#!/usr/bin/env python3
"""Read the public Harmony4D release without leaking GT into runtime.

The release stores static exocentric calibration as COLMAP text and SMPL GT
as per-frame dictionaries.  COLMAP poses are world-to-camera.  Four sparse
calibration frames are available per exocentric camera; this adapter averages
camera centres and rotations and records their scatter instead of silently
selecting one estimate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class CameraCalibration:
    name: str
    camera_id: int
    width: int
    height: int
    model: str
    intrinsic: np.ndarray
    distortion: np.ndarray
    world_to_camera: np.ndarray
    camera_to_world: np.ndarray
    calibration_views: int
    center_scatter_max_m: float
    rotation_scatter_max_deg: float
    canonical_world: str

    def jsonable(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("intrinsic", "distortion", "world_to_camera", "camera_to_world"):
            value[key] = np.asarray(value[key]).tolist()
        return value


def _noncomment_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def _camera_table(path: Path) -> dict[int, dict[str, Any]]:
    output: dict[int, dict[str, Any]] = {}
    for line in _noncomment_lines(path):
        fields = line.split()
        camera_id, model = int(fields[0]), fields[1]
        width, height = int(fields[2]), int(fields[3])
        params = np.asarray([float(value) for value in fields[4:]], dtype=np.float64)
        if model != "OPENCV_FISHEYE" or len(params) != 8:
            raise ValueError(f"Unsupported Harmony4D COLMAP camera: {line}")
        fx, fy, cx, cy = params[:4]
        output[camera_id] = {
            "camera_id": camera_id,
            "model": model,
            "width": width,
            "height": height,
            "intrinsic": np.asarray(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
            "distortion": params[4:].copy(),
        }
    return output


def _image_poses(path: Path) -> list[dict[str, Any]]:
    lines = _noncomment_lines(path)
    if len(lines) % 2:
        raise ValueError(f"COLMAP images.txt has an odd number of data lines: {path}")
    output = []
    for index in range(0, len(lines), 2):
        fields = lines[index].split()
        if len(fields) < 10:
            raise ValueError(f"Malformed COLMAP image row: {lines[index]}")
        qw, qx, qy, qz = [float(value) for value in fields[1:5]]
        translation = np.asarray([float(value) for value in fields[5:8]], dtype=np.float64)
        rotation = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        world_to_camera = np.eye(4, dtype=np.float64)
        world_to_camera[:3, :3] = rotation
        world_to_camera[:3, 3] = translation
        output.append(
            {
                "image_id": int(fields[0]),
                "camera_id": int(fields[8]),
                "name": fields[9],
                "world_to_camera": world_to_camera,
                "camera_to_world": np.linalg.inv(world_to_camera),
            }
        )
    return output


def _rotation_distance_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first)[:3, :3].T @ np.asarray(second)[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _load_transform_dict(path: Path) -> dict[str, np.ndarray]:
    import pickle

    with path.open("rb") as handle:
        value = pickle.load(handle, encoding="latin1")
    return {str(key): np.asarray(item, dtype=np.float64) for key, item in value.items()}


def _similarity_rotation(transform: np.ndarray) -> tuple[float, np.ndarray]:
    linear = np.asarray(transform, dtype=np.float64)[:3, :3]
    scales = np.linalg.norm(linear, axis=0)
    scale = float(scales.mean())
    if scale <= 0 or float(np.max(np.abs(scales - scale))) > 1e-5:
        raise ValueError(f"Non-uniform Harmony4D similarity: scales={scales}")
    rotation = linear / scale
    if np.linalg.det(rotation) < 0:
        raise ValueError("Reflection in Harmony4D coordinate similarity")
    return scale, rotation


def transform_camera_similarity(transform: np.ndarray, camera_to_world: np.ndarray) -> np.ndarray:
    scale, rotation = _similarity_rotation(transform)
    output = np.eye(4, dtype=np.float64)
    output[:3, :3] = rotation @ np.asarray(camera_to_world)[:3, :3]
    output[:3, 3] = (
        scale * rotation @ np.asarray(camera_to_world)[:3, 3]
        + np.asarray(transform, dtype=np.float64)[:3, 3]
    )
    return output


def transform_points_similarity(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    value = np.asarray(points, dtype=np.float64)
    return value @ np.asarray(transform, dtype=np.float64)[:3, :3].T + np.asarray(transform)[:3, 3]


def load_exo_calibrations(
    sequence_root: Path, canonical_identity: str = "aria01"
) -> dict[str, CameraCalibration]:
    workplace = sequence_root / "colmap/workplace"
    cameras = _camera_table(workplace / "cameras.txt")
    rows = _image_poses(workplace / "images.txt")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        name = str(row["name"]).split("/")[0]
        if name.startswith("cam"):
            grouped.setdefault(name, []).append(row)
    aria_from_colmap = _load_transform_dict(workplace / "aria_from_colmap_transforms.pkl")
    if canonical_identity not in aria_from_colmap:
        raise KeyError(f"Canonical identity {canonical_identity} absent from aria-from-COLMAP transforms")
    canonical_from_colmap = aria_from_colmap[canonical_identity]
    output: dict[str, CameraCalibration] = {}
    for name, values in sorted(grouped.items()):
        camera_ids = {int(value["camera_id"]) for value in values}
        if len(camera_ids) != 1:
            raise ValueError(f"Camera {name} maps to multiple COLMAP intrinsics: {camera_ids}")
        camera_id = next(iter(camera_ids))
        table = cameras[camera_id]
        centres = np.stack([value["camera_to_world"][:3, 3] for value in values])
        rotations = Rotation.from_matrix(
            np.stack([value["camera_to_world"][:3, :3] for value in values])
        )
        mean_rotation = rotations.mean().as_matrix()
        mean_centre = centres.mean(axis=0)
        camera_to_colmap = np.eye(4, dtype=np.float64)
        camera_to_colmap[:3, :3] = mean_rotation
        camera_to_colmap[:3, 3] = mean_centre
        camera_to_world = transform_camera_similarity(canonical_from_colmap, camera_to_colmap)
        canonical_centres = np.stack(
            [transform_camera_similarity(canonical_from_colmap, value["camera_to_world"])[:3, 3] for value in values]
        )
        center_scatter = np.linalg.norm(canonical_centres - camera_to_world[:3, 3], axis=1)
        rotation_scatter = [
            _rotation_distance_deg(
                camera_to_world,
                transform_camera_similarity(canonical_from_colmap, value["camera_to_world"]),
            )
            for value in values
        ]
        output[name] = CameraCalibration(
            name=name,
            camera_id=camera_id,
            width=int(table["width"]),
            height=int(table["height"]),
            model=str(table["model"]),
            intrinsic=np.asarray(table["intrinsic"], dtype=np.float64),
            distortion=np.asarray(table["distortion"], dtype=np.float64),
            world_to_camera=np.linalg.inv(camera_to_world),
            camera_to_world=camera_to_world,
            calibration_views=len(values),
            center_scatter_max_m=float(center_scatter.max(initial=0.0)),
            rotation_scatter_max_deg=float(max(rotation_scatter, default=0.0)),
            canonical_world=canonical_identity,
        )
    if not output:
        raise ValueError(f"No exocentric cameras found under {sequence_root}")
    return output


def locate_sequence_root(extracted_root: Path) -> Path:
    candidates = sorted(
        path.parent.parent.parent
        for path in extracted_root.rglob("colmap/workplace/cameras.txt")
    )
    if len(candidates) != 1:
        raise ValueError(f"Expected one Harmony4D capture below {extracted_root}, found {candidates}")
    return candidates[0]


def frame_numbers(sequence_root: Path, camera_names: list[str] | None = None) -> list[int]:
    names = camera_names or sorted(path.name for path in (sequence_root / "exo").glob("cam*"))
    sets = []
    for name in names:
        images = sequence_root / "exo" / name / "images"
        values = {
            int(path.stem)
            for path in images.glob("*.jpg")
            if path.stem.isdigit()
        }
        if not values:
            raise ValueError(f"No numbered RGB frames for {name}: {images}")
        sets.append(values)
    smpl = {
        int(path.stem)
        for path in (sequence_root / "processed_data/smpl").glob("*.npy")
        if path.stem.isdigit()
    }
    common = set.intersection(*sets, smpl)
    return sorted(common)


def image_path(sequence_root: Path, camera_name: str, frame: int) -> Path:
    path = sequence_root / "exo" / camera_name / "images" / f"{int(frame):05d}.jpg"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_gt_people(
    sequence_root: Path, frame: int, canonical_identity: str = "aria01"
) -> dict[str, dict[str, np.ndarray]]:
    path = sequence_root / "processed_data/smpl" / f"{int(frame):05d}.npy"
    value = np.load(path, allow_pickle=True)
    if value.shape != () or not isinstance(value.item(), dict):
        raise ValueError(f"Unexpected Harmony4D SMPL payload: {path}")
    workplace = sequence_root / "colmap/workplace"
    aria_from_colmap = _load_transform_dict(workplace / "aria_from_colmap_transforms.pkl")
    if canonical_identity not in aria_from_colmap:
        raise KeyError(canonical_identity)
    output = {}
    for identity, person in value.item().items():
        identity = str(identity)
        # Harmony4D's processed SMPL dictionaries are already fused into the
        # primary (aria01) metric world, despite keys naming the two wearers.
        # The per-Aria COLMAP transforms localise sensor trajectories; applying
        # them again per person corrupts the second body.  We therefore move
        # COLMAP exocentric cameras into aria01 above and keep all processed
        # bodies in their published shared metric frame here.
        if canonical_identity != "aria01":
            raise ValueError("The public processed SMPL world is anchored to aria01")
        canonical_from_identity = np.eye(4, dtype=np.float64)
        vertices = np.asarray(person["vertices"], dtype=np.float64)
        joints = np.asarray(person["joints"], dtype=np.float64)
        if vertices.shape != (6890, 3) or joints.ndim != 2 or joints.shape[1] != 3:
            raise ValueError(f"Unexpected GT topology for {identity} at {path}: {vertices.shape}, {joints.shape}")
        output[identity] = {
            "vertices": vertices,
            "joints_stored": joints,
            "transl": np.asarray(person["transl"], dtype=np.float64),
            "global_orient": np.asarray(person["global_orient"], dtype=np.float64),
            "canonical_from_identity": canonical_from_identity,
        }
    return output


def project_fisheye(points_world: np.ndarray, calibration: CameraCalibration) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    world_to_camera = calibration.world_to_camera
    camera = points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    valid_depth = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 1e-6)
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    if valid_depth.any():
        rotation_vector, _ = cv2.Rodrigues(world_to_camera[:3, :3])
        projected, _ = cv2.fisheye.projectPoints(
            points[valid_depth, None, :],
            rotation_vector,
            world_to_camera[:3, 3],
            calibration.intrinsic,
            calibration.distortion,
        )
        pixels[valid_depth] = projected[:, 0]
    return pixels, valid_depth


def projected_visibility(points_world: np.ndarray, calibration: CameraCalibration) -> dict[str, Any]:
    pixels, valid_depth = project_fisheye(points_world, calibration)
    inside = valid_depth.copy()
    inside &= np.isfinite(pixels).all(axis=1)
    inside &= (pixels[:, 0] >= 0.0) & (pixels[:, 0] < calibration.width)
    inside &= (pixels[:, 1] >= 0.0) & (pixels[:, 1] < calibration.height)
    valid_pixels = pixels[inside]
    bbox = (
        np.r_[valid_pixels.min(axis=0), valid_pixels.max(axis=0)]
        if len(valid_pixels)
        else np.full(4, np.nan, dtype=np.float64)
    )
    area = 0.0 if not len(valid_pixels) else float(np.prod(np.maximum(bbox[2:] - bbox[:2], 0.0)))
    return {
        "visible_vertex_fraction": float(inside.mean()) if len(inside) else 0.0,
        "positive_depth_fraction": float(valid_depth.mean()) if len(valid_depth) else 0.0,
        "bbox_xyxy": bbox.tolist(),
        "bbox_area_fraction": area / float(calibration.width * calibration.height),
        "visible_vertices": int(inside.sum()),
    }


def infer_fps(sequence_root: Path) -> float:
    videos = sorted((sequence_root / "exo").glob("cam*/images/rgb.mp4"))
    for video in videos:
        capture = cv2.VideoCapture(str(video))
        value = float(capture.get(cv2.CAP_PROP_FPS))
        capture.release()
        if np.isfinite(value) and value > 0:
            return value
    return 30.0
