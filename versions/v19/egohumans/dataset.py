#!/usr/bin/env python3
"""EgoHumans calibration, GT, visibility, and capture audit utilities.

The functions in this module are evaluator-only.  Movie3R inference must not
import this module because it opens calibration, identity, bbox, and SMPL GT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.spatial.transform import Rotation


PROTOCOL_NAME = "Movie3R-EgoHumans-CS100-v1"
PROTOCOL_SEED = 20260820
FPS = 20.0
CLIP_LENGTH = 100
PRE_COUNT = 50
POST_COUNT = 50
ANGLE_QUANTILES = {
    "small": 0.10,
    "medium": 0.40,
    "large": 0.70,
    "extreme": 1.00,
}


@dataclass(frozen=True)
class ExoCalibration:
    name: str
    camera_id: int
    model: str
    width: int
    height: int
    params: np.ndarray
    camera_to_world: np.ndarray
    world_to_camera: np.ndarray
    pose_record_count: int
    centre_spread_max_m_colmap: float


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(jsonable(value), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)


def rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first)[:3, :3].T @ np.asarray(second)[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _camera_table(path: Path) -> dict[int, dict[str, Any]]:
    output: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields or not fields[0].isdigit():
            continue
        output[int(fields[0])] = {
            "model": fields[1],
            "width": int(fields[2]),
            "height": int(fields[3]),
            "params": np.asarray(fields[4:], dtype=np.float64),
        }
    if not output:
        raise ValueError(f"No COLMAP cameras in {path}")
    return output


def _image_pose_records(path: Path) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) != 10 or not fields[0].isdigit() or "/" not in fields[-1]:
                continue
            name = fields[-1].split("/", 1)[0]
            if not name.startswith("cam"):
                continue
            quaternion = np.asarray(fields[1:5], dtype=np.float64)
            world_to_camera_rotation = Rotation.from_quat(
                [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]
            ).as_matrix()
            translation = np.asarray(fields[5:8], dtype=np.float64)
            output.setdefault(name, []).append(
                {
                    "rotation": world_to_camera_rotation,
                    "translation": translation,
                    "camera_id": int(fields[8]),
                }
            )
    if not output:
        raise ValueError(f"No exo poses in {path}")
    return output


def load_exo_calibrations(capture_root: Path) -> dict[str, ExoCalibration]:
    """Return static exo c2w poses in the metric Aria-01 GT world."""

    workplace = capture_root / "colmap/workplace"
    camera_rows = _camera_table(workplace / "cameras.txt")
    pose_rows = _image_pose_records(workplace / "images.txt")
    with (workplace / "colmap_from_aria_transforms.pkl").open("rb") as handle:
        transforms = pickle.load(handle)
    if "aria01" not in transforms:
        raise KeyError("colmap_from_aria_transforms.pkl lacks aria01")
    aria_to_colmap = np.asarray(transforms["aria01"], dtype=np.float64)
    if aria_to_colmap.shape != (4, 4):
        raise ValueError(f"Unexpected Aria-to-COLMAP shape {aria_to_colmap.shape}")
    linear = aria_to_colmap[:3, :3]
    scale = float(np.cbrt(np.linalg.det(linear)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid Aria-to-COLMAP scale {scale}")
    aria_to_colmap_rotation = linear / scale
    if not np.allclose(
        aria_to_colmap_rotation.T @ aria_to_colmap_rotation,
        np.eye(3),
        atol=2e-3,
    ):
        raise ValueError("Aria-to-COLMAP linear block is not a similarity")
    aria_to_colmap_translation = aria_to_colmap[:3, 3]

    output: dict[str, ExoCalibration] = {}
    for name, rows in sorted(pose_rows.items()):
        camera_ids = {int(row["camera_id"]) for row in rows}
        if len(camera_ids) != 1:
            raise ValueError(f"{name} has inconsistent camera IDs: {camera_ids}")
        camera_id = next(iter(camera_ids))
        intrinsics = camera_rows[camera_id]
        rotations = np.stack([row["rotation"] for row in rows])
        world_to_camera_colmap = Rotation.from_matrix(rotations).mean().as_matrix()
        centres_colmap = np.stack(
            [-row["rotation"].T @ row["translation"] for row in rows]
        )
        centre_colmap = centres_colmap.mean(axis=0)
        camera_to_colmap = world_to_camera_colmap.T
        centre_aria = np.linalg.solve(
            linear, centre_colmap - aria_to_colmap_translation
        )
        camera_to_aria = aria_to_colmap_rotation.T @ camera_to_colmap
        camera_to_world = np.eye(4, dtype=np.float64)
        camera_to_world[:3, :3] = camera_to_aria
        camera_to_world[:3, 3] = centre_aria
        world_to_camera = np.linalg.inv(camera_to_world)
        output[name] = ExoCalibration(
            name=name,
            camera_id=camera_id,
            model=str(intrinsics["model"]),
            width=int(intrinsics["width"]),
            height=int(intrinsics["height"]),
            params=np.asarray(intrinsics["params"], dtype=np.float64),
            camera_to_world=camera_to_world,
            world_to_camera=world_to_camera,
            pose_record_count=len(rows),
            centre_spread_max_m_colmap=float(
                np.linalg.norm(centres_colmap - centre_colmap, axis=1).max()
            ),
        )
    return output


def load_smpl_people(capture_root: Path, frame: int) -> dict[str, dict[str, Any]]:
    path = capture_root / f"processed_data/smpl/{int(frame):05d}.npy"
    value = np.load(path, allow_pickle=True)
    if not isinstance(value, np.ndarray) or value.shape != ():
        raise ValueError(f"Unexpected SMPL container in {path}: {value.shape}")
    people = value.item()
    if not isinstance(people, dict) or not people:
        raise ValueError(f"No SMPL people in {path}")
    output = {}
    for identity, person in people.items():
        vertices = np.asarray(person["vertices"], dtype=np.float64)
        if vertices.shape != (6890, 3) or not np.isfinite(vertices).all():
            raise ValueError(f"Invalid SMPL vertices for {identity} at {frame}: {vertices.shape}")
        output[str(identity)] = {**person, "vertices": vertices}
    return output


def image_frames(capture_root: Path, camera: str) -> set[int]:
    return {
        int(path.stem)
        for path in (capture_root / "exo" / camera / "images").glob("*.jpg")
        if path.stem.isdigit()
    }


def smpl_frames(capture_root: Path) -> set[int]:
    return {
        int(path.stem)
        for path in (capture_root / "processed_data/smpl").glob("*.npy")
        if path.stem.isdigit()
    }


def longest_contiguous(values: Iterable[int]) -> list[int]:
    ordered = sorted(set(int(value) for value in values))
    if not ordered:
        return []
    best: list[int] = []
    start = previous = ordered[0]
    for value in ordered[1:] + [ordered[-1] + 2]:
        if value != previous + 1:
            run = list(range(start, previous + 1))
            if len(run) > len(best):
                best = run
            start = value
        previous = value
    return best


def _fisheye_pixels(camera_points: np.ndarray, calibration: ExoCalibration) -> np.ndarray:
    points = np.asarray(camera_points, dtype=np.float64)
    normal = points[:, :2] / np.maximum(points[:, 2:3], 1e-12)
    if calibration.model == "OPENCV_FISHEYE":
        radius = np.linalg.norm(normal, axis=1)
        theta = np.arctan(radius)
        theta2 = theta * theta
        k1, k2, k3, k4 = calibration.params[4:8]
        distorted = theta * (
            1.0 + k1 * theta2 + k2 * theta2**2 + k3 * theta2**3 + k4 * theta2**4
        )
        factor = np.divide(
            distorted,
            radius,
            out=np.ones_like(distorted),
            where=radius > 1e-12,
        )
        normal = normal * factor[:, None]
    elif calibration.model not in {"PINHOLE", "OPENCV"}:
        raise ValueError(f"Unsupported camera model {calibration.model}")
    pixels = normal.copy()
    pixels[:, 0] = calibration.params[0] * pixels[:, 0] + calibration.params[2]
    pixels[:, 1] = calibration.params[1] * pixels[:, 1] + calibration.params[3]
    return pixels


def projected_visible_fraction(vertices_world: np.ndarray, calibration: ExoCalibration) -> float:
    camera = (
        np.asarray(vertices_world) - calibration.camera_to_world[:3, 3]
    ) @ calibration.camera_to_world[:3, :3]
    pixels = _fisheye_pixels(camera, calibration)
    visible = (
        (camera[:, 2] > 1e-6)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < calibration.width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < calibration.height)
    )
    return float(visible.mean())


def bbox_annotations(capture_root: Path, camera: str, frame: int) -> dict[str, dict[str, Any]]:
    path = capture_root / f"processed_data/bboxes/{camera}/rgb/{int(frame):05d}.npy"
    if not path.is_file():
        return {}
    output = {}
    for row in np.load(path, allow_pickle=True):
        if not isinstance(row, dict) or "human_name" not in row or "bbox" not in row:
            continue
        output[str(row["human_name"])] = row
    return output


def annotated_visible(
    capture_root: Path,
    camera: str,
    frame: int,
    identity: str,
    calibration: ExoCalibration,
) -> bool:
    row = bbox_annotations(capture_root, camera, frame).get(identity)
    if row is None:
        return False
    bbox = np.asarray(row["bbox"], dtype=np.float64).reshape(-1)
    if len(bbox) < 4 or not np.isfinite(bbox[:4]).all():
        return False
    x1, y1, x2, y2 = bbox[:4]
    width = max(0.0, min(x2, calibration.width) - max(x1, 0.0))
    height = max(0.0, min(y2, calibration.height) - max(y1, 0.0))
    return width * height > 0.0


def visibility_fraction(
    capture_root: Path,
    camera: str,
    frame: int,
    identity: str,
    vertices_world: np.ndarray,
    calibration: ExoCalibration,
) -> float:
    if not annotated_visible(capture_root, camera, frame, identity, calibration):
        return 0.0
    return projected_visible_fraction(vertices_world, calibration)


def _directed_pair(capture_name: str, first: str, second: str) -> tuple[str, str]:
    digest = hashlib.sha256(f"{PROTOCOL_SEED}:{capture_name}:{first}:{second}".encode()).digest()
    return (first, second) if digest[0] % 2 == 0 else (second, first)


def select_camera_pairs(
    capture_name: str,
    calibrations: dict[str, ExoCalibration],
    visible_counts: dict[str, int],
    identity_count: int,
) -> list[dict[str, Any]]:
    required_visible = min(2, int(identity_count))
    pairs = []
    cameras = sorted(calibrations)
    for index, first in enumerate(cameras):
        for second in cameras[index + 1 :]:
            row = {
                "first": first,
                "second": second,
                "angle_deg": rotation_error_deg(
                    calibrations[first].camera_to_world,
                    calibrations[second].camera_to_world,
                ),
                "baseline_m": float(
                    np.linalg.norm(
                        calibrations[first].camera_to_world[:3, 3]
                        - calibrations[second].camera_to_world[:3, 3]
                    )
                ),
                "minimum_visible_people_at_boundary": min(
                    visible_counts.get(first, 0), visible_counts.get(second, 0)
                ),
            }
            pairs.append(row)
    eligible = [
        row for row in pairs
        if row["minimum_visible_people_at_boundary"] >= required_visible
    ]
    visibility_fallback = False
    if len(eligible) < len(ANGLE_QUANTILES):
        eligible = pairs
        visibility_fallback = True
    eligible.sort(key=lambda row: (row["angle_deg"], row["first"], row["second"]))
    if len(eligible) < len(ANGLE_QUANTILES):
        raise ValueError(f"Only {len(eligible)} exo camera pairs are available")
    selected: list[dict[str, Any]] = []
    unused = set(range(len(eligible)))
    for stratum, quantile in ANGLE_QUANTILES.items():
        target = int(round(float(quantile) * (len(eligible) - 1)))
        chosen = min(
            unused,
            key=lambda index: (
                abs(index - target),
                eligible[index]["angle_deg"],
                eligible[index]["first"],
                eligible[index]["second"],
            ),
        )
        unused.remove(chosen)
        row = dict(eligible[chosen])
        pre, post = _directed_pair(capture_name, row.pop("first"), row.pop("second"))
        selected.append(
            {
                "angle_stratum": stratum,
                "pre_camera": pre,
                "post_camera": post,
                "selection_quantile": float(quantile),
                "visibility_fallback": visibility_fallback,
                **row,
            }
        )
    return selected


def audit_capture(
    capture_root: Path,
    archive_entry: str,
    capture_relative: str | None = None,
) -> dict[str, Any]:
    capture_root = capture_root.resolve()
    if not (capture_root / "exo").is_dir() or not (capture_root / "processed_data/smpl").is_dir():
        raise FileNotFoundError(f"Not an EgoHumans capture root: {capture_root}")
    calibrations = load_exo_calibrations(capture_root)
    cameras = sorted(
        name for name in calibrations
        if (capture_root / "exo" / name / "images").is_dir()
    )
    if len(cameras) < 3:
        raise ValueError(f"Only {len(cameras)} exo cameras have RGB")
    common = smpl_frames(capture_root)
    per_camera_counts = {}
    for camera in cameras:
        frames = image_frames(capture_root, camera)
        per_camera_counts[camera] = len(frames)
        common &= frames
    contiguous = longest_contiguous(common)
    if len(contiguous) < CLIP_LENGTH:
        raise ValueError(
            f"Longest synchronized RGB/SMPL run has {len(contiguous)} frames; need {CLIP_LENGTH}"
        )
    offset = (len(contiguous) - CLIP_LENGTH) // 2
    clip_frames = contiguous[offset : offset + CLIP_LENGTH]
    boundary_frame = clip_frames[PRE_COUNT]
    checked_frames = sorted({clip_frames[0], boundary_frame, clip_frames[-1]})
    people_by_frame = {frame: load_smpl_people(capture_root, frame) for frame in checked_frames}
    identities = sorted(people_by_frame[checked_frames[0]])
    for frame, people in people_by_frame.items():
        if sorted(people) != identities:
            raise ValueError(
                f"GT identity set changes at frame {frame}: {sorted(people)} vs {identities}"
            )
    boundary_people = people_by_frame[boundary_frame]
    visible_counts = {
        camera: sum(
            annotated_visible(
                capture_root, camera, boundary_frame, identity, calibrations[camera]
            )
            for identity in identities
        )
        for camera in cameras
    }
    pairs = select_camera_pairs(
        capture_root.name, calibrations, visible_counts, len(identities)
    )
    projection = {
        camera: {
            identity: visibility_fraction(
                capture_root,
                camera,
                boundary_frame,
                identity,
                boundary_people[identity]["vertices"],
                calibrations[camera],
            )
            for identity in identities
        }
        for camera in cameras
    }
    return {
        "schema_version": "Movie3R-v19-EgoHumans-capture-audit-v1",
        "protocol": PROTOCOL_NAME,
        "protocol_seed": PROTOCOL_SEED,
        "archive_entry": archive_entry,
        "capture_name": capture_root.name,
        "capture_root": str(capture_root),
        "capture_relative": capture_relative or capture_root.name,
        "fps": FPS,
        "all_common_frame_count": len(common),
        "common_frame_min": min(common),
        "common_frame_max": max(common),
        "longest_contiguous_count": len(contiguous),
        "longest_contiguous_min": contiguous[0],
        "longest_contiguous_max": contiguous[-1],
        "clip_frames": clip_frames,
        "boundary_frame": boundary_frame,
        "boundary_index": PRE_COUNT,
        "camera_count": len(cameras),
        "cameras": cameras,
        "per_camera_rgb_counts": per_camera_counts,
        "identity_count": len(identities),
        "identities": identities,
        "visible_people_at_boundary": visible_counts,
        "projected_visible_fraction_at_boundary": projection,
        "calibrations": {
            name: {
                "camera_id": value.camera_id,
                "model": value.model,
                "width": value.width,
                "height": value.height,
                "params": value.params,
                "camera_to_world": value.camera_to_world,
                "pose_record_count": value.pose_record_count,
                "centre_spread_max_m_colmap": value.centre_spread_max_m_colmap,
            }
            for name, value in calibrations.items()
        },
        "selected_protocol_pairs": pairs,
        "selection_depends_on_model_result": False,
        "runtime_may_read_audit": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--archive-entry", required=True)
    parser.add_argument("--capture-relative")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    value = audit_capture(
        args.capture_root, args.archive_entry, args.capture_relative
    )
    atomic_json(args.output, value)
    print(json.dumps(jsonable(value), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

