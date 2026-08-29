#!/usr/bin/env python3
"""Render real scene--camera--human panels for the paper qualitative figure.

Only method-native payloads are used.  Strict Human3R and BRIDGE3R contribute
their own replayed depth/confidence, camera poses, and formal-test human
meshes.  No ground truth or geometry from another method is used.  External
methods are intentionally left mesh-only in the surrounding SVG because they
do not expose an equivalent dense scene/camera representation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
import pyrender
import trimesh


MOVIE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = MOVIE_ROOT.parent
for item in (MOVIE_ROOT, MOVIE_ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from publication.bridge3r_iclr2027.launch_demo_payload_viewer import load_payload  # noqa: E402


PAYLOAD_ROOT = MOVIE_ROOT / "output/bridge3r_two_dataset_demo_v2/egohumans/payloads"
DEFAULT_OUTPUT = (
    WORKSPACE
    / "ICLR-paper/bridge3r_iclr2027/versions/"
    "v026_20260830_mvhuman_and_real_qualitative/manuscript/figures/"
    "qualitative_real_large_view_panels"
)
METHOD_COLOURS = (
    (65, 105, 225),
    (238, 99, 82),
    (46, 160, 108),
    (155, 89, 182),
    (238, 174, 49),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload-root", type=Path, default=PAYLOAD_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=874)
    parser.add_argument("--point-stride", type=int, default=12)
    return parser.parse_args()


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    backward = eye - target
    backward /= max(float(np.linalg.norm(backward)), 1e-9)
    right = np.cross(up, backward)
    right /= max(float(np.linalg.norm(right)), 1e-9)
    camera_up = np.cross(backward, right)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = right
    pose[:3, 1] = camera_up
    pose[:3, 2] = backward
    pose[:3, 3] = eye
    return pose


def transformed_pose(pose: np.ndarray, flip: np.ndarray) -> np.ndarray:
    output = np.asarray(pose, dtype=np.float64).copy()
    output[:3, :3] = flip @ output[:3, :3]
    output[:3, 3] = flip @ output[:3, 3]
    return output


def cylinder_between(start: np.ndarray, end: np.ndarray, radius: float) -> trimesh.Trimesh:
    start, end = np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64)
    vector = end - start
    length = float(np.linalg.norm(vector))
    if length < 1e-9:
        raise ValueError("Degenerate camera-frustum edge")
    transform = trimesh.geometry.align_vectors(np.asarray([0.0, 0.0, 1.0]), vector / length)
    transform[:3, 3] = 0.5 * (start + end)
    return trimesh.creation.cylinder(radius=radius, height=length, sections=10, transform=transform)


def add_coloured_trimesh(
    scene: pyrender.Scene,
    mesh: trimesh.Trimesh,
    rgb: tuple[int, int, int],
    *,
    roughness: float = 0.8,
) -> None:
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=tuple(value / 255.0 for value in rgb) + (1.0,),
        metallicFactor=0.0,
        roughnessFactor=roughness,
    )
    scene.add(pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True))


def frustum_vertices(pose: np.ndarray, intrinsics: np.ndarray, width: int, height: int, depth: float) -> np.ndarray:
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    pixels = np.asarray([[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]])
    corners_camera = np.column_stack(
        ((pixels[:, 0] - cx) * depth / fx, (pixels[:, 1] - cy) * depth / fy, np.full(4, depth))
    )
    corners_world = corners_camera @ pose[:3, :3].T + pose[:3, 3]
    return np.vstack((pose[:3, 3], corners_world))


def add_frustum(
    scene: pyrender.Scene,
    pose: np.ndarray,
    intrinsics: np.ndarray,
    image_shape: tuple[int, int],
    rgb: tuple[int, int, int],
    scale: float,
    radius: float,
) -> np.ndarray:
    height, width = image_shape
    vertices = frustum_vertices(pose, intrinsics, width, height, scale)
    edges = ((0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1))
    for first, second in edges:
        add_coloured_trimesh(scene, cylinder_between(vertices[first], vertices[second], radius), rgb)
    centre = trimesh.creation.icosphere(subdivisions=2, radius=radius * 2.6)
    centre.apply_translation(vertices[0])
    add_coloured_trimesh(scene, centre, rgb)
    return vertices


def add_lights(scene: pyrender.Scene, camera_pose: np.ndarray) -> None:
    for yaw in (-0.75, 0.0, 0.75):
        pose = camera_pose.copy()
        rotation = np.asarray(
            [[np.cos(yaw), 0.0, np.sin(yaw)], [0.0, 1.0, 0.0], [-np.sin(yaw), 0.0, np.cos(yaw)]]
        )
        pose[:3, :3] = pose[:3, :3] @ rotation
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.2), pose=pose)


def method_scene(payload: dict[str, Any], method: str, width: int, height: int, stride: int) -> tuple[np.ndarray, dict[str, Any]]:
    # The first post-cut payload frame is local index 5; index 4 is the last
    # pre-cut observation.  Both point clouds are shown to make the inter-view
    # alignment visible.
    frame_indices = (4, 5)
    current = 5
    flip = np.diag([1.0, -1.0, 1.0]).astype(np.float64)
    points_all, colours_all = [], []
    for frame_index in frame_indices:
        points = np.asarray(payload["pc_list"][frame_index], dtype=np.float64)
        colours = np.asarray(payload["colour_list"][frame_index][0], dtype=np.float64)
        confidence = np.asarray(payload["confidence_list"][frame_index][0], dtype=np.float64)
        mask_entry = payload["mask_list"][frame_index]
        valid = np.isfinite(points).all(axis=-1) & np.isfinite(confidence) & (confidence > 1.0)
        if mask_entry is not None:
            valid &= np.asarray(mask_entry[0]) < 0.1
        points = points[valid][::stride] @ flip.T
        colours = colours[valid][::stride]
        points_all.append(points)
        colours_all.append(colours)
    points = np.concatenate(points_all, axis=0)
    colours = np.clip(np.concatenate(colours_all, axis=0) * 255.0, 0, 255).astype(np.uint8)

    vertices = np.asarray(payload["vertices"][current], dtype=np.float64) @ flip.T
    identities = np.asarray(payload["identities"][current], dtype=np.int64)
    faces = np.asarray(payload["faces"], dtype=np.int32)
    human_centre = np.nanmedian(vertices.reshape(-1, 3), axis=0)
    # Exclude distant floaters only for display framing; the same deterministic
    # radius is used for both internal methods and does not modify predictions.
    human_extent = max(float(np.ptp(vertices.reshape(-1, 3), axis=0).max()), 1.5)
    display_radius = max(4.0 * human_extent, 7.0)
    keep = np.linalg.norm(points - human_centre, axis=1) <= display_radius
    points, colours = points[keep], colours[keep]

    cameras = []
    intrinsics = []
    image_shape = None
    root = Path(payload["_root"])
    for frame_index in frame_indices:
        with np.load(root / "camera" / f"{frame_index:06d}.npz", allow_pickle=False) as camera:
            cameras.append(transformed_pose(np.asarray(camera["pose"]), flip))
            intrinsics.append(np.asarray(camera["intrinsics"], dtype=np.float64))
        colour = cv2.imread(str(root / "color" / f"{frame_index:06d}.png"), cv2.IMREAD_COLOR)
        if colour is None:
            raise OSError(root / "color" / f"{frame_index:06d}.png")
        image_shape = colour.shape[:2]

    scene = pyrender.Scene(
        bg_color=np.asarray([0.965, 0.97, 0.98, 1.0]),
        ambient_light=np.ones(3) * 0.55,
    )
    scene.add(pyrender.Mesh.from_points(points, colors=colours))
    for person_index, person_vertices in enumerate(vertices):
        identity = int(identities[person_index]) if person_index < len(identities) else person_index
        mesh = trimesh.Trimesh(vertices=person_vertices, faces=faces, process=False)
        add_coloured_trimesh(scene, mesh, METHOD_COLOURS[identity % len(METHOD_COLOURS)], roughness=0.58)

    framing_points = [points[:: max(1, len(points) // 4000)], vertices.reshape(-1, 3)[::80]]
    camera_scale = max(0.22 * human_extent, 0.35)
    frustum_radius = max(0.0045 * display_radius, 0.012)
    frustum_colours = ((38, 99, 185), (220, 68, 45))
    for pose, intrinsic, rgb in zip(cameras, intrinsics, frustum_colours):
        frustum = add_frustum(
            scene, pose, intrinsic, image_shape, rgb, camera_scale, frustum_radius
        )
        framing_points.append(frustum)

    sampled = np.concatenate(framing_points, axis=0)
    lower, upper = np.nanpercentile(sampled, [1.0, 99.0], axis=0)
    centre = 0.5 * (lower + upper)
    span = max(float(np.max(upper - lower)), 2.0)
    eye = centre + span * np.asarray([1.25, 0.85, 1.45])
    camera_pose = look_at(eye, centre)
    world_to_camera = np.linalg.inv(camera_pose)
    projected = np.column_stack((sampled, np.ones(len(sampled)))) @ world_to_camera.T
    x_low, x_high = np.nanpercentile(projected[:, 0], [0.25, 99.75])
    y_low, y_high = np.nanpercentile(projected[:, 1], [0.25, 99.75])
    xmag = max(float(x_high - x_low) * 0.58, 1.0)
    ymag = max(float(y_high - y_low) * 0.62, 1.0)
    aspect = width / height
    if xmag / ymag < aspect:
        xmag = ymag * aspect
    else:
        ymag = xmag / aspect
    scene.add(pyrender.OrthographicCamera(xmag=xmag, ymag=ymag, znear=0.01, zfar=1000.0), pose=camera_pose)
    add_lights(scene, camera_pose)
    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height, point_size=2.0)
    rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SKIP_CULL_FACES)
    renderer.delete()
    image = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
    return image, {
        "method": method,
        "payload": str(root.resolve()),
        "frames": list(frame_indices),
        "point_count_after_deterministic_sampling": int(len(points)),
        "camera_colours": {"pre_cut": "#2663B9", "first_post_cut": "#DC442D"},
        "gt_used": False,
        "cross_method_geometry_used": False,
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    provenance = {
        "schema_version": "Bridge3R-real-qualitative-scene-camera-panels-v1",
        "case_id": "ego_test_legoassemble_003_legoassemble_extreme_cam03_cam04_b00301",
        "selection": "metric-selected extreme-view example; not claimed representative",
        "methods": {},
    }
    for method in ("strict", "bridge3r"):
        root = (args.payload_root / method).resolve()
        payload = load_payload(root)
        payload["_root"] = str(root)
        image, record = method_scene(
            payload, method, int(args.width), int(args.height), int(args.point_stride)
        )
        destination = output / f"{method}_scene_camera_post_local000005_original0050.png"
        if not cv2.imwrite(str(destination), image):
            raise OSError(destination)
        record["output"] = str(destination)
        provenance["methods"][method] = record
    provenance_path = output / "scene_camera_panel_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "provenance": str(provenance_path)}, indent=2))


if __name__ == "__main__":
    main()
