#!/usr/bin/env python3
"""Generate V14.3 RGB, fixed-world, timeline, and correction videos.

DA3 and Human Projection are evaluated only at the first post-cut frame.  The
resulting local-world human translation and one Boundary transform are fixed
for the complete post-cut shot.  SMPL-X construction is intentionally batched
on CUDA; no frame-wise Boundary solve is performed by this script.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import matplotlib
import numpy as np
import pyrender
import torch
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402


ROOT = Path(__file__).resolve().parents[3]
for path in (str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from dust3r.utils.smpl_layer import SMPL_Layer  # noqa: E402
from versions.v12.experiments.v14_2_canonical_human_memory_probe import (  # noqa: E402
    blend_rotations,
    physical_scale,
)


DEFAULT_QUANTITATIVE = (
    ROOT
    / "output/v14_3_projection_consistent_reanchoring/quantitative"
    / "v14_3_projection_consistent_reanchoring.json"
)
DEFAULT_V14_2 = (
    ROOT
    / "output/v14_2_canonical_human_memory/single_cut"
    / "v14_2_canonical_human_memory_probe.json"
)
DEFAULT_CACHE = ROOT / "output/v52_long_sequence_visualization/cache"
DEFAULT_OUTPUT = ROOT / "output/v14_3_projection_consistent_reanchoring/visualization"
METHOD_LABELS = (
    "Hard Reset + Camera-only",
    "Continuity + Camera-only",
    "Coupled Human-Camera",
    "Coupled + Continuity",
)
METHOD_COLORS = (
    (214, 92, 73),
    (45, 141, 161),
    (232, 153, 57),
    (47, 139, 87),
)
ROLE_LABELS = {
    "continuity_gain": "Continuity improves clearly",
    "continuity_neutral": "Continuity changes little",
    "memory_regression": "Memory slightly hurts absolute body accuracy",
    "camera_human_conflict": "Camera improves while raw human worsens",
    "coupled_success": "Coupled correction succeeds",
    "coupled_failure": "Coupled correction remains difficult",
    "da3_wins": "DA3 is better than Human Projection",
    "human_projection_wins": "Human Projection is better than DA3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quantitative", type=Path, default=DEFAULT_QUANTITATIVE)
    parser.add_argument("--v14_2_report", type=Path, default=DEFAULT_V14_2)
    parser.add_argument("--cache_dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--alignment_method", default="da3_coupled_alpha_0p75")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--point_stride", type=int, default=24)
    parser.add_argument("--confidence_threshold", type=float, default=1.5)
    parser.add_argument("--mask_dilate", type=int, default=9)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--cases", nargs="*", default=())
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


@dataclass
class SequenceData:
    name: str
    source: str
    count: int
    images: list[np.ndarray]
    intrinsics: np.ndarray
    local_poses: np.ndarray
    points_local: list[np.ndarray]
    point_colors: list[np.ndarray]
    raw_joints_camera: np.ndarray
    raw_vertices_camera: np.ndarray
    continuity_joints_centered: np.ndarray
    continuity_vertices_centered: np.ndarray
    raw_betas: np.ndarray
    output_betas: np.ndarray
    raw_scales: np.ndarray
    output_scales: np.ndarray
    raw_local_residual_deg: np.ndarray
    output_local_residual_deg: np.ndarray
    rotvecs: np.ndarray
    boundary: np.ndarray
    correction_local: np.ndarray
    calibrated_root_first: np.ndarray
    raw_root_reference: np.ndarray
    v18_depth: float
    da3_depth: float
    quantitative_case: dict
    v14_2_case: dict


def transform_points(pose: np.ndarray, points: np.ndarray) -> np.ndarray:
    return (np.asarray(points) @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32)


def camera_points(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    yy, xx = np.indices(depth.shape, dtype=np.float32)
    return np.stack(
        [
            (xx - K[0, 2]) * depth / K[0, 0],
            (yy - K[1, 2]) * depth / K[1, 1],
            depth,
        ],
        axis=-1,
    )


def load_frame(path: Path, index: int) -> dict:
    with np.load(path / "camera" / f"{index:06d}.npz") as camera:
        pose = np.asarray(camera["pose"], dtype=np.float32)
        K = np.asarray(camera["intrinsics"], dtype=np.float32)
    with np.load(path / "smpl" / f"{index:06d}.npz", allow_pickle=True) as smpl_file:
        smpl = {name: np.asarray(smpl_file[name]) for name in smpl_file.files}
    expression = smpl["expression"]
    return {
        "pose": pose,
        "K": K,
        "image": np.asarray(Image.open(path / "color" / f"{index:06d}.png").convert("RGB")),
        "depth": np.load(path / "depth" / f"{index:06d}.npy").astype(np.float32),
        "confidence": np.load(path / "conf" / f"{index:06d}.npy").astype(np.float32),
        "mask": np.asarray(smpl["msk"][0] > 0.10, dtype=np.uint8),
        "rotvec": np.asarray(smpl["rotvec"][0], dtype=np.float32),
        "shape": np.asarray(smpl["shape"][0], dtype=np.float32),
        "transl": np.asarray(smpl["transl"][0], dtype=np.float32),
        "expression": (
            np.zeros(10, dtype=np.float32)
            if expression is None or len(expression) == 0
            else np.asarray(expression[0], dtype=np.float32)
        ),
    }


def body_batch(
    layer: SMPL_Layer,
    rotvecs: np.ndarray,
    shapes: np.ndarray,
    translations: np.ndarray,
    expressions: np.ndarray,
    intrinsics: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        output = layer(
            torch.from_numpy(rotvecs).to(device),
            torch.from_numpy(shapes).to(device),
            torch.from_numpy(translations).to(device),
            None,
            None,
            K=torch.from_numpy(intrinsics).to(device),
            expression=torch.from_numpy(expressions).to(device),
        )
    return (
        output["smpl_j3d"].detach().float().cpu().numpy().astype(np.float32),
        output["smpl_v3d"].detach().float().cpu().numpy().astype(np.float32),
    )


def geodesic_mean_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.swapaxes(first, -1, -2) @ second
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)).mean())


def continuity_parameters(
    rotvecs: np.ndarray,
    shapes: np.ndarray,
    raw_joints: np.ndarray,
    count: int,
    canonical_beta: np.ndarray,
    canonical_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    output_rotvecs = rotvecs.copy()
    output_shapes = shapes.copy()
    output_scales = np.asarray(
        [physical_scale(joints - joints[:1]) for joints in raw_joints], dtype=np.float32
    )
    raw_scales = output_scales.copy()
    raw_residual = np.zeros(len(rotvecs), dtype=np.float32)
    output_residual = np.zeros(len(rotvecs), dtype=np.float32)

    history = [Rotation.from_rotvec(value[1:].reshape(-1, 3)).as_matrix() for value in rotvecs[max(0, count - 5) : count]]
    memory = history[0].astype(np.float32)
    for value in history[1:]:
        memory = blend_rotations(memory, value, 0.20)

    for index in range(count, len(rotvecs)):
        current = Rotation.from_rotvec(rotvecs[index, 1:].reshape(-1, 3)).as_matrix()
        blended = blend_rotations(current, memory, 0.15)
        output_rotvecs[index, 1:] = Rotation.from_matrix(blended).as_rotvec().astype(np.float32)
        output_shapes[index] = shapes[index] + 0.25 * (canonical_beta - shapes[index])
        output_scales[index] = raw_scales[index] + 0.25 * (canonical_scale - raw_scales[index])
        raw_residual[index] = geodesic_mean_deg(memory, current)
        output_residual[index] = geodesic_mean_deg(memory, blended)
        memory = blend_rotations(memory, current, 0.20)
    return output_rotvecs, output_shapes, raw_scales, output_scales, raw_residual, output_residual


def normalize_centered_bodies(
    joints: np.ndarray,
    vertices: np.ndarray,
    target_scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    centered_joints = joints - joints[:, :1]
    centered_vertices = vertices - joints[:, :1]
    for index, target in enumerate(target_scales):
        scale = physical_scale(centered_joints[index])
        factor = 1.0 if not np.isfinite(scale) or scale < 1e-8 else float(target) / float(scale)
        centered_joints[index] *= factor
        centered_vertices[index] *= factor
    return centered_joints.astype(np.float32), centered_vertices.astype(np.float32)


def load_sequence(
    name: str,
    cache_dir: Path,
    quantitative_case: dict,
    v14_2_case: dict,
    layer: SMPL_Layer,
    device: torch.device,
    args: argparse.Namespace,
) -> SequenceData:
    manifest_path = cache_dir / name / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    count = int(manifest["frames_per_shot"])
    frames: list[dict] = []
    local_poses = []
    for side, gauge_name in (("pre", "pre_to_v10_gauge"), ("post", "post_to_v10_gauge")):
        path = Path(manifest[f"{side}_dir"])
        gauge = np.asarray(manifest[gauge_name], dtype=np.float32)
        for index in range(count):
            frame = load_frame(path, index)
            frames.append(frame)
            local_poses.append((gauge @ frame["pose"]).astype(np.float32))

    local_poses_array = np.stack(local_poses)
    intrinsics = np.stack([frame["K"] for frame in frames]).astype(np.float32)
    rotvecs = np.stack([frame["rotvec"] for frame in frames]).astype(np.float32)
    shapes = np.stack([frame["shape"] for frame in frames]).astype(np.float32)
    translations = np.stack([frame["transl"] for frame in frames]).astype(np.float32)
    expressions = np.stack([frame["expression"] for frame in frames]).astype(np.float32)
    raw_joints, raw_vertices = body_batch(
        layer, rotvecs, shapes, translations, expressions, intrinsics, device
    )
    memory = v14_2_case["memory"]
    output_rotvecs, output_shapes, raw_scales, output_scales, raw_residual, output_residual = (
        continuity_parameters(
            rotvecs,
            shapes,
            raw_joints,
            count,
            np.asarray(memory["canonical_beta"], dtype=np.float32),
            float(memory["canonical_physical_scale"]),
        )
    )
    zero_translation = np.zeros_like(translations)
    output_joints, output_vertices = body_batch(
        layer,
        output_rotvecs,
        output_shapes,
        zero_translation,
        expressions,
        intrinsics,
        device,
    )
    output_joints, output_vertices = normalize_centered_bodies(
        output_joints, output_vertices, output_scales
    )

    points_local: list[np.ndarray] = []
    point_colors: list[np.ndarray] = []
    kernel = np.ones((int(args.mask_dilate), int(args.mask_dilate)), dtype=np.uint8)
    for frame, pose in zip(frames, local_poses_array):
        depth = frame["depth"]
        confidence = frame["confidence"]
        mask = frame["mask"]
        if mask.shape != depth.shape:
            mask = cv2.resize(mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
        if int(args.mask_dilate) > 1:
            mask = cv2.dilate(mask, kernel, iterations=1)
        if confidence.shape != depth.shape:
            confidence = cv2.resize(
                confidence, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_LINEAR
            )
        points_camera = camera_points(depth, frame["K"])
        valid = (
            np.isfinite(points_camera).all(axis=-1)
            & np.isfinite(confidence)
            & (depth > 0.05)
            & (depth < 30.0)
            & (confidence > float(args.confidence_threshold))
            & (mask == 0)
        )
        ids = np.flatnonzero(valid.reshape(-1))[:: max(int(args.point_stride), 1)]
        points_local.append(transform_points(pose, points_camera.reshape(-1, 3)[ids]))
        point_colors.append(frame["image"].reshape(-1, 3)[ids])

    method = quantitative_case["methods"][str(args.alignment_method)]
    boundary = np.asarray(method["transform"], dtype=np.float32)
    calibrated_root = np.asarray(method["human"]["camera_root"], dtype=np.float32)
    raw_reference = np.asarray(quantitative_case["roots"]["raw_camera"], dtype=np.float32)
    correction_local = local_poses_array[count, :3, :3] @ (calibrated_root - raw_reference)
    generated_reference = raw_joints[count, 0]
    reference_delta = float(np.linalg.norm(generated_reference - raw_reference))
    if reference_delta > 0.08:
        raise RuntimeError(
            f"Long cache root differs from quantitative cache for {name}: {reference_delta:.4f} m"
        )
    return SequenceData(
        name=name,
        source=str(quantitative_case["source"]),
        count=count,
        images=[frame["image"] for frame in frames],
        intrinsics=intrinsics,
        local_poses=local_poses_array,
        points_local=points_local,
        point_colors=point_colors,
        raw_joints_camera=raw_joints,
        raw_vertices_camera=raw_vertices,
        continuity_joints_centered=output_joints,
        continuity_vertices_centered=output_vertices,
        raw_betas=shapes,
        output_betas=output_shapes,
        raw_scales=raw_scales,
        output_scales=output_scales,
        raw_local_residual_deg=raw_residual,
        output_local_residual_deg=output_residual,
        rotvecs=output_rotvecs,
        boundary=boundary,
        correction_local=correction_local.astype(np.float32),
        calibrated_root_first=calibrated_root,
        raw_root_reference=raw_reference,
        v18_depth=float(quantitative_case["roots"]["v18_calibrated_camera"][2]),
        da3_depth=float(quantitative_case["roots"]["da3_calibrated_camera"][2]),
        quantitative_case=quantitative_case,
        v14_2_case=v14_2_case,
    )


def continuity_score(case: dict) -> float:
    raw = case["continuity"]["hard_reset"]
    memory = case["continuity"]["shape_scale_local_pose_memory"]
    return float(
        raw["shape_jump_l2"]
        - memory["shape_jump_l2"]
        + 20.0 * (raw["body_scale_jump_abs"] - memory["body_scale_jump_abs"])
        + 0.1 * (raw["local_pose_jump_residual_deg"] - memory["local_pose_jump_residual_deg"])
    )


def select_roles(quantitative: dict, v14_2: dict, available: set[str]) -> dict[str, list[str]]:
    v14_map = {row["case_name"]: row for row in v14_2["cases"]}
    selected: dict[str, list[str]] = {}
    sources = sorted({row["source"] for row in quantitative["cases"]})
    for source in sources:
        rows = [
            row
            for row in quantitative["cases"]
            if row["source"] == source and row["case_name"] in available
        ]
        if not rows:
            continue

        def v14(row: dict) -> dict:
            return v14_map[row["case_name"]]

        def da3_human_gain(row: dict) -> float:
            methods = row["methods"]
            return float(
                methods["da3_camera_only"]["human"]["world_root_error_m"]
                - methods["da3_coupled_full"]["human"]["world_root_error_m"]
            )

        def da3_over_human(row: dict) -> float:
            methods = row["methods"]
            return float(
                methods["v18_coupled_full"]["human"]["world_root_error_m"]
                - methods["da3_coupled_full"]["human"]["world_root_error_m"]
            )

        def memory_accuracy_harm(row: dict) -> float:
            continuity = v14(row)["continuity"]
            raw = continuity["hard_reset"]
            memory = continuity["shape_scale_local_pose_memory"]
            return float(
                memory["gt_beta_error_l2"]
                - raw["gt_beta_error_l2"]
                + 10.0 * (memory["gt_body_scale_error_abs"] - raw["gt_body_scale_error_abs"])
            )

        def camera_human_conflict(row: dict) -> float:
            methods = row["methods"]
            return float(
                methods["v18_camera_only"]["human"]["world_root_error_m"]
                - methods["fixed_explicit"]["human"]["world_root_error_m"]
            )

        role_rows = {
            "continuity_gain": max(rows, key=lambda row: continuity_score(v14(row))),
            "continuity_neutral": min(rows, key=lambda row: abs(continuity_score(v14(row)))),
            "memory_regression": max(rows, key=memory_accuracy_harm),
            "camera_human_conflict": max(rows, key=camera_human_conflict),
            "coupled_success": max(rows, key=da3_human_gain),
            "coupled_failure": max(
                rows,
                key=lambda row: row["methods"]["da3_coupled_full"]["camera"]["translation_m"]
                + row["methods"]["da3_coupled_full"]["human"]["world_root_error_m"],
            ),
            "da3_wins": max(rows, key=da3_over_human),
            "human_projection_wins": min(rows, key=da3_over_human),
        }
        for role, row in role_rows.items():
            selected.setdefault(row["case_name"], []).append(role)
    return selected


def method_geometry(sequence: SequenceData, method_index: int, frame: int) -> dict:
    post = frame >= sequence.count
    continuity = method_index in (1, 3)
    coupled = method_index in (2, 3)
    local_pose = sequence.local_poses[frame]
    boundary = sequence.boundary if post else np.eye(4, dtype=np.float32)
    camera_pose = (boundary @ local_pose).astype(np.float32)
    raw_root_camera = sequence.raw_joints_camera[frame, 0]
    raw_root_local = local_pose[:3, :3] @ raw_root_camera + local_pose[:3, 3]
    root_local = raw_root_local + sequence.correction_local if post and coupled else raw_root_local
    root_world = boundary[:3, :3] @ root_local + boundary[:3, 3]
    if continuity:
        joints_centered = sequence.continuity_joints_centered[frame]
        vertices_centered = sequence.continuity_vertices_centered[frame]
    else:
        joints_centered = sequence.raw_joints_camera[frame] - raw_root_camera
        vertices_centered = sequence.raw_vertices_camera[frame] - raw_root_camera
    joints_world = joints_centered @ camera_pose[:3, :3].T + root_world
    vertices_world = vertices_centered @ camera_pose[:3, :3].T + root_world
    inverse_camera = np.linalg.inv(camera_pose)
    joints_camera = transform_points(inverse_camera, joints_world)
    vertices_camera = transform_points(inverse_camera, vertices_world)
    points_world = transform_points(boundary, sequence.points_local[frame])
    return {
        "camera_pose": camera_pose,
        "root_world": root_world.astype(np.float32),
        "joints_world": joints_world.astype(np.float32),
        "vertices_world": vertices_world.astype(np.float32),
        "joints_camera": joints_camera.astype(np.float32),
        "vertices_camera": vertices_camera.astype(np.float32),
        "points_world": points_world.astype(np.float32),
    }


class MeshOverlayRenderer:
    def __init__(self, faces: np.ndarray) -> None:
        self.faces = np.asarray(faces, dtype=np.int32)
        self.renderers: dict[tuple[int, int], pyrender.OffscreenRenderer] = {}

    def close(self) -> None:
        for renderer in self.renderers.values():
            renderer.delete()
        self.renderers.clear()

    def render(
        self,
        image: np.ndarray,
        vertices_camera: np.ndarray,
        joints_camera: np.ndarray,
        K: np.ndarray,
        color_rgb: tuple[int, int, int],
    ) -> np.ndarray:
        height, width = image.shape[:2]
        key = (width, height)
        if key not in self.renderers:
            self.renderers[key] = pyrender.OffscreenRenderer(width, height)
        flip = np.asarray([1.0, -1.0, -1.0], dtype=np.float32)
        mesh = trimesh.Trimesh(
            vertices=np.asarray(vertices_camera) * flip[None],
            faces=self.faces,
            process=False,
        )
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            roughnessFactor=0.75,
            baseColorFactor=tuple(float(value) / 255.0 for value in color_rgb) + (1.0,),
        )
        scene = pyrender.Scene(bg_color=(0.0, 0.0, 0.0, 0.0), ambient_light=(0.75, 0.75, 0.75))
        scene.add(pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True))
        camera = pyrender.IntrinsicsCamera(
            fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2])
        )
        scene.add(camera, pose=np.eye(4, dtype=np.float32))
        rgba, depth = self.renderers[key].render(
            scene, flags=pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SKIP_CULL_FACES
        )
        visible = np.isfinite(depth) & (depth > 0)
        output = image.copy()
        output[visible] = np.clip(
            0.52 * output[visible].astype(np.float32) + 0.48 * rgba[visible, :3].astype(np.float32),
            0,
            255,
        ).astype(np.uint8)
        projected = project_points(joints_camera, K)
        for first, second in body_edges():
            if first >= len(projected) or second >= len(projected):
                continue
            a, b = projected[first], projected[second]
            if np.isfinite(a).all() and np.isfinite(b).all():
                cv2.line(output, tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)), (245, 245, 245), 1, cv2.LINE_AA)
        for point in projected[:22]:
            if np.isfinite(point).all() and 0 <= point[0] < width and 0 <= point[1] < height:
                cv2.circle(output, tuple(np.rint(point).astype(int)), 2, (255, 255, 255), -1, cv2.LINE_AA)
        return output


def body_edges() -> tuple[tuple[int, int], ...]:
    return (
        (0, 1), (0, 2), (0, 3), (1, 4), (4, 7), (7, 10), (2, 5), (5, 8), (8, 11),
        (3, 6), (6, 9), (9, 12), (12, 15), (12, 13), (13, 16), (16, 18),
        (18, 20), (12, 14), (14, 17), (17, 19), (19, 21),
    )


def project_points(points: np.ndarray, K: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    result = np.full((len(points), 2), np.nan, dtype=np.float32)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0.05)
    result[valid, 0] = K[0, 0] * points[valid, 0] / points[valid, 2] + K[0, 2]
    result[valid, 1] = K[1, 1] * points[valid, 1] / points[valid, 2] + K[1, 2]
    return result


def label_panel(image: np.ndarray, label: str, frame: int, count: int) -> np.ndarray:
    band = np.full((42, image.shape[1], 3), 24, dtype=np.uint8)
    cv2.putText(band, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (244, 244, 244), 1, cv2.LINE_AA)
    if frame == count:
        cv2.putText(band, "CUT", (image.shape[1] - 52, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (80, 98, 246), 2, cv2.LINE_AA)
    return np.concatenate([band, image], axis=0)


def resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    width = int(round(image.shape[1] * height / image.shape[0]))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def encode_video(frames: list[np.ndarray], path: Path, fps: int) -> None:
    if not frames:
        raise ValueError(f"No frames for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    width += width % 2
    height += height % 2
    temp = path.with_name(path.stem + ".mpeg4.mp4")
    writer = cv2.VideoWriter(str(temp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {temp}")
    for frame in frames:
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.copyMakeBorder(
                frame,
                0,
                height - frame.shape[0],
                0,
                width - frame.shape[1],
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    subprocess.run(
        [
            "ffmpeg", "-loglevel", "error", "-y", "-i", str(temp), "-c:v", "libx264",
            "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
        ],
        check=True,
    )
    temp.unlink()


def render_rgb_overlay_video(
    sequence: SequenceData,
    faces: np.ndarray,
    output: Path,
    fps: int,
) -> None:
    renderer = MeshOverlayRenderer(faces)
    frames = []
    try:
        for frame_index, image in enumerate(sequence.images):
            panels = []
            for method_index, (label, color) in enumerate(zip(METHOD_LABELS, METHOD_COLORS)):
                geometry = method_geometry(sequence, method_index, frame_index)
                overlay = renderer.render(
                    image,
                    geometry["vertices_camera"],
                    geometry["joints_camera"],
                    sequence.intrinsics[frame_index],
                    color,
                )
                panels.append(label_panel(overlay, label, frame_index, sequence.count))
            target_height = min(panel.shape[0] for panel in panels)
            frames.append(np.concatenate([resize_to_height(panel, target_height) for panel in panels], axis=1))
    finally:
        renderer.close()
    encode_video(frames, output, fps)


def camera_frustum(pose: np.ndarray, size: float = 0.18) -> list[np.ndarray]:
    corners_camera = np.asarray(
        [[0, 0, 0], [-0.7, -0.5, 1], [0.7, -0.5, 1], [0.7, 0.5, 1], [-0.7, 0.5, 1]],
        dtype=np.float32,
    ) * size
    corners = transform_points(pose, corners_camera)
    return [
        corners[[0, 1]], corners[[0, 2]], corners[[0, 3]], corners[[0, 4]],
        corners[[1, 2, 3, 4, 1]],
    ]


def figure_rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    return np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)[..., :3].copy()


def world_limits(sequence: SequenceData) -> tuple[np.ndarray, float]:
    values = []
    for frame in range(len(sequence.images)):
        for method in (0, 3):
            geometry = method_geometry(sequence, method, frame)
            values.append(geometry["vertices_world"][::40])
            values.append(geometry["root_world"][None])
            values.append(geometry["camera_pose"][:3, 3][None])
    points = np.concatenate(values)
    lower, upper = np.percentile(points, [1, 99], axis=0)
    center = (lower + upper) * 0.5
    radius = min(max(float(np.max(upper - lower)) * 0.60, 1.25), 6.0)
    return center.astype(np.float32), radius


def render_fixed_world_video(
    sequence: SequenceData,
    faces: np.ndarray,
    output: Path,
    fps: int,
) -> None:
    center, radius = world_limits(sequence)
    face_subset = faces[::14]
    video_frames = []
    geometries = [
        [method_geometry(sequence, method, frame) for frame in range(len(sequence.images))]
        for method in range(4)
    ]
    for frame in range(len(sequence.images)):
        fig = plt.figure(figsize=(16, 7.2), dpi=100, facecolor="#f4f4f1")
        for method, (label, color) in enumerate(zip(METHOD_LABELS, METHOD_COLORS)):
            ax = fig.add_subplot(2, 2, method + 1, projection="3d")
            geometry = geometries[method][frame]
            points = geometry["points_world"]
            colors = sequence.point_colors[frame]
            if len(points):
                visible = np.max(np.abs(points - center[None]), axis=1) <= radius
                points = points[visible]
                colors = colors[visible]
                stride = max(1, len(points) // 1800)
                ax.scatter(
                    points[::stride, 0], points[::stride, 1], points[::stride, 2],
                    c=colors[::stride].astype(np.float32) / 255.0, s=0.7, alpha=0.11, depthshade=False,
                )
            triangles = geometry["vertices_world"][face_subset]
            collection = Poly3DCollection(
                triangles,
                facecolor=tuple(value / 255.0 for value in color) + (0.48,),
                edgecolor="none",
            )
            ax.add_collection3d(collection)
            vertices = geometry["vertices_world"][::28]
            ax.scatter(
                vertices[:, 0], vertices[:, 1], vertices[:, 2],
                color=tuple(value / 255.0 for value in color), s=2.2, alpha=0.78, depthshade=False,
            )
            joints = geometry["joints_world"][:22]
            ax.scatter(
                joints[:, 0], joints[:, 1], joints[:, 2], color="#fff7df", edgecolor="#20242a",
                linewidth=0.25, s=8, alpha=0.95, depthshade=False,
            )
            camera_path = np.stack([row["camera_pose"][:3, 3] for row in geometries[method][: frame + 1]])
            root_path = np.stack([row["root_world"] for row in geometries[method][: frame + 1]])
            ax.plot(camera_path[:, 0], camera_path[:, 1], camera_path[:, 2], color="#1f2937", linewidth=1.5)
            ax.plot(root_path[:, 0], root_path[:, 1], root_path[:, 2], color="#d94841", linewidth=2.2)
            for line in camera_frustum(geometry["camera_pose"]):
                ax.plot(line[:, 0], line[:, 1], line[:, 2], color="#20252b", linewidth=1.0)
            ax.set_xlim(center[0] - radius, center[0] + radius)
            ax.set_ylim(center[1] - radius, center[1] + radius)
            ax.set_zlim(center[2] - radius, center[2] + radius)
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=20, azim=-62)
            ax.set_title(label, fontsize=10, pad=3)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        phase = "POST-CUT" if frame >= sequence.count else "PRE-CUT"
        fig.suptitle(
            f"{sequence.name} | fixed world view | frame {frame + 1:02d}/{len(sequence.images):02d} | {phase}",
            fontsize=13,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        video_frames.append(figure_rgb(fig))
        plt.close(fig)
    encode_video(video_frames, output, fps)


def torso_heading_deg(joints_world: np.ndarray) -> float:
    left = 0.5 * (joints_world[1] + joints_world[16])
    right = 0.5 * (joints_world[2] + joints_world[17])
    lateral = left - right
    return float(np.degrees(np.arctan2(lateral[2], lateral[0])))


def timeline_values(sequence: SequenceData) -> dict[str, np.ndarray]:
    hard = [method_geometry(sequence, 0, index) for index in range(len(sequence.images))]
    coupled = [method_geometry(sequence, 3, index) for index in range(len(sequence.images))]
    return {
        "raw_beta": np.linalg.norm(sequence.raw_betas, axis=1),
        "memory_beta": np.linalg.norm(sequence.output_betas, axis=1),
        "raw_scale": sequence.raw_scales,
        "memory_scale": sequence.output_scales,
        "raw_pose": sequence.raw_local_residual_deg,
        "memory_pose": sequence.output_local_residual_deg,
        "raw_heading": np.asarray([torso_heading_deg(row["joints_world"]) for row in hard]),
        "memory_heading": np.asarray([torso_heading_deg(row["joints_world"]) for row in coupled]),
        "camera_only_root": np.stack([row["root_world"] for row in hard]),
        "coupled_root": np.stack([row["root_world"] for row in coupled]),
    }


def draw_timeline(sequence: SequenceData, values: dict[str, np.ndarray], until: int | None) -> plt.Figure:
    total = len(sequence.images)
    limit = total if until is None else until + 1
    x = np.arange(total)
    fig, axes = plt.subplots(3, 2, figsize=(13, 8), dpi=110, facecolor="#f7f7f4")
    specs = (
        ("SMPL-X beta norm", values["raw_beta"], values["memory_beta"], "L2"),
        ("Body scale", values["raw_scale"], values["memory_scale"], "m"),
        ("Local-pose residual", values["raw_pose"], values["memory_pose"], "deg"),
        ("Torso heading", values["raw_heading"], values["memory_heading"], "deg"),
    )
    for ax, (title, raw, memory, unit) in zip(axes.flat[:4], specs):
        ax.plot(x[:limit], raw[:limit], color="#d45b49", label="Hard reset", linewidth=1.8)
        ax.plot(x[:limit], memory[:limit], color="#2f8b57", label="Continuity memory", linewidth=1.8)
        ax.axvline(sequence.count - 0.5, color="#20242a", linestyle="--", linewidth=1.1)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(unit)
        ax.grid(alpha=0.2)
    ax = axes.flat[4]
    for axis, color in zip(range(3), ("#d14b4b", "#2b8a3e", "#2d6cc0")):
        ax.plot(x[:limit], values["camera_only_root"][:limit, axis], color=color, linestyle="--", alpha=0.55)
        ax.plot(x[:limit], values["coupled_root"][:limit, axis], color=color, linewidth=1.6, label="XYZ"[axis])
    ax.axvline(sequence.count - 0.5, color="#20242a", linestyle="--", linewidth=1.1)
    ax.set_title("World root: dashed camera-only, solid coupled", fontsize=10)
    ax.set_ylabel("m"); ax.grid(alpha=0.2); ax.legend(ncol=3, fontsize=8)
    ax = axes.flat[5]
    correction = np.linalg.norm(values["coupled_root"] - values["camera_only_root"], axis=1)
    ax.plot(x[:limit], correction[:limit], color="#7950a8", linewidth=2.0)
    ax.axvline(sequence.count - 0.5, color="#20242a", linestyle="--", linewidth=1.1)
    ax.set_title("Human correction magnitude", fontsize=10)
    ax.set_ylabel("m"); ax.grid(alpha=0.2)
    for ax in axes.flat:
        ax.set_xlim(0, total - 1)
        ax.set_xlabel("stream frame")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"V14.3 continuity timeline | {sequence.name}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def render_timeline(sequence: SequenceData, png: Path, video: Path, fps: int) -> None:
    values = timeline_values(sequence)
    fig = draw_timeline(sequence, values, None)
    fig.savefig(png, dpi=140)
    plt.close(fig)
    frames = []
    for index in range(len(sequence.images)):
        fig = draw_timeline(sequence, values, index)
        frames.append(figure_rgb(fig))
        plt.close(fig)
    encode_video(frames, video, fps)


def draw_difference(sequence: SequenceData, until: int) -> plt.Figure:
    values = timeline_values(sequence)
    x = np.arange(len(sequence.images))
    correction = np.linalg.norm(values["coupled_root"] - values["camera_only_root"], axis=1)
    q = sequence.quantitative_case["methods"]
    fig = plt.figure(figsize=(13, 6.8), dpi=110, facecolor="#f7f7f4")
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    raw = sequence.raw_root_reference
    da3 = sequence.calibrated_root_first
    v18 = np.asarray(sequence.quantitative_case["roots"]["v18_calibrated_camera"], dtype=np.float32)
    points = ((raw, "Raw Human3R", "#d45b49"), (v18, "Human projection", "#2d6cc0"), (da3, "DA3 / selected", "#2f8b57"))
    for point, label, color in points:
        ax3d.scatter(point[0], point[1], point[2], color=color, s=55, label=label)
    ax3d.plot([raw[0], da3[0]], [raw[1], da3[1]], [raw[2], da3[2]], color="#7950a8", linewidth=2.4)
    ax3d.set_xlabel("camera X"); ax3d.set_ylabel("camera Y"); ax3d.set_zlabel("depth Z")
    ax3d.set_title("One cut-time root calibration")
    ax3d.view_init(elev=20, azim=-58)
    ax3d.legend(fontsize=8)

    ax = fig.add_subplot(2, 2, 2)
    ax.plot(x[: until + 1], correction[: until + 1], color="#7950a8", linewidth=2.2)
    ax.axvline(sequence.count - 0.5, color="#20242a", linestyle="--")
    ax.set_xlim(0, len(x) - 1); ax.set_ylabel("m"); ax.set_title("Camera-only vs coupled human position")
    ax.grid(alpha=0.2)
    ax = fig.add_subplot(2, 2, 4)
    ax.axis("off")
    metrics = (
        f"Raw depth: {raw[2]:.3f} m\n"
        f"Human projection depth: {sequence.v18_depth:.3f} m\n"
        f"DA3 depth: {sequence.da3_depth:.3f} m\n"
        f"Cut-time root correction: {np.linalg.norm(da3 - raw):.3f} m\n\n"
        f"V18 coupled camera / human: "
        f"{q['v18_coupled_full']['camera']['translation_m']:.3f} / "
        f"{q['v18_coupled_full']['human']['world_root_error_m']:.3f} m\n"
        f"DA3 coupled camera / human: "
        f"{q['da3_coupled_full']['camera']['translation_m']:.3f} / "
        f"{q['da3_coupled_full']['human']['world_root_error_m']:.3f} m\n"
        f"DA3 scene discontinuity: {q['da3_coupled_full']['scene']['trimmed_mean_m']:.3f} m"
    )
    ax.text(0.02, 0.96, metrics, va="top", fontsize=11, linespacing=1.55, family="monospace")
    fig.suptitle(f"V14.3 root/depth/correction difference | {sequence.name}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def render_difference(sequence: SequenceData, png: Path, video: Path, fps: int) -> None:
    frames = []
    for index in range(len(sequence.images)):
        fig = draw_difference(sequence, index)
        if index == len(sequence.images) - 1:
            fig.savefig(png, dpi=140)
        frames.append(figure_rgb(fig))
        plt.close(fig)
    encode_video(frames, video, fps)


def case_metrics(case: dict) -> dict:
    methods = case["methods"]
    return {
        "fixed_camera": methods["fixed_explicit"]["camera"]["translation_m"],
        "v18_camera": methods["v18_coupled_full"]["camera"]["translation_m"],
        "v18_human": methods["v18_coupled_full"]["human"]["world_root_error_m"],
        "da3_camera": methods["da3_coupled_full"]["camera"]["translation_m"],
        "da3_human": methods["da3_coupled_full"]["human"]["world_root_error_m"],
        "da3_scene": methods["da3_coupled_full"]["scene"]["trimmed_mean_m"],
    }


def mean_projected_displacement(
    first: np.ndarray,
    second: np.ndarray,
    K: np.ndarray,
) -> float:
    first_2d = project_points(first, K)
    second_2d = project_points(second, K)
    valid = np.isfinite(first_2d).all(axis=1) & np.isfinite(second_2d).all(axis=1)
    return float(np.mean(np.linalg.norm(first_2d[valid] - second_2d[valid], axis=1))) if valid.any() else float("nan")


def visual_change_metrics(sequence: SequenceData) -> dict:
    joint_shift = []
    mesh_shift = []
    coupled_root_shift = []
    for frame in range(sequence.count, len(sequence.images)):
        hard = method_geometry(sequence, 0, frame)
        memory = method_geometry(sequence, 1, frame)
        coupled = method_geometry(sequence, 2, frame)
        K = sequence.intrinsics[frame]
        joint_shift.append(
            mean_projected_displacement(hard["joints_camera"][:22], memory["joints_camera"][:22], K)
        )
        mesh_shift.append(
            mean_projected_displacement(hard["vertices_camera"][::40], memory["vertices_camera"][::40], K)
        )
        coupled_root_shift.append(
            mean_projected_displacement(
                hard["joints_camera"][:1], coupled["joints_camera"][:1], K
            )
        )
    return {
        "continuity_joint_shift_mean_px": float(np.nanmean(joint_shift)),
        "continuity_mesh_shift_mean_px": float(np.nanmean(mesh_shift)),
        "coupled_root_shift_mean_px": float(np.nanmean(coupled_root_shift)),
    }


def write_html(output_dir: Path, records: list[dict], alignment_method: str) -> None:
    cards = []
    for record in records:
        roles = "".join(f"<span class='role'>{ROLE_LABELS[role]}</span>" for role in record["roles"])
        metrics = record["metrics"]
        visual = record["visual_change"]
        cards.append(
            f"""
<article class="case" data-source="{record['source']}">
  <header><div><p class="source">{record['source']}</p><h2>{record['case_name']}</h2></div><div class="roles">{roles}</div></header>
  <p class="metrics">Fixed camera {metrics['fixed_camera']:.3f} m &nbsp; | &nbsp; V18 camera/human {metrics['v18_camera']:.3f}/{metrics['v18_human']:.3f} m &nbsp; | &nbsp; DA3 camera/human {metrics['da3_camera']:.3f}/{metrics['da3_human']:.3f} m &nbsp; | &nbsp; DA3 scene {metrics['da3_scene']:.3f} m<br>Visual change: continuity joints/mesh {visual['continuity_joint_shift_mean_px']:.1f}/{visual['continuity_mesh_shift_mean_px']:.1f} px &nbsp; | &nbsp; coupled root {visual['coupled_root_shift_mean_px']:.1f} px</p>
  <div class="videos"><figure><video controls preload="metadata" src="{record['rgb_video']}"></video><figcaption>RGB mesh + joints</figcaption></figure><figure><video controls preload="metadata" src="{record['world_video']}"></video><figcaption>Fixed third-person world view</figcaption></figure><figure><video controls preload="metadata" src="{record['timeline_video']}"></video><figcaption>Shape / scale / pose / trajectory</figcaption></figure><figure><video controls preload="metadata" src="{record['difference_video']}"></video><figcaption>Depth and root correction</figcaption></figure></div>
  <div class="links"><a href="{record['timeline_png']}">Timeline PNG</a><a href="{record['difference_png']}">Difference PNG</a></div>
</article>"""
        )
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V14.3 Human-Camera Re-anchoring</title><style>
:root {{ color-scheme: light; font-family: Inter, Arial, sans-serif; color:#1f2428; background:#eef0ed; }} body {{ margin:0; }} .top {{ padding:24px 30px 18px; background:#fff; border-bottom:1px solid #ccd1cc; position:sticky; top:0; z-index:2; }} h1 {{ font-size:24px; margin:0 0 8px; letter-spacing:0; }} .top p {{ margin:4px 0; color:#555f59; }} nav {{ margin-top:14px; display:flex; gap:8px; flex-wrap:wrap; }} button {{ border:1px solid #aeb6b0; background:#fff; padding:7px 12px; border-radius:5px; cursor:pointer; }} button.active {{ background:#24342d; color:#fff; }} main {{ padding:22px 28px 48px; display:grid; gap:22px; }} .case {{ background:#fff; border:1px solid #cbd1cc; border-radius:6px; overflow:hidden; }} header {{ display:flex; gap:16px; justify-content:space-between; padding:16px 18px 10px; }} h2 {{ font-size:15px; margin:2px 0; overflow-wrap:anywhere; }} .source {{ text-transform:uppercase; font-size:11px; color:#6b746e; margin:0; }} .roles {{ display:flex; gap:5px; flex-wrap:wrap; justify-content:flex-end; }} .role {{ background:#e7ece8; border-radius:4px; padding:4px 7px; font-size:11px; }} .metrics {{ margin:0; padding:0 18px 13px; color:#4f5953; font-size:12px; }} .videos {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); border-top:1px solid #dde1de; }} figure {{ margin:0; padding:12px; border-right:1px solid #e2e5e3; border-bottom:1px solid #e2e5e3; }} video {{ display:block; width:100%; background:#171a18; aspect-ratio:16/9; }} figcaption {{ margin-top:7px; font-size:12px; color:#58615b; }} .links {{ padding:10px 16px 14px; display:flex; gap:14px; }} a {{ color:#175c45; font-size:12px; }} @media(max-width:850px) {{ .videos {{ grid-template-columns:1fr; }} header {{ display:block; }} .roles {{ justify-content:flex-start; margin-top:8px; }} .top {{ position:static; }} }}
</style></head><body><section class="top"><h1>V14.3 Projection-Consistent Human-Camera Re-anchoring</h1><p>Streaming protocol: DA3 runs once on 5 pre-cut + first post-cut frame; one fixed shot transform is reused. Visualization alignment: <strong>{alignment_method}</strong>.</p><p>Camera-human correction is evaluated separately from raw Human3R scene-scale consistency. Success, neutral, and failure cases are all included.</p><nav><button class="active" data-filter="all">All</button><button data-filter="avatarrex">AvatarReX</button><button data-filter="thuman">THuman</button><button data-filter="mvhuman100">MVHuman100</button><button data-filter="mvhuman200">MVHuman200</button></nav></section><main>{''.join(cards)}</main><script>
document.querySelectorAll('button').forEach(button=>button.addEventListener('click',()=>{{document.querySelectorAll('button').forEach(x=>x.classList.remove('active'));button.classList.add('active');const f=button.dataset.filter;document.querySelectorAll('.case').forEach(card=>card.style.display=(f==='all'||card.dataset.source===f)?'block':'none');}}));
</script></body></html>"""
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("V14.3 SMPL-X visualization requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    quantitative = json.loads(args.quantitative.read_text(encoding="utf-8"))
    v14_2 = json.loads(args.v14_2_report.read_text(encoding="utf-8"))
    quantitative_map = {row["case_name"]: row for row in quantitative["cases"]}
    v14_map = {row["case_name"]: row for row in v14_2["cases"]}
    available = {path.parent.name for path in args.cache_dir.glob("*/manifest.json")}
    selection = select_roles(quantitative, v14_2, available)
    requested = set(args.cases)
    names = sorted(name for name in selection if not requested or name in requested)
    if int(args.max_cases) > 0:
        names = names[: int(args.max_cases)]
    if not names:
        raise RuntimeError("No selected V14.3 cases have complete long-sequence caches")

    device = torch.device(args.device)
    layer = SMPL_Layer(type="smplx", gender="neutral", num_betas=10, kid=False, person_center="head").to(device).eval()
    faces = np.asarray(layer.bm_x.faces, dtype=np.int32)
    records = []
    for index, name in enumerate(names):
        print(f">> V14.3 visualize {index + 1}/{len(names)} {name}", flush=True)
        case_dir = args.output_dir / name
        case_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "rgb_video": case_dir / "rgb_mesh_overlay.mp4",
            "world_video": case_dir / "fixed_world_3d.mp4",
            "timeline_video": case_dir / "continuity_timeline.mp4",
            "timeline_png": case_dir / "continuity_timeline.png",
            "difference_video": case_dir / "root_depth_difference.mp4",
            "difference_png": case_dir / "root_depth_difference.png",
        }
        sequence = load_sequence(
            name, args.cache_dir, quantitative_map[name], v14_map[name], layer, device, args
        )
        if args.overwrite or not paths["rgb_video"].is_file():
            render_rgb_overlay_video(sequence, faces, paths["rgb_video"], int(args.fps))
        if args.overwrite or not paths["world_video"].is_file():
            render_fixed_world_video(sequence, faces, paths["world_video"], int(args.fps))
        if args.overwrite or not paths["timeline_video"].is_file() or not paths["timeline_png"].is_file():
            render_timeline(sequence, paths["timeline_png"], paths["timeline_video"], int(args.fps))
        if args.overwrite or not paths["difference_video"].is_file() or not paths["difference_png"].is_file():
            render_difference(sequence, paths["difference_png"], paths["difference_video"], int(args.fps))
        record = {
            "case_name": name,
            "source": sequence.source,
            "roles": selection[name],
            "metrics": case_metrics(quantitative_map[name]),
            "visual_change": visual_change_metrics(sequence),
            **{key: str(path.relative_to(args.output_dir)) for key, path in paths.items()},
        }
        records.append(record)
        (case_dir / "visualization.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8"
        )
    write_html(args.output_dir, records, str(args.alignment_method))
    summary = {
        "experiment": "V14.3 Human Continuity Visualization",
        "device": str(args.device),
        "alignment_method": str(args.alignment_method),
        "case_count": len(records),
        "cases": records,
        "streaming": {
            "da3_runs": "once at cut on 5 pre + 1 post frames",
            "boundary_updates_per_post_shot": 1,
            "post_frames_reuse_fixed_transform": True,
        },
    }
    (args.output_dir / "visualization_manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8"
    )
    print(f">> wrote {args.output_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
