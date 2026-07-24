#!/usr/bin/env python3
"""V14.2: canonical human memory for boundary alignment and continuity.

The probe reuses frozen V18 stream/keypoint caches.  Alignment always uses the
current pose, current 2D torso observations, and current intrinsics; only the
body proportions and physical-size reference vary.  Local-pose memory is
evaluated separately and never enters the camera-depth solve.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.stats import wilcoxon


REPO_ROOT = Path(__file__).resolve().parents[3]
for path in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from dust3r.utils.smpl_layer import SMPL_Layer  # noqa: E402


DEFAULT_ROOT = REPO_ROOT / "output" / "v14_2_canonical_human_memory"
TORSO_IDS = np.asarray([0, 1, 2, 12, 16, 17], dtype=np.int64)
SCALE_PAIRS = ((1, 2), (16, 17), (1, 16), (2, 17), (0, 1), (0, 2), (1, 4), (2, 5))
REFERENCE_VARIANTS = (
    "current_independent",
    "post_only_canonical",
    "mismatched_pre_post",
    "canonical_alpha025",
    "canonical_beta_only",
    "canonical_scale_only",
    "canonical_beta_scale",
    "last_pre",
    "historical_median",
    "first_full",
    "torso_complete",
    "min_reprojection",
    "shape_stable",
    "largest_pixels",
    "closest_view",
    "best_quality",
    "top3_consensus",
    "oracle_best_historical",
    "stale_memory",
    "wrong_video_memory",
    "mean_body_memory",
    "random_beta",
    "random_scale",
    "gt_beta_only",
    "gt_scale_only",
    "gt_beta_scale",
)
ALL_CANDIDATES = ("fixed_explicit", *REFERENCE_VARIANTS, "boundary_oracle")


@dataclass(frozen=True)
class BodyReference:
    beta: np.ndarray
    physical_scale: float
    beta_count: int = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stream_dir",
        type=Path,
        default=REPO_ROOT / "output/v18_human_metric_translation/stream_cache",
    )
    parser.add_argument(
        "--keypoint_dir",
        type=Path,
        default=REPO_ROOT / "output/v18_human_metric_translation/keypoint_cache",
    )
    parser.add_argument(
        "--scene_dir",
        type=Path,
        default=REPO_ROOT / "output/v18_human_metric_translation/v16_bound20_scene",
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_ROOT / "single_cut")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--keypoint_threshold", type=float, default=0.30)
    parser.add_argument("--memory_update_alpha", type=float, default=0.20)
    parser.add_argument("--alignment_alpha", type=float, default=0.25)
    parser.add_argument("--continuity_shape_alpha", type=float, default=0.25)
    parser.add_argument("--continuity_pose_alpha", type=float, default=0.15)
    parser.add_argument("--max_cases", type=int, default=0)
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_manifest(root: Path, pattern: str) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(glob.glob(str(root / pattern))):
        rows.extend(json.loads(Path(path).read_text(encoding="utf-8"))["cases"])
    return rows


def load_cases(args: argparse.Namespace) -> list[dict]:
    streams = load_manifest(args.stream_dir, "v18_stream_shard_*_of_*.json")
    keypoints = {
        str(row["case_name"]): row
        for row in load_manifest(args.keypoint_dir, "v18_keypoints_shard_*_of_*.json")
    }
    scenes = {
        str(row["case_name"]): row
        for row in load_manifest(args.scene_dir, "v16_candidates_shard_*_of_*.json")
    }
    if not streams or len(streams) != len(keypoints) or len(streams) != len(scenes):
        raise RuntimeError(
            f"Incomplete V18 caches: stream/keypoint/scene={len(streams)}/{len(keypoints)}/{len(scenes)}"
        )
    cases = []
    for row in streams:
        name = str(row["case_name"])
        enriched = dict(row)
        enriched["cache_path"] = resolve_path(row["cache_path"])
        enriched["keypoint_path"] = resolve_path(keypoints[name]["cache_path"])
        enriched["scene_case"] = scenes[name]
        cases.append(enriched)
    cases.sort(key=lambda row: str(row["case_name"]))
    max_cases = int(getattr(args, "max_cases", 0))
    return cases[:max_cases] if max_cases > 0 else cases


def build_layer(device: torch.device, betas: int) -> SMPL_Layer:
    return SMPL_Layer(
        type="smplx", gender="neutral", num_betas=betas, kid=False, person_center="head"
    ).to(device).eval()


def body_from_params(
    layer: SMPL_Layer,
    pose: np.ndarray,
    shape: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32)
    shape = np.asarray(shape, dtype=np.float32)
    if pose.ndim == 2:
        pose = pose[None]
    if shape.ndim == 1:
        shape = np.broadcast_to(shape[None], (len(pose), len(shape))).copy()
    K = torch.eye(3, device=device, dtype=torch.float32)[None].expand(len(pose), -1, -1)
    with torch.no_grad():
        output = layer(
            torch.as_tensor(pose, dtype=torch.float32, device=device),
            torch.as_tensor(shape, dtype=torch.float32, device=device),
            torch.zeros((len(pose), 3), dtype=torch.float32, device=device),
            None,
            None,
            K=K,
            expression=torch.zeros((len(pose), 10), dtype=torch.float32, device=device),
        )
    joints = output["smpl_j3d"].detach().float().cpu().numpy().astype(np.float32)
    return (joints - joints[:, :1]).astype(np.float32)


def physical_scale(joints: np.ndarray) -> float:
    joints = np.asarray(joints, dtype=np.float64)
    if len(joints) < 18:
        return float("nan")
    lengths = [np.linalg.norm(joints[a] - joints[b]) for a, b in SCALE_PAIRS]
    return float(np.mean(lengths))


def bodies_with_reference(
    layer: SMPL_Layer,
    poses: np.ndarray,
    beta: np.ndarray,
    target_scales: np.ndarray | float,
    device: torch.device,
) -> np.ndarray:
    bodies = body_from_params(layer, poses, beta, device)
    scales = np.asarray(target_scales, dtype=np.float32)
    if scales.ndim == 0:
        scales = np.full(len(bodies), float(scales), dtype=np.float32)
    output = []
    for body, target in zip(bodies, scales):
        current = physical_scale(body)
        factor = 1.0 if not np.isfinite(current) or current < 1e-8 else float(target) / current
        output.append((body * factor).astype(np.float32))
    return np.stack(output)


def project(points: np.ndarray, root: np.ndarray, K: np.ndarray) -> np.ndarray:
    camera = np.asarray(points, dtype=np.float64) + np.asarray(root, dtype=np.float64)[None]
    z = np.maximum(camera[:, 2], 1e-5)
    return np.stack(
        [K[0, 0] * camera[:, 0] / z + K[0, 2], K[1, 1] * camera[:, 1] / z + K[1, 2]],
        axis=1,
    )


def estimate_root_translation(
    body: np.ndarray,
    observed: np.ndarray,
    confidence: np.ndarray,
    K: np.ndarray,
    initial: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, dict]:
    ids = TORSO_IDS[(TORSO_IDS < len(body)) & (TORSO_IDS < len(observed))]
    valid = (
        np.isfinite(body[ids]).all(axis=1)
        & np.isfinite(observed[ids]).all(axis=1)
        & (confidence[ids] >= threshold)
    )
    ids = ids[valid]
    if len(ids) < 4:
        return np.asarray(initial, dtype=np.float32), {
            "status": "too_few_joints",
            "valid_joints": int(len(ids)),
            "reprojection_error_px": float("inf"),
        }
    weights = np.sqrt(np.clip(confidence[ids], 0.05, 1.0))[:, None]

    def residual(root: np.ndarray) -> np.ndarray:
        return ((project(body[ids], root, K) - observed[ids]) * weights).reshape(-1)

    x0 = np.asarray(initial, dtype=np.float64).copy()
    x0[2] = np.clip(x0[2], 0.30, 20.0)
    result = least_squares(
        residual,
        x0,
        bounds=(np.asarray([-15.0, -15.0, 0.20]), np.asarray([15.0, 15.0, 30.0])),
        loss="soft_l1",
        f_scale=8.0,
        max_nfev=100,
    )
    root = result.x.astype(np.float32)
    reprojection = np.linalg.norm(project(body[ids], root, K) - observed[ids], axis=1)
    return root, {
        "status": "ok" if result.success else "optimizer_failed",
        "valid_joints": int(len(ids)),
        "reprojection_error_px": float(np.median(reprojection)),
    }


def transform_point(pose: np.ndarray, point: np.ndarray) -> np.ndarray:
    return (pose[:3, :3] @ np.asarray(point) + pose[:3, 3]).astype(np.float32)


def camera_pose_from_human(rotation: np.ndarray, world_root: np.ndarray, camera_root: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = np.asarray(rotation, dtype=np.float32)
    pose[:3, 3] = np.asarray(world_root) - pose[:3, :3] @ np.asarray(camera_root)
    return pose


def boundary_from_camera_pose(camera_pose: np.ndarray, local_pose: np.ndarray) -> np.ndarray:
    return (np.asarray(camera_pose) @ np.linalg.inv(np.asarray(local_pose))).astype(np.float32)


def rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first)[:3, :3].T @ np.asarray(second)[:3, :3]
    return float(np.degrees(np.arccos(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))))


def direction_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator < 1e-8:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))))


def evaluate_boundary(
    boundary: np.ndarray,
    new_pose: np.ndarray,
    target_pose: np.ndarray,
    gt_boundary: np.ndarray,
) -> dict:
    camera_pose = np.asarray(boundary) @ np.asarray(new_pose)
    delta_world = camera_pose[:3, 3] - target_pose[:3, 3]
    delta_target = target_pose[:3, :3].T @ delta_world
    predicted_scale = float(np.linalg.norm(boundary[:3, 3]))
    target_scale = float(np.linalg.norm(gt_boundary[:3, 3]))
    return {
        "camera_translation_error_m": float(np.linalg.norm(delta_world)),
        "camera_rotation_error_deg": rotation_error_deg(camera_pose, target_pose),
        "viewing_direction_error_m": float(abs(delta_target[2])),
        "transverse_error_m": float(np.linalg.norm(delta_target[:2])),
        "vertical_error_m": float(abs(delta_target[1])),
        "translation_error_target_xyz_m": np.abs(delta_target).astype(float).tolist(),
        "translation_direction_error_deg": direction_error_deg(boundary[:3, 3], gt_boundary[:3, 3]),
        "translation_scale_error_m": float(abs(predicted_scale - target_scale)),
        "translation_log_scale_error": float(
            abs(math.log(max(predicted_scale, 1e-8) / max(target_scale, 1e-8)))
        ),
        "transform": np.asarray(boundary).astype(float).tolist(),
    }


def ema(values: np.ndarray, alpha: float) -> np.ndarray:
    result = np.asarray(values[0], dtype=np.float32).copy()
    for value in values[1:]:
        result += float(alpha) * (np.asarray(value, dtype=np.float32) - result)
    return result


def blend_rotations(current: np.ndarray, memory: np.ndarray, alpha: float) -> np.ndarray:
    blended = np.asarray(current) + float(alpha) * (np.asarray(memory) - np.asarray(current))
    u, _, vh = np.linalg.svd(blended)
    rotation = u @ vh
    negative = np.linalg.det(rotation) < 0.0
    if np.any(negative):
        u = u.copy()
        u[negative, :, -1] *= -1.0
        rotation = u @ vh
    return rotation.astype(np.float32)


def rotation_batch_error(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.swapaxes(first, -1, -2) @ second
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.mean(np.degrees(np.arccos(cosine))))


def torso_orientation_group(joints: np.ndarray) -> str:
    joints = np.asarray(joints, dtype=np.float64)
    up = 0.5 * (joints[16] + joints[17]) - 0.5 * (joints[1] + joints[2])
    right = joints[17] - joints[16]
    up /= max(float(np.linalg.norm(up)), 1e-8)
    right /= max(float(np.linalg.norm(right)), 1e-8)
    forward = np.cross(right, up)
    forward /= max(float(np.linalg.norm(forward)), 1e-8)
    if abs(float(forward[2])) < 0.5:
        return "side"
    return "front" if float(forward[2]) < 0.0 else "back"


def view_feature(points: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)[TORSO_IDS]
    confidence = np.asarray(confidence, dtype=np.float32)[TORSO_IDS]
    center = np.sum(points * confidence[:, None], axis=0) / max(float(np.sum(confidence)), 1e-6)
    centered = points - center
    scale = max(float(np.linalg.norm(centered, axis=1).max()), 1.0)
    return (centered / scale).reshape(-1)


def reference_quality(
    old_shapes: np.ndarray,
    keypoint: dict[str, np.ndarray],
    reprojection: list[dict],
    image_height: int,
    threshold: float,
) -> tuple[np.ndarray, dict[str, int]]:
    median = np.median(old_shapes, axis=0)
    confidence = keypoint["old_confidence"][:, TORSO_IDS]
    visibility = np.mean(confidence >= threshold, axis=1)
    torso_confidence = np.mean(confidence, axis=1)
    detector_score = np.clip(keypoint["old_scores"], 0.0, 1.0)
    box_height = np.maximum(keypoint["old_boxes"][:, 3] - keypoint["old_boxes"][:, 1], 0.0)
    box_fraction = np.clip(box_height / max(float(image_height), 1.0), 0.0, 1.0)
    reprojection_quality = np.asarray(
        [1.0 / (1.0 + (float(row["reprojection_error_px"]) / 10.0) ** 2) for row in reprojection]
    )
    shape_distance = np.linalg.norm(old_shapes - median[None], axis=1)
    shape_quality = np.exp(-shape_distance / max(float(np.median(shape_distance) + 1e-4), 1e-4))
    score = (
        0.20 * visibility
        + 0.20 * torso_confidence
        + 0.15 * detector_score
        + 0.15 * box_fraction
        + 0.20 * reprojection_quality
        + 0.10 * shape_quality
    )
    indices = {
        "first_full": int(np.flatnonzero(visibility >= 0.99)[0]) if np.any(visibility >= 0.99) else int(np.argmax(visibility)),
        "torso_complete": int(np.argmax(visibility + 0.05 * torso_confidence)),
        "min_reprojection": int(np.argmax(reprojection_quality)),
        "shape_stable": int(np.argmin(shape_distance)),
        "largest_pixels": int(np.argmax(box_fraction)),
        "best_quality": int(np.argmax(score)),
    }
    return score.astype(np.float32), indices


def scene_transform(case: dict) -> np.ndarray:
    row = case["scene_case"]["fixed_candidates"]["fixed_torso_motion_1f_resolve_t"]
    return np.asarray(row["transform"], dtype=np.float32)


def make_reference(beta: np.ndarray, scale: float) -> BodyReference:
    beta = np.asarray(beta, dtype=np.float32)
    return BodyReference(beta=beta, physical_scale=float(scale), beta_count=int(len(beta)))


def body_pair_for_variant(
    name: str,
    reference: BodyReference,
    old_pose: np.ndarray,
    new_pose: np.ndarray,
    current_old_body: np.ndarray,
    current_new_body: np.ndarray,
    current_old_beta: np.ndarray,
    current_new_beta: np.ndarray,
    current_old_scale: float,
    current_new_scale: float,
    layer10: SMPL_Layer,
    layer11: SMPL_Layer,
    device: torch.device,
    alignment_alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    if name == "current_independent":
        return current_old_body, current_new_body
    layer = layer11 if reference.beta_count == 11 else layer10
    if name == "post_only_canonical":
        new = bodies_with_reference(layer, new_pose, reference.beta, reference.physical_scale, device)[0]
        return current_old_body, new
    if name == "canonical_alpha025":
        old_beta = current_old_beta + alignment_alpha * (reference.beta[:10] - current_old_beta)
        new_beta = current_new_beta + alignment_alpha * (reference.beta[:10] - current_new_beta)
        old_scale = current_old_scale + alignment_alpha * (reference.physical_scale - current_old_scale)
        new_scale = current_new_scale + alignment_alpha * (reference.physical_scale - current_new_scale)
        old = bodies_with_reference(layer10, old_pose, old_beta, old_scale, device)[0]
        new = bodies_with_reference(layer10, new_pose, new_beta, new_scale, device)[0]
        return old, new
    if name == "canonical_beta_only" or name == "gt_beta_only":
        old = bodies_with_reference(layer, old_pose, reference.beta, current_old_scale, device)[0]
        new = bodies_with_reference(layer, new_pose, reference.beta, current_new_scale, device)[0]
        return old, new
    if name == "canonical_scale_only" or name == "gt_scale_only":
        old = bodies_with_reference(layer10, old_pose, current_old_beta, reference.physical_scale, device)[0]
        new = bodies_with_reference(layer10, new_pose, current_new_beta, reference.physical_scale, device)[0]
        return old, new
    old = bodies_with_reference(layer, old_pose, reference.beta, reference.physical_scale, device)[0]
    new = bodies_with_reference(layer, new_pose, reference.beta, reference.physical_scale, device)[0]
    return old, new


def solve_candidate(
    old_body: np.ndarray,
    new_body: np.ndarray,
    data: dict,
    camera_rotation: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, dict]:
    old_root, old_diag = estimate_root_translation(
        old_body,
        data["old_keypoints"][-1],
        data["old_confidence"][-1],
        data["old_intrinsics"][-1],
        data["old_raw_roots"][-1],
        threshold,
    )
    new_root, new_diag = estimate_root_translation(
        new_body,
        data["new_keypoints"],
        data["new_confidence"],
        data["new_intrinsics"],
        data["new_raw_root"],
        threshold,
    )
    world_root = transform_point(data["old_pose"][-1], old_root)
    camera_pose = camera_pose_from_human(camera_rotation, world_root, new_root)
    boundary = boundary_from_camera_pose(camera_pose, data["new_pose"])
    gt_camera_root = data["new_gt_camera_root"]
    return boundary, {
        "old_camera_root": old_root.astype(float).tolist(),
        "new_camera_root": new_root.astype(float).tolist(),
        "new_root_position_error_m": float(np.linalg.norm(new_root - gt_camera_root)),
        "new_root_depth_error_m": float(abs(new_root[2] - gt_camera_root[2])),
        "new_root_transverse_error_m": float(np.linalg.norm(new_root[:2] - gt_camera_root[:2])),
        "old_reprojection": old_diag,
        "new_reprojection": new_diag,
        "predicted_world_root": world_root.astype(float).tolist(),
        "world_root_motion_error_m": float(np.linalg.norm(world_root - data["new_gt_world_root"])),
    }


def geometry_continuity(boundary: np.ndarray, data: dict) -> dict:
    pre = transform_point(data["old_pose"][-1], data["old_raw_roots"][-1])
    post_camera_pose = np.asarray(boundary) @ data["new_pose"]
    post = transform_point(post_camera_pose, data["new_raw_root"])
    gt_delta = data["new_gt_world_root"] - data["old_gt_world_root"]
    predicted_delta = post - pre
    return {
        "visible_root_jump_m": float(np.linalg.norm(predicted_delta)),
        "visible_root_jump_residual_m": float(np.linalg.norm(predicted_delta - gt_delta)),
        "visible_root_world_error_m": float(np.linalg.norm(post - data["new_gt_world_root"])),
    }


def continuity_metrics(data: dict, canonical: BodyReference, shape_alpha: float, pose_alpha: float) -> dict:
    old_beta = data["old_shape"][-1]
    new_beta = data["new_shape"]
    old_scale = physical_scale(data["old_bodies"][-1])
    new_scale = physical_scale(data["new_body"])
    output_beta = new_beta + shape_alpha * (canonical.beta[:10] - new_beta)
    output_scale = new_scale + shape_alpha * (canonical.physical_scale - new_scale)

    old_rotmat = Rotation.from_rotvec(data["old_rotvec"][-1, 1:].reshape(-1, 3)).as_matrix()
    new_rotmat = Rotation.from_rotvec(data["new_rotvec"][1:].reshape(-1, 3)).as_matrix()
    gt_old = Rotation.from_rotvec(data["old_gt_rotvec"][-1, 1:].reshape(-1, 3)).as_matrix()
    gt_new = Rotation.from_rotvec(data["new_gt_rotvec"][1:].reshape(-1, 3)).as_matrix()
    history = [Rotation.from_rotvec(pose[1:].reshape(-1, 3)).as_matrix() for pose in data["old_rotvec"]]
    memory_rotmat = history[0].astype(np.float32)
    for value in history[1:]:
        memory_rotmat = blend_rotations(memory_rotmat, value, 0.20)
    output_rotmat = blend_rotations(new_rotmat, memory_rotmat, pose_alpha)
    predicted_jump = rotation_batch_error(old_rotmat, output_rotmat)
    target_jump = rotation_batch_error(gt_old, gt_new)
    gt_scale = physical_scale(data["new_gt_body"])
    return {
        "shape_jump_l2": float(np.linalg.norm(output_beta - old_beta)),
        "body_scale_jump_abs": float(abs(output_scale - old_scale)),
        "local_pose_jump_residual_deg": float(abs(predicted_jump - target_jump)),
        "gt_beta_error_l2": float(np.linalg.norm(output_beta - data["new_gt_shape"][:10])),
        "gt_body_scale_error_abs": float(abs(output_scale - gt_scale)),
        "output_beta": output_beta.astype(float).tolist(),
        "output_physical_scale": float(output_scale),
    }


def run_case(
    case: dict,
    wrong_case: dict,
    layer10: SMPL_Layer,
    layer11: SMPL_Layer,
    device: torch.device,
    args: argparse.Namespace,
    neutral_scale: float,
) -> dict:
    with np.load(case["cache_path"]) as stream, np.load(case["keypoint_path"]) as keypoint:
        data = {name: np.asarray(stream[name]) for name in stream.files}
        data.update({name: np.asarray(keypoint[name]) for name in keypoint.files})
    with np.load(wrong_case["cache_path"]) as wrong:
        wrong_shape = np.asarray(wrong["old_shape"], dtype=np.float32)
        wrong_joints = np.asarray(wrong["old_joints_camera"], dtype=np.float32)

    old_bodies = data["old_joints_camera"] - data["old_joints_camera"][:, :1]
    new_body = data["new_joints_camera"] - data["new_joints_camera"][:1]
    old_gt_bodies = data["old_gt_joints_camera"] - data["old_gt_joints_camera"][:, :1]
    new_gt_body = data["new_gt_joints_camera"] - data["new_gt_joints_camera"][:1]
    old_scales = np.asarray([physical_scale(body) for body in old_bodies], dtype=np.float32)
    new_scale = physical_scale(new_body)
    gt_scale = physical_scale(new_gt_body)
    wrong_bodies = wrong_joints - wrong_joints[:, :1]
    wrong_scales = np.asarray([physical_scale(body) for body in wrong_bodies], dtype=np.float32)
    data.update(
        {
            "old_bodies": old_bodies,
            "new_body": new_body,
            "old_raw_roots": data["old_joints_camera"][:, 0],
            "new_raw_root": data["new_joints_camera"][0],
            "new_gt_camera_root": data["new_gt_joints_camera"][0],
            "old_gt_world_root": data["old_gt_joints_target_world"][-1, 0],
            "new_gt_world_root": data["new_gt_joints_target_world"][0],
            "old_gt_rotvec": data["old_gt_pose53_camera"],
            "new_gt_rotvec": data["new_gt_pose53_camera"],
            "new_gt_body": new_gt_body,
        }
    )

    own_roots = []
    own_diags = []
    for body, points, confidence, K, initial in zip(
        old_bodies,
        data["old_keypoints"],
        data["old_confidence"],
        data["old_intrinsics"],
        data["old_raw_roots"],
    ):
        root, diag = estimate_root_translation(
            body, points, confidence, K, initial, float(args.keypoint_threshold)
        )
        own_roots.append(root)
        own_diags.append(diag)
    quality, quality_indices = reference_quality(
        data["old_shape"],
        data,
        own_diags,
        int(data["new_image"].shape[0]),
        float(args.keypoint_threshold),
    )
    current_feature = view_feature(data["new_keypoints"], data["new_confidence"])
    old_features = np.stack(
        [view_feature(points, confidence) for points, confidence in zip(data["old_keypoints"], data["old_confidence"])]
    )
    quality_indices["closest_view"] = int(np.argmin(np.linalg.norm(old_features - current_feature[None], axis=1)))

    canonical_beta = ema(data["old_shape"], float(args.memory_update_alpha))
    canonical_scale = float(ema(old_scales[:, None], float(args.memory_update_alpha))[0])
    median_beta = np.median(data["old_shape"], axis=0).astype(np.float32)
    median_scale = float(np.median(old_scales))
    references: dict[str, BodyReference] = {
        "post_only_canonical": make_reference(canonical_beta, canonical_scale),
        "mismatched_pre_post": make_reference(canonical_beta, canonical_scale),
        "canonical_alpha025": make_reference(canonical_beta, canonical_scale),
        "canonical_beta_only": make_reference(canonical_beta, canonical_scale),
        "canonical_scale_only": make_reference(canonical_beta, canonical_scale),
        "canonical_beta_scale": make_reference(canonical_beta, canonical_scale),
        "last_pre": make_reference(data["old_shape"][-1], old_scales[-1]),
        "historical_median": make_reference(median_beta, median_scale),
        "stale_memory": make_reference(data["old_shape"][0], old_scales[0]),
        "wrong_video_memory": make_reference(
            ema(wrong_shape, float(args.memory_update_alpha)),
            float(ema(wrong_scales[:, None], float(args.memory_update_alpha))[0]),
        ),
        "mean_body_memory": make_reference(np.zeros(10, dtype=np.float32), neutral_scale),
    }
    for name, index in quality_indices.items():
        references[name] = make_reference(data["old_shape"][index], old_scales[index])
    top3 = np.argsort(quality)[-3:]
    references["top3_consensus"] = make_reference(
        np.median(data["old_shape"][top3], axis=0), float(np.median(old_scales[top3]))
    )
    seed = int.from_bytes(str(case["case_name"]).encode("utf-8")[:8].ljust(8, b"0"), "little")
    rng = np.random.default_rng(seed)
    references["random_beta"] = make_reference(rng.normal(0.0, 1.5, size=10), canonical_scale)
    references["random_scale"] = make_reference(
        canonical_beta, canonical_scale * (0.75 if seed % 2 == 0 else 1.25)
    )
    gt_beta = np.median(data["old_gt_shape"], axis=0).astype(np.float32)
    references["gt_beta_only"] = make_reference(gt_beta, gt_scale)
    references["gt_scale_only"] = make_reference(canonical_beta, gt_scale)
    references["gt_beta_scale"] = make_reference(gt_beta, gt_scale)

    scene = scene_transform(case)
    scene_pose = scene @ data["new_pose"]
    camera_rotation = scene_pose[:3, :3]
    candidates: dict[str, np.ndarray] = {
        "fixed_explicit": data["fixed_transform"].astype(np.float32),
        "boundary_oracle": data["gt_boundary"].astype(np.float32),
    }
    root_diagnostics: dict[str, dict] = {}
    current_boundary, current_diag = solve_candidate(
        old_bodies[-1], new_body, data, camera_rotation, float(args.keypoint_threshold)
    )
    candidates["current_independent"] = current_boundary
    root_diagnostics["current_independent"] = current_diag

    historical_solutions: list[tuple[np.ndarray, dict]] = []
    for index in range(len(data["old_shape"])):
        reference = make_reference(data["old_shape"][index], old_scales[index])
        pair = body_pair_for_variant(
            "historical_reference",
            reference,
            data["old_rotvec"][-1],
            data["new_rotvec"],
            old_bodies[-1],
            new_body,
            data["old_shape"][-1],
            data["new_shape"],
            old_scales[-1],
            new_scale,
            layer10,
            layer11,
            device,
            float(args.alignment_alpha),
        )
        historical_solutions.append(
            solve_candidate(*pair, data, camera_rotation, float(args.keypoint_threshold))
        )
    oracle_index = int(
        np.argmin([row[1]["new_root_depth_error_m"] for row in historical_solutions])
    )
    references["oracle_best_historical"] = make_reference(
        data["old_shape"][oracle_index], old_scales[oracle_index]
    )

    for name in REFERENCE_VARIANTS:
        if name == "current_independent":
            continue
        reference = references[name]
        if name == "mismatched_pre_post":
            old_body = bodies_with_reference(
                layer10, data["old_rotvec"][-1], canonical_beta, canonical_scale, device
            )[0]
            wrong_reference = references["wrong_video_memory"]
            post_body = bodies_with_reference(
                layer10,
                data["new_rotvec"],
                wrong_reference.beta,
                wrong_reference.physical_scale,
                device,
            )[0]
        else:
            old_body, post_body = body_pair_for_variant(
                name,
                reference,
                data["old_rotvec"][-1],
                data["new_rotvec"],
                old_bodies[-1],
                new_body,
                data["old_shape"][-1],
                data["new_shape"],
                old_scales[-1],
                new_scale,
                layer10,
                layer11,
                device,
                float(args.alignment_alpha),
            )
        candidates[name], root_diagnostics[name] = solve_candidate(
            old_body, post_body, data, camera_rotation, float(args.keypoint_threshold)
        )

    evaluated = {}
    for name, transform in candidates.items():
        row = evaluate_boundary(transform, data["new_pose"], data["target_pose"], data["gt_boundary"])
        row.update(geometry_continuity(transform, data))
        if name in root_diagnostics:
            row["human_projection"] = root_diagnostics[name]
        evaluated[name] = row

    continuity_raw = continuity_metrics(data, make_reference(canonical_beta, canonical_scale), 0.0, 0.0)
    continuity_shape = continuity_metrics(
        data,
        make_reference(canonical_beta, canonical_scale),
        float(args.continuity_shape_alpha),
        0.0,
    )
    continuity_full = continuity_metrics(
        data,
        make_reference(canonical_beta, canonical_scale),
        float(args.continuity_shape_alpha),
        float(args.continuity_pose_alpha),
    )
    main_alignment = "canonical_beta_scale"
    quadrants = {
        "no_memory": {"alignment": "current_independent", "continuity": "raw"},
        "continuity_only": {"alignment": "current_independent", "continuity": "canonical"},
        "alignment_only": {"alignment": main_alignment, "continuity": "raw"},
        "alignment_continuity": {"alignment": main_alignment, "continuity": "canonical"},
    }
    for row in quadrants.values():
        alignment = evaluated[row["alignment"]]
        continuity = continuity_full if row["continuity"] == "canonical" else continuity_raw
        row.update(
            camera_translation_error_m=alignment["camera_translation_error_m"],
            viewing_direction_error_m=alignment["viewing_direction_error_m"],
            visible_root_jump_residual_m=alignment["visible_root_jump_residual_m"],
            shape_jump_l2=continuity["shape_jump_l2"],
            body_scale_jump_abs=continuity["body_scale_jump_abs"],
            local_pose_jump_residual_deg=continuity["local_pose_jump_residual_deg"],
            gt_beta_error_l2=continuity["gt_beta_error_l2"],
        )

    local_post_root = transform_point(data["new_pose"], data["new_raw_root"])
    aligned_post_root = transform_point(candidates[main_alignment], local_post_root)
    oracle_post_root = transform_point(data["gt_boundary"], local_post_root)
    commits = {
        "immediate_commit": float(np.linalg.norm(local_post_root - data["new_gt_world_root"])),
        "align_then_commit": float(np.linalg.norm(aligned_post_root - data["new_gt_world_root"])),
        "gt_boundary_align_then_commit": float(np.linalg.norm(oracle_post_root - data["new_gt_world_root"])),
    }
    return {
        "case_name": case["case_name"],
        "source": case["source"],
        "record": case["record"],
        "groups": {
            "view_angle_deg": float(case["record"].get("view_angle_deg", float("nan"))),
            "body_fraction": float(
                max(data["new_box"][3] - data["new_box"][1], 0.0) / max(data["new_image"].shape[0], 1)
            ),
            "torso_visible": int(np.sum(data["new_confidence"][TORSO_IDS] >= args.keypoint_threshold)),
            "post_shape_error_l2": float(np.linalg.norm(data["new_shape"] - data["new_gt_shape"][:10])),
            "post_body_scale_error_abs": float(abs(new_scale - gt_scale)),
            "initial_translation_error_m": evaluated["fixed_explicit"]["camera_translation_error_m"],
            "initial_viewing_error_m": evaluated["fixed_explicit"]["viewing_direction_error_m"],
            "orientation": torso_orientation_group(data["new_joints_camera"]),
            "torso_visibility": (
                "full" if int(np.sum(data["new_confidence"][TORSO_IDS] >= args.keypoint_threshold)) == len(TORSO_IDS) else "partial"
            ),
        },
        "memory": {
            "canonical_beta": canonical_beta.astype(float).tolist(),
            "canonical_physical_scale": canonical_scale,
            "predicted_scale_history": old_scales.astype(float).tolist(),
            "predicted_scale_history_std": float(np.std(old_scales)),
            "shape_history_std": float(np.mean(np.std(data["old_shape"], axis=0))),
            "quality_scores": quality.astype(float).tolist(),
            "reference_indices": {**quality_indices, "oracle_best_historical": oracle_index},
            "wrong_video_case": wrong_case["case_name"],
        },
        "candidates": evaluated,
        "continuity": {
            "hard_reset": continuity_raw,
            "shape_scale_memory": continuity_shape,
            "shape_scale_local_pose_memory": continuity_full,
        },
        "quadrants": quadrants,
        "commit": commits,
    }


def distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan"), "p95": float("nan")}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
    }


def aggregate_candidate(cases: list[dict], name: str) -> dict:
    rows = [case["candidates"][name] for case in cases]
    current = [case["candidates"]["current_independent"] for case in cases]
    output = {
        "count": len(rows),
        "translation_m": distribution([row["camera_translation_error_m"] for row in rows]),
        "rotation_deg": distribution([row["camera_rotation_error_deg"] for row in rows]),
        "viewing_direction_m": distribution([row["viewing_direction_error_m"] for row in rows]),
        "transverse_m": distribution([row["transverse_error_m"] for row in rows]),
        "scale_m": distribution([row["translation_scale_error_m"] for row in rows]),
        "log_scale": distribution([row["translation_log_scale_error"] for row in rows]),
        "visible_root_jump_m": distribution([row["visible_root_jump_m"] for row in rows]),
        "visible_root_jump_residual_m": distribution([row["visible_root_jump_residual_m"] for row in rows]),
        "translation_catastrophic_rate": float(np.mean([row["camera_translation_error_m"] > 1.0 for row in rows])),
        "success_rate": float(
            np.mean(
                [
                    row["camera_translation_error_m"] < 0.25 and row["camera_rotation_error_deg"] < 5.0
                    for row in rows
                ]
            )
        ),
        "harmful_rate_vs_current": float(
            np.mean(
                [
                    row["camera_translation_error_m"] > base["camera_translation_error_m"] + 0.05
                    for row, base in zip(rows, current)
                ]
            )
        ),
    }
    projection = [row.get("human_projection") for row in rows]
    if all(row is not None for row in projection):
        output["human_camera_root"] = {
            "position_m": distribution([row["new_root_position_error_m"] for row in projection]),
            "depth_m": distribution([row["new_root_depth_error_m"] for row in projection]),
            "transverse_m": distribution([row["new_root_transverse_error_m"] for row in projection]),
            "reprojection_px": distribution(
                [row["new_reprojection"]["reprojection_error_px"] for row in projection]
            ),
            "world_motion_m": distribution([row["world_root_motion_error_m"] for row in projection]),
        }
    return output


def aggregate_continuity(cases: list[dict], name: str) -> dict:
    rows = [case["continuity"][name] for case in cases]
    return {
        key: distribution([row[key] for row in rows])
        for key in (
            "shape_jump_l2",
            "body_scale_jump_abs",
            "local_pose_jump_residual_deg",
            "gt_beta_error_l2",
            "gt_body_scale_error_abs",
        )
    }


def aggregate_quadrants(cases: list[dict]) -> dict:
    output = {}
    for name in cases[0]["quadrants"]:
        rows = [case["quadrants"][name] for case in cases]
        output[name] = {
            metric: distribution([row[metric] for row in rows])
            for metric in (
                "camera_translation_error_m",
                "viewing_direction_error_m",
                "visible_root_jump_residual_m",
                "shape_jump_l2",
                "body_scale_jump_abs",
                "local_pose_jump_residual_deg",
                "gt_beta_error_l2",
            )
        }
    return output


def aggregate(cases: list[dict]) -> dict:
    by_candidate = {name: aggregate_candidate(cases, name) for name in ALL_CANDIDATES}
    return {
        "candidates": by_candidate,
        "continuity": {
            name: aggregate_continuity(cases, name)
            for name in cases[0]["continuity"]
        },
        "quadrants": aggregate_quadrants(cases),
        "commit": {
            name: distribution([case["commit"][name] for case in cases])
            for name in cases[0]["commit"]
        },
        "memory": {
            "predicted_scale_history_std": distribution(
                [case["memory"]["predicted_scale_history_std"] for case in cases]
            ),
            "shape_history_std": distribution([case["memory"]["shape_history_std"] for case in cases]),
        },
    }


def difficulty_groups(cases: list[dict]) -> dict:
    definitions = {
        "post_shape_error": "post_shape_error_l2",
        "post_body_scale_error": "post_body_scale_error_abs",
        "body_fraction": "body_fraction",
        "view_angle": "view_angle_deg",
        "initial_translation_error": "initial_translation_error_m",
        "initial_viewing_error": "initial_viewing_error_m",
    }
    output = {}
    for label, field in definitions.items():
        values = np.asarray([case["groups"][field] for case in cases], dtype=np.float64)
        low, high = np.nanquantile(values, [1.0 / 3.0, 2.0 / 3.0])
        masks = {
            "low": values <= low,
            "mid": (values > low) & (values <= high),
            "high": values > high,
        }
        output[label] = {
            bucket: {
                name: aggregate_candidate([case for case, keep in zip(cases, mask) if keep], name)
                for name in ("current_independent", "canonical_beta_scale", "best_quality", "top3_consensus")
            }
            for bucket, mask in masks.items()
        }
    return output


def categorical_groups(cases: list[dict]) -> dict:
    output = {}
    for field in ("orientation", "torso_visibility"):
        output[field] = {}
        for value in sorted({str(case["groups"][field]) for case in cases}):
            selected = [case for case in cases if str(case["groups"][field]) == value]
            output[field][value] = {
                name: aggregate_candidate(selected, name)
                for name in ("current_independent", "canonical_beta_scale", "top3_consensus")
            }
    return output


def paired_alignment(cases: list[dict]) -> dict:
    baseline = np.asarray(
        [case["candidates"]["current_independent"]["camera_translation_error_m"] for case in cases],
        dtype=np.float64,
    )
    output = {}
    for name in REFERENCE_VARIANTS:
        if name == "current_independent":
            continue
        values = np.asarray(
            [case["candidates"][name]["camera_translation_error_m"] for case in cases],
            dtype=np.float64,
        )
        delta = values - baseline
        output[name] = {
            "mean_delta_m": float(np.mean(delta)),
            "median_delta_m": float(np.median(delta)),
            "improvement_rate": float(np.mean(delta < 0.0)),
            "improvement_over_5cm_rate": float(np.mean(delta < -0.05)),
            "harm_over_5cm_rate": float(np.mean(delta > 0.05)),
            "wilcoxon_p": float(wilcoxon(delta).pvalue),
        }
    return output


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# V14.2 Canonical Human Memory Probe",
        "",
        f"Cases: {report['case_count']}",
        "",
        "## Alignment",
        "",
        "| Candidate | T mean | T median | T P90 | T P95 | View | T-cat | Harmful vs current | Root residual |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ALL_CANDIDATES:
        row = report["overall"]["candidates"][name]
        lines.append(
            "| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.1f}% | {:.1f}% | {:.3f} |".format(
                name,
                row["translation_m"]["mean"],
                row["translation_m"]["median"],
                row["translation_m"]["p90"],
                row["translation_m"]["p95"],
                row["viewing_direction_m"]["mean"],
                100.0 * row["translation_catastrophic_rate"],
                100.0 * row["harmful_rate_vs_current"],
                row["visible_root_jump_residual_m"]["mean"],
            )
        )
    lines.extend(
        [
            "",
            "## Continuity",
            "",
            "| Method | Shape jump | Scale jump | Pose residual | GT beta | GT scale |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, row in report["overall"]["continuity"].items():
        lines.append(
            "| {} | {:.3f} | {:.5f} | {:.2f} | {:.3f} | {:.5f} |".format(
                name,
                row["shape_jump_l2"]["mean"],
                row["body_scale_jump_abs"]["mean"],
                row["local_pose_jump_residual_deg"]["mean"],
                row["gt_beta_error_l2"]["mean"],
                row["gt_body_scale_error_abs"]["mean"],
            )
        )
    lines.extend(["", "## By Source", ""])
    for source, metrics in report["by_source"].items():
        current = metrics["candidates"]["current_independent"]
        canonical = metrics["candidates"]["canonical_beta_scale"]
        lines.append(
            f"- **{source}**: current `{current['translation_m']['mean']:.3f} m`, "
            f"canonical `{canonical['translation_m']['mean']:.3f} m`; root residual "
            f"`{current['visible_root_jump_residual_m']['mean']:.3f} -> "
            f"{canonical['visible_root_jump_residual_m']['mean']:.3f} m`."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def choose_wrong_cases(cases: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        grouped[str(case["source"])].append(case)
    result = {}
    for source_cases in grouped.values():
        for index, case in enumerate(source_cases):
            own_group = str(case["record"].get("group", ""))
            wrong = next(
                (
                    candidate
                    for offset in range(1, len(source_cases) + 1)
                    if str((candidate := source_cases[(index + offset) % len(source_cases)])["record"].get("group", ""))
                    != own_group
                ),
                source_cases[(index + 1) % len(source_cases)],
            )
            result[str(case["case_name"])] = wrong
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("V14.2 SMPL-X projection probe requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args)
    wrong_cases = choose_wrong_cases(cases)
    device = torch.device(args.device)
    layer10 = build_layer(device, 10)
    layer11 = build_layer(device, 11)
    neutral_pose = np.zeros((53, 3), dtype=np.float32)
    neutral_body = body_from_params(layer10, neutral_pose, np.zeros(10, dtype=np.float32), device)[0]
    neutral_scale = physical_scale(neutral_body)

    rows = []
    started = time.perf_counter()
    for index, case in enumerate(cases):
        rows.append(
            run_case(
                case,
                wrong_cases[str(case["case_name"])],
                layer10,
                layer11,
                device,
                args,
                neutral_scale,
            )
        )
        if (index + 1) % 10 == 0 or index + 1 == len(cases):
            print(f">> V14.2 {index + 1}/{len(cases)}", flush=True)

    sources = sorted({str(row["source"]) for row in rows})
    report = {
        "experiment": "V14.2 Canonical Human Memory for Alignment and Continuity",
        "case_count": len(rows),
        "elapsed_seconds": time.perf_counter() - started,
        "protocol": {
            "human3r_frozen": True,
            "inputs": "V18 cached current pose, frozen 2D torso joints, and processed intrinsics",
            "rotation": "V16 torso-motion rotation with one global 20 degree bound",
            "motion": "last pre-cut projected world root",
            "alignment_pose_memory_used": False,
            "gt_use": "evaluation and explicit oracle variants only",
            "body_scale_definition": "mean metric length of eight fixed pelvis/hip/shoulder/leg segments",
            "body_scale_note": "Human3R has no independent predicted world-scale head; the scalar is derived from its SMPL-X body geometry",
            "memory_update_alpha": float(args.memory_update_alpha),
            "alignment_alpha": float(args.alignment_alpha),
            "continuity_shape_alpha": float(args.continuity_shape_alpha),
            "continuity_pose_alpha": float(args.continuity_pose_alpha),
            "max_humans": 1,
        },
        "overall": aggregate(rows),
        "by_source": {
            source: aggregate([row for row in rows if row["source"] == source]) for source in sources
        },
        "difficulty_groups": difficulty_groups(rows),
        "categorical_groups": categorical_groups(rows),
        "paired_alignment_vs_current": paired_alignment(rows),
        "decision": {
            "alignment_contribution": False,
            "continuity_contribution": True,
            "recommended_role": "shot-aware human continuity memory only",
            "canonical_memory_for_boundary_translation": False,
            "align_then_commit_required": True,
        },
        "cases": rows,
    }
    output = args.output_dir / "v14_2_canonical_human_memory_probe.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    write_markdown(args.output_dir / "v14_2_canonical_human_memory_probe.md", report)
    print(f">> wrote {output}", flush=True)


if __name__ == "__main__":
    main()
