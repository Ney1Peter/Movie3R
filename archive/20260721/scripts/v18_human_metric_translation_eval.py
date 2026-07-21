#!/usr/bin/env python3
"""V18 stages 3-5: causal human metric calibration and translation candidates."""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from v13_scene_coordinate_oracle import direct_transform_error  # noqa: E402
from v18_projection_depth_probe import (  # noqa: E402
    TORSO_IDS,
    body_from_params,
    build_layer,
    estimate_root_translation,
)


DEFAULT_ROOT = REPO_ROOT / "output" / "v18_human_metric_translation"
DEFAULT_STREAM = DEFAULT_ROOT / "stream_cache"
DEFAULT_KEYPOINT = DEFAULT_ROOT / "keypoint_cache"
DEFAULT_SCENE = DEFAULT_ROOT / "v16_bound20_scene"
DEFAULT_PROJECTION = DEFAULT_ROOT / "projection_depth" / "v18_projection_depth_probe.json"
DEFAULT_OUTPUT = DEFAULT_ROOT / "final_candidates"
DEPTH_METHODS = (
    "raw_human3r",
    "projection_no_calibration",
    "projection_shape_median",
    "projection_depth_ratio",
    "projection_depth_affine",
    "projection_shape_depth_ratio",
)
MOTION_METHODS = (
    "last_root",
    "constant_velocity_last2",
    "constant_acceleration",
    "robust_velocity_3f",
    "robust_velocity_5f",
    "torso_compatible",
)
PARTIAL_ROTATIONS = ("gt_rotation", "torso20_rotation", "fixed_rotation")
PARTIAL_WORLD_ROOTS = ("predicted_last", "predicted_constant_velocity", "gt_current")
PARTIAL_CAMERA_ROOTS = ("predicted", "gt_depth_only", "gt_transverse_only", "gt_full")
KEY_CANDIDATES = (
    "fixed_explicit",
    "v16_torso20_scene_resolve",
    "human_no_calibration",
    "human_shape_median",
    "human_depth_ratio",
    "human_depth_affine",
    "human_shape_depth_calibration",
    "human_scene_view_fusion",
    "human_scene_robust_fusion",
    "gt_camera_depth_upper",
    "gt_human_motion_upper",
    "gt_human_depth_and_motion_upper",
    "boundary_oracle",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream_dir", type=Path, default=DEFAULT_STREAM)
    parser.add_argument("--keypoint_dir", type=Path, default=DEFAULT_KEYPOINT)
    parser.add_argument("--scene_dir", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--projection_report", type=Path, default=DEFAULT_PROJECTION)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--keypoint_threshold", type=float, default=0.30)
    parser.add_argument("--ratio_min", type=float, default=0.50)
    parser.add_argument("--ratio_max", type=float, default=2.00)
    parser.add_argument("--robust_fusion_max_weight", type=float, default=0.75)
    parser.add_argument("--robust_fusion_max_depth_correction", type=float, default=1.50)
    return parser.parse_args()


def load_manifest(root: Path, pattern: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(str(root / pattern))):
        rows.extend(json.loads(Path(path).read_text(encoding="utf-8"))["cases"])
    return rows


def load_cases(args: argparse.Namespace) -> list[dict]:
    stream = load_manifest(args.stream_dir, "v18_stream_shard_*_of_*.json")
    keypoints = {
        row["case_name"]: row
        for row in load_manifest(args.keypoint_dir, "v18_keypoints_shard_*_of_*.json")
    }
    scene = {
        row["case_name"]: row
        for row in load_manifest(args.scene_dir, "v16_candidates_shard_*_of_*.json")
    }
    projection = json.loads(args.projection_report.read_text(encoding="utf-8"))
    groups = {row["case_name"]: row["groups"] for row in projection["cases"]}
    if len(stream) != 180 or len(keypoints) != 180 or len(scene) != 180:
        raise RuntimeError(
            f"Expected 180 stream/keypoint/scene cases, got {len(stream)}/{len(keypoints)}/{len(scene)}"
        )
    output = []
    for row in stream:
        case_name = row["case_name"]
        enriched = dict(row)
        enriched["keypoint_path"] = keypoints[case_name]["cache_path"]
        enriched["scene_case"] = scene[case_name]
        enriched["projection_groups"] = groups[case_name]
        output.append(enriched)
    return sorted(output, key=lambda row: str(row["case_name"]))


def camera_pose_from_boundary(boundary: np.ndarray, predicted_pose: np.ndarray) -> np.ndarray:
    return (np.asarray(boundary) @ np.asarray(predicted_pose)).astype(np.float32)


def boundary_from_camera_pose(camera_pose: np.ndarray, predicted_pose: np.ndarray) -> np.ndarray:
    return (np.asarray(camera_pose) @ np.linalg.inv(np.asarray(predicted_pose))).astype(np.float32)


def camera_pose_from_human(rotation: np.ndarray, world_root: np.ndarray, camera_root: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = np.asarray(rotation, dtype=np.float32)
    pose[:3, 3] = np.asarray(world_root) - pose[:3, :3] @ np.asarray(camera_root)
    return pose


def transform_point(pose: np.ndarray, point: np.ndarray) -> np.ndarray:
    return (pose[:3, :3] @ np.asarray(point) + pose[:3, 3]).astype(np.float32)


def direction_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-8 or norm_b < 1e-8:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (norm_a * norm_b), -1.0, 1.0))))


def evaluate_boundary(
    boundary: np.ndarray,
    predicted_pose: np.ndarray,
    target_pose: np.ndarray,
    target_boundary: np.ndarray,
) -> dict:
    row = direct_transform_error(boundary, predicted_pose, target_pose)
    camera_pose = camera_pose_from_boundary(boundary, predicted_pose)
    delta_world = camera_pose[:3, 3] - target_pose[:3, 3]
    delta_target = target_pose[:3, :3].T @ delta_world
    predicted_scale = float(np.linalg.norm(boundary[:3, 3]))
    target_scale = float(np.linalg.norm(target_boundary[:3, 3]))
    return {
        **row,
        "camera_translation_error_target_xyz_m": np.abs(delta_target).astype(float).tolist(),
        "viewing_direction_error_m": float(abs(delta_target[2])),
        "transverse_error_m": float(np.linalg.norm(delta_target[:2])),
        "vertical_error_m": float(abs(delta_target[1])),
        "translation_direction_error_deg": direction_error_deg(boundary[:3, 3], target_boundary[:3, 3]),
        "translation_scale_error_m": float(abs(predicted_scale - target_scale)),
        "translation_log_scale_error": float(
            abs(math.log(max(predicted_scale, 1e-8) / max(target_scale, 1e-8)))
        ),
        "transform": np.asarray(boundary).astype(float).tolist(),
    }


def estimate_roots(
    bodies: np.ndarray,
    observed: np.ndarray,
    confidence: np.ndarray,
    intrinsics: np.ndarray,
    initial: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, list[dict]]:
    roots = []
    diagnostics = []
    for body, points, scores, K, root0 in zip(bodies, observed, confidence, intrinsics, initial):
        root, row = estimate_root_translation(body, points, scores, K, root0, TORSO_IDS, threshold)
        roots.append(root)
        diagnostics.append(row)
    return np.stack(roots).astype(np.float32), diagnostics


def root_at_depth(root: np.ndarray, depth: float) -> np.ndarray:
    root = np.asarray(root, dtype=np.float64)
    depth = float(np.clip(depth, 0.20, 30.0))
    if abs(float(root[2])) < 1e-8:
        output = root.copy()
        output[2] = depth
        return output.astype(np.float32)
    return (root * (depth / float(root[2]))).astype(np.float32)


def ratio_calibration(
    roots: np.ndarray,
    raw_roots: np.ndarray,
    ratio_min: float,
    ratio_max: float,
) -> tuple[np.ndarray, dict]:
    valid = (
        np.isfinite(roots).all(axis=1)
        & np.isfinite(raw_roots).all(axis=1)
        & (roots[:, 2] > 0.20)
        & (raw_roots[:, 2] > 0.20)
    )
    ratios = raw_roots[valid, 2] / roots[valid, 2]
    ratio = float(np.median(ratios)) if len(ratios) else 1.0
    ratio = float(np.clip(ratio, ratio_min, ratio_max))
    calibrated = np.stack([root_at_depth(root, ratio * root[2]) for root in roots])
    return calibrated, {
        "status": "ok" if len(ratios) else "no_valid_history",
        "ratio": ratio,
        "ratio_mad": float(np.median(np.abs(ratios - np.median(ratios)))) if len(ratios) else float("inf"),
        "sample_count": int(len(ratios)),
    }


def affine_calibration(
    roots: np.ndarray,
    raw_roots: np.ndarray,
    ratio_min: float,
    ratio_max: float,
) -> tuple[np.ndarray, dict]:
    valid = (
        np.isfinite(roots).all(axis=1)
        & np.isfinite(raw_roots).all(axis=1)
        & (roots[:, 2] > 0.20)
        & (raw_roots[:, 2] > 0.20)
    )
    x = roots[valid, 2].astype(np.float64)
    y = raw_roots[valid, 2].astype(np.float64)
    if len(x) < 3 or float(np.ptp(x)) < 0.03:
        calibrated, ratio = ratio_calibration(roots, raw_roots, ratio_min, ratio_max)
        return calibrated, {"status": "ratio_fallback", "scale": ratio["ratio"], "offset_m": 0.0, **ratio}

    median_ratio = float(np.clip(np.median(y / x), ratio_min, ratio_max))

    def residual(parameters: np.ndarray) -> np.ndarray:
        scale, offset = parameters
        data = scale * x + offset - y
        regularizer = np.asarray([0.20 * (scale - median_ratio), 0.10 * offset], dtype=np.float64)
        return np.concatenate([data, regularizer])

    result = least_squares(
        residual,
        np.asarray([median_ratio, 0.0], dtype=np.float64),
        bounds=(np.asarray([ratio_min, -2.0]), np.asarray([ratio_max, 2.0])),
        loss="soft_l1",
        f_scale=0.05,
        max_nfev=100,
    )
    scale, offset = result.x
    calibrated = np.stack([root_at_depth(root, scale * root[2] + offset) for root in roots])
    return calibrated, {
        "status": "ok" if result.success else "optimizer_failed",
        "scale": float(scale),
        "offset_m": float(offset),
        "history_depth_residual_m": float(np.median(np.abs(scale * x + offset - y))),
        "sample_count": int(len(x)),
    }


def torso_angular_speed_deg(joints_world: np.ndarray) -> float:
    headings = []
    for joints in joints_world:
        hip = 0.5 * (joints[1] + joints[2])
        shoulder = 0.5 * (joints[16] + joints[17])
        up = shoulder - hip
        right = joints[17] - joints[16]
        up /= max(float(np.linalg.norm(up)), 1e-8)
        right /= max(float(np.linalg.norm(right)), 1e-8)
        forward = np.cross(right, up)
        forward /= max(float(np.linalg.norm(forward)), 1e-8)
        headings.append(forward)
    angles = []
    for first, second in zip(headings[:-1], headings[1:]):
        angles.append(np.degrees(np.arccos(np.clip(np.dot(first, second), -1.0, 1.0))))
    return float(np.median(angles)) if angles else 0.0


def motion_predictions(history: np.ndarray, torso_speed_deg: float) -> dict[str, np.ndarray]:
    history = np.asarray(history, dtype=np.float32)
    differences = np.diff(history, axis=0)
    last = history[-1]
    velocity_last = differences[-1] if len(differences) else np.zeros(3, dtype=np.float32)
    velocity_3 = np.median(differences[-2:], axis=0) if len(differences) else velocity_last
    velocity_5 = np.median(differences[-4:], axis=0) if len(differences) else velocity_last
    acceleration = differences[-1] - differences[-2] if len(differences) >= 2 else np.zeros(3, dtype=np.float32)
    damping = 1.0 / (1.0 + max(float(torso_speed_deg), 0.0) / 20.0)
    return {
        "last_root": last.copy(),
        "constant_velocity_last2": (last + velocity_last).astype(np.float32),
        "constant_acceleration": (last + velocity_last + acceleration).astype(np.float32),
        "robust_velocity_3f": (last + velocity_3).astype(np.float32),
        "robust_velocity_5f": (last + velocity_5).astype(np.float32),
        "torso_compatible": (last + damping * velocity_5).astype(np.float32),
    }


def scene_transform(case: dict) -> np.ndarray:
    row = case["scene_case"]["fixed_candidates"]["fixed_torso_motion_1f_resolve_t"]
    return np.asarray(row["transform"], dtype=np.float32)


def robust_fusion(
    scene_pose: np.ndarray,
    human_pose: np.ndarray,
    reprojection_px: float,
    valid_joints: int,
    calibration_mad: float,
    maximum_weight: float,
    maximum_depth_correction: float,
) -> tuple[np.ndarray, dict]:
    axis = scene_pose[:3, 2]
    correction = float(np.dot(human_pose[:3, 3] - scene_pose[:3, 3], axis))
    reprojection_weight = 1.0 / (1.0 + (max(float(reprojection_px), 0.0) / 8.0) ** 2)
    joint_weight = float(np.clip(valid_joints / max(float(len(TORSO_IDS)), 1.0), 0.0, 1.0))
    stability_weight = 1.0 if not np.isfinite(calibration_mad) else 1.0 / (1.0 + (calibration_mad / 0.05) ** 2)
    weight = float(min(maximum_weight, reprojection_weight * joint_weight * stability_weight))
    bounded = float(np.clip(correction, -maximum_depth_correction, maximum_depth_correction))
    fused = scene_pose.copy()
    fused[:3, 3] = scene_pose[:3, 3] + weight * bounded * axis
    return fused, {
        "weight": weight,
        "raw_depth_correction_m": correction,
        "bounded_depth_correction_m": bounded,
        "reprojection_weight": reprojection_weight,
        "joint_weight": joint_weight,
        "stability_weight": stability_weight,
    }


def run_case(case: dict, layer, device: torch.device, args: argparse.Namespace) -> dict:
    with np.load(case["cache_path"]) as stream, np.load(case["keypoint_path"]) as keypoint:
        old_pose = stream["old_pose"].astype(np.float32)
        new_pose = stream["new_pose"].astype(np.float32)
        target_pose = stream["target_pose"].astype(np.float32)
        target_boundary = stream["gt_boundary"].astype(np.float32)
        fixed = stream["fixed_transform"].astype(np.float32)
        old_joints = stream["old_joints_camera"].astype(np.float32)
        new_joints = stream["new_joints_camera"].astype(np.float32)
        old_gt_world = stream["old_gt_joints_target_world"].astype(np.float32)
        new_gt_world = stream["new_gt_joints_target_world"].astype(np.float32)
        new_gt_camera = stream["new_gt_joints_camera"].astype(np.float32)
        old_intrinsics = stream["old_intrinsics"].astype(np.float32)
        new_intrinsics = stream["new_intrinsics"].astype(np.float32)
        old_rotvec = stream["old_rotvec"].astype(np.float32)
        new_rotvec = stream["new_rotvec"].astype(np.float32)
        old_shape = stream["old_shape"].astype(np.float32)
        old_keypoints = keypoint["old_keypoints"].astype(np.float32)
        new_keypoints = keypoint["new_keypoints"].astype(np.float32)
        old_confidence = keypoint["old_confidence"].astype(np.float32)
        new_confidence = keypoint["new_confidence"].astype(np.float32)

    all_raw_joints = np.concatenate([old_joints, new_joints[None]], axis=0)
    all_raw_roots = all_raw_joints[:, 0]
    all_predicted_bodies = all_raw_joints - all_raw_roots[:, None]
    all_keypoints = np.concatenate([old_keypoints, new_keypoints[None]], axis=0)
    all_confidence = np.concatenate([old_confidence, new_confidence[None]], axis=0)
    all_intrinsics = np.concatenate([old_intrinsics, new_intrinsics[None]], axis=0)
    median_shape = np.median(old_shape, axis=0).astype(np.float32)
    all_rotvec = np.concatenate([old_rotvec, new_rotvec[None]], axis=0)
    median_bodies = np.stack(
        [body_from_params(layer, pose, median_shape, 1.0, device)[0] for pose in all_rotvec]
    ).astype(np.float32)

    projection_roots, projection_diagnostics = estimate_roots(
        all_predicted_bodies,
        all_keypoints,
        all_confidence,
        all_intrinsics,
        all_raw_roots,
        float(args.keypoint_threshold),
    )
    median_roots, median_diagnostics = estimate_roots(
        median_bodies,
        all_keypoints,
        all_confidence,
        all_intrinsics,
        all_raw_roots,
        float(args.keypoint_threshold),
    )
    _, ratio_diagnostics = ratio_calibration(
        projection_roots[:-1], all_raw_roots[:-1], float(args.ratio_min), float(args.ratio_max)
    )
    projection_ratio = np.stack(
        [root_at_depth(root, ratio_diagnostics["ratio"] * root[2]) for root in projection_roots]
    )
    _, affine_diagnostics = affine_calibration(
        projection_roots[:-1], all_raw_roots[:-1], float(args.ratio_min), float(args.ratio_max)
    )
    projection_affine = np.stack(
        [
            root_at_depth(
                root,
                affine_diagnostics["scale"] * root[2] + affine_diagnostics["offset_m"],
            )
            for root in projection_roots
        ]
    )
    _, median_ratio_diagnostics = ratio_calibration(
        median_roots[:-1], all_raw_roots[:-1], float(args.ratio_min), float(args.ratio_max)
    )
    median_ratio = np.stack(
        [root_at_depth(root, median_ratio_diagnostics["ratio"] * root[2]) for root in median_roots]
    )
    root_sequences = {
        "raw_human3r": all_raw_roots,
        "projection_no_calibration": projection_roots,
        "projection_shape_median": median_roots,
        "projection_depth_ratio": projection_ratio,
        "projection_depth_affine": projection_affine,
        "projection_shape_depth_ratio": median_ratio,
    }
    depth_diagnostics = {
        "raw_human3r": {"status": "native"},
        "projection_no_calibration": projection_diagnostics[-1],
        "projection_shape_median": median_diagnostics[-1],
        "projection_depth_ratio": {**projection_diagnostics[-1], **ratio_diagnostics},
        "projection_depth_affine": {**projection_diagnostics[-1], **affine_diagnostics},
        "projection_shape_depth_ratio": {**median_diagnostics[-1], **median_ratio_diagnostics},
    }

    torso_speed = torso_angular_speed_deg(old_joints)
    target_world_root = new_gt_world[0]
    gt_camera_root = new_gt_camera[0]
    motion = {}
    human_boundaries = {}
    root_rows = {}
    scene = scene_transform(case)
    scene_pose = camera_pose_from_boundary(scene, new_pose)
    fixed_pose = camera_pose_from_boundary(fixed, new_pose)
    camera_rotation = scene_pose[:3, :3]
    for depth_name, roots in root_sequences.items():
        history_world = np.stack([transform_point(pose, root) for pose, root in zip(old_pose, roots[:-1])])
        predictions = motion_predictions(history_world, torso_speed)
        motion[depth_name] = {
            name: {
                "predicted_world_root": root.astype(float).tolist(),
                "world_root_error_m": float(np.linalg.norm(root - target_world_root)),
            }
            for name, root in predictions.items()
        }
        human_boundaries[depth_name] = {}
        for motion_name, world_root in predictions.items():
            camera_pose = camera_pose_from_human(camera_rotation, world_root, roots[-1])
            human_boundaries[depth_name][motion_name] = boundary_from_camera_pose(camera_pose, new_pose)
        delta = roots[-1] - gt_camera_root
        root_rows[depth_name] = {
            "estimated_camera_root": roots[-1].astype(float).tolist(),
            "root_position_error_m": float(np.linalg.norm(delta)),
            "root_depth_error_m": float(abs(delta[2])),
            "root_transverse_error_m": float(np.linalg.norm(delta[:2])),
            "diagnostics": depth_diagnostics[depth_name],
        }

    default_motion = "last_root"
    selected = {
        "human_no_calibration": ("projection_no_calibration", default_motion),
        "human_shape_median": ("projection_shape_median", default_motion),
        "human_depth_ratio": ("projection_depth_ratio", default_motion),
        "human_depth_affine": ("projection_depth_affine", default_motion),
        "human_shape_depth_calibration": ("projection_shape_depth_ratio", default_motion),
    }
    candidate_transforms = {
        "fixed_explicit": fixed,
        "v16_torso20_scene_resolve": scene,
    }
    for name, (depth_name, motion_name) in selected.items():
        candidate_transforms[name] = human_boundaries[depth_name][motion_name]

    fusion_human_pose = camera_pose_from_boundary(candidate_transforms["human_shape_median"], new_pose)
    axis = scene_pose[:3, 2]
    view_only_pose = scene_pose.copy()
    view_only_pose[:3, 3] += float(np.dot(fusion_human_pose[:3, 3] - scene_pose[:3, 3], axis)) * axis
    candidate_transforms["human_scene_view_fusion"] = boundary_from_camera_pose(view_only_pose, new_pose)
    median_diag = depth_diagnostics["projection_shape_median"]
    robust_pose, fusion_diagnostics = robust_fusion(
        scene_pose,
        fusion_human_pose,
        float(median_diag.get("reprojection_error_px", float("inf"))),
        int(median_diag.get("valid_joints", 0)),
        float(median_ratio_diagnostics.get("ratio_mad", float("inf"))),
        float(args.robust_fusion_max_weight),
        float(args.robust_fusion_max_depth_correction),
    )
    candidate_transforms["human_scene_robust_fusion"] = boundary_from_camera_pose(robust_pose, new_pose)

    base_roots = root_sequences["projection_shape_median"]
    base_history = np.stack([transform_point(pose, root) for pose, root in zip(old_pose, base_roots[:-1])])
    predicted_world = motion_predictions(base_history, torso_speed)[default_motion]
    gt_depth_root = root_at_depth(base_roots[-1], gt_camera_root[2])
    candidate_transforms["gt_camera_depth_upper"] = boundary_from_camera_pose(
        camera_pose_from_human(camera_rotation, predicted_world, gt_depth_root), new_pose
    )
    candidate_transforms["gt_human_motion_upper"] = boundary_from_camera_pose(
        camera_pose_from_human(camera_rotation, target_world_root, base_roots[-1]), new_pose
    )
    candidate_transforms["gt_human_depth_and_motion_upper"] = boundary_from_camera_pose(
        camera_pose_from_human(camera_rotation, target_world_root, gt_camera_root), new_pose
    )
    candidate_transforms["boundary_oracle"] = target_boundary

    raw_history_world = np.stack(
        [transform_point(pose, root) for pose, root in zip(old_pose, all_raw_roots[:-1])]
    )
    raw_motion = motion_predictions(raw_history_world, torso_speed)
    partial_rotations = {
        "gt_rotation": target_pose[:3, :3],
        "torso20_rotation": camera_rotation,
        "fixed_rotation": fixed_pose[:3, :3],
    }
    partial_world_roots = {
        "predicted_last": raw_motion["last_root"],
        "predicted_constant_velocity": raw_motion["constant_velocity_last2"],
        "gt_current": target_world_root,
    }
    partial_camera_roots = {
        "predicted": all_raw_roots[-1],
        "gt_depth_only": np.asarray(
            [all_raw_roots[-1, 0], all_raw_roots[-1, 1], gt_camera_root[2]], dtype=np.float32
        ),
        "gt_transverse_only": np.asarray(
            [gt_camera_root[0], gt_camera_root[1], all_raw_roots[-1, 2]], dtype=np.float32
        ),
        "gt_full": gt_camera_root,
    }
    partial_oracles = {}
    for rotation_name, rotation in partial_rotations.items():
        partial_oracles[rotation_name] = {}
        for world_name, world_root in partial_world_roots.items():
            partial_oracles[rotation_name][world_name] = {}
            for camera_name, camera_root in partial_camera_roots.items():
                partial_pose = camera_pose_from_human(rotation, world_root, camera_root)
                partial_boundary = boundary_from_camera_pose(partial_pose, new_pose)
                partial_oracles[rotation_name][world_name][camera_name] = evaluate_boundary(
                    partial_boundary, new_pose, target_pose, target_boundary
                )

    recovered_gt_pose = camera_pose_from_human(
        target_pose[:3, :3], target_world_root, gt_camera_root
    )
    recovered_gt_boundary = boundary_from_camera_pose(recovered_gt_pose, new_pose)
    coordinate_audit = {
        "equation_residual_m": float(
            np.linalg.norm(
                target_world_root
                - (target_pose[:3, :3] @ gt_camera_root + target_pose[:3, 3])
            )
        ),
        "recovered_camera_translation_error_m": float(
            np.linalg.norm(recovered_gt_pose[:3, 3] - target_pose[:3, 3])
        ),
        "recovered_boundary_translation_error_m": float(
            np.linalg.norm(recovered_gt_boundary[:3, 3] - target_boundary[:3, 3])
        ),
    }

    candidates = {
        name: evaluate_boundary(transform, new_pose, target_pose, target_boundary)
        for name, transform in candidate_transforms.items()
    }
    all_human_candidates = {
        depth_name: {
            motion_name: evaluate_boundary(transform, new_pose, target_pose, target_boundary)
            for motion_name, transform in variants.items()
        }
        for depth_name, variants in human_boundaries.items()
    }
    fixed_row = candidates["fixed_explicit"]
    initial_scale_error = fixed_row["translation_scale_error_m"]
    initial_direction_error = fixed_row["translation_direction_error_deg"]
    gt_speed = float(np.linalg.norm(target_world_root - old_gt_world[-1, 0]))
    projection_groups = dict(case["projection_groups"])
    projection_groups.update(
        {
            "motion_speed_m_per_frame": gt_speed,
            "initial_scale_error_m": initial_scale_error,
            "initial_direction_error_deg": initial_direction_error,
            "person_count": 1,
        }
    )
    return {
        "case_name": case["case_name"],
        "source": case["source"],
        "groups": projection_groups,
        "torso_angular_speed_deg_per_frame": torso_speed,
        "human_camera_root": root_rows,
        "human_motion": motion,
        "calibration": {
            "shape_history_std": float(np.mean(np.std(old_shape, axis=0))),
            "predicted_shape_median": median_shape.astype(float).tolist(),
            "depth_ratio": ratio_diagnostics,
            "depth_affine": affine_diagnostics,
            "shape_depth_ratio": median_ratio_diagnostics,
        },
        "fusion_diagnostics": fusion_diagnostics,
        "coordinate_audit": coordinate_audit,
        "partial_oracles": partial_oracles,
        "candidates": candidates,
        "all_human_candidates": all_human_candidates,
    }


def distribution(values: list[float]) -> dict:
    array = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if not len(array):
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan"), "p95": float("nan")}
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
    }


def aggregate_candidate(cases: list[dict], candidate: str) -> dict:
    rows = [case["candidates"][candidate] for case in cases]
    fixed = [case["candidates"]["fixed_explicit"] for case in cases]
    return {
        "count": len(rows),
        "translation_m": distribution([row["camera_translation_error_m"] for row in rows]),
        "rotation_deg": distribution([row["camera_rotation_error_deg"] for row in rows]),
        "viewing_direction_m": distribution([row["viewing_direction_error_m"] for row in rows]),
        "transverse_m": distribution([row["transverse_error_m"] for row in rows]),
        "vertical_m": distribution([row["vertical_error_m"] for row in rows]),
        "direction_deg": distribution([row["translation_direction_error_deg"] for row in rows]),
        "scale_m": distribution([row["translation_scale_error_m"] for row in rows]),
        "log_scale": distribution([row["translation_log_scale_error"] for row in rows]),
        "catastrophic_rate": float(
            np.mean(
                [
                    row["camera_translation_error_m"] > 1.0
                    or row["camera_rotation_error_deg"] > 30.0
                    for row in rows
                ]
            )
        ),
        "translation_catastrophic_rate": float(
            np.mean([row["camera_translation_error_m"] > 1.0 for row in rows])
        ),
        "rotation_catastrophic_rate": float(
            np.mean([row["camera_rotation_error_deg"] > 30.0 for row in rows])
        ),
        "success_rate": float(
            np.mean(
                [
                    row["camera_translation_error_m"] < 0.25
                    and row["camera_rotation_error_deg"] < 5.0
                    for row in rows
                ]
            )
        ),
        "harmful_translation_rate_vs_fixed": float(
            np.mean(
                [
                    row["camera_translation_error_m"] > base["camera_translation_error_m"] + 0.05
                    for row, base in zip(rows, fixed)
                ]
            )
        ),
    }


def aggregate_candidates(cases: list[dict]) -> dict:
    return {candidate: aggregate_candidate(cases, candidate) for candidate in KEY_CANDIDATES}


def aggregate_roots(cases: list[dict]) -> dict:
    return {
        method: {
            "position_m": distribution([case["human_camera_root"][method]["root_position_error_m"] for case in cases]),
            "depth_m": distribution([case["human_camera_root"][method]["root_depth_error_m"] for case in cases]),
            "transverse_m": distribution(
                [case["human_camera_root"][method]["root_transverse_error_m"] for case in cases]
            ),
        }
        for method in DEPTH_METHODS
    }


def aggregate_motion(cases: list[dict]) -> dict:
    return {
        depth: {
            motion: distribution(
                [case["human_motion"][depth][motion]["world_root_error_m"] for case in cases]
            )
            for motion in MOTION_METHODS
        }
        for depth in DEPTH_METHODS
    }


def aggregate_partial_variant(cases: list[dict], rotation: str, world: str, camera: str) -> dict:
    rows = [case["partial_oracles"][rotation][world][camera] for case in cases]
    return {
        "translation_m": distribution([row["camera_translation_error_m"] for row in rows]),
        "rotation_deg": distribution([row["camera_rotation_error_deg"] for row in rows]),
        "viewing_direction_m": distribution([row["viewing_direction_error_m"] for row in rows]),
        "transverse_m": distribution([row["transverse_error_m"] for row in rows]),
    }


def aggregate_partial_oracles(cases: list[dict]) -> dict:
    return {
        rotation: {
            world: {
                camera: aggregate_partial_variant(cases, rotation, world, camera)
                for camera in PARTIAL_CAMERA_ROOTS
            }
            for world in PARTIAL_WORLD_ROOTS
        }
        for rotation in PARTIAL_ROTATIONS
    }


def group_label(case: dict, group: str) -> str:
    value = case["groups"].get(group)
    if group in {"orientation", "visibility"}:
        return str(value)
    if group == "body_size":
        fraction = float(case["groups"]["body_fraction"])
        return "small" if fraction < 0.45 else "medium" if fraction < 0.75 else "large"
    if group == "motion_speed":
        speed = float(case["groups"]["motion_speed_m_per_frame"])
        return "low" if speed < 0.03 else "medium" if speed < 0.10 else "high"
    if group == "initial_scale_error":
        error = float(case["groups"]["initial_scale_error_m"])
        return "low" if error < 0.5 else "medium" if error < 1.0 else "high"
    if group == "initial_direction_error":
        error = float(case["groups"]["initial_direction_error_deg"])
        return "low" if error < 15.0 else "medium" if error < 45.0 else "high"
    raise KeyError(group)


def grouped_results(cases: list[dict]) -> dict:
    output = {}
    for group in ("orientation", "visibility", "body_size", "motion_speed", "initial_scale_error", "initial_direction_error"):
        buckets: dict[str, list[dict]] = defaultdict(list)
        for case in cases:
            buckets[group_label(case, group)].append(case)
        output[group] = {
            label: {
                "case_count": len(rows),
                "fixed_explicit": aggregate_candidate(rows, "fixed_explicit"),
                "human_shape_median": aggregate_candidate(rows, "human_shape_median"),
                "human_scene_robust_fusion": aggregate_candidate(rows, "human_scene_robust_fusion"),
            }
            for label, rows in sorted(buckets.items())
        }
    return output


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# V18 Human-Calibrated Metric Translation",
        "",
        "## Main Candidates",
        "",
        "| Candidate | T mean | T median | T P90 | T P95 | View | R mean | T-cat | R-cat | Success | Harmful T |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in KEY_CANDIDATES:
        row = report["overall"][name]
        lines.append(
            f"| {name} | {row['translation_m']['mean']:.3f} | {row['translation_m']['median']:.3f} | "
            f"{row['translation_m']['p90']:.3f} | {row['translation_m']['p95']:.3f} | "
            f"{row['viewing_direction_m']['mean']:.3f} | {row['rotation_deg']['mean']:.2f} | "
            f"{100.0 * row['translation_catastrophic_rate']:.1f}% | "
            f"{100.0 * row['rotation_catastrophic_rate']:.1f}% | {100.0 * row['success_rate']:.1f}% | "
            f"{100.0 * row['harmful_translation_rate_vs_fixed']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Camera-Frame Human Root",
            "",
            "| Method | Position | Depth | Transverse |",
            "|---|---:|---:|---:|",
        ]
    )
    for method in DEPTH_METHODS:
        row = report["human_camera_root"][method]
        lines.append(
            f"| {method} | {row['position_m']['mean']:.3f} | {row['depth_m']['mean']:.3f} | "
            f"{row['transverse_m']['mean']:.3f} |"
        )
    lines.extend(["", "## Motion Prediction", "", "Depth method: projection_shape_median", ""])
    for method in MOTION_METHODS:
        row = report["human_motion"]["projection_shape_median"][method]
        lines.append(f"- `{method}`: mean `{row['mean']:.3f} m`, P90 `{row['p90']:.3f} m`.")
    lines.extend(["", "## By Source", ""])
    for source, metrics in report["by_source"].items():
        fixed = metrics["fixed_explicit"]["translation_m"]["mean"]
        human = metrics["human_shape_median"]["translation_m"]["mean"]
        fusion = metrics["human_scene_robust_fusion"]["translation_m"]["mean"]
        lines.append(f"- **{source}**: Fixed `{fixed:.3f} m`; Human `{human:.3f} m`; Fusion `{fusion:.3f} m`.")
    lines.extend(["", "## Consistent-Gauge Partial Oracles", ""])
    for rotation, world, camera in (
        ("gt_rotation", "predicted_constant_velocity", "predicted"),
        ("gt_rotation", "gt_current", "predicted"),
        ("gt_rotation", "predicted_constant_velocity", "gt_full"),
        ("gt_rotation", "gt_current", "gt_depth_only"),
        ("gt_rotation", "gt_current", "gt_transverse_only"),
        ("gt_rotation", "gt_current", "gt_full"),
        ("torso20_rotation", "gt_current", "gt_full"),
    ):
        row = report["partial_oracles"][rotation][world][camera]
        lines.append(
            f"- `{rotation} / {world} / {camera}`: T `{row['translation_m']['mean']:.3f} m`, "
            f"view `{row['viewing_direction_m']['mean']:.3f} m`, R `{row['rotation_deg']['mean']:.2f} deg`."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("V18 final translation evaluation requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args)
    device = torch.device(args.device)
    layer = build_layer(device, 10)
    rows = []
    for index, case in enumerate(cases):
        rows.append(run_case(case, layer, device, args))
        if (index + 1) % 20 == 0:
            print(f"V18 final candidates {index + 1}/{len(cases)}", flush=True)
    overall = aggregate_candidates(rows)
    by_source = {
        source: aggregate_candidates([row for row in rows if row["source"] == source])
        for source in sorted({row["source"] for row in rows})
    }
    report = {
        "experiment": "V18 Human-Calibrated Metric Translation Probe",
        "case_count": len(rows),
        "protocol": {
            "human3r_frozen": True,
            "post_cut_frames": 1,
            "rotation": "V16 torso-motion, uniform 20-degree residual bound",
            "default_motion": "last root, selected from the stage-1 causal diagnostic before final candidates",
            "deployable_2d": "frozen torchvision Keypoint R-CNN",
            "self_calibration_history_frames": 5,
            "learned_alignment_used": False,
            "raw_tokens_used": False,
            "gt_depth_or_scene_mesh_used": False,
            "multiple_people_supported": False,
        },
        "overall": overall,
        "by_source": by_source,
        "human_camera_root": aggregate_roots(rows),
        "human_motion": aggregate_motion(rows),
        "coordinate_audit": {
            "max_equation_residual_m": max(row["coordinate_audit"]["equation_residual_m"] for row in rows),
            "max_recovered_camera_translation_error_m": max(
                row["coordinate_audit"]["recovered_camera_translation_error_m"] for row in rows
            ),
            "max_recovered_boundary_translation_error_m": max(
                row["coordinate_audit"]["recovered_boundary_translation_error_m"] for row in rows
            ),
        },
        "partial_oracles": aggregate_partial_oracles(rows),
        "grouped": grouped_results(rows),
        "cases": rows,
    }
    output = args.output_dir / "v18_human_metric_translation_eval.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    write_markdown(args.output_dir / "v18_human_metric_translation_summary.md", report)
    print(f">> wrote {output}", flush=True)


if __name__ == "__main__":
    main()
