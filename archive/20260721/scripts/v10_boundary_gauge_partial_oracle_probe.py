#!/usr/bin/env python3
"""Boundary-gauge partial-oracle diagnostic on cached AABB Human3R outputs.

This script does not train a network and does not rerun Human3R.  It decomposes
the boundary SE(3) problem into rotation, translation, gravity, human root and
human torso-heading components using the same 180 cached AABB cases as the V10
Oracle Candidate Selection probe.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for path in (str(REPO_ROOT), str(SRC_ROOT), str(SCRIPTS_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from dust3r.datasets.avatarrex import AvatarReX_AABB  # noqa: E402
from dust3r.smpl_model import SMPLModel  # noqa: E402
from dust3r.utils.device import todevice  # noqa: E402
from dust3r.utils.smpl_layer import SMPL_Layer  # noqa: E402
from v10_implicit_explicit_cross_shot_probe import (  # noqa: E402
    background_cloud,
    history_background_cloud,
    transform_points,
)
from v10_oracle_candidate_selection_probe import (  # noqa: E402
    ANGLE_BUCKETS,
    camera_errors,
    predicted_poses,
    transform_camera_poses,
)
from v10_oracle_state_vs_gauge_probe import rotation_error_deg  # noqa: E402
from v10_token_alignment_4source_probe import (  # noqa: E402
    aabb_tuple_from_record,
    raw_roots_for_record,
    source_split_and_scope,
)
from v9_learned_stream_alignment_overfit import gt_pose_from_view  # noqa: E402


DEFAULT_CANDIDATE_REPORT = (
    REPO_ROOT
    / "output"
    / "v10_candidate_selection"
    / "oracle_gt_4source"
    / "oracle_candidate_selection_metrics.json"
)
METHOD_ORDER = (
    "current_best_explicit",
    "current_candidate_oracle",
    "candidate_rotation_oracle_resolved_t",
    "candidate_translation_oracle",
    "factorized_candidate_oracle",
    "gt_rotation_predicted_translation",
    "predicted_rotation_gt_translation",
    "gt_gravity_proxy",
    "gt_human_root",
    "gt_human_torso_heading",
    "gt_human_gravity",
    "full_boundary_oracle",
)
METHOD_LABELS = {
    "current_best_explicit": "Current Best Explicit",
    "current_candidate_oracle": "Current Candidate Oracle",
    "candidate_rotation_oracle_resolved_t": "Candidate Rotation Oracle + Resolved T",
    "candidate_translation_oracle": "Candidate Translation Oracle",
    "factorized_candidate_oracle": "Factorized Candidate Oracle",
    "gt_rotation_predicted_translation": "GT Rotation + Predicted Translation",
    "predicted_rotation_gt_translation": "Predicted Rotation + GT Translation",
    "gt_gravity_proxy": "GT Gravity Proxy",
    "gt_human_root": "GT Human Root",
    "gt_human_torso_heading": "GT Human Torso Heading",
    "gt_human_gravity": "GT Human + GT Gravity",
    "full_boundary_oracle": "Full Boundary Oracle",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_report", type=Path, default=DEFAULT_CANDIDATE_REPORT)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=REPO_ROOT / "output" / "v10_candidate_selection" / "boundary_gauge_partial_oracle",
    )
    parser.add_argument("--data_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data"))
    parser.add_argument("--device", default="cuda:6" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resolution", type=int, nargs=2, default=(512, 288), metavar=("W", "H"))
    parser.add_argument("--resize_mode", default="human3r_demo")
    parser.add_argument("--boundary", type=int, default=2)
    parser.add_argument("--gt_batch_size", type=int, default=4)
    parser.add_argument("--cloud_points_per_frame", type=int, default=4000)
    parser.add_argument("--translation_iters", type=int, default=8)
    parser.add_argument("--translation_max_distance", type=float, default=0.60)
    parser.add_argument("--translation_min_distance", type=float, default=0.12)
    parser.add_argument("--cases_per_source", type=int, default=0)
    parser.add_argument("--overwrite_gt_cache", action="store_true")
    return parser.parse_args()


def normalize(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < eps:
        raise ValueError(f"Degenerate vector: {vector}")
    return (vector / norm).astype(np.float32)


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float32)
    transform[:3, 3] = np.asarray(translation, dtype=np.float32)
    return transform


def transform_directions(rotation: np.ndarray, directions: np.ndarray) -> np.ndarray:
    return np.einsum("ij,...j->...i", rotation, directions).astype(np.float32)


def torso_frame(joints: np.ndarray) -> np.ndarray:
    left_hip, right_hip = joints[1], joints[2]
    left_shoulder, right_shoulder = joints[16], joints[17]
    hip_mid = 0.5 * (left_hip + right_hip)
    shoulder_mid = 0.5 * (left_shoulder + right_shoulder)
    up = normalize(shoulder_mid - hip_mid)
    right = normalize(right_shoulder - left_shoulder)
    forward = normalize(np.cross(right, up))
    right = normalize(np.cross(up, forward))
    return np.stack([right, up, forward], axis=1).astype(np.float32)


def rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = normalize(source)
    target = normalize(target)
    cross = np.cross(source, target)
    sin_angle = float(np.linalg.norm(cross))
    cos_angle = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if sin_angle < 1e-7:
        if cos_angle > 0.0:
            return np.eye(3, dtype=np.float32)
        fallback = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(fallback, source))) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        axis = normalize(np.cross(source, fallback))
        return Rotation.from_rotvec(math.pi * axis).as_matrix().astype(np.float32)
    axis = cross / sin_angle
    angle = math.atan2(sin_angle, cos_angle)
    return Rotation.from_rotvec(angle * axis).as_matrix().astype(np.float32)


def signed_angle_about_axis(source: np.ndarray, target: np.ndarray, axis: np.ndarray) -> float:
    axis = normalize(axis)
    source = source - axis * float(np.dot(source, axis))
    target = target - axis * float(np.dot(target, axis))
    source = normalize(source)
    target = normalize(target)
    return math.atan2(float(np.dot(axis, np.cross(source, target))), float(np.clip(np.dot(source, target), -1.0, 1.0)))


def correct_gravity(rotation: np.ndarray, up_local: np.ndarray, up_target: np.ndarray) -> np.ndarray:
    mapped_up = rotation @ normalize(up_local)
    correction = rotation_between(mapped_up, up_target)
    return (correction @ rotation).astype(np.float32)


def correct_heading(
    rotation: np.ndarray,
    heading_local: np.ndarray,
    heading_target: np.ndarray,
    axis_target: np.ndarray,
) -> np.ndarray:
    mapped_heading = rotation @ normalize(heading_local)
    angle = signed_angle_about_axis(mapped_heading, heading_target, axis_target)
    correction = Rotation.from_rotvec(angle * normalize(axis_target)).as_matrix().astype(np.float32)
    return (correction @ rotation).astype(np.float32)


def load_candidate_report(path: Path, cases_per_source: int) -> list[dict]:
    report = json.loads(path.read_text(encoding="utf-8"))
    cases = list(report["cases"])
    if cases_per_source <= 0:
        return cases
    selected = []
    counts: dict[str, int] = defaultdict(int)
    for case in cases:
        source = str(case["record"]["source"])
        if counts[source] >= cases_per_source:
            continue
        selected.append(case)
        counts[source] += 1
    return selected


def candidate_lookup(case: dict) -> dict[str, dict]:
    return {candidate["name"]: candidate for candidate in case["candidates"]}


def gt_cache_path(args: argparse.Namespace, case: dict) -> Path:
    return args.output_dir / "gt_cache" / f"{case['case_name']}.npz"


def build_source_dataset(records: list[dict], args: argparse.Namespace) -> AvatarReX_AABB:
    split, pair_scope = source_split_and_scope(records[0])
    fixed_samples = [aabb_tuple_from_record(record) for record in records]
    return AvatarReX_AABB(
        split=split,
        ROOT=str(args.data_root),
        resolution=tuple(args.resolution),
        num_views=4,
        aug_crop=0,
        allow_repeat=True,
        seed=401,
        n_corres=0,
        fixed_samples=fixed_samples,
        load_da3_depth=False,
        raw_calibration_root=raw_roots_for_record(records[0]),
        resize_mode=str(args.resize_mode),
        max_humans=1,
        pair_scope=pair_scope,
    )


def extract_gt_batch(
    views: list[dict],
    smpl_model: SMPLModel,
    batch_records: list[dict],
    args: argparse.Namespace,
) -> None:
    gt_poses = []
    for view in views:
        pose = view.get("raw_camera_pose", view["camera_pose"])
        gt_poses.append(pose.detach().float().cpu().numpy().astype(np.float32))
    gt_poses_np = np.stack(gt_poses, axis=1)
    smpl_model.update_smpl_gt(views)
    for batch_idx, case in enumerate(batch_records):
        joints_world = []
        for frame_idx, view in enumerate(views):
            valid = bool(view["smpl_mask"][batch_idx, 0].detach().cpu())
            if not valid:
                raise ValueError(f"GT human is not visible for {case['case_name']} frame {frame_idx}")
            joints_cam = view["smpl_j3d"][batch_idx, 0].detach().float().cpu().numpy().astype(np.float32)
            pose = gt_poses_np[batch_idx, frame_idx]
            joints_world.append(transform_points(pose, joints_cam))
        path = gt_cache_path(args, case)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            gt_poses=gt_poses_np[batch_idx],
            gt_joints_world=np.stack(joints_world).astype(np.float32),
        )


def ensure_gt_cache(cases: list[dict], args: argparse.Namespace, device: torch.device) -> None:
    missing_by_source: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        if args.overwrite_gt_cache or not gt_cache_path(args, case).is_file():
            missing_by_source[str(case["record"]["source"])].append(case)
    if not missing_by_source:
        return
    smpl_model = SMPLModel(
        device,
        model_args={"patch_size": 16, "mhmr_img_res": 896, "bb_patch_size": 14},
    )
    for source, source_cases in missing_by_source.items():
        records = [case["record"] for case in source_cases]
        dataset = build_source_dataset(records, args)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=int(args.gt_batch_size),
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        offset = 0
        for views in loader:
            batch_size = int(views[0]["img"].shape[0])
            batch_cases = source_cases[offset : offset + batch_size]
            offset += batch_size
            views = todevice(views, device)
            extract_gt_batch(views, smpl_model, batch_cases, args)
            print(f">> GT cache {source}: {offset}/{len(source_cases)}", flush=True)
            del views
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if offset != len(source_cases):
            raise RuntimeError(f"GT cache count mismatch for {source}: {offset}/{len(source_cases)}")


def build_pred_smpl_layer(device: torch.device) -> SMPL_Layer:
    layer = SMPL_Layer(
        type="smplx",
        gender="neutral",
        num_betas=10,
        kid=False,
        person_center="head",
    ).to(device)
    layer.eval()
    return layer


def load_predicted_joints(local_dir: Path, layer: SMPL_Layer, device: torch.device) -> np.ndarray:
    poses, intrinsics, rotvec, shape, transl, expression = [], [], [], [], [], []
    for idx in range(4):
        with np.load(local_dir / "camera" / f"{idx:06d}.npz") as camera:
            poses.append(camera["pose"].astype(np.float32))
            intrinsics.append(camera["intrinsics"].astype(np.float32))
        with np.load(local_dir / "smpl" / f"{idx:06d}.npz", allow_pickle=True) as smpl:
            if len(smpl["shape"]) == 0:
                raise ValueError(f"No predicted human in {local_dir} frame {idx}")
            rotvec.append(smpl["rotvec"][0].astype(np.float32))
            shape.append(smpl["shape"][0].astype(np.float32))
            transl.append(smpl["transl"][0].astype(np.float32))
            expr = smpl["expression"]
            expression.append(np.zeros(10, dtype=np.float32) if expr is None or len(expr) == 0 else expr[0].astype(np.float32))
    with torch.no_grad():
        output = layer(
            torch.from_numpy(np.stack(rotvec)).to(device),
            torch.from_numpy(np.stack(shape)).to(device),
            torch.from_numpy(np.stack(transl)).to(device),
            None,
            None,
            K=torch.from_numpy(np.stack(intrinsics)).to(device),
            expression=torch.from_numpy(np.stack(expression)).to(device),
        )
    joints_cam = output["smpl_j3d"].detach().float().cpu().numpy().astype(np.float32)
    poses_np = np.stack(poses)
    return (
        np.einsum("nij,nkj->nki", poses_np[:, :3, :3], joints_cam)
        + poses_np[:, None, :3, 3]
    ).astype(np.float32)


def solve_translation_fixed_rotation(
    rotation: np.ndarray,
    human_target: np.ndarray,
    human_source: np.ndarray,
    source_cloud: np.ndarray,
    target_cloud: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict]:
    translation = (human_target - rotation @ human_source).astype(np.float32)
    initial = translation.copy()
    if len(source_cloud) < 32 or len(target_cloud) < 32:
        return translation, {"status": "human_only_too_few_background_points", "iterations": []}
    tree = cKDTree(target_cloud)
    history = []
    for iteration in range(int(args.translation_iters)):
        transformed = transform_directions(rotation, source_cloud) + translation[None]
        distances, indices = tree.query(transformed, k=1, workers=-1)
        alpha = iteration / max(int(args.translation_iters) - 1, 1)
        max_distance = (1.0 - alpha) * float(args.translation_max_distance) + alpha * float(args.translation_min_distance)
        valid = np.isfinite(distances) & (distances < max_distance)
        if int(valid.sum()) < 32:
            history.append({"iteration": iteration, "pairs": int(valid.sum()), "status": "too_few_pairs"})
            break
        ids = np.where(valid)[0]
        trim = float(np.quantile(distances[ids], 0.70))
        ids = ids[distances[ids] <= trim]
        if len(ids) < 32:
            break
        residual = target_cloud[indices[ids]] - transformed[ids]
        weights = 1.0 / np.maximum(distances[ids], 0.01)
        delta = np.average(residual, axis=0, weights=weights).astype(np.float32)
        translation += delta
        history.append(
            {
                "iteration": iteration,
                "pairs": int(len(ids)),
                "median_distance": float(np.median(distances[ids])),
                "delta_norm": float(np.linalg.norm(delta)),
            }
        )
    return translation, {
        "status": "ok",
        "initial_human_translation": initial.tolist(),
        "residual_from_human_initial": float(np.linalg.norm(translation - initial)),
        "iterations": history,
    }


def transform_diagnostics(transform: np.ndarray, oracle: np.ndarray) -> dict:
    rotation_delta = transform[:3, :3] @ oracle[:3, :3].T
    yaw, pitch, roll = Rotation.from_matrix(rotation_delta.astype(np.float64)).as_euler("ZYX", degrees=True)
    translation_delta = transform[:3, 3] - oracle[:3, 3]
    return {
        "translation_xyz_signed_m": translation_delta.astype(float).tolist(),
        "translation_xyz_abs_m": np.abs(translation_delta).astype(float).tolist(),
        "translation_norm_m": float(np.linalg.norm(translation_delta)),
        "yaw_pitch_roll_signed_deg": [float(yaw), float(pitch), float(roll)],
        "yaw_pitch_roll_abs_deg": [float(abs(yaw)), float(abs(pitch)), float(abs(roll))],
        "rotation_geodesic_deg": rotation_error_deg(transform, oracle),
        "euler_convention": "ZYX in the frame-0-aligned Human3R target gauge",
    }


def human_jump(
    transform: np.ndarray,
    predicted_joints: np.ndarray,
    boundary: int,
    target_human_root: np.ndarray,
) -> dict:
    pre_frame = torso_frame(predicted_joints[boundary - 1])
    post_frame = torso_frame(predicted_joints[boundary])
    pre_root = predicted_joints[boundary - 1, 0]
    post_root = transform[:3, :3] @ predicted_joints[boundary, 0] + transform[:3, 3]
    post_frame_aligned = transform[:3, :3] @ post_frame
    pre_pose = make_transform(pre_frame, pre_root)
    post_pose = make_transform(post_frame_aligned, post_root)
    return {
        "world_root_jump_m": float(np.linalg.norm(post_root - pre_root)),
        "aligned_root_to_gt_m": float(np.linalg.norm(post_root - target_human_root)),
        "torso_orientation_jump_deg": rotation_error_deg(pre_pose, post_pose),
    }


def evaluate_transform(
    name: str,
    transform: np.ndarray,
    pred_poses: np.ndarray,
    target_poses: np.ndarray,
    oracle_transform: np.ndarray,
    predicted_joints: np.ndarray,
    boundary: int,
    target_human_root: np.ndarray,
    diagnostics: dict | None = None,
) -> dict:
    aligned = transform_camera_poses(pred_poses, transform, boundary)
    return {
        "name": name,
        "label": METHOD_LABELS[name],
        "transform": transform.astype(np.float32).tolist(),
        "camera": camera_errors(aligned, target_poses, boundary),
        "transform_error": transform_diagnostics(transform, oracle_transform),
        "human_jump": human_jump(transform, predicted_joints, boundary, target_human_root),
        "diagnostics": diagnostics or {},
    }


def run_case(
    case: dict,
    args: argparse.Namespace,
    pred_layer: SMPL_Layer,
    device: torch.device,
    case_index: int,
) -> dict:
    local_dir = Path(case["paths"]["human3r_local_reset"])
    pred_poses = predicted_poses(local_dir)
    predicted_joints = load_predicted_joints(local_dir, pred_layer, device)
    with np.load(gt_cache_path(args, case)) as gt:
        gt_poses = gt["gt_poses"].astype(np.float32)
        gt_joints_world = gt["gt_joints_world"].astype(np.float32)
    boundary = int(args.boundary)
    gauge_target = pred_poses[0] @ np.linalg.inv(gt_poses[0])
    gauge_local = pred_poses[boundary] @ np.linalg.inv(gt_poses[boundary])
    target_poses = np.stack([(gauge_target @ pose).astype(np.float32) for pose in gt_poses])
    oracle_transform = (target_poses[boundary] @ np.linalg.inv(pred_poses[boundary])).astype(np.float32)

    target_joints = np.stack([transform_points(gauge_target, joints) for joints in gt_joints_world])
    local_gt_joints = np.stack([transform_points(gauge_local, joints) for joints in gt_joints_world])
    target_torso = torso_frame(target_joints[boundary])
    local_torso = torso_frame(local_gt_joints[boundary])
    up_target, heading_target = target_torso[:, 1], target_torso[:, 2]
    up_local, heading_local = local_torso[:, 1], local_torso[:, 2]

    point_limit = int(args.cloud_points_per_frame)
    target_cloud, target_cloud_debug = history_background_cloud(
        local_dir, list(range(boundary)), point_limit
    )
    source_cloud, source_cloud_debug = background_cloud(
        local_dir, boundary, point_limit, seed=20260717 + case_index
    )
    human_target = predicted_joints[boundary - 1, 0]
    human_source = predicted_joints[boundary, 0]

    candidates = candidate_lookup(case)
    current = case["best_single_fixed"]
    current_transform = np.asarray(current["transform"], dtype=np.float32)
    current_candidate_oracle = np.asarray(case["oracle_selected"]["transform"], dtype=np.float32)
    translation_candidate = min(
        case["candidates"], key=lambda row: row["metrics"]["mean_translation_m"]
    )
    translation_candidate_transform = np.asarray(translation_candidate["transform"], dtype=np.float32)
    rotation_candidate = min(
        case["candidates"],
        key=lambda row: rotation_error_deg(np.asarray(row["transform"], dtype=np.float32), oracle_transform),
    )
    rotation_candidate_R = np.asarray(rotation_candidate["transform"], dtype=np.float32)[:3, :3]
    resolved_t, resolved_debug = solve_translation_fixed_rotation(
        rotation_candidate_R, human_target, human_source, source_cloud, target_cloud, args
    )
    candidate_rotation_transform = make_transform(rotation_candidate_R, resolved_t)

    factorized = []
    for candidate in case["candidates"]:
        candidate_R = np.asarray(candidate["transform"], dtype=np.float32)[:3, :3]
        candidate_t, candidate_debug = solve_translation_fixed_rotation(
            candidate_R, human_target, human_source, source_cloud, target_cloud, args
        )
        candidate_transform = make_transform(candidate_R, candidate_t)
        metrics = camera_errors(
            transform_camera_poses(pred_poses, candidate_transform, boundary), target_poses, boundary
        )
        factorized.append(
            {
                "rotation_source": candidate["name"],
                "transform": candidate_transform,
                "camera": metrics,
                "translation_solver": candidate_debug,
            }
        )
    factorized.sort(key=lambda row: row["camera"]["joint_oracle_cost"])
    factorized_best = factorized[0]

    gt_R = oracle_transform[:3, :3]
    gt_t = oracle_transform[:3, 3]
    gt_rotation_t, gt_rotation_t_debug = solve_translation_fixed_rotation(
        gt_R, human_target, human_source, source_cloud, target_cloud, args
    )
    gt_rotation_transform = make_transform(gt_R, gt_rotation_t)
    pred_rotation_gt_t = make_transform(current_transform[:3, :3], gt_t)

    gravity_R = correct_gravity(current_transform[:3, :3], up_local, up_target)
    gravity_t, gravity_t_debug = solve_translation_fixed_rotation(
        gravity_R, human_target, human_source, source_cloud, target_cloud, args
    )
    gravity_transform = make_transform(gravity_R, gravity_t)

    gt_root_target = target_joints[boundary, 0]
    human_root_t = gt_root_target - current_transform[:3, :3] @ human_source
    human_root_transform = make_transform(current_transform[:3, :3], human_root_t)

    mapped_up_current = current_transform[:3, :3] @ up_local
    heading_R = correct_heading(
        current_transform[:3, :3], heading_local, heading_target, mapped_up_current
    )
    heading_t, heading_t_debug = solve_translation_fixed_rotation(
        heading_R, human_target, human_source, source_cloud, target_cloud, args
    )
    heading_transform = make_transform(heading_R, heading_t)

    human_gravity_R = correct_heading(gravity_R, heading_local, heading_target, up_target)
    human_gravity_t = gt_root_target - human_gravity_R @ human_source
    human_gravity_transform = make_transform(human_gravity_R, human_gravity_t)

    variants = {
        "current_best_explicit": evaluate_transform(
            "current_best_explicit", current_transform, pred_poses, target_poses, oracle_transform,
            predicted_joints, boundary, gt_root_target, {"candidate": current["name"]}
        ),
        "current_candidate_oracle": evaluate_transform(
            "current_candidate_oracle", current_candidate_oracle, pred_poses, target_poses,
            oracle_transform, predicted_joints, boundary, gt_root_target,
            {"candidate": case["oracle_selected"]["name"]}
        ),
        "candidate_rotation_oracle_resolved_t": evaluate_transform(
            "candidate_rotation_oracle_resolved_t", candidate_rotation_transform, pred_poses,
            target_poses, oracle_transform, predicted_joints, boundary, gt_root_target,
            {"rotation_candidate": rotation_candidate["name"], "translation_solver": resolved_debug}
        ),
        "candidate_translation_oracle": evaluate_transform(
            "candidate_translation_oracle", translation_candidate_transform, pred_poses, target_poses,
            oracle_transform, predicted_joints, boundary, gt_root_target,
            {"candidate": translation_candidate["name"]}
        ),
        "factorized_candidate_oracle": evaluate_transform(
            "factorized_candidate_oracle", factorized_best["transform"], pred_poses, target_poses,
            oracle_transform, predicted_joints, boundary, gt_root_target,
            {
                "rotation_candidate": factorized_best["rotation_source"],
                "translation_solver": factorized_best["translation_solver"],
            }
        ),
        "gt_rotation_predicted_translation": evaluate_transform(
            "gt_rotation_predicted_translation", gt_rotation_transform, pred_poses, target_poses,
            oracle_transform, predicted_joints, boundary, gt_root_target,
            {"translation_solver": gt_rotation_t_debug}
        ),
        "predicted_rotation_gt_translation": evaluate_transform(
            "predicted_rotation_gt_translation", pred_rotation_gt_t, pred_poses, target_poses,
            oracle_transform, predicted_joints, boundary, gt_root_target
        ),
        "gt_gravity_proxy": evaluate_transform(
            "gt_gravity_proxy", gravity_transform, pred_poses, target_poses, oracle_transform,
            predicted_joints, boundary, gt_root_target,
            {
                "gravity_source": "GT torso-up proxy because no unified ground-normal GT is stored",
                "translation_solver": gravity_t_debug,
            }
        ),
        "gt_human_root": evaluate_transform(
            "gt_human_root", human_root_transform, pred_poses, target_poses, oracle_transform,
            predicted_joints, boundary, gt_root_target,
            {"root_definition": "GT SMPL-X pelvis in frame-0-aligned target gauge"}
        ),
        "gt_human_torso_heading": evaluate_transform(
            "gt_human_torso_heading", heading_transform, pred_poses, target_poses, oracle_transform,
            predicted_joints, boundary, gt_root_target,
            {"translation_solver": heading_t_debug, "heading_axis": "predicted mapped torso-up"}
        ),
        "gt_human_gravity": evaluate_transform(
            "gt_human_gravity", human_gravity_transform, pred_poses, target_poses, oracle_transform,
            predicted_joints, boundary, gt_root_target,
            {
                "rotation": "GT torso-up plus GT torso heading",
                "translation": "GT pelvis target minus rotated predicted local pelvis",
            }
        ),
        "full_boundary_oracle": evaluate_transform(
            "full_boundary_oracle", oracle_transform, pred_poses, target_poses, oracle_transform,
            predicted_joints, boundary, gt_root_target
        ),
    }
    return {
        "case_name": case["case_name"],
        "record": case["record"],
        "variants": variants,
        "oracle_transform": oracle_transform.tolist(),
        "pointmap_debug": {"target": target_cloud_debug, "source": source_cloud_debug},
    }


def percentile(values: list[float], quantile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def aggregate_method(cases: list[dict], method: str) -> dict:
    rows = [case["variants"][method] for case in cases]
    camera = [row["camera"] for row in rows]
    transform_error = [row["transform_error"] for row in rows]
    human = [row["human_jump"] for row in rows]
    translation = [row["mean_translation_m"] for row in camera]
    rotation = [row["mean_rotation_deg"] for row in camera]
    root_jump = [row["world_root_jump_m"] for row in human]
    root_to_gt = [row["aligned_root_to_gt_m"] for row in human]
    axis_t = np.asarray([row["translation_xyz_abs_m"] for row in transform_error], dtype=np.float64)
    axis_r = np.asarray([row["yaw_pitch_roll_abs_deg"] for row in transform_error], dtype=np.float64)
    boundary_t = [row["boundary_translation_m"] for row in camera]
    boundary_r = [row["boundary_rotation_deg"] for row in camera]
    return {
        "count": len(rows),
        "translation_mean_m": float(np.mean(translation)),
        "translation_median_m": float(np.median(translation)),
        "translation_p90_m": percentile(translation, 90),
        "translation_p95_m": percentile(translation, 95),
        "rotation_mean_deg": float(np.mean(rotation)),
        "rotation_median_deg": float(np.median(rotation)),
        "rotation_p90_deg": percentile(rotation, 90),
        "rotation_p95_deg": percentile(rotation, 95),
        "boundary_translation_mean_m": float(np.mean(boundary_t)),
        "boundary_rotation_mean_deg": float(np.mean(boundary_r)),
        "success_strict_rate": float(np.mean([row["success_strict"] for row in camera])),
        "success_relaxed_rate": float(np.mean([row["success_relaxed"] for row in camera])),
        "catastrophic_rate": float(np.mean([row["catastrophic"] for row in camera])),
        "human_root_jump_mean_m": float(np.mean(root_jump)),
        "human_root_to_gt_mean_m": float(np.mean(root_to_gt)),
        "translation_axis_abs_mean_m": {
            "x": float(axis_t[:, 0].mean()),
            "y": float(axis_t[:, 1].mean()),
            "z": float(axis_t[:, 2].mean()),
        },
        "rotation_axis_abs_mean_deg": {
            "yaw": float(axis_r[:, 0].mean()),
            "pitch": float(axis_r[:, 1].mean()),
            "roll": float(axis_r[:, 2].mean()),
        },
        "mean_per_frame": [
            {
                "post_offset": offset,
                "translation_m": float(np.mean([row["per_frame"][offset]["translation_m"] for row in camera])),
                "rotation_deg": float(np.mean([row["per_frame"][offset]["rotation_deg"] for row in camera])),
            }
            for offset in range(len(camera[0]["per_frame"]))
        ],
    }


def aggregate_group(cases: list[dict]) -> dict:
    aggregate = {method: aggregate_method(cases, method) for method in METHOD_ORDER}
    full = aggregate["full_boundary_oracle"]
    for metrics in aggregate.values():
        metrics["gap_to_full_boundary_oracle"] = {
            "translation_mean_m": metrics["translation_mean_m"] - full["translation_mean_m"],
            "rotation_mean_deg": metrics["rotation_mean_deg"] - full["rotation_mean_deg"],
            "joint_cost": (
                metrics["translation_mean_m"]
                + math.radians(metrics["rotation_mean_deg"])
                - full["translation_mean_m"]
                - math.radians(full["rotation_mean_deg"])
            ),
        }
    return aggregate


def relative_reduction(base: float, improved: float) -> float:
    return float((base - improved) / max(abs(base), 1e-8))


def automatic_diagnosis(overall: dict) -> dict:
    current = overall["current_best_explicit"]
    candidate = overall["current_candidate_oracle"]
    factorized = overall["factorized_candidate_oracle"]
    gt_rotation = overall["gt_rotation_predicted_translation"]
    gt_translation = overall["predicted_rotation_gt_translation"]
    gravity = overall["gt_gravity_proxy"]
    human_root = overall["gt_human_root"]
    heading = overall["gt_human_torso_heading"]
    combined = overall["gt_human_gravity"]
    full = overall["full_boundary_oracle"]

    rotation_t_reduction = relative_reduction(current["translation_mean_m"], gt_rotation["translation_mean_m"])
    root_t_reduction = relative_reduction(current["translation_mean_m"], human_root["translation_mean_m"])
    full_rotation_gain = current["rotation_mean_deg"] - gt_rotation["rotation_mean_deg"]
    gravity_rotation_gain = current["rotation_mean_deg"] - gravity["rotation_mean_deg"]
    heading_rotation_gain = current["rotation_mean_deg"] - heading["rotation_mean_deg"]
    gravity_fraction = gravity_rotation_gain / max(full_rotation_gain, 1e-8)
    factorized_joint = factorized["translation_mean_m"] + math.radians(factorized["rotation_mean_deg"])
    candidate_joint = candidate["translation_mean_m"] + math.radians(candidate["rotation_mean_deg"])
    factorized_gain = relative_reduction(candidate_joint, factorized_joint)
    combined_joint = combined["translation_mean_m"] + math.radians(combined["rotation_mean_deg"])
    full_joint = full["translation_mean_m"] + math.radians(full["rotation_mean_deg"])

    combined_far_from_full = bool(
        combined["translation_mean_m"] > 0.50
        or combined["success_relaxed_rate"] < 0.50
    )
    if combined["success_relaxed_rate"] >= 0.80 or combined_joint <= max(0.10, 5.0 * full_joint):
        priority = "human_motion_plus_gravity"
        conclusion = "人体 root/运动与重力/heading 已接近完整 Boundary Oracle，应优先发展分解式显式 SE(3)。"
    elif combined_far_from_full:
        priority = "scene_relocalization"
        conclusion = (
            "即使提供 GT human root、torso heading 和 gravity，结果仍远离 Boundary Oracle；"
            "下一步必须优先开发场景重定位或世界坐标记忆，人体 motion/root 作为平移辅助，torso heading 作为旋转辅助。"
        )
    elif gt_rotation["translation_mean_m"] > 0.50 and root_t_reduction > 0.30:
        priority = "human_motion_bridge"
        conclusion = "GT Rotation 后平移仍大，而 GT Human Root 明显改善，优先开发 Human Motion Bridge。"
    elif gravity_fraction >= 0.80:
        priority = "gravity_and_ground"
        conclusion = "GT gravity proxy 已解释大部分旋转收益，优先开发地面和重力模块。"
    elif heading_rotation_gain > gravity_rotation_gain + 2.0:
        priority = "human_torso_heading"
        conclusion = "人体 torso heading 对旋转的帮助明显大于 gravity，优先解决 yaw。"
    elif rotation_t_reduction > 0.30:
        priority = "rotation_first"
        conclusion = "GT Rotation 同时显著降低平移误差，旋转是当前首要瓶颈。"
    else:
        priority = "scene_relocalization"
        conclusion = "人体和简单重力信息仍不足，优先开发真正的场景重定位或世界坐标记忆。"

    return {
        "priority": priority,
        "conclusion": conclusion,
        "rotation_oracle_translation_reduction": rotation_t_reduction,
        "human_root_translation_reduction": root_t_reduction,
        "gravity_fraction_of_gt_rotation_gain": gravity_fraction,
        "heading_rotation_gain_deg": heading_rotation_gain,
        "gravity_rotation_gain_deg": gravity_rotation_gain,
        "factorized_joint_improvement_over_candidate_oracle": factorized_gain,
        "factorized_recommended": bool(factorized_gain > 0.10),
        "combined_human_gravity_gap_to_full_joint": float(combined_joint - full_joint),
        "combined_human_gravity_far_from_full": combined_far_from_full,
        "gt_translation_rotation_mean_deg": gt_translation["rotation_mean_deg"],
    }


def write_summary_csv(path: Path, aggregate: dict) -> None:
    rows = []
    for method in METHOD_ORDER:
        metrics = aggregate[method]
        rows.append(
            {
                "method": method,
                "label": METHOD_LABELS[method],
                "translation_mean_m": metrics["translation_mean_m"],
                "translation_median_m": metrics["translation_median_m"],
                "translation_p90_m": metrics["translation_p90_m"],
                "translation_p95_m": metrics["translation_p95_m"],
                "rotation_mean_deg": metrics["rotation_mean_deg"],
                "rotation_median_deg": metrics["rotation_median_deg"],
                "rotation_p90_deg": metrics["rotation_p90_deg"],
                "rotation_p95_deg": metrics["rotation_p95_deg"],
                "gap_to_full_t_m": metrics["gap_to_full_boundary_oracle"]["translation_mean_m"],
                "gap_to_full_r_deg": metrics["gap_to_full_boundary_oracle"]["rotation_mean_deg"],
                "success_relaxed_rate": metrics["success_relaxed_rate"],
                "catastrophic_rate": metrics["catastrophic_rate"],
                "root_jump_m": metrics["human_root_jump_mean_m"],
                "root_to_gt_m": metrics["human_root_to_gt_mean_m"],
                "yaw_abs_deg": metrics["rotation_axis_abs_mean_deg"]["yaw"],
                "pitch_abs_deg": metrics["rotation_axis_abs_mean_deg"]["pitch"],
                "roll_abs_deg": metrics["rotation_axis_abs_mean_deg"]["roll"],
                "x_abs_m": metrics["translation_axis_abs_mean_m"]["x"],
                "y_abs_m": metrics["translation_axis_abs_mean_m"]["y"],
                "z_abs_m": metrics["translation_axis_abs_mean_m"]["z"],
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, report: dict) -> None:
    overall = report["overall"]
    diagnosis = report["automatic_diagnosis"]
    lines = [
        "# V10 Boundary Gauge Partial-Oracle Probe",
        "",
        "## 实验设置",
        "",
        f"- Case 数量：`{report['case_count']}`。",
        "- 不训练网络，不运行 Selector，不重新运行 Human3R。",
        "- cut 后保持 fresh state；所有方法只估计一个固定 shot-level SE(3)。",
        "- 数据没有统一 ground-normal GT，因此 GT Gravity 使用 GT SMPL-X torso-up 作为重力代理。",
        "- yaw/pitch/roll 使用 frame-0 对齐后的 Human3R target gauge 中的 ZYX 欧拉角。",
        "",
        "## 总结表",
        "",
        "| 方法 | T mean | R mean | Gap T | Gap R | T P90 | R P90 | Relaxed | Catastrophic | Root jump | Root-to-GT |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHOD_ORDER:
        metrics = overall[method]
        lines.append(
            f"| {METHOD_LABELS[method]} | {metrics['translation_mean_m']:.4f} | "
            f"{metrics['rotation_mean_deg']:.2f} | "
            f"{metrics['gap_to_full_boundary_oracle']['translation_mean_m']:.4f} | "
            f"{metrics['gap_to_full_boundary_oracle']['rotation_mean_deg']:.2f} | "
            f"{metrics['translation_p90_m']:.4f} | {metrics['rotation_p90_deg']:.2f} | "
            f"{100.0 * metrics['success_relaxed_rate']:.1f}% | "
            f"{100.0 * metrics['catastrophic_rate']:.1f}% | {metrics['human_root_jump_mean_m']:.4f} | "
            f"{metrics['human_root_to_gt_mean_m']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 自动判断",
            "",
            f"- 下一步优先级：`{diagnosis['priority']}`。",
            f"- 结论：{diagnosis['conclusion']}",
            f"- GT Rotation 对平移误差的相对降低：`{100.0 * diagnosis['rotation_oracle_translation_reduction']:.1f}%`。",
            f"- GT Human Root 对平移误差的相对降低：`{100.0 * diagnosis['human_root_translation_reduction']:.1f}%`。",
            f"- Gravity proxy 解释的 GT Rotation 收益比例：`{100.0 * diagnosis['gravity_fraction_of_gt_rotation_gain']:.1f}%`。",
            f"- Factorized 相对完整 Candidate Oracle 的 joint 改善：`{100.0 * diagnosis['factorized_joint_improvement_over_candidate_oracle']:.1f}%`。",
            f"- 是否建议拆分 R/T：`{diagnosis['factorized_recommended']}`。",
            "",
            "## 分数据源",
            "",
        ]
    )
    for source, group in report["by_source"].items():
        current = group["current_best_explicit"]
        factorized = group["factorized_candidate_oracle"]
        combined = group["gt_human_gravity"]
        lines.append(
            f"- **{source}**：Current `{current['translation_mean_m']:.3f} m / {current['rotation_mean_deg']:.1f} deg`； "
            f"Factorized `{factorized['translation_mean_m']:.3f} m / {factorized['rotation_mean_deg']:.1f} deg`； "
            f"Human+Gravity `{combined['translation_mean_m']:.3f} m / {combined['rotation_mean_deg']:.1f} deg`."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_candidate_report(args.candidate_report, int(args.cases_per_source))
    print(f">> partial-oracle cases={len(cases)} device={args.device}", flush=True)
    device = torch.device(args.device)
    ensure_gt_cache(cases, args, device)
    pred_layer = build_pred_smpl_layer(device)
    case_reports = []
    progress_path = args.output_dir / "case_progress.jsonl"
    if progress_path.exists():
        progress_path.unlink()
    start = time.perf_counter()
    for index, case in enumerate(cases):
        print(f">> [{index + 1}/{len(cases)}] {case['case_name']}", flush=True)
        report = run_case(case, args, pred_layer, device, index)
        case_reports.append(report)
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "case_name": report["case_name"],
                        "source": report["record"]["source"],
                        "current_t": report["variants"]["current_best_explicit"]["camera"]["mean_translation_m"],
                        "factorized_t": report["variants"]["factorized_candidate_oracle"]["camera"]["mean_translation_m"],
                        "human_gravity_t": report["variants"]["gt_human_gravity"]["camera"]["mean_translation_m"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    overall = aggregate_group(case_reports)
    by_source = {}
    for source in sorted({str(case["record"]["source"]) for case in case_reports}):
        by_source[source] = aggregate_group(
            [case for case in case_reports if str(case["record"]["source"]) == source]
        )
    by_angle = {}
    for bucket in ANGLE_BUCKETS:
        subset = [case for case in case_reports if str(case["record"].get("angle_bucket")) == bucket]
        if subset:
            by_angle[bucket] = aggregate_group(subset)
    diagnosis = automatic_diagnosis(overall)
    report = {
        "experiment": "Boundary Gauge Partial-Oracle Probe",
        "case_count": len(case_reports),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "protocol": {
            "human3r_rerun": False,
            "network_training": False,
            "selector": False,
            "gt_gravity_definition": "GT SMPL-X torso-up proxy; unified ground-normal GT is unavailable",
            "factorized_translation_solver": "predicted pelvis initialization plus fixed-rotation translation-only pointmap refinement",
        },
        "overall": overall,
        "by_source": by_source,
        "by_angle_bucket": by_angle,
        "automatic_diagnosis": diagnosis,
        "elapsed_seconds": time.perf_counter() - start,
        "cases": case_reports,
    }
    (args.output_dir / "boundary_gauge_partial_oracle_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_summary_csv(args.output_dir / "boundary_gauge_partial_oracle_summary.csv", overall)
    write_markdown(args.output_dir / "boundary_gauge_partial_oracle_metrics.md", report)
    print(json.dumps({"overall": overall, "automatic_diagnosis": diagnosis}, indent=2, ensure_ascii=False), flush=True)
    print(f">> report: {args.output_dir / 'boundary_gauge_partial_oracle_metrics.md'}", flush=True)


if __name__ == "__main__":
    main()
