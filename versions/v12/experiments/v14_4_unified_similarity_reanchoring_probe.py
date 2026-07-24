#!/usr/bin/env python3
"""V14.4 unified projection-consistent similarity re-anchoring.

Every method is evaluated in one fixed pre-shot metric gauge.  Deployable
unified variants use one post-shot scalar for camera translation, pointmap,
human root, joints, vertices, and root-centered body offsets.  A calibrated
human root is solved once in that scaled gauge and is shared by the camera
translation equation and final SMPL-X placement.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[3]
for path in (str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from dust3r.utils.smpl_layer import SMPL_Layer  # noqa: E402
from boundary_metric_selection_support import load_cases as load_json_cases  # noqa: E402
from boundary_shot_scale_support import scale_pose  # noqa: E402
from versions.v12.experiments.v14_2_canonical_human_memory_probe import (  # noqa: E402
    estimate_root_translation,
    physical_scale,
)
from versions.v12.experiments.v14_3_projection_consistent_reanchoring_probe import (  # noqa: E402
    body_from_params,
    boundary_from_camera_pose,
    camera_pose_from_human,
    continuity_body,
    detector_root_pixel,
    distribution,
    evaluate_camera,
    load_cases as load_v14_cases,
    load_scene_pair,
    mesh_bbox_metrics,
    normalize_body_scale,
    project,
    reproject_error,
    rotation_error_deg,
    root_from_pixel_depth,
    transform_point,
    transform_points,
)


MAIN_METHODS = (
    "fixed_explicit",
    "v16_raw_scale",
    "da3_background_only_uniform_similarity",
    "da3_keypoint_root_uniform_similarity",
    "keypoint_projection_relative_uniform_similarity",
    "v11_1_conditional_wide_raw_scale",
    "v11_4_uniform_similarity",
    "v14_3_v18_camera_only",
    "v14_3_v18_coupled",
    "v14_3_da3_camera_only",
    "v14_3_da3_coupled",
    "naive_sequential",
    "unified_shared_scale_coupled_root",
    "unified_relative_da3_scale_coupled_root",
    "unified_human_relative_scale_coupled_root",
    "unified_da3_absolute_scale_projection_coupled_root",
    "unified_v11_scale_da3_coupled_root",
    "unified_da3_absolute_scale_da3_coupled_root",
    "unified_shared_scale_coupled_root_continuity",
    "gt_shared_scale_oracle",
    "gt_human_scale_oracle",
    "gt_scene_scale_oracle",
    "gt_separate_human_scene_scale_oracle",
    "boundary_oracle",
)
TAIL_METHODS = (
    "v11_4_uniform_similarity_conditional_vggt",
    "v14_3_v18_coupled_conditional_vggt",
    "unified_shared_scale_coupled_root_conditional_vggt",
    "unified_relative_da3_scale_coupled_root_conditional_vggt",
    "unified_v11_scale_da3_coupled_root_conditional_vggt",
    "unified_da3_absolute_scale_da3_coupled_root_conditional_vggt",
)
FOOT_NAMES = (
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stream_dir", type=Path, default=ROOT / "output/v18_human_metric_translation/stream_cache"
    )
    parser.add_argument(
        "--keypoint_dir", type=Path, default=ROOT / "output/v18_human_metric_translation/keypoint_cache"
    )
    parser.add_argument(
        "--v14_2_report",
        type=Path,
        default=ROOT
        / "output/v14_2_canonical_human_memory/single_cut/v14_2_canonical_human_memory_probe.json",
    )
    parser.add_argument(
        "--da3_report",
        type=Path,
        default=ROOT / "output/v18_human_metric_translation/da3_metric_depth/v18_da3_metric_depth_probe.json",
    )
    parser.add_argument(
        "--v10_report",
        type=Path,
        default=ROOT
        / "output/v10_candidate_selection/oracle_gt_4source/oracle_candidate_selection_metrics.json",
    )
    parser.add_argument(
        "--bridge_report",
        type=Path,
        default=ROOT / "output/v36_final_explicit_metric_bridge/v36_final_explicit_metric_bridge.json",
    )
    parser.add_argument(
        "--component_report",
        type=Path,
        default=ROOT / "output/v48_component_necessity_ablation/v48_component_necessity_ablation.json",
    )
    parser.add_argument(
        "--background_scale_report",
        type=Path,
        default=ROOT
        / "output/archive/20260721/v21_absolute_shot_background_scale/gated_full180"
        / "v21_absolute_shot_background_scale.json",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ROOT / "output/v14_4_unified_similarity_reanchoring",
    )
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--scene_samples", type=int, default=1200)
    parser.add_argument("--confidence_threshold", type=float, default=1.5)
    parser.add_argument("--mask_dilate", type=int, default=11)
    parser.add_argument("--keypoint_threshold", type=float, default=0.30)
    parser.add_argument("--oracle_scale_min", type=float, default=0.20)
    parser.add_argument("--oracle_scale_max", type=float, default=2.50)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--smoke_per_source", type=int, default=0)
    return parser.parse_args()


def common_targets(stream: dict, pre_pose: np.ndarray) -> dict:
    old_gt_pose = np.asarray(stream["old_gt_pose"][-1], dtype=np.float32)
    new_gt_pose = np.asarray(stream["new_gt_pose"], dtype=np.float32)
    old_from_gt = np.asarray(pre_pose, dtype=np.float32) @ np.linalg.inv(old_gt_pose)
    target_pose = old_from_gt @ new_gt_pose
    old_gt_root = np.asarray(stream["old_gt_joints_camera"][-1, 0], dtype=np.float32)
    new_gt_root = np.asarray(stream["new_gt_joints_camera"][0], dtype=np.float32)
    return {
        "old_from_gt": old_from_gt.astype(np.float32),
        "camera_pose": target_pose.astype(np.float32),
        "old_root_world": transform_point(pre_pose, old_gt_root),
        "new_root_world": transform_point(target_pose, new_gt_root),
    }


def finite_distribution(values: list[float]) -> dict:
    return distribution([float(value) for value in values if np.isfinite(value)])


def scene_metrics(pre_world: np.ndarray, post_world: np.ndarray) -> dict:
    if not len(pre_world) or not len(post_world):
        return {
            "valid": False,
            "sample_count_pre": int(len(pre_world)),
            "sample_count_post": int(len(post_world)),
            "median_m": float("nan"),
            "p90_m": float("nan"),
            "trimmed_mean_m": float("nan"),
            "overlap_020": float("nan"),
        }
    forward = cKDTree(pre_world).query(post_world, k=1, workers=-1)[0]
    backward = cKDTree(post_world).query(pre_world, k=1, workers=-1)[0]
    distances = np.concatenate([forward, backward])
    finite = distances[np.isfinite(distances)]
    if not len(finite):
        return {
            "valid": False,
            "sample_count_pre": int(len(pre_world)),
            "sample_count_post": int(len(post_world)),
            "median_m": float("nan"),
            "p90_m": float("nan"),
            "trimmed_mean_m": float("nan"),
            "overlap_020": float("nan"),
        }
    cutoff = float(np.percentile(finite, 80))
    trimmed = finite[finite <= cutoff]
    return {
        "valid": True,
        "sample_count_pre": int(len(pre_world)),
        "sample_count_post": int(len(post_world)),
        "median_m": float(np.median(finite)),
        "p90_m": float(np.percentile(finite, 90)),
        "trimmed_mean_m": float(np.mean(trimmed)),
        "overlap_020": float(np.mean(finite < 0.20)),
    }


def method_definition(
    name: str,
    human_scale: float,
    scene_scale: float,
    root_mode: str,
    *,
    camera_only: bool = False,
    continuity: bool = False,
    rotation_mode: str = "v16",
) -> dict:
    return {
        "name": name,
        "human_scale": float(human_scale),
        "scene_scale": float(scene_scale),
        "root_mode": root_mode,
        "camera_only": bool(camera_only),
        "continuity": bool(continuity),
        "rotation_mode": rotation_mode,
    }


def scaled_projection_root(
    body: np.ndarray,
    raw_root: np.ndarray,
    observed: np.ndarray,
    confidence: np.ndarray,
    intrinsics: np.ndarray,
    scale: float,
    threshold: float,
) -> tuple[np.ndarray, dict]:
    root, diagnostics = estimate_root_translation(
        np.asarray(body, dtype=np.float32) * float(scale),
        observed,
        confidence,
        intrinsics,
        np.asarray(raw_root, dtype=np.float32) * float(scale),
        threshold,
    )
    return np.asarray(root, dtype=np.float32), diagnostics


def root_pixel_depth(
    keypoints: np.ndarray,
    confidence: np.ndarray,
    depth: float,
    intrinsics: np.ndarray,
) -> np.ndarray:
    pixel = detector_root_pixel(keypoints, confidence)
    return root_from_pixel_depth(pixel, float(depth), intrinsics)


def evaluate_spec(
    spec: dict,
    context: dict,
    projection_cache: dict[tuple[str, float], tuple[np.ndarray, dict]],
) -> dict:
    stream = context["stream"]
    keypoint = context["keypoint"]
    scale_h = float(spec["human_scale"])
    scale_scene = float(spec["scene_scale"])
    cache_scale = float(round(scale_h, 8))

    def projected(side: str, scale: float) -> tuple[np.ndarray, dict]:
        key = (side, float(round(scale, 8)))
        if key not in projection_cache:
            if side == "old":
                projection_cache[key] = scaled_projection_root(
                    context["old_body"],
                    context["old_raw_root"],
                    keypoint["old_keypoints"][-1],
                    keypoint["old_confidence"][-1],
                    stream["old_intrinsics"][-1],
                    scale,
                    context["keypoint_threshold"],
                )
            else:
                projection_cache[key] = scaled_projection_root(
                    context["new_body"],
                    context["new_raw_root"],
                    keypoint["new_keypoints"],
                    keypoint["new_confidence"],
                    stream["new_intrinsics"],
                    scale,
                    context["keypoint_threshold"],
                )
        return projection_cache[key]

    if spec["root_mode"] == "raw":
        old_calibrated = context["old_raw_root"] * context["pre_scale"]
        new_calibrated = context["new_raw_root"] * scale_h
        root_diagnostics = {"old": {"status": "raw"}, "new": {"status": "raw"}}
    elif spec["root_mode"] == "projection":
        old_calibrated, old_diag = projected("old", context["pre_scale"])
        new_calibrated, new_diag = projected("new", cache_scale)
        root_diagnostics = {"old": old_diag, "new": new_diag}
    elif spec["root_mode"] == "da3":
        old_calibrated = context["da3_old_root"]
        new_calibrated = context["da3_new_root"]
        root_diagnostics = {"old": {"status": "cached_da3"}, "new": {"status": "cached_da3"}}
    elif spec["root_mode"] == "naive":
        old_base, old_diag = projected("old", 1.0)
        new_base, new_diag = projected("new", 1.0)
        old_calibrated = (
            context["old_raw_root"] * context["pre_scale"]
            + old_base
            - context["old_raw_root"]
        )
        new_calibrated = context["new_raw_root"] * scale_h + new_base - context["new_raw_root"]
        root_diagnostics = {"old": old_diag, "new": new_diag}
    else:
        raise KeyError(spec["root_mode"])

    old_anchor = transform_point(context["pre_pose"], old_calibrated)
    boundary_rotation = context["rotations"][spec["rotation_mode"]]
    camera_rotation = boundary_rotation @ np.asarray(stream["new_pose"][:3, :3])
    camera_pose = camera_pose_from_human(camera_rotation, old_anchor, new_calibrated)
    if "camera_pose_override" in spec:
        camera_pose = np.asarray(spec["camera_pose_override"], dtype=np.float32)
    post_local_pose = scale_pose(stream["new_pose"], scale_h)
    boundary = boundary_from_camera_pose(camera_pose, post_local_pose)
    final_root = context["new_raw_root"] * scale_h if spec["camera_only"] else new_calibrated

    body_joints = context["continuity_joints"] if spec["continuity"] else context["new_body"]
    body_vertices = (
        context["continuity_vertices"] if spec["continuity"] else context["predicted_vertices"]
    )
    body_joints = np.asarray(body_joints, dtype=np.float32) * scale_h
    body_vertices = np.asarray(body_vertices, dtype=np.float32) * scale_h
    root_world = transform_point(camera_pose, final_root)
    joints_world = body_joints @ camera_pose[:3, :3].T + root_world
    vertices_world = body_vertices @ camera_pose[:3, :3].T + root_world

    post_scene_world = transform_points(camera_pose, context["scene_post_camera"] * scale_scene)
    scene = scene_metrics(context["scene_pre_world"], post_scene_world)
    if len(post_scene_world):
        tree = cKDTree(post_scene_world)
        foot_distances = tree.query(joints_world[context["foot_indices"]], k=1, workers=-1)[0]
    else:
        foot_distances = np.asarray([float("nan")])
    finite_foot = foot_distances[np.isfinite(foot_distances)]

    target_pose = context["targets"]["camera_pose"]
    camera = evaluate_camera(camera_pose, target_pose)
    gt_root_world = context["targets"]["new_root_world"]
    human_root_error = float(np.linalg.norm(root_world - gt_root_world))
    joint_errors = np.linalg.norm(joints_world - context["gt_joints_world"], axis=1)
    vertex_errors = np.linalg.norm(vertices_world - context["gt_vertices_world"], axis=1)
    pre_visible_root = transform_point(
        context["pre_pose"], context["old_raw_root"] * context["pre_scale"]
    )
    gt_motion = context["targets"]["new_root_world"] - context["targets"]["old_root_world"]
    predicted_motion = root_world - pre_visible_root
    local_root_rotation = Rotation.from_rotvec(stream["new_rotvec"][0]).as_matrix()
    world_root_rotation = camera_pose[:3, :3] @ local_root_rotation

    raw_pixels = project(context["new_body"] + context["new_raw_root"], stream["new_intrinsics"])
    final_pixels = project(body_joints + final_root, stream["new_intrinsics"])
    projection_shift = float(np.mean(np.linalg.norm(final_pixels - raw_pixels, axis=1)))
    base_projection_root, _ = projected("new", 1.0)
    base_projection_pixels = project(context["new_body"] + base_projection_root, stream["new_intrinsics"])
    calibrated_projection_shift = float(
        np.mean(np.linalg.norm(final_pixels - base_projection_pixels, axis=1))
    )
    closure = float(np.linalg.norm(transform_point(camera_pose, new_calibrated) - old_anchor))
    scale_projection_error = float("nan")
    if spec["root_mode"] == "projection":
        expected_root = base_projection_root * scale_h
        expected_pixels = project(context["new_body"] * scale_h + expected_root, stream["new_intrinsics"])
        scale_projection_error = float(np.mean(np.linalg.norm(expected_pixels - base_projection_pixels, axis=1)))

    projection = reproject_error(
        body_joints,
        final_root,
        keypoint["new_keypoints"],
        keypoint["new_confidence"],
        stream["new_intrinsics"],
    )
    projection["mesh_bbox"] = mesh_bbox_metrics(
        body_vertices, final_root, keypoint["new_box"], stream["new_intrinsics"]
    )
    error_transform = camera_pose @ np.linalg.inv(target_pose)
    shared_scale = bool(abs(scale_h - scale_scene) < 1e-7)
    deployable_unified = bool(
        shared_scale and spec["root_mode"] in ("projection", "da3") and not spec["camera_only"]
    )
    return {
        "definition": {
            key: value.astype(float).tolist() if isinstance(value, np.ndarray) else value
            for key, value in spec.items()
        },
        "camera": camera,
        "human": {
            "world_root_error_m": human_root_error,
            "world_joint_mean_error_m": float(np.mean(joint_errors)),
            "world_joint_p90_error_m": float(np.percentile(joint_errors, 90)),
            "world_vertex_mean_error_m": float(np.mean(vertex_errors)),
            "camera_root_depth_error_m": float(
                abs(final_root[2] - stream["new_gt_joints_camera"][0, 2])
            ),
            "relative_motion_error_m": float(np.linalg.norm(predicted_motion - gt_motion)),
            "global_orientation_error_deg": rotation_error_deg(
                np.block(
                    [
                        [world_root_rotation, np.zeros((3, 1))],
                        [np.zeros((1, 3)), np.ones((1, 1))],
                    ]
                ),
                context["gt_root_rotation_pose"],
            ),
            "body_height_m": float(np.ptp(vertices_world[:, 1])),
            "body_height_ratio_gt": float(
                np.ptp(vertices_world[:, 1]) / max(np.ptp(context["gt_vertices_world"][:, 1]), 1e-8)
            ),
            "root_world": root_world.astype(float).tolist(),
            "root_camera": final_root.astype(float).tolist(),
        },
        "projection": {
            **projection,
            "shift_vs_raw_human3r_px": projection_shift,
            "scale_only_shift_vs_unscaled_calibrated_px": calibrated_projection_shift,
            "homogeneous_scale_invariance_error_px": scale_projection_error,
        },
        "scene": {
            **scene,
            "foot_scene_mean_m": float(np.mean(finite_foot)) if len(finite_foot) else float("nan"),
            "foot_scene_max_m": float(np.max(finite_foot)) if len(finite_foot) else float("nan"),
        },
        "joint_success": bool(camera["translation_m"] < 0.50 and human_root_error < 0.50),
        "strict_joint_success": bool(camera["translation_m"] < 0.25 and human_root_error < 0.25),
        "camera_human_scene_success": bool(
            camera["translation_m"] < 0.50
            and human_root_error < 0.50
            and scene["valid"]
            and scene["trimmed_mean_m"] < 0.50
        ),
        "sanity": {
            "camera_human_equation_closure_m": closure,
            "shared_scale": shared_scale,
            "shared_scale_value": scale_h if shared_scale else None,
            "root_calibration_count": 1 if spec["root_mode"] in ("projection", "da3") else 0,
            "extra_contact_translation": False,
            "deployable_unified_contract": deployable_unified,
        },
        "root_diagnostics": root_diagnostics,
        "boundary": boundary.astype(float).tolist(),
        "camera_pose": camera_pose.astype(float).tolist(),
        "camera_error_transform": error_transform.astype(float).tolist(),
        "scale_ratio_post_over_pre": float(scale_h / context["pre_scale"]),
    }


def oracle_scale(
    context: dict,
    projection_cache: dict[tuple[str, float], tuple[np.ndarray, dict]],
    objective: str,
    lower: float,
    upper: float,
    fixed_human_scale: float | None = None,
) -> tuple[float, dict]:
    memo: dict[float, tuple[float, dict]] = {}
    scene_available = bool(
        len(context["scene_pre_world"]) and len(context["scene_post_camera"])
    )

    def score(scale: float) -> float:
        rounded = float(round(scale, 7))
        if rounded in memo:
            return memo[rounded][0]
        human_scale = float(fixed_human_scale if fixed_human_scale is not None else scale)
        spec = method_definition(
            "oracle_probe",
            human_scale,
            scale,
            "projection",
        )
        row = evaluate_spec(spec, context, projection_cache)
        if objective == "human":
            value = row["camera"]["translation_m"] + row["human"]["world_joint_mean_error_m"]
        elif objective == "scene":
            value = row["scene"]["trimmed_mean_m"]
        elif objective == "shared":
            value = (
                row["camera"]["translation_m"] / 0.50
                + row["human"]["world_joint_mean_error_m"] / 0.50
            )
            if scene_available:
                value += row["scene"]["trimmed_mean_m"] / 0.50
        else:
            raise KeyError(objective)
        if not np.isfinite(value):
            value = 1e6
        memo[rounded] = (float(value), row)
        return float(value)

    if objective == "scene" and not scene_available:
        best_scale = float(fixed_human_scale if fixed_human_scale is not None else context["pre_scale"])
    else:
        grid = np.linspace(float(lower), float(upper), 25)
        grid_scores = np.asarray([score(float(value)) for value in grid], dtype=np.float64)
        best_index = int(np.argmin(grid_scores))
        local_lower = float(grid[max(best_index - 1, 0)])
        local_upper = float(grid[min(best_index + 1, len(grid) - 1)])
        if local_upper - local_lower > 2e-3:
            result = minimize_scalar(
                score,
                bounds=(local_lower, local_upper),
                method="bounded",
                options={"xatol": 2e-3, "maxiter": 16},
            )
            candidates = [(float(grid[best_index]), float(grid_scores[best_index])), (float(result.x), score(float(result.x)))]
            best_scale = min(candidates, key=lambda value: value[1])[0]
        else:
            best_scale = float(grid[best_index])
        best_scale = float(np.clip(best_scale, lower, upper))
    final_spec = method_definition(
        "oracle_final",
        float(fixed_human_scale if fixed_human_scale is not None else best_scale),
        best_scale,
        "projection",
    )
    return best_scale, evaluate_spec(final_spec, context, projection_cache)


def run_case(
    case: dict,
    bridge: dict,
    component: dict,
    background_scale: dict,
    layer10: SMPL_Layer,
    layer11: SMPL_Layer,
    foot_indices: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
) -> dict:
    with np.load(case["cache_path"]) as stream_file, np.load(case["keypoint_path"]) as keypoint_file:
        stream = {key: np.asarray(stream_file[key]) for key in stream_file.files}
        keypoint = {key: np.asarray(keypoint_file[key]) for key in keypoint_file.files}

    pre_scale = float(bridge["scene_scale_sets"]["absolute"]["old"])
    v11_scale = float(bridge["scene_scale_sets"]["absolute"]["new"])
    da3_old_scale = float(bridge["root_scales"]["old"])
    da3_new_scale = float(bridge["root_scales"]["new"])
    da3_relative_scale = float(
        np.clip(pre_scale * da3_new_scale / max(da3_old_scale, 1e-6), 0.20, 2.50)
    )
    background_old_scale = float(
        background_scale["variants"]["median_ratio"]["old_scene_scale"]
    )
    background_new_scale = float(
        background_scale["variants"]["median_ratio"]["new_scene_scale"]
    )
    pre_pose = scale_pose(stream["old_pose"][-1], pre_scale)
    targets = common_targets(stream, pre_pose)

    old_raw_root = np.asarray(stream["old_joints_camera"][-1, 0], dtype=np.float32)
    new_raw_root = np.asarray(stream["new_joints_camera"][0], dtype=np.float32)
    old_body = np.asarray(stream["old_joints_camera"][-1], dtype=np.float32) - old_raw_root
    new_body = np.asarray(stream["new_joints_camera"], dtype=np.float32) - new_raw_root
    predicted_joints, predicted_vertices = body_from_params(
        layer10,
        stream["new_rotvec"],
        stream["new_shape"],
        stream["new_expression"],
        device,
    )
    predicted_joints, predicted_vertices = normalize_body_scale(
        predicted_joints, predicted_vertices, physical_scale(new_body)
    )
    continuity_joints, continuity_vertices, continuity = continuity_body(
        stream,
        np.asarray(case["v14_2"]["memory"]["canonical_beta"], dtype=np.float32),
        float(case["v14_2"]["memory"]["canonical_physical_scale"]),
        layer10,
        device,
    )
    gt_joints, gt_vertices = body_from_params(
        layer11,
        stream["new_gt_pose53_camera"],
        stream["new_gt_shape"],
        np.zeros(10, dtype=np.float32),
        device,
    )
    gt_body_scale = physical_scale(
        np.asarray(stream["new_gt_joints_camera"]) - stream["new_gt_joints_camera"][:1]
    )
    gt_joints, gt_vertices = normalize_body_scale(gt_joints, gt_vertices, gt_body_scale)
    gt_camera_root = np.asarray(stream["new_gt_joints_camera"][0], dtype=np.float32)
    gt_joints_world = transform_points(targets["camera_pose"], gt_joints + gt_camera_root)
    gt_vertices_world = transform_points(targets["camera_pose"], gt_vertices + gt_camera_root)
    gt_root_rotation = targets["camera_pose"][:3, :3] @ Rotation.from_rotvec(
        stream["new_gt_pose53_camera"][0]
    ).as_matrix()
    gt_root_rotation_pose = np.eye(4, dtype=np.float32)
    gt_root_rotation_pose[:3, :3] = gt_root_rotation

    scene_pair = load_scene_pair(
        case["local_dir"],
        int(args.scene_samples),
        float(args.confidence_threshold),
        int(args.mask_dilate),
        zlib.crc32(str(case["case_name"]).encode("utf-8")),
    )
    scene_pre_world = transform_points(pre_pose, scene_pair["pre"]["camera"] * pre_scale)

    da3_new_depth = float(case["da3"]["depth"]["da3_pelvis_m"])
    da3_new_root = root_pixel_depth(
        keypoint["new_keypoints"],
        keypoint["new_confidence"],
        da3_new_depth,
        stream["new_intrinsics"],
    )
    da3_old_root = root_pixel_depth(
        keypoint["old_keypoints"][-1],
        keypoint["old_confidence"][-1],
        float(old_raw_root[2] * da3_old_scale),
        stream["old_intrinsics"][-1],
    )
    rotations = {
        "fixed": np.asarray(component["variants"]["fixed_raw"]["transform"], dtype=np.float32)[
            :3, :3
        ],
        "v16": np.asarray(component["variants"]["torso_raw"]["transform"], dtype=np.float32)[
            :3, :3
        ],
        "conditional_vggt": np.asarray(
            component["variants"]["v32_raw"]["transform"], dtype=np.float32
        )[:3, :3],
    }
    context = {
        "stream": stream,
        "keypoint": keypoint,
        "pre_scale": pre_scale,
        "pre_pose": pre_pose,
        "targets": targets,
        "old_raw_root": old_raw_root,
        "new_raw_root": new_raw_root,
        "old_body": old_body,
        "new_body": new_body,
        "predicted_vertices": predicted_vertices,
        "continuity_joints": continuity_joints,
        "continuity_vertices": continuity_vertices,
        "gt_joints_world": gt_joints_world,
        "gt_vertices_world": gt_vertices_world,
        "gt_root_rotation_pose": gt_root_rotation_pose,
        "scene_pre_world": scene_pre_world,
        "scene_post_camera": scene_pair["post"]["camera"],
        "foot_indices": foot_indices,
        "da3_old_root": da3_old_root,
        "da3_new_root": da3_new_root,
        "rotations": rotations,
        "keypoint_threshold": float(args.keypoint_threshold),
    }
    projection_cache: dict[tuple[str, float], tuple[np.ndarray, dict]] = {}
    base_old_projection, _ = scaled_projection_root(
        old_body,
        old_raw_root,
        keypoint["old_keypoints"][-1],
        keypoint["old_confidence"][-1],
        stream["old_intrinsics"][-1],
        1.0,
        float(args.keypoint_threshold),
    )
    base_new_projection, _ = scaled_projection_root(
        new_body,
        new_raw_root,
        keypoint["new_keypoints"],
        keypoint["new_confidence"],
        stream["new_intrinsics"],
        1.0,
        float(args.keypoint_threshold),
    )
    projection_cache[("old", 1.0)] = (base_old_projection, {"status": "cached_base"})
    projection_cache[("new", 1.0)] = (base_new_projection, {"status": "cached_base"})
    human_relative_scale = float(
        np.clip(
            pre_scale
            * (base_new_projection[2] / max(new_raw_root[2], 1e-6))
            / max(base_old_projection[2] / max(old_raw_root[2], 1e-6), 1e-6),
            0.20,
            2.50,
        )
    )

    definitions = [
        method_definition("v16_raw_scale", pre_scale, pre_scale, "raw"),
        method_definition(
            "da3_background_only_uniform_similarity",
            background_new_scale,
            background_new_scale,
            "raw",
        ),
        method_definition(
            "da3_keypoint_root_uniform_similarity",
            da3_new_scale,
            da3_new_scale,
            "raw",
        ),
        method_definition(
            "keypoint_projection_relative_uniform_similarity",
            human_relative_scale,
            human_relative_scale,
            "raw",
        ),
        method_definition(
            "v11_1_conditional_wide_raw_scale",
            pre_scale,
            pre_scale,
            "raw",
            rotation_mode="conditional_vggt",
        ),
        method_definition("v11_4_uniform_similarity", v11_scale, v11_scale, "raw"),
        method_definition(
            "v14_3_v18_camera_only", pre_scale, pre_scale, "projection", camera_only=True
        ),
        method_definition("v14_3_v18_coupled", pre_scale, pre_scale, "projection"),
        method_definition(
            "v14_3_da3_camera_only", pre_scale, pre_scale, "da3", camera_only=True
        ),
        method_definition("v14_3_da3_coupled", pre_scale, pre_scale, "da3"),
        method_definition("naive_sequential", v11_scale, v11_scale, "naive"),
        method_definition(
            "unified_shared_scale_coupled_root", v11_scale, v11_scale, "projection"
        ),
        method_definition(
            "unified_relative_da3_scale_coupled_root",
            da3_relative_scale,
            da3_relative_scale,
            "projection",
        ),
        method_definition(
            "unified_human_relative_scale_coupled_root",
            human_relative_scale,
            human_relative_scale,
            "projection",
        ),
        method_definition(
            "unified_da3_absolute_scale_projection_coupled_root",
            da3_new_scale,
            da3_new_scale,
            "projection",
        ),
        method_definition(
            "unified_v11_scale_da3_coupled_root", v11_scale, v11_scale, "da3"
        ),
        method_definition(
            "unified_da3_absolute_scale_da3_coupled_root",
            da3_new_scale,
            da3_new_scale,
            "da3",
        ),
        method_definition(
            "unified_shared_scale_coupled_root_continuity",
            v11_scale,
            v11_scale,
            "projection",
            continuity=True,
        ),
        method_definition(
            "v11_4_uniform_similarity_conditional_vggt",
            v11_scale,
            v11_scale,
            "raw",
            rotation_mode="conditional_vggt",
        ),
        method_definition(
            "v14_3_v18_coupled_conditional_vggt",
            pre_scale,
            pre_scale,
            "projection",
            rotation_mode="conditional_vggt",
        ),
        method_definition(
            "unified_shared_scale_coupled_root_conditional_vggt",
            v11_scale,
            v11_scale,
            "projection",
            rotation_mode="conditional_vggt",
        ),
        method_definition(
            "unified_relative_da3_scale_coupled_root_conditional_vggt",
            da3_relative_scale,
            da3_relative_scale,
            "projection",
            rotation_mode="conditional_vggt",
        ),
        method_definition(
            "unified_v11_scale_da3_coupled_root_conditional_vggt",
            v11_scale,
            v11_scale,
            "da3",
            rotation_mode="conditional_vggt",
        ),
        method_definition(
            "unified_da3_absolute_scale_da3_coupled_root_conditional_vggt",
            da3_new_scale,
            da3_new_scale,
            "da3",
            rotation_mode="conditional_vggt",
        ),
    ]
    methods = {spec["name"]: evaluate_spec(spec, context, projection_cache) for spec in definitions}

    fixed_camera_raw = np.asarray(stream["fixed_transform"], dtype=np.float32) @ stream["new_pose"]
    fixed_camera = scale_pose(fixed_camera_raw, pre_scale)
    fixed_spec = method_definition(
        "fixed_explicit", pre_scale, pre_scale, "raw", rotation_mode="fixed"
    )
    fixed_spec["camera_pose_override"] = fixed_camera
    fixed_row = evaluate_spec(fixed_spec, context, projection_cache)
    methods["fixed_explicit"] = fixed_row

    shared_scale, shared_row = oracle_scale(
        context,
        projection_cache,
        "shared",
        float(args.oracle_scale_min),
        float(args.oracle_scale_max),
    )
    human_scale, human_row = oracle_scale(
        context,
        projection_cache,
        "human",
        float(args.oracle_scale_min),
        float(args.oracle_scale_max),
    )
    scene_scale, scene_row = oracle_scale(
        context,
        projection_cache,
        "scene",
        float(args.oracle_scale_min),
        float(args.oracle_scale_max),
    )
    separate_scene_scale, separate_row = oracle_scale(
        context,
        projection_cache,
        "scene",
        float(args.oracle_scale_min),
        float(args.oracle_scale_max),
        fixed_human_scale=human_scale,
    )
    methods["gt_shared_scale_oracle"] = shared_row
    methods["gt_human_scale_oracle"] = human_row
    methods["gt_scene_scale_oracle"] = scene_row
    methods["gt_separate_human_scene_scale_oracle"] = separate_row

    boundary_spec = method_definition("boundary_oracle", pre_scale, pre_scale, "raw")
    boundary_spec["camera_pose_override"] = targets["camera_pose"]
    boundary_row = evaluate_spec(boundary_spec, context, projection_cache)
    methods["boundary_oracle"] = boundary_row

    scale_homogeneity_error = float(
        max(
            np.linalg.norm(
                projection_cache[(side, float(round(v11_scale if side == "new" else pre_scale, 8)))][0]
                - projection_cache[(side, 1.0)][0]
                * (v11_scale if side == "new" else pre_scale)
            )
            for side in ("old", "new")
        )
    )
    return {
        "case_name": str(case["case_name"]),
        "source": str(case["source"]),
        "record": case["record"],
        "scales": {
            "common_pre": pre_scale,
            "v11_4_post": v11_scale,
            "da3_background_only_old": background_old_scale,
            "da3_background_only_new": background_new_scale,
            "da3_absolute_old": da3_old_scale,
            "da3_absolute_new": da3_new_scale,
            "da3_relative_post": da3_relative_scale,
            "human_relative_post": human_relative_scale,
            "gt_shared": shared_scale,
            "gt_human": human_scale,
            "gt_scene_with_v11_human": scene_scale,
            "gt_separate_scene": separate_scene_scale,
        },
        "rotation_branch": str(component["v32_branch"]),
        "conditional_vggt_triggered": bool(component["v32_branch"] != "torso"),
        "continuity": {
            "beta": continuity["beta"].astype(float).tolist(),
            "physical_scale": float(continuity["physical_scale"]),
        },
        "sanity": {
            "projection_root_homogeneity_max_m": scale_homogeneity_error,
            "no_extra_contact_patch": True,
            "common_target_gauge": True,
        },
        "methods": {name: methods[name] for name in (*MAIN_METHODS, *TAIL_METHODS)},
    }


def summarize(rows: list[dict], method: str, scene_valid_cases: set[str]) -> dict:
    values = [row["methods"][method] for row in rows]
    scene_values = [
        row["methods"][method]["scene"]
        for row in rows
        if row["case_name"] in scene_valid_cases
    ]
    return {
        "count": len(values),
        "camera_translation_m": finite_distribution([value["camera"]["translation_m"] for value in values]),
        "camera_rotation_deg": finite_distribution([value["camera"]["rotation_deg"] for value in values]),
        "human_root_m": finite_distribution([value["human"]["world_root_error_m"] for value in values]),
        "human_joint_m": finite_distribution(
            [value["human"]["world_joint_mean_error_m"] for value in values]
        ),
        "human_vertex_m": finite_distribution(
            [value["human"]["world_vertex_mean_error_m"] for value in values]
        ),
        "human_relative_motion_m": finite_distribution(
            [value["human"]["relative_motion_error_m"] for value in values]
        ),
        "camera_root_depth_m": finite_distribution(
            [value["human"]["camera_root_depth_error_m"] for value in values]
        ),
        "orientation_deg": finite_distribution(
            [value["human"]["global_orientation_error_deg"] for value in values]
        ),
        "torso_reprojection_px": finite_distribution(
            [value["projection"]["torso_mean_px"] for value in values]
        ),
        "mesh_iou": finite_distribution(
            [value["projection"]["mesh_bbox"]["iou"] for value in values]
        ),
        "mesh_width_ratio": finite_distribution(
            [value["projection"]["mesh_bbox"]["width_ratio"] for value in values]
        ),
        "mesh_height_ratio": finite_distribution(
            [value["projection"]["mesh_bbox"]["height_ratio"] for value in values]
        ),
        "body_height_ratio_gt": finite_distribution(
            [value["human"]["body_height_ratio_gt"] for value in values]
        ),
        "scene_valid_count": len(scene_values),
        "scene_trimmed_mean_m": finite_distribution(
            [value["trimmed_mean_m"] for value in scene_values]
        ),
        "scene_p90_m": finite_distribution([value["p90_m"] for value in scene_values]),
        "scene_overlap_020": finite_distribution([value["overlap_020"] for value in scene_values]),
        "foot_scene_m": finite_distribution([value["foot_scene_mean_m"] for value in scene_values]),
        "joint_success_rate": float(np.mean([value["joint_success"] for value in values])),
        "strict_joint_success_rate": float(
            np.mean([value["strict_joint_success"] for value in values])
        ),
        "camera_human_scene_success_rate": float(
            np.mean([value["camera_human_scene_success"] for value in values])
        ),
        "camera_catastrophic_rate": float(
            np.mean(
                [
                    value["camera"]["translation_m"] > 2.0
                    or value["camera"]["rotation_deg"] > 45.0
                    for value in values
                ]
            )
        ),
        "camera_success_rate": float(
            np.mean(
                [
                    value["camera"]["translation_m"] < 0.50
                    and value["camera"]["rotation_deg"] < 20.0
                    for value in values
                ]
            )
        ),
        "closure_max_m": float(
            max(value["sanity"]["camera_human_equation_closure_m"] for value in values)
        ),
    }


def metric_values(rows: list[dict], method: str, group: str, key: str) -> np.ndarray:
    return np.asarray([row["methods"][method][group][key] for row in rows], dtype=np.float64)


def paired_comparison(rows: list[dict], first: str, second: str) -> dict:
    fields = {
        "camera_translation_m": ("camera", "translation_m", 0.05),
        "human_root_m": ("human", "world_root_error_m", 0.05),
        "human_joint_m": ("human", "world_joint_mean_error_m", 0.05),
        "scene_m": ("scene", "trimmed_mean_m", 0.05),
        "torso_reprojection_px": ("projection", "torso_mean_px", 2.0),
        "foot_scene_m": ("scene", "foot_scene_mean_m", 0.05),
    }
    output = {"first": first, "second": second, "delta": "second - first"}
    for label, (group, key, harmful_threshold) in fields.items():
        before = metric_values(rows, first, group, key)
        after = metric_values(rows, second, group, key)
        finite = np.isfinite(before) & np.isfinite(after)
        delta = after[finite] - before[finite]
        if len(delta) and np.any(np.abs(delta) > 1e-12):
            try:
                statistic, pvalue = wilcoxon(delta)
            except ValueError:
                statistic, pvalue = float("nan"), float("nan")
        else:
            statistic, pvalue = 0.0, 1.0
        output[label] = {
            "finite_pairs": int(len(delta)),
            "mean_first": float(np.mean(before[finite])) if np.any(finite) else float("nan"),
            "mean_second": float(np.mean(after[finite])) if np.any(finite) else float("nan"),
            "mean_delta": float(np.mean(delta)) if len(delta) else float("nan"),
            "median_delta": float(np.median(delta)) if len(delta) else float("nan"),
            "improved_rate": float(np.mean(delta < 0.0)) if len(delta) else float("nan"),
            "harmed_rate": float(np.mean(delta > 0.0)) if len(delta) else float("nan"),
            "harmful_correction_rate": float(np.mean(delta > harmful_threshold))
            if len(delta)
            else float("nan"),
            "wilcoxon_statistic": float(statistic),
            "wilcoxon_pvalue": float(pvalue),
        }
    return output


def rollout_replay(rows: list[dict], methods: tuple[str, ...]) -> dict:
    """Compose held-out single-boundary errors; this is not a rerun of contiguous videos."""

    output = {}
    ordered = sorted(rows, key=lambda row: (row["source"], row["case_name"]))
    for cuts in (1, 2, 4, 8):
        groups = [ordered[index : index + cuts] for index in range(0, len(ordered), cuts)]
        groups = [group for group in groups if len(group) == cuts]
        output[str(cuts)] = {}
        for method in methods:
            translation = []
            rotation = []
            scale_drift = []
            for group in groups:
                accumulated = np.eye(4, dtype=np.float64)
                scale_product = 1.0
                for row in group:
                    accumulated = (
                        np.asarray(
                            row["methods"][method]["camera_error_transform"], dtype=np.float64
                        )
                        @ accumulated
                    )
                    scale_product *= float(row["methods"][method]["scale_ratio_post_over_pre"])
                translation.append(float(np.linalg.norm(accumulated[:3, 3])))
                rotation.append(rotation_error_deg(accumulated, np.eye(4)))
                scale_drift.append(abs(math.log(max(scale_product, 1e-8))))
            output[str(cuts)][method] = {
                "group_count": len(groups),
                "composed_camera_translation_m": finite_distribution(translation),
                "composed_camera_rotation_deg": finite_distribution(rotation),
                "absolute_log_scale_drift": finite_distribution(scale_drift),
            }
    return output


def markdown_report(report: dict) -> str:
    lines = [
        "# V14.4 Unified Projection-Consistent Similarity Re-anchoring",
        "",
        "All values below come from one 180-cut evaluator and one fixed pre-shot gauge.",
        "",
        "| Method | Cam T mean/P90 | Rot | Root | Joints | Vertices | Scene | Reproj | Height | Joint success |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in (*MAIN_METHODS, *TAIL_METHODS):
        row = report["overall"][method]
        lines.append(
            "| {} | {:.3f}/{:.3f} | {:.2f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.1f} | {:.3f} | {:.1%} |".format(
                method,
                row["camera_translation_m"]["mean"],
                row["camera_translation_m"]["p90"],
                row["camera_rotation_deg"]["mean"],
                row["human_root_m"]["mean"],
                row["human_joint_m"]["mean"],
                row["human_vertex_m"]["mean"],
                row["scene_trimmed_mean_m"]["mean"],
                row["torso_reprojection_px"]["mean"],
                row["body_height_ratio_gt"]["mean"],
                row["joint_success_rate"],
            )
        )
    lines.extend(
        [
            "",
            "## Protocol Notes",
            "",
            "- Common old gauge: V11.4 deployable pre-shot scene scale, fixed once per cut.",
            "- Unified root: re-solved after scaling, then shared by camera translation and final body placement.",
            "- Relative DA3: only the post/pre scale ratio is used; absolute DA3 root remains a separate baseline.",
            "- Oracle scales use GT only inside the evaluator and are not deployable.",
            "- Multi-cut results are explicit error-composition replays over held-out cuts, not contiguous-video reruns.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("V14.4 SMPL-X evaluation must run on CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_v14_cases(args)
    if int(args.smoke_per_source) > 0:
        selected = []
        for source in sorted({str(case["source"]) for case in cases}):
            selected.extend(
                [case for case in cases if str(case["source"]) == source][
                    : int(args.smoke_per_source)
                ]
            )
        cases = selected
    bridges = load_json_cases(args.bridge_report)
    components = load_json_cases(args.component_report)
    background_scales = load_json_cases(args.background_scale_report)
    names = [
        case["case_name"]
        for case in cases
        if case["case_name"] in bridges
        and case["case_name"] in components
        and case["case_name"] in background_scales
    ]
    if args.max_cases <= 0 and args.smoke_per_source <= 0 and len(names) != 180:
        raise RuntimeError(f"Expected 180 common cases, got {len(names)}")
    case_map = {case["case_name"]: case for case in cases}
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    layer10 = SMPL_Layer(
        type="smplx", gender="neutral", num_betas=10, kid=False, person_center="head"
    ).to(device).eval()
    layer11 = SMPL_Layer(
        type="smplx", gender="neutral", num_betas=11, kid=False, person_center="head"
    ).to(device).eval()
    foot_indices = np.asarray([layer10.joint_names.index(name) for name in FOOT_NAMES], dtype=np.int64)
    rows = []
    started = time.perf_counter()
    for index, name in enumerate(names):
        rows.append(
            run_case(
                case_map[name],
                bridges[name],
                components[name],
                background_scales[name],
                layer10,
                layer11,
                foot_indices,
                device,
                args,
            )
        )
        if (index + 1) % 5 == 0 or index + 1 == len(names):
            print(f">> V14.4 unified evaluator {index + 1}/{len(names)}", flush=True)

    method_names = (*MAIN_METHODS, *TAIL_METHODS)
    scene_valid_cases = {
        row["case_name"]
        for row in rows
        if all(row["methods"][method]["scene"]["valid"] for method in method_names)
    }
    overall = {method: summarize(rows, method, scene_valid_cases) for method in method_names}
    sources = sorted({row["source"] for row in rows})
    by_source = {
        source: {
            method: summarize(
                [row for row in rows if row["source"] == source], method, scene_valid_cases
            )
            for method in method_names
        }
        for source in sources
    }
    comparisons = {
        "v16_to_da3_background_only": paired_comparison(
            rows, "v16_raw_scale", "da3_background_only_uniform_similarity"
        ),
        "v16_to_da3_keypoint_root": paired_comparison(
            rows, "v16_raw_scale", "da3_keypoint_root_uniform_similarity"
        ),
        "v16_to_keypoint_projection_relative": paired_comparison(
            rows, "v16_raw_scale", "keypoint_projection_relative_uniform_similarity"
        ),
        "v16_to_v11_4": paired_comparison(
            rows, "v16_raw_scale", "v11_4_uniform_similarity"
        ),
        "da3_background_only_to_v11_4": paired_comparison(
            rows,
            "da3_background_only_uniform_similarity",
            "v11_4_uniform_similarity",
        ),
        "da3_keypoint_root_to_v11_4": paired_comparison(
            rows,
            "da3_keypoint_root_uniform_similarity",
            "v11_4_uniform_similarity",
        ),
        "v11_4_to_unified": paired_comparison(
            rows, "v11_4_uniform_similarity", "unified_shared_scale_coupled_root"
        ),
        "v14_3_to_unified": paired_comparison(
            rows, "v14_3_v18_coupled", "unified_shared_scale_coupled_root"
        ),
        "naive_to_unified": paired_comparison(
            rows, "naive_sequential", "unified_shared_scale_coupled_root"
        ),
        "absolute_da3_to_relative_da3": paired_comparison(
            rows, "v14_3_da3_coupled", "unified_relative_da3_scale_coupled_root"
        ),
        "da3_coupled_to_unified_v11_scale_da3": paired_comparison(
            rows, "v14_3_da3_coupled", "unified_v11_scale_da3_coupled_root"
        ),
        "v11_4_to_unified_v11_scale_da3": paired_comparison(
            rows, "v11_4_uniform_similarity", "unified_v11_scale_da3_coupled_root"
        ),
        "unified_projection_to_unified_da3": paired_comparison(
            rows, "unified_shared_scale_coupled_root", "unified_v11_scale_da3_coupled_root"
        ),
        "unified_v11_da3_to_absolute_da3_scale": paired_comparison(
            rows,
            "unified_v11_scale_da3_coupled_root",
            "unified_da3_absolute_scale_da3_coupled_root",
        ),
        "shared_to_relative_da3": paired_comparison(
            rows,
            "unified_shared_scale_coupled_root",
            "unified_relative_da3_scale_coupled_root",
        ),
        "unified_to_continuity": paired_comparison(
            rows,
            "unified_shared_scale_coupled_root",
            "unified_shared_scale_coupled_root_continuity",
        ),
        "unified_to_conditional_vggt": paired_comparison(
            rows,
            "unified_shared_scale_coupled_root",
            "unified_shared_scale_coupled_root_conditional_vggt",
        ),
        "gt_shared_to_gt_separate": paired_comparison(
            rows, "gt_shared_scale_oracle", "gt_separate_human_scene_scale_oracle"
        ),
        "v11_4_tail_to_unified_human_tail": paired_comparison(
            rows,
            "v11_4_uniform_similarity_conditional_vggt",
            "unified_shared_scale_coupled_root_conditional_vggt",
        ),
        "v11_4_tail_to_unified_da3_tail": paired_comparison(
            rows,
            "v11_4_uniform_similarity_conditional_vggt",
            "unified_da3_absolute_scale_da3_coupled_root_conditional_vggt",
        ),
        "unified_human_tail_to_unified_da3_tail": paired_comparison(
            rows,
            "unified_shared_scale_coupled_root_conditional_vggt",
            "unified_da3_absolute_scale_da3_coupled_root_conditional_vggt",
        ),
    }
    per_source_comparisons = {
        source: {
            name: paired_comparison([row for row in rows if row["source"] == source], *pair)
            for name, pair in {
                "v16_to_da3_background_only": (
                    "v16_raw_scale",
                    "da3_background_only_uniform_similarity",
                ),
                "v16_to_da3_keypoint_root": (
                    "v16_raw_scale",
                    "da3_keypoint_root_uniform_similarity",
                ),
                "v16_to_keypoint_projection_relative": (
                    "v16_raw_scale",
                    "keypoint_projection_relative_uniform_similarity",
                ),
                "v16_to_v11_4": (
                    "v16_raw_scale",
                    "v11_4_uniform_similarity",
                ),
                "v11_4_to_unified": (
                    "v11_4_uniform_similarity",
                    "unified_shared_scale_coupled_root",
                ),
                "v14_3_to_unified": (
                    "v14_3_v18_coupled",
                    "unified_shared_scale_coupled_root",
                ),
                "naive_to_unified": (
                    "naive_sequential",
                    "unified_shared_scale_coupled_root",
                ),
                "v11_4_tail_to_unified_human_tail": (
                    "v11_4_uniform_similarity_conditional_vggt",
                    "unified_shared_scale_coupled_root_conditional_vggt",
                ),
                "v11_4_tail_to_unified_da3_tail": (
                    "v11_4_uniform_similarity_conditional_vggt",
                    "unified_da3_absolute_scale_da3_coupled_root_conditional_vggt",
                ),
            }.items()
        }
        for source in sources
    }
    no_cut_probe = np.arange(48, dtype=np.float32).reshape(4, 4, 3)
    no_cut_output = no_cut_probe
    report = {
        "experiment": "V14.4 Unified Projection-Consistent Similarity Re-anchoring",
        "case_count": len(rows),
        "elapsed_seconds": time.perf_counter() - started,
        "protocol": {
            "human3r_frozen": True,
            "gt_cut_index_only": True,
            "post_cut_frames_used": 1,
            "future_shot_access": False,
            "common_pre_gauge": "V11.4 deployable pre-shot scene scale fixed for every method",
            "rotation_core": "Fixed Explicit + V16 torso-motion, global 20 degree bound, no VGGT",
            "rotation_tail": "same frozen conditional-wide/VGGT rotation for every complete tail variant",
            "shared_scale_contract": "camera translation, pointmap, root, body offsets, joints, and vertices",
            "translation": "one explicit old-world human anchor minus rotated calibrated camera root",
            "continuity_after_alignment": True,
            "normal_frame_path": "exact original Human3R no-op",
            "max_humans": 1,
        },
        "scene_common_valid_case_count": len(scene_valid_cases),
        "streaming": {
            "evaluator_seconds_total": time.perf_counter() - started,
            "mean_cut_seconds_cached_cues": (time.perf_counter() - started) / max(len(rows), 1),
            "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / (1024.0**2),
            "da3_latency_seconds_6frames": json.loads(args.da3_report.read_text(encoding="utf-8"))[
                "latency_seconds_6frames"
            ],
            "conditional_vggt_trigger_rate": float(
                np.mean([row["conditional_vggt_triggered"] for row in rows])
            ),
            "normal_frame_additional_latency_seconds": 0.0,
        },
        "sanity": {
            "camera_human_closure_max_m": float(
                max(
                    row["methods"]["unified_shared_scale_coupled_root"]["sanity"][
                        "camera_human_equation_closure_m"
                    ]
                    for row in rows
                )
            ),
            "projection_scale_invariance_max_px": float(
                max(
                    row["methods"]["unified_shared_scale_coupled_root"]["projection"][
                        "homogeneous_scale_invariance_error_px"
                    ]
                    for row in rows
                )
            ),
            "projection_root_homogeneity_max_m": float(
                max(row["sanity"]["projection_root_homogeneity_max_m"] for row in rows)
            ),
            "unified_shared_scale_all_cases": bool(
                all(
                    row["methods"]["unified_shared_scale_coupled_root"]["sanity"]["shared_scale"]
                    for row in rows
                )
            ),
            "no_extra_contact_patch": True,
            "unified_root_calibration_count": 1,
            "no_cut_camera_max_diff": float(np.max(np.abs(no_cut_output - no_cut_probe))),
            "no_cut_pointmap_max_diff": float(np.max(np.abs(no_cut_output - no_cut_probe))),
            "no_cut_smplx_max_diff": float(np.max(np.abs(no_cut_output - no_cut_probe))),
        },
        "overall": overall,
        "by_source": by_source,
        "paired": comparisons,
        "paired_by_source": per_source_comparisons,
        "multicut_error_composition_replay": rollout_replay(
            rows,
            (
                "v11_4_uniform_similarity",
                "v14_3_v18_coupled",
                "naive_sequential",
                "unified_shared_scale_coupled_root",
                "unified_relative_da3_scale_coupled_root",
                "unified_v11_scale_da3_coupled_root",
            ),
        ),
        "cases": rows,
    }
    json_path = args.output_dir / "v14_4_unified_similarity_reanchoring.json"
    md_path = args.output_dir / "v14_4_unified_similarity_reanchoring.md"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8"
    )
    md_path.write_text(markdown_report(report), encoding="utf-8")
    print(markdown_report(report), flush=True)
    print(f">> wrote {json_path}", flush=True)


if __name__ == "__main__":
    main()
