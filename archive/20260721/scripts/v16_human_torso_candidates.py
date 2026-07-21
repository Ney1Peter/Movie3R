#!/usr/bin/env python3
"""Generate deployable V16 torso-geometry rotation residual candidates on CUDA."""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from v10_boundary_gauge_partial_oracle_probe import (  # noqa: E402
    correct_gravity,
    normalize,
    signed_angle_about_axis,
)
from v10_latent_activation_patching_probe import camera_matrix  # noqa: E402
from v10_token_alignment_4source_probe import token_debug_to_arrays  # noqa: E402
from v11_gauge_neutral_first_write_probe import (  # noqa: E402
    DEFAULT_RECORDS,
    build_dataset,
    build_model,
    build_smpl_models,
    configure_views,
    predicted_human,
    read_jsonl,
    record_spec,
    texture_score,
)
from v12_build_gauge_neutral_teacher_cache import old_a_dataset  # noqa: E402
from v13_scene_coordinate_oracle import (  # noqa: E402
    camera_points,
    confidence,
    direct_transform_error,
    human_mask,
    transform_points,
    valid_points,
)
from v14_depth_free_world_memory_candidates import (  # noqa: E402
    fixed_explicit_human3r_gauge,
    human_diagnostics,
)
from v15_wide_baseline_boundary_bridge_candidates import (  # noqa: E402
    candidate_row,
    dense_clouds,
    predicted_human_summary,
)
from v9_learned_stream_alignment_overfit import gt_pose_from_view  # noqa: E402


DEFAULT_OUTPUT = REPO_ROOT / "output" / "v16_human_aware_rotation_residual" / "candidate_cache"
DEFAULT_V15 = REPO_ROOT / "output" / "v15_wide_baseline_boundary_bridge" / "candidate_cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--v15_candidate_dir", type=Path, default=DEFAULT_V15)
    parser.add_argument(
        "--candidate_root",
        type=Path,
        default=REPO_ROOT / "output" / "v10_candidate_selection" / "oracle_gt_4source" / "cases",
    )
    parser.add_argument("--data_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data"))
    parser.add_argument("--model_path", type=Path, default=REPO_ROOT / "src" / "human3r_896L.pth")
    parser.add_argument("--device", required=True)
    parser.add_argument("--resolution", type=int, nargs=2, default=(512, 288))
    parser.add_argument("--resize_mode", default="human3r_demo")
    parser.add_argument("--max_post_frames", type=int, default=3)
    parser.add_argument("--warmup_frames", type=int, default=8)
    parser.add_argument("--history_frames", type=int, default=5)
    parser.add_argument("--point_samples_per_frame", type=int, default=768)
    parser.add_argument("--max_yaw_residual_deg", type=float, default=45.0)
    parser.add_argument("--max_gravity_residual_deg", type=float, default=15.0)
    parser.add_argument("--translation_iters", type=int, default=8)
    parser.add_argument("--translation_max_distance", type=float, default=0.60)
    parser.add_argument("--translation_min_distance", type=float, default=0.12)
    parser.add_argument("--root_check_margin_m", type=float, default=0.25)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--sources", nargs="*", default=())
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def one_batch(dataset) -> list[dict]:
    return next(iter(torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)))


def load_v15_cases(root: Path) -> dict[str, dict]:
    cases = {}
    for path in sorted(glob.glob(str(root / "v15_candidates_shard_*_of_*.json"))):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for case in payload["cases"]:
            cases[str(case["case_name"])] = case
    return cases


def rotation_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = normalize(axis)
    return Rotation.from_rotvec(float(angle_rad) * axis).as_matrix().astype(np.float32)


def bounded_rotation(rotation: np.ndarray, maximum_deg: float) -> tuple[np.ndarray, float, bool]:
    rotvec = Rotation.from_matrix(np.asarray(rotation, dtype=np.float64)).as_rotvec()
    angle = float(np.linalg.norm(rotvec))
    maximum = math.radians(float(maximum_deg))
    clipped = angle > maximum
    if clipped and angle > 1e-8:
        rotvec *= maximum / angle
    return Rotation.from_rotvec(rotvec).as_matrix().astype(np.float32), math.degrees(min(angle, maximum)), clipped


def robust_mean_rotvec(deltas: list[np.ndarray]) -> tuple[np.ndarray, dict]:
    if not deltas:
        return np.zeros(3, dtype=np.float32), {"count": 0, "spread_deg": float("nan")}
    rotvecs = np.stack([Rotation.from_matrix(delta.astype(np.float64)).as_rotvec() for delta in deltas])
    center = np.median(rotvecs, axis=0)
    distances = np.linalg.norm(rotvecs - center[None], axis=1)
    keep = distances <= max(math.radians(10.0), float(np.median(distances) + 2.5 * np.median(np.abs(distances - np.median(distances)))))
    if int(keep.sum()):
        center = np.mean(rotvecs[keep], axis=0)
    return center.astype(np.float32), {
        "count": int(len(rotvecs)),
        "inlier_count": int(keep.sum()),
        "spread_deg": float(np.degrees(np.median(distances))),
        "angular_speed_deg_per_frame": float(np.degrees(np.linalg.norm(center))),
    }


def predict_torso_frames(old_humans: list[dict | None], post_count: int) -> tuple[list[np.ndarray] | None, dict]:
    valid = [human for human in old_humans if human is not None]
    if not valid:
        return None, {"status": "no_pre_human"}
    frames = [human["torso"] for human in valid]
    deltas = [frames[index] @ frames[index - 1].T for index in range(1, len(frames))]
    omega, diagnostics = robust_mean_rotvec(deltas)
    predicted = []
    for offset in range(1, post_count + 1):
        predicted.append(
            (Rotation.from_rotvec(offset * omega.astype(np.float64)).as_matrix() @ frames[-1]).astype(np.float32)
        )
    return predicted, {"status": "ok", **diagnostics}


def yaw_residual(
    coarse_rotation: np.ndarray,
    source_frames: list[np.ndarray],
    target_frames: list[np.ndarray],
    maximum_deg: float,
) -> tuple[np.ndarray, dict]:
    angles = []
    for source, target in zip(source_frames, target_frames):
        target_up = normalize(target[:, 1])
        mapped_heading = coarse_rotation @ normalize(source[:, 2])
        angles.append(signed_angle_about_axis(mapped_heading, target[:, 2], target_up))
    if not angles:
        return coarse_rotation.copy(), {"status": "no_valid_torso", "angle_count": 0}
    angle = float(np.median(np.unwrap(np.asarray(angles, dtype=np.float64))))
    bounded = float(np.clip(angle, -math.radians(maximum_deg), math.radians(maximum_deg)))
    axis = normalize(target_frames[0][:, 1])
    corrected = rotation_from_axis_angle(axis, bounded) @ coarse_rotation
    return corrected.astype(np.float32), {
        "status": "ok",
        "raw_residual_deg": math.degrees(angle),
        "bounded_residual_deg": math.degrees(bounded),
        "clipped": bool(abs(angle) > math.radians(maximum_deg)),
        "angle_count": len(angles),
        "angle_median_abs_deviation_deg": float(np.degrees(np.median(np.abs(angles - np.median(angles))))),
    }


def scene_translation_fixed_rotation(
    rotation: np.ndarray,
    initial_translation: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict]:
    translation = np.asarray(initial_translation, dtype=np.float32).copy()
    initial = translation.copy()
    if len(source) < 32 or len(target) < 32:
        return translation, {"status": "too_few_points", "human_root_used": False}
    tree = cKDTree(target)
    iterations = []
    for iteration in range(int(args.translation_iters)):
        transformed = source @ rotation.T + translation[None]
        distance, nearest = tree.query(transformed, k=1, workers=-1)
        alpha = iteration / max(int(args.translation_iters) - 1, 1)
        threshold = (
            (1.0 - alpha) * float(args.translation_max_distance)
            + alpha * float(args.translation_min_distance)
        )
        valid = np.isfinite(distance) & (distance < threshold)
        if int(valid.sum()) < 32:
            break
        ids = np.flatnonzero(valid)
        trim = float(np.quantile(distance[ids], 0.70))
        ids = ids[distance[ids] <= trim]
        if len(ids) < 32:
            break
        residual = target[nearest[ids]] - transformed[ids]
        weights = 1.0 / np.maximum(distance[ids], 0.01)
        delta = np.average(residual, axis=0, weights=weights).astype(np.float32)
        translation += delta
        iterations.append(
            {
                "iteration": iteration,
                "pairs": int(len(ids)),
                "median_distance_m": float(np.median(distance[ids])),
                "delta_translation_m": float(np.linalg.norm(delta)),
            }
        )
    return translation, {
        "status": "ok",
        "human_root_used": False,
        "residual_from_t0_m": float(np.linalg.norm(translation - initial)),
        "iterations": iterations,
    }


def estimate_ground_up(
    prediction: dict,
    view: dict,
    pose: np.ndarray,
    reference_up: np.ndarray,
    seed: int,
) -> tuple[np.ndarray | None, dict]:
    points = camera_points(prediction)
    shape = prediction["pts3d_in_self_view"].shape[-3:-1]
    height, width = int(shape[0]), int(shape[1])
    conf = confidence(prediction, len(points))
    mask = human_mask(view, len(points))
    valid = valid_points(points, conf, mask, True)
    yy = np.repeat(np.arange(height), width)
    valid &= yy >= int(round(0.55 * height))
    if int(valid.sum()) < 64:
        return None, {"status": "too_few_lower_background_points", "count": int(valid.sum())}
    threshold = float(np.quantile(conf[valid], 0.50))
    ids = np.flatnonzero(valid & (conf >= threshold))
    rng = np.random.default_rng(seed)
    if len(ids) > 1600:
        ids = rng.choice(ids, size=1600, replace=False)
    world = transform_points(pose, points[ids]).astype(np.float64)
    reference = normalize(reference_up).astype(np.float64)
    best = None
    for _ in range(128):
        sample = rng.choice(len(world), size=3, replace=False)
        a, b, c = world[sample]
        normal = np.cross(b - a, c - a)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-7:
            continue
        normal /= norm
        if float(np.dot(normal, reference)) < 0.0:
            normal = -normal
        alignment = float(np.dot(normal, reference))
        if alignment < 0.45:
            continue
        distance = np.abs((world - a) @ normal)
        inliers = distance < 0.08
        count = int(inliers.sum())
        if count < 48:
            continue
        residual = float(np.median(distance[inliers]))
        score = count * (0.5 + 0.5 * alignment) - 100.0 * residual
        if best is None or score > best[0]:
            best = (score, normal.copy(), inliers, residual, alignment)
    if best is None:
        return None, {"status": "ransac_failed", "count": int(len(world))}
    _, normal, inliers, residual, alignment = best
    centered = world[inliers] - np.median(world[inliers], axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    refined = vt[-1]
    if float(np.dot(refined, reference)) < 0.0:
        refined = -refined
    refined /= max(float(np.linalg.norm(refined)), 1e-12)
    return refined.astype(np.float32), {
        "status": "ok",
        "count": int(len(world)),
        "inlier_count": int(inliers.sum()),
        "inlier_ratio": float(np.mean(inliers)),
        "residual_m": residual,
        "reference_alignment": alignment,
    }


def aggregate_ground_up(
    predictions: list[dict],
    views: list[dict],
    poses: list[np.ndarray],
    reference_up: np.ndarray,
    seed: int,
) -> tuple[np.ndarray | None, dict]:
    normals, rows = [], []
    for index, (prediction, view, pose) in enumerate(zip(predictions, views, poses)):
        normal, diagnostics = estimate_ground_up(prediction, view, pose, reference_up, seed + index)
        rows.append(diagnostics)
        if normal is not None:
            normals.append(normal)
    if not normals:
        return None, {"status": "no_valid_frame", "frames": rows}
    normal = normalize(np.median(np.stack(normals), axis=0))
    spread = [math.degrees(math.acos(float(np.clip(np.dot(normal, row), -1.0, 1.0)))) for row in normals]
    return normal, {
        "status": "ok",
        "valid_frames": len(normals),
        "spread_deg": float(np.median(spread)),
        "frames": rows,
    }


def gravity_residual(
    coarse_rotation: np.ndarray,
    source_up: np.ndarray | None,
    target_up: np.ndarray | None,
    maximum_deg: float,
) -> tuple[np.ndarray, dict]:
    if source_up is None or target_up is None:
        return coarse_rotation.copy(), {"status": "gravity_unavailable"}
    raw = correct_gravity(coarse_rotation, source_up, target_up) @ coarse_rotation.T
    bounded, magnitude, clipped = bounded_rotation(raw, maximum_deg)
    return (bounded @ coarse_rotation).astype(np.float32), {
        "status": "ok",
        "bounded_residual_deg": magnitude,
        "clipped": clipped,
    }


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def root_prediction(old_humans: list[dict | None], post_offset: int = 1) -> np.ndarray | None:
    roots = [human["root"] for human in old_humans if human is not None]
    if not roots:
        return None
    velocity = np.median(np.diff(np.stack(roots), axis=0), axis=0) if len(roots) >= 2 else np.zeros(3)
    return (roots[-1] + post_offset * velocity).astype(np.float32)


def root_motion_error(transform: np.ndarray, query_human: dict | None, predicted_root: np.ndarray | None) -> float:
    if query_human is None or predicted_root is None:
        return float("nan")
    aligned = transform[:3, :3] @ query_human["root"] + transform[:3, 3]
    return float(np.linalg.norm(aligned - predicted_root))


def add_candidate(
    outputs: dict,
    name: str,
    rotation: np.ndarray,
    initial_translation: np.ndarray,
    clouds: tuple[np.ndarray, np.ndarray],
    context: dict,
    diagnostics: dict,
    resolve_translation: bool,
) -> np.ndarray:
    source, target = clouds
    if resolve_translation:
        translation, solver = scene_translation_fixed_rotation(
            rotation, initial_translation, source, target, context["args"]
        )
    else:
        translation = initial_translation
        solver = {"status": "keep_t0", "human_root_used": False}
    transform = make_transform(rotation, translation)
    outputs[name] = candidate_row(
        transform,
        context["pred_pose0"],
        context["target_pose0"],
        context["old_pose0"],
        context["query_human"],
        context["old_humans"],
        {**diagnostics, "translation_solver": solver},
    )
    return transform


def build_family(
    prefix: str,
    coarse: np.ndarray,
    old_humans: list[dict | None],
    new_humans: list[dict | None],
    predicted_frames: list[np.ndarray] | None,
    ground: dict,
    clouds_1f: tuple[np.ndarray, np.ndarray],
    clouds_3f: tuple[np.ndarray, np.ndarray],
    context: dict,
) -> dict:
    outputs = {}
    rotation0, translation0 = coarse[:3, :3], coarse[:3, 3]
    valid_new = [human for human in new_humans if human is not None]
    valid_old = [human for human in old_humans if human is not None]
    if not valid_new or not valid_old or predicted_frames is None:
        outputs[f"{prefix}_torso_unavailable"] = {"fit_failed": True, "transform": None}
        return outputs

    last_target = [valid_old[-1]["torso"]]
    current_source = [valid_new[0]["torso"]]
    last_rotation, last_diag = yaw_residual(
        rotation0, current_source, last_target, context["args"].max_yaw_residual_deg
    )
    add_candidate(outputs, f"{prefix}_torso_last_1f_keep_t0", last_rotation, translation0, clouds_1f, context, last_diag, False)
    add_candidate(outputs, f"{prefix}_torso_last_1f_resolve_t", last_rotation, translation0, clouds_1f, context, last_diag, True)

    motion_rotation, motion_diag = yaw_residual(
        rotation0,
        current_source,
        predicted_frames[:1],
        context["args"].max_yaw_residual_deg,
    )
    add_candidate(outputs, f"{prefix}_torso_motion_1f_keep_t0", motion_rotation, translation0, clouds_1f, context, motion_diag, False)
    motion_1f = add_candidate(
        outputs, f"{prefix}_torso_motion_1f_resolve_t", motion_rotation, translation0, clouds_1f, context, motion_diag, True
    )

    count = min(3, len(valid_new), len(predicted_frames))
    three_rotation, three_diag = yaw_residual(
        rotation0,
        [human["torso"] for human in valid_new[:count]],
        predicted_frames[:count],
        context["args"].max_yaw_residual_deg,
    )
    three_transform = add_candidate(
        outputs, f"{prefix}_torso_motion_3f_resolve_t", three_rotation, translation0, clouds_3f, context, three_diag, True
    )

    gravity_1f, gravity_diag = gravity_residual(
        rotation0, ground["new_1f"], ground["old"], context["args"].max_gravity_residual_deg
    )
    add_candidate(outputs, f"{prefix}_gravity_1f_resolve_t", gravity_1f, translation0, clouds_1f, context, gravity_diag, True)
    torso_gravity_rotation, torso_gravity_diag = yaw_residual(
        gravity_1f,
        current_source,
        predicted_frames[:1],
        context["args"].max_yaw_residual_deg,
    )
    add_candidate(
        outputs,
        f"{prefix}_torso_motion_gravity_1f_resolve_t",
        torso_gravity_rotation,
        translation0,
        clouds_1f,
        context,
        {"gravity": gravity_diag, "torso": torso_gravity_diag},
        True,
    )

    predicted_root = root_prediction(old_humans)
    coarse_root_error = root_motion_error(coarse, context["query_human"], predicted_root)
    one_root_error = root_motion_error(motion_1f, context["query_human"], predicted_root)
    three_root_error = root_motion_error(three_transform, context["query_human"], predicted_root)
    chosen = motion_1f if (
        not np.isfinite(coarse_root_error)
        or one_root_error <= coarse_root_error + float(context["args"].root_check_margin_m)
    ) else coarse
    outputs[f"{prefix}_torso_motion_1f_root_check"] = candidate_row(
        chosen,
        context["pred_pose0"],
        context["target_pose0"],
        context["old_pose0"],
        context["query_human"],
        context["old_humans"],
        {
            "accepted_torso": bool(chosen is motion_1f),
            "coarse_root_motion_error_m": coarse_root_error,
            "corrected_root_motion_error_m": one_root_error,
        },
    )
    chosen_three = three_transform if (
        not np.isfinite(coarse_root_error)
        or three_root_error <= coarse_root_error + float(context["args"].root_check_margin_m)
    ) else coarse
    outputs[f"{prefix}_torso_motion_3f_root_check"] = candidate_row(
        chosen_three,
        context["pred_pose0"],
        context["target_pose0"],
        context["old_pose0"],
        context["query_human"],
        context["old_humans"],
        {
            "accepted_torso": bool(chosen_three is three_transform),
            "coarse_root_motion_error_m": coarse_root_error,
            "corrected_root_motion_error_m": three_root_error,
        },
    )
    return outputs


def token_array(debug: list[dict], count: int) -> np.ndarray:
    arrays = token_debug_to_arrays(debug)
    token = arrays.get("human_token_out")
    if token is None:
        return np.empty((count, 0), dtype=np.float16)
    token = token[-count:]
    if len(token) < count:
        token = np.concatenate([np.zeros((count - len(token), token.shape[1]), dtype=token.dtype), token], axis=0)
    return token.astype(np.float16)


def run_case(record: dict, human3r, pred_layer, v15_case: dict, args: argparse.Namespace, case_index: int) -> tuple[dict, np.ndarray, np.ndarray]:
    spec = record_spec(record, args)
    device = torch.device(args.device)
    reset_views = configure_views(one_batch(build_dataset([spec], False, args)), device, human3r.mhmr_img_res)[:3]
    old_views_all = configure_views(one_batch(old_a_dataset(spec, args)), device, human3r.mhmr_img_res)
    started = time.perf_counter()
    with torch.no_grad():
        old_predictions_all, _, old_debug = human3r.forward_recurrent_lighter(
            old_views_all, str(device), ret_state=False, use_ttt3r=False, return_token_debug=True
        )
        reset_predictions, _, reset_debug = human3r.forward_recurrent_lighter(
            reset_views, str(device), ret_state=False, use_ttt3r=False, return_token_debug=True
        )
    inference_seconds = time.perf_counter() - started

    history_count = min(int(args.history_frames), len(old_predictions_all))
    old_predictions = old_predictions_all[-history_count:]
    old_views = old_views_all[-history_count:]
    old_poses = [camera_matrix(row) for row in old_predictions]
    new_poses = [camera_matrix(row) for row in reset_predictions]
    old_humans = [predicted_human_summary(prediction, view, pred_layer) for prediction, view in zip(old_predictions, old_views)]
    new_humans = [predicted_human_summary(prediction, view, pred_layer) for prediction, view in zip(reset_predictions, reset_views)]
    predicted_frames, motion_diagnostics = predict_torso_frames(old_humans, len(reset_predictions))

    pred_pose0 = new_poses[0]
    old_pose0 = old_poses[-1]
    gt_pose0 = gt_pose_from_view(reset_views[0]).detach().float().cpu().numpy().astype(np.float32)
    old_gt_pose = gt_pose_from_view(old_views[-1]).detach().float().cpu().numpy().astype(np.float32)
    old_from_raw = old_pose0 @ np.linalg.inv(old_gt_pose)
    target_pose0 = old_from_raw @ gt_pose0
    boundary_gt = target_pose0 @ np.linalg.inv(pred_pose0)
    fixed, fixed_name = fixed_explicit_human3r_gauge(args.candidate_root, record["pattern_id"])
    wide_row = v15_case["windows"]["full_rgb_1p1"]["candidates"]["coarse"]
    wide = np.asarray(wide_row["transform"], dtype=np.float32)

    cloud_1f_full = dense_clouds(
        old_predictions[-3:], reset_predictions[:1], old_views[-3:], reset_views[:1], old_poses[-3:], new_poses[:1],
        int(args.point_samples_per_frame), False,
    )
    cloud_3f_full = dense_clouds(
        old_predictions[-3:], reset_predictions[:3], old_views[-3:], reset_views[:3], old_poses[-3:], new_poses[:3],
        int(args.point_samples_per_frame), False,
    )
    clouds_1f = (cloud_1f_full[0], cloud_1f_full[1])
    clouds_3f = (cloud_3f_full[0], cloud_3f_full[1])

    valid_old = [human for human in old_humans if human is not None]
    valid_new = [human for human in new_humans if human is not None]
    old_reference_up = valid_old[-1]["torso"][:, 1] if valid_old else np.array([0.0, -1.0, 0.0], dtype=np.float32)
    new_reference_up = valid_new[0]["torso"][:, 1] if valid_new else np.array([0.0, -1.0, 0.0], dtype=np.float32)
    old_ground, old_ground_diag = aggregate_ground_up(
        old_predictions[-3:], old_views[-3:], old_poses[-3:], old_reference_up, args.seed + 1000 * case_index
    )
    new_ground_1f, new_ground_1f_diag = aggregate_ground_up(
        reset_predictions[:1], reset_views[:1], new_poses[:1], new_reference_up, args.seed + 1000 * case_index + 100
    )
    new_ground_3f, new_ground_3f_diag = aggregate_ground_up(
        reset_predictions[:3], reset_views[:3], new_poses[:3], new_reference_up, args.seed + 1000 * case_index + 200
    )
    ground = {"old": old_ground, "new_1f": new_ground_1f, "new_3f": new_ground_3f}

    context = {
        "args": args,
        "pred_pose0": pred_pose0,
        "target_pose0": target_pose0,
        "old_pose0": old_pose0,
        "query_human": new_humans[0] if new_humans else None,
        "old_humans": old_humans,
    }
    fixed_candidates = build_family(
        "fixed", fixed, old_humans, new_humans, predicted_frames, ground, clouds_1f, clouds_3f, context
    )
    wide_candidates = build_family(
        "v15", wide, old_humans, new_humans, predicted_frames, ground, clouds_1f, clouds_3f, context
    )

    baselines = {
        "fixed_explicit": candidate_row(
            fixed, pred_pose0, target_pose0, old_pose0, new_humans[0], old_humans, {"name": fixed_name}
        ),
        "v15_coarse": candidate_row(wide, pred_pose0, target_pose0, old_pose0, new_humans[0], old_humans),
        "boundary_oracle": candidate_row(boundary_gt, pred_pose0, target_pose0, old_pose0, new_humans[0], old_humans),
        "hard_reset": candidate_row(np.eye(4, dtype=np.float32), pred_pose0, target_pose0, old_pose0, new_humans[0], old_humans),
    }
    original_continue = v15_case["baselines"]["original_continue"]
    baselines["original_continue"] = original_continue
    case = {
        "case_name": record["pattern_id"],
        "record": record,
        "protocol": {
            "human3r_frozen": True,
            "gt_camera_use": "evaluation only",
            "gt_depth_used": False,
            "human_root_translation_used": False,
            "postcut_frames": 3,
            "precut_history_frames": history_count,
            "max_humans": 1,
            "max_yaw_residual_deg": float(args.max_yaw_residual_deg),
            "max_gravity_residual_deg": float(args.max_gravity_residual_deg),
        },
        "texture_score": texture_score(reset_views[0]),
        "baselines": baselines,
        "fixed_candidates": fixed_candidates,
        "v15_candidates": wide_candidates,
        "motion_diagnostics": motion_diagnostics,
        "ground_diagnostics": {
            "old": old_ground_diag,
            "new_1f": new_ground_1f_diag,
            "new_3f": new_ground_3f_diag,
        },
        "inference_seconds": inference_seconds,
        "peak_gpu_memory_gb": float(torch.cuda.max_memory_allocated(device) / (1024**3)),
        "token_feature_index": case_index,
    }
    return case, token_array(old_debug, history_count), token_array(reset_debug, 3)


def stack_tokens(rows: list[np.ndarray]) -> np.ndarray:
    maximum_frames = max(row.shape[0] for row in rows)
    maximum_dim = max(row.shape[1] for row in rows)
    output = np.zeros((len(rows), maximum_frames, maximum_dim), dtype=np.float16)
    for index, row in enumerate(rows):
        output[index, -row.shape[0] :, : row.shape[1]] = row
    return output


def main() -> None:
    args = parse_args()
    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("V16 Human3R inference requires CUDA")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard index")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"v16_candidates_shard_{args.shard_index:02d}_of_{args.num_shards:02d}.json"
    feature_path = args.output_dir / f"v16_tokens_shard_{args.shard_index:02d}_of_{args.num_shards:02d}.npz"
    if json_path.is_file() and feature_path.is_file() and not args.overwrite:
        print(f">> exists {json_path}")
        return
    records = read_jsonl(args.records)
    if args.sources:
        allowed = set(args.sources)
        records = [record for record in records if str(record["source"]) in allowed]
    selected = [record for index, record in enumerate(records) if index % args.num_shards == args.shard_index]
    if args.max_cases > 0:
        selected = selected[: args.max_cases]
    v15_cases = load_v15_cases(args.v15_candidate_dir)
    missing = [record["pattern_id"] for record in selected if record["pattern_id"] not in v15_cases]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} V15 cases; first={missing[0]}")
    human3r = build_model(args)
    _, pred_layer = build_smpl_models(human3r, torch.device(args.device))
    cases, old_tokens, new_tokens = [], [], []
    started = time.perf_counter()
    for index, record in enumerate(selected):
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))
        case, old_token, new_token = run_case(record, human3r, pred_layer, v15_cases[record["pattern_id"]], args, index)
        case["token_feature_index"] = len(cases)
        cases.append(case)
        old_tokens.append(old_token)
        new_tokens.append(new_token)
        candidate = case["fixed_candidates"].get("fixed_torso_motion_3f_resolve_t", {})
        print(
            f">> [{index + 1}/{len(selected)}] {record['pattern_id']} "
            f"R={candidate.get('camera_rotation_error_deg', float('nan')):.2f} "
            f"mem={case['peak_gpu_memory_gb']:.1f}GB",
            flush=True,
        )
        torch.cuda.empty_cache()
    payload = {
        "experiment": "V16 Explicit-First Human-Aware Torso Geometry Candidates",
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "case_count": len(cases),
        "elapsed_seconds": time.perf_counter() - started,
        "cases": cases,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    np.savez_compressed(
        feature_path,
        case_names=np.asarray([case["case_name"] for case in cases]),
        old_human_token=stack_tokens(old_tokens),
        new_human_token=stack_tokens(new_tokens),
    )
    print(f">> wrote {json_path}", flush=True)
    print(f">> wrote {feature_path}", flush=True)


if __name__ == "__main__":
    main()
