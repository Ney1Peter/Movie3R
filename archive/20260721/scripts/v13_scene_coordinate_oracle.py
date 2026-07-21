#!/usr/bin/env python3
"""V13 stage-1 scene-coordinate pseudo-oracle SE(3)/Sim(3) ladder."""

from __future__ import annotations

import argparse
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

from v10_latent_activation_patching_probe import run_branch  # noqa: E402
from v10_oracle_state_vs_gauge_probe import rotation_error_deg  # noqa: E402
from v11_gauge_neutral_first_write_oracle import fixed_explicit_transform  # noqa: E402
from v11_gauge_neutral_first_write_probe import (  # noqa: E402
    DEFAULT_RECORDS,
    build_dataset,
    build_model,
    configure_views,
    read_jsonl,
    record_spec,
    texture_score,
)
from v12_build_gauge_neutral_teacher_cache import old_a_dataset  # noqa: E402
from v9_learned_stream_alignment_overfit import gt_pose_from_view  # noqa: E402
from v10_latent_activation_patching_probe import camera_matrix, pointmap  # noqa: E402


DEFAULT_OUTPUT = REPO_ROOT / "output" / "v13_world_coordinate_memory" / "stage1_scene_coordinate_oracle"
DEFAULT_CANDIDATES = REPO_ROOT / "output" / "v10_candidate_selection" / "oracle_gt_4source" / "cases"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidate_root", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--data_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data"))
    parser.add_argument("--model_path", type=Path, default=REPO_ROOT / "src" / "human3r_896L.pth")
    parser.add_argument("--device", required=True)
    parser.add_argument("--resolution", type=int, nargs=2, default=(512, 288))
    parser.add_argument("--resize_mode", default="human3r_demo")
    parser.add_argument("--max_post_frames", type=int, default=9)
    parser.add_argument("--warmup_frames", type=int, default=8)
    parser.add_argument("--memory_points_per_frame", type=int, default=2048)
    parser.add_argument("--memory_match_radius", type=float, default=0.50)
    parser.add_argument("--point_counts", type=int, nargs="*", default=(64, 256, 1024, 4096))
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def one_batch(dataset) -> list[dict]:
    return next(iter(torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)))


def camera_points(prediction: dict) -> np.ndarray:
    value = prediction["pts3d_in_self_view"]
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32).reshape(-1, 3)


def point_shape(prediction: dict) -> tuple[int, int]:
    value = prediction["pts3d_in_self_view"]
    shape = tuple(value.shape)
    return int(shape[-3]), int(shape[-2])


def confidence(prediction: dict, count: int) -> np.ndarray:
    value = prediction.get("conf_self")
    if value is None:
        return np.ones(count, dtype=np.float32)
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    return array[:count] if len(array) >= count else np.ones(count, dtype=np.float32)


def human_mask(view: dict, count: int) -> np.ndarray:
    value = view.get("msk", False)
    if isinstance(value, bool):
        return np.zeros(count, dtype=bool)
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value).squeeze().reshape(-1)
    if len(array) != count:
        return np.zeros(count, dtype=bool)
    return array > 0


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def valid_points(points: np.ndarray, conf: np.ndarray, mask: np.ndarray, static_only: bool) -> np.ndarray:
    valid = np.isfinite(points).all(axis=1)
    valid &= points[:, 2] > 0.05
    valid &= points[:, 2] < 50.0
    valid &= np.isfinite(conf)
    if static_only:
        valid &= ~mask
    return valid


def spatial_select(
    ids: np.ndarray,
    conf: np.ndarray,
    height: int,
    width: int,
    count: int,
    mode: str,
    seed: int,
) -> np.ndarray:
    if len(ids) <= count:
        return ids
    if mode == "confidence":
        return ids[np.argsort(conf[ids])[-count:]]
    side = max(2, int(math.ceil(math.sqrt(count))))
    rows, cols = ids // width, ids % width
    row_bin = np.minimum(rows * side // max(height, 1), side - 1)
    col_bin = np.minimum(cols * side // max(width, 1), side - 1)
    chosen = []
    for key in np.unique(row_bin * side + col_bin):
        local = ids[(row_bin * side + col_bin) == key]
        chosen.append(int(local[np.argmax(conf[local])]))
    chosen = np.asarray(chosen, dtype=np.int64)
    if len(chosen) >= count:
        return chosen[np.argsort(conf[chosen])[-count:]]
    remaining = np.setdiff1d(ids, chosen, assume_unique=False)
    generator = np.random.default_rng(seed)
    weights = np.maximum(conf[remaining] - np.nanmin(conf[remaining]), 1e-5)
    weights = weights / weights.sum()
    extra = generator.choice(remaining, size=count - len(chosen), replace=False, p=weights)
    return np.concatenate([chosen, extra])


def weighted_similarity(source: np.ndarray, target: np.ndarray, weight: np.ndarray, allow_scale: bool):
    weight = np.maximum(weight.astype(np.float64), 1e-8)
    weight = weight / weight.sum()
    source64 = source.astype(np.float64)
    target64 = target.astype(np.float64)
    source_mean = (weight[:, None] * source64).sum(axis=0)
    target_mean = (weight[:, None] * target64).sum(axis=0)
    source_centered = source64 - source_mean
    target_centered = target64 - target_mean
    covariance = (weight[:, None] * target_centered).T @ source_centered
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.ones(3, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0:
        sign[-1] = -1.0
    rotation = u @ np.diag(sign) @ vt
    denominator = (weight * np.sum(source_centered**2, axis=1)).sum()
    scale = float(np.dot(singular, sign) / max(denominator, 1e-12)) if allow_scale else 1.0
    translation = target_mean - scale * rotation @ source_mean
    return rotation.astype(np.float32), scale, translation.astype(np.float32)


def robust_fit(source: np.ndarray, target: np.ndarray, weight: np.ndarray, allow_scale: bool) -> dict | None:
    if len(source) < 6:
        return None
    active = np.ones(len(source), dtype=bool)
    rotation = np.eye(3, dtype=np.float32)
    scale = 1.0
    translation = np.zeros(3, dtype=np.float32)
    for _ in range(6):
        if active.sum() < 6:
            break
        rotation, scale, translation = weighted_similarity(
            source[active], target[active], weight[active], allow_scale
        )
        predicted = scale * (source @ rotation.T) + translation
        residual = np.linalg.norm(predicted - target, axis=1)
        median = float(np.median(residual[active]))
        mad = float(np.median(np.abs(residual[active] - median)))
        threshold = min(0.75, max(0.05, median + 2.5 * 1.4826 * mad))
        next_active = residual <= threshold
        if next_active.sum() < 6 or np.array_equal(next_active, active):
            active = next_active if next_active.sum() >= 6 else active
            break
        active = next_active
    predicted = scale * (source @ rotation.T) + translation
    residual = np.linalg.norm(predicted - target, axis=1)
    return {
        "rotation": rotation,
        "scale": float(scale),
        "translation": translation,
        "residual": residual,
        "active": active,
    }


def geometry_diagnostics(
    source: np.ndarray,
    target: np.ndarray,
    pixels: np.ndarray,
    target_ids: np.ndarray,
    height: int,
    width: int,
    fit: dict,
) -> dict:
    centered = target - target.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(target), 1)
    eigenvalues = np.sort(np.maximum(np.linalg.eigvalsh(covariance), 0.0))
    condition = float(eigenvalues[-1] / max(eigenvalues[0], 1e-10))
    bins = 8
    row_bin = np.minimum(pixels[:, 0] * bins // max(height, 1), bins - 1)
    col_bin = np.minimum(pixels[:, 1] * bins // max(width, 1), bins - 1)
    image_coverage = float(len(np.unique(row_bin * bins + col_bin)) / (bins * bins))
    residual = fit["residual"]
    unique_ratio = float(len(np.unique(target_ids)) / max(len(target_ids), 1))
    return {
        "correspondence_count": int(len(source)),
        "fit_residual_mean_m": float(residual.mean()),
        "fit_residual_median_m": float(np.median(residual)),
        "fit_residual_p90_m": float(np.percentile(residual, 90)),
        "inlier_ratio_0_10m": float(np.mean(residual < 0.10)),
        "inlier_ratio_0_20m": float(np.mean(residual < 0.20)),
        "robust_inlier_ratio": float(fit["active"].mean()),
        "image_coverage_8x8": image_coverage,
        "extent_xyz_m": (target.max(axis=0) - target.min(axis=0)).tolist(),
        "covariance_eigenvalues": eigenvalues.tolist(),
        "geometry_condition_number": condition,
        "planarity_ratio": float(eigenvalues[0] / max(eigenvalues.sum(), 1e-10)),
        "unique_target_ratio": unique_ratio,
        "duplicate_geometry_ratio": 1.0 - unique_ratio,
    }


def pose_error(
    fit: dict,
    pred_pose: np.ndarray,
    target_pose: np.ndarray,
) -> dict:
    rotation = fit["rotation"]
    scale = float(fit["scale"])
    translation = fit["translation"]
    aligned_rotation = rotation @ pred_pose[:3, :3]
    aligned_translation = scale * (rotation @ pred_pose[:3, 3]) + translation
    relative = aligned_rotation.T @ target_pose[:3, :3]
    yaw, pitch, roll = np.abs(Rotation.from_matrix(relative).as_euler("zyx", degrees=True))
    delta = aligned_translation - target_pose[:3, 3]
    return {
        "camera_translation_error_m": float(np.linalg.norm(delta)),
        "camera_rotation_error_deg": rotation_error_deg(aligned_rotation, target_pose[:3, :3]),
        "translation_error_xyz_m": np.abs(delta).tolist(),
        "yaw_error_deg": float(yaw),
        "pitch_error_deg": float(pitch),
        "roll_error_deg": float(roll),
        "estimated_scale": scale,
        "scale_log_abs": abs(math.log(max(scale, 1e-8))),
    }


def direct_transform_error(transform: np.ndarray, pred_pose: np.ndarray, target_pose: np.ndarray) -> dict:
    fit = {
        "rotation": transform[:3, :3],
        "scale": 1.0,
        "translation": transform[:3, 3],
    }
    return pose_error(fit, pred_pose, target_pose)


def frame_correspondences(
    reset_prediction: dict,
    teacher_prediction: dict,
    view: dict,
    old_from_raw: np.ndarray,
    memory_tree: cKDTree,
    memory_points: np.ndarray,
    memory_confidence: np.ndarray,
    static_only: bool,
    match_radius: float,
    candidate_limit: int,
    seed: int,
) -> dict:
    source_camera = camera_points(reset_prediction)
    teacher_camera = camera_points(teacher_prediction)
    count = min(len(source_camera), len(teacher_camera))
    source_camera = source_camera[:count]
    teacher_camera = teacher_camera[:count]
    conf = confidence(reset_prediction, count)
    teacher_conf = confidence(teacher_prediction, count)
    mask = human_mask(view, count)
    valid = valid_points(source_camera, conf, mask, static_only)
    valid &= valid_points(teacher_camera, teacher_conf, mask, static_only)
    ids = np.nonzero(valid)[0]
    height, width = point_shape(reset_prediction)
    if len(ids) > candidate_limit:
        half = max(candidate_limit // 2, 1)
        confidence_ids = spatial_select(ids, conf * teacher_conf, height, width, half, "confidence", seed)
        spatial_ids = spatial_select(ids, conf * teacher_conf, height, width, half, "spatial", seed + 1)
        ids = np.unique(np.concatenate([confidence_ids, spatial_ids]))
    pred_pose = camera_matrix(reset_prediction)
    gt_pose = gt_pose_from_view(view).detach().float().cpu().numpy().astype(np.float32)
    source_world = transform_points(pred_pose, source_camera)
    teacher_old = transform_points(old_from_raw @ gt_pose, teacher_camera)
    distances, memory_ids = memory_tree.query(teacher_old[ids], k=1, workers=-1)
    keep = distances <= match_radius
    memory_source_ids = ids[keep]
    matched_memory_ids = memory_ids[keep].astype(np.int64)
    pixels = np.stack([ids // width, ids % width], axis=1).astype(np.int32)
    memory_pixels = np.stack([memory_source_ids // width, memory_source_ids % width], axis=1).astype(np.int32)
    return {
        "height": height,
        "width": width,
        "source_world": source_world,
        "teacher_target": teacher_old,
        "valid_ids": ids,
        "pixels": pixels,
        "weight": np.sqrt(np.maximum(conf * teacher_conf, 1e-6)),
        "memory_source_ids": memory_source_ids,
        "memory_target_ids": matched_memory_ids,
        "memory_target": memory_points[matched_memory_ids],
        "memory_pixels": memory_pixels,
        "memory_weight": np.sqrt(
            np.maximum(conf[memory_source_ids] * teacher_conf[memory_source_ids] * memory_confidence[matched_memory_ids], 1e-6)
        ) * np.exp(-distances[keep] / max(match_radius, 1e-6)),
        "memory_distances": distances,
        "memory_keep": keep,
        "overlap_ratio_0_20m": float(np.mean(distances < 0.20)) if len(distances) else 0.0,
        "overlap_ratio_0_50m": float(np.mean(distances < 0.50)) if len(distances) else 0.0,
    }


def build_memory(old_predictions: list[dict], old_views: list[dict], points_per_frame: int, seed: int):
    points, confidence_rows, frame_ids = [], [], []
    for frame_index, (prediction, view) in enumerate(zip(old_predictions, old_views)):
        camera = camera_points(prediction)
        conf = confidence(prediction, len(camera))
        mask = human_mask(view, len(camera))
        valid = np.nonzero(valid_points(camera, conf, mask, True))[0]
        height, width = point_shape(prediction)
        selected = spatial_select(valid, conf, height, width, points_per_frame, "spatial", seed + frame_index)
        world = transform_points(camera_matrix(prediction), camera[selected])
        points.append(world)
        confidence_rows.append(conf[selected])
        frame_ids.append(np.full(len(selected), frame_index, dtype=np.int32))
    memory_points = np.concatenate(points, axis=0).astype(np.float32)
    memory_confidence = np.concatenate(confidence_rows, axis=0).astype(np.float32)
    memory_frames = np.concatenate(frame_ids, axis=0)
    return memory_points, memory_confidence, memory_frames


def collect_fit_data(
    rows: list[dict],
    mode: str,
    frame_count: int,
    point_count: int,
    selection: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int] | None:
    source_rows, target_rows, weight_rows, pixel_rows, target_id_rows = [], [], [], [], []
    target_offset = 0
    for frame_index, row in enumerate(rows[:frame_count]):
        if mode == "same_view_teacher":
            ids = spatial_select(
                row["valid_ids"], row["weight"], row["height"], row["width"], point_count, selection, seed + frame_index
            )
            if not len(ids):
                continue
            source_rows.append(row["source_world"][ids])
            target_rows.append(row["teacher_target"][ids])
            weight_rows.append(row["weight"][ids])
            pixel_rows.append(np.stack([ids // row["width"], ids % row["width"]], axis=1))
            target_id_rows.append(np.arange(len(ids), dtype=np.int64) + target_offset)
            target_offset += len(ids)
        else:
            ids = np.arange(len(row["memory_source_ids"]), dtype=np.int64)
            source_pixel_ids = row["memory_source_ids"]
            local_conf = row["memory_weight"]
            if len(ids) > point_count:
                dense_conf = np.zeros(row["height"] * row["width"], dtype=np.float32)
                dense_conf[source_pixel_ids] = local_conf
                chosen_source_pixels = spatial_select(
                    source_pixel_ids,
                    dense_conf,
                    row["height"], row["width"], point_count, selection, seed + frame_index
                )
                position = {int(value): idx for idx, value in enumerate(source_pixel_ids)}
                ids = np.asarray([position[int(value)] for value in chosen_source_pixels if int(value) in position], dtype=np.int64)
            if not len(ids):
                continue
            source_rows.append(row["source_world"][source_pixel_ids[ids]])
            if mode == "history_coverage_teacher":
                target_rows.append(row["teacher_target"][source_pixel_ids[ids]])
            else:
                target_rows.append(row["memory_target"][ids])
            weight_rows.append(row["memory_weight"][ids])
            pixel_rows.append(row["memory_pixels"][ids])
            if mode == "history_coverage_teacher":
                target_id_rows.append(np.arange(len(ids), dtype=np.int64) + target_offset)
                target_offset += len(ids)
            else:
                target_id_rows.append(row["memory_target_ids"][ids])
    if not source_rows:
        return None
    return (
        np.concatenate(source_rows),
        np.concatenate(target_rows),
        np.concatenate(weight_rows),
        np.concatenate(pixel_rows),
        np.concatenate(target_id_rows),
        rows[0]["height"],
        rows[0]["width"],
    )


def run_case(record: dict, model, args: argparse.Namespace, device: torch.device, case_index: int) -> dict:
    spec = record_spec(record, args)
    reset_views = configure_views(one_batch(build_dataset([spec], False, args)), device, model.mhmr_img_res)
    teacher_views = configure_views(one_batch(build_dataset([spec], True, args)), device, model.mhmr_img_res)
    old_views = configure_views(one_batch(old_a_dataset(spec, args)), device, model.mhmr_img_res)
    reset_predictions, _, reset_seconds, _ = run_branch(model, reset_views, device, 0, capture=False)
    teacher_predictions, _, teacher_seconds, _ = run_branch(
        model, teacher_views, device, spec["warmup_count"], capture=False
    )
    old_predictions, _, memory_seconds, _ = run_branch(
        model, old_views, device, len(old_views) - 1, capture=False
    )
    teacher_post = teacher_predictions[spec["warmup_count"] :]
    pred_pose0 = camera_matrix(reset_predictions[0])
    gt_pose0 = gt_pose_from_view(reset_views[0]).detach().float().cpu().numpy().astype(np.float32)
    old_pred_pose = camera_matrix(old_predictions[-1])
    old_gt_pose = gt_pose_from_view(old_views[-1]).detach().float().cpu().numpy().astype(np.float32)
    old_from_raw = old_pred_pose @ np.linalg.inv(old_gt_pose)
    target_pose0 = old_from_raw @ gt_pose0
    boundary_gt = old_from_raw @ gt_pose0 @ np.linalg.inv(pred_pose0)
    explicit_raw, explicit_name = fixed_explicit_transform(args.candidate_root, record["pattern_id"])
    explicit_old = old_from_raw @ explicit_raw
    memory_points, memory_confidence, memory_frames = build_memory(
        old_predictions, old_views, args.memory_points_per_frame, args.seed + case_index * 101
    )
    memory_tree = cKDTree(memory_points)
    variants = {}
    correspondence_meta = {}
    for static_only in (False, True):
        key = "static" if static_only else "all"
        rows = [
            frame_correspondences(
                reset_predictions[offset],
                teacher_post[offset],
                reset_views[offset],
                old_from_raw,
                memory_tree,
                memory_points,
                memory_confidence,
                static_only,
                args.memory_match_radius,
                max(args.point_counts) * 4,
                args.seed + case_index * 1009 + offset,
            )
            for offset in range(min(3, len(reset_predictions), len(teacher_post)))
        ]
        correspondence_meta[key] = {
            "frame_overlap_ratio_0_20m": [row["overlap_ratio_0_20m"] for row in rows],
            "frame_overlap_ratio_0_50m": [row["overlap_ratio_0_50m"] for row in rows],
            "frame_valid_points": [int(len(row["valid_ids"])) for row in rows],
            "frame_memory_matches": [int(len(row["memory_source_ids"])) for row in rows],
        }
        for mode in ("same_view_teacher", "history_coverage_teacher", "history_memory"):
            for frame_count in (1, 3):
                for point_count in args.point_counts:
                    for selection in ("confidence", "spatial"):
                        data = collect_fit_data(
                            rows,
                            mode,
                            frame_count,
                            int(point_count),
                            selection,
                            args.seed + case_index * 1009 + point_count,
                        )
                        if data is None:
                            continue
                        source, target, weights, pixels, target_ids, height, width = data
                        for transform_type, allow_scale in (("se3", False), ("sim3", True)):
                            fit = robust_fit(source, target, weights, allow_scale)
                            if fit is None:
                                continue
                            name = f"{mode}_{key}_{frame_count}f_{point_count}_{selection}_{transform_type}"
                            variants[name] = {
                                **pose_error(fit, pred_pose0, target_pose0),
                                **geometry_diagnostics(source, target, pixels, target_ids, height, width, fit),
                                "correspondence_mode": mode,
                                "static_only": static_only,
                                "frame_count": frame_count,
                                "point_budget_per_frame": int(point_count),
                                "selection": selection,
                                "transform_type": transform_type,
                            }
    baselines = {
        "hard_reset_no_alignment": direct_transform_error(np.eye(4, dtype=np.float32), pred_pose0, target_pose0),
        "fixed_explicit": {
            **direct_transform_error(explicit_old, pred_pose0, target_pose0),
            "name": explicit_name,
        },
        "boundary_oracle": direct_transform_error(boundary_gt, pred_pose0, target_pose0),
    }
    output = {
        "case_name": record["pattern_id"],
        "record": record,
        "oracle_type": "offline_teacher_pseudo_oracle",
        "true_scene_coordinate_oracle_available": False,
        "memory": {
            "frame_count": len(old_predictions),
            "point_count": int(len(memory_points)),
            "points_per_frame_limit": int(args.memory_points_per_frame),
            "source_frame_histogram": np.bincount(memory_frames, minlength=len(old_predictions)).tolist(),
        },
        "correspondence": correspondence_meta,
        "texture_score": texture_score(reset_views[0]),
        "baselines": baselines,
        "variants": variants,
        "timing_seconds": {
            "reset": reset_seconds,
            "same_camera_teacher": teacher_seconds,
            "historical_memory_build": memory_seconds,
        },
        "coordinate_convention": {
            "camera_pose": "camera_to_world",
            "fit_source": "fresh Human3R world gauge",
            "fit_target": "pre-cut Human3R world gauge",
            "gt_only_for": "oracle correspondence and final evaluation",
        },
    }
    del reset_views, teacher_views, old_views, reset_predictions, teacher_predictions, old_predictions
    torch.cuda.empty_cache()
    return output


def main() -> None:
    args = parse_args()
    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("V13 scene-coordinate Oracle requires CUDA Human3R inference")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard index")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"stage1_shard_{args.shard_index:02d}_of_{args.num_shards:02d}.json"
    if output.is_file() and not args.overwrite:
        print(f">> exists {output}")
        return
    records = read_jsonl(args.records)
    selected = [row for index, row in enumerate(records) if index % args.num_shards == args.shard_index]
    if args.max_cases > 0:
        selected = selected[: args.max_cases]
    model = build_model(args)
    device = torch.device(args.device)
    cases = []
    started = time.perf_counter()
    for index, record in enumerate(selected):
        case = run_case(record, model, args, device, index)
        cases.append(case)
        canonical = case["variants"].get("history_memory_static_3f_1024_spatial_se3", {})
        print(
            f">> [{index + 1}/{len(selected)}] {record['pattern_id']} "
            f"T={canonical.get('camera_translation_error_m', float('nan')):.3f} "
            f"R={canonical.get('camera_rotation_error_deg', float('nan')):.2f}",
            flush=True,
        )
    report = {
        "experiment": "V13 Stage-1 Scene-Coordinate Oracle",
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "case_count": len(cases),
        "elapsed_seconds": time.perf_counter() - started,
        "true_oracle_status": "unavailable_in_current_180_cut_data",
        "cases": cases,
    }
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    print(f">> wrote {output}", flush=True)


if __name__ == "__main__":
    main()
