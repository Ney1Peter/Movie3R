#!/usr/bin/env python3
"""Generate V14 depth-free streaming World-Memory candidates and diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from v10_latent_activation_patching_probe import camera_matrix  # noqa: E402
from v10_oracle_state_vs_gauge_probe import rotation_error_deg  # noqa: E402
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
from v13_frozen_world_memory_probe import (  # noqa: E402
    capture_descriptors,
    descriptor_table,
    normalize_rows,
    ransac_fit,
)
from v13_scene_coordinate_oracle import (  # noqa: E402
    camera_points,
    confidence,
    direct_transform_error,
    human_mask,
    pose_error,
    robust_fit,
    transform_points,
    valid_points,
)
from v9_learned_stream_alignment_overfit import gt_pose_from_view  # noqa: E402


DEFAULT_OUTPUT = REPO_ROOT / "output" / "v14_selective_world_memory" / "candidate_cache"
DEFAULT_CANDIDATES = REPO_ROOT / "output" / "v10_candidate_selection" / "oracle_gt_4source" / "cases"
ANCHOR_STRATEGIES = (
    "confidence",
    "spatial",
    "temporal",
    "confidence_spatial",
    "temporal_spatial",
    "temporal_spatial_static",
)


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
    parser.add_argument("--max_post_frames", type=int, default=3)
    parser.add_argument("--warmup_frames", type=int, default=8)
    parser.add_argument("--patch_samples", type=int, default=128)
    parser.add_argument("--memory_capacity", type=int, default=256)
    parser.add_argument("--voxel_size", type=float, default=0.20)
    parser.add_argument("--ransac_steps", type=int, default=64)
    parser.add_argument(
        "--descriptors",
        nargs="*",
        default=("dino_mhmr",),
    )
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def one_batch(dataset) -> list[dict]:
    return next(iter(torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)))


def fixed_explicit_human3r_gauge(root: Path, case_name: str) -> tuple[np.ndarray, str]:
    payload = json.loads((root / case_name / "case_metrics.json").read_text(encoding="utf-8"))
    transform = np.asarray(payload["fixed_explicit"]["transform"], dtype=np.float32)
    return transform, str(payload["fixed_explicit_name"])


def point_shape(prediction: dict) -> tuple[int, int]:
    shape = tuple(prediction["pts3d_in_self_view"].shape)
    return int(shape[-3]), int(shape[-2])


def patch_pixels(frame_data: dict, height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.asarray(frame_data["positions"], dtype=np.float32)
    patch_ids = np.asarray(frame_data["patch_ids"], dtype=np.int64)
    selected = positions[patch_ids]
    y_extent = max(float(positions[:, 0].max() + 1), 1.0)
    x_extent = max(float(positions[:, 1].max() + 1), 1.0)
    yy = np.clip(((selected[:, 0] + 0.5) / y_extent * height).astype(np.int64), 0, height - 1)
    xx = np.clip(((selected[:, 1] + 0.5) / x_extent * width).astype(np.int64), 0, width - 1)
    return yy, xx, yy * width + xx


def sampled_normals(points: np.ndarray, pixel_ids: np.ndarray, height: int, width: int, stride: int = 4) -> np.ndarray:
    grid = points.reshape(height, width, 3)
    yy, xx = pixel_ids // width, pixel_ids % width
    y0, y1 = np.clip(yy - stride, 0, height - 1), np.clip(yy + stride, 0, height - 1)
    x0, x1 = np.clip(xx - stride, 0, width - 1), np.clip(xx + stride, 0, width - 1)
    dx = grid[yy, x1] - grid[yy, x0]
    dy = grid[y1, xx] - grid[y0, xx]
    normal = np.cross(dx, dy)
    norm = np.linalg.norm(normal, axis=1, keepdims=True)
    valid = np.isfinite(normal).all(axis=1) & (norm[:, 0] > 1e-6)
    output = np.zeros_like(normal, dtype=np.float32)
    output[valid] = normal[valid] / norm[valid]
    return output


def frame_patches(prediction: dict, view: dict, frame_data: dict, descriptor_names: tuple[str, ...]) -> dict:
    points = camera_points(prediction)
    height, width = point_shape(prediction)
    yy, xx, pixel_ids = patch_pixels(frame_data, height, width)
    conf = confidence(prediction, len(points))[pixel_ids]
    mask = human_mask(view, len(points))[pixel_ids]
    camera = points[pixel_ids]
    valid = valid_points(camera, conf, mask, False)
    pose = camera_matrix(prediction)
    normal_camera = sampled_normals(points, pixel_ids, height, width)
    normal_world = normal_camera @ pose[:3, :3].T
    normal_valid = np.linalg.norm(normal_world, axis=1) > 0.5
    table = descriptor_table(frame_data)
    descriptors = {
        name: np.asarray(table[name], dtype=np.float32)[valid]
        for name in descriptor_names
        if name in table
    }
    edge = np.minimum.reduce([yy, height - 1 - yy, xx, width - 1 - xx]).astype(np.float32)
    edge /= max(float(min(height, width)) * 0.5, 1.0)
    return {
        "world": transform_points(pose, camera[valid]),
        "camera": camera[valid],
        "confidence": conf[valid],
        "static": (~mask[valid]).astype(np.float32),
        "normal": normal_world[valid],
        "normal_valid": normal_valid[valid],
        "pixels": np.stack([yy[valid], xx[valid]], axis=1),
        "edge": np.clip(edge[valid], 0.0, 1.0),
        "height": height,
        "width": width,
        "descriptors": descriptors,
    }


def human_summary(prediction: dict, view: dict, pred_layer) -> dict | None:
    human = predicted_human(prediction, view["camera_intrinsics"], pred_layer)
    if human is None:
        return None
    pose = camera_matrix(prediction)
    return {
        "root": transform_points(pose, human["root"][None])[0],
        "torso": (pose[:3, :3] @ human["torso"]).astype(np.float32),
    }


def aligned_mean_normal(normals: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, float]:
    rows = normals[valid]
    if not len(rows):
        return np.zeros(3, dtype=np.float32), 0.0
    reference = rows[0]
    rows = rows.copy()
    rows[(rows @ reference) < 0] *= -1.0
    mean = rows.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm < 1e-6:
        return np.zeros(3, dtype=np.float32), 0.0
    mean /= norm
    stability = float(np.mean(np.abs(rows @ mean)))
    return mean.astype(np.float32), stability


def build_temporal_memory(frames: list[dict], descriptor_names: tuple[str, ...], voxel_size: float) -> dict:
    groups: dict[tuple[int, int, int], list[tuple[int, int]]] = defaultdict(list)
    for frame_index, frame in enumerate(frames):
        voxels = np.floor(frame["world"] / float(voxel_size)).astype(np.int64)
        for point_index, voxel in enumerate(voxels):
            groups[tuple(int(value) for value in voxel)].append((frame_index, point_index))
    rows = []
    for observations in groups.values():
        world = np.stack([frames[f]["world"][i] for f, i in observations])
        conf = np.asarray([frames[f]["confidence"][i] for f, i in observations], dtype=np.float32)
        static = np.asarray([frames[f]["static"][i] for f, i in observations], dtype=np.float32)
        edge = np.asarray([frames[f]["edge"][i] for f, i in observations], dtype=np.float32)
        normals = np.stack([frames[f]["normal"][i] for f, i in observations])
        normal_valid = np.asarray([frames[f]["normal_valid"][i] for f, i in observations], dtype=bool)
        normal, normal_stability = aligned_mean_normal(normals, normal_valid)
        frame_presence = np.zeros(len(frames), dtype=bool)
        frame_presence[[f for f, _ in observations]] = True
        descriptor_mean, descriptor_variance = {}, {}
        for name in descriptor_names:
            values = np.stack([frames[f]["descriptors"][name][i] for f, i in observations])
            values = normalize_rows(values)
            mean = values.mean(axis=0)
            mean /= max(float(np.linalg.norm(mean)), 1e-8)
            descriptor_mean[name] = mean.astype(np.float32)
            descriptor_variance[name] = float(np.mean(1.0 - values @ mean))
        center = world.mean(axis=0)
        rows.append(
            {
                "world": center.astype(np.float32),
                "xyz_variance": float(np.mean(np.sum((world - center) ** 2, axis=1))),
                "confidence": float(conf.mean()),
                "static_rate": float(static.mean()),
                "edge_score": float(edge.mean()),
                "observation_count": int(frame_presence.sum()),
                "normal": normal,
                "normal_stability": normal_stability,
                "frame_presence": frame_presence,
                "descriptors": descriptor_mean,
                "descriptor_variance": descriptor_variance,
            }
        )
    return {
        "world": np.stack([row["world"] for row in rows]),
        "xyz_variance": np.asarray([row["xyz_variance"] for row in rows], dtype=np.float32),
        "confidence": np.asarray([row["confidence"] for row in rows], dtype=np.float32),
        "static_rate": np.asarray([row["static_rate"] for row in rows], dtype=np.float32),
        "edge_score": np.asarray([row["edge_score"] for row in rows], dtype=np.float32),
        "observation_count": np.asarray([row["observation_count"] for row in rows], dtype=np.float32),
        "normal": np.stack([row["normal"] for row in rows]),
        "normal_stability": np.asarray([row["normal_stability"] for row in rows], dtype=np.float32),
        "frame_presence": np.stack([row["frame_presence"] for row in rows]),
        "descriptors": {
            name: np.stack([row["descriptors"][name] for row in rows]) for name in descriptor_names
        },
        "descriptor_variance": {
            name: np.asarray([row["descriptor_variance"][name] for row in rows], dtype=np.float32)
            for name in descriptor_names
        },
    }


def robust_unit(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    low, high = np.percentile(value, [5, 95]) if len(value) > 1 else (float(value[0]), float(value[0]) + 1.0)
    return np.clip((value - low) / max(float(high - low), 1e-6), 0.0, 1.0)


def spatial_select_3d(world: np.ndarray, score: np.ndarray, count: int) -> np.ndarray:
    if len(world) <= count:
        return np.arange(len(world), dtype=np.int64)
    minimum, maximum = world.min(axis=0), world.max(axis=0)
    unit = (world - minimum) / np.maximum(maximum - minimum, 1e-6)
    side = max(2, int(math.ceil(count ** (1.0 / 3.0))))
    bins = np.minimum((unit * side).astype(np.int64), side - 1)
    keys = bins[:, 0] * side * side + bins[:, 1] * side + bins[:, 2]
    selected = []
    for key in np.unique(keys):
        ids = np.flatnonzero(keys == key)
        selected.append(int(ids[np.argmax(score[ids])]))
    selected = np.asarray(selected, dtype=np.int64)
    if len(selected) > count:
        return selected[np.argsort(score[selected])[-count:]]
    mask = np.ones(len(world), dtype=bool)
    mask[selected] = False
    remaining = np.flatnonzero(mask)
    extra = remaining[np.argsort(score[remaining])[-(count - len(selected)) :]]
    return np.concatenate([selected, extra])


def select_anchors(memory: dict, descriptor: str, strategy: str, capacity: int) -> np.ndarray:
    conf = robust_unit(np.log1p(np.maximum(memory["confidence"], 0.0)))
    observations = robust_unit(memory["observation_count"])
    xyz_stability = np.exp(-np.sqrt(np.maximum(memory["xyz_variance"], 0.0)) / 0.10)
    descriptor_stability = np.exp(-memory["descriptor_variance"][descriptor] / 0.10)
    temporal = observations * xyz_stability * descriptor_stability * np.maximum(memory["normal_stability"], 0.25)
    if strategy == "confidence":
        score, spatial = conf, False
    elif strategy == "spatial":
        score, spatial = np.ones_like(conf), True
    elif strategy == "temporal":
        score, spatial = temporal, False
    elif strategy == "confidence_spatial":
        score, spatial = conf, True
    elif strategy == "temporal_spatial":
        score, spatial = temporal, True
    elif strategy == "temporal_spatial_static":
        score = temporal * np.maximum(memory["static_rate"], 0.05) ** 2
        score *= np.maximum(memory["edge_score"], 0.10)
        spatial = True
    else:
        raise KeyError(strategy)
    if strategy == "temporal_spatial_static":
        eligible = np.flatnonzero(memory["static_rate"] >= 0.80)
        if len(eligible) >= 16:
            local = spatial_select_3d(memory["world"][eligible], score[eligible], min(capacity, len(eligible)))
            return eligible[local]
    if spatial:
        return spatial_select_3d(memory["world"], score, min(capacity, len(score)))
    return np.argsort(score)[-min(capacity, len(score)) :].astype(np.int64)


def global_descriptor(frame: dict, descriptor: str) -> np.ndarray:
    rows = normalize_rows(frame["descriptors"][descriptor])
    pooled = rows.mean(axis=0)
    return pooled / max(float(np.linalg.norm(pooled)), 1e-8)


def keyframe_order(query_frames: list[dict], old_frames: list[dict], descriptor: str) -> tuple[list[int], list[float]]:
    query = np.stack([global_descriptor(frame, descriptor) for frame in query_frames]).mean(axis=0)
    query /= max(float(np.linalg.norm(query)), 1e-8)
    old = np.stack([global_descriptor(frame, descriptor) for frame in old_frames])
    scores = old @ query
    return np.argsort(-scores).astype(int).tolist(), scores.astype(float).tolist()


def covariance_diagnostics(points: np.ndarray) -> dict:
    if len(points) < 3:
        return {
            "extent_xyz_m": [0.0, 0.0, 0.0],
            "condition_number": float("inf"),
            "planarity_ratio": 0.0,
            "linearity_ratio": 0.0,
            "volume_proxy": 0.0,
        }
    centered = points - points.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(points), 1)
    eigen = np.sort(np.maximum(np.linalg.eigvalsh(covariance), 0.0))
    return {
        "extent_xyz_m": (points.max(axis=0) - points.min(axis=0)).tolist(),
        "condition_number": float(eigen[-1] / max(eigen[0], 1e-10)),
        "planarity_ratio": float(eigen[0] / max(eigen.sum(), 1e-10)),
        "linearity_ratio": float(eigen[1] / max(eigen[-1], 1e-10)),
        "volume_proxy": float(np.prod(np.sqrt(np.maximum(eigen, 0.0)))),
    }


def transform_difference(first: dict | None, second: dict | None) -> tuple[float, float]:
    if first is None or second is None:
        return float("nan"), float("nan")
    translation = float(np.linalg.norm(first["translation"] - second["translation"]))
    rotation = rotation_error_deg(first["rotation"], second["rotation"])
    return translation, rotation


def human_diagnostics(transform: np.ndarray | None, query_human: dict | None, old_humans: list[dict | None]) -> dict:
    valid_old = [human for human in old_humans if human is not None]
    if transform is None or query_human is None or not valid_old:
        return {
            "human_root_jump_m": float("nan"),
            "human_torso_jump_deg": float("nan"),
            "precut_human_speed_m": float("nan"),
        }
    old = valid_old[-1]
    root = transform_points(transform, query_human["root"][None])[0]
    torso = transform[:3, :3] @ query_human["torso"]
    speed = float("nan")
    if len(valid_old) >= 2:
        speed = float(np.linalg.norm(valid_old[-1]["root"] - valid_old[-2]["root"]))
    return {
        "human_root_jump_m": float(np.linalg.norm(root - old["root"])),
        "human_torso_jump_deg": rotation_error_deg(torso, old["torso"]),
        "precut_human_speed_m": speed,
    }


def fit_candidate(
    query_frames: list[dict],
    memory: dict,
    anchor_ids: np.ndarray,
    keyframes: list[int],
    descriptor: str,
    pred_pose: np.ndarray,
    target_pose: np.ndarray,
    ransac_steps: int,
    seed: int,
    device: torch.device,
    query_human: dict | None,
    old_humans: list[dict | None],
) -> dict:
    query_source = np.concatenate([frame["world"] for frame in query_frames], axis=0)
    query_conf = np.concatenate([frame["confidence"] for frame in query_frames], axis=0)
    query_static = np.concatenate([frame["static"] for frame in query_frames], axis=0)
    query_normal = np.concatenate([frame["normal"] for frame in query_frames], axis=0)
    query_normal_valid = np.concatenate([frame["normal_valid"] for frame in query_frames], axis=0)
    query_pixels = np.concatenate([frame["pixels"] for frame in query_frames], axis=0)
    query_frame_ids = np.concatenate(
        [np.full(len(frame["world"]), index, dtype=np.int32) for index, frame in enumerate(query_frames)]
    )
    query_desc = np.concatenate([frame["descriptors"][descriptor] for frame in query_frames], axis=0)
    allowed = memory["frame_presence"][anchor_ids][:, keyframes].any(axis=1)
    selected = anchor_ids[allowed]
    if len(query_source) < 8 or len(selected) < 8:
        return {"fit_failed": True, "correspondence_count": 0, "transform": None}
    query_gpu = torch.as_tensor(normalize_rows(query_desc), dtype=torch.float32, device=device)
    memory_gpu = torch.as_tensor(memory["descriptors"][descriptor][selected], dtype=torch.float32, device=device)
    similarity_gpu = query_gpu @ memory_gpu.T
    top_count = min(2, similarity_gpu.shape[1])
    values, ids = torch.topk(similarity_gpu, k=top_count, dim=1)
    best = ids[:, 0].detach().cpu().numpy()
    best_value = values[:, 0].detach().cpu().numpy()
    second_value = values[:, 1].detach().cpu().numpy() if top_count > 1 else best_value
    margin = best_value - second_value
    reverse = torch.argmax(similarity_gpu, dim=0).detach().cpu().numpy()
    mutual = reverse[best] == np.arange(len(best))
    candidate = np.flatnonzero(mutual)
    if len(candidate) < 16:
        candidate = np.argsort(margin)[-min(max(32, len(candidate)), len(margin)) :]
    if len(candidate) > 384:
        candidate = candidate[np.argsort(margin[candidate])[-384:]]
    target_ids = selected[best[candidate]]
    source = query_source[candidate]
    target = memory["world"][target_ids]
    similarity = best_value[candidate]
    match_margin = margin[candidate]
    weight = np.sqrt(np.maximum(query_conf[candidate] * memory["confidence"][target_ids], 1e-6))
    weight *= np.maximum(similarity + 1.0, 1e-3) * np.maximum(match_margin + 0.02, 0.01)
    fit = ransac_fit(source, target, weight, ransac_steps, seed, device)
    if fit is None:
        return {"fit_failed": True, "correspondence_count": int(len(source)), "transform": None}
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = fit["rotation"]
    transform[:3, 3] = fit["translation"]
    residual = fit["residual"]
    bins = 8
    height, width = query_frames[0]["height"], query_frames[0]["width"]
    row_bin = np.minimum(query_pixels[candidate, 0] * bins // max(height, 1), bins - 1)
    col_bin = np.minimum(query_pixels[candidate, 1] * bins // max(width, 1), bins - 1)
    image_keys = query_frame_ids[candidate] * bins * bins + row_bin * bins + col_bin
    source_normal = query_normal[candidate]
    source_normal_valid = query_normal_valid[candidate]
    target_normal = memory["normal"][target_ids]
    target_normal_valid = np.linalg.norm(target_normal, axis=1) > 0.5
    normal_valid = source_normal_valid & target_normal_valid
    normal_conflict = float("nan")
    if normal_valid.any():
        mapped = source_normal[normal_valid] @ fit["rotation"].T
        cosine = np.abs(np.sum(mapped * target_normal[normal_valid], axis=1))
        normal_conflict = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))).mean())
    return {
        **pose_error(fit, pred_pose, target_pose),
        **human_diagnostics(transform, query_human, old_humans),
        "fit_failed": False,
        "transform": transform.tolist(),
        "correspondence_count": int(len(source)),
        "mutual_match_count": int(mutual.sum()),
        "fit_residual_mean_m": float(residual.mean()),
        "fit_residual_median_m": float(np.median(residual)),
        "fit_residual_p90_m": float(np.percentile(residual, 90)),
        "inlier_ratio_0_10m": float(np.mean(residual < 0.10)),
        "inlier_ratio_0_20m": float(np.mean(residual < 0.20)),
        "robust_inlier_ratio": float(fit["active"].mean()),
        "image_coverage_8x8": float(len(np.unique(image_keys)) / (len(query_frames) * bins * bins)),
        "mean_cosine": float(similarity.mean()),
        "mean_margin": float(match_margin.mean()),
        "query_confidence_mean": float(query_conf[candidate].mean()),
        "query_static_rate": float(query_static[candidate].mean()),
        "anchor_confidence_mean": float(memory["confidence"][target_ids].mean()),
        "anchor_observation_count_mean": float(memory["observation_count"][target_ids].mean()),
        "anchor_xyz_variance_mean": float(memory["xyz_variance"][target_ids].mean()),
        "anchor_descriptor_variance_mean": float(memory["descriptor_variance"][descriptor][target_ids].mean()),
        "anchor_normal_stability_mean": float(memory["normal_stability"][target_ids].mean()),
        "anchor_static_rate_mean": float(memory["static_rate"][target_ids].mean()),
        "anchor_edge_score_mean": float(memory["edge_score"][target_ids].mean()),
        "normal_conflict_deg": normal_conflict,
        "source_geometry": covariance_diagnostics(source),
        "target_geometry": covariance_diagnostics(target),
    }


def fit_icp_candidate(
    query_frames: list[dict],
    memory: dict,
    anchor_ids: np.ndarray,
    keyframes: list[int],
    initial_transform: np.ndarray,
    pred_pose: np.ndarray,
    target_pose: np.ndarray,
    device: torch.device,
    query_human: dict | None,
    old_humans: list[dict | None],
) -> dict:
    query_source_all = np.concatenate([frame["world"] for frame in query_frames], axis=0)
    query_conf_all = np.concatenate([frame["confidence"] for frame in query_frames], axis=0)
    query_static_all = np.concatenate([frame["static"] for frame in query_frames], axis=0)
    query_normal_all = np.concatenate([frame["normal"] for frame in query_frames], axis=0)
    query_normal_valid_all = np.concatenate([frame["normal_valid"] for frame in query_frames], axis=0)
    query_pixels_all = np.concatenate([frame["pixels"] for frame in query_frames], axis=0)
    query_frame_ids_all = np.concatenate(
        [np.full(len(frame["world"]), index, dtype=np.int32) for index, frame in enumerate(query_frames)]
    )
    source_ids = np.flatnonzero(query_static_all > 0.5)
    allowed = memory["frame_presence"][anchor_ids][:, keyframes].any(axis=1)
    selected = anchor_ids[allowed]
    if len(source_ids) < 12 or len(selected) < 12:
        return {"fit_failed": True, "correspondence_count": 0, "transform": None}
    source = query_source_all[source_ids]
    source_gpu = torch.as_tensor(source, dtype=torch.float32, device=device)
    target_all = memory["world"][selected]
    target_gpu = torch.as_tensor(target_all, dtype=torch.float32, device=device)
    current = initial_transform.astype(np.float32).copy()
    schedule = (1.00, 0.80, 0.60, 0.45, 0.35, 0.28, 0.22, 0.18)
    iterations = []
    for iteration, threshold in enumerate(schedule):
        transformed = torch.as_tensor(transform_points(current, source), dtype=torch.float32, device=device)
        distance_matrix = torch.cdist(transformed, target_gpu)
        distance, nearest = torch.min(distance_matrix, dim=1)
        keep = distance < float(threshold)
        if int(keep.sum().item()) < 12:
            break
        keep_np = keep.detach().cpu().numpy()
        nearest_np = nearest.detach().cpu().numpy()
        current_source = transformed.detach().cpu().numpy()[keep_np]
        current_target = target_all[nearest_np[keep_np]]
        current_weight = np.sqrt(
            np.maximum(
                query_conf_all[source_ids[keep_np]] * memory["confidence"][selected[nearest_np[keep_np]]],
                1e-6,
            )
        )
        delta = robust_fit(current_source, current_target, current_weight, False)
        if delta is None:
            break
        delta_translation = float(np.linalg.norm(delta["translation"]))
        delta_rotation = rotation_error_deg(delta["rotation"], np.eye(3, dtype=np.float32))
        if delta_translation > 0.75 or delta_rotation > 20.0:
            break
        update = np.eye(4, dtype=np.float32)
        update[:3, :3] = delta["rotation"]
        update[:3, 3] = delta["translation"]
        current = update @ current
        iterations.append(
            {
                "iteration": iteration,
                "threshold_m": float(threshold),
                "pair_count": int(keep_np.sum()),
                "median_distance_m": float(np.median(distance.detach().cpu().numpy()[keep_np])),
                "delta_translation_m": delta_translation,
                "delta_rotation_deg": delta_rotation,
            }
        )
    transformed = torch.as_tensor(transform_points(current, source), dtype=torch.float32, device=device)
    distance_matrix = torch.cdist(transformed, target_gpu)
    distance, nearest = torch.min(distance_matrix, dim=1)
    keep = distance < 0.30
    if int(keep.sum().item()) < 12:
        return {
            "fit_failed": True,
            "correspondence_count": int(keep.sum().item()),
            "transform": None,
            "icp_iterations": iterations,
        }
    keep_np = keep.detach().cpu().numpy()
    nearest_np = nearest.detach().cpu().numpy()
    matched_source_ids = source_ids[keep_np]
    target_ids = selected[nearest_np[keep_np]]
    matched_source = source[keep_np]
    matched_target = target_all[nearest_np[keep_np]]
    predicted = transform_points(current, matched_source)
    residual = np.linalg.norm(predicted - matched_target, axis=1)
    fit = {
        "rotation": current[:3, :3],
        "scale": 1.0,
        "translation": current[:3, 3],
        "residual": residual,
        "active": residual < 0.20,
    }
    bins = 8
    height, width = query_frames[0]["height"], query_frames[0]["width"]
    pixels = query_pixels_all[matched_source_ids]
    row_bin = np.minimum(pixels[:, 0] * bins // max(height, 1), bins - 1)
    col_bin = np.minimum(pixels[:, 1] * bins // max(width, 1), bins - 1)
    image_keys = query_frame_ids_all[matched_source_ids] * bins * bins + row_bin * bins + col_bin
    source_normal = query_normal_all[matched_source_ids]
    source_normal_valid = query_normal_valid_all[matched_source_ids]
    target_normal = memory["normal"][target_ids]
    target_normal_valid = np.linalg.norm(target_normal, axis=1) > 0.5
    normal_valid = source_normal_valid & target_normal_valid
    normal_conflict = float("nan")
    if normal_valid.any():
        mapped = source_normal[normal_valid] @ current[:3, :3].T
        cosine = np.abs(np.sum(mapped * target_normal[normal_valid], axis=1))
        normal_conflict = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))).mean())
    refinement = current @ np.linalg.inv(initial_transform)
    return {
        **pose_error(fit, pred_pose, target_pose),
        **human_diagnostics(current, query_human, old_humans),
        "fit_failed": False,
        "transform": current.tolist(),
        "correspondence_count": int(len(matched_source)),
        "mutual_match_count": int(len(np.unique(target_ids))),
        "fit_residual_mean_m": float(residual.mean()),
        "fit_residual_median_m": float(np.median(residual)),
        "fit_residual_p90_m": float(np.percentile(residual, 90)),
        "inlier_ratio_0_10m": float(np.mean(residual < 0.10)),
        "inlier_ratio_0_20m": float(np.mean(residual < 0.20)),
        "robust_inlier_ratio": float(np.mean(residual < 0.20)),
        "image_coverage_8x8": float(len(np.unique(image_keys)) / (len(query_frames) * bins * bins)),
        "mean_cosine": float("nan"),
        "mean_margin": float("nan"),
        "query_confidence_mean": float(query_conf_all[matched_source_ids].mean()),
        "query_static_rate": float(query_static_all[matched_source_ids].mean()),
        "anchor_confidence_mean": float(memory["confidence"][target_ids].mean()),
        "anchor_observation_count_mean": float(memory["observation_count"][target_ids].mean()),
        "anchor_xyz_variance_mean": float(memory["xyz_variance"][target_ids].mean()),
        "anchor_descriptor_variance_mean": float("nan"),
        "anchor_normal_stability_mean": float(memory["normal_stability"][target_ids].mean()),
        "anchor_static_rate_mean": float(memory["static_rate"][target_ids].mean()),
        "anchor_edge_score_mean": float(memory["edge_score"][target_ids].mean()),
        "normal_conflict_deg": normal_conflict,
        "source_geometry": covariance_diagnostics(matched_source),
        "target_geometry": covariance_diagnostics(matched_target),
        "icp_iterations": iterations,
        "refinement_translation_m": float(np.linalg.norm(refinement[:3, 3])),
        "refinement_rotation_deg": rotation_error_deg(refinement[:3, :3], np.eye(3, dtype=np.float32)),
    }


def run_case(record: dict, model, pred_layer, args: argparse.Namespace, device: torch.device, case_index: int) -> dict:
    spec = record_spec(record, args)
    reset_views = configure_views(one_batch(build_dataset([spec], False, args)), device, model.mhmr_img_res)
    old_views = configure_views(one_batch(old_a_dataset(spec, args)), device, model.mhmr_img_res)
    reset_predictions, reset_data, reset_seconds = capture_descriptors(
        model, reset_views[:3], args.patch_samples, args.seed + case_index * 101, device
    )
    old_predictions, old_data, old_seconds = capture_descriptors(
        model, old_views, args.patch_samples, args.seed + case_index * 101 + 17, device
    )
    available = sorted(set(descriptor_table(reset_data[0])) & set(descriptor_table(old_data[0])))
    descriptor_names = tuple(name for name in args.descriptors if name in available)
    if not descriptor_names:
        raise RuntimeError(f"No requested descriptors available: {available}")
    reset_frames = [
        frame_patches(reset_predictions[index], reset_views[index], reset_data[index], descriptor_names)
        for index in range(len(reset_predictions))
    ]
    old_frames = [
        frame_patches(old_predictions[index], old_views[index], old_data[index], descriptor_names)
        for index in range(len(old_predictions))
    ]
    memory = build_temporal_memory(old_frames, descriptor_names, args.voxel_size)
    old_humans = [human_summary(prediction, view, pred_layer) for prediction, view in zip(old_predictions, old_views)]
    reset_humans = [
        human_summary(prediction, view, pred_layer) for prediction, view in zip(reset_predictions, reset_views[:3])
    ]
    pred_pose0 = camera_matrix(reset_predictions[0])
    gt_pose0 = gt_pose_from_view(reset_views[0]).detach().float().cpu().numpy().astype(np.float32)
    old_pred_pose = camera_matrix(old_predictions[-1])
    old_gt_pose = gt_pose_from_view(old_views[-1]).detach().float().cpu().numpy().astype(np.float32)
    old_from_raw = old_pred_pose @ np.linalg.inv(old_gt_pose)
    target_pose0 = old_from_raw @ gt_pose0
    boundary_gt = old_from_raw @ gt_pose0 @ np.linalg.inv(pred_pose0)
    explicit_old, explicit_name = fixed_explicit_human3r_gauge(args.candidate_root, record["pattern_id"])
    variants = {}
    anchor_meta = {}
    retrieval_meta = {}
    for descriptor in descriptor_names:
        anchor_meta[descriptor] = {}
        for strategy in ANCHOR_STRATEGIES:
            anchor_ids = select_anchors(memory, descriptor, strategy, args.memory_capacity)
            anchor_meta[descriptor][strategy] = {
                "count": int(len(anchor_ids)),
                "observation_count_mean": float(memory["observation_count"][anchor_ids].mean()),
                "xyz_variance_mean": float(memory["xyz_variance"][anchor_ids].mean()),
                "descriptor_variance_mean": float(memory["descriptor_variance"][descriptor][anchor_ids].mean()),
                "static_rate_mean": float(memory["static_rate"][anchor_ids].mean()),
                "normal_stability_mean": float(memory["normal_stability"][anchor_ids].mean()),
                "geometry": covariance_diagnostics(memory["world"][anchor_ids]),
            }
            frame_results = {}
            for frame_count in (1, 3):
                current = reset_frames[:frame_count]
                order, scores = keyframe_order(current, old_frames, descriptor)
                retrieval_meta.setdefault(descriptor, {})[str(frame_count)] = {
                    "order": order,
                    "scores": scores,
                    "top1_score": float(scores[order[0]]),
                    "top1_margin": float(scores[order[0]] - scores[order[1]]) if len(order) > 1 else 0.0,
                }
                fits = {}
                for topk in (1, 3, 5):
                    fits[topk] = fit_candidate(
                        current,
                        memory,
                        anchor_ids,
                        order[:topk],
                        descriptor,
                        pred_pose0,
                        target_pose0,
                        args.ransac_steps,
                        args.seed + case_index * 10007 + frame_count * 101 + topk,
                        device,
                        reset_humans[0],
                        old_humans,
                    )
                candidate = dict(fits[3])
                top1_fit = None if fits[1].get("transform") is None else {
                    "rotation": np.asarray(fits[1]["transform"], dtype=np.float32)[:3, :3],
                    "translation": np.asarray(fits[1]["transform"], dtype=np.float32)[:3, 3],
                }
                top3_fit = None if fits[3].get("transform") is None else {
                    "rotation": np.asarray(fits[3]["transform"], dtype=np.float32)[:3, :3],
                    "translation": np.asarray(fits[3]["transform"], dtype=np.float32)[:3, 3],
                }
                top5_fit = None if fits[5].get("transform") is None else {
                    "rotation": np.asarray(fits[5]["transform"], dtype=np.float32)[:3, :3],
                    "translation": np.asarray(fits[5]["transform"], dtype=np.float32)[:3, 3],
                }
                t13, r13 = transform_difference(top1_fit, top3_fit)
                t35, r35 = transform_difference(top3_fit, top5_fit)
                candidate.update(
                    {
                        "descriptor": descriptor,
                        "anchor_strategy": strategy,
                        "frame_count": frame_count,
                        "topk": 3,
                        "keyframes": order[:3],
                        "global_top1_score": retrieval_meta[descriptor][str(frame_count)]["top1_score"],
                        "global_top1_margin": retrieval_meta[descriptor][str(frame_count)]["top1_margin"],
                        "top1_top3_translation_consistency_m": t13,
                        "top1_top3_rotation_consistency_deg": r13,
                        "top3_top5_translation_consistency_m": t35,
                        "top3_top5_rotation_consistency_deg": r35,
                    }
                )
                name = f"{descriptor}__{strategy}__{frame_count}f"
                variants[name] = candidate
                frame_results[frame_count] = candidate
            first = frame_results[1]
            third = frame_results[3]
            if first.get("transform") is not None and third.get("transform") is not None:
                first_fit = {
                    "rotation": np.asarray(first["transform"], dtype=np.float32)[:3, :3],
                    "translation": np.asarray(first["transform"], dtype=np.float32)[:3, 3],
                }
                third_fit = {
                    "rotation": np.asarray(third["transform"], dtype=np.float32)[:3, :3],
                    "translation": np.asarray(third["transform"], dtype=np.float32)[:3, 3],
                }
                delta_t, delta_r = transform_difference(first_fit, third_fit)
                variants[f"{descriptor}__{strategy}__1f"]["one_three_translation_consistency_m"] = delta_t
                variants[f"{descriptor}__{strategy}__1f"]["one_three_rotation_consistency_deg"] = delta_r
                variants[f"{descriptor}__{strategy}__3f"]["one_three_translation_consistency_m"] = delta_t
                variants[f"{descriptor}__{strategy}__3f"]["one_three_rotation_consistency_deg"] = delta_r
    retrieval_descriptor = "dino_mhmr" if "dino_mhmr" in descriptor_names else descriptor_names[0]
    icp_descriptor = f"explicit_icp_{retrieval_descriptor}"
    anchor_meta[icp_descriptor] = {}
    for strategy in ANCHOR_STRATEGIES:
        anchor_ids = select_anchors(memory, retrieval_descriptor, strategy, args.memory_capacity)
        anchor_meta[icp_descriptor][strategy] = {
            "count": int(len(anchor_ids)),
            "observation_count_mean": float(memory["observation_count"][anchor_ids].mean()),
            "xyz_variance_mean": float(memory["xyz_variance"][anchor_ids].mean()),
            "descriptor_variance_mean": float(
                memory["descriptor_variance"][retrieval_descriptor][anchor_ids].mean()
            ),
            "static_rate_mean": float(memory["static_rate"][anchor_ids].mean()),
            "normal_stability_mean": float(memory["normal_stability"][anchor_ids].mean()),
            "geometry": covariance_diagnostics(memory["world"][anchor_ids]),
        }
        frame_results = {}
        for frame_count in (1, 3):
            current = reset_frames[:frame_count]
            order, scores = keyframe_order(current, old_frames, retrieval_descriptor)
            retrieval_meta.setdefault(icp_descriptor, {})[str(frame_count)] = {
                "retrieval_descriptor": retrieval_descriptor,
                "order": order,
                "scores": scores,
                "top1_score": float(scores[order[0]]),
                "top1_margin": float(scores[order[0]] - scores[order[1]]) if len(order) > 1 else 0.0,
            }
            fits = {
                topk: fit_icp_candidate(
                    current,
                    memory,
                    anchor_ids,
                    order[:topk],
                    explicit_old,
                    pred_pose0,
                    target_pose0,
                    device,
                    reset_humans[0],
                    old_humans,
                )
                for topk in (1, 3, 5)
            }
            candidate = dict(fits[3])
            top1_fit = None if fits[1].get("transform") is None else {
                "rotation": np.asarray(fits[1]["transform"], dtype=np.float32)[:3, :3],
                "translation": np.asarray(fits[1]["transform"], dtype=np.float32)[:3, 3],
            }
            top3_fit = None if fits[3].get("transform") is None else {
                "rotation": np.asarray(fits[3]["transform"], dtype=np.float32)[:3, :3],
                "translation": np.asarray(fits[3]["transform"], dtype=np.float32)[:3, 3],
            }
            top5_fit = None if fits[5].get("transform") is None else {
                "rotation": np.asarray(fits[5]["transform"], dtype=np.float32)[:3, :3],
                "translation": np.asarray(fits[5]["transform"], dtype=np.float32)[:3, 3],
            }
            t13, r13 = transform_difference(top1_fit, top3_fit)
            t35, r35 = transform_difference(top3_fit, top5_fit)
            candidate.update(
                {
                    "descriptor": icp_descriptor,
                    "retrieval_descriptor": retrieval_descriptor,
                    "anchor_strategy": strategy,
                    "frame_count": frame_count,
                    "topk": 3,
                    "keyframes": order[:3],
                    "global_top1_score": retrieval_meta[icp_descriptor][str(frame_count)]["top1_score"],
                    "global_top1_margin": retrieval_meta[icp_descriptor][str(frame_count)]["top1_margin"],
                    "top1_top3_translation_consistency_m": t13,
                    "top1_top3_rotation_consistency_deg": r13,
                    "top3_top5_translation_consistency_m": t35,
                    "top3_top5_rotation_consistency_deg": r35,
                }
            )
            variants[f"{icp_descriptor}__{strategy}__{frame_count}f"] = candidate
            frame_results[frame_count] = candidate
        first, third = frame_results[1], frame_results[3]
        if first.get("transform") is not None and third.get("transform") is not None:
            first_fit = {
                "rotation": np.asarray(first["transform"], dtype=np.float32)[:3, :3],
                "translation": np.asarray(first["transform"], dtype=np.float32)[:3, 3],
            }
            third_fit = {
                "rotation": np.asarray(third["transform"], dtype=np.float32)[:3, :3],
                "translation": np.asarray(third["transform"], dtype=np.float32)[:3, 3],
            }
            delta_t, delta_r = transform_difference(first_fit, third_fit)
            for frame_count in (1, 3):
                name = f"{icp_descriptor}__{strategy}__{frame_count}f"
                variants[name]["one_three_translation_consistency_m"] = delta_t
                variants[name]["one_three_rotation_consistency_deg"] = delta_r
    output = {
        "case_name": record["pattern_id"],
        "record": record,
        "protocol": {
            "gt_depth_used": False,
            "gt_correspondence_used": False,
            "gt_camera_use": "evaluation_labels_only",
            "history_is_causal": True,
            "memory_capacity": int(args.memory_capacity),
            "precut_frame_count": len(old_frames),
            "postcut_frame_count": len(reset_frames),
        },
        "texture_score": texture_score(reset_views[0]),
        "memory": {
            "raw_patch_count": int(sum(len(frame["world"]) for frame in old_frames)),
            "voxel_anchor_count": int(len(memory["world"])),
            "capacity": int(args.memory_capacity),
            "voxel_size_m": float(args.voxel_size),
        },
        "baselines": {
            "hard_reset": direct_transform_error(np.eye(4, dtype=np.float32), pred_pose0, target_pose0),
            "fixed_explicit": {**direct_transform_error(explicit_old, pred_pose0, target_pose0), "name": explicit_name},
            "boundary_oracle": direct_transform_error(boundary_gt, pred_pose0, target_pose0),
        },
        "anchor_meta": anchor_meta,
        "retrieval": retrieval_meta,
        "variants": variants,
        "timing_seconds": {"reset_capture": reset_seconds, "history_capture": old_seconds},
    }
    del reset_views, old_views, reset_predictions, old_predictions
    torch.cuda.empty_cache()
    return output


def main() -> None:
    args = parse_args()
    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("V14 candidate generation requires CUDA Human3R inference")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard index")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"v14_candidates_shard_{args.shard_index:02d}_of_{args.num_shards:02d}.json"
    if output_path.is_file() and not args.overwrite:
        print(f">> exists {output_path}")
        return
    records = read_jsonl(args.records)
    selected = [row for index, row in enumerate(records) if index % args.num_shards == args.shard_index]
    if args.max_cases > 0:
        selected = selected[: args.max_cases]
    model = build_model(args)
    _, pred_layer = build_smpl_models(model, torch.device(args.device))
    device = torch.device(args.device)
    cases = []
    started = time.perf_counter()
    for index, record in enumerate(selected):
        case = run_case(record, model, pred_layer, args, device, index)
        cases.append(case)
        valid = [
            value
            for value in case["variants"].values()
            if not value.get("fit_failed", True) and value.get("frame_count") == 3
        ]
        best = min(
            valid,
            key=lambda row: row["camera_translation_error_m"] / 0.25 + row["camera_rotation_error_deg"] / 5.0,
            default={},
        )
        print(
            f">> [{index + 1}/{len(selected)}] {record['pattern_id']} "
            f"bestT={best.get('camera_translation_error_m', float('nan')):.3f} "
            f"bestR={best.get('camera_rotation_error_deg', float('nan')):.2f}",
            flush=True,
        )
    report = {
        "experiment": "V14 Depth-Free Selective World-Memory Candidates",
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "case_count": len(cases),
        "elapsed_seconds": time.perf_counter() - started,
        "cases": cases,
    }
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    print(f">> wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
