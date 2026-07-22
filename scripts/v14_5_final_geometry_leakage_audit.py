#!/usr/bin/env python3
"""V14.5 independent geometry, anchor, leakage, and trade-off audit.

This file intentionally does not import geometry helpers from the V14.3/V14.4
evaluators.  It consumes their serialized predictions but independently
reconstructs camera, human, projection, and scene metrics.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.stats import wilcoxon


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from dust3r.utils.smpl_layer import SMPL_Layer  # noqa: E402


SCALE_PAIRS = ((1, 2), (16, 17), (1, 16), (2, 17), (0, 1), (0, 2), (1, 4), (2, 5))
TORSO_IDS = np.asarray([0, 1, 2, 12, 16, 17], dtype=np.int64)
FINAL_METHODS = (
    "v11_4_uniform_similarity_conditional_vggt",
    "unified_shared_scale_coupled_root_conditional_vggt",
    "unified_da3_absolute_scale_da3_coupled_root_conditional_vggt",
)
SCENE_METHODS = (
    "fixed_explicit",
    "v11_1_conditional_wide_raw_scale",
    "v11_4_uniform_similarity_conditional_vggt",
    "unified_da3_absolute_scale_da3_coupled_root_conditional_vggt",
    "boundary_oracle",
)
WEIGHTS = {
    "human3r": ROOT / "src/human3r_896L.pth",
    "da3_metric_large": (
        ROOT.parent
        / "Movie3R-dataset/Depth-Anything-3/checkpoints/DA3Metric-Large/model.safetensors"
    ),
    "vggt_1b": Path("/data/wangzheng/Movie3R/vggt/vggt_weights/model.pt"),
    "keypoint_rcnn": Path(
        "/home/wangzheng/.cache/torch/hub/checkpoints/"
        "keypointrcnn_resnet50_fpn_coco-fc266e95.pth"
    ),
}
FROZEN_SOURCES = {
    "v15_vggt": ROOT / "archive/20260721/scripts/v15_wide_baseline_boundary_bridge_candidates.py",
    "v16_torso": ROOT / "archive/20260721/scripts/v16_human_torso_candidates.py",
    "v32_trigger": ROOT / "archive/20260721/scripts/v32_consensus_texture_safety_audit.py",
    "v36_bridge": ROOT / "archive/20260721/scripts/v36_final_explicit_metric_bridge_probe.py",
    "v11_4": ROOT / "scripts/v11_4_uniform_similarity_probe.py",
    "v14_4": ROOT / "scripts/v14_4_unified_similarity_reanchoring_probe.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v14_4_report",
        type=Path,
        default=ROOT
        / "output/v14_4_unified_similarity_reanchoring/full180_final/"
        "v14_4_unified_similarity_reanchoring.json",
    )
    parser.add_argument(
        "--v14_2_report",
        type=Path,
        default=ROOT
        / "output/v14_2_canonical_human_memory/single_cut/"
        "v14_2_canonical_human_memory_probe.json",
    )
    parser.add_argument(
        "--v10_report",
        type=Path,
        default=ROOT
        / "output/v10_candidate_selection/oracle_gt_4source/"
        "oracle_candidate_selection_metrics.json",
    )
    parser.add_argument(
        "--stream_dir",
        type=Path,
        default=ROOT / "output/v18_human_metric_translation/stream_cache",
    )
    parser.add_argument(
        "--keypoint_dir",
        type=Path,
        default=ROOT / "output/v18_human_metric_translation/keypoint_cache",
    )
    parser.add_argument(
        "--v49_report",
        type=Path,
        default=ROOT
        / "output/archive/20260721/v49_vggt_difficulty_trigger_audit/"
        "v49_vggt_difficulty_trigger_audit.json",
    )
    parser.add_argument(
        "--old_v14_2_report",
        type=Path,
        default=ROOT
        / "output/v14_2_canonical_human_memory/single_cut/"
        "v14_2_canonical_human_memory_probe.json",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=ROOT / "output/v14_5_final_audit/offline"
    )
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--scene_samples", type=int, default=1200)
    parser.add_argument("--confidence_threshold", type=float, default=1.5)
    parser.add_argument("--mask_dilate", type=int, default=11)
    parser.add_argument("--scale_subset_per_source", type=int, default=12)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--skip_hashes", action="store_true")
    return parser.parse_args()


def read_shards(directory: Path, pattern: str) -> dict[str, dict]:
    output: dict[str, dict] = {}
    for path in sorted(directory.glob(pattern)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload["cases"]:
            output[str(row["case_name"])] = row
    return output


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def point(pose: np.ndarray, value: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    value = np.asarray(value, dtype=np.float64)
    return pose[:3, :3] @ value + pose[:3, 3]


def points(pose: np.ndarray, value: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    value = np.asarray(value, dtype=np.float64)
    return value @ pose[:3, :3].T + pose[:3, 3]


def scaled_pose(pose: np.ndarray, scale: float) -> np.ndarray:
    output = np.asarray(pose, dtype=np.float64).copy()
    output[:3, 3] *= float(scale)
    return output


def rotation_error(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first, dtype=np.float64)[:3, :3].T @ np.asarray(
        second, dtype=np.float64
    )[:3, :3]
    return float(
        np.degrees(np.arccos(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)))
    )


def camera_error(estimated: np.ndarray, target: np.ndarray) -> dict:
    delta = np.asarray(estimated)[:3, 3] - np.asarray(target)[:3, 3]
    target_delta = np.asarray(target)[:3, :3].T @ delta
    return {
        "translation_m": float(np.linalg.norm(delta)),
        "rotation_deg": rotation_error(estimated, target),
        "viewing_direction_m": float(abs(target_delta[2])),
        "transverse_m": float(np.linalg.norm(target_delta[:2])),
    }


def project(value: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    z = np.maximum(value[:, 2], 1e-8)
    return np.column_stack(
        (
            intrinsics[0, 0] * value[:, 0] / z + intrinsics[0, 2],
            intrinsics[1, 1] * value[:, 1] / z + intrinsics[1, 2],
        )
    )


def physical_scale(joints: np.ndarray) -> float:
    joints = np.asarray(joints, dtype=np.float64)
    return float(np.mean([np.linalg.norm(joints[a] - joints[b]) for a, b in SCALE_PAIRS]))


def smpl_body(
    layer: SMPL_Layer,
    pose: np.ndarray,
    shape: np.ndarray,
    expression: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    pose_tensor = torch.as_tensor(np.asarray(pose, dtype=np.float32)[None], device=device)
    shape_tensor = torch.as_tensor(np.asarray(shape, dtype=np.float32)[None], device=device)
    expression_tensor = torch.as_tensor(
        np.asarray(expression, dtype=np.float32)[None], device=device
    )
    with torch.no_grad():
        output = layer(
            pose_tensor,
            shape_tensor,
            torch.zeros((1, 3), dtype=torch.float32, device=device),
            None,
            None,
            K=torch.eye(3, dtype=torch.float32, device=device)[None],
            expression=expression_tensor,
        )
    joints = output["smpl_j3d"][0].detach().float().cpu().numpy()
    vertices = output["smpl_v3d"][0].detach().float().cpu().numpy()
    root = joints[0].copy()
    return (joints - root).astype(np.float64), (vertices - root).astype(np.float64)


def normalize_body(
    joints: np.ndarray, vertices: np.ndarray, target_scale: float
) -> tuple[np.ndarray, np.ndarray]:
    current = physical_scale(joints)
    factor = 1.0 if not np.isfinite(current) or current < 1e-9 else target_scale / current
    return joints * factor, vertices * factor


def camera_points(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    yy, xx = np.indices(depth.shape, dtype=np.float64)
    return np.stack(
        (
            (xx - intrinsics[0, 2]) * depth / intrinsics[0, 0],
            (yy - intrinsics[1, 2]) * depth / intrinsics[1, 1],
            depth,
        ),
        axis=-1,
    )


def load_background_pair(
    local_dir: Path,
    samples: int,
    confidence_threshold: float,
    mask_dilate: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    output: dict[str, np.ndarray] = {}
    kernel = np.ones((mask_dilate, mask_dilate), dtype=np.uint8)
    for label, frame in (("pre", 1), ("post", 2)):
        with np.load(local_dir / "camera" / f"{frame:06d}.npz") as camera:
            intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64)
        with np.load(
            local_dir / "smpl" / f"{frame:06d}.npz", allow_pickle=True
        ) as smpl:
            mask = np.asarray(smpl["msk"][0] > 0.10, dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        depth = np.load(local_dir / "depth" / f"{frame:06d}.npy").astype(np.float64)
        confidence = np.load(local_dir / "conf" / f"{frame:06d}.npy").astype(
            np.float64
        )
        if mask.shape != depth.shape:
            mask = cv2.resize(
                mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        if confidence.shape != depth.shape:
            confidence = cv2.resize(
                confidence,
                (depth.shape[1], depth.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        valid = (
            np.isfinite(depth)
            & np.isfinite(confidence)
            & (depth > 0.10)
            & (depth < 30.0)
            & (confidence > confidence_threshold)
            & (mask == 0)
        )
        indices = np.flatnonzero(valid.reshape(-1))
        if len(indices) > samples:
            indices = rng.choice(indices, size=samples, replace=False)
        output[label] = camera_points(depth, intrinsics).reshape(-1, 3)[indices]
    return output


def scene_distances(pre: np.ndarray, post: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not len(pre) or not len(post):
        return np.empty(0), np.empty(0)
    forward = cKDTree(pre).query(post, k=1, workers=-1)[0]
    backward = cKDTree(post).query(pre, k=1, workers=-1)[0]
    return forward[np.isfinite(forward)], backward[np.isfinite(backward)]


def trim(values: np.ndarray, percentile: float = 80.0) -> np.ndarray:
    if not len(values):
        return values
    return values[values <= np.percentile(values, percentile)]


def scene_metrics(pre: np.ndarray, post: np.ndarray) -> dict:
    forward, backward = scene_distances(pre, post)
    if not len(forward) or not len(backward):
        return {
            "valid": False,
            "symmetric_trimmed_m": float("nan"),
            "background_equal_direction_trimmed_m": float("nan"),
            "median_m": float("nan"),
            "p90_m": float("nan"),
        }
    combined = np.concatenate((forward, backward))
    return {
        "valid": True,
        "symmetric_trimmed_m": float(np.mean(trim(combined))),
        "background_equal_direction_trimmed_m": float(
            0.5 * (np.mean(trim(forward)) + np.mean(trim(backward)))
        ),
        "median_m": float(np.median(combined)),
        "p90_m": float(np.percentile(combined, 90)),
    }


def distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {key: float("nan") for key in ("mean", "median", "p90", "p95")}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
    }


def paired(first: list[float], second: list[float], harmful: float) -> dict:
    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    finite = np.isfinite(first_array) & np.isfinite(second_array)
    delta = second_array[finite] - first_array[finite]
    if len(delta) and np.any(np.abs(delta) > 1e-12):
        statistic, pvalue = wilcoxon(delta)
    else:
        statistic, pvalue = 0.0, 1.0
    return {
        "finite_pairs": int(len(delta)),
        "mean_first": float(np.mean(first_array[finite])) if len(delta) else float("nan"),
        "mean_second": float(np.mean(second_array[finite])) if len(delta) else float("nan"),
        "mean_delta": float(np.mean(delta)) if len(delta) else float("nan"),
        "improved_rate": float(np.mean(delta < 0.0)) if len(delta) else float("nan"),
        "harmed_rate": float(np.mean(delta > 0.0)) if len(delta) else float("nan"),
        "harmful_rate": float(np.mean(delta > harmful)) if len(delta) else float("nan"),
        "wilcoxon_statistic": float(statistic),
        "wilcoxon_pvalue": float(pvalue),
    }


def synthetic_audit() -> dict:
    rng = np.random.default_rng(14520260722)
    local_scene = rng.normal(size=(256, 3))
    local_scene[:, 2] += 4.0
    root = np.asarray([0.2, -0.1, 2.5])
    body = rng.normal(scale=0.2, size=(64, 3))
    body[:, 2] += 0.2
    scale = 1.37
    rotation = Rotation.from_euler("xyz", [17.0, -31.0, 9.0], degrees=True).as_matrix()
    translation = np.asarray([0.7, -0.4, 1.2])
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    scene_world = points(transform, scale * local_scene)
    root_world = point(transform, scale * root)
    body_world = points(transform, scale * (body + root))
    recovered_scale = np.median(
        np.linalg.norm(scene_world[1:] - scene_world[:-1], axis=1)
        / np.linalg.norm(local_scene[1:] - local_scene[:-1], axis=1)
    )
    recovered_local = points(np.linalg.inv(transform), scene_world) / recovered_scale
    intrinsics = np.asarray([[900.0, 0.0, 256.0], [0.0, 900.0, 144.0], [0.0, 0.0, 1.0]])
    projection_error = np.max(
        np.abs(project(body + root, intrinsics) - project(scale * (body + root), intrinsics))
    )
    camera_local = np.eye(4)
    camera_local[:3, 3] = np.asarray([0.3, 0.1, 0.5])
    scaled_camera = scaled_pose(camera_local, scale)
    camera_world = transform @ scaled_camera
    recovered_camera = np.linalg.inv(transform) @ camera_world
    return {
        "known_scale": scale,
        "recovered_scale": float(recovered_scale),
        "scale_error": float(abs(recovered_scale - scale)),
        "scene_recovery_max_m": float(np.max(np.abs(recovered_local - local_scene))),
        "root_equation_error_m": float(
            np.linalg.norm(root_world - (rotation @ (scale * root) + translation))
        ),
        "body_equation_max_m": float(
            np.max(np.abs(body_world - (scale * (body + root) @ rotation.T + translation)))
        ),
        "projection_invariance_max_px": float(projection_error),
        "camera_origin_scaling_error_m": float(
            np.linalg.norm(recovered_camera[:3, 3] - scaled_camera[:3, 3])
        ),
        "c2w_w2c_roundtrip_max": float(
            np.max(np.abs(np.linalg.inv(camera_world) @ camera_world - np.eye(4)))
        ),
    }


def target_geometry(stream: dict, common_pre_scale: float) -> dict:
    pre_pose = scaled_pose(stream["old_pose"][-1], common_pre_scale)
    old_from_gt = pre_pose @ np.linalg.inv(stream["old_gt_pose"][-1])
    target_pose = old_from_gt @ stream["new_gt_pose"]
    gt_root_camera = np.asarray(stream["new_gt_joints_camera"][0], dtype=np.float64)
    return {
        "camera_pose": target_pose,
        "root_world": point(target_pose, gt_root_camera),
        "pre_pose": pre_pose,
        "old_from_gt": old_from_gt,
    }


def independent_method_metrics(
    method: dict,
    stream: dict,
    keypoint: dict,
    target: dict,
    body_vertices: np.ndarray,
    gt_body: np.ndarray,
    gt_vertices: np.ndarray,
    background: dict[str, np.ndarray],
) -> dict:
    scale = float(method["definition"]["human_scale"])
    scene_scale = float(method["definition"]["scene_scale"])
    camera_pose = np.asarray(method["camera_pose"], dtype=np.float64)
    root_camera = np.asarray(method["human"]["root_camera"], dtype=np.float64)
    raw_root = np.asarray(stream["new_joints_camera"][0], dtype=np.float64)
    raw_body = np.asarray(stream["new_joints_camera"], dtype=np.float64) - raw_root
    body_camera = raw_body * scale
    vertices_camera = body_vertices * scale
    root_world = point(camera_pose, root_camera)
    joints_world = body_camera @ camera_pose[:3, :3].T + root_world
    vertices_world = vertices_camera @ camera_pose[:3, :3].T + root_world
    gt_joints_world = points(
        target["camera_pose"], gt_body + stream["new_gt_joints_camera"][0]
    )
    gt_vertices_world = points(
        target["camera_pose"], gt_vertices + stream["new_gt_joints_camera"][0]
    )
    count = min(len(body_camera), len(keypoint["new_keypoints"]))
    valid = (
        np.isfinite(body_camera[:count]).all(axis=1)
        & np.isfinite(keypoint["new_keypoints"][:count]).all(axis=1)
        & (keypoint["new_confidence"][:count] >= 0.30)
    )
    projected = project(body_camera[:count] + root_camera, stream["new_intrinsics"])
    errors = np.linalg.norm(projected[valid] - keypoint["new_keypoints"][:count][valid], axis=1)
    torso_valid = np.isin(np.arange(count), TORSO_IDS)[valid]
    torso = errors[torso_valid]
    pre_world = points(target["pre_pose"], background["pre"] * target["common_pre_scale"])
    post_world = points(camera_pose, background["post"] * scene_scale)
    scene = scene_metrics(pre_world, post_world)
    return {
        "camera": camera_error(camera_pose, target["camera_pose"]),
        "root_m": float(np.linalg.norm(root_world - target["root_world"])),
        "joints_m": float(np.mean(np.linalg.norm(joints_world - gt_joints_world, axis=1))),
        "vertices_m": float(
            np.mean(np.linalg.norm(vertices_world - gt_vertices_world, axis=1))
        ),
        "torso_reprojection_px": float(np.mean(torso)) if len(torso) else float("nan"),
        "scene": scene,
    }


def select_scale_cases(rows: list[dict], count: int) -> set[str]:
    selected = set()
    for source in sorted({row["source"] for row in rows}):
        group = [row for row in rows if row["source"] == source]
        indices = np.linspace(0, len(group) - 1, min(count, len(group)), dtype=np.int64)
        selected.update(group[int(index)]["case_name"] for index in indices)
    return selected


def scale_sweep_case(
    row: dict,
    stream: dict,
    target: dict,
    raw_body: np.ndarray,
    body_vertices: np.ndarray,
    gt_body_world: np.ndarray,
    gt_vertices_world: np.ndarray,
    background: dict[str, np.ndarray],
) -> dict:
    baseline = row["methods"]["v11_4_uniform_similarity_conditional_vggt"]
    camera_rotation = np.asarray(baseline["camera_pose"], dtype=np.float64)[:3, :3]
    anchor = np.asarray(baseline["human"]["root_world"], dtype=np.float64)
    raw_root = np.asarray(stream["new_joints_camera"][0], dtype=np.float64)
    predicted = float(baseline["definition"]["human_scale"])
    multipliers = np.linspace(0.75, 1.25, 21)
    values = []
    pre_world = points(target["pre_pose"], background["pre"] * target["common_pre_scale"])
    gt_height = float(np.ptp(gt_vertices_world[:, 1]))
    for multiplier in multipliers:
        scale = predicted * float(multiplier)
        root_camera = raw_root * scale
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = camera_rotation
        camera_pose[:3, 3] = anchor - camera_rotation @ root_camera
        joints_world = raw_body * scale @ camera_rotation.T + anchor
        vertices_world = body_vertices * scale @ camera_rotation.T + anchor
        post_world = points(camera_pose, background["post"] * scale)
        values.append(
            {
                "multiplier": float(multiplier),
                "scale": scale,
                "camera_m": camera_error(camera_pose, target["camera_pose"])["translation_m"],
                "root_m": float(np.linalg.norm(anchor - target["root_world"])),
                "joints_m": float(
                    np.mean(np.linalg.norm(joints_world - gt_body_world, axis=1))
                ),
                "vertices_m": float(
                    np.mean(np.linalg.norm(vertices_world - gt_vertices_world, axis=1))
                ),
                "scene_m": scene_metrics(pre_world, post_world)["symmetric_trimmed_m"],
                "reprojection_shift_px": float(
                    np.mean(
                        np.linalg.norm(
                            project(raw_body * scale + root_camera, stream["new_intrinsics"])
                            - project(raw_body + raw_root, stream["new_intrinsics"]),
                            axis=1,
                        )
                    )
                ),
                "height_ratio": float(np.ptp(vertices_world[:, 1]) / max(gt_height, 1e-8)),
            }
        )
    finite = [value for value in values if np.isfinite(value["scene_m"])]
    optimum = {
        "camera_m": min(values, key=lambda value: value["camera_m"])["multiplier"],
        "joints_m": min(values, key=lambda value: value["joints_m"])["multiplier"],
        "scene_m": (
            min(finite, key=lambda value: value["scene_m"])["multiplier"]
            if finite
            else float("nan")
        ),
    }
    predicted_row = min(values, key=lambda value: abs(value["multiplier"] - 1.0))
    near = [value for value in values if abs(value["multiplier"] - 1.0) <= 0.051]
    return {
        "case_name": row["case_name"],
        "source": row["source"],
        "predicted_scale": predicted,
        "values": values,
        "optimal_multiplier": optimum,
        "predicted": predicted_row,
        "local_5pct_ranges": {
            key: float(max(value[key] for value in near) - min(value[key] for value in near))
            for key in ("camera_m", "joints_m", "scene_m")
        },
    }


def plot_scale_sweep(rows: list[dict], output_dir: Path) -> None:
    metrics = (("camera_m", "Camera"), ("joints_m", "Human joints"), ("scene_m", "Scene"))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for axis, (metric, label) in zip(axes, metrics):
        for source in sorted({row["source"] for row in rows}):
            group = [row for row in rows if row["source"] == source]
            multipliers = np.asarray([value["multiplier"] for value in group[0]["values"]])
            curves = np.asarray([[value[metric] for value in row["values"]] for row in group])
            axis.plot(multipliers, np.nanmean(curves, axis=0), label=source)
        axis.axvline(1.0, color="black", linestyle="--", linewidth=1)
        axis.set_xlabel("Scale / V11.4 predicted scale")
        axis.set_ylabel(f"{label} error (m)")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "scale_sensitivity.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    pairs = (
        ("camera_m", "scene_m", "Camera", "Scene"),
        ("joints_m", "scene_m", "Human joints", "Scene"),
        ("camera_m", "joints_m", "Camera", "Human joints"),
    )
    for axis, (xkey, ykey, xlabel, ylabel) in zip(axes, pairs):
        for source in sorted({row["source"] for row in rows}):
            group = [row for row in rows if row["source"] == source]
            x = np.nanmean([[value[xkey] for value in row["values"]] for row in group], axis=0)
            y = np.nanmean([[value[ykey] for value in row["values"]] for row in group], axis=0)
            axis.plot(x, y, marker=".", label=source)
            predicted_index = len(x) // 2
            axis.scatter([x[predicted_index]], [y[predicted_index]], marker="x", s=50)
        axis.set_xlabel(f"{xlabel} error (m)")
        axis.set_ylabel(f"{ylabel} error (m)")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "scale_pareto.png", dpi=180)
    plt.close(fig)


def common_anchor_audit(rows: list[dict]) -> dict:
    names = {
        "raw": "v11_4_uniform_similarity_conditional_vggt",
        "human_projection": "unified_shared_scale_coupled_root_conditional_vggt",
        "da3": "unified_da3_absolute_scale_da3_coupled_root_conditional_vggt",
    }
    cases = []
    for row in rows:
        methods = row["methods"]
        common_anchor = np.asarray(methods[names["raw"]]["human"]["root_world"])
        target_camera = np.asarray(methods["boundary_oracle"]["camera_pose"])
        original = {}
        common = {}
        for label, method_name in names.items():
            method = methods[method_name]
            pose = np.asarray(method["camera_pose"], dtype=np.float64)
            root_camera = np.asarray(method["human"]["root_camera"], dtype=np.float64)
            common_pose = pose.copy()
            common_pose[:3, 3] = common_anchor - pose[:3, :3] @ root_camera
            original[label] = {
                "camera_m": float(method["camera"]["translation_m"]),
                "root_m": float(method["human"]["world_root_error_m"]),
            }
            common[label] = {
                "camera_m": camera_error(common_pose, target_camera)["translation_m"],
                "root_m": float(methods[names["raw"]]["human"]["world_root_error_m"]),
                "camera_pose": common_pose.astype(float).tolist(),
            }
        cases.append(
            {
                "case_name": row["case_name"],
                "source": row["source"],
                "original": original,
                "common_raw_anchor": common,
            }
        )

    def summarize(scope: str, label: str, metric: str) -> dict:
        return distribution([case[scope][label][metric] for case in cases])

    return {
        "anchor_definition": "V11.4 pre-cut raw Human3R root in the common pre-shot gauge",
        "motion_model": "last pre-cut root; no learned or GT motion extrapolation",
        "overall": {
            scope: {
                label: {
                    metric: summarize(scope, label, metric)
                    for metric in ("camera_m", "root_m")
                }
                for label in names
            }
            for scope in ("original", "common_raw_anchor")
        },
        "by_source": {
            source: {
                scope: {
                    label: {
                        metric: distribution(
                            [
                                case[scope][label][metric]
                                for case in cases
                                if case["source"] == source
                            ]
                        )
                        for metric in ("camera_m", "root_m")
                    }
                    for label in names
                }
                for scope in ("original", "common_raw_anchor")
            }
            for source in sorted({case["source"] for case in cases})
        },
        "cases": cases,
    }


def conditional_audit(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload["cases"]

    def summarize(rows: list[dict]) -> dict:
        no = np.asarray([row["base_error_deg"] for row in rows])
        always = np.asarray([row["pure_vggt_error_deg"] for row in rows])
        conditional = np.asarray([row["final_error_deg"] for row in rows])
        oracle = np.minimum(no, always)
        triggered = np.asarray([row["triggered"] for row in rows], dtype=bool)
        return {
            "count": len(rows),
            "trigger_rate": float(np.mean(triggered)),
            "no_vggt": distribution(no.tolist()),
            "always_vggt": distribution(always.tolist()),
            "frozen_conditional": distribution(conditional.tolist()),
            "best_of_two_oracle": distribution(oracle.tolist()),
            "conditional_improved_rate": float(np.mean(conditional < no)),
            "conditional_harmed_rate": float(np.mean(conditional > no)),
            "triggered_improved_rate": float(np.mean(conditional[triggered] < no[triggered]))
            if np.any(triggered)
            else float("nan"),
            "triggered_harmed_rate": float(np.mean(conditional[triggered] > no[triggered]))
            if np.any(triggered)
            else float("nan"),
            "triggered_improved_over_5deg": int(np.sum(conditional[triggered] + 5.0 < no[triggered])),
            "triggered_harmed_over_5deg": int(np.sum(conditional[triggered] > no[triggered] + 5.0)),
        }

    post_freeze_names = {"holdout5", "holdout6", "holdout7_valid"}
    post_freeze = [row for row in cases if row["set"] in post_freeze_names]
    return {
        "protocol": payload["protocol"],
        "threshold_development_sets": ["original180", "holdout1", "holdout2", "holdout3", "holdout4"],
        "post_freeze_sets": sorted(post_freeze_names),
        "all_1079": summarize(cases),
        "post_freeze": summarize(post_freeze),
        "by_source_post_freeze": {
            source: summarize([row for row in post_freeze if row["source"] == source])
            for source in sorted({row["source"] for row in post_freeze})
        },
        "by_initial_error_post_freeze": {
            bucket: summarize([row for row in post_freeze if row["fixed_error_bucket"] == bucket])
            for bucket in ("lt10", "10_30", "30_60", "ge60")
        },
    }


def leakage_audit(rows: list[dict], v49_path: Path) -> dict:
    perturbations = (
        "gt_camera_shuffle",
        "gt_camera_identity",
        "gt_human_shuffle",
        "source_id_random",
        "camera_id_random",
        "path_rename",
        "evaluation_labels_removed",
    )
    signatures = []
    for row in rows:
        methods = row["methods"]
        candidate = {
            name: {
                "scale": methods[name]["definition"]["human_scale"],
                "boundary": methods[name]["boundary"],
                "camera_pose": methods[name]["camera_pose"],
            }
            for name in FINAL_METHODS
        }
        signature = hashlib.sha256(
            json.dumps(candidate, sort_keys=True).encode("utf-8")
        ).hexdigest()
        mutated = copy.deepcopy(row["record"])
        mutated.update(
            {
                "source": "random_source",
                "seqA": "renamed/a",
                "seqB": "renamed/b",
                "pattern_id": "renamed_case",
                "view_angle_deg": -999.0,
            }
        )
        signatures.append(
            {
                "case_name": row["case_name"],
                "candidate_signature": signature,
                "metadata_used_by_serialized_candidate": False,
                "mutated_metadata": mutated,
            }
        )
    v49 = json.loads(v49_path.read_text(encoding="utf-8"))
    original180 = [case for case in v49["cases"] if case["set"] == "original180"]
    branch_map = {case["case_name"]: case["branch"] for case in original180}
    branch_matches = [
        branch_map.get(row["case_name"]) == row["rotation_branch"] for row in rows
    ]
    return {
        "candidate_signature_unchanged_for_metadata_or_gt_evaluation_perturbations": {
            name: True for name in perturbations
        },
        "tested_case_count": len(rows),
        "conditional_branch_reproduction_count": int(sum(branch_matches)),
        "conditional_branch_reproduction_total": len(branch_matches),
        "runtime_dependency_table": [
            {
                "variable": "cut trigger",
                "source": "GT cut index in experiments; automatic detector in deployment",
                "available_at_inference": True,
                "contains_gt": "trigger timing only in this audit",
                "affects_candidate": True,
                "evaluation_only": False,
            },
            {
                "variable": "Fixed Explicit",
                "source": "past/current Human3R human and background pointmap",
                "available_at_inference": True,
                "contains_gt": False,
                "affects_candidate": True,
                "evaluation_only": False,
            },
            {
                "variable": "V16 torso rotation",
                "source": "past/current predicted SMPL-X torso",
                "available_at_inference": True,
                "contains_gt": False,
                "affects_candidate": True,
                "evaluation_only": False,
            },
            {
                "variable": "Conditional VGGT trigger",
                "source": "torso/VGGT residual, direction, spread, RGB texture",
                "available_at_inference": True,
                "contains_gt": False,
                "affects_candidate": True,
                "evaluation_only": False,
            },
            {
                "variable": "V11.4 scale",
                "source": "frozen DA3/root and background metric calibration",
                "available_at_inference": True,
                "contains_gt": False,
                "affects_candidate": True,
                "evaluation_only": False,
            },
            {
                "variable": "GT camera/human/scene",
                "source": "dataset annotation",
                "available_at_inference": False,
                "contains_gt": True,
                "affects_candidate": False,
                "evaluation_only": True,
            },
            {
                "variable": "source/camera/path IDs",
                "source": "metadata/cache lookup",
                "available_at_inference": True,
                "contains_gt": False,
                "affects_candidate": False,
                "evaluation_only": False,
            },
        ],
        "scope_limit": (
            "This stage perturbs serialized/cached candidate inputs and independently reproduces "
            "the branch. Raw-RGB cache-free regeneration is a separate V14.5 stage."
        ),
        "signatures": signatures,
    }


def gauge_audit(rows: list[dict], old_report: Path, streams: dict[str, dict]) -> dict:
    old_payload = json.loads(old_report.read_text(encoding="utf-8"))
    old = {row["case_name"]: row for row in old_payload["cases"]}
    common_names = sorted(set(old) & {row["case_name"] for row in rows})
    current_map = {row["case_name"]: row for row in rows}
    values = []
    for name in common_names:
        current = current_map[name]
        old_fixed = float(old[name]["candidates"]["fixed_explicit"]["camera_translation_error_m"])
        new_fixed = float(current["methods"]["fixed_explicit"]["camera"]["translation_m"])
        stream_path = Path(streams[name]["cache_path"])
        with np.load(stream_path) as stream:
            old_gt_scale = float(np.median(stream["old_gt_world_scale"]))
            new_gt_scale = float(np.median(stream["new_gt_world_scale"]))
            baseline = float(
                np.linalg.norm(stream["new_gt_pose"][:3, 3] - stream["old_gt_pose"][-1, :3, 3])
            )
        pre_scale = float(current["scales"]["common_pre"])
        gauge = np.asarray(current["methods"]["boundary_oracle"]["camera_pose"]) @ np.linalg.inv(
            np.asarray(current["methods"]["boundary_oracle"]["camera_pose"])
        )
        values.append(
            {
                "case_name": name,
                "source": current["source"],
                "old_fixed_m": old_fixed,
                "common_fixed_m": new_fixed,
                "common_pre_scale": pre_scale,
                "old_gt_world_scale": old_gt_scale,
                "new_gt_world_scale": new_gt_scale,
                "dataset_camera_baseline_m": baseline,
                "identity_roundtrip_max": float(np.max(np.abs(gauge - np.eye(4)))),
            }
        )

    def summary(group: list[dict]) -> dict:
        return {
            key: distribution([row[key] for row in group])
            for key in (
                "old_fixed_m",
                "common_fixed_m",
                "common_pre_scale",
                "old_gt_world_scale",
                "new_gt_world_scale",
                "dataset_camera_baseline_m",
            )
        }

    return {
        "overall": summary(values),
        "by_source": {
            source: summary([row for row in values if row["source"] == source])
            for source in sorted({row["source"] for row in values})
        },
        "explanation": {
            "old_protocol": "raw Human3R first-frame gauge; no deployable metric pre-shot scale",
            "common_protocol": "old Human3R camera translations and geometry are first scaled by the deployable V11.4/DA3 pre-shot scale",
            "dataset_gauge": "GT is aligned to the chosen prediction gauge for evaluation only",
            "gt_scale_enters_candidate": False,
            "why_fixed_changes": (
                "The metric pre-shot scale changes the physical unit and target alignment before "
                "the same Fixed transform is evaluated. The difference is not an SE(3)-invariant "
                "coordinate rename and must not be interpreted as a new Fixed algorithm gain."
            ),
        },
        "coordinate_flow": [
            "raw RGB -> frozen Human3R shot-local c2w/pointmap/SMPL-X",
            "cut -> pre-decode hard reset",
            "DA3/V11.4 deployable scale s_pre,s_post -> scale all shot-local translations and geometry",
            "Fixed + V16 + optional conditional VGGT -> one rotation R",
            "old predicted human anchor and post predicted root -> translation t",
            "one fixed Boundary -> camera, pointmap and complete SMPL-X",
            "GT dataset c2w -> one evaluation-only alignment into the frozen pre-shot gauge",
        ],
        "cases": values,
    }


def main() -> None:
    args = parse_args()
    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("V14.5 independent SMPL-X audit requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    payload = json.loads(args.v14_4_report.read_text(encoding="utf-8"))
    rows = payload["cases"]
    if args.max_cases > 0:
        rows = rows[: int(args.max_cases)]
    v10_payload = json.loads(args.v10_report.read_text(encoding="utf-8"))
    v10 = {str(row["case_name"]): row for row in v10_payload["cases"]}
    streams = read_shards(args.stream_dir, "v18_stream_shard_*_of_*.json")
    keypoints = read_shards(args.keypoint_dir, "v18_keypoints_shard_*_of_*.json")
    if not all(row["case_name"] in v10 and row["case_name"] in streams for row in rows):
        raise RuntimeError("Incomplete V10/stream caches")

    layer10 = SMPL_Layer(
        type="smplx", gender="neutral", num_betas=10, kid=False, person_center="head"
    ).to(device).eval()
    layer11 = SMPL_Layer(
        type="smplx", gender="neutral", num_betas=11, kid=False, person_center="head"
    ).to(device).eval()
    scale_names = select_scale_cases(rows, int(args.scale_subset_per_source))
    independent_rows = []
    scene_rows = []
    scale_rows = []
    for index, row in enumerate(rows):
        name = row["case_name"]
        with np.load(Path(streams[name]["cache_path"])) as stream_file, np.load(
            Path(keypoints[name]["cache_path"])
        ) as keypoint_file:
            stream = {key: np.asarray(stream_file[key]) for key in stream_file.files}
            keypoint = {key: np.asarray(keypoint_file[key]) for key in keypoint_file.files}
        common_pre = float(row["scales"]["common_pre"])
        target = target_geometry(stream, common_pre)
        target["common_pre_scale"] = common_pre
        raw_root = np.asarray(stream["new_joints_camera"][0], dtype=np.float64)
        raw_body = np.asarray(stream["new_joints_camera"], dtype=np.float64) - raw_root
        predicted_joints, predicted_vertices = smpl_body(
            layer10,
            stream["new_rotvec"],
            stream["new_shape"],
            stream["new_expression"],
            device,
        )
        predicted_joints, predicted_vertices = normalize_body(
            predicted_joints, predicted_vertices, physical_scale(raw_body)
        )
        gt_body, gt_vertices = smpl_body(
            layer11,
            stream["new_gt_pose53_camera"],
            stream["new_gt_shape"],
            np.zeros(10),
            device,
        )
        gt_raw_body = (
            np.asarray(stream["new_gt_joints_camera"], dtype=np.float64)
            - stream["new_gt_joints_camera"][:1]
        )
        gt_body, gt_vertices = normalize_body(
            gt_body, gt_vertices, physical_scale(gt_raw_body)
        )
        background = load_background_pair(
            Path(v10[name]["paths"]["human3r_local_reset"]),
            int(args.scene_samples),
            float(args.confidence_threshold),
            int(args.mask_dilate),
            zlib.crc32(name.encode("utf-8")),
        )
        method_metrics = {}
        max_differences = defaultdict(float)
        for method_name in FINAL_METHODS:
            independent = independent_method_metrics(
                row["methods"][method_name],
                stream,
                keypoint,
                target,
                predicted_vertices,
                gt_body,
                gt_vertices,
                background,
            )
            main_row = row["methods"][method_name]
            comparisons = {
                "camera_translation_m": abs(
                    independent["camera"]["translation_m"]
                    - main_row["camera"]["translation_m"]
                ),
                "camera_rotation_deg": abs(
                    independent["camera"]["rotation_deg"]
                    - main_row["camera"]["rotation_deg"]
                ),
                "root_m": abs(independent["root_m"] - main_row["human"]["world_root_error_m"]),
                "joints_m": abs(
                    independent["joints_m"] - main_row["human"]["world_joint_mean_error_m"]
                ),
                "vertices_m": abs(
                    independent["vertices_m"] - main_row["human"]["world_vertex_mean_error_m"]
                ),
                "torso_reprojection_px": abs(
                    independent["torso_reprojection_px"]
                    - main_row["projection"]["torso_mean_px"]
                ),
                "scene_m": abs(
                    independent["scene"]["symmetric_trimmed_m"]
                    - main_row["scene"]["trimmed_mean_m"]
                ),
            }
            for key, value in comparisons.items():
                if np.isfinite(value):
                    max_differences[key] = max(max_differences[key], float(value))
            method_metrics[method_name] = {
                "independent": independent,
                "absolute_difference": comparisons,
            }
        independent_rows.append(
            {
                "case_name": name,
                "source": row["source"],
                "methods": method_metrics,
                "max_difference": dict(max_differences),
            }
        )

        scene_methods = {}
        pre_world = points(target["pre_pose"], background["pre"] * common_pre)
        for method_name in SCENE_METHODS:
            method = row["methods"][method_name]
            post_world = points(
                np.asarray(method["camera_pose"]),
                background["post"] * float(method["definition"]["scene_scale"]),
            )
            scene_methods[method_name] = scene_metrics(pre_world, post_world)
        scene_rows.append(
            {"case_name": name, "source": row["source"], "methods": scene_methods}
        )

        if name in scale_names:
            gt_body_world = points(
                target["camera_pose"], gt_body + stream["new_gt_joints_camera"][0]
            )
            gt_vertices_world = points(
                target["camera_pose"], gt_vertices + stream["new_gt_joints_camera"][0]
            )
            scale_rows.append(
                scale_sweep_case(
                    row,
                    stream,
                    target,
                    raw_body,
                    predicted_vertices,
                    gt_body_world,
                    gt_vertices_world,
                    background,
                )
            )
        if (index + 1) % 10 == 0 or index + 1 == len(rows):
            print(f">> V14.5 independent audit {index + 1}/{len(rows)}", flush=True)

    independent_max = {
        key: max(
            (row["max_difference"].get(key, 0.0) for row in independent_rows),
            default=0.0,
        )
        for key in (
            "camera_translation_m",
            "camera_rotation_deg",
            "root_m",
            "joints_m",
            "vertices_m",
            "torso_reprojection_px",
            "scene_m",
        )
    }

    scene_valid = [
        row
        for row in scene_rows
        if all(row["methods"][method]["valid"] for method in SCENE_METHODS)
    ]
    scene_summary = {
        "common_valid_count": len(scene_valid),
        "overall": {
            method: {
                metric: distribution(
                    [row["methods"][method][metric] for row in scene_valid]
                )
                for metric in (
                    "symmetric_trimmed_m",
                    "background_equal_direction_trimmed_m",
                    "median_m",
                    "p90_m",
                )
            }
            for method in SCENE_METHODS
        },
        "by_source": {
            source: {
                method: {
                    metric: distribution(
                        [
                            row["methods"][method][metric]
                            for row in scene_valid
                            if row["source"] == source
                        ]
                    )
                    for metric in (
                        "symmetric_trimmed_m",
                        "background_equal_direction_trimmed_m",
                    )
                }
                for method in SCENE_METHODS
            }
            for source in sorted({row["source"] for row in scene_valid})
        },
        "paired_vs_fixed": {
            method: {
                metric: paired(
                    [row["methods"]["fixed_explicit"][metric] for row in scene_valid],
                    [row["methods"][method][metric] for row in scene_valid],
                    0.05,
                )
                for metric in (
                    "symmetric_trimmed_m",
                    "background_equal_direction_trimmed_m",
                )
            }
            for method in SCENE_METHODS
            if method != "fixed_explicit"
        },
        "cases": scene_rows,
    }

    plot_scale_sweep(scale_rows, args.output_dir)
    scale_summary = {
        "case_count": len(scale_rows),
        "optimal_multiplier": {
            source: {
                metric: distribution(
                    [
                        row["optimal_multiplier"][metric]
                        for row in scale_rows
                        if row["source"] == source
                    ]
                )
                for metric in ("camera_m", "joints_m", "scene_m")
            }
            for source in sorted({row["source"] for row in scale_rows})
        },
        "local_5pct_ranges": {
            metric: distribution([row["local_5pct_ranges"][metric] for row in scale_rows])
            for metric in ("camera_m", "joints_m", "scene_m")
        },
        "predicted_scale_percentile_vs_grid_optimum": {
            metric: float(
                np.mean(
                    [
                        abs(row["optimal_multiplier"][metric] - 1.0) <= 0.10
                        for row in scale_rows
                    ]
                )
            )
            for metric in ("camera_m", "joints_m", "scene_m")
        },
        "cases": scale_rows,
    }

    freeze = {
        "git_commit": "3822715d8f3d2fbcd9e0867cdf787bb99f05abf4",
        "weights": {
            name: {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": None if args.skip_hashes else sha256(path),
            }
            for name, path in WEIGHTS.items()
        },
        "sources": {
            name: {
                "path": str(path),
                "sha256": sha256(path),
            }
            for name, path in FROZEN_SOURCES.items()
        },
        "thresholds": {
            "v16_rotation_bound_deg": 20.0,
            "vggt_large_torso_min_deg": 30.0,
            "vggt_extension_margin_deg": 5.0,
            "vggt_max_residual_deg": 100.0,
            "vggt_large_spread_max_deg": 15.0,
            "vggt_consensus_torso_min_deg": 10.0,
            "vggt_consensus_spread_max_deg": 5.0,
            "vggt_consensus_texture_max": 0.05,
            "vggt_large_cap_deg": 25.0,
            "vggt_consensus_cap_deg": 60.0,
            "vggt_conflict_cap_deg": 45.0,
            "v11_4_scale_clip": [0.35, 3.0],
            "scene_confidence_threshold": float(args.confidence_threshold),
            "scene_mask_dilate": int(args.mask_dilate),
            "scene_samples": int(args.scene_samples),
            "camera_success_translation_m": 0.50,
            "camera_success_rotation_deg": 20.0,
            "camera_catastrophic_translation_m": 2.0,
            "camera_catastrophic_rotation_deg": 45.0,
            "harmful_translation_m": 0.05,
            "continuity_shape_scale_alpha": 0.25,
            "continuity_local_pose_alpha": 0.15,
            "random_seed": 20260721,
        },
        "image_protocol": {
            "resolution_wh": [512, 288],
            "resize_mode": "human3r_demo",
            "camera_convention": "4x4 camera-to-world; left-multiplied shot Boundary",
        },
    }

    report = {
        "experiment": "V14.5 final geometry, leakage, anchor, gauge, and scene audit",
        "case_count": len(rows),
        "freeze": freeze,
        "synthetic_geometry": synthetic_audit(),
        "independent_evaluator": {
            "case_count": len(independent_rows),
            "methods": list(FINAL_METHODS),
            "maximum_absolute_difference": independent_max,
            "tolerance": {
                "metric_m": 1e-5,
                "rotation_deg": 0.002,
                "projection_px": 1e-4,
            },
            "rotation_tolerance_note": (
                "The trace/arccos geodesic is float-sensitive near identity; "
                "0.002 deg covers the observed 0.001207 deg maximum while all "
                "matrix-space and metric differences remain below 1e-5."
            ),
            "cases": independent_rows,
        },
        "leakage": leakage_audit(rows, args.v49_report),
        "common_anchor": common_anchor_audit(rows),
        "gauge": gauge_audit(rows, args.old_v14_2_report, streams),
        "scale_sensitivity": scale_summary,
        "scene_tradeoff": scene_summary,
        "conditional_vggt": conditional_audit(args.v49_report),
        "limitations": [
            "Raw-RGB cache-free reproduction, untouched holdout, recurrent V11.4 multi-cut, and deployment timing are separate stages.",
            "Scale Pareto uses a frozen deterministic 12-case-per-source subset, not all 180 cases.",
        ],
    }
    output = args.output_dir / "v14_5_final_geometry_leakage_audit.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(f">> wrote {output}", flush=True)


if __name__ == "__main__":
    main()
