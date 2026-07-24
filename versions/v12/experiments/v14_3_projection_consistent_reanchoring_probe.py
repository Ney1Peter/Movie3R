#!/usr/bin/env python3
"""V14.3 projection-consistent human-camera re-anchoring probe.

The same calibrated camera-frame root is used to solve camera translation and
to translate the complete post-cut SMPL-X body.  Human Projection and cached
DA3 metric depth are evaluated under identical V16 rotation and motion rules.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
import zlib
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[3]
for path in (str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from dust3r.utils.smpl_layer import SMPL_Layer  # noqa: E402
from versions.v12.experiments.v14_2_canonical_human_memory_probe import (  # noqa: E402
    TORSO_IDS,
    blend_rotations,
    physical_scale,
)


DEFAULT_V18 = ROOT / "output/v18_human_metric_translation"
DEFAULT_V14_2 = ROOT / "output/v14_2_canonical_human_memory/single_cut"
ALPHAS = (0.25, 0.50, 0.75, 1.00)
FOUR_QUADRANTS = (
    "v18_camera_only",
    "v18_camera_only_continuity",
    "v18_coupled_full",
    "v18_coupled_full_continuity",
    "da3_camera_only",
    "da3_camera_only_continuity",
    "da3_coupled_full",
    "da3_coupled_full_continuity",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream_dir", type=Path, default=DEFAULT_V18 / "stream_cache")
    parser.add_argument("--keypoint_dir", type=Path, default=DEFAULT_V18 / "keypoint_cache")
    parser.add_argument(
        "--v14_2_report",
        type=Path,
        default=DEFAULT_V14_2 / "v14_2_canonical_human_memory_probe.json",
    )
    parser.add_argument(
        "--da3_report",
        type=Path,
        default=DEFAULT_V18 / "da3_metric_depth/v18_da3_metric_depth_probe.json",
    )
    parser.add_argument(
        "--v10_report",
        type=Path,
        default=ROOT
        / "output/v10_candidate_selection/oracle_gt_4source/oracle_candidate_selection_metrics.json",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ROOT / "output/v14_3_projection_consistent_reanchoring/quantitative",
    )
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--scene_samples", type=int, default=1200)
    parser.add_argument("--confidence_threshold", type=float, default=1.5)
    parser.add_argument("--mask_dilate", type=int, default=11)
    parser.add_argument("--max_cases", type=int, default=0)
    return parser.parse_args()


def load_manifest(root: Path, pattern: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(str(root / pattern))):
        rows.extend(json.loads(Path(path).read_text(encoding="utf-8"))["cases"])
    return rows


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_cases(args: argparse.Namespace) -> list[dict]:
    streams = load_manifest(args.stream_dir, "v18_stream_shard_*_of_*.json")
    keypoints = {
        str(row["case_name"]): row
        for row in load_manifest(args.keypoint_dir, "v18_keypoints_shard_*_of_*.json")
    }
    v14_2_payload = json.loads(args.v14_2_report.read_text(encoding="utf-8"))
    v14_2 = {str(row["case_name"]): row for row in v14_2_payload["cases"]}
    da3_payload = json.loads(args.da3_report.read_text(encoding="utf-8"))
    da3 = {str(row["case_name"]): row for row in da3_payload["cases"]}
    v10_payload = json.loads(args.v10_report.read_text(encoding="utf-8"))
    v10 = {str(row["case_name"]): row for row in v10_payload["cases"]}
    if not streams or not all(len(table) == len(streams) for table in (keypoints, v14_2, da3, v10)):
        raise RuntimeError(
            f"Incomplete caches: stream/keypoint/v14.2/DA3/V10="
            f"{len(streams)}/{len(keypoints)}/{len(v14_2)}/{len(da3)}/{len(v10)}"
        )
    cases = []
    for row in streams:
        name = str(row["case_name"])
        cases.append(
            {
                **row,
                "cache_path": resolve(row["cache_path"]),
                "keypoint_path": resolve(keypoints[name]["cache_path"]),
                "v14_2": v14_2[name],
                "da3": da3[name],
                "local_dir": Path(v10[name]["paths"]["human3r_local_reset"]),
            }
        )
    cases.sort(key=lambda row: str(row["case_name"]))
    return cases[: args.max_cases] if args.max_cases > 0 else cases


def transform_point(pose: np.ndarray, point: np.ndarray) -> np.ndarray:
    return (pose[:3, :3] @ np.asarray(point) + pose[:3, 3]).astype(np.float32)


def transform_points(pose: np.ndarray, points: np.ndarray) -> np.ndarray:
    return (np.asarray(points) @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32)


def camera_pose_from_human(rotation: np.ndarray, world_root: np.ndarray, camera_root: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = np.asarray(rotation, dtype=np.float32)
    pose[:3, 3] = np.asarray(world_root) - pose[:3, :3] @ np.asarray(camera_root)
    return pose


def boundary_from_camera_pose(camera_pose: np.ndarray, local_pose: np.ndarray) -> np.ndarray:
    return (np.asarray(camera_pose) @ np.linalg.inv(np.asarray(local_pose))).astype(np.float32)


def root_from_pixel_depth(pixel: np.ndarray, depth: float, K: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            (float(pixel[0]) - float(K[0, 2])) * depth / float(K[0, 0]),
            (float(pixel[1]) - float(K[1, 2])) * depth / float(K[1, 1]),
            depth,
        ],
        dtype=np.float32,
    )


def detector_root_pixel(keypoints: np.ndarray, confidence: np.ndarray, threshold: float = 0.30) -> np.ndarray:
    if confidence[0] >= threshold and np.isfinite(keypoints[0]).all():
        return np.asarray(keypoints[0], dtype=np.float32)
    hips = [index for index in (1, 2) if confidence[index] >= threshold and np.isfinite(keypoints[index]).all()]
    if hips:
        return np.mean(keypoints[hips], axis=0).astype(np.float32)
    valid = np.flatnonzero((confidence >= threshold) & np.isfinite(keypoints).all(axis=1))
    return (
        np.mean(keypoints[valid], axis=0).astype(np.float32)
        if len(valid)
        else np.asarray([np.nan, np.nan], dtype=np.float32)
    )


def project(points: np.ndarray, K: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    z = np.maximum(points[:, 2], 1e-6)
    return np.stack(
        [K[0, 0] * points[:, 0] / z + K[0, 2], K[1, 1] * points[:, 1] / z + K[1, 2]],
        axis=1,
    )


def rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first)[:3, :3].T @ np.asarray(second)[:3, :3]
    return float(np.degrees(np.arccos(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))))


def evaluate_camera(camera_pose: np.ndarray, target_pose: np.ndarray) -> dict:
    delta = camera_pose[:3, 3] - target_pose[:3, 3]
    target_delta = target_pose[:3, :3].T @ delta
    return {
        "translation_m": float(np.linalg.norm(delta)),
        "rotation_deg": rotation_error_deg(camera_pose, target_pose),
        "viewing_direction_m": float(abs(target_delta[2])),
        "transverse_m": float(np.linalg.norm(target_delta[:2])),
        "vertical_m": float(abs(target_delta[1])),
        "translation_target_xyz_m": np.abs(target_delta).astype(float).tolist(),
    }


def body_from_params(
    layer: SMPL_Layer,
    pose: np.ndarray,
    shape: np.ndarray,
    expression: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    pose = np.asarray(pose, dtype=np.float32)[None]
    shape = np.asarray(shape, dtype=np.float32)[None]
    expression = np.asarray(expression, dtype=np.float32)[None]
    with torch.no_grad():
        output = layer(
            torch.from_numpy(pose).to(device),
            torch.from_numpy(shape).to(device),
            torch.zeros((1, 3), dtype=torch.float32, device=device),
            None,
            None,
            K=torch.eye(3, dtype=torch.float32, device=device)[None],
            expression=torch.from_numpy(expression).to(device),
        )
    joints = output["smpl_j3d"][0].detach().float().cpu().numpy().astype(np.float32)
    vertices = output["smpl_v3d"][0].detach().float().cpu().numpy().astype(np.float32)
    root = joints[0].copy()
    return (joints - root).astype(np.float32), (vertices - root).astype(np.float32)


def normalize_body_scale(joints: np.ndarray, vertices: np.ndarray, target: float) -> tuple[np.ndarray, np.ndarray]:
    scale = physical_scale(joints)
    factor = 1.0 if not np.isfinite(scale) or scale < 1e-8 else float(target) / float(scale)
    return (joints * factor).astype(np.float32), (vertices * factor).astype(np.float32)


def continuity_body(
    stream: dict,
    canonical_beta: np.ndarray,
    canonical_scale: float,
    layer: SMPL_Layer,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict]:
    current_pose = np.asarray(stream["new_rotvec"], dtype=np.float32)
    current_beta = np.asarray(stream["new_shape"], dtype=np.float32)
    beta = current_beta + 0.25 * (np.asarray(canonical_beta, dtype=np.float32) - current_beta)

    history = [Rotation.from_rotvec(pose[1:].reshape(-1, 3)).as_matrix() for pose in stream["old_rotvec"]]
    memory = history[0].astype(np.float32)
    for value in history[1:]:
        memory = blend_rotations(memory, value, 0.20)
    current_local = Rotation.from_rotvec(current_pose[1:].reshape(-1, 3)).as_matrix()
    blended_local = blend_rotations(current_local, memory, 0.15)
    pose = current_pose.copy()
    pose[1:] = Rotation.from_matrix(blended_local).as_rotvec().astype(np.float32)
    joints, vertices = body_from_params(
        layer, pose, beta, np.asarray(stream["new_expression"], dtype=np.float32), device
    )
    current_scale = physical_scale(np.asarray(stream["new_joints_camera"]) - stream["new_joints_camera"][:1])
    output_scale = current_scale + 0.25 * (float(canonical_scale) - current_scale)
    joints, vertices = normalize_body_scale(joints, vertices, output_scale)
    return joints, vertices, {
        "beta": beta,
        "physical_scale": float(output_scale),
        "pose": pose,
    }


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


def load_scene_pair(
    local_dir: Path,
    samples: int,
    confidence_threshold: float,
    mask_dilate: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    result = {}
    kernel = np.ones((mask_dilate, mask_dilate), dtype=np.uint8)
    for label, frame in (("pre", 1), ("post", 2)):
        with np.load(local_dir / "camera" / f"{frame:06d}.npz") as camera:
            pose = np.asarray(camera["pose"], dtype=np.float32)
            K = np.asarray(camera["intrinsics"], dtype=np.float32)
        with np.load(local_dir / "smpl" / f"{frame:06d}.npz", allow_pickle=True) as smpl:
            mask = np.asarray(smpl["msk"][0] > 0.10, dtype=np.uint8)
        if mask_dilate > 1:
            mask = cv2.dilate(mask, kernel, iterations=1)
        depth = np.load(local_dir / "depth" / f"{frame:06d}.npy").astype(np.float32)
        confidence = np.load(local_dir / "conf" / f"{frame:06d}.npy").astype(np.float32)
        if mask.shape != depth.shape:
            mask = cv2.resize(mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
        if confidence.shape != depth.shape:
            confidence = cv2.resize(
                confidence, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_LINEAR
            )
        valid = (
            np.isfinite(depth)
            & np.isfinite(confidence)
            & (depth > 0.10)
            & (depth < 30.0)
            & (confidence > confidence_threshold)
            & (mask == 0)
        )
        ids = np.flatnonzero(valid.reshape(-1))
        if len(ids) > samples:
            ids = rng.choice(ids, size=samples, replace=False)
        points_camera = camera_points(depth, K).reshape(-1, 3)[ids].astype(np.float32)
        result[label] = {
            "camera": points_camera,
            "world": transform_points(pose, points_camera),
            "pose": pose,
        }
    return result


def scene_discontinuity(pre_world: np.ndarray, post_world: np.ndarray) -> dict:
    if not len(pre_world) or not len(post_world):
        return {"median_m": float("nan"), "p90_m": float("nan"), "trimmed_mean_m": float("nan")}
    forward = cKDTree(pre_world).query(post_world, k=1, workers=-1)[0]
    backward = cKDTree(post_world).query(pre_world, k=1, workers=-1)[0]
    distance = np.concatenate([forward, backward])
    cutoff = np.percentile(distance, 80)
    trimmed = distance[distance <= cutoff]
    return {
        "median_m": float(np.median(distance)),
        "p90_m": float(np.percentile(distance, 90)),
        "trimmed_mean_m": float(np.mean(trimmed)),
    }


def reproject_error(
    body: np.ndarray,
    root: np.ndarray,
    observed: np.ndarray,
    confidence: np.ndarray,
    K: np.ndarray,
) -> dict:
    count = min(len(body), len(observed))
    points = body[:count] + np.asarray(root)[None]
    valid = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(observed[:count]).all(axis=1)
        & (points[:, 2] > 0.05)
        & (confidence[:count] >= 0.30)
    )
    errors = np.linalg.norm(project(points[valid], K) - observed[:count][valid], axis=1)
    torso_mask = np.asarray([index in set(TORSO_IDS.tolist()) for index in np.arange(count)])[valid]
    torso = errors[torso_mask]
    return {
        "mean_px": float(np.mean(errors)) if len(errors) else float("nan"),
        "median_px": float(np.median(errors)) if len(errors) else float("nan"),
        "torso_mean_px": float(np.mean(torso)) if len(torso) else float("nan"),
        "torso_median_px": float(np.median(torso)) if len(torso) else float("nan"),
    }


def mesh_bbox_metrics(
    vertices: np.ndarray,
    root: np.ndarray,
    target_box: np.ndarray,
    K: np.ndarray,
) -> dict:
    points = np.asarray(vertices) + np.asarray(root)[None]
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0.05)
    projected = project(points[valid], K)
    if not len(projected):
        return {"iou": float("nan"), "width_ratio": float("nan"), "height_ratio": float("nan")}
    predicted = np.asarray(
        [projected[:, 0].min(), projected[:, 1].min(), projected[:, 0].max(), projected[:, 1].max()],
        dtype=np.float64,
    )
    target = np.asarray(target_box, dtype=np.float64).reshape(-1)[:4]
    if len(target) != 4 or not np.isfinite(target).all():
        return {"iou": float("nan"), "width_ratio": float("nan"), "height_ratio": float("nan")}
    intersection_min = np.maximum(predicted[:2], target[:2])
    intersection_max = np.minimum(predicted[2:], target[2:])
    intersection_size = np.maximum(intersection_max - intersection_min, 0.0)
    intersection = float(np.prod(intersection_size))
    predicted_size = np.maximum(predicted[2:] - predicted[:2], 1e-6)
    target_size = np.maximum(target[2:] - target[:2], 1e-6)
    union = float(np.prod(predicted_size) + np.prod(target_size) - intersection)
    return {
        "iou": intersection / union if union > 0 else float("nan"),
        "width_ratio": float(predicted_size[0] / target_size[0]),
        "height_ratio": float(predicted_size[1] / target_size[1]),
    }


def method_spec(
    name: str,
    camera_pose: np.ndarray,
    root_camera: np.ndarray,
    body: str = "raw",
    body_scale: float = 1.0,
) -> dict:
    return {
        "name": name,
        "camera_pose": np.asarray(camera_pose, dtype=np.float32),
        "root_camera": np.asarray(root_camera, dtype=np.float32),
        "body": body,
        "body_scale": float(body_scale),
    }


def coupled_pose(rotation: np.ndarray, world_root: np.ndarray, camera_root: np.ndarray) -> np.ndarray:
    return camera_pose_from_human(rotation, world_root, camera_root)


def run_case(
    case: dict,
    layer10: SMPL_Layer,
    layer11: SMPL_Layer,
    foot_indices: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
) -> dict:
    with np.load(case["cache_path"]) as stream_file, np.load(case["keypoint_path"]) as keypoint_file:
        stream = {name: np.asarray(stream_file[name]) for name in stream_file.files}
        keypoint = {name: np.asarray(keypoint_file[name]) for name in keypoint_file.files}

    raw_root = np.asarray(stream["new_joints_camera"][0], dtype=np.float32)
    raw_body = np.asarray(stream["new_joints_camera"], dtype=np.float32) - raw_root[None]
    predicted_joints, predicted_vertices = body_from_params(
        layer10,
        stream["new_rotvec"],
        stream["new_shape"],
        stream["new_expression"],
        device,
    )
    # Cached joints are the authoritative Human3R output; generated vertices share its root and body scale.
    predicted_scale = physical_scale(raw_body)
    predicted_joints, predicted_vertices = normalize_body_scale(
        predicted_joints, predicted_vertices, predicted_scale
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
    gt_scale = physical_scale(
        np.asarray(stream["new_gt_joints_camera"]) - stream["new_gt_joints_camera"][:1]
    )
    gt_joints, gt_vertices = normalize_body_scale(gt_joints, gt_vertices, gt_scale)
    gt_world_root = np.asarray(stream["new_gt_joints_target_world"][0], dtype=np.float32)
    gt_joints_world = transform_points(stream["target_pose"], gt_joints + stream["new_gt_joints_camera"][0])
    gt_vertices_world = transform_points(stream["target_pose"], gt_vertices + stream["new_gt_joints_camera"][0])

    v18_root = np.asarray(
        case["v14_2"]["candidates"]["current_independent"]["human_projection"]["new_camera_root"],
        dtype=np.float32,
    )
    v18_boundary = np.asarray(
        case["v14_2"]["candidates"]["current_independent"]["transform"], dtype=np.float32
    )
    v18_pose = v18_boundary @ stream["new_pose"]
    v18_world = transform_point(v18_pose, v18_root)

    da3_depth = float(case["da3"]["depth"]["da3_pelvis_m"])
    da3_pixel = detector_root_pixel(keypoint["new_keypoints"], keypoint["new_confidence"])
    da3_root = root_from_pixel_depth(da3_pixel, da3_depth, stream["new_intrinsics"])
    da3_boundary = np.asarray(case["da3"]["candidates"]["da3_pelvis_depth"]["transform"], dtype=np.float32)
    da3_pose = da3_boundary @ stream["new_pose"]
    da3_world = transform_point(da3_pose, da3_root)

    fixed_pose = np.asarray(stream["fixed_transform"], dtype=np.float32) @ stream["new_pose"]
    target_pose = np.asarray(stream["target_pose"], dtype=np.float32)
    gt_boundary_pose = np.asarray(stream["gt_boundary"], dtype=np.float32) @ stream["new_pose"]
    rotation = v18_pose[:3, :3]
    if np.max(np.abs(rotation - da3_pose[:3, :3])) > 1e-5:
        raise RuntimeError(f"V18 and DA3 rotations differ for {case['case_name']}")

    v18_depth_root = raw_root.copy()
    v18_depth_root[2] = v18_root[2]
    da3_depth_root = raw_root.copy()
    da3_depth_root[2] = da3_root[2]
    gt_depth_root = raw_root.copy()
    gt_depth_root[2] = stream["new_gt_joints_camera"][0, 2]
    human_fusion_root = 0.5 * (v18_root + da3_root)
    human_fusion_world = 0.5 * (v18_world + da3_world)

    methods: dict[str, dict] = {
        "fixed_explicit": method_spec("fixed_explicit", fixed_pose, raw_root),
        "v18_camera_only": method_spec("v18_camera_only", v18_pose, raw_root),
        "v18_camera_only_continuity": method_spec(
            "v18_camera_only_continuity", v18_pose, raw_root, "continuity"
        ),
        "human_only_projection_depth": method_spec(
            "human_only_projection_depth", fixed_pose, v18_depth_root
        ),
        "v18_coupled_depth_only": method_spec(
            "v18_coupled_depth_only",
            coupled_pose(rotation, v18_world, v18_depth_root),
            v18_depth_root,
        ),
        "v18_coupled_full": method_spec("v18_coupled_full", v18_pose, v18_root),
        "v18_coupled_projective_body": method_spec(
            "v18_coupled_projective_body",
            v18_pose,
            v18_root,
            body_scale=float(v18_root[2] / raw_root[2]),
        ),
        "v18_coupled_full_continuity": method_spec(
            "v18_coupled_full_continuity", v18_pose, v18_root, "continuity"
        ),
        "da3_camera_only": method_spec("da3_camera_only", da3_pose, raw_root),
        "da3_camera_only_continuity": method_spec(
            "da3_camera_only_continuity", da3_pose, raw_root, "continuity"
        ),
        "da3_coupled_depth_only": method_spec(
            "da3_coupled_depth_only",
            coupled_pose(rotation, da3_world, da3_depth_root),
            da3_depth_root,
        ),
        "da3_coupled_full": method_spec("da3_coupled_full", da3_pose, da3_root),
        "da3_coupled_projective_body": method_spec(
            "da3_coupled_projective_body",
            da3_pose,
            da3_root,
            body_scale=float(da3_root[2] / raw_root[2]),
        ),
        "da3_coupled_full_continuity": method_spec(
            "da3_coupled_full_continuity", da3_pose, da3_root, "continuity"
        ),
        "human_da3_fusion_coupled": method_spec(
            "human_da3_fusion_coupled",
            coupled_pose(rotation, human_fusion_world, human_fusion_root),
            human_fusion_root,
        ),
        "gt_root_depth_coupled": method_spec(
            "gt_root_depth_coupled",
            coupled_pose(rotation, v18_world, gt_depth_root),
            gt_depth_root,
        ),
        "gt_depth_motion_coupled": method_spec(
            "gt_depth_motion_coupled",
            coupled_pose(rotation, gt_world_root, gt_depth_root),
            gt_depth_root,
        ),
        "boundary_oracle": method_spec("boundary_oracle", gt_boundary_pose, raw_root),
        "boundary_oracle_continuity": method_spec(
            "boundary_oracle_continuity", gt_boundary_pose, raw_root, "continuity"
        ),
    }
    for cue, calibrated, world in (
        ("v18", v18_root, v18_world),
        ("da3", da3_root, da3_world),
    ):
        for alpha in ALPHAS:
            root = raw_root + float(alpha) * (calibrated - raw_root)
            name = f"{cue}_coupled_alpha_{alpha:.2f}".replace(".", "p")
            methods[name] = method_spec(name, coupled_pose(rotation, world, root), root)
            projective_name = f"{cue}_projective_alpha_{alpha:.2f}".replace(".", "p")
            methods[projective_name] = method_spec(
                projective_name,
                coupled_pose(rotation, world, root),
                root,
                body_scale=float(root[2] / raw_root[2]),
            )

    scene = load_scene_pair(
        case["local_dir"],
        int(args.scene_samples),
        float(args.confidence_threshold),
        int(args.mask_dilate),
        zlib.crc32(str(case["case_name"]).encode("utf-8")),
    )
    pre_display_root = transform_point(stream["old_pose"][-1], stream["old_joints_camera"][-1, 0])
    gt_delta = gt_world_root - stream["old_gt_joints_target_world"][-1, 0]
    gt_root_rotation = target_pose[:3, :3] @ Rotation.from_rotvec(
        stream["new_gt_pose53_camera"][0]
    ).as_matrix()

    evaluated = {}
    for name, spec in methods.items():
        camera_pose = spec["camera_pose"]
        root_camera = spec["root_camera"]
        body_joints = continuity_joints if spec["body"] == "continuity" else raw_body
        body_vertices = continuity_vertices if spec["body"] == "continuity" else predicted_vertices
        body_joints = body_joints * float(spec["body_scale"])
        body_vertices = body_vertices * float(spec["body_scale"])
        root_world = transform_point(camera_pose, root_camera)
        joints_world = body_joints @ camera_pose[:3, :3].T + root_world
        vertices_world = body_vertices @ camera_pose[:3, :3].T + root_world
        predicted_delta = root_world - pre_display_root
        local_root_rotation = Rotation.from_rotvec(
            continuity["pose"][0] if spec["body"] == "continuity" else stream["new_rotvec"][0]
        ).as_matrix()
        world_root_rotation = camera_pose[:3, :3] @ local_root_rotation
        feet_camera = body_joints[foot_indices] + root_camera
        foot_distances = cKDTree(scene["post"]["camera"]).query(feet_camera, k=1, workers=-1)[0]
        post_scene_world = transform_points(
            boundary_from_camera_pose(camera_pose, stream["new_pose"]), scene["post"]["world"]
        )
        # scene['post']['world'] is in the fresh local world; Boundary maps it to the pre-cut gauge.
        scene_row = scene_discontinuity(scene["pre"]["world"], post_scene_world)
        camera_row = evaluate_camera(camera_pose, target_pose)
        human_root_error = float(np.linalg.norm(root_world - gt_world_root))
        evaluated[name] = {
            "camera": camera_row,
            "human": {
                "world_root_error_m": human_root_error,
                "world_root": root_world.astype(float).tolist(),
                "root_jump_m": float(np.linalg.norm(predicted_delta)),
                "root_jump_residual_m": float(np.linalg.norm(predicted_delta - gt_delta)),
                "world_joint_mean_error_m": float(
                    np.mean(np.linalg.norm(joints_world - gt_joints_world[: len(joints_world)], axis=1))
                ),
                "world_joint_p90_error_m": float(
                    np.percentile(
                        np.linalg.norm(joints_world - gt_joints_world[: len(joints_world)], axis=1), 90
                    )
                ),
                "world_vertex_mean_error_m": float(
                    np.mean(np.linalg.norm(vertices_world - gt_vertices_world, axis=1))
                ),
                "global_orientation_error_deg": rotation_error_deg(
                    np.block(
                        [
                            [world_root_rotation, np.zeros((3, 1))],
                            [np.zeros((1, 3)), np.ones((1, 1))],
                        ]
                    ),
                    np.block(
                        [
                            [gt_root_rotation, np.zeros((3, 1))],
                            [np.zeros((1, 3)), np.ones((1, 1))],
                        ]
                    ),
                ),
                "camera_root": root_camera.astype(float).tolist(),
                "camera_root_depth_error_m": float(
                    abs(root_camera[2] - stream["new_gt_joints_camera"][0, 2])
                ),
            },
            "projection": reproject_error(
                body_joints,
                root_camera,
                keypoint["new_keypoints"],
                keypoint["new_confidence"],
                stream["new_intrinsics"],
            ),
            "scene": {
                **scene_row,
                "foot_nearest_mean_m": float(np.mean(foot_distances)),
                "foot_nearest_max_m": float(np.max(foot_distances)),
                "local_pointmap_max_diff_vs_hard_reset": 0.0,
            },
            "joint_success": bool(camera_row["translation_m"] < 0.50 and human_root_error < 0.50),
            "strict_joint_success": bool(
                camera_row["translation_m"] < 0.25 and human_root_error < 0.25
            ),
            "transform": boundary_from_camera_pose(camera_pose, stream["new_pose"])
            .astype(float)
            .tolist(),
            "root_correction_m": float(np.linalg.norm(root_camera - raw_root)),
            "body_variant": spec["body"],
            "body_scale_factor": float(spec["body_scale"]),
        }
        evaluated[name]["projection"]["mesh_bbox"] = mesh_bbox_metrics(
            body_vertices,
            root_camera,
            keypoint["new_box"],
            stream["new_intrinsics"],
        )

    closure = {
        name: float(
            np.linalg.norm(
                transform_point(spec["camera_pose"], spec["root_camera"])
                - (
                    v18_world
                    if name.startswith("v18_coupled") or name == "gt_root_depth_coupled"
                    else da3_world
                    if name.startswith("da3_coupled")
                    else human_fusion_world
                    if name == "human_da3_fusion_coupled"
                    else gt_world_root
                    if name == "gt_depth_motion_coupled"
                    else transform_point(spec["camera_pose"], spec["root_camera"])
                )
            )
        )
        for name, spec in methods.items()
        if "coupled" in name
    }
    return {
        "case_name": case["case_name"],
        "source": case["source"],
        "record": case["record"],
        "roots": {
            "raw_camera": raw_root.astype(float).tolist(),
            "v18_calibrated_camera": v18_root.astype(float).tolist(),
            "v18_predicted_world": v18_world.astype(float).tolist(),
            "da3_calibrated_camera": da3_root.astype(float).tolist(),
            "da3_predicted_world": da3_world.astype(float).tolist(),
            "gt_camera": stream["new_gt_joints_camera"][0].astype(float).tolist(),
            "gt_world": gt_world_root.astype(float).tolist(),
        },
        "continuity": {
            "beta": continuity["beta"].astype(float).tolist(),
            "physical_scale": continuity["physical_scale"],
        },
        "coupled_closure_max_m": float(max(closure.values())),
        "methods": evaluated,
    }


def distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    total = len(array)
    array = array[np.isfinite(array)]
    if not len(array):
        return {
            "count": 0,
            "non_finite_count": total,
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
        }
    return {
        "count": int(len(array)),
        "non_finite_count": int(total - len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
    }


def aggregate_method(cases: list[dict], name: str) -> dict:
    rows = [case["methods"][name] for case in cases]
    fixed = [case["methods"]["fixed_explicit"] for case in cases]
    camera_errors = np.asarray(
        [row["camera"]["translation_m"] for row in rows], dtype=np.float64
    )
    human_errors = np.asarray(
        [row["human"]["world_root_error_m"] for row in rows], dtype=np.float64
    )
    return {
        "count": len(rows),
        "camera_translation_m": distribution([row["camera"]["translation_m"] for row in rows]),
        "camera_rotation_deg": distribution([row["camera"]["rotation_deg"] for row in rows]),
        "camera_view_m": distribution([row["camera"]["viewing_direction_m"] for row in rows]),
        "human_root_m": distribution([row["human"]["world_root_error_m"] for row in rows]),
        "human_joint_m": distribution([row["human"]["world_joint_mean_error_m"] for row in rows]),
        "human_vertex_m": distribution([row["human"]["world_vertex_mean_error_m"] for row in rows]),
        "root_jump_residual_m": distribution([row["human"]["root_jump_residual_m"] for row in rows]),
        "root_depth_m": distribution([row["human"]["camera_root_depth_error_m"] for row in rows]),
        "orientation_deg": distribution(
            [row["human"]["global_orientation_error_deg"] for row in rows]
        ),
        "reprojection_torso_px": distribution([row["projection"]["torso_mean_px"] for row in rows]),
        "reprojection_full_px": distribution([row["projection"]["mean_px"] for row in rows]),
        "mesh_bbox_iou": distribution([row["projection"]["mesh_bbox"]["iou"] for row in rows]),
        "mesh_bbox_width_ratio": distribution(
            [row["projection"]["mesh_bbox"]["width_ratio"] for row in rows]
        ),
        "mesh_bbox_height_ratio": distribution(
            [row["projection"]["mesh_bbox"]["height_ratio"] for row in rows]
        ),
        "scene_discontinuity_m": distribution([row["scene"]["trimmed_mean_m"] for row in rows]),
        "foot_scene_m": distribution([row["scene"]["foot_nearest_mean_m"] for row in rows]),
        "root_correction_m": distribution([row["root_correction_m"] for row in rows]),
        "camera_catastrophic_rate": float(
            np.mean([row["camera"]["translation_m"] > 1.0 for row in rows])
        ),
        "joint_success_rate": float(np.mean([row["joint_success"] for row in rows])),
        "strict_joint_success_rate": float(np.mean([row["strict_joint_success"] for row in rows])),
        "harmful_camera_rate_vs_fixed": float(
            np.mean(
                [
                    row["camera"]["translation_m"] > base["camera"]["translation_m"] + 0.05
                    for row, base in zip(rows, fixed)
                ]
            )
        ),
        "harmful_human_rate_vs_fixed": float(
            np.mean(
                [
                    row["human"]["world_root_error_m"] > base["human"]["world_root_error_m"] + 0.05
                    for row, base in zip(rows, fixed)
                ]
            )
        ),
        "camera_better_human_worse_vs_fixed_rate": float(
            np.mean(
                [
                    row["camera"]["translation_m"] < base["camera"]["translation_m"] - 0.05
                    and row["human"]["world_root_error_m"] > base["human"]["world_root_error_m"] + 0.05
                    for row, base in zip(rows, fixed)
                ]
            )
        ),
        "camera_human_error_pearson": safe_pearson(camera_errors, human_errors),
    }


def safe_pearson(first: np.ndarray, second: np.ndarray) -> float:
    valid = np.isfinite(first) & np.isfinite(second)
    first = first[valid]
    second = second[valid]
    if len(first) < 2 or np.std(first) < 1e-12 or np.std(second) < 1e-12:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def aggregate(cases: list[dict]) -> dict:
    names = tuple(cases[0]["methods"])
    return {name: aggregate_method(cases, name) for name in names}


def comparison_rates(cases: list[dict], camera_only: str, coupled: str) -> dict:
    camera_rows = [case["methods"][camera_only] for case in cases]
    coupled_rows = [case["methods"][coupled] for case in cases]
    camera_delta = np.asarray(
        [b["camera"]["translation_m"] - a["camera"]["translation_m"] for a, b in zip(camera_rows, coupled_rows)]
    )
    human_delta = np.asarray(
        [b["human"]["world_root_error_m"] - a["human"]["world_root_error_m"] for a, b in zip(camera_rows, coupled_rows)]
    )
    reprojection_delta = finite_pair_delta(
        [b["projection"]["torso_mean_px"] - a["projection"]["torso_mean_px"] for a, b in zip(camera_rows, coupled_rows)]
    )
    scene_delta = finite_pair_delta(
        [b["scene"]["trimmed_mean_m"] - a["scene"]["trimmed_mean_m"] for a, b in zip(camera_rows, coupled_rows)]
    )
    foot_delta = finite_pair_delta(
        [b["scene"]["foot_nearest_mean_m"] - a["scene"]["foot_nearest_mean_m"] for a, b in zip(camera_rows, coupled_rows)]
    )
    return {
        "camera_mean_delta_m": float(np.mean(camera_delta)),
        "human_root_mean_delta_m": float(np.mean(human_delta)),
        "human_improvement_rate": float(np.mean(human_delta < -0.05)),
        "human_harm_rate": float(np.mean(human_delta > 0.05)),
        "torso_reprojection_mean_delta_px": finite_mean(reprojection_delta),
        "scene_discontinuity_mean_delta_m": finite_mean(scene_delta),
        "foot_scene_mean_delta_m": finite_mean(foot_delta),
        "finite_reprojection_pairs": int(len(reprojection_delta)),
        "finite_scene_pairs": int(len(scene_delta)),
        "finite_foot_pairs": int(len(foot_delta)),
    }


def finite_pair_delta(values: list[float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def finite_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else float("nan")


def method_tradeoff(cases: list[dict], first: str, second: str) -> dict:
    """Report second-minus-first paired deltas for the joint geometry objectives."""

    first_rows = [case["methods"][first] for case in cases]
    second_rows = [case["methods"][second] for case in cases]
    fields = {
        "camera_translation_m": lambda row: row["camera"]["translation_m"],
        "human_root_m": lambda row: row["human"]["world_root_error_m"],
        "human_joint_m": lambda row: row["human"]["world_joint_mean_error_m"],
        "torso_reprojection_px": lambda row: row["projection"]["torso_mean_px"],
        "scene_discontinuity_m": lambda row: row["scene"]["trimmed_mean_m"],
        "foot_scene_m": lambda row: row["scene"]["foot_nearest_mean_m"],
    }
    result = {"first": first, "second": second, "delta_definition": "second - first"}
    for label, getter in fields.items():
        delta = finite_pair_delta(
            [getter(second_row) - getter(first_row) for first_row, second_row in zip(first_rows, second_rows)]
        )
        result[label] = {
            "mean_delta": finite_mean(delta),
            "improvement_rate": float(np.mean(delta < -0.05)) if len(delta) else float("nan"),
            "harm_rate": float(np.mean(delta > 0.05)) if len(delta) else float("nan"),
            "finite_pairs": int(len(delta)),
        }
    return result


def write_markdown(path: Path, report: dict) -> None:
    names = (
        "fixed_explicit",
        "v18_camera_only",
        "human_only_projection_depth",
        "v18_coupled_depth_only",
        "v18_coupled_full",
        "v18_coupled_projective_body",
        "v18_coupled_full_continuity",
        "da3_camera_only",
        "da3_coupled_depth_only",
        "da3_coupled_full",
        "da3_coupled_projective_body",
        "da3_coupled_full_continuity",
        "human_da3_fusion_coupled",
        "gt_root_depth_coupled",
        "gt_depth_motion_coupled",
        "boundary_oracle",
    )
    lines = [
        "# V14.3 Projection-Consistent Re-anchoring",
        "",
        "| Method | Camera T | T P90 | Human root | Joints | Vertices | Joint success | Reproj torso | Scene | Foot-scene |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        row = report["overall"][name]
        lines.append(
            "| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.1f}% | {:.1f} | {:.3f} | {:.3f} |".format(
                name,
                row["camera_translation_m"]["mean"],
                row["camera_translation_m"]["p90"],
                row["human_root_m"]["mean"],
                row["human_joint_m"]["mean"],
                row["human_vertex_m"]["mean"],
                100.0 * row["joint_success_rate"],
                row["reprojection_torso_px"]["mean"],
                row["scene_discontinuity_m"]["mean"],
                row["foot_scene_m"]["mean"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("V14.3 SMPL-X body/vertex evaluation requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args)
    device = torch.device(args.device)
    layer10 = SMPL_Layer(
        type="smplx", gender="neutral", num_betas=10, kid=False, person_center="head"
    ).to(device).eval()
    layer11 = SMPL_Layer(
        type="smplx", gender="neutral", num_betas=11, kid=False, person_center="head"
    ).to(device).eval()
    names = layer10.joint_names
    foot_indices = np.asarray(
        [
            names.index("left_big_toe"),
            names.index("left_small_toe"),
            names.index("left_heel"),
            names.index("right_big_toe"),
            names.index("right_small_toe"),
            names.index("right_heel"),
        ],
        dtype=np.int64,
    )
    rows = []
    started = time.perf_counter()
    for index, case in enumerate(cases):
        rows.append(run_case(case, layer10, layer11, foot_indices, device, args))
        if (index + 1) % 10 == 0 or index + 1 == len(cases):
            print(f">> V14.3 coupled {index + 1}/{len(cases)}", flush=True)

    overall = aggregate(rows)
    sources = sorted({str(row["source"]) for row in rows})
    report = {
        "experiment": "V14.3 Projection-Consistent Human-Camera Re-anchoring",
        "case_count": len(rows),
        "elapsed_seconds": time.perf_counter() - started,
        "protocol": {
            "human3r_frozen": True,
            "da3_inference": "cached frozen DA3Metric-Large output; originally run on 5 pre + 1 post frames",
            "rotation": "same V16 torso-motion rotation, global 20 degree bound",
            "coupled_definition": "same calibrated camera root solves camera translation and translates all joints/vertices",
            "pointmap_modified": False,
            "continuity_shape_scale_alpha": 0.25,
            "continuity_local_pose_alpha": 0.15,
            "max_humans": 1,
        },
        "coupled_closure_max_m": float(max(row["coupled_closure_max_m"] for row in rows)),
        "overall": overall,
        "by_source": {
            source: aggregate([row for row in rows if row["source"] == source]) for source in sources
        },
        "camera_only_vs_coupled": {
            "v18": comparison_rates(rows, "v18_camera_only", "v18_coupled_full"),
            "da3": comparison_rates(rows, "da3_camera_only", "da3_coupled_full"),
        },
        "method_tradeoffs": {
            "v18_depth_only_to_full_root": method_tradeoff(
                rows, "v18_coupled_depth_only", "v18_coupled_full"
            ),
            "da3_depth_only_to_full_root": method_tradeoff(
                rows, "da3_coupled_depth_only", "da3_coupled_full"
            ),
            "da3_alpha_0p75_to_1p00": method_tradeoff(
                rows, "da3_coupled_alpha_0p75", "da3_coupled_alpha_1p00"
            ),
            "v18_full_to_da3_full": method_tradeoff(
                rows, "v18_coupled_full", "da3_coupled_full"
            ),
            "da3_full_to_fixed_fusion": method_tradeoff(
                rows, "da3_coupled_full", "human_da3_fusion_coupled"
            ),
            "v18_rigid_body_to_projective_body": method_tradeoff(
                rows, "v18_coupled_full", "v18_coupled_projective_body"
            ),
            "da3_rigid_body_to_projective_body": method_tradeoff(
                rows, "da3_coupled_full", "da3_coupled_projective_body"
            ),
            "da3_alpha_0p75_rigid_to_projective_body": method_tradeoff(
                rows, "da3_coupled_alpha_0p75", "da3_projective_alpha_0p75"
            ),
        },
        "four_quadrants": {name: overall[name] for name in FOUR_QUADRANTS},
        "cases": rows,
    }
    output = args.output_dir / "v14_3_projection_consistent_reanchoring.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    write_markdown(args.output_dir / "v14_3_projection_consistent_reanchoring.md", report)
    print(f">> wrote {output}", flush=True)


if __name__ == "__main__":
    main()
