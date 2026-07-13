#!/usr/bin/env python3
"""Floor-locked cached segment alignment for saved Human3R outputs.

This is a diagnostic V10 probe.  It assumes an oracle boundary and computes one
cached transform for the post-boundary segment:

1. align the boundary-frame floor normal to the history floor normal;
2. solve only yaw + in-plane translation from human anchors;
3. optionally add normal-axis translation from floor centers or human centroids;
4. apply the same transform to all post-boundary frames.

The purpose is to test whether the visible A/B tilt comes from unconstrained
human full-SE3 alignment.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
ARCHIVE_V7 = SCRIPTS_ROOT / "archive_v7"
for path in (str(REPO_ROOT), str(SRC_ROOT), str(SCRIPTS_ROOT), str(ARCHIVE_V7)):
    if path not in sys.path:
        sys.path.insert(0, path)

from overfit_human_anchor_pose_correction import FOOT_JOINTS, STABLE_JOINTS, load_sequence
from v10_static_alignment_4source_probe import body_frame_from_joints
from v9_learned_stream_alignment_overfit import rotation_geodesic
from v9_segment_human3r_yaw_align_probe import copy_np_payload, copy_smpl, save_camera


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--floor_json", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--boundary", type=int, default=2)
    parser.add_argument(
        "--normal_translation_source",
        choices=["none", "floor_center", "human_centroid"],
        default="floor_center",
    )
    parser.add_argument("--stable_weight", type=float, default=1.0)
    parser.add_argument("--foot_weight", type=float, default=2.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / max(float(np.linalg.norm(v)), eps)


def angle_deg(a: np.ndarray, b: np.ndarray, unsigned: bool = True) -> float:
    a = normalize(a)
    b = normalize(b)
    dot = float(np.dot(a, b))
    if unsigned:
        dot = abs(dot)
    return math.degrees(math.acos(float(np.clip(dot, -1.0, 1.0))))


def rotation_angle_deg(R: np.ndarray) -> float:
    cos = np.clip((float(np.trace(R)) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(cos))


def rotation_from_vectors(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src = normalize(src)
    dst = normalize(dst)
    dot = float(np.clip(np.dot(src, dst), -1.0, 1.0))
    if dot > 1.0 - 1e-10:
        return np.eye(3, dtype=np.float64)
    if dot < -1.0 + 1e-10:
        axis = np.cross(src, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        if np.linalg.norm(axis) < 1e-8:
            axis = np.cross(src, np.array([0.0, 1.0, 0.0], dtype=np.float64))
        axis = normalize(axis)
        K = skew(axis)
        return np.eye(3, dtype=np.float64) + 2.0 * (K @ K)
    axis = np.cross(src, dst)
    angle = math.acos(dot)
    return rotation_about_axis(axis, angle)


def skew(axis: np.ndarray) -> np.ndarray:
    x, y, z = normalize(axis)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def rotation_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    K = skew(axis)
    return np.eye(3, dtype=np.float64) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def make_plane_basis(normal: np.ndarray) -> np.ndarray:
    n = normalize(normal)
    seed = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(seed, n))) > 0.9:
        seed = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    u = normalize(seed - n * float(np.dot(seed, n)))
    v = normalize(np.cross(n, u))
    return np.stack([u, v], axis=0)


def project_to_plane(v: np.ndarray, normal: np.ndarray) -> np.ndarray:
    n = normalize(normal)
    return v - np.sum(v * n, axis=-1, keepdims=True) * n


def orient_like(normal: np.ndarray, ref: np.ndarray) -> np.ndarray:
    normal = normalize(normal)
    ref = normalize(ref)
    return normal if float(np.dot(normal, ref)) >= 0.0 else -normal


def mean_oriented(vectors: list[np.ndarray]) -> np.ndarray:
    ref = normalize(vectors[0])
    oriented = [orient_like(v, ref) for v in vectors]
    return normalize(np.mean(np.stack(oriented), axis=0))


def load_floor_planes(path: Path) -> dict[int, dict]:
    debug = json.loads(path.read_text())
    return {int(p["viewer_frame"]): p for p in debug.get("planes", [])}


def plane_center(plane: dict) -> np.ndarray:
    if "center" in plane:
        return np.asarray(plane["center"], dtype=np.float64)
    return np.asarray(plane["start"], dtype=np.float64)


def weighted_joint_ids(stable_weight: float, foot_weight: float) -> tuple[np.ndarray, np.ndarray]:
    weights: dict[int, float] = {}
    for idx in STABLE_JOINTS:
        weights[int(idx)] = weights.get(int(idx), 0.0) + float(stable_weight)
    for idx in FOOT_JOINTS:
        weights[int(idx)] = weights.get(int(idx), 0.0) + float(foot_weight)
    joint_ids = np.asarray(sorted(weights), dtype=np.int64)
    weight_arr = np.asarray([weights[int(i)] for i in joint_ids], dtype=np.float64)
    weight_arr /= max(float(weight_arr.sum()), 1e-12)
    return joint_ids, weight_arr


def solve_yaw_plane_translation(
    ref_points: np.ndarray,
    cur_points_after_floor: np.ndarray,
    normal: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    basis = make_plane_basis(normal)
    ref_2d = ref_points @ basis.T
    cur_2d = cur_points_after_floor @ basis.T
    ref_centroid = np.sum(ref_2d * weights[:, None], axis=0)
    cur_centroid = np.sum(cur_2d * weights[:, None], axis=0)
    ref_c = ref_2d - ref_centroid
    cur_c = cur_2d - cur_centroid
    a = float(np.sum(weights * (cur_c[:, 0] * ref_c[:, 0] + cur_c[:, 1] * ref_c[:, 1])))
    b = float(np.sum(weights * (cur_c[:, 0] * ref_c[:, 1] - cur_c[:, 1] * ref_c[:, 0])))
    yaw = math.atan2(b, a)
    R_yaw = rotation_about_axis(normal, yaw)
    ref_centroid_3d = np.sum(ref_points * weights[:, None], axis=0)
    cur_centroid_3d = np.sum(cur_points_after_floor * weights[:, None], axis=0)
    t_plane = project_to_plane(ref_centroid_3d - R_yaw @ cur_centroid_3d, normal)
    return R_yaw, t_plane, yaw


def transform_points(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return points @ R.T + t[None]


def transform_pose(pose: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = pose.astype(np.float64).copy()
    out[:3, :3] = R @ out[:3, :3]
    out[:3, 3] = R @ out[:3, 3] + t
    return out.astype(np.float32)


def body_frame_metrics(joints: np.ndarray, boundary: int) -> dict:
    frames = body_frame_from_joints(torch.from_numpy(joints.astype(np.float32)))
    hist = frames[:boundary].mean(dim=0)
    u, _, vh = torch.linalg.svd(hist)
    hist = u @ vh
    post = frames[boundary : boundary + 2]
    err = rotation_geodesic(post, hist[None].expand_as(post))
    return {
        "B0_to_Amean_deg": float(torch.rad2deg(err[0]).item()),
        "B1_to_Amean_deg": float(torch.rad2deg(err[1]).item()),
        "Bmean_to_Amean_deg": float(torch.rad2deg(err).mean().item()),
    }


def segment_anchor_metrics(joints: np.ndarray, boundary: int, joint_ids: np.ndarray) -> dict:
    hist = joints[:boundary, joint_ids].mean(axis=0)
    b0 = joints[boundary, joint_ids]
    b1 = joints[boundary + 1, joint_ids]
    return {
        "Amean_B0_m": float(np.linalg.norm(hist - b0, axis=-1).mean()),
        "Amean_B1_m": float(np.linalg.norm(hist - b1, axis=-1).mean()),
        "BB_m": float(np.linalg.norm(b0 - b1, axis=-1).mean()),
    }


def write_output(input_dir: Path, output_dir: Path, poses: np.ndarray, R: np.ndarray, t: np.ndarray, boundary: int, overwrite: bool) -> None:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    for sub in ["camera", "color", "conf", "depth", "smpl"]:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)
    for frame in range(poses.shape[0]):
        cam_path = input_dir / "camera" / f"{frame:06d}.npz"
        save_camera(cam_path, output_dir / "camera" / f"{frame:06d}.npz", poses[frame])
        copy_smpl(
            input_dir / "smpl" / f"{frame:06d}.npz",
            output_dir / "smpl" / f"{frame:06d}.npz",
            R if frame >= boundary else None,
            t if frame >= boundary else None,
        )
        for sub, ext in [("color", ".png"), ("conf", ".npy"), ("depth", ".npy")]:
            copy_np_payload(input_dir / sub / f"{frame:06d}{ext}", output_dir / sub / f"{frame:06d}{ext}")


def main() -> None:
    args = parse_args()
    boundary = int(args.boundary)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    data = load_sequence(args.input_dir, len(sorted((args.input_dir / "camera").glob("*.npz"))), device)
    planes = load_floor_planes(args.floor_json)
    if boundary <= 0 or boundary >= data.poses.shape[0]:
        raise ValueError(f"boundary must be in [1, {data.poses.shape[0] - 1}], got {boundary}")

    hist_frames = list(range(boundary))
    post_frames = list(range(boundary, data.poses.shape[0]))
    missing = [idx for idx in hist_frames + [boundary] if idx not in planes]
    if missing:
        raise KeyError(f"Missing floor planes for frames: {missing}")

    hist_normals = [np.asarray(planes[i]["normal"], dtype=np.float64) for i in hist_frames]
    ref_normal = mean_oriented(hist_normals)
    cur_normal = orient_like(np.asarray(planes[boundary]["normal"], dtype=np.float64), ref_normal)
    R_floor = rotation_from_vectors(cur_normal, ref_normal)

    joint_ids, weights = weighted_joint_ids(args.stable_weight, args.foot_weight)
    ref_points = data.joints_world[:boundary, joint_ids].mean(axis=0).astype(np.float64)
    cur_points = data.joints_world[boundary, joint_ids].astype(np.float64)
    cur_after_floor = transform_points(cur_points, R_floor, np.zeros(3, dtype=np.float64))
    R_yaw, t_plane, yaw = solve_yaw_plane_translation(ref_points, cur_after_floor, ref_normal, weights)
    R_total = R_yaw @ R_floor

    if args.normal_translation_source == "floor_center":
        ref_center = np.mean(np.stack([plane_center(planes[i]) for i in hist_frames]), axis=0)
        cur_center = plane_center(planes[boundary])
        amount = float(np.dot(ref_center - R_total @ cur_center, ref_normal))
        t_normal = amount * ref_normal
    elif args.normal_translation_source == "human_centroid":
        ref_centroid = np.sum(ref_points * weights[:, None], axis=0)
        cur_centroid = np.sum(cur_points * weights[:, None], axis=0)
        amount = float(np.dot(ref_centroid - R_total @ cur_centroid, ref_normal))
        t_normal = amount * ref_normal
    else:
        t_normal = np.zeros(3, dtype=np.float64)
    t_total = t_plane + t_normal

    corrected_poses = data.poses.copy()
    corrected_joints = data.joints_world.copy()
    for frame in post_frames:
        corrected_poses[frame] = transform_pose(data.poses[frame], R_total, t_total)
        corrected_joints[frame] = transform_points(data.joints_world[frame], R_total, t_total).astype(np.float32)

    write_output(args.input_dir, args.output_dir, corrected_poses, R_total, t_total, boundary, bool(args.overwrite))

    transformed_normals = {}
    transformed_centers = {}
    for frame, plane in planes.items():
        normal = orient_like(np.asarray(plane["normal"], dtype=np.float64), ref_normal)
        center = plane_center(plane)
        if frame >= boundary:
            normal = normalize(R_total @ normal)
            center = R_total @ center + t_total
        transformed_normals[frame] = normal
        transformed_centers[frame] = center

    hist_after = mean_oriented([transformed_normals[i] for i in hist_frames])
    b_after = mean_oriented([transformed_normals[i] for i in post_frames])
    raw_hist = mean_oriented([orient_like(np.asarray(planes[i]["normal"], dtype=np.float64), ref_normal) for i in hist_frames])
    raw_b = mean_oriented([orient_like(np.asarray(planes[i]["normal"], dtype=np.float64), ref_normal) for i in post_frames])
    metrics = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "floor_json": str(args.floor_json),
        "boundary": boundary,
        "strict_streaming": {
            "uses_future_frames": False,
            "floor_transform_uses": f"history frames {hist_frames} + boundary frame {boundary}",
            "post_boundary_frames_use_cached_transform": True,
        },
        "normal_translation_source": args.normal_translation_source,
        "transform": {
            "floor_rotation_deg": rotation_angle_deg(R_floor),
            "yaw_deg": math.degrees(float(yaw)),
            "total_rotation_deg": rotation_angle_deg(R_total),
            "translation": t_total.astype(np.float32).tolist(),
            "translation_plane": t_plane.astype(np.float32).tolist(),
            "translation_normal": t_normal.astype(np.float32).tolist(),
        },
        "floor_normal": {
            "raw_Amean_Bmean_deg": angle_deg(raw_hist, raw_b),
            "aligned_Amean_Bmean_deg": angle_deg(hist_after, b_after),
            "aligned_Amean_B0_deg": angle_deg(hist_after, transformed_normals[boundary]),
            "aligned_Amean_B1_deg": angle_deg(hist_after, transformed_normals[min(boundary + 1, data.poses.shape[0] - 1)]),
        },
        "human": {
            "raw_segment_anchor": segment_anchor_metrics(data.joints_world, boundary, joint_ids),
            "aligned_segment_anchor": segment_anchor_metrics(corrected_joints, boundary, joint_ids),
            "raw_body_frame": body_frame_metrics(data.joints_world, boundary),
            "aligned_body_frame": body_frame_metrics(corrected_joints, boundary),
        },
        "joint_ids": joint_ids.astype(int).tolist(),
    }
    (args.output_dir / "floor_locked_segment_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
