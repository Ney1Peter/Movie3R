#!/usr/bin/env python3
"""Overfit one boundary frame with human anchor plus scene-plane normal guard.

This minimal smoke test corrects only the first post-boundary frame.  It keeps the
human-anchor pair loss from the single-frame SE(3) test, but adds a dominant
background plane constraint estimated from Human3R depth/confidence after masking
out the detected person:

    human anchor:  align frame boundary human to frame boundary-1 human
    scene guard:   keep dominant background plane normal/offset consistent

The goal is to allow necessary 3D correction while discouraging the scene from
tilting or drifting vertically just to satisfy the human loss.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates

from overfit_human_anchor_pose_correction import (
    FOOT_JOINTS,
    STABLE_JOINTS,
    apply_correction,
    compute_metrics,
    infer_num_frames,
    load_sequence,
    so3_exp_map,
    torso_frame,
)
from overfit_memory_human_correction import chamfer_per_frame
from overfit_single_boundary_frame_human_anchor import write_outputs_with_links
from overfit_transient_gate_human_correction import compute_transition_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--source_video", type=Path, default=None)
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--boundary", type=int, required=True, help="Frame to correct; first frame after shot boundary.")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--max_delta_t", type=float, default=6.0)
    parser.add_argument("--max_delta_r_deg", type=float, default=90.0)
    parser.add_argument("--stable_weight", type=float, default=1.0)
    parser.add_argument("--foot_weight", type=float, default=2.0)
    parser.add_argument("--orient_weight", type=float, default=5.0)
    parser.add_argument("--normal_weight", type=float, default=20.0)
    parser.add_argument("--plane_offset_weight", type=float, default=2.0)
    parser.add_argument("--prior_weight", type=float, default=0.01)
    parser.add_argument("--conf_threshold", type=float, default=1.5)
    parser.add_argument("--mask_threshold", type=float, default=0.1)
    parser.add_argument("--plane_max_points", type=int, default=20000)
    parser.add_argument("--plane_iterations", type=int, default=2048)
    parser.add_argument("--plane_threshold", type=float, default=0.04)
    parser.add_argument("--min_inlier_ratio", type=float, default=0.03)
    parser.add_argument("--subset_output_dir", type=Path, default=None)
    parser.add_argument("--raw_subset_output_dir", type=Path, default=None)
    parser.add_argument("--subset_start", type=int, default=None)
    parser.add_argument("--subset_count", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log_every", type=int, default=200)
    return parser.parse_args()


def load_background_points(input_dir: Path, frame_id: int, pose: np.ndarray, intrinsics: np.ndarray, conf_threshold: float, mask_threshold: float) -> np.ndarray:
    depth = np.load(input_dir / "depth" / f"{frame_id:06d}.npy").astype(np.float32)
    conf = np.load(input_dir / "conf" / f"{frame_id:06d}.npy").astype(np.float32)
    points_world, _ = depthmap_to_absolute_camera_coordinates(depth, intrinsics.astype(np.float32), pose.astype(np.float32))

    valid = np.isfinite(points_world).all(axis=-1) & np.isfinite(depth) & (depth > 0.0) & np.isfinite(conf) & (conf >= float(conf_threshold))
    smpl = np.load(input_dir / "smpl" / f"{frame_id:06d}.npz", allow_pickle=True)
    if "msk" in smpl.files:
        msk = smpl["msk"]
        if msk is not None and msk.size > 0:
            human_mask = np.max(msk.astype(np.float32), axis=0) > float(mask_threshold)
            if human_mask.shape == valid.shape:
                valid &= ~human_mask
    points = points_world[valid].astype(np.float32)
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] < 100:
        raise ValueError(f"Too few background points for frame {frame_id}: {points.shape[0]}")
    return points


def estimate_dominant_plane(points: np.ndarray, max_points: int, iterations: int, threshold: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    points = points.astype(np.float32)
    if points.shape[0] > max_points:
        points = points[rng.choice(points.shape[0], size=max_points, replace=False)]

    n_points = points.shape[0]
    best_count = -1
    best_normal = None
    best_d = None
    batch = 128
    for start in range(0, iterations, batch):
        b = min(batch, iterations - start)
        idx = rng.integers(0, n_points, size=(b, 3))
        p0, p1, p2 = points[idx[:, 0]], points[idx[:, 1]], points[idx[:, 2]]
        normals = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normals, axis=1, keepdims=True)
        valid = norm[:, 0] > 1e-6
        if not np.any(valid):
            continue
        normals = normals[valid] / norm[valid]
        p0 = p0[valid]
        d = -np.sum(normals * p0, axis=1)
        dist = np.abs(points @ normals.T + d[None])
        counts = np.sum(dist < float(threshold), axis=0)
        local_best = int(np.argmax(counts))
        if int(counts[local_best]) > best_count:
            best_count = int(counts[local_best])
            best_normal = normals[local_best].astype(np.float32)
            best_d = float(d[local_best])

    if best_normal is None:
        raise ValueError("RANSAC failed to estimate a plane")

    inliers = np.abs(points @ best_normal + best_d) < float(threshold)
    inlier_points = points[inliers]
    if inlier_points.shape[0] >= 3:
        center = inlier_points.mean(axis=0)
        _, _, vh = np.linalg.svd(inlier_points - center, full_matrices=False)
        normal = vh[-1].astype(np.float32)
        normal /= max(float(np.linalg.norm(normal)), 1e-8)
        d = -float(normal @ center)
        best_normal, best_d = normal, d
        inliers = np.abs(points @ best_normal + best_d) < float(threshold)

    return {
        "normal": best_normal.astype(np.float32),
        "d": np.float32(best_d),
        "inlier_ratio": float(np.mean(inliers)),
        "num_points": int(n_points),
        "num_inliers": int(np.sum(inliers)),
    }


def estimate_scene_planes(data, args: argparse.Namespace) -> dict:
    ref_id = int(args.boundary) - 1
    cur_id = int(args.boundary)
    pts_ref = load_background_points(args.input_dir, ref_id, data.poses[ref_id], data.intrinsics[ref_id], args.conf_threshold, args.mask_threshold)
    pts_cur = load_background_points(args.input_dir, cur_id, data.poses[cur_id], data.intrinsics[cur_id], args.conf_threshold, args.mask_threshold)
    plane_ref = estimate_dominant_plane(pts_ref, args.plane_max_points, args.plane_iterations, args.plane_threshold, seed=101)
    plane_cur = estimate_dominant_plane(pts_cur, args.plane_max_points, args.plane_iterations, args.plane_threshold, seed=103)
    if float(np.dot(plane_ref["normal"], plane_cur["normal"])) < 0.0:
        plane_cur["normal"] = -plane_cur["normal"]
        plane_cur["d"] = np.float32(-float(plane_cur["d"]))
    scene_weight = min(1.0, plane_ref["inlier_ratio"] / max(args.min_inlier_ratio, 1e-6)) * min(1.0, plane_cur["inlier_ratio"] / max(args.min_inlier_ratio, 1e-6))
    return {"ref": plane_ref, "cur": plane_cur, "scene_weight": float(scene_weight)}


def jsonable_plane(plane: dict) -> dict:
    return {k: (v.tolist() if isinstance(v, np.ndarray) else float(v) if isinstance(v, np.floating) else v) for k, v in plane.items()}


def train_scene_normal_single_frame(data, plane_info: dict, args: argparse.Namespace, device: torch.device) -> tuple[np.ndarray, dict]:
    boundary = int(args.boundary)
    num_frames = data.poses.shape[0]
    if boundary <= 0 or boundary >= num_frames:
        raise ValueError(f"boundary must be in [1, {num_frames - 1}], got {boundary}")

    poses = torch.from_numpy(data.poses).to(device=device, dtype=torch.float32)
    joints_world = torch.from_numpy(data.joints_world).to(device=device, dtype=torch.float32)
    raw_delta_t = torch.nn.Parameter(torch.zeros(1, 3, device=device))
    raw_delta_r = torch.nn.Parameter(torch.zeros(1, 3, device=device))
    optimizer = torch.optim.AdamW([raw_delta_t, raw_delta_r], lr=args.lr, weight_decay=1e-4)
    max_delta_r = math.radians(float(args.max_delta_r_deg))

    ref_joints = joints_world[boundary - 1]
    ref_frame = torso_frame(joints_world[boundary - 1 : boundary])[0]
    n_ref = torch.from_numpy(plane_info["ref"]["normal"]).to(device=device, dtype=torch.float32)
    n_cur = torch.from_numpy(plane_info["cur"]["normal"]).to(device=device, dtype=torch.float32)
    d_ref = torch.tensor(float(plane_info["ref"]["d"]), device=device, dtype=torch.float32)
    d_cur = torch.tensor(float(plane_info["cur"]["d"]), device=device, dtype=torch.float32)
    scene_weight = torch.tensor(float(plane_info["scene_weight"]), device=device, dtype=torch.float32)

    history = []
    for step in range(args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        delta_t_one = float(args.max_delta_t) * torch.tanh(raw_delta_t)
        delta_r_one = max_delta_r * torch.tanh(raw_delta_r)
        corrected_one, _, _ = apply_correction(joints_world[boundary : boundary + 1], poses[boundary : boundary + 1], delta_t_one, delta_r_one)
        corrected_one = corrected_one[0]
        R_delta = so3_exp_map(delta_r_one)[0]
        frame_corr = torso_frame(corrected_one.unsqueeze(0))[0]

        stable_loss = F.smooth_l1_loss(corrected_one[STABLE_JOINTS], ref_joints[STABLE_JOINTS], beta=0.05)
        foot_loss = chamfer_per_frame(corrected_one[FOOT_JOINTS], ref_joints[FOOT_JOINTS])
        orient_loss = (1.0 - (frame_corr * ref_frame).sum(dim=-1).clamp(-1.0, 1.0)).mean()

        n_corr = R_delta @ n_cur
        normal_dot = (n_corr * n_ref).sum().clamp(-1.0, 1.0)
        normal_loss = 1.0 - normal_dot.abs()
        d_corr = d_cur - (n_corr * delta_t_one[0]).sum()
        offset_loss = torch.minimum((d_corr - d_ref).pow(2), (d_corr + d_ref).pow(2))
        prior_loss = delta_t_one.pow(2).mean() + delta_r_one.pow(2).mean()

        loss = (
            args.stable_weight * stable_loss
            + args.foot_weight * foot_loss
            + args.orient_weight * orient_loss
            + scene_weight * args.normal_weight * normal_loss
            + scene_weight * args.plane_offset_weight * offset_loss
            + args.prior_weight * prior_loss
        )
        loss.backward()
        optimizer.step()

        if step % args.log_every == 0 or step == args.steps:
            record = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "stable_loss": float(stable_loss.detach().cpu()),
                "foot_loss": float(foot_loss.detach().cpu()),
                "orient_loss": float(orient_loss.detach().cpu()),
                "normal_loss": float(normal_loss.detach().cpu()),
                "normal_dot_abs": float(normal_dot.abs().detach().cpu()),
                "offset_loss": float(offset_loss.detach().cpu()),
                "prior_loss": float(prior_loss.detach().cpu()),
                "delta_t_norm": float(delta_t_one.norm().detach().cpu()),
                "delta_r_deg": float(torch.rad2deg(delta_r_one.norm()).detach().cpu()),
                "scene_weight": float(scene_weight.detach().cpu()),
            }
            print(json.dumps(record, sort_keys=True), flush=True)
            history.append(record)

    with torch.no_grad():
        delta_t_one = float(args.max_delta_t) * torch.tanh(raw_delta_t)
        delta_r_one = max_delta_r * torch.tanh(raw_delta_r)
        _, R_corr_one, t_corr_one = apply_correction(joints_world[boundary : boundary + 1], poses[boundary : boundary + 1], delta_t_one, delta_r_one)
        corrected_poses = data.poses.copy()
        corrected_poses[boundary, :3, :3] = R_corr_one[0].detach().cpu().numpy().astype(np.float32)
        corrected_poses[boundary, :3, 3] = t_corr_one[0].detach().cpu().numpy().astype(np.float32)
        debug = {
            "delta_t": delta_t_one.detach().cpu().numpy().astype(np.float32),
            "delta_r": delta_r_one.detach().cpu().numpy().astype(np.float32),
            "history": history,
        }
    return corrected_poses, debug


def main() -> None:
    args = parse_args()
    torch.manual_seed(41)
    np.random.seed(41)
    device = torch.device(args.device)
    num_frames = infer_num_frames(args.input_dir, args.source_video, args.num_frames)
    print(f"Using {num_frames} frames from {args.input_dir}")
    data = load_sequence(args.input_dir, num_frames, device)
    plane_info = estimate_scene_planes(data, args)
    print("plane_info", json.dumps({
        "ref": jsonable_plane(plane_info["ref"]),
        "cur": jsonable_plane(plane_info["cur"]),
        "scene_weight": plane_info["scene_weight"],
    }, indent=2, sort_keys=True))

    raw_metrics = compute_metrics(data, data.poses, args.boundary)
    raw_transition_metrics = compute_transition_metrics(data, data.poses, args.boundary)
    print("raw_metrics", json.dumps(raw_metrics, indent=2, sort_keys=True))
    print("raw_transition_metrics", json.dumps(raw_transition_metrics, indent=2, sort_keys=True))

    corrected_poses, debug = train_scene_normal_single_frame(data, plane_info, args, device)
    corrected_metrics = compute_metrics(data, corrected_poses, args.boundary)
    corrected_transition_metrics = compute_transition_metrics(data, corrected_poses, args.boundary)
    print("corrected_metrics", json.dumps(corrected_metrics, indent=2, sort_keys=True))
    print("corrected_transition_metrics", json.dumps(corrected_transition_metrics, indent=2, sort_keys=True))

    all_frames = list(range(num_frames))
    write_outputs_with_links(args.input_dir, args.output_dir, corrected_poses, data.intrinsics, all_frames, args.overwrite)
    if args.subset_output_dir is not None:
        subset_start = args.boundary - 1 if args.subset_start is None else args.subset_start
        subset_frames = list(range(subset_start, min(num_frames, subset_start + args.subset_count)))
        write_outputs_with_links(args.input_dir, args.subset_output_dir, corrected_poses, data.intrinsics, subset_frames, args.overwrite)
    if args.raw_subset_output_dir is not None:
        subset_start = args.boundary - 1 if args.subset_start is None else args.subset_start
        subset_frames = list(range(subset_start, min(num_frames, subset_start + args.subset_count)))
        write_outputs_with_links(args.input_dir, args.raw_subset_output_dir, data.poses, data.intrinsics, subset_frames, args.overwrite)

    np.savez(
        args.output_dir / "single_boundary_frame_scene_normal_debug.npz",
        delta_t=debug["delta_t"],
        delta_r=debug["delta_r"],
        plane_ref_normal=plane_info["ref"]["normal"],
        plane_cur_normal=plane_info["cur"]["normal"],
        plane_ref_d=np.array(plane_info["ref"]["d"], dtype=np.float32),
        plane_cur_d=np.array(plane_info["cur"]["d"], dtype=np.float32),
        raw_metrics=np.array(json.dumps(raw_metrics), dtype=object),
        corrected_metrics=np.array(json.dumps(corrected_metrics), dtype=object),
        raw_transition_metrics=np.array(json.dumps(raw_transition_metrics), dtype=object),
        corrected_transition_metrics=np.array(json.dumps(corrected_transition_metrics), dtype=object),
    )
    with open(args.output_dir / "single_boundary_frame_scene_normal_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "raw": raw_metrics,
                "corrected": corrected_metrics,
                "raw_transition": raw_transition_metrics,
                "corrected_transition": corrected_transition_metrics,
                "history": debug["history"],
                "plane_info": {
                    "ref": jsonable_plane(plane_info["ref"]),
                    "cur": jsonable_plane(plane_info["cur"]),
                    "scene_weight": plane_info["scene_weight"],
                },
                "corrected_frame": int(args.boundary),
                "reference_frame": int(args.boundary - 1),
            },
            f,
            indent=2,
            sort_keys=True,
        )
    print(f"Wrote scene-normal single-frame corrected output to {args.output_dir}")


if __name__ == "__main__":
    main()
