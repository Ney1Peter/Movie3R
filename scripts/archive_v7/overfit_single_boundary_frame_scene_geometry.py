#!/usr/bin/env python3
"""Overfit one boundary frame with human anchor and stronger scene geometry guard.

Compared with ``overfit_single_boundary_frame_scene_normal.py``, this version
adds two lightweight causal geometry cues from the previous/current frame only:

1. top-K dominant background planes instead of a single plane normal;
2. a weak robust background Chamfer loss between corrected current background
   points and previous-frame background points.

The goal is to keep the successful scene-normal tilt guard while reducing the
remaining in-plane horizontal/yaw ambiguity left by a single dominant plane.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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
from overfit_single_boundary_frame_scene_normal import (
    estimate_dominant_plane,
    jsonable_plane,
    load_background_points,
)
from overfit_transient_gate_human_correction import compute_transition_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--source_video", type=Path, default=None)
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--boundary", type=int, required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--max_delta_t", type=float, default=6.0)
    parser.add_argument("--max_delta_r_deg", type=float, default=90.0)
    parser.add_argument("--stable_weight", type=float, default=0.8)
    parser.add_argument("--foot_weight", type=float, default=1.5)
    parser.add_argument("--orient_weight", type=float, default=4.0)
    parser.add_argument("--normal_weight", type=float, default=20.0)
    parser.add_argument("--plane_offset_weight", type=float, default=4.0)
    parser.add_argument("--bg_chamfer_weight", type=float, default=1.0)
    parser.add_argument("--bg_chamfer_cap", type=float, default=0.35)
    parser.add_argument("--prior_weight", type=float, default=0.03)
    parser.add_argument("--conf_threshold", type=float, default=1.5)
    parser.add_argument("--mask_threshold", type=float, default=0.1)
    parser.add_argument("--plane_max_points", type=int, default=20000)
    parser.add_argument("--plane_iterations", type=int, default=2048)
    parser.add_argument("--plane_threshold", type=float, default=0.04)
    parser.add_argument("--num_planes", type=int, default=3)
    parser.add_argument("--min_plane_inlier_ratio", type=float, default=0.03)
    parser.add_argument("--min_plane_dot", type=float, default=0.65)
    parser.add_argument("--bg_chamfer_points", type=int, default=2500)
    parser.add_argument("--subset_output_dir", type=Path, default=None)
    parser.add_argument("--raw_subset_output_dir", type=Path, default=None)
    parser.add_argument("--subset_start", type=int, default=None)
    parser.add_argument("--subset_count", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log_every", type=int, default=200)
    return parser.parse_args()


def estimate_top_planes(points: np.ndarray, args: argparse.Namespace, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    points = points.astype(np.float32)
    if points.shape[0] > args.plane_max_points:
        points = points[rng.choice(points.shape[0], size=args.plane_max_points, replace=False)]
    remaining = points
    planes = []
    total = max(points.shape[0], 1)
    for i in range(int(args.num_planes)):
        if remaining.shape[0] < 200:
            break
        plane = estimate_dominant_plane(
            remaining,
            max_points=min(args.plane_max_points, remaining.shape[0]),
            iterations=args.plane_iterations,
            threshold=args.plane_threshold,
            seed=seed + 17 * i,
        )
        normal = plane["normal"].astype(np.float32)
        d = float(plane["d"])
        inliers = np.abs(remaining @ normal + d) < float(args.plane_threshold)
        global_ratio = float(np.sum(inliers) / total)
        plane["global_inlier_ratio"] = global_ratio
        if global_ratio < float(args.min_plane_inlier_ratio):
            break
        planes.append(plane)
        remaining = remaining[~inliers]
    return planes


def match_planes(ref_planes: list[dict], cur_planes: list[dict], min_dot: float) -> list[dict]:
    matches = []
    used = set()
    for ref_i, ref in enumerate(ref_planes):
        n_ref = np.asarray(ref["normal"], dtype=np.float32)
        best = None
        for cur_i, cur in enumerate(cur_planes):
            if cur_i in used:
                continue
            n_cur = np.asarray(cur["normal"], dtype=np.float32)
            dot = float(np.dot(n_ref, n_cur))
            score = abs(dot)
            if best is None or score > best[0]:
                best = (score, dot, cur_i, cur)
        if best is None or best[0] < float(min_dot):
            continue
        score, dot, cur_i, cur = best
        used.add(cur_i)
        cur_copy = dict(cur)
        if dot < 0.0:
            cur_copy["normal"] = -np.asarray(cur_copy["normal"], dtype=np.float32)
            cur_copy["d"] = np.float32(-float(cur_copy["d"]))
        weight = min(float(ref.get("global_inlier_ratio", ref["inlier_ratio"])), float(cur_copy.get("global_inlier_ratio", cur_copy["inlier_ratio"])))
        matches.append({"ref_index": ref_i, "cur_index": cur_i, "dot_abs": score, "weight": weight, "ref": ref, "cur": cur_copy})
    return matches


def sample_points(points: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    points = points.astype(np.float32)
    if points.shape[0] > n:
        return points[rng.choice(points.shape[0], size=n, replace=False)]
    return points


def build_scene_geometry(data, args: argparse.Namespace, device: torch.device) -> dict:
    ref_id = int(args.boundary) - 1
    cur_id = int(args.boundary)
    pts_ref = load_background_points(args.input_dir, ref_id, data.poses[ref_id], data.intrinsics[ref_id], args.conf_threshold, args.mask_threshold)
    pts_cur = load_background_points(args.input_dir, cur_id, data.poses[cur_id], data.intrinsics[cur_id], args.conf_threshold, args.mask_threshold)
    ref_planes = estimate_top_planes(pts_ref, args, seed=211)
    cur_planes = estimate_top_planes(pts_cur, args, seed=223)
    matches = match_planes(ref_planes, cur_planes, args.min_plane_dot)
    if not matches:
        raise ValueError("No reliable matched background planes found")
    ref_bg = torch.from_numpy(sample_points(pts_ref, args.bg_chamfer_points, seed=227)).to(device=device, dtype=torch.float32)
    cur_bg = torch.from_numpy(sample_points(pts_cur, args.bg_chamfer_points, seed=229)).to(device=device, dtype=torch.float32)
    tensors = []
    total_weight = sum(max(float(m["weight"]), 1e-6) for m in matches)
    for m in matches:
        w = max(float(m["weight"]), 1e-6) / max(total_weight, 1e-6)
        tensors.append({
            "n_ref": torch.from_numpy(np.asarray(m["ref"]["normal"], dtype=np.float32)).to(device=device),
            "n_cur": torch.from_numpy(np.asarray(m["cur"]["normal"], dtype=np.float32)).to(device=device),
            "d_ref": torch.tensor(float(m["ref"]["d"]), device=device, dtype=torch.float32),
            "d_cur": torch.tensor(float(m["cur"]["d"]), device=device, dtype=torch.float32),
            "weight": torch.tensor(w, device=device, dtype=torch.float32),
        })
    return {
        "ref_planes": ref_planes,
        "cur_planes": cur_planes,
        "matches": matches,
        "match_tensors": tensors,
        "ref_bg": ref_bg,
        "cur_bg": cur_bg,
    }


def robust_bg_chamfer(cur_corr: torch.Tensor, ref_bg: torch.Tensor, cap: float) -> torch.Tensor:
    d = torch.cdist(cur_corr.unsqueeze(0), ref_bg.unsqueeze(0), p=2)[0]
    cur_to_ref = d.min(dim=1).values.clamp_max(float(cap)).mean()
    ref_to_cur = d.min(dim=0).values.clamp_max(float(cap)).mean()
    return 0.5 * (cur_to_ref + ref_to_cur)


def train_scene_geometry_single_frame(data, scene: dict, args: argparse.Namespace, device: torch.device) -> tuple[np.ndarray, dict]:
    boundary = int(args.boundary)
    num_frames = data.poses.shape[0]
    poses = torch.from_numpy(data.poses).to(device=device, dtype=torch.float32)
    joints_world = torch.from_numpy(data.joints_world).to(device=device, dtype=torch.float32)
    raw_delta_t = torch.nn.Parameter(torch.zeros(1, 3, device=device))
    raw_delta_r = torch.nn.Parameter(torch.zeros(1, 3, device=device))
    optimizer = torch.optim.AdamW([raw_delta_t, raw_delta_r], lr=args.lr, weight_decay=1e-4)
    max_delta_r = math.radians(float(args.max_delta_r_deg))

    ref_joints = joints_world[boundary - 1]
    ref_frame = torso_frame(joints_world[boundary - 1 : boundary])[0]
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

        normal_losses, offset_losses, dots = [], [], []
        for m in scene["match_tensors"]:
            n_corr = R_delta @ m["n_cur"]
            normal_dot = (n_corr * m["n_ref"]).sum().clamp(-1.0, 1.0)
            d_corr = m["d_cur"] - (n_corr * delta_t_one[0]).sum()
            normal_losses.append(m["weight"] * (1.0 - normal_dot.abs()))
            offset_losses.append(m["weight"] * torch.minimum((d_corr - m["d_ref"]).pow(2), (d_corr + m["d_ref"]).pow(2)))
            dots.append(normal_dot.abs())
        normal_loss = torch.stack(normal_losses).sum()
        offset_loss = torch.stack(offset_losses).sum()
        mean_normal_dot = torch.stack(dots).mean()
        bg_corr = torch.einsum("ij,kj->ki", R_delta, scene["cur_bg"]) + delta_t_one[0]
        bg_chamfer = robust_bg_chamfer(bg_corr, scene["ref_bg"], args.bg_chamfer_cap)
        prior_loss = delta_t_one.pow(2).mean() + delta_r_one.pow(2).mean()

        loss = (
            args.stable_weight * stable_loss
            + args.foot_weight * foot_loss
            + args.orient_weight * orient_loss
            + args.normal_weight * normal_loss
            + args.plane_offset_weight * offset_loss
            + args.bg_chamfer_weight * bg_chamfer
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
                "mean_normal_dot_abs": float(mean_normal_dot.detach().cpu()),
                "offset_loss": float(offset_loss.detach().cpu()),
                "bg_chamfer": float(bg_chamfer.detach().cpu()),
                "prior_loss": float(prior_loss.detach().cpu()),
                "delta_t_norm": float(delta_t_one.norm().detach().cpu()),
                "delta_r_deg": float(torch.rad2deg(delta_r_one.norm()).detach().cpu()),
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


def scene_json(scene: dict) -> dict:
    return {
        "ref_planes": [jsonable_plane(p) for p in scene["ref_planes"]],
        "cur_planes": [jsonable_plane(p) for p in scene["cur_planes"]],
        "matches": [
            {
                "ref_index": int(m["ref_index"]),
                "cur_index": int(m["cur_index"]),
                "dot_abs": float(m["dot_abs"]),
                "weight": float(m["weight"]),
            }
            for m in scene["matches"]
        ],
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(43)
    np.random.seed(43)
    device = torch.device(args.device)
    num_frames = infer_num_frames(args.input_dir, args.source_video, args.num_frames)
    print(f"Using {num_frames} frames from {args.input_dir}")
    data = load_sequence(args.input_dir, num_frames, device)
    scene = build_scene_geometry(data, args, device)
    print("scene_geometry", json.dumps(scene_json(scene), indent=2, sort_keys=True))

    raw_metrics = compute_metrics(data, data.poses, args.boundary)
    raw_transition_metrics = compute_transition_metrics(data, data.poses, args.boundary)
    print("raw_metrics", json.dumps(raw_metrics, indent=2, sort_keys=True))
    print("raw_transition_metrics", json.dumps(raw_transition_metrics, indent=2, sort_keys=True))

    corrected_poses, debug = train_scene_geometry_single_frame(data, scene, args, device)
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
        args.output_dir / "single_boundary_frame_scene_geometry_debug.npz",
        delta_t=debug["delta_t"],
        delta_r=debug["delta_r"],
        raw_metrics=np.array(json.dumps(raw_metrics), dtype=object),
        corrected_metrics=np.array(json.dumps(corrected_metrics), dtype=object),
        raw_transition_metrics=np.array(json.dumps(raw_transition_metrics), dtype=object),
        corrected_transition_metrics=np.array(json.dumps(corrected_transition_metrics), dtype=object),
    )
    with open(args.output_dir / "single_boundary_frame_scene_geometry_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "raw": raw_metrics,
                "corrected": corrected_metrics,
                "raw_transition": raw_transition_metrics,
                "corrected_transition": corrected_transition_metrics,
                "history": debug["history"],
                "scene_geometry": scene_json(scene),
                "corrected_frame": int(args.boundary),
                "reference_frame": int(args.boundary - 1),
            },
            f,
            indent=2,
            sort_keys=True,
        )
    print(f"Wrote scene-geometry single-frame corrected output to {args.output_dir}")


if __name__ == "__main__":
    main()
