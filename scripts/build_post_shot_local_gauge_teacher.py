#!/usr/bin/env python3
"""Build an offline post-shot local-gauge teacher for boundary settling frames.

This is an oracle / pseudo-label generation smoke test, not a causal inference
method. It is allowed to read a future post-shot stable frame/window to estimate
the local cam2 gauge, then it corrects boundary settling frames toward that gauge.
The future information must not be used as input to the final online student.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

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
from overfit_single_boundary_frame_scene_geometry import estimate_top_planes, match_planes, robust_bg_chamfer, sample_points
from overfit_single_boundary_frame_scene_normal import jsonable_plane, load_background_points
from overfit_transient_gate_human_correction import compute_transition_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--source_video", type=Path, default=None)
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--boundary", type=int, required=True, help="First post-shot frame to correct.")
    parser.add_argument("--target_count", type=int, default=3, help="Number of settling frames to correct from boundary onward.")
    parser.add_argument("--stable_start", type=int, required=True, help="First future stable frame used by the teacher.")
    parser.add_argument("--stable_end", type=int, required=True, help="Inclusive future stable window end.")
    parser.add_argument("--gauge_anchor_frame", type=int, default=None, help="Future frame used as local gauge anchor. Defaults to stable_start.")
    parser.add_argument("--steps_per_frame", type=int, default=800)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--max_delta_t", type=float, default=5.0)
    parser.add_argument("--max_delta_r_deg", type=float, default=90.0)
    parser.add_argument("--stable_weight", type=float, default=0.6)
    parser.add_argument("--foot_weight", type=float, default=1.0)
    parser.add_argument("--orient_weight", type=float, default=2.5)
    parser.add_argument("--normal_weight", type=float, default=20.0)
    parser.add_argument("--plane_offset_weight", type=float, default=5.0)
    parser.add_argument("--bg_chamfer_weight", type=float, default=1.5)
    parser.add_argument("--bg_chamfer_cap", type=float, default=0.35)
    parser.add_argument("--temporal_t_weight", type=float, default=0.08)
    parser.add_argument("--temporal_r_weight", type=float, default=0.08)
    parser.add_argument("--prior_weight", type=float, default=0.04)
    parser.add_argument("--conf_threshold", type=float, default=1.5)
    parser.add_argument("--mask_threshold", type=float, default=0.1)
    parser.add_argument("--plane_max_points", type=int, default=16000)
    parser.add_argument("--plane_iterations", type=int, default=1024)
    parser.add_argument("--plane_threshold", type=float, default=0.04)
    parser.add_argument("--num_planes", type=int, default=3)
    parser.add_argument("--min_plane_inlier_ratio", type=float, default=0.03)
    parser.add_argument("--min_plane_dot", type=float, default=0.9)
    parser.add_argument("--bg_chamfer_points", type=int, default=1200)
    parser.add_argument("--subset_output_dir", type=Path, default=None)
    parser.add_argument("--raw_subset_output_dir", type=Path, default=None)
    parser.add_argument("--subset_start", type=int, default=None)
    parser.add_argument("--subset_count", type=int, default=31)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log_every", type=int, default=200)
    return parser.parse_args()


def build_scene(input_dir: Path, frame_id: int, pose: np.ndarray, intrinsics: np.ndarray, args: argparse.Namespace, device: torch.device, seed_base: int) -> dict:
    points = load_background_points(input_dir, frame_id, pose, intrinsics, args.conf_threshold, args.mask_threshold)
    plane_args = SimpleNamespace(
        plane_max_points=args.plane_max_points,
        plane_iterations=args.plane_iterations,
        plane_threshold=args.plane_threshold,
        num_planes=args.num_planes,
        min_plane_inlier_ratio=args.min_plane_inlier_ratio,
    )
    planes = estimate_top_planes(points, plane_args, seed=seed_base + frame_id * 17)
    bg = sample_points(points, args.bg_chamfer_points, seed=seed_base + frame_id * 17 + 5)
    return {
        "frame": int(frame_id),
        "num_points": int(points.shape[0]),
        "planes": planes,
        "bg": torch.from_numpy(bg).to(device=device, dtype=torch.float32),
    }


def make_match_tensors(ref_scene: dict, cur_scene: dict, min_dot: float, device: torch.device) -> tuple[list[dict], list[dict]]:
    matches = match_planes(ref_scene["planes"], cur_scene["planes"], min_dot)
    tensors = []
    total_weight = sum(max(float(m["weight"]), 1e-6) for m in matches)
    for m in matches:
        w = max(float(m["weight"]), 1e-6) / max(total_weight, 1e-6)
        tensors.append(
            {
                "n_ref": torch.from_numpy(np.asarray(m["ref"]["normal"], dtype=np.float32)).to(device=device),
                "n_cur": torch.from_numpy(np.asarray(m["cur"]["normal"], dtype=np.float32)).to(device=device),
                "d_ref": torch.tensor(float(m["ref"]["d"]), device=device, dtype=torch.float32),
                "d_cur": torch.tensor(float(m["cur"]["d"]), device=device, dtype=torch.float32),
                "weight": torch.tensor(w, device=device, dtype=torch.float32),
            }
        )
    return matches, tensors


def rotation_temporal_loss(R_corr: torch.Tensor, R_prev: torch.Tensor) -> torch.Tensor:
    rel = R_corr @ R_prev.transpose(0, 1)
    trace = torch.diagonal(rel, dim1=-2, dim2=-1).sum()
    cos = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    return 1.0 - cos


def optimize_target_frame(
    frame_id: int,
    data,
    gauge_scene: dict,
    cur_scene: dict,
    corrected_poses: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict]:
    matches, match_tensors = make_match_tensors(gauge_scene, cur_scene, args.min_plane_dot, device)
    if not match_tensors:
        pose = data.poses[frame_id]
        return pose[:3, :3].copy(), pose[:3, 3].copy(), {"frame": int(frame_id), "skipped": True, "reason": "no_plane_matches"}

    poses = torch.from_numpy(data.poses).to(device=device, dtype=torch.float32)
    joints_world = torch.from_numpy(data.joints_world).to(device=device, dtype=torch.float32)
    raw_delta_t = torch.nn.Parameter(torch.zeros(1, 3, device=device))
    raw_delta_r = torch.nn.Parameter(torch.zeros(1, 3, device=device))
    optimizer = torch.optim.AdamW([raw_delta_t, raw_delta_r], lr=args.lr, weight_decay=1e-4)
    max_delta_r = math.radians(float(args.max_delta_r_deg))

    boundary = int(args.boundary)
    prev_R = torch.from_numpy(corrected_poses[frame_id - 1, :3, :3]).to(device=device, dtype=torch.float32)
    prev_t = torch.from_numpy(corrected_poses[frame_id - 1, :3, 3]).to(device=device, dtype=torch.float32)
    ref_joints = joints_world[boundary - 1]
    ref_frame = torso_frame(joints_world[boundary - 1 : boundary])[0]
    history = []

    for step in range(int(args.steps_per_frame) + 1):
        optimizer.zero_grad(set_to_none=True)
        delta_t = float(args.max_delta_t) * torch.tanh(raw_delta_t)
        delta_r = max_delta_r * torch.tanh(raw_delta_r)
        corrected_one, R_corr_one, t_corr_one = apply_correction(joints_world[frame_id : frame_id + 1], poses[frame_id : frame_id + 1], delta_t, delta_r)
        corrected_one = corrected_one[0]
        R_corr = R_corr_one[0]
        t_corr = t_corr_one[0]
        R_delta = so3_exp_map(delta_r)[0]

        frame_corr = torso_frame(corrected_one.unsqueeze(0))[0]
        if frame_id == boundary:
            stable_loss = F.smooth_l1_loss(corrected_one[STABLE_JOINTS], ref_joints[STABLE_JOINTS], beta=0.05)
            foot_loss = chamfer_per_frame(corrected_one[FOOT_JOINTS], ref_joints[FOOT_JOINTS])
            orient_loss = (1.0 - (frame_corr * ref_frame).sum(dim=-1).clamp(-1.0, 1.0)).mean()
        else:
            stable_loss = delta_t.new_tensor(0.0)
            foot_loss = delta_t.new_tensor(0.0)
            orient_loss = delta_t.new_tensor(0.0)

        normal_losses, offset_losses, dots = [], [], []
        for m in match_tensors:
            n_corr = R_delta @ m["n_cur"]
            normal_dot = (n_corr * m["n_ref"]).sum().clamp(-1.0, 1.0)
            d_corr = m["d_cur"] - (n_corr * delta_t[0]).sum()
            normal_losses.append(m["weight"] * (1.0 - normal_dot.abs()))
            offset_losses.append(m["weight"] * torch.minimum((d_corr - m["d_ref"]).pow(2), (d_corr + m["d_ref"]).pow(2)))
            dots.append(normal_dot.abs())
        normal_loss = torch.stack(normal_losses).sum()
        offset_loss = torch.stack(offset_losses).sum()
        bg_corr = torch.einsum("ij,kj->ki", R_delta, cur_scene["bg"]) + delta_t[0]
        bg_chamfer = robust_bg_chamfer(bg_corr, gauge_scene["bg"], args.bg_chamfer_cap)
        temporal_t_loss = (t_corr - prev_t).pow(2).mean()
        temporal_r_loss = rotation_temporal_loss(R_corr, prev_R)
        prior_loss = delta_t.pow(2).mean() + delta_r.pow(2).mean()

        loss = (
            args.stable_weight * stable_loss
            + args.foot_weight * foot_loss
            + args.orient_weight * orient_loss
            + args.normal_weight * normal_loss
            + args.plane_offset_weight * offset_loss
            + args.bg_chamfer_weight * bg_chamfer
            + args.temporal_t_weight * temporal_t_loss
            + args.temporal_r_weight * temporal_r_loss
            + args.prior_weight * prior_loss
        )
        loss.backward()
        optimizer.step()

        if step % int(args.log_every) == 0 or step == int(args.steps_per_frame):
            history.append(
                {
                    "step": int(step),
                    "loss": float(loss.detach().cpu()),
                    "stable_loss": float(stable_loss.detach().cpu()),
                    "foot_loss": float(foot_loss.detach().cpu()),
                    "orient_loss": float(orient_loss.detach().cpu()),
                    "normal_loss": float(normal_loss.detach().cpu()),
                    "offset_loss": float(offset_loss.detach().cpu()),
                    "bg_chamfer": float(bg_chamfer.detach().cpu()),
                    "temporal_t_loss": float(temporal_t_loss.detach().cpu()),
                    "temporal_r_loss": float(temporal_r_loss.detach().cpu()),
                    "prior_loss": float(prior_loss.detach().cpu()),
                    "mean_normal_dot_abs": float(torch.stack(dots).mean().detach().cpu()),
                    "delta_t_norm": float(delta_t.norm().detach().cpu()),
                    "delta_r_deg": float(torch.rad2deg(delta_r.norm()).detach().cpu()),
                }
            )

    with torch.no_grad():
        delta_t = float(args.max_delta_t) * torch.tanh(raw_delta_t)
        delta_r = max_delta_r * torch.tanh(raw_delta_r)
        _, R_corr_one, t_corr_one = apply_correction(joints_world[frame_id : frame_id + 1], poses[frame_id : frame_id + 1], delta_t, delta_r)
        debug = {
            "frame": int(frame_id),
            "skipped": False,
            "matches": [
                {
                    "ref_index": int(m["ref_index"]),
                    "cur_index": int(m["cur_index"]),
                    "dot_abs": float(m["dot_abs"]),
                    "weight": float(m["weight"]),
                }
                for m in matches
            ],
            "history": history,
            "delta_t": delta_t[0].detach().cpu().numpy().astype(np.float32).tolist(),
            "delta_r": delta_r[0].detach().cpu().numpy().astype(np.float32).tolist(),
        }
    return R_corr_one[0].detach().cpu().numpy().astype(np.float32), t_corr_one[0].detach().cpu().numpy().astype(np.float32), debug


def scene_summary(scene: dict) -> dict:
    return {
        "frame": int(scene["frame"]),
        "num_points": int(scene["num_points"]),
        "planes": [jsonable_plane(p) for p in scene["planes"]],
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(53)
    np.random.seed(53)
    device = torch.device(args.device)
    num_frames = infer_num_frames(args.input_dir, args.source_video, args.num_frames)
    boundary = int(args.boundary)
    target_frames = list(range(boundary, min(num_frames, boundary + int(args.target_count))))
    stable_start = int(args.stable_start)
    stable_end = min(int(args.stable_end), num_frames - 1)
    gauge_anchor = stable_start if args.gauge_anchor_frame is None else int(args.gauge_anchor_frame)
    if boundary <= 0 or stable_start <= target_frames[-1] or stable_end < stable_start:
        raise ValueError("Expected boundary targets before a valid future stable window")

    data = load_sequence(args.input_dir, num_frames, device)
    corrected_poses = data.poses.copy()
    gauge_scene = build_scene(args.input_dir, gauge_anchor, data.poses[gauge_anchor], data.intrinsics[gauge_anchor], args, device, seed_base=7101)
    stable_scenes = []
    for frame_id in np.linspace(stable_start, stable_end, num=min(5, stable_end - stable_start + 1), dtype=int):
        stable_scenes.append(build_scene(args.input_dir, int(frame_id), data.poses[int(frame_id)], data.intrinsics[int(frame_id)], args, device, seed_base=7201))

    frame_debug = []
    for frame_id in target_frames:
        cur_scene = build_scene(args.input_dir, frame_id, data.poses[frame_id], data.intrinsics[frame_id], args, device, seed_base=7301)
        R_corr, t_corr, debug = optimize_target_frame(frame_id, data, gauge_scene, cur_scene, corrected_poses, args, device)
        corrected_poses[frame_id, :3, :3] = R_corr
        corrected_poses[frame_id, :3, 3] = t_corr
        frame_debug.append(debug)
        final = debug["history"][-1] if debug.get("history") else {}
        print(json.dumps({"frame": int(frame_id), "skipped": debug.get("skipped", False), **final}, sort_keys=True), flush=True)

    raw_metrics = compute_metrics(data, data.poses, boundary)
    raw_transition = compute_transition_metrics(data, data.poses, boundary)
    corrected_metrics = compute_metrics(data, corrected_poses, boundary)
    corrected_transition = compute_transition_metrics(data, corrected_poses, boundary)
    print("raw_transition_metrics", json.dumps(raw_transition, indent=2, sort_keys=True))
    print("teacher_transition_metrics", json.dumps(corrected_transition, indent=2, sort_keys=True))

    write_outputs_with_links(args.input_dir, args.output_dir, corrected_poses, data.intrinsics, list(range(num_frames)), args.overwrite)
    if args.subset_output_dir is not None:
        subset_start = max(0, boundary - 1) if args.subset_start is None else int(args.subset_start)
        subset_frames = list(range(subset_start, min(num_frames, subset_start + int(args.subset_count))))
        write_outputs_with_links(args.input_dir, args.subset_output_dir, corrected_poses, data.intrinsics, subset_frames, args.overwrite)
    if args.raw_subset_output_dir is not None:
        subset_start = max(0, boundary - 1) if args.subset_start is None else int(args.subset_start)
        subset_frames = list(range(subset_start, min(num_frames, subset_start + int(args.subset_count))))
        write_outputs_with_links(args.input_dir, args.raw_subset_output_dir, data.poses, data.intrinsics, subset_frames, args.overwrite)

    metrics = {
        "teacher_type": "offline_post_shot_local_gauge_teacher",
        "causal_inference_allowed": False,
        "note": "Uses future stable frames only to build pseudo labels / oracle targets.",
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "boundary": int(boundary),
        "target_frames": [int(x) for x in target_frames],
        "stable_start": int(stable_start),
        "stable_end": int(stable_end),
        "gauge_anchor_frame": int(gauge_anchor),
        "gauge_scene": scene_summary(gauge_scene),
        "stable_scenes": [scene_summary(s) for s in stable_scenes],
        "frames": frame_debug,
        "raw": raw_metrics,
        "corrected": corrected_metrics,
        "raw_transition": raw_transition,
        "corrected_transition": corrected_transition,
    }
    with open(args.output_dir / "post_shot_local_gauge_teacher_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    print(f"Wrote post-shot local gauge teacher output to {args.output_dir}")


if __name__ == "__main__":
    main()
