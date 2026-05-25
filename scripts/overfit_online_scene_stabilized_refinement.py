#!/usr/bin/env python3
"""Causal post-boundary scene-stabilized camera refinement.

This smoke test targets the smaller post-boundary drift after the first boundary
frame has been fixed. It uses a past scene-memory anchor and current-frame
background geometry only. Human root/world position is intentionally not used as
an absolute constraint, so real human motion is not suppressed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from overfit_human_anchor_pose_correction import infer_num_frames, load_sequence, so3_exp_map
from overfit_single_boundary_frame_human_anchor import write_outputs_with_links
from overfit_single_boundary_frame_scene_geometry import estimate_top_planes, match_planes, sample_points
from overfit_single_boundary_frame_scene_geometry import robust_bg_chamfer as robust_bg_chamfer_torch
from overfit_single_boundary_frame_scene_geometry import scene_json as single_pair_scene_json
from overfit_single_boundary_frame_scene_normal import jsonable_plane, load_background_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--source_video", type=Path, default=None)
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--anchor_frame", type=int, required=True, help="Past frame used as fixed scene memory.")
    parser.add_argument("--start_frame", type=int, default=None, help="First frame to refine. Defaults to anchor_frame+1.")
    parser.add_argument("--end_frame", type=int, default=None, help="Inclusive end frame. Defaults to sequence end.")
    parser.add_argument("--steps_per_frame", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--max_delta_t", type=float, default=1.5)
    parser.add_argument("--max_delta_r_deg", type=float, default=8.0)
    parser.add_argument("--normal_weight", type=float, default=10.0)
    parser.add_argument("--plane_offset_weight", type=float, default=5.0)
    parser.add_argument("--bg_chamfer_weight", type=float, default=2.0)
    parser.add_argument("--bg_chamfer_cap", type=float, default=0.35)
    parser.add_argument("--smooth_t_weight", type=float, default=0.20)
    parser.add_argument("--smooth_r_weight", type=float, default=0.20)
    parser.add_argument("--prior_weight", type=float, default=0.08)
    parser.add_argument("--conf_threshold", type=float, default=1.5)
    parser.add_argument("--mask_threshold", type=float, default=0.1)
    parser.add_argument("--plane_max_points", type=int, default=12000)
    parser.add_argument("--plane_iterations", type=int, default=768)
    parser.add_argument("--plane_threshold", type=float, default=0.04)
    parser.add_argument("--num_planes", type=int, default=3)
    parser.add_argument("--min_plane_inlier_ratio", type=float, default=0.03)
    parser.add_argument("--min_plane_dot", type=float, default=0.9)
    parser.add_argument("--bg_chamfer_points", type=int, default=700)
    parser.add_argument("--subset_output_dir", type=Path, default=None)
    parser.add_argument("--raw_subset_output_dir", type=Path, default=None)
    parser.add_argument("--raw_input_dir", type=Path, default=None, help="Optional raw output dir for raw subset overlay.")
    parser.add_argument("--subset_start", type=int, default=None)
    parser.add_argument("--subset_count", type=int, default=31)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log_every", type=int, default=50)
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
    planes = estimate_top_planes(points, plane_args, seed=seed_base + frame_id * 19)
    sampled = sample_points(points, args.bg_chamfer_points, seed=seed_base + frame_id * 19 + 7)
    return {
        "num_points": int(points.shape[0]),
        "planes": planes,
        "bg": torch.from_numpy(sampled).to(device=device, dtype=torch.float32),
    }


def make_match_tensors(anchor_scene: dict, cur_scene: dict, min_dot: float, device: torch.device) -> tuple[list[dict], list[dict]]:
    matches = match_planes(anchor_scene["planes"], cur_scene["planes"], min_dot)
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


def rotation_smooth_loss(R_corr: torch.Tensor, R_prev: torch.Tensor) -> torch.Tensor:
    rel = R_corr @ R_prev.transpose(0, 1)
    trace = torch.diagonal(rel, dim1=-2, dim2=-1).sum()
    cos = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    return 1.0 - cos


def optimize_frame(
    frame_id: int,
    data,
    anchor_scene: dict,
    cur_scene: dict,
    prev_R_corr: torch.Tensor,
    prev_t_corr: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict]:
    matches, match_tensors = make_match_tensors(anchor_scene, cur_scene, args.min_plane_dot, device)
    if not match_tensors:
        pose = data.poses[frame_id]
        return pose[:3, :3].copy(), pose[:3, 3].copy(), {"frame": frame_id, "skipped": True, "reason": "no_plane_matches"}

    raw_R = torch.from_numpy(data.poses[frame_id, :3, :3]).to(device=device, dtype=torch.float32)
    raw_t = torch.from_numpy(data.poses[frame_id, :3, 3]).to(device=device, dtype=torch.float32)
    raw_delta_t = torch.nn.Parameter(torch.zeros(1, 3, device=device))
    raw_delta_r = torch.nn.Parameter(torch.zeros(1, 3, device=device))
    optimizer = torch.optim.AdamW([raw_delta_t, raw_delta_r], lr=args.lr, weight_decay=1e-4)
    max_delta_r = math.radians(float(args.max_delta_r_deg))
    history = []

    for step in range(int(args.steps_per_frame) + 1):
        optimizer.zero_grad(set_to_none=True)
        delta_t = float(args.max_delta_t) * torch.tanh(raw_delta_t)[0]
        delta_r = max_delta_r * torch.tanh(raw_delta_r)
        R_delta = so3_exp_map(delta_r)[0]
        R_corr = R_delta @ raw_R
        t_corr = R_delta @ raw_t + delta_t

        normal_losses, offset_losses, dots = [], [], []
        for m in match_tensors:
            n_corr = R_delta @ m["n_cur"]
            normal_dot = (n_corr * m["n_ref"]).sum().clamp(-1.0, 1.0)
            d_corr = m["d_cur"] - (n_corr * delta_t).sum()
            normal_losses.append(m["weight"] * (1.0 - normal_dot.abs()))
            offset_losses.append(m["weight"] * torch.minimum((d_corr - m["d_ref"]).pow(2), (d_corr + m["d_ref"]).pow(2)))
            dots.append(normal_dot.abs())
        normal_loss = torch.stack(normal_losses).sum()
        offset_loss = torch.stack(offset_losses).sum()
        bg_corr = torch.einsum("ij,kj->ki", R_delta, cur_scene["bg"]) + delta_t
        bg_chamfer = robust_bg_chamfer_torch(bg_corr, anchor_scene["bg"], args.bg_chamfer_cap)
        smooth_t_loss = (t_corr - prev_t_corr).pow(2).mean()
        smooth_r_loss = rotation_smooth_loss(R_corr, prev_R_corr)
        prior_loss = delta_t.pow(2).mean() + delta_r.pow(2).mean()

        loss = (
            args.normal_weight * normal_loss
            + args.plane_offset_weight * offset_loss
            + args.bg_chamfer_weight * bg_chamfer
            + args.smooth_t_weight * smooth_t_loss
            + args.smooth_r_weight * smooth_r_loss
            + args.prior_weight * prior_loss
        )
        loss.backward()
        optimizer.step()

        if step % int(args.log_every) == 0 or step == int(args.steps_per_frame):
            history.append(
                {
                    "step": int(step),
                    "loss": float(loss.detach().cpu()),
                    "normal_loss": float(normal_loss.detach().cpu()),
                    "offset_loss": float(offset_loss.detach().cpu()),
                    "bg_chamfer": float(bg_chamfer.detach().cpu()),
                    "smooth_t_loss": float(smooth_t_loss.detach().cpu()),
                    "smooth_r_loss": float(smooth_r_loss.detach().cpu()),
                    "prior_loss": float(prior_loss.detach().cpu()),
                    "mean_normal_dot_abs": float(torch.stack(dots).mean().detach().cpu()),
                    "delta_t_norm": float(delta_t.norm().detach().cpu()),
                    "delta_r_deg": float(torch.rad2deg(delta_r.norm()).detach().cpu()),
                }
            )

    with torch.no_grad():
        delta_t = float(args.max_delta_t) * torch.tanh(raw_delta_t)[0]
        delta_r = max_delta_r * torch.tanh(raw_delta_r)
        R_delta = so3_exp_map(delta_r)[0]
        R_corr = R_delta @ raw_R
        t_corr = R_delta @ raw_t + delta_t
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
            "delta_t": delta_t.detach().cpu().numpy().astype(np.float32).tolist(),
            "delta_r": delta_r.detach().cpu().numpy().astype(np.float32).tolist(),
        }
    return R_corr.detach().cpu().numpy().astype(np.float32), t_corr.detach().cpu().numpy().astype(np.float32), debug


def main() -> None:
    args = parse_args()
    torch.manual_seed(47)
    np.random.seed(47)
    device = torch.device(args.device)
    num_frames = infer_num_frames(args.input_dir, args.source_video, args.num_frames)
    start_frame = int(args.anchor_frame) + 1 if args.start_frame is None else int(args.start_frame)
    end_frame = num_frames - 1 if args.end_frame is None else min(int(args.end_frame), num_frames - 1)
    if not (0 <= int(args.anchor_frame) < start_frame <= end_frame < num_frames):
        raise ValueError(f"Invalid frame range: anchor={args.anchor_frame}, start={start_frame}, end={end_frame}, num_frames={num_frames}")

    data = load_sequence(args.input_dir, num_frames, device)
    corrected_poses = data.poses.copy()
    anchor_scene = build_scene(args.input_dir, int(args.anchor_frame), corrected_poses[int(args.anchor_frame)], data.intrinsics[int(args.anchor_frame)], args, device, seed_base=5101)

    frame_debug = []
    for frame_id in range(start_frame, end_frame + 1):
        cur_scene = build_scene(args.input_dir, frame_id, data.poses[frame_id], data.intrinsics[frame_id], args, device, seed_base=6101)
        prev_R = torch.from_numpy(corrected_poses[frame_id - 1, :3, :3]).to(device=device, dtype=torch.float32)
        prev_t = torch.from_numpy(corrected_poses[frame_id - 1, :3, 3]).to(device=device, dtype=torch.float32)
        R_corr, t_corr, debug = optimize_frame(frame_id, data, anchor_scene, cur_scene, prev_R, prev_t, args, device)
        corrected_poses[frame_id, :3, :3] = R_corr
        corrected_poses[frame_id, :3, 3] = t_corr
        frame_debug.append(debug)
        final = debug["history"][-1] if debug.get("history") else {}
        print(json.dumps({"frame": frame_id, "skipped": debug.get("skipped", False), **final}, sort_keys=True), flush=True)

    all_frames = list(range(num_frames))
    write_outputs_with_links(args.input_dir, args.output_dir, corrected_poses, data.intrinsics, all_frames, args.overwrite)

    if args.subset_output_dir is not None:
        subset_start = int(args.anchor_frame) - 3 if args.subset_start is None else int(args.subset_start)
        subset_start = max(0, subset_start)
        subset_frames = list(range(subset_start, min(num_frames, subset_start + int(args.subset_count))))
        write_outputs_with_links(args.input_dir, args.subset_output_dir, corrected_poses, data.intrinsics, subset_frames, args.overwrite)
    if args.raw_subset_output_dir is not None:
        raw_input = args.raw_input_dir if args.raw_input_dir is not None else args.input_dir
        raw_num_frames = infer_num_frames(raw_input, args.source_video, args.num_frames)
        raw_data = load_sequence(raw_input, raw_num_frames, device)
        subset_start = int(args.anchor_frame) - 3 if args.subset_start is None else int(args.subset_start)
        subset_start = max(0, subset_start)
        subset_frames = list(range(subset_start, min(raw_num_frames, subset_start + int(args.subset_count))))
        write_outputs_with_links(raw_input, args.raw_subset_output_dir, raw_data.poses, raw_data.intrinsics, subset_frames, args.overwrite)

    metrics = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "anchor_frame": int(args.anchor_frame),
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "anchor_scene": {
            "num_points": int(anchor_scene["num_points"]),
            "planes": [jsonable_plane(p) for p in anchor_scene["planes"]],
        },
        "frames": frame_debug,
    }
    with open(args.output_dir / "online_scene_stabilized_refinement_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    print(f"Wrote online scene-stabilized refinement to {args.output_dir}")


if __name__ == "__main__":
    main()
