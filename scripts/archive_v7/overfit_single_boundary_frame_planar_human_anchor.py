#!/usr/bin/env python3
"""Overfit a planar human-anchor correction for one boundary frame only.

This is the constrained version of
``overfit_single_boundary_frame_human_anchor.py``.  It still uses only the
boundary pair (boundary-1 -> boundary) as the human anchor constraint, but it
does not allow arbitrary SE(3):

    allowed:   horizontal translation in the reference body plane + yaw
    forbidden: vertical translation + pitch / roll tilt

Only the boundary frame is corrected.  All other frames are left unchanged.
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
    parser.add_argument("--max_horizontal_t", type=float, default=6.0)
    parser.add_argument("--max_yaw_deg", type=float, default=90.0)
    parser.add_argument("--stable_weight", type=float, default=1.0)
    parser.add_argument("--foot_weight", type=float, default=2.0)
    parser.add_argument("--orient_weight", type=float, default=5.0)
    parser.add_argument("--prior_weight", type=float, default=0.01)
    parser.add_argument("--subset_output_dir", type=Path, default=None)
    parser.add_argument("--raw_subset_output_dir", type=Path, default=None)
    parser.add_argument("--subset_start", type=int, default=None)
    parser.add_argument("--subset_count", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log_every", type=int, default=200)
    return parser.parse_args()


def planar_correction(
    joints_world_one: torch.Tensor,
    pose_one: torch.Tensor,
    raw_horizontal: torch.Tensor,
    raw_yaw: torch.Tensor,
    ref_basis: torch.Tensor,
    max_horizontal_t: float,
    max_yaw: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    right = ref_basis[0]
    up = ref_basis[1]
    forward = ref_basis[2]
    horizontal = float(max_horizontal_t) * torch.tanh(raw_horizontal[0])
    yaw = float(max_yaw) * torch.tanh(raw_yaw[0, 0])
    delta_t = horizontal[0] * right + horizontal[1] * forward
    delta_r = yaw * up
    R_delta = so3_exp_map(delta_r.unsqueeze(0))[0]
    joints_corr = torch.einsum("ij,kj->ki", R_delta, joints_world_one) + delta_t[None]
    R_corr = R_delta @ pose_one[:3, :3]
    t_corr = R_delta @ pose_one[:3, 3] + delta_t
    return joints_corr, R_corr, t_corr, delta_t, delta_r


def train_planar_single_frame(data, args: argparse.Namespace, device: torch.device) -> tuple[np.ndarray, dict]:
    boundary = int(args.boundary)
    num_frames = data.poses.shape[0]
    if boundary <= 0 or boundary >= num_frames:
        raise ValueError(f"boundary must be in [1, {num_frames - 1}], got {boundary}")

    poses = torch.from_numpy(data.poses).to(device=device, dtype=torch.float32)
    joints_world = torch.from_numpy(data.joints_world).to(device=device, dtype=torch.float32)
    raw_horizontal = torch.nn.Parameter(torch.zeros(1, 2, device=device))
    raw_yaw = torch.nn.Parameter(torch.zeros(1, 1, device=device))
    optimizer = torch.optim.AdamW([raw_horizontal, raw_yaw], lr=args.lr, weight_decay=1e-4)
    max_yaw = math.radians(float(args.max_yaw_deg))

    ref_joints = joints_world[boundary - 1]
    ref_basis = torso_frame(joints_world[boundary - 1 : boundary])[0].detach()
    ref_frame = ref_basis
    history = []
    for step in range(args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        corrected_one, _, _, delta_t, delta_r = planar_correction(
            joints_world[boundary],
            poses[boundary],
            raw_horizontal,
            raw_yaw,
            ref_basis,
            args.max_horizontal_t,
            max_yaw,
        )
        frame_corr = torso_frame(corrected_one.unsqueeze(0))[0]

        stable_loss = F.smooth_l1_loss(corrected_one[STABLE_JOINTS], ref_joints[STABLE_JOINTS], beta=0.05)
        foot_loss = chamfer_per_frame(corrected_one[FOOT_JOINTS], ref_joints[FOOT_JOINTS])
        orient_loss = (1.0 - (frame_corr * ref_frame).sum(dim=-1).clamp(-1.0, 1.0)).mean()
        prior_loss = delta_t.pow(2).mean() + delta_r.pow(2).mean()
        loss = args.stable_weight * stable_loss + args.foot_weight * foot_loss + args.orient_weight * orient_loss + args.prior_weight * prior_loss
        loss.backward()
        optimizer.step()

        if step % args.log_every == 0 or step == args.steps:
            record = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "stable_loss": float(stable_loss.detach().cpu()),
                "foot_loss": float(foot_loss.detach().cpu()),
                "orient_loss": float(orient_loss.detach().cpu()),
                "prior_loss": float(prior_loss.detach().cpu()),
                "horizontal_t_norm": float(delta_t.norm().detach().cpu()),
                "yaw_deg": float(torch.rad2deg(delta_r.norm()).detach().cpu()),
                "vertical_t_along_ref_up": float((delta_t * ref_basis[1]).sum().detach().cpu()),
            }
            print(json.dumps(record, sort_keys=True), flush=True)
            history.append(record)

    with torch.no_grad():
        _, R_corr_one, t_corr_one, delta_t, delta_r = planar_correction(
            joints_world[boundary],
            poses[boundary],
            raw_horizontal,
            raw_yaw,
            ref_basis,
            args.max_horizontal_t,
            max_yaw,
        )
        corrected_poses = data.poses.copy()
        corrected_poses[boundary, :3, :3] = R_corr_one.detach().cpu().numpy().astype(np.float32)
        corrected_poses[boundary, :3, 3] = t_corr_one.detach().cpu().numpy().astype(np.float32)
        debug = {
            "delta_t": delta_t.detach().cpu().numpy().astype(np.float32),
            "delta_r": delta_r.detach().cpu().numpy().astype(np.float32),
            "ref_basis": ref_basis.detach().cpu().numpy().astype(np.float32),
            "history": history,
        }
    return corrected_poses, debug


def main() -> None:
    args = parse_args()
    torch.manual_seed(37)
    np.random.seed(37)
    device = torch.device(args.device)
    num_frames = infer_num_frames(args.input_dir, args.source_video, args.num_frames)
    print(f"Using {num_frames} frames from {args.input_dir}")
    data = load_sequence(args.input_dir, num_frames, device)

    raw_metrics = compute_metrics(data, data.poses, args.boundary)
    raw_transition_metrics = compute_transition_metrics(data, data.poses, args.boundary)
    print("raw_metrics", json.dumps(raw_metrics, indent=2, sort_keys=True))
    print("raw_transition_metrics", json.dumps(raw_transition_metrics, indent=2, sort_keys=True))

    corrected_poses, debug = train_planar_single_frame(data, args, device)
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
        args.output_dir / "single_boundary_frame_planar_human_anchor_debug.npz",
        delta_t=debug["delta_t"],
        delta_r=debug["delta_r"],
        ref_basis=debug["ref_basis"],
        raw_metrics=np.array(json.dumps(raw_metrics), dtype=object),
        corrected_metrics=np.array(json.dumps(corrected_metrics), dtype=object),
        raw_transition_metrics=np.array(json.dumps(raw_transition_metrics), dtype=object),
        corrected_transition_metrics=np.array(json.dumps(corrected_transition_metrics), dtype=object),
    )
    with open(args.output_dir / "single_boundary_frame_planar_human_anchor_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "raw": raw_metrics,
                "corrected": corrected_metrics,
                "raw_transition": raw_transition_metrics,
                "corrected_transition": corrected_transition_metrics,
                "history": debug["history"],
                "corrected_frame": int(args.boundary),
                "reference_frame": int(args.boundary - 1),
                "allowed_dof": "horizontal translation in frame boundary-1 body plane + yaw around frame boundary-1 up axis",
            },
            f,
            indent=2,
            sort_keys=True,
        )
    print(f"Wrote planar single-frame corrected output to {args.output_dir}")


if __name__ == "__main__":
    main()
