#!/usr/bin/env python3
"""Overfit camera correction from boundary-pair human consistency.

This is a second V7 smoke test.  Unlike
``overfit_human_anchor_pose_correction.py``, it does not use the whole
pre-boundary segment as a long-term static reference for every post-boundary
frame.  Instead, it trains a tiny MLP so adjacent corrected human anchors stay
continuous, with the strongest supervision at the shot boundary.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from overfit_human_anchor_pose_correction import (
    FOOT_JOINTS,
    STABLE_JOINTS,
    HumanAnchorCorrectionMLP,
    apply_correction,
    compute_metrics,
    copy_or_write_outputs,
    infer_num_frames,
    load_sequence,
    normalize_vectors,
    rotation_angle_deg,
    torso_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--source_video", type=Path, default=None)
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--boundary", type=int, required=True, help="First frame index after shot boundary.")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--max_delta_t", type=float, default=4.0)
    parser.add_argument("--max_delta_r_deg", type=float, default=70.0)
    parser.add_argument("--boundary_pair_weight", type=float, default=30.0)
    parser.add_argument("--post_pair_weight", type=float, default=3.0)
    parser.add_argument("--foot_pair_weight", type=float, default=2.0)
    parser.add_argument("--orient_pair_weight", type=float, default=6.0)
    parser.add_argument("--delta_smooth_weight", type=float, default=0.25)
    parser.add_argument("--camera_smooth_weight", type=float, default=0.05)
    parser.add_argument("--prior_weight", type=float, default=0.01)
    parser.add_argument("--pre_noop_weight", type=float, default=2.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log_every", type=int, default=250)
    return parser.parse_args()


def build_pair_features(data, boundary: int, device: torch.device) -> torch.Tensor:
    joints_world = torch.from_numpy(data.joints_world).to(device=device, dtype=torch.float32)
    poses = torch.from_numpy(data.poses).to(device=device, dtype=torch.float32)
    num_frames = joints_world.shape[0]

    stable = joints_world[:, STABLE_JOINTS]
    foot = joints_world[:, FOOT_JOINTS]
    frame = torso_frame(joints_world)

    stable_prev = torch.cat([stable[:1], stable[:-1]], dim=0)
    foot_prev = torch.cat([foot[:1], foot[:-1]], dim=0)
    frame_prev = torch.cat([frame[:1], frame[:-1]], dim=0)
    cam_t = poses[:, :3, 3]
    cam_t_prev = torch.cat([cam_t[:1], cam_t[:-1]], dim=0)

    stable_pair = (stable - stable_prev).flatten(1)
    foot_pair = (foot - foot_prev).flatten(1)
    frame_pair = (frame - frame_prev).flatten(1)
    cam_pair = cam_t - cam_t_prev
    cam_centered = cam_t - cam_t[:boundary].mean(dim=0, keepdim=True)

    idx = torch.arange(num_frames, device=device)
    t = torch.linspace(0.0, 1.0, num_frames, device=device).unsqueeze(1)
    post = (idx >= boundary).float().unsqueeze(1)
    boundary_flag = (idx == boundary).float().unsqueeze(1)
    pe = torch.cat([
        t,
        post,
        boundary_flag,
        torch.sin(2.0 * math.pi * t),
        torch.cos(2.0 * math.pi * t),
        torch.sin(4.0 * math.pi * t),
        torch.cos(4.0 * math.pi * t),
    ], dim=1)
    return torch.cat([stable_pair, foot_pair, frame_pair, cam_pair, cam_centered, pe], dim=1)


def pair_weights(num_frames: int, boundary: int, args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    weights = torch.zeros(num_frames - 1, device=device, dtype=torch.float32)
    # Pairs are indexed by their dst frame: pair k compares k -> k+1.
    dst = torch.arange(1, num_frames, device=device)
    weights = torch.where(dst >= boundary, torch.full_like(weights, args.post_pair_weight), weights)
    weights = torch.where(dst == boundary, torch.full_like(weights, args.boundary_pair_weight), weights)
    return weights


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    denom = weights.sum().clamp_min(1.0)
    return (values * weights).sum() / denom


def train_boundary_pair_correction(data, args: argparse.Namespace, device: torch.device) -> tuple[np.ndarray, dict]:
    boundary = int(args.boundary)
    num_frames = data.poses.shape[0]
    if boundary <= 0 or boundary >= num_frames:
        raise ValueError(f"boundary must be in [1, {num_frames - 1}], got {boundary}")

    poses = torch.from_numpy(data.poses).to(device=device, dtype=torch.float32)
    joints_world = torch.from_numpy(data.joints_world).to(device=device, dtype=torch.float32)
    features = build_pair_features(data, boundary, device)
    weights = pair_weights(num_frames, boundary, args, device)

    model = HumanAnchorCorrectionMLP(
        in_dim=features.shape[1],
        hidden_dim=args.hidden_dim,
        max_delta_t=args.max_delta_t,
        max_delta_r_deg=args.max_delta_r_deg,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    pre = slice(0, boundary)
    post = slice(boundary, num_frames)
    history = []
    for step in range(args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        delta_t, delta_r = model(features)
        joints_corr, R_corr, t_corr = apply_correction(joints_world, poses, delta_t, delta_r)

        stable_diff = joints_corr[1:, STABLE_JOINTS] - joints_corr[:-1, STABLE_JOINTS]
        stable_pair_loss = weighted_mean(F.smooth_l1_loss(stable_diff, torch.zeros_like(stable_diff), beta=0.05, reduction="none"), weights)

        # Feet can move or cross, so use set distance between adjacent foot-support points.
        foot_curr = joints_corr[1:, FOOT_JOINTS]
        foot_prev = joints_corr[:-1, FOOT_JOINTS]
        d = torch.cdist(foot_curr, foot_prev, p=2)
        foot_chamfer = 0.5 * (d.min(dim=2).values.mean(dim=1) + d.min(dim=1).values.mean(dim=1))
        foot_pair_loss = weighted_mean(foot_chamfer, weights)

        frame_corr = torso_frame(joints_corr)
        orient_dot = (frame_corr[1:] * frame_corr[:-1]).sum(dim=-1).clamp(-1.0, 1.0)
        orient_pair_loss = weighted_mean(1.0 - orient_dot, weights)

        # Correction should propagate smoothly inside each shot, but no smoothing is imposed across boundary.
        smooth_losses = []
        if boundary > 2:
            smooth_losses.append((delta_t[1:boundary] - delta_t[: boundary - 1]).pow(2).mean())
            smooth_losses.append((delta_r[1:boundary] - delta_r[: boundary - 1]).pow(2).mean())
        if num_frames - boundary > 2:
            smooth_losses.append((delta_t[boundary + 1 :] - delta_t[boundary:-1]).pow(2).mean())
            smooth_losses.append((delta_r[boundary + 1 :] - delta_r[boundary:-1]).pow(2).mean())
        delta_smooth_loss = torch.stack(smooth_losses).mean() if smooth_losses else delta_t.new_tensor(0.0)

        cam_smooth_losses = []
        if boundary > 2:
            cam_smooth_losses.append((t_corr[1:boundary] - t_corr[: boundary - 1]).pow(2).mean())
        if num_frames - boundary > 2:
            cam_smooth_losses.append((t_corr[boundary + 1 :] - t_corr[boundary:-1]).pow(2).mean())
        camera_smooth_loss = torch.stack(cam_smooth_losses).mean() if cam_smooth_losses else delta_t.new_tensor(0.0)

        prior_loss = delta_t.pow(2).mean() + delta_r.pow(2).mean()
        pre_noop_loss = delta_t[pre].pow(2).mean() + delta_r[pre].pow(2).mean()

        loss = (
            stable_pair_loss
            + args.foot_pair_weight * foot_pair_loss
            + args.orient_pair_weight * orient_pair_loss
            + args.delta_smooth_weight * delta_smooth_loss
            + args.camera_smooth_weight * camera_smooth_loss
            + args.prior_weight * prior_loss
            + args.pre_noop_weight * pre_noop_loss
        )
        loss.backward()
        optimizer.step()

        if step % args.log_every == 0 or step == args.steps:
            with torch.no_grad():
                boundary_stable = stable_diff[boundary - 1].norm(dim=-1).mean()
                boundary_foot = foot_chamfer[boundary - 1]
                boundary_orient = (1.0 - orient_dot[boundary - 1]).mean()
            record = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "stable_pair_loss": float(stable_pair_loss.detach().cpu()),
                "foot_pair_loss": float(foot_pair_loss.detach().cpu()),
                "orient_pair_loss": float(orient_pair_loss.detach().cpu()),
                "delta_smooth_loss": float(delta_smooth_loss.detach().cpu()),
                "camera_smooth_loss": float(camera_smooth_loss.detach().cpu()),
                "pre_noop_loss": float(pre_noop_loss.detach().cpu()),
                "delta_t_norm_post": float(delta_t[post].norm(dim=-1).mean().detach().cpu()),
                "delta_r_deg_post": float(torch.rad2deg(delta_r[post].norm(dim=-1)).mean().detach().cpu()),
                "boundary_stable_joint_dist": float(boundary_stable.detach().cpu()),
                "boundary_foot_chamfer": float(boundary_foot.detach().cpu()),
                "boundary_orient_loss": float(boundary_orient.detach().cpu()),
            }
            print(json.dumps(record, sort_keys=True))
            history.append(record)

    with torch.no_grad():
        delta_t, delta_r = model(features)
        _, R_corr, t_corr = apply_correction(joints_world, poses, delta_t, delta_r)
        corrected = poses.detach().cpu().numpy().copy()
        corrected[:, :3, :3] = R_corr.detach().cpu().numpy().astype(np.float32)
        corrected[:, :3, 3] = t_corr.detach().cpu().numpy().astype(np.float32)
        delta_stats = {
            "delta_t": delta_t.detach().cpu().numpy().astype(np.float32),
            "delta_r": delta_r.detach().cpu().numpy().astype(np.float32),
            "history": history,
        }
    return corrected, delta_stats


def main() -> None:
    args = parse_args()
    torch.manual_seed(11)
    np.random.seed(11)
    device = torch.device(args.device)
    num_frames = infer_num_frames(args.input_dir, args.source_video, args.num_frames)
    print(f"Using {num_frames} frames from {args.input_dir}")
    data = load_sequence(args.input_dir, num_frames, device)

    raw_metrics = compute_metrics(data, data.poses, args.boundary)
    print("raw_metrics", json.dumps(raw_metrics, indent=2, sort_keys=True))
    corrected_poses, delta_stats = train_boundary_pair_correction(data, args, device)
    corrected_metrics = compute_metrics(data, corrected_poses, args.boundary)
    print("corrected_metrics", json.dumps(corrected_metrics, indent=2, sort_keys=True))

    copy_or_write_outputs(args.input_dir, args.output_dir, corrected_poses, data.intrinsics, num_frames, args.overwrite)
    np.savez(
        args.output_dir / "boundary_pair_human_correction_debug.npz",
        delta_t=delta_stats["delta_t"],
        delta_r=delta_stats["delta_r"],
        raw_metrics=np.array(json.dumps(raw_metrics), dtype=object),
        corrected_metrics=np.array(json.dumps(corrected_metrics), dtype=object),
    )
    with open(args.output_dir / "boundary_pair_human_correction_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"raw": raw_metrics, "corrected": corrected_metrics, "history": delta_stats["history"]}, f, indent=2, sort_keys=True)
    print(f"Wrote boundary-pair corrected output to {args.output_dir}")


if __name__ == "__main__":
    main()
