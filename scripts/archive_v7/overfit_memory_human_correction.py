#!/usr/bin/env python3
"""Overfit causal human-memory pose correction on saved Human3R output.

This third V7 smoke test addresses a limitation of pure boundary-pair matching:
the first post-boundary frame can be corrected, but later frames may drift if they
continue to follow the raw Human3R state.  Here each post-boundary frame reads a
corrected human memory, predicts a residual, then writes the corrected anchors
back into memory with an EMA update.
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
    HumanAnchorCorrectionMLP,
    apply_correction,
    compute_metrics,
    copy_or_write_outputs,
    infer_num_frames,
    load_sequence,
    normalize_vectors,
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
    parser.add_argument("--memory_alpha", type=float, default=0.18, help="EMA update weight for corrected human memory.")
    parser.add_argument("--boundary_weight", type=float, default=8.0)
    parser.add_argument("--post_weight", type=float, default=2.0)
    parser.add_argument("--foot_weight", type=float, default=2.0)
    parser.add_argument("--orient_weight", type=float, default=5.0)
    parser.add_argument("--stable_weight", type=float, default=1.0)
    parser.add_argument("--delta_smooth_weight", type=float, default=0.3)
    parser.add_argument("--camera_smooth_weight", type=float, default=0.08)
    parser.add_argument("--prior_weight", type=float, default=0.01)
    parser.add_argument("--pre_noop_weight", type=float, default=2.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log_every", type=int, default=250)
    parser.add_argument("--mode", choices=["fast", "mlp"], default="fast", help="fast directly optimizes per-frame residuals; mlp keeps the original sequential MLP smoke test.")
    parser.add_argument("--residual_lr", type=float, default=5e-2, help="Learning rate for --mode fast per-frame residual parameters.")
    return parser.parse_args()


def chamfer_per_frame(points: torch.Tensor, ref_points: torch.Tensor) -> torch.Tensor:
    d = torch.cdist(points.unsqueeze(0), ref_points.unsqueeze(0), p=2)[0]
    return 0.5 * (d.min(dim=1).values.mean() + d.min(dim=0).values.mean())


def build_memory_feature(
    raw_joints_world_t: torch.Tensor,
    raw_cam_t: torch.Tensor,
    raw_cam_prev_t: torch.Tensor,
    memory_stable: torch.Tensor,
    memory_foot: torch.Tensor,
    memory_frame: torch.Tensor,
    frame_index: int,
    num_frames: int,
    boundary: int,
    device: torch.device,
) -> torch.Tensor:
    raw_frame = torso_frame(raw_joints_world_t.unsqueeze(0))[0]
    t = torch.tensor([frame_index / max(num_frames - 1, 1)], device=device, dtype=raw_joints_world_t.dtype)
    post = torch.tensor([1.0 if frame_index >= boundary else 0.0], device=device, dtype=raw_joints_world_t.dtype)
    boundary_flag = torch.tensor([1.0 if frame_index == boundary else 0.0], device=device, dtype=raw_joints_world_t.dtype)
    pe = torch.cat([
        t,
        post,
        boundary_flag,
        torch.sin(2.0 * math.pi * t),
        torch.cos(2.0 * math.pi * t),
        torch.sin(4.0 * math.pi * t),
        torch.cos(4.0 * math.pi * t),
    ])
    return torch.cat([
        (raw_joints_world_t[STABLE_JOINTS] - memory_stable).flatten(),
        (raw_joints_world_t[FOOT_JOINTS] - memory_foot).flatten(),
        (raw_frame - memory_frame).flatten(),
        raw_cam_t - raw_cam_prev_t,
        raw_cam_t,
        pe,
    ]).unsqueeze(0)


def sequential_forward(
    model: HumanAnchorCorrectionMLP,
    joints_world: torch.Tensor,
    poses: torch.Tensor,
    boundary: int,
    memory_alpha: float,
):
    num_frames = joints_world.shape[0]
    device = joints_world.device
    dtype = joints_world.dtype

    zero_t = torch.zeros(1, 3, device=device, dtype=dtype)
    zero_r = torch.zeros(1, 3, device=device, dtype=dtype)
    delta_t_list = [zero_t[0] for _ in range(boundary)]
    delta_r_list = [zero_r[0] for _ in range(boundary)]
    corrected_list = [joints_world[i] for i in range(boundary)]

    memory_stable = joints_world[boundary - 1, STABLE_JOINTS]
    memory_foot = joints_world[boundary - 1, FOOT_JOINTS]
    memory_frame = torso_frame(joints_world[boundary - 1 : boundary])[0]
    memory_alpha_t = torch.tensor(float(memory_alpha), device=device, dtype=dtype)

    loss_records = []
    prev_delta_t = zero_t[0]
    prev_delta_r = zero_r[0]
    prev_t_corr = poses[boundary - 1, :3, 3]

    for frame_index in range(boundary, num_frames):
        feature = build_memory_feature(
            joints_world[frame_index],
            poses[frame_index, :3, 3],
            poses[frame_index - 1, :3, 3],
            memory_stable,
            memory_foot,
            memory_frame,
            frame_index,
            num_frames,
            boundary,
            device,
        )
        delta_t, delta_r = model(feature)
        corrected, _, t_corr = apply_correction(
            joints_world[frame_index : frame_index + 1],
            poses[frame_index : frame_index + 1],
            delta_t,
            delta_r,
        )
        corrected = corrected[0]
        t_corr = t_corr[0]

        stable_loss = F.smooth_l1_loss(corrected[STABLE_JOINTS], memory_stable, beta=0.05)
        foot_loss = chamfer_per_frame(corrected[FOOT_JOINTS], memory_foot)
        frame_corr = torso_frame(corrected.unsqueeze(0))[0]
        orient_loss = (1.0 - (frame_corr * memory_frame).sum(dim=-1).clamp(-1.0, 1.0)).mean()
        delta_smooth_loss = (delta_t[0] - prev_delta_t).pow(2).mean() + (delta_r[0] - prev_delta_r).pow(2).mean()
        camera_smooth_loss = (t_corr - prev_t_corr).pow(2).mean()

        frame_weight = 8.0 if frame_index == boundary else 1.0
        loss_records.append({
            "frame_weight": corrected.new_tensor(frame_weight),
            "stable": stable_loss,
            "foot": foot_loss,
            "orient": orient_loss,
            "delta_smooth": delta_smooth_loss,
            "camera_smooth": camera_smooth_loss,
            "boundary_stable_dist": (corrected[STABLE_JOINTS] - memory_stable).norm(dim=-1).mean(),
            "boundary_foot_chamfer": foot_loss,
            "boundary_orient_loss": orient_loss,
        })

        delta_t_list.append(delta_t[0])
        delta_r_list.append(delta_r[0])
        corrected_list.append(corrected)
        prev_delta_t = delta_t[0]
        prev_delta_r = delta_r[0]
        prev_t_corr = t_corr

        # Write corrected anchors back into causal memory.  This is the key
        # difference from pure boundary-pair correction.
        # The state cache is an online memory, not a target optimized by future
        # frames. Detaching keeps this smoke test fast and mirrors deployment:
        # each frame reads the previous corrected state as fixed context.
        memory_stable = ((1.0 - memory_alpha_t) * memory_stable + memory_alpha_t * corrected[STABLE_JOINTS]).detach()
        memory_foot = ((1.0 - memory_alpha_t) * memory_foot + memory_alpha_t * corrected[FOOT_JOINTS]).detach()
        memory_frame = normalize_vectors((1.0 - memory_alpha_t) * memory_frame + memory_alpha_t * frame_corr).detach()

    return (
        torch.stack(delta_t_list, dim=0),
        torch.stack(delta_r_list, dim=0),
        torch.stack(corrected_list, dim=0),
        loss_records,
    )


def train_memory_correction(data, args: argparse.Namespace, device: torch.device) -> tuple[np.ndarray, dict]:
    boundary = int(args.boundary)
    num_frames = data.poses.shape[0]
    if boundary <= 0 or boundary >= num_frames:
        raise ValueError(f"boundary must be in [1, {num_frames - 1}], got {boundary}")

    poses = torch.from_numpy(data.poses).to(device=device, dtype=torch.float32)
    joints_world = torch.from_numpy(data.joints_world).to(device=device, dtype=torch.float32)

    # Infer feature dimension from one memory feature.
    feature0 = build_memory_feature(
        joints_world[boundary],
        poses[boundary, :3, 3],
        poses[boundary - 1, :3, 3],
        joints_world[boundary - 1, STABLE_JOINTS],
        joints_world[boundary - 1, FOOT_JOINTS],
        torso_frame(joints_world[boundary - 1 : boundary])[0],
        boundary,
        num_frames,
        boundary,
        device,
    )
    model = HumanAnchorCorrectionMLP(
        in_dim=feature0.shape[1],
        hidden_dim=args.hidden_dim,
        max_delta_t=args.max_delta_t,
        max_delta_r_deg=args.max_delta_r_deg,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    for step in range(args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        delta_t, delta_r, _, records = sequential_forward(model, joints_world, poses, boundary, args.memory_alpha)

        stable_losses, foot_losses, orient_losses = [], [], []
        delta_smooth_losses, camera_smooth_losses = [], []
        for rec in records:
            w = args.boundary_weight if float(rec["frame_weight"].detach().cpu()) > 1.0 else args.post_weight
            stable_losses.append(w * rec["stable"])
            foot_losses.append(w * rec["foot"])
            orient_losses.append(w * rec["orient"])
            delta_smooth_losses.append(rec["delta_smooth"])
            camera_smooth_losses.append(rec["camera_smooth"])

        stable_loss = torch.stack(stable_losses).mean()
        foot_loss = torch.stack(foot_losses).mean()
        orient_loss = torch.stack(orient_losses).mean()
        delta_smooth_loss = torch.stack(delta_smooth_losses).mean()
        camera_smooth_loss = torch.stack(camera_smooth_losses).mean()
        prior_loss = delta_t.pow(2).mean() + delta_r.pow(2).mean()
        pre_noop_loss = delta_t[:boundary].pow(2).mean() + delta_r[:boundary].pow(2).mean()

        loss = (
            args.stable_weight * stable_loss
            + args.foot_weight * foot_loss
            + args.orient_weight * orient_loss
            + args.delta_smooth_weight * delta_smooth_loss
            + args.camera_smooth_weight * camera_smooth_loss
            + args.prior_weight * prior_loss
            + args.pre_noop_weight * pre_noop_loss
        )
        loss.backward()
        optimizer.step()

        if step % args.log_every == 0 or step == args.steps:
            boundary_rec = records[0]
            record = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "stable_loss": float(stable_loss.detach().cpu()),
                "foot_loss": float(foot_loss.detach().cpu()),
                "orient_loss": float(orient_loss.detach().cpu()),
                "delta_smooth_loss": float(delta_smooth_loss.detach().cpu()),
                "camera_smooth_loss": float(camera_smooth_loss.detach().cpu()),
                "pre_noop_loss": float(pre_noop_loss.detach().cpu()),
                "delta_t_norm_post": float(delta_t[boundary:].norm(dim=-1).mean().detach().cpu()),
                "delta_r_deg_post": float(torch.rad2deg(delta_r[boundary:].norm(dim=-1)).mean().detach().cpu()),
                "boundary_stable_joint_dist": float(boundary_rec["boundary_stable_dist"].detach().cpu()),
                "boundary_foot_chamfer": float(boundary_rec["boundary_foot_chamfer"].detach().cpu()),
                "boundary_orient_loss": float(boundary_rec["boundary_orient_loss"].detach().cpu()),
            }
            print(json.dumps(record, sort_keys=True))
            history.append(record)

    with torch.no_grad():
        delta_t, delta_r, _, _ = sequential_forward(model, joints_world, poses, boundary, args.memory_alpha)
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


def sequential_forward_deltas(
    joints_world: torch.Tensor,
    poses: torch.Tensor,
    boundary: int,
    memory_alpha: float,
    delta_t: torch.Tensor,
    delta_r: torch.Tensor,
):
    num_frames = joints_world.shape[0]
    device = joints_world.device
    dtype = joints_world.dtype

    joints_corr, R_corr, t_corr = apply_correction(joints_world, poses, delta_t, delta_r)

    memory_stable = joints_world[boundary - 1, STABLE_JOINTS]
    memory_foot = joints_world[boundary - 1, FOOT_JOINTS]
    memory_frame = torso_frame(joints_world[boundary - 1 : boundary])[0]
    memory_alpha_t = torch.tensor(float(memory_alpha), device=device, dtype=dtype)

    loss_records = []
    prev_delta_t = torch.zeros(3, device=device, dtype=dtype)
    prev_delta_r = torch.zeros(3, device=device, dtype=dtype)
    prev_t_corr = poses[boundary - 1, :3, 3]

    for frame_index in range(boundary, num_frames):
        corrected = joints_corr[frame_index]
        frame_corr = torso_frame(corrected.unsqueeze(0))[0]

        stable_loss = F.smooth_l1_loss(corrected[STABLE_JOINTS], memory_stable, beta=0.05)
        foot_loss = chamfer_per_frame(corrected[FOOT_JOINTS], memory_foot)
        orient_loss = (1.0 - (frame_corr * memory_frame).sum(dim=-1).clamp(-1.0, 1.0)).mean()
        delta_smooth_loss = (delta_t[frame_index] - prev_delta_t).pow(2).mean() + (delta_r[frame_index] - prev_delta_r).pow(2).mean()
        camera_smooth_loss = (t_corr[frame_index] - prev_t_corr).pow(2).mean()

        frame_weight = 8.0 if frame_index == boundary else 1.0
        loss_records.append({
            "frame_weight": corrected.new_tensor(frame_weight),
            "stable": stable_loss,
            "foot": foot_loss,
            "orient": orient_loss,
            "delta_smooth": delta_smooth_loss,
            "camera_smooth": camera_smooth_loss,
            "boundary_stable_dist": (corrected[STABLE_JOINTS] - memory_stable).norm(dim=-1).mean(),
            "boundary_foot_chamfer": foot_loss,
            "boundary_orient_loss": orient_loss,
        })

        prev_delta_t = delta_t[frame_index]
        prev_delta_r = delta_r[frame_index]
        prev_t_corr = t_corr[frame_index]

        # Same causal state update as sequential_forward(): the corrected frame is
        # written into memory for future frames, but future losses do not
        # backpropagate through old cache entries.
        memory_stable = ((1.0 - memory_alpha_t) * memory_stable + memory_alpha_t * corrected[STABLE_JOINTS]).detach()
        memory_foot = ((1.0 - memory_alpha_t) * memory_foot + memory_alpha_t * corrected[FOOT_JOINTS]).detach()
        memory_frame = normalize_vectors((1.0 - memory_alpha_t) * memory_frame + memory_alpha_t * frame_corr).detach()

    return joints_corr, R_corr, t_corr, loss_records


def train_memory_residual_correction(data, args: argparse.Namespace, device: torch.device) -> tuple[np.ndarray, dict]:
    boundary = int(args.boundary)
    num_frames = data.poses.shape[0]
    if boundary <= 0 or boundary >= num_frames:
        raise ValueError(f"boundary must be in [1, {num_frames - 1}], got {boundary}")

    poses = torch.from_numpy(data.poses).to(device=device, dtype=torch.float32)
    joints_world = torch.from_numpy(data.joints_world).to(device=device, dtype=torch.float32)
    post_len = num_frames - boundary

    raw_delta_t_post = torch.nn.Parameter(torch.zeros(post_len, 3, device=device, dtype=torch.float32))
    raw_delta_r_post = torch.nn.Parameter(torch.zeros(post_len, 3, device=device, dtype=torch.float32))
    optimizer = torch.optim.AdamW([raw_delta_t_post, raw_delta_r_post], lr=args.residual_lr, weight_decay=1e-4)

    zero_pre = torch.zeros(boundary, 3, device=device, dtype=torch.float32)
    max_delta_r = math.radians(float(args.max_delta_r_deg))
    history = []

    for step in range(args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        delta_t_post = float(args.max_delta_t) * torch.tanh(raw_delta_t_post)
        delta_r_post = max_delta_r * torch.tanh(raw_delta_r_post)
        delta_t = torch.cat([zero_pre, delta_t_post], dim=0)
        delta_r = torch.cat([zero_pre, delta_r_post], dim=0)

        _, _, _, records = sequential_forward_deltas(joints_world, poses, boundary, args.memory_alpha, delta_t, delta_r)

        stable_losses, foot_losses, orient_losses = [], [], []
        delta_smooth_losses, camera_smooth_losses = [], []
        for rec in records:
            w = args.boundary_weight if float(rec["frame_weight"].detach().cpu()) > 1.0 else args.post_weight
            stable_losses.append(w * rec["stable"])
            foot_losses.append(w * rec["foot"])
            orient_losses.append(w * rec["orient"])
            delta_smooth_losses.append(rec["delta_smooth"])
            camera_smooth_losses.append(rec["camera_smooth"])

        stable_loss = torch.stack(stable_losses).mean()
        foot_loss = torch.stack(foot_losses).mean()
        orient_loss = torch.stack(orient_losses).mean()
        delta_smooth_loss = torch.stack(delta_smooth_losses).mean()
        camera_smooth_loss = torch.stack(camera_smooth_losses).mean()
        prior_loss = delta_t_post.pow(2).mean() + delta_r_post.pow(2).mean()
        pre_noop_loss = delta_t.new_tensor(0.0)

        loss = (
            args.stable_weight * stable_loss
            + args.foot_weight * foot_loss
            + args.orient_weight * orient_loss
            + args.delta_smooth_weight * delta_smooth_loss
            + args.camera_smooth_weight * camera_smooth_loss
            + args.prior_weight * prior_loss
            + args.pre_noop_weight * pre_noop_loss
        )
        loss.backward()
        optimizer.step()

        if step % args.log_every == 0 or step == args.steps:
            boundary_rec = records[0]
            record = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "stable_loss": float(stable_loss.detach().cpu()),
                "foot_loss": float(foot_loss.detach().cpu()),
                "orient_loss": float(orient_loss.detach().cpu()),
                "delta_smooth_loss": float(delta_smooth_loss.detach().cpu()),
                "camera_smooth_loss": float(camera_smooth_loss.detach().cpu()),
                "pre_noop_loss": float(pre_noop_loss.detach().cpu()),
                "delta_t_norm_post": float(delta_t_post.norm(dim=-1).mean().detach().cpu()),
                "delta_r_deg_post": float(torch.rad2deg(delta_r_post.norm(dim=-1)).mean().detach().cpu()),
                "boundary_stable_joint_dist": float(boundary_rec["boundary_stable_dist"].detach().cpu()),
                "boundary_foot_chamfer": float(boundary_rec["boundary_foot_chamfer"].detach().cpu()),
                "boundary_orient_loss": float(boundary_rec["boundary_orient_loss"].detach().cpu()),
            }
            print(json.dumps(record, sort_keys=True), flush=True)
            history.append(record)

    with torch.no_grad():
        delta_t_post = float(args.max_delta_t) * torch.tanh(raw_delta_t_post)
        delta_r_post = max_delta_r * torch.tanh(raw_delta_r_post)
        delta_t = torch.cat([zero_pre, delta_t_post], dim=0)
        delta_r = torch.cat([zero_pre, delta_r_post], dim=0)
        _, R_corr, t_corr, _ = sequential_forward_deltas(joints_world, poses, boundary, args.memory_alpha, delta_t, delta_r)
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
    torch.manual_seed(17)
    np.random.seed(17)
    device = torch.device(args.device)
    num_frames = infer_num_frames(args.input_dir, args.source_video, args.num_frames)
    print(f"Using {num_frames} frames from {args.input_dir}")
    data = load_sequence(args.input_dir, num_frames, device)

    raw_metrics = compute_metrics(data, data.poses, args.boundary)
    print("raw_metrics", json.dumps(raw_metrics, indent=2, sort_keys=True))
    # **========== 原始代码 ==========**
    # corrected_poses, delta_stats = train_memory_correction(data, args, device)
    # **========== 新代码 ==========**
    if args.mode == "mlp":
        corrected_poses, delta_stats = train_memory_correction(data, args, device)
    else:
        corrected_poses, delta_stats = train_memory_residual_correction(data, args, device)
    # **========== 结束 ==========**
    corrected_metrics = compute_metrics(data, corrected_poses, args.boundary)
    print("corrected_metrics", json.dumps(corrected_metrics, indent=2, sort_keys=True))

    copy_or_write_outputs(args.input_dir, args.output_dir, corrected_poses, data.intrinsics, num_frames, args.overwrite)
    np.savez(
        args.output_dir / "memory_human_correction_debug.npz",
        delta_t=delta_stats["delta_t"],
        delta_r=delta_stats["delta_r"],
        raw_metrics=np.array(json.dumps(raw_metrics), dtype=object),
        corrected_metrics=np.array(json.dumps(corrected_metrics), dtype=object),
    )
    with open(args.output_dir / "memory_human_correction_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"raw": raw_metrics, "corrected": corrected_metrics, "history": delta_stats["history"]}, f, indent=2, sort_keys=True)
    print(f"Wrote memory-propagation corrected output to {args.output_dir}")


if __name__ == "__main__":
    main()
