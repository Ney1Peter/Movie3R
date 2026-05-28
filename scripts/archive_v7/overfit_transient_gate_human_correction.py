#!/usr/bin/env python3
"""Overfit learnable transient-gate pose correction on saved Human3R output.

This V7 smoke test targets shot-boundary settling errors: the first post-cut
frame may be badly shifted while later frames in the new shot are already much
more reliable.  Instead of freely correcting every frame, it learns a bounded
persistent residual plus a bounded transient residual modulated by a learned
per-frame gate:

    delta_xi_t = delta_xi_persistent + alpha_t * delta_xi_transient
    T_corr_t = exp(delta_xi_t) @ T_hat_t

Only camera poses are changed in the output directory.  Depth, confidence,
color, and SMPL parameters are copied unchanged so the result remains compatible
with the existing Human3R viewer.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from overfit_human_anchor_pose_correction import (
    FOOT_JOINTS,
    STABLE_JOINTS,
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
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--max_transient_t", type=float, default=6.0)
    parser.add_argument("--max_transient_r_deg", type=float, default=80.0)
    parser.add_argument("--max_persistent_t", type=float, default=1.0)
    parser.add_argument("--max_persistent_r_deg", type=float, default=20.0)
    parser.add_argument("--anomaly_z", type=float, default=6.0)
    parser.add_argument("--anomaly_sharpness", type=float, default=1.0)
    parser.add_argument("--stable_weight", type=float, default=1.0)
    parser.add_argument("--foot_weight", type=float, default=2.0)
    parser.add_argument("--orient_weight", type=float, default=5.0)
    parser.add_argument("--relative_preserve_weight", type=float, default=2.0)
    parser.add_argument("--camera_preserve_weight", type=float, default=1.0)
    parser.add_argument("--gate_sparsity_weight", type=float, default=0.08)
    parser.add_argument("--gate_reliable_sparsity_weight", type=float, default=5.0)
    parser.add_argument("--gate_decay_weight", type=float, default=0.04)
    parser.add_argument("--gate_smooth_weight", type=float, default=0.1)
    parser.add_argument("--persistent_prior_weight", type=float, default=0.5)
    parser.add_argument("--prior_weight", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log_every", type=int, default=250)
    return parser.parse_args()


class TransientGateCorrection(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        max_transient_t: float,
        max_transient_r_deg: float,
        max_persistent_t: float,
        max_persistent_r_deg: float,
    ):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.zeros_(self.gate_net[-1].bias)
        self.raw_transient = nn.Parameter(torch.zeros(6))
        self.raw_persistent = nn.Parameter(torch.zeros(6))
        self.max_transient_t = float(max_transient_t)
        self.max_transient_r = math.radians(float(max_transient_r_deg))
        self.max_persistent_t = float(max_persistent_t)
        self.max_persistent_r = math.radians(float(max_persistent_r_deg))

    def forward(self, features: torch.Tensor, post_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        gate = torch.sigmoid(self.gate_net(features)).squeeze(-1) * post_mask
        transient_t = self.max_transient_t * torch.tanh(self.raw_transient[:3])
        transient_r = self.max_transient_r * torch.tanh(self.raw_transient[3:])
        persistent_t = self.max_persistent_t * torch.tanh(self.raw_persistent[:3])
        persistent_r = self.max_persistent_r * torch.tanh(self.raw_persistent[3:])
        delta_t = post_mask[:, None] * (persistent_t[None] + gate[:, None] * transient_t[None])
        delta_r = post_mask[:, None] * (persistent_r[None] + gate[:, None] * transient_r[None])
        parts = {
            "gate": gate,
            "transient_t": transient_t,
            "transient_r": transient_r,
            "persistent_t": persistent_t,
            "persistent_r": persistent_r,
        }
        return delta_t, delta_r, gate, parts


def batched_chamfer(points: torch.Tensor, ref_points: torch.Tensor) -> torch.Tensor:
    d = torch.cdist(points, ref_points, p=2)
    return 0.5 * (d.min(dim=2).values.mean(dim=1) + d.min(dim=1).values.mean(dim=1))


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    denom = weights.sum().clamp_min(1.0)
    return (values * weights).sum() / denom


def robust_z(values: torch.Tensor) -> torch.Tensor:
    median = values.median()
    mad = (values - median).abs().median()
    return (values - median) / (1.4826 * mad + 1e-4)


def build_gate_features(
    joints_world: torch.Tensor,
    poses: torch.Tensor,
    boundary: int,
    anomaly_score: torch.Tensor,
    anomaly_weight: torch.Tensor,
) -> torch.Tensor:
    num_frames = joints_world.shape[0]
    device = joints_world.device
    dtype = joints_world.dtype

    stable = joints_world[:, STABLE_JOINTS]
    foot = joints_world[:, FOOT_JOINTS]
    frame = torso_frame(joints_world)
    cam_t = poses[:, :3, 3]

    stable_prev = torch.cat([stable[:1], stable[:-1]], dim=0)
    foot_prev = torch.cat([foot[:1], foot[:-1]], dim=0)
    frame_prev = torch.cat([frame[:1], frame[:-1]], dim=0)
    cam_t_prev = torch.cat([cam_t[:1], cam_t[:-1]], dim=0)

    stable_pair = (stable - stable_prev).flatten(1)
    foot_pair = (foot - foot_prev).flatten(1)
    frame_pair = (frame - frame_prev).flatten(1)
    cam_step = cam_t - cam_t_prev

    score_frame = torch.zeros(num_frames, device=device, dtype=dtype)
    weight_frame = torch.zeros(num_frames, device=device, dtype=dtype)
    score_frame[1:] = anomaly_score
    weight_frame[1:] = anomaly_weight

    idx = torch.arange(num_frames, device=device, dtype=dtype)
    post_age = (idx - float(boundary)).clamp_min(0.0)
    post_mask = (idx >= boundary).to(dtype)
    boundary_flag = (idx == boundary).to(dtype)
    t = idx / max(num_frames - 1, 1)
    pe = torch.stack(
        [
            t,
            post_mask,
            boundary_flag,
            post_age / max(num_frames - boundary, 1),
            torch.exp(-post_age / 1.5),
            torch.exp(-post_age / 4.0),
            torch.sin(2.0 * math.pi * t),
            torch.cos(2.0 * math.pi * t),
            score_frame,
            weight_frame,
            cam_step.norm(dim=-1),
        ],
        dim=1,
    )
    return torch.cat([stable_pair, foot_pair, frame_pair, cam_step, pe], dim=1)


def compute_anomaly_weights(
    joints_world: torch.Tensor,
    poses: torch.Tensor,
    boundary: int,
    anomaly_z: float,
    anomaly_sharpness: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    stable = joints_world[:, STABLE_JOINTS]
    foot = joints_world[:, FOOT_JOINTS]
    frame = torso_frame(joints_world)
    cam_t = poses[:, :3, 3]

    stable_jump = (stable[1:] - stable[:-1]).norm(dim=-1).mean(dim=-1)
    foot_jump = batched_chamfer(foot[1:], foot[:-1])
    orient_jump = (1.0 - (frame[1:] * frame[:-1]).sum(dim=-1).clamp(-1.0, 1.0)).mean(dim=-1)
    camera_jump = (cam_t[1:] - cam_t[:-1]).norm(dim=-1)

    score = torch.maximum(torch.maximum(robust_z(stable_jump), robust_z(foot_jump)), torch.maximum(robust_z(orient_jump), robust_z(camera_jump)))
    weight = torch.sigmoid((score - float(anomaly_z)) * float(anomaly_sharpness))
    dst = torch.arange(1, joints_world.shape[0], device=joints_world.device)
    weight = torch.where(dst >= boundary, weight, torch.zeros_like(weight))
    components = {
        "stable_jump": stable_jump,
        "foot_jump": foot_jump,
        "orient_jump": orient_jump,
        "camera_jump": camera_jump,
        "score": score,
        "weight": weight,
    }
    return score.detach(), weight.detach(), components


def rotation_relative_loss(R_corr: torch.Tensor, R_raw: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    rel_corr = torch.einsum("nij,njk->nik", R_corr[:-1].transpose(1, 2), R_corr[1:])
    rel_raw = torch.einsum("nij,njk->nik", R_raw[:-1].transpose(1, 2), R_raw[1:])
    return weighted_mean((rel_corr - rel_raw).pow(2), weights)


def transform_joints_np(data, poses: np.ndarray) -> np.ndarray:
    return np.einsum("nij,nkj->nki", poses[:, :3, :3], data.joints_cam) + poses[:, None, :3, 3]


def compute_transition_metrics(data, poses: np.ndarray, boundary: int) -> dict:
    joints = transform_joints_np(data, poses)
    frame = torso_frame(torch.from_numpy(joints).float()).numpy()
    cam_t = poses[:, :3, 3]
    cam_R = poses[:, :3, :3]

    foot_jump = np.linalg.norm(joints[1:, FOOT_JOINTS] - joints[:-1, FOOT_JOINTS], axis=-1).mean(axis=-1)
    stable_jump = np.linalg.norm(joints[1:, STABLE_JOINTS] - joints[:-1, STABLE_JOINTS], axis=-1).mean(axis=-1)
    orient_step = np.degrees(np.arccos(np.clip((frame[1:] * frame[:-1]).sum(axis=-1), -1.0, 1.0))).mean(axis=-1)
    cam_step_t = np.linalg.norm(cam_t[1:] - cam_t[:-1], axis=-1)
    cam_step_r = rotation_angle_deg(np.einsum("nij,njk->nik", np.transpose(cam_R[:-1], (0, 2, 1)), cam_R[1:]))

    settle_idx = min(boundary, len(foot_jump) - 1)
    post_slice = slice(boundary - 1, None)
    return {
        "boundary_foot_jump": float(foot_jump[boundary - 1]),
        "settle_foot_jump": float(foot_jump[settle_idx]),
        "post_pair_foot_jump_max": float(np.max(foot_jump[post_slice])),
        "post_pair_foot_jump_mean": float(np.mean(foot_jump[post_slice])),
        "boundary_stable_jump": float(stable_jump[boundary - 1]),
        "settle_stable_jump": float(stable_jump[settle_idx]),
        "boundary_orient_step_deg": float(orient_step[boundary - 1]),
        "settle_orient_step_deg": float(orient_step[settle_idx]),
        "boundary_camera_step_t": float(cam_step_t[boundary - 1]),
        "settle_camera_step_t": float(cam_step_t[settle_idx]),
        "boundary_camera_step_r_deg": float(cam_step_r[boundary - 1]),
        "settle_camera_step_r_deg": float(cam_step_r[settle_idx]),
    }


def train_transient_gate_correction(data, args: argparse.Namespace, device: torch.device) -> tuple[np.ndarray, dict]:
    boundary = int(args.boundary)
    num_frames = data.poses.shape[0]
    if boundary <= 0 or boundary >= num_frames:
        raise ValueError(f"boundary must be in [1, {num_frames - 1}], got {boundary}")

    poses = torch.from_numpy(data.poses).to(device=device, dtype=torch.float32)
    joints_world = torch.from_numpy(data.joints_world).to(device=device, dtype=torch.float32)
    anomaly_score, anomaly_weight, anomaly_components = compute_anomaly_weights(
        joints_world,
        poses,
        boundary,
        args.anomaly_z,
        args.anomaly_sharpness,
    )
    features = build_gate_features(joints_world, poses, boundary, anomaly_score, anomaly_weight)
    frame_idx = torch.arange(num_frames, device=device)
    post_mask = (frame_idx >= boundary).float()
    pair_dst = torch.arange(1, num_frames, device=device)
    post_pair_mask = (pair_dst >= boundary).float()
    continuity_weight = anomaly_weight * post_pair_mask
    reliable_weight = (1.0 - anomaly_weight) * post_pair_mask

    model = TransientGateCorrection(
        in_dim=features.shape[1],
        hidden_dim=args.hidden_dim,
        max_transient_t=args.max_transient_t,
        max_transient_r_deg=args.max_transient_r_deg,
        max_persistent_t=args.max_persistent_t,
        max_persistent_r_deg=args.max_persistent_r_deg,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    raw_stable_diff = joints_world[1:, STABLE_JOINTS] - joints_world[:-1, STABLE_JOINTS]
    raw_frame = torso_frame(joints_world)
    raw_frame_diff = raw_frame[1:] - raw_frame[:-1]
    raw_t_step = poses[1:, :3, 3] - poses[:-1, :3, 3]

    history = []
    for step in range(args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        delta_t, delta_r, gate, parts = model(features, post_mask)
        joints_corr, R_corr, t_corr = apply_correction(joints_world, poses, delta_t, delta_r)
        frame_corr = torso_frame(joints_corr)

        stable_diff = joints_corr[1:, STABLE_JOINTS] - joints_corr[:-1, STABLE_JOINTS]
        stable_continuity = weighted_mean(F.smooth_l1_loss(stable_diff, torch.zeros_like(stable_diff), beta=0.05, reduction="none"), continuity_weight)
        foot_continuity = weighted_mean(batched_chamfer(joints_corr[1:, FOOT_JOINTS], joints_corr[:-1, FOOT_JOINTS]), continuity_weight)
        orient_dot = (frame_corr[1:] * frame_corr[:-1]).sum(dim=-1).clamp(-1.0, 1.0)
        orient_continuity = weighted_mean(1.0 - orient_dot, continuity_weight)

        stable_preserve = weighted_mean(F.smooth_l1_loss(stable_diff - raw_stable_diff, torch.zeros_like(raw_stable_diff), beta=0.05, reduction="none"), reliable_weight)
        frame_preserve = weighted_mean((frame_corr[1:] - frame_corr[:-1] - raw_frame_diff).pow(2), reliable_weight)
        camera_t_preserve = weighted_mean((t_corr[1:] - t_corr[:-1] - raw_t_step).pow(2), reliable_weight)
        camera_r_preserve = rotation_relative_loss(R_corr, poses[:, :3, :3], reliable_weight)
        relative_preserve = stable_preserve + frame_preserve
        camera_preserve = camera_t_preserve + camera_r_preserve

        gate_post = gate[boundary:]
        post_age = torch.arange(num_frames - boundary, device=device, dtype=torch.float32)
        # **========== 原始代码 ==========**
        # gate_sparsity = gate_post.mean()
        # **========== 新代码 ==========**
        gate_sparsity = gate_post.mean()
        reliable_gate_sparsity = weighted_mean(gate[1:].pow(2), reliable_weight)
        # **========== 结束 ==========**
        gate_decay = (gate_post * (post_age / max(num_frames - boundary - 1, 1))).mean()
        gate_smooth = (gate[boundary + 1 :] - gate[boundary:-1]).pow(2).mean() if num_frames - boundary > 1 else gate.new_tensor(0.0)

        persistent_prior = parts["persistent_t"].pow(2).mean() + parts["persistent_r"].pow(2).mean()
        prior = delta_t.pow(2).mean() + delta_r.pow(2).mean() + 0.1 * (parts["transient_t"].pow(2).mean() + parts["transient_r"].pow(2).mean())

        loss = (
            args.stable_weight * stable_continuity
            + args.foot_weight * foot_continuity
            + args.orient_weight * orient_continuity
            + args.relative_preserve_weight * relative_preserve
            + args.camera_preserve_weight * camera_preserve
            + args.gate_sparsity_weight * gate_sparsity
            + args.gate_reliable_sparsity_weight * reliable_gate_sparsity
            + args.gate_decay_weight * gate_decay
            + args.gate_smooth_weight * gate_smooth
            + args.persistent_prior_weight * persistent_prior
            + args.prior_weight * prior
        )
        loss.backward()
        optimizer.step()

        if step % args.log_every == 0 or step == args.steps:
            gate_slice = gate[boundary : min(boundary + 12, num_frames)].detach().cpu().tolist()
            anomaly_slice = anomaly_weight[boundary - 1 : min(boundary + 11, num_frames - 1)].detach().cpu().tolist()
            record = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "stable_continuity": float(stable_continuity.detach().cpu()),
                "foot_continuity": float(foot_continuity.detach().cpu()),
                "orient_continuity": float(orient_continuity.detach().cpu()),
                "relative_preserve": float(relative_preserve.detach().cpu()),
                "camera_preserve": float(camera_preserve.detach().cpu()),
                "gate_sparsity": float(gate_sparsity.detach().cpu()),
                "reliable_gate_sparsity": float(reliable_gate_sparsity.detach().cpu()),
                "gate_decay": float(gate_decay.detach().cpu()),
                "gate_smooth": float(gate_smooth.detach().cpu()),
                "persistent_t_norm": float(parts["persistent_t"].norm().detach().cpu()),
                "persistent_r_deg": float(torch.rad2deg(parts["persistent_r"].norm()).detach().cpu()),
                "transient_t_norm": float(parts["transient_t"].norm().detach().cpu()),
                "transient_r_deg": float(torch.rad2deg(parts["transient_r"].norm()).detach().cpu()),
                "gate_first_post": [round(float(x), 4) for x in gate_slice],
                "anomaly_first_post_pairs": [round(float(x), 4) for x in anomaly_slice],
            }
            print(json.dumps(record, sort_keys=True), flush=True)
            history.append(record)

    with torch.no_grad():
        delta_t, delta_r, gate, parts = model(features, post_mask)
        _, R_corr, t_corr = apply_correction(joints_world, poses, delta_t, delta_r)
        corrected = poses.detach().cpu().numpy().copy()
        corrected[:, :3, :3] = R_corr.detach().cpu().numpy().astype(np.float32)
        corrected[:, :3, 3] = t_corr.detach().cpu().numpy().astype(np.float32)
        debug = {
            "delta_t": delta_t.detach().cpu().numpy().astype(np.float32),
            "delta_r": delta_r.detach().cpu().numpy().astype(np.float32),
            "gate": gate.detach().cpu().numpy().astype(np.float32),
            "anomaly_weight": anomaly_weight.detach().cpu().numpy().astype(np.float32),
            "anomaly_score": anomaly_score.detach().cpu().numpy().astype(np.float32),
            "persistent_t": parts["persistent_t"].detach().cpu().numpy().astype(np.float32),
            "persistent_r": parts["persistent_r"].detach().cpu().numpy().astype(np.float32),
            "transient_t": parts["transient_t"].detach().cpu().numpy().astype(np.float32),
            "transient_r": parts["transient_r"].detach().cpu().numpy().astype(np.float32),
            "history": history,
            "anomaly_components": {k: v.detach().cpu().numpy().astype(np.float32) for k, v in anomaly_components.items()},
        }
    return corrected, debug


def main() -> None:
    args = parse_args()
    torch.manual_seed(23)
    np.random.seed(23)
    device = torch.device(args.device)
    num_frames = infer_num_frames(args.input_dir, args.source_video, args.num_frames)
    print(f"Using {num_frames} frames from {args.input_dir}")
    data = load_sequence(args.input_dir, num_frames, device)

    raw_metrics = compute_metrics(data, data.poses, args.boundary)
    raw_pair_metrics = compute_transition_metrics(data, data.poses, args.boundary)
    print("raw_metrics", json.dumps(raw_metrics, indent=2, sort_keys=True))
    print("raw_transition_metrics", json.dumps(raw_pair_metrics, indent=2, sort_keys=True))

    corrected_poses, debug = train_transient_gate_correction(data, args, device)
    corrected_metrics = compute_metrics(data, corrected_poses, args.boundary)
    corrected_pair_metrics = compute_transition_metrics(data, corrected_poses, args.boundary)
    print("corrected_metrics", json.dumps(corrected_metrics, indent=2, sort_keys=True))
    print("corrected_transition_metrics", json.dumps(corrected_pair_metrics, indent=2, sort_keys=True))

    copy_or_write_outputs(args.input_dir, args.output_dir, corrected_poses, data.intrinsics, num_frames, args.overwrite)
    np.savez(
        args.output_dir / "transient_gate_human_correction_debug.npz",
        delta_t=debug["delta_t"],
        delta_r=debug["delta_r"],
        gate=debug["gate"],
        anomaly_weight=debug["anomaly_weight"],
        anomaly_score=debug["anomaly_score"],
        persistent_t=debug["persistent_t"],
        persistent_r=debug["persistent_r"],
        transient_t=debug["transient_t"],
        transient_r=debug["transient_r"],
        raw_metrics=np.array(json.dumps(raw_metrics), dtype=object),
        corrected_metrics=np.array(json.dumps(corrected_metrics), dtype=object),
        raw_transition_metrics=np.array(json.dumps(raw_pair_metrics), dtype=object),
        corrected_transition_metrics=np.array(json.dumps(corrected_pair_metrics), dtype=object),
    )
    with open(args.output_dir / "transient_gate_human_correction_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "raw": raw_metrics,
                "corrected": corrected_metrics,
                "raw_transition": raw_pair_metrics,
                "corrected_transition": corrected_pair_metrics,
                "history": debug["history"],
                "gate_first_post": debug["gate"][args.boundary : min(args.boundary + 20, num_frames)].tolist(),
                "anomaly_first_post_pairs": debug["anomaly_weight"][args.boundary - 1 : min(args.boundary + 19, num_frames - 1)].tolist(),
            },
            f,
            indent=2,
            sort_keys=True,
        )
    print(f"Wrote transient-gate corrected output to {args.output_dir}")


if __name__ == "__main__":
    main()
