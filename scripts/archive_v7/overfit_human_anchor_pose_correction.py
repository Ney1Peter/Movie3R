#!/usr/bin/env python3
"""Overfit a lightweight human-anchor camera pose correction on saved Human3R output.

This script is intentionally independent from the main training pipeline.  It reads
the directory produced by ``demo.py --save`` and learns a tiny MLP that predicts a
left-multiplied SE(3) residual from human-anchor features:

    T_corr = Delta_T @ T_hat

Only camera poses are changed in the output directory.  Depth, confidence, color,
and SMPL parameters are copied unchanged so the corrected result remains compatible
with the existing Human3R viewer.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dust3r.utils.smpl_layer import SMPL_Layer


FOOT_JOINTS = [7, 8, 10, 11, 60, 61, 62, 63, 64, 65]
TORSO_JOINTS = [0, 1, 2, 3, 6, 9, 15, 16, 17]
STABLE_JOINTS = [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 15, 16, 17, 62, 65]


@dataclass
class SequenceData:
    poses: np.ndarray
    intrinsics: np.ndarray
    rotvec: np.ndarray
    shape: np.ndarray
    transl: np.ndarray
    expression: np.ndarray
    joints_cam: np.ndarray
    joints_world: np.ndarray


class HumanAnchorCorrectionMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, max_delta_t: float = 4.0, max_delta_r_deg: float = 60.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 6),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.max_delta_t = float(max_delta_t)
        self.max_delta_r = math.radians(float(max_delta_r_deg))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.net(x)
        delta_t = self.max_delta_t * torch.tanh(raw[:, :3])
        delta_r = self.max_delta_r * torch.tanh(raw[:, 3:])
        return delta_t, delta_r


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True, help="Human3R demo --save output directory.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Corrected output directory to write.")
    parser.add_argument("--source_video", type=Path, default=None, help="Optional source mp4 used to infer frame count.")
    parser.add_argument("--num_frames", type=int, default=None, help="Number of frames to use. Overrides source_video inference.")
    parser.add_argument("--boundary", type=int, required=True, help="First frame index after the shot boundary, e.g. 63 means 62->63.")
    parser.add_argument("--steps", type=int, default=2500, help="MLP overfit optimization steps.")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate.")
    parser.add_argument("--hidden_dim", type=int, default=256, help="MLP hidden dimension.")
    parser.add_argument("--max_delta_t", type=float, default=4.0, help="Bound on correction translation norm components.")
    parser.add_argument("--max_delta_r_deg", type=float, default=60.0, help="Bound on correction axis-angle components in degrees.")
    parser.add_argument("--foot_weight", type=float, default=8.0)
    parser.add_argument("--stable_weight", type=float, default=1.0)
    parser.add_argument("--orient_weight", type=float, default=4.0)
    parser.add_argument("--static_weight", type=float, default=0.3)
    parser.add_argument("--smooth_weight", type=float, default=0.1)
    parser.add_argument("--prior_weight", type=float, default=0.01)
    parser.add_argument("--pre_noop_weight", type=float, default=2.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output_dir if it already exists.")
    parser.add_argument("--log_every", type=int, default=250)
    return parser.parse_args()


def infer_num_frames(input_dir: Path, source_video: Path | None, explicit_num_frames: int | None) -> int:
    if explicit_num_frames is not None:
        return int(explicit_num_frames)
    if source_video is not None:
        cap = cv2.VideoCapture(str(source_video))
        if not cap.isOpened():
            raise ValueError(f"Could not open source video: {source_video}")
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if n > 0:
            return n
    camera_files = sorted((input_dir / "camera").glob("*.npz"))
    if not camera_files:
        raise FileNotFoundError(f"No camera npz files found under {input_dir / 'camera'}")
    return len(camera_files)


def load_sequence(input_dir: Path, num_frames: int, device: torch.device) -> SequenceData:
    poses, intrinsics = [], []
    rotvec, shape, transl, expression = [], [], [], []
    for i in range(num_frames):
        camera_path = input_dir / "camera" / f"{i:06d}.npz"
        smpl_path = input_dir / "smpl" / f"{i:06d}.npz"
        if not camera_path.is_file():
            raise FileNotFoundError(camera_path)
        if not smpl_path.is_file():
            raise FileNotFoundError(smpl_path)
        cam = np.load(camera_path)
        poses.append(cam["pose"].astype(np.float32))
        intrinsics.append(cam["intrinsics"].astype(np.float32))
        smpl = np.load(smpl_path, allow_pickle=True)
        if smpl["shape"].shape[0] < 1:
            raise ValueError(f"No detected human in {smpl_path}; this quick overfit expects one visible person.")
        rotvec.append(smpl["rotvec"][0].astype(np.float32))
        shape.append(smpl["shape"][0].astype(np.float32))
        transl.append(smpl["transl"][0].astype(np.float32))
        expr_value = smpl["expression"]
        if expr_value is None or expr_value.shape[0] < 1:
            expr_value = np.zeros((1, 10), dtype=np.float32)
        expression.append(expr_value[0].astype(np.float32))

    poses_np = np.stack(poses)
    intrinsics_np = np.stack(intrinsics)
    rotvec_np = np.stack(rotvec)
    shape_np = np.stack(shape)
    transl_np = np.stack(transl)
    expression_np = np.stack(expression)

    smpl_layer = SMPL_Layer(
        type="smplx",
        gender="neutral",
        num_betas=shape_np.shape[-1],
        kid=False,
        person_center="head",
    ).to(device)
    with torch.no_grad():
        out = smpl_layer(
            torch.from_numpy(rotvec_np).to(device=device, dtype=torch.float32),
            torch.from_numpy(shape_np).to(device=device, dtype=torch.float32),
            torch.from_numpy(transl_np).to(device=device, dtype=torch.float32),
            None,
            None,
            K=torch.from_numpy(intrinsics_np).to(device=device, dtype=torch.float32),
            expression=torch.from_numpy(expression_np).to(device=device, dtype=torch.float32),
        )
        joints_cam_np = out["smpl_j3d"].detach().cpu().numpy().astype(np.float32)

    joints_world_np = np.einsum("nij,nkj->nki", poses_np[:, :3, :3], joints_cam_np) + poses_np[:, None, :3, 3]
    return SequenceData(
        poses=poses_np,
        intrinsics=intrinsics_np,
        rotvec=rotvec_np,
        shape=shape_np,
        transl=transl_np,
        expression=expression_np,
        joints_cam=joints_cam_np,
        joints_world=joints_world_np.astype(np.float32),
    )


def normalize_vectors(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def torso_frame(joints: torch.Tensor) -> torch.Tensor:
    left_hip, right_hip = joints[:, 1], joints[:, 2]
    left_shoulder, right_shoulder = joints[:, 16], joints[:, 17]
    hip_mid = 0.5 * (left_hip + right_hip)
    shoulder_mid = 0.5 * (left_shoulder + right_shoulder)
    up = normalize_vectors(shoulder_mid - hip_mid)
    right = normalize_vectors(right_shoulder - left_shoulder)
    forward = normalize_vectors(torch.cross(right, up, dim=-1))
    return torch.stack([right, up, forward], dim=1)


def so3_exp_map(rotvec: torch.Tensor) -> torch.Tensor:
    theta = rotvec.norm(dim=-1, keepdim=True)
    small = theta < 1e-6
    axis = rotvec / theta.clamp_min(1e-6)
    x, y, z = axis.unbind(dim=-1)
    zero = torch.zeros_like(x)
    K = torch.stack(
        [
            zero, -z, y,
            z, zero, -x,
            -y, x, zero,
        ],
        dim=-1,
    ).reshape(-1, 3, 3)
    eye = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype).expand(rotvec.shape[0], 3, 3)
    sin_t = torch.sin(theta).reshape(-1, 1, 1)
    cos_t = torch.cos(theta).reshape(-1, 1, 1)
    R = eye + sin_t * K + (1.0 - cos_t) * (K @ K)
    if small.any():
        # First-order approximation keeps gradients well behaved near zero.
        R_small = eye + skew(rotvec)
        R = torch.where(small.reshape(-1, 1, 1), R_small, R)
    return R


def skew(v: torch.Tensor) -> torch.Tensor:
    x, y, z = v.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        [
            zero, -z, y,
            z, zero, -x,
            -y, x, zero,
        ],
        dim=-1,
    ).reshape(-1, 3, 3)


def chamfer_l1(points: torch.Tensor, ref_points: torch.Tensor) -> torch.Tensor:
    d = torch.cdist(points, ref_points.unsqueeze(0).expand(points.shape[0], -1, -1), p=2)
    return 0.5 * (d.min(dim=2).values.mean() + d.min(dim=1).values.mean())


def rotation_angle_deg(R_rel: np.ndarray) -> np.ndarray:
    tr = np.trace(R_rel, axis1=-2, axis2=-1)
    return np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


def build_features(data: SequenceData, boundary: int, device: torch.device) -> torch.Tensor:
    joints_world = torch.from_numpy(data.joints_world).to(device=device, dtype=torch.float32)
    poses = torch.from_numpy(data.poses).to(device=device, dtype=torch.float32)
    num_frames = joints_world.shape[0]

    foot_ref = joints_world[:boundary, FOOT_JOINTS].median(dim=0).values
    stable_ref = joints_world[:boundary, STABLE_JOINTS].median(dim=0).values
    frame_world = torso_frame(joints_world)
    orient_ref = normalize_vectors(frame_world[:boundary].mean(dim=0))

    foot_res = (joints_world[:, FOOT_JOINTS] - foot_ref.unsqueeze(0)).flatten(1)
    stable_res = (joints_world[:, STABLE_JOINTS] - stable_ref.unsqueeze(0)).flatten(1)
    orient_res = (frame_world - orient_ref.unsqueeze(0)).flatten(1)
    cam_t = poses[:, :3, 3]
    cam_t = cam_t - cam_t[:boundary].mean(dim=0, keepdim=True)

    t = torch.linspace(0.0, 1.0, num_frames, device=device).unsqueeze(1)
    post = (torch.arange(num_frames, device=device) >= boundary).float().unsqueeze(1)
    pe = torch.cat([
        t,
        post,
        torch.sin(2.0 * math.pi * t),
        torch.cos(2.0 * math.pi * t),
        torch.sin(4.0 * math.pi * t),
        torch.cos(4.0 * math.pi * t),
    ], dim=1)
    return torch.cat([foot_res, stable_res, orient_res, cam_t, pe], dim=1)


def apply_correction(joints_world: torch.Tensor, poses: torch.Tensor, delta_t: torch.Tensor, delta_r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    R_delta = so3_exp_map(delta_r)
    joints_corr = torch.einsum("nij,nkj->nki", R_delta, joints_world) + delta_t[:, None, :]

    R_raw = poses[:, :3, :3]
    t_raw = poses[:, :3, 3]
    R_corr = torch.einsum("nij,njk->nik", R_delta, R_raw)
    t_corr = torch.einsum("nij,nj->ni", R_delta, t_raw) + delta_t
    return joints_corr, R_corr, t_corr


def train_correction(data: SequenceData, args: argparse.Namespace, device: torch.device) -> tuple[np.ndarray, dict]:
    boundary = int(args.boundary)
    num_frames = data.poses.shape[0]
    if boundary <= 0 or boundary >= num_frames:
        raise ValueError(f"boundary must be in [1, {num_frames - 1}], got {boundary}")

    poses = torch.from_numpy(data.poses).to(device=device, dtype=torch.float32)
    joints_world = torch.from_numpy(data.joints_world).to(device=device, dtype=torch.float32)
    features = build_features(data, boundary, device)

    model = HumanAnchorCorrectionMLP(
        in_dim=features.shape[1],
        hidden_dim=args.hidden_dim,
        max_delta_t=args.max_delta_t,
        max_delta_r_deg=args.max_delta_r_deg,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    foot_ref = joints_world[:boundary, FOOT_JOINTS].median(dim=0).values
    stable_ref = joints_world[:boundary, STABLE_JOINTS].median(dim=0).values
    orient_ref = normalize_vectors(torso_frame(joints_world)[:boundary].mean(dim=0))

    pre = slice(0, boundary)
    post = slice(boundary, num_frames)

    history = []
    for step in range(args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        delta_t, delta_r = model(features)
        joints_corr, R_corr, t_corr = apply_correction(joints_world, poses, delta_t, delta_r)
        frame_corr = torso_frame(joints_corr)

        foot_loss = chamfer_l1(joints_corr[post][:, FOOT_JOINTS], foot_ref)
        stable_loss = F.smooth_l1_loss(joints_corr[post][:, STABLE_JOINTS], stable_ref.unsqueeze(0).expand(num_frames - boundary, -1, -1), beta=0.05)
        orient_dot = (frame_corr[post] * orient_ref.unsqueeze(0)).sum(dim=-1).clamp(-1.0, 1.0)
        orient_loss = (1.0 - orient_dot).mean()

        t_pre_var = (t_corr[pre] - t_corr[pre].mean(dim=0, keepdim=True)).pow(2).mean()
        t_post_var = (t_corr[post] - t_corr[post].mean(dim=0, keepdim=True)).pow(2).mean()
        static_loss = t_pre_var + t_post_var

        if boundary > 2:
            smooth_pre = (delta_t[1:boundary] - delta_t[: boundary - 1]).pow(2).mean() + (delta_r[1:boundary] - delta_r[: boundary - 1]).pow(2).mean()
        else:
            smooth_pre = delta_t.new_tensor(0.0)
        if num_frames - boundary > 2:
            smooth_post = (delta_t[boundary + 1 :] - delta_t[boundary:-1]).pow(2).mean() + (delta_r[boundary + 1 :] - delta_r[boundary:-1]).pow(2).mean()
        else:
            smooth_post = delta_t.new_tensor(0.0)
        smooth_loss = smooth_pre + smooth_post

        prior_loss = delta_t.pow(2).mean() + delta_r.pow(2).mean()
        pre_noop_loss = delta_t[pre].pow(2).mean() + delta_r[pre].pow(2).mean()

        loss = (
            args.foot_weight * foot_loss
            + args.stable_weight * stable_loss
            + args.orient_weight * orient_loss
            + args.static_weight * static_loss
            + args.smooth_weight * smooth_loss
            + args.prior_weight * prior_loss
            + args.pre_noop_weight * pre_noop_loss
        )
        loss.backward()
        optimizer.step()

        if step % args.log_every == 0 or step == args.steps:
            record = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "foot_loss": float(foot_loss.detach().cpu()),
                "stable_loss": float(stable_loss.detach().cpu()),
                "orient_loss": float(orient_loss.detach().cpu()),
                "static_loss": float(static_loss.detach().cpu()),
                "smooth_loss": float(smooth_loss.detach().cpu()),
                "prior_loss": float(prior_loss.detach().cpu()),
                "pre_noop_loss": float(pre_noop_loss.detach().cpu()),
                "delta_t_norm_post": float(delta_t[post].norm(dim=-1).mean().detach().cpu()),
                "delta_r_deg_post": float(torch.rad2deg(delta_r[post].norm(dim=-1)).mean().detach().cpu()),
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


def compute_metrics(data: SequenceData, poses: np.ndarray, boundary: int) -> dict:
    joints_corr = np.einsum("nij,nkj->nki", poses[:, :3, :3], data.joints_cam) + poses[:, None, :3, 3]
    foot_ref = np.median(data.joints_world[:boundary, FOOT_JOINTS], axis=0)
    foot_err = np.linalg.norm(joints_corr[:, FOOT_JOINTS] - foot_ref[None], axis=-1)

    raw_frame_ref = np.mean(torso_frame(torch.from_numpy(data.joints_world[:boundary]).float()).numpy(), axis=0)
    raw_frame_ref = raw_frame_ref / np.maximum(np.linalg.norm(raw_frame_ref, axis=-1, keepdims=True), 1e-8)
    frame_corr = torso_frame(torch.from_numpy(joints_corr).float()).numpy()
    orient_dot = np.clip((frame_corr * raw_frame_ref[None]).sum(axis=-1), -1.0, 1.0)
    orient_angle = np.degrees(np.arccos(orient_dot))

    cam_t = poses[:, :3, 3]
    cam_R = poses[:, :3, :3]
    step_t = np.linalg.norm(cam_t[1:] - cam_t[:-1], axis=-1)
    step_r = rotation_angle_deg(np.einsum("nij,njk->nik", np.transpose(cam_R[:-1], (0, 2, 1)), cam_R[1:]))

    return {
        "pre_foot_err": float(np.mean(foot_err[:boundary])),
        "post_foot_err": float(np.mean(foot_err[boundary:])),
        "boundary_foot_jump": float(np.mean(np.linalg.norm(joints_corr[boundary, FOOT_JOINTS] - joints_corr[boundary - 1, FOOT_JOINTS], axis=-1))),
        "post_torso_right_deg": float(np.mean(orient_angle[boundary:, 0])),
        "post_torso_up_deg": float(np.mean(orient_angle[boundary:, 1])),
        "post_torso_forward_deg": float(np.mean(orient_angle[boundary:, 2])),
        "boundary_camera_step_t": float(step_t[boundary - 1]),
        "boundary_camera_step_r_deg": float(step_r[boundary - 1]),
        "post_camera_mean_step_t": float(np.mean(step_t[boundary:])),
        "post_camera_max_step_t": float(np.max(step_t[boundary:])),
    }


def copy_or_write_outputs(input_dir: Path, output_dir: Path, corrected_poses: np.ndarray, intrinsics: np.ndarray, num_frames: int, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists; pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    for subdir in ["camera", "camera_raw", "color", "conf", "depth", "smpl"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    for i in range(num_frames):
        raw_cam = np.load(input_dir / "camera" / f"{i:06d}.npz")
        np.savez(output_dir / "camera_raw" / f"{i:06d}.npz", pose=raw_cam["pose"].astype(np.float32), intrinsics=raw_cam["intrinsics"].astype(np.float32))
        np.savez(output_dir / "camera" / f"{i:06d}.npz", pose=corrected_poses[i].astype(np.float32), intrinsics=intrinsics[i].astype(np.float32))
        for subdir, ext in [("color", ".png"), ("conf", ".npy"), ("depth", ".npy"), ("smpl", ".npz")]:
            src = input_dir / subdir / f"{i:06d}{ext}"
            dst = output_dir / subdir / f"{i:06d}{ext}"
            if not src.is_file():
                raise FileNotFoundError(src)
            shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    torch.manual_seed(7)
    np.random.seed(7)
    device = torch.device(args.device)

    num_frames = infer_num_frames(args.input_dir, args.source_video, args.num_frames)
    print(f"Using {num_frames} frames from {args.input_dir}")
    data = load_sequence(args.input_dir, num_frames, device)

    raw_metrics = compute_metrics(data, data.poses, args.boundary)
    print("raw_metrics", json.dumps(raw_metrics, indent=2, sort_keys=True))

    corrected_poses, delta_stats = train_correction(data, args, device)
    corrected_metrics = compute_metrics(data, corrected_poses, args.boundary)
    print("corrected_metrics", json.dumps(corrected_metrics, indent=2, sort_keys=True))

    copy_or_write_outputs(args.input_dir, args.output_dir, corrected_poses, data.intrinsics, num_frames, args.overwrite)
    np.savez(
        args.output_dir / "human_anchor_correction_debug.npz",
        delta_t=delta_stats["delta_t"],
        delta_r=delta_stats["delta_r"],
        raw_metrics=np.array(json.dumps(raw_metrics), dtype=object),
        corrected_metrics=np.array(json.dumps(corrected_metrics), dtype=object),
    )
    with open(args.output_dir / "human_anchor_correction_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"raw": raw_metrics, "corrected": corrected_metrics, "history": delta_stats["history"]}, f, indent=2, sort_keys=True)
    print(f"Wrote corrected Human3R-compatible output to {args.output_dir}")


if __name__ == "__main__":
    main()
