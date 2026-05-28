#!/usr/bin/env python3
"""Overfit a human-anchor correction for one shot-boundary frame only.

This smoke test is intentionally stricter than transient-gate correction.  It
uses only the boundary pair (frame boundary-1 -> boundary) as the human anchor
constraint, corrects only the first post-boundary frame, and leaves all other
frames exactly unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
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
    torso_frame,
)
from overfit_memory_human_correction import chamfer_per_frame
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
    parser.add_argument("--prior_weight", type=float, default=0.01)
    parser.add_argument("--subset_output_dir", type=Path, default=None, help="Optional reindexed subset output for visualization.")
    parser.add_argument("--raw_subset_output_dir", type=Path, default=None, help="Optional raw reindexed subset for raw camera overlay.")
    parser.add_argument("--subset_start", type=int, default=None)
    parser.add_argument("--subset_count", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log_every", type=int, default=200)
    return parser.parse_args()


def link_or_symlink(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        rel_src = os.path.relpath(src, start=dst.parent)
        os.symlink(rel_src, dst)


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists; pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    for subdir in ["camera", "camera_raw", "color", "conf", "depth", "smpl"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)


def write_outputs_with_links(input_dir: Path, output_dir: Path, poses: np.ndarray, intrinsics: np.ndarray, frame_ids: list[int], overwrite: bool) -> None:
    prepare_output_dir(output_dir, overwrite)
    for out_i, src_i in enumerate(frame_ids):
        raw_cam = np.load(input_dir / "camera" / f"{src_i:06d}.npz")
        np.savez(output_dir / "camera_raw" / f"{out_i:06d}.npz", pose=raw_cam["pose"].astype(np.float32), intrinsics=raw_cam["intrinsics"].astype(np.float32))
        np.savez(output_dir / "camera" / f"{out_i:06d}.npz", pose=poses[src_i].astype(np.float32), intrinsics=intrinsics[src_i].astype(np.float32))
        for subdir, ext in [("color", ".png"), ("conf", ".npy"), ("depth", ".npy"), ("smpl", ".npz")]:
            src = input_dir / subdir / f"{src_i:06d}{ext}"
            dst = output_dir / subdir / f"{out_i:06d}{ext}"
            if not src.is_file():
                raise FileNotFoundError(src)
            link_or_symlink(src, dst)


def train_single_frame(data, args: argparse.Namespace, device: torch.device) -> tuple[np.ndarray, dict]:
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
    history = []
    for step in range(args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        delta_t_one = float(args.max_delta_t) * torch.tanh(raw_delta_t)
        delta_r_one = max_delta_r * torch.tanh(raw_delta_r)
        corrected_one, _, _ = apply_correction(joints_world[boundary : boundary + 1], poses[boundary : boundary + 1], delta_t_one, delta_r_one)
        corrected_one = corrected_one[0]
        frame_corr = torso_frame(corrected_one.unsqueeze(0))[0]

        stable_loss = F.smooth_l1_loss(corrected_one[STABLE_JOINTS], ref_joints[STABLE_JOINTS], beta=0.05)
        foot_loss = chamfer_per_frame(corrected_one[FOOT_JOINTS], ref_joints[FOOT_JOINTS])
        orient_loss = (1.0 - (frame_corr * ref_frame).sum(dim=-1).clamp(-1.0, 1.0)).mean()
        prior_loss = delta_t_one.pow(2).mean() + delta_r_one.pow(2).mean()
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


def main() -> None:
    args = parse_args()
    torch.manual_seed(31)
    np.random.seed(31)
    device = torch.device(args.device)
    num_frames = infer_num_frames(args.input_dir, args.source_video, args.num_frames)
    print(f"Using {num_frames} frames from {args.input_dir}")
    data = load_sequence(args.input_dir, num_frames, device)

    raw_metrics = compute_metrics(data, data.poses, args.boundary)
    raw_transition_metrics = compute_transition_metrics(data, data.poses, args.boundary)
    print("raw_metrics", json.dumps(raw_metrics, indent=2, sort_keys=True))
    print("raw_transition_metrics", json.dumps(raw_transition_metrics, indent=2, sort_keys=True))

    corrected_poses, debug = train_single_frame(data, args, device)
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
        args.output_dir / "single_boundary_frame_human_anchor_debug.npz",
        delta_t=debug["delta_t"],
        delta_r=debug["delta_r"],
        raw_metrics=np.array(json.dumps(raw_metrics), dtype=object),
        corrected_metrics=np.array(json.dumps(corrected_metrics), dtype=object),
        raw_transition_metrics=np.array(json.dumps(raw_transition_metrics), dtype=object),
        corrected_transition_metrics=np.array(json.dumps(corrected_transition_metrics), dtype=object),
    )
    with open(args.output_dir / "single_boundary_frame_human_anchor_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "raw": raw_metrics,
                "corrected": corrected_metrics,
                "raw_transition": raw_transition_metrics,
                "corrected_transition": corrected_transition_metrics,
                "history": debug["history"],
                "corrected_frame": int(args.boundary),
                "reference_frame": int(args.boundary - 1),
            },
            f,
            indent=2,
            sort_keys=True,
        )
    print(f"Wrote single-frame corrected output to {args.output_dir}")


if __name__ == "__main__":
    main()
