#!/usr/bin/env python3
"""Apply a learned streaming alignment checkpoint to a saved Human3R output."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
ARCHIVE_V7 = SCRIPTS_ROOT / "archive_v7"
for path in (str(REPO_ROOT), str(SRC_ROOT), str(SCRIPTS_ROOT), str(ARCHIVE_V7)):
    if path not in sys.path:
        sys.path.insert(0, path)

from overfit_human_anchor_pose_correction import FOOT_JOINTS, STABLE_JOINTS, load_sequence
from v9_learned_stream_alignment_4source_probe import segment_anchor_metrics
from v9_learned_stream_alignment_overfit import (
    StreamingAlignmentMLP,
    apply_transform_to_post,
    build_feature,
    so3_exp,
    write_aligned_output,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--boundary", type=int, default=2)
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    ckpt_args = checkpoint.get("args", {})
    data = load_sequence(args.input_dir, int(args.num_frames), device)
    pred_poses = torch.from_numpy(data.poses).to(device=device, dtype=torch.float32)
    pred_joints = torch.from_numpy(data.joints_world).to(device=device, dtype=torch.float32)
    joint_ids_np = np.asarray(sorted(set(STABLE_JOINTS + FOOT_JOINTS)), dtype=np.int64)
    joint_ids = torch.from_numpy(joint_ids_np).to(device=device, dtype=torch.long)
    feature = build_feature(pred_joints, int(args.boundary), joint_ids)

    model = StreamingAlignmentMLP(
        in_dim=feature.shape[-1],
        hidden_dim=int(ckpt_args.get("hidden_dim", 256)),
        max_rot_deg=float(ckpt_args.get("max_rot_deg", 180.0)),
        max_trans=float(ckpt_args.get("max_trans", 12.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    with torch.no_grad():
        rotvec, trans = model(feature)
        R = so3_exp(rotvec)[0]
        t = trans[0]
        aligned_poses, aligned_joints = apply_transform_to_post(
            pred_poses,
            pred_joints,
            R,
            t,
            int(args.boundary),
        )

    write_aligned_output(
        args.input_dir,
        args.output_dir,
        aligned_poses.detach().cpu().numpy().astype(np.float32),
        int(args.boundary),
        overwrite=True,
    )
    raw_anchor = segment_anchor_metrics(data.joints_world, int(args.boundary), joint_ids_np)
    aligned_anchor = segment_anchor_metrics(
        aligned_joints.detach().cpu().numpy().astype(np.float32),
        int(args.boundary),
        joint_ids_np,
    )
    summary = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "checkpoint": str(args.checkpoint),
        "boundary": int(args.boundary),
        "num_frames": int(args.num_frames),
        "device": str(device),
        "streaming_semantics": {
            "uses_future_frames_as_input": False,
            "alignment_head_runs_on_frame": int(args.boundary),
            "later_segment_frames_use_cached_transform": True,
            "boundary_is_oracle_for_this_probe": True,
        },
        "learned_transform": {
            "rotvec": rotvec.detach().cpu().numpy().astype(float).reshape(-1).tolist(),
            "rotvec_deg_norm": float(torch.rad2deg(rotvec.norm(dim=-1)).detach().cpu()[0]),
            "translation": trans.detach().cpu().numpy().astype(float).reshape(-1).tolist(),
            "translation_norm": float(trans.norm(dim=-1).detach().cpu()[0]),
        },
        "raw_segment_anchor": raw_anchor,
        "aligned_segment_anchor": aligned_anchor,
    }
    (args.output_dir / "learned_alignment_saved_output_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
