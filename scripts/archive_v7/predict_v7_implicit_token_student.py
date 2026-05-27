#!/usr/bin/env python3
"""Run a trained V7 implicit pose adapter on dumped token features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dust3r.v7_pose_adapter import HumanSceneTokenPoseAdapter, apply_left_se3_delta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens_npz", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_npz", type=Path, required=True)
    parser.add_argument("--frame_ids", type=int, nargs="*", default=None)
    parser.add_argument("--frame_start", type=int, default=None)
    parser.add_argument("--frame_count", type=int, default=None)
    parser.add_argument("--all_frames", action="store_true")
    parser.add_argument(
        "--zero_raw_camera_pose_input",
        action="store_true",
        help="Force token-only adapter input by feeding zeros as the raw-pose prior. Defaults to the checkpoint setting if omitted.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def select_indices(frame_ids: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.all_frames:
        return np.arange(len(frame_ids), dtype=np.int64)
    if args.frame_ids is not None and len(args.frame_ids) > 0:
        wanted = {int(x) for x in args.frame_ids}
        indices = [i for i, frame in enumerate(frame_ids.tolist()) if int(frame) in wanted]
        missing = sorted(wanted - {int(frame_ids[i]) for i in indices})
        if missing:
            raise ValueError(f"Requested frame ids not present in token dump: {missing}")
        return np.asarray(indices, dtype=np.int64)
    if args.frame_start is not None and args.frame_count is not None:
        start = int(args.frame_start)
        end = start + int(args.frame_count)
        indices = np.where((frame_ids >= start) & (frame_ids < end))[0]
        if len(indices) == 0:
            raise ValueError(f"No frames selected by [{start}, {end})")
        return indices.astype(np.int64)
    raise ValueError("Select frames with --frame_ids, --frame_start/--frame_count, or --all_frames")


def to_tensor(array: np.ndarray, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    return torch.from_numpy(array).to(device=device, dtype=dtype)


def load_adapter(checkpoint_path: Path, device: torch.device) -> tuple[HumanSceneTokenPoseAdapter, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    adapter = HumanSceneTokenPoseAdapter(
        dec_dim=int(checkpoint["dec_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        input_mode=checkpoint.get("input_mode", "human"),
        max_delta_t=float(checkpoint["max_delta_t"]),
        max_delta_r=float(checkpoint["max_delta_r"]),
    ).to(device)
    adapter.load_state_dict(checkpoint["adapter"])
    adapter.eval()
    return adapter, checkpoint


def main() -> None:
    args = parse_args()
    if args.output_npz.exists() and not args.overwrite:
        raise FileExistsError(args.output_npz)
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    data = np.load(args.tokens_npz)
    frame_ids = data["frame_ids"].astype(np.int64)
    indices_np = select_indices(frame_ids, args)
    indices = torch.from_numpy(indices_np).to(device=device, dtype=torch.long)
    adapter, checkpoint = load_adapter(args.checkpoint, device)
    zero_raw_camera_pose_input = bool(args.zero_raw_camera_pose_input or checkpoint.get("zero_raw_camera_pose_input", False))
    tensors = {
        "pose_tokens": to_tensor(data["pose_tokens"], device),
        "scene_tokens": to_tensor(data["scene_tokens"], device),
        "human_tokens": to_tensor(data["human_tokens"], device),
        "memory_tokens": to_tensor(data["memory_tokens"], device),
        "raw_camera_pose": to_tensor(data["raw_camera_pose"], device),
        "human_token_mask": torch.from_numpy(data["human_token_mask"].astype(np.bool_)).to(device=device),
    }
    raw_camera_pose = tensors["raw_camera_pose"][indices]
    camera_pose_input = torch.zeros_like(raw_camera_pose) if zero_raw_camera_pose_input else raw_camera_pose
    with torch.no_grad():
        corrected_pose, info = adapter(
            pose_token=tensors["pose_tokens"][indices],
            scene_tokens=tensors["scene_tokens"][indices],
            human_tokens=tensors["human_tokens"][indices],
            memory_tokens=tensors["memory_tokens"][indices],
            camera_pose=camera_pose_input,
            human_token_mask=tensors["human_token_mask"][indices],
        )
        if zero_raw_camera_pose_input:
            corrected_pose = apply_left_se3_delta(
                raw_camera_pose,
                info["v7_pose_delta_t"],
                info["v7_pose_delta_rotvec"],
                info["v7_pose_alpha"],
            )
    out_frame_ids = frame_ids[indices_np].astype(np.int32)
    pred_delta_t = info["v7_pose_delta_t"].detach().cpu().numpy().astype(np.float32)
    pred_delta_rotvec = info["v7_pose_delta_rotvec"].detach().cpu().numpy().astype(np.float32)
    pred_alpha = info["v7_pose_alpha"].detach().cpu().numpy().astype(np.float32)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        frame_ids=out_frame_ids,
        corrected_camera_pose=corrected_pose.detach().cpu().numpy().astype(np.float32),
        pred_delta_t=pred_delta_t,
        pred_delta_rotvec=pred_delta_rotvec,
        pred_alpha=pred_alpha,
        pred_r_human=info["v7_pose_r_human"].detach().cpu().numpy().astype(np.float32),
        pred_r_scene=info["v7_pose_r_scene"].detach().cpu().numpy().astype(np.float32),
        target_mask=np.ones((len(out_frame_ids),), dtype=np.bool_),
    )
    summary = {
        "tokens_npz": str(args.tokens_npz),
        "checkpoint": str(args.checkpoint),
        "output_npz": str(args.output_npz),
        "input_mode": checkpoint.get("input_mode", "human"),
        "zero_raw_camera_pose_input": zero_raw_camera_pose_input,
        "selected_frame_ids": out_frame_ids.astype(int).tolist(),
        "pred_delta_t_norm": np.linalg.norm(pred_delta_t, axis=1).astype(float).tolist(),
        "pred_delta_rot_deg": (np.linalg.norm(pred_delta_rotvec, axis=1) * 180.0 / np.pi).astype(float).tolist(),
        "pred_alpha": pred_alpha.reshape(-1).astype(float).tolist(),
    }
    args.output_npz.with_suffix(".json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
