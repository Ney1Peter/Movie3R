#!/usr/bin/env python3
"""Evaluate Human3R/V8.1 camera pose on explicit AvatarReX AABB samples.

This script runs the same recurrent dataloader inference path as training, but
does not save point clouds or meshes. It compares predicted camera poses with
the raw-calibration relative target:

    T_target_i = inv(raw_camera_pose_0) @ raw_camera_pose_i
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import torch

from add_ckpt_path import add_path_to_dust3r
from dust3r.datasets.avatarrex import AvatarReX_AABB
from dust3r.model import ARCroco3DStereo
from dust3r.smpl_model import SMPLModel
from dust3r.utils.camera import pose_encoding_to_camera
from dust3r.utils.device import todevice


def parse_samples(text: str) -> list[tuple[str, str, int]]:
    samples = ast.literal_eval(text)
    out = []
    for sample in samples:
        if len(sample) != 3:
            raise ValueError(f"Bad sample {sample!r}; expected (seq_a, seq_b, start_frame)")
        out.append((str(sample[0]), str(sample[1]), int(sample[2])))
    return out


def parse_manifest(path: Path) -> list[tuple[str, str, int]]:
    records = []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        parsed = json.loads(text)
    else:
        parsed = [json.loads(line) for line in text.splitlines() if line.strip()]
    for record in parsed:
        if isinstance(record, (list, tuple)) and len(record) == 3:
            seq_a, seq_b, start_frame = record
        else:
            seq_a = record.get("seqA", record.get("seq_a"))
            seq_b = record.get("seqB", record.get("seq_b"))
            start_frame = record.get("start_frame", record.get("frame", record.get("t")))
        if seq_a is None or seq_b is None or start_frame is None:
            raise ValueError(f"Bad manifest record in {path}: {record!r}")
        records.append((str(seq_a), str(seq_b), int(start_frame)))
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--samples", default=None, help="Python literal list of (seq_a, seq_b, start_frame)")
    parser.add_argument("--manifest_path", type=Path, default=None, help="JSON/JSONL manifest of AvatarReX AABB samples")
    parser.add_argument("--name", default="eval")
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--avatarrex_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data"))
    parser.add_argument(
        "--avatarrex_raw_root",
        default="/data/wangzheng/iJCV-CODE/data/avatarrex_lbn1",
        help="Raw AvatarReX calibration root, or a Python dict string for grouped roots.",
    )
    parser.add_argument("--split", default="Training")
    parser.add_argument("--resolution", type=int, nargs=2, default=(512, 288), metavar=("W", "H"))
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def parse_raw_root(value: str):
    text = str(value)
    if text.strip().startswith("{"):
        parsed = ast.literal_eval(text)
        if not isinstance(parsed, dict):
            raise ValueError(f"--avatarrex_raw_root dict string expected, got {type(parsed).__name__}")
        return {str(k): str(v) for k, v in parsed.items()}
    return text


def rotation_error_deg(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    rel = torch.matmul(pred[..., :3, :3], target[..., :3, :3].transpose(-1, -2))
    trace = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    angle = torch.acos(((trace - 1.0) * 0.5).clamp(-1.0, 1.0))
    return torch.rad2deg(angle)


def summarize_frames(frame_rows: list[dict]) -> dict:
    trans = np.array([r["trans_err"] for r in frame_rows], dtype=np.float64)
    rot = np.array([r["rot_err_deg"] for r in frame_rows], dtype=np.float64)
    b_trans = np.array([r["trans_err"] for r in frame_rows if r["view_idx"] >= 2], dtype=np.float64)
    b_rot = np.array([r["rot_err_deg"] for r in frame_rows if r["view_idx"] >= 2], dtype=np.float64)
    return {
        "mean_trans_err": float(trans.mean()) if trans.size else None,
        "mean_rot_err_deg": float(rot.mean()) if rot.size else None,
        "b_frames_mean_trans_err": float(b_trans.mean()) if b_trans.size else None,
        "b_frames_mean_rot_err_deg": float(b_rot.mean()) if b_rot.size else None,
        "num_frames": int(len(frame_rows)),
    }


def main() -> None:
    args = parse_args()
    if (args.samples is None) == (args.manifest_path is None):
        raise ValueError("Pass exactly one of --samples or --manifest_path")
    samples = parse_samples(args.samples) if args.samples is not None else parse_manifest(args.manifest_path)
    if not samples:
        raise ValueError("No samples to evaluate")
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    dataset = AvatarReX_AABB(
        split=args.split,
        ROOT=str(args.avatarrex_root),
        resolution=tuple(args.resolution),
        num_views=4,
        aug_crop=0,
        allow_repeat=True,
        seed=args.seed,
        n_corres=0,
        fixed_samples=samples,
        load_da3_depth=False,
        raw_calibration_root=parse_raw_root(args.avatarrex_raw_root),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    add_path_to_dust3r(str(args.model_path))
    model = ARCroco3DStereo.from_pretrained(str(args.model_path)).to(device).float().eval()
    smpl_model = SMPLModel(
        device,
        model_args={
            "patch_size": model.croco_args["patch_size"],
            "mhmr_img_res": model.mhmr_img_res,
            "bb_patch_size": model.bb_patch_size,
        },
    )

    sample_rows = []
    frame_rows = []
    with torch.no_grad():
        for sample_idx, views in enumerate(loader):
            views = todevice(views, device)
            smpl_model.update_smpl_gt(views)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                output, _ = model(views, ret_state=True, inference=True)

            raw_poses = torch.cat([view["raw_camera_pose"].float() for view in views], dim=0)
            target = torch.matmul(torch.linalg.inv(raw_poses[:1]), raw_poses)

            per_sample_frames = []
            for view_idx, pred in enumerate(output.ress):
                pred_pose = pose_encoding_to_camera(pred["camera_pose"].float()).detach()
                gt_pose = target[view_idx:view_idx + 1].to(pred_pose.device, pred_pose.dtype)
                trans_err = torch.linalg.norm(pred_pose[:, :3, 3] - gt_pose[:, :3, 3], dim=-1)
                rot_err = rotation_error_deg(pred_pose, gt_pose)
                row = {
                    "sample_idx": int(sample_idx),
                    "sample": list(samples[sample_idx]),
                    "view_idx": int(view_idx),
                    "trans_err": float(trans_err.item()),
                    "rot_err_deg": float(rot_err.item()),
                }
                frame_rows.append(row)
                per_sample_frames.append(row)

            sample_rows.append({
                "sample_idx": int(sample_idx),
                "sample": list(samples[sample_idx]),
                **summarize_frames(per_sample_frames),
            })

    result = {
        "name": args.name,
        "model_path": str(args.model_path),
        "manifest_path": None if args.manifest_path is None else str(args.manifest_path),
        "samples": [list(s) for s in samples],
        "summary": summarize_frames(frame_rows),
        "sample_metrics": sample_rows,
        "frame_metrics": frame_rows,
        "target": "T_target_i = inv(raw_camera_pose_0) @ raw_camera_pose_i",
        "load_da3_depth": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
