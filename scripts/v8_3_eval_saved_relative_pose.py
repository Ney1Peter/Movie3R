#!/usr/bin/env python3
"""Evaluate saved 4-frame AvatarReX image-only outputs against GT camera poses.

The model is not run here. This script only reads saved ``camera/*.npz`` files
and the RGB source paths recorded in ``input_manifest.json``.

For V8 pose-prompt experiments, the correct GT target is the dataloader's
``raw_camera_pose`` from AvatarReX ``calibration_full.json``:

    T_target_i = inv(raw_camera_pose_0) @ raw_camera_pose_i

Do not use ``training/<group>/<seq>/cam/*.npz`` as the pose target here.  Those
processed person-relative poses are useful for SMPL/depth preprocessing, but
they are not the camera target used by the V8 losses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_manifest", type=Path, required=True)
    parser.add_argument("--pred_dir", type=Path, required=True)
    parser.add_argument("--raw_dir", type=Path, default=None)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data"))
    parser.add_argument(
        "--raw_calibration_root",
        type=Path,
        default=None,
        help="Raw AvatarReX root with calibration_full.json. Defaults to data/avatarrex_<group>.",
    )
    return parser.parse_args()


def load_pose(path: Path) -> np.ndarray:
    data = np.load(path)
    return data["pose"].astype(np.float64)


def load_raw_calibration(root: Path) -> dict:
    path = root / "calibration_full.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def raw_calibration_c2w(calibration: dict, seq: str) -> np.ndarray:
    """Match AvatarReX_AABB raw_camera_pose target: X_cam = R @ X_world + T."""
    seq_key = seq.split("/", 1)[1] if "/" in seq else seq
    if seq_key not in calibration:
        raise KeyError(f"{seq_key} not found in raw calibration")
    cal = calibration[seq_key]
    R_w2c = np.asarray(cal["R"], dtype=np.float32).reshape(3, 3)
    T_w2c = np.asarray(cal["T"], dtype=np.float32).reshape(3)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R_w2c.T
    pose[:3, 3] = -R_w2c.T @ T_w2c
    return pose.astype(np.float64)


def rotation_error_deg(pred: np.ndarray, target: np.ndarray) -> float:
    rel = pred[:3, :3] @ target[:3, :3].T
    trace = np.trace(rel)
    cos_angle = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def pose_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    return {
        "rot_err_deg": rotation_error_deg(pred, target),
        "trans_err": float(np.linalg.norm(pred[:3, 3] - target[:3, 3])),
        "pred_trans_norm": float(np.linalg.norm(pred[:3, 3])),
        "gt_trans_norm": float(np.linalg.norm(target[:3, 3])),
    }


def relative_rows(poses: list[np.ndarray], gt_poses: list[np.ndarray], mode: str) -> list[dict]:
    rows = []
    if mode == "adjacent":
        pairs = [(i, i + 1) for i in range(len(poses) - 1)]
    elif mode == "from_first":
        pairs = [(0, i) for i in range(len(poses))]
    else:
        raise ValueError(mode)

    for src, dst in pairs:
        pred_rel = np.linalg.inv(poses[src]) @ poses[dst]
        gt_rel = np.linalg.inv(gt_poses[src]) @ gt_poses[dst]
        rows.append({
            "transition": f"{src}->{dst}",
            **pose_metrics(pred_rel, gt_rel),
        })
    return rows


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {}
    rot = np.asarray([row["rot_err_deg"] for row in rows], dtype=np.float64)
    trans = np.asarray([row["trans_err"] for row in rows], dtype=np.float64)
    return {
        "mean_rot_err_deg": float(rot.mean()),
        "mean_trans_err": float(trans.mean()),
        "max_rot_err_deg": float(rot.max()),
        "max_trans_err": float(trans.max()),
    }


def load_saved_poses(output_dir: Path, num_frames: int) -> list[np.ndarray]:
    return [load_pose(output_dir / "camera" / f"{idx:06d}.npz") for idx in range(num_frames)]


def evaluate_output(name: str, output_dir: Path, gt_poses: list[np.ndarray]) -> dict:
    poses = load_saved_poses(output_dir, len(gt_poses))
    adjacent = relative_rows(poses, gt_poses, mode="adjacent")
    from_first = relative_rows(poses, gt_poses, mode="from_first")
    boundary = next((row for row in adjacent if row["transition"] == "1->2"), None)
    return {
        "name": name,
        "output_dir": str(output_dir),
        "adjacent": adjacent,
        "from_first": from_first,
        "summary_adjacent": summarize(adjacent),
        "summary_from_first": summarize(from_first),
        "boundary_1_to_2": boundary,
    }


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    group = manifest.get("group") or str(manifest["frames"][0]["seq"]).split("/", 1)[0]
    raw_calibration_root = args.raw_calibration_root or (args.data_root / f"avatarrex_{group}")
    calibration = load_raw_calibration(raw_calibration_root)
    gt_poses = [raw_calibration_c2w(calibration, frame["seq"]) for frame in manifest["frames"]]

    results = {
        "input_manifest": str(args.input_manifest),
        "view_angle_deg": manifest.get("view_angle_deg"),
        "group": manifest.get("group"),
        "seqA": manifest.get("seqA"),
        "seqB": manifest.get("seqB"),
        "start_frame": manifest.get("start_frame"),
        "gt_source": "raw_calibration",
        "raw_calibration_root": str(raw_calibration_root),
        "target": "T_target_i = inv(raw_camera_pose_0) @ raw_camera_pose_i",
        "pred": evaluate_output("pred", args.pred_dir, gt_poses),
    }
    if args.raw_dir is not None:
        results["raw"] = evaluate_output("raw", args.raw_dir, gt_poses)

    if "raw" in results:
        pred_boundary = results["pred"]["boundary_1_to_2"]
        raw_boundary = results["raw"]["boundary_1_to_2"]
        pred_summary = results["pred"]["summary_adjacent"]
        raw_summary = results["raw"]["summary_adjacent"]
        results["comparison"] = {
            "boundary_rot_improvement_deg": raw_boundary["rot_err_deg"] - pred_boundary["rot_err_deg"],
            "boundary_trans_improvement": raw_boundary["trans_err"] - pred_boundary["trans_err"],
            "mean_adjacent_rot_improvement_deg": raw_summary["mean_rot_err_deg"] - pred_summary["mean_rot_err_deg"],
            "mean_adjacent_trans_improvement": raw_summary["mean_trans_err"] - pred_summary["mean_trans_err"],
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(results.get("comparison", results["pred"]["summary_adjacent"]), indent=2, sort_keys=True))
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
