#!/usr/bin/env python3
"""Evaluate saved 4-frame AvatarReX image-only outputs against GT camera poses.

The model is not run here. This script only reads saved ``camera/*.npz`` files
and the RGB source paths recorded in ``input_manifest.json``.
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
    return parser.parse_args()


def load_pose(path: Path) -> np.ndarray:
    data = np.load(path)
    return data["pose"].astype(np.float64)


def source_to_gt_cam_path(source: str) -> Path:
    path = Path(source)
    parts = list(path.parts)
    try:
        rgb_idx = parts.index("rgb")
    except ValueError as exc:
        raise ValueError(f"Source path does not contain /rgb/: {source}") from exc
    parts[rgb_idx] = "cam"
    return Path(*parts).with_suffix(".npz")


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
    sources = [frame["source"] for frame in manifest["frames"]]
    gt_cam_paths = [source_to_gt_cam_path(source) for source in sources]
    gt_poses = [load_pose(path) for path in gt_cam_paths]

    results = {
        "input_manifest": str(args.input_manifest),
        "view_angle_deg": manifest.get("view_angle_deg"),
        "group": manifest.get("group"),
        "seqA": manifest.get("seqA"),
        "seqB": manifest.get("seqB"),
        "start_frame": manifest.get("start_frame"),
        "gt_cam_paths": [str(path) for path in gt_cam_paths],
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
