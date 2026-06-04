#!/usr/bin/env python3
"""Scan V8.4 manifests for GT human and mask coverage.

This script is intentionally lightweight: it does not run Human3R/MHMR. It
checks whether each manifest clip has SMPL annotations and non-empty foreground
masks for every frame. This separates bad GT/data samples from model-side human
detection failures.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np


def frame_specs(record: dict) -> list[tuple[str, int]]:
    start = int(record["start_frame"])
    if record.get("clip_type") == "aabb":
        return [
            (record["seqA"], start),
            (record["seqA"], start + 1),
            (record["seqB"], start + 2),
            (record["seqB"], start + 3),
        ]
    return [(record["seq"], start + i) for i in range(4)]


def scene_path(root: Path, seq: str) -> Path:
    return root / "Training" / seq


def smpl_count(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        with path.open("rb") as f:
            data = pickle.load(f)
    except Exception:
        return 0
    return len(data) if isinstance(data, list) else 0


def mask_pixels(path: Path) -> int:
    if not path.is_file():
        return 0
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return 0
    return int((mask > 10).sum())


def scan_manifest(manifest: Path, root: Path, limit: int | None, groups: set[str] | None) -> dict:
    stats = {
        "manifest": str(manifest),
        "clips": 0,
        "frames": 0,
        "clips_all_smpl": 0,
        "clips_all_mask": 0,
        "clips_all_smpl_and_mask": 0,
        "frames_missing_smpl": 0,
        "frames_missing_mask": 0,
        "frames_tiny_mask": 0,
        "groups": {},
        "bad_examples": [],
    }
    tiny_threshold = 64

    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            if limit is not None and stats["clips"] >= limit:
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            group = rec.get("group", "unknown")
            if groups is not None and group not in groups:
                continue
            stats["clips"] += 1
            g = stats["groups"].setdefault(
                group,
                {"clips": 0, "all_smpl": 0, "all_mask": 0, "all_smpl_and_mask": 0},
            )
            g["clips"] += 1

            has_smpl = []
            has_mask = []
            mask_counts = []
            for seq, frame in frame_specs(rec):
                base = scene_path(root, seq)
                smpl_path = base / "smpl" / f"{frame:08d}.pkl"
                mask_path = base / "mask" / f"{frame:08d}.png"
                n_smpl = smpl_count(smpl_path)
                n_mask = mask_pixels(mask_path)
                stats["frames"] += 1
                has_smpl.append(n_smpl > 0)
                has_mask.append(n_mask > tiny_threshold)
                mask_counts.append(n_mask)
                if n_smpl <= 0:
                    stats["frames_missing_smpl"] += 1
                if n_mask <= 0:
                    stats["frames_missing_mask"] += 1
                if 0 < n_mask <= tiny_threshold:
                    stats["frames_tiny_mask"] += 1

            all_smpl = all(has_smpl)
            all_mask = all(has_mask)
            if all_smpl:
                stats["clips_all_smpl"] += 1
                g["all_smpl"] += 1
            if all_mask:
                stats["clips_all_mask"] += 1
                g["all_mask"] += 1
            if all_smpl and all_mask:
                stats["clips_all_smpl_and_mask"] += 1
                g["all_smpl_and_mask"] += 1
            elif len(stats["bad_examples"]) < 20:
                stats["bad_examples"].append(
                    {
                        "record": rec,
                        "has_smpl": has_smpl,
                        "has_mask": has_mask,
                        "mask_pixels": mask_counts,
                    }
                )

    for key in ["clips_all_smpl", "clips_all_mask", "clips_all_smpl_and_mask"]:
        stats[key + "_ratio"] = stats[key] / max(stats["clips"], 1)
    stats["frames_missing_smpl_ratio"] = stats["frames_missing_smpl"] / max(stats["frames"], 1)
    stats["frames_missing_mask_ratio"] = stats["frames_missing_mask"] / max(stats["frames"], 1)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/wangzheng/iJCV-CODE/data")
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--groups",
        default="",
        help="Comma-separated group allow-list, e.g. lbn1,zxc,zzr. Empty means all groups.",
    )
    parser.add_argument("--output", default="output/v8_4_batch_probe/manifest_human_coverage.json")
    args = parser.parse_args()

    root = Path(args.root)
    groups = {x.strip() for x in args.groups.split(",") if x.strip()} or None
    results = [scan_manifest(Path(path), root, args.limit, groups) for path in args.manifest]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
