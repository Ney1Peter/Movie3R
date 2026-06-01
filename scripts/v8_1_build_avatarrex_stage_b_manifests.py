#!/usr/bin/env python3
"""Build grouped AvatarReX Stage-B manifests.

Stage B uses grouped training data:

  Training/lbn1/<camera_seq>/
  Training/zxc/<camera_seq>/
  Training/zzr/<camera_seq>/

The manifest stores sequence names as relative paths, e.g.:

  {"seqA": "lbn1/22053903", "seqB": "lbn1/22139907", ...}

This lets one AvatarReX_AABB dataset read multiple people/actions while still
keeping each AABB clip inside one group.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training_root",
        type=Path,
        default=Path("/data/wangzheng/iJCV-CODE/data/Avatarrex_output/Training"),
    )
    parser.add_argument("--output_dir", type=Path, default=Path("output/v8_1_aabb_manifests/stage_b_40k"))
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--min_angle", type=float, default=60.0)
    parser.add_argument("--train_groups", nargs="+", default=["lbn1", "zxc"])
    parser.add_argument("--test_new_group", default="zzr")
    parser.add_argument("--train_clips", type=int, default=40000)
    parser.add_argument("--val_in_clips", type=int, default=2000)
    parser.add_argument("--test_in_clips", type=int, default=2000)
    parser.add_argument("--test_new_clips", type=int, default=2000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_pose(seq_dir: Path, frame_id: int) -> np.ndarray:
    return np.load(seq_dir / "cam" / f"{frame_id:08d}.npz")["pose"].astype(np.float64)


def camera_angle_deg(pose_a: np.ndarray, pose_b: np.ndarray) -> float:
    dir_a = pose_a[:3, 2]
    dir_b = pose_b[:3, 2]
    dir_a = dir_a / max(np.linalg.norm(dir_a), 1e-12)
    dir_b = dir_b / max(np.linalg.norm(dir_b), 1e-12)
    return math.degrees(math.acos(float(np.clip(np.dot(dir_a, dir_b), -1.0, 1.0))))


def angle_bucket(angle: float) -> str:
    if angle < 90.0:
        return "060_090"
    if angle < 120.0:
        return "090_120"
    if angle < 150.0:
        return "120_150"
    return "150_180"


def frame_ids(seq_dir: Path) -> list[int]:
    return sorted(int(path.stem) for path in (seq_dir / "rgb").glob("*.png"))


def has_clip(group_dir: Path, seq_a: str, seq_b: str, start: int) -> bool:
    return (
        (group_dir / seq_a / "rgb" / f"{start:08d}.png").is_file()
        and (group_dir / seq_a / "rgb" / f"{start + 1:08d}.png").is_file()
        and (group_dir / seq_b / "rgb" / f"{start + 2:08d}.png").is_file()
        and (group_dir / seq_b / "rgb" / f"{start + 3:08d}.png").is_file()
        and (group_dir / seq_a / "cam" / f"{start:08d}.npz").is_file()
        and (group_dir / seq_b / "cam" / f"{start + 2:08d}.npz").is_file()
        and (group_dir / seq_a / "smpl" / f"{start:08d}.pkl").is_file()
        and (group_dir / seq_b / "smpl" / f"{start + 2:08d}.pkl").is_file()
    )


def group_sequences(training_root: Path, group: str) -> list[str]:
    group_dir = training_root / group
    if not group_dir.is_dir():
        raise FileNotFoundError(group_dir)
    seqs = sorted(path.name for path in group_dir.iterdir() if (path / "rgb").is_dir())
    if len(seqs) < 2:
        raise ValueError(f"Need at least two sequence folders under {group_dir}")
    return seqs


def group_windows(training_root: Path, group: str, seqs: list[str]) -> dict[str, list[int]]:
    first_ids = frame_ids(training_root / group / seqs[0])
    max_start = first_ids[-1] - 3
    train_hi = int(max_start * 0.70)
    val_hi = int(max_start * 0.85)
    return {
        "train": list(range(first_ids[0], train_hi + 1)),
        "val_in": list(range(train_hi + 1, val_hi + 1)),
        "test_in": list(range(val_hi + 1, max_start + 1)),
        "test_new": list(range(first_ids[0], max_start + 1)),
    }


def ordered_pairs(training_root: Path, group: str, seqs: list[str], min_angle: float):
    group_dir = training_root / group
    pairs = []
    probe_frame = frame_ids(group_dir / seqs[0])[0]
    for seq_a in seqs:
        for seq_b in seqs:
            if seq_a == seq_b:
                continue
            angle = camera_angle_deg(
                load_pose(group_dir / seq_a, probe_frame),
                load_pose(group_dir / seq_b, probe_frame),
            )
            if angle >= min_angle:
                pairs.append((seq_a, seq_b, float(angle), angle_bucket(angle)))
    return pairs


def candidate_records(training_root: Path, group: str, split_name: str, min_angle: float):
    seqs = group_sequences(training_root, group)
    windows = group_windows(training_root, group, seqs)
    pairs = ordered_pairs(training_root, group, seqs, min_angle)
    group_dir = training_root / group
    records = []
    for seq_a, seq_b, angle, bucket in pairs:
        for start in windows[split_name]:
            if has_clip(group_dir, seq_a, seq_b, int(start)):
                records.append(
                    {
                        "group": group,
                        "seqA": f"{group}/{seq_a}",
                        "seqB": f"{group}/{seq_b}",
                        "start_frame": int(start),
                        "view_angle_deg": round(angle, 6),
                        "angle_bucket": bucket,
                    }
                )
    return records, {"sequences": seqs, "windows": {k: [v[0], v[-1]] for k, v in windows.items()}, "pairs": len(pairs)}


def balanced_sample(records: list[dict], total: int, rng: random.Random, strata_keys=("group", "angle_bucket")):
    if total <= 0:
        return []
    strata = defaultdict(list)
    for record in records:
        key = tuple(record[k] for k in strata_keys)
        strata[key].append(record)
    keys = sorted(strata)
    if not keys:
        raise ValueError("No candidate records to sample from")

    base = total // len(keys)
    counts = {key: base for key in keys}
    for key in keys[: total % len(keys)]:
        counts[key] += 1

    sampled = []
    for key in keys:
        pool = strata[key][:]
        need = counts[key]
        if len(pool) < need:
            raise ValueError(f"Need {need} records from stratum {key}, only {len(pool)} available")
        sampled.extend(rng.sample(pool, need))
    rng.shuffle(sampled)
    return sampled


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def summarize(records: list[dict]) -> dict:
    by_group = defaultdict(int)
    by_bucket = defaultdict(int)
    by_pair = defaultdict(int)
    keys = set()
    angles, starts = [], []
    for record in records:
        by_group[record["group"]] += 1
        by_bucket[record["angle_bucket"]] += 1
        by_pair[f"{record['seqA']}->{record['seqB']}"] += 1
        key = (record["seqA"], record["seqB"], int(record["start_frame"]))
        keys.add(key)
        angles.append(float(record["view_angle_deg"]))
        starts.append(int(record["start_frame"]))
    return {
        "num_records": len(records),
        "num_unique_clip_keys": len(keys),
        "has_duplicate_clip_keys": len(keys) != len(records),
        "num_ordered_pairs": len(by_pair),
        "angle_min": min(angles) if angles else None,
        "angle_max": max(angles) if angles else None,
        "start_min": min(starts) if starts else None,
        "start_max": max(starts) if starts else None,
        "group_counts": dict(sorted(by_group.items())),
        "angle_bucket_counts": dict(sorted(by_bucket.items())),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} already exists; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    train_candidates, group_meta = [], {}
    val_candidates, test_in_candidates = [], []
    for group in args.train_groups:
        train_group_records, meta = candidate_records(args.training_root, group, "train", args.min_angle)
        val_group_records, _ = candidate_records(args.training_root, group, "val_in", args.min_angle)
        test_group_records, _ = candidate_records(args.training_root, group, "test_in", args.min_angle)
        train_candidates.extend(train_group_records)
        val_candidates.extend(val_group_records)
        test_in_candidates.extend(test_group_records)
        group_meta[group] = meta

    test_new_candidates, meta = candidate_records(args.training_root, args.test_new_group, "test_new", args.min_angle)
    group_meta[args.test_new_group] = meta

    outputs = {
        "stage_b_train_40k.jsonl": balanced_sample(train_candidates, args.train_clips, rng),
        "stage_b_val_in_2k.jsonl": balanced_sample(val_candidates, args.val_in_clips, rng),
        "stage_b_test_in_2k.jsonl": balanced_sample(test_in_candidates, args.test_in_clips, rng),
        "stage_b_test_new_zzr_2k.jsonl": balanced_sample(
            test_new_candidates,
            args.test_new_clips,
            rng,
            strata_keys=("angle_bucket",),
        ),
    }
    for filename, records in outputs.items():
        write_jsonl(args.output_dir / filename, records)

    metadata = {
        "training_root": str(args.training_root),
        "seed": args.seed,
        "min_angle": args.min_angle,
        "train_groups": args.train_groups,
        "test_new_group": args.test_new_group,
        "group_metadata": group_meta,
        "candidate_counts": {
            "train": len(train_candidates),
            "val_in": len(val_candidates),
            "test_in": len(test_in_candidates),
            "test_new": len(test_new_candidates),
        },
        "manifests": {filename: summarize(records) for filename, records in outputs.items()},
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata["manifests"], indent=2, sort_keys=True))
    print(f"Wrote manifests to {args.output_dir}")


if __name__ == "__main__":
    main()
