#!/usr/bin/env python3
"""Build fixed AvatarReX AABB manifests for V8.1 large-scale training."""

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
    parser.add_argument("--avatarrex_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data"))
    parser.add_argument("--split", default="Training")
    parser.add_argument("--raw_calibration_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data/AvatarReX_raw_meta/lbn1"))
    parser.add_argument("--output_dir", type=Path, default=Path("output/v8_1_aabb_manifests/stage_a_10k"))
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--min_angle", type=float, default=60.0)
    parser.add_argument("--train_clips", type=int, default=10000)
    parser.add_argument("--val_same_clips", type=int, default=200)
    parser.add_argument("--test_same_clips", type=int, default=200)
    parser.add_argument("--val_new_clips", type=int, default=200)
    parser.add_argument("--test_new_clips", type=int, default=200)
    parser.add_argument("--train_pair_count", type=int, default=61)
    parser.add_argument("--val_new_pair_count", type=int, default=13)
    parser.add_argument("--test_new_pair_count", type=int, default=13)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def raw_c2w(calibration: dict, seq: str) -> np.ndarray:
    cal = calibration[seq]
    r_w2c = np.asarray(cal["R"], dtype=np.float64).reshape(3, 3)
    t_w2c = np.asarray(cal["T"], dtype=np.float64).reshape(3)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = r_w2c.T
    pose[:3, 3] = -r_w2c.T @ t_w2c
    return pose


def camera_angle_deg(pose_a: np.ndarray, pose_b: np.ndarray) -> float:
    dir_a = pose_a[:3, 2]
    dir_b = pose_b[:3, 2]
    dir_a = dir_a / max(np.linalg.norm(dir_a), 1e-12)
    dir_b = dir_b / max(np.linalg.norm(dir_b), 1e-12)
    cos_angle = float(np.clip(np.dot(dir_a, dir_b), -1.0, 1.0))
    return math.degrees(math.acos(cos_angle))


def list_frame_ids(split_dir: Path, seqs: list[str]) -> list[int]:
    all_ids = []
    for seq in seqs:
        ids = sorted(int(path.stem) for path in (split_dir / seq / "rgb").glob("*.png"))
        if not ids:
            raise FileNotFoundError(f"No RGB frames found for {seq}")
        all_ids.append(ids)
    first = all_ids[0]
    for seq, ids in zip(seqs, all_ids):
        if ids != first:
            raise ValueError(f"Frame ids differ for {seq}")
    return first


def angle_bucket(angle: float) -> str:
    if angle < 90.0:
        return "060_090"
    if angle < 120.0:
        return "090_120"
    if angle < 150.0:
        return "120_150"
    return "150_180"


def sample_records(
    ordered_pairs: list[tuple[str, str, float]],
    start_frames: list[int],
    total: int,
    rng: random.Random,
    angle_balanced: bool,
) -> list[dict]:
    if not ordered_pairs or not start_frames or total <= 0:
        return []

    by_bucket: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
    for pair in ordered_pairs:
        by_bucket[angle_bucket(pair[2])].append(pair)

    if angle_balanced:
        buckets = [bucket for bucket in ["060_090", "090_120", "120_150", "150_180"] if by_bucket.get(bucket)]
        counts = {bucket: total // len(buckets) for bucket in buckets}
        for bucket in buckets[: total % len(buckets)]:
            counts[bucket] += 1
    else:
        buckets = ["all"]
        by_bucket = {"all": ordered_pairs}
        counts = {"all": total}

    records = []
    seen = set()
    for bucket in buckets:
        pairs = by_bucket[bucket]
        target = counts[bucket]
        max_unique = len(pairs) * len(start_frames)
        if target > max_unique:
            raise ValueError(f"Requested {target} clips from bucket {bucket}, but only {max_unique} unique clips exist")
        while sum(1 for r in records if r["angle_bucket"] == bucket) < target:
            seq_a, seq_b, angle = rng.choice(pairs)
            start = rng.choice(start_frames)
            key = (seq_a, seq_b, start)
            if key in seen:
                continue
            seen.add(key)
            records.append(
                {
                    "seqA": seq_a,
                    "seqB": seq_b,
                    "start_frame": int(start),
                    "view_angle_deg": round(float(angle), 6),
                    "angle_bucket": bucket,
                }
            )
    rng.shuffle(records)
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def summarize(records: list[dict]) -> dict:
    by_bucket = defaultdict(int)
    by_pair = defaultdict(int)
    for record in records:
        by_bucket[record["angle_bucket"]] += 1
        by_pair[f"{record['seqA']}->{record['seqB']}"] += 1
    angles = [float(record["view_angle_deg"]) for record in records]
    starts = [int(record["start_frame"]) for record in records]
    return {
        "num_records": len(records),
        "num_ordered_pairs": len(by_pair),
        "angle_min": min(angles) if angles else None,
        "angle_max": max(angles) if angles else None,
        "start_min": min(starts) if starts else None,
        "start_max": max(starts) if starts else None,
        "angle_bucket_counts": dict(sorted(by_bucket.items())),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} already exists; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    split_dir = args.avatarrex_root / args.split
    seqs = sorted(path.name for path in split_dir.iterdir() if path.is_dir())
    calibration = json.loads((args.raw_calibration_root / "calibration_full.json").read_text(encoding="utf-8"))
    frame_ids = list_frame_ids(split_dir, seqs)
    max_start = frame_ids[-1] - 3

    poses = {seq: raw_c2w(calibration, seq) for seq in seqs}
    unordered_pairs = []
    for i, seq_a in enumerate(seqs):
        for seq_b in seqs[i + 1 :]:
            angle = camera_angle_deg(poses[seq_a], poses[seq_b])
            if angle >= args.min_angle:
                unordered_pairs.append((seq_a, seq_b, angle))
    unordered_pairs.sort(key=lambda item: (-item[2], item[0], item[1]))
    rng.shuffle(unordered_pairs)

    needed_pairs = args.train_pair_count + args.val_new_pair_count + args.test_new_pair_count
    if needed_pairs > len(unordered_pairs):
        raise ValueError(f"Requested {needed_pairs} unordered pairs, only {len(unordered_pairs)} pass min_angle")

    train_pairs = unordered_pairs[: args.train_pair_count]
    val_new_pairs = unordered_pairs[args.train_pair_count : args.train_pair_count + args.val_new_pair_count]
    test_new_pairs = unordered_pairs[
        args.train_pair_count + args.val_new_pair_count : needed_pairs
    ]

    def ordered(pairs: list[tuple[str, str, float]]) -> list[tuple[str, str, float]]:
        out = []
        for seq_a, seq_b, angle in pairs:
            out.append((seq_a, seq_b, angle))
            out.append((seq_b, seq_a, angle))
        return out

    # Use gaps between time windows so AABB clips do not share frames across splits.
    train_starts = list(range(0, min(1496, max_start) + 1))
    val_same_starts = list(range(1500, min(1696, max_start) + 1))
    test_same_starts = list(range(1700, max_start + 1))
    new_pair_starts = list(range(0, max_start + 1))

    outputs = {
        "stage_a_train_10k.jsonl": sample_records(ordered(train_pairs), train_starts, args.train_clips, rng, True),
        "stage_a_val_same_200.jsonl": sample_records(ordered(train_pairs), val_same_starts, args.val_same_clips, rng, True),
        "stage_a_test_same_200.jsonl": sample_records(ordered(train_pairs), test_same_starts, args.test_same_clips, rng, True),
        "stage_a_val_new_200.jsonl": sample_records(ordered(val_new_pairs), new_pair_starts, args.val_new_clips, rng, True),
        "stage_a_test_new_200.jsonl": sample_records(ordered(test_new_pairs), new_pair_starts, args.test_new_clips, rng, True),
    }

    for filename, records in outputs.items():
        write_jsonl(args.output_dir / filename, records)

    metadata = {
        "avatarrex_root": str(args.avatarrex_root),
        "split": args.split,
        "raw_calibration_root": str(args.raw_calibration_root),
        "seed": args.seed,
        "min_angle": args.min_angle,
        "num_sequences": len(seqs),
        "sequences": seqs,
        "num_frames": len(frame_ids),
        "frame_min": frame_ids[0],
        "frame_max": frame_ids[-1],
        "valid_start_min": frame_ids[0],
        "valid_start_max": max_start,
        "num_unordered_pairs_passing_angle": len(unordered_pairs),
        "split_pair_counts": {
            "train_unordered": len(train_pairs),
            "val_new_unordered": len(val_new_pairs),
            "test_new_unordered": len(test_new_pairs),
        },
        "time_windows": {
            "train": [train_starts[0], train_starts[-1]],
            "val_same": [val_same_starts[0], val_same_starts[-1]],
            "test_same": [test_same_starts[0], test_same_starts[-1]],
            "new_pair": [new_pair_starts[0], new_pair_starts[-1]],
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
