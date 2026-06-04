#!/usr/bin/env python3
"""Build mixed AABB/AAAA manifests with an explicit held-out test folder.

The goal is to train V8 pose correction on both:
  - AABB clips: camera/view jump clips, shot_label=[0,0,1,0]
  - AAAA clips: same-camera continuous clips, shot_label=[0,0,0,0]

Test clips are selected from frame windows that do not overlap train/val frame
windows. The script materializes only the selected test files as symlinks under
``data/Test/<split_name>/...`` so the test set is explicit without duplicating
large image files.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


NUM_VIEWS = 4
FILE_SUBDIRS = ("rgb", "cam", "smpl", "mask")
FRAME_IDS_CACHE: dict[Path, list[int]] = {}
FRAME_SET_CACHE: dict[tuple[Path, str], set[int]] = {}
CLIP_START_CACHE: dict[Path, list[int]] = {}
POSE_CACHE: dict[tuple[Path, int], np.ndarray] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training_root",
        type=Path,
        default=Path("/data/wangzheng/iJCV-CODE/data/Training"),
    )
    parser.add_argument(
        "--test_root",
        type=Path,
        default=Path("/data/wangzheng/iJCV-CODE/data/Test/v8_4_mixed_aabb_aaaa"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output/v8_4_mixed_aabb_aaaa_manifests"),
    )
    parser.add_argument("--groups", nargs="+", default=["lbn1", "zxc", "zzr", "thuman00", "thuman02"])
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--train_aabb", type=int, default=60000)
    parser.add_argument("--train_aaaa", type=int, default=20000)
    parser.add_argument("--val_aabb", type=int, default=2000)
    parser.add_argument("--val_aaaa", type=int, default=1000)
    parser.add_argument("--test_aabb", type=int, default=2000)
    parser.add_argument("--test_aaaa", type=int, default=1000)
    parser.add_argument("--min_aabb_angle", type=float, default=15.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_test_symlinks", action="store_true")
    return parser.parse_args()


def is_sequence_dir(path: Path) -> bool:
    return (path / "rgb").is_dir() and (path / "cam").is_dir() and (path / "smpl").is_dir()


def group_sequences(training_root: Path, group: str) -> list[str]:
    group_dir = training_root / group
    if not group_dir.is_dir():
        raise FileNotFoundError(group_dir)
    seqs = sorted(path.name for path in group_dir.iterdir() if path.is_dir() and is_sequence_dir(path))
    if not seqs:
        raise ValueError(f"No sequence dirs found under {group_dir}")
    return seqs


def frame_ids(seq_dir: Path) -> list[int]:
    seq_dir = seq_dir.resolve()
    if seq_dir not in FRAME_IDS_CACHE:
        FRAME_IDS_CACHE[seq_dir] = sorted(int(path.stem) for path in (seq_dir / "rgb").glob("*.png"))
    return FRAME_IDS_CACHE[seq_dir]


def available_frame_set(seq_dir: Path, subdir: str) -> set[int]:
    seq_dir = seq_dir.resolve()
    key = (seq_dir, subdir)
    if key in FRAME_SET_CACHE:
        return FRAME_SET_CACHE[key]
    suffix = {"rgb": ".png", "cam": ".npz", "smpl": ".pkl", "mask": ".png"}[subdir]
    path = seq_dir / subdir
    if not path.is_dir():
        frames = set()
    else:
        frames = {int(file.stem) for file in path.glob(f"*{suffix}")}
    FRAME_SET_CACHE[key] = frames
    return frames


def numeric_clip_starts(seq_dir: Path) -> list[int]:
    seq_dir = seq_dir.resolve()
    if seq_dir not in CLIP_START_CACHE:
        ids = set(frame_ids(seq_dir))
        CLIP_START_CACHE[seq_dir] = sorted(
            fid for fid in ids if all((fid + off) in ids for off in range(NUM_VIEWS))
        )
    return CLIP_START_CACHE[seq_dir]


def split_starts(starts: list[int]) -> dict[str, list[int]]:
    """Split start frames with a guard band to avoid cross-split image reuse."""
    if len(starts) < NUM_VIEWS * 6:
        return {"train": [], "val": [], "test": []}
    train_cut = int(len(starts) * 0.70)
    val_cut = int(len(starts) * 0.85)
    guard = NUM_VIEWS
    train = starts[: max(0, train_cut - guard)]
    val = starts[min(len(starts), train_cut + guard) : max(train_cut + guard, val_cut - guard)]
    test = starts[min(len(starts), val_cut + guard) :]
    return {"train": train, "val": val, "test": test}


def load_pose(seq_dir: Path, frame_id: int) -> np.ndarray:
    seq_dir = seq_dir.resolve()
    key = (seq_dir, int(frame_id))
    if key not in POSE_CACHE:
        POSE_CACHE[key] = np.load(seq_dir / "cam" / f"{frame_id:08d}.npz")["pose"].astype(np.float64)
    return POSE_CACHE[key]


def camera_angle_deg(pose_a: np.ndarray, pose_b: np.ndarray) -> float:
    dir_a = pose_a[:3, 2]
    dir_b = pose_b[:3, 2]
    dir_a = dir_a / max(np.linalg.norm(dir_a), 1e-12)
    dir_b = dir_b / max(np.linalg.norm(dir_b), 1e-12)
    return math.degrees(math.acos(float(np.clip(np.dot(dir_a, dir_b), -1.0, 1.0))))


def angle_bucket(angle: float) -> str:
    if angle < 30.0:
        return "015_030"
    if angle < 60.0:
        return "030_060"
    if angle < 90.0:
        return "060_090"
    if angle < 120.0:
        return "090_120"
    if angle < 150.0:
        return "120_150"
    return "150_180"


def has_files_for_frames(group_dir: Path, seq: str, frames: Iterable[int]) -> bool:
    seq_dir = group_dir / seq
    rgb_frames = available_frame_set(seq_dir, "rgb")
    cam_frames = available_frame_set(seq_dir, "cam")
    smpl_frames = available_frame_set(seq_dir, "smpl")
    for frame in frames:
        frame = int(frame)
        if frame not in rgb_frames or frame not in cam_frames or frame not in smpl_frames:
            return False
    return True


def pair_angle(group_dir: Path, seq_a: str, seq_b: str) -> float:
    common = sorted(set(frame_ids(group_dir / seq_a)).intersection(frame_ids(group_dir / seq_b)))
    if not common:
        raise ValueError(f"No common frames for {group_dir.name}/{seq_a} and {seq_b}")
    probe_frame = common[0]
    return camera_angle_deg(load_pose(group_dir / seq_a, probe_frame), load_pose(group_dir / seq_b, probe_frame))


def build_aaaa_candidates(training_root: Path, groups: list[str]) -> tuple[dict[str, list[dict]], dict]:
    candidates = {"train": [], "val": [], "test": []}
    meta = {}
    for group in groups:
        group_dir = training_root / group
        group_meta = {"sequences": {}, "num_sequences": 0}
        for seq in group_sequences(training_root, group):
            starts_by_split = split_starts(numeric_clip_starts(group_dir / seq))
            group_meta["sequences"][seq] = {name: len(vals) for name, vals in starts_by_split.items()}
            for split_name, starts in starts_by_split.items():
                for start in starts:
                    if not has_files_for_frames(group_dir, seq, range(start, start + NUM_VIEWS)):
                        continue
                    candidates[split_name].append(
                        {
                            "clip_type": "aaaa",
                            "group": group,
                            "seq": f"{group}/{seq}",
                            "start_frame": int(start),
                            "view_angle_deg": 0.0,
                            "angle_bucket": "same_camera",
                        }
                    )
        group_meta["num_sequences"] = len(group_meta["sequences"])
        meta[group] = group_meta
    return candidates, meta


def build_aabb_candidates(
    training_root: Path,
    groups: list[str],
    min_angle: float,
) -> tuple[dict[str, list[dict]], dict]:
    candidates = {"train": [], "val": [], "test": []}
    meta = {}
    for group in groups:
        group_dir = training_root / group
        seqs = group_sequences(training_root, group)
        first_starts = numeric_clip_starts(group_dir / seqs[0])
        starts_by_split = split_starts(first_starts)
        pair_count = 0
        bucket_counts = defaultdict(int)
        for seq_a in seqs:
            for seq_b in seqs:
                if seq_a == seq_b:
                    continue
                angle = pair_angle(group_dir, seq_a, seq_b)
                if angle < min_angle:
                    continue
                bucket = angle_bucket(angle)
                pair_count += 1
                bucket_counts[bucket] += 1
                for split_name, starts in starts_by_split.items():
                    for start in starts:
                        if not has_files_for_frames(group_dir, seq_a, (start, start + 1)):
                            continue
                        if not has_files_for_frames(group_dir, seq_b, (start + 2, start + 3)):
                            continue
                        candidates[split_name].append(
                            {
                                "clip_type": "aabb",
                                "group": group,
                                "seqA": f"{group}/{seq_a}",
                                "seqB": f"{group}/{seq_b}",
                                "start_frame": int(start),
                                "view_angle_deg": round(float(angle), 6),
                                "angle_bucket": bucket,
                            }
                        )
        meta[group] = {
            "num_sequences": len(seqs),
            "num_ordered_pairs_after_angle_filter": pair_count,
            "pair_angle_bucket_counts": dict(sorted(bucket_counts.items())),
            "starts_by_split": {name: len(vals) for name, vals in starts_by_split.items()},
        }
    return candidates, meta


def balanced_sample(records: list[dict], total: int, rng: random.Random, keys: tuple[str, ...]) -> list[dict]:
    if total <= 0:
        return []
    strata = defaultdict(list)
    for record in records:
        strata[tuple(record.get(key, "") for key in keys)].append(record)
    strata_keys = sorted(strata)
    if not strata_keys:
        raise ValueError("No candidate records to sample from")
    base = total // len(strata_keys)
    counts = {key: base for key in strata_keys}
    for key in strata_keys[: total % len(strata_keys)]:
        counts[key] += 1

    sampled = []
    leftovers = []
    shortage = 0
    for key in strata_keys:
        pool = strata[key][:]
        rng.shuffle(pool)
        need = counts[key]
        sampled.extend(pool[: min(need, len(pool))])
        leftovers.extend(pool[min(need, len(pool)) :])
        shortage += max(0, need - len(pool))
    if shortage:
        if len(leftovers) < shortage:
            raise ValueError(
                f"Requested {total} samples but only {len(records)} unique candidates are available"
            )
        rng.shuffle(leftovers)
        sampled.extend(leftovers[:shortage])
    rng.shuffle(sampled)
    return sampled


def distribute_counts(keys: list[tuple], total: int) -> dict[tuple, int]:
    if total <= 0:
        return {key: 0 for key in keys}
    if not keys:
        raise ValueError("No strata available for sampling")
    base = total // len(keys)
    counts = {key: base for key in keys}
    for key in keys[: total % len(keys)]:
        counts[key] += 1
    return counts


def build_aabb_sampling_strata(
    training_root: Path,
    groups: list[str],
    min_angle: float,
) -> tuple[dict[str, dict[tuple, dict]], dict]:
    strata = {"train": {}, "val": {}, "test": {}}
    meta = {}
    for group in groups:
        group_dir = training_root / group
        seqs = group_sequences(training_root, group)
        starts_by_split = split_starts(numeric_clip_starts(group_dir / seqs[0]))
        pair_buckets = defaultdict(list)
        for seq_a in seqs:
            for seq_b in seqs:
                if seq_a == seq_b:
                    continue
                angle = pair_angle(group_dir, seq_a, seq_b)
                if angle < min_angle:
                    continue
                bucket = angle_bucket(angle)
                pair_buckets[bucket].append((seq_a, seq_b, float(angle)))

        for split_name, starts in starts_by_split.items():
            for bucket, pairs in pair_buckets.items():
                if not starts or not pairs:
                    continue
                strata[split_name][(group, bucket)] = {
                    "group": group,
                    "group_dir": str(group_dir),
                    "bucket": bucket,
                    "pairs": pairs,
                    "starts": starts,
                    "capacity_upper_bound": len(pairs) * len(starts),
                }

        meta[group] = {
            "num_sequences": len(seqs),
            "starts_by_split": {name: len(vals) for name, vals in starts_by_split.items()},
            "pair_angle_bucket_counts": {bucket: len(pairs) for bucket, pairs in sorted(pair_buckets.items())},
            "capacity_upper_bound_by_split": {
                split_name: sum(
                    len(pairs) * len(starts_by_split[split_name])
                    for pairs in pair_buckets.values()
                )
                for split_name in ("train", "val", "test")
            },
        }
    return strata, meta


def sample_aabb_records(strata: dict[tuple, dict], total: int, rng: random.Random) -> list[dict]:
    keys = sorted(strata)
    counts = distribute_counts(keys, total)
    sampled = []
    global_seen = set()
    shortage = 0

    for key in keys:
        data = strata[key]
        need = counts[key]
        if need <= 0:
            continue
        capacity = int(data["capacity_upper_bound"])
        if capacity < need:
            shortage += need - capacity
            need = capacity

        local = []
        local_seen = set()
        pairs = data["pairs"]
        starts = data["starts"]
        group_dir = Path(data["group_dir"])
        max_attempts = max(need * 50, 1000)
        attempts = 0
        while len(local) < need and attempts < max_attempts:
            attempts += 1
            seq_a, seq_b, angle = rng.choice(pairs)
            start = int(rng.choice(starts))
            clip_key = (data["group"], seq_a, seq_b, start)
            if clip_key in local_seen or clip_key in global_seen:
                continue
            if not has_files_for_frames(group_dir, seq_a, (start, start + 1)):
                continue
            if not has_files_for_frames(group_dir, seq_b, (start + 2, start + 3)):
                continue
            local_seen.add(clip_key)
            global_seen.add(clip_key)
            local.append((seq_a, seq_b, start, angle))

        if len(local) < need:
            for seq_a, seq_b, angle in pairs:
                for start in starts:
                    clip_key = (data["group"], seq_a, seq_b, int(start))
                    if clip_key in local_seen or clip_key in global_seen:
                        continue
                    if not has_files_for_frames(group_dir, seq_a, (int(start), int(start) + 1)):
                        continue
                    if not has_files_for_frames(group_dir, seq_b, (int(start) + 2, int(start) + 3)):
                        continue
                    local_seen.add(clip_key)
                    global_seen.add(clip_key)
                    local.append((seq_a, seq_b, int(start), angle))
                    if len(local) >= need:
                        break
                if len(local) >= need:
                    break

        if len(local) < need:
            shortage += need - len(local)

        for seq_a, seq_b, start, angle in local:
            group = data["group"]
            bucket = data["bucket"]
            sampled.append(
                {
                    "clip_type": "aabb",
                    "group": group,
                    "seqA": f"{group}/{seq_a}",
                    "seqB": f"{group}/{seq_b}",
                    "start_frame": int(start),
                    "view_angle_deg": round(float(angle), 6),
                    "angle_bucket": bucket,
                }
            )

    if shortage:
        raise ValueError(f"AABB sampling shortage: requested {total}, missing {shortage} clips")
    rng.shuffle(sampled)
    return sampled


def record_frames(record: dict) -> set[tuple[str, int]]:
    start = int(record["start_frame"])
    if record["clip_type"] == "aaaa":
        seq = record["seq"]
        return {(seq, start + off) for off in range(NUM_VIEWS)}
    if record["clip_type"] == "aabb":
        return {
            (record["seqA"], start),
            (record["seqA"], start + 1),
            (record["seqB"], start + 2),
            (record["seqB"], start + 3),
        }
    raise ValueError(f"Unknown clip_type: {record['clip_type']}")


def record_frame_set(records: Iterable[dict]) -> set[tuple[str, int]]:
    used = set()
    for record in records:
        used.update(record_frames(record))
    return used


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def summarize(records: list[dict]) -> dict:
    by_group = defaultdict(int)
    by_bucket = defaultdict(int)
    by_type = defaultdict(int)
    keys = set()
    angles = []
    starts = []
    for record in records:
        by_group[record["group"]] += 1
        by_bucket[record["angle_bucket"]] += 1
        by_type[record["clip_type"]] += 1
        if record["clip_type"] == "aaaa":
            key = (record["clip_type"], record["seq"], int(record["start_frame"]))
        else:
            key = (record["clip_type"], record["seqA"], record["seqB"], int(record["start_frame"]))
        keys.add(key)
        angles.append(float(record["view_angle_deg"]))
        starts.append(int(record["start_frame"]))
    return {
        "num_records": len(records),
        "num_unique_clip_keys": len(keys),
        "has_duplicate_clip_keys": len(keys) != len(records),
        "num_unique_images": len(record_frame_set(records)),
        "angle_min": min(angles) if angles else None,
        "angle_max": max(angles) if angles else None,
        "start_min": min(starts) if starts else None,
        "start_max": max(starts) if starts else None,
        "clip_type_counts": dict(sorted(by_type.items())),
        "group_counts": dict(sorted(by_group.items())),
        "angle_bucket_counts": dict(sorted(by_bucket.items())),
    }


def source_file(training_root: Path, seq: str, subdir: str, frame: int) -> Path:
    suffix = {"rgb": ".png", "cam": ".npz", "smpl": ".pkl", "mask": ".png"}[subdir]
    return training_root / seq / subdir / f"{int(frame):08d}{suffix}"


def symlink_file(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    rel_src = os.path.relpath(src, dst.parent)
    os.symlink(rel_src, dst)


def materialize_test_folder(training_root: Path, test_root: Path, records: list[dict]) -> dict:
    linked = 0
    missing = []
    for seq, frame in sorted(record_frame_set(records)):
        for subdir in FILE_SUBDIRS:
            src = source_file(training_root, seq, subdir, frame)
            dst = test_root / seq / subdir / src.name
            before = dst.exists() or dst.is_symlink()
            symlink_file(src, dst)
            after = dst.exists() or dst.is_symlink()
            linked += int((not before) and after)
            if subdir != "mask" and not src.exists():
                missing.append(str(src))
    return {"test_root": str(test_root), "new_symlinks": linked, "missing_required_files": missing[:50]}


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} already exists; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    aabb_strata, aabb_meta = build_aabb_sampling_strata(args.training_root, args.groups, args.min_aabb_angle)
    aaaa_candidates, aaaa_meta = build_aaaa_candidates(args.training_root, args.groups)

    outputs = {
        "train_aabb_60k.jsonl": sample_aabb_records(aabb_strata["train"], args.train_aabb, rng),
        "train_aaaa_20k.jsonl": balanced_sample(
            aaaa_candidates["train"], args.train_aaaa, rng, keys=("group",)
        ),
        "val_aabb_2k.jsonl": sample_aabb_records(aabb_strata["val"], args.val_aabb, rng),
        "val_aaaa_1k.jsonl": balanced_sample(
            aaaa_candidates["val"], args.val_aaaa, rng, keys=("group",)
        ),
        "test_aabb_2k.jsonl": sample_aabb_records(aabb_strata["test"], args.test_aabb, rng),
        "test_aaaa_1k.jsonl": balanced_sample(
            aaaa_candidates["test"], args.test_aaaa, rng, keys=("group",)
        ),
    }

    for filename, records in outputs.items():
        write_jsonl(args.output_dir / filename, records)

    train_records = outputs["train_aabb_60k.jsonl"] + outputs["train_aaaa_20k.jsonl"]
    val_records = outputs["val_aabb_2k.jsonl"] + outputs["val_aaaa_1k.jsonl"]
    test_records = outputs["test_aabb_2k.jsonl"] + outputs["test_aaaa_1k.jsonl"]
    train_frames = record_frame_set(train_records)
    val_frames = record_frame_set(val_records)
    test_frames = record_frame_set(test_records)

    test_symlink_summary = None
    if not args.no_test_symlinks:
        test_symlink_summary = materialize_test_folder(args.training_root, args.test_root, test_records)

    metadata = {
        "training_root": str(args.training_root),
        "test_root": str(args.test_root),
        "seed": args.seed,
        "groups": args.groups,
        "num_views": NUM_VIEWS,
        "min_aabb_angle": args.min_aabb_angle,
        "candidate_counts": {
            "aabb_capacity_upper_bound": {
                split: int(sum(data["capacity_upper_bound"] for data in split_strata.values()))
                for split, split_strata in aabb_strata.items()
            },
            "aaaa": {split: len(records) for split, records in aaaa_candidates.items()},
        },
        "aabb_group_metadata": aabb_meta,
        "aaaa_group_metadata": aaaa_meta,
        "manifests": {filename: summarize(records) for filename, records in outputs.items()},
        "frame_overlap": {
            "train_val": len(train_frames.intersection(val_frames)),
            "train_test": len(train_frames.intersection(test_frames)),
            "val_test": len(val_frames.intersection(test_frames)),
        },
        "test_symlinks": test_symlink_summary,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "manifests": metadata["manifests"],
        "frame_overlap": metadata["frame_overlap"],
        "test_symlinks": metadata["test_symlinks"],
    }, indent=2, sort_keys=True))
    print(f"Wrote manifests to {args.output_dir}")


if __name__ == "__main__":
    main()
