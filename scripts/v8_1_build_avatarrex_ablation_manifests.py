#!/usr/bin/env python3
"""Build small grouped AvatarReX AABB manifests for V8.1 ablation.

Unlike the large Stage-B manifest builder, this script is meant for controlled
small experiments. It mixes lbn1/zxc/zzr in every split and avoids reusing the
same RGB frame inside a split.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


RAW_ROOTS = {
    "lbn1": "/data/wangzheng/iJCV-CODE/data/avatarrex_lbn1",
    "zxc": "/data/wangzheng/iJCV-CODE/data/avatarrex_zxc",
    "zzr": "/data/wangzheng/iJCV-CODE/data/avatarrex_zzr",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data/Avatarrex_output/Training"))
    parser.add_argument("--output_dir", type=Path, default=Path("output/v8_1_aabb_manifests/round1_ablation_small"))
    parser.add_argument("--groups", nargs="+", default=["lbn1", "zxc", "zzr"])
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--min_angle", type=float, default=120.0)
    parser.add_argument("--train_clips", type=int, default=240)
    parser.add_argument("--val_clips", type=int, default=60)
    parser.add_argument("--test_clips", type=int, default=60)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def frame_ids(seq_dir: Path) -> list[int]:
    return sorted(int(path.stem) for path in (seq_dir / "rgb").glob("*.png"))


def load_raw_c2w(group: str, seq: str) -> np.ndarray:
    cal_path = Path(RAW_ROOTS[group]) / "calibration_full.json"
    cal = json.loads(cal_path.read_text(encoding="utf-8"))[seq]
    r_w2c = np.asarray(cal["R"], dtype=np.float64).reshape(3, 3)
    t_w2c = np.asarray(cal["T"], dtype=np.float64).reshape(3)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = r_w2c.T
    pose[:3, 3] = -r_w2c.T @ t_w2c
    return pose


def camera_angle_deg(group: str, seq_a: str, seq_b: str) -> float:
    pose_a = load_raw_c2w(group, seq_a)
    pose_b = load_raw_c2w(group, seq_b)
    dir_a = pose_a[:3, 2]
    dir_b = pose_b[:3, 2]
    dir_a = dir_a / max(np.linalg.norm(dir_a), 1e-12)
    dir_b = dir_b / max(np.linalg.norm(dir_b), 1e-12)
    return math.degrees(math.acos(float(np.clip(np.dot(dir_a, dir_b), -1.0, 1.0))))


def angle_bucket(angle: float) -> str:
    if angle < 150.0:
        return "120_150"
    return "150_180"


def split_windows(first_id: int, last_valid_start: int) -> dict[str, list[int]]:
    train_hi = int(last_valid_start * 0.60)
    val_hi = int(last_valid_start * 0.80)
    return {
        "train": list(range(first_id, train_hi + 1)),
        "val": list(range(train_hi + 1, val_hi + 1)),
        "test": list(range(val_hi + 1, last_valid_start + 1)),
    }


def has_clip(group_dir: Path, seq_a: str, seq_b: str, start: int) -> bool:
    required = [
        group_dir / seq_a / "rgb" / f"{start:08d}.png",
        group_dir / seq_a / "rgb" / f"{start + 1:08d}.png",
        group_dir / seq_b / "rgb" / f"{start + 2:08d}.png",
        group_dir / seq_b / "rgb" / f"{start + 3:08d}.png",
        group_dir / seq_a / "smpl" / f"{start:08d}.pkl",
        group_dir / seq_b / "smpl" / f"{start + 2:08d}.pkl",
    ]
    return all(path.is_file() for path in required)


def clip_frames(group: str, seq_a: str, seq_b: str, start: int) -> set[tuple[str, str, int]]:
    return {
        (group, seq_a, start),
        (group, seq_a, start + 1),
        (group, seq_b, start + 2),
        (group, seq_b, start + 3),
    }


def candidates_for_group(training_root: Path, group: str, split: str, min_angle: float) -> list[dict]:
    group_dir = training_root / group
    seqs = sorted(path.name for path in group_dir.iterdir() if (path / "rgb").is_dir())
    if len(seqs) < 2:
        raise ValueError(f"Need at least two sequence folders under {group_dir}")
    ids = frame_ids(group_dir / seqs[0])
    windows = split_windows(ids[0], ids[-1] - 3)
    records = []
    for seq_a in seqs:
        for seq_b in seqs:
            if seq_a == seq_b:
                continue
            angle = camera_angle_deg(group, seq_a, seq_b)
            if angle < min_angle:
                continue
            for start in windows[split]:
                if has_clip(group_dir, seq_a, seq_b, start):
                    records.append(
                        {
                            "group": group,
                            "seqA": f"{group}/{seq_a}",
                            "seqB": f"{group}/{seq_b}",
                            "start_frame": int(start),
                            "view_angle_deg": round(float(angle), 6),
                            "angle_bucket": angle_bucket(angle),
                        }
                    )
    return records


def sample_no_reused_frames(records: list[dict], total: int, rng: random.Random) -> list[dict]:
    strata = defaultdict(list)
    for record in records:
        strata[(record["group"], record["angle_bucket"])].append(record)
    keys = sorted(strata)
    if not keys:
        raise ValueError("No candidates available")
    counts = {key: total // len(keys) for key in keys}
    for key in keys[: total % len(keys)]:
        counts[key] += 1

    selected = []
    used_frames: set[tuple[str, str, int]] = set()
    for key in keys:
        pool = strata[key][:]
        rng.shuffle(pool)
        need = counts[key]
        picked = 0
        for record in pool:
            group = record["group"]
            seq_a = record["seqA"].split("/", 1)[1]
            seq_b = record["seqB"].split("/", 1)[1]
            frames = clip_frames(group, seq_a, seq_b, int(record["start_frame"]))
            if frames & used_frames:
                continue
            selected.append(record)
            used_frames.update(frames)
            picked += 1
            if picked == need:
                break
        if picked < need:
            raise RuntimeError(f"Only picked {picked}/{need} for stratum {key}; reduce clip count or angle threshold")
    rng.shuffle(selected)
    return selected


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def summarize(records: list[dict]) -> dict:
    by_group = defaultdict(int)
    by_bucket = defaultdict(int)
    used_frames = set()
    angles = []
    for record in records:
        by_group[record["group"]] += 1
        by_bucket[record["angle_bucket"]] += 1
        seq_a = record["seqA"].split("/", 1)[1]
        seq_b = record["seqB"].split("/", 1)[1]
        used_frames.update(clip_frames(record["group"], seq_a, seq_b, int(record["start_frame"])))
        angles.append(float(record["view_angle_deg"]))
    return {
        "num_clips": len(records),
        "num_used_frames": len(used_frames),
        "has_reused_frames": len(used_frames) != len(records) * 4,
        "angle_min": min(angles) if angles else None,
        "angle_max": max(angles) if angles else None,
        "group_counts": dict(sorted(by_group.items())),
        "angle_bucket_counts": dict(sorted(by_bucket.items())),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} already exists; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    outputs = {}
    for split, total in [("train", args.train_clips), ("val", args.val_clips), ("test", args.test_clips)]:
        candidates = []
        for group in args.groups:
            candidates.extend(candidates_for_group(args.training_root, group, split, args.min_angle))
        outputs[f"round1_{split}.jsonl"] = sample_no_reused_frames(candidates, total, rng)

    for name, records in outputs.items():
        write_jsonl(args.output_dir / name, records)

    metadata = {
        "training_root": str(args.training_root),
        "raw_roots": RAW_ROOTS,
        "seed": args.seed,
        "min_angle": args.min_angle,
        "groups": args.groups,
        "manifests": {name: summarize(records) for name, records in outputs.items()},
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata["manifests"], indent=2, sort_keys=True))
    print(f"Wrote manifests to {args.output_dir}")


if __name__ == "__main__":
    main()
