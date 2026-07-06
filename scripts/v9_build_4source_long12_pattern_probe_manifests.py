#!/usr/bin/env python3
"""Build tiny 12-frame four-source explicit-pattern overfit manifests."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np


PATTERNS = {
    "same12": tuple("AAAAAAAAAAAA"),
    "block3": tuple("AAAABBBBCCCC"),
    "block4": tuple("AAABBBCCCDDD"),
    "alternate2": tuple("ABABABABABAB"),
    "repeat_aabb": tuple("AABBAABBAABB"),
    "cycle4": tuple("ABCDABCDABCD"),
}

SOURCE_SPECS = {
    "avatarrex": {
        "manifest_root": "config/manifests/v9_4source_baseline_avatarrex_lbn1_lbn2_zzr_angle60_manifests",
        "split_root": "/data/wangzheng/iJCV-CODE/data/Training",
    },
    "thuman": {
        "manifest_root": "config/manifests/v9_4source_baseline_thuman00_02_angle60_manifests",
        "split_root": "/data/wangzheng/iJCV-CODE/data/Training",
    },
    "mvhuman100": {
        "manifest_root": "config/manifests/v9_4source_baseline_mvhuman100_angle60_manifests",
        "split_root": "/data/wangzheng/iJCV-CODE/data/Training/mvhuman",
    },
    "mvhuman200": {
        "manifest_root": "config/manifests/v9_4source_baseline_mvhuman200_angle60_manifests",
        "split_root": "/data/wangzheng/iJCV-CODE/data/Training/mvhuman",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/Movie3R"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("config/manifests/v9_4source_long12_pattern_probe"),
    )
    parser.add_argument("--per-pattern-per-source", type=int, default=1)
    parser.add_argument("--num-frames", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def has_frame(split_root: Path, seq: str, frame: int) -> bool:
    seq_root = split_root / seq
    stem = f"{int(frame):08d}"
    return (
        (seq_root / "rgb" / f"{stem}.png").is_file()
        and (seq_root / "cam" / f"{stem}.npz").is_file()
        and (seq_root / "smpl" / f"{stem}.pkl").is_file()
    )


def has_all_frames(split_root: Path, seq: str, start: int, num_frames: int) -> bool:
    return all(has_frame(split_root, seq, start + offset) for offset in range(num_frames))


def load_pose(split_root: Path, seq: str, frame: int) -> np.ndarray:
    return np.load(split_root / seq / "cam" / f"{int(frame):08d}.npz")["pose"].astype(np.float64)


def camera_angle_deg(pose_a: np.ndarray, pose_b: np.ndarray) -> float:
    dir_a = pose_a[:3, 2]
    dir_b = pose_b[:3, 2]
    dir_a = dir_a / max(np.linalg.norm(dir_a), 1e-12)
    dir_b = dir_b / max(np.linalg.norm(dir_b), 1e-12)
    return math.degrees(math.acos(float(np.clip(np.dot(dir_a, dir_b), -1.0, 1.0))))


def angle_bucket(angle: float) -> str:
    if angle <= 1e-6:
        return "same_camera"
    if angle < 90.0:
        return "060_090"
    if angle < 120.0:
        return "090_120"
    if angle < 150.0:
        return "120_150"
    return "150_180"


def collect_group_sequences(records: list[dict]) -> dict[str, list[str]]:
    seqs_by_group = defaultdict(set)
    for record in records:
        group = str(record["group"])
        seqs_by_group[group].add(str(record["seqA"]))
        seqs_by_group[group].add(str(record["seqB"]))
    return {group: sorted(seqs) for group, seqs in seqs_by_group.items()}


def shot_labels(seqs: list[str]) -> list[int]:
    labels = [0]
    for idx in range(1, len(seqs)):
        labels.append(0 if seqs[idx] == seqs[idx - 1] else 1)
    return labels


def transition_angles(split_root: Path, seqs: list[str], frames: list[int]) -> list[float]:
    angles = [0.0]
    for idx in range(1, len(seqs)):
        if seqs[idx] == seqs[idx - 1]:
            angles.append(0.0)
            continue
        angle = camera_angle_deg(
            load_pose(split_root, seqs[idx - 1], frames[idx - 1]),
            load_pose(split_root, seqs[idx], frames[idx]),
        )
        angles.append(round(float(angle), 6))
    return angles


def build_source_records(source: str, spec: dict, repo_root: Path, per_pattern: int, num_frames: int) -> dict[str, list[dict]]:
    manifest_root = repo_root / spec["manifest_root"]
    split_root = Path(spec["split_root"])
    aabb_records = read_jsonl(manifest_root / "train_aabb_60k.jsonl")
    seqs_by_group = collect_group_sequences(aabb_records)
    pattern_records = {name: [] for name in PATTERNS}
    used_case_keys = set()

    for base in aabb_records:
        if all(len(records) >= per_pattern for records in pattern_records.values()):
            break
        group = str(base["group"])
        start = int(base["start_frame"])
        seq_a = str(base["seqA"])
        seq_b = str(base["seqB"])
        if not has_all_frames(split_root, seq_a, start, num_frames):
            continue
        if not has_all_frames(split_root, seq_b, start, num_frames):
            continue

        extras = [
            seq
            for seq in seqs_by_group[group]
            if seq not in {seq_a, seq_b}
            and has_all_frames(split_root, seq, start, num_frames)
        ]
        if len(extras) < 2:
            continue

        seq_map = {"A": seq_a, "B": seq_b, "C": extras[0], "D": extras[1]}
        case_key = (group, seq_a, seq_b, extras[0], extras[1], start)
        if case_key in used_case_keys:
            continue

        frames = [start + offset for offset in range(num_frames)]
        candidate_records = {}
        for pattern_name, letters in PATTERNS.items():
            if len(letters) != num_frames:
                raise ValueError(f"Pattern {pattern_name} length {len(letters)} != num_frames {num_frames}")
            seqs = [seq_map[letter] for letter in letters]
            trans_angles = transition_angles(split_root, seqs, frames)
            max_angle = max(trans_angles)
            candidate_records[pattern_name] = {
                "angle_bucket": angle_bucket(max_angle),
                "clip_type": pattern_name,
                "group": group,
                "pattern_id": f"{source}_{group}_{start}_{len(pattern_records[pattern_name])}",
                "seqs": seqs,
                "frames": frames,
                "shot_labels": shot_labels(seqs),
                "transition_angles_deg": trans_angles,
                "view_angle_deg": round(float(max_angle), 6),
            }

        used_case_keys.add(case_key)
        for pattern_name, record in candidate_records.items():
            if len(pattern_records[pattern_name]) < per_pattern:
                pattern_records[pattern_name].append(record)

    missing = {name: per_pattern - len(records) for name, records in pattern_records.items() if len(records) < per_pattern}
    if missing:
        raise RuntimeError(f"{source} did not have enough long12 pattern records: {missing}")
    return pattern_records


def summarize(records_by_source: dict[str, dict[str, list[dict]]], num_frames: int) -> dict:
    summary = {"num_frames": num_frames, "sources": {}, "total_records": 0}
    for source, by_pattern in records_by_source.items():
        source_total = sum(len(records) for records in by_pattern.values())
        summary["sources"][source] = {
            "total_records": source_total,
            "pattern_counts": {name: len(records) for name, records in sorted(by_pattern.items())},
            "angle_ranges": {
                name: [
                    min(record["view_angle_deg"] for record in records),
                    max(record["view_angle_deg"] for record in records),
                ]
                for name, records in sorted(by_pattern.items())
            },
        }
        summary["total_records"] += source_total
    return summary


def main() -> None:
    args = parse_args()
    output_dir = args.repo_root / args.output_dir
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} already exists; pass --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records_by_source = {}
    for source, spec in SOURCE_SPECS.items():
        records_by_source[source] = build_source_records(
            source,
            spec,
            args.repo_root,
            args.per_pattern_per_source,
            args.num_frames,
        )
        source_all = []
        for pattern_name, records in sorted(records_by_source[source].items()):
            write_jsonl(output_dir / source / f"train_{pattern_name}.jsonl", records)
            source_all.extend(records)
        write_jsonl(output_dir / source / "train_all_patterns.jsonl", source_all)

    metadata = summarize(records_by_source, args.num_frames)
    metadata.update(
        {
            "patterns": {key: "".join(value) for key, value in PATTERNS.items()},
            "source_specs": SOURCE_SPECS,
            "per_pattern_per_source": args.per_pattern_per_source,
        }
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
