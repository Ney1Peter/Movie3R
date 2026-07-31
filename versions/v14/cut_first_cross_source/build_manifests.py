#!/usr/bin/env python3
"""Build deterministic, camera-pair-disjoint V14 cut-event manifests.

The source manifests contain AABB clips.  V14 uses only the last two A views
and the first B view so that supervision is attached exclusively to the first
post-cut frame:

    A(t-1), A(t), B(t) with shot_labels [0, 0, 1]

The frozen ten-event camera pairs are excluded in both directions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "manifests"
FROZEN_ROOT = REPO_ROOT / "config/manifests/v14_1_cut_event/ten"
SOURCE_MANIFESTS = {
    "avatarrex": REPO_ROOT
    / "config/manifests/v9_4source_baseline_avatarrex_lbn1_lbn2_zzr_angle60_manifests/train_aabb_60k.jsonl",
    "thuman": REPO_ROOT
    / "config/manifests/v9_4source_baseline_thuman00_02_angle60_manifests/train_aabb_60k.jsonl",
    "mvhuman100": REPO_ROOT
    / "config/manifests/v9_4source_baseline_mvhuman100_angle60_manifests/train_aabb_60k.jsonl",
    "mvhuman200": REPO_ROOT
    / "config/manifests/v9_4source_baseline_mvhuman200_angle60_manifests/train_aabb_60k.jsonl",
}
FROZEN_FILES = {
    "avatarrex": "avatarrex.jsonl",
    "thuman": "thuman.jsonl",
    "mvhuman100": "mvhuman100.jsonl",
    "mvhuman200": "mvhuman200.jsonl",
}
STAGES = {
    "train10": {
        "avatarrex": 3,
        "thuman": 2,
        "mvhuman100": 3,
        "mvhuman200": 2,
    },
    "train24ps": {source: 24 for source in SOURCE_MANIFESTS},
    "train96ps": {source: 96 for source in SOURCE_MANIFESTS},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260801)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n"
        for record in records
    )
    path.write_text(payload, encoding="utf-8")


def canonical_pair(first: str, second: str) -> tuple[str, str]:
    return tuple(sorted((str(first), str(second))))


def frozen_pairs() -> dict[str, set[tuple[str, str]]]:
    output: dict[str, set[tuple[str, str]]] = {}
    for source, filename in FROZEN_FILES.items():
        records = read_jsonl(FROZEN_ROOT / filename)
        output[source] = {
            canonical_pair(record["seqs"][-2], record["seqs"][-1])
            for record in records
        }
    return output


def stable_rank(record: dict[str, Any], source: str, seed: int) -> str:
    identity = "|".join(
        (
            str(seed),
            source,
            str(record.get("group", "")),
            str(record["seqA"]),
            str(record["seqB"]),
            str(record["start_frame"]),
        )
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def to_cut_event(record: dict[str, Any], source: str) -> dict[str, Any]:
    frame = int(record["start_frame"])
    if frame < 1:
        raise ValueError(f"Cannot create t-1 context for frame {frame}")
    seq_a = str(record["seqA"])
    seq_b = str(record["seqB"])
    pair_name = f"{seq_a.split('/')[-1]}_{seq_b.split('/')[-1]}"
    return {
        "angle_bucket": str(record.get("angle_bucket", "unknown")),
        "clip_type": "cut_event",
        "frames": [frame - 1, frame, frame],
        "group": str(record.get("group", "")),
        "pattern_id": f"v14_cross_{source}_{record.get('group', '')}_{pair_name}_{frame}",
        "seqs": [seq_a, seq_a, seq_b],
        "shot_labels": [0, 0, 1],
        "source": source,
        "source_manifest_index": int(record.get("source_manifest_index", -1)),
        "transition_angles_deg": [
            0.0,
            0.0,
            float(record.get("view_angle_deg", 0.0)),
        ],
        "view_angle_deg": float(record.get("view_angle_deg", 0.0)),
    }


def ordered_candidates(
    source: str,
    seed: int,
    excluded: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    raw = read_jsonl(SOURCE_MANIFESTS[source])
    valid = []
    seen_events = set()
    for index, record in enumerate(raw):
        if int(record["start_frame"]) < 1:
            continue
        pair = canonical_pair(record["seqA"], record["seqB"])
        if pair in excluded:
            continue
        event = (pair, int(record["start_frame"]))
        if event in seen_events:
            continue
        seen_events.add(event)
        item = dict(record)
        item["source_manifest_index"] = index
        valid.append(item)

    # Interleave angle buckets.  The stable hash makes the selection exactly
    # reproducible without inheriting ordering bias from generated manifests.
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in valid:
        buckets[str(record.get("angle_bucket", "unknown"))].append(record)
    for records in buckets.values():
        records.sort(key=lambda row: stable_rank(row, source, seed))

    ordered = []
    names = sorted(buckets)
    offset = 0
    while True:
        appended = False
        for name in names:
            if offset < len(buckets[name]):
                ordered.append(buckets[name][offset])
                appended = True
        if not appended:
            break
        offset += 1

    # Prefer camera-pair diversity before taking additional timestamps from a
    # pair.  This is useful for 10/24-per-source stages while still allowing
    # the 96-per-source stage to draw enough clips.
    unique_pair, repeated_pair = [], []
    seen_pairs = set()
    for record in ordered:
        pair = canonical_pair(record["seqA"], record["seqB"])
        if pair in seen_pairs:
            repeated_pair.append(record)
        else:
            seen_pairs.add(pair)
            unique_pair.append(record)
    return unique_pair + repeated_pair


def audit_stage(
    stage: str,
    records_by_source: dict[str, list[dict[str, Any]]],
    excluded: dict[str, set[tuple[str, str]]],
) -> dict[str, Any]:
    report: dict[str, Any] = {"stage": stage, "sources": {}}
    for source, records in records_by_source.items():
        pairs = [canonical_pair(row["seqs"][-2], row["seqs"][-1]) for row in records]
        events = [(pair, int(row["frames"][-1])) for pair, row in zip(pairs, records)]
        overlap = sorted(set(pairs) & excluded[source])
        if overlap:
            raise RuntimeError(f"{stage}/{source} leaks frozen pairs: {overlap}")
        if len(events) != len(set(events)):
            raise RuntimeError(f"{stage}/{source} contains duplicate cut events")
        if any(row["shot_labels"] != [0, 0, 1] for row in records):
            raise RuntimeError(f"{stage}/{source} has invalid shot labels")
        report["sources"][source] = {
            "records": len(records),
            "unique_unordered_camera_pairs": len(set(pairs)),
            "groups": dict(sorted(Counter(row["group"] for row in records).items())),
            "angle_buckets": dict(
                sorted(Counter(row["angle_bucket"] for row in records).items())
            ),
            "min_angle_deg": min(row["view_angle_deg"] for row in records),
            "max_angle_deg": max(row["view_angle_deg"] for row in records),
            "frozen_pair_overlap": 0,
        }
    report["total_records"] = sum(len(rows) for rows in records_by_source.values())
    return report


def main() -> None:
    args = parse_args()
    excluded = frozen_pairs()
    candidates = {
        source: ordered_candidates(source, args.seed, excluded[source])
        for source in SOURCE_MANIFESTS
    }
    max_required = max(max(counts.values()) for counts in STAGES.values())
    for source, records in candidates.items():
        if len(records) < max_required:
            raise RuntimeError(
                f"Only {len(records)} eligible records for {source}; need {max_required}"
            )

    metadata = {
        "seed": args.seed,
        "source_manifests": {key: str(value) for key, value in SOURCE_MANIFESTS.items()},
        "frozen_eval_root": str(FROZEN_ROOT),
        "pair_disjoint_rule": "unordered (seqA, seqB), source-local",
        "stages": {},
    }
    for stage, counts in STAGES.items():
        records_by_source = {
            source: [to_cut_event(row, source) for row in candidates[source][:count]]
            for source, count in counts.items()
        }
        for source, records in records_by_source.items():
            write_jsonl(args.output_dir / stage / f"{source}.jsonl", records)
        metadata["stages"][stage] = audit_stage(stage, records_by_source, excluded)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "metadata.json"
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(path)
    for stage, report in metadata["stages"].items():
        print(f"{stage}: {report['total_records']} records")
        for source, source_report in report["sources"].items():
            print(
                f"  {source}: n={source_report['records']} "
                f"pairs={source_report['unique_unordered_camera_pairs']} "
                f"angle={source_report['min_angle_deg']:.1f}-"
                f"{source_report['max_angle_deg']:.1f}"
            )


if __name__ == "__main__":
    main()
