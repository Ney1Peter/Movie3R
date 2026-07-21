#!/usr/bin/env python3
"""Build a disjoint 120-cut holdout for V24/V25 rotation-rule validation."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from v10_oracle_candidate_selection_probe import (  # noqa: E402
    ANGLE_BUCKETS,
    aabb_tuple_from_any_record,
    bad_sample_key,
    load_bad_sample_keys,
    load_manifest_map,
    read_jsonl,
)


DEFAULT_MANIFEST_MAP = (
    REPO_ROOT
    / "config"
    / "manifests"
    / "v10_oracle_candidate_selection_gt_sources"
    / "manifest_map.json"
)
DEFAULT_SELECTED = (
    REPO_ROOT
    / "output"
    / "v10_candidate_selection"
    / "oracle_gt_4source"
    / "selected_records.jsonl"
)
DEFAULT_BAD = REPO_ROOT / "config" / "manifests" / "v10_static_alignment_bad_samples.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "v25_holdout_rotation_validation" / "records"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest_map", type=Path, default=DEFAULT_MANIFEST_MAP)
    parser.add_argument("--selected_records", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument(
        "--additional_exclude_records",
        type=Path,
        nargs="*",
        default=(),
        help="Additional JSONL record sets that must remain disjoint from this holdout.",
    )
    parser.add_argument("--bad_sample_registry", type=Path, default=DEFAULT_BAD)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples_per_bucket", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


def record_key(record: dict) -> tuple[str, str, int]:
    seq_a, seq_b, start = aabb_tuple_from_any_record(record)
    return str(seq_a), str(seq_b), int(start)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_map = load_manifest_map(args.manifest_map)
    excluded_paths = [args.selected_records, *args.additional_exclude_records]
    existing = {
        record_key(row)
        for path in excluded_paths
        for row in read_jsonl(path)
    }
    bad_keys = load_bad_sample_keys(args.bad_sample_registry)
    rng = np.random.default_rng(int(args.seed))
    selected = []
    source_files = {}

    for source, manifest in manifest_map.items():
        by_bucket: dict[str, list[dict]] = defaultdict(list)
        for manifest_index, raw in enumerate(read_jsonl(manifest)):
            record = dict(raw)
            record["source"] = source
            record["source_manifest_index"] = manifest_index
            if record_key(record) in existing or bad_sample_key(record) in bad_keys:
                continue
            by_bucket[str(record.get("angle_bucket", "unknown"))].append(record)

        source_rows = []
        for bucket in ANGLE_BUCKETS:
            pool = by_bucket.get(bucket, [])
            if not pool:
                continue
            order = rng.permutation(len(pool)).tolist()
            take = min(int(args.samples_per_bucket), len(order))
            for local_index in order[:take]:
                record = dict(pool[local_index])
                seq_a, seq_b, start = aabb_tuple_from_any_record(record)
                record.setdefault(
                    "pattern_id",
                    f"{source}_{bucket}_{record.get('group', 'group')}_{start}_{seq_a.split('/')[-1]}_{seq_b.split('/')[-1]}",
                )
                source_rows.append(record)
                selected.append(record)

        source_rows.sort(
            key=lambda row: (
                str(row.get("angle_bucket", "")),
                str(row["pattern_id"]),
            )
        )
        source_path = args.output_dir / f"{source}.jsonl"
        source_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in source_rows),
            encoding="utf-8",
        )
        source_files[source] = str(source_path)

    selected.sort(
        key=lambda row: (
            str(row["source"]),
            str(row.get("angle_bucket", "")),
            str(row["pattern_id"]),
        )
    )
    if len(selected) != len({record_key(row) for row in selected}):
        raise RuntimeError("Holdout records are not unique")
    if existing & {record_key(row) for row in selected}:
        raise RuntimeError("Holdout overlaps the original 180 records")
    records_path = args.output_dir / "holdout_records.jsonl"
    records_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
    )
    map_path = args.output_dir / "manifest_map.json"
    map_path.write_text(
        json.dumps({"source_manifests": source_files}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary = {
        "experiment": "V25 disjoint rotation holdout records",
        "case_count": len(selected),
        "samples_per_available_source_bucket": int(args.samples_per_bucket),
        "seed": int(args.seed),
        "overlap_with_original_180": 0,
        "overlap_with_all_excluded_sets": 0,
        "excluded_record_sets": [str(path) for path in excluded_paths],
        "by_source": {
            source: sum(row["source"] == source for row in selected)
            for source in sorted(source_files)
        },
        "by_source_bucket": {
            f"{source}:{bucket}": sum(
                row["source"] == source and row.get("angle_bucket") == bucket
                for row in selected
            )
            for source in sorted(source_files)
            for bucket in ANGLE_BUCKETS
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
