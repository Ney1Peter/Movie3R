#!/usr/bin/env python3
"""Freeze a capture-disjoint untouched holdout for the V14.5 final audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_MAP = (
    ROOT
    / "config/manifests/v10_oracle_candidate_selection_gt_sources/manifest_map.json"
)
DEFAULT_ORIGINAL = (
    ROOT / "output/v10_candidate_selection/oracle_gt_4source/selected_records.jsonl"
)
DEFAULT_BAD = ROOT / "config/manifests/v10_static_alignment_bad_samples.jsonl"
DEFAULT_HISTORY = (
    ROOT / "output/archive/20260721/v25_holdout_rotation_validation/records/holdout_records.jsonl",
    ROOT / "output/archive/20260721/v27_consensus_holdout2/records/holdout_records.jsonl",
    ROOT / "output/archive/20260721/v29_rotation_rule_holdout3/records/holdout_records.jsonl",
    ROOT / "output/archive/20260721/v31_metric_fit_holdout4/records/holdout_records.jsonl",
    ROOT / "output/archive/20260721/v32_texture_safe_holdout5/records/holdout_records.jsonl",
    ROOT / "output/archive/20260721/v36_human_jump_holdout6/records/holdout_records.jsonl",
    ROOT / "output/archive/20260721/v37_human_jump_holdout7/records/holdout_records.jsonl",
    ROOT
    / "output/archive/20260721/v37_human_jump_holdout7/records/holdout_records_v10_valid.jsonl",
)
DEFAULT_OUTPUT = ROOT / "output/v14_5_final_audit/untouched_holdout/records"
ANGLE_BUCKETS = ("060_090", "090_120", "120_150", "150_180")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest_map", type=Path, default=DEFAULT_MANIFEST_MAP)
    parser.add_argument("--original_records", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument(
        "--historical_records", type=Path, nargs="*", default=DEFAULT_HISTORY
    )
    parser.add_argument("--bad_sample_registry", type=Path, default=DEFAULT_BAD)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples_per_bucket", type=int, default=4)
    parser.add_argument("--seed", type=int, default=14520260722)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def manifest_map(path: Path) -> dict[str, Path]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(source): Path(value)
        for source, value in payload["source_manifests"].items()
    }


def capture_key(record: dict) -> tuple[str, str, int]:
    return (
        str(record.get("source", "")),
        str(record.get("group", "")),
        int(record.get("start_frame", -1)),
    )


def case_key(record: dict) -> str:
    if record.get("pattern_id"):
        return str(record["pattern_id"])
    seq_a = str(record.get("seqA", "")).split("/")[-1]
    seq_b = str(record.get("seqB", "")).split("/")[-1]
    return (
        f"{record.get('source', '')}_{record.get('angle_bucket', 'unknown')}_"
        f"{record.get('group', 'group')}_{record.get('start_frame', -1)}_"
        f"{seq_a}_{seq_b}"
    )


def bad_key(record: dict) -> tuple[str, str, str, str, int]:
    return (
        str(record.get("source", "")),
        str(record.get("group", "")),
        str(record.get("seqA", "")),
        str(record.get("seqB", "")),
        int(record.get("start_frame", -1)),
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    required = [
        args.manifest_map,
        args.original_records,
        args.bad_sample_registry,
        *args.historical_records,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen split inputs: {missing}")

    maps = manifest_map(args.manifest_map)
    original = read_jsonl(args.original_records)
    historical = [row for path in args.historical_records for row in read_jsonl(path)]
    excluded_rows = original + historical
    excluded_captures = {capture_key(row) for row in excluded_rows}
    excluded_cases = {case_key(row) for row in excluded_rows}
    bad_samples = {bad_key(row) for row in read_jsonl(args.bad_sample_registry)}
    rng = np.random.default_rng(int(args.seed))

    selected: list[dict] = []
    available = Counter()
    for source, path in sorted(maps.items()):
        by_bucket: dict[str, list[dict]] = defaultdict(list)
        for index, raw in enumerate(read_jsonl(path)):
            record = dict(raw)
            record["source"] = source
            record["source_manifest_index"] = index
            record["pattern_id"] = case_key(record)
            if capture_key(record) in excluded_captures:
                continue
            if case_key(record) in excluded_cases or bad_key(record) in bad_samples:
                continue
            bucket = str(record.get("angle_bucket", "unknown"))
            by_bucket[bucket].append(record)
            available[(source, bucket)] += 1

        for bucket in ANGLE_BUCKETS:
            pool = sorted(by_bucket.get(bucket, []), key=case_key)
            if not pool:
                continue
            order = rng.permutation(len(pool))
            for index in order[: min(int(args.samples_per_bucket), len(pool))]:
                selected.append(dict(pool[int(index)]))

    selected.sort(
        key=lambda row: (
            str(row["source"]),
            str(row.get("angle_bucket", "")),
            case_key(row),
        )
    )
    selected_captures = {capture_key(row) for row in selected}
    selected_cases = {case_key(row) for row in selected}
    if len(selected_captures) != len(selected) or len(selected_cases) != len(selected):
        raise RuntimeError("Selected holdout is not unique at capture and case level")
    if selected_captures & excluded_captures or selected_cases & excluded_cases:
        raise RuntimeError("Untouched holdout overlaps a historical set")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_path = args.output_dir / "holdout_records.jsonl"
    write_jsonl(all_path, selected)
    source_paths = {}
    for source in sorted(maps):
        source_path = args.output_dir / f"{source}.jsonl"
        write_jsonl(source_path, [row for row in selected if row["source"] == source])
        source_paths[source] = str(source_path)
    (args.output_dir / "manifest_map.json").write_text(
        json.dumps({"source_manifests": source_paths}, indent=2) + "\n",
        encoding="utf-8",
    )

    inputs = [
        args.manifest_map,
        args.original_records,
        args.bad_sample_registry,
        *args.historical_records,
        *maps.values(),
    ]
    summary = {
        "experiment": "V14.5 capture-disjoint untouched holdout freeze",
        "selection_frozen_before_inference": True,
        "seed": int(args.seed),
        "samples_per_available_source_bucket": int(args.samples_per_bucket),
        "case_count": len(selected),
        "capture_count": len(selected_captures),
        "historical_case_count": len(excluded_cases),
        "historical_capture_count": len(excluded_captures),
        "case_overlap_with_history": len(selected_cases & excluded_cases),
        "capture_overlap_with_history": len(selected_captures & excluded_captures),
        "selection_sha256": sha256(all_path),
        "input_sha256": {str(path): sha256(path) for path in inputs},
        "by_source": dict(Counter(row["source"] for row in selected)),
        "by_source_bucket": {
            f"{source}:{bucket}": sum(
                row["source"] == source and row.get("angle_bucket") == bucket
                for row in selected
            )
            for source in sorted(maps)
            for bucket in ANGLE_BUCKETS
        },
        "available_after_exclusion": {
            f"{source}:{bucket}": available[(source, bucket)]
            for source in sorted(maps)
            for bucket in ANGLE_BUCKETS
        },
        "historical_record_files": [str(path) for path in args.historical_records],
        "capture_key": ["source", "group", "start_frame"],
    }
    (args.output_dir / "freeze_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
