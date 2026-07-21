#!/usr/bin/env python3
"""Align holdout records to the cases that completed V10 successfully."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--merged_cases", type=Path, required=True)
    parser.add_argument("--output_records", type=Path, required=True)
    parser.add_argument("--output_audit", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.records)
    merged = json.loads(args.merged_cases.read_text(encoding="utf-8"))

    record_by_id = {record["pattern_id"]: record for record in records}
    successful_ids = [case["case_name"] for case in merged["cases"]]
    missing_from_records = [case_id for case_id in successful_ids if case_id not in record_by_id]
    if missing_from_records:
        raise RuntimeError(f"Merged V10 cases are absent from the record file: {missing_from_records}")

    aligned_records = [record_by_id[case_id] for case_id in successful_ids]
    successful_id_set = set(successful_ids)
    excluded_records = [record for record in records if record["pattern_id"] not in successful_id_set]

    args.output_records.parent.mkdir(parents=True, exist_ok=True)
    with args.output_records.open("w", encoding="utf-8") as handle:
        for record in aligned_records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    audit = {
        "input_record_count": len(records),
        "successful_v10_case_count": len(aligned_records),
        "excluded_record_count": len(excluded_records),
        "excluded_records": excluded_records,
        "v10_failures": merged.get("failures", []),
        "policy": "Cases without a valid post-reset human are excluded from torso-rule evaluation and must use the Fixed Explicit fallback at runtime.",
    }
    args.output_audit.write_text(json.dumps(audit, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print(json.dumps({
        "output_records": str(args.output_records),
        "valid_count": len(aligned_records),
        "excluded_count": len(excluded_records),
    }))


if __name__ == "__main__":
    main()
