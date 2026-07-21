#!/usr/bin/env python3
"""Merge per-source V10 holdout outputs into one candidate-root namespace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "output" / "v25_holdout_rotation_validation" / "v10"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "v25_holdout_rotation_validation" / "v10_merged"
SOURCES = ("avatarrex", "thuman", "mvhuman100", "mvhuman200")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases_dir = args.output_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []
    source_counts = {}
    for source in SOURCES:
        source_root = args.input_root / source
        report_path = source_root / "oracle_candidate_selection_metrics.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        source_counts[source] = len(report["cases"])
        rows.extend(report["cases"])
        failures.extend(report.get("failures", []))
        for source_case in sorted((source_root / "cases").iterdir()):
            target = cases_dir / source_case.name
            if target.is_symlink():
                if target.resolve() != source_case.resolve():
                    raise RuntimeError(f"Conflicting link for {target.name}")
                continue
            if target.exists():
                raise RuntimeError(f"Unexpected existing path: {target}")
            target.symlink_to(source_case.resolve(), target_is_directory=True)

    names = [row["case_name"] for row in rows]
    expected = sum(source_counts.values())
    if len(names) != expected or len(names) != len(set(names)):
        raise RuntimeError("Merged V10 cases are incomplete or duplicated")
    payload = {
        "experiment": "V25 disjoint holdout merged V10 cache",
        "case_count": len(rows),
        "source_counts": source_counts,
        "failure_count": len(failures),
        "cases": rows,
        "failures": failures,
    }
    output = args.output_dir / "merged_cases.json"
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: payload[key] for key in ("case_count", "source_counts", "failure_count")}, indent=2))


if __name__ == "__main__":
    main()
