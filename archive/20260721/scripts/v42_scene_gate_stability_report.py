#!/usr/bin/env python3
"""Aggregate repeated V41 scene-gate audits across seeds and sample counts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(args.root.glob("*/v41_background_scale_scene_safety_audit.json"))
    if not paths:
        raise RuntimeError(f"No V41 reports under {args.root}")
    runs = {}
    frequency: Counter[str] = Counter()
    row_by_case = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        selected = []
        harmful = []
        improved = []
        for row in payload["rows"]:
            row_by_case[row["case_name"]] = row
            if row["scene_delta_m"] < -float(args.threshold):
                selected.append(row["case_name"])
                frequency[row["case_name"]] += 1
                if row["camera_delta_m"] > 0.05:
                    harmful.append(row["case_name"])
                if row["camera_delta_m"] < -0.05:
                    improved.append(row["case_name"])
        runs[path.parent.name] = {
            "selected": sorted(selected),
            "selected_count": len(selected),
            "improved_005m": sorted(improved),
            "harmful_005m": sorted(harmful),
        }
    selected_sets = [set(run["selected"]) for run in runs.values()]
    intersection = set.intersection(*selected_sets)
    union = set.union(*selected_sets)
    report = {
        "experiment": "V42 scene-gated background-scale stability report",
        "threshold_m": float(args.threshold),
        "run_count": len(runs),
        "runs": runs,
        "intersection": sorted(intersection),
        "intersection_count": len(intersection),
        "union": sorted(union),
        "union_count": len(union),
        "selection_frequency": dict(sorted(frequency.items(), key=lambda item: (-item[1], item[0]))),
        "all_runs_harmful_005m_count": sum(len(run["harmful_005m"]) for run in runs.values()),
        "stable_cases": [
            {
                "case_name": name,
                "source": row_by_case[name]["source"],
                "camera_delta_m": row_by_case[name]["camera_delta_m"],
                "scene_delta_m_last_run": row_by_case[name]["scene_delta_m"],
            }
            for name in sorted(intersection)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
