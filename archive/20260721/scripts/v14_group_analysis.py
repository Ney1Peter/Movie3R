#!/usr/bin/env python3
"""Build V14 texture, pseudo-overlap, geometry and angle group summaries."""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate_dir",
        type=Path,
        default=REPO_ROOT / "output" / "v14_selective_world_memory" / "candidate_cache",
    )
    parser.add_argument(
        "--eval_json",
        type=Path,
        default=REPO_ROOT / "output" / "v14_selective_world_memory" / "evaluation" / "v14_eval.json",
    )
    parser.add_argument(
        "--v13_dir",
        type=Path,
        default=REPO_ROOT / "output" / "v13_world_coordinate_memory" / "stage1_scene_coordinate_oracle",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "output" / "v14_selective_world_memory" / "evaluation" / "v14_groups.json",
    )
    return parser.parse_args()


def load_cases(pattern: str) -> list[dict]:
    cases = []
    for path in sorted(glob.glob(pattern)):
        cases.extend(json.loads(Path(path).read_text(encoding="utf-8"))["cases"])
    return cases


def failed(row: dict | None) -> bool:
    return row is None or bool(row.get("fit_failed", False))


def catastrophic(row: dict | None) -> bool:
    return failed(row) or float(row["camera_translation_error_m"]) > 1.0 or float(row["camera_rotation_error_deg"]) > 30.0


def cost(row: dict | None) -> float:
    if failed(row):
        return 100.0
    return float(row["camera_translation_error_m"]) / 0.25 + float(row["camera_rotation_error_deg"]) / 5.0


def oracle(options: list[dict | None]) -> dict:
    return min(options, key=lambda row: (catastrophic(row), cost(row)))


def stats(rows: list[dict], method: str) -> dict:
    values = [row[method] for row in rows]
    valid = [row for row in values if not failed(row)]
    translation = np.asarray([row["camera_translation_error_m"] for row in valid], dtype=np.float64)
    rotation = np.asarray([row["camera_rotation_error_deg"] for row in valid], dtype=np.float64)
    return {
        "count": len(values),
        "translation_m": float(translation.mean()) if valid else None,
        "translation_median_m": float(np.median(translation)) if valid else None,
        "translation_p90_m": float(np.percentile(translation, 90)) if valid else None,
        "translation_p95_m": float(np.percentile(translation, 95)) if valid else None,
        "rotation_deg": float(rotation.mean()) if valid else None,
        "rotation_median_deg": float(np.median(rotation)) if valid else None,
        "rotation_p90_deg": float(np.percentile(rotation, 90)) if valid else None,
        "rotation_p95_deg": float(np.percentile(rotation, 95)) if valid else None,
        "catastrophic_rate": float(np.mean([catastrophic(row) for row in values])),
        "success_rate": float(
            np.mean(
                [
                    not failed(row)
                    and float(row["camera_translation_error_m"]) < 0.25
                    and float(row["camera_rotation_error_deg"]) < 5.0
                    for row in values
                ]
            )
        ),
        "improvement_vs_fixed_joint": float(np.mean([cost(row["fixed"]) - cost(row[method]) for row in rows])),
    }


def main() -> None:
    args = parse_args()
    candidates = load_cases(str(args.candidate_dir / "v14_candidates_shard_*_of_*.json"))
    v13_cases = load_cases(str(args.v13_dir / "stage1_shard_*_of_*.json"))
    overlap = {
        case["case_name"]: float(np.mean(case["correspondence"]["static"]["frame_overlap_ratio_0_20m"]))
        for case in v13_cases
    }
    evaluation = json.loads(args.eval_json.read_text(encoding="utf-8"))
    rows = []
    for case in candidates:
        fold = evaluation["folds"][str(case["record"]["source"])]
        descriptor = fold["selected_config"]["descriptor"]
        strategy = fold["selected_config"]["anchor_strategy"]
        fixed = case["baselines"]["fixed_explicit"]
        one = case["variants"][f"{descriptor}__{strategy}__1f"]
        three = case["variants"][f"{descriptor}__{strategy}__3f"]
        rows.append(
            {
                "case_name": case["case_name"],
                "texture": float(case["texture_score"]),
                "overlap": overlap[case["case_name"]],
                "planarity": float(three.get("target_geometry", {}).get("planarity_ratio", float("nan"))),
                "angle_bucket": str(case["record"].get("angle_bucket", "unknown")),
                "fixed": fixed,
                "world_1f": one,
                "world_3f": three,
                "oracle_select": oracle([fixed, one, three]),
                "oracle_all": oracle([fixed, *case["variants"].values()]),
            }
        )
    group_specs = {
        "texture": ("texture", ("low", "medium", "high")),
        "pseudo_overlap": ("overlap", ("low", "medium", "high")),
        "geometry": ("planarity", ("planar", "medium", "nondegenerate")),
    }
    groups = {}
    for group_name, (key, names) in group_specs.items():
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        low, high = np.nanpercentile(values, [33.333, 66.667])
        labels = [names[0] if value <= low else names[1] if value <= high else names[2] for value in values]
        groups[group_name] = {}
        for label in names:
            subset = [row for row, current in zip(rows, labels) if current == label]
            groups[group_name][label] = {
                method: stats(subset, method)
                for method in ("fixed", "world_1f", "world_3f", "oracle_select", "oracle_all")
            }
    groups["angle_bucket"] = {}
    for label in sorted({row["angle_bucket"] for row in rows}):
        subset = [row for row in rows if row["angle_bucket"] == label]
        groups["angle_bucket"][label] = {
            method: stats(subset, method)
            for method in ("fixed", "world_1f", "world_3f", "oracle_select", "oracle_all")
        }
    report = {
        "experiment": "V14 grouped candidate and Oracle analysis",
        "case_count": len(rows),
        "overlap_label": "V13 offline-teacher pseudo overlap; evaluation grouping only",
        "overall": {
            method: stats(rows, method)
            for method in ("fixed", "world_1f", "world_3f", "oracle_select", "oracle_all")
        },
        "groups": groups,
    }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    print(json.dumps(groups, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
