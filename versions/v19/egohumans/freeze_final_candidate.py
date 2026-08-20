#!/usr/bin/env python3
"""Apply the pre-registered independent-holdout gate and freeze one method."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


BASELINE = "v16_0_m15_geometry"
FALLBACK = "v17_harmony_multicue_safe"
FINAL_NAME = "v19_egohumans_frozen"
CORE = ("W-MPJPE_mm", "WA-MPJPE_mm", "Accel_mm_frame2", "ATE_SE3_m", "Seam_root_m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-summary", type=Path, required=True)
    parser.add_argument("--holdout-summary", type=Path, required=True)
    parser.add_argument("--holdout-candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def finite(value: Any) -> float | None:
    if value in (None, ""):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def ratio(first: float | None, second: float | None) -> float | None:
    return None if first is None or second is None or abs(first) < 1e-12 else second / first


def case_rows(csv_path: Path) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[str(row["method"])].append(row)
    return grouped


def action_decision(
    candidate: list[dict[str, str]], parent: list[dict[str, str]]
) -> dict[str, Any]:
    parent_by_case = {str(row["case_id"]): row for row in parent}
    by_action: dict[str, list[float]] = defaultdict(list)
    worst_w = []
    for row in candidate:
        other = parent_by_case.get(str(row["case_id"]))
        if other is None:
            continue
        ratios = []
        for metric in CORE:
            value = ratio(finite(other.get(metric)), finite(row.get(metric)))
            if value is not None and value > 0:
                ratios.append(value)
        if len(ratios) == len(CORE):
            by_action[str(row["action"])].append(float(np.exp(np.mean(np.log(ratios)))))
        w_value = ratio(finite(other.get("W-MPJPE_mm")), finite(row.get("W-MPJPE_mm")))
        if w_value is not None:
            worst_w.append(w_value)
    action_ratios = {
        action: float(np.mean(values)) for action, values in sorted(by_action.items()) if values
    }
    nonworse = sum(value <= 1.0 for value in action_ratios.values())
    return {
        "action_core_geomean_ratio": action_ratios,
        "nonworse_actions": nonworse,
        "action_count": len(action_ratios),
        "at_least_five_of_seven_nonworse": len(action_ratios) == 7 and nonworse >= 5,
        "worst_case_w_ratio": max(worst_w, default=None),
        "worst_case_w_harm_le_20pct": not worst_w or max(worst_w) <= 1.20,
    }


def main() -> None:
    args = parse_args()
    development = json.loads(args.development_summary.read_text(encoding="utf-8"))
    holdout = json.loads(args.holdout_summary.read_text(encoding="utf-8"))
    frozen = json.loads(args.holdout_candidates.read_text(encoding="utf-8"))
    if development.get("split") != "development" or holdout.get("split") != "holdout":
        raise ValueError("Expected development and independent holdout summaries")
    if holdout.get("protocol", {}).get("parameter_selection_allowed"):
        raise ValueError("Holdout incorrectly marked as parameter-selection data")
    csv_path = args.holdout_summary.parent / "case_metrics.csv"
    rows = case_rows(csv_path)
    selected_names = list(frozen["selected_names"])
    candidate_by_name = {str(row["name"]): row for row in frozen["candidates"]}
    decisions = {}
    qualified = []
    parent_metrics = holdout["methods"][BASELINE]["case_macro"]
    for name in selected_names:
        if name not in development["methods"] or name not in holdout["methods"]:
            raise ValueError(f"Candidate {name} missing from development/holdout summary")
        dev = development["methods"][name]["development_promotion"]
        current = holdout["methods"][name]
        ratios = {
            metric: ratio(finite(parent_metrics.get(metric)), finite(current["case_macro"].get(metric)))
            for metric in set(CORE) | {"MPJPE_mm", "MPVPE_mm"}
        }
        core = [ratios[key] for key in CORE if ratios[key] is not None and ratios[key] > 0]
        geometric = float(np.exp(np.mean(np.log(core)))) if len(core) == len(CORE) else None
        development_improved = {
            key for key in CORE
            if finite(dev.get("ratios_to_parent", {}).get(key)) is not None
            and float(dev["ratios_to_parent"][key]) < 1.0
        }
        consistent = [key for key in CORE if key in development_improved and ratios[key] is not None and ratios[key] < 1.0]
        action = action_decision(rows.get(name, []), rows.get(BASELINE, []))
        idf1 = finite(current["case_macro"].get("IDF1"))
        parent_idf1 = finite(parent_metrics.get("IDF1"))
        coverage = finite(current["case_macro"].get("Coverage"))
        parent_coverage = finite(parent_metrics.get("Coverage"))
        checks = {
            "all_required_metrics_defined": all(
                int(current["finite_case_count"].get(key, 0)) == int(current["case_count"])
                for key in (*CORE, "MPJPE_mm", "MPVPE_mm", "IDF1", "Coverage")
            ),
            "three_core_directions_consistent_with_development": len(consistent) >= 3,
            "holdout_core_improvement_ge_2pct": geometric is not None and geometric <= 0.98,
            "mpjpe_noninferior_2pct": ratios["MPJPE_mm"] is not None and ratios["MPJPE_mm"] <= 1.02,
            "mpvpe_noninferior_2pct": ratios["MPVPE_mm"] is not None and ratios["MPVPE_mm"] <= 1.02,
            "coverage_drop_le_1pp": coverage is not None and parent_coverage is not None and coverage >= parent_coverage - 0.01,
            "idf1_drop_le_0p01": idf1 is not None and parent_idf1 is not None and idf1 >= parent_idf1 - 0.01,
            "five_of_seven_actions_nonworse": action["at_least_five_of_seven_nonworse"],
            "worst_case_w_harm_le_20pct": action["worst_case_w_harm_le_20pct"],
        }
        passed = bool(all(checks.values()))
        decisions[name] = {
            "passed": passed,
            "checks": checks,
            "holdout_ratios_to_parent": ratios,
            "holdout_core_geometric_mean_ratio": geometric,
            "consistent_improved_core_metrics": consistent,
            "action_safety": action,
        }
        if passed:
            qualified.append(name)
    qualified.sort(key=lambda name: decisions[name]["holdout_core_geometric_mean_ratio"])
    source_name = qualified[0] if qualified else FALLBACK
    if source_name not in candidate_by_name:
        raise ValueError(f"Fallback/final source {source_name} was not evaluated on holdout")
    source = json.loads(json.dumps(candidate_by_name[source_name]))
    final_name = FINAL_NAME if qualified else FALLBACK
    source["name"] = final_name
    source["geometry"]["name"] = final_name
    baseline = {
        "name": BASELINE,
        "geometry": {"name": BASELINE},
        "identity": None,
    }
    output = {
        "schema_version": "Movie3R-v19-EgoHumans-final-candidate-v1",
        "protocol": "Movie3R-EgoHumans-CS100-v1",
        "frozen_before_test": True,
        "test_metrics_read": False,
        "source_candidate_name": source_name,
        "final_method_name": final_name,
        "qualified_holdout_candidates": qualified,
        "fallback_used": not bool(qualified),
        "decisions": decisions,
        "candidates": [baseline, source],
        "provenance": {
            "development_summary": str(args.development_summary.resolve()),
            "development_summary_sha256": sha256(args.development_summary),
            "holdout_summary": str(args.holdout_summary.resolve()),
            "holdout_summary_sha256": sha256(args.holdout_summary),
            "holdout_case_metrics": str(csv_path.resolve()),
            "holdout_case_metrics_sha256": sha256(csv_path),
            "holdout_candidates": str(args.holdout_candidates.resolve()),
            "holdout_candidates_sha256": sha256(args.holdout_candidates),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    partial.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(partial, args.output)
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
