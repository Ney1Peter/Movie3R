#!/usr/bin/env python3
"""Apply the preregistered v18 promotion rule to independent Harmony4D holdout reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from select_development_candidate import (
    REFERENCE,
    assess_candidates,
    finite,
    load_reports,
    mean,
    summarize_length,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = REPO_ROOT / "output/v18_harmony4d/holdout/per_sequence"
DEFAULT_OUTPUT = REPO_ROOT / "output/v18_harmony4d/holdout/decision.json"
DEFAULT_TARGET = "v18_dev_selected"
CORE_SHORTFALLS = ("W-MPJPE_mm", "WA-MPJPE_mm", "Accel_mm_frame2")
MIN_RELATIVE_GAIN = 0.001  # 0.1%; excludes reduction-level numerical noise.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--length", type=int, default=150)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_comparisons(
    reports: list[dict[str, Any]], target_name: str,
) -> list[dict[str, Any]]:
    comparisons = []
    for payload in reports:
        by_candidate: dict[str, list[dict[str, Any]]] = {}
        for row in payload.get("rows", []):
            if row.get("status") == "complete":
                by_candidate.setdefault(str(row["candidate"]), []).append(row)
        if REFERENCE not in by_candidate or target_name not in by_candidate:
            continue
        metrics = {}
        for metric in CORE_SHORTFALLS:
            reference = mean([
                finite(row["metrics"].get(metric)) for row in by_candidate[REFERENCE]
            ])
            target = mean([
                finite(row["metrics"].get(metric)) for row in by_candidate[target_name]
            ])
            ratio = None if reference in (None, 0) or target is None else target / reference
            metrics[metric] = {
                "v17": reference,
                "v18": target,
                "ratio": ratio,
                "improved_by_at_least_0.1pct": (
                    ratio is not None and ratio <= 1.0 - MIN_RELATIVE_GAIN
                ),
            }
        comparisons.append({
            "sequence": Path(payload["_source"]).parent.name,
            "evaluable_cases": int(payload.get("complete_case_count", 0)),
            "evaluator_unavailable": len(payload.get("skipped_cases", [])),
            "metrics": metrics,
        })
    return comparisons


def main() -> None:
    args = parse_args()
    grouped = load_reports(args.input)
    if args.length not in grouped:
        raise KeyError(f"length {args.length} is unavailable")
    reports = grouped[args.length]
    summary = summarize_length(reports)
    ranking = assess_candidates(summary)
    by_name = {row["name"]: row for row in ranking}
    if args.target not in by_name or REFERENCE not in by_name:
        raise KeyError(f"required candidates missing: {REFERENCE}, {args.target}")
    target, reference = by_name[args.target], by_name[REFERENCE]
    comparisons = sequence_comparisons(reports, args.target)
    majority = math.floor(len(comparisons) / 2) + 1

    core = {}
    stable_improvements = []
    for metric in CORE_SHORTFALLS:
        ratio = target["ratios_to_v17"].get(metric)
        sequence_wins = sum(
            bool(row["metrics"][metric]["improved_by_at_least_0.1pct"])
            for row in comparisons
        )
        stable = bool(
            ratio is not None and ratio <= 1.0 - MIN_RELATIVE_GAIN
            and sequence_wins >= majority
        )
        core[metric] = {
            "ratio_to_v17": ratio,
            "relative_change_percent": None if ratio is None else 100.0 * (ratio - 1.0),
            "sequence_wins": sequence_wins,
            "required_sequence_wins": majority,
            "stable_improvement": stable,
        }
        if stable:
            stable_improvements.append(metric)

    ref_metrics, metrics = reference["metrics"], target["metrics"]
    safety = dict(target["constraints"])
    safety.update({
        "ate_se3_within_5pct": (
            metrics.get("ATE_SE3_m") is not None and ref_metrics.get("ATE_SE3_m") not in (None, 0)
            and metrics["ATE_SE3_m"] <= 1.05 * ref_metrics["ATE_SE3_m"]
        ),
        "seam_root_within_5pct": (
            metrics.get("Seam_root_m") is not None and ref_metrics.get("Seam_root_m") not in (None, 0)
            and metrics["Seam_root_m"] <= 1.05 * ref_metrics["Seam_root_m"]
        ),
        "accepted_case_present": int(target["accepted"]) > 0,
    })
    promote = bool(
        all(safety.values())
        and len(stable_improvements) >= 2
        and target["score_to_v17"] < 1.0
    )
    sources = [Path(payload["_source"]) for payload in reports]
    result = {
        "schema_version": "Movie3R-v18-Harmony4D-independent-holdout-decision-v1",
        "decision_scope": "three preregistered Harmony4D train holdout actions; official test unused",
        "length": args.length,
        "target_name": args.target,
        "promotion_rule": {
            "minimum_stable_core_improvements": 2,
            "core_metrics": list(CORE_SHORTFALLS),
            "minimum_relative_gain_per_metric": MIN_RELATIVE_GAIN,
            "stability": "aggregate gain and gain on a strict majority of holdout actions",
            "all_safety_constraints_required": True,
            "overall_multimetric_score_must_improve": True,
        },
        "evaluable_case_count": summary["case_count"],
        "evaluator_unavailable": len(summary["skipped_cases"]),
        "target": target,
        "reference": reference,
        "core_assessment": core,
        "stable_improvements": stable_improvements,
        "safety": safety,
        "sequence_comparisons": comparisons,
        "promote_v18": promote,
        "decision": "promote_v18" if promote else "retain_v17",
        "source_reports": [
            {"path": str(path.resolve()), "sha256": sha256(path)} for path in sources
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "decision": result["decision"],
        "stable_improvements": stable_improvements,
        "safety_pass": all(safety.values()),
        "score_to_v17": target["score_to_v17"],
        "output": str(args.output.resolve()),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
