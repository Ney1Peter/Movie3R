#!/usr/bin/env python3
"""Select a GT-free, BRTC-fallback VSP root policy on a dev report only.

The evaluated action is a convex blend of two causal root corrections already
available at the first post-cut frame:

    c = c_BRTC + alpha * (c_shadow - c_BRTC)

It is committed only if frozen BRTC accepted the person and the two predicted
corrections agree in direction and magnitude.  Otherwise the output is exact
BRTC.  GT is used solely in this script to choose one policy on the pair-
disjoint development split; the generated JSON contains only runtime-observed
quantities and can be evaluated unchanged on confirmation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = REPO_ROOT / "output/v14_cut_first_cross_source/eval_vsp_dev_96/report.json"
DEFAULT_OUTPUT = REPO_ROOT / "output/v14/fine_alignment_research/vsp_root_agreement"
SOURCES = ("avatarrex", "thuman", "mvhuman100", "mvhuman200")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
    }


def candidate_rows(report: dict[str, Any], alpha: float, min_cosine: float, max_disagreement: float | None) -> dict[str, Any]:
    cases = []
    for row in report["cases"]:
        typed = row["shadow_typed_residual"]
        if not typed["available"]:
            continue
        root = typed["root"]
        shadow = np.asarray(root["shadow_minus_b0_world"], dtype=np.float64)
        brtc = np.asarray(root["brtc_minus_b0_world"], dtype=np.float64)
        oracle = np.asarray(root["oracle_minus_b0_world"], dtype=np.float64)
        disagreement = float(np.linalg.norm(shadow - brtc))
        norm_product = float(np.linalg.norm(shadow) * np.linalg.norm(brtc))
        agreement = float(np.dot(shadow, brtc) / norm_product) if norm_product > 1e-10 else float("nan")
        brtc_accepted = bool(row["diagnostics"]["brtc"]["accepted_count"] > 0)
        accepted = bool(
            brtc_accepted
            and np.isfinite(agreement)
            and agreement >= min_cosine
            and (max_disagreement is None or disagreement <= max_disagreement)
        )
        correction = brtc + alpha * (shadow - brtc) if accepted else brtc
        baseline_error = float(np.linalg.norm(brtc - oracle))
        candidate_error = float(np.linalg.norm(correction - oracle))
        cases.append(
            {
                "source": row["source"],
                "accepted": accepted,
                "agreement_cosine": agreement,
                "disagreement_m": disagreement,
                "baseline_error_m": baseline_error,
                "candidate_error_m": candidate_error,
                "delta_m": candidate_error - baseline_error,
            }
        )
    baseline = np.asarray([row["baseline_error_m"] for row in cases])
    candidate = np.asarray([row["candidate_error_m"] for row in cases])
    delta = candidate - baseline
    by_source = {}
    for source in SOURCES:
        subset = [row for row in cases if row["source"] == source]
        if not subset:
            continue
        source_baseline = np.asarray([row["baseline_error_m"] for row in subset])
        source_candidate = np.asarray([row["candidate_error_m"] for row in subset])
        by_source[source] = {
            "count": len(subset),
            "accepted_count": int(sum(row["accepted"] for row in subset)),
            "baseline": stats(source_baseline),
            "candidate": stats(source_candidate),
            "relative_gain": float(1.0 - source_candidate.mean() / source_baseline.mean()),
        }
    return {
        "policy": {
            "alpha": alpha,
            "min_cosine": min_cosine,
            "max_disagreement_m": max_disagreement,
        },
        "count": len(cases),
        "accepted_count": int(sum(row["accepted"] for row in cases)),
        "coverage": float(np.mean([row["accepted"] for row in cases])),
        "baseline": stats(baseline),
        "candidate": stats(candidate),
        "relative_gain": float(1.0 - candidate.mean() / baseline.mean()),
        "improvement_rate": float(np.mean(delta < 0.0)),
        "harm_over_5cm_rate": float(np.mean(delta > 0.05)),
        "p95_delta_m": float(np.quantile(delta, 0.95)),
        "by_source": by_source,
    }


def eligible(row: dict[str, Any]) -> bool:
    sources = row["by_source"]
    nonnegative = sum(values["relative_gain"] >= -1e-9 for values in sources.values())
    improving = sum(values["relative_gain"] >= 0.01 for values in sources.values())
    return bool(
        row["accepted_count"] >= 8
        and row["relative_gain"] >= 0.01
        and row["harm_over_5cm_rate"] <= 0.10
        and row["candidate"]["p95"] <= row["baseline"]["p95"] + 1e-12
        and nonnegative >= 3
        and improving >= 2
    )


def selection_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    policy = row["policy"]
    max_disagreement = float("inf") if policy["max_disagreement_m"] is None else float(policy["max_disagreement_m"])
    # Larger gain first; then lower harm and deliberately more conservative
    # interpolation/gating choices.
    return (
        -float(row["relative_gain"]),
        float(row["harm_over_5cm_rate"]),
        float(policy["alpha"]),
        -float(policy["min_cosine"]),
        max_disagreement,
    )


def main() -> None:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if len(report.get("failures", [])):
        raise RuntimeError("Development report has failures; do not select a policy from it")
    grid = []
    for alpha in (0.125, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875, 1.0):
        for min_cosine in (0.0, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99):
            for max_disagreement in (0.05, 0.10, 0.20, 0.40, 0.80, 1.60, None):
                row = candidate_rows(report, alpha, min_cosine, max_disagreement)
                row["eligible"] = eligible(row)
                grid.append(row)
    qualified = [row for row in grid if row["eligible"]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scan = {
        "experiment": "VSP root agreement policy development scan",
        "input_report": str(args.report),
        "input_checkpoint": report["checkpoint"],
        "development_protocol": "pair-disjoint dev96; GT used only for policy selection",
        "candidate_count": len(grid),
        "qualified_count": len(qualified),
        "candidates": sorted(grid, key=selection_key),
    }
    (args.output_dir / "DEV_SCAN.json").write_text(
        json.dumps(scan, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    if not qualified:
        print("NO_GO_VSP_ROOT_AGREEMENT")
        return
    winner = min(qualified, key=selection_key)
    policy = {
        "freeze_id": "VSP_ROOT_AGREEMENT_V1_20260803",
        "status": "frozen_after_pair_disjoint_dev_before_confirmation",
        "method_name": "Verified Shadow Projection: BRTC/Shadow Root Agreement Blend",
        "base": "cross96 B0 + frozen BRTC-LC v1",
        "runtime_inputs": [
            "B0 raw-root correction from last-pre/first-post", "shadow-root correction",
            "frozen BRTC accepted flag",
        ],
        "parameters": {
            **winner["policy"],
            "require_brtc_accepted": True,
        },
        "action": "brtc + alpha * (shadow - brtc) only when both agreement gates pass",
        "fallback": "exact BRTC-LC v1 geometry",
        "prohibitions": ["no GT", "no future frame", "no camera update", "no shadow-state commit"],
        "selection_requirements": {
            "relative_root_gain_at_least": 0.01,
            "root_harm_over_5cm_at_most": 0.10,
            "p95_not_worse": True,
            "at_least_3_sources_nonnegative": True,
            "at_least_2_sources_gain_at_least": 0.01,
        },
        "development_result": winner,
        "confirmation_status": "not_run",
    }
    (args.output_dir / "FROZEN_VSP_ROOT_POLICY_BEFORE_CONFIRM.json").write_text(
        json.dumps(policy, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"winner": winner["policy"], "gain": winner["relative_gain"]}, indent=2))


if __name__ == "__main__":
    main()
