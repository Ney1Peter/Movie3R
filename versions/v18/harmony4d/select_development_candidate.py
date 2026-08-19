#!/usr/bin/env python3
"""Select a constrained Movie3R candidate from frozen Harmony4D dev reports."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = REPO_ROOT / "output/v18_harmony4d/dev/per_sequence"
DEFAULT_REPORT = REPO_ROOT / "output/v18_harmony4d/dev/selection.json"
DEFAULT_CANDIDATE = REPO_ROOT / "versions/v18/harmony4d/frozen_dev_candidate.json"
PARENT = "v16_0_m15_geometry"
REFERENCE = "v17_reference"
CORE_SCORE = {
    "W-MPJPE_mm": 0.25,
    "WA-MPJPE_mm": 0.25,
    "Accel_mm_frame2": 0.20,
    "ATE_Sim3_m": 0.15,
    "Seam_root_m": 0.15,
}
LOWER_METRICS = (
    "W-MPJPE_mm", "WA-MPJPE_mm", "MPJPE_mm", "MPVPE_mm",
    "Accel_mm_frame2", "ATE_Sim3_m", "ATE_SE3_m", "Seam_root_m",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--candidate-output", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--main-length", type=int, default=150)
    return parser.parse_args()


def finite(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def mean(values: list[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(valid)) if valid else None


def geometric_score(metrics: dict[str, float | None], reference: dict[str, float | None]) -> float:
    terms = []
    for metric, weight in CORE_SCORE.items():
        value, base = metrics.get(metric), reference.get(metric)
        if value is None or base is None or base <= 1e-12:
            return float("inf")
        terms.append(weight * math.log(max(value / base, 1e-12)))
    return float(math.exp(sum(terms)))


def load_reports(root: Path) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(root.rglob("cs*.json")):
        match = re.fullmatch(r"cs(\d+)", path.stem)
        if not match:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("errors"):
            raise ValueError(f"incomplete report {path}: {payload['errors']}")
        payload["_source"] = str(path.resolve())
        grouped[int(match.group(1))].append(payload)
    if not grouped:
        raise ValueError(f"no cs*.json reports below {root}")
    return grouped


def summarize_length(reports: list[dict[str, Any]]) -> dict[str, Any]:
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    configurations: dict[str, dict[str, Any]] = {}
    sources = []
    skipped = []
    for payload in reports:
        sources.append(payload["_source"])
        skipped.extend(payload.get("skipped_cases", []))
        for name, value in payload["aggregate"]["summary"].items():
            configurations[name] = value["candidate"]
        for row in payload.get("rows", []):
            if row.get("status") == "complete":
                rows[str(row["candidate"])].append(row)
    case_ids = {row["case_id"] for values in rows.values() for row in values}
    summary = {}
    for name, values in rows.items():
        if len(values) != len(case_ids) or len({row["case_id"] for row in values}) != len(case_ids):
            raise ValueError(f"candidate {name} has {len(values)} rows for {len(case_ids)} cases")
        metrics = {
            metric: mean([finite(row["metrics"].get(metric)) for row in values])
            for metric in (*LOWER_METRICS, "IDF1", "Coverage")
        }
        summary[name] = {
            "candidate": configurations[name], "case_count": len(values), "metrics": metrics,
            "rows": values,
        }
    return {
        "sources": sources, "case_count": len(case_ids), "skipped_cases": skipped,
        "candidates": summary,
    }


def assess_candidates(length_summary: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = length_summary["candidates"]
    if REFERENCE not in candidates or PARENT not in candidates:
        raise KeyError(f"required candidates missing: {REFERENCE}, {PARENT}")
    reference = candidates[REFERENCE]
    parent_by_case = {row["case_id"]: row for row in candidates[PARENT]["rows"]}
    ranking = []
    for name, value in candidates.items():
        metrics = value["metrics"]
        ref = reference["metrics"]
        ratios = {
            metric: None if metrics.get(metric) is None or ref.get(metric) in (None, 0)
            else float(metrics[metric] / ref[metric])
            for metric in metrics
        }
        accepted_rows = [
            row for row in value["rows"]
            if bool(row.get("diagnostics", {}).get("reliability_gate", {}).get("accepted"))
        ]
        catastrophic = []
        fallback_exact = True
        for row in value["rows"]:
            parent = parent_by_case[row["case_id"]]
            accepted = bool(row.get("diagnostics", {}).get("reliability_gate", {}).get("accepted"))
            w, parent_w = finite(row["metrics"].get("W-MPJPE_mm")), finite(parent["metrics"].get("W-MPJPE_mm"))
            if accepted and w is not None and parent_w is not None and w > 1.5 * parent_w:
                catastrophic.append(row["case_id"])
            if not accepted:
                for metric in (*LOWER_METRICS, "IDF1", "Coverage"):
                    first, second = finite(row["metrics"].get(metric)), finite(parent["metrics"].get(metric))
                    if first is None or second is None:
                        continue
                    # The candidate and parent arrays are exact fallbacks, but
                    # independent evaluator calls may differ at roughly
                    # 1e-5--1e-4 after float32/float64 reductions.  One
                    # thousandth of a reported metric unit is far below the
                    # displayed precision while still detecting real changes.
                    if not math.isclose(first, second, rel_tol=1e-6, abs_tol=1e-3):
                        fallback_exact = False
        constraints = {
            "ate_within_5pct": ratios.get("ATE_Sim3_m") is not None and ratios["ATE_Sim3_m"] <= 1.05,
            "idf1_drop_at_most_0005": metrics.get("IDF1") is not None and ref.get("IDF1") is not None
            and metrics["IDF1"] >= ref["IDF1"] - 0.005,
            "mpjpe_within_2pct": ratios.get("MPJPE_mm") is not None and ratios["MPJPE_mm"] <= 1.02,
            "mpvpe_within_2pct": ratios.get("MPVPE_mm") is not None and ratios["MPVPE_mm"] <= 1.02,
            "w_within_1pct": ratios.get("W-MPJPE_mm") is not None and ratios["W-MPJPE_mm"] <= 1.01,
            "accel_within_5pct": ratios.get("Accel_mm_frame2") is not None and ratios["Accel_mm_frame2"] <= 1.05,
            "no_catastrophic_accept": not catastrophic,
            "fallback_exact": fallback_exact,
        }
        ranking.append({
            "name": name, "candidate": value["candidate"], "metrics": metrics,
            "ratios_to_v17": ratios, "score_to_v17": geometric_score(metrics, ref),
            "accepted": len(accepted_rows), "catastrophic_accepts": catastrophic,
            "constraints": constraints, "feasible": all(constraints.values()),
        })
    ranking.sort(key=lambda row: (not row["feasible"], row["score_to_v17"], row["name"]))
    return ranking


def recommended_length(
    summaries: dict[int, dict[str, Any]], selected_name: str, main_length: int,
) -> dict[str, Any]:
    main = summaries[main_length]["candidates"][selected_name]["metrics"]
    rows = []
    for length in sorted(summaries):
        metrics = summaries[length]["candidates"][selected_name]["metrics"]
        quality = mean([
            metrics[key] / main[key]
            for key in ("W-MPJPE_mm", "WA-MPJPE_mm", "Accel_mm_frame2")
            if metrics.get(key) is not None and main.get(key) not in (None, 0)
        ])
        eligible = bool(
            length < main_length and quality is not None and quality <= 0.95
            and metrics.get("ATE_Sim3_m") is not None and main.get("ATE_Sim3_m") is not None
            and metrics["ATE_Sim3_m"] <= 1.05 * main["ATE_Sim3_m"]
            and metrics.get("IDF1") is not None and main.get("IDF1") is not None
            and metrics["IDF1"] >= main["IDF1"] - 0.005
            and metrics.get("MPJPE_mm") is not None and main.get("MPJPE_mm") is not None
            and metrics["MPJPE_mm"] <= 1.02 * main["MPJPE_mm"]
        )
        rows.append({
            "length": length, "metrics": metrics,
            "mean_W_WA_Accel_ratio_to_150": quality, "low_latency_eligible": eligible,
        })
    eligible = [row["length"] for row in rows if row["low_latency_eligible"]]
    return {
        "main_length": main_length,
        "recommended_low_latency_length": max(eligible) if eligible else main_length,
        "selection_rule": "longest shorter window with >=5% mean W/WA/Accel gain while preserving ATE, IDF1 and MPJPE",
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    reports = load_reports(args.input)
    if args.main_length not in reports:
        raise KeyError(f"main length {args.main_length} missing")
    summaries = {length: summarize_length(values) for length, values in reports.items()}
    ranking = assess_candidates(summaries[args.main_length])
    feasible = [row for row in ranking if row["feasible"]]
    if not feasible:
        raise ValueError("no candidate satisfies preservation constraints")
    best = feasible[0]
    selected_name = best["name"] if best["score_to_v17"] < 1.0 else REFERENCE
    selected = next(row for row in ranking if row["name"] == selected_name)
    length_result = recommended_length(summaries, selected_name, args.main_length)
    result = {
        "schema_version": "Movie3R-v18-Harmony4D-development-selection-v1",
        "selection_scope": "Harmony4D train development only; official test metrics unused",
        "main_length": args.main_length, "selected_source_name": selected_name,
        "selected": selected, "ranking_cs150": ranking,
        "length_ablation": length_result,
        "per_length": {
            str(length): {
                "sources": value["sources"], "case_count": value["case_count"],
                "evaluator_unavailable": len(value["skipped_cases"]),
            }
            for length, value in summaries.items()
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    candidate = dict(selected["candidate"])
    candidate["name"] = "v18_dev_selected"
    reference_candidate = dict(
        summaries[args.main_length]["candidates"][REFERENCE]["candidate"]
    )
    reference_candidate["name"] = REFERENCE
    frozen = {
        "phase": "frozen after three-action train development and before independent train holdout",
        "selection_report": str(args.report.resolve()),
        "main_length": args.main_length,
        "recommended_low_latency_length": length_result["recommended_low_latency_length"],
        "source_candidate": selected_name,
        # Keep the exact v17 configuration in the holdout file.  The parent is
        # needed to audit prediction-only fallbacks, while v17 is the actual
        # promotion baseline for the development-selected candidate.
        "candidates": [{"name": PARENT}, reference_candidate, candidate],
    }
    args.candidate_output.parent.mkdir(parents=True, exist_ok=True)
    args.candidate_output.write_text(json.dumps(frozen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "selected": selected_name, "score_to_v17": selected["score_to_v17"],
        "low_latency_length": length_result["recommended_low_latency_length"],
        "report": str(args.report.resolve()), "candidate": str(args.candidate_output.resolve()),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
