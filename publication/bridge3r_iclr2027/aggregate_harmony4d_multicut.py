#!/usr/bin/env python3
"""Fail closed aggregation for frozen Harmony4D multi-cut auxiliary evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


METHODS = ("strict_human3r", "bridge3r")
METRICS = {
    "W-MPJPE_mm": ("multi_thumbs_named_provisional", "w_mpjpe_mm", "mean"),
    "WA-MPJPE_mm": ("multi_thumbs_named_provisional", "wa_mpjpe_mm", "mean"),
    "ATE-Sim3_m": ("multi_thumbs_named_provisional", "ate_sim3_m", "mean"),
    "IDF1": ("identity", "idf1"),
    "Coverage": ("coverage", "coverage"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", type=Path, nargs="+", required=True)
    parser.add_argument("--no-cut-runtimes", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table-output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def nested(row: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = row
    for key in path:
        value = value[key]
    return value


def finite_mean(values: list[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return None if not valid else float(np.mean(valid))


def detector_totals(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [row["detectors"][key] for row in rows]
    tp = sum(int(value["tp"]) for value in values)
    fp = sum(int(value["fp"]) for value in values)
    fn = sum(int(value["fn"]) for value in values)
    noncut = sum(
        int(row["record"]["clip_length"]) - len(row["record"]["boundaries"])
        for row in rows
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "false_positive_rate_per_noncut_pair": fp / max(noncut, 1),
        "brier_macro": finite_mean([value.get("brier") for value in values]),
    }


def no_cut_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    detector_rows = [row["runtime"][key] for row in rows]
    labels = [int(label) for row in detector_rows for label in row["labels"]]
    exact = [
        bool(value)
        for row in rows
        for value in row["bridge3r"]["no_cut_array_equal"].values()
    ]
    return {
        "case_count": len(rows),
        "positive_count": int(sum(labels)),
        "frame_pair_count": len(labels),
        "false_positive_rate_per_pair": float(sum(labels) / max(len(labels), 1)),
        "first_positive_indices": [row.get("first_positive_index") for row in detector_rows],
        "no_cut_array_checks": len(exact),
        "no_cut_array_checks_all_exact": bool(exact and all(exact)),
    }


def main() -> None:
    args = parse_args()
    evaluations = [load(path) for path in args.evaluations]
    controls = [load(path) for path in args.no_cut_runtimes]
    case_ids = [str(row["case_id"]) for row in evaluations]
    control_ids = [str(row["record"]["case_id"]) for row in controls]
    if len(evaluations) != 4 or len(set(case_ids)) != 4:
        raise ValueError("expected exactly four unique multi-cut evaluations")
    if len(controls) != 4 or len(set(control_ids)) != 4:
        raise ValueError("expected exactly four unique no-cut controls")
    if any(tuple(row["methods"]) != METHODS for row in evaluations):
        raise ValueError("multi-cut method order differs from locked comparison")
    if any(row["record"]["boundaries"] != [50, 100] for row in evaluations):
        raise ValueError("unexpected multi-cut boundaries")
    if any(row["record"]["boundaries"] != [] for row in controls):
        raise ValueError("unexpected no-cut boundaries")

    aggregate: dict[str, Any] = {}
    for method in METHODS:
        values = {
            name: finite_mean([nested(row["methods"][method], path) for row in evaluations])
            for name, path in METRICS.items()
        }
        seams = [
            seam["cut_seam"].get("root_excess_m")
            for row in evaluations
            for seam in row["methods"][method]["cut_seams"].values()
        ]
        values["Seam-root_m"] = finite_mean(seams)
        aggregate[method] = values

    result = {
        "schema_version": "Bridge3R-Harmony4D-multicut-summary-v1",
        "protocol": "Bridge3R-Harmony4D-MultiCut-v1",
        "case_count": len(evaluations),
        "boundaries_per_case": 2,
        "methods": aggregate,
        "multicut_detector": {
            key: detector_totals(evaluations, key)
            for key in ("causal_gru", "static_logistic")
        },
        "no_cut_controls": {
            key: no_cut_summary(controls, key)
            for key in ("causal_gru_detector", "static_logistic_detector")
        },
        "case_reports": [str(path.resolve()) for path in args.evaluations],
        "no_cut_runtime_reports": [str(path.resolve()) for path in args.no_cut_runtimes],
        "selection": "all four pre-registered cases retained; no result-based exclusion",
        "caveat": (
            "Auxiliary same-scene three-shot evidence only. It does not alter "
            "the primary single-cut protocol or support cross-scene claims."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    def number(value: float | None, digits: int) -> str:
        return "N/A" if value is None else f"{value:.{digits}f}"

    strict, bridge = aggregate["strict_human3r"], aggregate["bridge3r"]
    table = "\n".join([
        "% Auto-generated frozen Harmony4D multi-cut auxiliary result.",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Method & W $\downarrow$ & WA $\downarrow$ & ATE-Sim3 $\downarrow$ & IDF1 $\uparrow$ & Coverage $\uparrow$ & Seam-root $\downarrow$ \\",
        r"\midrule",
        "Strict \\humanthree{} & "
        + " & ".join([
            number(strict["W-MPJPE_mm"], 1), number(strict["WA-MPJPE_mm"], 1),
            number(strict["ATE-Sim3_m"], 3), number(strict["IDF1"], 3),
            number(strict["Coverage"], 3), number(strict["Seam-root_m"], 3),
        ]) + r" \\",
        "\\method{} (ours) & "
        + " & ".join([
            number(bridge["W-MPJPE_mm"], 1), number(bridge["WA-MPJPE_mm"], 1),
            number(bridge["ATE-Sim3_m"], 3), number(bridge["IDF1"], 3),
            number(bridge["Coverage"], 3), number(bridge["Seam-root_m"], 3),
        ]) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ])
    args.table_output.parent.mkdir(parents=True, exist_ok=True)
    args.table_output.write_text(table, encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
