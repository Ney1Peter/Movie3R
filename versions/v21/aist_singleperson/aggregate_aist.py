#!/usr/bin/env python3
"""Aggregate frozen AIST++ CS150 case reports without silently dropping cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .protocol import atomic_json, canonical_json_digest, sha256_file
except ImportError:
    from protocol import atomic_json, canonical_json_digest, sha256_file  # type: ignore


SCHEMA = "Bridge3R-AIST-SinglePerson-CS150-aggregate-v1"
METRICS = (
    "pa_mpjpe_body12_mm",
    "first_shot_anchor_mpjpe_body12_mm",
    "first_shot_anchor_root_error_mm",
    "first_shot_anchor_orientation_proxy_deg",
    "seam_root_excess_mm",
    "seam_orientation_excess_deg",
    "post_camera_relative_rotation_deg",
    "post_camera_relative_translation_m",
)
DISPLAY = {
    "m0_strict_human3r": "Strict Human3R",
    "m1_clean_reset": "Clean reset",
    "m3_b0_only": "Coarse alignment only",
    "m4_b0_identity": "Coarse alignment + identity",
    "m6_b0_identity_brtc_c1": "Fine alignment / history transaction",
    "m14_safe_boundary_permutation_causal_gru": "Gated parent",
    "m15_bridge3r_fixed_v19": "Bridge3R (fixed, causal)",
    "prompthmr_official": "PromptHMR (official, offline)",
    "gvhmr_official": "GVHMR (official, offline)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--evaluator-manifest", type=Path, required=True)
    parser.add_argument("--role", choices=("pilot", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tex-output", type=Path)
    parser.add_argument("--bootstrap-seed", type=int, default=20260826)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    return parser.parse_args()


def read_manifest(path: Path, role: str) -> list[str]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [str(row.get("case_id")) for row in rows]
    if len(ids) != len(set(ids)) or not ids or any(row.get("role") != role or row.get("protocol") != "CS150" for row in rows):
        raise ValueError("Evaluator manifest is not a unique CS150 role manifest")
    return sorted(ids)


def metric_value(method: dict[str, Any], name: str) -> float | None:
    value = method.get("metrics", {}).get(name, {}).get("mean")
    return None if value is None else float(value)


def bootstrap_improvement(
    rows: dict[str, dict[str, Any]], bridge: str, baseline: str, metric: str, seed: int, reps: int
) -> dict[str, Any]:
    case_ids = sorted(set(rows.get(bridge, {})).intersection(rows.get(baseline, {})))
    values = []
    for case_id in case_ids:
        before, after = metric_value(rows[baseline][case_id], metric), metric_value(rows[bridge][case_id], metric)
        if before is not None and after is not None:
            values.append(before - after)
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"paired_case_count": 0, "mean_improvement": None, "ci95": None}
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(array), size=(int(reps), len(array)))
    statistics = array[samples].mean(axis=1)
    return {
        "paired_case_count": int(len(array)),
        "mean_improvement": float(array.mean()),
        "ci95": [float(np.quantile(statistics, .025)), float(np.quantile(statistics, .975))],
        "definition": f"{DISPLAY.get(baseline, baseline)} minus {DISPLAY.get(bridge, bridge)}; positive favours Bridge3R",
    }


def tex(value: float | None, digits: int = 1) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def build_tex(summary: dict[str, Any]) -> str:
    rows = []
    for method, values in summary["methods"].items():
        metric = values["metrics"]
        rows.append(
            f"{DISPLAY.get(method, method)} & {tex(metric['pa_mpjpe_body12_mm']['case_macro_mean'])} & "
            f"{tex(metric['first_shot_anchor_mpjpe_body12_mm']['case_macro_mean'])} & "
            f"{tex(metric['seam_root_excess_mm']['case_macro_mean'])} & "
            f"{tex(metric['post_camera_relative_rotation_deg']['case_macro_mean'])} & "
            f"{tex(values['coverage']['case_macro_mean'] * 100.0)} \\\\"
        )
    return "\n".join([
        "% Auto-generated from frozen AIST++ CS150 evaluator reports; do not hand-edit.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{AIST++ single-person cross-view CS150 results. Scores are case-macro means over the frozen official test sources; lower is better except coverage. PA and anchor errors use the declared AIST COCO body-12 joint set.}",
        "\\label{tab:aist-singleperson-cs150}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Method & PA-MPJPE $\\downarrow$ & Anchor-MPJPE $\\downarrow$ & Seam-root $\\downarrow$ & Rel. camera rot. $\\downarrow$ & Coverage $\\uparrow$ \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}}",
        "\\end{table*}",
        "",
    ])


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 100:
        raise SystemExit("--bootstrap-replicates must be at least 100")
    expected = read_manifest(args.evaluator_manifest, args.role)
    reports: dict[str, dict[str, Any]] = {}
    for path in sorted(args.metrics_dir.glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        case_id = str(value.get("case_id"))
        if case_id in reports:
            raise ValueError(f"Duplicate metric report for {case_id}")
        reports[case_id] = value
    unexpected = sorted(set(reports).difference(expected))
    if unexpected:
        raise ValueError(f"Metric reports outside frozen manifest: {unexpected[:3]}")
    method_cases: dict[str, dict[str, Any]] = {}
    detector_rows = []
    for case_id, report in reports.items():
        detector_rows.append(report.get("detector", {}))
        for method, row in report.get("methods", {}).items():
            method_cases.setdefault(method, {})[case_id] = row
    methods = {}
    for method, rows in sorted(method_cases.items()):
        metric_rows = {}
        for metric in METRICS:
            array = np.asarray([
                value for value in (metric_value(row, metric) for row in rows.values()) if value is not None
            ], dtype=np.float64)
            metric_rows[metric] = {
                "case_count": int(len(array)),
                "case_macro_mean": None if not len(array) else float(array.mean()),
                "case_macro_median": None if not len(array) else float(np.median(array)),
            }
        coverage = [float(row.get("coverage", {}).get("valid_frame_coverage", 0.0)) for row in rows.values()]
        completion = [float(row.get("coverage", {}).get("completion", 0.0)) for row in rows.values()]
        methods[method] = {
            "reported_case_count": len(rows),
            "formal_manifest_denominator": len(expected),
            "missing_metric_report_case_count": len(expected) - len(rows),
            "metrics": metric_rows,
            "coverage": {
                "case_macro_mean": float(np.mean(coverage)) if coverage else 0.0,
                "completion_case_macro_mean": float(np.mean(completion)) if completion else 0.0,
            },
        }
    detector = {
        "reported_case_count": len(detector_rows),
        "formal_manifest_denominator": len(expected),
        "tp": int(sum(int(row.get("tp", 0)) for row in detector_rows if row.get("available"))),
        "fp": int(sum(int(row.get("fp", 0)) for row in detector_rows if row.get("available"))),
        "fn": int(sum(int(row.get("fn", 0)) for row in detector_rows if row.get("available"))),
    }
    detector["precision"] = detector["tp"] / max(detector["tp"] + detector["fp"], 1)
    detector["recall"] = detector["tp"] / max(detector["tp"] + detector["fn"], 1)
    detector["f1"] = 2 * detector["tp"] / max(2 * detector["tp"] + detector["fp"] + detector["fn"], 1)
    bridge, baseline = "m15_bridge3r_fixed_v19", "m0_strict_human3r"
    paired = {
        metric: bootstrap_improvement(method_cases, bridge, baseline, metric, args.bootstrap_seed, args.bootstrap_replicates)
        for metric in METRICS
    } if bridge in method_cases and baseline in method_cases else {}
    payload = {
        "schema_version": SCHEMA,
        "role": args.role,
        "evaluator_manifest": str(args.evaluator_manifest.resolve()),
        "evaluator_manifest_sha256": sha256_file(args.evaluator_manifest.resolve()),
        "formal_manifest_case_count": len(expected),
        "metric_report_case_count": len(reports),
        "missing_reports": sorted(set(expected).difference(reports)),
        "methods": methods,
        "detector": detector,
        "paired_bridge3r_vs_strict_human3r": paired,
        "aggregation": "Each case contributes one within-case mean; final means are unweighted case macro averages. Missing/inference-failed frozen cases remain in the denominator through completion and coverage.",
    }
    payload["content_sha256"] = canonical_json_digest(payload)
    atomic_json(args.output, payload)
    if args.tex_output:
        args.tex_output.parent.mkdir(parents=True, exist_ok=True)
        args.tex_output.write_text(build_tex(payload), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "formal_cases": len(expected), "reports": len(reports), "methods": list(methods)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
