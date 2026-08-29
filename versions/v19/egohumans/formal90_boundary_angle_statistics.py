#!/usr/bin/env python3
"""Reaggregate boundary-only EgoHumans formal-90 metrics by viewpoint span.

The source artifacts are the immutable per-capture evaluator reports produced
for Strict Human3R, the no-learned-correction route, alignment-only, and the
full Bridge3R route.  This script does not read RGB, predictions, calibration,
or GT directly.  It verifies exact parity with the formal 90-case manifest,
retains every finite-support denominator, and reports equal-capture macro
means with capture-resampling confidence intervals.

Evaluator metrics and prediction-only transform diagnostics are deliberately
separated.  The latter are unavailable for Strict Human3R and are never
presented as GT accuracy.
"""

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


SCHEMA = "Bridge3R-EgoHumans-formal90-boundary-angle-statistics-v1"
CANDIDATE = "v19_egohumans_frozen"
STRICT_KEY = "m0_strict_human3r"
METHOD_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("Strict Human3R", "full_replay", "reference", STRICT_KEY),
    ("No learned correction branch", "native", "candidate", CANDIDATE),
    ("Alignment only", "alignment_only", "candidate", CANDIDATE),
    ("Bridge3R (full)", "full_replay", "candidate", CANDIDATE),
)

# key, paper label, unit, direction, source class
METRICS: tuple[tuple[str, str, str, str, str], ...] = (
    ("Boundary_camera_t_m", "Boundary camera T", "m", "lower", "evaluator"),
    ("Boundary_camera_R_deg", "Boundary camera R", "deg", "lower", "evaluator"),
    ("Seam_camera_t_m", "Seam camera T", "m", "lower", "evaluator"),
    ("Seam_camera_R_deg", "Seam camera R", "deg", "lower", "evaluator"),
    ("Boundary_root_m", "Boundary root", "m", "lower", "evaluator"),
    ("Post_root_m", "Post root", "m", "lower", "evaluator"),
    ("Seam_root_m", "Seam root", "m", "lower", "evaluator"),
    ("Seam_CHRGE_m", "Seam camera--human residual", "m", "lower", "evaluator"),
    ("CHRGE_m", "Camera--human root-gauge residual", "m", "lower", "evaluator"),
    (
        "Transform_residual_before_m",
        "Prediction-only transform residual (before)",
        "m",
        "lower",
        "runtime_diagnostic",
    ),
    (
        "Transform_residual_after_m",
        "Prediction-only transform residual (after)",
        "m",
        "lower",
        "runtime_diagnostic",
    ),
    (
        "Transform_residual_reduction_m",
        "Prediction-only transform residual reduction",
        "m",
        "higher",
        "runtime_diagnostic",
    ),
)

METRIC_DEFINITIONS = {
    "Boundary_camera_t_m": (
        "Euclidean camera-centre error at the first post-cut frame after the "
        "evaluator's one shared initial Sim(3) alignment."
    ),
    "Boundary_camera_R_deg": (
        "Camera rotation geodesic error at the first post-cut frame after the same "
        "shared initial alignment."
    ),
    "Seam_camera_t_m": (
        "Euclidean error between predicted and GT camera-centre displacement from the "
        "last pre-cut frame to the first post-cut frame."
    ),
    "Seam_camera_R_deg": (
        "Geodesic error between predicted and GT relative camera rotation across the cut."
    ),
    "Boundary_root_m": (
        "Mean pelvis world-space error at the first post-cut frame under the shared "
        "initial Sim(3), over evaluator-matched people."
    ),
    "Post_root_m": (
        "Mean pelvis world-space error over all evaluator-matched post-cut person frames "
        "under the shared initial Sim(3)."
    ),
    "Seam_root_m": (
        "Mean discrepancy between predicted and GT pelvis displacement across the cut "
        "for identities evaluator-matched on both boundary frames."
    ),
    "Seam_CHRGE_m": (
        "Mean discrepancy of the cross-cut pelvis displacement expressed in each "
        "instantaneous camera coordinate frame; a camera--human relative seam residual."
    ),
    "CHRGE_m": (
        "Mean per-frame pelvis residual in camera coordinates after the shared world "
        "alignment; this measures camera--human relative root gauge."
    ),
    "Transform_residual_before_m": (
        "Prediction-only matched-torso residual before the shared boundary translation. "
        "It uses no GT and is not defined for Strict Human3R."
    ),
    "Transform_residual_after_m": (
        "Prediction-only matched-torso residual after the shared boundary translation. "
        "It uses no GT and is not defined for Strict Human3R."
    ),
    "Transform_residual_reduction_m": (
        "Before minus after prediction-only matched-torso residual; positive values mean "
        "that the boundary translation reduced its own matching residual."
    ),
}

GROUP_ORDER = ("all", "small", "medium", "large", "extreme", "ge150")
NUMERICAL_TIE_ATOL = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parents[3]
    parser.add_argument(
        "--parent",
        type=Path,
        default=repo_root / "output/bridge3r_egohumans_ablation_v1",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            repo_root.parents[0]
            / "ICLR-paper/bridge3r_iclr2027/private_audit/egohumans_formal90_manifest.jsonl"
        ),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-tex", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260829)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(value, encoding="utf-8")
    os.replace(partial, path)


def finite(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = {str(row["case_id"]): row for row in rows}
    if len(rows) != 90 or len(result) != 90:
        raise ValueError("formal manifest must contain exactly 90 unique cases")
    if {str(row.get("formal_protocol")) for row in rows} != {
        "Bridge3R-EgoHumans-formal90-v1"
    }:
        raise ValueError("unexpected formal manifest protocol")
    strata = {str(row["angle_stratum"]) for row in rows}
    if strata != {"small", "medium", "large", "extreme"}:
        raise ValueError(f"unexpected viewpoint strata: {sorted(strata)}")
    return result


def report_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.glob("test/captures/*/candidate_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if payload.get("schema_version") == "Movie3R-v16-Harmony4D-causal-stabilization-probe-v1":
            files.append(path)
    if len(files) != 27:
        raise ValueError(f"{root}: expected 27 evaluator reports, found {len(files)}")
    return files


def diagnostic_metrics(row: dict[str, Any]) -> dict[str, float | None]:
    diagnostics = row.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return {
            "Transform_residual_before_m": None,
            "Transform_residual_after_m": None,
            "Transform_residual_reduction_m": None,
        }
    boundary = diagnostics.get("boundary")
    if not isinstance(boundary, dict):
        boundary = {}
    before = finite(boundary.get("torso_residual_before_m"))
    after = finite(boundary.get("torso_residual_after_m"))
    return {
        "Transform_residual_before_m": before,
        "Transform_residual_after_m": after,
        "Transform_residual_reduction_m": (
            before - after if before is not None and after is not None else None
        ),
    }


def load_method(
    parent: Path,
    route: str,
    kind: str,
    source_name: str,
    expected_cases: set[str],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    root = parent / f"formal90_{route}"
    rows: dict[str, dict[str, Any]] = {}
    sources: list[dict[str, str]] = []
    for path in report_files(root):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("errors") or payload.get("skipped_cases"):
            raise ValueError(f"{path}: evaluator errors/skips are not allowed")
        sources.append({"path": str(path.resolve()), "sha256": sha256(path)})
        source_rows = payload.get("reference_rows" if kind == "reference" else "rows", [])
        for row in source_rows:
            observed = str(row.get("method" if kind == "reference" else "candidate", ""))
            if observed != source_name:
                continue
            if row.get("status") != "complete":
                raise ValueError(f"{path}: incomplete {source_name}/{row.get('case_id')}")
            case_id = str(row.get("case_id", ""))
            if case_id not in expected_cases:
                raise ValueError(f"{path}: unexpected case {case_id}")
            if case_id in rows:
                raise ValueError(f"duplicate {source_name}/{case_id}")
            values = {
                metric: finite(row.get("metrics", {}).get(metric))
                for metric, _, _, _, source_class in METRICS
                if source_class == "evaluator"
            }
            values.update(diagnostic_metrics(row) if kind == "candidate" else diagnostic_metrics({}))
            rows[case_id] = {
                "case_id": case_id,
                "sequence": str(row.get("sequence", "")),
                "capture": str(row.get("capture", "")),
                "angle_stratum": str(row.get("angle_stratum", "")),
                "metrics": values,
            }
    observed_cases = set(rows)
    if observed_cases != expected_cases:
        raise ValueError(
            f"{root}/{source_name}: case mismatch; "
            f"missing={sorted(expected_cases-observed_cases)}, "
            f"extra={sorted(observed_cases-expected_cases)}"
        )
    return rows, sources


def group_members(formal: dict[str, dict[str, Any]]) -> dict[str, set[str]]:
    groups: dict[str, set[str]] = {name: set() for name in GROUP_ORDER}
    for case_id, row in formal.items():
        groups["all"].add(case_id)
        groups[str(row["angle_stratum"])].add(case_id)
        if float(row["camera_rotation_span_deg_evaluator_only"]) >= 150.0:
            groups["ge150"].add(case_id)
    if len(groups["all"]) != 90 or not groups["ge150"]:
        raise ValueError("invalid all/ge150 formal groups")
    return groups


def capture_bootstrap(
    cluster_values: np.ndarray, *, samples: int, rng: np.random.Generator
) -> list[float]:
    if cluster_values.size == 0:
        return [None, None]  # type: ignore[list-item]
    indices = rng.integers(
        0,
        cluster_values.size,
        size=(samples, cluster_values.size),
    )
    draws = cluster_values[indices].mean(axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def summarize_values(
    rows: dict[str, dict[str, Any]],
    members: set[str],
    metric: str,
    *,
    samples: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    values: list[tuple[str, float]] = []
    for case_id in sorted(members):
        value = finite(rows[case_id]["metrics"].get(metric))
        if value is None:
            continue
        cluster = f"{rows[case_id]['sequence']}/{rows[case_id]['capture']}"
        values.append((cluster, value))
    by_cluster: dict[str, list[float]] = defaultdict(list)
    for cluster, value in values:
        by_cluster[cluster].append(value)
    cluster_means = np.asarray(
        [np.mean(by_cluster[key]) for key in sorted(by_cluster)], dtype=np.float64
    )
    return {
        "group_case_denominator": len(members),
        "finite_case_support": len(values),
        "finite_capture_support": len(cluster_means),
        "case_macro_mean": float(np.mean([value for _, value in values])) if values else None,
        "capture_macro_mean": float(cluster_means.mean()) if len(cluster_means) else None,
        "capture_bootstrap_ci95": capture_bootstrap(
            cluster_means, samples=samples, rng=rng
        ),
    }


def summarize_paired(
    full: dict[str, dict[str, Any]],
    comparator: dict[str, dict[str, Any]],
    members: set[str],
    metric: str,
    direction: str,
    *,
    samples: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    values: list[tuple[str, float]] = []
    for case_id in sorted(members):
        full_value = finite(full[case_id]["metrics"].get(metric))
        other_value = finite(comparator[case_id]["metrics"].get(metric))
        if full_value is None or other_value is None:
            continue
        gain = other_value - full_value if direction == "lower" else full_value - other_value
        if abs(gain) <= NUMERICAL_TIE_ATOL:
            gain = 0.0
        cluster = f"{full[case_id]['sequence']}/{full[case_id]['capture']}"
        values.append((cluster, gain))
    by_cluster: dict[str, list[float]] = defaultdict(list)
    for cluster, value in values:
        by_cluster[cluster].append(value)
    cluster_means = np.asarray(
        [np.mean(by_cluster[key]) for key in sorted(by_cluster)], dtype=np.float64
    )
    gains = np.asarray([value for _, value in values], dtype=np.float64)
    wins = int(np.count_nonzero(gains > 0)) if len(gains) else 0
    ties = int(np.count_nonzero(gains == 0)) if len(gains) else 0
    return {
        "group_case_denominator": len(members),
        "paired_case_support": len(gains),
        "paired_capture_support": len(cluster_means),
        "case_macro_improvement": float(gains.mean()) if len(gains) else None,
        "capture_macro_improvement": float(cluster_means.mean()) if len(cluster_means) else None,
        "capture_bootstrap_ci95": capture_bootstrap(
            cluster_means, samples=samples, rng=rng
        ),
        "case_win_tie_loss": {
            "wins": wins,
            "ties": ties,
            "losses": int(len(gains) - wins - ties),
        },
        "convention": "positive favours Bridge3R (full)",
    }


def fmt(value: Any, unit: str) -> str:
    number = finite(value)
    if number is None:
        return "--"
    return f"{number:.2f}" if unit == "deg" else f"{number:.3f}"


def summary_cell(value: dict[str, Any], unit: str) -> str:
    low, high = value["capture_bootstrap_ci95"]
    return (
        f"{fmt(value['capture_macro_mean'], unit)} "
        f"[{fmt(low, unit)}, {fmt(high, unit)}] "
        f"({value['finite_case_support']}/{value['finite_capture_support']})"
    )


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# EgoHumans formal-90 boundary and viewpoint-span statistics",
        "",
        f"- Formal manifest SHA-256: `{payload['formal_manifest_sha256']}`",
        f"- Cases / captures: {payload['case_count']} / {payload['capture_count']}",
        (
            "- Main values are equal-capture macro means; square brackets are 95% "
            "capture-resampling bootstrap intervals and parentheses are finite "
            "cases/captures."
        ),
        (
            "- Evaluator boundary metrics use GT only in evaluation. Runtime transform residuals "
            "are prediction-only diagnostics and are never treated as GT accuracy."
        ),
        "",
        "## Metric definitions",
        "",
    ]
    for metric, label, unit, _, source_class in METRICS:
        lines.append(
            f"- **{label}** (`{metric}`, {unit}; {source_class}): {METRIC_DEFINITIONS[metric]}"
        )
    for group in GROUP_ORDER:
        group_payload = payload["groups"][group]
        lines.extend(
            [
                "",
                f"## Group: {group} (N={group_payload['case_denominator']})",
                "",
                "### Evaluator boundary metrics",
                "",
                "| Method | B-Cam T | B-Cam R | Seam-Cam T | Seam-Cam R | B-Root | Post-Root | Seam-Root | Seam-CHRGE | CHRGE |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        keys = [row[0] for row in METRICS if row[4] == "evaluator"]
        units = {row[0]: row[2] for row in METRICS}
        for method, values in group_payload["methods"].items():
            cells = [summary_cell(values[key], units[key]) for key in keys]
            lines.append(f"| {method} | " + " | ".join(cells) + " |")
        lines.extend(
            [
                "",
                "### Full-route paired improvements",
                "",
                (
                    "Positive values favour Bridge3R (full). Intervals resample the "
                    "shared finite capture pairs."
                ),
                "",
                "| Comparator | Metric | Support (cases/captures) | Full improvement [95% CI] | Win/tie/loss |",
                "|---|---|---:|---:|---:|",
            ]
        )
        labels = {row[0]: (row[1], row[2]) for row in METRICS}
        for comparator, metric_values in group_payload["full_paired_comparisons"].items():
            for key in keys:
                value = metric_values[key]
                low, high = value["capture_bootstrap_ci95"]
                wins = value["case_win_tie_loss"]
                label, unit = labels[key]
                lines.append(
                    f"| {comparator} | {label} | "
                    f"{value['paired_case_support']}/{value['paired_capture_support']} | "
                    f"{fmt(value['capture_macro_improvement'], unit)} "
                    f"[{fmt(low, unit)}, {fmt(high, unit)}] | "
                    f"{wins['wins']}/{wins['ties']}/{wins['losses']} |"
                )
        lines.extend(
            [
                "",
                "### Prediction-only boundary transform diagnostics",
                "",
                "| Method | Residual before | Residual after | Reduction |",
                "|---|---:|---:|---:|",
            ]
        )
        for method, values in group_payload["methods"].items():
            cells = [
                summary_cell(values[key], "m")
                for key in (
                    "Transform_residual_before_m",
                    "Transform_residual_after_m",
                    "Transform_residual_reduction_m",
                )
            ]
            lines.append(f"| {method} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Paired Full-route comparisons",
            "",
            (
                "The JSON retains Full-minus-comparator capture-cluster intervals for every "
                "metric and group. Positive improvement always favours Full; these are "
                "retrospective Test analyses and must not select a new formal route."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def tex_cell(value: dict[str, Any], unit: str) -> str:
    return (
        f"{fmt(value['capture_macro_mean'], unit)} "
        f"[{value['finite_case_support']}/{value['finite_capture_support']}]"
    )


def render_tex(payload: dict[str, Any]) -> str:
    units = {row[0]: row[2] for row in METRICS}
    overall = payload["groups"]["all"]["methods"]
    lines = [
        "% Generated by formal90_boundary_angle_statistics.py.",
        "% Cell support is finite cases/captures; all group denominators are retained in JSON/CSV.",
        r"\begin{table*}[t]",
        r"\centering\scriptsize",
        r"\caption{EgoHumans formal-90 boundary-specific metrics. Values are equal-capture means; brackets give finite case/capture support. Lower is better. Prediction-only transform residuals are reported separately and are not GT accuracy.}",
        r"\label{tab:egohumans-boundary-formal90}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\toprule",
        r"Method & B-Cam T & B-Cam R & Seam-Cam T & Seam-Cam R & B-Root & Post-Root & Seam-Root & Seam-CHRGE & CHRGE \\",
        r"\midrule",
    ]
    evaluator_keys = [row[0] for row in METRICS if row[4] == "evaluator"]
    for method, values in overall.items():
        lines.append(
            method
            + " & "
            + " & ".join(tex_cell(values[key], units[key]) for key in evaluator_keys)
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}}",
            r"\end{table*}",
            "",
            r"\begin{table*}[t]",
            r"\centering\scriptsize",
            r"\caption{Viewpoint-span boundary stress test on the same frozen EgoHumans formal-90 manifest. Values are equal-capture means and brackets give finite case/capture support. The $\geq150^\circ$ row is evaluator-defined before aggregation.}",
            r"\label{tab:egohumans-boundary-angle}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{llrrrrr}",
            r"\toprule",
            r"Group & Method & B-Cam T & Seam-Cam T & B-Root & Seam-Root & Seam-CHRGE \\",
            r"\midrule",
        ]
    )
    angle_keys = (
        "Boundary_camera_t_m",
        "Seam_camera_t_m",
        "Boundary_root_m",
        "Seam_root_m",
        "Seam_CHRGE_m",
    )
    for group in ("small", "medium", "large", "extreme", "ge150"):
        methods = payload["groups"][group]["methods"]
        for index, (method, values) in enumerate(methods.items()):
            group_label = (
                r"$\geq150^\circ$" if group == "ge150" else group.capitalize()
            ) if index == 0 else ""
            lines.append(
                group_label
                + " & "
                + method
                + " & "
                + " & ".join(tex_cell(values[key], units[key]) for key in angle_keys)
                + r" \\"
            )
        lines.append(r"\addlinespace")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}}",
            r"\end{table*}",
            "",
            r"\begin{table}[t]",
            r"\centering\small",
            r"\caption{Prediction-only matched-torso transform residual on EgoHumans formal-90. Values are equal-capture means in metres and brackets give finite case/capture support. Reduction is before minus after; higher is better. Strict Human3R has no boundary transform and is therefore not applicable.}",
            r"\label{tab:egohumans-transform-residual}",
            r"\resizebox{\linewidth}{!}{%",
            r"\begin{tabular}{lrrr}",
            r"\toprule",
            r"Method & Before $\downarrow$ & After $\downarrow$ & Reduction $\uparrow$ \\",
            r"\midrule",
        ]
    )
    for method, values in overall.items():
        lines.append(
            method
            + " & "
            + " & ".join(
                tex_cell(values[key], "m")
                for key in (
                    "Transform_residual_before_m",
                    "Transform_residual_after_m",
                    "Transform_residual_reduction_m",
                )
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("--bootstrap-samples must be at least 100")
    parent = args.parent.resolve()
    manifest_path = args.manifest.resolve()
    formal = load_manifest(manifest_path)
    expected_cases = set(formal)
    methods: dict[str, dict[str, dict[str, Any]]] = {}
    sources: dict[str, list[dict[str, str]]] = {}
    for label, route, kind, source_name in METHOD_SPECS:
        methods[label], sources[label] = load_method(
            parent, route, kind, source_name, expected_cases
        )
        for case_id, row in methods[label].items():
            manifest_row = formal[case_id]
            if row["sequence"] != str(manifest_row["sequence"]):
                raise ValueError(f"{label}/{case_id}: sequence mismatch")
            if row["capture"] != str(manifest_row["capture"]):
                raise ValueError(f"{label}/{case_id}: capture mismatch")
            if row["angle_stratum"] != str(manifest_row["angle_stratum"]):
                raise ValueError(f"{label}/{case_id}: angle-stratum mismatch")

    groups = group_members(formal)
    rng = np.random.default_rng(args.seed)
    group_results: dict[str, Any] = {}
    full = methods["Bridge3R (full)"]
    for group in GROUP_ORDER:
        members = groups[group]
        method_results: dict[str, Any] = {}
        for label, rows in methods.items():
            method_results[label] = {
                metric: summarize_values(
                    rows,
                    members,
                    metric,
                    samples=args.bootstrap_samples,
                    rng=rng,
                )
                for metric, _, _, _, _ in METRICS
            }
        paired_results: dict[str, Any] = {}
        for label, rows in methods.items():
            if label == "Bridge3R (full)":
                continue
            paired_results[label] = {
                metric: summarize_paired(
                    full,
                    rows,
                    members,
                    metric,
                    direction,
                    samples=args.bootstrap_samples,
                    rng=rng,
                )
                for metric, _, _, direction, _ in METRICS
            }
        group_results[group] = {
            "case_denominator": len(members),
            "capture_denominator": len(
                {
                    f"{formal[case]['sequence']}/{formal[case]['capture']}"
                    for case in members
                }
            ),
            "case_ids": sorted(members),
            "methods": method_results,
            "full_paired_comparisons": paired_results,
        }

    payload = {
        "schema_version": SCHEMA,
        "protocol": "Bridge3R-EgoHumans-formal90-v1",
        "analysis_scope": (
            "retrospective boundary-only and viewpoint-span aggregation of immutable Test "
            "evaluator reports; not a Test-time method-selection rule"
        ),
        "formal_manifest": str(manifest_path),
        "formal_manifest_sha256": sha256(manifest_path),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "case_count": 90,
        "capture_count": 27,
        "bootstrap": {
            "unit": "capture",
            "aggregation": (
                "mean finite cases within capture, then equal-weight mean/resampling over "
                "finite captures"
            ),
            "samples": args.bootstrap_samples,
            "seed": args.seed,
        },
        "metric_definitions": {
            metric: {
                "label": label,
                "unit": unit,
                "direction": direction,
                "source_class": source_class,
                "definition": METRIC_DEFINITIONS[metric],
            }
            for metric, label, unit, direction, source_class in METRICS
        },
        "source_reports": sources,
        "groups": group_results,
    }

    case_rows: list[dict[str, Any]] = []
    for method, rows in methods.items():
        for case_id in sorted(rows):
            formal_row = formal[case_id]
            output = {
                "method": method,
                "case_id": case_id,
                "sequence": rows[case_id]["sequence"],
                "capture": rows[case_id]["capture"],
                "angle_stratum": rows[case_id]["angle_stratum"],
                "angle_deg": float(formal_row["camera_rotation_span_deg_evaluator_only"]),
                "ge150": int(
                    float(formal_row["camera_rotation_span_deg_evaluator_only"]) >= 150.0
                ),
            }
            output.update(rows[case_id]["metrics"])
            case_rows.append(output)

    for path in (args.output_json, args.output_csv, args.output_md, args.output_tex):
        path.resolve().parent.mkdir(parents=True, exist_ok=True)
    atomic_text(
        args.output_json.resolve(),
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    )
    csv_path = args.output_csv.resolve()
    partial_csv = csv_path.with_suffix(csv_path.suffix + ".partial")
    with partial_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(case_rows[0]))
        writer.writeheader()
        writer.writerows(case_rows)
    os.replace(partial_csv, csv_path)
    atomic_text(args.output_md.resolve(), render_markdown(payload) + "\n")
    atomic_text(args.output_tex.resolve(), render_tex(payload))
    print(
        json.dumps(
            {
                "output_json": str(args.output_json.resolve()),
                "output_csv": str(csv_path),
                "output_md": str(args.output_md.resolve()),
                "output_tex": str(args.output_tex.resolve()),
                "case_rows": len(case_rows),
                "groups": {key: len(value) for key, value in groups.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
