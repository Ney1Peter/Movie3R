#!/usr/bin/env python3
"""Aggregate frozen v16 Harmony4D results at clip and sequence levels."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PRIMARY = "v16_harmony_safe"
PARENT = "v16_0_m15_geometry"
METHOD_ORDER = (
    "m0_strict_human3r",
    "m15_safe_boundary_permutation_causal_gru",
    PARENT,
    PRIMARY,
)
DISPLAY_NAMES = {
    "m0_strict_human3r": "Strict Human3R",
    "m15_safe_boundary_permutation_causal_gru": "Movie3R-v15",
    PARENT: "B0 + boundary ID (v16 parent)",
    PRIMARY: "Movie3R-v16 Harmony-Safe",
}
METRICS: tuple[tuple[str, str, bool], ...] = (
    ("W-MPJPE_mm", "W-MPJPE", False),
    ("WA-MPJPE_mm", "WA-MPJPE", False),
    ("MPJPE_mm", "MPJPE", False),
    ("PA-MPJPE_mm", "PA-MPJPE", False),
    ("MPVPE_mm", "MPVPE", False),
    ("Accel_mm_frame2", "Accel", False),
    ("RTE_H3R_percent", "RTE-H3R", False),
    ("ROE_joint_proxy_deg", "ROE-joint-proxy", False),
    ("Jitter_H3R", "Jitter", False),
    ("Foot_sliding_cm", "Foot sliding", False),
    ("ATE_Sim3_m", "ATE-Sim3", False),
    ("ATE_SE3_m", "ATE-SE3", False),
    ("Boundary_camera_t_m", "Boundary camera-t", False),
    ("Boundary_camera_R_deg", "Boundary camera-R", False),
    ("Boundary_root_m", "Boundary root", False),
    ("Post_root_m", "Post root", False),
    ("Seam_root_m", "Seam root", False),
    ("Seam_CHRGE_m", "Seam CHRGE", False),
    ("Pair_vector_m", "Pair vector", False),
    ("IDs", "IDs", False),
    ("IDF1", "IDF1", True),
    ("Coverage", "Coverage", True),
    ("Detection_precision", "Detection precision", True),
)
LITERATURE_MULTI_THUMBS = {
    "W-MPJPE_mm": 221.0,
    "WA-MPJPE_mm": 116.9,
    "MPJPE_mm": 215.9,
    "MPVPE_mm": 278.3,
    "Accel_mm_frame2": 17.4,
    "ATE_Sim3_m": 0.7,
    "IDs": 0.46,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--primary", default=PRIMARY)
    parser.add_argument("--parent", default=PARENT)
    parser.add_argument("--primary-display-name")
    parser.add_argument("--parent-display-name")
    parser.add_argument("--title", default="Movie3R-v16 Harmony4D frozen test summary")
    parser.add_argument("--test-used-for-parameter-selection", action="store_true")
    parser.add_argument(
        "--primary-ungated",
        action="store_true",
        help=(
            "Declare that the primary candidate applies its fixed boundary "
            "correction without a reliability gate. This suppresses gate "
            "acceptance/fallback statistics and preserves all primary-vs-parent "
            "differences in paired tests."
        ),
    )
    return parser.parse_args()


def finite(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def source_files(inputs: list[Path]) -> list[Path]:
    files = set()
    for source in inputs:
        if source.is_file():
            files.add(source.resolve())
        elif source.is_dir():
            files.update(source.resolve().rglob("*.json"))
        else:
            raise FileNotFoundError(source)
    output = []
    for path in sorted(files):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        if payload.get("schema_version") == "Movie3R-v16-Harmony4D-causal-stabilization-probe-v1":
            output.append(path)
    if not output:
        raise ValueError("No v16 per-sequence probe reports found")
    return output


def distribution(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {key: None for key in ("count", "mean", "median", "p90", "p95", "std")}
    return {
        "count": int(len(array)), "mean": float(array.mean()),
        "median": float(np.median(array)), "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)), "std": float(array.std()),
    }


def sequence_macro(rows: list[dict[str, Any]], metric: str) -> float | None:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = finite(row["metrics"].get(metric))
        if value is not None:
            grouped[str(row["sequence"])].append(value)
    values = [float(np.mean(group)) for group in grouped.values() if group]
    return float(np.mean(values)) if values else None


def hierarchical_bootstrap(
    rows: list[dict[str, Any]], metric: str, samples: int, rng: np.random.Generator,
) -> dict[str, Any]:
    grouped: dict[str, np.ndarray] = {}
    for sequence in sorted({str(row["sequence"]) for row in rows}):
        values = [
            finite(row["metrics"].get(metric)) for row in rows
            if str(row["sequence"]) == sequence
        ]
        grouped[sequence] = np.asarray([value for value in values if value is not None])
    grouped = {key: value for key, value in grouped.items() if len(value)}
    sequences = np.asarray(sorted(grouped), dtype=object)
    if not len(sequences):
        return {"low": None, "high": None, "unit": "sequence_then_clip", "samples": samples}
    estimates = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        chosen = rng.choice(sequences, size=len(sequences), replace=True)
        means = [
            float(rng.choice(grouped[str(sequence)], size=len(grouped[str(sequence)]), replace=True).mean())
            for sequence in chosen
        ]
        estimates[sample] = float(np.mean(means))
    return {
        "low": float(np.percentile(estimates, 2.5)),
        "high": float(np.percentile(estimates, 97.5)),
        "unit": "sequence_then_clip", "samples": samples,
    }


def exact_or_mc_sign_p(differences: np.ndarray, rng: np.random.Generator) -> dict[str, Any]:
    delta = np.asarray(differences, dtype=np.float64)
    delta = delta[np.isfinite(delta)]
    if not len(delta) or np.allclose(delta, 0.0):
        return {"p_two_sided": 1.0, "mode": "degenerate", "count": int(len(delta))}
    observed = abs(float(delta.mean()))
    if len(delta) <= 20:
        values = [
            abs(float(np.mean(delta * np.asarray(signs))))
            for signs in itertools.product((-1.0, 1.0), repeat=len(delta))
        ]
        p_value = float(np.mean(np.asarray(values) >= observed - 1e-12))
        mode = "exact_sign_randomization"
    else:
        samples = 200_000
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(samples, len(delta)))
        null = np.abs((signs * delta[None]).mean(axis=1))
        p_value = float((1 + np.count_nonzero(null >= observed)) / (samples + 1))
        mode = "monte_carlo_sign_randomization_200000"
    return {"p_two_sided": p_value, "mode": mode, "count": int(len(delta))}


def paired_sequence_test(
    primary: list[dict[str, Any]], baseline: list[dict[str, Any]], metric: str,
    higher: bool, rng: np.random.Generator,
    zero_all_fallback_sequences: bool = False,
) -> dict[str, Any]:
    def means(rows: list[dict[str, Any]]) -> dict[str, float]:
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            value = finite(row["metrics"].get(metric))
            if value is not None:
                grouped[str(row["sequence"])].append(value)
        return {key: float(np.mean(value)) for key, value in grouped.items() if value}

    first, second = means(primary), means(baseline)
    sequences = sorted(set(first) & set(second))
    exact_fallback_sequences: set[str] = set()
    if zero_all_fallback_sequences:
        gate_by_sequence: dict[str, list[bool]] = defaultdict(list)
        for row in primary:
            accepted = bool(
                row.get("diagnostics", {}).get("reliability_gate", {}).get("accepted")
            )
            gate_by_sequence[str(row["sequence"])].append(accepted)
        exact_fallback_sequences = {
            sequence for sequence, decisions in gate_by_sequence.items()
            if decisions and not any(decisions)
        }
    delta = np.asarray([
        0.0 if key in exact_fallback_sequences else first[key] - second[key]
        for key in sequences
    ], dtype=np.float64)
    oriented = delta if higher else -delta
    return {
        "unit": "sequence", "sequences": sequences,
        "exact_fallback_zeroed_sequences": sorted(exact_fallback_sequences),
        "primary_minus_baseline_mean": float(delta.mean()) if len(delta) else None,
        "primary_better_fraction": float(np.mean(oriented > 0)) if len(oriented) else None,
        **exact_or_mc_sign_p(delta, rng),
    }


def tex_table(
    summary: dict[str, Any], method_order: tuple[str, ...], display_names: dict[str, str],
) -> str:
    columns = (
        "W-MPJPE_mm", "WA-MPJPE_mm", "RTE_H3R_percent", "ATE_Sim3_m",
        "IDF1", "IDs", "Accel_mm_frame2",
    )
    labels = ("W-MPJPE", "WA-MPJPE", "RTE-H3R", "ATE", "IDF1", "IDs", "Accel")
    row_end = r" \\"
    lines = [
        "% Auto-generated; all methods use the same Movie3R-CS150-Harmony manifest.",
        "\\begin{tabular}{l" + "r" * len(columns) + "}", "\\toprule",
        "Method & " + " & ".join(labels) + row_end, "\\midrule",
    ]
    for method in method_order:
        if method not in summary:
            continue
        values = []
        for metric in columns:
            value = summary[method]["clip_macro"].get(metric)
            if value is None:
                values.append("--")
            elif metric == "IDF1":
                values.append(f"{100.0 * value:.1f}")
            elif metric == "ATE_Sim3_m":
                values.append(f"{value:.3f}")
            else:
                values.append(f"{value:.1f}")
        lines.append(display_names[method] + " & " + " & ".join(values) + row_end)
    lines.extend(("\\bottomrule", "\\end{tabular}", ""))
    return "\n".join(lines)


def markdown_summary(
    result: dict[str, Any], method_order: tuple[str, ...], display_names: dict[str, str],
) -> str:
    columns = ("W-MPJPE_mm", "WA-MPJPE_mm", "MPJPE_mm", "MPVPE_mm", "Accel_mm_frame2", "ATE_Sim3_m", "IDF1", "IDs")
    labels = ("W", "WA", "MPJPE", "MPVPE", "Accel", "ATE", "IDF1", "IDs")
    lines = [
        f"# {result['title']}", "",
        f"- Sequences: {result['sequence_count']}",
        f"- Cases: {result['case_count']} evaluable / {result['manifest_case_count']} preregistered (150 frames, 75+75)",
        (
            f"- Shared-internal evaluator unavailable: {result['evaluator_unavailable_count']} "
            "(legacy prediction-dependent initial-match fit; not an inference failure)"
        ),
        (
            f"- Reliability gate: {result['gate']['accepted']} accepted / "
            f"{result['gate']['fallback']} fallback"
            if result["gate"]["enabled"]
            else "- Reliability gate: not used; the fixed shared translation is applied to every evaluator-available case"
        ),
        "",
        "| Method | " + " | ".join(labels) + " |",
        "|---|" + "---:|" * len(labels),
    ]
    for method in method_order:
        if method not in result["methods"]:
            continue
        values = []
        for metric in columns:
            value = result["methods"][method]["clip_macro"].get(metric)
            if value is None:
                values.append("--")
            elif metric == "IDF1":
                values.append(f"{value:.3f}")
            elif metric == "ATE_Sim3_m":
                values.append(f"{value:.4f}")
            else:
                values.append(f"{value:.1f}")
        lines.append(f"| {display_names[method]} | " + " | ".join(values) + " |")
    lines.extend((
        "", "Multi-THuMBS literature values are stored only as protocol-different context; they are not treated as a same-manifest leaderboard comparison.", "",
    ))
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    primary = str(args.primary)
    parent = str(args.parent)
    method_order = (
        "m0_strict_human3r",
        "m15_safe_boundary_permutation_causal_gru",
        parent,
        primary,
    )
    method_order = tuple(dict.fromkeys(method_order))
    display_names = dict(DISPLAY_NAMES)
    display_names[parent] = args.parent_display_name or display_names.get(parent, parent)
    display_names[primary] = args.primary_display_name or display_names.get(primary, primary)
    files = source_files(args.inputs)
    rng = np.random.default_rng(int(args.seed))
    rows_by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sources = []
    skipped_cases = []
    manifest_case_count = 0
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("errors"):
            raise ValueError(f"Incomplete probe report: {path}: {payload['errors']}")
        sources.append(str(path))
        manifest_case_count += int(payload.get("case_count", 0))
        for skipped in payload.get("skipped_cases", []):
            # Older reports carried a `method_independent` field even though
            # these cases arise from a prediction-dependent initial-match fit
            # in the shared internal evaluator. Preserve that legacy detail
            # but do not repeat the incorrect publication-facing label.
            corrected = dict(skipped)
            if "method_independent" in corrected:
                corrected["legacy_method_independent_field"] = corrected["method_independent"]
            corrected["method_independent"] = False
            corrected["availability_scope"] = "shared_internal_legacy_initial_match"
            skipped_cases.append(corrected)
        for row in payload.get("reference_rows", []):
            if row.get("status") == "complete":
                rows_by_method[str(row["method"])].append(row)
        for row in payload.get("rows", []):
            if row.get("status") == "complete":
                rows_by_method[str(row["candidate"])].append(row)
    case_ids = {row["case_id"] for rows in rows_by_method.values() for row in rows}
    sequences = {row["sequence"] for rows in rows_by_method.values() for row in rows}
    for method in method_order:
        ids = [row["case_id"] for row in rows_by_method.get(method, [])]
        if len(ids) != len(case_ids) or len(ids) != len(set(ids)):
            raise ValueError(f"Method {method} has {len(ids)} rows for {len(case_ids)} cases")

    summary = {}
    for method in method_order:
        rows = rows_by_method[method]
        summary[method] = {
            "display_name": display_names[method],
            "case_count": len(rows), "sequence_count": len({row["sequence"] for row in rows}),
            "clip_macro": {
                metric: distribution([
                    value for row in rows
                    if (value := finite(row["metrics"].get(metric))) is not None
                ])["mean"]
                for metric, _, _ in METRICS
            },
            "sequence_macro": {
                metric: sequence_macro(rows, metric) for metric, _, _ in METRICS
            },
            "clip_distributions": {
                metric: distribution([
                    value for row in rows
                    if (value := finite(row["metrics"].get(metric))) is not None
                ])
                for metric, _, _ in METRICS
            },
            "hierarchical_bootstrap_ci95": {
                metric: hierarchical_bootstrap(rows, metric, int(args.bootstrap), rng)
                for metric, _, _ in METRICS
            },
        }

    significance = {}
    for method in method_order:
        if method == primary:
            continue
        significance[method] = {
            metric: paired_sequence_test(
                rows_by_method[primary], rows_by_method[method], metric, higher, rng,
                zero_all_fallback_sequences=(method == parent and not args.primary_ungated),
            )
            for metric, _, higher in METRICS
        }

    primary_rows = rows_by_method[primary]
    if args.primary_ungated:
        gate = {
            "enabled": False,
            "applied_cases": len(primary_rows),
            "accepted": None,
            "fallback": None,
            "strata": None,
        }
    else:
        accepted = [
            row for row in primary_rows
            if bool(row.get("diagnostics", {}).get("reliability_gate", {}).get("accepted"))
        ]
        parent_by_case = {row["case_id"]: row for row in rows_by_method[parent]}
        gate_strata = {}
        for name, rows in (("accepted", accepted), ("fallback", [row for row in primary_rows if row not in accepted])):
            gate_strata[name] = {
                "case_count": len(rows),
                "mean_delta_primary_minus_parent": {
                    metric: distribution([
                        float(row["metrics"][metric]) - float(parent_by_case[row["case_id"]]["metrics"][metric])
                        for row in rows
                        if finite(row["metrics"].get(metric)) is not None
                        and finite(parent_by_case[row["case_id"]]["metrics"].get(metric)) is not None
                    ])["mean"]
                    for metric, _, _ in METRICS
                },
            }
        gate = {
            "enabled": True,
            "applied_cases": len(primary_rows),
            "accepted": len(accepted),
            "fallback": len(primary_rows) - len(accepted),
            "strata": gate_strata,
        }

    result = {
        "schema_version": "Movie3R-v16-Harmony4D-multisequence-summary-v1",
        "sources": sources, "method": primary, "parent": parent,
        "title": str(args.title),
        "case_count": len(case_ids), "manifest_case_count": manifest_case_count,
        "evaluator_unavailable_count": len(skipped_cases),
        "skipped_cases": skipped_cases, "sequence_count": len(sequences),
        "sequences": sorted(sequences),
        "protocol": {
            "frames": 150, "pre_frames": 75, "post_frames": 75,
            "aggregation_primary": "clip_macro",
            "uncertainty": "hierarchical sequence-then-clip bootstrap",
            "paired_test_unit": "sequence",
            "test_used_for_parameter_selection": bool(args.test_used_for_parameter_selection),
            "exclusion_rule": "shared-internal legacy evaluator unavailable only; not an external inference failure",
            "evaluator_unavailable_scope": "prediction-dependent initial-match fit in the legacy shared internal evaluator",
        },
        "gate": gate,
        "methods": summary, "paired_significance_vs_primary": significance,
        "multi_thumbs_literature_reference": {
            "values": LITERATURE_MULTI_THUMBS,
            "contract": "context only; exact Multi-THuMBS manifest/evaluator unavailable",
        },
        "metric_caveats": {
            "RTE_H3R_percent": "exact NumPy port of repository Human3R/WHAM rigid path-normalized RTE",
            "Jitter_H3R": "exact third-difference fps^3/10 convention from repository Human3R evaluator",
            "Foot_sliding_cm": "SMPL-6890 four-foot-vertex contact metric, converted to cm",
            "ROE_joint_proxy_deg": "joint-derived torso Kabsch proxy; not official HumanMM SMPL-root ROE",
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    fields = ["case_id", "sequence", "capture", "angle_stratum", "person_count", "method", *[key for key, _, _ in METRICS]]
    with (args.output / "case_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method in method_order:
            for row in sorted(rows_by_method[method], key=lambda value: value["case_id"]):
                writer.writerow({
                    "case_id": row["case_id"], "sequence": row["sequence"],
                    "capture": row.get("capture"), "angle_stratum": row.get("angle_stratum"),
                    "person_count": row.get("person_count"), "method": method,
                    **{key: row["metrics"].get(key) for key, _, _ in METRICS},
                })
    (args.output / "main_table.tex").write_text(
        tex_table(summary, method_order, display_names), encoding="utf-8"
    )
    (args.output / "SUMMARY.md").write_text(
        markdown_summary(result, method_order, display_names), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output.resolve()), "cases": len(case_ids),
        "sequences": len(sequences), "gate_enabled": gate["enabled"],
        "W_primary": summary[primary]["clip_macro"]["W-MPJPE_mm"],
        "W_parent": summary[parent]["clip_macro"]["W-MPJPE_mm"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
