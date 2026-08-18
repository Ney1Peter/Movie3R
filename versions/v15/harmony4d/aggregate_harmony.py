#!/usr/bin/env python3
"""Aggregate frozen Harmony4D case reports into paper-ready statistics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wilcoxon


DEFAULT_PRIMARY = "m10_observability_safe_oracle"
METHOD_ORDER = (
    "m0_strict_human3r",
    "m1_clean_reset",
    "m2_no_v9_raw_se3",
    "m3_b0_only",
    "m4_b0_identity",
    "m5_b0_identity_brtc",
    "m6_b0_identity_brtc_c1",
    "m7_full_v15_oracle",
    "m8_full_v15_causal_gru",
    "m9_full_v15_static_logistic",
    "m10_observability_safe_oracle",
    "m11_observability_safe_causal_gru",
    "m12_observability_safe_static_logistic",
)

# path, output label, higher-is-better
METRICS: tuple[tuple[tuple[str, ...], str, bool], ...] = (
    (("multi_thumbs_named_provisional", "w_mpjpe_mm", "mean"), "W-MPJPE_mm", False),
    (("multi_thumbs_named_provisional", "wa_mpjpe_mm", "mean"), "WA-MPJPE_mm", False),
    (("multi_thumbs_named_provisional", "mpjpe_mm", "mean"), "MPJPE_mm", False),
    (("multi_thumbs_named_provisional", "mpvpe_mm", "mean"), "MPVPE_mm", False),
    (("multi_thumbs_named_provisional", "accel_delta2_mm_per_frame2", "mean"), "Accel_mm_frame2", False),
    (("multi_thumbs_named_provisional", "ate_sim3_m", "mean"), "ATE_Sim3_m", False),
    (("multi_thumbs_named_provisional", "ate_se3_m", "mean"), "ATE_SE3_m", False),
    (("multi_thumbs_named_provisional", "ate_metric_initial_se3_m", "mean"), "ATE_metric_initial_SE3_m", False),
    (("camera", "first_post_translation_m"), "Boundary_camera_t_m", False),
    (("camera", "first_post_rotation_deg"), "Boundary_camera_R_deg", False),
    (("camera", "boundary_rpe_translation_m"), "Boundary_RPE_t_m", False),
    (("camera", "boundary_rpe_rotation_deg"), "Boundary_RPE_R_deg", False),
    (("fixed_world", "first_post_root_m", "mean"), "Boundary_root_m", False),
    (("fixed_world", "post_root_m", "mean"), "Post_root_m", False),
    (("fixed_world", "post_vertex_m", "mean"), "Post_vertex_m", False),
    (("camera_human_relative", "first_post_root_gauge_m", "mean"), "Boundary_CHRGE_m", False),
    (("pairwise_layout", "first_post_vector_m", "mean"), "Boundary_pair_vector_m", False),
    (("cut_seam", "camera_translation_excess_m"), "Seam_camera_t_m", False),
    (("cut_seam", "camera_rotation_excess_deg"), "Seam_camera_R_deg", False),
    (("cut_seam", "root_excess_m"), "Seam_root_m", False),
    (("cut_seam", "camera_human_relative_excess_m"), "Seam_CHRGE_m", False),
    (("identity", "ids_total"), "IDs", False),
    (("identity", "idf1"), "IDF1", True),
    (("coverage", "coverage"), "Coverage", True),
    (("coverage", "precision"), "Detection_precision", True),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--primary", default=DEFAULT_PRIMARY)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--permutations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260818)
    return parser.parse_args()


def nested(value: dict[str, Any], path: tuple[str, ...]) -> float | None:
    current: Any = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    if current is None:
        return None
    try:
        number = float(current)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def input_reports(paths: list[Path]) -> list[tuple[Path, dict[str, Any]]]:
    files: set[Path] = set()
    for path in paths:
        if path.is_file():
            files.add(path.resolve())
        elif path.is_dir():
            files.update(item.resolve() for item in path.rglob("*.json"))
        else:
            raise FileNotFoundError(path)
    reports = []
    for path in sorted(files):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if report.get("schema_version") == "Movie3R-Harmony4D-evaluation-v1":
            reports.append((path, report))
    if not reports:
        raise ValueError("No Movie3R-Harmony4D evaluation reports found")
    case_ids = [report["case_id"] for _, report in reports]
    if len(case_ids) != len(set(case_ids)):
        duplicates = sorted({case for case in case_ids if case_ids.count(case) > 1})
        raise ValueError(f"Duplicate case reports: {duplicates}")
    return reports


def distribution(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {key: None for key in ("count", "mean", "median", "p90", "p95", "std")}
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "std": float(array.std()),
    }


def bootstrap_ci(
    rows: list[dict[str, Any]], metric: str, samples: int, rng: np.random.Generator
) -> dict[str, Any]:
    valid = [row for row in rows if row.get(metric) is not None]
    if not valid:
        return {"low": None, "high": None, "unit": None, "samples": samples}
    by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid:
        by_sequence[str(row["sequence"])].append(row)
    sequences = sorted(by_sequence)
    estimates = np.empty(samples, dtype=np.float64)
    if len(sequences) >= 2:
        for index in range(samples):
            selected = rng.choice(sequences, size=len(sequences), replace=True)
            sequence_means = [
                np.mean([float(row[metric]) for row in by_sequence[str(sequence)]])
                for sequence in selected
            ]
            estimates[index] = float(np.mean(sequence_means))
        unit = "sequence"
    else:
        values = np.asarray([float(row[metric]) for row in valid])
        for index in range(samples):
            estimates[index] = float(rng.choice(values, size=len(values), replace=True).mean())
        unit = "clip"
    return {
        "low": float(np.percentile(estimates, 2.5)),
        "high": float(np.percentile(estimates, 97.5)),
        "unit": unit,
        "samples": samples,
    }


def paired_test(
    primary: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    metric: str,
    higher_is_better: bool,
    permutations: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    first = {row["case_id"]: row.get(metric) for row in primary}
    second = {row["case_id"]: row.get(metric) for row in baseline}
    ids = sorted(
        case for case in set(first).intersection(second)
        if first[case] is not None and second[case] is not None
    )
    if not ids:
        return {"count": 0, "mean_primary_minus_baseline": None, "p_wilcoxon": None, "p_permutation": None}
    delta = np.asarray([float(first[case]) - float(second[case]) for case in ids])
    oriented_improvement = delta if higher_is_better else -delta
    if np.allclose(delta, 0.0):
        p_wilcoxon = 1.0
    else:
        p_wilcoxon = float(wilcoxon(delta, zero_method="zsplit", alternative="two-sided").pvalue)
    observed = abs(float(delta.mean()))
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(permutations, len(delta)))
    null = np.abs((signs * delta[None]).mean(axis=1))
    p_permutation = float((1 + np.count_nonzero(null >= observed)) / (permutations + 1))
    return {
        "count": len(ids),
        "mean_primary_minus_baseline": float(delta.mean()),
        "mean_oriented_improvement": float(oriented_improvement.mean()),
        "primary_better_fraction": float(np.mean(oriented_improvement > 0)),
        "p_wilcoxon": p_wilcoxon,
        "p_permutation": p_permutation,
    }


def latex_table(summary: dict[str, Any], methods: list[str]) -> str:
    columns = ("W-MPJPE_mm", "WA-MPJPE_mm", "MPJPE_mm", "MPVPE_mm", "ATE_Sim3_m", "IDs", "Coverage")
    labels = ("W-MPJPE", "WA-MPJPE", "MPJPE", "MPVPE", "ATE", "IDs", "Cov.")
    lines = [
        "% Auto-generated by aggregate_harmony.py; protocol-local values.",
        "\\begin{tabular}{l" + "r" * len(columns) + "}",
        "\\toprule",
        "Method & " + " & ".join(labels) + " \\\\",
        "\\midrule",
    ]
    for method in methods:
        row = summary[method]["clip_macro"]
        values = []
        for column in columns:
            value = row.get(column)
            if value is None:
                values.append("--")
            elif column == "Coverage":
                values.append(f"{100.0 * value:.1f}")
            elif column in {"ATE_Sim3_m", "IDs"}:
                values.append(f"{value:.3f}")
            else:
                values.append(f"{value:.1f}")
        lines.append(method.replace("_", "\\_") + " & " + " & ".join(values) + " \\\\")
    lines.extend(("\\bottomrule", "\\end{tabular}", ""))
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.bootstrap < 100 or args.permutations < 100:
        raise ValueError("bootstrap and permutation counts must each be >= 100")
    inputs = input_reports(args.metrics)
    rng = np.random.default_rng(args.seed)
    rows_by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    case_rows: list[dict[str, Any]] = []
    detector_rows = []
    gate_rows = []
    provenance = []
    for path, report in inputs:
        record = report["record"]
        provenance.append({
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
        for method, result in report["methods"].items():
            row: dict[str, Any] = {
                "case_id": report["case_id"],
                "sequence": record["sequence"],
                "capture": record.get("capture", Path(record["capture_relative"]).name),
                "angle_stratum": record["angle_stratum"],
                "angle_deg": record["camera_rotation_span_deg_evaluator_only"],
                "method": method,
            }
            for metric_path, label, _ in METRICS:
                row[label] = nested(result, metric_path)
                if metric_path[-1] == "mean":
                    row[label + "__weight"] = nested(result, metric_path[:-1] + ("count",))
                elif label in {"Coverage", "IDF1"}:
                    row[label + "__weight"] = nested(
                        result, ("coverage", "visible_gt_person_frames")
                    )
                elif label == "Detection_precision":
                    row[label + "__weight"] = nested(
                        result, ("coverage", "predicted_person_frames")
                    )
                else:
                    row[label + "__weight"] = 1.0
            row["__idtp"] = nested(result, ("identity", "idtp"))
            row["__idfp"] = nested(result, ("identity", "idfp"))
            row["__idfn"] = nested(result, ("identity", "idfn"))
            rows_by_method[method].append(row)
            case_rows.append(row)
        for detector_name, values in report.get("detectors", {}).items():
            detector_rows.append({
                "case_id": report["case_id"],
                "sequence": record["sequence"],
                "detector": detector_name,
                **values,
            })
        safe = report.get("observability_safe_gate")
        if safe:
            gate_rows.append({"case_id": report["case_id"], "sequence": record["sequence"], **safe})

    methods = [method for method in METHOD_ORDER if method in rows_by_method]
    methods.extend(sorted(set(rows_by_method) - set(methods)))
    if args.primary not in rows_by_method:
        raise KeyError(f"Primary method absent: {args.primary}")
    summary: dict[str, Any] = {}
    for method in methods:
        rows = rows_by_method[method]
        clip_macro = {
            label: distribution([float(row[label]) for row in rows if row[label] is not None])["mean"]
            for _, label, _ in METRICS
        }
        sequence_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            sequence_rows[str(row["sequence"])].append(row)
        sequence_macro = {}
        for _, label, _ in METRICS:
            values = [
                np.mean([float(row[label]) for row in group if row[label] is not None])
                for group in sequence_rows.values()
                if any(row[label] is not None for row in group)
            ]
            sequence_macro[label] = float(np.mean(values)) if values else None
        micro = {}
        for _, label, _ in METRICS:
            valid = [
                row for row in rows
                if row[label] is not None and row.get(label + "__weight") is not None
                and float(row[label + "__weight"]) > 0
            ]
            denominator = sum(float(row[label + "__weight"]) for row in valid)
            micro[label] = (
                sum(float(row[label]) * float(row[label + "__weight"]) for row in valid)
                / denominator
                if denominator > 0
                else None
            )
        idtp = sum(float(row["__idtp"]) for row in rows if row["__idtp"] is not None)
        idfp = sum(float(row["__idfp"]) for row in rows if row["__idfp"] is not None)
        idfn = sum(float(row["__idfn"]) for row in rows if row["__idfn"] is not None)
        micro["IDF1"] = 2.0 * idtp / max(2.0 * idtp + idfp + idfn, 1.0)
        summary[method] = {
            "case_count": len(rows),
            "sequence_count": len(sequence_rows),
            "clip_macro": clip_macro,
            "sequence_macro": sequence_macro,
            "person_frame_weighted_micro": micro,
            "totals": {
                "IDs_total": int(sum(float(row["IDs"]) for row in rows if row["IDs"] is not None)),
            },
            "case_distributions": {
                label: distribution([float(row[label]) for row in rows if row[label] is not None])
                for _, label, _ in METRICS
            },
            "confidence_intervals_95": {
                label: bootstrap_ci(rows, label, args.bootstrap, rng)
                for _, label, _ in METRICS
            },
        }

    significance = {}
    primary_rows = rows_by_method[args.primary]
    for method in methods:
        if method == args.primary:
            continue
        significance[method] = {
            label: paired_test(
                primary_rows, rows_by_method[method], label, higher,
                args.permutations, rng,
            )
            for _, label, higher in METRICS
        }

    detector_summary = {}
    for name in sorted({row["detector"] for row in detector_rows}):
        values = [row for row in detector_rows if row["detector"] == name]
        tp, fp, fn = (sum(int(row[key]) for row in values) for key in ("tp", "fp", "fn"))
        detector_summary[name] = {
            "case_count": len(values),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "first_positive_boundary_rate": float(np.mean([bool(row["first_positive_is_boundary"]) for row in values])),
            "latency_seconds_mean": float(np.mean([float(row["latency_seconds"]) for row in values])),
        }

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "Movie3R-Harmony4D-aggregate-v1",
        "primary_method": args.primary,
        "case_count": len(inputs),
        "sequences": sorted({report["record"]["sequence"] for _, report in inputs}),
        "methods": methods,
        "summary": summary,
        "paired_significance_primary_vs_baseline": significance,
        "detectors": detector_summary,
        "gate_rows": gate_rows,
        "bootstrap": {"samples": args.bootstrap, "seed": args.seed},
        "permutation_test": {"samples": args.permutations, "seed": args.seed},
        "input_reports": provenance,
        "aggregation_note": (
            "clip_macro averages case means; sequence_macro first averages within sequence. "
            "Frame/person-level distribution summaries remain in each immutable case report."
        ),
    }
    (output / "aggregate.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    fieldnames = [
        name for name in case_rows[0]
        if not name.endswith("__weight") and not name.startswith("__")
    ]
    with (output / "case_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            [{name: row.get(name) for name in fieldnames} for row in case_rows]
        )
    (output / "main_table.tex").write_text(latex_table(summary, methods), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "cases": len(inputs),
        "methods": methods,
        "sequences": report["sequences"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
