#!/usr/bin/env python3
"""Aggregate frozen Test evaluations and generate the protocol return tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np


METHODS = ("r0", "r1", "r2", "r3")
DISPLAY = {
    "r0": "Human3R per-shot reset",
    "r1": "+ pelvis translation",
    "r2": "+ human-joint SE(3)",
    "r3": "+ scene registration + ICP",
    "shot3r": "Shot3R",
}
ACCESS = {
    "r0": "per-shot online",
    "r1": "boundary-time",
    "r2": "boundary-time",
    "r3": "offline post-hoc",
    "shot3r": "streaming",
}
METRICS = (
    "w_mpjpe_mm", "wa_mpjpe_mm", "mpjpe_mm", "mpvpe_mm", "accel_mm_frame2",
    "ate_sim3_m", "ate_se3_m", "ate_m", "idf1", "coverage", "ids",
    "boundary_camera_translation_m", "boundary_camera_rotation_deg",
    "fixed_root_m", "post_root_m", "seam_root_m", "seam_camera_translation_m",
    "seam_camera_rotation_deg", "fit_runtime_seconds",
)
SHOT3R_SOURCES = {
    "egobody": {
        "csv": "Movie3R/output/v20_egobody/formal/test/aggregate/case_metrics.csv",
        "method": "v19_ungated_translation_b050",
        "summary": "Movie3R/output/v20_egobody/formal/test/aggregate/summary.json",
    },
    "egohumans": {
        "csv": "Movie3R/output/v19_egohumans/test/summary/case_metrics.csv",
        "method": "v19_egohumans_frozen",
        "summary": "Movie3R/output/v19_egohumans/test/summary/summary.json",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260915)
    return parser.parse_args()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(text, encoding="utf-8")
    os.replace(partial, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def summary_mean(section: dict[str, Any], key: str) -> float | None:
    value = section.get(key)
    return finite(value.get("mean")) if isinstance(value, dict) else finite(value)


def metric_row(result: dict[str, Any], registration: dict[str, Any], dataset: str) -> dict[str, Any]:
    named = result.get("multi_thumbs_named_provisional", {})
    camera = result.get("camera", {})
    fixed = result.get("fixed_world", {})
    seam = result.get("cut_seam", {})
    identity = result.get("identity", {})
    coverage = result.get("coverage", {})
    ate_sim3 = summary_mean(named, "ate_sim3_m")
    ate_se3 = summary_mean(named, "ate_se3_m")
    return {
        "w_mpjpe_mm": summary_mean(named, "w_mpjpe_mm"),
        "wa_mpjpe_mm": summary_mean(named, "wa_mpjpe_mm"),
        "mpjpe_mm": summary_mean(named, "mpjpe_mm"),
        "mpvpe_mm": summary_mean(named, "mpvpe_mm"),
        "accel_mm_frame2": summary_mean(named, "accel_delta2_mm_per_frame2"),
        "ate_sim3_m": ate_sim3,
        "ate_se3_m": ate_se3,
        "ate_m": ate_sim3 if dataset == "egobody" else ate_se3,
        "idf1": finite(identity.get("idf1")),
        "coverage": finite(coverage.get("coverage")),
        "ids": finite(identity.get("ids_total")),
        "boundary_camera_translation_m": finite(camera.get("boundary_rpe_translation_m")),
        "boundary_camera_rotation_deg": finite(camera.get("boundary_rpe_rotation_deg")),
        "fixed_root_m": summary_mean(fixed, "root_m"),
        "post_root_m": summary_mean(fixed, "post_root_m"),
        "seam_root_m": finite(seam.get("root_excess_m")),
        "seam_camera_translation_m": finite(seam.get("camera_translation_excess_m")),
        "seam_camera_rotation_deg": finite(seam.get("camera_rotation_excess_deg")),
        "fit_runtime_seconds": finite(registration.get("runtime_seconds")),
    }


def load_test_rows(root: Path, dataset: str, expected: int) -> list[dict[str, Any]]:
    directory = root / "metrics/evaluations" / dataset / "test"
    files = sorted(directory.glob("*.evaluation.json"))
    if len(files) != expected:
        raise ValueError(f"{dataset} evaluation count {len(files)} != {expected}")
    rows = []
    for path in files:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("errors"):
            raise ValueError(f"method evaluation errors in {path}: {report['errors']}")
        case_id = str(report["case_id"])
        record = report["record_runtime_fields"]
        metadata = report.get("evaluator_metadata", {})
        for method in METHODS:
            if method not in report.get("methods", {}) or method not in report.get("registration", {}):
                raise ValueError(f"incomplete {method} evaluation: {path}")
            registration = report["registration"][method]
            rows.append({
                "dataset": dataset,
                "case_id": case_id,
                "method": method,
                "recording": record.get("recording"),
                "capture": record.get("capture"),
                "sequence": record.get("sequence"),
                "angle_stratum": metadata.get("angle_stratum"),
                "angle_deg": finite(metadata.get("camera_rotation_span_deg")),
                "person_count": metadata.get("person_count"),
                "status": str(registration.get("status")),
                "failure_reason": registration.get("failure_reason"),
                "diagnostics": registration.get("diagnostics", {}),
                "world_alignment_available": report["methods"][method].get(
                    "metric_availability", {}
                ).get("world_alignment", True),
                "metric_unavailability_reason": report["methods"][method].get(
                    "metric_availability", {}
                ).get("reason"),
                "metrics": metric_row(report["methods"][method], registration, dataset),
            })
    return rows


def load_shot3r_rows(workspace: Path, dataset: str, metadata: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    source = SHOT3R_SOURCES[dataset]
    path = workspace / source["csv"]
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            if raw.get("method", raw.get("name")) != source["method"]:
                continue
            case_id = str(raw["case_id"])
            meta = metadata[case_id]
            metric = {
                "w_mpjpe_mm": finite(raw.get("W-MPJPE_mm")),
                "wa_mpjpe_mm": finite(raw.get("WA-MPJPE_mm")),
                "mpjpe_mm": finite(raw.get("MPJPE_mm")),
                "mpvpe_mm": finite(raw.get("MPVPE_mm")),
                "accel_mm_frame2": finite(raw.get("Accel_mm_frame2")),
                "ate_sim3_m": finite(raw.get("ATE_Sim3_m")),
                "ate_se3_m": finite(raw.get("ATE_SE3_m")),
                "idf1": finite(raw.get("IDF1")),
                "coverage": finite(raw.get("Coverage")),
                "ids": finite(raw.get("IDs")),
                "boundary_camera_translation_m": finite(raw.get("Boundary_camera_t_m")),
                "boundary_camera_rotation_deg": finite(raw.get("Boundary_camera_R_deg")),
                "fixed_root_m": finite(raw.get("CHRGE_m")),
                "post_root_m": finite(raw.get("Post_root_m")),
                "seam_root_m": finite(raw.get("Seam_root_m")),
                "seam_camera_translation_m": finite(raw.get("Seam_camera_t_m")),
                "seam_camera_rotation_deg": finite(raw.get("Seam_camera_R_deg")),
                "fit_runtime_seconds": None,
            }
            metric["ate_m"] = metric["ate_sim3_m"] if dataset == "egobody" else metric["ate_se3_m"]
            rows.append({
                "dataset": dataset, "case_id": case_id, "method": "shot3r",
                "recording": meta.get("recording"), "capture": meta.get("capture"),
                "sequence": meta.get("sequence"), "angle_stratum": meta.get("angle_stratum"),
                "angle_deg": meta.get("angle_deg"), "person_count": meta.get("person_count"),
                "status": "streaming", "failure_reason": None, "diagnostics": {}, "metrics": metric,
                "world_alignment_available": metric["w_mpjpe_mm"] is not None,
                "metric_unavailability_reason": None,
            })
    if len(rows) != len(metadata):
        raise ValueError(f"Shot3R {dataset} common cases {len(rows)} != {len(metadata)}")
    return rows


def unit_key(row: dict[str, Any], dataset: str) -> str:
    return str(row["recording"] if dataset == "egobody" else row["capture"])


def mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def aggregate_metric(rows: list[dict[str, Any]], metric: str, dataset: str, unit_macro: bool) -> dict[str, Any]:
    valid = [(row, finite(row["metrics"].get(metric))) for row in rows]
    valid = [(row, value) for row, value in valid if value is not None]
    case_mean = mean([value for _, value in valid])
    by_unit: dict[str, list[float]] = defaultdict(list)
    for row, value in valid:
        by_unit[unit_key(row, dataset)].append(value)
    unit_values = [float(np.mean(values)) for values in by_unit.values()]
    return {
        "mean": mean(unit_values) if unit_macro else case_mean,
        "case_macro_mean": case_mean,
        "unit_macro_mean": mean(unit_values),
        "valid_case_count": len(valid),
        "valid_unit_count": len(unit_values),
    }


def aggregate_methods(rows: list[dict[str, Any]], dataset: str) -> dict[str, Any]:
    output = {}
    # The frozen publication contract uses recording macro for EgoBody and the
    # existing case macro headline for EgoHumans. Capture macro remains explicit.
    publication_unit_macro = dataset == "egobody"
    for method in (*METHODS, "shot3r"):
        selected = [row for row in rows if row["method"] == method]
        if not selected:
            continue
        success = None if method in {"r0", "shot3r"} else sum(row["status"] == "ok" for row in selected) / len(selected)
        output[method] = {
            "case_count": len(selected),
            "unit_count": len({unit_key(row, dataset) for row in selected}),
            "publication_aggregation": "recording_macro" if publication_unit_macro else "case_macro",
            "metrics": {
                metric: aggregate_metric(selected, metric, dataset, publication_unit_macro)
                for metric in METRICS
            },
            "registration_success_rate": success,
            "status_counts": dict(Counter(row["status"] for row in selected)),
            "evaluator_unavailable_case_count": sum(
                not row.get("world_alignment_available", True) for row in selected
            ),
        }
    return output


def strata_specs(dataset: str) -> list[tuple[str, Callable[[dict[str, Any]], bool]]]:
    if dataset == "egobody":
        return [("all", lambda row: True)] + [
            (name, lambda row, value=name: row["angle_stratum"] == value)
            for name in ("small", "medium", "extreme")
        ]
    return [
        ("all", lambda row: True),
        ("small", lambda row: row["angle_stratum"] == "small"),
        ("medium", lambda row: row["angle_stratum"] == "medium"),
        ("large", lambda row: row["angle_stratum"] == "large"),
        ("extreme", lambda row: row["angle_stratum"] == "extreme"),
        ("ge150deg", lambda row: row["angle_deg"] is not None and row["angle_deg"] >= 150.0),
    ]


def angle_tables(rows: list[dict[str, Any]], dataset: str) -> dict[str, Any]:
    output = {}
    for name, predicate in strata_specs(dataset):
        selected = [row for row in rows if predicate(row)]
        cases = {row["case_id"] for row in selected if row["method"] == "r0"}
        output[name] = {
            "case_count": len(cases),
            "methods": aggregate_methods(selected, dataset),
        }
    return output


def bootstrap_unit_differences(values: dict[str, list[float]], samples: int, seed: int) -> dict[str, Any]:
    units = sorted(values)
    unit_values = np.asarray([np.mean(values[unit]) for unit in units], dtype=np.float64)
    if not len(unit_values):
        return {"mean": None, "ci95": [None, None], "unit_count": 0, "samples": samples}
    rng = np.random.default_rng(seed)
    draws = unit_values[rng.integers(0, len(unit_values), size=(samples, len(unit_values)))].mean(axis=1)
    return {
        "mean": float(unit_values.mean()),
        "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        "unit_count": len(units), "samples": samples,
    }


def paired(rows: list[dict[str, Any]], dataset: str, samples: int, seed: int) -> dict[str, Any]:
    lookup = {(row["case_id"], row["method"]): row for row in rows}
    output = {}
    for method_index, method in enumerate(("r1", "r2", "r3")):
        result = {}
        for metric_index, metric in enumerate(("w_mpjpe_mm", "wa_mpjpe_mm", "ate_m")):
            for comparison in ("r0", "shot3r"):
                by_unit: dict[str, list[float]] = defaultdict(list)
                case_count = 0
                for case_id in sorted({row["case_id"] for row in rows if row["method"] == method}):
                    candidate = finite(lookup[(case_id, method)]["metrics"].get(metric))
                    reference = finite(lookup[(case_id, comparison)]["metrics"].get(metric))
                    if candidate is None or reference is None:
                        continue
                    # Positive always means the named comparator has lower error:
                    # R0-method for vs-R0; method-Shot3R for vs-Shot3R.
                    delta = reference - candidate if comparison == "r0" else candidate - reference
                    by_unit[unit_key(lookup[(case_id, method)], dataset)].append(delta)
                    case_count += 1
                label = f"{metric}_vs_{comparison}"
                result[label] = {
                    **bootstrap_unit_differences(
                        by_unit, samples, seed + method_index * 100 + metric_index * 10 + (comparison == "shot3r")
                    ),
                    "common_case_count": case_count,
                    "delta_definition": "R0 - method" if comparison == "r0" else "method - Shot3R",
                    "positive_means_comparator_lower_error": True,
                }
        output[method] = result
    return output


def distribution(values: list[Any]) -> dict[str, Any]:
    array = np.asarray([value for value in (finite(item) for item in values) if value is not None], dtype=np.float64)
    if not len(array):
        return {"count": 0, "mean": None, "median": None, "p90": None, "p95": None}
    return {
        "count": len(array), "mean": float(array.mean()), "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)), "p95": float(np.percentile(array, 95)),
    }


def failure_analysis(rows: list[dict[str, Any]], dataset: str) -> dict[str, Any]:
    output = {}
    for method in ("r1", "r2", "r3"):
        selected = [row for row in rows if row["method"] == method]
        catastrophic_states = []
        for row in selected:
            translation = finite(row["metrics"].get("boundary_camera_translation_m"))
            rotation = finite(row["metrics"].get("boundary_camera_rotation_deg"))
            state = None if translation is None and rotation is None else (
                (translation is not None and translation > 1.0)
                or (rotation is not None and rotation > 30.0)
            )
            catastrophic_states.append((row, state))
        catastrophic_valid = [row for row, state in catastrophic_states if state is not None]
        catastrophic = [
            row for row, state in catastrophic_states if state is True
        ]
        output[method] = {
            "case_count": len(selected),
            "algorithmic_failure_count": sum(row["status"] != "ok" for row in selected),
            "algorithmic_failure_rate": sum(row["status"] != "ok" for row in selected) / len(selected),
            "status_counts": dict(Counter(row["status"] for row in selected)),
            "catastrophic_evaluation_count": len(catastrophic),
            "catastrophic_evaluation_valid_count": len(catastrophic_valid),
            "catastrophic_evaluation_unavailable_count": len(selected) - len(catastrophic_valid),
            "catastrophic_evaluation_rate": (
                len(catastrophic) / len(catastrophic_valid)
                if catastrophic_valid else None
            ),
            "catastrophic_rule": "boundary translation > 1 m OR boundary rotation > 30 deg; diagnostic only",
            "diagnostic_distributions": {
                key: distribution([row["diagnostics"].get(key) for row in selected])
                for key in (
                    "scene_point_count_pre", "scene_point_count_post", "correspondence_count",
                    "inlier_ratio", "fit_residual", "global_fitness", "global_residual",
                    "matched_person_count", "runtime_seconds",
                )
            },
            "by_angle": {
                name: {
                    "case_count": len(subset := [row for row in selected if predicate(row)]),
                    "algorithmic_failure_rate": (
                        sum(row["status"] != "ok" for row in subset) / len(subset) if subset else None
                    ),
                }
                for name, predicate in strata_specs(dataset)
            },
            "by_person_count": {
                str(count): {
                    "case_count": len(subset := [row for row in selected if row["person_count"] == count]),
                    "algorithmic_failure_rate": sum(row["status"] != "ok" for row in subset) / len(subset),
                }
                for count in sorted({row["person_count"] for row in selected if row["person_count"] is not None})
            },
        }
    return output


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(partial, path)


def format_metric(value: dict[str, Any], digits: int) -> str:
    mean_value = value["mean"]
    return "--" if mean_value is None else f"{mean_value:.{digits}f} (n={value['valid_case_count']})"


def main_tables(root: Path, aggregates: dict[str, Any]) -> None:
    lines = [
        "# Traditional registration baselines",
        "",
        "| Method | Access | EgoBody W | EgoBody WA | EgoBody ATE-Sim3 | Reg. success | EgoHumans W | EgoHumans WA | EgoHumans ATE-SE3 | Reg. success |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    csv_rows = []
    for method in (*METHODS, "shot3r"):
        eb, eh = aggregates["egobody"][method], aggregates["egohumans"][method]
        row = {
            "Method": DISPLAY[method], "Access": ACCESS[method],
            "EgoBody_W_mm": eb["metrics"]["w_mpjpe_mm"]["mean"],
            "EgoBody_WA_mm": eb["metrics"]["wa_mpjpe_mm"]["mean"],
            "EgoBody_ATE_Sim3_m": eb["metrics"]["ate_sim3_m"]["mean"],
            "EgoBody_Reg_success": eb["registration_success_rate"],
            "EgoHumans_W_mm": eh["metrics"]["w_mpjpe_mm"]["mean"],
            "EgoHumans_WA_mm": eh["metrics"]["wa_mpjpe_mm"]["mean"],
            "EgoHumans_ATE_SE3_m": eh["metrics"]["ate_se3_m"]["mean"],
            "EgoHumans_Reg_success": eh["registration_success_rate"],
        }
        csv_rows.append(row)
        success_eb = "N/A" if row["EgoBody_Reg_success"] is None else f"{100*row['EgoBody_Reg_success']:.1f}%"
        success_eh = "N/A" if row["EgoHumans_Reg_success"] is None else f"{100*row['EgoHumans_Reg_success']:.1f}%"
        lines.append(
            f"| {DISPLAY[method]} | {ACCESS[method]} | "
            f"{format_metric(eb['metrics']['w_mpjpe_mm'], 1)} | {format_metric(eb['metrics']['wa_mpjpe_mm'], 1)} | "
            f"{format_metric(eb['metrics']['ate_sim3_m'], 3)} | {success_eb} | "
            f"{format_metric(eh['metrics']['w_mpjpe_mm'], 1)} | {format_metric(eh['metrics']['wa_mpjpe_mm'], 1)} | "
            f"{format_metric(eh['metrics']['ate_se3_m'], 3)} | {success_eh} |"
        )
    lines.extend((
        "", "EgoBody uses recording-macro aggregation. EgoHumans retains the frozen paper headline case macro; capture-macro values and paired capture bootstrap are included in `aggregate.json`. Traditional methods receive annotated cut boundaries; Shot3R remains streaming with its online detector.", "",
    ))
    atomic_text(root / "tables/registration_main.md", "\n".join(lines))
    write_csv(root / "tables/registration_main.csv", csv_rows, list(csv_rows[0]))
    tex = [
        "% Auto-generated from frozen case-level evaluations.",
        "\\begin{tabular}{llrrrrrrrr}", "\\toprule",
        "Method & Access & EB W & EB WA & EB ATE & Succ. & EH W & EH WA & EH ATE & Succ. \\\\",
        "\\midrule",
    ]
    for row in csv_rows:
        def val(key: str, digits: int) -> str:
            return "--" if row[key] is None else f"{row[key]:.{digits}f}"
        def succ(key: str) -> str:
            return "--" if row[key] is None else f"{100*row[key]:.1f}\\%"
        tex.append(
            f"{row['Method']} & {row['Access']} & {val('EgoBody_W_mm',1)} & {val('EgoBody_WA_mm',1)} & {val('EgoBody_ATE_Sim3_m',3)} & {succ('EgoBody_Reg_success')} & "
            f"{val('EgoHumans_W_mm',1)} & {val('EgoHumans_WA_mm',1)} & {val('EgoHumans_ATE_SE3_m',3)} & {succ('EgoHumans_Reg_success')} \\\\"
        )
    tex.extend(("\\bottomrule", "\\end{tabular}", ""))
    atomic_text(root / "tables/registration_main.tex", "\n".join(tex))


def angle_markdown(root: Path, angle: dict[str, Any]) -> None:
    lines = [
        "# Viewpoint-stratified traditional registration results", "",
        "Each cell is the frozen publication aggregation within the pre-defined stratum. Failure denotes prediction-only fitting failure followed by the mandatory R0 fallback.", "",
        "| Dataset | Stratum | Cases | Method | W (mm) | WA (mm) | ATE (m) | Failure |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for dataset in ("egobody", "egohumans"):
        for stratum, value in angle[dataset].items():
            for method in (*METHODS, "shot3r"):
                summary = value["methods"].get(method)
                if summary is None:
                    continue
                failure = None if summary["registration_success_rate"] is None else 1.0 - summary["registration_success_rate"]
                lines.append(
                    f"| {dataset} | {stratum} | {value['case_count']} | {DISPLAY[method]} | "
                    f"{format_metric(summary['metrics']['w_mpjpe_mm'],1)} | {format_metric(summary['metrics']['wa_mpjpe_mm'],1)} | "
                    f"{format_metric(summary['metrics']['ate_m'],3)} | {'N/A' if failure is None else f'{100*failure:.1f}%'} |"
                )
    lines.append("")
    atomic_text(root / "tables/registration_angle_strata.md", "\n".join(lines))


def paired_markdown(root: Path, results: dict[str, Any]) -> None:
    lines = [
        "# Paired differences with 95% confidence intervals", "",
        "Positive `vs R0` means the registration method has lower error than R0. Positive `vs Shot3R` means Shot3R has lower error than the registration method. Resampling unit is recording for EgoBody and capture for EgoHumans.", "",
        "| Dataset | Method | Delta W vs R0 | Delta ATE vs R0 | Delta W vs Shot3R | Delta ATE vs Shot3R | Bootstrap unit |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for dataset in ("egobody", "egohumans"):
        for method in ("r1", "r2", "r3"):
            value = results[dataset][method]
            def f(key: str, digits: int) -> str:
                row = value[key]
                return "--" if row["mean"] is None else (
                    f"{row['mean']:.{digits}f} "
                    f"[{row['ci95'][0]:.{digits}f}, {row['ci95'][1]:.{digits}f}] "
                    f"(n={row['common_case_count']}, u={row['unit_count']})"
                )
            lines.append(
                f"| {dataset} | {DISPLAY[method]} | {f('w_mpjpe_mm_vs_r0',1)} | {f('ate_m_vs_r0',3)} | "
                f"{f('w_mpjpe_mm_vs_shot3r',1)} | {f('ate_m_vs_shot3r',3)} | {'recording' if dataset == 'egobody' else 'capture'} |"
            )
    lines.append("")
    atomic_text(root / "tables/registration_paired.md", "\n".join(lines))


def secondary_markdown(root: Path, aggregates: dict[str, Any]) -> None:
    lines = [
        "# Secondary registration metrics", "",
        "IDF1 and Coverage use the full fixed case denominator. Local MPJPE/MPVPE verify that a shared rigid transform does not alter within-camera pose or shape. Boundary and seam quantities diagnose cross-shot geometry; fitting time covers registration only and is not end-to-end inference time.", "",
        "| Dataset | Method | IDF1 | Coverage | Local MPJPE (mm) | Local MPVPE (mm) | Boundary t (m) | Boundary R (deg) | Post root (m) | Seam root (m) | Fit time (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("egobody", "egohumans"):
        for method in (*METHODS, "shot3r"):
            summary = aggregates[dataset][method]
            metrics = summary["metrics"]
            lines.append(
                f"| {dataset} | {DISPLAY[method]} | "
                f"{format_metric(metrics['idf1'],3)} | {format_metric(metrics['coverage'],3)} | "
                f"{format_metric(metrics['mpjpe_mm'],1)} | {format_metric(metrics['mpvpe_mm'],1)} | "
                f"{format_metric(metrics['boundary_camera_translation_m'],3)} | "
                f"{format_metric(metrics['boundary_camera_rotation_deg'],1)} | "
                f"{format_metric(metrics['post_root_m'],3)} | {format_metric(metrics['seam_root_m'],3)} | "
                f"{format_metric(metrics['fit_runtime_seconds'],3)} |"
            )
    lines.extend((
        "",
        "A missing fitting-time entry for Shot3R is intentional: the traditional methods report post-processing time, whereas Shot3R is a streaming reconstruction model and its end-to-end runtime is not directly comparable to registration-only time.",
        "",
    ))
    atomic_text(root / "tables/registration_secondary.md", "\n".join(lines))


def failure_markdown(root: Path, failures: dict[str, Any]) -> None:
    lines = [
        "# Registration failure analysis", "",
        "Algorithmic failure is decided from prediction-only fitting diagnostics and triggers the mandatory exact-R0 fallback. Catastrophic evaluation failure is a post-seal diagnostic only: boundary translation error greater than 1 m or boundary rotation error greater than 30 degrees. It never selects a Test transform.", "",
        "| Dataset | Method | Status counts | Algorithmic failure | Catastrophic evaluation | Fit time mean / median / p90 (s) |",
        "|---|---|---|---:|---:|---:|",
    ]
    for dataset in ("egobody", "egohumans"):
        for method in ("r1", "r2", "r3"):
            value = failures[dataset][method]
            runtime = value["diagnostic_distributions"]["runtime_seconds"]
            status = ", ".join(f"{key}: {count}" for key, count in sorted(value["status_counts"].items()))
            timing = "--" if runtime["mean"] is None else (
                f"{runtime['mean']:.3f} / {runtime['median']:.3f} / {runtime['p90']:.3f}"
            )
            catastrophic_rate = value["catastrophic_evaluation_rate"]
            catastrophic_text = (
                "--" if catastrophic_rate is None else f"{100*catastrophic_rate:.1f}%"
            )
            lines.append(
                f"| {dataset} | {DISPLAY[method]} | {status} | "
                f"{value['algorithmic_failure_count']}/{value['case_count']} "
                f"({100*value['algorithmic_failure_rate']:.1f}%) | "
                f"{value['catastrophic_evaluation_count']}/{value['catastrophic_evaluation_valid_count']} valid "
                f"({catastrophic_text}; "
                f"unavailable={value['catastrophic_evaluation_unavailable_count']}) | {timing} |"
            )
    lines.extend(("", "## Prediction-only scene-registration diagnostics", ""))
    lines.extend((
        "| Dataset | Pre-scene points (median / p90) | Post-scene points (median / p90) | Correspondences (median / p90) | Inlier ratio (median / p90) | Residual (median / p90, m) | Global fitness (median / p90) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ))
    for dataset in ("egobody", "egohumans"):
        diagnostics = failures[dataset]["r3"]["diagnostic_distributions"]
        def pair(name: str, digits: int = 1) -> str:
            value = diagnostics[name]
            if value["median"] is None:
                return "--"
            return f"{value['median']:.{digits}f} / {value['p90']:.{digits}f} (n={value['count']})"
        lines.append(
            f"| {dataset} | {pair('scene_point_count_pre',0)} | {pair('scene_point_count_post',0)} | "
            f"{pair('correspondence_count',0)} | {pair('inlier_ratio',3)} | "
            f"{pair('fit_residual',3)} | {pair('global_fitness',3)} |"
        )
    lines.extend(("", "## Algorithmic failure by pre-defined viewpoint stratum", ""))
    lines.extend((
        "| Dataset | Stratum | Cases | Pelvis translation | Human-joint SE(3) | Scene + ICP |",
        "|---|---|---:|---:|---:|---:|",
    ))
    for dataset in ("egobody", "egohumans"):
        for stratum, item in failures[dataset]["r1"]["by_angle"].items():
            cells = []
            for method in ("r1", "r2", "r3"):
                value = failures[dataset][method]["by_angle"][stratum]["algorithmic_failure_rate"]
                cells.append("--" if value is None else f"{100*value:.1f}%")
            lines.append(f"| {dataset} | {stratum} | {item['case_count']} | {' | '.join(cells)} |")
    lines.extend(("", "## Algorithmic failure by person count", ""))
    lines.extend((
        "| Dataset | People | Cases | Pelvis translation | Human-joint SE(3) | Scene + ICP |",
        "|---|---:|---:|---:|---:|---:|",
    ))
    for dataset in ("egobody", "egohumans"):
        for people, item in failures[dataset]["r1"]["by_person_count"].items():
            cells = [
                f"{100*failures[dataset][method]['by_person_count'][people]['algorithmic_failure_rate']:.1f}%"
                for method in ("r1", "r2", "r3")
            ]
            lines.append(f"| {dataset} | {people} | {item['case_count']} | {' | '.join(cells)} |")
    lines.append("")
    atomic_text(root / "tables/registration_failures.md", "\n".join(lines))


def make_figures(root: Path, angle: dict[str, Any], failures: dict[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"r0": "#777777", "r1": "#4C78A8", "r2": "#F58518", "r3": "#54A24B", "shot3r": "#B279A2"}
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4), constrained_layout=True)
    for axis, dataset in zip(axes, ("egobody", "egohumans")):
        strata = [name for name in ("small", "medium", "large", "extreme") if name in angle[dataset] and angle[dataset][name]["case_count"]]
        x = np.arange(len(strata))
        for method in (*METHODS, "shot3r"):
            values = [angle[dataset][name]["methods"][method]["metrics"]["w_mpjpe_mm"]["mean"] for name in strata]
            axis.plot(x, values, marker="o", linewidth=1.8, label=DISPLAY[method], color=colors[method])
        axis.set_xticks(x, strata)
        axis.set_ylabel("W-MPJPE (mm)")
        axis.set_title("EgoBody" if dataset == "egobody" else "EgoHumans")
        axis.grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=7, loc="best")
    fig.savefig(root / "figures/registration_viewpoint_curve.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), constrained_layout=True)
    for axis, dataset in zip(axes, ("egobody", "egohumans")):
        methods = ("r1", "r2", "r3")
        algorithmic = [100 * failures[dataset][method]["algorithmic_failure_rate"] for method in methods]
        catastrophic = [100 * failures[dataset][method]["catastrophic_evaluation_rate"] for method in methods]
        x = np.arange(len(methods)); width = 0.36
        axis.bar(x - width/2, algorithmic, width, label="Algorithmic", color="#4C78A8")
        axis.bar(x + width/2, catastrophic, width, label="Catastrophic", color="#E45756")
        axis.set_xticks(x, ("Pelvis", "Human SE(3)", "Scene+ICP"))
        axis.set_ylabel("Failure rate (%)")
        axis.set_title("EgoBody" if dataset == "egobody" else "EgoHumans")
        axis.grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(root / "figures/registration_failure_breakdown.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = args.output_root.resolve()
    workspace = REPO_WORKSPACE = Path(__file__).resolve().parents[3]
    all_rows, aggregates, angle, pair, failures = {}, {}, {}, {}, {}
    expected = {"egobody": 129, "egohumans": 90}
    for dataset in ("egobody", "egohumans"):
        current = load_test_rows(root, dataset, expected[dataset])
        metadata = {
            row["case_id"]: {
                key: row[key] for key in ("recording", "capture", "sequence", "angle_stratum", "angle_deg", "person_count")
            }
            for row in current if row["method"] == "r0"
        }
        shot = load_shot3r_rows(workspace, dataset, metadata)
        rows = current + shot
        all_rows[dataset] = rows
        aggregates[dataset] = aggregate_methods(rows, dataset)
        angle[dataset] = angle_tables(rows, dataset)
        pair[dataset] = paired(rows, dataset, int(args.bootstrap), int(args.seed))
        failures[dataset] = failure_analysis(rows, dataset)
    payload = {
        "schema_version": "Shot3R-traditional-registration-final-aggregate-v1",
        "bootstrap_samples": int(args.bootstrap), "bootstrap_seed": int(args.seed),
        "datasets": aggregates, "angle_strata": angle, "paired": pair,
        "failures": failures,
        "aggregation_contract": {
            "egobody_headline": "recording macro over 43 recordings",
            "egohumans_headline": "frozen paper case macro; capture macro also retained per metric",
            "paired_bootstrap": "recording for EgoBody; capture for EgoHumans",
            "failure_fallback": "R0 retained in fixed denominator",
        },
    }
    atomic_json(root / "metrics/aggregate.json", payload)
    atomic_json(root / "metrics/paired_bootstrap.json", pair)
    atomic_json(root / "metrics/angle_strata.json", angle)
    flat = []
    failure_rows = []
    for dataset, rows in all_rows.items():
        for row in rows:
            flat.append({
                "dataset": dataset, "case_id": row["case_id"], "method": row["method"],
                "recording": row["recording"], "capture": row["capture"],
                "angle_stratum": row["angle_stratum"], "angle_deg": row["angle_deg"],
                "person_count": row["person_count"], "status": row["status"], **row["metrics"],
                "world_alignment_available": row.get("world_alignment_available", True),
                "metric_unavailability_reason": row.get("metric_unavailability_reason"),
            })
            if row["method"] in {"r1", "r2", "r3"}:
                failure_rows.append({
                    "dataset": dataset, "case_id": row["case_id"], "method": row["method"],
                    "angle_stratum": row["angle_stratum"], "angle_deg": row["angle_deg"],
                    "person_count": row["person_count"], "status": row["status"],
                    "failure_reason": row["failure_reason"],
                    "catastrophic_evaluation": (
                        None
                        if finite(row["metrics"].get("boundary_camera_translation_m")) is None
                        and finite(row["metrics"].get("boundary_camera_rotation_deg")) is None
                        else (
                            (finite(row["metrics"].get("boundary_camera_translation_m")) is not None
                             and finite(row["metrics"].get("boundary_camera_translation_m")) > 1.0)
                            or (finite(row["metrics"].get("boundary_camera_rotation_deg")) is not None
                                and finite(row["metrics"].get("boundary_camera_rotation_deg")) > 30.0)
                        )
                    ),
                    **{key: row["diagnostics"].get(key) for key in (
                        "scene_point_count_pre", "scene_point_count_post", "correspondence_count",
                        "inlier_ratio", "fit_residual", "global_fitness", "global_residual",
                        "matched_person_count", "runtime_seconds",
                    )},
                })
    write_csv(root / "metrics/case_metrics.csv", flat, list(flat[0]))
    write_csv(root / "metrics/failures.csv", failure_rows, list(failure_rows[0]))
    main_tables(root, aggregates)
    angle_markdown(root, angle)
    paired_markdown(root, pair)
    secondary_markdown(root, aggregates)
    failure_markdown(root, failures)
    make_figures(root, angle, failures)
    print(json.dumps({
        "aggregate": str((root / "metrics/aggregate.json").resolve()),
        "case_rows": len(flat), "bootstrap": int(args.bootstrap),
    }, indent=2))


if __name__ == "__main__":
    main()
