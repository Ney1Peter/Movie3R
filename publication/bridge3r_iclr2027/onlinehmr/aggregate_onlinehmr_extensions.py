#!/usr/bin/env python3
"""Aggregate OnlineHMR extension results on their fixed protocol denominators."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


METHOD = "onlinehmr_official"
AIST_METRICS = {
    "PA-MPJPE_mm": "pa_mpjpe_body12_mm",
    "Anchor-MPJPE_mm": "first_shot_anchor_mpjpe_body12_mm",
    "Anchor-root_mm": "first_shot_anchor_root_error_mm",
    "Orientation_deg": "first_shot_anchor_orientation_proxy_deg",
    "Seam-root_mm": "seam_root_excess_mm",
    "Seam-orientation_deg": "seam_orientation_excess_deg",
    "Camera-rotation_deg": "post_camera_relative_rotation_deg",
    "Camera-translation_m": "post_camera_relative_translation_m",
}
MULTICUT_METRICS = {
    **{key: value for key, value in AIST_METRICS.items() if not key.startswith("Seam-")},
    "Camera-rotation_deg": "post_first_cut_camera_relative_rotation_deg",
    "Camera-translation_m": "post_first_cut_camera_relative_translation_m",
    "Seam-root_mm": "mean_boundary_seam_root_excess_mm",
    "Seam-orientation_deg": "mean_boundary_seam_orientation_excess_deg",
    "Boundary-camera-rotation_deg": "mean_boundary_camera_relative_rotation_deg",
    "Boundary-camera-translation_m": "mean_boundary_camera_relative_translation_m",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def nested(row: dict[str, Any], *keys: str) -> Any:
    value: Any = row
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(partial, path)


def summarize(values: list[float | None]) -> dict[str, Any]:
    array = np.asarray([value for value in values if value is not None], dtype=np.float64)
    return {
        "support": int(len(array)),
        "mean": None if not len(array) else float(array.mean()),
        "median": None if not len(array) else float(np.median(array)),
        "std": None if not len(array) else float(array.std()),
    }


def bootstrap(values: list[float | None], seed: int, samples: int = 10000) -> dict[str, Any]:
    array = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if not len(array):
        return {"support": 0, "mean": None, "ci95": None}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    distribution = array[indices].mean(axis=1)
    return {
        "support": int(len(array)),
        "mean": float(array.mean()),
        "ci95": [float(np.quantile(distribution, 0.025)), float(np.quantile(distribution, 0.975))],
        "bootstrap_samples": samples,
    }


def harmony_values(value: dict[str, Any]) -> dict[str, float | None]:
    named = value.get("multi_thumbs_named_provisional", {})
    seams = [
        finite(nested(row, "cut_seam", "root_excess_m"))
        for row in value.get("cut_seams", {}).values()
    ]
    seam = [item for item in seams if item is not None]
    return {
        "W-MPJPE_mm": finite(nested(named, "w_mpjpe_mm", "mean")),
        "WA-MPJPE_mm": finite(nested(named, "wa_mpjpe_mm", "mean")),
        "ATE-Sim3_m": finite(nested(named, "ate_sim3_m", "mean")),
        "IDF1": finite(nested(value, "identity", "idf1")),
        "Seam-root_m": None if not seam else float(np.mean(seam)),
    }


def aist_values(value: dict[str, Any], protocol: str) -> dict[str, float | None]:
    names = MULTICUT_METRICS if protocol.startswith("MC150") else AIST_METRICS
    return {
        display: finite(nested(value, "metrics", source, "mean"))
        for display, source in names.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--evaluator-manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--bootstrap-seed", type=int, default=20260905)
    args = parser.parse_args()
    runtime_rows = read_jsonl(args.runtime_manifest.resolve())
    evaluator_rows = read_jsonl(args.evaluator_manifest.resolve())
    if [row["case_id"] for row in runtime_rows] != [row["case_id"] for row in evaluator_rows]:
        raise ValueError("runtime/evaluator manifests differ")
    run_root = args.run_root.resolve()
    rows, missing = [], []
    boundary_rows = []
    for line, (record, evaluator) in enumerate(zip(runtime_rows, evaluator_rows), 1):
        root = run_root / f"line{line:03d}"
        raw_path, eval_path = root / "onlinehmr.runtime.json", root / "onlinehmr.evaluation.json"
        if not raw_path.is_file() or not eval_path.is_file():
            missing.append(str(record["case_id"]))
            continue
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        report = json.loads(eval_path.read_text(encoding="utf-8"))
        if raw.get("case_id") != record["case_id"] or report.get("case_id") != record["case_id"]:
            raise ValueError(f"case mismatch at line {line}")
        value = report.get("methods", {}).get(METHOD)
        if not isinstance(value, dict):
            raise ValueError(f"missing OnlineHMR method at line {line}")
        protocol = str(record["protocol"])
        metrics = harmony_values(value) if record["dataset"] == "Harmony4D" else aist_values(value, protocol)
        coverage = finite(nested(value, "coverage", "coverage"))
        if coverage is None:
            coverage = finite(nested(value, "coverage", "valid_frame_coverage"))
        if coverage is None:
            coverage = 0.0
        row = {
            "line": line,
            "case_id": record["case_id"],
            "dataset": record["dataset"],
            "protocol": protocol,
            "raw_status": raw.get("status"),
            "evaluation_status": value.get("status", "ok"),
            "failure_reason": raw.get("failure_reason") or value.get("failure_reason"),
            "wall_time_seconds": finite(raw.get("wall_time_seconds")),
            "native_track_count": int(raw.get("native_track_count", 0)),
            "Coverage": coverage,
            "Completion": 1.0 if raw.get("status") == "success" else 0.0,
            "angle_stratum": evaluator.get("angle_stratum"),
            "subject": evaluator.get("subject"),
            **metrics,
        }
        rows.append(row)
        if protocol.startswith("MC150"):
            transitions = evaluator.get("camera_transition_angles_degrees_evaluator_only", [])
            for detail in value.get("boundary_metrics", []):
                order = int(detail["boundary_order"])
                boundary_rows.append({
                    "case_id": record["case_id"],
                    "boundary_order": order,
                    "transition_angle_deg": finite(transitions[order - 1]) if order <= len(transitions) else None,
                    **{key: finite(detail.get(key)) for key in (
                        "seam_root_excess_mm", "seam_orientation_excess_deg",
                        "camera_relative_rotation_deg", "camera_relative_translation_m",
                    )},
                })
        elif record["dataset"] == "Harmony4D":
            transitions = evaluator.get("camera_transitions_evaluator_only", [])
            for order, (cut, detail) in enumerate(sorted(value.get("cut_seams", {}).items(), key=lambda item: int(item[0])), 1):
                boundary_rows.append({
                    "case_id": record["case_id"],
                    "boundary_order": order,
                    "cut_index": int(cut),
                    "transition_angle_deg": finite(transitions[order - 1].get("rotation_deg")) if order <= len(transitions) else None,
                    "seam_root_m": finite(nested(detail, "cut_seam", "root_excess_m")),
                    "camera_relative_rotation_deg": finite(detail.get("boundary_camera_rpe_rotation_deg")),
                    "camera_relative_translation_m": finite(detail.get("boundary_camera_rpe_translation_m")),
                })
    if args.require_complete and missing:
        raise ValueError(f"missing {len(missing)} fixed-denominator results; first={missing[0]}")
    metric_names = sorted({key for row in rows for key, value in row.items() if key not in {
        "line", "case_id", "dataset", "protocol", "raw_status", "evaluation_status", "angle_stratum", "subject"
    } and isinstance(value, (int, float))})
    overall = {metric: summarize([finite(row.get(metric)) for row in rows]) for metric in metric_names}
    intervals = {metric: bootstrap([finite(row.get(metric)) for row in rows], args.bootstrap_seed + index) for index, metric in enumerate(metric_names)}
    strata = {}
    for label in sorted({str(row["angle_stratum"]) for row in rows if row.get("angle_stratum")}):
        selected = [row for row in rows if row.get("angle_stratum") == label]
        strata[label] = {
            "case_count": len(selected),
            "metrics": {metric: summarize([finite(row.get(metric)) for row in selected]) for metric in metric_names},
        }
    boundary_summary = {}
    for order in sorted({int(row["boundary_order"]) for row in boundary_rows}):
        selected = [row for row in boundary_rows if int(row["boundary_order"]) == order]
        names = sorted({key for row in selected for key, value in row.items() if key not in {"case_id", "boundary_order", "cut_index"} and isinstance(value, (int, float))})
        boundary_summary[str(order)] = {
            "boundary_count": len(selected),
            "metrics": {name: summarize([finite(row.get(name)) for row in selected]) for name in names},
        }
    payload = {
        "schema_version": "Bridge3R-OnlineHMR-extension-aggregate-v1",
        "dataset": runtime_rows[0]["dataset"],
        "protocol": runtime_rows[0]["protocol"],
        "method": METHOD,
        "fixed_manifest_denominator": len(runtime_rows),
        "reported_case_count": len(rows),
        "missing_case_count": len(missing),
        "missing_cases": missing,
        "successful_inference_cases": sum(row["raw_status"] == "success" for row in rows),
        "failed_inference_cases": sum(row["raw_status"] != "success" for row in rows),
        "failure_reason_counts": {
            reason: sum(row.get("failure_reason") == reason for row in rows)
            for reason in sorted({
                str(row["failure_reason"])
                for row in rows
                if row.get("failure_reason")
            })
        },
        "valid_geometry_cases": sum(row["evaluation_status"] in {"ok", None} for row in rows),
        "overall": overall,
        "bootstrap_95_ci": intervals,
        "angle_strata": strata,
        "boundary_order": boundary_summary,
        "aggregation_contract": {
            "case_macro": True,
            "failure_and_empty_output_retained_in_coverage_denominator": True,
            "geometry_means_use_finite_support_and_report_support": True,
            "no_result_based_case_exclusion": True,
        },
        "cases": rows,
        "boundaries": boundary_rows,
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "onlinehmr_extension_aggregate.json", payload)
    fields = sorted({key for row in rows for key in row})
    with (output / "onlinehmr_extension_cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    if boundary_rows:
        fields = sorted({key for row in boundary_rows for key in row})
        with (output / "onlinehmr_extension_boundaries.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(boundary_rows)
    print(json.dumps({
        "output": str(output), "protocol": payload["protocol"],
        "reported": len(rows), "denominator": len(runtime_rows), "missing": len(missing),
        "successful_inference": payload["successful_inference_cases"],
    }, indent=2))


if __name__ == "__main__":
    main()
