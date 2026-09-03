#!/usr/bin/env python3
"""Aggregate fixed-denominator OnlineHMR evaluations for publication."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean
from typing import Any


METHOD = "onlinehmr_official"
SCHEMA = "Bridge3R-OnlineHMR-publication-aggregate-v1"
METRICS = (
    "W-MPJPE_mm", "WA-MPJPE_mm", "MPJPE_mm", "PA-MPJPE_mm", "MPVPE_mm",
    "ATE-Sim3_m", "ATE-SE3_m", "RPE-translation_m", "RPE-rotation_deg",
    "Camera-seam-translation_m", "Camera-seam-rotation_deg", "Human-seam_mm",
    "IDF1", "IDs", "Coverage", "Precision",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def nested_mean(value: dict[str, Any], *keys: str) -> float | None:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return finite(current)


def angle(case_id: str) -> str:
    for label in ("small", "medium", "large", "extreme"):
        if f"_{label}_" in case_id:
            return label
    raise ValueError(f"case has no angle stratum: {case_id}")


def summarize(rows: list[dict[str, Any]], scope: str) -> dict[str, Any]:
    output: dict[str, Any] = {
        "scope": scope,
        "case_count": len(rows),
        "successful_inference_cases": sum(row["raw_status"] == "success" for row in rows),
        "failed_inference_cases": sum(row["raw_status"] != "success" for row in rows),
        "W_available_cases": sum(bool(row["W_available"]) for row in rows),
        "WA_available_cases": sum(bool(row["WA_available"]) for row in rows),
        "camera_reportable_cases": sum(bool(row["camera_reportable"]) for row in rows),
    }
    for metric in METRICS:
        values = [row[metric] for row in rows if row[metric] is not None]
        output[metric] = mean(values) if values else None
        output[f"{metric}_available_cases"] = len(values)
    return output


def unit_macro(rows: list[dict[str, Any]], scope: str) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(str(row["unit"]), []).append(row)
    unit_rows: list[dict[str, Any]] = []
    for unit, values in sorted(buckets.items()):
        item: dict[str, Any] = {"unit": unit}
        for metric in METRICS:
            finite_values = [row[metric] for row in values if row[metric] is not None]
            item[metric] = mean(finite_values) if finite_values else None
        unit_rows.append(item)
    output: dict[str, Any] = {
        "scope": scope,
        "unit_count": len(unit_rows),
        "case_count": len(rows),
    }
    for metric in METRICS:
        values = [row[metric] for row in unit_rows if row[metric] is not None]
        output[metric] = mean(values) if values else None
        output[f"{metric}_available_units"] = len(values)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("egobody", "egohumans", "harmony4d"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lines", help="optional subset for pilot aggregation")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    manifest = args.manifest.resolve()
    manifest_rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.lines:
        selected = [int(value) for value in args.lines.split(",") if value.strip()]
    else:
        selected = list(range(1, len(manifest_rows) + 1))
    if len(selected) != len(set(selected)) or any(line < 1 or line > len(manifest_rows) for line in selected):
        raise ValueError("invalid selected lines")

    rows: list[dict[str, Any]] = []
    missing = []
    for line in selected:
        record = manifest_rows[line - 1]
        case_id = str(record["case_id"])
        evaluation = args.run_root.resolve() / f"line{line:03d}/onlinehmr.evaluation.json"
        raw_path = args.run_root.resolve() / f"line{line:03d}/onlinehmr.runtime.json"
        if not evaluation.is_file() or not raw_path.is_file():
            missing.append(case_id)
            continue
        payload = json.loads(evaluation.read_text(encoding="utf-8"))
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        if payload.get("case_id") != case_id or raw.get("case_id") != case_id:
            raise ValueError(f"case mismatch at line {line}")
        if set(payload.get("methods", {})) != {METHOD}:
            raise ValueError(f"unexpected method at line {line}")
        value = payload["methods"][METHOD]
        named = value.get("multi_thumbs_named_provisional", {})
        coverage = value.get("coverage", {})
        identity = value.get("identity", {})
        camera = value.get("camera", {})
        seam = value.get("cut_seam", {})
        unit = record.get("recording") if args.dataset == "egobody" else record.get("capture")
        if not unit:
            raise ValueError(f"missing independent unit at line {line}")
        rows.append({
            "line": line,
            "case_id": case_id,
            "unit": str(unit),
            "sequence": record.get("sequence"),
            "angle_stratum": angle(case_id),
            "raw_status": raw.get("status"),
            "failure_reason": raw.get("failure_reason"),
            "wall_time_seconds": finite(raw.get("wall_time_seconds")),
            "native_track_count": int(raw.get("native_track_count", 0)),
            "W-MPJPE_mm": nested_mean(named, "w_mpjpe_mm", "mean"),
            "WA-MPJPE_mm": nested_mean(named, "wa_mpjpe_mm", "mean"),
            "MPJPE_mm": nested_mean(named, "mpjpe_mm", "mean"),
            "PA-MPJPE_mm": nested_mean(named, "pa_mpjpe_mm", "mean"),
            "MPVPE_mm": nested_mean(named, "mpvpe_mm", "mean"),
            "ATE-Sim3_m": nested_mean(camera, "ate_sim3_m", "mean"),
            "ATE-SE3_m": nested_mean(camera, "ate_se3_m", "mean"),
            "RPE-translation_m": nested_mean(camera, "rpe_translation_m", "mean"),
            "RPE-rotation_deg": nested_mean(camera, "rpe_rotation_deg", "mean"),
            "Camera-seam-translation_m": finite(seam.get("camera_translation_excess_m")),
            "Camera-seam-rotation_deg": finite(seam.get("camera_rotation_excess_deg")),
            "Human-seam_mm": nested_mean(seam, "human_joint_excess_mm", "mean"),
            "IDF1": finite(identity.get("idf1")),
            "IDs": finite(identity.get("ids_total")),
            "Coverage": finite(coverage.get("coverage")),
            "Precision": finite(coverage.get("precision")),
            "W_available": bool(value.get("world_alignment", {}).get("w_available")),
            "WA_available": bool(value.get("world_alignment", {}).get("wa_available")),
            "camera_reportable": bool(camera.get("reportable")),
            "evaluation": str(evaluation),
            "evaluation_sha256": sha256(evaluation),
        })
    if args.require_complete and missing:
        raise FileNotFoundError(f"missing {len(missing)} evaluations; examples={missing[:3]}")

    case_summary = summarize(rows, "case-macro")
    independent_summary = unit_macro(rows, "independent-unit-macro")
    angle_rows = []
    for label in ("small", "medium", "large", "extreme"):
        subset = [row for row in rows if row["angle_stratum"] == label]
        if subset:
            angle_rows.append({"angle_stratum": label, **summarize(subset, "case-macro")})
            angle_rows.append({"angle_stratum": label, **unit_macro(subset, "independent-unit-macro")})
    payload = {
        "schema_version": SCHEMA,
        "dataset": args.dataset,
        "method": METHOD,
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "selected_lines": selected,
        "expected_cases": len(selected),
        "observed_cases": len(rows),
        "missing_case_ids": missing,
        "fixed_denominator_complete": not missing,
        "case_macro": case_summary,
        "independent_unit_macro": independent_summary,
        "angle_summaries": angle_rows,
        "runtime_gt_access": False,
        "conditional_errors_accompanied_by_availability": True,
        "cases": rows,
    }
    output = args.output_dir.resolve()
    atomic_json(output / "onlinehmr_aggregate.json", payload)
    fields = [
        "line", "case_id", "unit", "sequence", "angle_stratum", "raw_status",
        "failure_reason", "wall_time_seconds", "native_track_count", *METRICS,
        "W_available", "WA_available", "camera_reportable", "evaluation",
        "evaluation_sha256",
    ]
    write_csv(output / "onlinehmr_case_metrics.csv", rows, fields)
    summary_fields = [
        "scope", "case_count", "unit_count", "successful_inference_cases",
        "failed_inference_cases", *METRICS, "W_available_cases", "WA_available_cases",
        "camera_reportable_cases",
    ]
    write_csv(
        output / "onlinehmr_summary.csv",
        [case_summary, independent_summary], summary_fields,
    )
    write_csv(
        output / "onlinehmr_angle_metrics.csv", angle_rows,
        ["angle_stratum", *summary_fields],
    )
    write_csv(
        output / "onlinehmr_failures.csv",
        [row for row in rows if row["raw_status"] != "success" or row["Coverage"] in (None, 0.0)],
        fields,
    )
    print(json.dumps({
        "dataset": args.dataset,
        "observed": len(rows),
        "missing": len(missing),
        "case_macro": case_summary,
        "independent_unit_macro": independent_summary,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
