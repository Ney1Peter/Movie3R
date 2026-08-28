#!/usr/bin/env python3
"""Aggregate sealed EgoHumans external-baseline evaluations.

The aggregator never filters cases by metric availability: Coverage/IDF1 use
all selected case evaluations, while conditional body/world metrics report an
explicit available-case count.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean


def finite(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def collect(
    paths: list[Path], method_label: str, selected_case_ids: set[str] | None = None
) -> dict:
    rows = []
    for path in sorted(paths):
        try:
            payload = json.loads(path.read_text())
            methods = payload.get("methods", {})
            if not methods:
                continue
            method, value = next(iter(methods.items()))
            runtime = payload.get("record_runtime_fields", {}) or {}
            evaluator = payload.get("record_evaluator_fields", {}) or {}
            named = value.get("multi_thumbs_named_provisional", {})
            coverage = value.get("coverage", {})
            identity = value.get("identity", {})
            rows.append({
                "path": str(path), "case_id": payload.get("case_id"), "method": method,
                "status": value.get("status", payload.get("inference_status", "success")),
                "capture": evaluator.get("capture", runtime.get("capture")),
                "sequence": evaluator.get("sequence", runtime.get("sequence")),
                "angle_stratum": evaluator.get("angle_stratum", runtime.get("angle_stratum")),
                "person_count": evaluator.get("person_count_evaluator_only", runtime.get("person_count")),
                "failure_reason": value.get("failure_reason") or value.get("camera", {}).get("reason"),
                "W-MPJPE_mm": finite(named.get("w_mpjpe_mm", {}).get("mean")),
                "WA-MPJPE_mm": finite(named.get("wa_mpjpe_mm", {}).get("mean")),
                "MPJPE_mm": finite(named.get("mpjpe_mm", {}).get("mean")),
                "PA-MPJPE_mm": finite(named.get("pa_mpjpe_mm", {}).get("mean")),
                "MPVPE_mm": finite(named.get("mpvpe_mm", {}).get("mean")),
                "Accel_mm_frame2": finite(named.get("accel_delta2_mm_per_frame2", {}).get("mean")),
                "ATE-Sim3_m": finite(named.get("ate_sim3_m", {}).get("mean")),
                "ATE-SE3_m": finite(named.get("ate_se3_m", {}).get("mean")),
                "RPE-translation_m": finite(value.get("camera", {}).get("rpe_translation_m", {}).get("mean")),
                "RPE-rotation_deg": finite(value.get("camera", {}).get("rpe_rotation_deg", {}).get("mean")),
                "Camera-seam-translation_m": finite(value.get("cut_seam", {}).get("camera_translation_excess_m")),
                "Camera-seam-rotation_deg": finite(value.get("cut_seam", {}).get("camera_rotation_excess_deg")),
                "Human-seam_mm": finite(value.get("cut_seam", {}).get("human_joint_excess_mm", {}).get("mean")),
                "IDF1": finite(identity.get("idf1")),
                "IDs": finite(identity.get("ids_total")),
                "Coverage": finite(coverage.get("coverage")),
                "Precision": finite(coverage.get("precision")),
                "W_available": bool(value.get("world_alignment", {}).get("w_available")),
                "WA_available": bool(value.get("world_alignment", {}).get("wa_available")),
                "camera_reportable": bool(value.get("camera", {}).get("reportable")),
            })
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
    source_case_count = len(rows)
    if selected_case_ids is not None:
        observed = [str(row.get("case_id")) for row in rows]
        if len(observed) != len(set(observed)):
            raise RuntimeError(f"{method_label}: duplicate case_id rows detected in source")
        missing = selected_case_ids - set(observed)
        if missing:
            raise RuntimeError(
                f"{method_label}: selected formal cases are missing from source ({len(missing)}): "
                f"{sorted(missing)[:3]}"
            )
        rows = [row for row in rows if str(row.get("case_id")) in selected_case_ids]
    metrics = ["W-MPJPE_mm", "WA-MPJPE_mm", "MPJPE_mm", "PA-MPJPE_mm", "MPVPE_mm", "Accel_mm_frame2", "ATE-Sim3_m", "ATE-SE3_m", "RPE-translation_m", "RPE-rotation_deg", "Camera-seam-translation_m", "Camera-seam-rotation_deg", "Human-seam_mm", "IDF1", "IDs", "Coverage", "Precision"]
    summary = {"method": method_label, "case_count": len(rows), "source_case_count": source_case_count, "success_cases": sum(row["status"] not in {"inference_failed", "failed"} for row in rows), "failed_cases": sum(row["status"] in {"inference_failed", "failed"} for row in rows)}
    for metric in metrics:
        values = [row[metric] for row in rows if row[metric] is not None]
        summary[metric] = mean(values) if values else None
        summary[f"{metric}_available_cases"] = len(values)
    summary["W_available_cases"] = sum(row["W_available"] for row in rows)
    summary["WA_available_cases"] = sum(row["WA_available"] for row in rows)
    summary["camera_reportable_cases"] = sum(row["camera_reportable"] for row in rows)
    summary["case_rows"] = rows
    return summary


def grouped_rows(results: dict, key: str) -> list[dict]:
    """Return case-macro summaries for an interpretable stratum.

    Every group is computed from the complete case rows for that method.  The
    conditional body/world availability is exposed explicitly instead of
    silently dropping failed or unmatched cases.
    """
    metrics = ["W-MPJPE_mm", "WA-MPJPE_mm", "MPJPE_mm", "PA-MPJPE_mm",
               "MPVPE_mm", "Accel_mm_frame2", "ATE-Sim3_m", "ATE-SE3_m",
               "RPE-translation_m", "RPE-rotation_deg", "Camera-seam-translation_m",
               "Camera-seam-rotation_deg", "Human-seam_mm", "IDF1", "IDs", "Coverage", "Precision"]
    out = []
    for result in results.values():
        buckets = {}
        for row in result.get("case_rows", []):
            group = row.get(key) or "unknown"
            buckets.setdefault(group, []).append(row)
        for group, rows in sorted(buckets.items()):
            item = {"method": result["method"], key: group,
                    "case_count": len(rows),
                    "success_cases": sum(row.get("status") not in {"inference_failed", "failed"} for row in rows),
                    "failed_cases": sum(row.get("status") in {"inference_failed", "failed"} for row in rows),
                    "W_available_cases": sum(bool(row.get("W_available")) for row in rows),
                    "WA_available_cases": sum(bool(row.get("WA_available")) for row in rows),
                    "camera_reportable_cases": sum(bool(row.get("camera_reportable")) for row in rows)}
            for metric in metrics:
                values = [row[metric] for row in rows if row.get(metric) is not None]
                item[metric] = mean(values) if values else None
                item[f"{metric}_available_cases"] = len(values)
            out.append(item)
    return out


def write_rows(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data/EgoHuman_work_v19/external_predictions"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-cases", type=int, default=None,
                   help="fail closed if a source does not have this many evaluation rows")
    p.add_argument("--manifest", type=Path,
                   default=Path("data/EgoHuman_work_v19/external_predictions/trace_egohumans_v2/manifests/egohumans_test.runtime.jsonl"),
                   help="frozen runtime manifest used to verify case identity and sequence parity")
    p.add_argument(
        "--selected-case-manifest",
        type=Path,
        help=(
            "Optional immutable formal JSONL manifest. Sources may contain a strict "
            "superset, but every listed case must exist and only listed cases enter "
            "the reported denominator."
        ),
    )
    args = p.parse_args()
    root = args.root.resolve()
    expected_case_ids = None
    if args.selected_case_manifest is not None:
        selected = args.selected_case_manifest.resolve()
        if not selected.is_file():
            raise FileNotFoundError(selected)
        expected_case_ids = {
            str(json.loads(line)["case_id"])
            for line in selected.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        if not expected_case_ids:
            raise RuntimeError("selected case manifest is empty")
    elif args.manifest.is_file():
        expected_case_ids = []
        for line in args.manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                expected_case_ids.append(json.loads(line)["case_id"])
        expected_case_ids = set(expected_case_ids)
    sources = {
        "TRACE (official, subject=4)": root / "trace_egohumans_v2/test/evaluations",
        "PromptHMR (official SPEC)": root / "prompthmr_egohumans/spec/egohumans_test_spec",
        "PromptHMR (no-SPEC adapter)": root / "prompthmr_egohumans/nospec/egohumans_test_nospec",
    }
    results = {}
    for label, path in sources.items():
        if path.is_dir():
            # TRACE writes ``evaluations/line001.evaluation.json`` directly;
            # PromptHMR keeps the variant filename inside ``line001/``.
            pattern = "line*.evaluation.json" if "trace_egohumans" in str(path) else "line*/prompthmr_*.evaluation.json"
            results[label] = collect(list(path.glob(pattern)), label, expected_case_ids)
        else:
            results[label] = {"method": label, "case_count": 0, "success_cases": 0, "failed_cases": 0, "case_rows": []}
        if args.expected_cases is not None and results[label].get("case_count", 0) != args.expected_cases:
            raise RuntimeError(
                f"{label}: expected {args.expected_cases} evaluation rows, "
                f"found {results[label].get('case_count', 0)}"
            )
        if expected_case_ids is not None:
            actual_case_ids = [row.get("case_id") for row in results[label].get("case_rows", [])]
            actual_set = set(actual_case_ids)
            if len(actual_case_ids) != len(actual_set):
                raise RuntimeError(f"{label}: duplicate case_id rows detected")
            missing = expected_case_ids - actual_set
            extra = actual_set - expected_case_ids
            if missing or extra:
                raise RuntimeError(
                    f"{label}: case identity mismatch against frozen manifest "
                    f"(missing={len(missing)}, extra={len(extra)})"
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fields = ["method", "case_count", "success_cases", "failed_cases", "W-MPJPE_mm", "WA-MPJPE_mm", "MPJPE_mm", "PA-MPJPE_mm", "MPVPE_mm", "Accel_mm_frame2", "ATE-Sim3_m", "ATE-SE3_m", "RPE-translation_m", "RPE-rotation_deg", "Camera-seam-translation_m", "Camera-seam-rotation_deg", "Human-seam_mm", "IDF1", "IDs", "Coverage", "Precision", "W_available_cases", "WA_available_cases", "camera_reportable_cases"]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results.values():
            writer.writerow({key: result.get(key) for key in fields})
    case_fields = ["method", "case_id", "capture", "sequence", "angle_stratum", "person_count", "status", "failure_reason",
                   "W-MPJPE_mm", "WA-MPJPE_mm", "MPJPE_mm", "PA-MPJPE_mm", "MPVPE_mm",
                   "Accel_mm_frame2", "ATE-Sim3_m", "ATE-SE3_m", "RPE-translation_m", "RPE-rotation_deg", "Camera-seam-translation_m", "Camera-seam-rotation_deg", "Human-seam_mm", "IDF1", "IDs", "Coverage",
                   "Precision", "W_available", "WA_available", "camera_reportable", "path"]
    case_rows = [row for result in results.values() for row in result.get("case_rows", [])]
    write_rows(args.output.with_name("external_baseline_case_metrics.csv"), case_rows, case_fields)
    angle_fields = ["method", "angle_stratum", "case_count", "success_cases", "failed_cases",
                     "W-MPJPE_mm", "WA-MPJPE_mm", "MPJPE_mm", "PA-MPJPE_mm", "MPVPE_mm",
                     "Accel_mm_frame2", "ATE-Sim3_m", "ATE-SE3_m", "RPE-translation_m", "RPE-rotation_deg", "Camera-seam-translation_m", "Camera-seam-rotation_deg", "Human-seam_mm", "IDF1", "IDs", "Coverage", "Precision",
                     "W_available_cases", "WA_available_cases", "camera_reportable_cases"]
    # Angle rows and action/sequence rows use one stable schema; the grouping
    # key is made explicit in the filename and column name.
    write_rows(args.output.with_name("external_baseline_angle_metrics.csv"),
               grouped_rows(results, "angle_stratum"), angle_fields)
    action_rows = grouped_rows(results, "sequence")
    write_rows(args.output.with_name("external_baseline_action_metrics.csv"), action_rows,
               ["method", "sequence", "case_count", "success_cases", "failed_cases",
                "W-MPJPE_mm", "WA-MPJPE_mm", "MPJPE_mm", "PA-MPJPE_mm", "MPVPE_mm",
                "Accel_mm_frame2", "ATE-Sim3_m", "ATE-SE3_m", "RPE-translation_m", "RPE-rotation_deg", "Camera-seam-translation_m", "Camera-seam-rotation_deg", "Human-seam_mm", "IDF1", "IDs", "Coverage", "Precision",
                "W_available_cases", "WA_available_cases", "camera_reportable_cases"])
    failure_rows = []
    for result in results.values():
        for row in result.get("case_rows", []):
            if row.get("status") in {"inference_failed", "failed"} or row.get("Coverage") in {None, 0.0}:
                failure_rows.append({"method": result["method"], "case_id": row.get("case_id"),
                                     "capture": row.get("capture"), "sequence": row.get("sequence"),
                                     "angle_stratum": row.get("angle_stratum"), "status": row.get("status"),
                                     "failure_reason": row.get("failure_reason"), "coverage": row.get("Coverage"), "path": row.get("path")})
    write_rows(args.output.with_name("external_baseline_failures.csv"), failure_rows,
               ["method", "case_id", "capture", "sequence", "angle_stratum", "status", "failure_reason", "coverage", "path"])
    denominator = len(expected_case_ids) if expected_case_ids is not None else "the frozen source"
    md = ["# EgoHumans external baseline results", "", f"All rows use the same frozen {denominator}-case CS100 selection. Coverage and IDF1 retain every selected case; conditional metrics expose their availability counts. Camera ATE/RPE and seam metrics are N/A when the method has no reportable trajectory.", "", "| Method | Cases | Success | Failed | W-MPJPE | WA-MPJPE | MPJPE | PA-MPJPE | MPVPE | Accel | ATE-Sim3 | ATE-SE3 | RPE-T | RPE-R | C-seam-T | C-seam-R | H-seam | IDF1 | IDs | Coverage | W avail. | WA avail. |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for result in results.values():
        fmt = lambda key: "—" if result.get(key) is None else f"{float(result[key]):.3f}"
        md.append(f"| {result['method']} | {result.get('case_count',0)} | {result.get('success_cases',0)} | {result.get('failed_cases',0)} | {fmt('W-MPJPE_mm')} | {fmt('WA-MPJPE_mm')} | {fmt('MPJPE_mm')} | {fmt('PA-MPJPE_mm')} | {fmt('MPVPE_mm')} | {fmt('Accel_mm_frame2')} | {fmt('ATE-Sim3_m')} | {fmt('ATE-SE3_m')} | {fmt('RPE-translation_m')} | {fmt('RPE-rotation_deg')} | {fmt('Camera-seam-translation_m')} | {fmt('Camera-seam-rotation_deg')} | {fmt('Human-seam_mm')} | {fmt('IDF1')} | {fmt('IDs')} | {fmt('Coverage')} | {result.get('W_available_cases',0)} | {result.get('WA_available_cases',0)} |")
    args.output.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({label: {key: value for key, value in result.items() if key != "case_rows"} for label, result in results.items()}, indent=2))


if __name__ == "__main__":
    main()
