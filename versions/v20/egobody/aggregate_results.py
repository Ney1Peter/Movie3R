#!/usr/bin/env python3
"""Recording-macro aggregation and safety audit for one EgoBody split."""

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


SCHEMA = "Bridge3R-EgoBody-CS150-aggregate-v1"
PROBE_SCHEMA = "Movie3R-v16-Harmony4D-causal-stabilization-probe-v1"
STATE_SCHEMA = "Bridge3R-EgoBody-CS150-protocol-state-v1"
BASELINE = "v16_0_m15_geometry"
REFERENCE_METHODS = (
    "m0_strict_human3r",
    "m0r_original_clean_reset",
    "m1_current_clean_reset",
    "m3_b0_only",
    "m15_v17_gated_parent",
)
CANDIDATE_SCHEMAS = {
    "development": "Bridge3R-EgoBody-development-candidates-v1",
    "holdout": "Bridge3R-EgoBody-frozen-holdout-candidates-v1",
    "test": "Bridge3R-EgoBody-frozen-final-candidate-v1",
}
LOWER_IS_BETTER = {
    "W-MPJPE_mm", "WA-MPJPE_mm", "MPJPE_mm", "PA-MPJPE_mm", "MPVPE_mm",
    "Accel_mm_frame2", "RTE_H3R_percent", "ROE_joint_proxy_deg", "Jitter_H3R",
    "Foot_sliding_cm", "ATE_Sim3_m", "ATE_SE3_m", "Boundary_camera_t_m",
    "Boundary_camera_R_deg", "Boundary_root_m", "Post_root_m", "Seam_camera_t_m",
    "Seam_camera_R_deg", "Seam_root_m", "Seam_CHRGE_m", "CHRGE_m",
    "Pair_vector_m", "IDs",
}
HIGHER_IS_BETTER = {"IDF1", "Coverage", "Detection_precision"}
CORE = ("W-MPJPE_mm", "WA-MPJPE_mm", "Accel_mm_frame2", "ATE_SE3_m", "Seam_root_m")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def mean(values: list[float]) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if len(array) else None


def geometric_mean(values: list[float]) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return None
    if (array < 0).any():
        raise ValueError("geometric mean received a negative ratio")
    if (array == 0).any():
        return 0.0
    return float(np.exp(np.log(array).mean()))


def bootstrap(values: list[float], seed: int, samples: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0, "mean": None, "ci95": [None, None]}
    if len(array) == 1:
        return {"count": 1, "mean": float(array[0]), "ci95": [float(array[0]), float(array[0])]}
    rng = np.random.default_rng(seed)
    draws = array[rng.integers(0, len(array), size=(samples, len(array)))].mean(axis=1)
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--protocol-state", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("development", "holdout", "test"), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parent", default="v16_0_m15_geometry")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260821)
    return parser.parse_args()


def validate_protocol_source(
    report_path: Path, state_path: Path, split: str
) -> dict[str, Any]:
    report_path = report_path.resolve()
    state_path = state_path.resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != PROBE_SCHEMA:
        raise ValueError("unexpected candidate-report schema")
    if state.get("schema_version") != STATE_SCHEMA:
        raise ValueError("unexpected protocol-state schema")
    if state.get("status") != "complete" or state.get("split") != split:
        raise ValueError("candidate report is not from a completed matching split")
    if state.get("smoke_subset") is not False or state.get("max_cases") is not None:
        raise ValueError("formal aggregation may not consume a smoke subset")
    report_sha = sha256(report_path)
    state_reports = state.get("candidate_reports", {})
    if not isinstance(state_reports, dict) or not any(
        isinstance(value, dict)
        and value.get("status") == "complete"
        and value.get("output_sha256") == report_sha
        for value in state_reports.values()
    ):
        raise ValueError("candidate report SHA is absent from the completed protocol state")
    errors = report.get("errors")
    if not isinstance(errors, list) or errors:
        raise ValueError(f"candidate report has errors: {errors}")
    contract = report.get("contract", {})
    if (
        not isinstance(contract, dict)
        or contract.get("candidate_runtime_uses_gt") is not False
        or int(contract.get("future_frames_at_boundary", -1)) != 0
        or contract.get("source_cache_immutable") is not True
    ):
        raise ValueError("candidate report violates the prediction-only contract")
    candidate_path = Path(str(report.get("candidate_source", ""))).resolve()
    if not candidate_path.is_file():
        raise FileNotFoundError(candidate_path)
    candidate_sha = sha256(candidate_path)
    candidate_state = [
        value
        for value in state.get("candidate_json", [])
        if isinstance(value, dict)
        and Path(str(value.get("path", ""))).resolve() == candidate_path
        and value.get("sha256") == candidate_sha
    ]
    if len(candidate_state) != 1:
        raise ValueError("candidate source path/SHA is not frozen in protocol state")
    candidate_spec = json.loads(candidate_path.read_text(encoding="utf-8"))
    if candidate_spec.get("schema_version") != CANDIDATE_SCHEMAS[split]:
        raise ValueError("candidate source schema does not match split")
    source_names = [str(row.get("name", "")) for row in candidate_spec.get("candidates", [])]
    if not source_names or not all(source_names) or len(source_names) != len(set(source_names)):
        raise ValueError("candidate source has invalid/duplicate names")
    expected_candidates = set(source_names) | {BASELINE}
    expected_cases = {
        str(value) for value in state.get("run_identity", {}).get("selected_case_ids", [])
    }
    if not expected_cases or len(expected_cases) != int(state.get("selected_case_count", -1)):
        raise ValueError("protocol state has inconsistent selected cases")
    rows = report.get("rows")
    references = report.get("reference_rows")
    skipped_rows = report.get("skipped_cases")
    if not isinstance(rows, list) or not isinstance(references, list) or not isinstance(skipped_rows, list):
        raise ValueError("candidate report rows are malformed")
    skipped: set[str] = set()
    for row in skipped_rows:
        case_id = str(row.get("case_id", ""))
        if (
            not case_id
            or case_id in skipped
            or row.get("status") != "evaluator_unavailable"
            or row.get("method_independent") is not True
        ):
            raise ValueError("invalid/duplicate evaluator-unavailable record")
        skipped.add(case_id)
    if skipped - expected_cases:
        raise ValueError("candidate report skips an unexpected case")

    def index(values: list[dict[str, Any]], name_field: str) -> dict[str, dict[str, dict[str, Any]]]:
        output: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in values:
            if not isinstance(row, dict):
                raise ValueError("candidate report contains a non-object row")
            case_id, name = str(row.get("case_id", "")), str(row.get(name_field, ""))
            if case_id not in expected_cases or not name or name in output[case_id]:
                raise ValueError(f"unexpected/duplicate report row {(case_id, name)}")
            output[case_id][name] = row
        return output

    candidate_index = index(rows, "candidate")
    reference_index = index(references, "method")
    for case_id in expected_cases:
        if set(candidate_index.get(case_id, {})) != expected_candidates:
            raise ValueError(f"candidate coverage mismatch in {case_id}")
        if set(reference_index.get(case_id, {})) != set(REFERENCE_METHODS):
            raise ValueError(f"reference coverage mismatch in {case_id}")
        expected_status = "error" if case_id in skipped else "complete"
        case_rows = [
            *candidate_index[case_id].values(), *reference_index[case_id].values()
        ]
        if any(row.get("status") != expected_status for row in case_rows):
            raise ValueError(f"method-dependent status in {case_id}")
    if int(report.get("candidate_count", -1)) != len(expected_candidates):
        raise ValueError("candidate_count mismatch")
    if int(report.get("case_count", -1)) != len(expected_cases):
        raise ValueError("case_count mismatch")
    if int(report.get("complete_case_count", -1)) != len(expected_cases - skipped):
        raise ValueError("complete_case_count mismatch")
    return {
        "report": report,
        "report_sha256": report_sha,
        "state": state,
        "state_sha256": sha256(state_path),
        "candidate_source": str(candidate_path),
        "candidate_source_sha256": candidate_sha,
        "selected_case_count": len(expected_cases),
        "evaluator_unavailable_case_count": len(skipped),
    }


def row_name(row: dict[str, Any]) -> str:
    value = row.get("candidate", row.get("method"))
    if not value:
        raise ValueError(f"row lacks method/candidate: {row}")
    return str(value)


def recording_name(row: dict[str, Any]) -> str:
    value = row.get("capture") or row.get("recording")
    if not value:
        case = str(row["case_id"])
        marker = "egobody_recording_"
        if not case.startswith(marker):
            raise ValueError(f"cannot infer recording from {case}")
        # recording_YYYYMMDD_Sxx_Syy_take ends before the angle name.
        fields = case[len("egobody_"):].split("_")
        value = "_".join(fields[:5])
    return str(value)


def normalized_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    seen = set()
    for row in list(payload.get("rows", [])) + list(payload.get("reference_rows", [])):
        if row.get("status") != "complete":
            continue
        metrics = {str(key): finite(value) for key, value in dict(row.get("metrics", {})).items()}
        key = (str(row["case_id"]), row_name(row))
        if key in seen:
            # The B0 source may also appear as a reference.  Identical duplicates
            # are harmless, but non-identical duplicates are a structural error.
            previous = next(value for value in output if (value["case_id"], value["name"]) == key)
            if previous["metrics"] != metrics:
                raise ValueError(f"non-identical duplicate {key}")
            continue
        seen.add(key)
        output.append({
            "case_id": key[0],
            "name": key[1],
            "recording": recording_name(row),
            "angle_stratum": row.get("angle_stratum"),
            "metrics": metrics,
            "diagnostics": row.get("diagnostics"),
        })
    if not output:
        raise ValueError("candidate report has no complete rows")
    metric_sets = {tuple(sorted(row["metrics"])) for row in output}
    if len(metric_sets) != 1:
        raise ValueError("complete rows do not expose one identical metric set")
    return output


def case_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(row["case_id"], row["name"]): row for row in rows}


def metric_names(rows: list[dict[str, Any]]) -> list[str]:
    values = sorted({key for row in rows for key in row["metrics"]})
    unknown = set(values) - LOWER_IS_BETTER - HIGHER_IS_BETTER
    if unknown:
        raise ValueError(f"metric direction is not registered: {sorted(unknown)}")
    return values


def macro_tables(
    rows: list[dict[str, Any]], metrics: list[str], seed: int, samples: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["name"], row["recording"])].append(row)
    recording_rows = []
    for (name, recording), values in sorted(grouped.items()):
        recording_rows.append({
            "name": name,
            "recording": recording,
            "case_count": len(values),
            **{metric: mean([row["metrics"][metric] for row in values if row["metrics"].get(metric) is not None]) for metric in metrics},
        })
    aggregate: dict[str, Any] = {}
    for name in sorted({row["name"] for row in recording_rows}):
        values = [row for row in recording_rows if row["name"] == name]
        aggregate[name] = {
            "recording_count": len(values),
            "case_count": sum(int(row["case_count"]) for row in values),
            "metrics": {
                metric: bootstrap(
                    [row[metric] for row in values if row[metric] is not None],
                    seed=int(seed) ^ int(hashlib.sha256(f"{name}:{metric}".encode()).hexdigest()[:8], 16),
                    samples=samples,
                )
                for metric in metrics
            },
        }
    return recording_rows, aggregate


def paired_analysis(rows: list[dict[str, Any]], parent: str, metrics: list[str]) -> dict[str, Any]:
    lookup = case_lookup(rows)
    names = sorted({row["name"] for row in rows} - {parent})
    output: dict[str, Any] = {}
    parent_cases = {row["case_id"] for row in rows if row["name"] == parent}
    if not parent_cases:
        raise ValueError(f"parent {parent!r} is absent")
    for name in names:
        cases = sorted(parent_cases & {row["case_id"] for row in rows if row["name"] == name})
        metric_rows = {}
        for metric in metrics:
            ratios, deltas, improved = [], [], 0
            for case in cases:
                base = lookup[(case, parent)]["metrics"].get(metric)
                value = lookup[(case, name)]["metrics"].get(metric)
                if base is None or value is None:
                    continue
                if metric in LOWER_IS_BETTER:
                    ratio = value / max(base, 1e-12)
                    delta = base - value
                    improved += int(value < base)
                else:
                    ratio = max(base, 1e-12) / max(value, 1e-12)
                    delta = value - base
                    improved += int(value > base)
                ratios.append(float(ratio))
                deltas.append(float(delta))
            metric_rows[metric] = {
                "paired_count": len(ratios),
                "mean_candidate_to_parent_error_ratio": mean(ratios),
                "mean_improvement_in_metric_direction": mean(deltas),
                "improved_cases": improved,
                "nonworse_cases": sum(value <= 1.0 + 1e-12 for value in ratios),
                "worst_error_ratio": max(ratios) if ratios else None,
            }
        core_ratios = [metric_rows[metric]["mean_candidate_to_parent_error_ratio"] for metric in CORE]
        output[name] = {
            "paired_case_count": len(cases),
            "metrics": metric_rows,
            "core_geometric_mean_error_ratio": geometric_mean([value for value in core_ratios if value is not None]),
        }
    return output


def safety_analysis(rows: list[dict[str, Any]], parent: str) -> dict[str, Any]:
    lookup = case_lookup(rows)
    output = {}
    for name in sorted({row["name"] for row in rows if row.get("diagnostics") is not None}):
        values = [row for row in rows if row["name"] == name]
        accepted, fallback, missing_gate = [], [], []
        exact_mismatch = []
        array_audit_missing = []
        array_mismatch = []
        accepted_w_rows = []
        for row in values:
            gate = (row.get("diagnostics") or {}).get("reliability_gate")
            if not isinstance(gate, dict):
                missing_gate.append(row["case_id"])
                continue
            is_accepted = bool(gate.get("accepted"))
            (accepted if is_accepted else fallback).append(row["case_id"])
            parent_row = lookup.get((row["case_id"], parent))
            if parent_row is None:
                continue
            if not is_accepted:
                audit = (row.get("diagnostics") or {}).get("fallback_audit")
                if not isinstance(audit, dict) or audit.get(
                    "declared_exact_fallback"
                ) is not True:
                    array_audit_missing.append(row["case_id"])
                elif audit.get("bit_exact") is not True:
                    failed_keys = [
                        key
                        for key, value in dict(audit.get("per_key", {})).items()
                        if not (
                            value.get("dtype_equal") is True
                            and value.get("shape_equal") is True
                            and value.get("raw_tobytes_equal") is True
                        )
                    ]
                    array_mismatch.append(
                        {"case_id": row["case_id"], "failed_keys": failed_keys}
                    )
                for metric, value in row["metrics"].items():
                    base = parent_row["metrics"].get(metric)
                    if value is None or base is None:
                        continue
                    if not math.isclose(value, base, rel_tol=1e-10, abs_tol=1e-10):
                        exact_mismatch.append({"case_id": row["case_id"], "metric": metric, "candidate": value, "parent": base})
            else:
                value = row["metrics"].get("W-MPJPE_mm")
                base = parent_row["metrics"].get("W-MPJPE_mm")
                if value is not None and base is not None:
                    accepted_w_rows.append({
                        "case_id": row["case_id"],
                        "candidate_W-MPJPE_mm": value,
                        "parent_W-MPJPE_mm": base,
                        "ratio": value / max(base, 1e-12),
                        "absolute_harm_mm": value - base,
                    })
        accepted_w_ratios = [value["ratio"] for value in accepted_w_rows]
        harm_5 = [value for value in accepted_w_rows if value["ratio"] > 1.05]
        harm_10 = [value for value in accepted_w_rows if value["ratio"] > 1.10]
        harm_20 = [value for value in accepted_w_rows if value["ratio"] > 1.20]
        worst = (
            max(accepted_w_rows, key=lambda value: value["ratio"])
            if accepted_w_rows
            else None
        )
        output[name] = {
            "gate_enabled": any(
                bool(((row.get("diagnostics") or {}).get("reliability_gate") or {}).get("enabled"))
                for row in values
            ),
            "case_count": len(values),
            "accepted_count": len(accepted),
            "fallback_count": len(fallback),
            "acceptance_rate": len(accepted) / max(len(accepted) + len(fallback), 1),
            "missing_gate_cases": missing_gate,
            "fallback_array_exactness_passed": not array_audit_missing and not array_mismatch,
            "fallback_array_audit_missing_cases": array_audit_missing,
            "fallback_array_mismatches": array_mismatch,
            "fallback_metric_exactness_passed": not exact_mismatch,
            "fallback_metric_mismatches": exact_mismatch,
            "accepted_W_evaluable_count": len(accepted_w_rows),
            "accepted_W_harm_over_5pct": len(harm_5),
            "accepted_W_harm_over_10pct": len(harm_10),
            "accepted_W_harm_over_20pct": len(harm_20),
            "accepted_W_harm_over_5pct_rate": len(harm_5) / max(len(accepted_w_rows), 1),
            "accepted_W_harm_over_10pct_rate": len(harm_10) / max(len(accepted_w_rows), 1),
            "accepted_W_harm_over_20pct_rate": len(harm_20) / max(len(accepted_w_rows), 1),
            "accepted_W_harm_over_5pct_cases": harm_5,
            "accepted_W_harm_over_10pct_cases": harm_10,
            "accepted_W_harm_over_20pct_cases": harm_20,
            "worst_accepted_W_ratio": None if worst is None else worst["ratio"],
            "worst_accepted_W_case": worst,
            "accepted_W_improvement_rate": sum(value < 1.0 for value in accepted_w_ratios) / max(len(accepted_w_ratios), 1),
        }
    return output


def stratified(rows: list[dict[str, Any]], metrics: list[str]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for angle in sorted({str(row["angle_stratum"]) for row in rows}):
        output[angle] = {}
        for name in sorted({row["name"] for row in rows}):
            values = [row for row in rows if str(row["angle_stratum"]) == angle and row["name"] == name]
            output[angle][name] = {
                "case_count": len(values),
                "metrics": {metric: mean([row["metrics"][metric] for row in values if row["metrics"].get(metric) is not None]) for metric in metrics},
            }
    return output


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(partial, path)


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be positive")
    provenance = validate_protocol_source(
        args.candidate_report, args.protocol_state, args.split
    )
    source = provenance["report"]
    rows = normalized_rows(source)
    metrics = metric_names(rows)
    recording_rows, aggregate = macro_tables(
        rows, metrics, int(args.seed), int(args.bootstrap_samples)
    )
    paired = paired_analysis(rows, args.parent, metrics)
    safety = safety_analysis(rows, args.parent)
    payload = {
        "schema_version": SCHEMA,
        "split": args.split,
        "candidate_report": str(args.candidate_report.resolve()),
        "candidate_report_sha256": provenance["report_sha256"],
        "protocol_state": str(args.protocol_state.resolve()),
        "protocol_state_sha256": provenance["state_sha256"],
        "run_identity_sha256": provenance["state"]["run_identity_sha256"],
        "candidate_source": provenance["candidate_source"],
        "candidate_source_sha256": provenance["candidate_source_sha256"],
        "parent": args.parent,
        "bootstrap_samples": int(args.bootstrap_samples),
        "bootstrap_seed": int(args.seed),
        "aggregation": "mean over three cases within recording, then equal-weight mean/bootstrap over recordings",
        "case_count": len({row["case_id"] for row in rows}),
        "selected_case_count": provenance["selected_case_count"],
        "evaluator_unavailable_case_count": provenance[
            "evaluator_unavailable_case_count"
        ],
        "recording_count": len({row["recording"] for row in rows}),
        "methods": aggregate,
        "paired_to_parent": paired,
        "safety": safety,
        "angle_strata": stratified(rows, metrics),
        "protocol_notes": {
            "test_tuning": False,
            "literature_context_directly_comparable": False,
            "w_wa_alignment": "one shared Sim(3) per clip across both people",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    case_flat = [
        {"case_id": row["case_id"], "recording": row["recording"], "angle_stratum": row["angle_stratum"], "name": row["name"], **row["metrics"]}
        for row in rows
    ]
    write_csv(args.output_dir / "case_metrics.csv", case_flat, ["case_id", "recording", "angle_stratum", "name", *metrics])
    write_csv(args.output_dir / "recording_metrics.csv", recording_rows, ["name", "recording", "case_count", *metrics])
    payload["case_metrics"] = str((args.output_dir / "case_metrics.csv").resolve())
    payload["case_metrics_sha256"] = sha256(args.output_dir / "case_metrics.csv")
    payload["recording_metrics"] = str(
        (args.output_dir / "recording_metrics.csv").resolve()
    )
    payload["recording_metrics_sha256"] = sha256(
        args.output_dir / "recording_metrics.csv"
    )
    output = args.output_dir / "summary.json"
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(partial, output)
    print(json.dumps({"output": str(output.resolve()), "cases": payload["case_count"], "recordings": payload["recording_count"], "methods": len(aggregate)}, indent=2))


if __name__ == "__main__":
    main()
