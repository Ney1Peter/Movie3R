#!/usr/bin/env python3
"""Audit a completed AIST++ MC150 run before it is promoted to the next stage.

The audit does not judge scores or choose examples.  It only verifies the
frozen manifest denominator, RGB/evaluator separation, runtime transaction
contract, cache shapes, and presence of every immutable output.  A failed
audit stops promotion rather than silently substituting a different result.
"""

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


ALLOWED_PROTOCOLS = {"MC150-3", "MC150-4"}
METHODS = ("m0_strict_human3r", "m1_clean_reset", "m3_b0_only", "m4_b0_identity", "m15_bridge3r_fixed_v19")
ARRAYS = ("cameras_c2w", "joints_world", "persistent_ids", "valid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--evaluator-manifest", type=Path, required=True)
    parser.add_argument("--protocol", choices=sorted(ALLOWED_PROTOCOLS), required=True)
    parser.add_argument("--role", choices=("pilot", "test"), required=True)
    parser.add_argument("--max-cases", type=int, default=0, help="Audit the first N frozen rows (0 means all).")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fail(errors: list[str], value: bool, text: str) -> None:
    if not value:
        errors.append(text)


def main() -> None:
    args = parse_args()
    run = args.run_dir.resolve()
    if args.max_cases < 0:
        raise SystemExit("--max-cases must be nonnegative")
    runtime_rows, evaluator_rows = read_jsonl(args.runtime_manifest.resolve()), read_jsonl(args.evaluator_manifest.resolve())
    if args.max_cases:
        runtime_rows = runtime_rows[:args.max_cases]
        selected_ids = {str(row.get("case_id")) for row in runtime_rows}
        evaluator_rows = [row for row in evaluator_rows if str(row.get("case_id")) in selected_ids]
    expected = {str(row.get("case_id")) for row in runtime_rows}
    evaluator_ids = {str(row.get("case_id")) for row in evaluator_rows}
    errors: list[str] = []
    fail(errors, bool(expected) and expected == evaluator_ids, "runtime/evaluator frozen case sets differ")
    fail(errors, all(row.get("protocol") == args.protocol and row.get("role") == args.role for row in runtime_rows + evaluator_rows), "manifest protocol or role drift")
    summary_path = run / "run_summary.json"
    fail(errors, summary_path.is_file(), "missing run_summary.json")
    summary: dict[str, Any] = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    result_rows = summary.get("results", [])
    result_ids = [str(row.get("case_id")) for row in result_rows]
    fail(errors, set(result_ids) == expected and len(result_ids) == len(expected), "run summary does not contain every frozen case exactly once")
    allowed_status = {"ok", "resumed_complete"}
    fail(errors, all(str(row.get("status")) in allowed_status for row in result_rows), "run summary contains a non-complete status")
    per_case: dict[str, Any] = {}
    detector_totals = {"tp": 0, "fp": 0, "fn": 0, "reported_cases": 0}
    for case_id in sorted(expected):
        safe = case_id.replace("/", "_")
        cache = run / "predictions" / f"{safe}.npz"
        runtime_path = cache.with_suffix(".runtime.json")
        metric_path = run / "metrics" / f"{safe}.json"
        for path in (cache, runtime_path, metric_path):
            fail(errors, path.is_file(), f"{case_id}: missing {path.name}")
        if not all(path.is_file() for path in (cache, runtime_path, metric_path)):
            continue
        report, metric = json.loads(runtime_path.read_text(encoding="utf-8")), json.loads(metric_path.read_text(encoding="utf-8"))
        record, contract = report.get("record", {}), report.get("runtime_contract", {})
        fail(errors, record.get("case_id") == case_id and record.get("protocol") == args.protocol and record.get("role") == args.role, f"{case_id}: runtime record drift")
        fail(errors, metric.get("case_id") == case_id and metric.get("protocol") == args.protocol, f"{case_id}: evaluator record drift")
        fail(errors, not bool(contract.get("gt_in_runtime")) and not bool(contract.get("camera_or_cut_in_runtime")), f"{case_id}: runtime accessed protected evaluator fields")
        fail(errors, int(contract.get("future_frames_at_boundary", -1)) == 0 and not bool(contract.get("pre_frames_rewritten_after_event")), f"{case_id}: non-causal prefix rewrite contract")
        detector = report.get("runtime", {}).get("causal_gru_detector", {})
        labels, events = detector.get("labels", []), detector.get("all_positive_indices", [])
        fail(errors, len(labels) == 150 and events == sorted(set(events)) and all(0 < int(item) < 150 for item in events), f"{case_id}: invalid causal detector event record")
        transactions = report.get("runtime", {}).get("multicut_transaction", {})
        fail(errors, transactions.get("events") == events, f"{case_id}: transaction events differ from detector events")
        fail(errors, not metric.get("errors", {}), f"{case_id}: evaluator method errors: {metric.get('errors')}")
        with np.load(cache, allow_pickle=False) as archive:
            for method in METHODS:
                for array in ARRAYS:
                    key = f"{method}__{array}"
                    fail(errors, key in archive.files, f"{case_id}: missing cache array {key}")
                valid_key = f"{method}__valid"
                if valid_key in archive.files:
                    valid = np.asarray(archive[valid_key], dtype=bool)
                    fail(errors, valid.ndim == 2 and valid.shape[0] == 150, f"{case_id}: invalid valid shape for {method}: {valid.shape}")
        scores = metric.get("detector", {})
        if scores.get("available"):
            for key in ("tp", "fp", "fn"):
                detector_totals[key] += int(scores.get(key, 0))
            detector_totals["reported_cases"] += 1
        per_case[case_id] = {
            "runtime_report_sha256": sha256_file(runtime_path),
            "metric_sha256": sha256_file(metric_path),
            "detector_events": events,
            "detector": scores,
            "method_coverage": {method: metric.get("methods", {}).get(method, {}).get("coverage", {}).get("valid_frame_coverage") for method in METHODS},
        }
    detector_totals["precision"] = detector_totals["tp"] / max(detector_totals["tp"] + detector_totals["fp"], 1)
    detector_totals["recall"] = detector_totals["tp"] / max(detector_totals["tp"] + detector_totals["fn"], 1)
    detector_totals["f1"] = 2 * detector_totals["tp"] / max(2 * detector_totals["tp"] + detector_totals["fp"] + detector_totals["fn"], 1)
    payload = {
        "schema_version": "Bridge3R-AIST-SinglePerson-MC150-campaign-audit-v1",
        "run_dir": str(run), "protocol": args.protocol, "role": args.role,
        "runtime_manifest": str(args.runtime_manifest.resolve()), "runtime_manifest_sha256": sha256_file(args.runtime_manifest.resolve()),
        "evaluator_manifest": str(args.evaluator_manifest.resolve()), "evaluator_manifest_sha256": sha256_file(args.evaluator_manifest.resolve()),
        "formal_manifest_case_count": len(expected), "audit_passed": not errors, "errors": errors,
        "detector": detector_totals, "per_case": per_case,
        "policy": "This audit checks execution integrity only; it neither filters cases nor gates promotion on metric values or detector accuracy.",
    }
    payload["content_sha256"] = canonical_json_digest(payload)
    atomic_json(args.output.resolve(), payload)
    print(json.dumps({"output": str(args.output), "passed": not errors, "errors": errors[:5], "cases": len(per_case)}, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
