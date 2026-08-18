#!/usr/bin/env python3
"""Audit frozen Harmony4D test caches, reports, and explicit failures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_METHODS = 17
EXPECTED_COMMIT = "3d022495bd9f3e870ef4de924e7a939042f33887"
EVALUATION_SCHEMA = "Movie3R-Harmony4D-evaluation-v1"
RUNTIME_SCHEMA = "Movie3R-Harmony4D-runtime-cache-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--metrics", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-commit", default=EXPECTED_COMMIT)
    parser.add_argument("--verify-cache-sha256", action="store_true")
    parser.add_argument("--allow-no-initial-match", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_by_case(paths: list[Path], suffix: str) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for root in paths:
        for path in sorted(root.rglob(f"*{suffix}")):
            case_id = path.name.removesuffix(suffix)
            if case_id in output:
                raise ValueError(f"Duplicate artifact for {case_id}: {output[case_id]} and {path}")
            output[case_id] = path.resolve()
    return output


def evaluation_reports(roots: list[Path]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for root in roots:
        for path in sorted(root.rglob("h4d_test_*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if payload.get("schema_version") != EVALUATION_SCHEMA:
                continue
            case_id = str(payload["case_id"])
            if case_id in output:
                raise ValueError(f"Duplicate evaluation for {case_id}")
            output[case_id] = path.resolve()
    return output


def evaluator_error(case_id: str, roots: list[Path]) -> dict[str, Any] | None:
    candidates: list[Path] = []
    for root in roots:
        candidates.extend(root.rglob(f"{case_id}.evaluation.json"))
        candidates.extend(root.rglob(f"{case_id}.reevaluation.json"))
    for path in sorted(candidates):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        stderr = str(payload.get("stderr", ""))
        if int(payload.get("returncode", 0)) != 0:
            reason = (
                "no_initial_matched_people_for_shared_world_fit"
                if "No initial matched people for shared world fit" in stderr
                else "unclassified_evaluator_error"
            )
            return {
                "reason": reason,
                "log": str(path.resolve()),
                "returncode": int(payload.get("returncode", -1)),
                "stderr_tail": stderr[-2000:],
            }
    return None


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    records = [
        json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected = {str(record["case_id"]): record for record in records}
    if len(expected) != len(records):
        raise ValueError("Manifest contains duplicate case IDs")
    runtimes = unique_by_case([path.resolve() for path in args.predictions], ".runtime.json")
    reports = evaluation_reports([path.resolve() for path in args.metrics])
    rows = []
    hard_errors: list[str] = []
    provenance_errors: list[str] = []
    verified_cache_bytes = 0
    for case_id, record in expected.items():
        runtime_path = runtimes.get(case_id)
        metric_path = reports.get(case_id)
        row: dict[str, Any] = {
            "case_id": case_id,
            "sequence": record["sequence"],
            "angle_stratum": record["angle_stratum"],
            "runtime": str(runtime_path) if runtime_path else None,
            "evaluation": str(metric_path) if metric_path else None,
        }
        if runtime_path is None:
            row["status"] = "missing_runtime"
            hard_errors.append(case_id + ":missing_runtime")
            rows.append(row)
            continue
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        cache_path = runtime_path.with_name(case_id + ".npz")
        checks = {
            "runtime_schema": runtime.get("schema_version") == RUNTIME_SCHEMA,
            "record_exact": runtime.get("record") == record,
            "method_count": len(runtime.get("methods", [])) == EXPECTED_METHODS,
            "commit": runtime.get("provenance", {}).get("commit") == args.expected_commit,
            "tracked_worktree_clean": not bool(runtime.get("provenance", {}).get("tracked_worktree_dirty", True)),
            "gt_not_in_runtime": runtime.get("runtime_contract", {}).get("gt_in_runtime") is False,
            "no_future_at_boundary": int(runtime.get("runtime_contract", {}).get("future_frames_at_boundary", -1)) == 0,
            "pre_cut_not_mutated": runtime.get("runtime_contract", {}).get("pre_cut_frames_mutated") is False,
            "cache_exists": cache_path.is_file(),
        }
        if args.verify_cache_sha256 and cache_path.is_file():
            actual = sha256(cache_path)
            checks["cache_sha256"] = actual == runtime.get("cache_sha256")
            verified_cache_bytes += cache_path.stat().st_size
            row["cache_sha256"] = actual
        failed_checks = sorted(key for key, value in checks.items() if not value)
        if failed_checks:
            provenance_errors.extend(f"{case_id}:{key}" for key in failed_checks)
        row["runtime_checks"] = checks
        row["cache"] = str(cache_path.resolve()) if cache_path.is_file() else None
        if metric_path is not None:
            report = json.loads(metric_path.read_text(encoding="utf-8"))
            metric_checks = {
                "evaluation_schema": report.get("schema_version") == EVALUATION_SCHEMA,
                "record_exact": report.get("record") == record,
                "method_count": len(report.get("methods", {})) == EXPECTED_METHODS,
            }
            row["evaluation_checks"] = metric_checks
            metric_failures = sorted(key for key, value in metric_checks.items() if not value)
            if metric_failures:
                hard_errors.extend(f"{case_id}:evaluation:{key}" for key in metric_failures)
                row["status"] = "invalid_evaluation"
            else:
                row["status"] = "complete"
        else:
            error = evaluator_error(case_id, [path.resolve() for path in args.metrics])
            row["evaluator_error"] = error
            if error and error["reason"] == "no_initial_matched_people_for_shared_world_fit":
                row["status"] = "evaluator_unavailable_no_initial_match"
                if not args.allow_no_initial_match:
                    hard_errors.append(case_id + ":no_initial_match_not_allowed")
            else:
                row["status"] = "missing_or_unknown_evaluation"
                hard_errors.append(case_id + ":missing_or_unknown_evaluation")
        rows.append(row)

    unexpected_runtime = sorted(set(runtimes) - set(expected))
    unexpected_evaluation = sorted(set(reports) - set(expected))
    if unexpected_runtime:
        hard_errors.extend("unexpected_runtime:" + value for value in unexpected_runtime)
    if unexpected_evaluation:
        hard_errors.extend("unexpected_evaluation:" + value for value in unexpected_evaluation)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    payload = {
        "schema_version": "Movie3R-Harmony4D-test-audit-v1",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "expected_cases": len(records),
        "runtime_cases": len(set(runtimes).intersection(expected)),
        "evaluation_cases": len(set(reports).intersection(expected)),
        "status_counts": counts,
        "evaluator_unavailable_fraction": counts.get("evaluator_unavailable_no_initial_match", 0) / max(len(records), 1),
        "cache_sha256_verified": bool(args.verify_cache_sha256),
        "verified_cache_bytes": verified_cache_bytes,
        "expected_commit": args.expected_commit,
        "provenance_errors": provenance_errors,
        "hard_errors": hard_errors,
        "unexpected_runtime": unexpected_runtime,
        "unexpected_evaluation": unexpected_evaluation,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "expected": len(records),
        "runtime": payload["runtime_cases"],
        "evaluation": payload["evaluation_cases"],
        "status_counts": counts,
        "provenance_errors": len(provenance_errors),
        "hard_errors": len(hard_errors),
    }, indent=2, ensure_ascii=False))
    if provenance_errors or hard_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
