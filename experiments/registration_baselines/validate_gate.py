#!/usr/bin/env python3
"""Validate a one-shot Holdout executable gate without selecting parameters."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


METHODS = ("r0", "r1", "r2", "r3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", choices=("egobody", "egohumans"), required=True)
    parser.add_argument("--expected-cases", type=int, required=True)
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(partial, path)


def main() -> None:
    args = parse_args()
    root = args.output_root.resolve()
    rows = [json.loads(line) for line in args.runtime_manifest.resolve().read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != int(args.expected_cases) or any(row.get("split") != "holdout" for row in rows):
        raise ValueError("unexpected Holdout manifest")
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    if config.get("selection_split") != "development" or config.get("per_case_oracle") is not False:
        raise ValueError("configuration was not frozen on Development")
    status = {method: Counter() for method in METHODS}
    errors, evaluated, evaluator_unavailable = [], [], []
    for record in rows:
        case_id = str(record["case_id"])
        prediction = root / "predictions" / args.dataset / "holdout" / f"{case_id}.npz"
        runtime = prediction.with_suffix(".runtime.json")
        evaluation = root / "metrics/evaluations" / args.dataset / "holdout" / f"{case_id}.evaluation.json"
        required = [prediction, runtime, evaluation] + [
            root / "transforms" / args.dataset / "holdout" / case_id / f"{method}.json"
            for method in METHODS
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            errors.append({"case_id": case_id, "missing": missing})
            continue
        report = json.loads(evaluation.read_text(encoding="utf-8"))
        if report.get("errors") or set(report.get("methods", {})) != set(METHODS):
            errors.append({"case_id": case_id, "evaluation_errors": report.get("errors")})
            continue
        availability = {
            method: report["methods"][method].get("metric_availability", {}).get(
                "world_alignment", True
            )
            for method in METHODS
        }
        unavailable_methods = [method for method, available in availability.items() if not available]
        if unavailable_methods:
            reasons = {
                report["methods"][method].get("metric_availability", {}).get("reason")
                for method in unavailable_methods
            }
            if set(unavailable_methods) != set(METHODS) or len(reasons) != 1:
                errors.append({
                    "case_id": case_id,
                    "inconsistent_metric_availability": availability,
                    "reasons": sorted(str(value) for value in reasons),
                })
                continue
            evaluator_unavailable.append({
                "case_id": case_id,
                "reason": next(iter(reasons)),
                "retained_in_fixed_denominator": True,
            })
        for method in METHODS:
            status[method][str(report["registration"][method]["status"])] += 1
        evaluated.append(case_id)
    payload = {
        "schema_version": "Shot3R-traditional-registration-Holdout-gate-v1",
        "dataset": args.dataset, "split": "holdout",
        "expected_case_count": int(args.expected_cases), "evaluated_case_count": len(evaluated),
        "status_counts": {method: dict(values) for method, values in status.items()},
        "evaluator_unavailable_case_count": len(evaluator_unavailable),
        "evaluator_unavailable_cases": evaluator_unavailable,
        "errors": errors, "passed": len(evaluated) == int(args.expected_cases) and not errors,
        "parameter_changes_after_gate_allowed": False,
        "gate_role": "executability and leakage check only; no parameter selection",
        "frozen_config": str(args.config.resolve()),
    }
    output = root / "config" / f"holdout_gate_{args.dataset}.json"
    atomic_json(output, payload)
    print(json.dumps({"output": str(output), "passed": payload["passed"], "evaluated": len(evaluated)}, indent=2))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
