#!/usr/bin/env python3
"""Materialize an exact v17 regression table from frozen v16 evaluations.

The v17 candidate changes only the prediction-side reliability decision.  If
accepted, its geometry is bit-identical to v16 Harmony-Safe; if rejected, it
is bit-identical to the frozen parent.  Therefore existing metric rows can be
selected exactly without loading GT or rerunning the evaluator.  Candidate
diagnostics are nevertheless recomputed from the immutable prediction cache.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from versions.v16.harmony4d.causal_stabilization import (
    Candidate,
    coupled_boundary_register,
    reliability_gate_diagnostics,
)


OLD = "v16_harmony_safe"
PARENT = "v16_0_m15_geometry"
SOURCE = "m3_b0_only"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--prediction-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def input_reports(inputs: list[Path]) -> list[Path]:
    values: set[Path] = set()
    for source in inputs:
        if source.is_file():
            values.add(source.resolve())
        elif source.is_dir():
            values.update(source.resolve().glob("*.json"))
        else:
            raise FileNotFoundError(source)
    reports = []
    for path in sorted(values):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") == "Movie3R-v16-Harmony4D-causal-stabilization-probe-v1":
            reports.append(path)
    if not reports:
        raise ValueError("no v16 per-sequence reports")
    return reports


def load_candidate(path: Path) -> Candidate:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [row for row in payload["candidates"] if row["name"] != PARENT]
    if len(rows) != 1:
        raise ValueError(f"expected one non-parent candidate, got {len(rows)}")
    return Candidate(**rows[0])


def runtime_index(roots: list[Path]) -> dict[str, tuple[Path, dict[str, Any]]]:
    output: dict[str, tuple[Path, dict[str, Any]]] = {}
    for root in roots:
        for path in sorted(root.resolve().glob("*.runtime.json")):
            runtime = json.loads(path.read_text(encoding="utf-8"))
            case_id = str(runtime["record"]["case_id"])
            if case_id in output:
                raise ValueError(f"duplicate runtime for {case_id}")
            output[case_id] = (path, runtime)
    return output


def cache_path(runtime_path: Path, runtime: dict[str, Any]) -> Path:
    adjacent = runtime_path.with_name(
        runtime_path.name.removesuffix(".runtime.json") + ".npz"
    )
    path = adjacent if adjacent.is_file() else Path(runtime["cache"])
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def main() -> None:
    args = parse_args()
    candidate = load_candidate(args.candidate_json)
    runtimes = runtime_index(args.prediction_roots)
    args.output.mkdir(parents=True, exist_ok=True)
    written = []
    accepted_total = 0
    complete_total = 0
    for report_path in input_reports(args.inputs):
        source_payload = json.loads(report_path.read_text(encoding="utf-8"))
        payload = copy.deepcopy(source_payload)
        old_by_case = {
            row["case_id"]: row for row in source_payload.get("rows", [])
            if row.get("candidate") == OLD
        }
        parent_by_case = {
            row["case_id"]: row for row in source_payload.get("rows", [])
            if row.get("candidate") == PARENT
        }
        new_rows = [
            row for row in payload.get("rows", [])
            if row.get("candidate") != candidate.name
        ]
        for case_id, old_row in sorted(old_by_case.items()):
            if old_row.get("status") != "complete":
                failed = copy.deepcopy(old_row)
                failed["candidate"] = candidate.name
                new_rows.append(failed)
                continue
            if case_id not in runtimes or case_id not in parent_by_case:
                raise KeyError(case_id)
            runtime_path, runtime = runtimes[case_id]
            with np.load(cache_path(runtime_path, runtime), allow_pickle=False) as cache:
                prefix = SOURCE + "__"
                arrays = {
                    "joints_world": cache[prefix + "joints_world"],
                    "valid": cache[prefix + "valid"],
                    "persistent_ids": cache[prefix + "persistent_ids"],
                }
            pairs = [
                tuple(map(int, pair))
                for pair in runtime.get("geometry", {}).get("association", {}).get("pairs", [])
            ]
            _, boundary_debug = coupled_boundary_register(
                arrays, int(runtime["record"]["boundary_index"]), pairs,
                candidate.boundary_kind, candidate.boundary_blend,
                candidate.use_velocity_target, materialize=False,
            )
            gate_debug = reliability_gate_diagnostics(boundary_debug, candidate)
            diagnostics = {
                "candidate": candidate.__dict__,
                "boundary": boundary_debug,
                "reliability_gate": gate_debug,
                "runtime_contract": {
                    "gt_used": False,
                    "future_frames_used": 0,
                    "exact_m15_fallback": not gate_debug["accepted"],
                },
            }
            accepted = bool(gate_debug["accepted"])
            metric_source = old_row if accepted else parent_by_case[case_id]
            row = copy.deepcopy(old_row)
            row["candidate"] = candidate.name
            row["metrics"] = copy.deepcopy(metric_source["metrics"])
            row["within_shot_motion"] = copy.deepcopy(metric_source["within_shot_motion"])
            row["diagnostics"] = diagnostics
            row["exact_metric_source"] = OLD if accepted else PARENT
            row["regression_only"] = True
            new_rows.append(row)
            accepted_total += int(accepted)
            complete_total += 1
        payload["rows"] = new_rows
        payload["candidate_source"] = str(args.candidate_json.resolve())
        payload["candidate_count"] = len({
            row.get("candidate") for row in new_rows if row.get("candidate")
        })
        payload["aggregate"] = {
            "status": "deferred_to_multisequence_aggregator",
            "regression_exactness": (
                "accepted rows select bit-identical v16 metrics; fallback rows select "
                "bit-identical parent metrics"
            ),
        }
        payload["v17_regression_contract"] = {
            "gt_loaded": False,
            "test_used_as_fresh_validation": False,
            "purpose": "seen-test regression diagnostic only",
        }
        destination = args.output / report_path.name
        destination.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        written.append(str(destination.resolve()))
    print(json.dumps({
        "output": str(args.output.resolve()),
        "reports": len(written),
        "complete_cases": complete_total,
        "accepted": accepted_total,
        "fallback": complete_total - accepted_total,
        "files": written,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
