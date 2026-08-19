#!/usr/bin/env python3
"""Evaluate independent Harmony4D cases in parallel and merge exact probe rows.

The frozen evaluator and all metric definitions remain in the v16 probe.  This
wrapper changes only scheduling: every immutable case cache is evaluated by an
ordinary one-case probe process, then the rows are deterministically merged and
the original aggregate function is rerun.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from versions.v16.harmony4d.probe_causal_stabilization import (  # noqa: E402
    aggregate,
    jsonable,
    load_candidates,
    runtime_paths,
    write_csv,
)


PROBE = REPO_ROOT / "versions/v16/harmony4d/probe_causal_stabilization.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--extracted-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-json", type=Path)
    parser.add_argument("--source-method", default="m3_b0_only")
    parser.add_argument("--reference-methods", nargs="*", default=[])
    parser.add_argument("--include-case-regex")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(jsonable(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(partial, path)


def case_ids(args: argparse.Namespace) -> list[str]:
    values = []
    for path in runtime_paths(args.prediction_roots):
        runtime = json.loads(path.read_text(encoding="utf-8"))
        case_id = str(runtime["record"]["case_id"])
        if args.include_case_regex and not re.search(args.include_case_regex, case_id):
            continue
        values.append(case_id)
    if not values:
        raise ValueError("no cases match the requested roots/regex")
    if len(values) != len(set(values)):
        raise ValueError("duplicate case IDs after filtering")
    return values


def valid_cached_shard(path: Path, case_id: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    observed = {
        str(row["case_id"])
        for row in payload.get("rows", []) + payload.get("reference_rows", [])
    }
    return (
        payload.get("schema_version") == "Movie3R-v16-Harmony4D-causal-stabilization-probe-v1"
        and int(payload.get("case_count", 0)) == 1
        and observed == {case_id}
    )


def command_for(args: argparse.Namespace, case_id: str, output: Path) -> list[str]:
    command = [
        sys.executable, str(PROBE),
        "--prediction-roots", *[str(path) for path in args.prediction_roots],
        "--extracted-root", str(args.extracted_root),
        "--output", str(output),
        "--source-method", str(args.source_method),
        "--include-case-regex", f"^{re.escape(case_id)}$",
    ]
    if args.candidate_json is not None:
        command.extend(("--candidate-json", str(args.candidate_json)))
    if args.reference_methods:
        command.extend(("--reference-methods", *args.reference_methods))
    return command


def run_shards(args: argparse.Namespace, cases: list[str], root: Path) -> list[Path]:
    workers = max(1, min(int(args.workers), len(cases)))
    root.mkdir(parents=True, exist_ok=True)
    (root / "tmp").mkdir(exist_ok=True)
    pending: deque[tuple[int, str, Path]] = deque()
    outputs = []
    for index, case_id in enumerate(cases):
        path = root / f"case_{index:04d}.json.partial"
        outputs.append(path)
        if not valid_cached_shard(path, case_id):
            pending.append((index, case_id, path))
    active: dict[int, tuple[subprocess.Popen[Any], Any, int, str, Path, float]] = {}
    failures = []
    while pending or active:
        while pending and len(active) < workers:
            index, case_id, output = pending.popleft()
            log = root / f"case_{index:04d}.log"
            handle = log.open("w", encoding="utf-8")
            command = command_for(args, case_id, output)
            handle.write("COMMAND " + json.dumps(command, ensure_ascii=False) + "\n")
            handle.flush()
            process = subprocess.Popen(
                command, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT,
                env={**os.environ, "TMPDIR": str(root / "tmp")},
            )
            active[process.pid] = (process, handle, index, case_id, output, time.perf_counter())
        if not active:
            continue
        time.sleep(0.5)
        for pid, (process, handle, index, case_id, output, started) in list(active.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            handle.close()
            if returncode or not valid_cached_shard(output, case_id):
                failures.append({
                    "case_id": case_id, "index": index, "returncode": returncode,
                    "seconds": time.perf_counter() - started,
                })
            del active[pid]
    if failures:
        raise RuntimeError(f"parallel probe failures: {failures}")
    return outputs


def merge(args: argparse.Namespace, cases: list[str], shards: list[Path]) -> dict[str, Any]:
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in shards]
    rows = sorted(
        [row for payload in payloads for row in payload.get("rows", [])],
        key=lambda row: (str(row["case_id"]), str(row["candidate"])),
    )
    references = sorted(
        [row for payload in payloads for row in payload.get("reference_rows", [])],
        key=lambda row: (str(row["case_id"]), str(row["method"])),
    )
    observed = {str(row["case_id"]) for row in rows}
    if observed != set(cases):
        raise ValueError(f"merged rows cover {len(observed)} cases, expected {len(cases)}")
    candidates = load_candidates(args.candidate_json)
    skipped = sorted(
        [row for payload in payloads for row in payload.get("skipped_cases", [])],
        key=lambda row: str(row["case_id"]),
    )
    errors = [error for payload in payloads for error in payload.get("errors", [])]
    base = payloads[0]
    result = {
        "schema_version": base["schema_version"],
        "source_method": args.source_method,
        "reference_methods": list(args.reference_methods),
        "prediction_roots": [str(path.resolve()) for path in args.prediction_roots],
        "extracted_root": str(args.extracted_root.resolve()),
        "candidate_source": (
            str(args.candidate_json.resolve()) if args.candidate_json
            else "frozen_exploration_grid_in_code"
        ),
        "include_case_regex": args.include_case_regex,
        "candidate_count": len(candidates),
        "case_count": len(cases),
        "complete_case_count": len({
            str(row["case_id"]) for row in rows if row.get("status") == "complete"
        }),
        "skipped_cases": skipped,
        "errors": errors,
        "aggregate": aggregate(rows, candidates),
        "rows": rows,
        "reference_rows": references,
        "contract": {
            **base.get("contract", {}),
            "parallel_scheduling_only": True,
            "parallel_workers": max(1, min(int(args.workers), len(cases))),
            "merge_order": "case_id_then_method",
            "metric_implementation": str(PROBE.resolve()),
        },
    }
    atomic_json(args.output, result)
    write_csv(args.output.with_suffix(".csv"), rows)
    return result


def main() -> None:
    args = parse_args()
    cases = case_ids(args)
    shard_root = args.output.parent / f".{args.output.stem}_parallel_shards"
    shards = run_shards(args, cases, shard_root)
    result = merge(args, cases, shards)
    print(json.dumps({
        "output": str(args.output.resolve()), "cases": len(cases),
        "complete_cases": result["complete_case_count"],
        "evaluator_unavailable": len(result["skipped_cases"]),
        "errors": len(result["errors"]), "workers": int(args.workers),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
