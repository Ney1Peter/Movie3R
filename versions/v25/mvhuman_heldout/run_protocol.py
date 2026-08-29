#!/usr/bin/env python3
"""Resumable multi-GPU driver for the frozen MVHuman MVH150 protocol."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
RUNNER = HERE / "run_case.py"
EVALUATOR = HERE / "evaluate_case.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--python", dest="python_executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--max-cases", type=int)
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_one(
    line: int,
    record: dict[str, Any],
    gpu: int,
    args: argparse.Namespace,
    runtime_manifest: Path,
    evaluator_manifest: Path,
) -> dict[str, Any]:
    case_id = str(record["case_id"])
    cache = args.output_root / "predictions" / f"{case_id}.npz"
    report = cache.with_suffix(".runtime.json")
    metric = args.output_root / "metrics" / f"{case_id}.json"
    log = args.output_root / "logs" / f"{case_id}.log"
    for parent in (cache.parent, metric.parent, log.parent, args.output_root / "work"):
        parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    commands = []
    if not (cache.is_file() and report.is_file()):
        commands.append(
            [
                str(args.python_executable), str(RUNNER),
                "--runtime-manifest", str(runtime_manifest), "--line", str(line),
                "--derived-root", str(args.protocol_root / "derived"),
                "--work-dir", str(args.output_root / "work" / f"gpu{gpu}"),
                "--output", str(cache), "--device", f"cuda:{gpu}",
            ]
        )
    if not metric.is_file():
        commands.append(
            [
                str(args.python_executable), str(EVALUATOR),
                "--cache", str(cache), "--runtime-report", str(report),
                "--evaluator-manifest", str(evaluator_manifest), "--case-id", case_id,
                "--audit-root", str(args.audit_root), "--output", str(metric),
            ]
        )
    with log.open("a", encoding="utf-8") as handle:
        for command in commands:
            handle.write(f"COMMAND {json.dumps(command)}\n")
            handle.flush()
            completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, text=True)
            if completed.returncode:
                return {"case_id": case_id, "line": line, "gpu": gpu, "status": "failed", "returncode": completed.returncode, "log": str(log), "seconds": time.time() - started}
    return {"case_id": case_id, "line": line, "gpu": gpu, "status": "ok", "resumed": not commands, "seconds": time.time() - started}


def worker(gpu: int, tasks: list[tuple[int, dict[str, Any]]], args: argparse.Namespace, runtime: Path, evaluator: Path) -> list[dict[str, Any]]:
    results = []
    for line, record in tasks:
        result = run_one(line, record, gpu, args, runtime, evaluator)
        results.append(result)
        print(json.dumps(result), flush=True)
    return results


def main() -> None:
    args = parse_args()
    args.protocol_root = args.protocol_root.resolve()
    args.audit_root = args.audit_root.resolve()
    args.output_root = args.output_root.resolve()
    args.python_executable = args.python_executable.expanduser()
    if not args.python_executable.is_absolute():
        args.python_executable = (Path.cwd() / args.python_executable).absolute()
    runtime = args.protocol_root / "manifests" / "test_runtime.jsonl"
    evaluator = args.protocol_root / "manifests" / "test_evaluator.jsonl"
    if not runtime.is_file() or not evaluator.is_file() or not args.python_executable.is_file():
        raise FileNotFoundError("Protocol manifests or requested interpreter are missing")
    rows = [json.loads(line) for line in runtime.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.max_cases is not None:
        rows = rows[: args.max_cases]
    tasks = [(index + 1, record) for index, record in enumerate(rows)]
    shards = {gpu: tasks[offset::len(args.gpus)] for offset, gpu in enumerate(args.gpus)}
    launched = time.time()
    outcomes = []
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = {executor.submit(worker, gpu, shard, args, runtime, evaluator): gpu for gpu, shard in shards.items()}
        for future in as_completed(futures):
            outcomes.extend(future.result())
    summary = {
        "schema_version": "Bridge3R-MVHuman-Heldout-MVH150-run-summary-v1",
        "protocol_root": str(args.protocol_root),
        "gpus": args.gpus,
        "case_count": len(tasks),
        "ok": sum(row["status"] == "ok" for row in outcomes),
        "failed": [row for row in outcomes if row["status"] != "ok"],
        "seconds": time.time() - launched,
        "outcomes": sorted(outcomes, key=lambda row: row["line"]),
    }
    atomic_json(args.output_root / "run_summary.json", summary)
    print(json.dumps({key: summary[key] for key in ("case_count", "ok", "failed", "seconds")}, indent=2))
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
