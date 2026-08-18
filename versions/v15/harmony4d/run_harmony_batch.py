#!/usr/bin/env python3
"""Resumable GPU inference plus evaluator loop for one extracted sequence."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = Path(__file__).resolve().with_name("run_harmony_case.py")
EVALUATOR = Path(__file__).resolve().with_name("evaluate_harmony.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--extracted-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--start-line", type=int, default=1)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def run(command: list[str], env: dict[str, str], log: Path) -> tuple[int, float]:
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=REPO_ROOT, env=env, text=True, capture_output=True)
    elapsed = time.perf_counter() - started
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps({
        "command": command, "returncode": completed.returncode,
        "elapsed_seconds": elapsed, "stdout": completed.stdout, "stderr": completed.stderr,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(completed.stdout[-2000:], flush=True)
    if completed.returncode:
        print(completed.stderr[-4000:], file=sys.stderr, flush=True)
    return completed.returncode, elapsed


def main() -> None:
    args = parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = [(index, row) for index, row in enumerate(rows, 1) if index >= args.start_line]
    if args.max_cases:
        selected = selected[: args.max_cases]
    args.predictions.mkdir(parents=True, exist_ok=True)
    args.metrics.mkdir(parents=True, exist_ok=True)
    ledger_path = args.predictions / "batch_ledger.jsonl"
    env = dict(os.environ)
    env["TMPDIR"] = "/data/wangzheng/iJCV-CODE/data/Harmony4D_work/tmp"
    ledger = []
    for line, record in selected:
        case_id = str(record["case_id"])
        cache = args.predictions / f"{case_id}.npz"
        runtime = cache.with_suffix(".runtime.json")
        metric = args.metrics / f"{case_id}.json"
        if cache.is_file() and runtime.is_file() and metric.is_file() and not args.overwrite:
            item = {"line": line, "case_id": case_id, "status": "resumed_complete"}
            ledger.append(item)
            print(json.dumps(item), flush=True)
            continue
        runner = [
            sys.executable, str(RUNNER), "--manifest", str(args.manifest.resolve()),
            "--line", str(line), "--extracted-root", str(args.extracted_root.resolve()),
            "--output", str(cache.resolve()), "--device", args.device,
        ]
        if args.overwrite:
            runner.append("--overwrite")
        returncode, inference_seconds = run(runner, env, args.predictions / "logs" / f"{case_id}.inference.json")
        if returncode:
            item = {"line": line, "case_id": case_id, "status": "inference_error", "returncode": returncode}
            ledger.append(item)
            if not args.continue_on_error:
                break
            continue
        evaluator = [
            sys.executable, str(EVALUATOR), "--cache", str(cache.resolve()),
            "--runtime-report", str(runtime.resolve()),
            "--extracted-root", str(args.extracted_root.resolve()),
            "--output", str(metric.resolve()),
        ]
        returncode, evaluation_seconds = run(evaluator, env, args.metrics / "logs" / f"{case_id}.evaluation.json")
        item = {
            "line": line, "case_id": case_id,
            "status": "ok" if returncode == 0 else "evaluation_error",
            "inference_seconds": inference_seconds, "evaluation_seconds": evaluation_seconds,
        }
        ledger.append(item)
        if returncode and not args.continue_on_error:
            break
    with ledger_path.open("a", encoding="utf-8") as handle:
        for item in ledger:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    summary = {
        "manifest": str(args.manifest.resolve()),
        "selected": len(selected),
        "ok": sum(item["status"] in {"ok", "resumed_complete"} for item in ledger),
        "errors": sum("error" in item["status"] for item in ledger),
        "ledger": str(ledger_path),
    }
    (args.predictions / "batch_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["errors"] and not args.continue_on_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
