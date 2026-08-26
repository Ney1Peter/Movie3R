#!/usr/bin/env python3
"""Wait for a declared PromptHMR shard partition, then seal its test ledger.

This monitor has no access to scores and never manages GPU workers.  It only
checks the artifact presence required by the pre-written partition lock and
invokes the immutable finalizer once all assigned cases exist.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition-lock", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--evaluator-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    return parser.parse_args()


def safe_name(case_id: str) -> str:
    return case_id.replace("/", "_")


def missing(lock: dict, runtime: list[dict]) -> list[str]:
    absent = []
    for item in lock["case_intervals"]:
        start, stop = item["inclusive_lines"]
        root = Path(item["output_root"])
        for line in range(int(start), int(stop) + 1):
            name = safe_name(str(runtime[line - 1]["case_id"]))
            expected = (root / "metrics" / f"{name}.json", root / "predictions" / f"{name}.npz", root / "predictions" / f"{name}.runtime.json", root / "predictions" / f"{name}.adapter.json")
            if not all(path.is_file() for path in expected):
                absent.append(f"line {line}")
    return absent


def main() -> None:
    args = parse_args()
    if args.poll_seconds < 10:
        raise ValueError("--poll-seconds must be at least ten seconds")
    lock = json.loads(args.partition_lock.read_text(encoding="utf-8"))
    runtime = [json.loads(line) for line in args.runtime_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.output_dir.exists():
        print(f"final ledger already exists: {args.output_dir}", flush=True); return
    while True:
        absent = missing(lock, runtime)
        if not absent:
            command = [
                sys.executable, str(Path(__file__).with_name("finalize_prompthmr_aist_test.py")),
                "--partition-lock", str(args.partition_lock.resolve()), "--runtime-manifest", str(args.runtime_manifest.resolve()),
                "--evaluator-manifest", str(args.evaluator_manifest.resolve()), "--output-dir", str(args.output_dir.resolve()),
            ]
            completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            print(completed.stdout, flush=True)
            if completed.returncode:
                raise SystemExit(completed.returncode)
            return
        print(json.dumps({"status": "waiting_for_declared_promptHMR_test_shards", "missing_artifact_cases": len(absent), "first": absent[:5]}), flush=True)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
