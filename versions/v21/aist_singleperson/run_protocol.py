#!/usr/bin/env python3
"""Resumable multi-GPU AIST++ CS150 runner with runtime/evaluator separation.

Only the per-case GPU subprocess receives a row from the runtime manifest.
The coordinating process joins its cache with an evaluator row *after*
inference, then invokes the CPU evaluator.  This explicit separation makes it
impossible for a runner command to receive calibration, labels, or a cut
index through its manifest argument.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    from .protocol import DEFAULT_DERIVED_ROOT, atomic_json, canonical_json_digest, sha256_file
except ImportError:
    from protocol import DEFAULT_DERIVED_ROOT, atomic_json, canonical_json_digest, sha256_file  # type: ignore


SCHEMA = "Bridge3R-AIST-SinglePerson-CS150-run-protocol-v1"
RUNNER = Path(__file__).with_name("run_aist_case.py")
EVALUATOR = Path(__file__).with_name("evaluate_aist.py")
EXPECTED_RUNTIME_KEYS = {"case_id", "dataset", "protocol", "role", "input_video", "fps", "num_frames"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--evaluator-manifest", type=Path, required=True)
    parser.add_argument("--derived-root", type=Path, default=DEFAULT_DERIVED_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--devices", required=True, help="Comma-separated physical CUDA indices, at most three.")
    parser.add_argument("--role", choices=("pilot", "test"), required=True)
    parser.add_argument("--start-line", type=int, default=1)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--keep-decoded-frames", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"Blank JSONL row at {path}:{number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Non-object JSONL row at {path}:{number}")
        rows.append(value)
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def resolve_safe(derived_root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise ValueError(f"Unsafe derived path: {relative!r}")
    root = derived_root.resolve()
    resolved = (root / candidate).resolve()
    if root not in resolved.parents or not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def validate_pair(args: argparse.Namespace) -> list[tuple[int, dict[str, Any], dict[str, Any]]]:
    runtime_rows = read_jsonl(args.runtime_manifest)
    evaluator_rows = read_jsonl(args.evaluator_manifest)
    evaluator_by_id = {str(row.get("case_id")): row for row in evaluator_rows}
    if len(evaluator_by_id) != len(evaluator_rows):
        raise ValueError("Duplicate evaluator case IDs")
    joined = []
    for line, runtime in enumerate(runtime_rows, start=1):
        if set(runtime) != EXPECTED_RUNTIME_KEYS:
            raise ValueError(f"Runtime schema drift at line {line}: {sorted(runtime)}")
        if runtime["dataset"] != "AIST++" or runtime["protocol"] != "CS150" or runtime["role"] != args.role:
            raise ValueError(f"Unexpected runtime contract at line {line}")
        if int(runtime["fps"]) != 30 or int(runtime["num_frames"]) != 150:
            raise ValueError(f"Unexpected runtime temporal contract at line {line}")
        case_id = str(runtime["case_id"])
        if case_id not in evaluator_by_id:
            raise ValueError(f"No evaluator counterpart for {case_id}")
        evaluator = evaluator_by_id[case_id]
        if evaluator.get("role") != args.role or evaluator.get("protocol") != "CS150":
            raise ValueError(f"Evaluator contract mismatch for {case_id}")
        resolve_safe(args.derived_root, str(runtime["input_video"]))
        resolve_safe(args.derived_root, str(evaluator["label"]))
        joined.append((line, runtime, evaluator))
    if set(evaluator_by_id) != {str(row["case_id"]) for _, row, _ in joined}:
        raise ValueError("Runtime/evaluator manifest case sets differ")
    selected = [row for row in joined if row[0] >= args.start_line]
    return selected[:args.max_cases] if args.max_cases else selected


def command_log(command: list[str], cwd: Path) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return {"command": command, "returncode": completed.returncode, "seconds": time.perf_counter() - started, "stdout": completed.stdout, "stderr": completed.stderr}


def execute_case(args: argparse.Namespace, device: int, item: tuple[int, dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    line, runtime, evaluator = item
    case_id = str(runtime["case_id"])
    safe_name = case_id.replace("/", "_")
    output = args.output_dir.resolve()
    cache = output / "predictions" / f"{safe_name}.npz"
    report = cache.with_suffix(".runtime.json")
    metric = output / "metrics" / f"{safe_name}.json"
    log_path = output / "logs" / f"{safe_name}.json"
    if cache.is_file() and report.is_file() and metric.is_file():
        return {"line": line, "case_id": case_id, "device": device, "status": "resumed_complete"}
    command = [
        sys.executable, str(RUNNER), "--runtime-manifest", str(args.runtime_manifest.resolve()), "--line", str(line),
        "--derived-root", str(args.derived_root.resolve()), "--work-dir", str((output / "frame_work" / f"cuda{device}").resolve()),
        "--output", str(cache), "--device", f"cuda:{device}", "--size", str(args.size),
    ]
    if args.keep_decoded_frames:
        command.append("--keep-decoded-frames")
    inference = command_log(command, Path.cwd())
    log = {"line": line, "case_id": case_id, "device": device, "inference": inference}
    if inference["returncode"]:
        atomic_json(log_path, log)
        return {"line": line, "case_id": case_id, "device": device, "status": "inference_error", "log": str(log_path), "returncode": inference["returncode"]}
    evaluation = command_log([
        sys.executable, str(EVALUATOR), "--cache", str(cache), "--runtime-report", str(report),
        "--label", str(resolve_safe(args.derived_root, str(evaluator["label"]))), "--derived-root", str(args.derived_root.resolve()), "--output", str(metric),
    ], Path.cwd())
    log["evaluation"] = evaluation
    atomic_json(log_path, log)
    return {"line": line, "case_id": case_id, "device": device, "status": "ok" if not evaluation["returncode"] else "evaluation_error", "log": str(log_path), "returncode": evaluation["returncode"]}


def main() -> None:
    args = parse_args()
    devices = tuple(int(value.strip()) for value in args.devices.split(",") if value.strip())
    if not devices or len(devices) > 3 or any(value < 0 for value in devices) or len(set(devices)) != len(devices):
        raise SystemExit("--devices must contain one to three distinct nonnegative CUDA indices")
    if args.size != 512 or args.start_line < 1 or args.max_cases < 0:
        raise SystemExit("invalid --size, --start-line, or --max-cases")
    items = validate_pair(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    # One queue per GPU prevents accidental duplicate case writes and keeps a
    # loaded CUDA process isolated to one physical device at a time.
    buckets = [items[index::len(devices)] for index in range(len(devices))]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [executor.submit(lambda dev=device, rows=bucket: [execute_case(args, dev, row) for row in rows]) for device, bucket in zip(devices, buckets)]
        for future in concurrent.futures.as_completed(futures):
            rows = future.result()
            results.extend(rows)
            for row in rows:
                print(json.dumps(row, ensure_ascii=False), flush=True)
                if "error" in row["status"] and not args.continue_on_error:
                    # The other device workers may finish their current case;
                    # no new source selection or output overwrite occurs.
                    raise SystemExit("Stopping after first case error; rerun with --continue-on-error after inspection")
    results.sort(key=lambda row: int(row["line"]))
    payload = {
        "schema_version": SCHEMA, "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "role": args.role, "runtime_manifest": str(args.runtime_manifest.resolve()), "runtime_manifest_sha256": sha256_file(args.runtime_manifest.resolve()),
        "evaluator_manifest": str(args.evaluator_manifest.resolve()), "evaluator_manifest_sha256": sha256_file(args.evaluator_manifest.resolve()),
        "devices": list(devices), "selected_case_count": len(items), "results": results,
        "runtime_evaluator_separation": "GPU runner received runtime rows only; evaluator rows were consumed only after cache creation.",
    }
    payload["content_sha256"] = canonical_json_digest(payload)
    summary = args.output_dir / "run_summary.json"
    atomic_json(summary, payload)
    error_count = sum("error" in row["status"] for row in results)
    print(json.dumps({"summary": str(summary), "selected": len(items), "ok": len(results) - error_count, "errors": error_count}, ensure_ascii=False, indent=2))
    if error_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
