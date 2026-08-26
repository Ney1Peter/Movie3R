#!/usr/bin/env python3
"""Resumable, separated AIST CS150 protocol runner for official GVHMR."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    from .protocol import DEFAULT_DERIVED_ROOT, atomic_json, canonical_json_digest, sha256_file
except ImportError:
    from protocol import DEFAULT_DERIVED_ROOT, atomic_json, canonical_json_digest, sha256_file  # type: ignore


SCHEMA = "Bridge3R-AIST-GVHMR-official-protocol-v1"
RUNNER, EVALUATOR = Path(__file__).with_name("run_gvhmr_case.py"), Path(__file__).with_name("evaluate_aist.py")
EXPECTED_RUNTIME_KEYS = {"case_id", "dataset", "protocol", "role", "input_video", "fps", "num_frames"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--evaluator-manifest", type=Path, required=True)
    parser.add_argument("--derived-root", type=Path, default=DEFAULT_DERIVED_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--converter", type=Path, required=True)
    parser.add_argument("--python", dest="python_executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--devices", required=True, help="one to three distinct physical CUDA indices")
    parser.add_argument("--role", choices=("pilot", "test"), required=True)
    parser.add_argument("--start-line", type=int, default=1)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def rows(path: Path) -> list[dict[str, Any]]:
    output = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"blank JSONL row at {path}:{number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"non-object JSONL row at {path}:{number}")
        output.append(value)
    if not output:
        raise ValueError(f"empty manifest: {path}")
    return output


def safe_file(root: Path, relative: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts or not value.parts:
        raise ValueError(f"unsafe relative data path: {relative!r}")
    root = root.resolve(); path = (root / value).resolve()
    if root not in path.parents or not path.is_file():
        raise FileNotFoundError(path)
    return path


def validate_pair(args: argparse.Namespace) -> list[tuple[int, dict[str, Any], dict[str, Any]]]:
    runtime_rows, evaluator_rows = rows(args.runtime_manifest.resolve()), rows(args.evaluator_manifest.resolve())
    evaluator_by_id = {str(value.get("case_id")): value for value in evaluator_rows}
    if len(evaluator_by_id) != len(evaluator_rows):
        raise ValueError("duplicate evaluator case ID")
    joined = []
    for line, runtime in enumerate(runtime_rows, start=1):
        if set(runtime) != EXPECTED_RUNTIME_KEYS or runtime.get("dataset") != "AIST++" or runtime.get("protocol") != "CS150" or runtime.get("role") != args.role:
            raise ValueError(f"runtime row {line} is not the frozen AIST CS150 {args.role} contract")
        if int(runtime.get("fps", -1)) != 30 or int(runtime.get("num_frames", -1)) != 150:
            raise ValueError(f"runtime row {line} temporal contract drifted")
        case_id = str(runtime["case_id"]); evaluator = evaluator_by_id.get(case_id)
        if evaluator is None or evaluator.get("protocol") != "CS150" or evaluator.get("role") != args.role:
            raise ValueError(f"missing paired evaluator row for {case_id}")
        safe_file(args.derived_root, str(runtime["input_video"])); safe_file(args.derived_root, str(evaluator["label"]))
        joined.append((line, runtime, evaluator))
    if set(evaluator_by_id) != {str(value[1]["case_id"]) for value in joined}:
        raise ValueError("runtime/evaluator case sets differ")
    selected = [value for value in joined if value[0] >= args.start_line]
    return selected[:args.max_cases] if args.max_cases else selected


def run_logged(command: list[str], cwd: Path) -> dict[str, Any]:
    started = time.perf_counter(); completed = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return {"command": command, "returncode": completed.returncode, "seconds": time.perf_counter() - started, "stdout": completed.stdout, "stderr": completed.stderr}


def complete(cache: Path, report: Path, metric: Path, case_id: str) -> bool:
    if not all(path.is_file() for path in (cache, report, metric)):
        return False
    try:
        return json.loads(report.read_text(encoding="utf-8")).get("record", {}).get("case_id") == case_id and json.loads(metric.read_text(encoding="utf-8")).get("case_id") == case_id
    except (OSError, json.JSONDecodeError):
        return False


def execute_case(args: argparse.Namespace, device: int, item: tuple[int, dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    line, runtime, evaluator = item; case_id = str(runtime["case_id"]); name = case_id.replace("/", "_")
    root = args.output_dir.resolve(); official = root / "official" / name; cache = root / "predictions" / f"{name}.npz"
    report, metadata = cache.with_suffix(".runtime.json"), cache.with_suffix(".adapter.json")
    metric, log = root / "metrics" / f"{name}.json", root / "logs" / f"{name}.json"
    if complete(cache, report, metric, case_id):
        return {"line": line, "case_id": case_id, "device": device, "status": "resumed_complete"}
    if any(path.exists() for path in (official, cache, report, metadata, metric)):
        return {"line": line, "case_id": case_id, "device": device, "status": "stale_or_partial_output", "reason": "immutable output exists; inspect rather than overwriting"}
    inference = run_logged([
        str(args.python_executable), str(RUNNER), "--runtime-manifest", str(args.runtime_manifest.resolve()), "--line", str(line),
        "--derived-root", str(args.derived_root.resolve()), "--repo", str(args.repo.resolve()), "--output-dir", str(official),
        "--cache-output", str(cache), "--runtime-report", str(report), "--adapter-metadata", str(metadata),
        "--converter", str(args.converter.resolve()), "--python", str(args.python_executable), "--physical-device", str(device), "--batch-size", str(args.batch_size),
    ], Path.cwd())
    payload: dict[str, Any] = {"line": line, "case_id": case_id, "device": device, "inference": inference}
    if inference["returncode"]:
        atomic_json(log, payload)
        return {"line": line, "case_id": case_id, "device": device, "status": "inference_error", "log": str(log)}
    evaluation = run_logged([
        str(args.python_executable), str(EVALUATOR), "--cache", str(cache), "--runtime-report", str(report),
        "--label", str(safe_file(args.derived_root, str(evaluator["label"]))), "--derived-root", str(args.derived_root.resolve()), "--output", str(metric),
    ], Path.cwd())
    payload["evaluation"] = evaluation; atomic_json(log, payload)
    return {"line": line, "case_id": case_id, "device": device, "status": "ok" if not evaluation["returncode"] else "evaluation_error", "log": str(log)}


def main() -> None:
    args = parse_args()
    devices = tuple(int(value.strip()) for value in args.devices.split(",") if value.strip())
    if not devices or len(devices) > 3 or len(devices) != len(set(devices)) or any(value < 0 for value in devices):
        raise SystemExit("--devices must contain one to three unique nonnegative physical CUDA indices")
    if args.start_line < 1 or args.max_cases < 0 or args.batch_size <= 0:
        raise SystemExit("invalid start-line, max-cases, or batch-size")
    args.python_executable = args.python_executable.expanduser()
    if not args.python_executable.is_absolute():
        args.python_executable = (Path.cwd() / args.python_executable).absolute()
    if not args.python_executable.is_file():
        raise FileNotFoundError(args.python_executable)
    items = validate_pair(args); args.output_dir.mkdir(parents=True, exist_ok=True)
    buckets = [items[index::len(devices)] for index in range(len(devices))]
    outcomes: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [executor.submit(lambda gpu=device, values=bucket: [execute_case(args, gpu, value) for value in values]) for device, bucket in zip(devices, buckets)]
        for future in concurrent.futures.as_completed(futures):
            completed_rows = future.result()
            outcomes.extend(completed_rows)
            for value in completed_rows:
                print(json.dumps(value, ensure_ascii=False), flush=True)
    outcomes.sort(key=lambda value: int(value["line"]))
    payload = {
        "schema_version": SCHEMA, "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "role": args.role, "devices": list(devices),
        "selected_case_count": len(items), "runtime_manifest": str(args.runtime_manifest.resolve()), "runtime_manifest_sha256": sha256_file(args.runtime_manifest.resolve()),
        "evaluator_manifest": str(args.evaluator_manifest.resolve()), "evaluator_manifest_sha256": sha256_file(args.evaluator_manifest.resolve()),
        "results": outcomes, "runtime_evaluator_separation": "Each official GVHMR process receives only one RGB runtime row. Evaluator labels are opened only after external inference and conversion exit.",
    }
    payload["content_sha256"] = canonical_json_digest(payload); atomic_json(args.output_dir / "run_summary.json", payload)
    errors = sum("error" in value["status"] or value["status"] == "stale_or_partial_output" for value in outcomes)
    print(json.dumps({"summary": str(args.output_dir / "run_summary.json"), "selected": len(items), "ok": len(outcomes) - errors, "errors": errors}, ensure_ascii=False, indent=2))
    if errors and not args.continue_on_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
