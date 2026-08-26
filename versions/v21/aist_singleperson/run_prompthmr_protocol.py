#!/usr/bin/env python3
"""Resumable AIST CS150 PromptHMR protocol runner with strict separation.

Only a compact RGB runtime row is passed to the GPU subprocess.  The
coordinator opens the corresponding evaluator row only after the external
method has produced its cache, then invokes the evaluator.  This matches the
separation used by Bridge3R's internal AIST protocol and prevents a baseline
from receiving ground truth, calibration, a camera identity, or a cut index.
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


SCHEMA = "Bridge3R-AIST-PromptHMR-official-protocol-v1"
RUNNER = Path(__file__).with_name("run_prompthmr_case.py")
EVALUATOR = Path(__file__).with_name("evaluate_aist.py")
EXPECTED_RUNTIME_KEYS = {"case_id", "dataset", "protocol", "role", "input_video", "fps", "num_frames"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--evaluator-manifest", type=Path, required=True)
    parser.add_argument("--derived-root", type=Path, default=DEFAULT_DERIVED_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--license-attestation", type=Path, required=True)
    parser.add_argument("--converter", type=Path, required=True)
    parser.add_argument("--python", dest="python_executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--devices", required=True, help="One to three physical CUDA indices.")
    parser.add_argument("--role", choices=("pilot", "test"), required=True)
    parser.add_argument("--start-line", type=int, default=1)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"blank JSONL row at {path}:{line_number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"non-object JSONL row at {path}:{line_number}")
        rows.append(value)
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def resolve_safe(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise ValueError(f"unsafe derived relative path {relative!r}")
    root = root.resolve()
    output = (root / candidate).resolve()
    if root not in output.parents or not output.is_file():
        raise FileNotFoundError(output)
    return output


def validate_pair(args: argparse.Namespace) -> list[tuple[int, dict[str, Any], dict[str, Any]]]:
    runtime_rows, evaluator_rows = read_jsonl(args.runtime_manifest.resolve()), read_jsonl(args.evaluator_manifest.resolve())
    evaluator_by_id = {str(row.get("case_id")): row for row in evaluator_rows}
    if len(evaluator_by_id) != len(evaluator_rows):
        raise ValueError("duplicate evaluator case IDs")
    joined = []
    for line, runtime in enumerate(runtime_rows, start=1):
        if set(runtime) != EXPECTED_RUNTIME_KEYS:
            raise ValueError(f"runtime schema drift at line {line}: {sorted(runtime)}")
        if runtime["dataset"] != "AIST++" or runtime["protocol"] != "CS150" or runtime["role"] != args.role:
            raise ValueError(f"unexpected runtime protocol at line {line}")
        if int(runtime["fps"]) != 30 or int(runtime["num_frames"]) != 150:
            raise ValueError(f"unexpected temporal contract at line {line}")
        case_id = str(runtime["case_id"])
        evaluator = evaluator_by_id.get(case_id)
        if evaluator is None or evaluator.get("protocol") != "CS150" or evaluator.get("role") != args.role:
            raise ValueError(f"missing or incompatible evaluator row for {case_id}")
        resolve_safe(args.derived_root, str(runtime["input_video"]))
        resolve_safe(args.derived_root, str(evaluator["label"]))
        joined.append((line, runtime, evaluator))
    if set(evaluator_by_id) != {str(runtime["case_id"]) for _, runtime, _ in joined}:
        raise ValueError("runtime and evaluator case sets differ")
    selected = [row for row in joined if row[0] >= args.start_line]
    return selected[:args.max_cases] if args.max_cases else selected


def run_logged(command: list[str], *, cwd: Path) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return {
        "command": command, "returncode": completed.returncode,
        "seconds": time.perf_counter() - started,
        "stdout": completed.stdout, "stderr": completed.stderr,
    }


def is_complete(cache: Path, report: Path, metric: Path, case_id: str) -> bool:
    if not all(path.is_file() for path in (cache, report, metric)):
        return False
    try:
        return (
            json.loads(report.read_text(encoding="utf-8")).get("record", {}).get("case_id") == case_id
            and json.loads(metric.read_text(encoding="utf-8")).get("case_id") == case_id
        )
    except (OSError, json.JSONDecodeError):
        return False


def execute_case(args: argparse.Namespace, device: int, item: tuple[int, dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    line, runtime, evaluator = item
    case_id = str(runtime["case_id"])
    safe_name = case_id.replace("/", "_")
    output = args.output_dir.resolve()
    official = output / "official" / safe_name
    cache = output / "predictions" / f"{safe_name}.npz"
    report, metadata = cache.with_suffix(".runtime.json"), cache.with_suffix(".adapter.json")
    metric, log_path = output / "metrics" / f"{safe_name}.json", output / "logs" / f"{safe_name}.json"
    if is_complete(cache, report, metric, case_id):
        return {"line": line, "case_id": case_id, "device": device, "status": "resumed_complete"}
    if official.exists() or cache.exists() or report.exists() or metadata.exists() or metric.exists():
        return {
            "line": line, "case_id": case_id, "device": device, "status": "stale_or_partial_output",
            "reason": "inspect immutable partial output; this runner never overwrites or silently resumes it",
        }
    runner_command = [
        str(args.python_executable), str(RUNNER),
        "--runtime-manifest", str(args.runtime_manifest.resolve()), "--line", str(line),
        "--derived-root", str(args.derived_root.resolve()), "--repo", str(args.repo.resolve()),
        "--license-attestation", str(args.license_attestation.resolve()),
        "--output-dir", str(official), "--cache-output", str(cache),
        "--runtime-report", str(report), "--adapter-metadata", str(metadata),
        "--converter", str(args.converter.resolve()), "--python", str(args.python_executable),
        "--device", f"cuda:{device}", "--batch-size", str(args.batch_size),
    ]
    inference = run_logged(runner_command, cwd=Path.cwd())
    log: dict[str, Any] = {"line": line, "case_id": case_id, "device": device, "inference": inference}
    if inference["returncode"]:
        atomic_json(log_path, log)
        return {"line": line, "case_id": case_id, "device": device, "status": "inference_error", "log": str(log_path)}
    # Evaluator-only file access begins strictly after the GPU process exits.
    evaluation = run_logged([
        str(args.python_executable), str(EVALUATOR), "--cache", str(cache),
        "--runtime-report", str(report),
        "--label", str(resolve_safe(args.derived_root, str(evaluator["label"]))),
        "--derived-root", str(args.derived_root.resolve()), "--output", str(metric),
    ], cwd=Path.cwd())
    log["evaluation"] = evaluation
    atomic_json(log_path, log)
    status = "ok" if not evaluation["returncode"] else "evaluation_error"
    return {"line": line, "case_id": case_id, "device": device, "status": status, "log": str(log_path)}


def main() -> None:
    args = parse_args()
    devices = tuple(int(part.strip()) for part in args.devices.split(",") if part.strip())
    if not devices or len(devices) > 3 or len(set(devices)) != len(devices) or any(device < 0 for device in devices):
        raise SystemExit("--devices must provide one to three distinct nonnegative CUDA indices")
    if args.start_line < 1 or args.max_cases < 0 or args.batch_size <= 0:
        raise SystemExit("invalid --start-line, --max-cases, or --batch-size")
    # Preserve the virtualenv path; resolving a symlink would invoke system
    # Python and lose PromptHMR's installed dependencies.
    args.python_executable = args.python_executable.expanduser()
    if not args.python_executable.is_absolute():
        args.python_executable = (Path.cwd() / args.python_executable).absolute()
    if not args.python_executable.is_file():
        raise FileNotFoundError(args.python_executable)
    items = validate_pair(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    buckets = [items[index::len(devices)] for index in range(len(devices))]
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [executor.submit(lambda gpu=device, rows=bucket: [execute_case(args, gpu, row) for row in rows]) for device, bucket in zip(devices, buckets)]
        for future in concurrent.futures.as_completed(futures):
            rows = future.result()
            results.extend(rows)
            for row in rows:
                print(json.dumps(row, ensure_ascii=False), flush=True)
                if "error" in row["status"] and not args.continue_on_error:
                    raise SystemExit("stopping after first error; inspect it before continuing")
    results.sort(key=lambda row: int(row["line"]))
    payload = {
        "schema_version": SCHEMA, "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "role": args.role, "devices": list(devices), "selected_case_count": len(items),
        "runtime_manifest": str(args.runtime_manifest.resolve()),
        "runtime_manifest_sha256": sha256_file(args.runtime_manifest.resolve()),
        "evaluator_manifest": str(args.evaluator_manifest.resolve()),
        "evaluator_manifest_sha256": sha256_file(args.evaluator_manifest.resolve()),
        "results": results,
        "runtime_evaluator_separation": "Each GPU subprocess receives only a runtime row; evaluator labels are read after native external inference and conversion finish.",
    }
    payload["content_sha256"] = canonical_json_digest(payload)
    summary = args.output_dir / "run_summary.json"
    atomic_json(summary, payload)
    errors = sum("error" in row["status"] for row in results)
    print(json.dumps({"summary": str(summary), "selected": len(items), "ok": len(results) - errors, "errors": errors}, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
