#!/usr/bin/env python3
"""Run and evaluate the alignment-plus-continued-state counterfactual.

The protocol is resumable and writes only to its dedicated output directory.
Each CUDA device owns a deterministic subset of manifest lines.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "versions/v20/egobody/run_alignment_continued_case.py"
EVALUATOR = REPO_ROOT / "versions/v20/egobody/evaluate_egobody.py"
METHOD = "m3_alignment_continued"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--staged-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--current-checkpoint", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def valid_evaluation(path: Path, case_id: str) -> bool:
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return (
        payload.get("case_id") == case_id
        and METHOD in payload.get("methods", {})
        and not payload.get("errors")
    )


def main() -> None:
    args = parse_args()
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    if not devices or len(devices) != len(set(devices)):
        raise ValueError("--devices must contain unique CUDA indices")
    rows = [
        json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("empty manifest")
    case_ids = [str(row["case_id"]) for row in rows]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("duplicate case IDs")

    output = args.output_dir.resolve()
    predictions = output / "predictions"
    evaluations = output / "evaluations"
    logs = output / "logs"
    for path in (predictions, evaluations, logs):
        path.mkdir(parents=True, exist_ok=True)
    state_path = output / "protocol_state.json"
    state: dict[str, Any] = {
        "schema_version": "Shot3R-EgoBody-alignment-continued-protocol-v1",
        "status": "running",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest.resolve()),
        "staged_root": str(args.staged_root.resolve()),
        "gt_root": str(args.gt_root.resolve()),
        "devices": devices,
        "method": METHOD,
        "case_count": len(rows),
        "cases": {},
    }
    lock = threading.Lock()
    completed = 0
    failed = 0

    def update(case_id: str, payload: dict[str, Any]) -> None:
        nonlocal completed, failed
        with lock:
            state["cases"][case_id] = payload
            completed = sum(
                value.get("status") in {"complete", "reused"}
                for value in state["cases"].values()
            )
            failed = sum(
                value.get("status") == "failed" for value in state["cases"].values()
            )
            state["complete_count"] = completed
            state["failed_count"] = failed
            atomic_json(state_path, state)
            print(
                f">> progress {completed}/{len(rows)} complete, {failed} failed; "
                f"latest={case_id} ({payload['status']})",
                flush=True,
            )

    def process(device: str, indexed_rows: list[tuple[int, dict[str, Any]]]) -> None:
        for line_number, record in indexed_rows:
            case_id = str(record["case_id"])
            cache = predictions / f"{case_id}.npz"
            runtime = predictions / f"{case_id}.runtime.json"
            evaluation = evaluations / f"{case_id}.evaluation.json"
            log_path = logs / f"{case_id}.log"
            if cache.is_file() and runtime.is_file() and valid_evaluation(evaluation, case_id):
                update(case_id, {
                    "status": "reused",
                    "device": device,
                    "manifest_line": line_number,
                    "cache": str(cache),
                    "runtime": str(runtime),
                    "evaluation": str(evaluation),
                })
                continue
            if cache.exists() != runtime.exists():
                update(case_id, {
                    "status": "failed",
                    "device": device,
                    "manifest_line": line_number,
                    "error": "incomplete existing prediction pair",
                })
                continue
            runner = [
                sys.executable, str(RUNNER),
                "--manifest", str(args.manifest.resolve()),
                "--line", str(line_number),
                "--staged-root", str(args.staged_root.resolve()),
                "--output", str(cache),
                "--device", f"cuda:{device}",
                "--size", str(args.size),
            ]
            if args.current_checkpoint is not None:
                runner.extend(["--current-checkpoint", str(args.current_checkpoint.resolve())])
            evaluator = [
                sys.executable, str(EVALUATOR),
                "--cache", str(cache),
                "--runtime-report", str(runtime),
                "--gt-root", str(args.gt_root.resolve()),
                "--output", str(evaluation),
            ]
            case_started = time.perf_counter()
            try:
                with log_path.open("a", encoding="utf-8") as log:
                    if not cache.is_file():
                        log.write("\n=== inference ===\n")
                        log.flush()
                        subprocess.run(
                            runner, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
                            text=True, check=True,
                        )
                    log.write("\n=== evaluation ===\n")
                    log.flush()
                    subprocess.run(
                        evaluator, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
                        text=True, check=True,
                    )
                if not valid_evaluation(evaluation, case_id):
                    raise ValueError("evaluation failed structural validation")
                update(case_id, {
                    "status": "complete",
                    "device": device,
                    "manifest_line": line_number,
                    "seconds": time.perf_counter() - case_started,
                    "cache": str(cache),
                    "cache_sha256": sha256(cache),
                    "runtime": str(runtime),
                    "evaluation": str(evaluation),
                    "evaluation_sha256": sha256(evaluation),
                    "log": str(log_path),
                })
            except Exception as error:
                update(case_id, {
                    "status": "failed",
                    "device": device,
                    "manifest_line": line_number,
                    "seconds": time.perf_counter() - case_started,
                    "error": f"{type(error).__name__}: {error}",
                    "log": str(log_path),
                })

    partitions: list[list[tuple[int, dict[str, Any]]]] = [[] for _ in devices]
    for index, record in enumerate(rows):
        partitions[index % len(devices)].append((index + 1, record))
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [
            executor.submit(process, device, partition)
            for device, partition in zip(devices, partitions)
        ]
        for future in futures:
            future.result()

    state["status"] = "complete" if completed == len(rows) and failed == 0 else "failed"
    state["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    atomic_json(state_path, state)
    print(json.dumps({
        "status": state["status"],
        "complete": completed,
        "failed": failed,
        "state": str(state_path),
    }, indent=2), flush=True)
    if state["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
