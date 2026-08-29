#!/usr/bin/env python3
"""Resumable official PromptHMR/GVHMR execution for frozen MVH150 rows."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
WORKSPACE_ROOT = HERE.parents[3]
PROMPT_RUNNER = HERE / "run_prompthmr_case.py"
GVHMR_RUNNER = HERE / "run_gvhmr_case.py"
EVALUATOR = HERE / "evaluate_case.py"
PROMPT_CONVERTER = HERE / "convert_prompthmr_result.py"
GVHMR_CONVERTER = HERE / "convert_gvhmr_result.py"
DEFAULT_PROMPT_PYTHON = WORKSPACE_ROOT / "external_baselines/.venvs/prompthmr-py311-pt24/bin/python"
DEFAULT_GVHMR_PYTHON = WORKSPACE_ROOT / "external_baselines/.venvs/gvhmr-py310-pt23/bin/python"
DEFAULT_PROMPT_REPO = WORKSPACE_ROOT / "external_baselines/PromptHMR"
DEFAULT_GVHMR_REPO = WORKSPACE_ROOT / "external_baselines/GVHMR"
DEFAULT_LICENSE_ATTESTATION = WORKSPACE_ROOT / "data/prompthmr_spec_license_attestation.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("prompthmr", "gvhmr"), required=True)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--prompthmr-python", type=Path, default=DEFAULT_PROMPT_PYTHON)
    parser.add_argument("--gvhmr-python", type=Path, default=DEFAULT_GVHMR_PYTHON)
    parser.add_argument("--prompthmr-repo", type=Path, default=DEFAULT_PROMPT_REPO)
    parser.add_argument("--gvhmr-repo", type=Path, default=DEFAULT_GVHMR_REPO)
    parser.add_argument("--license-attestation", type=Path, default=DEFAULT_LICENSE_ATTESTATION)
    parser.add_argument("--pilot", action="store_true", help="Run the predeclared 12-case tracker/availability pilot.")
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def pilot_lines(total: int) -> list[int]:
    declared = [1, 5, 6, 10, 21, 25, 26, 30, 31, 35, 46, 50]
    if total < max(declared):
        raise ValueError("Frozen 12-case pilot indices exceed runtime manifest")
    return declared


def infer_command(args: argparse.Namespace, line: int, case_id: str, gpu: int, cache: Path, report: Path, adapter: Path, official_dir: Path, runtime: Path) -> list[str]:
    if args.method == "prompthmr":
        return [
            str(args.prompthmr_python), str(PROMPT_RUNNER), "--runtime-manifest", str(runtime), "--line", str(line),
            "--derived-root", str(args.protocol_root / "derived"),
            "--repo", str(args.prompthmr_repo),
            "--license-attestation", str(args.license_attestation),
            "--output-dir", str(official_dir), "--cache-output", str(cache), "--runtime-report", str(report),
            "--adapter-metadata", str(adapter),
            "--converter", str(PROMPT_CONVERTER),
            # Each subprocess is restricted to one physical GPU below.  The
            # unchanged PromptHMR pipeline and its bundled camera components
            # therefore consistently address that device as logical cuda:0.
            "--python", str(args.prompthmr_python), "--device", "cuda:0", "--batch-size", "32",
        ]
    return [
        str(args.gvhmr_python), str(GVHMR_RUNNER), "--runtime-manifest", str(runtime), "--line", str(line),
        "--derived-root", str(args.protocol_root / "derived"),
        "--repo", str(args.gvhmr_repo),
        "--output-dir", str(official_dir), "--cache-output", str(cache), "--runtime-report", str(report),
        "--adapter-metadata", str(adapter),
        "--converter", str(GVHMR_CONVERTER),
        "--python", str(args.gvhmr_python), "--physical-device", str(gpu), "--batch-size", "64",
    ]


def run_one(args: argparse.Namespace, runtime: Path, evaluator: Path, line: int, record: dict[str, Any], gpu: int) -> dict[str, Any]:
    case_id = str(record["case_id"])
    cache = args.output_root / "predictions" / f"{case_id}.npz"
    report = cache.with_suffix(".runtime.json")
    adapter = cache.with_suffix(".adapter.json")
    metric = args.output_root / "metrics" / f"{case_id}.json"
    official_dir = args.output_root / "official" / case_id
    log = args.output_root / "logs" / f"{case_id}.log"
    for parent in (cache.parent, metric.parent, log.parent, official_dir.parent):
        parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    commands = []
    if not (cache.is_file() and report.is_file() and adapter.is_file()):
        commands.append(infer_command(args, line, case_id, gpu, cache, report, adapter, official_dir, runtime))
    if not metric.is_file():
        python_executable = args.prompthmr_python if args.method == "prompthmr" else args.gvhmr_python
        commands.append(
            [
                str(python_executable), str(EVALUATOR), "--cache", str(cache), "--runtime-report", str(report),
                "--evaluator-manifest", str(evaluator), "--case-id", case_id,
                "--audit-root", str(args.audit_root), "--output", str(metric),
            ]
        )
    with log.open("a", encoding="utf-8") as handle:
        for command in commands:
            handle.write(f"COMMAND {json.dumps(command)}\n"); handle.flush()
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            completed = subprocess.run(
                command,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                env=environment,
            )
            if completed.returncode:
                return {"case_id": case_id, "line": line, "gpu": gpu, "status": "failed", "returncode": completed.returncode, "seconds": time.time() - started, "log": str(log)}
    return {"case_id": case_id, "line": line, "gpu": gpu, "status": "ok", "resumed": not commands, "seconds": time.time() - started}


def worker(args: argparse.Namespace, runtime: Path, evaluator: Path, gpu: int, tasks: list[tuple[int, dict[str, Any]]]) -> list[dict[str, Any]]:
    outcomes = []
    for line, record in tasks:
        result = run_one(args, runtime, evaluator, line, record, gpu)
        outcomes.append(result); print(json.dumps(result), flush=True)
    return outcomes


def main() -> None:
    args = parse_args()
    args.protocol_root = args.protocol_root.resolve(); args.audit_root = args.audit_root.resolve(); args.output_root = args.output_root.resolve()
    args.prompthmr_python = args.prompthmr_python.resolve()
    args.gvhmr_python = args.gvhmr_python.resolve()
    args.prompthmr_repo = args.prompthmr_repo.resolve()
    args.gvhmr_repo = args.gvhmr_repo.resolve()
    args.license_attestation = args.license_attestation.resolve()
    runtime = args.protocol_root / "manifests" / "test_runtime.jsonl"
    evaluator = args.protocol_root / "manifests" / "test_evaluator.jsonl"
    rows = [json.loads(line) for line in runtime.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = pilot_lines(len(rows)) if args.pilot else list(range(1, len(rows) + 1))
    tasks = [(line, rows[line - 1]) for line in selected]
    shards = {gpu: tasks[index::len(args.gpus)] for index, gpu in enumerate(args.gpus)}
    started = time.time(); outcomes = []
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = [executor.submit(worker, args, runtime, evaluator, gpu, shard) for gpu, shard in shards.items()]
        for future in as_completed(futures):
            outcomes.extend(future.result())
    payload = {
        "schema_version": "Bridge3R-MVHuman-external-run-summary-v1", "method": args.method,
        "pilot": args.pilot, "predeclared_lines": selected, "gpus": args.gpus,
        "case_count": len(tasks), "ok": sum(row["status"] == "ok" for row in outcomes),
        "failed": [row for row in outcomes if row["status"] != "ok"], "seconds": time.time() - started,
        "outcomes": sorted(outcomes, key=lambda row: row["line"]),
    }
    atomic_json(args.output_root / "run_summary.json", payload)
    print(json.dumps({key: payload[key] for key in ("method", "pilot", "case_count", "ok", "failed", "seconds")}, indent=2))
    if payload["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
