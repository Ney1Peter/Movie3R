#!/usr/bin/env python3
"""Stream OnlineHMR extension cases across GPUs with bounded disk usage."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


SCRIPT = Path(__file__).resolve()
WORKSPACE = SCRIPT.parents[4]
MOVIE_PYTHON = WORKSPACE / "Movie3R/.venv/bin/python"
ONLINE_PYTHON = WORKSPACE / "external_baselines/.venvs/onlinehmr-py311-pt25-cu118/bin/python"
INVOKER = SCRIPT.with_name("invoke_onlinehmr.py")
STAGER = SCRIPT.with_name("stage_extension_case.py")
EVALUATOR = SCRIPT.with_name("evaluate_onlinehmr_extension.py")
PRINT_LOCK = threading.Lock()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(partial, path)


def parse_ints(value: str, minimum: int) -> list[int]:
    values = [int(item) for item in value.split(",") if item.strip()]
    if not values or len(values) != len(set(values)) or any(item < minimum for item in values):
        raise ValueError(value)
    return values


def remove_stage(path: Path, parent: Path) -> int:
    if not path.exists():
        return 0
    resolved, root = path.resolve(), parent.resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"unsafe stage cleanup: {resolved}")
    size = sum(item.stat().st_size for item in resolved.rglob("*") if item.is_file())
    shutil.rmtree(resolved)
    return int(size)


def run_logged(command: list[str], path: Path, *, allow_failure: bool = False) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, cwd=WORKSPACE, capture_output=True, text=True)
    path.write_text(
        "COMMAND " + json.dumps(command) + "\n" + completed.stdout + "\nSTDERR\n" + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode and not allow_failure:
        raise RuntimeError(f"command failed ({completed.returncode}); see {path}")
    return int(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--evaluator-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4")
    parser.add_argument("--lines")
    parser.add_argument("--reserve-gib", type=float, default=8.0)
    parser.add_argument("--timeout-seconds", type=int, default=5400)
    parser.add_argument("--mvhuman-audit-root", type=Path)
    args = parser.parse_args()
    runtime_manifest = args.runtime_manifest.resolve()
    evaluator_manifest = args.evaluator_manifest.resolve()
    rows = read_jsonl(runtime_manifest)
    evaluator_rows = read_jsonl(evaluator_manifest)
    if [row["case_id"] for row in rows] != [row["case_id"] for row in evaluator_rows]:
        raise ValueError("runtime/evaluator manifest case order mismatch")
    lines = parse_ints(args.lines, 1) if args.lines else list(range(1, len(rows) + 1))
    if any(line > len(rows) for line in lines):
        raise IndexError("selected line outside manifest")
    gpus = parse_ints(args.gpus, 0)
    work_root = args.work_root.resolve()
    input_root = work_root / "runtime_inputs"
    run_root = args.run_root.resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)
    state_path = run_root / "protocol_state.json"
    state: dict[str, Any] = {
        "schema_version": "Bridge3R-OnlineHMR-extension-stream-v1",
        "status": "running",
        "runtime_manifest": str(runtime_manifest),
        "evaluator_manifest": str(evaluator_manifest),
        "source_root": str(args.source_root.resolve()),
        "selected_lines": lines,
        "gpus": gpus,
        "fixed_denominator": len(lines),
        "runtime_gt_access": False,
        "cases": {},
    }
    if state_path.is_file():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        if previous.get("runtime_manifest") != str(runtime_manifest) or previous.get("selected_lines") != lines:
            raise RuntimeError("existing protocol state belongs to a different frozen run")
        state["cases"] = previous.get("cases", {})
    atomic_json(state_path, state)

    assignments = {gpu: lines[index::len(gpus)] for index, gpu in enumerate(gpus)}

    def update(line: int, value: dict[str, Any]) -> None:
        with PRINT_LOCK:
            state["cases"][str(line)] = value
            atomic_json(state_path, state)
            done = sum(item.get("status") == "complete" for item in state["cases"].values())
            print(f"[extension {done}/{len(lines)}] line={line:03d} {value.get('status')}", flush=True)

    def worker(gpu: int, assigned: list[int]) -> list[dict[str, Any]]:
        output = []
        for line in assigned:
            row = rows[line - 1]
            case_id = str(row["case_id"])
            case_root = run_root / f"line{line:03d}"
            evaluation = case_root / "onlinehmr.evaluation.json"
            runtime = case_root / "onlinehmr.runtime.json"
            if evaluation.is_file() and runtime.is_file():
                result = {"line": line, "case_id": case_id, "gpu": gpu, "status": "complete", "resumed": True}
                update(line, result)
                output.append(result)
                continue
            usage = shutil.disk_usage(work_root.parent)
            if usage.free < int(args.reserve_gib * 1024 ** 3):
                result = {"line": line, "case_id": case_id, "gpu": gpu, "status": "disk_blocked", "free_bytes": usage.free}
                update(line, result)
                output.append(result)
                break
            started = time.time()
            stage_case = input_root / case_id
            try:
                update(line, {"line": line, "case_id": case_id, "gpu": gpu, "status": "staging"})
                run_logged([
                    str(MOVIE_PYTHON), str(STAGER),
                    "--runtime-manifest", str(runtime_manifest),
                    "--evaluator-manifest", str(evaluator_manifest),
                    "--line", str(line), "--source-root", str(args.source_root.resolve()),
                    "--output-root", str(input_root),
                ], case_root / "stage.log")
                update(line, {"line": line, "case_id": case_id, "gpu": gpu, "status": "inference"})
                inference_returncode = run_logged([
                    str(MOVIE_PYTHON), str(INVOKER),
                    "--manifest", str(runtime_manifest), "--line", str(line),
                    "--image-dir", str(stage_case / "images"),
                    "--output-root", str(run_root), "--gpu", str(gpu),
                    "--timeout-seconds", str(args.timeout_seconds),
                ], case_root / "invoke.log", allow_failure=True)
                if not runtime.is_file():
                    raise FileNotFoundError(f"invoker did not retain a runtime record: {runtime}")
                update(line, {"line": line, "case_id": case_id, "gpu": gpu, "status": "evaluation", "inference_returncode": inference_returncode})
                command = [
                    str(MOVIE_PYTHON), str(EVALUATOR),
                    "--runtime-manifest", str(runtime_manifest),
                    "--evaluator-manifest", str(evaluator_manifest),
                    "--line", str(line), "--run-root", str(run_root),
                    "--source-root", str(args.source_root.resolve()),
                    "--adapter-python", str(ONLINE_PYTHON),
                    "--evaluator-python", str(MOVIE_PYTHON),
                ]
                if args.mvhuman_audit_root is not None:
                    command += ["--mvhuman-audit-root", str(args.mvhuman_audit_root.resolve())]
                run_logged(command, case_root / "evaluate.log")
                if not evaluation.is_file():
                    raise FileNotFoundError(evaluation)
                result = {
                    "line": line, "case_id": case_id, "gpu": gpu,
                    "status": "complete", "inference_returncode": inference_returncode,
                    "wall_time_seconds": time.time() - started,
                    "stage_bytes_removed": remove_stage(stage_case, input_root),
                }
            except Exception as error:
                result = {
                    "line": line, "case_id": case_id, "gpu": gpu,
                    "status": "orchestration_error",
                    "error": f"{type(error).__name__}: {error}",
                    "wall_time_seconds": time.time() - started,
                }
                try:
                    result["stage_bytes_removed"] = remove_stage(stage_case, input_root)
                except Exception as cleanup_error:
                    result["cleanup_error"] = f"{type(cleanup_error).__name__}: {cleanup_error}"
            update(line, result)
            output.append(result)
        return output

    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(worker, gpu, assigned) for gpu, assigned in assignments.items() if assigned]
        for future in concurrent.futures.as_completed(futures):
            results.extend(future.result())
    results.sort(key=lambda item: int(item["line"]))
    complete = sum(item["status"] == "complete" for item in results)
    state["status"] = "complete" if complete == len(lines) else "partial"
    state["completed_cases"] = complete
    state["orchestration_error_cases"] = sum(item["status"] == "orchestration_error" for item in results)
    state["disk_blocked_workers"] = sum(item["status"] == "disk_blocked" for item in results)
    atomic_json(state_path, state)
    print(json.dumps(state, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
