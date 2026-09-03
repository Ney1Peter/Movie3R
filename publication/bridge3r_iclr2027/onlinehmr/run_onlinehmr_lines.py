#!/usr/bin/env python3
"""Run frozen OnlineHMR inference for selected manifest lines in parallel."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


SCRIPT = Path(__file__).resolve()
WORKSPACE = SCRIPT.parents[4]
INVOKER = SCRIPT.with_name("invoke_onlinehmr.py")
DEFAULT_PYTHON = WORKSPACE / "Movie3R/.venv/bin/python"
SCHEMA = "Bridge3R-OnlineHMR-parallel-runtime-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def parse_values(value: str, *, minimum: int = 1) -> list[int]:
    output = [int(item) for item in value.split(",") if item.strip()]
    if not output or len(output) != len(set(output)) or any(item < minimum for item in output):
        raise ValueError(f"expected unique comma-separated integers >= {minimum}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lines", required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4")
    parser.add_argument(
        "--python", type=Path, default=DEFAULT_PYTHON,
        help="Python used only to launch the lightweight invoker; the invoker uses its frozen OnlineHMR environment",
    )
    parser.add_argument("--timeout-seconds", type=int, default=5400)
    parser.add_argument("--allow-failures", action="store_true")
    args = parser.parse_args()

    manifest = args.manifest.resolve()
    lines = parse_values(args.lines)
    gpus = parse_values(args.gpus, minimum=0)
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(line > len(rows) for line in lines):
        raise IndexError("selected line is outside manifest")
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    log_root = output_root / "invoke_logs"
    log_root.mkdir(parents=True, exist_ok=True)

    assignments = {gpu: lines[index::len(gpus)] for index, gpu in enumerate(gpus)}

    def worker(gpu: int, assigned: list[int]) -> list[dict[str, Any]]:
        results = []
        for line in assigned:
            row = rows[line - 1]
            case_id = str(row["case_id"])
            image_dir = input_root / case_id / "images"
            command = [
                str(args.python.resolve()), str(INVOKER),
                "--manifest", str(manifest), "--line", str(line),
                "--image-dir", str(image_dir), "--output-root", str(output_root),
                "--gpu", str(gpu), "--timeout-seconds", str(args.timeout_seconds),
            ]
            completed = subprocess.run(
                command, cwd=WORKSPACE, capture_output=True, text=True
            )
            log = log_root / f"line{line:03d}.invoke.log"
            log.write_text(
                "COMMAND " + json.dumps(command) + "\n" + completed.stdout
                + "\nSTDERR\n" + completed.stderr,
                encoding="utf-8",
            )
            runtime = output_root / f"line{line:03d}/onlinehmr.runtime.json"
            status = None
            if runtime.is_file():
                status = json.loads(runtime.read_text(encoding="utf-8")).get("status")
            results.append({
                "line": line, "case_id": case_id, "gpu": gpu,
                "returncode": completed.returncode, "status": status,
                "runtime": str(runtime), "log": str(log),
            })
            if completed.returncode and not args.allow_failures:
                raise RuntimeError(f"OnlineHMR line {line} failed; see {log}")
        return results

    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [
            pool.submit(worker, gpu, assigned)
            for gpu, assigned in assignments.items() if assigned
        ]
        for future in concurrent.futures.as_completed(futures):
            results.extend(future.result())
    results.sort(key=lambda item: int(item["line"]))
    payload = {
        "schema_version": SCHEMA,
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "selected_lines": lines,
        "gpus": gpus,
        "attempted_cases": len(results),
        "successful_cases": sum(item["status"] == "success" for item in results),
        "failed_cases": sum(item["status"] != "success" for item in results),
        "runtime_gt_access": False,
        "cases": results,
    }
    name = "summary.lines_" + "-".join(f"{line:03d}" for line in lines) + ".json"
    atomic_json(output_root / name, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if payload["failed_cases"] and not args.allow_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
