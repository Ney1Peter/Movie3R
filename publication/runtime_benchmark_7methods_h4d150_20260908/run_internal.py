#!/usr/bin/env python3
"""Launch process-level Human3R/Shot3R timing over the frozen five cases."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
WORKSPACE = THIS_DIR.parents[2]
MOVIE = WORKSPACE / "Movie3R"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("human3r", "shot3r", "both"), default="both")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--selection", type=Path, default=THIS_DIR / "selected_cases.json")
    parser.add_argument("--output-root", type=Path, default=THIS_DIR / "runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selection = json.loads(args.selection.resolve().read_text(encoding="utf-8"))
    methods = ("human3r", "shot3r") if args.method == "both" else (args.method,)
    gpu_index = int(str(args.device).split(":", 1)[1])
    hardware = subprocess.run(
        [
            "nvidia-smi", f"--id={gpu_index}",
            "--query-gpu=index,name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ], text=True, capture_output=True, check=True,
    ).stdout.strip()
    for method in methods:
        for case in selection["cases"]:
            case_id = str(case["case_id"])
            case_root = args.output_root.resolve() / method / case_id
            case_root.mkdir(parents=True, exist_ok=True)
            report_path = case_root / "runtime.json"
            if report_path.is_file():
                print(f"SKIP completed {method} {case_id}", flush=True)
                continue
            output = case_root / "prediction.npz"
            log_path = case_root / "process.log"
            command = [
                str(MOVIE / ".venv/bin/python"),
                str(THIS_DIR / "run_human3r_shot3r_case.py"),
                "--method", method,
                "--case-id", case_id,
                "--input-dir", str(WORKSPACE / case["input_dir"]),
                "--output", str(output),
                "--device", args.device,
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
            # The child sees only one device after masking.
            command[-1] = "cuda:0"
            started = time.perf_counter()
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    cwd=MOVIE,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            wall = time.perf_counter() - started
            report = {
                "schema_version": "Shot3R-H4D-CS150-process-runtime-v1",
                "method": method,
                "case_id": case_id,
                "frames": 150,
                "wall_time_seconds": wall,
                "returncode": completed.returncode,
                "status": "success" if completed.returncode == 0 and output.is_file() else "failed",
                "command": command,
                "CUDA_VISIBLE_DEVICES": str(gpu_index),
                "hardware": hardware,
                "host": platform.node(),
                "timing_scope": "process launch through native prediction save",
                "runtime_gt_access": False,
                "log": str(log_path),
                "native_output": str(output),
            }
            report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(report), flush=True)
            if report["status"] != "success":
                raise RuntimeError(f"{method} failed for {case_id}; see {log_path}")


if __name__ == "__main__":
    main()

