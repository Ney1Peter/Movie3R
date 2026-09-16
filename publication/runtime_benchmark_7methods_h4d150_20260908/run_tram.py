#!/usr/bin/env python3
"""Launch process-level official TRAM timing over the frozen five cases."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[2]
TRAM = WORKSPACE / "external_baselines/TRAM"
PYTHON = WORKSPACE / "external_baselines/.venvs/onlinehmr-py311-pt25-cu118/bin/python"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=3)
    parser.add_argument("--selection", type=Path, default=ROOT / "selected_cases.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs/tram")
    parser.add_argument("--only-line", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selection = json.loads(args.selection.resolve().read_text(encoding="utf-8"))
    hardware = subprocess.run(
        [
            "nvidia-smi", f"--id={args.device}",
            "--query-gpu=index,name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ], text=True, capture_output=True, check=True,
    ).stdout.strip()
    for case in selection["cases"]:
        if args.only_line is not None and int(case["line"]) != args.only_line:
            continue
        case_id = str(case["case_id"])
        case_root = args.output_root.resolve() / case_id
        case_root.mkdir(parents=True, exist_ok=True)
        report_path = case_root / "runtime.json"
        if report_path.is_file():
            print(f"SKIP completed TRAM {case_id}", flush=True)
            continue
        native = case_root / "native"
        command = [
            str(PYTHON), str(ROOT / "run_tram_case.py"),
            "--case-id", case_id,
            "--input-dir", str(WORKSPACE / case["input_dir"]),
            "--output-dir", str(native),
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(args.device)
        environment["CUDA_HOME"] = "/usr/local/cuda-11.8"
        environment["TORCH_HOME"] = str(TRAM / "data/cache/torch")
        environment["MIDAS_HUB_DIR"] = str(
            TRAM / "data/cache/torch/hub/intel-isl_MiDaS_master"
        )
        log_path = case_root / "process.log"
        started = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=TRAM,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        wall = time.perf_counter() - started
        internal = native / "tram.internal.json"
        report = {
            "schema_version": "Shot3R-H4D-CS150-TRAM-process-runtime-v1",
            "method": "tram",
            "case_id": case_id,
            "frames": 150,
            "wall_time_seconds": wall,
            "returncode": completed.returncode,
            "status": "success" if completed.returncode == 0 and internal.is_file() else "failed",
            "command": command,
            "CUDA_VISIBLE_DEVICES": str(args.device),
            "hardware": hardware,
            "host": platform.node(),
            "timing_scope": "process launch through official camera/track and VIMO prediction save; rendering excluded",
            "runtime_gt_access": False,
            "log": str(log_path),
            "native_output": str(native),
            "internal_report": str(internal),
        }
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report), flush=True)
        if report["status"] != "success":
            raise RuntimeError(f"TRAM failed for {case_id}; see {log_path}")


if __name__ == "__main__":
    main()
