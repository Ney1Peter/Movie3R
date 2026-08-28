#!/usr/bin/env python3
"""Wait for the formal-90 GPU queue and seal its non-GPU result artifacts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
PARENT = REPO_ROOT / "output/bridge3r_egohumans_ablation_v1"
QUEUE_STATE = PARENT / "formal90_ablation_queue_state.json"
FORMAL_MANIFEST = (
    REPO_ROOT.parents[0]
    / "ICLR-paper/bridge3r_iclr2027/private_audit/egohumans_formal90_manifest.jsonl"
)
AGGREGATE = REPO_ROOT / "versions/v19/egohumans/aggregate_formal90_ablation.py"
EXTERNAL = REPO_ROOT / "versions/v19/egohumans/aggregate_external_baselines.py"
EXTERNAL_ROOT = REPO_ROOT.parents[0] / "data/EgoHuman_work_v19/external_predictions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-state", type=Path, default=QUEUE_STATE)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(partial, path)


def main() -> None:
    args = parse_args()
    queue_state = args.queue_state.resolve()
    output = PARENT / "formal90_final"
    state_path = output / "finalization_state.json"
    state: dict[str, Any] = {
        "schema_version": "Bridge3R-EgoHumans-formal90-finalization-v1",
        "status": "waiting_for_gpu_queue",
        "queue_state": str(queue_state),
        "started_at": time.time(),
        "stages": {},
    }
    atomic_json(state_path, state)
    while True:
        if not queue_state.is_file():
            time.sleep(float(args.poll_seconds))
            continue
        queue = json.loads(queue_state.read_text(encoding="utf-8"))
        status = queue.get("status")
        if status == "complete":
            break
        if status == "error":
            state.update(status="blocked_by_gpu_queue", queue_errors=queue.get("errors", []), completed_at=time.time())
            atomic_json(state_path, state)
            raise SystemExit("formal GPU queue ended with errors")
        time.sleep(float(args.poll_seconds))

    stages = [
        (
            "aggregate_internal",
            [
                sys.executable,
                str(AGGREGATE),
                "--parent",
                str(PARENT),
                "--manifest",
                str(FORMAL_MANIFEST),
                "--output",
                str(output / "internal"),
                "--bootstrap",
                str(args.bootstrap),
            ],
        ),
        (
            "aggregate_external",
            [
                sys.executable,
                str(EXTERNAL),
                "--root",
                str(EXTERNAL_ROOT),
                "--selected-case-manifest",
                str(FORMAL_MANIFEST),
                "--expected-cases",
                "90",
                "--output",
                str(output / "external/external_baseline_metrics.csv"),
            ],
        ),
    ]
    for name, command in stages:
        state["status"] = "running"
        state["stages"][name] = {"status": "running", "command": command, "started_at": time.time()}
        atomic_json(state_path, state)
        log = output / f"{name}.log"
        with log.open("w", encoding="utf-8") as handle:
            handle.write("COMMAND " + json.dumps(command, ensure_ascii=False) + "\n")
            handle.flush()
            completed = subprocess.run(command, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT)
        state["stages"][name].update(
            status="complete" if completed.returncode == 0 else "error",
            returncode=completed.returncode,
            completed_at=time.time(),
            log=str(log),
        )
        atomic_json(state_path, state)
        if completed.returncode:
            state.update(status="error", completed_at=time.time())
            atomic_json(state_path, state)
            raise SystemExit(f"{name} failed")
    state.update(status="complete", completed_at=time.time())
    atomic_json(state_path, state)
    print(json.dumps(state, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
