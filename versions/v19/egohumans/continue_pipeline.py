#!/usr/bin/env python3
"""Resume the frozen EgoHumans pipeline after the base development run.

This driver is intentionally small: every expensive stage remains owned by
the resumable split runner, while this file records and launches the fixed
development -> holdout -> test sequence from the preregistered protocol.
"""

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
RUNNER = REPO_ROOT / "versions/v19/egohumans/run_protocol.py"
AGGREGATE = REPO_ROOT / "versions/v19/egohumans/aggregate_results.py"
FREEZE_HOLDOUT = REPO_ROOT / "versions/v19/egohumans/freeze_holdout_candidates.py"
FREEZE_FINAL = REPO_ROOT / "versions/v19/egohumans/freeze_final_candidate.py"
COMBINED = REPO_ROOT / "versions/v19/egohumans/development_combined_candidates.json"
OUTPUT = REPO_ROOT / "output/v19_egohumans"
HOLDOUT_CANDIDATES = REPO_ROOT / "versions/v19/egohumans/frozen_holdout_candidates.json"
FINAL_CANDIDATE = REPO_ROOT / "versions/v19/egohumans/frozen_final_candidate.json"
WORK = Path("/data/wangzheng/iJCV-CODE/data/EgoHuman_work_v19")
STATE = OUTPUT / "pipeline_state.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument("--devices", default="cuda:0,cuda:2,cuda:3,cuda:5")
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(partial, path)


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def development_base_complete() -> bool:
    state_path = OUTPUT / "development/protocol_state.json"
    if not state_path.is_file():
        return False
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("status") != "complete" or int(state.get("complete_captures", 0)) != 7:
        return False
    for item in state.get("captures", {}).values():
        state_file = Path(item["state"])
        capture = json.loads(state_file.read_text(encoding="utf-8"))
        # A later combined-candidate pass legitimately rewrites state.json,
        # while the immutable base reports remain beside it.
        if capture.get("status") != "complete" or not (state_file.parent / "exploration_grid.json").is_file():
            return False
        if not list(state_file.parent.glob("candidate_frozen_multicue_candidate_*.json")):
            return False
    return True


def run_stage(name: str, command: list[str], state: dict[str, Any]) -> None:
    stages = state.setdefault("stages", {})
    stages[name] = {
        "status": "running",
        "started_at": time.time(),
        "command": command,
    }
    state["status"] = "running"
    state["current_stage"] = name
    atomic_json(STATE, state)
    print(f"\n===== {name} =====", flush=True)
    print(json.dumps(command, ensure_ascii=False), flush=True)
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env={**os.environ, "TMPDIR": str(WORK / "tmp")},
    )
    stages[name].update(
        status="complete" if completed.returncode == 0 else "error",
        returncode=completed.returncode,
        completed_at=time.time(),
    )
    atomic_json(STATE, state)
    if completed.returncode:
        state["status"] = "error"
        atomic_json(STATE, state)
        raise SystemExit(f"Stage {name} failed with return code {completed.returncode}")


def main() -> None:
    args = parse_args()
    python = sys.executable
    state: dict[str, Any] = {
        "schema_version": "Movie3R-v19-EgoHumans-pipeline-state-v1",
        "protocol": "Movie3R-EgoHumans-CS100-v1",
        "status": "waiting_for_base_development",
        "started_at": time.time(),
        "wait_pid": args.wait_pid,
        "stages": {},
    }
    if STATE.is_file():
        previous = json.loads(STATE.read_text(encoding="utf-8"))
        state["started_at"] = previous.get("started_at", state["started_at"])
        state["stages"] = previous.get("stages", {})
    atomic_json(STATE, state)
    if args.wait_pid is not None:
        print(f"Waiting for base development PID {args.wait_pid}", flush=True)
        while process_exists(args.wait_pid):
            time.sleep(float(args.poll_seconds))
    if not development_base_complete():
        state.update(status="error", current_stage="validate_base_development")
        atomic_json(STATE, state)
        raise SystemExit("Base development did not finish with 7 complete captures and both reports")

    dev_summary = OUTPUT / "development/summary"
    holdout_summary = OUTPUT / "holdout/summary"
    test_summary = OUTPUT / "test/summary"
    stages = [
        (
            "development_combined_candidates",
            [
                python,
                str(RUNNER),
                "--split", "development",
                "--devices", args.devices,
                "--candidate-json", str(COMBINED),
                "--continue-on-structural-error",
            ],
        ),
        (
            "aggregate_development",
            [
                python,
                str(AGGREGATE),
                "--inputs", str(OUTPUT / "development/captures"),
                "--output", str(dev_summary),
                "--split", "development",
            ],
        ),
        (
            "freeze_holdout_candidates",
            [
                python,
                str(FREEZE_HOLDOUT),
                "--development-summary", str(dev_summary / "summary.json"),
                "--output", str(HOLDOUT_CANDIDATES),
            ],
        ),
        (
            "run_holdout",
            [
                python,
                str(RUNNER),
                "--split", "holdout",
                "--devices", args.devices,
                "--candidate-json", str(HOLDOUT_CANDIDATES),
                "--continue-on-structural-error",
            ],
        ),
        (
            "aggregate_holdout",
            [
                python,
                str(AGGREGATE),
                "--inputs", str(OUTPUT / "holdout/captures"),
                "--output", str(holdout_summary),
                "--split", "holdout",
            ],
        ),
        (
            "freeze_final_candidate",
            [
                python,
                str(FREEZE_FINAL),
                "--development-summary", str(dev_summary / "summary.json"),
                "--holdout-summary", str(holdout_summary / "summary.json"),
                "--holdout-candidates", str(HOLDOUT_CANDIDATES),
                "--output", str(FINAL_CANDIDATE),
            ],
        ),
        (
            "run_test",
            [
                python,
                str(RUNNER),
                "--split", "test",
                "--devices", args.devices,
                "--candidate-json", str(FINAL_CANDIDATE),
                "--continue-on-structural-error",
            ],
        ),
        (
            "aggregate_test",
            [
                python,
                str(AGGREGATE),
                "--inputs", str(OUTPUT / "test/captures"),
                "--output", str(test_summary),
                "--split", "test",
            ],
        ),
    ]
    for name, command in stages:
        if state.get("stages", {}).get(name, {}).get("status") == "complete":
            print(f"Skipping completed stage: {name}", flush=True)
            continue
        run_stage(name, command, state)
    state.update(status="complete", current_stage=None, completed_at=time.time())
    atomic_json(STATE, state)
    print(json.dumps(state, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
