#!/usr/bin/env python3
"""Keep the sealed EgoHumans component-ablation queue running on bounded GPU groups.

Each route has an independent, disk-bounded staging root and output root.  The
queue can assign one GPU per route (legacy comma-separated syntax) or a group
of GPUs per route (semicolon-separated groups).  The latter lets the protocol
runner execute the independent angle-stratum cases of one capture in parallel.
All routes receive the same immutable formal case manifest; a route failure is
recorded but never silently retried with a different case set or checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
INFERENCE_PYTHON = REPO_ROOT / ".venv/bin/python"
RUNNER = REPO_ROOT / "versions/v19/egohumans/run_protocol.py"
FORMAL_MANIFEST = (
    REPO_ROOT.parents[0]
    / "ICLR-paper/bridge3r_iclr2027/private_audit/egohumans_formal90_manifest.jsonl"
)
OUTER = REPO_ROOT.parents[0] / "data/EgoHuman.zip"
OUTER_INDEX = REPO_ROOT.parents[0] / "data/EgoHuman_work_v19/outer_index.json"
CANDIDATE = REPO_ROOT / "versions/v19/egohumans/frozen_final_candidate.json"
OUTPUT_PARENT = REPO_ROOT / "output/bridge3r_egohumans_ablation_v1"
WORK_PARENT = REPO_ROOT.parents[0] / "data"
SEALED_FULL_PREDICTIONS = REPO_ROOT / "output/v19_egohumans/test/predictions"


ROUTES: tuple[tuple[str, tuple[str, ...]], ...] = (
    # Native disables the learned correction branch while preserving the same
    # checkpoint, detector stream, formal manifest, and evaluator contract.
    ("native", ("--ablation-token-mode", "native")),
    ("semantic_only", ("--ablation-token-mode", "semantic_only")),
    ("alignment_only", ("--ablation-token-mode", "alignment_only")),
    ("semantic_alignment", ("--ablation-token-mode", "semantic_alignment")),
    ("lora_off", ("--ablation-token-mode", "full", "--ablation-disable-head-lora")),
    (
        "camera_residual_off",
        ("--ablation-token-mode", "full", "--ablation-disable-camera-residual-head"),
    ),
    (
        "human_residual_off",
        ("--ablation-token-mode", "full", "--ablation-disable-human-latent-head"),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--devices",
        default="cuda:1,cuda:5,cuda:6",
        help=(
            "Legacy: comma-separated one-GPU route slots. Multi-GPU routes: "
            "semicolon-separated groups, e.g. "
            "'cuda:0,cuda:1,cuda:2;cuda:3,cuda:4'."
        ),
    )
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--reserve-gib", type=float, default=120.0)
    parser.add_argument(
        "--wait-pid",
        type=int,
        help="Do not launch a GPU route until this prerequisite smoke PID has exited.",
    )
    parser.add_argument(
        "--skip-full-replay",
        action="store_true",
        help="Do not re-evaluate the sealed full-model caches on the formal 90 cases.",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=OUTPUT_PARENT / "formal90_ablation_queue_state.json",
    )
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


def route_complete(name: str) -> bool:
    state_path = OUTPUT_PARENT / f"formal90_{name}/test/protocol_state.json"
    if not state_path.is_file():
        return False
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        state.get("status") == "complete"
        and int(state.get("complete_captures", 0)) == 27
        and int(state.get("formal_case_count", 0)) == 90
    )


def command_for(name: str, extra: tuple[str, ...], device: str, reserve_gib: float) -> list[str]:
    return [
        str(INFERENCE_PYTHON),
        str(RUNNER),
        "--split",
        "test",
        "--formal-manifest",
        str(FORMAL_MANIFEST),
        "--devices",
        device,
        "--candidate-json",
        str(CANDIDATE),
        "--outer",
        str(OUTER),
        "--outer-index",
        str(OUTER_INDEX),
        "--work-root",
        str(WORK_PARENT / f"EgoHuman_ablation_formal90_{name}"),
        "--output-root",
        str(OUTPUT_PARENT / f"formal90_{name}"),
        "--detector-cache-prediction-root",
        str(SEALED_FULL_PREDICTIONS),
        "--reserve-gib",
        str(reserve_gib),
        "--runner-extra-args",
        *extra,
    ]


def replay_command(reserve_gib: float) -> list[str]:
    """Evaluator-only replay of the sealed full Bridge3R predictions.

    The runner hard-links every existing cache and rejects a missing case, so
    this stage consumes disk/CPU for staging and GT evaluation but no GPU model
    forward.  Keeping it in the queue ensures the full row receives exactly the
    same formal 90-case denominator as every inference-time ablation.
    """

    return [
        str(INFERENCE_PYTHON),
        str(RUNNER),
        "--split",
        "test",
        "--formal-manifest",
        str(FORMAL_MANIFEST),
        "--devices",
        "cuda:1",
        "--candidate-json",
        str(CANDIDATE),
        "--outer",
        str(OUTER),
        "--outer-index",
        str(OUTER_INDEX),
        "--work-root",
        str(WORK_PARENT / "EgoHuman_ablation_formal90_full_replay"),
        "--output-root",
        str(OUTPUT_PARENT / "formal90_full_replay"),
        "--replay-prediction-root",
        str(SEALED_FULL_PREDICTIONS),
        "--reserve-gib",
        str(reserve_gib),
    ]


def main() -> None:
    args = parse_args()
    if ";" in args.devices:
        device_groups = [value.strip() for value in args.devices.split(";") if value.strip()]
    else:
        device_groups = [value.strip() for value in args.devices.split(",") if value.strip()]
    physical_devices = [
        device.strip()
        for group in device_groups
        for device in group.split(",")
        if device.strip()
    ]
    if not 1 <= len(physical_devices) <= 5 or len(set(physical_devices)) != len(physical_devices):
        raise ValueError("--devices must name one to five distinct GPUs")
    required_paths = [INFERENCE_PYTHON, RUNNER, FORMAL_MANIFEST, OUTER, OUTER_INDEX, CANDIDATE]
    if not args.skip_full_replay:
        required_paths.append(SEALED_FULL_PREDICTIONS)
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    state: dict[str, Any] = {
        "schema_version": "Bridge3R-EgoHumans-formal90-ablation-queue-v1",
        "protocol": "Bridge3R-EgoHumans-formal90-v1",
        "formal_manifest": str(FORMAL_MANIFEST),
        "device_groups": device_groups,
        "physical_devices": physical_devices,
        "routes": {},
        "started_at": time.time(),
        "status": "waiting_for_prerequisite" if args.wait_pid is not None else "running",
        "wait_pid": args.wait_pid,
    }
    if args.state.is_file():
        previous = json.loads(args.state.read_text(encoding="utf-8"))
        state["started_at"] = previous.get("started_at", state["started_at"])
        state["routes"] = previous.get("routes", {})
    pending = [(name, extra) for name, extra in ROUTES if not route_complete(name)]
    for name, _ in ROUTES:
        if route_complete(name):
            state["routes"][name] = {"status": "complete_reused", "completed_at": time.time()}
    active: dict[str, tuple[str, subprocess.Popen[Any], Any, float, list[str]]] = {}
    logs = OUTPUT_PARENT / "formal90_ablation_queue_logs"
    logs.mkdir(parents=True, exist_ok=True)
    atomic_json(args.state, state)

    if args.wait_pid is not None:
        print(f"waiting for prerequisite pid {args.wait_pid}", flush=True)
        while process_exists(int(args.wait_pid)):
            time.sleep(float(args.poll_seconds))
        state["status"] = "running"
        state["prerequisite_completed_at"] = time.time()
        atomic_json(args.state, state)

    while pending or active:
        while pending and len(active) < len(device_groups):
            device = next(value for value in device_groups if value not in active)
            name, extra = pending.pop(0)
            command = command_for(name, extra, device, float(args.reserve_gib))
            log_path = logs / f"{name}.launch.log"
            handle = log_path.open("a", encoding="utf-8")
            handle.write("\nCOMMAND " + json.dumps(command, ensure_ascii=False) + "\n")
            handle.flush()
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env={**os.environ, "TMPDIR": str(WORK_PARENT / f"EgoHuman_ablation_formal90_{name}/tmp")},
            )
            active[device] = (name, process, handle, time.time(), command)
            state["routes"][name] = {
                "status": "running",
                "device": device,
                "pid": process.pid,
                "started_at": time.time(),
                "command": command,
            }
            atomic_json(args.state, state)
            print(f"started {name} on {device} (pid {process.pid})", flush=True)
        if not active:
            continue
        time.sleep(float(args.poll_seconds))
        for device, (name, process, handle, started, command) in list(active.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            handle.close()
            done = route_complete(name)
            state["routes"][name].update(
                status="complete" if returncode == 0 and done else "error",
                returncode=returncode,
                elapsed_seconds=time.time() - started,
                completed_at=time.time(),
                protocol_complete=done,
            )
            del active[device]
            atomic_json(args.state, state)
            print(
                f"finished {name} on {device}: returncode={returncode}, protocol_complete={done}",
                flush=True,
            )
    errors = [name for name, value in state["routes"].items() if value.get("status") == "error"]
    if not errors and not args.skip_full_replay:
        replay_state = state["routes"].get("full_replay", {})
        if not route_complete("full_replay"):
            command = replay_command(float(args.reserve_gib))
            log_path = logs / "full_replay.launch.log"
            started = time.time()
            state["routes"]["full_replay"] = {
                "status": "running",
                "kind": "sealed_prediction_evaluator_replay",
                "started_at": started,
                "command": command,
            }
            atomic_json(args.state, state)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write("\nCOMMAND " + json.dumps(command, ensure_ascii=False) + "\n")
                handle.flush()
                completed = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "TMPDIR": str(WORK_PARENT / "EgoHuman_ablation_formal90_full_replay/tmp")},
                )
            done = route_complete("full_replay")
            state["routes"]["full_replay"].update(
                status="complete" if completed.returncode == 0 and done else "error",
                returncode=completed.returncode,
                elapsed_seconds=time.time() - started,
                completed_at=time.time(),
                protocol_complete=done,
            )
            atomic_json(args.state, state)
            if completed.returncode or not done:
                errors.append("full_replay")
        elif replay_state.get("status") != "complete":
            state["routes"]["full_replay"] = {
                "status": "complete_reused",
                "kind": "sealed_prediction_evaluator_replay",
                "completed_at": time.time(),
            }
            atomic_json(args.state, state)
    state.update(
        status="complete" if not errors else "error",
        errors=errors,
        completed_at=time.time(),
    )
    atomic_json(args.state, state)
    if errors:
        raise SystemExit(f"Formal ablation queue completed with failures: {errors}")


if __name__ == "__main__":
    main()
