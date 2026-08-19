#!/usr/bin/env python3
"""Run frozen Harmony4D length/hyperparameter development or holdout phases."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path, PurePosixPath
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
STAGER = REPO_ROOT / "versions/v15/harmony4d/stage_archive.py"
BUILDER = REPO_ROOT / "versions/v15/harmony4d/build_manifest.py"
RUN_CASE = REPO_ROOT / "versions/v15/harmony4d/run_harmony_case.py"
PROBE = REPO_ROOT / "versions/v16/harmony4d/probe_causal_stabilization.py"
GRID = REPO_ROOT / "versions/v18/harmony4d/development_grid.json"
OUTER = Path("/data/wangzheng/iJCV-CODE/data/Harmony4D.zip")
DEV_ENTRIES = (
    "train/02_grappling.zip",
    "train/07_ballroom.zip",
    "train/12_mma.zip",
)
HOLDOUT_ENTRIES = (
    "train/04_sword_part1.zip",
    "train/08_ballroom2.zip",
    "train/13_mma2.zip",
)
DEFAULT_DEVICES = ("cuda:0", "cuda:2", "cuda:3", "cuda:6")
REFERENCES = ("m0_strict_human3r", "m15_safe_boundary_permutation_causal_gru")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("dev", "holdout"), required=True)
    parser.add_argument("--entries", nargs="*")
    parser.add_argument("--lengths", default="60,90,120,150")
    parser.add_argument("--devices", default=",".join(DEFAULT_DEVICES))
    parser.add_argument("--candidate-json", type=Path)
    parser.add_argument("--reserve-gib", type=float, default=80.0)
    parser.add_argument("--keep-staging", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(partial, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run(command: list[str], log: Path, work: Path) -> tuple[int, float]:
    started = time.perf_counter()
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        handle.write("COMMAND " + json.dumps(command, ensure_ascii=False) + "\n")
        handle.flush()
        completed = subprocess.run(
            command, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT,
            env={**os.environ, "TMPDIR": str(work / "tmp")},
        )
    return completed.returncode, time.perf_counter() - started


def safe_remove_staging(path: Path, work: Path) -> None:
    resolved = path.resolve()
    parent = (work / "staging").resolve()
    if resolved == parent or parent not in resolved.parents:
        raise ValueError(f"refusing cleanup outside staging: {resolved}")
    if resolved.is_dir():
        shutil.rmtree(resolved)


def infer_manifest(
    manifest: Path, staging: Path, prediction_root: Path, devices: list[str],
    state: dict[str, Any], state_path: Path, work: Path,
) -> None:
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    prediction_root.mkdir(parents=True, exist_ok=True)
    pending: deque[tuple[int, dict[str, Any]]] = deque()
    progress: dict[str, dict[str, Any]] = {}
    for line, record in enumerate(rows, 1):
        case_id = str(record["case_id"])
        cache = prediction_root / f"{case_id}.npz"
        runtime = prediction_root / f"{case_id}.runtime.json"
        if cache.is_file() and runtime.is_file() and read_json(runtime)["record"]["case_id"] == case_id:
            progress[case_id] = {"status": "cached"}
        else:
            pending.append((line, record))
    active: dict[str, tuple[subprocess.Popen[Any], Any, str, float]] = {}
    failures = []
    while pending or active:
        free = [device for device in devices if device not in active]
        while pending and free:
            line, record = pending.popleft()
            device = free.pop(0)
            case_id = str(record["case_id"])
            cache = prediction_root / f"{case_id}.npz"
            log = prediction_root / "logs" / f"{case_id}.inference.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("w", encoding="utf-8")
            command = [
                sys.executable, str(RUN_CASE), "--manifest", str(manifest), "--line", str(line),
                "--extracted-root", str(staging), "--output", str(cache), "--device", device,
            ]
            handle.write("COMMAND " + json.dumps(command, ensure_ascii=False) + "\n")
            handle.flush()
            process = subprocess.Popen(
                command, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT,
                env={**os.environ, "TMPDIR": str(work / "tmp")},
            )
            active[device] = (process, handle, case_id, time.perf_counter())
            progress[case_id] = {"status": "running", "device": device, "pid": process.pid}
        state["inference"] = progress
        atomic_json(state_path, state)
        if not active:
            continue
        time.sleep(1.0)
        for device, (process, handle, case_id, started) in list(active.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            handle.close()
            row = {
                "status": "complete" if returncode == 0 else "error", "device": device,
                "returncode": returncode, "seconds": time.perf_counter() - started,
            }
            progress[case_id] = row
            if returncode:
                failures.append({"case_id": case_id, **row})
            del active[device]
    state["inference"] = progress
    atomic_json(state_path, state)
    if failures:
        raise RuntimeError(f"inference failures: {failures}")


def validate_probe(path: Path, expected: int) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("errors"):
        raise ValueError(payload["errors"])
    complete = int(payload.get("complete_case_count", 0))
    skipped = len(payload.get("skipped_cases", []))
    if complete + skipped != expected:
        raise ValueError(f"{complete} complete + {skipped} skipped != {expected}")
    return payload


def process_entry(
    phase: str, entry: str, lengths: list[int], devices: list[str], candidate: Path,
    reserve_gib: float, keep_staging: bool, work: Path, output: Path,
) -> None:
    sequence = PurePosixPath(entry).stem
    slug = "_".join(PurePosixPath(entry).with_suffix("").parts)
    stage_meta = output / "staging" / sequence
    state_path = stage_meta / "state.json"
    expected_reports = [output / "per_sequence" / sequence / f"cs{length}.json" for length in lengths]
    if state_path.is_file() and read_json(state_path).get("status") == "complete" and all(
        path.is_file() for path in expected_reports
    ):
        print(json.dumps({"entry": entry, "status": "already_complete"}), flush=True)
        return
    staging = work / "staging" / slug
    state: dict[str, Any] = {
        "schema_version": "Movie3R-v18-Harmony4D-development-state-v1",
        "phase": phase, "entry": entry, "sequence": sequence, "lengths": lengths,
        "candidate_json": str(candidate.resolve()), "status": "started", "started_at": time.time(),
    }
    atomic_json(state_path, state)
    stage_command = [
        sys.executable, str(STAGER), "--outer", str(OUTER), "--entry", entry,
        "--work-root", str(work), "--audit-output", str(stage_meta / "audit.json"),
        "--index-output", str(stage_meta / "index.json"),
        "--manifest-output", str(stage_meta / "seed_cs150.jsonl"),
        "--ledger-output", str(stage_meta / "stage_ledger.json"),
        "--reserve-gib", str(reserve_gib),
    ]
    returncode, seconds = run(stage_command, output / "logs" / f"{sequence}.stage.log", work)
    state["stage_seconds"] = seconds
    if returncode:
        state.update(status="stage_error", returncode=returncode)
        atomic_json(state_path, state)
        raise RuntimeError(f"stage failed for {entry}")
    audit = Path(read_json(stage_meta / "stage_ledger.json")["selected_audit"])
    for length in lengths:
        if length <= 0 or length % 2:
            raise ValueError(f"length must be a positive even number: {length}")
        length_root = output / "lengths" / f"cs{length}" / sequence
        manifest = length_root / "manifest.jsonl"
        build_command = [
            sys.executable, str(BUILDER), "--audits", str(audit),
            "--split", "dev" if phase == "dev" else "test", "--output", str(manifest),
            "--pre-count", str(length // 2), "--post-count", str(length // 2),
        ]
        returncode, _ = run(build_command, output / "logs" / f"{sequence}.cs{length}.manifest.log", work)
        if returncode:
            raise RuntimeError(f"manifest failed for {entry} length {length}")
        rows = [line for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        predictions = length_root / "predictions"
        state["active_length"] = length
        atomic_json(state_path, state)
        infer_manifest(manifest, staging, predictions, devices, state, state_path, work)
        report = output / "per_sequence" / sequence / f"cs{length}.json"
        probe_command = [
            sys.executable, str(PROBE), "--prediction-roots", str(predictions),
            "--extracted-root", str(staging), "--output", str(report),
            "--candidate-json", str(candidate), "--reference-methods", *REFERENCES,
        ]
        returncode, probe_seconds = run(
            probe_command, output / "logs" / f"{sequence}.cs{length}.probe.log", work,
        )
        if returncode:
            raise RuntimeError(f"probe failed for {entry} length {length}")
        payload = validate_probe(report, len(rows))
        state.setdefault("length_results", {})[str(length)] = {
            "manifest_cases": len(rows), "evaluable_cases": payload["complete_case_count"],
            "evaluator_unavailable": len(payload.get("skipped_cases", [])),
            "probe_seconds": probe_seconds, "report": str(report.resolve()),
        }
        atomic_json(state_path, state)
    if not keep_staging:
        safe_remove_staging(staging, work)
    state.update(status="complete", completed_at=time.time(), staging_removed=not keep_staging)
    state.pop("active_length", None)
    atomic_json(state_path, state)


def main() -> None:
    args = parse_args()
    default_entries = DEV_ENTRIES if args.phase == "dev" else HOLDOUT_ENTRIES
    entries = list(args.entries) if args.entries else list(default_entries)
    allowed = set(DEV_ENTRIES + HOLDOUT_ENTRIES)
    unknown = [entry for entry in entries if entry not in allowed]
    if unknown:
        raise ValueError(f"entries outside frozen development plan: {unknown}")
    lengths = sorted({int(value) for value in args.lengths.split(",") if value.strip()})
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    if not devices:
        raise ValueError("at least one CUDA device is required")
    if args.candidate_json is None:
        if args.phase != "dev":
            raise ValueError("holdout requires an explicitly frozen --candidate-json")
        candidate = GRID
    else:
        candidate = args.candidate_json.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    work = Path(f"/data/wangzheng/iJCV-CODE/data/Harmony4D_work_v18_{args.phase}")
    output = REPO_ROOT / f"output/v18_harmony4d/{args.phase}"
    for entry in entries:
        process_entry(
            args.phase, entry, lengths, devices, candidate, float(args.reserve_gib),
            bool(args.keep_staging), work, output,
        )
    print(json.dumps({
        "status": "complete", "phase": args.phase, "entries": entries,
        "lengths": lengths, "candidate": str(candidate), "output": str(output.resolve()),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
