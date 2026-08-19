#!/usr/bin/env python3
"""Stage, infer, evaluate, and clean one preregistered Harmony4D holdout."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
STAGER = REPO_ROOT / "versions/v15/harmony4d/stage_archive.py"
RUN_CASE = REPO_ROOT / "versions/v15/harmony4d/run_harmony_case.py"
PROBE = REPO_ROOT / "versions/v16/harmony4d/probe_causal_stabilization.py"
OUTER = Path("/data/wangzheng/iJCV-CODE/data/Harmony4D.zip")
WORK = Path("/data/wangzheng/iJCV-CODE/data/Harmony4D_work_v17_holdout")
OUTPUT = REPO_ROOT / "output/v17_harmony4d/new_holdout"
DEVICES = ("cuda:0", "cuda:2", "cuda:3", "cuda:6")
REFERENCES = ("m0_strict_human3r", "m15_safe_boundary_permutation_causal_gru")
REQUIRED_METRICS = {
    "W-MPJPE_mm", "WA-MPJPE_mm", "MPJPE_mm", "MPVPE_mm",
    "Accel_mm_frame2", "RTE_H3R_percent", "ATE_Sim3_m", "IDF1", "Coverage",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", required=True, choices=("train/10_karate2.zip", "train/11_karate3.zip"))
    parser.add_argument("--candidate-json", type=Path, default=REPO_ROOT / "versions/v17/harmony4d/frozen_multicue_candidate.json")
    parser.add_argument("--devices", default=",".join(DEVICES))
    parser.add_argument("--reserve-gib", type=float, default=80.0)
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(partial, path)


def run(command: list[str], log: Path) -> tuple[int, float]:
    started = time.perf_counter()
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        handle.write("COMMAND " + json.dumps(command, ensure_ascii=False) + "\n")
        handle.flush()
        result = subprocess.run(
            command, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT,
            env={**os.environ, "TMPDIR": str(WORK / "tmp")},
        )
    return result.returncode, time.perf_counter() - started


def safe_remove(path: Path) -> None:
    resolved = path.resolve()
    parent = (WORK / "staging").resolve()
    if resolved == parent or parent not in resolved.parents:
        raise ValueError(f"refusing cleanup outside per-sequence staging: {resolved}")
    if resolved.is_dir():
        shutil.rmtree(resolved)


def validate_report(path: Path, expected_cases: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("errors"):
        raise ValueError(payload["errors"])
    complete = {
        row["case_id"] for row in payload.get("rows", [])
        if row.get("candidate") == "v17_harmony_multicue_safe" and row.get("status") == "complete"
    }
    skipped = payload.get("skipped_cases", [])
    if len(complete) + len(skipped) != expected_cases:
        raise ValueError(f"{len(complete)} complete + {len(skipped)} skipped != {expected_cases}")
    for row in payload.get("rows", []):
        if row.get("candidate") == "v17_harmony_multicue_safe" and row.get("status") == "complete":
            if not REQUIRED_METRICS <= set(row.get("metrics", {})):
                raise ValueError(f"missing metrics in {row['case_id']}")
    return payload


def main() -> None:
    args = parse_args()
    entry = PurePosixPath(args.entry)
    slug = "_".join(entry.with_suffix("").parts)
    label = entry.stem
    stage_meta = OUTPUT / "staging" / label
    manifest = stage_meta / "generated_manifest.jsonl"
    staging = WORK / "staging" / slug
    prediction_root = OUTPUT / "predictions" / label
    report = OUTPUT / "per_sequence" / f"{label}.json"
    state_path = OUTPUT / "state" / f"{label}.json"
    state: dict[str, Any] = {
        "entry": args.entry, "status": "started", "started_at": time.time(),
        "devices": args.devices.split(","),
    }
    atomic_json(state_path, state)
    stage_command = [
        sys.executable, str(STAGER), "--outer", str(OUTER), "--entry", args.entry,
        "--work-root", str(WORK),
        "--audit-output", str(stage_meta / "audit.json"),
        "--index-output", str(stage_meta / "index.json"),
        "--manifest-output", str(manifest),
        "--ledger-output", str(stage_meta / "stage_ledger.json"),
        "--reserve-gib", str(args.reserve_gib),
    ]
    returncode, seconds = run(stage_command, OUTPUT / "logs" / f"{label}.stage.log")
    state["stage_seconds"] = seconds
    if returncode:
        state.update(status="stage_error", returncode=returncode)
        atomic_json(state_path, state)
        raise SystemExit(returncode)
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    if len(devices) < len(rows):
        raise ValueError(f"need {len(rows)} devices, got {len(devices)}")
    prediction_root.mkdir(parents=True, exist_ok=True)
    processes = []
    inference_rows = []
    for line, (record, device) in enumerate(zip(rows, devices), 1):
        case_id = str(record["case_id"])
        cache = prediction_root / f"{case_id}.npz"
        runtime = prediction_root / f"{case_id}.runtime.json"
        if cache.is_file() and runtime.is_file():
            inference_rows.append({"case_id": case_id, "device": device, "status": "cached"})
            continue
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
            env={**os.environ, "TMPDIR": str(WORK / "tmp")},
        )
        processes.append((case_id, device, process, handle, time.perf_counter()))
    for case_id, device, process, handle, started in processes:
        returncode = process.wait()
        handle.close()
        inference_rows.append({
            "case_id": case_id, "device": device,
            "status": "complete" if returncode == 0 else "error",
            "returncode": returncode, "elapsed_seconds": time.perf_counter() - started,
        })
    state["inference"] = inference_rows
    failures = [row for row in inference_rows if row["status"] == "error"]
    if failures:
        state.update(status="inference_error", failures=failures)
        atomic_json(state_path, state)
        raise SystemExit(2)
    probe_command = [
        sys.executable, str(PROBE), "--prediction-roots", str(prediction_root),
        "--extracted-root", str(staging), "--output", str(report),
        "--candidate-json", str(args.candidate_json.resolve()),
        "--reference-methods", *REFERENCES,
    ]
    returncode, seconds = run(probe_command, OUTPUT / "logs" / f"{label}.probe.log")
    state["probe_seconds"] = seconds
    if returncode:
        state.update(status="probe_error", returncode=returncode)
        atomic_json(state_path, state)
        raise SystemExit(returncode)
    payload = validate_report(report, len(rows))
    safe_remove(staging)
    state.update(
        status="complete", report=str(report.resolve()), manifest=str(manifest.resolve()),
        case_count=len(rows), evaluator_unavailable=len(payload.get("skipped_cases", [])),
        staging_removed=True, completed_at=time.time(),
    )
    atomic_json(state_path, state)
    print(json.dumps(state, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
