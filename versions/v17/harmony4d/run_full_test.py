#!/usr/bin/env python3
"""Run frozen Movie3R-v17 on every evaluable Harmony4D test capture.

The outer archive is kept immutable.  One nested action archive is staged at
a time, every structurally eligible 150-frame capture is audited, and only
coordinate-valid captures enter the deterministic four-angle manifest.  The
runner reuses the already frozen first-capture v15 caches, is resumable, and
removes expanded data only after inference and evaluation validate.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
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
V15 = REPO_ROOT / "versions/v15/harmony4d"
V16 = REPO_ROOT / "versions/v16/harmony4d"
STAGER = V15 / "stage_archive.py"
AUDITOR = V15 / "audit_sequence.py"
BUILDER = V15 / "build_manifest.py"
RUN_CASE = V15 / "run_harmony_case.py"
PROBE = REPO_ROOT / "versions/v17/harmony4d/probe_parallel.py"
AGGREGATE = V16 / "aggregate_multisequence.py"
CANDIDATE = REPO_ROOT / "versions/v17/harmony4d/frozen_multicue_candidate.json"
OUTER = Path("/data/wangzheng/iJCV-CODE/data/Harmony4D.zip")
WORK = Path("/data/wangzheng/iJCV-CODE/data/Harmony4D_work_v17_full_test")
OUTPUT = REPO_ROOT / "output/v17_harmony4d/full_test"
ENTRIES = (
    "test/01_hugging.zip",
    "test/03_grappling2.zip",
    "test/05_sword2.zip",
    "test/06_sword3.zip",
    "test/08_ballroom2.zip",
    "test/15_mma4.zip",
    "test/16_mma5.zip",
)
DEFAULT_DEVICES = ("cuda:0", "cuda:2", "cuda:3", "cuda:6")
REFERENCES = ("m0_strict_human3r", "m15_safe_boundary_permutation_causal_gru")
PRIMARY = "v17_harmony_multicue_safe"
PARENT = "v16_0_m15_geometry"
REQUIRED_METRICS = {
    "W-MPJPE_mm", "WA-MPJPE_mm", "MPJPE_mm", "MPVPE_mm",
    "Accel_mm_frame2", "RTE_H3R_percent", "ATE_Sim3_m", "IDF1", "Coverage",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entries", nargs="*", choices=ENTRIES, default=list(ENTRIES))
    parser.add_argument("--devices", default=",".join(DEFAULT_DEVICES))
    parser.add_argument("--reserve-gib", type=float, default=80.0)
    parser.add_argument("--candidate-json", type=Path, default=CANDIDATE)
    parser.add_argument("--keep-staging", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(partial, path)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def run(command: list[str], log: Path) -> tuple[int, float]:
    started = time.perf_counter()
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        handle.write("COMMAND " + json.dumps(command, ensure_ascii=False) + "\n")
        handle.flush()
        completed = subprocess.run(
            command, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT,
            env={**os.environ, "TMPDIR": str(WORK / "tmp")},
        )
    return completed.returncode, time.perf_counter() - started


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def primary_from_candidate(path: Path) -> str:
    payload = read_json(path)
    names = [str(row["name"]) for row in payload["candidates"] if str(row["name"]) != PARENT]
    if len(names) != 1:
        raise ValueError(f"expected one non-parent candidate in {path}, got {names}")
    return names[0]


def safe_remove_staging(path: Path) -> None:
    resolved = path.resolve()
    parent = (WORK / "staging").resolve()
    if resolved == parent or parent not in resolved.parents:
        raise ValueError(f"refusing cleanup outside per-entry staging: {resolved}")
    if resolved.is_dir():
        shutil.rmtree(resolved)


def existing_prediction_roots(sequence: str) -> list[Path]:
    pattern = str(REPO_ROOT / "output/v15_harmony4d/predictions" / f"test_{sequence}*")
    return [Path(value).resolve() for value in sorted(glob.glob(pattern)) if Path(value).is_dir()]


def existing_cases(roots: list[Path]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for root in roots:
        for runtime_path in sorted(root.glob("*.runtime.json")):
            runtime = read_json(runtime_path)
            case_id = str(runtime["record"]["case_id"])
            cache = runtime_path.with_name(runtime_path.name.removesuffix(".runtime.json") + ".npz")
            if not cache.is_file():
                raise FileNotFoundError(cache)
            if case_id in output:
                raise ValueError(f"duplicate frozen cache for {case_id}")
            output[case_id] = runtime_path
    return output


def audit_all_captures(
    entry: str, stage_meta: Path, staging: Path, index: dict[str, Any], state: dict[str, Any],
) -> list[Path]:
    audit_root = stage_meta / "audits"
    audit_root.mkdir(parents=True, exist_ok=True)
    successful: list[Path] = []
    attempts: list[dict[str, Any]] = []
    for candidate in index.get("eligible_ranked", []):
        capture = str(candidate["capture_relative"])
        capture_name = Path(capture).name
        output = audit_root / f"{capture_name}.json"
        if output.is_file() and read_json(output).get("capture_relative") == capture:
            attempts.append({"capture_relative": capture, "status": "cached_valid", "audit": str(output)})
            successful.append(output)
            continue
        command = [
            sys.executable, str(AUDITOR), "--extracted-root", str(staging),
            "--archive-entry", entry, "--capture-relative", capture,
            "--output", str(output), "--overlay-cameras", "cam01,cam08,cam15",
        ]
        returncode, seconds = run(
            command, OUTPUT / "logs" / f"{'_'.join(PurePosixPath(entry).with_suffix('').parts)}.audit.{capture_name}.log",
        )
        row = {
            "capture_relative": capture, "returncode": returncode, "seconds": seconds,
            "status": "coordinate_valid" if returncode == 0 else "evaluator_unavailable_coordinate_audit",
            "audit": str(output) if returncode == 0 else None,
        }
        attempts.append(row)
        state["audit_attempts"] = attempts
        atomic_json(stage_meta / "state.json", state)
        if returncode == 0:
            successful.append(output)
    state["audit_attempts"] = attempts
    state["structurally_eligible_captures"] = len(index.get("eligible_ranked", []))
    state["coordinate_valid_captures"] = len(successful)
    state["coordinate_invalid_captures"] = len(attempts) - len(successful)
    atomic_json(stage_meta / "state.json", state)
    return successful


def launch_inference(
    rows: list[dict[str, Any]], manifest: Path, staging: Path, prediction_root: Path,
    devices: list[str], frozen: dict[str, Path], state: dict[str, Any], stage_meta: Path,
) -> None:
    prediction_root.mkdir(parents=True, exist_ok=True)
    pending: deque[tuple[int, dict[str, Any]]] = deque()
    inference: dict[str, dict[str, Any]] = {}
    for line, record in enumerate(rows, 1):
        case_id = str(record["case_id"])
        cache = prediction_root / f"{case_id}.npz"
        runtime = prediction_root / f"{case_id}.runtime.json"
        if case_id in frozen:
            inference[case_id] = {"status": "reused_frozen_v15", "runtime": str(frozen[case_id])}
        elif cache.is_file() and runtime.is_file() and read_json(runtime)["record"]["case_id"] == case_id:
            inference[case_id] = {"status": "cached_full_test", "runtime": str(runtime)}
        else:
            pending.append((line, record))
    active: dict[str, tuple[subprocess.Popen[Any], Any, str, float]] = {}
    failures: list[dict[str, Any]] = []
    while pending or active:
        free_devices = [device for device in devices if device not in active]
        while pending and free_devices:
            line, record = pending.popleft()
            device = free_devices.pop(0)
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
                env={**os.environ, "TMPDIR": str(WORK / "tmp")},
            )
            active[device] = (process, handle, case_id, time.perf_counter())
            inference[case_id] = {"status": "running", "device": device, "pid": process.pid}
        state["inference"] = inference
        atomic_json(stage_meta / "state.json", state)
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
            inference[case_id] = row
            if returncode:
                failures.append({"case_id": case_id, **row})
            del active[device]
    state["inference"] = inference
    atomic_json(stage_meta / "state.json", state)
    if failures:
        raise RuntimeError(f"inference failures: {failures}")


def validate_report(path: Path, expected_cases: int, primary: str) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("errors"):
        raise ValueError(payload["errors"])
    complete = {
        row["case_id"] for row in payload.get("rows", [])
        if row.get("candidate") == primary and row.get("status") == "complete"
    }
    skipped = payload.get("skipped_cases", [])
    if len(complete) + len(skipped) != expected_cases:
        raise ValueError(f"{len(complete)} complete + {len(skipped)} skipped != {expected_cases}")
    for row in payload.get("rows", []):
        if row.get("candidate") == primary and row.get("status") == "complete":
            if not REQUIRED_METRICS <= set(row.get("metrics", {})):
                raise ValueError(f"missing metrics in {row['case_id']}")
    return payload


def process_entry(
    entry: str, devices: list[str], reserve_gib: float, keep_staging: bool,
    candidate: Path,
) -> None:
    entry_path = PurePosixPath(entry)
    sequence = entry_path.stem
    slug = "_".join(entry_path.with_suffix("").parts)
    stage_meta = OUTPUT / "staging" / sequence
    staging = WORK / "staging" / slug
    manifest = stage_meta / "full_capture_manifest.jsonl"
    prediction_root = OUTPUT / "predictions" / sequence
    report = OUTPUT / "per_sequence" / f"{sequence}.json"
    state_path = stage_meta / "state.json"
    candidate_digest = sha256(candidate)
    primary = primary_from_candidate(candidate)
    if state_path.is_file() and report.is_file():
        previous = read_json(state_path)
        if (
            previous.get("status") == "complete"
            and previous.get("candidate_sha256") == candidate_digest
            and previous.get("primary") == primary
        ):
            payload = validate_report(
                report, int(previous["manifest_cases"]), primary,
            )
            print(json.dumps({
                "entry": entry, "status": "already_complete",
                "evaluable_cases": payload["complete_case_count"],
                "evaluator_unavailable_cases": len(payload.get("skipped_cases", [])),
            }), flush=True)
            return
    state: dict[str, Any] = {
        "schema_version": "Movie3R-v17-Harmony4D-full-test-state-v1",
        "entry": entry, "sequence": sequence, "status": "started",
        "started_at": time.time(), "devices": devices,
        "candidate": str(candidate.resolve()), "candidate_sha256": candidate_digest,
    }
    state["primary"] = primary
    atomic_json(state_path, state)
    stage_ledger = stage_meta / "stage_ledger.json"
    reusable_stage = bool(
        staging.is_dir() and stage_ledger.is_file() and (stage_meta / "index.json").is_file()
        and read_json(stage_ledger).get("status") == "staged_audited_manifest_frozen"
        and read_json(stage_ledger).get("entry") == entry
    )
    stage_command = [
        sys.executable, str(STAGER), "--outer", str(OUTER), "--entry", entry,
        "--work-root", str(WORK), "--audit-output", str(stage_meta / "seed_audit.json"),
        "--index-output", str(stage_meta / "index.json"),
        "--manifest-output", str(stage_meta / "seed_manifest.jsonl"),
        "--ledger-output", str(stage_meta / "stage_ledger.json"),
        "--reserve-gib", str(reserve_gib),
    ]
    if reusable_stage:
        returncode, seconds = 0, 0.0
        state["stage_reused_after_resume"] = True
    else:
        returncode, seconds = run(stage_command, OUTPUT / "logs" / f"{sequence}.stage.log")
    state["stage_seconds"] = seconds
    if returncode:
        state.update(status="stage_error", returncode=returncode)
        atomic_json(state_path, state)
        raise RuntimeError(f"stage failed for {entry}")
    index = read_json(stage_meta / "index.json")
    audits = audit_all_captures(entry, stage_meta, staging, index, state)
    if not audits:
        state.update(status="no_coordinate_valid_capture")
        atomic_json(state_path, state)
        if not keep_staging:
            safe_remove_staging(staging)
        return
    build_command = [
        sys.executable, str(BUILDER), "--audits", *[str(path) for path in audits],
        "--split", "test", "--output", str(manifest), "--pre-count", "75", "--post-count", "75",
    ]
    returncode, seconds = run(build_command, OUTPUT / "logs" / f"{sequence}.manifest.log")
    state["manifest_seconds"] = seconds
    if returncode:
        state.update(status="manifest_error", returncode=returncode)
        atomic_json(state_path, state)
        raise RuntimeError(f"manifest failed for {entry}")
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    state.update(manifest=str(manifest.resolve()), manifest_sha256=sha256(manifest), manifest_cases=len(rows))
    atomic_json(state_path, state)
    frozen_roots = existing_prediction_roots(sequence)
    frozen = existing_cases(frozen_roots)
    launch_inference(rows, manifest, staging, prediction_root, devices, frozen, state, stage_meta)
    roots = [*frozen_roots, prediction_root]
    probe_command = [
        sys.executable, str(PROBE), "--prediction-roots", *[str(path) for path in roots],
        "--extracted-root", str(staging), "--output", str(report),
        "--candidate-json", str(candidate), "--reference-methods", *REFERENCES,
    ]
    returncode, seconds = run(probe_command, OUTPUT / "logs" / f"{sequence}.probe.log")
    state["probe_seconds"] = seconds
    if returncode:
        state.update(status="probe_error", returncode=returncode)
        atomic_json(state_path, state)
        raise RuntimeError(f"probe failed for {entry}")
    payload = validate_report(report, len(rows), primary)
    if not keep_staging:
        safe_remove_staging(staging)
    state.update(
        status="complete", report=str(report.resolve()), completed_at=time.time(),
        evaluable_cases=payload["complete_case_count"],
        evaluator_unavailable_cases=len(payload.get("skipped_cases", [])),
        staging_removed=not keep_staging,
    )
    atomic_json(state_path, state)


def aggregate(entries: list[str], candidate: Path) -> None:
    reports = [OUTPUT / "per_sequence" / f"{PurePosixPath(entry).stem}.json" for entry in entries]
    missing = [str(path) for path in reports if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing per-sequence reports: {missing}")
    command = [
        sys.executable, str(AGGREGATE), "--inputs", *[str(path) for path in reports],
        "--output", str(OUTPUT / "paper"), "--primary", primary_from_candidate(candidate), "--parent", PARENT,
        "--primary-display-name", "Movie3R frozen full method",
        "--parent-display-name", "Movie3R-v17 parent",
        "--title", "Movie3R-v17 Harmony4D full test-capture summary",
        "--test-used-for-parameter-selection",
    ]
    returncode, _ = run(command, OUTPUT / "logs/aggregate.log")
    if returncode:
        raise RuntimeError("full-test aggregation failed")


def main() -> None:
    args = parse_args()
    entries = list(args.entries)
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    if not devices:
        raise ValueError("at least one device is required")
    if not args.aggregate_only:
        for entry in entries:
            process_entry(
                entry, devices, float(args.reserve_gib), bool(args.keep_staging),
                args.candidate_json.resolve(),
            )
    aggregate(entries, args.candidate_json.resolve())
    print(json.dumps({
        "status": "complete", "entries": entries,
        "paper": str((OUTPUT / "paper").resolve()),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
