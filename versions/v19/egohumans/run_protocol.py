#!/usr/bin/env python3
"""Resumable, disk-bounded Movie3R-EgoHumans-CS100-v1 runner.

The outer ZIP is immutable.  Exactly one capture is staged at a time; four
pre-registered camera spans are inferred in parallel, evaluated while GT is
available, and only then is the expanded copy removed.  Development may run
the finite v16 exploration grid and additional frozen candidate JSON files.
Holdout/test are expected to receive only pre-frozen candidate JSON files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from versions.v19.egohumans.stage_capture import capture_name, safe_stage_path, slug


OUTER = Path("/data/wangzheng/iJCV-CODE/data/EgoHuman.zip")
WORK = Path("/data/wangzheng/iJCV-CODE/data/EgoHuman_work_v19")
INDEX = WORK / "outer_index.json"
OUTPUT = REPO_ROOT / "output/v19_egohumans"
STAGER = REPO_ROOT / "versions/v19/egohumans/stage_capture.py"
BUILDER = REPO_ROOT / "versions/v19/egohumans/build_manifest.py"
RUN_CASE = REPO_ROOT / "versions/v15/harmony4d/run_harmony_case.py"
PROBE = REPO_ROOT / "versions/v19/egohumans/probe_parallel.py"
V19_PROBE = REPO_ROOT / "versions/v19/egohumans/probe_v19_candidates.py"
REFERENCES = ("m0_strict_human3r", "m15_safe_boundary_permutation_causal_gru")
DEFAULT_DEVICES = ("cuda:0", "cuda:2", "cuda:3", "cuda:5")
REQUIRED_METRICS = {
    "W-MPJPE_mm",
    "WA-MPJPE_mm",
    "MPJPE_mm",
    "MPVPE_mm",
    "Accel_mm_frame2",
    "ATE_Sim3_m",
    "ATE_SE3_m",
    "IDs",
    "IDF1",
    "Coverage",
    "Seam_root_m",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("development", "holdout", "test"), required=True)
    parser.add_argument("--devices", default=",".join(DEFAULT_DEVICES))
    parser.add_argument("--entries", nargs="*", help="Optional exact outer-ZIP entries from the frozen split.")
    parser.add_argument("--candidate-json", type=Path, nargs="*", default=[])
    parser.add_argument(
        "--include-exploration",
        action="store_true",
        help="Evaluate the code-frozen finite grid; permitted only on development.",
    )
    parser.add_argument("--reserve-gib", type=float, default=80.0)
    parser.add_argument("--keep-staging", action="store_true")
    parser.add_argument("--continue-on-structural-error", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(partial, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
            command,
            cwd=REPO_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env={**os.environ, "TMPDIR": str(WORK / "tmp")},
        )
    return completed.returncode, time.perf_counter() - started


def selected_entries(split: str, requested: list[str] | None) -> list[dict[str, Any]]:
    payload = read_json(INDEX)
    rows = [row for row in payload["entries"] if row["split"] == split]
    by_entry = {str(row["entry"]): row for row in rows}
    if requested:
        unknown = sorted(set(requested) - set(by_entry))
        if unknown:
            raise ValueError(f"Entries are not in frozen {split} split: {unknown}")
        return [by_entry[value] for value in requested]
    return sorted(rows, key=lambda row: (str(row["action"]), str(row["entry"])))


def report_tag(candidate: Path | None) -> str:
    if candidate is None:
        return "exploration_grid"
    return f"candidate_{candidate.stem}_{sha256(candidate)[:10]}"


def probe_for_candidate(candidate: Path | None) -> Path:
    if candidate is None:
        return PROBE
    try:
        payload = read_json(candidate)
        is_v19 = any("geometry" in row for row in payload.get("candidates", []))
    except (OSError, json.JSONDecodeError):
        is_v19 = False
    return V19_PROBE if is_v19 else PROBE


def valid_cache(cache: Path, runtime_path: Path, case_id: str) -> bool:
    if not cache.is_file() or not runtime_path.is_file():
        return False
    try:
        runtime = read_json(runtime_path)
        if str(runtime["record"]["case_id"]) != case_id:
            return False
        if not {"m0_strict_human3r", "m3_b0_only", "m15_safe_boundary_permutation_causal_gru"} <= set(runtime["methods"]):
            return False
        with np.load(cache, allow_pickle=False) as data:
            required = {
                "m0_strict_human3r__cameras_c2w",
                "m3_b0_only__cameras_c2w",
                "m3_b0_only__vertices_world",
                "m3_b0_only__joints_world",
                "m3_b0_only__valid",
                "m15_safe_boundary_permutation_causal_gru__cameras_c2w",
            }
            return required <= set(data.files)
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return False


def link_smoke_cache(prediction_root: Path, case_id: str) -> bool:
    """Reuse the audited smoke case without copying its 300+ MiB cache."""

    smoke = OUTPUT / "smoke/predictions"
    source_cache = smoke / f"{case_id}.npz"
    source_runtime = smoke / f"{case_id}.runtime.json"
    if not valid_cache(source_cache, source_runtime, case_id):
        return False
    prediction_root.mkdir(parents=True, exist_ok=True)
    target_cache = prediction_root / source_cache.name
    target_runtime = prediction_root / source_runtime.name
    if not target_cache.exists():
        os.link(source_cache, target_cache)
    if not target_runtime.exists():
        os.link(source_runtime, target_runtime)
    return valid_cache(target_cache, target_runtime, case_id)


def launch_inference(
    rows: list[dict[str, Any]],
    manifest: Path,
    extracted_root: Path,
    prediction_root: Path,
    log_root: Path,
    devices: list[str],
    state: dict[str, Any],
    state_path: Path,
) -> None:
    prediction_root.mkdir(parents=True, exist_ok=True)
    pending: deque[tuple[int, dict[str, Any]]] = deque()
    inference = dict(state.get("inference", {}))
    for line, row in enumerate(rows, start=1):
        case_id = str(row["case_id"])
        cache = prediction_root / f"{case_id}.npz"
        runtime = prediction_root / f"{case_id}.runtime.json"
        if valid_cache(cache, runtime, case_id) or link_smoke_cache(prediction_root, case_id):
            inference[case_id] = {"status": "complete", "cache_reused": True}
        else:
            pending.append((line, row))
    active: dict[str, tuple[subprocess.Popen[Any], Any, str, float, list[str]]] = {}
    failures = []
    while pending or active:
        while pending and len(active) < len(devices):
            free_device = next(value for value in devices if value not in active)
            line, row = pending.popleft()
            case_id = str(row["case_id"])
            output = prediction_root / f"{case_id}.npz"
            command = [
                sys.executable,
                str(RUN_CASE),
                "--manifest",
                str(manifest),
                "--line",
                str(line),
                "--extracted-root",
                str(extracted_root),
                "--output",
                str(output),
                "--device",
                free_device,
            ]
            log = log_root / f"{case_id}.inference.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("w", encoding="utf-8")
            handle.write("COMMAND " + json.dumps(command, ensure_ascii=False) + "\n")
            handle.flush()
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env={**os.environ, "TMPDIR": str(WORK / "tmp")},
            )
            active[free_device] = (process, handle, case_id, time.perf_counter(), command)
            inference[case_id] = {"status": "running", "device": free_device, "pid": process.pid}
            state["inference"] = inference
            atomic_json(state_path, state)
        if not active:
            continue
        time.sleep(1.0)
        for device, (process, handle, case_id, started, command) in list(active.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            handle.close()
            cache = prediction_root / f"{case_id}.npz"
            runtime = prediction_root / f"{case_id}.runtime.json"
            ok = returncode == 0 and valid_cache(cache, runtime, case_id)
            inference[case_id] = {
                "status": "complete" if ok else "error",
                "device": device,
                "returncode": returncode,
                "seconds": time.perf_counter() - started,
                "command": command,
            }
            if not ok:
                failures.append({"case_id": case_id, **inference[case_id]})
            del active[device]
            state["inference"] = inference
            atomic_json(state_path, state)
    if failures:
        raise RuntimeError(f"inference failures: {failures}")


def valid_probe(path: Path, expected_cases: int) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("errors"):
        raise ValueError(f"Probe errors: {payload['errors']}")
    observed = {str(row["case_id"]) for row in payload.get("rows", [])}
    skipped = {str(row["case_id"]) for row in payload.get("skipped_cases", [])}
    if len(observed | skipped) != expected_cases:
        raise ValueError(f"Probe covers {len(observed | skipped)} cases, expected {expected_cases}")
    complete_rows = [row for row in payload.get("rows", []) if row.get("status") == "complete"]
    for row in complete_rows:
        missing = REQUIRED_METRICS - set(row.get("metrics", {}))
        if missing:
            raise ValueError(
                f"Missing metric fields in {row['case_id']}:{row['candidate']}: {missing}"
            )
        if str(row.get("candidate")) == "v16_0_m15_geometry":
            invalid = [
                key for key in REQUIRED_METRICS - {"Seam_root_m"}
                if row.get("metrics", {}).get(key) is None
                or not np.isfinite(float(row["metrics"][key]))
            ]
            if invalid:
                raise ValueError(
                    f"Parent has invalid metrics in {row['case_id']}: {invalid}"
                )
    return payload


def safe_remove(path: Path, allowed_parent: Path) -> bool:
    if not path.exists():
        return False
    resolved = path.resolve()
    parent = allowed_parent.resolve()
    if resolved == parent or parent not in resolved.parents:
        raise ValueError(f"Refusing cleanup outside {parent}: {resolved}")
    if resolved.is_dir():
        shutil.rmtree(resolved)
    else:
        resolved.unlink()
    return True


def process_entry(
    row: dict[str, Any],
    split: str,
    devices: list[str],
    candidates: list[Path | None],
    reserve_gib: float,
    keep_staging: bool,
    continue_on_structural_error: bool,
) -> dict[str, Any]:
    entry = str(row["entry"])
    token = slug(entry)
    split_root = OUTPUT / split
    meta = split_root / "captures" / token
    logs = split_root / "logs" / token
    prediction_root = split_root / "predictions" / token
    state_path = meta / "state.json"
    audit = meta / "audit.json"
    ledger = meta / "stage_ledger.json"
    manifest = meta / "manifest.jsonl"
    archive_path, stage_root, capture_root = safe_stage_path(WORK, entry)
    expected_reports = {
        report_tag(candidate): meta / f"{report_tag(candidate)}.json" for candidate in candidates
    }
    if state_path.is_file():
        previous = read_json(state_path)
        if previous.get("status") == "complete" and all(path.is_file() for path in expected_reports.values()):
            for path in expected_reports.values():
                valid_probe(path, int(previous["manifest_cases"]))
            return previous
    state: dict[str, Any] = {
        "schema_version": "Movie3R-v19-EgoHumans-capture-state-v1",
        "protocol": "Movie3R-EgoHumans-CS100-v1",
        "split": split,
        "entry": entry,
        "action": row["action"],
        "status": "started",
        "started_at": time.time(),
        "devices": devices,
        "candidate_reports": {key: str(value.resolve()) for key, value in expected_reports.items()},
    }
    atomic_json(state_path, state)
    stage_command = [
        sys.executable,
        str(STAGER),
        "--outer",
        str(OUTER),
        "--entry",
        entry,
        "--work-root",
        str(WORK),
        "--audit-output",
        str(audit),
        "--ledger-output",
        str(ledger),
        "--reserve-gib",
        str(reserve_gib),
    ]
    returncode, seconds = run(stage_command, logs / "stage.log")
    state["stage_seconds"] = seconds
    if returncode:
        state.update(status="structural_error", returncode=returncode, completed_at=time.time())
        atomic_json(state_path, state)
        if continue_on_structural_error:
            safe_remove(stage_root, WORK / "staging")
            safe_remove(archive_path, WORK / "archives")
            return state
        raise RuntimeError(f"stage/audit failed for {entry}; see {logs / 'stage.log'}")
    manifest_command = [
        sys.executable,
        str(BUILDER),
        "--audits",
        str(audit),
        "--split",
        split,
        "--output",
        str(manifest),
    ]
    returncode, seconds = run(manifest_command, logs / "manifest.log")
    state["manifest_seconds"] = seconds
    if returncode:
        state.update(status="manifest_error", returncode=returncode)
        atomic_json(state_path, state)
        raise RuntimeError(f"manifest failed for {entry}")
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 4:
        raise ValueError(f"Frozen CS100 protocol requires four angle cases, got {len(rows)} for {entry}")
    state.update(
        manifest=str(manifest.resolve()),
        manifest_sha256=sha256(manifest),
        manifest_cases=len(rows),
        capture_root=str(capture_root.resolve()),
    )
    atomic_json(state_path, state)
    launch_inference(rows, manifest, stage_root, prediction_root, logs, devices, state, state_path)
    reports = {}
    for candidate in candidates:
        tag = report_tag(candidate)
        report = expected_reports[tag]
        if report.is_file():
            try:
                payload = valid_probe(report, len(rows))
                reports[tag] = {
                    "status": "reused",
                    "complete_cases": payload.get("complete_case_count"),
                }
                continue
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                report.unlink()
        command = [
            sys.executable,
            str(probe_for_candidate(candidate)),
            "--prediction-roots",
            str(prediction_root),
            "--extracted-root",
            str(stage_root),
            "--output",
            str(report),
            "--reference-methods",
            *REFERENCES,
        ]
        if candidate is not None:
            command.extend(("--candidate-json", str(candidate)))
        returncode, seconds = run(command, logs / f"{tag}.probe.log")
        if returncode:
            state.update(status="probe_error", failed_report=tag, returncode=returncode)
            atomic_json(state_path, state)
            raise RuntimeError(f"probe failed for {entry}:{tag}")
        payload = valid_probe(report, len(rows))
        reports[tag] = {
            "status": "complete",
            "seconds": seconds,
            "complete_cases": payload.get("complete_case_count"),
            "skipped_cases": len(payload.get("skipped_cases", [])),
        }
        state["reports"] = reports
        atomic_json(state_path, state)
    removed = {"staging": False, "inner_archive": False}
    if not keep_staging:
        removed["staging"] = safe_remove(stage_root, WORK / "staging")
        removed["inner_archive"] = safe_remove(archive_path, WORK / "archives")
    state.update(
        status="complete",
        completed_at=time.time(),
        reports=reports,
        temporary_removed=removed,
        staging_kept=bool(keep_staging),
    )
    atomic_json(state_path, state)
    return state


def main() -> None:
    args = parse_args()
    if args.include_exploration and args.split != "development":
        raise ValueError("The finite exploration grid is forbidden outside development")
    candidates: list[Path | None] = []
    if args.include_exploration:
        candidates.append(None)
    for path in args.candidate_json:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        candidates.append(resolved)
    if not candidates:
        raise ValueError("Provide --include-exploration and/or --candidate-json")
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    if not devices:
        raise ValueError("At least one device is required")
    entries = selected_entries(args.split, args.entries)
    split_state = OUTPUT / args.split / "protocol_state.json"
    summary = {
        "schema_version": "Movie3R-v19-EgoHumans-split-state-v1",
        "protocol": "Movie3R-EgoHumans-CS100-v1",
        "split": args.split,
        "outer_index": str(INDEX.resolve()),
        "outer_index_sha256": sha256(INDEX),
        "entries": [row["entry"] for row in entries],
        "devices": devices,
        "candidate_sources": ["finite_exploration_grid" if path is None else str(path) for path in candidates],
        "status": "running",
        "captures": {},
        "started_at": time.time(),
    }
    if split_state.is_file():
        previous = read_json(split_state)
        if previous.get("outer_index_sha256") == summary["outer_index_sha256"]:
            summary["started_at"] = previous.get("started_at", summary["started_at"])
            summary["captures"] = previous.get("captures", {})
    atomic_json(split_state, summary)
    fatal = []
    for index, row in enumerate(entries, start=1):
        entry = str(row["entry"])
        print(f">> [{index}/{len(entries)}] {args.split}: {entry}", flush=True)
        try:
            result = process_entry(
                row,
                args.split,
                devices,
                candidates,
                float(args.reserve_gib),
                bool(args.keep_staging),
                bool(args.continue_on_structural_error),
            )
            summary["captures"][entry] = {
                "status": result["status"],
                "state": str((OUTPUT / args.split / "captures" / slug(entry) / "state.json").resolve()),
            }
        except Exception as error:
            summary["captures"][entry] = {
                "status": "fatal_error",
                "error": f"{type(error).__name__}: {error}",
            }
            fatal.append({"entry": entry, **summary["captures"][entry]})
            atomic_json(split_state, summary)
            raise
        atomic_json(split_state, summary)
    statuses = [value["status"] for value in summary["captures"].values()]
    summary.update(
        status="complete" if not fatal else "error",
        completed_at=time.time(),
        complete_captures=sum(value == "complete" for value in statuses),
        structural_exclusions=sum(value == "structural_error" for value in statuses),
        fatal_errors=fatal,
    )
    atomic_json(split_state, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
