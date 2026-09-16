#!/usr/bin/env python3
"""Disk-bounded orchestrator for the EgoHumans Human3R state-isolation run.

The immutable formal90 continuous-state cache is never modified.  For each
capture, this script stages the corresponding EgoHumans inner archive, runs
the exact original Human3R checkpoint on only the 50 post-cut RGB frames from
fresh state, and removes the expanded capture after both GPU shards finish.
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
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
STAGER = REPO_ROOT / "versions/v19/egohumans/stage_capture.py"
RUNNER = Path(__file__).with_name("run_egohumans_original_human3r_reset.py")
DEFAULT_FORMAL = (
    REPO_ROOT.parent
    / "ICLR-paper/bridge3r_iclr2027/private_audit/egohumans_formal90_manifest.jsonl"
)
DEFAULT_SOURCE = (
    REPO_ROOT / "output/bridge3r_egohumans_ablation_v1/formal90_native/test/predictions"
)
DEFAULT_OUTPUT = REPO_ROOT / "output/egohumans_state_isolation_formal90/original_reset"
DEFAULT_WORK = REPO_ROOT.parent / "data/EgoHuman_state_isolation_work"
DEFAULT_OUTER = REPO_ROOT.parent / "data/EgoHuman.zip"
SCHEMA = "Shot3R-EgoHumans-state-isolation-orchestrator-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer", type=Path, default=DEFAULT_OUTER)
    parser.add_argument("--formal-manifest", type=Path, default=DEFAULT_FORMAL)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--reserve-gib", type=float, default=50.0)
    parser.add_argument("--entries", nargs="*", help="Optional exact archive entries for a smoke test.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def slug(entry: str) -> str:
    import re

    value = (entry[:-7] if entry.endswith(".tar.gz") else entry).replace("/", "__")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def capture_name(entry: str) -> str:
    import re

    name = Path(entry).name
    stem = name[:-7] if name.endswith(".tar.gz") else name
    return re.sub(r"-\d{3}$", "", stem)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, path)


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def source_by_case(source_root: Path) -> dict[str, Path]:
    output: dict[str, Path] = {}
    paths = sorted(source_root.glob("*/*.npz"))
    if len(paths) != 90:
        raise ValueError(f"formal90 source must contain 90 caches, found {len(paths)}")
    for path in paths:
        runtime = json.loads(path.with_suffix(".runtime.json").read_text(encoding="utf-8"))
        case_id = str(runtime["record"]["case_id"])
        if case_id in output:
            raise ValueError(f"duplicate source case: {case_id}")
        output[case_id] = path
    return output


def output_paths(source: Path, source_root: Path, output_root: Path) -> tuple[Path, Path]:
    output = output_root / source.relative_to(source_root)
    return output, output.with_suffix(".runtime.json")


def entry_looks_complete(
    rows: list[dict[str, Any]], sources: dict[str, Path], source_root: Path, output_root: Path
) -> bool:
    for row in rows:
        output, runtime = output_paths(sources[str(row["case_id"])], source_root, output_root)
        if not output.is_file() or not runtime.is_file():
            return False
        try:
            record = json.loads(runtime.read_text(encoding="utf-8"))
            if (
                record.get("case_id") != row["case_id"]
                or record.get("archive_entry") != row["archive_entry"]
                or record.get("output_sha256") != sha256(output)
            ):
                return False
        except (OSError, KeyError, json.JSONDecodeError):
            return False
    return True


def run_logged(command: list[str], log: Path) -> subprocess.Popen[str]:
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("w", encoding="utf-8")
    handle.write("COMMAND " + json.dumps(command, ensure_ascii=False) + "\n")
    handle.flush()
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, "TMPDIR": str(log.parent.parent / "tmp")},
    )
    process._shot3r_log_handle = handle  # type: ignore[attr-defined]
    return process


def wait_logged(process: subprocess.Popen[str]) -> int:
    code = process.wait()
    process._shot3r_log_handle.close()  # type: ignore[attr-defined]
    return int(code)


def safe_remove(path: Path, allowed_parent: Path, directory: bool) -> None:
    if not path.exists():
        return
    resolved = path.resolve()
    parent = allowed_parent.resolve()
    if parent not in resolved.parents or resolved == parent:
        raise ValueError(f"unsafe cleanup target: {resolved}")
    if directory:
        shutil.rmtree(resolved)
    else:
        resolved.unlink()


def main() -> None:
    options = parse_args()
    outer = options.outer.resolve(strict=True)
    formal = options.formal_manifest.resolve(strict=True)
    source_root = options.source_root.resolve(strict=True)
    output_root = options.output_root.resolve()
    work_root = options.work_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    devices = [value.strip() for value in options.devices.split(",") if value.strip()]
    if not devices:
        raise ValueError("at least one CUDA device is required")
    rows = read_rows(formal)
    if len(rows) != 90 or len({row["case_id"] for row in rows}) != 90:
        raise ValueError("the frozen formal manifest must contain 90 unique cases")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["archive_entry"])].append(row)
    if len(grouped) != 27:
        raise ValueError(f"formal90 must contain 27 captures, found {len(grouped)}")
    if options.entries:
        requested = set(options.entries)
        unknown = sorted(requested - set(grouped))
        if unknown:
            raise ValueError(f"requested entries are not in formal90: {unknown}")
        grouped = {key: value for key, value in grouped.items() if key in requested}

    sources = source_by_case(source_root)
    if set(sources) != {str(row["case_id"]) for row in rows}:
        raise ValueError("source caches do not exactly match the frozen formal90 case IDs")
    with zipfile.ZipFile(outer) as archive:
        sizes = {entry: int(archive.getinfo(entry).file_size) for entry in grouped}
    entries = sorted(grouped, key=lambda entry: (sizes[entry], entry))

    state_path = work_root / "orchestrator_state.json"
    state: dict[str, Any] = {
        "schema_version": SCHEMA,
        "outer": str(outer),
        "formal_manifest": str(formal),
        "formal_manifest_sha256": sha256(formal),
        "source_root": str(source_root),
        "output_root": str(output_root),
        "devices": devices,
        "selected_entries": entries,
        "selected_case_count": sum(len(grouped[entry]) for entry in entries),
        "status": "running",
        "started_at": time.time(),
        "captures": {},
    }
    atomic_json(state_path, state)

    for ordinal, entry in enumerate(entries, start=1):
        entry_rows = grouped[entry]
        entry_slug = slug(entry)
        if not options.overwrite and entry_looks_complete(
            entry_rows, sources, source_root, output_root
        ):
            state["captures"][entry] = {"status": "cached", "case_count": len(entry_rows)}
            atomic_json(state_path, state)
            print(f"[{ordinal}/{len(entries)}] cached {entry}", flush=True)
            continue

        archive_path = work_root / "archives" / f"{entry_slug}.tar.gz"
        stage_root = work_root / "staging" / entry_slug
        capture_root = stage_root / capture_name(entry)
        audit = work_root / "audits" / f"{entry_slug}.json"
        ledger = work_root / "ledgers" / f"{entry_slug}.json"
        logs = work_root / "logs" / entry_slug
        capture_state: dict[str, Any] = {
            "status": "staging",
            "case_count": len(entry_rows),
            "inner_tar_bytes": sizes[entry],
            "started_at": time.time(),
        }
        state["captures"][entry] = capture_state
        atomic_json(state_path, state)
        print(f"[{ordinal}/{len(entries)}] staging {entry} ({sizes[entry] / 2**30:.1f} GiB)", flush=True)
        stage_command = [
            sys.executable,
            str(STAGER),
            "--outer",
            str(outer),
            "--entry",
            entry,
            "--work-root",
            str(work_root),
            "--audit-output",
            str(audit),
            "--ledger-output",
            str(ledger),
            "--reserve-gib",
            str(options.reserve_gib),
        ]
        stage_process = run_logged(stage_command, logs / "stage.log")
        if wait_logged(stage_process):
            capture_state.update(status="stage_failed", completed_at=time.time())
            state["status"] = "failed"
            atomic_json(state_path, state)
            raise RuntimeError(f"staging failed for {entry}; see {logs / 'stage.log'}")
        if not capture_root.is_dir():
            raise FileNotFoundError(capture_root)

        capture_state["status"] = "inference"
        atomic_json(state_path, state)
        processes = []
        for shard, device in enumerate(devices):
            command = [
                sys.executable,
                str(RUNNER),
                "--source-root",
                str(source_root),
                "--output-root",
                str(output_root),
                "--archive-entry",
                entry,
                "--extracted-root",
                str(stage_root),
                "--device",
                device,
                "--shard-index",
                str(shard),
                "--num-shards",
                str(len(devices)),
            ]
            if options.overwrite:
                command.append("--overwrite")
            processes.append((device, run_logged(command, logs / f"inference_{device.replace(':', '_')}.log")))
        failures = [(device, wait_logged(process)) for device, process in processes]
        failures = [(device, code) for device, code in failures if code]
        if failures:
            capture_state.update(status="inference_failed", failures=failures, completed_at=time.time())
            state["status"] = "failed"
            atomic_json(state_path, state)
            raise RuntimeError(f"inference failed for {entry}: {failures}; staging retained")
        if not entry_looks_complete(entry_rows, sources, source_root, output_root):
            raise RuntimeError(f"inference returned success but outputs are incomplete for {entry}")

        safe_remove(stage_root, work_root / "staging", directory=True)
        safe_remove(archive_path, work_root / "archives", directory=False)
        capture_state.update(status="complete", completed_at=time.time(), temporary_removed=True)
        atomic_json(state_path, state)
        print(f"[{ordinal}/{len(entries)}] complete {entry}; temporary data removed", flush=True)

    complete_cases = sum(
        len(entry_rows)
        for entry, entry_rows in grouped.items()
        if entry_looks_complete(entry_rows, sources, source_root, output_root)
    )
    state.update(status="complete", completed_at=time.time(), complete_case_count=complete_cases)
    atomic_json(state_path, state)
    print(json.dumps({"status": "complete", "cases": complete_cases, "captures": len(entries)}))


if __name__ == "__main__":
    main()
