#!/usr/bin/env python3
"""Safely stage, audit, and freeze one nested Harmony4D archive.

The outer archive remains read-only.  The script is resumable, validates the
published byte size and SHA-256, performs a full inner ZIP CRC test, applies a
free-space gate, and selects the first coordinate-valid capture in a frozen
SHA-256 order.  It performs no Movie3R forward and reads no test result.
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
from pathlib import Path, PurePosixPath
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
INDEXER = Path(__file__).resolve().with_name("index_archive.py")
AUDITOR = Path(__file__).resolve().with_name("audit_sequence.py")
BUILDER = Path(__file__).resolve().with_name("build_manifest.py")
DEFAULT_RESERVE_GIB = 80.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer", type=Path, required=True)
    parser.add_argument("--entry", required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--index-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--ledger-output", type=Path, required=True)
    parser.add_argument("--reserve-gib", type=float, default=DEFAULT_RESERVE_GIB)
    parser.add_argument("--overlay-cameras", default="cam01,cam08,cam15")
    return parser.parse_args()


def digest(path: Path, chunk: int = 16 * 1024 * 1024) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(partial, path)


def run(command: list[str], log: Path) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command, cwd=REPO_ROOT, text=True, capture_output=True, env={
            **os.environ,
            "TMPDIR": "/data/wangzheng/iJCV-CODE/data/Harmony4D_work/tmp",
        }
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps({
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return completed


def safe_members(inner: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = inner.infolist()
    for member in members:
        path = PurePosixPath(member.filename)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe inner ZIP member: {member.filename}")
    return members


def within(child: Path, parent: Path) -> bool:
    child, parent = child.resolve(), parent.resolve()
    return child == parent or parent in child.parents


def main() -> None:
    args = parse_args()
    started = time.time()
    outer = args.outer.resolve()
    work = args.work_root.resolve()
    entry_path = PurePosixPath(args.entry)
    if not outer.is_file():
        raise FileNotFoundError(outer)
    if entry_path.is_absolute() or ".." in entry_path.parts or entry_path.suffix != ".zip":
        raise ValueError(f"Unsafe or non-ZIP archive entry: {args.entry}")
    work.mkdir(parents=True, exist_ok=True)
    slug = "_".join(entry_path.with_suffix("").parts)
    inner_path = work / "tmp" / entry_path
    staging = work / "staging" / slug
    staging_partial = staging.with_name(staging.name + ".partial")
    for target in (inner_path, staging, staging_partial):
        if not within(target, work):
            raise ValueError(f"Resolved target escapes work root: {target}")
    state_name = ".harmony4d_download_state.json"
    with zipfile.ZipFile(outer) as archive:
        if args.entry not in archive.namelist() or state_name not in archive.namelist():
            raise KeyError(f"Missing entry/state in outer archive: {args.entry}")
        state = json.loads(archive.read(state_name).decode("utf-8-sig"))
        expected = state["files"][args.entry]
        outer_info = archive.getinfo(args.entry)
        if int(outer_info.file_size) != int(expected["size"]):
            raise ValueError("Outer central-directory size disagrees with published state")
        inner_path.parent.mkdir(parents=True, exist_ok=True)
        if not inner_path.is_file():
            partial = inner_path.with_suffix(inner_path.suffix + ".partial")
            if partial.exists():
                partial.unlink()
            with archive.open(args.entry) as source, partial.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
            os.replace(partial, inner_path)
    if inner_path.stat().st_size != int(expected["size"]):
        raise ValueError(f"Nested ZIP size mismatch: {inner_path}")
    inner_sha = digest(inner_path)
    if inner_sha != str(expected["oid"]):
        raise ValueError(f"Nested ZIP SHA-256 mismatch: {inner_sha} != {expected['oid']}")

    with zipfile.ZipFile(inner_path) as inner:
        members = safe_members(inner)
        bad = inner.testzip()
        if bad is not None:
            raise ValueError(f"CRC failure in nested ZIP member: {bad}")
        uncompressed = sum(int(member.file_size) for member in members)
        free = shutil.disk_usage(work).free
        required = uncompressed + int(args.reserve_gib * 1024**3)
        if not staging.is_dir() and free <= required:
            raise OSError(
                f"Free-space gate failed: free={free}, uncompressed={uncompressed}, "
                f"reserve={int(args.reserve_gib * 1024**3)}"
            )
        if not staging.is_dir():
            if staging_partial.exists():
                shutil.rmtree(staging_partial)
            staging_partial.mkdir(parents=True)
            inner.extractall(staging_partial)
            os.replace(staging_partial, staging)

    log_root = work / "logs" / "stage_test"
    index_result = run([
        sys.executable, str(INDEXER),
        "--extracted-root", str(staging),
        "--archive-entry", args.entry,
        "--output", str(args.index_output.resolve()),
        "--select-count", "9999",
    ], log_root / f"{slug}_index.json")
    if index_result.returncode:
        raise RuntimeError(index_result.stderr)
    index = json.loads(args.index_output.read_text(encoding="utf-8"))
    attempts = []
    selected = None
    for candidate in index["eligible_ranked"]:
        capture = str(candidate["capture_relative"])
        candidate_output = args.audit_output.with_name(
            args.audit_output.stem + "_" + Path(capture).name + args.audit_output.suffix
        )
        completed = run([
            sys.executable, str(AUDITOR),
            "--extracted-root", str(staging),
            "--archive-entry", args.entry,
            "--capture-relative", capture,
            "--output", str(candidate_output.resolve()),
            "--overlay-cameras", args.overlay_cameras,
        ], log_root / f"{slug}_audit_{Path(capture).name}.json")
        attempts.append({
            "capture_relative": capture,
            "audit_output": str(candidate_output.resolve()),
            "returncode": completed.returncode,
            "stderr_tail": completed.stderr[-2000:],
        })
        if completed.returncode == 0:
            selected = candidate_output
            break
    if selected is None:
        ledger = {
            "status": "skipped_no_coordinate_valid_capture",
            "entry": args.entry,
            "staging": str(staging),
            "attempts": attempts,
            "outer_preserved": str(outer),
        }
        atomic_json(args.ledger_output, ledger)
        raise SystemExit(3)

    build_result = run([
        sys.executable, str(BUILDER),
        "--audits", str(selected.resolve()),
        "--split", "test",
        "--output", str(args.manifest_output.resolve()),
        "--pre-count", "75",
        "--post-count", "75",
    ], log_root / f"{slug}_manifest.json")
    if build_result.returncode:
        raise RuntimeError(build_result.stderr)
    manifest_sha = hashlib.sha256(args.manifest_output.read_bytes()).hexdigest()
    spec = json.loads(args.manifest_output.with_suffix(".spec.json").read_text(encoding="utf-8"))
    if manifest_sha != spec["manifest_sha256"]:
        raise ValueError("Frozen manifest file digest mismatch")
    # The nested archive is now exactly reproducible from the preserved outer
    # archive.  Removing it recovers space while keeping the expanded capture
    # available for the later, globally frozen test pass.
    inner_path.unlink()
    ledger = {
        "status": "staged_audited_manifest_frozen",
        "entry": args.entry,
        "published_size": int(expected["size"]),
        "published_sha256": str(expected["oid"]),
        "inner_uncompressed_bytes": uncompressed,
        "inner_crc_pass": True,
        "staging": str(staging),
        "index": str(args.index_output.resolve()),
        "capture_selection_rule": (
            "first projection-valid capture in frozen SHA256 structural order; "
            "no Movie3R prediction or test metric observed"
        ),
        "selected_audit": str(selected.resolve()),
        "attempts": attempts,
        "manifest": str(args.manifest_output.resolve()),
        "manifest_sha256": manifest_sha,
        "case_count": int(spec["case_count"]),
        "outer_preserved": str(outer),
        "temporary_nested_zip_deleted": str(inner_path),
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(args.ledger_output, ledger)
    print(json.dumps(ledger, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
