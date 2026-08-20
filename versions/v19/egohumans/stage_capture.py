#!/usr/bin/env python3
"""Disk-bounded extraction and GT-only audit of one EgoHumans capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from versions.v19.egohumans.dataset import atomic_json, audit_capture


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer", type=Path, required=True)
    parser.add_argument("--entry", required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--ledger-output", type=Path, required=True)
    parser.add_argument("--reserve-gib", type=float, default=80.0)
    return parser.parse_args()


def slug(entry: str) -> str:
    value = (entry[:-7] if entry.endswith(".tar.gz") else entry).replace("/", "__")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def capture_name(entry: str) -> str:
    name = Path(entry).name
    stem = name[:-7] if name.endswith(".tar.gz") else name
    return re.sub(r"-\d{3}$", "", stem)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def safe_stage_path(work_root: Path, entry: str) -> tuple[Path, Path, Path]:
    token = slug(entry)
    archive_path = work_root / "archives" / f"{token}.tar.gz"
    stage_root = work_root / "staging" / token
    root = stage_root / capture_name(entry)
    return archive_path, stage_root, root


def reusable(
    args: argparse.Namespace, archive_path: Path, capture_root: Path
) -> bool:
    if not args.audit_output.is_file() or not args.ledger_output.is_file():
        return False
    try:
        audit = json.loads(args.audit_output.read_text(encoding="utf-8"))
        ledger = json.loads(args.ledger_output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        ledger.get("status") == "staged_and_audited"
        and ledger.get("entry") == args.entry
        and audit.get("archive_entry") == args.entry
        and capture_root.is_dir()
        and archive_path.is_file()
    )


def extract_inner(outer: Path, entry: str, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    if partial.exists():
        partial.unlink()
    with zipfile.ZipFile(outer) as archive:
        info = archive.getinfo(entry)
        with archive.open(info) as source, partial.open("wb") as target:
            shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
    if partial.stat().st_size != info.file_size:
        raise ValueError(
            f"Inner tar size mismatch: {partial.stat().st_size} vs {info.file_size}"
        )
    partial.replace(destination)
    return {
        "outer_crc32": f"{info.CRC:08x}",
        "inner_tar_size_bytes": int(info.file_size),
        "inner_tar_sha256": sha256(destination),
    }


def extract_tar(archive_path: Path, stage_root: Path, expected_root: Path, tmp: Path) -> None:
    if stage_root.exists():
        resolved = stage_root.resolve()
        allowed = stage_root.parent.resolve()
        if allowed not in resolved.parents or resolved == allowed:
            raise ValueError(f"Unsafe staging cleanup target: {resolved}")
        shutil.rmtree(resolved)
    stage_root.mkdir(parents=True)
    tmp.mkdir(parents=True, exist_ok=True)
    command = [
        "tar",
        "-xzf",
        str(archive_path),
        "-C",
        str(stage_root),
        "--strip-components=8",
        "--no-same-owner",
        "--no-same-permissions",
    ]
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        env={**os.environ, "TMPDIR": str(tmp)},
    )
    if completed.returncode:
        raise RuntimeError(
            f"tar extraction failed ({completed.returncode}): {completed.stderr[-2000:]}"
        )
    if not expected_root.is_dir():
        roots = sorted(path.name for path in stage_root.iterdir())
        raise FileNotFoundError(f"Expected {expected_root}; extracted roots={roots}")
    links = list(expected_root.rglob("*"))
    unsafe_links = [path for path in links if path.is_symlink()]
    if unsafe_links:
        raise ValueError(f"Archive contains symbolic links: {unsafe_links[:3]}")
    required = (
        expected_root / "exo",
        expected_root / "processed_data/smpl",
        expected_root / "colmap/workplace/cameras.txt",
        expected_root / "colmap/workplace/images.txt",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Staged capture misses required paths: {missing}")


def main() -> None:
    args = parse_args()
    args.work_root.mkdir(parents=True, exist_ok=True)
    archive_path, stage_root, capture_root = safe_stage_path(args.work_root, args.entry)
    if reusable(args, archive_path, capture_root):
        print(
            json.dumps(
                {
                    "status": "reused",
                    "entry": args.entry,
                    "archive": str(archive_path),
                    "stage_root": str(stage_root),
                    "capture_root": str(capture_root),
                    "audit": str(args.audit_output),
                },
                indent=2,
            )
        )
        return
    with zipfile.ZipFile(args.outer) as outer:
        info = outer.getinfo(args.entry)
    usage = shutil.disk_usage(args.work_root)
    reserve = float(args.reserve_gib) * 1024**3
    conservative_required = 2.5 * float(info.file_size)
    if usage.free - reserve < conservative_required:
        raise OSError(
            f"Insufficient free space: free={usage.free}, reserve={reserve}, "
            f"conservative_required={conservative_required}"
        )
    metadata = extract_inner(args.outer, args.entry, archive_path)
    extract_tar(archive_path, stage_root, capture_root, args.work_root / "tmp")
    audit = audit_capture(
        capture_root,
        args.entry,
        capture_relative=capture_root.name,
    )
    atomic_json(args.audit_output, audit)
    ledger = {
        "schema_version": "Movie3R-v19-EgoHumans-stage-ledger-v1",
        "status": "staged_and_audited",
        "entry": args.entry,
        "outer": str(args.outer.resolve()),
        "archive": str(archive_path.resolve()),
        "stage_root": str(stage_root.resolve()),
        "capture_root": str(capture_root.resolve()),
        "capture_relative": capture_root.name,
        "audit": str(args.audit_output.resolve()),
        "disk_free_before_bytes": usage.free,
        "reserve_gib": float(args.reserve_gib),
        **metadata,
    }
    atomic_json(args.ledger_output, ledger)
    print(json.dumps(ledger, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
