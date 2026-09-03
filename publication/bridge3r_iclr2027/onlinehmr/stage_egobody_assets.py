#!/usr/bin/env python3
"""Materialize only evaluator assets required by the frozen EgoBody Test protocol."""

from __future__ import annotations

import argparse
import binascii
import hashlib
import json
import os
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA = "Bridge3R-OnlineHMR-EgoBody-assets-v1"
COMMON = (
    "data_info_release.csv",
    "data_splits.csv",
    "calibrations.zip",
    "kinect_cam_params.zip",
)
SPLIT_ASSETS = {
    "test": ("smplx_interactee_test.zip", "smplx_camera_wearer_test.zip"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def crc32(path: Path) -> int:
    value = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            value = binascii.crc32(block, value)
    return value & 0xFFFFFFFF


def nearest_existing(path: Path) -> Path:
    current = path.absolute()
    while not current.exists():
        if current.parent == current:
            raise FileNotFoundError(f"no existing ancestor for {path}")
        current = current.parent
    return current


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def safe_relative(name: str) -> Path:
    value = PurePosixPath(name)
    if value.is_absolute() or not value.parts or ".." in value.parts:
        raise ValueError(f"unsafe ZIP member {name!r}")
    return Path(*value.parts)


def unique_info(archive: zipfile.ZipFile, basename: str) -> zipfile.ZipInfo:
    matches = [
        info for info in archive.infolist()
        if not info.is_dir() and PurePosixPath(info.filename).name == basename
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one outer member named {basename!r}, found {len(matches)}")
    info = matches[0]
    if info.flag_bits & 0x1:
        raise ValueError(f"encrypted member is unsupported: {info.filename}")
    return info


def copy_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    try:
        with archive.open(info, "r") as source, partial.open("wb") as destination:
            shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
            destination.flush()
            os.fsync(destination.fileno())
        if partial.stat().st_size != info.file_size:
            raise ValueError(f"size mismatch while extracting {info.filename}")
        os.replace(partial, target)
    finally:
        if partial.exists():
            partial.unlink()


def expand_zip(source: Path, output: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    root = output.resolve()
    with zipfile.ZipFile(source) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            relative = safe_relative(info.filename)
            target = (output / relative).resolve()
            if root not in target.parents:
                raise ValueError(f"nested ZIP member escapes output root: {info.filename}")
            unix_mode = (info.external_attr >> 16) & 0xFFFF
            if unix_mode and stat.S_ISLNK(unix_mode):
                raise ValueError(f"symlink member is unsupported: {info.filename}")
            if target.is_file():
                if target.stat().st_size != info.file_size or crc32(target) != info.CRC:
                    raise ValueError(f"existing expanded asset differs: {target}")
            else:
                copy_member(archive, info, target)
            records.append({
                "member": info.filename,
                "path": str(target),
                "bytes": int(info.file_size),
                "crc32": f"{info.CRC:08x}",
            })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--official-split", choices=tuple(SPLIT_ASSETS), default="test")
    parser.add_argument("--reserve-gib", type=float, default=50.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    outer = args.outer.resolve()
    output = args.output_root.resolve()
    if not outer.is_file():
        raise FileNotFoundError(outer)
    required = COMMON + SPLIT_ASSETS[args.official_split]
    with zipfile.ZipFile(outer) as archive:
        infos = {name: unique_info(archive, name) for name in required}
        total_asset_bytes = sum(info.file_size for info in infos.values())
        required_bytes = sum(
            info.file_size for name, info in infos.items()
            if not (output / name).is_file()
        )
        free = shutil.disk_usage(nearest_existing(output)).free
        reserve = int(args.reserve_gib * 1024**3)
        if free - reserve < required_bytes:
            raise OSError(
                f"insufficient disk: free={free}, reserve={reserve}, "
                f"required={required_bytes}"
            )
        summary = {
            "schema_version": SCHEMA,
            "status": "dry_run" if args.dry_run else "running",
            "outer": str(outer),
            "outer_size_bytes": outer.stat().st_size,
            "outer_mtime_ns": outer.stat().st_mtime_ns,
            "official_split": args.official_split,
            "total_asset_bytes": total_asset_bytes,
            "required_uncompressed_bytes": required_bytes,
            "free_bytes_preflight": free,
            "reserve_bytes": reserve,
            "assets": [],
        }
        if args.dry_run:
            summary["assets"] = [
                {
                    "name": name,
                    "outer_member": infos[name].filename,
                    "bytes": infos[name].file_size,
                    "crc32": f"{infos[name].CRC:08x}",
                }
                for name in required
            ]
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            return
        output.mkdir(parents=True, exist_ok=True)
        ledger = output / "assets.provenance.json"
        previous = json.loads(ledger.read_text(encoding="utf-8")) if ledger.is_file() else None
        for name in required:
            info = infos[name]
            target = output / name
            old = next(
                (row for row in (previous or {}).get("assets", []) if row.get("name") == name),
                None,
            )
            if target.is_file() and old is not None:
                if (
                    previous.get("outer_size_bytes") != outer.stat().st_size
                    or previous.get("outer_mtime_ns") != outer.stat().st_mtime_ns
                    or target.stat().st_size != old.get("bytes")
                    or sha256(target) != old.get("sha256")
                ):
                    raise ValueError(f"existing asset provenance differs: {target}")
            elif target.is_file():
                if target.stat().st_size != info.file_size or crc32(target) != info.CRC:
                    raise ValueError(f"unproven existing asset differs from ZIP entry: {target}")
            else:
                copy_member(archive, info, target)
            summary["assets"].append({
                "name": name,
                "outer_member": info.filename,
                "path": str(target.resolve()),
                "bytes": target.stat().st_size,
                "sha256": sha256(target),
                "crc32": f"{info.CRC:08x}",
            })
            atomic_json(ledger, summary)

    expanded = output / "expanded"
    calibration_rows = expand_zip(output / "calibrations.zip", expanded)
    parameter_rows = expand_zip(output / "kinect_cam_params.zip", expanded)
    summary.update({
        "status": "complete",
        "expanded_root": str(expanded),
        "expanded_calibration_files": len(calibration_rows),
        "expanded_camera_parameter_files": len(parameter_rows),
    })
    atomic_json(output / "assets.provenance.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
