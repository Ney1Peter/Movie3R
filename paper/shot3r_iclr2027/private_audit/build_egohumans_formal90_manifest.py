#!/usr/bin/env python3
"""Materialize and validate the private immutable EgoHumans formal-90 manifest.

The public paper reports only the fixed N=90 protocol.  This helper stays in
``private_audit`` because it contains archive locations, checksums, and other
operational provenance that must not enter an anonymous submission package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any


AUDIT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = AUDIT_ROOT.parents[2]
DEFAULT_IDS = AUDIT_ROOT / "egohumans_cs100_formal_case_ids.txt"
DEFAULT_RUNTIME = WORKSPACE_ROOT / "Movie3R/output/v19_egohumans/test/captures"
DEFAULT_ARCHIVE = WORKSPACE_ROOT / "data/EgoHuman.zip"
DEFAULT_OUTPUT = AUDIT_ROOT / "egohumans_formal90_manifest.jsonl"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_ids(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(ids) != 90:
        raise ValueError(f"formal protocol requires exactly 90 IDs, found {len(ids)}")
    if len(ids) != len(set(ids)):
        raise ValueError("formal case-ID file has duplicates")
    return ids


def parse_test_manifests(root: Path) -> dict[str, tuple[dict[str, Any], Path]]:
    rows: dict[str, tuple[dict[str, Any], Path]] = {}
    manifests = sorted(root.glob("*/manifest.jsonl"))
    if not manifests:
        raise FileNotFoundError(f"no per-capture Test manifests under {root}")
    for path in manifests:
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            case_id = str(row.get("case_id", ""))
            if not case_id:
                raise ValueError(f"missing case ID: {path}:{line_no}")
            if row.get("split") != "test":
                raise ValueError(f"non-Test row in Test manifest: {path}:{line_no}")
            if case_id in rows:
                raise ValueError(f"duplicate Test case ID {case_id}: {rows[case_id][1]} and {path}")
            rows[case_id] = (row, path)
    return rows


def build(ids: list[str], runtime_root: Path, archive: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_rows = parse_test_manifests(runtime_root)
    missing = [case_id for case_id in ids if case_id not in source_rows]
    if missing:
        raise ValueError(f"{len(missing)} formal cases missing from Test manifests: {missing[:5]}")
    if not archive.is_file():
        raise FileNotFoundError(archive)

    archive_infos: dict[str, zipfile.ZipInfo] = {}
    with zipfile.ZipFile(archive) as payload:
        for info in payload.infolist():
            if info.filename.endswith(".tar.gz"):
                archive_infos[info.filename] = info

    records: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    for case_id in ids:
        row, source = source_rows[case_id]
        archive_entry = str(row["archive_entry"])
        info = archive_infos.get(archive_entry)
        if info is None:
            raise ValueError(f"formal case {case_id} refers to missing archive entry {archive_entry}")
        source_key = str(source.resolve())
        source_hashes.setdefault(source_key, sha256(source))
        record = dict(row)
        record.update(
            {
                "formal_protocol": "Bridge3R-EgoHumans-formal90-v1",
                "manifest_source": source_key,
                "manifest_source_sha256": source_hashes[source_key],
                "archive_file_size": int(info.file_size),
                "archive_crc32": f"{int(info.CRC):08x}",
            }
        )
        if record.get("gt_available_to_runtime") is not False:
            raise ValueError(f"GT runtime boundary violated by {case_id}")
        if record.get("selection_depends_on_model_result") is not False:
            raise ValueError(f"selection boundary violated by {case_id}")
        records.append(record)

    required_archives = sorted({str(row["archive_entry"]) for row in records})
    angles = [float(row["camera_rotation_span_deg_evaluator_only"]) for row in records]
    stratum_counts = {
        stratum: sum(row.get("angle_stratum") == stratum for row in records)
        for stratum in ("small", "medium", "large", "extreme")
    }
    spec = {
        "schema_version": "Bridge3R-EgoHumans-formal90-manifest-spec-v1",
        "protocol": "Bridge3R-EgoHumans-formal90-v1",
        "case_count": len(records),
        "capture_count": len({(str(row["sequence"]), str(row["capture"])) for row in records}),
        "archive_entry_count": len(required_archives),
        "archive_entry_names": required_archives,
        "angle_range_deg": [min(angles), max(angles)],
        "angle_stratum_counts": stratum_counts,
        "ge150_case_count": sum(angle >= 150.0 for angle in angles),
        "case_id_file": str(DEFAULT_IDS.resolve()),
        "case_id_file_sha256": sha256(DEFAULT_IDS),
        "outer_archive": str(archive.resolve()),
        "outer_archive_size": archive.stat().st_size,
        "source_manifest_count": len(source_hashes),
        "source_manifests": source_hashes,
        "guarantees": {
            "split": "test",
            "gt_available_to_runtime": False,
            "selection_depends_on_model_result": False,
        },
    }
    return records, spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-ids", type=Path, default=DEFAULT_IDS)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    ids = read_ids(args.case_ids.resolve())
    records, spec = build(ids, args.runtime_root.resolve(), args.archive.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    output_hash = sha256(args.output)
    spec.update({"manifest": str(args.output.resolve()), "manifest_sha256": output_hash})
    spec_path = args.output.with_suffix(".spec.json")
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(spec, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
