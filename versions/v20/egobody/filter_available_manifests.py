#!/usr/bin/env python3
"""Filter a paired EgoBody manifest by staged-RGB availability.

The filter is deliberately recording-level: if any selected case from one
recording is incomplete, every case from that recording is excluded.  This
preserves the protocol's fixed number of camera-pair strata per recording and
prevents an availability accident from reweighting the recording macro.

Runtime and evaluator rows are otherwise byte-for-value unchanged.  The
generated spec binds both parent and child manifests by SHA-256 and records
every excluded case/member.  No GT or model result is read by this tool.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from versions.v20.egobody.dataset import (  # noqa: E402
    canonical_json,
    file_sha256,
    value_sha256,
)


SPEC_SCHEMA = "Bridge3R-v20-EgoBody-availability-filter-spec-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--evaluator-manifest", type=Path, required=True)
    parser.add_argument("--staged-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument(
        "--split",
        choices=("development", "holdout"),
        required=True,
        help="Test filtering is intentionally forbidden after the protocol is frozen.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"Blank JSONL row: {path}:{line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row is not an object: {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


def index_rows(rows: list[dict[str, Any]], path: Path) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(rows, 1):
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"Missing case_id: {path}:{line_number}")
        if case_id in indexed:
            raise ValueError(f"Duplicate case_id {case_id!r}: {path}:{line_number}")
        indexed[case_id] = row
    return indexed


def safe_member_path(staged_root: Path, raw_member: Any) -> tuple[str, Path]:
    if not isinstance(raw_member, str) or not raw_member:
        raise ValueError(f"Invalid image member: {raw_member!r}")
    if "\\" in raw_member or "\x00" in raw_member or raw_member.startswith("/"):
        raise ValueError(f"Unsafe image member: {raw_member!r}")
    parts = PurePosixPath(raw_member).parts
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"Unsafe image member: {raw_member!r}")
    canonical = "/".join(parts)
    if canonical != raw_member:
        raise ValueError(f"Non-canonical image member: {raw_member!r}")
    root = staged_root.resolve()
    path = staged_root.joinpath(*parts)
    resolved = path.resolve(strict=False)
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Image member escapes staged root: {raw_member!r}")
    return canonical, path


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial.{os.getpid()}")
    try:
        with partial.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
    finally:
        try:
            partial.unlink()
        except FileNotFoundError:
            pass


def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")


def main() -> None:
    args = parse_args()
    runtime_path = args.runtime_manifest.resolve(strict=True)
    evaluator_path = args.evaluator_manifest.resolve(strict=True)
    staged_root = args.staged_root.resolve(strict=True)

    runtime_rows = read_jsonl(runtime_path)
    evaluator_rows = read_jsonl(evaluator_path)
    runtime_by_id = index_rows(runtime_rows, runtime_path)
    evaluator_by_id = index_rows(evaluator_rows, evaluator_path)
    if set(runtime_by_id) != set(evaluator_by_id):
        raise ValueError("Runtime/evaluator case-id sets differ")

    parent_counts: Counter[str] = Counter()
    missing_by_case: dict[str, list[str]] = {}
    for case_id, runtime in runtime_by_id.items():
        if runtime.get("split") != args.split:
            raise ValueError(f"Runtime split mismatch in {case_id}")
        evaluator = evaluator_by_id[case_id]
        if evaluator.get("split") != args.split:
            raise ValueError(f"Evaluator split mismatch in {case_id}")
        expected_hash = evaluator.get("runtime_row_sha256")
        actual_hash = value_sha256(runtime)
        if expected_hash != actual_hash:
            raise ValueError(f"Runtime-row hash mismatch in {case_id}")
        recording = runtime.get("recording")
        if not isinstance(recording, str) or not recording:
            raise ValueError(f"Missing recording in {case_id}")
        if evaluator.get("recording") != recording:
            raise ValueError(f"Runtime/evaluator recording mismatch in {case_id}")
        parent_counts[recording] += 1
        members = runtime.get("image_members")
        if not isinstance(members, list) or not members:
            raise ValueError(f"Missing image_members in {case_id}")
        missing: list[str] = []
        for raw_member in members:
            member, path = safe_member_path(staged_root, raw_member)
            if not path.is_file():
                missing.append(member)
        if missing:
            missing_by_case[case_id] = sorted(missing)

    distinct_parent_counts = set(parent_counts.values())
    if len(distinct_parent_counts) != 1:
        raise ValueError(f"Parent recording groups are imbalanced: {parent_counts}")
    cases_per_recording = next(iter(distinct_parent_counts))
    excluded_recordings = sorted(
        {str(runtime_by_id[case_id]["recording"]) for case_id in missing_by_case}
    )
    excluded_set = set(excluded_recordings)
    retained_runtime = [
        row for row in runtime_rows if str(row["recording"]) not in excluded_set
    ]
    retained_ids = {str(row["case_id"]) for row in retained_runtime}
    retained_evaluator = [
        row for row in evaluator_rows if str(row["case_id"]) in retained_ids
    ]
    if not retained_runtime:
        raise ValueError("Availability filter removed every recording")

    retained_counts = Counter(str(row["recording"]) for row in retained_runtime)
    if set(retained_counts.values()) != {cases_per_recording}:
        raise AssertionError(f"Filtered recording groups are imbalanced: {retained_counts}")
    if [row["case_id"] for row in retained_runtime] != [
        row["case_id"] for row in retained_evaluator
    ]:
        raise ValueError("Paired output row order differs")

    output_dir = args.output_dir.resolve()
    runtime_output = output_dir / f"{args.output_prefix}.runtime.jsonl"
    evaluator_output = output_dir / f"{args.output_prefix}.evaluator.jsonl"
    spec_output = output_dir / f"{args.output_prefix}.spec.json"
    for output in (runtime_output, evaluator_output, spec_output):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite filtered artifact: {output}")

    atomic_bytes(runtime_output, jsonl_bytes(retained_runtime))
    atomic_bytes(evaluator_output, jsonl_bytes(retained_evaluator))

    missing_members = sorted({m for values in missing_by_case.values() for m in values})
    excluded_cases = [
        str(row["case_id"])
        for row in runtime_rows
        if str(row["recording"]) in excluded_set
    ]
    spec = {
        "schema_version": SPEC_SCHEMA,
        "split": args.split,
        "filter_rule": (
            "Exclude an entire recording if any parent runtime case references "
            "an image member that is not a regular file below staged_root."
        ),
        "filter_inputs_use_gt": False,
        "filter_inputs_use_model_results": False,
        "staged_root": str(staged_root),
        "parent_runtime_manifest": str(runtime_path),
        "parent_runtime_manifest_sha256": file_sha256(runtime_path),
        "parent_evaluator_manifest": str(evaluator_path),
        "parent_evaluator_manifest_sha256": file_sha256(evaluator_path),
        "parent_case_count": len(runtime_rows),
        "parent_recording_count": len(parent_counts),
        "runtime_manifest": str(runtime_output),
        "runtime_manifest_sha256": file_sha256(runtime_output),
        "evaluator_manifest": str(evaluator_output),
        "evaluator_manifest_sha256": file_sha256(evaluator_output),
        "case_count": len(retained_runtime),
        "recording_count": len(retained_counts),
        "cases_per_recording": cases_per_recording,
        "recording_macro_balanced": True,
        "runtime_evaluator_case_ids_equal": True,
        "runtime_rows_unchanged_from_parent": True,
        "evaluator_rows_unchanged_from_parent": True,
        "excluded_recordings": excluded_recordings,
        "excluded_cases": excluded_cases,
        "cases_with_missing_members": missing_by_case,
        "missing_members": missing_members,
        "missing_member_count": len(missing_members),
    }
    atomic_bytes(
        spec_output,
        (json.dumps(spec, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
            "utf-8"
        ),
    )
    print(json.dumps({**spec, "spec": str(spec_output)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
