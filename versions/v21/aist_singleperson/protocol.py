#!/usr/bin/env python3
"""Shared immutable-input helpers for the Bridge3R AIST++ v1 protocol.

The source-selection manifests in the uploaded AIST++ bundle are already
frozen.  This module may read and validate them, but it deliberately contains
no operation that can regenerate or overwrite them.  All outputs belong in
``data/bridge3r_singleperson_v1``.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = REPO_ROOT.parent
DEFAULT_BUNDLE_ROOT = (
    WORKSPACE_ROOT / "data/HumanMM_AIST_Bridge3R_v1_20260826"
)
DEFAULT_DERIVED_ROOT = WORKSPACE_ROOT / "data/bridge3r_singleperson_v1"
PROTOCOL_NAME = "Bridge3R-AIST-SinglePerson-v1"
OUTPUT_FPS = 30
OUTPUT_FRAMES = 150
GT_FPS = 60
PROTOCOLS: dict[str, dict[str, Any]] = {
    "CS150": {"shot_lengths": (75, 75), "cut_indices": (74,)},
    "MC150-3": {"shot_lengths": (50, 50, 50), "cut_indices": (49, 99)},
    "MC150-4": {"shot_lengths": (38, 38, 37, 37), "cut_indices": (37, 75, 112)},
}


def sha256_file(path: Path) -> str:
    """Return a SHA-256 digest without loading the complete file in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_digest(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    """Atomically write a JSON artifact without partially replacing a valid one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def bundle_manifest_root(bundle_root: Path) -> Path:
    return bundle_root / "data/manifests/bridge3r_singleperson_v1"


def verify_input_manifest_freeze(bundle_root: Path) -> dict[str, str]:
    """Verify the four small immutable source-selection manifests.

    This is intentionally narrower than re-hashing the full 6.5-GB uploaded
    package on every builder invocation.  The package-wide checksum was
    already checked during intake; this guard protects the source/camera/window
    definitions that must not drift during construction.
    """
    manifest_root = bundle_manifest_root(bundle_root)
    checksum = manifest_root / "aist_manifest_freeze_sha256.txt"
    if not checksum.is_file():
        raise FileNotFoundError(f"Missing frozen-manifest hash list: {checksum}")
    expected: dict[str, str] = {}
    for line in checksum.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, filename = line.split(maxsplit=1)
        expected[filename.strip()] = digest
    if set(expected) != {
        "aist_source_selection_v1.json",
        "source_index_v1.json",
        "aist_selected_urls.txt",
        "aist_selected_files_v1.tsv",
    }:
        raise ValueError(f"Unexpected frozen-manifest members: {sorted(expected)}")
    observed = {
        filename: sha256_file(manifest_root / filename) for filename in expected
    }
    mismatches = {
        filename: {"expected": expected[filename], "observed": observed[filename]}
        for filename in expected
        if expected[filename] != observed[filename]
    }
    if mismatches:
        raise ValueError(f"Frozen AIST++ input manifests differ: {mismatches}")
    return observed


def load_frozen_sources(bundle_root: Path, roles: Iterable[str]) -> list[dict[str, Any]]:
    """Load selected Test/Pilot sources in frozen rank order."""
    requested = set(roles)
    if not requested or requested - {"test", "pilot"}:
        raise ValueError(f"Roles must be a nonempty subset of test,pilot; got {roles}")
    manifest_root = bundle_manifest_root(bundle_root)
    payload = json.loads((manifest_root / "source_index_v1.json").read_text(encoding="utf-8"))
    if payload.get("manifest_status") != "frozen":
        raise ValueError("source_index_v1.json is not frozen")
    sources = [row for row in payload["sources"] if row.get("role") in requested]
    if not sources:
        raise ValueError(f"No frozen AIST++ sources for roles={sorted(requested)}")
    source_ids = [str(row["source_id"]) for row in sources]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Duplicate source IDs in frozen source index")
    for row in sources:
        if row.get("paper_metrics_allowed") != (row.get("role") == "test"):
            raise ValueError(f"Unexpected paper-metric flag for {row['source_id']}")
        if int(row.get("output_num_frames", -1)) != OUTPUT_FRAMES:
            raise ValueError(f"Unexpected output length for {row['source_id']}")
        if int(row.get("output_fps", -1)) != OUTPUT_FPS:
            raise ValueError(f"Unexpected output FPS for {row['source_id']}")
        if len(row.get("camera_ids", [])) != 4:
            raise ValueError(f"Need exactly four frozen cameras for {row['source_id']}")
    return sorted(sources, key=lambda row: (str(row["role"]), int(row["selection_rank"])))


def source_video_path(bundle_root: Path, source: dict[str, Any], camera_id: str) -> Path:
    sequence = str(source["sequence_name"])
    if "cAll" not in sequence:
        raise ValueError(f"AIST source has no cAll component: {sequence}")
    path = (
        bundle_root
        / "data/raw/aistplusplus_v1/videos_selected"
        / (sequence.replace("cAll", camera_id) + ".mp4")
    )
    if not path.is_file():
        raise FileNotFoundError(f"Frozen RGB video is absent: {path}")
    return path


def output_gt_ticks(source: dict[str, Any]) -> np.ndarray:
    start = int(source["source_start_gt_index_60fps"])
    ticks = start + 2 * np.arange(OUTPUT_FRAMES, dtype=np.int64)
    if int(ticks[-1]) != int(source["output_last_gt_index"]):
        raise ValueError(f"Output GT tick rule drifted for {source['source_id']}")
    return ticks


def camera_records(bundle_root: Path, source: dict[str, Any]) -> dict[str, dict[str, Any]]:
    path = (
        bundle_root
        / "data/raw/aistplusplus_v1/annotations/cameras"
        / str(source["camera_setting"])
    )
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Expected list-valued AIST camera setting: {path}")
    result = {str(row["name"]): row for row in records}
    missing = set(source["camera_ids"]) - set(result)
    if missing:
        raise ValueError(f"Frozen camera(s) missing from {path}: {sorted(missing)}")
    return result


def camera_world_to_camera(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return AIST official K, world-to-camera R and t in native centimetres."""
    K = np.asarray(record["matrix"], dtype=np.float64).reshape(3, 3)
    rotation, _ = cv2.Rodrigues(
        np.asarray(record["rotation"], dtype=np.float64).reshape(3, 1)
    )
    translation = np.asarray(record["translation"], dtype=np.float64).reshape(3)
    return K, rotation, translation


def camera_to_world(record: dict[str, Any]) -> np.ndarray:
    K, rotation, translation = camera_world_to_camera(record)
    del K
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return np.linalg.inv(transform)


def run_checked(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout
