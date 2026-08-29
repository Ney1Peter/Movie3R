#!/usr/bin/env python3
"""Strict MVH150 schema layer for the frozen AIST GVHMR converter.

This wrapper does not alter SMPL-X/SMPL24 conversion, tracking, or the
declared unavailable-camera contract.  It only admits the frozen MVH150
prediction-only row and adds a two-file converter provenance chain.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
MOVIE3R = HERE.parents[2]
if str(MOVIE3R) not in sys.path:
    sys.path.insert(0, str(MOVIE3R))

from versions.v21.aist_singleperson import convert_gvhmr_result as base  # noqa: E402


BASE = Path(base.__file__).resolve()
EXPECTED_KEYS = {"case_id", "dataset", "protocol", "role", "input_video", "fps", "num_frames"}
FRAMES, FPS = 150, 30
SCHEMA = "Bridge3R-MVHuman-GVHMR-adapter-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_runtime(path: Path, line_number: int) -> dict[str, Any]:
    rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if line_number < 1 or line_number > len(rows):
        raise IndexError(f"line {line_number} outside manifest with {len(rows)} rows")
    row = json.loads(rows[line_number - 1])
    if not isinstance(row, dict) or set(row) != EXPECTED_KEYS:
        raise ValueError("MVH150 compact runtime row schema drifted or evaluator-only data leaked")
    if row.get("dataset") != "MVHuman" or row.get("protocol") != "MVH150" or row.get("role") != "test":
        raise ValueError("this schema layer accepts only frozen MVHuman/MVH150 Test rows")
    if int(row.get("num_frames", -1)) != FRAMES or int(row.get("fps", -1)) != FPS:
        raise ValueError("MVH150 temporal contract drifted")
    relative = Path(str(row.get("input_video", "")))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("unsafe MVH150 runtime video path")
    return row


def main() -> None:
    base_schema = str(base.SCHEMA)
    original_atomic_json = base.atomic_json

    def provenance_atomic_json(path: Path, payload: dict[str, Any]) -> None:
        value = dict(payload)
        value["schema_version"] = SCHEMA
        value["schema_layer"] = {
            "contract": "MVHuman/MVH150 Test, 150 frames at 30 FPS; no evaluator fields",
            "base_converter_schema": base_schema,
            "base_converter": str(BASE),
            "base_converter_sha256": sha256(BASE),
            "mvhuman_wrapper": str(Path(__file__).resolve()),
            "mvhuman_wrapper_sha256": sha256(Path(__file__).resolve()),
            "numerical_conversion_changed": False,
        }
        original_atomic_json(path, value)

    base.read_runtime = read_runtime
    base.atomic_json = provenance_atomic_json
    base.main()


if __name__ == "__main__":
    main()
