#!/usr/bin/env python3
"""Strict MVH150 schema layer for the frozen PromptHMR converter.

The numerical conversion remains in
``external_baselines/bridge3r_eval/convert_prompthmr_result.py``.  That
converter originally accepted AIST CS150 compact rows only.  This wrapper
changes only the prediction-only manifest parser, and records both source
files in the adapter metadata so a formal audit can bind the exact code.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[3]
BASE = WORKSPACE / "external_baselines" / "bridge3r_eval" / "convert_prompthmr_result.py"
EXPECTED_KEYS = {"case_id", "dataset", "protocol", "role", "input_video", "fps", "num_frames"}
FRAMES, FPS = 150, 30
SCHEMA = "Bridge3R-MVHuman-PromptHMR-adapter-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_row(path: Path, line_number: int) -> dict[str, Any]:
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
    row["_adapter_frame_count"] = FRAMES
    return row


def load_base() -> Any:
    if not BASE.is_file():
        raise FileNotFoundError(BASE)
    spec = importlib.util.spec_from_file_location("bridge3r_base_prompthmr_converter", BASE)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import frozen PromptHMR converter: {BASE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    base = load_base()
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

    base.read_row = read_row
    base.atomic_json = provenance_atomic_json
    base.main()


if __name__ == "__main__":
    main()
