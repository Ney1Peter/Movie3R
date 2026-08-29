#!/usr/bin/env python3
"""MVH150 schema adapter for the unchanged official GVHMR runner."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from versions.v21.aist_singleperson import run_gvhmr_case as official


def read_runtime(path: Path, line_number: int) -> dict[str, Any]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if line_number < 1 or line_number > len(lines):
        raise IndexError(f"--line {line_number} outside manifest with {len(lines)} rows")
    row = json.loads(lines[line_number - 1])
    if not isinstance(row, dict) or set(row) != official.EXPECTED_KEYS:
        raise ValueError("MVH150 runtime row schema drifted")
    if row["dataset"] != "MVHuman" or row["protocol"] != "MVH150" or row["role"] != "test":
        raise ValueError("This adapter accepts only frozen MVHuman MVH150 Test rows")
    if int(row["fps"]) != 30 or int(row["num_frames"]) != 150:
        raise ValueError("MVH150 temporal contract drifted")
    return row


official.SCHEMA = "Bridge3R-MVHuman-GVHMR-official-runtime-v1"
official.read_runtime = read_runtime


if __name__ == "__main__":
    official.main()
