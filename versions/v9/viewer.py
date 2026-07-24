#!/usr/bin/env python3
"""View a saved Movie3R-Learned V9 inference result."""

from __future__ import annotations

import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    runpy.run_path(
        str(ROOT / "scripts/view_human3r_saved_output.py"),
        run_name="__main__",
    )
