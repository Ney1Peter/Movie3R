#!/usr/bin/env python3
"""Run Movie3R-Learned V9 with the preserved inference interface."""

from __future__ import annotations

import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    runpy.run_path(
        str(ROOT / "scripts/run_human3r_save_output.py"),
        run_name="__main__",
    )
