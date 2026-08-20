#!/usr/bin/env python3
"""Parallel scheduling wrapper for the EgoHumans-fitted frozen evaluator."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from versions.v17.harmony4d import probe_parallel as frozen_parallel


def main() -> None:
    # Scheduling/merge logic remains byte-for-byte the frozen v17 wrapper; only
    # each shard's executable is replaced by the EgoHumans GT-loader adapter.
    frozen_parallel.PROBE = REPO_ROOT / "versions/v19/egohumans/probe_candidates.py"
    frozen_parallel.main()


if __name__ == "__main__":
    main()
