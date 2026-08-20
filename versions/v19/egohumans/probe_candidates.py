#!/usr/bin/env python3
"""Run the frozen v16/v17 prediction-only grid with EgoHumans GT loading.

The candidate implementation and metric formulas stay bit-identical to the
Harmony4D probe.  Only the GT loader is replaced after import; GT remains
strictly evaluator-only and never enters ``apply_candidate``.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from versions.v16.harmony4d import probe_causal_stabilization as frozen_probe  # noqa: E402
from versions.v19.egohumans.evaluate_egohumans import load_gt  # noqa: E402


def main() -> None:
    frozen_probe.load_gt = load_gt
    frozen_probe.main()


if __name__ == "__main__":
    main()

