#!/usr/bin/env python3
"""Run an archived experiment script with its removed dust3r modules loaded."""

from __future__ import annotations

import importlib.util
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FROZEN_MODULES = {
    "dust3r.v10_causal_state_query_prompt": (
        ROOT / "archive/20260721/src/dust3r/v10_causal_state_query_prompt.py"
    ),
    "dust3r.v12_gated_gauge_neutral_prompt": (
        ROOT / "archive/20260721/src/dust3r/v12_gated_gauge_neutral_prompt.py"
    ),
}


def load_module(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load frozen module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: v14_5_run_frozen_script.py SCRIPT [ARGS ...]")
    script = Path(sys.argv[1]).resolve()
    if not script.is_file():
        raise FileNotFoundError(script)
    for name, path in FROZEN_MODULES.items():
        load_module(name, path)
    sys.argv = [str(script), *sys.argv[2:]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
