#!/usr/bin/env python3
"""Train one cached V10 causal state-query adapter on a dedicated GPU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from v10_causal_state_query_prompt_validation import train_one_adapter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--variant", choices=("raw", "early"), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--train_clips", nargs="+", default=("clip01", "clip02"))
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--overwrite_train", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required")
    index_path = args.output_dir / "state_pair_cache" / "index.json"
    rows = json.loads(index_path.read_text(encoding="utf-8"))
    if args.variant == "raw":
        name, fresh_key, camera_key = "raw_first_write", "raw_fresh_state", "raw_camera"
    else:
        name, fresh_key, camera_key = (
            "early_query_first_write",
            "early_fresh_state",
            "early_camera",
        )
    _model, report = train_one_adapter(
        name,
        rows,
        fresh_key,
        camera_key,
        args,
        torch.device(args.device),
    )
    report_path = args.output_dir / f"{name}_training_report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    print(f">> wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
