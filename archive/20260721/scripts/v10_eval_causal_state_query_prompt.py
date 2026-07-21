#!/usr/bin/env python3
"""Evaluate cached V10 causal state-query adapters on one or more clips."""

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

from dust3r.v10_causal_state_query_prompt import CausalStateQueryFirstWritePrompt  # noqa: E402
from v10_causal_state_query_prompt_validation import evaluate_adapters  # noqa: E402
from v10_latent_activation_patching_probe import build_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--input_dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--report_name", required=True)
    parser.add_argument("--model_path", type=Path, default=REPO_ROOT / "src" / "human3r_896L.pth")
    parser.add_argument("--data_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data"))
    parser.add_argument("--device", required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--max_frames", type=int, default=24)
    parser.add_argument("--boundaries", type=int, nargs="+", default=(6, 9, 12, 15))
    parser.add_argument("--train_clips", nargs="+", default=("clip01", "clip02"))
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--point_sample", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--resolution", type=int, nargs=2, default=(512, 288))
    parser.add_argument("--resize_mode", default="human3r_demo")
    return parser.parse_args()


def load_adapter(path: Path, hidden_dim: int, device: torch.device):
    payload = torch.load(path, map_location=device, weights_only=False)
    model = CausalStateQueryFirstWritePrompt(hidden_dim=hidden_dim).to(device).eval()
    model.load_state_dict(payload["model"])
    return model


def main() -> None:
    args = parse_args()
    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required")
    device = torch.device(args.device)
    human3r = build_model(args)
    raw_adapter = load_adapter(
        args.output_dir / "checkpoints" / "raw_first_write.pth",
        int(args.hidden_dim),
        device,
    )
    early_adapter = load_adapter(
        args.output_dir / "checkpoints" / "early_query_first_write.pth",
        int(args.hidden_dim),
        device,
    )
    rows = json.loads(
        (args.output_dir / "state_pair_cache" / "index.json").read_text(encoding="utf-8")
    )
    rollout = evaluate_adapters(human3r, raw_adapter, early_adapter, rows, args, device)
    report_path = args.output_dir / f"rollout_{args.report_name}.json"
    report_path.write_text(json.dumps(rollout, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    print(f">> wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
