#!/usr/bin/env python3
"""Fine-tune a V12 first-write adapter through future gauge-neutral rollout."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from dust3r.v12_gated_gauge_neutral_prompt import GatedGaugeNeutralFirstWritePrompt  # noqa: E402
from v10_causal_state_query_prompt_validation import image_paths, prepare_views  # noqa: E402
from v11_gauge_neutral_first_write_oracle import gpu_output_filter  # noqa: E402
from v11_gauge_neutral_first_write_probe import build_dataset, build_model, configure_views  # noqa: E402
from v12_gated_first_write_runtime import (  # noqa: E402
    GatedFirstWriteController,
    cached_gauge_neutral_loss,
    distillation_auxiliary,
)


DEFAULT_CACHE = REPO_ROOT / "output" / "v12_gated_first_write" / "teacher_cache_loso_mvhuman200"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "v12_gated_first_write" / "training_loso_mvhuman200"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--variant", choices=("gated", "ungated", "no_old"), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--model_path", type=Path, default=REPO_ROOT / "src" / "human3r_896L.pth")
    parser.add_argument("--data_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data"))
    parser.add_argument("--resolution", type=int, nargs=2, default=(512, 288))
    parser.add_argument("--resize_mode", default="human3r_demo")
    parser.add_argument("--max_post_frames", type=int, default=9)
    parser.add_argument("--warmup_frames", type=int, default=8)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--hidden_dim", type=int, default=192)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--max_frames", type=int, default=24)
    return parser.parse_args()


def one_batch(dataset) -> list[dict]:
    return next(iter(torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)))


def real_views(pair: dict, human3r, args, device: torch.device) -> list[dict]:
    metadata = pair["metadata"]
    record = {
        "angle_bucket": metadata["angle_bucket"],
        "clip_type": "aabb",
        "group": metadata["group"],
        "seqA": metadata["seqA"],
        "seqB": metadata["seqB"],
        "start_frame": int(metadata["start_frame"]),
        "view_angle_deg": float(metadata["view_angle_deg"]),
        "source": metadata["source"],
        "pattern_id": metadata["case_name"],
    }
    spec = {
        "record": record,
        "post_frames": list(metadata["post_frames"]),
        "warmup_frames": [],
        "post_count": len(metadata["post_frames"]),
        "warmup_count": 0,
    }
    return configure_views(one_batch(build_dataset([spec], False, args)), device, human3r.mhmr_img_res)


def pseudo_views(pair: dict, human3r, args, device: torch.device) -> list[dict]:
    metadata = pair["metadata"]
    paths = image_paths(Path(metadata["input_dir"]), args.max_frames)
    views = prepare_views(paths, human3r, args, device)
    boundary = int(metadata["boundary"])
    return views[boundary : boundary + args.max_post_frames]


def views_for_pair(pair: dict, human3r, args, device: torch.device) -> list[dict]:
    return real_views(pair, human3r, args, device) if pair["metadata"]["kind"] == "real" else pseudo_views(pair, human3r, args, device)


def main() -> None:
    args = parse_args()
    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("V12 rollout fine-tuning requires CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    rows = json.loads((args.cache_dir / "index_train.json").read_text(encoding="utf-8"))
    distill = args.output_dir / "checkpoints" / f"{args.variant}_distill.pth"
    output = args.output_dir / "checkpoints" / f"{args.variant}_rollout.pth"
    if output.is_file() and not args.overwrite:
        print(f">> exists {output}")
        return
    adapter = GatedGaugeNeutralFirstWritePrompt(hidden_dim=args.hidden_dim).to(device)
    payload = torch.load(distill, map_location=device, weights_only=False)
    adapter.load_state_dict(payload["model"])
    human3r = build_model(args)
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    history = []
    adapter.train()
    for step in range(1, args.steps + 1):
        row = rows[random.randrange(len(rows))]
        pair = torch.load(row["path"], map_location="cpu", weights_only=False)
        views = views_for_pair(pair, human3r, args, device)
        optimizer.zero_grad(set_to_none=True)
        with gpu_output_filter(), GatedFirstWriteController(
            human3r,
            adapter,
            pair,
            args.variant,
            seed=args.seed + step,
        ) as controller:
            predictions, _ = human3r.forward_recurrent_lighter(
                views,
                str(device),
                ret_state=False,
                use_ttt3r=False,
                return_token_debug=False,
            )
        rollout_loss, terms = cached_gauge_neutral_loss(predictions, pair, device)
        auxiliary = distillation_auxiliary(controller.output, pair, device, args.variant)
        loss = rollout_loss + auxiliary
        loss.backward()
        nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 20 == 0 or step == args.steps:
            item = {
                "step": step,
                "case_name": pair["metadata"]["case_name"],
                "kind": pair["metadata"]["kind"],
                "loss": float(loss.detach()),
                "rollout_loss": float(rollout_loss.detach()),
                "auxiliary": float(auxiliary.detach()),
                "gate": float(controller.output.gate.mean().detach()),
                "terms": terms,
            }
            history.append(item)
            print(f">> {args.variant} {item}", flush=True)
        del views, predictions, pair, controller, loss, rollout_loss, auxiliary
        torch.cuda.empty_cache()
    report = {
        "variant": args.variant,
        "steps": args.steps,
        "train_cases": len(rows),
        "history": history,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": adapter.state_dict(), "report": report, "args": vars(args)}, output)
    report_path = args.output_dir / f"{args.variant}_rollout_report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    print(f">> wrote {output}", flush=True)


if __name__ == "__main__":
    main()
