#!/usr/bin/env python3
"""Run the Human3R latent activation-patching probe on an image directory."""

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

from demo import prepare_input  # noqa: E402
from dust3r.utils.device import todevice  # noqa: E402
from v10_latent_activation_patching_probe import (  # noqa: E402
    aggregate,
    architecture_judgement,
    build_model,
    plot_layer_curves,
    plot_recovery,
    run_case,
    write_csv,
    write_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, default=REPO_ROOT / "src" / "human3r_896L.pth")
    parser.add_argument("--data_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data"))
    parser.add_argument("--device", default="cuda:6" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--boundary", type=int, default=6)
    parser.add_argument("--max_frames", type=int, default=12)
    parser.add_argument("--point_sample", type=int, default=20000)
    parser.add_argument("--decoder_layers", type=int, nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resolution", type=int, nargs=2, default=(512, 288))
    parser.add_argument("--resize_mode", default="human3r_demo")
    return parser.parse_args()


def image_paths(input_dir: Path, max_frames: int) -> list[str]:
    paths = sorted(
        path
        for path in input_dir.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    )
    if max_frames > 0:
        paths = paths[:max_frames]
    if not paths:
        raise FileNotFoundError(f"No images found in {input_dir}")
    return [str(path) for path in paths]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(args)
    paths = image_paths(args.input_dir, int(args.max_frames))
    if int(args.boundary) <= 0 or int(args.boundary) >= len(paths):
        raise ValueError(f"boundary={args.boundary} must be within 1..{len(paths) - 1}")
    views = prepare_input(
        paths,
        [True] * len(paths),
        int(args.size),
        revisit=1,
        update=True,
        img_res=model.mhmr_img_res,
        reset_interval=1000000,
    )
    device = torch.device(args.device)
    views = todevice(views, device)
    record = {
        "source": "in_the_wild_video",
        "group": args.input_dir.parent.name,
        "seqA": args.input_dir.name,
        "start_frame": 0,
        "pattern_id": args.input_dir.name,
    }
    case = run_case(model, record, [record], args, device, 0, views_override=views)
    overall = aggregate([case])
    report = {
        "experiment": "Human3R latent activation patching on moving video",
        "input_images": paths,
        "boundary": int(args.boundary),
        "overall": overall,
        "architecture_judgement": architecture_judgement(overall),
        "cases": [case],
    }
    serializable = json.loads(json.dumps(report, default=str, allow_nan=True))
    (args.output_dir / "activation_patching_metrics.json").write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "activation_patching_cases.csv", [case])
    plot_recovery(args.output_dir / "activation_recovery_matrix.png", overall)
    plot_layer_curves(args.output_dir / "decoder_layer_recovery_curves.png", overall)
    write_markdown(args.output_dir / "activation_patching_metrics.md", serializable)
    print(f">> wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
