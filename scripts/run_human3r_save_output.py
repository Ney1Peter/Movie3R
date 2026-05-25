#!/usr/bin/env python3
"""Run Human3R inference and save outputs without launching the viewer."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from add_ckpt_path import add_path_to_dust3r
from demo import parse_seq_path, prepare_input, prepare_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--seq_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--subsample", type=int, default=1)
    parser.add_argument("--reset_interval", type=int, default=10000000)
    parser.add_argument("--use_ttt3r", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render_video", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def output_complete(output_dir: Path) -> bool:
    return (output_dir / "camera").is_dir() and any((output_dir / "camera").glob("*.npz"))


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and output_complete(args.output_dir) and not args.overwrite:
        print(json.dumps({"output_dir": str(args.output_dir), "status": "exists"}, sort_keys=True))
        return
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    add_path_to_dust3r(str(args.model_path))
    from src.dust3r.inference import inference_recurrent_lighter
    from src.dust3r.model import ARCroco3DStereo

    img_paths, tmpdirname = parse_seq_path(str(args.seq_path))
    if args.max_frames is not None:
        img_paths = img_paths[: int(args.max_frames)]
    img_paths = img_paths[:: int(args.subsample)]
    if not img_paths:
        raise FileNotFoundError(args.seq_path)

    model = ARCroco3DStereo.from_pretrained(str(args.model_path)).to(device)
    model.eval()
    img_res = getattr(model, "mhmr_img_res", None)
    views = prepare_input(
        img_paths=img_paths,
        img_mask=[True] * len(img_paths),
        size=args.size,
        revisit=1,
        update=True,
        img_res=img_res,
        reset_interval=args.reset_interval,
    )
    if tmpdirname is not None:
        shutil.rmtree(tmpdirname)

    with torch.no_grad():
        outputs, _ = inference_recurrent_lighter(
            views, model, str(device), use_ttt3r=args.use_ttt3r
        )
    prepare_output(
        outputs,
        str(args.output_dir),
        1,
        True,
        True,
        args.render,
        args.render_video,
        img_res,
        args.subsample,
    )
    summary = {
        "model_path": str(args.model_path),
        "seq_path": str(args.seq_path),
        "output_dir": str(args.output_dir),
        "num_frames": int(len(img_paths)),
        "subsample": int(args.subsample),
        "saved_output": True,
    }
    with open(args.output_dir / "human3r_save_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
