#!/usr/bin/env python3
"""Save Human3R/V8.1 outputs from one AvatarReX_AABB dataloader sample.

This intentionally uses the dataset preprocessing and full recurrent inference
path used by training/eval, instead of demo.py's image-folder lightweight path.
It is useful for strict overfit visualization.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from add_ckpt_path import add_path_to_dust3r
from demo import prepare_output
from dust3r.datasets.avatarrex import AvatarReX_AABB
from dust3r.model import ARCroco3DStereo
from dust3r.smpl_model import SMPLModel
from dust3r.utils.device import to_cpu, todevice


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--avatarrex_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data/Avatarrex_output"))
    parser.add_argument("--split", default="Training")
    parser.add_argument("--seq_a", default="22010710")
    parser.add_argument("--seq_b", default="22053923")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resolution", type=int, nargs=2, default=(512, 288), metavar=("W", "H"))
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument(
        "--load_da3_depth",
        action="store_true",
        help="Load AvatarReX_output depth into GT views. Disabled by default for V8.1 pose-only checks.",
    )
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

    dataset = AvatarReX_AABB(
        split=args.split,
        ROOT=str(args.avatarrex_root),
        resolution=tuple(args.resolution),
        num_views=4,
        aug_crop=0,
        allow_repeat=True,
        seed=args.seed,
        n_corres=0,
        fixed_samples=[(args.seq_a, args.seq_b, int(args.start_frame))],
        load_da3_depth=bool(args.load_da3_depth),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    views = next(iter(loader))

    add_path_to_dust3r(str(args.model_path))
    model = ARCroco3DStereo.from_pretrained(str(args.model_path)).to(device).eval()
    img_res = getattr(model, "mhmr_img_res", None)
    smpl_model = SMPLModel(
        device,
        model_args={
            "patch_size": model.croco_args["patch_size"],
            "mhmr_img_res": model.mhmr_img_res,
            "bb_patch_size": model.bb_patch_size,
        },
    )

    with torch.no_grad():
        views = todevice(views, device)
        smpl_model.update_smpl_gt(views)
        print(f">> Inference with model on {len(views)} AvatarReX_AABB dataloader views")
        output, _ = model(views, ret_state=True, inference=True)
        outputs = to_cpu({"views": output.views, "pred": output.ress})

    prepare_output(
        outputs,
        str(args.output_dir),
        revisit=1,
        use_pose=True,
        save=True,
        render=args.render,
        render_video=args.render_video,
        img_res=img_res,
        subsample=1,
    )

    summary = {
        "model_path": str(args.model_path),
        "avatarrex_root": str(args.avatarrex_root),
        "split": args.split,
        "seq_a": args.seq_a,
        "seq_b": args.seq_b,
        "start_frame": int(args.start_frame),
        "output_dir": str(args.output_dir),
        "resolution": list(args.resolution),
        "inference_path": "AvatarReX_AABB dataloader + inference",
        "load_da3_depth": bool(args.load_da3_depth),
        "note": "Saved depth/pointmap comes from the model prediction; AvatarReX DA3 depth is not used unless --load_da3_depth is set.",
        "saved_output": True,
    }
    with open(args.output_dir / "avatarrex_aabb_save_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
