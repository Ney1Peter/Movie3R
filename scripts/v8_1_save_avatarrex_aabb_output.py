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
    parser.add_argument("--avatarrex_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data"))
    parser.add_argument("--split", default="Training")
    parser.add_argument("--seq_a", default="lbn1/22010710")
    parser.add_argument("--seq_b", default="lbn1/22053923")
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
    parser.add_argument(
        "--oracle_correction_gate",
        action="store_true",
        help="Use view['shot_label'] as an oracle V9 correction gate during inference.",
    )
    parser.add_argument(
        "--oracle_correction_cache",
        action="store_true",
        help="When the oracle gate is off, reuse the previous correction residual within the same shot.",
    )
    parser.add_argument(
        "--oracle_pre_decoder_gate",
        action="store_true",
        help="Apply oracle gate before the decoder. By default oracle only gates/caches decoder-after residuals.",
    )
    parser.add_argument(
        "--freeze_updates_from_view",
        type=int,
        default=-1,
        help=(
            "Inference-only state freeze probe. If >=0, views from this index "
            "onward do not update recurrent state, pose memory, or V9 history."
        ),
    )
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
    if args.freeze_updates_from_view >= 0:
        for view_idx, view in enumerate(views):
            if view_idx < args.freeze_updates_from_view:
                continue
            update_mask = torch.zeros_like(view["img_mask"], dtype=torch.bool)
            view["update"] = update_mask
            view["update_state"] = update_mask
            view["update_mem"] = update_mask
            view["update_v8_history"] = update_mask

    add_path_to_dust3r(str(args.model_path))
    model = ARCroco3DStereo.from_pretrained(str(args.model_path)).to(device).eval()
    if args.oracle_correction_gate or args.oracle_correction_cache:
        model.v9_oracle_correction_gate_enabled = True
        model.v9_oracle_correction_cache_enabled = bool(args.oracle_correction_cache)
        model.v9_oracle_correction_inference_only = True
        model.v9_oracle_correction_post_residual_only = not bool(args.oracle_pre_decoder_gate)
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

    diagnostics = []
    for view_idx, pred in enumerate(outputs["pred"]):
        row = {"view_idx": int(view_idx)}
        view = outputs["views"][view_idx]
        if "shot_label" in view:
            shot_value = view["shot_label"]
            if hasattr(shot_value, "detach"):
                shot_value = shot_value.detach().float().reshape(-1)[0].item()
            row["shot_label"] = float(shot_value)
        for update_key in ("update", "update_state", "update_mem", "update_v8_history"):
            if update_key not in view:
                continue
            update_value = view[update_key]
            if hasattr(update_value, "detach"):
                update_value = update_value.detach().float().reshape(-1)[0].item()
            row[update_key] = float(update_value)
        for key in (
            "v9_oracle_new_correction_gate",
            "v9_oracle_pose_cache_applied",
            "v9_oracle_human_cache_applied",
            "v9_oracle_force_gate",
            "v9_pre_decoder_effective_gate",
            "v8_pose_prompt_gate",
            "v8_pose_prompt_delta_applied",
            "v8_human_latent_corr_delta_applied",
        ):
            value = pred.get(key, None)
            if value is None:
                continue
            if hasattr(value, "detach"):
                tensor = value.detach().float()
                row[f"{key}_mean"] = float(tensor.mean().item())
                if "delta" in key:
                    row[f"{key}_norm"] = float(tensor.norm(dim=-1).mean().item())
            else:
                row[key] = value
        diagnostics.append(row)
    with open(args.output_dir / "v9_oracle_correction_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2, sort_keys=True)
        f.write("\n")

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
        "oracle_correction_gate": bool(args.oracle_correction_gate or args.oracle_correction_cache),
        "oracle_correction_cache": bool(args.oracle_correction_cache),
        "oracle_correction_stage": "pre_decoder" if args.oracle_pre_decoder_gate else "post_residual",
        "freeze_updates_from_view": int(args.freeze_updates_from_view),
        "note": "Saved depth/pointmap comes from the model prediction; AvatarReX DA3 depth is not used unless --load_da3_depth is set.",
        "saved_output": True,
    }
    with open(args.output_dir / "avatarrex_aabb_save_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
