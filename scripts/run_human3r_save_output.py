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
    parser.add_argument(
        "--freeze_state_after",
        type=int,
        default=None,
        help=(
            "Inference-only state-write ablation. Frames with original index >= "
            "this value still run forward but do not update recurrent state, "
            "pose memory, or V9 correction history."
        ),
    )
    parser.add_argument(
        "--freeze_state_feat_after",
        type=int,
        default=None,
        help=(
            "Inference-only ablation. Frames with original index >= this value "
            "do not update the global recurrent state_feat."
        ),
    )
    parser.add_argument(
        "--freeze_pose_memory_after",
        type=int,
        default=None,
        help=(
            "Inference-only ablation. Frames with original index >= this value "
            "do not update the pose retriever memory."
        ),
    )
    parser.add_argument(
        "--freeze_v9_history_after",
        type=int,
        default=None,
        help=(
            "Inference-only ablation. Frames with original index >= this value "
            "do not update V9 correction-token history."
        ),
    )
    parser.add_argument("--use_ttt3r", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render_video", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--strict_original_human3r",
        action="store_true",
        help="Disable all Movie3R pose/human adaptation branches after loading the checkpoint.",
    )
    parser.add_argument(
        "--disable_v8_pose_prompt",
        action="store_true",
        help="Disable V8/V9 correct-token branch during inference.",
    )
    parser.add_argument(
        "--disable_v8_human_latent_corr",
        action="store_true",
        help="Disable V8/V9 human latent correction during inference.",
    )
    parser.add_argument(
        "--disable_v8_human_trans_corr",
        action="store_true",
        help="Disable V8/V9 human translation correction during inference.",
    )
    parser.add_argument(
        "--disable_v8_head_lora",
        action="store_true",
        help="Disable both pose-head and human-head LoRA branches during inference.",
    )
    parser.add_argument(
        "--disable_v8_pose_head_lora",
        action="store_true",
        help="Disable only the pose-head LoRA branch during inference.",
    )
    parser.add_argument(
        "--disable_v8_human_head_lora",
        action="store_true",
        help="Disable only the human-head LoRA branches during inference.",
    )
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
    from src.dust3r.v8_head_lora import set_lora_enabled

    img_paths, tmpdirname = parse_seq_path(str(args.seq_path))
    if args.max_frames is not None:
        img_paths = img_paths[: int(args.max_frames)]
    img_paths = img_paths[:: int(args.subsample)]
    if not img_paths:
        raise FileNotFoundError(args.seq_path)

    model = ARCroco3DStereo.from_pretrained(str(args.model_path)).to(device)

    disabled_lora = {}
    if args.disable_v8_head_lora or args.disable_v8_pose_head_lora:
        n = 0
        if hasattr(model.downstream_head, "pose_head"):
            n = set_lora_enabled(model.downstream_head.pose_head, False)
        disabled_lora["pose_head"] = n
    if args.disable_v8_head_lora or args.disable_v8_human_head_lora:
        n = 0
        for attr in ("deccam", "decpose", "decshape", "decexpression"):
            if hasattr(model.downstream_head, attr):
                n += set_lora_enabled(getattr(model.downstream_head, attr), False)
        disabled_lora["human_head"] = n

    if args.strict_original_human3r:
        disabled_flags = [
            "enable_shot_adaptation",
            "enable_shot_decoder_token",
            "enable_anchor_pose_adapter",
            "enable_anchor_decoder_tokens",
            "enable_anchor_pose_token_adapter",
            "enable_v7_pose_adapter",
            "enable_v8_pose_prompt",
            "enable_v8_human_trans_corr",
            "enable_v8_human_latent_corr",
            "enable_v8_head_lora",
            "enable_layerwise_pose_shot_adapter",
            "enable_pose_alignment_adapter",
            "enable_pose_translation_adapter",
            "enable_pose_lora",
            "enable_human_lora",
            "enable_world_lora",
        ]
        for name in disabled_flags:
            if hasattr(model, name):
                setattr(model, name, False)
        disabled_lora["pose_head"] = (
            set_lora_enabled(model.downstream_head.pose_head, False)
            if hasattr(model.downstream_head, "pose_head")
            else 0
        )
        human_lora_count = 0
        for attr in ("deccam", "decpose", "decshape", "decexpression"):
            if hasattr(model.downstream_head, attr):
                human_lora_count += set_lora_enabled(getattr(model.downstream_head, attr), False)
        disabled_lora["human_head"] = human_lora_count
        print("Strict original Human3R mode: disabled Movie3R adaptation branches.")
    if args.disable_v8_pose_prompt and hasattr(model, "enable_v8_pose_prompt"):
        model.enable_v8_pose_prompt = False
        print("V8/V9 pose prompt disabled for ablation.")
    if args.disable_v8_human_latent_corr and hasattr(model, "enable_v8_human_latent_corr"):
        model.enable_v8_human_latent_corr = False
        print("V8/V9 human latent correction disabled for ablation.")
    if args.disable_v8_human_trans_corr and hasattr(model, "enable_v8_human_trans_corr"):
        model.enable_v8_human_trans_corr = False
        print("V8/V9 human translation correction disabled for ablation.")
    if disabled_lora:
        print(f"Disabled LoRA branches for ablation: {disabled_lora}")
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
        freeze_state_after=args.freeze_state_after,
        freeze_state_feat_after=args.freeze_state_feat_after,
        freeze_pose_memory_after=args.freeze_pose_memory_after,
        freeze_v9_history_after=args.freeze_v9_history_after,
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
        "strict_original_human3r": bool(args.strict_original_human3r),
        "disable_v8_pose_prompt": bool(args.disable_v8_pose_prompt),
        "disable_v8_human_latent_corr": bool(args.disable_v8_human_latent_corr),
        "disable_v8_human_trans_corr": bool(args.disable_v8_human_trans_corr),
        "disable_v8_head_lora": bool(args.disable_v8_head_lora),
        "disable_v8_pose_head_lora": bool(args.disable_v8_pose_head_lora),
        "disable_v8_human_head_lora": bool(args.disable_v8_human_head_lora),
        "disabled_lora": disabled_lora,
    }
    with open(args.output_dir / "human3r_save_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
