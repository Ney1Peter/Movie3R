#!/usr/bin/env python3
"""Diagnose whether V9 branches disturb frame-0/frame-1 human alignment."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from add_ckpt_path import add_path_to_dust3r  # noqa: E402
from demo import parse_seq_path, prepare_input  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq_path", type=Path, required=True)
    parser.add_argument("--v9_checkpoint", type=Path, required=True)
    parser.add_argument("--original_checkpoint", type=Path, default=REPO_ROOT / "src/human3r_896L.pth")
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--variants", nargs="*", default=None)
    return parser.parse_args()


def rot_angle_deg(r0: torch.Tensor, r1: torch.Tensor) -> float:
    rel = r0.transpose(-1, -2) @ r1
    tr = torch.diagonal(rel, dim1=-2, dim2=-1).sum()
    cos = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0)
    return float(torch.rad2deg(torch.acos(cos)).item())


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.detach().float().mean().cpu().item())
    try:
        return float(value)
    except Exception:
        return None


def set_head_lora(model, *, pose: bool | None = None, human: bool | None = None) -> dict[str, int]:
    from src.dust3r.v8_head_lora import set_lora_enabled

    counts: dict[str, int] = {}
    if pose is not None and hasattr(model.downstream_head, "pose_head"):
        counts["pose_head"] = set_lora_enabled(model.downstream_head.pose_head, pose)
    if human is not None:
        n = 0
        for attr in ("deccam", "decpose", "decshape", "decexpression"):
            if hasattr(model.downstream_head, attr):
                n += set_lora_enabled(getattr(model.downstream_head, attr), human)
        counts["human_head"] = n
    return counts


def apply_variant(model, variant: str) -> dict[str, Any]:
    info: dict[str, Any] = {}
    if variant == "v9_disable_human_corr":
        if hasattr(model, "enable_v8_human_latent_corr"):
            model.enable_v8_human_latent_corr = False
        if hasattr(model, "enable_v8_human_trans_corr"):
            model.enable_v8_human_trans_corr = False
    elif variant == "v9_disable_pose_prompt":
        if hasattr(model, "enable_v8_pose_prompt"):
            model.enable_v8_pose_prompt = False
    elif variant == "v9_disable_pose_head_lora":
        info["disabled_lora"] = set_head_lora(model, pose=False)
    elif variant == "v9_disable_human_head_lora":
        info["disabled_lora"] = set_head_lora(model, human=False)
    elif variant == "v9_disable_all_head_lora":
        info["disabled_lora"] = set_head_lora(model, pose=False, human=False)
    elif variant == "v9_strict_original_from_v9":
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
        info["disabled_lora"] = set_head_lora(model, pose=False, human=False)
    return info


def first_human_transl(pred: dict[str, Any], key: str = "smpl_transl") -> torch.Tensor | None:
    value = pred.get(key)
    if value is None:
        return None
    value = value.detach().float().cpu()
    if value.ndim == 3:
        value = value[0]
    if value.numel() == 0 or value.shape[0] == 0:
        return None
    return value[0, :3]


def run_one(
    *,
    model_path: Path,
    variant: str,
    seq_path: Path,
    img_paths: list[str],
    size: int,
    device: torch.device,
) -> dict[str, Any]:
    add_path_to_dust3r(str(model_path))
    from src.dust3r.inference import inference_recurrent_lighter
    from src.dust3r.model import ARCroco3DStereo
    from src.dust3r.utils.camera import pose_encoding_to_camera

    model = ARCroco3DStereo.from_pretrained(str(model_path)).to(device)
    variant_info = apply_variant(model, variant)
    model.eval()

    views = prepare_input(
        img_paths=img_paths,
        img_mask=[True] * len(img_paths),
        size=size,
        revisit=1,
        update=True,
        img_res=getattr(model, "mhmr_img_res", None),
        reset_interval=10000000,
    )
    with torch.no_grad():
        outputs, _ = inference_recurrent_lighter(views, model, str(device), verbose=False)

    preds = outputs["pred"]
    poses = torch.cat(
        [pose_encoding_to_camera(pred["camera_pose"].detach().cpu()) for pred in preds],
        dim=0,
    )
    local_trans = [first_human_transl(pred) for pred in preds]
    raw_local_trans = [first_human_transl(pred, "v8_human_latent_corr_smpl_transl_raw") for pred in preds]

    def world_trans(local: torch.Tensor | None, idx: int) -> torch.Tensor | None:
        if local is None:
            return None
        r = poses[idx, :3, :3]
        t = poses[idx, :3, 3]
        return r @ local + t

    world = [world_trans(local_trans[i], i) for i in range(len(preds))]
    raw_world = [world_trans(raw_local_trans[i], i) for i in range(len(preds))]

    def point_diff(points: list[torch.Tensor | None], i: int, j: int) -> float | None:
        if i >= len(points) or j >= len(points) or points[i] is None or points[j] is None:
            return None
        return float(torch.linalg.norm(points[i] - points[j]).item())

    def local_diff(points: list[torch.Tensor | None], i: int, j: int) -> float | None:
        return point_diff(points, i, j)

    pairs: dict[str, Any] = {}
    for name, i, j in (("f0_f1", 0, 1), ("f2_f3", 2, 3), ("f0_f2", 0, 2)):
        if j >= len(preds):
            continue
        pairs[name] = {
            "cam_trans_m": float(torch.linalg.norm(poses[i, :3, 3] - poses[j, :3, 3]).item()),
            "cam_rot_deg": rot_angle_deg(poses[i, :3, :3], poses[j, :3, :3]),
            "human_local_trans_m": local_diff(local_trans, i, j),
            "human_world_trans_m": point_diff(world, i, j),
            "raw_human_world_trans_m": point_diff(raw_world, i, j),
        }

    gate_values = [
        to_float(pred.get("v8_pose_prompt_gate"))
        for pred in preds
        if pred.get("v8_pose_prompt_gate") is not None
    ]
    delta_values = [
        to_float(pred.get("v8_pose_prompt_delta_norm"))
        for pred in preds
        if pred.get("v8_pose_prompt_delta_norm") is not None
    ]
    human_delta_values = [
        to_float(pred.get("v8_human_latent_corr_delta_norm"))
        for pred in preds
        if pred.get("v8_human_latent_corr_delta_norm") is not None
    ]

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "variant": variant,
        "model_path": str(model_path),
        "seq_path": str(seq_path),
        "num_frames": len(preds),
        "pairs": pairs,
        "v8_pose_gate_mean": sum(gate_values) / len(gate_values) if gate_values else None,
        "v8_pose_delta_norm_mean": sum(delta_values) / len(delta_values) if delta_values else None,
        "v8_human_delta_norm_mean": sum(human_delta_values) / len(human_delta_values) if human_delta_values else None,
        **variant_info,
    }


def main() -> None:
    args = parse_args()
    output_json = args.output_json
    output_json.parent.mkdir(parents=True, exist_ok=True)
    md_path = output_json.with_suffix(".md")
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    img_paths, tmpdirname = parse_seq_path(str(args.seq_path))
    try:
        img_paths = img_paths[:4]
        default_variants = [
            "original_human3r",
            "v9_full",
            "v9_disable_human_corr",
            "v9_disable_pose_prompt",
            "v9_disable_pose_head_lora",
            "v9_disable_human_head_lora",
            "v9_disable_all_head_lora",
            "v9_strict_original_from_v9",
        ]
        variants = args.variants or default_variants
        results = []

        def write_summary() -> None:
            summary = {
                "seq_path": str(args.seq_path),
                "v9_checkpoint": str(args.v9_checkpoint),
                "original_checkpoint": str(args.original_checkpoint),
                "completed_variants": [item["variant"] for item in results],
                "results": results,
            }
            output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            lines = [
                "# V9 Frame 1/2 Instability Diagnosis",
                "",
                f"- Sequence: `{args.seq_path}`",
                f"- V9 checkpoint: `{args.v9_checkpoint}`",
                f"- Completed variants: {len(results)}/{len(variants)}",
                "",
                "| Variant | cam 0-1 m | cam 0-1 deg | human world 0-1 m | human local 0-1 m | cam 2-3 m | human world 2-3 m |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
            for item in results:
                p01 = item["pairs"].get("f0_f1", {})
                p23 = item["pairs"].get("f2_f3", {})

                def fmt(v):
                    if v is None:
                        return "-"
                    if isinstance(v, float) and math.isnan(v):
                        return "-"
                    return f"{v:.4f}"

                lines.append(
                    "| {variant} | {cam01} | {rot01} | {hw01} | {hl01} | {cam23} | {hw23} |".format(
                        variant=item["variant"],
                        cam01=fmt(p01.get("cam_trans_m")),
                        rot01=fmt(p01.get("cam_rot_deg")),
                        hw01=fmt(p01.get("human_world_trans_m")),
                        hl01=fmt(p01.get("human_local_trans_m")),
                        cam23=fmt(p23.get("cam_trans_m")),
                        hw23=fmt(p23.get("human_world_trans_m")),
                    )
                )
            md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        for variant in variants:
            model_path = args.original_checkpoint if variant == "original_human3r" else args.v9_checkpoint
            print(f"[diagnose] running {variant} on {device}", flush=True)
            results.append(
                run_one(
                    model_path=model_path,
                    variant=variant,
                    seq_path=args.seq_path,
                    img_paths=img_paths,
                    size=args.size,
                    device=device,
                )
            )
            write_summary()
        print(json.dumps({"output_json": str(output_json), "output_md": str(md_path)}, indent=2))
    finally:
        if tmpdirname is not None:
            shutil.rmtree(tmpdirname)


if __name__ == "__main__":
    main()
