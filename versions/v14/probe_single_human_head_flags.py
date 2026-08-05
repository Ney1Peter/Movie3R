#!/usr/bin/env python3
"""CPU-only singleton ablation for the current human prediction branch.

The probe keeps the current event/B0 camera transaction and reports the
post-human image bounding box after each inference-time flag setting.  It is
intended to identify whether the singleton failure is introduced by the
factorized V8 human latent correction or by the V8 human head LoRA.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from versions.v13 import gt_id_consensus as gt  # noqa: E402
from versions.v14.probe_p1_foot_scene_observability import (  # noqa: E402
    configure_model,
    decode_people,
    set_event_indices,
    transform_person,
)
from versions.v14.run_v14_2_single_sequence import camera_matrix  # noqa: E402


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pre-camera", default="22070935")
    p.add_argument("--post-camera", default="22053912")
    p.add_argument("--frame", type=int, default=1836)
    p.add_argument("--pre-frames", type=int, default=5)
    p.add_argument("--size", type=int, default=512)
    return p.parse_args()


def paths(a: argparse.Namespace) -> tuple[list[Path], list[Path]]:
    root = Path("/data/wangzheng/iJCV-CODE/data/Training/lbn1")
    pre = [root / a.pre_camera / "rgb" / f"{i:08d}.png" for i in range(a.frame - a.pre_frames + 1, a.frame + 1)]
    post = [root / a.post_camera / "rgb" / f"{a.frame:08d}.png"]
    if any(not p.is_file() for p in pre + post):
        raise FileNotFoundError([str(p) for p in pre + post if not p.is_file()])
    return pre, post


def run_variant(model, layer, pre: list[Path], post: list[Path], size: int, flags: dict) -> dict:
    # Restore the frozen runtime defaults before every ablation; otherwise a
    # previous variant's disabled flag would silently leak into the next one.
    configure_model(model)
    for key, value in flags.items():
        setattr(model, key, bool(value))
    shadow_views = set_event_indices(
        gt.prepare_full_square_input(model, pre + post, SimpleNamespace(size=size)), {len(pre)}
    )
    raw_views = set_event_indices(
        gt.prepare_full_square_input(model, post, SimpleNamespace(size=size)), set()
    )
    with torch.no_grad():
        shadow, shadow_returned, shadow_debug = model.forward_recurrent_lighter(
            shadow_views, "cpu", ret_state=False, use_ttt3r=False, return_token_debug=True
        )
        raw, raw_returned, raw_debug = model.forward_recurrent_lighter(
            raw_views, "cpu", ret_state=False, use_ttt3r=False, return_token_debug=True
        )
    b0 = camera_matrix(shadow[-1]).astype(np.float64) @ np.linalg.inv(camera_matrix(raw[0]).astype(np.float64))
    people = decode_people(raw[0], raw_returned[0], raw_debug[0], layer)
    transformed = [transform_person(b0, person) for person in people]
    camera = camera_matrix(shadow[-1]).astype(np.float64)
    K = shadow_returned[-1]["K_mhmr"][0].detach().cpu().numpy().astype(np.float64)
    rows = []
    for person in transformed:
        xyz = (camera[:3, :3].T @ (person["vertices"] - camera[:3, 3]).T).T
        valid = xyz[:, 2] > 0.05
        uv = xyz[valid, :2] / xyz[valid, 2:3] * K.diagonal()[:2] + K[:2, 2]
        rows.append({
            "bbox_uv": [float(v) for v in np.r_[uv.min(0), uv.max(0)]],
            "center_world": np.asarray(person["vertices"]).mean(0).tolist(),
            "positive_depth_fraction": float(valid.mean()),
        })
    return {
        "flags": {k: bool(v) for k, v in flags.items()},
        "b0_camera": camera.tolist(),
        "people": rows,
    }


def main() -> None:
    a = args()
    from dust3r.model import ARCroco3DStereo
    from dust3r.utils.smpl_layer import SMPL_Layer

    pre, post = paths(a)
    model = ARCroco3DStereo.from_pretrained(str(a.checkpoint)).to("cpu")
    configure_model(model)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layer = SMPL_Layer(type="smplx", gender="neutral", num_betas=10, kid=False, person_center="head").to("cpu").eval()
    variants = {
        "current": {},
        "no_human_latent_corr": {"enable_v8_human_latent_corr": False},
        "no_human_head_lora": {"enable_v8_head_lora": False},
        "no_latent_no_head_lora": {"enable_v8_human_latent_corr": False, "enable_v8_head_lora": False},
        "no_pose_prompt": {"enable_v8_pose_prompt": False},
        "no_pose_prompt_no_human_head_lora": {"enable_v8_pose_prompt": False, "enable_v8_head_lora": False},
    }
    report = {name: run_variant(model, layer, pre, post, int(a.size), flags) for name, flags in variants.items()}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
