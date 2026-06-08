#!/usr/bin/env python3
"""Check whether V8 image-only forward is sample-local for batch_size > 1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from dust3r.datasets.avatarrex import AvatarReX_AABB, AvatarReX_Video
from dust3r.inference import _make_v8_image_only_model_batch
from dust3r.model import ARCroco3DStereo, ARCroco3DStereoConfig, inf, strip_module  # noqa: F401
from dust3r.smpl_model import SMPLModel
from dust3r.utils.device import to_cpu, todevice


RAW_ROOTS = {
    "lbn1": "/data/wangzheng/iJCV-CODE/data/AvatarReX_raw_meta/lbn1",
    "zzr": "/data/wangzheng/iJCV-CODE/data/AvatarReX_raw_meta/zzr",
}


def build_dataset(kind: str):
    root = "/data/wangzheng/iJCV-CODE/data"
    manifest_root = Path("/data/wangzheng/iJCV-CODE/Movie3R/output/v8_4_mixed_aabb_aaaa_manifests_no_zxc")
    common = dict(
        allow_repeat=True,
        split="Training",
        ROOT=root,
        aug_crop=0,
        resolution=(512, 288),
        num_views=4,
        n_corres=0,
        load_da3_depth=False,
        raw_calibration_root=RAW_ROOTS,
    )
    if kind == "aabb":
        return AvatarReX_AABB(
            seed=101,
            manifest_path=str(manifest_root / "train_aabb_no_zxc.jsonl"),
            **common,
        )
    if kind == "aaaa":
        return AvatarReX_Video(
            seed=102,
            manifest_path=str(manifest_root / "train_aaaa_no_zxc.jsonl"),
            **common,
        )
    raise ValueError(f"Unknown dataset kind: {kind}")


def collate_indices(dataset, indices, device):
    batch = default_collate([dataset[int(idx)] for idx in indices])
    return todevice(batch, device)


def load_model(config_path: str, device: torch.device):
    cfg = OmegaConf.load(config_path)
    model = eval(cfg.model)
    ckpt = torch.load(cfg.pretrained, map_location="cpu")
    model.load_state_dict(strip_module(ckpt["model"]), strict=False)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def forward_image_only(model, smpl_model, batch):
    smpl_model.update_smpl_gt(batch)
    model_batch = _make_v8_image_only_model_batch(batch)
    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        output = model(model_batch)
    return to_cpu(output.ress)


def tensor_max_abs(a, b):
    a = a.float()
    b = b.float()
    if a.numel() == 0 and b.numel() == 0:
        return 0.0
    return float((a - b).abs().max().item())


def human_counts(preds):
    counts = []
    for view in preds:
        scores = view.get("smpl_scores")
        if torch.is_tensor(scores):
            counts.append([int(x) for x in (scores.reshape(scores.shape[0], -1).amax(dim=1) > 0.3).cpu().tolist()])
        elif "smpl_loc" in view and torch.is_tensor(view["smpl_loc"]):
            loc = view["smpl_loc"]
            counts.append([int(loc.shape[1])] * int(loc.shape[0]))
        else:
            counts.append([])
    return counts


def compare_preds(single_preds, batched_preds, batch_idx: int, keys):
    diffs = {}
    for view_idx, (single_view, batch_view) in enumerate(zip(single_preds, batched_preds)):
        for key in keys:
            if key not in single_view or key not in batch_view:
                continue
            single_value = single_view[key]
            batch_value = batch_view[key]
            if not torch.is_tensor(single_value) or not torch.is_tensor(batch_value):
                continue
            if batch_value.shape[0] <= batch_idx:
                continue
            diff = tensor_max_abs(single_value[0:1], batch_value[batch_idx:batch_idx + 1])
            diffs[f"view{view_idx}.{key}"] = diff
    return diffs


def labels_from_batch(batch):
    labels = []
    for view in batch:
        label = view.get("label", [""])
        if isinstance(label, (list, tuple)):
            labels.append(list(label))
        else:
            labels.append([str(label)])
    return labels


def run_check(model, smpl_model, dataset, indices, device, keys):
    batched = collate_indices(dataset, indices, device)
    batched_preds = forward_image_only(model, smpl_model, batched)
    result = {
        "indices": [int(i) for i in indices],
        "batched_labels": labels_from_batch(to_cpu(batched)),
        "batched_detected_human_counts": human_counts(batched_preds),
        "samples": [],
    }

    for batch_idx, sample_idx in enumerate(indices):
        single = collate_indices(dataset, [sample_idx], device)
        single_preds = forward_image_only(model, smpl_model, single)
        diffs = compare_preds(single_preds, batched_preds, batch_idx, keys)
        result["samples"].append(
            {
                "sample_idx": int(sample_idx),
                "single_labels": labels_from_batch(to_cpu(single)),
                "single_detected_human_counts": human_counts(single_preds),
                "max_abs_diff": max(diffs.values()) if diffs else None,
                "diffs": diffs,
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/train_v8_4_mixed_no_zxc_bs10_long.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--indices", default="0,1")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(args.config, device)
    smpl_model = SMPLModel(
        device,
        model_args={
            "patch_size": model.croco_args["patch_size"],
            "mhmr_img_res": model.mhmr_img_res,
            "bb_patch_size": model.bb_patch_size,
        },
    )
    indices = [int(x) for x in args.indices.split(",") if x.strip()]
    keys = [
        "camera_pose",
        "v8_raw_camera_pose",
        "v8_pose_prompt_gate",
        "v8_pose_prompt_delta_applied",
        "v8_pose_prompt_delta_raw",
        "v8_pose_prompt_delta_norm",
        "v8_pose_prompt_drift_logit",
    ]

    results = {}
    for kind in ["aabb", "aaaa"]:
        dataset = build_dataset(kind)
        results[kind] = run_check(model, smpl_model, dataset, indices, device, keys)

    text = json.dumps(results, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
