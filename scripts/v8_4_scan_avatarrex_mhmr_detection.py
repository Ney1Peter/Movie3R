#!/usr/bin/env python3
"""Scan AvatarReX clips for frozen Human3R/MHMR human-token detection.

The GT SMPL/mask files can be valid while the frozen human-token branch still
misses a person after Human3R resizing/cropping. This script runs image-only
forward at batch size 1 and reports clips whose four frames do not all produce
at least one detected human.
"""

from __future__ import annotations

import argparse
import json
import random
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


def build_dataset(kind: str, manifest_root: Path):
    common = dict(
        allow_repeat=True,
        split="Training",
        ROOT="/data/wangzheng/iJCV-CODE/data",
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


def load_model(config_path: str, device: torch.device):
    print(f"[model] loading config: {config_path}", flush=True)
    cfg = OmegaConf.load(config_path)
    print("[model] instantiating model", flush=True)
    model = eval(cfg.model)
    print(f"[model] loading checkpoint: {cfg.pretrained}", flush=True)
    ckpt = torch.load(cfg.pretrained, map_location="cpu")
    print("[model] loading state dict", flush=True)
    model.load_state_dict(strip_module(ckpt["model"]), strict=False)
    print(f"[model] moving to device: {device}", flush=True)
    model.to(device)
    model.eval()
    print("[model] ready", flush=True)
    return model


def sample_indices(num_items: int, limit: int, seed: int) -> list[int]:
    if limit <= 0 or limit >= num_items:
        return list(range(num_items))
    rng = random.Random(seed)
    return sorted(rng.sample(range(num_items), limit))


def labels_from_batch(batch) -> list[str]:
    labels = []
    for view in batch:
        label = view.get("label", [""])
        if isinstance(label, (list, tuple)):
            labels.append(str(label[0]) if label else "")
        else:
            labels.append(str(label))
    return labels


def group_from_labels(labels: list[str]) -> str:
    if not labels:
        return "unknown"
    first = labels[0]
    return first.split("/", 1)[0] if "/" in first else "unknown"


def count_detected_humans(pred_view: dict, score_threshold: float) -> int:
    scores = pred_view.get("smpl_scores")
    if torch.is_tensor(scores):
        scores = scores.detach().float().cpu()
        if scores.ndim == 0:
            return int(float(scores) > score_threshold)
        if scores.ndim == 1:
            return int((scores > score_threshold).sum().item())
        per_human = scores.reshape(scores.shape[0], -1).amax(dim=1)
        return int((per_human > score_threshold).sum().item())
    loc = pred_view.get("smpl_loc")
    if torch.is_tensor(loc) and loc.ndim >= 2:
        return int(loc.shape[1])
    return 0


@torch.no_grad()
def scan_dataset(
    *,
    model,
    smpl_model,
    dataset,
    kind: str,
    indices: list[int],
    device: torch.device,
    score_threshold: float,
    max_bad_examples: int,
    progress_interval: int,
) -> dict:
    stats = {
        "kind": kind,
        "clips_scanned": 0,
        "frames_scanned": 0,
        "clips_all_frames_detected": 0,
        "clips_any_frame_missing": 0,
        "frames_missing_human": 0,
        "groups": {},
        "bad_examples": [],
    }

    print(f"[{kind}] scanning {len(indices)} clips", flush=True)
    for scan_pos, item_idx in enumerate(indices):
        if scan_pos == 0 or (progress_interval > 0 and scan_pos % progress_interval == 0):
            print(f"[{kind}] {scan_pos}/{len(indices)} idx={item_idx}", flush=True)
        batch = default_collate([dataset[item_idx]])
        labels = labels_from_batch(batch)
        group = group_from_labels(labels)
        group_stats = stats["groups"].setdefault(
            group,
            {
                "clips_scanned": 0,
                "frames_scanned": 0,
                "clips_any_frame_missing": 0,
                "frames_missing_human": 0,
            },
        )
        batch = todevice(batch, device)
        smpl_model.update_smpl_gt(batch)
        model_batch = _make_v8_image_only_model_batch(batch)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            preds = model(model_batch).ress

        counts = [count_detected_humans(view, score_threshold) for view in preds]
        missing = [count <= 0 for count in counts]
        stats["clips_scanned"] += 1
        stats["frames_scanned"] += len(counts)
        stats["frames_missing_human"] += int(sum(missing))
        group_stats["clips_scanned"] += 1
        group_stats["frames_scanned"] += len(counts)
        group_stats["frames_missing_human"] += int(sum(missing))
        if any(missing):
            stats["clips_any_frame_missing"] += 1
            group_stats["clips_any_frame_missing"] += 1
            if len(stats["bad_examples"]) < max_bad_examples:
                meta = dataset.get_sample_metadata(item_idx) if hasattr(dataset, "get_sample_metadata") else {}
                stats["bad_examples"].append(
                    {
                        "dataset_index": int(item_idx),
                        "metadata": meta,
                        "group": group,
                        "labels": labels,
                        "detected_counts": counts,
                        "missing_frames": [int(i) for i, flag in enumerate(missing) if flag],
                    }
                )
        else:
            stats["clips_all_frames_detected"] += 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(
        f"[{kind}] done: missing_clip_ratio={stats['clips_any_frame_missing'] / max(stats['clips_scanned'], 1):.4f}",
        flush=True,
    )
    denom_clips = max(stats["clips_scanned"], 1)
    denom_frames = max(stats["frames_scanned"], 1)
    stats["clips_all_frames_detected_ratio"] = stats["clips_all_frames_detected"] / denom_clips
    stats["clips_any_frame_missing_ratio"] = stats["clips_any_frame_missing"] / denom_clips
    stats["frames_missing_human_ratio"] = stats["frames_missing_human"] / denom_frames
    for group_stats in stats["groups"].values():
        group_stats["clips_any_frame_missing_ratio"] = (
            group_stats["clips_any_frame_missing"] / max(group_stats["clips_scanned"], 1)
        )
        group_stats["frames_missing_human_ratio"] = (
            group_stats["frames_missing_human"] / max(group_stats["frames_scanned"], 1)
        )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/train_v8_4_mixed_no_zxc_bs10_long.yaml")
    parser.add_argument(
        "--manifest_root",
        default="output/v8_4_mixed_aabb_aaaa_manifests_no_zxc",
    )
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score_threshold", type=float, default=0.3)
    parser.add_argument("--max_bad_examples", type=int, default=50)
    parser.add_argument("--progress_interval", type=int, default=20)
    parser.add_argument("--output", default="output/v8_4_batch_probe/avatarrex_mhmr_detection_scan.json")
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

    results = []
    manifest_root = Path(args.manifest_root)
    for offset, kind in enumerate(["aabb", "aaaa"]):
        dataset = build_dataset(kind, manifest_root)
        indices = sample_indices(len(dataset), args.limit, args.seed + offset)
        results.append(
            scan_dataset(
                model=model,
                smpl_model=smpl_model,
                dataset=dataset,
                kind=kind,
                indices=indices,
                device=device,
                score_threshold=args.score_threshold,
                max_bad_examples=args.max_bad_examples,
                progress_interval=args.progress_interval,
            )
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
