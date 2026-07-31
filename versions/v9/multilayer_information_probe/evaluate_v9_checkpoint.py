#!/usr/bin/env python3
"""Evaluate the frozen formal V9 checkpoint on the probe's ten camera cuts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from run_probe import (
    REPO_ROOT,
    TEN_ROOT,
    camera_matrix_from_prediction,
    gt_camera,
    make_dataset,
    pose_error,
    prepare_batch,
    read_jsonl,
    record_key,
    relative_pose,
)


DEFAULT_CHECKPOINT = (
    REPO_ROOT / "checkpoints/v9_mixed_60h_pose_human_lora_bs10/checkpoint-final.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "output/v9_multilayer_information_probe/v9_checkpoint_eval.json",
    )
    return parser.parse_args()


def eval_records() -> list[dict]:
    records = []
    for source, filename in (
        ("avatarrex", "avatarrex.jsonl"),
        ("thuman", "thuman.jsonl"),
        ("mvhuman100", "mvhuman100.jsonl"),
        ("mvhuman200", "mvhuman200.jsonl"),
    ):
        for record in read_jsonl(TEN_ROOT / filename):
            item = dict(record)
            item["source"] = source
            records.append(item)
    return records


def summarize(cases: list[dict]) -> dict:
    result = {"count": len(cases), "per_case": cases}
    for metric in ("translation_m", "rotation_deg", "composite"):
        values = np.asarray([case[metric] for case in cases], dtype=np.float64)
        result[metric] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.9)),
        }
    return result


def configure_variant(model, variant: str) -> None:
    model.enable_v8_pose_prompt = variant != "no_prompt"
    model.enable_v8_human_latent_corr = variant != "no_human_corr"


def evaluate_variant(model, records: list[dict], variant: str, device: str) -> dict:
    configure_variant(model, variant)
    grouped = defaultdict(list)
    for record in records:
        grouped[record["source"]].append(record)

    datasets = {source: make_dataset(items, source, 20260731) for source, items in grouped.items()}
    indices = {
        (source, record_key(sample)): index
        for source, dataset in datasets.items()
        for index, sample in enumerate(dataset.samples)
    }
    cases = []
    for record in records:
        source = record["source"]
        key = (source, record_key(record))
        if key not in indices:
            raise RuntimeError(f"Frozen evaluation record is incomplete: {record['pattern_id']}")
        gt_batch, model_batch = prepare_batch(datasets[source], indices[key])
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            predictions, _ = model.forward_recurrent_lighter(
                model_batch, device, ret_state=False, use_ttt3r=False
            )
        target = relative_pose(gt_camera(gt_batch[-2]), gt_camera(gt_batch[-1]))
        estimate = relative_pose(
            camera_matrix_from_prediction(predictions[-2]),
            camera_matrix_from_prediction(predictions[-1]),
        )
        case = {
            "pattern_id": record["pattern_id"],
            "source": source,
            **pose_error(estimate, target),
        }
        cases.append(case)
        print(f"{variant:14s} {record['pattern_id']:36s} {case['composite']:.4f}", flush=True)
    return summarize(cases)


def main() -> None:
    args = parse_args()
    from dust3r.model import ARCroco3DStereo

    model = ARCroco3DStereo.from_pretrained(str(args.checkpoint)).to(args.device).float().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    flags = {
        name: getattr(model, name, None)
        for name in (
            "enable_v8_pose_prompt",
            "enable_v8_human_latent_corr",
            "enable_v8_head_lora",
            "v8_pose_prompt_use_decoder_token",
            "v8_pose_prompt_pooling",
        )
    }
    records = eval_records()
    report = {
        "checkpoint": str(args.checkpoint),
        "initial_flags": flags,
        "variants": {
            variant: evaluate_variant(model, records, variant, args.device)
            for variant in ("full", "no_human_corr", "no_prompt")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
