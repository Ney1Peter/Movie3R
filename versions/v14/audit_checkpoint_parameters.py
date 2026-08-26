#!/usr/bin/env python3
"""Audit the exact parameter scope encoded by the publication checkpoint.

The checkpoint stores the resolved model expression and its state dictionary.
This audit constructs that model on PyTorch's ``meta`` device, so it never
materialises the 4.6-GB tensors or executes an inference.  It then reports
the total and ``requires_grad`` parameter counts imposed by the checkpoint's
saved freeze configuration.  The checkpoint is a local, author-produced
artifact; loading its OmegaConf envelope with ``weights_only=False`` is
intentional and is not appropriate for an untrusted file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


SCHEMA = "Bridge3R-V14.1-checkpoint-parameter-audit-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def family(name: str) -> str:
    if name.startswith("downstream_head.pose_head.") and ".lora_" in name:
        return "pose_head_lora"
    if (
        name.startswith(
            (
                "downstream_head.deccam.",
                "downstream_head.decpose.",
                "downstream_head.decshape.",
                "downstream_head.decexpression.",
            )
        )
        and ".lora_" in name
    ):
        return "human_head_lora"
    if name.startswith("v8_pose_prompt."):
        return "correction_token_pathway"
    if name.startswith("v8_pose_residual_head."):
        return "camera_correction_head"
    if name.startswith("v8_human_latent_corr_head."):
        return "human_latent_correction_head"
    return "other_trainable"


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))
    from dust3r.model import ARCroco3DStereo, ARCroco3DStereoConfig

    # The checkpoint was produced within this workspace and its model
    # expression is a resolved ARCroco3DStereo configuration. Mapping tensors
    # to meta and memory-mapping the archive prevents a large RAM allocation.
    checkpoint_payload = torch.load(
        checkpoint, map_location="meta", weights_only=False, mmap=True
    )
    if not isinstance(checkpoint_payload, dict):
        raise TypeError("checkpoint envelope is not a dictionary")
    model_expression = checkpoint_payload.get("args", {}).get("model")
    if not isinstance(model_expression, str) or not model_expression.startswith(
        "ARCroco3DStereo(ARCroco3DStereoConfig("
    ):
        raise ValueError("checkpoint does not contain the expected resolved model expression")
    with torch.device("meta"):
        model = eval(  # noqa: S307 -- restricted to author-owned model expression above.
            model_expression,
            {
                "ARCroco3DStereo": ARCroco3DStereo,
                "ARCroco3DStereoConfig": ARCroco3DStereoConfig,
                "inf": float("inf"),
            },
        )

    total = 0
    trainable = 0
    by_family: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"parameter_tensors": 0, "parameters": 0, "names": []}
    )
    for name, parameter in model.named_parameters():
        count = int(parameter.numel())
        total += count
        if parameter.requires_grad:
            trainable += count
            group = by_family[family(name)]
            group["parameter_tensors"] += 1
            group["parameters"] += count
            group["names"].append(name)

    state_dict = checkpoint_payload.get("model")
    if not isinstance(state_dict, dict):
        raise TypeError("checkpoint model state is not a dictionary")
    state_tensor_count = sum(1 for value in state_dict.values() if torch.is_tensor(value))
    state_numel = sum(
        int(value.numel()) for value in state_dict.values() if torch.is_tensor(value)
    )
    config_attribute = {
        "freeze": "freeze",
        "v8_pose_prompt_variant": "v8_pose_prompt_variant",
        "v8_human_latent_corr": "v8_human_latent_corr",
        "v8_pose_head_lora": "v8_pose_head_lora_enabled",
        "v8_human_head_lora": "v8_human_head_lora_enabled",
        "v8_head_lora_rank": "v8_head_lora_rank",
        "v14_1_event_only_head_lora": "v14_1_event_only_head_lora",
        "v14_1_freeze_unused_prompt_params": "v14_1_freeze_unused_prompt_params",
    }
    config_flags = {
        key: getattr(model, attribute) for key, attribute in config_attribute.items()
    }
    payload = {
        "schema_version": SCHEMA,
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": int(checkpoint.stat().st_size),
        "checkpoint_sha256": sha256(checkpoint),
        "construction": {
            "model_constructed_on": "meta",
            "inference_executed": False,
            "checkpoint_tensor_values_materialized": False,
            "saved_model_expression": model_expression,
            "saved_config_flags": config_flags,
        },
        "model_parameters": {
            "total": total,
            "trainable": trainable,
            "trainable_fraction": trainable / total,
            "trainable_tensor_count": sum(
                group["parameter_tensors"] for group in by_family.values()
            ),
            "trainable_by_family": dict(sorted(by_family.items())),
        },
        "checkpoint_state": {
            "tensor_count": state_tensor_count,
            "tensor_numel": state_numel,
            "non_parameter_tensor_numel": state_numel - total,
        },
        "interpretation": (
            "This is the exact parameter scope imposed by the saved V14.1 "
            "freeze configuration. It does not by itself establish training/"
            "benchmark clip disjointness."
        ),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    concise = {
        "output": str(output),
        "checkpoint_sha256": payload["checkpoint_sha256"],
        "total": total,
        "trainable": trainable,
        "trainable_fraction": trainable / total,
        "families": {
            key: value["parameters"] for key, value in payload["model_parameters"]["trainable_by_family"].items()
        },
    }
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()
