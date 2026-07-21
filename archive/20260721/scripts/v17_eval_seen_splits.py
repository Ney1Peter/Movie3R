#!/usr/bin/env python3
"""Evaluate V17 checkpoints on train, unseen-pair validation, and held-out source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO_ROOT / "scripts"))

from v17_train_loso_fold import (  # noqa: E402
    Arrays,
    BridgeMLP,
    METHODS,
    SOURCES,
    choose_validation,
    decode,
    make_input_numpy,
    plain_world_features,
)


DEFAULT_CACHE = REPO_ROOT / "output" / "v17_direct_vs_factorized" / "feature_cache"
DEFAULT_LOSO = REPO_ROOT / "output" / "v17_direct_vs_factorized" / "loso"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "v17_direct_vs_factorized" / "evaluation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--loso_dir", type=Path, default=DEFAULT_LOSO)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


class LoadedScaler:
    def __init__(self, mean: np.ndarray, scale: np.ndarray):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)

    def numpy(self, values: np.ndarray) -> np.ndarray:
        return np.clip((values - self.mean) / self.scale, -10.0, 10.0).astype(np.float32)


def aggregate(rotation: np.ndarray, translation: np.ndarray, target: np.ndarray) -> dict:
    relative = rotation @ np.swapaxes(target[:, :3, :3], -1, -2)
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5, -1.0, 1.0)
    rotation_error = np.degrees(np.arccos(cosine))
    translation_error = np.linalg.norm(translation - target[:, :3, 3], axis=-1)
    return {
        "count": int(len(target)),
        "translation_mean_m": float(translation_error.mean()),
        "translation_p90_m": float(np.percentile(translation_error, 90)),
        "rotation_mean_deg": float(rotation_error.mean()),
        "rotation_p90_deg": float(np.percentile(rotation_error, 90)),
        "catastrophic_rate": float(np.mean((translation_error > 1.0) | (rotation_error > 30.0))),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.cache_dir / "v17_explicit_features.json").read_text(encoding="utf-8"))
    rows = metadata["rows"]
    with np.load(args.cache_dir / "v17_explicit_features.npz") as data:
        arrays = Arrays(
            invariant=data["invariant"].astype(np.float32),
            world=data["world_matrices"].astype(np.float32),
            stats=data["stats"].astype(np.float32),
            fixed=data["relative_fixed"].astype(np.float32),
            torso=data["relative_torso"].astype(np.float32),
            v15=data["relative_v15"].astype(np.float32),
            target=data["relative_target"].astype(np.float32),
        )
    world_flat = plain_world_features(arrays.world)
    output = {}
    for held_out in SOURCES:
        fold = json.loads((args.loso_dir / f"v17_loso_{held_out}.json").read_text(encoding="utf-8"))
        train_pool = np.asarray([index for index, row in enumerate(rows) if row["source"] != held_out], dtype=np.int64)
        train_ids, val_ids, _ = choose_validation(rows, train_pool, held_out)
        test_ids = np.asarray([index for index, row in enumerate(rows) if row["source"] == held_out], dtype=np.int64)
        split_ids = {"seen_train": train_ids, "unseen_pair_validation": val_ids, "held_out_source": test_ids}
        fold_output = {}
        for method in METHODS:
            checkpoint = torch.load(
                args.loso_dir / "checkpoints" / held_out / f"{method}.pt",
                map_location="cpu",
                weights_only=False,
            )
            state = checkpoint["state_dict"]
            input_dim = int(state["net.1.weight"].shape[1])
            hidden_dim = int(state["net.1.weight"].shape[0])
            output_dim = int(state["net.6.weight"].shape[0])
            model = BridgeMLP(input_dim, hidden_dim, output_dim)
            model.load_state_dict(state)
            model.eval()
            invariant_scaler = LoadedScaler(checkpoint["invariant_mean"], checkpoint["invariant_scale"])
            world_scaler = LoadedScaler(checkpoint["world_mean"], checkpoint["world_scale"])
            stats_scaler = LoadedScaler(checkpoint["stats_mean"], checkpoint["stats_scale"])
            inputs = make_input_numpy(
                method,
                arrays.invariant,
                world_flat,
                arrays.stats,
                invariant_scaler,
                world_scaler,
                stats_scaler,
            )
            method_output = {}
            for split, ids in split_ids.items():
                with torch.no_grad():
                    prediction = model(torch.as_tensor(inputs[ids], dtype=torch.float32))
                    rotation, translation, _ = decode(
                        method,
                        prediction,
                        torch.as_tensor(arrays.fixed[ids]),
                        torch.as_tensor(arrays.torso[ids]),
                        torch.as_tensor(arrays.v15[ids]),
                        argparse.Namespace(**fold["args"]),
                    )
                method_output[split] = aggregate(
                    rotation.numpy(), translation.numpy(), arrays.target[ids]
                )
            fold_output[method] = method_output
        output[held_out] = fold_output
    report = {
        "experiment": "V17 seen versus held-out split checkpoint evaluation",
        "folds": output,
    }
    path = args.output_dir / "v17_seen_split_eval.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f">> wrote {path}", flush=True)


if __name__ == "__main__":
    main()
