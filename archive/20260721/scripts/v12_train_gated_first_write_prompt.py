#!/usr/bin/env python3
"""Train V12 gated/ungated/no-old first-write adapters from Oracle caches."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from dust3r.v12_gated_gauge_neutral_prompt import GatedGaugeNeutralFirstWritePrompt  # noqa: E402


DEFAULT_CACHE = REPO_ROOT / "output" / "v12_gated_first_write" / "teacher_cache_loso_mvhuman200"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "v12_gated_first_write" / "training_loso_mvhuman200"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--variant", choices=("gated", "ungated", "no_old"), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=192)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


class PairDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        pair = torch.load(self.rows[index]["path"], map_location="cpu", weights_only=False)
        labels = pair["labels"]
        return {
            "old_state": pair["old_state"],
            "fresh_state": pair["fresh_state"],
            "oracle_residual": pair["oracle_residual"],
            "image_summary": pair["image_summary"],
            "human_summary": pair["human_summary"],
            "camera_token": pair["camera_token"],
            "memory_summary": pair["memory_summary"],
            "diagnostics": pair["diagnostics"],
            "gate_target": torch.tensor(labels["gate_target"], dtype=torch.float32),
            "gain_target": torch.tensor(labels["gain_target"], dtype=torch.float32),
            "wait_target": torch.tensor(labels["wait_target"], dtype=torch.float32),
        }


def to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device=device, dtype=torch.float32) for key, value in batch.items()}


def adapter_inputs(batch: dict, variant: str) -> tuple:
    old_state = batch["old_state"]
    memory_summary = batch["memory_summary"]
    if variant == "no_old":
        old_state = torch.zeros_like(old_state)
        memory_summary = torch.zeros_like(memory_summary)
    return (
        old_state,
        batch["fresh_state"],
        batch["image_summary"],
        batch["human_summary"],
        batch["camera_token"],
        memory_summary,
        batch["diagnostics"],
    )


def loss_terms(model: nn.Module, batch: dict, variant: str) -> tuple[torch.Tensor, dict]:
    gate_override = 1.0 if variant == "ungated" else None
    output = model(*adapter_inputs(batch, variant), gate_override=gate_override)
    gate_target = torch.ones_like(batch["gate_target"]) if variant == "ungated" else batch["gate_target"]
    desired = batch["fresh_state"] + gate_target[:, None, None] * batch["oracle_residual"]
    residual_energy = batch["oracle_residual"].square().mean(dim=(1, 2)).clamp_min(1e-4)
    latent = ((output.corrected_state - desired).square().mean(dim=(1, 2)) / residual_energy).mean()
    positive = batch["gate_target"].clamp(0.0, 1.0)
    residual_scale = batch["fresh_state"].std(dim=(1, 2), unbiased=False).clamp_min(1e-4)
    residual_error = F.smooth_l1_loss(
        output.bounded_residual / residual_scale[:, None, None],
        batch["oracle_residual"] / residual_scale[:, None, None],
        reduction="none",
    ).mean(dim=(1, 2))
    residual = (residual_error * positive).sum() / positive.sum().clamp_min(1.0)
    if variant == "ungated":
        gate = latent.new_zeros(())
    else:
        gate = F.mse_loss(output.gate, batch["gate_target"])
    gain = F.smooth_l1_loss(output.predicted_gain, batch["gain_target"].clamp(-1.0, 1.0))
    wait = F.binary_cross_entropy(output.wait_score, batch["wait_target"])
    identity_weight = (batch["gate_target"] < 0.05).float()
    identity = (
        output.corrected_state.sub(batch["fresh_state"]).square().mean(dim=(1, 2))
        / residual_energy
    )
    identity = (identity * identity_weight).sum() / identity_weight.sum().clamp_min(1.0)
    total = latent + 0.20 * residual + 0.50 * gate + 0.10 * gain + 0.05 * wait + 0.20 * identity
    return total, {
        "latent": float(latent.detach()),
        "residual": float(residual.detach()),
        "gate": float(gate.detach()),
        "gain": float(gain.detach()),
        "wait": float(wait.detach()),
        "identity": float(identity.detach()),
        "mean_gate": float(output.gate.mean().detach()),
    }


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    positive = labels > 0.5
    negative = ~positive
    if not positive.any() or not negative.any():
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[positive].sum() - positive.sum() * (positive.sum() + 1) / 2) / (positive.sum() * negative.sum()))


def calibration_error(scores: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    error = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        mask = (scores >= low) & (scores < high if index + 1 < bins else scores <= high)
        if mask.any():
            error += mask.mean() * abs(scores[mask].mean() - labels[mask].mean())
    return float(error)


def evaluate(model: nn.Module, rows: list[dict], variant: str, device: torch.device) -> dict:
    loader = DataLoader(PairDataset(rows), batch_size=1, shuffle=False, num_workers=0)
    baseline, predicted, gates, targets, gains, gain_targets = [], [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            output = model(
                *adapter_inputs(batch, variant),
                gate_override=1.0 if variant == "ungated" else None,
            )
            gate_target = torch.ones_like(batch["gate_target"]) if variant == "ungated" else batch["gate_target"]
            desired = batch["fresh_state"] + gate_target[:, None, None] * batch["oracle_residual"]
            baseline.append((batch["fresh_state"] - desired).square().mean().sqrt())
            predicted.append((output.corrected_state - desired).square().mean().sqrt())
            gates.append(output.gate)
            targets.append(batch["gate_target"])
            gains.append(output.predicted_gain)
            gain_targets.append(batch["gain_target"])
    baseline_value = float(torch.stack(baseline).mean())
    predicted_value = float(torch.stack(predicted).mean())
    gate_array = torch.cat(gates).cpu().numpy()
    target_array = torch.cat(targets).cpu().numpy()
    gain_array = torch.cat(gains).cpu().numpy()
    gain_target_array = torch.cat(gain_targets).cpu().numpy()
    return {
        "cases": len(rows),
        "baseline_target_rmse": baseline_value,
        "predicted_target_rmse": predicted_value,
        "latent_recovery": 1.0 - predicted_value / max(baseline_value, 1e-8),
        "mean_gate": float(gate_array.mean()),
        "gate_target_mean": float(target_array.mean()),
        "difficulty_auroc": auroc(gate_array, target_array > 0.05),
        "gate_calibration_error": calibration_error(gate_array, target_array),
        "gate_gain_correlation": float(np.corrcoef(gate_array, gain_target_array)[0, 1]),
        "predicted_gain_correlation": float(np.corrcoef(gain_array, gain_target_array)[0, 1]),
    }


def main() -> None:
    args = parse_args()
    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("V12 adapter training requires CUDA")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    train_rows = json.loads((args.cache_dir / "index_train.json").read_text(encoding="utf-8"))
    val_rows = json.loads((args.cache_dir / "index_validation.json").read_text(encoding="utf-8"))
    if not val_rows:
        val_rows = train_rows
    checkpoint = args.output_dir / "checkpoints" / f"{args.variant}_distill.pth"
    if checkpoint.is_file() and not args.overwrite:
        print(f">> exists {checkpoint}")
        return
    loader = DataLoader(
        PairDataset(train_rows),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    model = GatedGaugeNeutralFirstWritePrompt(hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    iterator = iter(loader)
    history = []
    model.train()
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss, terms = loss_terms(model, batch, args.variant)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            row = {"step": step, "loss": float(loss.detach()), **terms}
            history.append(row)
            print(f">> {args.variant} {row}", flush=True)
    report = {
        "variant": args.variant,
        "steps": args.steps,
        "train_cases": len(train_rows),
        "validation_cases": len(val_rows),
        "train": evaluate(model, train_rows, args.variant, device),
        "validation": evaluate(model, val_rows, args.variant, device),
        "history": history,
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "report": report, "args": vars(args)}, checkpoint)
    report_path = args.output_dir / f"{args.variant}_distill_report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    print(f">> wrote {checkpoint}", flush=True)


if __name__ == "__main__":
    main()
