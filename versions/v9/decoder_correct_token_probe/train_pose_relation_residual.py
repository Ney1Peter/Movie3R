#!/usr/bin/env python3
"""Train an isolated final-decoder pose-relation residual head."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

from run_token_depth_probe import REPO_ROOT, pose_error


DEFAULT_CACHE = REPO_ROOT / "output/v9_decoder_correct_token_probe/full96/token_cache.pt"
DEFAULT_OUTPUT = REPO_ROOT / "output/v9_decoder_correct_token_probe/pose_relation_head"
TOKEN_DIM = 768
STAT_DIM = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", choices=("single", "small", "full"), required=True)
    parser.add_argument("--architecture", choices=("structured", "flat"), default="structured")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--rank", type=int, default=48)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--translation-bound", type=float, default=5.0)
    return parser.parse_args()


def relative_residual(row: dict) -> np.ndarray:
    return row["gt_relative"].numpy() @ np.linalg.inv(row["full_relative"].numpy())


def matrix_to_twist(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [Rotation.from_matrix(matrix[:3, :3]).as_rotvec(), matrix[:3, 3]], axis=0
    ).astype(np.float32)


def twist_to_matrix(twist: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = Rotation.from_rotvec(twist[:3]).as_matrix().astype(np.float32)
    matrix[:3, 3] = twist[3:]
    return matrix


def matrix_context(row: dict) -> np.ndarray:
    raw = row["raw_relative"].numpy()
    full = row["full_relative"].numpy()
    formal_delta = full @ np.linalg.inv(raw)
    values = []
    for matrix in (raw, full, formal_delta):
        values.extend(Rotation.from_matrix(matrix[:3, :3]).as_rotvec().tolist())
        values.extend(matrix[:3, 3].tolist())
    return np.asarray(values, dtype=np.float32)


def split_descriptor(row: dict) -> tuple[np.ndarray, ...]:
    descriptor = row["descriptors"]["decoder_l11_pose"].numpy().astype(np.float32)
    expected = 4 * TOKEN_DIM + STAT_DIM
    if descriptor.shape != (expected,):
        raise ValueError(f"Unexpected L11 pose descriptor shape: {descriptor.shape}")
    return (
        descriptor[0:TOKEN_DIM],
        descriptor[TOKEN_DIM:2 * TOKEN_DIM],
        descriptor[2 * TOKEN_DIM:3 * TOKEN_DIM],
        descriptor[3 * TOKEN_DIM:4 * TOKEN_DIM],
        descriptor[4 * TOKEN_DIM:],
    )


def row_arrays(row: dict) -> tuple[np.ndarray, np.ndarray]:
    pre, post, difference, product, statistics = split_descriptor(row)
    context = np.concatenate([matrix_context(row), statistics], axis=0)
    features = np.concatenate([pre, post, difference, product, context], axis=0)
    return features, matrix_to_twist(relative_residual(row))


def stack_rows(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    values = [row_arrays(row) for row in rows]
    return np.stack([value[0] for value in values]), np.stack([value[1] for value in values])


class StructuredPoseRelationHead(nn.Module):
    def __init__(self, rank: int, hidden: int, translation_bound: float):
        super().__init__()
        self.translation_bound = translation_bound
        self.token_projections = nn.ModuleList(
            [
                nn.Sequential(nn.LayerNorm(TOKEN_DIM), nn.Linear(TOKEN_DIM, rank), nn.GELU())
                for _ in range(4)
            ]
        )
        self.context = nn.Sequential(
            nn.LayerNorm(3 * 6 + STAT_DIM),
            nn.Linear(3 * 6 + STAT_DIM, rank),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(5 * rank, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 6),
        )
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

    @staticmethod
    def radial_bound(values: torch.Tensor, maximum: float) -> torch.Tensor:
        norm = values.norm(dim=-1, keepdim=True)
        bounded_scale = maximum * torch.tanh(norm / maximum) / norm.clamp_min(1e-8)
        scale = torch.where(norm > 1e-4, bounded_scale, torch.ones_like(norm))
        return values * scale

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        token_values = [
            values[:, index * TOKEN_DIM:(index + 1) * TOKEN_DIM]
            for index in range(4)
        ]
        context = values[:, 4 * TOKEN_DIM:]
        features = [
            projection(token)
            for projection, token in zip(self.token_projections, token_values)
        ]
        output = self.fusion(torch.cat([*features, self.context(context)], dim=-1))
        rotation = self.radial_bound(output[:, :3], math.pi)
        translation = self.radial_bound(output[:, 3:], self.translation_bound)
        return torch.cat([rotation, translation], dim=-1)


class FlatPoseRelationHead(nn.Module):
    def __init__(self, input_dim: int, hidden: int, translation_bound: float):
        super().__init__()
        self.translation_bound = translation_bound
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 6),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output = self.network(values)
        rotation = StructuredPoseRelationHead.radial_bound(output[:, :3], math.pi)
        translation = StructuredPoseRelationHead.radial_bound(
            output[:, 3:], self.translation_bound
        )
        return torch.cat([rotation, translation], dim=-1)


def make_model(args: argparse.Namespace, input_dim: int) -> nn.Module:
    if args.architecture == "structured":
        return StructuredPoseRelationHead(args.rank, args.hidden, args.translation_bound)
    return FlatPoseRelationHead(input_dim, args.hidden, args.translation_bound)


def pair_fold(row: dict, folds: int = 5) -> int:
    pre, post = map(str, row["seqs"][-2:])
    pair = "|".join(sorted((pre, post)))
    digest = hashlib.sha1(f"{row['source']}|{pair}".encode()).hexdigest()
    return int(digest[:8], 16) % folds


def stratified_small(rows: list[dict], seed: int) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["source"]].append(row)
    rng = random.Random(seed)
    selected = []
    counts = {"avatarrex": 3, "thuman": 3, "mvhuman100": 2, "mvhuman200": 2}
    for source, count in counts.items():
        values = list(grouped[source])
        rng.shuffle(values)
        selected.extend(values[:count])
    return selected


def predict(model: nn.Module, x: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(x).float().to(device)).cpu().numpy()


def summarize(rows: list[dict], predictions: np.ndarray) -> dict[str, Any]:
    per_case = []
    for row, twist in zip(rows, predictions):
        estimated = twist_to_matrix(twist) @ row["full_relative"].numpy()
        error = pose_error(estimated, row["gt_relative"].numpy())
        per_case.append(
            {"pattern_id": row["pattern_id"], "source": row["source"], **error}
        )
    summary: dict[str, Any] = {}
    for key in ("translation_m", "rotation_deg", "composite"):
        values = np.asarray([case[key] for case in per_case])
        summary[key] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.9)),
        }
    summary["per_case"] = per_case
    return summary


def baseline_summary(rows: list[dict]) -> dict[str, Any]:
    return summarize(rows, np.zeros((len(rows), 6), dtype=np.float32))


def train_fixed_steps(
    args: argparse.Namespace,
    train_rows: list[dict],
    steps: int,
    seed: int,
) -> tuple[nn.Module, list[float], np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    x, y = stack_rows(train_rows)
    scale = np.maximum(y.std(axis=0), np.asarray([0.05] * 3 + [0.1] * 3))
    model = make_model(args, x.shape[1]).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    tx = torch.from_numpy(x).float().to(args.device)
    ty = torch.from_numpy(y).float().to(args.device)
    ts = torch.from_numpy(scale).float().to(args.device)
    losses = []
    model.train()
    for _ in range(steps):
        output = model(tx)
        loss = F.smooth_l1_loss(output / ts, ty / ts, beta=0.25)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return model, losses, scale


def select_steps(
    args: argparse.Namespace, train_rows: list[dict], validation_rows: list[dict]
) -> tuple[int, list[dict]]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    x, y = stack_rows(train_rows)
    vx, _ = stack_rows(validation_rows)
    scale = np.maximum(y.std(axis=0), np.asarray([0.05] * 3 + [0.1] * 3))
    model = make_model(args, x.shape[1]).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    tx = torch.from_numpy(x).float().to(args.device)
    ty = torch.from_numpy(y).float().to(args.device)
    ts = torch.from_numpy(scale).float().to(args.device)
    history = []
    best_step = args.eval_every
    best_score = float("inf")
    for step in range(1, args.steps + 1):
        model.train()
        output = model(tx)
        loss = F.smooth_l1_loss(output / ts, ty / ts, beta=0.25)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            prediction = predict(model, vx, args.device)
            score = summarize(validation_rows, prediction)["composite"]["mean"]
            history.append({"step": step, "train_loss": float(loss.detach().cpu()), "val": score})
            if score < best_score:
                best_score = score
                best_step = step
    return best_step, history


def select_residual_scale(
    args: argparse.Namespace,
    train_rows: list[dict],
    validation_rows: list[dict],
    steps: int,
) -> tuple[float, dict[str, float]]:
    model, _, _ = train_fixed_steps(args, train_rows, steps, args.seed + 10000)
    validation_x, _ = stack_rows(validation_rows)
    prediction = predict(model, validation_x, args.device)
    candidates = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
    scores = {
        str(scale): summarize(validation_rows, prediction * scale)["composite"]["mean"]
        for scale in candidates
    }
    selected = min(candidates, key=lambda scale: scores[str(scale)])
    return selected, scores


def source_means(summary: dict) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for case in summary["per_case"]:
        grouped[case["source"]].append(case["composite"])
    return {source: float(np.mean(values)) for source, values in sorted(grouped.items())}


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    args = parse_args()
    rows = torch.load(args.cache, map_location="cpu", weights_only=False)
    train_pool = [row for row in rows if row["split"] == "train"]
    frozen_rows = [row for row in rows if row["split"] == "eval10"]
    validation_history = []
    residual_scale = 1.0
    residual_scale_scores = {"1.0": float("nan")}
    if args.stage == "single":
        train_rows = [row for row in frozen_rows if "lbn1_1192" in row["pattern_id"]]
        eval_rows = train_rows
        selected_steps = args.steps
    elif args.stage == "small":
        train_rows = stratified_small(train_pool, args.seed)
        eval_rows = frozen_rows
        selected_steps = args.steps
    else:
        fit_rows = [row for row in train_pool if pair_fold(row) != 0]
        validation_rows = [row for row in train_pool if pair_fold(row) == 0]
        selected_steps, validation_history = select_steps(args, fit_rows, validation_rows)
        residual_scale, residual_scale_scores = select_residual_scale(
            args, fit_rows, validation_rows, selected_steps
        )
        train_rows = train_pool
        eval_rows = frozen_rows
    model, losses, target_scale = train_fixed_steps(
        args, train_rows, selected_steps, args.seed + 10000
    )
    eval_x, _ = stack_rows(eval_rows)
    prediction = predict(model, eval_x, args.device) * residual_scale
    result = summarize(eval_rows, prediction)
    baseline = baseline_summary(eval_rows)
    output_dir = args.output_dir / f"{args.stage}_{args.architecture}"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "stage": args.stage,
        "architecture": args.architecture,
        "rank": args.rank,
        "hidden": args.hidden,
        "translation_bound": args.translation_bound,
        "selected_steps": selected_steps,
        "residual_scale": residual_scale,
        "target_scale": target_scale,
        "train_pattern_ids": [row["pattern_id"] for row in train_rows],
    }
    torch.save(checkpoint, output_dir / "checkpoint.pth")
    report = {
        "args": vars(args),
        "train_cases": len(train_rows),
        "eval_cases": len(eval_rows),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "selected_steps": selected_steps,
        "residual_scale": residual_scale,
        "residual_scale_scores": residual_scale_scores,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "validation_history": validation_history,
        "formal_v9_baseline": baseline,
        "pose_relation_residual": result,
        "baseline_per_source": source_means(baseline),
        "result_per_source": source_means(result),
    }
    (output_dir / "report.json").write_text(
        json.dumps(json_ready(report), indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(
        f"stage={args.stage} arch={args.architecture} train={len(train_rows)} "
        f"steps={selected_steps} scale={residual_scale:.2f} "
        f"loss={losses[0]:.6f}->{losses[-1]:.6f} "
        f"formal={baseline['composite']['mean']:.4f} result={result['composite']['mean']:.4f} "
        f"p90={result['composite']['p90']:.4f}",
        flush=True,
    )
    print(f"per-source={source_means(result)}", flush=True)
    print(f"wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
