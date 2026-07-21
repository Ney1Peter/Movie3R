#!/usr/bin/env python3
"""Train and evaluate one held-out-source fold for the V17 fair comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = REPO_ROOT / "output" / "v17_direct_vs_factorized" / "feature_cache"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "v17_direct_vs_factorized" / "loso"
SOURCES = ("avatarrex", "thuman", "mvhuman100", "mvhuman200")
METHODS = (
    "weak_stats_absolute",
    "direct_absolute",
    "direct_residual",
    "direct_residual_no_gauge_aug",
    "direct_residual_gauge_aug",
    "direct_residual_uncertainty",
    "factor_scale_only",
    "factor_direction_scale",
    "factor_translation_residual",
    "factor_direction_scale_uncertainty",
    "vggt_direction_learned_scale",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--held_out_source", required=True, choices=SOURCES)
    parser.add_argument("--device", required=True)
    parser.add_argument("--hidden_dim", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--patience", type=int, default=160)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--max_rotation_residual_deg", type=float, default=90.0)
    parser.add_argument("--max_translation_residual_m", type=float, default=2.5)
    parser.add_argument("--max_log_scale_residual", type=float, default=1.8)
    parser.add_argument("--gauge_translation_range_m", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260725)
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stable_hash(value: str) -> int:
    return int(hashlib.sha1(value.encode("utf-8")).hexdigest()[:12], 16)


def choose_validation(rows: list[dict], train_pool: np.ndarray, held_out: str) -> tuple[np.ndarray, np.ndarray, dict]:
    validation: list[int] = []
    held_captures: dict[str, str] = {}
    train_sources = sorted({rows[index]["source"] for index in train_pool})
    for source in train_sources:
        groups: dict[str, list[int]] = {}
        for index in train_pool:
            if rows[index]["source"] == source:
                groups.setdefault(rows[index]["capture"], []).append(int(index))
        target = max(4, round(sum(map(len, groups.values())) * 0.20))
        selected = min(
            groups,
            key=lambda group: (
                abs(len(groups[group]) - target),
                stable_hash(f"{held_out}:{source}:{group}"),
            ),
        )
        held_captures[source] = selected
        validation.extend(groups[selected])
    val_ids = np.asarray(sorted(validation), dtype=np.int64)
    train_ids = np.asarray(sorted(set(map(int, train_pool)) - set(validation)), dtype=np.int64)
    train_pairs = {rows[index]["camera_pair"] for index in train_ids}
    val_pairs = {rows[index]["camera_pair"] for index in val_ids}
    if train_pairs & val_pairs:
        raise RuntimeError("Internal validation contains a camera pair seen in training")
    return train_ids, val_ids, {
        "held_captures": held_captures,
        "train_count": int(len(train_ids)),
        "validation_count": int(len(val_ids)),
        "unseen_camera_pair_count": int(len(val_pairs)),
    }


class Standardizer:
    def __init__(self, values: np.ndarray):
        self.mean = np.mean(values, axis=0).astype(np.float32)
        self.scale = np.std(values, axis=0).astype(np.float32)
        self.scale = np.maximum(self.scale, 1e-4)

    def numpy(self, values: np.ndarray) -> np.ndarray:
        return np.clip((values - self.mean) / self.scale, -10.0, 10.0).astype(np.float32)


class BridgeMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


def rotation_to_6d(rotation: np.ndarray) -> np.ndarray:
    return np.asarray(rotation, dtype=np.float32)[..., :, :2].swapaxes(-1, -2).reshape(*rotation.shape[:-2], 6)


def rotation_6d_to_matrix(values: torch.Tensor) -> torch.Tensor:
    first = F.normalize(values[..., :3], dim=-1, eps=1e-6)
    second_raw = values[..., 3:6]
    second = F.normalize(second_raw - (first * second_raw).sum(dim=-1, keepdim=True) * first, dim=-1, eps=1e-6)
    third = torch.cross(first, second, dim=-1)
    return torch.stack([first, second, third], dim=-1)


def quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    q = F.normalize(quaternion, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def axis_angle_to_matrix(vector: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    skew = torch.zeros((*vector.shape[:-1], 3, 3), dtype=vector.dtype, device=vector.device)
    skew[..., 0, 1] = -vector[..., 2]
    skew[..., 0, 2] = vector[..., 1]
    skew[..., 1, 0] = vector[..., 2]
    skew[..., 1, 2] = -vector[..., 0]
    skew[..., 2, 0] = -vector[..., 1]
    skew[..., 2, 1] = vector[..., 0]
    identity = torch.eye(3, dtype=vector.dtype, device=vector.device).expand_as(skew)
    sine_over_angle = torch.sinc(angle / math.pi)[..., None]
    one_minus_cos_over_angle2 = (0.5 * torch.sinc(angle / (2.0 * math.pi)).square())[..., None]
    return identity + sine_over_angle * skew + one_minus_cos_over_angle2 * (skew @ skew)


def geodesic_radians(estimated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    relative = estimated @ target.transpose(-1, -2)
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cosine)


def bounded_rotvec(raw: torch.Tensor, maximum_rad: float) -> torch.Tensor:
    vector = torch.tanh(raw) * maximum_rad
    norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    return vector * torch.clamp(maximum_rad / norm.clamp_min(1e-6), max=1.0)


def normalize_direction(vector: torch.Tensor) -> torch.Tensor:
    fallback = torch.zeros_like(vector)
    fallback[..., 2] = 1.0
    norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    return torch.where(norm > 1e-6, vector / norm.clamp_min(1e-6), fallback)


def random_gauge_world_features(world: torch.Tensor, translation_range: float) -> torch.Tensor:
    batch = world.shape[0]
    rotation = quaternion_to_matrix(torch.randn((batch, 4), device=world.device, dtype=world.dtype))
    translation = (torch.rand((batch, 3), device=world.device, dtype=world.dtype) * 2.0 - 1.0) * translation_range
    gauge = torch.eye(4, device=world.device, dtype=world.dtype).repeat(batch, 1, 1)
    gauge[:, :3, :3] = rotation
    gauge[:, :3, 3] = translation
    inverse = torch.linalg.inv(gauge)
    output = torch.empty_like(world)
    output[:, 0] = gauge @ world[:, 0]
    output[:, 1] = gauge @ world[:, 1]
    output[:, 2:] = gauge[:, None] @ world[:, 2:] @ inverse[:, None]
    return output[:, :, :3, :4].reshape(batch, -1)


def plain_world_features(world: np.ndarray) -> np.ndarray:
    return world[:, :, :3, :4].reshape(len(world), -1).astype(np.float32)


@dataclass
class Arrays:
    invariant: np.ndarray
    world: np.ndarray
    stats: np.ndarray
    fixed: np.ndarray
    torso: np.ndarray
    v15: np.ndarray
    target: np.ndarray


def output_dim(method: str) -> int:
    return {
        "weak_stats_absolute": 9,
        "direct_absolute": 9,
        "direct_residual": 6,
        "direct_residual_no_gauge_aug": 6,
        "direct_residual_gauge_aug": 6,
        "direct_residual_uncertainty": 8,
        "factor_scale_only": 1,
        "factor_direction_scale": 4,
        "factor_translation_residual": 3,
        "factor_direction_scale_uncertainty": 6,
        "vggt_direction_learned_scale": 1,
    }[method]


def uses_uncertainty(method: str) -> bool:
    return method in {"direct_residual_uncertainty", "factor_direction_scale_uncertainty"}


def decode(method: str, output: torch.Tensor, base_fixed: torch.Tensor, base_torso: torch.Tensor, base_v15: torch.Tensor, args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    fixed_rotation, fixed_translation = base_fixed[:, :3, :3], base_fixed[:, :3, 3]
    torso_rotation, torso_translation = base_torso[:, :3, :3], base_torso[:, :3, 3]
    v15_translation = base_v15[:, :3, 3]
    uncertainty = None
    if method in {"weak_stats_absolute", "direct_absolute"}:
        rotation = rotation_6d_to_matrix(output[:, :6])
        translation = output[:, 6:9]
    elif method in {"direct_residual", "direct_residual_no_gauge_aug", "direct_residual_gauge_aug", "direct_residual_uncertainty"}:
        residual = bounded_rotvec(output[:, :3], math.radians(float(args.max_rotation_residual_deg)))
        rotation = axis_angle_to_matrix(residual) @ fixed_rotation
        translation = fixed_translation + float(args.max_translation_residual_m) * torch.tanh(output[:, 3:6])
        if uses_uncertainty(method):
            uncertainty = output[:, 6:8].clamp(-4.0, 4.0)
    elif method == "factor_scale_only":
        direction = normalize_direction(torso_translation)
        initial_scale = torch.linalg.vector_norm(torso_translation, dim=-1, keepdim=True).clamp_min(1e-4)
        scale = initial_scale * torch.exp(float(args.max_log_scale_residual) * torch.tanh(output[:, :1]))
        rotation, translation = torso_rotation, direction * scale
    elif method in {"factor_direction_scale", "factor_direction_scale_uncertainty"}:
        initial_direction = normalize_direction(torso_translation)
        direction = normalize_direction(initial_direction + 0.75 * torch.tanh(output[:, :3]))
        initial_scale = torch.linalg.vector_norm(torso_translation, dim=-1, keepdim=True).clamp_min(1e-4)
        scale = initial_scale * torch.exp(float(args.max_log_scale_residual) * torch.tanh(output[:, 3:4]))
        rotation, translation = torso_rotation, direction * scale
        if uses_uncertainty(method):
            uncertainty = output[:, 4:6].clamp(-4.0, 4.0)
    elif method == "factor_translation_residual":
        rotation = torso_rotation
        translation = torso_translation + float(args.max_translation_residual_m) * torch.tanh(output[:, :3])
    elif method == "vggt_direction_learned_scale":
        direction = normalize_direction(v15_translation)
        initial_scale = torch.linalg.vector_norm(torso_translation, dim=-1, keepdim=True).clamp_min(1e-4)
        scale = initial_scale * torch.exp(float(args.max_log_scale_residual) * torch.tanh(output[:, :1]))
        rotation, translation = torso_rotation, direction * scale
    else:
        raise KeyError(method)
    return rotation, translation, uncertainty


def prediction_loss(method: str, output: torch.Tensor, fixed: torch.Tensor, torso: torch.Tensor, v15: torch.Tensor, target: torch.Tensor, hard_weight: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    rotation, translation, uncertainty = decode(method, output, fixed, torso, v15, args)
    target_rotation, target_translation = target[:, :3, :3], target[:, :3, 3]
    rotation_error = geodesic_radians(rotation, target_rotation)
    translation_error = F.smooth_l1_loss(translation, target_translation, reduction="none", beta=0.25).mean(dim=-1)
    target_direction = normalize_direction(target_translation)
    direction_error = 1.0 - (normalize_direction(translation) * target_direction).sum(dim=-1).clamp(-1.0, 1.0)
    target_log_scale = torch.log(torch.linalg.vector_norm(target_translation, dim=-1).clamp_min(1e-4))
    predicted_log_scale = torch.log(torch.linalg.vector_norm(translation, dim=-1).clamp_min(1e-4))
    scale_error = F.smooth_l1_loss(predicted_log_scale, target_log_scale, reduction="none", beta=0.20)

    if method in {"weak_stats_absolute", "direct_absolute", "direct_residual", "direct_residual_no_gauge_aug", "direct_residual_gauge_aug", "direct_residual_uncertainty"}:
        if uncertainty is None:
            per_sample = 3.0 * rotation_error + 2.0 * translation_error
        else:
            per_sample = (
                torch.exp(-uncertainty[:, 0]) * (3.0 * rotation_error)
                + uncertainty[:, 0]
                + torch.exp(-uncertainty[:, 1]) * (2.0 * translation_error)
                + uncertainty[:, 1]
            )
    elif method in {"factor_direction_scale", "factor_direction_scale_uncertainty"}:
        if uncertainty is None:
            per_sample = direction_error + scale_error + 2.0 * translation_error
        else:
            per_sample = (
                torch.exp(-uncertainty[:, 0]) * direction_error
                + uncertainty[:, 0]
                + torch.exp(-uncertainty[:, 1]) * (scale_error + 2.0 * translation_error)
                + uncertainty[:, 1]
            )
    elif method in {"factor_scale_only", "vggt_direction_learned_scale"}:
        per_sample = scale_error + 2.0 * translation_error
    else:
        per_sample = 2.0 * translation_error

    regularizer = torch.zeros_like(per_sample)
    if method.startswith("direct_residual"):
        regularizer = 0.005 * output[:, :6].square().mean(dim=-1)
    elif method.startswith("factor_") or method.startswith("vggt_"):
        regularizer = 0.003 * output[:, : min(output.shape[1], 4)].square().mean(dim=-1)
    return ((per_sample + regularizer) * hard_weight).mean()


def validation_objective(rotation: torch.Tensor, translation: torch.Tensor, target: torch.Tensor) -> float:
    rotation_deg = torch.rad2deg(geodesic_radians(rotation, target[:, :3, :3])).mean()
    translation_m = torch.linalg.vector_norm(translation - target[:, :3, 3], dim=-1).mean()
    return float((translation_m + rotation_deg / 30.0).item())


def initialize_output(model: BridgeMLP, method: str, target: np.ndarray, train_ids: np.ndarray) -> None:
    layer = model.net[-1]
    assert isinstance(layer, nn.Linear)
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    if method in {"weak_stats_absolute", "direct_absolute"}:
        mean_rotation = Rotation.from_matrix(target[train_ids, :3, :3].astype(np.float64)).mean().as_matrix().astype(np.float32)
        mean_translation = target[train_ids, :3, 3].mean(axis=0)
        initial = np.concatenate([rotation_to_6d(mean_rotation[None])[0], mean_translation])
        with torch.no_grad():
            layer.bias.copy_(torch.as_tensor(initial, device=layer.bias.device))


def make_input_numpy(method: str, invariant: np.ndarray, world_flat: np.ndarray, stats: np.ndarray, invariant_scaler: Standardizer, world_scaler: Standardizer, stats_scaler: Standardizer) -> np.ndarray:
    if method == "weak_stats_absolute":
        return stats_scaler.numpy(stats)
    if method in {"direct_residual_no_gauge_aug", "direct_residual_gauge_aug"}:
        return np.concatenate([invariant_scaler.numpy(invariant), world_scaler.numpy(world_flat)], axis=1)
    return invariant_scaler.numpy(invariant)


def model_input_batch(method: str, ids: np.ndarray, arrays: Arrays, invariant_scaled: torch.Tensor, stats_scaled: torch.Tensor, world_scaler: Standardizer, device: torch.device, augment_gauge: bool, args: argparse.Namespace) -> torch.Tensor:
    index = torch.as_tensor(ids, dtype=torch.long, device=device)
    if method == "weak_stats_absolute":
        return stats_scaled[index]
    invariant = invariant_scaled[index]
    if method not in {"direct_residual_no_gauge_aug", "direct_residual_gauge_aug"}:
        return invariant
    world = torch.as_tensor(arrays.world[ids], dtype=torch.float32, device=device)
    if augment_gauge:
        world_flat = random_gauge_world_features(world, float(args.gauge_translation_range_m))
    else:
        world_flat = world[:, :, :3, :4].reshape(len(ids), -1)
    mean = torch.as_tensor(world_scaler.mean, dtype=torch.float32, device=device)
    scale = torch.as_tensor(world_scaler.scale, dtype=torch.float32, device=device)
    world_scaled = ((world_flat - mean) / scale).clamp(-10.0, 10.0)
    return torch.cat([invariant, world_scaled], dim=-1)


def train_method(method: str, arrays: Arrays, train_ids: np.ndarray, val_ids: np.ndarray, invariant_scaler: Standardizer, world_scaler: Standardizer, stats_scaler: Standardizer, args: argparse.Namespace, method_index: int) -> tuple[BridgeMLP, dict]:
    device = torch.device(args.device)
    seed_all(int(args.seed) + 100 * method_index + stable_hash(args.held_out_source) % 97)
    if method == "weak_stats_absolute":
        input_dim = int(arrays.stats.shape[1])
    elif method in {"direct_residual_no_gauge_aug", "direct_residual_gauge_aug"}:
        input_dim = int(arrays.invariant.shape[1] + 60)
    else:
        input_dim = int(arrays.invariant.shape[1])
    model = BridgeMLP(input_dim, int(args.hidden_dim), output_dim(method)).to(device)
    initialize_output(model, method, arrays.target, train_ids)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(args.epochs), eta_min=float(args.learning_rate) * 0.05)

    invariant_scaled = torch.as_tensor(invariant_scaler.numpy(arrays.invariant), dtype=torch.float32, device=device)
    stats_scaled = torch.as_tensor(stats_scaler.numpy(arrays.stats), dtype=torch.float32, device=device)
    fixed = torch.as_tensor(arrays.fixed, dtype=torch.float32, device=device)
    torso = torch.as_tensor(arrays.torso, dtype=torch.float32, device=device)
    v15 = torch.as_tensor(arrays.v15, dtype=torch.float32, device=device)
    target = torch.as_tensor(arrays.target, dtype=torch.float32, device=device)
    with torch.no_grad():
        fixed_rotation_error = geodesic_radians(fixed[:, :3, :3], target[:, :3, :3])
        fixed_translation_error = torch.linalg.vector_norm(fixed[:, :3, 3] - target[:, :3, 3], dim=-1)
        hard_weight = 1.0 + 0.25 * (fixed_rotation_error / math.radians(60.0)).clamp(0.0, 1.0) + 0.25 * (fixed_translation_error / 2.0).clamp(0.0, 1.0)

    best_state, best_objective, best_epoch = None, float("inf"), -1
    no_improvement = 0
    rng = np.random.default_rng(int(args.seed) + method_index * 997)
    started = time.perf_counter()
    for epoch in range(int(args.epochs)):
        model.train()
        shuffled = rng.permutation(train_ids)
        for start in range(0, len(shuffled), int(args.batch_size)):
            ids = shuffled[start : start + int(args.batch_size)]
            augment = method == "direct_residual_gauge_aug"
            inputs = model_input_batch(method, ids, arrays, invariant_scaled, stats_scaled, world_scaler, device, augment, args)
            index = torch.as_tensor(ids, dtype=torch.long, device=device)
            output = model(inputs)
            loss = prediction_loss(method, output, fixed[index], torso[index], v15[index], target[index], hard_weight[index], args)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        scheduler.step()

        if epoch % 5 == 0 or epoch == int(args.epochs) - 1:
            model.eval()
            with torch.no_grad():
                inputs = model_input_batch(method, val_ids, arrays, invariant_scaled, stats_scaled, world_scaler, device, False, args)
                index = torch.as_tensor(val_ids, dtype=torch.long, device=device)
                output = model(inputs)
                rotation, translation, _ = decode(method, output, fixed[index], torso[index], v15[index], args)
                objective = validation_objective(rotation, translation, target[index])
            if objective < best_objective - 1e-5:
                best_objective = objective
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                no_improvement = 0
            else:
                no_improvement += 5
            if no_improvement >= int(args.patience):
                break
    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for {method}")
    model.load_state_dict(best_state)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return model.eval(), {
        "best_epoch": int(best_epoch),
        "best_validation_objective": float(best_objective),
        "trained_epochs": int(epoch + 1),
        "parameter_count": int(parameter_count),
        "training_seconds": float(time.perf_counter() - started),
        "gauge_handling": (
            "random_global_gauge_augmentation"
            if method == "direct_residual_gauge_aug"
            else "raw_world_gauge_no_augmentation"
            if method == "direct_residual_no_gauge_aug"
            else "analytic_camera_frame_invariance"
        ),
    }


def predict_method(method: str, model: BridgeMLP, ids: np.ndarray, arrays: Arrays, invariant_scaler: Standardizer, world_scaler: Standardizer, stats_scaler: Standardizer, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, float]:
    device = torch.device(args.device)
    world_flat = plain_world_features(arrays.world)
    inputs = make_input_numpy(method, arrays.invariant, world_flat, arrays.stats, invariant_scaler, world_scaler, stats_scaler)
    tensor = torch.as_tensor(inputs[ids], dtype=torch.float32, device=device)
    fixed = torch.as_tensor(arrays.fixed[ids], dtype=torch.float32, device=device)
    torso = torch.as_tensor(arrays.torso[ids], dtype=torch.float32, device=device)
    v15 = torch.as_tensor(arrays.v15[ids], dtype=torch.float32, device=device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        output = model(tensor)
        rotation, translation, uncertainty = decode(method, output, fixed, torso, v15, args)
    torch.cuda.synchronize(device)
    elapsed_ms = 1000.0 * (time.perf_counter() - started) / max(len(ids), 1)
    return (
        rotation.cpu().numpy(),
        translation.cpu().numpy(),
        uncertainty.cpu().numpy() if uncertainty is not None else None,
        elapsed_ms,
    )


def direction_error_deg(estimated: np.ndarray, target: np.ndarray) -> float:
    a = estimated / max(float(np.linalg.norm(estimated)), 1e-8)
    b = target / max(float(np.linalg.norm(target)), 1e-8)
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def evaluated_row(rotation: np.ndarray, translation: np.ndarray, target: np.ndarray, uncertainty: np.ndarray | None = None) -> dict:
    rotation_error = Rotation.from_matrix((rotation @ target[:3, :3].T).astype(np.float64)).magnitude()
    delta = np.asarray(translation) - target[:3, 3]
    scale = float(np.linalg.norm(translation))
    target_scale = float(np.linalg.norm(target[:3, 3]))
    row = {
        "fit_failed": False,
        "camera_translation_error_m": float(np.linalg.norm(delta)),
        "camera_rotation_error_deg": float(np.degrees(rotation_error)),
        "translation_direction_error_deg": direction_error_deg(translation, target[:3, 3]),
        "translation_scale_abs_error_m": abs(scale - target_scale),
        "translation_scale_log_abs": abs(math.log(max(scale, 1e-8) / max(target_scale, 1e-8))),
        "translation_error_xyz_m": np.abs(delta).astype(float).tolist(),
        "translation_view_direction_error_m": float(abs(delta[2])),
        "translation_transverse_error_m": float(np.linalg.norm(delta[:2])),
        "relative_rotation": np.asarray(rotation, dtype=float).tolist(),
        "relative_translation": np.asarray(translation, dtype=float).tolist(),
    }
    if uncertainty is not None:
        row["predicted_rotation_log_variance"] = float(uncertainty[0])
        row["predicted_translation_log_variance"] = float(uncertainty[1])
    return row


def partial_oracle_rows(rotation: np.ndarray, translation: np.ndarray, target: np.ndarray) -> tuple[dict, dict]:
    return (
        evaluated_row(rotation, target[:3, 3], target),
        evaluated_row(target[:3, :3], translation, target),
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("V17 fold training requires CUDA")
    seed_all(int(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.cache_dir / "v17_explicit_features.json").read_text(encoding="utf-8"))
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
    rows = metadata["rows"]
    held_out = str(args.held_out_source)
    train_pool = np.asarray([index for index, row in enumerate(rows) if row["source"] != held_out], dtype=np.int64)
    test_ids = np.asarray([index for index, row in enumerate(rows) if row["source"] == held_out], dtype=np.int64)
    train_ids, val_ids, split = choose_validation(rows, train_pool, held_out)
    invariant_scaler = Standardizer(arrays.invariant[train_ids])
    world_scaler = Standardizer(plain_world_features(arrays.world[train_ids]))
    stats_scaler = Standardizer(arrays.stats[train_ids])

    trained: dict[str, dict] = {}
    predictions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None]] = {}
    checkpoint_dir = args.output_dir / "checkpoints" / held_out
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for method_index, method in enumerate(METHODS):
        print(f"[{held_out}] training {method} ({method_index + 1}/{len(METHODS)})", flush=True)
        model, diagnostics = train_method(
            method,
            arrays,
            train_ids,
            val_ids,
            invariant_scaler,
            world_scaler,
            stats_scaler,
            args,
            method_index,
        )
        rotation, translation, uncertainty, latency = predict_method(
            method,
            model,
            test_ids,
            arrays,
            invariant_scaler,
            world_scaler,
            stats_scaler,
            args,
        )
        diagnostics["test_latency_ms_per_cut"] = latency
        trained[method] = diagnostics
        predictions[method] = (rotation, translation, uncertainty)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "method": method,
                "held_out_source": held_out,
                "args": vars(args),
                "invariant_mean": invariant_scaler.mean,
                "invariant_scale": invariant_scaler.scale,
                "world_mean": world_scaler.mean,
                "world_scale": world_scaler.scale,
                "stats_mean": stats_scaler.mean,
                "stats_scale": stats_scaler.scale,
            },
            checkpoint_dir / f"{method}.pt",
        )

    output_rows = []
    for local_index, sample_index in enumerate(test_ids):
        metadata_row = rows[int(sample_index)]
        methods = dict(metadata_row["stored_metrics"])
        target = arrays.target[sample_index]
        for method in METHODS:
            rotation, translation, uncertainty = predictions[method]
            uncertainty_row = uncertainty[local_index] if uncertainty is not None else None
            methods[method] = evaluated_row(rotation[local_index], translation[local_index], target, uncertainty_row)
        learned_rotation, learned_translation, _ = predictions["direct_residual"]
        learned_rot_gt_t, gt_rot_learned_t = partial_oracle_rows(
            learned_rotation[local_index], learned_translation[local_index], target
        )
        methods["learned_rotation_gt_translation"] = learned_rot_gt_t
        methods["gt_rotation_learned_translation"] = gt_rot_learned_t
        output_rows.append(
            {
                "case_name": metadata_row["case_name"],
                "source": metadata_row["source"],
                "capture": metadata_row["capture"],
                "camera_pair": metadata_row["camera_pair"],
                "methods": methods,
            }
        )

    report = {
        "experiment": "V17 Direct SE(3) vs Factorized Translation Bridge",
        "held_out_source": held_out,
        "protocol": {
            "leave_one_source_out": True,
            "human3r_frozen": True,
            "raw_tokens_used": False,
            "gt_depth_used": False,
            "post_cut_frames": 1,
            "target_frame": "last pre-cut Human3R camera frame",
            "normalization_fit": "training subset only",
            "model_selection": "capture and camera-pair held-out validation only",
        },
        "split": split,
        "test_count": int(len(test_ids)),
        "test_camera_pair_count": int(len({rows[index]["camera_pair"] for index in test_ids})),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "training": trained,
        "rows": output_rows,
    }
    output_path = args.output_dir / f"v17_loso_{held_out}.json"
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f">> wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
