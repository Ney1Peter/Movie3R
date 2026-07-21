#!/usr/bin/env python3
"""Train one V54 explicit correspondence bridge and evaluate synthetic/real cuts."""

from __future__ import annotations

import argparse
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
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = REPO_ROOT / "output" / "v54_synthetic_explicit_shot_bridge" / "cache"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "v54_synthetic_explicit_shot_bridge" / "models"
SOURCES = ("avatarrex", "thuman", "mvhuman100", "mvhuman200")
VARIANTS = ("raw_se3", "raw_sim3", "da3_se3", "da3_se3_human")
BUCKETS = {
    "small": {"rotation_deg": 10.0, "translation_m": 0.30, "scale": (0.90, 1.10)},
    "medium": {"rotation_deg": 30.0, "translation_m": 1.00, "scale": (0.75, 1.25)},
    "large": {"rotation_deg": 60.0, "translation_m": 3.00, "scale": (0.60, 1.40)},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--held_out_source", choices=SOURCES, required=True)
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--steps", type=int, default=2200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--points", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--eval_augmentations", type=int, default=4)
    parser.add_argument("--real_eval_repeats", type=int, default=3)
    parser.add_argument("--real_cut_training_probability", type=float, default=0.35)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--eval_only", action="store_true")
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    return (vector / norm).astype(np.float32)


def sample_perturbation(rng: np.random.Generator, bucket: str, with_scale: bool) -> tuple[float, np.ndarray, np.ndarray]:
    config = BUCKETS[bucket]
    axis = normalize_vector(rng.normal(size=3).astype(np.float32))
    angle = math.radians(rng.uniform(0.15, 1.0) * float(config["rotation_deg"]))
    rotation = Rotation.from_rotvec(axis.astype(np.float64) * angle).as_matrix().astype(np.float32)
    direction = normalize_vector(rng.normal(size=3).astype(np.float32))
    translation = direction * rng.uniform(0.15, 1.0) * float(config["translation_m"])
    if with_scale:
        lo, hi = config["scale"]
        scale = float(np.exp(rng.uniform(math.log(lo), math.log(hi))))
    else:
        scale = 1.0
    return scale, rotation, translation.astype(np.float32)


def compose_correction(
    base_rotation: np.ndarray,
    base_translation: np.ndarray,
    perturb_scale: float,
    perturb_rotation: np.ndarray,
    perturb_translation: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    scale = 1.0 / perturb_scale
    rotation = base_rotation @ perturb_rotation.T
    translation = base_translation - scale * rotation @ perturb_translation
    return float(scale), rotation.astype(np.float32), translation.astype(np.float32)


@dataclass
class Cache:
    points: np.ndarray
    anchors: np.ndarray
    poses: np.ndarray
    scales: np.ndarray
    target: np.ndarray
    fixed: np.ndarray
    torso: np.ndarray
    rows: list[dict]


def load_cache(cache_dir: Path) -> Cache:
    arrays = np.load(cache_dir / "v54_explicit_geometry.npz")
    metadata = json.loads((cache_dir / "v54_explicit_geometry.json").read_text(encoding="utf-8"))
    return Cache(
        points=np.asarray(arrays["points"], dtype=np.float32),
        anchors=np.asarray(arrays["human_anchors"], dtype=np.float32),
        poses=np.asarray(arrays["poses"], dtype=np.float32),
        scales=np.asarray(arrays["da3_scales"], dtype=np.float32),
        target=np.asarray(arrays["relative_target"], dtype=np.float32),
        fixed=np.asarray(arrays["relative_fixed"], dtype=np.float32),
        torso=np.asarray(arrays["relative_torso"], dtype=np.float32),
        rows=list(metadata["rows"]),
    )


def transform_points(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    output = points.copy()
    output[:, :3] = scale * (points[:, :3] @ rotation.T) + translation
    return output


def append_human(points: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    scene = np.concatenate(
        [points, np.zeros((len(points), 5), dtype=np.float32)],
        axis=-1,
    )
    anchor_features = np.zeros((4, 12), dtype=np.float32)
    anchor_features[:, :3] = anchors
    anchor_features[:, 6] = 1.0
    anchor_features[:, 7] = 1.0
    anchor_features[:, 8:] = np.eye(4, dtype=np.float32)
    return np.concatenate([scene, anchor_features], axis=0)


def scene_only(points: np.ndarray) -> np.ndarray:
    return np.concatenate([points, np.zeros((len(points), 5), dtype=np.float32)], axis=-1)


class SyntheticBatcher:
    def __init__(
        self,
        cache: Cache,
        indices: np.ndarray,
        variant: str,
        points: int,
        seed: int,
        real_cut_probability: float = 0.0,
    ):
        self.cache = cache
        self.indices = np.asarray(indices, dtype=np.int64)
        self.variant = variant
        self.points = int(points)
        self.rng = np.random.default_rng(seed)
        self.use_da3 = variant.startswith("da3")
        self.use_human = variant.endswith("human")
        self.use_sim3 = variant == "raw_sim3"
        self.real_cut_probability = float(real_cut_probability)

    def one(self, bucket: str | None = None, exact: bool | None = None) -> dict[str, np.ndarray | float | int]:
        case_index = int(self.rng.choice(self.indices))
        real_cut = exact is not True and bool(self.rng.random() < self.real_cut_probability)
        shot_start = int(self.rng.choice([0, 2]))
        if exact is None:
            exact = bool(self.rng.random() < 0.55)
        if real_cut:
            exact = False
            target_frame, source_frame = 1, 2
        else:
            target_frame = shot_start
            source_frame = shot_start if exact else shot_start + 1
        target_pool = self.cache.points[case_index, target_frame]
        source_pool = self.cache.points[case_index, source_frame]
        if exact:
            selected = self.rng.choice(len(target_pool), size=self.points, replace=self.points > len(target_pool))
            target = target_pool[selected].copy()
            source = target_pool[selected].copy()
            permutation = self.rng.permutation(self.points)
            source = source[permutation]
            correspondence = permutation.astype(np.int64)
        else:
            target = target_pool[self.rng.choice(len(target_pool), size=self.points, replace=False)].copy()
            source = source_pool[self.rng.choice(len(source_pool), size=self.points, replace=False)].copy()
            correspondence = np.full(self.points, -1, dtype=np.int64)

        metric_scale = float(self.cache.scales[case_index, target_frame]) if self.use_da3 else 1.0
        source_metric_scale = float(self.cache.scales[case_index, source_frame]) if self.use_da3 else 1.0
        target[:, :3] *= metric_scale
        source[:, :3] *= source_metric_scale
        target_anchor = self.cache.anchors[case_index, target_frame].copy() * metric_scale
        source_anchor = self.cache.anchors[case_index, source_frame].copy() * source_metric_scale

        if real_cut:
            fixed = self.cache.fixed[case_index]
            source[:, :3] = source[:, :3] @ fixed[:3, :3].T + fixed[:3, 3]
            source_anchor = source_anchor @ fixed[:3, :3].T + fixed[:3, 3]
            residual = self.cache.target[case_index] @ np.linalg.inv(fixed)
            base_rotation = np.asarray(residual[:3, :3], dtype=np.float32)
            base_translation = np.asarray(residual[:3, 3], dtype=np.float32)
        elif exact:
            base_rotation = np.eye(3, dtype=np.float32)
            base_translation = np.zeros(3, dtype=np.float32)
        else:
            base = np.linalg.inv(self.cache.poses[case_index, target_frame]) @ self.cache.poses[case_index, source_frame]
            base_rotation = np.asarray(base[:3, :3], dtype=np.float32)
            base_translation = np.asarray(base[:3, 3], dtype=np.float32) * metric_scale
        bucket = bucket or str(self.rng.choice(list(BUCKETS)))
        perturb_scale, perturb_rotation, perturb_translation = sample_perturbation(
            self.rng,
            bucket,
            with_scale=self.use_sim3,
        )
        source = transform_points(source, perturb_scale, perturb_rotation, perturb_translation)
        source_anchor = perturb_scale * (source_anchor @ perturb_rotation.T) + perturb_translation
        gt_scale, gt_rotation, gt_translation = compose_correction(
            base_rotation,
            base_translation,
            perturb_scale,
            perturb_rotation,
            perturb_translation,
        )
        if self.use_human:
            target = append_human(target, target_anchor)
            source = append_human(source, source_anchor)
            if exact:
                correspondence = np.concatenate([correspondence, np.arange(self.points, self.points + 4, dtype=np.int64)])
            else:
                correspondence = np.concatenate([correspondence, np.full(4, -1, dtype=np.int64)])
        else:
            target = scene_only(target)
            source = scene_only(source)
        return {
            "target": target.astype(np.float32),
            "source": source.astype(np.float32),
            "scale": np.float32(gt_scale),
            "rotation": gt_rotation,
            "translation": gt_translation,
            "correspondence": correspondence,
            "case_index": case_index,
            "bucket": bucket,
            "exact": int(exact),
            "real_cut": int(real_cut),
        }

    def batch(self, batch_size: int, pretrain: bool = False) -> dict[str, torch.Tensor]:
        rows = [self.one(exact=True if pretrain else None) for _ in range(batch_size)]
        return {
            key: torch.from_numpy(np.stack([row[key] for row in rows]))
            for key in ("target", "source", "scale", "rotation", "translation", "correspondence")
        }


def knn_indices(xyz: torch.Tensor, k: int) -> torch.Tensor:
    distance = torch.cdist(xyz, xyz)
    return distance.topk(k=k + 1, largest=False).indices[:, :, 1:]


def graph_features(features: torch.Tensor, xyz: torch.Tensor, k: int) -> torch.Tensor:
    batch, points, channels = features.shape
    indices = knn_indices(xyz, k)
    offset = torch.arange(batch, device=features.device).view(batch, 1, 1) * points
    neighbors = features.reshape(batch * points, channels)[(indices + offset).reshape(-1)].reshape(batch, points, k, channels)
    center = features[:, :, None, :].expand(-1, -1, k, -1)
    return torch.cat([center, neighbors - center], dim=-1).permute(0, 3, 1, 2).contiguous()


class EdgeEncoder(nn.Module):
    def __init__(self, input_dim: int = 12, embedding_dim: int = 96, k: int = 12):
        super().__init__()
        self.k = k
        self.edge1 = nn.Sequential(
            nn.Conv2d(input_dim * 2, 64, 1, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 64, 1, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )
        self.edge2 = nn.Sequential(
            nn.Conv2d(128, 96, 1, bias=False),
            nn.GroupNorm(8, 96),
            nn.SiLU(),
        )
        self.project = nn.Sequential(
            nn.Conv1d(160, embedding_dim, 1, bias=False),
            nn.GroupNorm(8, embedding_dim),
            nn.SiLU(),
        )

    def forward(self, values: torch.Tensor, normalized_xyz: torch.Tensor) -> torch.Tensor:
        first = self.edge1(graph_features(values, normalized_xyz, self.k)).max(dim=-1).values
        first_points = first.transpose(1, 2)
        second = self.edge2(graph_features(first_points, normalized_xyz, self.k)).max(dim=-1).values
        return self.project(torch.cat([first, second], dim=1)).transpose(1, 2)


def normalized_features(points: torch.Tensor, scene_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    xyz = points[..., :3]
    scene_xyz = xyz[:, :scene_count]
    center = scene_xyz.mean(dim=1, keepdim=True)
    radius = torch.sqrt(((scene_xyz - center).square().sum(dim=-1)).mean(dim=1, keepdim=True)).clamp_min(1e-3)
    normalized_xyz = (xyz - center) / radius[..., None]
    rgb = points[..., 3:6] * 2.0 - 1.0
    auxiliary = points[..., 6:]
    return torch.cat([normalized_xyz, rgb, auxiliary], dim=-1), normalized_xyz


def weighted_umeyama(
    source: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    estimate_scale: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weights = weights.clamp_min(1e-5)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    source_center = (weights[..., None] * source).sum(dim=1)
    target_center = (weights[..., None] * target).sum(dim=1)
    source_zero = source - source_center[:, None]
    target_zero = target - target_center[:, None]
    covariance = torch.einsum("bn,bni,bnj->bij", weights, target_zero, source_zero)
    u, singular, vh = torch.linalg.svd(covariance.float(), full_matrices=False)
    sign = torch.sign(torch.det(u @ vh)).detach()
    correction = torch.eye(3, device=source.device, dtype=torch.float32).repeat(len(source), 1, 1)
    correction[:, 2, 2] = sign
    rotation = (u @ correction @ vh).to(source.dtype)
    if estimate_scale:
        numerator = (singular * torch.stack([torch.ones_like(sign), torch.ones_like(sign), sign], dim=-1)).sum(dim=-1)
        denominator = (weights * source_zero.square().sum(dim=-1)).sum(dim=-1).clamp_min(1e-6)
        scale = (numerator / denominator).clamp(0.35, 2.5).to(source.dtype)
    else:
        scale = torch.ones(len(source), device=source.device, dtype=source.dtype)
    translation = target_center - scale[:, None] * torch.einsum("bij,bj->bi", rotation, source_center)
    return scale, rotation, translation


class ExplicitCorrespondenceBridge(nn.Module):
    def __init__(self, scene_count: int, estimate_scale: bool, hard_human_anchors: bool = False):
        super().__init__()
        self.scene_count = scene_count
        self.estimate_scale = estimate_scale
        self.hard_human_anchors = hard_human_anchors
        self.encoder = EdgeEncoder()
        self.target_attention = nn.MultiheadAttention(96, 4, batch_first=True, dropout=0.05)
        self.source_attention = nn.MultiheadAttention(96, 4, batch_first=True, dropout=0.05)
        self.fuse = nn.Sequential(nn.Linear(192, 96), nn.LayerNorm(96), nn.SiLU(), nn.Linear(96, 96))
        self.confidence = nn.Sequential(nn.Linear(97, 64), nn.SiLU(), nn.Linear(64, 1))
        self.anchor_strength = nn.Parameter(torch.tensor(-1.5, dtype=torch.float32))
        self.log_temperature = nn.Parameter(torch.tensor(math.log(0.10), dtype=torch.float32))

    def forward(self, target: torch.Tensor, source: torch.Tensor) -> dict[str, torch.Tensor]:
        target_values, target_normalized = normalized_features(target, self.scene_count)
        source_values, source_normalized = normalized_features(source, self.scene_count)
        target_embedding = self.encoder(target_values, target_normalized)
        source_embedding = self.encoder(source_values, source_normalized)
        target_cross, _ = self.target_attention(target_embedding, source_embedding, source_embedding, need_weights=False)
        source_cross, _ = self.source_attention(source_embedding, target_embedding, target_embedding, need_weights=False)
        target_embedding = F.normalize(self.fuse(torch.cat([target_embedding, target_cross], dim=-1)), dim=-1)
        source_embedding = F.normalize(self.fuse(torch.cat([source_embedding, source_cross], dim=-1)), dim=-1)
        temperature = self.log_temperature.exp().clamp(0.025, 0.40)
        logits = torch.einsum("bnd,bmd->bnm", source_embedding, target_embedding) / temperature
        probability = torch.softmax(logits, dim=-1)
        matched_target = probability @ target[..., :3]
        peak = probability.max(dim=-1).values
        weights = torch.sigmoid(self.confidence(torch.cat([source_embedding, peak[..., None]], dim=-1)).squeeze(-1)) * peak
        if self.hard_human_anchors:
            matched_target = torch.cat([matched_target[:, : self.scene_count], target[:, self.scene_count :, :3]], dim=1)
            anchor_weight = torch.sigmoid(self.anchor_strength)
            weights = torch.cat(
                [weights[:, : self.scene_count], anchor_weight * torch.ones_like(weights[:, self.scene_count :])],
                dim=1,
            )
        scale, rotation, translation = weighted_umeyama(
            source[..., :3],
            matched_target,
            weights,
            self.estimate_scale,
        )
        return {
            "scale": scale,
            "rotation": rotation,
            "translation": translation,
            "logits": logits,
            "weights": weights,
            "matched_target": matched_target,
        }


def geodesic_radians(estimated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    relative = estimated @ target.transpose(-1, -2)
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cosine)


def training_loss(output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    rotation = geodesic_radians(output["rotation"], batch["rotation"])
    translation = F.smooth_l1_loss(output["translation"], batch["translation"], reduction="none", beta=0.20).mean(dim=-1)
    scale = F.smooth_l1_loss(torch.log(output["scale"]), torch.log(batch["scale"]), reduction="none", beta=0.10)
    source = batch["source"][..., :3]
    estimated_points = output["scale"][:, None, None] * torch.einsum("bij,bnj->bni", output["rotation"], source) + output["translation"][:, None]
    target_points = batch["scale"][:, None, None] * torch.einsum("bij,bnj->bni", batch["rotation"], source) + batch["translation"][:, None]
    point = F.smooth_l1_loss(estimated_points, target_points, reduction="none", beta=0.20).mean(dim=(-1, -2))
    labels = batch["correspondence"]
    valid = labels >= 0
    if valid.any():
        correspondence = F.cross_entropy(output["logits"][valid], labels[valid])
    else:
        correspondence = torch.zeros((), device=source.device)
    loss = (2.5 * rotation + 2.0 * translation + 1.5 * scale + point).mean() + 0.35 * correspondence
    return loss, {
        "rotation_deg": float(torch.rad2deg(rotation).mean().detach()),
        "translation_m": float(torch.linalg.vector_norm(output["translation"] - batch["translation"], dim=-1).mean().detach()),
        "scale_log": float(torch.abs(torch.log(output["scale"] / batch["scale"])).mean().detach()),
        "correspondence": float(correspondence.detach()),
    }


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
    }


def rotation_error_deg(estimated: np.ndarray, target: np.ndarray) -> float:
    cosine = np.clip((np.trace(estimated @ target.T) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


@torch.no_grad()
def predict_numpy(model: ExplicitCorrespondenceBridge, target: np.ndarray, source: np.ndarray, device: torch.device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    output = model(
        torch.from_numpy(target[None]).to(device),
        torch.from_numpy(source[None]).to(device),
    )
    return (
        float(output["scale"][0].cpu()),
        output["rotation"][0].cpu().numpy(),
        output["translation"][0].cpu().numpy(),
    )


@torch.no_grad()
def predict_detailed(model: ExplicitCorrespondenceBridge, target: np.ndarray, source: np.ndarray, device: torch.device) -> dict[str, np.ndarray | float]:
    model.eval()
    output = model(torch.from_numpy(target[None]).to(device), torch.from_numpy(source[None]).to(device))
    return {
        "scale": float(output["scale"][0].cpu()),
        "rotation": output["rotation"][0].cpu().numpy(),
        "translation": output["translation"][0].cpu().numpy(),
        "weights": output["weights"][0].cpu().numpy(),
        "matched_target": output["matched_target"][0].cpu().numpy(),
    }


def synthetic_evaluation(
    model: ExplicitCorrespondenceBridge,
    batcher: SyntheticBatcher,
    device: torch.device,
    augmentations: int,
) -> dict:
    result = {}
    for bucket in BUCKETS:
        rows = []
        for _ in range(len(batcher.indices) * augmentations):
            sample = batcher.one(bucket=bucket, exact=False)
            scale, rotation, translation = predict_numpy(model, sample["target"], sample["source"], device)
            rows.append(
                {
                    "rotation_deg": rotation_error_deg(rotation, sample["rotation"]),
                    "translation_m": float(np.linalg.norm(translation - sample["translation"])),
                    "log_scale": float(abs(math.log(scale / float(sample["scale"])))),
                }
            )
        result[bucket] = {key: summarize([row[key] for row in rows]) for key in rows[0]}
    return result


def prepare_real_pair(
    cache: Cache,
    case_index: int,
    variant: str,
    points: int,
    rng: np.random.Generator,
    prealign_fixed: bool,
) -> tuple[np.ndarray, np.ndarray]:
    use_da3 = variant.startswith("da3")
    use_human = variant.endswith("human")
    target_pool = cache.points[case_index, 1]
    source_pool = cache.points[case_index, 2]
    target = target_pool[rng.choice(len(target_pool), size=points, replace=False)].copy()
    source = source_pool[rng.choice(len(source_pool), size=points, replace=False)].copy()
    target_scale = float(cache.scales[case_index, 1]) if use_da3 else 1.0
    source_scale = float(cache.scales[case_index, 2]) if use_da3 else 1.0
    target[:, :3] *= target_scale
    source[:, :3] *= source_scale
    fixed = cache.fixed[case_index]
    if prealign_fixed:
        source[:, :3] = source[:, :3] @ fixed[:3, :3].T + fixed[:3, 3]
    if use_human:
        target_anchor = cache.anchors[case_index, 1] * target_scale
        source_anchor = cache.anchors[case_index, 2] * source_scale
        if prealign_fixed:
            source_anchor = source_anchor @ fixed[:3, :3].T + fixed[:3, 3]
        target = append_human(target, target_anchor)
        source = append_human(source, source_anchor)
    else:
        target = scene_only(target)
        source = scene_only(source)
    return target, source


def prepare_real_translation_pair(
    cache: Cache,
    case_index: int,
    variant: str,
    points: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    use_da3 = variant.startswith("da3")
    use_human = variant.endswith("human")
    target_pool = cache.points[case_index, 1]
    source_pool = cache.points[case_index, 2]
    target = target_pool[rng.choice(len(target_pool), size=points, replace=False)].copy()
    source_raw = source_pool[rng.choice(len(source_pool), size=points, replace=False)].copy()
    target_scale = float(cache.scales[case_index, 1]) if use_da3 else 1.0
    source_scale = float(cache.scales[case_index, 2]) if use_da3 else 1.0
    target[:, :3] *= target_scale
    source_raw[:, :3] *= source_scale
    fixed = cache.fixed[case_index]
    source_fixed = source_raw.copy()
    source_fixed[:, :3] = source_fixed[:, :3] @ fixed[:3, :3].T + fixed[:3, 3]
    if use_human:
        target_anchor = cache.anchors[case_index, 1] * target_scale
        source_anchor_raw = cache.anchors[case_index, 2] * source_scale
        source_anchor_fixed = source_anchor_raw @ fixed[:3, :3].T + fixed[:3, 3]
        target = append_human(target, target_anchor)
        source_raw = append_human(source_raw, source_anchor_raw)
        source_fixed = append_human(source_fixed, source_anchor_fixed)
    else:
        target = scene_only(target)
        source_raw = scene_only(source_raw)
        source_fixed = scene_only(source_fixed)
    return target, source_fixed, source_raw


def robust_translation_from_matches(
    source_raw: np.ndarray,
    matched_target: np.ndarray,
    weights: np.ndarray,
    rotation: np.ndarray,
) -> np.ndarray:
    rotated = source_raw[:, :3] @ rotation.T
    offsets = matched_target - rotated
    initial = np.median(offsets, axis=0)
    residual = np.linalg.norm(offsets - initial[None], axis=-1)
    keep = residual <= np.quantile(residual, 0.60)
    robust_weights = np.maximum(weights, 1e-5) * keep.astype(np.float32)
    if float(robust_weights.sum()) < 1e-6:
        robust_weights = np.maximum(weights, 1e-5)
    return ((robust_weights[:, None] * offsets).sum(axis=0) / robust_weights.sum()).astype(np.float32)


def alignment_score(target: np.ndarray, source: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray, scene_count: int) -> float:
    transformed = scale * (source[:scene_count, :3] @ rotation.T) + translation
    distances = cKDTree(target[:scene_count, :3]).query(transformed, k=1)[0]
    keep = distances <= np.quantile(distances, 0.70)
    return float(np.mean(distances[keep])) if keep.any() else float(np.mean(distances))


def compose_with_fixed(scale: float, rotation: np.ndarray, translation: np.ndarray, fixed: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    return (
        scale,
        (rotation @ fixed[:3, :3]).astype(np.float32),
        (scale * rotation @ fixed[:3, 3] + translation).astype(np.float32),
    )


@torch.no_grad()
def real_cut_evaluation(
    model: ExplicitCorrespondenceBridge,
    cache: Cache,
    indices: np.ndarray,
    variant: str,
    points: int,
    repeats: int,
    device: torch.device,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    cases = []
    for case_index in indices:
        mode_predictions = {}
        for mode, prealign_fixed in (("direct", False), ("after_fixed", True)):
            predictions = []
            for _ in range(repeats):
                target_points, source_points = prepare_real_pair(
                    cache,
                    int(case_index),
                    variant,
                    points,
                    rng,
                    prealign_fixed=prealign_fixed,
                )
                prediction = predict_numpy(model, target_points, source_points, device)
                score = alignment_score(target_points, source_points, *prediction, scene_count=points)
                predictions.append((*prediction, score))
            scale, rotation, translation, score = min(predictions, key=lambda item: item[-1])
            if prealign_fixed:
                scale, rotation, translation = compose_with_fixed(scale, rotation, translation, cache.fixed[case_index])
            mode_predictions[mode] = (scale, rotation, translation, score)
        target = cache.target[case_index]
        fixed = cache.fixed[case_index]
        direct = mode_predictions["direct"]
        after_fixed = mode_predictions["after_fixed"]
        torso_predictions = []
        for _ in range(repeats):
            target_points, source_fixed, source_raw = prepare_real_translation_pair(
                cache,
                int(case_index),
                variant,
                points,
                rng,
            )
            detail = predict_detailed(model, target_points, source_fixed, device)
            torso_rotation = cache.torso[case_index, :3, :3]
            torso_translation = robust_translation_from_matches(
                source_raw,
                detail["matched_target"],
                detail["weights"],
                torso_rotation,
            )
            torso_score = alignment_score(
                target_points,
                source_raw,
                1.0,
                torso_rotation,
                torso_translation,
                scene_count=points,
            )
            torso_predictions.append((torso_translation, torso_score))
        torso_translation, torso_score = min(torso_predictions, key=lambda item: item[-1])
        cases.append(
            {
                "case_name": cache.rows[case_index]["case_name"],
                "source": cache.rows[case_index]["source"],
                "view_angle_deg": float(cache.rows[case_index]["view_angle_deg"]),
                "direct_scale": direct[0],
                "direct_rotation_deg": rotation_error_deg(direct[1], target[:3, :3]),
                "direct_translation_m": float(np.linalg.norm(direct[2] - target[:3, 3])),
                "direct_alignment_score": direct[3],
                "after_fixed_scale": after_fixed[0],
                "after_fixed_rotation_deg": rotation_error_deg(after_fixed[1], target[:3, :3]),
                "after_fixed_translation_m": float(np.linalg.norm(after_fixed[2] - target[:3, 3])),
                "after_fixed_alignment_score": after_fixed[3],
                "torso_learned_translation_rotation_deg": rotation_error_deg(cache.torso[case_index, :3, :3], target[:3, :3]),
                "torso_learned_translation_m": float(np.linalg.norm(torso_translation - target[:3, 3])),
                "torso_learned_alignment_score": torso_score,
                "fixed_rotation_deg": rotation_error_deg(fixed[:3, :3], target[:3, :3]),
                "fixed_translation_m": float(np.linalg.norm(fixed[:3, 3] - target[:3, 3])),
            }
        )
    overall = {
        "learned_direct": {
            "rotation_deg": summarize([case["direct_rotation_deg"] for case in cases]),
            "translation_m": summarize([case["direct_translation_m"] for case in cases]),
        },
        "learned_after_fixed": {
            "rotation_deg": summarize([case["after_fixed_rotation_deg"] for case in cases]),
            "translation_m": summarize([case["after_fixed_translation_m"] for case in cases]),
        },
        "torso_rotation_learned_translation": {
            "rotation_deg": summarize([case["torso_learned_translation_rotation_deg"] for case in cases]),
            "translation_m": summarize([case["torso_learned_translation_m"] for case in cases]),
        },
        "fixed_explicit": {
            "rotation_deg": summarize([case["fixed_rotation_deg"] for case in cases]),
            "translation_m": summarize([case["fixed_translation_m"] for case in cases]),
        },
    }
    return {"overall": overall, "cases": cases}


def main() -> None:
    args = parse_args()
    seed_all(int(args.seed))
    device = torch.device(args.device)
    cache = load_cache(args.cache_dir)
    train_indices = np.asarray([i for i, row in enumerate(cache.rows) if row["source"] != args.held_out_source], dtype=np.int64)
    test_indices = np.asarray([i for i, row in enumerate(cache.rows) if row["source"] == args.held_out_source], dtype=np.int64)
    train_batcher = SyntheticBatcher(
        cache,
        train_indices,
        args.variant,
        args.points,
        int(args.seed),
        real_cut_probability=float(args.real_cut_training_probability),
    )
    test_batcher = SyntheticBatcher(cache, test_indices, args.variant, args.points, int(args.seed) + 17)
    model = ExplicitCorrespondenceBridge(
        scene_count=int(args.points),
        estimate_scale=args.variant == "raw_sim3",
        hard_human_anchors=args.variant.endswith("human"),
    ).to(device)
    if args.checkpoint:
        checkpoint_data = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint_data["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(args.steps), 1), eta_min=2e-6)
    start = time.perf_counter()
    history = []
    for step in range(0 if args.eval_only else int(args.steps)):
        model.train()
        batch = to_device(
            train_batcher.batch(int(args.batch_size), pretrain=step < min(350, int(args.steps) // 4)),
            device,
        )
        optimizer.zero_grad(set_to_none=True)
        output = model(batch["target"], batch["source"])
        loss, metrics = training_loss(output, batch)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {float(loss)}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        if step % 100 == 0 or step + 1 == int(args.steps):
            row = {"step": step + 1, "loss": float(loss.detach()), **metrics}
            history.append(row)
            print(json.dumps(row), flush=True)

    synthetic = synthetic_evaluation(model, test_batcher, device, int(args.eval_augmentations))
    real = real_cut_evaluation(
        model,
        cache,
        test_indices,
        args.variant,
        int(args.points),
        int(args.real_eval_repeats),
        device,
        int(args.seed) + 101,
    )
    output_dir = args.output_dir / f"heldout_{args.held_out_source}" / args.variant
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "model.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "variant": args.variant,
            "held_out_source": args.held_out_source,
            "points": int(args.points),
        },
        checkpoint,
    )
    report = {
        "experiment": "V54 synthetic explicit geometry shot bridge",
        "variant": args.variant,
        "held_out_source": args.held_out_source,
        "train_case_count": int(len(train_indices)),
        "test_case_count": int(len(test_indices)),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "elapsed_seconds": float(time.perf_counter() - start),
        "protocol": {
            "raw_tokens_used": False,
            "gt_depth_used": False,
            "training_target": "known synthetic SE(3)/Sim(3) perturbation on continuous-shot Human3R geometry",
            "test": "held-out data source synthetic perturbations and real camera cuts",
            "post_cut_frames": 1,
            "inference": "one correspondence forward pass plus differentiable Umeyama solve",
        },
        "history": history,
        "synthetic_held_out": synthetic,
        "real_held_out": real,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "synthetic": synthetic, "real": real["overall"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
