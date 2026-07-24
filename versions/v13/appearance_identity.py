"""Frozen predicted-bbox appearance cues and precision-first identity gating.

This module only answers WHO.  It deliberately has no camera, Boundary, or
SE(3) solver and never consumes evaluator identity labels.
"""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from versions.v13.identity_bridge import (
    MatchConfig,
    build_identity_bank,
    feature_distance,
    frame_feature,
    match_identity_bank,
)


ENCODER_NAME = "dinov2_vits14"
ENCODER_DIM = 768
ENCODER_INPUT_SIZE = 224
ENCODER_CHECKPOINT_SHA256 = (
    "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
)
IMAGE_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGE_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_rows(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-12)


def equalized_concat(*values: np.ndarray) -> np.ndarray:
    return normalize_rows(np.concatenate([normalize_rows(value) for value in values], axis=1))


def crop_predicted_bbox(
    image_rgb: np.ndarray,
    bbox_processed: np.ndarray,
    processed_shape: tuple[int, int],
    padding_ratio: float = 0.08,
) -> tuple[np.ndarray | None, dict]:
    """Crop a Human3R-predicted body box from the uncropped RGB image."""
    height, width = image_rgb.shape[:2]
    processed_height, processed_width = (int(value) for value in processed_shape)
    bbox = np.asarray(bbox_processed, dtype=np.float64).reshape(-1)
    if bbox.shape != (4,) or not np.isfinite(bbox).all():
        return None, {"valid": False, "reason": "invalid_bbox"}
    scaled = bbox.copy()
    scaled[[0, 2]] *= width / max(processed_width, 1)
    scaled[[1, 3]] *= height / max(processed_height, 1)
    box_width = float(scaled[2] - scaled[0])
    box_height = float(scaled[3] - scaled[1])
    if box_width < 4.0 or box_height < 8.0:
        return None, {"valid": False, "reason": "bbox_too_small", "bbox_rgb": scaled}
    scaled[[0, 2]] += np.asarray((-1.0, 1.0)) * padding_ratio * box_width
    scaled[[1, 3]] += np.asarray((-1.0, 1.0)) * padding_ratio * box_height
    x0 = int(np.clip(math.floor(scaled[0]), 0, width - 1))
    y0 = int(np.clip(math.floor(scaled[1]), 0, height - 1))
    x1 = int(np.clip(math.ceil(scaled[2]), x0 + 1, width))
    y1 = int(np.clip(math.ceil(scaled[3]), y0 + 1, height))
    crop = image_rgb[y0:y1, x0:x1]
    valid = crop.shape[0] >= 8 and crop.shape[1] >= 4
    return (
        crop if valid else None,
        {
            "valid": bool(valid),
            "reason": "ok" if valid else "empty_crop",
            "bbox_rgb": np.asarray((x0, y0, x1, y1), dtype=np.int64),
            "bbox_processed": bbox,
            "crop_shape": crop.shape[:2],
        },
    )


def prepare_crop(crop_rgb: np.ndarray, size: int = ENCODER_INPUT_SIZE) -> torch.Tensor:
    height, width = crop_rgb.shape[:2]
    scale = min(size / max(width, 1), size / max(height, 1))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(
        crop_rgb, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    )
    canvas = np.empty((size, size, 3), dtype=np.uint8)
    canvas[...] = np.round(255.0 * IMAGE_MEAN).astype(np.uint8)
    x0 = (size - resized_width) // 2
    y0 = (size - resized_height) // 2
    canvas[y0 : y0 + resized_height, x0 : x0 + resized_width] = resized
    value = canvas.astype(np.float32) / 255.0
    value = (value - IMAGE_MEAN) / IMAGE_STD
    return torch.from_numpy(value).permute(2, 0, 1).contiguous()


class FrozenDinoAppearance:
    """Official DINOv2-S/14 with fixed CLS plus mean-patch pooling."""

    def __init__(
        self,
        device: str,
        hub_dir: Path,
        checkpoint: Path,
        batch_size: int = 32,
    ) -> None:
        if sha256(checkpoint) != ENCODER_CHECKPOINT_SHA256:
            raise ValueError(f"Unexpected DINOv2 checkpoint hash: {checkpoint}")
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        model = torch.hub.load(
            str(hub_dir), ENCODER_NAME, source="local", pretrained=False
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        self.model = model.eval().to(self.device)
        self.model.requires_grad_(False)

    @torch.inference_mode()
    def encode(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.empty((0, ENCODER_DIM), dtype=np.float32)
        output = []
        for start in range(0, len(crops), self.batch_size):
            batch = torch.stack(
                [prepare_crop(crop) for crop in crops[start : start + self.batch_size]]
            ).to(self.device, non_blocking=True)
            with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                features = self.model.forward_features(batch)
                cls = F.normalize(features["x_norm_clstoken"].float(), dim=-1)
                patch = F.normalize(
                    features["x_norm_patchtokens"].float().mean(dim=1), dim=-1
                )
                embedding = F.normalize(torch.cat((cls, patch), dim=-1), dim=-1)
            output.append(embedding.cpu().numpy().astype(np.float32))
        return np.concatenate(output, axis=0)


def augment_frame(frame: dict, appearance: np.ndarray, valid: np.ndarray) -> dict:
    output = copy.deepcopy(frame)
    count = int(output["count"])
    appearance = np.asarray(appearance, dtype=np.float32).reshape(count, ENCODER_DIM)
    valid = np.asarray(valid, dtype=bool).reshape(count)
    if len(appearance) and appearance.shape[1] != ENCODER_DIM:
        raise ValueError(f"Unexpected appearance dimension: {appearance.shape}")
    beta_value = np.asarray(output["features"]["smpl_beta"], dtype=np.float32)
    pose_value = np.asarray(output["features"]["local_pose"], dtype=np.float32)
    native_value = np.asarray(output["features"]["mhmr_head_tokens"], dtype=np.float32)
    beta = beta_value.reshape(count, beta_value.shape[-1])
    pose = pose_value.reshape(count, pose_value.shape[-1])
    native = np.asarray(
        native_value, dtype=np.float32
    ).reshape(count, native_value.shape[-1])
    output["features"].update(
        {
            "appearance": appearance,
            "appearance_beta": equalized_concat(appearance, beta),
            "appearance_pose": equalized_concat(appearance, pose),
            "appearance_beta_pose": equalized_concat(appearance, beta, pose),
            "appearance_native": equalized_concat(appearance, native),
        }
    )
    output["appearance_valid"] = valid
    return output


@dataclass(frozen=True)
class PrecisionGateConfig:
    feature: str
    prototype: str
    distance: str
    max_primary_distance: float = float("inf")
    min_primary_margin: float = 0.0
    min_vote_fraction: float = 0.0
    max_beta_distance: float = float("inf")
    max_pose_distance: float = float("inf")
    min_valid_observations: int = 1
    require_mutual: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def _cue_cost(
    pre_frames: list[dict],
    post_frame: dict,
    cue: str,
    prototype: str,
    distance: str,
) -> tuple[np.ndarray, np.ndarray]:
    bank = build_identity_bank(pre_frames, cue, prototype, distance)
    target = frame_feature(post_frame, cue)
    return np.asarray(bank["track_ids"], dtype=np.int64), feature_distance(
        bank["prototypes"], target, distance
    )


def _vote_fraction(
    pre_frames: list[dict], post_frame: dict, feature: str, distance: str
) -> dict[int, np.ndarray]:
    target = frame_feature(post_frame, feature)
    votes: dict[int, list[np.ndarray]] = {}
    for frame in pre_frames:
        source = frame_feature(frame, feature)
        track_ids = np.asarray(frame["track_ids"], dtype=np.int64)
        if not len(source) or not len(target):
            continue
        nearest = np.argmin(feature_distance(source, target, distance), axis=1)
        for detection, track_id in enumerate(track_ids):
            if int(track_id) < 0:
                continue
            row = np.zeros(len(target), dtype=np.float64)
            row[int(nearest[detection])] = 1.0
            votes.setdefault(int(track_id), []).append(row)
    return {
        track_id: np.mean(np.stack(rows), axis=0)
        for track_id, rows in votes.items()
    }


def precision_signals(
    pre_frames: list[dict], post_frame: dict, config: PrecisionGateConfig
) -> tuple[dict, list[dict]]:
    """Return complete Hungarian proposals plus causal gate diagnostics."""
    base = match_identity_bank(
        pre_frames,
        post_frame,
        MatchConfig(
            feature=config.feature,
            prototype=config.prototype,
            distance=config.distance,
            matcher="hungarian",
            max_cost=float("inf"),
        ),
    )
    track_ids = np.asarray(base["bank"]["track_ids"], dtype=np.int64)
    beta_ids, beta_cost = _cue_cost(
        pre_frames, post_frame, "smpl_beta", config.prototype, "normalized_l2"
    )
    pose_ids, pose_cost = _cue_cost(
        pre_frames, post_frame, "local_pose", config.prototype, "cosine"
    )
    if not np.array_equal(track_ids, beta_ids) or not np.array_equal(track_ids, pose_ids):
        raise ValueError("Cue banks do not share the same external track IDs")
    cost = np.asarray(base["cost"], dtype=np.float64)
    row_best = np.argmin(cost, axis=1) if cost.size else np.empty(0, dtype=np.int64)
    col_best = np.argmin(cost, axis=0) if cost.size else np.empty(0, dtype=np.int64)
    votes = _vote_fraction(pre_frames, post_frame, config.feature, config.distance)
    valid_counts: dict[int, int] = {}
    for frame in pre_frames:
        ids = np.asarray(frame["track_ids"], dtype=np.int64)
        valid = np.asarray(frame.get("appearance_valid", np.ones(len(ids))), dtype=bool)
        for detection, track_id in enumerate(ids):
            if int(track_id) >= 0 and bool(valid[detection]):
                valid_counts[int(track_id)] = valid_counts.get(int(track_id), 0) + 1
    target_valid = np.asarray(
        post_frame.get("appearance_valid", np.ones(int(post_frame["count"]))),
        dtype=bool,
    )
    signals = []
    for pair_index, pair in enumerate(base["accepted_pairs"]):
        source = int(pair["source_index"])
        target = int(pair["target_index"])
        row_values = np.sort(cost[source])
        col_values = np.sort(cost[:, target])
        row_margin = float(row_values[1] - row_values[0]) if len(row_values) > 1 else float("inf")
        col_margin = float(col_values[1] - col_values[0]) if len(col_values) > 1 else float("inf")
        track_id = int(track_ids[source])
        vote = votes.get(track_id, np.zeros(cost.shape[1], dtype=np.float64))
        signals.append(
            {
                "pair_index": pair_index,
                "source_index": source,
                "target_index": target,
                "track_id": track_id,
                "primary_distance": float(cost[source, target]),
                "primary_row_margin": row_margin,
                "primary_column_margin": col_margin,
                "primary_margin": min(row_margin, col_margin),
                "mutual_nearest": bool(
                    len(row_best)
                    and len(col_best)
                    and int(row_best[source]) == target
                    and int(col_best[target]) == source
                ),
                "vote_fraction": float(vote[target]) if len(vote) else 0.0,
                "beta_distance": float(beta_cost[source, target]),
                "pose_distance": float(pose_cost[source, target]),
                "valid_observations": int(valid_counts.get(track_id, 0)),
                "target_appearance_valid": bool(target_valid[target]),
            }
        )
    return base, signals


def signal_is_accepted(signal: dict, config: PrecisionGateConfig) -> bool:
    return bool(
        (signal["mutual_nearest"] or not config.require_mutual)
        and signal["target_appearance_valid"]
        and signal["valid_observations"] >= int(config.min_valid_observations)
        and signal["primary_distance"] <= float(config.max_primary_distance)
        and signal["primary_margin"] >= float(config.min_primary_margin)
        and signal["vote_fraction"] >= float(config.min_vote_fraction)
        and signal["beta_distance"] <= float(config.max_beta_distance)
        and signal["pose_distance"] <= float(config.max_pose_distance)
    )


def apply_precision_gate(
    base: dict, signals: list[dict], config: PrecisionGateConfig
) -> dict:
    output = copy.deepcopy(base)
    accepted_pair_indices = {
        int(signal["pair_index"])
        for signal in signals
        if signal_is_accepted(signal, config)
    }
    pairs = []
    for pair_index, pair in enumerate(output["accepted_pairs"]):
        row = dict(pair)
        row["accepted"] = pair_index in accepted_pair_indices
        pairs.append(row)
    accepted = [row for row in pairs if row["accepted"]]
    accepted_source = {int(row["source_index"]) for row in accepted}
    accepted_target = {int(row["target_index"]) for row in accepted}
    source_count, target_count = output["cost"].shape
    output["pairs"] = pairs
    output["accepted_pairs"] = accepted
    output["unmatched_source"] = np.asarray(
        [index for index in range(source_count) if index not in accepted_source],
        dtype=np.int64,
    )
    output["unmatched_target"] = np.asarray(
        [index for index in range(target_count) if index not in accepted_target],
        dtype=np.int64,
    )
    return output
