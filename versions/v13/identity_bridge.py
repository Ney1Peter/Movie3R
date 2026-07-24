"""Training-free cross-shot identity association for Movie3R-Multi V13.

This module deliberately contains no Boundary solver. It maps post-cut
detections to pre-cut external track IDs; the frozen geometric path remains
responsible for the shared shot transform.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from dust3r.utils.image import log_optimal_transport


BASE_FEATURES = (
    "refined_human_tokens",
    "cut3r_head_tokens",
    "mhmr_head_tokens",
    "fused_human_prompts",
    "smpl_beta",
    "local_pose",
)

FEATURE_COMPONENTS = {
    "refined_human_tokens": ("refined_human_tokens",),
    "cut3r_head_tokens": ("cut3r_head_tokens",),
    "mhmr_head_tokens": ("mhmr_head_tokens",),
    "fused_human_prompts": ("fused_human_prompts",),
    "smpl_beta": ("smpl_beta",),
    "local_pose": ("local_pose",),
    "cut3r_beta": ("cut3r_head_tokens", "smpl_beta"),
    "refined_beta": ("refined_human_tokens", "smpl_beta"),
    "refined_beta_pose": (
        "refined_human_tokens",
        "smpl_beta",
        "local_pose",
    ),
    "beta_pose": ("smpl_beta", "local_pose"),
}


@dataclass(frozen=True)
class MatchConfig:
    feature: str
    prototype: str = "mean"
    distance: str = "normalized_l2"
    matcher: str = "hungarian"
    max_cost: float = float("inf")
    sinkhorn_score_threshold: float = 0.2
    sinkhorn_alpha: float = -10.0
    sinkhorn_iterations: int = 20


@dataclass
class IdentityTracklet:
    track_id: int
    observations: list[dict] = field(default_factory=list)
    last_seen: int = 0
    active: bool = True


class CausalIdentityMemory:
    """External WHO memory with Match-Then-Align / Align-Then-Commit semantics."""

    def __init__(self, ttl: int = 8, prototype_window: int = 5):
        if ttl < 1 or prototype_window < 1:
            raise ValueError("ttl and prototype_window must be positive")
        self.ttl = int(ttl)
        self.prototype_window = int(prototype_window)
        self.tracklets: dict[int, IdentityTracklet] = {}
        self.next_track_id = 0
        self.commit_count = 0

    @staticmethod
    def _observation(frame: dict, detection_index: int) -> dict:
        return {
            "count": 1,
            "track_ids": np.asarray([-1], dtype=np.int64),
            "head_scores": np.asarray(
                [np.asarray(frame.get("head_scores", np.ones(frame["count"])))[detection_index]]
            ),
            "features": {
                name: _rows(value, int(frame["count"]), name)[detection_index : detection_index + 1].copy()
                for name, value in frame["features"].items()
            },
        }

    def _append(self, track_id: int, frame: dict, detection_index: int, timestamp: int) -> None:
        observation = self._observation(frame, detection_index)
        observation["track_ids"][0] = int(track_id)
        tracklet = self.tracklets.setdefault(
            int(track_id), IdentityTracklet(int(track_id))
        )
        tracklet.observations.append(observation)
        tracklet.observations = tracklet.observations[-self.prototype_window :]
        tracklet.last_seen = int(timestamp)
        tracklet.active = True
        self.next_track_id = max(self.next_track_id, int(track_id) + 1)

    def bootstrap(self, frame: dict, timestamp: int, use_native_ids: bool = True) -> np.ndarray:
        """Initialize a no-cut shot before the first cross-shot association."""
        count = int(frame["count"])
        native = np.asarray(frame.get("track_ids", np.full(count, -1)), dtype=np.int64)
        output = np.full(count, -1, dtype=np.int64)
        for detection_index in range(count):
            if use_native_ids and detection_index < len(native) and int(native[detection_index]) >= 0:
                track_id = int(native[detection_index])
            else:
                track_id = self.next_track_id
            output[detection_index] = track_id
            self._append(track_id, frame, detection_index, timestamp)
        self.commit_count += 1
        return output

    def observe(self, frame: dict, track_ids: np.ndarray, timestamp: int) -> None:
        """Commit a normal no-cut frame whose external IDs are already known."""
        track_ids = np.asarray(track_ids, dtype=np.int64).reshape(-1)
        if len(track_ids) != int(frame["count"]):
            raise ValueError("track_ids and frame detections have different lengths")
        for tracklet in self.tracklets.values():
            tracklet.active = False
        for detection_index, track_id in enumerate(track_ids):
            if int(track_id) < 0:
                continue
            self._append(int(track_id), frame, detection_index, timestamp)
        self.commit_count += 1
        self.expire(timestamp)

    def _prototype_frames(self, timestamp: int) -> list[dict]:
        frames = []
        for track_id in sorted(self.tracklets):
            tracklet = self.tracklets[track_id]
            if int(timestamp) - int(tracklet.last_seen) <= self.ttl:
                frames.extend(tracklet.observations)
        return frames

    def tentative_match(self, post_frame: dict, config: MatchConfig, timestamp: int) -> dict:
        """Compute association without mutating any tracklet or prototype."""
        return match_identity_bank(self._prototype_frames(timestamp), post_frame, config)

    def commit(self, post_frame: dict, result: dict, timestamp: int) -> np.ndarray:
        """Commit only after alignment/verification and allocate IDs for new people."""
        for tracklet in self.tracklets.values():
            tracklet.active = False
        output = np.full(int(post_frame["count"]), -1, dtype=np.int64)
        bank_ids = np.asarray(result["bank"]["track_ids"], dtype=np.int64)
        for row in result["accepted_pairs"]:
            source = int(row["source_index"])
            target = int(row["target_index"])
            if source >= len(bank_ids) or target >= len(output):
                continue
            track_id = int(bank_ids[source])
            output[target] = track_id
            self._append(track_id, post_frame, target, timestamp)
        for target in range(len(output)):
            if output[target] >= 0:
                continue
            track_id = self.next_track_id
            self.next_track_id += 1
            output[target] = track_id
            self._append(track_id, post_frame, target, timestamp)
        self.commit_count += 1
        self.expire(timestamp)
        return output

    def expire(self, timestamp: int) -> None:
        expired = [
            track_id
            for track_id, tracklet in self.tracklets.items()
            if int(timestamp) - int(tracklet.last_seen) > self.ttl
        ]
        for track_id in expired:
            del self.tracklets[track_id]

    def snapshot(self) -> dict:
        return {
            "next_track_id": self.next_track_id,
            "commit_count": self.commit_count,
            "tracks": {
                track_id: {
                    "history_count": len(tracklet.observations),
                    "last_seen": tracklet.last_seen,
                    "active": tracklet.active,
                }
                for track_id, tracklet in sorted(self.tracklets.items())
            },
        }


def _rows(value: np.ndarray, count: int, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    while value.ndim > 1 and value.shape[0] == 1 and value.shape[0] != count:
        value = value[0]
    if count == 0:
        return np.empty((0, 0), dtype=np.float32)
    if value.ndim == 1 and count == 1:
        value = value[None]
    if value.ndim < 2 or value.shape[0] != count:
        raise ValueError(f"Cannot interpret {name} shape {value.shape} for {count} people")
    return value.reshape(count, -1)


def _normalize_rows(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-12)


def frame_feature(frame: dict, name: str) -> np.ndarray:
    """Return one person row per detection, preserving detection order."""
    if name not in FEATURE_COMPONENTS:
        raise KeyError(name)
    count = int(frame["count"])
    components = [
        _rows(frame["features"][component], count, component)
        for component in FEATURE_COMPONENTS[name]
    ]
    if len(components) == 1:
        return components[0]
    # Equalize cue magnitudes before concatenation; this is a fixed, untrained
    # fusion and prevents a 1024-D token from numerically erasing beta.
    return np.concatenate([_normalize_rows(value) for value in components], axis=1)


def feature_distance(first: np.ndarray, second: np.ndarray, mode: str) -> np.ndarray:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if not len(first) or not len(second):
        return np.empty((len(first), len(second)), dtype=np.float64)
    if mode == "raw_l2":
        return cdist(first, second, metric="euclidean")
    if mode == "normalized_l2":
        return cdist(_normalize_rows(first), _normalize_rows(second), metric="euclidean")
    if mode == "cosine":
        first_normalized = _normalize_rows(first)
        second_normalized = _normalize_rows(second)
        return np.clip(1.0 - first_normalized @ second_normalized.T, 0.0, 2.0)
    raise ValueError(f"Unsupported feature distance: {mode}")


def _prototype(values: np.ndarray, mode: str, distance: str) -> np.ndarray:
    if len(values) == 1 or mode == "last":
        return values[-1]
    if mode == "mean":
        return np.mean(values, axis=0)
    if mode == "medoid":
        pairwise = feature_distance(values, values, distance)
        return values[int(np.argmin(pairwise.sum(axis=1)))]
    raise ValueError(f"Unsupported prototype: {mode}")


def build_identity_bank(
    frames: Iterable[dict], feature: str, prototype: str, distance: str
) -> dict:
    histories: dict[int, list[np.ndarray]] = {}
    frame_presence: dict[int, list[int]] = {}
    for frame_index, frame in enumerate(frames):
        values = frame_feature(frame, feature)
        track_ids = np.asarray(frame["track_ids"], dtype=np.int64).reshape(-1)
        if len(track_ids) != len(values):
            raise ValueError("track_ids and detection features have different lengths")
        for detection_index, track_id in enumerate(track_ids):
            if int(track_id) < 0:
                continue
            histories.setdefault(int(track_id), []).append(values[detection_index])
            frame_presence.setdefault(int(track_id), []).append(frame_index)
    track_ids = np.asarray(sorted(histories), dtype=np.int64)
    if len(track_ids):
        prototypes = np.stack(
            [
                _prototype(np.stack(histories[int(track_id)]), prototype, distance)
                for track_id in track_ids
            ]
        )
    else:
        prototypes = np.empty((0, 0), dtype=np.float64)
    return {
        "track_ids": track_ids,
        "prototypes": prototypes,
        "history_count": np.asarray(
            [len(histories[int(track_id)]) for track_id in track_ids], dtype=np.int64
        ),
        "last_frame": np.asarray(
            [frame_presence[int(track_id)][-1] for track_id in track_ids], dtype=np.int64
        ),
    }


def _hungarian(cost: np.ndarray, max_cost: float) -> dict:
    source_count, target_count = cost.shape
    pairs = []
    if source_count and target_count:
        source, target = linear_sum_assignment(cost)
        for source_index, target_index in zip(source, target):
            value = float(cost[source_index, target_index])
            pairs.append(
                {
                    "source_index": int(source_index),
                    "target_index": int(target_index),
                    "cost": value,
                    "score": float(-value),
                    "accepted": bool(value <= max_cost),
                }
            )
    return _finish_assignment(source_count, target_count, pairs)


def _sinkhorn(cost: np.ndarray, config: MatchConfig) -> dict:
    source_count, target_count = cost.shape
    pairs = []
    if source_count and target_count:
        scores = torch.as_tensor(-cost, dtype=torch.float32)[None]
        transport = log_optimal_transport(
            scores,
            alpha=torch.tensor(float(config.sinkhorn_alpha), dtype=scores.dtype),
            iters=int(config.sinkhorn_iterations),
        )
        matches = transport[:, :-1, :-1]
        max_source, max_target = matches.max(2), matches.max(1)
        indices_source = max_source.indices
        indices_target = max_target.indices
        mutual_source = (
            torch.arange(source_count)[None]
            == indices_target.gather(1, indices_source)
        )
        mutual_target = (
            torch.arange(target_count)[None]
            == indices_source.gather(1, indices_target)
        )
        probabilities_source = torch.where(
            mutual_source, max_source.values.exp(), torch.zeros_like(max_source.values)
        )
        for source_index in range(source_count):
            target_index = int(indices_source[0, source_index])
            mutual = bool(mutual_source[0, source_index] and mutual_target[0, target_index])
            probability = float(probabilities_source[0, source_index]) if mutual else 0.0
            pairs.append(
                {
                    "source_index": source_index,
                    "target_index": target_index,
                    "cost": float(cost[source_index, target_index]),
                    "score": probability,
                    "accepted": bool(
                        mutual and probability > float(config.sinkhorn_score_threshold)
                    ),
                }
            )
    output = _finish_assignment(source_count, target_count, pairs)
    output["log_transport"] = (
        transport[0].detach().cpu().numpy()
        if source_count and target_count
        else np.empty((source_count + 1, target_count + 1), dtype=np.float32)
    )
    return output


def _finish_assignment(source_count: int, target_count: int, pairs: list[dict]) -> dict:
    accepted = [row for row in pairs if row["accepted"]]
    accepted_source = {row["source_index"] for row in accepted}
    accepted_target = {row["target_index"] for row in accepted}
    return {
        "pairs": pairs,
        "accepted_pairs": accepted,
        "unmatched_source": np.asarray(
            [index for index in range(source_count) if index not in accepted_source],
            dtype=np.int64,
        ),
        "unmatched_target": np.asarray(
            [index for index in range(target_count) if index not in accepted_target],
            dtype=np.int64,
        ),
    }


def match_identity_bank(pre_frames: list[dict], post_frame: dict, config: MatchConfig) -> dict:
    bank = build_identity_bank(
        pre_frames, config.feature, config.prototype, config.distance
    )
    target = frame_feature(post_frame, config.feature)
    cost = feature_distance(bank["prototypes"], target, config.distance)
    if config.matcher == "hungarian":
        assignment = _hungarian(cost, float(config.max_cost))
    elif config.matcher == "sinkhorn":
        assignment = _sinkhorn(cost, config)
    else:
        raise ValueError(f"Unsupported matcher: {config.matcher}")
    return {"bank": bank, "target": target, "cost": cost, **assignment}


def majority_track_labels(pre_labels: list[np.ndarray], pre_frames: list[dict]) -> dict[int, int]:
    """Evaluator-only map from deployable track IDs to GT labels."""
    votes: dict[int, list[int]] = {}
    for labels, frame in zip(pre_labels, pre_frames):
        track_ids = np.asarray(frame["track_ids"], dtype=np.int64)
        for detection_index, track_id in enumerate(track_ids):
            label = int(labels[detection_index])
            if int(track_id) >= 0 and label >= 0:
                votes.setdefault(int(track_id), []).append(label)
    output = {}
    for track_id, labels in votes.items():
        values, counts = np.unique(labels, return_counts=True)
        output[track_id] = int(values[int(np.argmax(counts))])
    return output


def evaluate_assignment(
    result: dict,
    bank_gt_labels: dict[int, int],
    target_gt_labels: np.ndarray,
) -> dict:
    """Score identity matching without feeding labels back into assignment."""
    target_gt_labels = np.asarray(target_gt_labels, dtype=np.int64)
    bank_track_ids = np.asarray(result["bank"]["track_ids"], dtype=np.int64)
    target_to_pair = {
        int(row["target_index"]): row for row in result["accepted_pairs"]
    }
    source_to_pair = {
        int(row["source_index"]): row for row in result["accepted_pairs"]
    }
    available_labels = set(bank_gt_labels.values())
    target_labels = {int(value) for value in target_gt_labels if int(value) >= 0}
    true_positive = false_positive = false_negative = 0
    assignment_rows = []
    for target_index, target_label in enumerate(target_gt_labels):
        matchable = int(target_label) >= 0 and int(target_label) in available_labels
        pair = target_to_pair.get(target_index)
        predicted_label = None
        if pair is not None:
            track_id = int(bank_track_ids[int(pair["source_index"])])
            predicted_label = bank_gt_labels.get(track_id)
        correct = bool(matchable and predicted_label == int(target_label))
        if correct:
            true_positive += 1
        elif pair is not None:
            false_positive += 1
        if matchable and not correct:
            false_negative += 1
        assignment_rows.append(
            {
                "target_index": target_index,
                "target_gt_label": int(target_label),
                "matchable": matchable,
                "predicted_gt_label": predicted_label,
                "correct": correct,
            }
        )

    expected_post_dustbin = [
        index
        for index, label in enumerate(target_gt_labels)
        if int(label) < 0 or int(label) not in available_labels
    ]
    predicted_post_dustbin = set(int(value) for value in result["unmatched_target"])
    post_dustbin_tp = len(set(expected_post_dustbin) & predicted_post_dustbin)
    post_dustbin_fp = len(predicted_post_dustbin - set(expected_post_dustbin))
    post_dustbin_fn = len(set(expected_post_dustbin) - predicted_post_dustbin)

    expected_source_dustbin = []
    for source_index, track_id in enumerate(bank_track_ids):
        label = bank_gt_labels.get(int(track_id), -1)
        if label < 0 or label not in target_labels:
            expected_source_dustbin.append(source_index)
    predicted_source_dustbin = set(int(value) for value in result["unmatched_source"])
    source_dustbin_tp = len(set(expected_source_dustbin) & predicted_source_dustbin)
    source_dustbin_fp = len(predicted_source_dustbin - set(expected_source_dustbin))
    source_dustbin_fn = len(set(expected_source_dustbin) - predicted_source_dustbin)

    dustbin_tp = post_dustbin_tp + source_dustbin_tp
    dustbin_fp = post_dustbin_fp + source_dustbin_fp
    dustbin_fn = post_dustbin_fn + source_dustbin_fn
    accepted = len(result["accepted_pairs"])
    matchable = sum(row["matchable"] for row in assignment_rows)
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "accepted": accepted,
        "matchable": matchable,
        "assignment_accuracy": true_positive / max(accepted, 1),
        "recall_at_1": true_positive / max(matchable, 1),
        "idf1": 2 * true_positive / max(2 * true_positive + false_positive + false_negative, 1),
        "id_switches": false_positive,
        "dustbin_true_positive": dustbin_tp,
        "dustbin_false_positive": dustbin_fp,
        "dustbin_false_negative": dustbin_fn,
        "dustbin_precision": dustbin_tp / max(dustbin_tp + dustbin_fp, 1),
        "dustbin_recall": dustbin_tp / max(dustbin_tp + dustbin_fn, 1),
        "assignments": assignment_rows,
    }


def labeled_pair_distances(
    result: dict, bank_gt_labels: dict[int, int], target_gt_labels: np.ndarray
) -> tuple[list[float], list[float]]:
    same, different = [], []
    track_ids = np.asarray(result["bank"]["track_ids"], dtype=np.int64)
    for source_index, track_id in enumerate(track_ids):
        source_label = bank_gt_labels.get(int(track_id), -1)
        for target_index, target_label in enumerate(target_gt_labels):
            if source_label < 0 or int(target_label) < 0:
                continue
            bucket = same if source_label == int(target_label) else different
            bucket.append(float(result["cost"][source_index, target_index]))
    return same, different
