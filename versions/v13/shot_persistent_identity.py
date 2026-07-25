"""Fixed-size causal identity state and bounded cross-shot hypotheses.

This module answers WHO only.  It stores no camera transform and never predicts
rotation, translation, scale, fusion weights, or a Boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, permutations

import numpy as np
from scipy.optimize import linear_sum_assignment

from versions.v13.identity_bridge import frame_feature


def _unit(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(value))
    return value / max(norm, 1e-12)


def _cosine_cost(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    if not len(first) or not len(second):
        return np.empty((len(first), len(second)), dtype=np.float64)
    first = np.stack([_unit(row) for row in first])
    second = np.stack([_unit(row) for row in second])
    return np.clip(1.0 - first @ second.T, 0.0, 2.0)


@dataclass
class RunningFeature:
    """Constant-memory running statistics plus a small medoid buffer."""

    history_size: int = 5
    count: int = 0
    mean: np.ndarray | None = None
    m2: np.ndarray | None = None
    last: np.ndarray | None = None
    history: list[np.ndarray] = field(default_factory=list)

    def update(self, value: np.ndarray) -> None:
        value = _unit(value)
        if not np.isfinite(value).all() or float(np.linalg.norm(value)) < 1e-10:
            return
        self.count += 1
        if self.mean is None:
            self.mean = value.copy()
            self.m2 = np.zeros_like(value)
        else:
            delta = value - self.mean
            self.mean = self.mean + delta / self.count
            self.m2 = self.m2 + delta * (value - self.mean)
        self.last = value.copy()
        self.history.append(value.copy())
        self.history = self.history[-int(self.history_size) :]

    def prototype(self, mode: str) -> np.ndarray:
        if self.last is None:
            raise ValueError("Feature state has no valid observation")
        if mode == "last":
            return self.last.copy()
        if mode == "mean":
            return _unit(self.mean)
        if mode == "medoid":
            values = np.stack(self.history)
            cost = _cosine_cost(values, values)
            return values[int(np.argmin(cost.sum(axis=1)))].copy()
        raise ValueError(f"Unsupported state prototype: {mode}")

    def dispersion(self, floor: float) -> float:
        if self.count < 2 or self.m2 is None:
            return float(floor)
        rms = float(np.sqrt(np.maximum(self.m2.sum() / (self.count - 1), 0.0)))
        return max(rms, float(floor))


@dataclass
class PersistentTrack:
    track_id: int
    features: dict[str, RunningFeature] = field(default_factory=dict)
    observation_count: int = 0
    last_seen: int = 0
    active: bool = True
    valid_appearance_count: int = 0
    quality_sum: float = 0.0


class ShotPersistentIdentityState:
    """External identity state with read-only query and explicit commit."""

    def __init__(
        self,
        ttl: int = 30,
        history_size: int = 5,
        max_tracks: int = 8,
    ) -> None:
        if ttl < 1 or history_size < 1 or max_tracks < 1:
            raise ValueError("ttl, history_size and max_tracks must be positive")
        self.ttl = int(ttl)
        self.history_size = int(history_size)
        self.max_tracks = int(max_tracks)
        self.tracks: dict[int, PersistentTrack] = {}
        self.next_track_id = 0
        self.commit_count = 0

    def allocate(self) -> int:
        track_id = int(self.next_track_id)
        self.next_track_id += 1
        self.tracks[track_id] = PersistentTrack(track_id=track_id)
        return track_id

    def _update_one(
        self,
        frame: dict,
        detection_index: int,
        track_id: int,
        timestamp: int,
    ) -> None:
        track = self.tracks.setdefault(int(track_id), PersistentTrack(int(track_id)))
        self.next_track_id = max(self.next_track_id, int(track_id) + 1)
        appearance_valid = bool(
            np.asarray(
                frame.get("appearance_valid", np.ones(int(frame["count"]))),
                dtype=bool,
            )[int(detection_index)]
        )
        for name in sorted(frame.get("features", {})):
            if name.startswith("appearance") and not appearance_valid:
                continue
            values = frame_feature(frame, name)
            if int(detection_index) >= len(values):
                continue
            feature = track.features.setdefault(
                name, RunningFeature(history_size=self.history_size)
            )
            feature.update(values[int(detection_index)])
        track.observation_count += 1
        track.valid_appearance_count += int(appearance_valid)
        scores = np.asarray(
            frame.get("head_scores", np.ones(int(frame["count"]))), dtype=np.float64
        ).reshape(-1)
        if int(detection_index) < len(scores) and np.isfinite(scores[detection_index]):
            track.quality_sum += float(scores[detection_index])
        track.last_seen = int(timestamp)
        track.active = True

    def bootstrap(self, frame: dict, timestamp: int) -> np.ndarray:
        output = np.empty(int(frame["count"]), dtype=np.int64)
        for detection_index in range(len(output)):
            output[detection_index] = self.allocate()
            self._update_one(
                frame, detection_index, int(output[detection_index]), timestamp
            )
        self.commit_count += 1
        return output

    def observe(
        self, frame: dict, track_ids: np.ndarray, timestamp: int, valid: np.ndarray | None = None
    ) -> None:
        track_ids = np.asarray(track_ids, dtype=np.int64).reshape(-1)
        if len(track_ids) != int(frame["count"]):
            raise ValueError("track_ids and detections have different lengths")
        keep = (
            np.ones(len(track_ids), dtype=bool)
            if valid is None
            else np.asarray(valid, dtype=bool).reshape(-1)
        )
        if len(keep) != len(track_ids):
            raise ValueError("valid mask and detections have different lengths")
        for track in self.tracks.values():
            track.active = False
        for detection_index, track_id in enumerate(track_ids):
            if int(track_id) >= 0 and bool(keep[detection_index]):
                self._update_one(frame, detection_index, int(track_id), timestamp)
        self.commit_count += 1
        self.expire(timestamp)

    def commit(
        self, frame: dict, result: dict, timestamp: int, allocate_unmatched: bool = True
    ) -> np.ndarray:
        """Commit an already selected hypothesis; matching itself is read-only."""
        output = np.full(int(frame["count"]), -1, dtype=np.int64)
        bank_ids = np.asarray(result["bank"]["track_ids"], dtype=np.int64)
        for pair in result["accepted_pairs"]:
            source = int(pair["source_index"])
            target = int(pair["target_index"])
            if source < len(bank_ids) and target < len(output):
                output[target] = int(bank_ids[source])
        if allocate_unmatched:
            for target in range(len(output)):
                if output[target] < 0:
                    output[target] = self.allocate()
        self.observe(frame, output, timestamp)
        return output

    def expire(self, timestamp: int) -> None:
        expired = [
            track_id
            for track_id, track in self.tracks.items()
            if int(timestamp) - int(track.last_seen) > self.ttl
        ]
        for track_id in expired:
            del self.tracks[track_id]

    def candidate_tracks(self, timestamp: int, feature: str) -> list[PersistentTrack]:
        tracks = [
            track
            for track in self.tracks.values()
            if int(timestamp) - int(track.last_seen) <= self.ttl
            and feature in track.features
            and track.features[feature].count > 0
        ]
        tracks.sort(key=lambda row: (-int(row.last_seen), int(row.track_id)))
        tracks = tracks[: self.max_tracks]
        return sorted(tracks, key=lambda row: int(row.track_id))

    def cost_matrix(
        self,
        post_frame: dict,
        feature: str,
        prototype: str,
        timestamp: int,
        track_normalized: bool = False,
        dispersion_floor: float = 0.05,
    ) -> tuple[dict, np.ndarray]:
        tracks = self.candidate_tracks(timestamp, feature)
        target = frame_feature(post_frame, feature)
        if tracks:
            source = np.stack(
                [track.features[feature].prototype(prototype) for track in tracks]
            )
            cost = _cosine_cost(source, target)
            dispersions = np.asarray(
                [
                    track.features[feature].dispersion(dispersion_floor)
                    for track in tracks
                ],
                dtype=np.float64,
            )
            if track_normalized:
                cost = cost / dispersions[:, None]
        else:
            source = np.empty((0, target.shape[1] if target.ndim == 2 else 0))
            cost = np.empty((0, len(target)), dtype=np.float64)
            dispersions = np.empty(0, dtype=np.float64)
        bank = {
            "track_ids": np.asarray([track.track_id for track in tracks], dtype=np.int64),
            "prototypes": source,
            "history_count": np.asarray(
                [track.features[feature].count for track in tracks], dtype=np.int64
            ),
            "dispersion": dispersions,
        }
        return {"bank": bank, "target": target}, cost

    def hungarian(
        self,
        post_frame: dict,
        feature: str,
        prototype: str,
        timestamp: int,
        track_normalized: bool = False,
        dispersion_floor: float = 0.05,
        max_cost: float = float("inf"),
    ) -> dict:
        payload, cost = self.cost_matrix(
            post_frame,
            feature,
            prototype,
            timestamp,
            track_normalized,
            dispersion_floor,
        )
        pairs = []
        if cost.size:
            sources, targets = linear_sum_assignment(cost)
            for source, target in zip(sources, targets):
                value = float(cost[source, target])
                pairs.append(
                    {
                        "source_index": int(source),
                        "target_index": int(target),
                        "cost": value,
                        "score": -value,
                        "accepted": bool(value <= float(max_cost)),
                    }
                )
        return assignment_result(payload, cost, pairs)

    def snapshot(self) -> dict:
        return {
            "next_track_id": int(self.next_track_id),
            "commit_count": int(self.commit_count),
            "tracks": {
                int(track_id): {
                    "observation_count": int(track.observation_count),
                    "valid_appearance_count": int(track.valid_appearance_count),
                    "last_seen": int(track.last_seen),
                    "active": bool(track.active),
                    "feature_counts": {
                        name: int(value.count)
                        for name, value in sorted(track.features.items())
                    },
                }
                for track_id, track in sorted(self.tracks.items())
            },
        }


def assignment_result(payload: dict, cost: np.ndarray, pairs: list[dict]) -> dict:
    accepted = [dict(row) for row in pairs if bool(row.get("accepted", True))]
    source_count, target_count = cost.shape
    accepted_source = {int(row["source_index"]) for row in accepted}
    accepted_target = {int(row["target_index"]) for row in accepted}
    return {
        **payload,
        "cost": np.asarray(cost, dtype=np.float64),
        "pairs": [dict(row) for row in pairs],
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


def enumerate_topk_hypotheses(
    payload: dict,
    cost: np.ndarray,
    top_k: int = 6,
    unmatched_penalty: float | None = None,
) -> list[dict]:
    """Enumerate bounded partial one-to-one assignments for 2-3 person cuts."""
    if top_k < 1 or top_k > 6:
        raise ValueError("Phase 5 requires 1 <= top_k <= 6")
    cost = np.asarray(cost, dtype=np.float64)
    source_count, target_count = cost.shape
    finite = cost[np.isfinite(cost)]
    if unmatched_penalty is None:
        unmatched_penalty = (
            max(float(np.median(finite) + 0.5), 1.0) if finite.size else 1.0
        )
    hypotheses = []
    maximum = min(source_count, target_count)
    for match_count in range(maximum + 1):
        for sources in combinations(range(source_count), match_count):
            for targets in combinations(range(target_count), match_count):
                for target_order in permutations(targets):
                    pairs = [
                        {
                            "source_index": int(source),
                            "target_index": int(target),
                            "cost": float(cost[source, target]),
                            "score": float(-cost[source, target]),
                            "accepted": True,
                        }
                        for source, target in zip(sources, target_order)
                    ]
                    matched_cost = sum(row["cost"] for row in pairs)
                    unmatched = source_count + target_count - 2 * match_count
                    total = matched_cost + float(unmatched_penalty) * unmatched
                    signature = tuple(
                        sorted(
                            (
                                int(payload["bank"]["track_ids"][row["source_index"]]),
                                int(row["target_index"]),
                            )
                            for row in pairs
                        )
                    )
                    hypotheses.append(
                        {
                            "result": assignment_result(payload, cost, pairs),
                            "identity_cost": float(total),
                            "matched_cost": float(matched_cost),
                            "unmatched_penalty": float(unmatched_penalty),
                            "matched_count": int(match_count),
                            "signature": signature,
                        }
                    )
    hypotheses.sort(
        key=lambda row: (
            float(row["identity_cost"]),
            -int(row["matched_count"]),
            row["signature"],
        )
    )
    return hypotheses[: int(top_k)]
