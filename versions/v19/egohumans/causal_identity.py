"""Prediction-only causal identity transport for EgoHumans experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


TORSO = np.asarray([0, 1, 2, 3, 6, 9, 12, 13, 14, 16, 17], dtype=np.int64)


@dataclass(frozen=True)
class IdentityConfig:
    max_cost_m: float = 1.0
    max_gap_frames: int = 5
    body_weight: float = 0.25
    velocity_alpha: float = 0.5


def pelvis(joints: np.ndarray) -> np.ndarray:
    return np.asarray(joints)[..., [1, 2], :].mean(axis=-2)


def clone(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.asarray(value).copy() for key, value in arrays.items()}


def retrack_causally(
    arrays: dict[str, np.ndarray],
    boundary: int,
    boundary_pairs: list[tuple[int, int]],
    config: IdentityConfig,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Transport IDs online using emitted geometry and one B0 boundary match.

    Geometry is never changed.  Inside each shot, a constant-velocity world-root
    prediction and root-centred torso descriptor determine a gated Hungarian
    match.  At the cut, the already available B0 association transports IDs
    from the last pre frame to the first post frame; unmatched detections start
    new tracks.  No RGB future frame, GT identity, calibration, or action label
    is read.
    """

    output = clone(arrays)
    valid = np.asarray(output["valid"]).astype(bool)
    frame_count, slot_count = valid.shape
    output["persistent_ids"][:] = -1
    next_id = 0
    tracks: dict[int, dict[str, Any]] = {}
    assignments = []
    unmatched_detections = 0
    expired_tracks = 0

    def observation(frame: int, slot: int) -> tuple[np.ndarray, np.ndarray]:
        joints = np.asarray(output["joints_world"][frame, slot], dtype=np.float64)
        root = pelvis(joints)
        body = joints[TORSO] - root
        return root, body

    def create(frame: int, slot: int) -> int:
        nonlocal next_id
        identity = next_id
        next_id += 1
        root, body = observation(frame, slot)
        tracks[identity] = {
            "last_frame": frame,
            "root": root,
            "body": body,
            "velocity": np.zeros(3, dtype=np.float64),
        }
        output["persistent_ids"][frame, slot] = identity
        return identity

    def update(identity: int, frame: int, slot: int) -> None:
        root, body = observation(frame, slot)
        track = tracks[identity]
        gap = max(frame - int(track["last_frame"]), 1)
        measured = (root - np.asarray(track["root"])) / gap
        track["velocity"] = (
            (1.0 - float(config.velocity_alpha)) * np.asarray(track["velocity"])
            + float(config.velocity_alpha) * measured
        )
        track.update(last_frame=frame, root=root, body=body)
        output["persistent_ids"][frame, slot] = identity

    for frame in range(frame_count):
        detections = [int(value) for value in np.flatnonzero(valid[frame])]
        if not detections:
            assignments.append({"frame": frame, "pairs": [], "new": []})
            continue
        if frame == 0:
            new = [(slot, create(frame, slot)) for slot in detections]
            assignments.append({"frame": frame, "policy": "initialize", "pairs": [], "new": new})
            continue
        if frame == int(boundary):
            used_slots = set()
            used_ids = set()
            pairs_out = []
            for pre_slot, post_slot in boundary_pairs:
                pre_slot, post_slot = int(pre_slot), int(post_slot)
                if post_slot not in detections or pre_slot < 0 or pre_slot >= slot_count:
                    continue
                identity = int(output["persistent_ids"][frame - 1, pre_slot])
                if identity < 0 or identity in used_ids or post_slot in used_slots:
                    continue
                update(identity, frame, post_slot)
                used_slots.add(post_slot)
                used_ids.add(identity)
                pairs_out.append((identity, post_slot))
            new = [(slot, create(frame, slot)) for slot in detections if slot not in used_slots]
            unmatched_detections += len(new)
            assignments.append(
                {"frame": frame, "policy": "B0_boundary_transport", "pairs": pairs_out, "new": new}
            )
            continue

        active = [
            identity
            for identity, track in sorted(tracks.items())
            if 0 < frame - int(track["last_frame"]) <= int(config.max_gap_frames)
        ]
        matched_slots = set()
        pairs_out = []
        if active:
            cost = np.full((len(active), len(detections)), np.inf, dtype=np.float64)
            for row, identity in enumerate(active):
                track = tracks[identity]
                gap = frame - int(track["last_frame"])
                target_root = np.asarray(track["root"]) + gap * np.asarray(track["velocity"])
                target_body = np.asarray(track["body"])
                for column, slot in enumerate(detections):
                    root, body = observation(frame, slot)
                    root_cost = float(np.linalg.norm(root - target_root))
                    body_cost = float(np.linalg.norm(body - target_body, axis=1).mean())
                    cost[row, column] = root_cost + float(config.body_weight) * body_cost
            rows, columns = linear_sum_assignment(cost)
            for row, column in zip(rows, columns):
                if float(cost[row, column]) > float(config.max_cost_m):
                    continue
                identity, slot = active[int(row)], detections[int(column)]
                update(identity, frame, slot)
                matched_slots.add(slot)
                pairs_out.append((identity, slot, float(cost[row, column])))
        new = [(slot, create(frame, slot)) for slot in detections if slot not in matched_slots]
        unmatched_detections += len(new)
        expired_tracks += sum(
            frame - int(track["last_frame"]) == int(config.max_gap_frames) + 1
            for track in tracks.values()
        )
        assignments.append({"frame": frame, "policy": "causal_hungarian", "pairs": pairs_out, "new": new})

    before = np.asarray(arrays["persistent_ids"])
    changed = int(np.count_nonzero(valid & (before != output["persistent_ids"])))
    return output, {
        "policy": "causal_geometry_identity_transport_v1",
        "config": config.__dict__,
        "track_count": next_id,
        "changed_valid_assignments": changed,
        "unmatched_detection_track_starts": unmatched_detections,
        "expired_track_events": expired_tracks,
        "assignments": assignments,
        "geometry_changed": False,
        "runtime_contract": {
            "gt_used": False,
            "calibration_used": False,
            "future_frames_used": 0,
            "boundary_information": "B0 prediction-only association",
        },
    }
