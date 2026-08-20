"""Causal decoupled camera/person correction used by Movie3R-v19."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PersonCorrectionConfig:
    blend: float = 1.0
    use_velocity_target: bool = True
    velocity_horizon: int = 4
    max_shift_m: float = 5.0


def pelvis(joints: np.ndarray) -> np.ndarray:
    return np.asarray(joints)[..., [1, 2], :].mean(axis=-2)


def person_boundary_correct(
    arrays: dict[str, np.ndarray],
    boundary: int,
    boundary_pairs: list[tuple[int, int]],
    config: PersonCorrectionConfig,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Translate each matched post-shot trajectory toward its causal target.

    The input must already carry causal persistent IDs.  A separate constant
    translation is estimated for each B0-matched person at the cut and applied
    only to that person's future trajectory.  Cameras, body pose, pre-shot
    outputs, unmatched people, and already emitted frames are untouched.
    """

    if not 0.0 <= float(config.blend) <= 1.0:
        raise ValueError(f"blend must be in [0,1], got {config.blend}")
    output = {key: np.asarray(value).copy() for key, value in arrays.items()}
    if boundary <= 0 or boundary >= len(output["valid"]):
        return output, {"accepted": False, "reason": "boundary_out_of_range"}
    rows = []
    used_ids = set()
    for pre_slot, post_slot in boundary_pairs:
        pre_slot, post_slot = int(pre_slot), int(post_slot)
        if (
            pre_slot < 0
            or post_slot < 0
            or pre_slot >= output["valid"].shape[1]
            or post_slot >= output["valid"].shape[1]
            or not output["valid"][boundary - 1, pre_slot]
            or not output["valid"][boundary, post_slot]
        ):
            continue
        pre_identity = int(output["persistent_ids"][boundary - 1, pre_slot])
        post_identity = int(output["persistent_ids"][boundary, post_slot])
        if pre_identity < 0 or post_identity < 0 or post_identity in used_ids:
            continue
        target = pelvis(output["joints_world"][boundary - 1, pre_slot]).astype(np.float64)
        velocity = np.zeros(3, dtype=np.float64)
        if config.use_velocity_target:
            history = []
            for frame in range(max(0, boundary - int(config.velocity_horizon) - 1), boundary):
                slots = np.flatnonzero(
                    output["valid"][frame].astype(bool)
                    & (output["persistent_ids"][frame] == pre_identity)
                )
                if len(slots):
                    history.append(pelvis(output["joints_world"][frame, int(slots[0])]))
            if len(history) >= 2:
                velocity = np.median(np.diff(np.asarray(history), axis=0), axis=0)
                speed = float(np.linalg.norm(velocity))
                if speed > 0.15:
                    velocity *= 0.15 / speed
                target = target + velocity
        source = pelvis(output["joints_world"][boundary, post_slot]).astype(np.float64)
        full_shift = target - source
        norm = float(np.linalg.norm(full_shift))
        accepted = norm <= float(config.max_shift_m)
        shift = float(config.blend) * full_shift if accepted else np.zeros(3, dtype=np.float64)
        affected = 0
        if accepted:
            for frame in range(boundary, len(output["valid"])):
                slots = np.flatnonzero(
                    output["valid"][frame].astype(bool)
                    & (output["persistent_ids"][frame] == post_identity)
                )
                for slot in slots:
                    output["joints_world"][frame, int(slot)] += shift
                    output["vertices_world"][frame, int(slot)] += shift
                    affected += 1
        used_ids.add(post_identity)
        rows.append(
            {
                "pre_slot": pre_slot,
                "post_slot": post_slot,
                "pre_identity": pre_identity,
                "post_identity": post_identity,
                "full_shift_m": norm,
                "applied_shift": shift.tolist(),
                "applied_shift_m": float(np.linalg.norm(shift)),
                "velocity_target": velocity.tolist(),
                "accepted": accepted,
                "affected_post_frames": affected,
            }
        )
    return output, {
        "policy": "B0_associated_causal_person_trajectory_translation_v1",
        "config": config.__dict__,
        "people": rows,
        "accepted_count": sum(bool(row["accepted"]) for row in rows),
        "matched_count": len(rows),
        "camera_changed": False,
        "body_pose_changed": False,
        "pre_frames_changed": False,
        "unmatched_people_changed": False,
        "runtime_contract": {
            "gt_used": False,
            "calibration_used": False,
            "future_frames_used": 0,
            "association": "B0 prediction-only boundary match",
        },
    }
