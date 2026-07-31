"""Accepted-person shared SO(3) orientation refinement after frozen BRTC.

All BRTC-accepted people at one boundary contribute their root-centered torso4
correspondences to one Kabsch estimate.  A frozen bounded fraction of that
single rotation is applied around each accepted person's already-corrected
root.  Roots, cameras, rejected people, unmatched people, and pair layout are
unchanged by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np
from scipy.spatial.transform import Rotation

from versions.v14.b0_person_triangulation import (
    DEFAULT_CONFIG,
    PersonTriangulationConfig,
    refine_matched_people,
)


TORSO4 = (1, 2, 16, 17)


@dataclass(frozen=True)
class SharedOrientationKabschConfig:
    max_angle_deg: float = 25.0
    rotation_fraction: float = 0.5
    min_observable_relative_improvement: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.max_angle_deg) or not (0.0 < self.max_angle_deg <= 180.0):
            raise ValueError("max_angle_deg must be finite and in (0, 180]")
        if not np.isfinite(self.rotation_fraction) or not (0.0 < self.rotation_fraction <= 1.0):
            raise ValueError("rotation_fraction must be finite and in (0, 1]")
        if not np.isfinite(self.min_observable_relative_improvement):
            raise ValueError("min_observable_relative_improvement must be finite")


DEFAULT_SHARED_ORIENTATION_KABSCH_CONFIG = SharedOrientationKabschConfig()


def _points(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim < 1 or result.shape[-1] != 3 or not np.isfinite(result).all():
        raise ValueError("Points must be finite with final dimension 3")
    return result


def kabsch_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = _points(source).reshape(-1, 3)
    target = _points(target).reshape(-1, 3)
    if source.shape != target.shape or len(source) < 3:
        raise ValueError("Kabsch requires at least three corresponding points")
    covariance = source.T @ target
    left, _, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_t[-1] *= -1.0
        rotation = right_t.T @ left.T
    return rotation


def bounded_rotation(
    rotation: np.ndarray,
    config: SharedOrientationKabschConfig,
) -> tuple[np.ndarray, float, float]:
    vector = Rotation.from_matrix(np.asarray(rotation, dtype=np.float64)).as_rotvec()
    raw_angle = float(np.linalg.norm(vector))
    if raw_angle <= 1e-12:
        return np.eye(3, dtype=np.float64), 0.0, 0.0
    applied = min(
        raw_angle * float(config.rotation_fraction),
        math.radians(float(config.max_angle_deg)),
    )
    value = Rotation.from_rotvec(vector * (applied / raw_angle)).as_matrix()
    return value, math.degrees(raw_angle), math.degrees(applied)


def refine_matched_people_shared_orientation_kabsch(
    pre_camera: np.ndarray,
    post_camera: np.ndarray,
    pre_people: list[dict[str, Any]],
    post_people: list[dict[str, Any]],
    matches: Iterable[tuple[int, int]],
    config: PersonTriangulationConfig = DEFAULT_CONFIG,
    orientation_config: SharedOrientationKabschConfig = DEFAULT_SHARED_ORIENTATION_KABSCH_CONFIG,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply one accepted-set Kabsch rotation around each corrected root."""

    materialized_matches = tuple(
        (int(pre_index), int(post_index)) for pre_index, post_index in matches
    )
    corrected, base_debug = refine_matched_people(
        pre_camera,
        post_camera,
        pre_people,
        post_people,
        materialized_matches,
        config,
    )
    records_by_post = {
        int(record["post_index"]): record for record in base_debug["people"]
    }
    accepted_pairs = [
        (pre_index, post_index)
        for pre_index, post_index in materialized_matches
        if bool(records_by_post[post_index]["accepted"])
    ]
    pre_centered, post_centered = [], []
    accepted_post_indices = []
    insufficient = False
    for pre_index, post_index in accepted_pairs:
        pre_joints = _points(pre_people[pre_index]["joints"]).reshape(-1, 3)
        post_joints = _points(post_people[post_index]["joints"]).reshape(-1, 3)
        ids = [index for index in TORSO4 if index < min(len(pre_joints), len(post_joints))]
        if len(ids) < 3:
            insufficient = True
            break
        pre_root = _points(pre_people[pre_index]["root"]).reshape(3)
        post_root = _points(post_people[post_index]["root"]).reshape(3)
        pre_centered.append(pre_joints[ids] - pre_root)
        post_centered.append(post_joints[ids] - post_root)
        accepted_post_indices.append(post_index)

    if accepted_pairs and not insufficient:
        pre_stack = np.concatenate(pre_centered, axis=0)
        post_stack = np.concatenate(post_centered, axis=0)
        before = float(np.linalg.norm(post_stack - pre_stack, axis=1).mean())
        raw_rotation = kabsch_rotation(post_stack, pre_stack)
        candidate_rotation, raw_angle_deg, applied_angle_deg = bounded_rotation(
            raw_rotation, orientation_config
        )
        after = float(
            np.linalg.norm(post_stack @ candidate_rotation.T - pre_stack, axis=1).mean()
        )
        relative = float((before - after) / max(before, 1e-12))
        apply = bool(
            applied_angle_deg > 1e-8
            and after < before
            and relative
            >= float(orientation_config.min_observable_relative_improvement)
        )
        reason = "applied" if apply else "observable_gate"
    else:
        candidate_rotation = np.eye(3, dtype=np.float64)
        raw_angle_deg = applied_angle_deg = 0.0
        before = after = relative = 0.0
        apply = False
        reason = "insufficient_torso_joints" if insufficient else "no_accepted_people"

    if apply:
        for post_index in accepted_post_indices:
            root = _points(corrected[post_index]["root"]).reshape(3)
            for key in ("joints", "vertices"):
                if key in corrected[post_index]:
                    points = _points(corrected[post_index][key])
                    corrected[post_index][key] = (
                        points - root
                    ) @ candidate_rotation.T + root

    people_debug = []
    for base_record in base_debug["people"]:
        record = dict(base_record)
        accepted = bool(base_record["accepted"])
        record.update(
            {
                "shared_orientation_applied": bool(apply and accepted),
                "shared_rotation_world": candidate_rotation if apply and accepted else np.eye(3),
                "shared_applied_angle_deg": applied_angle_deg if apply and accepted else 0.0,
            }
        )
        people_debug.append(record)

    debug = dict(base_debug)
    debug.update(
        {
            "camera_update": "none",
            "root_update": "frozen BRTC only",
            "orientation_update": "shared accepted-set Kabsch around each corrected root",
            "accepted_orientation_person_count": len(accepted_pairs),
            "shared_orientation_applied": apply,
            "shared_orientation_reason": reason,
            "shared_raw_torso_residual_m": before,
            "shared_candidate_torso_residual_m": after,
            "shared_observable_relative_improvement": relative,
            "shared_raw_angle_deg": raw_angle_deg,
            "shared_applied_angle_deg": applied_angle_deg,
            "shared_rotation_world": candidate_rotation,
            "people": people_debug,
        }
    )
    return corrected, debug

