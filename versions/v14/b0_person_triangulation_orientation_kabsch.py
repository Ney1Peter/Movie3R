"""Person-local global-orientation refinement after frozen BRTC-LC v1.

For every BRTC-accepted match, the runtime aligns root-centred torso joints
from the current post-cut prediction to the last pre-cut prediction with a
bounded Kabsch SO(3) rotation.  It rotates only body-local geometry around the
already-corrected root.  Camera matrices and roots are never updated;
rejected and unmatched post people remain exact frozen-B0 outputs.

The callable consumes no image, GT, identity label, future post frame, or
additional pretrained model.
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
class OrientationKabschConfig:
    max_angle_deg: float = 25.0
    rotation_fraction: float = 0.5
    min_observable_relative_improvement: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.max_angle_deg) or not (0.0 < self.max_angle_deg <= 180.0):
            raise ValueError("max_angle_deg must be finite and in (0, 180]")
        if not np.isfinite(self.rotation_fraction) or not (
            0.0 < self.rotation_fraction <= 1.0
        ):
            raise ValueError("rotation_fraction must be finite and in (0, 1]")
        if not np.isfinite(self.min_observable_relative_improvement):
            raise ValueError("min_observable_relative_improvement must be finite")


DEFAULT_ORIENTATION_KABSCH_CONFIG = OrientationKabschConfig()


def _points(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim < 1 or result.shape[-1] != 3 or not np.isfinite(result).all():
        raise ValueError("Points must be finite with final dimension 3")
    return result


def kabsch_rotation(source: Any, target: Any) -> np.ndarray:
    """Return R such that row-vector ``source @ R.T`` fits ``target``."""

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
    rotation: Any,
    config: OrientationKabschConfig = DEFAULT_ORIENTATION_KABSCH_CONFIG,
) -> tuple[np.ndarray, float, float]:
    """Return bounded rotation and raw/applied angles in degrees."""

    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("rotation must be a finite 3x3 matrix")
    vector = Rotation.from_matrix(matrix).as_rotvec()
    raw_angle = float(np.linalg.norm(vector))
    if raw_angle <= 1e-12:
        return np.eye(3, dtype=np.float64), 0.0, 0.0
    applied = min(
        raw_angle * float(config.rotation_fraction),
        math.radians(float(config.max_angle_deg)),
    )
    bounded = Rotation.from_rotvec(vector * (applied / raw_angle)).as_matrix()
    return bounded, math.degrees(raw_angle), math.degrees(applied)


def orientation_candidate(
    pre_person: dict[str, Any],
    post_person: dict[str, Any],
    corrected_person: dict[str, Any],
    config: OrientationKabschConfig = DEFAULT_ORIENTATION_KABSCH_CONFIG,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Estimate and apply one bounded person-local rotation."""

    pre_joints = _points(pre_person["joints"]).reshape(-1, 3)
    post_joints = _points(post_person["joints"]).reshape(-1, 3)
    ids = [index for index in TORSO4 if index < min(len(pre_joints), len(post_joints))]
    if len(ids) < 3:
        return corrected_person, {
            "applied": False,
            "reason": "insufficient_torso_joints",
        }
    pre_root = _points(pre_person["root"]).reshape(3)
    post_root = _points(post_person["root"]).reshape(3)
    pre = pre_joints[ids] - pre_root
    post = post_joints[ids] - post_root
    before = float(np.linalg.norm(post - pre, axis=1).mean())
    raw_rotation = kabsch_rotation(post, pre)
    candidate_rotation, raw_angle_deg, applied_angle_deg = bounded_rotation(
        raw_rotation, config
    )
    after = float(
        np.linalg.norm(post @ candidate_rotation.T - pre, axis=1).mean()
    )
    relative = float((before - after) / max(before, 1e-12))
    apply = bool(
        applied_angle_deg > 1e-8
        and after < before
        and relative >= float(config.min_observable_relative_improvement)
    )
    debug = {
        "applied": apply,
        "reason": "applied" if apply else "observable_gate",
        "raw_torso_residual_m": before,
        "candidate_torso_residual_m": after,
        "observable_relative_improvement": relative,
        "raw_angle_deg": raw_angle_deg,
        "applied_angle_deg": applied_angle_deg,
        "rotation_world": candidate_rotation,
    }
    if not apply:
        return corrected_person, debug

    output = {
        key: (value.copy() if isinstance(value, np.ndarray) else value)
        for key, value in corrected_person.items()
    }
    root = _points(corrected_person["root"]).reshape(3)
    output["root"] = root.copy()
    for key in ("joints", "vertices"):
        if key in corrected_person:
            points = _points(corrected_person[key])
            output[key] = (points - root) @ candidate_rotation.T + root
    for key in ("torso", "root_rotation"):
        if key in corrected_person:
            output[key] = candidate_rotation @ np.asarray(
                corrected_person[key], dtype=np.float64
            )
    return output, debug


def refine_matched_people_orientation_kabsch(
    pre_camera: np.ndarray,
    post_camera: np.ndarray,
    pre_people: list[dict[str, Any]],
    post_people: list[dict[str, Any]],
    matches: Iterable[tuple[int, int]],
    config: PersonTriangulationConfig = DEFAULT_CONFIG,
    orientation_config: OrientationKabschConfig = DEFAULT_ORIENTATION_KABSCH_CONFIG,
    orientation_pre_people: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run frozen BRTC, then rotate only its accepted matched people.

    ``pre_people`` is the frozen translation-state input consumed by BRTC.
    A causal multi-cut deployment may additionally supply the already-rotated
    last-pre state through ``orientation_pre_people``.  Keeping these states
    separate prevents a previous orientation update from changing the next
    cut's frozen BRTC roots while still letting Kabsch consume causal pose.
    """

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
    orientation_pre = (
        pre_people if orientation_pre_people is None else orientation_pre_people
    )
    if len(orientation_pre) != len(pre_people):
        raise ValueError(
            "orientation_pre_people must use the same indexing as pre_people"
        )
    for pre_index in {pre_index for pre_index, _ in materialized_matches}:
        for identity_key in ("global_track_id", "native_track_id"):
            translation_identity = pre_people[pre_index].get(identity_key)
            orientation_identity = orientation_pre[pre_index].get(identity_key)
            if (
                translation_identity is not None
                and orientation_identity is not None
                and int(translation_identity) != int(orientation_identity)
            ):
                raise ValueError(
                    "orientation_pre_people must use the same indexing as "
                    f"pre_people ({identity_key} differs at index {pre_index})"
                )
    records_by_post = {
        int(record["post_index"]): record for record in base_debug["people"]
    }
    orientation_by_post: dict[int, dict[str, Any]] = {}
    for pre_index, post_index in materialized_matches:
        record = records_by_post[post_index]
        if not bool(record["accepted"]):
            orientation_by_post[post_index] = {
                "applied": False,
                "reason": "brtc_rejected_exact_b0_fallback",
            }
            continue
        corrected[post_index], orientation_by_post[post_index] = orientation_candidate(
            orientation_pre[pre_index],
            post_people[post_index],
            corrected[post_index],
            orientation_config,
        )

    people_debug = []
    for base_record in base_debug["people"]:
        record = dict(base_record)
        record["orientation"] = orientation_by_post[int(record["post_index"])]
        people_debug.append(record)
    debug = dict(base_debug)
    debug.update(
        {
            "camera_update": "none",
            "root_update": "frozen BRTC only",
            "orientation_update": "person-local bounded torso4 Kabsch",
            "orientation_pre_state": (
                "shared_with_brtc_translation_state"
                if orientation_pre_people is None
                else "separate_causal_orientation_state"
            ),
            "orientation_applied_count": int(
                sum(bool(value["applied"]) for value in orientation_by_post.values())
            ),
            "unmatched_policy": "exact frozen B0",
            "rejected_policy": "exact frozen B0",
            "people": people_debug,
        }
    )
    return corrected, debug
