"""Causal common-gauge and root-trajectory stabilization.

Every scene transform in this module is estimated from the current or past
prediction only.  A common transform is applied to the camera and all people
in a frame, so their camera-coordinate geometry is invariant by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.spatial.transform import Rotation


ArrayDict = dict[str, np.ndarray]
RegistrationKind = Literal["none", "translation", "se3"]
TORSO_JOINTS = np.asarray([0, 1, 2, 3, 6, 9, 12, 13, 14, 16, 17], dtype=np.int64)


@dataclass(frozen=True)
class Candidate:
    """A finite-grid v16 candidate; values are frozen before validation."""

    name: str
    camera_alpha: float = 1.0
    boundary_kind: RegistrationKind = "none"
    boundary_blend: float = 0.0
    use_velocity_target: bool = False
    root_alpha: float | None = None
    root_beta: float = 0.0
    gate_max_boundary_residual_m: float | None = None
    gate_min_matches: int = 0


def clone_arrays(arrays: ArrayDict) -> ArrayDict:
    return {key: np.asarray(value).copy() for key, value in arrays.items()}


def camera_coordinates(camera_c2w: np.ndarray, points_world: np.ndarray) -> np.ndarray:
    rotation = np.asarray(camera_c2w)[:3, :3]
    centre = np.asarray(camera_c2w)[:3, 3]
    return (np.asarray(points_world) - centre) @ rotation


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.asarray(points) @ np.asarray(transform)[:3, :3].T + np.asarray(transform)[:3, 3]


def _transform_frame_inplace(arrays: ArrayDict, frame: int, transform: np.ndarray) -> None:
    arrays["cameras_c2w"][frame] = transform @ arrays["cameras_c2w"][frame]
    for key in ("joints_world", "vertices_world"):
        arrays[key][frame] = transform_points(transform, arrays[key][frame])


def _rotation_step(previous: np.ndarray, observation: np.ndarray, alpha: float) -> np.ndarray:
    relative = previous.T @ observation
    rotvec = Rotation.from_matrix(relative).as_rotvec()
    return previous @ Rotation.from_rotvec(float(alpha) * rotvec).as_matrix()


def rotation_angle_deg(rotation: np.ndarray) -> float:
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(rotation).as_rotvec())))


def causal_shot_gauge_stabilize(
    arrays: ArrayDict,
    boundary: int,
    alpha: float,
) -> tuple[ArrayDict, dict[str, Any]]:
    """Causally filter each shot's camera gauge and move all people with it.

    ``alpha=0`` freezes a shot to its first predicted camera. ``alpha=1`` is
    an exact no-op.  Intermediate values are a causal SE(3) exponential moving
    average.  No already-emitted frame is changed when a future frame arrives.
    """

    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError(f"camera alpha must be in [0, 1], got {alpha}")
    output = clone_arrays(arrays)
    frame_count = len(arrays["cameras_c2w"])
    corrections = []
    shot_rows = []
    for start, end in ((0, int(boundary)), (int(boundary), frame_count)):
        if start >= end:
            continue
        first = np.asarray(arrays["cameras_c2w"][start], dtype=np.float64)
        filtered_rotation = first[:3, :3].copy()
        filtered_translation = first[:3, 3].copy()
        raw_centres = []
        raw_rotations = []
        for frame in range(start, end):
            observed = np.asarray(arrays["cameras_c2w"][frame], dtype=np.float64)
            raw_centres.append(observed[:3, 3])
            raw_rotations.append(observed[:3, :3])
            if frame > start:
                filtered_translation = (
                    (1.0 - float(alpha)) * filtered_translation
                    + float(alpha) * observed[:3, 3]
                )
                filtered_rotation = _rotation_step(
                    filtered_rotation, observed[:3, :3], float(alpha)
                )
            filtered = np.eye(4, dtype=np.float64)
            filtered[:3, :3] = filtered_rotation
            filtered[:3, 3] = filtered_translation
            correction = filtered @ np.linalg.inv(observed)
            _transform_frame_inplace(output, frame, correction)
            corrections.append({
                "frame": frame,
                "translation_m": float(np.linalg.norm(correction[:3, 3])),
                "rotation_deg": rotation_angle_deg(correction[:3, :3]),
            })
        centres = np.asarray(raw_centres)
        first_rotation = raw_rotations[0]
        shot_rows.append({
            "start": start,
            "end": end,
            "raw_max_centre_drift_m": float(np.linalg.norm(centres - centres[0], axis=1).max()),
            "raw_max_rotation_drift_deg": float(max(
                rotation_angle_deg(first_rotation.T @ value) for value in raw_rotations
            )),
        })
    return output, {
        "policy": "causal_shot_local_common_gauge_ema",
        "alpha": float(alpha),
        "shots": shot_rows,
        "max_applied_translation_m": max((row["translation_m"] for row in corrections), default=0.0),
        "max_applied_rotation_deg": max((row["rotation_deg"] for row in corrections), default=0.0),
        "camera_human_relative_geometry_invariant": True,
    }


def boundary_permutation_ids(
    arrays: ArrayDict,
    boundary: int,
    pairs: list[tuple[int, int]],
) -> tuple[ArrayDict, dict[str, Any]]:
    """Apply the v15 single-boundary native-slot permutation without geometry changes."""

    output = clone_arrays(arrays)
    if boundary <= 0 or boundary >= len(output["valid"]):
        return output, {"mapping": {}, "reason": "boundary_out_of_range"}
    pre_valid = output["valid"][boundary - 1].astype(bool)
    post_valid = output["valid"][boundary].astype(bool)
    used = set(int(value) for value in output["persistent_ids"][boundary - 1, pre_valid] if value >= 0)
    next_id = max(used, default=-1) + 1
    native_to_persistent: dict[int, int] = {}
    for pre_index, post_index in pairs:
        if not pre_valid[int(pre_index)] or not post_valid[int(post_index)]:
            continue
        native = int(output["native_ids"][boundary, int(post_index)])
        persistent = int(output["persistent_ids"][boundary - 1, int(pre_index)])
        if native >= 0 and persistent >= 0:
            native_to_persistent[native] = persistent
    new_slots = []
    for frame in range(boundary, len(output["valid"])):
        for index in np.flatnonzero(output["valid"][frame].astype(bool)):
            native = int(output["native_ids"][frame, index])
            if native not in native_to_persistent:
                native_to_persistent[native] = next_id
                new_slots.append({"frame": frame, "native_id": native, "persistent_id": next_id})
                next_id += 1
            output["persistent_ids"][frame, index] = native_to_persistent[native]
    return output, {
        "policy": "single_boundary_permutation_then_native_slot",
        "mapping": {str(key): int(value) for key, value in sorted(native_to_persistent.items())},
        "new_slots": new_slots,
        "valid_detection_count_preserved": int(output["valid"].sum()) == int(arrays["valid"].sum()),
    }


def _pelvis(joints: np.ndarray) -> np.ndarray:
    return np.asarray(joints)[..., [1, 2], :].mean(axis=-2)


def _pre_velocity(arrays: ArrayDict, boundary: int, person_index: int, horizon: int = 4) -> np.ndarray:
    identity = int(arrays["persistent_ids"][boundary - 1, person_index])
    points = []
    for frame in range(max(0, boundary - horizon - 1), boundary):
        indices = np.flatnonzero(
            arrays["valid"][frame].astype(bool)
            & (arrays["persistent_ids"][frame] == identity)
        )
        if len(indices):
            points.append(_pelvis(arrays["joints_world"][frame, indices[0]]))
    if len(points) < 2:
        return np.zeros(3, dtype=np.float64)
    steps = np.diff(np.asarray(points, dtype=np.float64), axis=0)
    velocity = np.median(steps, axis=0)
    norm = float(np.linalg.norm(velocity))
    return velocity * min(1.0, 0.10 / max(norm, 1e-12))


def _rigid_fit(target: np.ndarray, source: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    source = np.asarray(source, dtype=np.float64).reshape(-1, 3)
    if target.shape != source.shape or len(target) < 3:
        raise ValueError(f"rigid fit shape mismatch: {target.shape} vs {source.shape}")
    target_mean = target.mean(axis=0)
    source_mean = source.mean(axis=0)
    left, _, right = np.linalg.svd((source - source_mean).T @ (target - target_mean))
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0:
        right[-1] *= -1
        rotation = right.T @ left.T
    translation = target_mean - rotation @ source_mean
    return rotation, translation


def coupled_boundary_register(
    arrays: ArrayDict,
    boundary: int,
    pairs: list[tuple[int, int]],
    kind: RegistrationKind,
    blend: float,
    use_velocity_target: bool = False,
) -> tuple[ArrayDict, dict[str, Any]]:
    """Estimate one human-anchored SE(3) and move the entire post shot with it."""

    if kind == "none" or float(blend) == 0.0:
        return clone_arrays(arrays), {
            "policy": "none", "accepted": False,
            "camera_human_relative_geometry_invariant": True,
        }
    if kind not in {"translation", "se3"}:
        raise ValueError(kind)
    if not 0.0 <= float(blend) <= 1.0:
        raise ValueError(f"boundary blend must be in [0, 1], got {blend}")
    if boundary <= 0 or boundary >= len(arrays["valid"]):
        return clone_arrays(arrays), {"policy": kind, "accepted": False, "reason": "boundary_out_of_range"}
    target_rows, source_rows, root_offsets = [], [], []
    accepted_pairs = []
    for pre_index, post_index in pairs:
        pre_index, post_index = int(pre_index), int(post_index)
        if not arrays["valid"][boundary - 1, pre_index] or not arrays["valid"][boundary, post_index]:
            continue
        target = np.asarray(arrays["joints_world"][boundary - 1, pre_index, TORSO_JOINTS], dtype=np.float64)
        if use_velocity_target:
            target = target + _pre_velocity(arrays, boundary, pre_index)
        source = np.asarray(arrays["joints_world"][boundary, post_index, TORSO_JOINTS], dtype=np.float64)
        target_rows.append(target)
        source_rows.append(source)
        root_offsets.append(_pelvis(target) - _pelvis(source))
        accepted_pairs.append((pre_index, post_index))
    if not accepted_pairs:
        return clone_arrays(arrays), {
            "policy": kind, "accepted": False, "reason": "no_valid_boundary_pairs",
            "camera_human_relative_geometry_invariant": True,
        }
    target = np.concatenate(target_rows, axis=0)
    source = np.concatenate(source_rows, axis=0)
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    if kind == "translation":
        full_rotation = np.eye(3, dtype=np.float64)
        full_translation = np.median(np.asarray(root_offsets), axis=0)
        blended_rotation = full_rotation
        blended_translation = float(blend) * full_translation
    else:
        full_rotation, full_translation = _rigid_fit(target, source)
        blended_rotation = Rotation.from_rotvec(
            float(blend) * Rotation.from_matrix(full_rotation).as_rotvec()
        ).as_matrix()
        desired_mean = (1.0 - float(blend)) * source_mean + float(blend) * target_mean
        blended_translation = desired_mean - blended_rotation @ source_mean
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = blended_rotation
    transform[:3, 3] = blended_translation
    output = clone_arrays(arrays)
    for frame in range(boundary, len(output["valid"])):
        _transform_frame_inplace(output, frame, transform)
    residual_before = np.linalg.norm(source - target, axis=1)
    residual_after = np.linalg.norm(transform_points(transform, source) - target, axis=1)
    return output, {
        "policy": f"human_anchored_coupled_boundary_{kind}",
        "accepted": True,
        "matched_pairs": accepted_pairs,
        "matched_count": len(accepted_pairs),
        "blend": float(blend),
        "use_velocity_target": bool(use_velocity_target),
        "translation_m": float(np.linalg.norm(transform[:3, 3])),
        "rotation_deg": rotation_angle_deg(transform[:3, :3]),
        "full_translation_m": float(np.linalg.norm(full_translation)),
        "full_rotation_deg": rotation_angle_deg(full_rotation),
        "torso_residual_before_m": float(residual_before.mean()),
        "torso_residual_after_m": float(residual_after.mean()),
        "camera_human_relative_geometry_invariant": True,
        "transform": transform.tolist(),
    }


def causal_root_filter(
    arrays: ArrayDict,
    alpha: float,
    beta: float,
) -> tuple[ArrayDict, dict[str, Any]]:
    """Causal alpha-beta filter of person translation only."""

    if not 0.0 <= float(alpha) <= 1.0 or not 0.0 <= float(beta) <= 1.0:
        raise ValueError((alpha, beta))
    output = clone_arrays(arrays)
    state: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    applied = []
    for frame in range(len(output["valid"])):
        for index in np.flatnonzero(output["valid"][frame].astype(bool)):
            identity = int(output["persistent_ids"][frame, index])
            observed = _pelvis(output["joints_world"][frame, index]).astype(np.float64)
            if identity not in state:
                state[identity] = (observed.copy(), np.zeros(3, dtype=np.float64))
                continue
            previous, velocity = state[identity]
            predicted = previous + velocity
            residual = observed - predicted
            filtered = predicted + float(alpha) * residual
            velocity = velocity + float(beta) * residual
            shift = filtered - observed
            output["joints_world"][frame, index] += shift
            output["vertices_world"][frame, index] += shift
            state[identity] = (filtered, velocity)
            applied.append(float(np.linalg.norm(shift)))
    return output, {
        "policy": "causal_alpha_beta_root_translation",
        "alpha": float(alpha), "beta": float(beta),
        "max_shift_m": max(applied, default=0.0),
        "mean_shift_m": float(np.mean(applied)) if applied else 0.0,
        "camera_changed": False,
        "body_pose_changed": False,
    }


def apply_candidate(
    arrays: ArrayDict,
    boundary: int,
    pairs: list[tuple[int, int]],
    candidate: Candidate,
) -> tuple[ArrayDict, dict[str, Any]]:
    """Materialize one candidate from the same frozen B0 cache."""

    current, identity_debug = boundary_permutation_ids(arrays, boundary, pairs)
    exact_m15_fallback = clone_arrays(current)
    current, gauge_debug = causal_shot_gauge_stabilize(
        current, boundary, candidate.camera_alpha
    )
    current, boundary_debug = coupled_boundary_register(
        current, boundary, pairs, candidate.boundary_kind,
        candidate.boundary_blend, candidate.use_velocity_target,
    )
    gate_enabled = (
        candidate.gate_max_boundary_residual_m is not None
        or candidate.gate_min_matches > 0
    )
    residual = boundary_debug.get("torso_residual_after_m")
    matched_count = int(boundary_debug.get("matched_count", 0))
    gate_reasons = []
    if not boundary_debug.get("accepted", False):
        gate_reasons.append("boundary_registration_unavailable")
    if matched_count < int(candidate.gate_min_matches):
        gate_reasons.append("insufficient_boundary_matches")
    if (
        candidate.gate_max_boundary_residual_m is not None
        and (residual is None or float(residual) > candidate.gate_max_boundary_residual_m)
    ):
        gate_reasons.append("boundary_residual_too_large")
    gate_accepted = not gate_enabled or not gate_reasons
    gate_debug = {
        "enabled": gate_enabled,
        "accepted": gate_accepted,
        "max_boundary_residual_m": candidate.gate_max_boundary_residual_m,
        "min_matches": int(candidate.gate_min_matches),
        "observed_boundary_residual_m": residual,
        "observed_matches": matched_count,
        "reasons": gate_reasons,
        "fallback": "exact_m15_geometry_after_boundary_permutation",
        "gt_used": False,
    }
    if gate_enabled and not gate_accepted:
        return exact_m15_fallback, {
            "candidate": candidate.__dict__,
            "identity": identity_debug,
            "shot_gauge": gauge_debug,
            "boundary": boundary_debug,
            "reliability_gate": gate_debug,
            "root_filter": None,
            "runtime_contract": {
                "gt_used": False,
                "future_frames_used": 0,
                "pre_frames_rewritten_after_emission": False,
                "exact_m15_fallback": True,
            },
        }
    root_debug = None
    if candidate.root_alpha is not None:
        current, root_debug = causal_root_filter(
            current, candidate.root_alpha, candidate.root_beta
        )
    return current, {
        "candidate": candidate.__dict__,
        "identity": identity_debug,
        "shot_gauge": gauge_debug,
        "boundary": boundary_debug,
        "reliability_gate": gate_debug,
        "root_filter": root_debug,
        "runtime_contract": {
            "gt_used": False,
            "future_frames_used": 0,
            "pre_frames_rewritten_after_emission": False,
            "exact_m15_fallback": False,
        },
    }


def exploration_candidates() -> list[Candidate]:
    """Finite train01-only grid; do not expand after validation is read."""

    values = [Candidate("v16_0_m15_geometry")]
    values.extend(Candidate(f"csgs_a{alpha:g}", camera_alpha=alpha) for alpha in (0.0, 0.05, 0.10, 0.25, 0.50))
    for kind in ("translation", "se3"):
        for blend in (0.25, 0.50, 0.75, 1.0):
            values.append(Candidate(f"hcbr_{kind}_b{blend:g}", boundary_kind=kind, boundary_blend=blend))
            values.append(Candidate(
                f"csgs0_hcbr_{kind}_b{blend:g}", camera_alpha=0.0,
                boundary_kind=kind, boundary_blend=blend,
            ))
            values.append(Candidate(
                f"csgs0_hcbr_{kind}_b{blend:g}_vel", camera_alpha=0.0,
                boundary_kind=kind, boundary_blend=blend, use_velocity_target=True,
            ))
    for camera_alpha in (0.0, 0.10):
        for root_alpha, root_beta in ((0.6, 0.05), (0.8, 0.05), (0.9, 0.02)):
            values.append(Candidate(
                f"csgs_a{camera_alpha:g}_root_a{root_alpha:g}_b{root_beta:g}",
                camera_alpha=camera_alpha, root_alpha=root_alpha, root_beta=root_beta,
            ))
    names = [value.name for value in values]
    if len(names) != len(set(names)):
        raise AssertionError("duplicate candidate names")
    return values
