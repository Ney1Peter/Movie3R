"""Angular/reprojection-safe shared-group damping for frozen BRTC-LC v1.

The runtime first executes frozen BRTC-LC v1 unchanged.  It may change only
the shared group component, and only for a complete one-to-one association in
which every frozen ray proposal was accepted.  For each candidate group alpha
it measures the angular displacement of fixed-post-camera rays from the
uncorrected post joints to the candidate corrected joints.  The largest alpha
that satisfies a frozen angular budget is selected; the frozen individual
residual remains exact.

No intrinsics, image features, GT, future frame, persistent state, camera
update, or extra model is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from versions.v14.b0_person_triangulation import (
    DEFAULT_CONFIG,
    PersonTriangulationConfig,
    refine_matched_people,
)


CORE_JOINT_INDICES = (0, 1, 2, 3, 6, 9, 12, 13, 14, 16, 17)
DEFAULT_ALPHA_VALUES = tuple(float(value) for value in np.linspace(0.5, 1.0, 11))
VALID_STATISTICS = frozenset(("all_median", "all_p90", "core_median", "core_p90"))


@dataclass(frozen=True)
class AngularSafeFAGDConfig:
    angular_budget_deg: float = 2.0
    statistic: str = "core_p90"
    alpha_values: tuple[float, ...] = DEFAULT_ALPHA_VALUES

    def __post_init__(self) -> None:
        budget = float(self.angular_budget_deg)
        if not np.isfinite(budget) or budget <= 0.0:
            raise ValueError("angular_budget_deg must be finite and positive")
        if self.statistic not in VALID_STATISTICS:
            raise ValueError(f"Unknown angular statistic: {self.statistic}")
        values = tuple(float(value) for value in self.alpha_values)
        if not values or any(not np.isfinite(value) for value in values):
            raise ValueError("alpha_values must contain finite values")
        if tuple(sorted(set(values))) != values:
            raise ValueError("alpha_values must be sorted and unique")
        if values[0] < 0.5 or values[-1] > 1.0 or 1.0 not in values:
            raise ValueError("alpha_values must remain in [0.5, 1.0] and include 1.0")


DEFAULT_ANGULAR_SAFE_FAGD_CONFIG = AngularSafeFAGDConfig()


def _points(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim < 1 or result.shape[-1] != 3 or not np.isfinite(result).all():
        raise ValueError("Points must be finite with final dimension 3")
    return result


def _selected_joints(joints: Any, statistic: str) -> np.ndarray:
    points = _points(joints).reshape(-1, 3)
    if statistic.startswith("all_"):
        return points
    indices = [index for index in CORE_JOINT_INDICES if index < len(points)]
    return points[indices] if indices else points


def ray_angular_displacement_deg(
    points_world: Any,
    shift_world: Any,
    camera_center_world: Any,
) -> np.ndarray:
    """Return per-point angle between the before/after fixed-camera rays."""

    points = _points(points_world).reshape(-1, 3)
    shift = _points(shift_world).reshape(3)
    center = _points(camera_center_world).reshape(3)
    before = points - center
    after = points + shift - center
    before_norm = np.linalg.norm(before, axis=1)
    after_norm = np.linalg.norm(after, axis=1)
    valid = (before_norm > 1e-12) & (after_norm > 1e-12)
    if not np.any(valid):
        return np.empty((0,), dtype=np.float64)
    cosine = np.sum(before[valid] * after[valid], axis=1) / (
        before_norm[valid] * after_norm[valid]
    )
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def angular_statistic_deg(values: Any, statistic: str) -> float:
    angles = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(angles) or not np.isfinite(angles).all():
        return float("inf")
    if statistic.endswith("_median"):
        return float(np.median(angles))
    if statistic.endswith("_p90"):
        return float(np.quantile(angles, 0.9))
    raise ValueError(f"Unknown angular statistic: {statistic}")


def refine_matched_people_angular_safe_fagd(
    pre_camera: np.ndarray,
    post_camera: np.ndarray,
    pre_people: list[dict[str, Any]],
    post_people: list[dict[str, Any]],
    matches: Iterable[tuple[int, int]],
    config: PersonTriangulationConfig = DEFAULT_CONFIG,
    angular_config: AngularSafeFAGDConfig = DEFAULT_ANGULAR_SAFE_FAGD_CONFIG,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a shared-group alpha from fixed-post-camera ray displacement."""

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
    previous_count = len(pre_people)
    current_count = len(post_people)
    population_max = max(previous_count, current_count)
    matched_count = int(base_debug["matched_count"])
    accepted_count = int(base_debug["accepted_count"])
    strict_gate = bool(
        accepted_count == matched_count == population_max and population_max > 0
    )
    group = _points(base_debug["group_shift_world"]).reshape(3).copy()
    residual_lambda = float(base_debug["selected_residual_lambda"])
    records_by_post = {
        int(record["post_index"]): record for record in base_debug["people"]
    }
    score_by_alpha: dict[float, float] = {}
    shifts_by_alpha: dict[float, dict[int, np.ndarray]] = {}
    budget = float(angular_config.angular_budget_deg)

    if strict_gate:
        post_camera_array = np.asarray(post_camera, dtype=np.float64)
        if post_camera_array.shape != (4, 4) or not np.isfinite(post_camera_array).all():
            raise ValueError("post_camera must be a finite 4x4 matrix")
        camera_center = post_camera_array[:3, 3]
        for alpha in angular_config.alpha_values:
            per_post: dict[int, np.ndarray] = {}
            angles = []
            for _, post_index in materialized_matches:
                record = records_by_post[post_index]
                individual = _points(record["individual_shift_world"]).reshape(3)
                residual = residual_lambda * (individual - group)
                shift = float(alpha) * group + residual
                per_post[post_index] = shift
                joints = _selected_joints(
                    post_people[post_index]["joints"], angular_config.statistic
                )
                angles.append(
                    ray_angular_displacement_deg(joints, shift, camera_center)
                )
            flattened = (
                np.concatenate(angles) if angles else np.empty((0,), dtype=np.float64)
            )
            score_by_alpha[float(alpha)] = angular_statistic_deg(
                flattened, angular_config.statistic
            )
            shifts_by_alpha[float(alpha)] = per_post

        eligible = [
            alpha
            for alpha in angular_config.alpha_values
            if score_by_alpha[float(alpha)] <= budget + 1e-12
        ]
        if eligible:
            selected_alpha = float(max(eligible))
            budget_satisfied = True
        else:
            selected_alpha = float(
                min(
                    angular_config.alpha_values,
                    key=lambda alpha: (score_by_alpha[float(alpha)], -float(alpha)),
                )
            )
            budget_satisfied = False
    else:
        selected_alpha = 1.0
        budget_satisfied = False

    # Alpha 1.0 and failed strict gates return the exact frozen-v1 geometry.
    # Rebuild only when a genuine shared-group damping action is selected.
    damping_applied = bool(strict_gate and selected_alpha < 1.0 - 1e-15)
    records = []
    selected_shifts = shifts_by_alpha.get(selected_alpha, {})
    for base_record in base_debug["people"]:
        record = dict(base_record)
        post_index = int(base_record["post_index"])
        base_final = _points(base_record["final_shift_world"]).reshape(3).copy()
        final_shift = selected_shifts.get(post_index, base_final)
        if damping_applied:
            for key in ("root", "joints", "vertices"):
                if key in corrected[post_index]:
                    corrected[post_index][key] = (
                        _points(post_people[post_index][key]) + final_shift
                    )
        record.update(
            {
                "base_final_shift_world": base_final,
                "final_shift_world": final_shift,
                "selected_group_alpha": selected_alpha,
            }
        )
        records.append(record)

    debug = dict(base_debug)
    debug.update(
        {
            "camera_update": "none",
            "previous_observable_count": previous_count,
            "current_observable_count": current_count,
            "population_max": population_max,
            "strict_full_one_to_one_all_accepted": strict_gate,
            "strict_gate_formula": (
                "accepted_count == matched_count == "
                "max(previous_observable_count, current_observable_count) > 0"
            ),
            "angular_statistic": angular_config.statistic,
            "angular_budget_deg": budget,
            "angular_score_by_alpha_deg": score_by_alpha,
            "angular_budget_satisfied": budget_satisfied,
            "selected_group_alpha": selected_alpha,
            "angular_damping_applied": damping_applied,
            "exact_v1_output": not damping_applied,
            "base_group_shift_world": group,
            "group_shift_world": selected_alpha * group if strict_gate else group,
            "people": records,
        }
    )
    return corrected, debug
