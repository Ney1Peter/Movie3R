"""Strictly-online damped BRTC-LC person refinement.

This module leaves the frozen BRTC-LC v1 implementation untouched.  It tests
one additional scalar: multiply every accepted triangulated person shift by a
fixed observable-independent factor before layout consensus.  Cameras, scene,
pose, shape and orientation remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from versions.v14.b0_person_triangulation import (
    DEFAULT_CONFIG,
    PersonTriangulationConfig,
    gated_shift,
    person_evidence,
)


@dataclass(frozen=True)
class DampedTriangulationConfig:
    action_scale: float = 0.80

    def __post_init__(self) -> None:
        if not (0.0 < float(self.action_scale) <= 1.0):
            raise ValueError("action_scale must be in (0, 1]")


DEFAULT_DAMPED_CONFIG = DampedTriangulationConfig()


def _points(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape[-1] != 3 or not np.isfinite(result).all():
        raise ValueError("Person points must be finite with final dimension 3")
    return result


def _copy_person(person: dict[str, Any]) -> dict[str, Any]:
    output = dict(person)
    for key in ("root", "joints", "vertices"):
        if key in person:
            output[key] = _points(person[key]).copy()
    return output


def refine_matched_people_damped(
    pre_camera: np.ndarray,
    post_camera: np.ndarray,
    pre_people: list[dict[str, Any]],
    post_people: list[dict[str, Any]],
    matches: Iterable[tuple[int, int]],
    base_config: PersonTriangulationConfig = DEFAULT_CONFIG,
    damped_config: DampedTriangulationConfig = DEFAULT_DAMPED_CONFIG,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply fixed damping before the frozen BRTC-LC layout consensus."""

    pre_camera = np.asarray(pre_camera, dtype=np.float64)
    post_camera = np.asarray(post_camera, dtype=np.float64)
    matches = tuple((int(first), int(second)) for first, second in matches)
    if pre_camera.shape != (4, 4) or post_camera.shape != (4, 4):
        raise ValueError("Cameras must be 4x4 camera-to-world matrices")
    if len({first for first, _ in matches}) != len(matches) or len(
        {second for _, second in matches}
    ) != len(matches):
        raise ValueError("Matches must be one-to-one")

    records = []
    for pre_index, post_index in matches:
        if not (0 <= pre_index < len(pre_people) and 0 <= post_index < len(post_people)):
            raise IndexError("Match index outside person arrays")
        evidence = person_evidence(
            pre_people[pre_index],
            post_people[post_index],
            pre_camera,
            post_camera,
            base_config,
        )
        undamped_shift, accepted = gated_shift(evidence, base_config)
        damped_shift = float(damped_config.action_scale) * undamped_shift
        records.append(
            {
                "pre_index": pre_index,
                "post_index": post_index,
                "evidence": evidence,
                "undamped_shift_world": undamped_shift,
                "individual_shift_world": damped_shift,
                "accepted": accepted,
            }
        )

    accepted_shifts = [row["individual_shift_world"] for row in records if row["accepted"]]
    group_shift = (
        np.median(np.stack(accepted_shifts), axis=0)
        if accepted_shifts
        else np.zeros(3, dtype=np.float64)
    )
    objectives: dict[float, float] = {}
    for residual_lambda in base_config.residual_lambda_grid:
        proposed = {}
        for row in records:
            if row["accepted"]:
                shift = group_shift + residual_lambda * (
                    row["individual_shift_world"] - group_shift
                )
            else:
                shift = np.zeros(3, dtype=np.float64)
            proposed[row["post_index"]] = _points(
                post_people[row["post_index"]]["root"]
            ) + shift
        errors = []
        for first_index, first in enumerate(records):
            for second in records[first_index + 1 :]:
                post_vector = proposed[first["post_index"]] - proposed[second["post_index"]]
                pre_vector = _points(pre_people[first["pre_index"]]["root"]) - _points(
                    pre_people[second["pre_index"]]["root"]
                )
                errors.append(float(np.linalg.norm(post_vector - pre_vector)))
        objectives[float(residual_lambda)] = float(np.mean(errors)) if errors else 0.0
    selected_lambda = min(
        base_config.residual_lambda_grid,
        key=lambda value: objectives[float(value)],
    )

    corrected = [_copy_person(person) for person in post_people]
    for row in records:
        if row["accepted"]:
            shift = group_shift + selected_lambda * (
                row["individual_shift_world"] - group_shift
            )
        else:
            shift = np.zeros(3, dtype=np.float64)
        row["final_shift_world"] = shift
        for key in ("root", "joints", "vertices"):
            if key in corrected[row["post_index"]]:
                corrected[row["post_index"]][key] += shift

    return corrected, {
        "camera_update": "none",
        "action_scale": float(damped_config.action_scale),
        "matched_count": len(records),
        "accepted_count": sum(row["accepted"] for row in records),
        "group_shift_world": group_shift,
        "selected_residual_lambda": float(selected_lambda),
        "observable_layout_objective_by_lambda": objectives,
        "people": records,
    }
