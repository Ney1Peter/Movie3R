"""Joint ray-space layout least squares on frozen BRTC-LC proposals.

For every matched person, frozen BRTC-LC v1 supplies an individual signed
action along the current post-camera pelvis ray and an observable reliability
gate.  This candidate keeps the gate and ray action prior, but replaces the
group-median/layout-grid consensus with one joint least-squares solve.

Only accepted people are variables.  Rejected matched people are fixed at zero
and may serve as layout anchors; unmatched post people are copied exactly.  No
camera, scene, pose, shape or orientation parameter is optimized.
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
class RayLayoutLeastSquaresConfig:
    """Configuration for the normalized layout-plus-prior objective."""

    prior_weight: float = 1.0

    def __post_init__(self) -> None:
        value = float(self.prior_weight)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("prior_weight must be finite and non-negative")


DEFAULT_RAY_LAYOUT_CONFIG = RayLayoutLeastSquaresConfig()


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


def _mean_pair_vector_objective(
    records: list[dict[str, Any]], actions: np.ndarray
) -> float:
    errors = []
    for first_index, first in enumerate(records):
        for second in records[first_index + 1 :]:
            first_action = (
                float(actions[first["variable_index"]])
                if first["variable_index"] is not None
                else 0.0
            )
            second_action = (
                float(actions[second["variable_index"]])
                if second["variable_index"] is not None
                else 0.0
            )
            corrected_vector = (
                first["post_root"]
                + first_action * first["ray_world"]
                - second["post_root"]
                - second_action * second["ray_world"]
            )
            pre_vector = first["pre_root"] - second["pre_root"]
            errors.append(float(np.sum(np.square(corrected_vector - pre_vector))))
    return float(np.mean(errors)) if errors else 0.0


def refine_matched_people_ray_layout_least_squares(
    pre_camera: np.ndarray,
    post_camera: np.ndarray,
    pre_people: list[dict[str, Any]],
    post_people: list[dict[str, Any]],
    matches: Iterable[tuple[int, int]],
    config: PersonTriangulationConfig = DEFAULT_CONFIG,
    layout_config: RayLayoutLeastSquaresConfig = DEFAULT_RAY_LAYOUT_CONFIG,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Solve accepted scalar ray actions from observable pre-root layout.

    The normalized objective is

    ``mean_pairs ||post_vector(a) - pre_vector||^2``
    ``+ prior_weight * mean_accepted (a - a_brtc_raw)^2``.

    ``a_brtc_raw`` is the frozen BRTC individual gated/capped action before its
    group/layout consensus.  If fewer than two matched people provide a usable
    pair constraint, the observable layout term is absent and the BRTC action
    prior is returned exactly for accepted people.
    """

    pre_camera = np.asarray(pre_camera, dtype=np.float64)
    post_camera = np.asarray(post_camera, dtype=np.float64)
    if pre_camera.shape != (4, 4) or post_camera.shape != (4, 4):
        raise ValueError("Cameras must be 4x4 camera-to-world matrices")
    materialized_matches = tuple(
        (int(pre_index), int(post_index)) for pre_index, post_index in matches
    )
    if len({first for first, _ in materialized_matches}) != len(materialized_matches) or len(
        {second for _, second in materialized_matches}
    ) != len(materialized_matches):
        raise ValueError("Matches must be one-to-one")

    records: list[dict[str, Any]] = []
    for pre_index, post_index in materialized_matches:
        if not (0 <= pre_index < len(pre_people) and 0 <= post_index < len(post_people)):
            raise IndexError("Match index outside person arrays")
        evidence = person_evidence(
            pre_people[pre_index],
            post_people[post_index],
            pre_camera,
            post_camera,
            config,
        )
        individual_shift, accepted = gated_shift(evidence, config)
        ray_world = _points(evidence["ray_world"])
        raw_action = float(np.dot(individual_shift, ray_world)) if accepted else 0.0
        records.append(
            {
                "pre_index": pre_index,
                "post_index": post_index,
                "pre_root": _points(pre_people[pre_index]["root"]),
                "post_root": _points(post_people[post_index]["root"]),
                "ray_world": ray_world,
                "evidence": evidence,
                "accepted": bool(accepted),
                "brtc_raw_action_m": raw_action,
                "brtc_individual_shift_world": individual_shift,
                "variable_index": None,
            }
        )

    accepted_records = [record for record in records if record["accepted"]]
    for variable_index, record in enumerate(accepted_records):
        record["variable_index"] = variable_index
    variable_count = len(accepted_records)
    prior_actions = np.asarray(
        [record["brtc_raw_action_m"] for record in accepted_records],
        dtype=np.float64,
    )

    layout_blocks = []
    layout_targets = []
    constrained_pair_count = 0
    for first_index, first in enumerate(records):
        for second in records[first_index + 1 :]:
            block = np.zeros((3, variable_count), dtype=np.float64)
            if first["variable_index"] is not None:
                block[:, first["variable_index"]] += first["ray_world"]
            if second["variable_index"] is not None:
                block[:, second["variable_index"]] -= second["ray_world"]
            if not np.any(block):
                continue
            target = (
                first["pre_root"]
                - second["pre_root"]
                - (first["post_root"] - second["post_root"])
            )
            layout_blocks.append(block)
            layout_targets.append(target)
            constrained_pair_count += 1

    prior_weight = float(layout_config.prior_weight)
    if variable_count == 0:
        actions = np.empty(0, dtype=np.float64)
        matrix_rank = 0
        singular_values = np.empty(0, dtype=np.float64)
    elif constrained_pair_count == 0:
        # No observable pair layout exists.  Returning the prior is causal and
        # deterministic even when prior_weight is zero.
        actions = prior_actions.copy()
        matrix_rank = 0
        singular_values = np.empty(0, dtype=np.float64)
    else:
        layout_matrix = np.concatenate(layout_blocks, axis=0) / np.sqrt(
            float(constrained_pair_count)
        )
        layout_target = np.concatenate(layout_targets, axis=0) / np.sqrt(
            float(constrained_pair_count)
        )
        solve_matrix_parts = [layout_matrix]
        solve_target_parts = [layout_target]
        if prior_weight > 0.0:
            ridge_scale = np.sqrt(prior_weight / float(variable_count))
            solve_matrix_parts.append(ridge_scale * np.eye(variable_count))
            solve_target_parts.append(ridge_scale * prior_actions)
        solve_matrix = np.concatenate(solve_matrix_parts, axis=0)
        solve_target = np.concatenate(solve_target_parts, axis=0)
        actions, _, matrix_rank, singular_values = np.linalg.lstsq(
            solve_matrix, solve_target, rcond=None
        )
        actions = np.asarray(actions, dtype=np.float64)

    unclipped_actions = actions.copy()
    if len(actions):
        actions = np.clip(actions, -float(config.cap_m), float(config.cap_m))

    corrected = [_copy_person(person) for person in post_people]
    for record in records:
        if record["accepted"]:
            action = float(actions[record["variable_index"]])
            shift = action * record["ray_world"]
        else:
            action = 0.0
            shift = np.zeros(3, dtype=np.float64)
        record["solved_action_unclipped_m"] = (
            float(unclipped_actions[record["variable_index"]])
            if record["accepted"]
            else 0.0
        )
        record["final_action_m"] = action
        record["final_shift_world"] = shift
        for key in ("root", "joints", "vertices"):
            if key in corrected[record["post_index"]]:
                corrected[record["post_index"]][key] += shift

    condition_number = (
        float(singular_values[0] / singular_values[-1])
        if len(singular_values) and singular_values[-1] > 0.0
        else float("inf")
    )
    debug = {
        "camera_update": "none",
        "matched_count": len(records),
        "accepted_count": variable_count,
        "rejected_count": len(records) - variable_count,
        "prior_weight": prior_weight,
        "objective_normalization": "mean_pair_squared_vector_error + prior_weight * mean_action_prior_squared_error",
        "constrained_pair_count": constrained_pair_count,
        "variable_count": variable_count,
        "matrix_rank": int(matrix_rank),
        "condition_number": condition_number,
        "clipped_action_count": int(
            np.count_nonzero(np.abs(unclipped_actions - actions) > 1e-12)
        ),
        "observable_layout_objective_before": _mean_pair_vector_objective(
            records, prior_actions
        ),
        "observable_layout_objective_after": _mean_pair_vector_objective(
            records, actions
        ),
        "people": records,
    }
    return corrected, debug
