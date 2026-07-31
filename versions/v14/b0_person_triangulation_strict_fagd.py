"""Strict deployable full-accept group-only damping for frozen BRTC-LC v1.

The candidate first runs frozen BRTC-LC v1 unchanged.  Group-only ``alpha``
damping is allowed only when the current boundary is a complete one-to-one
association and every matched person's frozen ray evidence is accepted::

    accepted_count == matched_count
        == max(len(pre_people), len(post_people)) > 0

When the predicate is false, the frozen-v1 corrected people are returned
without rebuilding their geometry, giving an exact-v1 fallback.  When true,
only the shared group median is scaled; the frozen individual residual and
selected residual lambda are preserved.
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


@dataclass(frozen=True)
class StrictFAGDConfig:
    alpha: float = 0.9

    def __post_init__(self) -> None:
        value = float(self.alpha)
        if not np.isfinite(value) or not (0.0 < value <= 1.0):
            raise ValueError("alpha must be finite and in (0, 1]")


DEFAULT_STRICT_FAGD_CONFIG = StrictFAGDConfig()


def _points(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape[-1] != 3 or not np.isfinite(result).all():
        raise ValueError("Person points must be finite with final dimension 3")
    return result


def refine_matched_people_strict_fagd(
    pre_camera: np.ndarray,
    post_camera: np.ndarray,
    pre_people: list[dict[str, Any]],
    post_people: list[dict[str, Any]],
    matches: Iterable[tuple[int, int]],
    config: PersonTriangulationConfig = DEFAULT_CONFIG,
    fagd_config: StrictFAGDConfig = DEFAULT_STRICT_FAGD_CONFIG,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply alpha to the frozen group median only under the strict predicate."""

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
    alpha = float(fagd_config.alpha)

    # Preserve the frozen runtime object directly on fallback.  In particular,
    # do not subtract/re-add a zero shift, cast arrays, or deep-copy geometry.
    records = []
    group = _points(base_debug["group_shift_world"]).copy()
    residual_lambda = float(base_debug["selected_residual_lambda"])
    for base_record in base_debug["people"]:
        record = dict(base_record)
        base_final = _points(base_record["final_shift_world"]).copy()
        if strict_gate:
            individual = _points(base_record["individual_shift_world"])
            residual = residual_lambda * (individual - group)
            final_shift = alpha * group + residual
            post_index = int(base_record["post_index"])
            for key in ("root", "joints", "vertices"):
                if key in corrected[post_index]:
                    corrected[post_index][key] = (
                        _points(post_people[post_index][key]) + final_shift
                    )
        else:
            residual = base_final - group if bool(base_record["accepted"]) else np.zeros(3)
            final_shift = base_final
        record.update(
            {
                "base_final_shift_world": base_final,
                "frozen_individual_residual_world": residual,
                "final_shift_world": final_shift,
                "group_only_alpha": alpha if strict_gate else 1.0,
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
            "matched_count": matched_count,
            "accepted_count": accepted_count,
            "strict_full_one_to_one_all_accepted": strict_gate,
            "strict_gate_formula": (
                "accepted_count == matched_count == "
                "max(previous_observable_count, current_observable_count) > 0"
            ),
            "exact_v1_fallback": not strict_gate,
            "group_damping_applied": strict_gate and alpha != 1.0,
            "group_only_alpha": alpha if strict_gate else 1.0,
            "base_group_shift_world": group,
            "group_shift_world": alpha * group if strict_gate else group,
            "people": records,
        }
    )
    return corrected, debug
