"""Strictly-online association-completeness weighting for frozen BRTC-LC v1.

The frozen :mod:`versions.v14.b0_person_triangulation` runtime remains the
sole source of triangulation proposals, gates, group consensus and layout
selection.  This wrapper only scales each *accepted final* BRTC translation by
the observable association completeness at the current boundary::

    completeness = matched_count / max(previous_count, current_count)

``previous_count`` and ``current_count`` are the lengths of the supplied
``pre_people`` and ``post_people`` arrays.  Thus the wrapper needs no identity
labels, ground truth, future frame or persistent non-causal state.  A complete
one-to-one association has scale one and is exactly equivalent to frozen BRTC;
people entering, leaving or remaining unmatched conservatively reduce the
accepted action.  Rejected and unmatched post people retain frozen BRTC's
exact-B0 fallback.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from versions.v14.b0_person_triangulation import (
    DEFAULT_CONFIG,
    PersonTriangulationConfig,
    refine_matched_people,
)


def association_completeness(
    previous_observable_count: int,
    current_observable_count: int,
    matched_count: int,
) -> float:
    """Return the conservative observable matching fraction in ``[0, 1]``.

    Dividing by the larger visible population makes any entry/exit, miss or
    unmatched observation reduce confidence.  An empty boundary returns zero:
    there is no person action to apply or evidence to call complete.
    """

    previous = int(previous_observable_count)
    current = int(current_observable_count)
    matched = int(matched_count)
    if min(previous, current, matched) < 0:
        raise ValueError("Observable and matched person counts must be non-negative")
    if matched > min(previous, current):
        raise ValueError("matched_count cannot exceed either observable person count")
    denominator = max(previous, current)
    return float(matched / denominator) if denominator else 0.0


def _points(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape[-1] != 3 or not np.isfinite(result).all():
        raise ValueError("Person points must be finite with final dimension 3")
    return result


def refine_matched_people_completeness_weighted(
    pre_camera: np.ndarray,
    post_camera: np.ndarray,
    pre_people: list[dict[str, Any]],
    post_people: list[dict[str, Any]],
    matches: Iterable[tuple[int, int]],
    config: PersonTriangulationConfig = DEFAULT_CONFIG,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run frozen BRTC and weight accepted actions by match completeness.

    The signature intentionally matches frozen BRTC-LC v1.  The person arrays
    must contain exactly the people observable at the last pre-cut and first
    post-cut frames.  ``matches`` is materialized once so generators remain
    valid for both the frozen runtime and the completeness audit.
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

    previous_count = len(pre_people)
    current_count = len(post_people)
    matched_count = int(base_debug["matched_count"])
    completeness = association_completeness(
        previous_count,
        current_count,
        matched_count,
    )
    action_scale = completeness

    # Preserve frozen BRTC output directly when scale is one.  This makes the
    # full-match equivalence bit-exact, rather than merely numerically close.
    # For partial association, only accepted matched people are rebuilt from
    # the original B0 geometry; rejected and unmatched entries are untouched.
    weighted_people = []
    for base_record in base_debug["people"]:
        record = dict(base_record)
        base_shift = _points(base_record["final_shift_world"]).copy()
        weighted_shift = action_scale * base_shift
        record["base_final_shift_world"] = base_shift
        record["final_shift_world"] = weighted_shift
        record["action_scale"] = action_scale
        weighted_people.append(record)

        if action_scale == 1.0 or not bool(base_record["accepted"]):
            continue
        post_index = int(base_record["post_index"])
        for key in ("root", "joints", "vertices"):
            if key in corrected[post_index]:
                corrected[post_index][key] = (
                    _points(post_people[post_index][key]) + weighted_shift
                )

    base_group_shift = _points(base_debug["group_shift_world"]).copy()
    debug = dict(base_debug)
    debug.update(
        {
            "camera_update": "none",
            "previous_observable_count": previous_count,
            "current_observable_count": current_count,
            "matched_count": matched_count,
            "completeness_denominator": max(previous_count, current_count),
            "completeness": completeness,
            "action_scale": action_scale,
            "weight_formula": "matched_count / max(previous_observable_count, current_observable_count)",
            "weight_application_stage": "after_frozen_brtc_layout_consensus",
            "base_group_shift_world": base_group_shift,
            "group_shift_world": action_scale * base_group_shift,
            "people": weighted_people,
        }
    )
    return corrected, debug
