"""GT-free runtime for B0-frozen two-view person fine alignment.

The caller supplies cameras and already-associated anonymous person geometries
in one shared world gauge.  The module never estimates or modifies a camera.
It returns copied post-cut people with bounded rigid translations only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


CORE5 = (0, 1, 2, 16, 17)


@dataclass(frozen=True)
class PersonTriangulationConfig:
    joint_ids: tuple[int, ...] = CORE5
    min_valid: int = 1
    max_median_gap_m: float = 0.20
    max_mad_m: float = 0.40
    min_median_sine: float = 0.025
    min_abs_raw_m: float = 0.0
    cap_m: float = 2.0
    residual_lambda_grid: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.0)


DEFAULT_CONFIG = PersonTriangulationConfig()


def _points(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape[-1] != 3 or not np.isfinite(result).all():
        raise ValueError("Person points must be finite with final dimension 3")
    return result


def closest_rays(
    origin_a: np.ndarray,
    direction_a: np.ndarray,
    origin_b: np.ndarray,
    direction_b: np.ndarray,
) -> tuple[np.ndarray, float, float, float, float]:
    origin_a, origin_b = _points(origin_a), _points(origin_b)
    direction_a, direction_b = _points(direction_a), _points(direction_b)
    direction_a = direction_a / max(float(np.linalg.norm(direction_a)), 1e-12)
    direction_b = direction_b / max(float(np.linalg.norm(direction_b)), 1e-12)
    system = np.stack((direction_a, -direction_b), axis=1)
    depths, _, _, _ = np.linalg.lstsq(system, origin_b - origin_a, rcond=None)
    point_a = origin_a + float(depths[0]) * direction_a
    point_b = origin_b + float(depths[1]) * direction_b
    dot = float(np.clip(np.dot(direction_a, direction_b), -1.0, 1.0))
    sine = float(np.sqrt(max(0.0, 1.0 - dot * dot)))
    return (
        0.5 * (point_a + point_b),
        float(depths[0]),
        float(depths[1]),
        float(np.linalg.norm(point_a - point_b)),
        sine,
    )


def person_evidence(
    pre_person: dict[str, Any],
    post_person: dict[str, Any],
    pre_camera: np.ndarray,
    post_camera: np.ndarray,
    config: PersonTriangulationConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    pre_camera = np.asarray(pre_camera, dtype=np.float64)
    post_camera = np.asarray(post_camera, dtype=np.float64)
    if pre_camera.shape != (4, 4) or post_camera.shape != (4, 4):
        raise ValueError("Cameras must be 4x4 camera-to-world matrices")
    pre_joints = _points(pre_person["joints"])
    post_joints = _points(post_person["joints"])
    post_root = _points(post_person["root"])
    ray = post_root - post_camera[:3, 3]
    ray = ray / max(float(np.linalg.norm(ray)), 1e-12)
    candidates, gaps, sines = [], [], []
    for joint_id in config.joint_ids:
        if joint_id >= len(pre_joints) or joint_id >= len(post_joints):
            continue
        joint_a, joint_b = pre_joints[joint_id], post_joints[joint_id]
        direction_a = joint_a - pre_camera[:3, 3]
        direction_b = joint_b - post_camera[:3, 3]
        if min(np.linalg.norm(direction_a), np.linalg.norm(direction_b)) <= 1e-8:
            continue
        midpoint, depth_a, depth_b, gap, sine = closest_rays(
            pre_camera[:3, 3], direction_a, post_camera[:3, 3], direction_b
        )
        if depth_a <= 0.0 or depth_b <= 0.0 or sine <= 1e-5:
            continue
        candidate_root = midpoint - (joint_b - post_root)
        delta = float(np.dot(candidate_root - post_root, ray))
        if np.isfinite(delta):
            candidates.append(delta)
            gaps.append(gap)
            sines.append(sine)
    if not candidates:
        return {
            "raw_m": float("nan"), "valid_count": 0,
            "median_gap_m": float("inf"), "max_gap_m": float("inf"),
            "median_sine": 0.0, "min_sine": 0.0,
            "mad_m": float("inf"), "ray_world": ray,
        }
    values = np.asarray(candidates, dtype=np.float64)
    center = float(np.median(values))
    return {
        "raw_m": center,
        "valid_count": len(values),
        "median_gap_m": float(np.median(gaps)),
        "max_gap_m": float(np.max(gaps)),
        "median_sine": float(np.median(sines)),
        "min_sine": float(np.min(sines)),
        "mad_m": float(np.median(np.abs(values - center))),
        "ray_world": ray,
    }


def gated_shift(
    evidence: dict[str, Any],
    config: PersonTriangulationConfig = DEFAULT_CONFIG,
) -> tuple[np.ndarray, bool]:
    raw = float(evidence["raw_m"])
    accepted = bool(
        np.isfinite(raw)
        and int(evidence["valid_count"]) >= config.min_valid
        and float(evidence["median_gap_m"]) <= config.max_median_gap_m
        and float(evidence["mad_m"]) <= config.max_mad_m
        and float(evidence["median_sine"]) >= config.min_median_sine
        and abs(raw) >= config.min_abs_raw_m
    )
    action = float(np.clip(raw, -config.cap_m, config.cap_m)) if accepted else 0.0
    return action * _points(evidence["ray_world"]), accepted


def _copy_person(person: dict[str, Any]) -> dict[str, Any]:
    output = dict(person)
    for key in ("root", "joints", "vertices"):
        if key in person:
            output[key] = _points(person[key]).copy()
    return output


def refine_matched_people(
    pre_camera: np.ndarray,
    post_camera: np.ndarray,
    pre_people: list[dict[str, Any]],
    post_people: list[dict[str, Any]],
    matches: Iterable[tuple[int, int]],
    config: PersonTriangulationConfig = DEFAULT_CONFIG,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Refine matched post people and preserve unmatched/rejected people exactly.

    ``matches`` contains ``(pre_index, post_index)`` pairs produced by any
    upstream association module.  No identity label or dataset field is used.
    """
    pre_camera = np.asarray(pre_camera, dtype=np.float64)
    post_camera = np.asarray(post_camera, dtype=np.float64)
    matches = tuple((int(first), int(second)) for first, second in matches)
    if len({first for first, _ in matches}) != len(matches) or len(
        {second for _, second in matches}
    ) != len(matches):
        raise ValueError("Matches must be one-to-one")
    records = []
    for pre_index, post_index in matches:
        if not (0 <= pre_index < len(pre_people) and 0 <= post_index < len(post_people)):
            raise IndexError("Match index outside person arrays")
        evidence = person_evidence(
            pre_people[pre_index], post_people[post_index], pre_camera, post_camera, config
        )
        shift, accepted = gated_shift(evidence, config)
        records.append({
            "pre_index": pre_index,
            "post_index": post_index,
            "evidence": evidence,
            "individual_shift_world": shift,
            "accepted": accepted,
        })
    accepted_shifts = [row["individual_shift_world"] for row in records if row["accepted"]]
    group_shift = (
        np.median(np.stack(accepted_shifts), axis=0)
        if accepted_shifts else np.zeros(3, dtype=np.float64)
    )
    objectives = {}
    for residual_lambda in config.residual_lambda_grid:
        proposed = {}
        for row in records:
            if row["accepted"]:
                shift = group_shift + residual_lambda * (
                    row["individual_shift_world"] - group_shift
                )
            else:
                shift = np.zeros(3, dtype=np.float64)
            proposed[row["post_index"]] = _points(post_people[row["post_index"]]["root"]) + shift
        errors = []
        for first_index, first in enumerate(records):
            for second in records[first_index + 1:]:
                post_vector = proposed[first["post_index"]] - proposed[second["post_index"]]
                pre_vector = (
                    _points(pre_people[first["pre_index"]]["root"])
                    - _points(pre_people[second["pre_index"]]["root"])
                )
                errors.append(float(np.linalg.norm(post_vector - pre_vector)))
        objectives[float(residual_lambda)] = float(np.mean(errors)) if errors else 0.0
    selected_lambda = min(config.residual_lambda_grid, key=lambda value: objectives[float(value)])
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
    debug = {
        "camera_update": "none",
        "matched_count": len(records),
        "accepted_count": sum(row["accepted"] for row in records),
        "group_shift_world": group_shift,
        "selected_residual_lambda": float(selected_lambda),
        "observable_layout_objective_by_lambda": objectives,
        "people": records,
    }
    return corrected, debug
