"""Causal reliability-weighted Huber-IRLS person refinement after frozen B0.

The runtime contract is intentionally narrow:

* cameras are inputs and are never estimated or modified;
* only an already-matched last pre-cut person and current post-cut person are
  used to form five SMPL-X core-joint ray pairs;
* each accepted post person receives one bounded rigid world translation;
* unmatched or rejected people are copied exactly (exact-B0 fallback);
* no dataset identity, ground truth, image encoder, or pretrained model is
  referenced by this module.

Compared with the original median BRTC proposal, each ray receives an
observable geometric reliability weight from its triangulation sine and ray
gap.  A scalar Huber IRLS solve then robustly combines the per-ray depth
proposals.  The existing observable pre/post layout selector is retained for
multi-person consensus.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np


CORE5 = (0, 1, 2, 16, 17)


@dataclass(frozen=True)
class ReliabilityHuberConfig:
    joint_ids: tuple[int, ...] = CORE5
    sine_power: float = 1.0
    gap_power: float = 1.0
    gap_scale_m: float = 0.10
    huber_delta_m: float = 0.10
    max_irls_iterations: int = 25
    irls_tolerance_m: float = 1e-8
    min_valid: int = 1
    max_median_gap_m: float = 0.20
    max_weighted_mad_m: float = 0.40
    min_median_sine: float = 0.025
    min_effective_rays: float = 1.0
    min_huber_inlier_weight: float = 0.0
    min_abs_raw_m: float = 0.0
    cap_m: float = 2.0
    residual_lambda_grid: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.0)


DEFAULT_CONFIG = ReliabilityHuberConfig()


def config_dict(config: ReliabilityHuberConfig) -> dict[str, Any]:
    return asdict(config)


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
    """Return midpoint, two signed depths, ray gap, and conditioning sine."""
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


def ray_candidates(
    pre_person: dict[str, Any],
    post_person: dict[str, Any],
    pre_camera: np.ndarray,
    post_camera: np.ndarray,
    joint_ids: tuple[int, ...] = CORE5,
) -> dict[str, Any]:
    """Construct causal per-joint post-root ray-depth proposals."""
    pre_camera = np.asarray(pre_camera, dtype=np.float64)
    post_camera = np.asarray(post_camera, dtype=np.float64)
    if pre_camera.shape != (4, 4) or post_camera.shape != (4, 4):
        raise ValueError("Cameras must be 4x4 camera-to-world matrices")
    pre_joints = _points(pre_person["joints"])
    post_joints = _points(post_person["joints"])
    post_root = _points(post_person["root"])
    ray = post_root - post_camera[:3, 3]
    ray = ray / max(float(np.linalg.norm(ray)), 1e-12)
    candidates, gaps, sines, valid_joint_ids = [], [], [], []
    for joint_id in joint_ids:
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
        if not np.isfinite(delta):
            continue
        candidates.append(delta)
        gaps.append(gap)
        sines.append(sine)
        valid_joint_ids.append(int(joint_id))
    return {
        "candidate_depths_m": np.asarray(candidates, dtype=np.float64),
        "ray_gaps_m": np.asarray(gaps, dtype=np.float64),
        "conditioning_sines": np.asarray(sines, dtype=np.float64),
        "joint_ids": np.asarray(valid_joint_ids, dtype=np.int64),
        "ray_world": ray,
    }


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(values) != len(weights) or not len(values):
        raise ValueError("weighted_median expects equal non-empty arrays")
    if not (np.isfinite(values).all() and np.isfinite(weights).all()):
        raise ValueError("weighted_median inputs must be finite")
    if np.any(weights < 0.0) or float(weights.sum()) <= 0.0:
        raise ValueError("weighted_median weights must be nonnegative and nonzero")
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, 0.5 * float(weights.sum()), side="left"))
    return float(ordered_values[min(index, len(ordered_values) - 1)])


def reliability_weights(
    gaps_m: np.ndarray,
    sines: np.ndarray,
    config: ReliabilityHuberConfig,
) -> np.ndarray:
    """Observable ray reliability: high crossing angle and small ray gap."""
    gaps = np.asarray(gaps_m, dtype=np.float64).reshape(-1)
    sines = np.asarray(sines, dtype=np.float64).reshape(-1)
    if len(gaps) != len(sines) or not len(gaps):
        raise ValueError("Reliability arrays must be equal and non-empty")
    sine_weight = (
        np.ones_like(sines)
        if config.sine_power == 0.0
        else np.maximum(sines, 1e-8) ** float(config.sine_power)
    )
    if config.gap_power == 0.0:
        gap_weight = np.ones_like(gaps)
    else:
        scale = max(float(config.gap_scale_m), 1e-8)
        gap_weight = 1.0 / (
            1.0 + (np.maximum(gaps, 0.0) / scale) ** float(config.gap_power)
        )
    weights = sine_weight * gap_weight
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.full(len(weights), 1.0 / len(weights), dtype=np.float64)
    return weights / total


def huber_irls_center(
    values_m: np.ndarray,
    base_weights: np.ndarray,
    delta_m: float,
    max_iterations: int = 25,
    tolerance_m: float = 1e-8,
) -> dict[str, Any]:
    """Solve a weighted scalar Huber location problem by deterministic IRLS."""
    values = np.asarray(values_m, dtype=np.float64).reshape(-1)
    weights = np.asarray(base_weights, dtype=np.float64).reshape(-1)
    if len(values) != len(weights) or not len(values):
        raise ValueError("IRLS expects equal non-empty value and weight arrays")
    weights = weights / max(float(weights.sum()), 1e-12)
    center = weighted_median(values, weights)
    iterations = 0
    final_weights = weights.copy()
    delta = float(delta_m)
    for iterations in range(1, max(1, int(max_iterations)) + 1):
        residual = np.abs(values - center)
        if np.isinf(delta):
            robust = np.ones_like(residual)
        else:
            robust = np.minimum(1.0, max(delta, 1e-12) / np.maximum(residual, 1e-12))
        combined = weights * robust
        total = float(combined.sum())
        if not np.isfinite(total) or total <= 0.0:
            combined = weights.copy()
            total = float(combined.sum())
        new_center = float(np.dot(combined, values) / total)
        final_weights = combined / total
        if abs(new_center - center) <= float(tolerance_m):
            center = new_center
            break
        center = new_center
    residual = np.abs(values - center)
    inlier_weight = (
        1.0
        if np.isinf(delta)
        else float(weights[residual <= max(delta, 1e-12)].sum())
    )
    return {
        "center_m": center,
        "base_weights": weights,
        "final_weights": final_weights,
        "iterations": iterations,
        "huber_inlier_weight": inlier_weight,
    }


def aggregate_candidates(
    candidates: dict[str, Any],
    config: ReliabilityHuberConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    values = np.asarray(candidates["candidate_depths_m"], dtype=np.float64)
    gaps = np.asarray(candidates["ray_gaps_m"], dtype=np.float64)
    sines = np.asarray(candidates["conditioning_sines"], dtype=np.float64)
    if not len(values):
        return {
            "raw_m": float("nan"),
            "valid_count": 0,
            "median_gap_m": float("inf"),
            "max_gap_m": float("inf"),
            "median_sine": 0.0,
            "min_sine": 0.0,
            "weighted_mad_m": float("inf"),
            "unweighted_mad_m": float("inf"),
            "effective_rays": 0.0,
            "huber_inlier_weight": 0.0,
            "base_weights": np.empty(0, dtype=np.float64),
            "final_weights": np.empty(0, dtype=np.float64),
            "irls_iterations": 0,
            "ray_world": np.asarray(candidates["ray_world"], dtype=np.float64),
        }
    weights = reliability_weights(gaps, sines, config)
    fit = huber_irls_center(
        values,
        weights,
        float(config.huber_delta_m),
        int(config.max_irls_iterations),
        float(config.irls_tolerance_m),
    )
    center = float(fit["center_m"])
    absolute_residual = np.abs(values - center)
    effective = float(1.0 / max(float(np.sum(weights * weights)), 1e-12))
    return {
        "raw_m": center,
        "valid_count": len(values),
        "median_gap_m": float(np.median(gaps)),
        "max_gap_m": float(np.max(gaps)),
        "median_sine": float(np.median(sines)),
        "min_sine": float(np.min(sines)),
        "weighted_mad_m": weighted_median(absolute_residual, weights),
        "unweighted_mad_m": float(np.median(absolute_residual)),
        "effective_rays": effective,
        "huber_inlier_weight": float(fit["huber_inlier_weight"]),
        "base_weights": np.asarray(fit["base_weights"], dtype=np.float64),
        "final_weights": np.asarray(fit["final_weights"], dtype=np.float64),
        "irls_iterations": int(fit["iterations"]),
        "ray_world": np.asarray(candidates["ray_world"], dtype=np.float64),
    }


def accepted_shift(
    evidence: dict[str, Any],
    config: ReliabilityHuberConfig = DEFAULT_CONFIG,
) -> tuple[np.ndarray, bool]:
    raw = float(evidence["raw_m"])
    accepted = bool(
        np.isfinite(raw)
        and int(evidence["valid_count"]) >= int(config.min_valid)
        and float(evidence["median_gap_m"]) <= float(config.max_median_gap_m)
        and float(evidence["weighted_mad_m"]) <= float(config.max_weighted_mad_m)
        and float(evidence["median_sine"]) >= float(config.min_median_sine)
        and float(evidence["effective_rays"]) >= float(config.min_effective_rays)
        and float(evidence["huber_inlier_weight"])
        >= float(config.min_huber_inlier_weight)
        and abs(raw) >= float(config.min_abs_raw_m)
    )
    action = float(np.clip(raw, -config.cap_m, config.cap_m)) if accepted else 0.0
    return action * _points(evidence["ray_world"]), accepted


def person_evidence(
    pre_person: dict[str, Any],
    post_person: dict[str, Any],
    pre_camera: np.ndarray,
    post_camera: np.ndarray,
    config: ReliabilityHuberConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    candidates = ray_candidates(
        pre_person, post_person, pre_camera, post_camera, config.joint_ids
    )
    evidence = aggregate_candidates(candidates, config)
    evidence["candidate_depths_m"] = candidates["candidate_depths_m"]
    evidence["ray_gaps_m"] = candidates["ray_gaps_m"]
    evidence["conditioning_sines"] = candidates["conditioning_sines"]
    evidence["joint_ids"] = candidates["joint_ids"]
    return evidence


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
    config: ReliabilityHuberConfig = DEFAULT_CONFIG,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Translate accepted matched post people; preserve all fallbacks exactly."""
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
            pre_people[pre_index],
            post_people[post_index],
            pre_camera,
            post_camera,
            config,
        )
        shift, accepted = accepted_shift(evidence, config)
        records.append(
            {
                "pre_index": pre_index,
                "post_index": post_index,
                "evidence": evidence,
                "individual_shift_world": shift,
                "accepted": accepted,
            }
        )
    accepted_shifts = [row["individual_shift_world"] for row in records if row["accepted"]]
    group_shift = (
        np.median(np.stack(accepted_shifts), axis=0)
        if accepted_shifts
        else np.zeros(3, dtype=np.float64)
    )
    objectives = {}
    for residual_lambda in config.residual_lambda_grid:
        proposed = {}
        for row in records:
            shift = (
                group_shift
                + residual_lambda * (row["individual_shift_world"] - group_shift)
                if row["accepted"]
                else np.zeros(3, dtype=np.float64)
            )
            proposed[row["post_index"]] = (
                _points(post_people[row["post_index"]]["root"]) + shift
            )
        errors = []
        for first_index, first in enumerate(records):
            for second in records[first_index + 1 :]:
                post_vector = (
                    proposed[first["post_index"]] - proposed[second["post_index"]]
                )
                pre_vector = (
                    _points(pre_people[first["pre_index"]]["root"])
                    - _points(pre_people[second["pre_index"]]["root"])
                )
                errors.append(float(np.linalg.norm(post_vector - pre_vector)))
        objectives[float(residual_lambda)] = (
            float(np.mean(errors)) if errors else 0.0
        )
    selected_lambda = min(
        config.residual_lambda_grid, key=lambda value: objectives[float(value)]
    )
    corrected = [_copy_person(person) for person in post_people]
    fallback_max_abs_change = 0.0
    for row in records:
        shift = (
            group_shift
            + selected_lambda * (row["individual_shift_world"] - group_shift)
            if row["accepted"]
            else np.zeros(3, dtype=np.float64)
        )
        row["final_shift_world"] = shift
        for key in ("root", "joints", "vertices"):
            if key in corrected[row["post_index"]]:
                before = corrected[row["post_index"]][key].copy()
                corrected[row["post_index"]][key] += shift
                if not row["accepted"]:
                    fallback_max_abs_change = max(
                        fallback_max_abs_change,
                        float(np.max(np.abs(corrected[row["post_index"]][key] - before))),
                    )
    debug = {
        "camera_update": "none",
        "matched_count": len(records),
        "accepted_count": sum(row["accepted"] for row in records),
        "group_shift_world": group_shift,
        "selected_residual_lambda": float(selected_lambda),
        "observable_layout_objective_by_lambda": objectives,
        "exact_b0_fallback_max_abs_change": fallback_max_abs_change,
        "people": records,
    }
    return corrected, debug
