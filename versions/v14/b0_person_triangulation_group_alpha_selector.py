"""Lightweight causal selector for the shared BRTC group translation.

The frozen BRTC-LC v1 runtime remains the only source of ray proposals,
acceptance, the shared group shift, the residual lambda, and person-specific
residuals.  This module extracts a small set of aggregate observable features
from v1 debug values and applies a frozen multinomial linear classifier to
choose ``alpha`` from ``{0.8, 0.9, 1.0}``::

    final_i = alpha * group + lambda * (individual_i - group)

Only the shared group term changes.  A feature outside the frozen development
support, a low classifier probability, an invalid policy, or any prediction
error falls back to exact v1 (``alpha=1``).  Cameras, images, GT, future frames,
identity names, and additional pretrained models are never consumed.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from versions.v14.b0_person_triangulation import (
    DEFAULT_CONFIG,
    PersonTriangulationConfig,
    refine_matched_people,
)


ALPHAS = (0.8, 0.9, 1.0)
FEATURE_NAMES = (
    "group_norm_m",
    "accepted_count",
    "matched_count",
    "pre_count",
    "post_count",
    "acceptance_ratio",
    "raw_mean_m",
    "raw_std_m",
    "raw_abs_mean_m",
    "raw_abs_max_m",
    "raw_abs_min_m",
    "raw_sign_balance",
    "median_gap_mean_m",
    "median_gap_max_m",
    "mad_mean_m",
    "mad_max_m",
    "median_sine_mean",
    "median_sine_min",
    "valid_count_mean",
    "selected_residual_lambda",
    "layout_selected_objective_m",
    "layout_objective_range_m",
    "layout_objective_slope_m",
    "individual_residual_norm_mean_m",
    "individual_residual_norm_max_m",
)


def _finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    return array[np.isfinite(array)]


def _mean(values: Iterable[float]) -> float:
    array = _finite(values)
    return float(array.mean()) if len(array) else 0.0


def _max(values: Iterable[float]) -> float:
    array = _finite(values)
    return float(array.max()) if len(array) else 0.0


def _min(values: Iterable[float]) -> float:
    array = _finite(values)
    return float(array.min()) if len(array) else 0.0


def observable_feature_dict(
    base_debug: dict[str, Any], pre_count: int, post_count: int
) -> dict[str, float]:
    """Aggregate strictly observable v1 evidence without person identities."""

    people = list(base_debug.get("people", ()))
    accepted = [row for row in people if bool(row.get("accepted", False))]
    evidence = [row.get("evidence", {}) for row in people]
    raw = _finite(float(row.get("raw_m", float("nan"))) for row in evidence)
    absolute_raw = np.abs(raw)
    gaps = [float(row.get("median_gap_m", float("nan"))) for row in evidence]
    mads = [float(row.get("mad_m", float("nan"))) for row in evidence]
    sines = [float(row.get("median_sine", float("nan"))) for row in evidence]
    valid = [float(row.get("valid_count", 0.0)) for row in evidence]
    group = np.asarray(base_debug.get("group_shift_world", np.zeros(3)), dtype=np.float64)
    residuals = [
        float(
            np.linalg.norm(
                np.asarray(row.get("individual_shift_world", group), dtype=np.float64)
                - group
            )
        )
        for row in accepted
    ]
    objectives = {
        float(key): float(value)
        for key, value in base_debug.get(
            "observable_layout_objective_by_lambda", {}
        ).items()
        if np.isfinite(float(value))
    }
    selected_lambda = float(base_debug.get("selected_residual_lambda", 0.0))
    selected_objective = float(objectives.get(selected_lambda, 0.0))
    objective_values = list(objectives.values())
    matched_count = int(base_debug.get("matched_count", len(people)))
    accepted_count = int(base_debug.get("accepted_count", len(accepted)))
    output = {
        "group_norm_m": float(np.linalg.norm(group)),
        "accepted_count": float(accepted_count),
        "matched_count": float(matched_count),
        "pre_count": float(pre_count),
        "post_count": float(post_count),
        "acceptance_ratio": float(accepted_count / max(matched_count, 1)),
        "raw_mean_m": float(raw.mean()) if len(raw) else 0.0,
        "raw_std_m": float(raw.std()) if len(raw) else 0.0,
        "raw_abs_mean_m": float(absolute_raw.mean()) if len(raw) else 0.0,
        "raw_abs_max_m": float(absolute_raw.max()) if len(raw) else 0.0,
        "raw_abs_min_m": float(absolute_raw.min()) if len(raw) else 0.0,
        "raw_sign_balance": float(np.sign(raw).mean()) if len(raw) else 0.0,
        "median_gap_mean_m": _mean(gaps),
        "median_gap_max_m": _max(gaps),
        "mad_mean_m": _mean(mads),
        "mad_max_m": _max(mads),
        "median_sine_mean": _mean(sines),
        "median_sine_min": _min(sines),
        "valid_count_mean": _mean(valid),
        "selected_residual_lambda": selected_lambda,
        "layout_selected_objective_m": selected_objective,
        "layout_objective_range_m": (
            float(max(objective_values) - min(objective_values))
            if objective_values
            else 0.0
        ),
        "layout_objective_slope_m": float(
            objectives.get(1.0, selected_objective)
            - objectives.get(0.0, selected_objective)
        ),
        "individual_residual_norm_mean_m": _mean(residuals),
        "individual_residual_norm_max_m": _max(residuals),
    }
    if tuple(output) != FEATURE_NAMES:
        raise RuntimeError("Observable group-alpha feature schema changed")
    if not np.isfinite(np.asarray(list(output.values()), dtype=np.float64)).all():
        raise ValueError("Non-finite group-alpha observable feature")
    return output


def feature_vector(features: dict[str, float]) -> np.ndarray:
    return np.asarray([features[name] for name in FEATURE_NAMES], dtype=np.float64)


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - float(np.max(logits))
    values = np.exp(shifted)
    return values / max(float(values.sum()), 1e-12)


def select_group_alpha(
    features: dict[str, float], frozen_policy: dict[str, Any]
) -> dict[str, Any]:
    """Select an alpha or conservatively return exact-v1 fallback."""

    try:
        model = frozen_policy["model"]
        if tuple(model["feature_names"]) != FEATURE_NAMES:
            raise ValueError("feature_schema_mismatch")
        vector = feature_vector(features)
        lower = np.asarray(model["feature_lower"], dtype=np.float64)
        upper = np.asarray(model["feature_upper"], dtype=np.float64)
        if not (
            vector.shape == lower.shape == upper.shape
            and np.isfinite(vector).all()
            and np.isfinite(lower).all()
            and np.isfinite(upper).all()
        ):
            raise ValueError("invalid_feature_support")
        outside = (vector < lower) | (vector > upper)
        if bool(np.any(outside)):
            return {
                "selected_alpha": 1.0,
                "fallback": True,
                "fallback_reason": "feature_out_of_development_support",
                "outside_feature_names": [
                    FEATURE_NAMES[index] for index in np.flatnonzero(outside)
                ],
                "confidence": 0.0,
                "probability_by_alpha": {},
            }
        mean = np.asarray(model["feature_mean"], dtype=np.float64)
        scale = np.asarray(model["feature_scale"], dtype=np.float64)
        coefficients = np.asarray(model["coefficients"], dtype=np.float64)
        intercept = np.asarray(model["intercept"], dtype=np.float64)
        classes = np.asarray(model["classes"], dtype=np.float64)
        if not (
            mean.shape == scale.shape == vector.shape
            and coefficients.ndim == 2
            and coefficients.shape == (len(classes), len(vector))
            and intercept.shape == classes.shape
            and len(classes) == len(ALPHAS)
            and np.isfinite(mean).all()
            and np.isfinite(scale).all()
            and np.isfinite(coefficients).all()
            and np.isfinite(intercept).all()
            and np.isfinite(classes).all()
        ):
            raise ValueError("invalid_classifier_shape_or_value")
        if any(
            min(abs(float(value) - alpha) for alpha in ALPHAS) > 1e-8
            for value in classes
        ):
            raise ValueError("invalid_alpha_classes")
        normalized = (vector - mean) / np.maximum(scale, 1e-12)
        logits = coefficients @ normalized + intercept
        probabilities = _softmax(logits)
        index = int(np.argmax(probabilities))
        confidence = float(probabilities[index])
        threshold = float(frozen_policy["confidence_threshold"])
        probability_by_alpha = {
            str(float(alpha)): float(probability)
            for alpha, probability in zip(classes, probabilities)
        }
        if confidence < threshold:
            return {
                "selected_alpha": 1.0,
                "fallback": True,
                "fallback_reason": "low_classifier_confidence",
                "outside_feature_names": [],
                "confidence": confidence,
                "probability_by_alpha": probability_by_alpha,
            }
        selected = float(classes[index])
        if min(abs(selected - alpha) for alpha in ALPHAS) > 1e-8:
            raise ValueError("invalid_alpha_class")
        return {
            "selected_alpha": selected,
            "fallback": bool(abs(selected - 1.0) <= 1e-12),
            "fallback_reason": (
                "classifier_selected_exact_v1" if abs(selected - 1.0) <= 1e-12 else None
            ),
            "outside_feature_names": [],
            "confidence": confidence,
            "probability_by_alpha": probability_by_alpha,
        }
    except Exception as error:  # exact-v1 is the deployment safety contract
        return {
            "selected_alpha": 1.0,
            "fallback": True,
            "fallback_reason": f"policy_error:{type(error).__name__}:{error}",
            "outside_feature_names": [],
            "confidence": 0.0,
            "probability_by_alpha": {},
        }


def refine_matched_people_group_alpha_selector(
    pre_camera: np.ndarray,
    post_camera: np.ndarray,
    pre_people: list[dict[str, Any]],
    post_people: list[dict[str, Any]],
    matches: Iterable[tuple[int, int]],
    frozen_policy: dict[str, Any],
    config: PersonTriangulationConfig = DEFAULT_CONFIG,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run v1 and modify only its shared group component when selected."""

    materialized = tuple((int(first), int(second)) for first, second in matches)
    corrected, base_debug = refine_matched_people(
        pre_camera, post_camera, pre_people, post_people, materialized, config
    )
    features = observable_feature_dict(base_debug, len(pre_people), len(post_people))
    decision = select_group_alpha(features, frozen_policy)
    alpha = float(decision["selected_alpha"])
    group = np.asarray(base_debug["group_shift_world"], dtype=np.float64)
    residual_lambda = float(base_debug["selected_residual_lambda"])
    records = []
    for base_record in base_debug["people"]:
        record = dict(base_record)
        base_final = np.asarray(base_record["final_shift_world"], dtype=np.float64)
        if bool(base_record["accepted"]) and abs(alpha - 1.0) > 1e-12:
            individual = np.asarray(
                base_record["individual_shift_world"], dtype=np.float64
            )
            new_shift = alpha * group + residual_lambda * (individual - group)
            post_index = int(base_record["post_index"])
            for key in ("root", "joints", "vertices"):
                if key in corrected[post_index]:
                    corrected[post_index][key] = (
                        np.asarray(post_people[post_index][key], dtype=np.float64)
                        + new_shift
                    )
        else:
            new_shift = base_final
        record["base_final_shift_world"] = base_final
        record["final_shift_world"] = new_shift
        record["selected_group_alpha"] = alpha
        records.append(record)
    debug = dict(base_debug)
    debug.update(
        {
            "camera_update": "none",
            "group_alpha_selector": True,
            "observable_features": features,
            "selector_decision": decision,
            "selected_group_alpha": alpha,
            "base_group_shift_world": group,
            "group_shift_world": alpha * group,
            "individual_residual_unchanged": True,
            "people": records,
        }
    )
    return corrected, debug
