from __future__ import annotations

import copy

import numpy as np

from versions.v14 import b0_person_triangulation_group_alpha_selector as selector


def _people():
    person = {
        "root": np.zeros(3, dtype=np.float64),
        "joints": np.zeros((5, 3), dtype=np.float64),
        "vertices": np.zeros((7, 3), dtype=np.float64),
    }
    return [copy.deepcopy(person)], [copy.deepcopy(person)]


def _fake_v1(pre_camera, post_camera, pre_people, post_people, matches, config):
    del pre_camera, post_camera, pre_people, matches, config
    group = np.asarray([1.0, 0.0, 0.0])
    individual = np.asarray([1.4, 0.2, 0.0])
    residual_lambda = 0.5
    final = group + residual_lambda * (individual - group)
    corrected = [copy.deepcopy(person) for person in post_people]
    for key in ("root", "joints", "vertices"):
        corrected[0][key] = np.asarray(corrected[0][key]) + final
    return corrected, {
        "camera_update": "none",
        "matched_count": 1,
        "accepted_count": 1,
        "group_shift_world": group,
        "selected_residual_lambda": residual_lambda,
        "observable_layout_objective_by_lambda": {
            0.0: 0.2,
            0.25: 0.15,
            0.5: 0.1,
            0.75: 0.12,
            1.0: 0.18,
        },
        "people": [
            {
                "pre_index": 0,
                "post_index": 0,
                "accepted": True,
                "individual_shift_world": individual,
                "final_shift_world": final,
                "evidence": {
                    "raw_m": 1.0,
                    "valid_count": 5,
                    "median_gap_m": 0.01,
                    "max_gap_m": 0.02,
                    "median_sine": 0.2,
                    "min_sine": 0.1,
                    "mad_m": 0.02,
                    "ray_world": np.asarray([1.0, 0.0, 0.0]),
                },
            }
        ],
    }


def _policy(intercept, threshold=0.5, lower=-1e6, upper=1e6):
    count = len(selector.FEATURE_NAMES)
    return {
        "confidence_threshold": threshold,
        "model": {
            "feature_names": list(selector.FEATURE_NAMES),
            "feature_mean": [0.0] * count,
            "feature_scale": [1.0] * count,
            "feature_lower": [lower] * count,
            "feature_upper": [upper] * count,
            "classes": list(selector.ALPHAS),
            "coefficients": [[0.0] * count for _ in selector.ALPHAS],
            "intercept": list(intercept),
        },
    }


def _run(monkeypatch, policy):
    monkeypatch.setattr(selector, "refine_matched_people", _fake_v1)
    pre, post = _people()
    corrected, debug = selector.refine_matched_people_group_alpha_selector(
        np.eye(4), np.eye(4), pre, post, [(0, 0)], policy
    )
    return post, corrected, debug


def test_alpha_one_is_exact_v1(monkeypatch):
    post, corrected, debug = _run(monkeypatch, _policy((-20.0, -20.0, 20.0)))
    expected = np.asarray([1.2, 0.1, 0.0])
    assert debug["selected_group_alpha"] == 1.0
    for key in ("root", "joints", "vertices"):
        np.testing.assert_array_equal(corrected[0][key], post[0][key] + expected)


def test_alpha_point_eight_changes_only_group_and_preserves_residual(monkeypatch):
    post, corrected, debug = _run(monkeypatch, _policy((20.0, -20.0, -20.0)))
    expected = np.asarray([1.0, 0.1, 0.0])
    assert debug["selected_group_alpha"] == 0.8
    for key in ("root", "joints", "vertices"):
        np.testing.assert_allclose(corrected[0][key], post[0][key] + expected)
    base_residual = np.asarray([1.4, 0.2, 0.0]) - np.asarray([1.0, 0.0, 0.0])
    observed_residual = (
        debug["people"][0]["final_shift_world"]
        - debug["group_shift_world"]
    ) / debug["selected_residual_lambda"]
    np.testing.assert_allclose(observed_residual, base_residual)


def test_ood_is_exact_v1(monkeypatch):
    post, corrected, debug = _run(
        monkeypatch, _policy((20.0, -20.0, -20.0), lower=0.0, upper=0.0)
    )
    assert debug["selected_group_alpha"] == 1.0
    assert debug["selector_decision"]["fallback_reason"] == (
        "feature_out_of_development_support"
    )
    expected = np.asarray([1.2, 0.1, 0.0])
    np.testing.assert_array_equal(corrected[0]["root"], post[0]["root"] + expected)


def test_low_confidence_is_exact_v1(monkeypatch):
    post, corrected, debug = _run(
        monkeypatch, _policy((0.0, 0.0, 0.0), threshold=0.9)
    )
    assert debug["selected_group_alpha"] == 1.0
    assert debug["selector_decision"]["fallback_reason"] == (
        "low_classifier_confidence"
    )
    expected = np.asarray([1.2, 0.1, 0.0])
    np.testing.assert_array_equal(corrected[0]["root"], post[0]["root"] + expected)
