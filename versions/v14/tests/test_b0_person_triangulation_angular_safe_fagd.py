from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from versions.v14.b0_person_triangulation import (
    PersonTriangulationConfig,
    refine_matched_people,
)
from versions.v14.b0_person_triangulation_angular_safe_fagd import (
    AngularSafeFAGDConfig,
    refine_matched_people_angular_safe_fagd,
)
from versions.v14.tests.test_b0_person_triangulation_strict_fagd import (
    assert_geometry_exact,
    scene,
)


def test_angular_budget_selects_largest_safe_alpha_and_preserves_residual():
    pre_camera, post_camera, pre, post = scene(2)
    matches = [(0, 0), (1, 1)]
    pre_before, post_before = pre_camera.copy(), post_camera.copy()
    base, base_debug = refine_matched_people(
        pre_camera, post_camera, pre, post, matches
    )
    candidate, debug = refine_matched_people_angular_safe_fagd(
        pre_camera,
        post_camera,
        pre,
        post,
        matches,
        angular_config=AngularSafeFAGDConfig(
            angular_budget_deg=2.0,
            statistic="all_p90",
        ),
    )
    assert debug["strict_full_one_to_one_all_accepted"] is True
    assert debug["angular_budget_satisfied"] is True
    assert debug["selected_group_alpha"] == 0.8
    assert debug["angular_score_by_alpha_deg"][0.8] <= 2.0
    assert debug["angular_score_by_alpha_deg"][0.8500000000000001] > 2.0
    group = np.asarray(base_debug["group_shift_world"])
    residual_lambda = float(base_debug["selected_residual_lambda"])
    for index, record in enumerate(base_debug["people"]):
        individual = np.asarray(record["individual_shift_world"])
        expected = 0.8 * group + residual_lambda * (individual - group)
        actual = candidate[index]["root"] - post[index]["root"]
        assert np.allclose(actual, expected, atol=1e-12)
        base_shift = base[index]["root"] - post[index]["root"]
        assert np.allclose(actual - base_shift, -0.2 * group, atol=1e-12)
    assert np.array_equal(pre_camera, pre_before)
    assert np.array_equal(post_camera, post_before)


def test_unachievable_budget_selects_minimum_observed_angle():
    pre_camera, post_camera, pre, post = scene(2)
    _, debug = refine_matched_people_angular_safe_fagd(
        pre_camera,
        post_camera,
        pre,
        post,
        [(0, 0), (1, 1)],
        angular_config=AngularSafeFAGDConfig(
            angular_budget_deg=0.01,
            statistic="all_p90",
        ),
    )
    assert debug["angular_budget_satisfied"] is False
    assert debug["selected_group_alpha"] == 0.5
    selected_score = debug["angular_score_by_alpha_deg"][0.5]
    assert selected_score == min(debug["angular_score_by_alpha_deg"].values())


def test_population_change_is_exact_frozen_v1():
    pre_camera, post_camera, pre, post = scene(3)
    matches = [(0, 0), (1, 1)]
    base, _ = refine_matched_people(
        pre_camera, post_camera, pre, post[:2], matches
    )
    candidate, debug = refine_matched_people_angular_safe_fagd(
        pre_camera,
        post_camera,
        pre,
        post[:2],
        matches,
        angular_config=AngularSafeFAGDConfig(
            angular_budget_deg=2.0,
            statistic="core_median",
        ),
    )
    assert debug["strict_full_one_to_one_all_accepted"] is False
    assert debug["exact_v1_output"] is True
    for first, second in zip(candidate, base):
        assert_geometry_exact(first, second)


def test_incomplete_equal_population_and_unmatched_person_are_exact_v1():
    pre_camera, post_camera, pre, post = scene(2)
    matches = [(0, 0)]
    base, _ = refine_matched_people(pre_camera, post_camera, pre, post, matches)
    candidate, debug = refine_matched_people_angular_safe_fagd(
        pre_camera,
        post_camera,
        pre,
        post,
        matches,
        angular_config=AngularSafeFAGDConfig(
            angular_budget_deg=2.0,
            statistic="all_median",
        ),
    )
    assert debug["matched_count"] == 1 < debug["population_max"]
    assert debug["strict_full_one_to_one_all_accepted"] is False
    for first, second in zip(candidate, base):
        assert_geometry_exact(first, second)


def test_any_rejected_evidence_is_exact_v1():
    pre_camera, post_camera, pre, post = scene(2)
    matches = [(0, 0), (1, 1)]
    config = PersonTriangulationConfig(max_median_gap_m=-1.0)
    base, _ = refine_matched_people(
        pre_camera, post_camera, pre, post, matches, config
    )
    candidate, debug = refine_matched_people_angular_safe_fagd(
        pre_camera,
        post_camera,
        pre,
        post,
        matches,
        config=config,
        angular_config=AngularSafeFAGDConfig(
            angular_budget_deg=2.0,
            statistic="core_p90",
        ),
    )
    assert debug["accepted_count"] == 0
    assert debug["strict_full_one_to_one_all_accepted"] is False
    for first, second in zip(candidate, base):
        assert_geometry_exact(first, second)
