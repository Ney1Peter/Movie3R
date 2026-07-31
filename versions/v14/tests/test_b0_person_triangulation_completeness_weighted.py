from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from versions.v14.b0_person_triangulation import (
    PersonTriangulationConfig,
    refine_matched_people,
)
from versions.v14.b0_person_triangulation_completeness_weighted import (
    association_completeness,
    refine_matched_people_completeness_weighted,
)


def camera(center):
    value = np.eye(4, dtype=np.float64)
    value[:3, 3] = np.asarray(center, dtype=np.float64)
    return value


def person(root, joint_count=20):
    root = np.asarray(root, dtype=np.float64)
    offsets = np.zeros((joint_count, 3), dtype=np.float64)
    offsets[:, 1] = np.linspace(-0.4, 0.4, joint_count)
    joints = root + offsets
    return {
        "root": root.copy(),
        "joints": joints,
        "vertices": joints.copy(),
        "opaque_metadata": {"must": "survive"},
    }


def project_person(true_person, center, wrong_depth_scale=1.0):
    center = np.asarray(center, dtype=np.float64)
    output = dict(true_person)
    for key in ("root", "joints", "vertices"):
        rays = true_person[key] - center
        output[key] = center + wrong_depth_scale * rays
    return output


def assert_person_geometry_equal(first, second):
    for key in ("root", "joints", "vertices"):
        assert np.array_equal(first[key], second[key])


def scene_with_accepted_people(count):
    pre_camera = camera([0.0, 0.0, 0.0])
    post_camera = camera([1.0, 0.0, 0.0])
    truths = [person([0.1 + 0.45 * index, 0.0, 3.0]) for index in range(count)]
    pre_people = [project_person(item, pre_camera[:3, 3], 1.0) for item in truths]
    post_people = [project_person(item, post_camera[:3, 3], 1.35) for item in truths]
    return pre_camera, post_camera, pre_people, post_people


def test_completeness_formula_and_invalid_counts():
    assert association_completeness(3, 3, 3) == 1.0
    assert association_completeness(3, 2, 2) == pytest.approx(2.0 / 3.0)
    assert association_completeness(2, 3, 2) == pytest.approx(2.0 / 3.0)
    assert association_completeness(0, 0, 0) == 0.0
    with pytest.raises(ValueError):
        association_completeness(1, 1, 2)
    with pytest.raises(ValueError):
        association_completeness(-1, 0, 0)


def test_camera_is_bit_exact_and_full_match_is_frozen_brtc_equivalent():
    pre_camera, post_camera, pre_people, post_people = scene_with_accepted_people(2)
    matches = [(0, 0), (1, 1)]
    pre_camera_before = pre_camera.copy()
    post_camera_before = post_camera.copy()

    baseline, baseline_debug = refine_matched_people(
        pre_camera, post_camera, pre_people, post_people, matches
    )
    weighted, debug = refine_matched_people_completeness_weighted(
        pre_camera, post_camera, pre_people, post_people, matches
    )

    assert np.array_equal(pre_camera, pre_camera_before)
    assert np.array_equal(post_camera, post_camera_before)
    assert debug["camera_update"] == "none"
    assert debug["completeness"] == 1.0
    assert debug["action_scale"] == 1.0
    assert debug["accepted_count"] == baseline_debug["accepted_count"] == 2
    for weighted_person, baseline_person in zip(weighted, baseline):
        assert_person_geometry_equal(weighted_person, baseline_person)


@pytest.mark.parametrize(
    ("previous_count", "current_count", "matches", "expected_scale"),
    [
        (3, 2, [(0, 0), (1, 1)], 2.0 / 3.0),
        (2, 3, [(0, 0), (1, 1)], 2.0 / 3.0),
        (3, 3, [(0, 0), (1, 1)], 2.0 / 3.0),
    ],
)
def test_population_change_or_incomplete_matching_scales_frozen_brtc_action(
    previous_count,
    current_count,
    matches,
    expected_scale,
):
    maximum_count = max(previous_count, current_count)
    pre_camera, post_camera, all_pre, all_post = scene_with_accepted_people(maximum_count)
    pre_people = all_pre[:previous_count]
    post_people = all_post[:current_count]

    baseline, _ = refine_matched_people(
        pre_camera, post_camera, pre_people, post_people, matches
    )
    weighted, debug = refine_matched_people_completeness_weighted(
        pre_camera, post_camera, pre_people, post_people, matches
    )

    assert debug["completeness"] == pytest.approx(expected_scale)
    assert debug["action_scale"] == pytest.approx(expected_scale)
    for _, post_index in matches:
        baseline_shift = baseline[post_index]["root"] - post_people[post_index]["root"]
        weighted_shift = weighted[post_index]["root"] - post_people[post_index]["root"]
        assert np.allclose(weighted_shift, expected_scale * baseline_shift, atol=1e-12)


def test_unmatched_and_rejected_people_are_exact_b0_fallbacks():
    pre_camera, post_camera, pre_people, post_people = scene_with_accepted_people(3)
    matches = [(0, 0), (1, 1)]
    weighted, debug = refine_matched_people_completeness_weighted(
        pre_camera,
        post_camera,
        pre_people,
        post_people,
        matches,
        config=PersonTriangulationConfig(max_median_gap_m=-1.0),
    )

    assert debug["accepted_count"] == 0
    assert debug["completeness"] == pytest.approx(2.0 / 3.0)
    for weighted_person, original_person in zip(weighted, post_people):
        assert_person_geometry_equal(weighted_person, original_person)
        assert weighted_person["opaque_metadata"] is original_person["opaque_metadata"]


def test_accepted_match_is_scaled_while_unmatched_post_is_exact_b0():
    pre_camera, post_camera, pre_people, post_people = scene_with_accepted_people(3)
    matches = [(0, 0), (1, 1)]
    baseline, _ = refine_matched_people(
        pre_camera, post_camera, pre_people[:2], post_people, matches
    )
    weighted, debug = refine_matched_people_completeness_weighted(
        pre_camera, post_camera, pre_people[:2], post_people, matches
    )

    assert debug["accepted_count"] == 2
    assert debug["action_scale"] == pytest.approx(2.0 / 3.0)
    for post_index in (0, 1):
        base_shift = baseline[post_index]["root"] - post_people[post_index]["root"]
        actual_shift = weighted[post_index]["root"] - post_people[post_index]["root"]
        assert np.allclose(actual_shift, (2.0 / 3.0) * base_shift, atol=1e-12)
    assert_person_geometry_equal(weighted[2], post_people[2])


def test_empty_boundary_is_a_strict_noop():
    pre_camera, post_camera = camera([0, 0, 0]), camera([1, 0, 0])
    corrected, debug = refine_matched_people_completeness_weighted(
        pre_camera, post_camera, [], [], []
    )
    assert corrected == []
    assert debug["completeness"] == 0.0
    assert debug["action_scale"] == 0.0
    assert debug["matched_count"] == debug["accepted_count"] == 0
