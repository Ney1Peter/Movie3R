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
from versions.v14.b0_person_triangulation_strict_fagd import (
    StrictFAGDConfig,
    refine_matched_people_strict_fagd,
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
    return {"root": root.copy(), "joints": joints, "vertices": joints.copy()}


def project_person(true_person, center, wrong_depth_scale=1.0):
    center = np.asarray(center, dtype=np.float64)
    return {
        key: center + wrong_depth_scale * (true_person[key] - center)
        for key in ("root", "joints", "vertices")
    }


def scene(count=2):
    pre_camera, post_camera = camera([0, 0, 0]), camera([1, 0, 0])
    truths = [person([0.1 + 0.55 * index, 0.1 * index, 3.0 + 0.2 * index]) for index in range(count)]
    pre = [project_person(item, pre_camera[:3, 3], 1.0) for item in truths]
    post = [project_person(item, post_camera[:3, 3], 1.35 + 0.05 * index) for index, item in enumerate(truths)]
    return pre_camera, post_camera, pre, post


def assert_geometry_exact(first, second):
    for key in ("root", "joints", "vertices"):
        assert np.array_equal(first[key], second[key])


def test_strict_full_all_accepted_scales_group_only_and_preserves_residual():
    pre_camera, post_camera, pre, post = scene(2)
    matches = [(0, 0), (1, 1)]
    base, base_debug = refine_matched_people(pre_camera, post_camera, pre, post, matches)
    candidate, debug = refine_matched_people_strict_fagd(
        pre_camera, post_camera, pre, post, matches
    )
    group = np.asarray(base_debug["group_shift_world"])
    residual_lambda = float(base_debug["selected_residual_lambda"])
    assert debug["strict_full_one_to_one_all_accepted"] is True
    assert debug["group_only_alpha"] == 0.9
    for index, record in enumerate(base_debug["people"]):
        individual = np.asarray(record["individual_shift_world"])
        expected = 0.9 * group + residual_lambda * (individual - group)
        actual = candidate[index]["root"] - post[index]["root"]
        assert np.allclose(actual, expected, atol=1e-12)
        base_shift = base[index]["root"] - post[index]["root"]
        assert np.allclose(actual - base_shift, -0.1 * group, atol=1e-12)


def test_population_change_is_bit_exact_frozen_v1_fallback():
    pre_camera, post_camera, pre, post = scene(3)
    base, _ = refine_matched_people(
        pre_camera, post_camera, pre, post[:2], [(0, 0), (1, 1)]
    )
    candidate, debug = refine_matched_people_strict_fagd(
        pre_camera, post_camera, pre, post[:2], [(0, 0), (1, 1)]
    )
    assert debug["strict_full_one_to_one_all_accepted"] is False
    assert debug["exact_v1_fallback"] is True
    for first, second in zip(candidate, base):
        assert_geometry_exact(first, second)


def test_incomplete_matching_with_equal_populations_is_bit_exact_v1():
    pre_camera, post_camera, pre, post = scene(2)
    matches = [(0, 0)]
    base, _ = refine_matched_people(pre_camera, post_camera, pre, post, matches)
    candidate, debug = refine_matched_people_strict_fagd(
        pre_camera, post_camera, pre, post, matches
    )
    assert debug["matched_count"] == 1 < debug["population_max"]
    assert debug["exact_v1_fallback"] is True
    for first, second in zip(candidate, base):
        assert_geometry_exact(first, second)


def test_any_rejection_forces_bit_exact_v1_and_camera_is_unchanged():
    pre_camera, post_camera, pre, post = scene(2)
    matches = [(0, 0), (1, 1)]
    config = PersonTriangulationConfig(max_median_gap_m=-1.0)
    pre_before, post_before = pre_camera.copy(), post_camera.copy()
    base, _ = refine_matched_people(pre_camera, post_camera, pre, post, matches, config)
    candidate, debug = refine_matched_people_strict_fagd(
        pre_camera,
        post_camera,
        pre,
        post,
        matches,
        config=config,
        fagd_config=StrictFAGDConfig(alpha=0.9),
    )
    assert debug["accepted_count"] == 0
    assert debug["exact_v1_fallback"] is True
    for first, second in zip(candidate, base):
        assert_geometry_exact(first, second)
    assert np.array_equal(pre_camera, pre_before)
    assert np.array_equal(post_camera, post_before)
