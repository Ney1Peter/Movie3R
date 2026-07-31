from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from versions.v14.b0_person_triangulation import (
    PersonTriangulationConfig,
    refine_matched_people,
)
from versions.v14.b0_person_triangulation_orientation_kabsch import (
    kabsch_rotation,
    refine_matched_people_orientation_kabsch,
)
from versions.v14.tests.test_b0_person_triangulation_strict_fagd import (
    assert_geometry_exact,
    scene,
)


def rotated_scene(count: int = 2, angle_deg: float = 20.0):
    pre_camera, post_camera, pre, post = scene(count)
    rotation = Rotation.from_euler("z", angle_deg, degrees=True).as_matrix()
    for pre_person, post_person in zip(pre, post):
        pre_person["torso"] = np.eye(3)
        pre_person["root_rotation"] = np.eye(3)
        post_person["torso"] = np.eye(3)
        post_person["root_rotation"] = np.eye(3)
        for key in ("joints", "vertices"):
            post_person[key] = (
                pre_person[key] - pre_person["root"]
            ) @ rotation.T + post_person["root"]
        post_person["torso"] = rotation @ post_person["torso"]
        post_person["root_rotation"] = rotation @ post_person["root_rotation"]
    return pre_camera, post_camera, pre, post


def test_kabsch_row_vector_direction():
    generator = np.random.default_rng(4)
    source = generator.normal(size=(20, 3))
    expected = Rotation.from_euler("xyz", [20.0, -10.0, 15.0], degrees=True).as_matrix()
    target = source @ expected.T
    actual = kabsch_rotation(source, target)
    assert np.allclose(actual, expected, atol=1e-12)
    assert np.allclose(source @ actual.T, target, atol=1e-12)


def test_accepted_people_rotate_but_roots_and_cameras_stay_exact():
    pre_camera, post_camera, pre, post = rotated_scene()
    pre_camera_before, post_camera_before = pre_camera.copy(), post_camera.copy()
    matches = [(0, 0), (1, 1)]
    base, _ = refine_matched_people(pre_camera, post_camera, pre, post, matches)
    candidate, debug = refine_matched_people_orientation_kabsch(
        pre_camera, post_camera, pre, post, matches
    )
    assert debug["orientation_applied_count"] == 2
    for index, record in enumerate(debug["people"]):
        assert record["orientation"]["applied"] is True
        assert np.isclose(record["orientation"]["raw_angle_deg"], 20.0, atol=1e-10)
        assert np.isclose(record["orientation"]["applied_angle_deg"], 10.0, atol=1e-10)
        assert np.array_equal(candidate[index]["root"], base[index]["root"])
        assert not np.array_equal(candidate[index]["joints"], base[index]["joints"])
    assert np.array_equal(pre_camera, pre_camera_before)
    assert np.array_equal(post_camera, post_camera_before)


def test_rejected_people_are_exact_b0():
    pre_camera, post_camera, pre, post = rotated_scene()
    matches = [(0, 0), (1, 1)]
    config = PersonTriangulationConfig(max_median_gap_m=-1.0)
    base, _ = refine_matched_people(
        pre_camera, post_camera, pre, post, matches, config
    )
    candidate, debug = refine_matched_people_orientation_kabsch(
        pre_camera, post_camera, pre, post, matches, config=config
    )
    assert debug["accepted_count"] == 0
    assert debug["orientation_applied_count"] == 0
    for index, (actual, expected) in enumerate(zip(candidate, base)):
        assert_geometry_exact(actual, expected)
        assert_geometry_exact(actual, post[index])


def test_unmatched_people_are_exact_b0():
    pre_camera, post_camera, pre, post = rotated_scene()
    base, _ = refine_matched_people(pre_camera, post_camera, pre, post, [(0, 0)])
    candidate, debug = refine_matched_people_orientation_kabsch(
        pre_camera, post_camera, pre, post, [(0, 0)]
    )
    assert debug["orientation_applied_count"] == 1
    assert_geometry_exact(candidate[1], base[1])
    assert_geometry_exact(candidate[1], post[1])
