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
from versions.v14.b0_person_triangulation_shared_orientation_kabsch import (
    refine_matched_people_shared_orientation_kabsch,
)
from versions.v14.tests.test_b0_person_triangulation_strict_fagd import (
    assert_geometry_exact,
    scene,
)


def rotated_scene(count=2, angle_deg=20.0):
    pre_camera, post_camera, pre, post = scene(count)
    rotation = Rotation.from_euler("z", angle_deg, degrees=True).as_matrix()
    for pre_person, post_person in zip(pre, post):
        for key in ("joints", "vertices"):
            post_person[key] = (
                pre_person[key] - pre_person["root"]
            ) @ rotation.T + post_person["root"]
    return pre_camera, post_camera, pre, post


def test_shared_rotation_is_common_bounded_and_roots_stay_exact_brtc():
    pre_camera, post_camera, pre, post = rotated_scene(2)
    matches = [(0, 0), (1, 1)]
    base, _ = refine_matched_people(pre_camera, post_camera, pre, post, matches)
    candidate, debug = refine_matched_people_shared_orientation_kabsch(
        pre_camera, post_camera, pre, post, matches
    )
    assert debug["accepted_count"] == 2
    assert debug["shared_orientation_applied"] is True
    assert np.isclose(debug["shared_raw_angle_deg"], 20.0, atol=1e-10)
    assert np.isclose(debug["shared_applied_angle_deg"], 10.0, atol=1e-10)
    assert debug["shared_candidate_torso_residual_m"] < debug["shared_raw_torso_residual_m"]
    rotations = []
    for index, record in enumerate(debug["people"]):
        assert np.array_equal(candidate[index]["root"], base[index]["root"])
        assert record["shared_orientation_applied"] is True
        rotations.append(record["shared_rotation_world"])
    assert np.array_equal(rotations[0], rotations[1])


def test_rejected_people_are_exact_b0_and_camera_is_unchanged():
    pre_camera, post_camera, pre, post = rotated_scene(2)
    matches = [(0, 0), (1, 1)]
    config = PersonTriangulationConfig(max_median_gap_m=-1.0)
    pre_before, post_before = pre_camera.copy(), post_camera.copy()
    base, _ = refine_matched_people(
        pre_camera, post_camera, pre, post, matches, config
    )
    candidate, debug = refine_matched_people_shared_orientation_kabsch(
        pre_camera, post_camera, pre, post, matches, config=config
    )
    assert debug["accepted_count"] == 0
    assert debug["shared_orientation_applied"] is False
    for index, (first, second) in enumerate(zip(candidate, base)):
        assert_geometry_exact(first, second)
        assert_geometry_exact(first, post[index])
    assert np.array_equal(pre_camera, pre_before)
    assert np.array_equal(post_camera, post_before)


def test_unmatched_person_is_exact_b0_while_matched_person_can_rotate():
    pre_camera, post_camera, pre, post = rotated_scene(2)
    base, _ = refine_matched_people(
        pre_camera, post_camera, pre, post, [(0, 0)]
    )
    candidate, debug = refine_matched_people_shared_orientation_kabsch(
        pre_camera, post_camera, pre, post, [(0, 0)]
    )
    assert debug["accepted_orientation_person_count"] == 1
    assert np.array_equal(candidate[0]["root"], base[0]["root"])
    assert not np.array_equal(candidate[0]["joints"], base[0]["joints"])
    assert_geometry_exact(candidate[1], base[1])
    assert_geometry_exact(candidate[1], post[1])

