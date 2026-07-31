from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
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
    orientation_candidate,
    refine_matched_people_orientation_kabsch,
)
from versions.v14.tests.test_b0_person_triangulation_strict_fagd import (
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
        for key in ("root", "joints", "vertices", "torso", "root_rotation"):
            assert np.array_equal(actual[key], expected[key])
            assert np.array_equal(actual[key], post[index][key])


def test_unmatched_people_are_exact_b0():
    pre_camera, post_camera, pre, post = rotated_scene()
    base, _ = refine_matched_people(pre_camera, post_camera, pre, post, [(0, 0)])
    candidate, debug = refine_matched_people_orientation_kabsch(
        pre_camera, post_camera, pre, post, [(0, 0)]
    )
    assert debug["orientation_applied_count"] == 1
    for key in ("root", "joints", "vertices", "torso", "root_rotation"):
        assert np.array_equal(candidate[1][key], base[1][key])
        assert np.array_equal(candidate[1][key], post[1][key])


def test_accepted_orientation_metadata_follows_applied_world_rotation():
    pre_camera, post_camera, pre, post = rotated_scene(count=1)
    base, _ = refine_matched_people(
        pre_camera, post_camera, pre, post, [(0, 0)]
    )
    candidate, debug = refine_matched_people_orientation_kabsch(
        pre_camera, post_camera, pre, post, [(0, 0)]
    )
    rotation = np.asarray(debug["people"][0]["orientation"]["rotation_world"])
    assert debug["people"][0]["orientation"]["applied"] is True
    assert np.array_equal(candidate[0]["root"], base[0]["root"])
    assert np.allclose(candidate[0]["torso"], rotation @ base[0]["torso"])
    assert np.allclose(
        candidate[0]["root_rotation"], rotation @ base[0]["root_rotation"]
    )


def test_separate_orientation_state_preserves_brtc_root_and_controls_kabsch():
    pre_camera, post_camera, pre, post = rotated_scene()
    matches = [(0, 0), (1, 1)]
    base, _ = refine_matched_people(
        pre_camera, post_camera, pre, post, matches
    )
    orientation_pre = copy.deepcopy(pre)
    inherited = Rotation.from_euler("z", 30.0, degrees=True).as_matrix()
    for person in orientation_pre:
        root = person["root"]
        for key in ("joints", "vertices"):
            person[key] = (person[key] - root) @ inherited.T + root
        person["torso"] = inherited @ person["torso"]
        person["root_rotation"] = inherited @ person["root_rotation"]

    shared, _ = refine_matched_people_orientation_kabsch(
        pre_camera, post_camera, pre, post, matches
    )
    separate, debug = refine_matched_people_orientation_kabsch(
        pre_camera,
        post_camera,
        pre,
        post,
        matches,
        orientation_pre_people=orientation_pre,
    )
    assert debug["orientation_pre_state"] == "separate_causal_orientation_state"
    for index in range(len(post)):
        expected, expected_debug = orientation_candidate(
            orientation_pre[index], post[index], base[index]
        )
        assert expected_debug["applied"] is True
        for key in ("root", "joints", "vertices", "torso", "root_rotation"):
            assert np.allclose(separate[index][key], expected[key], atol=1e-12)
        assert np.array_equal(separate[index]["root"], base[index]["root"])
        assert not np.allclose(separate[index]["joints"], shared[index]["joints"])


def test_omitting_orientation_state_is_backward_compatible():
    pre_camera, post_camera, pre, post = rotated_scene()
    matches = [(0, 0), (1, 1)]
    implicit, implicit_debug = refine_matched_people_orientation_kabsch(
        pre_camera, post_camera, pre, post, matches
    )
    explicit, explicit_debug = refine_matched_people_orientation_kabsch(
        pre_camera,
        post_camera,
        pre,
        post,
        matches,
        orientation_pre_people=pre,
    )
    assert implicit_debug["orientation_pre_state"] == (
        "shared_with_brtc_translation_state"
    )
    assert explicit_debug["orientation_pre_state"] == (
        "separate_causal_orientation_state"
    )
    for implicit_person, explicit_person in zip(implicit, explicit):
        for key in ("root", "joints", "vertices", "torso", "root_rotation"):
            assert np.array_equal(implicit_person[key], explicit_person[key])


def test_separate_orientation_state_requires_same_length_and_track_indexing():
    pre_camera, post_camera, pre, post = rotated_scene()
    matches = [(0, 0), (1, 1)]
    with pytest.raises(ValueError, match="same indexing"):
        refine_matched_people_orientation_kabsch(
            pre_camera,
            post_camera,
            pre,
            post,
            matches,
            orientation_pre_people=pre[:1],
        )

    for index, person in enumerate(pre):
        person["global_track_id"] = index
    reversed_orientation = list(reversed(copy.deepcopy(pre)))
    with pytest.raises(ValueError, match="global_track_id differs"):
        refine_matched_people_orientation_kabsch(
            pre_camera,
            post_camera,
            pre,
            post,
            matches,
            orientation_pre_people=reversed_orientation,
        )
