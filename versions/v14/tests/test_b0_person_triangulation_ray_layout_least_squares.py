from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from versions.v14.b0_person_triangulation import PersonTriangulationConfig
from versions.v14.b0_person_triangulation_ray_layout_least_squares import (
    RayLayoutLeastSquaresConfig,
    refine_matched_people_ray_layout_least_squares,
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


def test_actions_remain_on_post_rays_and_camera_is_bit_exact():
    pre_camera, post_camera = camera([0, 0, 0]), camera([1, 0, 0])
    truths = [person([0.1, 0.0, 3.0]), person([0.8, 0.1, 3.4])]
    pre = [project_person(item, pre_camera[:3, 3], 1.0) for item in truths]
    post = [project_person(item, post_camera[:3, 3], scale) for item, scale in zip(truths, (1.4, 1.2))]
    pre_before, post_before = pre_camera.copy(), post_camera.copy()
    corrected, debug = refine_matched_people_ray_layout_least_squares(
        pre_camera,
        post_camera,
        pre,
        post,
        [(0, 0), (1, 1)],
        layout_config=RayLayoutLeastSquaresConfig(prior_weight=1.0),
    )
    assert debug["accepted_count"] == 2
    for index, record in enumerate(debug["people"]):
        shift = corrected[index]["root"] - post[index]["root"]
        assert np.linalg.norm(np.cross(shift, record["ray_world"])) < 1e-10
    assert np.array_equal(pre_camera, pre_before)
    assert np.array_equal(post_camera, post_before)


def test_layout_solve_does_not_increase_its_observable_objective_without_prior():
    pre_camera, post_camera = camera([0, 0, 0]), camera([1, 0, 0])
    truths = [person([0.1, 0.0, 3.0]), person([0.8, 0.1, 3.4]), person([-0.4, 0.0, 2.8])]
    pre = [project_person(item, pre_camera[:3, 3], 1.0) for item in truths]
    post = [project_person(item, post_camera[:3, 3], scale) for item, scale in zip(truths, (1.4, 1.2, 1.5))]
    _, debug = refine_matched_people_ray_layout_least_squares(
        pre_camera,
        post_camera,
        pre,
        post,
        [(0, 0), (1, 1), (2, 2)],
        layout_config=RayLayoutLeastSquaresConfig(prior_weight=0.0),
    )
    assert debug["observable_layout_objective_after"] <= (
        debug["observable_layout_objective_before"] + 1e-12
    )


def test_single_person_falls_back_to_original_brtc_individual_action():
    pre_camera, post_camera = camera([0, 0, 0]), camera([1, 0, 0])
    truth = person([0.1, 0.0, 3.0])
    pre = project_person(truth, pre_camera[:3, 3], 1.0)
    post = project_person(truth, post_camera[:3, 3], 1.4)
    corrected, debug = refine_matched_people_ray_layout_least_squares(
        pre_camera,
        post_camera,
        [pre],
        [post],
        [(0, 0)],
        layout_config=RayLayoutLeastSquaresConfig(prior_weight=0.0),
    )
    record = debug["people"][0]
    assert debug["constrained_pair_count"] == 0
    assert record["final_action_m"] == record["brtc_raw_action_m"]
    assert np.allclose(
        corrected[0]["root"] - post["root"],
        record["brtc_individual_shift_world"],
        atol=1e-12,
    )


def test_rejected_and_unmatched_people_are_exact_noops():
    pre_camera, post_camera = camera([0, 0, 0]), camera([1, 0, 0])
    pre = [person([0, 0, 3])]
    post = [person([0, 0, 4]), person([1, 0, 4])]
    corrected, debug = refine_matched_people_ray_layout_least_squares(
        pre_camera,
        post_camera,
        pre,
        post,
        [(0, 0)],
        config=PersonTriangulationConfig(max_median_gap_m=-1.0),
    )
    assert debug["accepted_count"] == 0
    for candidate, original in zip(corrected, post):
        for key in ("root", "joints", "vertices"):
            assert np.array_equal(candidate[key], original[key])
