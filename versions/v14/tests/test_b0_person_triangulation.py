from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from versions.v14.b0_person_triangulation import (
    PersonTriangulationConfig,
    closest_rays,
    refine_matched_people,
)


def camera(center):
    value = np.eye(4)
    value[:3, 3] = center
    return value


def person(root, joint_count=20):
    root = np.asarray(root, dtype=np.float64)
    offsets = np.zeros((joint_count, 3), dtype=np.float64)
    offsets[:, 1] = np.linspace(-0.4, 0.4, joint_count)
    joints = root + offsets
    return {"root": root.copy(), "joints": joints, "vertices": joints.copy()}


def project_person(true_person, center, wrong_depth_scale=1.0):
    center = np.asarray(center, dtype=np.float64)
    output = {}
    for key in ("root", "joints", "vertices"):
        rays = true_person[key] - center
        output[key] = center + wrong_depth_scale * rays
    return output


def test_closest_rays_recovers_intersection():
    point = np.array([0.2, -0.1, 3.0])
    first = np.array([0.0, 0.0, 0.0])
    second = np.array([1.0, 0.0, 0.0])
    midpoint, depth_a, depth_b, gap, sine = closest_rays(
        first, point - first, second, point - second
    )
    assert np.allclose(midpoint, point, atol=1e-10)
    assert depth_a > 0 and depth_b > 0 and sine > 0
    assert gap < 1e-10


def test_single_person_corrects_depth_without_camera_mutation():
    pre_camera, post_camera = camera([0, 0, 0]), camera([1, 0, 0])
    true = person([0.1, 0.0, 3.0])
    pre = project_person(true, pre_camera[:3, 3], 1.0)
    post = project_person(true, post_camera[:3, 3], 1.4)
    pre_before, post_before = pre_camera.copy(), post_camera.copy()
    corrected, debug = refine_matched_people(
        pre_camera, post_camera, [pre], [post], [(0, 0)]
    )
    assert debug["accepted_count"] == 1
    assert np.linalg.norm(corrected[0]["root"] - true["root"]) < np.linalg.norm(
        post["root"] - true["root"]
    )
    assert np.array_equal(pre_camera, pre_before)
    assert np.array_equal(post_camera, post_before)


def test_rejection_and_unmatched_are_bit_exact():
    config = PersonTriangulationConfig(max_median_gap_m=-1.0)
    pre_camera, post_camera = camera([0, 0, 0]), camera([1, 0, 0])
    pre = person([0, 0, 3])
    post = person([0, 0, 4])
    unmatched = person([1, 0, 4])
    corrected, debug = refine_matched_people(
        pre_camera, post_camera, [pre], [post, unmatched], [(0, 0)], config
    )
    assert debug["accepted_count"] == 0
    for index, original in enumerate((post, unmatched)):
        for key in ("root", "joints", "vertices"):
            assert np.array_equal(corrected[index][key], original[key])


def test_common_world_gauge_equivariance():
    pre_camera, post_camera = camera([0, 0, 0]), camera([1, 0, 0])
    true = person([0.1, 0.0, 3.0])
    pre = project_person(true, pre_camera[:3, 3], 1.0)
    post = project_person(true, post_camera[:3, 3], 1.3)
    corrected, _ = refine_matched_people(pre_camera, post_camera, [pre], [post], [(0, 0)])
    angle = 0.4
    transform = np.eye(4)
    transform[:3, :3] = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ])
    transform[:3, 3] = [0.3, -0.2, 0.1]

    def moved(item):
        return {
            key: item[key] @ transform[:3, :3].T + transform[:3, 3]
            for key in ("root", "joints", "vertices")
        }

    corrected_moved, _ = refine_matched_people(
        transform @ pre_camera,
        transform @ post_camera,
        [moved(pre)],
        [moved(post)],
        [(0, 0)],
    )
    expected = moved(corrected[0])
    for key in ("root", "joints", "vertices"):
        assert np.allclose(corrected_moved[0][key], expected[key], atol=1e-9)
