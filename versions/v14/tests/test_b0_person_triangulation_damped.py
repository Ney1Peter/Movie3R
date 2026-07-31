from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from versions.v14.b0_person_triangulation import refine_matched_people
from versions.v14.b0_person_triangulation_damped import (
    DampedTriangulationConfig,
    refine_matched_people_damped,
)


def camera(center):
    value = np.eye(4)
    value[:3, 3] = center
    return value


def person(root, scale=1.0, camera_center=None):
    root = np.asarray(root, dtype=np.float64)
    offsets = np.zeros((20, 3), dtype=np.float64)
    offsets[:, 1] = np.linspace(-0.4, 0.4, 20)
    points = {"root": root.copy(), "joints": root + offsets, "vertices": root + offsets}
    if camera_center is None or scale == 1.0:
        return points
    center = np.asarray(camera_center, dtype=np.float64)
    return {key: center + scale * (value - center) for key, value in points.items()}


def test_single_person_shift_is_exactly_damped_and_cameras_are_unchanged():
    pre_camera = camera([0, 0, 0])
    post_camera = camera([1, 0, 0])
    truth = person([0.1, 0.0, 3.0])
    pre = person([0.1, 0.0, 3.0])
    post = person([0.1, 0.0, 3.0], scale=1.4, camera_center=[1, 0, 0])
    pre_before, post_before = pre_camera.copy(), post_camera.copy()
    baseline, _ = refine_matched_people(pre_camera, post_camera, [pre], [post], [(0, 0)])
    damped, debug = refine_matched_people_damped(
        pre_camera, post_camera, [pre], [post], [(0, 0)]
    )
    baseline_shift = baseline[0]["root"] - post["root"]
    damped_shift = damped[0]["root"] - post["root"]
    assert np.allclose(damped_shift, 0.8 * baseline_shift, atol=1e-10)
    assert np.linalg.norm(damped[0]["root"] - truth["root"]) < np.linalg.norm(
        post["root"] - truth["root"]
    )
    assert debug["action_scale"] == 0.8
    assert np.array_equal(pre_camera, pre_before)
    assert np.array_equal(post_camera, post_before)


def test_rejected_and_unmatched_remain_bit_exact():
    from versions.v14.b0_person_triangulation import PersonTriangulationConfig

    pre_camera, post_camera = camera([0, 0, 0]), camera([1, 0, 0])
    pre = person([0, 0, 3])
    post = person([0, 0, 4])
    unmatched = person([1, 0, 4])
    corrected, debug = refine_matched_people_damped(
        pre_camera,
        post_camera,
        [pre],
        [post, unmatched],
        [(0, 0)],
        base_config=PersonTriangulationConfig(max_median_gap_m=-1.0),
    )
    assert debug["accepted_count"] == 0
    for index, original in enumerate((post, unmatched)):
        for key in ("root", "joints", "vertices"):
            assert np.array_equal(corrected[index][key], original[key])


def test_invalid_damping_is_rejected():
    with pytest.raises(ValueError):
        DampedTriangulationConfig(action_scale=0.0)
    with pytest.raises(ValueError):
        DampedTriangulationConfig(action_scale=1.01)
