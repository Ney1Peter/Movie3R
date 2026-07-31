from __future__ import annotations

import copy

import numpy as np
from scipy.spatial.transform import Rotation

from versions.v14.b0_person_body_scale_consistency import (
    BodyScaleConfig,
    scale_person_about_root,
)
from versions.v14.eval_brtc_global_orientation_kabsch_egohumans import (
    rotate_person_around_root,
)
from versions.v14.eval_brtc_kabsch_body_scale_composition import (
    replay_scale_over_kabsch,
)


def _person(factor: float, root=(2.0, -1.0, 3.0)):
    root = np.asarray(root, dtype=np.float64)
    joints = np.stack(
        [
            root
            + factor
            * np.asarray([0.07 * index, 0.04 * index, 0.025 * index])
            for index in range(22)
        ]
    )
    vertices = np.stack(
        [
            root + factor * np.asarray([0.01 * index, -0.02 * index, 0.03])
            for index in range(30)
        ]
    )
    return {
        "root": root,
        "joints": joints,
        "vertices": vertices,
        "global_track_id": 1,
        "native_track_id": 7,
    }


def test_rotation_and_uniform_scale_commute_about_same_native_root():
    person = _person(0.9)
    rotation = Rotation.from_rotvec(np.asarray([0.1, -0.2, 0.15])).as_matrix()
    first = scale_person_about_root(
        rotate_person_around_root(person, rotation), 1.08
    )
    second = rotate_person_around_root(
        scale_person_about_root(person, 1.08), rotation
    )
    for key in ("root", "joints", "vertices"):
        np.testing.assert_allclose(first[key], second[key], atol=1e-12)
    np.testing.assert_array_equal(first["root"], person["root"])


def test_scale_replay_consumes_inherited_local_state_and_preserves_roots():
    def frame(factor):
        return {
            "people": [_person(factor)],
            "method_camera_c2w": np.eye(4),
        }

    chain = {
        "chain_index": 0,
        "segments": [[frame(1.0)], [frame(0.8)], [frame(0.9)]],
    }
    chain["frames"] = [row for segment in chain["segments"] for row in segment]
    runtime_rows = []
    for cut in (0, 1):
        runtime_rows.append(
            {
                "chain_index": 0,
                "cut_index": cut,
                "association": {"track_to_post_index": {1: 0}},
                "action_by_native_track": {
                    7: {
                        "brtc_accepted": True,
                        "orientation_applied": True,
                        "shift_world": np.zeros(3),
                        "rotation_world": np.eye(3),
                    }
                },
            }
        )
    config = BodyScaleConfig(
        fraction=1.0,
        relative_cap=0.5,
        max_log_mad=1e-10,
        min_valid_edges=12,
    )
    output, debug = replay_scale_over_kabsch([copy.deepcopy(chain)], runtime_rows, config)
    first_extent = np.linalg.norm(
        output[0]["segments"][0][0]["people"][0]["joints"]
        - output[0]["segments"][0][0]["people"][0]["root"],
        axis=1,
    ).mean()
    for segment in (1, 2):
        person = output[0]["segments"][segment][0]["people"][0]
        extent = np.linalg.norm(person["joints"] - person["root"], axis=1).mean()
        np.testing.assert_allclose(extent, first_extent, atol=1e-12)
        np.testing.assert_array_equal(
            person["root"], chain["segments"][segment][0]["people"][0]["root"]
        )
    assert debug[1]["inherited_scaled_pre"] is True
    assert max(row["first_frame_replay_max_abs_delta"] for row in debug) == 0.0
