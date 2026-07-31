from __future__ import annotations

import copy

import numpy as np

from versions.v14 import b0_person_body_scale_consistency as scale


def _person(factor: float = 1.0, root=(2.0, -1.0, 3.0)):
    root = np.asarray(root, dtype=np.float64)
    joints = np.stack(
        [root + factor * np.asarray([0.1 * index, 0.03 * index, 0.02 * index]) for index in range(22)]
    )
    vertices = np.stack(
        [root + factor * np.asarray([0.01 * index, -0.02 * index, 0.03]) for index in range(30)]
    )
    return {"root": root, "joints": joints, "vertices": vertices}


def test_same_person_bone_ratio_recovers_uniform_scale():
    pre = _person(1.0)
    post = _person(0.8)
    config = scale.BodyScaleConfig(
        fraction=1.0,
        relative_cap=0.5,
        max_log_mad=1e-10,
        min_valid_edges=12,
    )
    evidence = scale.robust_body_scale_evidence(pre, post, config)
    factor, accepted, reason = scale.bounded_scale_factor(evidence, config)
    assert accepted
    assert reason is None
    assert evidence["valid_edge_count"] == len(scale.STABLE_BODY_EDGES)
    np.testing.assert_allclose(factor, 1.25, atol=1e-12)


def test_scaling_preserves_native_root_bit_exact():
    person = _person(1.0)
    corrected = scale.scale_person_about_root(person, 1.1)
    np.testing.assert_array_equal(corrected["root"], person["root"])
    np.testing.assert_allclose(
        corrected["joints"] - person["root"],
        1.1 * (person["joints"] - person["root"]),
    )
    np.testing.assert_allclose(
        corrected["vertices"] - person["root"],
        1.1 * (person["vertices"] - person["root"]),
    )


def test_high_mad_fallback_is_bit_exact():
    pre = _person(1.0)
    post = _person(1.0)
    # Distort alternating topology endpoints so no common scalar explains it.
    post["joints"][1::2] += np.asarray([0.4, -0.2, 0.1])
    config = scale.BodyScaleConfig(max_log_mad=1e-6, min_valid_edges=8)
    corrected, debug = scale.refine_brtc_output_body_scale(
        [pre], [post], [(0, 0)], config
    )
    assert debug["people"][0]["accepted"] is False
    assert debug["people"][0]["fallback_reason"] == "log_ratio_mad_gate"
    for key in ("root", "joints", "vertices"):
        np.testing.assert_array_equal(corrected[0][key], post[key])


def test_unmatched_is_exact_and_matched_only_changes_body():
    pre = [_person(1.0), _person(1.0, root=(5.0, 0.0, 2.0))]
    post = [_person(0.8), _person(0.7, root=(5.0, 0.0, 2.0))]
    config = scale.BodyScaleConfig(
        fraction=1.0,
        relative_cap=0.5,
        max_log_mad=1e-10,
        min_valid_edges=12,
    )
    corrected, debug = scale.refine_brtc_output_body_scale(
        pre, post, [(0, 0)], config
    )
    assert debug["root_max_abs_change"] == 0.0
    np.testing.assert_array_equal(corrected[0]["root"], post[0]["root"])
    assert not np.array_equal(corrected[0]["joints"], post[0]["joints"])
    for key in ("root", "joints", "vertices"):
        np.testing.assert_array_equal(corrected[1][key], post[1][key])


def test_combined_runtime_scales_only_brtc_accepted(monkeypatch):
    pre = [_person(1.0), _person(1.0, root=(5.0, 0.0, 2.0))]
    post = [_person(0.8), _person(0.8, root=(5.0, 0.0, 2.0))]

    def fake_brtc(pre_camera, post_camera, pre_people, post_people, matches, config):
        del pre_camera, post_camera, pre_people, matches, config
        corrected = copy.deepcopy(post_people)
        shift = np.asarray([0.2, 0.0, 0.1])
        for key in ("root", "joints", "vertices"):
            corrected[0][key] = np.asarray(corrected[0][key]) + shift
        return corrected, {
            "camera_update": "none",
            "matched_count": 2,
            "accepted_count": 1,
            "group_shift_world": shift,
            "selected_residual_lambda": 0.0,
            "observable_layout_objective_by_lambda": {0.0: 0.0},
            "people": [
                {
                    "pre_index": 0,
                    "post_index": 0,
                    "accepted": True,
                    "individual_shift_world": shift,
                    "final_shift_world": shift,
                    "evidence": {},
                },
                {
                    "pre_index": 1,
                    "post_index": 1,
                    "accepted": False,
                    "individual_shift_world": np.zeros(3),
                    "final_shift_world": np.zeros(3),
                    "evidence": {},
                },
            ],
        }

    monkeypatch.setattr(scale, "refine_matched_people", fake_brtc)
    config = scale.BodyScaleConfig(
        fraction=1.0,
        relative_cap=0.5,
        max_log_mad=1e-10,
        min_valid_edges=12,
    )
    corrected, debug = scale.refine_matched_people_body_scale_consistency(
        np.eye(4),
        np.eye(4),
        pre,
        post,
        [(0, 0), (1, 1)],
        scale_config=config,
    )
    expected_root = post[0]["root"] + np.asarray([0.2, 0.0, 0.1])
    np.testing.assert_array_equal(corrected[0]["root"], expected_root)
    np.testing.assert_array_equal(corrected[1]["root"], post[1]["root"])
    np.testing.assert_array_equal(corrected[1]["joints"], post[1]["joints"])
    assert debug["root_max_abs_change_vs_brtc"] == 0.0
    assert debug["body_scale_debug"]["matched_count"] == 1
