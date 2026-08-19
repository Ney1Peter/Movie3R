from __future__ import annotations

import numpy as np

from versions.v16.harmony4d.causal_stabilization import (
    Candidate,
    apply_candidate,
    boundary_permutation_ids,
    camera_coordinates,
    causal_shot_gauge_stabilize,
    coupled_boundary_register,
)


def synthetic_arrays(frames: int = 6, people: int = 2) -> dict[str, np.ndarray]:
    joints_camera = np.zeros((people, 24, 3), dtype=np.float64)
    for person in range(people):
        joints_camera[person, :, 0] = person * 1.2 + np.linspace(-0.2, 0.2, 24)
        joints_camera[person, :, 1] = np.linspace(-0.8, 0.8, 24)
        joints_camera[person, :, 2] = 3.0 + 0.1 * person
    vertices_camera = joints_camera[:, :5].copy()
    cameras, joints, vertices = [], [], []
    for frame in range(frames):
        camera = np.eye(4, dtype=np.float64)
        camera[:3, 3] = [0.1 * (frame % 3), 0.02 * frame, 0.0]
        if frame >= 3:
            camera[:3, 3] += [2.0, 0.0, 1.0]
        cameras.append(camera)
        joints.append(joints_camera @ camera[:3, :3].T + camera[:3, 3])
        vertices.append(vertices_camera @ camera[:3, :3].T + camera[:3, 3])
    valid = np.ones((frames, people), dtype=np.uint8)
    native = np.tile(np.arange(people, dtype=np.int32), (frames, 1))
    persistent = native.copy()
    return {
        "cameras_c2w": np.stack(cameras),
        "joints_world": np.stack(joints),
        "vertices_world": np.stack(vertices),
        "valid": valid,
        "native_ids": native,
        "persistent_ids": persistent,
    }


def assert_relative_geometry_equal(first: dict[str, np.ndarray], second: dict[str, np.ndarray]) -> None:
    for frame in range(len(first["valid"])):
        np.testing.assert_allclose(
            camera_coordinates(first["cameras_c2w"][frame], first["joints_world"][frame]),
            camera_coordinates(second["cameras_c2w"][frame], second["joints_world"][frame]),
            atol=1e-9,
        )
        np.testing.assert_allclose(
            camera_coordinates(first["cameras_c2w"][frame], first["vertices_world"][frame]),
            camera_coordinates(second["cameras_c2w"][frame], second["vertices_world"][frame]),
            atol=1e-9,
        )


def test_shot_gauge_freeze_is_relative_invariant_and_causal() -> None:
    arrays = synthetic_arrays()
    stabilized, _ = causal_shot_gauge_stabilize(arrays, boundary=3, alpha=0.0)
    assert_relative_geometry_equal(arrays, stabilized)
    np.testing.assert_allclose(
        stabilized["cameras_c2w"][:3],
        np.repeat(stabilized["cameras_c2w"][0][None], 3, axis=0),
    )
    np.testing.assert_allclose(
        stabilized["cameras_c2w"][3:],
        np.repeat(stabilized["cameras_c2w"][3][None], 3, axis=0),
    )

    prefix = {key: value[:5].copy() for key, value in arrays.items()}
    prefix_stabilized, _ = causal_shot_gauge_stabilize(prefix, boundary=3, alpha=0.0)
    for key in ("cameras_c2w", "joints_world", "vertices_world"):
        np.testing.assert_allclose(prefix_stabilized[key], stabilized[key][:5], atol=1e-9)


def test_boundary_registration_moves_only_post_common_gauge() -> None:
    arrays = synthetic_arrays()
    registered, debug = coupled_boundary_register(
        arrays, boundary=3, pairs=[(0, 0), (1, 1)],
        kind="translation", blend=1.0,
    )
    assert debug["accepted"]
    np.testing.assert_allclose(registered["cameras_c2w"][:3], arrays["cameras_c2w"][:3])
    np.testing.assert_allclose(registered["joints_world"][:3], arrays["joints_world"][:3])
    assert_relative_geometry_equal(arrays, registered)
    before = np.linalg.norm(arrays["joints_world"][3, :, :3] - arrays["joints_world"][2, :, :3])
    after = np.linalg.norm(registered["joints_world"][3, :, :3] - registered["joints_world"][2, :, :3])
    assert after < before


def test_boundary_permutation_preserves_every_detection() -> None:
    arrays = synthetic_arrays()
    arrays["native_ids"][3:] = arrays["native_ids"][3:, ::-1]
    output, debug = boundary_permutation_ids(arrays, boundary=3, pairs=[(0, 1), (1, 0)])
    assert debug["valid_detection_count_preserved"]
    np.testing.assert_array_equal(output["valid"], arrays["valid"])
    assert output["persistent_ids"][3, 1] == output["persistent_ids"][2, 0]
    assert output["persistent_ids"][3, 0] == output["persistent_ids"][2, 1]


def test_reliability_gate_returns_exact_m15_fallback() -> None:
    arrays = synthetic_arrays()
    baseline, _ = apply_candidate(
        arrays, boundary=3, pairs=[(0, 0), (1, 1)],
        candidate=Candidate("baseline"),
    )
    gated, debug = apply_candidate(
        arrays, boundary=3, pairs=[(0, 0), (1, 1)],
        candidate=Candidate(
            "forced_fallback",
            camera_alpha=0.0,
            boundary_kind="translation",
            boundary_blend=1.0,
            root_alpha=0.5,
            root_beta=0.02,
            gate_max_boundary_residual_m=0.25,
            gate_min_matches=3,
        ),
    )
    assert not debug["reliability_gate"]["accepted"]
    assert debug["runtime_contract"]["exact_m15_fallback"]
    for key in arrays:
        np.testing.assert_allclose(gated[key], baseline[key])


def test_reliability_gate_accepts_well_registered_boundary() -> None:
    arrays = synthetic_arrays()
    gated, debug = apply_candidate(
        arrays, boundary=3, pairs=[(0, 0), (1, 1)],
        candidate=Candidate(
            "accepted",
            boundary_kind="translation",
            boundary_blend=1.0,
            gate_max_boundary_residual_m=0.25,
            gate_min_matches=2,
        ),
    )
    assert debug["reliability_gate"]["accepted"]
    assert not debug["runtime_contract"]["exact_m15_fallback"]
    assert not np.allclose(gated["cameras_c2w"][3:], arrays["cameras_c2w"][3:])


def test_multicue_gate_rejects_low_coverage_and_large_translation() -> None:
    arrays = synthetic_arrays(people=3)
    gated, debug = apply_candidate(
        arrays, boundary=3, pairs=[(0, 0), (1, 1)],
        candidate=Candidate(
            "multicue_fallback",
            boundary_kind="translation",
            boundary_blend=1.0,
            gate_max_boundary_residual_m=0.25,
            gate_min_matches=2,
            gate_min_match_fraction=0.75,
            gate_max_translation_m=0.5,
        ),
    )
    gate = debug["reliability_gate"]
    assert not gate["accepted"]
    assert gate["observed_match_fraction"] == 2.0 / 3.0
    assert "insufficient_match_fraction" in gate["reasons"]
    assert "boundary_translation_outside_trust_region" in gate["reasons"]
    assert debug["runtime_contract"]["exact_m15_fallback"]
    np.testing.assert_allclose(gated["cameras_c2w"], arrays["cameras_c2w"])
