from __future__ import annotations

import numpy as np

from publication.bridge3r_iclr2027.runtime_contract import (
    apply_locked_multicut,
    apply_locked_transaction,
    load_method_lock,
    locked_candidate,
)
from publication.bridge3r_iclr2027.bridge3r import validate_bindings
from publication.bridge3r_iclr2027.audit_egohumans_publication_equivalence import (
    arrays_numerically_identical,
)


def synthetic_arrays(frames: int = 8, people: int = 2) -> dict[str, np.ndarray]:
    cameras = np.repeat(np.eye(4, dtype=np.float64)[None], frames, axis=0)
    cameras[:, 0, 3] = np.linspace(0.0, 0.7, frames)
    joints = np.zeros((frames, people, 24, 3), dtype=np.float64)
    vertices = np.zeros((frames, people, 6, 3), dtype=np.float64)
    for frame in range(frames):
        for person in range(people):
            centre = np.array([person * 1.2, 0.05 * frame, 3.0 + 0.1 * person])
            joints[frame, person] = centre + np.linspace(-0.1, 0.1, 24)[:, None] * np.array([1.0, 0.2, 0.1])
            vertices[frame, person] = centre + np.linspace(-0.05, 0.05, 6)[:, None]
    return {
        "cameras_c2w": cameras,
        "joints_world": joints,
        "vertices_world": vertices,
        "valid": np.ones((frames, people), dtype=np.uint8),
        "native_ids": np.tile(np.arange(people, dtype=np.int32), (frames, 1)),
        "persistent_ids": np.tile(np.arange(people, dtype=np.int32), (frames, 1)),
    }


def assert_prefix_equal(left: dict[str, np.ndarray], right: dict[str, np.ndarray], end: int) -> None:
    for key in left:
        np.testing.assert_allclose(left[key][:end], right[key][:end], atol=0.0, rtol=0.0)


def test_method_lock_has_exactly_one_ungated_half_translation() -> None:
    lock = load_method_lock()
    candidate = locked_candidate(lock)
    assert candidate.camera_alpha == 1.0
    assert candidate.boundary_kind == "translation"
    assert candidate.boundary_blend == 0.5
    assert candidate.root_alpha is None
    assert candidate.gate_max_boundary_residual_m is None


def test_all_frozen_result_bindings_match_the_lock() -> None:
    result = validate_bindings()
    assert result["status"] == "PASS"
    assert [row["dataset"] for row in result["bindings"]] == [
        "EgoBody-CS150", "EgoHumans-CS100", "Harmony4D-CS150"
    ]


def test_no_cut_is_bit_exact() -> None:
    arrays = synthetic_arrays()
    output, debug = apply_locked_transaction(
        arrays, boundary=None, pairs=[], cut_detected=False
    )
    assert_prefix_equal(arrays, output, len(arrays["valid"]))
    assert debug["runtime_contract"]["no_cut_bit_exact"]


def test_publication_array_audit_preserves_identical_nan_padding() -> None:
    left = np.array([[1.0, np.nan], [2.0, np.nan]], dtype=np.float32)
    same = left.copy()
    changed = left.copy()
    changed[0, 0] = 1.25
    changed_mask = left.copy()
    changed_mask[1, 1] = 0.0
    assert arrays_numerically_identical(left, same)
    assert not arrays_numerically_identical(left, changed)
    assert not arrays_numerically_identical(left, changed_mask)


def test_prefix_and_future_suffix_contracts() -> None:
    arrays = synthetic_arrays()
    full, debug = apply_locked_transaction(
        arrays, boundary=4, pairs=[(0, 0), (1, 1)], cut_detected=True
    )
    prefix = {key: value[:6].copy() for key, value in arrays.items()}
    prefix_output, _ = apply_locked_transaction(
        prefix, boundary=4, pairs=[(0, 0), (1, 1)], cut_detected=True
    )
    assert_prefix_equal(full, prefix_output, 6)

    perturbed = {key: value.copy() for key, value in arrays.items()}
    perturbed["cameras_c2w"][6:, :3, 3] += 99.0
    perturbed["joints_world"][6:] -= 88.0
    perturbed["vertices_world"][6:] += 77.0
    perturbed_output, _ = apply_locked_transaction(
        perturbed, boundary=4, pairs=[(0, 0), (1, 1)], cut_detected=True
    )
    assert_prefix_equal(full, perturbed_output, 6)
    assert debug["runtime_contract"]["future_frames_used"] == 0


def test_pre_cut_history_is_immutable_and_lineage_is_explicit() -> None:
    arrays = synthetic_arrays()
    output, debug = apply_locked_transaction(
        arrays, boundary=4, pairs=[(0, 1), (1, 0)], cut_detected=True
    )
    assert_prefix_equal(arrays, output, 4)
    assert debug["state_lineage"]["pre_cut_state"] == "read_only_shadow"
    assert debug["state_lineage"]["post_cut_state"] == "clean_reset_input"
    assert not debug["reliability_gate"]["enabled"]
    assert debug["root_filter"] is None


def test_multicut_composes_without_rewriting_previous_prefixes() -> None:
    arrays = synthetic_arrays(frames=10)
    output, transactions = apply_locked_multicut(
        arrays,
        [(3, [(0, 0), (1, 1)]), (6, [(0, 0), (1, 1)])],
    )
    assert len(transactions) == 2
    assert_prefix_equal(arrays, output, 3)
    assert all(item["runtime_contract"]["pre_frames_rewritten_after_emission"] is False for item in transactions)
