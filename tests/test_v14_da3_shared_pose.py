import cv2
import numpy as np
import pytest
from types import SimpleNamespace

from versions.v14.b0_da3_fine_alignment import (
    DA3FineAligner,
    apply_boundary_to_points,
    apply_boundary_to_pose,
    b0_camera_center,
    da3_proposal,
    direction_angle_deg,
    refine_b0_with_da3,
    rotation_angle_deg,
)
from versions.v14.probe_b0_da3_shared_pose import (
    SAFE_DIRECTION_SPREAD_DEG,
    SAFE_DIRECTION_VS_B0_DEG,
    SAFE_RIGHT_ROTATION_DEG,
    SAFE_ROTATION_SPREAD_DEG,
    consensus_proposal,
    proposal_from_prediction,
    safe_gate_decision,
)
from versions.v14.probe_b0_residual_observability import transform


def rotation(vector):
    return cv2.Rodrigues(np.asarray(vector, dtype=np.float64))[0]


def accepted_fixture(proposal_rotation_deg=10.0, proposal_direction_deg=20.0):
    pre = np.eye(4, dtype=np.float64)
    raw_post = transform(np.eye(3), [1.0, 0.0, 0.0])
    b0 = np.eye(4, dtype=np.float64)
    angle = np.radians(proposal_direction_deg)
    desired_post = transform(
        rotation([0.0, np.radians(proposal_rotation_deg), 0.0]),
        [np.cos(angle), 0.0, np.sin(angle)],
    )
    forward = np.stack([pre, desired_post])
    reverse = np.stack([desired_post, pre])
    return b0, pre, raw_post, forward, reverse


def test_da3_proposal_is_invariant_to_shared_da3_world_gauge():
    pre = transform(rotation([0.08, -0.04, 0.02]), [0.4, -0.2, 0.1])
    raw_post = transform(rotation([-0.05, 0.03, 0.06]), [-0.3, 0.5, 0.2])
    true_boundary = transform(rotation([0.12, 0.07, -0.03]), [1.0, -0.4, 0.3])
    desired_post = true_boundary @ raw_post

    da3_pre = transform(rotation([-0.2, 0.1, 0.04]), [2.0, 1.0, -0.5])
    da3_world_to_pre_world = pre @ np.linalg.inv(da3_pre)
    da3_post = np.linalg.inv(da3_world_to_pre_world) @ desired_post
    cache = {"poses": [pre, raw_post]}

    forward = proposal_from_prediction(
        cache,
        {"status": "ok", "camera_to_world": np.stack([da3_pre, da3_post])},
        reverse=False,
    )
    reverse = proposal_from_prediction(
        cache,
        {"status": "ok", "camera_to_world": np.stack([da3_post, da3_pre])},
        reverse=True,
    )
    expected_direction = desired_post[:3, 3] - pre[:3, 3]
    expected_direction /= np.linalg.norm(expected_direction)

    for proposal in (forward, reverse):
        np.testing.assert_allclose(
            proposal["boundary_rotation"], true_boundary[:3, :3], atol=1e-10
        )
        np.testing.assert_allclose(
            proposal["baseline_direction_world"], expected_direction, atol=1e-10
        )

    runtime_forward = da3_proposal(pre, raw_post, np.stack([da3_pre, da3_post]), False)
    runtime_reverse = da3_proposal(pre, raw_post, np.stack([da3_post, da3_pre]), True)
    for proposal in (runtime_forward, runtime_reverse):
        np.testing.assert_allclose(
            proposal["boundary_rotation"], true_boundary[:3, :3], atol=1e-10
        )
        np.testing.assert_allclose(
            proposal["baseline_direction_world"], expected_direction, atol=1e-10
        )


def test_consensus_preserves_identical_pose_proposals():
    proposal = {
        "boundary_rotation": rotation([0.1, -0.2, 0.03]),
        "baseline_direction_world": np.asarray([1.0, 2.0, 3.0]),
        "da3_baseline_units": 1.5,
    }
    output = consensus_proposal([proposal, proposal])
    np.testing.assert_allclose(
        output["boundary_rotation"], proposal["boundary_rotation"], atol=1e-10
    )
    np.testing.assert_allclose(
        output["baseline_direction_world"],
        proposal["baseline_direction_world"]
        / np.linalg.norm(proposal["baseline_direction_world"]),
        atol=1e-10,
    )
    assert output["da3_baseline_units"] == 1.5


def test_safe_gate_accepts_limits_and_rejects_missing_nonfinite_or_excess():
    accepted = {
        "forward_reverse_rotation_spread_deg": SAFE_ROTATION_SPREAD_DEG,
        "forward_reverse_direction_spread_deg": SAFE_DIRECTION_SPREAD_DEG,
        "right_rotation_deg": SAFE_RIGHT_ROTATION_DEG,
        "direction_vs_b0_deg": SAFE_DIRECTION_VS_B0_DEG,
    }
    assert safe_gate_decision(accepted)
    for key in tuple(accepted):
        missing = dict(accepted)
        missing.pop(key)
        assert not safe_gate_decision(missing)

        nonfinite = dict(accepted)
        nonfinite[key] = np.nan
        assert not safe_gate_decision(nonfinite)

        excess = dict(accepted)
        excess[key] = accepted[key] + 1e-3
        assert not safe_gate_decision(excess)


def test_runtime_caps_residual_and_preserves_b0_baseline_length():
    b0, pre, raw_post, forward, reverse = accepted_fixture()
    output, diagnostics = refine_b0_with_da3(
        b0, pre, raw_post, forward, reverse
    )

    assert diagnostics["accepted"]
    assert diagnostics["selected"] == "da3_bounded_consensus"
    assert diagnostics["applied_rotation_deg"] <= 3.0 + 1e-9
    assert diagnostics["applied_direction_deg"] <= 5.0 + 1e-9
    assert rotation_angle_deg(b0, output) == pytest.approx(3.0, abs=1e-8)

    coarse_center = b0_camera_center(b0, raw_post)
    output_center = output[:3, :3] @ raw_post[:3, 3] + output[:3, 3]
    assert np.linalg.norm(output_center - pre[:3, 3]) == pytest.approx(
        np.linalg.norm(coarse_center - pre[:3, 3]), abs=1e-10
    )
    assert direction_angle_deg(
        coarse_center - pre[:3, 3], output_center - pre[:3, 3]
    ) == pytest.approx(5.0, abs=1e-8)


@pytest.mark.parametrize(
    "failure",
    ("missing", "nan", "rotation_spread", "direction_spread", "bad_pre"),
)
def test_runtime_returns_bit_exact_b0_on_invalid_or_conflicting_cue(failure):
    b0, pre, raw_post, forward, reverse = accepted_fixture()
    if failure == "missing":
        forward = None
    elif failure == "nan":
        forward = forward.copy()
        forward[1, 0, 0] = np.nan
    elif failure == "rotation_spread":
        reverse = reverse.copy()
        reverse[0, :3, :3] = rotation([0.0, np.radians(30.0), 0.0])
    elif failure == "direction_spread":
        reverse = reverse.copy()
        reverse[0, :3, 3] = [0.0, 0.0, 1.0]
    elif failure == "bad_pre":
        pre = np.zeros((2, 2), dtype=np.float64)

    output, diagnostics = refine_b0_with_da3(
        b0, pre, raw_post, forward, reverse
    )
    assert not diagnostics["accepted"]
    assert diagnostics["selected"] == "b0"
    assert np.array_equal(output, b0)


def test_runtime_returns_bit_exact_b0_for_degenerate_b0_camera_baseline():
    b0, pre, raw_post, forward, reverse = accepted_fixture()
    raw_post = np.eye(4, dtype=np.float64)
    output, diagnostics = refine_b0_with_da3(
        b0, pre, raw_post, forward, reverse
    )
    assert not diagnostics["accepted"]
    assert np.array_equal(output, b0)


def test_runtime_fallback_preserves_b0_dtype_and_bytes():
    b0, pre, raw_post, _, reverse = accepted_fixture()
    b0 = b0.astype(np.float32)
    output, diagnostics = refine_b0_with_da3(
        b0, pre, raw_post, None, reverse
    )
    assert not diagnostics["accepted"]
    assert output.dtype == b0.dtype
    assert output.tobytes() == b0.tobytes()


def test_one_boundary_is_shared_by_camera_pose_and_all_points():
    boundary = transform(rotation([0.1, -0.2, 0.03]), [0.4, -0.3, 0.2])
    pose = transform(rotation([-0.04, 0.08, 0.02]), [1.0, 2.0, 3.0])
    points = np.asarray([[0.0, 0.0, 0.0], [1.0, -2.0, 0.5]])

    expected_pose = boundary @ pose
    expected_points = points @ boundary[:3, :3].T + boundary[:3, 3]
    np.testing.assert_allclose(apply_boundary_to_pose(boundary, pose), expected_pose)
    np.testing.assert_allclose(
        apply_boundary_to_points(boundary, points), expected_points
    )


class FakeDA3:
    def __init__(self, camera_to_world_pairs):
        self.camera_to_world_pairs = list(camera_to_world_pairs)

    def inference(self, images, **kwargs):
        camera_to_world = self.camera_to_world_pairs.pop(0)
        return SimpleNamespace(extrinsics=np.linalg.inv(camera_to_world))


def test_deployment_adapter_runs_bidirectional_da3_without_evaluator_inputs():
    b0, pre, raw_post, forward, reverse = accepted_fixture()
    adapter = DA3FineAligner(FakeDA3([forward, reverse]))
    output, diagnostics = adapter.refine_images(
        b0,
        pre,
        raw_post,
        np.zeros((16, 16, 3), dtype=np.uint8),
        np.ones((16, 16, 3), dtype=np.uint8),
    )
    expected, expected_diagnostics = refine_b0_with_da3(
        b0, pre, raw_post, forward, reverse
    )
    np.testing.assert_allclose(output, expected, atol=1e-12)
    assert diagnostics["accepted"] == expected_diagnostics["accepted"]
    assert diagnostics["da3_forward_seconds"] >= 0.0
    assert diagnostics["da3_reverse_seconds"] >= 0.0


def test_deployment_adapter_returns_bit_exact_b0_on_da3_runtime_failure():
    class BrokenDA3:
        def inference(self, images, **kwargs):
            raise RuntimeError("synthetic inference failure")

    b0, pre, raw_post, _, _ = accepted_fixture()
    output, diagnostics = DA3FineAligner(BrokenDA3()).refine_images(
        b0, pre, raw_post, "pre.png", "post.png"
    )
    assert np.array_equal(output, b0)
    assert not diagnostics["accepted"]
    assert diagnostics["selected"] == "b0"
    assert "synthetic inference failure" in diagnostics["reason"]
