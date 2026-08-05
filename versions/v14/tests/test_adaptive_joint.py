import numpy as np

from src.dust3r.adaptive_joint import (
    AdaptiveJointConfig,
    apply_to_arrays,
    estimate_shared_boundary,
    gate_boundary,
)


def _rot_z(deg: float) -> np.ndarray:
    angle = np.deg2rad(deg)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_shared_boundary_matches_anonymous_people():
    rng = np.random.default_rng(3)
    pre = rng.normal(size=(2, 32, 3)).astype(np.float64)
    forward = np.eye(4)
    forward[:3, :3] = _rot_z(65.0)
    forward[:3, 3] = [1.2, -0.3, 2.0]
    post = pre @ forward[:3, :3].T + forward[:3, 3]
    post = post[[1, 0]]
    inverse, diagnostics = estimate_shared_boundary(pre, post)
    assert diagnostics["valid"]
    assert diagnostics["selected_permutation_post_index_by_pre_index"] == [1, 0]
    assert diagnostics["shared_rotation_deg"] > 60.0
    assert np.allclose(inverse @ forward, np.eye(4), atol=1e-6)


def test_small_residual_falls_back_exactly():
    rng = np.random.default_rng(7)
    pre = rng.normal(size=(3, 32, 3))
    post = pre + 0.01 * rng.normal(size=pre.shape)
    transform, diagnostics = estimate_shared_boundary(pre, post)
    accepted, reason = gate_boundary(diagnostics, AdaptiveJointConfig())
    assert transform is not None
    assert not accepted
    assert reason == "small_boundary_residual_baseline_kept"


def test_apply_changes_only_post_after_accepted_event():
    rng = np.random.default_rng(11)
    pre = rng.normal(size=(1, 64, 3))
    forward = np.eye(4)
    forward[:3, :3] = _rot_z(90.0)
    forward[:3, 3] = [2.0, 0.0, 0.0]
    post = pre @ forward[:3, :3].T + forward[:3, 3]
    meshes = [pre.copy(), post.copy(), post.copy()]
    cameras = np.tile(np.eye(4), (3, 1, 1))
    points = [np.zeros((1, 4, 4, 3)) for _ in range(3)]
    cameras_new, meshes_new, points_new, records = apply_to_arrays(
        cameras, meshes, points, [1], AdaptiveJointConfig()
    )
    assert records[0]["accepted"]
    assert np.allclose(meshes_new[0], meshes[0])
    assert not np.allclose(meshes_new[1], meshes[1])
    assert np.allclose(meshes_new[1], pre, atol=1e-6)
    assert np.allclose(meshes_new[2], pre, atol=1e-6)
    assert np.allclose(points_new[0], points[0])
