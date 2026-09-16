import numpy as np

from experiments.registration_baselines.geometry import (
    _kabsch,
    apply_shared_transform,
    transform_points,
)


def synthetic_arrays(frames=6, people=2):
    cameras = np.repeat(np.eye(4, dtype=np.float64)[None], frames, axis=0)
    joints = np.arange(frames * people * 24 * 3, dtype=np.float64).reshape(frames, people, 24, 3) / 100.0
    vertices = np.arange(frames * people * 11 * 3, dtype=np.float64).reshape(frames, people, 11, 3) / 100.0
    return {
        "cameras_c2w": cameras,
        "joints_world": joints,
        "vertices_world": vertices,
        "persistent_ids": np.tile(np.arange(people), (frames, 1)).astype(np.int32),
        "native_ids": np.tile(np.arange(people), (frames, 1)).astype(np.int32),
        "valid": np.ones((frames, people), dtype=np.uint8),
    }


def known_transform():
    angle = np.deg2rad(31.0)
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    value[:3, 3] = [0.7, -1.2, 0.25]
    return value


def test_identity_transform_changes_nothing():
    arrays = synthetic_arrays()
    mapped = apply_shared_transform(arrays, np.eye(4), 3)
    for key in arrays:
        np.testing.assert_array_equal(mapped[key], arrays[key])


def test_known_se3_is_recovered_and_shared_by_camera_human_scene():
    rng = np.random.default_rng(7)
    source = rng.normal(size=(40, 3))
    truth = known_transform()
    target = transform_points(source, truth)
    recovered = _kabsch(source, target)
    np.testing.assert_allclose(recovered, truth, atol=1e-10)
    arrays = synthetic_arrays()
    mapped = apply_shared_transform(arrays, recovered, 3)
    np.testing.assert_allclose(mapped["cameras_c2w"][3], truth, atol=1e-12)
    np.testing.assert_allclose(mapped["joints_world"][3], transform_points(arrays["joints_world"][3], truth), atol=1e-12)
    np.testing.assert_allclose(mapped["vertices_world"][3], transform_points(arrays["vertices_world"][3], truth), atol=1e-12)
    np.testing.assert_allclose(transform_points(source, recovered), target, atol=1e-12)


def test_only_post_cut_is_modified():
    arrays = synthetic_arrays()
    mapped = apply_shared_transform(arrays, known_transform(), 3)
    for key in ("cameras_c2w", "joints_world", "vertices_world"):
        np.testing.assert_array_equal(mapped[key][:3], arrays[key][:3])
        assert not np.array_equal(mapped[key][3:], arrays[key][3:])
