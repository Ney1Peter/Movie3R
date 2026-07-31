import numpy as np

from versions.v14.probe_b0_anchor_conflict import b0_centered_solutions


def _human(root):
    return {
        "root": np.asarray(root, dtype=np.float64),
        "torso": np.eye(3, dtype=np.float64),
        "root_rotation": np.eye(3, dtype=np.float64),
        "joints": np.asarray([root], dtype=np.float64),
        "vertices": np.asarray([root], dtype=np.float64),
        "score": 1.0,
        "completeness": 1.0,
        "bbox_area": 1.0,
        "detection_index": 0,
    }


def _cache():
    pre = {"person0": _human([1.0, 0.0, 2.0])}
    post = {"person0": _human([0.0, 0.0, 2.0])}
    return {
        "case": {"pre_frames": [0], "post_frame": 1},
        "humans": [pre, post],
        "clouds": [np.empty((0, 3)), np.empty((0, 3))],
    }


def test_rotation_only_keeps_b0_translation_exactly():
    b0 = np.eye(4, dtype=np.float64)
    b0[:3, 3] = [3.0, 4.0, 5.0]

    solutions = b0_centered_solutions(_cache(), b0)

    np.testing.assert_array_equal(
        solutions["b0_rotation_only"]["translation"], b0[:3, 3]
    )


def test_translation_only_keeps_b0_rotation_exactly():
    b0 = np.eye(4, dtype=np.float64)
    b0[:3, :3] = np.diag([-1.0, 1.0, -1.0])

    solutions = b0_centered_solutions(_cache(), b0)

    np.testing.assert_array_equal(
        solutions["b0_translation_only"]["rotation"], b0[:3, :3]
    )
