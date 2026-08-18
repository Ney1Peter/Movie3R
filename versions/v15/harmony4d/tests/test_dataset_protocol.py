from __future__ import annotations

import numpy as np

from versions.v15.harmony4d.dataset import locate_sequence_root, load_exo_calibrations
from versions.v15.harmony4d.protocol import camera_pair_rows, select_balanced_pairs


DATA = "/data/wangzheng/iJCV-CODE/data/Harmony4D_work/staging/train_01_hugging"


def test_camera_round_trip_and_static_scatter() -> None:
    root = locate_sequence_root(__import__("pathlib").Path(DATA))
    cameras = load_exo_calibrations(root)
    assert len(cameras) == 22
    for camera in cameras.values():
        np.testing.assert_allclose(
            camera.world_to_camera @ camera.camera_to_world,
            np.eye(4),
            atol=1e-9,
        )
        assert camera.calibration_views == 4
        assert camera.width == 3840 and camera.height == 2160
        assert camera.center_scatter_max_m < 0.1
        assert camera.rotation_scatter_max_deg < 2.0


def test_protocol_pair_selection_is_deterministic() -> None:
    root = locate_sequence_root(__import__("pathlib").Path(DATA))
    cameras = load_exo_calibrations(root)
    first = select_balanced_pairs(cameras)
    second = select_balanced_pairs(cameras)
    assert first == second
    assert {row["angle_stratum"] for row in first} == {"small", "medium", "large", "extreme"}
    assert len(camera_pair_rows(cameras)) == 22 * 21

