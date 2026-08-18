from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from versions.v15.harmony4d.dataset import locate_sequence_root, load_exo_calibrations
from versions.v15.harmony4d.protocol import (
    camera_pair_rows,
    select_balanced_pairs,
    write_jsonl,
)


DATA = "/data/wangzheng/iJCV-CODE/data/Harmony4D_work/staging/train_01_hugging"
GRAPPLING = (
    "/data/wangzheng/iJCV-CODE/data/Harmony4D_work/staging/"
    "train_03_grappling2/03_grappling2/008_grappling2"
)


def require_staged_data(path: str) -> Path:
    staged = Path(path)
    if not staged.exists():
        pytest.skip("requires optional Harmony4D staging restored from Harmony4D.zip")
    return staged


def test_camera_round_trip_and_static_scatter() -> None:
    root = locate_sequence_root(require_staged_data(DATA))
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
    root = locate_sequence_root(require_staged_data(DATA))
    cameras = load_exo_calibrations(root)
    first = select_balanced_pairs(cameras)
    second = select_balanced_pairs(cameras)
    assert first == second
    assert {row["angle_stratum"] for row in first} == {"small", "medium", "large", "extreme"}
    assert len(camera_pair_rows(cameras)) == 22 * 21


def test_missing_aria_transform_uses_annotation_pnp() -> None:
    cameras = load_exo_calibrations(require_staged_data(GRAPPLING))
    assert len(cameras) == 20
    for camera in cameras.values():
        assert camera.extrinsic_source == "published_smpl45_to_poses2d45_static_pnp"
        assert camera.reprojection_median_px <= 5.0
        assert camera.reprojection_p95_px <= 15.0
        np.testing.assert_allclose(
            camera.world_to_camera @ camera.camera_to_world,
            np.eye(4),
            atol=1e-9,
        )


def test_jsonl_digest_is_exact_file_sha256(tmp_path) -> None:
    import hashlib

    path = tmp_path / "manifest.jsonl"
    digest = write_jsonl(path, [{"z": 1, "a": "value"}])
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
