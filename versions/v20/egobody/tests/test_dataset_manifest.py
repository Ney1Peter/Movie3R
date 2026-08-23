from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from versions.v20.egobody.build_manifest import (
    build_rows,
    jsonl_bytes,
    write_split_manifests,
)
from versions.v20.egobody.dataset import (
    CASES_PER_RECORDING,
    CLIP_LENGTH,
    OFFICIAL_TO_PROTOCOL_SPLIT,
    POST_COUNT,
    PRE_COUNT,
    PROTOCOL_NAME,
    file_sha256,
    load_recording_calibrations,
    load_recording_metadata,
    value_sha256,
)


WORKSPACE = Path("/data/wangzheng/iJCV-CODE")
REAL_OUTER = WORKSPACE / "data/EgoBody_work_v20/outer"
REAL_METADATA = WORKSPACE / "data/EgoBody_work_v20/metadata"


def _yaw(degrees: float, translation: tuple[float, float, float]) -> list[list[float]]:
    angle = np.radians(degrees)
    cosine, sine = float(np.cos(angle)), float(np.sin(angle))
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]]
    value[:3, 3] = translation
    return value.tolist()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture()
def synthetic_release(tmp_path: Path) -> dict[str, Path]:
    recordings = [
        ("recording_20210101_S01_S02_01", "scene_train", "train"),
        ("recording_20210102_S03_S04_01", "scene_val", "val"),
        ("recording_20210103_S05_S06_01", "scene_test", "test"),
    ]
    info = tmp_path / "data_info_release.csv"
    splits = tmp_path / "data_splits.csv"
    calibrations = tmp_path / "calibrations"
    params = tmp_path / "kinect_cam_params"
    _write_csv(
        info,
        [
            "scene_name",
            "body_idx_0",
            "body_idx_1",
            "start_frame",
            "end_frame",
            "body_idx_fpv",
            "recording_name",
        ],
        [
            {
                "scene_name": scene,
                "body_idx_0": "0 female",
                "body_idx_1": "1 male",
                "start_frame": 101,
                "end_frame": 400,
                "body_idx_fpv": "0 female",
                "recording_name": recording,
            }
            for recording, scene, _ in recordings
        ],
    )
    split_row = {"train": "", "val": "", "test": ""}
    for recording, _, split in recordings:
        split_row[split] = recording
    _write_csv(splits, ["train", "val", "test"], [split_row])
    intrinsic = {
        "camera_mtx": [[900.0, 0.0, 960.0], [0.0, 901.0, 550.0], [0.0, 0.0, 1.0]],
        "k": [0.1, -0.2, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
    }
    for role in ("kinect_master", "kinect_sub_1", "kinect_sub_2"):
        _write_json(params / role / "Color.json", intrinsic)
    for recording, scene, _ in recordings:
        root = calibrations / recording / "cal_trans"
        _write_json(root / "kinect12_to_world" / f"{scene}.json", {"trans": _yaw(5, (1, 2, 3))})
        _write_json(root / "kinect_11to12_color.json", {"trans": _yaw(-30, (-1, 0, 0))})
        _write_json(root / "kinect_13to12_color.json", {"trans": _yaw(80, (2, 0, 1))})
    return {
        "info": info,
        "splits": splits,
        "calibrations": calibrations,
        "params": params,
        "output": tmp_path / "manifests",
    }


def test_official_split_mapping_and_runtime_gt_isolation(
    synthetic_release: dict[str, Path],
) -> None:
    release = synthetic_release
    recordings = load_recording_metadata(release["info"], release["splits"])
    assert {row.official_split: row.protocol_split for row in recordings} == OFFICIAL_TO_PROTOCOL_SPLIT
    runtime, evaluator, _ = build_rows(
        recordings, release["calibrations"], release["params"]
    )
    forbidden = {
        "official_split",
        "scene_name",
        "subjects_evaluator_only",
        "person_count_evaluator_only",
        "angle_stratum_evaluator_only",
        "camera_calibration_evaluator_only",
    }
    for split in ("development", "holdout", "test"):
        assert len(runtime[split]) == CASES_PER_RECORDING
        assert len(evaluator[split]) == CASES_PER_RECORDING
        for runtime_row, evaluator_row in zip(runtime[split], evaluator[split]):
            assert forbidden.isdisjoint(runtime_row)
            assert not any(key.endswith("_evaluator_only") for key in runtime_row)
            assert evaluator_row["runtime_row_sha256"] == value_sha256(runtime_row)
            assert runtime_row["protocol"] == PROTOCOL_NAME
            assert len(runtime_row["image_members"]) == CLIP_LENGTH
            assert runtime_row["post_frame_numbers"][0] == runtime_row["pre_frame_numbers"][-1] + 1
            assert len(runtime_row["pre_frame_numbers"]) == PRE_COUNT
            assert len(runtime_row["post_frame_numbers"]) == POST_COUNT
            assert all("/frame_" in value for value in runtime_row["image_members"])
            assert all("/kinect_" not in value for value in runtime_row["image_members"])


def test_calibration_chain_is_composed_in_named_direction(
    synthetic_release: dict[str, Path],
) -> None:
    release = synthetic_release
    recording = load_recording_metadata(release["info"], release["splits"])[0]
    cameras = load_recording_calibrations(
        recording, release["calibrations"], release["params"]
    )
    assert sorted(cameras) == ["kinect_11", "kinect_12", "kinect_13"]
    for camera in cameras.values():
        assert np.allclose(
            camera.camera_to_world,
            camera.master_to_world @ camera.camera_to_master,
        )
        assert np.allclose(camera.camera_to_world @ camera.world_to_camera, np.eye(4))
    assert cameras["kinect_12"].intrinsic_role == "kinect_master"
    assert cameras["kinect_11"].intrinsic_role == "kinect_sub_1"
    assert cameras["kinect_13"].intrinsic_role == "kinect_sub_2"


def test_manifest_bytes_and_hashes_are_deterministic(
    synthetic_release: dict[str, Path],
) -> None:
    release = synthetic_release
    recordings = load_recording_metadata(release["info"], release["splits"])
    first_runtime, first_evaluator, _ = build_rows(
        recordings, release["calibrations"], release["params"]
    )
    second_runtime, second_evaluator, _ = build_rows(
        list(reversed(recordings)), release["calibrations"], release["params"]
    )
    assert jsonl_bytes(first_runtime["development"]) == jsonl_bytes(
        second_runtime["development"]
    )
    assert jsonl_bytes(first_evaluator["development"]) == jsonl_bytes(
        second_evaluator["development"]
    )
    source = {"synthetic": True}
    first = write_split_manifests(
        "development",
        first_runtime["development"],
        first_evaluator["development"],
        release["output"],
        source,
    )
    runtime_sha = first["runtime_manifest_sha256"]
    evaluator_sha = first["evaluator_manifest_sha256"]
    second = write_split_manifests(
        "development",
        second_runtime["development"],
        second_evaluator["development"],
        release["output"],
        source,
    )
    assert second["runtime_manifest_sha256"] == runtime_sha
    assert second["evaluator_manifest_sha256"] == evaluator_sha
    assert file_sha256(Path(second["runtime_manifest"])) == runtime_sha
    assert second["recording_macro_balanced"] is True


@pytest.mark.skipif(
    not (REAL_OUTER / "data_info_release.csv").is_file(),
    reason="EgoBody metadata is not present",
)
def test_actual_release_metadata_and_calibration_contract() -> None:
    recordings = load_recording_metadata(
        REAL_OUTER / "data_info_release.csv", REAL_OUTER / "data_splits.csv"
    )
    assert len(recordings) == 125
    assert {
        split: sum(row.protocol_split == split for row in recordings)
        for split in ("development", "holdout", "test")
    } == {"development": 65, "holdout": 17, "test": 43}
    three_camera = next(
        row
        for row in recordings
        if not (
            REAL_METADATA
            / "calibrations"
            / row.recording
            / "cal_trans/kinect_14to12_color.json"
        ).is_file()
    )
    five_camera = next(
        row
        for row in recordings
        if (
            REAL_METADATA
            / "calibrations"
            / row.recording
            / "cal_trans/kinect_14to12_color.json"
        ).is_file()
    )
    assert len(
        load_recording_calibrations(
            three_camera,
            REAL_METADATA / "calibrations",
            REAL_METADATA / "kinect_cam_params",
        )
    ) == 3
    assert len(
        load_recording_calibrations(
            five_camera,
            REAL_METADATA / "calibrations",
            REAL_METADATA / "kinect_cam_params",
        )
    ) == 5
