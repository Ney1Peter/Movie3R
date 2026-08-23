#!/usr/bin/env python3
"""Frozen EgoBody metadata, calibration, and CS150 audit utilities.

This module is evaluator/audit-only.  Runtime code must consume the separate
runtime manifest produced by :mod:`build_manifest` and must not import this
module: it opens official split metadata and metric camera calibration.

EgoBody names Kinect 12 as the master color camera.  Per-recording files named
``kinect_<id>to12_color.json`` map the named camera's color coordinates into
Kinect-12 color coordinates, while ``kinect12_to_world/<scene>.json`` maps
Kinect-12 color coordinates into the metric scene world.  Consequently,

    T_world_from_camera = T_world_from_12 @ T_12_from_camera.

The explicit physical-camera/intrinsic-directory mapping below follows the
official release naming: 12 is ``kinect_master`` and 11/13/14/15 are
``kinect_sub_1``/``sub_2``/``sub_3``/``sub_4`` respectively.  Every transform,
intrinsic matrix, and composed camera pose is validated before a case can enter
the manifest.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


PROTOCOL_NAME = "Bridge3R-EgoBody-CS150-v1"
PROTOCOL_SEED = 20260821
FPS = 30.0
CLIP_LENGTH = 150
PRE_COUNT = 75
POST_COUNT = 75
BOUNDARY_INDEX = PRE_COUNT
CASES_PER_RECORDING = 3

OFFICIAL_TO_PROTOCOL_SPLIT = {
    "train": "development",
    "val": "holdout",
    "test": "test",
}

# This is an explicit release mapping, not an order inferred from directory
# iteration.  Keeping it here makes a wrong physical-camera mapping fail fast.
KINECT_ID_TO_INTRINSIC_ROLE = {
    11: "kinect_sub_1",
    12: "kinect_master",
    13: "kinect_sub_2",
    14: "kinect_sub_3",
    15: "kinect_sub_4",
}

PAIR_STRATA = ("small", "medium", "extreme")
_RECORDING_RE = re.compile(
    r"^recording_(?P<date>\d{8})_(?P<subject0>S\d+)_(?P<subject1>S\d+)_(?P<take>\d+)$"
)
_BODY_RE = re.compile(r"^\s*(?P<index>\d+)\s+(?P<gender>\S+)\s*$")
_CAMERA_TRANSFORM_RE = re.compile(r"^kinect_(?P<camera>\d+)to12_color\.json$")


@dataclass(frozen=True)
class BodyDescriptor:
    index: int
    gender: str


@dataclass(frozen=True)
class RecordingInfo:
    recording: str
    scene: str
    start_frame: int
    end_frame: int
    body0: BodyDescriptor
    body1: BodyDescriptor
    fpv_body: BodyDescriptor
    subject0: str
    subject1: str
    official_split: str
    protocol_split: str

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame + 1

    @property
    def fpv_subject(self) -> str:
        return (self.subject0, self.subject1)[self.fpv_body.index]

    @property
    def interactee_subject(self) -> str:
        return (self.subject0, self.subject1)[1 - self.fpv_body.index]


@dataclass(frozen=True)
class CameraCalibration:
    name: str
    physical_id: int
    intrinsic_role: str
    camera_to_world: np.ndarray
    world_to_camera: np.ndarray
    intrinsic: np.ndarray
    distortion: np.ndarray
    camera_to_master: np.ndarray
    master_to_world: np.ndarray
    transform_source: str
    world_source: str
    intrinsic_source: str


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def value_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_body(value: str, field: str, recording: str) -> BodyDescriptor:
    match = _BODY_RE.fullmatch(str(value))
    if match is None:
        raise ValueError(f"Invalid {field} for {recording}: {value!r}")
    return BodyDescriptor(int(match.group("index")), match.group("gender"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def official_split_map(path: Path) -> dict[str, str]:
    """Map every official recording to exactly one train/val/test split."""

    rows = _read_csv(path)
    output: dict[str, str] = {}
    for row in rows:
        unknown = set(row) - set(OFFICIAL_TO_PROTOCOL_SPLIT)
        if unknown:
            raise ValueError(f"Unexpected official split columns: {sorted(unknown)}")
        for official in OFFICIAL_TO_PROTOCOL_SPLIT:
            recording = str(row.get(official, "")).strip()
            if not recording:
                continue
            if recording in output:
                raise ValueError(
                    f"Recording appears in multiple official splits: {recording}"
                )
            output[recording] = official
    if not output:
        raise ValueError(f"No official split recordings in {path}")
    return output


def load_recording_metadata(
    data_info_path: Path, data_splits_path: Path
) -> list[RecordingInfo]:
    """Load and cross-validate the official release metadata and split files."""

    split_by_recording = official_split_map(data_splits_path)
    rows = _read_csv(data_info_path)
    required = {
        "scene_name",
        "body_idx_0",
        "body_idx_1",
        "start_frame",
        "end_frame",
        "body_idx_fpv",
        "recording_name",
    }
    values: list[RecordingInfo] = []
    seen = set()
    for row in rows:
        missing = required - set(row)
        if missing:
            raise ValueError(f"Missing data-info columns: {sorted(missing)}")
        recording = str(row["recording_name"]).strip()
        if recording in seen:
            raise ValueError(f"Duplicate data-info recording: {recording}")
        seen.add(recording)
        match = _RECORDING_RE.fullmatch(recording)
        if match is None:
            raise ValueError(f"Unexpected EgoBody recording name: {recording}")
        if recording not in split_by_recording:
            raise ValueError(f"Recording absent from official splits: {recording}")
        body0 = _parse_body(row["body_idx_0"], "body_idx_0", recording)
        body1 = _parse_body(row["body_idx_1"], "body_idx_1", recording)
        fpv = _parse_body(row["body_idx_fpv"], "body_idx_fpv", recording)
        if {body0.index, body1.index} != {0, 1}:
            raise ValueError(f"Expected body indices 0/1 for {recording}")
        if fpv.index not in {0, 1}:
            raise ValueError(f"Invalid FPV body index for {recording}: {fpv.index}")
        official = split_by_recording[recording]
        start, end = int(row["start_frame"]), int(row["end_frame"])
        if end < start or end - start + 1 < CLIP_LENGTH:
            raise ValueError(
                f"Recording {recording} has {end - start + 1} frames; need {CLIP_LENGTH}"
            )
        values.append(
            RecordingInfo(
                recording=recording,
                scene=str(row["scene_name"]).strip(),
                start_frame=start,
                end_frame=end,
                body0=body0,
                body1=body1,
                fpv_body=fpv,
                subject0=match.group("subject0"),
                subject1=match.group("subject1"),
                official_split=official,
                protocol_split=OFFICIAL_TO_PROTOCOL_SPLIT[official],
            )
        )
    extra = set(split_by_recording) - seen
    if extra:
        raise ValueError(f"Official split recordings absent from data info: {sorted(extra)}")
    return sorted(values, key=lambda value: value.recording)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_rigid(transform: np.ndarray, label: str, atol: float = 5e-3) -> np.ndarray:
    value = np.asarray(transform, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError(f"Invalid rigid transform {label}: shape={value.shape}")
    if not np.allclose(value[3], [0.0, 0.0, 0.0, 1.0], atol=atol):
        raise ValueError(f"Invalid homogeneous row in {label}")
    rotation = value[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=atol):
        raise ValueError(f"Non-orthonormal rotation in {label}")
    determinant = float(np.linalg.det(rotation))
    if not math.isclose(determinant, 1.0, rel_tol=0.0, abs_tol=atol):
        raise ValueError(f"Improper rotation in {label}: det={determinant}")
    inverse = np.linalg.inv(value)
    if not np.allclose(value @ inverse, np.eye(4), atol=1e-8):
        raise ValueError(f"Non-invertible rigid transform {label}")
    return value


def _load_transform(path: Path) -> np.ndarray:
    payload = _load_json(path)
    if set(payload) != {"trans"}:
        raise ValueError(f"Unexpected transform payload fields in {path}: {sorted(payload)}")
    return validate_rigid(np.asarray(payload["trans"], dtype=np.float64), str(path))


def _load_intrinsic(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = _load_json(path)
    intrinsic = np.asarray(payload.get("camera_mtx"), dtype=np.float64)
    distortion = np.asarray(payload.get("k"), dtype=np.float64)
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError(f"Invalid color intrinsic in {path}")
    if distortion.ndim != 1 or len(distortion) < 4 or not np.isfinite(distortion).all():
        raise ValueError(f"Invalid color distortion in {path}")
    if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0 or not np.allclose(
        intrinsic[2], [0.0, 0.0, 1.0], atol=1e-8
    ):
        raise ValueError(f"Invalid focal/homogeneous intrinsic in {path}")
    return intrinsic, distortion


def load_recording_calibrations(
    recording: RecordingInfo,
    calibrations_root: Path,
    kinect_params_root: Path,
) -> dict[str, CameraCalibration]:
    """Load validated metric color-camera poses for one recording."""

    trans_root = calibrations_root / recording.recording / "cal_trans"
    if not trans_root.is_dir():
        raise FileNotFoundError(trans_root)
    world_path = trans_root / "kinect12_to_world" / f"{recording.scene}.json"
    if not world_path.is_file():
        alternatives = sorted((trans_root / "kinect12_to_world").glob("*.json"))
        if len(alternatives) == 1:
            raise FileNotFoundError(
                f"Scene-calibration mismatch for {recording.recording}: expected {world_path}, "
                f"found {alternatives[0]}"
            )
        raise FileNotFoundError(world_path)
    master_to_world = _load_transform(world_path)
    camera_transforms: dict[int, tuple[np.ndarray, Path]] = {
        12: (np.eye(4, dtype=np.float64), world_path)
    }
    for path in sorted(trans_root.glob("kinect_*to12_color.json")):
        match = _CAMERA_TRANSFORM_RE.fullmatch(path.name)
        if match is None:
            continue
        camera_id = int(match.group("camera"))
        if camera_id not in KINECT_ID_TO_INTRINSIC_ROLE:
            raise ValueError(f"Unsupported physical Kinect ID {camera_id} in {path}")
        if camera_id in camera_transforms:
            raise ValueError(f"Duplicate calibration for Kinect {camera_id}")
        camera_transforms[camera_id] = (_load_transform(path), path)
    if len(camera_transforms) not in {3, 5}:
        raise ValueError(
            f"Expected 3 or 5 calibrated Kinect color cameras for {recording.recording}; "
            f"got {sorted(camera_transforms)}"
        )
    expected = {11, 12, 13} if len(camera_transforms) == 3 else {11, 12, 13, 14, 15}
    if set(camera_transforms) != expected:
        raise ValueError(
            f"Non-official camera set for {recording.recording}: {sorted(camera_transforms)}"
        )

    output: dict[str, CameraCalibration] = {}
    for camera_id, (camera_to_master, transform_path) in sorted(camera_transforms.items()):
        role = KINECT_ID_TO_INTRINSIC_ROLE[camera_id]
        intrinsic_path = kinect_params_root / role / "Color.json"
        if not intrinsic_path.is_file():
            raise FileNotFoundError(intrinsic_path)
        intrinsic, distortion = _load_intrinsic(intrinsic_path)
        camera_to_world = validate_rigid(
            master_to_world @ camera_to_master,
            f"{recording.recording}:world_from_kinect_{camera_id}",
        )
        world_to_camera = validate_rigid(
            np.linalg.inv(camera_to_world),
            f"{recording.recording}:kinect_{camera_id}_from_world",
        )
        if not np.allclose(camera_to_world @ world_to_camera, np.eye(4), atol=1e-8):
            raise ValueError(f"Camera roundtrip failed for Kinect {camera_id}")
        name = f"kinect_{camera_id}"
        output[name] = CameraCalibration(
            name=name,
            physical_id=camera_id,
            intrinsic_role=role,
            camera_to_world=camera_to_world,
            world_to_camera=world_to_camera,
            intrinsic=intrinsic,
            distortion=distortion,
            camera_to_master=camera_to_master,
            master_to_world=master_to_world,
            transform_source=(
                "identity_master_kinect_12"
                if camera_id == 12
                else str(transform_path.resolve())
            ),
            world_source=str(world_path.resolve()),
            intrinsic_source=str(intrinsic_path.resolve()),
        )
    return output


def rotation_span_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first)[:3, :3].T @ np.asarray(second)[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _directed_pair(recording: str, first: str, second: str) -> tuple[str, str]:
    digest = hashlib.sha256(
        f"{PROTOCOL_SEED}:{recording}:{first}:{second}".encode("utf-8")
    ).digest()
    return (first, second) if digest[0] % 2 == 0 else (second, first)


def select_balanced_camera_pairs(
    recording: str,
    calibrations: Mapping[str, CameraCalibration],
) -> list[dict[str, Any]]:
    """Select exactly three angle-spread pairs for recording-macro balance."""

    names = sorted(calibrations)
    pairs = []
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            first_pose = calibrations[first].camera_to_world
            second_pose = calibrations[second].camera_to_world
            pairs.append(
                {
                    "first": first,
                    "second": second,
                    "rotation_span_deg": rotation_span_deg(first_pose, second_pose),
                    "camera_center_baseline_m": float(
                        np.linalg.norm(first_pose[:3, 3] - second_pose[:3, 3])
                    ),
                }
            )
    pairs.sort(
        key=lambda row: (
            row["rotation_span_deg"],
            row["camera_center_baseline_m"],
            row["first"],
            row["second"],
        )
    )
    if len(pairs) < CASES_PER_RECORDING:
        raise ValueError(f"Only {len(pairs)} camera pairs for {recording}")
    target_indices = (0, (len(pairs) - 1) // 2, len(pairs) - 1)
    if len(set(target_indices)) != CASES_PER_RECORDING:
        raise AssertionError(target_indices)
    selected = []
    for stratum, pair_index in zip(PAIR_STRATA, target_indices):
        row = dict(pairs[pair_index])
        pre, post = _directed_pair(recording, row.pop("first"), row.pop("second"))
        selected.append(
            {
                "angle_stratum": stratum,
                "pre_camera": pre,
                "post_camera": post,
                "selection_rank": int(pair_index),
                "available_pair_count": len(pairs),
                **row,
            }
        )
    return selected


def centered_clip_frames(recording: RecordingInfo) -> list[int]:
    offset = (recording.frame_count - CLIP_LENGTH) // 2
    first = recording.start_frame + offset
    frames = list(range(first, first + CLIP_LENGTH))
    if frames[-1] > recording.end_frame:
        raise AssertionError(recording)
    if frames[PRE_COUNT] != frames[PRE_COUNT - 1] + 1:
        raise AssertionError("The first camera-B frame must advance source time by one frame")
    return frames


def audit_recording(
    recording: RecordingInfo,
    calibrations_root: Path,
    kinect_params_root: Path,
) -> dict[str, Any]:
    calibrations = load_recording_calibrations(
        recording, calibrations_root, kinect_params_root
    )
    frames = centered_clip_frames(recording)
    pairs = select_balanced_camera_pairs(recording.recording, calibrations)
    return {
        "schema_version": "Bridge3R-v20-EgoBody-recording-audit-v1",
        "protocol": PROTOCOL_NAME,
        "protocol_seed": PROTOCOL_SEED,
        "recording": recording.recording,
        "scene": recording.scene,
        "official_split": recording.official_split,
        "protocol_split": recording.protocol_split,
        "metadata_frame_range": [recording.start_frame, recording.end_frame],
        "metadata_frame_count": recording.frame_count,
        "clip_frames": frames,
        "pre_frame_numbers": frames[:PRE_COUNT],
        "post_frame_numbers": frames[PRE_COUNT:],
        "boundary_index": BOUNDARY_INDEX,
        "boundary_source_step": frames[PRE_COUNT] - frames[PRE_COUNT - 1],
        "fps": FPS,
        "subjects": [recording.subject0, recording.subject1],
        "fpv_subject": recording.fpv_subject,
        "interactee_subject": recording.interactee_subject,
        "bodies": [
            {"index": recording.body0.index, "gender": recording.body0.gender},
            {"index": recording.body1.index, "gender": recording.body1.gender},
        ],
        "fpv_body_index": recording.fpv_body.index,
        "camera_count": len(calibrations),
        "cameras": {
            name: {
                "physical_id": value.physical_id,
                "intrinsic_role": value.intrinsic_role,
                "camera_to_world": value.camera_to_world,
                "world_to_camera": value.world_to_camera,
                "intrinsic": value.intrinsic,
                "distortion": value.distortion,
                "camera_to_master": value.camera_to_master,
                "master_to_world": value.master_to_world,
                "transform_source": value.transform_source,
                "world_source": value.world_source,
                "intrinsic_source": value.intrinsic_source,
            }
            for name, value in sorted(calibrations.items())
        },
        "selected_camera_pairs": pairs,
        "camera_chain_contract": (
            "world_from_camera = world_from_kinect12 @ kinect12_from_camera_color"
        ),
        "recording_macro_weight_contract": (
            "exactly three camera pairs per official recording"
        ),
        "selection_depends_on_model_result": False,
        "runtime_may_read_audit": False,
    }


def split_counts(recordings: Iterable[RecordingInfo]) -> dict[str, int]:
    values = list(recordings)
    return {
        split: sum(row.protocol_split == split for row in values)
        for split in ("development", "holdout", "test")
    }
