#!/usr/bin/env python3
"""Build isolated, reproducible EgoBody GT caches for the v20 evaluator.

This is an evaluator-only/offline program.  In particular,
``run_egobody_case.py`` must never import it: this module opens official split
metadata, calibration, and SMPL-X ground truth archives.

The official SMPL-X parameters are expressed in Kinect-12 (``master``) color
coordinates.  We render the gendered 10,475-vertex bodies there, transform
them with ``T_world_from_kinect12``, transfer the surface to the common
SMPL-6890 topology, and regress the frozen SMPL 24 joints.  A camera pose is
composed independently as::

    T_world_from_camera = (
        T_world_from_kinect12 @ T_kinect12_from_camera_color
    )

Dataset PKL files are decoded by an allow-list unpickler.  The two repository
body-topology PKLs use a separate, still restricted loader that additionally
permits only the known SciPy CSC and Chumpy container classes required by the
frozen SMPL asset.  No general ``pickle.load`` is used in this file.
"""

from __future__ import annotations

import argparse
import codecs
import copyreg
import csv
import hashlib
import io
import json
import math
import os
import pickle
import re
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = REPO_ROOT.parent
DEFAULT_OUTER_ROOT = WORKSPACE_ROOT / "data/EgoBody_work_v20/outer"
DEFAULT_MODEL_ROOT = REPO_ROOT / "src/models"

SCHEMA = "Bridge3R-EgoBody-CS150-ground-truth-v1"
MIN_VISIBLE_VERTEX_FRACTION = 0.01
DEFAULT_IMAGE_WIDTH = 1920
DEFAULT_IMAGE_HEIGHT = 1080
MAX_PARAMETER_PICKLE_BYTES = 1 << 20

PROTOCOL_SPLIT_TO_OFFICIAL = {
    "development": "train",
    "holdout": "val",
    "test": "test",
    "train": "train",
    "val": "val",
}
CAMERA_TO_PHYSICAL_ID = {
    "master": 12,
    "sub_1": 11,
    "sub_2": 13,
    "sub_3": 14,
    "sub_4": 15,
    "kinect_11": 11,
    "kinect_12": 12,
    "kinect_13": 13,
    "kinect_14": 14,
    "kinect_15": 15,
}
PHYSICAL_ID_TO_ROLE = {
    11: "sub_1",
    12: "master",
    13: "sub_2",
    14: "sub_3",
    15: "sub_4",
}
PHYSICAL_ID_TO_INTRINSIC_DIR = {
    camera: f"kinect_{role}" for camera, role in PHYSICAL_ID_TO_ROLE.items()
}
CASE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
BODY_VALUE_RE = re.compile(r"^\s*([01])\s+(\S+)\s*$")

PARAMETER_SHAPES = {
    "betas": (10,),
    "global_orient": (3,),
    "transl": (3,),
    "body_pose": (63,),
    "left_hand_pose": (12,),
    "right_hand_pose": (12,),
    "jaw_pose": (3,),
    "leye_pose": (3,),
    "reye_pose": (3,),
    "expression": (10,),
}
IGNORED_PARAMETER_KEYS = {
    "pose_embedding",
    "camera_rotation",
    "camera_translation",
    "gender",
}


def jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
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


_HASH_CACHE: dict[Path, str] = {}


def file_sha256(path: Path) -> str:
    resolved = path.resolve()
    cached = _HASH_CACHE.get(resolved)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    value = digest.hexdigest()
    _HASH_CACHE[resolved] = value
    return value


def value_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _numpy_pickle_globals() -> dict[tuple[str, str], Any]:
    reconstruct = np.core.multiarray._reconstruct
    return {
        ("numpy.core.multiarray", "_reconstruct"): reconstruct,
        ("numpy._core.multiarray", "_reconstruct"): reconstruct,
        ("numpy", "ndarray"): np.ndarray,
        ("numpy", "dtype"): np.dtype,
        ("_codecs", "encode"): codecs.encode,
    }


class NumpyRestrictedUnpickler(pickle.Unpickler):
    """Decode plain NumPy dictionaries without permitting arbitrary globals."""

    ALLOWED = _numpy_pickle_globals()

    def find_class(self, module: str, name: str) -> Any:
        value = self.ALLOWED.get((module, name))
        if value is None:
            raise pickle.UnpicklingError(
                f"forbidden pickle global {module}.{name}"
            )
        return value


class TopologyRestrictedUnpickler(NumpyRestrictedUnpickler):
    """Restricted decoder for the two fixed repository SMPL assets."""

    def find_class(self, module: str, name: str) -> Any:
        value = self.ALLOWED.get((module, name))
        if value is not None:
            return value
        # Imported lazily so the basic self-test does not require the model env.
        import chumpy
        from scipy.sparse import csc_matrix

        extra = {
            ("copy_reg", "_reconstructor"): copyreg._reconstructor,
            ("copyreg", "_reconstructor"): copyreg._reconstructor,
            ("scipy.sparse.csc", "csc_matrix"): csc_matrix,
            ("scipy.sparse._csc", "csc_matrix"): csc_matrix,
            ("__builtin__", "object"): object,
            ("builtins", "object"): object,
            ("__builtin__", "set"): set,
            ("builtins", "set"): set,
            ("chumpy.ch", "Ch"): chumpy.Ch,
        }
        value = extra.get((module, name))
        if value is None:
            raise pickle.UnpicklingError(
                f"forbidden topology pickle global {module}.{name}"
            )
        return value


def restricted_load(
    handle: BinaryIO, *, topology_asset: bool = False
) -> Any:
    loader = TopologyRestrictedUnpickler if topology_asset else NumpyRestrictedUnpickler
    return loader(handle, fix_imports=True, encoding="latin1").load()


def validate_rigid(transform: np.ndarray, label: str) -> np.ndarray:
    value = np.asarray(transform, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError(f"invalid transform {label}: {value.shape}")
    if not np.allclose(value[3], [0.0, 0.0, 0.0, 1.0], atol=5e-3):
        raise ValueError(f"invalid homogeneous row in {label}")
    rotation = value[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=5e-3):
        raise ValueError(f"non-orthonormal rotation in {label}")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=5e-3):
        raise ValueError(f"improper rotation in {label}")
    return value


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    value = np.asarray(points)
    matrix = np.asarray(transform)
    return value @ matrix[:3, :3].T + matrix[:3, 3]


@dataclass(frozen=True)
class RecordingInfo:
    recording: str
    scene: str
    start_frame: int
    end_frame: int
    body_genders: tuple[str, str]
    interactee_index: int
    official_split: str


class ReleaseIndex:
    def __init__(self, outer_root: Path):
        self.outer_root = outer_root.resolve()
        self.data_info_path = self.outer_root / "data_info_release.csv"
        self.data_splits_path = self.outer_root / "data_splits.csv"
        for path in (self.data_info_path, self.data_splits_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        split_by_recording: dict[str, str] = {}
        with self.data_splits_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            for row in csv.DictReader(handle):
                for split in ("train", "val", "test"):
                    recording = str(row.get(split, "")).strip()
                    if not recording:
                        continue
                    if recording in split_by_recording:
                        raise ValueError(f"duplicate split recording {recording}")
                    split_by_recording[recording] = split
        values: dict[str, RecordingInfo] = {}
        with self.data_info_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            for row in csv.DictReader(handle):
                recording = str(row["recording_name"]).strip()
                if recording in values or recording not in split_by_recording:
                    raise ValueError(f"invalid metadata recording {recording}")
                body_values = []
                for field in ("body_idx_0", "body_idx_1"):
                    match = BODY_VALUE_RE.fullmatch(str(row[field]))
                    if match is None or int(match.group(1)) != len(body_values):
                        raise ValueError(f"invalid {field} for {recording}")
                    body_values.append(match.group(2).lower())
                fpv = BODY_VALUE_RE.fullmatch(str(row["body_idx_fpv"]))
                if fpv is None:
                    raise ValueError(f"invalid body_idx_fpv for {recording}")
                interactee = int(fpv.group(1))
                if fpv.group(2).lower() != body_values[interactee]:
                    raise ValueError(f"FPV body gender mismatch for {recording}")
                values[recording] = RecordingInfo(
                    recording=recording,
                    scene=str(row["scene_name"]).strip(),
                    start_frame=int(row["start_frame"]),
                    end_frame=int(row["end_frame"]),
                    body_genders=(body_values[0], body_values[1]),
                    interactee_index=interactee,
                    official_split=split_by_recording[recording],
                )
        if set(values) != set(split_by_recording):
            raise ValueError("data_info_release.csv and data_splits.csv disagree")
        self.recordings = values

    def get(self, name: str) -> RecordingInfo:
        try:
            return self.recordings[name]
        except KeyError as error:
            raise KeyError(f"unknown EgoBody recording {name}") from error


class ArchivePool:
    """Open evaluator-only ZIPs lazily and perform O(1) member lookup."""

    def __init__(self, outer_root: Path):
        self.outer_root = outer_root.resolve()
        self._archives: dict[str, zipfile.ZipFile] = {}

    def archive_path(self, name: str) -> Path:
        path = self.outer_root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def get(self, name: str) -> zipfile.ZipFile:
        value = self._archives.get(name)
        if value is None:
            value = zipfile.ZipFile(self.archive_path(name), "r")
            self._archives[name] = value
        return value

    def read_json(self, archive: str, member: str) -> dict[str, Any]:
        handle = self.get(archive)
        try:
            info = handle.getinfo(member)
        except KeyError as error:
            raise FileNotFoundError(f"{archive}!/{member}") from error
        if info.is_dir() or info.file_size > (1 << 20):
            raise ValueError(f"unexpected JSON member {archive}!/{member}")
        return json.loads(handle.read(info).decode("utf-8"))

    def read_parameters(self, archive: str, member: str) -> dict[str, Any]:
        handle = self.get(archive)
        try:
            info = handle.getinfo(member)
        except KeyError as error:
            raise FileNotFoundError(f"{archive}!/{member}") from error
        if info.is_dir() or info.file_size > MAX_PARAMETER_PICKLE_BYTES:
            raise ValueError(
                f"unexpected parameter member size {archive}!/{member}: "
                f"{info.file_size}"
            )
        with handle.open(info, "r") as source:
            payload = restricted_load(source)
        if not isinstance(payload, dict):
            raise ValueError(f"parameter payload is not a dict: {archive}!/{member}")
        return payload

    def close(self) -> None:
        for archive in self._archives.values():
            archive.close()
        self._archives.clear()

    def __enter__(self) -> "ArchivePool":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


@dataclass(frozen=True)
class Camera:
    name: str
    physical_id: int
    camera_to_world: np.ndarray
    camera_to_master: np.ndarray
    master_to_world: np.ndarray
    intrinsic: np.ndarray
    distortion: np.ndarray
    transform_member: str
    world_member: str
    intrinsic_member: str


class CalibrationStore:
    def __init__(self, archives: ArchivePool):
        self.archives = archives
        self._cache: dict[tuple[str, str, str], Camera] = {}

    @staticmethod
    def _transform(payload: Mapping[str, Any], label: str) -> np.ndarray:
        if set(payload) != {"trans"}:
            raise ValueError(f"unexpected transform keys in {label}")
        return validate_rigid(np.asarray(payload["trans"], dtype=np.float64), label)

    def camera(self, recording: str, scene: str, camera_name: str) -> Camera:
        key = (recording, scene, camera_name)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        try:
            physical_id = CAMERA_TO_PHYSICAL_ID[camera_name]
        except KeyError as error:
            raise KeyError(f"unsupported EgoBody camera {camera_name}") from error
        world_member = (
            f"calibrations/{recording}/cal_trans/"
            f"kinect12_to_world/{scene}.json"
        )
        master_to_world = self._transform(
            self.archives.read_json("calibrations.zip", world_member), world_member
        )
        if physical_id == 12:
            camera_to_master = np.eye(4, dtype=np.float64)
            transform_member = "identity_master_kinect_12"
        else:
            transform_member = (
                f"calibrations/{recording}/cal_trans/"
                f"kinect_{physical_id}to12_color.json"
            )
            camera_to_master = self._transform(
                self.archives.read_json("calibrations.zip", transform_member),
                transform_member,
            )
        camera_to_world = validate_rigid(
            master_to_world @ camera_to_master,
            f"{recording}:world_from_kinect_{physical_id}",
        )
        intrinsic_dir = PHYSICAL_ID_TO_INTRINSIC_DIR[physical_id]
        intrinsic_member = f"kinect_cam_params/{intrinsic_dir}/Color.json"
        intrinsics = self.archives.read_json(
            "kinect_cam_params.zip", intrinsic_member
        )
        intrinsic = np.asarray(intrinsics.get("camera_mtx"), dtype=np.float64)
        distortion = np.asarray(intrinsics.get("k"), dtype=np.float64).reshape(-1)
        if (
            intrinsic.shape != (3, 3)
            or len(distortion) not in {4, 5, 8, 12, 14}
            or not np.isfinite(intrinsic).all()
            or not np.isfinite(distortion).all()
        ):
            raise ValueError(f"invalid intrinsic {intrinsic_member}")
        value = Camera(
            name=camera_name,
            physical_id=physical_id,
            camera_to_world=camera_to_world,
            camera_to_master=camera_to_master,
            master_to_world=master_to_world,
            intrinsic=intrinsic,
            distortion=distortion,
            transform_member=transform_member,
            world_member=world_member,
            intrinsic_member=intrinsic_member,
        )
        self._cache[key] = value
        return value


@dataclass
class CommonTopology:
    indices: np.ndarray
    weights: np.ndarray
    joint_regressor: np.ndarray
    transfer_path: Path
    smpl_model_path: Path

    @classmethod
    def load(cls, model_root: Path) -> "CommonTopology":
        transfer_path = model_root / "smplx/smplx2smpl.pkl"
        smpl_model_path = model_root / "smpl/SMPL_NEUTRAL.pkl"
        for path in (transfer_path, smpl_model_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        with transfer_path.open("rb") as handle:
            transfer_payload = restricted_load(handle)
        if not isinstance(transfer_payload, dict) or "matrix" not in transfer_payload:
            raise ValueError(f"invalid topology payload {transfer_path}")
        matrix = np.asarray(transfer_payload["matrix"])
        del transfer_payload
        if matrix.shape != (6890, 10475) or not np.isfinite(matrix).all():
            raise ValueError(f"invalid SMPL-X-to-SMPL matrix {matrix.shape}")
        rows, columns = np.nonzero(matrix)
        counts = np.bincount(rows, minlength=6890)
        if not len(rows) or np.any(counts == 0):
            raise ValueError("SMPL-X-to-SMPL transfer has empty output rows")
        width = int(counts.max())
        indices = np.full((6890, width), -1, dtype=np.int32)
        weights = np.zeros((6890, width), dtype=np.float32)
        offsets = np.zeros(6890, dtype=np.int32)
        for row, column in zip(rows.tolist(), columns.tolist()):
            slot = int(offsets[row])
            indices[row, slot] = column
            weights[row, slot] = float(matrix[row, column])
            offsets[row] += 1
        row_sums = weights.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=5e-4):
            raise ValueError("SMPL-X-to-SMPL transfer rows do not sum to one")
        del matrix, rows, columns, counts, offsets, row_sums

        with smpl_model_path.open("rb") as handle:
            model_payload = restricted_load(handle, topology_asset=True)
        if not isinstance(model_payload, dict) or "J_regressor" not in model_payload:
            raise ValueError(f"invalid SMPL model payload {smpl_model_path}")
        regressor = model_payload["J_regressor"]
        if hasattr(regressor, "toarray"):
            regressor = regressor.toarray()
        joint_regressor = np.asarray(regressor, dtype=np.float32)[:24]
        del model_payload, regressor
        if (
            joint_regressor.shape != (24, 6890)
            or not np.isfinite(joint_regressor).all()
        ):
            raise ValueError(f"invalid SMPL joint regressor {joint_regressor.shape}")
        return cls(
            indices=indices,
            weights=weights,
            joint_regressor=joint_regressor,
            transfer_path=transfer_path.resolve(),
            smpl_model_path=smpl_model_path.resolve(),
        )

    def to_smpl(self, vertices: np.ndarray) -> np.ndarray:
        value = np.asarray(vertices, dtype=np.float32)
        if value.ndim != 3 or value.shape[1:] != (10475, 3):
            raise ValueError(f"expected [B,10475,3], got {value.shape}")
        safe = np.maximum(self.indices, 0)
        gathered = value[:, safe, :]
        return (gathered * self.weights[None, :, :, None]).sum(axis=2)

    def joints(self, vertices: np.ndarray) -> np.ndarray:
        value = np.asarray(vertices, dtype=np.float32)
        if value.ndim != 3 or value.shape[1:] != (6890, 3):
            raise ValueError(f"expected [B,6890,3], got {value.shape}")
        return np.einsum("jv,bvc->bjc", self.joint_regressor, value)


class BodyModelPool:
    def __init__(self, model_root: Path, device: str, batch_size: int):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.model_root = model_root.resolve()
        self.device_name = device
        self.batch_size = batch_size
        self._models: dict[str, Any] = {}

    def model_path(self, gender: str) -> Path:
        path = self.model_root / f"smplx/SMPLX_{gender.upper()}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def get(self, gender: str) -> Any:
        gender = gender.lower()
        model = self._models.get(gender)
        if model is not None:
            return model
        import smplx

        self.model_path(gender)
        model = smplx.create(
            str(self.model_root),
            model_type="smplx",
            gender=gender,
            ext="npz",
            num_betas=10,
            num_expression_coeffs=10,
            num_pca_comps=12,
            use_pca=True,
            flat_hand_mean=False,
        ).to(self.device_name).eval()
        self._models[gender] = model
        return model

    def render(
        self, gender: str, parameters: Sequence[Mapping[str, np.ndarray]]
    ) -> np.ndarray:
        import torch

        model = self.get(gender)
        chunks = []
        with torch.no_grad():
            for first in range(0, len(parameters), self.batch_size):
                rows = parameters[first : first + self.batch_size]
                tensors = {
                    key: torch.from_numpy(
                        np.concatenate([np.asarray(row[key]) for row in rows], axis=0)
                    ).to(self.device_name)
                    for key in PARAMETER_SHAPES
                }
                output = model(return_verts=True, **tensors)
                vertices = output.vertices.detach().cpu().numpy().astype(
                    np.float32, copy=False
                )
                if vertices.shape != (len(rows), 10475, 3):
                    raise ValueError(f"unexpected SMPL-X output {vertices.shape}")
                if not np.isfinite(vertices).all():
                    raise ValueError("non-finite SMPL-X vertices")
                chunks.append(vertices)
        return np.concatenate(chunks, axis=0)


def validate_parameters(
    payload: Mapping[str, Any], expected_gender: str, source: str
) -> dict[str, np.ndarray]:
    unknown = set(payload) - set(PARAMETER_SHAPES) - IGNORED_PARAMETER_KEYS
    if unknown:
        raise ValueError(f"unexpected parameter keys in {source}: {sorted(unknown)}")
    missing = set(PARAMETER_SHAPES) - set(payload)
    if missing:
        raise ValueError(f"missing parameter keys in {source}: {sorted(missing)}")
    gender = str(payload.get("gender", expected_gender)).lower()
    if gender != expected_gender.lower():
        raise ValueError(
            f"parameter gender mismatch in {source}: {gender} vs {expected_gender}"
        )
    output = {}
    for key, tail_shape in PARAMETER_SHAPES.items():
        value = np.asarray(payload[key], dtype=np.float32)
        if value.shape != (1, *tail_shape) or not np.isfinite(value).all():
            raise ValueError(f"invalid {key} in {source}: {value.shape}")
        output[key] = value
    return output


def parameter_member(
    role: str, official_split: str, recording: str, body_index: int, frame: int
) -> tuple[str, str]:
    archive = f"smplx_{role}_{official_split}.zip"
    root = f"smplx_{role}_{official_split}"
    member = (
        f"{root}/{recording}/body_idx_{body_index}/results/"
        f"frame_{frame:05d}/000.pkl"
    )
    return archive, member


@dataclass
class Geometry:
    recording: str
    official_split: str
    scene: str
    frames: np.ndarray
    identities: tuple[str, str]
    genders: tuple[str, str]
    roles: tuple[str, str]
    vertices_world: np.ndarray
    joints_world: np.ndarray
    parameter_members: dict[str, list[str]]


def load_geometry(
    info: RecordingInfo,
    frames: Sequence[int],
    archives: ArchivePool,
    calibrations: CalibrationStore,
    topology: CommonTopology,
    models: BodyModelPool,
) -> Geometry:
    frame_values = np.asarray(frames, dtype=np.int64)
    if (
        frame_values.ndim != 1
        or len(frame_values) == 0
        or np.any(frame_values < info.start_frame)
        or np.any(frame_values > info.end_frame)
    ):
        raise ValueError(f"frames outside metadata range for {info.recording}")
    identities = ("body_idx_0", "body_idx_1")
    roles = tuple(
        "interactee" if index == info.interactee_index else "camera_wearer"
        for index in range(2)
    )
    master_to_world = calibrations.camera(
        info.recording, info.scene, "kinect_12"
    ).master_to_world
    vertices_world = np.empty((len(frames), 2, 6890, 3), dtype=np.float32)
    parameter_members: dict[str, list[str]] = {}
    for body_index in range(2):
        role = roles[body_index]
        archive_name = f"smplx_{role}_{info.official_split}.zip"
        rows = []
        member_values = []
        for frame in frame_values.tolist():
            archive, member = parameter_member(
                role, info.official_split, info.recording, body_index, int(frame)
            )
            if archive != archive_name:
                raise AssertionError(archive)
            payload = archives.read_parameters(archive, member)
            rows.append(
                validate_parameters(
                    payload, info.body_genders[body_index], f"{archive}!/{member}"
                )
            )
            member_values.append(member)
        master_vertices = models.render(info.body_genders[body_index], rows)
        smpl_master = topology.to_smpl(master_vertices)
        vertices_world[:, body_index] = transform_points(
            master_to_world, smpl_master
        ).astype(np.float32)
        parameter_members[archive_name] = member_values
    flat = vertices_world.reshape(-1, 6890, 3)
    joints_world = topology.joints(flat).reshape(len(frames), 2, 24, 3)
    if not np.isfinite(vertices_world).all() or not np.isfinite(joints_world).all():
        raise ValueError(f"non-finite world geometry for {info.recording}")
    return Geometry(
        recording=info.recording,
        official_split=info.official_split,
        scene=info.scene,
        frames=frame_values,
        identities=identities,
        genders=info.body_genders,
        roles=roles,
        vertices_world=vertices_world,
        joints_world=joints_world.astype(np.float32, copy=False),
        parameter_members=parameter_members,
    )


def camera_points(camera_to_world: np.ndarray, points_world: np.ndarray) -> np.ndarray:
    transform = np.asarray(camera_to_world, dtype=np.float64)
    points = np.asarray(points_world, dtype=np.float64)
    # Row-vector form of inv(c2w): (p_world - t_c2w) @ R_c2w.
    return (points - transform[:3, 3]) @ transform[:3, :3]


def distorted_pixels(points_camera: np.ndarray, camera: Camera) -> np.ndarray:
    value = np.asarray(points_camera, dtype=np.float64)
    normalized = value[:, :2] / np.maximum(value[:, 2:3], 1e-12)
    x, y = normalized[:, 0], normalized[:, 1]
    r2 = x * x + y * y
    coefficients = np.pad(camera.distortion, (0, max(0, 8 - len(camera.distortion))))
    k1, k2, p1, p2, k3, k4, k5, k6 = coefficients[:8]
    numerator = 1.0 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    denominator = 1.0 + k4 * r2 + k5 * r2**2 + k6 * r2**3
    radial = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=np.abs(denominator) > 1e-12,
    )
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    pixels = np.empty((len(value), 2), dtype=np.float64)
    pixels[:, 0] = camera.intrinsic[0, 0] * xd + camera.intrinsic[0, 2]
    pixels[:, 1] = camera.intrinsic[1, 1] * yd + camera.intrinsic[1, 2]
    return pixels


def visible_fraction(
    vertices_world: np.ndarray, camera: Camera, width: int, height: int
) -> float:
    points = camera_points(camera.camera_to_world, vertices_world)
    pixels = distorted_pixels(points, camera)
    visible = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(pixels).all(axis=1)
        & (points[:, 2] > 1e-6)
        & (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] < height)
    )
    return float(visible.mean())


def record_name(record: Mapping[str, Any]) -> str:
    for key in ("recording", "recording_name", "capture"):
        value = str(record.get(key, "")).strip()
        if value:
            return value
    raise KeyError("manifest row has no recording/recording_name/capture")


def validate_record(record: Mapping[str, Any], index: ReleaseIndex) -> RecordingInfo:
    required = {
        "case_id",
        "pre_camera",
        "post_camera",
        "pre_frame_numbers",
        "post_frame_numbers",
        "boundary_index",
    }
    missing = required - set(record)
    if missing:
        raise ValueError(f"manifest row misses {sorted(missing)}")
    case_id = str(record["case_id"])
    if CASE_ID_RE.fullmatch(case_id) is None:
        raise ValueError(f"unsafe case_id {case_id!r}")
    info = index.get(record_name(record))
    scene = str(record.get("scene_name", info.scene))
    if scene != info.scene:
        raise ValueError(f"scene mismatch for {info.recording}: {scene} vs {info.scene}")
    official = str(record.get("official_split", info.official_split))
    if official != info.official_split:
        raise ValueError(
            f"official split mismatch for {info.recording}: {official} vs {info.official_split}"
        )
    protocol_split = str(record.get("split", ""))
    if protocol_split:
        expected = PROTOCOL_SPLIT_TO_OFFICIAL.get(protocol_split)
        if expected is None or expected != info.official_split:
            raise ValueError(
                f"protocol split mismatch for {info.recording}: {protocol_split}"
            )
    pre = [int(value) for value in record["pre_frame_numbers"]]
    post = [int(value) for value in record["post_frame_numbers"]]
    if not pre or not post or int(record["boundary_index"]) != len(pre):
        raise ValueError(f"invalid boundary for {case_id}")
    if any(second != first + 1 for first, second in zip(pre + post, (pre + post)[1:])):
        raise ValueError(f"non-consecutive source frames for {case_id}")
    for camera in (str(record["pre_camera"]), str(record["post_camera"])):
        if camera not in CAMERA_TO_PHYSICAL_ID:
            raise ValueError(f"unsupported camera {camera} in {case_id}")
    descriptors = record.get("body_descriptors_evaluator_only")
    if descriptors is not None:
        expected = [
            {"index": index_value, "gender": info.body_genders[index_value]}
            for index_value in range(2)
        ]
        normalized = [
            {"index": int(row["index"]), "gender": str(row["gender"]).lower()}
            for row in descriptors
        ]
        if normalized != expected:
            raise ValueError(f"body descriptor mismatch for {case_id}")
    return info


def manifest_calibration_check(record: Mapping[str, Any], camera: Camera) -> None:
    manifest = record.get("camera_calibration_evaluator_only")
    if manifest is None:
        return
    payload = manifest.get(camera.name)
    if payload is None:
        raise ValueError(f"manifest calibration misses {camera.name}")
    expected = {
        "camera_to_world": camera.camera_to_world,
        "camera_to_master": camera.camera_to_master,
        "master_to_world": camera.master_to_world,
        "intrinsic": camera.intrinsic,
        "distortion": camera.distortion,
    }
    for key, value in expected.items():
        observed = np.asarray(payload[key], dtype=np.float64)
        if observed.shape != value.shape or not np.allclose(observed, value, atol=1e-8):
            raise ValueError(
                f"manifest/official calibration mismatch for {camera.name}:{key}"
            )


def cache_is_complete(
    npz_path: Path,
    json_path: Path,
    case_id: str,
    record: Mapping[str, Any] | None = None,
) -> bool:
    if not npz_path.is_file() or not json_path.is_file():
        return False
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        return (
            payload.get("schema_version") == SCHEMA
            and payload.get("case_id") == case_id
            and (
                record is None
                or payload.get("record_sha256") == value_sha256(record)
            )
            and payload.get("output", {}).get("npz_sha256") == file_sha256(npz_path)
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temp_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        _HASH_CACHE.pop(path.resolve(), None)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = (
        json.dumps(jsonable(payload), sort_keys=True, indent=2, ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temp_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        _HASH_CACHE.pop(path.resolve(), None)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def build_case(
    record: Mapping[str, Any],
    geometry: Geometry,
    archives: ArchivePool,
    calibrations: CalibrationStore,
    topology: CommonTopology,
    models: BodyModelPool,
    manifest_path: Path,
    output_root: Path,
    width: int,
    height: int,
    min_visible_fraction: float,
    overwrite: bool,
) -> dict[str, Any]:
    case_id = str(record["case_id"])
    npz_path = output_root / f"{case_id}.gt.npz"
    json_path = output_root / f"{case_id}.gt.json"
    complete = cache_is_complete(npz_path, json_path, case_id, record)
    if complete and not overwrite:
        return {"case_id": case_id, "status": "skipped_complete", "npz": str(npz_path)}
    if not overwrite and npz_path.exists() and json_path.exists() and not complete:
        raise RuntimeError(
            f"existing GT pair is invalid for {case_id}; use --overwrite to replace it"
        )
    pre_frames = [int(value) for value in record["pre_frame_numbers"]]
    post_frames = [int(value) for value in record["post_frame_numbers"]]
    frames = np.asarray(pre_frames + post_frames, dtype=np.int64)
    if not np.array_equal(frames, geometry.frames):
        raise ValueError(f"geometry frame mismatch for {case_id}")
    camera_names = (
        [str(record["pre_camera"])] * len(pre_frames)
        + [str(record["post_camera"])] * len(post_frames)
    )
    cameras_by_name = {
        name: calibrations.camera(geometry.recording, geometry.scene, name)
        for name in sorted(set(camera_names))
    }
    for camera in cameras_by_name.values():
        manifest_calibration_check(record, camera)
    cameras_c2w = np.stack(
        [cameras_by_name[name].camera_to_world for name in camera_names]
    ).astype(np.float64)
    fractions = np.empty((len(frames), 2), dtype=np.float32)
    for frame_index, camera_name in enumerate(camera_names):
        camera = cameras_by_name[camera_name]
        for body_index in range(2):
            fractions[frame_index, body_index] = visible_fraction(
                geometry.vertices_world[frame_index, body_index], camera, width, height
            )
    visible = fractions >= float(min_visible_fraction)
    arrays = {
        "cameras_c2w": cameras_c2w,
        "vertices_world": geometry.vertices_world,
        "joints_world": geometry.joints_world,
        "frames": frames,
        "visible_fraction": fractions,
        "visible": visible,
        "identities": np.asarray(geometry.identities, dtype="<U10"),
    }
    atomic_npz(npz_path, arrays)
    used_archive_paths = {
        name: archives.archive_path(name).resolve()
        for name in sorted(geometry.parameter_members)
    }
    source_archives = {
        name: {
            "path": path,
            "sha256": file_sha256(path),
            "member_count_used": len(geometry.parameter_members[name]),
            "members": geometry.parameter_members[name],
        }
        for name, path in used_archive_paths.items()
    }
    used_genders = sorted(set(geometry.genders))
    camera_sources = {
        name: {
            "physical_id": camera.physical_id,
            "camera_to_world": camera.camera_to_world,
            "camera_to_master": camera.camera_to_master,
            "master_to_world": camera.master_to_world,
            "transform_member": camera.transform_member,
            "world_member": camera.world_member,
            "intrinsic_member": camera.intrinsic_member,
        }
        for name, camera in cameras_by_name.items()
    }
    provenance = {
        "schema_version": SCHEMA,
        "case_id": case_id,
        "protocol": record.get("protocol"),
        "recording": geometry.recording,
        "scene_name": geometry.scene,
        "official_split": geometry.official_split,
        "protocol_split": record.get("split"),
        "record_sha256": value_sha256(record),
        "manifest": {
            "path": manifest_path.resolve(),
            "sha256": file_sha256(manifest_path),
        },
        "frames": frames,
        "boundary_index": int(record["boundary_index"]),
        "camera_names": camera_names,
        "identities": list(geometry.identities),
        "genders": list(geometry.genders),
        "roles": list(geometry.roles),
        "coordinate_contract": {
            "parameter_frame": "Kinect-12/master color camera, metric metres",
            "world_vertices": "T_world_from_kinect12 applied to SMPL-6890 vertices",
            "camera_c2w": (
                "T_world_from_kinect12 @ T_kinect12_from_camera_color"
            ),
            "manifest_calibration_verified_against_official_zip": True,
        },
        "topology": {
            "source": "gendered SMPL-X 10475",
            "target": "common SMPL 6890 / SMPL 24-joint regressor",
            "smplx_to_smpl": topology.transfer_path,
            "smplx_to_smpl_sha256": file_sha256(topology.transfer_path),
            "smpl_neutral": topology.smpl_model_path,
            "smpl_neutral_sha256": file_sha256(topology.smpl_model_path),
            "gendered_smplx_models": {
                gender: {
                    "path": models.model_path(gender),
                    "sha256": file_sha256(models.model_path(gender)),
                }
                for gender in used_genders
            },
        },
        "release_metadata": {
            "data_info_release": str((archives.outer_root / "data_info_release.csv").resolve()),
            "data_info_release_sha256": file_sha256(
                archives.outer_root / "data_info_release.csv"
            ),
            "data_splits": str((archives.outer_root / "data_splits.csv").resolve()),
            "data_splits_sha256": file_sha256(
                archives.outer_root / "data_splits.csv"
            ),
            "calibrations_zip_sha256": file_sha256(
                archives.archive_path("calibrations.zip")
            ),
            "kinect_cam_params_zip_sha256": file_sha256(
                archives.archive_path("kinect_cam_params.zip")
            ),
        },
        "source_parameter_archives": source_archives,
        "camera_sources": camera_sources,
        "visibility": {
            "definition": (
                "fraction of SMPL-6890 vertices with positive depth and distorted "
                "projection inside the Kinect color image"
            ),
            "image_width": width,
            "image_height": height,
            "threshold": float(min_visible_fraction),
            "visible_person_frames": int(visible.sum()),
        },
        "pickle_security": {
            "dataset_parameters": "NumPy-only allow-list unpickler",
            "repository_topology_assets": (
                "allow-list unpickler limited to NumPy, SciPy CSC, Chumpy Ch, "
                "copyreg reconstructor, object, and set"
            ),
            "general_pickle_load_used": False,
            "offline_evaluator_only": True,
        },
        "output": {
            "npz": npz_path.resolve(),
            "npz_sha256": file_sha256(npz_path),
            "arrays": {name: list(value.shape) for name, value in arrays.items()},
        },
        "created_unix_seconds": time.time(),
    }
    atomic_json(json_path, provenance)
    return {
        "case_id": case_id,
        "status": "written",
        "npz": str(npz_path.resolve()),
        "json": str(json_path.resolve()),
        "visible_person_frames": int(visible.sum()),
    }


def read_manifest(path: Path) -> list[dict[str, Any]]:
    value = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in value.splitlines() if line.strip()]
    else:
        payload = json.loads(value)
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict) and "case_id" in payload:
            rows = [payload]
        elif isinstance(payload, dict):
            for key in ("rows", "cases", "records"):
                if isinstance(payload.get(key), list):
                    rows = payload[key]
                    break
            else:
                raise ValueError(f"cannot find manifest rows in {path}")
        else:
            raise ValueError(f"unexpected manifest payload in {path}")
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"empty or invalid manifest {path}")
    case_ids = [str(row.get("case_id", "")) for row in rows]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError(f"duplicate case_id in {path}")
    return rows


def select_rows(
    rows: list[dict[str, Any]], case_ids: Sequence[str], lines: Sequence[int], limit: int
) -> list[dict[str, Any]]:
    selected = rows
    if case_ids:
        wanted = set(case_ids)
        selected = [row for row in selected if str(row.get("case_id")) in wanted]
        missing = wanted - {str(row.get("case_id")) for row in selected}
        if missing:
            raise KeyError(f"case ids absent from manifest: {sorted(missing)}")
    if lines:
        invalid = [line for line in lines if line < 1 or line > len(rows)]
        if invalid:
            raise IndexError(f"manifest lines out of range: {invalid}")
        line_ids = {str(rows[line - 1]["case_id"]) for line in lines}
        selected = [row for row in selected if str(row.get("case_id")) in line_ids]
    if limit > 0:
        selected = selected[:limit]
    if not selected:
        raise ValueError("no manifest rows selected")
    return selected


def synthetic_self_test() -> dict[str, Any]:
    malicious = b"cos\nsystem\n(S'false'\ntR."
    try:
        restricted_load(io.BytesIO(malicious))
    except pickle.UnpicklingError:
        rejected = True
    else:
        rejected = False
    if not rejected:
        raise AssertionError("restricted unpickler accepted os.system")
    first = np.eye(4)
    first[:3, 3] = [1.0, 2.0, 3.0]
    second = np.eye(4)
    second[:3, 3] = [-0.5, 0.25, 2.0]
    points = np.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    composed = first @ second
    np.testing.assert_allclose(
        transform_points(composed, points),
        transform_points(first, transform_points(second, points)),
    )
    recovered = camera_points(composed, transform_points(composed, points))
    np.testing.assert_allclose(recovered, points, atol=1e-12)
    return {
        "restricted_unpickler_rejected_os_system": rejected,
        "transform_composition": "passed",
        "c2w_roundtrip": "passed",
    }


def real_smoke(
    outer_root: Path,
    model_root: Path,
    device: str,
    recording_name_value: str | None,
) -> dict[str, Any]:
    index = ReleaseIndex(outer_root)
    info = (
        index.get(recording_name_value)
        if recording_name_value
        else index.recordings[sorted(index.recordings)[0]]
    )
    frame = info.start_frame
    with ArchivePool(outer_root) as archives:
        calibrations = CalibrationStore(archives)
        topology = CommonTopology.load(model_root)
        models = BodyModelPool(model_root, device, batch_size=1)
        geometry = load_geometry(
            info, [frame], archives, calibrations, topology, models
        )
        camera = calibrations.camera(info.recording, info.scene, "kinect_12")
        fractions = [
            visible_fraction(geometry.vertices_world[0, index_value], camera, 1920, 1080)
            for index_value in range(2)
        ]
    if geometry.vertices_world.shape != (1, 2, 6890, 3):
        raise AssertionError(geometry.vertices_world.shape)
    if geometry.joints_world.shape != (1, 2, 24, 3):
        raise AssertionError(geometry.joints_world.shape)
    return {
        "recording": info.recording,
        "official_split": info.official_split,
        "frame": frame,
        "genders": info.body_genders,
        "roles": geometry.roles,
        "vertices_shape": geometry.vertices_world.shape,
        "joints_shape": geometry.joints_world.shape,
        "master_visible_fraction": fractions,
        "finite": bool(
            np.isfinite(geometry.vertices_world).all()
            and np.isfinite(geometry.joints_world).all()
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--outer-root", type=Path, default=DEFAULT_OUTER_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--line", type=int, action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-width", type=int, default=DEFAULT_IMAGE_WIDTH)
    parser.add_argument("--image-height", type=int, default=DEFAULT_IMAGE_HEIGHT)
    parser.add_argument(
        "--min-visible-fraction", type=float, default=MIN_VISIBLE_VERTEX_FRACTION
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--real-smoke",
        nargs="?",
        const="",
        metavar="RECORDING",
        help="Render one real frame (auto-select recording when value is omitted)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        print(json.dumps(synthetic_self_test(), indent=2, ensure_ascii=False))
        return
    if args.real_smoke is not None:
        result = real_smoke(
            args.outer_root.resolve(),
            args.model_root.resolve(),
            args.device,
            args.real_smoke or None,
        )
        print(json.dumps(jsonable(result), indent=2, ensure_ascii=False))
        return
    if args.manifest is None or args.output_root is None:
        raise ValueError("--manifest and --output-root are required in batch mode")
    if args.image_width < 1 or args.image_height < 1:
        raise ValueError("image dimensions must be positive")
    if not 0.0 <= args.min_visible_fraction <= 1.0:
        raise ValueError("--min-visible-fraction must be in [0,1]")
    manifest_path = args.manifest.resolve()
    rows = select_rows(
        read_manifest(manifest_path), args.case_id, args.line, args.limit
    )
    outer_root = args.outer_root.resolve()
    output_root = args.output_root.resolve()
    results, errors = [], []
    pending = []
    # A JSON sidecar is the commit marker for its NPZ.  Verify both the NPZ
    # digest and the exact current manifest row before doing any expensive
    # topology/model work, so a resumed batch is genuinely cheap.
    for record in rows:
        case_id = str(record.get("case_id", ""))
        npz_path = output_root / f"{case_id}.gt.npz"
        json_path = output_root / f"{case_id}.gt.json"
        if (
            not args.overwrite
            and cache_is_complete(npz_path, json_path, case_id, record)
        ):
            results.append(
                {
                    "case_id": case_id,
                    "status": "skipped_complete",
                    "npz": str(npz_path),
                }
            )
        else:
            pending.append(record)
    if not pending:
        summary = {
            "schema_version": SCHEMA,
            "manifest": str(manifest_path),
            "selected_cases": len(rows),
            "written": 0,
            "skipped_complete": len(results),
            "errors": [],
            "results": results,
        }
        print(json.dumps(jsonable(summary), indent=2, ensure_ascii=False))
        return

    release = ReleaseIndex(outer_root)
    topology = CommonTopology.load(args.model_root.resolve())
    models = BodyModelPool(
        args.model_root.resolve(), args.device, int(args.batch_size)
    )
    # The three camera-pair cases for one recording share the same 150 source
    # frames.  Retaining one geometry entry avoids rendering them three times.
    geometry_key: tuple[str, tuple[int, ...]] | None = None
    geometry_value: Geometry | None = None
    with ArchivePool(outer_root) as archives:
        calibrations = CalibrationStore(archives)
        for record in pending:
            case_id = str(record.get("case_id", "<missing>"))
            try:
                info = validate_record(record, release)
                frames = tuple(
                    int(value)
                    for value in (
                        list(record["pre_frame_numbers"])
                        + list(record["post_frame_numbers"])
                    )
                )
                key = (info.recording, frames)
                if geometry_key != key or geometry_value is None:
                    geometry_value = load_geometry(
                        info,
                        frames,
                        archives,
                        calibrations,
                        topology,
                        models,
                    )
                    geometry_key = key
                results.append(
                    build_case(
                        record,
                        geometry_value,
                        archives,
                        calibrations,
                        topology,
                        models,
                        manifest_path,
                        output_root,
                        int(args.image_width),
                        int(args.image_height),
                        float(args.min_visible_fraction),
                        bool(args.overwrite),
                    )
                )
            except Exception as error:
                row = {
                    "case_id": case_id,
                    "error": f"{type(error).__name__}: {error}",
                }
                errors.append(row)
                if args.fail_fast:
                    raise
                print(json.dumps(row, ensure_ascii=False), file=sys.stderr)
    summary = {
        "schema_version": SCHEMA,
        "manifest": str(manifest_path),
        "selected_cases": len(rows),
        "written": sum(row["status"] == "written" for row in results),
        "skipped_complete": sum(
            row["status"] == "skipped_complete" for row in results
        ),
        "errors": errors,
        "results": results,
    }
    print(json.dumps(jsonable(summary), indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
