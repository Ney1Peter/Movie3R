#!/usr/bin/env python3
"""Convert native OnlineHMR outputs to the frozen BRIDGE3R prediction schema."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCHEMA = "Bridge3R-OnlineHMR-adapter-v1"
METHOD = "onlinehmr_official"
SCRIPT = Path(__file__).resolve()
WORKSPACE = SCRIPT.parents[4]
ONLINEHMR_REPO = WORKSPACE / "external_baselines/Video-OnlineHMR"
if str(ONLINEHMR_REPO) not in sys.path:
    sys.path.insert(0, str(ONLINEHMR_REPO))


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(partial, path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(jsonable(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def read_row(path: Path, line_number: int) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if line_number < 1 or line_number > len(rows):
        raise IndexError(f"line {line_number} outside manifest with {len(rows)} rows")
    row = rows[line_number - 1]
    if row.get("runtime_gt_access") is not False:
        raise ValueError("runtime row is not prediction-only")
    if len(row["image_members"]) != int(row["clip_length"]):
        raise ValueError("runtime RGB count differs from clip length")
    return row


def quaternion_matrix(values: np.ndarray) -> np.ndarray:
    """Convert normalized xyzw quaternions to rotation matrices."""

    q = np.asarray(values, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4 or not np.isfinite(q).all():
        raise ValueError("camera quaternion array must be finite N x 4")
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError("camera trajectory contains a zero quaternion")
    x, y, z, w = (q / norm).T
    output = np.empty((len(q), 3, 3), dtype=np.float64)
    output[:, 0, 0] = 1 - 2 * (y * y + z * z)
    output[:, 0, 1] = 2 * (x * y - z * w)
    output[:, 0, 2] = 2 * (x * z + y * w)
    output[:, 1, 0] = 2 * (x * y + z * w)
    output[:, 1, 1] = 1 - 2 * (x * x + z * z)
    output[:, 1, 2] = 2 * (y * z - x * w)
    output[:, 2, 0] = 2 * (x * z - y * w)
    output[:, 2, 1] = 2 * (y * z + x * w)
    output[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return output


def load_camera(path: Path, frame_count: int) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.loadtxt(path, dtype=np.float64)
    values = np.atleast_2d(values)
    if values.shape != (frame_count - 1, 8):
        raise ValueError(
            f"incremental trajectory shape {values.shape}, expected {(frame_count - 1, 8)}"
        )
    if not np.isfinite(values).all() or np.any(values[:, 0] <= 0):
        raise ValueError("incremental trajectory contains invalid values")

    # run_custom_mt fixes the metric scale when processing physical frame 2,
    # then uses that one value for all subsequent world-human transforms.
    # Row zero corresponds to frame 1 and row one to frame 2.
    scale_row = 1 if len(values) > 1 else 0
    fixed_scale = float(values[scale_row, 0])
    cameras = np.repeat(np.eye(4, dtype=np.float64)[None], frame_count, axis=0)
    cameras[1:, :3, :3] = quaternion_matrix(values[:, 4:8])
    cameras[1:, :3, 3] = fixed_scale * values[:, 1:4]
    rotation_residual = float(
        np.max(
            np.abs(
                cameras[:, :3, :3]
                @ np.swapaxes(cameras[:, :3, :3], -1, -2)
                - np.eye(3)
            )
        )
    )
    determinant_residual = float(
        np.max(np.abs(np.linalg.det(cameras[:, :3, :3]) - 1.0))
    )
    if rotation_residual > 1e-5 or determinant_residual > 1e-5:
        raise ValueError("camera trajectory rotations are not valid SO(3) matrices")
    return cameras.astype(np.float32), {
        "trajectory_rows": len(values),
        "frame_zero_convention": "MASt3R-SLAM identity initialization",
        "fixed_metric_scale": fixed_scale,
        "fixed_metric_scale_source_frame": scale_row + 1,
        "rotation_orthogonality_max_residual": rotation_residual,
        "rotation_determinant_max_residual": determinant_residual,
        "trajectory_semantics": "online incremental c2w used by OnlineHMR human-world composition",
    }


def native_label(path: Path, case_id: str) -> str:
    suffix = f"_{case_id}.npz"
    if not path.name.endswith(suffix):
        raise ValueError(f"unexpected OnlineHMR track filename: {path.name}")
    value = path.name[: -len(suffix)]
    if not value:
        raise ValueError(f"empty native track label in {path.name}")
    return value


def track_key(value: str) -> tuple[int, int | str]:
    try:
        return 0, int(value)
    except ValueError:
        return 1, value


def load_tracks(root: Path, case_id: str, frame_count: int) -> list[dict[str, Any]]:
    tracks = []
    for path in sorted(root.glob(f"*_{case_id}.npz")):
        label = native_label(path, case_id)
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "frame_ids", "pred_cam", "pred_pose", "pred_shape",
                "pred_rotmat", "pred_trans",
            }
            missing = required.difference(archive.files)
            if missing:
                raise KeyError(f"{path} misses {sorted(missing)}")
            raw = {key: np.asarray(archive[key]) for key in required}
        frames = raw["frame_ids"].astype(np.int64)
        count = len(frames)
        expected = {
            "pred_shape": (count, 10),
            "pred_rotmat": (count, 24, 3, 3),
            "pred_trans": (count, 1, 3),
        }
        if frames.ndim != 1 or np.any(frames < 0) or np.any(frames >= frame_count):
            raise ValueError(f"bad frame IDs in {path}")
        for name, shape in expected.items():
            if raw[name].shape != shape or not np.isfinite(raw[name]).all():
                raise ValueError(f"bad {name} in {path}: {raw[name].shape}")
        # The official entry intentionally repeats the first cached prediction
        # at frame 1.  Keep the last occurrence deterministically; never use GT
        # or a quality score to choose between duplicates.
        last_index = {int(frame): index for index, frame in enumerate(frames)}
        indices = np.asarray([last_index[frame] for frame in sorted(last_index)], dtype=np.int64)
        tracks.append(
            {
                "label": label,
                "path": path,
                "path_bytes": path.stat().st_size,
                "frame_ids": frames[indices],
                "pred_shape": raw["pred_shape"][indices].astype(np.float32),
                "pred_rotmat": raw["pred_rotmat"][indices].astype(np.float32),
                "pred_trans": raw["pred_trans"][indices, 0].astype(np.float32),
                "duplicate_rows_removed": count - len(indices),
            }
        )
    tracks.sort(key=lambda item: track_key(str(item["label"])))
    return tracks


def reconstruct(
    model: Any,
    track: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    vertices, joints = [], []
    with torch.inference_mode():
        for start in range(0, len(track["frame_ids"]), batch_size):
            stop = min(start + batch_size, len(track["frame_ids"]))
            rotation = torch.from_numpy(track["pred_rotmat"][start:stop]).to(device)
            prediction = model(
                body_pose=rotation[:, 1:],
                global_orient=rotation[:, [0]],
                betas=torch.from_numpy(track["pred_shape"][start:stop]).to(device),
                transl=torch.from_numpy(track["pred_trans"][start:stop]).to(device),
                pose2rot=False,
                default_smpl=True,
            )
            vertices.append(prediction.vertices.detach().cpu().numpy().astype(np.float32))
            joints.append(prediction.joints[:, :24].detach().cpu().numpy().astype(np.float32))
    return np.concatenate(vertices), np.concatenate(joints)


def pack(
    cameras: np.ndarray,
    tracks: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    from lib.models.smpl import SMPL

    frame_count = len(cameras)
    if not tracks:
        arrays = {
            "cameras_c2w": cameras,
            "vertices_world": np.full((frame_count, 1, 6890, 3), np.nan, np.float32),
            "joints_world": np.full((frame_count, 1, 24, 3), np.nan, np.float32),
            "persistent_ids": np.full((frame_count, 1), -1, np.int32),
            "native_ids": np.full((frame_count, 1), -1, np.int32),
            "valid": np.zeros((frame_count, 1), np.uint8),
        }
        return arrays, {"native_tracks": 0, "valid_person_frames": 0, "empty_prediction": True}

    # The upstream SMPL wrapper hard-codes ``data/smpl`` relative to the
    # repository working directory.  Resolve that dependency locally without
    # imposing a cwd requirement on the publication adapter.
    original_cwd = Path.cwd()
    try:
        os.chdir(ONLINEHMR_REPO)
        model = SMPL(gender="neutral").to(device).eval()
    finally:
        os.chdir(original_cwd)
    active = [[] for _ in range(frame_count)]
    reconstructed = []
    for mapped_id, track in enumerate(tracks):
        surface, joints = reconstruct(model, track, device, batch_size)
        reconstructed.append((mapped_id, track, surface, joints))
        for frame in track["frame_ids"]:
            active[int(frame)].append(mapped_id)
    people = max(max(map(len, active), default=0), 1)
    vertices = np.full((frame_count, people, 6890, 3), np.nan, np.float32)
    joints = np.full((frame_count, people, 24, 3), np.nan, np.float32)
    persistent = np.full((frame_count, people), -1, np.int32)
    native = np.full((frame_count, people), -1, np.int32)
    valid = np.zeros((frame_count, people), np.uint8)
    slots = [{value: slot for slot, value in enumerate(sorted(values))} for values in active]
    max_roundtrip = 0.0
    for mapped_id, track, camera_vertices, camera_joints in reconstructed:
        for index, frame_value in enumerate(track["frame_ids"]):
            frame = int(frame_value)
            slot = slots[frame][mapped_id]
            rotation = cameras[frame, :3, :3]
            translation = cameras[frame, :3, 3]
            world_vertices = camera_vertices[index] @ rotation.T + translation
            world_joints = camera_joints[index] @ rotation.T + translation
            vertices[frame, slot] = world_vertices
            joints[frame, slot] = world_joints
            persistent[frame, slot] = mapped_id
            native[frame, slot] = mapped_id
            valid[frame, slot] = 1
            roundtrip = (world_joints - translation) @ rotation
            max_roundtrip = max(
                max_roundtrip,
                float(np.max(np.abs(roundtrip - camera_joints[index]))),
            )
    if max_roundtrip > 2e-4:
        raise ValueError(f"camera/world round-trip residual is too large: {max_roundtrip}")
    arrays = {
        "cameras_c2w": cameras,
        "vertices_world": vertices,
        "joints_world": joints,
        "persistent_ids": persistent,
        "native_ids": native,
        "valid": valid,
    }
    return arrays, {
        "native_tracks": len(tracks),
        "raw_to_packed_id": {str(track["label"]): index for index, track in enumerate(tracks)},
        "valid_person_frames": int(valid.sum()),
        "predicted_frames": int(np.count_nonzero(valid.any(axis=1))),
        "max_people_in_one_frame": int(valid.sum(axis=1).max()),
        "camera_world_roundtrip_max_residual_m": max_roundtrip,
        "duplicate_rows_removed": int(sum(track["duplicate_rows_removed"] for track in tracks)),
        "empty_prediction": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--camera-trajectory", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--line", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--method", default=METHOD)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    row = read_row(args.manifest.resolve(), int(args.line))
    frame_count = int(row["clip_length"])
    cameras, camera_audit = load_camera(args.camera_trajectory.resolve(), frame_count)
    tracks = load_tracks(args.native_root.resolve(), str(row["case_id"]), frame_count)
    arrays, summary = pack(cameras, tracks, torch.device(args.device), int(args.batch_size))
    prefix = str(args.method) + "__"
    packed = {prefix + key: value for key, value in arrays.items()}
    atomic_npz(args.output.resolve(), packed)
    metadata = {
        "schema_version": SCHEMA,
        "case_id": row["case_id"],
        "record": row,
        "methods": [str(args.method)],
        "summary": summary,
        "camera_audit": camera_audit,
        "cache": str(args.output.resolve()),
        "cache_bytes": args.output.resolve().stat().st_size,
        "native_root": str(args.native_root.resolve()),
        "camera_trajectory": str(args.camera_trajectory.resolve()),
        "camera_trajectory_bytes": args.camera_trajectory.resolve().stat().st_size,
        "native_track_files": [
            {"path": str(track["path"]), "bytes": track["path_bytes"]}
            for track in tracks
        ],
        "method": str(args.method),
        "topology": "official neutral SMPL-6890 / first 24 official SMPL joints",
        "coordinate_contract": (
            "native camera-coordinate SMPL transformed by OnlineHMR's independent "
            "incremental MASt3R-SLAM c2w trajectory"
        ),
        "runtime_gt_access": False,
        "gt_assisted_track_repair": False,
        "post_cut_geometric_correction": False,
    }
    atomic_json(args.metadata_output.resolve(), metadata)
    print(json.dumps(jsonable({"case_id": row["case_id"], **summary}), indent=2))


if __name__ == "__main__":
    main()
