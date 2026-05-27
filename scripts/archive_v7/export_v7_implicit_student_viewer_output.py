#!/usr/bin/env python3
"""Export V7 implicit-student predictions to a Human3R viewer directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# **========== 原始代码 ==========**
# import torch
#
# from dust3r.utils.camera import pose_encoding_to_camera
# **========== 新代码 ==========**
# Keep this script independent from dust3r.utils.camera because that module can
# circularly import dust3r.heads when used as a standalone script.
# **========== 结束 ==========**
from overfit_single_boundary_frame_human_anchor import write_outputs_with_links


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True, help="Raw Human3R saved-output directory.")
    parser.add_argument("--predictions_npz", type=Path, required=True, help="v7_implicit_student_predictions.npz.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Viewer-ready corrected output directory.")
    parser.add_argument("--raw_output_dir", type=Path, default=None, help="Optional viewer-ready raw camera overlay directory.")
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--subset_start", type=int, default=None)
    parser.add_argument("--subset_count", type=int, default=31)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def infer_num_frames(input_dir: Path, explicit_num_frames: int | None) -> int:
    if explicit_num_frames is not None:
        return int(explicit_num_frames)
    files = sorted((input_dir / "camera").glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No camera files found under {input_dir / 'camera'}")
    return len(files)


def load_raw_cameras(input_dir: Path, num_frames: int) -> tuple[np.ndarray, np.ndarray]:
    poses = []
    intrinsics = []
    for frame_id in range(num_frames):
        cam = np.load(input_dir / "camera" / f"{frame_id:06d}.npz")
        poses.append(cam["pose"].astype(np.float32))
        intrinsics.append(cam["intrinsics"].astype(np.float32))
    return np.stack(poses), np.stack(intrinsics)


# **========== 原始代码 ==========**
# def pose7_to_matrix(pose7: np.ndarray) -> np.ndarray:
#     pose = torch.from_numpy(pose7.astype(np.float32))
#     with torch.no_grad():
#         return pose_encoding_to_camera(pose).cpu().numpy().astype(np.float32)
# **========== 新代码 ==========**
def quaternion_to_matrix_np(quaternions: np.ndarray) -> np.ndarray:
    quaternions = quaternions.astype(np.float32)
    quaternions = quaternions / np.maximum(np.linalg.norm(quaternions, axis=-1, keepdims=True), 1e-8)
    r, i, j, k = np.moveaxis(quaternions, -1, 0)
    two_s = 2.0 / np.maximum(np.sum(quaternions * quaternions, axis=-1), 1e-8)
    matrix = np.stack(
        [
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ],
        axis=-1,
    )
    return matrix.reshape(quaternions.shape[:-1] + (3, 3)).astype(np.float32)


def pose7_to_matrix(pose7: np.ndarray) -> np.ndarray:
    pose7 = pose7.astype(np.float32)
    mats = np.repeat(np.eye(4, dtype=np.float32)[None], pose7.shape[0], axis=0)
    mats[:, :3, :3] = quaternion_to_matrix_np(pose7[:, 3:7])
    mats[:, :3, 3] = pose7[:, :3]
    return mats
# **========== 结束 ==========**


def select_frame_ids(num_frames: int, subset_start: int | None, subset_count: int) -> list[int]:
    if subset_start is None:
        return list(range(num_frames))
    start = max(0, int(subset_start))
    end = min(num_frames, start + int(subset_count))
    if end <= start:
        raise ValueError("Selected empty subset")
    return list(range(start, end))


def main() -> None:
    args = parse_args()
    num_frames = infer_num_frames(args.input_dir, args.num_frames)
    raw_poses, intrinsics = load_raw_cameras(args.input_dir, num_frames)

    pred = np.load(args.predictions_npz)
    frame_ids = pred["frame_ids"].astype(np.int64)
    corrected_pose7 = pred["corrected_camera_pose"].astype(np.float32)
    if corrected_pose7.shape[0] != frame_ids.shape[0]:
        raise ValueError("frame_ids and corrected_camera_pose length mismatch")

    corrected_poses = raw_poses.copy()
    corrected_mats = pose7_to_matrix(corrected_pose7)
    for pred_i, frame_id in enumerate(frame_ids.tolist()):
        if frame_id < 0 or frame_id >= num_frames:
            raise ValueError(f"Predicted frame id {frame_id} outside [0, {num_frames})")
        corrected_poses[frame_id] = corrected_mats[pred_i]

    viewer_frame_ids = select_frame_ids(num_frames, args.subset_start, args.subset_count)
    write_outputs_with_links(args.input_dir, args.output_dir, corrected_poses, intrinsics, viewer_frame_ids, args.overwrite)
    if args.raw_output_dir is not None:
        write_outputs_with_links(args.input_dir, args.raw_output_dir, raw_poses, intrinsics, viewer_frame_ids, args.overwrite)

    target_frames = []
    if "target_mask" in pred.files:
        target_frames = frame_ids[pred["target_mask"].astype(bool)].astype(int).tolist()
    metadata = {
        "input_dir": str(args.input_dir),
        "predictions_npz": str(args.predictions_npz),
        "output_dir": str(args.output_dir),
        "raw_output_dir": str(args.raw_output_dir) if args.raw_output_dir else None,
        "num_frames": int(num_frames),
        "viewer_frame_ids": [int(x) for x in viewer_frame_ids],
        "target_frames": target_frames,
        "corrected_pointcloud_and_humans": True,
        "raw_overlay_cameras_only": args.raw_output_dir is not None,
    }
    with open(args.output_dir / "v7_implicit_student_viewer_export.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
