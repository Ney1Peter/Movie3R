#!/usr/bin/env python3
"""Probe streaming segment alignment with original Human3R outputs.

The prototype is intentionally inference-only:

1. split a 4-frame AABB input into two continuous shot segments;
2. run strict original Human3R separately on each segment;
3. concatenate segment outputs without alignment as a control;
4. align segment B into segment A's world gauge with a single yaw + translation
   transform estimated from predicted SMPL joints.

This tests whether the hard part is frame-wise correction or segment-level gauge
alignment.  It never uses V9 raw/corrected outputs as "original Human3R".
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
ARCHIVE_V7 = REPO_ROOT / "scripts" / "archive_v7"
for path in (str(REPO_ROOT), str(SRC_ROOT), str(ARCHIVE_V7)):
    if path not in sys.path:
        sys.path.insert(0, path)

from overfit_human_anchor_pose_correction import FOOT_JOINTS, STABLE_JOINTS, load_sequence, torso_frame


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--case_name", default=None)
    parser.add_argument("--output_root", type=Path, default=REPO_ROOT / "output" / "v9_segment_state_probe")
    parser.add_argument("--model_path", type=Path, default=REPO_ROOT / "src" / "human3r_896L.pth")
    parser.add_argument("--boundary", type=int, default=2, help="First frame index of segment B.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_inference", action="store_true")
    parser.add_argument("--stable_weight", type=float, default=1.0)
    parser.add_argument("--foot_weight", type=float, default=1.5)
    parser.add_argument(
        "--alignment_mode",
        choices=["segment_mean", "stream_first"],
        default="segment_mean",
        help=(
            "segment_mean uses both frames in segment B to estimate the transform. "
            "stream_first estimates the transform from historical segment A and "
            "only the first frame of segment B, then applies it to all B frames."
        ),
    )
    return parser.parse_args()


def list_images(input_dir: Path) -> list[Path]:
    files = [p for p in sorted(input_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if len(files) < 4:
        raise ValueError(f"Expected at least 4 images under {input_dir}, found {len(files)}")
    return files[:4]


def reset_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_segment_inputs(images: list[Path], segment_dir: Path, frame_ids: list[int], overwrite: bool) -> None:
    reset_dir(segment_dir, overwrite)
    existing = sorted(p for p in segment_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if len(existing) == len(frame_ids) and not overwrite:
        return
    for p in existing:
        p.unlink()
    for out_i, src_idx in enumerate(frame_ids, start=1):
        src = images[src_idx]
        dst = segment_dir / f"{out_i:06d}{src.suffix.lower()}"
        shutil.copy2(src, dst)


def output_complete(output_dir: Path) -> bool:
    return (output_dir / "camera" / "000000.npz").is_file()


def run_original_human3r(seq_path: Path, output_dir: Path, args: argparse.Namespace) -> None:
    if output_complete(output_dir) and not args.overwrite:
        return
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_human3r_save_output.py"),
        "--model_path",
        str(args.model_path),
        "--seq_path",
        str(seq_path),
        "--output_dir",
        str(output_dir),
        "--device",
        str(args.device),
        "--size",
        str(args.size),
        "--strict_original_human3r",
        "--overwrite",
    ]
    env = os.environ.copy()
    pythonpath = os.pathsep.join([str(REPO_ROOT), str(SRC_ROOT), env.get("PYTHONPATH", "")])
    env["PYTHONPATH"] = pythonpath
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)


def copy_np_payload(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def save_camera(src_camera: Path, dst_camera: Path, pose: np.ndarray | None = None) -> None:
    cam = np.load(src_camera)
    out_pose = cam["pose"].astype(np.float32) if pose is None else pose.astype(np.float32)
    np.savez(dst_camera, pose=out_pose, intrinsics=cam["intrinsics"].astype(np.float32))


def copy_smpl(src_smpl: Path, dst_smpl: Path, R: np.ndarray | None = None, t: np.ndarray | None = None) -> None:
    data = np.load(src_smpl, allow_pickle=True)
    payload = {key: data[key] for key in data.files}
    if R is not None and t is not None and "verts_world" in payload:
        verts = payload["verts_world"].astype(np.float64)
        payload["verts_world"] = (verts @ R.T + t.reshape(1, 1, 3)).astype(np.float32)
    np.savez(dst_smpl, **payload)


def prepare_saved_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    for sub in ["camera", "color", "conf", "depth", "smpl"]:
        (path / sub).mkdir(parents=True, exist_ok=True)


def transform_pose(pose: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = pose.astype(np.float64).copy()
    out[:3, :3] = R @ out[:3, :3]
    out[:3, 3] = R @ out[:3, 3] + t
    return out.astype(np.float32)


def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / max(float(np.linalg.norm(v)), eps)


def rotation_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = normalize(axis)
    x, y, z = axis
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    eye = np.eye(3, dtype=np.float64)
    return eye + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def make_plane_basis(normal: np.ndarray) -> np.ndarray:
    n = normalize(normal)
    seed = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(seed, n))) > 0.9:
        seed = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    u = normalize(seed - n * float(np.dot(seed, n)))
    v = normalize(np.cross(n, u))
    return np.stack([u, v], axis=0)


def weighted_joint_ids(stable_weight: float, foot_weight: float) -> tuple[np.ndarray, np.ndarray]:
    weights_by_joint: dict[int, float] = {}
    for idx in STABLE_JOINTS:
        weights_by_joint[int(idx)] = weights_by_joint.get(int(idx), 0.0) + float(stable_weight)
    for idx in FOOT_JOINTS:
        weights_by_joint[int(idx)] = weights_by_joint.get(int(idx), 0.0) + float(foot_weight)
    joint_ids = np.asarray(sorted(weights_by_joint), dtype=np.int64)
    weights = np.asarray([weights_by_joint[int(idx)] for idx in joint_ids], dtype=np.float64)
    weights /= max(float(weights.sum()), 1e-12)
    return joint_ids, weights


def solve_yaw_translation(
    ref_points: np.ndarray,
    cur_points: np.ndarray,
    normal: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    basis = make_plane_basis(normal)
    ref_2d = ref_points @ basis.T
    cur_2d = cur_points @ basis.T
    ref_centroid_2d = np.sum(ref_2d * weights[:, None], axis=0)
    cur_centroid_2d = np.sum(cur_2d * weights[:, None], axis=0)
    ref_centered = ref_2d - ref_centroid_2d
    cur_centered = cur_2d - cur_centroid_2d

    a = float(np.sum(weights * (cur_centered[:, 0] * ref_centered[:, 0] + cur_centered[:, 1] * ref_centered[:, 1])))
    b = float(np.sum(weights * (cur_centered[:, 0] * ref_centered[:, 1] - cur_centered[:, 1] * ref_centered[:, 0])))
    yaw = math.atan2(b, a)
    R = rotation_about_axis(normal, yaw)

    ref_centroid = np.sum(ref_points * weights[:, None], axis=0)
    cur_centroid = np.sum(cur_points * weights[:, None], axis=0)
    t = ref_centroid - R @ cur_centroid
    return R, t, yaw


def mean_body_up(joints_world: np.ndarray) -> np.ndarray:
    joints_t = torch.from_numpy(joints_world.astype(np.float32))
    frame = torso_frame(joints_t).numpy()
    up = normalize(frame[:, 1].mean(axis=0))
    if not np.isfinite(up).all() or float(np.linalg.norm(up)) < 1e-6:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return up


def root_orientation_translation(output_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    root_dirs, root_trans = [], []
    camera_files = sorted((output_dir / "camera").glob("*.npz"))
    for i, camera_path in enumerate(camera_files):
        cam = np.load(camera_path)["pose"].astype(np.float64)
        smpl = np.load(output_dir / "smpl" / f"{i:06d}.npz", allow_pickle=True)
        if smpl["rotvec"].shape[0] < 1:
            raise ValueError(f"No detected human in {output_dir / 'smpl' / f'{i:06d}.npz'}")
        root_R_cam, _ = cv2.Rodrigues(smpl["rotvec"][0, 0].astype(np.float64))
        root_R_world = cam[:3, :3] @ root_R_cam
        root_t_world = cam[:3, :3] @ smpl["transl"][0].astype(np.float64) + cam[:3, 3]
        root_dirs.append(root_R_world[:, 2])
        root_trans.append(root_t_world)
    return np.stack(root_dirs), np.stack(root_trans)


def signed_yaw_between(src_dir: np.ndarray, dst_dir: np.ndarray, up: np.ndarray) -> float:
    up = normalize(up)
    src = normalize(src_dir - up * float(np.dot(src_dir, up)))
    dst = normalize(dst_dir - up * float(np.dot(dst_dir, up)))
    return math.atan2(float(np.dot(up, np.cross(src, dst))), float(np.dot(src, dst)))


def solve_root_yaw_translation(seg_a_dir: Path, seg_b_dir: Path, mode: str) -> tuple[np.ndarray, np.ndarray, float, dict]:
    a_dirs, a_trans = root_orientation_translation(seg_a_dir)
    b_dirs, b_trans = root_orientation_translation(seg_b_dir)
    # Human3R demo/world outputs consistently use y as the vertical axis in
    # these saved-output probes.  Keeping the rotation to yaw prevents arbitrary
    # roll/pitch flips while still fixing the front/back ambiguity.
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    a_dir = normalize(a_dirs.mean(axis=0))
    if mode == "stream_first":
        b_dir = normalize(b_dirs[0])
        b_anchor = b_trans[0]
    else:
        b_dir = normalize(b_dirs.mean(axis=0))
        b_anchor = b_trans.mean(axis=0)
    yaw = signed_yaw_between(b_dir, a_dir, up)
    R = rotation_about_axis(up, yaw)
    a_anchor = a_trans.mean(axis=0)
    t = a_anchor - R @ b_anchor
    debug = {
        "alignment_mode": mode,
        "root_axis": "SMPL root local +Z transformed to Human3R world",
        "up_axis": up.astype(np.float32).tolist(),
        "a_root_dir_mean": a_dir.astype(np.float32).tolist(),
        "b_root_dir_mean": b_dir.astype(np.float32).tolist(),
        "a_root_translation_mean": a_anchor.astype(np.float32).tolist(),
        "b_root_translation_anchor_raw": b_anchor.astype(np.float32).tolist(),
        "b_root_translation_anchor_aligned": (R @ b_anchor + t).astype(np.float32).tolist(),
    }
    return R, t, yaw, debug


def load_data(output_dir: Path, device: str):
    n = len(sorted((output_dir / "camera").glob("*.npz")))
    return load_sequence(output_dir, n, torch.device(device))


def pose_delta_metrics(poses: np.ndarray, i: int, j: int) -> dict[str, float]:
    t_err = float(np.linalg.norm(poses[i, :3, 3] - poses[j, :3, 3]))
    r_rel = poses[i, :3, :3].T @ poses[j, :3, :3]
    cos = float(np.clip((np.trace(r_rel) - 1.0) * 0.5, -1.0, 1.0))
    r_deg = float(math.degrees(math.acos(cos)))
    return {"t_m": t_err, "r_deg": r_deg}


def joint_dist(a: np.ndarray, b: np.ndarray, joint_ids: np.ndarray) -> float:
    return float(np.linalg.norm(a[joint_ids] - b[joint_ids], axis=-1).mean())


def build_outputs(
    case_dir: Path,
    seg_a_dir: Path,
    seg_b_dir: Path,
    R: np.ndarray,
    t: np.ndarray,
    overwrite: bool,
    aligned_subdir: str,
) -> tuple[Path, Path]:
    raw_dir = case_dir / "human3r_segments_concat_raw"
    aligned_dir = case_dir / aligned_subdir
    prepare_saved_output_dir(raw_dir, overwrite)
    prepare_saved_output_dir(aligned_dir, overwrite)

    for out_i, (src_dir, src_i, apply_align) in enumerate(
        [
            (seg_a_dir, 0, False),
            (seg_a_dir, 1, False),
            (seg_b_dir, 0, True),
            (seg_b_dir, 1, True),
        ]
    ):
        for dst_root in [raw_dir, aligned_dir]:
            for sub, ext in [("color", ".png"), ("conf", ".npy"), ("depth", ".npy")]:
                copy_np_payload(src_dir / sub / f"{src_i:06d}{ext}", dst_root / sub / f"{out_i:06d}{ext}")

        src_camera = src_dir / "camera" / f"{src_i:06d}.npz"
        save_camera(src_camera, raw_dir / "camera" / f"{out_i:06d}.npz")
        src_smpl = src_dir / "smpl" / f"{src_i:06d}.npz"
        copy_smpl(src_smpl, raw_dir / "smpl" / f"{out_i:06d}.npz")

        cam = np.load(src_camera)
        aligned_pose = transform_pose(cam["pose"].astype(np.float32), R, t) if apply_align else cam["pose"].astype(np.float32)
        save_camera(src_camera, aligned_dir / "camera" / f"{out_i:06d}.npz", aligned_pose)
        copy_smpl(src_smpl, aligned_dir / "smpl" / f"{out_i:06d}.npz", R if apply_align else None, t if apply_align else None)

    return raw_dir, aligned_dir


def compute_summary(
    case_dir: Path,
    seg_a_dir: Path,
    seg_b_dir: Path,
    R: np.ndarray,
    t: np.ndarray,
    yaw: float,
    args: argparse.Namespace,
    alignment_debug: dict,
) -> dict:
    data_a = load_data(seg_a_dir, args.device)
    data_b = load_data(seg_b_dir, args.device)
    joint_ids, weights = weighted_joint_ids(args.stable_weight, args.foot_weight)
    a_ref = np.sum(data_a.joints_world[:, joint_ids].mean(axis=0) * weights[:, None], axis=0)
    b_ref = np.sum(data_b.joints_world[:, joint_ids].mean(axis=0) * weights[:, None], axis=0)
    b_world_aligned = data_b.joints_world @ R.T + t.reshape(1, 1, 3)
    poses_b_aligned = np.stack([transform_pose(p, R, t) for p in data_b.poses], axis=0)
    all_raw_poses = np.concatenate([data_a.poses, data_b.poses], axis=0)
    all_aligned_poses = np.concatenate([data_a.poses, poses_b_aligned], axis=0)
    return {
        "case_dir": str(case_dir),
        "segment_A": str(seg_a_dir),
        "segment_B": str(seg_b_dir),
        "alignment": {
            "type": "segment root-yaw plus 3D root-translation from original Human3R SMPL",
            "alignment_mode": args.alignment_mode,
            "yaw_deg": float(math.degrees(yaw)),
            "translation": t.astype(np.float32).tolist(),
            "translation_norm_m": float(np.linalg.norm(t)),
            "weighted_A_centroid": a_ref.astype(np.float32).tolist(),
            "weighted_B_centroid_raw": b_ref.astype(np.float32).tolist(),
            "root_debug": alignment_debug,
        },
        "human_joint_mean_m": {
            "AA_0_1": joint_dist(data_a.joints_world[0], data_a.joints_world[1], joint_ids),
            "BB_0_1_raw": joint_dist(data_b.joints_world[0], data_b.joints_world[1], joint_ids),
            "BB_0_1_aligned": joint_dist(b_world_aligned[0], b_world_aligned[1], joint_ids),
            "A_mean_to_B0_raw": joint_dist(data_a.joints_world[:, joint_ids].mean(axis=0), data_b.joints_world[0, joint_ids], np.arange(len(joint_ids))),
            "A_mean_to_B0_aligned": joint_dist(data_a.joints_world[:, joint_ids].mean(axis=0), b_world_aligned[0, joint_ids], np.arange(len(joint_ids))),
            "A_mean_to_B1_aligned": joint_dist(data_a.joints_world[:, joint_ids].mean(axis=0), b_world_aligned[1, joint_ids], np.arange(len(joint_ids))),
        },
        "camera_delta": {
            "AA_0_1": pose_delta_metrics(all_aligned_poses, 0, 1),
            "BB_2_3_raw": pose_delta_metrics(all_raw_poses, 2, 3),
            "BB_2_3_aligned": pose_delta_metrics(all_aligned_poses, 2, 3),
            "boundary_1_2_raw": pose_delta_metrics(all_raw_poses, 1, 2),
            "boundary_1_2_aligned": pose_delta_metrics(all_aligned_poses, 1, 2),
        },
        "joint_ids": joint_ids.astype(int).tolist(),
        "joint_weights": weights.astype(float).tolist(),
    }


def main() -> None:
    args = parse_args()
    case_name = args.case_name or args.input_dir.name
    case_dir = args.output_root / case_name
    if args.overwrite and case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(args.input_dir)
    input_a = case_dir / "inputs_A"
    input_b = case_dir / "inputs_B"
    copy_segment_inputs(images, input_a, list(range(args.boundary)), args.overwrite)
    copy_segment_inputs(images, input_b, list(range(args.boundary, 4)), args.overwrite)

    seg_a_dir = case_dir / "human3r_segment_A"
    seg_b_dir = case_dir / "human3r_segment_B"
    if not args.skip_inference:
        run_original_human3r(input_a, seg_a_dir, args)
        run_original_human3r(input_b, seg_b_dir, args)

    R, t, yaw, alignment_debug = solve_root_yaw_translation(seg_a_dir, seg_b_dir, args.alignment_mode)

    aligned_subdir = (
        "human3r_segments_align_yaw_trans_stream_first"
        if args.alignment_mode == "stream_first"
        else "human3r_segments_align_yaw_trans_stream"
    )
    raw_dir, aligned_dir = build_outputs(case_dir, seg_a_dir, seg_b_dir, R, t, args.overwrite, aligned_subdir)
    summary = compute_summary(case_dir, seg_a_dir, seg_b_dir, R, t, yaw, args, alignment_debug)
    summary.update(
        {
            "input_dir": str(args.input_dir),
            "images": [str(p) for p in images],
            "raw_output_dir": str(raw_dir),
            "aligned_output_dir": str(aligned_dir),
        }
    )
    summary_path = case_dir / (
        "segment_alignment_yaw_trans_stream_first_summary.json"
        if args.alignment_mode == "stream_first"
        else "segment_alignment_yaw_trans_summary.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
