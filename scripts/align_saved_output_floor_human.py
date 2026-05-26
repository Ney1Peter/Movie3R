#!/usr/bin/env python3
"""Align saved-output humans after floor normal leveling.

This debug utility assumes the input saved-output directory already has parallel
floor normals. It keeps that floor normal fixed, then left-multiplies selected
camera poses by a yaw rotation around the floor normal plus an in-plane
translation so the selected frames' human joints align to a reference frame.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import torch

from overfit_human_anchor_pose_correction import FOOT_JOINTS, STABLE_JOINTS, load_sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True, help="Floor-parallel saved-output directory.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Human-aligned saved-output directory.")
    parser.add_argument("--normal_debug_json", type=Path, required=True, help="Floor-normal overlay JSON from the input directory.")
    parser.add_argument("--reference_viewer_frame", type=int, default=0)
    parser.add_argument("--align_viewer_frames", type=int, nargs="*", default=None, help="Viewer frame ids to align. Defaults to all non-reference frames.")
    parser.add_argument("--stable_weight", type=float, default=1.0)
    parser.add_argument("--foot_weight", type=float, default=2.0)
    parser.add_argument(
        "--normal_translation_source",
        choices=["none", "human_centroid", "floor_center"],
        default="none",
        help="Optional translation along the floor normal; rotations still keep normals parallel.",
    )
    parser.add_argument("--line_length", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def infer_num_frames(input_dir: Path) -> int:
    files = sorted((input_dir / "camera").glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No camera npz files under {input_dir / 'camera'}")
    return len(files)


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    for subdir in ["camera", "camera_raw", "color", "conf", "depth", "smpl"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)


def link_or_symlink(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        rel_src = os.path.relpath(src, dst.parent)
        os.symlink(rel_src, dst)


def copy_payload(input_dir: Path, output_dir: Path, frame_id: int) -> None:
    for subdir, ext in [("camera_raw", ".npz"), ("color", ".png"), ("conf", ".npy"), ("depth", ".npy"), ("smpl", ".npz")]:
        src = input_dir / subdir / f"{frame_id:06d}{ext}"
        if not src.is_file():
            if subdir == "camera_raw":
                continue
            raise FileNotFoundError(src)
        link_or_symlink(src, output_dir / subdir / f"{frame_id:06d}{ext}")


def plane_center(plane: dict) -> np.ndarray:
    if "center" in plane:
        return np.asarray(plane["center"], dtype=np.float64)
    return np.asarray(plane["start"], dtype=np.float64)


def make_plane_basis(normal: np.ndarray) -> np.ndarray:
    n = normalize(normal)
    seed = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(seed, n))) > 0.9:
        seed = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    u = normalize(seed - n * float(np.dot(seed, n)))
    v = normalize(np.cross(n, u))
    return np.stack([u, v], axis=0)


def rotation_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = normalize(axis)
    x, y, z = axis
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    eye = np.eye(3, dtype=np.float64)
    return eye + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def project_to_plane(v: np.ndarray, normal: np.ndarray) -> np.ndarray:
    n = normalize(normal)
    return v - np.sum(v * n, axis=-1, keepdims=True) * n


def build_joint_selection(stable_weight: float, foot_weight: float) -> tuple[np.ndarray, np.ndarray]:
    weights_by_joint: dict[int, float] = {}
    for idx in STABLE_JOINTS:
        weights_by_joint[int(idx)] = weights_by_joint.get(int(idx), 0.0) + float(stable_weight)
    for idx in FOOT_JOINTS:
        weights_by_joint[int(idx)] = weights_by_joint.get(int(idx), 0.0) + float(foot_weight)
    joint_ids = np.asarray(sorted(weights_by_joint), dtype=np.int64)
    weights = np.asarray([weights_by_joint[int(idx)] for idx in joint_ids], dtype=np.float64)
    weights /= max(float(weights.sum()), 1e-12)
    return joint_ids, weights


def solve_yaw_and_plane_translation(ref_points: np.ndarray, cur_points: np.ndarray, normal: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
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
    # **========== 原始代码 ==========**
    # t = project_to_plane(ref_centroid - R @ cur_centroid, normal)
    # return R, t, yaw

    # **========== 新代码 ==========**
    t_plane = project_to_plane(ref_centroid - R @ cur_centroid, normal)
    return R, t_plane, yaw
    # **========== 结束 ==========**


def compute_normal_translation(
    source: str,
    ref_points: np.ndarray,
    cur_points: np.ndarray,
    weights: np.ndarray,
    normal: np.ndarray,
    R: np.ndarray,
    ref_floor_center: np.ndarray,
    cur_floor_center: np.ndarray,
) -> np.ndarray:
    n = normalize(normal)
    if source == "human_centroid":
        ref_centroid = np.sum(ref_points * weights[:, None], axis=0)
        cur_centroid = np.sum(cur_points * weights[:, None], axis=0)
        amount = float(np.dot(ref_centroid - R @ cur_centroid, n))
        return amount * n
    if source == "floor_center":
        amount = float(np.dot(ref_floor_center - R @ cur_floor_center, n))
        return amount * n
    return np.zeros(3, dtype=np.float64)


def transform_pose(pose: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = pose.astype(np.float64).copy()
    out[:3, :3] = R @ out[:3, :3]
    out[:3, 3] = R @ out[:3, 3] + t
    return out.astype(np.float32)


def transform_points(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return points @ R.T + t[None]


def joint_metrics(ref: np.ndarray, cur: np.ndarray, normal: np.ndarray, joint_ids: np.ndarray, weights: np.ndarray) -> dict:
    diff = cur[joint_ids] - ref[joint_ids]
    plane_diff = project_to_plane(diff, normal)
    normal_diff = np.abs(diff @ normalize(normal))
    stable = np.asarray(STABLE_JOINTS, dtype=np.int64)
    foot = np.asarray(FOOT_JOINTS, dtype=np.int64)
    return {
        "selected_full_mean": float(np.sum(np.linalg.norm(diff, axis=1) * weights)),
        "selected_plane_mean": float(np.sum(np.linalg.norm(plane_diff, axis=1) * weights)),
        "selected_normal_mean": float(np.sum(normal_diff * weights)),
        "stable_full_mean": float(np.linalg.norm(cur[stable] - ref[stable], axis=1).mean()),
        "foot_full_mean": float(np.linalg.norm(cur[foot] - ref[foot], axis=1).mean()),
    }


def transform_overlay(debug: dict, transforms: dict[int, tuple[np.ndarray, np.ndarray]], line_length: float) -> dict:
    overlay_planes = []
    overlay_segments = []
    overlay_labels = []
    for plane in debug.get("planes", []):
        frame = int(plane["viewer_frame"])
        normal = normalize(np.asarray(plane["normal"], dtype=np.float64))
        center = plane_center(plane)
        if frame in transforms:
            R, t = transforms[frame]
            normal = normalize(R @ normal)
            center = R @ center + t
        start = center.astype(np.float32)
        end = (center + float(line_length) * normal).astype(np.float32)
        label_position = (center + float(line_length) * 1.15 * normal).astype(np.float32)
        color = plane.get("color", [255, 255, 0])
        overlay_segments.append({"label": plane.get("label", f"viewer{frame}"), "start": start.tolist(), "end": end.tolist(), "color": color})
        overlay_labels.append({"text": f"floor n + human align raw{plane.get('raw_frame', frame)} v{frame}", "position": label_position.tolist(), "height": 0.12})
        overlay_plane = dict(plane)
        overlay_plane["normal"] = normal.astype(np.float32).tolist()
        overlay_plane["center"] = start.tolist()
        overlay_plane["label_position"] = label_position.tolist()
        overlay_planes.append(overlay_plane)
    return {
        "description": "Floor normals after yaw + in-plane human alignment.",
        "segments": overlay_segments,
        "labels": overlay_labels,
        "planes": overlay_planes,
        "line_width": debug.get("line_width", 6.0),
    }


def main() -> None:
    args = parse_args()
    num_frames = infer_num_frames(args.input_dir)
    debug = json.loads(args.normal_debug_json.read_text())
    planes = {int(p["viewer_frame"]): p for p in debug.get("planes", [])}
    ref_frame = int(args.reference_viewer_frame)
    if ref_frame not in planes:
        raise KeyError(f"Reference viewer frame {ref_frame} missing from debug JSON")
    normal = normalize(np.asarray(planes[ref_frame]["normal"], dtype=np.float64))

    align_frames = args.align_viewer_frames
    if align_frames is None:
        align_frames = [frame for frame in range(num_frames) if frame != ref_frame]
    align_frames = [int(x) for x in align_frames]

    data = load_sequence(args.input_dir, num_frames, torch.device(args.device))
    joint_ids, weights = build_joint_selection(args.stable_weight, args.foot_weight)
    ref_joints = data.joints_world[ref_frame].astype(np.float64)

    transforms: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    records = []
    corrected_poses = data.poses.copy()
    for frame in align_frames:
        if frame < 0 or frame >= num_frames:
            raise ValueError(f"Viewer frame {frame} outside saved-output frame count {num_frames}")
        cur_joints = data.joints_world[frame].astype(np.float64)
        before = joint_metrics(ref_joints, cur_joints, normal, joint_ids, weights)
        # **========== 原始代码 ==========**
        # R, t, yaw = solve_yaw_and_plane_translation(ref_joints[joint_ids], cur_joints[joint_ids], normal, weights)
        # aligned_joints = transform_points(cur_joints, R, t)

        # **========== 新代码 ==========**
        R, t_plane, yaw = solve_yaw_and_plane_translation(ref_joints[joint_ids], cur_joints[joint_ids], normal, weights)
        t_normal = compute_normal_translation(
            args.normal_translation_source,
            ref_joints[joint_ids],
            cur_joints[joint_ids],
            weights,
            normal,
            R,
            plane_center(planes[ref_frame]),
            plane_center(planes.get(frame, {"center": [0.0, 0.0, 0.0]})),
        )
        t = t_plane + t_normal
        aligned_joints = transform_points(cur_joints, R, t)
        # **========== 结束 ==========**
        after = joint_metrics(ref_joints, aligned_joints, normal, joint_ids, weights)
        corrected_poses[frame] = transform_pose(data.poses[frame], R, t)
        transforms[frame] = (R, t)
        records.append(
            {
                "viewer_frame": int(frame),
                "raw_frame": int(planes.get(frame, {}).get("raw_frame", frame)),
                "yaw_deg": math.degrees(float(yaw)),
                "translation": t.astype(np.float32).tolist(),
                "translation_plane": t_plane.astype(np.float32).tolist(),
                "translation_normal": t_normal.astype(np.float32).tolist(),
                "translation_norm": float(np.linalg.norm(t)),
                "translation_normal_component": float(np.dot(t, normal)),
                "floor_normal_after_dot": float(np.dot(R @ normal, normal)),
                "before": before,
                "after": after,
            }
        )

    prepare_output_dir(args.output_dir, args.overwrite)
    for frame in range(num_frames):
        cam = np.load(args.input_dir / "camera" / f"{frame:06d}.npz")
        np.savez(args.output_dir / "camera" / f"{frame:06d}.npz", pose=corrected_poses[frame].astype(np.float32), intrinsics=cam["intrinsics"].astype(np.float32))
        copy_payload(args.input_dir, args.output_dir, frame)

    overlay = transform_overlay(debug, transforms, args.line_length)
    overlay.update(
        {
            "input_dir": str(args.input_dir),
            "output_dir": str(args.output_dir),
            "source_normal_debug_json": str(args.normal_debug_json),
            "reference_viewer_frame": ref_frame,
            "aligned_viewer_frames": align_frames,
        }
    )
    overlay_path = args.output_dir / "floor_human_alignment_debug.json"
    overlay_path.write_text(json.dumps(overlay, indent=2, sort_keys=True), encoding="utf-8")

    metrics = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "normal_debug_json": str(args.normal_debug_json),
        "reference_viewer_frame": ref_frame,
        "reference_normal": normal.astype(np.float32).tolist(),
        "normal_translation_source": args.normal_translation_source,
        "aligned_viewer_frames": align_frames,
        "joint_ids": joint_ids.astype(int).tolist(),
        "joint_weights": weights.astype(float).tolist(),
        "aligned": records,
        "overlay_json": str(overlay_path),
    }
    metrics_path = args.output_dir / "floor_human_alignment_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
