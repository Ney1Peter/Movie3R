#!/usr/bin/env python3
"""Rigidly align selected saved-output floor normals to a reference frame.

This is a debug-only transform for saved ``demo.py --save`` style directories.
It left-multiplies the selected camera poses by a rotation that makes the stored
floor normal parallel to the reference frame's floor normal. Depth, color,
confidence, and SMPL files are hard-linked unchanged, so the Human3R viewer will
render the original per-frame geometry in the newly leveled world gauge.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True, help="Saved-output directory to transform.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Transformed saved-output directory.")
    parser.add_argument("--normal_debug_json", type=Path, required=True, help="Existing floor-normal debug JSON.")
    parser.add_argument("--reference_viewer_frame", type=int, default=0)
    parser.add_argument("--align_viewer_frames", type=int, nargs="*", default=None, help="Viewer frame ids to align. Defaults to all non-reference planes in the debug JSON.")
    parser.add_argument("--line_length", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def rotation_from_vectors(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src = normalize(src)
    dst = normalize(dst)
    dot = float(np.clip(np.dot(src, dst), -1.0, 1.0))
    if dot > 1.0 - 1e-10:
        return np.eye(3, dtype=np.float64)
    if dot < -1.0 + 1e-10:
        axis = np.cross(src, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        if np.linalg.norm(axis) < 1e-8:
            axis = np.cross(src, np.array([0.0, 1.0, 0.0], dtype=np.float64))
        axis = normalize(axis)
        kx, ky, kz = axis
        K = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]], dtype=np.float64)
        return np.eye(3, dtype=np.float64) + 2.0 * (K @ K)

    axis = np.cross(src, dst)
    s = float(np.linalg.norm(axis))
    kx, ky, kz = axis
    K = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + K + K @ K * ((1.0 - dot) / max(s * s, 1e-12))


def rotation_angle_deg(R: np.ndarray) -> float:
    cos = np.clip((float(np.trace(R)) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(cos))


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


def transform_pose(pose: np.ndarray, R: np.ndarray, center: np.ndarray) -> np.ndarray:
    t = center - R @ center
    out = pose.astype(np.float64).copy()
    out[:3, :3] = R @ out[:3, :3]
    out[:3, 3] = R @ out[:3, 3] + t
    return out.astype(np.float32)


def main() -> None:
    args = parse_args()
    num_frames = infer_num_frames(args.input_dir)
    debug = json.loads(args.normal_debug_json.read_text())
    planes = {int(p["viewer_frame"]): p for p in debug.get("planes", [])}
    if int(args.reference_viewer_frame) not in planes:
        raise KeyError(f"Reference viewer frame {args.reference_viewer_frame} missing from debug JSON")

    ref_frame = int(args.reference_viewer_frame)
    ref_normal = normalize(np.asarray(planes[ref_frame]["normal"], dtype=np.float64))
    align_frames = args.align_viewer_frames
    if align_frames is None:
        align_frames = sorted(frame for frame in planes if frame != ref_frame)
    align_frames = [int(x) for x in align_frames]

    transforms: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    records = []
    for frame in align_frames:
        if frame not in planes:
            raise KeyError(f"Viewer frame {frame} missing from debug JSON")
        if frame < 0 or frame >= num_frames:
            raise ValueError(f"Viewer frame {frame} outside saved-output frame count {num_frames}")
        cur_normal = normalize(np.asarray(planes[frame]["normal"], dtype=np.float64))
        signed_dot = float(np.dot(cur_normal, ref_normal))
        src_normal = cur_normal if signed_dot >= 0.0 else -cur_normal
        R = rotation_from_vectors(src_normal, ref_normal)
        center = plane_center(planes[frame])
        after_normal = normalize(R @ cur_normal)
        transforms[frame] = (R, center)
        records.append(
            {
                "viewer_frame": int(frame),
                "raw_frame": int(planes[frame].get("raw_frame", frame)),
                "before_dot": signed_dot,
                "before_dot_abs": abs(signed_dot),
                "after_dot": float(np.dot(after_normal, ref_normal)),
                "after_dot_abs": abs(float(np.dot(after_normal, ref_normal))),
                "rotation_deg": rotation_angle_deg(R),
                "rotate_about_center": center.astype(float).tolist(),
            }
        )

    prepare_output_dir(args.output_dir, args.overwrite)
    for frame in range(num_frames):
        cam = np.load(args.input_dir / "camera" / f"{frame:06d}.npz")
        pose = cam["pose"].astype(np.float32)
        if frame in transforms:
            R, center = transforms[frame]
            pose = transform_pose(pose, R, center)
        np.savez(args.output_dir / "camera" / f"{frame:06d}.npz", pose=pose, intrinsics=cam["intrinsics"].astype(np.float32))
        copy_payload(args.input_dir, args.output_dir, frame)

    overlay_segments = []
    overlay_planes = []
    overlay_labels = []
    for frame in sorted(planes):
        plane = planes[frame]
        normal = normalize(np.asarray(plane["normal"], dtype=np.float64))
        center = plane_center(plane)
        if frame in transforms:
            R, c = transforms[frame]
            t = c - R @ c
            normal = normalize(R @ normal)
            center = R @ center + t
        color = plane.get("color", [255, 255, 0])
        start = center.astype(np.float32)
        end = (center + float(args.line_length) * normal).astype(np.float32)
        label_position = (center + float(args.line_length) * 1.15 * normal).astype(np.float32)
        overlay_segments.append({"label": plane.get("label", f"viewer{frame}"), "start": start.tolist(), "end": end.tolist(), "color": color})
        overlay_labels.append({"text": f"aligned floor n raw{plane.get('raw_frame', frame)} v{frame}", "position": label_position.tolist(), "height": 0.12})
        overlay_plane = dict(plane)
        overlay_plane["normal"] = normal.astype(np.float32).tolist()
        overlay_plane["center"] = start.tolist()
        overlay_plane["label_position"] = label_position.tolist()
        overlay_planes.append(overlay_plane)

    overlay = {
        "description": "Floor normals after rigidly aligning selected frames to the reference normal.",
        "source_normal_debug_json": str(args.normal_debug_json),
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "reference_viewer_frame": ref_frame,
        "aligned_viewer_frames": align_frames,
        "segments": overlay_segments,
        "labels": overlay_labels,
        "planes": overlay_planes,
        "line_width": debug.get("line_width", 6.0),
    }
    overlay_path = args.output_dir / "floor_normal_alignment_debug.json"
    overlay_path.write_text(json.dumps(overlay, indent=2, sort_keys=True), encoding="utf-8")

    metrics = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "normal_debug_json": str(args.normal_debug_json),
        "reference_viewer_frame": ref_frame,
        "reference_normal": ref_normal.astype(np.float32).tolist(),
        "aligned": records,
        "overlay_json": str(overlay_path),
    }
    metrics_path = args.output_dir / "floor_normal_alignment_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
