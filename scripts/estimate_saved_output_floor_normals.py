#!/usr/bin/env python3
"""Estimate floor-normal debug overlays from a saved Human3R output directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True, help="Saved-output directory to inspect.")
    parser.add_argument("--json_out", type=Path, required=True, help="Debug overlay JSON to write.")
    parser.add_argument("--frames", type=int, nargs="*", default=None, help="Viewer frame ids. Defaults to all camera frames.")
    parser.add_argument("--raw_frame_offset", type=int, default=0, help="Raw-frame offset used only for labels.")
    parser.add_argument("--conf_threshold", type=float, default=1.5)
    parser.add_argument("--mask_threshold", type=float, default=0.1)
    parser.add_argument("--bottom_start", type=float, default=0.55)
    parser.add_argument("--max_points", type=int, default=20000)
    parser.add_argument("--iterations", type=int, default=2048)
    parser.add_argument("--plane_threshold", type=float, default=0.04)
    parser.add_argument("--line_length", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=9401)
    return parser.parse_args()


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def infer_num_frames(output_dir: Path) -> int:
    files = sorted((output_dir / "camera").glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No camera npz files under {output_dir / 'camera'}")
    return len(files)


def load_bottom_background_points(output_dir: Path, frame: int, conf_threshold: float, mask_threshold: float, bottom_start: float) -> np.ndarray:
    cam = np.load(output_dir / "camera" / f"{frame:06d}.npz")
    pose = cam["pose"].astype(np.float32)
    intrinsics = cam["intrinsics"].astype(np.float32)
    depth = np.load(output_dir / "depth" / f"{frame:06d}.npy").astype(np.float32)
    conf = np.load(output_dir / "conf" / f"{frame:06d}.npy").astype(np.float32)
    points_world, _ = depthmap_to_absolute_camera_coordinates(depth, intrinsics, pose)

    valid = np.isfinite(points_world).all(axis=-1) & np.isfinite(depth) & (depth > 0.0) & np.isfinite(conf) & (conf >= float(conf_threshold))
    y0 = int(np.clip(float(bottom_start), 0.0, 0.95) * depth.shape[0])
    region = np.zeros_like(valid, dtype=bool)
    region[y0:, :] = True
    valid &= region

    smpl_path = output_dir / "smpl" / f"{frame:06d}.npz"
    if smpl_path.is_file():
        smpl = np.load(smpl_path, allow_pickle=True)
        if "msk" in smpl.files:
            msk = smpl["msk"]
            if msk is not None and np.size(msk) > 0:
                human_mask = np.max(msk.astype(np.float32), axis=0) > float(mask_threshold)
                valid &= ~human_mask

    points = points_world[valid].astype(np.float32)
    if points.shape[0] < 3:
        raise ValueError(f"Too few bottom background points for frame {frame}: {points.shape[0]}")
    return points


def estimate_plane(points: np.ndarray, max_points: int, iterations: int, threshold: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    all_count = int(points.shape[0])
    pts = points.astype(np.float32)
    if pts.shape[0] > int(max_points):
        pts = pts[rng.choice(pts.shape[0], size=int(max_points), replace=False)]

    best_count = -1
    best_normal = None
    best_d = None
    batch = 128
    for start in range(0, int(iterations), batch):
        count = min(batch, int(iterations) - start)
        idx = rng.integers(0, pts.shape[0], size=(count, 3))
        p0, p1, p2 = pts[idx[:, 0]], pts[idx[:, 1]], pts[idx[:, 2]]
        normals = np.cross(p1 - p0, p2 - p0)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        valid = norms[:, 0] > 1e-6
        if not np.any(valid):
            continue
        normals = normals[valid] / norms[valid]
        p0 = p0[valid]
        d = -np.sum(normals * p0, axis=1)
        dist = np.abs(pts @ normals.T + d[None])
        inlier_counts = np.sum(dist < float(threshold), axis=0)
        local_best = int(np.argmax(inlier_counts))
        if int(inlier_counts[local_best]) > best_count:
            best_count = int(inlier_counts[local_best])
            best_normal = normals[local_best].astype(np.float32)
            best_d = float(d[local_best])

    if best_normal is None:
        raise ValueError("RANSAC failed to estimate a floor plane")

    inliers = np.abs(pts @ best_normal + best_d) < float(threshold)
    inlier_pts = pts[inliers]
    if inlier_pts.shape[0] >= 3:
        center = inlier_pts.mean(axis=0)
        _, _, vh = np.linalg.svd(inlier_pts - center, full_matrices=False)
        best_normal = normalize(vh[-1]).astype(np.float32)
        best_d = -float(best_normal @ center)
        inliers = np.abs(pts @ best_normal + best_d) < float(threshold)
        inlier_pts = pts[inliers]
    else:
        center = pts.mean(axis=0)

    normal = normalize(best_normal).astype(np.float32)
    if normal[1] > 0.0:
        normal = -normal
        best_d = -best_d
    center = inlier_pts.mean(axis=0).astype(np.float32) if inlier_pts.shape[0] else center.astype(np.float32)
    return {
        "normal": normal,
        "d": np.float32(best_d),
        "center": center,
        "num_points_bottom": all_count,
        "num_points_sampled": int(pts.shape[0]),
        "num_inliers": int(np.sum(inliers)),
        "inlier_ratio": float(np.mean(inliers)),
    }


def main() -> None:
    args = parse_args()
    num_frames = infer_num_frames(args.output_dir)
    frames = list(range(num_frames)) if args.frames is None else [int(x) for x in args.frames]
    colors = [[255, 64, 64], [64, 255, 64], [64, 128, 255], [255, 192, 64], [192, 64, 255], [64, 255, 255]]
    segments = []
    labels = []
    planes = []
    for order, frame in enumerate(frames):
        if frame < 0 or frame >= num_frames:
            raise ValueError(f"Frame {frame} outside saved-output frame count {num_frames}")
        points = load_bottom_background_points(args.output_dir, frame, args.conf_threshold, args.mask_threshold, args.bottom_start)
        plane = estimate_plane(points, args.max_points, args.iterations, args.plane_threshold, args.seed + 17 * frame)
        color = colors[order % len(colors)]
        start = plane["center"].astype(np.float32)
        end = (start + float(args.line_length) * plane["normal"]).astype(np.float32)
        label_position = (start + float(args.line_length) * 1.15 * plane["normal"]).astype(np.float32)
        raw_frame = int(frame + int(args.raw_frame_offset))
        segments.append({"label": f"raw{raw_frame}/viewer{frame}", "start": start.tolist(), "end": end.tolist(), "color": color})
        labels.append({"text": f"floor n raw{raw_frame} v{frame}", "position": label_position.tolist(), "height": 0.12})
        planes.append(
            {
                "viewer_frame": int(frame),
                "raw_frame": raw_frame,
                "normal": plane["normal"].astype(np.float32).tolist(),
                "d": float(plane["d"]),
                "num_points_bottom": int(plane["num_points_bottom"]),
                "num_points_sampled": int(plane["num_points_sampled"]),
                "num_inliers": int(plane["num_inliers"]),
                "inlier_ratio": float(plane["inlier_ratio"]),
                "center": start.tolist(),
                "label_position": label_position.tolist(),
                "color": color,
            }
        )

    output = {
        "description": "Bottom-region floor plane normals estimated from saved-output frames.",
        "output_dir": str(args.output_dir),
        "frames": frames,
        "raw_frame_offset": int(args.raw_frame_offset),
        "bottom_start": float(args.bottom_start),
        "conf_threshold": float(args.conf_threshold),
        "mask_threshold": float(args.mask_threshold),
        "plane_threshold": float(args.plane_threshold),
        "segments": segments,
        "labels": labels,
        "planes": planes,
        "line_width": 6.0,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
