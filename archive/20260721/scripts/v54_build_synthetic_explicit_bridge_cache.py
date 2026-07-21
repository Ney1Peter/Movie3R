#!/usr/bin/env python3
"""Build compact explicit-geometry caches for the V54 synthetic shot bridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V10 = (
    REPO_ROOT
    / "output"
    / "v10_candidate_selection"
    / "oracle_gt_4source"
    / "oracle_candidate_selection_metrics.json"
)
DEFAULT_V17 = REPO_ROOT / "output" / "v17_direct_vs_factorized" / "feature_cache"
DEFAULT_V53 = (
    REPO_ROOT
    / "output"
    / "v53_uniform_similarity_integrity"
    / "v53_uniform_similarity_integrity_probe.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "output" / "v54_synthetic_explicit_shot_bridge" / "cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v10_report", type=Path, default=DEFAULT_V10)
    parser.add_argument("--v17_dir", type=Path, default=DEFAULT_V17)
    parser.add_argument("--v53_report", type=Path, default=DEFAULT_V53)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--points_per_frame", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260729)
    return parser.parse_args()


def sample_background_pixels(
    depth: np.ndarray,
    confidence: np.ndarray,
    mask: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    height, width = depth.shape
    valid = (
        np.isfinite(depth)
        & np.isfinite(confidence)
        & (depth > 0.05)
        & (depth < 50.0)
        & (mask < 0.10)
    )
    if int(valid.sum()) < count:
        valid = np.isfinite(depth) & np.isfinite(confidence) & (depth > 0.05) & (depth < 50.0)
    ys, xs = np.nonzero(valid)
    if len(ys) == 0:
        raise RuntimeError("Frame has no valid depth pixels")

    # Keep broad image coverage before filling the remaining slots randomly.
    selected: list[int] = []
    grid_y, grid_x = 24, 18
    for gy in range(grid_y):
        y0, y1 = gy * height // grid_y, (gy + 1) * height // grid_y
        for gx in range(grid_x):
            x0, x1 = gx * width // grid_x, (gx + 1) * width // grid_x
            inside = np.flatnonzero((ys >= y0) & (ys < y1) & (xs >= x0) & (xs < x1))
            if len(inside):
                selected.append(int(inside[np.argmax(confidence[ys[inside], xs[inside]])]))
    selected = list(dict.fromkeys(selected))
    remaining = np.setdiff1d(np.arange(len(ys)), np.asarray(selected, dtype=np.int64), assume_unique=False)
    need = max(0, count - len(selected))
    if need:
        extra = rng.choice(remaining if len(remaining) else np.arange(len(ys)), size=need, replace=need > len(remaining))
        selected.extend(map(int, extra))
    if len(selected) > count:
        selected = selected[:count]
    return np.stack([ys[selected], xs[selected]], axis=-1).astype(np.int64)


def frame_geometry(local_dir: Path, index: int, count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth = np.load(local_dir / "depth" / f"{index:06d}.npy").astype(np.float32)
    confidence = np.load(local_dir / "conf" / f"{index:06d}.npy").astype(np.float32)
    with np.load(local_dir / "camera" / f"{index:06d}.npz") as camera:
        pose = np.asarray(camera["pose"], dtype=np.float32)
        intrinsics = np.asarray(camera["intrinsics"], dtype=np.float32)
    with np.load(local_dir / "smpl" / f"{index:06d}.npz", allow_pickle=True) as human:
        mask = np.asarray(human["msk"], dtype=np.float32)
        mask = mask[0] if mask.ndim == 3 and len(mask) else np.zeros_like(depth)
        translation = np.asarray(human["transl"], dtype=np.float32)
        rotvec = np.asarray(human["rotvec"], dtype=np.float32)
    color = np.asarray(Image.open(local_dir / "color" / f"{index:06d}.png").convert("RGB"), dtype=np.float32) / 255.0
    pixels = sample_background_pixels(depth, confidence, mask, count, rng)
    yy, xx = pixels[:, 0], pixels[:, 1]
    z = depth[yy, xx]
    x = (xx.astype(np.float32) - intrinsics[0, 2]) / max(float(intrinsics[0, 0]), 1e-6) * z
    y = (yy.astype(np.float32) - intrinsics[1, 2]) / max(float(intrinsics[1, 1]), 1e-6) * z
    xyz = np.stack([x, y, z], axis=-1).astype(np.float32)
    rgb = color[yy, xx].astype(np.float32)
    conf = confidence[yy, xx].astype(np.float32)
    conf_lo, conf_hi = np.quantile(conf, [0.10, 0.90])
    conf = np.clip((conf - conf_lo) / max(float(conf_hi - conf_lo), 1e-6), 0.0, 1.0)
    points = np.concatenate([xyz, rgb, conf[:, None]], axis=-1).astype(np.float32)

    anchors = np.zeros((4, 3), dtype=np.float32)
    if len(translation):
        root = translation[0]
        root_rotation = Rotation.from_rotvec(rotvec[0, 0].astype(np.float64)).as_matrix().astype(np.float32)
        anchors[0] = root
        anchors[1] = root + 0.30 * root_rotation[:, 0]
        anchors[2] = root + 0.30 * root_rotation[:, 1]
        anchors[3] = root + 0.30 * root_rotation[:, 2]
    return points, anchors, pose


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    v10 = json.loads(args.v10_report.read_text(encoding="utf-8"))
    v17_meta = json.loads((args.v17_dir / "v17_explicit_features.json").read_text(encoding="utf-8"))
    v17_arrays = np.load(args.v17_dir / "v17_explicit_features.npz")
    v53 = json.loads(args.v53_report.read_text(encoding="utf-8"))
    v53_cases = {str(case["case_name"]): case for case in v53["cases"]}
    v17_index = {str(row["case_name"]): index for index, row in enumerate(v17_meta["rows"])}

    cases = sorted(v10["cases"], key=lambda case: str(case["case_name"]))
    rng = np.random.default_rng(int(args.seed))
    point_rows, anchor_rows, pose_rows, scale_rows, metadata = [], [], [], [], []
    relative_target, relative_fixed, relative_torso = [], [], []
    for case_index, case in enumerate(cases):
        name = str(case["case_name"])
        local_dir = Path(case["paths"]["human3r_local_reset"])
        frame_points, frame_anchors, frame_poses = [], [], []
        for frame_index in range(4):
            points, anchors, pose = frame_geometry(
                local_dir,
                frame_index,
                int(args.points_per_frame),
                rng,
            )
            frame_points.append(points)
            frame_anchors.append(anchors)
            frame_poses.append(pose)
        scale_case = v53_cases[name]
        scales = np.asarray(
            [
                scale_case["scene_scales"]["old"],
                scale_case["scene_scales"]["old"],
                scale_case["scene_scales"]["new"],
                scale_case["scene_scales"]["new"],
            ],
            dtype=np.float32,
        )
        idx = v17_index[name]
        row = v17_meta["rows"][idx]
        point_rows.append(np.stack(frame_points))
        anchor_rows.append(np.stack(frame_anchors))
        pose_rows.append(np.stack(frame_poses))
        scale_rows.append(scales)
        relative_target.append(v17_arrays["relative_target"][idx])
        relative_fixed.append(v17_arrays["relative_fixed"][idx])
        relative_torso.append(v17_arrays["relative_torso"][idx])
        metadata.append(
            {
                "case_name": name,
                "source": str(row["source"]),
                "capture": str(row["capture"]),
                "camera_pair": str(row["camera_pair"]),
                "view_angle_deg": float(row["view_angle_deg"]),
                "angle_bucket": str(row["angle_bucket"]),
            }
        )
        if (case_index + 1) % 20 == 0:
            print(f"V54 cache {case_index + 1}/{len(cases)}", flush=True)

    cache_path = args.output_dir / "v54_explicit_geometry.npz"
    np.savez_compressed(
        cache_path,
        points=np.stack(point_rows).astype(np.float32),
        human_anchors=np.stack(anchor_rows).astype(np.float32),
        poses=np.stack(pose_rows).astype(np.float32),
        da3_scales=np.stack(scale_rows).astype(np.float32),
        relative_target=np.stack(relative_target).astype(np.float32),
        relative_fixed=np.stack(relative_fixed).astype(np.float32),
        relative_torso=np.stack(relative_torso).astype(np.float32),
    )
    report = {
        "experiment": "V54 synthetic explicit geometry shot bridge cache",
        "case_count": len(metadata),
        "points_per_frame": int(args.points_per_frame),
        "point_features": ["x", "y", "z", "r", "g", "b", "confidence"],
        "human_anchors": ["root", "root_x_axis", "root_y_axis", "root_z_axis"],
        "protocol": {
            "geometry": "frozen Human3R predicted depth, camera, confidence, mask, and SMPL-X",
            "scale": "V53 causal DA3 scene scale, stored separately from raw geometry",
            "gt_use": "relative camera pose target for training/evaluation only",
            "raw_tokens_used": False,
            "gt_depth_used": False,
        },
        "rows": metadata,
    }
    report_path = args.output_dir / "v54_explicit_geometry.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"cache": str(cache_path), "shape": list(np.stack(point_rows).shape)}, indent=2))


if __name__ == "__main__":
    main()
