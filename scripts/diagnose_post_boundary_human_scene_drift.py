#!/usr/bin/env python3
"""Diagnose whether post-boundary camera drift is human-driven or scene-driven.

This script is read-only for Human3R saved outputs. It measures consecutive
post-boundary pairs using only each pair's current saved pose / SMPL / depth / conf
data, then writes compact JSON/CSV diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from overfit_human_anchor_pose_correction import (
    FOOT_JOINTS,
    STABLE_JOINTS,
    infer_num_frames,
    load_sequence,
)
from overfit_single_boundary_frame_scene_geometry import estimate_top_planes, match_planes, sample_points
from overfit_single_boundary_frame_scene_normal import load_background_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--source_video", type=Path, default=None)
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--boundary", type=int, required=True)
    parser.add_argument("--end_frame", type=int, default=None, help="Inclusive end frame. Defaults to boundary+30 or sequence end.")
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, default=None)
    parser.add_argument("--conf_threshold", type=float, default=1.5)
    parser.add_argument("--mask_threshold", type=float, default=0.1)
    parser.add_argument("--plane_max_points", type=int, default=12000)
    parser.add_argument("--plane_iterations", type=int, default=768)
    parser.add_argument("--plane_threshold", type=float, default=0.04)
    parser.add_argument("--num_planes", type=int, default=3)
    parser.add_argument("--min_plane_inlier_ratio", type=float, default=0.03)
    parser.add_argument("--min_plane_dot", type=float, default=0.9)
    parser.add_argument("--bg_chamfer_points", type=int, default=1500)
    parser.add_argument("--bg_chamfer_cap", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def rotation_angle_deg(R_rel: np.ndarray) -> float:
    tr = float(np.trace(R_rel))
    return float(np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))))


def normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def torso_frame_np(joints: np.ndarray) -> np.ndarray:
    left_hip, right_hip = joints[:, 1], joints[:, 2]
    left_shoulder, right_shoulder = joints[:, 16], joints[:, 17]
    hip_mid = 0.5 * (left_hip + right_hip)
    shoulder_mid = 0.5 * (left_shoulder + right_shoulder)
    up = normalize(shoulder_mid - hip_mid)
    right = normalize(right_shoulder - left_shoulder)
    forward = normalize(np.cross(right, up))
    return np.stack([right, up, forward], axis=1)


def chamfer_np(points: np.ndarray, ref_points: np.ndarray, cap: float | None = None) -> float:
    diff = points[:, None, :] - ref_points[None, :, :]
    d = np.linalg.norm(diff, axis=-1)
    cur_to_ref = d.min(axis=1)
    ref_to_cur = d.min(axis=0)
    if cap is not None:
        cur_to_ref = np.minimum(cur_to_ref, float(cap))
        ref_to_cur = np.minimum(ref_to_cur, float(cap))
    return float(0.5 * (cur_to_ref.mean() + ref_to_cur.mean()))


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or len(y) < 3:
        return None
    x_np = np.asarray(x, dtype=np.float64)
    y_np = np.asarray(y, dtype=np.float64)
    if float(x_np.std()) < 1e-8 or float(y_np.std()) < 1e-8:
        return None
    return float(np.corrcoef(x_np, y_np)[0, 1])


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def build_frame_scene(input_dir: Path, frame_id: int, pose: np.ndarray, intrinsics: np.ndarray, args: argparse.Namespace) -> dict:
    points = load_background_points(input_dir, frame_id, pose, intrinsics, args.conf_threshold, args.mask_threshold)
    plane_args = SimpleNamespace(
        plane_max_points=args.plane_max_points,
        plane_iterations=args.plane_iterations,
        plane_threshold=args.plane_threshold,
        num_planes=args.num_planes,
        min_plane_inlier_ratio=args.min_plane_inlier_ratio,
    )
    planes = estimate_top_planes(points, plane_args, seed=3001 + 17 * frame_id)
    sampled = sample_points(points, args.bg_chamfer_points, seed=4001 + 17 * frame_id)
    return {"num_points": int(points.shape[0]), "planes": planes, "sampled_bg": sampled}


def plane_summary(planes: list[dict]) -> list[dict]:
    out = []
    for p in planes:
        out.append(
            {
                "normal": np.asarray(p["normal"], dtype=np.float32).tolist(),
                "d": float(p["d"]),
                "inlier_ratio": float(p["inlier_ratio"]),
                "global_inlier_ratio": float(p.get("global_inlier_ratio", p["inlier_ratio"])),
                "num_inliers": int(p["num_inliers"]),
                "num_points": int(p["num_points"]),
            }
        )
    return out


def diagnose(args: argparse.Namespace) -> dict:
    num_frames = infer_num_frames(args.input_dir, args.source_video, args.num_frames)
    start = int(args.boundary)
    end = int(args.end_frame) if args.end_frame is not None else min(num_frames - 1, start + 30)
    end = min(end, num_frames - 1)
    if start <= 0 or start >= num_frames:
        raise ValueError(f"boundary must be in [1, {num_frames - 1}], got {start}")
    if end <= start:
        raise ValueError(f"end_frame must be > boundary, got {end}")

    data = load_sequence(args.input_dir, num_frames, torch.device(args.device))
    torso_world = torso_frame_np(data.joints_world)
    torso_cam = torso_frame_np(data.joints_cam)

    frame_ids = list(range(start - 1, end + 1))
    scenes = {}
    scene_errors = {}
    for frame_id in frame_ids:
        try:
            scenes[frame_id] = build_frame_scene(args.input_dir, frame_id, data.poses[frame_id], data.intrinsics[frame_id], args)
        except Exception as exc:  # noqa: BLE001 - diagnostics should keep going.
            scene_errors[frame_id] = str(exc)

    rows = []
    for cur in range(start, end + 1):
        prev = cur - 1
        R_prev = data.poses[prev, :3, :3]
        R_cur = data.poses[cur, :3, :3]
        t_prev = data.poses[prev, :3, 3]
        t_cur = data.poses[cur, :3, 3]
        cam_delta = t_cur - t_prev

        torso_world_dots = np.sum(torso_world[prev] * torso_world[cur], axis=-1).clip(-1.0, 1.0)
        torso_cam_dots = np.sum(torso_cam[prev] * torso_cam[cur], axis=-1).clip(-1.0, 1.0)
        pelvis_cam_delta_world_axis = R_prev @ (data.joints_cam[cur, 0] - data.joints_cam[prev, 0])
        denom = max(float(np.linalg.norm(cam_delta) * np.linalg.norm(pelvis_cam_delta_world_axis)), 1e-8)

        row = {
            "pair": f"{prev}->{cur}",
            "prev": prev,
            "cur": cur,
            "camera_step_t": float(np.linalg.norm(cam_delta)),
            "camera_step_r_deg": rotation_angle_deg(R_cur @ R_prev.T),
            "pelvis_world_step": float(np.linalg.norm(data.joints_world[cur, 0] - data.joints_world[prev, 0])),
            "pelvis_cam_step": float(np.linalg.norm(data.joints_cam[cur, 0] - data.joints_cam[prev, 0])),
            "stable_world_chamfer": chamfer_np(data.joints_world[cur, STABLE_JOINTS], data.joints_world[prev, STABLE_JOINTS]),
            "stable_cam_chamfer": chamfer_np(data.joints_cam[cur, STABLE_JOINTS], data.joints_cam[prev, STABLE_JOINTS]),
            "foot_world_chamfer": chamfer_np(data.joints_world[cur, FOOT_JOINTS], data.joints_world[prev, FOOT_JOINTS]),
            "foot_cam_chamfer": chamfer_np(data.joints_cam[cur, FOOT_JOINTS], data.joints_cam[prev, FOOT_JOINTS]),
            "torso_world_step_deg": float(np.degrees(np.arccos(torso_world_dots)).mean()),
            "torso_cam_step_deg": float(np.degrees(np.arccos(torso_cam_dots)).mean()),
            "camera_vs_pelvis_cam_cos": float(np.dot(cam_delta, pelvis_cam_delta_world_axis) / denom),
        }

        if prev in scenes and cur in scenes:
            matches = match_planes(scenes[prev]["planes"], scenes[cur]["planes"], args.min_plane_dot)
            offset_diffs = [abs(float(m["cur"]["d"]) - float(m["ref"]["d"])) for m in matches]
            row.update(
                {
                    "bg_chamfer": chamfer_np(scenes[cur]["sampled_bg"], scenes[prev]["sampled_bg"], cap=args.bg_chamfer_cap),
                    "plane_matches": int(len(matches)),
                    "plane_mean_dot": float(np.mean([float(m["dot_abs"]) for m in matches])) if matches else None,
                    "plane_max_offset_diff": float(max(offset_diffs)) if offset_diffs else None,
                    "plane_weight_sum": float(sum(float(m["weight"]) for m in matches)),
                    "prev_bg_points": int(scenes[prev]["num_points"]),
                    "cur_bg_points": int(scenes[cur]["num_points"]),
                }
            )
        else:
            row.update(
                {
                    "bg_chamfer": None,
                    "plane_matches": 0,
                    "plane_mean_dot": None,
                    "plane_max_offset_diff": None,
                    "plane_weight_sum": 0.0,
                    "prev_bg_points": int(scenes[prev]["num_points"]) if prev in scenes else 0,
                    "cur_bg_points": int(scenes[cur]["num_points"]) if cur in scenes else 0,
                }
            )
        rows.append(row)

    numeric = {k: [float(r[k]) for r in rows if r.get(k) is not None] for k in rows[0].keys() if isinstance(rows[0].get(k), (int, float))}
    summary = {
        "input_dir": str(args.input_dir),
        "num_frames": int(num_frames),
        "boundary": int(start),
        "end_frame": int(end),
        "num_pairs": len(rows),
        "mean_camera_step_t": float(np.mean(numeric["camera_step_t"])),
        "p90_camera_step_t": percentile(numeric["camera_step_t"], 90),
        "mean_pelvis_world_step": float(np.mean(numeric["pelvis_world_step"])),
        "mean_pelvis_cam_step": float(np.mean(numeric["pelvis_cam_step"])),
        "mean_bg_chamfer": float(np.mean(numeric["bg_chamfer"])) if numeric.get("bg_chamfer") else None,
        "p90_bg_chamfer": percentile(numeric.get("bg_chamfer", []), 90),
        "corr_camera_step_vs_pelvis_cam_step": pearson(numeric["camera_step_t"], numeric["pelvis_cam_step"]),
        "corr_camera_step_vs_pelvis_world_step": pearson(numeric["camera_step_t"], numeric["pelvis_world_step"]),
        "corr_camera_step_vs_bg_chamfer": pearson(numeric["camera_step_t"], numeric.get("bg_chamfer", [])),
        "top_camera_step_pairs": sorted(rows, key=lambda r: r["camera_step_t"], reverse=True)[:8],
        "top_bg_chamfer_pairs": sorted([r for r in rows if r.get("bg_chamfer") is not None], key=lambda r: r["bg_chamfer"], reverse=True)[:8],
        "scene_errors": scene_errors,
    }
    return {
        "summary": summary,
        "rows": rows,
        "frames": {
            str(k): {
                "num_points": int(v["num_points"]),
                "planes": plane_summary(v["planes"]),
            }
            for k, v in scenes.items()
        },
    }


def main() -> None:
    args = parse_args()
    result = diagnose(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(result["rows"][0].keys()))
            writer.writeheader()
            writer.writerows(result["rows"])
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
