#!/usr/bin/env python3
"""Probe whether token-aligned near-foot regions provide stable floor normals.

The goal is diagnostic only:

1. reuse the same near-foot token heatmap used in V8.1 token validation;
2. gate the explicit near-foot background region by high token similarity;
3. fit a local 3D plane normal from GT depth and Human3R predicted depth;
4. visualize the candidate floor region and normal arrows.

This checks whether low-texture floor/wall geometry can provide a usable cue
even when background feature matching is weak.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.v8_1_probe_aabb_case import (  # noqa: E402
    AvatarReXRawProjector,
    encode_views,
    load_view,
    patch_index,
    token_similarity_heatmap,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data/Avatarrex_output"))
    parser.add_argument("--avatarrex_raw_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data/avatarrex_lbn1"))
    parser.add_argument("--split", default="Training")
    parser.add_argument("--seq_a", default="22070932")
    parser.add_argument("--seq_b", default="22070935")
    parser.add_argument("--start_frame", type=int, default=820)
    parser.add_argument("--model_path", type=Path, default=REPO_ROOT / "src" / "human3r_896L.pth")
    parser.add_argument("--human3r_raw_dir", type=Path, default=REPO_ROOT / "output" / "v8_1_human3r_aabb_compare" / "raw")
    parser.add_argument(
        "--human3r_pose_dir",
        type=Path,
        default=REPO_ROOT / "output" / "v8_1_human3r_aabb_compare" / "token_aligned_human_only",
        help="Pose directory used to transform Human3R depth normals into corrected world coordinates.",
    )
    parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "output" / "v8_1_floor_normal_token_probe")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--heat_quantile", type=float, default=0.65)
    parser.add_argument("--max_points", type=int, default=4096)
    parser.add_argument("--normal_length", type=float, default=0.35)
    return parser.parse_args()


def backproject(depth: np.ndarray, K: np.ndarray, region: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    valid = region & np.isfinite(depth) & (depth > 0.05) & (depth < 20.0)
    ys, xs = np.where(valid)
    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.float64)
    if len(xs) > max_points:
        rng = np.random.default_rng(17)
        keep = rng.choice(len(xs), size=max_points, replace=False)
        ys, xs = ys[keep], xs[keep]
    z = depth[ys, xs].astype(np.float64)
    x = (xs.astype(np.float64) - K[0, 2]) / K[0, 0] * z
    y = (ys.astype(np.float64) - K[1, 2]) / K[1, 1] * z
    return np.stack([x, y, z], axis=1), np.stack([xs, ys], axis=1).astype(np.float64)


def fit_plane_normal(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    if points.shape[0] < 8:
        return np.full(3, np.nan), np.full(3, np.nan), float("nan")
    centroid = np.median(points, axis=0)
    centered = points - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1].astype(np.float64)
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    # OpenCV camera coordinates: +y is downward, so a floor upward normal tends
    # to have negative y in the image camera.
    if normal[1] > 0:
        normal = -normal
    residual = np.abs(centered @ normal)
    return normal, centroid, float(np.median(residual))


def project(K: np.ndarray, xyz: np.ndarray) -> tuple[int, int] | None:
    if not np.isfinite(xyz).all() or xyz[2] <= 1e-5:
        return None
    uv = xyz[:2] / xyz[2]
    x = int(round(uv[0] * K[0, 0] + K[0, 2]))
    y = int(round(uv[1] * K[1, 1] + K[1, 2]))
    return x, y


def draw_normal_arrow(
    image: np.ndarray,
    K: np.ndarray,
    centroid: np.ndarray,
    normal: np.ndarray,
    length: float,
    color: tuple[int, int, int],
    label: str,
    y_offset: int,
) -> None:
    p0 = project(K, centroid)
    p1 = project(K, centroid + normal * length)
    if p0 is None or p1 is None:
        return
    cv2.arrowedLine(image, p0, p1, color, 3, cv2.LINE_AA, tipLength=0.18)
    cv2.circle(image, p0, 5, color, -1, cv2.LINE_AA)
    cv2.putText(image, label, (12, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)


def load_camera(output_dir: Path, idx: int) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(output_dir / "camera" / f"{idx:06d}.npz")
    return data["pose"].astype(np.float64), data["intrinsics"].astype(np.float64)


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return float("nan")
    a = a / max(float(np.linalg.norm(a)), 1e-12)
    b = b / max(float(np.linalg.norm(b)), 1e-12)
    return float(np.degrees(np.arccos(np.clip(abs(float(a @ b)), -1.0, 1.0))))


def make_token_candidate(heat: np.ndarray, explicit_near_foot: np.ndarray, quantile: float) -> tuple[np.ndarray, float]:
    if not explicit_near_foot.any():
        return explicit_near_foot.copy(), float("nan")
    threshold = float(np.quantile(heat[explicit_near_foot], quantile))
    candidate = explicit_near_foot & (heat >= threshold)
    if candidate.sum() < 64:
        threshold = float(np.quantile(heat[explicit_near_foot], 0.35))
        candidate = explicit_near_foot & (heat >= threshold)
    return candidate, threshold


def normal_world(pose: np.ndarray, normal_cam: np.ndarray) -> np.ndarray:
    if not np.isfinite(normal_cam).all():
        return np.full(3, np.nan)
    n = pose[:3, :3] @ normal_cam
    return n / max(float(np.linalg.norm(n)), 1e-12)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = args.output_dir / "normal_heatmaps"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    projector = AvatarReXRawProjector(args.avatarrex_raw_root)
    case = [
        (args.seq_a, args.start_frame, "view0_A_t"),
        (args.seq_a, args.start_frame + 1, "view1_A_t1"),
        (args.seq_b, args.start_frame + 2, "view2_B_t2_boundary"),
        (args.seq_b, args.start_frame + 3, "view3_B_t3"),
    ]
    views = [load_view(args, projector, i, seq, frame, label) for i, (seq, frame, label) in enumerate(case)]
    tokens, grid_hw = encode_views(args, views)

    rows: list[dict] = []
    per_view_images = []
    for view in views:
        token_idx, _, _ = patch_index(view.anchors["near_foot"], grid_hw)
        heat = token_similarity_heatmap(tokens[view.view_idx], token_idx, grid_hw, view.rgb.shape[:2])
        candidate, threshold = make_token_candidate(heat, view.masks["near_foot"], args.heat_quantile)

        gt_points, _ = backproject(view.depth_m.astype(np.float64), view.intrinsics.astype(np.float64), candidate, args.max_points)
        gt_normal_cam, gt_centroid, gt_residual = fit_plane_normal(gt_points)
        gt_normal_world = normal_world(view.pose.astype(np.float64), gt_normal_cam)

        raw_pose, raw_K = load_camera(args.human3r_raw_dir, view.view_idx)
        corr_pose, _ = load_camera(args.human3r_pose_dir, view.view_idx)
        h3r_depth = np.load(args.human3r_raw_dir / "depth" / f"{view.view_idx:06d}.npy").astype(np.float64)
        h3r_points, _ = backproject(h3r_depth, raw_K, candidate, args.max_points)
        h3r_normal_cam, h3r_centroid, h3r_residual = fit_plane_normal(h3r_points)
        h3r_normal_world_raw = normal_world(raw_pose, h3r_normal_cam)
        h3r_normal_world_corr = normal_world(corr_pose, h3r_normal_cam)

        heat_u8 = np.clip(heat * 255.0, 0, 255).astype(np.uint8)
        heat_color = cv2.cvtColor(cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        overlay = (0.52 * view.rgb + 0.48 * heat_color).astype(np.uint8)
        candidate_color = np.zeros_like(overlay)
        candidate_color[candidate] = np.array([0, 255, 255], dtype=np.uint8)
        overlay = np.where(candidate[..., None], (0.60 * overlay + 0.40 * candidate_color).astype(np.uint8), overlay)
        contours, _ = cv2.findContours(candidate.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)

        draw_normal_arrow(overlay, view.intrinsics.astype(np.float64), gt_centroid, gt_normal_cam, args.normal_length, (255, 230, 0), "GT depth normal", 42)
        draw_normal_arrow(overlay, raw_K, h3r_centroid, h3r_normal_cam, args.normal_length, (40, 255, 255), "Human3R depth normal", 64)
        title = f"{view.label} {view.seq}@{view.frame} near-foot token floor normal"
        cv2.putText(overlay, title, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            overlay,
            f"candidate={int(candidate.sum())} heat_thr={threshold:.3f} GT_res={gt_residual:.4f} H3R_res={h3r_residual:.4f}",
            (12, overlay.shape[0] - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        out_path = overlay_dir / f"view{view.view_idx}_near_foot_floor_normal.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        per_view_images.append(overlay)

        rows.append(
            {
                "view_idx": view.view_idx,
                "label": view.label,
                "seq": view.seq,
                "frame": view.frame,
                "candidate_pixels": int(candidate.sum()),
                "heat_threshold": threshold,
                "gt_num_points": int(gt_points.shape[0]),
                "gt_plane_residual_median": gt_residual,
                "gt_normal_cam_x": float(gt_normal_cam[0]),
                "gt_normal_cam_y": float(gt_normal_cam[1]),
                "gt_normal_cam_z": float(gt_normal_cam[2]),
                "gt_normal_world_x": float(gt_normal_world[0]),
                "gt_normal_world_y": float(gt_normal_world[1]),
                "gt_normal_world_z": float(gt_normal_world[2]),
                "h3r_num_points": int(h3r_points.shape[0]),
                "h3r_plane_residual_median": h3r_residual,
                "h3r_normal_cam_x": float(h3r_normal_cam[0]),
                "h3r_normal_cam_y": float(h3r_normal_cam[1]),
                "h3r_normal_cam_z": float(h3r_normal_cam[2]),
                "h3r_normal_world_raw_x": float(h3r_normal_world_raw[0]),
                "h3r_normal_world_raw_y": float(h3r_normal_world_raw[1]),
                "h3r_normal_world_raw_z": float(h3r_normal_world_raw[2]),
                "h3r_normal_world_token_human_x": float(h3r_normal_world_corr[0]),
                "h3r_normal_world_token_human_y": float(h3r_normal_world_corr[1]),
                "h3r_normal_world_token_human_z": float(h3r_normal_world_corr[2]),
                "gt_floor_like_minus_y_dot": float(-gt_normal_cam[1]) if np.isfinite(gt_normal_cam).all() else float("nan"),
                "h3r_floor_like_minus_y_dot": float(-h3r_normal_cam[1]) if np.isfinite(h3r_normal_cam).all() else float("nan"),
            }
        )

    grid = np.concatenate(
        [
            np.concatenate(per_view_images[:2], axis=1),
            np.concatenate(per_view_images[2:], axis=1),
        ],
        axis=0,
    )
    cv2.imwrite(str(args.output_dir / "floor_normal_heatmap_grid.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

    angle_summary = {}
    for prefix in ["gt_normal_world", "h3r_normal_world_raw", "h3r_normal_world_token_human"]:
        normals = [np.array([row[f"{prefix}_x"], row[f"{prefix}_y"], row[f"{prefix}_z"]], dtype=np.float64) for row in rows]
        angle_summary[prefix] = {
            "view0_to_view1_deg": angle_deg(normals[0], normals[1]),
            "view1_to_view2_boundary_deg": angle_deg(normals[1], normals[2]),
            "view2_to_view3_deg": angle_deg(normals[2], normals[3]),
        }

    csv_path = args.output_dir / "floor_normal_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "case": [{"idx": i, "seq": seq, "frame": frame, "label": label} for i, (seq, frame, label) in enumerate(case)],
        "region": "near_foot token heatmap gated explicit near-foot background",
        "heat_quantile": args.heat_quantile,
        "angle_summary": angle_summary,
        "outputs": {
            "grid": str(args.output_dir / "floor_normal_heatmap_grid.png"),
            "overlays": str(overlay_dir),
            "metrics_csv": str(csv_path),
        },
    }
    with open(args.output_dir / "floor_normal_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
