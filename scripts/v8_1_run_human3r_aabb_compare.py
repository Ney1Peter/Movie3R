#!/usr/bin/env python3
"""Run Human3R on one AvatarReX AABB case and build raw/corrected visual outputs.

The script reuses demo.py's model/input/output code but skips the blocking viser
viewer.  It writes two saved-output directories:

  raw/       original Human3R camera poses
  corrected/ oracle GT-relative camera correction for visual sanity checking

The correction here is deliberately not a learned module.  It replaces the saved
camera trajectory with the AvatarReX GT relative camera jump, aligned to the raw
first-frame coordinate system.  Depth, confidence, colors, and SMPL camera-space
predictions are unchanged, so the comparison isolates the effect of camera pose.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from add_ckpt_path import add_path_to_dust3r  # noqa: E402
from demo import prepare_input, prepare_output  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--avatarrex_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data"))
    parser.add_argument("--avatarrex_raw_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data/avatarrex_lbn1"))
    parser.add_argument("--split", default="training")
    parser.add_argument("--seq_a", default="22070932")
    parser.add_argument("--seq_b", default="22070935")
    parser.add_argument("--start_frame", type=int, default=820)
    parser.add_argument("--model_path", type=Path, default=REPO_ROOT / "src" / "human3r_896L.pth")
    parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "output" / "v8_1_human3r_aabb_compare")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reset_interval", type=int, default=100)
    parser.add_argument("--vis_threshold", type=float, default=2.0)
    parser.add_argument("--downsample_factor", type=int, default=1)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render_video", action="store_true")
    parser.add_argument(
        "--skip_inference",
        action="store_true",
        help="Reuse an existing raw/ saved-output directory and only rebuild corrected/ plus comparison visualizations.",
    )
    parser.add_argument("--max_points_per_frame", type=int, default=4500)
    return parser.parse_args()


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_aabb_input_frames(args: argparse.Namespace, input_dir: Path) -> list[dict]:
    input_dir.mkdir(parents=True, exist_ok=True)
    case = [
        (args.seq_a, args.start_frame, "view0_A_t"),
        (args.seq_a, args.start_frame + 1, "view1_A_t1"),
        (args.seq_b, args.start_frame + 2, "view2_B_t2_boundary"),
        (args.seq_b, args.start_frame + 3, "view3_B_t3"),
    ]
    specs = []
    for i, (seq, frame, label) in enumerate(case):
        src = args.avatarrex_root / args.split / seq / "rgb" / f"{frame:08d}.png"
        dst = input_dir / f"{i:06d}_{label}_{seq}_{frame:08d}.png"
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(src)
        cv2.imwrite(str(dst), img)
        specs.append({"idx": i, "seq": seq, "frame": frame, "label": label, "path": str(dst)})
    return specs


def load_gt_c2w(raw_root: Path, seq: str) -> np.ndarray:
    with open(raw_root / "calibration_full.json", "r", encoding="utf-8") as f:
        calibration = json.load(f)
    cal = calibration[seq]
    R_w2c = np.asarray(cal["R"], dtype=np.float32).reshape(3, 3)
    T_w2c = np.asarray(cal["T"], dtype=np.float32).reshape(3)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R_w2c.T
    c2w[:3, 3] = -R_w2c.T @ T_w2c
    return c2w


def run_human3r(args: argparse.Namespace, input_dir: Path, raw_dir: Path) -> None:
    import torch
    from src.dust3r.inference import inference_recurrent_lighter
    from src.dust3r.model import ARCroco3DStereo

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    add_path_to_dust3r(str(args.model_path))
    img_paths = sorted(str(p) for p in input_dir.glob("*.png"))
    if not img_paths:
        raise FileNotFoundError(input_dir)

    model = ARCroco3DStereo.from_pretrained(str(args.model_path)).to(args.device).eval()
    img_res = getattr(model, "mhmr_img_res", None)
    views = prepare_input(
        img_paths=img_paths,
        img_mask=[True] * len(img_paths),
        size=args.size,
        revisit=1,
        update=True,
        img_res=img_res,
        reset_interval=args.reset_interval,
    )
    outputs, _ = inference_recurrent_lighter(views, model, args.device, use_ttt3r=False)
    prepare_output(
        outputs,
        str(raw_dir),
        revisit=1,
        use_pose=True,
        save=True,
        render=args.render,
        render_video=args.render_video,
        img_res=img_res,
        subsample=1,
    )


def copy_saved_output_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def load_camera_poses(output_dir: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    cam_files = sorted((output_dir / "camera").glob("*.npz"))
    poses, intrinsics = [], []
    for path in cam_files:
        data = np.load(path)
        poses.append(data["pose"].astype(np.float32))
        intrinsics.append(data["intrinsics"].astype(np.float32))
    return poses, intrinsics


def write_camera_poses(output_dir: Path, poses: list[np.ndarray], intrinsics: list[np.ndarray]) -> None:
    cam_dir = output_dir / "camera"
    for i, (pose, K) in enumerate(zip(poses, intrinsics)):
        np.savez(cam_dir / f"{i:06d}.npz", pose=pose.astype(np.float32), intrinsics=K.astype(np.float32))


def build_oracle_corrected_output(args: argparse.Namespace, specs: list[dict], raw_dir: Path, corrected_dir: Path) -> dict:
    copy_saved_output_tree(raw_dir, corrected_dir)
    # ``demo.py --render_video`` creates a frame-local SMPL projection video. It
    # does not depend on the saved world camera trajectory, so a copied video is
    # misleading for pose-correction inspection.
    copied_video = corrected_dir / "output_video.mp4"
    if copied_video.exists():
        copied_video.unlink()
    raw_poses, intrinsics = load_camera_poses(raw_dir)
    if len(raw_poses) != len(specs):
        raise RuntimeError(f"Expected {len(specs)} raw poses, found {len(raw_poses)}")

    gt_poses = [load_gt_c2w(args.avatarrex_raw_root, spec["seq"]) for spec in specs]
    align = raw_poses[0] @ np.linalg.inv(gt_poses[0])
    corrected = [(align @ gt_pose).astype(np.float32) for gt_pose in gt_poses]
    write_camera_poses(corrected_dir, corrected, intrinsics)

    summary = {
        "correction_type": "oracle_gt_relative_camera_pose",
        "note": "This is only a visual sanity check; it is not the learned V8 pose correction module.",
        "alignment": "T_corr_i = T_raw_0 @ inv(T_gt_0) @ T_gt_i",
        "raw_camera_centers": [pose[:3, 3].tolist() for pose in raw_poses],
        "corrected_camera_centers": [pose[:3, 3].tolist() for pose in corrected],
        "gt_camera_centers": [pose[:3, 3].tolist() for pose in gt_poses],
    }
    with open(corrected_dir / "correction_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary


def backproject_depth(depth: np.ndarray, K: np.ndarray, pose: np.ndarray) -> np.ndarray:
    h, w = depth.shape
    ys, xs = np.mgrid[:h, :w]
    z = depth.astype(np.float32)
    valid = np.isfinite(z) & (z > 0.05) & (z < 50.0)
    x = (xs.astype(np.float32) - K[0, 2]) / K[0, 0] * z
    y = (ys.astype(np.float32) - K[1, 2]) / K[1, 1] * z
    pts_cam = np.stack([x, y, z], axis=-1)
    pts = pts_cam[valid]
    pts_world = pts @ pose[:3, :3].T + pose[:3, 3]
    return pts_world.astype(np.float32)


def sample_output_points(output_dir: Path, max_points_per_frame: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    poses, Ks = load_camera_poses(output_dir)
    pts_list, colors_list = [], []
    rng = np.random.default_rng(42)
    for i, (pose, K) in enumerate(zip(poses, Ks)):
        depth = np.load(output_dir / "depth" / f"{i:06d}.npy").astype(np.float32)
        conf = np.load(output_dir / "conf" / f"{i:06d}.npy").astype(np.float32)
        color_bgr = cv2.imread(str(output_dir / "color" / f"{i:06d}.png"), cv2.IMREAD_COLOR)
        color = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        threshold = np.quantile(conf[np.isfinite(conf)], 0.65)
        valid = np.isfinite(depth) & (depth > 0.05) & (depth < 50.0) & (conf >= threshold)
        ys, xs = np.where(valid)
        if len(xs) > max_points_per_frame:
            keep = rng.choice(len(xs), size=max_points_per_frame, replace=False)
            ys, xs = ys[keep], xs[keep]
        z = depth[ys, xs]
        x = (xs.astype(np.float32) - K[0, 2]) / K[0, 0] * z
        y = (ys.astype(np.float32) - K[1, 2]) / K[1, 1] * z
        pts_cam = np.stack([x, y, z], axis=-1)
        pts_world = pts_cam @ pose[:3, :3].T + pose[:3, 3]
        pts_list.append(pts_world.astype(np.float32))
        colors_list.append(color[ys, xs].astype(np.float32))
    return pts_list, colors_list


def make_camera_frustum(pose: np.ndarray, K: np.ndarray, image_hw: tuple[int, int], scale: float) -> list[np.ndarray]:
    h, w = image_hw
    corners_px = np.asarray(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
        dtype=np.float32,
    )
    z = np.full((4, 1), scale, dtype=np.float32)
    corners_cam = np.concatenate(
        [
            ((corners_px[:, :1] - K[0, 2]) / K[0, 0]) * z,
            ((corners_px[:, 1:] - K[1, 2]) / K[1, 1]) * z,
            z,
        ],
        axis=1,
    )
    center = pose[:3, 3]
    corners = corners_cam @ pose[:3, :3].T + center
    return [
        np.stack([center, corners[0]]),
        np.stack([center, corners[1]]),
        np.stack([center, corners[2]]),
        np.stack([center, corners[3]]),
        np.stack([corners[0], corners[1], corners[2], corners[3], corners[0]]),
    ]


def set_equal_3d_axes(ax, pts_list: list[np.ndarray], poses: list[np.ndarray]) -> None:
    valid_chunks = [p[np.all(np.isfinite(p), axis=1)] for p in pts_list if len(p) > 0]
    centers = np.stack([p[:3, 3] for p in poses])
    all_pts = np.concatenate(valid_chunks + [centers], axis=0)
    lo, hi = np.percentile(all_pts, [2, 98], axis=0)
    center = 0.5 * (lo + hi)
    radius = float(np.max(hi - lo) * 0.55)
    radius = max(radius, 0.5)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def draw_world_scene(
    ax,
    pts_list: list[np.ndarray],
    color_list: list[np.ndarray],
    poses: list[np.ndarray],
    intrinsics: list[np.ndarray],
    depth_shape: tuple[int, int],
    title: str,
    azim: float,
    use_rgb: bool,
) -> None:
    frame_colors = ["tab:blue", "tab:green", "tab:red", "tab:purple"]
    set_equal_3d_axes(ax, pts_list, poses)
    ranges = np.asarray([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
    frustum_scale = float(np.max(ranges[:, 1] - ranges[:, 0]) * 0.06)
    for i, (pts, colors) in enumerate(zip(pts_list, color_list)):
        step = max(1, len(pts) // 2000)
        if use_rgb:
            scatter_color = colors[::step]
        else:
            scatter_color = frame_colors[i % len(frame_colors)]
        ax.scatter(
            pts[::step, 0],
            pts[::step, 1],
            pts[::step, 2],
            s=1.3,
            c=scatter_color,
            alpha=0.5,
            depthshade=False,
        )
    for i, (pose, K) in enumerate(zip(poses, intrinsics)):
        center = pose[:3, 3]
        ax.scatter(center[0], center[1], center[2], s=34, color=frame_colors[i % len(frame_colors)], depthshade=False)
        ax.text(center[0], center[1], center[2], f" {i}", fontsize=9)
        for line in make_camera_frustum(pose, K, depth_shape, frustum_scale):
            ax.plot(line[:, 0], line[:, 1], line[:, 2], color=frame_colors[i % len(frame_colors)], linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=22, azim=azim)
    ax.grid(True, alpha=0.2)


def figure_to_bgr(fig) -> np.ndarray:
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    rgb = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def make_world_pose_visualizations(args: argparse.Namespace, raw_dir: Path, corrected_dir: Path, vis_dir: Path) -> dict:
    raw_poses, raw_Ks = load_camera_poses(raw_dir)
    corr_poses, corr_Ks = load_camera_poses(corrected_dir)
    raw_pts, raw_colors = sample_output_points(raw_dir, args.max_points_per_frame)
    corr_pts, corr_colors = sample_output_points(corrected_dir, args.max_points_per_frame)
    depth_shape = np.load(raw_dir / "depth" / "000000.npy").shape

    saved = {}
    for use_rgb, stem in [(False, "framecolor"), (True, "rgb")]:
        fig = plt.figure(figsize=(13, 6))
        ax_raw = fig.add_subplot(1, 2, 1, projection="3d")
        ax_corr = fig.add_subplot(1, 2, 2, projection="3d")
        draw_world_scene(
            ax_raw,
            raw_pts,
            raw_colors,
            raw_poses,
            raw_Ks,
            depth_shape,
            "Raw Human3R world placement",
            azim=-62,
            use_rgb=use_rgb,
        )
        draw_world_scene(
            ax_corr,
            corr_pts,
            corr_colors,
            corr_poses,
            corr_Ks,
            depth_shape,
            "Oracle-corrected world placement",
            azim=-62,
            use_rgb=use_rgb,
        )
        fig.tight_layout()
        out_path = vis_dir / f"world_3d_raw_vs_corrected_{stem}.png"
        fig.savefig(out_path, dpi=170)
        plt.close(fig)
        saved[f"world_3d_{stem}_png"] = str(out_path)

    video_path = vis_dir / "world_3d_raw_vs_corrected_turntable.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 12, (1560, 720))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {video_path}")
    for azim in np.linspace(-80, 280, 72, endpoint=False):
        fig = plt.figure(figsize=(13, 6), dpi=120)
        ax_raw = fig.add_subplot(1, 2, 1, projection="3d")
        ax_corr = fig.add_subplot(1, 2, 2, projection="3d")
        draw_world_scene(
            ax_raw,
            raw_pts,
            raw_colors,
            raw_poses,
            raw_Ks,
            depth_shape,
            "Raw Human3R world placement",
            azim=float(azim),
            use_rgb=False,
        )
        draw_world_scene(
            ax_corr,
            corr_pts,
            corr_colors,
            corr_poses,
            corr_Ks,
            depth_shape,
            "Oracle-corrected world placement",
            azim=float(azim),
            use_rgb=False,
        )
        fig.tight_layout()
        frame = figure_to_bgr(fig)
        plt.close(fig)
        writer.write(frame)
    writer.release()
    saved["world_3d_turntable_mp4"] = str(video_path)
    return saved


def rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    R = a[:3, :3] @ b[:3, :3].T
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def make_comparison_figures(args: argparse.Namespace, raw_dir: Path, corrected_dir: Path, vis_dir: Path) -> dict:
    vis_dir.mkdir(parents=True, exist_ok=True)
    raw_poses, _ = load_camera_poses(raw_dir)
    corr_poses, _ = load_camera_poses(corrected_dir)
    raw_pts, _ = sample_output_points(raw_dir, args.max_points_per_frame)
    corr_pts, _ = sample_output_points(corrected_dir, args.max_points_per_frame)
    colors = ["tab:blue", "tab:green", "tab:red", "tab:purple"]

    raw_centers = np.stack([p[:3, 3] for p in raw_poses])
    corr_centers = np.stack([p[:3, 3] for p in corr_poses])
    raw_jumps = np.linalg.norm(np.diff(raw_centers, axis=0), axis=1)
    corr_jumps = np.linalg.norm(np.diff(corr_centers, axis=0), axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, centers, title in [
        (axes[0], raw_centers, "Raw Human3R camera centers"),
        (axes[1], corr_centers, "Oracle-corrected camera centers"),
    ]:
        ax.plot(centers[:, 0], centers[:, 2], "-o", color="black")
        for i, c in enumerate(centers):
            ax.text(c[0], c[2], str(i), fontsize=11)
        ax.set_xlabel("world X")
        ax.set_ylabel("world Z")
        ax.set_title(title)
        ax.axis("equal")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(vis_dir / "camera_centers_xz_raw_vs_corrected.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, pts_list, title in [
        (axes[0], raw_pts, "Raw Human3R point clouds"),
        (axes[1], corr_pts, "Pose-corrected point clouds"),
    ]:
        for i, pts in enumerate(pts_list):
            step = max(1, len(pts) // 1800)
            ax.scatter(pts[::step, 0], pts[::step, 2], s=2, color=colors[i], alpha=0.45, label=f"view{i}")
        ax.set_xlabel("world X")
        ax.set_ylabel("world Z")
        ax.set_title(title)
        ax.axis("equal")
        ax.grid(True, alpha=0.2)
    axes[0].legend(markerscale=3)
    fig.tight_layout()
    fig.savefig(vis_dir / "pointcloud_xz_raw_vs_corrected.png", dpi=180)
    plt.close(fig)

    raw_rel_20 = np.linalg.inv(raw_poses[0]) @ raw_poses[2]
    corr_rel_20 = np.linalg.inv(corr_poses[0]) @ corr_poses[2]
    metrics = {
        "raw_camera_centers": raw_centers.tolist(),
        "corrected_camera_centers": corr_centers.tolist(),
        "raw_jump_lengths": raw_jumps.tolist(),
        "corrected_jump_lengths": corr_jumps.tolist(),
        "raw_boundary_translation_norm_0_to_2": float(np.linalg.norm(raw_rel_20[:3, 3])),
        "corrected_boundary_translation_norm_0_to_2": float(np.linalg.norm(corr_rel_20[:3, 3])),
        "raw_boundary_rotation_deg_0_to_2": rotation_error_deg(raw_poses[2], raw_poses[0]),
        "corrected_boundary_rotation_deg_0_to_2": rotation_error_deg(corr_poses[2], corr_poses[0]),
    }
    with open(vis_dir / "comparison_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    return metrics


def main() -> None:
    args = parse_args()
    if not args.model_path.is_file():
        raise FileNotFoundError(args.model_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = args.output_dir / "input_frames"
    raw_dir = args.output_dir / "raw"
    corrected_dir = args.output_dir / "corrected"
    vis_dir = args.output_dir / "comparison"
    ensure_clean_dir(input_dir)
    ensure_clean_dir(raw_dir)
    ensure_clean_dir(vis_dir)

    specs = copy_aabb_input_frames(args, input_dir)
    with open(args.output_dir / "case_manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_path": str(args.model_path),
                "input_frames": specs,
                "raw_dir": str(raw_dir),
                "corrected_dir": str(corrected_dir),
                "comparison_dir": str(vis_dir),
                "note": "Corrected output uses oracle GT-relative cameras for visualization only.",
            },
            f,
            indent=2,
            sort_keys=True,
        )

    run_human3r(args, input_dir, raw_dir)
    correction_summary = build_oracle_corrected_output(args, specs, raw_dir, corrected_dir)
    metrics = make_comparison_figures(args, raw_dir, corrected_dir, vis_dir)
    world_visuals = make_world_pose_visualizations(args, raw_dir, corrected_dir, vis_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "correction": correction_summary, "metrics": metrics, "world_visuals": world_visuals}, indent=2))


if __name__ == "__main__":
    main()
