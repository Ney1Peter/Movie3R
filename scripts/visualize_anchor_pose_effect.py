#!/usr/bin/env python3
"""V6.1 diagnostic: visualize anchor matches and manual pose correction."""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in [REPO_ROOT, REPO_ROOT / "src"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from add_ckpt_path import add_path_to_dust3r  # noqa: E402
from demo import parse_seq_path, prepare_input  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--seq_path", required=True)
    parser.add_argument("--anchor_path", required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--max_frames", type=int, default=66)
    parser.add_argument("--reset_interval", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out_dir", default="output/anchor_pose_visualization/h36_new")
    return parser.parse_args()


def read_video_frame(seq_path, frame_index):
    cap = cv2.VideoCapture(str(seq_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {seq_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"failed to read frame {frame_index}")
    return frame


def human3r_crop(frame_bgr, size):
    h0, w0 = frame_bgr.shape[:2]
    scale = float(size) / float(max(w0, h0))
    w1 = int(round(w0 * scale))
    h1 = int(round(h0 * scale))
    resized = cv2.resize(frame_bgr, (w1, h1), interpolation=cv2.INTER_AREA)
    cx, cy = w1 // 2, h1 // 2
    halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
    if w1 == h1:
        halfh = int(3 * halfw / 4)
    left, top, right, bottom = cx - halfw, cy - halfh, cx + halfw, cy + halfh
    return resized[top:bottom, left:right].copy()


def colors_for_n(n):
    colors = []
    for i in range(n):
        hue = int(round(179 * i / max(n, 1)))
        hsv = np.uint8([[[hue, 210, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        colors.append(tuple(int(v) for v in bgr.tolist()))
    return colors


def draw_points(image, points, colors, title):
    out = image.copy()
    cv2.putText(out, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(out, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 1, cv2.LINE_AA)
    for i, (point, color) in enumerate(zip(points, colors)):
        x, y = int(round(point[0])), int(round(point[1]))
        cv2.circle(out, (x, y), 7, color, -1, cv2.LINE_AA)
        cv2.circle(out, (x, y), 8, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(out, str(i), (x + 7, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(out, str(i), (x + 7, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


def save_anchor_point_images(ref_img, cur_img, ref_points, cur_points, out_dir, ref_idx, cur_idx):
    colors = colors_for_n(len(ref_points))
    ref_vis = draw_points(ref_img, ref_points, colors, f"frame {ref_idx}: ref anchors")
    cur_vis = draw_points(cur_img, cur_points, colors, f"frame {cur_idx}: cur anchors")
    cv2.imwrite(str(out_dir / "frame_0062_ref_anchor_points.jpg"), ref_vis)
    cv2.imwrite(str(out_dir / "frame_0063_cur_anchor_points.jpg"), cur_vis)

    h = max(ref_vis.shape[0], cur_vis.shape[0])
    gap = 30
    canvas = np.full((h, ref_vis.shape[1] + cur_vis.shape[1] + gap, 3), 245, dtype=np.uint8)
    canvas[: ref_vis.shape[0], : ref_vis.shape[1]] = ref_vis
    xoff = ref_vis.shape[1] + gap
    canvas[: cur_vis.shape[0], xoff : xoff + cur_vis.shape[1]] = cur_vis
    for i, color in enumerate(colors):
        p0 = (int(round(ref_points[i][0])), int(round(ref_points[i][1])))
        p1 = (int(round(cur_points[i][0] + xoff)), int(round(cur_points[i][1])))
        cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / "frame_0062_0063_anchor_matches.jpg"), canvas)


def pose_mats(outputs, pose_encoding_to_camera):
    mats = []
    for pred in outputs["pred"]:
        pose = pred["camera_pose"].detach().float()
        mat = pose_encoding_to_camera(pose.clone()).detach().float().reshape(-1, 4, 4)[0]
        mats.append(mat.cpu().numpy())
    return np.stack(mats, axis=0)


def apply_camera(points, c2w):
    return points @ c2w[:3, :3].T + c2w[:3, 3][None]


def fit_rigid(src, dst):
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src0 = src - src_mean
    dst0 = dst - dst_mean
    u, _, vt = np.linalg.svd(src0.T @ dst0)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = dst_mean - rotation @ src_mean
    return rotation.astype(np.float32), translation.astype(np.float32)


def robust_rigid(src, dst):
    rotation, translation = fit_rigid(src, dst)
    residuals = np.linalg.norm(apply_rigid(src, rotation, translation) - dst, axis=1)
    threshold = max(0.05, 2.5 * float(np.median(residuals)))
    inliers = residuals <= threshold
    if int(inliers.sum()) >= 3:
        rotation, translation = fit_rigid(src[inliers], dst[inliers])
    return rotation, translation


def apply_rigid(points, rotation, translation):
    return points @ rotation.T + translation[None]


def residual_stats(values):
    values = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "max": float(values.max()),
    }


def make_mapper(point_sets, width, height):
    all_points = np.concatenate([p[:, [0, 2]] for p in point_sets], axis=0)
    lo = all_points.min(axis=0)
    hi = all_points.max(axis=0)
    center = (lo + hi) * 0.5
    span = float(max((hi - lo).max(), 1e-4)) * 1.18
    margin_x, margin_y = 55, 60

    def mapper(points):
        xz = points[:, [0, 2]]
        norm = (xz - center[None]) / span + 0.5
        x = margin_x + norm[:, 0] * (width - 2 * margin_x)
        y = height - margin_y - norm[:, 1] * (height - 2 * margin_y)
        return np.stack([x, y], axis=1).astype(np.int32)

    return mapper


def draw_scatter_panel(ref_world, cur_world, cam_ref, cam_cur, mapper, title, stats, color_cur):
    width, height = 720, 620
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    ref_px = mapper(ref_world)
    cur_px = mapper(cur_world)
    cam_px = mapper(np.stack([cam_ref, cam_cur], axis=0))
    colors = colors_for_n(len(ref_world))
    for i in range(len(ref_world)):
        cv2.line(canvas, tuple(ref_px[i]), tuple(cur_px[i]), (170, 170, 170), 1, cv2.LINE_AA)
        cv2.circle(canvas, tuple(ref_px[i]), 6, (0, 160, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas, tuple(cur_px[i]), 6, color_cur, -1, cv2.LINE_AA)
        cv2.circle(canvas, tuple(ref_px[i]), 7, colors[i], 1, cv2.LINE_AA)
    cv2.circle(canvas, tuple(cam_px[0]), 10, (0, 120, 0), -1, cv2.LINE_AA)
    cv2.circle(canvas, tuple(cam_px[1]), 10, color_cur, -1, cv2.LINE_AA)
    cv2.putText(canvas, "green=ref anchors/camera", (20, height - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 120, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, title, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    stat_text = f"anchor residual median={stats['median']:.3f}, mean={stats['mean']:.3f}, max={stats['max']:.3f}"
    cv2.putText(canvas, stat_text, (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, "XZ top-down view", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1, cv2.LINE_AA)
    return canvas


def save_correction_image(ref_world, cur_before, cur_after, cam_ref, cam_before, cam_after, out_dir):
    before_residuals = np.linalg.norm(cur_before - ref_world, axis=1)
    after_residuals = np.linalg.norm(cur_after - ref_world, axis=1)
    mapper = make_mapper([ref_world, cur_before, cur_after, np.stack([cam_ref, cam_before, cam_after])], 720, 620)
    before = draw_scatter_panel(
        ref_world,
        cur_before,
        cam_ref,
        cam_before,
        mapper,
        "before manual correction",
        residual_stats(before_residuals),
        (0, 0, 230),
    )
    after = draw_scatter_panel(
        ref_world,
        cur_after,
        cam_ref,
        cam_after,
        mapper,
        "after manual rigid correction",
        residual_stats(after_residuals),
        (230, 80, 0),
    )
    canvas = np.concatenate([before, after], axis=1)
    cv2.imwrite(str(out_dir / "manual_anchor_correction_before_after_xz.jpg"), canvas)
    return residual_stats(before_residuals), residual_stats(after_residuals)


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    anchor = np.load(args.anchor_path)
    ref_idx = int(anchor["ref_view_idx"][0])
    cur_idx = int(anchor["cur_view_idx"][0])
    mask = np.asarray(anchor["anchor_mask"], dtype=bool)
    ref_patch_xy = np.asarray(anchor["ref_patch_xy"], dtype=np.float32)[mask]
    cur_patch_xy = np.asarray(anchor["cur_patch_xy"], dtype=np.float32)[mask]
    ref_points = (ref_patch_xy + 0.5) * 16.0
    cur_points = (cur_patch_xy + 0.5) * 16.0

    ref_img = human3r_crop(read_video_frame(args.seq_path, ref_idx), args.size)
    cur_img = human3r_crop(read_video_frame(args.seq_path, cur_idx), args.size)
    save_anchor_point_images(ref_img, cur_img, ref_points, cur_points, out_dir, ref_idx, cur_idx)

    add_path_to_dust3r(args.model_path)
    from src.dust3r.inference import inference_recurrent_lighter  # noqa: E402
    from src.dust3r.model import ARCroco3DStereo  # noqa: E402
    from src.dust3r.utils.camera import pose_encoding_to_camera  # noqa: E402

    print(f"Loading model from {args.model_path}...")
    model = ARCroco3DStereo.from_pretrained(args.model_path).to(args.device)
    model.eval()

    img_paths, tmpdirname = parse_seq_path(args.seq_path)
    try:
        img_paths = img_paths[: args.max_frames]
        views = prepare_input(
            img_paths=img_paths,
            img_mask=[True] * len(img_paths),
            size=args.size,
            revisit=1,
            update=True,
            img_res=getattr(model, "mhmr_img_res", None),
            reset_interval=args.reset_interval,
        )
    finally:
        if tmpdirname is not None:
            shutil.rmtree(tmpdirname)

    print("Running no-anchor inference for manual correction visualization...")
    start = time.time()
    with torch.no_grad():
        outputs, _ = inference_recurrent_lighter(views, model, args.device, verbose=True)
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"Inference finished in {time.time() - start:.2f}s")

    mats = pose_mats(outputs, pose_encoding_to_camera)
    ref_pts = outputs["pred"][ref_idx]["pts3d_in_self_view"].detach().float().cpu().numpy()[0]
    cur_pts = outputs["pred"][cur_idx]["pts3d_in_self_view"].detach().float().cpu().numpy()[0]
    ref_samples = []
    cur_samples = []
    for ref_point, cur_point in zip(ref_points, cur_points):
        ref_x = int(np.clip(round(ref_point[0]), 0, ref_pts.shape[1] - 1))
        ref_y = int(np.clip(round(ref_point[1]), 0, ref_pts.shape[0] - 1))
        cur_x = int(np.clip(round(cur_point[0]), 0, cur_pts.shape[1] - 1))
        cur_y = int(np.clip(round(cur_point[1]), 0, cur_pts.shape[0] - 1))
        ref_samples.append(ref_pts[ref_y, ref_x])
        cur_samples.append(cur_pts[cur_y, cur_x])
    ref_samples = np.asarray(ref_samples, dtype=np.float32)
    cur_samples = np.asarray(cur_samples, dtype=np.float32)
    finite = np.isfinite(ref_samples).all(axis=1) & np.isfinite(cur_samples).all(axis=1)
    ref_samples = ref_samples[finite]
    cur_samples = cur_samples[finite]

    ref_world = apply_camera(ref_samples, mats[ref_idx])
    cur_world_before = apply_camera(cur_samples, mats[cur_idx])
    rotation, translation = robust_rigid(cur_samples, ref_world)
    cur_world_after = apply_rigid(cur_samples, rotation, translation)
    cam_ref = mats[ref_idx, :3, 3]
    cam_before = mats[cur_idx, :3, 3]
    cam_after = translation
    before_stats, after_stats = save_correction_image(
        ref_world, cur_world_before, cur_world_after, cam_ref, cam_before, cam_after, out_dir
    )

    summary = {
        "ref_idx": ref_idx,
        "cur_idx": cur_idx,
        "quality_gate": float(np.asarray(anchor["quality_gate"]).reshape(-1)[0]),
        "valid_anchor_count": int(mask.sum()),
        "finite_anchor_count": int(finite.sum()),
        "before_residual": before_stats,
        "after_residual": after_stats,
        "outputs": {
            "ref_points": str(out_dir / "frame_0062_ref_anchor_points.jpg"),
            "cur_points": str(out_dir / "frame_0063_cur_anchor_points.jpg"),
            "matches": str(out_dir / "frame_0062_0063_anchor_matches.jpg"),
            "correction": str(out_dir / "manual_anchor_correction_before_after_xz.jpg"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
