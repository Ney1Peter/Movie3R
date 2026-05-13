#!/usr/bin/env python3
"""Verify whether mesh-verified XFeat anchors are visible in Human3R encoder tokens.

Step 1 experiment for ShotToken V6:
1. Build one RICH AABB shot pair: A@(t+1) -> B@(t+2).
2. Run XFeat semi-dense matching on RGB images.
3. Keep only matches verified by the official RICH static scan mesh.
4. Map verified 2D anchor points into Human3R/CUT3R encoder patch indices.
5. Compare encoder-token cosine similarity for anchor pairs vs random negatives.

The script does not modify the encoder or decoder and does not use Movie3R
ShotToken modules. It only tests whether existing Human3R encoder tokens already
carry useful anchor correspondence information.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from PIL.ImageOps import exif_transpose
import torchvision.transforms as tvf


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
ACCEL_ROOT = Path("/workspace/code/accelerated_features")
ACCEL_SCRIPTS = ACCEL_ROOT / "scripts"

for path in [REPO_ROOT, SRC_ROOT, ACCEL_ROOT, ACCEL_SCRIPTS]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dust3r.model_human3r import load_model  # noqa: E402
from modules.xfeat import XFeat  # noqa: E402
from test_rich_aabb_xfeat_geometry import (  # noqa: E402
    compute_ransac_inliers,
    load_mask,
    resize_for_matching,
    to_original_coords,
)
from test_rich_aabb_xfeat_mesh_geometry import (  # noqa: E402
    build_visible_vertex_map,
    compute_fundamental_inliers,
    compute_visible_overlap,
    evaluate_mesh_geometry,
)
from visualize_rich_mesh_projection import load_ply_vertices  # noqa: E402


IMG_NORM = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rich_root", default="/workspace/data/RICH")
    parser.add_argument("--data_root", default="/workspace/data/RICH/RICH_4Human3R/Training")
    parser.add_argument("--source_sequence", default="BBQ_001_guitar")
    parser.add_argument("--cam_a", type=int, default=6)
    parser.add_argument("--cam_b", type=int, default=7)
    parser.add_argument("--start_frame", type=int, default=244)
    parser.add_argument("--model_path", default=str(REPO_ROOT / "src" / "human3r_896L.pth"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--top_k", type=int, default=8192)
    parser.add_argument("--max_dim", type=int, default=1200)
    parser.add_argument("--mesh_max_dim", type=int, default=1400)
    parser.add_argument("--mesh_lookup_radius", type=int, default=4)
    parser.add_argument("--mesh_z_tol", type=float, default=0.03)
    parser.add_argument("--reproj_thresh", type=float, default=24.0)
    parser.add_argument("--ransac_thresh", type=float, default=4.0)
    parser.add_argument("--fundamental_thresh", type=float, default=2.0)
    parser.add_argument("--max_draw", type=int, default=120)
    parser.add_argument("--num_similarity_examples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", default=None)
    return parser.parse_args()


def seq_name(source_sequence, cam):
    return f"{source_sequence}_cam_{cam:02d}"


def load_rgb(data_root, seq, frame):
    path = Path(data_root) / seq / "rgb" / f"{frame:08d}.png"
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"failed to read image: {path}")
    return img, path


def load_human3r_image(image_path, size):
    img = exif_transpose(Image.open(image_path)).convert("RGB")
    w0, h0 = img.size
    scale = float(size) / float(max(w0, h0))
    new_size = (int(round(w0 * scale)), int(round(h0 * scale)))
    interp = Image.Resampling.LANCZOS if max(w0, h0) > size else Image.Resampling.BICUBIC
    img = img.resize(new_size, interp)

    w1, h1 = img.size
    cx, cy = w1 // 2, h1 // 2
    halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
    if w1 == h1:
        halfh = int(3 * halfw / 4)
    crop_box = (cx - halfw, cy - halfh, cx + halfw, cy + halfh)
    img = img.crop(crop_box)
    w2, h2 = img.size
    tensor = IMG_NORM(img)[None]
    meta = {
        "original_size_wh": [int(w0), int(h0)],
        "resized_size_wh": [int(w1), int(h1)],
        "crop_box_xyxy": [int(v) for v in crop_box],
        "final_size_wh": [int(w2), int(h2)],
        "scale_xy": [float(w1 / w0), float(h1 / h0)],
    }
    return tensor, torch.from_numpy(np.int32([[h2, w2]])), np.asarray(img), meta


def raw_to_crop_xy(points_xy, meta):
    points = np.asarray(points_xy, dtype=np.float32).copy()
    sx, sy = meta["scale_xy"]
    left, top, _, _ = meta["crop_box_xyxy"]
    points[:, 0] = points[:, 0] * sx - left
    points[:, 1] = points[:, 1] * sy - top
    w, h = meta["final_size_wh"]
    valid = (
        np.isfinite(points).all(axis=1)
        & (points[:, 0] >= 0)
        & (points[:, 0] < w)
        & (points[:, 1] >= 0)
        & (points[:, 1] < h)
    )
    return points, valid


def crop_xy_to_patch(points_xy, valid, patch_size, grid_hw):
    points = np.asarray(points_xy, dtype=np.float32)
    patch_xy = np.floor(points / float(patch_size)).astype(np.int32)
    gh, gw = grid_hw
    valid = valid.copy()
    valid &= patch_xy[:, 0] >= 0
    valid &= patch_xy[:, 0] < gw
    valid &= patch_xy[:, 1] >= 0
    valid &= patch_xy[:, 1] < gh
    patch_idx = patch_xy[:, 1] * gw + patch_xy[:, 0]
    return patch_xy, patch_idx, valid


def normalize01(x):
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.float32)
    lo = np.percentile(x[finite], 1)
    hi = np.percentile(x[finite], 99)
    if hi <= lo + 1e-8:
        lo = float(x[finite].min())
        hi = float(x[finite].max())
    if hi <= lo + 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def heatmap_image(values, out_hw):
    values = normalize01(values)
    hm = cv2.applyColorMap((values * 255).round().astype(np.uint8), cv2.COLORMAP_TURBO)
    hm = cv2.resize(hm, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)


def save_rgb(path, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def draw_patch_grid(rgb, patch_size):
    out = rgb.copy()
    h, w = out.shape[:2]
    for x in range(0, w + 1, patch_size):
        cv2.line(out, (x, 0), (x, h), (255, 255, 0), 1, cv2.LINE_AA)
    for y in range(0, h + 1, patch_size):
        cv2.line(out, (0, y), (w, y), (255, 255, 0), 1, cv2.LINE_AA)
    return out


def patch_center(patch_idx, patch_size, grid_hw):
    gh, gw = grid_hw
    y = int(patch_idx) // gw
    x = int(patch_idx) % gw
    return np.array([(x + 0.5) * patch_size, (y + 0.5) * patch_size], dtype=np.float32)


# **========== 原始代码 ==========
# def draw_side_by_side_matches(ref_rgb, cur_rgb, anchors, patch_size, grid_hw, out_path, max_draw):
#     # Assumed ref/cur Human3R crops share one patch grid.
# **========== 新代码 ==========
def draw_side_by_side_matches(ref_rgb, cur_rgb, anchors, patch_size, ref_grid_hw, cur_grid_hw, out_path, max_draw):
    # **========== 结束 ==========
    ref = draw_patch_grid(ref_rgb, patch_size)
    cur = draw_patch_grid(cur_rgb, patch_size)
    h = max(ref.shape[0], cur.shape[0])
    w = ref.shape[1] + cur.shape[1]
    canvas = np.zeros((h + 64, w, 3), dtype=np.uint8)
    canvas[:64] = 20
    canvas[64 : 64 + ref.shape[0], : ref.shape[1]] = ref
    canvas[64 : 64 + cur.shape[0], ref.shape[1] : ref.shape[1] + cur.shape[1]] = cur
    cv2.putText(canvas, f"mesh-verified anchors mapped to Human3R patches: {len(anchors)}", (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    rng = np.random.default_rng(0)
    draw_ids = np.arange(len(anchors))
    if len(draw_ids) > max_draw:
        draw_ids = rng.choice(draw_ids, size=max_draw, replace=False)
    for n, idx in enumerate(draw_ids):
        a = anchors[int(idx)]
        color = tuple(int(c) for c in cv2.applyColorMap(np.array([[int(255 * (n % 32) / 31)]], dtype=np.uint8), cv2.COLORMAP_HSV)[0, 0])
        color = (color[2], color[1], color[0])
        p0 = patch_center(a["ref_patch_idx"], patch_size, ref_grid_hw)
        p1 = patch_center(a["cur_patch_idx"], patch_size, cur_grid_hw)
        p0 = (int(round(p0[0])), int(round(64 + p0[1])))
        p1 = (int(round(ref.shape[1] + p1[0])), int(round(64 + p1[1])))
        cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, p0, 4, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 4, color, -1, cv2.LINE_AA)
    save_rgb(out_path, canvas)


def draw_raw_mesh_matches(ref_bgr, cur_bgr, mkpts_ref, mkpts_cur, mesh_indices, out_path, max_draw):
    scale = min(1.0, 900.0 / max(ref_bgr.shape[:2] + cur_bgr.shape[:2]))
    ref = cv2.resize(ref_bgr, (int(ref_bgr.shape[1] * scale), int(ref_bgr.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    cur = cv2.resize(cur_bgr, (int(cur_bgr.shape[1] * scale), int(cur_bgr.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    h = max(ref.shape[0], cur.shape[0])
    canvas = np.zeros((h + 56, ref.shape[1] + cur.shape[1], 3), dtype=np.uint8)
    canvas[:56] = 20
    canvas[56 : 56 + ref.shape[0], : ref.shape[1]] = ref
    canvas[56 : 56 + cur.shape[0], ref.shape[1] : ref.shape[1] + cur.shape[1]] = cur
    cv2.putText(canvas, f"semi-dense XFeat + RICH mesh inliers: {len(mesh_indices)}", (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    ids = np.asarray(mesh_indices, dtype=np.int64)[:max_draw]
    for n, idx in enumerate(ids):
        color = tuple(int(c) for c in cv2.applyColorMap(np.array([[int(255 * (n % 32) / 31)]], dtype=np.uint8), cv2.COLORMAP_HSV)[0, 0])
        p0 = (int(round(mkpts_ref[idx, 0] * scale)), int(round(56 + mkpts_ref[idx, 1] * scale)))
        p1 = (int(round(ref.shape[1] + mkpts_cur[idx, 0] * scale)), int(round(56 + mkpts_cur[idx, 1] * scale)))
        cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, p0, 4, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 4, color, -1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)


def save_histogram(path, series, labels, title, bins=40):
    width, height = 900, 420
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    cv2.putText(canvas, title, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (20, 20, 20), 2, cv2.LINE_AA)
    colors = [(40, 120, 240), (220, 80, 80), (80, 170, 80)]
    left, right, top, bottom = 64, 28, 64, 54
    plot_w = width - left - right
    plot_h = height - top - bottom
    cv2.rectangle(canvas, (left, top), (left + plot_w, top + plot_h), (220, 220, 220), 1)
    all_vals = np.concatenate([np.asarray(v, dtype=np.float32) for v in series if len(v) > 0]) if any(len(v) > 0 for v in series) else np.array([0.0, 1.0])
    lo = min(-1.0, float(all_vals.min()))
    hi = max(1.0, float(all_vals.max()))
    max_count = 1
    hists = []
    edges = None
    for vals in series:
        hist, edges = np.histogram(vals, bins=bins, range=(lo, hi))
        hists.append(hist)
        max_count = max(max_count, int(hist.max()))
    for si, hist in enumerate(hists):
        pts = []
        for bi, count in enumerate(hist):
            x = left + int(round((bi + 0.5) / bins * plot_w))
            y = top + plot_h - int(round(count / max_count * plot_h))
            pts.append((x, y))
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(canvas, a, b, colors[si % len(colors)], 2, cv2.LINE_AA)
        cv2.putText(canvas, labels[si], (left + 12, top + 24 + si * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colors[si % len(colors)], 2, cv2.LINE_AA)
    cv2.putText(canvas, f"cosine range [{lo:.2f}, {hi:.2f}]", (left, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def save_rank_chart(path, ranks, title):
    ranks = np.asarray(ranks, dtype=np.float32)
    width, height = 900, 360
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    cv2.putText(canvas, title, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (20, 20, 20), 2, cv2.LINE_AA)
    if len(ranks) == 0:
        cv2.imwrite(str(path), canvas)
        return
    sorted_ranks = np.sort(ranks)
    left, right, top, bottom = 64, 28, 70, 54
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_rank = max(float(sorted_ranks.max()), 1.0)
    prev = None
    for i, rank in enumerate(sorted_ranks):
        x = left + int(round(i / max(len(sorted_ranks) - 1, 1) * plot_w))
        y = top + int(round(math.log1p(rank) / math.log1p(max_rank) * plot_h))
        if prev is not None:
            cv2.line(canvas, prev, (x, y), (80, 120, 240), 2, cv2.LINE_AA)
        prev = (x, y)
    cv2.rectangle(canvas, (left, top), (left + plot_w, top + plot_h), (220, 220, 220), 1)
    cv2.putText(canvas, f"median rank={np.median(ranks):.1f}, top10={(ranks <= 10).mean():.3f}", (left, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


# **========== 原始代码 ==========
# def save_similarity_examples(out_dir, anchors, ref_rgb, cur_rgb, sim_matrix, patch_size, grid_hw, max_examples):
#     # Assumed the similarity row can be reshaped with the same grid as ref.
# **========== 新代码 ==========
def save_similarity_examples(out_dir, anchors, ref_rgb, cur_rgb, sim_matrix, patch_size, ref_grid_hw, cur_grid_hw, max_examples):
    # **========== 结束 ==========
    if not anchors:
        return []
    sims = np.array([a["encoder_cosine"] for a in anchors], dtype=np.float32)
    order = np.argsort(-sims)
    chosen = []
    if len(order) <= max_examples:
        chosen = order.tolist()
    else:
        half = max_examples // 2
        chosen = order[:half].tolist() + order[-(max_examples - half) :].tolist()
    paths = []
    for n, idx in enumerate(chosen):
        a = anchors[int(idx)]
        ref_center = patch_center(a["ref_patch_idx"], patch_size, ref_grid_hw)
        cur_center = patch_center(a["cur_patch_idx"], patch_size, cur_grid_hw)
        sim_map = sim_matrix[a["ref_patch_idx"]].reshape(cur_grid_hw)
        hm = heatmap_image(sim_map, cur_rgb.shape[:2])
        overlay = np.clip(0.45 * cur_rgb.astype(np.float32) + 0.55 * hm.astype(np.float32), 0, 255).astype(np.uint8)
        ref_marked = ref_rgb.copy()
        cur_marked = overlay.copy()
        cv2.circle(ref_marked, (int(round(ref_center[0])), int(round(ref_center[1]))), 7, (255, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(cur_marked, (int(round(cur_center[0])), int(round(cur_center[1]))), 7, (255, 0, 255), -1, cv2.LINE_AA)
        h = max(ref_marked.shape[0], cur_marked.shape[0])
        canvas = np.zeros((h + 64, ref_marked.shape[1] + cur_marked.shape[1], 3), dtype=np.uint8)
        canvas[:64] = 20
        canvas[64 : 64 + ref_marked.shape[0], : ref_marked.shape[1]] = ref_marked
        canvas[64 : 64 + cur_marked.shape[0], ref_marked.shape[1] :] = cur_marked
        title = f"anchor #{idx}: cosine={a['encoder_cosine']:.3f}, rank={a['encoder_rank']}, mesh_err={a['mesh_error_px']:.2f}px"
        cv2.putText(canvas, title, (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        path = out_dir / f"60_similarity_map_anchor_{n:02d}.jpg"
        save_rgb(path, canvas)
        paths.append(str(path))
    return paths


def compute_stats(values):
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return None
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    seq_a = seq_name(args.source_sequence, args.cam_a)
    seq_b = seq_name(args.source_sequence, args.cam_b)
    ref_frame = args.start_frame + 1
    cur_frame = args.start_frame + 2
    ref_bgr, ref_path = load_rgb(args.data_root, seq_a, ref_frame)
    cur_bgr, cur_path = load_rgb(args.data_root, seq_b, cur_frame)

    if args.out_dir is None:
        args.out_dir = str(
            REPO_ROOT
            / "output"
            / "rich_anchor_encoder_step1"
            / f"{args.source_sequence}_cam{args.cam_a:02d}_cam{args.cam_b:02d}_f{args.start_frame:08d}"
        )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Running semi-dense XFeat + RICH mesh verification...")
    ref_match, sx_ref, sy_ref = resize_for_matching(ref_bgr, args.max_dim)
    cur_match, sx_cur, sy_cur = resize_for_matching(cur_bgr, args.max_dim)
    xfeat = XFeat(top_k=args.top_k)
    mkpts_ref, mkpts_cur = xfeat.match_xfeat_star(ref_match, cur_match, top_k=args.top_k)
    mkpts_ref = np.asarray(mkpts_ref, dtype=np.float32)
    mkpts_cur = np.asarray(mkpts_cur, dtype=np.float32)
    mkpts_ref_orig = to_original_coords(mkpts_ref, sx_ref, sy_ref)
    mkpts_cur_orig = to_original_coords(mkpts_cur, sx_cur, sy_cur)
    del xfeat
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mesh_path = Path(args.rich_root) / "scan_calibration" / "BBQ" / "scan_camcoord.ply"
    xyz, _ = load_ply_vertices(mesh_path)
    ref_map = build_visible_vertex_map(xyz, args.rich_root, seq_a, ref_bgr.shape, args.mesh_max_dim, args.mesh_z_tol)
    cur_map = build_visible_vertex_map(xyz, args.rich_root, seq_b, cur_bgr.shape, args.mesh_max_dim, args.mesh_z_tol)
    mask_ref = load_mask(args.data_root, seq_a, ref_frame, ref_bgr.shape)
    mask_cur = load_mask(args.data_root, seq_b, cur_frame, cur_bgr.shape)
    eval_items = evaluate_mesh_geometry(mkpts_ref_orig, mkpts_cur_orig, ref_map, cur_map, mask_ref, mask_cur, args)
    mesh_mask = np.array([item["mesh_inlier"] for item in eval_items], dtype=bool)
    human_mask = np.array([item["on_human"] for item in eval_items], dtype=bool)
    mesh_indices = np.flatnonzero(mesh_mask)
    ransac_mask = compute_ransac_inliers(mkpts_ref, mkpts_cur, args.ransac_thresh)
    fundamental_mask = compute_fundamental_inliers(mkpts_ref, mkpts_cur, args.fundamental_thresh)
    visible_overlap = compute_visible_overlap(ref_map, cur_map, mask_ref, mask_cur)
    draw_raw_mesh_matches(ref_bgr, cur_bgr, mkpts_ref_orig, mkpts_cur_orig, mesh_indices, out_dir / "00_semidense_mesh_inliers_raw_space.jpg", args.max_draw)

    print("Loading original Human3R encoder...")
    model = load_model(args.model_path, device=device, verbose=True).eval()
    model.gradient_checkpointing = False
    patch_size = int(model.croco_args["patch_size"])
    ref_img_tensor, ref_true_shape, ref_crop_rgb, ref_meta = load_human3r_image(ref_path, args.size)
    cur_img_tensor, cur_true_shape, cur_crop_rgb, cur_meta = load_human3r_image(cur_path, args.size)
    ref_img_tensor = ref_img_tensor.to(device)
    cur_img_tensor = cur_img_tensor.to(device)
    ref_true_shape = ref_true_shape.to(device)
    cur_true_shape = cur_true_shape.to(device)
    with torch.no_grad():
        ref_feat = model._encode_image(ref_img_tensor, ref_true_shape)[0][-1]
        cur_feat = model._encode_image(cur_img_tensor, cur_true_shape)[0][-1]
        ref_dec = model.decoder_embed(ref_feat)
        cur_dec = model.decoder_embed(cur_feat)

    h_ref, w_ref = map(int, ref_true_shape[0].detach().cpu().numpy().tolist())
    h_cur, w_cur = map(int, cur_true_shape[0].detach().cpu().numpy().tolist())
    # **========== 原始代码 ==========
    # if (h_ref, w_ref) != (h_cur, w_cur):
    #     raise RuntimeError(f"Human3R crops must match for direct token comparison: ref={(h_ref,w_ref)} cur={(h_cur,w_cur)}")
    # grid_hw = (h_ref // patch_size, w_ref // patch_size)
    # if ref_feat.shape[1] != grid_hw[0] * grid_hw[1]:
    #     raise RuntimeError(f"token/grid mismatch: {ref_feat.shape[1]} vs {grid_hw}")
    # **========== 新代码 ==========
    ref_grid_hw = (h_ref // patch_size, w_ref // patch_size)
    cur_grid_hw = (h_cur // patch_size, w_cur // patch_size)
    if ref_feat.shape[1] != ref_grid_hw[0] * ref_grid_hw[1]:
        raise RuntimeError(f"ref token/grid mismatch: {ref_feat.shape[1]} vs {ref_grid_hw}")
    if cur_feat.shape[1] != cur_grid_hw[0] * cur_grid_hw[1]:
        raise RuntimeError(f"cur token/grid mismatch: {cur_feat.shape[1]} vs {cur_grid_hw}")
    # **========== 结束 ==========

    ref_crop_xy, ref_crop_valid = raw_to_crop_xy(mkpts_ref_orig, ref_meta)
    cur_crop_xy, cur_crop_valid = raw_to_crop_xy(mkpts_cur_orig, cur_meta)
    # **========== 原始代码 ==========
    # ref_patch_xy, ref_patch_idx, ref_patch_valid = crop_xy_to_patch(ref_crop_xy, ref_crop_valid, patch_size, grid_hw)
    # cur_patch_xy, cur_patch_idx, cur_patch_valid = crop_xy_to_patch(cur_crop_xy, cur_crop_valid, patch_size, grid_hw)
    # **========== 新代码 ==========
    ref_patch_xy, ref_patch_idx, ref_patch_valid = crop_xy_to_patch(ref_crop_xy, ref_crop_valid, patch_size, ref_grid_hw)
    cur_patch_xy, cur_patch_idx, cur_patch_valid = crop_xy_to_patch(cur_crop_xy, cur_crop_valid, patch_size, cur_grid_hw)
    # **========== 结束 ==========
    mapped_valid = mesh_mask & ref_patch_valid & cur_patch_valid

    best_by_pair = {}
    for idx in np.flatnonzero(mapped_valid):
        pair = (int(ref_patch_idx[idx]), int(cur_patch_idx[idx]))
        err = eval_items[int(idx)]["best_mesh_reproj_error_px"]
        err_val = float(err) if err is not None else float("inf")
        if pair not in best_by_pair or err_val < best_by_pair[pair]["mesh_error_px"]:
            best_by_pair[pair] = {
                "match_index": int(idx),
                "ref_patch_idx": pair[0],
                "cur_patch_idx": pair[1],
                "ref_patch_xy": ref_patch_xy[idx].astype(int).tolist(),
                "cur_patch_xy": cur_patch_xy[idx].astype(int).tolist(),
                "ref_xy_crop": ref_crop_xy[idx].astype(float).tolist(),
                "cur_xy_crop": cur_crop_xy[idx].astype(float).tolist(),
                "ref_xy_original": mkpts_ref_orig[idx].astype(float).tolist(),
                "cur_xy_original": mkpts_cur_orig[idx].astype(float).tolist(),
                "mesh_error_px": err_val,
            }
    anchors = list(best_by_pair.values())

    ref_norm = F.normalize(ref_feat[0].float(), dim=-1)
    cur_norm = F.normalize(cur_feat[0].float(), dim=-1)
    ref_dec_norm = F.normalize(ref_dec[0].float(), dim=-1)
    cur_dec_norm = F.normalize(cur_dec[0].float(), dim=-1)
    sim_matrix = (ref_norm @ cur_norm.T).detach().cpu().numpy()
    dec_sim_matrix = (ref_dec_norm @ cur_dec_norm.T).detach().cpu().numpy()

    positive = []
    positive_dec = []
    ranks = []
    ranks_dec = []
    for a in anchors:
        ri = a["ref_patch_idx"]
        ci = a["cur_patch_idx"]
        sim = float(sim_matrix[ri, ci])
        sim_dec = float(dec_sim_matrix[ri, ci])
        rank = int((sim_matrix[ri] > sim).sum() + 1)
        rank_dec = int((dec_sim_matrix[ri] > sim_dec).sum() + 1)
        a["encoder_cosine"] = sim
        a["decoder_embed_cosine"] = sim_dec
        a["encoder_rank"] = rank
        a["decoder_embed_rank"] = rank_dec
        positive.append(sim)
        positive_dec.append(sim_dec)
        ranks.append(rank)
        ranks_dec.append(rank_dec)

    # **========== 原始代码 ==========
    # n_tokens = ref_feat.shape[1]
    # **========== 新代码 ==========
    n_cur_tokens = cur_feat.shape[1]
    # **========== 结束 ==========
    negative = []
    negative_dec = []
    shuffled = []
    shuffled_dec = []
    if anchors:
        cur_anchor_indices = np.array([a["cur_patch_idx"] for a in anchors], dtype=np.int64)
        shifted_cur_anchor_indices = np.roll(cur_anchor_indices, 1)
        for a, shifted_ci in zip(anchors, shifted_cur_anchor_indices):
            ri = a["ref_patch_idx"]
            ci = a["cur_patch_idx"]
            rand_ci = int(rng.integers(0, n_cur_tokens - 1))
            if rand_ci >= ci:
                rand_ci += 1
            negative.append(float(sim_matrix[ri, rand_ci]))
            negative_dec.append(float(dec_sim_matrix[ri, rand_ci]))
            shuffled.append(float(sim_matrix[ri, int(shifted_ci)]))
            shuffled_dec.append(float(dec_sim_matrix[ri, int(shifted_ci)]))

    draw_side_by_side_matches(ref_crop_rgb, cur_crop_rgb, anchors, patch_size, ref_grid_hw, cur_grid_hw, out_dir / "01_anchor_patches_on_human3r_crop.jpg", args.max_draw)
    save_rgb(out_dir / "02_ref_human3r_crop_grid.jpg", draw_patch_grid(ref_crop_rgb, patch_size))
    save_rgb(out_dir / "03_cur_human3r_crop_grid.jpg", draw_patch_grid(cur_crop_rgb, patch_size))
    save_histogram(out_dir / "10_encoder_cosine_pos_vs_neg.jpg", [positive, negative, shuffled], ["mesh anchor positives", "random negatives", "shuffled anchor negatives"], "Human3R encoder token cosine")
    save_histogram(out_dir / "11_decoder_embed_cosine_pos_vs_neg.jpg", [positive_dec, negative_dec, shuffled_dec], ["mesh anchor positives", "random negatives", "shuffled anchor negatives"], "Human3R decoder_embed token cosine")
    save_rank_chart(out_dir / "12_encoder_true_match_rank.jpg", ranks, "rank of true mesh anchor patch among all current patches")
    save_rank_chart(out_dir / "13_decoder_embed_true_match_rank.jpg", ranks_dec, "decoder_embed rank of true mesh anchor patch")
    example_paths = save_similarity_examples(out_dir, anchors, ref_crop_rgb, cur_crop_rgb, sim_matrix, patch_size, ref_grid_hw, cur_grid_hw, args.num_similarity_examples)

    pos_arr = np.asarray(positive, dtype=np.float32)
    neg_arr = np.asarray(negative, dtype=np.float32)
    pairwise_accuracy = None
    if len(pos_arr) > 0 and len(neg_arr) > 0:
        pairwise_accuracy = float((pos_arr > neg_arr).mean())

    summary = {
        "args": vars(args),
        "ref": {"seq": seq_a, "frame": int(ref_frame), "image": str(ref_path)},
        "cur": {"seq": seq_b, "frame": int(cur_frame), "image": str(cur_path)},
        "match_mode": "semidense",
        "raw_matches": int(len(mkpts_ref)),
        "homography_ransac_inliers": int(ransac_mask.sum()),
        "fundamental_ransac_inliers": int(fundamental_mask.sum()),
        "mesh_geometry_inliers": int(mesh_mask.sum()),
        "mesh_inliers_inside_fundamental": int((mesh_mask & fundamental_mask).sum()),
        "matches_on_human": int(human_mask.sum()),
        "mesh_visible_overlap": visible_overlap,
        "human3r_ref_crop_meta": ref_meta,
        "human3r_cur_crop_meta": cur_meta,
        "human3r_patch_size": int(patch_size),
        # **========== 原始代码 ==========
        # "human3r_grid_hw": list(grid_hw),
        # **========== 新代码 ==========
        "human3r_ref_grid_hw": list(ref_grid_hw),
        "human3r_cur_grid_hw": list(cur_grid_hw),
        # **========== 结束 ==========
        "mesh_inliers_after_human3r_crop": int(mapped_valid.sum()),
        "unique_anchor_patch_pairs": int(len(anchors)),
        "encoder_positive_cosine": compute_stats(positive),
        "encoder_random_negative_cosine": compute_stats(negative),
        "encoder_shuffled_negative_cosine": compute_stats(shuffled),
        "decoder_embed_positive_cosine": compute_stats(positive_dec),
        "decoder_embed_random_negative_cosine": compute_stats(negative_dec),
        "decoder_embed_shuffled_negative_cosine": compute_stats(shuffled_dec),
        "encoder_true_match_rank": {
            "count": int(len(ranks)),
            "median": float(np.median(ranks)) if ranks else None,
            "mean": float(np.mean(ranks)) if ranks else None,
            "top1": float((np.asarray(ranks) <= 1).mean()) if ranks else None,
            "top5": float((np.asarray(ranks) <= 5).mean()) if ranks else None,
            "top10": float((np.asarray(ranks) <= 10).mean()) if ranks else None,
            "top32": float((np.asarray(ranks) <= 32).mean()) if ranks else None,
        },
        "decoder_embed_true_match_rank": {
            "count": int(len(ranks_dec)),
            "median": float(np.median(ranks_dec)) if ranks_dec else None,
            "mean": float(np.mean(ranks_dec)) if ranks_dec else None,
            "top1": float((np.asarray(ranks_dec) <= 1).mean()) if ranks_dec else None,
            "top5": float((np.asarray(ranks_dec) <= 5).mean()) if ranks_dec else None,
            "top10": float((np.asarray(ranks_dec) <= 10).mean()) if ranks_dec else None,
            "top32": float((np.asarray(ranks_dec) <= 32).mean()) if ranks_dec else None,
        },
        "positive_greater_than_random_negative": pairwise_accuracy,
        "anchors": anchors,
        "visualizations": {
            "raw_mesh_matches": str(out_dir / "00_semidense_mesh_inliers_raw_space.jpg"),
            "anchor_patch_matches": str(out_dir / "01_anchor_patches_on_human3r_crop.jpg"),
            "encoder_histogram": str(out_dir / "10_encoder_cosine_pos_vs_neg.jpg"),
            "decoder_embed_histogram": str(out_dir / "11_decoder_embed_cosine_pos_vs_neg.jpg"),
            "encoder_rank_chart": str(out_dir / "12_encoder_true_match_rank.jpg"),
            "decoder_embed_rank_chart": str(out_dir / "13_decoder_embed_true_match_rank.jpg"),
            "similarity_examples": example_paths,
        },
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "out_dir": str(out_dir),
        "raw_matches": summary["raw_matches"],
        "mesh_geometry_inliers": summary["mesh_geometry_inliers"],
        "unique_anchor_patch_pairs": summary["unique_anchor_patch_pairs"],
        "encoder_positive_cosine": summary["encoder_positive_cosine"],
        "encoder_random_negative_cosine": summary["encoder_random_negative_cosine"],
        "encoder_true_match_rank": summary["encoder_true_match_rank"],
        "positive_greater_than_random_negative": pairwise_accuracy,
    }, indent=2))


if __name__ == "__main__":
    main()
