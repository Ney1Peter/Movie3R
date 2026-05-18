#!/usr/bin/env python3
"""Build external AnchorToken metadata for a shot boundary in a video.

The output npz can be passed to demo.py with --anchor_path. This script does
not use GT camera/depth/SMPL. It matches the boundary frames with XFeat, filters
matches with a fundamental-matrix RANSAC, maps points to Human3R crop patches,
and writes the top-K patch correspondences expected by AnchorPoseAdapter.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
XFEAT_ROOT = REPO_ROOT.parent / "xfeat-for-Movie3R"
for path in [REPO_ROOT, REPO_ROOT / "src", XFEAT_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from modules.xfeat import XFeat  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq_path", required=True, help="Input video path.")
    parser.add_argument("--out_path", required=True, help="Output anchor npz path.")
    parser.add_argument("--ref_index", type=int, required=True, help="0-based frame index before the cut.")
    parser.add_argument("--cur_index", type=int, required=True, help="0-based frame index after the cut.")
    parser.add_argument("--size", type=int, default=512, help="Human3R demo input size.")
    parser.add_argument("--top_k_xfeat", type=int, default=8192)
    parser.add_argument("--top_k_tokens", type=int, default=16)
    parser.add_argument("--fundamental_thresh", type=float, default=2.0)
    parser.add_argument("--min_unique_anchors", type=int, default=4)
    parser.add_argument("--max_dim", type=int, default=1200)
    parser.add_argument("--summary_path", default=None)
    return parser.parse_args()


def read_video_frame(seq_path, frame_index):
    cap = cv2.VideoCapture(str(seq_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {seq_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"failed to read frame {frame_index} from {seq_path}")
    return frame


def resize_for_matching(img, max_dim):
    h, w = img.shape[:2]
    if max_dim <= 0 or max(h, w) <= max_dim:
        return img, 1.0, 1.0
    scale = float(max_dim) / float(max(h, w))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, float(new_w) / float(w), float(new_h) / float(h)


def to_original_coords(points, sx, sy):
    pts = np.asarray(points, dtype=np.float32).copy()
    pts[:, 0] /= sx
    pts[:, 1] /= sy
    return pts


def human3r_crop_meta(image_shape, size):
    h0, w0 = image_shape[:2]
    scale = float(size) / float(max(w0, h0))
    w1 = int(round(w0 * scale))
    h1 = int(round(h0 * scale))
    cx, cy = w1 // 2, h1 // 2
    halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
    if w1 == h1:
        halfh = int(3 * halfw / 4)
    crop_box = (cx - halfw, cy - halfh, cx + halfw, cy + halfh)
    return {
        "original_size_wh": [int(w0), int(h0)],
        "resized_size_wh": [int(w1), int(h1)],
        "crop_box_xyxy": [int(v) for v in crop_box],
        "final_size_wh": [int(crop_box[2] - crop_box[0]), int(crop_box[3] - crop_box[1])],
        "scale_xy": [float(w1 / w0), float(h1 / h0)],
    }


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


def patch_center_norm(patch_idx, grid_hw):
    gh, gw = grid_hw
    y = int(patch_idx) // gw
    x = int(patch_idx) % gw
    return np.array([(x + 0.5) / gw, (y + 0.5) / gh], dtype=np.float32)


def compute_fundamental_inliers(points_ref, points_cur, threshold):
    if len(points_ref) < 8:
        return np.zeros((len(points_ref),), dtype=bool)
    _, mask = cv2.findFundamentalMat(
        np.asarray(points_ref, dtype=np.float32),
        np.asarray(points_cur, dtype=np.float32),
        cv2.FM_RANSAC,
        float(threshold),
        0.999,
    )
    if mask is None:
        return np.zeros((len(points_ref),), dtype=bool)
    return mask.reshape(-1).astype(bool)


def fit_affine(ref_norm, cur_norm, weights, ridge=1e-4):
    if len(ref_norm) < 3:
        return None
    x = np.concatenate([ref_norm, np.ones((len(ref_norm), 1), dtype=np.float32)], axis=1)
    w = np.sqrt(np.clip(weights, 1e-6, None)).reshape(-1, 1).astype(np.float32)
    lhs = (x * w).T @ (x * w) + ridge * np.eye(3, dtype=np.float32)
    rhs = (x * w).T @ (cur_norm * w)
    try:
        beta = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return None
    return beta.T.astype(np.float32)


def invert_affine(affine):
    if affine is None:
        return None
    m = affine[:, :2]
    t = affine[:, 2]
    try:
        inv_m = np.linalg.inv(m)
    except np.linalg.LinAlgError:
        return None
    inv_t = -inv_m @ t
    return np.concatenate([inv_m, inv_t[:, None]], axis=1).astype(np.float32)


def apply_affine(points, affine):
    x = np.concatenate([points, np.ones((len(points), 1), dtype=np.float32)], axis=1)
    return (x @ affine.T).astype(np.float32)


def patch_error(pred_cur_norm, cur_norm, grid_hw):
    gh, gw = grid_hw
    diff = (pred_cur_norm - cur_norm) * np.array([gw, gh], dtype=np.float32)[None]
    return np.linalg.norm(diff, axis=1)


def compute_local_residual(ref_pos_norm, cur_pos_norm, affine_inverse):
    if affine_inverse is None:
        return ref_pos_norm - cur_pos_norm
    cur_h = np.concatenate([cur_pos_norm, np.ones((cur_pos_norm.shape[0], 1), dtype=np.float32)], axis=1)
    base_ref = cur_h @ affine_inverse.astype(np.float32).T
    return ref_pos_norm - base_ref.astype(np.float32)


def select_spatial_diverse(anchors, top_k):
    if len(anchors) <= top_k:
        return list(range(len(anchors)))
    order = sorted(range(len(anchors)), key=lambda i: anchors[i]["confidence"], reverse=True)
    cur = np.array([a["cur_pos_norm"] for a in anchors], dtype=np.float32)
    selected = []
    for min_dist in [0.18, 0.14, 0.10, 0.06, 0.0]:
        selected = []
        for idx in order:
            if len(selected) >= top_k:
                break
            if not selected:
                selected.append(idx)
                continue
            d = np.linalg.norm(cur[idx] - cur[np.array(selected, dtype=np.int64)], axis=1)
            if float(d.min()) >= min_dist:
                selected.append(idx)
        if len(selected) >= top_k:
            break
    if len(selected) < top_k:
        for idx in order:
            if idx not in selected:
                selected.append(idx)
                if len(selected) >= top_k:
                    break
    return selected[:top_k]


def quality_gate(num_anchors, affine_median_error):
    count_gate = float(np.clip((num_anchors - 4.0) / 12.0, 0.0, 1.0))
    residual_gate = float(np.clip(1.0 - affine_median_error / 4.0, 0.0, 1.0))
    return count_gate * residual_gate


def main():
    args = parse_args()
    ref_bgr = read_video_frame(args.seq_path, args.ref_index)
    cur_bgr = read_video_frame(args.seq_path, args.cur_index)

    ref_match, sx_ref, sy_ref = resize_for_matching(ref_bgr, args.max_dim)
    cur_match, sx_cur, sy_cur = resize_for_matching(cur_bgr, args.max_dim)
    xfeat = XFeat(top_k=args.top_k_xfeat)
    mkpts_ref, mkpts_cur = xfeat.match_xfeat_star(ref_match, cur_match, top_k=args.top_k_xfeat)
    mkpts_ref = np.asarray(mkpts_ref, dtype=np.float32)
    mkpts_cur = np.asarray(mkpts_cur, dtype=np.float32)
    mkpts_ref_orig = to_original_coords(mkpts_ref, sx_ref, sy_ref)
    mkpts_cur_orig = to_original_coords(mkpts_cur, sx_cur, sy_cur)
    fundamental_mask = compute_fundamental_inliers(mkpts_ref, mkpts_cur, args.fundamental_thresh)

    ref_meta = human3r_crop_meta(ref_bgr.shape, args.size)
    cur_meta = human3r_crop_meta(cur_bgr.shape, args.size)
    patch_size = 16
    ref_h, ref_w = ref_meta["final_size_wh"][1], ref_meta["final_size_wh"][0]
    cur_h, cur_w = cur_meta["final_size_wh"][1], cur_meta["final_size_wh"][0]
    ref_grid_hw = (ref_h // patch_size, ref_w // patch_size)
    cur_grid_hw = (cur_h // patch_size, cur_w // patch_size)

    ref_crop_xy, ref_crop_valid = raw_to_crop_xy(mkpts_ref_orig, ref_meta)
    cur_crop_xy, cur_crop_valid = raw_to_crop_xy(mkpts_cur_orig, cur_meta)
    ref_patch_xy, ref_patch_idx, ref_patch_valid = crop_xy_to_patch(ref_crop_xy, ref_crop_valid, patch_size, ref_grid_hw)
    cur_patch_xy, cur_patch_idx, cur_patch_valid = crop_xy_to_patch(cur_crop_xy, cur_crop_valid, patch_size, cur_grid_hw)
    valid = fundamental_mask & ref_patch_valid & cur_patch_valid

    best_by_pair = {}
    for idx in np.flatnonzero(valid):
        pair = (int(ref_patch_idx[idx]), int(cur_patch_idx[idx]))
        if pair in best_by_pair:
            continue
        ref_pos = patch_center_norm(pair[0], ref_grid_hw)
        cur_pos = patch_center_norm(pair[1], cur_grid_hw)
        best_by_pair[pair] = {
            "ref_patch_idx": pair[0],
            "cur_patch_idx": pair[1],
            "ref_patch_xy": ref_patch_xy[idx].astype(np.int32).tolist(),
            "cur_patch_xy": cur_patch_xy[idx].astype(np.int32).tolist(),
            "ref_pos_norm": ref_pos,
            "cur_pos_norm": cur_pos,
            "confidence": 1.0,
        }
    anchors = list(best_by_pair.values())
    if len(anchors) < args.min_unique_anchors:
        raise RuntimeError(
            f"too few anchors after filtering: {len(anchors)} < {args.min_unique_anchors}; raw_matches={len(mkpts_ref)}"
        )

    ref_norm = np.array([a["ref_pos_norm"] for a in anchors], dtype=np.float32)
    cur_norm = np.array([a["cur_pos_norm"] for a in anchors], dtype=np.float32)
    weights = np.array([a["confidence"] for a in anchors], dtype=np.float32)
    affine = fit_affine(ref_norm, cur_norm, weights)
    inv_affine = invert_affine(affine)
    if affine is None or inv_affine is None:
        affine = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        inv_affine = affine.copy()
    affine_err = patch_error(apply_affine(ref_norm, affine), cur_norm, cur_grid_hw)
    median_affine_err = float(np.median(affine_err))
    q_gate = quality_gate(len(anchors), median_affine_err)

    selected = select_spatial_diverse(anchors, args.top_k_tokens)
    top = [anchors[i] for i in selected]
    top_ref_norm = np.array([a["ref_pos_norm"] for a in top], dtype=np.float32)
    top_cur_norm = np.array([a["cur_pos_norm"] for a in top], dtype=np.float32)
    local_residual = compute_local_residual(top_ref_norm, top_cur_norm, inv_affine)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        ref_view_idx=np.array([args.ref_index], dtype=np.int64),
        cur_view_idx=np.array([args.cur_index], dtype=np.int64),
        top_k_tokens=np.array([args.top_k_tokens], dtype=np.int64),
        ref_patch_idx=np.array([a["ref_patch_idx"] for a in top], dtype=np.int64),
        cur_patch_idx=np.array([a["cur_patch_idx"] for a in top], dtype=np.int64),
        ref_patch_xy=np.array([a["ref_patch_xy"] for a in top], dtype=np.int32),
        cur_patch_xy=np.array([a["cur_patch_xy"] for a in top], dtype=np.int32),
        ref_pos_norm=top_ref_norm.astype(np.float32),
        cur_pos_norm=top_cur_norm.astype(np.float32),
        local_residual_norm=local_residual.astype(np.float32),
        confidence=np.array([a["confidence"] for a in top], dtype=np.float32),
        anchor_mask=np.ones((len(top),), dtype=np.bool_),
        quality_gate=np.array([q_gate], dtype=np.float32),
        affine_forward=affine.astype(np.float32),
        affine_inverse=inv_affine.astype(np.float32),
        ref_grid_hw=np.array(ref_grid_hw, dtype=np.int32),
        cur_grid_hw=np.array(cur_grid_hw, dtype=np.int32),
    )

    summary = {
        "seq_path": str(args.seq_path),
        "out_path": str(out_path),
        "ref_index": int(args.ref_index),
        "cur_index": int(args.cur_index),
        "raw_matches": int(len(mkpts_ref)),
        "fundamental_inliers": int(fundamental_mask.sum()),
        "unique_anchor_patch_pairs": int(len(anchors)),
        "top_k_tokens": int(len(top)),
        "quality_gate": float(q_gate),
        "affine_median_patch_error": median_affine_err,
        "ref_grid_hw": list(map(int, ref_grid_hw)),
        "cur_grid_hw": list(map(int, cur_grid_hw)),
    }
    summary_path = Path(args.summary_path) if args.summary_path else out_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
