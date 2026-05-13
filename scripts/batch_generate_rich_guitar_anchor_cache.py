#!/usr/bin/env python3
"""Batch-generate a small RICH guitar AnchorToken cache.

This is an offline training/validation cache generator. It intentionally stores
only top-K anchor metadata and affine evidence, not full-frame encoder tokens.
During model training, encoder tokens can be gathered online from cached patch
indices.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
SRC_ROOT = REPO_ROOT / "src"
ACCEL_ROOT = Path("/workspace/code/accelerated_features")
ACCEL_SCRIPTS = ACCEL_ROOT / "scripts"
for path in [REPO_ROOT, SRC_ROOT, ACCEL_ROOT, ACCEL_SCRIPTS, SCRIPT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from verify_rich_anchor_encoder_similarity import (  # noqa: E402
    XFeat,
    build_visible_vertex_map,
    compute_fundamental_inliers,
    crop_xy_to_patch,
    evaluate_mesh_geometry,
    load_human3r_image,
    load_mask,
    load_ply_vertices,
    load_rgb,
    raw_to_crop_xy,
    resize_for_matching,
    seq_name,
    to_original_coords,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rich_root", default="/workspace/data/RICH")
    parser.add_argument("--data_root", default="/workspace/data/RICH/RICH_4Human3R/Training")
    parser.add_argument("--out_root", default="/workspace/data/RICH/RICH_4Human3R/anchor_cache_guitar_v1")
    parser.add_argument("--source_sequence", default="BBQ_001_guitar")
    parser.add_argument("--camera_pairs", default="6-7,5-6,4-5,3-4,1-2")
    parser.add_argument("--start_frame", type=int, default=5)
    parser.add_argument("--end_frame", type=int, default=374)
    parser.add_argument("--frame_stride", type=int, default=30)
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--top_k_xfeat", type=int, default=8192)
    parser.add_argument("--top_k_tokens", type=int, default=16)
    parser.add_argument("--min_unique_anchors", type=int, default=4)
    parser.add_argument("--max_dim", type=int, default=1200)
    parser.add_argument("--mesh_max_dim", type=int, default=1400)
    parser.add_argument("--mesh_lookup_radius", type=int, default=4)
    parser.add_argument("--mesh_z_tol", type=float, default=0.03)
    parser.add_argument("--reproj_thresh", type=float, default=24.0)
    parser.add_argument("--fundamental_thresh", type=float, default=2.0)
    parser.add_argument("--human3r_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def parse_camera_pairs(text):
    pairs = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        a, b = item.split("-", 1)
        pairs.append((int(a), int(b)))
    return pairs


def patch_center_norm(patch_idx, grid_hw):
    gh, gw = grid_hw
    y = int(patch_idx) // gw
    x = int(patch_idx) % gw
    return np.array([(x + 0.5) / gw, (y + 0.5) / gh], dtype=np.float32)


def fit_affine(ref_norm, cur_norm, weights, ridge=1e-4):
    if len(ref_norm) < 3:
        return None
    x = np.concatenate([ref_norm, np.ones((len(ref_norm), 1), dtype=np.float32)], axis=1)
    w = np.sqrt(weights).reshape(-1, 1).astype(np.float32)
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


def build_candidates(args):
    pairs = parse_camera_pairs(args.camera_pairs)
    frames = list(range(args.start_frame, args.end_frame + 1, args.frame_stride))
    candidates = []
    for frame in frames:
        for cam_a, cam_b in pairs:
            candidates.append((cam_a, cam_b, frame))
    return candidates[: args.max_samples]


def cache_path_for(out_root, source_sequence, cam_a, cam_b, frame):
    rel = Path("samples") / f"{source_sequence}_cam{cam_a:02d}_cam{cam_b:02d}" / f"start_{frame:08d}.npz"
    return Path(out_root) / rel, rel


def process_candidate(args, xfeat, xyz, map_cache, cam_a, cam_b, start_frame, out_root):
    seq_a = seq_name(args.source_sequence, cam_a)
    seq_b = seq_name(args.source_sequence, cam_b)
    ref_frame = start_frame + 1
    cur_frame = start_frame + 2
    ref_bgr, ref_path = load_rgb(args.data_root, seq_a, ref_frame)
    cur_bgr, cur_path = load_rgb(args.data_root, seq_b, cur_frame)

    ref_match, sx_ref, sy_ref = resize_for_matching(ref_bgr, args.max_dim)
    cur_match, sx_cur, sy_cur = resize_for_matching(cur_bgr, args.max_dim)
    mkpts_ref, mkpts_cur = xfeat.match_xfeat_star(ref_match, cur_match, top_k=args.top_k_xfeat)
    mkpts_ref = np.asarray(mkpts_ref, dtype=np.float32)
    mkpts_cur = np.asarray(mkpts_cur, dtype=np.float32)
    mkpts_ref_orig = to_original_coords(mkpts_ref, sx_ref, sy_ref)
    mkpts_cur_orig = to_original_coords(mkpts_cur, sx_cur, sy_cur)

    if seq_a not in map_cache:
        map_cache[seq_a] = build_visible_vertex_map(xyz, args.rich_root, seq_a, ref_bgr.shape, args.mesh_max_dim, args.mesh_z_tol)
    if seq_b not in map_cache:
        map_cache[seq_b] = build_visible_vertex_map(xyz, args.rich_root, seq_b, cur_bgr.shape, args.mesh_max_dim, args.mesh_z_tol)

    mask_ref = load_mask(args.data_root, seq_a, ref_frame, ref_bgr.shape)
    mask_cur = load_mask(args.data_root, seq_b, cur_frame, cur_bgr.shape)
    eval_items = evaluate_mesh_geometry(mkpts_ref_orig, mkpts_cur_orig, map_cache[seq_a], map_cache[seq_b], mask_ref, mask_cur, args)
    mesh_mask = np.array([item["mesh_inlier"] for item in eval_items], dtype=bool)
    fundamental_mask = compute_fundamental_inliers(mkpts_ref, mkpts_cur, args.fundamental_thresh)

    _, ref_true_shape, _, ref_meta = load_human3r_image(ref_path, args.human3r_size)
    _, cur_true_shape, _, cur_meta = load_human3r_image(cur_path, args.human3r_size)
    patch_size = 16
    h_ref, w_ref = map(int, ref_true_shape[0].detach().cpu().numpy().tolist())
    h_cur, w_cur = map(int, cur_true_shape[0].detach().cpu().numpy().tolist())
    ref_grid_hw = (h_ref // patch_size, w_ref // patch_size)
    cur_grid_hw = (h_cur // patch_size, w_cur // patch_size)

    ref_crop_xy, ref_crop_valid = raw_to_crop_xy(mkpts_ref_orig, ref_meta)
    cur_crop_xy, cur_crop_valid = raw_to_crop_xy(mkpts_cur_orig, cur_meta)
    ref_patch_xy, ref_patch_idx, ref_patch_valid = crop_xy_to_patch(ref_crop_xy, ref_crop_valid, patch_size, ref_grid_hw)
    cur_patch_xy, cur_patch_idx, cur_patch_valid = crop_xy_to_patch(cur_crop_xy, cur_crop_valid, patch_size, cur_grid_hw)
    valid = mesh_mask & ref_patch_valid & cur_patch_valid

    best_by_pair = {}
    for idx in np.flatnonzero(valid):
        pair = (int(ref_patch_idx[idx]), int(cur_patch_idx[idx]))
        err = eval_items[int(idx)]["best_mesh_reproj_error_px"]
        err_val = float(err) if err is not None else float("inf")
        if pair in best_by_pair and err_val >= best_by_pair[pair]["mesh_error_px"]:
            continue
        conf = float(np.exp(-err_val / max(args.reproj_thresh, 1e-6)))
        if fundamental_mask[int(idx)]:
            conf *= 1.25
        conf = float(np.clip(conf, 0.0, 1.0))
        ref_pos = patch_center_norm(pair[0], ref_grid_hw)
        cur_pos = patch_center_norm(pair[1], cur_grid_hw)
        best_by_pair[pair] = {
            "match_index": int(idx),
            "ref_patch_idx": pair[0],
            "cur_patch_idx": pair[1],
            "ref_patch_xy": ref_patch_xy[idx].astype(np.int32).tolist(),
            "cur_patch_xy": cur_patch_xy[idx].astype(np.int32).tolist(),
            "ref_pos_norm": ref_pos.astype(float).tolist(),
            "cur_pos_norm": cur_pos.astype(float).tolist(),
            "delta_uv_norm": (cur_pos - ref_pos).astype(float).tolist(),
            "confidence": conf,
            "mesh_error_px": err_val,
            "fundamental_inlier": bool(fundamental_mask[int(idx)]),
        }
    anchors = list(best_by_pair.values())
    if len(anchors) < args.min_unique_anchors:
        return None, {
            "status": "skipped_few_anchors",
            "cam_a": cam_a,
            "cam_b": cam_b,
            "start_frame": start_frame,
            "raw_matches": int(len(mkpts_ref)),
            "mesh_geometry_inliers": int(mesh_mask.sum()),
            "unique_anchor_patch_pairs": int(len(anchors)),
        }

    ref_norm = np.array([a["ref_pos_norm"] for a in anchors], dtype=np.float32)
    cur_norm = np.array([a["cur_pos_norm"] for a in anchors], dtype=np.float32)
    weights = np.array([a["confidence"] for a in anchors], dtype=np.float32)
    affine = fit_affine(ref_norm, cur_norm, weights)
    inv_affine = invert_affine(affine)
    if affine is None:
        affine = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        inv_affine = affine.copy()
    affine_err = patch_error(apply_affine(ref_norm, affine), cur_norm, cur_grid_hw)
    median_affine_err = float(np.median(affine_err))
    q_gate = quality_gate(len(anchors), median_affine_err)

    selected = select_spatial_diverse(anchors, args.top_k_tokens)
    top = [anchors[i] for i in selected]
    out_path, rel_path = cache_path_for(out_root, args.source_sequence, cam_a, cam_b, start_frame)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        ref_patch_idx=np.array([a["ref_patch_idx"] for a in top], dtype=np.int32),
        cur_patch_idx=np.array([a["cur_patch_idx"] for a in top], dtype=np.int32),
        ref_patch_xy=np.array([a["ref_patch_xy"] for a in top], dtype=np.int32),
        cur_patch_xy=np.array([a["cur_patch_xy"] for a in top], dtype=np.int32),
        ref_pos_norm=np.array([a["ref_pos_norm"] for a in top], dtype=np.float32),
        cur_pos_norm=np.array([a["cur_pos_norm"] for a in top], dtype=np.float32),
        delta_uv_norm=np.array([a["delta_uv_norm"] for a in top], dtype=np.float32),
        confidence=np.array([a["confidence"] for a in top], dtype=np.float32),
        mesh_error_px=np.array([a["mesh_error_px"] for a in top], dtype=np.float32),
        fundamental_inlier=np.array([a["fundamental_inlier"] for a in top], dtype=np.bool_),
        affine_forward=affine.astype(np.float32),
        affine_inverse=inv_affine.astype(np.float32),
        ref_grid_hw=np.array(ref_grid_hw, dtype=np.int32),
        cur_grid_hw=np.array(cur_grid_hw, dtype=np.int32),
        quality_gate=np.array([q_gate], dtype=np.float32),
    )
    record = {
        "status": "ok",
        "source_sequence": args.source_sequence,
        "cam_a": int(cam_a),
        "cam_b": int(cam_b),
        "start_frame": int(start_frame),
        "aabb_frames": [int(start_frame), int(start_frame + 1), int(start_frame + 2), int(start_frame + 3)],
        "boundary": {
            "ref_seq": seq_a,
            "ref_frame": int(ref_frame),
            "cur_seq": seq_b,
            "cur_frame": int(cur_frame),
        },
        "cache_path": str(rel_path),
        "raw_matches": int(len(mkpts_ref)),
        "mesh_geometry_inliers": int(mesh_mask.sum()),
        "mesh_inliers_inside_fundamental": int((mesh_mask & fundamental_mask).sum()),
        "unique_anchor_patch_pairs": int(len(anchors)),
        "top_k_tokens": int(len(top)),
        "quality_gate": q_gate,
        "affine_median_patch_error": median_affine_err,
        "ref_grid_hw": list(map(int, ref_grid_hw)),
        "cur_grid_hw": list(map(int, cur_grid_hw)),
        "confidence_mean": float(np.mean([a["confidence"] for a in top])) if top else None,
        "confidence_min": float(np.min([a["confidence"] for a in top])) if top else None,
    }
    return record, None


def main():
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "manifest.jsonl"
    skipped_path = out_root / "skipped.jsonl"
    summary_path = out_root / "summary.json"
    candidates = build_candidates(args)

    print(f"Generating {len(candidates)} guitar AnchorToken cache candidates into {out_root}")
    mesh_path = Path(args.rich_root) / "scan_calibration" / "BBQ" / "scan_camcoord.ply"
    xyz, _ = load_ply_vertices(mesh_path)
    xfeat = XFeat(top_k=args.top_k_xfeat)
    map_cache = {}
    records = []
    skipped = []
    with manifest_path.open("w", encoding="utf-8") as mf, skipped_path.open("w", encoding="utf-8") as sf:
        for index, (cam_a, cam_b, frame) in enumerate(candidates):
            print(f"[{index + 1}/{len(candidates)}] {args.source_sequence} cam{cam_a:02d}->cam{cam_b:02d} start={frame}")
            try:
                record, skip = process_candidate(args, xfeat, xyz, map_cache, cam_a, cam_b, frame, out_root)
            except Exception as exc:
                record, skip = None, {
                    "status": "error",
                    "cam_a": int(cam_a),
                    "cam_b": int(cam_b),
                    "start_frame": int(frame),
                    "error": repr(exc),
                }
            if record is not None:
                records.append(record)
                mf.write(json.dumps(record, ensure_ascii=False) + "\n")
                mf.flush()
            if skip is not None:
                skipped.append(skip)
                sf.write(json.dumps(skip, ensure_ascii=False) + "\n")
                sf.flush()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = {
        "args": vars(args),
        "out_root": str(out_root),
        "num_candidates": int(len(candidates)),
        "num_cached": int(len(records)),
        "num_skipped": int(len(skipped)),
        "manifest": str(manifest_path),
        "skipped_manifest": str(skipped_path),
        "quality_gate_mean": float(np.mean([r["quality_gate"] for r in records])) if records else None,
        "unique_anchor_patch_pairs_mean": float(np.mean([r["unique_anchor_patch_pairs"] for r in records])) if records else None,
        "top_k_tokens": int(args.top_k_tokens),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
