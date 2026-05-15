#!/usr/bin/env python3
"""AABB-level Step1 verification for external anchors and Human3R encoder tokens.

This script keeps the model untouched. It evaluates an AABB sample:

    [A@t, A@t+1, B@t+2, B@t+3]

and compares three pairs:

    A@t   -> A@t+1     contiguous reference pair
    A@t+1 -> B@t+2     shot-boundary pair, the main target
    B@t+2 -> B@t+3     contiguous current pair

For each pair it runs semi-dense XFeat matching, filters matches with the RICH
official static mesh, maps surviving anchors to Human3R encoder patches, and
reports whether corresponding patch tokens are more similar than random patches.
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from report_image_style import patch_cv2_text

patch_cv2_text(cv2)

from verify_rich_anchor_encoder_similarity import (  # noqa: E402
    REPO_ROOT,
    XFeat,
    build_visible_vertex_map,
    compute_fundamental_inliers,
    compute_ransac_inliers,
    compute_stats,
    compute_visible_overlap,
    crop_xy_to_patch,
    draw_patch_grid,
    draw_raw_mesh_matches,
    draw_side_by_side_matches,
    evaluate_mesh_geometry,
    heatmap_image,
    load_human3r_image,
    load_mask,
    load_model,
    load_ply_vertices,
    load_rgb,
    raw_to_crop_xy,
    resize_for_matching,
    save_histogram,
    save_rank_chart,
    save_rgb,
    seq_name,
    to_original_coords,
)


PAIR_SPECS = [
    ("pair_00_A_t0_to_A_t1", 0, 1, "contiguous_ref"),
    ("pair_01_A_t1_to_B_t2_BOUNDARY", 1, 2, "shot_boundary"),
    ("pair_02_B_t2_to_B_t3", 2, 3, "contiguous_cur"),
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    # **========== 原始代码：旧服务器 RICH 路径 ==========**
    # parser.add_argument("--rich_root", default="/workspace/data/RICH")
    # parser.add_argument("--data_root", default="/workspace/data/RICH/RICH_4Human3R/Training")
    # **========== 新代码：当前服务器 RICH 路径 ==========**
    parser.add_argument("--rich_root", default=str(REPO_ROOT.parent / "data"))
    parser.add_argument("--data_root", default=str(REPO_ROOT.parent / "data" / "RICH_4Human3R" / "Training"))
    # **========== 结束 ==========**
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
    parser.add_argument("--num_similarity_examples", type=int, default=4)
    parser.add_argument("--sample_counts", default="4,8,16,32")
    parser.add_argument("--sample_trials", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", default=None)
    return parser.parse_args()


def parse_sample_counts(text):
    counts = []
    for item in text.split(","):
        item = item.strip()
        if item:
            counts.append(int(item))
    return counts


def patch_center(patch_idx, patch_size, grid_hw):
    _, gw = grid_hw
    y = int(patch_idx) // gw
    x = int(patch_idx) % gw
    return np.array([(x + 0.5) * patch_size, (y + 0.5) * patch_size], dtype=np.float32)


def encode_view(model, view, size, device):
    img_tensor, true_shape, crop_rgb, meta = load_human3r_image(view["path"], size)
    img_tensor = img_tensor.to(device)
    true_shape = true_shape.to(device)
    with torch.no_grad():
        enc = model._encode_image(img_tensor, true_shape)[0][-1]
        dec = model.decoder_embed(enc)
    h, w = map(int, true_shape[0].detach().cpu().numpy().tolist())
    patch_size = int(model.croco_args["patch_size"])
    grid_hw = (h // patch_size, w // patch_size)
    if enc.shape[1] != grid_hw[0] * grid_hw[1]:
        raise RuntimeError(f"token/grid mismatch for {view['name']}: {enc.shape[1]} vs {grid_hw}")
    return {
        "enc": enc,
        "dec": dec,
        "crop_rgb": crop_rgb,
        "meta": meta,
        "true_shape_hw": [h, w],
        "grid_hw": grid_hw,
    }


def draw_delta_scatter(path, anchors, title):
    width, height = 760, 560
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    cv2.putText(canvas, title, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2, cv2.LINE_AA)
    if not anchors:
        cv2.imwrite(str(path), canvas)
        return
    deltas = np.array([a["delta_uv_norm"] for a in anchors], dtype=np.float32)
    center = np.median(deltas, axis=0)
    span = np.percentile(np.abs(deltas - center), 95, axis=0)
    span = np.maximum(span, 0.02)
    span = float(max(span[0], span[1]))
    left, right, top, bottom = 76, 34, 68, 62
    plot_w = width - left - right
    plot_h = height - top - bottom
    cv2.rectangle(canvas, (left, top), (left + plot_w, top + plot_h), (220, 220, 220), 1)
    cv2.line(canvas, (left, top + plot_h // 2), (left + plot_w, top + plot_h // 2), (210, 210, 210), 1)
    cv2.line(canvas, (left + plot_w // 2, top), (left + plot_w // 2, top + plot_h), (210, 210, 210), 1)
    for a in anchors:
        dx, dy = a["delta_uv_norm"]
        sim = a.get("encoder_cosine", 0.0)
        x = left + int(round(((dx - center[0]) / (2 * span) + 0.5) * plot_w))
        y = top + int(round(((dy - center[1]) / (2 * span) + 0.5) * plot_h))
        x = int(np.clip(x, left, left + plot_w))
        y = int(np.clip(y, top, top + plot_h))
        color_val = int(np.clip((sim + 1.0) / 2.0 * 255.0, 0, 255))
        color = cv2.applyColorMap(np.array([[color_val]], dtype=np.uint8), cv2.COLORMAP_TURBO)[0, 0]
        cv2.circle(canvas, (x, y), 4, tuple(int(c) for c in color), -1, cv2.LINE_AA)
    med_text = f"median delta_norm=({center[0]:.4f}, {center[1]:.4f}), anchors={len(anchors)}"
    cv2.putText(canvas, med_text, (left, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def save_similarity_examples(out_dir, anchors, ref_rgb, cur_rgb, sim_matrix, patch_size, ref_grid_hw, cur_grid_hw, max_examples):
    if not anchors:
        return []
    sims = np.array([a["encoder_cosine"] for a in anchors], dtype=np.float32)
    order = np.argsort(-sims)
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
        title = f"anchor #{idx}: cosine={a['encoder_cosine']:.3f}, rank={a['encoder_rank']}, delta_norm=({a['delta_uv_norm'][0]:.3f},{a['delta_uv_norm'][1]:.3f})"
        cv2.putText(canvas, title, (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
        path = out_dir / f"60_similarity_map_anchor_{n:02d}.jpg"
        save_rgb(path, canvas)
        paths.append(str(path))
    return paths


def summarize_ranks(ranks):
    ranks = np.asarray(ranks, dtype=np.float32)
    if len(ranks) == 0:
        return {
            "count": 0,
            "median": None,
            "mean": None,
            "top1": None,
            "top5": None,
            "top10": None,
            "top32": None,
        }
    return {
        "count": int(len(ranks)),
        "median": float(np.median(ranks)),
        "mean": float(ranks.mean()),
        "top1": float((ranks <= 1).mean()),
        "top5": float((ranks <= 5).mean()),
        "top10": float((ranks <= 10).mean()),
        "top32": float((ranks <= 32).mean()),
    }


def summarize_delta(anchors):
    if not anchors:
        return None
    deltas = np.array([a["delta_uv_norm"] for a in anchors], dtype=np.float32)
    patch_deltas = np.array([a["delta_patch_xy"] for a in anchors], dtype=np.float32)
    return {
        "delta_uv_norm_mean": deltas.mean(axis=0).astype(float).tolist(),
        "delta_uv_norm_median": np.median(deltas, axis=0).astype(float).tolist(),
        "delta_uv_norm_std": deltas.std(axis=0).astype(float).tolist(),
        "delta_patch_xy_mean": patch_deltas.mean(axis=0).astype(float).tolist(),
        "delta_patch_xy_median": np.median(patch_deltas, axis=0).astype(float).tolist(),
        "delta_patch_xy_std": patch_deltas.std(axis=0).astype(float).tolist(),
    }


def sample_delta_stability(anchors, sample_counts, trials, rng):
    if not anchors:
        return {}
    deltas = np.array([a["delta_uv_norm"] for a in anchors], dtype=np.float32)
    weights = np.array([max(float(a.get("encoder_cosine", 0.0)), 0.05) for a in anchors], dtype=np.float32)
    full = np.average(deltas, axis=0, weights=weights)
    out = {
        "full_weighted_delta_uv_norm": full.astype(float).tolist(),
    }
    for n in sample_counts:
        if len(anchors) < n:
            continue
        estimates = []
        errors = []
        for _ in range(trials):
            idx = rng.choice(len(anchors), size=n, replace=False)
            est = np.average(deltas[idx], axis=0, weights=weights[idx])
            estimates.append(est)
            errors.append(float(np.linalg.norm(est - full)))
        estimates = np.asarray(estimates, dtype=np.float32)
        errors = np.asarray(errors, dtype=np.float32)
        out[str(n)] = {
            "trials": int(trials),
            "estimate_mean": estimates.mean(axis=0).astype(float).tolist(),
            "estimate_std": estimates.std(axis=0).astype(float).tolist(),
            "error_to_full_mean": float(errors.mean()),
            "error_to_full_median": float(np.median(errors)),
            "error_to_full_p90": float(np.percentile(errors, 90)),
        }
    return out


def analyze_pair(args, pair_name, pair_kind, ref_view, cur_view, ref_enc, cur_enc, xfeat, xyz, visible_maps, masks, patch_size, rng, sample_counts, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_bgr = ref_view["bgr"]
    cur_bgr = cur_view["bgr"]
    ref_match, sx_ref, sy_ref = resize_for_matching(ref_bgr, args.max_dim)
    cur_match, sx_cur, sy_cur = resize_for_matching(cur_bgr, args.max_dim)
    mkpts_ref, mkpts_cur = xfeat.match_xfeat_star(ref_match, cur_match, top_k=args.top_k)
    mkpts_ref = np.asarray(mkpts_ref, dtype=np.float32)
    mkpts_cur = np.asarray(mkpts_cur, dtype=np.float32)
    mkpts_ref_orig = to_original_coords(mkpts_ref, sx_ref, sy_ref)
    mkpts_cur_orig = to_original_coords(mkpts_cur, sx_cur, sy_cur)

    ref_map = visible_maps[ref_view["name"]]
    cur_map = visible_maps[cur_view["name"]]
    mask_ref = masks[ref_view["name"]]
    mask_cur = masks[cur_view["name"]]
    eval_items = evaluate_mesh_geometry(mkpts_ref_orig, mkpts_cur_orig, ref_map, cur_map, mask_ref, mask_cur, args)
    mesh_mask = np.array([item["mesh_inlier"] for item in eval_items], dtype=bool)
    human_mask = np.array([item["on_human"] for item in eval_items], dtype=bool)
    mesh_indices = np.flatnonzero(mesh_mask)
    ransac_mask = compute_ransac_inliers(mkpts_ref, mkpts_cur, args.ransac_thresh)
    fundamental_mask = compute_fundamental_inliers(mkpts_ref, mkpts_cur, args.fundamental_thresh)
    visible_overlap = compute_visible_overlap(ref_map, cur_map, mask_ref, mask_cur)

    draw_raw_mesh_matches(ref_bgr, cur_bgr, mkpts_ref_orig, mkpts_cur_orig, mesh_indices, out_dir / "00_semidense_mesh_inliers_raw_space.jpg", args.max_draw)

    ref_crop_xy, ref_crop_valid = raw_to_crop_xy(mkpts_ref_orig, ref_enc["meta"])
    cur_crop_xy, cur_crop_valid = raw_to_crop_xy(mkpts_cur_orig, cur_enc["meta"])
    ref_patch_xy, ref_patch_idx, ref_patch_valid = crop_xy_to_patch(ref_crop_xy, ref_crop_valid, patch_size, ref_enc["grid_hw"])
    cur_patch_xy, cur_patch_idx, cur_patch_valid = crop_xy_to_patch(cur_crop_xy, cur_crop_valid, patch_size, cur_enc["grid_hw"])
    mapped_valid = mesh_mask & ref_patch_valid & cur_patch_valid

    best_by_pair = {}
    ref_w, ref_h = ref_enc["meta"]["final_size_wh"]
    cur_w, cur_h = cur_enc["meta"]["final_size_wh"]
    for idx in np.flatnonzero(mapped_valid):
        pair = (int(ref_patch_idx[idx]), int(cur_patch_idx[idx]))
        err = eval_items[int(idx)]["best_mesh_reproj_error_px"]
        err_val = float(err) if err is not None else float("inf")
        if pair in best_by_pair and err_val >= best_by_pair[pair]["mesh_error_px"]:
            continue
        ref_center = patch_center(pair[0], patch_size, ref_enc["grid_hw"])
        cur_center = patch_center(pair[1], patch_size, cur_enc["grid_hw"])
        ref_norm = ref_center / np.array([ref_w, ref_h], dtype=np.float32)
        cur_norm = cur_center / np.array([cur_w, cur_h], dtype=np.float32)
        best_by_pair[pair] = {
            "match_index": int(idx),
            "ref_patch_idx": pair[0],
            "cur_patch_idx": pair[1],
            "ref_patch_xy": ref_patch_xy[idx].astype(int).tolist(),
            "cur_patch_xy": cur_patch_xy[idx].astype(int).tolist(),
            "delta_patch_xy": (cur_patch_xy[idx] - ref_patch_xy[idx]).astype(int).tolist(),
            "ref_patch_center_crop_xy": ref_center.astype(float).tolist(),
            "cur_patch_center_crop_xy": cur_center.astype(float).tolist(),
            "delta_uv_crop_px": (cur_center - ref_center).astype(float).tolist(),
            "delta_uv_norm": (cur_norm - ref_norm).astype(float).tolist(),
            "ref_xy_crop": ref_crop_xy[idx].astype(float).tolist(),
            "cur_xy_crop": cur_crop_xy[idx].astype(float).tolist(),
            "ref_xy_original": mkpts_ref_orig[idx].astype(float).tolist(),
            "cur_xy_original": mkpts_cur_orig[idx].astype(float).tolist(),
            "mesh_error_px": err_val,
        }
    anchors = list(best_by_pair.values())

    ref_token = F.normalize(ref_enc["enc"][0].float(), dim=-1)
    cur_token = F.normalize(cur_enc["enc"][0].float(), dim=-1)
    ref_dec = F.normalize(ref_enc["dec"][0].float(), dim=-1)
    cur_dec = F.normalize(cur_enc["dec"][0].float(), dim=-1)
    sim_matrix = (ref_token @ cur_token.T).detach().cpu().numpy()
    dec_sim_matrix = (ref_dec @ cur_dec.T).detach().cpu().numpy()

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

    negative = []
    negative_dec = []
    shuffled = []
    shuffled_dec = []
    n_cur_tokens = cur_enc["enc"].shape[1]
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

    draw_side_by_side_matches(ref_enc["crop_rgb"], cur_enc["crop_rgb"], anchors, patch_size, ref_enc["grid_hw"], cur_enc["grid_hw"], out_dir / "01_anchor_patches_on_human3r_crop.jpg", args.max_draw)
    save_rgb(out_dir / "02_ref_human3r_crop_grid.jpg", draw_patch_grid(ref_enc["crop_rgb"], patch_size))
    save_rgb(out_dir / "03_cur_human3r_crop_grid.jpg", draw_patch_grid(cur_enc["crop_rgb"], patch_size))
    save_histogram(out_dir / "10_encoder_cosine_pos_vs_neg.jpg", [positive, negative, shuffled], ["mesh anchor positives", "random negatives", "shuffled anchor negatives"], "Human3R encoder token cosine")
    save_histogram(out_dir / "11_decoder_embed_cosine_pos_vs_neg.jpg", [positive_dec, negative_dec, shuffled_dec], ["mesh anchor positives", "random negatives", "shuffled anchor negatives"], "Human3R decoder_embed token cosine")
    save_rank_chart(out_dir / "12_encoder_true_match_rank.jpg", ranks, "rank of true mesh anchor patch among all current patches")
    save_rank_chart(out_dir / "13_decoder_embed_true_match_rank.jpg", ranks_dec, "decoder_embed rank of true mesh anchor patch")
    draw_delta_scatter(out_dir / "20_delta_uv_norm_scatter.jpg", anchors, f"{pair_name}: normalized patch displacement")
    example_paths = save_similarity_examples(out_dir, anchors, ref_enc["crop_rgb"], cur_enc["crop_rgb"], sim_matrix, patch_size, ref_enc["grid_hw"], cur_enc["grid_hw"], args.num_similarity_examples)

    pos_arr = np.asarray(positive, dtype=np.float32)
    neg_arr = np.asarray(negative, dtype=np.float32)
    pairwise_accuracy = float((pos_arr > neg_arr).mean()) if len(pos_arr) and len(neg_arr) else None

    summary = {
        "pair_name": pair_name,
        "pair_kind": pair_kind,
        "ref": {"name": ref_view["name"], "seq": ref_view["seq"], "frame": int(ref_view["frame"]), "image": str(ref_view["path"])},
        "cur": {"name": cur_view["name"], "seq": cur_view["seq"], "frame": int(cur_view["frame"]), "image": str(cur_view["path"])},
        "raw_matches": int(len(mkpts_ref)),
        "homography_ransac_inliers": int(ransac_mask.sum()),
        "fundamental_ransac_inliers": int(fundamental_mask.sum()),
        "mesh_geometry_inliers": int(mesh_mask.sum()),
        "mesh_inliers_inside_fundamental": int((mesh_mask & fundamental_mask).sum()),
        "matches_on_human": int(human_mask.sum()),
        "mesh_visible_overlap": visible_overlap,
        "human3r_ref_grid_hw": list(ref_enc["grid_hw"]),
        "human3r_cur_grid_hw": list(cur_enc["grid_hw"]),
        "mesh_inliers_after_human3r_crop": int(mapped_valid.sum()),
        "unique_anchor_patch_pairs": int(len(anchors)),
        "encoder_positive_cosine": compute_stats(positive),
        "encoder_random_negative_cosine": compute_stats(negative),
        "encoder_shuffled_negative_cosine": compute_stats(shuffled),
        "decoder_embed_positive_cosine": compute_stats(positive_dec),
        "decoder_embed_random_negative_cosine": compute_stats(negative_dec),
        "decoder_embed_shuffled_negative_cosine": compute_stats(shuffled_dec),
        "encoder_true_match_rank": summarize_ranks(ranks),
        "decoder_embed_true_match_rank": summarize_ranks(ranks_dec),
        "positive_greater_than_random_negative": pairwise_accuracy,
        "delta_summary": summarize_delta(anchors),
        "sample_delta_stability": sample_delta_stability(anchors, sample_counts, args.sample_trials, rng),
        "anchors": anchors,
        "visualizations": {
            "raw_mesh_matches": str(out_dir / "00_semidense_mesh_inliers_raw_space.jpg"),
            "anchor_patch_matches": str(out_dir / "01_anchor_patches_on_human3r_crop.jpg"),
            "encoder_histogram": str(out_dir / "10_encoder_cosine_pos_vs_neg.jpg"),
            "encoder_rank_chart": str(out_dir / "12_encoder_true_match_rank.jpg"),
            "delta_scatter": str(out_dir / "20_delta_uv_norm_scatter.jpg"),
            "similarity_examples": example_paths,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def draw_aabb_comparison(path, pair_summaries):
    width, height = 1280, 360
    row_h = height // 4
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    cv2.putText(canvas, "AABB Step1 anchor/encoder comparison", (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20, 20, 20), 2, cv2.LINE_AA)
    headers = ["pair", "mesh", "patch", "pos cos", "rand cos", "rank med", "pos>rand", "delta med"]
    xs = [18, 385, 500, 615, 750, 890, 1025, 1145]
    for x, text in zip(xs, headers):
        cv2.putText(canvas, text, (x, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (50, 50, 50), 2, cv2.LINE_AA)
    for i, s in enumerate(pair_summaries):
        y = 118 + i * row_h
        kind_color = (35, 70, 220) if s["pair_kind"] == "shot_boundary" else (40, 130, 40)
        cv2.putText(canvas, s["pair_name"], (xs[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, kind_color, 2, cv2.LINE_AA)
        pos = s["encoder_positive_cosine"] or {}
        neg = s["encoder_random_negative_cosine"] or {}
        rank = s["encoder_true_match_rank"] or {}
        delta = s["delta_summary"] or {}
        delta_med = delta.get("delta_uv_norm_median")
        delta_text = "None" if delta_med is None else f"({delta_med[0]:.3f},{delta_med[1]:.3f})"
        vals = [
            str(s["mesh_geometry_inliers"]),
            str(s["unique_anchor_patch_pairs"]),
            f"{pos.get('mean'):.3f}" if pos.get("mean") is not None else "None",
            f"{neg.get('mean'):.3f}" if neg.get("mean") is not None else "None",
            f"{rank.get('median'):.1f}" if rank.get("median") is not None else "None",
            f"{s['positive_greater_than_random_negative']:.3f}" if s["positive_greater_than_random_negative"] is not None else "None",
            delta_text,
        ]
        for x, text in zip(xs[1:], vals):
            cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    sample_counts = parse_sample_counts(args.sample_counts)

    seq_a = seq_name(args.source_sequence, args.cam_a)
    seq_b = seq_name(args.source_sequence, args.cam_b)
    views = [
        {"name": "A_t0", "seq": seq_a, "frame": args.start_frame},
        {"name": "A_t1", "seq": seq_a, "frame": args.start_frame + 1},
        {"name": "B_t2", "seq": seq_b, "frame": args.start_frame + 2},
        {"name": "B_t3", "seq": seq_b, "frame": args.start_frame + 3},
    ]

    if args.out_dir is None:
        args.out_dir = str(
            REPO_ROOT
            / "output"
            / "rich_aabb_anchor_step1"
            / f"{args.source_sequence}_cam{args.cam_a:02d}_cam{args.cam_b:02d}_f{args.start_frame:08d}"
        )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for view in views:
        bgr, path = load_rgb(args.data_root, view["seq"], view["frame"])
        view["bgr"] = bgr
        view["path"] = path
        view["shape"] = bgr.shape

    print("Loading original Human3R encoder once for all AABB views...")
    model = load_model(args.model_path, device=device, verbose=True).eval()
    model.gradient_checkpointing = False
    patch_size = int(model.croco_args["patch_size"])
    encoded = {view["name"]: encode_view(model, view, args.size, device) for view in views}

    print("Building RICH static mesh visibility maps...")
    mesh_path = Path(args.rich_root) / "scan_calibration" / "BBQ" / "scan_camcoord.ply"
    xyz, _ = load_ply_vertices(mesh_path)
    visible_maps = {}
    masks = {}
    for view in views:
        visible_maps[view["name"]] = build_visible_vertex_map(xyz, args.rich_root, view["seq"], view["shape"], args.mesh_max_dim, args.mesh_z_tol)
        masks[view["name"]] = load_mask(args.data_root, view["seq"], view["frame"], view["shape"])

    print("Running semi-dense XFeat for AABB pairs...")
    xfeat = XFeat(top_k=args.top_k)
    pair_summaries = []
    for pair_name, ref_idx, cur_idx, pair_kind in PAIR_SPECS:
        print(f"Analyzing {pair_name} ({pair_kind})...")
        summary = analyze_pair(
            args=args,
            pair_name=pair_name,
            pair_kind=pair_kind,
            ref_view=views[ref_idx],
            cur_view=views[cur_idx],
            ref_enc=encoded[views[ref_idx]["name"]],
            cur_enc=encoded[views[cur_idx]["name"]],
            xfeat=xfeat,
            xyz=xyz,
            visible_maps=visible_maps,
            masks=masks,
            patch_size=patch_size,
            rng=rng,
            sample_counts=sample_counts,
            out_dir=out_dir / pair_name,
        )
        pair_summaries.append(summary)

    draw_aabb_comparison(out_dir / "aabb_comparison.jpg", pair_summaries)
    boundary = next(s for s in pair_summaries if s["pair_kind"] == "shot_boundary")
    root_summary = {
        "args": vars(args),
        "aabb": [
            {"name": view["name"], "seq": view["seq"], "frame": int(view["frame"]), "image": str(view["path"])}
            for view in views
        ],
        "human3r_patch_size": int(patch_size),
        "pair_summaries": pair_summaries,
        "boundary_key_metrics": {
            "mesh_geometry_inliers": boundary["mesh_geometry_inliers"],
            "unique_anchor_patch_pairs": boundary["unique_anchor_patch_pairs"],
            "encoder_positive_cosine": boundary["encoder_positive_cosine"],
            "encoder_random_negative_cosine": boundary["encoder_random_negative_cosine"],
            "encoder_true_match_rank": boundary["encoder_true_match_rank"],
            "positive_greater_than_random_negative": boundary["positive_greater_than_random_negative"],
            "delta_summary": boundary["delta_summary"],
            "sample_delta_stability": boundary["sample_delta_stability"],
        },
        "visualizations": {
            "aabb_comparison": str(out_dir / "aabb_comparison.jpg"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(root_summary, indent=2), encoding="utf-8")

    concise = {
        "out_dir": str(out_dir),
        "boundary_mesh_geometry_inliers": boundary["mesh_geometry_inliers"],
        "boundary_unique_anchor_patch_pairs": boundary["unique_anchor_patch_pairs"],
        "boundary_encoder_positive_cosine": boundary["encoder_positive_cosine"],
        "boundary_encoder_random_negative_cosine": boundary["encoder_random_negative_cosine"],
        "boundary_encoder_true_match_rank": boundary["encoder_true_match_rank"],
        "boundary_positive_greater_than_random_negative": boundary["positive_greater_than_random_negative"],
        "boundary_delta_summary": boundary["delta_summary"],
    }
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()
