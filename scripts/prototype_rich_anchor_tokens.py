#!/usr/bin/env python3
"""Prototype local AnchorTokens for AABB shot-boundary re-anchoring.

This script keeps Human3R/Movie3R unchanged. It reads mesh-verified AABB
boundary anchors, reruns the frozen Human3R encoder for the boundary images, and
builds structured local AnchorTokens:

    key:   current patch encoder token + current patch position
    value: reference patch position + local correction residual
    meta:  confidence, mesh error, delta_uv, patch ids

It then validates whether those AnchorTokens can predict held-out reference
patches better than same-position / average translation / global affine.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
SRC_ROOT = REPO_ROOT / "src"
for path in [REPO_ROOT, SRC_ROOT, SCRIPT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from report_image_style import patch_cv2_text  # noqa: E402

patch_cv2_text(cv2)

from analyze_rich_aabb_anchor_correction import (  # noqa: E402
    fit_affine,
    load_boundary_summary,
    stats,
    weighted_translation,
)
from verify_rich_anchor_encoder_similarity import (  # noqa: E402
    draw_patch_grid,
    load_human3r_image,
    load_model,
    save_rgb,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aabb_dir", action="append", required=True, help="AABB Step1 output directory. Can be repeated.")
    parser.add_argument("--model_path", default=str(REPO_ROOT / "src" / "human3r_896L.pth"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--sample_counts", default="4,8,16,32")
    parser.add_argument("--trials", type=int, default=256)
    parser.add_argument("--feature_weight", type=float, default=2.0)
    parser.add_argument("--spatial_weight", type=float, default=30.0)
    parser.add_argument("--attention_temperature", type=float, default=1.0)
    parser.add_argument("--max_draw", type=int, default=120)
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


def encode_image(model, image_path, size, device):
    img_tensor, true_shape, crop_rgb, meta = load_human3r_image(image_path, size)
    img_tensor = img_tensor.to(device)
    true_shape = true_shape.to(device)
    with torch.no_grad():
        enc = model._encode_image(img_tensor, true_shape)[0][-1]
    h, w = map(int, true_shape[0].detach().cpu().numpy().tolist())
    patch_size = int(model.croco_args["patch_size"])
    grid_hw = (h // patch_size, w // patch_size)
    if enc.shape[1] != grid_hw[0] * grid_hw[1]:
        raise RuntimeError(f"token/grid mismatch for {image_path}: {enc.shape[1]} vs {grid_hw}")
    enc = F.normalize(enc[0].float(), dim=-1).detach().cpu().numpy().astype(np.float32)
    return {
        "enc": enc,
        "crop_rgb": crop_rgb,
        "meta": meta,
        "grid_hw": grid_hw,
        "patch_size": patch_size,
        "true_shape_hw": [h, w],
    }


def patch_idx_to_xy(idx, grid_hw):
    _, gw = grid_hw
    return np.array([int(idx) % gw, int(idx) // gw], dtype=np.float32)


def patch_idx_to_center_norm(idx, grid_hw):
    gh, gw = grid_hw
    xy = patch_idx_to_xy(idx, grid_hw)
    return np.array([(xy[0] + 0.5) / gw, (xy[1] + 0.5) / gh], dtype=np.float32)


def patch_idx_to_center_px(idx, grid_hw, patch_size):
    return (patch_idx_to_xy(idx, grid_hw) + 0.5) * float(patch_size)


def nearest_patch_indices(points_norm, grid_hw):
    gh, gw = grid_hw
    pts = np.asarray(points_norm, dtype=np.float32)
    x = np.floor(pts[:, 0] * gw).astype(np.int32)
    y = np.floor(pts[:, 1] * gh).astype(np.int32)
    x = np.clip(x, 0, gw - 1)
    y = np.clip(y, 0, gh - 1)
    return (y * gw + x).astype(np.int64)


def patch_error(pred_ref_norm, true_ref_norm, ref_grid_hw):
    gh, gw = ref_grid_hw
    diff = (np.asarray(pred_ref_norm, dtype=np.float32) - np.asarray(true_ref_norm, dtype=np.float32))
    diff_patch = diff * np.array([gw, gh], dtype=np.float32)[None]
    return np.linalg.norm(diff_patch, axis=1)


def token_cosine_for_pred(pred_ref_norm, cur_idx, ref_features, cur_features, ref_grid_hw):
    pred_idx = nearest_patch_indices(pred_ref_norm, ref_grid_hw)
    return np.array([float(ref_features[int(r)] @ cur_features[int(c)]) for r, c in zip(pred_idx, cur_idx)], dtype=np.float32)


def rank_for_pred(pred_ref_norm, cur_idx, ref_features, cur_features, ref_grid_hw):
    pred_idx = nearest_patch_indices(pred_ref_norm, ref_grid_hw)
    sims = ref_features @ cur_features[cur_idx].T
    ranks = []
    for n, ref_i in enumerate(pred_idx):
        val = sims[int(ref_i), n]
        ranks.append(int((sims[:, n] > val).sum() + 1))
    return np.asarray(ranks, dtype=np.float32)


def fit_models(train, ref_norm, cur_norm, weights):
    translation = weighted_translation(ref_norm[train], cur_norm[train], weights[train])
    affine = fit_affine(ref_norm[train], cur_norm[train], weights[train])
    inv_affine = invert_affine(affine)
    return translation, affine, inv_affine


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


def apply_affine(points_norm, affine):
    pts = np.asarray(points_norm, dtype=np.float32)
    x = np.concatenate([pts, np.ones((len(pts), 1), dtype=np.float32)], axis=1)
    return (x @ affine.T).astype(np.float32)


def softmax(x):
    x = np.asarray(x, dtype=np.float32)
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-8)


def anchor_attention_weights(query_feat, query_pos, token_feat, token_pos, args):
    feat_score = query_feat @ token_feat.T
    spatial_d2 = ((query_pos[:, None, :] - token_pos[None, :, :]) ** 2).sum(axis=-1)
    score = args.feature_weight * feat_score - args.spatial_weight * spatial_d2
    return softmax(score / max(args.attention_temperature, 1e-6))


def predict_with_anchor_tokens(test, train, data, translation, inv_affine, args):
    ref_norm = data["ref_norm"]
    cur_norm = data["cur_norm"]
    cur_feat = data["cur_feat"]
    train_cur_feat = cur_feat[train]
    train_cur_pos = cur_norm[train]
    test_cur_feat = cur_feat[test]
    test_cur_pos = cur_norm[test]

    weights = anchor_attention_weights(test_cur_feat, test_cur_pos, train_cur_feat, train_cur_pos, args)
    token_soft = weights @ ref_norm[train]

    top1 = np.argmax(weights, axis=1)
    token_top1 = ref_norm[train[top1]]

    if inv_affine is None:
        base_test = test_cur_pos - translation[None]
        base_train = train_cur_pos - translation[None]
    else:
        base_test = apply_affine(test_cur_pos, inv_affine)
        base_train = apply_affine(train_cur_pos, inv_affine)
    residual_train = ref_norm[train] - base_train
    token_residual = base_test + weights @ residual_train
    return token_top1, token_soft, token_residual


def summarize_method(name, pred_ref_norm, test, data):
    true_ref = data["ref_norm"][test]
    cur_idx = data["cur_idx"][test]
    err = patch_error(pred_ref_norm, true_ref, data["ref_grid_hw"])
    cos = token_cosine_for_pred(pred_ref_norm, cur_idx, data["ref_feat"], data["cur_feat"], data["ref_grid_hw"])
    ranks = rank_for_pred(pred_ref_norm, cur_idx, data["ref_feat"], data["cur_feat"], data["ref_grid_hw"])
    return {
        "name": name,
        "patch_error": stats(err),
        "token_cosine": stats(cos),
        "rank_median": float(np.median(ranks)) if len(ranks) else None,
        "rank_mean": float(ranks.mean()) if len(ranks) else None,
        "top1_rate": float((ranks <= 1).mean()) if len(ranks) else None,
        "top5_rate": float((ranks <= 5).mean()) if len(ranks) else None,
        "top10_rate": float((ranks <= 10).mean()) if len(ranks) else None,
        "within_1_patch_rate": float((err <= 1.0).mean()) if len(err) else None,
        "within_2_patch_rate": float((err <= 2.0).mean()) if len(err) else None,
    }


def evaluate_split(train, test, data, args):
    train = np.asarray(train, dtype=np.int64)
    test = np.asarray(test, dtype=np.int64)
    translation, affine, inv_affine = fit_models(train, data["ref_norm"], data["cur_norm"], data["weights"])
    cur_test = data["cur_norm"][test]
    pred_same = cur_test.copy()
    pred_translation = cur_test - translation[None]
    pred_affine = pred_translation if inv_affine is None else apply_affine(cur_test, inv_affine)
    token_top1, token_soft, token_residual = predict_with_anchor_tokens(test, train, data, translation, inv_affine, args)
    oracle = data["ref_norm"][test]
    return {
        "same_position": summarize_method("same_position", pred_same, test, data),
        "translation": summarize_method("translation", pred_translation, test, data),
        "affine": summarize_method("affine", pred_affine, test, data),
        "anchor_token_top1": summarize_method("anchor_token_top1", token_top1, test, data),
        "anchor_token_soft": summarize_method("anchor_token_soft", token_soft, test, data),
        "anchor_token_affine_residual": summarize_method("anchor_token_affine_residual", token_residual, test, data),
        "oracle_anchor": summarize_method("oracle_anchor", oracle, test, data),
    }


def aggregate_method_rows(rows):
    methods = rows[0].keys()
    out = {}
    for method in methods:
        out[method] = {}
        for metric_group in ["patch_error", "token_cosine"]:
            vals = np.array([row[method][metric_group]["median"] for row in rows if row[method][metric_group] is not None], dtype=np.float32)
            out[method][f"{metric_group}_median_over_trials"] = stats(vals)
        for scalar in ["rank_median", "top1_rate", "top5_rate", "top10_rate", "within_1_patch_rate", "within_2_patch_rate"]:
            vals = np.array([row[method][scalar] for row in rows if row[method][scalar] is not None], dtype=np.float32)
            out[method][f"{scalar}_over_trials"] = stats(vals)
    return out


def leave_one_out(data, args):
    n = len(data["ref_norm"])
    rows = []
    if n < 2:
        return {}
    for i in range(n):
        train = np.array([j for j in range(n) if j != i], dtype=np.int64)
        test = np.array([i], dtype=np.int64)
        if len(train) < 3:
            continue
        rows.append(evaluate_split(train, test, data, args))
    return aggregate_method_rows(rows) if rows else {}


def sampled_trials(data, sample_counts, trials, rng, args):
    n = len(data["ref_norm"])
    out = {}
    for count in sample_counts:
        if n <= count or count < 3:
            continue
        rows = []
        for _ in range(trials):
            train = rng.choice(n, size=count, replace=False)
            mask = np.ones(n, dtype=bool)
            mask[train] = False
            test = np.flatnonzero(mask)
            rows.append(evaluate_split(train, test, data, args))
        out[str(count)] = aggregate_method_rows(rows)
        out[str(count)]["trials"] = int(trials)
        out[str(count)]["train_count"] = int(count)
        out[str(count)]["heldout_count_mean"] = float(n - count)
    return out


def build_data(summary, anchors, ref_info, cur_info):
    ref_idx = np.array([a["ref_patch_idx"] for a in anchors], dtype=np.int64)
    cur_idx = np.array([a["cur_patch_idx"] for a in anchors], dtype=np.int64)
    ref_norm = np.array([patch_idx_to_center_norm(i, ref_info["grid_hw"]) for i in ref_idx], dtype=np.float32)
    cur_norm = np.array([patch_idx_to_center_norm(i, cur_info["grid_hw"]) for i in cur_idx], dtype=np.float32)
    cosine = np.array([a.get("encoder_cosine", float(ref_info["enc"][r] @ cur_info["enc"][c])) for a, r, c in zip(anchors, ref_idx, cur_idx)], dtype=np.float32)
    mesh_err = np.array([a.get("mesh_error_px", 24.0) for a in anchors], dtype=np.float32)
    weights = np.clip(cosine, 0.05, 1.0) * np.exp(-np.clip(mesh_err, 0.0, 96.0) / 48.0)
    return {
        "ref_idx": ref_idx,
        "cur_idx": cur_idx,
        "ref_norm": ref_norm,
        "cur_norm": cur_norm,
        "ref_feat": ref_info["enc"],
        "cur_feat": cur_info["enc"],
        "anchor_ref_feat": ref_info["enc"][ref_idx],
        "anchor_cur_feat": cur_info["enc"][cur_idx],
        "cosine": cosine,
        "mesh_err": mesh_err,
        "weights": weights.astype(np.float32),
        "ref_grid_hw": ref_info["grid_hw"],
        "cur_grid_hw": cur_info["grid_hw"],
        "patch_size": ref_info["patch_size"],
        "mesh_visible_overlap": summary.get("mesh_visible_overlap", {}),
    }


def build_anchor_tokens_npz(path, data, summary):
    ref_norm = data["ref_norm"]
    cur_norm = data["cur_norm"]
    delta = cur_norm - ref_norm
    np.savez_compressed(
        path,
        ref_patch_idx=data["ref_idx"].astype(np.int32),
        cur_patch_idx=data["cur_idx"].astype(np.int32),
        ref_pos_norm=ref_norm.astype(np.float32),
        cur_pos_norm=cur_norm.astype(np.float32),
        delta_uv_norm=delta.astype(np.float32),
        key_cur_feature=data["anchor_cur_feat"].astype(np.float32),
        value_ref_feature=data["anchor_ref_feat"].astype(np.float32),
        confidence=data["weights"].astype(np.float32),
        encoder_cosine=data["cosine"].astype(np.float32),
        mesh_error_px=data["mesh_err"].astype(np.float32),
        ref_grid_hw=np.array(data["ref_grid_hw"], dtype=np.int32),
        cur_grid_hw=np.array(data["cur_grid_hw"], dtype=np.int32),
        mesh_geometry_inliers=np.array([summary.get("mesh_geometry_inliers", 0)], dtype=np.int32),
    )


def draw_method_chart(path, metrics):
    methods = ["same_position", "translation", "affine", "anchor_token_top1", "anchor_token_soft", "anchor_token_affine_residual", "oracle_anchor"]
    labels = ["same", "trans", "affine", "tok-top1", "tok-soft", "tok-resid", "oracle"]
    values = []
    cosines = []
    for method in methods:
        m = metrics.get(method, {})
        values.append(m.get("patch_error_median_over_trials", {}).get("median"))
        cosines.append(m.get("token_cosine_median_over_trials", {}).get("median"))
    valid_vals = [v for v in values if v is not None]
    max_val = max(valid_vals + [1.0])
    width, height = 1220, 450
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    cv2.putText(canvas, "Held-out reference lookup using structured AnchorTokens", (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (20, 20, 20), 2, cv2.LINE_AA)
    left, top, plot_w, plot_h = 72, 78, 1080, 260
    cv2.rectangle(canvas, (left, top), (left + plot_w, top + plot_h), (220, 220, 220), 1)
    group_w = plot_w / len(methods)
    for i, (label, err, cos) in enumerate(zip(labels, values, cosines)):
        x0 = left + int(i * group_w + group_w * 0.22)
        bar_w = max(12, int(group_w * 0.18))
        if err is not None:
            h_err = int(err / max_val * plot_h)
            cv2.rectangle(canvas, (x0, top + plot_h - h_err), (x0 + bar_w, top + plot_h), (230, 80, 80), -1)
            cv2.putText(canvas, f"{err:.2f}", (x0 - 6, max(top + 16, top + plot_h - h_err - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (30, 30, 30), 1, cv2.LINE_AA)
        if cos is not None:
            h_cos = int(np.clip((cos + 1.0) / 2.0, 0.0, 1.0) * plot_h)
            cv2.rectangle(canvas, (x0 + bar_w + 8, top + plot_h - h_cos), (x0 + 2 * bar_w + 8, top + plot_h), (80, 150, 230), -1)
            cv2.putText(canvas, f"{cos:.2f}", (x0 + bar_w, max(top + 16, top + plot_h - h_cos - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (30, 30, 30), 1, cv2.LINE_AA)
        cv2.putText(canvas, label, (x0 - 18, height - 64), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(canvas, "red=median patch error lower is better, blue=median token cosine higher is better", (left, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (50, 50, 50), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def draw_anchor_token_overlay(path, ref_rgb, cur_rgb, anchors, data, pred_residual, max_draw):
    patch_size = data["patch_size"]
    ref_grid = data["ref_grid_hw"]
    cur_grid = data["cur_grid_hw"]
    ref = draw_patch_grid(ref_rgb, patch_size)
    cur = draw_patch_grid(cur_rgb, patch_size)
    h = max(ref.shape[0], cur.shape[0])
    canvas = np.zeros((h + 82, ref.shape[1] + cur.shape[1], 3), dtype=np.uint8)
    canvas[:82] = 20
    canvas[82 : 82 + ref.shape[0], : ref.shape[1]] = ref
    canvas[82 : 82 + cur.shape[0], ref.shape[1] : ref.shape[1] + cur.shape[1]] = cur
    cv2.putText(canvas, "AnchorToken residual lookup: true ref=magenta, predicted ref=orange, current=white", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "shown with leave-one-out style token prediction", (12, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 1, cv2.LINE_AA)
    ids = np.arange(len(anchors))
    # **========== 原始代码：最多按 max_draw 画所有跨图连线 ==========**
    # if len(ids) > max_draw:
    #     ids = np.linspace(0, len(ids) - 1, max_draw).round().astype(np.int64)
    # **========== 新代码：总览图只画少量稀疏样本，避免线段过密 ==========**
    sparse_limit = min(max_draw, 24)
    if len(ids) > sparse_limit:
        ids = np.linspace(0, len(ids) - 1, sparse_limit).round().astype(np.int64)
    # **========== 结束 ==========**
    pred_idx = nearest_patch_indices(pred_residual, ref_grid)
    for idx in ids:
        true_ref = patch_idx_to_center_px(data["ref_idx"][idx], ref_grid, patch_size)
        pred_ref = patch_idx_to_center_px(pred_idx[idx], ref_grid, patch_size)
        cur = patch_idx_to_center_px(data["cur_idx"][idx], cur_grid, patch_size)
        true_pt = (int(round(true_ref[0])), int(round(82 + true_ref[1])))
        pred_pt = (int(round(pred_ref[0])), int(round(82 + pred_ref[1])))
        cur_pt = (int(round(ref_rgb.shape[1] + cur[0])), int(round(82 + cur[1])))
        # **========== 原始代码：较大的点 ==========**
        # cv2.circle(canvas, cur_pt, 3, (255, 255, 255), -1, cv2.LINE_AA)
        # cv2.circle(canvas, true_pt, 5, (255, 0, 255), -1, cv2.LINE_AA)
        # cv2.circle(canvas, pred_pt, 4, (40, 170, 255), -1, cv2.LINE_AA)
        # **========== 新代码：更小的点 ==========**
        cv2.circle(canvas, cur_pt, 2, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, true_pt, 3, (255, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, pred_pt, 3, (255, 150, 0), -1, cv2.LINE_AA)
        # **========== 结束 ==========**
        cv2.line(canvas, cur_pt, true_pt, (255, 0, 255), 1, cv2.LINE_AA)
        cv2.line(canvas, cur_pt, pred_pt, (255, 150, 0), 1, cv2.LINE_AA)
    save_rgb(path, canvas)


def token_prediction_errors(data, pred_residual):
    return patch_error(pred_residual, data["ref_norm"], data["ref_grid_hw"])


def error_color_rgb(err):
    if err <= 1.0:
        return (70, 220, 90)
    if err <= 2.0:
        return (255, 200, 40)
    return (255, 70, 70)


def draw_anchor_token_ref_error_overlay(path, ref_rgb, data, pred_residual, max_draw):
    patch_size = data["patch_size"]
    ref_grid = data["ref_grid_hw"]
    ref = draw_patch_grid(ref_rgb, patch_size)
    header_h = 96
    canvas = np.zeros((header_h + ref.shape[0], ref.shape[1], 3), dtype=np.uint8)
    canvas[:header_h] = 20
    canvas[header_h : header_h + ref.shape[0], : ref.shape[1]] = ref
    errors = token_prediction_errors(data, pred_residual)
    ids = np.arange(len(errors))
    if len(ids) > max_draw:
        ids = np.linspace(0, len(ids) - 1, max_draw).round().astype(np.int64)

    ref_wh = np.array([ref_grid[1] * patch_size, ref_grid[0] * patch_size], dtype=np.float32)
    pred_px_all = np.clip(pred_residual * ref_wh[None], [0.0, 0.0], ref_wh[None] - 1.0)
    for idx in ids:
        gt_ref = patch_idx_to_center_px(data["ref_idx"][idx], ref_grid, patch_size)
        pred_ref = pred_px_all[idx]
        gt_pt = (int(round(gt_ref[0])), int(round(header_h + gt_ref[1])))
        pred_pt = (int(round(pred_ref[0])), int(round(header_h + pred_ref[1])))
        color = error_color_rgb(float(errors[idx]))
        cv2.line(canvas, pred_pt, gt_pt, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, pred_pt, 2, (255, 150, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas, gt_pt, 2, (255, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, gt_pt, 4, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(canvas, "AnchorToken prediction on reference image", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"magenta=GT ref patch, orange=predicted ref position, line=prediction error | median={np.median(errors):.2f} patches", (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, "line color: green<=1 patch, yellow<=2, red>2", (12, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (190, 190, 190), 1, cv2.LINE_AA)
    save_rgb(path, canvas)


def draw_anchor_token_cur_query_error_overlay(path, cur_rgb, data, pred_residual, max_draw):
    patch_size = data["patch_size"]
    cur_grid = data["cur_grid_hw"]
    cur = draw_patch_grid(cur_rgb, patch_size)
    header_h = 96
    canvas = np.zeros((header_h + cur.shape[0], cur.shape[1], 3), dtype=np.uint8)
    canvas[:header_h] = 20
    canvas[header_h : header_h + cur.shape[0], : cur.shape[1]] = cur
    errors = token_prediction_errors(data, pred_residual)
    ids = np.arange(len(errors))
    if len(ids) > max_draw:
        ids = np.linspace(0, len(ids) - 1, max_draw).round().astype(np.int64)
    good = int((errors <= 1.0).sum())
    mid = int(((errors > 1.0) & (errors <= 2.0)).sum())
    bad = int((errors > 2.0).sum())
    for idx in ids:
        cur_pt_px = patch_idx_to_center_px(data["cur_idx"][idx], cur_grid, patch_size)
        cur_pt = (int(round(cur_pt_px[0])), int(round(header_h + cur_pt_px[1])))
        color = error_color_rgb(float(errors[idx]))
        cv2.circle(canvas, cur_pt, 3, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, cur_pt, 5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(canvas, "Current-frame AnchorToken queries", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"each dot is a held-out cur patch query; color is ref prediction error | green={good}, yellow={mid}, red={bad}", (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.49, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, "this separates where the query came from from where it predicts in the ref image", (12, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (190, 190, 190), 1, cv2.LINE_AA)
    save_rgb(path, canvas)


def select_example_indices(errors, max_examples=8):
    n = len(errors)
    if n <= max_examples:
        return np.arange(n, dtype=np.int64)
    order = np.argsort(errors)
    chosen = []
    for idx in order[:2]:
        chosen.append(int(idx))
    median_center = n // 2
    for idx in order[max(0, median_center - 1) : min(n, median_center + 2)]:
        chosen.append(int(idx))
    for idx in order[-3:]:
        chosen.append(int(idx))
    out = []
    for idx in chosen:
        if idx not in out:
            out.append(idx)
    return np.array(out[:max_examples], dtype=np.int64)


def draw_anchor_token_lookup_examples(path, ref_rgb, cur_rgb, data, pred_residual, max_examples=8):
    patch_size = data["patch_size"]
    ref_grid = data["ref_grid_hw"]
    cur_grid = data["cur_grid_hw"]
    errors = token_prediction_errors(data, pred_residual)
    ids = select_example_indices(errors, max_examples=max_examples)
    ref_grid_img = draw_patch_grid(ref_rgb, patch_size)
    cur_grid_img = draw_patch_grid(cur_rgb, patch_size)
    scale = min(1.0, 720.0 / float(ref_grid_img.shape[1] + cur_grid_img.shape[1]))
    ref_small = cv2.resize(ref_grid_img, (int(round(ref_grid_img.shape[1] * scale)), int(round(ref_grid_img.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    cur_small = cv2.resize(cur_grid_img, (int(round(cur_grid_img.shape[1] * scale)), int(round(cur_grid_img.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    row_h = max(ref_small.shape[0], cur_small.shape[0]) + 56
    width = ref_small.shape[1] + cur_small.shape[1] + 24
    header_h = 88
    canvas = np.zeros((header_h + row_h * len(ids), width, 3), dtype=np.uint8)
    canvas[:header_h] = 20
    cv2.putText(canvas, "Representative leave-one-out AnchorToken lookups", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "left=ref GT/pred, right=cur query; one row per anchor, avoiding dense cross-image lines", (12, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1, cv2.LINE_AA)

    ref_wh = np.array([ref_grid[1] * patch_size, ref_grid[0] * patch_size], dtype=np.float32)
    pred_px_all = np.clip(pred_residual * ref_wh[None], [0.0, 0.0], ref_wh[None] - 1.0)
    for row, idx in enumerate(ids):
        y0 = header_h + row * row_h
        canvas[y0 : y0 + row_h] = 245
        canvas[y0 + 44 : y0 + 44 + ref_small.shape[0], : ref_small.shape[1]] = ref_small
        x_cur = ref_small.shape[1] + 24
        canvas[y0 + 44 : y0 + 44 + cur_small.shape[0], x_cur : x_cur + cur_small.shape[1]] = cur_small

        gt_ref = patch_idx_to_center_px(data["ref_idx"][idx], ref_grid, patch_size) * scale
        pred_ref = pred_px_all[idx] * scale
        cur_pt_px = patch_idx_to_center_px(data["cur_idx"][idx], cur_grid, patch_size) * scale
        gt_pt = (int(round(gt_ref[0])), int(round(y0 + 44 + gt_ref[1])))
        pred_pt = (int(round(pred_ref[0])), int(round(y0 + 44 + pred_ref[1])))
        cur_pt = (int(round(x_cur + cur_pt_px[0])), int(round(y0 + 44 + cur_pt_px[1])))
        color = error_color_rgb(float(errors[idx]))
        cv2.line(canvas, pred_pt, gt_pt, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, gt_pt, 3, (255, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, pred_pt, 3, (255, 150, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas, cur_pt, 3, color, -1, cv2.LINE_AA)
        cv2.putText(canvas, f"#{int(idx)} err={errors[idx]:.2f} patch", (12, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(canvas, "ref: magenta GT, orange pred", (180, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (40, 40, 40), 1, cv2.LINE_AA)
        cv2.putText(canvas, "cur query", (x_cur + 8, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (40, 40, 40), 1, cv2.LINE_AA)
    save_rgb(path, canvas)


def make_leave_one_out_predictions(data, args):
    n = len(data["ref_norm"])
    pred = np.zeros_like(data["ref_norm"])
    if n < 4:
        pred[:] = data["ref_norm"]
        return pred
    for i in range(n):
        train = np.array([j for j in range(n) if j != i], dtype=np.int64)
        test = np.array([i], dtype=np.int64)
        translation, _, inv_affine = fit_models(train, data["ref_norm"], data["cur_norm"], data["weights"])
        _, _, token_residual = predict_with_anchor_tokens(test, train, data, translation, inv_affine, args)
        pred[i] = token_residual[0]
    return pred


def make_leave_one_out_comparison_predictions(data, args):
    n = len(data["ref_norm"])
    pred_affine = data["ref_norm"].copy()
    pred_residual = data["ref_norm"].copy()
    valid = np.zeros(n, dtype=bool)
    for i in range(n):
        train = np.array([j for j in range(n) if j != i], dtype=np.int64)
        test = np.array([i], dtype=np.int64)
        if len(train) < 3:
            continue
        translation, _, inv_affine = fit_models(train, data["ref_norm"], data["cur_norm"], data["weights"])
        cur_test = data["cur_norm"][test]
        if inv_affine is None:
            affine_pred = cur_test - translation[None]
        else:
            affine_pred = apply_affine(cur_test, inv_affine)
        _, _, residual_pred = predict_with_anchor_tokens(test, train, data, translation, inv_affine, args)
        pred_affine[i] = affine_pred[0]
        pred_residual[i] = residual_pred[0]
        valid[i] = True
    return pred_affine, pred_residual, valid


def affine_residual_metrics(data, pred_affine, pred_residual, valid):
    gt = data["ref_norm"]
    valid = np.asarray(valid, dtype=bool)
    affine_err = patch_error(pred_affine[valid], gt[valid], data["ref_grid_hw"]) if valid.any() else np.asarray([], dtype=np.float32)
    residual_err = patch_error(pred_residual[valid], gt[valid], data["ref_grid_hw"]) if valid.any() else np.asarray([], dtype=np.float32)
    improvement = affine_err - residual_err
    return {
        "valid_count": int(valid.sum()),
        "affine_patch_error": stats(affine_err),
        "affine_residual_patch_error": stats(residual_err),
        "improvement_patch_error": stats(improvement),
        "improved_count": int((improvement > 0.0).sum()) if len(improvement) else 0,
        "improved_rate": float((improvement > 0.0).mean()) if len(improvement) else None,
        "within_1_patch_affine": float((affine_err <= 1.0).mean()) if len(affine_err) else None,
        "within_1_patch_residual": float((residual_err <= 1.0).mean()) if len(residual_err) else None,
        "within_2_patch_affine": float((affine_err <= 2.0).mean()) if len(affine_err) else None,
        "within_2_patch_residual": float((residual_err <= 2.0).mean()) if len(residual_err) else None,
    }


def draw_affine_vs_residual_overlay(path, ref_rgb, data, pred_affine, pred_residual, valid, max_draw):
    patch_size = data["patch_size"]
    ref_grid = data["ref_grid_hw"]
    ref = draw_patch_grid(ref_rgb, patch_size)
    header_h = 112
    canvas = np.zeros((header_h + ref.shape[0], ref.shape[1], 3), dtype=np.uint8)
    canvas[:header_h] = 20
    canvas[header_h : header_h + ref.shape[0], : ref.shape[1]] = ref

    ids = np.flatnonzero(valid)
    if len(ids) > max_draw:
        ids = np.linspace(0, len(ids) - 1, max_draw).round().astype(np.int64)
        ids = np.flatnonzero(valid)[ids]

    ref_wh = np.array([ref_grid[1] * patch_size, ref_grid[0] * patch_size], dtype=np.float32)
    affine_px = np.clip(pred_affine * ref_wh[None], [0.0, 0.0], ref_wh[None] - 1.0)
    residual_px = np.clip(pred_residual * ref_wh[None], [0.0, 0.0], ref_wh[None] - 1.0)
    affine_err = patch_error(pred_affine[valid], data["ref_norm"][valid], ref_grid) if valid.any() else np.asarray([], dtype=np.float32)
    residual_err = patch_error(pred_residual[valid], data["ref_norm"][valid], ref_grid) if valid.any() else np.asarray([], dtype=np.float32)

    for idx in ids:
        gt_ref = patch_idx_to_center_px(data["ref_idx"][idx], ref_grid, patch_size)
        gt_pt = (int(round(gt_ref[0])), int(round(header_h + gt_ref[1])))
        aff_pt = (int(round(affine_px[idx, 0])), int(round(header_h + affine_px[idx, 1])))
        res_pt = (int(round(residual_px[idx, 0])), int(round(header_h + residual_px[idx, 1])))
        cv2.line(canvas, aff_pt, gt_pt, (70, 150, 255), 1, cv2.LINE_AA)
        cv2.line(canvas, res_pt, gt_pt, (255, 150, 0), 1, cv2.LINE_AA)
        cv2.circle(canvas, gt_pt, 2, (255, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, gt_pt, 4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(canvas, aff_pt, 2, (70, 150, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, res_pt, 2, (255, 150, 0), -1, cv2.LINE_AA)

    aff_med = float(np.median(affine_err)) if len(affine_err) else float("nan")
    res_med = float(np.median(residual_err)) if len(residual_err) else float("nan")
    gain = aff_med - res_med if np.isfinite(aff_med) and np.isfinite(res_med) else float("nan")
    cv2.putText(canvas, "Affine-only vs affine + local AnchorToken residual", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "magenta=GT ref, blue=affine-only pred, orange=affine+residual pred", (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"median patch error: affine={aff_med:.2f}, affine+residual={res_med:.2f}, gain={gain:.2f}", (12, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, "shorter error line means better held-out anchor correction", (12, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (190, 190, 190), 1, cv2.LINE_AA)
    save_rgb(path, canvas)


def draw_affine_residual_table(path, metrics):
    # **========== 原始代码：OpenCV Hershey 字体表格 ==========**
    # width, height = 1120, 440
    # canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    # cv2.putText(canvas, "Step3 held-out correction: affine-only vs affine + local residual", (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (20, 20, 20), 2, cv2.LINE_AA)
    # rows = [
    #     ("valid held-out anchors", str(metrics["valid_count"]), "-", "-"),
    #     ("median patch error", fmt_metric(metrics["affine_patch_error"], "median"), fmt_metric(metrics["affine_residual_patch_error"], "median"), fmt_metric(metrics["improvement_patch_error"], "median")),
    #     ("mean patch error", fmt_metric(metrics["affine_patch_error"], "mean"), fmt_metric(metrics["affine_residual_patch_error"], "mean"), fmt_metric(metrics["improvement_patch_error"], "mean")),
    #     ("p75 patch error", fmt_metric(metrics["affine_patch_error"], "p75"), fmt_metric(metrics["affine_residual_patch_error"], "p75"), fmt_metric(metrics["improvement_patch_error"], "p75")),
    #     ("within 1 patch", fmt_rate(metrics["within_1_patch_affine"]), fmt_rate(metrics["within_1_patch_residual"]), fmt_delta_rate(metrics["within_1_patch_residual"], metrics["within_1_patch_affine"])),
    #     ("within 2 patches", fmt_rate(metrics["within_2_patch_affine"]), fmt_rate(metrics["within_2_patch_residual"]), fmt_delta_rate(metrics["within_2_patch_residual"], metrics["within_2_patch_affine"])),
    #     ("anchors improved", "-", "-", f"{metrics['improved_count']} ({fmt_rate(metrics['improved_rate'])})"),
    # ]
    # xs = [34, 420, 650, 850]
    # headers = ["metric", "affine-only", "affine + residual", "residual gain"]
    # y = 92
    # for x, text in zip(xs, headers):
    #     cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (50, 50, 50), 2, cv2.LINE_AA)
    # cv2.line(canvas, (24, y + 18), (width - 24, y + 18), (180, 180, 180), 1, cv2.LINE_AA)
    # for i, row in enumerate(rows):
    #     y = 136 + i * 42
    #     bg = (235, 235, 235) if i % 2 else (248, 248, 248)
    #     cv2.rectangle(canvas, (24, y - 26), (width - 24, y + 12), bg, -1)
    #     for x, text in zip(xs, row):
    #         color = (30, 120, 40) if x == xs[-1] and text not in {"-", "None"} and not text.startswith("-") else (20, 20, 20)
    #         cv2.putText(canvas, str(text), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, color, 1, cv2.LINE_AA)
    # cv2.putText(canvas, "positive gain means local residual improved over affine-only", (34, height - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (70, 70, 70), 1, cv2.LINE_AA)
    # cv2.imwrite(str(path), canvas)
    # **========== 新代码：PIL/DejaVu 字体的汇报版表格 ==========**
    rows = [
        ("valid held-out anchors", str(metrics["valid_count"]), "-", "-"),
        ("median patch error", fmt_metric(metrics["affine_patch_error"], "median"), fmt_metric(metrics["affine_residual_patch_error"], "median"), fmt_metric(metrics["improvement_patch_error"], "median")),
        ("mean patch error", fmt_metric(metrics["affine_patch_error"], "mean"), fmt_metric(metrics["affine_residual_patch_error"], "mean"), fmt_metric(metrics["improvement_patch_error"], "mean")),
        ("p75 patch error", fmt_metric(metrics["affine_patch_error"], "p75"), fmt_metric(metrics["affine_residual_patch_error"], "p75"), fmt_metric(metrics["improvement_patch_error"], "p75")),
        ("within 1 patch", fmt_rate(metrics["within_1_patch_affine"]), fmt_rate(metrics["within_1_patch_residual"]), fmt_delta_rate(metrics["within_1_patch_residual"], metrics["within_1_patch_affine"])),
        ("within 2 patches", fmt_rate(metrics["within_2_patch_affine"]), fmt_rate(metrics["within_2_patch_residual"]), fmt_delta_rate(metrics["within_2_patch_residual"], metrics["within_2_patch_affine"])),
        ("anchors improved", "-", "-", f"{metrics['improved_count']} ({fmt_rate(metrics['improved_rate'])})"),
    ]

    width, height = 1280, 540
    margin = 48
    title_y = 38
    table_x = margin
    table_y = 132
    row_h = 46
    header_h = 52
    col_w = [410, 230, 270, 240]
    headers = ["Metric", "Affine-only", "Affine + residual", "Residual gain"]
    font_dir = Path("/usr/share/fonts/truetype/dejavu")
    title_font = ImageFont.truetype(str(font_dir / "DejaVuSans-Bold.ttf"), 31)
    subtitle_font = ImageFont.truetype(str(font_dir / "DejaVuSans.ttf"), 16)
    header_font = ImageFont.truetype(str(font_dir / "DejaVuSans-Bold.ttf"), 17)
    body_font = ImageFont.truetype(str(font_dir / "DejaVuSans.ttf"), 17)
    bold_font = ImageFont.truetype(str(font_dir / "DejaVuSans-Bold.ttf"), 17)

    img = Image.new("RGB", (width, height), (247, 249, 252))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((24, 24, width - 24, height - 24), radius=24, fill=(255, 255, 255), outline=(224, 229, 236), width=1)
    draw.text((margin, title_y), "Step3: Affine-Only vs Affine + Local Residual", font=title_font, fill=(20, 28, 38))
    draw.text((margin, title_y + 42), "Held-out anchors: lower patch error is better; positive gain means local residual improved over affine-only.", font=subtitle_font, fill=(91, 102, 118))

    table_w = sum(col_w)
    draw.rounded_rectangle((table_x, table_y, table_x + table_w, table_y + header_h + row_h * len(rows)), radius=16, fill=(255, 255, 255), outline=(220, 226, 235), width=1)
    draw.rounded_rectangle((table_x, table_y, table_x + table_w, table_y + header_h), radius=16, fill=(238, 243, 250))
    draw.rectangle((table_x, table_y + header_h - 16, table_x + table_w, table_y + header_h), fill=(238, 243, 250))

    x = table_x
    for text, w in zip(headers, col_w):
        draw.text((x + 18, table_y + 16), text, font=header_font, fill=(38, 48, 62))
        x += w
    for i in range(1, len(col_w)):
        x_sep = table_x + sum(col_w[:i])
        draw.line((x_sep, table_y + 10, x_sep, table_y + header_h + row_h * len(rows) - 10), fill=(230, 234, 240), width=1)

    for i, row in enumerate(rows):
        y0 = table_y + header_h + i * row_h
        bg = (250, 252, 255) if i % 2 == 0 else (255, 255, 255)
        draw.rectangle((table_x + 1, y0, table_x + table_w - 1, y0 + row_h), fill=bg)
        draw.line((table_x + 12, y0, table_x + table_w - 12, y0), fill=(235, 239, 245), width=1)
        x = table_x
        for j, (text, w) in enumerate(zip(row, col_w)):
            text = str(text)
            if j == 0:
                font = bold_font if i in {1, 4, 5} else body_font
                color = (35, 45, 58)
            elif j == 3 and text not in {"-", "None"}:
                font = bold_font
                color = (30, 132, 73) if not text.startswith("-") else (190, 75, 65)
            else:
                font = body_font
                color = (35, 45, 58)
            draw.text((x + 18, y0 + 13), text, font=font, fill=color)
            x += w

    legend_y = table_y + header_h + row_h * len(rows) + 34
    draw.rounded_rectangle((margin, legend_y - 12, margin + 440, legend_y + 36), radius=12, fill=(246, 249, 255), outline=(225, 232, 242), width=1)
    draw.ellipse((margin + 18, legend_y + 2, margin + 30, legend_y + 14), fill=(70, 150, 255))
    draw.text((margin + 40, legend_y - 1), "blue = affine-only", font=subtitle_font, fill=(70, 82, 98))
    draw.ellipse((margin + 202, legend_y + 2, margin + 214, legend_y + 14), fill=(255, 150, 0))
    draw.text((margin + 224, legend_y - 1), "orange = affine + local residual", font=subtitle_font, fill=(70, 82, 98))

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    # **========== 结束 ==========**


def fmt_metric(metric, key):
    if not metric or metric.get(key) is None:
        return "None"
    return f"{float(metric[key]):.2f}"


def fmt_rate(value):
    if value is None:
        return "None"
    return f"{100.0 * float(value):.1f}%"


def fmt_delta_rate(new, old):
    if new is None or old is None:
        return "None"
    return f"{100.0 * (float(new) - float(old)):+.1f} pp"


def analyze_one(aabb_dir, out_dir, model, args, rng, sample_counts):
    aabb_dir, boundary_dir, summary = load_boundary_summary(aabb_dir)
    anchors = summary.get("anchors", [])
    if len(anchors) < 3:
        raise RuntimeError(f"need at least 3 anchors for {aabb_dir}, got {len(anchors)}")
    ref_info = encode_image(model, summary["ref"]["image"], args.size, args.device)
    cur_info = encode_image(model, summary["cur"]["image"], args.size, args.device)
    data = build_data(summary, anchors, ref_info, cur_info)

    loo = leave_one_out(data, args)
    sampled = sampled_trials(data, sample_counts, args.trials, rng, args)
    full_train = np.arange(len(anchors), dtype=np.int64)
    full_eval = evaluate_split(full_train, full_train, data, args)
    loo_pred = make_leave_one_out_predictions(data, args)
    affine_pred, residual_pred, comparison_valid = make_leave_one_out_comparison_predictions(data, args)
    comparison_metrics = affine_residual_metrics(data, affine_pred, residual_pred, comparison_valid)

    sample_out = out_dir / aabb_dir.name
    sample_out.mkdir(parents=True, exist_ok=True)
    build_anchor_tokens_npz(sample_out / "anchor_tokens_structured.npz", data, summary)
    draw_method_chart(sample_out / "anchor_token_leave_one_out_chart.jpg", loo)
    draw_affine_vs_residual_overlay(sample_out / "anchor_token_affine_vs_residual_overlay.jpg", ref_info["crop_rgb"], data, affine_pred, residual_pred, comparison_valid, args.max_draw)
    draw_affine_residual_table(sample_out / "anchor_token_affine_residual_table.jpg", comparison_metrics)
    draw_anchor_token_overlay(sample_out / "anchor_token_lookup_overlay.jpg", ref_info["crop_rgb"], cur_info["crop_rgb"], anchors, data, loo_pred, args.max_draw)
    draw_anchor_token_ref_error_overlay(sample_out / "anchor_token_ref_error_overlay.jpg", ref_info["crop_rgb"], data, loo_pred, args.max_draw)
    draw_anchor_token_cur_query_error_overlay(sample_out / "anchor_token_cur_query_error_overlay.jpg", cur_info["crop_rgb"], data, loo_pred, args.max_draw)
    draw_anchor_token_lookup_examples(sample_out / "anchor_token_lookup_examples.jpg", ref_info["crop_rgb"], cur_info["crop_rgb"], data, loo_pred)

    result = {
        "aabb_dir": str(aabb_dir),
        "boundary_dir": str(boundary_dir),
        "ref": summary["ref"],
        "cur": summary["cur"],
        "n_anchor_tokens": int(len(anchors)),
        "mesh_geometry_inliers": int(summary.get("mesh_geometry_inliers", 0)),
        "unique_anchor_patch_pairs": int(summary.get("unique_anchor_patch_pairs", len(anchors))),
        "ref_grid_hw": list(data["ref_grid_hw"]),
        "cur_grid_hw": list(data["cur_grid_hw"]),
        "anchor_token_fields": [
            "key_cur_feature",
            "value_ref_feature",
            "ref_pos_norm",
            "cur_pos_norm",
            "delta_uv_norm",
            "confidence",
            "mesh_error_px",
            "encoder_cosine",
        ],
        "leave_one_out": loo,
        "sampled_trials": sampled,
        "full_self_lookup": full_eval,
        "affine_vs_residual": comparison_metrics,
        "visualizations": {
            "leave_one_out_chart": str(sample_out / "anchor_token_leave_one_out_chart.jpg"),
            "affine_vs_residual_overlay": str(sample_out / "anchor_token_affine_vs_residual_overlay.jpg"),
            "affine_residual_table": str(sample_out / "anchor_token_affine_residual_table.jpg"),
            "lookup_overlay": str(sample_out / "anchor_token_lookup_overlay.jpg"),
            "ref_error_overlay": str(sample_out / "anchor_token_ref_error_overlay.jpg"),
            "cur_query_error_overlay": str(sample_out / "anchor_token_cur_query_error_overlay.jpg"),
            "lookup_examples": str(sample_out / "anchor_token_lookup_examples.jpg"),
            "structured_tokens_npz": str(sample_out / "anchor_tokens_structured.npz"),
        },
    }
    (sample_out / "anchor_token_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def draw_root_summary(path, results):
    width, height = 1380, 130 + 86 * len(results)
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    cv2.putText(canvas, "Structured AnchorToken prototype validation", (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (20, 20, 20), 2, cv2.LINE_AA)
    headers = ["sample", "tokens", "same", "affine", "tok-soft", "tok-resid", "oracle", "resid cos"]
    xs = [18, 500, 610, 730, 850, 980, 1120, 1250]
    for x, text in zip(xs, headers):
        cv2.putText(canvas, text, (x, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (50, 50, 50), 2, cv2.LINE_AA)
    for i, r in enumerate(results):
        y = 130 + 86 * i
        loo = r["leave_one_out"]
        def med(method):
            return loo.get(method, {}).get("patch_error_median_over_trials", {}).get("median")
        def cos(method):
            return loo.get(method, {}).get("token_cosine_median_over_trials", {}).get("median")
        vals = [
            str(r["n_anchor_tokens"]),
            f"{med('same_position'):.2f}" if med("same_position") is not None else "None",
            f"{med('affine'):.2f}" if med("affine") is not None else "None",
            f"{med('anchor_token_soft'):.2f}" if med("anchor_token_soft") is not None else "None",
            f"{med('anchor_token_affine_residual'):.2f}" if med("anchor_token_affine_residual") is not None else "None",
            f"{med('oracle_anchor'):.2f}" if med("oracle_anchor") is not None else "None",
            f"{cos('anchor_token_affine_residual'):.2f}" if cos("anchor_token_affine_residual") is not None else "None",
        ]
        cv2.putText(canvas, Path(r["aabb_dir"]).name[:60], (xs[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (20, 20, 20), 2, cv2.LINE_AA)
        for x, val in zip(xs[1:], vals):
            cv2.putText(canvas, val, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    sample_counts = parse_sample_counts(args.sample_counts)
    rng = np.random.default_rng(args.seed)
    if args.out_dir is None:
        args.out_dir = str(REPO_ROOT / "output" / "rich_anchor_token_prototype")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading original Human3R encoder...")
    model = load_model(args.model_path, device=args.device, verbose=True).eval()
    model.gradient_checkpointing = False

    results = []
    for aabb_dir in args.aabb_dir:
        print(f"Prototyping AnchorTokens for {aabb_dir}...")
        results.append(analyze_one(Path(aabb_dir), out_dir, model, args, rng, sample_counts))

    draw_root_summary(out_dir / "anchor_token_prototype_summary.jpg", results)
    root = {
        "args": vars(args),
        "results": results,
        "visualizations": {
            "summary": str(out_dir / "anchor_token_prototype_summary.jpg"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(root, indent=2), encoding="utf-8")
    print(json.dumps(
        {
            "out_dir": str(out_dir),
            "samples": [
                {
                    "name": Path(r["aabb_dir"]).name,
                    "tokens": r["n_anchor_tokens"],
                    "same_error": r["leave_one_out"].get("same_position", {}).get("patch_error_median_over_trials", {}).get("median"),
                    "affine_error": r["leave_one_out"].get("affine", {}).get("patch_error_median_over_trials", {}).get("median"),
                    "token_soft_error": r["leave_one_out"].get("anchor_token_soft", {}).get("patch_error_median_over_trials", {}).get("median"),
                    "token_residual_error": r["leave_one_out"].get("anchor_token_affine_residual", {}).get("patch_error_median_over_trials", {}).get("median"),
                    "oracle_error": r["leave_one_out"].get("oracle_anchor", {}).get("patch_error_median_over_trials", {}).get("median"),
                    "token_residual_cosine": r["leave_one_out"].get("anchor_token_affine_residual", {}).get("token_cosine_median_over_trials", {}).get("median"),
                }
                for r in results
            ],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
