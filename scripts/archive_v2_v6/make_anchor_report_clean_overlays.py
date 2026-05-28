#!/usr/bin/env python3
"""Create presentation-friendly AnchorToken report overlays.

This script only redraws existing report artifacts. It does not rerun XFeat,
mesh verification, Human3R, or any model code.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PATCH_SIZE = 16

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (170, 170, 170)
CYAN = (0, 220, 255)
MAGENTA = (255, 0, 255)
RED = (255, 50, 50)
GREEN = (0, 230, 90)
ORANGE = (255, 165, 0)
YELLOW = (255, 255, 0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report_root", default="output/anchor_token_report_v1")
    parser.add_argument("--sample", default="BBQ_001_guitar_cam06_cam07_f00000244")
    parser.add_argument("--max_points", type=int, default=8)
    return parser.parse_args()


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def read_json(path):
    return json.loads(resolve_path(path).read_text(encoding="utf-8"))


def read_rgb(path):
    img = cv2.imread(str(resolve_path(path)), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_rgb(path, rgb):
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def put_text(img, text, xy, color=WHITE, scale=0.52, thickness=1):
    x, y = xy
    cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_marker(img, pt, color, label=None, radius=4, filled=True):
    pt = tuple(np.round(pt).astype(int).tolist())
    cv2.circle(img, pt, radius + 2, BLACK, -1, cv2.LINE_AA)
    cv2.circle(img, pt, radius, color, -1 if filled else 2, cv2.LINE_AA)
    if label is not None:
        put_text(img, str(label), (pt[0] + 7, pt[1] - 7), WHITE, 0.42, 1)


def draw_cross(img, pt, color):
    pt = tuple(np.round(pt).astype(int).tolist())
    cv2.drawMarker(img, pt, color, markerType=cv2.MARKER_TILTED_CROSS, markerSize=10, thickness=2, line_type=cv2.LINE_AA)


def draw_diamond(img, pt, color):
    pt = tuple(np.round(pt).astype(int).tolist())
    cv2.drawMarker(img, pt, color, markerType=cv2.MARKER_DIAMOND, markerSize=11, thickness=2, line_type=cv2.LINE_AA)


def weighted_translation(ref, cur, weights):
    return np.average(cur - ref, axis=0, weights=weights).astype(np.float32)


def fit_affine(ref, cur, weights, ridge=1e-4):
    if len(ref) < 3:
        return None
    x = np.concatenate([ref, np.ones((len(ref), 1), dtype=np.float32)], axis=1)
    w = np.sqrt(weights).reshape(-1, 1).astype(np.float32)
    lhs = (x * w).T @ (x * w) + ridge * np.eye(3, dtype=np.float32)
    rhs = (x * w).T @ (cur * w)
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


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-8)


def select_diverse(ref_px, cur_px, score, max_points):
    order = np.argsort(-score)
    selected = []
    min_dist = 42.0
    while len(selected) < max_points and min_dist >= 16.0:
        for idx in order:
            if idx in selected:
                continue
            if all(np.linalg.norm(cur_px[idx] - cur_px[j]) >= min_dist and np.linalg.norm(ref_px[idx] - ref_px[j]) >= min_dist for j in selected):
                selected.append(int(idx))
                if len(selected) >= max_points:
                    break
        min_dist *= 0.75
    for idx in order:
        if len(selected) >= max_points:
            break
        if idx not in selected:
            selected.append(int(idx))
    return np.array(selected[:max_points], dtype=np.int64)


def anchor_arrays(summary):
    anchors = summary["anchors"]
    ref_px = np.array([a["ref_patch_center_crop_xy"] for a in anchors], dtype=np.float32)
    cur_px = np.array([a["cur_patch_center_crop_xy"] for a in anchors], dtype=np.float32)
    ref_grid = np.array(summary["human3r_ref_grid_hw"], dtype=np.float32)
    cur_grid = np.array(summary["human3r_cur_grid_hw"], dtype=np.float32)
    ref_wh = np.array([ref_grid[1] * PATCH_SIZE, ref_grid[0] * PATCH_SIZE], dtype=np.float32)
    cur_wh = np.array([cur_grid[1] * PATCH_SIZE, cur_grid[0] * PATCH_SIZE], dtype=np.float32)
    ref_norm = ref_px / ref_wh[None]
    cur_norm = cur_px / cur_wh[None]
    cosine = np.array([a.get("encoder_cosine", 0.0) for a in anchors], dtype=np.float32)
    mesh_err = np.array([a.get("mesh_error_px", 24.0) for a in anchors], dtype=np.float32)
    weights = np.clip(cosine, 0.05, 1.0) * np.exp(-np.clip(mesh_err, 0.0, 96.0) / 48.0)
    return ref_px, cur_px, ref_norm, cur_norm, ref_wh, cur_wh, weights.astype(np.float32), cosine, mesh_err


def make_canvas(ref_img, cur_img, title, subtitle, legend_items):
    gap = 24
    header = 108
    footer = 36
    h = max(ref_img.shape[0], cur_img.shape[0])
    w = ref_img.shape[1] + gap + cur_img.shape[1]
    canvas = np.zeros((header + h + footer, w, 3), dtype=np.uint8)
    canvas[header : header + ref_img.shape[0], : ref_img.shape[1]] = ref_img
    canvas[header : header + cur_img.shape[0], ref_img.shape[1] + gap : ref_img.shape[1] + gap + cur_img.shape[1]] = cur_img
    put_text(canvas, title, (14, 30), WHITE, 0.7, 2)
    put_text(canvas, subtitle, (14, 62), GRAY, 0.52, 1)
    x = 14
    y = 92
    for color, text in legend_items:
        cv2.rectangle(canvas, (x, y - 10), (x + 16, y + 6), color, -1)
        put_text(canvas, text, (x + 24, y + 5), WHITE, 0.42, 1)
        x += 24 + len(text) * 9 + 28
    put_text(canvas, "LEFT: reference A@t+1", (14, header + 24), YELLOW, 0.52, 1)
    put_text(canvas, "RIGHT: current B@t+2", (ref_img.shape[1] + gap + 14, header + 24), YELLOW, 0.52, 1)
    return canvas, header, ref_img.shape[1] + gap


def draw_correction_overlay(report_root, sample, max_points):
    boundary_dir = report_root / "01_aabb_step1" / sample / "pair_01_A_t1_to_B_t2_BOUNDARY"
    summary = read_json(boundary_dir / "summary.json")
    ref_img = read_rgb(boundary_dir / "02_ref_human3r_crop_grid.jpg")
    cur_img = read_rgb(boundary_dir / "03_cur_human3r_crop_grid.jpg")

    ref_px, cur_px, ref_norm, cur_norm, ref_wh, cur_wh, weights, cosine, mesh_err = anchor_arrays(summary)
    delta = weighted_translation(ref_norm, cur_norm, weights)
    affine = fit_affine(ref_norm, cur_norm, weights)
    pred_zero = ref_norm * cur_wh[None]
    pred_aff = apply_affine(ref_norm, affine) * cur_wh[None] if affine is not None else pred_zero
    quality = weights + cosine - mesh_err / 96.0
    ids = select_diverse(ref_px, cur_px, quality, max_points)

    canvas, top, cur_x0 = make_canvas(
        ref_img,
        cur_img,
        "Clean correction overlay: where should a reference anchor land after the shot cut?",
        "same ID links source ref anchor to its current-view target; shorter line to magenta means better correction",
        [(CYAN, "ref source"), (MAGENTA, "true current"), (RED, "no correction"), (GREEN, "affine prediction")],
    )

    for n, idx in enumerate(ids, start=1):
        ref_pt = ref_px[idx] + np.array([0, top])
        true_pt = cur_px[idx] + np.array([cur_x0, top])
        zero_pt = pred_zero[idx] + np.array([cur_x0, top])
        aff_pt = pred_aff[idx] + np.array([cur_x0, top])
        cv2.line(canvas, tuple(np.round(zero_pt).astype(int)), tuple(np.round(true_pt).astype(int)), RED, 1, cv2.LINE_AA)
        cv2.line(canvas, tuple(np.round(aff_pt).astype(int)), tuple(np.round(true_pt).astype(int)), GREEN, 1, cv2.LINE_AA)
        draw_marker(canvas, ref_pt, CYAN, n, radius=4, filled=False)
        draw_marker(canvas, true_pt, MAGENTA, n, radius=4, filled=True)
        draw_cross(canvas, zero_pt, RED)
        draw_diamond(canvas, aff_pt, GREEN)

    out = report_root / "02_correction_proxy" / sample / "correction_prediction_overlay_clean.jpg"
    save_rgb(out, canvas)
    return out


def patch_center_px(norm, grid_hw):
    wh = np.array([grid_hw[1] * PATCH_SIZE, grid_hw[0] * PATCH_SIZE], dtype=np.float32)
    return np.asarray(norm, dtype=np.float32) * wh[None]


def make_token_predictions(npz):
    ref_norm = npz["ref_pos_norm"].astype(np.float32)
    cur_norm = npz["cur_pos_norm"].astype(np.float32)
    cur_feat = npz["key_cur_feature"].astype(np.float32)
    confidence = npz["confidence"].astype(np.float32)
    pred = np.zeros_like(ref_norm)
    n = len(ref_norm)
    for i in range(n):
        train = np.array([j for j in range(n) if j != i], dtype=np.int64)
        translation = weighted_translation(ref_norm[train], cur_norm[train], confidence[train])
        affine = fit_affine(ref_norm[train], cur_norm[train], confidence[train])
        inv_affine = invert_affine(affine)
        query_feat = cur_feat[[i]]
        token_feat = cur_feat[train]
        feat_score = query_feat @ token_feat.T
        spatial_d2 = ((cur_norm[[i], None, :] - cur_norm[train][None, :, :]) ** 2).sum(axis=-1)
        attn = softmax(2.0 * feat_score - 30.0 * spatial_d2)
        if inv_affine is None:
            base_test = cur_norm[[i]] - translation[None]
            base_train = cur_norm[train] - translation[None]
        else:
            base_test = apply_affine(cur_norm[[i]], inv_affine)
            base_train = apply_affine(cur_norm[train], inv_affine)
        residual_train = ref_norm[train] - base_train
        pred[i] = base_test[0] + (attn @ residual_train)[0]
    return pred


def draw_token_overlay(report_root, sample, max_points):
    boundary_dir = report_root / "01_aabb_step1" / sample / "pair_01_A_t1_to_B_t2_BOUNDARY"
    token_dir = report_root / "03_anchor_token_prototype" / sample
    ref_img = read_rgb(boundary_dir / "02_ref_human3r_crop_grid.jpg")
    cur_img = read_rgb(boundary_dir / "03_cur_human3r_crop_grid.jpg")
    npz = np.load(resolve_path(token_dir / "anchor_tokens_structured.npz"), allow_pickle=True)
    ref_norm = npz["ref_pos_norm"].astype(np.float32)
    cur_norm = npz["cur_pos_norm"].astype(np.float32)
    ref_grid = npz["ref_grid_hw"].astype(np.int32)
    cur_grid = npz["cur_grid_hw"].astype(np.int32)
    confidence = npz["confidence"].astype(np.float32)
    pred_norm = make_token_predictions(npz)
    ref_px = patch_center_px(ref_norm, ref_grid)
    cur_px = patch_center_px(cur_norm, cur_grid)
    pred_px = patch_center_px(pred_norm, ref_grid)
    error = np.linalg.norm((pred_norm - ref_norm) * np.array([ref_grid[1], ref_grid[0]], dtype=np.float32)[None], axis=1)
    score = confidence - 0.15 * error
    ids = select_diverse(ref_px, cur_px, score, max_points)

    canvas, top, cur_x0 = make_canvas(
        ref_img,
        cur_img,
        "Clean AnchorToken lookup: can current anchors find their reference patches?",
        "orange ring should sit near magenta dot; this shows local residual re-anchor prediction",
        [(WHITE, "current anchor"), (MAGENTA, "true ref"), (ORANGE, "AnchorToken prediction")],
    )

    for n, idx in enumerate(ids, start=1):
        cur_pt = cur_px[idx] + np.array([cur_x0, top])
        true_pt = ref_px[idx] + np.array([0, top])
        pred_pt = pred_px[idx] + np.array([0, top])
        cv2.line(canvas, tuple(np.round(cur_pt).astype(int)), tuple(np.round(true_pt).astype(int)), MAGENTA, 1, cv2.LINE_AA)
        cv2.line(canvas, tuple(np.round(cur_pt).astype(int)), tuple(np.round(pred_pt).astype(int)), ORANGE, 1, cv2.LINE_AA)
        draw_marker(canvas, cur_pt, WHITE, n, radius=4, filled=True)
        draw_marker(canvas, true_pt, MAGENTA, n, radius=4, filled=True)
        draw_marker(canvas, pred_pt, ORANGE, None, radius=7, filled=False)
        put_text(canvas, str(n), (int(round(pred_pt[0] + 9)), int(round(pred_pt[1] + 9))), ORANGE, 0.42, 1)

    out = token_dir / "anchor_token_lookup_overlay_clean.jpg"
    save_rgb(out, canvas)
    return out


def main():
    args = parse_args()
    report_root = resolve_path(args.report_root)
    correction = draw_correction_overlay(report_root, args.sample, args.max_points)
    token = draw_token_overlay(report_root, args.sample, args.max_points)
    print(correction)
    print(token)


if __name__ == "__main__":
    main()
