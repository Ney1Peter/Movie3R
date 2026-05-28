#!/usr/bin/env python3
"""Analyze whether AABB boundary anchors provide useful correction evidence.

Input is the output directory produced by verify_rich_aabb_anchor_step1.py.
This script does not rerun XFeat, mesh verification, Human3R, or any model code.
It only reads saved mesh-verified boundary anchors and asks:

    If we use a few anchors to estimate a simple 2D correction, can that
    correction predict held-out anchor positions better than no correction?

The experiment is intentionally a proxy. Translation / affine transforms are
not the final camera correction, but they test whether anchors carry a stable
shot-boundary re-alignment signal.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from report_image_style import patch_cv2_text

patch_cv2_text(cv2)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aabb_dir",
        action="append",
        required=True,
        help="AABB Step1 output directory. Can be provided multiple times.",
    )
    parser.add_argument("--sample_counts", default="4,8,16,32")
    parser.add_argument("--trials", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_draw", type=int, default=160)
    parser.add_argument("--out_dir", default=None)
    return parser.parse_args()


def parse_sample_counts(text):
    out = []
    for item in text.split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    return out


def load_boundary_summary(aabb_dir):
    aabb_dir = Path(aabb_dir)
    candidates = sorted(aabb_dir.glob("pair_*BOUNDARY/summary.json"))
    if not candidates:
        candidates = sorted(aabb_dir.glob("pair_01_*/summary.json"))
    if not candidates:
        raise RuntimeError(f"cannot find boundary pair summary under {aabb_dir}")
    summary = json.loads(candidates[0].read_text(encoding="utf-8"))
    return aabb_dir, candidates[0].parent, summary


def anchors_to_arrays(anchors):
    ref_px = np.array([a["ref_patch_center_crop_xy"] for a in anchors], dtype=np.float32)
    cur_px = np.array([a["cur_patch_center_crop_xy"] for a in anchors], dtype=np.float32)
    ref_wh = np.array([max(ref_px[:, 0].max() + 8.0, 1.0), max(ref_px[:, 1].max() + 8.0, 1.0)], dtype=np.float32)
    cur_wh = np.array([max(cur_px[:, 0].max() + 8.0, 1.0), max(cur_px[:, 1].max() + 8.0, 1.0)], dtype=np.float32)

    # Prefer exact crop sizes from patch grids when available.
    ref_grid = anchors[0].get("ref_grid_hw")
    cur_grid = anchors[0].get("cur_grid_hw")
    if ref_grid is not None:
        ref_wh = np.array([ref_grid[1] * 16.0, ref_grid[0] * 16.0], dtype=np.float32)
    if cur_grid is not None:
        cur_wh = np.array([cur_grid[1] * 16.0, cur_grid[0] * 16.0], dtype=np.float32)

    ref_norm = ref_px / ref_wh[None]
    cur_norm = cur_px / cur_wh[None]
    cosine = np.array([a.get("encoder_cosine", 0.0) for a in anchors], dtype=np.float32)
    mesh_err = np.array([a.get("mesh_error_px", 24.0) for a in anchors], dtype=np.float32)
    weights = np.clip(cosine, 0.05, 1.0) * np.exp(-np.clip(mesh_err, 0.0, 96.0) / 48.0)
    weights = np.clip(weights, 1e-4, None).astype(np.float32)
    return {
        "ref_px": ref_px,
        "cur_px": cur_px,
        "ref_wh": ref_wh,
        "cur_wh": cur_wh,
        "ref_norm": ref_norm,
        "cur_norm": cur_norm,
        "weights": weights,
        "cosine": cosine,
        "mesh_err": mesh_err,
    }


def weighted_translation(ref, cur, weights):
    delta = cur - ref
    return np.average(delta, axis=0, weights=weights).astype(np.float32)


def fit_affine(ref, cur, weights, ridge=1e-4):
    if len(ref) < 3:
        return None
    x = np.concatenate([ref, np.ones((len(ref), 1), dtype=np.float32)], axis=1)
    w = np.sqrt(weights).reshape(-1, 1).astype(np.float32)
    xw = x * w
    yw = cur * w
    lhs = xw.T @ xw + ridge * np.eye(3, dtype=np.float32)
    rhs = xw.T @ yw
    try:
        beta = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return None
    return beta.T.astype(np.float32)


def predict_translation(ref, delta):
    return ref + delta[None]


def predict_affine(ref, affine):
    x = np.concatenate([ref, np.ones((len(ref), 1), dtype=np.float32)], axis=1)
    return x @ affine.T


def endpoint_errors(pred_norm, cur_norm, cur_wh, patch_size=16.0):
    diff_norm = pred_norm - cur_norm
    diff_px = diff_norm * cur_wh[None]
    diff_patch = diff_px / float(patch_size)
    return {
        "norm": np.linalg.norm(diff_norm, axis=1),
        "px": np.linalg.norm(diff_px, axis=1),
        "patch": np.linalg.norm(diff_patch, axis=1),
    }


def stats(values):
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return None
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "max": float(values.max()),
    }


def evaluate_predictions(ref, cur, cur_wh, weights, train_idx=None):
    if train_idx is None:
        train_idx = np.arange(len(ref))
    train_idx = np.asarray(train_idx, dtype=np.int64)
    delta = weighted_translation(ref[train_idx], cur[train_idx], weights[train_idx])
    pred_zero = ref.copy()
    pred_trans = predict_translation(ref, delta)
    affine = fit_affine(ref[train_idx], cur[train_idx], weights[train_idx])
    pred_affine = None if affine is None else predict_affine(ref, affine)
    return {
        "translation_delta_norm": delta,
        "affine_matrix_norm": affine,
        "pred_zero": pred_zero,
        "pred_translation": pred_trans,
        "pred_affine": pred_affine,
        "errors_zero": endpoint_errors(pred_zero, cur, cur_wh),
        "errors_translation": endpoint_errors(pred_trans, cur, cur_wh),
        "errors_affine": None if pred_affine is None else endpoint_errors(pred_affine, cur, cur_wh),
    }


def trial_analysis(arrays, sample_counts, trials, rng):
    ref = arrays["ref_norm"]
    cur = arrays["cur_norm"]
    cur_wh = arrays["cur_wh"]
    weights = arrays["weights"]
    n = len(ref)
    out = {}
    full_eval = evaluate_predictions(ref, cur, cur_wh, weights)
    out["full"] = summarize_eval(full_eval, np.arange(n))

    for count in sample_counts:
        if n < count:
            continue
        rows = []
        for _ in range(trials):
            train = rng.choice(n, size=count, replace=False)
            if n > count:
                mask = np.ones(n, dtype=bool)
                mask[train] = False
                test = np.flatnonzero(mask)
            else:
                test = np.arange(n)
            ev = evaluate_predictions(ref, cur, cur_wh, weights, train)
            row = summarize_eval(ev, test)
            rows.append(row)
        out[str(count)] = aggregate_trial_rows(rows)
    return out


def summarize_eval(ev, idx):
    idx = np.asarray(idx, dtype=np.int64)
    zero = ev["errors_zero"]["patch"][idx]
    trans = ev["errors_translation"]["patch"][idx]
    aff = None if ev["errors_affine"] is None else ev["errors_affine"]["patch"][idx]
    return {
        "eval_count": int(len(idx)),
        "zero_patch_median": float(np.median(zero)) if len(zero) else None,
        "translation_patch_median": float(np.median(trans)) if len(trans) else None,
        "affine_patch_median": float(np.median(aff)) if aff is not None and len(aff) else None,
        "zero_patch_mean": float(np.mean(zero)) if len(zero) else None,
        "translation_patch_mean": float(np.mean(trans)) if len(trans) else None,
        "affine_patch_mean": float(np.mean(aff)) if aff is not None and len(aff) else None,
        "translation_improvement_median": float(np.median(zero) - np.median(trans)) if len(zero) else None,
        "affine_improvement_median": float(np.median(zero) - np.median(aff)) if aff is not None and len(aff) else None,
    }


def aggregate_trial_rows(rows):
    keys = [k for k in rows[0].keys() if k != "eval_count"]
    out = {"trials": int(len(rows)), "eval_count_mean": float(np.mean([r["eval_count"] for r in rows]))}
    for key in keys:
        vals = np.array([r[key] for r in rows if r[key] is not None], dtype=np.float32)
        out[key] = stats(vals)
    return out


def draw_prediction_overlay(path, boundary_dir, arrays, ev, max_draw):
    cur_grid_path = boundary_dir / "03_cur_human3r_crop_grid.jpg"
    img = cv2.imread(str(cur_grid_path), cv2.IMREAD_COLOR)
    if img is None:
        cur_w, cur_h = arrays["cur_wh"]
        img = np.full((int(cur_h), int(cur_w), 3), 245, dtype=np.uint8)
    canvas = img.copy()
    n = len(arrays["cur_norm"])
    ids = np.arange(n)
    if n > max_draw:
        ids = np.linspace(0, n - 1, max_draw).round().astype(np.int64)
    cur_wh = arrays["cur_wh"]
    true_px = arrays["cur_norm"] * cur_wh[None]
    zero_px = ev["pred_zero"] * cur_wh[None]
    trans_px = ev["pred_translation"] * cur_wh[None]
    aff_px = None if ev["pred_affine"] is None else ev["pred_affine"] * cur_wh[None]
    for idx in ids:
        true = tuple(np.round(true_px[idx]).astype(int).tolist())
        zero = tuple(np.round(zero_px[idx]).astype(int).tolist())
        trans = tuple(np.round(trans_px[idx]).astype(int).tolist())
        cv2.line(canvas, zero, true, (255, 80, 80), 1, cv2.LINE_AA)
        cv2.line(canvas, trans, true, (80, 190, 80), 1, cv2.LINE_AA)
        cv2.circle(canvas, true, 4, (255, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, zero, 3, (255, 80, 80), -1, cv2.LINE_AA)
        cv2.circle(canvas, trans, 3, (80, 190, 80), -1, cv2.LINE_AA)
        if aff_px is not None:
            aff = tuple(np.round(aff_px[idx]).astype(int).tolist())
            cv2.line(canvas, aff, true, (40, 170, 255), 1, cv2.LINE_AA)
            cv2.circle(canvas, aff, 3, (40, 170, 255), -1, cv2.LINE_AA)
    legend = np.zeros((64, canvas.shape[1], 3), dtype=np.uint8)
    cv2.putText(legend, "true anchor=magenta, no correction=blue/red, translation=green, affine=orange", (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 255, 255), 2, cv2.LINE_AA)
    out = np.concatenate([legend, canvas], axis=0)
    cv2.imwrite(str(path), out)


def draw_single_method_overlay(path, boundary_dir, arrays, ev, method, max_draw):
    specs = {
        "no_correction": {
            "title": "No correction: reference patch keeps the same normalized position",
            "pred_key": "pred_zero",
            "err_key": "errors_zero",
            "color": (230, 80, 80),
        },
        "translation": {
            "title": "Translation correction: one global average dx,dy from all anchors",
            "pred_key": "pred_translation",
            "err_key": "errors_translation",
            "color": (80, 180, 80),
        },
        "affine": {
            "title": "Affine correction: 2D coarse re-anchor fitted from anchors",
            "pred_key": "pred_affine",
            "err_key": "errors_affine",
            "color": (40, 170, 255),
        },
    }
    spec = specs[method]
    pred_norm = ev[spec["pred_key"]]
    errors = ev[spec["err_key"]]
    cur_grid_path = boundary_dir / "03_cur_human3r_crop_grid.jpg"
    img = cv2.imread(str(cur_grid_path), cv2.IMREAD_COLOR)
    if img is None:
        cur_w, cur_h = arrays["cur_wh"]
        img = np.full((int(cur_h), int(cur_w), 3), 245, dtype=np.uint8)

    canvas = img.copy()
    n = len(arrays["cur_norm"])
    ids = np.arange(n)
    if n > max_draw:
        ids = np.linspace(0, n - 1, max_draw).round().astype(np.int64)

    cur_wh = arrays["cur_wh"]
    true_px = arrays["cur_norm"] * cur_wh[None]
    if pred_norm is None:
        pred_px = None
    else:
        pred_px = pred_norm * cur_wh[None]

    method_color = spec["color"]
    true_color = (255, 0, 255)
    if pred_px is not None:
        for idx in ids:
            pred = tuple(np.round(pred_px[idx]).astype(int).tolist())
            true = tuple(np.round(true_px[idx]).astype(int).tolist())
            # **========== 原始代码：较粗箭头和较大点 ==========**
            # cv2.arrowedLine(canvas, pred, true, method_color, 2, cv2.LINE_AA, tipLength=0.20)
            # cv2.circle(canvas, pred, 4, method_color, -1, cv2.LINE_AA)
            # cv2.circle(canvas, true, 5, true_color, -1, cv2.LINE_AA)
            # cv2.circle(canvas, true, 8, (255, 255, 255), 1, cv2.LINE_AA)
            # **========== 新代码：更细的误差线和更小的点 ==========**
            cv2.line(canvas, pred, true, method_color, 1, cv2.LINE_AA)
            cv2.circle(canvas, pred, 2, method_color, -1, cv2.LINE_AA)
            cv2.circle(canvas, true, 2, true_color, -1, cv2.LINE_AA)
            cv2.circle(canvas, true, 4, (255, 255, 255), 1, cv2.LINE_AA)
            # **========== 结束 ==========**

    if errors is None:
        err_text = "patch error: unavailable"
    else:
        patch_err = np.asarray(errors["patch"], dtype=np.float32)
        err_text = f"anchors={n} | median patch error={np.median(patch_err):.2f} | mean={patch_err.mean():.2f}"

    legend = np.zeros((92, canvas.shape[1], 3), dtype=np.uint8)
    cv2.putText(legend, spec["title"], (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(legend, err_text, (12, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 1, cv2.LINE_AA)
    # **========== 原始代码：箭头说明 ==========**
    # cv2.putText(legend, "arrow: predicted current position -> true mesh-verified current anchor; shorter is better", (12, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (190, 190, 190), 1, cv2.LINE_AA)
    # **========== 新代码：明确 GT、预测和误差线含义 ==========**
    cv2.putText(legend, "magenta=GT mesh anchor, colored=method prediction, thin line=prediction error; shorter is better", (12, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (190, 190, 190), 1, cv2.LINE_AA)
    # **========== 结束 ==========**
    out = np.concatenate([legend, canvas], axis=0)
    cv2.imwrite(str(path), out)


def draw_error_chart(path, correction_summary):
    width, height = 980, 460
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    cv2.putText(canvas, "Held-out patch error: no correction vs translation vs affine", (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (20, 20, 20), 2, cv2.LINE_AA)
    trial_items = [(k, v) for k, v in correction_summary["sampling"].items() if k != "full"]
    if not trial_items:
        cv2.imwrite(str(path), canvas)
        return
    labels = [k for k, _ in trial_items]
    zero = [v["zero_patch_median"]["median"] for _, v in trial_items]
    trans = [v["translation_patch_median"]["median"] for _, v in trial_items]
    aff = [v["affine_patch_median"]["median"] if v.get("affine_patch_median") else None for _, v in trial_items]
    values = [x for x in zero + trans + [a for a in aff if a is not None] if x is not None]
    max_val = max(values) if values else 1.0
    left, right, top, bottom = 72, 36, 74, 72
    plot_w = width - left - right
    plot_h = height - top - bottom
    cv2.rectangle(canvas, (left, top), (left + plot_w, top + plot_h), (220, 220, 220), 1)
    group_w = plot_w / max(len(labels), 1)
    colors = [(230, 80, 80), (80, 180, 80), (40, 170, 255)]
    for i, label in enumerate(labels):
        base_x = left + int(i * group_w + group_w * 0.16)
        bar_w = max(12, int(group_w * 0.16))
        vals = [zero[i], trans[i], aff[i]]
        for j, val in enumerate(vals):
            if val is None:
                continue
            h = int((val / max_val) * plot_h)
            x0 = base_x + j * (bar_w + 6)
            y0 = top + plot_h - h
            cv2.rectangle(canvas, (x0, y0), (x0 + bar_w, top + plot_h), colors[j], -1)
            cv2.putText(canvas, f"{val:.2f}", (x0 - 6, max(top + 16, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1, cv2.LINE_AA)
        cv2.putText(canvas, label, (base_x, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, "red=no correction, green=translation, orange=affine", (left, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (50, 50, 50), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def analyze_one(aabb_dir, out_dir, sample_counts, trials, rng, max_draw):
    aabb_dir, boundary_dir, summary = load_boundary_summary(aabb_dir)
    anchors = summary.get("anchors", [])
    if not anchors:
        raise RuntimeError(f"no anchors in {boundary_dir / 'summary.json'}")

    ref_grid = summary.get("human3r_ref_grid_hw")
    cur_grid = summary.get("human3r_cur_grid_hw")
    for anchor in anchors:
        anchor["ref_grid_hw"] = ref_grid
        anchor["cur_grid_hw"] = cur_grid

    arrays = anchors_to_arrays(anchors)
    ev = evaluate_predictions(arrays["ref_norm"], arrays["cur_norm"], arrays["cur_wh"], arrays["weights"])
    sampling = trial_analysis(arrays, sample_counts, trials, rng)

    pair_out = out_dir / aabb_dir.name
    pair_out.mkdir(parents=True, exist_ok=True)
    draw_prediction_overlay(pair_out / "correction_prediction_overlay.jpg", boundary_dir, arrays, ev, max_draw)
    draw_single_method_overlay(pair_out / "correction_overlay_no_correction.jpg", boundary_dir, arrays, ev, "no_correction", max_draw)
    draw_single_method_overlay(pair_out / "correction_overlay_translation.jpg", boundary_dir, arrays, ev, "translation", max_draw)
    draw_single_method_overlay(pair_out / "correction_overlay_affine.jpg", boundary_dir, arrays, ev, "affine", max_draw)
    correction_summary = {
        "aabb_dir": str(aabb_dir),
        "boundary_dir": str(boundary_dir),
        "pair_name": summary.get("pair_name"),
        "pair_kind": summary.get("pair_kind"),
        "n_anchors": int(len(anchors)),
        "mesh_geometry_inliers": int(summary.get("mesh_geometry_inliers", 0)),
        "unique_anchor_patch_pairs": int(summary.get("unique_anchor_patch_pairs", len(anchors))),
        "encoder_positive_cosine": summary.get("encoder_positive_cosine"),
        "encoder_true_match_rank": summary.get("encoder_true_match_rank"),
        "full_fit": {
            "translation_delta_norm": ev["translation_delta_norm"].astype(float).tolist(),
            "affine_matrix_norm": None if ev["affine_matrix_norm"] is None else ev["affine_matrix_norm"].astype(float).tolist(),
            "zero_error_patch": stats(ev["errors_zero"]["patch"]),
            "translation_error_patch": stats(ev["errors_translation"]["patch"]),
            "affine_error_patch": None if ev["errors_affine"] is None else stats(ev["errors_affine"]["patch"]),
        },
        "sampling": sampling,
        "visualizations": {
            "prediction_overlay": str(pair_out / "correction_prediction_overlay.jpg"),
            "no_correction_overlay": str(pair_out / "correction_overlay_no_correction.jpg"),
            "translation_overlay": str(pair_out / "correction_overlay_translation.jpg"),
            "affine_overlay": str(pair_out / "correction_overlay_affine.jpg"),
            "error_chart": str(pair_out / "correction_sampling_error_chart.jpg"),
        },
    }
    draw_error_chart(pair_out / "correction_sampling_error_chart.jpg", correction_summary)
    (pair_out / "correction_summary.json").write_text(json.dumps(correction_summary, indent=2), encoding="utf-8")
    return correction_summary


def draw_multi_summary(path, summaries):
    width, height = 1280, 130 + 80 * len(summaries)
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    cv2.putText(canvas, "AABB boundary anchor correction proxy", (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.88, (20, 20, 20), 2, cv2.LINE_AA)
    headers = ["sample", "anchors", "no corr", "translation", "affine", "best 8", "best 16", "signal"]
    xs = [18, 430, 545, 670, 810, 940, 1060, 1180]
    for x, h in zip(xs, headers):
        cv2.putText(canvas, h, (x, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (50, 50, 50), 2, cv2.LINE_AA)
    for i, s in enumerate(summaries):
        y = 128 + 80 * i
        name = Path(s["aabb_dir"]).name
        full = s["full_fit"]
        zero = full["zero_error_patch"]["median"]
        trans = full["translation_error_patch"]["median"]
        aff = full["affine_error_patch"]["median"] if full["affine_error_patch"] else None
        samp8 = s["sampling"].get("8", {}).get("translation_patch_median", {}).get("median")
        samp16 = s["sampling"].get("16", {}).get("translation_patch_median", {}).get("median")
        best = min([v for v in [trans, aff] if v is not None])
        signal = "good" if best < zero * 0.5 else "weak" if best < zero else "bad"
        vals = [
            str(s["unique_anchor_patch_pairs"]),
            f"{zero:.2f}",
            f"{trans:.2f}",
            f"{aff:.2f}" if aff is not None else "None",
            f"{samp8:.2f}" if samp8 is not None else "skip",
            f"{samp16:.2f}" if samp16 is not None else "skip",
            signal,
        ]
        cv2.putText(canvas, name[:52], (xs[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (20, 20, 20), 2, cv2.LINE_AA)
        for x, val in zip(xs[1:], vals):
            color = (40, 140, 40) if val == "good" else (30, 110, 220) if val == "weak" else (20, 20, 20)
            cv2.putText(canvas, val, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, color, 2, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    sample_counts = parse_sample_counts(args.sample_counts)
    if args.out_dir is None:
        # **========== 原始代码：旧服务器输出路径 ==========**
        # args.out_dir = "/workspace/code/Movie3R/output/rich_aabb_anchor_correction_proxy"
        # **========== 新代码：当前仓库输出路径 ==========**
        args.out_dir = str(Path(__file__).resolve().parents[1] / "output" / "rich_aabb_anchor_correction_proxy")
        # **========== 结束 ==========**
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for aabb_dir in args.aabb_dir:
        summaries.append(analyze_one(Path(aabb_dir), out_dir, sample_counts, args.trials, rng, args.max_draw))

    root = {
        "args": vars(args),
        "summaries": summaries,
        "visualizations": {
            "multi_summary": str(out_dir / "correction_proxy_summary.jpg"),
        },
    }
    draw_multi_summary(out_dir / "correction_proxy_summary.jpg", summaries)
    (out_dir / "summary.json").write_text(json.dumps(root, indent=2), encoding="utf-8")
    print(json.dumps(
        {
            "out_dir": str(out_dir),
            "samples": [
                {
                    "name": Path(s["aabb_dir"]).name,
                    "anchors": s["unique_anchor_patch_pairs"],
                    "no_correction_patch_median": s["full_fit"]["zero_error_patch"]["median"],
                    "translation_patch_median": s["full_fit"]["translation_error_patch"]["median"],
                    "affine_patch_median": None if s["full_fit"]["affine_error_patch"] is None else s["full_fit"]["affine_error_patch"]["median"],
                    "sample_8_translation_patch_median": s["sampling"].get("8", {}).get("translation_patch_median", {}).get("median"),
                    "sample_16_translation_patch_median": s["sampling"].get("16", {}).get("translation_patch_median", {}).get("median"),
                }
                for s in summaries
            ],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
