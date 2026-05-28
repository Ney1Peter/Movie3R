#!/usr/bin/env python3
"""Build model-facing anchor evidence from AABB boundary anchors.

This is the next step after the correction proxy experiment. It still does not
modify Human3R/Movie3R. It reads mesh-verified boundary anchors from
verify_rich_aabb_anchor_step1.py outputs, reruns the original Human3R encoder for
the two boundary frames, and converts anchors into:

1. A fixed-size evidence vector suitable for a future pose-only adapter.
2. A patch lookup map: for each current patch, where should it look in the
   reference frame after no/translation/affine correction?
3. Metrics showing whether affine evidence predicts the correct reference patch
   better than no correction.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
SRC_ROOT = REPO_ROOT / "src"
for path in [REPO_ROOT, SRC_ROOT, SCRIPT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from report_image_style import patch_cv2_text  # noqa: E402

patch_cv2_text(cv2)

from analyze_rich_aabb_anchor_correction import (  # noqa: E402
    anchors_to_arrays,
    evaluate_predictions,
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
    parser.add_argument("--max_draw", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", default=None)
    return parser.parse_args()


def encode_image(model, image_path, size, device):
    img_tensor, true_shape, crop_rgb, meta = load_human3r_image(image_path, size)
    img_tensor = img_tensor.to(device)
    true_shape = true_shape.to(device)
    with torch.no_grad():
        enc = model._encode_image(img_tensor, true_shape)[0][-1]
        dec = model.decoder_embed(enc)
    h, w = map(int, true_shape[0].detach().cpu().numpy().tolist())
    patch_size = int(model.croco_args["patch_size"])
    grid_hw = (h // patch_size, w // patch_size)
    if enc.shape[1] != grid_hw[0] * grid_hw[1]:
        raise RuntimeError(f"token/grid mismatch for {image_path}: {enc.shape[1]} vs {grid_hw}")
    return {
        "enc": enc,
        "dec": dec,
        "crop_rgb": crop_rgb,
        "meta": meta,
        "grid_hw": grid_hw,
        "patch_size": patch_size,
        "true_shape_hw": [h, w],
    }


def patch_centers_norm(grid_hw):
    gh, gw = grid_hw
    yy, xx = np.mgrid[0:gh, 0:gw]
    centers = np.stack([(xx.reshape(-1) + 0.5) / gw, (yy.reshape(-1) + 0.5) / gh], axis=1)
    return centers.astype(np.float32)


def patch_idx_to_xy(idx, grid_hw):
    _, gw = grid_hw
    return np.array([idx % gw, idx // gw], dtype=np.float32)


def patch_idx_to_center_px(idx, grid_hw, patch_size):
    xy = patch_idx_to_xy(int(idx), grid_hw)
    return (xy + 0.5) * float(patch_size)


def nearest_patch_indices(points_norm, grid_hw):
    gh, gw = grid_hw
    pts = np.asarray(points_norm, dtype=np.float32)
    x = np.floor(pts[:, 0] * gw).astype(np.int32)
    y = np.floor(pts[:, 1] * gh).astype(np.int32)
    x = np.clip(x, 0, gw - 1)
    y = np.clip(y, 0, gh - 1)
    return (y * gw + x).astype(np.int64)


def apply_affine(points_norm, affine):
    pts = np.asarray(points_norm, dtype=np.float32)
    x = np.concatenate([pts, np.ones((len(pts), 1), dtype=np.float32)], axis=1)
    return (x @ affine.T).astype(np.float32)


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


def endpoint_patch_error(pred_ref_idx, true_ref_idx, ref_grid_hw):
    pred_xy = np.array([patch_idx_to_xy(int(i), ref_grid_hw) for i in pred_ref_idx], dtype=np.float32)
    true_xy = np.array([patch_idx_to_xy(int(i), ref_grid_hw) for i in true_ref_idx], dtype=np.float32)
    return np.linalg.norm(pred_xy - true_xy, axis=1)


def summarize_lookup(name, pred_ref_idx, true_ref_idx, cur_idx, sim_matrix, ref_grid_hw):
    err = endpoint_patch_error(pred_ref_idx, true_ref_idx, ref_grid_hw)
    cos = np.array([sim_matrix[int(r), int(c)] for r, c in zip(pred_ref_idx, cur_idx)], dtype=np.float32)
    exact = pred_ref_idx == true_ref_idx
    within1 = err <= 1.0
    within2 = err <= 2.0
    ranks = []
    for r, c in zip(pred_ref_idx, cur_idx):
        sim = sim_matrix[int(r), int(c)]
        ranks.append(int((sim_matrix[:, int(c)] > sim).sum() + 1))
    ranks = np.asarray(ranks, dtype=np.float32)
    return {
        "name": name,
        "patch_error": stats(err),
        "token_cosine": stats(cos),
        "exact_match_rate": float(exact.mean()) if len(exact) else None,
        "within_1_patch_rate": float(within1.mean()) if len(within1) else None,
        "within_2_patch_rate": float(within2.mean()) if len(within2) else None,
        "rank_median": float(np.median(ranks)) if len(ranks) else None,
        "rank_mean": float(ranks.mean()) if len(ranks) else None,
        "top1_rate": float((ranks <= 1).mean()) if len(ranks) else None,
        "top5_rate": float((ranks <= 5).mean()) if len(ranks) else None,
        "top10_rate": float((ranks <= 10).mean()) if len(ranks) else None,
    }


def build_evidence_vector(summary, anchors, arrays, full_eval, lookup_summaries):
    n = len(anchors)
    count_norm = min(float(n) / 64.0, 1.0)
    count_log = float(np.log1p(n) / np.log1p(256.0))
    cos = arrays["cosine"]
    mesh_err = arrays["mesh_err"]
    weights = arrays["weights"]
    affine = full_eval["affine_matrix_norm"]
    translation = full_eval["translation_delta_norm"]
    if affine is None:
        affine_residual = np.zeros((2, 3), dtype=np.float32)
    else:
        affine_residual = affine - np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    aff_err = full_eval["errors_affine"]["patch"] if full_eval["errors_affine"] is not None else np.full((n,), np.nan, dtype=np.float32)
    trans_err = full_eval["errors_translation"]["patch"]
    zero_err = full_eval["errors_zero"]["patch"]
    improvement = np.nanmedian(zero_err) - np.nanmedian(aff_err)
    residual_gate = float(np.clip(1.0 - np.nanmedian(aff_err) / 4.0, 0.0, 1.0)) if np.isfinite(aff_err).any() else 0.0
    count_gate = float(np.clip((n - 4.0) / 12.0, 0.0, 1.0))
    quality_gate = float(count_gate * residual_gate)
    overlap = summary.get("mesh_visible_overlap", {}) or {}
    positive = summary.get("encoder_positive_cosine", {}) or {}
    rank = summary.get("encoder_true_match_rank", {}) or {}

    vector = np.array(
        [
            count_norm,
            count_log,
            quality_gate,
            float(np.mean(cos)) if len(cos) else 0.0,
            float(np.median(cos)) if len(cos) else 0.0,
            float(np.mean(mesh_err)) if len(mesh_err) else 24.0,
            float(np.median(mesh_err)) if len(mesh_err) else 24.0,
            float(np.sum(weights) / max(n, 1)),
            float(overlap.get("overlap_min", 0.0)),
            float(overlap.get("jaccard", 0.0)),
            float(positive.get("mean", 0.0) or 0.0),
            float(rank.get("top10", 0.0) or 0.0),
            float(translation[0]),
            float(translation[1]),
            *affine_residual.reshape(-1).astype(float).tolist(),
            float(np.nanmedian(zero_err)) if len(zero_err) else 0.0,
            float(np.nanmedian(trans_err)) if len(trans_err) else 0.0,
            float(np.nanmedian(aff_err)) if len(aff_err) else 0.0,
            float(improvement) if np.isfinite(improvement) else 0.0,
        ],
        dtype=np.float32,
    )
    names = [
        "count_norm",
        "count_log",
        "quality_gate",
        "anchor_cos_mean",
        "anchor_cos_median",
        "mesh_err_mean",
        "mesh_err_median",
        "weight_mean",
        "overlap_min",
        "overlap_jaccard",
        "encoder_positive_cos_mean",
        "encoder_true_rank_top10",
        "translation_dx_norm",
        "translation_dy_norm",
        "affine_a_minus_1",
        "affine_b",
        "affine_tx",
        "affine_c",
        "affine_d_minus_1",
        "affine_ty",
        "no_correction_error_patch_median",
        "translation_error_patch_median",
        "affine_error_patch_median",
        "affine_improvement_patch_median",
    ]
    return {
        "names": names,
        "values": vector.astype(float).tolist(),
        "quality_gate": quality_gate,
        "count_gate": count_gate,
        "residual_gate": residual_gate,
        "lookup_summary_names": [item["name"] for item in lookup_summaries],
    }


def draw_lookup_comparison(path, ref_rgb, cur_rgb, anchors, lookup_indices, ref_grid_hw, cur_grid_hw, patch_size, max_draw):
    ref = draw_patch_grid(ref_rgb, patch_size)
    cur = draw_patch_grid(cur_rgb, patch_size)
    h = max(ref.shape[0], cur.shape[0])
    canvas = np.zeros((h + 82, ref.shape[1] + cur.shape[1], 3), dtype=np.uint8)
    canvas[:82] = 20
    canvas[82 : 82 + ref.shape[0], : ref.shape[1]] = ref
    canvas[82 : 82 + cur.shape[0], ref.shape[1] : ref.shape[1] + cur.shape[1]] = cur
    cv2.putText(canvas, "anchor lookup: true=magenta, no-correction=red, translation=green, affine=orange", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "lines connect current anchor to predicted reference patch", (12, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 1, cv2.LINE_AA)
    ids = np.arange(len(anchors))
    if len(ids) > max_draw:
        ids = np.linspace(0, len(ids) - 1, max_draw).round().astype(np.int64)
    colors = {
        "same_position": (255, 80, 80),
        "translation": (80, 190, 80),
        "affine": (40, 170, 255),
    }
    for idx in ids:
        a = anchors[int(idx)]
        cur_center = patch_idx_to_center_px(a["cur_patch_idx"], cur_grid_hw, patch_size)
        cur_pt = (int(round(ref.shape[1] + cur_center[0])), int(round(82 + cur_center[1])))
        true_center = patch_idx_to_center_px(a["ref_patch_idx"], ref_grid_hw, patch_size)
        true_pt = (int(round(true_center[0])), int(round(82 + true_center[1])))
        cv2.circle(canvas, cur_pt, 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, true_pt, 5, (255, 0, 255), -1, cv2.LINE_AA)
        cv2.line(canvas, true_pt, cur_pt, (255, 0, 255), 1, cv2.LINE_AA)
        for key, color in colors.items():
            pred_idx = int(lookup_indices[key][idx])
            pred_center = patch_idx_to_center_px(pred_idx, ref_grid_hw, patch_size)
            pred_pt = (int(round(pred_center[0])), int(round(82 + pred_center[1])))
            cv2.circle(canvas, pred_pt, 3, color, -1, cv2.LINE_AA)
            cv2.line(canvas, pred_pt, cur_pt, color, 1, cv2.LINE_AA)
    save_rgb(path, canvas)


def draw_lookup_error_chart(path, lookup_summaries):
    width, height = 980, 420
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    cv2.putText(canvas, "Reference patch lookup from anchor evidence", (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (20, 20, 20), 2, cv2.LINE_AA)
    names = [s["name"] for s in lookup_summaries]
    errors = [s["patch_error"]["median"] for s in lookup_summaries]
    cosines = [s["token_cosine"]["median"] for s in lookup_summaries]
    max_err = max(errors + [1.0])
    left, top = 72, 76
    plot_w, plot_h = 850, 250
    cv2.rectangle(canvas, (left, top), (left + plot_w, top + plot_h), (220, 220, 220), 1)
    group_w = plot_w / max(len(names), 1)
    for i, (name, err, cos) in enumerate(zip(names, errors, cosines)):
        x0 = left + int(i * group_w + group_w * 0.25)
        bar_w = int(group_w * 0.22)
        h_err = int(err / max_err * plot_h)
        h_cos = int(np.clip((cos + 1.0) / 2.0, 0.0, 1.0) * plot_h)
        cv2.rectangle(canvas, (x0, top + plot_h - h_err), (x0 + bar_w, top + plot_h), (230, 80, 80), -1)
        cv2.rectangle(canvas, (x0 + bar_w + 8, top + plot_h - h_cos), (x0 + 2 * bar_w + 8, top + plot_h), (80, 150, 230), -1)
        cv2.putText(canvas, f"{err:.2f}", (x0 - 4, max(top + 15, top + plot_h - h_err - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (30, 30, 30), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{cos:.2f}", (x0 + bar_w, max(top + 15, top + plot_h - h_cos - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (30, 30, 30), 1, cv2.LINE_AA)
        cv2.putText(canvas, name[:15], (x0 - 16, height - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(canvas, "red=median patch error lower is better, blue=median token cosine higher is better", (left, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def draw_flow_map(path, cur_rgb, cur_grid_hw, patch_size, inv_affine, translation_delta, max_draw):
    cur = draw_patch_grid(cur_rgb, patch_size)
    canvas = cur.copy()
    centers_cur_norm = patch_centers_norm(cur_grid_hw)
    ids = np.arange(len(centers_cur_norm))
    if len(ids) > max_draw:
        ids = np.linspace(0, len(ids) - 1, max_draw).round().astype(np.int64)
    if inv_affine is not None:
        ref_pred = apply_affine(centers_cur_norm, inv_affine)
    else:
        ref_pred = centers_cur_norm - translation_delta[None]
    delta = centers_cur_norm - ref_pred
    gh, gw = cur_grid_hw
    for idx in ids:
        cur_xy = patch_idx_to_center_px(idx, cur_grid_hw, patch_size)
        d = delta[idx] * np.array([gw * patch_size, gh * patch_size], dtype=np.float32)
        start = (int(round(cur_xy[0] - 0.5 * d[0])), int(round(cur_xy[1] - 0.5 * d[1])))
        end = (int(round(cur_xy[0] + 0.5 * d[0])), int(round(cur_xy[1] + 0.5 * d[1])))
        cv2.arrowedLine(canvas, start, end, (40, 170, 255), 1, cv2.LINE_AA, tipLength=0.25)
    legend = np.zeros((56, canvas.shape[1], 3), dtype=np.uint8)
    cv2.putText(legend, "affine correction field on current patch grid", (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    out = np.concatenate([legend, canvas], axis=0)
    save_rgb(path, out)


def analyze_one(aabb_dir, out_dir, model, size, device, max_draw):
    aabb_dir, boundary_dir, summary = load_boundary_summary(aabb_dir)
    anchors = summary.get("anchors", [])
    if not anchors:
        raise RuntimeError(f"no anchors found in {boundary_dir / 'summary.json'}")
    ref_grid = summary.get("human3r_ref_grid_hw")
    cur_grid = summary.get("human3r_cur_grid_hw")
    for a in anchors:
        a["ref_grid_hw"] = ref_grid
        a["cur_grid_hw"] = cur_grid

    arrays = anchors_to_arrays(anchors)
    ref_info = encode_image(model, summary["ref"]["image"], size, device)
    cur_info = encode_image(model, summary["cur"]["image"], size, device)
    patch_size = ref_info["patch_size"]
    ref_grid_hw = ref_info["grid_hw"]
    cur_grid_hw = cur_info["grid_hw"]

    if tuple(ref_grid_hw) != tuple(ref_grid):
        raise RuntimeError(f"ref grid mismatch: encoded={ref_grid_hw} summary={ref_grid}")
    if tuple(cur_grid_hw) != tuple(cur_grid):
        raise RuntimeError(f"cur grid mismatch: encoded={cur_grid_hw} summary={cur_grid}")

    ref_token = F.normalize(ref_info["enc"][0].float(), dim=-1)
    cur_token = F.normalize(cur_info["enc"][0].float(), dim=-1)
    sim_matrix = (ref_token @ cur_token.T).detach().cpu().numpy()

    ref_norm = arrays["ref_norm"]
    cur_norm = arrays["cur_norm"]
    weights = arrays["weights"]
    full_eval = evaluate_predictions(ref_norm, cur_norm, arrays["cur_wh"], weights)
    translation = weighted_translation(ref_norm, cur_norm, weights)
    affine = fit_affine(ref_norm, cur_norm, weights)
    inv_affine = invert_affine(affine)

    true_ref_idx = np.array([a["ref_patch_idx"] for a in anchors], dtype=np.int64)
    cur_idx = np.array([a["cur_patch_idx"] for a in anchors], dtype=np.int64)
    same_ref_idx = nearest_patch_indices(cur_norm, ref_grid_hw)
    trans_ref_idx = nearest_patch_indices(cur_norm - translation[None], ref_grid_hw)
    if inv_affine is None:
        affine_ref_idx = trans_ref_idx.copy()
        affine_pred_norm = cur_norm - translation[None]
    else:
        affine_pred_norm = apply_affine(cur_norm, inv_affine)
        affine_ref_idx = nearest_patch_indices(affine_pred_norm, ref_grid_hw)
    oracle_ref_idx = true_ref_idx.copy()
    lookup_indices = {
        "same_position": same_ref_idx,
        "translation": trans_ref_idx,
        "affine": affine_ref_idx,
        "oracle_anchor": oracle_ref_idx,
    }

    lookup_summaries = [
        summarize_lookup("same_position", same_ref_idx, true_ref_idx, cur_idx, sim_matrix, ref_grid_hw),
        summarize_lookup("translation", trans_ref_idx, true_ref_idx, cur_idx, sim_matrix, ref_grid_hw),
        summarize_lookup("affine", affine_ref_idx, true_ref_idx, cur_idx, sim_matrix, ref_grid_hw),
        summarize_lookup("oracle_anchor", oracle_ref_idx, true_ref_idx, cur_idx, sim_matrix, ref_grid_hw),
    ]
    evidence = build_evidence_vector(summary, anchors, arrays, full_eval, lookup_summaries)

    sample_out = out_dir / aabb_dir.name
    sample_out.mkdir(parents=True, exist_ok=True)
    draw_lookup_comparison(sample_out / "anchor_lookup_comparison.jpg", ref_info["crop_rgb"], cur_info["crop_rgb"], anchors, lookup_indices, ref_grid_hw, cur_grid_hw, patch_size, max_draw)
    draw_lookup_error_chart(sample_out / "lookup_error_chart.jpg", lookup_summaries)
    draw_flow_map(sample_out / "affine_correction_field.jpg", cur_info["crop_rgb"], cur_grid_hw, patch_size, inv_affine, translation, max_draw)
    np.save(sample_out / "anchor_evidence_vector.npy", np.array(evidence["values"], dtype=np.float32))

    result = {
        "aabb_dir": str(aabb_dir),
        "boundary_dir": str(boundary_dir),
        "ref": summary["ref"],
        "cur": summary["cur"],
        "n_anchors": int(len(anchors)),
        "mesh_geometry_inliers": int(summary.get("mesh_geometry_inliers", 0)),
        "unique_anchor_patch_pairs": int(summary.get("unique_anchor_patch_pairs", len(anchors))),
        "ref_grid_hw": list(ref_grid_hw),
        "cur_grid_hw": list(cur_grid_hw),
        "forward_translation_norm": translation.astype(float).tolist(),
        "forward_affine_norm": None if affine is None else affine.astype(float).tolist(),
        "inverse_affine_norm": None if inv_affine is None else inv_affine.astype(float).tolist(),
        "lookup_summaries": lookup_summaries,
        "evidence_vector": evidence,
        "visualizations": {
            "anchor_lookup_comparison": str(sample_out / "anchor_lookup_comparison.jpg"),
            "lookup_error_chart": str(sample_out / "lookup_error_chart.jpg"),
            "affine_correction_field": str(sample_out / "affine_correction_field.jpg"),
            "anchor_evidence_vector_npy": str(sample_out / "anchor_evidence_vector.npy"),
        },
    }
    (sample_out / "anchor_evidence_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def draw_root_summary(path, results):
    width, height = 1320, 120 + 82 * len(results)
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    cv2.putText(canvas, "Anchor evidence: current patch -> reference patch lookup", (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.86, (20, 20, 20), 2, cv2.LINE_AA)
    headers = ["sample", "anchors", "gate", "same err", "trans err", "affine err", "oracle cos", "affine cos"]
    xs = [18, 470, 580, 690, 820, 950, 1080, 1210]
    for x, h in zip(xs, headers):
        cv2.putText(canvas, h, (x, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (50, 50, 50), 2, cv2.LINE_AA)
    for i, r in enumerate(results):
        y = 126 + 82 * i
        name = Path(r["aabb_dir"]).name[:58]
        lookup = {s["name"]: s for s in r["lookup_summaries"]}
        vals = [
            str(r["unique_anchor_patch_pairs"]),
            f"{r['evidence_vector']['quality_gate']:.2f}",
            f"{lookup['same_position']['patch_error']['median']:.2f}",
            f"{lookup['translation']['patch_error']['median']:.2f}",
            f"{lookup['affine']['patch_error']['median']:.2f}",
            f"{lookup['oracle_anchor']['token_cosine']['median']:.2f}",
            f"{lookup['affine']['token_cosine']['median']:.2f}",
        ]
        cv2.putText(canvas, name, (xs[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (20, 20, 20), 2, cv2.LINE_AA)
        for x, val in zip(xs[1:], vals):
            cv2.putText(canvas, val, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def main():
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    if args.out_dir is None:
        args.out_dir = str(REPO_ROOT / "output" / "rich_anchor_evidence")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading original Human3R encoder...")
    model = load_model(args.model_path, device=device, verbose=True).eval()
    model.gradient_checkpointing = False

    results = []
    for aabb_dir in args.aabb_dir:
        print(f"Building anchor evidence for {aabb_dir}...")
        results.append(analyze_one(Path(aabb_dir), out_dir, model, args.size, device, args.max_draw))

    draw_root_summary(out_dir / "anchor_evidence_summary.jpg", results)
    root = {
        "args": vars(args),
        "results": results,
        "visualizations": {
            "summary": str(out_dir / "anchor_evidence_summary.jpg"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(root, indent=2), encoding="utf-8")
    print(json.dumps(
        {
            "out_dir": str(out_dir),
            "samples": [
                {
                    "name": Path(r["aabb_dir"]).name,
                    "anchors": r["unique_anchor_patch_pairs"],
                    "quality_gate": r["evidence_vector"]["quality_gate"],
                    "same_position_error": r["lookup_summaries"][0]["patch_error"]["median"],
                    "translation_error": r["lookup_summaries"][1]["patch_error"]["median"],
                    "affine_error": r["lookup_summaries"][2]["patch_error"]["median"],
                    "oracle_cosine": r["lookup_summaries"][3]["token_cosine"]["median"],
                    "affine_cosine": r["lookup_summaries"][2]["token_cosine"]["median"],
                }
                for r in results
            ],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
