#!/usr/bin/env python3
"""Validate that local AnchorTokens carry specific correction information.

This is a decoder-before proxy. It keeps Human3R/Movie3R unchanged, uses frozen
encoder patch tokens, and compares correct AnchorTokens against negative controls:

1. affine_only: global coarse re-anchor without local token residual.
2. correct_anchor_token: local residual from the correct boundary anchors.
3. spatial_only_token: same residual values, but token attention ignores features.
4. shuffled_value_token: correct keys/positions, shuffled residual values.
5. wrong_boundary_token: residuals from another AABB boundary.

If correct_anchor_token beats affine_only and the negative controls degrade, then
the token contains concrete local correction evidence rather than a generic shot
label or empty anchor metadata.
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
for path in [REPO_ROOT, SRC_ROOT, SCRIPT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from report_image_style import patch_cv2_text  # noqa: E402

patch_cv2_text(cv2)

from analyze_rich_aabb_anchor_correction import load_boundary_summary, stats, weighted_translation  # noqa: E402
from prototype_rich_anchor_tokens import (  # noqa: E402
    apply_affine,
    build_data,
    encode_image,
    fit_affine,
    invert_affine,
    nearest_patch_indices,
    patch_error,
    rank_for_pred,
    token_cosine_for_pred,
)
from verify_rich_anchor_encoder_similarity import load_model  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aabb_dir", action="append", required=True, help="AABB Step1 output directory. Can be repeated.")
    parser.add_argument("--model_path", default=str(REPO_ROOT / "src" / "human3r_896L.pth"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--feature_weight", type=float, default=2.0)
    parser.add_argument("--spatial_weight", type=float, default=30.0)
    parser.add_argument("--attention_temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", default=None)
    return parser.parse_args()


def softmax(x):
    x = np.asarray(x, dtype=np.float32)
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-8)


def attention_weights(query_feat, query_pos, token_feat, token_pos, feature_weight, spatial_weight, temperature):
    feat_score = query_feat @ token_feat.T
    spatial_d2 = ((query_pos[:, None, :] - token_pos[None, :, :]) ** 2).sum(axis=-1)
    score = feature_weight * feat_score - spatial_weight * spatial_d2
    return softmax(score / max(temperature, 1e-6))


def fit_target_coarse(data, train):
    ref_norm = data["ref_norm"]
    cur_norm = data["cur_norm"]
    weights = data["weights"]
    translation = weighted_translation(ref_norm[train], cur_norm[train], weights[train])
    affine = fit_affine(ref_norm[train], cur_norm[train], weights[train])
    inv_affine = invert_affine(affine)
    return translation, affine, inv_affine


def affine_base(cur_pos, translation, inv_affine):
    if inv_affine is None:
        return cur_pos - translation[None]
    return apply_affine(cur_pos, inv_affine)


def token_residual_prediction(
    query_feat,
    query_pos,
    base_query,
    token_feat,
    token_pos,
    token_residual,
    feature_weight,
    spatial_weight,
    temperature,
):
    weights = attention_weights(query_feat, query_pos, token_feat, token_pos, feature_weight, spatial_weight, temperature)
    return base_query + weights @ token_residual


def donor_residuals(donor):
    idx = np.arange(len(donor["ref_norm"]), dtype=np.int64)
    translation, _, inv_affine = fit_target_coarse(donor, idx)
    base = affine_base(donor["cur_norm"], translation, inv_affine)
    return donor["ref_norm"] - base


def summarize_method(name, pred_ref_norm, test, data):
    true_ref = data["ref_norm"][test]
    cur_idx = data["cur_idx"][test]
    err = patch_error(pred_ref_norm, true_ref, data["ref_grid_hw"])
    cos = token_cosine_for_pred(pred_ref_norm, cur_idx, data["ref_feat"], data["cur_feat"], data["ref_grid_hw"])
    ranks = rank_for_pred(pred_ref_norm, cur_idx, data["ref_feat"], data["cur_feat"], data["ref_grid_hw"])
    pred_idx = nearest_patch_indices(pred_ref_norm, data["ref_grid_hw"])
    exact = pred_idx == data["ref_idx"][test]
    return {
        "name": name,
        "patch_error": stats(err),
        "token_cosine": stats(cos),
        "rank_median": float(np.median(ranks)) if len(ranks) else None,
        "top1_rate": float((ranks <= 1).mean()) if len(ranks) else None,
        "top5_rate": float((ranks <= 5).mean()) if len(ranks) else None,
        "top10_rate": float((ranks <= 10).mean()) if len(ranks) else None,
        "exact_match_rate": float(exact.mean()) if len(exact) else None,
        "within_1_patch_rate": float((err <= 1.0).mean()) if len(err) else None,
        "within_2_patch_rate": float((err <= 2.0).mean()) if len(err) else None,
    }


def aggregate(rows):
    methods = rows[0].keys()
    out = {}
    for method in methods:
        out[method] = {}
        for group in ["patch_error", "token_cosine"]:
            vals = np.array([row[method][group]["median"] for row in rows if row[method][group] is not None], dtype=np.float32)
            out[method][f"{group}_median_over_anchors"] = stats(vals)
        for scalar in ["rank_median", "top1_rate", "top5_rate", "top10_rate", "exact_match_rate", "within_1_patch_rate", "within_2_patch_rate"]:
            vals = np.array([row[method][scalar] for row in rows if row[method][scalar] is not None], dtype=np.float32)
            out[method][f"{scalar}_over_anchors"] = stats(vals)
    return out


def evaluate_specificity(target, donor, rng, args):
    data = target["data"]
    n = len(data["ref_norm"])
    rows = []
    if n < 4:
        return {}
    donor_res = donor_residuals(donor["data"]) if donor is not None else None
    for i in range(n):
        train = np.array([j for j in range(n) if j != i], dtype=np.int64)
        test = np.array([i], dtype=np.int64)
        translation, _, inv_affine = fit_target_coarse(data, train)
        cur_test = data["cur_norm"][test]
        base_test = affine_base(cur_test, translation, inv_affine)
        base_train = affine_base(data["cur_norm"][train], translation, inv_affine)
        residual_train = data["ref_norm"][train] - base_train
        query_feat = data["cur_feat"][data["cur_idx"][test]]
        query_pos = data["cur_norm"][test]
        token_feat = data["cur_feat"][data["cur_idx"][train]]
        token_pos = data["cur_norm"][train]

        shuffled = residual_train.copy()
        perm = rng.permutation(len(shuffled))
        if len(perm) > 1 and np.all(perm == np.arange(len(perm))):
            perm = np.roll(perm, 1)
        shuffled = shuffled[perm]

        pred_same = cur_test.copy()
        pred_affine = base_test
        pred_correct = token_residual_prediction(query_feat, query_pos, base_test, token_feat, token_pos, residual_train, args.feature_weight, args.spatial_weight, args.attention_temperature)
        pred_spatial = token_residual_prediction(query_feat, query_pos, base_test, token_feat, token_pos, residual_train, 0.0, args.spatial_weight, args.attention_temperature)
        pred_shuffled = token_residual_prediction(query_feat, query_pos, base_test, token_feat, token_pos, shuffled, args.feature_weight, args.spatial_weight, args.attention_temperature)
        pred_oracle = data["ref_norm"][test]
        row = {
            "same_position": summarize_method("same_position", pred_same, test, data),
            "affine_only": summarize_method("affine_only", pred_affine, test, data),
            "correct_anchor_token": summarize_method("correct_anchor_token", pred_correct, test, data),
            "spatial_only_token": summarize_method("spatial_only_token", pred_spatial, test, data),
            "shuffled_value_token": summarize_method("shuffled_value_token", pred_shuffled, test, data),
            "oracle_anchor": summarize_method("oracle_anchor", pred_oracle, test, data),
        }
        if donor is not None and donor_res is not None and len(donor_res):
            donor_data = donor["data"]
            pred_wrong = token_residual_prediction(
                query_feat,
                query_pos,
                base_test,
                donor_data["cur_feat"][donor_data["cur_idx"]],
                donor_data["cur_norm"],
                donor_res,
                args.feature_weight,
                args.spatial_weight,
                args.attention_temperature,
            )
            row["wrong_boundary_token"] = summarize_method("wrong_boundary_token", pred_wrong, test, data)
        rows.append(row)
    return aggregate(rows)


def metric(summary, method, group="patch_error_median_over_anchors", field="median"):
    return summary.get(method, {}).get(group, {}).get(field)


def draw_summary_chart(path, results):
    methods = ["same_position", "affine_only", "correct_anchor_token", "spatial_only_token", "shuffled_value_token", "wrong_boundary_token", "oracle_anchor"]
    labels = ["same", "affine", "correct", "spatial", "shuffled", "wrong", "oracle"]
    width = 1480
    height = 140 + 92 * len(results)
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    cv2.putText(canvas, "AnchorToken specificity before decoder", (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, "red bars = median patch error; correct should beat affine and negative controls", (18, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (60, 60, 60), 1, cv2.LINE_AA)
    left = 330
    bar_area = width - left - 40
    for row_idx, result in enumerate(results):
        y0 = 122 + 92 * row_idx
        cv2.putText(canvas, Path(result["aabb_dir"]).name[:44], (18, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 2, cv2.LINE_AA)
        values = [metric(result["specificity"], m) for m in methods]
        valid = [v for v in values if v is not None]
        max_val = max(valid + [1.0])
        group_w = bar_area / len(methods)
        for i, (label, value) in enumerate(zip(labels, values)):
            x = left + int(i * group_w)
            cv2.putText(canvas, label, (x, y0 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (45, 45, 45), 1, cv2.LINE_AA)
            if value is None:
                cv2.putText(canvas, "NA", (x + 8, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 80), 1, cv2.LINE_AA)
                continue
            bar_w = max(16, int(group_w * 0.35))
            bar_h = int(np.clip(value / max_val, 0.0, 1.0) * 54)
            color = (70, 170, 70) if label == "correct" else (80, 80, 230)
            if label in {"shuffled", "wrong"}:
                color = (180, 120, 60)
            cv2.rectangle(canvas, (x + 4, y0 + 58 - bar_h), (x + 4 + bar_w, y0 + 58), color, -1)
            cv2.putText(canvas, f"{value:.2f}", (x + 2, y0 + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (25, 25, 25), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def load_samples(aabb_dirs, model, args):
    samples = []
    for aabb in aabb_dirs:
        aabb_dir, boundary_dir, summary = load_boundary_summary(aabb)
        anchors = summary.get("anchors", [])
        if len(anchors) < 4:
            print(f"Skipping {aabb_dir}: only {len(anchors)} anchors")
            continue
        print(f"Encoding boundary images for {aabb_dir} ({len(anchors)} anchors)")
        ref_info = encode_image(model, summary["ref"]["image"], args.size, args.device)
        cur_info = encode_image(model, summary["cur"]["image"], args.size, args.device)
        samples.append(
            {
                "aabb_dir": str(aabb_dir),
                "boundary_dir": str(boundary_dir),
                "summary": summary,
                "anchors": anchors,
                "data": build_data(summary, anchors, ref_info, cur_info),
            }
        )
    return samples


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    if args.out_dir is None:
        args.out_dir = str(REPO_ROOT / "output" / "rich_anchor_token_specificity")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("Loading original Human3R encoder...")
    model = load_model(args.model_path, device=args.device, verbose=True).eval()
    model.gradient_checkpointing = False
    samples = load_samples(args.aabb_dir, model, args)
    if not samples:
        raise RuntimeError("no valid samples to evaluate")

    results = []
    for i, sample in enumerate(samples):
        donor = samples[(i + 1) % len(samples)] if len(samples) > 1 else None
        print(f"Evaluating specificity for {sample['aabb_dir']}")
        specificity = evaluate_specificity(sample, donor, rng, args)
        aff = metric(specificity, "affine_only")
        corr = metric(specificity, "correct_anchor_token")
        shuf = metric(specificity, "shuffled_value_token")
        wrong = metric(specificity, "wrong_boundary_token")
        result = {
            "aabb_dir": sample["aabb_dir"],
            "boundary_dir": sample["boundary_dir"],
            "donor_aabb_dir": donor["aabb_dir"] if donor is not None else None,
            "n_anchor_tokens": int(len(sample["anchors"])),
            "mesh_geometry_inliers": int(sample["summary"].get("mesh_geometry_inliers", 0)),
            "unique_anchor_patch_pairs": int(sample["summary"].get("unique_anchor_patch_pairs", len(sample["anchors"]))),
            "specificity": specificity,
            "key_deltas": {
                "affine_minus_correct_patch_error": float(aff - corr) if aff is not None and corr is not None else None,
                "shuffled_minus_correct_patch_error": float(shuf - corr) if shuf is not None and corr is not None else None,
                "wrong_minus_correct_patch_error": float(wrong - corr) if wrong is not None and corr is not None else None,
            },
        }
        sample_out = out_dir / Path(sample["aabb_dir"]).name
        sample_out.mkdir(parents=True, exist_ok=True)
        (sample_out / "specificity_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        results.append(result)

    draw_summary_chart(out_dir / "anchor_token_specificity_summary.jpg", results)
    root = {
        "args": vars(args),
        "results": results,
        "visualizations": {
            "summary": str(out_dir / "anchor_token_specificity_summary.jpg"),
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
                    "same": metric(r["specificity"], "same_position"),
                    "affine": metric(r["specificity"], "affine_only"),
                    "correct": metric(r["specificity"], "correct_anchor_token"),
                    "spatial_only": metric(r["specificity"], "spatial_only_token"),
                    "shuffled": metric(r["specificity"], "shuffled_value_token"),
                    "wrong_boundary": metric(r["specificity"], "wrong_boundary_token"),
                    "oracle": metric(r["specificity"], "oracle_anchor"),
                    "deltas": r["key_deltas"],
                }
                for r in results
            ],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
