#!/usr/bin/env python3
"""Validate top-K / quality-gated AnchorToken selection strategies.

The previous prototype showed that global affine + local AnchorToken residual can
improve held-out reference lookup when enough anchors are available. This script
answers the next practical question:

    At inference time, can we keep only a small set of AnchorTokens?

It compares confidence top-K, spatially diverse top-K, and random-K token banks.
For each selected bank, the script predicts all unselected anchors and measures
held-out patch error.
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

from prototype_rich_anchor_tokens import (  # noqa: E402
    aggregate_method_rows,
    build_data,
    encode_image,
    evaluate_split,
    load_boundary_summary,
    load_model,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aabb_dir", action="append", required=True, help="AABB Step1 output directory. Can be repeated.")
    parser.add_argument("--model_path", default=str(REPO_ROOT / "src" / "human3r_896L.pth"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--k_values", default="4,8,16,32,64")
    parser.add_argument("--random_trials", type=int, default=256)
    parser.add_argument("--feature_weight", type=float, default=2.0)
    parser.add_argument("--spatial_weight", type=float, default=30.0)
    parser.add_argument("--attention_temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", default=None)
    return parser.parse_args()


def parse_ints(text):
    out = []
    for item in text.split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    return out


def top_confidence_indices(data, k):
    order = np.argsort(-data["weights"])
    return order[:k].astype(np.int64)


def diverse_confidence_indices(data, k):
    order = np.argsort(-data["weights"])
    selected = []
    cur_pos = data["cur_norm"]
    # Greedy confidence-first diversity. The threshold relaxes until K tokens are selected.
    for min_dist in [0.18, 0.14, 0.10, 0.06, 0.0]:
        selected = []
        for idx in order:
            if len(selected) >= k:
                break
            if not selected:
                selected.append(int(idx))
                continue
            d = np.linalg.norm(cur_pos[int(idx)] - cur_pos[np.array(selected, dtype=np.int64)], axis=1)
            if float(d.min()) >= min_dist:
                selected.append(int(idx))
        if len(selected) >= k:
            break
    if len(selected) < k:
        for idx in order:
            if int(idx) not in selected:
                selected.append(int(idx))
                if len(selected) >= k:
                    break
    return np.array(selected[:k], dtype=np.int64)


def test_for_indices(data, selected, args):
    n = len(data["ref_norm"])
    selected = np.asarray(selected, dtype=np.int64)
    mask = np.ones(n, dtype=bool)
    mask[selected] = False
    test = np.flatnonzero(mask)
    if len(selected) < 3 or len(test) == 0:
        return None
    return evaluate_split(selected, test, data, args)


def compact_metrics(row):
    if row is None:
        return None
    out = {}
    for method in ["same_position", "translation", "affine", "anchor_token_soft", "anchor_token_affine_residual", "oracle_anchor"]:
        method_row = row.get(method)
        if not method_row:
            continue
        out[method] = {
            "patch_error_median": method_row["patch_error"]["median"] if method_row.get("patch_error") else None,
            "patch_error_mean": method_row["patch_error"]["mean"] if method_row.get("patch_error") else None,
            "token_cosine_median": method_row["token_cosine"]["median"] if method_row.get("token_cosine") else None,
            "top10_rate": method_row.get("top10_rate"),
            "within_1_patch_rate": method_row.get("within_1_patch_rate"),
            "within_2_patch_rate": method_row.get("within_2_patch_rate"),
        }
    return out


def compact_random_metrics(agg):
    if not agg:
        return None
    out = {}
    for method in ["same_position", "translation", "affine", "anchor_token_soft", "anchor_token_affine_residual", "oracle_anchor"]:
        method_row = agg.get(method)
        if not method_row:
            continue
        err = method_row.get("patch_error_median_over_trials", {})
        cos = method_row.get("token_cosine_median_over_trials", {})
        out[method] = {
            "patch_error_median": err.get("median"),
            "patch_error_p75": err.get("p75"),
            "token_cosine_median": cos.get("median"),
        }
    return out


def random_strategy(data, k, trials, rng, args):
    n = len(data["ref_norm"])
    if n <= k or k < 3:
        return None
    rows = []
    for _ in range(trials):
        selected = rng.choice(n, size=k, replace=False)
        row = test_for_indices(data, selected, args)
        if row is not None:
            rows.append(row)
    return aggregate_method_rows(rows) if rows else None


def quality_gate(n_tokens, residual_error=None):
    count_gate = float(np.clip((n_tokens - 4.0) / 12.0, 0.0, 1.0))
    if residual_error is None:
        return count_gate
    residual_gate = float(np.clip(1.0 - residual_error / 4.0, 0.0, 1.0))
    return count_gate * residual_gate


def strategy_improvement(metrics):
    if not metrics:
        return None
    affine = metrics.get("affine", {}).get("patch_error_median")
    token = metrics.get("anchor_token_affine_residual", {}).get("patch_error_median")
    if affine is None or token is None:
        return None
    return float(affine - token)


def analyze_one(aabb_dir, model, args, k_values, rng, out_dir):
    aabb_dir, boundary_dir, summary = load_boundary_summary(aabb_dir)
    anchors = summary.get("anchors", [])
    if len(anchors) < 3:
        raise RuntimeError(f"need at least 3 anchors for {aabb_dir}")
    ref_info = encode_image(model, summary["ref"]["image"], args.size, args.device)
    cur_info = encode_image(model, summary["cur"]["image"], args.size, args.device)
    data = build_data(summary, anchors, ref_info, cur_info)
    n = len(anchors)

    results = []
    for k in k_values:
        if k >= n or k < 3:
            continue
        conf_idx = top_confidence_indices(data, k)
        div_idx = diverse_confidence_indices(data, k)
        conf_row = compact_metrics(test_for_indices(data, conf_idx, args))
        div_row = compact_metrics(test_for_indices(data, div_idx, args))
        rand_agg = compact_random_metrics(random_strategy(data, k, args.random_trials, rng, args))
        results.append(
            {
                "k": int(k),
                "heldout_count": int(n - k),
                "confidence_topk": conf_row,
                "diverse_topk": div_row,
                "random_k": rand_agg,
                "confidence_improvement_vs_affine": strategy_improvement(conf_row),
                "diverse_improvement_vs_affine": strategy_improvement(div_row),
                "random_improvement_vs_affine": strategy_improvement(rand_agg),
                "quality_gate_count_only": quality_gate(k),
            }
        )

    best = choose_best(results)
    sample_out = out_dir / aabb_dir.name
    sample_out.mkdir(parents=True, exist_ok=True)
    draw_sample_chart(sample_out / "anchor_token_selection_chart.jpg", results)
    result = {
        "aabb_dir": str(aabb_dir),
        "boundary_dir": str(boundary_dir),
        "n_anchor_tokens": int(n),
        "mesh_geometry_inliers": int(summary.get("mesh_geometry_inliers", 0)),
        "unique_anchor_patch_pairs": int(summary.get("unique_anchor_patch_pairs", n)),
        "k_results": results,
        "best_strategy": best,
        "recommended_gate": {
            "strong_enable": bool(n >= 16),
            "weak_enable": bool(8 <= n < 16),
            "fallback": bool(n < 8),
        },
        "visualizations": {
            "selection_chart": str(sample_out / "anchor_token_selection_chart.jpg"),
        },
    }
    (sample_out / "anchor_token_selection_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def choose_best(results):
    candidates = []
    for r in results:
        for strategy in ["confidence_topk", "diverse_topk", "random_k"]:
            metrics = r.get(strategy)
            if not metrics:
                continue
            token_err = metrics.get("anchor_token_affine_residual", {}).get("patch_error_median")
            affine_err = metrics.get("affine", {}).get("patch_error_median")
            if token_err is None:
                continue
            candidates.append(
                {
                    "k": r["k"],
                    "strategy": strategy,
                    "token_residual_error": token_err,
                    "affine_error": affine_err,
                    "improvement_vs_affine": None if affine_err is None else float(affine_err - token_err),
                }
            )
    if not candidates:
        return None
    candidates.sort(key=lambda x: x["token_residual_error"])
    return candidates[0]


def draw_sample_chart(path, results):
    width, height = 1180, 460
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    cv2.putText(canvas, "Top-K AnchorToken selection: held-out patch error", (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (20, 20, 20), 2, cv2.LINE_AA)
    if not results:
        cv2.imwrite(str(path), canvas)
        return
    labels = [str(r["k"]) for r in results]
    series = {
        "conf_affine": [],
        "conf_token": [],
        "div_token": [],
        "rand_token": [],
    }
    for r in results:
        series["conf_affine"].append(value(r, "confidence_topk", "affine"))
        series["conf_token"].append(value(r, "confidence_topk", "anchor_token_affine_residual"))
        series["div_token"].append(value(r, "diverse_topk", "anchor_token_affine_residual"))
        series["rand_token"].append(value(r, "random_k", "anchor_token_affine_residual"))
    all_vals = [v for vals in series.values() for v in vals if v is not None]
    max_val = max(all_vals + [1.0])
    left, top, plot_w, plot_h = 72, 78, 1040, 270
    cv2.rectangle(canvas, (left, top), (left + plot_w, top + plot_h), (220, 220, 220), 1)
    colors = {
        "conf_affine": (90, 90, 90),
        "conf_token": (40, 170, 255),
        "div_token": (80, 180, 80),
        "rand_token": (230, 80, 80),
    }
    names = {
        "conf_affine": "topK affine",
        "conf_token": "topK token-residual",
        "div_token": "diverse token-residual",
        "rand_token": "random token-residual",
    }
    group_w = plot_w / max(len(labels), 1)
    bar_w = max(10, int(group_w * 0.12))
    for i, label in enumerate(labels):
        x_base = left + int(i * group_w + group_w * 0.16)
        for j, key in enumerate(["conf_affine", "conf_token", "div_token", "rand_token"]):
            val = series[key][i]
            if val is None:
                continue
            h = int(val / max_val * plot_h)
            x0 = x_base + j * (bar_w + 6)
            cv2.rectangle(canvas, (x0, top + plot_h - h), (x0 + bar_w, top + plot_h), colors[key], -1)
            cv2.putText(canvas, f"{val:.2f}", (x0 - 8, max(top + 14, top + plot_h - h - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (30, 30, 30), 1, cv2.LINE_AA)
        cv2.putText(canvas, label, (x_base, height - 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 1, cv2.LINE_AA)
    y = height - 24
    x = left
    for key in ["conf_affine", "conf_token", "div_token", "rand_token"]:
        cv2.rectangle(canvas, (x, y - 14), (x + 16, y), colors[key], -1)
        cv2.putText(canvas, names[key], (x + 22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 40, 40), 1, cv2.LINE_AA)
        x += 250
    cv2.imwrite(str(path), canvas)


def value(row, strategy, method):
    metrics = row.get(strategy) or {}
    method_metrics = metrics.get(method) or {}
    return method_metrics.get("patch_error_median")


def draw_root_summary(path, summaries):
    width, height = 1380, 130 + 88 * len(summaries)
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    cv2.putText(canvas, "AnchorToken top-K / quality-gate validation", (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.88, (20, 20, 20), 2, cv2.LINE_AA)
    headers = ["sample", "tokens", "gate", "best k", "strategy", "token err", "affine err", "improve"]
    xs = [18, 500, 610, 710, 820, 980, 1110, 1250]
    for x, text in zip(xs, headers):
        cv2.putText(canvas, text, (x, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (50, 50, 50), 2, cv2.LINE_AA)
    for i, s in enumerate(summaries):
        y = 130 + 88 * i
        best = s.get("best_strategy") or {}
        gate = "fallback" if s["recommended_gate"]["fallback"] else "weak" if s["recommended_gate"]["weak_enable"] else "strong"
        vals = [
            str(s["n_anchor_tokens"]),
            gate,
            str(best.get("k")),
            str(best.get("strategy")),
            fmt(best.get("token_residual_error")),
            fmt(best.get("affine_error")),
            fmt(best.get("improvement_vs_affine")),
        ]
        cv2.putText(canvas, Path(s["aabb_dir"]).name[:60], (xs[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (20, 20, 20), 2, cv2.LINE_AA)
        for x, val in zip(xs[1:], vals):
            color = (40, 130, 40) if val == "strong" else (30, 100, 220) if val == "weak" else (20, 20, 20)
            cv2.putText(canvas, val, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.54, color, 2, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def fmt(x):
    return "None" if x is None else f"{float(x):.2f}"


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    rng = np.random.default_rng(args.seed)
    k_values = parse_ints(args.k_values)
    if args.out_dir is None:
        args.out_dir = str(REPO_ROOT / "output" / "rich_anchor_token_selection")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading original Human3R encoder...")
    model = load_model(args.model_path, device=args.device, verbose=True).eval()
    model.gradient_checkpointing = False

    summaries = []
    for aabb_dir in args.aabb_dir:
        print(f"Validating AnchorToken selection for {aabb_dir}...")
        summaries.append(analyze_one(Path(aabb_dir), model, args, k_values, rng, out_dir))

    draw_root_summary(out_dir / "anchor_token_selection_summary.jpg", summaries)
    root = {
        "args": vars(args),
        "summaries": summaries,
        "visualizations": {
            "summary": str(out_dir / "anchor_token_selection_summary.jpg"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(root, indent=2), encoding="utf-8")
    print(json.dumps(
        {
            "out_dir": str(out_dir),
            "samples": [
                {
                    "name": Path(s["aabb_dir"]).name,
                    "tokens": s["n_anchor_tokens"],
                    "gate": "fallback" if s["recommended_gate"]["fallback"] else "weak" if s["recommended_gate"]["weak_enable"] else "strong",
                    "best_strategy": s["best_strategy"],
                }
                for s in summaries
            ],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
