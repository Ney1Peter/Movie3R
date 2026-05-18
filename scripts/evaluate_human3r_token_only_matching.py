#!/usr/bin/env python3
"""Evaluate whether Human3R tokens can discover anchor correspondences.

This script does not use XFeat to generate matches. It only uses previously
saved Step1 mesh-verified anchor pairs as evaluation targets, then performs
global nearest-neighbor matching between Human3R patch tokens.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from PIL.ImageOps import exif_transpose
import torchvision.transforms as tvf


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in [REPO_ROOT, SRC_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dust3r.model_human3r import load_model  # noqa: E402


IMG_NORM = tvf.Compose(
    [tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)


DEFAULT_SUMMARIES = [
    REPO_ROOT
    / "output"
    / "anchor_token_report_v1_guitar_only_step2_5"
    / "01_aabb_step1"
    / "BBQ_001_guitar_cam01_cam03_f00000005"
    / "pair_01_A_t1_to_B_t2_BOUNDARY"
    / "summary.json",
    REPO_ROOT
    / "output"
    / "anchor_token_report_v1_guitar_only_step2_5"
    / "01_aabb_step1"
    / "BBQ_001_guitar_cam06_cam07_f00000244"
    / "pair_01_A_t1_to_B_t2_BOUNDARY"
    / "summary.json",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        action="append",
        default=None,
        help="Step1 pair summary.json. Can be passed multiple times.",
    )
    parser.add_argument(
        "--model_path",
        default=str(REPO_ROOT / "src" / "human3r_896L.pth"),
    )
    parser.add_argument("--out_dir", default=str(REPO_ROOT / "output" / "human3r_token_only_matching_v1"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--margin_thresholds", default="0.02,0.05,0.10")
    return parser.parse_args()


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
    return tensor, torch.from_numpy(np.int32([[h2, w2]]))


def idx_to_xy(indices, grid_hw):
    idx = np.asarray(indices, dtype=np.int64)
    gh, gw = grid_hw
    y = idx // gw
    x = idx % gw
    return np.stack([x, y], axis=-1).astype(np.float32)


def stats(values):
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return None
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def topk_hit(sim_matrix, ref_idx, cur_idx, k):
    row = sim_matrix[ref_idx]
    score = row[cur_idx]
    rank = int((row > score).sum() + 1)
    return rank <= k, rank


def evaluate_ref_groups(sim_np, nn_cur, cur_best_ref, anchors, cur_grid_hw):
    groups = {}
    for anchor in anchors:
        ri = int(anchor["ref_patch_idx"])
        groups.setdefault(ri, set()).add(int(anchor["cur_patch_idx"]))

    exact_any = []
    within_1 = []
    within_2 = []
    errors = []
    mnn_any = []
    ranks = []
    top_hits = {"top1": [], "top5": [], "top10": [], "top32": []}
    per_group = []
    for ri in sorted(groups):
        gt_cur = sorted(groups[ri])
        pred = int(nn_cur[ri])
        gt_xy = idx_to_xy(gt_cur, cur_grid_hw)
        pred_xy = idx_to_xy([pred], cur_grid_hw)[0]
        err = float(np.linalg.norm(gt_xy - pred_xy[None], axis=1).min())
        gt_ranks = [int((sim_np[ri] > sim_np[ri, ci]).sum() + 1) for ci in gt_cur]
        best_rank = int(min(gt_ranks))
        ranks.append(best_rank)

        exact = pred in groups[ri]
        exact_any.append(exact)
        within_1.append(err <= 1.0)
        within_2.append(err <= 2.0)
        errors.append(err)
        mnn = any(int(cur_best_ref[ci]) == ri for ci in gt_cur)
        mnn_any.append(mnn)
        for key, k in [("top1", 1), ("top5", 5), ("top10", 10), ("top32", 32)]:
            top_hits[key].append(best_rank <= k)
        per_group.append(
            {
                "ref_patch_idx": int(ri),
                "gt_cur_patch_idx": [int(ci) for ci in gt_cur],
                "pred_cur_patch_idx": pred,
                "best_rank_true_cur": best_rank,
                "patch_error_to_nearest_gt": err,
                "exact_any": bool(exact),
                "true_pair_mnn_any": bool(mnn),
            }
        )

    return {
        "num_unique_ref_patches": int(len(groups)),
        "topk_true_cur_any_rate": {key: float(np.mean(vals)) if vals else None for key, vals in top_hits.items()},
        "true_cur_best_rank": stats(ranks),
        "nn_exact_any_rate": float(np.mean(exact_any)) if exact_any else None,
        "nn_within_1_patch_any_rate": float(np.mean(within_1)) if within_1 else None,
        "nn_within_2_patch_any_rate": float(np.mean(within_2)) if within_2 else None,
        "nn_patch_error_to_nearest_gt": stats(errors),
        "true_pair_mnn_any_rate": float(np.mean(mnn_any)) if mnn_any else None,
        "per_ref_group": per_group,
    }


def evaluate_feature(name, ref_tokens, cur_tokens, anchors, ref_grid_hw, cur_grid_hw, margin_thresholds):
    sim = (F.normalize(ref_tokens.float(), dim=-1) @ F.normalize(cur_tokens.float(), dim=-1).T)
    sim_np = sim.detach().cpu().numpy()

    top_vals, top_idx = torch.topk(sim, k=min(2, sim.shape[1]), dim=1)
    nn_cur = top_idx[:, 0].detach().cpu().numpy().astype(np.int64)
    nn_score = top_vals[:, 0].detach().cpu().numpy().astype(np.float32)
    if top_vals.shape[1] > 1:
        margins = (top_vals[:, 0] - top_vals[:, 1]).detach().cpu().numpy().astype(np.float32)
    else:
        margins = np.zeros((sim.shape[0],), dtype=np.float32)
    cur_best_ref = sim.argmax(dim=0).detach().cpu().numpy().astype(np.int64)

    ref_idx = np.array([int(a["ref_patch_idx"]) for a in anchors], dtype=np.int64)
    cur_idx = np.array([int(a["cur_patch_idx"]) for a in anchors], dtype=np.int64)
    pred_cur = nn_cur[ref_idx]
    pred_xy = idx_to_xy(pred_cur, cur_grid_hw)
    gt_xy = idx_to_xy(cur_idx, cur_grid_hw)
    patch_error = np.linalg.norm(pred_xy - gt_xy, axis=1)
    exact = pred_cur == cur_idx
    mnn = cur_best_ref[cur_idx] == ref_idx
    pred_is_mnn = cur_best_ref[pred_cur] == ref_idx

    ranks = []
    top_hits = {"top1": [], "top5": [], "top10": [], "top32": []}
    for ri, ci in zip(ref_idx, cur_idx):
        for key, k in [("top1", 1), ("top5", 5), ("top10", 10), ("top32", 32)]:
            hit, rank = topk_hit(sim_np, int(ri), int(ci), k)
            top_hits[key].append(hit)
        ranks.append(rank)

    margin_eval = {}
    for threshold in margin_thresholds:
        selected_all = margins >= threshold
        selected_gt = selected_all[ref_idx]
        margin_eval[f">={threshold:.2f}"] = {
            "all_ref_selected": int(selected_all.sum()),
            "gt_ref_selected": int(selected_gt.sum()),
            "gt_ref_selected_rate": float(selected_gt.mean()) if len(selected_gt) else None,
            "selected_exact_rate": float(exact[selected_gt].mean()) if selected_gt.any() else None,
            "selected_within_1_patch_rate": float((patch_error[selected_gt] <= 1.0).mean()) if selected_gt.any() else None,
            "selected_median_patch_error": float(np.median(patch_error[selected_gt])) if selected_gt.any() else None,
        }

    gt_pair_set = {(int(ri), int(ci)) for ri, ci in zip(ref_idx, cur_idx)}
    all_mnn_pairs = []
    for ri, ci in enumerate(nn_cur):
        if cur_best_ref[ci] == ri:
            all_mnn_pairs.append((int(ri), int(ci)))
    exact_mnn_pairs = [p for p in all_mnn_pairs if p in gt_pair_set]

    grouped_by_ref = evaluate_ref_groups(sim_np, nn_cur, cur_best_ref, anchors, cur_grid_hw)

    return {
        "feature": name,
        "num_ref_tokens": int(sim.shape[0]),
        "num_cur_tokens": int(sim.shape[1]),
        "num_gt_anchors": int(len(anchors)),
        "topk_true_cur_rate": {key: float(np.mean(vals)) if vals else None for key, vals in top_hits.items()},
        "true_cur_rank": stats(ranks),
        "nn_exact_rate": float(exact.mean()) if len(exact) else None,
        "nn_within_1_patch_rate": float((patch_error <= 1.0).mean()) if len(patch_error) else None,
        "nn_within_2_patch_rate": float((patch_error <= 2.0).mean()) if len(patch_error) else None,
        "nn_patch_error": stats(patch_error),
        "nn_score": stats(nn_score[ref_idx]),
        "nn_margin": stats(margins[ref_idx]),
        "true_pair_mnn_rate": float(mnn.mean()) if len(mnn) else None,
        "pred_pair_mnn_rate": float(pred_is_mnn.mean()) if len(pred_is_mnn) else None,
        "all_mnn_pairs": int(len(all_mnn_pairs)),
        "all_mnn_pairs_exact_gt": int(len(exact_mnn_pairs)),
        "all_mnn_exact_gt_precision_sparse": float(len(exact_mnn_pairs) / len(all_mnn_pairs)) if all_mnn_pairs else None,
        "margin_eval": margin_eval,
        "grouped_by_ref": grouped_by_ref,
        "per_anchor": [
            {
                "ref_patch_idx": int(ri),
                "gt_cur_patch_idx": int(ci),
                "pred_cur_patch_idx": int(pi),
                "rank_true_cur": int(rank),
                "patch_error": float(err),
                "exact": bool(ex),
                "true_pair_mnn": bool(mm),
                "pred_pair_mnn": bool(pm),
                "nn_score": float(score),
                "nn_margin": float(margin),
            }
            for ri, ci, pi, rank, err, ex, mm, pm, score, margin in zip(
                ref_idx,
                cur_idx,
                pred_cur,
                ranks,
                patch_error,
                exact,
                mnn,
                pred_is_mnn,
                nn_score[ref_idx],
                margins[ref_idx],
            )
        ],
    }


def write_markdown(path, results):
    lines = [
        "# Human3R Token-Only Matching Evaluation",
        "",
        "Only Human3R tokens are used for matching. Step1 mesh-verified anchors are evaluation targets.",
        "",
        # **========== Original table: per-anchor metrics only ==========**
        # "| sample | feature | anchors | top1 | top5 | top10 | top32 | exact NN | <=1 patch | <=2 patch | median err | MNN true |",
        # "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        # **========== End ==========**
        "| sample | feature | anchors | ref groups | top1 | top5 | exact NN | <=1 patch | median err | group top1 | group top5 | group exact | group <=1 | group median err |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for sample in results:
        sample_name = sample["sample"]
        for item in sample["features"]:
            top = item["topk_true_cur_rate"]
            err = item["nn_patch_error"] or {}
            group = item["grouped_by_ref"]
            group_top = group["topk_true_cur_any_rate"]
            group_err = group["nn_patch_error_to_nearest_gt"] or {}
            lines.append(
                "| "
                + " | ".join(
                    [
                        sample_name,
                        item["feature"],
                        str(item["num_gt_anchors"]),
                        str(group["num_unique_ref_patches"]),
                        f"{top['top1']:.3f}",
                        f"{top['top5']:.3f}",
                        f"{item['nn_exact_rate']:.3f}",
                        f"{item['nn_within_1_patch_rate']:.3f}",
                        f"{err.get('median', float('nan')):.2f}",
                        f"{group_top['top1']:.3f}",
                        f"{group_top['top5']:.3f}",
                        f"{group['nn_exact_any_rate']:.3f}",
                        f"{group['nn_within_1_patch_any_rate']:.3f}",
                        f"{group_err.get('median', float('nan')):.2f}",
                    ]
                )
                + " |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    summary_paths = [Path(p) for p in args.summary] if args.summary else DEFAULT_SUMMARIES
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    margin_thresholds = [float(x) for x in args.margin_thresholds.split(",") if x.strip()]

    model = load_model(args.model_path, device=device, verbose=True).eval()
    model.gradient_checkpointing = False
    patch_size = int(model.croco_args["patch_size"])

    results = []
    with torch.no_grad():
        for summary_path in summary_paths:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            anchors = summary["anchors"]
            sample = summary_path.parent.parent.name
            pair_name = summary.get("pair_name", summary_path.parent.name)
            ref_img = summary["ref"]["image"]
            cur_img = summary["cur"]["image"]

            ref_tensor, ref_true_shape = load_human3r_image(ref_img, args.size)
            cur_tensor, cur_true_shape = load_human3r_image(cur_img, args.size)
            ref_tensor = ref_tensor.to(device)
            cur_tensor = cur_tensor.to(device)
            ref_true_shape = ref_true_shape.to(device)
            cur_true_shape = cur_true_shape.to(device)

            ref_feat = model._encode_image(ref_tensor, ref_true_shape)[0][-1][0]
            cur_feat = model._encode_image(cur_tensor, cur_true_shape)[0][-1][0]
            ref_dec = model.decoder_embed(ref_feat[None])[0]
            cur_dec = model.decoder_embed(cur_feat[None])[0]

            h_ref, w_ref = map(int, ref_true_shape[0].detach().cpu().numpy().tolist())
            h_cur, w_cur = map(int, cur_true_shape[0].detach().cpu().numpy().tolist())
            ref_grid_hw = (h_ref // patch_size, w_ref // patch_size)
            cur_grid_hw = (h_cur // patch_size, w_cur // patch_size)

            features = [
                evaluate_feature(
                    "encoder",
                    ref_feat,
                    cur_feat,
                    anchors,
                    ref_grid_hw,
                    cur_grid_hw,
                    margin_thresholds,
                ),
                evaluate_feature(
                    "decoder_embed",
                    ref_dec,
                    cur_dec,
                    anchors,
                    ref_grid_hw,
                    cur_grid_hw,
                    margin_thresholds,
                ),
            ]
            result = {
                "sample": sample,
                "pair_name": pair_name,
                "summary_path": str(summary_path),
                "ref": summary["ref"],
                "cur": summary["cur"],
                "patch_size": int(patch_size),
                "ref_grid_hw": list(ref_grid_hw),
                "cur_grid_hw": list(cur_grid_hw),
                "features": features,
            }
            results.append(result)
            (out_dir / f"{sample}_{pair_name}.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )

    combined = {"args": vars(args), "results": results}
    (out_dir / "summary.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    write_markdown(out_dir / "summary.md", results)
    print(json.dumps({"out_dir": str(out_dir), "results": results}, indent=2))


if __name__ == "__main__":
    main()
