#!/usr/bin/env python3
"""Analyze AvatarReX shot labels and ShotTokenGenerator behavior."""

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dust3r.datasets.avatarrex import AvatarReX_AABB, AvatarReX_Video
from dust3r.datasets.utils.transforms import ImgNorm
from dust3r.model import ARCroco3DStereo


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", default=None, help="Model/checkpoint path. Omit for label-only checks.")
    parser.add_argument("--roots", nargs="+", default=[
        "/workspace/data/Avatarrex/avatarrex_zzr_output",
        "/workspace/data/Avatarrex/avatarrex_lbn1_output",
        "/workspace/data/Avatarrex/avatarrex_zxc_output",
    ])
    parser.add_argument("--num_samples", type=int, default=20, help="Samples per dataset/root.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    parser.add_argument("--quiet", action="store_true", help="Only print compact per-dataset status and overall summary.")
    return parser.parse_args()


def make_dataset(kind, root, seed):
    cls = AvatarReX_Video if kind == "video" else AvatarReX_AABB
    return cls(
        split="Training",
        ROOT=root,
        resolution=(512, 288),
        transform=ImgNorm,
        num_views=4,
        seed=seed,
        n_corres=0,
    )


def sample_indices(length, num_samples, rng):
    n = min(length, num_samples)
    if n <= 0:
        return []
    return rng.choice(length, size=n, replace=False).tolist()


def path_parts(instance_path):
    path = Path(instance_path)
    seq = path.parents[1].name
    frame = int(path.stem)
    return seq, frame


def validate_sample(kind, views):
    labels = [int(v["shot_label"]) for v in views]
    seqs, frames = zip(*(path_parts(v["instance"]) for v in views))
    frame_step_ok = all(frames[i] == frames[0] + i for i in range(len(frames)))

    if kind == "video":
        ok = labels == [0, 0, 0, 0] and len(set(seqs)) == 1 and frame_step_ok
    else:
        ok = (
            labels == [0, 0, 1, 0]
            and seqs[0] == seqs[1]
            and seqs[2] == seqs[3]
            and seqs[0] != seqs[2]
            and frame_step_ok
        )
    return {
        "ok": bool(ok),
        "labels": labels,
        "seqs": list(seqs),
        "frames": list(frames),
    }


def tensor_stats(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def auc_pairwise(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    total = len(pos) * len(neg)
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / total


def centroid_embedding_auc(rows, key):
    vectors = [row.get(key) for row in rows]
    labels = [int(row["shot_label"]) for row in rows]
    if not vectors or len(set(labels)) < 2:
        return None

    x = np.stack(vectors).astype(np.float64)
    x = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
    labels_np = np.asarray(labels)
    scores = []

    for i in range(len(rows)):
        pos_mask = labels_np == 1
        neg_mask = labels_np == 0
        if labels_np[i] == 1 and pos_mask.sum() > 1:
            pos_mask[i] = False
        if labels_np[i] == 0 and neg_mask.sum() > 1:
            neg_mask[i] = False
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return None

        pos_center = x[pos_mask].mean(axis=0)
        neg_center = x[neg_mask].mean(axis=0)
        pos_center = pos_center / max(np.linalg.norm(pos_center), 1e-12)
        neg_center = neg_center / max(np.linalg.norm(neg_center), 1e-12)
        scores.append(float(x[i].dot(pos_center) - x[i].dot(neg_center)))

    return auc_pairwise(scores, labels)


@torch.no_grad()
def analyze_features(model, device, views):
    imgs = torch.stack([v["img"] for v in views], dim=0).to(device, non_blocking=True)
    shapes = torch.as_tensor(
        np.stack([v["true_shape"] for v in views]), device=device, dtype=torch.int32
    )
    feat_ls, _, _ = model._encode_image(imgs, shapes)
    f_dec = model.decoder_embed(feat_ls[-1])
    g = f_dec.mean(dim=1)

    q_tokens = []
    for i in range(len(views)):
        if i == 0:
            q_tokens.append(model.shot_token_generator(f_dec[i:i + 1], f_dec[i:i + 1], i=0))
        else:
            q_tokens.append(model.shot_token_generator(f_dec[i:i + 1], f_dec[i - 1:i], i=i))

    rows = []
    for i in range(1, len(views)):
        q = q_tokens[i].reshape(-1)
        q_prev = q_tokens[i - 1].reshape(-1)
        rows.append({
            "view": i,
            "shot_label": int(views[i]["shot_label"]),
            "g_cos_prev": float(F.cosine_similarity(g[i:i + 1], g[i - 1:i], dim=-1).item()),
            "g_diff_norm": float(torch.linalg.norm(g[i] - g[i - 1]).item()),
            "q_norm": float(torch.linalg.norm(q).item()),
            "q_cos_prev": float(F.cosine_similarity(q[None], q_prev[None], dim=-1).item()),
            "q_delta_norm": float(torch.linalg.norm(q - q_prev).item()),
            "_q_vec": q.detach().float().cpu().numpy(),
        })
    return rows


def summarize_feature_rows(rows):
    summary = {}

    metric_keys = ["g_cos_prev", "g_diff_norm", "q_norm", "q_cos_prev", "q_delta_norm"]
    by_label = defaultdict(list)
    for row in rows:
        by_label[int(row["shot_label"])].append(row)

    for label, items in sorted(by_label.items()):
        summary[f"label_{label}"] = {
            key: tensor_stats([item[key] for item in items])
            for key in metric_keys
        }

    by_view_label = defaultdict(list)
    for row in rows:
        by_view_label[(int(row["view"]), int(row["shot_label"]))].append(row)
    summary["by_view_label"] = {
        f"view_{view}_label_{label}": {
            key: tensor_stats([item[key] for item in items])
            for key in metric_keys
        }
        for (view, label), items in sorted(by_view_label.items())
    }

    labels = [int(r["shot_label"]) for r in rows]
    summary["auc"] = {
        "neg_g_cos_prev": auc_pairwise([-r["g_cos_prev"] for r in rows], labels),
        "g_diff_norm": auc_pairwise([r["g_diff_norm"] for r in rows], labels),
        "q_norm": auc_pairwise([r["q_norm"] for r in rows], labels),
        "neg_q_cos_prev": auc_pairwise([-r["q_cos_prev"] for r in rows], labels),
        "q_delta_norm": auc_pairwise([r["q_delta_norm"] for r in rows], labels),
        "q_vec_centroid": centroid_embedding_auc(rows, "_q_vec"),
    }
    return summary


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = None
    if args.model_path:
        print(f"Loading model: {args.model_path}")
        model = ARCroco3DStereo.from_pretrained(args.model_path).to(device).eval()

    all_feature_rows = []
    dataset_summaries = []

    for root in args.roots:
        if not os.path.isdir(root):
            print(f"[skip] root not found: {root}")
            continue
        for kind in ["video", "aabb"]:
            print(f"\nDataset: {kind} root={root}")
            dataset = make_dataset(kind, root, args.seed)
            indices = sample_indices(len(dataset), args.num_samples, rng)
            label_patterns = Counter()
            invalid = []
            feature_rows = []

            for idx in indices:
                views = dataset[idx]
                check = validate_sample(kind, views)
                label_patterns[tuple(check["labels"])] += 1
                if not check["ok"]:
                    invalid.append({"idx": int(idx), **check})
                if model is not None:
                    rows = analyze_features(model, device, views)
                    for row in rows:
                        row.update({"dataset": kind, "root": root, "idx": int(idx)})
                    feature_rows.extend(rows)

            summary = {
                "kind": kind,
                "root": root,
                "num_checked": len(indices),
                "label_patterns": {str(list(k)): v for k, v in label_patterns.items()},
                "invalid_count": len(invalid),
                "invalid_examples": invalid[:5],
            }
            if feature_rows:
                summary["features"] = summarize_feature_rows(feature_rows)
            dataset_summaries.append(summary)
            all_feature_rows.extend(feature_rows)

            print(f"checked={len(indices)} invalid={len(invalid)} patterns={dict(label_patterns)}")
            if feature_rows and not args.quiet:
                print(json.dumps(summary["features"], indent=2, ensure_ascii=False))

    result = {
        "model_path": args.model_path,
        "num_samples_per_dataset": args.num_samples,
        "datasets": dataset_summaries,
    }
    if all_feature_rows:
        result["overall_features"] = summarize_feature_rows(all_feature_rows)
        print("\nOverall feature summary:")
        print(json.dumps(result["overall_features"], indent=2, ensure_ascii=False))

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
