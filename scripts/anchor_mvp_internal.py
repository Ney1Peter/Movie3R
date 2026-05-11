#!/usr/bin/env python3
"""Offline MVP for matching static background anchors with internal encoder tokens.

This script does not change the model forward path. It probes whether the frozen
Human3R/Movie3R encoder tokens already contain enough evidence to match static
background patches across a shot boundary.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dust3r.datasets.avatarrex import AvatarReX_AABB  # noqa: E402
from dust3r.datasets.utils.transforms import ImgNorm  # noqa: E402
from dust3r.model import load_model  # noqa: E402


DEFAULT_MODEL = (
    REPO_ROOT
    / "experiments/training-4gpu-bz24-30ep-shot-v5_1-last2-20260510-095305/checkpoint-final.pth"
)
DEFAULT_ROOT = Path("/workspace/data/avatarrex_zzr_output")
DEFAULT_OUTPUT = REPO_ROOT / "output/anchor_mvp_internal/a5b5_0304_0305"


class FeatureProjectionMLP(nn.Module):
    """Optional untrained projection hook for later anchor-detector experiments."""

    def __init__(self, dim):
        super().__init__()
        hidden = max(128, dim // 2)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        return self.net(x)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", default=str(DEFAULT_MODEL))
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--split", default="Training")
    parser.add_argument("--ref_seq", default="22010708", help="Reference frame sequence, drawn on the left.")
    parser.add_argument("--ref_frame", type=int, default=304)
    parser.add_argument("--cur_seq", default="22010710", help="Current frame sequence, drawn on the right.")
    parser.add_argument("--cur_frame", type=int, default=305)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--topk", type=int, default=128, help="Background candidates per frame.")
    parser.add_argument("--max_matches", type=int, default=64)
    parser.add_argument("--sim_thresh", type=float, default=0.30)
    parser.add_argument("--inlier_px", type=float, default=32.0)
    parser.add_argument("--depth_rel_thresh", type=float, default=0.20)
    parser.add_argument("--projection", choices=["identity", "random_mlp", "decoder_embed"], default="identity")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def to_numpy_mask(mask, height, width):
    if mask is False or mask is None:
        return np.zeros((height, width), dtype=np.float32)
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    if mask.ndim == 3:
        mask = mask[..., 0]
    return (mask.astype(np.float32) > 0.5).astype(np.float32)


def load_processed_view(dataset, seq, frame, view_idx):
    split_path = Path(dataset.ROOT) / dataset.split
    view = dataset._load_view(
        str(split_path),
        seq,
        0,
        frame,
        annots=[],
        resolution=(dataset._resolutions[0][0], dataset._resolutions[0][1]),
        rng=np.random.default_rng(0),
        v=view_idx,
        shot_label=0,
    )

    image = view["img"]
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image = image.convert("RGB")
    width, height = image.size

    view["rgb_pil"] = image
    view["true_shape"] = np.int32((height, width))
    view["img"] = ImgNorm(image)[None]
    view["human_mask"] = to_numpy_mask(view.get("msk"), height, width)
    view["depthmap"] = np.asarray(view["depthmap"], dtype=np.float32)
    return view


@torch.no_grad()
def extract_features(model, views, device, projection):
    imgs = torch.cat([view["img"] for view in views], dim=0).to(device, non_blocking=True)
    shapes = torch.as_tensor(
        np.stack([view["true_shape"] for view in views]), device=device, dtype=torch.int32
    )
    feat_ls, pos, _ = model._encode_image(imgs, shapes)
    feats = feat_ls[-1]

    if projection == "decoder_embed":
        feats = model.decoder_embed(feats)
    elif projection == "random_mlp":
        torch.manual_seed(0)
        mlp = FeatureProjectionMLP(feats.shape[-1]).to(device).eval()
        feats = mlp(feats)

    feats = F.normalize(feats.float(), dim=-1)
    return feats.cpu(), pos.cpu()


def infer_patch_grid(num_tokens, height, width):
    patch_size = None
    for candidate in (16, 14, 8):
        gh = height // candidate
        gw = width // candidate
        if gh * gw == num_tokens:
            patch_size = candidate
            return gh, gw, patch_size

    aspect = width / max(height, 1)
    gh = int(round(math.sqrt(num_tokens / aspect)))
    gh = max(1, gh)
    gw = max(1, num_tokens // gh)
    if gh * gw != num_tokens:
        raise ValueError(f"Cannot infer patch grid for {num_tokens=} and image {width}x{height}")
    patch_size = max(1, height // gh)
    return gh, gw, patch_size


def adaptive_avg(array, grid_h, grid_w):
    tensor = torch.as_tensor(array, dtype=torch.float32)[None, None]
    return F.adaptive_avg_pool2d(tensor, (grid_h, grid_w))[0, 0].numpy()


def adaptive_max(array, grid_h, grid_w):
    tensor = torch.as_tensor(array, dtype=torch.float32)[None, None]
    return F.adaptive_max_pool2d(tensor, (grid_h, grid_w))[0, 0].numpy()


def image_gradient_score(image, grid_h, grid_w):
    rgb = np.asarray(image).astype(np.float32) / 255.0
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    grad = np.zeros_like(gray, dtype=np.float32)
    grad[:, 1:] += np.abs(gray[:, 1:] - gray[:, :-1])
    grad[1:, :] += np.abs(gray[1:, :] - gray[:-1, :])
    patch_grad = adaptive_avg(grad, grid_h, grid_w)
    lo, hi = np.percentile(patch_grad, [5, 95])
    if hi <= lo:
        return np.zeros_like(patch_grad, dtype=np.float32)
    return np.clip((patch_grad - lo) / (hi - lo), 0.0, 1.0)


def candidate_scores(view, grid_h, grid_w):
    human = adaptive_max(view["human_mask"], grid_h, grid_w)
    depth_valid = adaptive_avg((view["depthmap"] > 0).astype(np.float32), grid_h, grid_w)
    texture = image_gradient_score(view["rgb_pil"], grid_h, grid_w)

    bg = human < 0.05
    valid = depth_valid > 0.25
    score = (0.10 + texture) * bg.astype(np.float32) * valid.astype(np.float32)
    return {
        "score": score,
        "human": human,
        "depth_valid": depth_valid,
        "texture": texture,
        "bg": bg,
        "valid": valid,
    }


def topk_indices(score, topk):
    flat = score.reshape(-1)
    valid = np.flatnonzero(flat > 0)
    if valid.size == 0:
        return np.zeros((0,), dtype=np.int64)
    order = valid[np.argsort(flat[valid])[::-1]]
    return order[:topk].astype(np.int64)


def mutual_matches(feats_ref, feats_cur, idx_ref, idx_cur, sim_thresh, max_matches):
    if len(idx_ref) == 0 or len(idx_cur) == 0:
        return []

    ref = feats_ref[idx_ref]
    cur = feats_cur[idx_cur]
    sim = cur @ ref.T
    cur_best_ref = sim.argmax(dim=1)
    ref_best_cur = sim.argmax(dim=0)

    matches = []
    for cur_pos, ref_pos in enumerate(cur_best_ref.tolist()):
        if ref_best_cur[ref_pos].item() != cur_pos:
            continue
        score = float(sim[cur_pos, ref_pos].item())
        if score < sim_thresh:
            continue
        matches.append({
            "idx_ref": int(idx_ref[ref_pos]),
            "idx_cur": int(idx_cur[cur_pos]),
            "similarity": score,
        })

    matches.sort(key=lambda item: item["similarity"], reverse=True)
    return matches[:max_matches]


def patch_center(index, grid_w, patch_size, width, height):
    row = int(index // grid_w)
    col = int(index % grid_w)
    x = min(width - 1.0, (col + 0.5) * patch_size)
    y = min(height - 1.0, (row + 0.5) * patch_size)
    return x, y, row, col


def patch_median_depth(depth, row, col, patch_size):
    y0 = row * patch_size
    x0 = col * patch_size
    patch = depth[y0:y0 + patch_size, x0:x0 + patch_size]
    vals = patch[np.isfinite(patch) & (patch > 0)]
    if vals.size == 0:
        return None
    return float(np.median(vals))


def project_cur_to_ref(cur_view, ref_view, x_cur, y_cur, depth_cur):
    k_cur = cur_view["camera_intrinsics"]
    k_ref = ref_view["camera_intrinsics"]
    c2w_cur = cur_view["camera_pose"]
    c2w_ref = ref_view["camera_pose"]

    pix = np.array([x_cur, y_cur, 1.0], dtype=np.float32)
    xyz_cur = np.linalg.inv(k_cur) @ pix * depth_cur
    xyz_cur_h = np.concatenate([xyz_cur, np.ones(1, dtype=np.float32)])
    xyz_world = c2w_cur @ xyz_cur_h
    xyz_ref = np.linalg.inv(c2w_ref) @ xyz_world
    z_ref = float(xyz_ref[2])
    if z_ref <= 1e-6:
        return None
    pix_ref = k_ref @ xyz_ref[:3]
    x_ref = float(pix_ref[0] / pix_ref[2])
    y_ref = float(pix_ref[1] / pix_ref[2])
    return x_ref, y_ref, z_ref


def evaluate_matches(matches, ref_view, cur_view, grid_h, grid_w, patch_size, inlier_px, depth_rel_thresh):
    width, height = ref_view["rgb_pil"].size
    evaluated = []

    for match in matches:
        x_ref, y_ref, row_ref, col_ref = patch_center(match["idx_ref"], grid_w, patch_size, width, height)
        x_cur, y_cur, row_cur, col_cur = patch_center(match["idx_cur"], grid_w, patch_size, width, height)

        depth_cur = patch_median_depth(cur_view["depthmap"], row_cur, col_cur, patch_size)
        depth_ref = patch_median_depth(ref_view["depthmap"], row_ref, col_ref, patch_size)
        reproj = None
        depth_rel = None
        inlier = False
        in_image = False

        if depth_cur is not None:
            projected = project_cur_to_ref(cur_view, ref_view, x_cur, y_cur, depth_cur)
            if projected is not None:
                px, py, z_ref = projected
                in_image = 0 <= px < width and 0 <= py < height
                reproj = float(math.hypot(px - x_ref, py - y_ref))
                if depth_ref is not None:
                    depth_rel = float(abs(z_ref - depth_ref) / max(abs(depth_ref), 1e-6))
                depth_ok = depth_rel is None or depth_rel < depth_rel_thresh
                inlier = bool(in_image and reproj < inlier_px and depth_ok)

        item = dict(match)
        item.update({
            "ref_xy": [float(x_ref), float(y_ref)],
            "cur_xy": [float(x_cur), float(y_cur)],
            "ref_rc": [int(row_ref), int(col_ref)],
            "cur_rc": [int(row_cur), int(col_cur)],
            "depth_cur": depth_cur,
            "depth_ref": depth_ref,
            "reproj_error_px": reproj,
            "depth_rel_error": depth_rel,
            "in_image": bool(in_image),
            "inlier": bool(inlier),
        })
        evaluated.append(item)

    return evaluated


def summarize_matches(matches):
    sims = [m["similarity"] for m in matches]
    reproj = [m["reproj_error_px"] for m in matches if m["reproj_error_px"] is not None]
    inliers = [m for m in matches if m["inlier"]]
    return {
        "num_matches": len(matches),
        "num_inliers": len(inliers),
        "inlier_ratio": float(len(inliers) / len(matches)) if matches else 0.0,
        "similarity_mean": float(np.mean(sims)) if sims else None,
        "similarity_max": float(np.max(sims)) if sims else None,
        "reproj_error_mean_px": float(np.mean(reproj)) if reproj else None,
        "reproj_error_median_px": float(np.median(reproj)) if reproj else None,
    }


def draw_matches(ref_view, cur_view, matches, output_path):
    ref_img = ref_view["rgb_pil"].copy()
    cur_img = cur_view["rgb_pil"].copy()
    width, height = ref_img.size
    canvas = Image.new("RGB", (width * 2, height), color=(0, 0, 0))
    canvas.paste(ref_img, (0, 0))
    canvas.paste(cur_img, (width, 0))
    draw = ImageDraw.Draw(canvas)

    for match in matches:
        x_ref, y_ref = match["ref_xy"]
        x_cur, y_cur = match["cur_xy"]
        x_cur += width
        color = (0, 220, 0) if match["inlier"] else (230, 60, 60)
        draw.line([(x_ref, y_ref), (x_cur, y_cur)], fill=color, width=2)
        r = 4
        draw.ellipse((x_ref - r, y_ref - r, x_ref + r, y_ref + r), outline=color, width=2)
        draw.ellipse((x_cur - r, y_cur - r, x_cur + r, y_cur + r), outline=color, width=2)

    canvas.save(output_path)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = AvatarReX_AABB(
        split=args.split,
        ROOT=args.root,
        resolution=(args.width, args.height),
        transform=ImgNorm,
        aug_crop=1,
        num_views=4,
        seed=args.seed,
        n_corres=0,
    )
    ref_view = load_processed_view(dataset, args.ref_seq, args.ref_frame, view_idx=0)
    cur_view = load_processed_view(dataset, args.cur_seq, args.cur_frame, view_idx=1)

    model = load_model(args.model_path, args.device, verbose=True).eval()
    feats, pos = extract_features(model, [ref_view, cur_view], args.device, args.projection)

    num_tokens = feats.shape[1]
    grid_h, grid_w, patch_size = infer_patch_grid(num_tokens, args.height, args.width)
    ref_scores = candidate_scores(ref_view, grid_h, grid_w)
    cur_scores = candidate_scores(cur_view, grid_h, grid_w)
    idx_ref = topk_indices(ref_scores["score"], args.topk)
    idx_cur = topk_indices(cur_scores["score"], args.topk)

    matches = mutual_matches(
        feats_ref=feats[0],
        feats_cur=feats[1],
        idx_ref=idx_ref,
        idx_cur=idx_cur,
        sim_thresh=args.sim_thresh,
        max_matches=args.max_matches,
    )
    matches = evaluate_matches(
        matches,
        ref_view,
        cur_view,
        grid_h,
        grid_w,
        patch_size,
        args.inlier_px,
        args.depth_rel_thresh,
    )

    draw_matches(ref_view, cur_view, matches, output_dir / "matches.png")

    summary = {
        "model_path": args.model_path,
        "projection": args.projection,
        "root": args.root,
        "ref": {"seq": args.ref_seq, "frame": args.ref_frame, "instance": ref_view["instance"]},
        "cur": {"seq": args.cur_seq, "frame": args.cur_frame, "instance": cur_view["instance"]},
        "image_size": [args.width, args.height],
        "grid": {"height": grid_h, "width": grid_w, "patch_size": patch_size, "num_tokens": int(num_tokens)},
        "num_ref_candidates": int(len(idx_ref)),
        "num_cur_candidates": int(len(idx_cur)),
        "pos_minmax": {
            "min": pos.reshape(-1, 2).min(dim=0).values.tolist(),
            "max": pos.reshape(-1, 2).max(dim=0).values.tolist(),
        },
        "thresholds": {
            "sim_thresh": args.sim_thresh,
            "inlier_px": args.inlier_px,
            "depth_rel_thresh": args.depth_rel_thresh,
        },
        "summary": summarize_matches(matches),
        "matches": matches,
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    np.savez_compressed(
        output_dir / "anchor_debug.npz",
        idx_ref=idx_ref,
        idx_cur=idx_cur,
        ref_score=ref_scores["score"],
        cur_score=cur_scores["score"],
        ref_texture=ref_scores["texture"],
        cur_texture=cur_scores["texture"],
        ref_human=ref_scores["human"],
        cur_human=cur_scores["human"],
    )

    print(json.dumps(summary["summary"], indent=2, ensure_ascii=False))
    print(f"Wrote {output_dir / 'summary.json'}")
    print(f"Wrote {output_dir / 'matches.png'}")


if __name__ == "__main__":
    main()
