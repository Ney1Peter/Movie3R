#!/usr/bin/env python3
"""Visualize ShotToken inputs and pose-only decoder effects on one AABB sample."""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dust3r.datasets.avatarrex import AvatarReX_AABB  # noqa: E402
from dust3r.datasets.utils.transforms import ImgNorm  # noqa: E402
from dust3r.model import ARCroco3DStereo  # noqa: E402
from dust3r.utils.image import rgb as tensor_to_rgb  # noqa: E402


DEFAULT_MODEL = (
    REPO_ROOT
    / "experiments/training-4gpu-bz24-30ep-shot-v5_1-last2-20260510-095305/checkpoint-best.pth"
)
DEFAULT_ROOT = Path("/workspace/data/Avatarrex/avatarrex_zzr_output")
DEFAULT_OUT = REPO_ROOT / "output/shot_token_visualization/aabb_sample"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", default=str(DEFAULT_MODEL))
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--split", default="Training")
    parser.add_argument("--sample_idx", type=int, default=304)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resolution", type=int, nargs=2, default=(512, 288), metavar=("W", "H"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT))
    parser.add_argument("--overlay_alpha", type=float, default=0.55)
    parser.add_argument("--human_patch_threshold", type=float, default=0.10)
    parser.add_argument("--no_decoder_probe", action="store_true", help="Skip layerwise pose-shot adapter probe.")
    return parser.parse_args()


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def as_float(value):
    if torch.is_tensor(value):
        value = value.detach().float().cpu().reshape(-1)[0].item()
    elif isinstance(value, np.ndarray):
        value = value.reshape(-1)[0].item()
    return float(value)


def to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def view_image(view):
    true_shape = view.get("true_shape", None)
    if isinstance(true_shape, np.ndarray):
        true_shape = tuple(int(x) for x in true_shape.tolist())
    return tensor_to_rgb(view["img"], true_shape=true_shape)


def normalize01(values, lo_pct=2.0, hi_pct=98.0):
    arr = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32)
    lo = np.percentile(arr[finite], lo_pct)
    hi = np.percentile(arr[finite], hi_pct)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(arr[finite]))
        hi = float(np.nanmax(arr[finite]))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def resize_map(values, size_hw):
    h, w = size_hw
    norm = normalize01(values)
    img = Image.fromarray((norm * 255).astype(np.uint8))
    img = img.resize((w, h), Image.BILINEAR)
    return np.asarray(img).astype(np.float32) / 255.0


def resize_signed_map(values, size_hw):
    h, w = size_hw
    arr = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        scaled = np.zeros_like(arr, dtype=np.float32)
    else:
        denom = np.percentile(np.abs(arr[finite]), 98.0)
        if not np.isfinite(denom) or denom <= 1e-12:
            denom = float(np.max(np.abs(arr[finite]))) if finite.any() else 1.0
        denom = max(denom, 1e-12)
        scaled = np.clip(arr / denom, -1.0, 1.0)
    img = Image.fromarray(((scaled + 1.0) * 127.5).astype(np.uint8))
    img = img.resize((w, h), Image.BILINEAR)
    return np.asarray(img).astype(np.float32) / 127.5 - 1.0


def overlay_heatmap(image, heatmap, cmap_name="magma", alpha=0.55, signed=False):
    h, w = image.shape[:2]
    if signed:
        resized = resize_signed_map(heatmap, (h, w))
        colors = plt.get_cmap(cmap_name)((resized + 1.0) * 0.5)[..., :3]
    else:
        resized = resize_map(heatmap, (h, w))
        colors = plt.get_cmap(cmap_name)(resized)[..., :3]
    return np.clip((1.0 - alpha) * image + alpha * colors, 0.0, 1.0)


def save_image(path, image):
    arr = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def infer_patch_grid(n_tokens, true_shape, patch_size):
    h, w = [int(x) for x in true_shape]
    gh = max(1, h // patch_size)
    gw = max(1, w // patch_size)
    if gh * gw == n_tokens:
        return gh, gw

    target_ratio = h / max(w, 1)
    candidates = []
    for a in range(1, int(math.sqrt(n_tokens)) + 1):
        if n_tokens % a != 0:
            continue
        for gh_c, gw_c in ((a, n_tokens // a), (n_tokens // a, a)):
            score = abs((gh_c / max(gw_c, 1)) - target_ratio)
            candidates.append((score, gh_c, gw_c))
    if not candidates:
        raise ValueError(f"Cannot infer patch grid for {n_tokens} tokens and true_shape={true_shape}")
    _, gh, gw = min(candidates, key=lambda x: x[0])
    return gh, gw


def mask_to_patch(view, grid_hw):
    mask = view.get("msk", None)
    if mask is None or isinstance(mask, bool):
        return None
    mask_np = to_numpy(mask).astype(np.float32)
    if mask_np.ndim == 3:
        mask_np = mask_np[0] if mask_np.shape[0] <= 4 else mask_np[..., 0]
    if mask_np.size == 0:
        return None
    gh, gw = grid_hw
    mask_img = Image.fromarray((mask_np > 0.5).astype(np.uint8) * 255)
    mask_img = mask_img.resize((gw, gh), Image.BILINEAR)
    return np.asarray(mask_img).astype(np.float32) / 255.0


def patch_stats(values, human_patch, human_threshold):
    arr = np.asarray(values, dtype=np.float64)
    stats = {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
        "p95": float(np.percentile(arr, 95.0)),
    }
    if human_patch is None:
        stats.update(
            {
                "mask_available": False,
                "human_patch_fraction": None,
                "human_mean": None,
                "background_mean": None,
                "human_sum_share": None,
            }
        )
        return stats

    human = human_patch.reshape(-1) >= human_threshold
    background = ~human
    total = float(arr.sum())
    stats["mask_available"] = True
    stats["human_patch_fraction"] = float(human.mean())
    stats["human_mean"] = float(arr[human].mean()) if human.any() else None
    stats["background_mean"] = float(arr[background].mean()) if background.any() else None
    stats["human_sum_share"] = float(arr[human].sum() / total) if human.any() and abs(total) > 1e-12 else None
    return stats


def batch_encode_images(model, views, device):
    imgs = torch.stack([v["img"] for v in views], dim=0).to(device, non_blocking=True)
    shapes = torch.as_tensor(
        np.stack([v["true_shape"] for v in views]),
        device=device,
        dtype=torch.int32,
    )
    feat_ls, pos, _ = model._encode_image(imgs, shapes)
    return feat_ls[-1], pos, shapes


def compute_shot_tokens(model, feat):
    f_dec_all = model.decoder_embed(feat)
    f_dec = [f_dec_all[i : i + 1] for i in range(f_dec_all.shape[0])]
    q_tokens = []
    shot_infos = []
    for i in range(len(f_dec)):
        if i == 0:
            q_t, info = model.shot_token_generator(f_dec[0], f_dec[0], i=0, return_info=True)
        else:
            q_t, info = model.shot_token_generator(f_dec[i], f_dec[i - 1], i=i, return_info=True)
        q_tokens.append(q_t)
        shot_infos.append(info)
    return f_dec, q_tokens, shot_infos


def compute_timeline_rows(model, views, f_dec, q_tokens, shot_infos):
    rows = []
    prev_q = None
    for i, (q_t, info) in enumerate(zip(q_tokens, shot_infos)):
        g_curr = f_dec[i].mean(dim=1)
        if i == 0:
            g_prev = g_curr
            q_delta_norm = 0.0
            q_cos_prev = 1.0
        else:
            g_prev = f_dec[i - 1].mean(dim=1)
            q_prev = prev_q.reshape(1, -1)
            q_flat = q_t.reshape(1, -1)
            q_delta_norm = float(torch.linalg.norm(q_flat - q_prev).detach().cpu())
            q_cos_prev = float(F.cosine_similarity(q_flat, q_prev, dim=-1).detach().cpu())

        g_diff = g_curr - g_prev
        g_cos = float(F.cosine_similarity(g_curr, g_prev, dim=-1).detach().cpu())
        q_context_norm = None
        q_context_amplification = None
        if hasattr(model, "layerwise_pose_shot_adapter"):
            q_context = model.layerwise_pose_shot_adapter.context_norm(q_t)
            q_context_norm = float(torch.linalg.norm(q_context).detach().cpu())
            q_norm = float(torch.linalg.norm(q_t).detach().cpu())
            q_context_amplification = q_context_norm / max(q_norm, 1e-12)
        rows.append(
            {
                "view": i,
                "label": str(views[i].get("label", i)),
                "instance": str(views[i].get("instance", "")),
                "shot_label": int(views[i].get("shot_label", 0)),
                "g_diff_norm": float(torch.linalg.norm(g_diff).detach().cpu()),
                "g_cos_prev": g_cos,
                "shot_logit": as_float(info["shot_logit"]),
                "shot_prob": as_float(info["shot_prob"]),
                "shot_q_norm": as_float(info["shot_q_norm"]),
                "shot_q_raw_norm": as_float(info["shot_q_raw_norm"]),
                "shot_q_energy": as_float(info["shot_q_energy"]),
                "shot_scale": as_float(info["shot_scale"]),
                "shot_q_context_norm": q_context_norm,
                "shot_q_context_amplification": q_context_amplification,
                "q_delta_norm": q_delta_norm,
                "q_cos_prev": q_cos_prev,
            }
        )
        prev_q = q_t.detach()
    return rows


def compute_pair_maps(views, f_dec, shapes, patch_size, human_threshold):
    pair_rows = []
    pair_maps = []
    for i in range(1, len(f_dec)):
        prev_tokens = f_dec[i - 1]
        curr_tokens = f_dec[i]
        diff_tokens = curr_tokens - prev_tokens
        diff_norm = torch.linalg.norm(diff_tokens, dim=-1).squeeze(0)
        cos_dist = 1.0 - F.cosine_similarity(curr_tokens, prev_tokens, dim=-1).squeeze(0)
        g_diff = curr_tokens.mean(dim=1) - prev_tokens.mean(dim=1)
        contribution = (diff_tokens * g_diff[:, None, :]).sum(dim=-1).squeeze(0)

        true_shape = shapes[i].detach().cpu().numpy().tolist()
        gh, gw = infer_patch_grid(diff_norm.numel(), true_shape, patch_size)
        diff_map = diff_norm.detach().float().cpu().numpy().reshape(gh, gw)
        cos_map = cos_dist.detach().float().cpu().numpy().reshape(gh, gw)
        contrib_map = contribution.detach().float().cpu().numpy().reshape(gh, gw)

        prev_mask = mask_to_patch(views[i - 1], (gh, gw))
        curr_mask = mask_to_patch(views[i], (gh, gw))
        if prev_mask is None and curr_mask is None:
            human_patch = None
        elif prev_mask is None:
            human_patch = curr_mask
        elif curr_mask is None:
            human_patch = prev_mask
        else:
            human_patch = np.maximum(prev_mask, curr_mask)

        stats = patch_stats(diff_map.reshape(-1), human_patch, human_threshold)
        contrib_abs = np.abs(contrib_map.reshape(-1))
        top_k = min(32, contrib_abs.size)
        if top_k > 0 and contrib_abs.sum() > 1e-12:
            top_share = float(np.sort(contrib_abs)[-top_k:].sum() / contrib_abs.sum())
        else:
            top_share = None
        signed_sum = float(contrib_map.sum())
        abs_sum = float(np.abs(contrib_map).sum())
        cancellation = float(abs(signed_sum) / abs_sum) if abs_sum > 1e-12 else None

        row = {
            "pair": f"{i - 1}->{i}",
            "prev_label": str(views[i - 1].get("label", i - 1)),
            "curr_label": str(views[i].get("label", i)),
            "shot_label": int(views[i].get("shot_label", 0)),
            "grid_hw": [int(gh), int(gw)],
            "diff_stats": stats,
            "contribution_top32_abs_share": top_share,
            "contribution_signed_sum": signed_sum,
            "contribution_abs_sum": abs_sum,
            "contribution_cancellation_ratio": cancellation,
        }
        pair_rows.append(row)
        pair_maps.append(
            {
                "i": i,
                "diff": diff_map,
                "cos_dist": cos_map,
                "contribution": contrib_map,
                "human_patch": human_patch,
                "stats": row,
            }
        )
    return pair_rows, pair_maps


def save_pair_visualizations(views, pair_maps, out_dir, alpha):
    image_dir = ensure_dir(out_dir / "pair_heatmaps")
    saved = []
    for item in pair_maps:
        i = item["i"]
        prev_img = view_image(views[i - 1])
        curr_img = view_image(views[i])
        diff_overlay = overlay_heatmap(curr_img, item["diff"], "magma", alpha=alpha)
        cos_overlay = overlay_heatmap(curr_img, item["cos_dist"], "viridis", alpha=alpha)
        contrib_overlay = overlay_heatmap(curr_img, item["contribution"], "coolwarm", alpha=alpha, signed=True)

        stem = f"pair_{i - 1}_{i}_shot{int(views[i].get('shot_label', 0))}"
        paths = {
            "diff_overlay": image_dir / f"{stem}_patch_diff_overlay.png",
            "cos_overlay": image_dir / f"{stem}_patch_cosdist_overlay.png",
            "contribution_overlay": image_dir / f"{stem}_mean_pool_contribution_overlay.png",
            "panel": image_dir / f"{stem}_panel.png",
        }
        save_image(paths["diff_overlay"], diff_overlay)
        save_image(paths["cos_overlay"], cos_overlay)
        save_image(paths["contribution_overlay"], contrib_overlay)

        fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        axes = axes.reshape(-1)
        axes[0].imshow(prev_img)
        axes[0].set_title(f"prev {i - 1}: {views[i - 1].get('label', '')}")
        axes[1].imshow(curr_img)
        axes[1].set_title(f"curr {i}: {views[i].get('label', '')}")
        axes[2].imshow(diff_overlay)
        axes[2].set_title("patch diff norm")
        axes[3].imshow(contrib_overlay)
        axes[3].set_title("mean-pool contribution +/-")
        axes[4].imshow(cos_overlay)
        axes[4].set_title("1 - patch cosine")
        if item["human_patch"] is not None:
            axes[5].imshow(curr_img)
            axes[5].imshow(
                resize_map(item["human_patch"], curr_img.shape[:2]),
                cmap="Greens",
                alpha=0.45,
                vmin=0,
                vmax=1,
            )
            axes[5].set_title("human mask patches")
        else:
            axes[5].axis("off")
            axes[5].set_title("no human mask")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"pair {i - 1}->{i}, shot_label={int(views[i].get('shot_label', 0))}")
        fig.savefig(paths["panel"], dpi=160)
        plt.close(fig)

        saved.append({k: str(v) for k, v in paths.items()})
    return saved


def plot_timeline(rows, out_dir):
    path = out_dir / "timeline_metrics.png"
    x = np.asarray([r["view"] for r in rows])
    labels = np.asarray([r["shot_label"] for r in rows], dtype=np.float32)

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True, constrained_layout=True)
    axes[0].plot(x, [r["shot_prob"] for r in rows], marker="o", label="shot_prob")
    axes[0].bar(x, labels, alpha=0.25, label="shot_label")
    axes[0].set_ylabel("prob / label")
    axes[0].legend(loc="best")

    axes[1].plot(x, [r["g_diff_norm"] for r in rows], marker="o", label="g_diff_norm")
    axes[1].plot(x, [1.0 - r["g_cos_prev"] for r in rows], marker="s", label="1 - g_cos_prev")
    axes[1].set_ylabel("global diff")
    axes[1].legend(loc="best")

    axes[2].plot(x, [r["shot_q_norm"] for r in rows], marker="o", label="q_norm gated")
    axes[2].plot(x, [r["shot_q_raw_norm"] for r in rows], marker="s", label="q_raw_norm")
    if rows[0].get("shot_q_context_norm") is not None:
        axes[2].plot(x, [r["shot_q_context_norm"] for r in rows], marker="^", label="q_norm after adapter LayerNorm")
        axes[2].set_yscale("log")
    axes[2].set_ylabel("q norm")
    axes[2].legend(loc="best")

    axes[3].plot(x, [r["q_delta_norm"] for r in rows], marker="o", label="q_delta_norm")
    axes[3].plot(x, [1.0 - r["q_cos_prev"] for r in rows], marker="s", label="1 - q_cos_prev")
    axes[3].bar(x, labels, alpha=0.20)
    axes[3].set_ylabel("q temporal")
    axes[3].set_xlabel("view index")
    axes[3].legend(loc="best")

    for ax in axes:
        ax.grid(alpha=0.25)
        ax.set_xticks(x)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def collect_decoder_probe(model, feat, pos, q_tokens):
    if not getattr(model, "pose_head_flag", False):
        return []

    records = []
    feat_list = [feat[i : i + 1] for i in range(feat.shape[0])]
    pos_list = [pos[i : i + 1] for i in range(pos.shape[0])]
    state_feat, state_pos = model._init_state(feat_list[0], pos_list[0])
    init_state_feat = state_feat.clone()
    mem = model.pose_retriever.mem.expand(feat_list[0].shape[0], -1, -1)
    current = {"view": None}
    orig_apply = model._apply_layerwise_pose_shot

    def instrumented_apply(f_img, q_t, layer_idx):
        if q_t is None:
            return f_img
        if not getattr(model, "enable_layerwise_pose_shot_adapter", False):
            return f_img
        if layer_idx not in getattr(model, "shot_pose_layers", set()):
            return f_img

        adapter = model.layerwise_pose_shot_adapter
        pose_before = f_img[:, 0:1]
        pose_query = adapter.pose_norm(pose_before)
        context = adapter.context_norm(torch.cat([pose_before, q_t], dim=1))
        attn_out, attn_weights = adapter.cross_attn(
            pose_query,
            context,
            context,
            need_weights=True,
            average_attn_weights=False,
        )
        delta_pose = adapter.ffn(attn_out)
        scaled_delta = adapter.pose_update_scale * delta_pose
        pose_after = pose_before + scaled_delta

        weights = attn_weights.detach().float().cpu().numpy()[0, :, 0, :]
        q_context = adapter.context_norm(q_t)
        q_norm = float(torch.linalg.norm(q_t).detach().cpu())
        q_context_norm = float(torch.linalg.norm(q_context).detach().cpu())
        records.append(
            {
                "view": int(current["view"]),
                "layer": int(layer_idx),
                "attn_pose_mean": float(weights[:, 0].mean()),
                "attn_shot_mean": float(weights[:, 1].mean()),
                "attn_pose_by_head": [float(x) for x in weights[:, 0].tolist()],
                "attn_shot_by_head": [float(x) for x in weights[:, 1].tolist()],
                "delta_pose_norm_raw": float(torch.linalg.norm(delta_pose).detach().cpu()),
                "delta_pose_norm_scaled": float(torch.linalg.norm(scaled_delta).detach().cpu()),
                "pose_before_norm": float(torch.linalg.norm(pose_before).detach().cpu()),
                "q_norm": q_norm,
                "q_context_norm": q_context_norm,
                "q_context_amplification": q_context_norm / max(q_norm, 1e-12),
                "pose_update_scale": float(adapter.pose_update_scale.detach().cpu()),
            }
        )
        return torch.cat([pose_after, f_img[:, 1:]], dim=1)

    model._apply_layerwise_pose_shot = instrumented_apply
    try:
        for i, (feat_i, pos_i) in enumerate(zip(feat_list, pos_list)):
            current["view"] = i
            global_img_feat_i = model._get_img_level_feat(feat_i)
            if i == 0:
                pose_feat_i = model.pose_token.expand(feat_i.shape[0], -1, -1)
            else:
                pose_feat_i = model.pose_retriever.inquire(global_img_feat_i, mem)
            pose_pos_i = -torch.ones(
                feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
            )
            new_state_feat, dec, _ = model._recurrent_rollout(
                state_feat,
                state_pos,
                feat_i,
                pos_i,
                pose_feat_i,
                pose_pos_i,
                None,
                None,
                init_state_feat,
                f_shot=None,
                f_pose_shot=q_tokens[i],
            )
            out_pose_feat_i = dec[-1][:, 0:1]
            mem = model.pose_retriever.update_mem(mem, global_img_feat_i, out_pose_feat_i)
            state_feat = new_state_feat
    finally:
        model._apply_layerwise_pose_shot = orig_apply
    return records


def plot_decoder_probe(records, out_dir):
    if not records:
        return None
    layers = sorted({r["layer"] for r in records})
    views = sorted({r["view"] for r in records})
    layer_to_col = {layer: idx for idx, layer in enumerate(layers)}
    view_to_row = {view: idx for idx, view in enumerate(views)}
    attn = np.full((len(views), len(layers)), np.nan, dtype=np.float32)
    delta = np.full_like(attn, np.nan)
    for r in records:
        row = view_to_row[r["view"]]
        col = layer_to_col[r["layer"]]
        attn[row, col] = r["attn_shot_mean"]
        delta[row, col] = r["delta_pose_norm_scaled"]

    path = out_dir / "decoder_pose_shot_probe.png"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    im0 = axes[0].imshow(attn, vmin=0.0, vmax=1.0, cmap="magma")
    axes[0].set_title("pose attention to q_t")
    axes[0].set_xlabel("decoder layer")
    axes[0].set_ylabel("view")
    axes[0].set_xticks(range(len(layers)), labels=[str(x) for x in layers])
    axes[0].set_yticks(range(len(views)), labels=[str(x) for x in views])
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(delta, cmap="viridis")
    axes[1].set_title("scaled pose delta norm")
    axes[1].set_xlabel("decoder layer")
    axes[1].set_ylabel("view")
    axes[1].set_xticks(range(len(layers)), labels=[str(x) for x in layers])
    axes[1].set_yticks(range(len(views)), labels=[str(x) for x in views])
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def save_sample_strip(views, out_dir):
    path = out_dir / "sample_views.png"
    fig, axes = plt.subplots(1, len(views), figsize=(4 * len(views), 3), constrained_layout=True)
    if len(views) == 1:
        axes = [axes]
    for i, (ax, view) in enumerate(zip(axes, views)):
        ax.imshow(view_image(view))
        ax.set_title(f"view {i}\nshot={int(view.get('shot_label', 0))}\n{view.get('label', '')}")
        ax.axis("off")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def write_markdown_report(out_dir, summary):
    path = out_dir / "README.md"
    lines = [
        "# ShotToken Visualization",
        "",
        f"sample_idx: `{summary['sample']['sample_idx']}`",
        f"model: `{summary['model_path']}`",
        "",
        "## Views",
        "",
    ]
    for row in summary["timeline"]:
        lines.append(
            f"- view {row['view']}: `{row['label']}`, shot_label={row['shot_label']}, "
            f"shot_prob={row['shot_prob']:.4f}, q_norm={row['shot_q_norm']:.4f}, "
            f"q_context_norm={row.get('shot_q_context_norm'):.4f}, "
            f"g_diff_norm={row['g_diff_norm']:.4f}"
        )
    lines.extend(["", "## Main Files", ""])
    lines.append(f"- sample strip: `{Path(summary['files']['sample_strip']).name}`")
    lines.append(f"- timeline: `{Path(summary['files']['timeline']).name}`")
    if summary["files"].get("decoder_probe"):
        lines.append(f"- decoder probe: `{Path(summary['files']['decoder_probe']).name}`")
    lines.append("- pair heatmaps: `pair_heatmaps/`")
    lines.extend(
        [
            "",
            "## Adapter LayerNorm Check",
            "",
            "`q_context_norm` is `LayerwisePoseShotAdapter.context_norm(q_t).norm()`. If it is much larger than `q_norm`, the ShotToken gate/scale magnitude is mostly removed before pose-shot attention.",
            "",
        ]
    )
    for row in summary["timeline"]:
        lines.append(
            f"- view {row['view']}: q_norm={row['shot_q_norm']:.4f}, "
            f"q_context_norm={row.get('shot_q_context_norm'):.4f}, "
            f"amplification={row.get('shot_q_context_amplification'):.1f}x"
        )
    lines.extend(
        [
            "",
            "Note: the decoder probe runs the CUT3R image-token + pose-token path to inspect the layerwise pose-shot adapter. It does not include MHMR/SMPL human tokens, so use it as an adapter-mechanics probe rather than a full demo prediction trace.",
        ]
    )
    lines.extend(["", "## Pair Stats", ""])
    for row in summary["pairs"]:
        diff = row["diff_stats"]
        lines.append(
            f"- pair {row['pair']} shot_label={row['shot_label']}: "
            f"diff_mean={diff['mean']:.4f}, diff_p95={diff['p95']:.4f}, "
            f"human_share={diff.get('human_sum_share')}, "
            f"contrib_top32_abs_share={row['contribution_top32_abs_share']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def main():
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    out_dir = ensure_dir(args.out_dir)

    print(f"Loading AABB dataset: root={args.root}, sample_idx={args.sample_idx}")
    dataset = AvatarReX_AABB(
        split=args.split,
        ROOT=args.root,
        resolution=tuple(args.resolution),
        transform=ImgNorm,
        num_views=4,
        seed=args.seed,
        n_corres=0,
    )
    views = dataset[args.sample_idx]
    print("Sample views:")
    for i, view in enumerate(views):
        print(f"  view {i}: label={view.get('label')} shot_label={view.get('shot_label')} instance={view.get('instance')}")

    print(f"Loading model: {args.model_path}")
    model = ARCroco3DStereo.from_pretrained(args.model_path).to(device).eval()
    model.enable_shot_adaptation = True
    model.enable_shot_decoder_token = False
    model.enable_layerwise_pose_shot_adapter = True
    for p in model.parameters():
        p.requires_grad_(False)

    patch_size = int(model.croco_args.get("patch_size", 16))

    with torch.no_grad():
        feat, pos, shapes = batch_encode_images(model, views, device)
        f_dec, q_tokens, shot_infos = compute_shot_tokens(model, feat)
        timeline = compute_timeline_rows(model, views, f_dec, q_tokens, shot_infos)
        pairs, pair_maps = compute_pair_maps(
            views, f_dec, shapes, patch_size, args.human_patch_threshold
        )
        decoder_records = [] if args.no_decoder_probe else collect_decoder_probe(model, feat, pos, q_tokens)

    files = {
        "sample_strip": save_sample_strip(views, out_dir),
        "timeline": plot_timeline(timeline, out_dir),
        "pair_visualizations": save_pair_visualizations(views, pair_maps, out_dir, args.overlay_alpha),
        "decoder_probe": plot_decoder_probe(decoder_records, out_dir),
    }

    summary = {
        "model_path": str(args.model_path),
        "root": str(args.root),
        "device": str(device),
        "sample": {
            "sample_idx": int(args.sample_idx),
            "seed": int(args.seed),
            "resolution": [int(args.resolution[0]), int(args.resolution[1])],
        },
        "timeline": timeline,
        "pairs": pairs,
        "decoder_probe": decoder_records,
        "files": files,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path = write_markdown_report(out_dir, summary)

    print(f"Wrote summary: {summary_path}")
    print(f"Wrote report: {report_path}")
    print(f"Wrote visualizations under: {out_dir}")


if __name__ == "__main__":
    main()
