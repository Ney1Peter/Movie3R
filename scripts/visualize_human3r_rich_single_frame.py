#!/usr/bin/env python3
"""Visualize original Human3R intermediate tokens on one RICH frame.

This script is intentionally standalone: it does not use Movie3R ShotToken code
or training dataloaders. It loads the original Human3R checkpoint and follows the
demo-style image path to inspect CUT3R image tokens, MHMR tokens, human tokens,
and decoder tokens for a single RICH RGB frame.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dust3r.model_human3r import load_model  # noqa: E402
from dust3r.smpl_model import apply_threshold, nms  # noqa: E402
from dust3r.utils.image import load_images, pad_image, unpad_uv  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default="/workspace/data/RICH/RICH_4Human3R/Training")
    parser.add_argument("--source_sequence", default="BBQ_001_guitar")
    parser.add_argument("--cam", type=int, default=6)
    parser.add_argument("--frame", type=int, default=6)
    parser.add_argument("--model_path", default=str(REPO_ROOT / "src" / "human3r_896L.pth"))
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--human_threshold", type=float, default=0.3)
    parser.add_argument("--out_dir", default=None)
    return parser.parse_args()


def seq_name(source_sequence, cam):
    return f"{source_sequence}_cam_{cam:02d}"


def tensor_to_rgb_uint8(img_tensor):
    arr = img_tensor.detach().float().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    arr = np.transpose(arr, (1, 2, 0))
    arr = np.clip(arr * 0.5 + 0.5, 0.0, 1.0)
    return (arr * 255.0).round().astype(np.uint8)


def save_rgb(path, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def normalize01(x):
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.float32)
    lo = np.percentile(x[finite], 1)
    hi = np.percentile(x[finite], 99)
    if hi <= lo + 1e-8:
        lo = float(x[finite].min())
        hi = float(x[finite].max())
    if hi <= lo + 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def heatmap_image(values, out_hw):
    values = normalize01(values)
    values_u8 = (values * 255.0).round().astype(np.uint8)
    hm = cv2.applyColorMap(values_u8, cv2.COLORMAP_TURBO)
    hm = cv2.resize(hm, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)


def overlay_heatmap(rgb, values, alpha=0.45):
    hm = heatmap_image(values, rgb.shape[:2])
    return np.clip((1.0 - alpha) * rgb.astype(np.float32) + alpha * hm.astype(np.float32), 0, 255).astype(np.uint8)


def token_pca_rgb(tokens, grid_hw, out_hw):
    arr = tokens.detach().float().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[0]
    arr = arr.astype(np.float32)
    arr = arr - arr.mean(axis=0, keepdims=True)
    if arr.shape[0] < 3:
        img = np.zeros((*grid_hw, 3), dtype=np.uint8)
    else:
        try:
            _, _, vh = np.linalg.svd(arr, full_matrices=False)
            comp = arr @ vh[:3].T
        except np.linalg.LinAlgError:
            comp = np.zeros((arr.shape[0], 3), dtype=np.float32)
        comp = normalize01(comp)
        img = (comp.reshape(grid_hw[0], grid_hw[1], 3) * 255.0).round().astype(np.uint8)
    return cv2.resize(img, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_NEAREST)


def draw_patch_grid(rgb, patch_size):
    out = rgb.copy()
    h, w = out.shape[:2]
    color = (255, 255, 0)
    for x in range(0, w + 1, patch_size):
        cv2.line(out, (x, 0), (x, h), color, 1, cv2.LINE_AA)
    for y in range(0, h + 1, patch_size):
        cv2.line(out, (0, y), (w, y), color, 1, cv2.LINE_AA)
    return out


def draw_points(rgb, points_xy, labels=None, color=(255, 0, 255), radius=7):
    out = rgb.copy()
    for i, pt in enumerate(points_xy):
        x, y = float(pt[0]), float(pt[1])
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        p = (int(round(x)), int(round(y)))
        cv2.circle(out, p, radius, color, -1, cv2.LINE_AA)
        cv2.circle(out, p, radius + 2, (255, 255, 255), 2, cv2.LINE_AA)
        if labels is not None:
            cv2.putText(out, str(labels[i]), (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return out


def save_token_visuals(out_dir, prefix, tokens, grid_hw, base_rgb=None):
    tokens_2d = tokens.detach().float()
    if tokens_2d.ndim == 3:
        tokens_2d = tokens_2d[0]
    norm = torch.linalg.vector_norm(tokens_2d, dim=-1).cpu().numpy().reshape(grid_hw)
    out_hw = base_rgb.shape[:2] if base_rgb is not None else (grid_hw[0] * 16, grid_hw[1] * 16)

    norm_hm = heatmap_image(norm, out_hw)
    pca = token_pca_rgb(tokens_2d, grid_hw, out_hw)
    save_rgb(out_dir / f"{prefix}_token_norm_heatmap.jpg", norm_hm)
    save_rgb(out_dir / f"{prefix}_token_pca_rgb.jpg", pca)
    if base_rgb is not None:
        save_rgb(out_dir / f"{prefix}_token_norm_overlay.jpg", overlay_heatmap(base_rgb, norm))
        pca_overlay = np.clip(0.45 * base_rgb.astype(np.float32) + 0.55 * pca.astype(np.float32), 0, 255).astype(np.uint8)
        save_rgb(out_dir / f"{prefix}_token_pca_overlay.jpg", pca_overlay)
    return {
        "shape": list(tokens.shape),
        "grid_hw": list(grid_hw),
        "norm_min": float(norm.min()),
        "norm_max": float(norm.max()),
        "norm_mean": float(norm.mean()),
    }


def save_bar_chart(path, values, title, height=260, width=880):
    values = np.asarray(values, dtype=np.float32)
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    cv2.putText(canvas, title, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
    if len(values) == 0:
        cv2.imwrite(str(path), canvas)
        return
    vmax = max(float(values.max()), 1e-6)
    left, right, top, bottom = 54, 20, 58, 42
    plot_w = width - left - right
    plot_h = height - top - bottom
    bar_w = max(1, plot_w // len(values))
    for i, v in enumerate(values):
        x0 = left + i * bar_w
        x1 = min(left + (i + 1) * bar_w - 2, width - right)
        y1 = height - bottom
        y0 = int(round(y1 - plot_h * float(v) / vmax))
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (40, 140, 240), -1)
        cv2.putText(canvas, str(i), (x0 + 2, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def make_view(image_path, size, img_res):
    from dust3r.utils.geometry import get_camera_parameters

    images = load_images([str(image_path)], size=size, verbose=True)
    view = {
        "img": images[0]["img"],
        "ray_map": torch.full((images[0]["img"].shape[0], 6, images[0]["img"].shape[-2], images[0]["img"].shape[-1]), torch.nan),
        "true_shape": torch.from_numpy(images[0]["true_shape"]),
        "idx": 0,
        "instance": str(image_path),
        "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
        "img_mask": torch.tensor(True).unsqueeze(0),
        "ray_mask": torch.tensor(False).unsqueeze(0),
        "update": torch.tensor(True).unsqueeze(0),
        "reset": torch.tensor(False).unsqueeze(0),
    }
    if img_res is not None:
        view["img_mhmr"] = pad_image(view["img"], img_res)
        view["K_mhmr"] = get_camera_parameters(img_res, device="cpu")
    return view


def to_device_view(view, device):
    out = {}
    ignore = {"idx", "instance"}
    for key, value in view.items():
        if key in ignore:
            out[key] = value
        elif isinstance(value, torch.Tensor):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def main():
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    seq = seq_name(args.source_sequence, args.cam)
    image_path = Path(args.data_root) / seq / "rgb" / f"{args.frame:08d}.png"
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    if args.out_dir is None:
        args.out_dir = str(
            REPO_ROOT
            / "output"
            / "human3r_rich_single_frame_tokens"
            / f"{args.source_sequence}_cam{args.cam:02d}_f{args.frame:08d}"
        )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading original Human3R model: {args.model_path}")
    model = load_model(args.model_path, device=device, verbose=True).eval()
    model.gradient_checkpointing = False
    patch_size = int(model.croco_args["patch_size"])
    mhmr_img_res = getattr(model, "mhmr_img_res", None)
    bb_patch_size = int(getattr(model, "bb_patch_size", 14))
    bb_token_res = int(getattr(model, "bb_token_res", mhmr_img_res // bb_patch_size))

    view_cpu = make_view(image_path, args.size, mhmr_img_res)
    input_rgb = tensor_to_rgb_uint8(view_cpu["img"])
    mhmr_rgb = tensor_to_rgb_uint8(view_cpu["img_mhmr"])
    save_rgb(out_dir / "00_input_resized_crop.jpg", input_rgb)
    save_rgb(out_dir / "01_input_patch_grid.jpg", draw_patch_grid(input_rgb, patch_size))
    save_rgb(out_dir / "02_mhmr_padded_input.jpg", mhmr_rgb)
    save_rgb(out_dir / "03_mhmr_patch_grid.jpg", draw_patch_grid(mhmr_rgb, bb_patch_size))

    raw_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if raw_bgr is not None:
        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        scale = min(1.0, 1200 / max(raw_rgb.shape[:2]))
        raw_small = cv2.resize(raw_rgb, (int(raw_rgb.shape[1] * scale), int(raw_rgb.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        save_rgb(out_dir / "00_original_rgb_preview.jpg", raw_small)

    view = to_device_view(view_cpu, device)
    summary = {
        "args": vars(args),
        "image_path": str(image_path),
        "model_path": str(args.model_path),
        "device": device,
        "patch_size": patch_size,
        "mhmr_img_res": mhmr_img_res,
        "bb_patch_size": bb_patch_size,
        "input_shape": list(view_cpu["img"].shape),
        "true_shape": view_cpu["true_shape"].cpu().numpy().tolist(),
    }

    with torch.no_grad():
        img = view["img"]
        true_shape = view["true_shape"]
        img_out, img_pos, _ = model._encode_image(img, true_shape)
        cut3r_feat = img_out[-1]
        h, w = map(int, true_shape[0].detach().cpu().numpy().tolist())
        cut3r_grid = (h // patch_size, w // patch_size)
        summary["cut3r_encoder"] = save_token_visuals(out_dir, "10_cut3r_encoder", cut3r_feat, cut3r_grid, input_rgb)

        cut3r_dec_embed = model.decoder_embed(cut3r_feat)
        summary["cut3r_decoder_embed"] = save_token_visuals(out_dir, "11_cut3r_decoder_embed", cut3r_dec_embed, cut3r_grid, input_rgb)

        selected_imgs_mhmr = view["img_mhmr"]
        mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
        selected_imgs_mhmr = (selected_imgs_mhmr * 0.5 + 0.5 - mean) / std
        mhmr_feat = model.backbone(selected_imgs_mhmr)
        mhmr_grid = (bb_token_res, bb_token_res)
        summary["mhmr_encoder"] = save_token_visuals(out_dir, "20_mhmr_encoder", mhmr_feat, mhmr_grid, mhmr_rgb)

        scores_raw = model.downstream_head.detect_mhmr(mhmr_feat)
        scores_raw_grid = scores_raw.detach().float().reshape(1, bb_token_res, bb_token_res, 1).permute(0, 3, 1, 2)
        scores_nms = nms(scores_raw_grid, kernel=3).permute(0, 2, 3, 1)
        score_map = scores_nms[0, :, :, 0].detach().float().cpu().numpy()
        save_rgb(out_dir / "21_mhmr_human_score_heatmap.jpg", heatmap_image(score_map, mhmr_rgb.shape[:2]))
        save_rgb(out_dir / "22_mhmr_human_score_overlay.jpg", overlay_heatmap(mhmr_rgb, score_map, alpha=0.5))

        idx = apply_threshold(args.human_threshold, scores_nms)
        peak_img_id, peak_h, peak_w = idx[0], idx[1], idx[2]
        peak_points = []
        peak_scores = []
        for h_id, w_id in zip(peak_h.detach().cpu().numpy().tolist(), peak_w.detach().cpu().numpy().tolist()):
            peak_points.append([(w_id + 0.5) * bb_patch_size, (h_id + 0.5) * bb_patch_size])
            peak_scores.append(float(score_map[h_id, w_id]))
        save_rgb(out_dir / "23_mhmr_detected_human_peaks.jpg", draw_points(mhmr_rgb, peak_points, labels=list(range(len(peak_points))), color=(255, 0, 255), radius=7))

        scores, smpl_tk_mhmr, pos_mhmr, smpl_loc, msks = model.smpl_tokenizer_mhmr([mhmr_feat], [img_pos], [view], inference=True)
        smpl_tk_cut3r, pos_cut3r, smpl_uv_cut3r = model.smpl_tokenizer_cut3r([cut3r_feat], [img_pos], [view], smpl_loc, inference=True)
        smpl_query = model.token_fuse(smpl_tk_mhmr, smpl_tk_cut3r, inference=True)

        n_humans = int(smpl_query[0].shape[1])
        summary["human_tokens"] = {
            "num_detected": n_humans,
            "threshold": float(args.human_threshold),
            "mhmr_peak_points_xy": peak_points,
            "mhmr_peak_scores": peak_scores,
            "smpl_tk_mhmr_shape": list(smpl_tk_mhmr[0].shape),
            "smpl_tk_cut3r_shape": list(smpl_tk_cut3r[0].shape),
            "fused_smpl_query_shape": list(smpl_query[0].shape),
        }

        cut3r_points = []
        if n_humans > 0:
            uv = smpl_uv_cut3r[0][0].detach().cpu().numpy()
            for w_id, h_id in uv:
                cut3r_points.append([(float(w_id) + 0.5) * patch_size, (float(h_id) + 0.5) * patch_size])
            save_rgb(out_dir / "24_cut3r_human_patch_tokens.jpg", draw_points(input_rgb, cut3r_points, labels=list(range(len(cut3r_points))), color=(255, 0, 255), radius=6))
            summary["human_tokens"]["cut3r_human_patch_points_xy"] = cut3r_points
        else:
            save_rgb(out_dir / "24_cut3r_human_patch_tokens.jpg", input_rgb)
            summary["human_tokens"]["cut3r_human_patch_points_xy"] = []

        state_feat, state_pos = model._init_state(cut3r_feat, img_pos)
        pose_feat = model.pose_token.expand(cut3r_feat.shape[0], -1, -1)
        pose_pos = -torch.ones(cut3r_feat.shape[0], 1, 2, device=device, dtype=img_pos.dtype)
        new_state_feat, dec, cross_attn_states = model._recurrent_rollout(
            state_feat,
            state_pos,
            cut3r_feat,
            img_pos,
            pose_feat,
            pose_pos,
            smpl_query[0],
            pos_cut3r[0],
            state_feat.clone(),
            img_mask=view["img_mask"],
            reset_mask=view["reset"],
            update=view.get("update", None),
        )

        summary["decoder"] = {
            "num_layers_plus_input": len(dec),
            "token_shapes": [list(x.shape) for x in dec],
            "cross_attn_states": [list(x.shape) if hasattr(x, "shape") else str(type(x)) for x in cross_attn_states],
        }

        pose_norms = []
        human_norms = []
        selected_layers = sorted(set([0, 1, len(dec) // 2, len(dec) - 1]))
        for layer_idx in selected_layers:
            tokens = dec[layer_idx].detach().float()
            layer_prefix = f"30_decoder_layer_{layer_idx:02d}"
            if layer_idx == 0:
                summary["decoder"][f"layer_{layer_idx:02d}_image"] = save_token_visuals(out_dir, f"{layer_prefix}_image", tokens, cut3r_grid, input_rgb)
                continue
            pose_token = tokens[:, 0:1]
            image_tokens = tokens[:, 1 : 1 + cut3r_feat.shape[1]]
            human_tokens = tokens[:, 1 + cut3r_feat.shape[1] :]
            summary["decoder"][f"layer_{layer_idx:02d}_image"] = save_token_visuals(out_dir, f"{layer_prefix}_image", image_tokens, cut3r_grid, input_rgb)
            summary["decoder"][f"layer_{layer_idx:02d}_pose_norm"] = float(torch.linalg.vector_norm(pose_token, dim=-1).mean().item())
            summary["decoder"][f"layer_{layer_idx:02d}_human_shape"] = list(human_tokens.shape)

        for layer_idx, tokens in enumerate(dec):
            tokens = tokens.detach().float()
            if layer_idx == 0:
                pose_norms.append(float("nan"))
                human_norms.append(float("nan"))
            else:
                pose_norms.append(float(torch.linalg.vector_norm(tokens[:, 0:1], dim=-1).mean().item()))
                h_tokens = tokens[:, 1 + cut3r_feat.shape[1] :]
                human_norms.append(float(torch.linalg.vector_norm(h_tokens, dim=-1).mean().item()) if h_tokens.numel() else 0.0)
        summary["decoder"]["pose_token_norm_by_layer"] = pose_norms
        summary["decoder"]["human_token_norm_by_layer"] = human_norms
        save_bar_chart(out_dir / "40_decoder_pose_token_norms.jpg", [v for v in pose_norms[1:] if np.isfinite(v)], "pose token norm by decoder layer")
        save_bar_chart(out_dir / "41_decoder_human_token_norms.jpg", [v for v in human_norms[1:] if np.isfinite(v)], "human token norm by decoder layer")

        if msks is not None and len(msks) > 0:
            msk = msks[0][0, ..., 0].detach().float().cpu().numpy()
            save_rgb(out_dir / "50_predicted_human_mask_heatmap.jpg", heatmap_image(msk, input_rgb.shape[:2]))
            save_rgb(out_dir / "51_predicted_human_mask_overlay.jpg", overlay_heatmap(input_rgb, msk, alpha=0.45))
            summary["predicted_mask"] = {"shape": list(msks[0].shape), "min": float(msk.min()), "max": float(msk.max()), "mean": float(msk.mean())}

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
