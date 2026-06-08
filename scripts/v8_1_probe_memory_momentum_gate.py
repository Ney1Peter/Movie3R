#!/usr/bin/env python3
"""Probe token-aligned memory, temporal momentum, and gate cues for V8.1.

This is a diagnostic script.  It does not train a module and does not modify
Human3R outputs.  The goal is to make the three candidate prompt sources more
visible without introducing information that a future token-level branch could
not access:

1. Recurrent latent memory:
   dump ``state_feat`` and ``mem`` from ``forward_recurrent_lighter`` with
   ``ret_state=True``; compare token-aligned body anchor tokens and global image
   tokens to these latent slots.
2. Temporal momentum:
   measure raw camera jumps and token-aligned human-anchor jumps using only
   Human3R predicted SMPL anchors that were already validated at the token
   locations: pelvis, torso, left foot, right foot.
3. Current-vs-memory gate:
   build a simple proxy gate from camera jump, human-anchor jump, previous-human
   fit delta, state update norm, and memory retrieval disagreement.

GT AvatarReX SMPL is only used to choose/verify the image locations of the four
body anchors, matching the previous token-validation probes.  The gate proxy
itself does not use GT camera pose, GT pointmap, or background matching.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from add_ckpt_path import add_path_to_dust3r  # noqa: E402
from demo import prepare_input  # noqa: E402
from scripts.v8_1_build_human_only_pose_correction import (  # noqa: E402
    TOKEN_BODY_ANCHOR_SPECS,
    fit_rigid,
    load_camera_poses,
    load_human3r_smpl_joints,
    transform_points,
)
from scripts.v8_1_probe_aabb_case import (  # noqa: E402
    AvatarReXRawProjector,
    draw_overlay,
    load_view,
    patch_index,
)


ANCHOR_NAMES = [name for name, _ in TOKEN_BODY_ANCHOR_SPECS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare_root", type=Path, default=REPO_ROOT / "output" / "v8_1_human3r_aabb_compare")
    parser.add_argument("--raw_dir", type=Path, default=None)
    parser.add_argument("--token_aligned_dir", type=Path, default=None)
    parser.add_argument("--root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data"))
    parser.add_argument("--avatarrex_raw_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data/AvatarReX_raw_meta/lbn1"))
    parser.add_argument("--split", default="Training")
    parser.add_argument("--model_path", type=Path, default=REPO_ROOT / "src" / "human3r_896L.pth")
    parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "output" / "v8_1_memory_momentum_gate_probe")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reset_interval", type=int, default=100)
    parser.add_argument("--min_smpl_iou", type=float, default=0.50)
    parser.add_argument("--topk_mem", type=int, default=8)
    return parser.parse_args()


def ensure_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "anchors": root / "anchor_overlays",
        "state": root / "state_memory_heatmaps",
        "curves": root / "momentum_gate_curves",
        "dump": root / "compact_token_dump",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def read_case_manifest(compare_root: Path) -> list[dict]:
    manifest_path = compare_root / "case_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    specs = data["input_frames"]
    for item in specs:
        item["path"] = str((REPO_ROOT / item["path"]).resolve()) if not Path(item["path"]).is_absolute() else item["path"]
    return specs


def rotation_step_deg(pose_a: np.ndarray, pose_b: np.ndarray) -> float:
    rel = pose_b[:3, :3] @ pose_a[:3, :3].T
    value = np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(value)))


def transform_delta_metrics(delta: np.ndarray) -> tuple[float, float]:
    t = float(np.linalg.norm(delta[:3, 3]))
    r = rotation_step_deg(np.eye(4, dtype=np.float64), delta)
    return t, r


def cosine_scores(query: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    query = query.astype(np.float64)
    tokens = tokens.astype(np.float64)
    query = query / max(float(np.linalg.norm(query)), 1e-12)
    tokens = tokens / np.maximum(np.linalg.norm(tokens, axis=-1, keepdims=True), 1e-12)
    return tokens @ query


def normalize01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).any():
        return np.zeros_like(values)
    finite = values[np.isfinite(values)]
    lo, hi = float(finite.min()), float(finite.max())
    if hi - lo < 1e-9:
        return np.zeros_like(values)
    out = (values - lo) / (hi - lo)
    out[~np.isfinite(out)] = 0.0
    return out


def normalize01_after_first(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.zeros_like(values)
    if out.shape[0] <= 1:
        return out
    out[1:] = normalize01(values[1:])
    return out


def token_grid_hw_from_views(views: list[dict], ntokens: int) -> tuple[int, int]:
    h, w = views[0]["img"].shape[-2:]
    gh, gw = int(h // 16), int(w // 16)
    if gh * gw == ntokens:
        return gh, gw
    side = int(round(math.sqrt(ntokens)))
    if side * side == ntokens:
        return side, side
    raise RuntimeError(f"Cannot infer token grid for {ntokens} tokens from image shape {(h, w)}")


def scatter_scores_to_position_grid(scores: np.ndarray, pos: np.ndarray | None) -> np.ndarray:
    if pos is None:
        side = int(math.ceil(math.sqrt(scores.shape[0])))
        grid = np.full((side, side), np.nan, dtype=np.float32)
        grid.flat[: scores.shape[0]] = scores.astype(np.float32)
        return grid
    pos = np.asarray(pos)
    ys = pos[:, 0].astype(np.int64)
    xs = pos[:, 1].astype(np.int64)
    h, w = int(ys.max()) + 1, int(xs.max()) + 1
    grid = np.full((h, w), np.nan, dtype=np.float32)
    grid[ys, xs] = scores.astype(np.float32)
    return grid


def normalize_grid(grid: np.ndarray) -> np.ndarray:
    out = grid.astype(np.float32).copy()
    valid = np.isfinite(out)
    if not valid.any():
        return np.nan_to_num(out)
    lo, hi = float(out[valid].min()), float(out[valid].max())
    if hi - lo < 1e-8:
        out[valid] = 0.0
    else:
        out[valid] = (out[valid] - lo) / (hi - lo)
    return np.nan_to_num(out)


def snapshot_to_numpy(state_args: list[tuple]) -> list[dict[str, np.ndarray | None]]:
    snapshots = []
    for state_feat, state_pos, init_state_feat, mem, init_mem in state_args:
        snapshots.append(
            {
                "state_feat": state_feat.detach().float().cpu().numpy()[0],
                "state_pos": None if state_pos is None else state_pos.detach().cpu().numpy()[0],
                "init_state_feat": init_state_feat.detach().float().cpu().numpy()[0],
                "mem": mem.detach().float().cpu().numpy()[0],
                "init_mem": init_mem.detach().float().cpu().numpy()[0],
            }
        )
    return snapshots


def run_human3r_state_dump(args: argparse.Namespace, specs: list[dict]) -> dict:
    from src.dust3r.inference import inference_recurrent_lighter
    from src.dust3r.model import ARCroco3DStereo

    add_path_to_dust3r(str(args.model_path))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    model = ARCroco3DStereo.from_pretrained(str(args.model_path)).to(args.device).eval()
    img_res = getattr(model, "mhmr_img_res", None)
    img_paths = [item["path"] for item in specs]
    views = prepare_input(
        img_paths=img_paths,
        img_mask=[True] * len(img_paths),
        size=args.size,
        revisit=1,
        update=True,
        img_res=img_res,
        reset_interval=args.reset_interval,
    )
    with torch.no_grad():
        _, state_args = inference_recurrent_lighter(views, model, args.device, verbose=True, use_ttt3r=False)
        imgs = torch.cat([view["img"] for view in views], dim=0).to(args.device)
        shapes = torch.cat([view["true_shape"] for view in views], dim=0).to(args.device)
        feats, _, _ = model._encode_image(imgs, shapes)
        enc_tokens = feats[-1]
        dec_tokens = model.decoder_embed(enc_tokens)
        global_img_feat = model._get_img_level_feat(enc_tokens)
        mem_queries = model.pose_retriever.proj_q(global_img_feat)

    snapshots = snapshot_to_numpy(state_args)
    dec_np = dec_tokens.detach().float().cpu().numpy()
    enc_np = enc_tokens.detach().float().cpu().numpy()
    query_np = mem_queries.detach().float().cpu().numpy()[:, 0]
    grid_hw = token_grid_hw_from_views(views, dec_np.shape[1])

    return {
        "snapshots": snapshots,
        "enc_tokens": enc_np,
        "dec_tokens": dec_np,
        "mem_queries": query_np,
        "grid_hw": grid_hw,
    }


def build_processed_views(args: argparse.Namespace, specs: list[dict]):
    projector = AvatarReXRawProjector(args.avatarrex_raw_root)
    views = []
    for item in specs:
        views.append(load_view(args, projector, int(item["idx"]), item["seq"], int(item["frame"]), item["label"]))
    low_iou = [(v.label, v.seq, v.frame, v.smpl_projection_iou) for v in views if v.smpl_projection_iou < args.min_smpl_iou]
    if low_iou:
        raise RuntimeError(f"SMPL projection sanity check failed: {low_iou}")
    return views


def save_anchor_overlays(views, out_dir: Path) -> None:
    for view in views:
        draw_overlay(view, out_dir / f"view{view.view_idx}_{view.label}.png")


def save_state_anchor_panels(
    views,
    snapshots: list[dict],
    dec_tokens: np.ndarray,
    grid_hw: tuple[int, int],
    out_dir: Path,
) -> list[dict]:
    rows = []
    fig, axes = plt.subplots(len(views), len(ANCHOR_NAMES), figsize=(3.0 * len(ANCHOR_NAMES), 2.7 * len(views)))
    for i, view in enumerate(views):
        state_after = snapshots[i + 1]["state_feat"]
        state_pos = snapshots[i + 1]["state_pos"]
        for j, anchor in enumerate(ANCHOR_NAMES):
            patch_idx, px, py = patch_index(view.anchors[anchor], grid_hw)
            query = dec_tokens[i, patch_idx]
            scores = cosine_scores(query, state_after)
            grid = normalize_grid(scatter_scores_to_position_grid(scores, state_pos))
            ax = axes[i, j]
            ax.imshow(grid, cmap="magma", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"v{i} {anchor}", fontsize=8)
            rows.append(
                {
                    "view_idx": i,
                    "label": view.label,
                    "seq": view.seq,
                    "frame": view.frame,
                    "anchor": anchor,
                    "patch_idx": patch_idx,
                    "patch_x": px,
                    "patch_y": py,
                    "state_top1_sim": float(scores.max()),
                    "state_top5_sim": float(np.sort(scores)[-5:].mean()),
                    "state_mean_sim": float(scores.mean()),
                    "state_std_sim": float(scores.std()),
                }
            )
            cv2.imwrite(
                str(out_dir / f"view{i}_{anchor}_state_similarity.png"),
                cv2.applyColorMap(np.uint8(grid * 255), cv2.COLORMAP_MAGMA),
            )
    fig.suptitle("Body anchor decoder-token similarity to recurrent state after each frame", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "state_anchor_similarity_panel.png", dpi=180)
    plt.close(fig)
    return rows


def save_memory_query_panels(
    specs: list[dict],
    snapshots: list[dict],
    mem_queries: np.ndarray,
    out_dir: Path,
    topk: int,
) -> list[dict]:
    rows = []
    fig, axes = plt.subplots(1, len(specs), figsize=(3.1 * len(specs), 3.0))
    if len(specs) == 1:
        axes = [axes]
    for i, item in enumerate(specs):
        mem_before = snapshots[i]["mem"]
        d = mem_queries.shape[-1]
        mem_key = mem_before[:, :d]
        scores = cosine_scores(mem_queries[i], mem_key)
        grid = normalize_grid(scatter_scores_to_position_grid(scores, None))
        axes[i].imshow(grid, cmap="viridis", vmin=0.0, vmax=1.0)
        axes[i].set_title(f"v{i} before\n{item['label']}", fontsize=8)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        rows.append(
            {
                "view_idx": i,
                "label": item["label"],
                "seq": item["seq"],
                "frame": int(item["frame"]),
                "memory_query_top1_sim": float(scores.max()),
                "memory_query_topk_sim": float(np.sort(scores)[-topk:].mean()),
                "memory_query_mean_sim": float(scores.mean()),
                "memory_query_std_sim": float(scores.std()),
            }
        )
        cv2.imwrite(
            str(out_dir / f"view{i}_global_query_to_memory_before.png"),
            cv2.applyColorMap(np.uint8(grid * 255), cv2.COLORMAP_VIRIDIS),
        )
    fig.suptitle("Current global-image token similarity to memory before processing frame", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "memory_query_similarity_panel.png", dpi=180)
    plt.close(fig)
    return rows


def compute_state_update_rows(specs: list[dict], snapshots: list[dict]) -> list[dict]:
    rows = []
    for i, item in enumerate(specs):
        before = snapshots[i]
        after = snapshots[i + 1]
        state_delta = after["state_feat"] - before["state_feat"]
        mem_delta = after["mem"] - before["mem"]
        rows.append(
            {
                "view_idx": i,
                "label": item["label"],
                "seq": item["seq"],
                "frame": int(item["frame"]),
                "state_update_mean_norm": float(np.linalg.norm(state_delta, axis=-1).mean()),
                "state_update_max_norm": float(np.linalg.norm(state_delta, axis=-1).max()),
                "mem_update_mean_norm": float(np.linalg.norm(mem_delta, axis=-1).mean()),
                "mem_update_max_norm": float(np.linalg.norm(mem_delta, axis=-1).max()),
            }
        )
    return rows


def load_anchor_world_metrics(args: argparse.Namespace, specs: list[dict]) -> tuple[list[dict], np.ndarray, np.ndarray]:
    raw_dir = args.raw_dir or (args.compare_root / "raw")
    token_dir = args.token_aligned_dir or (args.compare_root / "token_aligned_human_only")
    raw_poses, intrinsics = load_camera_poses(raw_dir)
    pred_cam_anchors = load_human3r_smpl_joints(raw_dir, intrinsics, TOKEN_BODY_ANCHOR_SPECS, args.device)
    raw_world = np.stack([transform_points(pose, anchors) for pose, anchors in zip(raw_poses, pred_cam_anchors)], axis=0)

    token_poses = None
    if token_dir.is_dir():
        token_poses, _ = load_camera_poses(token_dir)

    rows = []
    for i, item in enumerate(specs):
        row = {
            "view_idx": i,
            "label": item["label"],
            "seq": item["seq"],
            "frame": int(item["frame"]),
            "camera_center_step": 0.0,
            "camera_rotation_step_deg": 0.0,
            "raw_anchor_step_mean": 0.0,
            "raw_anchor_step_max": 0.0,
            "prev_human_fit_delta_t": 0.0,
            "prev_human_fit_delta_r_deg": 0.0,
            "prev_human_fit_rmse": 0.0,
            "token_aligned_delta_t_eval_only": 0.0,
            "token_aligned_delta_r_deg_eval_only": 0.0,
        }
        if i > 0:
            row["camera_center_step"] = float(np.linalg.norm(raw_poses[i][:3, 3] - raw_poses[i - 1][:3, 3]))
            row["camera_rotation_step_deg"] = rotation_step_deg(raw_poses[i - 1], raw_poses[i])
            anchor_step = np.linalg.norm(raw_world[i] - raw_world[i - 1], axis=-1)
            row["raw_anchor_step_mean"] = float(anchor_step.mean())
            row["raw_anchor_step_max"] = float(anchor_step.max())

            fit_pose = fit_rigid(pred_cam_anchors[i], raw_world[i - 1])
            fit_world = transform_points(fit_pose, pred_cam_anchors[i])
            fit_rmse = np.sqrt(np.mean(np.sum((fit_world - raw_world[i - 1]) ** 2, axis=-1)))
            delta_prev = fit_pose @ np.linalg.inv(raw_poses[i])
            row["prev_human_fit_delta_t"], row["prev_human_fit_delta_r_deg"] = transform_delta_metrics(delta_prev)
            row["prev_human_fit_rmse"] = float(fit_rmse)

        if token_poses is not None:
            delta_token = token_poses[i] @ np.linalg.inv(raw_poses[i])
            row["token_aligned_delta_t_eval_only"], row["token_aligned_delta_r_deg_eval_only"] = transform_delta_metrics(delta_token)
        rows.append(row)
    return rows, raw_world, np.asarray(raw_poses)


def merge_rows(*row_groups: list[dict]) -> list[dict]:
    merged: dict[int, dict] = {}
    for group in row_groups:
        for row in group:
            idx = int(row["view_idx"])
            merged.setdefault(idx, {}).update(row)
    return [merged[i] for i in sorted(merged)]


def add_gate_scores(rows: list[dict]) -> None:
    camera_jump = np.asarray([r["camera_center_step"] + 0.03 * r["camera_rotation_step_deg"] for r in rows])
    anchor_jump = np.asarray([r["raw_anchor_step_mean"] for r in rows])
    prev_fit_delta = np.asarray([r["prev_human_fit_delta_t"] + 0.03 * r["prev_human_fit_delta_r_deg"] for r in rows])
    state_update = np.asarray([r["state_update_mean_norm"] for r in rows])
    mem_disagree = np.asarray([1.0 - r["memory_query_topk_sim"] for r in rows])

    components = {
        "camera_jump_score": normalize01_after_first(camera_jump),
        "human_anchor_jump_score": normalize01_after_first(anchor_jump),
        "prev_human_fit_delta_score": normalize01_after_first(prev_fit_delta),
        "state_update_score": normalize01_after_first(state_update),
        "memory_disagreement_score": normalize01_after_first(mem_disagree),
    }
    no_memory_names = [
        "camera_jump_score",
        "human_anchor_jump_score",
        "prev_human_fit_delta_score",
        "state_update_score",
    ]
    for i, row in enumerate(rows):
        for name, values in components.items():
            row[name] = float(values[i])
        row["token_aligned_gate_with_memory"] = float(np.mean([values[i] for values in components.values()]))
        row["token_aligned_gate_no_memory"] = float(np.mean([components[name][i] for name in no_memory_names]))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(rows: list[dict], out_dir: Path) -> None:
    x = np.asarray([r["view_idx"] for r in rows])
    labels = [r["label"] for r in rows]

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(x, [r["camera_center_step"] for r in rows], "-o", label="raw camera center step")
    axes[0].plot(x, [r["raw_anchor_step_mean"] for r in rows], "-o", label="raw token-human anchor step")
    axes[0].plot(x, [r["prev_human_fit_delta_t"] for r in rows], "-o", label="prev-human fit delta t")
    axes[0].set_ylabel("meters / model units")
    axes[0].legend()

    axes[1].plot(x, [r["camera_rotation_step_deg"] for r in rows], "-o", label="raw camera rot step")
    axes[1].plot(x, [r["prev_human_fit_delta_r_deg"] for r in rows], "-o", label="prev-human fit delta rot")
    axes[1].plot(x, [r["token_aligned_delta_r_deg_eval_only"] for r in rows], "--o", label="token-aligned correction rot eval")
    axes[1].set_ylabel("degrees")
    axes[1].legend()

    axes[2].plot(x, [r["token_aligned_delta_t_eval_only"] for r in rows], "--o", label="token-aligned correction t eval")
    axes[2].plot(x, [r["prev_human_fit_rmse"] for r in rows], "-o", label="prev-human fit RMSE")
    axes[2].set_ylabel("m / model units")
    axes[2].legend()

    for ax in axes:
        ax.axvline(2, color="orange", linestyle="--", alpha=0.75)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=18, ha="right")
    fig.suptitle("Temporal momentum from token-aligned human anchors and raw camera pose")
    fig.tight_layout()
    fig.savefig(out_dir / "temporal_momentum_curves.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(x, [r["state_update_mean_norm"] for r in rows], "-o", label="state update mean norm")
    axes[0].plot(x, [r["mem_update_mean_norm"] for r in rows], "-o", label="memory update mean norm")
    axes[0].set_ylabel("latent update norm")
    axes[0].legend()
    axes[1].plot(x, [r["memory_query_topk_sim"] for r in rows], "-o", label="current global token -> memory top-k sim")
    axes[1].plot(x, [r["memory_disagreement_score"] for r in rows], "-o", label="memory disagreement score")
    axes[1].set_ylabel("similarity / normalized")
    axes[1].legend()
    for ax in axes:
        ax.axvline(2, color="orange", linestyle="--", alpha=0.75)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=18, ha="right")
    fig.suptitle("Recurrent latent memory visibility")
    fig.tight_layout()
    fig.savefig(out_dir / "state_memory_curves.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    component_names = [
        "camera_jump_score",
        "human_anchor_jump_score",
        "prev_human_fit_delta_score",
        "state_update_score",
        "memory_disagreement_score",
        "token_aligned_gate_with_memory",
        "token_aligned_gate_no_memory",
    ]
    for name in component_names:
        style = "-o"
        linewidth = 1.2
        if name in {"token_aligned_gate_with_memory", "token_aligned_gate_no_memory"}:
            linewidth = 2.6
        ax.plot(x, [r[name] for r in rows], style, label=name, linewidth=linewidth)
    ax.axvline(2, color="orange", linestyle="--", alpha=0.75, label="A->B boundary")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("normalized score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    ax.set_title("Current-vs-memory gate proxy from token-accessible cues")
    fig.tight_layout()
    fig.savefig(out_dir / "gate_proxy_curves.png", dpi=180)
    plt.close(fig)


def save_latent_pca(anchor_rows: list[dict], views, snapshots, dec_tokens, grid_hw, out_dir: Path) -> None:
    token_blocks = []
    colors = []
    labels = []
    for i in range(len(views)):
        state = snapshots[i + 1]["state_feat"]
        token_blocks.append(state)
        colors.extend(["#bbbbbb"] * len(state))
        labels.extend(["state"] * len(state))
    anchor_tokens = []
    anchor_labels = []
    for i, view in enumerate(views):
        for anchor in ANCHOR_NAMES:
            patch_idx, _, _ = patch_index(view.anchors[anchor], grid_hw)
            anchor_tokens.append(dec_tokens[i, patch_idx])
            anchor_labels.append(f"v{i}:{anchor}")
    token_blocks.append(np.stack(anchor_tokens, axis=0))
    X = np.concatenate(token_blocks, axis=0).astype(np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    Y = X @ vt[:2].T
    n_state = sum(len(snapshots[i + 1]["state_feat"]) for i in range(len(views)))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(Y[:n_state, 0], Y[:n_state, 1], s=4, c="#bbbbbb", alpha=0.25, label="state tokens")
    anchor_y = Y[n_state:]
    palette = {"pelvis": "red", "torso": "orange", "left_foot": "deepskyblue", "right_foot": "blue"}
    for point, label in zip(anchor_y, anchor_labels):
        anchor = label.split(":")[1]
        ax.scatter(point[0], point[1], s=38, c=palette.get(anchor, "black"), edgecolors="black", linewidths=0.4)
        ax.text(point[0], point[1], label, fontsize=7)
    ax.set_title("PCA of recurrent state tokens and token-aligned body anchor tokens")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "latent_state_anchor_pca.png", dpi=180)
    plt.close(fig)


def save_compact_dump(path: Path, dump: dict, views) -> None:
    anchor_uv = {
        f"view{view.view_idx}_{name}": np.asarray(view.anchors[name], dtype=np.float32)
        for view in views
        for name in ANCHOR_NAMES
    }
    np.savez_compressed(
        path,
        enc_tokens=dump["enc_tokens"].astype(np.float16),
        dec_tokens=dump["dec_tokens"].astype(np.float16),
        mem_queries=dump["mem_queries"].astype(np.float16),
        grid_hw=np.asarray(dump["grid_hw"], dtype=np.int32),
        **anchor_uv,
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dirs = ensure_dirs(args.output_dir)
    specs = read_case_manifest(args.compare_root)
    views = build_processed_views(args, specs)
    save_anchor_overlays(views, dirs["anchors"])

    dump = run_human3r_state_dump(args, specs)
    snapshots = dump["snapshots"]
    if len(snapshots) != len(specs) + 1:
        raise RuntimeError(f"Expected {len(specs) + 1} state snapshots, got {len(snapshots)}")

    state_anchor_rows = save_state_anchor_panels(
        views,
        snapshots,
        dump["dec_tokens"],
        dump["grid_hw"],
        dirs["state"],
    )
    memory_rows = save_memory_query_panels(specs, snapshots, dump["mem_queries"], dirs["state"], args.topk_mem)
    state_update_rows = compute_state_update_rows(specs, snapshots)
    momentum_rows, raw_world_anchors, raw_poses = load_anchor_world_metrics(args, specs)

    merged = merge_rows(memory_rows, state_update_rows, momentum_rows)
    add_gate_scores(merged)
    write_csv(args.output_dir / "state_anchor_similarity_metrics.csv", state_anchor_rows)
    write_csv(args.output_dir / "memory_momentum_gate_metrics.csv", merged)
    plot_curves(merged, dirs["curves"])
    save_latent_pca(state_anchor_rows, views, snapshots, dump["dec_tokens"], dump["grid_hw"], dirs["state"])
    save_compact_dump(dirs["dump"] / "token_state_memory_compact.npz", dump, views)

    summary = {
        "output_dir": str(args.output_dir),
        "case": [{k: item[k] for k in ("idx", "label", "seq", "frame")} for item in specs],
        "token_aligned_sources": {
            "body_anchors": ANCHOR_NAMES,
            "anchor_location_source": "AvatarReX SMPL projection is used only to locate body-part image tokens for validation.",
            "memory_source": "Human3R forward_recurrent_lighter ret_state snapshots: state_feat/state_pos/mem.",
            "global_memory_query_source": "mean image encoder token projected by pose_retriever.proj_q.",
            "gate_proxy_sources": [
                "raw camera pose jump",
                "raw world motion of Human3R predicted pelvis/torso/feet anchors",
                "current Human3R camera-space anchors fitted to previous raw world anchors",
                "recurrent state update norm",
                "current global image token vs memory top-k disagreement",
            ],
        },
        "not_used_for_gate": [
            "GT camera pose",
            "GT depth or pointmap",
            "background feature matching",
            "non-token body anchors outside pelvis/torso/left_foot/right_foot",
        ],
        "state_snapshots": len(snapshots),
        "state_shape": list(snapshots[0]["state_feat"].shape),
        "mem_shape": list(snapshots[0]["mem"].shape),
        "image_token_grid_hw": list(dump["grid_hw"]),
        "main_outputs": [
            str(dirs["anchors"] / "view0_view0_A_t.png"),
            str(dirs["state"] / "state_anchor_similarity_panel.png"),
            str(dirs["state"] / "memory_query_similarity_panel.png"),
            str(dirs["curves"] / "temporal_momentum_curves.png"),
            str(dirs["curves"] / "state_memory_curves.png"),
            str(dirs["curves"] / "gate_proxy_curves.png"),
        ],
        "per_frame_gate": merged,
    }
    with open(args.output_dir / "memory_momentum_gate_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
