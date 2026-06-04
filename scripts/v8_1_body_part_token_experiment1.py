#!/usr/bin/env python3
"""V8.1 body-part token validation, experiment 1.

This probe answers a narrow question:

    Given a GT-projected body-part anchor, does the corresponding frozen
    Human3R encoder patch token distinguish that body part from the rest of the
    image?

GT is used only to choose the query anchor and to score the response.  The
search itself is token similarity over the whole patch grid.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from v8_1_probe_aabb_case import (  # noqa: E402
    AvatarReXRawProjector,
    ProcessedView,
    draw_overlay,
    encode_views,
    load_view,
    patch_index,
)


BODY_PARTS = ["pelvis", "torso", "left_foot", "right_foot"]

DEFAULT_CASES = [
    {
        "case_id": "lbn1_22053925_22010708_00000692",
        "split": "training/lbn1",
        "raw_root": "/data/wangzheng/iJCV-CODE/data/avatarrex_lbn1",
        "seq_a": "22053925",
        "seq_b": "22010708",
        "start_frame": 692,
    },
    {
        "case_id": "zxc_22053925_22010710_00000289",
        "split": "training/zxc",
        "raw_root": "/data/wangzheng/iJCV-CODE/data/avatarrex_zxc",
        "seq_a": "22053925",
        "seq_b": "22010710",
        "start_frame": 289,
    },
    {
        "case_id": "zzr_22070935_22053926_00001117",
        "split": "training/zzr",
        "raw_root": "/data/wangzheng/iJCV-CODE/data/avatarrex_zzr",
        "seq_a": "22070935",
        "seq_b": "22053926",
        "start_frame": 1117,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data"))
    parser.add_argument("--model_path", type=Path, default=REPO_ROOT / "src" / "human3r_896L.pth")
    parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "output" / "v8_1_body_part_token_experiment1")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--min_smpl_iou", type=float, default=0.50)
    parser.add_argument("--case_id", default="all", help="Run one default case_id or all.")
    return parser.parse_args()


def ensure_case_dirs(case_dir: Path) -> dict[str, Path]:
    dirs = {
        "explicit": case_dir / "explicit_overlays",
        "heatmaps": case_dir / "body_part_heatmaps",
        "topk": case_dir / "topk_overlays",
        "panels": case_dir / "panels",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def cosine_similarity_grid(tokens: np.ndarray, token_idx: int, grid_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    query = tokens[token_idx]
    query = query / max(float(np.linalg.norm(query)), 1e-8)
    feats = tokens / np.maximum(np.linalg.norm(tokens, axis=1, keepdims=True), 1e-8)
    sim = feats @ query
    sim_grid = sim.reshape(grid_hw)
    norm_grid = (sim_grid - sim_grid.min()) / max(float(sim_grid.max() - sim_grid.min()), 1e-8)
    return sim_grid.astype(np.float32), norm_grid.astype(np.float32)


def patch_mask(mask: np.ndarray, grid_hw: tuple[int, int], patch_size: int, threshold: float = 0.10) -> np.ndarray:
    gh, gw = grid_hw
    out = np.zeros((gh, gw), dtype=bool)
    for y in range(gh):
        for x in range(gw):
            y0, y1 = y * patch_size, min((y + 1) * patch_size, mask.shape[0])
            x0, x1 = x * patch_size, min((x + 1) * patch_size, mask.shape[1])
            crop = mask[y0:y1, x0:x1]
            if crop.size and float(crop.mean()) >= threshold:
                out[y, x] = True
    return out


def patch_center(px: int, py: int, patch_size: int) -> tuple[float, float]:
    return (px + 0.5) * patch_size, (py + 0.5) * patch_size


def topk_indices(norm_grid: np.ndarray, k: int, exclude_idx: int | None = None) -> np.ndarray:
    flat = norm_grid.reshape(-1)
    if exclude_idx is not None:
        flat = flat.copy()
        flat[int(exclude_idx)] = -np.inf
    k = min(k, flat.size)
    return np.argsort(-flat)[:k]


def save_body_part_heatmap(rgb: np.ndarray, norm_grid: np.ndarray, out_path: Path, title: str) -> np.ndarray:
    heat = cv2.resize(norm_grid, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
    heat_u8 = np.clip(heat * 255.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    overlay = (0.52 * rgb + 0.48 * color).astype(np.uint8)
    cv2.putText(overlay, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return overlay


def draw_topk_overlay(
    rgb: np.ndarray,
    norm_grid: np.ndarray,
    view: ProcessedView,
    part: str,
    top_idx: np.ndarray,
    grid_hw: tuple[int, int],
    patch_size: int,
    out_path: Path,
) -> None:
    title = f"{view.label} {part}: top-{len(top_idx)} similarity excl. query"
    overlay = save_body_part_heatmap(rgb, norm_grid, out_path.with_suffix(".heat.png"), title)
    canvas = overlay.copy()
    target = view.masks[part].astype(np.uint8) * 255
    contours, _ = cv2.findContours(target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, (255, 0, 255), 2)

    anchor = view.anchors[part]
    cv2.drawMarker(
        canvas,
        (int(round(anchor[0])), int(round(anchor[1]))),
        (0, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=18,
        thickness=2,
    )

    gh, gw = grid_hw
    for rank, idx in enumerate(top_idx, start=1):
        py, px = divmod(int(idx), gw)
        x0, y0 = px * patch_size, py * patch_size
        x1, y1 = min(x0 + patch_size, rgb.shape[1] - 1), min(y0 + patch_size, rgb.shape[0] - 1)
        color = (255, 40, 40) if rank == 1 else (255, 220, 0)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
        cx, cy = patch_center(px, py, patch_size)
        cv2.putText(canvas, str(rank), (int(cx) - 5, int(cy) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    cv2.putText(canvas, "cyan cross=query/GT anchor, magenta=target, red/yellow=top-k excluding query", (10, canvas.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def score_part(
    view: ProcessedView,
    part: str,
    raw_grid: np.ndarray,
    norm_grid: np.ndarray,
    top_idx: np.ndarray,
    grid_hw: tuple[int, int],
    patch_size: int,
) -> dict[str, float | int | str]:
    target = patch_mask(view.masks[part], grid_hw, patch_size)
    human = patch_mask(view.masks["human"], grid_hw, patch_size)
    part_masks = {name: patch_mask(view.masks[name], grid_hw, patch_size) for name in BODY_PARTS}
    outside = ~target
    flat_target = target.reshape(-1)
    flat_human = human.reshape(-1)
    flat_norm = norm_grid.reshape(-1)
    flat_raw = raw_grid.reshape(-1)

    top_target = flat_target[top_idx]
    top_human = flat_human[top_idx]
    top_background = ~top_human
    top1 = int(top_idx[0])
    top1_py, top1_px = divmod(top1, grid_hw[1])
    top1_cx, top1_cy = patch_center(top1_px, top1_py, patch_size)
    anchor = view.anchors[part]
    pixel_error = math.sqrt((top1_cx - anchor[0]) ** 2 + (top1_cy - anchor[1]) ** 2)

    row: dict[str, float | int | str] = {
        "case_id": "",
        "view_idx": view.view_idx,
        "label": view.label,
        "seq": view.seq,
        "frame": view.frame,
        "query_part": part,
        "anchor_u": float(anchor[0]),
        "anchor_v": float(anchor[1]),
        "topk_excludes_query": 1,
        "top1_patch_x": int(top1_px),
        "top1_patch_y": int(top1_py),
        "top1_norm_sim": float(flat_norm[top1]),
        "top1_raw_sim": float(flat_raw[top1]),
        "top1_pixel_error": float(pixel_error),
        "target_mean_sim": float(flat_norm[flat_target].mean()) if flat_target.any() else 0.0,
        "outside_mean_sim": float(flat_norm[outside.reshape(-1)].mean()) if outside.any() else 0.0,
        "human_mean_sim": float(flat_norm[flat_human].mean()) if flat_human.any() else 0.0,
        "background_mean_sim": float(flat_norm[(~flat_human)].mean()) if (~flat_human).any() else 0.0,
        "top1_target_hit": int(bool(top_target[0])),
        "top1_human_hit": int(bool(top_human[0])),
        "top5_target_hit_rate": float(top_target.mean()),
        "top5_human_hit_rate": float(top_human.mean()),
        "top5_background_hit_rate": float(top_background.mean()),
    }
    row["target_outside_margin"] = float(row["target_mean_sim"]) - float(row["outside_mean_sim"])
    row["human_background_margin"] = float(row["human_mean_sim"]) - float(row["background_mean_sim"])
    for name, mask in part_masks.items():
        row[f"top5_{name}_hit_rate"] = float(mask.reshape(-1)[top_idx].mean())
    return row


def load_case_views(args: argparse.Namespace, case: dict) -> list[ProcessedView]:
    probe_args = SimpleNamespace(
        root=args.root,
        split=case["split"],
        size=args.size,
        device=args.device,
        model_path=args.model_path,
        min_smpl_iou=args.min_smpl_iou,
    )
    projector = AvatarReXRawProjector(Path(case["raw_root"]))
    frame = int(case["start_frame"])
    seq_a, seq_b = case["seq_a"], case["seq_b"]
    spec = [
        (seq_a, frame, "view0_A_t"),
        (seq_a, frame + 1, "view1_A_t1"),
        (seq_b, frame + 2, "view2_B_t2_boundary"),
        (seq_b, frame + 3, "view3_B_t3"),
    ]
    views = [load_view(probe_args, projector, i, seq, fr, label) for i, (seq, fr, label) in enumerate(spec)]
    low_iou = [(v.label, v.seq, v.frame, v.smpl_projection_iou) for v in views if v.smpl_projection_iou < args.min_smpl_iou]
    if low_iou:
        raise RuntimeError(f"SMPL projection sanity check failed for {case['case_id']}: {low_iou}")
    return views


def make_case_panel(case_dir: Path, case_id: str) -> Path:
    view_labels = ["A_t", "A_t+1", "B_t+2", "B_t+3"]
    overlay_files = [
        "view0_view0_A_t.png",
        "view1_view1_A_t1.png",
        "view2_view2_B_t2_boundary.png",
        "view3_view3_B_t3.png",
    ]
    col_labels = ["explicit overlay"] + BODY_PARTS
    cell_w, cell_h = 230, 230
    label_w, header_h, gap = 72, 38, 8
    width = label_w + len(col_labels) * cell_w + (len(col_labels) - 1) * gap
    height = header_h + len(view_labels) * cell_h + (len(view_labels) - 1) * gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    except Exception:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    for c, label in enumerate(col_labels):
        x = label_w + c * (cell_w + gap)
        draw.text((x + 6, 8), label, fill=(20, 20, 20), font=font_small)
    for r, vlabel in enumerate(view_labels):
        y = header_h + r * (cell_h + gap)
        draw.text((8, y + 96), vlabel, fill=(20, 20, 20), font=font)
        paths = [case_dir / "explicit_overlays" / overlay_files[r]]
        paths += [case_dir / "topk_overlays" / f"view{r}_{part}_topk.png" for part in BODY_PARTS]
        for c, path in enumerate(paths):
            x = label_w + c * (cell_w + gap)
            if not path.is_file():
                draw.rectangle([x, y, x + cell_w, y + cell_h], outline=(220, 0, 0), width=2)
                draw.text((x + 20, y + 100), "missing", fill=(220, 0, 0), font=font)
                continue
            im = Image.open(path).convert("RGB")
            im.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
            canvas.paste(im, (x + (cell_w - im.width) // 2, y + (cell_h - im.height) // 2))
            draw.rectangle([x, y, x + cell_w - 1, y + cell_h - 1], outline=(220, 220, 220), width=1)

    out = case_dir / "experiment1_body_part_topk_panel.png"
    canvas.save(out)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(output_dir: Path, rows: list[dict]) -> None:
    summary_rows = []
    for part in BODY_PARTS:
        part_rows = [r for r in rows if r["query_part"] == part]
        if not part_rows:
            continue
        summary_rows.append(
            {
                "part": part,
                "num": len(part_rows),
                "target_mean_sim": np.mean([float(r["target_mean_sim"]) for r in part_rows]),
                "outside_mean_sim": np.mean([float(r["outside_mean_sim"]) for r in part_rows]),
                "target_outside_margin": np.mean([float(r["target_outside_margin"]) for r in part_rows]),
                "top1_target_acc": np.mean([float(r["top1_target_hit"]) for r in part_rows]),
                "top1_human_acc": np.mean([float(r["top1_human_hit"]) for r in part_rows]),
                "top5_target_hit_rate": np.mean([float(r["top5_target_hit_rate"]) for r in part_rows]),
                "top5_human_hit_rate": np.mean([float(r["top5_human_hit_rate"]) for r in part_rows]),
                "top5_background_hit_rate": np.mean([float(r["top5_background_hit_rate"]) for r in part_rows]),
                "top1_pixel_error": np.mean([float(r["top1_pixel_error"]) for r in part_rows]),
            }
        )

    write_csv(output_dir / "experiment1_summary_by_part.csv", summary_rows)
    md = [
        "# V8.1 Body-Part Token Experiment 1",
        "",
        "GT SMPL projection is used to select the query patch token and to score the result. The search itself is cosine similarity from that query token to all frozen Human3R encoder patch tokens in the same frame. Top-k metrics exclude the query patch itself, so they measure whether other high-response patches stay on the correct body part/human rather than trivially returning the source patch.",
        "",
        "| Part | N | Target Sim | Outside Sim | Margin | Top1 Target | Top1 Human | Top5 Target | Top5 Human | Top5 BG | Top1 Pixel Err |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in summary_rows:
        md.append(
            "| {part} | {num} | {target_mean_sim:.3f} | {outside_mean_sim:.3f} | {target_outside_margin:.3f} | {top1_target_acc:.3f} | {top1_human_acc:.3f} | {top5_target_hit_rate:.3f} | {top5_human_hit_rate:.3f} | {top5_background_hit_rate:.3f} | {top1_pixel_error:.1f} |".format(
                **r
            )
        )
    md += [
        "",
        "Interpretation:",
        "",
        "- `Margin > 0` means the target body-part region is more similar to the query token than non-target patches.",
        "- `Top1/Top5 Human` checks whether high-response patches stay on the human instead of drifting to background.",
        "- Feet may respond to both feet or lower-body patches; this is acceptable for a first pose-anchor token, but persistent background hits are not acceptable.",
    ]
    (output_dir / "experiment1_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def stack_panels(output_dir: Path, panel_paths: list[Path]) -> Path:
    panels = [Image.open(p).convert("RGB") for p in panel_paths]
    max_w = max(p.width for p in panels)
    total_h = sum(p.height for p in panels) + 44 * len(panels)
    canvas = Image.new("RGB", (max_w, total_h), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    y = 0
    for panel_path, panel in zip(panel_paths, panels):
        draw.text((12, y + 12), panel_path.parent.name, fill=(20, 20, 20), font=font)
        y += 36
        canvas.paste(panel, (0, y))
        y += panel.height + 8
    out = output_dir / "all_experiment1_body_part_topk_panels.png"
    canvas.save(out)
    return out


def run_case(args: argparse.Namespace, case: dict) -> tuple[list[dict], Path]:
    case_dir = args.output_dir / case["case_id"]
    dirs = ensure_case_dirs(case_dir)
    views = load_case_views(args, case)
    for view in views:
        draw_overlay(view, dirs["explicit"] / f"view{view.view_idx}_{view.label}.png")

    probe_args = SimpleNamespace(model_path=args.model_path, device=args.device)
    tokens, grid_hw = encode_views(probe_args, views)
    rows = []
    for view in views:
        for part in BODY_PARTS:
            query_idx, _, _ = patch_index(view.anchors[part], grid_hw)
            raw_grid, norm_grid = cosine_similarity_grid(tokens[view.view_idx], query_idx, grid_hw)
            top_idx = topk_indices(norm_grid, args.topk, exclude_idx=query_idx)
            heat_path = dirs["heatmaps"] / f"view{view.view_idx}_{part}_heatmap.png"
            save_body_part_heatmap(view.rgb, norm_grid, heat_path, f"{view.label} {part} token similarity")
            topk_path = dirs["topk"] / f"view{view.view_idx}_{part}_topk.png"
            draw_topk_overlay(view.rgb, norm_grid, view, part, top_idx, grid_hw, args.patch_size, topk_path)
            row = score_part(view, part, raw_grid, norm_grid, top_idx, grid_hw, args.patch_size)
            row["case_id"] = case["case_id"]
            rows.append(row)

    write_csv(case_dir / "experiment1_metrics.csv", rows)
    with (case_dir / "case_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(case, f, indent=2, sort_keys=True)
    panel_path = make_case_panel(case_dir, case["case_id"])
    return rows, panel_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = DEFAULT_CASES
    if args.case_id != "all":
        cases = [case for case in DEFAULT_CASES if case["case_id"] == args.case_id]
        if not cases:
            raise ValueError(f"Unknown case_id {args.case_id}. Valid: {[c['case_id'] for c in DEFAULT_CASES]}")

    all_rows = []
    panel_paths = []
    for case in cases:
        print(f"Running Experiment 1 case: {case['case_id']}")
        rows, panel = run_case(args, case)
        all_rows.extend(rows)
        panel_paths.append(panel)

    write_csv(args.output_dir / "experiment1_metrics.csv", all_rows)
    write_summary(args.output_dir, all_rows)
    combined = stack_panels(args.output_dir, panel_paths)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "num_rows": len(all_rows),
                "combined_panel": str(combined),
                "summary": str(args.output_dir / "experiment1_summary.md"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
