#!/usr/bin/env python3
"""Redraw a compact Step1 report for one RICH AABB boundary pair.

This keeps only the figures needed to show that XFeat+mesh anchors remain
visible in Human3R encoder patch tokens.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
ACCEL_ROOT = Path("/workspace/code/accelerated_features")
ACCEL_SCRIPTS = ACCEL_ROOT / "scripts"

for path in [REPO_ROOT, SRC_ROOT, ACCEL_ROOT, ACCEL_SCRIPTS, Path(__file__).resolve().parent]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import verify_rich_anchor_encoder_similarity as base  # noqa: E402
from dust3r.model_human3r import load_model  # noqa: E402
from modules.xfeat import XFeat  # noqa: E402
from test_rich_aabb_xfeat_geometry import (  # noqa: E402
    load_mask,
    resize_for_matching,
    to_original_coords,
)
from test_rich_aabb_xfeat_mesh_geometry import (  # noqa: E402
    build_visible_vertex_map,
    evaluate_mesh_geometry,
)
from visualize_rich_mesh_projection import load_ply_vertices  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rich_root", default="/workspace/data/RICH")
    parser.add_argument("--data_root", default="/workspace/data/RICH/RICH_4Human3R/Training")
    parser.add_argument("--source_sequence", default="BBQ_001_guitar")
    parser.add_argument("--cam_a", type=int, default=6)
    parser.add_argument("--cam_b", type=int, default=7)
    parser.add_argument("--start_frame", type=int, default=244)
    parser.add_argument("--model_path", default=str(REPO_ROOT / "src" / "human3r_896L.pth"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--top_k", type=int, default=8192)
    parser.add_argument("--max_dim", type=int, default=1200)
    parser.add_argument("--mesh_max_dim", type=int, default=1400)
    parser.add_argument("--mesh_lookup_radius", type=int, default=4)
    parser.add_argument("--mesh_z_tol", type=float, default=0.03)
    parser.add_argument("--reproj_thresh", type=float, default=24.0)
    parser.add_argument("--max_raw_draw", type=int, default=80)
    parser.add_argument("--max_patch_draw", type=int, default=48)
    parser.add_argument("--num_similarity_examples", type=int, default=4)
    parser.add_argument(
        "--out_dir",
        default=str(
            REPO_ROOT
            / "output"
            / "anchor_token_report_v1"
            / "01_aabb_step1"
            / "BBQ_001_guitar_cam06_cam07_f00000244"
            / "pair_01_A_t1_to_B_t2_BOUNDARY"
            / "clean_step1_v2"
        ),
    )
    return parser.parse_args()


def save_rgb(path, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def put_label(img, text, org, scale=0.62, color=(255, 255, 255), thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_grid(rgb, patch_size, color=(245, 230, 60)):
    out = rgb.copy()
    h, w = out.shape[:2]
    overlay = out.copy()
    for x in range(0, w + 1, patch_size):
        cv2.line(overlay, (x, 0), (x, h), color, 1, cv2.LINE_AA)
    for y in range(0, h + 1, patch_size):
        cv2.line(overlay, (0, y), (w, y), color, 1, cv2.LINE_AA)
    return cv2.addWeighted(overlay, 0.55, out, 0.45, 0)


def draw_grid_light(rgb, patch_size, color=(225, 210, 60), alpha=0.28):
    out = rgb.copy()
    h, w = out.shape[:2]
    overlay = out.copy()
    for x in range(0, w + 1, patch_size):
        cv2.line(overlay, (x, 0), (x, h), color, 1, cv2.LINE_AA)
    for y in range(0, h + 1, patch_size):
        cv2.line(overlay, (0, y), (w, y), color, 1, cv2.LINE_AA)
    return cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0)


def draw_raw_mesh_inliers(ref_bgr, cur_bgr, mkpts_ref, mkpts_cur, mesh_indices, out_path, max_draw):
    scale = min(1.0, 920.0 / max(ref_bgr.shape[:2] + cur_bgr.shape[:2]))
    ref = cv2.resize(ref_bgr, (int(ref_bgr.shape[1] * scale), int(ref_bgr.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    cur = cv2.resize(cur_bgr, (int(cur_bgr.shape[1] * scale), int(cur_bgr.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
    cur = cv2.cvtColor(cur, cv2.COLOR_BGR2RGB)
    h = max(ref.shape[0], cur.shape[0])
    w = ref.shape[1] + cur.shape[1]
    canvas = np.full((h + 78, w, 3), 18, dtype=np.uint8)
    canvas[78 : 78 + ref.shape[0], : ref.shape[1]] = ref
    canvas[78 : 78 + cur.shape[0], ref.shape[1] : ref.shape[1] + cur.shape[1]] = cur
    put_label(canvas, f"1. semi-dense XFeat + RICH mesh inliers: {len(mesh_indices)}", (14, 32), 0.72)
    put_label(canvas, "left: reference A@t+1     right: current B@t+2", (14, 62), 0.54, (210, 210, 210), 1)

    ids = np.asarray(mesh_indices, dtype=np.int64)
    if len(ids) > max_draw:
        # Prefer spatial coverage rather than random clutter.
        ids = ids[np.linspace(0, len(ids) - 1, max_draw).round().astype(np.int64)]
    overlay = canvas.copy()
    for n, idx in enumerate(ids):
        hue = int(255 * (n % 32) / 31)
        bgr = cv2.applyColorMap(np.array([[hue]], dtype=np.uint8), cv2.COLORMAP_HSV)[0, 0]
        color = (int(bgr[2]), int(bgr[1]), int(bgr[0]))
        p0 = (int(round(mkpts_ref[idx, 0] * scale)), int(round(78 + mkpts_ref[idx, 1] * scale)))
        p1 = (int(round(ref.shape[1] + mkpts_cur[idx, 0] * scale)), int(round(78 + mkpts_cur[idx, 1] * scale)))
        cv2.line(overlay, p0, p1, color, 1, cv2.LINE_AA)
        cv2.circle(overlay, p0, 3, color, -1, cv2.LINE_AA)
        cv2.circle(overlay, p1, 3, color, -1, cv2.LINE_AA)
    canvas[78:] = cv2.addWeighted(overlay[78:], 0.82, canvas[78:], 0.18, 0)
    save_rgb(out_path, canvas)


def patch_center(patch_idx, patch_size, grid_hw):
    gh, gw = grid_hw
    y = int(patch_idx) // gw
    x = int(patch_idx) % gw
    return np.array([(x + 0.5) * patch_size, (y + 0.5) * patch_size], dtype=np.float32)


def draw_crop(rgb, anchors, patch_size, grid_hw, side, out_path):
    out = draw_grid(rgb, patch_size)
    for a in anchors:
        key = "ref_patch_idx" if side == "ref" else "cur_patch_idx"
        c = patch_center(a[key], patch_size, grid_hw)
        cv2.circle(out, (int(round(c[0])), int(round(c[1]))), 4, (255, 0, 255), -1, cv2.LINE_AA)
    h, w = out.shape[:2]
    canvas = np.full((h + 58, w, 3), 18, dtype=np.uint8)
    canvas[58 : 58 + h] = out
    label = "2. reference Human3R crop" if side == "ref" else "3. current Human3R crop"
    put_label(canvas, f"{label}: {grid_hw[0]} x {grid_hw[1]} patches, anchors={len(anchors)}", (12, 36), 0.62)
    save_rgb(out_path, canvas)


def draw_patch_correspondences(ref_rgb, cur_rgb, anchors, patch_size, ref_grid_hw, cur_grid_hw, out_path, max_draw):
    ref = draw_grid(ref_rgb, patch_size)
    cur = draw_grid(cur_rgb, patch_size)
    h = max(ref.shape[0], cur.shape[0])
    w = ref.shape[1] + cur.shape[1]
    canvas = np.full((h + 78, w, 3), 18, dtype=np.uint8)
    canvas[78 : 78 + ref.shape[0], : ref.shape[1]] = ref
    canvas[78 : 78 + cur.shape[0], ref.shape[1] : ref.shape[1] + cur.shape[1]] = cur
    put_label(canvas, f"4. mesh anchors mapped to Human3R patch tokens: {len(anchors)}", (14, 32), 0.72)
    put_label(canvas, "each line: one verified ref-patch -> cur-patch correspondence", (14, 62), 0.54, (210, 210, 210), 1)

    order = np.argsort([a["mesh_error_px"] for a in anchors])
    if len(order) > max_draw:
        order = order[:max_draw]
    overlay = canvas.copy()
    for n, idx in enumerate(order):
        a = anchors[int(idx)]
        hue = int(255 * (n % 32) / 31)
        bgr = cv2.applyColorMap(np.array([[hue]], dtype=np.uint8), cv2.COLORMAP_HSV)[0, 0]
        color = (int(bgr[2]), int(bgr[1]), int(bgr[0]))
        p0 = patch_center(a["ref_patch_idx"], patch_size, ref_grid_hw)
        p1 = patch_center(a["cur_patch_idx"], patch_size, cur_grid_hw)
        p0 = (int(round(p0[0])), int(round(78 + p0[1])))
        p1 = (int(round(ref.shape[1] + p1[0])), int(round(78 + p1[1])))
        cv2.line(overlay, p0, p1, color, 1, cv2.LINE_AA)
        cv2.circle(overlay, p0, 4, color, -1, cv2.LINE_AA)
        cv2.circle(overlay, p1, 4, color, -1, cv2.LINE_AA)
    canvas[78:] = cv2.addWeighted(overlay[78:], 0.88, canvas[78:], 0.12, 0)
    save_rgb(out_path, canvas)


def draw_marker_with_label(img, center, label, color, radius=8):
    x, y = int(round(center[0])), int(round(center[1]))
    cv2.circle(img, (x, y), radius, color, 2, cv2.LINE_AA)
    cv2.circle(img, (x, y), 2, color, -1, cv2.LINE_AA)
    put_label(img, label, (x + 8, max(18, y - 8)), 0.48, color, 2)


def boxes_overlap(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def draw_marker_with_label_light(img, center, label, color, radius=5, existing_boxes=None, offsets=None):
    x, y = int(round(center[0])), int(round(center[1]))
    cv2.circle(img, (x, y), radius, color, 1, cv2.LINE_AA)
    cv2.circle(img, (x, y), 1, color, -1, cv2.LINE_AA)
    if not label:
        return

    if existing_boxes is None:
        existing_boxes = []
    if offsets is None:
        offsets = [(7, -7), (7, 13), (-18, -7), (-18, 13), (10, -20), (-24, 24), (18, 24)]

    scale = 0.34
    thickness = 1
    text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    img_h, img_w = img.shape[:2]
    chosen = None
    chosen_box = None
    for dx, dy in offsets:
        org_x = int(np.clip(x + dx, 2, max(2, img_w - text_size[0] - 3)))
        org_y = int(np.clip(y + dy, text_size[1] + 3, max(text_size[1] + 3, img_h - baseline - 3)))
        box = (org_x - 2, org_y - text_size[1] - 2, org_x + text_size[0] + 2, org_y + baseline + 2)
        if not any(boxes_overlap(box, prev) for prev in existing_boxes):
            chosen = (org_x, org_y)
            chosen_box = box
            break
    if chosen is None:
        org_x = int(np.clip(x + offsets[0][0], 2, max(2, img_w - text_size[0] - 3)))
        org_y = int(np.clip(y + offsets[0][1], text_size[1] + 3, max(text_size[1] + 3, img_h - baseline - 3)))
        chosen = (org_x, org_y)
        chosen_box = (org_x - 2, org_y - text_size[1] - 2, org_x + text_size[0] + 2, org_y + baseline + 2)
    existing_boxes.append(chosen_box)
    cv2.putText(img, label, (chosen[0] + 1, chosen[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (12, 12, 12), thickness, cv2.LINE_AA)
    cv2.putText(img, label, chosen, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_similarity_examples(out_dir, anchors, ref_rgb, cur_rgb, sim_matrix, patch_size, ref_grid_hw, cur_grid_hw, max_examples):
    if not anchors:
        return []
    order = sorted(range(len(anchors)), key=lambda i: (anchors[i]["encoder_rank"], -anchors[i]["encoder_cosine"]))
    chosen = order[:max_examples]
    paths = []
    for n, idx in enumerate(chosen):
        a = anchors[idx]
        row = sim_matrix[a["ref_patch_idx"]]
        top = np.argsort(-row)[:5]
        top1 = int(top[0])
        top2 = int(top[1]) if len(top) > 1 else top1
        margin = float(row[top1] - row[top2])

        sim_map = row.reshape(cur_grid_hw)
        hm = base.heatmap_image(sim_map, cur_rgb.shape[:2])
        overlay = np.clip(0.42 * cur_rgb.astype(np.float32) + 0.58 * hm.astype(np.float32), 0, 255).astype(np.uint8)
        # **========== 原始代码 ==========
        # ref_marked = draw_grid(ref_rgb, patch_size)
        # cur_marked = draw_grid(overlay, patch_size)
        # **========== 新代码 ==========
        ref_marked = draw_grid_light(ref_rgb, patch_size)
        cur_marked = draw_grid_light(overlay, patch_size)
        # **========== 结束 ==========

        ref_center = patch_center(a["ref_patch_idx"], patch_size, ref_grid_hw)
        true_center = patch_center(a["cur_patch_idx"], patch_size, cur_grid_hw)
        # **========== 原始代码 ==========
        # draw_marker_with_label(ref_marked, ref_center, "ref", (255, 0, 255), 9)
        # draw_marker_with_label(cur_marked, true_center, "true", (255, 0, 255), 10)
        # **========== 新代码 ==========
        draw_marker_with_label_light(ref_marked, ref_center, "ref", (255, 0, 255), 6)
        label_boxes = []
        draw_marker_with_label_light(cur_marked, true_center, "true", (255, 0, 255), 7, label_boxes)
        # **========== 结束 ==========
        for rank, cur_idx in enumerate(top, start=1):
            center = patch_center(cur_idx, patch_size, cur_grid_hw)
            color = (255, 220, 40) if cur_idx != a["cur_patch_idx"] else (40, 255, 80)
            # **========== 原始代码 ==========
            # draw_marker_with_label(cur_marked, center, f"top{rank}", color, 7)
            # **========== 新代码 ==========
            draw_marker_with_label_light(cur_marked, center, str(rank), color, 5, label_boxes)
            # **========== 结束 ==========

        h = max(ref_marked.shape[0], cur_marked.shape[0])
        w = ref_marked.shape[1] + cur_marked.shape[1]
        # **========== 原始代码 ==========
        # canvas = np.full((h + 92, w, 3), 18, dtype=np.uint8)
        # canvas[92 : 92 + ref_marked.shape[0], : ref_marked.shape[1]] = ref_marked
        # canvas[92 : 92 + cur_marked.shape[0], ref_marked.shape[1] : ref_marked.shape[1] + cur_marked.shape[1]] = cur_marked
        # **========== 新代码 ==========
        canvas = np.full((h + 76, w, 3), 18, dtype=np.uint8)
        canvas[76 : 76 + ref_marked.shape[0], : ref_marked.shape[1]] = ref_marked
        canvas[76 : 76 + cur_marked.shape[0], ref_marked.shape[1] : ref_marked.shape[1] + cur_marked.shape[1]] = cur_marked
        # **========== 结束 ==========
        put_label(
            canvas,
            # **========== 原始代码 ==========
            # f"5. encoder similarity map anchor #{idx}: true-rank={a['encoder_rank']}, cosine={a['encoder_cosine']:.3f}, top1-top2 margin={margin:.3f}",
            # **========== 新代码 ==========
            f"5. encoder similarity anchor #{idx}: rank={a['encoder_rank']}, cos={a['encoder_cosine']:.3f}, margin={margin:.3f}",
            # **========== 结束 ==========
            (12, 28),
            0.47,
            (255, 255, 255),
            1,
        )
        # **========== 原始代码 ==========
        # put_label(canvas, "right: red/yellow = higher cosine; magenta=true mesh anchor; green/yellow rings=top similar patches", (12, 66), 0.48, (210, 210, 210), 1)
        # **========== 新代码 ==========
        put_label(canvas, "right: warm=higher cosine | magenta=true anchor | small numbers=top-5 token matches", (12, 55), 0.38, (210, 210, 210), 1)
        # **========== 结束 ==========
        path = out_dir / f"04_similarity_map_anchor_{n:02d}.jpg"
        save_rgb(path, canvas)
        paths.append(path)
    return paths


def main():
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seq_a = base.seq_name(args.source_sequence, args.cam_a)
    seq_b = base.seq_name(args.source_sequence, args.cam_b)
    ref_frame = args.start_frame + 1
    cur_frame = args.start_frame + 2
    ref_bgr, ref_path = base.load_rgb(args.data_root, seq_a, ref_frame)
    cur_bgr, cur_path = base.load_rgb(args.data_root, seq_b, cur_frame)

    print("Running semi-dense XFeat + mesh verification...")
    ref_match, sx_ref, sy_ref = resize_for_matching(ref_bgr, args.max_dim)
    cur_match, sx_cur, sy_cur = resize_for_matching(cur_bgr, args.max_dim)
    xfeat = XFeat(top_k=args.top_k)
    mkpts_ref, mkpts_cur = xfeat.match_xfeat_star(ref_match, cur_match, top_k=args.top_k)
    mkpts_ref = np.asarray(mkpts_ref, dtype=np.float32)
    mkpts_cur = np.asarray(mkpts_cur, dtype=np.float32)
    mkpts_ref_orig = to_original_coords(mkpts_ref, sx_ref, sy_ref)
    mkpts_cur_orig = to_original_coords(mkpts_cur, sx_cur, sy_cur)
    del xfeat
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mesh_path = Path(args.rich_root) / "scan_calibration" / "BBQ" / "scan_camcoord.ply"
    xyz, _ = load_ply_vertices(mesh_path)
    ref_map = build_visible_vertex_map(xyz, args.rich_root, seq_a, ref_bgr.shape, args.mesh_max_dim, args.mesh_z_tol)
    cur_map = build_visible_vertex_map(xyz, args.rich_root, seq_b, cur_bgr.shape, args.mesh_max_dim, args.mesh_z_tol)
    mask_ref = load_mask(args.data_root, seq_a, ref_frame, ref_bgr.shape)
    mask_cur = load_mask(args.data_root, seq_b, cur_frame, cur_bgr.shape)
    eval_items = evaluate_mesh_geometry(mkpts_ref_orig, mkpts_cur_orig, ref_map, cur_map, mask_ref, mask_cur, args)
    mesh_mask = np.array([item["mesh_inlier"] for item in eval_items], dtype=bool)
    mesh_indices = np.flatnonzero(mesh_mask)
    draw_raw_mesh_inliers(
        ref_bgr,
        cur_bgr,
        mkpts_ref_orig,
        mkpts_cur_orig,
        mesh_indices,
        out_dir / "00_semidense_xfeat_mesh_inliers.jpg",
        args.max_raw_draw,
    )

    print("Encoding Human3R crops...")
    model = load_model(args.model_path, device=device, verbose=True).eval()
    model.gradient_checkpointing = False
    patch_size = int(model.croco_args["patch_size"])
    ref_img_tensor, ref_true_shape, ref_crop_rgb, ref_meta = base.load_human3r_image(ref_path, args.size)
    cur_img_tensor, cur_true_shape, cur_crop_rgb, cur_meta = base.load_human3r_image(cur_path, args.size)
    ref_img_tensor = ref_img_tensor.to(device)
    cur_img_tensor = cur_img_tensor.to(device)
    ref_true_shape = ref_true_shape.to(device)
    cur_true_shape = cur_true_shape.to(device)
    with torch.no_grad():
        ref_feat = model._encode_image(ref_img_tensor, ref_true_shape)[0][-1]
        cur_feat = model._encode_image(cur_img_tensor, cur_true_shape)[0][-1]

    h_ref, w_ref = map(int, ref_true_shape[0].detach().cpu().numpy().tolist())
    h_cur, w_cur = map(int, cur_true_shape[0].detach().cpu().numpy().tolist())
    ref_grid_hw = (h_ref // patch_size, w_ref // patch_size)
    cur_grid_hw = (h_cur // patch_size, w_cur // patch_size)

    ref_crop_xy, ref_crop_valid = base.raw_to_crop_xy(mkpts_ref_orig, ref_meta)
    cur_crop_xy, cur_crop_valid = base.raw_to_crop_xy(mkpts_cur_orig, cur_meta)
    ref_patch_xy, ref_patch_idx, ref_patch_valid = base.crop_xy_to_patch(ref_crop_xy, ref_crop_valid, patch_size, ref_grid_hw)
    cur_patch_xy, cur_patch_idx, cur_patch_valid = base.crop_xy_to_patch(cur_crop_xy, cur_crop_valid, patch_size, cur_grid_hw)
    mapped_valid = mesh_mask & ref_patch_valid & cur_patch_valid

    best_by_pair = {}
    for idx in np.flatnonzero(mapped_valid):
        pair = (int(ref_patch_idx[idx]), int(cur_patch_idx[idx]))
        err = eval_items[int(idx)]["best_mesh_reproj_error_px"]
        err_val = float(err) if err is not None else float("inf")
        if pair not in best_by_pair or err_val < best_by_pair[pair]["mesh_error_px"]:
            best_by_pair[pair] = {
                "match_index": int(idx),
                "ref_patch_idx": pair[0],
                "cur_patch_idx": pair[1],
                "ref_patch_xy": ref_patch_xy[idx].astype(int).tolist(),
                "cur_patch_xy": cur_patch_xy[idx].astype(int).tolist(),
                "mesh_error_px": err_val,
            }
    anchors = list(best_by_pair.values())

    ref_norm = F.normalize(ref_feat[0].float(), dim=-1)
    cur_norm = F.normalize(cur_feat[0].float(), dim=-1)
    sim_matrix = (ref_norm @ cur_norm.T).detach().cpu().numpy()
    for a in anchors:
        ri = a["ref_patch_idx"]
        ci = a["cur_patch_idx"]
        sim = float(sim_matrix[ri, ci])
        a["encoder_cosine"] = sim
        a["encoder_rank"] = int((sim_matrix[ri] > sim).sum() + 1)

    draw_crop(ref_crop_rgb, anchors, patch_size, ref_grid_hw, "ref", out_dir / "01_ref_human3r_crop.jpg")
    draw_crop(cur_crop_rgb, anchors, patch_size, cur_grid_hw, "cur", out_dir / "02_cur_human3r_crop.jpg")
    draw_patch_correspondences(
        ref_crop_rgb,
        cur_crop_rgb,
        anchors,
        patch_size,
        ref_grid_hw,
        cur_grid_hw,
        out_dir / "03_human3r_patch_anchor_correspondences.jpg",
        args.max_patch_draw,
    )
    sim_paths = draw_similarity_examples(
        out_dir,
        anchors,
        ref_crop_rgb,
        cur_crop_rgb,
        sim_matrix,
        patch_size,
        ref_grid_hw,
        cur_grid_hw,
        args.num_similarity_examples,
    )

    print(
        {
            "out_dir": str(out_dir),
            "mesh_geometry_inliers": int(mesh_mask.sum()),
            "unique_anchor_patch_pairs": len(anchors),
            "ref_grid_hw": ref_grid_hw,
            "cur_grid_hw": cur_grid_hw,
            "similarity_examples": [str(p) for p in sim_paths],
        }
    )


if __name__ == "__main__":
    main()
