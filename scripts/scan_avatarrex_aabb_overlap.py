#!/usr/bin/env python3
"""Scan AvatarReX AABB boundary candidates by feature overlap.

For each candidate (camA, camB, start_frame), the AABB views are:
  (camA, start), (camA, start+1), (camB, start+2), (camB, start+3)

The shot boundary used by the anchor path is therefore:
  (camA, start+1) -> (camB, start+2)

This script ranks candidates by boundary feature overlap while also reporting
GT camera pose jump, so we can find samples that are both matchable and visibly
discontinuous.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
SRC_ROOT = REPO_ROOT / "src"
XFEAT_ROOT = REPO_ROOT.parent / "xfeat-for-Movie3R"
for path in [REPO_ROOT, SRC_ROOT, XFEAT_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from modules.xfeat import XFeat  # noqa: E402
from scripts.build_video_anchor_cache import (  # noqa: E402
    apply_affine,
    compute_fundamental_inliers,
    crop_xy_to_patch,
    fit_affine,
    human3r_crop_meta,
    patch_center_norm,
    patch_error,
    raw_to_crop_xy,
    resize_for_matching,
    to_original_coords,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(REPO_ROOT.parent / "data" / "RICH_4Human3R"))
    parser.add_argument("--split", default="Training")
    parser.add_argument("--source_sequence", default="BBQ_001_guitar")
    parser.add_argument("--camera_pairs", default="all_ordered", help="all_ordered, all_unordered, or comma list like 0-1,1-2")
    parser.add_argument("--start_frame", type=int, default=None)
    parser.add_argument("--end_frame", type=int, default=None)
    parser.add_argument("--frame_stride", type=int, default=10)
    parser.add_argument("--max_candidates", type=int, default=None)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--top_k_xfeat", type=int, default=8192)
    parser.add_argument("--top_k_tokens", type=int, default=16)
    parser.add_argument("--fundamental_thresh", type=float, default=2.0)
    parser.add_argument("--max_dim", type=int, default=1200)
    parser.add_argument("--out_dir", default=str(REPO_ROOT / "output" / "guitar_aabb_overlap_scan"))
    parser.add_argument("--top_n", type=int, default=100)
    return parser.parse_args()


def seq_name(source_sequence, cam_idx):
    return f"{source_sequence}_cam_{int(cam_idx):02d}"


def available_cameras(split_root, source_sequence):
    cams = []
    for path in sorted(split_root.glob(f"{source_sequence}_cam_*")):
        if not path.is_dir():
            continue
        try:
            cams.append(int(path.name.rsplit("_", 1)[1]))
        except ValueError:
            continue
    return cams


def parse_camera_pairs(text, cameras):
    if text == "all_ordered":
        return [(a, b) for a in cameras for b in cameras if a != b]
    if text == "all_unordered":
        return [(a, b) for i, a in enumerate(cameras) for b in cameras[i + 1 :]]
    pairs = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        a, b = item.split("-", 1)
        pairs.append((int(a), int(b)))
    return pairs


def available_frames(split_root, source_sequence, cameras):
    frame_sets = []
    for cam in cameras:
        rgb_dir = split_root / seq_name(source_sequence, cam) / "rgb"
        frames = {int(p.stem) for p in rgb_dir.glob("*.png")}
        frame_sets.append(frames)
    common = sorted(set.intersection(*frame_sets))
    if not common:
        raise RuntimeError(f"no common frames found under {split_root}")
    return common


def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def load_pose(cam_path):
    data = np.load(cam_path)
    if "pose" in data.files:
        return np.asarray(data["pose"], dtype=np.float64)
    if "camera_pose" in data.files:
        return np.asarray(data["camera_pose"], dtype=np.float64)
    raise KeyError(f"pose key not found in {cam_path}: {data.files}")


def rotation_angle_deg(pose_a, pose_b):
    ra = pose_a[:3, :3]
    rb = pose_b[:3, :3]
    rel = ra.T @ rb
    cos_angle = (np.trace(rel) - 1.0) / 2.0
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return float(math.degrees(math.acos(cos_angle)))


def camera_jump(split_root, seq_a, ref_frame, seq_b, cur_frame):
    pose_a = load_pose(split_root / seq_a / "cam" / f"{ref_frame:08d}.npz")
    pose_b = load_pose(split_root / seq_b / "cam" / f"{cur_frame:08d}.npz")
    t_jump = float(np.linalg.norm(pose_b[:3, 3] - pose_a[:3, 3]))
    r_jump = rotation_angle_deg(pose_a, pose_b)
    return t_jump, r_jump


def quality_gate(num_anchors, affine_median_error):
    count_gate = float(np.clip((num_anchors - 4.0) / 12.0, 0.0, 1.0))
    residual_gate = float(np.clip(1.0 - affine_median_error / 4.0, 0.0, 1.0))
    return count_gate * residual_gate


def process_candidate(args, split_root, xfeat, cam_a, cam_b, start_frame):
    seq_a = seq_name(args.source_sequence, cam_a)
    seq_b = seq_name(args.source_sequence, cam_b)
    ref_frame = int(start_frame) + 1
    cur_frame = int(start_frame) + 2
    ref_path = split_root / seq_a / "rgb" / f"{ref_frame:08d}.png"
    cur_path = split_root / seq_b / "rgb" / f"{cur_frame:08d}.png"
    ref_bgr = read_image(ref_path)
    cur_bgr = read_image(cur_path)

    ref_match, sx_ref, sy_ref = resize_for_matching(ref_bgr, args.max_dim)
    cur_match, sx_cur, sy_cur = resize_for_matching(cur_bgr, args.max_dim)
    mkpts_ref, mkpts_cur = xfeat.match_xfeat_star(ref_match, cur_match, top_k=args.top_k_xfeat)
    mkpts_ref = np.asarray(mkpts_ref, dtype=np.float32)
    mkpts_cur = np.asarray(mkpts_cur, dtype=np.float32)
    mkpts_ref_orig = to_original_coords(mkpts_ref, sx_ref, sy_ref)
    mkpts_cur_orig = to_original_coords(mkpts_cur, sx_cur, sy_cur)
    fundamental_mask = compute_fundamental_inliers(mkpts_ref, mkpts_cur, args.fundamental_thresh)

    ref_meta = human3r_crop_meta(ref_bgr.shape, args.size)
    cur_meta = human3r_crop_meta(cur_bgr.shape, args.size)
    patch_size = 16
    ref_h, ref_w = ref_meta["final_size_wh"][1], ref_meta["final_size_wh"][0]
    cur_h, cur_w = cur_meta["final_size_wh"][1], cur_meta["final_size_wh"][0]
    ref_grid_hw = (ref_h // patch_size, ref_w // patch_size)
    cur_grid_hw = (cur_h // patch_size, cur_w // patch_size)

    ref_crop_xy, ref_crop_valid = raw_to_crop_xy(mkpts_ref_orig, ref_meta)
    cur_crop_xy, cur_crop_valid = raw_to_crop_xy(mkpts_cur_orig, cur_meta)
    _, ref_patch_idx, ref_patch_valid = crop_xy_to_patch(ref_crop_xy, ref_crop_valid, patch_size, ref_grid_hw)
    _, cur_patch_idx, cur_patch_valid = crop_xy_to_patch(cur_crop_xy, cur_crop_valid, patch_size, cur_grid_hw)
    valid = fundamental_mask & ref_patch_valid & cur_patch_valid

    best_by_pair = {}
    for idx in np.flatnonzero(valid):
        pair = (int(ref_patch_idx[idx]), int(cur_patch_idx[idx]))
        if pair in best_by_pair:
            continue
        best_by_pair[pair] = {
            "ref_pos_norm": patch_center_norm(pair[0], ref_grid_hw),
            "cur_pos_norm": patch_center_norm(pair[1], cur_grid_hw),
        }

    anchors = list(best_by_pair.values())
    if len(anchors) >= 3:
        ref_norm = np.array([a["ref_pos_norm"] for a in anchors], dtype=np.float32)
        cur_norm = np.array([a["cur_pos_norm"] for a in anchors], dtype=np.float32)
        weights = np.ones((len(anchors),), dtype=np.float32)
        affine = fit_affine(ref_norm, cur_norm, weights)
        if affine is None:
            affine = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        affine_err = patch_error(apply_affine(ref_norm, affine), cur_norm, cur_grid_hw)
        affine_median_error = float(np.median(affine_err))
        delta_patch = np.linalg.norm((cur_norm - ref_norm) * np.array(cur_grid_hw[::-1], dtype=np.float32), axis=1)
        delta_patch_median = float(np.median(delta_patch))
        delta_patch_p90 = float(np.percentile(delta_patch, 90))
    else:
        affine_median_error = float("inf")
        delta_patch_median = None
        delta_patch_p90 = None

    q_gate = quality_gate(len(anchors), affine_median_error if np.isfinite(affine_median_error) else 4.0)
    pose_t_jump, pose_r_jump_deg = camera_jump(split_root, seq_a, ref_frame, seq_b, cur_frame)

    return {
        "source_sequence": args.source_sequence,
        "cam_a": int(cam_a),
        "cam_b": int(cam_b),
        "start_frame": int(start_frame),
        "aabb_frames": [int(start_frame), int(start_frame + 1), int(start_frame + 2), int(start_frame + 3)],
        "ref_seq": seq_a,
        "ref_frame": int(ref_frame),
        "cur_seq": seq_b,
        "cur_frame": int(cur_frame),
        "raw_matches": int(len(mkpts_ref)),
        "fundamental_inliers": int(fundamental_mask.sum()),
        "unique_anchor_patch_pairs": int(len(anchors)),
        "top_k_tokens": int(min(args.top_k_tokens, len(anchors))),
        "quality_gate": float(q_gate),
        "affine_median_patch_error": affine_median_error,
        "anchor_delta_patch_median": delta_patch_median,
        "anchor_delta_patch_p90": delta_patch_p90,
        "pose_jump_t": pose_t_jump,
        "pose_jump_deg": pose_r_jump_deg,
    }


def sort_key(record):
    return (
        record["unique_anchor_patch_pairs"],
        record["quality_gate"],
        record["fundamental_inliers"],
        record["raw_matches"],
        record["pose_jump_t"],
    )


def write_csv(path, records):
    fieldnames = [
        "rank",
        "cam_a",
        "cam_b",
        "start_frame",
        "ref_frame",
        "cur_frame",
        "unique_anchor_patch_pairs",
        "quality_gate",
        "fundamental_inliers",
        "raw_matches",
        "affine_median_patch_error",
        "anchor_delta_patch_median",
        "anchor_delta_patch_p90",
        "pose_jump_t",
        "pose_jump_deg",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, record in enumerate(records, start=1):
            row = {key: record.get(key) for key in fieldnames if key != "rank"}
            row["rank"] = rank
            writer.writerow(row)


def write_top_markdown(path, records, top_n):
    lines = [
        "# AvatarReX Guitar AABB Overlap Candidates",
        "",
        "Sorted by unique anchor patch pairs, then quality gate, fundamental inliers, raw matches, and camera translation jump.",
        "",
        "|rank|camA->camB|start|boundary|unique|q_gate|F inliers|raw|affine err|pose t|pose deg|",
        "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, r in enumerate(records[:top_n], start=1):
        lines.append(
            f"|{rank}|{r['cam_a']:02d}->{r['cam_b']:02d}|{r['start_frame']}|{r['ref_frame']}->{r['cur_frame']}|"
            f"{r['unique_anchor_patch_pairs']}|{r['quality_gate']:.4f}|{r['fundamental_inliers']}|{r['raw_matches']}|"
            f"{r['affine_median_patch_error']:.4f}|{r['pose_jump_t']:.4f}|{r['pose_jump_deg']:.2f}|"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    split_root = Path(args.root) / args.split
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cameras = available_cameras(split_root, args.source_sequence)
    pairs = parse_camera_pairs(args.camera_pairs, cameras)
    frames = available_frames(split_root, args.source_sequence, cameras)
    min_start = frames[0] if args.start_frame is None else args.start_frame
    max_start = frames[-1] - 3 if args.end_frame is None else args.end_frame
    frame_set = set(frames)
    starts = [
        f for f in range(min_start, max_start + 1, args.frame_stride)
        if all((f + offset) in frame_set for offset in range(4))
    ]
    candidates = [(cam_a, cam_b, start) for start in starts for cam_a, cam_b in pairs]
    if args.max_candidates is not None:
        candidates = candidates[: args.max_candidates]

    print(f"cameras: {cameras}")
    print(f"pairs: {len(pairs)}, starts: {len(starts)}, candidates: {len(candidates)}")
    print(f"out_dir: {out_dir}")

    xfeat = XFeat(top_k=args.top_k_xfeat)
    raw_path = out_dir / "results.jsonl"
    skipped_path = out_dir / "skipped.jsonl"
    records = []
    skipped = []
    with raw_path.open("w", encoding="utf-8") as raw_f, skipped_path.open("w", encoding="utf-8") as skip_f:
        for cam_a, cam_b, start in tqdm(candidates):
            try:
                record = process_candidate(args, split_root, xfeat, cam_a, cam_b, start)
            except Exception as exc:
                skip = {"cam_a": int(cam_a), "cam_b": int(cam_b), "start_frame": int(start), "error": repr(exc)}
                skipped.append(skip)
                skip_f.write(json.dumps(skip, ensure_ascii=False) + "\n")
                skip_f.flush()
                continue
            records.append(record)
            raw_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            raw_f.flush()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    records_sorted = sorted(records, key=sort_key, reverse=True)
    sorted_jsonl = out_dir / "results_sorted.jsonl"
    with sorted_jsonl.open("w", encoding="utf-8") as f:
        for record in records_sorted:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_csv(out_dir / "results_sorted.csv", records_sorted)
    write_top_markdown(out_dir / "top_candidates.md", records_sorted, args.top_n)

    summary = {
        "args": vars(args),
        "cameras": cameras,
        "num_pairs": len(pairs),
        "num_starts": len(starts),
        "num_candidates": len(candidates),
        "num_ok": len(records),
        "num_skipped": len(skipped),
        "sorted_jsonl": str(sorted_jsonl),
        "sorted_csv": str(out_dir / "results_sorted.csv"),
        "top_candidates_md": str(out_dir / "top_candidates.md"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
