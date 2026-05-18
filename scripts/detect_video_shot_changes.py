#!/usr/bin/env python3
"""Detect candidate shot-change frame pairs in a video.

The detector is intentionally two-stage:
1. Scan all adjacent frame pairs with cheap image-difference metrics.
2. Run XFeat + fundamental-matrix RANSAC only on the top candidates.

It writes JSON/Markdown summaries and preview images so candidates can be
checked visually before using them as anchor boundaries.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
XFEAT_ROOT = REPO_ROOT.parent / "xfeat-for-Movie3R"
for path in [REPO_ROOT, REPO_ROOT / "src", XFEAT_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq_path", required=True, help="Input video path.")
    parser.add_argument(
        "--out_dir",
        default=str(REPO_ROOT / "output" / "shot_change_probe_v1"),
        help="Output root directory.",
    )
    parser.add_argument("--fast_max_dim", type=int, default=320, help="Resize max dimension for cheap scan.")
    parser.add_argument("--xfeat_max_dim", type=int, default=1200, help="Resize max dimension for XFeat.")
    parser.add_argument("--top_k_candidates", type=int, default=8)
    parser.add_argument("--top_k_xfeat", type=int, default=8192)
    parser.add_argument("--fundamental_thresh", type=float, default=2.0)
    parser.add_argument("--no_xfeat", action="store_true", help="Skip XFeat verification.")
    parser.add_argument("--max_draw_matches", type=int, default=120)
    return parser.parse_args()


def resize_keep_aspect(img, max_dim):
    h, w = img.shape[:2]
    if max_dim <= 0 or max(h, w) <= max_dim:
        return img
    scale = float(max_dim) / float(max(h, w))
    return cv2.resize(
        img,
        (int(round(w * scale)), int(round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )


def read_video_frames(video_path, max_dim):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frames.append(resize_keep_aspect(frame, max_dim))
    cap.release()
    if len(frames) < 2:
        raise RuntimeError(f"need at least two frames, got {len(frames)}")
    return frames, {"fps": fps, "width": width, "height": height, "frame_count": len(frames)}


def read_frame(video_path, frame_index, max_dim):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"failed to read frame {frame_index} from {video_path}")
    return resize_keep_aspect(frame, max_dim)


def hsv_hist(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist = hist.astype(np.float32)
    hist /= float(hist.sum() + 1e-8)
    return hist


def pair_metrics(frame_a, frame_b):
    if frame_a.shape[:2] != frame_b.shape[:2]:
        frame_b = cv2.resize(frame_b, (frame_a.shape[1], frame_a.shape[0]), interpolation=cv2.INTER_AREA)
    abs_diff = cv2.absdiff(frame_a, frame_b).astype(np.float32) / 255.0
    mean_abs_diff = float(abs_diff.mean())
    p95_abs_diff = float(np.percentile(abs_diff, 95))

    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    gray_abs_diff = float(cv2.absdiff(gray_a, gray_b).mean() / 255.0)

    corr = float(cv2.compareHist(hsv_hist(frame_a), hsv_hist(frame_b), cv2.HISTCMP_CORREL))
    hist_diff = float(np.clip(1.0 - corr, 0.0, 2.0))
    return {
        "mean_abs_diff": mean_abs_diff,
        "p95_abs_diff": p95_abs_diff,
        "gray_abs_diff": gray_abs_diff,
        "hist_diff": hist_diff,
    }


def robust_z(values):
    arr = np.asarray(values, dtype=np.float32)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    # **========== Original code: tiny MAD on quantized metrics could create huge z-scores ==========**
    # scale = 1.4826 * mad + 1e-6
    # return (arr - med) / scale
    # **========== End ==========**
    if mad < 1e-5:
        scale = float(arr.std()) + 1e-6
    else:
        scale = 1.4826 * mad + 1e-6
    return np.clip((arr - med) / scale, -50.0, 50.0)


def add_scores(rows):
    mean_z = robust_z([r["mean_abs_diff"] for r in rows])
    gray_z = robust_z([r["gray_abs_diff"] for r in rows])
    hist_z = robust_z([r["hist_diff"] for r in rows])
    p95_z = robust_z([r["p95_abs_diff"] for r in rows])
    for i, row in enumerate(rows):
        row["mean_abs_z"] = float(mean_z[i])
        row["gray_abs_z"] = float(gray_z[i])
        row["hist_z"] = float(hist_z[i])
        row["p95_abs_z"] = float(p95_z[i])
        row["shot_score"] = float(
            max(0.0, hist_z[i])
            + 0.7 * max(0.0, mean_z[i])
            + 0.5 * max(0.0, gray_z[i])
            + 0.3 * max(0.0, p95_z[i])
        )
    return rows


def compute_fundamental_inliers(points_ref, points_cur, threshold):
    if len(points_ref) < 8:
        return 0
    _, mask = cv2.findFundamentalMat(
        np.asarray(points_ref, dtype=np.float32),
        np.asarray(points_cur, dtype=np.float32),
        cv2.FM_RANSAC,
        float(threshold),
        0.999,
    )
    if mask is None:
        return 0
    return int(mask.reshape(-1).astype(bool).sum())


def put_label(img, text, xy):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)


def display_path(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def save_pair_preview(frame_a, frame_b, row, out_path):
    h = min(frame_a.shape[0], frame_b.shape[0])
    if frame_a.shape[0] != h:
        scale = h / float(frame_a.shape[0])
        frame_a = cv2.resize(frame_a, (int(frame_a.shape[1] * scale), h), interpolation=cv2.INTER_AREA)
    if frame_b.shape[0] != h:
        scale = h / float(frame_b.shape[0])
        frame_b = cv2.resize(frame_b, (int(frame_b.shape[1] * scale), h), interpolation=cv2.INTER_AREA)
    canvas = np.concatenate([frame_a, frame_b], axis=1)
    put_label(canvas, f"frame {row['ref_index']}", (12, 30))
    put_label(canvas, f"frame {row['cur_index']}", (frame_a.shape[1] + 12, 30))
    put_label(canvas, f"score={row['shot_score']:.2f} hist={row['hist_diff']:.3f} abs={row['mean_abs_diff']:.3f}", (12, canvas.shape[0] - 18))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def save_match_preview(frame_a, frame_b, pts_a, pts_b, row, out_path, max_draw):
    h = max(frame_a.shape[0], frame_b.shape[0])
    w_a = frame_a.shape[1]
    canvas = np.zeros((h, w_a + frame_b.shape[1], 3), dtype=np.uint8)
    canvas[: frame_a.shape[0], :w_a] = frame_a
    canvas[: frame_b.shape[0], w_a : w_a + frame_b.shape[1]] = frame_b
    n = min(len(pts_a), max_draw)
    if n > 0:
        indices = np.linspace(0, len(pts_a) - 1, n).round().astype(np.int64)
        for idx in indices:
            rng = np.random.default_rng(int(idx))
            color = tuple(int(v) for v in rng.integers(40, 255, size=3))
            p0 = tuple(np.round(pts_a[idx]).astype(int).tolist())
            p1_raw = np.round(pts_b[idx]).astype(int)
            p1 = (int(p1_raw[0] + w_a), int(p1_raw[1]))
            cv2.circle(canvas, p0, 2, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, p1, 2, color, -1, cv2.LINE_AA)
            cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)
    put_label(
        canvas,
        f"{row['ref_index']}->{row['cur_index']} matches={row.get('xfeat_matches', 0)} inliers={row.get('fundamental_inliers', 0)}",
        (12, 30),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def save_score_plot(rows, out_path):
    width = max(900, len(rows) * 7)
    height = 420
    margin = 52
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    keys = [
        ("shot_score", (20, 20, 220)),
        ("hist_diff", (20, 160, 20)),
        ("mean_abs_diff", (220, 120, 20)),
    ]
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    cv2.rectangle(canvas, (margin, margin), (margin + plot_w, margin + plot_h), (230, 230, 230), 1)
    for key, color in keys:
        vals = np.asarray([r[key] for r in rows], dtype=np.float32)
        lo = float(vals.min())
        hi = float(vals.max())
        denom = hi - lo + 1e-8
        pts = []
        for i, val in enumerate(vals):
            x = margin + int(round(i / max(1, len(vals) - 1) * plot_w))
            y = margin + plot_h - int(round((float(val) - lo) / denom * plot_h))
            pts.append((x, y))
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(canvas, a, b, color, 2, cv2.LINE_AA)
        put_label(canvas, f"{key}: min={lo:.3f} max={hi:.3f}", (margin + 12, margin + 26 + 28 * keys.index((key, color))))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def run_xfeat_for_candidates(video_path, rows, out_dir, args):
    from modules.xfeat import XFeat

    xfeat = XFeat(top_k=args.top_k_xfeat)
    for rank, row in enumerate(rows):
        frame_a = read_frame(video_path, row["ref_index"], args.xfeat_max_dim)
        frame_b = read_frame(video_path, row["cur_index"], args.xfeat_max_dim)
        pts_a, pts_b = xfeat.match_xfeat_star(frame_a, frame_b, top_k=args.top_k_xfeat)
        inliers = compute_fundamental_inliers(pts_a, pts_b, args.fundamental_thresh)
        row["xfeat_matches"] = int(len(pts_a))
        row["fundamental_inliers"] = int(inliers)
        row["fundamental_inlier_rate"] = float(inliers / len(pts_a)) if len(pts_a) else 0.0
        match_path = out_dir / f"candidate_{rank:02d}_frames_{row['ref_index']:04d}_{row['cur_index']:04d}_matches.jpg"
        save_match_preview(frame_a, frame_b, pts_a, pts_b, row, match_path, args.max_draw_matches)
        # **========== Original code: assumed output path was absolute under repo root ==========**
        # row["match_preview"] = str(match_path.relative_to(REPO_ROOT))
        # **========== End ==========**
        row["match_preview"] = display_path(match_path)
    return rows


def write_markdown(out_path, video_path, rows, meta):
    lines = [
        f"# Shot Change Probe: {video_path.name}",
        "",
        f"video: `{video_path}`",
        f"frames: {meta['frame_count']}, fps: {meta['fps']:.3f}, size: {meta['width']}x{meta['height']}",
        "",
        "| rank | frame pair | time | score | hist diff | mean abs | XFeat matches | F inliers | F inlier rate | preview |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    fps = meta["fps"] if meta["fps"] > 0 else 1.0
    for rank, row in enumerate(rows):
        pair = f"{row['ref_index']}->{row['cur_index']}"
        time = f"{row['ref_index'] / fps:.3f}s->{row['cur_index'] / fps:.3f}s"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    pair,
                    time,
                    f"{row['shot_score']:.2f}",
                    f"{row['hist_diff']:.3f}",
                    f"{row['mean_abs_diff']:.3f}",
                    str(row.get("xfeat_matches", "-")),
                    str(row.get("fundamental_inliers", "-")),
                    f"{row.get('fundamental_inlier_rate', 0.0):.3f}" if "fundamental_inlier_rate" in row else "-",
                    row.get("preview", ""),
                ]
            )
            + " |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    video_path = Path(args.seq_path)
    out_dir = Path(args.out_dir) / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    frames, meta = read_video_frames(video_path, args.fast_max_dim)
    rows = []
    for idx in range(len(frames) - 1):
        row = {
            "ref_index": int(idx),
            "cur_index": int(idx + 1),
            **pair_metrics(frames[idx], frames[idx + 1]),
        }
        rows.append(row)
    add_scores(rows)
    rows_sorted = sorted(rows, key=lambda r: r["shot_score"], reverse=True)
    candidates = rows_sorted[: args.top_k_candidates]

    for rank, row in enumerate(candidates):
        frame_a = read_frame(video_path, row["ref_index"], args.fast_max_dim)
        frame_b = read_frame(video_path, row["cur_index"], args.fast_max_dim)
        preview_path = out_dir / f"candidate_{rank:02d}_frames_{row['ref_index']:04d}_{row['cur_index']:04d}.jpg"
        save_pair_preview(frame_a, frame_b, row, preview_path)
        # **========== Original code: assumed output path was absolute under repo root ==========**
        # row["preview"] = str(preview_path.relative_to(REPO_ROOT))
        # **========== End ==========**
        row["preview"] = display_path(preview_path)

    if not args.no_xfeat:
        run_xfeat_for_candidates(video_path, candidates, out_dir, args)

    score_plot = out_dir / "score_plot.jpg"
    save_score_plot(rows, score_plot)

    summary = {
        "video": str(video_path),
        "metadata": meta,
        # **========== Original code: assumed output path was absolute under repo root ==========**
        # "score_plot": str(score_plot.relative_to(REPO_ROOT)),
        # **========== End ==========**
        "score_plot": display_path(score_plot),
        "top_candidates": candidates,
        "all_pairs": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(out_dir / "summary.md", video_path, candidates, meta)
    print(json.dumps({"out_dir": str(out_dir), "top_candidates": candidates}, indent=2))


if __name__ == "__main__":
    main()
