#!/usr/bin/env python3
"""P4 CPU-only visibility audit for SIFT correspondences inside runtime person boxes."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = REPO_ROOT / "output/v14/fine_alignment_research/p1_foot_scene_observability_v2"
DEFAULT_OUTPUT = REPO_ROOT / "output/v14/fine_alignment_research/p4_rgb_person_correspondence_visibility"
RATIO, INNER_MARGIN, MIN_MATCHES, MIN_CELLS = .70, .10, 8, 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def workspace(path: Path) -> None:
    if not str(path.resolve()).startswith(str(REPO_ROOT.resolve())):
        raise ValueError(f"P4 artifact outside Movie3R workspace: {path}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def image_path(cache_dir: Path, camera: int, frame: int) -> Path:
    return cache_dir / "cache/frame_cache/input_frames" / f"cam{camera}" / f"{frame:06d}.jpg"


def load_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None: raise FileNotFoundError(path)
    return cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)


def in_inner_bbox(point: tuple[float, float], bbox: Any) -> bool:
    box = np.asarray(bbox, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all(): return False
    delta = (box[2:] - box[:2]) * INNER_MARGIN
    return bool(box[0] + delta[0] <= point[0] <= box[2] - delta[0] and box[1] + delta[1] <= point[1] <= box[3] - delta[1])


def cell(point: tuple[float, float], bbox: Any) -> int | None:
    box = np.asarray(bbox, dtype=np.float64); span = box[2:] - box[:2]
    if np.any(span <= 1e-6): return None
    xy = (np.asarray(point) - box[:2]) / span
    if not ((xy >= 0).all() and (xy <= 1).all()): return None
    x, y = np.minimum((xy * 3).astype(int), 2)
    return int(y * 3 + x)


def audit(args: argparse.Namespace) -> Path:
    index_path = args.cache_dir / "P1_CACHE_INDEX.json"; index = json.loads(index_path.read_text(encoding="utf-8"))
    if int(index.get("schema", -1)) != 2: raise RuntimeError("P4 requires P1 schema-2")
    sift, matcher, rows = cv2.SIFT_create(), cv2.BFMatcher(cv2.NORM_L2), []
    for path_text in index["case_paths"]:
        cached = torch.load(path_text, map_location="cpu", weights_only=False); runtime = cached["runtime"]
        if runtime["runtime_contract"]["gt_used"] or runtime["runtime_contract"]["future_post_frames_used"]: raise RuntimeError("invalid runtime contract")
        record, event = runtime["record"], str(runtime["record"]["event_id"])
        pre = load_gray(image_path(args.cache_dir, int(record["pre_camera"]), int(record["frame"])))
        post = load_gray(image_path(args.cache_dir, int(record["post_camera"]), int(record["frame"])))
        kp_pre, desc_pre = sift.detectAndCompute(pre, None); kp_post, desc_post = sift.detectAndCompute(post, None)
        raw = [] if desc_pre is None or desc_post is None else [first for first, second in matcher.knnMatch(desc_pre, desc_post, k=2) if first.distance < RATIO * second.distance]
        for pre_index, post_index in runtime["association"]["pairs"]:
            person_pre, person_post = runtime["pre_people"][int(pre_index)], runtime["b0_post_people"][int(post_index)]
            local = [match for match in raw if in_inner_bbox(kp_pre[match.queryIdx].pt, person_pre["bbox"]) and in_inner_bbox(kp_post[match.trainIdx].pt, person_post["bbox"])]
            cells_pre = {cell(kp_pre[m.queryIdx].pt, person_pre["bbox"]) for m in local}; cells_post = {cell(kp_post[m.trainIdx].pt, person_post["bbox"]) for m in local}
            cells_pre.discard(None); cells_post.discard(None)
            rows.append({"event_id": event, "pre_index": int(pre_index), "post_index": int(post_index), "runtime_match_count": len(local), "pre_cell_count": len(cells_pre), "post_cell_count": len(cells_post), "pnp_observable": bool(len(local) >= MIN_MATCHES and len(cells_pre) >= MIN_CELLS and len(cells_post) >= MIN_CELLS)})
    observable = [row for row in rows if row["pnp_observable"]]
    report = {"experiment": "v14_p4_rgb_person_sift_visibility", "status": "GO_TO_SEPARATE_PNP_PROTOCOL" if len(observable) / max(len(rows), 1) >= .20 else "NO_GO_RGB_PERSON_SIFT_OBSERVABILITY", "cache_index": str(index_path), "cache_index_sha256": sha256(index_path), "policy": {"resize": 512, "descriptor": "opencv_sift", "ratio": RATIO, "bbox_inner_margin": INNER_MARGIN, "pnp_min_matches": MIN_MATCHES, "pnp_min_cells_each_image": MIN_CELLS, "selection": "none"}, "counts": {"matched_runtime_rows": len(rows), "pnp_observable_rows": len(observable), "coverage": float(len(observable) / max(len(rows), 1)), "match_count_mean": float(np.mean([r["runtime_match_count"] for r in rows])), "match_count_median": float(np.median([r["runtime_match_count"] for r in rows])), "match_count_max": int(max([r["runtime_match_count"] for r in rows], default=0))}, "runtime_invariants": {"gt_evaluator_access": False, "future_post_frames_used": 0, "camera_update": "none", "external_pretrained_models": []}, "rows": rows}
    args.output_dir.mkdir(parents=True, exist_ok=True); out = args.output_dir / "P4_RGB_PERSON_SIFT_VISIBILITY_REPORT.json"; out.write_text(json.dumps(jsonable(report), indent=2)+"\n", encoding="utf-8"); return out


def main() -> None:
    args = parse_args(); workspace(args.cache_dir); workspace(args.output_dir); print(audit(args), flush=True)


if __name__ == "__main__": main()
