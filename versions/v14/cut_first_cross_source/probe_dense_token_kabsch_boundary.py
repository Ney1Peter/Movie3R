#!/usr/bin/env python3
"""Probe an explicit boundary from built-in dense token correspondences.

Unlike B0's implicit camera proposal, this diagnostic obtains a second rigid
candidate directly from geometry already produced by frozen Human3R:

    last-pre encoder image tokens <-> first-post clean-raw encoder image tokens
    mutual descriptor matches + their predicted 3D pointmap locations
    robust SO(3)+translation Kabsch
    B_geom: post raw local gauge -> pre-shot gauge.

No image model is added: the encoder image tokens are already computed by
Human3R and are exposed read-only by an opt-in diagnostic flag.  They are used
before recurrent decoder/state mixing, unlike the rejected decoder-token
variant. The script is a diagnostic only.  It does not select, commit, or tune
a runtime policy; GT is read after both B0 and B_geom are constructed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
for path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts", Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dust3r.model import ARCroco3DStereo  # noqa: E402
from dust3r.utils.device import todevice  # noqa: E402
from dust3r.v14_outputs import boundary_from_camera_predictions  # noqa: E402
from evaluate_cut_events import add_mhmr_inputs, homogeneous, read_jsonl  # noqa: E402
from evaluate_four_source_b0 import load_views, safe_name  # noqa: E402
from v9_learned_stream_alignment_overfit import gt_pose_from_view  # noqa: E402
from versions.v14.run_v14_2_single_sequence import (  # noqa: E402
    camera_matrix,
    configure_model,
    model_batch_from_gt,
    rotation_error_deg,
    set_event_indices,
)


CHECKPOINT = REPO_ROOT / "output/v14_cut_first_cross_source/v14_cut_first_cross_source_96ps_e6/checkpoint-final.pth"
DEFAULT_RECORDS = REPO_ROOT / "config/manifests/v14_vsp_pair_disjoint_20260802/dev_all.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "output/v14/fine_alignment_research/dense_token_kabsch_vsp_dev"
SOURCES = ("avatarrex", "thuman", "mvhuman100", "mvhuman200")
PATCH_SIZE = 16
MAX_MATCHES = 256
RANSAC_ITERATIONS = 512
RANSAC_INLIER_M = .30
MIN_INLIERS = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=CHECKPOINT)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sources", nargs="+", choices=SOURCES, default=SOURCES)
    parser.add_argument("--max-cases-per-source", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0, "mean": float("nan"), "median": float("nan"), "p90": float("nan"), "p95": float("nan")}
    return {"count": int(len(array)), "mean": float(array.mean()), "median": float(np.median(array)), "p90": float(np.quantile(array, .90)), "p95": float(np.quantile(array, .95))}


def load_event_views(record: dict[str, Any], args: argparse.Namespace) -> list[dict]:
    loader_args = SimpleNamespace(data_root=args.data_root, resolution=(512, 288), resize_mode="human3r_demo", boundary=2)
    views = load_views(record, loader_args)
    if len(views) < 3:
        raise RuntimeError(f"Expected 3 views, got {len(views)}")
    return views[:3]


def camera_metrics(camera: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    translation = float(np.linalg.norm(camera[:3, 3] - target[:3, 3]))
    rotation = rotation_error_deg(camera, target)
    return {"translation_m": translation, "rotation_deg": rotation, "composite": translation + .02 * rotation, "catastrophic": bool(translation > 1.0 or rotation > 30.0)}


def rigid_fit(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return B such that row points satisfy ``target ~= source @ R.T + t``."""
    if source.shape != target.shape or len(source) < 3:
        raise ValueError("Kabsch needs matching Nx3 arrays with N>=3")
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    covariance = (source - source_mean).T @ (target - target_mean)
    left, _, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_t[-1] *= -1.0
        rotation = right_t.T @ left.T
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_mean - rotation @ source_mean
    return transform


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def mutual_matches(pre_tokens: torch.Tensor, post_tokens: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pre = torch.nn.functional.normalize(pre_tokens.detach().float()[0], dim=-1)
    post = torch.nn.functional.normalize(post_tokens.detach().float()[0], dim=-1)
    similarity = (pre @ post.T).cpu().numpy()
    pre_to_post = similarity.argmax(axis=1)
    post_to_pre = similarity.argmax(axis=0)
    pre_index = np.arange(len(pre_to_post), dtype=np.int64)
    keep = post_to_pre[pre_to_post] == pre_index
    pre_index, post_index = pre_index[keep], pre_to_post[keep]
    scores = similarity[pre_index, post_index]
    order = np.argsort(-scores)[:MAX_MATCHES]
    return pre_index[order], post_index[order], scores[order]


def token_point_samples(prediction: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tokens = prediction.get("v14_encoder_image_tokens")
    positions = prediction.get("v14_encoder_image_positions")
    points = prediction.get("pts3d_in_other_view")
    confidence = prediction.get("conf")
    if not all(torch.is_tensor(value) for value in (tokens, positions, points, confidence)):
        raise RuntimeError("Missing attached scene tokens / pointmap / confidence")
    tokens_np = tokens.detach().float().cpu().numpy()[0]
    positions_np = positions.detach().cpu().numpy()[0]
    points_np = points.detach().float().cpu().numpy()[0]
    confidence_np = confidence.detach().float().cpu().numpy()[0]
    grid_h, grid_w = points_np.shape[:2]
    token_h, token_w = int(positions_np[:, 0].max()) + 1, int(positions_np[:, 1].max()) + 1
    if token_h * token_w != len(tokens_np):
        raise RuntimeError(f"Token grid mismatch: N={len(tokens_np)}, positions={token_h}x{token_w}")
    # DINO positions enumerate the image patch lattice.  Sample the centre of
    # the corresponding pointmap cell instead of assuming a decoder layout.
    rr = np.clip(((positions_np[:, 0] + .5) * grid_h / token_h).astype(np.int64), 0, grid_h - 1)
    cc = np.clip(((positions_np[:, 1] + .5) * grid_w / token_w).astype(np.int64), 0, grid_w - 1)
    return tokens_np, points_np[rr, cc], confidence_np[rr, cc]


def robust_kabsch(source: np.ndarray, target: np.ndarray, scores: np.ndarray, seed_text: str) -> tuple[np.ndarray | None, dict[str, Any]]:
    valid = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    source, target, scores = source[valid], target[valid], scores[valid]
    if len(source) < MIN_INLIERS:
        return None, {"match_count": int(len(source)), "reason": "too_few_valid_matches"}
    seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16], 16) % (2**32)
    generator = np.random.default_rng(seed)
    best_inliers: np.ndarray | None = None
    best_key = (-1, float("inf"))
    for _ in range(RANSAC_ITERATIONS):
        indices = generator.choice(len(source), size=3, replace=False)
        try:
            candidate = rigid_fit(source[indices], target[indices])
        except np.linalg.LinAlgError:
            continue
        residual = np.linalg.norm(transform_points(candidate, source) - target, axis=1)
        inliers = residual <= RANSAC_INLIER_M
        key = (int(inliers.sum()), float(np.median(residual[inliers])) if inliers.any() else float("inf"))
        if key[0] > best_key[0] or (key[0] == best_key[0] and key[1] < best_key[1]):
            best_inliers, best_key = inliers, key
    if best_inliers is None or int(best_inliers.sum()) < MIN_INLIERS:
        return None, {"match_count": int(len(source)), "reason": "ransac_insufficient_inliers", "best_inlier_count": 0 if best_inliers is None else int(best_inliers.sum())}
    transform = rigid_fit(source[best_inliers], target[best_inliers])
    residual = np.linalg.norm(transform_points(transform, source) - target, axis=1)
    inliers = residual <= RANSAC_INLIER_M
    # One refinement is enough for this diagnostic.  Keep the fixed threshold,
    # rather than tuning a case-specific acceptance threshold from GT.
    if int(inliers.sum()) >= MIN_INLIERS:
        transform = rigid_fit(source[inliers], target[inliers])
        residual = np.linalg.norm(transform_points(transform, source) - target, axis=1)
    return transform, {
        "match_count": int(len(source)), "inlier_count": int(inliers.sum()), "inlier_ratio": float(inliers.mean()),
        "similarity_mean": float(scores.mean()), "similarity_p10": float(np.quantile(scores, .10)),
        "residual_median_m": float(np.median(residual)), "residual_p90_m": float(np.quantile(residual, .90)),
        "inlier_residual_median_m": float(np.median(residual[inliers])) if inliers.any() else float("nan"),
    }


def forward(model: ARCroco3DStereo, views: list[dict], device: torch.device, pattern_id: str) -> dict[str, Any]:
    clean = todevice(model_batch_from_gt(views), device)
    shadow_views = set_event_indices(clean, {2})
    raw_views = set_event_indices(clean[2:3], set())
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
        shadow, _ = model.forward_recurrent_lighter(shadow_views, str(device), ret_state=False, use_ttt3r=False)
        raw, _ = model.forward_recurrent_lighter(raw_views, str(device), ret_state=False, use_ttt3r=False)
    pre_camera = homogeneous(camera_matrix(shadow[1]))
    raw_camera = homogeneous(camera_matrix(raw[0]))
    shadow_camera = homogeneous(camera_matrix(shadow[2]))
    b0_boundary = boundary_from_camera_predictions(shadow[2], raw[0])[0].detach().float().cpu().numpy().astype(np.float64)
    b0 = b0_boundary @ raw_camera
    parity = float(np.max(np.abs(b0 - shadow_camera)))
    if parity > 1e-5:
        raise RuntimeError(f"B0 parity failed ({parity})")
    pre_tokens, pre_points, pre_confidence = token_point_samples(shadow[1])
    post_tokens, post_points, post_confidence = token_point_samples(raw[0])
    pre_index, post_index, similarity = mutual_matches(
        shadow[1]["v14_encoder_image_tokens"], raw[0]["v14_encoder_image_tokens"]
    )
    confidence = np.minimum(pre_confidence[pre_index], post_confidence[post_index])
    keep = np.isfinite(confidence) & (confidence > 0.0)
    geom, diagnostics = robust_kabsch(post_points[post_index][keep], pre_points[pre_index][keep], similarity[keep], pattern_id)
    diagnostics |= {"mutual_match_count_before_confidence": int(len(pre_index)), "matched_confidence_mean": float(confidence[keep].mean()) if keep.any() else float("nan")}
    return {"pre_camera": pre_camera, "b0": b0, "geom": geom, "diagnostics": diagnostics, "parity": parity}


def evaluate_case(model: ARCroco3DStereo, record: dict[str, Any], args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    views = load_event_views(record, args); add_mhmr_inputs(views)
    started = time.perf_counter()
    result = forward(model, views, device, str(record["pattern_id"]))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    gt_pre = homogeneous(gt_pose_from_view(views[1]).detach().float().cpu().numpy())
    gt_post = homogeneous(gt_pose_from_view(views[2]).detach().float().cpu().numpy())
    target = result["pre_camera"] @ np.linalg.inv(gt_pre) @ gt_post
    row: dict[str, Any] = {"b0_metrics": camera_metrics(result["b0"], target), "geometry": result["diagnostics"], "b0_parity_max_abs": result["parity"], "timing_seconds": time.perf_counter() - started}
    if result["geom"] is not None:
        row["geom_metrics"] = camera_metrics(result["geom"], target)
        row["geometry"]["candidate_available"] = True
        row["cameras_evaluation_only"] = {"b0": result["b0"], "geom": result["geom"], "target": target}
    else:
        row["geometry"]["candidate_available"] = False
    return json_ready(row)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    available = [row for row in rows if row["geometry"]["candidate_available"]]
    def method(key: str, items: list[dict[str, Any]]) -> dict[str, Any]:
        values = [row[key] for row in items]
        return {"count": len(values), "translation_m": stats([x["translation_m"] for x in values]), "rotation_deg": stats([x["rotation_deg"] for x in values]), "composite": stats([x["composite"] for x in values]), "catastrophic_count": int(sum(x["catastrophic"] for x in values))}
    result = {"case_count": len(rows), "candidate_available": len(available), "availability": float(len(available) / max(len(rows), 1)), "b0": method("b0_metrics", rows)}
    if available:
        result["b0_on_available"] = method("b0_metrics", available)
        result["geom"] = method("geom_metrics", available)
        result["geom_relative_composite_gain_on_available"] = float(1.0 - result["geom"]["composite"]["mean"] / result["b0_on_available"]["composite"]["mean"])
        result["geometry"] = {
            key: stats([row["geometry"].get(key, float("nan")) for row in available])
            for key in ("match_count", "inlier_count", "inlier_ratio", "similarity_mean", "residual_median_m", "residual_p90_m")
        }
    return result


def main() -> None:
    args = parse_args()
    for path in (args.model_path, args.records):
        if not path.is_file():
            raise FileNotFoundError(path)
    selected, counts = [], defaultdict(int)
    for record in read_jsonl(args.records):
        source = str(record["source"])
        if source not in args.sources or (args.max_cases_per_source and counts[source] >= args.max_cases_per_source):
            continue
        selected.append(record); counts[source] += 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = args.output_dir / "cases"; cases_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = ARCroco3DStereo.from_pretrained(str(args.model_path)).to(device)
    flags = configure_model(model)
    model.return_v14_encoder_tokens = True
    rows, failures = [], []
    for index, record in enumerate(selected, 1):
        path = cases_dir / f"{safe_name(record['pattern_id'])}.json"
        cached = json.loads(path.read_text(encoding="utf-8")) if path.is_file() and not args.overwrite else None
        if cached is not None and cached.get("status") == "ok":
            row = cached
        else:
            try:
                row = evaluate_case(model, record, args, device)
                row |= {"status": "ok", "source": record["source"], "record": record}
            except Exception as error:
                row = {"status": "failed", "source": record["source"], "record": record, "error": repr(error), "traceback": traceback.format_exc()}
            path.write_text(json.dumps(row, indent=2, allow_nan=True) + "\n", encoding="utf-8")
        if row["status"] == "ok":
            rows.append(row)
            print(f"[{index:03d}/{len(selected):03d}] {record['source']} b0={row['b0_metrics']['composite']:.3f} geom={row.get('geom_metrics', {}).get('composite', float('nan')):.3f} avail={row['geometry']['candidate_available']}", flush=True)
        else:
            failures.append(row); print(f"[{index:03d}/{len(selected):03d}] FAILED {row['error']}", flush=True)
        if device.type == "cuda" and index % 10 == 0:
            torch.cuda.empty_cache()
    report = {"experiment": "frozen_cross96_encoder_token_pointmap_kabsch_boundary_diagnostic", "records": str(args.records), "checkpoint": {"path": str(args.model_path), "sha256": sha256(args.model_path), "flags": flags}, "runtime_inputs": "existing Human3R pre-decoder encoder image tokens, predicted pointmaps/confidence, last-pre + first-post clean raw only", "fixed_parameters": {"patch_size": PATCH_SIZE, "max_mutual_matches": MAX_MATCHES, "ransac_iterations": RANSAC_ITERATIONS, "ransac_inlier_m": RANSAC_INLIER_M, "minimum_inliers": MIN_INLIERS}, "summary": summarize(rows), "by_source": {source: summarize([row for row in rows if row["source"] == source]) for source in args.sources}, "failures": failures, "cases": rows}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    print(args.output_dir / "report.json")


if __name__ == "__main__":
    main()
