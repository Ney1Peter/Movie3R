#!/usr/bin/env python3
"""Probe MASt3R background correspondences as a frozen wide-baseline rotation cue."""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import PIL.Image
import torch
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
MAST3R_ROOT = Path(
    "/data/wangzheng/Movie3R/Video-OnlineHMR/thirdparty/MASt3R-SLAM/thirdparty/mast3r"
)
for path in (str(MAST3R_ROOT), str(MAST3R_ROOT / "dust3r"), str(REPO_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from dust3r.inference import inference  # noqa: E402
from dust3r.utils.image import ImgNorm  # noqa: E402
from mast3r.fast_nn import fast_reciprocal_NNs  # noqa: E402
from mast3r.model import AsymmetricMASt3R  # noqa: E402

from v19_da3_depth_correction_ablation import load_raw_pair  # noqa: E402
from v22_explicit_metric_bridge_selection import load_cases, load_shards  # noqa: E402


DEFAULT_WEIGHTS = Path(
    "/data/wangzheng/Movie3R/Video-OnlineHMR/checkpoints/"
    "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
)
DEFAULT_V22 = (
    REPO_ROOT
    / "output"
    / "v22_explicit_metric_bridge"
    / "final_seed1"
    / "v22_explicit_metric_bridge.json"
)
DEFAULT_V21 = (
    REPO_ROOT
    / "output"
    / "v21_absolute_shot_background_scale"
    / "gated_full180"
    / "v21_absolute_shot_background_scale.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "output" / "v23_mast3r_rotation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--v22_report", type=Path, default=DEFAULT_V22)
    parser.add_argument("--v21_report", type=Path, default=DEFAULT_V21)
    parser.add_argument(
        "--v10_report",
        type=Path,
        default=REPO_ROOT
        / "output"
        / "v10_candidate_selection"
        / "oracle_gt_4source"
        / "oracle_candidate_selection_metrics.json",
    )
    parser.add_argument(
        "--stream_dir",
        type=Path,
        default=REPO_ROOT / "output" / "v18_human_metric_translation" / "stream_cache",
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--match_stride", type=int, default=8)
    parser.add_argument("--ransac_threshold_px", type=float, default=1.5)
    parser.add_argument("--mask_dilate", type=int, default=15)
    parser.add_argument("--min_matches", type=int, default=32)
    parser.add_argument("--min_fixed_rotation_deg", type=float, default=30.0)
    parser.add_argument("--max_cases", type=int, default=20)
    parser.add_argument("--sources", nargs="*", default=())
    parser.add_argument(
        "--trigger_mode", choices=("fixed_gt_pilot", "explicit_instability"), default="fixed_gt_pilot"
    )
    parser.add_argument("--max_scene_root_ratio", type=float, default=0.8)
    parser.add_argument("--min_torso_residual_deg", type=float, default=20.0)
    parser.add_argument(
        "--v16_dir",
        type=Path,
        default=REPO_ROOT / "output" / "v16_human_aware_rotation_residual" / "candidate_cache",
    )
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    return parser.parse_args()


def resize_crop(
    image: np.ndarray,
    intrinsics: np.ndarray,
    size: int,
) -> tuple[dict, np.ndarray, dict]:
    pil = PIL.Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
    width0, height0 = pil.size
    scale = float(size) / max(width0, height0)
    width = int(round(width0 * scale))
    height = int(round(height0 * scale))
    interpolation = PIL.Image.Resampling.LANCZOS if scale < 1.0 else PIL.Image.Resampling.BICUBIC
    pil = pil.resize((width, height), interpolation)
    center_x, center_y = width // 2, height // 2
    half_width = ((2 * center_x) // 16) * 8
    half_height = ((2 * center_y) // 16) * 8
    if width == height:
        half_height = 3 * half_width // 4
    left = int(center_x - half_width)
    top = int(center_y - half_height)
    right = int(center_x + half_width)
    bottom = int(center_y + half_height)
    pil = pil.crop((left, top, right, bottom))
    processed_k = np.asarray(intrinsics, dtype=np.float32).copy()
    processed_k[0] *= scale
    processed_k[1] *= scale
    processed_k[0, 2] -= left
    processed_k[1, 2] -= top
    view = {
        "img": ImgNorm(pil)[None],
        "true_shape": np.int32([[pil.size[1], pil.size[0]]]),
        "idx": 0,
        "instance": "0",
    }
    meta = {
        "scale": scale,
        "left": left,
        "top": top,
        "width": pil.size[0],
        "height": pil.size[1],
    }
    return view, processed_k, meta


def process_mask(mask: np.ndarray, meta: dict, dilation: int) -> np.ndarray:
    resized = cv2.resize(
        mask.astype(np.uint8),
        (int(round(mask.shape[1] * meta["scale"])), int(round(mask.shape[0] * meta["scale"]))),
        interpolation=cv2.INTER_NEAREST,
    )
    cropped = resized[
        meta["top"] : meta["top"] + meta["height"],
        meta["left"] : meta["left"] + meta["width"],
    ]
    kernel = np.ones((int(dilation), int(dilation)), dtype=np.uint8)
    return cv2.dilate(cropped, kernel, iterations=1) > 0


def rotation_error(rotation: np.ndarray, target: np.ndarray) -> float:
    return float(
        np.degrees(
            Rotation.from_matrix(rotation.astype(np.float64) @ target.astype(np.float64).T).magnitude()
        )
    )


def essential_rotation(
    old_points: np.ndarray,
    new_points: np.ndarray,
    old_k: np.ndarray,
    new_k: np.ndarray,
    threshold_px: float,
) -> tuple[np.ndarray | None, dict]:
    old_norm = cv2.undistortPoints(old_points[:, None].astype(np.float64), old_k, None)[:, 0]
    new_norm = cv2.undistortPoints(new_points[:, None].astype(np.float64), new_k, None)[:, 0]
    focal = float(np.mean([old_k[0, 0], old_k[1, 1], new_k[0, 0], new_k[1, 1]]))
    threshold = float(threshold_px) / max(focal, 1e-6)
    essential, mask = cv2.findEssentialMat(
        old_norm,
        new_norm,
        np.eye(3),
        method=cv2.RANSAC,
        prob=0.999,
        threshold=threshold,
    )
    if essential is None:
        return None, {"status": "essential_failed", "match_count": int(len(old_points))}
    candidates = [essential[index : index + 3] for index in range(0, essential.shape[0], 3)]
    best = None
    best_inliers = -1
    best_mask = None
    for candidate in candidates:
        inliers, rotation, _, pose_mask = cv2.recoverPose(
            candidate,
            old_norm,
            new_norm,
            np.eye(3),
            mask=mask,
        )
        if int(inliers) > best_inliers:
            best = rotation
            best_inliers = int(inliers)
            best_mask = pose_mask
    if best is None:
        return None, {"status": "recover_pose_failed", "match_count": int(len(old_points))}
    return best.T.astype(np.float32), {
        "status": "ok",
        "match_count": int(len(old_points)),
        "ransac_inliers": int(best_inliers),
        "inlier_ratio": float(best_inliers / max(len(old_points), 1)),
        "threshold_normalized": threshold,
        "pose_mask_count": int(np.count_nonzero(best_mask)) if best_mask is not None else 0,
    }


def pnp_rotation(
    points3d: np.ndarray,
    confidence: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    stride: int,
) -> tuple[np.ndarray | None, dict]:
    height, width = points3d.shape[:2]
    yy, xx = np.mgrid[stride // 2 : height : stride, stride // 2 : width : stride]
    xy = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1).astype(np.float32)
    xyz = points3d[yy, xx].reshape(-1, 3).astype(np.float32)
    score = confidence[yy, xx].reshape(-1).astype(np.float32)
    valid = (
        np.isfinite(xyz).all(axis=1)
        & (np.linalg.norm(xyz, axis=1) > 0.10)
        & (np.linalg.norm(xyz, axis=1) < 100.0)
        & ~mask[yy, xx].reshape(-1)
        & np.isfinite(score)
    )
    if int(valid.sum()) >= 32:
        cutoff = float(np.quantile(score[valid], 0.50))
        valid &= score >= cutoff
    xyz = xyz[valid]
    xy = xy[valid]
    if len(xyz) < 32:
        return None, {"status": "pnp_too_few_points", "point_count": int(len(xyz))}
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        xyz,
        xy,
        intrinsics,
        None,
        iterationsCount=2000,
        reprojectionError=3.0,
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success or inliers is None or len(inliers) < 12:
        return None, {
            "status": "pnp_failed",
            "point_count": int(len(xyz)),
            "inlier_count": int(len(inliers)) if inliers is not None else 0,
        }
    inlier_ids = inliers.reshape(-1)
    try:
        rvec, tvec = cv2.solvePnPRefineLM(
            xyz[inlier_ids], xy[inlier_ids], intrinsics, None, rvec, tvec
        )
    except cv2.error:
        pass
    rotation, _ = cv2.Rodrigues(rvec)
    return rotation.T.astype(np.float32), {
        "status": "ok",
        "point_count": int(len(xyz)),
        "inlier_count": int(len(inlier_ids)),
        "inlier_ratio": float(len(inlier_ids) / max(len(xyz), 1)),
        "translation_norm_native": float(np.linalg.norm(tvec)),
    }


def run_case(
    case_name: str,
    stream: dict,
    v10: dict,
    v22: dict,
    model: AsymmetricMASt3R,
    args: argparse.Namespace,
) -> dict:
    with np.load(stream["cache_path"]) as cache:
        images = [cache["old_images"][-1], cache["new_image"]]
        intrinsics = [cache["old_intrinsics"][-1], cache["new_intrinsics"]]
        old_pose = np.asarray(cache["old_pose"][-1], dtype=np.float32)
        target_pose = np.asarray(cache["target_pose"], dtype=np.float32)
    views = []
    processed_k = []
    metas = []
    for index in range(2):
        view, k, meta = resize_crop(images[index], intrinsics[index], int(args.image_size))
        view["idx"] = index
        view["instance"] = str(index)
        views.append(view)
        processed_k.append(k)
        metas.append(meta)

    started = time.perf_counter()
    with torch.inference_mode():
        output = inference([tuple(views)], model, args.device, batch_size=1, verbose=False)
    elapsed = time.perf_counter() - started
    descriptor_old = output["pred1"]["desc"].squeeze(0).detach()
    descriptor_new = output["pred2"]["desc"].squeeze(0).detach()
    confidence_old = output["pred1"]["desc_conf"].squeeze(0).detach().float().cpu().numpy()
    confidence_new = output["pred2"]["desc_conf"].squeeze(0).detach().float().cpu().numpy()
    points3d_new_in_old = (
        output["pred2"]["pts3d_in_other_view"].squeeze(0).detach().float().cpu().numpy()
    )
    points3d_confidence = output["pred2"]["conf"].squeeze(0).detach().float().cpu().numpy()
    old_points, new_points = fast_reciprocal_NNs(
        descriptor_old,
        descriptor_new,
        subsample_or_initxy1=int(args.match_stride),
        device=args.device,
        dist="dot",
        block_size=8192,
    )
    raw = load_raw_pair(Path(v10["paths"]["human3r_local_reset"]))
    masks = [process_mask(raw["mask"][i], metas[i], int(args.mask_dilate)) for i in range(2)]
    border = 4
    valid = (
        (old_points[:, 0] >= border)
        & (old_points[:, 0] < metas[0]["width"] - border)
        & (old_points[:, 1] >= border)
        & (old_points[:, 1] < metas[0]["height"] - border)
        & (new_points[:, 0] >= border)
        & (new_points[:, 0] < metas[1]["width"] - border)
        & (new_points[:, 1] >= border)
        & (new_points[:, 1] < metas[1]["height"] - border)
    )
    old_xy = old_points.astype(np.int64)
    new_xy = new_points.astype(np.int64)
    valid &= ~masks[0][old_xy[:, 1], old_xy[:, 0]]
    valid &= ~masks[1][new_xy[:, 1], new_xy[:, 0]]
    match_confidence = np.minimum(
        confidence_old[old_xy[:, 1], old_xy[:, 0]],
        confidence_new[new_xy[:, 1], new_xy[:, 0]],
    )
    if np.any(valid):
        cutoff = float(np.quantile(match_confidence[valid], 0.25))
        valid &= match_confidence >= cutoff
    old_points = old_points[valid].astype(np.float32)
    new_points = new_points[valid].astype(np.float32)
    if len(old_points) < int(args.min_matches):
        relative_rotation = None
        diagnostics = {"status": "too_few_matches", "match_count": int(len(old_points))}
    else:
        relative_rotation, diagnostics = essential_rotation(
            old_points,
            new_points,
            processed_k[0],
            processed_k[1],
            float(args.ransac_threshold_px),
        )
    target_rotation = target_pose[:3, :3]
    if relative_rotation is None:
        mast3r_error = float("nan")
        camera_rotation = None
    else:
        camera_rotation = old_pose[:3, :3] @ relative_rotation
        mast3r_error = rotation_error(camera_rotation, target_rotation)
    pnp_relative, pnp_diagnostics = pnp_rotation(
        points3d_new_in_old,
        points3d_confidence,
        masks[1],
        processed_k[1],
        int(args.match_stride),
    )
    if pnp_relative is None:
        pnp_error = float("nan")
        pnp_camera_rotation = None
    else:
        pnp_camera_rotation = old_pose[:3, :3] @ pnp_relative
        pnp_error = rotation_error(pnp_camera_rotation, target_rotation)
    return {
        "case_name": case_name,
        "source": v22["source"],
        "fixed_rotation_deg": float(
            v22["variants"]["fixed_explicit"]["camera"]["rotation_deg"]
        ),
        "torso_rotation_deg": float(
            v22["variants"]["torso_root_scale"]["camera"]["rotation_deg"]
        ),
        "v22_rotation_deg": float(
            v22["variants"]["safe_gravity_absolute_scene_scale"]["camera"]["rotation_deg"]
        ),
        "mast3r_rotation_deg": mast3r_error,
        "mast3r_camera_rotation": camera_rotation.astype(float).tolist()
        if camera_rotation is not None
        else None,
        "diagnostics": diagnostics,
        "mast3r_pnp_rotation_deg": pnp_error,
        "mast3r_pnp_camera_rotation": pnp_camera_rotation.astype(float).tolist()
        if pnp_camera_rotation is not None
        else None,
        "pnp_diagnostics": pnp_diagnostics,
        "inference_seconds": elapsed,
    }


def distribution(values: list[float]) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)) if len(values) else float("nan"),
        "median": float(np.median(values)) if len(values) else float("nan"),
        "p90": float(np.quantile(values, 0.90)) if len(values) else float("nan"),
        "p95": float(np.quantile(values, 0.95)) if len(values) else float("nan"),
    }


def summarize(rows: list[dict]) -> dict:
    return {
        "fixed_rotation_deg": distribution([row["fixed_rotation_deg"] for row in rows]),
        "torso_rotation_deg": distribution([row["torso_rotation_deg"] for row in rows]),
        "v22_rotation_deg": distribution([row["v22_rotation_deg"] for row in rows]),
        "mast3r_rotation_deg": distribution([row["mast3r_rotation_deg"] for row in rows]),
        "mast3r_pnp_rotation_deg": distribution(
            [row["mast3r_pnp_rotation_deg"] for row in rows]
        ),
        "success_rate": float(
            np.mean([row["diagnostics"].get("status") == "ok" for row in rows])
        ),
        "mast3r_better_than_v22_rate": float(
            np.mean(
                [
                    np.isfinite(row["mast3r_rotation_deg"])
                    and row["mast3r_rotation_deg"] < row["v22_rotation_deg"]
                    for row in rows
                ]
            )
        ),
        "mast3r_pnp_better_than_v22_rate": float(
            np.mean(
                [
                    np.isfinite(row["mast3r_pnp_rotation_deg"])
                    and row["mast3r_pnp_rotation_deg"] < row["v22_rotation_deg"]
                    for row in rows
                ]
            )
        ),
        "inference_seconds": distribution([row["inference_seconds"] for row in rows]),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not (0 <= int(args.shard_id) < int(args.num_shards)):
        raise ValueError("invalid shard")
    streams = load_shards(args.stream_dir, "v18_stream_shard_*_of_*.json")
    v10 = load_cases(args.v10_report)
    v22 = load_cases(args.v22_report)
    v21 = load_cases(args.v21_report)
    v16 = load_shards(args.v16_dir, "v16_candidates_shard_*_of_*.json")
    if args.trigger_mode == "explicit_instability":
        names = [
            name
            for name in sorted(v22)
            if float(v21[name]["calibration"]["new"]["scales"]["median_ratio"])
            / max(float(v21[name]["root_scales"]["new"]), 1e-6)
            < float(args.max_scene_root_ratio)
            and float(
                v16[name]["fixed_candidates"]["fixed_torso_motion_1f_resolve_t"][
                    "bounded_residual_deg"
                ]
            )
            >= float(args.min_torso_residual_deg)
            and (not args.sources or v22[name]["source"] in set(args.sources))
        ]
    else:
        names = [
            name
            for name in sorted(v22)
            if float(v22[name]["variants"]["fixed_explicit"]["camera"]["rotation_deg"])
            >= float(args.min_fixed_rotation_deg)
            and (not args.sources or v22[name]["source"] in set(args.sources))
        ]
    names = names[: int(args.max_cases)] if int(args.max_cases) > 0 else names
    names = [name for index, name in enumerate(names) if index % int(args.num_shards) == int(args.shard_id)]
    model = AsymmetricMASt3R.from_pretrained(str(args.weights)).to(args.device).eval()
    rows = []
    for index, name in enumerate(names):
        rows.append(run_case(name, streams[name], v10[name], v22[name], model, args))
        print(f"V23 MASt3R {index + 1}/{len(names)}", flush=True)
    report = {
        "experiment": "V23 frozen MASt3R background-correspondence rotation probe",
        "case_count": len(rows),
        "protocol": {
            "min_fixed_rotation_deg": float(args.min_fixed_rotation_deg),
            "trigger_mode": str(args.trigger_mode),
            "max_scene_root_ratio": float(args.max_scene_root_ratio),
            "min_torso_residual_deg": float(args.min_torso_residual_deg),
            "background_only": True,
            "match_stride": int(args.match_stride),
            "ransac_threshold_px": float(args.ransac_threshold_px),
            "post_cut_frames": 1,
            "learned_components": False,
        },
        "overall": summarize(rows),
        "by_source": {
            source: summarize([row for row in rows if row["source"] == source])
            for source in sorted({row["source"] for row in rows})
        },
        "cases": rows,
    }
    output = args.output_dir / f"v23_mast3r_rotation_shard_{int(args.shard_id):02d}_of_{int(args.num_shards):02d}.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(f">> wrote {output}", flush=True)


if __name__ == "__main__":
    main()
