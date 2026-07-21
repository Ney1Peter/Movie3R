#!/usr/bin/env python3
"""Generate V15 frozen wide-baseline Boundary Bridge candidates on CUDA."""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from v10_latent_activation_patching_probe import camera_matrix  # noqa: E402
from v10_oracle_state_vs_gauge_probe import rotation_error_deg  # noqa: E402
from v11_gauge_neutral_first_write_probe import (  # noqa: E402
    DEFAULT_RECORDS,
    build_dataset,
    build_model,
    build_smpl_models,
    configure_views,
    predicted_human,
    read_jsonl,
    record_spec,
    texture_score,
)
from v12_build_gauge_neutral_teacher_cache import old_a_dataset  # noqa: E402
from v13_scene_coordinate_oracle import (  # noqa: E402
    camera_points,
    confidence,
    direct_transform_error,
    human_mask,
    robust_fit,
    transform_points,
    valid_points,
)
from v14_depth_free_world_memory_candidates import (  # noqa: E402
    covariance_diagnostics,
    fixed_explicit_human3r_gauge,
    human_diagnostics,
)
from v9_learned_stream_alignment_overfit import gt_pose_from_view  # noqa: E402


DEFAULT_OUTPUT = REPO_ROOT / "output" / "v15_wide_baseline_boundary_bridge" / "candidate_cache"
DEFAULT_V14 = REPO_ROOT / "output" / "v14_selective_world_memory" / "candidate_cache"
DEFAULT_VGGT_ROOT = Path("/data/wangzheng/Movie3R/vggt")
DEFAULT_VGGT_WEIGHTS = DEFAULT_VGGT_ROOT / "vggt_weights" / "model.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--v14_candidate_dir", type=Path, default=DEFAULT_V14)
    parser.add_argument("--candidate_root", type=Path, default=REPO_ROOT / "output/v10_candidate_selection/oracle_gt_4source/cases")
    parser.add_argument("--data_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data"))
    parser.add_argument("--model_path", type=Path, default=REPO_ROOT / "src/human3r_896L.pth")
    parser.add_argument("--vggt_root", type=Path, default=DEFAULT_VGGT_ROOT)
    parser.add_argument("--vggt_weights", type=Path, default=DEFAULT_VGGT_WEIGHTS)
    parser.add_argument("--enable_da3_correspondence", action="store_true")
    parser.add_argument(
        "--da3_model_path",
        type=Path,
        default=REPO_ROOT.parent
        / "Movie3R-dataset"
        / "Depth-Anything-3"
        / "checkpoints"
        / "DA3Metric-Large",
    )
    parser.add_argument("--da3_process_res", type=int, default=504)
    parser.add_argument("--device", required=True)
    parser.add_argument("--resolution", type=int, nargs=2, default=(512, 288))
    parser.add_argument("--resize_mode", default="human3r_demo")
    parser.add_argument("--max_post_frames", type=int, default=3)
    parser.add_argument("--warmup_frames", type=int, default=8)
    parser.add_argument("--query_rows", type=int, default=20)
    parser.add_argument("--query_cols", type=int, default=16)
    parser.add_argument("--pair_batch_size", type=int, default=3)
    parser.add_argument("--point_samples_per_frame", type=int, default=768)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--sources", nargs="*", default=())
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def one_batch(dataset) -> list[dict]:
    return next(iter(torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)))


def load_v14_cases(root: Path) -> dict[str, dict]:
    rows = {}
    for path in sorted(glob.glob(str(root / "v14_candidates_shard_*_of_*.json"))):
        for case in json.loads(Path(path).read_text(encoding="utf-8"))["cases"]:
            rows[str(case["case_name"])] = case
    return rows


def build_vggt(args: argparse.Namespace):
    sys.path.insert(0, str(args.vggt_root))
    from vggt.models.vggt import VGGT

    model = VGGT(enable_depth=False, enable_point=False, enable_track=True)
    state = torch.load(args.vggt_weights, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=False)
    unexpected = [key for key in incompatible.unexpected_keys if not key.startswith(("depth_head.", "point_head."))]
    if incompatible.missing_keys or unexpected:
        raise RuntimeError(f"Unexpected VGGT checkpoint mismatch: {incompatible}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.to(args.device).eval()


def build_da3(args: argparse.Namespace):
    if not args.enable_da3_correspondence:
        return None
    from v18_da3_metric_depth_probe import DepthAnything3

    return DepthAnything3.from_pretrained(str(args.da3_model_path)).to(args.device).eval()


def square_view(view: dict, background_only: bool, query_rows: int, query_cols: int) -> dict:
    image = ((view["img"][0].detach().float().cpu() + 1.0) * 0.5).clamp(0.0, 1.0)
    height, width = int(image.shape[-2]), int(image.shape[-1])
    mask = view.get("msk", False)
    if isinstance(mask, torch.Tensor):
        mask = mask[0].detach().float().cpu()
    else:
        mask = torch.zeros((height, width), dtype=torch.float32)
    if mask.shape != (height, width):
        mask = F.interpolate(mask[None, None], size=(height, width), mode="nearest")[0, 0]
    if background_only:
        image = torch.where(mask[None] > 0.5, torch.ones_like(image), image)
    scale = 518.0 / max(height, width)
    resized_height = max(14, int(round(height * scale / 14.0)) * 14)
    resized_width = max(14, int(round(width * scale / 14.0)) * 14)
    resized_height, resized_width = min(resized_height, 518), min(resized_width, 518)
    image = F.interpolate(image[None], size=(resized_height, resized_width), mode="bilinear", align_corners=False)[0]
    pad_top = (518 - resized_height) // 2
    pad_left = (518 - resized_width) // 2
    image = F.pad(
        image,
        (pad_left, 518 - resized_width - pad_left, pad_top, 518 - resized_height - pad_top),
        value=1.0,
    )
    margin_y = min(12.0, max(height * 0.05, 2.0))
    margin_x = min(12.0, max(width * 0.05, 2.0))
    yy = np.linspace(margin_y, height - 1.0 - margin_y, query_rows)
    xx = np.linspace(margin_x, width - 1.0 - margin_x, query_cols)
    grid_x, grid_y = np.meshgrid(xx, yy)
    original_query = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1).astype(np.float32)
    square_query = np.stack(
        [
            original_query[:, 0] * resized_width / width + pad_left,
            original_query[:, 1] * resized_height / height + pad_top,
        ],
        axis=1,
    ).astype(np.float32)
    return {
        "image": image,
        "height": height,
        "width": width,
        "resized_height": resized_height,
        "resized_width": resized_width,
        "pad_top": pad_top,
        "pad_left": pad_left,
        "original_query": original_query,
        "square_query": square_query,
        "human_ratio": float((mask > 0.5).float().mean()),
    }


def square_to_original(points: np.ndarray, meta: dict) -> tuple[np.ndarray, np.ndarray]:
    output = np.stack(
        [
            (points[:, 0] - meta["pad_left"]) * meta["width"] / meta["resized_width"],
            (points[:, 1] - meta["pad_top"]) * meta["height"] / meta["resized_height"],
        ],
        axis=1,
    ).astype(np.float32)
    valid = (
        (output[:, 0] >= 0.0)
        & (output[:, 0] <= meta["width"] - 1.0)
        & (output[:, 1] >= 0.0)
        & (output[:, 1] <= meta["height"] - 1.0)
    )
    return output, valid


def pose_encoding_to_c2w(pose_encoding: torch.Tensor, image_shape: tuple[int, int]) -> np.ndarray:
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    extrinsic, _ = pose_encoding_to_extri_intri(pose_encoding.float().unsqueeze(0), image_shape)
    output = []
    for row in extrinsic[0].detach().cpu().numpy():
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3] = row
        output.append(np.linalg.inv(matrix).astype(np.float32))
    return np.stack(output)


def pair_specs(old_count: int, new_count: int) -> list[dict]:
    specs = []
    for old_index in range(old_count):
        for new_index in range(new_count):
            specs.append({"old": old_index, "new": new_index, "reverse": False})
            specs.append({"old": old_index, "new": new_index, "reverse": True})
    return specs


def run_vggt_pairs(
    model,
    old_meta: list[dict],
    new_meta: list[dict],
    specs: list[dict],
    args: argparse.Namespace,
) -> tuple[list[dict], float]:
    outputs = []
    started = time.perf_counter()
    device = torch.device(args.device)
    for start in range(0, len(specs), int(args.pair_batch_size)):
        current = specs[start : start + int(args.pair_batch_size)]
        images, queries = [], []
        for spec in current:
            first = new_meta[spec["new"]] if spec["reverse"] else old_meta[spec["old"]]
            second = old_meta[spec["old"]] if spec["reverse"] else new_meta[spec["new"]]
            images.append(torch.stack([first["image"], second["image"]]))
            queries.append(torch.from_numpy(first["square_query"]))
        image_batch = torch.stack(images).to(device, non_blocking=True)
        query_batch = torch.stack(queries).to(device, non_blocking=True)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            prediction = model(image_batch, query_batch)
        for local_index, spec in enumerate(current):
            c2w = pose_encoding_to_c2w(prediction["pose_enc"][local_index], (518, 518))
            raw_relative = np.linalg.inv(c2w[0]) @ c2w[1]
            old_from_new = np.linalg.inv(raw_relative) if spec["reverse"] else raw_relative
            tracked_square = prediction["track"][local_index, 1].detach().float().cpu().numpy()
            first = new_meta[spec["new"]] if spec["reverse"] else old_meta[spec["old"]]
            second = old_meta[spec["old"]] if spec["reverse"] else new_meta[spec["new"]]
            tracked_original, inside = square_to_original(tracked_square, second)
            query_original = first["original_query"]
            if spec["reverse"]:
                old_pixels, new_pixels = tracked_original, query_original
            else:
                old_pixels, new_pixels = query_original, tracked_original
            outputs.append(
                {
                    **spec,
                    "old_from_new": old_from_new.astype(np.float32),
                    "old_pixels": old_pixels,
                    "new_pixels": new_pixels,
                    "inside": inside,
                    "visibility": prediction["vis"][local_index, 1].detach().float().cpu().numpy(),
                    "track_confidence": prediction["conf"][local_index, 1].detach().float().cpu().numpy(),
                }
            )
        del image_batch, query_batch, prediction
    return outputs, time.perf_counter() - started


def project_rotation(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(matrix.astype(np.float64))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation.astype(np.float32)


def robust_rotation_consensus(rotations: list[np.ndarray]) -> tuple[np.ndarray, dict]:
    if not rotations:
        return np.eye(3, dtype=np.float32), {"count": 0, "inlier_count": 0, "spread_deg": float("nan")}
    pairwise = np.asarray(
        [[rotation_error_deg(first, second) for second in rotations] for first in rotations], dtype=np.float64
    )
    medoid = int(np.argmin(pairwise.sum(axis=1)))
    distance = pairwise[medoid]
    median = float(np.median(distance))
    mad = float(np.median(np.abs(distance - median)))
    threshold = min(45.0, max(8.0, median + 2.5 * 1.4826 * mad))
    active = distance <= threshold
    if int(active.sum()) < max(1, min(3, len(rotations))):
        active = np.argsort(distance)[: min(3, len(rotations))]
        selected = [rotations[int(index)] for index in active]
    else:
        selected = [rotation for rotation, keep in zip(rotations, active) if keep]
    consensus = project_rotation(np.sum(np.stack(selected), axis=0))
    errors = np.asarray([rotation_error_deg(consensus, rotation) for rotation in rotations], dtype=np.float64)
    return consensus, {
        "count": len(rotations),
        "inlier_count": len(selected),
        "spread_deg": float(np.median(errors)),
        "p90_spread_deg": float(np.percentile(errors, 90)),
        "max_spread_deg": float(errors.max()),
    }


def robust_vector_center(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = np.maximum(np.asarray(weights, dtype=np.float64), 1e-8)
    values = np.asarray(values, dtype=np.float64)
    center = np.average(values, axis=0, weights=weights)
    for _ in range(12):
        distance = np.linalg.norm(values - center, axis=1)
        current = weights / np.maximum(distance, 1e-4)
        updated = np.average(values, axis=0, weights=current)
        if np.linalg.norm(updated - center) < 1e-5:
            center = updated
            break
        center = updated
    return center.astype(np.float32)


def aggregate_coarse(pairs: list[dict], old_poses: list[np.ndarray], new_poses: list[np.ndarray]) -> tuple[np.ndarray, dict]:
    transforms = [old_poses[row["old"]] @ row["old_from_new"] @ np.linalg.inv(new_poses[row["new"]]) for row in pairs]
    rotation, diagnostics = robust_rotation_consensus([row[:3, :3] for row in transforms])
    translation = robust_vector_center(np.stack([row[:3, 3] for row in transforms]), np.ones(len(transforms)))
    output = np.eye(4, dtype=np.float32)
    output[:3, :3] = rotation
    output[:3, 3] = translation
    return output, diagnostics


def normalized_pixels(pixels: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate(
        [pixels.astype(np.float64), np.ones((len(pixels), 1), dtype=np.float64)],
        axis=1,
    )
    normalized = homogeneous @ np.linalg.inv(intrinsics.astype(np.float64)).T
    return normalized[:, :2] / np.maximum(normalized[:, 2:3], 1e-9)


def essential_rotation_candidate(
    pairs: list[dict],
    old_views: list[dict],
    new_views: list[dict],
    old_poses: list[np.ndarray],
    new_poses: list[np.ndarray],
) -> tuple[np.ndarray | None, dict]:
    rotations = []
    pair_rows = []
    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    for pair in pairs:
        valid = np.asarray(pair["inside"], dtype=bool).copy()
        confidence = np.asarray(pair["track_confidence"], dtype=np.float32)
        visibility = np.asarray(pair["visibility"], dtype=np.float32)
        finite = np.isfinite(confidence) & np.isfinite(visibility)
        valid &= finite & (visibility >= 0.25)
        if finite.any():
            threshold = max(0.002, float(np.percentile(confidence[finite], 35)))
            valid &= confidence >= threshold
        if int(valid.sum()) < 12:
            continue

        old_view = old_views[pair["old"]]
        new_view = new_views[pair["new"]]
        old_intrinsics = (
            old_view["camera_intrinsics"][0].detach().float().cpu().numpy()
        )
        new_intrinsics = (
            new_view["camera_intrinsics"][0].detach().float().cpu().numpy()
        )
        old_pixels = np.asarray(pair["old_pixels"], dtype=np.float32)[valid]
        new_pixels = np.asarray(pair["new_pixels"], dtype=np.float32)[valid]
        old_normalized = normalized_pixels(old_pixels, old_intrinsics)
        new_normalized = normalized_pixels(new_pixels, new_intrinsics)
        focal = float(
            np.mean(
                [
                    old_intrinsics[0, 0],
                    old_intrinsics[1, 1],
                    new_intrinsics[0, 0],
                    new_intrinsics[1, 1],
                ]
            )
        )
        essential, mask = cv2.findEssentialMat(
            old_normalized,
            new_normalized,
            np.eye(3, dtype=np.float64),
            method=method,
            prob=0.999,
            threshold=1.5 / max(focal, 1.0),
        )
        if essential is None or mask is None:
            continue
        essential = np.asarray(essential, dtype=np.float64).reshape(-1, 3, 3)
        best = None
        for matrix in essential:
            recovered, rotation, _, pose_mask = cv2.recoverPose(
                matrix,
                old_normalized,
                new_normalized,
                np.eye(3, dtype=np.float64),
                mask=mask.copy(),
            )
            if best is None or int(recovered) > best[0]:
                best = (int(recovered), rotation, pose_mask)
        if best is None or best[0] < 8:
            continue
        old_from_new = np.eye(4, dtype=np.float32)
        old_from_new[:3, :3] = best[1].T.astype(np.float32)
        boundary = (
            old_poses[pair["old"]]
            @ old_from_new
            @ np.linalg.inv(new_poses[pair["new"]])
        )
        rotations.append(boundary[:3, :3].astype(np.float32))
        pair_rows.append(
            {
                "old": int(pair["old"]),
                "new": int(pair["new"]),
                "reverse": bool(pair["reverse"]),
                "point_count": int(valid.sum()),
                "essential_inlier_count": int(mask.sum()),
                "cheirality_inlier_count": int(best[0]),
                "essential_inlier_ratio": float(mask.mean()),
                "cheirality_inlier_ratio": float(best[0] / max(int(valid.sum()), 1)),
            }
        )
    if not rotations:
        return None, {
            "essential_fit_failed": True,
            "essential_pair_count": 0,
            "essential_pairs": [],
        }
    rotation, consensus = robust_rotation_consensus(rotations)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation
    return transform, {
        "essential_fit_failed": False,
        "essential_pair_count": len(rotations),
        "essential_inlier_ratio": float(
            np.mean([row["essential_inlier_ratio"] for row in pair_rows])
        ),
        "essential_cheirality_ratio": float(
            np.mean([row["cheirality_inlier_ratio"] for row in pair_rows])
        ),
        "essential_rotation_spread_deg": float(consensus["spread_deg"]),
        "essential_pairs": pair_rows,
    }


def view_rgb(view: dict) -> np.ndarray:
    image = ((view["img"][0].detach().float().cpu() + 1.0) * 127.5).clamp(
        0.0, 255.0
    )
    return image.permute(1, 2, 0).byte().numpy()


def processed_pixels(
    pixels: np.ndarray, intrinsics: np.ndarray, processed_intrinsics: np.ndarray
) -> np.ndarray:
    return np.stack(
        [
            (pixels[:, 0] - intrinsics[0, 2])
            * (processed_intrinsics[0, 0] / intrinsics[0, 0])
            + processed_intrinsics[0, 2],
            (pixels[:, 1] - intrinsics[1, 2])
            * (processed_intrinsics[1, 1] / intrinsics[1, 1])
            + processed_intrinsics[1, 2],
        ],
        axis=1,
    ).astype(np.float32)


def sample_da3_points(
    depth: np.ndarray,
    original_intrinsics: np.ndarray,
    processed_intrinsics: np.ndarray,
    pixels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mapped = processed_pixels(pixels, original_intrinsics, processed_intrinsics)
    xx = np.rint(mapped[:, 0]).astype(np.int64)
    yy = np.rint(mapped[:, 1]).astype(np.int64)
    inside = (
        (xx >= 0)
        & (xx < depth.shape[1])
        & (yy >= 0)
        & (yy < depth.shape[0])
    )
    clipped_x = np.clip(xx, 0, depth.shape[1] - 1)
    clipped_y = np.clip(yy, 0, depth.shape[0] - 1)
    z = depth[clipped_y, clipped_x].astype(np.float32)
    inside &= np.isfinite(z) & (z > 0.05) & (z < 100.0)
    points = np.stack(
        [
            (mapped[:, 0] - processed_intrinsics[0, 2])
            * z
            / processed_intrinsics[0, 0],
            (mapped[:, 1] - processed_intrinsics[1, 2])
            * z
            / processed_intrinsics[1, 1],
            z,
        ],
        axis=1,
    ).astype(np.float32)
    return points, inside


def da3_correspondence_candidate(
    pairs: list[dict],
    old_views: list[dict],
    new_views: list[dict],
    old_poses: list[np.ndarray],
    new_poses: list[np.ndarray],
    old_depths: np.ndarray | None,
    new_depths: np.ndarray | None,
    old_processed_intrinsics: np.ndarray | None,
    new_processed_intrinsics: np.ndarray | None,
    fixed_rotation: np.ndarray | None = None,
) -> tuple[np.ndarray | None, dict]:
    if old_depths is None or new_depths is None:
        return None, {"da3_fit_failed": True, "da3_correspondence_count": 0}
    source_rows, target_rows, weight_rows = [], [], []
    for pair in pairs:
        old_intrinsics = (
            old_views[pair["old"]]["camera_intrinsics"][0]
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        new_intrinsics = (
            new_views[pair["new"]]["camera_intrinsics"][0]
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        old_points, old_inside = sample_da3_points(
            old_depths[pair["old"]],
            old_intrinsics,
            old_processed_intrinsics[pair["old"]],
            np.asarray(pair["old_pixels"], dtype=np.float32),
        )
        new_points, new_inside = sample_da3_points(
            new_depths[pair["new"]],
            new_intrinsics,
            new_processed_intrinsics[pair["new"]],
            np.asarray(pair["new_pixels"], dtype=np.float32),
        )
        confidence = np.asarray(pair["track_confidence"], dtype=np.float32)
        visibility = np.asarray(pair["visibility"], dtype=np.float32)
        valid = (
            np.asarray(pair["inside"], dtype=bool)
            & old_inside
            & new_inside
            & np.isfinite(confidence)
            & np.isfinite(visibility)
            & (visibility >= 0.25)
        )
        finite_confidence = confidence[np.isfinite(confidence)]
        if len(finite_confidence):
            threshold = max(
                0.002, float(np.percentile(finite_confidence, 35))
            )
            valid &= confidence >= threshold
        if int(valid.sum()) < 6:
            continue
        source_rows.append(
            transform_points(new_poses[pair["new"]], new_points[valid])
        )
        target_rows.append(
            transform_points(old_poses[pair["old"]], old_points[valid])
        )
        weight_rows.append(
            np.maximum(confidence[valid], 1e-4)
            * np.maximum(visibility[valid], 1e-3)
        )
    if not source_rows:
        return None, {"da3_fit_failed": True, "da3_correspondence_count": 0}
    source = np.concatenate(source_rows)
    target = np.concatenate(target_rows)
    weight = np.concatenate(weight_rows)
    fit = (
        robust_fit(source, target, weight, False)
        if fixed_rotation is None
        else fixed_rotation_fit(source, target, weight, fixed_rotation)
    )
    if fit is None:
        return None, {
            "da3_fit_failed": True,
            "da3_correspondence_count": int(len(source)),
        }
    return fit_to_transform(fit), {
        "da3_fit_failed": False,
        "da3_rotation_mode": "free" if fixed_rotation is None else "fixed_wide",
        "da3_correspondence_count": int(len(source)),
        "da3_fit_residual_mean_m": float(np.mean(fit["residual"])),
        "da3_fit_residual_median_m": float(np.median(fit["residual"])),
        "da3_fit_residual_p90_m": float(np.percentile(fit["residual"], 90)),
        "da3_robust_inlier_ratio": float(np.mean(fit["active"])),
    }


def sample_prediction(prediction: dict, view: dict, pixels: np.ndarray) -> dict:
    points = camera_points(prediction)
    height, width = tuple(int(value) for value in prediction["pts3d_in_self_view"].shape[-3:-1])
    xx = np.rint(pixels[:, 0]).astype(np.int64)
    yy = np.rint(pixels[:, 1]).astype(np.int64)
    inside = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
    ids = np.clip(yy, 0, height - 1) * width + np.clip(xx, 0, width - 1)
    return {
        "points": points[ids],
        "confidence": confidence(prediction, len(points))[ids],
        "human": human_mask(view, len(points))[ids],
        "inside": inside,
        "height": height,
        "width": width,
    }


def epipolar_distance(old_pixels: np.ndarray, new_pixels: np.ndarray, old_view: dict, new_view: dict) -> np.ndarray:
    old_pose = gt_pose_from_view(old_view).detach().float().cpu().numpy()
    new_pose = gt_pose_from_view(new_view).detach().float().cpu().numpy()
    new_from_old = np.linalg.inv(new_pose) @ old_pose
    rotation, translation = new_from_old[:3, :3], new_from_old[:3, 3]
    skew = np.asarray(
        [[0.0, -translation[2], translation[1]], [translation[2], 0.0, -translation[0]], [-translation[1], translation[0], 0.0]],
        dtype=np.float64,
    )
    old_k = old_view["camera_intrinsics"][0].detach().float().cpu().numpy()
    new_k = new_view["camera_intrinsics"][0].detach().float().cpu().numpy()
    fundamental = np.linalg.inv(new_k).T @ skew @ rotation @ np.linalg.inv(old_k)
    old_h = np.concatenate([old_pixels, np.ones((len(old_pixels), 1), dtype=np.float32)], axis=1)
    new_h = np.concatenate([new_pixels, np.ones((len(new_pixels), 1), dtype=np.float32)], axis=1)
    line_new = old_h @ fundamental.T
    line_old = new_h @ fundamental
    numerator = np.abs(np.sum(new_h * line_new, axis=1))
    first = numerator / np.maximum(np.linalg.norm(line_new[:, :2], axis=1), 1e-8)
    second = numerator / np.maximum(np.linalg.norm(line_old[:, :2], axis=1), 1e-8)
    return 0.5 * (first + second)


def mutual_track_ratio(rows: list[dict], radius: float = 8.0) -> float:
    groups = {}
    for row in rows:
        groups[(row["old"], row["new"], row["reverse"])] = row
    ratios = []
    for row in rows:
        if row["reverse"]:
            continue
        reverse = groups.get((row["old"], row["new"], True))
        if reverse is None:
            continue
        forward_new = row["new_pixels"]
        reverse_new = reverse["new_pixels"]
        distance = np.linalg.norm(forward_new[:, None] - reverse_new[None], axis=2)
        nearest = np.argmin(distance, axis=1)
        cycle = np.linalg.norm(row["old_pixels"] - reverse["old_pixels"][nearest], axis=1)
        valid = row["inside"] & reverse["inside"][nearest]
        if valid.any():
            ratios.append(float(np.mean((distance[np.arange(len(distance)), nearest] < radius) & (cycle < radius) & valid)))
    return float(np.mean(ratios)) if ratios else float("nan")


def collect_correspondences(
    pairs: list[dict],
    old_predictions: list[dict],
    new_predictions: list[dict],
    old_views: list[dict],
    new_views: list[dict],
    old_poses: list[np.ndarray],
    new_poses: list[np.ndarray],
    policy: str,
) -> dict:
    source_rows, target_rows, weight_rows = [], [], []
    old_pixel_rows, new_pixel_rows, epi_rows, human_rows = [], [], [], []
    pair_ids = []
    for pair_index, row in enumerate(pairs):
        old = sample_prediction(old_predictions[row["old"]], old_views[row["old"]], row["old_pixels"])
        new = sample_prediction(new_predictions[row["new"]], new_views[row["new"]], row["new_pixels"])
        valid = row["inside"] & old["inside"] & new["inside"]
        valid &= np.isfinite(old["points"]).all(axis=1) & np.isfinite(new["points"]).all(axis=1)
        valid &= (old["points"][:, 2] > 0.05) & (new["points"][:, 2] > 0.05)
        track_confidence = np.maximum(np.asarray(row["track_confidence"], dtype=np.float32), 0.0)
        visibility = np.clip(np.asarray(row["visibility"], dtype=np.float32), 0.0, 1.0)
        finite_conf = track_confidence[np.isfinite(track_confidence)]
        if len(finite_conf):
            threshold = max(0.002, float(np.percentile(finite_conf, 35)))
            valid &= track_confidence >= threshold
        human = old["human"] | new["human"]
        if policy == "background":
            valid &= ~human
        if int(valid.sum()) < 6:
            continue
        old_world = transform_points(old_poses[row["old"]], old["points"][valid])
        new_world = transform_points(new_poses[row["new"]], new["points"][valid])
        point_confidence = np.sqrt(np.maximum(old["confidence"][valid] * new["confidence"][valid], 1e-6))
        point_confidence /= max(float(np.median(point_confidence)), 1e-6)
        weight = point_confidence * np.maximum(track_confidence[valid], 1e-4) * np.maximum(visibility[valid], 1e-3)
        if policy == "downweighted":
            weight *= np.where(human[valid], 0.10, 1.0)
        source_rows.append(new_world)
        target_rows.append(old_world)
        weight_rows.append(weight)
        old_pixel_rows.append(row["old_pixels"][valid])
        new_pixel_rows.append(row["new_pixels"][valid])
        epi_rows.append(epipolar_distance(row["old_pixels"][valid], row["new_pixels"][valid], old_views[row["old"]], new_views[row["new"]]))
        human_rows.append(human[valid])
        pair_ids.append(np.full(int(valid.sum()), pair_index, dtype=np.int32))
    if not source_rows:
        return {"source": np.empty((0, 3)), "target": np.empty((0, 3)), "weight": np.empty(0)}
    return {
        "source": np.concatenate(source_rows),
        "target": np.concatenate(target_rows),
        "weight": np.concatenate(weight_rows),
        "old_pixels": np.concatenate(old_pixel_rows),
        "new_pixels": np.concatenate(new_pixel_rows),
        "epipolar": np.concatenate(epi_rows),
        "human": np.concatenate(human_rows),
        "pair_ids": np.concatenate(pair_ids),
    }


def fixed_rotation_fit(source: np.ndarray, target: np.ndarray, weight: np.ndarray, rotation: np.ndarray) -> dict | None:
    if len(source) < 6:
        return None
    translation_samples = target - source @ rotation.T
    active = np.ones(len(source), dtype=bool)
    translation = robust_vector_center(translation_samples, weight)
    for _ in range(5):
        residual = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
        median = float(np.median(residual[active]))
        mad = float(np.median(np.abs(residual[active] - median)))
        threshold = min(1.5, max(0.08, median + 2.5 * 1.4826 * mad))
        next_active = residual <= threshold
        if int(next_active.sum()) < 6 or np.array_equal(active, next_active):
            break
        active = next_active
        translation = robust_vector_center(translation_samples[active], weight[active])
    residual = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
    return {"rotation": rotation, "scale": 1.0, "translation": translation, "residual": residual, "active": active}


def fit_to_transform(fit: dict) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = fit["rotation"]
    transform[:3, 3] = fit["translation"]
    return transform


def metric_fit_diagnostics(fit: dict | None, correspondences: dict, pair_rows: list[dict]) -> dict:
    if fit is None:
        return {"fit_failed": True, "correspondence_count": int(len(correspondences.get("source", [])))}
    residual = fit["residual"]
    source = correspondences["source"]
    old_pixels = correspondences["old_pixels"]
    pair_ids = correspondences["pair_ids"]
    bins = 8
    image_keys = pair_ids * bins * bins + np.minimum((old_pixels[:, 1] / 512.0 * bins).astype(int), bins - 1) * bins
    image_keys += np.minimum((old_pixels[:, 0] / 512.0 * bins).astype(int), bins - 1)
    return {
        "fit_failed": False,
        "correspondence_count": int(len(source)),
        "pair_count": int(len(np.unique(pair_ids))),
        "fit_residual_mean_m": float(np.mean(residual)),
        "fit_residual_median_m": float(np.median(residual)),
        "fit_residual_p90_m": float(np.percentile(residual, 90)),
        "inlier_ratio_0_10m": float(np.mean(residual < 0.10)),
        "inlier_ratio_0_20m": float(np.mean(residual < 0.20)),
        "robust_inlier_ratio": float(np.mean(fit["active"])),
        "image_coverage_8x8": float(len(np.unique(image_keys)) / max(len(np.unique(pair_ids)) * bins * bins, 1)),
        "human_correspondence_ratio": float(np.mean(correspondences["human"])),
        "epipolar_median_px": float(np.median(correspondences["epipolar"])),
        "epipolar_p90_px": float(np.percentile(correspondences["epipolar"], 90)),
        "mutual_track_ratio": mutual_track_ratio(pair_rows),
        "source_geometry": covariance_diagnostics(source),
        "target_geometry": covariance_diagnostics(correspondences["target"]),
    }


def sampled_world_cloud(prediction: dict, view: dict, pose: np.ndarray, count: int, static_only: bool) -> tuple[np.ndarray, np.ndarray]:
    points = camera_points(prediction)
    height, width = tuple(int(value) for value in prediction["pts3d_in_self_view"].shape[-3:-1])
    conf = confidence(prediction, len(points))
    mask = human_mask(view, len(points))
    valid = valid_points(points, conf, mask, static_only)
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    cells_y, cells_x = 24, 32
    cell = np.minimum(yy.reshape(-1) * cells_y // height, cells_y - 1) * cells_x
    cell += np.minimum(xx.reshape(-1) * cells_x // width, cells_x - 1)
    selected = []
    for cell_id in range(cells_y * cells_x):
        ids = np.flatnonzero(valid & (cell == cell_id))
        if len(ids):
            selected.append(int(ids[np.argmax(conf[ids])]))
    selected = np.asarray(selected, dtype=np.int64)
    if len(selected) > count:
        selected = selected[np.argsort(conf[selected])[-count:]]
    return transform_points(pose, points[selected]), conf[selected]


def dense_clouds(
    old_predictions: list[dict],
    new_predictions: list[dict],
    old_views: list[dict],
    new_views: list[dict],
    old_poses: list[np.ndarray],
    new_poses: list[np.ndarray],
    count: int,
    static_only: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    old_rows, old_conf, new_rows, new_conf = [], [], [], []
    for prediction, view, pose in zip(old_predictions, old_views, old_poses):
        points, conf = sampled_world_cloud(prediction, view, pose, count, static_only)
        old_rows.append(points)
        old_conf.append(conf)
    for prediction, view, pose in zip(new_predictions, new_views, new_poses):
        points, conf = sampled_world_cloud(prediction, view, pose, count, static_only)
        new_rows.append(points)
        new_conf.append(conf)
    return np.concatenate(new_rows), np.concatenate(old_rows), np.concatenate(new_conf), np.concatenate(old_conf)


def residual_icp(
    initial: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    source_conf: np.ndarray,
    target_conf: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, dict]:
    current = initial.astype(np.float32).copy()
    source_gpu = torch.as_tensor(source, dtype=torch.float32, device=device)
    target_gpu = torch.as_tensor(target, dtype=torch.float32, device=device)
    iterations = []
    for threshold in (1.25, 0.85, 0.60, 0.42, 0.30, 0.22):
        transformed = torch.as_tensor(transform_points(current, source), dtype=torch.float32, device=device)
        distance, nearest = torch.min(torch.cdist(transformed, target_gpu), dim=1)
        keep = distance < threshold
        if int(keep.sum()) < 16:
            break
        keep_np = keep.detach().cpu().numpy()
        nearest_np = nearest.detach().cpu().numpy()[keep_np]
        delta = robust_fit(
            transformed.detach().cpu().numpy()[keep_np],
            target[nearest_np],
            np.sqrt(np.maximum(source_conf[keep_np] * target_conf[nearest_np], 1e-6)),
            False,
        )
        if delta is None:
            break
        delta_t = float(np.linalg.norm(delta["translation"]))
        delta_r = rotation_error_deg(delta["rotation"], np.eye(3, dtype=np.float32))
        if delta_t > 0.75 or delta_r > 12.0:
            break
        update = fit_to_transform(delta)
        current = update @ current
        iterations.append({"threshold_m": threshold, "pair_count": int(keep.sum()), "delta_translation_m": delta_t, "delta_rotation_deg": delta_r})
    transformed = transform_points(current, source)
    distance, _ = torch.min(torch.cdist(torch.as_tensor(transformed, device=device), target_gpu), dim=1)
    residual = distance.detach().cpu().numpy()
    refinement = current @ np.linalg.inv(initial)
    return current, {
        "icp_iterations": iterations,
        "fit_residual_mean_m": float(np.mean(residual)),
        "fit_residual_median_m": float(np.median(residual)),
        "fit_residual_p90_m": float(np.percentile(residual, 90)),
        "inlier_ratio_0_20m": float(np.mean(residual < 0.20)),
        "refinement_translation_m": float(np.linalg.norm(refinement[:3, 3])),
        "refinement_rotation_deg": rotation_error_deg(refinement[:3, :3], np.eye(3, dtype=np.float32)),
    }


def relative_translation_diagnostics(
    transform: np.ndarray,
    pred_pose: np.ndarray,
    old_pose: np.ndarray,
    target_pose: np.ndarray,
) -> dict:
    estimated = np.linalg.inv(old_pose) @ transform @ pred_pose
    target = np.linalg.inv(old_pose) @ target_pose
    estimated_vector, target_vector = estimated[:3, 3], target[:3, 3]
    estimated_norm, target_norm = float(np.linalg.norm(estimated_vector)), float(np.linalg.norm(target_vector))
    cosine = float(np.dot(estimated_vector, target_vector) / max(estimated_norm * target_norm, 1e-8))
    return {
        "translation_direction_error_deg": float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))),
        "translation_scale_ratio": estimated_norm / max(target_norm, 1e-8),
        "translation_scale_log_abs": abs(math.log(max(estimated_norm, 1e-8) / max(target_norm, 1e-8))),
    }


def candidate_row(
    transform: np.ndarray | None,
    pred_pose: np.ndarray,
    target_pose: np.ndarray,
    old_pose: np.ndarray,
    query_human: dict | None,
    old_humans: list[dict | None],
    diagnostics: dict | None = None,
) -> dict:
    if transform is None:
        return {"fit_failed": True, "transform": None, **(diagnostics or {})}
    return {
        **direct_transform_error(transform, pred_pose, target_pose),
        **relative_translation_diagnostics(transform, pred_pose, old_pose, target_pose),
        **human_diagnostics(transform, query_human, old_humans),
        "fit_failed": False,
        "transform": transform.tolist(),
        **(diagnostics or {}),
    }


def run_window(
    name: str,
    pair_rows: list[dict],
    old_predictions: list[dict],
    new_predictions: list[dict],
    old_views: list[dict],
    new_views: list[dict],
    old_poses: list[np.ndarray],
    new_poses: list[np.ndarray],
    fixed_transform: np.ndarray,
    pred_pose0: np.ndarray,
    target_pose0: np.ndarray,
    old_pose0: np.ndarray,
    query_human: dict | None,
    old_humans: list[dict | None],
    cloud_cache: dict,
    args: argparse.Namespace,
    old_da3_depths: np.ndarray | None = None,
    new_da3_depths: np.ndarray | None = None,
    old_da3_intrinsics: np.ndarray | None = None,
    new_da3_intrinsics: np.ndarray | None = None,
) -> dict:
    coarse, rotation_meta = aggregate_coarse(pair_rows, old_poses, new_poses)
    wide_rotation = coarse[:3, :3]
    rotation_fixed = np.eye(4, dtype=np.float32)
    rotation_fixed[:3, :3] = wide_rotation
    rotation_fixed[:3, 3] = fixed_transform[:3, 3]
    outputs = {
        "coarse": candidate_row(coarse, pred_pose0, target_pose0, old_pose0, query_human, old_humans, rotation_meta),
        "rotation_fixed_translation": candidate_row(
            rotation_fixed, pred_pose0, target_pose0, old_pose0, query_human, old_humans, rotation_meta
        ),
    }
    essential_transform, essential_meta = essential_rotation_candidate(
        pair_rows,
        old_views,
        new_views,
        old_poses,
        new_poses,
    )
    if essential_transform is not None:
        essential_transform[:3, 3] = fixed_transform[:3, 3]
    outputs["essential_rotation_fixed_translation"] = candidate_row(
        essential_transform,
        pred_pose0,
        target_pose0,
        old_pose0,
        query_human,
        old_humans,
        essential_meta,
    )
    da3_transform, da3_meta = da3_correspondence_candidate(
        pair_rows,
        old_views,
        new_views,
        old_poses,
        new_poses,
        old_da3_depths,
        new_da3_depths,
        old_da3_intrinsics,
        new_da3_intrinsics,
    )
    outputs["da3_metric_full"] = candidate_row(
        da3_transform,
        pred_pose0,
        target_pose0,
        old_pose0,
        query_human,
        old_humans,
        da3_meta,
    )
    da3_fixed_transform, da3_fixed_meta = da3_correspondence_candidate(
        pair_rows,
        old_views,
        new_views,
        old_poses,
        new_poses,
        old_da3_depths,
        new_da3_depths,
        old_da3_intrinsics,
        new_da3_intrinsics,
        fixed_rotation=wide_rotation,
    )
    outputs["da3_wide_rotation_metric_translation"] = candidate_row(
        da3_fixed_transform,
        pred_pose0,
        target_pose0,
        old_pose0,
        query_human,
        old_humans,
        da3_fixed_meta,
    )
    for policy in ("full", "downweighted", "background"):
        correspondences = collect_correspondences(
            pair_rows,
            old_predictions,
            new_predictions,
            old_views,
            new_views,
            old_poses,
            new_poses,
            policy,
        )
        full_fit = None
        rotation_fit = None
        if len(correspondences.get("source", [])) >= 6:
            full_fit = robust_fit(correspondences["source"], correspondences["target"], correspondences["weight"], False)
            rotation_fit = fixed_rotation_fit(
                correspondences["source"], correspondences["target"], correspondences["weight"], wide_rotation
            )
        fit_meta = metric_fit_diagnostics(rotation_fit, correspondences, pair_rows)
        outputs[f"metric_full_{policy}"] = candidate_row(
            None if full_fit is None else fit_to_transform(full_fit),
            pred_pose0,
            target_pose0,
            old_pose0,
            query_human,
            old_humans,
            metric_fit_diagnostics(full_fit, correspondences, pair_rows),
        )
        metric_rotation = (
            None if full_fit is None else fit_to_transform(full_fit)[:3, :3]
        )
        da3_metric_rotation_transform, da3_metric_rotation_meta = (
            da3_correspondence_candidate(
                pair_rows,
                old_views,
                new_views,
                old_poses,
                new_poses,
                old_da3_depths,
                new_da3_depths,
                old_da3_intrinsics,
                new_da3_intrinsics,
                fixed_rotation=metric_rotation,
            )
            if policy == "full" and metric_rotation is not None
            else (None, {"da3_fit_failed": True})
        )
        outputs[f"da3_metric_rotation_metric_translation_{policy}"] = candidate_row(
            da3_metric_rotation_transform,
            pred_pose0,
            target_pose0,
            old_pose0,
            query_human,
            old_humans,
            {
                **da3_metric_rotation_meta,
                **metric_fit_diagnostics(full_fit, correspondences, pair_rows),
            },
        )
        metric_transform = None if rotation_fit is None else fit_to_transform(rotation_fit)
        outputs[f"metric_rotation_{policy}"] = candidate_row(
            metric_transform,
            pred_pose0,
            target_pose0,
            old_pose0,
            query_human,
            old_humans,
            {**fit_meta, **rotation_meta},
        )
        mixed_transform = None
        if full_fit is not None:
            mixed_transform = np.eye(4, dtype=np.float32)
            mixed_transform[:3, :3] = wide_rotation
            mixed_transform[:3, 3] = full_fit["translation"]
        outputs[f"wide_rotation_metric_translation_{policy}"] = candidate_row(
            mixed_transform,
            pred_pose0,
            target_pose0,
            old_pose0,
            query_human,
            old_humans,
            {**metric_fit_diagnostics(full_fit, correspondences, pair_rows), **rotation_meta},
        )
        if mixed_transform is not None:
            static_only = policy == "background"
            cloud_key = (len(old_predictions), len(new_predictions), static_only)
            if cloud_key not in cloud_cache:
                cloud_cache[cloud_key] = dense_clouds(
                    old_predictions,
                    new_predictions,
                    old_views,
                    new_views,
                    old_poses,
                    new_poses,
                    int(args.point_samples_per_frame),
                    static_only,
                )
            source, target, source_conf, target_conf = cloud_cache[cloud_key]
            refined, refinement_meta = residual_icp(
                mixed_transform, source, target, source_conf, target_conf, torch.device(args.device)
            )
            outputs[f"hybrid_{policy}"] = candidate_row(
                refined,
                pred_pose0,
                target_pose0,
                old_pose0,
                query_human,
                old_humans,
                {**fit_meta, **rotation_meta, **refinement_meta},
            )
        else:
            outputs[f"hybrid_{policy}"] = candidate_row(
                None, pred_pose0, target_pose0, old_pose0, query_human, old_humans, fit_meta
            )
    return {"window": name, "rotation_consensus": rotation_meta, "candidates": outputs}


def predicted_human_summary(prediction: dict, view: dict, pred_layer) -> dict | None:
    human = predicted_human(prediction, view["camera_intrinsics"], pred_layer)
    if human is None:
        return None
    pose = camera_matrix(prediction)
    return {
        "root": transform_points(pose, human["root"][None])[0],
        "torso": (pose[:3, :3] @ human["torso"]).astype(np.float32),
    }


def run_case(
    record: dict,
    human3r,
    pred_layer,
    vggt,
    da3,
    v14_cases: dict[str, dict],
    args: argparse.Namespace,
) -> dict:
    spec = record_spec(record, args)
    device = torch.device(args.device)
    reset_views = configure_views(one_batch(build_dataset([spec], False, args)), device, human3r.mhmr_img_res)[:3]
    old_views_all = configure_views(one_batch(old_a_dataset(spec, args)), device, human3r.mhmr_img_res)
    started = time.perf_counter()
    with torch.no_grad():
        continuous_predictions, _ = human3r.forward_recurrent_lighter(
            old_views_all + reset_views, str(device), ret_state=False, use_ttt3r=False
        )
        reset_predictions, _ = human3r.forward_recurrent_lighter(
            reset_views, str(device), ret_state=False, use_ttt3r=False
        )
    human3r_seconds = time.perf_counter() - started
    old_predictions_all = continuous_predictions[: len(old_views_all)]
    continue_predictions = continuous_predictions[len(old_views_all) :]
    old_views = old_views_all[-3:]
    old_predictions = old_predictions_all[-3:]
    old_poses = [camera_matrix(row) for row in old_predictions]
    new_poses = [camera_matrix(row) for row in reset_predictions]
    pred_pose0 = new_poses[0]
    old_pose0 = old_poses[-1]
    gt_pose0 = gt_pose_from_view(reset_views[0]).detach().float().cpu().numpy().astype(np.float32)
    old_gt_pose = gt_pose_from_view(old_views[-1]).detach().float().cpu().numpy().astype(np.float32)
    old_from_raw = old_pose0 @ np.linalg.inv(old_gt_pose)
    target_pose0 = old_from_raw @ gt_pose0
    boundary_gt = target_pose0 @ np.linalg.inv(pred_pose0)
    fixed_transform, fixed_name = fixed_explicit_human3r_gauge(args.candidate_root, record["pattern_id"])
    old_humans = [predicted_human_summary(prediction, view, pred_layer) for prediction, view in zip(old_predictions, old_views)]
    reset_humans = [predicted_human_summary(prediction, view, pred_layer) for prediction, view in zip(reset_predictions, reset_views)]

    da3_old_depths = None
    da3_new_depths = None
    da3_old_intrinsics = None
    da3_new_intrinsics = None
    da3_seconds = 0.0
    if da3 is not None:
        from v18_da3_metric_depth_probe import metric_inference

        da3_views = [*old_views, *reset_views]
        da3_images = [view_rgb(view) for view in da3_views]
        da3_input_intrinsics = np.stack(
            [
                view["camera_intrinsics"][0]
                .detach()
                .float()
                .cpu()
                .numpy()
                for view in da3_views
            ]
        ).astype(np.float32)
        da3_depths, da3_processed_intrinsics, da3_seconds = metric_inference(
            da3,
            da3_images,
            da3_input_intrinsics,
            int(args.da3_process_res),
        )
        da3_old_depths = da3_depths[: len(old_views)]
        da3_new_depths = da3_depths[len(old_views) :]
        da3_old_intrinsics = da3_processed_intrinsics[: len(old_views)]
        da3_new_intrinsics = da3_processed_intrinsics[len(old_views) :]

    prepared = {}
    vggt_seconds = {}
    window_outputs = {}
    cloud_cache = {}
    for visual_policy, background_only in (("full_rgb", False), ("background_only", True)):
        old_meta = [square_view(view, background_only, args.query_rows, args.query_cols) for view in old_views]
        new_meta = [square_view(view, background_only, args.query_rows, args.query_cols) for view in reset_views]
        prepared[visual_policy] = {"old": old_meta, "new": new_meta}
        for window_name, old_count, new_count in (("1p1", 1, 1), ("3p3", 3, 3)):
            selected_old_meta = old_meta[-old_count:]
            selected_new_meta = new_meta[:new_count]
            specs = pair_specs(len(selected_old_meta), len(selected_new_meta))
            rows, elapsed = run_vggt_pairs(vggt, selected_old_meta, selected_new_meta, specs, args)
            selected_old_predictions = old_predictions[-old_count:]
            selected_new_predictions = reset_predictions[:new_count]
            selected_old_views = old_views[-old_count:]
            selected_new_views = reset_views[:new_count]
            selected_old_poses = old_poses[-old_count:]
            selected_new_poses = new_poses[:new_count]
            result = run_window(
                f"{visual_policy}_{window_name}",
                rows,
                selected_old_predictions,
                selected_new_predictions,
                selected_old_views,
                selected_new_views,
                selected_old_poses,
                selected_new_poses,
                fixed_transform,
                pred_pose0,
                target_pose0,
                old_pose0,
                reset_humans[0],
                old_humans,
                cloud_cache,
                args,
                None if da3_old_depths is None else da3_old_depths[-old_count:],
                None if da3_new_depths is None else da3_new_depths[:new_count],
                None
                if da3_old_intrinsics is None
                else da3_old_intrinsics[-old_count:],
                None
                if da3_new_intrinsics is None
                else da3_new_intrinsics[:new_count],
            )
            result["pair_count"] = len(rows)
            result["vggt_seconds"] = elapsed
            window_outputs[f"{visual_policy}_{window_name}"] = result
            vggt_seconds[f"{visual_policy}_{window_name}"] = elapsed

    v14 = v14_cases.get(record["pattern_id"], {})
    v14_variants = v14.get("variants", {})
    v14_one = v14_variants.get("explicit_icp_dino_mhmr__confidence_spatial__1f")
    v14_three = v14_variants.get("explicit_icp_dino_mhmr__confidence_spatial__3f")
    baselines = {
        "original_continue": {
            **direct_transform_error(np.eye(4, dtype=np.float32), camera_matrix(continue_predictions[0]), target_pose0),
            "fit_failed": False,
        },
        "hard_reset": candidate_row(
            np.eye(4, dtype=np.float32), pred_pose0, target_pose0, old_pose0, reset_humans[0], old_humans
        ),
        "fixed_explicit": candidate_row(
            fixed_transform, pred_pose0, target_pose0, old_pose0, reset_humans[0], old_humans, {"name": fixed_name}
        ),
        "v14_world_memory_1f": v14_one,
        "v14_world_memory_3f": v14_three,
        "boundary_oracle": candidate_row(
            boundary_gt, pred_pose0, target_pose0, old_pose0, reset_humans[0], old_humans
        ),
    }
    human_ratios = [meta["human_ratio"] for policy in prepared.values() for side in policy.values() for meta in side]
    return {
        "case_name": record["pattern_id"],
        "record": record,
        "protocol": {
            "human3r_frozen": True,
            "vggt_frozen": True,
            "da3_frozen": da3 is not None,
            "gt_cut_idx_used": True,
            "gt_camera_use": "evaluation_and_epipolar_diagnostics_only",
            "gt_depth_used": False,
            "gt_scene_mesh_used": False,
            "gt_correspondence_used": False,
            "precut_frames": 3,
            "postcut_frames": 3,
            "shot_transform_count": 1,
        },
        "texture_score": texture_score(reset_views[0]),
        "human_image_ratio": float(np.mean(human_ratios)),
        "baselines": baselines,
        "windows": window_outputs,
        "timing_seconds": {
            "human3r": human3r_seconds,
            "da3": da3_seconds,
            **vggt_seconds,
        },
        "peak_gpu_memory_gb": float(torch.cuda.max_memory_allocated(device) / (1024**3)),
    }


def main() -> None:
    args = parse_args()
    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("V15 Human3R and wide-baseline inference require CUDA")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard index")
    output_path = args.output_dir / f"v15_candidates_shard_{args.shard_index:02d}_of_{args.num_shards:02d}.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if output_path.is_file() and not args.overwrite:
        print(f">> exists {output_path}")
        return
    records = read_jsonl(args.records)
    if args.sources:
        allowed_sources = set(args.sources)
        records = [row for row in records if str(row.get("source")) in allowed_sources]
    selected = [row for index, row in enumerate(records) if index % args.num_shards == args.shard_index]
    if args.max_cases > 0:
        selected = selected[: args.max_cases]
    human3r = build_model(args)
    _, pred_layer = build_smpl_models(human3r, torch.device(args.device))
    vggt = build_vggt(args)
    da3 = build_da3(args)
    v14_cases = load_v14_cases(args.v14_candidate_dir)
    cases = []
    started = time.perf_counter()
    for index, record in enumerate(selected):
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))
        case = run_case(record, human3r, pred_layer, vggt, da3, v14_cases, args)
        cases.append(case)
        main_candidate = case["windows"]["full_rgb_3p3"]["candidates"]["hybrid_downweighted"]
        print(
            f">> [{index + 1}/{len(selected)}] {record['pattern_id']} "
            f"T={main_candidate.get('camera_translation_error_m', float('nan')):.3f} "
            f"R={main_candidate.get('camera_rotation_error_deg', float('nan')):.2f} "
            f"mem={case['peak_gpu_memory_gb']:.1f}GB",
            flush=True,
        )
        torch.cuda.empty_cache()
    report = {
        "experiment": "V15 Wide-Baseline Boundary Bridge Candidates",
        "wide_baseline_model": "VGGT-1B frozen camera and track heads",
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "case_count": len(cases),
        "elapsed_seconds": time.perf_counter() - started,
        "cases": cases,
    }
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    print(f">> wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
