#!/usr/bin/env python3
"""Probe Human3R-internal camera-local root-depth correction after frozen V14 B0.

The probe never changes the camera, pointmap, B0, or shared Boundary.  It only
tests whether Human3R's own dense pointmap and semantic person mask can provide
a conservative translation correction for each predicted SMPL-X body.

GT identity and geometry are used only by the evaluator.  Candidate generation
uses predicted outputs exclusively.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import roma
import torch
from scipy.optimize import least_squares


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


from scripts.boundary_human3r_reset_support import build_smpl_models  # noqa: E402
from versions.v13 import gt_id_consensus as geometry  # noqa: E402
from versions.v13.native_token_probe import jsonable, tensor_numpy  # noqa: E402
from versions.v14.probe_b0_identity_matching import strict_cache  # noqa: E402
from versions.v14.run_v14_2_single_sequence import (  # noqa: E402
    camera_matrix,
    configure_model,
    set_event_indices,
)


DEFAULT_MODEL = Path(
    "/dev/shm/movie3r_v14_1/"
    "v14_1_v9_event_only_boundary_geometry_self20_fp32_e80/checkpoint-best.pth"
)
DEFAULT_DATA = Path(
    "/data/wangzheng/iJCV-CODE/data/MultiHuman/Real-World-Capture/extracted"
)
DEFAULT_OUTPUT = REPO_ROOT / "output/v14/internal_root_depth"
SEQUENCE_INPUTS = {
    "three": {
        "report": REPO_ROOT
        / "output/v14/b0_identity_matching/v14_b0_identity_matching.json",
        "cache": REPO_ROOT
        / "output/v20_phase1_gt_id_multihuman_consensus/case_cache",
    },
    "dance": {
        "report": REPO_ROOT
        / "output/v14/b0_identity_matching_extended/dance/v14_b0_identity_matching.json",
        "cache": REPO_ROOT / "output/v13/dance_phase2/case_cache",
    },
    "box": {
        "report": REPO_ROOT
        / "output/v14/b0_identity_matching_extended/box/v14_b0_identity_matching.json",
        "cache": REPO_ROOT / "output/v13/box_phase3/case_cache",
    },
}
METHODS = (
    "raw",
    "pointmap_z",
    "mask_translation",
    "candidate_mean",
    "conservative_gate",
    "oracle_candidate",
    "oracle_gt_local_root",
    "oracle_gt_root",
)
METRICS = (
    "root_error_m",
    "joint_error_m",
    "vertex_error_m",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", nargs="+", choices=tuple(SEQUENCE_INPUTS), default=("three",))
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--evaluation_only", action="store_true")
    parser.add_argument("--mask_threshold", type=float, default=0.50)
    parser.add_argument("--point_radius", type=float, default=0.025)
    parser.add_argument("--max_point_iqr_m", type=float, default=0.45)
    parser.add_argument("--max_relative_shift", type=float, default=0.30)
    parser.add_argument("--agreement_m", type=float, default=0.22)
    parser.add_argument("--agreement_relative", type=float, default=0.12)
    parser.add_argument("--gate_multiscale_range_m", type=float, default=0.005)
    parser.add_argument("--gate_point_iqr_m", type=float, default=0.10)
    parser.add_argument("--gate_relative_shift", type=float, default=0.08)
    parser.add_argument("--gate_absolute_shift_m", type=float, default=0.20)
    return parser.parse_args()


def finite_distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {name: float("nan") for name in ("mean", "median", "p90", "p95", "max")}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return geometry.transform_points(
        np.asarray(transform, dtype=np.float64), np.asarray(points, dtype=np.float64)
    )


def read_report(path: Path) -> tuple[dict[str, dict], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = {str(row["case"]["key"]): row for row in payload["cases"]}
    return rows, payload


def prediction_mask(prediction: dict) -> np.ndarray | None:
    value = prediction.get("msk")
    if value is None:
        return None
    array = tensor_numpy(value)
    array = np.asarray(array, dtype=np.float32).squeeze()
    if array.ndim != 2:
        raise ValueError(f"Unexpected Human3R mask shape: {array.shape}")
    return array


def prediction_pointmap(prediction: dict) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(tensor_numpy(prediction["pts3d_in_self_view"]), dtype=np.float32).squeeze()
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"Unexpected pointmap shape: {points.shape}")
    confidence = prediction.get("conf_self")
    if confidence is None:
        confidence_array = np.ones(points.shape[:2], dtype=np.float32)
    else:
        confidence_array = np.asarray(tensor_numpy(confidence), dtype=np.float32).squeeze()
    if confidence_array.shape != points.shape[:2]:
        raise ValueError(
            f"Pointmap/confidence mismatch: {points.shape} vs {confidence_array.shape}"
        )
    return points, confidence_array


def decode_local_humans(prediction: dict, view: dict, debug: dict, layer) -> list[dict]:
    count = int(prediction["smpl_transl"].shape[1])
    if count == 0:
        return []
    device = next(layer.parameters()).device
    rotmat = prediction["smpl_rotmat"][0, :count].to(device=device, dtype=torch.float32)
    rotvec = roma.rotmat_to_rotvec(rotmat)
    shape = prediction["smpl_shape"][0, :count].to(device=device, dtype=torch.float32)
    transl = prediction["smpl_transl"][0, :count].to(device=device, dtype=torch.float32)
    expression = prediction.get("smpl_expression")
    expression = (
        torch.zeros((count, 10), device=device, dtype=torch.float32)
        if expression is None
        else expression[0, :count].to(device=device, dtype=torch.float32)
    )
    intrinsic = view["K_mhmr"][0].to(device=device, dtype=torch.float32)
    intrinsics = intrinsic[None].expand(count, -1, -1)
    with torch.no_grad():
        body = layer(rotvec, shape, transl, None, None, K=intrinsics, expression=expression)
    joints = body["smpl_j3d"].detach().float().cpu().numpy()
    vertices = body["smpl_v3d"].detach().float().cpu().numpy()
    pixels = body["smpl_v2d"].detach().float().cpu().numpy()
    scores = np.asarray(tensor_numpy(debug.get("head_scores", np.ones(count)))).reshape(-1)
    locations = np.asarray(tensor_numpy(debug.get("head_locations", np.zeros((count, 2))))).reshape(-1, 2)
    output = []
    for index in range(count):
        output.append(
            {
                "detection_index": index,
                "score": float(scores[index]) if index < len(scores) else 1.0,
                "location_mhmr": locations[index].astype(np.float64),
                "translation": tensor_numpy(transl[index]).astype(np.float64),
                "root": joints[index, 0].astype(np.float64),
                "joints": joints[index].astype(np.float64),
                "vertices": vertices[index].astype(np.float64),
                "vertices_uv_mhmr": pixels[index].astype(np.float64),
            }
        )
    return output


def pointmap_location(location_mhmr: np.ndarray, mhmr_size: int, height: int, width: int) -> np.ndarray:
    scale = max(height, width) / float(mhmr_size)
    offset = np.asarray(((max(height, width) - width) // 2, (max(height, width) - height) // 2))
    uv = np.asarray(location_mhmr, dtype=np.float64) * scale - offset
    return np.clip(uv, (0.0, 0.0), (width - 1.0, height - 1.0))


def predicted_surface_offset(human: dict, radius_px: float) -> tuple[float, dict]:
    uv = np.asarray(human["vertices_uv_mhmr"], dtype=np.float64)
    vertices = np.asarray(human["vertices"], dtype=np.float64)
    center = np.asarray(human["location_mhmr"], dtype=np.float64)
    distance = np.linalg.norm(uv - center[None], axis=1)
    valid = np.isfinite(distance) & np.isfinite(vertices).all(axis=1) & (vertices[:, 2] > 0.05)
    selected = valid & (distance <= float(radius_px))
    if int(selected.sum()) < 8:
        ids = np.flatnonzero(valid)
        if not len(ids):
            return 0.0, {"status": "no_projected_vertices"}
        ids = ids[np.argsort(distance[ids])[: min(32, len(ids))]]
        selected = np.zeros(len(vertices), dtype=bool)
        selected[ids] = True
    surface_depth = float(np.median(vertices[selected, 2]))
    return surface_depth - float(human["root"][2]), {
        "status": "ok",
        "vertex_count": int(selected.sum()),
        "surface_depth_m": surface_depth,
    }


def pointmap_candidate(
    human: dict,
    points: np.ndarray,
    confidence: np.ndarray,
    person_mask: np.ndarray | None,
    mhmr_size: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray | None, dict]:
    height, width = points.shape[:2]
    location = pointmap_location(human["location_mhmr"], mhmr_size, height, width)
    radius = max(3, int(round(float(args.point_radius) * min(height, width))))
    yy, xx = np.ogrid[:height, :width]
    base_valid = np.isfinite(points).all(axis=2) & np.isfinite(confidence)
    base_valid &= (points[:, :, 2] > 0.05) & (points[:, :, 2] < 50.0)
    if person_mask is not None:
        resized_mask = cv2.resize(
            person_mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        base_valid &= resized_mask
    radius_scales = (0.55, 1.0, 1.55)
    depth_rows = []
    for radius_scale in radius_scales:
        current_radius = max(2, int(round(radius * radius_scale)))
        patch = (
            (xx - location[0]) ** 2 + (yy - location[1]) ** 2
            <= current_radius * current_radius
        )
        valid = patch & base_valid
        if int(valid.sum()) < 8:
            continue
        depths = points[:, :, 2][valid]
        conf_values = confidence[valid]
        keep = conf_values >= np.percentile(conf_values, 35)
        depths = depths[keep]
        if len(depths) < 6:
            continue
        depth_rows.append(
            {
                "radius_px": current_radius,
                "point_count": int(len(depths)),
                "median_m": float(np.median(depths)),
                "iqr_m": float(np.percentile(depths, 75) - np.percentile(depths, 25)),
            }
        )
    if not depth_rows:
        return None, {"status": "too_few_masked_points", "point_count": 0}
    medians = np.asarray([row["median_m"] for row in depth_rows], dtype=np.float64)
    median = float(np.median(medians))
    all_radius = max(row["radius_px"] for row in depth_rows)
    all_patch = (xx - location[0]) ** 2 + (yy - location[1]) ** 2 <= all_radius * all_radius
    all_valid = all_patch & base_valid
    all_depth = points[:, :, 2][all_valid]
    q25, q75 = np.percentile(all_depth, (25, 75))
    iqr = float(q75 - q25)
    if iqr > float(args.max_point_iqr_m):
        return None, {
            "status": "high_depth_iqr",
            "depth_iqr_m": iqr,
            "point_count": int(len(all_depth)),
            "radius_rows": depth_rows,
        }
    surface_offset, surface_debug = predicted_surface_offset(
        human, radius_px=max(12.0, radius * mhmr_size / max(height, width))
    )
    target_root_depth = float(median - surface_offset)
    shift = np.asarray((0.0, 0.0, target_root_depth - float(human["root"][2])))
    return shift, {
        "status": "ok",
        "location_pointmap": location,
        "radius_px": radius,
        "point_count": int(len(all_depth)),
        "surface_depth_m": float(median),
        "depth_iqr_m": iqr,
        "multiscale_range_m": float(np.max(medians) - np.min(medians)),
        "radius_rows": depth_rows,
        "mask_restricted": person_mask is not None,
        "surface_offset_m": float(surface_offset),
        "root_shift_m": shift,
        "surface_debug": surface_debug,
    }


def split_person_mask(
    mask: np.ndarray,
    locations: np.ndarray,
    threshold: float,
) -> list[np.ndarray]:
    foreground = np.isfinite(mask) & (mask >= float(threshold))
    yy, xx = np.nonzero(foreground)
    output = [np.zeros_like(foreground) for _ in range(len(locations))]
    if not len(xx) or not len(locations):
        return output
    pixels = np.stack((xx, yy), axis=1).astype(np.float64)
    distances = np.sum((pixels[:, None] - locations[None]) ** 2, axis=2)
    owner = np.argmin(distances, axis=1)
    for index in range(len(locations)):
        selected = owner == index
        output[index][yy[selected], xx[selected]] = True
    return output


def quantile_box(binary: np.ndarray) -> np.ndarray | None:
    yy, xx = np.nonzero(binary)
    if len(xx) < 64:
        return None
    return np.asarray(
        (
            np.percentile(xx, 2),
            np.percentile(yy, 2),
            np.percentile(xx, 98),
            np.percentile(yy, 98),
        ),
        dtype=np.float64,
    )


def projected_quantile_box(vertices: np.ndarray, intrinsic: np.ndarray) -> np.ndarray | None:
    valid = np.isfinite(vertices).all(axis=1) & (vertices[:, 2] > 0.05)
    if int(valid.sum()) < 64:
        return None
    projected = vertices[valid, :2] / vertices[valid, 2:3]
    projected = projected @ intrinsic[:2, :2].T + intrinsic[:2, 2]
    return np.asarray(
        (
            np.percentile(projected[:, 0], 2),
            np.percentile(projected[:, 1], 2),
            np.percentile(projected[:, 0], 98),
            np.percentile(projected[:, 1], 98),
        ),
        dtype=np.float64,
    )


def mask_translation_candidate(
    human: dict,
    person_mask: np.ndarray,
    intrinsic: np.ndarray,
    image_size: int,
) -> tuple[np.ndarray | None, dict]:
    observed = quantile_box(person_mask)
    if observed is None:
        return None, {"status": "too_few_mask_pixels", "pixel_count": int(person_mask.sum())}
    height = max(float(observed[3] - observed[1]), 1.0)
    width = max(float(observed[2] - observed[0]), 1.0)
    touches_edge = bool(
        observed[0] <= 2 or observed[1] <= 2 or observed[2] >= image_size - 3 or observed[3] >= image_size - 3
    )
    if touches_edge:
        return None, {"status": "truncated_mask", "observed_box": observed}
    vertices = np.asarray(human["vertices"], dtype=np.float64)
    root = np.asarray(human["root"], dtype=np.float64)
    location = np.asarray(human["location_mhmr"], dtype=np.float64)
    scale = np.asarray((max(width, 40.0), max(height, 80.0), max(width, 40.0), max(height, 80.0)))

    def residual(delta: np.ndarray) -> np.ndarray:
        shifted = vertices + np.asarray(delta)[None]
        box = projected_quantile_box(shifted, intrinsic)
        if box is None:
            return np.full(7, 10.0, dtype=np.float64)
        shifted_root = root + np.asarray(delta)
        root_uv = shifted_root[:2] / max(shifted_root[2], 0.05)
        root_uv = intrinsic[:2, :2] @ root_uv + intrinsic[:2, 2]
        box_residual = (box - observed) / scale
        pelvis_residual = (root_uv - location) / np.asarray((max(width, 40.0), max(height, 80.0)))
        depth_regularizer = np.asarray((0.05 * delta[2] / max(root[2], 0.25),))
        return np.r_[box_residual, pelvis_residual, depth_regularizer]

    z = max(float(root[2]), 0.25)
    xy_bound = max(0.75, 0.35 * z)
    lower = np.asarray((-xy_bound, -xy_bound, max(-0.55 * z, 0.08 - np.min(vertices[:, 2]))))
    upper = np.asarray((xy_bound, xy_bound, 0.80 * z))
    before = residual(np.zeros(3, dtype=np.float64))
    result = least_squares(
        residual,
        np.zeros(3, dtype=np.float64),
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=0.20,
        max_nfev=40,
    )
    after = residual(result.x)
    before_cost = float(np.mean(np.square(before[:6])))
    after_cost = float(np.mean(np.square(after[:6])))
    if not result.success or not np.isfinite(result.x).all():
        return None, {"status": "optimization_failed", "message": str(result.message)}
    if after_cost >= before_cost * 0.95:
        return None, {
            "status": "no_reprojection_improvement",
            "before_cost": before_cost,
            "after_cost": after_cost,
        }
    return result.x.astype(np.float64), {
        "status": "ok",
        "pixel_count": int(person_mask.sum()),
        "observed_box": observed,
        "before_cost": before_cost,
        "after_cost": after_cost,
        "root_shift_m": result.x,
        "nfev": int(result.nfev),
    }


def conservative_fusion(
    human: dict,
    point_shift: np.ndarray | None,
    point_debug: dict,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict]:
    zero = np.zeros(3, dtype=np.float64)
    if point_shift is None or point_debug.get("status") != "ok":
        return zero, {"status": "fallback_raw", "reason": "missing_pointmap_candidate"}
    root_depth = max(abs(float(human["root"][2])), 0.25)
    absolute_shift = abs(float(point_shift[2]))
    relative_shift = absolute_shift / root_depth
    if relative_shift > float(args.gate_relative_shift):
        return zero, {
            "status": "fallback_raw",
            "reason": "large_relative_shift",
            "relative_shift": relative_shift,
        }
    if absolute_shift > float(args.gate_absolute_shift_m):
        return zero, {
            "status": "fallback_raw",
            "reason": "large_absolute_shift",
            "absolute_shift_m": absolute_shift,
        }
    multiscale_range = float(point_debug.get("multiscale_range_m", float("inf")))
    if multiscale_range > float(args.gate_multiscale_range_m):
        return zero, {
            "status": "fallback_raw",
            "reason": "multiscale_depth_disagreement",
            "multiscale_range_m": multiscale_range,
        }
    depth_iqr = float(point_debug.get("depth_iqr_m", float("inf")))
    if depth_iqr > float(args.gate_point_iqr_m):
        return zero, {
            "status": "fallback_raw",
            "reason": "high_point_depth_iqr",
            "depth_iqr_m": depth_iqr,
        }
    output = np.asarray(point_shift, dtype=np.float64).copy()
    return output, {
        "status": "accepted",
        "relative_shift": relative_shift,
        "absolute_shift_m": absolute_shift,
        "multiscale_range_m": multiscale_range,
        "depth_iqr_m": depth_iqr,
        "root_shift_m": output,
    }


def evaluate_shift(
    local_human: dict,
    shift: np.ndarray,
    final_camera: np.ndarray,
    target: dict,
    gauge: np.ndarray,
) -> dict:
    shifted_root = np.asarray(local_human["root"]) + shift
    shifted_joints = np.asarray(local_human["joints"]) + shift[None]
    shifted_vertices = np.asarray(local_human["vertices"]) + shift[None]
    final_root = transform_points(final_camera, shifted_root[None])[0]
    final_joints = transform_points(final_camera, shifted_joints)
    final_vertices = transform_points(final_camera, shifted_vertices)
    target_root = transform_points(gauge, target["root"][None])[0]
    target_joints = transform_points(gauge, target["joints"])
    target_vertices = transform_points(gauge, target["vertices"])
    joint_count = min(len(final_joints), len(target_joints))
    vertex_count = min(len(final_vertices), len(target_vertices))
    return {
        "root_error_m": float(np.linalg.norm(final_root - target_root)),
        "joint_error_m": float(
            np.linalg.norm(final_joints[:joint_count] - target_joints[:joint_count], axis=1).mean()
        ),
        "vertex_error_m": float(
            np.linalg.norm(final_vertices[:vertex_count] - target_vertices[:vertex_count], axis=1).mean()
        ),
        "final_root": final_root,
        "target_root": target_root,
    }


def oracle_shift(local_human: dict, final_camera: np.ndarray, target: dict, gauge: np.ndarray) -> np.ndarray:
    target_final = transform_points(gauge, target["root"][None])[0]
    target_local = transform_points(np.linalg.inv(final_camera), target_final[None])[0]
    return target_local - np.asarray(local_human["root"], dtype=np.float64)


def local_gt_shift(local_human: dict, target: dict, gt_post_camera: np.ndarray) -> np.ndarray:
    target_local = transform_points(
        np.linalg.inv(gt_post_camera), np.asarray(target["root"])[None]
    )[0]
    return target_local - np.asarray(local_human["root"], dtype=np.float64)


def local_root_metrics(
    local_human: dict,
    shift: np.ndarray,
    target: dict,
    gt_post_camera: np.ndarray,
) -> dict:
    predicted = np.asarray(local_human["root"], dtype=np.float64) + np.asarray(shift)
    target_local = transform_points(
        np.linalg.inv(gt_post_camera), np.asarray(target["root"])[None]
    )[0]
    return {
        "local_root_error_m": float(np.linalg.norm(predicted - target_local)),
        "local_depth_error_m": float(abs(predicted[2] - target_local[2])),
        "predicted_local_root": predicted,
        "target_local_root": target_local,
    }


def infer_raw_post(model, args: argparse.Namespace, cache: dict):
    frame_args = SimpleNamespace(**vars(args))
    # Keep extracted frames sequence-scoped; MultiHuman sequences reuse camera
    # indices and frame numbers but contain different images.
    frame_args.output_dir = Path(args.output_dir) / str(args.sequence)
    post_path = geometry.extract_video_frame(
        frame_args,
        int(cache["case"]["target_camera"]),
        int(cache["case"]["post_frame"]),
    )
    views = set_event_indices(geometry.prepare_full_square_input(model, [post_path], args), set())
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
        predictions, output_views, debug = model.forward_recurrent_lighter(
            views,
            str(args.device),
            ret_state=False,
            use_ttt3r=False,
            return_token_debug=True,
        )
    return predictions[0], output_views[0], debug[0]


def process_case(
    model,
    layer,
    args: argparse.Namespace,
    sequence: str,
    cache_path: Path,
    b0_row: dict,
) -> dict:
    cache = strict_cache(args, cache_path)
    started = time.perf_counter()
    prediction, view, debug = infer_raw_post(model, args, cache)
    raw_camera = camera_matrix(prediction).astype(np.float64)
    b0 = np.asarray(b0_row["boundaries"]["learned_b0"], dtype=np.float64)
    final_camera = b0 @ raw_camera
    cached_raw_camera = np.asarray(cache["poses"][-1], dtype=np.float64)
    camera_delta = {
        "translation_m": float(np.linalg.norm(raw_camera[:3, 3] - cached_raw_camera[:3, 3])),
        "rotation_deg": geometry.rotation_error_deg(raw_camera, cached_raw_camera),
    }
    local_humans = decode_local_humans(prediction, view, debug, layer)
    world_humans = geometry.layer_humans(prediction, view, debug, layer)
    height, width = [int(value) for value in tensor_numpy(view["true_shape"])[0]]
    assigned, assignment = geometry.assign_gt_identities(
        args,
        world_humans,
        raw_camera,
        int(cache["case"]["target_camera"]),
        int(cache["case"]["post_frame"]),
        width,
        height,
    )
    identity_by_detection = {
        int(human["detection_index"]): identity for identity, human in assigned.items()
    }
    points, confidence = prediction_pointmap(prediction)
    mask = prediction_mask(prediction)
    mhmr_size = int(view["img_mhmr"].shape[-1])
    if mask is not None and mask.shape != (mhmr_size, mhmr_size):
        mask = cv2.resize(mask, (mhmr_size, mhmr_size), interpolation=cv2.INTER_LINEAR)
    locations = np.stack([human["location_mhmr"] for human in local_humans]) if local_humans else np.empty((0, 2))
    person_masks = (
        split_person_mask(mask, locations, float(args.mask_threshold))
        if mask is not None
        else [np.zeros((mhmr_size, mhmr_size), dtype=bool) for _ in local_humans]
    )
    intrinsic = tensor_numpy(view["K_mhmr"])[0].astype(np.float64)
    pre_pose = np.asarray(cache["poses"][-2], dtype=np.float64)
    gt_pre = np.asarray(cache["gt"]["pre_c2w"], dtype=np.float64)
    gauge = pre_pose @ np.linalg.inv(gt_pre)
    gt_post_camera = np.asarray(cache["gt"]["post_c2w"], dtype=np.float64)
    people = []
    for human in local_humans:
        detection_index = int(human["detection_index"])
        identity = identity_by_detection.get(detection_index)
        if identity is None or identity not in cache["gt"]["post_humans"]:
            continue
        point_shift, point_debug = pointmap_candidate(
            human,
            points,
            confidence,
            person_masks[detection_index] if mask is not None else None,
            mhmr_size,
            args,
        )
        mask_shift, mask_debug = mask_translation_candidate(
            human, person_masks[detection_index], intrinsic, mhmr_size
        )
        mean_candidates = [value for value in (point_shift, mask_shift) if value is not None]
        mean_shift = (
            np.mean(np.stack(mean_candidates), axis=0)
            if mean_candidates
            else np.zeros(3, dtype=np.float64)
        )
        gated_shift, gate_debug = conservative_fusion(
            human, point_shift, point_debug, args
        )
        target = cache["gt"]["post_humans"][identity]
        gt_shift = oracle_shift(human, final_camera, target, gauge)
        gt_local_shift = local_gt_shift(human, target, gt_post_camera)
        candidates = {
            "raw": np.zeros(3, dtype=np.float64),
            "pointmap_z": point_shift if point_shift is not None else np.zeros(3, dtype=np.float64),
            "mask_translation": mask_shift if mask_shift is not None else np.zeros(3, dtype=np.float64),
            "candidate_mean": mean_shift,
            "conservative_gate": gated_shift,
            "oracle_gt_local_root": gt_local_shift,
            "oracle_gt_root": gt_shift,
        }
        evaluated = {
            name: {
                **evaluate_shift(human, shift, final_camera, target, gauge),
                **local_root_metrics(human, shift, target, gt_post_camera),
            }
            for name, shift in candidates.items()
        }
        valid_oracle = [
            name for name in ("raw", "pointmap_z", "mask_translation", "candidate_mean")
            if name == "raw" or (name == "pointmap_z" and point_shift is not None)
            or (name == "mask_translation" and mask_shift is not None)
            or (name == "candidate_mean" and bool(mean_candidates))
        ]
        best_name = min(valid_oracle, key=lambda name: evaluated[name]["root_error_m"])
        evaluated["oracle_candidate"] = dict(evaluated[best_name])
        candidates["oracle_candidate"] = candidates[best_name]
        people.append(
            {
                "identity": identity,
                "detection_index": detection_index,
                "raw_root_depth_m": float(human["root"][2]),
                "pointmap": point_debug,
                "mask": mask_debug,
                "gate": gate_debug,
                "oracle_candidate_name": best_name,
                "shifts": candidates,
                "metrics": evaluated,
            }
        )
    gt_post = gt_post_camera
    target_camera = gauge @ gt_post_camera
    camera_metrics = {
        "translation_error_m": float(np.linalg.norm(final_camera[:3, 3] - target_camera[:3, 3])),
        "rotation_error_deg": geometry.rotation_error_deg(final_camera, target_camera),
        "composite": float(
            np.linalg.norm(final_camera[:3, 3] - target_camera[:3, 3])
            + 0.02 * geometry.rotation_error_deg(final_camera, target_camera)
        ),
        "matrix": final_camera,
        "b0_matrix": b0,
    }
    output = {
        "sequence": sequence,
        "case": cache["case"],
        "runtime_seconds": time.perf_counter() - started,
        "camera_raw_vs_cache": camera_delta,
        "camera": camera_metrics,
        "mask_diagnostics": {
            "available": mask is not None,
            "shape": None if mask is None else mask.shape,
            "minimum": None if mask is None else float(np.nanmin(mask)),
            "maximum": None if mask is None else float(np.nanmax(mask)),
            "mean": None if mask is None else float(np.nanmean(mask)),
            "foreground_fraction": None if mask is None else float(np.mean(mask >= float(args.mask_threshold))),
        },
        "pointmap_diagnostics": {
            "shape": points.shape,
            "depth_median": float(np.nanmedian(points[:, :, 2])),
        },
        "assignment": assignment,
        "people": people,
    }
    del cache, prediction, view, debug
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output


def pairwise_metrics(case: dict, method: str) -> tuple[list[float], list[float]]:
    final = {
        person["identity"]: np.asarray(person["metrics"][method]["final_root"])
        for person in case["people"]
    }
    target = {
        person["identity"]: np.asarray(person["metrics"][method]["target_root"])
        for person in case["people"]
    }
    distance, vector = [], []
    for first, second in combinations(sorted(final), 2):
        predicted_vector = final[first] - final[second]
        target_vector = target[first] - target[second]
        distance.append(abs(float(np.linalg.norm(predicted_vector) - np.linalg.norm(target_vector))))
        vector.append(float(np.linalg.norm(predicted_vector - target_vector)))
    return distance, vector


def refresh_loaded_gate(case: dict, args: argparse.Namespace) -> dict:
    """Apply the frozen deployable gate to cached candidate rows without inference."""
    for person in case["people"]:
        point_debug = person["pointmap"]
        shift = np.asarray(person["shifts"]["pointmap_z"], dtype=np.float64)
        root_depth = max(abs(float(person["raw_root_depth_m"])), 0.25)
        accepted = (
            point_debug.get("status") == "ok"
            and abs(float(shift[2])) / root_depth <= float(args.gate_relative_shift)
            and abs(float(shift[2])) <= float(args.gate_absolute_shift_m)
            and float(point_debug.get("multiscale_range_m", float("inf")))
            <= float(args.gate_multiscale_range_m)
            and float(point_debug.get("depth_iqr_m", float("inf")))
            <= float(args.gate_point_iqr_m)
        )
        if accepted:
            person["gate"] = {
                "status": "accepted",
                "relative_shift": abs(float(shift[2])) / root_depth,
                "absolute_shift_m": abs(float(shift[2])),
                "multiscale_range_m": float(point_debug["multiscale_range_m"]),
                "depth_iqr_m": float(point_debug["depth_iqr_m"]),
                "root_shift_m": shift,
            }
            person["shifts"]["conservative_gate"] = shift
            person["metrics"]["conservative_gate"] = dict(
                person["metrics"]["pointmap_z"]
            )
        else:
            person["gate"] = {"status": "fallback_raw", "reason": "frozen_point_gate"}
            person["shifts"]["conservative_gate"] = np.zeros(3, dtype=np.float64)
            person["metrics"]["conservative_gate"] = dict(person["metrics"]["raw"])
    return case


def refresh_case_metrics_from_cache(case: dict, cache: dict) -> dict:
    """Rebuild local/world metrics from the strict Human3R case cache."""
    raw_camera = np.asarray(cache["poses"][-1], dtype=np.float64)
    final_camera = np.asarray(case["camera"]["matrix"], dtype=np.float64)
    gt_pre = np.asarray(cache["gt"]["pre_c2w"], dtype=np.float64)
    gt_post = np.asarray(cache["gt"]["post_c2w"], dtype=np.float64)
    gauge = np.asarray(cache["poses"][-2], dtype=np.float64) @ np.linalg.inv(gt_pre)
    raw_w2c = np.linalg.inv(raw_camera)
    for person in case["people"]:
        identity = person["identity"]
        predicted = cache["humans"][-1][identity]
        target = cache["gt"]["post_humans"][identity]
        local_human = {
            "root": transform_points(raw_w2c, np.asarray(predicted["root"])[None])[0],
            "joints": transform_points(raw_w2c, predicted["joints"]),
            "vertices": transform_points(raw_w2c, predicted["vertices"]),
        }
        person["raw_root_depth_m"] = float(local_human["root"][2])
        candidates = {
            name: np.asarray(person["shifts"].get(name, np.zeros(3)), dtype=np.float64)
            for name in (
                "raw",
                "pointmap_z",
                "mask_translation",
                "candidate_mean",
                "conservative_gate",
            )
        }
        candidates["oracle_gt_local_root"] = local_gt_shift(
            local_human, target, gt_post
        )
        candidates["oracle_gt_root"] = oracle_shift(
            local_human, final_camera, target, gauge
        )
        evaluated = {
            name: {
                **evaluate_shift(
                    local_human, shift, final_camera, target, gauge
                ),
                **local_root_metrics(local_human, shift, target, gt_post),
            }
            for name, shift in candidates.items()
        }
        valid_oracle = ["raw"]
        if person["pointmap"].get("status") == "ok":
            valid_oracle.append("pointmap_z")
        if person["mask"].get("status") == "ok":
            valid_oracle.append("mask_translation")
        if len(valid_oracle) > 1:
            valid_oracle.append("candidate_mean")
        best_name = min(
            valid_oracle, key=lambda name: evaluated[name]["root_error_m"]
        )
        candidates["oracle_candidate"] = candidates[best_name]
        evaluated["oracle_candidate"] = dict(evaluated[best_name])
        person["oracle_candidate_name"] = best_name
        person["shifts"] = candidates
        person["metrics"] = evaluated
    return case


def aggregate(cases: list[dict]) -> dict:
    output = {}
    for method in METHODS:
        people = [person for case in cases for person in case["people"]]
        row = {
            metric: finite_distribution(
                [float(person["metrics"][method][metric]) for person in people]
            )
            for metric in METRICS
        }
        pair_distance, pair_vector = [], []
        for case in cases:
            first, second = pairwise_metrics(case, method)
            pair_distance.extend(first)
            pair_vector.extend(second)
        row["pairwise_distance_error_m"] = finite_distribution(pair_distance)
        row["pairwise_vector_error_m"] = finite_distribution(pair_vector)
        row["local_root_error_m"] = finite_distribution(
            [float(person["metrics"][method]["local_root_error_m"]) for person in people]
        )
        row["local_depth_error_m"] = finite_distribution(
            [float(person["metrics"][method]["local_depth_error_m"]) for person in people]
        )
        raw = np.asarray([person["metrics"]["raw"]["root_error_m"] for person in people])
        current = np.asarray([person["metrics"][method]["root_error_m"] for person in people])
        row["improved_fraction"] = float(np.mean(current < raw - 1e-8)) if len(raw) else float("nan")
        row["harmed_over_5cm_fraction"] = float(np.mean(current > raw + 0.05)) if len(raw) else float("nan")
        row["catastrophic_root_over_2m_fraction"] = float(np.mean(current > 2.0)) if len(raw) else float("nan")
        output[method] = row
    gates = [person["gate"]["status"] == "accepted" for case in cases for person in case["people"]]
    output["coverage"] = {
        "person_gate_coverage": float(np.mean(gates)) if gates else 0.0,
        "cut_any_gate_coverage": float(np.mean([any(p["gate"]["status"] == "accepted" for p in case["people"]) for case in cases])) if cases else 0.0,
        "cut_all_gate_coverage": float(np.mean([bool(case["people"]) and all(p["gate"]["status"] == "accepted" for p in case["people"]) for case in cases])) if cases else 0.0,
    }
    output["camera_invariance"] = {
        "unique_b0_matrices": int(len({np.asarray(case["camera"]["b0_matrix"]).tobytes() for case in cases})),
        "all_methods_share_camera": True,
        "translation_error_m": finite_distribution([case["camera"]["translation_error_m"] for case in cases]),
        "rotation_error_deg": finite_distribution([case["camera"]["rotation_error_deg"] for case in cases]),
        "composite": finite_distribution([case["camera"]["composite"] for case in cases]),
    }
    return output


def markdown(report: dict) -> str:
    lines = [
        "# V14 Human3R-Internal Root Depth Probe",
        "",
        "Camera, pointmap, learned B0, and the shared Boundary are unchanged for every method. ",
        "Only each person's camera-local SMPL-X translation is tested.",
        "",
    ]
    for sequence, summary in report["summary"].items():
        lines.extend(
            [
                f"## {sequence}",
                "",
                "| Method | Root mean | Root median | Root P90 | Root P95 | Joint mean | Vertex mean | >5cm harm |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method in METHODS:
            row = summary[method]
            lines.append(
                f"| {method} | {row['root_error_m']['mean']:.3f} | "
                f"{row['root_error_m']['median']:.3f} | {row['root_error_m']['p90']:.3f} | "
                f"{row['root_error_m']['p95']:.3f} | {row['joint_error_m']['mean']:.3f} | "
                f"{row['vertex_error_m']['mean']:.3f} | {100.0 * row['harmed_over_5cm_fraction']:.1f}% |"
            )
        coverage = summary["coverage"]
        camera = summary["camera_invariance"]
        lines.extend(
            [
                "",
                f"Conservative person coverage: `{100.0 * coverage['person_gate_coverage']:.1f}%`; "
                f"all-person cut coverage: `{100.0 * coverage['cut_all_gate_coverage']:.1f}%`.",
                f"Frozen B0 camera composite: `{camera['composite']['mean']:.3f}` mean; "
                "identical for every root method.",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.model_path.is_file() and not args.evaluation_only:
        raise FileNotFoundError(args.model_path)
    model = None
    layer = None
    flags = None
    if not args.evaluation_only:
        from dust3r.model import ARCroco3DStereo

        model = ARCroco3DStereo.from_pretrained(str(args.model_path)).to(args.device)
        flags = configure_model(model)
        _, layer = build_smpl_models(model, torch.device(args.device))
    all_cases: dict[str, list[dict]] = {}
    source_reports = {}
    for sequence in args.sequences:
        geometry.IDENTITIES = geometry.SEQUENCE_IDENTITIES[sequence]
        args.sequence = sequence
        inputs = SEQUENCE_INPUTS[sequence]
        report_rows, source_report = read_report(inputs["report"])
        source_reports[sequence] = str(inputs["report"])
        keys = sorted(report_rows)
        if args.case:
            requested = set(args.case)
            keys = [key for key in keys if key in requested]
        if int(args.max_cases) > 0:
            keys = keys[: int(args.max_cases)]
        cases = []
        case_dir = args.output_dir / sequence / "cases"
        case_dir.mkdir(parents=True, exist_ok=True)
        for index, key in enumerate(keys, start=1):
            output_path = case_dir / f"{key}.json"
            cache_path = Path(inputs["cache"]) / f"{key}.pt"
            if not cache_path.is_file():
                raise FileNotFoundError(cache_path)
            if output_path.is_file() and not args.overwrite:
                row = json.loads(output_path.read_text(encoding="utf-8"))
                row = refresh_loaded_gate(row, args)
                row = refresh_case_metrics_from_cache(
                    row, strict_cache(args, cache_path)
                )
            else:
                if args.evaluation_only:
                    raise FileNotFoundError(output_path)
                row = process_case(model, layer, args, sequence, cache_path, report_rows[key])
                row = refresh_loaded_gate(row, args)
                row = refresh_case_metrics_from_cache(
                    row, strict_cache(args, cache_path)
                )
                output_path.write_text(
                    json.dumps(jsonable(row), indent=2, ensure_ascii=True), encoding="utf-8"
                )
            cases.append(row)
            print(
                f">> [{sequence} {index}/{len(keys)}] {key}: "
                f"people={len(row['people'])}, mask={row['mask_diagnostics']['available']}",
                flush=True,
            )
        all_cases[sequence] = cases
    report = {
        "experiment": "v14_internal_root_depth",
        "model_path": str(args.model_path),
        "model_flags": flags,
        "protocol": {
            "sequences": list(args.sequences),
            "candidate_generation_uses_gt": False,
            "gt_use": "identity assignment and metrics only",
            "camera_and_b0_unchanged": True,
            "per_person_boundary": False,
            "mask_threshold": float(args.mask_threshold),
            "point_radius": float(args.point_radius),
            "max_point_iqr_m": float(args.max_point_iqr_m),
            "max_relative_shift": float(args.max_relative_shift),
            "agreement_m": float(args.agreement_m),
            "agreement_relative": float(args.agreement_relative),
            "gate_multiscale_range_m": float(args.gate_multiscale_range_m),
            "gate_point_iqr_m": float(args.gate_point_iqr_m),
            "gate_relative_shift": float(args.gate_relative_shift),
            "gate_absolute_shift_m": float(args.gate_absolute_shift_m),
            "source_reports": source_reports,
        },
        "summary": {sequence: aggregate(cases) for sequence, cases in all_cases.items()},
        "cases": all_cases,
    }
    json_path = args.output_dir / "v14_internal_root_depth.json"
    md_path = args.output_dir / "README.md"
    json_path.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=True), encoding="utf-8")
    md_path.write_text(markdown(report), encoding="utf-8")
    print(f">> wrote {json_path}", flush=True)
    print(f">> wrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
