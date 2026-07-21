#!/usr/bin/env python3
"""V18 stage 2: recover metric human root depth from physical body projection."""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import least_squares


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from dust3r.utils.smpl_layer import SMPL_Layer  # noqa: E402


DEFAULT_STREAM = REPO_ROOT / "output" / "v18_human_metric_translation" / "stream_cache"
DEFAULT_KEYPOINT = REPO_ROOT / "output" / "v18_human_metric_translation" / "keypoint_cache"
DEFAULT_V10 = (
    REPO_ROOT
    / "output"
    / "v10_candidate_selection"
    / "oracle_gt_4source"
    / "oracle_candidate_selection_metrics.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "output" / "v18_human_metric_translation" / "projection_depth"
TORSO_IDS = np.asarray([0, 1, 2, 12, 16, 17], dtype=np.int64)
BODY_IDS = np.asarray([0, 1, 2, 4, 5, 7, 8, 12, 15, 16, 17, 18, 19, 20, 21], dtype=np.int64)
VARIANTS = (
    "gt_body_gt2d_gtK",
    "pred_pose_gt_shape_gt2d",
    "gt_pose_pred_shape_gt2d",
    "pred_body_gt2d",
    "pred_body_self2d",
    "gt_body_detector2d_torso",
    "gt_body_detector2d_full",
    "pred_pose_gt_shape_detector2d_torso",
    "pred_pose_gt_shape_detector2d_full",
    "gt_pose_pred_shape_detector2d_torso",
    "gt_pose_pred_shape_detector2d_full",
    "pred_body_detector2d_torso",
    "pred_body_detector2d_full",
    "median_shape_detector2d_full",
    "pred_mesh_mask_bbox",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream_dir", type=Path, default=DEFAULT_STREAM)
    parser.add_argument("--keypoint_dir", type=Path, default=DEFAULT_KEYPOINT)
    parser.add_argument("--v10_report", type=Path, default=DEFAULT_V10)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--keypoint_threshold", type=float, default=0.30)
    return parser.parse_args()


def load_manifest(root: Path, pattern: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(str(root / pattern))):
        rows.extend(json.loads(Path(path).read_text(encoding="utf-8"))["cases"])
    return rows


def load_cases(stream_dir: Path, keypoint_dir: Path, v10_report: Path) -> list[dict]:
    stream = load_manifest(stream_dir, "v18_stream_shard_*_of_*.json")
    keypoints = {row["case_name"]: row for row in load_manifest(keypoint_dir, "v18_keypoints_shard_*_of_*.json")}
    v10 = json.loads(v10_report.read_text(encoding="utf-8"))
    local = {row["case_name"]: row["paths"]["human3r_local_reset"] for row in v10["cases"]}
    if len(stream) != 180 or len(keypoints) != 180:
        raise RuntimeError(f"Expected 180 stream/keypoint cases, got {len(stream)}/{len(keypoints)}")
    for row in stream:
        row["keypoint_path"] = keypoints[row["case_name"]]["cache_path"]
        row["local_dir"] = local[row["case_name"]]
    return sorted(stream, key=lambda row: str(row["case_name"]))


def build_layer(device: torch.device, betas: int) -> SMPL_Layer:
    return SMPL_Layer(type="smplx", gender="neutral", num_betas=betas, kid=False, person_center="head").to(device).eval()


def body_from_params(
    layer: SMPL_Layer,
    pose: np.ndarray,
    shape: np.ndarray,
    world_scale: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    K = torch.eye(3, device=device, dtype=torch.float32)[None]
    with torch.no_grad():
        output = layer(
            torch.as_tensor(pose[None], dtype=torch.float32, device=device),
            torch.as_tensor(shape[None], dtype=torch.float32, device=device),
            torch.zeros((1, 3), dtype=torch.float32, device=device),
            None,
            None,
            K=K,
            expression=torch.zeros((1, 10), dtype=torch.float32, device=device),
        )
    joints = output["smpl_j3d"][0].detach().float().cpu().numpy().astype(np.float32)
    vertices = output["smpl_v3d"][0].detach().float().cpu().numpy().astype(np.float32)
    joints = (joints - joints[0]) * float(world_scale)
    vertices = (vertices - output["smpl_j3d"][0, 0].detach().float().cpu().numpy()) * float(world_scale)
    return joints.astype(np.float32), vertices.astype(np.float32)


def project(points: np.ndarray, translation: np.ndarray, K: np.ndarray) -> np.ndarray:
    camera = np.asarray(points, dtype=np.float64) + np.asarray(translation, dtype=np.float64)[None]
    z = np.maximum(camera[:, 2], 1e-5)
    return np.stack(
        [K[0, 0] * camera[:, 0] / z + K[0, 2], K[1, 1] * camera[:, 1] / z + K[1, 2]],
        axis=1,
    )


def estimate_root_translation(
    body: np.ndarray,
    observed: np.ndarray,
    confidence: np.ndarray,
    K: np.ndarray,
    initial: np.ndarray,
    ids: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, dict]:
    ids = np.asarray([index for index in ids if index < len(body) and index < len(observed)], dtype=np.int64)
    valid = np.isfinite(observed[ids]).all(axis=1) & np.isfinite(body[ids]).all(axis=1) & (confidence[ids] >= threshold)
    ids = ids[valid]
    if len(ids) < 4:
        return np.asarray(initial, dtype=np.float32), {
            "status": "too_few_joints",
            "valid_joints": int(len(ids)),
            "reprojection_error_px": float("inf"),
        }
    weights = np.sqrt(np.clip(confidence[ids], 0.05, 1.0))[:, None]

    def residual(translation: np.ndarray) -> np.ndarray:
        pixel = project(body[ids], translation, K)
        return ((pixel - observed[ids]) * weights).reshape(-1)

    x0 = np.asarray(initial, dtype=np.float64).copy()
    x0[2] = np.clip(x0[2], 0.3, 20.0)
    result = least_squares(
        residual,
        x0,
        bounds=(np.asarray([-15.0, -15.0, 0.20]), np.asarray([15.0, 15.0, 30.0])),
        loss="soft_l1",
        f_scale=8.0,
        max_nfev=100,
    )
    estimate = result.x.astype(np.float32)
    reprojection = np.linalg.norm(project(body[ids], estimate, K) - observed[ids], axis=1)
    return estimate, {
        "status": "ok" if result.success else "optimizer_failed",
        "valid_joints": int(len(ids)),
        "reprojection_error_px": float(np.median(reprojection)),
        "reprojection_p90_px": float(np.percentile(reprojection, 90)),
        "optimizer_cost": float(result.cost),
    }


def mask_bbox(local_dir: Path) -> tuple[np.ndarray | None, dict]:
    with np.load(local_dir / "smpl" / "000002.npz", allow_pickle=True) as smpl:
        mask = np.asarray(smpl["msk"], dtype=np.float32)
    if mask.ndim == 3:
        mask = mask[0]
    active = mask > 0.10
    yy, xx = np.where(active)
    if not len(xx):
        return None, {"status": "empty_mask", "mask_fraction": 0.0}
    bbox = np.asarray([xx.min(), yy.min(), xx.max(), yy.max()], dtype=np.float32)
    return bbox, {"status": "ok", "mask_fraction": float(active.mean())}


def estimate_bbox_translation(
    vertices: np.ndarray,
    observed_bbox: np.ndarray | None,
    K: np.ndarray,
    initial: np.ndarray,
) -> tuple[np.ndarray, dict]:
    if observed_bbox is None:
        return np.asarray(initial, dtype=np.float32), {"status": "empty_mask", "reprojection_error_px": float("inf")}
    sample = vertices[::10]
    width = max(float(observed_bbox[2] - observed_bbox[0]), 1.0)
    height = max(float(observed_bbox[3] - observed_bbox[1]), 1.0)

    def predicted_bbox(translation: np.ndarray) -> np.ndarray:
        pixel = project(sample, translation, K)
        low = np.quantile(pixel, 0.01, axis=0)
        high = np.quantile(pixel, 0.99, axis=0)
        return np.asarray([low[0], low[1], high[0], high[1]])

    def residual(translation: np.ndarray) -> np.ndarray:
        scale = np.asarray([width, height, width, height])
        return (predicted_bbox(translation) - observed_bbox) / scale

    x0 = np.asarray(initial, dtype=np.float64).copy()
    x0[2] = np.clip(x0[2], 0.3, 20.0)
    result = least_squares(
        residual,
        x0,
        bounds=(np.asarray([-15.0, -15.0, 0.20]), np.asarray([15.0, 15.0, 30.0])),
        loss="soft_l1",
        f_scale=0.10,
        max_nfev=80,
    )
    predicted = predicted_bbox(result.x)
    return result.x.astype(np.float32), {
        "status": "ok" if result.success else "optimizer_failed",
        "reprojection_error_px": float(np.mean(np.abs(predicted - observed_bbox))),
        "predicted_bbox": predicted.astype(float).tolist(),
        "observed_bbox": observed_bbox.astype(float).tolist(),
    }


def camera_translation_error(root: np.ndarray, gt_root: np.ndarray) -> dict:
    delta = np.asarray(root) - np.asarray(gt_root)
    return {
        "root_position_error_m": float(np.linalg.norm(delta)),
        "root_depth_error_m": float(abs(delta[2])),
        "root_transverse_error_m": float(np.linalg.norm(delta[:2])),
        "root_error_xyz_m": np.abs(delta).astype(float).tolist(),
    }


def torso_orientation_group(joints: np.ndarray) -> str:
    up = 0.5 * (joints[16] + joints[17]) - 0.5 * (joints[1] + joints[2])
    right = joints[17] - joints[16]
    up /= max(float(np.linalg.norm(up)), 1e-8)
    right /= max(float(np.linalg.norm(right)), 1e-8)
    forward = np.cross(right, up)
    forward /= max(float(np.linalg.norm(forward)), 1e-8)
    if abs(float(forward[2])) < 0.5:
        return "side"
    return "front" if float(forward[2]) < 0.0 else "back"


def run_case(case: dict, layer10: SMPL_Layer, layer11: SMPL_Layer, device: torch.device, threshold: float) -> dict:
    with np.load(case["cache_path"]) as stream, np.load(case["keypoint_path"]) as keypoint:
        K = stream["new_intrinsics"].astype(np.float32)
        predicted_joints = stream["new_joints_camera"].astype(np.float32)
        gt_joints = stream["new_gt_joints_camera"].astype(np.float32)
        predicted_root = predicted_joints[0]
        gt_root = gt_joints[0]
        predicted_body = predicted_joints - predicted_root
        gt_body = gt_joints - gt_root
        gt_2d = project(gt_body, gt_root, K).astype(np.float32)
        predicted_2d = project(predicted_body, predicted_root, K).astype(np.float32)
        full_confidence = np.ones(len(predicted_joints), dtype=np.float32)
        detector_2d = keypoint["new_keypoints"].astype(np.float32)
        detector_confidence = keypoint["new_confidence"].astype(np.float32)
        median_shape = np.median(stream["old_shape"].astype(np.float32), axis=0)
        pred_pose_gt_shape, _ = body_from_params(
            layer11,
            stream["new_rotvec"].astype(np.float32),
            stream["new_gt_shape"].astype(np.float32),
            float(stream["new_gt_world_scale"]),
            device,
        )
        gt_pose_pred_shape, _ = body_from_params(
            layer11,
            stream["new_gt_pose53_camera"].astype(np.float32),
            np.concatenate([stream["new_shape"].astype(np.float32), np.zeros(1, dtype=np.float32)]),
            1.0,
            device,
        )
        median_body, median_vertices = body_from_params(
            layer10,
            stream["new_rotvec"].astype(np.float32),
            median_shape,
            1.0,
            device,
        )
        _, predicted_vertices = body_from_params(
            layer10,
            stream["new_rotvec"].astype(np.float32),
            stream["new_shape"].astype(np.float32),
            1.0,
            device,
        )
        image_shape = stream["new_image"].shape[:2]

    configurations = {
        "gt_body_gt2d_gtK": (gt_body, gt_2d, full_confidence, BODY_IDS),
        "pred_pose_gt_shape_gt2d": (pred_pose_gt_shape, gt_2d, full_confidence, BODY_IDS),
        "gt_pose_pred_shape_gt2d": (gt_pose_pred_shape, gt_2d, full_confidence, BODY_IDS),
        "pred_body_gt2d": (predicted_body, gt_2d, full_confidence, BODY_IDS),
        "pred_body_self2d": (predicted_body, predicted_2d, full_confidence, BODY_IDS),
        "gt_body_detector2d_torso": (gt_body, detector_2d, detector_confidence, TORSO_IDS),
        "gt_body_detector2d_full": (gt_body, detector_2d, detector_confidence, BODY_IDS),
        "pred_pose_gt_shape_detector2d_torso": (
            pred_pose_gt_shape,
            detector_2d,
            detector_confidence,
            TORSO_IDS,
        ),
        "pred_pose_gt_shape_detector2d_full": (
            pred_pose_gt_shape,
            detector_2d,
            detector_confidence,
            BODY_IDS,
        ),
        "gt_pose_pred_shape_detector2d_torso": (
            gt_pose_pred_shape,
            detector_2d,
            detector_confidence,
            TORSO_IDS,
        ),
        "gt_pose_pred_shape_detector2d_full": (
            gt_pose_pred_shape,
            detector_2d,
            detector_confidence,
            BODY_IDS,
        ),
        "pred_body_detector2d_torso": (predicted_body, detector_2d, detector_confidence, TORSO_IDS),
        "pred_body_detector2d_full": (predicted_body, detector_2d, detector_confidence, BODY_IDS),
        "median_shape_detector2d_full": (median_body, detector_2d, detector_confidence, BODY_IDS),
    }
    variants = {}
    for name, (body, observed, confidence, ids) in configurations.items():
        root, diagnostics = estimate_root_translation(
            body, observed, confidence, K, predicted_root, ids, threshold
        )
        variants[name] = {
            "estimated_root_camera": root.astype(float).tolist(),
            **camera_translation_error(root, gt_root),
            **diagnostics,
        }
    observed_bbox, mask_diagnostics = mask_bbox(Path(case["local_dir"]))
    bbox_root, bbox_diagnostics = estimate_bbox_translation(
        predicted_vertices, observed_bbox, K, predicted_root
    )
    variants["pred_mesh_mask_bbox"] = {
        "estimated_root_camera": bbox_root.astype(float).tolist(),
        **camera_translation_error(bbox_root, gt_root),
        **mask_diagnostics,
        **bbox_diagnostics,
    }
    detector_box = np.asarray([np.nan, np.nan, np.nan, np.nan], dtype=np.float32)
    with np.load(case["keypoint_path"]) as keypoint:
        detector_box = keypoint["new_box"].astype(np.float32)
    height, width = image_shape
    box_height = max(float(detector_box[3] - detector_box[1]), 0.0)
    truncated = bool(
        detector_box[0] <= 2.0
        or detector_box[1] <= 2.0
        or detector_box[2] >= width - 3.0
        or detector_box[3] >= height - 3.0
    )
    with np.load(case["cache_path"]) as stream:
        shape_consistency = float(np.mean(np.std(stream["old_shape"].astype(np.float32), axis=0)))
    return {
        "case_name": case["case_name"],
        "source": case["source"],
        "groups": {
            "orientation": torso_orientation_group(predicted_joints),
            "body_fraction": box_height / max(float(height), 1.0),
            "visibility": "truncated" if truncated else "full_or_half",
            "valid_detector_joints": int(np.sum(detector_confidence >= threshold)),
        },
        "predicted_body_height_m": float(predicted_body[:, 1].max() - predicted_body[:, 1].min()),
        "gt_body_height_m": float(gt_body[:, 1].max() - gt_body[:, 1].min()),
        "shape_consistency_l2": shape_consistency,
        "variants": variants,
    }


def distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
    }


def aggregate(cases: list[dict]) -> dict:
    output = {}
    for variant in VARIANTS:
        rows = [case["variants"][variant] for case in cases]
        valid = [row for row in rows if row.get("status") == "ok"]
        selected = valid if valid else rows
        output[variant] = {
            "count": len(rows),
            "valid_rate": float(len(valid) / len(rows)),
            "root_position_m": distribution([float(row["root_position_error_m"]) for row in selected]),
            "root_depth_m": distribution([float(row["root_depth_error_m"]) for row in selected]),
            "root_transverse_m": distribution([float(row["root_transverse_error_m"]) for row in selected]),
            "reprojection_px": distribution(
                [float(row["reprojection_error_px"]) for row in valid if np.isfinite(row["reprojection_error_px"])]
            )
            if valid
            else None,
        }
    return output


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# V18 Human Projection Metric-Depth Probe",
        "",
        "| Variant | Valid | Root error | Depth error | Transverse | Reprojection |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        row = report["overall"][variant]
        reprojection = row["reprojection_px"]["mean"] if row["reprojection_px"] else float("nan")
        lines.append(
            f"| {variant} | {100.0 * row['valid_rate']:.1f}% | {row['root_position_m']['mean']:.3f} | "
            f"{row['root_depth_m']['mean']:.3f} | {row['root_transverse_m']['mean']:.3f} | {reprojection:.2f} |"
        )
    lines.extend(["", "## By Source", ""])
    for source, metrics in report["by_source"].items():
        deploy = metrics["median_shape_detector2d_full"]
        oracle = metrics["pred_body_gt2d"]
        lines.append(
            f"- **{source}**: predicted body + GT 2D depth `{oracle['root_depth_m']['mean']:.3f} m`; "
            f"median shape + detector 2D depth `{deploy['root_depth_m']['mean']:.3f} m`."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("V18 projection probe requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args.stream_dir, args.keypoint_dir, args.v10_report)
    device = torch.device(args.device)
    layer10 = build_layer(device, 10)
    layer11 = build_layer(device, 11)
    rows = []
    for index, case in enumerate(cases):
        rows.append(run_case(case, layer10, layer11, device, float(args.keypoint_threshold)))
        if (index + 1) % 20 == 0:
            print(f"V18 projection {index + 1}/{len(cases)}", flush=True)
    overall = aggregate(rows)
    by_source = {
        source: aggregate([row for row in rows if row["source"] == source])
        for source in sorted({row["source"] for row in rows})
    }
    report = {
        "experiment": "V18 Human Physical-Size and 2D Projection Depth Probe",
        "case_count": len(rows),
        "protocol": {
            "gt_depth_used": False,
            "gt_2d_use": "partial oracle only",
            "deployable_2d": "frozen torchvision Keypoint R-CNN",
            "predicted_intrinsics": "Human3R input/cropped intrinsics; no separate predicted-intrinsics head is available",
            "raw_tokens_used": False,
        },
        "overall": overall,
        "by_source": by_source,
        "cases": rows,
    }
    output = args.output_dir / "v18_projection_depth_probe.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    write_markdown(args.output_dir / "v18_projection_depth_probe_summary.md", report)
    print(f">> wrote {output}", flush=True)


if __name__ == "__main__":
    main()
