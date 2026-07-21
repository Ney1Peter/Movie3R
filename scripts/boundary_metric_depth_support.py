#!/usr/bin/env python3
"""Internal DA3 metric-depth support for active boundary experiments."""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
DA3_ROOT = REPO_ROOT.parent / "Movie3R-dataset" / "Depth-Anything-3"
if str(DA3_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(DA3_ROOT / "src"))

from depth_anything_3.api import DepthAnything3  # noqa: E402


DEFAULT_ROOT = REPO_ROOT / "output" / "v18_human_metric_translation"
DEFAULT_MODEL = DA3_ROOT / "checkpoints" / "DA3Metric-Large"
TORSO_IDS = np.asarray([0, 1, 2, 12, 16, 17], dtype=np.int64)
CANDIDATES = (
    "fixed_explicit",
    "v18_human_projection",
    "da3_pelvis_depth",
    "da3_torso_offset_depth",
    "da3_gt_pixel_depth_upper",
    "da3_pelvis_gt_motion_upper",
    "da3_motion_gt_camera_root_upper",
    "gt_depth_and_motion_torso_rotation_upper",
    "boundary_oracle",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream_dir", type=Path, default=DEFAULT_ROOT / "stream_cache")
    parser.add_argument("--keypoint_dir", type=Path, default=DEFAULT_ROOT / "keypoint_cache")
    parser.add_argument("--scene_dir", type=Path, default=DEFAULT_ROOT / "v16_bound20_scene")
    parser.add_argument(
        "--v18_report",
        type=Path,
        default=DEFAULT_ROOT / "final_candidates" / "v18_human_metric_translation_eval.json",
    )
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_ROOT / "da3_metric_depth")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--process_res", type=int, default=504)
    parser.add_argument("--keypoint_threshold", type=float, default=0.30)
    parser.add_argument("--sample_radius", type=int, default=3)
    parser.add_argument("--max_cases", type=int, default=0)
    return parser.parse_args()


def load_manifest(root: Path, pattern: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(str(root / pattern))):
        rows.extend(json.loads(Path(path).read_text(encoding="utf-8"))["cases"])
    return rows


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_cases(args: argparse.Namespace) -> list[dict]:
    stream = load_manifest(args.stream_dir, "v18_stream_shard_*_of_*.json")
    keypoints = {
        row["case_name"]: row
        for row in load_manifest(args.keypoint_dir, "v18_keypoints_shard_*_of_*.json")
    }
    scene = {
        row["case_name"]: row
        for row in load_manifest(args.scene_dir, "v16_candidates_shard_*_of_*.json")
    }
    v18 = json.loads(args.v18_report.read_text(encoding="utf-8"))
    v18_cases = {row["case_name"]: row for row in v18["cases"]}
    if len(stream) != 180 or len(keypoints) != 180 or len(scene) != 180 or len(v18_cases) != 180:
        raise RuntimeError(
            f"Expected 180 stream/keypoint/scene/V18 cases, got "
            f"{len(stream)}/{len(keypoints)}/{len(scene)}/{len(v18_cases)}"
        )
    cases = []
    for row in stream:
        case_name = row["case_name"]
        cases.append(
            {
                **row,
                "keypoint_path": keypoints[case_name]["cache_path"],
                "scene_case": scene[case_name],
                "v18_case": v18_cases[case_name],
            }
        )
    cases = sorted(cases, key=lambda row: str(row["case_name"]))
    return cases[: int(args.max_cases)] if int(args.max_cases) > 0 else cases


def processed_pixel(pixel: np.ndarray, K: np.ndarray, K_processed: np.ndarray) -> np.ndarray:
    x = (float(pixel[0]) - float(K[0, 2])) * float(K_processed[0, 0] / K[0, 0]) + float(
        K_processed[0, 2]
    )
    y = (float(pixel[1]) - float(K[1, 2])) * float(K_processed[1, 1] / K[1, 1]) + float(
        K_processed[1, 2]
    )
    return np.asarray([x, y], dtype=np.float32)


def sample_depth(depth: np.ndarray, pixel: np.ndarray, radius: int) -> float:
    x = int(round(float(pixel[0])))
    y = int(round(float(pixel[1])))
    x = int(np.clip(x, 0, depth.shape[1] - 1))
    y = int(np.clip(y, 0, depth.shape[0] - 1))
    patch = depth[
        max(0, y - radius) : min(depth.shape[0], y + radius + 1),
        max(0, x - radius) : min(depth.shape[1], x + radius + 1),
    ]
    valid = patch[np.isfinite(patch) & (patch > 0.05) & (patch < 100.0)]
    return float(np.median(valid)) if len(valid) else float("nan")


def gt_pixel(root: np.ndarray, K: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            K[0, 0] * root[0] / root[2] + K[0, 2],
            K[1, 1] * root[1] / root[2] + K[1, 2],
        ],
        dtype=np.float32,
    )


def root_from_pixel_depth(pixel: np.ndarray, depth: float, K: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            (float(pixel[0]) - float(K[0, 2])) * depth / float(K[0, 0]),
            (float(pixel[1]) - float(K[1, 2])) * depth / float(K[1, 1]),
            depth,
        ],
        dtype=np.float32,
    )


def camera_pose_from_human(rotation: np.ndarray, world_root: np.ndarray, camera_root: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = rotation
    pose[:3, 3] = world_root - rotation @ camera_root
    return pose


def boundary_from_camera_pose(camera_pose: np.ndarray, predicted_pose: np.ndarray) -> np.ndarray:
    return (camera_pose @ np.linalg.inv(predicted_pose)).astype(np.float32)


def transform_point(pose: np.ndarray, point: np.ndarray) -> np.ndarray:
    return (pose[:3, :3] @ point + pose[:3, 3]).astype(np.float32)


def rotation_error_deg(estimated: np.ndarray, target: np.ndarray) -> float:
    relative = estimated @ target.T
    return float(np.degrees(Rotation.from_matrix(relative.astype(np.float64)).magnitude()))


def evaluate(boundary: np.ndarray, predicted_pose: np.ndarray, target_pose: np.ndarray) -> dict:
    camera_pose = boundary @ predicted_pose
    delta_world = camera_pose[:3, 3] - target_pose[:3, 3]
    delta_target = target_pose[:3, :3].T @ delta_world
    return {
        "translation_m": float(np.linalg.norm(delta_world)),
        "rotation_deg": rotation_error_deg(camera_pose[:3, :3], target_pose[:3, :3]),
        "viewing_direction_m": float(abs(delta_target[2])),
        "transverse_m": float(np.linalg.norm(delta_target[:2])),
        "transform": boundary.astype(float).tolist(),
    }


def detector_root_pixel(keypoints: np.ndarray, confidence: np.ndarray, threshold: float) -> np.ndarray:
    if confidence[0] >= threshold and np.isfinite(keypoints[0]).all():
        return keypoints[0].astype(np.float32)
    valid_hips = [index for index in (1, 2) if confidence[index] >= threshold and np.isfinite(keypoints[index]).all()]
    if valid_hips:
        return np.mean(keypoints[valid_hips], axis=0).astype(np.float32)
    valid = np.flatnonzero((confidence >= threshold) & np.isfinite(keypoints).all(axis=1))
    if len(valid):
        return np.mean(keypoints[valid], axis=0).astype(np.float32)
    return np.asarray([np.nan, np.nan], dtype=np.float32)


def estimate_frame_roots(
    metric_depth: np.ndarray,
    K: np.ndarray,
    K_processed: np.ndarray,
    keypoints: np.ndarray,
    confidence: np.ndarray,
    predicted_joints: np.ndarray,
    threshold: float,
    radius: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    pixel = detector_root_pixel(keypoints, confidence, threshold)
    if not np.isfinite(pixel).all():
        fallback = predicted_joints[0].astype(np.float32)
        return fallback, fallback, {"status": "no_detector_root"}
    pixel_processed = processed_pixel(pixel, K, K_processed)
    pelvis_depth = sample_depth(metric_depth, pixel_processed, radius)
    if not np.isfinite(pelvis_depth):
        pelvis_depth = float(predicted_joints[0, 2])
    pelvis_root = root_from_pixel_depth(pixel, pelvis_depth, K)

    body = predicted_joints - predicted_joints[0]
    root_depths = []
    sampled_depths = []
    for joint_id in TORSO_IDS:
        if joint_id >= len(keypoints) or confidence[joint_id] < threshold:
            continue
        if not np.isfinite(keypoints[joint_id]).all():
            continue
        joint_pixel = processed_pixel(keypoints[joint_id], K, K_processed)
        joint_depth = sample_depth(metric_depth, joint_pixel, radius)
        if not np.isfinite(joint_depth):
            continue
        sampled_depths.append(joint_depth)
        root_depths.append(joint_depth - float(body[joint_id, 2]))
    torso_depth = float(np.median(root_depths)) if len(root_depths) >= 3 else pelvis_depth
    torso_root = root_from_pixel_depth(pixel, torso_depth, K)
    return pelvis_root, torso_root, {
        "status": "ok",
        "pelvis_depth_m": pelvis_depth,
        "torso_root_depth_m": torso_depth,
        "valid_torso_samples": len(root_depths),
        "torso_surface_depth_median": float(np.median(sampled_depths)) if sampled_depths else float("nan"),
    }


def scene_transform(case: dict) -> np.ndarray:
    row = case["scene_case"]["fixed_candidates"]["fixed_torso_motion_1f_resolve_t"]
    return np.asarray(row["transform"], dtype=np.float32)


def metric_inference(
    model: DepthAnything3,
    images: list[np.ndarray],
    intrinsics: np.ndarray,
    process_res: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    started = time.perf_counter()
    with torch.no_grad():
        prediction = model.inference(images, intrinsics=intrinsics, process_res=process_res)
    elapsed = time.perf_counter() - started
    _, _, processed_intrinsics = model.input_processor(
        images,
        intrinsics=intrinsics,
        process_res=process_res,
        process_res_method="upper_bound_resize",
        sequential=True,
    )
    K_processed = processed_intrinsics.numpy().astype(np.float32)
    raw_depth = prediction.depth.astype(np.float32)
    focal = 0.5 * (K_processed[:, 0, 0] + K_processed[:, 1, 1])
    metric_depth = raw_depth * (focal[:, None, None] / 300.0)
    return metric_depth.astype(np.float32), K_processed, elapsed


def run_case(case: dict, model: DepthAnything3, args: argparse.Namespace) -> dict:
    with np.load(resolve(case["cache_path"])) as stream, np.load(resolve(case["keypoint_path"])) as keypoint:
        images = [*stream["old_images"], stream["new_image"]]
        K = np.concatenate([stream["old_intrinsics"], stream["new_intrinsics"][None]], axis=0).astype(np.float32)
        poses = np.concatenate([stream["old_pose"], stream["new_pose"][None]], axis=0).astype(np.float32)
        joints = np.concatenate([stream["old_joints_camera"], stream["new_joints_camera"][None]], axis=0).astype(np.float32)
        gt_camera = np.concatenate(
            [stream["old_gt_joints_camera"], stream["new_gt_joints_camera"][None]], axis=0
        ).astype(np.float32)
        target_world_root = stream["new_gt_joints_target_world"][0].astype(np.float32)
        new_pose = stream["new_pose"].astype(np.float32)
        target_pose = stream["target_pose"].astype(np.float32)
        target_boundary = stream["gt_boundary"].astype(np.float32)
        fixed = stream["fixed_transform"].astype(np.float32)
        keypoints = np.concatenate([keypoint["old_keypoints"], keypoint["new_keypoints"][None]], axis=0)
        confidence = np.concatenate([keypoint["old_confidence"], keypoint["new_confidence"][None]], axis=0)

    metric_depth, K_processed, elapsed = metric_inference(model, images, K, int(args.process_res))
    pelvis_roots = []
    torso_roots = []
    root_diagnostics = []
    gt_pixel_depths = []
    for index in range(len(images)):
        pelvis, torso, diagnostics = estimate_frame_roots(
            metric_depth[index],
            K[index],
            K_processed[index],
            keypoints[index],
            confidence[index],
            joints[index],
            float(args.keypoint_threshold),
            int(args.sample_radius),
        )
        gt_uv = gt_pixel(gt_camera[index, 0], K[index])
        gt_uv_processed = processed_pixel(gt_uv, K[index], K_processed[index])
        gt_pixel_depths.append(sample_depth(metric_depth[index], gt_uv_processed, int(args.sample_radius)))
        pelvis_roots.append(pelvis)
        torso_roots.append(torso)
        root_diagnostics.append(diagnostics)
    pelvis_roots = np.stack(pelvis_roots).astype(np.float32)
    torso_roots = np.stack(torso_roots).astype(np.float32)
    gt_pixel_depths = np.asarray(gt_pixel_depths, dtype=np.float32)

    scene = scene_transform(case)
    scene_pose = scene @ new_pose
    rotation = scene_pose[:3, :3]
    old_pelvis_world = np.stack([transform_point(pose, root) for pose, root in zip(poses[:-1], pelvis_roots[:-1])])
    old_torso_world = np.stack([transform_point(pose, root) for pose, root in zip(poses[:-1], torso_roots[:-1])])
    pelvis_world = old_pelvis_world[-1]
    torso_world = old_torso_world[-1]

    gt_pixel_root = root_from_pixel_depth(
        detector_root_pixel(keypoints[-1], confidence[-1], float(args.keypoint_threshold)),
        float(gt_pixel_depths[-1]),
        K[-1],
    )
    gt_camera_root = gt_camera[-1, 0]
    candidates = {
        "fixed_explicit": fixed,
        "v18_human_projection": np.asarray(
            case["v18_case"]["candidates"]["human_no_calibration"]["transform"], dtype=np.float32
        ),
        "da3_pelvis_depth": boundary_from_camera_pose(
            camera_pose_from_human(rotation, pelvis_world, pelvis_roots[-1]), new_pose
        ),
        "da3_torso_offset_depth": boundary_from_camera_pose(
            camera_pose_from_human(rotation, torso_world, torso_roots[-1]), new_pose
        ),
        "da3_gt_pixel_depth_upper": boundary_from_camera_pose(
            camera_pose_from_human(rotation, pelvis_world, gt_pixel_root), new_pose
        ),
        "da3_pelvis_gt_motion_upper": boundary_from_camera_pose(
            camera_pose_from_human(rotation, target_world_root, pelvis_roots[-1]), new_pose
        ),
        "da3_motion_gt_camera_root_upper": boundary_from_camera_pose(
            camera_pose_from_human(rotation, pelvis_world, gt_camera_root), new_pose
        ),
        "gt_depth_and_motion_torso_rotation_upper": boundary_from_camera_pose(
            camera_pose_from_human(rotation, target_world_root, gt_camera_root), new_pose
        ),
        "boundary_oracle": target_boundary,
    }
    evaluated = {name: evaluate(transform, new_pose, target_pose) for name, transform in candidates.items()}
    new_gt_depth = float(gt_camera_root[2])
    return {
        "case_name": case["case_name"],
        "source": case["source"],
        "inference_seconds_6frames": elapsed,
        "depth": {
            "gt_root_m": new_gt_depth,
            "human3r_root_m": float(joints[-1, 0, 2]),
            "da3_gt_pixel_m": float(gt_pixel_depths[-1]),
            "da3_pelvis_m": float(pelvis_roots[-1, 2]),
            "da3_torso_offset_m": float(torso_roots[-1, 2]),
            "errors_m": {
                "human3r": float(abs(joints[-1, 0, 2] - new_gt_depth)),
                "da3_gt_pixel": float(abs(gt_pixel_depths[-1] - new_gt_depth)),
                "da3_pelvis": float(abs(pelvis_roots[-1, 2] - new_gt_depth)),
                "da3_torso_offset": float(abs(torso_roots[-1, 2] - new_gt_depth)),
            },
        },
        "motion": {
            "da3_pelvis_last_error_m": float(np.linalg.norm(pelvis_world - target_world_root)),
            "da3_torso_last_error_m": float(np.linalg.norm(torso_world - target_world_root)),
        },
        "root_diagnostics": root_diagnostics[-1],
        "candidates": evaluated,
    }


def distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
    }


def aggregate_candidate(cases: list[dict], name: str) -> dict:
    rows = [case["candidates"][name] for case in cases]
    fixed = [case["candidates"]["fixed_explicit"] for case in cases]
    return {
        "translation_m": distribution([row["translation_m"] for row in rows]),
        "rotation_deg": distribution([row["rotation_deg"] for row in rows]),
        "viewing_direction_m": distribution([row["viewing_direction_m"] for row in rows]),
        "transverse_m": distribution([row["transverse_m"] for row in rows]),
        "translation_catastrophic_rate": float(np.mean([row["translation_m"] > 1.0 for row in rows])),
        "harmful_translation_rate_vs_fixed": float(
            np.mean([row["translation_m"] > base["translation_m"] + 0.05 for row, base in zip(rows, fixed)])
        ),
    }


def aggregate(cases: list[dict]) -> dict:
    return {name: aggregate_candidate(cases, name) for name in CANDIDATES}


def aggregate_depth(cases: list[dict]) -> dict:
    keys = ("human3r", "da3_gt_pixel", "da3_pelvis", "da3_torso_offset")
    return {key: distribution([case["depth"]["errors_m"][key] for case in cases]) for key in keys}


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# DA3 Metric-Depth Support Diagnostic",
        "",
        "## Depth",
        "",
        "| Method | Mean | Median | P90 | P95 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in report["depth"]["overall"].items():
        lines.append(
            f"| {name} | {row['mean']:.3f} | {row['median']:.3f} | {row['p90']:.3f} | {row['p95']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Translation Candidates",
            "",
            "| Candidate | T mean | T median | T P90 | View | T-cat | Harmful T |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in CANDIDATES:
        row = report["overall"][name]
        lines.append(
            f"| {name} | {row['translation_m']['mean']:.3f} | {row['translation_m']['median']:.3f} | "
            f"{row['translation_m']['p90']:.3f} | {row['viewing_direction_m']['mean']:.3f} | "
            f"{100.0 * row['translation_catastrophic_rate']:.1f}% | "
            f"{100.0 * row['harmful_translation_rate_vs_fixed']:.1f}% |"
        )
    lines.extend(["", "## By Source", ""])
    for source, candidates in report["by_source"].items():
        fixed = candidates["fixed_explicit"]["translation_m"]["mean"]
        human = candidates["v18_human_projection"]["translation_m"]["mean"]
        da3 = candidates["da3_pelvis_depth"]["translation_m"]["mean"]
        depth = report["depth"]["by_source"][source]["da3_pelvis"]["mean"]
        lines.append(
            f"- **{source}**: depth error `{depth:.3f} m`; Fixed `{fixed:.3f} m`; "
            f"Human projection `{human:.3f} m`; DA3 `{da3:.3f} m`."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("DA3 metric-depth probe requires CUDA")
    if not (args.model_path / "config.json").exists() or not (args.model_path / "model.safetensors").exists():
        raise FileNotFoundError(f"Incomplete DA3 metric checkpoint: {args.model_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args)
    model = DepthAnything3.from_pretrained(str(args.model_path)).to(args.device).eval()
    rows = []
    for index, case in enumerate(cases):
        rows.append(run_case(case, model, args))
        if (index + 1) % 10 == 0 or index + 1 == len(cases):
            print(f"DA3 metric support {index + 1}/{len(cases)}", flush=True)
    overall = aggregate(rows)
    by_source = {
        source: aggregate([row for row in rows if row["source"] == source])
        for source in sorted({row["source"] for row in rows})
    }
    depth_by_source = {
        source: aggregate_depth([row for row in rows if row["source"] == source])
        for source in sorted({row["source"] for row in rows})
    }
    report = {
        "experiment": "External DA3 metric-depth support diagnostic",
        "case_count": len(rows),
        "protocol": {
            "checkpoint": str(args.model_path),
            "model": "DA3Metric-Large",
            "metric_conversion": "depth_meter = raw_depth * processed_focal / 300",
            "history_frames": 5,
            "post_cut_frames": 1,
            "rotation": "torso-motion 20-degree bound",
            "raw_tokens_used": False,
            "gt_depth_used": False,
            "gt_scene_mesh_used": False,
        },
        "latency_seconds_6frames": distribution([row["inference_seconds_6frames"] for row in rows]),
        "depth": {"overall": aggregate_depth(rows), "by_source": depth_by_source},
        "overall": overall,
        "by_source": by_source,
        "cases": rows,
    }
    output = args.output_dir / "v18_da3_metric_depth_probe.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    write_markdown(args.output_dir / "v18_da3_metric_depth_probe_summary.md", report)
    print(f">> wrote {output}", flush=True)


if __name__ == "__main__":
    main()
