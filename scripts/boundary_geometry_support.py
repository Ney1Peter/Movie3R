#!/usr/bin/env python3
"""Internal explicit-geometry utilities for boundary alignment probes."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from boundary_metric_depth_support import (  # noqa: E402
    DEFAULT_MODEL,
    DepthAnything3,
    boundary_from_camera_pose,
    camera_pose_from_human,
    estimate_frame_roots,
    evaluate,
    load_cases,
    metric_inference,
    resolve,
)


DEFAULT_ROOT = REPO_ROOT / "output" / "v19_da3_explicit_geometry_correction"
METHODS = (
    "fixed_raw_geometry",
    "da3_camera_raw_geometry",
    "fixed_da3_geometry",
    "da3_corrected_human_translation",
    "da3_corrected_scene_residual",
    "oracle_raw_geometry",
    "oracle_da3_geometry",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stream_dir",
        type=Path,
        default=REPO_ROOT / "output" / "v18_human_metric_translation" / "stream_cache",
    )
    parser.add_argument(
        "--keypoint_dir",
        type=Path,
        default=REPO_ROOT / "output" / "v18_human_metric_translation" / "keypoint_cache",
    )
    parser.add_argument(
        "--scene_dir",
        type=Path,
        default=REPO_ROOT / "output" / "v18_human_metric_translation" / "v16_bound20_scene",
    )
    parser.add_argument(
        "--v18_report",
        type=Path,
        default=REPO_ROOT
        / "output"
        / "v18_human_metric_translation"
        / "final_candidates"
        / "v18_human_metric_translation_eval.json",
    )
    parser.add_argument(
        "--v10_report",
        type=Path,
        default=REPO_ROOT
        / "output"
        / "v10_candidate_selection"
        / "oracle_gt_4source"
        / "oracle_candidate_selection_metrics.json",
    )
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--process_res", type=int, default=504)
    parser.add_argument("--keypoint_threshold", type=float, default=0.30)
    parser.add_argument("--sample_radius", type=int, default=3)
    parser.add_argument("--point_samples", type=int, default=4000)
    parser.add_argument("--raw_confidence_threshold", type=float, default=1.5)
    parser.add_argument("--mask_dilate", type=int, default=11)
    parser.add_argument("--scene_iters", type=int, default=8)
    parser.add_argument("--scene_max_distance", type=float, default=0.60)
    parser.add_argument("--scene_min_distance", type=float, default=0.12)
    parser.add_argument("--scene_max_correction", type=float, default=0.50)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260720)
    return parser.parse_args()


def load_v10_cases(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["case_name"]): row for row in payload["cases"]}


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def camera_points(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    return np.stack(
        [
            (xx - intrinsics[0, 2]) * depth / intrinsics[0, 0],
            (yy - intrinsics[1, 2]) * depth / intrinsics[1, 1],
            depth,
        ],
        axis=-1,
    )


def sample_cloud(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    pose: np.ndarray,
    human_mask: np.ndarray,
    confidence: np.ndarray | None,
    confidence_threshold: float,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if human_mask.shape != depth.shape:
        human_mask = cv2.resize(
            human_mask.astype(np.uint8), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST
        )
    valid = (
        np.isfinite(depth)
        & (depth > 0.10)
        & (depth < 30.0)
        & (human_mask == 0)
    )
    if confidence is not None:
        if confidence.shape != depth.shape:
            confidence = cv2.resize(
                confidence.astype(np.float32),
                (depth.shape[1], depth.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        valid &= np.isfinite(confidence) & (confidence > confidence_threshold)
    ids = np.flatnonzero(valid.reshape(-1))
    if len(ids) > count:
        ids = rng.choice(ids, size=count, replace=False)
    points = camera_points(depth, intrinsics).reshape(-1, 3)[ids]
    return transform_points(pose, points).astype(np.float32)


def load_raw_geometry(
    local_dir: Path,
    point_samples: int,
    confidence_threshold: float,
    mask_dilate: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    clouds = []
    roots = []
    poses = []
    images = []
    kernel = np.ones((mask_dilate, mask_dilate), dtype=np.uint8)
    for frame in (1, 2):
        with np.load(local_dir / "camera" / f"{frame:06d}.npz") as camera:
            pose = np.asarray(camera["pose"], dtype=np.float32)
            intrinsics = np.asarray(camera["intrinsics"], dtype=np.float32)
        with np.load(local_dir / "smpl" / f"{frame:06d}.npz", allow_pickle=True) as smpl:
            root = np.asarray(smpl["transl"][0], dtype=np.float32)
            mask = np.asarray(smpl["msk"][0] > 0.10, dtype=np.uint8)
        if mask_dilate > 1:
            mask = cv2.dilate(mask, kernel, iterations=1)
        depth = np.load(local_dir / "depth" / f"{frame:06d}.npy").astype(np.float32)
        confidence = np.load(local_dir / "conf" / f"{frame:06d}.npy").astype(np.float32)
        image = cv2.cvtColor(
            cv2.imread(str(local_dir / "color" / f"{frame:06d}.png"), cv2.IMREAD_COLOR),
            cv2.COLOR_BGR2RGB,
        )
        clouds.append(
            sample_cloud(
                depth,
                intrinsics,
                pose,
                mask,
                confidence,
                confidence_threshold,
                point_samples,
                rng,
            )
        )
        roots.append(pose[:3, :3] @ root + pose[:3, 3])
        poses.append(pose)
        images.append(image)
    return clouds[0], clouds[1], roots[0], roots[1], {"poses": poses, "images": images}


def scene_alignment_metrics(transform: np.ndarray, source: np.ndarray, target: np.ndarray) -> dict:
    transformed = transform_points(transform, source)
    source_tree = cKDTree(transformed)
    target_tree = cKDTree(target)
    source_to_target, _ = target_tree.query(transformed, k=1, workers=-1)
    target_to_source, _ = source_tree.query(target, k=1, workers=-1)
    combined = np.concatenate([source_to_target, target_to_source])
    finite = combined[np.isfinite(combined)]
    if not len(finite):
        return {
            "median_m": float("nan"),
            "trimmed_mean_m": float("nan"),
            "p90_m": float("nan"),
            "overlap_020": 0.0,
            "overlap_050": 0.0,
        }
    trim = finite[finite <= np.quantile(finite, 0.50)]
    return {
        "median_m": float(np.median(finite)),
        "trimmed_mean_m": float(np.mean(trim)),
        "p90_m": float(np.quantile(finite, 0.90)),
        "overlap_020": float(np.mean(finite < 0.20)),
        "overlap_050": float(np.mean(finite < 0.50)),
    }


def solve_scene_translation(
    initial: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict]:
    transform = np.asarray(initial, dtype=np.float32).copy()
    initial_translation = transform[:3, 3].copy()
    tree = cKDTree(target)
    history = []
    for iteration in range(int(args.scene_iters)):
        transformed = transform_points(transform, source)
        distance, nearest = tree.query(transformed, k=1, workers=-1)
        alpha = iteration / max(int(args.scene_iters) - 1, 1)
        threshold = (1.0 - alpha) * float(args.scene_max_distance) + alpha * float(
            args.scene_min_distance
        )
        valid = np.flatnonzero(np.isfinite(distance) & (distance < threshold))
        if len(valid) < 32:
            break
        trim = float(np.quantile(distance[valid], 0.70))
        valid = valid[distance[valid] <= trim]
        if len(valid) < 32:
            break
        residual = target[nearest[valid]] - transformed[valid]
        weights = 1.0 / np.maximum(distance[valid], 0.01)
        delta = np.average(residual, axis=0, weights=weights).astype(np.float32)
        proposed = transform[:3, 3] + delta
        total = proposed - initial_translation
        norm = float(np.linalg.norm(total))
        if norm > float(args.scene_max_correction):
            proposed = initial_translation + total * (float(args.scene_max_correction) / norm)
        transform[:3, 3] = proposed
        history.append(
            {
                "iteration": iteration,
                "pairs": int(len(valid)),
                "median_distance_m": float(np.median(distance[valid])),
                "delta_m": float(np.linalg.norm(delta)),
            }
        )
    return transform, {
        "iterations": history,
        "translation_correction_m": float(np.linalg.norm(transform[:3, 3] - initial_translation)),
    }


def human_metrics(
    transform: np.ndarray,
    old_root_camera: np.ndarray,
    new_root_camera: np.ndarray,
    old_pose: np.ndarray,
    new_pose: np.ndarray,
    gt_old_world: np.ndarray,
    gt_new_world: np.ndarray,
) -> dict:
    old_world = transform_points(old_pose, old_root_camera[None])[0]
    new_local = transform_points(new_pose, new_root_camera[None])[0]
    new_world = transform_points(transform, new_local[None])[0]
    displacement = new_world - old_world
    gt_displacement = gt_new_world - gt_old_world
    return {
        "root_jump_m": float(np.linalg.norm(displacement)),
        "root_motion_error_m": float(np.linalg.norm(displacement - gt_displacement)),
        "root_displacement_xyz_m": displacement.astype(float).tolist(),
    }


def distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
    }


def aggregate(rows: list[dict]) -> dict:
    output = {}
    for method in METHODS:
        method_rows = [row["methods"][method] for row in rows]
        output[method] = {
            "camera_translation_m": distribution([row["camera"]["translation_m"] for row in method_rows]),
            "camera_rotation_deg": distribution([row["camera"]["rotation_deg"] for row in method_rows]),
            "human_root_jump_m": distribution([row["human"]["root_jump_m"] for row in method_rows]),
            "human_root_motion_error_m": distribution(
                [row["human"]["root_motion_error_m"] for row in method_rows]
            ),
            "scene_trimmed_mean_m": distribution(
                [row["scene"]["trimmed_mean_m"] for row in method_rows]
            ),
            "scene_median_m": distribution([row["scene"]["median_m"] for row in method_rows]),
            "scene_overlap_020": distribution([row["scene"]["overlap_020"] for row in method_rows]),
        }
    return output


def run_case(
    case: dict,
    v10_case: dict,
    model: DepthAnything3,
    args: argparse.Namespace,
    index: int,
) -> dict:
    with np.load(resolve(case["cache_path"])) as stream, np.load(resolve(case["keypoint_path"])) as keypoint:
        images = [*stream["old_images"], stream["new_image"]]
        intrinsics = np.concatenate(
            [stream["old_intrinsics"], stream["new_intrinsics"][None]], axis=0
        ).astype(np.float32)
        poses = np.concatenate([stream["old_pose"], stream["new_pose"][None]], axis=0).astype(
            np.float32
        )
        joints = np.concatenate(
            [stream["old_joints_camera"], stream["new_joints_camera"][None]], axis=0
        ).astype(np.float32)
        translations = np.concatenate(
            [stream["old_transl"], stream["new_transl"][None]], axis=0
        ).astype(np.float32)
        fixed = stream["fixed_transform"].astype(np.float32)
        target_boundary = stream["gt_boundary"].astype(np.float32)
        target_pose = stream["target_pose"].astype(np.float32)
        new_pose = stream["new_pose"].astype(np.float32)
        gt_old_world = stream["old_gt_joints_target_world"][-1, 0].astype(np.float32)
        gt_new_world = stream["new_gt_joints_target_world"][0].astype(np.float32)
        keypoints = np.concatenate(
            [keypoint["old_keypoints"], keypoint["new_keypoints"][None]], axis=0
        )
        confidence = np.concatenate(
            [keypoint["old_confidence"], keypoint["new_confidence"][None]], axis=0
        )

    metric_depth, processed_intrinsics, elapsed = metric_inference(
        model, images, intrinsics, int(args.process_res)
    )
    da3_roots = []
    for frame in range(len(images)):
        pelvis, _, _ = estimate_frame_roots(
            metric_depth[frame],
            intrinsics[frame],
            processed_intrinsics[frame],
            keypoints[frame],
            confidence[frame],
            joints[frame],
            float(args.keypoint_threshold),
            int(args.sample_radius),
        )
        da3_roots.append(pelvis)
    da3_roots = np.stack(da3_roots).astype(np.float32)

    local_dir = Path(v10_case["paths"]["human3r_local_reset"])
    rng = np.random.default_rng(int(args.seed) + index)
    raw_old_cloud, raw_new_cloud, raw_old_transl_world, raw_new_transl_local, raw_debug = load_raw_geometry(
        local_dir,
        int(args.point_samples),
        float(args.raw_confidence_threshold),
        int(args.mask_dilate),
        rng,
    )
    old_mask = np.load(local_dir / "smpl" / "000001.npz", allow_pickle=True)["msk"][0] > 0.10
    new_mask = np.load(local_dir / "smpl" / "000002.npz", allow_pickle=True)["msk"][0] > 0.10
    kernel = np.ones((int(args.mask_dilate), int(args.mask_dilate)), dtype=np.uint8)
    old_mask = cv2.dilate(old_mask.astype(np.uint8), kernel, iterations=1)
    new_mask = cv2.dilate(new_mask.astype(np.uint8), kernel, iterations=1)
    da3_old_cloud = sample_cloud(
        metric_depth[-2],
        processed_intrinsics[-2],
        poses[-2],
        old_mask,
        None,
        0.0,
        int(args.point_samples),
        rng,
    )
    da3_new_cloud = sample_cloud(
        metric_depth[-1],
        processed_intrinsics[-1],
        poses[-1],
        new_mask,
        None,
        0.0,
        int(args.point_samples),
        rng,
    )

    scene_transform = np.asarray(
        case["scene_case"]["fixed_candidates"]["fixed_torso_motion_1f_resolve_t"]["transform"],
        dtype=np.float32,
    )
    scene_camera_pose = scene_transform @ new_pose
    old_da3_world = transform_points(poses[-2], da3_roots[-2][None])[0]
    da3_camera_pose = camera_pose_from_human(
        scene_camera_pose[:3, :3], old_da3_world, da3_roots[-1]
    )
    da3_boundary = boundary_from_camera_pose(da3_camera_pose, new_pose)
    scene_boundary, scene_debug = solve_scene_translation(
        da3_boundary, da3_new_cloud, da3_old_cloud, args
    )

    corrected_translations = translations + (da3_roots - joints[:, 0])
    method_specs = {
        "fixed_raw_geometry": (fixed, "raw"),
        "da3_camera_raw_geometry": (da3_boundary, "raw"),
        "fixed_da3_geometry": (fixed, "da3"),
        "da3_corrected_human_translation": (da3_boundary, "da3"),
        "da3_corrected_scene_residual": (scene_boundary, "da3"),
        "oracle_raw_geometry": (target_boundary, "raw"),
        "oracle_da3_geometry": (target_boundary, "da3"),
    }
    methods = {}
    for method, (transform, geometry) in method_specs.items():
        if geometry == "raw":
            old_cloud, new_cloud = raw_old_cloud, raw_new_cloud
            old_root, new_root = translations[-2], translations[-1]
        else:
            old_cloud, new_cloud = da3_old_cloud, da3_new_cloud
            old_root, new_root = corrected_translations[-2], corrected_translations[-1]
        methods[method] = {
            "geometry": geometry,
            "transform": transform.astype(float).tolist(),
            "camera": evaluate(transform, new_pose, target_pose),
            "human": human_metrics(
                transform,
                old_root,
                new_root,
                poses[-2],
                poses[-1],
                gt_old_world,
                gt_new_world,
            ),
            "scene": scene_alignment_metrics(transform, new_cloud, old_cloud),
        }
    methods["da3_corrected_scene_residual"]["scene_solver"] = scene_debug

    old_image = raw_debug["images"][0]
    new_image = raw_debug["images"][1]
    return {
        "case_name": case["case_name"],
        "source": case["source"],
        "inference_seconds_6frames": elapsed,
        "cache_consistency": {
            "old_rgb_mean_abs": float(np.mean(np.abs(old_image.astype(np.float32) - images[-2].astype(np.float32)))),
            "new_rgb_mean_abs": float(np.mean(np.abs(new_image.astype(np.float32) - images[-1].astype(np.float32)))),
            "raw_old_pose_max_abs": float(np.max(np.abs(raw_debug["poses"][0] - poses[-2]))),
            "raw_new_pose_max_abs": float(np.max(np.abs(raw_debug["poses"][1] - poses[-1]))),
        },
        "da3_root_depth": {
            "old_m": float(da3_roots[-2, 2]),
            "new_m": float(da3_roots[-1, 2]),
            "old_human3r_m": float(joints[-2, 0, 2]),
            "new_human3r_m": float(joints[-1, 0, 2]),
        },
        "methods": methods,
    }


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# DA3 Explicit Geometry Support",
        "",
        "| Method | Camera T | Camera R | Human jump | Human motion err | Scene trim | Scene overlap<20cm |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = report["overall"][method]
        lines.append(
            f"| {method} | {row['camera_translation_m']['mean']:.3f} | "
            f"{row['camera_rotation_deg']['mean']:.2f} | {row['human_root_jump_m']['mean']:.3f} | "
            f"{row['human_root_motion_error_m']['mean']:.3f} | "
            f"{row['scene_trimmed_mean_m']['mean']:.3f} | {row['scene_overlap_020']['mean']:.3f} |"
        )
    lines.extend(["", "## By Source", ""])
    for source, methods in report["by_source"].items():
        fixed = methods["fixed_raw_geometry"]
        corrected = methods["da3_corrected_human_translation"]
        residual = methods["da3_corrected_scene_residual"]
        lines.append(
            f"- **{source}**: Fixed camera/root/scene "
            f"`{fixed['camera_translation_m']['mean']:.3f}/"
            f"{fixed['human_root_jump_m']['mean']:.3f}/"
            f"{fixed['scene_trimmed_mean_m']['mean']:.3f}`; "
            f"DA3 corrected `{corrected['camera_translation_m']['mean']:.3f}/"
            f"{corrected['human_root_jump_m']['mean']:.3f}/"
            f"{corrected['scene_trimmed_mean_m']['mean']:.3f}`; "
            f"scene residual `{residual['camera_translation_m']['mean']:.3f}/"
            f"{residual['human_root_jump_m']['mean']:.3f}/"
            f"{residual['scene_trimmed_mean_m']['mean']:.3f}`."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("DA3 geometry support requires CUDA")
    if not (args.model_path / "config.json").exists() or not (
        args.model_path / "model.safetensors"
    ).exists():
        raise FileNotFoundError(f"Incomplete DA3 checkpoint: {args.model_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args)
    v10_cases = load_v10_cases(args.v10_report)
    model = DepthAnything3.from_pretrained(str(args.model_path)).to(args.device).eval()
    rows = []
    for index, case in enumerate(cases):
        rows.append(run_case(case, v10_cases[str(case["case_name"])], model, args, index))
        print(f"DA3 geometry support {index + 1}/{len(cases)}", flush=True)
    overall = aggregate(rows)
    by_source = {
        source: aggregate([row for row in rows if row["source"] == source])
        for source in sorted({row["source"] for row in rows})
    }
    report = {
        "experiment": "DA3 explicit geometry correction before Boundary SE(3)",
        "case_count": len(rows),
        "protocol": {
            "checkpoint": str(args.model_path),
            "history_frames": 5,
            "post_cut_frames": 1,
            "rotation": "torso-motion 20-degree bound",
            "geometry_correction": (
                "DA3 dense metric depth replaces boundary pointmaps; SMPL-X translation is shifted "
                "by DA3 pelvis root minus Human3R pelvis root in camera coordinates"
            ),
            "final_alignment": "one fixed shot-level SE(3)",
            "gt_use": "camera and human metrics only",
        },
        "overall": overall,
        "by_source": by_source,
        "cases": rows,
    }
    output = args.output_dir / "v19_da3_explicit_geometry_correction.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8"
    )
    write_markdown(args.output_dir / "v19_da3_explicit_geometry_correction.md", report)
    print(f">> wrote {output}", flush=True)


if __name__ == "__main__":
    main()
