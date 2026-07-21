#!/usr/bin/env python3
"""Compare one- and three-frame shot-start DA3 scale calibration."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from v18_da3_metric_depth_probe import (  # noqa: E402
    DEFAULT_MODEL,
    DepthAnything3,
    boundary_from_camera_pose,
    camera_pose_from_human,
    estimate_frame_roots,
    metric_inference,
)
from v19_da3_depth_correction_ablation import load_raw_pair  # noqa: E402
from v19_da3_explicit_geometry_correction_probe import (  # noqa: E402
    distribution,
    human_metrics,
    sample_cloud,
    scene_alignment_metrics,
    solve_scene_translation,
    transform_points,
)
from v20_shot_scale_consistency_probe import (  # noqa: E402
    bound_key,
    case_map,
    clip_translation_delta,
    clipped_scale,
    scale_pose,
)


DEFAULT_OUTPUT = REPO_ROOT / "output" / "v20_shot_scale_consistency" / "three_frame"
BOUNDS = (0.0, 0.25, 0.50)
SCALE_SOURCES = ("pelvis", "torso")
OLD_ESTIMATORS = ("first1",)
NEW_ESTIMATORS = ("first1", "median3")
ROTATIONS = ("rotation1", "rotation3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stream_dir",
        type=Path,
        default=REPO_ROOT / "output" / "v20_shot_scale_consistency" / "stream3_cache",
    )
    parser.add_argument(
        "--keypoint_dir",
        type=Path,
        default=REPO_ROOT / "output" / "v20_shot_scale_consistency" / "keypoint3_cache",
    )
    parser.add_argument(
        "--scene_dir",
        type=Path,
        default=REPO_ROOT / "output" / "v18_human_metric_translation" / "v16_bound20_scene",
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
    parser.add_argument(
        "--v19_report",
        type=Path,
        default=REPO_ROOT
        / "output"
        / "v19_da3_explicit_geometry_correction"
        / "full180"
        / "v19_da3_explicit_geometry_correction.json",
    )
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--process_res", type=int, default=504)
    parser.add_argument("--keypoint_threshold", type=float, default=0.30)
    parser.add_argument("--sample_radius", type=int, default=3)
    parser.add_argument("--point_samples", type=int, default=3000)
    parser.add_argument("--raw_confidence_threshold", type=float, default=1.0)
    parser.add_argument("--mask_dilate", type=int, default=11)
    parser.add_argument("--scene_iters", type=int, default=8)
    parser.add_argument("--scene_max_distance", type=float, default=0.60)
    parser.add_argument("--scene_min_distance", type=float, default=0.12)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args()


def manifest_rows(root: Path, pattern: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(str(root / pattern))):
        rows.extend(json.loads(Path(path).read_text(encoding="utf-8"))["cases"])
    return rows


def load_cases(args: argparse.Namespace) -> list[dict]:
    stream = manifest_rows(args.stream_dir, "v20_stream3_shard_*_of_*.json")
    keypoints = {
        row["case_name"]: row
        for row in manifest_rows(args.keypoint_dir, "v20_keypoints3_shard_*_of_*.json")
    }
    scene = {
        row["case_name"]: row
        for row in manifest_rows(args.scene_dir, "v16_candidates_shard_*_of_*.json")
    }
    if len(stream) != 180 or len(keypoints) != 180 or len(scene) != 180:
        raise RuntimeError(
            f"Expected 180 stream/keypoint/scene cases, got {len(stream)}/{len(keypoints)}/{len(scene)}"
        )
    rows = [
        {**row, "keypoint_path": keypoints[row["case_name"]]["cache_path"], "scene_case": scene[row["case_name"]]}
        for row in stream
    ]
    rows = sorted(rows, key=lambda row: str(row["case_name"]))
    return rows[: int(args.max_cases)] if int(args.max_cases) > 0 else rows


def estimate_scale(values: np.ndarray, estimator: str) -> float:
    if estimator == "first1":
        value = float(values[0])
        if np.isfinite(value):
            return clipped_scale(value)
        finite = values[np.isfinite(values)]
        return clipped_scale(float(np.median(finite))) if len(finite) else 1.0
    if estimator in ("first3", "median3"):
        finite = values[:3][np.isfinite(values[:3])]
        return clipped_scale(float(np.median(finite))) if len(finite) else 1.0
    raise KeyError(estimator)


def calibrated_targets_first(
    stream: np.lib.npyio.NpzFile, old_pose: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    old_gt_pose = np.asarray(stream["old_gt_pose"][-1], dtype=np.float32)
    new_gt_pose = np.asarray(stream["new_gt_pose"][0], dtype=np.float32)
    old_from_gt = old_pose @ np.linalg.inv(old_gt_pose)
    target_pose = old_from_gt @ new_gt_pose
    old_gt_root = np.asarray(stream["old_gt_joints_camera"][-1, 0], dtype=np.float32)
    new_gt_root = np.asarray(stream["new_gt_joints_camera"][0, 0], dtype=np.float32)
    old_gt_world = transform_points(old_pose, old_gt_root[None])[0]
    new_gt_native = transform_points(new_gt_pose, new_gt_root[None])[0]
    new_gt_world = transform_points(old_from_gt, new_gt_native[None])[0]
    return target_pose.astype(np.float32), old_gt_world, new_gt_world


def load_clouds(local_dir: Path, poses: np.ndarray, args: argparse.Namespace, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    raw = load_raw_pair(local_dir)
    kernel = np.ones((int(args.mask_dilate), int(args.mask_dilate)), dtype=np.uint8)
    clouds = []
    for index in range(2):
        mask = cv2.dilate(raw["mask"][index].astype(np.uint8), kernel, iterations=1)
        with np.load(local_dir / "camera" / f"{index + 1:06d}.npz") as camera:
            intrinsics = np.asarray(camera["intrinsics"], dtype=np.float32)
        clouds.append(
            sample_cloud(
                raw["depth"][index], intrinsics, poses[index], mask, raw["confidence"][index],
                float(args.raw_confidence_threshold), int(args.point_samples), rng,
            )
        )
    return clouds[0], clouds[1]


def run_case(case: dict, v10_case: dict, fixed_case: dict, model: DepthAnything3, args: argparse.Namespace, index: int) -> dict:
    with np.load(case["cache_path"]) as stream, np.load(case["keypoint_path"]) as keypoint:
        images = [*stream["old_images"], *stream["new_images"]]
        intrinsics = np.concatenate([stream["old_intrinsics"], stream["new_intrinsics"]]).astype(np.float32)
        joints = np.concatenate([stream["old_joints_camera"], stream["new_joints_camera"]]).astype(np.float32)
        poses = np.concatenate([stream["old_pose"], stream["new_pose"]]).astype(np.float32)
        translations = np.concatenate([stream["old_transl"], stream["new_transl"]]).astype(np.float32)
        keypoints = np.concatenate([keypoint["old_keypoints"], keypoint["new_keypoints"]])
        confidence = np.concatenate([keypoint["old_confidence"], keypoint["new_confidence"]])

        metric_depth, processed_intrinsics, elapsed = metric_inference(model, images, intrinsics, int(args.process_res))
        pelvis_roots, torso_roots = [], []
        for frame in range(len(images)):
            pelvis, torso, _ = estimate_frame_roots(
                metric_depth[frame], intrinsics[frame], processed_intrinsics[frame], keypoints[frame],
                confidence[frame], joints[frame], float(args.keypoint_threshold), int(args.sample_radius),
            )
            pelvis_roots.append(pelvis)
            torso_roots.append(torso)
        root_sets = {"pelvis": np.stack(pelvis_roots), "torso": np.stack(torso_roots)}
        scale_sets = {
            name: np.clip(roots[:, 2] / np.maximum(joints[:, 0, 2], 1e-4), 0.35, 3.0)
            for name, roots in root_sets.items()
        }

        local_dir = Path(v10_case["paths"]["human3r_local_reset"])
        rng = np.random.default_rng(int(args.seed) + index)
        raw_old_cloud, raw_new_cloud = load_clouds(local_dir, np.stack([poses[4], poses[5]]), args, rng)
        rotation_transforms = {
            "rotation1": np.asarray(case["scene_case"]["fixed_candidates"]["fixed_torso_motion_1f_resolve_t"]["transform"], dtype=np.float32),
            "rotation3": np.asarray(case["scene_case"]["fixed_candidates"]["fixed_torso_motion_3f_resolve_t"]["transform"], dtype=np.float32),
        }
        methods = {}
        for source in SCALE_SOURCES:
            old_values = scale_sets[source][:5]
            new_values = scale_sets[source][5:]
            for old_estimator in OLD_ESTIMATORS:
                old_s = estimate_scale(old_values, old_estimator)
                for new_estimator in NEW_ESTIMATORS:
                    new_s = estimate_scale(new_values, new_estimator)
                    old_pose = scale_pose(poses[4], old_s)
                    new_pose = scale_pose(poses[5], new_s)
                    target_pose, gt_old_world, gt_new_world = calibrated_targets_first(stream, old_pose)
                    old_root = translations[4] * old_s
                    new_root = translations[5] * new_s
                    old_cloud = raw_old_cloud * old_s
                    new_cloud = raw_new_cloud * new_s
                    for rotation_name in ROTATIONS:
                        camera_rotation = (rotation_transforms[rotation_name] @ poses[5])[:3, :3]
                        old_anchor_world = transform_points(old_pose, old_root[None])[0]
                        camera_pose = camera_pose_from_human(camera_rotation, old_anchor_world, new_root)
                        initial = boundary_from_camera_pose(camera_pose, new_pose)
                        solve_args = argparse.Namespace(**vars(args))
                        solve_args.scene_max_correction = 0.50
                        solved, solver = solve_scene_translation(initial, new_cloud, old_cloud, solve_args)
                        variants = {}
                        for bound in BOUNDS:
                            transform = clip_translation_delta(initial, solved, bound)
                            variants[bound_key(bound)] = {
                                "camera": evaluate_pose(transform, new_pose, target_pose),
                                "human": human_metrics(transform, old_root, new_root, old_pose, new_pose, gt_old_world, gt_new_world),
                                "scene": scene_alignment_metrics(transform, new_cloud, old_cloud),
                                "transform": transform.astype(float).tolist(),
                            }
                        variants["solver"] = solver
                        name = f"{source}_{old_estimator}_{new_estimator}_{rotation_name}"
                        methods[name] = {"old_scale": old_s, "new_scale": new_s, "variants": variants}
        return {
            "case_name": case["case_name"], "source": case["source"],
            "fixed": fixed_case["methods"]["fixed_raw_geometry"], "methods": methods,
            "scale_diagnostics": {
                source: {
                    "old": scale_sets[source][:5].astype(float).tolist(),
                    "new": scale_sets[source][5:].astype(float).tolist(),
                    "new_range": float(np.ptp(scale_sets[source][5:])),
                }
                for source in SCALE_SOURCES
            },
            "da3_inference_seconds_8frames": elapsed,
        }


def evaluate_pose(boundary: np.ndarray, pose: np.ndarray, target: np.ndarray) -> dict:
    from v18_da3_metric_depth_probe import evaluate
    return evaluate(boundary, pose, target)


def aggregate(rows: list[dict], method: str, bound: str) -> dict:
    values = [row["methods"][method]["variants"][bound] for row in rows]
    fixed = [row["fixed"] for row in rows]
    camera = np.asarray([row["camera"]["translation_m"] for row in values])
    human = np.asarray([row["human"]["root_motion_error_m"] for row in values])
    scene = np.asarray([row["scene"]["trimmed_mean_m"] for row in values])
    fixed_camera = np.asarray([row["camera"]["translation_m"] for row in fixed])
    fixed_human = np.asarray([row["human"]["root_motion_error_m"] for row in fixed])
    fixed_scene = np.asarray([row["scene"]["trimmed_mean_m"] for row in fixed])
    valid = np.isfinite(scene) & np.isfinite(fixed_scene)
    return {
        "camera_translation_m": distribution(camera.tolist()),
        "camera_rotation_deg": distribution([row["camera"]["rotation_deg"] for row in values]),
        "human_motion_error_m": distribution(human.tolist()),
        "scene_trimmed_mean_m": distribution(scene.tolist()),
        "camera_improved_rate": float(np.mean(camera < fixed_camera)),
        "human_improved_rate": float(np.mean(human < fixed_human)),
        "scene_improved_rate_valid": float(np.mean(scene[valid] < fixed_scene[valid])) if valid.any() else float("nan"),
        "scene_valid_count": int(valid.sum()),
        "all_three_improved_rate_valid": float(np.mean((camera[valid] < fixed_camera[valid]) & (human[valid] < fixed_human[valid]) & (scene[valid] < fixed_scene[valid]))) if valid.any() else float("nan"),
    }


def build_summary(rows: list[dict]) -> dict:
    return {
        method: {bound_key(bound): aggregate(rows, method, bound_key(bound)) for bound in BOUNDS}
        for method in sorted(rows[0]["methods"])
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("V20 three-frame probe requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args)
    v10, v19 = case_map(args.v10_report), case_map(args.v19_report)
    model = DepthAnything3.from_pretrained(str(args.model_path)).to(args.device).eval()
    rows = []
    for index, case in enumerate(cases):
        rows.append(run_case(case, v10[case["case_name"]], v19[case["case_name"]], model, args, index))
        print(f"V20 three-frame {index + 1}/{len(cases)}", flush=True)
    report = {
        "experiment": "V20 one-frame versus three-frame shot-scale calibration",
        "case_count": len(rows),
        "protocol": {"post_cut_frames": 3, "bounds_m": list(BOUNDS), "raw_confidence_threshold": float(args.raw_confidence_threshold)},
        "overall": build_summary(rows),
        "by_source": {source: build_summary([row for row in rows if row["source"] == source]) for source in sorted({row["source"] for row in rows})},
        "cases": rows,
    }
    output = args.output_dir / "v20_three_frame_shot_scale.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    print(f">> wrote {output}", flush=True)


if __name__ == "__main__":
    main()
