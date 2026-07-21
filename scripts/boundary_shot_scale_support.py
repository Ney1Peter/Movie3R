#!/usr/bin/env python3
"""Internal shot-scale utilities for coherent camera/scene/human scaling."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation


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
from boundary_depth_correction_support import load_raw_pair  # noqa: E402
from boundary_geometry_support import (  # noqa: E402
    distribution,
    human_metrics,
    sample_cloud,
    scene_alignment_metrics,
    solve_scene_translation,
    transform_points,
)


DEFAULT_OUTPUT = REPO_ROOT / "output" / "v20_shot_scale_consistency" / "one_frame"
SCALE_SOURCES = ("pelvis", "torso")
OLD_ESTIMATORS = ("first1", "first3_median", "median5", "last1")
BOUNDS = (0.0, 0.05, 0.10, 0.15, 0.25, 0.50)


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
    parser.add_argument("--da3_cache_dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--process_res", type=int, default=504)
    parser.add_argument("--keypoint_threshold", type=float, default=0.30)
    parser.add_argument("--sample_radius", type=int, default=3)
    parser.add_argument("--point_samples", type=int, default=3000)
    parser.add_argument("--raw_confidence_threshold", type=float, default=1.5)
    parser.add_argument("--mask_dilate", type=int, default=11)
    parser.add_argument("--scene_iters", type=int, default=8)
    parser.add_argument("--scene_max_distance", type=float, default=0.60)
    parser.add_argument("--scene_min_distance", type=float, default=0.12)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--overwrite_da3_cache", action="store_true")
    parser.add_argument("--rotation_label", default="torso-motion")
    parser.add_argument(
        "--rotation_candidate_key",
        default="fixed_torso_motion_1f_resolve_t",
        help="Scene-cache candidate used only for the corrected camera rotation.",
    )
    parser.add_argument(
        "--rotation_blend_candidate_key",
        default=None,
        help="Optional second candidate blended on SO(3) with rotation_candidate_key.",
    )
    parser.add_argument("--rotation_blend_alpha", type=float, default=0.0)
    parser.add_argument(
        "--da3_mode",
        choices=("joint_history", "independent_single"),
        default="joint_history",
    )
    return parser.parse_args()


def case_map(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["case_name"]): row for row in payload["cases"]}


def bound_key(bound: float) -> str:
    return f"b{int(round(100.0 * bound)):02d}"


def scale_pose(pose: np.ndarray, scale: float) -> np.ndarray:
    output = np.asarray(pose, dtype=np.float32).copy()
    output[:3, 3] *= float(scale)
    return output


def clipped_scale(value: float) -> float:
    return float(np.clip(value, 0.35, 3.0))


def old_scale(values: np.ndarray, estimator: str) -> float:
    if estimator == "first1":
        return clipped_scale(float(values[0]))
    if estimator == "first3_median":
        return clipped_scale(float(np.median(values[:3])))
    if estimator == "median5":
        return clipped_scale(float(np.median(values)))
    if estimator == "last1":
        return clipped_scale(float(values[-1]))
    raise KeyError(estimator)


def clip_translation_delta(initial: np.ndarray, solved: np.ndarray, bound: float) -> np.ndarray:
    output = np.asarray(initial, dtype=np.float32).copy()
    delta = np.asarray(solved[:3, 3] - initial[:3, 3], dtype=np.float32)
    norm = float(np.linalg.norm(delta))
    if norm > float(bound) > 0.0:
        delta *= float(bound) / norm
    if bound <= 0.0:
        delta[:] = 0.0
    output[:3, 3] += delta
    return output


def da3_cache(
    case: dict,
    model: DepthAnything3,
    args: argparse.Namespace,
) -> dict[str, np.ndarray | float]:
    cache_root = args.da3_cache_dir or (args.output_dir / "da3_cache")
    cache_path = cache_root / f"{case['case_name']}.npz"
    if cache_path.exists() and not args.overwrite_da3_cache:
        with np.load(cache_path) as cache:
            return {key: cache[key] for key in cache.files}

    with np.load(resolve(case["cache_path"])) as stream, np.load(
        resolve(case["keypoint_path"])
    ) as keypoint:
        images = [*stream["old_images"], stream["new_image"]]
        intrinsics = np.concatenate(
            [stream["old_intrinsics"], stream["new_intrinsics"][None]], axis=0
        ).astype(np.float32)
        joints = np.concatenate(
            [stream["old_joints_camera"], stream["new_joints_camera"][None]], axis=0
        ).astype(np.float32)
        keypoints = np.concatenate(
            [keypoint["old_keypoints"], keypoint["new_keypoints"][None]], axis=0
        )
        confidence = np.concatenate(
            [keypoint["old_confidence"], keypoint["new_confidence"][None]], axis=0
        )

    if args.da3_mode == "independent_single":
        selected = (0, len(images) - 1)
        pelvis_selected = []
        torso_selected = []
        elapsed = 0.0
        for frame in selected:
            depth, processed, frame_elapsed = metric_inference(
                model, [images[frame]], intrinsics[frame : frame + 1], int(args.process_res)
            )
            pelvis, torso, _ = estimate_frame_roots(
                depth[0],
                intrinsics[frame],
                processed[0],
                keypoints[frame],
                confidence[frame],
                joints[frame],
                float(args.keypoint_threshold),
                int(args.sample_radius),
            )
            pelvis_selected.append(pelvis)
            torso_selected.append(torso)
            elapsed += float(frame_elapsed)
        pelvis_roots = [pelvis_selected[0]] * (len(images) - 1) + [pelvis_selected[1]]
        torso_roots = [torso_selected[0]] * (len(images) - 1) + [torso_selected[1]]
        metric_depth = np.empty((0,), dtype=np.float32)
        processed_intrinsics = np.empty((0, 3, 3), dtype=np.float32)
    else:
        metric_depth, processed_intrinsics, elapsed = metric_inference(
            model, images, intrinsics, int(args.process_res)
        )
        pelvis_roots = []
        torso_roots = []
        for frame in range(len(images)):
            pelvis, torso, _ = estimate_frame_roots(
                metric_depth[frame],
                intrinsics[frame],
                processed_intrinsics[frame],
                keypoints[frame],
                confidence[frame],
                joints[frame],
                float(args.keypoint_threshold),
                int(args.sample_radius),
            )
            pelvis_roots.append(pelvis)
            torso_roots.append(torso)
    payload = {
        "metric_depth": metric_depth.astype(np.float32),
        "processed_intrinsics": processed_intrinsics.astype(np.float32),
        "pelvis_roots": np.stack(pelvis_roots).astype(np.float32),
        "torso_roots": np.stack(torso_roots).astype(np.float32),
        "inference_seconds": np.asarray(elapsed, dtype=np.float32),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **payload)
    return payload


def calibrated_targets(stream: np.lib.npyio.NpzFile, old_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    old_gt_pose = np.asarray(stream["old_gt_pose"][-1], dtype=np.float32)
    new_gt_pose = np.asarray(stream["new_gt_pose"], dtype=np.float32)
    old_from_gt = old_pose @ np.linalg.inv(old_gt_pose)
    target_pose = old_from_gt @ new_gt_pose
    old_gt_camera_root = np.asarray(stream["old_gt_joints_camera"][-1, 0], dtype=np.float32)
    new_gt_camera_root = np.asarray(stream["new_gt_joints_camera"][0], dtype=np.float32)
    old_gt_world = transform_points(old_pose, old_gt_camera_root[None])[0]
    new_gt_native = transform_points(new_gt_pose, new_gt_camera_root[None])[0]
    new_gt_world = transform_points(old_from_gt, new_gt_native[None])[0]
    return target_pose.astype(np.float32), old_gt_world, new_gt_world


def load_raw_clouds(
    local_dir: Path,
    poses: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    raw = load_raw_pair(local_dir)
    kernel = np.ones((int(args.mask_dilate), int(args.mask_dilate)), dtype=np.uint8)
    clouds = []
    for index in range(2):
        mask = cv2.dilate(raw["mask"][index].astype(np.uint8), kernel, iterations=1)
        with np.load(local_dir / "camera" / f"{index + 1:06d}.npz") as camera:
            intrinsics = np.asarray(camera["intrinsics"], dtype=np.float32)
        clouds.append(
            sample_cloud(
                raw["depth"][index],
                intrinsics,
                poses[index],
                mask,
                raw["confidence"][index],
                float(args.raw_confidence_threshold),
                int(args.point_samples),
                rng,
            )
        )
    return clouds[0], clouds[1]


def run_case(
    case: dict,
    v10_case: dict,
    fixed_case: dict,
    model: DepthAnything3,
    args: argparse.Namespace,
    index: int,
) -> dict:
    with np.load(resolve(case["cache_path"])) as stream:
        poses = np.concatenate([stream["old_pose"], stream["new_pose"][None]], axis=0).astype(
            np.float32
        )
        joints = np.concatenate(
            [stream["old_joints_camera"], stream["new_joints_camera"][None]], axis=0
        ).astype(np.float32)
        translations = np.concatenate(
            [stream["old_transl"], stream["new_transl"][None]], axis=0
        ).astype(np.float32)
        old_gt_pose = np.asarray(stream["old_gt_pose"][-1], dtype=np.float32)
        new_gt_pose = np.asarray(stream["new_gt_pose"], dtype=np.float32)
        raw_target_pose = np.asarray(stream["target_pose"], dtype=np.float32)
        raw_gt_old_world = np.asarray(stream["old_gt_joints_target_world"][-1, 0], dtype=np.float32)
        raw_gt_new_world = np.asarray(stream["new_gt_joints_target_world"][0], dtype=np.float32)

        cached = da3_cache(case, model, args)
        pelvis = np.asarray(cached["pelvis_roots"], dtype=np.float32)
        torso = np.asarray(cached["torso_roots"], dtype=np.float32)
        scales = {
            "pelvis": np.clip(pelvis[:, 2] / np.maximum(joints[:, 0, 2], 1e-4), 0.35, 3.0),
            "torso": np.clip(torso[:, 2] / np.maximum(joints[:, 0, 2], 1e-4), 0.35, 3.0),
        }

        raw_poses = np.stack([poses[-2], poses[-1]])
        local_dir = Path(v10_case["paths"]["human3r_local_reset"])
        rng = np.random.default_rng(int(args.seed) + index)
        raw_old_cloud, raw_new_cloud = load_raw_clouds(local_dir, raw_poses, args, rng)
        fixed_transform = np.asarray(
            fixed_case["methods"]["fixed_raw_geometry"]["transform"], dtype=np.float32
        )
        fixed_consistent = {
            "transform": fixed_transform.astype(float).tolist(),
            "camera": evaluate(fixed_transform, poses[-1], raw_target_pose),
            "human": human_metrics(
                fixed_transform,
                translations[-2],
                translations[-1],
                poses[-2],
                poses[-1],
                raw_gt_old_world,
                raw_gt_new_world,
            ),
            "scene": scene_alignment_metrics(fixed_transform, raw_new_cloud, raw_old_cloud),
        }

        torso_boundary = np.asarray(
            case["scene_case"]["fixed_candidates"][str(args.rotation_candidate_key)]["transform"],
            dtype=np.float32,
        )
        corrected_camera_rotation = (torso_boundary @ poses[-1])[:3, :3]
        if args.rotation_blend_candidate_key and float(args.rotation_blend_alpha) > 0.0:
            blend_boundary = np.asarray(
                case["scene_case"]["fixed_candidates"][str(args.rotation_blend_candidate_key)][
                    "transform"
                ],
                dtype=np.float32,
            )
            blend_camera_rotation = (blend_boundary @ poses[-1])[:3, :3]
            relative = Rotation.from_matrix(
                blend_camera_rotation.astype(np.float64)
                @ corrected_camera_rotation.astype(np.float64).T
            ).as_rotvec()
            corrected_camera_rotation = (
                Rotation.from_rotvec(float(args.rotation_blend_alpha) * relative).as_matrix()
                @ corrected_camera_rotation
            ).astype(np.float32)
        methods = {}
        for source in SCALE_SOURCES:
            for estimator in OLD_ESTIMATORS:
                old_s = old_scale(scales[source][:-1], estimator)
                new_s = clipped_scale(float(scales[source][-1]))
                old_pose = scale_pose(poses[-2], old_s)
                new_pose = scale_pose(poses[-1], new_s)
                target_pose, gt_old_world, gt_new_world = calibrated_targets(stream, old_pose)
                old_root = translations[-2] * old_s
                new_root = translations[-1] * new_s
                old_anchor_world = transform_points(old_pose, old_root[None])[0]
                camera_pose = camera_pose_from_human(
                    corrected_camera_rotation, old_anchor_world, new_root
                )
                initial = boundary_from_camera_pose(camera_pose, new_pose)
                old_cloud = raw_old_cloud * old_s
                new_cloud = raw_new_cloud * new_s

                bound_args = argparse.Namespace(**vars(args))
                bound_args.scene_max_correction = 0.50
                solved, solver = solve_scene_translation(initial, new_cloud, old_cloud, bound_args)
                variants = {}
                for bound in BOUNDS:
                    transform = clip_translation_delta(initial, solved, bound)
                    variants[bound_key(bound)] = {
                        "bound_m": bound,
                        "transform": transform.astype(float).tolist(),
                        "camera": evaluate(transform, new_pose, target_pose),
                        "human": human_metrics(
                            transform,
                            old_root,
                            new_root,
                            old_pose,
                            new_pose,
                            gt_old_world,
                            gt_new_world,
                        ),
                        "scene": scene_alignment_metrics(transform, new_cloud, old_cloud),
                    }
                variants["solver"] = solver
                methods[f"{source}_{estimator}"] = {
                    "old_scale": old_s,
                    "new_scale": new_s,
                    "variants": variants,
                }

        return {
            "case_name": case["case_name"],
            "source": case["source"],
            "fixed": fixed_consistent,
            "scale_diagnostics": {
                source: {
                    "old_values": scales[source][:-1].astype(float).tolist(),
                    "new_value": float(scales[source][-1]),
                    "old_range": float(np.ptp(scales[source][:-1])),
                    "old_mad": float(
                        np.median(
                            np.abs(scales[source][:-1] - np.median(scales[source][:-1]))
                        )
                    ),
                }
                for source in SCALE_SOURCES
            },
            "methods": methods,
            "target_camera_baseline_m": float(np.linalg.norm(new_gt_pose[:3, 3] - old_gt_pose[:3, 3])),
            "da3_inference_seconds": float(np.asarray(cached["inference_seconds"])),
        }


def aggregate(rows: list[dict], method: str, bound: str) -> dict:
    values = [row["methods"][method]["variants"][bound] for row in rows]
    fixed = [row["fixed"] for row in rows]
    camera = np.asarray([row["camera"]["translation_m"] for row in values])
    human = np.asarray([row["human"]["root_motion_error_m"] for row in values])
    scene = np.asarray([row["scene"]["trimmed_mean_m"] for row in values])
    fixed_camera = np.asarray([row["camera"]["translation_m"] for row in fixed])
    fixed_human = np.asarray([row["human"]["root_motion_error_m"] for row in fixed])
    fixed_scene = np.asarray([row["scene"]["trimmed_mean_m"] for row in fixed])
    improved = (camera < fixed_camera) & (human < fixed_human) & (scene < fixed_scene)
    return {
        "camera_translation_m": distribution(camera.tolist()),
        "camera_rotation_deg": distribution(
            [row["camera"]["rotation_deg"] for row in values]
        ),
        "human_motion_error_m": distribution(human.tolist()),
        "scene_trimmed_mean_m": distribution(scene.tolist()),
        "scene_overlap_020": distribution([row["scene"]["overlap_020"] for row in values]),
        "camera_improved_rate": float(np.mean(camera < fixed_camera)),
        "human_improved_rate": float(np.mean(human < fixed_human)),
        "scene_improved_rate": float(np.mean(scene < fixed_scene)),
        "all_three_improved_rate": float(np.mean(improved)),
        "camera_harmful_rate_010m": float(np.mean(camera > fixed_camera + 0.10)),
        "human_harmful_rate_010m": float(np.mean(human > fixed_human + 0.10)),
        "scene_harmful_rate_010m": float(np.mean(scene > fixed_scene + 0.10)),
        "translation_catastrophic_rate_2m": float(np.mean(camera > 2.0)),
    }


def build_summary(rows: list[dict]) -> dict:
    methods = sorted(rows[0]["methods"])
    return {
        method: {
            bound_key(bound): aggregate(rows, method, bound_key(bound)) for bound in BOUNDS
        }
        for method in methods
    }


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# Shot-Scale Consistency Support",
        "",
        "| Method | Bound | Camera T | R | Human | Scene | Scene<Fixed | All3<Fixed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, bounds in report["overall"].items():
        for bound in BOUNDS:
            row = bounds[bound_key(bound)]
            lines.append(
                f"| {method} | {bound:.2f} | {row['camera_translation_m']['mean']:.3f} | "
                f"{row['camera_rotation_deg']['mean']:.2f} | {row['human_motion_error_m']['mean']:.3f} | "
                f"{row['scene_trimmed_mean_m']['mean']:.3f} | "
                f"{100.0 * row['scene_improved_rate']:.1f}% | "
                f"{100.0 * row['all_three_improved_rate']:.1f}% |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("Shot-scale support requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args)
    v10 = case_map(args.v10_report)
    v19 = case_map(args.v19_report)
    model = DepthAnything3.from_pretrained(str(args.model_path)).to(args.device).eval()
    rows = []
    for index, case in enumerate(cases):
        rows.append(run_case(case, v10[case["case_name"]], v19[case["case_name"]], model, args, index))
        print(f"Shot-scale support {index + 1}/{len(cases)}", flush=True)
    overall = build_summary(rows)
    by_source = {
        source: build_summary([row for row in rows if row["source"] == source])
        for source in sorted({row["source"] for row in rows})
    }
    scale_stability = {
        source: {
            "old_range": distribution(
                [row["scale_diagnostics"][source]["old_range"] for row in rows]
            ),
            "old_mad": distribution(
                [row["scale_diagnostics"][source]["old_mad"] for row in rows]
            ),
        }
        for source in SCALE_SOURCES
    }
    report = {
        "experiment": "Fixed shot-level DA3 scale for Human3R explicit geometry",
        "case_count": len(rows),
        "protocol": {
            "post_cut_frames": 1,
            "old_shot_scale_estimators": list(OLD_ESTIMATORS),
            "scale_sources": list(SCALE_SOURCES),
            "scene_residual_bounds_m": list(BOUNDS),
            "rotation": str(args.rotation_label),
            "rotation_candidate_key": str(args.rotation_candidate_key),
            "rotation_blend_candidate_key": args.rotation_blend_candidate_key,
            "rotation_blend_alpha": float(args.rotation_blend_alpha),
            "da3_mode": str(args.da3_mode),
            "coupled_outputs": ["camera translation", "camera trajectory", "pointmap", "SMPL-X root"],
        },
        "scale_stability": scale_stability,
        "overall": overall,
        "by_source": by_source,
        "cases": rows,
    }
    output = args.output_dir / "v20_shot_scale_consistency.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(args.output_dir / "v20_shot_scale_consistency.md", report)
    print(f">> wrote {output}", flush=True)


if __name__ == "__main__":
    main()
