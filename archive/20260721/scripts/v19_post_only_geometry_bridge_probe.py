#!/usr/bin/env python3
"""Test causal post-only DA3 geometry correction and scene-residual bounds."""

from __future__ import annotations

import argparse
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
    evaluate,
    load_cases,
    metric_inference,
    resolve,
)
from v19_da3_depth_correction_ablation import corrected_depths, load_raw_pair  # noqa: E402
from v19_da3_explicit_geometry_correction_probe import (  # noqa: E402
    distribution,
    human_metrics,
    sample_cloud,
    scene_alignment_metrics,
    solve_scene_translation,
    transform_points,
)


DEFAULT_OUTPUT = REPO_ROOT / "output" / "v19_da3_explicit_geometry_correction" / "post_only"
GEOMETRIES = ("full_root_scale", "post_root_scale", "post_lowfreq_scale", "post_da3_dense")
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
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--process_res", type=int, default=504)
    parser.add_argument("--keypoint_threshold", type=float, default=0.30)
    parser.add_argument("--sample_radius", type=int, default=3)
    parser.add_argument("--point_samples", type=int, default=3000)
    parser.add_argument("--raw_confidence_threshold", type=float, default=1.5)
    parser.add_argument("--mask_dilate", type=int, default=11)
    parser.add_argument("--lowfreq_sigma", type=float, default=25.0)
    parser.add_argument("--scene_iters", type=int, default=8)
    parser.add_argument("--scene_max_distance", type=float, default=0.60)
    parser.add_argument("--scene_min_distance", type=float, default=0.12)
    parser.add_argument("--scene_max_correction", type=float, default=0.25)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args()


def case_map(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["case_name"]): row for row in payload["cases"]}


def bound_key(bound: float) -> str:
    return f"b{int(round(100.0 * bound)):02d}"


def run_case(
    case: dict,
    v10_case: dict,
    fixed_case: dict,
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
        target_pose = stream["target_pose"].astype(np.float32)
        new_pose = stream["new_pose"].astype(np.float32)
        gt_old_world = stream["old_gt_joints_target_world"][-1, 0].astype(np.float32)
        gt_new_world = stream["new_gt_joints_target_world"][0].astype(np.float32)
        keypoints = np.concatenate(
            [keypoint["old_keypoints"], keypoint["new_keypoints"][None]], axis=0
        )
        keypoint_confidence = np.concatenate(
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
            keypoint_confidence[frame],
            joints[frame],
            float(args.keypoint_threshold),
            int(args.sample_radius),
        )
        da3_roots.append(pelvis)
    da3_roots = np.stack(da3_roots).astype(np.float32)

    local_dir = Path(v10_case["paths"]["human3r_local_reset"])
    raw_pair = load_raw_pair(local_dir)
    kernel = np.ones((int(args.mask_dilate), int(args.mask_dilate)), dtype=np.uint8)
    for pair_index in range(2):
        raw_pair["mask"][pair_index] = cv2.dilate(
            raw_pair["mask"][pair_index], kernel, iterations=1
        )
    resized_da3 = [
        cv2.resize(
            metric_depth[frame],
            (raw_pair["depth"][pair_index].shape[1], raw_pair["depth"][pair_index].shape[0]),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        for pair_index, frame in enumerate((-2, -1))
    ]
    depth_modes = [
        corrected_depths(
            raw_pair["depth"][pair_index],
            resized_da3[pair_index],
            raw_pair["confidence"][pair_index],
            raw_pair["mask"][pair_index],
            float(joints[frame, 0, 2]),
            float(da3_roots[frame, 2]),
            args,
        )
        for pair_index, frame in enumerate((-2, -1))
    ]

    scene_transform = np.asarray(
        case["scene_case"]["fixed_candidates"]["fixed_torso_motion_1f_resolve_t"]["transform"],
        dtype=np.float32,
    )
    camera_rotation = (scene_transform @ new_pose)[:3, :3]
    corrected_translations = translations + (da3_roots - joints[:, 0])
    geometry_specs = {
        "full_root_scale": {
            "old_depth": depth_modes[0]["root_scale_h3r"],
            "new_depth": depth_modes[1]["root_scale_h3r"],
            "old_anchor": da3_roots[-2],
            "new_anchor": da3_roots[-1],
            "old_smpl": corrected_translations[-2],
            "new_smpl": corrected_translations[-1],
            "old_pelvis": da3_roots[-2],
            "new_pelvis": da3_roots[-1],
        },
        "post_root_scale": {
            "old_depth": raw_pair["depth"][0],
            "new_depth": depth_modes[1]["root_scale_h3r"],
            "old_anchor": joints[-2, 0],
            "new_anchor": da3_roots[-1],
            "old_smpl": translations[-2],
            "new_smpl": corrected_translations[-1],
            "old_pelvis": joints[-2, 0],
            "new_pelvis": da3_roots[-1],
        },
        "post_lowfreq_scale": {
            "old_depth": raw_pair["depth"][0],
            "new_depth": depth_modes[1]["lowfreq_scale_h3r"],
            "old_anchor": joints[-2, 0],
            "new_anchor": da3_roots[-1],
            "old_smpl": translations[-2],
            "new_smpl": corrected_translations[-1],
            "old_pelvis": joints[-2, 0],
            "new_pelvis": da3_roots[-1],
        },
        "post_da3_dense": {
            "old_depth": raw_pair["depth"][0],
            "new_depth": resized_da3[1],
            "old_anchor": joints[-2, 0],
            "new_anchor": da3_roots[-1],
            "old_smpl": translations[-2],
            "new_smpl": corrected_translations[-1],
            "old_pelvis": joints[-2, 0],
            "new_pelvis": da3_roots[-1],
        },
    }

    rng = np.random.default_rng(int(args.seed) + index)
    geometries = {}
    for geometry, spec in geometry_specs.items():
        old_anchor_world = transform_points(poses[-2], spec["old_anchor"][None])[0]
        camera_pose = camera_pose_from_human(
            camera_rotation, old_anchor_world, spec["new_anchor"]
        )
        initial = boundary_from_camera_pose(camera_pose, new_pose)
        old_cloud = sample_cloud(
            spec["old_depth"],
            intrinsics[-2],
            poses[-2],
            raw_pair["mask"][0],
            raw_pair["confidence"][0],
            float(args.raw_confidence_threshold),
            int(args.point_samples),
            rng,
        )
        new_cloud = sample_cloud(
            spec["new_depth"],
            intrinsics[-1],
            poses[-1],
            raw_pair["mask"][1],
            raw_pair["confidence"][1],
            float(args.raw_confidence_threshold),
            int(args.point_samples),
            rng,
        )
        bounds = {}
        for bound in BOUNDS:
            if bound == 0.0:
                transform = initial.copy()
                solver = {"translation_correction_m": 0.0, "iterations": []}
            else:
                bound_args = argparse.Namespace(**vars(args))
                bound_args.scene_max_correction = float(bound)
                transform, solver = solve_scene_translation(
                    initial, new_cloud, old_cloud, bound_args
                )
            bounds[bound_key(bound)] = {
                "bound_m": bound,
                "transform": transform.astype(float).tolist(),
                "camera": evaluate(transform, new_pose, target_pose),
                "smpl": human_metrics(
                    transform,
                    spec["old_smpl"],
                    spec["new_smpl"],
                    poses[-2],
                    poses[-1],
                    gt_old_world,
                    gt_new_world,
                ),
                "pelvis": human_metrics(
                    transform,
                    spec["old_pelvis"],
                    spec["new_pelvis"],
                    poses[-2],
                    poses[-1],
                    gt_old_world,
                    gt_new_world,
                ),
                "scene": scene_alignment_metrics(transform, new_cloud, old_cloud),
                "solver": solver,
            }
        geometries[geometry] = bounds
    return {
        "case_name": case["case_name"],
        "source": case["source"],
        "inference_seconds_6frames": elapsed,
        "fixed": fixed_case["methods"]["fixed_raw_geometry"],
        "geometries": geometries,
    }


def aggregate(rows: list[dict], geometry: str, bound: str) -> dict:
    values = [row["geometries"][geometry][bound] for row in rows]
    fixed = [row["fixed"] for row in rows]
    camera_improved = np.asarray(
        [value["camera"]["translation_m"] < base["camera"]["translation_m"] for value, base in zip(values, fixed)]
    )
    human_improved = np.asarray(
        [
            value["smpl"]["root_motion_error_m"] < base["human"]["root_motion_error_m"]
            for value, base in zip(values, fixed)
        ]
    )
    scene_improved = np.asarray(
        [
            value["scene"]["trimmed_mean_m"] < base["scene"]["trimmed_mean_m"]
            for value, base in zip(values, fixed)
        ]
    )
    return {
        "camera_translation_m": distribution([value["camera"]["translation_m"] for value in values]),
        "smpl_motion_error_m": distribution(
            [value["smpl"]["root_motion_error_m"] for value in values]
        ),
        "pelvis_motion_error_m": distribution(
            [value["pelvis"]["root_motion_error_m"] for value in values]
        ),
        "scene_trimmed_mean_m": distribution(
            [value["scene"]["trimmed_mean_m"] for value in values]
        ),
        "scene_overlap_020": distribution([value["scene"]["overlap_020"] for value in values]),
        "camera_improved_rate": float(np.mean(camera_improved)),
        "human_improved_rate": float(np.mean(human_improved)),
        "scene_improved_rate": float(np.mean(scene_improved)),
        "all_three_improved_rate": float(np.mean(camera_improved & human_improved & scene_improved)),
    }


def build_summary(rows: list[dict]) -> dict:
    return {
        geometry: {
            bound_key(bound): aggregate(rows, geometry, bound_key(bound)) for bound in BOUNDS
        }
        for geometry in GEOMETRIES
    }


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# V19 Post-Only Geometry Bridge",
        "",
        "| Geometry | Bound | Camera T | SMPL motion | Pelvis motion | Scene trim | Scene<Fixed | All3<Fixed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for geometry in GEOMETRIES:
        for bound in BOUNDS:
            row = report["overall"][geometry][bound_key(bound)]
            lines.append(
                f"| {geometry} | {bound:.2f} | {row['camera_translation_m']['mean']:.3f} | "
                f"{row['smpl_motion_error_m']['mean']:.3f} | {row['pelvis_motion_error_m']['mean']:.3f} | "
                f"{row['scene_trimmed_mean_m']['mean']:.3f} | "
                f"{100.0 * row['scene_improved_rate']:.1f}% | "
                f"{100.0 * row['all_three_improved_rate']:.1f}% |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("V19 post-only probe requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args)
    v10 = case_map(args.v10_report)
    v19 = case_map(args.v19_report)
    model = DepthAnything3.from_pretrained(str(args.model_path)).to(args.device).eval()
    rows = []
    for index, case in enumerate(cases):
        rows.append(run_case(case, v10[case["case_name"]], v19[case["case_name"]], model, args, index))
        print(f"V19 post-only {index + 1}/{len(cases)}", flush=True)
    overall = build_summary(rows)
    by_source = {
        source: build_summary([row for row in rows if row["source"] == source])
        for source in sorted({row["source"] for row in rows})
    }
    report = {
        "experiment": "V19 causal post-only DA3 geometry correction",
        "case_count": len(rows),
        "protocol": {
            "checkpoint": str(args.model_path),
            "old_shot_modified": False,
            "post_cut_frames": 1,
            "rotation": "V16 torso-motion 20-degree bound",
            "residual_bounds_m": list(BOUNDS),
        },
        "overall": overall,
        "by_source": by_source,
        "cases": rows,
    }
    output = args.output_dir / "v19_post_only_geometry_bridge.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8"
    )
    write_markdown(args.output_dir / "v19_post_only_geometry_bridge.md", report)
    print(f">> wrote {output}", flush=True)


if __name__ == "__main__":
    main()
