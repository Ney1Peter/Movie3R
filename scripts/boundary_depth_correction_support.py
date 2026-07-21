#!/usr/bin/env python3
"""Internal depth-correction utilities retained from the DA3 ablation."""

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
from boundary_geometry_support import (  # noqa: E402
    distribution,
    human_metrics,
    sample_cloud,
    scene_alignment_metrics,
    solve_scene_translation,
    transform_points,
)


DEFAULT_OUTPUT = REPO_ROOT / "output" / "v19_da3_explicit_geometry_correction" / "depth_ablation"
MODES = (
    "da3_dense",
    "root_shift_h3r",
    "root_scale_h3r",
    "scene_shift_h3r",
    "scene_scale_h3r",
    "scene_affine_h3r",
    "lowfreq_shift_h3r",
    "lowfreq_scale_h3r",
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


def load_raw_pair(local_dir: Path) -> dict:
    output = {"depth": [], "confidence": [], "mask": []}
    for frame in (1, 2):
        output["depth"].append(np.load(local_dir / "depth" / f"{frame:06d}.npy").astype(np.float32))
        output["confidence"].append(
            np.load(local_dir / "conf" / f"{frame:06d}.npy").astype(np.float32)
        )
        with np.load(local_dir / "smpl" / f"{frame:06d}.npz", allow_pickle=True) as smpl:
            output["mask"].append(np.asarray(smpl["msk"][0] > 0.10, dtype=np.uint8))
    return output


def robust_background_pairs(
    raw: np.ndarray,
    da3: np.ndarray,
    confidence: np.ndarray,
    mask: np.ndarray,
    confidence_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = (
        np.isfinite(raw)
        & np.isfinite(da3)
        & (raw > 0.10)
        & (raw < 30.0)
        & (da3 > 0.10)
        & (da3 < 30.0)
        & np.isfinite(confidence)
        & (confidence > confidence_threshold)
        & (mask == 0)
    )
    return raw[valid], da3[valid], valid


def weighted_smooth(values: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    numerator = cv2.GaussianBlur(
        np.where(valid, values, 0.0).astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma
    )
    denominator = cv2.GaussianBlur(
        valid.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma
    )
    fallback = float(np.median(values[valid])) if int(valid.sum()) else 0.0
    return np.where(denominator > 1e-4, numerator / np.maximum(denominator, 1e-4), fallback)


def robust_affine(source: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    if len(source) < 64:
        return 1.0, 0.0
    ids = np.linspace(0, len(source) - 1, min(len(source), 20000), dtype=np.int64)
    x = source[ids].astype(np.float64)
    y = target[ids].astype(np.float64)
    keep = np.ones(len(x), dtype=bool)
    scale, shift = 1.0, 0.0
    for _ in range(4):
        design = np.stack([x[keep], np.ones(int(keep.sum()))], axis=1)
        scale, shift = np.linalg.lstsq(design, y[keep], rcond=None)[0]
        residual = y - (scale * x + shift)
        center = float(np.median(residual[keep]))
        mad = float(np.median(np.abs(residual[keep] - center)))
        keep = np.abs(residual - center) <= max(3.0 * 1.4826 * mad, 0.10)
        if int(keep.sum()) < 64:
            break
    return float(np.clip(scale, 0.35, 3.0)), float(np.clip(shift, -3.0, 3.0))


def corrected_depths(
    raw: np.ndarray,
    da3: np.ndarray,
    confidence: np.ndarray,
    mask: np.ndarray,
    h3r_root_depth: float,
    da3_root_depth: float,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    source, target, valid = robust_background_pairs(
        raw, da3, confidence, mask, float(args.raw_confidence_threshold)
    )
    root_shift = float(da3_root_depth - h3r_root_depth)
    root_scale = float(da3_root_depth / max(h3r_root_depth, 1e-4))
    scene_shift = float(np.median(target - source)) if len(source) else root_shift
    ratios = target / np.maximum(source, 1e-4)
    ratios = ratios[np.isfinite(ratios) & (ratios > 0.2) & (ratios < 5.0)]
    scene_scale = float(np.median(ratios)) if len(ratios) else root_scale
    affine_scale, affine_shift = robust_affine(source, target)

    delta = da3 - raw
    lowfreq_delta = weighted_smooth(delta, valid, float(args.lowfreq_sigma))
    log_ratio = np.log(np.maximum(da3, 1e-4) / np.maximum(raw, 1e-4))
    lowfreq_log_ratio = weighted_smooth(log_ratio, valid, float(args.lowfreq_sigma))
    output = {
        "da3_dense": da3,
        "root_shift_h3r": raw + root_shift,
        "root_scale_h3r": raw * root_scale,
        "scene_shift_h3r": raw + scene_shift,
        "scene_scale_h3r": raw * scene_scale,
        "scene_affine_h3r": raw * affine_scale + affine_shift,
        "lowfreq_shift_h3r": raw + lowfreq_delta,
        "lowfreq_scale_h3r": raw * np.exp(np.clip(lowfreq_log_ratio, -1.2, 1.2)),
    }
    return {name: np.clip(depth, 0.05, 30.0).astype(np.float32) for name, depth in output.items()}


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
    scene_camera_pose = scene_transform @ new_pose
    old_da3_world = transform_points(poses[-2], da3_roots[-2][None])[0]
    da3_camera_pose = camera_pose_from_human(
        scene_camera_pose[:3, :3], old_da3_world, da3_roots[-1]
    )
    da3_boundary = boundary_from_camera_pose(da3_camera_pose, new_pose)
    corrected_translations = translations + (da3_roots - joints[:, 0])
    human_initial = human_metrics(
        da3_boundary,
        corrected_translations[-2],
        corrected_translations[-1],
        poses[-2],
        poses[-1],
        gt_old_world,
        gt_new_world,
    )

    modes = {}
    rng = np.random.default_rng(int(args.seed) + index)
    for mode in MODES:
        old_cloud = sample_cloud(
            depth_modes[0][mode],
            intrinsics[-2],
            poses[-2],
            raw_pair["mask"][0],
            raw_pair["confidence"][0],
            float(args.raw_confidence_threshold),
            int(args.point_samples),
            rng,
        )
        new_cloud = sample_cloud(
            depth_modes[1][mode],
            intrinsics[-1],
            poses[-1],
            raw_pair["mask"][1],
            raw_pair["confidence"][1],
            float(args.raw_confidence_threshold),
            int(args.point_samples),
            rng,
        )
        residual_transform, solver = solve_scene_translation(
            da3_boundary, new_cloud, old_cloud, args
        )
        modes[mode] = {
            "initial": {
                "transform": da3_boundary.astype(float).tolist(),
                "camera": evaluate(da3_boundary, new_pose, target_pose),
                "human": human_initial,
                "scene": scene_alignment_metrics(da3_boundary, new_cloud, old_cloud),
            },
            "residual": {
                "transform": residual_transform.astype(float).tolist(),
                "camera": evaluate(residual_transform, new_pose, target_pose),
                "human": human_metrics(
                    residual_transform,
                    corrected_translations[-2],
                    corrected_translations[-1],
                    poses[-2],
                    poses[-1],
                    gt_old_world,
                    gt_new_world,
                ),
                "scene": scene_alignment_metrics(residual_transform, new_cloud, old_cloud),
                "solver": solver,
            },
        }
    return {
        "case_name": case["case_name"],
        "source": case["source"],
        "inference_seconds_6frames": elapsed,
        "fixed": fixed_case["methods"]["fixed_raw_geometry"],
        "modes": modes,
    }


def aggregate(rows: list[dict], mode: str, variant: str) -> dict:
    values = [row["modes"][mode][variant] for row in rows]
    fixed = [row["fixed"] for row in rows]
    camera_improved = np.asarray(
        [value["camera"]["translation_m"] < base["camera"]["translation_m"] for value, base in zip(values, fixed)]
    )
    human_improved = np.asarray(
        [
            value["human"]["root_motion_error_m"] < base["human"]["root_motion_error_m"]
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
        "human_motion_error_m": distribution(
            [value["human"]["root_motion_error_m"] for value in values]
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
        mode: {variant: aggregate(rows, mode, variant) for variant in ("initial", "residual")}
        for mode in MODES
    }


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# DA3 Depth Correction Support",
        "",
        "| Mode | Variant | Camera T | Human motion | Scene trim | Scene<Fixed | All3<Fixed |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        for variant in ("initial", "residual"):
            row = report["overall"][mode][variant]
            lines.append(
                f"| {mode} | {variant} | {row['camera_translation_m']['mean']:.3f} | "
                f"{row['human_motion_error_m']['mean']:.3f} | "
                f"{row['scene_trimmed_mean_m']['mean']:.3f} | "
                f"{100.0 * row['scene_improved_rate']:.1f}% | "
                f"{100.0 * row['all_three_improved_rate']:.1f}% |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("DA3 depth correction ablation requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args)
    v10 = case_map(args.v10_report)
    v19 = case_map(args.v19_report)
    model = DepthAnything3.from_pretrained(str(args.model_path)).to(args.device).eval()
    rows = []
    for index, case in enumerate(cases):
        rows.append(run_case(case, v10[case["case_name"]], v19[case["case_name"]], model, args, index))
        print(f"DA3 depth support {index + 1}/{len(cases)}", flush=True)
    overall = build_summary(rows)
    by_source = {
        source: build_summary([row for row in rows if row["source"] == source])
        for source in sorted({row["source"] for row in rows})
    }
    report = {
        "experiment": "Conservative DA3 correction of Human3R depth",
        "case_count": len(rows),
        "protocol": {
            "checkpoint": str(args.model_path),
            "rotation": "torso-motion 20-degree bound",
            "human_geometry": "SMPL-X translated to DA3 pelvis depth",
            "scene_residual_max_m": float(args.scene_max_correction),
            "low_frequency_sigma_px": float(args.lowfreq_sigma),
        },
        "overall": overall,
        "by_source": by_source,
        "cases": rows,
    }
    output = args.output_dir / "v19_da3_depth_correction_ablation.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8"
    )
    write_markdown(args.output_dir / "v19_da3_depth_correction_ablation.md", report)
    print(f">> wrote {output}", flush=True)


if __name__ == "__main__":
    main()
