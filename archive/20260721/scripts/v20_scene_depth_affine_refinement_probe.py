#!/usr/bin/env python3
"""Fit a bounded affine background-depth correction with camera and human fixed."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from v19_da3_explicit_geometry_correction_probe import (  # noqa: E402
    distribution,
    scene_alignment_metrics,
    transform_points,
)
from v20_scene_depth_scale_refinement_probe import load_stream_cases, split_cloud  # noqa: E402
from v20_shot_scale_consistency_probe import case_map, load_raw_clouds, scale_pose  # noqa: E402


DEFAULT_INPUT = (
    REPO_ROOT
    / "output"
    / "v20_shot_scale_consistency"
    / "full180_independent_bound45_conf1_fair"
    / "v20_shot_scale_consistency.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "output" / "v20_shot_scale_consistency" / "scene_depth_affine"
METHODS = ("pelvis_first1", "torso_first1")
BOUNDS = ((0.20, 0.10), (0.20, 0.20), (0.20, 0.30), (0.30, 0.10), (0.30, 0.20), (0.30, 0.30))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_report", type=Path, default=DEFAULT_INPUT)
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
    parser.add_argument("--point_samples", type=int, default=6000)
    parser.add_argument("--raw_confidence_threshold", type=float, default=1.0)
    parser.add_argument("--mask_dilate", type=int, default=11)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args()


def affine_cloud(
    cloud: np.ndarray,
    center: np.ndarray,
    pose_rotation: np.ndarray,
    scale: float,
    shift: float,
) -> np.ndarray:
    relative = cloud - center[None]
    camera = relative @ pose_rotation
    depth = camera[:, 2]
    factor = float(scale) + float(shift) / np.maximum(depth, 0.10)
    return center[None] + relative * factor[:, None]


def fit_affine(
    transform: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    center: np.ndarray,
    pose_rotation: np.ndarray,
    scale_bound: float,
    shift_bound: float,
    iterations: int,
) -> tuple[float, float, dict]:
    tree = cKDTree(target)
    relative = source - center[None]
    camera = relative @ pose_rotation
    depth = camera[:, 2]
    ray = relative / np.maximum(depth[:, None], 0.10)
    base = transform_points(transform, center[None])[0]
    basis_scale = relative @ transform[:3, :3].T
    basis_shift = ray @ transform[:3, :3].T
    scale, shift = 1.0, 0.0
    history = []
    for iteration in range(int(iterations)):
        transformed = base[None] + scale * basis_scale + shift * basis_shift
        distance, nearest = tree.query(transformed, k=1, workers=1)
        finite = np.isfinite(distance)
        if int(finite.sum()) < 64:
            break
        threshold = float(np.quantile(distance[finite], 0.60))
        valid = np.flatnonzero(finite & (distance <= threshold))
        if len(valid) < 64:
            break
        residual_target = target[nearest[valid]] - base[None]
        b0 = basis_scale[valid]
        b1 = basis_shift[valid]
        weights = 1.0 / np.maximum(distance[valid], 0.02)
        a00 = float(np.sum(weights * np.sum(b0 * b0, axis=1)))
        a01 = float(np.sum(weights * np.sum(b0 * b1, axis=1)))
        a11 = float(np.sum(weights * np.sum(b1 * b1, axis=1)))
        y0 = float(np.sum(weights * np.sum(b0 * residual_target, axis=1)))
        y1 = float(np.sum(weights * np.sum(b1 * residual_target, axis=1)))
        ridge = 0.002 * max(a00 + a11, 1.0)
        matrix = np.asarray([[a00 + ridge, a01], [a01, a11 + ridge]], dtype=np.float64)
        target_vector = np.asarray([y0 + ridge, y1], dtype=np.float64)
        try:
            proposed_scale, proposed_shift = np.linalg.solve(matrix, target_vector)
        except np.linalg.LinAlgError:
            break
        proposed_scale = float(np.clip(proposed_scale, 1.0 - scale_bound, 1.0 + scale_bound))
        proposed_shift = float(np.clip(proposed_shift, -shift_bound, shift_bound))
        updated_scale = 0.5 * scale + 0.5 * proposed_scale
        updated_shift = 0.5 * shift + 0.5 * proposed_shift
        history.append(
            {
                "iteration": iteration,
                "scale": updated_scale,
                "shift_m": updated_shift,
                "pairs": int(len(valid)),
                "median_distance_m": float(np.median(distance[valid])),
            }
        )
        if abs(updated_scale - scale) < 1e-4 and abs(updated_shift - shift) < 1e-4:
            scale, shift = updated_scale, updated_shift
            break
        scale, shift = updated_scale, updated_shift
    return scale, shift, {"iterations": history}


def variant_key(scale_bound: float, shift_bound: float) -> str:
    return f"q{int(round(100 * scale_bound)):02d}_d{int(round(100 * shift_bound)):02d}"


def aggregate(rows: list[dict], method: str, variant: str) -> dict:
    values = [row["methods"][method][variant] for row in rows]
    fixed = [row["fixed"] for row in rows]
    scene = np.asarray([row["scene"]["trimmed_mean_m"] for row in values])
    fixed_scene = np.asarray([row["scene"]["trimmed_mean_m"] for row in fixed])
    camera = np.asarray([row["camera"]["translation_m"] for row in values])
    fixed_camera = np.asarray([row["camera"]["translation_m"] for row in fixed])
    human = np.asarray([row["human"]["root_motion_error_m"] for row in values])
    fixed_human = np.asarray([row["human"]["root_motion_error_m"] for row in fixed])
    return {
        "camera_translation_m": distribution(camera.tolist()),
        "human_motion_error_m": distribution(human.tolist()),
        "scene_trimmed_mean_m": distribution(scene.tolist()),
        "depth_scale": distribution([row["depth_scale"] for row in values]),
        "depth_shift_m": distribution([abs(row["depth_shift_m"]) for row in values]),
        "scene_improved_rate": float(np.mean(scene < fixed_scene)),
        "all_three_improved_rate": float(
            np.mean((scene < fixed_scene) & (camera < fixed_camera) & (human < fixed_human))
        ),
        "scene_harmful_rate_010m": float(np.mean(scene > fixed_scene + 0.10)),
    }


def build_summary(rows: list[dict]) -> dict:
    return {
        method: {
            variant_key(*bounds): aggregate(rows, method, variant_key(*bounds)) for bounds in BOUNDS
        }
        for method in METHODS
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads(args.input_report.read_text(encoding="utf-8"))
    input_cases = {row["case_name"]: row for row in report["cases"]}
    streams = load_stream_cases(args.stream_dir)
    v10 = case_map(args.v10_report)
    helper_args = argparse.Namespace(
        point_samples=int(args.point_samples),
        raw_confidence_threshold=float(args.raw_confidence_threshold),
        mask_dilate=int(args.mask_dilate),
    )
    rows = []
    for index, case_name in enumerate(sorted(input_cases)):
        row = input_cases[case_name]
        with np.load(streams[case_name]["cache_path"]) as stream:
            raw_poses = np.stack([stream["old_pose"][-1], stream["new_pose"]]).astype(np.float32)
        rng = np.random.default_rng(int(args.seed) + index)
        raw_old, raw_new = load_raw_clouds(
            Path(v10[case_name]["paths"]["human3r_local_reset"]), raw_poses, helper_args, rng
        )
        raw_old_solve, raw_old_eval = split_cloud(raw_old, rng)
        raw_new_solve, raw_new_eval = split_cloud(raw_new, rng)
        fixed_transform = np.asarray(row["fixed"]["transform"], dtype=np.float32)
        fixed = {
            **row["fixed"],
            "scene": scene_alignment_metrics(fixed_transform, raw_new_eval, raw_old_eval),
        }
        methods = {}
        for method in METHODS:
            source = row["methods"][method]
            old_scale, new_scale = float(source["old_scale"]), float(source["new_scale"])
            old_solve, old_eval = raw_old_solve * old_scale, raw_old_eval * old_scale
            new_solve, new_eval = raw_new_solve * new_scale, raw_new_eval * new_scale
            new_pose = scale_pose(raw_poses[1], new_scale)
            center, pose_rotation = new_pose[:3, 3], new_pose[:3, :3]
            initial = source["variants"]["b00"]
            transform = np.asarray(initial["transform"], dtype=np.float32)
            variants = {}
            for scale_bound, shift_bound in BOUNDS:
                scale, shift, diagnostics = fit_affine(
                    transform,
                    new_solve,
                    old_solve,
                    center,
                    pose_rotation,
                    scale_bound,
                    shift_bound,
                    int(args.iterations),
                )
                corrected = affine_cloud(new_eval, center, pose_rotation, scale, shift)
                variants[variant_key(scale_bound, shift_bound)] = {
                    "depth_scale": scale,
                    "depth_shift_m": shift,
                    "camera": initial["camera"],
                    "human": initial["human"],
                    "scene": scene_alignment_metrics(transform, corrected, old_eval),
                    "fit": diagnostics,
                }
            methods[method] = variants
        rows.append({"case_name": case_name, "source": row["source"], "fixed": fixed, "methods": methods})
        print(f"V20 scene depth affine {index + 1}/{len(input_cases)}", flush=True)
    output = {
        "experiment": "V20 bounded affine background-depth refinement",
        "case_count": len(rows),
        "protocol": {"fit_eval_point_split": True, "bounds": [list(row) for row in BOUNDS], "learned_components": False},
        "overall": build_summary(rows),
        "by_source": {source: build_summary([row for row in rows if row["source"] == source]) for source in sorted({row["source"] for row in rows})},
        "cases": rows,
    }
    path = args.output_dir / "v20_scene_depth_affine_refinement.json"
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    print(f">> wrote {path}", flush=True)


if __name__ == "__main__":
    main()
