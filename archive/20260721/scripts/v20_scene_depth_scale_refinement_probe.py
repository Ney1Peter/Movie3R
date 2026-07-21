#!/usr/bin/env python3
"""Refine only post-cut background pointmap depth after fixing camera and human alignment."""

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
from v20_shot_scale_consistency_probe import (  # noqa: E402
    case_map,
    load_raw_clouds,
    scale_pose,
)


DEFAULT_INPUT = (
    REPO_ROOT
    / "output"
    / "v20_shot_scale_consistency"
    / "full180_conf1_fair"
    / "v20_shot_scale_consistency.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "output" / "v20_shot_scale_consistency" / "scene_depth_scale"
METHODS = ("pelvis_first1", "torso_first1")
SCALE_BOUNDS = (0.0, 0.05, 0.10, 0.20, 0.30)


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
    parser.add_argument("--grid_steps", type=int, default=31)
    parser.add_argument("--scale_penalty_m", type=float, default=0.0)
    parser.add_argument("--solver", choices=("grid", "iterative_ls"), default="grid")
    parser.add_argument("--scale_iters", type=int, default=5)
    parser.add_argument("--min_absolute_gain_m", type=float, default=0.0)
    parser.add_argument("--min_relative_gain", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args()


def load_stream_cases(root: Path) -> dict[str, dict]:
    rows = {}
    for path in sorted(root.glob("v18_stream_shard_*_of_*.json")):
        for row in json.loads(path.read_text(encoding="utf-8"))["cases"]:
            rows[row["case_name"]] = row
    if len(rows) != 180:
        raise RuntimeError(f"Expected 180 stream cases, got {len(rows)}")
    return rows


def scale_cloud(cloud: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    return center[None] + float(scale) * (cloud - center[None])


def split_cloud(cloud: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    order = rng.permutation(len(cloud))
    middle = len(order) // 2
    return cloud[order[:middle]], cloud[order[middle:]]


def fit_scale(
    transform: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    center: np.ndarray,
    bound: float,
    steps: int,
    scale_penalty_m: float,
    min_absolute_gain_m: float,
    min_relative_gain: float,
) -> tuple[float, dict]:
    if bound <= 0.0:
        metric = scene_alignment_metrics(transform, source, target)
        return 1.0, {"objective_m": metric["trimmed_mean_m"], "evaluated": 1}
    best_scale = 1.0
    best_score = float("inf")
    baseline_score = float("nan")
    tree = cKDTree(target)
    for scale in np.linspace(1.0 - float(bound), 1.0 + float(bound), int(steps)):
        transformed = transform_points(
            transform, scale_cloud(source, center, float(scale))
        )
        distance, _ = tree.query(transformed, k=1, workers=1)
        finite = distance[np.isfinite(distance)]
        if not len(finite):
            continue
        keep = finite <= np.quantile(finite, 0.50)
        score = float(np.mean(finite[keep])) + float(scale_penalty_m) * abs(
            float(np.log(scale))
        )
        if abs(float(scale) - 1.0) < 1e-6:
            baseline_score = score
        if np.isfinite(score) and score < best_score:
            best_scale, best_score = float(scale), score
    absolute_gain = baseline_score - best_score
    relative_gain = absolute_gain / max(abs(baseline_score), 1e-6)
    accepted = bool(
        np.isfinite(absolute_gain)
        and absolute_gain >= float(min_absolute_gain_m)
        and relative_gain >= float(min_relative_gain)
    )
    if not accepted:
        best_scale, best_score = 1.0, baseline_score
    return best_scale, {
        "objective_m": best_score,
        "baseline_objective_m": baseline_score,
        "absolute_gain_m": absolute_gain,
        "relative_gain": relative_gain,
        "accepted": accepted,
        "evaluated": int(steps),
    }


def fit_scale_iterative(
    transform: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    center: np.ndarray,
    bound: float,
    iterations: int,
    min_absolute_gain_m: float,
    min_relative_gain: float,
) -> tuple[float, dict]:
    if bound <= 0.0:
        metric = scene_alignment_metrics(transform, source, target)
        return 1.0, {"objective_m": metric["trimmed_mean_m"], "iterations": []}
    tree = cKDTree(target)
    base = transform_points(transform, center[None])[0]
    direction = (source - center[None]) @ transform[:3, :3].T
    scale = 1.0
    history = []
    for iteration in range(int(iterations)):
        transformed = base[None] + scale * direction
        distance, nearest = tree.query(transformed, k=1, workers=1)
        finite = np.isfinite(distance)
        if int(finite.sum()) < 64:
            break
        threshold = float(np.quantile(distance[finite], 0.60))
        valid = np.flatnonzero(finite & (distance <= threshold))
        denominator = np.sum(direction[valid] * direction[valid], axis=1)
        keep = denominator > 1e-6
        valid = valid[keep]
        denominator = denominator[keep]
        if len(valid) < 64:
            break
        numerator = np.sum(
            direction[valid] * (target[nearest[valid]] - base[None]), axis=1
        )
        proposals = numerator / denominator
        proposals = proposals[np.isfinite(proposals)]
        if not len(proposals):
            break
        proposed = float(np.median(proposals))
        updated = float(np.clip(proposed, 1.0 - float(bound), 1.0 + float(bound)))
        history.append(
            {
                "iteration": iteration,
                "scale": updated,
                "proposal": proposed,
                "pairs": int(len(proposals)),
                "median_distance_m": float(np.median(distance[valid])),
            }
        )
        if abs(updated - scale) < 1e-4:
            scale = updated
            break
        scale = updated
    baseline = scene_alignment_metrics(transform, source, target)
    metric = scene_alignment_metrics(transform, scale_cloud(source, center, scale), target)
    absolute_gain = float(baseline["trimmed_mean_m"] - metric["trimmed_mean_m"])
    relative_gain = absolute_gain / max(abs(float(baseline["trimmed_mean_m"])), 1e-6)
    accepted = bool(
        absolute_gain >= float(min_absolute_gain_m)
        and relative_gain >= float(min_relative_gain)
    )
    if not accepted:
        scale, metric = 1.0, baseline
    return scale, {
        "objective_m": metric["trimmed_mean_m"],
        "baseline_objective_m": baseline["trimmed_mean_m"],
        "absolute_gain_m": absolute_gain,
        "relative_gain": relative_gain,
        "accepted": accepted,
        "iterations": history,
    }


def aggregate(rows: list[dict], method: str, bound_key: str) -> dict:
    values = [row["methods"][method][bound_key] for row in rows]
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
        "pointmap_scale": distribution([row["pointmap_scale"] for row in values]),
        "scene_improved_rate": float(np.mean(scene < fixed_scene)),
        "all_three_improved_rate": float(
            np.mean((scene < fixed_scene) & (camera < fixed_camera) & (human < fixed_human))
        ),
        "scene_harmful_rate_010m": float(np.mean(scene > fixed_scene + 0.10)),
    }


def build_summary(rows: list[dict]) -> dict:
    return {
        method: {
            f"q{int(round(100 * bound)):02d}": aggregate(
                rows, method, f"q{int(round(100 * bound)):02d}"
            )
            for bound in SCALE_BOUNDS
        }
        for method in METHODS
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads(args.input_report.read_text(encoding="utf-8"))
    input_cases = {row["case_name"]: row for row in report["cases"]}
    v10 = case_map(args.v10_report)
    streams = load_stream_cases(args.stream_dir)
    rows = []
    helper_args = argparse.Namespace(
        point_samples=int(args.point_samples),
        raw_confidence_threshold=float(args.raw_confidence_threshold),
        mask_dilate=int(args.mask_dilate),
    )
    for index, case_name in enumerate(sorted(input_cases)):
        row = input_cases[case_name]
        with np.load(streams[case_name]["cache_path"]) as stream:
            raw_poses = np.stack([stream["old_pose"][-1], stream["new_pose"]]).astype(np.float32)
        local_dir = Path(v10[case_name]["paths"]["human3r_local_reset"])
        rng = np.random.default_rng(int(args.seed) + index)
        raw_old, raw_new = load_raw_clouds(local_dir, raw_poses, helper_args, rng)
        methods = {}
        fixed_transform = np.asarray(row["fixed"]["transform"], dtype=np.float32)
        raw_old_solve, raw_old_eval = split_cloud(raw_old, rng)
        raw_new_solve, raw_new_eval = split_cloud(raw_new, rng)
        fixed = {
            **row["fixed"],
            "scene": scene_alignment_metrics(fixed_transform, raw_new_eval, raw_old_eval),
        }
        for method in METHODS:
            source = row["methods"][method]
            old_scale = float(source["old_scale"])
            new_scale = float(source["new_scale"])
            old_solve, old_eval = raw_old_solve * old_scale, raw_old_eval * old_scale
            new_solve, new_eval = raw_new_solve * new_scale, raw_new_eval * new_scale
            new_pose = scale_pose(raw_poses[1], new_scale)
            center = new_pose[:3, 3]
            initial = source["variants"]["b00"]
            transform = np.asarray(initial["transform"], dtype=np.float32)
            variants = {}
            for bound in SCALE_BOUNDS:
                if args.solver == "iterative_ls":
                    scale, diagnostics = fit_scale_iterative(
                        transform,
                        new_solve,
                        old_solve,
                        center,
                        bound,
                        int(args.scale_iters),
                        float(args.min_absolute_gain_m),
                        float(args.min_relative_gain),
                    )
                else:
                    scale, diagnostics = fit_scale(
                        transform,
                        new_solve,
                        old_solve,
                        center,
                        bound,
                        int(args.grid_steps),
                        float(args.scale_penalty_m),
                        float(args.min_absolute_gain_m),
                        float(args.min_relative_gain),
                    )
                scene = scene_alignment_metrics(
                    transform, scale_cloud(new_eval, center, scale), old_eval
                )
                variants[f"q{int(round(100 * bound)):02d}"] = {
                    "pointmap_scale": scale,
                    "camera": initial["camera"],
                    "human": initial["human"],
                    "scene": scene,
                    "fit": diagnostics,
                }
            methods[method] = variants
        rows.append({"case_name": case_name, "source": row["source"], "fixed": fixed, "methods": methods})
        print(f"V20 scene depth scale {index + 1}/{len(input_cases)}", flush=True)
    output = {
        "experiment": "V20 bounded post-cut background pointmap depth-scale refinement",
        "case_count": len(rows),
        "protocol": {
            "camera_and_human_boundary_fixed": True,
            "fit_eval_point_split": True,
            "pointmap_scale_bounds": list(SCALE_BOUNDS),
            "scale_penalty_m": float(args.scale_penalty_m),
            "solver": str(args.solver),
            "min_absolute_gain_m": float(args.min_absolute_gain_m),
            "min_relative_gain": float(args.min_relative_gain),
            "learned_components": False,
        },
        "overall": build_summary(rows),
        "by_source": {
            source: build_summary([row for row in rows if row["source"] == source])
            for source in sorted({row["source"] for row in rows})
        },
        "cases": rows,
    }
    path = args.output_dir / "v20_scene_depth_scale_refinement.json"
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    print(f">> wrote {path}", flush=True)


if __name__ == "__main__":
    main()
