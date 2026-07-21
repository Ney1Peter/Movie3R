#!/usr/bin/env python3
"""V16 Stage-1 partial oracle for bounded human-aware rotation residuals.

The probe starts from the deployed Fixed Explicit transform, changes rotation
only, and compares preserving its translation against scene-only translation
re-solving with the corrected rotation held fixed. GT is used only to construct
the partial-oracle rotation variants and to evaluate them.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for path in (str(REPO_ROOT), str(SRC_ROOT), str(SCRIPTS_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from dust3r.utils.smpl_layer import SMPL_Layer  # noqa: E402
from v10_boundary_gauge_partial_oracle_probe import (  # noqa: E402
    correct_gravity,
    correct_heading,
    evaluate_transform,
    load_predicted_joints,
    make_transform,
    torso_frame,
    transform_diagnostics,
)
from v10_implicit_explicit_cross_shot_probe import (  # noqa: E402
    background_cloud,
    history_background_cloud,
    transform_points,
)
from v10_oracle_candidate_selection_probe import (  # noqa: E402
    camera_errors,
    predicted_poses,
    transform_camera_poses,
)
from v10_oracle_state_vs_gauge_probe import rotation_error_deg  # noqa: E402


DEFAULT_REPORT = (
    REPO_ROOT
    / "output"
    / "v10_candidate_selection"
    / "oracle_gt_4source"
    / "oracle_candidate_selection_metrics.json"
)
DEFAULT_GT_CACHE = (
    REPO_ROOT
    / "output"
    / "v10_candidate_selection"
    / "boundary_gauge_partial_oracle"
    / "gt_cache"
)
DEFAULT_OUTPUT = REPO_ROOT / "output" / "v16_human_aware_rotation_residual" / "partial_oracle"

BASE_METHODS = (
    "fixed_explicit",
    "gt_delta_yaw",
    "gt_gravity",
    "gt_torso_heading",
    "gt_torso_gravity",
    "full_gt_rotation",
)
METHODS = tuple(
    method
    for base in BASE_METHODS
    for method in ((base,) if base == "fixed_explicit" else (f"{base}_keep_t0", f"{base}_resolve_t"))
) + ("full_boundary_oracle",)

LABELS = {
    "fixed_explicit": "Fixed Explicit",
    "gt_delta_yaw_keep_t0": "Fixed + GT Delta Yaw, keep t0",
    "gt_delta_yaw_resolve_t": "Fixed + GT Delta Yaw, resolve t",
    "gt_gravity_keep_t0": "Fixed + GT Gravity, keep t0",
    "gt_gravity_resolve_t": "Fixed + GT Gravity, resolve t",
    "gt_torso_heading_keep_t0": "Fixed + GT Torso Heading, keep t0",
    "gt_torso_heading_resolve_t": "Fixed + GT Torso Heading, resolve t",
    "gt_torso_gravity_keep_t0": "Fixed + GT Torso + Gravity, keep t0",
    "gt_torso_gravity_resolve_t": "Fixed + GT Torso + Gravity, resolve t",
    "full_gt_rotation_keep_t0": "Fixed + Full GT Rotation, keep t0",
    "full_gt_rotation_resolve_t": "Fixed + Full GT Rotation, resolve t",
    "full_boundary_oracle": "Boundary Oracle",
}

INITIAL_ROTATION_BUCKETS = (
    ("lt10", 0.0, 10.0),
    ("10_30", 10.0, 30.0),
    ("30_60", 30.0, 60.0),
    ("ge60", 60.0, float("inf")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--gt_cache", type=Path, default=DEFAULT_GT_CACHE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--boundary", type=int, default=2)
    parser.add_argument("--cloud_points_per_frame", type=int, default=5000)
    parser.add_argument("--translation_iters", type=int, default=8)
    parser.add_argument("--translation_max_distance", type=float, default=0.60)
    parser.add_argument("--translation_min_distance", type=float, default=0.12)
    parser.add_argument("--cases_per_source", type=int, default=0)
    return parser.parse_args()


def build_pred_smpl_layer(device: torch.device) -> SMPL_Layer:
    layer = SMPL_Layer(
        type="smplx",
        gender="neutral",
        num_betas=10,
        kid=False,
        person_center="head",
    ).to(device)
    layer.eval()
    return layer


def load_cases(path: Path, cases_per_source: int) -> list[dict]:
    cases = list(json.loads(path.read_text(encoding="utf-8"))["cases"])
    if cases_per_source <= 0:
        return cases
    counts: dict[str, int] = defaultdict(int)
    selected = []
    for case in cases:
        source = str(case["record"]["source"])
        if counts[source] >= cases_per_source:
            continue
        selected.append(case)
        counts[source] += 1
    return selected


def gt_cache_path(root: Path, case: dict) -> Path:
    return root / f"{case['case_name']}.npz"


def twist_rotation(delta: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Return the pure twist component of delta around axis in target gauge."""
    axis = np.asarray(axis, dtype=np.float64)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    quat = Rotation.from_matrix(np.asarray(delta, dtype=np.float64)).as_quat()
    vector, scalar = quat[:3], float(quat[3])
    projected = axis * float(np.dot(vector, axis))
    twist = np.concatenate([projected, [scalar]])
    norm = float(np.linalg.norm(twist))
    if norm < 1e-8:
        return np.eye(3, dtype=np.float32)
    twist /= norm
    return Rotation.from_quat(twist).as_matrix().astype(np.float32)


def solve_scene_translation_fixed_rotation(
    rotation: np.ndarray,
    initial_translation: np.ndarray,
    source_cloud: np.ndarray,
    target_cloud: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict]:
    """Translation-only robust ICP initialized from the explicit candidate."""
    translation = np.asarray(initial_translation, dtype=np.float32).copy()
    initial = translation.copy()
    if len(source_cloud) < 32 or len(target_cloud) < 32:
        return translation, {"status": "too_few_background_points", "iterations": []}
    tree = cKDTree(target_cloud)
    history = []
    for iteration in range(int(args.translation_iters)):
        transformed = np.einsum("ij,nj->ni", rotation, source_cloud) + translation[None]
        distances, indices = tree.query(transformed, k=1, workers=-1)
        alpha = iteration / max(int(args.translation_iters) - 1, 1)
        max_distance = (
            (1.0 - alpha) * float(args.translation_max_distance)
            + alpha * float(args.translation_min_distance)
        )
        valid = np.isfinite(distances) & (distances < max_distance)
        if int(valid.sum()) < 32:
            history.append({"iteration": iteration, "pairs": int(valid.sum()), "status": "too_few_pairs"})
            break
        ids = np.where(valid)[0]
        trim = float(np.quantile(distances[ids], 0.70))
        ids = ids[distances[ids] <= trim]
        if len(ids) < 32:
            break
        residual = target_cloud[indices[ids]] - transformed[ids]
        weights = 1.0 / np.maximum(distances[ids], 0.01)
        delta = np.average(residual, axis=0, weights=weights).astype(np.float32)
        translation += delta
        history.append(
            {
                "iteration": iteration,
                "pairs": int(len(ids)),
                "median_distance_m": float(np.median(distances[ids])),
                "delta_translation_m": float(np.linalg.norm(delta)),
            }
        )
    return translation, {
        "status": "ok",
        "initial_translation": initial.astype(float).tolist(),
        "residual_from_t0_m": float(np.linalg.norm(translation - initial)),
        "iterations": history,
        "human_root_used": False,
    }


def evaluate(
    name: str,
    transform: np.ndarray,
    pred_poses: np.ndarray,
    target_poses: np.ndarray,
    oracle_transform: np.ndarray,
    predicted_joints: np.ndarray,
    boundary: int,
    target_human_root: np.ndarray,
    diagnostics: dict | None = None,
) -> dict:
    row = evaluate_transform(
        "full_boundary_oracle" if name == "full_boundary_oracle" else "current_best_explicit",
        transform,
        pred_poses,
        target_poses,
        oracle_transform,
        predicted_joints,
        boundary,
        target_human_root,
        diagnostics,
    )
    row["name"] = name
    row["label"] = LABELS[name]
    return row


def run_case(
    case: dict,
    args: argparse.Namespace,
    pred_layer: SMPL_Layer,
    device: torch.device,
    case_index: int,
) -> dict:
    local_dir = Path(case["paths"]["human3r_local_reset"])
    pred_poses = predicted_poses(local_dir)
    predicted_joints = load_predicted_joints(local_dir, pred_layer, device)
    cache_path = gt_cache_path(args.gt_cache, case)
    if not cache_path.is_file():
        raise FileNotFoundError(f"Missing V10 GT cache: {cache_path}")
    with np.load(cache_path) as gt:
        gt_poses = gt["gt_poses"].astype(np.float32)
        gt_joints_world = gt["gt_joints_world"].astype(np.float32)

    boundary = int(args.boundary)
    gauge_target = pred_poses[0] @ np.linalg.inv(gt_poses[0])
    gauge_local = pred_poses[boundary] @ np.linalg.inv(gt_poses[boundary])
    target_poses = np.stack([(gauge_target @ pose).astype(np.float32) for pose in gt_poses])
    oracle_transform = (target_poses[boundary] @ np.linalg.inv(pred_poses[boundary])).astype(np.float32)
    target_joints = np.stack([transform_points(gauge_target, joints) for joints in gt_joints_world])
    local_joints = np.stack([transform_points(gauge_local, joints) for joints in gt_joints_world])
    target_torso = torso_frame(target_joints[boundary])
    local_torso = torso_frame(local_joints[boundary])
    up_target, heading_target = target_torso[:, 1], target_torso[:, 2]
    up_local, heading_local = local_torso[:, 1], local_torso[:, 2]

    fixed = case["fixed_explicit"]
    fixed_transform = np.asarray(fixed["transform"], dtype=np.float32)
    fixed_rotation = fixed_transform[:3, :3]
    fixed_translation = fixed_transform[:3, 3]

    target_cloud, target_debug = history_background_cloud(
        local_dir, list(range(boundary)), int(args.cloud_points_per_frame)
    )
    source_cloud, source_debug = background_cloud(
        local_dir,
        boundary,
        int(args.cloud_points_per_frame),
        seed=20260721 + case_index,
    )

    delta_to_gt = oracle_transform[:3, :3] @ fixed_rotation.T
    yaw_rotation = twist_rotation(delta_to_gt, up_target) @ fixed_rotation
    gravity_rotation = correct_gravity(fixed_rotation, up_local, up_target)
    mapped_up = fixed_rotation @ up_local
    torso_rotation = correct_heading(fixed_rotation, heading_local, heading_target, mapped_up)
    torso_gravity_rotation = correct_heading(gravity_rotation, heading_local, heading_target, up_target)
    rotation_variants = {
        "gt_delta_yaw": yaw_rotation.astype(np.float32),
        "gt_gravity": gravity_rotation.astype(np.float32),
        "gt_torso_heading": torso_rotation.astype(np.float32),
        "gt_torso_gravity": torso_gravity_rotation.astype(np.float32),
        "full_gt_rotation": oracle_transform[:3, :3].astype(np.float32),
    }

    target_human_root = target_joints[boundary, 0]
    variants = {
        "fixed_explicit": evaluate(
            "fixed_explicit",
            fixed_transform,
            pred_poses,
            target_poses,
            oracle_transform,
            predicted_joints,
            boundary,
            target_human_root,
            {"candidate": fixed["name"]},
        )
    }
    for base, rotation in rotation_variants.items():
        keep_transform = make_transform(rotation, fixed_translation)
        resolved_t, solver_debug = solve_scene_translation_fixed_rotation(
            rotation, fixed_translation, source_cloud, target_cloud, args
        )
        resolved_transform = make_transform(rotation, resolved_t)
        variants[f"{base}_keep_t0"] = evaluate(
            f"{base}_keep_t0",
            keep_transform,
            pred_poses,
            target_poses,
            oracle_transform,
            predicted_joints,
            boundary,
            target_human_root,
            {"translation": "preserved fixed explicit t0"},
        )
        variants[f"{base}_resolve_t"] = evaluate(
            f"{base}_resolve_t",
            resolved_transform,
            pred_poses,
            target_poses,
            oracle_transform,
            predicted_joints,
            boundary,
            target_human_root,
            {"translation_solver": solver_debug},
        )
    variants["full_boundary_oracle"] = evaluate(
        "full_boundary_oracle",
        oracle_transform,
        pred_poses,
        target_poses,
        oracle_transform,
        predicted_joints,
        boundary,
        target_human_root,
    )
    initial_rotation_error = rotation_error_deg(fixed_transform, oracle_transform)
    initial_translation_error = float(np.linalg.norm(fixed_translation - oracle_transform[:3, 3]))
    return {
        "case_name": case["case_name"],
        "record": case["record"],
        "initial_error": {
            "rotation_deg": initial_rotation_error,
            "translation_m": initial_translation_error,
        },
        "oracle_transform": oracle_transform.astype(float).tolist(),
        "variants": variants,
        "pointmap_debug": {"target": target_debug, "source": source_debug},
    }


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def aggregate_method(cases: list[dict], method: str) -> dict:
    rows = [case["variants"][method] for case in cases]
    camera = [row["camera"] for row in rows]
    transform = [row["transform_error"] for row in rows]
    translation = [row["mean_translation_m"] for row in camera]
    rotation = [row["mean_rotation_deg"] for row in camera]
    axis_r = np.asarray([row["yaw_pitch_roll_abs_deg"] for row in transform], dtype=np.float64)
    axis_t = np.asarray([row["translation_xyz_abs_m"] for row in transform], dtype=np.float64)
    return {
        "count": len(rows),
        "translation_mean_m": float(np.mean(translation)),
        "translation_median_m": float(np.median(translation)),
        "translation_p90_m": percentile(translation, 90),
        "translation_p95_m": percentile(translation, 95),
        "rotation_mean_deg": float(np.mean(rotation)),
        "rotation_median_deg": float(np.median(rotation)),
        "rotation_p90_deg": percentile(rotation, 90),
        "rotation_p95_deg": percentile(rotation, 95),
        "yaw_mean_abs_deg": float(axis_r[:, 0].mean()),
        "pitch_mean_abs_deg": float(axis_r[:, 1].mean()),
        "roll_mean_abs_deg": float(axis_r[:, 2].mean()),
        "translation_x_mean_abs_m": float(axis_t[:, 0].mean()),
        "translation_y_mean_abs_m": float(axis_t[:, 1].mean()),
        "translation_z_mean_abs_m": float(axis_t[:, 2].mean()),
        "success_strict_rate": float(np.mean([row["success_strict"] for row in camera])),
        "success_relaxed_rate": float(np.mean([row["success_relaxed"] for row in camera])),
        "catastrophic_rate": float(np.mean([row["catastrophic"] for row in camera])),
        "human_root_jump_mean_m": float(np.mean([row["human_jump"]["world_root_jump_m"] for row in rows])),
        "human_torso_jump_mean_deg": float(
            np.mean([row["human_jump"]["torso_orientation_jump_deg"] for row in rows])
        ),
    }


def aggregate(cases: list[dict]) -> dict:
    return {method: aggregate_method(cases, method) for method in METHODS}


def source_groups(cases: list[dict]) -> dict:
    return {
        source: aggregate([case for case in cases if str(case["record"]["source"]) == source])
        for source in sorted({str(case["record"]["source"]) for case in cases})
    }


def rotation_groups(cases: list[dict]) -> dict:
    groups = {}
    for name, lower, upper in INITIAL_ROTATION_BUCKETS:
        subset = [
            case
            for case in cases
            if lower <= float(case["initial_error"]["rotation_deg"]) < upper
        ]
        if subset:
            groups[name] = aggregate(subset)
    return groups


def automatic_decision(overall: dict) -> dict:
    fixed = overall["fixed_explicit"]
    torso = overall["gt_torso_heading_resolve_t"]
    torso_gravity = overall["gt_torso_gravity_resolve_t"]
    full_rotation = overall["full_gt_rotation_resolve_t"]
    heading_gain = fixed["rotation_mean_deg"] - torso["rotation_mean_deg"]
    heading_tail_gain = fixed["rotation_p90_deg"] - torso["rotation_p90_deg"]
    translation_change = torso["translation_mean_m"] - fixed["translation_mean_m"]
    full_rotation_gain = fixed["rotation_mean_deg"] - full_rotation["rotation_mean_deg"]
    fraction = heading_gain / max(full_rotation_gain, 1e-8)
    proceed = bool(heading_gain >= 2.0 and heading_tail_gain > 0.0)
    return {
        "proceed_to_predicted_torso_geometry": proceed,
        "gt_torso_rotation_gain_deg": heading_gain,
        "gt_torso_rotation_p90_gain_deg": heading_tail_gain,
        "gt_torso_fraction_of_full_rotation_gain": fraction,
        "gt_torso_resolved_translation_change_m": translation_change,
        "gt_torso_gravity_rotation_mean_deg": torso_gravity["rotation_mean_deg"],
        "full_gt_rotation_resolved_translation_m": full_rotation["translation_mean_m"],
        "reason": (
            "GT torso retains a material post-explicit rotation signal."
            if proceed
            else "GT torso adds too little post-explicit rotation value; stop before learned modules."
        ),
    }


def write_csv(path: Path, metrics: dict) -> None:
    rows = []
    for method in METHODS:
        rows.append({"method": method, "label": LABELS[method], **metrics[method]})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, report: dict) -> None:
    overall = report["overall"]
    lines = [
        "# V16 Rotation Residual Partial Oracle",
        "",
        "- Coarse candidate: V15/V14 deployment baseline `Fixed Explicit`.",
        "- GT is used only for rotation partial oracles and evaluation; translation never uses GT or human root.",
        "- `resolve t` holds the corrected rotation fixed and runs scene-only translation ICP from `t0`.",
        "",
        "| Method | T mean | R mean | Yaw | Pitch | Roll | T P90 | R P90 | Catastrophic |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = overall[method]
        lines.append(
            f"| {LABELS[method]} | {row['translation_mean_m']:.4f} | {row['rotation_mean_deg']:.2f} | "
            f"{row['yaw_mean_abs_deg']:.2f} | {row['pitch_mean_abs_deg']:.2f} | {row['roll_mean_abs_deg']:.2f} | "
            f"{row['translation_p90_m']:.4f} | {row['rotation_p90_deg']:.2f} | "
            f"{100.0 * row['catastrophic_rate']:.1f}% |"
        )
    decision = report["automatic_decision"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Proceed to predicted torso geometry: `{decision['proceed_to_predicted_torso_geometry']}`.",
            f"- GT torso mean rotation gain: `{decision['gt_torso_rotation_gain_deg']:.2f} deg`.",
            f"- GT torso P90 rotation gain: `{decision['gt_torso_rotation_p90_gain_deg']:.2f} deg`.",
            f"- Fraction of full rotation-oracle gain: `{100.0 * decision['gt_torso_fraction_of_full_rotation_gain']:.1f}%`.",
            f"- Translation change after scene-only re-solving: `{decision['gt_torso_resolved_translation_change_m']:+.4f} m`.",
            "",
            "## By Source",
            "",
        ]
    )
    for source, group in report["by_source"].items():
        fixed = group["fixed_explicit"]
        torso = group["gt_torso_heading_resolve_t"]
        lines.append(
            f"- **{source}**: Fixed `{fixed['translation_mean_m']:.3f} m / {fixed['rotation_mean_deg']:.2f} deg`; "
            f"GT torso + resolve t `{torso['translation_mean_m']:.3f} m / {torso['rotation_mean_deg']:.2f} deg`."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("V16 Human3R/SMPL-X evaluation must run on CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args.candidate_report, int(args.cases_per_source))
    missing = [case["case_name"] for case in cases if not gt_cache_path(args.gt_cache, case).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} V10 GT cache files; first={missing[0]}")
    device = torch.device(args.device)
    pred_layer = build_pred_smpl_layer(device)
    reports = []
    progress = args.output_dir / "case_progress.jsonl"
    if progress.exists():
        progress.unlink()
    started = time.perf_counter()
    for index, case in enumerate(cases):
        report = run_case(case, args, pred_layer, device, index)
        reports.append(report)
        fixed = report["variants"]["fixed_explicit"]["camera"]
        torso = report["variants"]["gt_torso_heading_resolve_t"]["camera"]
        with progress.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "case_name": report["case_name"],
                        "source": report["record"]["source"],
                        "fixed_t": fixed["mean_translation_m"],
                        "fixed_r": fixed["mean_rotation_deg"],
                        "torso_t": torso["mean_translation_m"],
                        "torso_r": torso["mean_rotation_deg"],
                    }
                )
                + "\n"
            )
        print(
            f">> [{index + 1}/{len(cases)}] {report['case_name']} "
            f"R {fixed['mean_rotation_deg']:.2f}->{torso['mean_rotation_deg']:.2f}",
            flush=True,
        )
        torch.cuda.empty_cache()

    overall = aggregate(reports)
    output = {
        "experiment": "V16 Explicit-First Human-Aware Rotation Residual Partial Oracle",
        "case_count": len(reports),
        "protocol": {
            "coarse_candidate": "V15/V14 Fixed Explicit (human_mean_pointmap_history_standard)",
            "gt_camera_use": "partial-oracle rotation construction and evaluation only",
            "gt_translation_used": False,
            "gt_depth_used": False,
            "human_root_translation_used": False,
            "translation_resolver": "fixed-rotation scene-only pointmap translation ICP initialized at t0",
            "human3r_rerun": False,
            "human3r_state_modified": False,
        },
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "overall": overall,
        "by_source": source_groups(reports),
        "by_initial_rotation": rotation_groups(reports),
        "automatic_decision": automatic_decision(overall),
        "elapsed_seconds": time.perf_counter() - started,
        "cases": reports,
    }
    metrics_path = args.output_dir / "v16_partial_oracle.json"
    metrics_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(args.output_dir / "v16_partial_oracle_summary.csv", overall)
    write_markdown(args.output_dir / "v16_partial_oracle_summary.md", output)
    print(json.dumps({"overall": overall, "decision": output["automatic_decision"]}, indent=2), flush=True)
    print(f">> wrote {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
