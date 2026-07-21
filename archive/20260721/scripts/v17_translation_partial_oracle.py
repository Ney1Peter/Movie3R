#!/usr/bin/env python3
"""V17 Stage-0 translation direction/scale partial-oracle audit.

Translation is represented by the post-cut boundary camera center in the last
pre-cut camera frame. This representation is invariant to a global world-gauge
change and matches the translation-direction quantity diagnosed in V15.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from v10_oracle_candidate_selection_probe import predicted_poses  # noqa: E402
from v10_implicit_explicit_cross_shot_probe import (  # noqa: E402
    background_cloud,
    history_background_cloud,
)
from v13_scene_coordinate_oracle import direct_transform_error  # noqa: E402
from v16_human_torso_candidates import (  # noqa: E402
    make_transform,
    scene_translation_fixed_rotation,
)


DEFAULT_V16 = REPO_ROOT / "output" / "v16_human_aware_rotation_residual" / "candidate_cache"
DEFAULT_V10 = (
    REPO_ROOT
    / "output"
    / "v10_candidate_selection"
    / "oracle_gt_4source"
    / "oracle_candidate_selection_metrics.json"
)
DEFAULT_PARTIAL = (
    REPO_ROOT
    / "output"
    / "v16_human_aware_rotation_residual"
    / "partial_oracle"
    / "v16_partial_oracle.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "output" / "v17_direct_vs_factorized" / "partial_oracle"

ROTATIONS = ("fixed_rotation", "torso_rotation", "gt_rotation")
VARIANTS = (
    "current_direction_current_scale",
    "gt_direction_current_scale",
    "current_direction_gt_scale",
    "vggt_direction_current_scale",
    "vggt_direction_gt_scale",
    "gt_direction_gt_scale",
    "scene_translation_resolve",
    "full_gt_transform_translation",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v16_dir", type=Path, default=DEFAULT_V16)
    parser.add_argument("--v10_report", type=Path, default=DEFAULT_V10)
    parser.add_argument("--v16_partial", type=Path, default=DEFAULT_PARTIAL)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--boundary", type=int, default=2)
    parser.add_argument("--cloud_points_per_frame", type=int, default=3000)
    parser.add_argument("--translation_iters", type=int, default=8)
    parser.add_argument("--translation_max_distance", type=float, default=0.60)
    parser.add_argument("--translation_min_distance", type=float, default=0.12)
    return parser.parse_args()


def load_v16_cases(root: Path) -> list[dict]:
    cases = []
    for path in sorted(glob.glob(str(root / "v16_candidates_shard_*_of_*.json"))):
        cases.extend(json.loads(Path(path).read_text(encoding="utf-8"))["cases"])
    names = [str(case["case_name"]) for case in cases]
    if len(names) != 180 or len(names) != len(set(names)):
        raise RuntimeError(f"Expected 180 unique V16 cases, got {len(names)}/{len(set(names))}")
    return cases


def load_maps(args: argparse.Namespace) -> tuple[dict[str, Path], dict[str, dict]]:
    v10 = json.loads(args.v10_report.read_text(encoding="utf-8"))
    local_dirs = {
        str(case["case_name"]): Path(case["paths"]["human3r_local_reset"])
        for case in v10["cases"]
    }
    partial = json.loads(args.v16_partial.read_text(encoding="utf-8"))
    partial_cases = {str(case["case_name"]): case for case in partial["cases"]}
    return local_dirs, partial_cases


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    return (vector / norm).astype(np.float32)


def relative_boundary_pose(old_pose: np.ndarray, transform: np.ndarray, pred_pose: np.ndarray) -> np.ndarray:
    return (np.linalg.inv(old_pose) @ transform @ pred_pose).astype(np.float32)


def relative_translation(old_pose: np.ndarray, transform: np.ndarray, pred_pose: np.ndarray) -> np.ndarray:
    return relative_boundary_pose(old_pose, transform, pred_pose)[:3, 3].astype(np.float32)


def transform_from_rotation_and_relative_translation(
    rotation: np.ndarray,
    translation_old_camera: np.ndarray,
    old_pose: np.ndarray,
    pred_pose: np.ndarray,
) -> np.ndarray:
    target_center_world = (
        old_pose[:3, :3] @ np.asarray(translation_old_camera, dtype=np.float32)
        + old_pose[:3, 3]
    )
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_center_world - rotation @ pred_pose[:3, 3]
    return transform


def direction_error_deg(estimated: np.ndarray, target: np.ndarray) -> float:
    cosine = float(np.dot(normalize(estimated), normalize(target)))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def translation_metrics(
    estimated: np.ndarray,
    target: np.ndarray,
    transform: np.ndarray,
    pred_pose: np.ndarray,
    target_pose: np.ndarray,
) -> dict:
    delta = np.asarray(estimated, dtype=np.float32) - np.asarray(target, dtype=np.float32)
    estimated_scale = float(np.linalg.norm(estimated))
    target_scale = float(np.linalg.norm(target))
    row = direct_transform_error(transform, pred_pose, target_pose)
    return {
        **row,
        "translation_direction_error_deg": direction_error_deg(estimated, target),
        "translation_scale_abs_error_m": abs(estimated_scale - target_scale),
        "translation_scale_log_abs": abs(
            math.log(max(estimated_scale, 1e-8) / max(target_scale, 1e-8))
        ),
        "translation_vector_error_m": float(np.linalg.norm(delta)),
        "translation_error_old_camera_xyz_m": np.abs(delta).astype(float).tolist(),
        "translation_view_direction_error_m": float(abs(delta[2])),
        "translation_transverse_error_m": float(np.linalg.norm(delta[:2])),
        "translation_horizontal_error_m": float(abs(delta[0])),
        "translation_vertical_error_m": float(abs(delta[1])),
        "estimated_translation_old_camera": np.asarray(estimated, dtype=float).tolist(),
        "target_translation_old_camera": np.asarray(target, dtype=float).tolist(),
        "transform": transform.astype(float).tolist(),
    }


def stored_transform(case: dict, group: str, key: str) -> np.ndarray:
    return np.asarray(case[group][key]["transform"], dtype=np.float32)


def partial_transform(partial_case: dict, key: str) -> np.ndarray:
    return np.asarray(partial_case["variants"][key]["transform"], dtype=np.float32)


def run_case(
    case: dict,
    local_dir: Path,
    partial_case: dict,
    args: argparse.Namespace,
) -> dict:
    boundary = int(args.boundary)
    poses = predicted_poses(local_dir)
    old_pose = poses[boundary - 1]
    pred_pose = poses[boundary]
    fixed = stored_transform(case, "baselines", "fixed_explicit")
    torso_keep = stored_transform(case, "fixed_candidates", "fixed_torso_motion_1f_keep_t0")
    torso_resolve = stored_transform(case, "fixed_candidates", "fixed_torso_motion_1f_resolve_t")
    vggt = stored_transform(case, "baselines", "v15_coarse")
    oracle = stored_transform(case, "baselines", "boundary_oracle")
    gt_keep = make_transform(oracle[:3, :3], fixed[:3, 3])
    target_cloud, _ = history_background_cloud(
        local_dir, list(range(boundary)), int(args.cloud_points_per_frame)
    )
    source_cloud, _ = background_cloud(
        local_dir, boundary, int(args.cloud_points_per_frame), seed=20260723
    )
    gt_resolved_translation, _ = scene_translation_fixed_rotation(
        oracle[:3, :3], fixed[:3, 3], source_cloud, target_cloud, args
    )
    gt_resolve = make_transform(oracle[:3, :3], gt_resolved_translation)
    target_pose = oracle @ pred_pose
    target_translation = relative_translation(old_pose, oracle, pred_pose)
    target_direction = normalize(target_translation)
    target_scale = float(np.linalg.norm(target_translation))
    vggt_direction = normalize(relative_translation(old_pose, vggt, pred_pose))

    configs = {
        "fixed_rotation": {
            "rotation": fixed[:3, :3],
            "current": fixed,
            "resolved": fixed,
        },
        "torso_rotation": {
            "rotation": torso_keep[:3, :3],
            "current": torso_keep,
            "resolved": torso_resolve,
        },
        "gt_rotation": {
            "rotation": oracle[:3, :3],
            "current": gt_keep,
            "resolved": gt_resolve,
        },
    }
    outputs = {}
    for rotation_name, config in configs.items():
        rotation = config["rotation"]
        current_translation = relative_translation(old_pose, config["current"], pred_pose)
        current_direction = normalize(current_translation)
        current_scale = float(np.linalg.norm(current_translation))
        vectors = {
            "current_direction_current_scale": current_translation,
            "gt_direction_current_scale": target_direction * current_scale,
            "current_direction_gt_scale": current_direction * target_scale,
            "vggt_direction_current_scale": vggt_direction * current_scale,
            "vggt_direction_gt_scale": vggt_direction * target_scale,
            "gt_direction_gt_scale": target_translation,
        }
        rows = {}
        for variant, vector in vectors.items():
            transform = transform_from_rotation_and_relative_translation(
                rotation, vector, old_pose, pred_pose
            )
            estimated = relative_translation(old_pose, transform, pred_pose)
            rows[variant] = translation_metrics(
                estimated, target_translation, transform, pred_pose, target_pose
            )

        resolved = config["resolved"]
        rows["scene_translation_resolve"] = translation_metrics(
            relative_translation(old_pose, resolved, pred_pose),
            target_translation,
            resolved,
            pred_pose,
            target_pose,
        )
        full_gt_t = np.eye(4, dtype=np.float32)
        full_gt_t[:3, :3] = rotation
        full_gt_t[:3, 3] = oracle[:3, 3]
        rows["full_gt_transform_translation"] = translation_metrics(
            relative_translation(old_pose, full_gt_t, pred_pose),
            target_translation,
            full_gt_t,
            pred_pose,
            target_pose,
        )
        outputs[rotation_name] = rows

    return {
        "case_name": case["case_name"],
        "source": case["record"]["source"],
        "record": case["record"],
        "target_translation_old_camera": target_translation.astype(float).tolist(),
        "target_translation_scale_m": target_scale,
        "rotations": outputs,
    }


def aggregate_rows(cases: list[dict], rotation: str, variant: str) -> dict:
    rows = [case["rotations"][rotation][variant] for case in cases]
    metrics = (
        "camera_translation_error_m",
        "camera_rotation_error_deg",
        "translation_direction_error_deg",
        "translation_scale_abs_error_m",
        "translation_scale_log_abs",
        "translation_vector_error_m",
        "translation_view_direction_error_m",
        "translation_transverse_error_m",
        "translation_horizontal_error_m",
        "translation_vertical_error_m",
    )
    output = {"case_count": len(rows)}
    for metric in metrics:
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        output[metric] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)),
        }
    return output


def build_aggregate(cases: list[dict]) -> dict:
    by_source: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        by_source[str(case["source"])].append(case)
    output = {"overall": {}, "by_source": {}}
    for rotation in ROTATIONS:
        output["overall"][rotation] = {
            variant: aggregate_rows(cases, rotation, variant) for variant in VARIANTS
        }
    for source, source_cases in sorted(by_source.items()):
        output["by_source"][source] = {}
        for rotation in ROTATIONS:
            output["by_source"][source][rotation] = {
                variant: aggregate_rows(source_cases, rotation, variant)
                for variant in VARIANTS
            }
    return output


def write_summary(aggregate: dict, output_dir: Path) -> None:
    rows = []
    for rotation in ROTATIONS:
        for variant in VARIANTS:
            row = aggregate["overall"][rotation][variant]
            rows.append(
                {
                    "rotation": rotation,
                    "variant": variant,
                    "vector_error_m": row["translation_vector_error_m"]["mean"],
                    "direction_error_deg": row["translation_direction_error_deg"]["mean"],
                    "scale_error_m": row["translation_scale_abs_error_m"]["mean"],
                    "camera_translation_m": row["camera_translation_error_m"]["mean"],
                    "camera_rotation_deg": row["camera_rotation_error_deg"]["mean"],
                }
            )
    with (output_dir / "v17_translation_partial_oracle_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# V17 Translation Direction-Scale Partial Oracle",
        "",
        "Translation is measured in the last pre-cut camera frame.",
        "",
        "| Rotation | Translation variant | Vector (m) | Direction (deg) | Scale (m) | Camera T (m) | Camera R (deg) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['rotation']} | {row['variant']} | {row['vector_error_m']:.3f} | "
            f"{row['direction_error_deg']:.2f} | {row['scale_error_m']:.3f} | "
            f"{row['camera_translation_m']:.3f} | {row['camera_rotation_deg']:.2f} |"
        )
    lines.extend(["", "## Direction versus scale gain", ""])
    for rotation in ROTATIONS:
        base = aggregate["overall"][rotation]["current_direction_current_scale"]
        gt_direction = aggregate["overall"][rotation]["gt_direction_current_scale"]
        gt_scale = aggregate["overall"][rotation]["current_direction_gt_scale"]
        base_error = base["translation_vector_error_m"]["mean"]
        direction_gain = base_error - gt_direction["translation_vector_error_m"]["mean"]
        scale_gain = base_error - gt_scale["translation_vector_error_m"]["mean"]
        lines.append(
            f"- `{rotation}`: GT direction gain `{direction_gain:.3f} m`; "
            f"GT scale gain `{scale_gain:.3f} m`."
        )
    (output_dir / "v17_translation_partial_oracle_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_v16_cases(args.v16_dir)
    local_dirs, partial_cases = load_maps(args)
    outputs = []
    for index, case in enumerate(cases):
        name = str(case["case_name"])
        outputs.append(
            run_case(case, local_dirs[name], partial_cases[name], args)
        )
        if (index + 1) % 20 == 0:
            print(f"Stage-0 {index + 1}/{len(cases)}", flush=True)
    aggregate = build_aggregate(outputs)
    payload = {
        "experiment": "V17 Translation Direction-Scale Partial Oracle",
        "case_count": len(outputs),
        "protocol": {
            "translation_frame": "last pre-cut camera frame",
            "global_gauge_invariant": True,
            "gt_depth_used": False,
            "gt_camera_use": "partial oracle and evaluation only",
        },
        "aggregate": aggregate,
        "cases": outputs,
    }
    (args.output_dir / "v17_translation_partial_oracle.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    write_summary(aggregate, args.output_dir)
    print(args.output_dir / "v17_translation_partial_oracle.json", flush=True)


if __name__ == "__main__":
    main()
