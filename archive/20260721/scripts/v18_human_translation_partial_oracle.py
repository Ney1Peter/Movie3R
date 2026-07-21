#!/usr/bin/env python3
"""V18 stages 0-1: coordinate audit and human-root translation partial oracles."""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from scipy.stats import pearsonr, spearmanr


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from dust3r.utils.smpl_layer import SMPL_Layer  # noqa: E402
from v10_boundary_gauge_partial_oracle_probe import load_predicted_joints  # noqa: E402
from v10_oracle_candidate_selection_probe import predicted_poses  # noqa: E402
from v13_scene_coordinate_oracle import direct_transform_error, transform_points  # noqa: E402


DEFAULT_V16 = REPO_ROOT / "output" / "v16_human_aware_rotation_residual" / "candidate_cache"
DEFAULT_V10 = (
    REPO_ROOT
    / "output"
    / "v10_candidate_selection"
    / "oracle_gt_4source"
    / "oracle_candidate_selection_metrics.json"
)
DEFAULT_GT = (
    REPO_ROOT
    / "output"
    / "v10_candidate_selection"
    / "boundary_gauge_partial_oracle"
    / "gt_cache"
)
DEFAULT_OUTPUT = REPO_ROOT / "output" / "v18_human_metric_translation" / "partial_oracle"
ROTATIONS = ("gt_rotation", "torso_rotation", "fixed_rotation")
WORLD_ROOTS = ("predicted_last", "predicted_constant_velocity", "gt_current")
CAMERA_ROOTS = (
    "predicted",
    "gt_depth_only",
    "gt_transverse_only",
    "gt_full",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v16_dir", type=Path, default=DEFAULT_V16)
    parser.add_argument("--v10_report", type=Path, default=DEFAULT_V10)
    parser.add_argument("--gt_cache", type=Path, default=DEFAULT_GT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--boundary", type=int, default=2)
    return parser.parse_args()


def load_cases(v16_dir: Path, v10_report: Path) -> tuple[list[dict], dict[str, Path]]:
    cases = []
    for path in sorted(glob.glob(str(v16_dir / "v16_candidates_shard_*_of_*.json"))):
        cases.extend(json.loads(Path(path).read_text(encoding="utf-8"))["cases"])
    report = json.loads(v10_report.read_text(encoding="utf-8"))
    local_dirs = {
        str(case["case_name"]): Path(case["paths"]["human3r_local_reset"])
        for case in report["cases"]
    }
    cases = sorted(cases, key=lambda row: str(row["case_name"]))
    if len(cases) != 180 or len({row["case_name"] for row in cases}) != 180:
        raise RuntimeError(f"Expected 180 unique V16 cases, got {len(cases)}")
    return cases, local_dirs


def build_layer(device: torch.device) -> SMPL_Layer:
    layer = SMPL_Layer(type="smplx", gender="neutral", num_betas=10, kid=False, person_center="head").to(device)
    return layer.eval()


def inverse_transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    return (transform[:3, :3].T @ (np.asarray(point) - transform[:3, 3])).astype(np.float32)


def camera_pose_from_human(rotation: np.ndarray, world_root: np.ndarray, camera_root: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = np.asarray(rotation, dtype=np.float32)
    pose[:3, 3] = np.asarray(world_root, dtype=np.float32) - pose[:3, :3] @ np.asarray(camera_root, dtype=np.float32)
    return pose


def boundary_from_camera_pose(camera_pose: np.ndarray, predicted_pose: np.ndarray) -> np.ndarray:
    return (camera_pose @ np.linalg.inv(predicted_pose)).astype(np.float32)


def direction_error(estimated: np.ndarray, target: np.ndarray) -> float:
    a = np.asarray(estimated, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64)
    a /= max(float(np.linalg.norm(a)), 1e-8)
    b /= max(float(np.linalg.norm(b)), 1e-8)
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def evaluate_camera_pose(
    camera_pose: np.ndarray,
    predicted_pose: np.ndarray,
    target_pose: np.ndarray,
    target_boundary: np.ndarray,
) -> dict:
    boundary = boundary_from_camera_pose(camera_pose, predicted_pose)
    row = direct_transform_error(boundary, predicted_pose, target_pose)
    delta_world = camera_pose[:3, 3] - target_pose[:3, 3]
    delta_camera = target_pose[:3, :3].T @ delta_world
    estimated_relative = np.linalg.inv(target_pose) @ camera_pose
    return {
        **row,
        "camera_translation_error_target_xyz_m": np.abs(delta_camera).astype(float).tolist(),
        "viewing_direction_error_m": float(abs(delta_camera[2])),
        "transverse_error_m": float(np.linalg.norm(delta_camera[:2])),
        "vertical_error_m": float(abs(delta_camera[1])),
        "camera_center_error_m": float(np.linalg.norm(delta_world)),
        "boundary_translation_error_m": float(np.linalg.norm(boundary[:3, 3] - target_boundary[:3, 3])),
        "camera_relative_translation": estimated_relative[:3, 3].astype(float).tolist(),
        "transform": boundary.astype(float).tolist(),
    }


def candidate_transform(case: dict, group: str, key: str) -> np.ndarray:
    return np.asarray(case[group][key]["transform"], dtype=np.float32)


def run_case(
    case: dict,
    local_dir: Path,
    gt_cache: Path,
    layer: SMPL_Layer,
    device: torch.device,
    boundary: int,
) -> dict:
    poses = predicted_poses(local_dir)
    predicted_joints_world = load_predicted_joints(local_dir, layer, device)
    predicted_joints_camera = np.stack(
        [
            np.einsum("ij,kj->ki", poses[index, :3, :3].T, predicted_joints_world[index] - poses[index, :3, 3])
            for index in range(len(poses))
        ]
    ).astype(np.float32)
    with np.load(gt_cache / f"{case['case_name']}.npz") as gt:
        gt_poses = gt["gt_poses"].astype(np.float32)
        gt_joints_world = gt["gt_joints_world"].astype(np.float32)

    gauge_target = poses[0] @ np.linalg.inv(gt_poses[0])
    target_poses = np.stack([gauge_target @ pose for pose in gt_poses]).astype(np.float32)
    target_joints_world = np.stack([transform_points(gauge_target, joints) for joints in gt_joints_world])
    target_boundary = (target_poses[boundary] @ np.linalg.inv(poses[boundary])).astype(np.float32)
    gt_camera_root = inverse_transform_point(gt_poses[boundary], gt_joints_world[boundary, 0])
    target_world_root = target_joints_world[boundary, 0]
    target_pose = target_poses[boundary]

    recovered_gt_pose = camera_pose_from_human(target_pose[:3, :3], target_world_root, gt_camera_root)
    recovered_gt_boundary = boundary_from_camera_pose(recovered_gt_pose, poses[boundary])
    coordinate_audit = {
        "equation_residual_m": float(
            np.linalg.norm(target_world_root - (target_pose[:3, :3] @ gt_camera_root + target_pose[:3, 3]))
        ),
        "recovered_camera_translation_error_m": float(np.linalg.norm(recovered_gt_pose[:3, 3] - target_pose[:3, 3])),
        "recovered_boundary_translation_error_m": float(
            np.linalg.norm(recovered_gt_boundary[:3, 3] - target_boundary[:3, 3])
        ),
        "root_definition": "SMPL-X joint 0 / pelvis for GT and Human3R prediction",
        "unit": "meter",
        "camera_convention": "camera-to-world pose; column-vector left multiplication",
    }

    fixed_boundary = candidate_transform(case, "baselines", "fixed_explicit")
    torso_boundary = candidate_transform(case, "fixed_candidates", "fixed_torso_motion_1f_resolve_t")
    camera_rotations = {
        "gt_rotation": target_pose[:3, :3],
        "torso_rotation": torso_boundary[:3, :3] @ poses[boundary, :3, :3],
        "fixed_rotation": fixed_boundary[:3, :3] @ poses[boundary, :3, :3],
    }
    predicted_world_roots = {
        "predicted_last": predicted_joints_world[boundary - 1, 0],
        "predicted_constant_velocity": (
            2.0 * predicted_joints_world[boundary - 1, 0] - predicted_joints_world[boundary - 2, 0]
        ).astype(np.float32),
        "gt_current": target_world_root,
    }
    predicted_camera_root = predicted_joints_camera[boundary, 0]
    camera_roots = {
        "predicted": predicted_camera_root,
        "gt_depth_only": np.asarray(
            [predicted_camera_root[0], predicted_camera_root[1], gt_camera_root[2]], dtype=np.float32
        ),
        "gt_transverse_only": np.asarray(
            [gt_camera_root[0], gt_camera_root[1], predicted_camera_root[2]], dtype=np.float32
        ),
        "gt_full": gt_camera_root,
    }

    variants = {}
    for rotation_name, rotation in camera_rotations.items():
        variants[rotation_name] = {}
        for world_name, world_root in predicted_world_roots.items():
            variants[rotation_name][world_name] = {}
            for camera_name, camera_root in camera_roots.items():
                camera_pose = camera_pose_from_human(rotation, world_root, camera_root)
                variants[rotation_name][world_name][camera_name] = evaluate_camera_pose(
                    camera_pose, poses[boundary], target_pose, target_boundary
                )

    camera_root_delta = predicted_camera_root - gt_camera_root
    motion_errors = {
        name: float(np.linalg.norm(root - target_world_root))
        for name, root in predicted_world_roots.items()
    }
    return {
        "case_name": case["case_name"],
        "source": case["record"]["source"],
        "record": case["record"],
        "coordinate_audit": coordinate_audit,
        "human_camera_root": {
            "predicted": predicted_camera_root.astype(float).tolist(),
            "gt": gt_camera_root.astype(float).tolist(),
            "error_xyz_m": np.abs(camera_root_delta).astype(float).tolist(),
            "position_error_m": float(np.linalg.norm(camera_root_delta)),
            "depth_error_m": float(abs(camera_root_delta[2])),
            "transverse_error_m": float(np.linalg.norm(camera_root_delta[:2])),
            "direction_error_deg": direction_error(predicted_camera_root, gt_camera_root),
        },
        "human_motion": {
            "predicted_last_world": predicted_world_roots["predicted_last"].astype(float).tolist(),
            "predicted_constant_velocity_world": predicted_world_roots["predicted_constant_velocity"].astype(float).tolist(),
            "gt_current_world": target_world_root.astype(float).tolist(),
            "last_error_m": motion_errors["predicted_last"],
            "constant_velocity_error_m": motion_errors["predicted_constant_velocity"],
        },
        "variants": variants,
    }


def distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
    }


def aggregate_variant(cases: list[dict], rotation: str, world: str, camera: str) -> dict:
    rows = [case["variants"][rotation][world][camera] for case in cases]
    return {
        "count": len(rows),
        "translation_m": distribution([float(row["camera_translation_error_m"]) for row in rows]),
        "rotation_deg": distribution([float(row["camera_rotation_error_deg"]) for row in rows]),
        "viewing_direction_m": distribution([float(row["viewing_direction_error_m"]) for row in rows]),
        "transverse_m": distribution([float(row["transverse_error_m"]) for row in rows]),
        "xyz_m": {
            axis: distribution([float(row["camera_translation_error_target_xyz_m"][index]) for row in rows])
            for index, axis in enumerate("xyz")
        },
        "catastrophic_rate": float(
            np.mean(
                [
                    float(row["camera_translation_error_m"]) > 1.0
                    or float(row["camera_rotation_error_deg"]) > 30.0
                    for row in rows
                ]
            )
        ),
    }


def aggregate(cases: list[dict]) -> dict:
    return {
        rotation: {
            world: {
                camera: aggregate_variant(cases, rotation, world, camera)
                for camera in CAMERA_ROOTS
            }
            for world in WORLD_ROOTS
        }
        for rotation in ROTATIONS
    }


def correlation(cases: list[dict], key: str, values: list[float]) -> dict:
    predictor = np.asarray(values, dtype=np.float64)
    target = np.asarray(
        [
            case["variants"]["torso_rotation"]["predicted_constant_velocity"]["predicted"][
                "camera_translation_error_m"
            ]
            for case in cases
        ],
        dtype=np.float64,
    )
    return {
        "predictor": key,
        "pearson": float(pearsonr(predictor, target).statistic),
        "spearman": float(spearmanr(predictor, target).statistic),
    }


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# V18 Human Translation Partial Oracle",
        "",
        "## Coordinate Audit",
        "",
        f"- Max equation residual: `{report['coordinate_audit']['max_equation_residual_m']:.8f} m`.",
        f"- Max recovered camera translation error: `{report['coordinate_audit']['max_camera_translation_error_m']:.8f} m`.",
        f"- Max recovered boundary translation error: `{report['coordinate_audit']['max_boundary_translation_error_m']:.8f} m`.",
        "",
        "## Main Partial Oracles",
        "",
        "| Rotation | World root | Camera root | T mean | View error | Transverse | R mean | Catastrophic |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    selections = [
        ("gt_rotation", "predicted_constant_velocity", "predicted"),
        ("gt_rotation", "gt_current", "predicted"),
        ("gt_rotation", "predicted_constant_velocity", "gt_full"),
        ("gt_rotation", "gt_current", "gt_full"),
        ("gt_rotation", "gt_current", "gt_depth_only"),
        ("gt_rotation", "gt_current", "gt_transverse_only"),
        ("torso_rotation", "predicted_constant_velocity", "predicted"),
        ("torso_rotation", "gt_current", "predicted"),
        ("torso_rotation", "predicted_constant_velocity", "gt_full"),
        ("torso_rotation", "gt_current", "gt_full"),
        ("fixed_rotation", "predicted_constant_velocity", "predicted"),
    ]
    for rotation, world, camera in selections:
        row = report["overall"][rotation][world][camera]
        lines.append(
            f"| {rotation} | {world} | {camera} | {row['translation_m']['mean']:.3f} | "
            f"{row['viewing_direction_m']['mean']:.3f} | {row['transverse_m']['mean']:.3f} | "
            f"{row['rotation_deg']['mean']:.2f} | {100.0 * row['catastrophic_rate']:.1f}% |"
        )
    lines.extend(["", "## Human Error", ""])
    for key, value in report["human_error"].items():
        if isinstance(value, dict) and "mean" in value:
            lines.append(f"- `{key}`: mean `{value['mean']:.3f}`, P90 `{value['p90']:.3f}`.")
    lines.extend(["", "## Correlation", ""])
    for row in report["correlations"]:
        lines.append(f"- `{row['predictor']}`: Pearson `{row['pearson']:.3f}`, Spearman `{row['spearman']:.3f}`.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("V18 SMPL-X partial oracle requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases, local_dirs = load_cases(args.v16_dir, args.v10_report)
    device = torch.device(args.device)
    layer = build_layer(device)
    rows = []
    for index, case in enumerate(cases):
        rows.append(
            run_case(
                case,
                local_dirs[str(case["case_name"])],
                args.gt_cache,
                layer,
                device,
                int(args.boundary),
            )
        )
        if (index + 1) % 20 == 0:
            print(f"V18 partial oracle {index + 1}/{len(cases)}", flush=True)
    overall = aggregate(rows)
    by_source = {
        source: aggregate([row for row in rows if row["source"] == source])
        for source in sorted({row["source"] for row in rows})
    }
    coordinate = {
        "max_equation_residual_m": float(max(row["coordinate_audit"]["equation_residual_m"] for row in rows)),
        "max_camera_translation_error_m": float(
            max(row["coordinate_audit"]["recovered_camera_translation_error_m"] for row in rows)
        ),
        "max_boundary_translation_error_m": float(
            max(row["coordinate_audit"]["recovered_boundary_translation_error_m"] for row in rows)
        ),
    }
    human_error = {
        "camera_root_position_m": distribution([row["human_camera_root"]["position_error_m"] for row in rows]),
        "camera_root_depth_m": distribution([row["human_camera_root"]["depth_error_m"] for row in rows]),
        "camera_root_transverse_m": distribution([row["human_camera_root"]["transverse_error_m"] for row in rows]),
        "motion_last_m": distribution([row["human_motion"]["last_error_m"] for row in rows]),
        "motion_constant_velocity_m": distribution([row["human_motion"]["constant_velocity_error_m"] for row in rows]),
    }
    correlations = [
        correlation(rows, "camera_root_depth_error", [row["human_camera_root"]["depth_error_m"] for row in rows]),
        correlation(rows, "camera_root_position_error", [row["human_camera_root"]["position_error_m"] for row in rows]),
        correlation(rows, "motion_constant_velocity_error", [row["human_motion"]["constant_velocity_error_m"] for row in rows]),
    ]
    report = {
        "experiment": "V18 Human-Calibrated Metric Translation Stages 0-1",
        "case_count": len(rows),
        "protocol": {
            "human3r_frozen": True,
            "gt_depth_used": False,
            "gt_camera_and_human_use": "coordinate audit and partial oracle only",
            "predicted_motion": "constant velocity from the two cached pre-cut frames",
            "camera_convention": "camera-to-world",
        },
        "coordinate_audit": coordinate,
        "human_error": human_error,
        "correlations": correlations,
        "overall": overall,
        "by_source": by_source,
        "cases": rows,
    }
    output = args.output_dir / "v18_human_translation_partial_oracle.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    write_markdown(args.output_dir / "v18_human_translation_partial_oracle_summary.md", report)
    print(json.dumps({"coordinate": coordinate, "human_error": human_error}, indent=2), flush=True)
    print(f">> wrote {output}", flush=True)


if __name__ == "__main__":
    main()
