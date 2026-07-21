#!/usr/bin/env python3
"""Build the shared explicit-only feature cache used by all V17 branches."""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from v10_oracle_candidate_selection_probe import predicted_poses  # noqa: E402
from v17_translation_partial_oracle import relative_boundary_pose  # noqa: E402


DEFAULT_V16 = REPO_ROOT / "output" / "v16_human_aware_rotation_residual" / "candidate_cache"
DEFAULT_V10 = (
    REPO_ROOT
    / "output"
    / "v10_candidate_selection"
    / "oracle_gt_4source"
    / "oracle_candidate_selection_metrics.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "output" / "v17_direct_vs_factorized" / "feature_cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v16_dir", type=Path, default=DEFAULT_V16)
    parser.add_argument("--v10_report", type=Path, default=DEFAULT_V10)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--boundary", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def load_cases(v16_dir: Path, v10_report: Path) -> tuple[list[dict], dict[str, Path]]:
    cases: list[dict] = []
    for path in sorted(glob.glob(str(v16_dir / "v16_candidates_shard_*_of_*.json"))):
        cases.extend(json.loads(Path(path).read_text(encoding="utf-8"))["cases"])
    report = json.loads(v10_report.read_text(encoding="utf-8"))
    local_dirs = {
        str(case["case_name"]): Path(case["paths"]["human3r_local_reset"])
        for case in report["cases"]
    }
    cases = sorted(cases, key=lambda case: str(case["case_name"]))
    names = [str(case["case_name"]) for case in cases]
    if len(names) != 180 or len(names) != len(set(names)):
        raise RuntimeError(f"Expected 180 unique V16 cases, got {len(names)}/{len(set(names))}")
    if set(names) != set(local_dirs):
        raise RuntimeError("V16 and V10 case sets differ")
    return cases, local_dirs


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def nested(row: dict, *keys: str, default: float = 0.0) -> float:
    value: object = row
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return safe_float(value, default)


def rotation_6d(rotation: np.ndarray) -> np.ndarray:
    return np.asarray(rotation, dtype=np.float32)[:, :2].T.reshape(-1).astype(np.float32)


def transform_feature(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float32)
    translation = transform[:3, 3]
    scale = float(np.linalg.norm(translation))
    direction = translation / max(scale, 1e-6)
    return np.concatenate(
        [
            rotation_6d(transform[:3, :3]),
            translation,
            direction.astype(np.float32),
            np.asarray([math.log(max(scale, 1e-6))], dtype=np.float32),
        ]
    )


def quantiles(values: np.ndarray, points: tuple[float, ...]) -> list[float]:
    if values.size == 0:
        return [0.0] * len(points)
    return [safe_float(value) for value in np.quantile(values, points)]


def pointmap_summary(local_dir: Path, idx: int) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    with np.load(local_dir / "camera" / f"{idx:06d}.npz") as camera:
        K = np.asarray(camera["intrinsics"], dtype=np.float32)
    depth = np.load(local_dir / "depth" / f"{idx:06d}.npy").astype(np.float32)
    conf = np.load(local_dir / "conf" / f"{idx:06d}.npy").astype(np.float32)
    with np.load(local_dir / "smpl" / f"{idx:06d}.npz", allow_pickle=True) as human:
        mask = np.asarray(human["msk"], dtype=np.float32)
        mask = mask[0] if mask.ndim == 3 and len(mask) else np.zeros_like(depth)
        shape = np.asarray(human["shape"], dtype=np.float32)
        root_rotvec = np.asarray(human["rotvec"], dtype=np.float32)
        root_translation = np.asarray(human["transl"], dtype=np.float32)
    height, width = depth.shape
    yy, xx = np.indices(depth.shape, dtype=np.float32)
    valid = np.isfinite(depth) & np.isfinite(conf) & (depth > 0.05) & (depth < 50.0)
    background = valid & (mask < 0.10)
    foreground = valid & (mask >= 0.10)
    x = (xx - K[0, 2]) / max(float(K[0, 0]), 1e-6) * depth
    y = (yy - K[1, 2]) / max(float(K[1, 1]), 1e-6) * depth
    points = np.stack([x, y, depth], axis=-1)

    values: list[float] = [
        float(K[0, 0] / width),
        float(K[1, 1] / height),
        float(K[0, 2] / width),
        float(K[1, 2] / height),
        float(width / height),
        float(valid.mean()),
        float(background.mean()),
        float(foreground.mean()),
    ]
    names = [
        "fx_over_w",
        "fy_over_h",
        "cx_over_w",
        "cy_over_h",
        "aspect",
        "valid_fraction",
        "background_fraction",
        "human_fraction",
    ]
    for region_name, region in (("all", valid), ("background", background), ("human", foreground)):
        region_points = points[region]
        region_depth = depth[region]
        region_conf = conf[region]
        if len(region_points):
            center = np.median(region_points, axis=0)
            spread = np.quantile(region_points, 0.90, axis=0) - np.quantile(region_points, 0.10, axis=0)
            covariance = np.cov(region_points.T) if len(region_points) > 3 else np.zeros((3, 3))
            eigenvalues = np.sort(np.maximum(np.linalg.eigvalsh(covariance), 0.0))[::-1]
        else:
            center = np.zeros(3)
            spread = np.zeros(3)
            eigenvalues = np.zeros(3)
        region_values = [
            *center.tolist(),
            *spread.tolist(),
            *eigenvalues.tolist(),
            *quantiles(region_depth, (0.10, 0.50, 0.90)),
            *quantiles(region_conf, (0.10, 0.50, 0.90)),
        ]
        values.extend(region_values)
        names.extend(
            [
                *(f"{region_name}_center_{axis}" for axis in "xyz"),
                *(f"{region_name}_spread_{axis}" for axis in "xyz"),
                *(f"{region_name}_cov_eig_{rank}" for rank in range(3)),
                *(f"{region_name}_depth_q{quantile}" for quantile in (10, 50, 90)),
                *(f"{region_name}_conf_q{quantile}" for quantile in (10, 50, 90)),
            ]
        )

    # A coarse spatial layout preserves physical depth structure without pixels or tokens.
    for gy in range(3):
        for gx in range(3):
            cell = background.copy()
            cell &= yy >= gy * height / 3.0
            cell &= yy < (gy + 1) * height / 3.0
            cell &= xx >= gx * width / 3.0
            cell &= xx < (gx + 1) * width / 3.0
            cell_points = points[cell]
            median = np.median(cell_points, axis=0) if len(cell_points) else np.zeros(3)
            values.extend([*median.tolist(), float(cell.mean()), safe_float(np.median(conf[cell])) if cell.any() else 0.0])
            names.extend(
                [
                    f"grid_{gy}_{gx}_median_x",
                    f"grid_{gy}_{gx}_median_y",
                    f"grid_{gy}_{gx}_median_z",
                    f"grid_{gy}_{gx}_fraction",
                    f"grid_{gy}_{gx}_conf",
                ]
            )

    human_values: list[float] = [float(len(root_translation) > 0)]
    human_names = ["human_present"]
    if len(root_translation):
        root_rotation = Rotation.from_rotvec(root_rotvec[0, 0].astype(np.float64)).as_matrix().astype(np.float32)
        human_values.extend(root_translation[0].tolist())
        human_values.extend(rotation_6d(root_rotation).tolist())
        human_values.extend(shape[0, :10].tolist())
    else:
        human_values.extend([0.0] * 19)
    human_names.extend([*(f"human_root_{axis}" for axis in "xyz"), *(f"human_rot6d_{i}" for i in range(6)), *(f"human_shape_{i}" for i in range(10))])
    return (
        np.asarray(values, dtype=np.float32),
        np.asarray(human_values, dtype=np.float32),
        names,
        human_names,
    )


def inference_diagnostics(case: dict) -> tuple[np.ndarray, list[str]]:
    fixed = case["baselines"]["fixed_explicit"]
    torso = case["fixed_candidates"]["fixed_torso_motion_1f_resolve_t"]
    root_check = case["fixed_candidates"]["fixed_torso_motion_1f_root_check"]
    motion = case.get("motion_diagnostics", {})
    old_ground = case.get("ground_diagnostics", {}).get("old", {})
    new_ground = case.get("ground_diagnostics", {}).get("new_1f", {})
    solver = torso.get("translation_solver", {})
    iterations = solver.get("iterations", [])
    last = iterations[-1] if iterations else {}
    names = [
        "fixed_estimated_scale",
        "fixed_human_root_jump",
        "fixed_human_torso_jump",
        "fixed_precut_human_speed",
        "torso_raw_residual",
        "torso_bounded_residual",
        "torso_residual_clipped",
        "torso_angle_mad",
        "motion_angular_speed",
        "motion_spread",
        "motion_inlier_fraction",
        "torso_human_root_jump",
        "torso_human_torso_jump",
        "root_check_coarse_error",
        "root_check_corrected_error",
        "translation_residual_from_t0",
        "translation_pairs",
        "translation_median_distance",
        "old_ground_valid_frames",
        "old_ground_spread",
        "new_ground_valid_frames",
        "new_ground_spread",
        "texture_score",
    ]
    values = [
        nested(fixed, "estimated_scale"),
        nested(fixed, "human_root_jump_m"),
        nested(fixed, "human_torso_jump_deg") / 30.0,
        nested(fixed, "precut_human_speed_m"),
        nested(torso, "raw_residual_deg") / 45.0,
        nested(torso, "bounded_residual_deg") / 45.0,
        float(bool(torso.get("clipped", False))),
        nested(torso, "angle_median_abs_deviation_deg") / 10.0,
        nested(motion, "angular_speed_deg_per_frame") / 10.0,
        nested(motion, "spread_deg") / 10.0,
        nested(motion, "inlier_count") / max(nested(motion, "count", default=1.0), 1.0),
        nested(torso, "human_root_jump_m"),
        nested(torso, "human_torso_jump_deg") / 30.0,
        nested(root_check, "coarse_root_motion_error_m"),
        nested(root_check, "corrected_root_motion_error_m"),
        nested(solver, "residual_from_t0_m"),
        nested(last, "pairs") / 1000.0,
        nested(last, "median_distance_m"),
        nested(old_ground, "valid_frames") / 3.0,
        nested(old_ground, "spread_deg") / 15.0,
        nested(new_ground, "valid_frames"),
        nested(new_ground, "spread_deg") / 15.0,
        safe_float(case.get("texture_score", 0.0)) * 10.0,
    ]
    return np.asarray(values, dtype=np.float32), names


def flatten_world_block(matrices: np.ndarray) -> np.ndarray:
    return matrices[:, :3, :4].reshape(-1).astype(np.float32)


def random_gauge(rng: np.random.Generator) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = Rotation.random(random_state=rng).as_matrix().astype(np.float32)
    transform[:3, 3] = rng.uniform(-2.0, 2.0, size=3).astype(np.float32)
    return transform


def gauge_world_matrices(matrices: np.ndarray, gauge: np.ndarray) -> np.ndarray:
    old_pose, fresh_pose, fixed, torso, v15 = matrices
    inverse = np.linalg.inv(gauge)
    return np.stack(
        [
            gauge @ old_pose,
            gauge @ fresh_pose,
            gauge @ fixed @ inverse,
            gauge @ torso @ inverse,
            gauge @ v15 @ inverse,
        ]
    ).astype(np.float32)


def run_case(case: dict, local_dir: Path, boundary: int) -> dict:
    poses = predicted_poses(local_dir)
    old_pose = poses[boundary - 1]
    fresh_pose = poses[boundary]
    fixed = np.asarray(case["baselines"]["fixed_explicit"]["transform"], dtype=np.float32)
    torso = np.asarray(
        case["fixed_candidates"]["fixed_torso_motion_1f_resolve_t"]["transform"], dtype=np.float32
    )
    v15 = np.asarray(case["baselines"]["v15_coarse"]["transform"], dtype=np.float32)
    oracle = np.asarray(case["baselines"]["boundary_oracle"]["transform"], dtype=np.float32)
    relative = {
        "fixed": relative_boundary_pose(old_pose, fixed, fresh_pose),
        "torso": relative_boundary_pose(old_pose, torso, fresh_pose),
        "v15": relative_boundary_pose(old_pose, v15, fresh_pose),
        "target": relative_boundary_pose(old_pose, oracle, fresh_pose),
    }
    point_values, point_names, human_values, human_names = [], [], [], []
    for idx, label in ((boundary - 2, "pre_previous"), (boundary - 1, "pre_last"), (boundary, "post_first")):
        point, human, pnames, hnames = pointmap_summary(local_dir, idx)
        point_values.append(point)
        human_values.append(human)
        point_names.extend([f"{label}_{name}" for name in pnames])
        human_names.extend([f"{label}_{name}" for name in hnames])
    diagnostics, diagnostic_names = inference_diagnostics(case)
    relative_values = np.concatenate(
        [transform_feature(relative[key]) for key in ("fixed", "torso", "v15")]
    )
    relative_names = [
        f"{key}_{name}"
        for key in ("fixed", "torso", "v15")
        for name in (
            *(f"rot6d_{idx}" for idx in range(6)),
            "tx",
            "ty",
            "tz",
            "dir_x",
            "dir_y",
            "dir_z",
            "log_scale",
        )
    ]
    invariant = np.concatenate([relative_values, *point_values, *human_values, diagnostics]).astype(np.float32)
    invariant_names = relative_names + point_names + human_names + diagnostic_names
    world_matrices = np.stack([old_pose, fresh_pose, fixed, torso, v15]).astype(np.float32)
    stats = np.concatenate(
        [
            point_values[1][:8],
            point_values[2][:8],
            diagnostics[-1:],
        ]
    ).astype(np.float32)
    record = case["record"]
    return {
        "case_name": str(case["case_name"]),
        "source": str(record["source"]),
        "capture": str(record.get("group", "unknown")),
        "camera_a": str(record.get("seqA", "unknown")).split("/")[-1],
        "camera_b": str(record.get("seqB", "unknown")).split("/")[-1],
        "camera_pair": "|".join(sorted((str(record.get("seqA", "unknown")), str(record.get("seqB", "unknown"))))),
        "angle_bucket": str(record.get("angle_bucket", "unknown")),
        "view_angle_deg": safe_float(record.get("view_angle_deg", 0.0)),
        "texture_score": safe_float(case.get("texture_score", 0.0)),
        "invariant": invariant,
        "invariant_names": invariant_names,
        "world_matrices": world_matrices,
        "stats": stats,
        "relative": relative,
        "stored_metrics": {
            key: case["baselines"][key]
            for key in ("original_continue", "hard_reset", "fixed_explicit", "boundary_oracle")
        }
        | {"torso_motion": case["fixed_candidates"]["fixed_torso_motion_1f_resolve_t"]},
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases, local_dirs = load_cases(args.v16_dir, args.v10_report)
    rows = []
    for index, case in enumerate(cases):
        rows.append(run_case(case, local_dirs[str(case["case_name"])], int(args.boundary)))
        if (index + 1) % 20 == 0:
            print(f"V17 feature cache {index + 1}/{len(cases)}", flush=True)

    invariant = np.stack([row.pop("invariant") for row in rows])
    world_matrices = np.stack([row.pop("world_matrices") for row in rows])
    stats = np.stack([row.pop("stats") for row in rows])
    relative = {
        key: np.stack([row["relative"].pop(key) for row in rows])
        for key in ("fixed", "torso", "v15", "target")
    }
    invariant_names = rows[0].pop("invariant_names")
    for row in rows[1:]:
        row.pop("invariant_names")
    for row in rows:
        row.pop("relative")

    # The learned target is invariant under arbitrary world-gauge changes.
    rng = np.random.default_rng(int(args.seed))
    maximum_error = 0.0
    for index in rng.choice(len(rows), size=min(32, len(rows)), replace=False):
        gauged = gauge_world_matrices(world_matrices[index], random_gauge(rng))
        old_pose, fresh_pose, fixed, torso, v15 = gauged
        for name, transform in (("fixed", fixed), ("torso", torso), ("v15", v15)):
            recomputed = relative_boundary_pose(old_pose, transform, fresh_pose)
            maximum_error = max(maximum_error, float(np.max(np.abs(recomputed - relative[name][index]))))

    np.savez_compressed(
        args.output_dir / "v17_explicit_features.npz",
        invariant=invariant,
        world_matrices=world_matrices,
        stats=stats,
        relative_fixed=relative["fixed"],
        relative_torso=relative["torso"],
        relative_v15=relative["v15"],
        relative_target=relative["target"],
    )
    metadata = {
        "experiment": "V17 shared explicit-only feature cache",
        "case_count": len(rows),
        "boundary": int(args.boundary),
        "protocol": {
            "raw_tokens_used": False,
            "gt_depth_used": False,
            "gt_camera_use": "relative SE(3) training target and evaluation only",
            "translation_frame": "last pre-cut Human3R camera frame",
            "post_cut_frames": 1,
            "pointmap_source": "frozen Human3R predicted depth and confidence",
        },
        "feature_dimensions": {
            "invariant": int(invariant.shape[1]),
            "world_gauge_sensitive": int(5 * 3 * 4),
            "weak_statistics": int(stats.shape[1]),
        },
        "invariant_feature_names": invariant_names,
        "gauge_audit_max_abs_relative_transform_error": maximum_error,
        "rows": rows,
    }
    (args.output_dir / "v17_explicit_features.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "cache": str(args.output_dir / "v17_explicit_features.npz"),
                "shape": list(invariant.shape),
                "gauge_audit_max_abs_error": maximum_error,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
