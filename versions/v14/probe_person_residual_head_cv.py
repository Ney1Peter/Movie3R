#!/usr/bin/env python3
"""Cross-validated predictability probe for a person-local root-ray residual.

The experiment is deliberately restricted to the 41 controlled ``three``
development cuts.  It consumes the frozen learned-B0 identity-matching report
and cached Human3R predictions.  Deployment features are built before the GT
dictionary is exposed.  GT camera, identity, and body geometry are used only to
create the signed ray-correction label and to score frozen fold predictions.

The post camera is never changed.  Both regressors output one scalar per
automatically matched person, clipped to a fixed trust region and applied as a
rigid translation along that person's frozen post-camera ray.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from scipy.stats import pearsonr, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from versions.v14.probe_b0_anchor_conflict import SEQUENCE_INPUTS  # noqa: E402


DEFAULT_MATCHING_REPORT = (
    REPO_ROOT / "output/v14/b0_identity_matching/v14_b0_identity_matching.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "output/v14/fine_alignment_research/person_residual_head_cv_three"
)
FORBIDDEN_FEATURE_TOKENS = (
    "gt",
    "identity",
    "actor",
    "source_camera",
    "target_camera",
    "camera_id",
    "timestamp",
    "case_key",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matching_report", type=Path, default=DEFAULT_MATCHING_REPORT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--cap_m", type=float, default=0.30)
    parser.add_argument("--ridge_alpha", type=float, default=10.0)
    parser.add_argument("--mlp_alpha", type=float, default=0.1)
    parser.add_argument("--mlp_max_iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument(
        "--split",
        choices=("camera_pair", "double_block_timestamp_actor"),
        default="camera_pair",
    )
    parser.add_argument(
        "--models", nargs="+", choices=("ridge", "mlp"), default=("ridge", "mlp")
    )
    return parser.parse_args()


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def finite_stats(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {
            "count": 0,
            "mean": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
        }
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
    }


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return points @ np.asarray(transform, dtype=np.float64)[:3, :3].T + np.asarray(
        transform, dtype=np.float64
    )[:3, 3]


def world_to_camera(pose: np.ndarray, points: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    return (points - pose[:3, 3]) @ pose[:3, :3]


def safe_rotvec(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    try:
        return Rotation.from_matrix(matrix).as_rotvec()
    except ValueError:
        left, _, right = np.linalg.svd(matrix)
        rotation = left @ right
        if np.linalg.det(rotation) < 0:
            left[:, -1] *= -1
            rotation = left @ right
        return Rotation.from_matrix(rotation).as_rotvec()


def add_vector(features: dict[str, float], prefix: str, value: np.ndarray) -> None:
    array = np.asarray(value, dtype=np.float64).reshape(3)
    for axis, item in zip(("x", "y", "z"), array):
        features[f"{prefix}_{axis}"] = float(item)


def person_observation_features(
    human: dict, pose: np.ndarray, prefix: str
) -> dict[str, float]:
    """Features from one predicted person observation in its predicted camera."""
    root = np.asarray(human["root"], dtype=np.float64)
    local_root = world_to_camera(pose, root)
    root_range = float(np.linalg.norm(local_root))
    ray = local_root / max(root_range, 1e-8)
    bbox = np.asarray(human["bbox"], dtype=np.float64) / 512.0
    bbox_width = max(float(bbox[2] - bbox[0]), 1e-8)
    bbox_height = max(float(bbox[3] - bbox[1]), 1e-8)
    bbox_center = 0.5 * (bbox[:2] + bbox[2:])

    output: dict[str, float] = {
        f"{prefix}_score": float(human.get("score", float("nan"))),
        f"{prefix}_completeness": float(
            human.get("completeness", float("nan"))
        ),
        f"{prefix}_bbox_center_x": float(bbox_center[0]),
        f"{prefix}_bbox_center_y": float(bbox_center[1]),
        f"{prefix}_bbox_width": bbox_width,
        f"{prefix}_bbox_height": bbox_height,
        f"{prefix}_bbox_area": bbox_width * bbox_height,
        f"{prefix}_bbox_aspect": bbox_width / bbox_height,
        f"{prefix}_root_range": root_range,
        f"{prefix}_root_inv_range": 1.0 / max(root_range, 1e-8),
        f"{prefix}_root_x_over_z": float(local_root[0] / max(abs(local_root[2]), 1e-8)),
        f"{prefix}_root_y_over_z": float(local_root[1] / max(abs(local_root[2]), 1e-8)),
    }
    add_vector(output, f"{prefix}_root_local", local_root)
    add_vector(output, f"{prefix}_root_ray", ray)

    camera_rotation = np.asarray(pose, dtype=np.float64)[:3, :3]
    for name in ("joints", "vertices"):
        points = np.asarray(human[name], dtype=np.float64)
        centered_local = (points - root) @ camera_rotation
        extent = np.ptp(centered_local, axis=0)
        rms = float(np.sqrt(np.mean(np.sum(centered_local**2, axis=1))))
        output[f"{prefix}_{name}_rms"] = rms
        add_vector(output, f"{prefix}_{name}_extent", extent)

    for name in ("torso", "root_rotation"):
        if name in human:
            local_rotation = camera_rotation.T @ np.asarray(
                human[name], dtype=np.float64
            )
            add_vector(output, f"{prefix}_{name}_rotvec", safe_rotvec(local_rotation))
    return output


def matcher_pairs(
    cache_view: dict, matching: dict
) -> list[tuple[str, str, dict[str, float]]]:
    """Read only the frozen root+torso assignment and its predicted costs."""
    matcher = matching["matchers"]["root_torso"]
    assignment = matcher["predicted_identity_by_pre_identity"]
    row_order = [str(value) for value in matching["identities"]]
    column_order = [str(value) for value in matching["post_detection_gt_order"]]
    total_cost = np.asarray(matcher["cost"], dtype=np.float64)
    components = {
        name: np.asarray(value, dtype=np.float64)
        for name, value in matching["components"].items()
    }
    output = []
    for pre_key, post_key in assignment.items():
        pre_key, post_key = str(pre_key), str(post_key)
        if (
            pre_key not in cache_view["humans"][-2]
            or post_key not in cache_view["humans"][-1]
            or pre_key not in row_order
            or post_key not in column_order
        ):
            continue
        row = row_order.index(pre_key)
        column = column_order.index(post_key)
        row_costs = np.sort(total_cost[row])
        chosen = float(total_cost[row, column])
        features = {
            "matcher_selected_total": chosen,
            "matcher_selected_root": float(components["root"][row, column]),
            "matcher_selected_torso_deg": float(components["torso"][row, column]),
            "matcher_selected_joints": float(components["joints"][row, column]),
            "matcher_row_chosen_minus_best": chosen - float(row_costs[0]),
            "matcher_row_best_second_margin": (
                float(row_costs[1] - row_costs[0]) if len(row_costs) > 1 else 0.0
            ),
            "matcher_assignment_cost_per_person": float(
                matcher["best_assignment_cost"] / max(int(matcher["person_count"]), 1)
            ),
            "matcher_assignment_margin": float(matcher["best_vs_second_margin"]),
        }
        output.append((pre_key, post_key, features))
    return output


def build_deployment_rows(
    cache_view: dict, boundary: np.ndarray, matching: dict
) -> list[dict]:
    """Build proposal inputs with no GT dictionary available to this function."""
    poses = [np.asarray(value, dtype=np.float64) for value in cache_view["poses"]]
    humans = cache_view["humans"]
    pre_frame_numbers = [int(value) for value in cache_view["pre_frames"]]
    pre_pose = poses[-2]
    raw_post_pose = poses[-1]
    final_post_pose = np.asarray(boundary, dtype=np.float64) @ raw_post_pose

    relative_camera = np.linalg.inv(pre_pose) @ final_post_pose
    camera_baseline_local = relative_camera[:3, 3]
    camera_rotvec = safe_rotvec(relative_camera[:3, :3])
    boundary_rotvec = safe_rotvec(np.asarray(boundary)[:3, :3])
    common_camera_features: dict[str, float] = {
        "camera_baseline_norm": float(np.linalg.norm(camera_baseline_local)),
        "camera_rotation_deg": float(np.degrees(np.linalg.norm(camera_rotvec))),
        "b0_translation_norm": float(np.linalg.norm(np.asarray(boundary)[:3, 3])),
        "b0_rotation_deg": float(np.degrees(np.linalg.norm(boundary_rotvec))),
    }
    add_vector(common_camera_features, "camera_baseline_local", camera_baseline_local)
    add_vector(common_camera_features, "camera_relative_rotvec", camera_rotvec)

    rows = []
    for pre_key, post_key, match_features in matcher_pairs(cache_view, matching):
        features = {**common_camera_features, **match_features}
        history = [
            (index, pre_frame_numbers[index], frame[pre_key])
            for index, frame in enumerate(humans[:-1])
            if pre_key in frame
        ][-3:]
        if not history:
            continue
        reversed_history = list(reversed(history))
        features["pre_history_count"] = float(len(history))
        for lag, (pose_index, _, human) in enumerate(reversed_history):
            features.update(
                person_observation_features(human, poses[pose_index], f"pre_lag{lag}")
            )

        post_human = humans[-1][post_key]
        # Human and camera must use the same gauge.  Raw post/raw post pose is
        # camera-locally identical to B0-aligned post/final post pose.
        features.update(person_observation_features(post_human, raw_post_pose, "post"))

        last_index, _, last_human = history[-1]
        last_root = np.asarray(last_human["root"], dtype=np.float64)
        raw_post_root = np.asarray(post_human["root"], dtype=np.float64)
        aligned_post_root = transform_points(boundary, raw_post_root)
        root_jump_local = (aligned_post_root - last_root) @ pre_pose[:3, :3]
        features["root_jump_norm"] = float(np.linalg.norm(root_jump_local))
        add_vector(features, "root_jump_pre_camera", root_jump_local)

        last_local = world_to_camera(pre_pose, last_root)
        post_local = world_to_camera(final_post_pose, aligned_post_root)
        last_range = float(np.linalg.norm(last_local))
        post_range = float(np.linalg.norm(post_local))
        last_ray = last_local / max(last_range, 1e-8)
        post_ray = post_local / max(post_range, 1e-8)
        features["pre_post_range_delta"] = post_range - last_range
        features["pre_post_ray_angle_deg"] = float(
            np.degrees(np.arccos(np.clip(last_ray @ post_ray, -1.0, 1.0)))
        )

        if len(history) >= 2:
            first_index, first_frame, first_human = history[0]
            _, last_frame, _ = history[-1]
            frame_delta = max(int(last_frame - first_frame), 1)
            velocity_world = (
                last_root - np.asarray(first_human["root"], dtype=np.float64)
            ) / frame_delta
            velocity_local = velocity_world @ pre_pose[:3, :3]
            features["pre_velocity_norm_per_frame"] = float(
                np.linalg.norm(velocity_local)
            )
            add_vector(features, "pre_velocity_local_per_frame", velocity_local)
            first_range = float(
                np.linalg.norm(
                    world_to_camera(poses[first_index], first_human["root"])
                )
            )
            features["pre_range_slope_per_frame"] = (
                last_range - first_range
            ) / frame_delta
        if len(history) >= 3:
            roots = [np.asarray(item[2]["root"], dtype=np.float64) for item in history]
            frames = [int(item[1]) for item in history]
            v0 = (roots[1] - roots[0]) / max(frames[1] - frames[0], 1)
            v1 = (roots[2] - roots[1]) / max(frames[2] - frames[1], 1)
            acceleration_local = (v1 - v0) @ pre_pose[:3, :3]
            features["pre_acceleration_norm_per_frame2"] = float(
                np.linalg.norm(acceleration_local)
            )
            add_vector(
                features, "pre_acceleration_local_per_frame2", acceleration_local
            )

        ratio_pairs = (
            "bbox_width",
            "bbox_height",
            "joints_rms",
            "vertices_rms",
            "joints_extent_x",
            "joints_extent_y",
            "joints_extent_z",
            "vertices_extent_x",
            "vertices_extent_y",
            "vertices_extent_z",
        )
        for suffix in ratio_pairs:
            pre_value = features.get(f"pre_lag0_{suffix}", float("nan"))
            post_value = features.get(f"post_{suffix}", float("nan"))
            features[f"pre_post_{suffix}_log_ratio"] = float(
                np.log(max(post_value, 1e-8) / max(pre_value, 1e-8))
            )

        final_root = aligned_post_root
        camera_center = final_post_pose[:3, 3]
        ray_vector = final_root - camera_center
        ray_depth = float(np.linalg.norm(ray_vector))
        ray = ray_vector / max(ray_depth, 1e-8)
        rows.append(
            {
                "pre_key": pre_key,
                "post_key": post_key,
                "features": features,
                "root": final_root,
                "ray": ray,
                "ray_depth_m": ray_depth,
            }
        )
    return rows


def attach_gt_label(
    deployment_row: dict,
    cache: dict,
    boundary: np.ndarray,
) -> dict:
    """GT-only label/evaluator path, called after deployment features are frozen."""
    pre_key = deployment_row["pre_key"]
    gt = cache["gt"]
    pre_pose = np.asarray(cache["poses"][-2], dtype=np.float64)
    gauge = pre_pose @ np.linalg.inv(np.asarray(gt["pre_c2w"], dtype=np.float64))
    if pre_key not in gt["post_humans"]:
        raise KeyError(f"No GT post human for automatically tracked key {pre_key}")
    target_root = transform_points(gauge, gt["post_humans"][pre_key]["root"])
    root = np.asarray(deployment_row["root"], dtype=np.float64)
    ray = np.asarray(deployment_row["ray"], dtype=np.float64)
    label = float(np.dot(target_root - root, ray))
    error = float(np.linalg.norm(root - target_root))
    oracle_root = root + label * ray
    return {
        **deployment_row,
        "label_correction_m": label,
        "b0_root_error_m": error,
        "oracle_root_error_m": float(np.linalg.norm(oracle_root - target_root)),
        "target_root": target_root,
    }


def validate_feature_contract(feature_names: list[str]) -> None:
    violations = [
        name
        for name in feature_names
        if any(token in name.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    if violations:
        raise RuntimeError(f"Forbidden feature names indicate leakage: {violations}")


def correlation(values: np.ndarray, labels: np.ndarray) -> dict:
    finite = np.isfinite(values) & np.isfinite(labels)
    values, labels = values[finite], labels[finite]
    if len(values) < 4 or np.std(values) < 1e-12 or np.std(labels) < 1e-12:
        return {"count": int(len(values)), "pearson": float("nan"), "spearman": float("nan")}
    return {
        "count": int(len(values)),
        "pearson": float(pearsonr(values, labels).statistic),
        "spearman": float(spearmanr(values, labels).statistic),
    }


def make_matrix(rows: list[dict], feature_names: list[str]) -> np.ndarray:
    return np.asarray(
        [
            [row["features"].get(name, float("nan")) for name in feature_names]
            for row in rows
        ],
        dtype=np.float64,
    )


def fit_fold_model(
    name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    if name == "ridge":
        estimator = Ridge(alpha=float(args.ridge_alpha))
    elif name == "mlp":
        estimator = MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation="tanh",
            solver="lbfgs",
            alpha=float(args.mlp_alpha),
            max_iter=int(args.mlp_max_iter),
            random_state=int(args.seed),
        )
    else:
        raise KeyError(name)
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("model", estimator),
        ]
    )
    target_mean = float(np.mean(train_y))
    target_scale = max(float(np.std(train_y)), 1e-8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline.fit(train_x, (train_y - target_mean) / target_scale)
    return np.asarray(pipeline.predict(test_x), dtype=np.float64) * target_scale + target_mean


def cross_validated_predictions(
    matrix: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    timestamps: np.ndarray,
    actors: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    predictions = {
        name: np.full(len(labels), np.nan, dtype=np.float64)
        for name in dict.fromkeys(args.models)
    }
    folds = []
    if args.split == "camera_pair":
        split_units = [
            (str(group), groups != group, groups == group)
            for group in sorted(set(groups.tolist()))
        ]
    else:
        split_units = []
        for timestamp, actor in sorted(
            set(zip(timestamps.tolist(), actors.tolist())),
            key=lambda item: (int(item[0]), str(item[1])),
        ):
            test = (timestamps == timestamp) & (actors == actor)
            # Double block: the train fold sees neither this time instant nor
            # this actor at any other instant.
            train = (timestamps != timestamp) & (actors != actor)
            split_units.append((f"t{int(timestamp)}_{actor}", train, test))

    for split_unit, train, test in split_units:
        excluded = ~(train | test)
        if int(np.sum(train)) < 8 or int(np.sum(test)) < 1:
            folds.append(
                {
                    "held_out_unit": split_unit,
                    "train_samples": int(np.sum(train)),
                    "test_samples": int(np.sum(test)),
                    "blocked_not_train_or_test": int(np.sum(excluded)),
                    "status": "insufficient_samples",
                    "test_indices": np.flatnonzero(test),
                }
            )
            continue
        fold = {
            "held_out_unit": split_unit,
            "train_samples": int(np.sum(train)),
            "test_samples": int(np.sum(test)),
            "blocked_not_train_or_test": int(np.sum(excluded)),
            "status": "ok",
            "train_label_mean_m": float(np.mean(labels[train])),
            "test_label_mean_m": float(np.mean(labels[test])),
            "test_indices": np.flatnonzero(test),
        }
        for name in predictions:
            predictions[name][test] = fit_fold_model(
                name, matrix[train], labels[train], matrix[test], args
            )
        folds.append(fold)
    return predictions, folds


def method_metrics(
    prediction: np.ndarray,
    rows: list[dict],
    cap_m: float,
    clip: bool = True,
) -> dict:
    raw_prediction = np.asarray(prediction, dtype=np.float64)
    correction = (
        np.clip(raw_prediction, -float(cap_m), float(cap_m))
        if clip
        else raw_prediction
    )
    labels = np.asarray([row["label_correction_m"] for row in rows], dtype=np.float64)
    before = np.asarray([row["b0_root_error_m"] for row in rows], dtype=np.float64)
    roots = np.stack([row["root"] for row in rows])
    rays = np.stack([row["ray"] for row in rows])
    targets = np.stack([row["target_root"] for row in rows])
    after = np.linalg.norm(roots + correction[:, None] * rays - targets, axis=1)
    delta = after - before
    valid_sign = np.abs(labels) > 1e-8
    prediction_correlation = correlation(correction, labels)
    return {
        "count": int(len(rows)),
        "raw_prediction": finite_stats(raw_prediction),
        "applied_correction": finite_stats(correction),
        "clipped_rate": float(np.mean(np.abs(raw_prediction) > float(cap_m))) if clip else 0.0,
        "root_before": finite_stats(before),
        "root_corrected": finite_stats(after),
        "mean_root_delta_m": float(np.mean(delta)),
        "improve_rate": float(np.mean(after < before - 1e-12)),
        "harm_over_5cm_rate": float(np.mean(delta > 0.05)),
        "harm_over_10cm_rate": float(np.mean(delta > 0.10)),
        "sign_accuracy": float(
            np.mean(np.sign(correction[valid_sign]) == np.sign(labels[valid_sign]))
        ),
        "prediction_vs_label": prediction_correlation,
        "per_sample_correction_m": correction,
        "per_sample_root_error_m": after,
    }


def metric_subset(metrics: dict) -> dict:
    return {
        key: value
        for key, value in metrics.items()
        if not key.startswith("per_sample_")
    }


def markdown_report(report: dict) -> str:
    protocol = report["protocol"]
    split_name = str(protocol["split"])
    split_description = (
        "leave-one-directed-camera-pair-group-out"
        if split_name == "camera_pair"
        else "double-block `(timestamp, actor)`: training excludes the test timestamp and actor"
    )
    lines = [
        "# V14 Person Residual-Head CV (`three` only)",
        "",
        f"Cases/people/camera-pair groups/folds: `{protocol['case_count']}/"
        f"{protocol['sample_count']}/{protocol['group_count']}/{protocol['fold_count']}`. ",
        "Only the controlled `three` development cases are used; `dance` and `box` are not loaded.",
        f"Split: **{split_description}**. Prediction coverage: "
        f"`{100*protocol['prediction_coverage']:.1f}%`.",
        "The frozen root+torso auto matcher supplies associations. GT is exposed only after ",
        "deployment features are frozen, to build the signed ray label and evaluate predictions.",
        "",
        "The post camera is bit-unchanged. Every learned output is clipped to ",
        f"`±{protocol['cap_m']:.2f} m` and applied as a rigid person translation along the frozen post-camera ray.",
        "",
        "## Cross-validated result",
        "",
        "| Method | Root mean | P50 | P90 | Delta | Improve | Sign | Pearson | Harm >5cm | Harm >10cm |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    display_names = {
        "b0": "frozen B0",
        "ridge": "ridge CV",
        "mlp": "MLP CV",
        "oracle_capped": "GT-ray oracle capped",
        "oracle": "GT-ray oracle",
    }
    method_order = ["b0"] + [
        name for name in ("ridge", "mlp") if name in report["methods"]
    ] + ["oracle_capped", "oracle"]
    for name in method_order:
        row = report["methods"][name]
        lines.append(
            f"| {display_names[name]} | {row['root_corrected']['mean']:.4f} | "
            f"{row['root_corrected']['p50']:.4f} | {row['root_corrected']['p90']:.4f} | "
            f"{row['mean_root_delta_m']:+.4f} | {100*row['improve_rate']:.1f}% | "
            f"{100*row['sign_accuracy']:.1f}% | {row['prediction_vs_label']['pearson']:.3f} | "
            f"{100*row['harm_over_5cm_rate']:.1f}% | {100*row['harm_over_10cm_rate']:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## Strongest single-feature signed correlations",
            "",
            "| Feature | Pearson | Spearman | N |",
            "|---|---:|---:|---:|",
        ]
    )
    ranked = sorted(
        report["feature_correlations"].items(),
        key=lambda item: abs(item[1]["pearson"])
        if item[1]["pearson"] is not None
        and math.isfinite(float(item[1]["pearson"]))
        else -1.0,
        reverse=True,
    )[:15]
    for name, row in ranked:
        pearson_value = row["pearson"]
        spearman_value = row["spearman"]
        lines.append(
            f"| `{name}` | "
            f"{pearson_value if pearson_value is not None else float('nan'):.3f} | "
            f"{spearman_value if spearman_value is not None else float('nan'):.3f} | "
            f"{row['count']} |"
        )

    lines.extend(
        [
            "",
            "## Per camera-pair diagnostic",
            "",
            "| Group | People | B0 root | Ridge root | MLP root | Ridge improve | MLP improve |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for group, row in report["per_group"].items():
        ridge = row.get("ridge")
        mlp = row.get("mlp")
        lines.append(
            f"| {group} | {row['count']} | {row['b0']['root_corrected']['mean']:.4f} | "
            f"{ridge['root_corrected']['mean']:.4f} | " if ridge else
            f"| {group} | {row['count']} | {row['b0']['root_corrected']['mean']:.4f} | -- | "
        )

        # Complete the row separately so ridge-only runs can be reported immediately.
        lines[-1] += (
            f"{mlp['root_corrected']['mean']:.4f} | " if mlp else "-- | "
        )
        lines[-1] += (
            f"{100*ridge['improve_rate']:.1f}% | " if ridge else "-- | "
        )
        lines[-1] += f"{100*mlp['improve_rate']:.1f}% |" if mlp else "-- |"

    for title, key, label in (
        ("Per actor", "per_actor", "Actor"),
        ("Per timestamp", "per_timestamp", "Timestamp"),
    ):
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                f"| {label} | People | B0 root | Ridge root | Ridge improve | Ridge sign | Ridge harm >5cm |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for value, row in report[key].items():
            ridge = row.get("ridge")
            if ridge is None:
                continue
            lines.append(
                f"| {value} | {row['count']} | {row['b0']['root_corrected']['mean']:.4f} | "
                f"{ridge['root_corrected']['mean']:.4f} | {100*ridge['improve_rate']:.1f}% | "
                f"{100*ridge['sign_accuracy']:.1f}% | {100*ridge['harm_over_5cm_rate']:.1f}% |"
            )

    lines.extend(
        [
            "",
            "## Fold audit",
            "",
            "| Test unit | Train | Test | Blocked | Ridge root | Ridge sign | Ridge harm >5cm |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for fold in report["folds"]:
        ridge = fold.get("methods", {}).get("ridge")
        lines.append(
            f"| {fold['held_out_unit']} | {fold['train_samples']} | {fold['test_samples']} | "
            f"{fold['blocked_not_train_or_test']} | "
            + (
                f"{ridge['root_corrected']['mean']:.4f} | {100*ridge['sign_accuracy']:.1f}% | "
                f"{100*ridge['harm_over_5cm_rate']:.1f}% |"
                if ridge is not None
                else "-- | -- | -- |"
            )
        )

    best_name = min(
        [name for name in ("ridge", "mlp") if name in report["methods"]],
        key=lambda name: report["methods"][name]["root_corrected"]["mean"],
    )
    best = report["methods"][best_name]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"The stronger fixed learned model is `{best_name}` with mean root delta "
            f"`{best['mean_root_delta_m']:+.4f} m`, sign accuracy "
            f"`{100*best['sign_accuracy']:.1f}%`, and signed Pearson "
            f"`{best['prediction_vs_label']['pearson']:.3f}`.",
            "This is a development predictability experiment, not a promoted person-refinement method. ",
            "It uses one capture, only nine directed camera-pair groups, and GT residuals as training labels. ",
            "No capture-disjoint or frozen-sequence generalization has been tested.",
            "",
            "Feature preprocessing is fitted inside each training fold. Camera-pair group, camera/source ID, ",
            "case key, timestamp, person identity, GT camera, and GT geometry are absent from the feature matrix.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if float(args.cap_m) <= 0:
        raise ValueError("--cap_m must be positive")
    report = json.loads(args.matching_report.read_text(encoding="utf-8"))
    protocol = report.get("protocol", {})
    if protocol.get("sequence") != "three":
        raise ValueError("This probe is locked to the `three` development sequence")
    report_cases = list(report["cases"])
    if int(args.max_cases) > 0:
        report_cases = report_cases[: int(args.max_cases)]

    cache_root = SEQUENCE_INPUTS["three"]["cache"]
    rows = []
    cases_used = []
    matcher_correct = 0
    matcher_total = 0
    for case_index, report_case in enumerate(report_cases, start=1):
        case = report_case["case"]
        key = str(case["key"])
        if not key.startswith("three_"):
            raise ValueError(f"Non-development case rejected: {key}")
        cache_path = cache_root / f"{key}.pt"
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        boundary = np.asarray(report_case["boundaries"]["learned_b0"], dtype=np.float64)
        frozen_matching = report_case["matching"]["learned_b0"]

        # Explicitly remove GT and camera/source metadata from the deployment view.
        cache_view = {
            "poses": cache["poses"],
            "humans": cache["humans"],
            "pre_frames": cache["case"]["pre_frames"],
        }
        deployment_rows = build_deployment_rows(cache_view, boundary, frozen_matching)
        group = f"c{int(case['source_camera'])}_c{int(case['target_camera'])}"
        for person in deployment_rows:
            labelled = attach_gt_label(person, cache, boundary)
            labelled["case_key"] = key
            labelled["group"] = group
            labelled["timestamp_evaluator_only"] = int(case["timestamp"])
            labelled["actor_evaluator_only"] = str(labelled["pre_key"])
            rows.append(labelled)
        matcher_row = frozen_matching["matchers"]["root_torso"]
        matcher_correct += int(matcher_row["correct_count"])
        matcher_total += int(matcher_row["person_count"])
        cases_used.append(key)
        print(
            f"[{case_index}/{len(report_cases)}] {key}: group={group} "
            f"people={len(deployment_rows)}",
            flush=True,
        )

    feature_names = sorted(set().union(*(row["features"].keys() for row in rows)))
    validate_feature_contract(feature_names)
    matrix = make_matrix(rows, feature_names)
    labels = np.asarray([row["label_correction_m"] for row in rows], dtype=np.float64)
    groups = np.asarray([row["group"] for row in rows], dtype=object)
    timestamps = np.asarray(
        [row["timestamp_evaluator_only"] for row in rows], dtype=np.int64
    )
    actors = np.asarray(
        [row["actor_evaluator_only"] for row in rows], dtype=object
    )
    if len(set(groups.tolist())) < 2:
        raise RuntimeError("Camera-pair leave-one-group-out requires at least two groups")

    feature_correlations = {
        name: correlation(matrix[:, index], labels)
        for index, name in enumerate(feature_names)
    }
    predictions, folds = cross_validated_predictions(
        matrix, labels, groups, timestamps, actors, args
    )
    if any(not np.isfinite(value).all() for value in predictions.values()):
        raise RuntimeError("At least one sample lacks an out-of-group prediction")

    zero = np.zeros(len(rows), dtype=np.float64)
    methods_full = {
        "b0": method_metrics(zero, rows, args.cap_m),
        "oracle_capped": method_metrics(labels, rows, args.cap_m),
        "oracle": method_metrics(labels, rows, args.cap_m, clip=False),
    }
    for name, values in predictions.items():
        methods_full[name] = method_metrics(values, rows, args.cap_m)

    for fold in folds:
        if fold["status"] != "ok":
            continue
        indices = np.asarray(fold["test_indices"], dtype=np.int64)
        fold_rows = [rows[int(index)] for index in indices]
        fold["methods"] = {
            "b0": metric_subset(
                method_metrics(zero[indices], fold_rows, args.cap_m)
            )
        }
        for name, values in predictions.items():
            fold["methods"][name] = metric_subset(
                method_metrics(values[indices], fold_rows, args.cap_m)
            )

    per_group = {}
    for group in sorted(set(groups.tolist())):
        indices = np.flatnonzero(groups == group)
        group_rows = [rows[int(index)] for index in indices]
        per_group[group] = {
            "count": int(len(indices)),
            "b0": metric_subset(method_metrics(zero[indices], group_rows, args.cap_m)),
        }
        for name, values in predictions.items():
            per_group[group][name] = metric_subset(
                method_metrics(values[indices], group_rows, args.cap_m)
            )

    def breakdown(values: np.ndarray) -> dict:
        output = {}
        for value in sorted(set(values.tolist()), key=str):
            indices = np.flatnonzero(values == value)
            subset_rows = [rows[int(index)] for index in indices]
            item = {
                "count": int(len(indices)),
                "b0": metric_subset(
                    method_metrics(zero[indices], subset_rows, args.cap_m)
                ),
            }
            for name, prediction in predictions.items():
                item[name] = metric_subset(
                    method_metrics(prediction[indices], subset_rows, args.cap_m)
                )
            output[str(value)] = item
        return output

    per_actor = breakdown(actors)
    per_timestamp = breakdown(timestamps)

    sample_rows = []
    for index, row in enumerate(rows):
        sample_rows.append(
            {
                "case_key": row["case_key"],
                "group": row["group"],
                "timestamp_evaluator_only": row["timestamp_evaluator_only"],
                "actor_evaluator_only": row["actor_evaluator_only"],
                "pre_track_key_evaluator_only": row["pre_key"],
                "post_detection_key_evaluator_only": row["post_key"],
                "features": row["features"],
                "label_correction_m": row["label_correction_m"],
                "b0_root_error_m": row["b0_root_error_m"],
                "oracle_root_error_m": row["oracle_root_error_m"],
                **{
                    f"{name}_raw_prediction_m": values[index]
                    for name, values in predictions.items()
                },
                **{
                    f"{name}_correction_m": methods_full[name]["per_sample_correction_m"][index]
                    for name in predictions
                },
                **{
                    f"{name}_root_error_m": methods_full[name]["per_sample_root_error_m"][index]
                    for name in predictions
                },
            }
        )

    output = {
        "experiment": "V14 person-local signed root-ray residual predictability",
        "protocol": {
            "sequence": "three",
            "development_only": True,
            "dance_or_box_loaded": False,
            "case_count": len(cases_used),
            "sample_count": len(rows),
            "group_count": len(set(groups.tolist())),
            "group_counts": dict(Counter(groups.tolist())),
            "split": str(args.split),
            "fold_count": len(folds),
            "prediction_coverage": float(
                np.mean(
                    np.logical_and.reduce(
                        [np.isfinite(value) for value in predictions.values()]
                    )
                )
            ),
            "auto_matcher": "frozen learned_b0/root_torso",
            "auto_matcher_accuracy_evaluator_only": matcher_correct / max(matcher_total, 1),
            "cap_m": float(args.cap_m),
            "ridge_alpha_fixed": float(args.ridge_alpha),
            "mlp_structure_fixed": [32, 16],
            "mlp_alpha_fixed": float(args.mlp_alpha),
            "seed": int(args.seed),
            "models": list(predictions),
            "label": "GT-only signed rigid translation along frozen B0 post-person camera ray",
            "camera_changed": False,
            "gt_feature_leakage": False,
            "forbidden_feature_tokens": FORBIDDEN_FEATURE_TOKENS,
            "feature_count": len(feature_names),
            "feature_names": feature_names,
            "cases": cases_used,
        },
        "label_stats": finite_stats(labels),
        "feature_correlations": feature_correlations,
        "folds": folds,
        "methods": {name: metric_subset(value) for name, value in methods_full.items()},
        "per_group": per_group,
        "per_actor": per_actor,
        "per_timestamp": per_timestamp,
        "samples": sample_rows,
        "limitations": [
            "Only one MultiHuman capture and the `three` development sequence are used.",
            "The nine held-out groups are directed camera pairs, not capture-disjoint datasets.",
            "GT supplies every training label and evaluator target, but no deployment feature.",
            "The controlled 41-cut subset excludes changing visibility sets and is easier for identity matching.",
            "No dance/box result, frozen holdout result, runtime head, or promoted inference method is claimed.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "v14_person_residual_head_cv.json"
    md_path = args.output_dir / "v14_person_residual_head_cv.md"
    json_path.write_text(
        json.dumps(jsonable(output), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown = markdown_report(output)
    md_path.write_text(markdown + "\n", encoding="utf-8")
    print(markdown, flush=True)
    print(f">> wrote {json_path}", flush=True)


if __name__ == "__main__":
    main()
