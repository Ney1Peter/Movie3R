#!/usr/bin/env python3
"""V14 invariant person root-ray residual predictability on ``three`` only.

This v2 probe removes camera-pair/capture shortcuts.  Its deployment feature
whitelist contains only relative, body-normalized, or dimensionless cues:

* pre/post bbox and predicted-body extent log ratios;
* root jump, velocity, and acceleration in the last pre torso frame, normalized
  by predicted body size;
* pre/post camera-local ray angle (never either absolute ray);
* frozen matcher costs/margins and prediction completeness;
* pre-to-post body-relative torso/root rotation;
* other explicitly named dimensionless ratios.

It excludes camera baseline, B0/camera rotation parameters, absolute root/ray/
range, absolute bbox location/size, absolute pose axes, and all camera/source/
person/time IDs.  GT is available only to create the signed root-ray label,
define timestamp+actor double-block folds, and evaluate OOF predictions.
``dance`` and ``box`` cannot be selected or loaded by this script.
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
from scipy.stats import pearsonr, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from versions.v14.probe_b0_anchor_conflict import SEQUENCE_INPUTS  # noqa: E402
from versions.v14.probe_person_residual_head_cv import (  # noqa: E402
    safe_rotvec,
    transform_points,
)


DEFAULT_MATCHING = (
    REPO_ROOT / "output/v14/b0_identity_matching/v14_b0_identity_matching.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "output/v14/fine_alignment_research/person_residual_head_invariant_cv_three"
)
RIDGE_ALPHA = 10.0
HUBER_ALPHA = 0.01
HUBER_EPSILON = 1.35
ACTION_CAP_M = 0.05
LOGISTIC_C = 1.0
SIGN_NEUTRAL_M = 0.02
LOGISTIC_GATE_THRESHOLDS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
FORBIDDEN_FEATURE_TOKENS = (
    "camera",
    "baseline",
    "b0",
    "absolute",
    "root_local",
    "range",
    "bbox_center",
    "root_ray_x",
    "root_ray_y",
    "root_ray_z",
    "pose_axis",
    "torso_axis",
    "identity",
    "actor",
    "person",
    "timestamp",
    "source",
    "target",
    "case_key",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matching_report", type=Path, default=DEFAULT_MATCHING)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("ridge", "huber", "logistic"),
        default=("ridge", "huber", "logistic"),
    )
    parser.add_argument("--seed", type=int, default=20260731)
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


def finite_stats(values: np.ndarray | list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {
            "count": 0,
            "mean": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
        }
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
    }


def correlation(first: np.ndarray, second: np.ndarray) -> dict:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    finite = np.isfinite(first) & np.isfinite(second)
    first, second = first[finite], second[finite]
    if len(first) < 4 or np.std(first) < 1e-12 or np.std(second) < 1e-12:
        return {"count": int(len(first)), "pearson": float("nan"), "spearman": float("nan")}
    return {
        "count": int(len(first)),
        "pearson": float(pearsonr(first, second).statistic),
        "spearman": float(spearmanr(first, second).statistic),
    }


def body_observation(human: dict) -> dict:
    root = np.asarray(human["root"], dtype=np.float64)
    torso = np.asarray(human["torso"], dtype=np.float64)
    bbox = np.asarray(human["bbox"], dtype=np.float64)
    bbox_width = max(float(bbox[2] - bbox[0]), 1e-8)
    bbox_height = max(float(bbox[3] - bbox[1]), 1e-8)
    output = {
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "bbox_area": bbox_width * bbox_height,
        "bbox_aspect": bbox_width / bbox_height,
        "completeness": float(human.get("completeness", float("nan"))),
    }
    for name in ("joints", "vertices"):
        centered_body = (np.asarray(human[name], dtype=np.float64) - root) @ torso
        extent = np.maximum(np.ptp(centered_body, axis=0), 1e-8)
        output[f"{name}_rms"] = float(
            np.sqrt(np.mean(np.sum(centered_body**2, axis=1)))
        )
        for axis, value in zip(("x", "y", "z"), extent):
            output[f"{name}_extent_{axis}"] = float(value)
        output[f"{name}_shape_x_over_y"] = float(extent[0] / extent[1])
        output[f"{name}_shape_z_over_y"] = float(extent[2] / extent[1])
    output["vertices_over_joints_rms"] = float(
        output["vertices_rms"] / max(output["joints_rms"], 1e-8)
    )
    return output


def log_ratio(first: float, second: float) -> float:
    return float(np.log(max(float(second), 1e-8) / max(float(first), 1e-8)))


def add_vector(features: dict[str, float], prefix: str, vector: np.ndarray) -> None:
    for axis, value in zip(("x", "y", "z"), np.asarray(vector).reshape(3)):
        features[f"{prefix}_{axis}"] = float(value)


def validate_whitelist(feature_names: list[str]) -> None:
    violations = [
        name
        for name in feature_names
        if any(token in name.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    if violations:
        raise RuntimeError(f"Invariant feature whitelist violation: {violations}")
    required_prefixes = (
        "cross_view_",
        "pre_temporal_",
        "motion_body_norm_",
        "body_relative_",
        "matcher_",
        "completeness_",
        "dimensionless_",
    )
    unclassified = [
        name for name in feature_names if not name.startswith(required_prefixes)
    ]
    if unclassified:
        raise RuntimeError(f"Features outside explicit invariant classes: {unclassified}")


def invariant_matcher_pairs(
    cache_view: dict, matching: dict
) -> list[tuple[str, str, dict[str, float]]]:
    """Frozen assignments with rank/self-normalized margins only.

    Raw metric root/joint costs, mixed-unit totals, and raw margins are omitted.
    Person keys route predictions but are not emitted as features.
    """
    matcher = matching["matchers"]["root_torso"]
    assignment = matcher["predicted_identity_by_pre_identity"]
    row_order = [str(value) for value in matching["identities"]]
    column_order = [str(value) for value in matching["post_detection_gt_order"]]
    costs = np.asarray(matcher["cost"], dtype=np.float64)
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
        row_values = costs[row]
        chosen = float(row_values[column])
        sorted_values = np.sort(row_values)
        median = float(np.median(row_values))
        mad = max(float(np.median(np.abs(row_values - median))), 1e-6)
        selected_rank = 1 + int(np.sum(row_values < chosen - 1e-12))
        assignment_scale = max(abs(float(matcher["best_assignment_cost"])), 1e-6)
        features = {
            "matcher_selected_rank_fraction": float(selected_rank / len(row_values)),
            "matcher_row_chosen_minus_best_over_mad": float(
                (chosen - sorted_values[0]) / mad
            ),
            "matcher_row_best_second_margin_over_mad": float(
                (sorted_values[1] - sorted_values[0]) / mad
                if len(sorted_values) > 1
                else 0.0
            ),
            "matcher_assignment_margin_over_cost": float(
                matcher["best_vs_second_margin"] / assignment_scale
            ),
            "matcher_assignment_valid_fraction": float(
                int(matcher["person_count"]) / max(len(column_order), 1)
            ),
        }
        output.append((pre_key, post_key, features))
    return output


def invariant_deployment_rows(
    cache_view: dict, boundary: np.ndarray, matching: dict
) -> list[dict]:
    """GT-free invariant feature builder; IDs are routing keys, never features."""
    poses = [np.asarray(value, dtype=np.float64) for value in cache_view["poses"]]
    humans = cache_view["humans"]
    frames = [int(value) for value in cache_view["pre_frames"]]
    pre_pose = poses[-2]
    raw_post_pose = poses[-1]
    rows = []
    for pre_key, post_key, matcher_features in invariant_matcher_pairs(
        cache_view, matching
    ):
        history = [
            (index, frames[index], frame[pre_key])
            for index, frame in enumerate(humans[:-1])
            if pre_key in frame
        ][-3:]
        if not history:
            continue
        last_index, last_frame, pre_human = history[-1]
        post_human = humans[-1][post_key]
        pre_observation = body_observation(pre_human)
        post_observation = body_observation(post_human)
        # ``assignment_cost_per_person`` is a scalar normalization from the
        # frozen matcher, not a person identifier.  Use a neutral deployment
        # name so the static leakage contract can continue to reject actual
        # identity/person fields without a false positive.
        features = {
            str(name).replace("_per_person", "_per_track"): float(value)
            for name, value in matcher_features.items()
        }

        ratio_names = (
            "bbox_width",
            "bbox_height",
            "bbox_area",
            "bbox_aspect",
            "joints_rms",
            "vertices_rms",
            "joints_extent_x",
            "joints_extent_y",
            "joints_extent_z",
            "vertices_extent_x",
            "vertices_extent_y",
            "vertices_extent_z",
            "joints_shape_x_over_y",
            "joints_shape_z_over_y",
            "vertices_shape_x_over_y",
            "vertices_shape_z_over_y",
            "vertices_over_joints_rms",
        )
        for name in ratio_names:
            features[f"cross_view_{name}_log_ratio"] = log_ratio(
                pre_observation[name], post_observation[name]
            )
        features["completeness_pre"] = pre_observation["completeness"]
        features["completeness_post"] = post_observation["completeness"]
        features["completeness_delta"] = (
            post_observation["completeness"] - pre_observation["completeness"]
        )
        features["dimensionless_completeness_ratio"] = float(
            post_observation["completeness"]
            / max(pre_observation["completeness"], 1e-8)
        )

        if len(history) >= 2:
            earlier_observation = body_observation(history[-2][2])
            for name in ratio_names:
                features[f"pre_temporal_{name}_log_ratio"] = log_ratio(
                    earlier_observation[name], pre_observation[name]
                )

        pre_root = np.asarray(pre_human["root"], dtype=np.float64)
        post_root_raw = np.asarray(post_human["root"], dtype=np.float64)
        post_root_aligned = transform_points(boundary, post_root_raw)
        pre_torso = np.asarray(pre_human["torso"], dtype=np.float64)
        body_scale = max(float(pre_observation["vertices_rms"]), 1e-8)
        jump_body_normalized = (post_root_aligned - pre_root) @ pre_torso / body_scale
        add_vector(features, "motion_body_norm_root_jump", jump_body_normalized)
        features["motion_body_norm_root_jump_norm"] = float(
            np.linalg.norm(jump_body_normalized)
        )

        if len(history) >= 2:
            first_index, first_frame, first_human = history[0]
            frame_delta = max(int(last_frame - first_frame), 1)
            velocity = (
                pre_root - np.asarray(first_human["root"], dtype=np.float64)
            ) / frame_delta
            velocity_body_normalized = velocity @ pre_torso / body_scale
            add_vector(
                features, "motion_body_norm_velocity_per_frame", velocity_body_normalized
            )
            features["motion_body_norm_velocity_per_frame_norm"] = float(
                np.linalg.norm(velocity_body_normalized)
            )
        if len(history) >= 3:
            roots = [np.asarray(item[2]["root"], dtype=np.float64) for item in history]
            frame_numbers = [int(item[1]) for item in history]
            velocity0 = (roots[1] - roots[0]) / max(
                frame_numbers[1] - frame_numbers[0], 1
            )
            velocity1 = (roots[2] - roots[1]) / max(
                frame_numbers[2] - frame_numbers[1], 1
            )
            acceleration_body_normalized = (velocity1 - velocity0) @ pre_torso / body_scale
            add_vector(
                features,
                "motion_body_norm_acceleration_per_frame2",
                acceleration_body_normalized,
            )
            features["motion_body_norm_acceleration_per_frame2_norm"] = float(
                np.linalg.norm(acceleration_body_normalized)
            )

        pre_ray_world = pre_root - poses[last_index][:3, 3]
        pre_ray_world /= max(float(np.linalg.norm(pre_ray_world)), 1e-8)
        final_post_pose = np.asarray(boundary) @ raw_post_pose
        post_ray_world = post_root_aligned - final_post_pose[:3, 3]
        post_ray_world /= max(float(np.linalg.norm(post_ray_world)), 1e-8)
        ray_cosine = float(np.clip(pre_ray_world @ post_ray_world, -1.0, 1.0))
        features["cross_view_ray_cosine"] = ray_cosine
        features["cross_view_ray_angle_sin"] = float(
            np.sqrt(max(1.0 - ray_cosine * ray_cosine, 0.0))
        )

        aligned_post_torso = np.asarray(boundary)[:3, :3] @ np.asarray(
            post_human["torso"], dtype=np.float64
        )
        torso_relative = pre_torso.T @ aligned_post_torso
        torso_rotvec = safe_rotvec(torso_relative)
        add_vector(features, "body_relative_torso_rotvec", torso_rotvec)
        features["body_relative_torso_angle_sin"] = float(
            np.sin(np.linalg.norm(torso_rotvec))
        )
        features["body_relative_torso_angle_cos"] = float(
            np.cos(np.linalg.norm(torso_rotvec))
        )
        if "root_rotation" in pre_human and "root_rotation" in post_human:
            aligned_post_root_rotation = np.asarray(boundary)[:3, :3] @ np.asarray(
                post_human["root_rotation"], dtype=np.float64
            )
            root_relative = np.asarray(pre_human["root_rotation"]).T @ aligned_post_root_rotation
            root_rotvec = safe_rotvec(root_relative)
            add_vector(features, "body_relative_root_rotvec", root_rotvec)
            features["body_relative_root_angle_sin"] = float(
                np.sin(np.linalg.norm(root_rotvec))
            )
            features["body_relative_root_angle_cos"] = float(
                np.cos(np.linalg.norm(root_rotvec))
            )

        final_root = post_root_aligned
        final_center = final_post_pose[:3, 3]
        ray_vector = final_root - final_center
        ray = ray_vector / max(float(np.linalg.norm(ray_vector)), 1e-8)
        rows.append(
            {
                "pre_key": pre_key,
                "post_key": post_key,
                "features": features,
                "root": final_root,
                "ray": ray,
            }
        )
    return rows


def attach_label(deployment: dict, cache: dict) -> dict:
    gt = cache["gt"]
    pre_pose = np.asarray(cache["poses"][-2], dtype=np.float64)
    gauge = pre_pose @ np.linalg.inv(np.asarray(gt["pre_c2w"], dtype=np.float64))
    target_root = transform_points(
        gauge, gt["post_humans"][deployment["pre_key"]]["root"]
    )
    root = np.asarray(deployment["root"], dtype=np.float64)
    ray = np.asarray(deployment["ray"], dtype=np.float64)
    return {
        **deployment,
        "target_root": target_root,
        "label_correction_m": float(np.dot(target_root - root, ray)),
        "b0_root_error_m": float(np.linalg.norm(root - target_root)),
    }


def feature_matrix(rows: list[dict], names: list[str]) -> np.ndarray:
    return np.asarray(
        [[row["features"].get(name, float("nan")) for name in names] for row in rows],
        dtype=np.float64,
    )


def regression_pipeline(model) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )


def double_block_oof(
    matrix: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
    actors: np.ndarray,
    models: tuple[str, ...],
    seed: int,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    predictions = {
        "train_fold_mean": np.full(len(labels), np.nan, dtype=np.float64),
        "train_fold_median": np.full(len(labels), np.nan, dtype=np.float64),
        "train_fold_majority_sign": np.full(len(labels), np.nan, dtype=np.float64),
    }
    for name in models:
        predictions[name] = np.full(len(labels), np.nan, dtype=np.float64)
    if "logistic" in models:
        predictions["logistic_probability_positive"] = np.full(
            len(labels), np.nan, dtype=np.float64
        )

    folds = []
    units = sorted(
        set(zip(timestamps.tolist(), actors.tolist())),
        key=lambda item: (int(item[0]), str(item[1])),
    )
    for timestamp, actor in units:
        test = (timestamps == timestamp) & (actors == actor)
        train = (timestamps != timestamp) & (actors != actor)
        blocked = ~(train | test)
        train_x, test_x = matrix[train], matrix[test]
        train_y = labels[train]
        target_mean = float(np.mean(train_y))
        target_scale = max(float(np.std(train_y)), 1e-8)
        predictions["train_fold_mean"][test] = target_mean
        predictions["train_fold_median"][test] = float(np.median(train_y))
        nonneutral_for_prior = np.abs(train_y) >= SIGN_NEUTRAL_M
        positive_fraction = float(
            np.mean(train_y[nonneutral_for_prior] >= 0.0)
        ) if np.any(nonneutral_for_prior) else 0.5
        predictions["train_fold_majority_sign"][test] = (
            ACTION_CAP_M if positive_fraction >= 0.5 else -ACTION_CAP_M
        )

        if "ridge" in models:
            model = regression_pipeline(Ridge(alpha=RIDGE_ALPHA))
            model.fit(train_x, (train_y - target_mean) / target_scale)
            predictions["ridge"][test] = (
                model.predict(test_x) * target_scale + target_mean
            )
        if "huber" in models:
            model = regression_pipeline(
                HuberRegressor(
                    epsilon=HUBER_EPSILON,
                    alpha=HUBER_ALPHA,
                    max_iter=2000,
                )
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(train_x, (train_y - target_mean) / target_scale)
            predictions["huber"][test] = (
                model.predict(test_x) * target_scale + target_mean
            )
        if "logistic" in models:
            nonneutral = np.abs(train_y) >= SIGN_NEUTRAL_M
            classes = (train_y[nonneutral] >= 0.0).astype(np.int64)
            if len(classes) == 0:
                probability = np.full(int(np.sum(test)), 0.5)
            elif len(np.unique(classes)) == 1:
                probability = np.full(int(np.sum(test)), float(classes[0]))
            else:
                classifier = regression_pipeline(
                    LogisticRegression(
                        C=LOGISTIC_C,
                        max_iter=1000,
                        random_state=int(seed),
                    )
                )
                classifier.fit(train_x[nonneutral], classes)
                probability = classifier.predict_proba(test_x)[:, 1]
            predictions["logistic_probability_positive"][test] = probability
            predictions["logistic"][test] = np.where(
                probability >= 0.5, ACTION_CAP_M, -ACTION_CAP_M
            )

        folds.append(
            {
                "held_out_unit": f"t{int(timestamp)}_{actor}",
                "train_samples": int(np.sum(train)),
                "test_samples": int(np.sum(test)),
                "blocked_samples": int(np.sum(blocked)),
                "train_label_mean_m": target_mean,
                "train_abs_label_median_m": float(np.median(np.abs(train_y))),
                "train_logistic_nonneutral": int(
                    np.sum(np.abs(train_y) >= SIGN_NEUTRAL_M)
                ),
                "test_indices": np.flatnonzero(test),
            }
        )
    return predictions, folds


def leave_camera_pair_oof(
    matrix: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Secondary proxy audit; never used to select the main candidate."""
    predictions = {
        "train_fold_mean": np.full(len(labels), np.nan, dtype=np.float64),
        "ridge": np.full(len(labels), np.nan, dtype=np.float64),
        "logistic": np.full(len(labels), np.nan, dtype=np.float64),
        "logistic_probability_positive": np.full(
            len(labels), np.nan, dtype=np.float64
        ),
    }
    folds = []
    for group in sorted(set(groups.tolist())):
        test = groups == group
        train = ~test
        train_x, test_x = matrix[train], matrix[test]
        train_y = labels[train]
        target_mean = float(np.mean(train_y))
        target_scale = max(float(np.std(train_y)), 1e-8)
        predictions["train_fold_mean"][test] = target_mean

        ridge = regression_pipeline(Ridge(alpha=RIDGE_ALPHA))
        ridge.fit(train_x, (train_y - target_mean) / target_scale)
        predictions["ridge"][test] = (
            ridge.predict(test_x) * target_scale + target_mean
        )

        nonneutral = np.abs(train_y) >= SIGN_NEUTRAL_M
        classes = (train_y[nonneutral] >= 0.0).astype(np.int64)
        if len(classes) == 0:
            probability = np.full(int(np.sum(test)), 0.5)
        elif len(np.unique(classes)) == 1:
            probability = np.full(int(np.sum(test)), float(classes[0]))
        else:
            logistic = regression_pipeline(
                LogisticRegression(
                    C=LOGISTIC_C,
                    max_iter=1000,
                    random_state=int(seed),
                )
            )
            logistic.fit(train_x[nonneutral], classes)
            probability = logistic.predict_proba(test_x)[:, 1]
        predictions["logistic_probability_positive"][test] = probability
        predictions["logistic"][test] = np.where(
            probability >= 0.5, ACTION_CAP_M, -ACTION_CAP_M
        )
        folds.append(
            {
                "held_out_camera_pair": str(group),
                "train_samples": int(np.sum(train)),
                "test_samples": int(np.sum(test)),
            }
        )
    return predictions, folds


def method_metrics(action: np.ndarray, score: np.ndarray, rows: list[dict]) -> dict:
    action = np.asarray(action, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    labels = np.asarray([row["label_correction_m"] for row in rows], dtype=np.float64)
    before = np.asarray([row["b0_root_error_m"] for row in rows], dtype=np.float64)
    roots = np.stack([row["root"] for row in rows])
    rays = np.stack([row["ray"] for row in rows])
    targets = np.stack([row["target_root"] for row in rows])
    after = np.linalg.norm(roots + action[:, None] * rays - targets, axis=1)
    delta = after - before
    accepted = np.abs(action) > 1e-12
    valid_sign = accepted & (np.abs(labels) >= SIGN_NEUTRAL_M)
    return {
        "count": len(rows),
        "coverage": float(np.mean(accepted)),
        "action": finite_stats(action),
        "root_before": finite_stats(before),
        "root_corrected": finite_stats(after),
        "mean_root_delta_m": float(np.mean(delta)),
        "improve_rate": float(np.mean(after < before - 1e-12)),
        "worsen_rate": float(np.mean(after > before + 1e-12)),
        "harm_over_1cm_rate": float(np.mean(delta > 0.01)),
        "harm_over_5cm_rate": float(np.mean(delta > 0.05)),
        "harm_over_10cm_rate": float(np.mean(delta > 0.10)),
        "accepted_sign_accuracy": float(
            np.mean(np.sign(action[valid_sign]) == np.sign(labels[valid_sign]))
        )
        if np.any(valid_sign)
        else float("nan"),
        "accepted_improve_rate": float(np.mean(after[accepted] < before[accepted]))
        if np.any(accepted)
        else float("nan"),
        "score_vs_label": correlation(score, labels),
        "per_sample_action_m": action,
        "per_sample_root_error_m": after,
    }


def without_samples(metrics: dict) -> dict:
    return {
        key: value
        for key, value in metrics.items()
        if not key.startswith("per_sample_")
    }


def regression_action(prediction: np.ndarray) -> np.ndarray:
    return np.clip(
        np.asarray(prediction, dtype=np.float64), -ACTION_CAP_M, ACTION_CAP_M
    )


def logistic_gate_grid(
    probabilities: np.ndarray,
    rows: list[dict],
    timestamps: np.ndarray,
    actors: np.ndarray,
) -> tuple[dict[str, dict], str | None]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    confidence = np.abs(2.0 * probabilities - 1.0)
    sign_action = np.where(probabilities >= 0.5, ACTION_CAP_M, -ACTION_CAP_M)
    grid = {}
    eligible = []
    for threshold in LOGISTIC_GATE_THRESHOLDS:
        action = np.where(confidence >= threshold, sign_action, 0.0)
        metrics = method_metrics(action, 2.0 * probabilities - 1.0, rows)
        relative_gain = -metrics["mean_root_delta_m"] / max(
            metrics["root_before"]["mean"], 1e-12
        )
        unit_noninferior = True
        for values in (timestamps, actors):
            for value in set(values.tolist()):
                indices = np.flatnonzero(values == value)
                subset = [rows[int(index)] for index in indices]
                unit_metrics = method_metrics(
                    action[indices],
                    (2.0 * probabilities - 1.0)[indices],
                    subset,
                )
                if unit_metrics["mean_root_delta_m"] > 1e-12:
                    unit_noninferior = False
                    break
            if not unit_noninferior:
                break
        key = f"confidence_{threshold:.2f}"
        grid[key] = {
            **without_samples(metrics),
            "relative_root_gain": relative_gain,
            "all_actor_timestamp_means_noninferior": unit_noninferior,
            "root_protocol_eligible": bool(
                metrics["coverage"] >= 0.15
                and metrics["mean_root_delta_m"] <= -0.020
                and relative_gain >= 0.05
                and metrics["harm_over_5cm_rate"] <= 0.05
                and metrics["accepted_sign_accuracy"] >= 0.85
                and metrics["accepted_improve_rate"] >= 0.80
                and metrics["root_corrected"]["p95"]
                <= metrics["root_before"]["p95"]
                and unit_noninferior
            ),
            "full_geometry_eligible": False,
        }
        if (
            grid[key]["root_protocol_eligible"]
        ):
            eligible.append((metrics["root_corrected"]["mean"], key))
    selected = min(eligible)[1] if eligible else None
    return grid, selected


def breakdown(
    values: np.ndarray,
    rows: list[dict],
    actions: dict[str, np.ndarray],
    scores: dict[str, np.ndarray],
) -> dict:
    output = {}
    for value in sorted(set(values.tolist()), key=str):
        indices = np.flatnonzero(values == value)
        subset = [rows[int(index)] for index in indices]
        output[str(value)] = {
            "count": int(len(indices)),
            "methods": {
                name: without_samples(
                    method_metrics(action[indices], scores[name][indices], subset)
                )
                for name, action in actions.items()
            },
        }
    return output


def markdown_report(report: dict) -> str:
    protocol = report["protocol"]
    methods = report["methods"]
    lines = [
        "# V14 Invariant Person Residual Head — `three` Double-Block CV",
        "",
        f"Cases/people/features/folds: `{protocol['case_count']}/{protocol['person_count']}/"
        f"{protocol['feature_count']}/{protocol['fold_count']}`. Prediction coverage is 100%.",
        "Only relative/body-normalized/dimensionless deployment features are present. "
        "Timestamp and actor are split/report metadata only. `dance` and `box` are not loaded.",
        "",
        "All ungated actions are capped at `±0.05 m`. Logistic uses a fixed sign step. "
        "Regression magnitudes and classifiers are fitted inside each training fold.",
        "",
        "## OOF comparison",
        "",
        "| Method | Root mean | P50 | P90 | Delta | Sign | Improve | Harm >1cm | Pearson | Feature gain vs mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    mean_root = methods["train_fold_mean"]["root_corrected"]["mean"]
    order = [
        "b0",
        "train_fold_mean",
        "train_fold_median",
        "train_fold_majority_sign",
        "ridge",
        "huber",
        "logistic",
    ]
    if report["selected_logistic_gate"] is not None:
        order.append("logistic_selected_gate")
    for name in order:
        if name not in methods:
            continue
        row = methods[name]
        lines.append(
            f"| {name} | {row['root_corrected']['mean']:.4f} | "
            f"{row['root_corrected']['p50']:.4f} | {row['root_corrected']['p90']:.4f} | "
            f"{row['mean_root_delta_m']:+.4f} | {100*row['accepted_sign_accuracy']:.1f}% | "
            f"{100*row['improve_rate']:.1f}% | {100*row['harm_over_1cm_rate']:.1f}% | "
            f"{row['score_vs_label']['pearson']:.3f} | "
            f"{mean_root-row['root_corrected']['mean']:+.4f} |"
        )

    lines.extend(
        [
            "",
            "## Strongest invariant feature correlations",
            "",
            "| Feature | Pearson | Spearman |",
            "|---|---:|---:|",
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
        lines.append(
            f"| `{name}` | {row['pearson']:.3f} | {row['spearman']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Logistic confidence grid",
            "",
            "| Gate | Coverage | Root | Sign | Improve | Harm >1cm |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, row in report["logistic_gate_grid"].items():
        lines.append(
            f"| {name} | {100*row['coverage']:.1f}% | "
            f"{row['root_corrected']['mean']:.4f} | "
            f"{100*row['accepted_sign_accuracy']:.1f}% | "
            f"{100*row['improve_rate']:.1f}% | {100*row['harm_over_1cm_rate']:.1f}% |"
        )
    lines.append("")
    lines.append(
        f"Selected development-only logistic gate: `{report['selected_logistic_gate']}`."
    )

    lines.extend(
        [
            "",
            "## Leave-camera-pair-out proxy audit",
            "",
            "| Method | Root | Delta | Sign | Harm >1cm | Pearson |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, row in report["camera_pair_proxy_audit"]["methods"].items():
        lines.append(
            f"| {name} | {row['root_corrected']['mean']:.4f} | "
            f"{row['mean_root_delta_m']:+.4f} | "
            f"{100*row['accepted_sign_accuracy']:.1f}% | "
            f"{100*row['harm_over_1cm_rate']:.1f}% | "
            f"{row['score_vs_label']['pearson']:.3f} |"
        )

    for title, key in (("Per actor", "per_actor"), ("Per timestamp", "per_timestamp")):
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Unit | N | Mean baseline | Ridge | Logistic | Selected logistic |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for unit, block in report[key].items():
            unit_methods = block["methods"]
            selected = unit_methods.get("logistic_selected_gate")
            lines.append(
                f"| {unit} | {block['count']} | "
                f"{unit_methods['train_fold_mean']['root_corrected']['mean']:.4f} | "
                f"{unit_methods['ridge']['root_corrected']['mean']:.4f} | "
                f"{unit_methods['logistic']['root_corrected']['mean']:.4f} | "
                + (
                    f"{selected['root_corrected']['mean']:.4f} |"
                    if selected is not None
                    else "-- |"
                )
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            report["interpretation"],
            "",
            "This is development model-selection evidence only. No final model is saved and no frozen split is inspected.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    matching = json.loads(args.matching_report.read_text(encoding="utf-8"))
    if matching.get("protocol", {}).get("sequence") != "three":
        raise ValueError("This probe is locked to the `three` development report")
    report_cases = list(matching["cases"])
    if int(args.max_cases) > 0:
        report_cases = report_cases[: int(args.max_cases)]
    cache_root = SEQUENCE_INPUTS["three"]["cache"]

    rows = []
    case_keys = []
    for case_index, report_case in enumerate(report_cases, start=1):
        case = report_case["case"]
        key = str(case["key"])
        if not key.startswith("three_"):
            raise ValueError(f"Non-development input rejected: {key}")
        cache = torch.load(
            cache_root / f"{key}.pt", map_location="cpu", weights_only=False
        )
        boundary = np.asarray(
            report_case["boundaries"]["learned_b0"], dtype=np.float64
        )
        cache_view = {
            "poses": cache["poses"],
            "humans": cache["humans"],
            "pre_frames": cache["case"]["pre_frames"],
        }
        deployment_rows = invariant_deployment_rows(
            cache_view, boundary, report_case["matching"]["learned_b0"]
        )
        # GT and split metadata enter only after invariant deployment features
        # and frozen rays have been built.
        for deployment in deployment_rows:
            row = attach_label(deployment, cache)
            row["case_key_evaluator_only"] = key
            row["timestamp_evaluator_only"] = int(case["timestamp"])
            row["actor_evaluator_only"] = str(deployment["pre_key"])
            row["camera_pair_evaluator_only"] = (
                f"c{int(case['source_camera'])}_c{int(case['target_camera'])}"
            )
            rows.append(row)
        case_keys.append(key)
        print(
            f"[{case_index}/{len(report_cases)}] {key}: people={len(deployment_rows)}",
            flush=True,
        )

    names = sorted(set().union(*(row["features"].keys() for row in rows)))
    validate_whitelist(names)
    matrix = feature_matrix(rows, names)
    labels = np.asarray([row["label_correction_m"] for row in rows], dtype=np.float64)
    timestamps = np.asarray(
        [row["timestamp_evaluator_only"] for row in rows], dtype=np.int64
    )
    actors = np.asarray([row["actor_evaluator_only"] for row in rows], dtype=object)
    camera_pairs = np.asarray(
        [row["camera_pair_evaluator_only"] for row in rows], dtype=object
    )
    models = tuple(dict.fromkeys(args.models))
    predictions, folds = double_block_oof(
        matrix, labels, timestamps, actors, models, int(args.seed)
    )
    if any(not np.isfinite(value).all() for value in predictions.values()):
        raise RuntimeError("Incomplete OOF prediction coverage")

    actions = {
        "b0": np.zeros(len(rows), dtype=np.float64),
        "train_fold_mean": regression_action(predictions["train_fold_mean"]),
        "train_fold_median": regression_action(predictions["train_fold_median"]),
        "train_fold_majority_sign": predictions["train_fold_majority_sign"],
    }
    scores = {
        "b0": np.zeros(len(rows), dtype=np.float64),
        "train_fold_mean": predictions["train_fold_mean"],
        "train_fold_median": predictions["train_fold_median"],
        "train_fold_majority_sign": predictions["train_fold_majority_sign"],
    }
    for name in ("ridge", "huber"):
        if name in predictions:
            actions[name] = regression_action(predictions[name])
            scores[name] = predictions[name]
    logistic_grid, selected_gate = {}, None
    if "logistic" in predictions:
        probabilities = predictions["logistic_probability_positive"]
        actions["logistic"] = predictions["logistic"]
        scores["logistic"] = 2.0 * probabilities - 1.0
        logistic_grid, selected_gate = logistic_gate_grid(
            probabilities, rows, timestamps, actors
        )
        if selected_gate is not None:
            threshold = float(selected_gate.rsplit("_", 1)[-1])
            confidence = np.abs(2.0 * probabilities - 1.0)
            actions["logistic_selected_gate"] = np.where(
                confidence >= threshold, predictions["logistic"], 0.0
            )
            scores["logistic_selected_gate"] = scores["logistic"]

    method_results = {
        name: without_samples(method_metrics(action, scores[name], rows))
        for name, action in actions.items()
    }
    correlations = {
        name: correlation(matrix[:, index], labels) for index, name in enumerate(names)
    }
    mean_root = method_results["train_fold_mean"]["root_corrected"]["mean"]
    feature_models = [name for name in ("ridge", "huber", "logistic") if name in method_results]
    best_feature = min(
        feature_models, key=lambda name: method_results[name]["root_corrected"]["mean"]
    )
    feature_increment = mean_root - method_results[best_feature]["root_corrected"]["mean"]
    if feature_increment <= 0.0:
        interpretation = (
            f"Best invariant feature model `{best_feature}` is "
            f"{1000*(-feature_increment):.1f} mm worse than the train-fold mean baseline. "
            "The apparent gain over B0 is a capture-level signed prior, not a learned individual residual."
        )
    elif feature_increment <= 0.005:
        interpretation = (
            f"Best invariant feature model `{best_feature}` improves only "
            f"{1000*feature_increment:.1f} mm over the train-fold mean baseline. "
            "The residual remains dominated by a capture-level signed prior; invariant individual cues "
            "do not yet provide a material safe estimator."
        )
    else:
        interpretation = (
            f"Best invariant feature model `{best_feature}` adds "
            f"{1000*feature_increment:.1f} mm over the train-fold mean baseline. "
            "This is nonzero development evidence, but it still requires a safe confidence gate and frozen validation."
        )

    pair_predictions, pair_folds = leave_camera_pair_oof(
        matrix, labels, camera_pairs, int(args.seed)
    )
    pair_actions = {
        "train_fold_mean": regression_action(pair_predictions["train_fold_mean"]),
        "ridge": regression_action(pair_predictions["ridge"]),
        "logistic": pair_predictions["logistic"],
    }
    pair_scores = {
        "train_fold_mean": pair_predictions["train_fold_mean"],
        "ridge": pair_predictions["ridge"],
        "logistic": 2.0 * pair_predictions["logistic_probability_positive"] - 1.0,
    }
    pair_audit = {
        "split": "leave_one_camera_pair_out_proxy_audit",
        "folds": pair_folds,
        "methods": {
            name: without_samples(method_metrics(action, pair_scores[name], rows))
            for name, action in pair_actions.items()
        },
        "per_camera_pair": breakdown(
            camera_pairs, rows, pair_actions, pair_scores
        ),
        "used_for_candidate_selection": False,
    }

    output = {
        "experiment": "V14 invariant person residual-head double-block predictability",
        "protocol": {
            "sequence": "three",
            "development_only": True,
            "dance_or_box_loaded": False,
            "case_count": len(case_keys),
            "person_count": len(rows),
            "feature_count": len(names),
            "feature_names": names,
            "feature_whitelist_classes": [
                "cross_view relative log ratios/ray angle",
                "pre temporal log ratios",
                "body-frame normalized root motion",
                "body-relative rotations",
                "matcher costs and margins",
                "prediction completeness",
                "dimensionless ratios",
            ],
            "forbidden_feature_tokens": FORBIDDEN_FEATURE_TOKENS,
            "split": "double_block_timestamp_actor",
            "fold_count": len(folds),
            "fold_train_range": [
                min(row["train_samples"] for row in folds),
                max(row["train_samples"] for row in folds),
            ],
            "fold_test_range": [
                min(row["test_samples"] for row in folds),
                max(row["test_samples"] for row in folds),
            ],
            "models": list(models),
            "ridge_alpha": RIDGE_ALPHA,
            "huber_alpha": HUBER_ALPHA,
            "huber_epsilon": HUBER_EPSILON,
            "logistic_c": LOGISTIC_C,
            "action_cap_m": ACTION_CAP_M,
            "gt_feature_leakage": False,
            "cases": case_keys,
        },
        "label_stats": finite_stats(labels),
        "feature_correlations": correlations,
        "methods": method_results,
        "feature_increment_over_train_mean_m": float(feature_increment),
        "best_feature_model": best_feature,
        "logistic_gate_selection_rule": {
            "thresholds": LOGISTIC_GATE_THRESHOLDS,
            "requirements": {
                "coverage_min": 0.15,
                "mean_root_gain_min_m": 0.020,
                "relative_root_gain_min": 0.05,
                "harm_over_5cm_max": 0.05,
                "accepted_sign_min": 0.85,
                "accepted_improve_min": 0.80,
                "root_p95_noninferior": True,
                "every_actor_timestamp_root_mean_noninferior": True,
                "full_geometry_pending": "joint/vertex/layout checks required before promotion",
            },
            "selection": "lowest OOF root mean among eligible thresholds",
        },
        "logistic_gate_grid": logistic_grid,
        "selected_logistic_gate": selected_gate,
        "folds": folds,
        "per_actor": breakdown(actors, rows, actions, scores),
        "per_timestamp": breakdown(timestamps, rows, actions, scores),
        "camera_pair_proxy_audit": pair_audit,
        "interpretation": interpretation,
        "limitations": [
            "One MultiHuman capture and `three` development only.",
            "Confidence threshold selection uses `three` OOF labels and is not holdout evidence.",
            "No final model is saved and no dance/box cache is loaded.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "v14_person_residual_head_invariant_cv.json"
    md_path = args.output_dir / "v14_person_residual_head_invariant_cv.md"
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
