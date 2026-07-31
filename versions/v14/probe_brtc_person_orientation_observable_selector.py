#!/usr/bin/env python3
"""Conservative observable selector between frozen torso4 and torso8 Kabsch.

The qualified orientation baseline uses four predicted torso joints.  This
experiment asks whether a slightly larger eight-joint fit can be selected
without GT at runtime.  GT is used only to form development regression labels
and to score predictions.  Selection is leave-one-timestamp-out on ``three
offset0``; the fitted tree and its OOD statistics are serialized before any
held-out split is read.

The default action is the already-qualified torso4 Kabsch.  A person switches
to torso8 only if a shallow regression tree predicts a sufficiently large
joint+vertex gain and every observable feature is within a frozen diagonal
z-score envelope.  BRTC roots, cameras, rejected/unmatched people and pair-root
layout are unchanged by construction.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from sklearn.tree import DecisionTreeRegressor


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (
    REPO_ROOT,
    REPO_ROOT / "src",
    REPO_ROOT / "scripts",
    REPO_ROOT / "versions/v14",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from versions.v14 import probe_b0_brtc_huber_irls as harness  # noqa: E402
from versions.v14 import probe_b0_two_view_person_triangulation as legacy  # noqa: E402
from versions.v14 import probe_brtc_global_orientation_kabsch as orientation  # noqa: E402
from versions.v14.b0_person_body_scale_consistency import (  # noqa: E402
    robust_body_scale_evidence,
)


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "output/v14/fine_alignment_research/brtc_person_orientation_selector"
)
DEFAULT_POLICY = DEFAULT_OUTPUT / "FROZEN_POLICY_BEFORE_HELDOUT.json"
TORSO4 = (1, 2, 16, 17)
TORSO8 = (1, 2, 3, 6, 9, 12, 16, 17)
ALL_BODY = tuple(range(22))
CURRENT_POLICY = orientation.OrientationPolicy(25.0, 0.5, 0.0)
METRICS = (
    "joint_error_m",
    "vertex_error_m",
    "pelvis_joint_error_m",
    "pelvis_vertex_error_m",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("dev", "freeze", "validate"))
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    return parser.parse_args()


def jsonable(value: Any) -> Any:
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def centered_error(
    predicted: dict[str, np.ndarray], target: dict[str, np.ndarray], key: str
) -> float:
    count = min(len(predicted[key]), len(target[key]))
    return float(
        np.linalg.norm(
            (predicted[key][:count] - predicted["root"])
            - (target[key][:count] - target["root"]),
            axis=1,
        ).mean()
    )


def person_metrics(
    predicted: dict[str, np.ndarray], target: dict[str, np.ndarray]
) -> dict[str, float]:
    output = harness.point_errors(predicted, target, full=True)
    output.update(
        {
            "pelvis_joint_error_m": centered_error(predicted, target, "joints"),
            "pelvis_vertex_error_m": centered_error(predicted, target, "vertices"),
        }
    )
    return output


def kabsch_for_ids(
    person: dict[str, Any], ids: tuple[int, ...]
) -> tuple[np.ndarray, float, bool]:
    pre_joints = np.asarray(person["pre"]["joints"], dtype=np.float64)
    post_joints = np.asarray(person["post"]["joints"], dtype=np.float64)
    valid = [index for index in ids if index < min(len(pre_joints), len(post_joints))]
    if len(valid) < 3:
        return np.eye(3, dtype=np.float64), 0.0, False
    pre_root = np.asarray(person["pre"]["root"], dtype=np.float64)
    post_root = np.asarray(person["post"]["root"], dtype=np.float64)
    pre = pre_joints[valid] - pre_root
    post = post_joints[valid] - post_root
    before = float(np.linalg.norm(post - pre, axis=1).mean())
    raw = orientation.kabsch_rotation(post, pre)
    rotation, angle_deg = orientation.bounded_rotation(raw, CURRENT_POLICY)
    after = float(np.linalg.norm(post @ rotation.T - pre, axis=1).mean())
    applied = bool(angle_deg > 1e-8 and after < before)
    return rotation if applied else np.eye(3, dtype=np.float64), angle_deg, applied


def rotate_about_native_root(
    person: dict[str, np.ndarray], rotation: np.ndarray
) -> dict[str, np.ndarray]:
    output = {key: np.asarray(value, dtype=np.float64).copy() for key, value in person.items()}
    root = output["root"]
    for key in ("joints", "vertices"):
        output[key] = (output[key] - root) @ rotation.T + root
    return output


def residual(
    person: dict[str, Any], rotation: np.ndarray, ids: tuple[int, ...]
) -> float:
    pre_joints = np.asarray(person["pre"]["joints"], dtype=np.float64)
    post_joints = np.asarray(person["post"]["joints"], dtype=np.float64)
    valid = [index for index in ids if index < min(len(pre_joints), len(post_joints))]
    pre_root = np.asarray(person["pre"]["root"], dtype=np.float64)
    post_root = np.asarray(person["post"]["root"], dtype=np.float64)
    return float(
        np.linalg.norm(
            (post_joints[valid] - post_root) @ rotation.T
            - (pre_joints[valid] - pre_root),
            axis=1,
        ).mean()
    )


def observable_features(
    case: dict[str, Any],
    person: dict[str, Any],
    evidence: dict[str, Any],
    action_m: float,
    torso4_rotation: np.ndarray,
    torso4_angle: float,
    torso8_rotation: np.ndarray,
    torso8_angle: float,
) -> dict[str, float]:
    pre_joints = np.asarray(person["pre"]["joints"], dtype=np.float64)
    post_joints = np.asarray(person["post"]["joints"], dtype=np.float64)
    pre_root = np.asarray(person["pre"]["root"], dtype=np.float64)
    post_root = np.asarray(person["post"]["root"], dtype=np.float64)
    relative_camera = np.linalg.inv(np.asarray(case["pre_camera"], dtype=np.float64)) @ np.asarray(
        case["post_camera"], dtype=np.float64
    )
    scale = robust_body_scale_evidence(person["pre"], person["post"])
    output = {
        "r4_before": residual(person, np.eye(3), TORSO4),
        "r4_after4": residual(person, torso4_rotation, TORSO4),
        "r4_after8": residual(person, torso8_rotation, TORSO4),
        "r8_before": residual(person, np.eye(3), TORSO8),
        "r8_after4": residual(person, torso4_rotation, TORSO8),
        "r8_after8": residual(person, torso8_rotation, TORSO8),
        "all_before": residual(person, np.eye(3), ALL_BODY),
        "all_after4": residual(person, torso4_rotation, ALL_BODY),
        "all_after8": residual(person, torso8_rotation, ALL_BODY),
        "angle4_deg": float(torso4_angle),
        "angle8_deg": float(torso8_angle),
        "rotation_difference_rad": float(
            np.linalg.norm(
                Rotation.from_matrix(torso4_rotation @ torso8_rotation.T).as_rotvec()
            )
        ),
        "joint0_native_root_pre_m": float(np.linalg.norm(pre_joints[0] - pre_root)),
        "joint0_native_root_post_m": float(np.linalg.norm(post_joints[0] - post_root)),
        "bone_log_center": float(scale["median_log_pre_over_post"]),
        "bone_log_mad": float(scale["log_ratio_mad"]),
        "brtc_action_abs_m": abs(float(action_m)),
        "brtc_median_gap_m": float(evidence["median_gap"]),
        "brtc_mad_m": float(evidence["mad"]),
        "brtc_median_sine": float(evidence["median_sine"]),
        "camera_relative_rotation_rad": float(
            np.linalg.norm(Rotation.from_matrix(relative_camera[:3, :3]).as_rotvec())
        ),
        "camera_relative_translation_m": float(np.linalg.norm(relative_camera[:3, 3])),
    }
    for prefix in ("r4", "r8", "all"):
        for candidate in ("4", "8"):
            output[f"{prefix}_after{candidate}_relative"] = float(
                output[f"{prefix}_after{candidate}"]
                / max(output[f"{prefix}_before"], 1e-12)
            )
    if not all(np.isfinite(list(output.values()))):
        raise ValueError("Non-finite observable orientation selector feature")
    return output


def prepare_rows(source_rows: list[dict[str, Any]]) -> dict[str, Any]:
    cases = harness.prepare_all(source_rows)
    frozen = harness.legacy_policy()
    records: list[dict[str, Any]] = []
    total_people_by_group: Counter[int] = Counter()
    camera_max_abs_change = 0.0
    root_max_abs_change = 0.0
    fallback_max_abs_change = 0.0
    for case in cases:
        group = int(case["case"]["timestamp"])
        total_people_by_group[group] += len(case["people"])
        proposals = []
        for person in case["people"]:
            shift, accepted, evidence, action = harness.legacy_proposal(person, frozen)
            proposals.append(
                {
                    "individual_shift": shift,
                    "accepted": accepted,
                    "evidence": evidence,
                    "action_m": action,
                }
            )
        shifts, _ = harness.observable_layout_consensus(case, proposals)
        for person_index, (person, proposal, shift) in enumerate(
            zip(case["people"], proposals, shifts)
        ):
            post = {
                key: np.asarray(person["post"][key], dtype=np.float64)
                for key in ("root", "joints", "vertices")
            }
            brtc = {
                key: value + np.asarray(shift, dtype=np.float64)
                for key, value in post.items()
            }
            if not bool(proposal["accepted"]):
                fallback_max_abs_change = max(
                    fallback_max_abs_change,
                    max(float(np.max(np.abs(brtc[key] - post[key]))) for key in brtc),
                )
                continue
            torso4_rotation, torso4_angle, _ = kabsch_for_ids(person, TORSO4)
            torso8_rotation, torso8_angle, _ = kabsch_for_ids(person, TORSO8)
            baseline = rotate_about_native_root(brtc, torso4_rotation)
            alternative = rotate_about_native_root(brtc, torso8_rotation)
            target = {
                key: np.asarray(person["target"][key], dtype=np.float64)
                for key in ("root", "joints", "vertices")
            }
            baseline_metrics = person_metrics(baseline, target)
            alternative_metrics = person_metrics(alternative, target)
            delta = {
                key: alternative_metrics[key] - baseline_metrics[key] for key in METRICS
            }
            features = observable_features(
                case,
                person,
                proposal["evidence"],
                proposal["action_m"],
                torso4_rotation,
                torso4_angle,
                torso8_rotation,
                torso8_angle,
            )
            root_max_abs_change = max(
                root_max_abs_change,
                float(np.max(np.abs(alternative["root"] - baseline["root"]))),
            )
            records.append(
                {
                    "group": group,
                    "case_key_evaluation_only": case["case"]["key"],
                    "person_index_evaluation_only": person_index,
                    "features": features,
                    "delta": delta,
                    "target_score": delta["joint_error_m"] + delta["vertex_error_m"],
                }
            )
        camera_max_abs_change = max(
            camera_max_abs_change, float(case["camera"]["candidate_max_abs_change"])
        )
    return {
        "records": records,
        "total_people_by_group": dict(total_people_by_group),
        "invariants": {
            "camera_max_abs_change": camera_max_abs_change,
            "alternative_vs_torso4_root_max_abs_change": root_max_abs_change,
            "rejected_exact_b0_max_abs_change": fallback_max_abs_change,
            "pair_root_layout_change": 0.0,
        },
    }


def matrix(dataset: dict[str, Any], feature_names: list[str] | None = None):
    records = dataset["records"]
    names = sorted(records[0]["features"]) if feature_names is None else feature_names
    values = np.asarray(
        [[float(row["features"][name]) for name in names] for row in records],
        dtype=np.float64,
    )
    targets = np.asarray([float(row["target_score"]) for row in records])
    groups = np.asarray([int(row["group"]) for row in records], dtype=np.int64)
    return names, values, targets, groups


def fit_tree(
    values: np.ndarray, targets: np.ndarray, max_depth: int, min_samples_leaf: int
) -> DecisionTreeRegressor:
    return DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=0,
    ).fit(values, targets)


def tree_dict(tree: DecisionTreeRegressor) -> dict[str, Any]:
    fitted = tree.tree_
    return {
        "children_left": fitted.children_left.astype(int).tolist(),
        "children_right": fitted.children_right.astype(int).tolist(),
        "feature": fitted.feature.astype(int).tolist(),
        "threshold": fitted.threshold.astype(float).tolist(),
        "value": fitted.value[:, 0, 0].astype(float).tolist(),
        "node_count": int(fitted.node_count),
        "leaf_count": int(tree.get_n_leaves()),
    }


def predict_tree(tree: dict[str, Any], values: np.ndarray) -> np.ndarray:
    predictions = []
    for row in np.asarray(values, dtype=np.float64):
        node = 0
        while int(tree["children_left"][node]) != int(tree["children_right"][node]):
            feature = int(tree["feature"][node])
            node = int(
                tree["children_left"][node]
                if row[feature] <= float(tree["threshold"][node])
                else tree["children_right"][node]
            )
        predictions.append(float(tree["value"][node]))
    return np.asarray(predictions, dtype=np.float64)


def selection(
    predictions: np.ndarray,
    values: np.ndarray,
    threshold_m: float,
    mean: np.ndarray,
    std: np.ndarray,
    max_abs_z: float,
) -> tuple[np.ndarray, np.ndarray]:
    zmax = np.max(np.abs((values - mean) / np.maximum(std, 1e-9)), axis=1)
    return (predictions < -float(threshold_m)) & (zmax <= max_abs_z), zmax


def summarize(
    dataset: dict[str, Any], selected: np.ndarray, zmax: np.ndarray
) -> dict[str, Any]:
    records = dataset["records"]
    totals = {int(key): int(value) for key, value in dataset["total_people_by_group"].items()}
    groups = sorted(totals)
    per_group = {}
    for group in groups:
        indices = [
            index
            for index, row in enumerate(records)
            if int(row["group"]) == group and bool(selected[index])
        ]
        per_group[str(group)] = {
            "total_people": totals[group],
            "eligible_accepted_people": sum(
                int(row["group"]) == group for row in records
            ),
            "torso8_selected_people": len(indices),
            "delta_candidate_minus_torso4": {
                metric: float(sum(records[index]["delta"][metric] for index in indices) / totals[group])
                for metric in METRICS
            },
        }
    indices = np.flatnonzero(selected).tolist()
    total_people = sum(totals.values())
    delta = {
        metric: float(sum(records[index]["delta"][metric] for index in indices) / total_people)
        for metric in METRICS
    }
    harms = {}
    for threshold_name, threshold in (("over_1cm", 0.01), ("over_5cm", 0.05)):
        harms[threshold_name] = {
            metric: int(
                sum(records[index]["delta"][metric] > threshold for index in indices)
            )
            for metric in METRICS
        }
    group_pass = {
        group: bool(
            all(value <= 1e-12 for value in row["delta_candidate_minus_torso4"].values())
        )
        for group, row in per_group.items()
    }
    return {
        "total_people": total_people,
        "eligible_accepted_people": len(records),
        "torso8_selected_people": len(indices),
        "selection_rate_over_eligible": float(len(indices) / max(len(records), 1)),
        "delta_candidate_minus_torso4": delta,
        "selected_person_harm_counts": harms,
        "per_timestamp": per_group,
        "timestamp_all_metric_pass": group_pass,
        "all_timestamps_pass": bool(all(group_pass.values())),
        "all_aggregate_metrics_pass": bool(all(value <= 1e-12 for value in delta.values())),
        "all_harm_counts_zero": bool(
            all(count == 0 for level in harms.values() for count in level.values())
        ),
        "max_observed_abs_z": float(np.max(zmax)) if len(zmax) else 0.0,
        "invariants": dataset["invariants"],
    }


def eligible(summary: dict[str, Any], require_each_timestamp: bool) -> bool:
    invariants = summary["invariants"]
    return bool(
        summary["torso8_selected_people"] > 0
        and summary["all_aggregate_metrics_pass"]
        and summary["all_harm_counts_zero"]
        and (summary["all_timestamps_pass"] or not require_each_timestamp)
        and float(invariants["camera_max_abs_change"]) <= 1e-12
        and float(invariants["alternative_vs_torso4_root_max_abs_change"]) <= 1e-12
        and float(invariants["rejected_exact_b0_max_abs_change"]) <= 1e-12
        and float(invariants["pair_root_layout_change"]) <= 1e-12
    )


def cross_validate(
    dataset: dict[str, Any], max_depth: int, min_samples_leaf: int, threshold_m: float
) -> dict[str, Any]:
    names, values, targets, groups = matrix(dataset)
    predictions = np.zeros(len(values), dtype=np.float64)
    zmax = np.zeros(len(values), dtype=np.float64)
    selected = np.zeros(len(values), dtype=bool)
    for heldout in sorted(set(groups.tolist())):
        train = groups != heldout
        test = ~train
        model = fit_tree(values[train], targets[train], max_depth, min_samples_leaf)
        mean = values[train].mean(axis=0)
        std = values[train].std(axis=0)
        predictions[test] = model.predict(values[test])
        selected[test], zmax[test] = selection(
            predictions[test], values[test], threshold_m, mean, std, 5.0
        )
    summary = summarize(dataset, selected, zmax)
    return {
        "config": {
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "predicted_gain_threshold_m": threshold_m,
            "ood_max_abs_z": 5.0,
        },
        "summary": summary,
        "eligible": eligible(summary, require_each_timestamp=True),
        "feature_names": names,
    }


def heldout_rows() -> dict[str, list[dict[str, Any]]]:
    return {
        "three_offset1": json.loads(
            harness.DEFAULT_CONFIRM_REPORT.read_text(encoding="utf-8")
        )["cases"],
        "dance": legacy.report_rows(("dance",)),
        "box": legacy.report_rows(("box",)),
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Observable torso4/torso8 orientation selector",
        "",
        f"Phase: `{report['phase']}`; status: `{report['status']}`.",
        "",
    ]
    if report["phase"] == "development":
        selected = report.get("selected")
        if selected:
            lines.extend(
                [
                    f"Eligible CV configs: `{report['eligible_count']}`; selected: `{selected['config']}`.",
                    "",
                    "| Timestamp | Selected | dJoint mm | dVertex mm | dCentered-joint mm | dCentered-vertex mm | Pass |",
                    "|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for group, row in selected["summary"]["per_timestamp"].items():
                delta = row["delta_candidate_minus_torso4"]
                lines.append(
                    f"| {group} | {row['torso8_selected_people']} | "
                    f"{1000*delta['joint_error_m']:.6f} | {1000*delta['vertex_error_m']:.6f} | "
                    f"{1000*delta['pelvis_joint_error_m']:.6f} | "
                    f"{1000*delta['pelvis_vertex_error_m']:.6f} | "
                    f"{selected['summary']['timestamp_all_metric_pass'][group]} |"
                )
    else:
        lines.extend(
            [
                "| Split | Selected | dJoint mm | dVertex mm | dCentered-joint mm | dCentered-vertex mm | Pass |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for split, value in report.get("splits", {}).items():
            delta = value["summary"]["delta_candidate_minus_torso4"]
            lines.append(
                f"| {split} | {value['summary']['torso8_selected_people']} | "
                f"{1000*delta['joint_error_m']:.6f} | {1000*delta['vertex_error_m']:.6f} | "
                f"{1000*delta['pelvis_joint_error_m']:.6f} | "
                f"{1000*delta['pelvis_vertex_error_m']:.6f} | {value['pass']} |"
            )
    return "\n".join(lines) + "\n"


def run_dev(args: argparse.Namespace) -> None:
    dataset = prepare_rows(legacy.report_rows(("three",)))
    scan = []
    for depth in (2, 3, 4):
        for leaf in (3, 5, 8):
            # Threshold is an expected combined joint+vertex improvement.
            for threshold_mm in np.linspace(0.0, 4.0, 41):
                scan.append(
                    cross_validate(dataset, depth, leaf, float(threshold_mm / 1000.0))
                )
    qualified = [row for row in scan if row["eligible"]]
    selected = (
        min(
            qualified,
            key=lambda row: (
                sum(row["summary"]["delta_candidate_minus_torso4"].values()),
                -row["summary"]["torso8_selected_people"],
                row["config"]["max_depth"],
                row["config"]["min_samples_leaf"],
                row["config"]["predicted_gain_threshold_m"],
            ),
        )
        if qualified
        else None
    )
    report = {
        "experiment": "v14_brtc_person_orientation_observable_selector",
        "phase": "development",
        "status": "DEV_PASS" if selected else "DEV_NO_GO",
        "protocol": {
            "data": "three offset0 only",
            "grouped_cv": "leave one timestamp out; groups 500/700/900/1000/1100/1300/1500",
            "runtime_inputs": "predicted pre/post joints, frozen B0 camera, frozen BRTC evidence only",
            "gt_use": "per-person training target and offline metrics only",
            "default": "exact current qualified torso4 Kabsch",
            "alternative": "same bounded policy with torso8 correspondences",
            "hard_gate": "every held-out timestamp raw+centered joint/vertex <= torso4; >1cm/>5cm harm zero",
        },
        "eligible_count": len(qualified),
        "selected": selected,
        "scan": scan,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "GROUPED_CV_DEV.json"
    path.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False) + "\n")
    (args.output_dir / "GROUPED_CV_DEV.md").write_text(markdown(report), encoding="utf-8")
    print(markdown(report), end="")


def run_freeze(args: argparse.Namespace) -> None:
    dev_path = args.output_dir / "GROUPED_CV_DEV.json"
    dev = json.loads(dev_path.read_text(encoding="utf-8"))
    if dev["selected"] is None:
        raise RuntimeError("No grouped-CV eligible observable selector")
    dataset = prepare_rows(legacy.report_rows(("three",)))
    names, values, targets, _ = matrix(dataset, dev["selected"]["feature_names"])
    config = dev["selected"]["config"]
    model = fit_tree(
        values, targets, int(config["max_depth"]), int(config["min_samples_leaf"])
    )
    policy = {
        "feature_names": names,
        "tree": tree_dict(model),
        "predicted_gain_threshold_m": float(config["predicted_gain_threshold_m"]),
        "ood_max_abs_z": float(config["ood_max_abs_z"]),
        "feature_mean": values.mean(axis=0),
        "feature_std": values.std(axis=0),
        "orientation_policy": asdict(CURRENT_POLICY),
        "baseline_joint_ids": TORSO4,
        "alternative_joint_ids": TORSO8,
    }
    frozen = {
        "experiment": "v14_brtc_person_orientation_observable_selector",
        "phase": "frozen_before_heldout",
        "frozen": True,
        "development_file": str(dev_path),
        "development_file_sha256": file_sha256(dev_path),
        "policy": policy,
        "policy_sha256": canonical_sha256(policy),
        "constraints": {
            "future_post_frames": 0,
            "extra_pretrained_models": [],
            "gt_runtime": False,
            "camera_update": "none",
            "root_update": "none beyond frozen BRTC",
            "rejected_unmatched": "exact current output",
        },
    }
    args.policy.write_text(json.dumps(jsonable(frozen), indent=2, ensure_ascii=False) + "\n")
    print(args.policy)


def run_validate(args: argparse.Namespace) -> None:
    frozen = json.loads(args.policy.read_text(encoding="utf-8"))
    if canonical_sha256(frozen["policy"]) != frozen["policy_sha256"]:
        raise ValueError("Frozen observable-selector policy checksum mismatch")
    policy = frozen["policy"]
    splits = {}
    for name, rows in heldout_rows().items():
        dataset = prepare_rows(rows)
        _, values, _, _ = matrix(dataset, list(policy["feature_names"]))
        predictions = predict_tree(policy["tree"], values)
        selected, zmax = selection(
            predictions,
            values,
            float(policy["predicted_gain_threshold_m"]),
            np.asarray(policy["feature_mean"], dtype=np.float64),
            np.asarray(policy["feature_std"], dtype=np.float64),
            float(policy["ood_max_abs_z"]),
        )
        summary = summarize(dataset, selected, zmax)
        splits[name] = {
            "summary": summary,
            "pass": eligible(summary, require_each_timestamp=False),
        }
    passed = bool(all(row["pass"] for row in splits.values()))
    report = {
        "experiment": "v14_brtc_person_orientation_observable_selector",
        "phase": "heldout_validation",
        "status": "MULTIHUMAN_PASS" if passed else "NO_GO_MULTIHUMAN",
        "policy_file": str(args.policy),
        "policy_file_sha256": file_sha256(args.policy),
        "splits": splits,
        "all_splits_pass": passed,
        "note": "EgoHumans must not be opened unless this frozen MultiHuman gate passes.",
    }
    path = args.output_dir / "HELDOUT_RESULTS.json"
    path.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False) + "\n")
    (args.output_dir / "HELDOUT_RESULTS.md").write_text(markdown(report), encoding="utf-8")
    print(markdown(report), end="")


def main() -> None:
    args = parse_args()
    if not str(args.output_dir.resolve()).startswith(str(REPO_ROOT.resolve())):
        raise ValueError("Output must stay under Movie3R on /data")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    original_load = torch.load

    @lru_cache(maxsize=None)
    def cached_load(path: str):
        return original_load(path, map_location="cpu", weights_only=False)

    torch.load = lambda path, *unused_args, **unused_kwargs: cached_load(str(path))
    try:
        if args.phase == "dev":
            run_dev(args)
        elif args.phase == "freeze":
            run_freeze(args)
        else:
            run_validate(args)
    finally:
        torch.load = original_load


if __name__ == "__main__":
    main()
