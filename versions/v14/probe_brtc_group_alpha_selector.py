#!/usr/bin/env python3
"""Train, freeze, and blindly validate a causal BRTC group-alpha selector.

Only ``three`` offset-0 is permitted during development.  Cross-validation
holds out complete timestamps, then a multinomial linear classifier is frozen
to JSON before offset-1, dance, box, or EgoHumans is opened.  The deployment
runtime can choose only alpha 0.8, 0.9, or 1.0 for the shared BRTC group shift;
the person-specific residual, camera, unmatched people, and rejected people
remain unchanged.  OOD or low-confidence input is exact BRTC v1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression


REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    REPO_ROOT,
    REPO_ROOT / "src",
    REPO_ROOT / "scripts",
    REPO_ROOT / "versions/v14",
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from versions.v14 import probe_b0_brtc_huber_irls as harness  # noqa: E402
from versions.v14 import probe_b0_two_view_person_triangulation as legacy  # noqa: E402
from versions.v14.b0_person_triangulation import refine_matched_people  # noqa: E402
from versions.v14.b0_person_triangulation_group_alpha_selector import (  # noqa: E402
    ALPHAS,
    FEATURE_NAMES,
    feature_vector,
    observable_feature_dict,
    refine_matched_people_group_alpha_selector,
    select_group_alpha,
)


DEFAULT_OUTPUT = (
    REPO_ROOT / "output/v14/fine_alignment_research/brtc_group_alpha_selector"
)
DEFAULT_POLICY = DEFAULT_OUTPUT / "FROZEN_POLICY_BEFORE_HELDOUT.json"
DEFAULT_EGO_CACHE = (
    REPO_ROOT
    / "output/v14/fine_alignment_research/brtc_multithumbs_egohumans/"
    "current_v14_cpu_geometry.pt"
)
ERROR_METRICS = (
    "root_error_m",
    "joint_error_m",
    "vertex_error_m",
    "pairwise_distance_error_m",
    "pairwise_vector_error_m",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("dev", "freeze", "validate"), required=True)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument(
        "--confirm_report", type=Path, default=harness.DEFAULT_CONFIRM_REPORT
    )
    parser.add_argument("--ego_geometry_cache", type=Path, default=DEFAULT_EGO_CACHE)
    parser.add_argument("--skip_ego", action="store_true")
    parser.add_argument("--max_cases", type=int, default=0)
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if len(array) else float("nan")


def dataset_fingerprint(records: list[dict[str, Any]]) -> str:
    rows = [
        {
            "key": row["case"]["case"]["key"],
            "timestamp": int(row["timestamp"]),
            "oracle_alpha": float(row["oracle_alpha"]),
            "features": feature_vector(row["features"]).round(12).tolist(),
        }
        for row in records
    ]
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def alpha_shifts(debug: dict[str, Any], alpha: float) -> list[np.ndarray]:
    group = np.asarray(debug["group_shift_world"], dtype=np.float64)
    residual_lambda = float(debug["selected_residual_lambda"])
    output = []
    for row in debug["people"]:
        if bool(row["accepted"]):
            individual = np.asarray(row["individual_shift_world"], dtype=np.float64)
            output.append(alpha * group + residual_lambda * (individual - group))
        else:
            output.append(np.zeros(3, dtype=np.float64))
    return output


def prepare_records(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for index, case in enumerate(cases, start=1):
        pre_people = [person["pre"] for person in case["people"]]
        post_people = [person["post"] for person in case["people"]]
        matches = [(person_index, person_index) for person_index in range(len(pre_people))]
        _, debug = refine_matched_people(
            case["pre_camera"], case["post_camera"], pre_people, post_people, matches
        )
        features = observable_feature_dict(debug, len(pre_people), len(post_people))
        shifts = {alpha: alpha_shifts(debug, alpha) for alpha in ALPHAS}
        root_by_alpha = {}
        for alpha in ALPHAS:
            errors = [
                float(
                    np.linalg.norm(
                        person["post"]["root"]
                        + shift
                        - person["target"]["root"]
                    )
                )
                for person, shift in zip(case["people"], shifts[alpha])
            ]
            root_by_alpha[alpha] = float(np.mean(errors))
        # Minimize case mean root; exact-v1 (largest alpha) wins every tie.
        oracle = min(ALPHAS, key=lambda alpha: (root_by_alpha[alpha], -alpha))
        records.append(
            {
                "case": case,
                "timestamp": int(case["case"]["timestamp"]),
                "features": features,
                "base_debug": debug,
                "shifts": shifts,
                "root_by_alpha": root_by_alpha,
                "oracle_alpha": float(oracle),
            }
        )
        print(
            f"[{index:03d}/{len(cases):03d}] features {case['case']['key']} "
            f"oracle={oracle:.1f}",
            flush=True,
        )
    return records


def feature_support(
    matrix: np.ndarray, margin: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    lower = matrix.min(axis=0) - float(margin) * scale
    upper = matrix.max(axis=0) + float(margin) * scale
    return mean, scale, lower, upper


def fit_policy(
    records: list[dict[str, Any]], spec: dict[str, Any]
) -> dict[str, Any]:
    matrix = np.stack([feature_vector(row["features"]) for row in records])
    alpha_to_class = {alpha: index for index, alpha in enumerate(ALPHAS)}
    labels = np.asarray(
        [alpha_to_class[float(row["oracle_alpha"])] for row in records],
        dtype=np.int64,
    )
    mean, scale, lower, upper = feature_support(matrix, float(spec["ood_margin_std"]))
    classifier = LogisticRegression(
        C=float(spec["C"]),
        class_weight=spec["class_weight"],
        solver="lbfgs",
        max_iter=2000,
        random_state=20260801,
    )
    classifier.fit((matrix - mean) / scale, labels)
    integer_classes = np.asarray(classifier.classes_, dtype=np.int64)
    classes = np.asarray([ALPHAS[index] for index in integer_classes], dtype=np.float64)
    coefficients = np.asarray(classifier.coef_, dtype=np.float64)
    intercept = np.asarray(classifier.intercept_, dtype=np.float64)
    # The frozen runtime deliberately supports only a true three-logit model.
    if tuple(classes.tolist()) != ALPHAS or coefficients.shape != (
        len(ALPHAS),
        len(FEATURE_NAMES),
    ):
        raise RuntimeError(
            f"Expected three-class alpha model, got {classes} {coefficients.shape}"
        )
    return {
        "confidence_threshold": float(spec["confidence_threshold"]),
        "model": {
            "type": "multinomial_logistic_regression_frozen_numpy",
            "feature_names": list(FEATURE_NAMES),
            "feature_mean": mean,
            "feature_scale": scale,
            "feature_lower": lower,
            "feature_upper": upper,
            "classes": classes,
            "coefficients": coefficients,
            "intercept": intercept,
        },
    }


def corrected_person(
    person: dict[str, Any], shift: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(person["post"][key], dtype=np.float64) + shift
        for key in ("root", "joints", "vertices")
    }


def evaluate_decisions(
    records: list[dict[str, Any]], decisions: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(records) != len(decisions):
        raise ValueError("Decision count mismatch")
    metrics = {key: [] for key in ERROR_METRICS}
    baseline_metrics = {key: [] for key in ERROR_METRICS}
    root_deltas, root_delta_vs_v1 = [], []
    decisions_by_alpha: Counter[float] = Counter()
    fallback_reasons: Counter[str] = Counter()
    ood_features: Counter[str] = Counter()
    case_rows = []
    for record, decision in zip(records, decisions):
        case = record["case"]
        alpha = float(decision["selected_alpha"])
        decisions_by_alpha[alpha] += 1
        if bool(decision.get("fallback", False)):
            fallback_reasons[str(decision.get("fallback_reason"))] += 1
        ood_features.update(decision.get("outside_feature_names", ()))
        shifts = record["shifts"][alpha]
        v1_shifts = record["shifts"][1.0]
        predicted_roots, v1_roots, target_roots = [], [], []
        person_rows = []
        for person, shift, v1_shift in zip(case["people"], shifts, v1_shifts):
            predicted = corrected_person(person, shift)
            v1_predicted = corrected_person(person, v1_shift)
            target = person["target"]
            row_metrics = harness.point_errors(predicted, target, full=True)
            v1_metrics = harness.point_errors(v1_predicted, target, full=True)
            for key in ("root_error_m", "joint_error_m", "vertex_error_m"):
                metrics[key].append(float(row_metrics[key]))
                baseline_metrics[key].append(float(v1_metrics[key]))
            b0_root = float(
                np.linalg.norm(person["post"]["root"] - target["root"])
            )
            root_deltas.append(float(row_metrics["root_error_m"] - b0_root))
            root_delta_vs_v1.append(
                float(row_metrics["root_error_m"] - v1_metrics["root_error_m"])
            )
            predicted_roots.append(predicted["root"])
            v1_roots.append(v1_predicted["root"])
            target_roots.append(target["root"])
            person_rows.append(
                {
                    "identity_evaluation_only": person["identity"],
                    "candidate": row_metrics,
                    "v1": v1_metrics,
                    "b0_root_error_m": b0_root,
                }
            )

        def layout(roots: list[np.ndarray]) -> dict[str, float]:
            distance, vector = [], []
            for first in range(len(roots)):
                for second in range(first + 1, len(roots)):
                    pred = roots[first] - roots[second]
                    target = target_roots[first] - target_roots[second]
                    distance.append(
                        abs(float(np.linalg.norm(pred) - np.linalg.norm(target)))
                    )
                    vector.append(float(np.linalg.norm(pred - target)))
            return {
                "pairwise_distance_error_m": finite_mean(distance),
                "pairwise_vector_error_m": finite_mean(vector),
            }

        candidate_layout = layout(predicted_roots)
        v1_layout = layout(v1_roots)
        for key in ("pairwise_distance_error_m", "pairwise_vector_error_m"):
            metrics[key].append(candidate_layout[key])
            baseline_metrics[key].append(v1_layout[key])
        case_rows.append(
            {
                "case": case["case"],
                "decision": decision,
                "oracle_alpha_evaluation_only": record["oracle_alpha"],
                "metrics": candidate_layout,
                "v1_metrics": v1_layout,
                "people": person_rows,
            }
        )
    mean = {key: finite_mean(value) for key, value in metrics.items()}
    baseline = {
        key: finite_mean(value) for key, value in baseline_metrics.items()
    }
    deltas = {key: float(mean[key] - baseline[key]) for key in ERROR_METRICS}
    return {
        "case_count": len(records),
        "person_count": sum(len(row["case"]["people"]) for row in records),
        "candidate": mean,
        "v1": baseline,
        "delta_vs_v1": deltas,
        "root_harm_over_1cm_rate": float(np.mean(np.asarray(root_deltas) > 0.01)),
        "root_harm_over_5cm_rate": float(np.mean(np.asarray(root_deltas) > 0.05)),
        "v1_root_harm_over_1cm_rate": float(
            np.mean(
                np.asarray(root_deltas, dtype=np.float64)
                - np.asarray(root_delta_vs_v1, dtype=np.float64)
                > 0.01
            )
        ),
        "v1_root_harm_over_5cm_rate": float(
            np.mean(
                np.asarray(root_deltas, dtype=np.float64)
                - np.asarray(root_delta_vs_v1, dtype=np.float64)
                > 0.05
            )
        ),
        "root_improve_vs_v1_rate": float(np.mean(np.asarray(root_delta_vs_v1) < 0.0)),
        "root_harm_vs_v1_rate": float(np.mean(np.asarray(root_delta_vs_v1) > 0.0)),
        "selected_alpha_counts": {
            str(alpha): int(decisions_by_alpha[alpha]) for alpha in ALPHAS
        },
        "non_v1_action_count": int(sum(decisions_by_alpha[a] for a in ALPHAS if a < 1.0)),
        "fallback_count": int(sum(fallback_reasons.values())),
        "fallback_reasons": dict(sorted(fallback_reasons.items())),
        "ood_feature_counts": dict(sorted(ood_features.items())),
        "cases": case_rows,
    }


def exact_v1_decisions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "selected_alpha": 1.0,
            "fallback": True,
            "fallback_reason": "exact_v1_reference",
            "outside_feature_names": [],
            "confidence": 1.0,
            "probability_by_alpha": {"1.0": 1.0},
        }
        for _ in records
    ]


def eligible(summary: dict[str, Any]) -> bool:
    tolerance = 1e-12
    return bool(
        summary["non_v1_action_count"] > 0
        and all(summary["delta_vs_v1"][key] <= tolerance for key in ERROR_METRICS)
        and summary["root_harm_over_1cm_rate"]
        <= summary["v1_root_harm_over_1cm_rate"] + tolerance
        and summary["root_harm_over_5cm_rate"]
        <= summary["v1_root_harm_over_5cm_rate"] + tolerance
    )


def cross_validate(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups = np.asarray([row["timestamp"] for row in records], dtype=np.int64)
    timestamps = sorted(set(groups.tolist()))
    folds = []
    for timestamp in timestamps:
        train = [index for index, value in enumerate(groups) if value != timestamp]
        test = [index for index, value in enumerate(groups) if value == timestamp]
        folds.append((timestamp, train, test))
    specs = [
        {
            "C": C,
            "class_weight": class_weight,
            "confidence_threshold": threshold,
            "ood_margin_std": margin,
        }
        for C in (0.01, 0.1, 1.0, 10.0)
        for class_weight in (None, "balanced")
        for threshold in (0.4, 0.5, 0.6, 0.7)
        for margin in (0.0, 0.25, 0.5)
    ]
    scan = []
    for spec_index, spec in enumerate(specs, start=1):
        decisions: list[dict[str, Any] | None] = [None] * len(records)
        fold_rows = []
        for timestamp, train_indices, test_indices in folds:
            policy = fit_policy([records[index] for index in train_indices], spec)
            fold_decisions = [
                select_group_alpha(records[index]["features"], policy)
                for index in test_indices
            ]
            for index, decision in zip(test_indices, fold_decisions):
                decisions[index] = decision
            fold_rows.append(
                {
                    "held_out_timestamp": timestamp,
                    "train_count": len(train_indices),
                    "validation_count": len(test_indices),
                    "selected_alpha_counts": dict(
                        Counter(
                            str(float(row["selected_alpha"]))
                            for row in fold_decisions
                        )
                    ),
                    "fallback_count": sum(bool(row["fallback"]) for row in fold_decisions),
                }
            )
        if any(row is None for row in decisions):
            raise RuntimeError("Incomplete grouped CV predictions")
        summary = evaluate_decisions(records, list(decisions))
        scan.append(
            {
                "spec": spec,
                "eligible": eligible(summary),
                "summary": {key: value for key, value in summary.items() if key != "cases"},
                "folds": fold_rows,
            }
        )
        if spec_index % 12 == 0 or spec_index == len(specs):
            print(f"[{spec_index:03d}/{len(specs):03d}] grouped CV", flush=True)
    eligible_rows = [row for row in scan if row["eligible"]]
    ranking_pool = eligible_rows or scan
    selected = min(
        ranking_pool,
        key=lambda row: (
            row["summary"]["candidate"]["root_error_m"],
            row["summary"]["root_harm_over_5cm_rate"],
            row["summary"]["candidate"]["joint_error_m"],
            row["summary"]["candidate"]["vertex_error_m"],
            row["summary"]["candidate"]["pairwise_vector_error_m"],
            -row["summary"]["non_v1_action_count"],
            json.dumps(row["spec"], sort_keys=True),
        ),
    )
    return {
        "split": "three offset0 only",
        "fold_definition": "leave one complete timestamp out (7 folds)",
        "timestamps": timestamps,
        "candidate_count": len(scan),
        "eligible_count": len(eligible_rows),
        "selection_pool": "eligible_only" if eligible_rows else "diagnostic_all_no_go",
        "selection_rule": (
            "all root/joint/vertex/pair means and harm >1cm/>5cm must not worsen "
            "v1 and at least one non-v1 action; then root, harm5, joint, vertex, "
            "pair-vector, negative action count, deterministic spec JSON"
        ),
        "selected": selected,
        "scan": scan,
    }


def oracle_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(str(row["oracle_alpha"]) for row in records)
    gains = [
        float(row["root_by_alpha"][1.0] - row["root_by_alpha"][row["oracle_alpha"]])
        for row in records
    ]
    return {
        "label_counts": dict(sorted(labels.items())),
        "mean_case_root_gain_vs_v1_m": float(np.mean(gains)),
        "improvable_case_count": int(sum(value > 1e-12 for value in gains)),
    }


def markdown_split(title: str, split: dict[str, Any]) -> list[str]:
    value = split["summary"]
    candidate, v1 = value["candidate"], value["v1"]
    return [
        f"## {title}",
        "",
        "| Method | Root | Joint | Vertex | Pair distance | Pair vector | Harm >1cm | Harm >5cm | Actions | Fallback |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| BRTC v1 | {v1['root_error_m']:.6f} | {v1['joint_error_m']:.6f} | "
        f"{v1['vertex_error_m']:.6f} | {v1['pairwise_distance_error_m']:.6f} | "
        f"{v1['pairwise_vector_error_m']:.6f} | {value['v1_root_harm_over_1cm_rate']:.1%} | "
        f"{value['v1_root_harm_over_5cm_rate']:.1%} | 0 | 0 |",
        f"| selector | {candidate['root_error_m']:.6f} | {candidate['joint_error_m']:.6f} | "
        f"{candidate['vertex_error_m']:.6f} | {candidate['pairwise_distance_error_m']:.6f} | "
        f"{candidate['pairwise_vector_error_m']:.6f} | {value['root_harm_over_1cm_rate']:.1%} | "
        f"{value['root_harm_over_5cm_rate']:.1%} | {value['non_v1_action_count']} | "
        f"{value['fallback_count']} |",
        "",
        f"- α counts: `{value['selected_alpha_counts']}`; fallback: `{value['fallback_reasons']}`.",
        f"- Post-shot root Accel: `{split.get('acceleration', {}).get('root_accel_delta2_mm_per_frame2', 'unavailable')}`.",
        "",
    ]


def dev_markdown(report: dict[str, Any]) -> str:
    selected = report["cv"]["selected"]
    value = selected["summary"]
    lines = [
        "# BRTC shared-group alpha selector: development CV",
        "",
        "Only `three offset0` was read. Each fold holds out one complete timestamp.",
        "",
        f"- Eligible policies: `{report['cv']['eligible_count']}/{report['cv']['candidate_count']}`.",
        f"- Decision: **{'GO_TO_FREEZE' if report['dev_pass'] else 'NO_GO'}**.",
        f"- Selected spec: `{selected['spec']}`.",
        f"- Oracle labels: `{report['oracle']['label_counts']}`.",
        "",
    ]
    lines.extend(markdown_split("Grouped out-of-fold result", {"summary": value}))
    return "\n".join(lines) + "\n"


def acceleration_summary(records: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    """Post-shot k=0/1/2 delta2; a constant boundary shift must cancel exactly."""
    groups: dict[tuple[Any, ...], dict[int, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        case = record["case"]["case"]
        key = (
            record["case"]["sequence"],
            int(case["timestamp"]),
            int(case["source_camera"]),
            int(case["target_camera"]),
        )
        groups[key][int(case["offset"])] = record
    errors = {"root": [], "joint": [], "vertex": []}
    v1_errors = {"root": [], "joint": [], "vertex": []}
    used = 0
    for rows in groups.values():
        if not all(offset in rows for offset in (0, 1, 2)):
            continue
        boundary = rows[0]
        decision = select_group_alpha(boundary["features"], policy)
        alpha = float(decision["selected_alpha"])
        identities = [person["identity"] for person in boundary["case"]["people"]]
        for identity_index, identity in enumerate(identities):
            series, v1_series, targets = defaultdict(list), defaultdict(list), defaultdict(list)
            valid = True
            for offset in (0, 1, 2):
                row = rows[offset]
                lookup = {
                    person["identity"]: index
                    for index, person in enumerate(row["case"]["people"])
                }
                if identity not in lookup:
                    valid = False
                    break
                person = row["case"]["people"][lookup[identity]]
                # Deployment propagates the shift selected at the boundary.
                selector_shift = boundary["shifts"][alpha][identity_index]
                v1_shift = boundary["shifts"][1.0][identity_index]
                for key in ("root", "joints", "vertices"):
                    series[key].append(np.asarray(person["post"][key]) + selector_shift)
                    v1_series[key].append(np.asarray(person["post"][key]) + v1_shift)
                    targets[key].append(np.asarray(person["target"][key]))
            if not valid:
                continue
            used += 1
            for key in ("root", "joints", "vertices"):
                metric_key = {
                    "root": "root",
                    "joints": "joint",
                    "vertices": "vertex",
                }[key]
                for source, destination in (
                    (series, errors[metric_key]),
                    (v1_series, v1_errors[metric_key]),
                ):
                    source_rows = source[key]
                    target_rows = targets[key]
                    if np.asarray(source_rows[0]).ndim > 1:
                        count = min(
                            *(len(value) for value in source_rows),
                            *(len(value) for value in target_rows),
                        )
                        source_rows = [value[:count] for value in source_rows]
                        target_rows = [value[:count] for value in target_rows]
                    target_delta2 = (
                        target_rows[2] - 2 * target_rows[1] + target_rows[0]
                    )
                    delta2 = source_rows[2] - 2 * source_rows[1] + source_rows[0]
                    norm = np.linalg.norm(delta2 - target_delta2, axis=-1)
                    destination.append(float(np.mean(norm)))
    return {
        "definition": "GT delta2 error over post-shot k=0,1,2; frozen boundary shift propagated",
        "trajectory_count": used,
        "root_accel_delta2_mm_per_frame2": finite_mean(errors["root"]) * 1000.0,
        "joint_accel_delta2_mm_per_frame2": finite_mean(errors["joint"]) * 1000.0,
        "vertex_accel_delta2_mm_per_frame2": finite_mean(errors["vertex"]) * 1000.0,
        "v1_root_accel_delta2_mm_per_frame2": finite_mean(v1_errors["root"]) * 1000.0,
        "v1_joint_accel_delta2_mm_per_frame2": finite_mean(v1_errors["joint"]) * 1000.0,
        "v1_vertex_accel_delta2_mm_per_frame2": finite_mean(v1_errors["vertex"]) * 1000.0,
    }


def load_prepared(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if rows and isinstance(rows[0].get("case"), dict):
        pass
    return prepare_records(harness.prepare_all(rows))


def run_dev(args: argparse.Namespace) -> None:
    rows = harness.load_rows("dev", args.confirm_report, args.max_cases)
    records = load_prepared(rows)
    cv = cross_validate(records)
    report = {
        "experiment": "v14_brtc_shared_group_alpha_selector",
        "phase": "development_grouped_cv",
        "protocol": {
            "data_read": "three offset0 only; no heldout data loaded",
            "unit": "case-level alpha, aggregate anonymous observable features",
            "labels": "GT case-mean root oracle among {0.8,0.9,1.0}; evaluator only",
            "camera": "bit-exact B0",
            "person_action": "alpha*group + lambda*(individual-group)",
            "safety": "OOD/low confidence/policy error exact v1",
        },
        "dataset_fingerprint": dataset_fingerprint(records),
        "oracle": oracle_report(records),
        "cv": cv,
        "dev_pass": bool(cv["eligible_count"] > 0),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "DEV_CV.json"
    path.write_text(
        json.dumps(jsonable(report), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "DEV_CV.md").write_text(dev_markdown(report), encoding="utf-8")
    print(dev_markdown(report), flush=True)
    print(f">> wrote {path}", flush=True)


def run_freeze(args: argparse.Namespace) -> None:
    dev_path = args.output_dir / "DEV_CV.json"
    if not dev_path.is_file():
        raise FileNotFoundError(f"Run --phase dev first: {dev_path}")
    dev = json.loads(dev_path.read_text(encoding="utf-8"))
    if not bool(dev["dev_pass"]):
        raise RuntimeError("Development CV has no eligible selector; freezing is forbidden")
    # Reopen only the development split and verify it is exactly the one scanned.
    rows = harness.load_rows("dev", args.confirm_report, args.max_cases)
    records = load_prepared(rows)
    fingerprint = dataset_fingerprint(records)
    if fingerprint != dev["dataset_fingerprint"]:
        raise RuntimeError("Development dataset changed between CV and freeze")
    spec = dev["cv"]["selected"]["spec"]
    policy = fit_policy(records, spec)
    decisions = [select_group_alpha(row["features"], policy) for row in records]
    in_sample = evaluate_decisions(records, decisions)
    frozen = {
        "experiment": "v14_brtc_shared_group_alpha_selector",
        "phase": "frozen_before_any_heldout_read",
        "frozen": True,
        "development_source": str(dev_path),
        "development_sha256": sha256(dev_path),
        "development_dataset_fingerprint": fingerprint,
        "selection_spec": spec,
        "cv_selected_summary": dev["cv"]["selected"]["summary"],
        "policy": policy,
        "in_sample_diagnostic_not_selection": {
            key: value for key, value in in_sample.items() if key != "cases"
        },
        "deployment_contract": {
            "strict_online": True,
            "future_or_previous_shot_history": "none beyond BRTC v1 pre/current boundary input",
            "gt_or_image_input": "none",
            "camera_update": "none",
            "allowed_alpha": list(ALPHAS),
            "changed_component": "shared group translation only",
            "individual_residual": "exactly unchanged",
            "ood_low_confidence_error": "exact BRTC v1",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.policy.resolve()
    path.write_text(
        json.dumps(jsonable(frozen), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "frozen_policy": str(path),
                "sha256": sha256(path),
                "spec": spec,
                "cv_summary": dev["cv"]["selected"]["summary"],
            },
            indent=2,
        ),
        flush=True,
    )


def evaluate_split(
    name: str, rows: list[dict[str, Any]], policy: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = load_prepared(rows)
    decisions = [select_group_alpha(row["features"], policy) for row in records]
    summary = evaluate_decisions(records, decisions)
    output = {"name": name, "summary": summary}
    if name in ("dance", "box"):
        output["acceleration"] = acceleration_summary(records, policy)
    else:
        output["acceleration"] = {
            "available": False,
            "reason": "offset1 confirmation has one frame per boundary",
        }
    return output, records


def validate_runtime_contract(policy: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    case = record["case"]
    pre = [person["pre"] for person in case["people"]]
    post = [person["post"] for person in case["people"]]
    matches = [(index, index) for index in range(len(pre))]
    corrected, debug = refine_matched_people_group_alpha_selector(
        case["pre_camera"], case["post_camera"], pre, post, matches, policy
    )
    expected = record["shifts"][float(debug["selected_group_alpha"])]
    shift_delta = 0.0
    residual_delta = 0.0
    rejected_delta = 0.0
    for index, (before, after, expected_shift, row) in enumerate(
        zip(post, corrected, expected, debug["people"])
    ):
        observed = np.asarray(after["root"]) - np.asarray(before["root"])
        shift_delta = max(shift_delta, float(np.max(np.abs(observed - expected_shift))))
        if bool(row["accepted"]):
            base_residual = np.asarray(row["individual_shift_world"]) - np.asarray(
                debug["base_group_shift_world"]
            )
            reconstructed = (
                np.asarray(row["final_shift_world"])
                - np.asarray(debug["group_shift_world"])
            ) / max(float(debug["selected_residual_lambda"]), 1e-12)
            if float(debug["selected_residual_lambda"]) > 0:
                residual_delta = max(
                    residual_delta,
                    float(np.max(np.abs(reconstructed - base_residual))),
                )
        else:
            rejected_delta = max(rejected_delta, float(np.max(np.abs(observed))))
    return {
        "runtime_formula_max_abs_delta": shift_delta,
        "individual_residual_reconstruction_max_abs_delta": residual_delta,
        "rejected_max_abs_change": rejected_delta,
        "camera_update": debug["camera_update"],
        "pass": bool(
            shift_delta <= 1e-12
            and residual_delta <= 1e-12
            and rejected_delta == 0.0
            and debug["camera_update"] == "none"
        ),
    }


def evaluate_ego(policy: dict[str, Any], cache_path: Path) -> dict[str, Any]:
    from versions.v14 import eval_brtc_multithumbs_egohumans as ego

    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    methods, boundary_rows = ego.method_chains(cache)

    def callback(pre_camera, post_camera, pre_people, post_people, matches):
        return refine_matched_people_group_alpha_selector(
            pre_camera, post_camera, pre_people, post_people, matches, policy
        )

    selector_chains, runtime_rows = ego.replay_refinement_variant(
        methods["b0"],
        boundary_rows,
        "b0_brtc_group_alpha_selector",
        callback,
        refine_matched_people_group_alpha_selector.__module__,
    )
    _, exo = ego.load_colmap(ego.DEFAULT_DATA)
    vertex_map, joint_regressor = ego.load_smpl_resources()
    reports, roots = {}, {}
    for name, chains in (
        ("b0", methods["b0"]),
        ("b0_brtc_lc", methods["b0_brtc_lc"]),
        ("b0_brtc_group_alpha_selector", selector_chains),
    ):
        per_chain, arrays, root_rows = [], [], []
        for chain in chains:
            result, raw_arrays, root_errors = ego.evaluate_chain(
                chain, ego.DEFAULT_DATA, exo, vertex_map, joint_regressor, 30.0
            )
            per_chain.append(result)
            arrays.append(raw_arrays)
            root_rows.append(root_errors)
        reports[name] = ego.aggregate_method(per_chain, arrays)
        roots[name] = root_rows
    return {
        "geometry_cache": str(cache_path),
        "methods": reports,
        "selector_harm_vs_b0": ego.harm_audit(
            roots["b0"], roots["b0_brtc_group_alpha_selector"]
        ),
        "v1_harm_vs_b0": ego.harm_audit(roots["b0"], roots["b0_brtc_lc"]),
        "camera_exactness": ego.camera_exactness_audit(methods["b0"], selector_chains),
        "runtime_audit": ego.refinement_runtime_audit(runtime_rows),
        "selected_alpha_counts": dict(
            Counter(
                str(float(row["refinement"]["selected_group_alpha"]))
                for row in runtime_rows
            )
        ),
        "fallback_reasons": dict(
            Counter(
                str(row["refinement"]["selector_decision"]["fallback_reason"])
                for row in runtime_rows
                if row["refinement"]["selector_decision"]["fallback"]
            )
        ),
        "boundary_rows": runtime_rows,
    }


def validation_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Frozen BRTC shared-group alpha selector: held-out validation",
        "",
        f"Frozen policy SHA256: `{report['policy_sha256_before_heldout']}`.",
        "",
    ]
    for name in ("three_offset1", "dance", "box"):
        lines.extend(markdown_split(name, report["splits"][name]))
    if "egohumans" in report:
        metrics = report["egohumans"]["methods"]
        lines.extend(
            [
                "## EgoHumans 001_legoassemble (same-forward CPU cache)",
                "",
                "| Method | W | WA | Root | Joint | Vertex | Pair distance | Pair vector | Root Accel | Harm >5cm |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name in ("b0_brtc_lc", "b0_brtc_group_alpha_selector"):
            value = metrics[name]["metrics"]
            harm = (
                report["egohumans"]["v1_harm_vs_b0"]
                if name == "b0_brtc_lc"
                else report["egohumans"]["selector_harm_vs_b0"]
            )["all_person_frames_in_corrected_post_shots"]["harm_over_5cm_rate"]
            lines.append(
                f"| {name} | {value['w_mpjpe_mm']:.3f} | {value['wa_mpjpe_mm']:.3f} | "
                f"{value['fixed_world_root_mm']:.3f} | {value['fixed_world_joint_mm']:.3f} | "
                f"{value['fixed_world_vertex_mm']:.3f} | {value['pairwise_root_distance_mm']:.3f} | "
                f"{value['pairwise_root_vector_mm']:.3f} | "
                f"{value['world_root_accel_delta2_mm_per_frame2']:.3f} | {harm:.1%} |"
            )
        lines.extend(
            [
                "",
                f"- α counts: `{report['egohumans']['selected_alpha_counts']}`; "
                f"fallback: `{report['egohumans']['fallback_reasons']}`.",
                "",
            ]
        )
    lines.extend(
        [
            f"- Held-out winner: **{report['heldout_winner']}**.",
            f"- Decision: **{report['decision']}**.",
            "- No held-out result was used to alter the frozen policy.",
        ]
    )
    return "\n".join(lines) + "\n"


def split_dominates_v1(split: dict[str, Any]) -> bool:
    value = split["summary"]
    return bool(
        value["non_v1_action_count"] > 0
        and all(value["delta_vs_v1"][key] <= 1e-12 for key in ERROR_METRICS)
        and value["root_harm_over_1cm_rate"] <= value["v1_root_harm_over_1cm_rate"] + 1e-12
        and value["root_harm_over_5cm_rate"] <= value["v1_root_harm_over_5cm_rate"] + 1e-12
    )


def run_validate(args: argparse.Namespace) -> None:
    policy_path = args.policy.resolve()
    frozen = json.loads(policy_path.read_text(encoding="utf-8"))
    if not frozen.get("frozen", False):
        raise RuntimeError("A pre-heldout frozen policy is required")
    policy_sha = sha256(policy_path)
    policy = frozen["policy"]
    # Only after policy bytes and hash are fixed do we open held-out reports.
    offset1_rows = harness.load_rows("confirm", args.confirm_report, args.max_cases)
    dance_rows = legacy.report_rows(("dance",))[: int(args.max_cases) or None]
    box_rows = legacy.report_rows(("box",))[: int(args.max_cases) or None]
    splits, records_by_split = {}, {}
    for name, rows in (
        ("three_offset1", offset1_rows),
        ("dance", dance_rows),
        ("box", box_rows),
    ):
        split, records = evaluate_split(name, rows, policy)
        splits[name] = split
        records_by_split[name] = records
    runtime_contract = validate_runtime_contract(
        policy, records_by_split["three_offset1"][0]
    )
    report = {
        "experiment": "v14_brtc_shared_group_alpha_selector",
        "phase": "frozen_heldout_validation",
        "policy_source": str(policy_path),
        "policy_sha256_before_heldout": policy_sha,
        "heldout_policy_unchanged": True,
        "splits": splits,
        "runtime_contract": runtime_contract,
    }
    if not args.skip_ego:
        report["egohumans"] = evaluate_ego(policy, args.ego_geometry_cache)
    split_wins = {name: split_dominates_v1(value) for name, value in splits.items()}
    ego_win = False
    if "egohumans" in report:
        ego = report["egohumans"]
        old = ego["methods"]["b0_brtc_lc"]["metrics"]
        new = ego["methods"]["b0_brtc_group_alpha_selector"]["metrics"]
        keys = (
            "fixed_world_root_mm",
            "fixed_world_joint_mm",
            "fixed_world_vertex_mm",
            "pairwise_root_distance_mm",
            "pairwise_root_vector_mm",
            "world_root_accel_delta2_mm_per_frame2",
        )
        old_harm = ego["v1_harm_vs_b0"]["all_person_frames_in_corrected_post_shots"]["harm_over_5cm_rate"]
        new_harm = ego["selector_harm_vs_b0"]["all_person_frames_in_corrected_post_shots"]["harm_over_5cm_rate"]
        ego_win = bool(
            sum(ego["selected_alpha_counts"].get(str(alpha), 0) for alpha in ALPHAS if alpha < 1.0) > 0
            and all(new[key] <= old[key] + 1e-12 for key in keys)
            and new_harm <= old_harm + 1e-12
            and ego["camera_exactness"]["bit_exact"]
        )
    report["split_dominates_v1"] = split_wins
    report["egohumans_dominates_v1"] = ego_win
    report["heldout_winner"] = bool(all(split_wins.values()) and (ego_win if "egohumans" in report else True))
    report["decision"] = "PROMOTE" if report["heldout_winner"] else "NO_GO_ARCHIVE"
    report["policy_sha256_after_heldout"] = sha256(policy_path)
    report["heldout_policy_unchanged"] = bool(
        report["policy_sha256_after_heldout"] == policy_sha
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "HELDOUT_RESULTS.json"
    path.write_text(
        json.dumps(jsonable(report), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    text = validation_markdown(report)
    (args.output_dir / "HELDOUT_RESULTS.md").write_text(text, encoding="utf-8")
    doc = REPO_ROOT / "versions/v14/docs/V14_BRTC_GROUP_ALPHA_SELECTOR_20260801.md"
    doc.write_text(text, encoding="utf-8")
    print(text, flush=True)
    print(f">> wrote {path}", flush=True)


def main() -> None:
    args = parse_args()
    if not str(args.output_dir.resolve()).startswith(str(REPO_ROOT.resolve())):
        raise ValueError("Output must remain inside Movie3R /data workspace")
    if args.phase == "dev":
        run_dev(args)
    elif args.phase == "freeze":
        run_freeze(args)
    else:
        run_validate(args)


if __name__ == "__main__":
    main()
