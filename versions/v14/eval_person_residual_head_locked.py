#!/usr/bin/env python3
"""Locked safety evaluation for the V14 person root-ray residual head.

The candidate is intentionally not tunable from the command line:

* Ridge alpha: 10
* feature contract: the frozen 172-dimensional deployment feature list
* acceptance gate: ``abs(raw_prediction) >= 0.20 m``
* applied correction: rigid root-ray translation clipped to ``+/-0.05 m``
* rejection: bit-exact frozen-B0 person geometry

The default input contains double-block timestamp/actor OOF predictions on the
``three`` development split.  This evaluator reconstructs predicted geometry,
freezes the candidate, and only then reads GT for scoring.  It never loads
``dance`` or ``box``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from versions.v14.probe_b0_anchor_conflict import SEQUENCE_INPUTS  # noqa: E402
from versions.v14.probe_person_residual_head_cv import (  # noqa: E402
    build_deployment_rows,
    transform_points,
    validate_feature_contract,
)


LOCKED_RIDGE_ALPHA = 10.0
LOCKED_GATE_ABS_RAW_M = 0.20
LOCKED_CORRECTION_CAP_M = 0.05
EXPECTED_FEATURE_COUNT = 172
DEFAULT_OOF = (
    REPO_ROOT
    / "output/v14/fine_alignment_research/"
    "person_residual_head_cv_three_double_block_ridge/"
    "v14_person_residual_head_cv.json"
)
DEFAULT_MATCHING = (
    REPO_ROOT / "output/v14/b0_identity_matching/v14_b0_identity_matching.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "output/v14/fine_alignment_research/person_residual_head_locked_three"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof_report", type=Path, default=DEFAULT_OOF)
    parser.add_argument("--matching_report", type=Path, default=DEFAULT_MATCHING)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
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


def stats(values: list[float] | np.ndarray) -> dict[str, float | int]:
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


def point_error(predicted: np.ndarray, target: np.ndarray) -> float:
    count = min(len(predicted), len(target))
    return float(
        np.linalg.norm(
            np.asarray(predicted[:count]) - np.asarray(target[:count]), axis=1
        ).mean()
    )


def error_summary(before: list[float], after: list[float]) -> dict:
    before_array = np.asarray(before, dtype=np.float64)
    after_array = np.asarray(after, dtype=np.float64)
    delta = after_array - before_array
    return {
        "b0": stats(before_array),
        "locked": stats(after_array),
        "mean_delta_m": float(np.mean(delta)),
        "relative_mean_change": float(
            np.mean(after_array) / max(float(np.mean(before_array)), 1e-12) - 1.0
        ),
        "improve_rate": float(np.mean(after_array < before_array - 1e-12)),
        "worsen_rate": float(np.mean(after_array > before_array + 1e-12)),
        "harm_over_1cm_rate": float(np.mean(delta > 0.01)),
        "harm_over_5cm_rate": float(np.mean(delta > 0.05)),
        "harm_over_10cm_rate": float(np.mean(delta > 0.10)),
        "max_harm_m": float(np.max(delta)),
    }


def feature_parity(
    rebuilt: dict[str, float], saved: dict[str, float], feature_names: list[str]
) -> float:
    maximum = 0.0
    for name in feature_names:
        first = float(rebuilt.get(name, float("nan")))
        second = float(saved.get(name, float("nan")))
        if math.isnan(first) and math.isnan(second):
            continue
        if not math.isfinite(first) or not math.isfinite(second):
            raise RuntimeError(f"Nonmatching nonfinite feature {name}: {first}, {second}")
        maximum = max(maximum, abs(first - second))
    return maximum


def locked_candidate(raw_prediction: float, ray: np.ndarray) -> dict:
    accepted = bool(abs(float(raw_prediction)) >= LOCKED_GATE_ABS_RAW_M)
    correction = (
        float(
            np.clip(
                float(raw_prediction),
                -LOCKED_CORRECTION_CAP_M,
                LOCKED_CORRECTION_CAP_M,
            )
        )
        if accepted
        else 0.0
    )
    return {
        "accepted": accepted,
        "raw_prediction_m": float(raw_prediction),
        "correction_m": correction,
        "shift": correction * np.asarray(ray, dtype=np.float64),
    }


def case_pairwise(people: list[dict], root_key: str) -> tuple[list[float], list[float]]:
    distances, vectors = [], []
    for first_index, first in enumerate(people):
        for second in people[first_index + 1 :]:
            predicted_vector = np.asarray(first[root_key]) - np.asarray(second[root_key])
            target_vector = np.asarray(first["target_root"]) - np.asarray(
                second["target_root"]
            )
            distances.append(
                abs(float(np.linalg.norm(predicted_vector) - np.linalg.norm(target_vector)))
            )
            vectors.append(float(np.linalg.norm(predicted_vector - target_vector)))
    return distances, vectors


def fit_final_head(oof: dict, feature_names: list[str]) -> dict:
    samples = oof["samples"]
    matrix = np.asarray(
        [
            [sample["features"].get(name, float("nan")) for name in feature_names]
            for sample in samples
        ],
        dtype=np.float64,
    )
    labels = np.asarray(
        [sample["label_correction_m"] for sample in samples], dtype=np.float64
    )
    target_mean = float(np.mean(labels))
    target_scale = max(float(np.std(labels)), 1e-8)
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=LOCKED_RIDGE_ALPHA)),
        ]
    )
    pipeline.fit(matrix, (labels - target_mean) / target_scale)
    return {
        "format": "v14_person_root_ray_ridge_v1",
        "sequence_trained": "three",
        "holdouts_evaluated": [],
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "pipeline": pipeline,
        "target_mean_m": target_mean,
        "target_scale_m": target_scale,
        "ridge_alpha": LOCKED_RIDGE_ALPHA,
        "gate_abs_raw_m": LOCKED_GATE_ABS_RAW_M,
        "correction_cap_m": LOCKED_CORRECTION_CAP_M,
        "output_semantics": (
            "predict signed root-ray residual; accept iff abs(raw)>=gate; "
            "clip accepted residual to cap; rigid-shift root/joints/vertices; camera unchanged"
        ),
    }


def acceptance_decision(metrics: dict, verification: dict) -> dict:
    """Predeclared development safety gate; this is not holdout promotion."""
    root = metrics["root"]
    joint = metrics["joint"]
    vertex = metrics["vertex"]
    pair_distance = metrics["pairwise_distance"]
    pair_vector = metrics["pairwise_vector"]
    checks = {
        "camera_bit_exact": bool(verification["camera_bit_exact"]),
        "shared_rigid_shift_exact": bool(
            verification["max_rigid_shift_deviation"] <= 1e-12
        ),
        "fallback_bit_exact": bool(verification["fallback_bit_exact"]),
        "root_mean_gain_at_least_3pct": bool(
            root["relative_mean_change"] <= -0.03
        ),
        "joint_mean_improves": bool(joint["mean_delta_m"] < 0.0),
        "vertex_mean_improves": bool(vertex["mean_delta_m"] < 0.0),
        "root_joint_vertex_p95_noninferior": bool(
            root["locked"]["p95"] <= root["b0"]["p95"]
            and joint["locked"]["p95"] <= joint["b0"]["p95"]
            and vertex["locked"]["p95"] <= vertex["b0"]["p95"]
        ),
        "pairwise_means_within_1pct": bool(
            pair_distance["locked"]["mean"]
            <= 1.01 * pair_distance["b0"]["mean"]
            and pair_vector["locked"]["mean"]
            <= 1.01 * pair_vector["b0"]["mean"]
        ),
        "no_metric_harm_over_5cm": bool(
            root["harm_over_5cm_rate"] == 0.0
            and joint["harm_over_5cm_rate"] == 0.0
            and vertex["harm_over_5cm_rate"] == 0.0
        ),
        "accepted_sign_at_least_75pct": bool(
            metrics["accepted"]["sign_accuracy"] >= 0.75
        ),
        "accepted_root_improve_at_least_75pct": bool(
            metrics["accepted"]["root_improve_rate"] >= 0.75
        ),
    }
    return {
        "status": "pass_development_gate" if all(checks.values()) else "fail_development_gate",
        "all_checks_pass": bool(all(checks.values())),
        "checks": checks,
        "meaning": (
            "Passing permits saving a `three`-trained candidate only. It does not "
            "authorize or imply dance/box holdout success."
        ),
    }


def markdown_report(report: dict) -> str:
    protocol = report["protocol"]
    metrics = report["metrics"]
    verification = report["verification"]
    decision = report["decision"]
    lines = [
        "# V14 Locked Person Residual Head — `three` Development Safety Evaluation",
        "",
        "Candidate is frozen: Ridge `alpha=10`, 172 features, gate "
        "`abs(raw)>=0.20 m`, accepted rigid root-ray correction clipped to `±0.05 m`.",
        "Rejected people are bit-exact B0 and the camera is never modified.",
        "",
        f"Cases/people/accepted: `{protocol['case_count']}/{protocol['person_count']}/"
        f"{protocol['accepted_count']}`; coverage `{100*protocol['coverage']:.1f}%`.",
        "Predictions are timestamp+actor double-block OOF. `dance` and `box` were not loaded.",
        "",
        "## Full geometry metrics",
        "",
        "| Metric | B0 mean | Locked mean | B0 P50 | Locked P50 | B0 P90 | Locked P90 | B0 P95 | Locked P95 | Delta | Improve | Harm >5cm |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, display in (
        ("root", "Root"),
        ("joint", "Joint"),
        ("vertex", "Vertex"),
        ("pairwise_distance", "Pair distance"),
        ("pairwise_vector", "Pair vector"),
    ):
        row = metrics[name]
        lines.append(
            f"| {display} | {row['b0']['mean']:.4f} | {row['locked']['mean']:.4f} | "
            f"{row['b0']['p50']:.4f} | {row['locked']['p50']:.4f} | "
            f"{row['b0']['p90']:.4f} | {row['locked']['p90']:.4f} | "
            f"{row['b0']['p95']:.4f} | {row['locked']['p95']:.4f} | "
            f"{row['mean_delta_m']:+.4f} | {100*row['improve_rate']:.1f}% | "
            f"{100*row['harm_over_5cm_rate']:.1f}% |"
        )

    accepted = metrics["accepted"]
    lines.extend(
        [
            "",
            "## Gate and safety",
            "",
            f"- accepted sign accuracy: `{100*accepted['sign_accuracy']:.1f}%`;",
            f"- accepted root/joint/vertex improve: `"
            f"{100*accepted['root_improve_rate']:.1f}% / "
            f"{100*accepted['joint_improve_rate']:.1f}% / "
            f"{100*accepted['vertex_improve_rate']:.1f}%`;",
            f"- camera bit-exact: `{verification['camera_bit_exact']}`;",
            f"- rigid shift max deviation: `{verification['max_rigid_shift_deviation']:.3e}`;",
            f"- rejected fallback bit-exact: `{verification['fallback_bit_exact']}` "
            f"on `{verification['fallback_people']}` people;",
            f"- rebuilt-vs-saved deployment feature max difference: "
            f"`{verification['feature_parity_max_abs']:.3e}`.",
            "",
            f"Development decision: **{decision['status']}**.",
            "",
            "| Check | Pass |",
            "|---|---:|",
        ]
    )
    for name, passed in decision["checks"].items():
        lines.append(f"| `{name}` | {passed} |")
    lines.extend(
        [
            "",
            "This is still a development-only candidate. A saved all-`three` model has not "
            "seen `dance` or `box`; no holdout claim is made.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    oof = json.loads(args.oof_report.read_text(encoding="utf-8"))
    matching = json.loads(args.matching_report.read_text(encoding="utf-8"))
    oof_protocol = oof["protocol"]
    if oof_protocol.get("sequence") != "three":
        raise ValueError("OOF report is not `three`")
    if oof_protocol.get("split") != "double_block_timestamp_actor":
        raise ValueError("Locked evaluation requires double-block OOF predictions")
    if oof_protocol.get("models") != ["ridge"]:
        raise ValueError("Locked evaluation requires the ridge-only OOF report")
    if float(oof_protocol["ridge_alpha_fixed"]) != LOCKED_RIDGE_ALPHA:
        raise ValueError("OOF ridge alpha differs from locked alpha=10")
    feature_names = [str(value) for value in oof_protocol["feature_names"]]
    if len(feature_names) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} frozen features, got {len(feature_names)}"
        )
    validate_feature_contract(feature_names)

    saved_samples = {
        (
            str(row["case_key"]),
            str(row["pre_track_key_evaluator_only"]),
            str(row["post_detection_key_evaluator_only"]),
        ): row
        for row in oof["samples"]
    }
    matching_cases = {str(row["case"]["key"]): row for row in matching["cases"]}
    cache_root = SEQUENCE_INPUTS["three"]["cache"]

    people_by_case: defaultdict[str, list[dict]] = defaultdict(list)
    root_before, root_after = [], []
    joint_before, joint_after = [], []
    vertex_before, vertex_after = [], []
    accepted_flags, accepted_sign = [], []
    camera_bit_exact = True
    fallback_bit_exact = True
    fallback_people = 0
    maximum_rigid_deviation = 0.0
    maximum_feature_difference = 0.0
    seen_samples = set()

    for case_index, case_key in enumerate(oof_protocol["cases"], start=1):
        case_key = str(case_key)
        report_case = matching_cases[case_key]
        cache = torch.load(
            cache_root / f"{case_key}.pt", map_location="cpu", weights_only=False
        )
        boundary = np.asarray(
            report_case["boundaries"]["learned_b0"], dtype=np.float64
        )
        frozen_matching = report_case["matching"]["learned_b0"]
        cache_view = {
            "poses": cache["poses"],
            "humans": cache["humans"],
            "pre_frames": cache["case"]["pre_frames"],
        }
        deployment_rows = build_deployment_rows(
            cache_view, boundary, frozen_matching
        )
        pre_pose = np.asarray(cache["poses"][-2], dtype=np.float64)
        raw_post_pose = np.asarray(cache["poses"][-1], dtype=np.float64)
        frozen_camera = boundary @ raw_post_pose
        candidate_camera = frozen_camera.copy()
        camera_bit_exact &= bool(np.array_equal(candidate_camera, frozen_camera))

        frozen_people = []
        for deployment in deployment_rows:
            sample_key = (
                case_key,
                str(deployment["pre_key"]),
                str(deployment["post_key"]),
            )
            if sample_key not in saved_samples:
                raise KeyError(f"Missing OOF prediction for {sample_key}")
            saved = saved_samples[sample_key]
            seen_samples.add(sample_key)
            maximum_feature_difference = max(
                maximum_feature_difference,
                feature_parity(deployment["features"], saved["features"], feature_names),
            )
            proposal = locked_candidate(
                float(saved["ridge_raw_prediction_m"]), deployment["ray"]
            )
            accepted_flags.append(proposal["accepted"])

            post_human = cache["humans"][-1][deployment["post_key"]]
            b0_root = np.asarray(deployment["root"], dtype=np.float64)
            b0_joints = transform_points(boundary, post_human["joints"])
            b0_vertices = transform_points(boundary, post_human["vertices"])
            shift = np.asarray(proposal["shift"], dtype=np.float64)
            locked_root = b0_root + shift
            locked_joints = b0_joints + shift
            locked_vertices = b0_vertices + shift

            maximum_rigid_deviation = max(
                maximum_rigid_deviation,
                float(np.max(np.abs((locked_root - b0_root) - shift))),
                float(np.max(np.abs((locked_joints - b0_joints) - shift[None]))),
                float(np.max(np.abs((locked_vertices - b0_vertices) - shift[None]))),
            )
            if not proposal["accepted"]:
                fallback_people += 1
                fallback_bit_exact &= bool(
                    np.array_equal(locked_root, b0_root)
                    and np.array_equal(locked_joints, b0_joints)
                    and np.array_equal(locked_vertices, b0_vertices)
                )

            frozen_people.append(
                {
                    "deployment": deployment,
                    "saved": saved,
                    "proposal": proposal,
                    "b0_root": b0_root,
                    "b0_joints": b0_joints,
                    "b0_vertices": b0_vertices,
                    "locked_root": locked_root,
                    "locked_joints": locked_joints,
                    "locked_vertices": locked_vertices,
                }
            )

        # GT enters only after every feature, gate decision, correction, full
        # person geometry, and the unchanged camera have been frozen.
        gt = cache["gt"]
        gauge = pre_pose @ np.linalg.inv(np.asarray(gt["pre_c2w"], dtype=np.float64))

        for frozen in frozen_people:
            deployment = frozen["deployment"]
            saved = frozen["saved"]
            proposal = frozen["proposal"]
            b0_root = frozen["b0_root"]
            b0_joints = frozen["b0_joints"]
            b0_vertices = frozen["b0_vertices"]
            locked_root = frozen["locked_root"]
            locked_joints = frozen["locked_joints"]
            locked_vertices = frozen["locked_vertices"]
            identity = deployment["pre_key"]
            target_human = gt["post_humans"][identity]
            target_root = transform_points(gauge, target_human["root"])
            target_joints = transform_points(gauge, target_human["joints"])
            target_vertices = transform_points(gauge, target_human["vertices"])

            before_root_error = float(np.linalg.norm(b0_root - target_root))
            after_root_error = float(np.linalg.norm(locked_root - target_root))
            before_joint_error = point_error(b0_joints, target_joints)
            after_joint_error = point_error(locked_joints, target_joints)
            before_vertex_error = point_error(b0_vertices, target_vertices)
            after_vertex_error = point_error(locked_vertices, target_vertices)
            root_before.append(before_root_error)
            root_after.append(after_root_error)
            joint_before.append(before_joint_error)
            joint_after.append(after_joint_error)
            vertex_before.append(before_vertex_error)
            vertex_after.append(after_vertex_error)

            if proposal["accepted"]:
                accepted_sign.append(
                    bool(
                        np.sign(proposal["raw_prediction_m"])
                        == np.sign(float(saved["label_correction_m"]))
                    )
                )
            people_by_case[case_key].append(
                {
                    "accepted": proposal["accepted"],
                    "b0_root": b0_root,
                    "locked_root": locked_root,
                    "target_root": target_root,
                    "root_before": before_root_error,
                    "root_after": after_root_error,
                    "joint_before": before_joint_error,
                    "joint_after": after_joint_error,
                    "vertex_before": before_vertex_error,
                    "vertex_after": after_vertex_error,
                }
            )
        print(
            f"[{case_index}/{len(oof_protocol['cases'])}] {case_key}: "
            f"people={len(deployment_rows)}",
            flush=True,
        )

    if seen_samples != set(saved_samples):
        missing = set(saved_samples) - seen_samples
        raise RuntimeError(f"Did not reconstruct all OOF samples: {sorted(missing)[:3]}")

    pair_distance_before, pair_distance_after = [], []
    pair_vector_before, pair_vector_after = [], []
    for people in people_by_case.values():
        distance, vector = case_pairwise(people, "b0_root")
        pair_distance_before.extend(distance)
        pair_vector_before.extend(vector)
        distance, vector = case_pairwise(people, "locked_root")
        pair_distance_after.extend(distance)
        pair_vector_after.extend(vector)

    accepted_people = [
        person
        for people in people_by_case.values()
        for person in people
        if person["accepted"]
    ]
    metrics = {
        "root": error_summary(root_before, root_after),
        "joint": error_summary(joint_before, joint_after),
        "vertex": error_summary(vertex_before, vertex_after),
        "pairwise_distance": error_summary(
            pair_distance_before, pair_distance_after
        ),
        "pairwise_vector": error_summary(pair_vector_before, pair_vector_after),
        "accepted": {
            "count": len(accepted_people),
            "sign_accuracy": float(np.mean(accepted_sign)),
            "root_improve_rate": float(
                np.mean(
                    [person["root_after"] < person["root_before"] for person in accepted_people]
                )
            ),
            "joint_improve_rate": float(
                np.mean(
                    [person["joint_after"] < person["joint_before"] for person in accepted_people]
                )
            ),
            "vertex_improve_rate": float(
                np.mean(
                    [person["vertex_after"] < person["vertex_before"] for person in accepted_people]
                )
            ),
        },
    }
    verification = {
        "camera_bit_exact": bool(camera_bit_exact),
        "camera_max_abs_difference": 0.0,
        "max_rigid_shift_deviation": maximum_rigid_deviation,
        "fallback_bit_exact": bool(fallback_bit_exact),
        "fallback_people": int(fallback_people),
        "feature_parity_max_abs": maximum_feature_difference,
        "candidate_built_before_gt_access": True,
    }
    decision = acceptance_decision(metrics, verification)

    output = {
        "experiment": "V14 locked person residual-head full-geometry development safety evaluation",
        "protocol": {
            "sequence": "three",
            "case_count": len(oof_protocol["cases"]),
            "person_count": len(saved_samples),
            "accepted_count": int(np.sum(accepted_flags)),
            "coverage": float(np.mean(accepted_flags)),
            "pair_count": len(pair_distance_before),
            "oof_split": "double_block_timestamp_actor",
            "ridge_alpha": LOCKED_RIDGE_ALPHA,
            "feature_count": len(feature_names),
            "gate_abs_raw_m": LOCKED_GATE_ABS_RAW_M,
            "correction_cap_m": LOCKED_CORRECTION_CAP_M,
            "camera_changed": False,
            "dance_or_box_loaded": False,
            "gt_use": "training label and evaluator only",
        },
        "verification": verification,
        "metrics": metrics,
        "decision": decision,
        "limitations": [
            "This is one-capture `three` development evidence only.",
            "OOF labels use GT and split metadata uses GT actor/timestamp, neither enters deployment features.",
            "Passing the development gate saves a candidate but does not promote it before frozen holdout evaluation.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "v14_person_residual_head_three.joblib"
    if decision["all_checks_pass"]:
        final_head = fit_final_head(oof, feature_names)
        joblib.dump(final_head, model_path)
        output["final_model"] = {
            "saved": True,
            "path": str(model_path),
            "trained_samples": len(oof["samples"]),
            "holdouts_evaluated": [],
        }
    else:
        output["final_model"] = {
            "saved": False,
            "reason": "locked candidate failed at least one predeclared development check",
            "holdouts_evaluated": [],
        }

    json_path = args.output_dir / "v14_person_residual_head_locked.json"
    md_path = args.output_dir / "v14_person_residual_head_locked.md"
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
