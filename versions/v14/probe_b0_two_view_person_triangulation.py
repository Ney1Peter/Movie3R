#!/usr/bin/env python3
"""Evaluate frozen-B0 two-view per-person triangulation on MultiHuman caches.

This is the real-camera follow-up to ``probe_two_view_person_triangulation.py``.
It uses the last pre-cut Human3R body, the first raw-reset post-cut body, and a
precomputed frozen B0.  Cameras are never changed.  For each already-associated
person, five core-joint ray pairs propose one post-cut pelvis depth; the only
action is a rigid translation along the aligned post camera ray.

GT identity is used to isolate WHERE in this probe.  The same B0 matcher was
already evaluated independently on these exact cases; automatic-ID integration
is a separate end-to-end step and cannot alter the depth proposal.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from smplx.joint_names import JOINT_NAMES


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from versions.v14.probe_b0_anchor_conflict import SEQUENCE_INPUTS
from versions.v14.probe_b0_da3_person_ray_anchor import FROZEN_REPORT, THREE_REPORT
from versions.v14.probe_two_view_person_triangulation import closest_rays


DEFAULT_OUTPUT = REPO_ROOT / "output/v14/fine_alignment_research/b0_two_view_person_triangulation"
DEFAULT_POLICY = (
    REPO_ROOT
    / "output/v14/fine_alignment_research/two_view_person_triangulation/"
    "FROZEN_TRIANGULATION_POLICY_BEFORE_CONFIRM.json"
)
DEFAULT_K1_REPORT = (
    REPO_ROOT / "output/v14/b0_identity_matching_offset1_confirm/v14_b0_identity_matching.json"
)
CORE5 = (0, 1, 2, 16, 17)
METRICS = ("root_error_m", "joint_error_m", "vertex_error_m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("dev", "frozen", "confirm"), required=True)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--confirm_report", type=Path, default=DEFAULT_K1_REPORT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max_cases_per_sequence", type=int, default=0)
    return parser.parse_args()


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return np.einsum("ij,...j->...i", transform[:3, :3], points) + transform[:3, 3]


def finite_stats(values: list[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {key: float("nan") for key in ("mean", "median", "p90", "p95", "max")}
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def report_rows(sequences: tuple[str, ...]) -> list[dict]:
    rows = []
    if "three" in sequences:
        rows.extend(json.loads(THREE_REPORT.read_text(encoding="utf-8"))["cases"])
    if any(name in sequences for name in ("dance", "box")):
        frozen = json.loads(FROZEN_REPORT.read_text(encoding="utf-8"))["cases"]
        rows.extend(row for row in frozen if row["sequence"] in sequences)
    return [row for row in rows if row["sequence"] in sequences]


def point_errors(predicted: dict[str, np.ndarray], target: dict[str, np.ndarray]) -> dict[str, float]:
    joint_count = min(len(predicted["joints"]), len(target["joints"]))
    vertex_count = min(len(predicted["vertices"]), len(target["vertices"]))
    return {
        "root_error_m": float(np.linalg.norm(predicted["root"] - target["root"])),
        "joint_error_m": float(
            np.linalg.norm(
                predicted["joints"][:joint_count] - target["joints"][:joint_count], axis=1
            ).mean()
        ),
        "vertex_error_m": float(
            np.linalg.norm(
                predicted["vertices"][:vertex_count] - target["vertices"][:vertex_count], axis=1
            ).mean()
        ),
    }


def person_evidence(
    pre: dict[str, np.ndarray],
    post: dict[str, np.ndarray],
    pre_camera: np.ndarray,
    post_camera: np.ndarray,
) -> dict[str, Any]:
    post_root = np.asarray(post["root"], dtype=np.float64)
    camera_center = post_camera[:3, 3]
    ray = post_root - camera_center
    ray = ray / max(float(np.linalg.norm(ray)), 1e-12)
    candidates, gaps, sines = [], [], []
    for joint_id in CORE5:
        if joint_id >= len(pre["joints"]) or joint_id >= len(post["joints"]):
            continue
        joint_a = np.asarray(pre["joints"][joint_id], dtype=np.float64)
        joint_b = np.asarray(post["joints"][joint_id], dtype=np.float64)
        direction_a = joint_a - pre_camera[:3, 3]
        direction_b = joint_b - post_camera[:3, 3]
        if min(np.linalg.norm(direction_a), np.linalg.norm(direction_b)) <= 1e-8:
            continue
        midpoint, depth_a, depth_b, gap, sine = closest_rays(
            pre_camera[:3, 3], direction_a, post_camera[:3, 3], direction_b
        )
        if depth_a <= 0.0 or depth_b <= 0.0 or sine <= 1e-5:
            continue
        candidate_root = midpoint - (joint_b - post_root)
        delta = float(np.dot(candidate_root - post_root, ray))
        if not np.isfinite(delta):
            continue
        candidates.append(delta)
        gaps.append(gap)
        sines.append(sine)
    if not candidates:
        return {
            "raw": float("nan"), "valid_count": 0, "median_gap": float("inf"),
            "max_gap": float("inf"), "median_sine": 0.0, "min_sine": 0.0,
            "mad": float("inf"), "ray": ray,
        }
    candidate_array = np.asarray(candidates, dtype=np.float64)
    center = float(np.median(candidate_array))
    return {
        "raw": center,
        "valid_count": len(candidates),
        "median_gap": float(np.median(gaps)),
        "max_gap": float(np.max(gaps)),
        "median_sine": float(np.median(sines)),
        "min_sine": float(np.min(sines)),
        "mad": float(np.median(np.abs(candidate_array - center))),
        "ray": ray,
    }


def accepted_action(evidence: dict[str, Any], policy: dict[str, Any]) -> tuple[float, bool]:
    valid = bool(
        np.isfinite(evidence["raw"])
        and evidence["valid_count"] >= int(policy["min_valid"])
        and evidence["median_gap"] <= float(policy["max_median_gap_m"])
        and evidence["mad"] <= float(policy["max_mad_m"])
        and evidence["median_sine"] >= float(policy["min_median_sine"])
        and abs(evidence["raw"]) >= float(policy["min_abs_raw_m"])
    )
    action = (
        float(np.clip(evidence["raw"], -float(policy["cap_m"]), float(policy["cap_m"])))
        if valid else 0.0
    )
    return action, valid


def evaluate_case(row: dict, policy: dict[str, Any]) -> dict[str, Any]:
    sequence = str(row.get("sequence", str(row["case"]["key"]).split("_", 1)[0]))
    cache_path = SEQUENCE_INPUTS[sequence]["cache"] / f"{row['case']['key']}.pt"
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    if "methods" in row:
        boundary = np.asarray(row["methods"]["b0"]["boundary"], dtype=np.float64)
        b0_rotation_error = float(row["methods"]["b0"]["camera_rotation_error_deg"])
    else:
        boundary = np.asarray(row["boundaries"]["learned_b0"], dtype=np.float64)
        b0_rotation_error = float(row["learned_b0_camera_error"]["rotation_deg"])
    pre_camera = np.asarray(cache["poses"][-2], dtype=np.float64)
    raw_post_camera = np.asarray(cache["poses"][-1], dtype=np.float64)
    post_camera = boundary @ raw_post_camera
    gt_pre = np.asarray(cache["gt"]["pre_c2w"], dtype=np.float64)
    gt_post = np.asarray(cache["gt"]["post_c2w"], dtype=np.float64)
    gauge = pre_camera @ np.linalg.inv(gt_pre)
    target_camera = gauge @ gt_post

    if "matching" in row:
        association = row["matching"]["learned_b0"]["matchers"][
            "root_torso_joints"
        ]["predicted_identity_by_pre_identity"]
        association_source = "anonymous B0 root+torso+joints Hungarian matcher"
    else:
        association = {identity: identity for identity in cache["humans"][-1]}
        association_source = "GT-ID WHERE isolation"
    identities = tuple(
        identity for identity in sorted(association)
        if identity in cache["humans"][-2]
        and association[identity] in cache["humans"][-1]
        and identity in cache["gt"]["post_humans"]
    )
    prepared = []
    target_roots = {}
    for identity in identities:
        post_identity = association[identity]
        pre_raw = cache["humans"][-2][identity]
        post_raw = cache["humans"][-1][post_identity]
        target_raw = cache["gt"]["post_humans"][identity]
        pre = {
            key: np.asarray(pre_raw[key], dtype=np.float64)
            for key in ("root", "joints", "vertices")
        }
        post = {
            key: transform_points(boundary, np.asarray(post_raw[key], dtype=np.float64))
            for key in ("root", "joints", "vertices")
        }
        target = {
            key: transform_points(gauge, np.asarray(target_raw[key], dtype=np.float64))
            for key in ("root", "joints", "vertices")
        }
        evidence = person_evidence(pre, post, pre_camera, post_camera)
        action, accepted = accepted_action(evidence, policy)
        oracle_label = float(np.dot(target["root"] - post["root"], evidence["ray"]))
        prepared.append({
            "identity": identity,
            "predicted_post_identity_evaluation_only": post_identity,
            "association_correct_evaluation_only": bool(post_identity == identity),
            "pre": pre,
            "post": post,
            "target": target,
            "evidence": evidence,
            "action": action,
            "accepted": accepted,
            "oracle_label": oracle_label,
            "individual_shift": action * evidence["ray"],
        })
        target_roots[identity] = target["root"]

    accepted_shifts = [item["individual_shift"] for item in prepared if item["accepted"]]
    group_shift = (
        np.median(np.stack(accepted_shifts), axis=0)
        if accepted_shifts else np.zeros(3, dtype=np.float64)
    )
    # Decompose each depth proposal into a shared human-layer translation and a
    # person residual.  Choose the residual strength only from the observable
    # pre/post predicted layout.  The ascending grid makes a tie conservative.
    lambda_grid = (0.0, 0.25, 0.50, 0.75, 1.0)
    lambda_objectives = {}
    for residual_lambda in lambda_grid:
        proposed_roots = {}
        for item in prepared:
            if item["accepted"]:
                shift = group_shift + residual_lambda * (
                    item["individual_shift"] - group_shift
                )
            else:
                # Preserve the per-person exact fallback contract.  A reliable
                # peer must not silently move a person whose own rays failed.
                shift = np.zeros(3, dtype=np.float64)
            proposed_roots[item["identity"]] = item["post"]["root"] + shift
        errors = []
        for person_index, first in enumerate(identities):
            for second in identities[person_index + 1:]:
                post_vector = proposed_roots[first] - proposed_roots[second]
                pre_vector = next(x for x in prepared if x["identity"] == first)["pre"]["root"] - next(
                    x for x in prepared if x["identity"] == second
                )["pre"]["root"]
                errors.append(float(np.linalg.norm(post_vector - pre_vector)))
        lambda_objectives[residual_lambda] = float(np.mean(errors)) if errors else 0.0
    selected_lambda = min(lambda_grid, key=lambda value: lambda_objectives[value])

    people = []
    roots_by_method: dict[str, dict[str, np.ndarray]] = {
        "baseline": {}, "individual": {}, "corrected": {}
    }
    for item in prepared:
        individual_shift = item["individual_shift"] if item["accepted"] else np.zeros(3)
        consensus_shift = (
            group_shift + selected_lambda * (item["individual_shift"] - group_shift)
            if item["accepted"] else np.zeros(3, dtype=np.float64)
        )
        individual = {key: value + individual_shift for key, value in item["post"].items()}
        corrected = {key: value + consensus_shift for key, value in item["post"].items()}
        baseline_error = point_errors(item["post"], item["target"])
        individual_error = point_errors(individual, item["target"])
        corrected_error = point_errors(corrected, item["target"])
        people.append({
            "identity": item["identity"],
            "predicted_post_identity_evaluation_only": item[
                "predicted_post_identity_evaluation_only"
            ],
            "association_correct_evaluation_only": item[
                "association_correct_evaluation_only"
            ],
            "accepted": item["accepted"],
            "action_m": item["action"],
            "individual_shift_world": individual_shift,
            "consensus_shift_world": consensus_shift,
            "oracle_ray_label_m_evaluation_only": item["oracle_label"],
            "sign_correct_evaluation_only": bool(
                np.sign(item["action"]) == np.sign(item["oracle_label"])
            ),
            "evidence": {
                key: value for key, value in item["evidence"].items() if key != "ray"
            },
            "baseline": baseline_error,
            "individual": individual_error,
            "corrected": corrected_error,
            "root_delta_m": corrected_error["root_error_m"] - baseline_error["root_error_m"],
            "individual_root_delta_m": (
                individual_error["root_error_m"] - baseline_error["root_error_m"]
            ),
        })
        roots_by_method["baseline"][item["identity"]] = item["post"]["root"]
        roots_by_method["individual"][item["identity"]] = individual["root"]
        roots_by_method["corrected"][item["identity"]] = corrected["root"]

    def layout(roots: dict[str, np.ndarray]) -> dict[str, float]:
        distance, vector = [], []
        for index, first in enumerate(identities):
            for second in identities[index + 1:]:
                predicted = roots[first] - roots[second]
                target = target_roots[first] - target_roots[second]
                distance.append(abs(float(np.linalg.norm(predicted) - np.linalg.norm(target))))
                vector.append(float(np.linalg.norm(predicted - target)))
        return {
            "pairwise_distance_error_m": float(np.mean(distance)) if distance else float("nan"),
            "pairwise_vector_error_m": float(np.mean(vector)) if vector else float("nan"),
        }

    camera_delta = float(np.max(np.abs(post_camera - boundary @ raw_post_camera)))
    return {
        "sequence": sequence,
        "case": row["case"],
        "camera_span_deg": float(row["camera_span_deg"]),
        "association_source": association_source,
        "camera": {
            "b0_translation_error_m": float(np.linalg.norm(post_camera[:3, 3] - target_camera[:3, 3])),
            "b0_rotation_error_deg": b0_rotation_error,
            "candidate_max_abs_change": camera_delta,
        },
        "people": people,
        "layout_consensus": {
            "group_shift_world": group_shift,
            "selected_residual_lambda": selected_lambda,
            "observable_pre_layout_objective_by_lambda": {
                str(key): value for key, value in lambda_objectives.items()
            },
        },
        "layout": {name: layout(roots) for name, roots in roots_by_method.items()},
    }


def summarize(cases: list[dict]) -> dict[str, Any]:
    people = [person for case in cases for person in case["people"]]
    output: dict[str, Any] = {
        "case_count": len(cases),
        "person_count": len(people),
        "coverage": float(np.mean([person["accepted"] for person in people])) if people else float("nan"),
        "accepted_sign_accuracy": float(np.mean([
            person["sign_correct_evaluation_only"] for person in people if person["accepted"]
        ])) if any(person["accepted"] for person in people) else float("nan"),
        "association_accuracy": float(np.mean([
            person["association_correct_evaluation_only"] for person in people
        ])) if people else float("nan"),
        "root_improve_rate": float(np.mean([person["root_delta_m"] < 0.0 for person in people])) if people else float("nan"),
        "root_harm_over_5cm_rate": float(np.mean([person["root_delta_m"] > 0.05 for person in people])) if people else float("nan"),
        "root_mean_delta_m": float(np.mean([person["root_delta_m"] for person in people])) if people else float("nan"),
        "camera_candidate_max_abs_change": float(max(
            (case["camera"]["candidate_max_abs_change"] for case in cases), default=0.0
        )),
    }
    for name in ("baseline", "individual", "corrected"):
        output[name] = {
            metric: finite_stats([person[name][metric] for person in people]) for metric in METRICS
        }
        for metric in ("pairwise_distance_error_m", "pairwise_vector_error_m"):
            output[name][metric] = finite_stats([case["layout"][name][metric] for case in cases])
    baseline = output["baseline"]["root_error_m"]["mean"]
    corrected = output["corrected"]["root_error_m"]["mean"]
    layout_base = output["baseline"]["pairwise_vector_error_m"]["mean"]
    layout_corrected = output["corrected"]["pairwise_vector_error_m"]["mean"]
    output["root_relative_gain"] = float((baseline - corrected) / baseline)
    output["layout_vector_relative_gain"] = float((layout_base - layout_corrected) / layout_base)
    output["selected_residual_lambda"] = finite_stats([
        case["layout_consensus"]["selected_residual_lambda"] for case in cases
    ])
    return output


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Frozen-B0 Two-View Person Triangulation",
        "",
        "The camera is bit-exact frozen B0. GT identity isolates WHERE; GT geometry is evaluator-only.",
        "",
        "| Split | Cases | People | Coverage | Root B0 | Root independent | Root consensus | Root gain | Layout B0 | Layout consensus | Layout gain | Improve | Harm >5cm | Sign | Lambda |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, value in report["by_sequence"].items():
        lines.append(
            f"| {split} | {value['case_count']} | {value['person_count']} | {value['coverage']:.1%} | "
            f"{value['baseline']['root_error_m']['mean']:.4f} | {value['individual']['root_error_m']['mean']:.4f} | "
            f"{value['corrected']['root_error_m']['mean']:.4f} | "
            f"{value['root_relative_gain']:+.1%} | {value['baseline']['pairwise_vector_error_m']['mean']:.4f} | "
            f"{value['corrected']['pairwise_vector_error_m']['mean']:.4f} | {value['layout_vector_relative_gain']:+.1%} | "
            f"{value['root_improve_rate']:.1%} | {value['root_harm_over_5cm_rate']:.1%} | "
            f"{value['accepted_sign_accuracy']:.1%} | {value['selected_residual_lambda']['mean']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if not str(args.output_dir.resolve()).startswith(str(REPO_ROOT.resolve())):
        raise ValueError("Output must remain in Movie3R under /data")
    frozen = json.loads(args.policy.read_text(encoding="utf-8"))
    if not frozen.get("dev_pass", False):
        raise RuntimeError("Controlled policy did not pass development")
    policy = frozen["policy"]
    if policy["joint_set"] != "torso5":
        raise ValueError("This locked probe expects the frozen torso5 policy")
    if args.phase == "dev":
        sequences = ("three",)
        candidate_rows = report_rows(sequences)
    elif args.phase == "frozen":
        sequences = ("dance", "box")
        candidate_rows = report_rows(sequences)
    else:
        sequences = ("three",)
        candidate_rows = json.loads(args.confirm_report.read_text(encoding="utf-8"))["cases"]
    selected, counts = [], defaultdict(int)
    for row in candidate_rows:
        sequence = row.get("sequence", str(row["case"]["key"]).split("_", 1)[0])
        if args.max_cases_per_sequence and counts[sequence] >= args.max_cases_per_sequence:
            continue
        selected.append(row)
        counts[sequence] += 1
    cases = []
    for index, row in enumerate(selected, start=1):
        case = evaluate_case(row, policy)
        cases.append(case)
        print(
            f"[{index:03d}/{len(selected):03d}] {case['sequence']} {case['case']['key']} "
            f"people={len(case['people'])}", flush=True,
        )
    by_sequence = {
        sequence: summarize([case for case in cases if case["sequence"] == sequence])
        for sequence in sequences
    }
    report = {
        "experiment": "v14_b0_two_view_person_triangulation",
        "phase": args.phase,
        "protocol": {
            "camera": "frozen learned B0; numerical change must be zero",
            "causal_frames": "last pre-cut plus first post-cut only",
            "person_action": "one rigid post-person translation along its aligned pelvis ray",
            "candidate_gt_use": "none",
        "identity": (
            "automatic anonymous B0 root+torso+joints Hungarian when present in the "
            "input report; otherwise GT-ID WHERE isolation"
        ),
            "frozen_policy_source": str(args.policy),
        },
        "policy": policy,
        "summary": summarize(cases),
        "by_sequence": by_sequence,
        "cases": cases,
    }
    summary = report["summary"]
    report["pass"] = bool(
        summary["root_relative_gain"] >= 0.08
        and summary["layout_vector_relative_gain"] >= 0.05
        and summary["root_harm_over_5cm_rate"] <= 0.10
        and summary["coverage"] >= 0.20
        and summary["camera_candidate_max_abs_change"] <= 1e-12
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stems = {
        "dev": "dev_three",
        "frozen": "posthoc_dance_box_layout_consensus",
        "confirm": "confirm_three_offset1",
    }
    stem = stems[args.phase]
    (args.output_dir / f"{stem}.json").write_text(
        json.dumps(jsonable(report), indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / f"{stem}.md").write_text(markdown(report), encoding="utf-8")
    print(markdown(report), flush=True)
    print(json.dumps({"pass": report["pass"], "summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
