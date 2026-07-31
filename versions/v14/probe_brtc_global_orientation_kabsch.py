#!/usr/bin/env python3
"""Freeze and validate person-local global-orientation refinement after BRTC.

Frozen B0+BRTC corrects person translation but never corrects global body
orientation.  This probe estimates a bounded SO(3) rotation from corresponding
pre/post torso joints after B0 camera alignment, then rotates the post joints
and vertices rigidly around the already-corrected root.  Root, camera and
multi-person layout are therefore unchanged by construction.

The estimator reads only last-pre and current-post predicted skeletons.  GT is
used after prediction for development policy selection and held-out metrics.
No image model, future post frame, dataset identity, or source label is used.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts", REPO_ROOT / "versions/v14"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from versions.v14 import probe_b0_brtc_huber_irls as harness  # noqa: E402
from versions.v14 import probe_b0_two_view_person_triangulation as legacy  # noqa: E402
from versions.v14 import probe_brtc_observable_action_shrinkage as common  # noqa: E402


DEFAULT_OUTPUT = REPO_ROOT / "output/v14/fine_alignment_research/brtc_global_orientation_kabsch"
DEFAULT_POLICY = DEFAULT_OUTPUT / "FROZEN_POLICY_BEFORE_VALIDATION.json"
TORSO4 = (1, 2, 16, 17)
POINT_KEYS = ("root", "joints", "vertices")
PRIMARY = (
    "root_error_m",
    "joint_error_m",
    "vertex_error_m",
    "pairwise_distance_error_m",
    "pairwise_vector_error_m",
)


@dataclass(frozen=True)
class OrientationPolicy:
    max_angle_deg: float
    rotation_fraction: float
    min_observable_relative_improvement: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("dev", "freeze", "validate"))
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    return parser.parse_args()


def policy_grid() -> list[OrientationPolicy]:
    return [
        OrientationPolicy(angle, fraction, improvement)
        for angle in (2.0, 5.0, 10.0, 15.0, 25.0)
        for fraction in (0.5, 0.75, 1.0)
        for improvement in (0.0, 0.05, 0.10, 0.20)
    ]


def kabsch_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    covariance = source.T @ target
    left, _, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_t[-1] *= -1.0
        rotation = right_t.T @ left.T
    return rotation


def bounded_rotation(rotation: np.ndarray, policy: OrientationPolicy) -> tuple[np.ndarray, float]:
    vector = Rotation.from_matrix(np.asarray(rotation, dtype=np.float64)).as_rotvec()
    angle = float(np.linalg.norm(vector))
    if angle <= 1e-12:
        return np.eye(3, dtype=np.float64), 0.0
    maximum = math.radians(float(policy.max_angle_deg))
    applied = min(angle * float(policy.rotation_fraction), maximum)
    return Rotation.from_rotvec(vector * (applied / angle)).as_matrix(), math.degrees(applied)


def orientation_candidate(
    person: dict[str, Any],
    corrected: dict[str, np.ndarray],
    policy: OrientationPolicy,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    pre_root = np.asarray(person["pre"]["root"], dtype=np.float64)
    post_root = np.asarray(person["post"]["root"], dtype=np.float64)
    joint_count = min(len(person["pre"]["joints"]), len(person["post"]["joints"]))
    ids = tuple(index for index in TORSO4 if index < joint_count)
    if len(ids) < 3:
        return corrected, {"applied": False, "reason": "insufficient_torso_joints"}
    pre = np.asarray(person["pre"]["joints"], dtype=np.float64)[list(ids)] - pre_root
    post = np.asarray(person["post"]["joints"], dtype=np.float64)[list(ids)] - post_root
    before = float(np.linalg.norm(post - pre, axis=1).mean())
    raw_rotation = kabsch_rotation(post, pre)
    candidate_rotation, angle_deg = bounded_rotation(raw_rotation, policy)
    after = float(np.linalg.norm(post @ candidate_rotation.T - pre, axis=1).mean())
    relative = float((before - after) / max(before, 1e-12))
    apply = bool(
        angle_deg > 1e-8
        and after < before
        and relative >= float(policy.min_observable_relative_improvement)
    )
    if not apply:
        return corrected, {
            "applied": False,
            "reason": "observable_gate",
            "raw_residual_m": before,
            "candidate_residual_m": after,
            "observable_relative_improvement": relative,
            "applied_angle_deg": angle_deg,
        }
    output = {key: np.asarray(value, dtype=np.float64).copy() for key, value in corrected.items()}
    root = output["root"]
    for key in ("joints", "vertices"):
        output[key] = (output[key] - root) @ candidate_rotation.T + root
    return output, {
        "applied": True,
        "raw_residual_m": before,
        "candidate_residual_m": after,
        "observable_relative_improvement": relative,
        "applied_angle_deg": angle_deg,
        "rotation_world": candidate_rotation,
    }


def point_errors(predicted: dict[str, np.ndarray], target: dict[str, np.ndarray]) -> dict[str, float]:
    return harness.point_errors(predicted, target, full=True)


def layout(people: list[dict[str, np.ndarray]], targets: list[dict[str, np.ndarray]]) -> dict[str, float]:
    distance, vector = [], []
    for first in range(len(people)):
        for second in range(first + 1, len(people)):
            predicted = people[first]["root"] - people[second]["root"]
            target = targets[first]["root"] - targets[second]["root"]
            distance.append(abs(float(np.linalg.norm(predicted) - np.linalg.norm(target))))
            vector.append(float(np.linalg.norm(predicted - target)))
    return {
        "pairwise_distance_error_m": float(np.mean(distance)) if distance else float("nan"),
        "pairwise_vector_error_m": float(np.mean(vector)) if vector else float("nan"),
    }


def evaluate_cases(
    prepared: list[dict[str, Any]],
    policy: OrientationPolicy | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frozen = harness.legacy_policy()
    cases = []
    for case in prepared:
        proposals = []
        for person in case["people"]:
            shift, accepted, evidence, action = harness.legacy_proposal(person, frozen)
            proposals.append({
                "individual_shift": shift,
                "accepted": accepted,
                "evidence": evidence,
                "action_m": action,
            })
        final_shifts, consensus = harness.observable_layout_consensus(case, proposals)
        baseline_people, brtc_people, candidate_people, targets = [], [], [], []
        person_rows = []
        for person, proposal, shift in zip(case["people"], proposals, final_shifts):
            baseline = {key: np.asarray(person["post"][key], dtype=np.float64) for key in POINT_KEYS}
            brtc = {key: value + np.asarray(shift, dtype=np.float64) for key, value in baseline.items()}
            if policy is None:
                candidate, debug = brtc, {"applied": False, "reason": "v1"}
            elif not bool(proposal["accepted"]):
                candidate, debug = brtc, {
                    "applied": False,
                    "reason": "rejected_exact_b0_fallback",
                }
            else:
                candidate, debug = orientation_candidate(person, brtc, policy)
            target = {key: np.asarray(person["target"][key], dtype=np.float64) for key in POINT_KEYS}
            baseline_error = point_errors(baseline, target)
            brtc_error = point_errors(brtc, target)
            candidate_error = point_errors(candidate, target)
            person_rows.append({
                "identity_evaluation_only": person["identity"],
                "accepted": bool(proposal["accepted"]),
                "baseline": baseline_error,
                "brtc": brtc_error,
                "candidate": candidate_error,
                "root_delta_m": candidate_error["root_error_m"] - baseline_error["root_error_m"],
                "orientation": debug,
            })
            baseline_people.append(baseline)
            brtc_people.append(brtc)
            candidate_people.append(candidate)
            targets.append(target)
        cases.append({
            "sequence": case["sequence"],
            "case": case["case"],
            "camera_candidate_max_abs_change": case["camera"]["candidate_max_abs_change"],
            "people": person_rows,
            "layout": {
                "baseline": layout(baseline_people, targets),
                "brtc": layout(brtc_people, targets),
                "candidate": layout(candidate_people, targets),
            },
            "consensus": consensus,
        })
    return cases, summarize(cases)


def finite_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if len(array) else float("nan")


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    people = [person for case in cases for person in case["people"]]
    output = {
        "case_count": len(cases),
        "person_count": len(people),
        "coverage": finite_mean([float(row["accepted"]) for row in people]),
        "orientation_applied_rate": finite_mean([
            float(row["orientation"]["applied"]) for row in people
        ]),
        "root_harm_over_5cm_rate": finite_mean([
            float(row["root_delta_m"] > 0.05) for row in people
        ]),
        "camera_max_abs_change": float(max(
            (case["camera_candidate_max_abs_change"] for case in cases), default=0.0
        )),
    }
    for method in ("baseline", "brtc", "candidate"):
        error_key = "candidate" if method == "candidate" else method
        output[method] = {
            metric: finite_mean([row[error_key][metric] for row in people])
            for metric in ("root_error_m", "joint_error_m", "vertex_error_m")
        } | {
            metric: finite_mean([case["layout"][method][metric] for case in cases])
            for metric in ("pairwise_distance_error_m", "pairwise_vector_error_m")
        }
    return output


def metrics(summary: dict[str, Any], method: str) -> dict[str, float]:
    return {key: float(summary[method][key]) for key in PRIMARY} | {
        "coverage": float(summary["coverage"]),
        "orientation_applied_rate": float(summary["orientation_applied_rate"]),
        "root_harm_over_5cm_rate": float(summary["root_harm_over_5cm_rate"]),
        "camera_max_abs_change": float(summary["camera_max_abs_change"]),
    }


def safe(candidate: dict[str, float], reference: dict[str, float]) -> bool:
    return bool(
        all(candidate[key] <= reference[key] + 1e-12 for key in PRIMARY)
        and candidate["root_harm_over_5cm_rate"] <= reference["root_harm_over_5cm_rate"] + 1e-12
        and candidate["coverage"] >= reference["coverage"] - 1e-12
        and candidate["camera_max_abs_change"] <= 1e-12
    )


def prepared_splits(phase: str) -> dict[str, list[dict[str, Any]]]:
    if phase == "dev":
        rows = {"three_offset0": legacy.report_rows(("three",))}
    else:
        rows = {
            "three_offset1": json.loads(
                harness.DEFAULT_CONFIRM_REPORT.read_text(encoding="utf-8")
            )["cases"],
            "dance": legacy.report_rows(("dance",)),
            "box": legacy.report_rows(("box",)),
        }
    return {name: harness.prepare_all(value) for name, value in rows.items()}


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# BRTC + bounded global-orientation Kabsch",
        "",
        f"Phase: `{report['phase']}`.",
        "",
        "| Split | Method | Root | Joint | Vertex | Pair distance | Pair vector | Applied | Harm >5cm |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, value in report["splits"].items():
        for method in ("brtc", "candidate"):
            row = value[method]
            lines.append(
                f"| {split} | {method} | {row['root_error_m']:.6f} | "
                f"{row['joint_error_m']:.6f} | {row['vertex_error_m']:.6f} | "
                f"{row['pairwise_distance_error_m']:.6f} | "
                f"{row['pairwise_vector_error_m']:.6f} | "
                f"{row['orientation_applied_rate']:.1%} | "
                f"{row['root_harm_over_5cm_rate']:.1%} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if not str(args.output_dir.resolve()).startswith(str(REPO_ROOT.resolve())):
        raise ValueError("Output must stay in Movie3R under /data")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    original_load = torch.load

    @lru_cache(maxsize=None)
    def cached_load(path: str):
        return original_load(path, map_location="cpu", weights_only=False)

    def adapter(path, *unused_args, **unused_kwargs):
        return cached_load(str(path))

    torch.load = adapter
    try:
        if args.phase == "dev":
            prepared = prepared_splits("dev")["three_offset0"]
            _, v1_summary = evaluate_cases(prepared, None)
            reference = metrics(v1_summary, "brtc")
            scan = []
            selected_cases = selected_summary = None
            for policy in policy_grid():
                cases, summary = evaluate_cases(prepared, policy)
                value = metrics(summary, "candidate")
                scan.append({"policy": asdict(policy), "metrics": value, "safe": safe(value, reference)})
            eligible = [row for row in scan if row["safe"] and row["metrics"]["orientation_applied_rate"] > 0.0]
            if eligible:
                selected = min(
                    eligible,
                    key=lambda row: (
                        row["metrics"]["joint_error_m"] + row["metrics"]["vertex_error_m"],
                        row["policy"]["max_angle_deg"],
                    ),
                )
            else:
                selected = {"policy": asdict(OrientationPolicy(2.0, 0.5, 1.0)), "metrics": reference, "safe": False}
            selected_policy = OrientationPolicy(**selected["policy"])
            selected_cases, selected_summary = evaluate_cases(prepared, selected_policy)
            report = {
                "experiment": "v14_brtc_global_orientation_kabsch",
                "phase": "dev",
                "selection": {
                    "rule": "all five primary/harm/coverage safe vs v1, then minimum joint+vertex",
                    "eligible_count": len(eligible),
                    "selected_policy": asdict(selected_policy),
                    "development_pass": bool(eligible),
                    "scan": scan,
                },
                "splits": {
                    "three_offset0": {
                        "brtc": reference,
                        "candidate": metrics(selected_summary, "candidate"),
                    }
                },
                "cases": selected_cases,
            }
            stem = "DEV_SCAN"
        elif args.phase == "freeze":
            dev = json.loads((args.output_dir / "DEV_SCAN.json").read_text(encoding="utf-8"))
            if not dev["selection"]["development_pass"]:
                raise RuntimeError("Orientation candidate did not pass development")
            frozen = {
                "experiment": "v14_brtc_global_orientation_kabsch",
                "frozen_before_validation": True,
                "policy": dev["selection"]["selected_policy"],
                "constraints": {
                    "future_frames": 0,
                    "extra_pretrained_models": [],
                    "camera_update": "none",
                    "root_update": "none beyond frozen BRTC",
                },
            }
            frozen["policy_sha256"] = common.canonical_sha256(frozen["policy"])
            args.policy.write_text(
                json.dumps(common.jsonable(frozen), indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(args.policy)
            return
        else:
            frozen = json.loads(args.policy.read_text(encoding="utf-8"))
            policy = OrientationPolicy(**frozen["policy"])
            splits = {}
            all_safe = True
            for name, prepared in prepared_splits("validate").items():
                _, v1_summary = evaluate_cases(prepared, None)
                cases, candidate_summary = evaluate_cases(prepared, policy)
                reference = metrics(v1_summary, "brtc")
                candidate = metrics(candidate_summary, "candidate")
                split_safe = safe(candidate, reference)
                all_safe = all_safe and split_safe
                splits[name] = {
                    "brtc": reference,
                    "candidate": candidate,
                    "safe": split_safe,
                    "cases": cases,
                }
            report = {
                "experiment": "v14_brtc_global_orientation_kabsch",
                "phase": "validate",
                "policy": asdict(policy),
                "splits": splits,
                "decision": {
                    "all_splits_safe": all_safe,
                    "status": "GO_GLOBAL_ORIENTATION_KABSCH" if all_safe else "NO_GO_GLOBAL_ORIENTATION_KABSCH",
                },
            }
            stem = "VALIDATION_RESULTS"
    finally:
        torch.load = original_load
    (args.output_dir / f"{stem}.json").write_text(
        json.dumps(common.jsonable(report), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    text = markdown(report)
    (args.output_dir / f"{stem}.md").write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
