#!/usr/bin/env python3
"""Develop and freeze reliability-weighted Huber-IRLS multi-ray BRTC.

Development uses the existing offset-0 ``three`` B0 cache set.  Confirmation
uses the offset-1 automatic-association set only after a policy JSON has been
written.  Both phases keep the learned B0 camera bit-exact, use only the last
pre-cut/current post-cut prediction, and make rejected people exact no-ops.

Ground-truth people are loaded only after a proposal and gate have been formed,
and are used solely to report world root/joint/vertex and pair-layout errors.
No pretrained model or image feature extractor is loaded by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from versions.v14.b0_brtc_huber_irls import (  # noqa: E402
    CORE5,
    ReliabilityHuberConfig,
    accepted_shift,
    aggregate_candidates,
    config_dict,
    ray_candidates,
)
from versions.v14.probe_b0_anchor_conflict import SEQUENCE_INPUTS  # noqa: E402
from versions.v14 import probe_b0_two_view_person_triangulation as legacy  # noqa: E402


DEFAULT_OUTPUT = (
    REPO_ROOT / "output/v14/fine_alignment_research/b0_brtc_huber_irls"
)
DEFAULT_POLICY = DEFAULT_OUTPUT / "FROZEN_HUBER_IRLS_POLICY_BEFORE_CONFIRM.json"
DEFAULT_CONFIRM_REPORT = (
    REPO_ROOT / "output/v14/b0_identity_matching_offset1_confirm/"
    "v14_b0_identity_matching.json"
)
LEGACY_POLICY_PATH = (
    REPO_ROOT / "output/v14/fine_alignment_research/two_view_person_triangulation/"
    "FROZEN_TRIANGULATION_POLICY_BEFORE_CONFIRM.json"
)
FULL_METRICS = ("root_error_m", "joint_error_m", "vertex_error_m")
LAYOUT_METRICS = ("pairwise_distance_error_m", "pairwise_vector_error_m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("dev", "confirm", "compare"), required=True)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--confirm_report", type=Path, default=DEFAULT_CONFIRM_REPORT)
    parser.add_argument("--max_cases", type=int, default=0)
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value > 0 else "-inf"
    return value


def finite_stats(values: list[float]) -> dict[str, float]:
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


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return np.einsum("ij,...j->...i", transform[:3, :3], points) + transform[:3, 3]


def point_errors(
    predicted: dict[str, np.ndarray], target: dict[str, np.ndarray], full: bool
) -> dict[str, float]:
    result = {
        "root_error_m": float(np.linalg.norm(predicted["root"] - target["root"]))
    }
    if not full:
        return result
    joint_count = min(len(predicted["joints"]), len(target["joints"]))
    vertex_count = min(len(predicted["vertices"]), len(target["vertices"]))
    result.update(
        {
            "joint_error_m": float(
                np.linalg.norm(
                    predicted["joints"][:joint_count] - target["joints"][:joint_count],
                    axis=1,
                ).mean()
            ),
            "vertex_error_m": float(
                np.linalg.norm(
                    predicted["vertices"][:vertex_count]
                    - target["vertices"][:vertex_count],
                    axis=1,
                ).mean()
            ),
        }
    )
    return result


def load_rows(phase: str, confirm_report: Path, max_cases: int) -> list[dict]:
    if phase == "dev":
        rows = legacy.report_rows(("three",))
    else:
        rows = json.loads(confirm_report.read_text(encoding="utf-8"))["cases"]
    if max_cases:
        rows = rows[: int(max_cases)]
    return rows


def prepare_case(row: dict) -> dict[str, Any]:
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
        association_source = "automatic anonymous B0 root+torso+joints Hungarian"
    else:
        # Existing development protocol is a WHERE isolation.  The runtime
        # module accepts anonymous match indices and never consumes these names.
        association = {identity: identity for identity in cache["humans"][-1]}
        association_source = "GT-ID WHERE isolation (development only)"
    identities = tuple(
        identity
        for identity in sorted(association)
        if identity in cache["humans"][-2]
        and association[identity] in cache["humans"][-1]
        and identity in cache["gt"]["post_humans"]
    )
    people = []
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
        candidates = ray_candidates(pre, post, pre_camera, post_camera, CORE5)
        oracle_ray_label = float(
            np.dot(target["root"] - post["root"], candidates["ray_world"])
        )
        people.append(
            {
                "identity": identity,
                "predicted_post_identity_evaluation_only": post_identity,
                "association_correct_evaluation_only": bool(post_identity == identity),
                "pre": pre,
                "post": post,
                "target": target,
                "candidates": candidates,
                "oracle_ray_label_evaluation_only": oracle_ray_label,
            }
        )
    return {
        "sequence": sequence,
        "case": row["case"],
        "camera_span_deg": float(row["camera_span_deg"]),
        "association_source": association_source,
        "pre_camera": pre_camera,
        "post_camera": post_camera,
        "people": people,
        "camera": {
            "b0_translation_error_m": float(
                np.linalg.norm(post_camera[:3, 3] - target_camera[:3, 3])
            ),
            "b0_rotation_error_deg": b0_rotation_error,
            "candidate_max_abs_change": float(
                np.max(np.abs(post_camera - boundary @ raw_post_camera))
            ),
        },
    }


def legacy_evidence(candidates: dict[str, Any]) -> dict[str, Any]:
    values = np.asarray(candidates["candidate_depths_m"], dtype=np.float64)
    gaps = np.asarray(candidates["ray_gaps_m"], dtype=np.float64)
    sines = np.asarray(candidates["conditioning_sines"], dtype=np.float64)
    ray = np.asarray(candidates["ray_world"], dtype=np.float64)
    if not len(values):
        return {
            "raw": float("nan"),
            "valid_count": 0,
            "median_gap": float("inf"),
            "max_gap": float("inf"),
            "median_sine": 0.0,
            "min_sine": 0.0,
            "mad": float("inf"),
            "ray": ray,
        }
    center = float(np.median(values))
    return {
        "raw": center,
        "valid_count": len(values),
        "median_gap": float(np.median(gaps)),
        "max_gap": float(np.max(gaps)),
        "median_sine": float(np.median(sines)),
        "min_sine": float(np.min(sines)),
        "mad": float(np.median(np.abs(values - center))),
        "ray": ray,
    }


def legacy_proposal(
    person: dict[str, Any], policy: dict[str, Any]
) -> tuple[np.ndarray, bool, dict[str, Any], float]:
    evidence = legacy_evidence(person["candidates"])
    action, accepted = legacy.accepted_action(evidence, policy)
    shift = action * evidence["ray"] if accepted else np.zeros(3, dtype=np.float64)
    return shift, accepted, evidence, action


def candidate_proposal(
    person: dict[str, Any], config: ReliabilityHuberConfig
) -> tuple[np.ndarray, bool, dict[str, Any], float]:
    evidence = aggregate_candidates(person["candidates"], config)
    shift, accepted = accepted_shift(evidence, config)
    action = float(np.dot(shift, evidence["ray_world"])) if accepted else 0.0
    return shift, accepted, evidence, action


def damped_legacy_proposal(
    person: dict[str, Any], policy: dict[str, Any], scale: float = 0.8
) -> tuple[np.ndarray, bool, dict[str, Any], float]:
    """Fixed causal shrinkage comparator; gate/fallback are unchanged."""
    shift, accepted, evidence, action = legacy_proposal(person, policy)
    return float(scale) * shift, accepted, evidence, float(scale) * action


def damped_candidate_proposal(
    person: dict[str, Any], config: ReliabilityHuberConfig, scale: float = 0.8
) -> tuple[np.ndarray, bool, dict[str, Any], float]:
    """Exploratory composition of the frozen robust center and fixed damping."""
    shift, accepted, evidence, action = candidate_proposal(person, config)
    return float(scale) * shift, accepted, evidence, float(scale) * action


def observable_layout_consensus(
    case: dict[str, Any], proposals: list[dict[str, Any]]
) -> tuple[list[np.ndarray], dict[str, Any]]:
    accepted_shifts = [row["individual_shift"] for row in proposals if row["accepted"]]
    group_shift = (
        np.median(np.stack(accepted_shifts), axis=0)
        if accepted_shifts
        else np.zeros(3, dtype=np.float64)
    )
    lambda_grid = (0.0, 0.25, 0.50, 0.75, 1.0)
    objectives = {}
    for residual_lambda in lambda_grid:
        roots = []
        for person, proposal in zip(case["people"], proposals):
            shift = (
                group_shift
                + residual_lambda * (proposal["individual_shift"] - group_shift)
                if proposal["accepted"]
                else np.zeros(3, dtype=np.float64)
            )
            roots.append(person["post"]["root"] + shift)
        errors = []
        for first_index, first in enumerate(case["people"]):
            for second_index in range(first_index + 1, len(case["people"])):
                post_vector = roots[first_index] - roots[second_index]
                pre_vector = (
                    first["pre"]["root"] - case["people"][second_index]["pre"]["root"]
                )
                errors.append(float(np.linalg.norm(post_vector - pre_vector)))
        objectives[residual_lambda] = float(np.mean(errors)) if errors else 0.0
    selected_lambda = min(lambda_grid, key=lambda value: objectives[value])
    shifts = [
        (
            group_shift
            + selected_lambda * (proposal["individual_shift"] - group_shift)
            if proposal["accepted"]
            else np.zeros(3, dtype=np.float64)
        )
        for proposal in proposals
    ]
    return shifts, {
        "group_shift_world": group_shift,
        "selected_residual_lambda": selected_lambda,
        "observable_pre_layout_objective_by_lambda": objectives,
    }


def case_layout(
    case: dict[str, Any], predicted_roots: list[np.ndarray]
) -> dict[str, float]:
    distance, vector = [], []
    for first_index, first in enumerate(case["people"]):
        for second_index in range(first_index + 1, len(case["people"])):
            predicted = predicted_roots[first_index] - predicted_roots[second_index]
            target = (
                first["target"]["root"]
                - case["people"][second_index]["target"]["root"]
            )
            distance.append(abs(float(np.linalg.norm(predicted) - np.linalg.norm(target))))
            vector.append(float(np.linalg.norm(predicted - target)))
    return {
        "pairwise_distance_error_m": float(np.mean(distance)) if distance else float("nan"),
        "pairwise_vector_error_m": float(np.mean(vector)) if vector else float("nan"),
    }


def evaluate_method(
    prepared_cases: list[dict[str, Any]],
    proposal_fn: Callable[[dict[str, Any]], tuple[np.ndarray, bool, dict[str, Any], float]],
    method_name: str,
    full: bool,
) -> list[dict[str, Any]]:
    evaluated = []
    for case in prepared_cases:
        proposals = []
        for person in case["people"]:
            shift, accepted, evidence, action = proposal_fn(person)
            proposals.append(
                {
                    "individual_shift": np.asarray(shift, dtype=np.float64),
                    "accepted": bool(accepted),
                    "evidence": evidence,
                    "action_m": float(action),
                }
            )
        final_shifts, layout_consensus = observable_layout_consensus(case, proposals)
        people = []
        baseline_roots, corrected_roots = [], []
        fallback_max_abs_change = 0.0
        for person, proposal, final_shift in zip(case["people"], proposals, final_shifts):
            corrected = {
                key: value + final_shift for key, value in person["post"].items()
            }
            baseline_error = point_errors(person["post"], person["target"], full)
            corrected_error = point_errors(corrected, person["target"], full)
            root_delta = corrected_error["root_error_m"] - baseline_error["root_error_m"]
            if not proposal["accepted"]:
                fallback_max_abs_change = max(
                    fallback_max_abs_change, float(np.max(np.abs(final_shift)))
                )
            evidence = proposal["evidence"]
            if method_name == "legacy_brtc_lc":
                evidence_output = {key: value for key, value in evidence.items() if key != "ray"}
            else:
                evidence_output = {
                    key: value for key, value in evidence.items() if key != "ray_world"
                }
                evidence_output.update(
                    {
                        "candidate_depths_m": person["candidates"]["candidate_depths_m"],
                        "ray_gaps_m": person["candidates"]["ray_gaps_m"],
                        "conditioning_sines": person["candidates"]["conditioning_sines"],
                        "joint_ids": person["candidates"]["joint_ids"],
                    }
                )
            people.append(
                {
                    "identity_evaluation_only": person["identity"],
                    "predicted_post_identity_evaluation_only": person[
                        "predicted_post_identity_evaluation_only"
                    ],
                    "association_correct_evaluation_only": person[
                        "association_correct_evaluation_only"
                    ],
                    "accepted": proposal["accepted"],
                    "action_m": proposal["action_m"],
                    "individual_shift_world": proposal["individual_shift"],
                    "final_shift_world": final_shift,
                    "oracle_ray_label_m_evaluation_only": person[
                        "oracle_ray_label_evaluation_only"
                    ],
                    "sign_correct_evaluation_only": bool(
                        np.sign(proposal["action_m"])
                        == np.sign(person["oracle_ray_label_evaluation_only"])
                    ),
                    "evidence": evidence_output,
                    "baseline": baseline_error,
                    "corrected": corrected_error,
                    "root_delta_m": root_delta,
                }
            )
            baseline_roots.append(person["post"]["root"])
            corrected_roots.append(corrected["root"])
        evaluated.append(
            {
                "sequence": case["sequence"],
                "case": case["case"],
                "camera_span_deg": case["camera_span_deg"],
                "association_source": case["association_source"],
                "camera": case["camera"],
                "people": people,
                "layout_consensus": layout_consensus,
                "layout": {
                    "baseline": case_layout(case, baseline_roots),
                    "corrected": case_layout(case, corrected_roots),
                },
                "exact_b0_fallback_max_abs_change": fallback_max_abs_change,
            }
        )
    return evaluated


def summarize(cases: list[dict[str, Any]], full: bool) -> dict[str, Any]:
    people = [person for case in cases for person in case["people"]]
    deltas = [float(person["root_delta_m"]) for person in people]
    accepted = [person for person in people if person["accepted"]]
    metrics = FULL_METRICS if full else ("root_error_m",)
    output: dict[str, Any] = {
        "case_count": len(cases),
        "person_count": len(people),
        "coverage": float(np.mean([person["accepted"] for person in people])),
        "accepted_count": len(accepted),
        "accepted_sign_accuracy": (
            float(np.mean([person["sign_correct_evaluation_only"] for person in accepted]))
            if accepted
            else float("nan")
        ),
        "association_accuracy": float(
            np.mean([person["association_correct_evaluation_only"] for person in people])
        ),
        "root_improve_rate": float(np.mean([value < 0.0 for value in deltas])),
        "root_harm_any_rate": float(np.mean([value > 0.0 for value in deltas])),
        "root_harm_over_1cm_rate": float(np.mean([value > 0.01 for value in deltas])),
        "root_harm_over_5cm_rate": float(np.mean([value > 0.05 for value in deltas])),
        "root_mean_delta_m": float(np.mean(deltas)),
        "fallback_max_abs_change": float(
            max((case["exact_b0_fallback_max_abs_change"] for case in cases), default=0.0)
        ),
        "camera_candidate_max_abs_change": float(
            max((case["camera"]["candidate_max_abs_change"] for case in cases), default=0.0)
        ),
        "selected_residual_lambda": finite_stats(
            [float(case["layout_consensus"]["selected_residual_lambda"]) for case in cases]
        ),
        "accepted_action_abs_m": finite_stats(
            [abs(float(person["action_m"])) for person in accepted]
        ),
        "accepted_by_valid_ray_count": dict(
            sorted(
                Counter(
                    int(person["evidence"]["valid_count"])
                    for person in accepted
                ).items()
            )
        ),
    }
    output["baseline"] = {
        metric: finite_stats([float(person["baseline"][metric]) for person in people])
        for metric in metrics
    }
    output["corrected"] = {
        metric: finite_stats([float(person["corrected"][metric]) for person in people])
        for metric in metrics
    }
    for method in ("baseline", "corrected"):
        for metric in LAYOUT_METRICS:
            output[method][metric] = finite_stats(
                [float(case["layout"][method][metric]) for case in cases]
            )
    baseline_root = output["baseline"]["root_error_m"]["mean"]
    corrected_root = output["corrected"]["root_error_m"]["mean"]
    output["root_relative_gain"] = float(
        (baseline_root - corrected_root) / baseline_root
    )
    for metric in LAYOUT_METRICS:
        baseline_value = output["baseline"][metric]["mean"]
        corrected_value = output["corrected"][metric]["mean"]
        output[f"{metric}_relative_gain"] = float(
            (baseline_value - corrected_value) / baseline_value
        )
    return output


def candidate_grid(base: ReliabilityHuberConfig) -> list[ReliabilityHuberConfig]:
    configs = []
    seen = set()
    for sine_power in (0.0, 1.0, 2.0):
        for gap_power in (0.0, 1.0, 2.0):
            gap_scales = (0.10,) if gap_power == 0.0 else (0.05, 0.10, 0.20)
            for gap_scale in gap_scales:
                for huber_delta in (0.025, 0.05, 0.10, 0.20, float("inf")):
                    for min_effective in (1.0, 1.5, 2.0):
                        for max_mad in (0.10, 0.20, 0.40):
                            for min_inlier in (0.0, 0.50):
                                config = replace(
                                    base,
                                    sine_power=sine_power,
                                    gap_power=gap_power,
                                    gap_scale_m=gap_scale,
                                    huber_delta_m=huber_delta,
                                    min_effective_rays=min_effective,
                                    max_weighted_mad_m=max_mad,
                                    min_huber_inlier_weight=min_inlier,
                                )
                                key = json.dumps(jsonable(config_dict(config)), sort_keys=True)
                                if key not in seen:
                                    seen.add(key)
                                    configs.append(config)
    return configs


def policy_from_json(values: dict[str, Any]) -> ReliabilityHuberConfig:
    restored = dict(values)
    restored["joint_ids"] = tuple(int(value) for value in restored["joint_ids"])
    restored["residual_lambda_grid"] = tuple(
        float(value) for value in restored["residual_lambda_grid"]
    )
    for key in ("huber_delta_m",):
        if restored[key] == "inf":
            restored[key] = float("inf")
    return ReliabilityHuberConfig(**restored)


def comparison_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Frozen-B0 reliability-weighted Huber-IRLS BRTC",
        "",
        "All geometry metrics are in the shared evaluation world gauge. Cameras are exact frozen B0.",
        "",
        "| Method | Cuts | People | Coverage | Root | Joint | Vertex | Pair distance | Pair vector | Improve | Harm >1cm | Harm >5cm |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = (
        "legacy_brtc_lc",
        "legacy_brtc_lc_damped_0p8",
        "huber_irls_brtc_lc",
        "huber_irls_brtc_lc_damped_0p8_exploratory",
    )
    for name in (value for value in order if value in report["methods"]):
        value = report["methods"][name]["summary"]
        corrected = value["corrected"]
        lines.append(
            f"| {name} | {value['case_count']} | {value['person_count']} | "
            f"{value['coverage']:.1%} | {corrected['root_error_m']['mean']:.4f} | "
            f"{corrected.get('joint_error_m', {}).get('mean', float('nan')):.4f} | "
            f"{corrected.get('vertex_error_m', {}).get('mean', float('nan')):.4f} | "
            f"{corrected['pairwise_distance_error_m']['mean']:.4f} | "
            f"{corrected['pairwise_vector_error_m']['mean']:.4f} | "
            f"{value['root_improve_rate']:.1%} | {value['root_harm_over_1cm_rate']:.1%} | "
            f"{value['root_harm_over_5cm_rate']:.1%} |"
        )
    candidate = report["methods"]["huber_irls_brtc_lc"]["summary"]
    lines.extend(
        [
            "",
            f"- Exact-B0 rejected fallback max change: `{candidate['fallback_max_abs_change']:.3e}`.",
            f"- Camera candidate max change: `{candidate['camera_candidate_max_abs_change']:.3e}`.",
            f"- Frozen policy: `{report['policy_source']}`.",
        ]
    )
    return "\n".join(lines) + "\n"


def strong_comparison_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Huber-IRLS BRTC versus fixed-0.8 damped BRTC-LC",
        "",
        "The fixed 0.8 scale is an independently supplied strong comparator and did not alter the already frozen Huber-IRLS policy.",
        "",
    ]
    split_order = ("dev_offset0", "confirm_offset1", "dance_box_frozen_external")
    for split in (value for value in split_order if value in report["splits"]):
        lines.extend(
            [
                f"## {split}",
                "",
                "| Method | Coverage | Root | Joint | Vertex | Pair distance | Pair vector | Harm >1cm | Harm >5cm |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name in (
            "legacy_brtc_lc",
            "legacy_brtc_lc_damped_0p8",
            "huber_irls_brtc_lc",
            "huber_irls_brtc_lc_damped_0p8_exploratory",
        ):
            value = report["splits"][split]["methods"][name]["summary"]
            corrected = value["corrected"]
            lines.append(
                f"| {name} | {value['coverage']:.1%} | "
                f"{corrected['root_error_m']['mean']:.6f} | "
                f"{corrected['joint_error_m']['mean']:.6f} | "
                f"{corrected['vertex_error_m']['mean']:.6f} | "
                f"{corrected['pairwise_distance_error_m']['mean']:.6f} | "
                f"{corrected['pairwise_vector_error_m']['mean']:.6f} | "
                f"{value['root_harm_over_1cm_rate']:.1%} | "
                f"{value['root_harm_over_5cm_rate']:.1%} |"
            )
        lines.append("")
    lines.extend(
        [
            f"- Frozen robust candidate beats strong baseline: `{report['robust_beats_damped_strong_baseline']}`.",
            f"- Decision: **{report['decision']}**.",
            "- Cameras remain bit-exact B0 and every rejected person remains an exact no-op.",
        ]
    )
    return "\n".join(lines) + "\n"


def prepare_all(rows: list[dict]) -> list[dict[str, Any]]:
    cases = []
    for index, row in enumerate(rows, start=1):
        case = prepare_case(row)
        cases.append(case)
        print(
            f"[{index:03d}/{len(rows):03d}] prepared {case['case']['key']} "
            f"people={len(case['people'])}",
            flush=True,
        )
    return cases


def legacy_policy() -> dict[str, Any]:
    frozen = json.loads(LEGACY_POLICY_PATH.read_text(encoding="utf-8"))
    return frozen["policy"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def run_dev(args: argparse.Namespace) -> None:
    rows = load_rows("dev", args.confirm_report, args.max_cases)
    prepared = prepare_all(rows)
    old_policy = legacy_policy()
    legacy_cases = evaluate_method(
        prepared,
        lambda person: legacy_proposal(person, old_policy),
        "legacy_brtc_lc",
        full=False,
    )
    legacy_summary = summarize(legacy_cases, full=False)

    base = ReliabilityHuberConfig(
        joint_ids=CORE5,
        min_valid=int(old_policy["min_valid"]),
        max_median_gap_m=float(old_policy["max_median_gap_m"]),
        max_weighted_mad_m=float(old_policy["max_mad_m"]),
        min_median_sine=float(old_policy["min_median_sine"]),
        min_abs_raw_m=float(old_policy["min_abs_raw_m"]),
        cap_m=float(old_policy["cap_m"]),
    )
    grid = candidate_grid(base)
    scan = []
    for index, config in enumerate(grid, start=1):
        cases = evaluate_method(
            prepared,
            lambda person, cfg=config: candidate_proposal(person, cfg),
            "huber_irls_brtc_lc",
            full=False,
        )
        summary = summarize(cases, full=False)
        mechanism_active = bool(
            (config.sine_power > 0.0 or config.gap_power > 0.0)
            and np.isfinite(config.huber_delta_m)
        )
        scan.append(
            {
                "config": config_dict(config),
                "mechanism_active": mechanism_active,
                "summary": summary,
            }
        )
        if index % 100 == 0 or index == len(grid):
            print(f"[{index:04d}/{len(grid):04d}] parameter scan", flush=True)

    thresholds = {
        "coverage_min": max(0.20, legacy_summary["coverage"] - 0.05),
        "root_harm_over_5cm_max": 0.10,
        "pairwise_vector_error_max_m": legacy_summary["corrected"][
            "pairwise_vector_error_m"
        ]["mean"]
        * 1.02,
        "pairwise_distance_error_max_m": legacy_summary["corrected"][
            "pairwise_distance_error_m"
        ]["mean"]
        * 1.02,
    }
    eligible = [
        row
        for row in scan
        if row["mechanism_active"]
        and row["summary"]["coverage"] >= thresholds["coverage_min"]
        and row["summary"]["root_harm_over_5cm_rate"]
        <= thresholds["root_harm_over_5cm_max"]
        and row["summary"]["corrected"]["pairwise_vector_error_m"]["mean"]
        <= thresholds["pairwise_vector_error_max_m"]
        and row["summary"]["corrected"]["pairwise_distance_error_m"]["mean"]
        <= thresholds["pairwise_distance_error_max_m"]
    ]
    if not eligible:
        raise RuntimeError("No Huber-IRLS configuration satisfies development safety")
    selected = min(
        eligible,
        key=lambda row: (
            row["summary"]["corrected"]["root_error_m"]["mean"],
            row["summary"]["root_harm_over_5cm_rate"],
            row["summary"]["corrected"]["pairwise_vector_error_m"]["mean"],
            -row["summary"]["coverage"],
            json.dumps(jsonable(row["config"]), sort_keys=True),
        ),
    )
    selected_config = policy_from_json(jsonable(selected["config"]))
    legacy_full_cases = evaluate_method(
        prepared,
        lambda person: legacy_proposal(person, old_policy),
        "legacy_brtc_lc",
        full=True,
    )
    candidate_full_cases = evaluate_method(
        prepared,
        lambda person: candidate_proposal(person, selected_config),
        "huber_irls_brtc_lc",
        full=True,
    )
    legacy_full = summarize(legacy_full_cases, full=True)
    candidate_full = summarize(candidate_full_cases, full=True)
    dev_pass = bool(
        candidate_full["coverage"] >= thresholds["coverage_min"]
        and candidate_full["root_harm_over_5cm_rate"]
        <= thresholds["root_harm_over_5cm_max"]
        and candidate_full["corrected"]["pairwise_vector_error_m"]["mean"]
        <= thresholds["pairwise_vector_error_max_m"]
        and candidate_full["corrected"]["pairwise_distance_error_m"]["mean"]
        <= thresholds["pairwise_distance_error_max_m"]
        and candidate_full["corrected"]["root_error_m"]["mean"]
        < legacy_full["corrected"]["root_error_m"]["mean"]
        and candidate_full["fallback_max_abs_change"] == 0.0
        and candidate_full["camera_candidate_max_abs_change"] == 0.0
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_path = output_dir / "FROZEN_HUBER_IRLS_POLICY_BEFORE_CONFIRM.json"
    frozen = {
        "protocol": {
            "selection_split": "three offset0 only (41 cuts / 122 people)",
            "confirmation_split": "three offset1; unopened by this dev phase",
            "causal_inputs": "last pre-cut and current first post-cut predicted core-joint rays",
            "camera": "frozen B0; no candidate camera update",
            "rejected_contract": "exact B0; zero person shift",
            "gt_candidate_or_gate_use": "none; evaluator only",
            "pretrained_model": "none",
        },
        "legacy_policy_source": str(LEGACY_POLICY_PATH),
        "parameter_count": len(scan),
        "eligibility_thresholds_frozen_before_confirm": thresholds,
        "selection_rule": (
            "among mechanism-active eligible configurations, minimize development "
            "world-root mean; then harm, pair-vector, negative coverage, config JSON"
        ),
        "policy": config_dict(selected_config),
        "dev_legacy_summary": legacy_full,
        "dev_selected_summary": candidate_full,
        "dev_pass": dev_pass,
    }
    policy_path.write_text(
        json.dumps(jsonable(frozen), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    frozen_policy_sha256 = sha256(policy_path)
    scan_sorted = sorted(
        scan,
        key=lambda row: row["summary"]["corrected"]["root_error_m"]["mean"],
    )
    (output_dir / "dev_parameter_scan.json").write_text(
        json.dumps(
            jsonable(
                {
                    "grid": {
                        "count": len(scan),
                        "mechanism_active_count": sum(row["mechanism_active"] for row in scan),
                    },
                    "eligibility_thresholds": thresholds,
                    "eligible_count": len(eligible),
                    "selected": selected,
                    "top100_by_root": scan_sorted[:100],
                    "all": scan,
                }
            ),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    report = {
        "experiment": "v14_b0_brtc_reliability_huber_irls",
        "phase": "dev",
        "policy_source": str(policy_path),
        "policy_sha256": frozen_policy_sha256,
        "policy": config_dict(selected_config),
        "dev_pass": dev_pass,
        "selection": {
            "grid_count": len(scan),
            "eligible_count": len(eligible),
            "thresholds": thresholds,
        },
        "methods": {
            "legacy_brtc_lc": {"summary": legacy_full, "cases": legacy_full_cases},
            "huber_irls_brtc_lc": {
                "summary": candidate_full,
                "cases": candidate_full_cases,
            },
        },
    }
    (output_dir / "dev_report.json").write_text(
        json.dumps(jsonable(report), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "dev_report.md").write_text(
        comparison_markdown(report), encoding="utf-8"
    )
    print(comparison_markdown(report), flush=True)
    print(
        json.dumps(
            jsonable(
                {
                    "dev_pass": dev_pass,
                    "selected_config": config_dict(selected_config),
                    "legacy": legacy_full,
                    "candidate": candidate_full,
                }
            ),
            indent=2,
            allow_nan=False,
        ),
        flush=True,
    )


def run_confirm(args: argparse.Namespace) -> None:
    policy_path = args.policy.resolve()
    if not policy_path.is_file():
        raise FileNotFoundError(f"Frozen development policy required: {policy_path}")
    frozen = json.loads(policy_path.read_text(encoding="utf-8"))
    if not frozen.get("dev_pass", False):
        raise RuntimeError("Development rule failed; confirmation is not authorized")
    selected_config = policy_from_json(frozen["policy"])
    rows = load_rows("confirm", args.confirm_report, args.max_cases)
    prepared = prepare_all(rows)
    old_policy = legacy_policy()
    legacy_cases = evaluate_method(
        prepared,
        lambda person: legacy_proposal(person, old_policy),
        "legacy_brtc_lc",
        full=True,
    )
    candidate_cases = evaluate_method(
        prepared,
        lambda person: candidate_proposal(person, selected_config),
        "huber_irls_brtc_lc",
        full=True,
    )
    legacy_summary = summarize(legacy_cases, full=True)
    candidate_summary = summarize(candidate_cases, full=True)
    confirm_beats_legacy = bool(
        candidate_summary["corrected"]["root_error_m"]["mean"]
        < legacy_summary["corrected"]["root_error_m"]["mean"]
        and candidate_summary["root_harm_over_5cm_rate"] <= 0.10
        and candidate_summary["coverage"]
        >= frozen["eligibility_thresholds_frozen_before_confirm"]["coverage_min"]
        and candidate_summary["corrected"]["pairwise_vector_error_m"]["mean"]
        <= legacy_summary["corrected"]["pairwise_vector_error_m"]["mean"] * 1.02
        and candidate_summary["corrected"]["pairwise_distance_error_m"]["mean"]
        <= legacy_summary["corrected"]["pairwise_distance_error_m"]["mean"] * 1.02
        and candidate_summary["fallback_max_abs_change"] == 0.0
        and candidate_summary["camera_candidate_max_abs_change"] == 0.0
    )
    report = {
        "experiment": "v14_b0_brtc_reliability_huber_irls",
        "phase": "frozen_confirm",
        "protocol": {
            "selection": "policy loaded without modification from offset0 dev",
            "confirmation": "three offset1, 42 cuts / 125 people",
            "camera": "frozen B0",
            "rejected_contract": "exact B0",
            "pretrained_model": "none",
        },
        "policy_source": str(policy_path),
        "policy_sha256_at_confirm": sha256(policy_path),
        "policy": config_dict(selected_config),
        "confirm_beats_legacy": confirm_beats_legacy,
        "methods": {
            "legacy_brtc_lc": {"summary": legacy_summary, "cases": legacy_cases},
            "huber_irls_brtc_lc": {
                "summary": candidate_summary,
                "cases": candidate_cases,
            },
        },
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "confirm_report.json").write_text(
        json.dumps(jsonable(report), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "confirm_report.md").write_text(
        comparison_markdown(report), encoding="utf-8"
    )
    print(comparison_markdown(report), flush=True)
    print(
        json.dumps(
            jsonable(
                {
                    "confirm_beats_legacy": confirm_beats_legacy,
                    "legacy": legacy_summary,
                    "candidate": candidate_summary,
                }
            ),
            indent=2,
            allow_nan=False,
        ),
        flush=True,
    )


def run_compare(args: argparse.Namespace) -> None:
    """Compare the frozen robust rule against the independently fixed 0.8 baseline."""
    policy_path = args.policy.resolve()
    if not policy_path.is_file():
        raise FileNotFoundError(f"Frozen development policy required: {policy_path}")
    frozen = json.loads(policy_path.read_text(encoding="utf-8"))
    if not frozen.get("dev_pass", False):
        raise RuntimeError("Development rule failed")
    selected_config = policy_from_json(frozen["policy"])
    old_policy = legacy_policy()
    splits = {}
    split_rows = (
        ("dev_offset0", load_rows("dev", args.confirm_report, args.max_cases)),
        (
            "confirm_offset1",
            load_rows("confirm", args.confirm_report, args.max_cases),
        ),
        (
            "dance_box_frozen_external",
            legacy.report_rows(("dance", "box"))[: int(args.max_cases) or None],
        ),
    )
    for split, rows in split_rows:
        prepared = prepare_all(rows)
        method_cases = {
            "legacy_brtc_lc": evaluate_method(
                prepared,
                lambda person: legacy_proposal(person, old_policy),
                "legacy_brtc_lc",
                full=True,
            ),
            "legacy_brtc_lc_damped_0p8": evaluate_method(
                prepared,
                lambda person: damped_legacy_proposal(person, old_policy, 0.8),
                "legacy_brtc_lc",
                full=True,
            ),
            "huber_irls_brtc_lc": evaluate_method(
                prepared,
                lambda person: candidate_proposal(person, selected_config),
                "huber_irls_brtc_lc",
                full=True,
            ),
            "huber_irls_brtc_lc_damped_0p8_exploratory": evaluate_method(
                prepared,
                lambda person: damped_candidate_proposal(
                    person, selected_config, 0.8
                ),
                "huber_irls_brtc_lc",
                full=True,
            ),
        }
        splits[split] = {
            "methods": {
                name: {"summary": summarize(cases, full=True), "cases": cases}
                for name, cases in method_cases.items()
            }
        }

    # Require the robust candidate to be a credible replacement for the strong
    # baseline on the untouched confirmation split, not merely to win one mean.
    confirm = splits["confirm_offset1"]["methods"]
    robust = confirm["huber_irls_brtc_lc"]["summary"]
    composite = confirm[
        "huber_irls_brtc_lc_damped_0p8_exploratory"
    ]["summary"]
    damped = confirm["legacy_brtc_lc_damped_0p8"]["summary"]
    full_error_metrics = FULL_METRICS + LAYOUT_METRICS
    error_not_worse = all(
        robust["corrected"][metric]["mean"]
        <= damped["corrected"][metric]["mean"] + 1e-12
        for metric in full_error_metrics
    )
    safety_not_worse = bool(
        robust["coverage"] >= damped["coverage"]
        and robust["root_harm_over_1cm_rate"] <= damped["root_harm_over_1cm_rate"]
        and robust["root_harm_over_5cm_rate"] <= damped["root_harm_over_5cm_rate"]
    )
    robust_beats = bool(error_not_worse and safety_not_worse)
    composite_error_not_worse = all(
        composite["corrected"][metric]["mean"]
        <= damped["corrected"][metric]["mean"] + 1e-12
        for metric in full_error_metrics
    )
    composite_safety_not_worse = bool(
        composite["coverage"] >= damped["coverage"]
        and composite["root_harm_over_1cm_rate"]
        <= damped["root_harm_over_1cm_rate"]
        and composite["root_harm_over_5cm_rate"]
        <= damped["root_harm_over_5cm_rate"]
    )
    report = {
        "experiment": "v14_b0_brtc_huber_irls_strong_baseline_comparison",
        "protocol": {
            "robust_policy": "frozen on dev_offset0 before robust confirm",
            "strong_baseline": (
                "legacy BRTC-LC individual proposal multiplied by fixed 0.8; "
                "group shift scales identically; observable lambda is reselected"
            ),
            "strong_baseline_role": (
                "independent comparator supplied after robust development freeze; "
                "not used to retune robust policy"
            ),
            "camera": "bit-exact frozen B0",
            "rejected_contract": "exact B0",
            "pretrained_model": "none",
        },
        "policy_source": str(policy_path),
        "policy_sha256": sha256(policy_path),
        "policy": config_dict(selected_config),
        "splits": splits,
        "robust_confirm_error_not_worse_all_metrics": error_not_worse,
        "robust_confirm_safety_not_worse": safety_not_worse,
        "robust_beats_damped_strong_baseline": robust_beats,
        "exploratory_composite_confirm_error_not_worse_all_metrics": (
            composite_error_not_worse
        ),
        "exploratory_composite_confirm_safety_not_worse": (
            composite_safety_not_worse
        ),
        "exploratory_composite_protocol_status": (
            "diagnostic only; fixed 0.8 comparator was supplied after robust policy "
            "freeze and this composition needs a new clean split before promotion"
        ),
        "decision": (
            "GO_HUBER_IRLS"
            if robust_beats
            else "NO_GO_HUBER_IRLS_KEEP_FIXED_0P8_DAMPED_BRTC_LC"
        ),
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "strong_damped_baseline_comparison.json").write_text(
        json.dumps(jsonable(report), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "strong_damped_baseline_comparison.md").write_text(
        strong_comparison_markdown(report), encoding="utf-8"
    )
    print(strong_comparison_markdown(report), flush=True)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if REPO_ROOT not in output.parents:
        raise ValueError("Output must stay under Movie3R on /data")
    if args.phase == "dev":
        run_dev(args)
    elif args.phase == "confirm":
        run_confirm(args)
    else:
        run_compare(args)


if __name__ == "__main__":
    main()
