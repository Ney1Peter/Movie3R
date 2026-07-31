#!/usr/bin/env python3
"""Develop and confirm an observable association dustbin for BRTC.

The strict FAGD runtime can still be triggered by a population-preserving
identity replacement (for example, 2 -> 2 with one person leaving and another
entering).  Population counts cannot detect that event.  This probe therefore
adds a per-Hungarian-edge confidence gate using only the already available B0
person geometry.  A rejected post person is sent to a dustbin and remains
bit-exact B0; retained matches are passed to the frozen BRTC/FAGD runtimes.

Protocol:

* development uses all ``three`` offset-0 same-visibility and variable-
  visibility cuts;
* the selected single-feature absolute threshold is serialized before any
  held-out read;
* confirmation uses ``three`` offset-1 same-visibility plus ``dance`` and
  ``box`` same/variable-visibility cuts;
* GT identity is consulted only after Hungarian prediction to label edges and
  evaluate the frozen gate.  It is never consumed by the gate or refinement;
* no image model, future frame, camera update, or GPU is used.

The normalized Hungarian cost and ambiguity margins are audited, but policy
selection deliberately uses raw absolute components.  This is essential for
1 x 1 boundaries, where median-normalized components are always constant.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from versions.v13 import gt_id_consensus as geometry  # noqa: E402
from versions.v14 import probe_b0_identity_matching as identity_probe  # noqa: E402
from versions.v14 import probe_brtc_variable_visibility as variable  # noqa: E402
from versions.v14.b0_person_triangulation import refine_matched_people  # noqa: E402
from versions.v14.b0_person_triangulation_strict_fagd import (  # noqa: E402
    StrictFAGDConfig,
    refine_matched_people_strict_fagd,
)
from versions.v14.probe_b0_anchor_conflict import SEQUENCE_INPUTS  # noqa: E402


DEFAULT_OUTPUT = (
    REPO_ROOT / "output/v14/fine_alignment_research/brtc_association_dustbin_gate"
)
DEFAULT_POLICY = DEFAULT_OUTPUT / "FROZEN_POLICY_BEFORE_HELDOUT.json"
DEFAULT_DATA_ROOT = Path(
    "/data/wangzheng/iJCV-CODE/data/MultiHuman/Real-World-Capture/extracted"
)
REPORTS = {
    "three_k0": REPO_ROOT / "output/v14/b0_identity_matching/v14_b0_identity_matching.json",
    "three_k1": REPO_ROOT
    / "output/v14/b0_identity_matching_offset1_confirm/v14_b0_identity_matching.json",
    "dance": REPO_ROOT
    / "output/v14/b0_identity_matching_extended/dance/v14_b0_identity_matching.json",
    "box": REPO_ROOT
    / "output/v14/b0_identity_matching_extended/box/v14_b0_identity_matching.json",
}
VARIABLE_BOUNDARY_DIRS = {
    "three": REPO_ROOT / "output/v14/fine_alignment_research/brtc_variable_visibility/cases",
    "dance": REPO_ROOT
    / "output/v14/fine_alignment_research/brtc_variable_visibility/dance/cases",
    "box": REPO_ROOT
    / "output/v14/fine_alignment_research/brtc_variable_visibility/box/cases",
}
RAW_FEATURE_GRIDS = {
    "root_m": (0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.75, 1.00),
    "torso_deg": (20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 75.0, 90.0, 120.0, 180.0),
    "centered_joint_m": (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50),
}
POINT_KEYS = ("root", "joints", "vertices")
PRIMARY = (
    "root_error_m",
    "joint_error_m",
    "vertex_error_m",
    "pairwise_distance_error_m",
    "pairwise_vector_error_m",
)
METHODS = ("b0", "brtc_v1", "strict_fagd", "dustbin_v1", "dustbin_strict_fagd")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("dev", "freeze", "confirm"), required=True)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--self_test", action="store_true")
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
        return "inf" if value > 0 else "-inf"
    return value


def canonical_sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(
        jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def finite_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if len(array) else float("nan")


def finite_stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {key: float("nan") for key in ("min", "median", "p90", "max")}
    return {
        "min": float(array.min()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(array.max()),
    }


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def boundary_from_row(row: dict[str, Any]) -> np.ndarray:
    return np.asarray(row["boundaries"]["learned_b0"], dtype=np.float64)


def case_specs(
    sequence: str,
    report_path: Path,
    include_same: bool,
    include_variable: bool,
    max_cases: int = 0,
) -> list[dict[str, Any]]:
    report = load_report(report_path)
    specs: list[dict[str, Any]] = []
    if include_same:
        for row in report["cases"]:
            specs.append(
                {
                    "sequence": sequence,
                    "visibility": "same",
                    "case_name": str(row["case"]["key"]),
                    "boundary": boundary_from_row(row),
                    "boundary_source": str(report_path),
                }
            )
    if include_variable:
        for name in report["protocol"]["excluded_variable_visibility_cases"]:
            path = VARIABLE_BOUNDARY_DIRS[sequence] / f"{name}.json"
            row = load_report(path)
            specs.append(
                {
                    "sequence": sequence,
                    "visibility": "variable",
                    "case_name": str(name),
                    "boundary": np.asarray(row["learned_b0"], dtype=np.float64),
                    "boundary_source": str(path),
                }
            )
    return specs[: int(max_cases)] if max_cases else specs


def load_cache(args: argparse.Namespace, spec: dict[str, Any]) -> dict[str, Any]:
    sequence = str(spec["sequence"])
    geometry.IDENTITIES = geometry.SEQUENCE_IDENTITIES[sequence]
    cache_args = argparse.Namespace(
        data_root=args.data_root,
        size=int(args.size),
        sequence=sequence,
    )
    return identity_probe.strict_cache(
        cache_args, SEQUENCE_INPUTS[sequence]["cache"] / f"{spec['case_name']}.pt"
    )


def second_assignment_margin(cost: np.ndarray, matches: list[tuple[int, int]]) -> float:
    """Return second-best minus best rectangular assignment cost."""

    if min(cost.shape) <= 0 or len(matches) <= 1 and cost.size <= 1:
        return float("inf")
    best = float(sum(cost[row, column] for row, column in matches))
    alternatives = []
    for blocked_row, blocked_column in matches:
        candidate = np.asarray(cost, dtype=np.float64).copy()
        candidate[blocked_row, blocked_column] = np.inf
        try:
            rows, columns = linear_sum_assignment(candidate)
        except ValueError:
            continue
        if len(rows) != min(cost.shape):
            continue
        values = candidate[rows, columns]
        if np.isfinite(values).all():
            alternatives.append(float(values.sum()))
    return min(alternatives) - best if alternatives else float("inf")


def edge_feature(
    matrix: np.ndarray, row: int, column: int, axis: int
) -> tuple[float, float]:
    """Return best alternative and alternative-minus-assigned margin."""

    values = matrix[row, :] if axis == 1 else matrix[:, column]
    assigned_index = column if axis == 1 else row
    alternatives = np.delete(values, assigned_index)
    alternative = float(np.min(alternatives)) if len(alternatives) else float("inf")
    return alternative, alternative - float(matrix[row, column])


def anonymous_association(
    pre_order: list[tuple[str, dict[str, Any]]],
    post_order: list[tuple[str, dict[str, Any]]],
    boundary: np.ndarray,
) -> tuple[list[tuple[int, int]], list[dict[str, Any]], dict[str, Any]]:
    """Predict anonymously, then attach evaluator-only identity correctness."""

    pre_geometry = {identity: person for identity, person in pre_order}
    anonymous_row_keys = tuple(identity for identity, _ in pre_order)
    components = identity_probe.identity_cost_components(
        pre_geometry, post_order, boundary, anonymous_row_keys
    )
    cost = identity_probe.matching_costs(components)["root_torso_joints"]
    rows, columns = linear_sum_assignment(cost)
    matches = [(int(row), int(column)) for row, column in zip(rows, columns)]
    global_margin = second_assignment_margin(cost, matches)
    edges = []
    for row, column in matches:
        row_alternative, row_margin = edge_feature(cost, row, column, axis=1)
        column_alternative, column_margin = edge_feature(cost, row, column, axis=0)
        # Identity strings enter only below, after the numeric assignment and
        # all inference features have already been computed.
        correct = pre_order[row][0] == post_order[column][0]
        edges.append(
            {
                "pre_index": row,
                "post_index": column,
                "root_m": float(components["root"][row, column]),
                "torso_deg": float(components["torso"][row, column]),
                "centered_joint_m": float(components["joints"][row, column]),
                "normalized_combined_cost": float(cost[row, column]),
                "row_alternative_cost": row_alternative,
                "row_margin": row_margin,
                "column_alternative_cost": column_alternative,
                "column_margin": column_margin,
                "global_assignment_margin": float(global_margin),
                "pre_identity_evaluator_only": str(pre_order[row][0]),
                "post_identity_evaluator_only": str(post_order[column][0]),
                "correct_evaluator_only": bool(correct),
            }
        )
    return matches, edges, {
        "cost_shape": [int(value) for value in cost.shape],
        "global_assignment_margin": float(global_margin),
        "normalized_combined_cost": cost,
    }


def gate_accepts(edge: dict[str, Any], policy: dict[str, Any]) -> bool:
    if policy["kind"] != "single_raw_absolute_max":
        raise ValueError(f"Unsupported gate policy: {policy['kind']}")
    return bool(float(edge[policy["feature"]]) <= float(policy["max_value"]))


def association_counts(cases: list[dict[str, Any]], gated: bool) -> dict[str, Any]:
    edges = [edge for case in cases for edge in case["association"]["edges"]]
    selected = [edge for edge in edges if edge["gate_accept"]] if gated else edges
    correct = sum(bool(edge["correct_evaluator_only"]) for edge in selected)
    wrong = len(selected) - correct
    shared = sum(int(case["association"]["shared_identity_count"]) for case in cases)
    predicted_correct = sum(bool(edge["correct_evaluator_only"]) for edge in edges)
    return {
        "edge_count": len(edges),
        "accepted_count": len(selected),
        "correct_accept_count": int(correct),
        "wrong_accept_count": int(wrong),
        "precision": float(correct / len(selected)) if selected else float("nan"),
        "recall_over_shared_identities": float(correct / shared) if shared else float("nan"),
        "coverage_over_correct_hungarian_edges": (
            float(correct / predicted_correct) if predicted_correct else float("nan")
        ),
        "rejected_count": int(len(edges) - len(selected)),
        "rejected_wrong_count": int(
            sum(
                (not edge["gate_accept"]) and (not edge["correct_evaluator_only"])
                for edge in edges
            )
        )
        if gated
        else 0,
    }


def scan_policy(feature_cases: list[dict[str, Any]]) -> dict[str, Any]:
    edges = [edge for case in feature_cases for edge in case["edges"]]
    correct_count = sum(bool(edge["correct_evaluator_only"]) for edge in edges)
    candidates = []
    for feature, thresholds in RAW_FEATURE_GRIDS.items():
        correct_values = np.asarray(
            [float(edge[feature]) for edge in edges if edge["correct_evaluator_only"]],
            dtype=np.float64,
        )
        wrong_values = np.asarray(
            [float(edge[feature]) for edge in edges if not edge["correct_evaluator_only"]],
            dtype=np.float64,
        )
        for threshold in thresholds:
            accepted = [edge for edge in edges if float(edge[feature]) <= threshold]
            accepted_correct = [edge for edge in accepted if edge["correct_evaluator_only"]]
            accepted_wrong = [edge for edge in accepted if not edge["correct_evaluator_only"]]
            lower = float(correct_values.max()) if len(correct_values) else float("nan")
            upper = float(wrong_values.min()) if len(wrong_values) else float("inf")
            robust_ratio = (
                min(float(threshold) / max(lower, 1e-12), upper / float(threshold))
                if len(correct_values) and len(wrong_values) and threshold >= lower
                else 0.0
            )
            candidates.append(
                {
                    "kind": "single_raw_absolute_max",
                    "feature": feature,
                    "max_value": float(threshold),
                    "accepted_count": len(accepted),
                    "correct_accept_count": len(accepted_correct),
                    "wrong_accept_count": len(accepted_wrong),
                    "correct_coverage": (
                        float(len(accepted_correct) / correct_count)
                        if correct_count
                        else float("nan")
                    ),
                    "robust_gap_ratio": float(robust_ratio),
                    "all_correct_feature_max": lower,
                    "all_wrong_feature_min": upper,
                }
            )
    selected = min(
        candidates,
        key=lambda row: (
            int(row["wrong_accept_count"]),
            -int(row["correct_accept_count"]),
            -float(row["robust_gap_ratio"]),
            str(row["feature"]),
            float(row["max_value"]),
        ),
    )
    policy = {
        "kind": selected["kind"],
        "feature": selected["feature"],
        "max_value": selected["max_value"],
        "selection_objective": (
            "minimize wrong accepts, maximize retained correct edges, then maximize "
            "multiplicative threshold gap; single raw absolute feature only"
        ),
        "development_wrong_accept_count": selected["wrong_accept_count"],
        "development_correct_coverage": selected["correct_coverage"],
        "development_robust_gap_ratio": selected["robust_gap_ratio"],
    }
    return {"policy": policy, "selected_scan_row": selected, "scan": candidates}


def feature_distributions(feature_cases: list[dict[str, Any]]) -> dict[str, Any]:
    edges = [edge for case in feature_cases for edge in case["edges"]]
    features = (
        "root_m",
        "torso_deg",
        "centered_joint_m",
        "normalized_combined_cost",
        "row_margin",
        "column_margin",
        "global_assignment_margin",
    )
    return {
        label: {
            feature: finite_stats(
                [
                    float(edge[feature])
                    for edge in edges
                    if bool(edge["correct_evaluator_only"]) == correct
                ]
            )
            for feature in features
        }
        for label, correct in (("correct", True), ("wrong", False))
    }


def collect_features(
    args: argparse.Namespace, specs: list[dict[str, Any]], tag: str
) -> list[dict[str, Any]]:
    output = []
    for index, spec in enumerate(specs, start=1):
        cache = load_cache(args, spec)
        pre_order = variable.ordered_people(cache["humans"][-2])
        post_order = variable.ordered_people(cache["humans"][-1])
        _, edges, audit = anonymous_association(
            pre_order, post_order, np.asarray(spec["boundary"], dtype=np.float64)
        )
        output.append(
            {
                "sequence": spec["sequence"],
                "visibility": spec["visibility"],
                "case_name": spec["case_name"],
                "pre_person_count": len(pre_order),
                "post_person_count": len(post_order),
                "shared_identity_count_evaluator_only": len(
                    set(identity for identity, _ in pre_order)
                    & set(identity for identity, _ in post_order)
                ),
                "cost_shape": audit["cost_shape"],
                "global_assignment_margin": audit["global_assignment_margin"],
                "edges": edges,
            }
        )
        print(
            f"[{tag} features {index:03d}/{len(specs):03d}] "
            f"{spec['case_name']} {len(pre_order)}->{len(post_order)}",
            flush=True,
        )
    return output


def person_geometry(
    person: dict[str, Any], transform: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    return variable.person_geometry(person, transform)


def exact_post_parity(
    baseline: list[dict[str, np.ndarray]],
    candidate: list[dict[str, np.ndarray]],
    post_indices: set[int],
) -> tuple[bool, float]:
    exact, maximum = True, 0.0
    for index in sorted(post_indices):
        for key in POINT_KEYS:
            first = np.asarray(baseline[index][key])
            second = np.asarray(candidate[index][key])
            exact = bool(exact and np.array_equal(first, second))
            maximum = max(maximum, float(np.max(np.abs(first - second))))
    return exact, maximum


def evaluate_case(
    args: argparse.Namespace,
    spec: dict[str, Any],
    policy: dict[str, Any],
    alpha: float,
) -> dict[str, Any]:
    cache = load_cache(args, spec)
    boundary = np.asarray(spec["boundary"], dtype=np.float64)
    pre_order = variable.ordered_people(cache["humans"][-2])
    post_order = variable.ordered_people(cache["humans"][-1])
    pre_people = [person_geometry(person) for _, person in pre_order]
    post_people = [person_geometry(person, boundary) for _, person in post_order]
    pre_camera = np.asarray(cache["poses"][-2], dtype=np.float64)
    post_camera = boundary @ np.asarray(cache["poses"][-1], dtype=np.float64)
    before_pre, before_post = pre_camera.copy(), post_camera.copy()

    matches, edges, association_audit = anonymous_association(
        pre_order, post_order, boundary
    )
    for edge in edges:
        edge["gate_accept"] = gate_accepts(edge, policy)
    gated_matches = [
        (int(edge["pre_index"]), int(edge["post_index"]))
        for edge in edges
        if edge["gate_accept"]
    ]

    v1, v1_debug = refine_matched_people(
        pre_camera, post_camera, pre_people, post_people, matches
    )
    strict, strict_debug = refine_matched_people_strict_fagd(
        pre_camera,
        post_camera,
        pre_people,
        post_people,
        matches,
        fagd_config=StrictFAGDConfig(alpha=alpha),
    )
    dustbin_v1, dustbin_v1_debug = refine_matched_people(
        pre_camera, post_camera, pre_people, post_people, gated_matches
    )
    dustbin_strict, dustbin_strict_debug = refine_matched_people_strict_fagd(
        pre_camera,
        post_camera,
        pre_people,
        post_people,
        gated_matches,
        fagd_config=StrictFAGDConfig(alpha=alpha),
    )
    camera_change = max(
        float(np.max(np.abs(pre_camera - before_pre))),
        float(np.max(np.abs(post_camera - before_post))),
    )

    gt_pre = np.asarray(cache["gt"]["pre_c2w"], dtype=np.float64)
    gauge = pre_camera @ np.linalg.inv(gt_pre)
    targets = [
        person_geometry(cache["gt"]["post_humans"][identity], gauge)
        for identity, _ in post_order
    ]
    predictions = {
        "b0": post_people,
        "brtc_v1": v1,
        "strict_fagd": strict,
        "dustbin_v1": dustbin_v1,
        "dustbin_strict_fagd": dustbin_strict,
    }
    metrics = {}
    b0_root = np.asarray(
        [
            variable.point_errors(predicted, target)["root_error_m"]
            for predicted, target in zip(post_people, targets)
        ],
        dtype=np.float64,
    )
    for name, people in predictions.items():
        errors = [
            variable.point_errors(predicted, target)
            for predicted, target in zip(people, targets)
        ]
        roots = np.asarray([row["root_error_m"] for row in errors], dtype=np.float64)
        metrics[name] = {
            "person_errors": errors,
            "layout": variable.layout_errors(people, targets),
            "root_deltas_vs_b0_m": (roots - b0_root).tolist(),
        }

    rejected_post = set(range(len(post_people))) - {
        int(column) for _, column in gated_matches
    }
    v1_exact, v1_max = exact_post_parity(post_people, dustbin_v1, rejected_post)
    strict_exact, strict_max = exact_post_parity(
        post_people, dustbin_strict, rejected_post
    )
    shared_count = len(
        set(identity for identity, _ in pre_order)
        & set(identity for identity, _ in post_order)
    )
    return {
        "sequence": spec["sequence"],
        "visibility": spec["visibility"],
        "case_name": spec["case_name"],
        "pre_person_count": len(pre_people),
        "post_person_count": len(post_people),
        "equal_count_replacement_evaluator_only": bool(
            len(pre_people) == len(post_people)
            and set(identity for identity, _ in pre_order)
            != set(identity for identity, _ in post_order)
        ),
        "association": {
            "matcher": "rectangular Hungarian on normalized root+torso+centered-joints",
            "cost_shape": association_audit["cost_shape"],
            "shared_identity_count": shared_count,
            "matches": matches,
            "gated_matches": gated_matches,
            "edges": edges,
        },
        "runtime": {
            "camera_max_abs_change": camera_change,
            "v1_accepted_count": int(v1_debug["accepted_count"]),
            "strict_fagd_gate": bool(
                strict_debug["strict_full_one_to_one_all_accepted"]
            ),
            "dustbin_v1_accepted_count": int(dustbin_v1_debug["accepted_count"]),
            "dustbin_strict_fagd_gate": bool(
                dustbin_strict_debug["strict_full_one_to_one_all_accepted"]
            ),
            "rejected_post_count": len(rejected_post),
            "rejected_post_exact_b0_dustbin_v1": v1_exact,
            "rejected_post_exact_b0_dustbin_strict_fagd": strict_exact,
            "rejected_post_max_abs_change": max(v1_max, strict_max),
        },
        "metrics": metrics,
    }


def aggregate_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "case_count": len(cases),
        "post_person_count": int(sum(row["post_person_count"] for row in cases)),
        "equal_count_replacement_case_count": int(
            sum(row["equal_count_replacement_evaluator_only"] for row in cases)
        ),
        "association_before_gate": association_counts(cases, gated=False),
        "association_after_gate": association_counts(cases, gated=True),
        "camera_max_abs_change": float(
            max((row["runtime"]["camera_max_abs_change"] for row in cases), default=0.0)
        ),
        "rejected_post_exact_b0": bool(
            all(
                row["runtime"]["rejected_post_exact_b0_dustbin_v1"]
                and row["runtime"]["rejected_post_exact_b0_dustbin_strict_fagd"]
                for row in cases
            )
        ),
        "rejected_post_max_abs_change": float(
            max(
                (row["runtime"]["rejected_post_max_abs_change"] for row in cases),
                default=0.0,
            )
        ),
        "strict_fagd_gate_case_count": int(
            sum(row["runtime"]["strict_fagd_gate"] for row in cases)
        ),
        "dustbin_strict_fagd_gate_case_count": int(
            sum(row["runtime"]["dustbin_strict_fagd_gate"] for row in cases)
        ),
        "methods": {},
    }
    for method in METHODS:
        person_errors = [
            error
            for case in cases
            for error in case["metrics"][method]["person_errors"]
        ]
        layouts = [case["metrics"][method]["layout"] for case in cases]
        deltas = np.asarray(
            [
                value
                for case in cases
                for value in case["metrics"][method]["root_deltas_vs_b0_m"]
            ],
            dtype=np.float64,
        )
        value = {
            key: finite_mean([float(row[key]) for row in person_errors])
            for key in ("root_error_m", "joint_error_m", "vertex_error_m")
        }
        value.update(
            {
                key: finite_mean([float(row[key]) for row in layouts])
                for key in (
                    "pairwise_distance_error_m",
                    "pairwise_vector_error_m",
                )
            }
        )
        value.update(
            {
                "root_improve_rate": float(np.mean(deltas < 0.0)) if len(deltas) else 0.0,
                "root_harm_over_1cm_rate": (
                    float(np.mean(deltas > 0.01)) if len(deltas) else 0.0
                ),
                "root_harm_over_5cm_rate": (
                    float(np.mean(deltas > 0.05)) if len(deltas) else 0.0
                ),
                "root_mean_delta_m": float(np.mean(deltas)) if len(deltas) else 0.0,
            }
        )
        result["methods"][method] = value
    return result


def evaluate_specs(
    args: argparse.Namespace,
    specs: list[dict[str, Any]],
    policy: dict[str, Any],
    tag: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cases = []
    for index, spec in enumerate(specs, start=1):
        case = evaluate_case(args, spec, policy, float(args.alpha))
        cases.append(case)
        accepted = len(case["association"]["gated_matches"])
        matched = len(case["association"]["matches"])
        print(
            f"[{tag} eval {index:03d}/{len(specs):03d}] {spec['case_name']} "
            f"gate={accepted}/{matched}",
            flush=True,
        )
    return cases, aggregate_cases(cases)


def summary_for_groups(cases: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {"all": cases}
    for visibility in ("same", "variable"):
        selected = [row for row in cases if row["visibility"] == visibility]
        if selected:
            groups[visibility] = selected
    replacement = [row for row in cases if row["equal_count_replacement_evaluator_only"]]
    if replacement:
        groups["equal_count_replacement"] = replacement
    return {name: aggregate_cases(rows) for name, rows in groups.items()}


def method_row(split: str, method: str, value: dict[str, Any]) -> str:
    metric = value["methods"][method]
    return (
        f"| {split} | {method} | {metric['root_error_m']:.6f} | "
        f"{metric['joint_error_m']:.6f} | {metric['vertex_error_m']:.6f} | "
        f"{metric['pairwise_distance_error_m']:.6f} | "
        f"{metric['pairwise_vector_error_m']:.6f} | "
        f"{metric['root_harm_over_1cm_rate']:.1%} | "
        f"{metric['root_harm_over_5cm_rate']:.1%} |"
    )


def development_markdown(report: dict[str, Any]) -> str:
    policy = report["selection"]["policy"]
    all_summary = report["runtime_summary"]["all"]
    before = all_summary["association_before_gate"]
    after = all_summary["association_after_gate"]
    lines = [
        "# Observable association dustbin: development",
        "",
        "Development data: all MultiHuman `three` offset-0 same-visibility and variable-visibility cuts.",
        "",
        f"Selected gate: `{policy['feature']} <= {policy['max_value']}`.",
        f"Association before gate: precision `{before['precision']:.1%}`, wrong accepts `{before['wrong_accept_count']}`.",
        f"Association after gate: precision `{after['precision']:.1%}`, correct coverage `{after['coverage_over_correct_hungarian_edges']:.1%}`, wrong accepts `{after['wrong_accept_count']}`.",
        "",
        "| Method | Root | Joint | Vertex | Pair distance | Pair vector | Harm >1cm | Harm >5cm |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = method_row("dev", method, all_summary).replace("| dev | ", "| ", 1)
        lines.append(row)
    lines.extend(
        [
            "",
            f"Equal-count replacement cuts: `{report['runtime_summary'].get('equal_count_replacement', {}).get('case_count', 0)}`.",
            f"Rejected people exact B0: `{all_summary['rejected_post_exact_b0']}`; camera max change `{all_summary['camera_max_abs_change']:.3e}`.",
            "Held-out data has not been read by this phase.",
        ]
    )
    return "\n".join(lines) + "\n"


def confirmation_markdown(report: dict[str, Any]) -> str:
    policy = report["policy"]
    lines = [
        "# Frozen association dustbin: held-out confirmation",
        "",
        f"Frozen gate: `{policy['feature']} <= {policy['max_value']}`.",
        "",
        "| Split | Cases | True equal-count replacements | Before precision | After precision | Correct coverage | Wrong accepts | Rejected wrong | Exact B0 fallback |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for split, value in report["splits"].items():
        before = value["summary"]["association_before_gate"]
        after = value["summary"]["association_after_gate"]
        lines.append(
            f"| {split} | {value['summary']['case_count']} | "
            f"{value['summary']['equal_count_replacement_case_count']} | "
            f"{before['precision']:.1%} | "
            f"{after['precision']:.1%} | {after['coverage_over_correct_hungarian_edges']:.1%} | "
            f"{after['wrong_accept_count']} | {after['rejected_wrong_count']} | "
            f"{value['summary']['rejected_post_exact_b0']} |"
        )
    lines.extend(
        [
            "",
            "| Split | Method | Root | Joint | Vertex | Pair distance | Pair vector | Harm >1cm | Harm >5cm |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split, value in report["splits"].items():
        for method in ("b0", "strict_fagd", "dustbin_strict_fagd"):
            lines.append(method_row(split, method, value["summary"]))
    decision = report["decision"]
    lines.extend(
        [
            "",
            f"All wrong held-out edges rejected: `{decision['all_wrong_edges_rejected']}`.",
            f"All rejected people exact B0: `{decision['all_rejected_people_exact_b0']}`.",
            f"Camera bit-exact B0: `{decision['camera_bit_exact_b0']}`.",
            f"All five metrics no worse than ungated strict FAGD on every split: `{decision['all_metrics_no_worse_everywhere']}`.",
            f"Decision: **{decision['status']}**.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_dev(args: argparse.Namespace) -> None:
    specs = case_specs("three", REPORTS["three_k0"], True, True, args.max_cases)
    feature_cases = collect_features(args, specs, "dev")
    selection = scan_policy(feature_cases)
    policy = selection["policy"]
    cases, _ = evaluate_specs(args, specs, policy, "dev")
    report = {
        "experiment": "v14_observable_association_dustbin_development",
        "protocol": {
            "development": "MultiHuman three offset-0: all same and variable visibility cuts",
            "case_count": len(specs),
            "gt_use": "edge labels and metrics only, after anonymous Hungarian inference",
            "future_frames": 0,
            "extra_pretrained_models": [],
            "camera_update": "none",
            "candidate_family": "one raw absolute max threshold on one feature",
            "normalized_cost_policy_excluded_reason": (
                "1x1 median normalization is constant and cannot identify replacement"
            ),
        },
        "selection": selection,
        "feature_distributions": feature_distributions(feature_cases),
        "feature_cases": feature_cases,
        "runtime_summary": summary_for_groups(cases),
        "runtime_cases": cases,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "development_report.json").write_text(
        json.dumps(jsonable(report), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    text = development_markdown(report)
    (args.output_dir / "development_report.md").write_text(text, encoding="utf-8")
    print(text, end="")


def run_freeze(args: argparse.Namespace) -> None:
    path = args.output_dir / "development_report.json"
    report = load_report(path)
    policy = report["selection"]["policy"]
    frozen = {
        "experiment": "v14_observable_association_dustbin_frozen_policy",
        "frozen_before_heldout": True,
        "development_report": str(path),
        "policy": policy,
        "policy_sha256": canonical_sha256(policy),
        "heldout_results_present_at_freeze": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.policy.write_text(
        json.dumps(jsonable(frozen), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Frozen {policy['feature']} <= {policy['max_value']} "
        f"sha256={frozen['policy_sha256']}\n"
    )


def run_confirm(args: argparse.Namespace) -> None:
    frozen = load_report(args.policy)
    policy = frozen["policy"]
    if canonical_sha256(policy) != frozen["policy_sha256"]:
        raise ValueError("Frozen association policy checksum mismatch")
    split_specs = {
        "three_offset1_same": case_specs(
            "three", REPORTS["three_k1"], True, False, args.max_cases
        ),
        "dance_same": case_specs("dance", REPORTS["dance"], True, False, args.max_cases),
        "dance_variable": case_specs(
            "dance", REPORTS["dance"], False, True, args.max_cases
        ),
        "box_same": case_specs("box", REPORTS["box"], True, False, args.max_cases),
        "box_variable": case_specs(
            "box", REPORTS["box"], False, True, args.max_cases
        ),
    }
    splits = {}
    for name, specs in split_specs.items():
        cases, summary = evaluate_specs(args, specs, policy, name)
        splits[name] = {"summary": summary, "cases": cases}

    all_wrong_rejected = all(
        value["summary"]["association_after_gate"]["wrong_accept_count"] == 0
        for value in splits.values()
    )
    exact = all(value["summary"]["rejected_post_exact_b0"] for value in splits.values())
    camera = all(value["summary"]["camera_max_abs_change"] <= 1e-12 for value in splits.values())
    def metric_no_worse(candidate: float, reference: float) -> bool:
        # One-person cuts have no pairwise-layout metric.  Two unavailable
        # values are equivalent rather than a regression; a one-sided missing
        # value remains a failure.
        if not np.isfinite(candidate) or not np.isfinite(reference):
            return bool(not np.isfinite(candidate) and not np.isfinite(reference))
        return bool(candidate <= reference + 1e-12)

    no_worse = all(
        all(
            metric_no_worse(
                value["summary"]["methods"]["dustbin_strict_fagd"][metric],
                value["summary"]["methods"]["strict_fagd"][metric],
            )
            for metric in PRIMARY
        )
        for value in splits.values()
    )
    coverage = min(
        float(
            value["summary"]["association_after_gate"][
                "coverage_over_correct_hungarian_edges"
            ]
        )
        for value in splits.values()
    )
    status = (
        "GO_ASSOCIATION_DUSTBIN_FOR_DEPLOYABLE_FAGD"
        if all_wrong_rejected and exact and camera and no_worse and coverage >= 0.95
        else "NO_GO_ASSOCIATION_DUSTBIN_AS_FROZEN_GENERAL_GATE"
    )
    report = {
        "experiment": "v14_frozen_observable_association_dustbin_confirmation",
        "policy_source": str(args.policy),
        "policy_sha256": frozen["policy_sha256"],
        "policy": policy,
        "protocol": {
            "heldout": (
                "MultiHuman three offset-1 same visibility; dance and box same/variable visibility"
            ),
            "gt_use": "edge labels and metrics only, after frozen candidate prediction",
            "future_frames": 0,
            "extra_pretrained_models": [],
            "camera_update": "none",
            "rejected_post_person": "bit-exact B0",
        },
        "splits": splits,
        "decision": {
            "all_wrong_edges_rejected": all_wrong_rejected,
            "all_rejected_people_exact_b0": exact,
            "camera_bit_exact_b0": camera,
            "all_metrics_no_worse_everywhere": no_worse,
            "minimum_correct_edge_coverage": coverage,
            "status": status,
        },
    }
    (args.output_dir / "confirmation_report.json").write_text(
        json.dumps(jsonable(report), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    text = confirmation_markdown(report)
    (args.output_dir / "confirmation_report.md").write_text(text, encoding="utf-8")
    print(text, end="")


def self_test() -> None:
    cost = np.asarray([[1.0, 3.0], [4.0, 1.0]], dtype=np.float64)
    matches = [(0, 0), (1, 1)]
    assert abs(second_assignment_margin(cost, matches) - 5.0) <= 1e-12
    row_alt, row_margin = edge_feature(cost, 0, 0, axis=1)
    assert row_alt == 3.0 and row_margin == 2.0
    policy = {
        "kind": "single_raw_absolute_max",
        "feature": "torso_deg",
        "max_value": 60.0,
    }
    assert gate_accepts({"torso_deg": 59.0}, policy)
    assert not gate_accepts({"torso_deg": 61.0}, policy)
    print("self-test passed")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    for path in (args.output_dir, args.policy):
        if not str(path.resolve()).startswith(str(REPO_ROOT.resolve())):
            raise ValueError("All output paths must stay under Movie3R on /data")
    if abs(float(args.alpha) - 0.9) > 1e-12:
        raise ValueError("This probe requires the already-frozen FAGD alpha=0.9")
    if args.phase == "dev":
        run_dev(args)
    elif args.phase == "freeze":
        run_freeze(args)
    else:
        run_confirm(args)


if __name__ == "__main__":
    main()
