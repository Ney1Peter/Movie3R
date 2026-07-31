#!/usr/bin/env python3
"""Evaluate strict full-one-to-one/all-accepted FAGD on CPU caches.

The alpha is reused from the already-frozen FAGD-0.9 policy.  This probe adds
no selection.  It evaluates the independent strict runtime on existing
same-visibility ``three offset1``/``dance``/``box`` and on all previously
generated variable-visibility cases for those sequences.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts", REPO_ROOT / "versions/v14"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from versions.v14 import probe_b0_brtc_huber_irls as harness  # noqa: E402
from versions.v14 import probe_b0_two_view_person_triangulation as legacy  # noqa: E402
from versions.v14 import probe_brtc_observable_action_shrinkage as common  # noqa: E402
from versions.v14 import probe_brtc_soft_completeness as variable_source  # noqa: E402
from versions.v14 import probe_brtc_variable_visibility as variable  # noqa: E402
from versions.v14.b0_person_triangulation import refine_matched_people  # noqa: E402
from versions.v14.b0_person_triangulation_strict_fagd import (  # noqa: E402
    StrictFAGDConfig,
    refine_matched_people_strict_fagd,
)
from versions.v14.probe_b0_anchor_conflict import SEQUENCE_INPUTS  # noqa: E402
from versions.v14.probe_b0_identity_matching import strict_cache  # noqa: E402
from versions.v13 import gt_id_consensus as geometry  # noqa: E402


DEFAULT_POLICY = (
    REPO_ROOT
    / "output/v14/fine_alignment_research/brtc_full_accept_group_damping/"
    "FROZEN_POLICY_BEFORE_HELDOUT.json"
)
DEFAULT_VARIABLE_ROOT = REPO_ROOT / "output/v14/fine_alignment_research/brtc_variable_visibility"
DEFAULT_OUTPUT = REPO_ROOT / "output/v14/fine_alignment_research/brtc_strict_deployable_fagd"
POINT_KEYS = ("root", "joints", "vertices")
PRIMARY = (
    "root_error_m",
    "joint_error_m",
    "vertex_error_m",
    "pairwise_distance_error_m",
    "pairwise_vector_error_m",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--variable_root", type=Path, default=DEFAULT_VARIABLE_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max_cases", type=int, default=0)
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


def callback_v1(pre_camera, post_camera, pre_people, post_people, matches):
    return refine_matched_people(pre_camera, post_camera, pre_people, post_people, matches)


def callback_strict(alpha: float) -> Callable[..., tuple[list[dict], dict]]:
    def callback(pre_camera, post_camera, pre_people, post_people, matches):
        return refine_matched_people_strict_fagd(
            pre_camera,
            post_camera,
            pre_people,
            post_people,
            matches,
            fagd_config=StrictFAGDConfig(alpha=alpha),
        )

    return callback


def evaluate_prepared_runtime(
    prepared: list[dict[str, Any]],
    callback: Callable[..., tuple[list[dict], dict]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    evaluated = []
    for case in prepared:
        pre_people = [
            {key: np.asarray(person["pre"][key], dtype=np.float64).copy() for key in POINT_KEYS}
            for person in case["people"]
        ]
        post_people = [
            {key: np.asarray(person["post"][key], dtype=np.float64).copy() for key in POINT_KEYS}
            for person in case["people"]
        ]
        matches = [(index, index) for index in range(len(case["people"]))]
        pre_camera = np.asarray(case["pre_camera"], dtype=np.float64)
        post_camera = np.asarray(case["post_camera"], dtype=np.float64)
        before_pre, before_post = pre_camera.copy(), post_camera.copy()
        corrected, debug = callback(
            pre_camera, post_camera, pre_people, post_people, matches
        )
        camera_change = max(
            float(np.max(np.abs(pre_camera - before_pre))),
            float(np.max(np.abs(post_camera - before_post))),
        )
        records = {int(row["post_index"]): row for row in debug["people"]}
        people, baseline_roots, corrected_roots = [], [], []
        fallback_change = 0.0
        for index, (prepared_person, source, result) in enumerate(
            zip(case["people"], post_people, corrected)
        ):
            record = records[index]
            baseline = harness.point_errors(source, prepared_person["target"], True)
            candidate = harness.point_errors(result, prepared_person["target"], True)
            root_delta = candidate["root_error_m"] - baseline["root_error_m"]
            shift = np.asarray(record["final_shift_world"], dtype=np.float64)
            if not bool(record["accepted"]):
                fallback_change = max(
                    fallback_change,
                    max(float(np.max(np.abs(result[key] - source[key]))) for key in POINT_KEYS),
                )
            evidence = record.get("evidence", {})
            action = float(
                np.dot(
                    np.asarray(record.get("individual_shift_world", shift), dtype=np.float64),
                    np.asarray(evidence.get("ray_world", np.zeros(3)), dtype=np.float64),
                )
            )
            oracle = float(prepared_person["oracle_ray_label_evaluation_only"])
            people.append(
                {
                    "identity_evaluation_only": prepared_person["identity"],
                    "association_correct_evaluation_only": prepared_person[
                        "association_correct_evaluation_only"
                    ],
                    "accepted": bool(record["accepted"]),
                    "action_m": action,
                    "sign_correct_evaluation_only": bool(np.sign(action) == np.sign(oracle)),
                    "evidence": {
                        "valid_count": int(evidence.get("valid_count", 0)),
                    },
                    "baseline": baseline,
                    "corrected": candidate,
                    "root_delta_m": root_delta,
                }
            )
            baseline_roots.append(source["root"])
            corrected_roots.append(result["root"])
        evaluated.append(
            {
                "sequence": case["sequence"],
                "case": case["case"],
                "people": people,
                "camera": {"candidate_max_abs_change": camera_change},
                "layout": {
                    "baseline": harness.case_layout(case, baseline_roots),
                    "corrected": harness.case_layout(case, corrected_roots),
                },
                "layout_consensus": {
                    "selected_residual_lambda": float(debug["selected_residual_lambda"]),
                },
                "exact_b0_fallback_max_abs_change": fallback_change,
                "runtime": {
                    key: debug.get(key)
                    for key in (
                        "matched_count",
                        "accepted_count",
                        "strict_full_one_to_one_all_accepted",
                        "exact_v1_fallback",
                        "group_damping_applied",
                    )
                },
            }
        )
    summary = harness.summarize(evaluated, full=True)
    summary["strict_gate_boundary_count"] = int(
        sum(bool(case["runtime"].get("strict_full_one_to_one_all_accepted")) for case in evaluated)
    )
    return evaluated, summary


def same_visibility(alpha: float, max_cases: int) -> dict[str, Any]:
    split_rows = {
        "confirm_three_offset1": harness.load_rows(
            "confirm", harness.DEFAULT_CONFIRM_REPORT, max_cases
        ),
        "dance": legacy.report_rows(("dance",)),
        "box": legacy.report_rows(("box",)),
    }
    if max_cases:
        split_rows = {name: rows[:max_cases] for name, rows in split_rows.items()}
    output = {}
    for split, rows in split_rows.items():
        prepared = common.prepared_rows(rows, 0)
        v1_cases, v1_summary = evaluate_prepared_runtime(prepared, callback_v1)
        candidate_cases, candidate_summary = evaluate_prepared_runtime(
            prepared, callback_strict(alpha)
        )
        output[split] = {
            "methods": {
                "brtc_v1": {
                    "metrics": common.compact(v1_summary),
                    "summary": v1_summary,
                    "cases": v1_cases,
                },
                "strict_fagd": {
                    "metrics": common.compact(candidate_summary),
                    "summary": candidate_summary,
                    "cases": candidate_cases,
                },
            }
        }
    return output


def exact_geometry(first: list[dict], second: list[dict]) -> tuple[bool, float]:
    maximum = 0.0
    exact = True
    for first_person, second_person in zip(first, second):
        for key in POINT_KEYS:
            a = np.asarray(first_person[key])
            b = np.asarray(second_person[key])
            exact = bool(exact and np.array_equal(a, b))
            maximum = max(maximum, float(np.max(np.abs(a - b))))
    return exact, maximum


def evaluate_variable_case(cache: dict[str, Any], boundary: np.ndarray, alpha: float) -> dict[str, Any]:
    pre_order = variable.ordered_people(cache["humans"][-2])
    post_order = variable.ordered_people(cache["humans"][-1])
    pre_people = [variable.person_geometry(person) for _, person in pre_order]
    post_people = [variable.person_geometry(person, boundary) for _, person in post_order]
    pre_camera = np.asarray(cache["poses"][-2], dtype=np.float64)
    post_camera = boundary @ np.asarray(cache["poses"][-1], dtype=np.float64)
    matches, association = variable.anonymous_matches(pre_order, post_order, boundary)
    v1, v1_debug = refine_matched_people(
        pre_camera, post_camera, pre_people, post_people, matches
    )
    candidate, debug = refine_matched_people_strict_fagd(
        pre_camera,
        post_camera,
        pre_people,
        post_people,
        matches,
        fagd_config=StrictFAGDConfig(alpha=alpha),
    )
    exact, max_delta = exact_geometry(v1, candidate)

    gt_pre = np.asarray(cache["gt"]["pre_c2w"], dtype=np.float64)
    gauge = pre_camera @ np.linalg.inv(gt_pre)
    targets = [
        variable.person_geometry(cache["gt"]["post_humans"][identity], gauge)
        for identity, _ in post_order
    ]
    predictions = {"b0": post_people, "brtc_v1": v1, "strict_fagd": candidate}
    base_roots = np.asarray(
        [
            legacy.point_errors(predicted, target)["root_error_m"]
            for predicted, target in zip(post_people, targets)
        ]
    )
    metrics = {}
    for name, people in predictions.items():
        errors = [legacy.point_errors(predicted, target) for predicted, target in zip(people, targets)]
        value = {
            key: variable.finite_mean([row[key] for row in errors])
            for key in ("root_error_m", "joint_error_m", "vertex_error_m")
        } | variable.layout_errors(people, targets)
        if name != "b0":
            roots = np.asarray([row["root_error_m"] for row in errors])
            delta = roots - base_roots
            value.update(
                {
                    "root_improve_rate": float(np.mean(delta < 0.0)),
                    "root_harm_over_1cm_rate": float(np.mean(delta > 0.01)),
                    "root_harm_over_5cm_rate": float(np.mean(delta > 0.05)),
                }
            )
        metrics[name] = value
    return {
        "case": cache["case"],
        "pre_person_count": len(pre_people),
        "post_person_count": len(post_people),
        "association": association,
        "runtime": {
            "matched_count": int(v1_debug["matched_count"]),
            "accepted_count": int(v1_debug["accepted_count"]),
            "strict_gate": bool(debug["strict_full_one_to_one_all_accepted"]),
            "exact_v1_fallback": bool(debug["exact_v1_fallback"]),
            "geometry_bit_exact_v1": exact,
            "geometry_max_abs_delta_v1": max_delta,
        },
        "metrics": metrics,
    }


def aggregate_variable(cases: list[dict[str, Any]], method: str) -> dict[str, float]:
    result = {
        key: variable.finite_mean([case["metrics"][method][key] for case in cases])
        for key in PRIMARY
    }
    if method != "b0":
        for key in (
            "root_improve_rate",
            "root_harm_over_1cm_rate",
            "root_harm_over_5cm_rate",
        ):
            result[key] = variable.finite_mean([case["metrics"][method][key] for case in cases])
    return result


def variable_visibility(alpha: float, variable_root: Path, max_cases: int) -> dict[str, Any]:
    report_paths = {
        "three": variable.DEFAULT_SOURCE_REPORT,
        "dance": REPO_ROOT / "output/v14/b0_identity_matching_extended/dance/v14_b0_identity_matching.json",
        "box": REPO_ROOT / "output/v14/b0_identity_matching_extended/box/v14_b0_identity_matching.json",
    }
    boundary_dirs = {
        "three": variable_root,
        "dance": variable_root / "dance",
        "box": variable_root / "box",
    }
    output = {}
    for sequence in ("three", "dance", "box"):
        geometry.IDENTITIES = geometry.SEQUENCE_IDENTITIES[sequence]
        names = variable_source.case_names(report_paths[sequence])
        if max_cases:
            names = names[:max_cases]
        cache_args = argparse.Namespace(
            data_root=Path("/data/wangzheng/iJCV-CODE/data/MultiHuman/Real-World-Capture/extracted"),
            size=512,
            sequence=sequence,
        )
        cases = []
        for index, name in enumerate(names, start=1):
            boundary_row = json.loads(
                (boundary_dirs[sequence] / "cases" / f"{name}.json").read_text(encoding="utf-8")
            )
            cache = strict_cache(cache_args, SEQUENCE_INPUTS[sequence]["cache"] / f"{name}.pt")
            cases.append(
                evaluate_variable_case(
                    cache,
                    np.asarray(boundary_row["learned_b0"], dtype=np.float64),
                    alpha,
                )
            )
            print(f"[variable {sequence} {index:02d}/{len(names):02d}] {name}", flush=True)
        changed = [case for case in cases if case["pre_person_count"] != case["post_person_count"]]
        equal = [case for case in cases if case["pre_person_count"] == case["post_person_count"]]
        output[sequence] = {
            "case_count": len(cases),
            "population_count_change_case_count": len(changed),
            "equal_count_identity_change_case_count": len(equal),
            "strict_gate_case_count": int(sum(case["runtime"]["strict_gate"] for case in cases)),
            "all_variable_bit_exact_v1": bool(all(case["runtime"]["geometry_bit_exact_v1"] for case in cases)),
            "population_count_change_bit_exact_v1": bool(
                all(case["runtime"]["geometry_bit_exact_v1"] for case in changed)
            ),
            "equal_count_bit_exact_v1": bool(
                all(case["runtime"]["geometry_bit_exact_v1"] for case in equal)
            ),
            "max_abs_delta_v1": float(
                max((case["runtime"]["geometry_max_abs_delta_v1"] for case in cases), default=0.0)
            ),
            "methods": {
                name: aggregate_variable(cases, name)
                for name in ("b0", "brtc_v1", "strict_fagd")
            },
            "cases": cases,
        }
    return output


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Strict deployable FAGD-0.9 CPU-cache evaluation",
        "",
        f"Frozen alpha: `{report['policy']['alpha']}`.",
        "",
        "## Same visibility",
        "",
        "| Split | Method | Root | Joint | Vertex | Pair distance | Pair vector | Harm >5cm | Gate cuts |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, split_value in report["same_visibility"].items():
        for method in ("brtc_v1", "strict_fagd"):
            value = split_value["methods"][method]
            metric = value["metrics"]
            gates = value["summary"].get("strict_gate_boundary_count", 0)
            lines.append(
                f"| {split} | {method} | {metric['root_error_m']:.6f} | "
                f"{metric['joint_error_m']:.6f} | {metric['vertex_error_m']:.6f} | "
                f"{metric['pairwise_distance_error_m']:.6f} | "
                f"{metric['pairwise_vector_error_m']:.6f} | "
                f"{metric['root_harm_over_5cm_rate']:.1%} | {gates} |"
            )
    lines.extend(
        [
            "",
            "## Variable visibility",
            "",
            "| Split | Cases | Count-change | Equal-count replacement | Gate cuts | All bit-exact v1 | Count-change bit-exact | Root v1 | Root strict | Pair vector v1 | Pair vector strict |",
            "|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|",
        ]
    )
    for split, value in report["variable_visibility"].items():
        v1, candidate = value["methods"]["brtc_v1"], value["methods"]["strict_fagd"]
        lines.append(
            f"| {split} | {value['case_count']} | {value['population_count_change_case_count']} | "
            f"{value['equal_count_identity_change_case_count']} | {value['strict_gate_case_count']} | "
            f"{value['all_variable_bit_exact_v1']} | "
            f"{value['population_count_change_bit_exact_v1']} | "
            f"{v1['root_error_m']:.6f} | {candidate['root_error_m']:.6f} | "
            f"{v1['pairwise_vector_error_m']:.6f} | {candidate['pairwise_vector_error_m']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"All same-visibility spatial gains preserved: `{report['decision']['same_visibility_spatial_gain_everywhere']}`.",
            f"All same-visibility layout invariant: `{report['decision']['same_visibility_layout_invariant_everywhere']}`.",
            f"All variable cases bit-exact v1: `{report['decision']['all_variable_bit_exact_v1']}`.",
            f"All population-count-change cases bit-exact v1: `{report['decision']['population_count_change_bit_exact_v1']}`.",
            f"Decision: **{report['decision']['status']}**.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    for path in (args.policy, args.variable_root, args.output_dir):
        if not str(path.resolve()).startswith(str(REPO_ROOT.resolve())):
            raise ValueError("All paths must remain under Movie3R on /data")
    frozen = json.loads(args.policy.read_text(encoding="utf-8"))
    policy = frozen["policy"]
    if common.canonical_sha256(policy) != frozen["policy_sha256"]:
        raise ValueError("Frozen FAGD policy checksum mismatch")
    alpha = float(policy["alpha"])
    if alpha != 0.9:
        raise ValueError(f"Expected frozen FAGD alpha 0.9, got {alpha}")
    original_load, _ = common.install_cached_torch_load()
    try:
        same = same_visibility(alpha, args.max_cases)
        visibility = variable_visibility(alpha, args.variable_root, args.max_cases)
    finally:
        torch.load = original_load

    same_spatial = all(
        all(
            split["methods"]["strict_fagd"]["metrics"][key]
            < split["methods"]["brtc_v1"]["metrics"][key] - 1e-12
            for key in ("root_error_m", "joint_error_m", "vertex_error_m")
        )
        for split in same.values()
    )
    same_layout = all(
        all(
            abs(
                split["methods"]["strict_fagd"]["metrics"][key]
                - split["methods"]["brtc_v1"]["metrics"][key]
            )
            <= 1e-12
            for key in ("pairwise_distance_error_m", "pairwise_vector_error_m")
        )
        for split in same.values()
    )
    all_variable_exact = all(value["all_variable_bit_exact_v1"] for value in visibility.values())
    count_change_exact = all(
        value["population_count_change_bit_exact_v1"] for value in visibility.values()
    )
    status = (
        "GO_STRICT_FAGD_GENERAL_VARIABLE_SAFE"
        if same_spatial and same_layout and all_variable_exact
        else "NO_GO_STRICT_FAGD_GENERAL_VARIABLE_SAFE"
    )
    report = {
        "experiment": "v14_strict_deployable_fagd_0p9",
        "policy_source": str(args.policy),
        "policy": policy,
        "strict_gate": (
            "accepted_count == matched_count == "
            "max(len(pre_people), len(post_people)) > 0"
        ),
        "same_visibility": same,
        "variable_visibility": visibility,
        "decision": {
            "same_visibility_spatial_gain_everywhere": same_spatial,
            "same_visibility_layout_invariant_everywhere": same_layout,
            "all_variable_bit_exact_v1": all_variable_exact,
            "population_count_change_bit_exact_v1": count_change_exact,
            "status": status,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "multihuman_report.json").write_text(
        json.dumps(jsonable(report), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    text = markdown(report)
    (args.output_dir / "multihuman_report.md").write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
