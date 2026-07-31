#!/usr/bin/env python3
"""Develop, freeze, and validate angular-safe shared-group BRTC damping.

Only ``three offset0`` is used to select an angular statistic and budget.
Frozen confirmation uses ``three offset1``, ``dance``, ``box``, and the
existing variable-visibility CPU caches.  Candidate decisions use no GT;
GT is read only by the existing evaluation harness after output is produced.
"""

from __future__ import annotations

import argparse
import hashlib
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

from versions.v13 import gt_id_consensus as geometry  # noqa: E402
from versions.v14 import probe_b0_brtc_huber_irls as harness  # noqa: E402
from versions.v14 import probe_b0_two_view_person_triangulation as legacy  # noqa: E402
from versions.v14 import probe_brtc_observable_action_shrinkage as common  # noqa: E402
from versions.v14 import probe_brtc_soft_completeness as variable_source  # noqa: E402
from versions.v14 import probe_brtc_strict_deployable_fagd as strict  # noqa: E402
from versions.v14 import probe_brtc_variable_visibility as variable  # noqa: E402
from versions.v14.b0_person_triangulation import refine_matched_people  # noqa: E402
from versions.v14.b0_person_triangulation_angular_safe_fagd import (  # noqa: E402
    AngularSafeFAGDConfig,
    DEFAULT_ALPHA_VALUES,
    refine_matched_people_angular_safe_fagd,
)
from versions.v14.probe_b0_anchor_conflict import SEQUENCE_INPUTS  # noqa: E402
from versions.v14.probe_b0_identity_matching import strict_cache  # noqa: E402


DEFAULT_OUTPUT = REPO_ROOT / "output/v14/fine_alignment_research/brtc_angular_safe_fagd"
DEFAULT_POLICY = DEFAULT_OUTPUT / "FROZEN_POLICY_BEFORE_HELDOUT.json"
DEFAULT_VARIABLE_ROOT = REPO_ROOT / "output/v14/fine_alignment_research/brtc_variable_visibility"
STATISTICS = ("core_median", "core_p90", "all_median", "all_p90")
BUDGETS_DEG = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0)
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
    parser.add_argument("--phase", required=True, choices=("dev", "freeze", "validate"))
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--variable_root", type=Path, default=DEFAULT_VARIABLE_ROOT)
    parser.add_argument("--max_cases", type=int, default=0)
    return parser.parse_args()


def runtime_callback(
    angular_config: AngularSafeFAGDConfig,
    audit: list[dict[str, Any]] | None = None,
) -> Callable[..., tuple[list[dict[str, Any]], dict[str, Any]]]:
    def callback(pre_camera, post_camera, pre_people, post_people, matches):
        corrected, debug = refine_matched_people_angular_safe_fagd(
            pre_camera,
            post_camera,
            pre_people,
            post_people,
            matches,
            angular_config=angular_config,
        )
        if audit is not None:
            audit.append(
                {
                    "previous_observable_count": int(debug["previous_observable_count"]),
                    "current_observable_count": int(debug["current_observable_count"]),
                    "matched_count": int(debug["matched_count"]),
                    "accepted_count": int(debug["accepted_count"]),
                    "strict_gate": bool(debug["strict_full_one_to_one_all_accepted"]),
                    "selected_alpha": float(debug["selected_group_alpha"]),
                    "angular_budget_satisfied": bool(debug["angular_budget_satisfied"]),
                    "angular_score_by_alpha_deg": debug["angular_score_by_alpha_deg"],
                    "damping_applied": bool(debug["angular_damping_applied"]),
                }
            )
        return corrected, debug

    return callback


def audit_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = tuple(float(value) for value in DEFAULT_ALPHA_VALUES)
    selected = [float(row["selected_alpha"]) for row in rows]
    finite_selected_scores = []
    for row in rows:
        score = row["angular_score_by_alpha_deg"].get(row["selected_alpha"])
        if score is not None and np.isfinite(score):
            finite_selected_scores.append(float(score))
    return {
        "case_count": len(rows),
        "strict_gate_count": int(sum(row["strict_gate"] for row in rows)),
        "damping_applied_count": int(sum(row["damping_applied"] for row in rows)),
        "budget_satisfied_count": int(sum(row["angular_budget_satisfied"] for row in rows)),
        "mean_selected_alpha": float(np.mean(selected)) if selected else 1.0,
        "alpha_counts": {
            str(alpha): int(sum(abs(value - alpha) <= 1e-12 for value in selected))
            for alpha in values
        },
        "mean_selected_angular_score_deg": (
            float(np.mean(finite_selected_scores)) if finite_selected_scores else None
        ),
        "rows": rows,
    }


def evaluate_prepared(
    prepared: list[dict[str, Any]],
    angular_config: AngularSafeFAGDConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    audit: list[dict[str, Any]] = []
    cases, summary = strict.evaluate_prepared_runtime(
        prepared, runtime_callback(angular_config, audit)
    )
    return cases, summary, audit_summary(audit)


def config_from_policy(policy: dict[str, Any]) -> AngularSafeFAGDConfig:
    return AngularSafeFAGDConfig(
        angular_budget_deg=float(policy["angular_budget_deg"]),
        statistic=str(policy["statistic"]),
        alpha_values=tuple(float(value) for value in policy["alpha_values"]),
    )


def safe_vs(reference: dict[str, float], candidate: dict[str, float]) -> bool:
    return bool(
        all(candidate[key] <= reference[key] + 1e-12 for key in PRIMARY)
        and candidate["root_harm_over_1cm_rate"]
        <= reference["root_harm_over_1cm_rate"] + 1e-12
        and candidate["root_harm_over_5cm_rate"]
        <= reference["root_harm_over_5cm_rate"] + 1e-12
        and candidate["coverage"] >= reference["coverage"] - 1e-12
    )


def layout_exact(reference: dict[str, float], candidate: dict[str, float]) -> bool:
    return bool(
        abs(candidate["pairwise_distance_error_m"] - reference["pairwise_distance_error_m"])
        <= 1e-12
        and abs(candidate["pairwise_vector_error_m"] - reference["pairwise_vector_error_m"])
        <= 1e-12
    )


def dev_markdown(report: dict[str, Any]) -> str:
    selected = report["selection"]
    lines = [
        "# Angular-safe FAGD development",
        "",
        "Selection data: `three offset0` only.",
        "",
        f"Selected statistic/budget: `{selected['statistic']}` / "
        f"`{selected['angular_budget_deg']} deg`.",
        f"Development passed: `{selected['passed']}`.",
        "",
        "| Method | Root | Joint | Vertex | Pair distance | Pair vector | Harm >1cm | Harm >5cm | Applied | Mean alpha |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("brtc_v1", "angular_safe"):
        value = report["selected_dev"][name]
        metric = value["metrics"]
        audit = value["audit"]
        lines.append(
            f"| {name} | {metric['root_error_m']:.6f} | "
            f"{metric['joint_error_m']:.6f} | {metric['vertex_error_m']:.6f} | "
            f"{metric['pairwise_distance_error_m']:.6f} | "
            f"{metric['pairwise_vector_error_m']:.6f} | "
            f"{metric['root_harm_over_1cm_rate']:.1%} | "
            f"{metric['root_harm_over_5cm_rate']:.1%} | "
            f"{audit['damping_applied_count']} | {audit['mean_selected_alpha']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Eligible safe policies: `{selected['eligible_policy_count']}`.",
            f"Layout invariant: `{selected['layout_invariant']}`.",
            "Held-out data was not loaded by this phase.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_dev(args: argparse.Namespace) -> None:
    rows = harness.load_rows("dev", harness.DEFAULT_CONFIRM_REPORT, args.max_cases)
    prepared = common.prepared_rows(rows, 0)
    v1_cases, v1_summary = strict.evaluate_prepared_runtime(prepared, strict.callback_v1)
    v1_metric = common.compact(v1_summary)
    v1_audit = {
        "case_count": len(prepared),
        "strict_gate_count": 0,
        "damping_applied_count": 0,
        "budget_satisfied_count": 0,
        "mean_selected_alpha": 1.0,
        "alpha_counts": {"1.0": len(prepared)},
        "mean_selected_angular_score_deg": None,
        "rows": [],
    }
    scan = []
    full_by_key = {}
    for statistic in STATISTICS:
        for budget in BUDGETS_DEG:
            config = AngularSafeFAGDConfig(
                angular_budget_deg=budget,
                statistic=statistic,
            )
            cases, summary, audit = evaluate_prepared(prepared, config)
            metric = common.compact(summary)
            key = (statistic, float(budget))
            safe = safe_vs(v1_metric, metric)
            invariant = layout_exact(v1_metric, metric)
            spatial = all(
                metric[name] < v1_metric[name] - 1e-12
                for name in ("root_error_m", "joint_error_m", "vertex_error_m")
            )
            row = {
                "statistic": statistic,
                "angular_budget_deg": float(budget),
                "metrics": metric,
                "audit": {key: value for key, value in audit.items() if key != "rows"},
                "safe_vs_v1": safe,
                "layout_invariant": invariant,
                "spatial_strict_improve": spatial,
                "nontrivial": audit["damping_applied_count"] > 0,
            }
            scan.append(row)
            full_by_key[key] = {"cases": cases, "summary": summary, "audit": audit}
    eligible = [
        row
        for row in scan
        if row["safe_vs_v1"]
        and row["layout_invariant"]
        and row["spatial_strict_improve"]
        and row["nontrivial"]
    ]
    ranked = eligible if eligible else scan
    selected = min(
        ranked,
        key=lambda row: (
            not row["safe_vs_v1"],
            row["metrics"]["root_error_m"],
            row["metrics"]["joint_error_m"],
            row["metrics"]["vertex_error_m"],
            -row["audit"]["mean_selected_alpha"],
            row["statistic"],
            row["angular_budget_deg"],
        ),
    )
    key = (selected["statistic"], selected["angular_budget_deg"])
    full = full_by_key[key]
    passed = bool(selected in eligible)
    report = {
        "experiment": "v14_brtc_angular_safe_fagd",
        "phase": "development_before_freeze",
        "protocol": {
            "development": "three offset0 only",
            "statistics": STATISTICS,
            "budgets_deg": BUDGETS_DEG,
            "alpha_values": DEFAULT_ALPHA_VALUES,
            "strict_gate": (
                "accepted_count == matched_count == "
                "max(len(pre_people), len(post_people)) > 0"
            ),
            "candidate_gt_use": "none",
            "future_frames": 0,
            "camera_update": "none",
        },
        "scan": scan,
        "selection": {
            "rule": (
                "all five errors/coverage/harm no worse than v1; root/joint/vertex "
                "strictly improve; nontrivial damping; then minimize root/joint/vertex"
            ),
            "statistic": selected["statistic"],
            "angular_budget_deg": selected["angular_budget_deg"],
            "eligible_policy_count": len(eligible),
            "layout_invariant": selected["layout_invariant"],
            "passed": passed,
        },
        "selected_dev": {
            "brtc_v1": {
                "metrics": v1_metric,
                "summary": v1_summary,
                "cases": v1_cases,
                "audit": v1_audit,
            },
            "angular_safe": {
                "metrics": selected["metrics"],
                "summary": full["summary"],
                "cases": full["cases"],
                "audit": full["audit"],
            },
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "DEV_SCAN.json").write_text(
        json.dumps(common.jsonable(report), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    text = dev_markdown(report)
    (args.output_dir / "DEV_RESULTS.md").write_text(text, encoding="utf-8")
    print(text, end="")


def run_freeze(args: argparse.Namespace) -> None:
    source = args.output_dir / "DEV_SCAN.json"
    dev = json.loads(source.read_text(encoding="utf-8"))
    if not bool(dev["selection"]["passed"]):
        raise RuntimeError("Angular-safe development policy did not pass")
    policy = {
        "statistic": str(dev["selection"]["statistic"]),
        "angular_budget_deg": float(dev["selection"]["angular_budget_deg"]),
        "alpha_values": [float(value) for value in DEFAULT_ALPHA_VALUES],
        "selection": "largest alpha within budget; otherwise minimum observed angle",
        "gate": (
            "accepted_count == matched_count == "
            "max(len(pre_people), len(post_people)) > 0"
        ),
        "application": "scale shared group only; keep frozen individual residual exact",
    }
    frozen = {
        "experiment": dev["experiment"],
        "status": "frozen_before_offset1_dance_box_variable_egohumans",
        "policy": policy,
        "policy_sha256": common.canonical_sha256(policy),
        "source_dev_report": str(source),
        "source_dev_report_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "constraints": {
            "future_frames": 0,
            "extra_pretrained_models": [],
            "candidate_gt_use": "none",
            "camera_update": "none",
        },
    }
    args.policy.parent.mkdir(parents=True, exist_ok=True)
    args.policy.write_text(
        json.dumps(common.jsonable(frozen), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(frozen, indent=2, ensure_ascii=False))


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


def evaluate_variable_case(
    cache: dict[str, Any],
    boundary: np.ndarray,
    angular_config: AngularSafeFAGDConfig,
) -> dict[str, Any]:
    pre_order = variable.ordered_people(cache["humans"][-2])
    post_order = variable.ordered_people(cache["humans"][-1])
    pre_people = [variable.person_geometry(person) for _, person in pre_order]
    post_people = [variable.person_geometry(person, boundary) for _, person in post_order]
    pre_camera = np.asarray(cache["poses"][-2], dtype=np.float64)
    post_camera = boundary @ np.asarray(cache["poses"][-1], dtype=np.float64)
    matches, association = variable.anonymous_matches(pre_order, post_order, boundary)
    v1, _ = refine_matched_people(
        pre_camera, post_camera, pre_people, post_people, matches
    )
    candidate, debug = refine_matched_people_angular_safe_fagd(
        pre_camera,
        post_camera,
        pre_people,
        post_people,
        matches,
        angular_config=angular_config,
    )
    exact, max_delta = exact_geometry(v1, candidate)
    gt_pre = np.asarray(cache["gt"]["pre_c2w"], dtype=np.float64)
    gauge = pre_camera @ np.linalg.inv(gt_pre)
    targets = [
        variable.person_geometry(cache["gt"]["post_humans"][identity], gauge)
        for identity, _ in post_order
    ]
    predictions = {"brtc_v1": v1, "angular_safe": candidate}
    metrics = {}
    for name, people in predictions.items():
        errors = [
            legacy.point_errors(predicted, target)
            for predicted, target in zip(people, targets)
        ]
        metrics[name] = {
            key: variable.finite_mean([row[key] for row in errors])
            for key in ("root_error_m", "joint_error_m", "vertex_error_m")
        } | variable.layout_errors(people, targets)
    return {
        "case": cache["case"],
        "pre_person_count": len(pre_people),
        "post_person_count": len(post_people),
        "association": association,
        "runtime": {
            "matched_count": int(debug["matched_count"]),
            "accepted_count": int(debug["accepted_count"]),
            "strict_gate": bool(debug["strict_full_one_to_one_all_accepted"]),
            "selected_alpha": float(debug["selected_group_alpha"]),
            "damping_applied": bool(debug["angular_damping_applied"]),
            "geometry_bit_exact_v1": exact,
            "geometry_max_abs_delta_v1": max_delta,
        },
        "metrics": metrics,
    }


def aggregate_variable(cases: list[dict[str, Any]], method: str) -> dict[str, float]:
    return {
        key: variable.finite_mean([case["metrics"][method][key] for case in cases])
        for key in PRIMARY
    }


def variable_validation(
    angular_config: AngularSafeFAGDConfig,
    variable_root: Path,
    max_cases: int,
) -> dict[str, Any]:
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
                (boundary_dirs[sequence] / "cases" / f"{name}.json").read_text(
                    encoding="utf-8"
                )
            )
            cache = strict_cache(
                cache_args, SEQUENCE_INPUTS[sequence]["cache"] / f"{name}.pt"
            )
            cases.append(
                evaluate_variable_case(
                    cache,
                    np.asarray(boundary_row["learned_b0"], dtype=np.float64),
                    angular_config,
                )
            )
            print(f"[variable {sequence} {index:02d}/{len(names):02d}] {name}", flush=True)
        changed = [
            case
            for case in cases
            if case["pre_person_count"] != case["post_person_count"]
        ]
        equal = [
            case
            for case in cases
            if case["pre_person_count"] == case["post_person_count"]
        ]
        output[sequence] = {
            "case_count": len(cases),
            "population_count_change_case_count": len(changed),
            "equal_count_identity_change_case_count": len(equal),
            "strict_gate_case_count": int(sum(case["runtime"]["strict_gate"] for case in cases)),
            "damping_case_count": int(sum(case["runtime"]["damping_applied"] for case in cases)),
            "all_variable_bit_exact_v1": bool(
                all(case["runtime"]["geometry_bit_exact_v1"] for case in cases)
            ),
            "population_count_change_bit_exact_v1": bool(
                all(case["runtime"]["geometry_bit_exact_v1"] for case in changed)
            ),
            "equal_count_bit_exact_v1": bool(
                all(case["runtime"]["geometry_bit_exact_v1"] for case in equal)
            ),
            "max_abs_delta_v1": float(
                max(
                    (case["runtime"]["geometry_max_abs_delta_v1"] for case in cases),
                    default=0.0,
                )
            ),
            "methods": {
                name: aggregate_variable(cases, name)
                for name in ("brtc_v1", "angular_safe")
            },
            "cases": cases,
        }
    return output


def validation_markdown(report: dict[str, Any]) -> str:
    policy = report["policy"]
    lines = [
        "# Frozen angular-safe FAGD held-out validation",
        "",
        f"Policy: `{policy['statistic']}`, budget `{policy['angular_budget_deg']} deg`.",
        "",
        "| Split | Method | Root | Joint | Vertex | Pair distance | Pair vector | Harm >5cm | Applied | Mean alpha |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, value in report["same_visibility"].items():
        for name in ("brtc_v1", "angular_safe"):
            method = value["methods"][name]
            metric = method["metrics"]
            audit = method["audit"]
            lines.append(
                f"| {split} | {name} | {metric['root_error_m']:.6f} | "
                f"{metric['joint_error_m']:.6f} | {metric['vertex_error_m']:.6f} | "
                f"{metric['pairwise_distance_error_m']:.6f} | "
                f"{metric['pairwise_vector_error_m']:.6f} | "
                f"{metric['root_harm_over_5cm_rate']:.1%} | "
                f"{audit['damping_applied_count']} | {audit['mean_selected_alpha']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Variable visibility",
            "",
            "| Split | Cases | Count-change | Equal-count replacement | Damped | All exact v1 | Count-change exact v1 |",
            "|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for split, value in report["variable_visibility"].items():
        lines.append(
            f"| {split} | {value['case_count']} | "
            f"{value['population_count_change_case_count']} | "
            f"{value['equal_count_identity_change_case_count']} | "
            f"{value['damping_case_count']} | {value['all_variable_bit_exact_v1']} | "
            f"{value['population_count_change_bit_exact_v1']} |"
        )
    lines.extend(
        [
            "",
            f"Spatial improves everywhere: `{report['decision']['same_visibility_spatial_improves']}`.",
            f"Layout invariant everywhere: `{report['decision']['same_visibility_layout_invariant']}`.",
            f"Harm non-regression everywhere: `{report['decision']['same_visibility_harm_safe']}`.",
            f"Count-change exact-v1 everywhere: `{report['decision']['population_count_change_exact_v1']}`.",
            f"Decision before Ego: **{report['decision']['status']}**.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_validate(args: argparse.Namespace) -> None:
    frozen = json.loads(args.policy.read_text(encoding="utf-8"))
    policy = frozen["policy"]
    if common.canonical_sha256(policy) != frozen["policy_sha256"]:
        raise ValueError("Frozen angular-safe policy checksum mismatch")
    angular_config = config_from_policy(policy)
    split_rows = {
        "confirm_three_offset1": harness.load_rows(
            "confirm", harness.DEFAULT_CONFIRM_REPORT, args.max_cases
        ),
        "dance": legacy.report_rows(("dance",)),
        "box": legacy.report_rows(("box",)),
    }
    if args.max_cases:
        split_rows = {name: rows[: args.max_cases] for name, rows in split_rows.items()}
    same = {}
    for split, rows in split_rows.items():
        prepared = common.prepared_rows(rows, 0)
        v1_cases, v1_summary = strict.evaluate_prepared_runtime(
            prepared, strict.callback_v1
        )
        candidate_cases, candidate_summary, candidate_audit = evaluate_prepared(
            prepared, angular_config
        )
        same[split] = {
            "methods": {
                "brtc_v1": {
                    "metrics": common.compact(v1_summary),
                    "summary": v1_summary,
                    "cases": v1_cases,
                    "audit": {
                        "damping_applied_count": 0,
                        "mean_selected_alpha": 1.0,
                    },
                },
                "angular_safe": {
                    "metrics": common.compact(candidate_summary),
                    "summary": candidate_summary,
                    "cases": candidate_cases,
                    "audit": candidate_audit,
                },
            }
        }
    visibility = variable_validation(
        angular_config, args.variable_root, args.max_cases
    )
    spatial = all(
        all(
            value["methods"]["angular_safe"]["metrics"][key]
            < value["methods"]["brtc_v1"]["metrics"][key] - 1e-12
            for key in ("root_error_m", "joint_error_m", "vertex_error_m")
        )
        for value in same.values()
    )
    layout = all(
        layout_exact(
            value["methods"]["brtc_v1"]["metrics"],
            value["methods"]["angular_safe"]["metrics"],
        )
        for value in same.values()
    )
    harm = all(
        value["methods"]["angular_safe"]["metrics"]["root_harm_over_5cm_rate"]
        <= value["methods"]["brtc_v1"]["metrics"]["root_harm_over_5cm_rate"] + 1e-12
        for value in same.values()
    )
    count_exact = all(
        value["population_count_change_bit_exact_v1"] for value in visibility.values()
    )
    status = (
        "GO_ANGULAR_SAFE_FAGD_TO_EGO"
        if spatial and layout and harm and count_exact
        else "NO_GO_ANGULAR_SAFE_FAGD_HELDOUT"
    )
    report = {
        "experiment": "v14_brtc_angular_safe_fagd",
        "phase": "heldout_after_policy_freeze",
        "policy_source": str(args.policy),
        "policy": policy,
        "policy_sha256": frozen["policy_sha256"],
        "same_visibility": same,
        "variable_visibility": visibility,
        "decision": {
            "same_visibility_spatial_improves": spatial,
            "same_visibility_layout_invariant": layout,
            "same_visibility_harm_safe": harm,
            "population_count_change_exact_v1": count_exact,
            "status": status,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "HELDOUT_RESULTS.json").write_text(
        json.dumps(common.jsonable(report), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    text = validation_markdown(report)
    (args.output_dir / "HELDOUT_RESULTS.md").write_text(text, encoding="utf-8")
    print(text, end="")


def main() -> None:
    args = parse_args()
    for path in (args.output_dir, args.policy, args.variable_root):
        if not str(path.resolve()).startswith(str(REPO_ROOT.resolve())):
            raise ValueError("All paths must remain in Movie3R under /data")
    original_load, _ = common.install_cached_torch_load()
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

