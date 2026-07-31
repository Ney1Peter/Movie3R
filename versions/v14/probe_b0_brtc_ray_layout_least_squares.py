#!/usr/bin/env python3
"""Freeze and evaluate joint ray-space layout least squares for BRTC-LC.

Development scans only the ridge prior weight on the existing ``three
offset0`` CPU geometry caches and writes a frozen policy before any
confirmation rows are loaded.  Confirmation then evaluates that immutable
weight on ``three offset1`` and the existing ``dance``/``box`` caches.

Ground truth is evaluator-only.  The runtime candidate consumes only frozen-B0
cameras, matched pre/post predicted people, frozen BRTC ray proposals/gates and
predicted pre-root pair vectors.  It does not load images or pretrained models.
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


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts", REPO_ROOT / "versions/v14"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from versions.v14.b0_person_triangulation import (  # noqa: E402
    DEFAULT_CONFIG,
    refine_matched_people,
)
from versions.v14.b0_person_triangulation_ray_layout_least_squares import (  # noqa: E402
    RayLayoutLeastSquaresConfig,
    refine_matched_people_ray_layout_least_squares,
)
from versions.v14 import probe_b0_brtc_huber_irls as helpers  # noqa: E402
from versions.v14 import probe_b0_two_view_person_triangulation as legacy  # noqa: E402


DEFAULT_OUTPUT = REPO_ROOT / "output/v14/fine_alignment_research/b0_brtc_ray_layout_ls"
DEFAULT_POLICY = DEFAULT_OUTPUT / "FROZEN_RAY_LAYOUT_LS_POLICY_BEFORE_CONFIRM.json"
DEFAULT_CONFIRM_REPORT = (
    REPO_ROOT
    / "output/v14/b0_identity_matching_offset1_confirm/v14_b0_identity_matching.json"
)
PRIOR_GRID = (0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
PERSON_METRICS = ("root_error_m", "joint_error_m", "vertex_error_m")
LAYOUT_METRICS = ("pairwise_distance_error_m", "pairwise_vector_error_m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("dev", "confirm"), required=True)
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
        return None
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def finite_stats(values: list[float]) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {key: None for key in ("mean", "median", "p90", "p95", "max")}
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def copied_geometry(person: dict[str, np.ndarray], full: bool) -> dict[str, np.ndarray]:
    keys = ("root", "joints", "vertices") if full else ("root", "joints")
    return {key: np.asarray(person[key], dtype=np.float64).copy() for key in keys}


def evaluate_method(
    prepared_cases: list[dict[str, Any]],
    method_name: str,
    full: bool,
    prior_weight: float | None = None,
) -> list[dict[str, Any]]:
    cases = []
    for case in prepared_cases:
        pre_people = [copied_geometry(person["pre"], full) for person in case["people"]]
        post_people = [copied_geometry(person["post"], full) for person in case["people"]]
        matches = [(index, index) for index in range(len(case["people"]))]
        pre_camera = np.asarray(case["pre_camera"], dtype=np.float64)
        post_camera = np.asarray(case["post_camera"], dtype=np.float64)
        pre_camera_before, post_camera_before = pre_camera.copy(), post_camera.copy()

        if method_name == "b0":
            corrected = [copied_geometry(person, full) for person in post_people]
            records = [
                {
                    "pre_index": index,
                    "post_index": index,
                    "accepted": False,
                    "final_shift_world": np.zeros(3, dtype=np.float64),
                }
                for index in range(len(case["people"]))
            ]
            debug = {
                "camera_update": "none",
                "accepted_count": 0,
                "people": records,
            }
        elif method_name == "brtc_lc_v1":
            corrected, debug = refine_matched_people(
                pre_camera,
                post_camera,
                pre_people,
                post_people,
                matches,
                DEFAULT_CONFIG,
            )
        elif method_name == "ray_layout_ls":
            if prior_weight is None:
                raise ValueError("ray_layout_ls requires prior_weight")
            corrected, debug = refine_matched_people_ray_layout_least_squares(
                pre_camera,
                post_camera,
                pre_people,
                post_people,
                matches,
                DEFAULT_CONFIG,
                RayLayoutLeastSquaresConfig(prior_weight=float(prior_weight)),
            )
        else:
            raise ValueError(f"Unknown method: {method_name}")

        camera_change = max(
            float(np.max(np.abs(pre_camera - pre_camera_before))),
            float(np.max(np.abs(post_camera - post_camera_before))),
        )
        record_by_post = {int(record["post_index"]): record for record in debug["people"]}
        people = []
        baseline_roots, corrected_roots = [], []
        fallback_max_abs_change = 0.0
        for index, (source, result, prepared_person) in enumerate(
            zip(post_people, corrected, case["people"])
        ):
            record = record_by_post[index]
            baseline_error = helpers.point_errors(source, prepared_person["target"], full)
            corrected_error = helpers.point_errors(result, prepared_person["target"], full)
            root_delta = corrected_error["root_error_m"] - baseline_error["root_error_m"]
            shift = np.asarray(record["final_shift_world"], dtype=np.float64)
            if not bool(record["accepted"]):
                for key in source:
                    fallback_max_abs_change = max(
                        fallback_max_abs_change,
                        float(np.max(np.abs(result[key] - source[key]))),
                    )
            action = float(
                record.get(
                    "final_action_m",
                    np.linalg.norm(shift),
                )
            )
            people.append(
                {
                    "identity_evaluation_only": prepared_person["identity"],
                    "association_correct_evaluation_only": prepared_person[
                        "association_correct_evaluation_only"
                    ],
                    "accepted": bool(record["accepted"]),
                    "action_m": action,
                    "final_shift_world": shift,
                    "baseline": baseline_error,
                    "corrected": corrected_error,
                    "root_delta_m": root_delta,
                }
            )
            baseline_roots.append(source["root"])
            corrected_roots.append(result["root"])
        case_output = {
            "sequence": case["sequence"],
            "case": case["case"],
            "association_source": case["association_source"],
            "camera_max_abs_change": camera_change,
            "people": people,
            "layout": {
                "baseline": helpers.case_layout(case, baseline_roots),
                "corrected": helpers.case_layout(case, corrected_roots),
            },
            "fallback_max_abs_change": fallback_max_abs_change,
        }
        if method_name == "ray_layout_ls":
            case_output["solver"] = {
                key: debug[key]
                for key in (
                    "prior_weight",
                    "constrained_pair_count",
                    "variable_count",
                    "matrix_rank",
                    "condition_number",
                    "clipped_action_count",
                    "observable_layout_objective_before",
                    "observable_layout_objective_after",
                )
            }
        cases.append(case_output)
    return cases


def summarize(cases: list[dict[str, Any]], method_name: str, full: bool) -> dict[str, Any]:
    people = [person for case in cases for person in case["people"]]
    accepted = [person for person in people if person["accepted"]]
    metrics = PERSON_METRICS if full else ("root_error_m",)
    output: dict[str, Any] = {
        "case_count": len(cases),
        "person_count": len(people),
        "coverage": None if method_name == "b0" else float(len(accepted) / len(people)),
        "accepted_count": None if method_name == "b0" else len(accepted),
        "association_accuracy": float(
            np.mean([person["association_correct_evaluation_only"] for person in people])
        ),
        "root_improve_rate": float(np.mean([person["root_delta_m"] < 0.0 for person in people])),
        "root_harm_over_1cm_rate": float(
            np.mean([person["root_delta_m"] > 0.01 for person in people])
        ),
        "root_harm_over_5cm_rate": float(
            np.mean([person["root_delta_m"] > 0.05 for person in people])
        ),
        "camera_max_abs_change": float(
            max((case["camera_max_abs_change"] for case in cases), default=0.0)
        ),
        "fallback_max_abs_change": float(
            max((case["fallback_max_abs_change"] for case in cases), default=0.0)
        ),
        "metrics": {
            metric: finite_stats(
                [float(person["corrected"][metric]) for person in people]
            )
            for metric in metrics
        },
        "action_abs_m": finite_stats([abs(float(person["action_m"])) for person in accepted]),
    }
    for metric in LAYOUT_METRICS:
        output["metrics"][metric] = finite_stats(
            [float(case["layout"]["corrected"][metric]) for case in cases]
        )
    if method_name == "ray_layout_ls":
        output["solver"] = {
            "clipped_action_count": int(
                sum(case["solver"]["clipped_action_count"] for case in cases)
            ),
            "condition_number": finite_stats(
                [float(case["solver"]["condition_number"]) for case in cases]
            ),
            "observable_layout_objective_before": finite_stats(
                [float(case["solver"]["observable_layout_objective_before"]) for case in cases]
            ),
            "observable_layout_objective_after": finite_stats(
                [float(case["solver"]["observable_layout_objective_after"]) for case in cases]
            ),
        }
    return output


def evaluate_summaries(
    prepared: list[dict[str, Any]], prior_weight: float, full: bool
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    cases_by_method = {
        "b0": evaluate_method(prepared, "b0", full),
        "brtc_lc_v1": evaluate_method(prepared, "brtc_lc_v1", full),
        "ray_layout_ls": evaluate_method(
            prepared, "ray_layout_ls", full, prior_weight=prior_weight
        ),
    }
    summaries = {
        name: summarize(cases, name, full) for name, cases in cases_by_method.items()
    }
    return summaries, cases_by_method


def mean(summary: dict[str, Any], metric: str) -> float:
    return float(summary["metrics"][metric]["mean"])


def candidate_not_worse_all(candidate: dict[str, Any], reference: dict[str, Any]) -> bool:
    return all(
        mean(candidate, metric) <= mean(reference, metric) + 1e-12
        for metric in PERSON_METRICS + LAYOUT_METRICS
    )


def markdown_table(title: str, methods: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| Method | Coverage | Root | Joint | Vertex | Pair distance | Pair vector | Improve | Harm >1cm | Harm >5cm |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("b0", "brtc_lc_v1", "ray_layout_ls"):
        value = methods[name]
        coverage = "-" if value["coverage"] is None else f"{value['coverage']:.1%}"
        lines.append(
            f"| {name} | {coverage} | {mean(value, 'root_error_m'):.6f} | "
            f"{mean(value, 'joint_error_m'):.6f} | {mean(value, 'vertex_error_m'):.6f} | "
            f"{mean(value, 'pairwise_distance_error_m'):.6f} | "
            f"{mean(value, 'pairwise_vector_error_m'):.6f} | "
            f"{value['root_improve_rate']:.1%} | {value['root_harm_over_1cm_rate']:.1%} | "
            f"{value['root_harm_over_5cm_rate']:.1%} |"
        )
    lines.append("")
    return lines


def run_dev(args: argparse.Namespace) -> None:
    rows = helpers.load_rows("dev", args.confirm_report, args.max_cases)
    prepared = helpers.prepare_all(rows)
    brtc_cases = evaluate_method(prepared, "brtc_lc_v1", full=True)
    brtc_summary = summarize(brtc_cases, "brtc_lc_v1", full=True)
    scan = []
    for prior_weight in PRIOR_GRID:
        candidate_cases = evaluate_method(
            prepared,
            "ray_layout_ls",
            full=True,
            prior_weight=prior_weight,
        )
        candidate_summary = summarize(candidate_cases, "ray_layout_ls", full=True)
        scan.append(
            {
                "prior_weight": prior_weight,
                "summary": candidate_summary,
            }
        )
        print(
            f"[prior={prior_weight:g}] root={mean(candidate_summary, 'root_error_m'):.6f} "
            f"pair_vector={mean(candidate_summary, 'pairwise_vector_error_m'):.6f} "
            f"harm5={candidate_summary['root_harm_over_5cm_rate']:.1%}",
            flush=True,
        )

    safety = {
        "coverage_exact_brtc": True,
        "root_harm_over_5cm_max": 0.10,
        "pair_distance_max_relative_to_brtc": 1.02,
        "pair_vector_max_relative_to_brtc": 1.02,
        "fallback_max_abs_change": 0.0,
        "camera_max_abs_change": 0.0,
    }
    eligible = [
        row
        for row in scan
        if row["summary"]["coverage"] == brtc_summary["coverage"]
        and row["summary"]["root_harm_over_5cm_rate"] <= safety["root_harm_over_5cm_max"]
        and mean(row["summary"], "pairwise_distance_error_m")
        <= mean(brtc_summary, "pairwise_distance_error_m")
        * safety["pair_distance_max_relative_to_brtc"]
        and mean(row["summary"], "pairwise_vector_error_m")
        <= mean(brtc_summary, "pairwise_vector_error_m")
        * safety["pair_vector_max_relative_to_brtc"]
        and row["summary"]["fallback_max_abs_change"] == 0.0
        and row["summary"]["camera_max_abs_change"] == 0.0
    ]
    selection_pool = eligible if eligible else scan
    selected = min(
        selection_pool,
        key=lambda row: (
            mean(row["summary"], "root_error_m"),
            mean(row["summary"], "joint_error_m"),
            mean(row["summary"], "vertex_error_m"),
            row["summary"]["root_harm_over_5cm_rate"],
            float(row["prior_weight"]),
        ),
    )
    selected_weight = float(selected["prior_weight"])
    dev_candidate = selected["summary"]
    dev_go = bool(
        candidate_not_worse_all(dev_candidate, brtc_summary)
        and mean(dev_candidate, "root_error_m") < mean(brtc_summary, "root_error_m")
        and dev_candidate["root_harm_over_5cm_rate"]
        <= brtc_summary["root_harm_over_5cm_rate"]
        and dev_candidate["coverage"] == brtc_summary["coverage"]
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_path = output_dir / "FROZEN_RAY_LAYOUT_LS_POLICY_BEFORE_CONFIRM.json"
    policy = {
        "protocol": {
            "selection_split": "three offset0 only (41 cuts / 122 matched people)",
            "confirmation_splits_unopened_by_dev": "three offset1 and dance/box",
            "runtime": "joint scalar actions along each accepted post-camera ray",
            "layout_evidence": "predicted pre-root pair vectors",
            "prior": "frozen BRTC individual gated/capped action",
            "camera": "bit-exact frozen B0",
            "rejected_unmatched": "exact no-op",
            "future_or_gt_candidate_use": "none",
            "pretrained_model": "none; CPU cache only",
        },
        "prior_grid": PRIOR_GRID,
        "selection_rule": (
            "among development-safety-eligible weights, minimize root mean; then joint, "
            "vertex, harm>5cm and prior weight; if none eligible freeze the same ordering "
            "over all weights for diagnostic confirmation"
        ),
        "development_safety": safety,
        "eligible_count": len(eligible),
        "selected_prior_weight": selected_weight,
        "dev_go": dev_go,
        "dev_brtc_summary": brtc_summary,
        "dev_selected_summary": dev_candidate,
    }
    policy_path.write_text(
        json.dumps(jsonable(policy), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    policy_sha = sha256(policy_path)
    report = {
        "experiment": "v14_brtc_joint_ray_space_layout_least_squares",
        "phase": "development_and_freeze",
        "policy_source": str(policy_path),
        "policy_sha256": policy_sha,
        "policy_mtime_ns": policy_path.stat().st_mtime_ns,
        "selected_prior_weight": selected_weight,
        "eligible_count": len(eligible),
        "dev_go": dev_go,
        "methods": {
            "brtc_lc_v1": {"summary": brtc_summary, "cases": brtc_cases},
            "ray_layout_ls": {
                "summary": dev_candidate,
                "cases": evaluate_method(
                    prepared,
                    "ray_layout_ls",
                    full=True,
                    prior_weight=selected_weight,
                ),
            },
        },
        "prior_scan": scan,
    }
    (output_dir / "dev_report.json").write_text(
        json.dumps(jsonable(report), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Joint ray-space layout least squares: development freeze",
        "",
        f"- Selected prior weight: `{selected_weight:g}`.",
        f"- Eligible weights: `{len(eligible)}/{len(scan)}`.",
        f"- Development GO: `{dev_go}`.",
        f"- Frozen policy SHA256: `{policy_sha}`.",
        "",
        "| Prior | Root | Joint | Vertex | Pair distance | Pair vector | Harm >5cm |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in scan:
        value = row["summary"]
        lines.append(
            f"| {row['prior_weight']:g} | {mean(value, 'root_error_m'):.6f} | "
            f"{mean(value, 'joint_error_m'):.6f} | {mean(value, 'vertex_error_m'):.6f} | "
            f"{mean(value, 'pairwise_distance_error_m'):.6f} | "
            f"{mean(value, 'pairwise_vector_error_m'):.6f} | "
            f"{value['root_harm_over_5cm_rate']:.1%} |"
        )
    (output_dir / "dev_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def split_decision(candidate: dict[str, Any], brtc: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "all_five_error_metrics_not_worse": candidate_not_worse_all(candidate, brtc),
        "root_strictly_better": mean(candidate, "root_error_m") < mean(brtc, "root_error_m"),
        "harm_5cm_not_worse": candidate["root_harm_over_5cm_rate"]
        <= brtc["root_harm_over_5cm_rate"],
        "coverage_equal": candidate["coverage"] == brtc["coverage"],
        "camera_bit_exact": candidate["camera_max_abs_change"] == 0.0,
        "fallback_exact": candidate["fallback_max_abs_change"] == 0.0,
    }
    return {"checks": checks, "pass": all(checks.values())}


def run_confirm(args: argparse.Namespace) -> None:
    policy_path = args.policy.resolve()
    if not policy_path.is_file():
        raise FileNotFoundError(f"Frozen development policy required: {policy_path}")
    frozen = json.loads(policy_path.read_text(encoding="utf-8"))
    selected_weight = float(frozen["selected_prior_weight"])
    policy_sha = sha256(policy_path)
    policy_mtime = policy_path.stat().st_mtime_ns

    # Confirmation rows are intentionally opened only after the immutable
    # development policy has been loaded and fingerprinted above.
    offset1_rows = helpers.load_rows("confirm", args.confirm_report, args.max_cases)
    dance_rows = legacy.report_rows(("dance",))[: int(args.max_cases) or None]
    box_rows = legacy.report_rows(("box",))[: int(args.max_cases) or None]
    prepared_by_split = {
        "confirm_three_offset1": helpers.prepare_all(offset1_rows),
        "dance_posthoc_support": helpers.prepare_all(dance_rows),
        "box_posthoc_support": helpers.prepare_all(box_rows),
    }
    reports = {}
    cases_for_combined = {name: [] for name in ("b0", "brtc_lc_v1", "ray_layout_ls")}
    for split, prepared in prepared_by_split.items():
        summaries, cases = evaluate_summaries(prepared, selected_weight, full=True)
        reports[split] = {"methods": summaries, "cases": cases}
        if split in ("dance_posthoc_support", "box_posthoc_support"):
            for method, values in cases.items():
                cases_for_combined[method].extend(values)
    combined_summaries = {
        name: summarize(cases, name, full=True) for name, cases in cases_for_combined.items()
    }
    reports["dance_box_combined_posthoc_support"] = {
        "methods": combined_summaries,
        "cases": cases_for_combined,
    }

    decisions = {}
    for split in ("confirm_three_offset1", "dance_box_combined_posthoc_support"):
        methods = reports[split]["methods"]
        decisions[split] = split_decision(methods["ray_layout_ls"], methods["brtc_lc_v1"])
    final_go = bool(frozen["dev_go"] and all(value["pass"] for value in decisions.values()))
    decision = "GO_JOINT_RAY_LAYOUT_LS" if final_go else "NO_GO_JOINT_RAY_LAYOUT_LS"

    report = {
        "experiment": "v14_brtc_joint_ray_space_layout_least_squares",
        "phase": "frozen_confirmation",
        "protocol": {
            "policy_loaded_before_confirmation": True,
            "selected_prior_weight": selected_weight,
            "camera": "bit-exact frozen B0",
            "rejected_unmatched": "exact no-op",
            "inputs": "CPU geometry caches only",
            "gt": "evaluation only; no candidate action or gate",
            "dance_box_status": "post-hoc support because these caches were previously consumed",
        },
        "policy_source": str(policy_path),
        "policy_sha256_at_confirm": policy_sha,
        "policy_mtime_ns_before_confirm": policy_mtime,
        "development_go": bool(frozen["dev_go"]),
        "splits": reports,
        "confirmation_checks": decisions,
        "decision": decision,
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    confirm_path = output_dir / "confirm_report.json"
    confirm_path.write_text(
        json.dumps(jsonable(report), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Joint ray-space layout least squares: frozen confirmation",
        "",
        f"- Frozen prior weight: `{selected_weight:g}`.",
        f"- Policy SHA256: `{policy_sha}`.",
        f"- Decision: **{decision}**.",
        "- Camera remains bit-exact B0; rejected/unmatched remain exact no-ops.",
        "",
    ]
    for split in (
        "confirm_three_offset1",
        "dance_posthoc_support",
        "box_posthoc_support",
        "dance_box_combined_posthoc_support",
    ):
        lines.extend(markdown_table(split, reports[split]["methods"]))
    lines.extend(["## Frozen decision checks", ""])
    for split, value in decisions.items():
        lines.append(f"- `{split}` pass: `{value['pass']}`; checks: `{value['checks']}`.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The candidate is promoted only if development and both untouched offset1 / combined external checks pass every root, joint, vertex, pair-distance, pair-vector and safety condition. Any failure is an explicit NO-GO; confirmation results never retune the prior weight.",
        ]
    )
    markdown = "\n".join(lines) + "\n"
    (output_dir / "confirm_report.md").write_text(markdown, encoding="utf-8")
    print(markdown, flush=True)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if REPO_ROOT not in output.parents:
        raise ValueError("Output must remain under Movie3R on /data")
    if args.phase == "dev":
        run_dev(args)
    else:
        run_confirm(args)


if __name__ == "__main__":
    main()
