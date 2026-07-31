#!/usr/bin/env python3
"""Freeze and validate full-accept group-only damping for BRTC-LC.

Frozen BRTC decomposes every accepted final translation as::

    final_i = group_median + lambda * (individual_i - group_median)

When every matched person passes the frozen ray gate, ``group_median`` is a
common translation.  Scaling only that common component cannot change any
pairwise root distance or vector.  This probe therefore tests the conservative
strictly-online residual policy::

    if accepted_count == matched_count > 0:
        final_i = alpha * group_median + lambda * individual_residual_i
    else:
        final_i = frozen_BRTC_final_i

The all-accepted predicate, group median, and individual residual are already
observable at the current last-pre/first-post boundary.  No GT, image model,
future frame, persistent state, or camera update is used by the candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts", REPO_ROOT / "versions/v14"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from versions.v14 import probe_b0_brtc_huber_irls as harness  # noqa: E402
from versions.v14 import probe_b0_two_view_person_triangulation as legacy  # noqa: E402
from versions.v14 import probe_brtc_observable_action_shrinkage as common  # noqa: E402


DEFAULT_OUTPUT = (
    REPO_ROOT / "output/v14/fine_alignment_research/brtc_full_accept_group_damping"
)
DEFAULT_POLICY = DEFAULT_OUTPUT / "FROZEN_POLICY_BEFORE_HELDOUT.json"
ALPHAS = (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("dev", "freeze", "validate"), required=True)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--max_cases", type=int, default=0)
    return parser.parse_args()


def evaluate_group_only(
    prepared: list[dict[str, Any]],
    frozen_brtc_policy: dict[str, Any],
    alpha: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    original = harness.observable_layout_consensus

    def group_damped(case, proposals):
        base_shifts, debug = original(case, proposals)
        full_accept = bool(proposals) and all(row["accepted"] for row in proposals)
        if full_accept:
            group = np.asarray(debug["group_shift_world"], dtype=np.float64)
            residual_lambda = float(debug["selected_residual_lambda"])
            shifts = [
                float(alpha) * group
                + residual_lambda
                * (np.asarray(row["individual_shift"], dtype=np.float64) - group)
                for row in proposals
            ]
        else:
            shifts = base_shifts
        output_debug = dict(debug)
        output_debug.update(
            {
                "full_accept_observable": full_accept,
                "group_only_alpha": float(alpha) if full_accept else 1.0,
                "group_damping_applied": full_accept and float(alpha) != 1.0,
            }
        )
        return shifts, output_debug

    harness.observable_layout_consensus = group_damped
    try:
        cases = harness.evaluate_method(
            prepared,
            lambda person: harness.legacy_proposal(person, frozen_brtc_policy),
            "legacy_brtc_lc",
            full=True,
        )
    finally:
        harness.observable_layout_consensus = original
    return cases, harness.summarize(cases, full=True)


def invariant_layout(first: dict[str, float], second: dict[str, float]) -> bool:
    return bool(
        abs(first["pairwise_distance_error_m"] - second["pairwise_distance_error_m"])
        <= 1e-12
        and abs(first["pairwise_vector_error_m"] - second["pairwise_vector_error_m"])
        <= 1e-12
    )


def dev_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Full-accept group-only BRTC damping: development",
        "",
        "Selection used only `three offset0`.",
        "",
        f"Selected `alpha={report['selection']['selected_alpha']}`.",
        "",
        "| Method | Root | Joint | Vertex | Pair distance | Pair vector | Coverage | Harm >1cm | Harm >5cm |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("brtc_v1", "fixed_0p8", "group_only"):
        value = report["selected_dev"][name]["metrics"]
        lines.append(
            f"| {name} | {value['root_error_m']:.6f} | "
            f"{value['joint_error_m']:.6f} | {value['vertex_error_m']:.6f} | "
            f"{value['pairwise_distance_error_m']:.6f} | "
            f"{value['pairwise_vector_error_m']:.6f} | {value['coverage']:.1%} | "
            f"{value['root_harm_over_1cm_rate']:.1%} | "
            f"{value['root_harm_over_5cm_rate']:.1%} |"
        )
    lines.extend(
        [
            "",
            f"Layout bit-exact to BRTC v1: `{report['selection']['layout_invariant']}`.",
            f"Development safety passed: `{report['selection']['passed']}`.",
            "Held-out sets were not evaluated by this phase.",
        ]
    )
    return "\n".join(lines) + "\n"


def validation_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Frozen full-accept group-only BRTC damping: held-out validation",
        "",
        f"Frozen `alpha={report['policy']['alpha']}`.",
        "",
        "| Split | Method | Root | Joint | Vertex | Pair distance | Pair vector | Coverage | Harm >1cm | Harm >5cm |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, split_value in report["splits"].items():
        for name in ("brtc_v1", "fixed_0p8", "group_only"):
            value = split_value["methods"][name]["metrics"]
            lines.append(
                f"| {split} | {name} | {value['root_error_m']:.6f} | "
                f"{value['joint_error_m']:.6f} | {value['vertex_error_m']:.6f} | "
                f"{value['pairwise_distance_error_m']:.6f} | "
                f"{value['pairwise_vector_error_m']:.6f} | {value['coverage']:.1%} | "
                f"{value['root_harm_over_1cm_rate']:.1%} | "
                f"{value['root_harm_over_5cm_rate']:.1%} |"
            )
    lines.extend(
        [
            "",
            f"Root/joint/vertex improve over v1 on every held-out split: "
            f"`{report['decision']['spatial_improves_everywhere']}`.",
            f"Pair metrics bit-exact v1 everywhere: "
            f"`{report['decision']['layout_invariant_everywhere']}`.",
            f"Harm >5cm no worse everywhere: "
            f"`{report['decision']['harm_safe_everywhere']}`.",
            f"Decision: **{report['decision']['status']}**.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_dev(args: argparse.Namespace) -> None:
    frozen_brtc = harness.legacy_policy()
    rows = harness.load_rows("dev", harness.DEFAULT_CONFIRM_REPORT, args.max_cases)
    prepared = common.prepared_rows(rows, 0)
    scan = []
    cases_by_alpha, summaries_by_alpha = {}, {}
    for alpha in ALPHAS:
        cases, summary = evaluate_group_only(prepared, frozen_brtc, alpha)
        metrics = common.compact(summary)
        cases_by_alpha[float(alpha)] = cases
        summaries_by_alpha[float(alpha)] = summary
        scan.append({"alpha": float(alpha), "metrics": metrics})
    v1 = next(row for row in scan if row["alpha"] == 1.0)
    eligible = []
    for row in scan:
        metric, reference = row["metrics"], v1["metrics"]
        safe = bool(
            all(metric[key] <= reference[key] + 1e-12 for key in common.PRIMARY)
            and metric["root_harm_over_1cm_rate"]
            <= reference["root_harm_over_1cm_rate"] + 1e-12
            and metric["root_harm_over_5cm_rate"]
            <= reference["root_harm_over_5cm_rate"] + 1e-12
            and metric["coverage"] >= reference["coverage"] - 1e-12
        )
        row["safe_vs_v1"] = safe
        if safe:
            eligible.append(row)
    selected = min(
        eligible,
        key=lambda row: (
            row["metrics"]["root_error_m"],
            row["metrics"]["joint_error_m"],
            row["metrics"]["vertex_error_m"],
            abs(1.0 - row["alpha"]),
        ),
    )
    alpha = float(selected["alpha"])
    fixed_cases, fixed_summary = common.evaluate(
        prepared,
        frozen_brtc,
        lambda person: harness.damped_legacy_proposal(person, frozen_brtc, 0.8),
    )
    layout_exact = invariant_layout(selected["metrics"], v1["metrics"])
    report = {
        "experiment": "v14_brtc_full_accept_group_only_damping",
        "phase": "development_before_freeze",
        "protocol": {
            "development": "three offset0 only",
            "observable_gate": "accepted_count == matched_count > 0",
            "damped_component": "group median only",
            "individual_residual": "exact frozen BRTC residual",
            "future_frames": 0,
            "extra_pretrained_models": [],
            "camera_update": "none",
        },
        "grid": ALPHAS,
        "scan": scan,
        "selection": {
            "rule": (
                "all five errors, coverage and harm no worse than v1; then minimize "
                "root, joint, vertex"
            ),
            "selected_alpha": alpha,
            "eligible_alphas": [row["alpha"] for row in eligible],
            "layout_invariant": layout_exact,
            "passed": bool(selected["safe_vs_v1"] and layout_exact),
        },
        "selected_dev": {
            "brtc_v1": {
                "metrics": v1["metrics"],
                "summary": summaries_by_alpha[1.0],
                "cases": cases_by_alpha[1.0],
            },
            "fixed_0p8": {
                "metrics": common.compact(fixed_summary),
                "summary": fixed_summary,
                "cases": fixed_cases,
            },
            "group_only": {
                "metrics": selected["metrics"],
                "summary": summaries_by_alpha[alpha],
                "cases": cases_by_alpha[alpha],
            },
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "DEV_SCAN.json").write_text(
        json.dumps(common.jsonable(report), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    text = dev_markdown(report)
    (args.output_dir / "DEV_RESULTS.md").write_text(text, encoding="utf-8")
    print(text, end="")


def run_freeze(args: argparse.Namespace) -> None:
    source = args.output_dir / "DEV_SCAN.json"
    dev = json.loads(source.read_text(encoding="utf-8"))
    if not dev["selection"]["passed"]:
        raise RuntimeError("Development policy did not pass")
    policy = {
        "alpha": float(dev["selection"]["selected_alpha"]),
        "gate": "accepted_count == matched_count > 0",
        "application": "scale group median only; keep individual residual exact",
    }
    frozen = {
        "experiment": dev["experiment"],
        "status": "frozen_before_offset1_dance_box_egohumans",
        "policy": policy,
        "policy_sha256": common.canonical_sha256(policy),
        "source_dev_report": str(source),
        "source_dev_report_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "constraints": {
            "future_frames": 0,
            "extra_pretrained_models": [],
            "camera_update": "none",
            "rejected_fallback": "exact frozen BRTC / exact B0 where BRTC rejects",
        },
    }
    args.policy.parent.mkdir(parents=True, exist_ok=True)
    args.policy.write_text(
        json.dumps(common.jsonable(frozen), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(frozen, indent=2, ensure_ascii=False))


def run_validate(args: argparse.Namespace) -> None:
    frozen = json.loads(args.policy.read_text(encoding="utf-8"))
    policy = frozen["policy"]
    if common.canonical_sha256(policy) != frozen["policy_sha256"]:
        raise ValueError("Frozen policy checksum mismatch")
    alpha = float(policy["alpha"])
    frozen_brtc = harness.legacy_policy()
    splits = {
        "confirm_three_offset1": harness.load_rows(
            "confirm", harness.DEFAULT_CONFIRM_REPORT, args.max_cases
        ),
        "dance": legacy.report_rows(("dance",)),
        "box": legacy.report_rows(("box",)),
    }
    if args.max_cases:
        splits = {key: value[: args.max_cases] for key, value in splits.items()}
    results = {}
    for split, rows in splits.items():
        prepared = common.prepared_rows(rows, 0)
        v1_cases, v1_summary = evaluate_group_only(prepared, frozen_brtc, 1.0)
        group_cases, group_summary = evaluate_group_only(prepared, frozen_brtc, alpha)
        fixed_cases, fixed_summary = common.evaluate(
            prepared,
            frozen_brtc,
            lambda person: harness.damped_legacy_proposal(person, frozen_brtc, 0.8),
        )
        results[split] = {
            "methods": {
                "brtc_v1": {
                    "metrics": common.compact(v1_summary),
                    "summary": v1_summary,
                    "cases": v1_cases,
                },
                "fixed_0p8": {
                    "metrics": common.compact(fixed_summary),
                    "summary": fixed_summary,
                    "cases": fixed_cases,
                },
                "group_only": {
                    "metrics": common.compact(group_summary),
                    "summary": group_summary,
                    "cases": group_cases,
                },
            }
        }
    spatial = all(
        all(
            split["methods"]["group_only"]["metrics"][key]
            < split["methods"]["brtc_v1"]["metrics"][key] - 1e-12
            for key in ("root_error_m", "joint_error_m", "vertex_error_m")
        )
        for split in results.values()
    )
    layout = all(
        invariant_layout(
            split["methods"]["group_only"]["metrics"],
            split["methods"]["brtc_v1"]["metrics"],
        )
        for split in results.values()
    )
    harm = all(
        split["methods"]["group_only"]["metrics"]["root_harm_over_5cm_rate"]
        <= split["methods"]["brtc_v1"]["metrics"]["root_harm_over_5cm_rate"]
        + 1e-12
        for split in results.values()
    )
    status = (
        "GO_FULL_ACCEPT_GROUP_ONLY_DAMPING"
        if spatial and layout and harm
        else "NO_GO_FULL_ACCEPT_GROUP_ONLY_DAMPING"
    )
    report = {
        "experiment": "v14_brtc_full_accept_group_only_damping",
        "phase": "heldout_after_policy_freeze",
        "policy_source": str(args.policy),
        "policy": policy,
        "policy_sha256": frozen["policy_sha256"],
        "splits": results,
        "decision": {
            "spatial_improves_everywhere": spatial,
            "layout_invariant_everywhere": layout,
            "harm_safe_everywhere": harm,
            "status": status,
        },
    }
    (args.output_dir / "HELDOUT_RESULTS.json").write_text(
        json.dumps(common.jsonable(report), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    text = validation_markdown(report)
    (args.output_dir / "HELDOUT_RESULTS.md").write_text(text, encoding="utf-8")
    print(text, end="")


def main() -> None:
    args = parse_args()
    if not str(args.output_dir.resolve()).startswith(str(REPO_ROOT.resolve())):
        raise ValueError("Output must stay in Movie3R under /data")
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
