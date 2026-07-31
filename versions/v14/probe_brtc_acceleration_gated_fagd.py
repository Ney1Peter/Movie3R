#!/usr/bin/env python3
"""Probe a causal acceleration-consistency gate for FAGD-0.9.

FAGD improves spatial errors but can worsen trajectory acceleration.  At a
boundary the online system already owns at least two pre-shot person roots.
This probe compares frozen BRTC and FAGD using only the predicted boundary
second difference::

    score = mean_i ||root_post_i - 2 root_pre_i + root_preprev_i||

FAGD is applied only when its observable score is no larger than frozen BRTC's
score.  The camera, evidence gate, residual lambda and individual residuals
remain frozen.  No GT, future post frame, or additional model is used to make
the decision.  GT is read only by the existing evaluator after prediction.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from functools import lru_cache
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
from versions.v14.probe_b0_anchor_conflict import SEQUENCE_INPUTS  # noqa: E402


DEFAULT_OUTPUT = REPO_ROOT / "output/v14/fine_alignment_research/brtc_acceleration_gated_fagd"
ALPHA = 0.9
ALPHA_GRID = (0.8, 0.9, 1.0, 1.1, 1.2)
PRIMARY = (
    "root_error_m",
    "joint_error_m",
    "vertex_error_m",
    "pairwise_distance_error_m",
    "pairwise_vector_error_m",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("dev", "validate"))
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def compact(summary: dict[str, Any]) -> dict[str, float]:
    corrected = summary["corrected"]
    return {
        **{key: float(corrected[key]["mean"]) for key in PRIMARY},
        "coverage": float(summary["coverage"]),
        "root_harm_over_5cm_rate": float(summary["root_harm_over_5cm_rate"]),
        "camera_max_abs_change": float(summary["camera_candidate_max_abs_change"]),
        "fallback_max_abs_change": float(summary["fallback_max_abs_change"]),
    }


def add_pre_history(prepared: list[dict[str, Any]]) -> None:
    for case in prepared:
        sequence = str(case["sequence"])
        key = str(case["case"]["key"])
        cache = torch.load(
            SEQUENCE_INPUTS[sequence]["cache"] / f"{key}.pt",
            map_location="cpu",
            weights_only=False,
        )
        history_frame = cache["humans"][-3] if len(cache["humans"]) >= 3 else {}
        for person in case["people"]:
            identity = str(person["identity"])
            person["pre_previous_root_observable"] = (
                np.asarray(history_frame[identity]["root"], dtype=np.float64)
                if identity in history_frame
                else None
            )


def evaluate(
    prepared: list[dict[str, Any]],
    frozen_policy: dict[str, Any],
    gated: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    original = harness.observable_layout_consensus
    decisions: list[dict[str, Any]] = []

    def consensus(case, proposals):
        base_shifts, debug = original(case, proposals)
        full_accept = bool(proposals) and all(row["accepted"] for row in proposals)
        group = np.asarray(debug["group_shift_world"], dtype=np.float64)
        residual_lambda = float(debug["selected_residual_lambda"])
        fagd_shifts = [
            ALPHA * group
            + residual_lambda
            * (np.asarray(row["individual_shift"], dtype=np.float64) - group)
            for row in proposals
        ]
        history_ok = full_accept and all(
            person.get("pre_previous_root_observable") is not None
            for person in case["people"]
        )
        if history_ok:
            base_score = float(np.mean([
                np.linalg.norm(
                    person["post"]["root"] + shift
                    - 2.0 * person["pre"]["root"]
                    + person["pre_previous_root_observable"]
                )
                for person, shift in zip(case["people"], base_shifts)
            ]))
            fagd_score = float(np.mean([
                np.linalg.norm(
                    person["post"]["root"] + shift
                    - 2.0 * person["pre"]["root"]
                    + person["pre_previous_root_observable"]
                )
                for person, shift in zip(case["people"], fagd_shifts)
            ]))
        else:
            base_score = fagd_score = float("inf")
        apply = bool(full_accept and (not gated or (history_ok and fagd_score <= base_score)))
        decisions.append({
            "case": case["case"],
            "full_accept": full_accept,
            "history_ok": history_ok,
            "brtc_predicted_acceleration_score_m": base_score,
            "fagd_predicted_acceleration_score_m": fagd_score,
            "fagd_applied": apply,
        })
        output_debug = dict(debug)
        output_debug.update({
            "acceleration_gate_enabled": bool(gated),
            "history_ok": history_ok,
            "brtc_predicted_acceleration_score_m": base_score,
            "fagd_predicted_acceleration_score_m": fagd_score,
            "group_damping_applied": apply,
            "group_only_alpha": ALPHA if apply else 1.0,
        })
        return (fagd_shifts if apply else base_shifts), output_debug

    harness.observable_layout_consensus = consensus
    try:
        cases = harness.evaluate_method(
            prepared,
            lambda person: harness.legacy_proposal(person, frozen_policy),
            "legacy_brtc_lc",
            full=True,
        )
    finally:
        harness.observable_layout_consensus = original
    summary = harness.summarize(cases, full=True)
    audit = {
        "case_count": len(decisions),
        "full_accept_count": int(sum(row["full_accept"] for row in decisions)),
        "history_ok_count": int(sum(row["history_ok"] for row in decisions)),
        "fagd_applied_count": int(sum(row["fagd_applied"] for row in decisions)),
        "decisions": decisions,
    }
    return cases, summary, audit


def evaluate_alpha_selection(
    prepared: list[dict[str, Any]],
    frozen_policy: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Choose a shared group alpha from a fixed symmetric observable grid."""

    original = harness.observable_layout_consensus
    decisions: list[dict[str, Any]] = []

    def consensus(case, proposals):
        base_shifts, debug = original(case, proposals)
        full_accept = bool(proposals) and all(row["accepted"] for row in proposals)
        history_ok = full_accept and all(
            person.get("pre_previous_root_observable") is not None
            for person in case["people"]
        )
        group = np.asarray(debug["group_shift_world"], dtype=np.float64)
        residual_lambda = float(debug["selected_residual_lambda"])
        score_by_alpha = {}
        shifts_by_alpha = {}
        for alpha in ALPHA_GRID:
            shifts = [
                float(alpha) * group
                + residual_lambda
                * (np.asarray(row["individual_shift"], dtype=np.float64) - group)
                for row in proposals
            ]
            shifts_by_alpha[float(alpha)] = shifts
            score_by_alpha[float(alpha)] = (
                float(np.mean([
                    np.linalg.norm(
                        person["post"]["root"] + shift
                        - 2.0 * person["pre"]["root"]
                        + person["pre_previous_root_observable"]
                    )
                    for person, shift in zip(case["people"], shifts)
                ]))
                if history_ok else float("inf")
            )
        selected = (
            min(ALPHA_GRID, key=lambda value: (score_by_alpha[float(value)], abs(value - 1.0)))
            if history_ok else 1.0
        )
        decisions.append({
            "case": case["case"],
            "full_accept": full_accept,
            "history_ok": history_ok,
            "score_by_alpha": score_by_alpha,
            "selected_alpha": float(selected),
        })
        output_debug = dict(debug)
        output_debug.update({
            "acceleration_alpha_selection": True,
            "history_ok": history_ok,
            "score_by_alpha": score_by_alpha,
            "selected_group_alpha": float(selected),
        })
        return shifts_by_alpha[float(selected)] if history_ok else base_shifts, output_debug

    harness.observable_layout_consensus = consensus
    try:
        cases = harness.evaluate_method(
            prepared,
            lambda person: harness.legacy_proposal(person, frozen_policy),
            "legacy_brtc_lc",
            full=True,
        )
    finally:
        harness.observable_layout_consensus = original
    summary = harness.summarize(cases, full=True)
    audit = {
        "case_count": len(decisions),
        "history_ok_count": int(sum(row["history_ok"] for row in decisions)),
        "alpha_counts": {
            str(alpha): int(sum(row["selected_alpha"] == alpha for row in decisions))
            for alpha in ALPHA_GRID
        },
        "fagd_applied_count": int(sum(row["selected_alpha"] != 1.0 for row in decisions)),
        "decisions": decisions,
    }
    return cases, summary, audit


def load_prepared(phase: str) -> dict[str, list[dict[str, Any]]]:
    if phase == "dev":
        split_rows = {"three_offset0": harness.load_rows("dev", harness.DEFAULT_CONFIRM_REPORT, 0)}
    else:
        split_rows = {
            "three_offset1": json.loads(
                harness.DEFAULT_CONFIRM_REPORT.read_text(encoding="utf-8")
            )["cases"],
            "dance": legacy.report_rows(("dance",)),
            "box": legacy.report_rows(("box",)),
        }
    output = {}
    for name, rows in split_rows.items():
        prepared = harness.prepare_all(rows)
        add_pre_history(prepared)
        output[name] = prepared
    return output


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Acceleration-gated FAGD-0.9",
        "",
        f"Phase: `{report['phase']}`.",
        "",
        "| Split | Method | Root | Joint | Vertex | Pair distance | Pair vector | Harm >5cm | FAGD applied |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, value in report["splits"].items():
        for method in (
            "v1",
            "ungated_fagd",
            "acceleration_gated_fagd",
            "acceleration_selected_alpha",
        ):
            metric = value[method]["metrics"]
            applied = value[method]["audit"]["fagd_applied_count"]
            lines.append(
                f"| {split} | {method} | {metric['root_error_m']:.6f} | "
                f"{metric['joint_error_m']:.6f} | {metric['vertex_error_m']:.6f} | "
                f"{metric['pairwise_distance_error_m']:.6f} | "
                f"{metric['pairwise_vector_error_m']:.6f} | "
                f"{metric['root_harm_over_5cm_rate']:.1%} | {applied} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if not str(args.output_dir.resolve()).startswith(str(REPO_ROOT.resolve())):
        raise ValueError("Output must stay in Movie3R under /data")
    original_load = torch.load

    @lru_cache(maxsize=None)
    def cached_load(path: str):
        return original_load(path, map_location="cpu", weights_only=False)

    def adapter(path, *unused_args, **unused_kwargs):
        return cached_load(str(path))

    torch.load = adapter
    try:
        prepared_by_split = load_prepared(args.phase)
        policy = harness.legacy_policy()
        splits = {}
        for split, prepared in prepared_by_split.items():
            methods = {}
            for name, gated in (
                ("v1", None),
                ("ungated_fagd", False),
                ("acceleration_gated_fagd", True),
                ("acceleration_selected_alpha", "grid"),
            ):
                if gated is None:
                    cases = harness.evaluate_method(
                        prepared,
                        lambda person: harness.legacy_proposal(person, policy),
                        "legacy_brtc_lc",
                        full=True,
                    )
                    summary = harness.summarize(cases, full=True)
                    audit = {"fagd_applied_count": 0, "decisions": []}
                elif gated == "grid":
                    cases, summary, audit = evaluate_alpha_selection(prepared, policy)
                else:
                    cases, summary, audit = evaluate(prepared, policy, gated)
                methods[name] = {"metrics": compact(summary), "audit": audit}
            splits[split] = methods
    finally:
        torch.load = original_load

    report = {
        "experiment": "v14_brtc_acceleration_gated_fagd",
        "phase": args.phase,
        "protocol": {
            "alpha": ALPHA,
            "decision": "FAGD score <= BRTC score using previous-two/current predicted roots",
            "future_frames": 0,
            "extra_pretrained_models": [],
            "camera_update": "none",
            "candidate_gt_use": "none",
        },
        "splits": splits,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = "DEV_RESULTS" if args.phase == "dev" else "VALIDATION_RESULTS"
    (args.output_dir / f"{stem}.json").write_text(
        json.dumps(common.jsonable(report), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    text = markdown(report)
    (args.output_dir / f"{stem}.md").write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
