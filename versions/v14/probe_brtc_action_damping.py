#!/usr/bin/env python3
"""Select a fixed BRTC action damping on dev, then evaluate held-out support.

The selection rule sees only ``three offset0``.  It requires every primary
metric and harm rate to be no worse than undamped BRTC-LC v1, then chooses the
eligible scale with the lowest root error.  Confirmation and dance/box support
are evaluated only after that scale is fixed by this script's protocol.
"""

from __future__ import annotations

import argparse
import json
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

import versions.v14.probe_b0_two_view_person_triangulation as baseline


DEFAULT_OUTPUT = REPO_ROOT / "output/v14/fine_alignment_research/brtc_action_damping"
DEFAULT_SCALES = (0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 1.00, 1.10)
PRIMARY = (
    "root_error_m",
    "joint_error_m",
    "vertex_error_m",
    "pairwise_distance_error_m",
    "pairwise_vector_error_m",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scales", type=float, nargs="+", default=DEFAULT_SCALES)
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
    return value


def summary_metrics(summary: dict[str, Any]) -> dict[str, float]:
    corrected = summary["corrected"]
    return {
        key: float(corrected[key]["mean"])
        for key in PRIMARY
    } | {
        "coverage": float(summary["coverage"]),
        "root_harm_over_5cm_rate": float(summary["root_harm_over_5cm_rate"]),
        "root_improve_rate": float(summary["root_improve_rate"]),
        "camera_max_abs_change": float(summary["camera_candidate_max_abs_change"]),
    }


def evaluate(rows: list[dict[str, Any]], policy: dict[str, Any], scale: float) -> tuple[list, dict]:
    original = baseline.accepted_action

    def damped(evidence: dict[str, Any], locked_policy: dict[str, Any]) -> tuple[float, bool]:
        action, accepted = original(evidence, locked_policy)
        return float(scale) * action, accepted

    baseline.accepted_action = damped
    try:
        cases = [baseline.evaluate_case(row, policy) for row in rows]
    finally:
        baseline.accepted_action = original
    return cases, baseline.summarize(cases)


def markdown(report: dict[str, Any]) -> str:
    selected = report["selection"]["selected_scale"]
    lines = [
        "# BRTC-LC fixed action damping",
        "",
        f"Selected on three offset0: `action_scale={selected:.2f}`.",
        "",
        "| Split | Root | Joint | Vertex | Pair distance | Pair vector | Coverage | Harm >5cm |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, value in report["selected_evaluation"].items():
        metric = value["metrics"]
        lines.append(
            f"| {split} | {metric['root_error_m']:.4f} | {metric['joint_error_m']:.4f} | "
            f"{metric['vertex_error_m']:.4f} | {metric['pairwise_distance_error_m']:.4f} | "
            f"{metric['pairwise_vector_error_m']:.4f} | {metric['coverage']:.1%} | "
            f"{metric['root_harm_over_5cm_rate']:.1%} |"
        )
    lines.extend(
        [
            "",
            "The camera is unchanged. Candidate construction uses last-pre/first-post only.",
            "Raw Human3R, B0 and BRTC-LC are not conflated in this report.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if not str(args.output_dir.resolve()).startswith(str(REPO_ROOT.resolve())):
        raise ValueError("Output must stay in the repository under /data")
    frozen = json.loads(baseline.DEFAULT_POLICY.read_text(encoding="utf-8"))
    policy = frozen["policy"]

    original_torch_load = torch.load

    @lru_cache(maxsize=None)
    def cached_load(path: str) -> dict:
        return original_torch_load(path, map_location="cpu", weights_only=False)

    def load_adapter(path, *unused_args, **unused_kwargs):
        return cached_load(str(path))

    baseline.torch.load = load_adapter
    try:
        dev_rows = baseline.report_rows(("three",))
        dev_by_scale = {}
        dev_cases_by_scale = {}
        for scale in sorted(set(float(value) for value in args.scales)):
            cases, summary = evaluate(dev_rows, policy, scale)
            dev_cases_by_scale[scale] = cases
            dev_by_scale[scale] = summary_metrics(summary)

        if 1.0 not in dev_by_scale:
            raise ValueError("The scale grid must include the BRTC-LC v1 scale 1.0")
        reference = dev_by_scale[1.0]
        eligible = []
        for scale, metric in dev_by_scale.items():
            primary_safe = all(metric[key] <= reference[key] + 1e-12 for key in PRIMARY)
            harm_safe = metric["root_harm_over_5cm_rate"] <= reference[
                "root_harm_over_5cm_rate"
            ] + 1e-12
            coverage_safe = metric["coverage"] >= reference["coverage"] - 1e-12
            if primary_safe and harm_safe and coverage_safe:
                eligible.append(scale)
        selected_scale = min(eligible, key=lambda value: dev_by_scale[value]["root_error_m"])

        confirm_rows = json.loads(
            baseline.DEFAULT_K1_REPORT.read_text(encoding="utf-8")
        )["cases"]
        posthoc_rows = baseline.report_rows(("dance", "box"))
        confirm_cases, confirm_summary = evaluate(confirm_rows, policy, selected_scale)
        posthoc_cases, posthoc_summary = evaluate(posthoc_rows, policy, selected_scale)
        report = {
            "experiment": "v14_brtc_action_damping",
            "protocol": {
                "base": "frozen BRTC-LC v1",
                "new_parameter": "one fixed action scale applied before group median/layout consensus",
                "dev_split": "three offset0",
                "confirmation_split": "three offset1",
                "posthoc_support": "dance+box",
                "future_frames": 0,
                "extra_pretrained_models": [],
                "camera_update": "none",
            },
            "selection": {
                "grid": sorted(dev_by_scale),
                "rule": (
                    "all five primary means, harm, and coverage no worse than scale=1 on dev; "
                    "then minimum dev root"
                ),
                "eligible_scales": eligible,
                "selected_scale": selected_scale,
                "dev_by_scale": dev_by_scale,
            },
            "selected_evaluation": {
                "dev_three_offset0": {
                    "metrics": dev_by_scale[selected_scale],
                    "summary": baseline.summarize(dev_cases_by_scale[selected_scale]),
                },
                "confirm_three_offset1": {
                    "metrics": summary_metrics(confirm_summary),
                    "summary": confirm_summary,
                },
                "posthoc_dance_box": {
                    "metrics": summary_metrics(posthoc_summary),
                    "summary": posthoc_summary,
                },
            },
            "cases": {
                "confirm_three_offset1": confirm_cases,
                "posthoc_dance_box": posthoc_cases,
            },
        }
    finally:
        baseline.torch.load = original_torch_load

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(jsonable(report), indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    text = markdown(report)
    (args.output_dir / "report.md").write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
