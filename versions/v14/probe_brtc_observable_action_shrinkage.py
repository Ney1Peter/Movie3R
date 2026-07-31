#!/usr/bin/env python3
"""Freeze and validate a strictly-online observable BRTC action shrinker.

The frozen BRTC-LC proposal and gate are left untouched.  For an accepted
person, this probe only rescales the already observable ray-depth action::

    confidence = min(1, abs(raw_action) / full_trust_action_m)
    scaled_action = max_scale * confidence * raw_action

Small actions have a poor signal-to-noise ratio in the development cache, so
they are smoothly shrunk toward exact B0 instead of being treated as equally
reliable as a large action.  No image, GT, future frame, learned visual model,
or camera update is used by the candidate.  GT is loaded only by the existing
evaluator after the action has been produced.

Protocol:

* ``dev`` scans the two-parameter policy only on ``three offset0``;
* ``freeze`` writes the selected policy before any held-out evaluation;
* ``validate`` evaluates the frozen policy on ``three offset1``, ``dance`` and
  ``box`` without changing it.

This is an independent probe.  It does not edit the completeness-weighted
BRTC runtime or the frozen BRTC-LC v1 implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from functools import lru_cache
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


DEFAULT_OUTPUT = (
    REPO_ROOT / "output/v14/fine_alignment_research/brtc_observable_action_shrinkage"
)
DEFAULT_POLICY = DEFAULT_OUTPUT / "FROZEN_POLICY_BEFORE_HELDOUT.json"
MAX_SCALES = (0.75, 0.80, 0.85, 0.90, 0.95, 1.00)
FULL_TRUST_ACTIONS_M = (0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30)
PRIMARY = (
    "root_error_m",
    "joint_error_m",
    "vertex_error_m",
    "pairwise_distance_error_m",
    "pairwise_vector_error_m",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("dev", "freeze", "validate"), required=True)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--max_cases", type=int, default=0)
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
    payload = json.dumps(jsonable(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def compact(summary: dict[str, Any]) -> dict[str, float]:
    corrected = summary["corrected"]
    return {
        **{key: float(corrected[key]["mean"]) for key in PRIMARY},
        "coverage": float(summary["coverage"]),
        "root_improve_rate": float(summary["root_improve_rate"]),
        "root_harm_over_1cm_rate": float(summary["root_harm_over_1cm_rate"]),
        "root_harm_over_5cm_rate": float(summary["root_harm_over_5cm_rate"]),
        "fallback_max_abs_change": float(summary["fallback_max_abs_change"]),
        "camera_max_abs_change": float(summary["camera_candidate_max_abs_change"]),
    }


def observable_proposal(
    person: dict[str, Any],
    frozen_brtc_policy: dict[str, Any],
    candidate_policy: dict[str, float],
) -> tuple[np.ndarray, bool, dict[str, Any], float]:
    shift, accepted, evidence, action = harness.legacy_proposal(
        person, frozen_brtc_policy
    )
    evidence = dict(evidence)
    if accepted:
        threshold = float(candidate_policy["full_trust_action_m"])
        confidence = min(1.0, abs(float(action)) / threshold)
        action_scale = float(candidate_policy["max_scale"]) * confidence
    else:
        confidence = 0.0
        action_scale = 0.0
    evidence["observable_action_confidence"] = float(confidence)
    evidence["observable_action_scale"] = float(action_scale)
    return action_scale * shift, accepted, evidence, action_scale * float(action)


def evaluate(
    prepared: list[dict[str, Any]],
    frozen_brtc_policy: dict[str, Any],
    proposal: Callable[[dict[str, Any]], tuple[np.ndarray, bool, dict[str, Any], float]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cases = harness.evaluate_method(
        prepared,
        proposal,
        # The candidate retains the legacy evidence schema (``ray`` rather
        # than ``ray_world``), so use that serialization branch.
        "legacy_brtc_lc",
        full=True,
    )
    return cases, harness.summarize(cases, full=True)


def prepared_rows(rows: list[dict[str, Any]], max_cases: int) -> list[dict[str, Any]]:
    if max_cases:
        rows = rows[: int(max_cases)]
    return harness.prepare_all(rows)


def safe_vs(candidate: dict[str, float], reference: dict[str, float]) -> bool:
    return bool(
        all(candidate[key] <= reference[key] + 1e-12 for key in PRIMARY)
        and candidate["root_harm_over_1cm_rate"]
        <= reference["root_harm_over_1cm_rate"] + 1e-12
        and candidate["root_harm_over_5cm_rate"]
        <= reference["root_harm_over_5cm_rate"] + 1e-12
        and candidate["coverage"] >= reference["coverage"] - 1e-12
        and candidate["fallback_max_abs_change"] <= 1e-12
        and candidate["camera_max_abs_change"] <= 1e-12
    )


def policy_markdown(report: dict[str, Any]) -> str:
    selected = report["selection"]["selected_policy"]
    lines = [
        "# Observable action-magnitude shrinkage: development",
        "",
        "Only `three offset0` was used for selection.",
        "",
        "```text",
        "confidence = min(1, abs(raw_action) / full_trust_action_m)",
        "scaled_action = max_scale * confidence * raw_action",
        "```",
        "",
        f"Selected: `max_scale={selected['max_scale']}`, "
        f"`full_trust_action_m={selected['full_trust_action_m']}`.",
        "",
        "| Method | Root | Joint | Vertex | Pair distance | Pair vector | Coverage | Harm >1cm | Harm >5cm |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("brtc_v1", "fixed_0p8", "observable_shrinkage"):
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
            f"Strictly safe versus fixed 0.8 on all five errors, coverage and harm: "
            f"`{report['selection']['selected_safe_vs_fixed_0p8']}`.",
            "Held-out sets have not been read by this phase.",
        ]
    )
    return "\n".join(lines) + "\n"


def validation_markdown(report: dict[str, Any]) -> str:
    policy = report["policy"]
    lines = [
        "# Frozen observable BRTC action shrinkage: held-out validation",
        "",
        f"Frozen policy: `max_scale={policy['max_scale']}`, "
        f"`full_trust_action_m={policy['full_trust_action_m']}`.",
        "",
        "| Split | Method | Root | Joint | Vertex | Pair distance | Pair vector | Coverage | Harm >1cm | Harm >5cm |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, split_value in report["splits"].items():
        for name in ("brtc_v1", "fixed_0p8", "observable_shrinkage"):
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
            f"Candidate not worse than BRTC v1 on every held-out split/all five errors: "
            f"`{report['decision']['not_worse_than_v1_everywhere']}`.",
            f"Candidate beats fixed 0.8 on every held-out split/all five errors: "
            f"`{report['decision']['beats_fixed_0p8_everywhere']}`.",
            f"Decision: **{report['decision']['status']}**.",
            "Camera remains bit-exact B0; rejected people are exact no-ops.",
        ]
    )
    return "\n".join(lines) + "\n"


def install_cached_torch_load() -> tuple[Callable[..., Any], Callable[..., Any]]:
    original = torch.load

    @lru_cache(maxsize=None)
    def cached(path: str) -> dict[str, Any]:
        return original(path, map_location="cpu", weights_only=False)

    def adapter(path, *unused_args, **unused_kwargs):
        return cached(str(path))

    torch.load = adapter
    return original, cached


def run_dev(args: argparse.Namespace) -> None:
    frozen_brtc = harness.legacy_policy()
    rows = harness.load_rows("dev", harness.DEFAULT_CONFIRM_REPORT, args.max_cases)
    prepared = prepared_rows(rows, 0)

    v1_cases, v1_summary = evaluate(
        prepared,
        frozen_brtc,
        lambda person: harness.legacy_proposal(person, frozen_brtc),
    )
    fixed_cases, fixed_summary = evaluate(
        prepared,
        frozen_brtc,
        lambda person: harness.damped_legacy_proposal(person, frozen_brtc, 0.8),
    )
    fixed_metrics = compact(fixed_summary)

    scan = []
    selected_cases_by_key: dict[tuple[float, float], list[dict[str, Any]]] = {}
    selected_summary_by_key: dict[tuple[float, float], dict[str, Any]] = {}
    for max_scale in MAX_SCALES:
        for threshold in FULL_TRUST_ACTIONS_M:
            candidate_policy = {
                "max_scale": float(max_scale),
                "full_trust_action_m": float(threshold),
            }
            cases, summary = evaluate(
                prepared,
                frozen_brtc,
                lambda person, policy=candidate_policy: observable_proposal(
                    person, frozen_brtc, policy
                ),
            )
            metrics = compact(summary)
            key = (float(max_scale), float(threshold))
            selected_cases_by_key[key] = cases
            selected_summary_by_key[key] = summary
            scan.append(
                {
                    "policy": candidate_policy,
                    "metrics": metrics,
                    "safe_vs_fixed_0p8": safe_vs(metrics, fixed_metrics),
                }
            )
    eligible = [row for row in scan if row["safe_vs_fixed_0p8"]]
    if not eligible:
        raise RuntimeError("No observable action shrinker passes the development safety gate")
    selected = min(
        eligible,
        key=lambda row: (
            row["metrics"]["root_error_m"],
            row["metrics"]["pairwise_vector_error_m"],
            row["metrics"]["root_harm_over_5cm_rate"],
            row["policy"]["full_trust_action_m"],
            -row["policy"]["max_scale"],
        ),
    )
    selected_policy = selected["policy"]
    selected_key = (
        float(selected_policy["max_scale"]),
        float(selected_policy["full_trust_action_m"]),
    )
    report = {
        "experiment": "v14_brtc_observable_action_shrinkage",
        "phase": "development_before_freeze",
        "protocol": {
            "development": "three offset0 only",
            "selection_reference": "fixed 0.8 damped BRTC-LC",
            "candidate_inputs": "accepted raw action magnitude only",
            "future_frames": 0,
            "extra_pretrained_models": [],
            "camera_update": "none",
            "gt_candidate_use": "none",
        },
        "grid": {
            "max_scale": MAX_SCALES,
            "full_trust_action_m": FULL_TRUST_ACTIONS_M,
        },
        "selection": {
            "rule": (
                "all five full-geometry means, coverage, >1cm harm and >5cm harm "
                "must be no worse than fixed 0.8; then minimize root, pair vector, harm"
            ),
            "eligible_count": len(eligible),
            "selected_policy": selected_policy,
            "selected_policy_sha256": canonical_sha256(selected_policy),
            "selected_safe_vs_fixed_0p8": bool(selected["safe_vs_fixed_0p8"]),
        },
        "scan": scan,
        "selected_dev": {
            "brtc_v1": {"metrics": compact(v1_summary), "summary": v1_summary},
            "fixed_0p8": {"metrics": fixed_metrics, "summary": fixed_summary},
            "observable_shrinkage": {
                "metrics": selected["metrics"],
                "summary": selected_summary_by_key[selected_key],
            },
        },
        "selected_cases": {
            "brtc_v1": v1_cases,
            "fixed_0p8": fixed_cases,
            "observable_shrinkage": selected_cases_by_key[selected_key],
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "DEV_SCAN.json").write_text(
        json.dumps(jsonable(report), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    text = policy_markdown(report)
    (args.output_dir / "DEV_RESULTS.md").write_text(text, encoding="utf-8")
    print(text, end="")


def run_freeze(args: argparse.Namespace) -> None:
    source = args.output_dir / "DEV_SCAN.json"
    if not source.is_file():
        raise FileNotFoundError(source)
    dev = json.loads(source.read_text(encoding="utf-8"))
    if dev.get("phase") != "development_before_freeze":
        raise ValueError("Development report is not a pre-freeze artifact")
    if not dev["selection"]["selected_safe_vs_fixed_0p8"]:
        raise RuntimeError("Development candidate did not pass its safety gate")
    policy = dev["selection"]["selected_policy"]
    frozen = {
        "experiment": dev["experiment"],
        "status": "frozen_before_offset1_dance_box_egohumans",
        "policy": policy,
        "policy_sha256": canonical_sha256(policy),
        "source_dev_report": str(source),
        "source_dev_report_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "formula": (
            "confidence=min(1,abs(raw_action)/full_trust_action_m); "
            "scaled_action=max_scale*confidence*raw_action"
        ),
        "constraints": {
            "future_frames": 0,
            "extra_pretrained_models": [],
            "camera_update": "none",
            "rejected_fallback": "exact B0",
        },
    }
    args.policy.parent.mkdir(parents=True, exist_ok=True)
    args.policy.write_text(
        json.dumps(jsonable(frozen), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(frozen, indent=2, ensure_ascii=False))


def run_validate(args: argparse.Namespace) -> None:
    if not args.policy.is_file():
        raise FileNotFoundError(args.policy)
    frozen = json.loads(args.policy.read_text(encoding="utf-8"))
    policy = frozen["policy"]
    if canonical_sha256(policy) != frozen["policy_sha256"]:
        raise ValueError("Frozen policy checksum mismatch")
    frozen_brtc = harness.legacy_policy()
    split_rows = {
        "confirm_three_offset1": harness.load_rows(
            "confirm", harness.DEFAULT_CONFIRM_REPORT, args.max_cases
        ),
        "dance": legacy.report_rows(("dance",)),
        "box": legacy.report_rows(("box",)),
    }
    if args.max_cases:
        split_rows = {
            key: value[: int(args.max_cases)] for key, value in split_rows.items()
        }
    report_splits = {}
    for split, rows in split_rows.items():
        prepared = prepared_rows(rows, 0)
        methods = {}
        for name, proposal in (
            ("brtc_v1", lambda person: harness.legacy_proposal(person, frozen_brtc)),
            (
                "fixed_0p8",
                lambda person: harness.damped_legacy_proposal(
                    person, frozen_brtc, 0.8
                ),
            ),
            (
                "observable_shrinkage",
                lambda person: observable_proposal(person, frozen_brtc, policy),
            ),
        ):
            cases, summary = evaluate(prepared, frozen_brtc, proposal)
            methods[name] = {
                "metrics": compact(summary),
                "summary": summary,
                "cases": cases,
            }
        report_splits[split] = {"methods": methods}

    def all_primary_leq(first: str, second: str) -> bool:
        return all(
            all(
                split["methods"][first]["metrics"][key]
                <= split["methods"][second]["metrics"][key] + 1e-12
                for key in PRIMARY
            )
            for split in report_splits.values()
        )

    not_worse_v1 = all_primary_leq("observable_shrinkage", "brtc_v1")
    beats_fixed = all_primary_leq("observable_shrinkage", "fixed_0p8")
    harm_safe = all(
        split["methods"]["observable_shrinkage"]["metrics"][
            "root_harm_over_5cm_rate"
        ]
        <= split["methods"]["brtc_v1"]["metrics"]["root_harm_over_5cm_rate"]
        + 1e-12
        for split in report_splits.values()
    )
    status = (
        "GO_OBSERVABLE_ACTION_SHRINKAGE"
        if not_worse_v1 and harm_safe
        else "NO_GO_OBSERVABLE_ACTION_SHRINKAGE"
    )
    report = {
        "experiment": "v14_brtc_observable_action_shrinkage",
        "phase": "heldout_after_policy_freeze",
        "policy_source": str(args.policy),
        "policy": policy,
        "policy_sha256": frozen["policy_sha256"],
        "splits": report_splits,
        "decision": {
            "not_worse_than_v1_everywhere": not_worse_v1,
            "beats_fixed_0p8_everywhere": beats_fixed,
            "harm_not_worse_than_v1_everywhere": harm_safe,
            "status": status,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "HELDOUT_RESULTS.json").write_text(
        json.dumps(jsonable(report), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    text = validation_markdown(report)
    (args.output_dir / "HELDOUT_RESULTS.md").write_text(text, encoding="utf-8")
    print(text, end="")


def main() -> None:
    args = parse_args()
    if not str(args.output_dir.resolve()).startswith(str(REPO_ROOT.resolve())):
        raise ValueError("Output must stay inside Movie3R under /data")
    original_load, _ = install_cached_torch_load()
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
