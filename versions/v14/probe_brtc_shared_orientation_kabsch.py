#!/usr/bin/env python3
"""Mechanism and blind held-out tests for shared accepted-set SO(3) Kabsch.

The policy is inherited without tuning from the already-frozen individual
orientation Kabsch policy.  ``three offset0`` is only a mechanism check;
``three offset1``/``dance``/``box`` are evaluated with the inherited policy.
"""

from __future__ import annotations

import argparse
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
from versions.v14 import probe_brtc_strict_deployable_fagd as strict  # noqa: E402
from versions.v14.b0_person_triangulation_shared_orientation_kabsch import (  # noqa: E402
    SharedOrientationKabschConfig,
    refine_matched_people_shared_orientation_kabsch,
)


DEFAULT_SOURCE_POLICY = (
    REPO_ROOT
    / "output/v14/fine_alignment_research/brtc_global_orientation_kabsch/"
    "FROZEN_POLICY_BEFORE_VALIDATION.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "output/v14/fine_alignment_research/brtc_shared_orientation_kabsch"
EXPECTED_POLICY = {
    "max_angle_deg": 25.0,
    "rotation_fraction": 0.5,
    "min_observable_relative_improvement": 0.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("dev", "validate"))
    parser.add_argument("--source_policy", type=Path, default=DEFAULT_SOURCE_POLICY)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max_cases", type=int, default=0)
    return parser.parse_args()


def load_config(path: Path) -> tuple[dict[str, Any], SharedOrientationKabschConfig]:
    frozen = json.loads(path.read_text(encoding="utf-8"))
    policy = frozen["policy"]
    if common.canonical_sha256(policy) != frozen["policy_sha256"]:
        raise ValueError("Inherited orientation policy checksum mismatch")
    normalized = {key: float(value) for key, value in policy.items()}
    if normalized != EXPECTED_POLICY:
        raise ValueError(f"Unexpected inherited policy: {normalized}")
    return frozen, SharedOrientationKabschConfig(**normalized)


def callback(config: SharedOrientationKabschConfig, rows: list[dict[str, Any]]):
    def apply(pre_camera, post_camera, pre_people, post_people, matches):
        corrected, debug = refine_matched_people_shared_orientation_kabsch(
            pre_camera,
            post_camera,
            pre_people,
            post_people,
            matches,
            orientation_config=config,
        )
        rows.append(
            {
                "matched_count": int(debug["matched_count"]),
                "accepted_count": int(debug["accepted_count"]),
                "accepted_orientation_person_count": int(
                    debug["accepted_orientation_person_count"]
                ),
                "shared_orientation_applied": bool(
                    debug["shared_orientation_applied"]
                ),
                "shared_orientation_reason": debug["shared_orientation_reason"],
                "raw_torso_residual_m": float(
                    debug["shared_raw_torso_residual_m"]
                ),
                "candidate_torso_residual_m": float(
                    debug["shared_candidate_torso_residual_m"]
                ),
                "observable_relative_improvement": float(
                    debug["shared_observable_relative_improvement"]
                ),
                "raw_angle_deg": float(debug["shared_raw_angle_deg"]),
                "applied_angle_deg": float(debug["shared_applied_angle_deg"]),
            }
        )
        return corrected, debug

    return apply


def runtime_audit(cases: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "case_count": len(rows),
        "matched_person_count": int(sum(row["matched_count"] for row in rows)),
        "accepted_person_count": int(sum(row["accepted_count"] for row in rows)),
        "applied_boundary_count": int(
            sum(row["shared_orientation_applied"] for row in rows)
        ),
        "applied_person_count": int(
            sum(
                row["accepted_orientation_person_count"]
                for row in rows
                if row["shared_orientation_applied"]
            )
        ),
        "mean_applied_angle_deg": float(
            np.mean(
                [
                    row["applied_angle_deg"]
                    for row in rows
                    if row["shared_orientation_applied"]
                ]
            )
        )
        if any(row["shared_orientation_applied"] for row in rows)
        else 0.0,
        "rejected_exact_b0_max_abs_change": float(
            max(
                (
                    case["exact_b0_fallback_max_abs_change"]
                    for case in cases
                ),
                default=0.0,
            )
        ),
        "rows": rows,
    }


def evaluate(
    prepared: list[dict[str, Any]],
    config: SharedOrientationKabschConfig,
) -> dict[str, Any]:
    v1_cases, v1_summary = strict.evaluate_prepared_runtime(
        prepared, strict.callback_v1
    )
    rows: list[dict[str, Any]] = []
    candidate_cases, candidate_summary = strict.evaluate_prepared_runtime(
        prepared, callback(config, rows)
    )
    return {
        "methods": {
            "brtc_v1": {
                "metrics": common.compact(v1_summary),
                "summary": v1_summary,
                "cases": v1_cases,
            },
            "shared_kabsch": {
                "metrics": common.compact(candidate_summary),
                "summary": candidate_summary,
                "cases": candidate_cases,
            },
        },
        "runtime_audit": runtime_audit(candidate_cases, rows),
    }


def split_checks(value: dict[str, Any]) -> dict[str, bool]:
    reference = value["methods"]["brtc_v1"]["metrics"]
    candidate = value["methods"]["shared_kabsch"]["metrics"]
    audit = value["runtime_audit"]
    return {
        "root_exact": abs(candidate["root_error_m"] - reference["root_error_m"]) <= 1e-12,
        "joint_not_worse": candidate["joint_error_m"] <= reference["joint_error_m"] + 1e-12,
        "vertex_not_worse": candidate["vertex_error_m"] <= reference["vertex_error_m"] + 1e-12,
        "layout_exact": all(
            abs(candidate[key] - reference[key]) <= 1e-12
            for key in ("pairwise_distance_error_m", "pairwise_vector_error_m")
        ),
        "harm_exact": abs(
            candidate["root_harm_over_5cm_rate"]
            - reference["root_harm_over_5cm_rate"]
        )
        <= 1e-12,
        "camera_exact": candidate["camera_max_abs_change"] <= 1e-12,
        "rejected_exact_b0": audit["rejected_exact_b0_max_abs_change"] <= 1e-12,
        "nontrivial": audit["applied_boundary_count"] > 0,
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Shared/group SO(3) Kabsch after frozen BRTC",
        "",
        f"Phase: `{report['phase']}`. Policy inherited without tuning: "
        "`25 deg × 0.5`, observable improvement `>=0`.",
        "",
        "| Split | Method | Root | Joint | Vertex | Pair distance | Pair vector | Harm >5cm | Applied boundaries | Applied people |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, value in report["splits"].items():
        audit = value["runtime_audit"]
        for method in ("brtc_v1", "shared_kabsch"):
            metric = value["methods"][method]["metrics"]
            lines.append(
                f"| {split} | {method} | {metric['root_error_m']:.6f} | "
                f"{metric['joint_error_m']:.6f} | {metric['vertex_error_m']:.6f} | "
                f"{metric['pairwise_distance_error_m']:.6f} | "
                f"{metric['pairwise_vector_error_m']:.6f} | "
                f"{metric['root_harm_over_5cm_rate']:.1%} | "
                f"{audit['applied_boundary_count'] if method == 'shared_kabsch' else 0} | "
                f"{audit['applied_person_count'] if method == 'shared_kabsch' else 0} |"
            )
    lines.extend(
        [
            "",
            f"All required checks pass: `{report['decision']['all_required_checks_pass']}`.",
            f"Decision: **{report['decision']['status']}**.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    for path in (args.source_policy, args.output_dir):
        if not str(path.resolve()).startswith(str(REPO_ROOT.resolve())):
            raise ValueError("All paths must remain in Movie3R under /data")
    frozen, config = load_config(args.source_policy)
    if args.phase == "dev":
        split_rows = {
            "three_offset0": harness.load_rows(
                "dev", harness.DEFAULT_CONFIRM_REPORT, args.max_cases
            )
        }
    else:
        split_rows = {
            "three_offset1": harness.load_rows(
                "confirm", harness.DEFAULT_CONFIRM_REPORT, args.max_cases
            ),
            "dance": legacy.report_rows(("dance",)),
            "box": legacy.report_rows(("box",)),
        }
        if args.max_cases:
            split_rows = {name: rows[: args.max_cases] for name, rows in split_rows.items()}
    splits = {}
    for name, rows in split_rows.items():
        prepared = common.prepared_rows(rows, 0)
        splits[name] = evaluate(prepared, config)
        splits[name]["checks"] = split_checks(splits[name])
    all_pass = all(all(value["checks"].values()) for value in splits.values())
    if args.phase == "dev":
        status = "GO_SHARED_KABSCH_TO_BLIND_HELDOUT" if all_pass else "NO_GO_SHARED_KABSCH_MECHANISM"
        stem = "DEV_MECHANISM"
    else:
        status = "GO_SHARED_KABSCH_TO_EGO" if all_pass else "NO_GO_SHARED_KABSCH_HELDOUT"
        stem = "HELDOUT_RESULTS"
    report = {
        "experiment": "v14_brtc_shared_orientation_kabsch",
        "phase": args.phase,
        "policy_source": str(args.source_policy),
        "policy": frozen["policy"],
        "policy_sha256": frozen["policy_sha256"],
        "protocol": {
            "parameter_selection": "none; inherited from frozen individual Kabsch",
            "candidate_gt_use": "none",
            "future_frames": 0,
            "camera_update": "none",
            "root_update": "frozen BRTC only",
            "rejected_unmatched": "exact B0",
        },
        "splits": splits,
        "decision": {
            "all_required_checks_pass": all_pass,
            "status": status,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{stem}.json").write_text(
        json.dumps(common.jsonable(report), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    text = markdown(report)
    (args.output_dir / f"{stem}.md").write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    original_load, _ = common.install_cached_torch_load()
    try:
        main()
    finally:
        torch.load = original_load
