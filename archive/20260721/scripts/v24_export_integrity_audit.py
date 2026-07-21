#!/usr/bin/env python3
"""Independently validate the selected V24 runtime export."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPORT = (
    REPO_ROOT
    / "output"
    / "v24_vggt_v22_rotation_bridge"
    / "selected_candidates"
    / "v24_selected_rotation_bridge.json"
)
DEFAULT_REPORT = (
    REPO_ROOT
    / "output"
    / "v24_vggt_v22_rotation_bridge"
    / "v24_vggt_v22_rotation_bridge.json"
)
DEFAULT_V22_EXPORT = (
    REPO_ROOT
    / "output"
    / "v22_explicit_metric_bridge"
    / "selected_candidates"
    / "v22_selected_explicit_bridge.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "output" / "v24_vggt_v22_rotation_bridge" / "integrity_audit"
)
SELECTED = "safe_tiered_extension_vggt"
EXPECTED_SOURCE_COUNTS = {
    "avatarrex": 48,
    "mvhuman100": 48,
    "mvhuman200": 36,
    "thuman": 48,
}
EXPECTED_RULE_COUNTS = {
    "keep_v22": 146,
    "large_torso_residual_cap25": 23,
    "torso_vggt_consensus_cap60": 7,
    "low_texture_conflict_cap45": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--v22_export", type=Path, default=DEFAULT_V22_EXPORT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rotation_tolerance", type=float, default=1e-4)
    parser.add_argument("--value_tolerance", type=float, default=1e-6)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def transform_diagnostics(transform: object) -> dict:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4):
        return {
            "valid_shape": False,
            "finite": bool(np.isfinite(matrix).all()),
            "bottom_row_error": float("inf"),
            "orthogonality_error": float("inf"),
            "determinant": float("nan"),
        }
    rotation = matrix[:3, :3]
    return {
        "valid_shape": True,
        "finite": bool(np.isfinite(matrix).all()),
        "bottom_row_error": float(
            np.max(np.abs(matrix[3] - np.asarray([0.0, 0.0, 0.0, 1.0])))
        ),
        "orthogonality_error": float(
            np.max(np.abs(rotation.T @ rotation - np.eye(3)))
        ),
        "determinant": float(np.linalg.det(rotation)),
    }


def max_numeric_difference(left: object, right: object) -> float:
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return float("inf")
        return max((max_numeric_difference(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return float("inf")
        return max((max_numeric_difference(a, b) for a, b in zip(left, right)), default=0.0)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    return 0.0 if left == right else float("inf")


def expected_rule(diagnostics: dict) -> str:
    if diagnostics["trigger_safe_low_texture_conflict"]:
        return "low_texture_conflict_cap45"
    if diagnostics["trigger_safe_large_residual"]:
        return "large_torso_residual_cap25"
    if diagnostics["trigger_safe_consensus"]:
        return "torso_vggt_consensus_cap60"
    return "keep_v22"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    export = load(args.export)
    report = load(args.report)
    v22_export = load(args.v22_export)
    report_cases = {row["case_name"]: row for row in report["cases"]}
    v22_cases = {row["case_name"]: row for row in v22_export["cases"]}
    names = [row["case_name"] for row in export["cases"]]

    invalid_transforms = []
    transform_mismatches = []
    evaluation_mismatches = []
    scale_state_mismatches = []
    rule_mismatches = []
    required_mismatches = []
    diagnostics_mismatches = []
    transform_rows = []

    scale_fields = (
        "old_shot_human_camera_scale",
        "new_shot_human_camera_scale",
        "old_shot_background_depth_scale",
        "new_shot_background_depth_scale",
    )
    for row in export["cases"]:
        name = row["case_name"]
        diagnostics = transform_diagnostics(row["boundary_transform"])
        diagnostics["case_name"] = name
        transform_rows.append(diagnostics)
        if (
            not diagnostics["valid_shape"]
            or not diagnostics["finite"]
            or diagnostics["bottom_row_error"] > args.rotation_tolerance
            or diagnostics["orthogonality_error"] > args.rotation_tolerance
            or abs(diagnostics["determinant"] - 1.0) > args.rotation_tolerance
        ):
            invalid_transforms.append(diagnostics)

        report_row = report_cases[name]
        selected = report_row["variants"][SELECTED]
        transform_difference = max_numeric_difference(
            row["boundary_transform"], selected["transform"]
        )
        evaluation_transform_difference = max_numeric_difference(
            row["boundary_transform"], row["evaluation"]["camera"]["transform"]
        )
        if (
            transform_difference > args.value_tolerance
            or evaluation_transform_difference > args.value_tolerance
        ):
            transform_mismatches.append(
                {
                    "case_name": name,
                    "report_max_abs_difference": transform_difference,
                    "evaluation_transform_max_abs_difference": evaluation_transform_difference,
                }
            )

        evaluation_difference = max_numeric_difference(
            row["evaluation"],
            {key: selected[key] for key in ("camera", "human", "scene")},
        )
        if evaluation_difference > args.value_tolerance:
            evaluation_mismatches.append(
                {"case_name": name, "max_abs_difference": evaluation_difference}
            )

        v22_row = v22_cases[name]
        field_differences = {
            field: abs(float(row[field]) - float(v22_row[field])) for field in scale_fields
        }
        gravity_matches = (
            row["gravity_residual_accepted"] == v22_row["gravity_residual_accepted"]
        )
        if max(field_differences.values()) > args.value_tolerance or not gravity_matches:
            scale_state_mismatches.append(
                {
                    "case_name": name,
                    "field_differences": field_differences,
                    "gravity_flag_matches": gravity_matches,
                }
            )

        expected_required = report_row["diagnostics"]["torso_residual_deg"] >= 10.0
        if bool(row["vggt_required"]) != bool(expected_required):
            required_mismatches.append(name)
        rule = expected_rule(report_row["diagnostics"])
        if row["accepted_rotation_rule"] != rule or bool(
            row["vggt_rotation_accepted"]
        ) != (rule != "keep_v22"):
            rule_mismatches.append(
                {
                    "case_name": name,
                    "exported": row["accepted_rotation_rule"],
                    "expected": rule,
                }
            )
        diagnostic_difference = max_numeric_difference(
            row["rotation_diagnostics"], report_row["diagnostics"]
        )
        if diagnostic_difference > args.value_tolerance:
            diagnostics_mismatches.append(
                {"case_name": name, "max_abs_difference": diagnostic_difference}
            )

    source_counts = Counter(row["source"] for row in export["cases"])
    rule_counts = Counter(row["accepted_rotation_rule"] for row in export["cases"])
    required_count = sum(bool(row["vggt_required"]) for row in export["cases"])
    accepted_count = sum(bool(row["vggt_rotation_accepted"]) for row in export["cases"])
    checks = {
        "case_count_180": len(export["cases"]) == 180,
        "unique_case_names": len(names) == len(set(names)),
        "matches_report_case_set": set(names) == set(report_cases),
        "matches_v22_scale_case_set": set(names) == set(v22_cases),
        "source_counts_match": dict(source_counts) == EXPECTED_SOURCE_COUNTS,
        "all_transforms_valid_se3": not invalid_transforms,
        "all_export_transforms_match_report": not transform_mismatches,
        "all_evaluation_values_match_report": not evaluation_mismatches,
        "all_v22_scale_state_is_preserved": not scale_state_mismatches,
        "all_vggt_required_flags_match": not required_mismatches,
        "all_rotation_rules_match_diagnostics": not rule_mismatches,
        "all_rotation_diagnostics_match_report": not diagnostics_mismatches,
        "accepted_rule_counts_match": dict(rule_counts) == EXPECTED_RULE_COUNTS,
        "vggt_required_count_88": required_count == 88,
        "vggt_accepted_count_34": accepted_count == 34,
    }
    output = {
        "experiment": "V24 selected runtime export integrity audit",
        "passed": all(checks.values()),
        "checks": checks,
        "case_count": len(export["cases"]),
        "source_counts": dict(sorted(source_counts.items())),
        "accepted_rule_counts": dict(sorted(rule_counts.items())),
        "vggt_required_count": required_count,
        "vggt_accepted_count": accepted_count,
        "transform_extrema": {
            "max_bottom_row_error": max(row["bottom_row_error"] for row in transform_rows),
            "max_orthogonality_error": max(
                row["orthogonality_error"] for row in transform_rows
            ),
            "min_determinant": min(row["determinant"] for row in transform_rows),
            "max_determinant": max(row["determinant"] for row in transform_rows),
        },
        "invalid_transforms": invalid_transforms,
        "transform_mismatches": transform_mismatches,
        "evaluation_mismatches": evaluation_mismatches,
        "scale_state_mismatches": scale_state_mismatches,
        "rule_mismatches": rule_mismatches,
        "required_mismatches": required_mismatches,
        "diagnostics_mismatches": diagnostics_mismatches,
    }
    output_path = args.output_dir / "v24_export_integrity_audit.json"
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))
    if not output["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
