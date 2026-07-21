#!/usr/bin/env python3
"""Validate the selected V22 export and propagated shot-scale state."""

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
    / "v22_explicit_metric_bridge"
    / "selected_candidates"
    / "v22_selected_explicit_bridge.json"
)
DEFAULT_REPORT = (
    REPO_ROOT
    / "output"
    / "v22_explicit_metric_bridge"
    / "final_seed1"
    / "v22_explicit_metric_bridge.json"
)
DEFAULT_CHAIN = (
    REPO_ROOT
    / "output"
    / "v22_explicit_metric_bridge"
    / "chain_audit"
    / "v22_chain_scale_propagation_audit.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "output" / "v22_explicit_metric_bridge" / "integrity_audit"
SELECTED = "safe_gravity_absolute_scene_scale"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--chain", type=Path, default=DEFAULT_CHAIN)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rotation_tolerance", type=float, default=1e-4)
    parser.add_argument("--transform_tolerance", type=float, default=1e-6)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_values(value: object) -> bool:
    if isinstance(value, dict):
        return all(finite_values(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_values(item) for item in value)
    if isinstance(value, (int, float)):
        return bool(np.isfinite(value))
    return True


def transform_diagnostics(transform: list[list[float]]) -> dict:
    matrix = np.asarray(transform, dtype=np.float64)
    valid_shape = matrix.shape == (4, 4)
    if not valid_shape:
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


def collect_numeric_ranges(cases: list[dict]) -> dict:
    fields = {
        "old_human_scale": [row["old_shot_human_camera_scale"] for row in cases],
        "new_human_scale": [row["new_shot_human_camera_scale"] for row in cases],
        "old_background_scale": [row["old_shot_background_depth_scale"] for row in cases],
        "new_background_scale": [row["new_shot_background_depth_scale"] for row in cases],
    }
    return {
        name: {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
        }
        for name, values in fields.items()
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    export = load(args.export)
    report = load(args.report)
    chain = load(args.chain)
    report_cases = {row["case_name"]: row for row in report["cases"]}
    names = [row["case_name"] for row in export["cases"]]
    source_counts = Counter(row["source"] for row in export["cases"])
    expected_names = set(report_cases)

    transform_rows = []
    mismatches = []
    scale_errors = []
    for row in export["cases"]:
        name = row["case_name"]
        diagnostics = transform_diagnostics(row["boundary_transform"])
        diagnostics["case_name"] = name
        transform_rows.append(diagnostics)
        expected = np.asarray(
            report_cases[name]["variants"][SELECTED]["transform"], dtype=np.float64
        )
        actual = np.asarray(row["boundary_transform"], dtype=np.float64)
        difference = float(np.max(np.abs(expected - actual)))
        evaluation = np.asarray(row["evaluation"]["camera"]["transform"], dtype=np.float64)
        evaluation_difference = float(np.max(np.abs(evaluation - actual)))
        if difference > args.transform_tolerance or evaluation_difference > args.transform_tolerance:
            mismatches.append(
                {
                    "case_name": name,
                    "report_max_abs_difference": difference,
                    "evaluation_max_abs_difference": evaluation_difference,
                }
            )
        scales = np.asarray(
            [
                row["old_shot_human_camera_scale"],
                row["new_shot_human_camera_scale"],
                row["old_shot_background_depth_scale"],
                row["new_shot_background_depth_scale"],
            ],
            dtype=np.float64,
        )
        if not np.isfinite(scales).all() or np.any(scales <= 0.0):
            scale_errors.append({"case_name": name, "reason": "non-positive or non-finite"})
        for side in ("old", "new"):
            root = float(row[f"{side}_shot_human_camera_scale"])
            scene = float(row[f"{side}_shot_background_depth_scale"])
            ratio = scene / root
            if not 0.85 - 1e-6 <= ratio <= 1.15 + 1e-6:
                scale_errors.append(
                    {
                        "case_name": name,
                        "reason": f"{side} background/root ratio outside 15 percent bound",
                        "ratio": ratio,
                    }
                )

    invalid_transforms = [
        row
        for row in transform_rows
        if not row["valid_shape"]
        or not row["finite"]
        or row["bottom_row_error"] > args.rotation_tolerance
        or row["orthogonality_error"] > args.rotation_tolerance
        or abs(row["determinant"] - 1.0) > args.rotation_tolerance
    ]
    chain_finite = finite_values(chain)
    checks = {
        "case_count_180": len(export["cases"]) == 180,
        "unique_case_names": len(names) == len(set(names)),
        "matches_evaluation_case_set": set(names) == expected_names,
        "all_transforms_valid_se3": not invalid_transforms,
        "all_export_transforms_match_evaluation": not mismatches,
        "all_scales_positive_finite_and_bounded": not scale_errors,
        "chain_count_38": chain.get("chain_count") == 38 and len(chain.get("cases", [])) == 38,
        "chain_all_numeric_values_finite": chain_finite,
    }
    report_out = {
        "experiment": "V22 selected export and scale-state integrity audit",
        "passed": all(checks.values()),
        "checks": checks,
        "case_count": len(export["cases"]),
        "source_counts": dict(sorted(source_counts.items())),
        "scale_ranges": collect_numeric_ranges(export["cases"]),
        "transform_extrema": {
            "max_bottom_row_error": max(row["bottom_row_error"] for row in transform_rows),
            "max_orthogonality_error": max(
                row["orthogonality_error"] for row in transform_rows
            ),
            "min_determinant": min(row["determinant"] for row in transform_rows),
            "max_determinant": max(row["determinant"] for row in transform_rows),
        },
        "invalid_transforms": invalid_transforms,
        "transform_mismatches": mismatches,
        "scale_errors": scale_errors,
    }
    json_path = args.output_dir / "v22_export_integrity_audit.json"
    json_path.write_text(
        json.dumps(report_out, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report_out, indent=2, ensure_ascii=False))
    if not report_out["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
