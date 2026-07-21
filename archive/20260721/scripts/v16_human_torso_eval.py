#!/usr/bin/env python3
"""Evaluate V16 torso-geometry and V15+torso candidate caches."""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATTERN = str(
    REPO_ROOT
    / "output"
    / "v16_human_aware_rotation_residual"
    / "candidate_cache"
    / "v16_candidates_shard_*_of_*.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "output" / "v16_human_aware_rotation_residual" / "evaluation"
DEFAULT_V15 = REPO_ROOT / "output" / "v15_wide_baseline_boundary_bridge" / "candidate_cache"

METHODS = {
    "original_continue": ("baselines", "original_continue"),
    "hard_reset": ("baselines", "hard_reset"),
    "fixed_explicit": ("baselines", "fixed_explicit"),
    "v15_coarse": ("baselines", "v15_coarse"),
    "fixed_torso_last_1f_keep_t0": ("fixed_candidates", "fixed_torso_last_1f_keep_t0"),
    "fixed_torso_last_1f_resolve_t": ("fixed_candidates", "fixed_torso_last_1f_resolve_t"),
    "fixed_torso_motion_1f_keep_t0": ("fixed_candidates", "fixed_torso_motion_1f_keep_t0"),
    "fixed_torso_motion_1f_resolve_t": ("fixed_candidates", "fixed_torso_motion_1f_resolve_t"),
    "fixed_torso_motion_3f_resolve_t": ("fixed_candidates", "fixed_torso_motion_3f_resolve_t"),
    "fixed_gravity_1f_resolve_t": ("fixed_candidates", "fixed_gravity_1f_resolve_t"),
    "fixed_torso_motion_gravity_1f_resolve_t": ("fixed_candidates", "fixed_torso_motion_gravity_1f_resolve_t"),
    "fixed_torso_motion_1f_root_check": ("fixed_candidates", "fixed_torso_motion_1f_root_check"),
    "fixed_torso_motion_3f_root_check": ("fixed_candidates", "fixed_torso_motion_3f_root_check"),
    "v15_torso_motion_1f_keep_t0": ("v15_candidates", "v15_torso_motion_1f_keep_t0"),
    "v15_torso_motion_1f_resolve_t": ("v15_candidates", "v15_torso_motion_1f_resolve_t"),
    "v15_torso_motion_3f_resolve_t": ("v15_candidates", "v15_torso_motion_3f_resolve_t"),
    "v15_torso_motion_1f_root_check": ("v15_candidates", "v15_torso_motion_1f_root_check"),
    "v15_torso_motion_3f_root_check": ("v15_candidates", "v15_torso_motion_3f_root_check"),
    "boundary_oracle": ("baselines", "boundary_oracle"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_pattern", default=DEFAULT_PATTERN)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--v15_candidate_dir", type=Path, default=DEFAULT_V15)
    return parser.parse_args()


def load_cases(pattern: str) -> list[dict]:
    cases = []
    for path in sorted(glob.glob(pattern)):
        cases.extend(json.loads(Path(path).read_text(encoding="utf-8"))["cases"])
    names = [case["case_name"] for case in cases]
    if len(names) != len(set(names)):
        raise RuntimeError("Duplicate V16 cases")
    return cases


def method_row(case: dict, method: str) -> dict | None:
    group, key = METHODS[method]
    return case.get(group, {}).get(key)


def failed(row: dict | None) -> bool:
    return row is None or bool(row.get("fit_failed", False)) or not np.isfinite(row.get("camera_rotation_error_deg", np.nan))


def catastrophic(row: dict | None) -> bool:
    return failed(row) or float(row["camera_translation_error_m"]) > 1.0 or float(row["camera_rotation_error_deg"]) > 30.0


def success(row: dict | None) -> bool:
    return not failed(row) and float(row["camera_translation_error_m"]) < 0.25 and float(row["camera_rotation_error_deg"]) < 5.0


def strict_success(row: dict | None) -> bool:
    return not failed(row) and float(row["camera_translation_error_m"]) < 0.10 and float(row["camera_rotation_error_deg"]) < 2.0


def joint_cost(row: dict | None) -> float:
    if failed(row):
        return 1e6
    return float(row["camera_translation_error_m"]) / 0.25 + float(row["camera_rotation_error_deg"]) / 5.0


def distribution(values: list[float]) -> dict:
    array = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if not len(array):
        return {key: None for key in ("mean", "median", "p90", "p95", "max")}
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def baseline_for(method: str) -> str:
    return "v15_coarse" if method.startswith("v15_") else "fixed_explicit"


def stats(cases: list[dict], method: str) -> dict:
    rows = [method_row(case, method) for case in cases]
    valid = [row for row in rows if not failed(row)]
    base_rows = [method_row(case, baseline_for(method)) for case in cases]
    improvements = [
        float(base["camera_rotation_error_deg"]) - float(row["camera_rotation_error_deg"])
        for row, base in zip(rows, base_rows)
        if not failed(row) and not failed(base)
    ]
    false_corrections = [
        float(base["camera_rotation_error_deg"]) < 10.0
        and float(row["camera_rotation_error_deg"]) > float(base["camera_rotation_error_deg"]) + 1.0
        for row, base in zip(rows, base_rows)
        if not failed(row) and not failed(base)
    ]
    return {
        "case_count": len(cases),
        "fit_failure_rate": float(np.mean([failed(row) for row in rows])),
        "translation_m": distribution([float(row["camera_translation_error_m"]) for row in valid]),
        "rotation_deg": distribution([float(row["camera_rotation_error_deg"]) for row in valid]),
        "yaw_deg": distribution([float(row.get("yaw_error_deg", np.nan)) for row in valid]),
        "pitch_deg": distribution([float(row.get("pitch_error_deg", np.nan)) for row in valid]),
        "roll_deg": distribution([float(row.get("roll_error_deg", np.nan)) for row in valid]),
        "human_root_jump_m": distribution([float(row.get("human_root_jump_m", np.nan)) for row in valid]),
        "human_torso_jump_deg": distribution([float(row.get("human_torso_jump_deg", np.nan)) for row in valid]),
        "catastrophic_rate": float(np.mean([catastrophic(row) for row in rows])),
        "success_rate": float(np.mean([success(row) for row in rows])),
        "strict_success_rate": float(np.mean([strict_success(row) for row in rows])),
        "rotation_helpful_rate": float(np.mean([gain > 0.5 for gain in improvements])) if improvements else None,
        "rotation_harmful_rate": float(np.mean([gain < -0.5 for gain in improvements])) if improvements else None,
        "rotation_gain_mean_deg": float(np.mean(improvements)) if improvements else None,
        "false_correction_rate_on_lt10": float(np.mean(false_corrections)) if false_corrections else None,
        "joint_gain_mean": float(
            np.mean([joint_cost(base) - joint_cost(row) for row, base in zip(rows, base_rows)])
        ),
    }


def aggregate(cases: list[dict]) -> dict:
    return {method: stats(cases, method) for method in METHODS}


def initial_rotation(case: dict) -> float:
    return float(method_row(case, "fixed_explicit")["camera_rotation_error_deg"])


def source_groups(cases: list[dict]) -> dict:
    return {
        source: aggregate([case for case in cases if str(case["record"]["source"]) == source])
        for source in sorted({str(case["record"]["source"]) for case in cases})
    }


def initial_rotation_groups(cases: list[dict]) -> dict:
    buckets = (("lt10", 0.0, 10.0), ("10_30", 10.0, 30.0), ("30_60", 30.0, 60.0), ("ge60", 60.0, float("inf")))
    output = {}
    for name, lower, upper in buckets:
        subset = [case for case in cases if lower <= initial_rotation(case) < upper]
        if subset:
            output[name] = aggregate(subset)
    return output


def motion_groups(cases: list[dict]) -> dict:
    buckets = (("slow", 0.0, 2.0), ("medium", 2.0, 5.0), ("fast", 5.0, float("inf")))
    output = {}
    for name, lower, upper in buckets:
        subset = [
            case
            for case in cases
            if lower <= float(case["motion_diagnostics"].get("angular_speed_deg_per_frame", np.nan)) < upper
        ]
        if subset:
            output[name] = aggregate(subset)
    return output


def texture_groups(cases: list[dict]) -> dict:
    values = np.asarray([float(case["texture_score"]) for case in cases])
    low, high = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    groups = {
        "low": [case for case in cases if float(case["texture_score"]) <= low],
        "medium": [case for case in cases if low < float(case["texture_score"]) <= high],
        "high": [case for case in cases if float(case["texture_score"]) > high],
    }
    return {name: aggregate(rows) for name, rows in groups.items() if rows}


def angle_groups(cases: list[dict]) -> dict:
    return {
        bucket: aggregate([case for case in cases if str(case["record"].get("angle_bucket")) == bucket])
        for bucket in sorted({str(case["record"].get("angle_bucket")) for case in cases})
    }


def load_human_ratios(root: Path) -> dict[str, float]:
    ratios = {}
    for path in sorted(glob.glob(str(root / "v15_candidates_shard_*_of_*.json"))):
        for case in json.loads(Path(path).read_text(encoding="utf-8"))["cases"]:
            ratios[str(case["case_name"])] = float(case.get("human_image_ratio", float("nan")))
    return ratios


def human_ratio_groups(cases: list[dict], ratios: dict[str, float]) -> dict:
    valid = np.asarray([ratios.get(str(case["case_name"]), np.nan) for case in cases], dtype=np.float64)
    finite = valid[np.isfinite(valid)]
    if not len(finite):
        return {}
    low, high = np.quantile(finite, [1.0 / 3.0, 2.0 / 3.0])
    groups = {
        "low_visible_area": [case for case in cases if ratios.get(str(case["case_name"]), np.nan) <= low],
        "medium_visible_area": [
            case for case in cases if low < ratios.get(str(case["case_name"]), np.nan) <= high
        ],
        "high_visible_area": [case for case in cases if ratios.get(str(case["case_name"]), np.nan) > high],
    }
    return {name: aggregate(rows) for name, rows in groups.items() if rows}


def residual_correlation(cases: list[dict], method: str) -> dict:
    rows = []
    for case in cases:
        candidate = method_row(case, method)
        base = method_row(case, baseline_for(method))
        if failed(candidate) or failed(base):
            continue
        raw = candidate.get("raw_residual_deg")
        if raw is None:
            torso = candidate.get("torso", {})
            raw = torso.get("raw_residual_deg")
        if raw is None or not np.isfinite(raw):
            continue
        gain = float(base["camera_rotation_error_deg"]) - float(candidate["camera_rotation_error_deg"])
        rows.append((abs(float(raw)), gain))
    if len(rows) < 3:
        return {"count": len(rows), "pearson": None}
    array = np.asarray(rows)
    return {"count": len(rows), "pearson": float(np.corrcoef(array[:, 0], array[:, 1])[0, 1])}


def automatic_decision(overall: dict, by_source: dict) -> dict:
    one = overall["fixed_torso_motion_1f_root_check"]
    three = overall["fixed_torso_motion_3f_root_check"]
    fixed = overall["fixed_explicit"]
    source_gains = {}
    for source, group in by_source.items():
        source_gains[source] = (
            group["fixed_explicit"]["rotation_deg"]["mean"]
            - group["fixed_torso_motion_1f_root_check"]["rotation_deg"]["mean"]
        )
    improved_sources = sum(gain > 0.5 for gain in source_gains.values())
    proceed_token = bool(
        improved_sources >= 3
        and one["rotation_deg"]["p90"] < fixed["rotation_deg"]["p90"] - 1.0
    )
    v15_gains = {}
    for source, group in by_source.items():
        v15_gains[source] = (
            group["v15_coarse"]["rotation_deg"]["mean"]
            - group["v15_torso_motion_1f_root_check"]["rotation_deg"]["mean"]
        )
    return {
        "proceed_to_loso_human_token": proceed_token,
        "improved_fixed_sources": improved_sources,
        "fixed_source_rotation_gains_deg": source_gains,
        "one_frame_rotation_gain_deg": fixed["rotation_deg"]["mean"] - one["rotation_deg"]["mean"],
        "three_frame_rotation_gain_deg": fixed["rotation_deg"]["mean"] - three["rotation_deg"]["mean"],
        "three_minus_one_rotation_deg": three["rotation_deg"]["mean"] - one["rotation_deg"]["mean"],
        "one_frame_false_correction_rate": one["false_correction_rate_on_lt10"],
        "three_frame_false_correction_rate": three["false_correction_rate_on_lt10"],
        "v15_source_rotation_gains_deg": v15_gains,
        "v15_improved_sources": sum(gain > 0.5 for gain in v15_gains.values()),
        "translation_resolving_delta_m": (
            overall["fixed_torso_motion_1f_resolve_t"]["translation_m"]["mean"]
            - overall["fixed_torso_motion_1f_keep_t0"]["translation_m"]["mean"]
        ),
    }


def write_markdown(path: Path, report: dict) -> None:
    overall = report["overall"]
    selected = (
        "fixed_explicit",
        "fixed_torso_last_1f_resolve_t",
        "fixed_torso_motion_1f_keep_t0",
        "fixed_torso_motion_1f_resolve_t",
        "fixed_torso_motion_3f_resolve_t",
        "fixed_gravity_1f_resolve_t",
        "fixed_torso_motion_gravity_1f_resolve_t",
        "fixed_torso_motion_1f_root_check",
        "fixed_torso_motion_3f_root_check",
        "v15_coarse",
        "v15_torso_motion_1f_root_check",
        "v15_torso_motion_3f_root_check",
        "boundary_oracle",
    )
    lines = [
        "# V16 Predicted Torso Geometry Evaluation",
        "",
        "| Method | T mean | R mean | R P90 | R P95 | Catastrophic | Helpful R | Harmful R | False correction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in selected:
        row = overall[method]
        false = row["false_correction_rate_on_lt10"]
        lines.append(
            f"| {method} | {row['translation_m']['mean']:.3f} | {row['rotation_deg']['mean']:.2f} | "
            f"{row['rotation_deg']['p90']:.2f} | {row['rotation_deg']['p95']:.2f} | "
            f"{100.0 * row['catastrophic_rate']:.1f}% | {100.0 * row['rotation_helpful_rate']:.1f}% | "
            f"{100.0 * row['rotation_harmful_rate']:.1f}% | "
            f"{'n/a' if false is None else f'{100.0 * false:.1f}%'} |"
        )
    decision = report["automatic_decision"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Proceed to LOSO human-token probe: `{decision['proceed_to_loso_human_token']}`.",
            f"- Fixed-source mean rotation gains: `{decision['fixed_source_rotation_gains_deg']}`.",
            f"- One-frame gain: `{decision['one_frame_rotation_gain_deg']:.2f} deg`.",
            f"- Three-frame gain: `{decision['three_frame_rotation_gain_deg']:.2f} deg`.",
            f"- Three-frame minus one-frame: `{decision['three_minus_one_rotation_deg']:+.2f} deg`.",
            f"- Translation resolve minus keep t0: `{decision['translation_resolving_delta_m']:+.4f} m`.",
            f"- V15-source gains: `{decision['v15_source_rotation_gains_deg']}`.",
            "- Current loader uses `max_humans=1`; this experiment cannot support a multi-person conclusion.",
            "",
            "## By Source",
            "",
        ]
    )
    for source, group in report["by_source"].items():
        fixed = group["fixed_explicit"]
        one = group["fixed_torso_motion_1f_root_check"]
        three = group["fixed_torso_motion_3f_root_check"]
        lines.append(
            f"- **{source}**: Fixed `{fixed['translation_m']['mean']:.3f} m / {fixed['rotation_deg']['mean']:.2f} deg`; "
            f"1f `{one['translation_m']['mean']:.3f} m / {one['rotation_deg']['mean']:.2f} deg`; "
            f"3f `{three['translation_m']['mean']:.3f} m / {three['rotation_deg']['mean']:.2f} deg`."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args.candidate_pattern)
    if not cases:
        raise RuntimeError(f"No V16 cases matched {args.candidate_pattern}")
    overall = aggregate(cases)
    by_source = source_groups(cases)
    human_ratios = load_human_ratios(args.v15_candidate_dir)
    report = {
        "experiment": "V16 Predicted Torso Geometry Evaluation",
        "case_count": len(cases),
        "protocol": {
            "human3r_frozen": True,
            "gt_camera_use": "evaluation only",
            "gt_depth_used": False,
            "max_humans": 1,
            "translation_from_human_root": False,
        },
        "overall": overall,
        "by_source": by_source,
        "by_initial_rotation": initial_rotation_groups(cases),
        "by_motion_speed": motion_groups(cases),
        "by_texture": texture_groups(cases),
        "by_angle_bucket": angle_groups(cases),
        "by_human_visible_area": human_ratio_groups(cases, human_ratios),
        "residual_gain_correlation": {
            method: residual_correlation(cases, method)
            for method in ("fixed_torso_motion_1f_resolve_t", "fixed_torso_motion_3f_resolve_t")
        },
        "automatic_decision": automatic_decision(overall, by_source),
        "cases": [
            {
                "case_name": case["case_name"],
                "source": case["record"]["source"],
                "texture_score": case["texture_score"],
                "motion_diagnostics": case["motion_diagnostics"],
            }
            for case in cases
        ],
    }
    path = args.output_dir / "v16_torso_geometry_eval.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    write_markdown(args.output_dir / "v16_torso_geometry_summary.md", report)
    print(json.dumps({"case_count": len(cases), "overall": overall, "decision": report["automatic_decision"]}, indent=2), flush=True)
    print(f">> wrote {path}", flush=True)


if __name__ == "__main__":
    main()
