#!/usr/bin/env python3
"""Merge V13 stage-1 scene-coordinate Oracle shards and apply the route gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "output" / "v13_world_coordinate_memory" / "stage1_scene_coordinate_oracle"
CANONICAL = (
    "same_view_teacher_static_1f_1024_spatial_se3",
    "same_view_teacher_static_1f_1024_spatial_sim3",
    "same_view_teacher_static_3f_1024_spatial_se3",
    "same_view_teacher_static_3f_1024_spatial_sim3",
    "history_coverage_teacher_static_1f_1024_spatial_se3",
    "history_coverage_teacher_static_1f_1024_spatial_sim3",
    "history_coverage_teacher_static_3f_1024_spatial_se3",
    "history_coverage_teacher_static_3f_1024_spatial_sim3",
    "history_memory_static_1f_1024_spatial_se3",
    "history_memory_static_1f_1024_spatial_sim3",
    "history_memory_static_3f_1024_spatial_se3",
    "history_memory_static_3f_1024_spatial_sim3",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", type=Path, default=None)
    return parser.parse_args()


def finite(values) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    return array[np.isfinite(array)]


def stats(values) -> dict:
    array = finite(values)
    if not len(array):
        return {key: None for key in ("count", "mean", "median", "p90", "p95", "max")}
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def method_row(case: dict, method: str) -> dict | None:
    if method in case["variants"]:
        return case["variants"][method]
    return case["baselines"].get(method)


def summarize(cases: list[dict], method: str) -> dict:
    rows = [method_row(case, method) for case in cases]
    valid_rows = [row for row in rows if row is not None]
    translation = finite(row.get("camera_translation_error_m", np.nan) for row in valid_rows)
    rotation = finite(row.get("camera_rotation_error_deg", np.nan) for row in valid_rows)
    paired = [
        (float(row["camera_translation_error_m"]), float(row["camera_rotation_error_deg"]))
        for row in valid_rows
        if math.isfinite(float(row.get("camera_translation_error_m", np.nan)))
        and math.isfinite(float(row.get("camera_rotation_error_deg", np.nan)))
    ]
    successful_fit_count = len(paired)
    total_count = len(cases)
    return {
        "case_count": total_count,
        "successful_fit_count": successful_fit_count,
        "fit_failure_rate": float(1.0 - successful_fit_count / max(total_count, 1)),
        "translation_m": stats(translation),
        "rotation_deg": stats(rotation),
        "yaw_deg": stats(row.get("yaw_error_deg", np.nan) for row in valid_rows),
        "pitch_deg": stats(row.get("pitch_error_deg", np.nan) for row in valid_rows),
        "roll_deg": stats(row.get("roll_error_deg", np.nan) for row in valid_rows),
        "translation_x_m": stats(row.get("translation_error_xyz_m", [np.nan] * 3)[0] for row in valid_rows),
        "translation_y_m": stats(row.get("translation_error_xyz_m", [np.nan] * 3)[1] for row in valid_rows),
        "translation_z_m": stats(row.get("translation_error_xyz_m", [np.nan] * 3)[2] for row in valid_rows),
        "scale": stats(row.get("estimated_scale", np.nan) for row in valid_rows),
        "scale_log_abs": stats(row.get("scale_log_abs", np.nan) for row in valid_rows),
        "fit_residual_m": stats(row.get("fit_residual_mean_m", np.nan) for row in valid_rows),
        "inlier_ratio_0_20m": stats(row.get("inlier_ratio_0_20m", np.nan) for row in valid_rows),
        "image_coverage": stats(row.get("image_coverage_8x8", np.nan) for row in valid_rows),
        "condition_number": stats(row.get("geometry_condition_number", np.nan) for row in valid_rows),
        "planarity_ratio": stats(row.get("planarity_ratio", np.nan) for row in valid_rows),
        "duplicate_geometry_ratio": stats(row.get("duplicate_geometry_ratio", np.nan) for row in valid_rows),
        "rates": {
            "strict_success": float(sum(t < 0.10 and r < 2.0 for t, r in paired) / max(total_count, 1)),
            "success": float(sum(t < 0.25 and r < 5.0 for t, r in paired) / max(total_count, 1)),
            "catastrophic": float(
                (total_count - successful_fit_count + sum(t > 1.0 or r > 30.0 for t, r in paired))
                / max(total_count, 1)
            ),
        },
    }


def tertile(cases: list[dict], getter, names=("low", "medium", "high")) -> dict[str, str]:
    values = {case["case_name"]: float(getter(case)) for case in cases}
    array = finite(values.values())
    low, high = np.percentile(array, [33.333, 66.667])
    return {
        name: names[0] if value <= low else names[1] if value <= high else names[2]
        for name, value in values.items()
    }


def grouped(cases: list[dict]) -> dict:
    texture = tertile(cases, lambda case: case["texture_score"])
    overlap = tertile(
        cases,
        lambda case: np.mean(case["correspondence"]["static"]["frame_overlap_ratio_0_20m"]),
    )
    translation = tertile(
        cases,
        lambda case: case["baselines"]["hard_reset_no_alignment"]["camera_translation_error_m"],
    )
    def planarity_value(case: dict) -> float:
        row = case["variants"].get("history_memory_static_3f_1024_spatial_se3")
        return float(row.get("planarity_ratio", 0.0)) if row is not None else 0.0

    planarity = tertile(cases, planarity_value, ("planar", "medium", "nondegenerate"))
    groups: dict[str, dict[str, list[dict]]] = {
        "source": defaultdict(list),
        "angle_bucket": defaultdict(list),
        "texture_tertile": defaultdict(list),
        "overlap_tertile": defaultdict(list),
        "translation_tertile": defaultdict(list),
        "geometry_tertile": defaultdict(list),
    }
    for case in cases:
        name = case["case_name"]
        groups["source"][str(case["record"].get("source", "unknown"))].append(case)
        groups["angle_bucket"][str(case["record"].get("angle_bucket", "unknown"))].append(case)
        groups["texture_tertile"][texture[name]].append(case)
        groups["overlap_tertile"][overlap[name]].append(case)
        groups["translation_tertile"][translation[name]].append(case)
        groups["geometry_tertile"][planarity[name]].append(case)
    methods = ("fixed_explicit",) + CANONICAL
    return {
        group_name: {
            label: {method: summarize(items, method) for method in methods}
            for label, items in sorted(rows.items())
        }
        for group_name, rows in groups.items()
    }


def paired_delta(cases: list[dict], first: str, second: str, metric: str) -> dict:
    before, after = [], []
    for case in cases:
        a, b = method_row(case, first), method_row(case, second)
        if a is None or b is None:
            continue
        va, vb = float(a.get(metric, np.nan)), float(b.get(metric, np.nan))
        if math.isfinite(va) and math.isfinite(vb):
            before.append(va)
            after.append(vb)
    before, after = np.asarray(before), np.asarray(after)
    delta = after - before
    return {
        "count": int(len(delta)),
        "first_mean": float(before.mean()),
        "second_mean": float(after.mean()),
        "mean_delta": float(delta.mean()),
        "second_better_rate": float(np.mean(delta < 0.0)),
    }


def point_budget_summary(cases: list[dict]) -> dict:
    output = {}
    for mode in ("same_view_teacher", "history_coverage_teacher", "history_memory"):
        for frame_count in (1, 3):
            for selection in ("confidence", "spatial"):
                for transform in ("se3", "sim3"):
                    key = f"{mode}_{frame_count}f_{selection}_{transform}"
                    output[key] = {}
                    for count in (64, 256, 1024, 4096):
                        method = f"{mode}_static_{frame_count}f_{count}_{selection}_{transform}"
                        output[key][str(count)] = summarize(cases, method)
    return output


def route_decision(report: dict) -> dict:
    fixed = report["overall"]["fixed_explicit"]
    same_se3 = report["overall"]["same_view_teacher_static_1f_1024_spatial_se3"]
    same_sim3 = report["overall"]["same_view_teacher_static_1f_1024_spatial_sim3"]
    covered_3f = report["overall"]["history_coverage_teacher_static_3f_1024_spatial_se3"]
    memory_1f = report["overall"]["history_memory_static_1f_1024_spatial_se3"]
    memory_3f = report["overall"]["history_memory_static_3f_1024_spatial_se3"]
    criteria = {
        "pseudo_oracle_rotation_below_2deg": same_se3["rotation_deg"]["mean"] < 2.0,
        "pseudo_oracle_translation_below_0_25m": same_se3["translation_m"]["mean"] < 0.25,
        "pseudo_oracle_translation_better_than_fixed": same_se3["translation_m"]["mean"] < 0.5 * fixed["translation_m"]["mean"],
        "pseudo_oracle_catastrophic_below_fixed": same_se3["rates"]["catastrophic"] < fixed["rates"]["catastrophic"],
        "history_covered_ideal_rotation_below_2deg": covered_3f["rotation_deg"]["mean"] < 2.0,
        "history_covered_ideal_translation_below_0_25m": covered_3f["translation_m"]["mean"] < 0.25,
        "history_covered_ideal_fit_failure_below_10_percent": covered_3f["fit_failure_rate"] < 0.10,
        "history_covered_ideal_catastrophic_below_10_percent": covered_3f["rates"]["catastrophic"] < 0.10,
        "historical_memory_better_than_fixed": (
            memory_3f["translation_m"]["mean"] < fixed["translation_m"]["mean"]
            and memory_3f["rotation_deg"]["mean"] < fixed["rotation_deg"]["mean"]
        ),
        "historical_memory_translation_below_0_5m": memory_3f["translation_m"]["mean"] < 0.50,
        "historical_memory_rotation_below_5deg": memory_3f["rotation_deg"]["mean"] < 5.0,
        "historical_memory_catastrophic_below_10_percent": memory_3f["rates"]["catastrophic"] < 0.10,
    }
    sim3_gain_t = same_se3["translation_m"]["mean"] - same_sim3["translation_m"]["mean"]
    sim3_gain_r = same_se3["rotation_deg"]["mean"] - same_sim3["rotation_deg"]["mean"]
    three_frame_gain_t = memory_1f["translation_m"]["mean"] - memory_3f["translation_m"]["mean"]
    three_frame_gain_r = memory_1f["rotation_deg"]["mean"] - memory_3f["rotation_deg"]["mean"]
    geometry_upper_bound_keys = (
        "pseudo_oracle_rotation_below_2deg",
        "pseudo_oracle_translation_below_0_25m",
        "pseudo_oracle_translation_better_than_fixed",
        "pseudo_oracle_catastrophic_below_fixed",
    )
    historical_coverage_keys = (
        "history_covered_ideal_rotation_below_2deg",
        "history_covered_ideal_translation_below_0_25m",
        "history_covered_ideal_fit_failure_below_10_percent",
        "history_covered_ideal_catastrophic_below_10_percent",
    )
    historical_anchor_keys = (
        "historical_memory_better_than_fixed",
        "historical_memory_translation_below_0_5m",
        "historical_memory_rotation_below_5deg",
        "historical_memory_catastrophic_below_10_percent",
    )
    geometry_upper_bound_pass = bool(all(criteria[key] for key in geometry_upper_bound_keys))
    historical_coverage_pass = bool(all(criteria[key] for key in historical_coverage_keys))
    historical_anchor_pass = bool(all(criteria[key] for key in historical_anchor_keys))
    continue_stage2 = geometry_upper_bound_pass
    return {
        "true_oracle_available": False,
        "stage1_gate_is_provisional": True,
        "criteria": criteria,
        "sim3_gain_translation_m": float(sim3_gain_t),
        "sim3_gain_rotation_deg": float(sim3_gain_r),
        "sim3_materially_better": bool(sim3_gain_t > 0.05 or sim3_gain_r > 1.0),
        "three_frame_gain_translation_m": float(three_frame_gain_t),
        "three_frame_gain_rotation_deg": float(three_frame_gain_r),
        "three_frame_materially_better": bool(three_frame_gain_t > 0.05 or three_frame_gain_r > 1.0),
        "geometry_upper_bound_pass": geometry_upper_bound_pass,
        "historical_scene_coverage_pass": historical_coverage_pass,
        "historical_anchor_coordinate_pass": historical_anchor_pass,
        "continue_to_stage2_frozen_descriptor_diagnosis": continue_stage2,
        "continue_to_deployable_world_memory_pipeline": bool(
            geometry_upper_bound_pass and historical_coverage_pass and historical_anchor_pass
        ),
        "recommended_next_step": (
            "stop_world_memory_matching_and_improve_local_geometry_or_scale"
            if not geometry_upper_bound_pass
            else (
                "increase_causal_scene_coverage_or_use_fixed_three_frame_wait"
                if not historical_coverage_pass
                else (
                    "improve_causal_world_anchor_coordinates_before_matcher_training"
                    if not historical_anchor_pass
                    else "diagnose_keyframe_retrieval_and_frozen_correspondence"
                )
            )
        ),
    }


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# V13 Stage-1 Scene-Coordinate Oracle",
        "",
        f"Cases: {report['case_count']}",
        "",
        "> True static-scene GT depth/scan is unavailable in the current 180-cut data. All scene-coordinate results below are explicitly labeled offline-teacher pseudo Oracle.",
        "",
        "| Method | T mean | T median | T P90 | R mean | R median | R P90 | Fit fail | Success | Catastrophic |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ("hard_reset_no_alignment", "fixed_explicit", "boundary_oracle") + CANONICAL:
        row = report["overall"][method]
        lines.append(
            f"| {method} | {row['translation_m']['mean']:.4f} | {row['translation_m']['median']:.4f} | "
            f"{row['translation_m']['p90']:.4f} | {row['rotation_deg']['mean']:.3f} | "
            f"{row['rotation_deg']['median']:.3f} | {row['rotation_deg']['p90']:.3f} | "
            f"{100.0 * row['fit_failure_rate']:.1f}% | {100.0 * row['rates']['success']:.1f}% | "
            f"{100.0 * row['rates']['catastrophic']:.1f}% |"
        )
    lines.extend(["", "## Decision", ""])
    for key, value in report["decision"]["criteria"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            f"- Sim(3) materially better: {report['decision']['sim3_materially_better']}",
            f"- Three-frame materially better: {report['decision']['three_frame_materially_better']}",
            f"- Geometry upper bound pass: {report['decision']['geometry_upper_bound_pass']}",
            f"- Historical scene coverage pass: {report['decision']['historical_scene_coverage_pass']}",
            f"- Historical anchor coordinate pass: {report['decision']['historical_anchor_coordinate_pass']}",
            f"- Continue frozen descriptor diagnosis: {report['decision']['continue_to_stage2_frozen_descriptor_diagnosis']}",
            f"- Continue deployable World Memory pipeline: {report['decision']['continue_to_deployable_world_memory_pipeline']}",
            f"- Recommended: `{report['decision']['recommended_next_step']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.input_dir / "merged")
    output_dir.mkdir(parents=True, exist_ok=True)
    shards = sorted(args.input_dir.glob("stage1_shard_*_of_*.json"))
    if not shards:
        raise FileNotFoundError(args.input_dir)
    cases = []
    for shard in shards:
        cases.extend(json.loads(shard.read_text(encoding="utf-8"))["cases"])
    names = [case["case_name"] for case in cases]
    if len(names) != len(set(names)):
        raise RuntimeError("Duplicate V13 Stage-1 cases")
    all_variants = sorted({name for case in cases for name in case["variants"]})
    methods = ("hard_reset_no_alignment", "fixed_explicit", "boundary_oracle") + tuple(all_variants)
    report = {
        "experiment": "V13 Stage-1 Scene-Coordinate Oracle",
        "case_count": len(cases),
        "true_oracle_status": "unavailable_in_current_180_cut_data",
        "overall": {method: summarize(cases, method) for method in methods},
        "groups": grouped(cases),
        "point_budget": point_budget_summary(cases),
        "paired": {
            "same_view_sim3_vs_se3_translation": paired_delta(
                cases,
                "same_view_teacher_static_1f_1024_spatial_se3",
                "same_view_teacher_static_1f_1024_spatial_sim3",
                "camera_translation_error_m",
            ),
            "history_three_vs_one_translation": paired_delta(
                cases,
                "history_memory_static_1f_1024_spatial_se3",
                "history_memory_static_3f_1024_spatial_se3",
                "camera_translation_error_m",
            ),
            "history_three_vs_one_rotation": paired_delta(
                cases,
                "history_memory_static_1f_1024_spatial_se3",
                "history_memory_static_3f_1024_spatial_se3",
                "camera_rotation_error_deg",
            ),
        },
    }
    report["decision"] = route_decision(report)
    (output_dir / "stage1_merged.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    rows = []
    for case in cases:
        row = {
            "case_name": case["case_name"],
            "source": case["record"].get("source"),
            "angle_bucket": case["record"].get("angle_bucket"),
            "texture_score": case["texture_score"],
            "overlap_020": np.mean(case["correspondence"]["static"]["frame_overlap_ratio_0_20m"]),
        }
        for method in ("fixed_explicit",) + CANONICAL:
            item = method_row(case, method)
            row[f"{method}_t"] = item["camera_translation_error_m"] if item is not None else float("nan")
            row[f"{method}_r"] = item["camera_rotation_error_deg"] if item is not None else float("nan")
        rows.append(row)
    with (output_dir / "stage1_cases.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(output_dir / "stage1_summary.md", report)
    print(f">> merged {len(cases)} cases", flush=True)
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
