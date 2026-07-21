#!/usr/bin/env python3
"""Merge V13 Stage-2 frozen world-memory shards and diagnose the bottleneck."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "output" / "v13_world_coordinate_memory" / "stage2_frozen_world_memory"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", type=Path, default=None)
    return parser.parse_args()


def stats(values) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {key: None for key in ("count", "mean", "median", "p90", "p95")}
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
    }


def method_row(case: dict, method: str) -> dict | None:
    if method in case["variants"]:
        return case["variants"][method]
    return case["baselines"].get(method)


def summarize(cases: list[dict], method: str) -> dict:
    rows = [method_row(case, method) for case in cases]
    valid_rows = [row for row in rows if row is not None]
    paired = [
        (float(row.get("camera_translation_error_m", np.nan)), float(row.get("camera_rotation_error_deg", np.nan)))
        for row in valid_rows
    ]
    paired = [(t, r) for t, r in paired if math.isfinite(t) and math.isfinite(r)]
    successful_fit_count = len(paired)
    total_count = len(cases)
    return {
        "case_count": total_count,
        "successful_fit_count": successful_fit_count,
        "fit_failure_rate": float(1.0 - successful_fit_count / max(total_count, 1)),
        "translation_m": stats(t for t, _ in paired),
        "rotation_deg": stats(r for _, r in paired),
        "physical_match_mean_m": stats(row.get("physical_match_mean_m", np.nan) for row in valid_rows),
        "physical_accuracy_010": stats(row.get("physical_accuracy_010", np.nan) for row in valid_rows),
        "physical_accuracy_020": stats(row.get("physical_accuracy_020", np.nan) for row in valid_rows),
        "physical_accuracy_050": stats(row.get("physical_accuracy_050", np.nan) for row in valid_rows),
        "inlier_ratio_020": stats(row.get("inlier_ratio_0_20m", np.nan) for row in valid_rows),
        "rates": {
            "strict_success": float(sum(t < 0.10 and r < 2.0 for t, r in paired) / max(total_count, 1)),
            "success": float(sum(t < 0.25 and r < 5.0 for t, r in paired) / max(total_count, 1)),
            "catastrophic": float(
                (total_count - successful_fit_count + sum(t > 1.0 or r > 30.0 for t, r in paired))
                / max(total_count, 1)
            ),
        },
    }


def retrieval_summary(cases: list[dict]) -> dict:
    output = {}
    descriptors = sorted({name for case in cases for name in case["descriptor_names"]})
    for frame_count in (1, 3):
        for descriptor in descriptors:
            rows = [case["retrieval"].get(str(frame_count), {}).get(descriptor) for case in cases]
            rows = [row for row in rows if row is not None]
            output[f"{descriptor}_{frame_count}f"] = {
                "case_count": len(rows),
                "mean_oracle_rank": float(np.mean([row["oracle_keyframe_rank"] for row in rows])) if rows else None,
                "recall_at_1": float(np.mean([row["recall_at_1"] for row in rows])) if rows else None,
                "recall_at_3": float(np.mean([row["recall_at_3"] for row in rows])) if rows else None,
                "recall_at_5": float(np.mean([row["recall_at_5"] for row in rows])) if rows else None,
            }
    return output


def grouped(cases: list[dict], methods: list[str]) -> dict:
    groups: dict[str, dict[str, list[dict]]] = {
        "source": defaultdict(list),
        "angle_bucket": defaultdict(list),
    }
    texture_values = np.asarray([float(case["texture_score"]) for case in cases])
    texture_low, texture_high = np.percentile(texture_values, [33.333, 66.667])
    overlap_values = np.asarray(
        [float(case["retrieval"]["1"]["oracle_diagnostics"][0]["overlap_050"]) for case in cases]
    )
    overlap_low, overlap_high = np.percentile(overlap_values, [33.333, 66.667])
    groups["texture"] = defaultdict(list)
    groups["overlap"] = defaultdict(list)
    for case in cases:
        groups["source"][str(case["record"].get("source", "unknown"))].append(case)
        groups["angle_bucket"][str(case["record"].get("angle_bucket", "unknown"))].append(case)
        score = float(case["texture_score"])
        label = "low" if score <= texture_low else "medium" if score <= texture_high else "high"
        groups["texture"][label].append(case)
        overlap = float(case["retrieval"]["1"]["oracle_diagnostics"][0]["overlap_050"])
        overlap_label = "low" if overlap <= overlap_low else "medium" if overlap <= overlap_high else "high"
        groups["overlap"][overlap_label].append(case)
    return {
        group_name: {
            label: {method: summarize(rows, method) for method in methods}
            for label, rows in sorted(items.items())
        }
        for group_name, items in groups.items()
    }


def valid_mean(summary: dict, metric: str) -> float:
    value = summary[metric]["mean"]
    return float(value) if value is not None else float("inf")


def score(summary: dict) -> float:
    return valid_mean(summary, "rotation_deg") + 5.0 * valid_mean(summary, "translation_m")


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# V13 Stage-2 Frozen World-Memory Probe",
        "",
        f"Cases: {report['case_count']}",
        "",
        "| Method | T mean | T P90 | R mean | R P90 | Phys@20cm | Fit fail | Success | Catastrophic |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    table_methods = [
        "fixed_explicit",
        "oracle_keyframe_oracle_corr_1f",
        "oracle_keyframe_oracle_corr_3f",
        report["decision"]["best_oracle_keyframe_frozen_method"],
        report["decision"]["best_auto_oracle_correspondence_method"],
        report["decision"]["best_auto_frozen_method"],
    ]
    for method in dict.fromkeys(table_methods):
        row = report["overall"][method]
        physical = row["physical_accuracy_020"]["mean"]
        lines.append(
            f"| {method} | {row['translation_m']['mean']:.4f} | {row['translation_m']['p90']:.4f} | "
            f"{row['rotation_deg']['mean']:.3f} | {row['rotation_deg']['p90']:.3f} | "
            f"{physical if physical is not None else float('nan'):.3f} | "
            f"{100.0 * row['fit_failure_rate']:.1f}% | {100.0 * row['rates']['success']:.1f}% | "
            f"{100.0 * row['rates']['catastrophic']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Best keyframe descriptor: `{report['decision']['best_retrieval_descriptor']}`",
            f"- Best Oracle-keyframe frozen matcher: `{report['decision']['best_oracle_keyframe_frozen_method']}`",
            f"- Best automatic retrieval + Oracle correspondence: `{report['decision']['best_auto_oracle_correspondence_method']}`",
            f"- Best automatic frozen pipeline: `{report['decision']['best_auto_frozen_method']}`",
            f"- Retrieval bottleneck: {report['decision']['retrieval_is_primary_bottleneck']}",
            f"- Correspondence bottleneck: {report['decision']['correspondence_is_primary_bottleneck']}",
            f"- Frozen matcher ready: {report['decision']['frozen_matcher_ready']}",
            f"- Recommended: `{report['decision']['recommended_next_step']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.input_dir / "merged")
    output_dir.mkdir(parents=True, exist_ok=True)
    shards = sorted(args.input_dir.glob("stage2_shard_*_of_*.json"))
    if not shards:
        raise FileNotFoundError(args.input_dir)
    cases = []
    for shard in shards:
        cases.extend(json.loads(shard.read_text(encoding="utf-8"))["cases"])
    names = [case["case_name"] for case in cases]
    if len(names) != len(set(names)):
        raise RuntimeError("Duplicate V13 Stage-2 cases")
    variants = sorted({name for case in cases for name in case["variants"]})
    methods = ["hard_reset_no_alignment", "fixed_explicit", "boundary_oracle", *variants]
    overall = {method: summarize(cases, method) for method in methods}
    retrieval = retrieval_summary(cases)
    retrieval_candidates = sorted(retrieval, key=lambda key: retrieval[key]["recall_at_3"], reverse=True)
    oracle_frozen = [name for name in variants if name.startswith("oracle_keyframe_") and "_frozen_" in name]
    auto_oracle = [name for name in variants if name.startswith("auto_") and "_oracle_corr_" in name]
    auto_frozen = [name for name in variants if name.startswith("auto_") and "_frozen_" in name]
    best_oracle_frozen = min(oracle_frozen, key=lambda name: score(overall[name]))
    best_auto_oracle = min(auto_oracle, key=lambda name: score(overall[name]))
    best_auto_frozen = min(auto_frozen, key=lambda name: score(overall[name]))
    fixed = overall["fixed_explicit"]
    auto_oracle_good = (
        valid_mean(overall[best_auto_oracle], "translation_m") < valid_mean(fixed, "translation_m")
        and valid_mean(overall[best_auto_oracle], "rotation_deg") < valid_mean(fixed, "rotation_deg")
    )
    oracle_frozen_good = (
        valid_mean(overall[best_oracle_frozen], "translation_m") < valid_mean(fixed, "translation_m")
        and valid_mean(overall[best_oracle_frozen], "rotation_deg") < valid_mean(fixed, "rotation_deg")
    )
    auto_frozen_good = (
        valid_mean(overall[best_auto_frozen], "translation_m") < valid_mean(fixed, "translation_m")
        and valid_mean(overall[best_auto_frozen], "rotation_deg") < valid_mean(fixed, "rotation_deg")
    )
    retrieval_good = retrieval[retrieval_candidates[0]]["recall_at_3"] >= 0.70
    decision = {
        "best_retrieval_descriptor": retrieval_candidates[0],
        "best_oracle_keyframe_frozen_method": best_oracle_frozen,
        "best_auto_oracle_correspondence_method": best_auto_oracle,
        "best_auto_frozen_method": best_auto_frozen,
        "retrieval_is_primary_bottleneck": bool(not retrieval_good and oracle_frozen_good),
        "correspondence_is_primary_bottleneck": bool(auto_oracle_good and not oracle_frozen_good),
        "frozen_matcher_ready": bool(auto_frozen_good),
        "recommended_next_step": (
            "build_frozen_streaming_world_memory_pipeline"
            if auto_frozen_good
            else (
                "train_world_anchor_descriptor_projector_not_se3_regressor"
                if auto_oracle_good and not oracle_frozen_good
                else (
                    "improve_global_keyframe_retrieval"
                    if not retrieval_good and oracle_frozen_good
                    else "improve_world_anchor_coordinates_and_correspondence"
                )
            )
        ),
    }
    report = {
        "experiment": "V13 Stage-2 Frozen World-Memory Probe",
        "case_count": len(cases),
        "overall": overall,
        "retrieval": retrieval,
        "decision": decision,
    }
    selected_methods = [
        "fixed_explicit",
        "oracle_keyframe_oracle_corr_1f",
        "oracle_keyframe_oracle_corr_3f",
        best_oracle_frozen,
        best_auto_oracle,
        best_auto_frozen,
    ]
    report["groups"] = grouped(cases, selected_methods)
    (output_dir / "stage2_merged.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8"
    )
    write_markdown(output_dir / "stage2_summary.md", report)
    print(json.dumps(decision, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
