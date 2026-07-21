#!/usr/bin/env python3
"""Merge and analyze V11 stage-2 gauge-neutral first-write oracle shards."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "output" / "v11_gauge_neutral_first_write" / "stage2_full"
DEFAULT_OUTPUT = DEFAULT_INPUT / "merged"
VARIANTS = (
    "reset_gt_boundary",
    "absolute_teacher_state_gt_boundary",
    "gauge_neutral_oracle_gt_boundary",
    "reset_fixed_explicit",
    "gauge_neutral_oracle_fixed_explicit",
    "boundary_output_only_gt_boundary",
)
METRICS = (
    "camera_relative_translation_m",
    "camera_relative_rotation_deg",
    "camera_frame_pointmap_m",
    "camera_frame_depth_m",
    "depth_consistency_m",
    "root_centered_human_m",
    "human_relative_root_m",
    "torso_relative_orientation_deg",
    "human_local_pose_deg",
    "world_camera_translation_m",
    "world_camera_rotation_deg",
    "world_pointmap_teacher_m",
    "world_human_root_m",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def finite(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def stats(values: list[float]) -> dict:
    array = finite(values)
    if not len(array):
        return {key: None for key in ("count", "mean", "median", "p90", "p95", "max")}
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def camera_rates(cases: list[dict], variant: str, relative: bool) -> dict:
    if relative:
        translation_key = "camera_relative_translation_m"
        rotation_key = "camera_relative_rotation_deg"
    else:
        translation_key = "world_camera_translation_m"
        rotation_key = "world_camera_rotation_deg"
    pairs = []
    for case in cases:
        row = case["variants"][variant]["mean_future"]
        translation = float(row[translation_key])
        rotation = float(row[rotation_key])
        if math.isfinite(translation) and math.isfinite(rotation):
            pairs.append((translation, rotation))
    return {
        "count": len(pairs),
        "strict_success_rate": float(np.mean([t < 0.10 and r < 5.0 for t, r in pairs])) if pairs else None,
        "relaxed_success_rate": float(np.mean([t < 0.25 and r < 10.0 for t, r in pairs])) if pairs else None,
        "catastrophic_rate": float(np.mean([t > 1.0 or r > 30.0 for t, r in pairs])) if pairs else None,
    }


def summarize_variant(cases: list[dict], variant: str) -> dict:
    result = {
        "case_count": len(cases),
        "mean_future": {
            metric: stats([
                float(case["variants"][variant]["mean_future"].get(metric, float("nan")))
                for case in cases
            ])
            for metric in METRICS
        },
        "relative_camera_rates": camera_rates(cases, variant, True),
        "world_camera_rates": camera_rates(cases, variant, False),
        "offsets": {},
    }
    for offset in (0, 1, 2, 4, 8):
        rows = []
        for case in cases:
            match = next(
                (
                    row
                    for row in case["variants"][variant]["per_frame"]
                    if int(row["offset"]) == offset
                ),
                None,
            )
            if match is not None:
                rows.append(match)
        result["offsets"][str(offset)] = {
            "case_count": len(rows),
            "metrics": {
                metric: stats([float(row.get(metric, float("nan"))) for row in rows])
                for metric in METRICS
            },
        }
    return result


def paired_comparison(cases: list[dict], baseline: str, method: str) -> dict:
    output = {}
    for metric in METRICS:
        before = np.asarray(
            [float(case["variants"][baseline]["mean_future"].get(metric, np.nan)) for case in cases],
            dtype=np.float64,
        )
        after = np.asarray(
            [float(case["variants"][method]["mean_future"].get(metric, np.nan)) for case in cases],
            dtype=np.float64,
        )
        valid = np.isfinite(before) & np.isfinite(after)
        delta = after[valid] - before[valid]
        relative = (before[valid] - after[valid]) / np.maximum(np.abs(before[valid]), 1e-8)
        output[metric] = {
            "count": int(valid.sum()),
            "mean_before": float(before[valid].mean()) if valid.any() else None,
            "mean_after": float(after[valid].mean()) if valid.any() else None,
            "mean_delta_after_minus_before": float(delta.mean()) if len(delta) else None,
            "median_relative_improvement": float(np.median(relative)) if len(relative) else None,
            "improved_case_rate": float(np.mean(delta < -1e-8)) if len(delta) else None,
            "unchanged_case_rate": float(np.mean(np.abs(delta) <= 1e-8)) if len(delta) else None,
        }
    return output


def tertiles(cases: list[dict], key: str) -> dict[str, str]:
    values = np.asarray(
        [float(case["variants"]["reset_gt_boundary"].get(key, 0.0)) for case in cases],
        dtype=np.float64,
    )
    low, high = np.percentile(values, [33.333, 66.667])
    labels = {}
    for case, value in zip(cases, values):
        labels[case["case_name"]] = "low" if value <= low else "medium" if value <= high else "high"
    return labels


def grouped(cases: list[dict]) -> dict:
    texture = tertiles(cases, "texture_score")
    speed = tertiles(cases, "human_speed_m_per_frame")
    collections: dict[str, dict[str, list[dict]]] = {
        "source": defaultdict(list),
        "angle_bucket": defaultdict(list),
        "texture_tertile": defaultdict(list),
        "human_speed_tertile": defaultdict(list),
        "human_count": defaultdict(list),
    }
    for case in cases:
        reset = case["variants"]["reset_gt_boundary"]
        collections["source"][str(case["record"]["source"])].append(case)
        collections["angle_bucket"][str(case["record"].get("angle_bucket", "unknown"))].append(case)
        collections["texture_tertile"][texture[case["case_name"]]].append(case)
        collections["human_speed_tertile"][speed[case["case_name"]]].append(case)
        collections["human_count"][str(reset.get("human_count", "unknown"))].append(case)
    output = {}
    for group_name, labels in collections.items():
        output[group_name] = {}
        for label, items in sorted(labels.items()):
            output[group_name][label] = {
                "case_count": len(items),
                "reset_gt_boundary": summarize_variant(items, "reset_gt_boundary"),
                "gauge_neutral_oracle_gt_boundary": summarize_variant(items, "gauge_neutral_oracle_gt_boundary"),
                "paired": paired_comparison(items, "reset_gt_boundary", "gauge_neutral_oracle_gt_boundary"),
            }
    return output


def audit(cases: list[dict]) -> dict:
    boundary = [float(case["boundary_lock"]["gauge_neutral_vs_reset"]["maximum"]) for case in cases]
    camera_invariance = [
        float(case["gauge_neutrality"]["camera_relative_loss_max_abs_change"]) for case in cases
    ]
    point_invariance = [
        float(case["gauge_neutrality"]["camera_frame_pointmap_max_abs_change_m"]) for case in cases
    ]
    peak_memory = [float(case["optimization"]["peak_gpu_memory_bytes"]) for case in cases]
    elapsed = [float(case["optimization"]["elapsed_seconds"]) for case in cases]
    return {
        "boundary_output_max_abs": stats(boundary),
        "all_boundaries_exactly_locked": bool(max(boundary, default=0.0) == 0.0),
        "global_se3_camera_loss_change": stats(camera_invariance),
        "global_se3_pointmap_change_m": stats(point_invariance),
        "optimization_seconds": stats(elapsed),
        "peak_gpu_memory_bytes": stats(peak_memory),
    }


def route_decision(report: dict) -> dict:
    local = report["paired"]["gauge_neutral_vs_reset_gt"]
    explicit = report["paired"]["gauge_neutral_plus_explicit_vs_explicit"]
    required_local = (
        "camera_relative_translation_m",
        "camera_relative_rotation_deg",
        "camera_frame_pointmap_m",
        "human_relative_root_m",
    )
    local_improved = {
        metric: bool(local[metric]["mean_after"] < 0.90 * local[metric]["mean_before"])
        for metric in required_local
    }
    explicit_improved = {
        metric: bool(explicit[metric]["mean_after"] < explicit[metric]["mean_before"])
        for metric in ("world_camera_translation_m", "world_camera_rotation_deg", "world_pointmap_teacher_m")
    }
    local_oracle_supports_training = bool(
        report["audit"]["all_boundaries_exactly_locked"]
        and all(local_improved.values())
    )
    world_gain = {
        metric: float(
            (explicit[metric]["mean_before"] - explicit[metric]["mean_after"])
            / max(abs(explicit[metric]["mean_before"]), 1e-8)
        )
        for metric in ("world_camera_translation_m", "world_camera_rotation_deg", "world_pointmap_teacher_m")
    }
    practical_world_complementarity = bool(sum(value >= 0.05 for value in world_gain.values()) >= 2)
    return {
        "local_metrics_improve_at_least_10_percent": local_improved,
        "explicit_combination_improves": explicit_improved,
        "explicit_world_relative_gain": world_gain,
        "local_first_write_complementarity_proven": local_oracle_supports_training,
        "practical_world_complementarity_at_least_5_percent": practical_world_complementarity,
        "continue_to_train_gauge_neutral_state_query_prompt": local_oracle_supports_training,
        "explicit_relocalization_remains_primary_global_bottleneck": bool(
            local_oracle_supports_training and not practical_world_complementarity
        ),
        "recommended_route": (
            "train_gauge_neutral_first_write_prompt_as_local_transition_module_and_improve_explicit_relocalization"
            if local_oracle_supports_training
            else "stop_latent_first_write_and_focus_on_explicit_world_memory_and_reliability"
        ),
    }


def write_cases(path: Path, cases: list[dict]) -> None:
    rows = []
    for case in cases:
        row = {
            "case_name": case["case_name"],
            "source": case["record"]["source"],
            "angle_bucket": case["record"].get("angle_bucket", "unknown"),
            "view_angle_deg": case["record"].get("view_angle_deg"),
            "boundary_max_abs": case["boundary_lock"]["gauge_neutral_vs_reset"]["maximum"],
            "optimization_seconds": case["optimization"]["elapsed_seconds"],
        }
        for variant in VARIANTS:
            for metric in METRICS:
                row[f"{variant}__{metric}"] = case["variants"][variant]["mean_future"].get(metric)
        rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def f(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# V11 Gauge-Neutral First-Write Complementarity Probe",
        "",
        f"Cases: {report['case_count']}",
        "",
        "## Main comparison",
        "",
        "| Variant | Rel camera t (m) | Rel camera R (deg) | Cam pointmap (m) | Human rel root (m) | World camera t (m) | World camera R (deg) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        row = report["overall"][variant]["mean_future"]
        lines.append(
            f"| {variant} | {f(row['camera_relative_translation_m']['mean'])} | "
            f"{f(row['camera_relative_rotation_deg']['mean'])} | "
            f"{f(row['camera_frame_pointmap_m']['mean'])} | "
            f"{f(row['human_relative_root_m']['mean'])} | "
            f"{f(row['world_camera_translation_m']['mean'])} | "
            f"{f(row['world_camera_rotation_deg']['mean'])} |"
        )
    paired = report["paired"]["gauge_neutral_vs_reset_gt"]
    lines.extend(
        [
            "",
            "## Gauge-neutral oracle gain over hard reset",
            "",
            "| Metric | Before | After | Improved cases | Median relative gain |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for metric in (
        "camera_relative_translation_m",
        "camera_relative_rotation_deg",
        "camera_frame_pointmap_m",
        "human_relative_root_m",
        "torso_relative_orientation_deg",
    ):
        row = paired[metric]
        lines.append(
            f"| {metric} | {f(row['mean_before'])} | {f(row['mean_after'])} | "
            f"{100.0 * row['improved_case_rate']:.1f}% | {100.0 * row['median_relative_improvement']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Per-source camera and geometry",
            "",
            "| Source | Cases | Reset t | Oracle t | Reset R | Oracle R | Reset pointmap | Oracle pointmap |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for source, group in report["groups"]["source"].items():
        reset_group = group["reset_gt_boundary"]["mean_future"]
        oracle_group = group["gauge_neutral_oracle_gt_boundary"]["mean_future"]
        lines.append(
            f"| {source} | {group['case_count']} | "
            f"{f(reset_group['camera_relative_translation_m']['mean'])} | "
            f"{f(oracle_group['camera_relative_translation_m']['mean'])} | "
            f"{f(reset_group['camera_relative_rotation_deg']['mean'])} | "
            f"{f(oracle_group['camera_relative_rotation_deg']['mean'])} | "
            f"{f(reset_group['camera_frame_pointmap_m']['mean'])} | "
            f"{f(oracle_group['camera_frame_pointmap_m']['mean'])} |"
        )
    for group_name, title in (
        ("angle_bucket", "View-angle bucket"),
        ("texture_tertile", "Texture tertile"),
        ("human_speed_tertile", "Human-speed tertile"),
    ):
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Group | Cases | Reset t | Oracle t | Reset R | Oracle R | Reset pointmap | Oracle pointmap |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for label, group in report["groups"][group_name].items():
            reset_group = group["reset_gt_boundary"]["mean_future"]
            oracle_group = group["gauge_neutral_oracle_gt_boundary"]["mean_future"]
            lines.append(
                f"| {label} | {group['case_count']} | "
                f"{f(reset_group['camera_relative_translation_m']['mean'])} | "
                f"{f(oracle_group['camera_relative_translation_m']['mean'])} | "
                f"{f(reset_group['camera_relative_rotation_deg']['mean'])} | "
                f"{f(oracle_group['camera_relative_rotation_deg']['mean'])} | "
                f"{f(reset_group['camera_frame_pointmap_m']['mean'])} | "
                f"{f(oracle_group['camera_frame_pointmap_m']['mean'])} |"
            )
    lines.extend(
        [
            "",
            "## Tail metrics",
            "",
            "| Metric | Reset P90 | Oracle P90 | Reset P95 | Oracle P95 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    reset_tail = report["overall"]["reset_gt_boundary"]["mean_future"]
    oracle_tail = report["overall"]["gauge_neutral_oracle_gt_boundary"]["mean_future"]
    for metric in (
        "camera_relative_translation_m",
        "camera_relative_rotation_deg",
        "camera_frame_pointmap_m",
        "human_relative_root_m",
    ):
        lines.append(
            f"| {metric} | {f(reset_tail[metric]['p90'])} | {f(oracle_tail[metric]['p90'])} | "
            f"{f(reset_tail[metric]['p95'])} | {f(oracle_tail[metric]['p95'])} |"
        )
    reset_rates = report["overall"]["reset_gt_boundary"]["relative_camera_rates"]
    oracle_rates = report["overall"]["gauge_neutral_oracle_gt_boundary"]["relative_camera_rates"]
    lines.extend(
        [
            "",
            f"- Relative-camera strict success: {100.0 * reset_rates['strict_success_rate']:.1f}% -> {100.0 * oracle_rates['strict_success_rate']:.1f}%",
            f"- Relative-camera relaxed success: {100.0 * reset_rates['relaxed_success_rate']:.1f}% -> {100.0 * oracle_rates['relaxed_success_rate']:.1f}%",
            f"- Relative-camera catastrophic rate: {100.0 * reset_rates['catastrophic_rate']:.1f}% -> {100.0 * oracle_rates['catastrophic_rate']:.1f}%",
        ]
    )
    lines.extend(
        [
            "",
            "## Offset curve",
            "",
            "| Offset | Reset t (m) | Oracle t (m) | Reset R (deg) | Oracle R (deg) | Reset pointmap (m) | Oracle pointmap (m) |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    reset = report["overall"]["reset_gt_boundary"]["offsets"]
    oracle = report["overall"]["gauge_neutral_oracle_gt_boundary"]["offsets"]
    for offset in (0, 1, 2, 4, 8):
        r = reset[str(offset)]["metrics"]
        o = oracle[str(offset)]["metrics"]
        lines.append(
            f"| {offset} | {f(r['camera_relative_translation_m']['mean'])} | "
            f"{f(o['camera_relative_translation_m']['mean'])} | "
            f"{f(r['camera_relative_rotation_deg']['mean'])} | "
            f"{f(o['camera_relative_rotation_deg']['mean'])} | "
            f"{f(r['camera_frame_pointmap_m']['mean'])} | "
            f"{f(o['camera_frame_pointmap_m']['mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Sanity checks",
            "",
            f"- Boundary outputs exactly locked: {report['audit']['all_boundaries_exactly_locked']}",
            f"- Max global-SE(3) camera-loss change: {f(report['audit']['global_se3_camera_loss_change']['max'], 8)}",
            f"- Max camera-frame pointmap change after global SE(3): {f(report['audit']['global_se3_pointmap_change_m']['max'], 8)} m",
            f"- Mean per-case oracle optimization: {f(report['audit']['optimization_seconds']['mean'], 2)} s",
            "- The current records contain no multi-person examples and no background-overlap labels.",
            "",
            "## Route decision",
            "",
            f"- Continue training Gauge-Neutral State-query Prompt: {report['decision']['continue_to_train_gauge_neutral_state_query_prompt']}",
            f"- Local first-write complementarity proven: {report['decision']['local_first_write_complementarity_proven']}",
            f"- Practical world-space complementarity >=5%: {report['decision']['practical_world_complementarity_at_least_5_percent']}",
            f"- Explicit relocalization remains primary global bottleneck: {report['decision']['explicit_relocalization_remains_primary_global_bottleneck']}",
            f"- Recommended route: `{report['decision']['recommended_route']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    shards = sorted(args.input_dir.glob("stage2_shard_*_of_*.json"))
    if not shards:
        raise FileNotFoundError(args.input_dir)
    cases = []
    for shard in shards:
        cases.extend(json.loads(shard.read_text(encoding="utf-8"))["cases"])
    names = [case["case_name"] for case in cases]
    if len(names) != len(set(names)):
        raise RuntimeError("Duplicate V11 stage-2 cases")
    missing = [variant for variant in VARIANTS if any(variant not in case["variants"] for case in cases)]
    if missing:
        raise RuntimeError(f"Missing variants: {missing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": "V11 Gauge-Neutral First-Write Complementarity Probe",
        "case_count": len(cases),
        "shards": [str(path) for path in shards],
        "overall": {variant: summarize_variant(cases, variant) for variant in VARIANTS},
        "paired": {
            "gauge_neutral_vs_reset_gt": paired_comparison(
                cases, "reset_gt_boundary", "gauge_neutral_oracle_gt_boundary"
            ),
            "gauge_neutral_plus_explicit_vs_explicit": paired_comparison(
                cases, "reset_fixed_explicit", "gauge_neutral_oracle_fixed_explicit"
            ),
            "absolute_teacher_vs_reset_gt": paired_comparison(
                cases, "reset_gt_boundary", "absolute_teacher_state_gt_boundary"
            ),
        },
        "groups": grouped(cases),
        "audit": audit(cases),
        "limitations": {
            "scene_depth_gt": False,
            "pointmap_target": "same-camera warmed Human3R teacher",
            "multi_person_cases": 0,
            "background_overlap_labels": False,
        },
    }
    report["decision"] = route_decision(report)
    (args.output_dir / "stage2_merged.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_cases(args.output_dir / "stage2_cases.csv", cases)
    write_markdown(args.output_dir / "stage2_summary.md", report)
    print(f">> merged {len(cases)} cases from {len(shards)} shards")
    print(f">> wrote {args.output_dir / 'stage2_summary.md'}")


if __name__ == "__main__":
    main()
