#!/usr/bin/env python3
"""Merge V12 evaluation shards and decide whether the learned route survives LOSO."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "output" / "v12_gated_first_write" / "eval_loso_mvhuman200"
VARIANTS = (
    "hard_reset",
    "boundary_output_only",
    "oracle",
    "ungated",
    "gate_only",
    "gated",
    "no_old",
    "zero_old",
    "shuffle_old",
    "wrong_old",
    "explicit_only",
    "gated_explicit",
    "oracle_explicit",
)
METRICS = (
    "camera_relative_translation_m",
    "camera_relative_rotation_deg",
    "camera_frame_pointmap_m",
    "camera_frame_depth_m",
    "depth_consistency_m",
    "local_scale_log_abs",
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
    parser.add_argument("--output_dir", type=Path, default=None)
    return parser.parse_args()


def values(cases: list[dict], variant: str, metric: str) -> np.ndarray:
    array = np.asarray(
        [float(case["variants"][variant]["mean_future"].get(metric, np.nan)) for case in cases],
        dtype=np.float64,
    )
    return array[np.isfinite(array)]


def stats(array: np.ndarray) -> dict:
    array = np.asarray(array, dtype=np.float64)
    array = array[np.isfinite(array)]
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


def rates(cases: list[dict], variant: str) -> dict:
    pairs = []
    for case in cases:
        row = case["variants"][variant]["mean_future"]
        translation = float(row.get("camera_relative_translation_m", np.nan))
        rotation = float(row.get("camera_relative_rotation_deg", np.nan))
        if math.isfinite(translation) and math.isfinite(rotation):
            pairs.append((translation, rotation))
    if not pairs:
        return {"count": 0, "strict_success": None, "relaxed_success": None, "catastrophic": None}
    translation = np.asarray([row[0] for row in pairs], dtype=np.float64)
    rotation = np.asarray([row[1] for row in pairs], dtype=np.float64)
    return {
        "count": len(pairs),
        "strict_success": float(np.mean((translation < 0.10) & (rotation < 5.0))),
        "relaxed_success": float(np.mean((translation < 0.25) & (rotation < 10.0))),
        "catastrophic": float(np.mean((translation > 1.0) | (rotation > 30.0))),
    }


def summarize(cases: list[dict], variant: str) -> dict:
    result = {
        "case_count": len(cases),
        "metrics": {metric: stats(values(cases, variant, metric)) for metric in METRICS},
        "rates": rates(cases, variant),
        "offsets": {},
    }
    for offset in (0, 1, 2, 4, 8):
        rows = []
        for case in cases:
            row = next(
                (item for item in case["variants"][variant]["per_frame"] if int(item["offset"]) == offset),
                None,
            )
            if row is not None:
                rows.append(row)
        result["offsets"][str(offset)] = {
            "case_count": len(rows),
            "metrics": {
                metric: stats(np.asarray([float(row.get(metric, np.nan)) for row in rows], dtype=np.float64))
                for metric in METRICS
            },
        }
    return result


def paired(cases: list[dict], baseline: str, method: str) -> dict:
    output = {}
    for metric in METRICS:
        before = np.asarray([float(case["variants"][baseline]["mean_future"].get(metric, np.nan)) for case in cases])
        after = np.asarray([float(case["variants"][method]["mean_future"].get(metric, np.nan)) for case in cases])
        valid = np.isfinite(before) & np.isfinite(after)
        delta = after[valid] - before[valid]
        if not valid.any():
            output[metric] = {
                "count": 0,
                "mean_before": None,
                "mean_after": None,
                "mean_delta": None,
                "improved_rate": None,
            }
            continue
        output[metric] = {
            "count": int(valid.sum()),
            "mean_before": float(before[valid].mean()),
            "mean_after": float(after[valid].mean()),
            "mean_delta": float(delta.mean()),
            "improved_rate": float(np.mean(delta < 0.0)),
        }
    return output


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    positive = labels.astype(bool)
    negative = ~positive
    if not positive.any() or not negative.any():
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[positive].sum() - positive.sum() * (positive.sum() + 1) / 2) / (positive.sum() * negative.sum()))


def calibration(scores: np.ndarray, targets: np.ndarray, bins: int = 10) -> float:
    error = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        mask = (scores >= low) & (scores < high if index + 1 < bins else scores <= high)
        if mask.any():
            error += mask.mean() * abs(scores[mask].mean() - targets[mask].mean())
    return float(error)


def gate_metrics(cases: list[dict]) -> dict:
    gate = np.asarray([case["gate_predictions"]["gated"] for case in cases], dtype=np.float64)
    gain = np.asarray([case["gate_predictions"]["predicted_gain"] for case in cases], dtype=np.float64)
    target = np.asarray([case["labels"]["gate_target"] for case in cases], dtype=np.float64)
    gain_target = np.asarray([case["labels"]["gain_target"] for case in cases], dtype=np.float64)
    helpful = np.asarray([case["labels"]["oracle_helpful"] for case in cases], dtype=bool)
    wait = np.asarray([case["gate_predictions"]["wait_score"] for case in cases], dtype=np.float64)
    wait_target = np.asarray([case["labels"]["wait_target"] for case in cases], dtype=bool)
    simple = target <= 0.05
    hard_rotation = np.asarray(
        [case["variants"]["hard_reset"]["mean_future"]["camera_relative_rotation_deg"] for case in cases],
        dtype=np.float64,
    )
    gated_rotation = np.asarray(
        [case["variants"]["gated"]["mean_future"]["camera_relative_rotation_deg"] for case in cases],
        dtype=np.float64,
    )
    simple_degraded = (gated_rotation[simple] > hard_rotation[simple] + 0.25) if simple.any() else np.asarray([])
    return {
        "mean_gate": float(gate.mean()),
        "identity_fallback_rate_gate_lt_0_1": float(np.mean(gate < 0.10)),
        "correction_rate_gate_gt_0_5": float(np.mean(gate > 0.50)),
        "difficulty_auroc": auroc(gate, helpful),
        "gate_target_correlation": float(np.corrcoef(gate, target)[0, 1]),
        "predicted_gain_correlation": float(np.corrcoef(gain, gain_target)[0, 1]),
        "wait_decision_auroc": auroc(wait, wait_target),
        "calibration_error": calibration(gate, target),
        "false_positive_correction_rate": float(np.mean(gate[simple] > 0.50)) if simple.any() else None,
        "simple_sample_rotation_degradation_rate": float(np.mean(simple_degraded)) if len(simple_degraded) else None,
        "helpful_target_rate": float(helpful.mean()),
    }


def tertile_labels(cases: list[dict], key: str, names: tuple[str, str, str]) -> dict[str, str]:
    case_values = {
        case["case_name"]: float(case["variants"]["hard_reset"].get(key, np.nan))
        for case in cases
    }
    finite_values = np.asarray([value for value in case_values.values() if math.isfinite(value)], dtype=np.float64)
    if not len(finite_values):
        return {case["case_name"]: "unknown" for case in cases}
    low, high = np.percentile(finite_values, [33.333, 66.667])
    labels = {}
    for name, value in case_values.items():
        if not math.isfinite(value):
            labels[name] = "unknown"
        else:
            labels[name] = names[0] if value <= low else names[1] if value <= high else names[2]
    return labels


def grouped_summary(cases: list[dict]) -> dict:
    texture = tertile_labels(cases, "texture_score", ("low", "medium", "high"))
    speed = tertile_labels(cases, "human_speed_m_per_frame", ("low", "medium", "high"))
    groups: dict[str, dict[str, list[dict]]] = {
        "source": defaultdict(list),
        "angle_bucket": defaultdict(list),
        "texture_tertile": defaultdict(list),
        "human_speed_tertile": defaultdict(list),
        "human_count": defaultdict(list),
        "oracle_helpful": defaultdict(list),
    }
    for case in cases:
        name = case["case_name"]
        record = case["record"]
        hard = case["variants"]["hard_reset"]
        groups["source"][str(record.get("source", "unknown"))].append(case)
        groups["angle_bucket"][str(record.get("angle_bucket", "unknown"))].append(case)
        groups["texture_tertile"][texture[name]].append(case)
        groups["human_speed_tertile"][speed[name]].append(case)
        groups["human_count"][str(hard.get("human_count", "unknown"))].append(case)
        groups["oracle_helpful"][str(bool(case["labels"]["oracle_helpful"]))].append(case)
    selected_variants = ("hard_reset", "oracle", "ungated", "gated", "no_old", "explicit_only", "gated_explicit")
    return {
        group_name: {
            label: {variant: summarize(items, variant) for variant in selected_variants}
            for label, items in sorted(group.items())
        }
        for group_name, group in groups.items()
    }


def runtime_summary(cases: list[dict]) -> dict:
    timing_keys = sorted({key for case in cases for key in case.get("timing_seconds", {})})
    memory_keys = sorted({key for case in cases for key in case.get("peak_memory_mb", {})})
    timing = {
        key: stats(np.asarray([float(case["timing_seconds"].get(key, np.nan)) for case in cases]))
        for key in timing_keys
    }
    memory = {
        key: stats(np.asarray([float(case["peak_memory_mb"].get(key, np.nan)) for case in cases]))
        for key in memory_keys
    }
    reset_mean = timing.get("reset", {}).get("mean")
    gated_mean = timing.get("gated", {}).get("mean")
    return {
        "sequence_seconds": timing,
        "peak_memory_mb": memory,
        "gated_extra_sequence_seconds": (
            float(gated_mean - reset_mean)
            if gated_mean is not None and reset_mean is not None
            else None
        ),
    }


def boundary_audit(cases: list[dict]) -> dict:
    result = {}
    for variant in ("oracle", "gated", "ungated", "no_old"):
        maxima = np.asarray([case["boundary_lock"][variant]["maximum"] for case in cases], dtype=np.float64)
        result[variant] = {
            "max": float(maxima.max()) if len(maxima) else None,
            "mean": float(maxima.mean()) if len(maxima) else None,
            "exact": bool(len(maxima) and maxima.max() == 0.0),
        }
    return result


def oracle_retention(report: dict, metric: str) -> float:
    hard = report["overall"]["hard_reset"]["metrics"][metric]["mean"]
    oracle = report["overall"]["oracle"]["metrics"][metric]["mean"]
    learned = report["overall"]["gated"]["metrics"][metric]["mean"]
    oracle_gain = hard - oracle
    if oracle_gain <= 1e-8:
        return float("nan")
    return float((hard - learned) / oracle_gain)


def decision(report: dict) -> dict:
    retention_t = oracle_retention(report, "camera_relative_translation_m")
    retention_r = oracle_retention(report, "camera_relative_rotation_deg")
    hard = report["overall"]["hard_reset"]
    gated = report["overall"]["gated"]
    ungated = report["overall"]["ungated"]
    world_hard = report["overall"]["explicit_only"]
    world_gated = report["overall"]["gated_explicit"]
    alternatives = ("no_old", "zero_old", "shuffle_old", "wrong_old")
    correct_old_advantages = {}
    for variant in alternatives:
        delta_t = (
            report["overall"][variant]["metrics"]["camera_relative_translation_m"]["mean"]
            - gated["metrics"]["camera_relative_translation_m"]["mean"]
        )
        delta_r = (
            report["overall"][variant]["metrics"]["camera_relative_rotation_deg"]["mean"]
            - gated["metrics"]["camera_relative_rotation_deg"]["mean"]
        )
        correct_old_advantages[variant] = {"translation_m": float(delta_t), "rotation_deg": float(delta_r)}
    criteria = {
        "oracle_gain_retention_at_least_30_percent": bool(min(retention_t, retention_r) >= 0.30),
        "translation_below_0_060_m": bool(gated["metrics"]["camera_relative_translation_m"]["mean"] < 0.060),
        "rotation_below_1_1_deg": bool(gated["metrics"]["camera_relative_rotation_deg"]["mean"] < 1.10),
        "strict_success_gain_at_least_7_points": bool(gated["rates"]["strict_success"] - hard["rates"]["strict_success"] >= 0.07),
        "translation_p90_improves": bool(
            gated["metrics"]["camera_relative_translation_m"]["p90"]
            < hard["metrics"]["camera_relative_translation_m"]["p90"]
        ),
        "rotation_p90_improves": bool(
            gated["metrics"]["camera_relative_rotation_deg"]["p90"]
            < hard["metrics"]["camera_relative_rotation_deg"]["p90"]
        ),
        "gated_better_than_ungated_rotation": bool(gated["metrics"]["camera_relative_rotation_deg"]["mean"] < ungated["metrics"]["camera_relative_rotation_deg"]["mean"]),
        "correct_old_state_beats_all_controls": bool(
            all(row["translation_m"] > 0.0 or row["rotation_deg"] > 0.0 for row in correct_old_advantages.values())
        ),
        "all_boundaries_locked": bool(all(row["exact"] for row in report["boundary_audit"].values())),
        "world_rotation_not_worse": bool(world_gated["metrics"]["world_camera_rotation_deg"]["mean"] <= world_hard["metrics"]["world_camera_rotation_deg"]["mean"] + 0.25),
    }
    continue_route = bool(all(criteria.values()))
    return {
        "oracle_retention_translation": retention_t,
        "oracle_retention_rotation": retention_r,
        "correct_old_state_advantage": correct_old_advantages,
        "criteria": criteria,
        "continue_gauge_neutral_recurrent_transition": continue_route,
        "recommended_route": (
            "continue_gated_first_write_prompt_and_expand_real_multicamera_training"
            if continue_route
            else "stop_state_modification_and_keep_state_tokens_for_reliability_only"
        ),
    }


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# V12 Learned Gated Gauge-Neutral First-Write Prompt",
        "",
        f"Test cases: {report['case_count']} unseen MVHuman200 cuts",
        "",
        "| Variant | Rel T | Rel R | Pointmap | Human root | Strict success | World T | World R |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        row = report["overall"][variant]
        lines.append(
            f"| {variant} | {row['metrics']['camera_relative_translation_m']['mean']:.4f} | "
            f"{row['metrics']['camera_relative_rotation_deg']['mean']:.4f} | "
            f"{row['metrics']['camera_frame_pointmap_m']['mean']:.4f} | "
            f"{row['metrics']['human_relative_root_m']['mean']:.4f} | "
            f"{100.0 * row['rates']['strict_success']:.1f}% | "
            f"{row['metrics']['world_camera_translation_m']['mean']:.4f} | "
            f"{row['metrics']['world_camera_rotation_deg']['mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Tail metrics",
            "",
            "| Variant | T median | T P90 | T P95 | R median | R P90 | R P95 | Catastrophic |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for variant in ("hard_reset", "oracle", "ungated", "gated", "no_old"):
        row = report["overall"][variant]
        t = row["metrics"]["camera_relative_translation_m"]
        r = row["metrics"]["camera_relative_rotation_deg"]
        lines.append(
            f"| {variant} | {t['median']:.4f} | {t['p90']:.4f} | {t['p95']:.4f} | "
            f"{r['median']:.4f} | {r['p90']:.4f} | {r['p95']:.4f} | "
            f"{100.0 * row['rates']['catastrophic']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Offset curve",
            "",
            "| Offset | Reset T | Gated T | Oracle T | Reset R | Gated R | Oracle R |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for offset in (1, 2, 4, 8):
        reset = report["overall"]["hard_reset"]["offsets"][str(offset)]["metrics"]
        gated = report["overall"]["gated"]["offsets"][str(offset)]["metrics"]
        oracle = report["overall"]["oracle"]["offsets"][str(offset)]["metrics"]
        lines.append(
            f"| {offset} | {reset['camera_relative_translation_m']['mean']:.4f} | "
            f"{gated['camera_relative_translation_m']['mean']:.4f} | "
            f"{oracle['camera_relative_translation_m']['mean']:.4f} | "
            f"{reset['camera_relative_rotation_deg']['mean']:.4f} | "
            f"{gated['camera_relative_rotation_deg']['mean']:.4f} | "
            f"{oracle['camera_relative_rotation_deg']['mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Oracle retention",
            "",
            f"- Translation gain retained: {100.0 * report['decision']['oracle_retention_translation']:.1f}%",
            f"- Rotation gain retained: {100.0 * report['decision']['oracle_retention_rotation']:.1f}%",
            "",
            "## Gate",
            "",
        ]
    )
    for key, value in report["gate_metrics"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Runtime", ""])
    runtime = report["runtime"]
    lines.append(f"- Gated extra sequence time: {runtime['gated_extra_sequence_seconds']} s")
    if "gated" in runtime["peak_memory_mb"]:
        lines.append(f"- Gated peak allocated memory: {runtime['peak_memory_mb']['gated']['mean']} MB")
    lines.extend(
        [
            "",
            "## Coverage limitations",
            "",
            "- Primary test source is the completely unseen MVHuman200 split.",
            "- Background-overlap labels are unavailable in the current 180-cut manifest.",
            "- The current benchmark contains one detected person per cut, so no multi-person subgroup can be estimated.",
        ]
    )
    lines.extend(["", "## Decision", ""])
    for key, value in report["decision"]["criteria"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            f"- Continue route: {report['decision']['continue_gauge_neutral_recurrent_transition']}",
            f"- Recommended: `{report['decision']['recommended_route']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.input_dir / "merged")
    output_dir.mkdir(parents=True, exist_ok=True)
    shards = sorted(args.input_dir.glob("eval_shard_*_of_*.json"))
    if not shards:
        raise FileNotFoundError(args.input_dir)
    cases = []
    for shard in shards:
        cases.extend(json.loads(shard.read_text(encoding="utf-8"))["cases"])
    names = [case["case_name"] for case in cases]
    if len(names) != len(set(names)):
        raise RuntimeError("Duplicate V12 evaluation cases")
    report = {
        "experiment": "V12 Learned Gated Gauge-Neutral First-Write Prompt",
        "case_count": len(cases),
        "overall": {variant: summarize(cases, variant) for variant in VARIANTS},
        "paired": {
            "gated_vs_reset": paired(cases, "hard_reset", "gated"),
            "ungated_vs_reset": paired(cases, "hard_reset", "ungated"),
            "gated_vs_ungated": paired(cases, "ungated", "gated"),
            "gated_explicit_vs_explicit": paired(cases, "explicit_only", "gated_explicit"),
        },
        "gate_metrics": gate_metrics(cases),
        "boundary_audit": boundary_audit(cases),
        "groups": grouped_summary(cases),
        "runtime": runtime_summary(cases),
        "limitations": {
            "background_overlap_labels": False,
            "multi_person_cases": int(
                sum(int(case["variants"]["hard_reset"].get("human_count", 1)) > 1 for case in cases)
            ),
            "primary_test_source": "mvhuman200",
        },
    }
    report["decision"] = decision(report)
    (output_dir / "v12_eval_merged.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    rows = []
    for case in cases:
        row = {"case_name": case["case_name"], **case["gate_predictions"], **case["labels"]}
        for variant in VARIANTS:
            row[f"{variant}_t"] = case["variants"][variant]["mean_future"]["camera_relative_translation_m"]
            row[f"{variant}_r"] = case["variants"][variant]["mean_future"]["camera_relative_rotation_deg"]
        rows.append(row)
    with (output_dir / "v12_eval_cases.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(output_dir / "v12_eval_summary.md", report)
    print(f">> merged {len(cases)} cases")


if __name__ == "__main__":
    main()
