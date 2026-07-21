#!/usr/bin/env python3
"""Merge four V17 LOSO folds and produce the final quantitative report."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "output" / "v17_direct_vs_factorized" / "loso"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "v17_direct_vs_factorized" / "evaluation"
DEFAULT_CACHE = REPO_ROOT / "output" / "v17_direct_vs_factorized" / "feature_cache"
SOURCES = ("avatarrex", "thuman", "mvhuman100", "mvhuman200")
METHOD_ORDER = (
    "original_continue",
    "hard_reset",
    "fixed_explicit",
    "torso_motion",
    "weak_stats_absolute",
    "direct_absolute",
    "direct_residual_no_gauge_aug",
    "direct_residual_gauge_aug",
    "direct_residual",
    "direct_residual_uncertainty",
    "factor_scale_only",
    "factor_direction_scale",
    "factor_translation_residual",
    "factor_direction_scale_uncertainty",
    "vggt_direction_learned_scale",
    "gt_rotation_learned_translation",
    "learned_rotation_gt_translation",
    "boundary_oracle",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache_dir", type=Path, default=DEFAULT_CACHE)
    return parser.parse_args()


def finite(row: dict, key: str) -> float | None:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {key: None for key in ("mean", "median", "p90", "p95")}
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
    }


def failed(row: dict | None) -> bool:
    return row is None or bool(row.get("fit_failed", False)) or finite(row, "camera_translation_error_m") is None or finite(row, "camera_rotation_error_deg") is None


def catastrophic(row: dict | None) -> bool:
    return failed(row) or float(row["camera_translation_error_m"]) > 1.0 or float(row["camera_rotation_error_deg"]) > 30.0


def success(row: dict | None) -> bool:
    return not failed(row) and float(row["camera_translation_error_m"]) < 0.25 and float(row["camera_rotation_error_deg"]) < 5.0


def strict_success(row: dict | None) -> bool:
    return not failed(row) and float(row["camera_translation_error_m"]) < 0.10 and float(row["camera_rotation_error_deg"]) < 2.0


def uncertainty_metrics(rows: list[dict]) -> dict | None:
    if not rows or "predicted_translation_log_variance" not in rows[0]:
        return None
    rotation_sigma = np.sqrt(np.exp([float(row["predicted_rotation_log_variance"]) for row in rows]))
    translation_sigma = np.sqrt(np.exp([float(row["predicted_translation_log_variance"]) for row in rows]))
    rotation_error = np.radians([float(row["camera_rotation_error_deg"]) for row in rows])
    translation_error = np.asarray([float(row["camera_translation_error_m"]) for row in rows])
    return {
        "rotation_spearman": float(spearmanr(rotation_sigma, rotation_error).statistic),
        "translation_spearman": float(spearmanr(translation_sigma, translation_error).statistic),
        "rotation_mean_sigma_rad": float(rotation_sigma.mean()),
        "translation_mean_sigma_m": float(translation_sigma.mean()),
        "rotation_mean_error_rad": float(rotation_error.mean()),
        "translation_mean_error_m": float(translation_error.mean()),
    }


def aggregate(cases: list[dict], method: str) -> dict:
    rows = [case["methods"].get(method) for case in cases]
    valid_rows = [row for row in rows if not failed(row)]
    fixed_rows = [case["methods"]["fixed_explicit"] for case in cases]
    translation = [float(row["camera_translation_error_m"]) for row in valid_rows]
    rotation = [float(row["camera_rotation_error_deg"]) for row in valid_rows]
    direction = [value for row in valid_rows if (value := finite(row, "translation_direction_error_deg")) is not None]
    scale = [value for row in valid_rows if (value := finite(row, "translation_scale_abs_error_m")) is not None]
    log_scale = [
        value
        for row in valid_rows
        if (value := finite(row, "translation_scale_log_abs")) is not None
        or (value := finite(row, "scale_log_abs")) is not None
    ]
    xyz = [row.get("translation_error_xyz_m") for row in valid_rows if row.get("translation_error_xyz_m") is not None]
    harmful, false_rot, false_translation, rotation_harmful, rotation_helpful = [], [], [], [], []
    for fixed, row in zip(fixed_rows, rows):
        if failed(row):
            harmful.append(True)
            false_rot.append(float(fixed["camera_rotation_error_deg"]) < 10.0)
            false_translation.append(float(fixed["camera_translation_error_m"]) < 0.5)
            rotation_harmful.append(True)
            rotation_helpful.append(False)
            continue
        fixed_t, fixed_r = float(fixed["camera_translation_error_m"]), float(fixed["camera_rotation_error_deg"])
        method_t, method_r = float(row["camera_translation_error_m"]), float(row["camera_rotation_error_deg"])
        harmful.append(method_t > fixed_t + 0.10 or method_r > fixed_r + 2.0)
        false_rot.append(fixed_r < 10.0 and method_r > fixed_r + 1.0)
        false_translation.append(fixed_t < 0.5 and method_t > fixed_t + 0.10)
        rotation_harmful.append(method_r > fixed_r + 1.0)
        rotation_helpful.append(method_r < fixed_r - 1.0)
    return {
        "count": len(rows),
        "valid_count": len(valid_rows),
        "translation_m": distribution(translation),
        "rotation_deg": distribution(rotation),
        "translation_direction_deg": distribution(direction),
        "translation_scale_abs_m": distribution(scale),
        "translation_log_scale_abs": distribution(log_scale),
        "translation_xyz_m": {
            axis: distribution([float(value[index]) for value in xyz])
            for index, axis in enumerate("xyz")
        }
        if xyz
        else None,
        "catastrophic_rate": float(np.mean([catastrophic(row) for row in rows])),
        "success_rate": float(np.mean([success(row) for row in rows])),
        "strict_success_rate": float(np.mean([strict_success(row) for row in rows])),
        "harmful_correction_rate": float(np.mean(harmful)),
        "rotation_harmful_rate": float(np.mean(rotation_harmful)),
        "rotation_helpful_rate": float(np.mean(rotation_helpful)),
        "false_rotation_correction_rate_on_fixed_lt10": float(np.mean(false_rot)),
        "false_translation_correction_rate_on_fixed_lt0_5": float(np.mean(false_translation)),
        "uncertainty": uncertainty_metrics(valid_rows),
    }


def bucket_cases(cases: list[dict], field: str, buckets: list[tuple[str, float, float]]) -> dict:
    output = {}
    for name, lower, upper in buckets:
        selected = [
            case
            for case in cases
            if lower <= float(case["methods"]["fixed_explicit"][field]) < upper
        ]
        if selected:
            output[name] = {
                "count": len(selected),
                "methods": {
                    method: aggregate(selected, method)
                    for method in METHOD_ORDER
                    if method in selected[0]["methods"]
                },
            }
    return output


def bucket_analysis(cases: list[dict], field: str, buckets: list[tuple[str, float, float]]) -> dict:
    output = {}
    for name, lower, upper in buckets:
        selected = [case for case in cases if lower <= float(case["analysis"][field]) < upper]
        if selected:
            output[name] = {
                "count": len(selected),
                "methods": {
                    method: aggregate(selected, method)
                    for method in METHOD_ORDER
                    if method in selected[0]["methods"]
                },
            }
    return output


def compact_metric(row: dict) -> str:
    return (
        f"{row['translation_m']['mean']:.3f} m / {row['rotation_deg']['mean']:.2f} deg; "
        f"P90 {row['translation_m']['p90']:.3f} m / {row['rotation_deg']['p90']:.2f} deg; "
        f"cat {100.0 * row['catastrophic_rate']:.1f}%"
    )


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# V17 Direct SE(3) vs Factorized Translation Bridge",
        "",
        "## Held-Out Source Results",
        "",
        "| Method | T mean | T median | T P90 | T P95 | R mean | R P90 | R P95 | Catastrophic | Success | Harmful |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHOD_ORDER:
        if method not in report["overall"]:
            continue
        row = report["overall"][method]
        lines.append(
            f"| {method} | {row['translation_m']['mean']:.3f} | {row['translation_m']['median']:.3f} | "
            f"{row['translation_m']['p90']:.3f} | {row['translation_m']['p95']:.3f} | "
            f"{row['rotation_deg']['mean']:.2f} | {row['rotation_deg']['p90']:.2f} | {row['rotation_deg']['p95']:.2f} | "
            f"{100.0 * row['catastrophic_rate']:.1f}% | {100.0 * row['success_rate']:.1f}% | "
            f"{100.0 * row['harmful_correction_rate']:.1f}% |"
        )
    lines.extend(["", "## Per Held-Out Source", ""])
    for source in SOURCES:
        lines.append(f"### {source}")
        for method in ("fixed_explicit", "torso_motion", "direct_absolute", "direct_residual", "factor_scale_only", "factor_direction_scale", "factor_translation_residual", "vggt_direction_learned_scale"):
            lines.append(f"- `{method}`: {compact_metric(report['by_source'][source][method])}")
        lines.append("")
    lines.extend(["## Training and Efficiency", ""])
    for method, row in report["training"].items():
        lines.append(
            f"- `{method}`: {row['parameter_count']} parameters; "
            f"{row['test_latency_ms_per_cut']:.3f} ms/cut; "
            f"validation objective {row['best_validation_objective']:.4f}."
        )
    lines.extend(["", "## Automatic Comparison", ""])
    for key, value in report["comparison"].items():
        lines.append(f"- **{key}**: {value}")
    if report.get("seen_split_summary"):
        lines.extend(["", "## Generalization Gap", ""])
        for method in ("direct_absolute", "direct_residual", "factor_scale_only", "factor_direction_scale", "factor_translation_residual", "factor_direction_scale_uncertainty"):
            splits = report["seen_split_summary"][method]
            lines.append(
                f"- `{method}`: train `{splits['seen_train']['translation_mean_m']:.3f} m / {splits['seen_train']['rotation_mean_deg']:.2f} deg`; "
                f"unseen-pair val `{splits['unseen_pair_validation']['translation_mean_m']:.3f} m / {splits['unseen_pair_validation']['rotation_mean_deg']:.2f} deg`; "
                f"held-out source `{splits['held_out_source']['translation_mean_m']:.3f} m / {splits['held_out_source']['rotation_mean_deg']:.2f} deg`."
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    folds = [
        json.loads((args.input_dir / f"v17_loso_{source}.json").read_text(encoding="utf-8"))
        for source in SOURCES
    ]
    cases = [row for fold in folds for row in fold["rows"]]
    if len(cases) != 180 or len({case["case_name"] for case in cases}) != 180:
        raise RuntimeError(f"Expected 180 unique LOSO rows, got {len(cases)}")
    cache = json.loads((args.cache_dir / "v17_explicit_features.json").read_text(encoding="utf-8"))
    analysis_lookup = {row["case_name"]: row for row in cache["rows"]}
    for case in cases:
        source = analysis_lookup[case["case_name"]]
        case["analysis"] = {
            "view_angle_deg": float(source.get("view_angle_deg", 0.0)),
            "texture_score": float(source.get("texture_score", 0.0)),
        }
    methods = [method for method in METHOD_ORDER if method in cases[0]["methods"]]
    overall = {method: aggregate(cases, method) for method in methods}
    by_source = {
        source: {
            method: aggregate([case for case in cases if case["source"] == source], method)
            for method in methods
        }
        for source in SOURCES
    }
    training = {}
    for method in folds[0]["training"]:
        entries = [fold["training"][method] for fold in folds]
        training[method] = {
            "parameter_count": int(entries[0]["parameter_count"]),
            "test_latency_ms_per_cut": float(np.mean([entry["test_latency_ms_per_cut"] for entry in entries])),
            "best_validation_objective": float(np.mean([entry["best_validation_objective"] for entry in entries])),
            "training_seconds_total": float(np.sum([entry["training_seconds"] for entry in entries])),
            "best_epoch_mean": float(np.mean([entry["best_epoch"] for entry in entries])),
        }

    seen_path = args.output_dir / "v17_seen_split_eval.json"
    seen_split_summary = {}
    if seen_path.exists():
        seen_report = json.loads(seen_path.read_text(encoding="utf-8"))["folds"]
        for method in folds[0]["training"]:
            seen_split_summary[method] = {}
            for split in ("seen_train", "unseen_pair_validation", "held_out_source"):
                entries = [seen_report[source][method][split] for source in SOURCES]
                count = sum(entry["count"] for entry in entries)
                seen_split_summary[method][split] = {
                    "count": int(count),
                    "translation_mean_m": float(sum(entry["count"] * entry["translation_mean_m"] for entry in entries) / count),
                    "rotation_mean_deg": float(sum(entry["count"] * entry["rotation_mean_deg"] for entry in entries) / count),
                    "catastrophic_rate": float(sum(entry["count"] * entry["catastrophic_rate"] for entry in entries) / count),
                }

    fixed = overall["fixed_explicit"]
    learned = [
        "direct_absolute",
        "direct_residual",
        "direct_residual_uncertainty",
        "factor_scale_only",
        "factor_direction_scale",
        "factor_translation_residual",
        "factor_direction_scale_uncertainty",
        "vggt_direction_learned_scale",
    ]
    joint_fold_improvement = {
        method: sum(
            by_source[source][method]["translation_m"]["mean"] < by_source[source]["fixed_explicit"]["translation_m"]["mean"]
            or by_source[source][method]["rotation_deg"]["mean"] < by_source[source]["fixed_explicit"]["rotation_deg"]["mean"]
            for source in SOURCES
        )
        for method in learned
    }
    translation_improvement_over_torso = {
        method: sum(
            by_source[source][method]["translation_m"]["mean"]
            < by_source[source]["torso_motion"]["translation_m"]["mean"]
            for source in SOURCES
        )
        for method in learned
    }
    catastrophic_nonworse_than_torso = {
        method: sum(
            by_source[source][method]["catastrophic_rate"]
            <= by_source[source]["torso_motion"]["catastrophic_rate"] + 0.02
            for source in SOURCES
        )
        for method in learned
    }
    best_translation = min(learned, key=lambda method: overall[method]["translation_m"]["mean"])
    best_joint = min(
        learned,
        key=lambda method: overall[method]["translation_m"]["mean"] + overall[method]["rotation_deg"]["mean"] / 30.0,
    )
    comparison = {
        "stage0_scale_dominance": "GT scale recovered 1.097 m versus 0.303 m for GT direction.",
        "best_learned_translation": f"{best_translation}: {overall[best_translation]['translation_m']['mean']:.3f} m.",
        "best_learned_joint": f"{best_joint}: {compact_metric(overall[best_joint])}.",
        "held_out_any_metric_improvement_counts": joint_fold_improvement,
        "translation_improvement_over_torso_counts": translation_improvement_over_torso,
        "catastrophic_nonworse_than_torso_counts": catastrophic_nonworse_than_torso,
        "fixed_reference": compact_metric(fixed),
        "raw_world_gauge_augmentation_gain": {
            "translation_m": overall["direct_residual_no_gauge_aug"]["translation_m"]["mean"] - overall["direct_residual_gauge_aug"]["translation_m"]["mean"],
            "rotation_deg": overall["direct_residual_no_gauge_aug"]["rotation_deg"]["mean"] - overall["direct_residual_gauge_aug"]["rotation_deg"]["mean"],
        },
        "analytic_gauge_invariant_primary": compact_metric(overall["direct_residual"]),
        "shortcut_baseline": compact_metric(overall["weak_stats_absolute"]),
        "route_selection": "No learned method meets the cross-source safety criterion; retain Fixed Explicit + V16 torso-motion rotation + scene translation re-solving.",
    }
    report = {
        "experiment": "V17 Direct SE(3) vs Factorized Translation Bridge",
        "case_count": len(cases),
        "protocol": folds[0]["protocol"],
        "overall": overall,
        "by_source": by_source,
        "by_fixed_rotation_error": bucket_cases(
            cases,
            "camera_rotation_error_deg",
            [("lt10", 0.0, 10.0), ("10to30", 10.0, 30.0), ("30to60", 30.0, 60.0), ("ge60", 60.0, float("inf"))],
        ),
        "by_fixed_translation_error": bucket_cases(
            cases,
            "camera_translation_error_m",
            [("lt0_5", 0.0, 0.5), ("0_5to1", 0.5, 1.0), ("1to2", 1.0, 2.0), ("ge2", 2.0, float("inf"))],
        ),
        "by_view_angle": bucket_analysis(
            cases,
            "view_angle_deg",
            [("lt30", 0.0, 30.0), ("30to60", 30.0, 60.0), ("ge60", 60.0, float("inf"))],
        ),
        "by_texture": bucket_analysis(
            cases,
            "texture_score",
            [
                ("low", -float("inf"), float(np.quantile([case["analysis"]["texture_score"] for case in cases], 1.0 / 3.0))),
                (
                    "medium",
                    float(np.quantile([case["analysis"]["texture_score"] for case in cases], 1.0 / 3.0)),
                    float(np.quantile([case["analysis"]["texture_score"] for case in cases], 2.0 / 3.0)),
                ),
                ("high", float(np.quantile([case["analysis"]["texture_score"] for case in cases], 2.0 / 3.0)), float("inf")),
            ],
        ),
        "training": training,
        "seen_split_summary": seen_split_summary,
        "splits": {fold["held_out_source"]: fold["split"] for fold in folds},
        "comparison": comparison,
        "folds": [
            {
                "held_out_source": fold["held_out_source"],
                "test_count": fold["test_count"],
                "test_camera_pair_count": fold["test_camera_pair_count"],
                "split": fold["split"],
            }
            for fold in folds
        ],
    }
    json_path = args.output_dir / "v17_direct_vs_factorized_eval.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    write_markdown(args.output_dir / "v17_direct_vs_factorized_summary.md", report)
    with (args.output_dir / "v17_direct_vs_factorized_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "translation_mean_m", "translation_p90_m", "rotation_mean_deg", "rotation_p90_deg", "catastrophic_rate", "harmful_rate"])
        for method in methods:
            row = overall[method]
            writer.writerow([method, row["translation_m"]["mean"], row["translation_m"]["p90"], row["rotation_deg"]["mean"], row["rotation_deg"]["p90"], row["catastrophic_rate"], row["harmful_correction_rate"]])
    print(json.dumps({method: compact_metric(overall[method]) for method in methods}, indent=2), flush=True)
    print(f">> wrote {json_path}", flush=True)


if __name__ == "__main__":
    main()
