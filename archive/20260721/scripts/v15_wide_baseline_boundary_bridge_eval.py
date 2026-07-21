#!/usr/bin/env python3
"""Aggregate V15 Boundary Bridge candidates, capture-basin groups and diagnostics."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "output/v15_wide_baseline_boundary_bridge"


METHODS = {
    "original_continue": ("baseline", "original_continue"),
    "hard_reset": ("baseline", "hard_reset"),
    "fixed_explicit": ("baseline", "fixed_explicit"),
    "v14_world_memory_icp": ("baseline", "v14_world_memory_3f"),
    "wide_1p1_coarse": ("window", "full_rgb_1p1", "coarse"),
    "wide_3p3_coarse": ("window", "full_rgb_3p3", "coarse"),
    "wide_1p1_rotation_fixed_translation": ("window", "full_rgb_1p1", "rotation_fixed_translation"),
    "wide_3p3_rotation_fixed_translation": ("window", "full_rgb_3p3", "rotation_fixed_translation"),
    "wide_1p1_correspondence_metric": ("window", "full_rgb_1p1", "wide_rotation_metric_translation_downweighted"),
    "wide_3p3_correspondence_metric": ("window", "full_rgb_3p3", "wide_rotation_metric_translation_downweighted"),
    "wide_1p1_hybrid": ("window", "full_rgb_1p1", "hybrid_downweighted"),
    "wide_3p3_hybrid": ("window", "full_rgb_3p3", "hybrid_downweighted"),
    "wide_3p3_full_rigid_metric": ("window", "full_rgb_3p3", "metric_full_downweighted"),
    "background_1p1_coarse": ("window", "background_only_1p1", "coarse"),
    "background_3p3_coarse": ("window", "background_only_3p3", "coarse"),
    "background_1p1_metric": ("window", "background_only_1p1", "wide_rotation_metric_translation_background"),
    "background_3p3_metric": ("window", "background_only_3p3", "wide_rotation_metric_translation_background"),
    "background_1p1_hybrid": ("window", "background_only_1p1", "hybrid_background"),
    "background_3p3_hybrid": ("window", "background_only_3p3", "hybrid_background"),
    "boundary_oracle": ("baseline", "boundary_oracle"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_dir", type=Path, default=DEFAULT_ROOT / "candidate_cache")
    parser.add_argument(
        "--v13_dir",
        type=Path,
        default=REPO_ROOT / "output/v13_world_coordinate_memory/stage1_scene_coordinate_oracle",
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_ROOT / "evaluation")
    return parser.parse_args()


def load_cases(pattern: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(pattern)):
        rows.extend(json.loads(Path(path).read_text(encoding="utf-8"))["cases"])
    return rows


def method_row(case: dict, method: str) -> dict | None:
    spec = METHODS[method]
    if spec[0] == "baseline":
        return case["baselines"].get(spec[1])
    return case["windows"].get(spec[1], {}).get("candidates", {}).get(spec[2])


def failed(row: dict | None) -> bool:
    return row is None or bool(row.get("fit_failed", False)) or not np.isfinite(row.get("camera_translation_error_m", np.nan))


def catastrophic(row: dict | None) -> bool:
    return failed(row) or float(row["camera_translation_error_m"]) > 1.0 or float(row["camera_rotation_error_deg"]) > 30.0


def success(row: dict | None) -> bool:
    return not failed(row) and float(row["camera_translation_error_m"]) < 0.25 and float(row["camera_rotation_error_deg"]) < 5.0


def strict_success(row: dict | None) -> bool:
    return not failed(row) and float(row["camera_translation_error_m"]) < 0.10 and float(row["camera_rotation_error_deg"]) < 2.0


def cost(row: dict | None) -> float:
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


def stats(cases: list[dict], method: str) -> dict:
    values = [method_row(case, method) for case in cases]
    valid = [row for row in values if not failed(row)]
    fixed = [method_row(case, "fixed_explicit") for case in cases]
    return {
        "case_count": len(cases),
        "fit_failure_rate": float(np.mean([failed(row) for row in values])) if values else None,
        "translation_m": distribution([float(row["camera_translation_error_m"]) for row in valid]),
        "rotation_deg": distribution([float(row["camera_rotation_error_deg"]) for row in valid]),
        "translation_direction_deg": distribution([float(row.get("translation_direction_error_deg", np.nan)) for row in valid]),
        "translation_scale_log_abs": distribution([float(row.get("translation_scale_log_abs", np.nan)) for row in valid]),
        "catastrophic_rate": float(np.mean([catastrophic(row) for row in values])) if values else None,
        "success_rate": float(np.mean([success(row) for row in values])) if values else None,
        "strict_success_rate": float(np.mean([strict_success(row) for row in values])) if values else None,
        "helpful_vs_fixed_rate": float(np.mean([cost(row) + 0.05 < cost(base) for row, base in zip(values, fixed)])) if values else None,
        "mean_joint_gain_vs_fixed": float(np.mean([cost(base) - cost(row) for row, base in zip(values, fixed)])) if values else None,
    }


def oracle_row(case: dict, methods: list[str]) -> tuple[str, dict]:
    candidates = [(method, method_row(case, method)) for method in methods]
    candidates = [(method, row) for method, row in candidates if not failed(row)]
    if not candidates:
        return "fixed_explicit", method_row(case, "fixed_explicit")
    return min(candidates, key=lambda item: (catastrophic(item[1]), cost(item[1])))


def oracle_stats(cases: list[dict], methods: list[str]) -> dict:
    selected = [oracle_row(case, methods) for case in cases]
    rows = [row for _, row in selected]
    actions = {}
    for method, _ in selected:
        actions[method] = actions.get(method, 0) + 1
    valid = [row for row in rows if not failed(row)]
    return {
        "case_count": len(cases),
        "translation_m": distribution([float(row["camera_translation_error_m"]) for row in valid]),
        "rotation_deg": distribution([float(row["camera_rotation_error_deg"]) for row in valid]),
        "catastrophic_rate": float(np.mean([catastrophic(row) for row in rows])),
        "success_rate": float(np.mean([success(row) for row in rows])),
        "strict_success_rate": float(np.mean([strict_success(row) for row in rows])),
        "actions": actions,
    }


def diagnostic_stats(cases: list[dict], method: str) -> dict:
    rows = [method_row(case, method) for case in cases]
    rows = [row for row in rows if not failed(row)]
    keys = (
        "correspondence_count",
        "mutual_track_ratio",
        "epipolar_median_px",
        "epipolar_p90_px",
        "image_coverage_8x8",
        "fit_residual_median_m",
        "refinement_translation_m",
        "refinement_rotation_deg",
        "human_correspondence_ratio",
        "human_root_jump_m",
        "human_torso_jump_deg",
    )
    return {key: distribution([float(row.get(key, np.nan)) for row in rows]) for key in keys}


def human_check_stats(cases: list[dict], method: str) -> dict:
    rows = [method_row(case, method) for case in cases]
    labels = np.asarray([catastrophic(row) for row in rows], dtype=bool)
    output = {}
    for key in ("human_root_jump_m", "human_torso_jump_deg"):
        values = np.asarray([float(row.get(key, np.nan)) for row in rows], dtype=np.float64)
        valid = np.isfinite(values)
        if int(valid.sum()) and len(np.unique(labels[valid])) == 2:
            auc = float(roc_auc_score(labels[valid], values[valid]))
        else:
            auc = None
        output[key] = {
            "failure_auroc": auc,
            "noncatastrophic_mean": float(values[valid & ~labels].mean()) if (valid & ~labels).any() else None,
            "catastrophic_mean": float(values[valid & labels].mean()) if (valid & labels).any() else None,
        }
    return output


def label_terciles(cases: list[dict], key: str, names: tuple[str, str, str]) -> dict[str, list[dict]]:
    values = np.asarray([float(case[key]) for case in cases], dtype=np.float64)
    low, high = np.nanpercentile(values, [33.333, 66.667])
    output = {name: [] for name in names}
    for case, value in zip(cases, values):
        label = names[0] if value <= low else names[1] if value <= high else names[2]
        output[label].append(case)
    return output


def bucket_groups(cases: list[dict], key_fn, buckets: list[tuple[str, float, float]]) -> dict[str, list[dict]]:
    output = {name: [] for name, _, _ in buckets}
    for case in cases:
        value = key_fn(case)
        for name, lower, upper in buckets:
            if lower <= value < upper:
                output[name].append(case)
                break
    return output


def group_report(groups: dict[str, list[dict]], methods: list[str]) -> dict:
    return {
        name: {method: stats(rows, method) for method in methods}
        for name, rows in groups.items()
        if rows
    }


def md_value(summary: dict, metric: str, stat: str = "mean", scale: float = 1.0) -> str:
    value = summary[metric][stat]
    return "n/a" if value is None else f"{value * scale:.3f}"


def build_markdown(report: dict, output: Path) -> None:
    methods = (
        "fixed_explicit",
        "v14_world_memory_icp",
        "wide_1p1_coarse",
        "wide_3p3_coarse",
        "wide_1p1_rotation_fixed_translation",
        "wide_3p3_rotation_fixed_translation",
        "wide_1p1_correspondence_metric",
        "wide_3p3_correspondence_metric",
        "wide_1p1_hybrid",
        "wide_3p3_hybrid",
        "background_1p1_metric",
        "background_3p3_metric",
        "boundary_oracle",
    )
    lines = ["# V15 Evaluation Summary", "", "| Method | T mean | T P90 | R mean | R P90 | Catastrophic | Success |", "|---|---:|---:|---:|---:|---:|---:|"]
    for method in methods:
        row = report["overall"][method]
        lines.append(
            f"| {method} | {md_value(row, 'translation_m')} | {md_value(row, 'translation_m', 'p90')} | "
            f"{md_value(row, 'rotation_deg')} | {md_value(row, 'rotation_deg', 'p90')} | "
            f"{row['catastrophic_rate'] * 100:.1f}% | {row['success_rate'] * 100:.1f}% |"
        )
    oracle = report["oracle"]
    lines.extend(
        [
            "",
            "## Oracle Candidate Complementarity",
            "",
            f"Best of Fixed/Wide/Hybrid: {oracle['translation_m']['mean']:.3f} m / "
            f"{oracle['rotation_deg']['mean']:.3f} deg, catastrophic {oracle['catastrophic_rate'] * 100:.1f}%.",
            "",
            "## Source",
            "",
        ]
    )
    for source, source_rows in report["sources"].items():
        fixed = source_rows["fixed_explicit"]
        wide = source_rows["wide_3p3_correspondence_metric"]
        lines.append(
            f"- {source}: Fixed {fixed['translation_m']['mean']:.3f} m / {fixed['rotation_deg']['mean']:.2f} deg; "
            f"Wide metric {wide['translation_m']['mean']:.3f} m / {wide['rotation_deg']['mean']:.2f} deg."
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    cases = load_cases(str(args.candidate_dir / "v15_candidates_shard_*_of_*.json"))
    if len(cases) != 180:
        raise RuntimeError(f"Expected 180 V15 cases, found {len(cases)}")
    names = [case["case_name"] for case in cases]
    if len(set(names)) != len(names):
        raise RuntimeError("Duplicate V15 cases")
    v13_cases = load_cases(str(args.v13_dir / "stage1_shard_*_of_*.json"))
    overlap = {
        case["case_name"]: float(np.mean(case["correspondence"]["static"]["frame_overlap_ratio_0_20m"]))
        for case in v13_cases
    }
    for case in cases:
        case["pseudo_overlap"] = overlap.get(case["case_name"], float("nan"))
        v14_row = case["baselines"].get("v14_world_memory_3f") or {}
        case["planarity_ratio"] = float(v14_row.get("target_geometry", {}).get("planarity_ratio", float("nan")))
    report_methods = list(METHODS)
    key_methods = [
        "fixed_explicit",
        "v14_world_memory_icp",
        "wide_1p1_coarse",
        "wide_3p3_coarse",
        "wide_1p1_rotation_fixed_translation",
        "wide_3p3_rotation_fixed_translation",
        "wide_1p1_correspondence_metric",
        "wide_3p3_correspondence_metric",
        "wide_1p1_hybrid",
        "wide_3p3_hybrid",
        "background_1p1_metric",
        "background_3p3_metric",
    ]
    initial_rotation_groups = bucket_groups(
        cases,
        lambda case: float(method_row(case, "fixed_explicit")["camera_rotation_error_deg"]),
        [("lt10", 0.0, 10.0), ("10to30", 10.0, 30.0), ("30to60", 30.0, 60.0), ("ge60", 60.0, float("inf"))],
    )
    initial_translation_groups = bucket_groups(
        cases,
        lambda case: float(method_row(case, "fixed_explicit")["camera_translation_error_m"]),
        [("lt0p5", 0.0, 0.5), ("0p5to1", 0.5, 1.0), ("1to2", 1.0, 2.0), ("ge2", 2.0, float("inf"))],
    )
    angle_groups = bucket_groups(
        cases,
        lambda case: float(case["record"].get("view_angle_deg", 0.0)),
        [("small", 0.0, 60.0), ("medium", 60.0, 120.0), ("large", 120.0, float("inf"))],
    )
    source_groups = {}
    for case in cases:
        source_groups.setdefault(str(case["record"]["source"]), []).append(case)
    timing_keys = sorted({key for case in cases for key in case["timing_seconds"]})
    report = {
        "experiment": "V15 Wide-Baseline Boundary Bridge Feasibility Probe",
        "case_count": len(cases),
        "protocol": cases[0]["protocol"],
        "overall": {method: stats(cases, method) for method in report_methods},
        "oracle": oracle_stats(cases, key_methods),
        "oracle_fixed_wide_1p1_coarse": oracle_stats(cases, ["fixed_explicit", "wide_1p1_coarse"]),
        "oracle_fixed_wide_1p1_hybrid": oracle_stats(cases, ["fixed_explicit", "wide_1p1_hybrid"]),
        "sources": {source: {method: stats(rows, method) for method in key_methods} for source, rows in source_groups.items()},
        "groups": {
            "fixed_initial_rotation": group_report(initial_rotation_groups, key_methods),
            "fixed_initial_translation": group_report(initial_translation_groups, key_methods),
            "view_angle": group_report(angle_groups, key_methods),
            "texture": group_report(label_terciles(cases, "texture_score", ("low", "medium", "high")), key_methods),
            "pseudo_overlap": group_report(label_terciles(cases, "pseudo_overlap", ("low", "medium", "high")), key_methods),
            "geometry": group_report(
                label_terciles(cases, "planarity_ratio", ("planar", "medium", "nondegenerate")), key_methods
            ),
            "human_image_ratio": group_report(label_terciles(cases, "human_image_ratio", ("low", "medium", "high")), key_methods),
        },
        "matching_diagnostics": {
            method: diagnostic_stats(cases, method)
            for method in (
                "wide_1p1_correspondence_metric",
                "wide_3p3_correspondence_metric",
                "wide_1p1_hybrid",
                "wide_3p3_hybrid",
                "background_1p1_metric",
                "background_3p3_metric",
            )
        },
        "human_motion_checks": {
            method: human_check_stats(cases, method)
            for method in ("wide_1p1_coarse", "wide_1p1_correspondence_metric", "wide_1p1_hybrid", "background_1p1_metric")
        },
        "timing_seconds": {key: distribution([float(case["timing_seconds"].get(key, np.nan)) for case in cases]) for key in timing_keys},
        "peak_gpu_memory_gb": distribution([float(case["peak_gpu_memory_gb"]) for case in cases]),
        "ordinary_frame_fps_change": 0.0,
        "shot_transform_count": 1,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_dir / "v15_eval.json"
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    build_markdown(report, args.output_dir / "v15_summary.md")
    print(json.dumps({"overall": report["overall"], "oracle": report["oracle"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
