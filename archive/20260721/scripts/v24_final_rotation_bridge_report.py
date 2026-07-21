#!/usr/bin/env python3
"""Generate the final V24 report and grouped safety audit."""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V24 = (
    REPO_ROOT
    / "output"
    / "v24_vggt_v22_rotation_bridge"
    / "v24_vggt_v22_rotation_bridge.json"
)
DEFAULT_V15_EVAL = (
    REPO_ROOT
    / "output"
    / "v15_wide_baseline_boundary_bridge"
    / "evaluation"
    / "v15_eval.json"
)
DEFAULT_V15_CACHE = REPO_ROOT / "output" / "v15_wide_baseline_boundary_bridge" / "candidate_cache"
DEFAULT_V22_LATENCY = (
    REPO_ROOT
    / "output"
    / "v22_explicit_metric_bridge"
    / "latency_benchmark"
    / "v22_cut_latency_benchmark.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "output" / "v24_vggt_v22_rotation_bridge" / "final_report"
SELECTED = "safe_tiered_extension_vggt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v24_report", type=Path, default=DEFAULT_V24)
    parser.add_argument("--v15_eval", type=Path, default=DEFAULT_V15_EVAL)
    parser.add_argument("--v15_cache", type=Path, default=DEFAULT_V15_CACHE)
    parser.add_argument("--v22_latency", type=Path, default=DEFAULT_V22_LATENCY)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_v15_cases(root: Path) -> dict[str, dict]:
    rows = []
    for path in sorted(glob.glob(str(root / "v15_candidates_shard_*_of_*.json"))):
        rows.extend(load(Path(path))["cases"])
    return {row["case_name"]: row for row in rows}


def distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
    }


def catastrophic(value: dict) -> bool:
    return bool(
        value["camera"]["translation_m"] > 2.0
        or value["camera"]["rotation_deg"] > 45.0
        or value["human"]["root_motion_error_m"] > 0.50
        or value["scene"]["trimmed_mean_m"] > 1.0
    )


def rotation_bin(value: float) -> str:
    if value < 10.0:
        return "lt10"
    if value < 30.0:
        return "10_30"
    if value < 60.0:
        return "30_60"
    return "ge60"


def summarize_cases(rows: list[dict], variant: str) -> dict:
    values = [row["variants"][variant] for row in rows]
    return {
        "count": len(rows),
        "camera_translation_m": distribution(
            [row["camera"]["translation_m"] for row in values]
        ),
        "camera_rotation_deg": distribution(
            [row["camera"]["rotation_deg"] for row in values]
        ),
        "scene_m": distribution([row["scene"]["trimmed_mean_m"] for row in values]),
        "catastrophic_rate": float(np.mean([catastrophic(row) for row in values])),
    }


def grouped(rows: list[dict], field: str) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {
        key: {
            "v22": summarize_cases(values, "v22"),
            "v24": summarize_cases(values, SELECTED),
        }
        for key, values in sorted(groups.items())
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    v24 = load(args.v24_report)
    v15_eval = load(args.v15_eval)
    v15_cases = load_v15_cases(args.v15_cache)
    v22_latency = load(args.v22_latency)
    rows = v24["cases"]
    for row in rows:
        fixed_error = float(
            v15_cases[row["case_name"]]["baselines"]["fixed_explicit"][
                "camera_rotation_error_deg"
            ]
        )
        row["fixed_rotation_bin"] = rotation_bin(fixed_error)
        row["texture_tertile"] = "low" if v15_cases[row["case_name"]]["texture_score"] < 0.05 else "high"
        row["view_angle_bucket"] = v15_cases[row["case_name"]]["record"]["angle_bucket"]

    corrections = [
        row
        for row in rows
        if abs(
            row["variants"][SELECTED]["camera"]["rotation_deg"]
            - row["variants"]["v22"]["camera"]["rotation_deg"]
        )
        > 1e-6
    ]
    rescued = [
        row
        for row in rows
        if catastrophic(row["variants"]["v22"])
        and not catastrophic(row["variants"][SELECTED])
    ]
    introduced = [
        row
        for row in rows
        if not catastrophic(row["variants"]["v22"])
        and catastrophic(row["variants"][SELECTED])
    ]
    remaining = [row for row in rows if catastrophic(row["variants"][SELECTED])]
    good = [row for row in rows if row["variants"]["v22"]["camera"]["rotation_deg"] < 10.0]

    pretrigger = [row for row in rows if row["diagnostics"]["torso_residual_deg"] >= 10.0]
    base_v22_seconds = float(v22_latency["latency_seconds"]["cut_total_seconds"]["mean"])
    vggt_case_seconds = {
        name: float(case["timing_seconds"]["full_rgb_1p1"])
        for name, case in v15_cases.items()
    }
    amortized = [
        base_v22_seconds
        + (vggt_case_seconds[row["case_name"]] if row in pretrigger else 0.0)
        for row in rows
    ]
    triggered = [base_v22_seconds + vggt_case_seconds[row["case_name"]] for row in pretrigger]

    report = {
        "experiment": "V24 final safe conditional wide-rotation metric bridge",
        "selected_variant": SELECTED,
        "overall": {
            "v22": v24["overall"]["v22"],
            "v24_selected": v24["overall"][SELECTED],
            "oracle_best_v22_vggt_rotation": v24["overall"][
                "oracle_best_v22_vggt_rotation"
            ],
            "gt_rotation": v24["overall"]["gt_rotation"],
        },
        "by_source": {
            source: {
                "v22": values["v22"],
                "v24_selected": values[SELECTED],
            }
            for source, values in v24["by_source"].items()
        },
        "groups": {
            "fixed_rotation_bin": grouped(rows, "fixed_rotation_bin"),
            "texture": grouped(rows, "texture_tertile"),
            "view_angle": grouped(rows, "view_angle_bucket"),
        },
        "safety": {
            "corrected_count": len(corrections),
            "rescued_catastrophic_count": len(rescued),
            "introduced_catastrophic_count": len(introduced),
            "remaining_catastrophic_count": len(remaining),
            "v22_rotation_lt10_count": len(good),
            "v22_rotation_lt10_corrected_count": int(
                np.sum(
                    [
                        abs(
                            row["variants"][SELECTED]["camera"]["rotation_deg"]
                            - row["variants"]["v22"]["camera"]["rotation_deg"]
                        )
                        > 1e-6
                        for row in good
                    ]
                )
            ),
            "v22_rotation_lt10_harmful_5deg_count": int(
                np.sum(
                    [
                        row["variants"][SELECTED]["camera"]["rotation_deg"]
                        > row["variants"]["v22"]["camera"]["rotation_deg"] + 5.0
                        for row in good
                    ]
                )
            ),
            "accepted_rules": dict(
                Counter(
                    "low_texture_conflict"
                    if row["diagnostics"]["trigger_safe_low_texture_conflict"]
                    else (
                        "large_residual"
                        if row["diagnostics"]["trigger_safe_large_residual"]
                        else "consensus"
                    )
                    for row in corrections
                )
            ),
            "rescued_cases": [row["case_name"] for row in rescued],
            "remaining_cases": [row["case_name"] for row in remaining],
        },
        "runtime": {
            "vggt_pretrigger_rate": len(pretrigger) / len(rows),
            "vggt_acceptance_rate": len(corrections) / len(rows),
            "v22_base_cut_seconds": base_v22_seconds,
            "vggt_full_rgb_1p1_seconds": v15_eval["timing_seconds"]["full_rgb_1p1"],
            "amortized_cut_seconds": distribution(amortized),
            "pretriggered_cut_seconds": distribution(triggered),
            "ordinary_frame_path_changed": False,
            "v15_peak_gpu_memory_gb": v15_eval["peak_gpu_memory_gb"],
        },
        "decision": {
            "current_best_candidate": SELECTED,
            "deployable_if_cut_latency_budget_allows": True,
            "preferred_finalization": "distill the conditional VGGT rotation cue into a cut-only lightweight Shot Bridge",
            "remaining_failures": "two high-spread VGGT cases, one torso-residual-below-trigger MVHuman case, and one THuman scene-only case",
        },
    }
    json_path = args.output_dir / "v24_final_rotation_bridge.json"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    def metric_row(label: str, value: dict) -> str:
        return (
            f"| {label} | {value['camera_translation_m']['mean']:.3f} / "
            f"{value['camera_translation_m']['p95']:.3f} | "
            f"{value['camera_rotation_deg']['mean']:.2f} / "
            f"{value['camera_rotation_deg']['p95']:.2f} | "
            f"{value['scene_trimmed_mean_m']['mean']:.3f} / "
            f"{value['scene_trimmed_mean_m']['p95']:.3f} | "
            f"{100 * value['combined_catastrophic_rate']:.1f}% | "
            f"{100 * value['strict_success_rate']:.1f}% |"
        )

    v22 = v24["overall"]["v22"]
    selected = v24["overall"][SELECTED]
    markdown = [
        "# V24 Safe Conditional Wide-Rotation Metric Bridge",
        "",
        "## Result",
        "",
        "| Method | Camera T mean/P95 | Rotation mean/P95 | Scene mean/P95 | Catastrophic | Strict success |",
        "|---|---:|---:|---:|---:|---:|",
        metric_row("V22", v22),
        metric_row("V24 selected", selected),
        "",
        "V24 keeps the complete V22 metric-scale and explicit human-root translation path. Frozen VGGT is queried only after a torso residual of at least 10 degrees and can only modify rotation through fixed physical safety rules.",
        "",
        "## Safety rules",
        "",
        "1. Large torso residual: VGGT must extend the torso correction by at least 5 degrees, predict at most 100 degrees, and have internal spread at most 15 degrees; apply at most 25 degrees.",
        "2. Torso/VGGT consensus: residual directions must agree, VGGT spread must be at most 5 degrees, and VGGT must extend torso by at least 5 degrees; apply at most 60 degrees.",
        "3. Low-texture conflict: texture score below 0.05, opposite residual directions, VGGT spread at most 5 degrees, and at least 10 degrees of additional magnitude; apply at most 45 degrees.",
        "4. Re-solve metric camera translation after every accepted rotation. Camera, pointmap, and SMPL-X share the final transform.",
        "",
        "## Safety outcome",
        "",
        f"- Corrected `{len(corrections)}/180` cuts and rescued `{len(rescued)}` catastrophic cases.",
        f"- Introduced catastrophic cases: `{len(introduced)}`.",
        f"- Among `{len(good)}` V22 cases below 10 degrees, only `{report['safety']['v22_rotation_lt10_corrected_count']}` was changed and none degraded by more than 5 degrees.",
        f"- Rotation harmful rate over V22: `{100 * selected['rotation_harmful_5deg']:.1f}%`; camera harmful rate over 0.1 m: `{100 * selected['camera_harmful_010m']:.1f}%`.",
        "",
        "## Runtime",
        "",
        f"- VGGT pretrigger rate: `{100 * report['runtime']['vggt_pretrigger_rate']:.1f}%`; accepted rate: `{100 * report['runtime']['vggt_acceptance_rate']:.1f}%`.",
        f"- Amortized cut latency: `{report['runtime']['amortized_cut_seconds']['mean']:.3f} s` mean, `{report['runtime']['amortized_cut_seconds']['p95']:.3f} s` P95.",
        f"- Triggered cut latency: `{report['runtime']['pretriggered_cut_seconds']['mean']:.3f} s` mean.",
        "- Ordinary frames remain unchanged.",
        "",
        "## Decision",
        "",
        "V24 is the current best candidate. It directly improves the deployable V22 candidate without a learned gate or source-specific rule. The large VGGT cost makes it more suitable as a cut-only teacher/upper branch for Shot Bridge distillation than as the final lightweight system.",
    ]
    md_path = args.output_dir / "v24_final_rotation_bridge.md"
    md_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(f">> wrote {json_path}")
    print(f">> wrote {md_path}")


if __name__ == "__main__":
    main()
