#!/usr/bin/env python3
"""Build paper-ready statistics for the frozen v16 Harmony4D candidate."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


BASELINE = "v16_0_m15_geometry"
PRIMARY = "v16_harmony_safe"
METRICS = (
    ("W-MPJPE_mm", "W-MPJPE", "mm", False),
    ("WA-MPJPE_mm", "WA-MPJPE", "mm", False),
    ("MPJPE_mm", "MPJPE", "mm", False),
    ("MPVPE_mm", "MPVPE", "mm", False),
    ("Accel_mm_frame2", "Accel", "mm/frame²", False),
    ("Seam_root_m", "Seam-root", "m", False),
    ("ATE_Sim3_m", "ATE-Sim3", "m", False),
    ("ATE_SE3_m", "ATE-SE3", "m", False),
    ("Boundary_root_m", "Boundary-root", "m", False),
    ("Post_root_m", "Post-root", "m", False),
    ("IDs", "IDs/clip", "count", False),
    ("IDF1", "IDF1", "ratio", True),
    ("Coverage", "Coverage", "ratio", True),
)
LITERATURE_MULTI_THUMBS = {
    "W-MPJPE_mm": 221.0,
    "WA-MPJPE_mm": 116.9,
    "MPJPE_mm": 215.9,
    "MPVPE_mm": 278.3,
    "Accel_mm_frame2": 17.4,
    "ATE_Sim3_m": 0.7,
    "IDs": 0.46,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--base-metrics", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260819)
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def percentile_interval(values: np.ndarray) -> dict[str, float]:
    return {
        "low": float(np.percentile(values, 2.5)),
        "high": float(np.percentile(values, 97.5)),
    }


def paired_bootstrap(
    baseline: np.ndarray,
    primary: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    count = len(baseline)
    indices = rng.integers(0, count, size=(samples, count))
    base_means = baseline[indices].mean(axis=1)
    primary_means = primary[indices].mean(axis=1)
    delta = primary_means - base_means
    relative = delta / np.maximum(np.abs(base_means), 1e-12)
    return {
        "samples": int(samples),
        "resampling_unit": "case_within_one_preregistered_sequence",
        "baseline_ci95": percentile_interval(base_means),
        "primary_ci95": percentile_interval(primary_means),
        "delta_ci95": percentile_interval(delta),
        "relative_delta_ci95": percentile_interval(relative),
    }


def exact_sign_permutation_p(differences: np.ndarray) -> float:
    """Two-sided exact paired sign-randomization p-value."""

    differences = np.asarray(differences, dtype=np.float64)
    observed = abs(float(differences.mean()))
    statistics = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        statistics.append(abs(float(np.mean(differences * np.asarray(signs)))))
    return float(np.mean(np.asarray(statistics) >= observed - 1e-12))


def nested(value: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    if current is None:
        return None
    number = float(current)
    return number if math.isfinite(number) else None


def base_method_summary(paths: list[Path]) -> dict[str, Any]:
    reports = []
    for root in paths:
        for path in sorted(root.rglob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if payload.get("schema_version") == "Movie3R-Harmony4D-evaluation-v1":
                reports.append(payload)
    methods = {
        "strict_human3r": "m0_strict_human3r",
        "m15": "m15_safe_boundary_permutation_causal_gru",
    }
    source_paths = {
        "W-MPJPE_mm": ("multi_thumbs_named_provisional", "w_mpjpe_mm", "mean"),
        "WA-MPJPE_mm": ("multi_thumbs_named_provisional", "wa_mpjpe_mm", "mean"),
        "MPJPE_mm": ("multi_thumbs_named_provisional", "mpjpe_mm", "mean"),
        "MPVPE_mm": ("multi_thumbs_named_provisional", "mpvpe_mm", "mean"),
        "Accel_mm_frame2": ("multi_thumbs_named_provisional", "accel_delta2_mm_per_frame2", "mean"),
        "ATE_Sim3_m": ("multi_thumbs_named_provisional", "ate_sim3_m", "mean"),
        "Seam_root_m": ("cut_seam", "root_excess_m"),
        "IDs": ("identity", "ids_total"),
        "IDF1": ("identity", "idf1"),
        "Coverage": ("coverage", "coverage"),
    }
    output: dict[str, Any] = {"case_count": len(reports), "methods": {}}
    for label, method in methods.items():
        values = {}
        for metric, path in source_paths.items():
            rows = [nested(report["methods"][method], path) for report in reports]
            valid = [value for value in rows if value is not None]
            values[metric] = mean(valid) if valid else None
        output["methods"][label] = values
    return output


def write_csv(path: Path, paired_rows: list[dict[str, Any]]) -> None:
    fields = ["case_id", "angle_stratum", "gate_accepted"]
    for key, _, _, _ in METRICS:
        fields.extend((f"baseline_{key}", f"primary_{key}", f"delta_{key}"))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(paired_rows)


def write_tex(path: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Metric & M15 & Movie3R-v16 & Change \\\\",
        "\\midrule",
    ]
    for key, label, unit, _ in METRICS:
        row = metrics[key]
        suffix = "\\%" if unit == "ratio" else ""
        scale = 100.0 if unit == "ratio" else 1.0
        lines.append(
            f"{label} & {row['baseline_mean'] * scale:.3f}{suffix} & "
            f"{row['primary_mean'] * scale:.3f}{suffix} & "
            f"{row['relative_change_percent']:+.1f}\\% \\\\" 
        )
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    payload = json.loads(args.final.read_text(encoding="utf-8"))
    baseline_rows = {
        row["case_id"]: row for row in payload["rows"]
        if row["candidate"] == BASELINE and row["status"] == "complete"
    }
    primary_rows = {
        row["case_id"]: row for row in payload["rows"]
        if row["candidate"] == PRIMARY and row["status"] == "complete"
    }
    case_ids = sorted(set(baseline_rows) & set(primary_rows))
    if not case_ids:
        raise ValueError("No paired final cases")
    rng = np.random.default_rng(int(args.seed))
    metric_summary = {}
    for key, label, unit, higher_is_better in METRICS:
        baseline = np.asarray([baseline_rows[case]["metrics"][key] for case in case_ids], dtype=np.float64)
        primary = np.asarray([primary_rows[case]["metrics"][key] for case in case_ids], dtype=np.float64)
        delta = primary - baseline
        base_mean, primary_mean = float(baseline.mean()), float(primary.mean())
        metric_summary[key] = {
            "label": label,
            "unit": unit,
            "higher_is_better": higher_is_better,
            "baseline_mean": base_mean,
            "primary_mean": primary_mean,
            "absolute_change": primary_mean - base_mean,
            "relative_change_percent": 100.0 * (primary_mean / base_mean - 1.0),
            "nonworse_cases": int(np.sum(primary >= baseline - 1e-3) if higher_is_better else np.sum(primary <= baseline + 1e-3)),
            "paired_exact_sign_permutation_p_two_sided": exact_sign_permutation_p(delta),
            "case_bootstrap": paired_bootstrap(baseline, primary, int(args.bootstrap), rng),
        }
    paired_rows = []
    gate_accepts = 0
    for case in case_ids:
        base, primary = baseline_rows[case], primary_rows[case]
        accepted = bool(primary["diagnostics"]["reliability_gate"]["accepted"])
        gate_accepts += int(accepted)
        row: dict[str, Any] = {
            "case_id": case,
            "angle_stratum": primary["angle_stratum"],
            "gate_accepted": accepted,
        }
        for key, _, _, _ in METRICS:
            first, second = base["metrics"][key], primary["metrics"][key]
            row[f"baseline_{key}"] = first
            row[f"primary_{key}"] = second
            row[f"delta_{key}"] = second - first
        paired_rows.append(row)
    literature = {}
    for key, reference in LITERATURE_MULTI_THUMBS.items():
        ours = metric_summary[key]["primary_mean"]
        literature[key] = {
            "multi_thumbs_public_value": reference,
            "movie3r_v16_value": ours,
            "ratio_movie3r_over_reference": ours / reference,
            "comparison_contract": "literature scale reference only; exact public protocol unavailable",
        }
    result = {
        "schema_version": "Movie3R-v16-Harmony4D-paper-summary-v1",
        "final_source": str(args.final.resolve()),
        "method": PRIMARY,
        "baseline": BASELINE,
        "case_count": len(case_ids),
        "gate": {"accepted": gate_accepts, "fallback": len(case_ids) - gate_accepts},
        "metrics": metric_summary,
        "same_protocol_baselines": base_method_summary(args.base_metrics),
        "multi_thumbs_literature_reference": literature,
        "paired_rows": paired_rows,
        "inference": {
            "warning": "All uncertainty intervals resample four clips from one preregistered sequence; they do not establish cross-sequence significance."
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_csv(args.output / "paired_case_metrics.csv", paired_rows)
    write_tex(args.output / "main_table.tex", metric_summary)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "cases": len(case_ids),
        "gate_accepts": gate_accepts,
        "W_change_percent": metric_summary["W-MPJPE_mm"]["relative_change_percent"],
        "Accel_change_percent": metric_summary["Accel_mm_frame2"]["relative_change_percent"],
        "Seam_change_percent": metric_summary["Seam_root_m"]["relative_change_percent"],
    }, indent=2))


if __name__ == "__main__":
    main()
