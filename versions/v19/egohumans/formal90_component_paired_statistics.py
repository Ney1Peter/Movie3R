#!/usr/bin/env python3
"""Capture-cluster paired statistics for the EgoHumans formal-90 ablations.

This is a retrospective analysis of the immutable per-case metric CSV emitted
by ``aggregate_formal90_ablation.py``.  It never reads model predictions, RGB,
or ground truth and cannot select or alter a Test-time route.  For every named
component route, the script pairs all cases with Bridge3R (full), keeps camera
pairs clustered by capture, and reports:

* a case-macro paired improvement (positive always favours the full route),
* a paired capture-resampling bootstrap with within-capture case resampling,
* a two-sided capture-cluster sign-flip randomisation test, and
* per-case win/tie/loss accounting.

The source digest, exact case-level paired values, JSON summary, and Markdown
summary are retained so that every displayed number can be reconstructed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "Bridge3R-EgoHumans-formal90-component-paired-statistics-v1"
FULL = "Bridge3R (full)"
COMPARATORS = (
    "No learned correction branch",
    "Alignment only",
    "Full token, LoRA off",
    "Full token, camera residual off",
)
METRICS: tuple[tuple[str, str, str, str], ...] = (
    ("W-MPJPE_mm", "W-MPJPE", "mm", "lower"),
    ("WA-MPJPE_mm", "WA-MPJPE", "mm", "lower"),
    ("MPJPE_mm", "MPJPE", "mm", "lower"),
    ("MPVPE_mm", "MPVPE", "mm", "lower"),
    ("ATE_SE3_m", "ATE-SE3", "m", "lower"),
    ("Seam_root_m", "Human seam", "m", "lower"),
    ("IDF1", "IDF1", "unitless", "higher"),
    ("Coverage", "Coverage", "unitless", "higher"),
)
NUMERICAL_TIE_ATOL = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--signflip-samples", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=20260829)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(raw: Any) -> float | None:
    if raw is None or str(raw).strip() == "":
        return None
    value = float(raw)
    return value if math.isfinite(value) else None


def load_rows(path: Path) -> dict[str, dict[str, dict[str, str]]]:
    by_method: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "method", "case_id", "capture", "sequence", "angle_stratum", "angle_deg",
            *[metric for metric, _, _, _ in METRICS],
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"input CSV misses fields: {sorted(missing)}")
        for row in reader:
            method = row["method"]
            case_id = row["case_id"]
            if case_id in by_method[method]:
                raise ValueError(f"duplicate {method!r} row for {case_id}")
            by_method[method][case_id] = row
    expected_methods = {FULL, *COMPARATORS}
    absent = expected_methods.difference(by_method)
    if absent:
        raise ValueError(f"input CSV misses methods: {sorted(absent)}")
    expected_cases = set(by_method[FULL])
    if len(expected_cases) != 90:
        raise ValueError(f"full route has {len(expected_cases)} cases, expected 90")
    for method in expected_methods:
        observed = set(by_method[method])
        if observed != expected_cases:
            raise ValueError(
                f"{method}: formal-case mismatch; "
                f"missing={sorted(expected_cases-observed)}, extra={sorted(observed-expected_cases)}"
            )
    clusters = {
        f"{row['sequence']}/{row['capture']}" for row in by_method[FULL].values()
    }
    if len(clusters) != 27:
        raise ValueError(f"found {len(clusters)} capture clusters, expected 27")
    return dict(by_method)


def bootstrap_cluster_case_macro(
    grouped_gains: list[np.ndarray], *, samples: int, rng: np.random.Generator
) -> tuple[float, float]:
    """Resample captures and then their within-capture cases with replacement."""

    clusters = len(grouped_gains)
    draws = np.empty(samples, dtype=np.float64)
    offset = 0
    batch_size = 5_000
    while offset < samples:
        count = min(batch_size, samples - offset)
        values = np.empty(count, dtype=np.float64)
        for draw in range(count):
            selected = rng.integers(0, clusters, size=clusters)
            pieces = [
                grouped_gains[index][
                    rng.integers(
                        0,
                        len(grouped_gains[index]),
                        size=len(grouped_gains[index]),
                    )
                ]
                for index in selected
            ]
            values[draw] = np.concatenate(pieces).mean()
        draws[offset : offset + count] = values
        offset += count
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def signflip_cluster_pvalue(
    grouped_gains: list[np.ndarray], *, samples: int, rng: np.random.Generator
) -> float:
    """Flip whole-capture contributions while retaining case-macro weights."""

    cluster_sums = np.asarray([values.sum() for values in grouped_gains], dtype=np.float64)
    denominator = sum(len(values) for values in grouped_gains)
    observed = abs(float(cluster_sums.sum() / denominator))
    extreme = 0
    remaining = samples
    batch_size = 20_000
    while remaining:
        count = min(batch_size, remaining)
        signs = rng.integers(0, 2, size=(count, len(cluster_sums)), dtype=np.int8)
        null_means = ((signs * 2 - 1) * cluster_sums).sum(axis=1) / denominator
        extreme += int(np.count_nonzero(np.abs(null_means) >= observed))
        remaining -= count
    return float((extreme + 1) / (samples + 1))


def format_number(value: float, unit: str) -> str:
    if unit == "mm":
        return f"{value:.1f}"
    return f"{value:.4f}"


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# EgoHumans formal-90 component paired statistics",
        "",
        f"- Source: `{payload['source_csv']}`",
        f"- Source SHA-256: `{payload['source_csv_sha256']}`",
        f"- Cases / capture clusters: {payload['case_count']} / {payload['capture_cluster_count']}",
        (
            "- Convention: paired improvement is positive when Bridge3R (full) is better; "
            "negative values favour the named comparator."
        ),
        (
            "- Inference: two-sided capture-cluster sign-flip p-values are retrospective "
            "and unadjusted; these Test results must not be used to select a new formal route."
        ),
        "",
    ]
    for comparison in payload["comparisons"]:
        lines.extend(
            [
                f"## Full vs. {comparison['comparator']}",
                "",
                "| Metric | Support (cases/captures) | Full | Comparator | Paired improvement | 95% cluster bootstrap CI | Win/tie/loss | $p_{sf}$ |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in comparison["metrics"]:
            low, high = row["bootstrap_ci95"]
            count = row["case_win_tie_loss"]
            lines.append(
                f"| {row['label']} | {row['case_count']}/{row['capture_cluster_count']} | "
                f"{format_number(row['full_case_macro'], row['unit'])} | "
                f"{format_number(row['comparator_case_macro'], row['unit'])} | "
                f"{format_number(row['mean_paired_improvement'], row['unit'])} | "
                f"[{format_number(low, row['unit'])}, {format_number(high, row['unit'])}] | "
                f"{count['wins']}/{count['ties']}/{count['losses']} | "
                f"{row['signflip_pvalue_two_sided_capture_cluster']:.6f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples < 1 or args.signflip_samples < 1:
        raise ValueError("bootstrap and sign-flip sample counts must be positive")
    source = args.input.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    by_method = load_rows(source)
    full_rows = by_method[FULL]
    bootstrap_rng = np.random.default_rng(args.seed)
    signflip_rng = np.random.default_rng(args.seed + 1)
    paired_csv_rows: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []

    for comparator in COMPARATORS:
        comparator_rows = by_method[comparator]
        metric_rows: list[dict[str, Any]] = []
        for metric, label, unit, direction in METRICS:
            usable: list[tuple[str, str, float, float, float]] = []
            for case_id in sorted(full_rows):
                full_row = full_rows[case_id]
                other_row = comparator_rows[case_id]
                binding_fields = ("capture", "sequence", "angle_stratum", "angle_deg")
                mismatch = [field for field in binding_fields if full_row[field] != other_row[field]]
                if mismatch:
                    raise ValueError(f"{comparator}/{case_id}: binding mismatch {mismatch}")
                full_value = finite(full_row[metric])
                other_value = finite(other_row[metric])
                if full_value is None or other_value is None:
                    continue
                gain = other_value - full_value if direction == "lower" else full_value - other_value
                if abs(gain) <= NUMERICAL_TIE_ATOL:
                    gain = 0.0
                cluster = f"{full_row['sequence']}/{full_row['capture']}"
                usable.append((cluster, case_id, gain, full_value, other_value))
                paired_csv_rows.append(
                    {
                        "comparator": comparator,
                        "metric": metric,
                        "case_id": case_id,
                        "capture_cluster": cluster,
                        "sequence": full_row["sequence"],
                        "capture": full_row["capture"],
                        "angle_stratum": full_row["angle_stratum"],
                        "angle_deg": full_row["angle_deg"],
                        "full_value": full_value,
                        "comparator_value": other_value,
                        "paired_improvement_positive_favours_full": gain,
                    }
                )
            if not usable:
                raise ValueError(f"no finite pairs for {comparator}/{metric}")
            by_cluster: dict[str, list[float]] = defaultdict(list)
            for cluster, _, gain, _, _ in usable:
                by_cluster[cluster].append(gain)
            grouped = [np.asarray(by_cluster[key], dtype=np.float64) for key in sorted(by_cluster)]
            if not grouped:
                raise ValueError(f"{comparator}/{metric}: no finite capture cluster")
            gains = np.asarray([row[2] for row in usable], dtype=np.float64)
            full_values = np.asarray([row[3] for row in usable], dtype=np.float64)
            comparator_values = np.asarray([row[4] for row in usable], dtype=np.float64)
            low, high = bootstrap_cluster_case_macro(
                grouped, samples=args.bootstrap_samples, rng=bootstrap_rng
            )
            wins = int(np.count_nonzero(gains > 0))
            ties = int(np.count_nonzero(gains == 0))
            metric_rows.append(
                {
                    "metric": metric,
                    "label": label,
                    "unit": unit,
                    "direction": direction,
                    "case_count": len(usable),
                    "capture_cluster_count": len(grouped),
                    "full_case_macro": float(full_values.mean()),
                    "comparator_case_macro": float(comparator_values.mean()),
                    "mean_paired_improvement": float(gains.mean()),
                    "median_paired_improvement": float(np.median(gains)),
                    "bootstrap_ci95": [low, high],
                    "case_win_tie_loss": {
                        "wins": wins,
                        "ties": ties,
                        "losses": int(len(gains) - wins - ties),
                    },
                    "signflip_pvalue_two_sided_capture_cluster": signflip_cluster_pvalue(
                        grouped, samples=args.signflip_samples, rng=signflip_rng
                    ),
                }
            )
        comparisons.append({"full": FULL, "comparator": comparator, "metrics": metric_rows})

    payload = {
        "schema_version": SCHEMA,
        "source_csv": str(source),
        "source_csv_sha256": sha256(source),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "protocol": "Bridge3R-EgoHumans-formal90-v1",
        "analysis_scope": (
            "retrospective component attribution on the immutable formal-90 Test; "
            "not a Test-time method-selection rule"
        ),
        "case_count": 90,
        "capture_cluster_count": 27,
        "full_method": FULL,
        "comparators": list(COMPARATORS),
        "paired_improvement_convention": (
            "comparator minus full for lower-is-better metrics; full minus comparator "
            "for higher-is-better metrics; positive always favours full"
        ),
        "uncertainty": {
            "bootstrap": (
                "paired capture-resampling bootstrap with within-capture case resampling; "
                "reported point estimates remain case macro"
            ),
            "bootstrap_samples": args.bootstrap_samples,
            "sign_flip": (
                "two-sided Monte-Carlo capture-cluster sign-flip test of the case-macro "
                "paired improvement"
            ),
            "signflip_samples": args.signflip_samples,
            "pvalues": "unadjusted retrospective component-wise tests",
            "seed": args.seed,
        },
        "comparisons": comparisons,
    }

    for path in (args.output_json, args.output_csv, args.output_md):
        path.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output_json.resolve().write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with args.output_csv.resolve().open("w", newline="", encoding="utf-8") as handle:
        fields = list(paired_csv_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(paired_csv_rows)
    args.output_md.resolve().write_text(markdown(payload) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_json": str(args.output_json.resolve()),
                "output_csv": str(args.output_csv.resolve()),
                "output_md": str(args.output_md.resolve()),
                "comparisons": len(comparisons),
                "paired_csv_rows": len(paired_csv_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
