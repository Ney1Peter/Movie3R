#!/usr/bin/env python3
"""Compute a frozen, recording-paired EgoBody Test comparison.

The formal EgoBody protocol aggregates the three camera-baseline cases in a
recording before averaging across recordings.  This utility consumes that
retained ``recording_metrics.csv`` artifact only: it neither reads RGB, ground
truth, nor re-evaluates a method.  It reports a paired comparison of the
pre-Test-selected causal Bridge3R candidate against strict Human3R.

For every metric, an improvement is positive: ``strict - Bridge3R`` for a
lower-is-better metric and ``Bridge3R - strict`` for a higher-is-better metric.
The output includes a paired percentile-bootstrap confidence interval, median
recording improvement, per-recording win/tie/loss accounting, and a
two-sided Monte-Carlo sign-flip randomisation test.  The primary metrics are
the five quantities displayed for EgoBody in the v011 main table, fixed here
before this retrospective statistical analysis is run.  The statistics are
therefore supplementary paired evidence, not a new Test-time selection rule.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "Bridge3R-EgoBody-CS150-recording-paired-statistics-v1"
STRICT = "m0_strict_human3r"
BRIDGE3R = "v19_ungated_translation_b050"

# These are exactly the EgoBody columns in the v011 multi-person main table,
# in the same order.  They are fixed before this post-hoc statistical script
# is run, rather than selected from the larger retained metric ledger.
PRIMARY_METRICS: tuple[tuple[str, str, str], ...] = (
    ("W-MPJPE_mm", "W-MPJPE", "mm"),
    ("WA-MPJPE_mm", "WA-MPJPE", "mm"),
    ("ATE_Sim3_m", "ATE-Sim3", "m"),
    ("IDF1", "IDF1", "unitless"),
    ("Coverage", "Coverage", "unitless"),
)
LOWER_IS_BETTER = {"W-MPJPE_mm", "WA-MPJPE_mm", "ATE_Sim3_m"}
HIGHER_IS_BETTER = {"IDF1", "Coverage"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=100000)
    parser.add_argument("--signflip-samples", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=20260826)
    return parser.parse_args()


def read_rows(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    """Return method -> recording -> metric dictionary and validate pairing."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("recording metric CSV has no header")
        required = {"name", "recording", *(item[0] for item in PRIMARY_METRICS)}
        absent = required - set(reader.fieldnames)
        if absent:
            raise ValueError(f"recording metric CSV lacks columns: {sorted(absent)}")
        result: dict[str, dict[str, dict[str, float]]] = {}
        for raw in reader:
            method = str(raw["name"])
            recording = str(raw["recording"])
            if not method or not recording:
                raise ValueError("encountered blank method or recording")
            metrics = {key: float(raw[key]) for key, _, _ in PRIMARY_METRICS}
            if not all(math.isfinite(value) for value in metrics.values()):
                raise ValueError(f"non-finite selected metric for {method}/{recording}")
            by_recording = result.setdefault(method, {})
            if recording in by_recording:
                raise ValueError(f"duplicate method/recording row: {method}/{recording}")
            by_recording[recording] = metrics

    missing = [name for name in (STRICT, BRIDGE3R) if name not in result]
    if missing:
        raise ValueError(f"required formal methods are absent: {missing}")
    strict_recordings = set(result[STRICT])
    bridge_recordings = set(result[BRIDGE3R])
    if strict_recordings != bridge_recordings:
        raise ValueError(
            "strict/Bridge3R recording sets differ: "
            f"strict_only={sorted(strict_recordings - bridge_recordings)}, "
            f"bridge_only={sorted(bridge_recordings - strict_recordings)}"
        )
    if not strict_recordings:
        raise ValueError("no paired recordings")
    return result


def paired_bootstrap(
    gains: np.ndarray, *, samples: int, rng: np.random.Generator
) -> tuple[float, float]:
    draws = rng.integers(0, len(gains), size=(samples, len(gains)))
    means = gains[draws].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def signflip_pvalue(
    gains: np.ndarray, *, samples: int, rng: np.random.Generator
) -> float:
    """Two-sided Monte-Carlo sign-flip p-value for mean paired gain.

    Batching avoids materialising a large samples-by-recordings matrix.  The
    +1 correction makes the result a valid randomisation-test estimate even
    when no sampled null draw is as extreme as the observation.
    """

    observed = abs(float(gains.mean()))
    extreme = 0
    remaining = samples
    batch_size = 20000
    while remaining:
        count = min(batch_size, remaining)
        signs = rng.integers(0, 2, size=(count, len(gains)), dtype=np.int8)
        null_means = ((signs * 2 - 1) * gains).mean(axis=1)
        extreme += int(np.count_nonzero(np.abs(null_means) >= observed))
        remaining -= count
    return float((extreme + 1) / (samples + 1))


def format_value(value: float, unit: str, digits: int = 3) -> str:
    if unit == "mm":
        return f"{value:.1f}"
    return f"{value:.{digits}f}"


def render_tex(rows: list[dict[str, Any]]) -> str:
    lines = [
        "% Generated by paired_recording_statistics.py; frozen Test artifact.",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Metric & Mean paired gain $\\uparrow$ & 95\\% CI & Median gain & Win/tie/loss & $p_{\\mathrm{sf}}$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        unit = row["unit"]
        mean = format_value(row["mean_improvement"], unit)
        low, high = row["bootstrap_ci95"]
        ci = f"[{format_value(low, unit)}, {format_value(high, unit)}]"
        median = format_value(row["median_improvement"], unit)
        wins = row["win_tie_loss"]
        pvalue = row["signflip_pvalue_two_sided"]
        ptext = "<0.000001" if pvalue < 0.000001 else f"{pvalue:.6f}"
        lines.append(
            f"{row['label']} & {mean} & {ci} & {median} & "
            f"{wins['wins']}/{wins['ties']}/{wins['losses']} & {ptext} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples < 1 or args.signflip_samples < 1:
        raise ValueError("bootstrap and sign-flip samples must be positive")
    source = args.recording_metrics.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    all_rows = read_rows(source)
    recordings = sorted(all_rows[STRICT])
    bootstrap_rng = np.random.default_rng(args.seed)
    signflip_rng = np.random.default_rng(args.seed + 1)
    report_rows: list[dict[str, Any]] = []
    for metric, label, unit in PRIMARY_METRICS:
        strict = np.asarray([all_rows[STRICT][key][metric] for key in recordings])
        bridge = np.asarray([all_rows[BRIDGE3R][key][metric] for key in recordings])
        if metric in LOWER_IS_BETTER:
            gains = strict - bridge
        elif metric in HIGHER_IS_BETTER:
            gains = bridge - strict
        else:  # Defensive guard when the fixed metric list changes.
            raise ValueError(f"unknown direction for {metric}")
        low, high = paired_bootstrap(
            gains, samples=args.bootstrap_samples, rng=bootstrap_rng
        )
        wins = int(np.count_nonzero(gains > 0))
        ties = int(np.count_nonzero(gains == 0))
        report_rows.append(
            {
                "metric": metric,
                "label": label,
                "unit": unit,
                "direction": "higher paired gain is better",
                "recording_count": len(recordings),
                "strict_mean": float(strict.mean()),
                "bridge3r_mean": float(bridge.mean()),
                "mean_improvement": float(gains.mean()),
                "bootstrap_ci95": [low, high],
                "median_improvement": float(np.median(gains)),
                "win_tie_loss": {
                    "wins": wins,
                    "ties": ties,
                    "losses": int(len(gains) - wins - ties),
                },
                "signflip_pvalue_two_sided": signflip_pvalue(
                    gains, samples=args.signflip_samples, rng=signflip_rng
                ),
            }
        )

    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "source_recording_metrics": str(source),
        "source_recording_metrics_sha256": sha256(source),
        "split": "test",
        "aggregation_unit": "recording",
        "recordings": recordings,
        "recording_count": len(recordings),
        "strict_method": STRICT,
        "bridge3r_method": BRIDGE3R,
        "primary_metrics": [item[0] for item in PRIMARY_METRICS],
        "analysis_scope": (
            "retrospective paired statistics over the v011 EgoBody main-table "
            "metrics; not a Test-time method-selection rule"
        ),
        "bootstrap": {
            "method": "paired percentile bootstrap of recording means",
            "samples": args.bootstrap_samples,
            "seed": args.seed,
        },
        "sign_flip": {
            "method": "two-sided Monte-Carlo sign-flip randomisation test of mean paired gain",
            "samples": args.signflip_samples,
            "seed": args.seed + 1,
        },
        "rows": report_rows,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    output.with_suffix(".tex").write_text(render_tex(report_rows), encoding="utf-8")
    print(json.dumps({"output": str(output), "rows": report_rows}, indent=2))


if __name__ == "__main__":
    main()
