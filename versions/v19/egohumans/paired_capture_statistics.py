#!/usr/bin/env python3
"""Compute cluster-paired statistics from retained EgoHumans Test artifacts.

This utility reads only the frozen v19 candidate reports and their Test
summary.  It pairs strict Human3R with the frozen causal Bridge3R candidate
on the common evaluator-available cases, then treats each capture as a
cluster.  The primary case-macro point estimate is verified against the
published summary.  Uncertainty uses a capture-resampling paired bootstrap;
the two-sided sign-flip test flips complete capture contributions, rather than
pretending the camera-pair cases within one capture are independent.

No RGB, GT, model checkpoint, or new prediction is read.  The analysis is a
retrospective supplementary summary and never selects a candidate on Test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "Bridge3R-EgoHumans-CS100-capture-paired-statistics-v1"
STRICT = "m0_strict_human3r"
BRIDGE3R = "v19_egohumans_frozen"
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
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=100000)
    parser.add_argument("--signflip-samples", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=20260826)
    return parser.parse_args()


def finite(raw: Any) -> float | None:
    if raw is None:
        return None
    value = float(raw)
    return value if math.isfinite(value) else None


def source_rows(summary: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    """Collect the strict and frozen-candidate row for each retained case."""

    rows: dict[str, dict[str, Any]] = {}
    source_hashes: list[dict[str, str]] = []
    source_paths = summary.get("sources")
    if not isinstance(source_paths, list) or not source_paths:
        raise ValueError("summary has no candidate-report sources")
    for raw_source in source_paths:
        source = Path(str(raw_source)).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        report = json.loads(source.read_text(encoding="utf-8"))
        source_hashes.append({"path": str(source), "sha256": sha256(source)})
        for field, name, role in (
            ("reference_rows", STRICT, "strict"),
            ("rows", BRIDGE3R, "bridge3r"),
        ):
            for row in report.get(field, []):
                observed = str(row.get("method" if field == "reference_rows" else "candidate", ""))
                if observed != name:
                    continue
                case_id = str(row.get("case_id", ""))
                capture = str(row.get("capture", ""))
                sequence = str(row.get("sequence", ""))
                if not case_id or not capture or not sequence:
                    raise ValueError(f"malformed {role} row in {source}")
                entry = rows.setdefault(
                    case_id,
                    {
                        "case_id": case_id,
                        "cluster": f"{sequence}/{capture}",
                        "sequence": sequence,
                        "capture": capture,
                        "angle_stratum": str(row.get("angle_stratum", "")),
                    },
                )
                if role in entry:
                    raise ValueError(f"duplicate {role} row for {case_id}")
                entry[role] = row.get("metrics", {})
    absent = [
        case_id for case_id, entry in rows.items()
        if "strict" not in entry or "bridge3r" not in entry
    ]
    if absent:
        raise ValueError(f"incomplete method pair for cases: {sorted(absent)}")
    return rows, source_hashes


def bootstrap_cluster_case_macro(
    grouped_gains: list[np.ndarray], *, samples: int, rng: np.random.Generator
) -> tuple[float, float]:
    """Resample captures, then their within-capture case rows, with replacement."""

    clusters = len(grouped_gains)
    results = np.empty(samples, dtype=np.float64)
    batch_size = 5000
    offset = 0
    while offset < samples:
        count = min(batch_size, samples - offset)
        values = np.empty(count, dtype=np.float64)
        for draw in range(count):
            selected = rng.integers(0, clusters, size=clusters)
            pieces = [
                grouped_gains[index][
                    rng.integers(0, len(grouped_gains[index]), size=len(grouped_gains[index]))
                ]
                for index in selected
            ]
            values[draw] = np.concatenate(pieces).mean()
        results[offset : offset + count] = values
        offset += count
    return float(np.percentile(results, 2.5)), float(np.percentile(results, 97.5))


def signflip_cluster_pvalue(
    grouped_gains: list[np.ndarray], *, samples: int, rng: np.random.Generator
) -> float:
    """Sign-flip complete capture contributions, retaining case-macro weights."""

    cluster_sums = np.asarray([value.sum() for value in grouped_gains], dtype=np.float64)
    denominator = sum(len(value) for value in grouped_gains)
    observed = abs(float(cluster_sums.sum() / denominator))
    extreme = 0
    remaining = samples
    batch_size = 20000
    while remaining:
        count = min(batch_size, remaining)
        signs = rng.integers(0, 2, size=(count, len(cluster_sums)), dtype=np.int8)
        null_means = ((signs * 2 - 1) * cluster_sums).sum(axis=1) / denominator
        extreme += int(np.count_nonzero(np.abs(null_means) >= observed))
        remaining -= count
    return float((extreme + 1) / (samples + 1))


def number(value: float, unit: str) -> str:
    return f"{value:.1f}" if unit == "mm" else f"{value:.3f}"


def render_tex(rows: list[dict[str, Any]]) -> str:
    lines = [
        "% Generated by paired_capture_statistics.py; frozen Test artifact.",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Metric & Mean paired gain $\\uparrow$ & 95\\% CI & Median gain & Case win/tie/loss & $p_{\\mathrm{sf}}$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        low, high = row["bootstrap_ci95"]
        wins = row["case_win_tie_loss"]
        pvalue = row["signflip_pvalue_two_sided_capture_cluster"]
        ptext = "<0.000001" if pvalue < 0.000001 else f"{pvalue:.6f}"
        lines.append(
            f"{row['label']} & {number(row['mean_improvement'], row['unit'])} & "
            f"[{number(low, row['unit'])}, {number(high, row['unit'])}] & "
            f"{number(row['median_improvement'], row['unit'])} & "
            f"{wins['wins']}/{wins['ties']}/{wins['losses']} & {ptext} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples < 1 or args.signflip_samples < 1:
        raise ValueError("bootstrap and sign-flip sample counts must be positive")
    summary_path = args.summary.resolve()
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema_version") != "Movie3R-v19-EgoHumans-CS100-summary-v1":
        raise ValueError("unexpected EgoHumans Test summary schema")
    if summary.get("split") != "test" or summary.get("primary") != BRIDGE3R:
        raise ValueError("summary is not the frozen Bridge3R Test result")
    paired, source_hashes = source_rows(summary)
    bootstrap_rng = np.random.default_rng(args.seed)
    signflip_rng = np.random.default_rng(args.seed + 1)
    report_rows: list[dict[str, Any]] = []
    used_case_ids: dict[str, list[str]] = {}
    for metric, label, unit in PRIMARY_METRICS:
        usable = []
        for case_id, entry in paired.items():
            strict = finite(entry["strict"].get(metric))
            bridge = finite(entry["bridge3r"].get(metric))
            if strict is None or bridge is None:
                continue
            gain = strict - bridge if metric in LOWER_IS_BETTER else bridge - strict
            usable.append((entry["cluster"], case_id, gain, strict, bridge))
        if not usable:
            raise ValueError(f"no usable paired values for {metric}")
        strict_values = np.asarray([row[3] for row in usable], dtype=np.float64)
        bridge_values = np.asarray([row[4] for row in usable], dtype=np.float64)
        gains = np.asarray([row[2] for row in usable], dtype=np.float64)
        by_cluster: dict[str, list[float]] = defaultdict(list)
        for cluster, _, gain, _, _ in usable:
            by_cluster[cluster].append(gain)
        grouped = [np.asarray(by_cluster[key], dtype=np.float64) for key in sorted(by_cluster)]
        expected = summary["methods"][BRIDGE3R]["case_macro"].get(metric)
        strict_expected = summary["methods"][STRICT]["case_macro"].get(metric)
        if not math.isclose(float(expected), float(bridge_values.mean()), abs_tol=1e-9):
            raise ValueError(f"Bridge3R case macro does not reproduce summary for {metric}")
        if not math.isclose(float(strict_expected), float(strict_values.mean()), abs_tol=1e-9):
            raise ValueError(f"strict case macro does not reproduce summary for {metric}")
        low, high = bootstrap_cluster_case_macro(
            grouped, samples=args.bootstrap_samples, rng=bootstrap_rng
        )
        wins = int(np.count_nonzero(gains > 0))
        ties = int(np.count_nonzero(gains == 0))
        report_rows.append(
            {
                "metric": metric,
                "label": label,
                "unit": unit,
                "direction": "higher paired gain is better",
                "case_count": len(usable),
                "capture_cluster_count": len(grouped),
                "strict_case_macro": float(strict_values.mean()),
                "bridge3r_case_macro": float(bridge_values.mean()),
                "mean_improvement": float(gains.mean()),
                "bootstrap_ci95": [low, high],
                "median_improvement": float(np.median(gains)),
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
        used_case_ids[metric] = sorted(row[1] for row in usable)
    payload = {
        "schema_version": SCHEMA,
        "summary": str(summary_path),
        "summary_sha256": sha256(summary_path),
        "source_reports": source_hashes,
        "split": "test",
        "primary_aggregation": "case_macro",
        "strict_method": STRICT,
        "bridge3r_method": BRIDGE3R,
        "analysis_scope": (
            "retrospective paired statistics for the v014 EgoHumans main-table "
            "metrics; not a Test-time method-selection rule"
        ),
        "uncertainty": {
            "bootstrap": "paired capture-resampling bootstrap with within-capture case resampling",
            "bootstrap_samples": args.bootstrap_samples,
            "sign_flip": "two-sided capture-cluster sign-flip test of the case-macro paired gain",
            "signflip_samples": args.signflip_samples,
            "seed": args.seed,
        },
        "used_case_ids": used_case_ids,
        "rows": report_rows,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(".tex").write_text(render_tex(report_rows), encoding="utf-8")
    print(json.dumps({"output": str(output), "rows": report_rows}, indent=2))


if __name__ == "__main__":
    main()
