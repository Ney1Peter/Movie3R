#!/usr/bin/env python3
"""Build availability-aware paired OnlineHMR/BRIDGE3R paper evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT = Path(__file__).resolve()
WORKSPACE = SCRIPT.parents[4]
MOVIE3R = WORKSPACE / "Movie3R"
SCHEMA = "Bridge3R-OnlineHMR-paired-comparison-v1"


@dataclass(frozen=True)
class Spec:
    key: str
    display: str
    expected: int
    internal_csv: Path
    method_column: str
    bridge_key: str
    unit_column: str
    camera_online: str
    camera_internal: str
    camera_label: str
    primary_scope: str


SPECS = {
    "egobody": Spec(
        "egobody", "EgoBody", 129,
        MOVIE3R / "output/v20_egobody/formal/test/aggregate/case_metrics.csv",
        "name", "v19_ungated_translation_b050", "recording",
        "ATE-Sim3_m", "ATE_Sim3_m", "ATE-Sim3", "independent_unit_macro",
    ),
    "egohumans": Spec(
        "egohumans", "EgoHumans", 90,
        MOVIE3R / "output/v19_egohumans/test/summary/case_metrics.csv",
        "method", "v19_egohumans_frozen", "capture",
        "ATE-SE3_m", "ATE_SE3_m", "ATE-SE3", "case_macro",
    ),
    "harmony4d": Spec(
        "harmony4d", "Harmony4D", 88,
        MOVIE3R / "output/v17_harmony4d/unified_half_translation_audit/paper/case_metrics.csv",
        "method", "bridge3r_unified_half_translation", "capture",
        "ATE-Sim3_m", "ATE_Sim3_m", "ATE-Sim3", "case_macro",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def cluster_bootstrap(
    values: list[tuple[str, float]], *, samples: int, seed: int
) -> tuple[float, float, float]:
    point = float(np.mean([value for _, value in values]))
    buckets: dict[str, list[float]] = {}
    for unit, value in values:
        buckets.setdefault(unit, []).append(value)
    units = sorted(buckets)
    rng = np.random.default_rng(seed)
    distribution = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected = rng.integers(0, len(units), size=len(units))
        draw = [value for unit_index in selected for value in buckets[units[unit_index]]]
        distribution[index] = np.mean(draw)
    low, high = np.quantile(distribution, [0.025, 0.975]).tolist()
    return point, float(low), float(high)


def paired_metric(
    pairs: list[tuple[dict[str, str], dict[str, str]]],
    *,
    online_column: str,
    bridge_column: str,
    higher_is_better: bool,
    samples: int,
    seed: int,
    conditional: bool,
) -> dict[str, Any]:
    differences: list[tuple[str, float]] = []
    online_values, bridge_values = [], []
    for online, bridge in pairs:
        online_value = number(online.get(online_column))
        bridge_value = number(bridge.get(bridge_column))
        if not np.isfinite(online_value) or not np.isfinite(bridge_value):
            continue
        benefit = bridge_value - online_value if higher_is_better else online_value - bridge_value
        differences.append((online["unit"], float(benefit)))
        online_values.append(float(online_value))
        bridge_values.append(float(bridge_value))
    if not differences:
        return {
            "estimate_bridge3r_benefit": None,
            "ci95_low": None,
            "ci95_high": None,
            "available_cases": 0,
            "available_units": 0,
            "conditional_on_onlinehmr_availability": conditional,
        }
    point, low, high = cluster_bootstrap(differences, samples=samples, seed=seed)
    return {
        "estimate_bridge3r_benefit": point,
        "ci95_low": low,
        "ci95_high": high,
        "onlinehmr_mean_on_shared_support": float(np.mean(online_values)),
        "bridge3r_mean_on_shared_support": float(np.mean(bridge_values)),
        "available_cases": len(differences),
        "available_units": len({unit for unit, _ in differences}),
        "conditional_on_onlinehmr_availability": conditional,
        "direction": "positive favours BRIDGE3R",
        "bootstrap_unit": "recording/capture",
        "bootstrap_samples": samples,
        "seed": seed,
    }


def f(value: Any, digits: int = 1) -> str:
    result = number(value)
    return "\\textemdash{}" if not np.isfinite(result) else f"{result:.{digits}f}"


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(value, encoding="utf-8")
    os.replace(partial, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=tuple(SPECS), default=list(SPECS))
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--allow-subset", action="store_true")
    args = parser.parse_args()
    if args.bootstrap_samples < 1000:
        raise ValueError("--bootstrap-samples must be at least 1000")

    root = args.online_root.resolve()
    output = args.output.resolve()
    results: dict[str, Any] = {}
    latex_rows = []
    csv_rows: list[dict[str, Any]] = []
    for dataset_index, key in enumerate(args.datasets):
        spec = SPECS[key]
        online_csv = root / key / "onlinehmr_case_metrics.csv"
        aggregate_json = root / key / "onlinehmr_aggregate.json"
        online_rows = read_csv(online_csv)
        internal_all = read_csv(spec.internal_csv)
        internal_rows = [row for row in internal_all if row.get(spec.method_column) == spec.bridge_key]
        online_by_id = {row["case_id"]: row for row in online_rows}
        internal_by_id = {row["case_id"]: row for row in internal_rows}
        if len(online_by_id) != len(online_rows) or len(internal_by_id) != len(internal_rows):
            raise ValueError(f"duplicate case IDs in {key}")
        if not set(online_by_id).issubset(internal_by_id):
            raise ValueError(f"OnlineHMR {key} cases are not a subset of BRIDGE3R cases")
        if not args.allow_subset and (
            len(online_by_id) != spec.expected or set(online_by_id) != set(internal_by_id)
        ):
            raise ValueError(f"{key} is not the complete fixed {spec.expected}-case protocol")
        pairs = []
        for case_id in sorted(online_by_id):
            online, bridge = online_by_id[case_id], internal_by_id[case_id]
            if online.get("unit") != bridge.get(spec.unit_column):
                raise ValueError(f"independent-unit mismatch for {case_id}")
            pairs.append((online, bridge))
        metrics = {
            "Coverage": ("Coverage", "Coverage", True, False),
            "IDF1": ("IDF1", "IDF1", True, False),
            "W-MPJPE_mm": ("W-MPJPE_mm", "W-MPJPE_mm", False, True),
            "WA-MPJPE_mm": ("WA-MPJPE_mm", "WA-MPJPE_mm", False, True),
            spec.camera_label: (spec.camera_online, spec.camera_internal, False, True),
        }
        comparisons = {
            name: paired_metric(
                pairs,
                online_column=online_column,
                bridge_column=bridge_column,
                higher_is_better=higher,
                samples=args.bootstrap_samples,
                seed=args.seed + dataset_index,
                conditional=conditional,
            )
            for name, (online_column, bridge_column, higher, conditional) in metrics.items()
        }
        angle_results = {}
        for label in sorted({online["angle_stratum"] for online, _ in pairs}):
            subset = [pair for pair in pairs if pair[0]["angle_stratum"] == label]
            angle_results[label] = {
                name: paired_metric(
                    subset,
                    online_column=online_column,
                    bridge_column=bridge_column,
                    higher_is_better=higher,
                    samples=args.bootstrap_samples,
                    seed=args.seed + dataset_index,
                    conditional=conditional,
                )
                for name, (online_column, bridge_column, higher, conditional) in metrics.items()
            }
        aggregate = json.loads(aggregate_json.read_text(encoding="utf-8"))
        if aggregate.get("observed_cases") != len(online_rows):
            raise ValueError(f"aggregate/case CSV mismatch for {key}")
        primary = aggregate[spec.primary_scope]
        case_scope = aggregate["case_macro"]
        successful = int(case_scope["successful_inference_cases"])
        zero_coverage = sum(number(row.get("Coverage")) == 0.0 for row in online_rows)
        results[key] = {
            "display": spec.display,
            "expected_cases": spec.expected,
            "observed_cases": len(pairs),
            "successful_inference_cases": successful,
            "zero_coverage_cases": zero_coverage,
            "primary_scope": spec.primary_scope,
            "camera_metric": spec.camera_label,
            "onlinehmr_primary": primary,
            "onlinehmr_case_availability": {
                "W": int(case_scope["W_available_cases"]),
                "WA": int(case_scope["WA_available_cases"]),
                "camera": int(case_scope["camera_reportable_cases"]),
            },
            "paired_bridge3r_benefit": comparisons,
            "paired_by_angle": angle_results,
            "sources": {
                "online_case_csv": str(online_csv),
                "online_case_csv_sha256": sha256(online_csv),
                "online_aggregate": str(aggregate_json),
                "online_aggregate_sha256": sha256(aggregate_json),
                "bridge3r_case_csv": str(spec.internal_csv),
                "bridge3r_case_csv_sha256": sha256(spec.internal_csv),
                "bridge3r_method_key": spec.bridge_key,
            },
        }
        for name, value in comparisons.items():
            csv_rows.append({"dataset": key, "angle_stratum": "all", "metric": name, **value})
        for label, values in angle_results.items():
            for name, value in values.items():
                csv_rows.append({"dataset": key, "angle_stratum": label, "metric": name, **value})
        latex_rows.append(
            f"{spec.display} & OnlineHMR & {successful}/{len(pairs)} & "
            f"{f(primary.get('W-MPJPE_mm'))} & {f(primary.get('WA-MPJPE_mm'))} & "
            f"{f(primary.get(spec.camera_online), 3)} & {f(primary.get('IDF1'), 3)} & "
            f"{f(primary.get('Coverage'), 3)} & "
            f"{int(case_scope['W_available_cases'])}/{int(case_scope['WA_available_cases'])} & "
            f"{int(case_scope['camera_reportable_cases'])} \\\\"
        )

    payload = {
        "schema_version": SCHEMA,
        "selection_depends_on_onlinehmr_results": False,
        "full_denominator_metrics": ["Coverage", "IDF1"],
        "conditional_metrics": ["W-MPJPE_mm", "WA-MPJPE_mm", "camera trajectory error"],
        "datasets": results,
    }
    atomic_text(output / "onlinehmr_bridge3r_comparison.json", json.dumps(payload, indent=2, sort_keys=True) + "\n")
    fields = [
        "dataset", "angle_stratum", "metric", "estimate_bridge3r_benefit",
        "ci95_low", "ci95_high", "onlinehmr_mean_on_shared_support",
        "bridge3r_mean_on_shared_support", "available_cases", "available_units",
        "conditional_on_onlinehmr_availability", "direction", "bootstrap_unit",
        "bootstrap_samples", "seed",
    ]
    output.mkdir(parents=True, exist_ok=True)
    with (output / "onlinehmr_bridge3r_paired.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)
    latex = """% Generated availability-aware OnlineHMR evidence. Do not edit by hand.
\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{llrrrrrrlr}
\\toprule
Dataset & Method & Completed & W $\\downarrow$ & WA $\\downarrow$ & ATE$^{*}$ $\\downarrow$ & IDF1 $\\uparrow$ & Coverage $\\uparrow$ & $N_W/N_{WA}$ & $N_{\\mathrm{cam}}$ \\\\
\\midrule
""" + "\n".join(latex_rows) + """
\\bottomrule
\\end{tabular}}
\\parbox{0.99\\textwidth}{\\footnotesize W, WA, and ATE are conditional errors and must be read with $N_W/N_{WA}$, $N_{\\mathrm{cam}}$, and Coverage. W and WA are in mm. ATE$^{*}$ is Sim(3)-aligned for EgoBody/Harmony4D and SE(3)-aligned for EgoHumans. Completed counts native inference completion; zero-match cases remain in the fixed Coverage and IDF1 denominators.}
"""
    atomic_text(output / "onlinehmr_public_reference.tex", latex)
    print(json.dumps({"output": str(output), "datasets": list(results)}, indent=2))


if __name__ == "__main__":
    main()
