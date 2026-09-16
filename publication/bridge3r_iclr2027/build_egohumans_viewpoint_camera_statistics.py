#!/usr/bin/env python3
"""Relate EgoHumans viewpoint change to Shot3R gains with capture clustering.

The analysis joins two retained, immutable per-case CSV files.  It uses the
exact camera angle stored by the component audit and the final Human3R/Shot3R
metrics stored by the formal test summary.  No prediction or ground truth is
opened.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr


ERROR_METRICS = (
    "W-MPJPE_mm",
    "WA-MPJPE_mm",
    "Boundary_camera_R_deg",
    "Boundary_camera_t_m",
)
HIGHER_METRICS = ("IDF1",)
METRICS = ERROR_METRICS + HIGHER_METRICS
BASELINE = "m0_strict_human3r"
METHOD = "v19_egohumans_frozen"
STRATA = ("small", "medium", "large", "extreme")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-csv", type=Path, required=True)
    parser.add_argument("--angle-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260908)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str) -> float:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def percentile_interval(values: np.ndarray) -> list[float | None]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return [None, None]
    return [float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))]


def main() -> None:
    args = parse_args()
    if args.bootstrap_draws < 1000:
        raise ValueError("bootstrap-draws must be at least 1000")
    formal = read_csv(args.case_csv)
    angle_rows = read_csv(args.angle_csv)
    angle = {
        row["case_id"]: {
            "angle_deg": number(row, "angle_deg"),
            "sequence": row["sequence"],
            "capture": row["capture"],
            "angle_stratum": row["angle_stratum"],
        }
        for row in angle_rows
        if row.get("method") == "Strict Human3R"
    }
    by_case: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in formal:
        if row.get("method") in {BASELINE, METHOD}:
            by_case[row["case_id"]][row["method"]] = row
    if len(by_case) != 90 or set(by_case) != set(angle):
        raise ValueError(
            f"expected the same 90 formal cases; metrics={len(by_case)}, angles={len(angle)}"
        )

    cases: list[dict[str, Any]] = []
    for case_id in sorted(by_case):
        pair = by_case[case_id]
        if set(pair) != {BASELINE, METHOD}:
            raise ValueError(f"incomplete method pair for {case_id}")
        metadata = angle[case_id]
        cluster = f"{metadata['sequence']}/{metadata['capture']}"
        row: dict[str, Any] = {
            "case_id": case_id,
            "capture_cluster": cluster,
            "angle_stratum": metadata["angle_stratum"],
            "angle_deg": metadata["angle_deg"],
        }
        for metric in METRICS:
            baseline = number(pair[BASELINE], metric)
            method = number(pair[METHOD], metric)
            row[f"human3r_{metric}"] = baseline
            row[f"shot3r_{metric}"] = method
            row[f"gain_{metric}"] = (
                baseline - method if metric in ERROR_METRICS else method - baseline
            )
        cases.append(row)

    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cases:
        clusters[row["capture_cluster"]].append(row)
    keys = sorted(clusters)
    if len(keys) != 27:
        raise ValueError(f"expected 27 capture clusters, found {len(keys)}")

    correlations: dict[str, Any] = {}
    rng = np.random.default_rng(args.seed)
    for metric_index, metric in enumerate(METRICS):
        usable = [
            row for row in cases
            if math.isfinite(row["angle_deg"]) and math.isfinite(row[f"gain_{metric}"])
        ]
        observed = float(spearmanr(
            [row["angle_deg"] for row in usable],
            [row[f"gain_{metric}"] for row in usable],
        ).statistic)
        draws = np.empty(args.bootstrap_draws, dtype=np.float64)
        metric_rng = np.random.default_rng(rng.integers(0, 2**63 - 1) + metric_index)
        for draw in range(args.bootstrap_draws):
            selected = metric_rng.integers(0, len(keys), size=len(keys))
            sampled: list[dict[str, Any]] = []
            for cluster_index in selected:
                rows = clusters[keys[int(cluster_index)]]
                local = metric_rng.integers(0, len(rows), size=len(rows))
                sampled.extend(rows[int(index)] for index in local)
            x = np.asarray([row["angle_deg"] for row in sampled], dtype=np.float64)
            y = np.asarray([row[f"gain_{metric}"] for row in sampled], dtype=np.float64)
            mask = np.isfinite(x) & np.isfinite(y)
            draws[draw] = (
                float(spearmanr(x[mask], y[mask]).statistic)
                if mask.sum() >= 3 and len(np.unique(x[mask])) >= 2
                else float("nan")
            )
        correlations[metric] = {
            "case_count": len(usable),
            "capture_count": len({row["capture_cluster"] for row in usable}),
            "spearman_rho": observed,
            "capture_cluster_two_stage_bootstrap_ci95": percentile_interval(draws),
            "bootstrap_finite_draws": int(np.isfinite(draws).sum()),
        }

    strata: dict[str, Any] = {}
    for stratum in STRATA:
        selected = [row for row in cases if row["angle_stratum"] == stratum]
        strata[stratum] = {
            "case_count": len(selected),
            "capture_count": len({row["capture_cluster"] for row in selected}),
            "mean_angle_deg": float(np.mean([row["angle_deg"] for row in selected])),
            "metrics": {},
        }
        for metric in METRICS:
            base = np.asarray([row[f"human3r_{metric}"] for row in selected], dtype=float)
            ours = np.asarray([row[f"shot3r_{metric}"] for row in selected], dtype=float)
            gain = np.asarray([row[f"gain_{metric}"] for row in selected], dtype=float)
            strata[stratum]["metrics"][metric] = {
                "human3r_mean": float(np.nanmean(base)),
                "shot3r_mean": float(np.nanmean(ours)),
                "gain_mean": float(np.nanmean(gain)),
                "support": int(np.isfinite(gain).sum()),
            }

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "Shot3R-EgoHumans-viewpoint-camera-statistics-v1",
        "definition": (
            "Positive gain favours Shot3R; errors use Human3R minus Shot3R, "
            "IDF1 uses Shot3R minus Human3R. Correlation is case-level Spearman "
            "with a two-stage capture-cluster bootstrap interval."
        ),
        "inputs": {
            "case_csv": str(args.case_csv.resolve()),
            "case_csv_sha256": sha256(args.case_csv.resolve()),
            "angle_csv": str(args.angle_csv.resolve()),
            "angle_csv_sha256": sha256(args.angle_csv.resolve()),
        },
        "bootstrap": {"draws": args.bootstrap_draws, "seed": args.seed},
        "case_count": len(cases),
        "capture_count": len(keys),
        "correlations": correlations,
        "strata": strata,
    }
    atomic_json(output / "viewpoint_camera_statistics.json", payload)

    fields = list(cases[0])
    with (output / "viewpoint_camera_cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(cases)
    with (output / "viewpoint_camera_correlations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "cases", "captures", "rho", "ci95_low", "ci95_high"])
        for metric in METRICS:
            row = correlations[metric]
            writer.writerow([
                metric, row["case_count"], row["capture_count"], row["spearman_rho"],
                *row["capture_cluster_two_stage_bootstrap_ci95"],
            ])
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
