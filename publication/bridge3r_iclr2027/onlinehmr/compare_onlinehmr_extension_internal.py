#!/usr/bin/env python3
"""Compare OnlineHMR extension metrics with internal methods on shared cases."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SINGLE_METRICS = {
    "PA-MPJPE_mm": "pa_mpjpe_body12_mm",
    "Anchor-MPJPE_mm": "first_shot_anchor_mpjpe_body12_mm",
    "Anchor-root_mm": "first_shot_anchor_root_error_mm",
    "Orientation_deg": "first_shot_anchor_orientation_proxy_deg",
    "Seam-root_mm": "seam_root_excess_mm",
    "Seam-orientation_deg": "seam_orientation_excess_deg",
    "Camera-rotation_deg": "post_camera_relative_rotation_deg",
    "Camera-translation_m": "post_camera_relative_translation_m",
}
MULTICUT_METRICS = {
    "PA-MPJPE_mm": "pa_mpjpe_body12_mm",
    "Anchor-MPJPE_mm": "first_shot_anchor_mpjpe_body12_mm",
    "Anchor-root_mm": "first_shot_anchor_root_error_mm",
    "Orientation_deg": "first_shot_anchor_orientation_proxy_deg",
    "Seam-root_mm": "mean_boundary_seam_root_excess_mm",
    "Seam-orientation_deg": "mean_boundary_seam_orientation_excess_deg",
    "Camera-rotation_deg": "post_first_cut_camera_relative_rotation_deg",
    "Camera-translation_m": "post_first_cut_camera_relative_translation_m",
    "Boundary-camera-rotation_deg": "mean_boundary_camera_relative_rotation_deg",
    "Boundary-camera-translation_m": "mean_boundary_camera_relative_translation_m",
}
HIGHER_IS_BETTER = {"Coverage", "IDF1"}
HARMONY_METRICS = {
    "W-MPJPE_mm": ("multi_thumbs_named_provisional", "w_mpjpe_mm", "mean"),
    "WA-MPJPE_mm": ("multi_thumbs_named_provisional", "wa_mpjpe_mm", "mean"),
    "ATE-Sim3_m": ("multi_thumbs_named_provisional", "ate_sim3_m", "mean"),
    "IDF1": ("identity", "idf1"),
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def parse_method(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--method must be LABEL=METHOD_KEY")
    label, key = value.split("=", 1)
    if not label or not key:
        raise argparse.ArgumentTypeError("--method must be LABEL=METHOD_KEY")
    return label, key


def nested(value: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def load_internal(directory: Path, pattern: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob(pattern)):
        payload = read_json(path)
        case_id = payload.get("case_id")
        if case_id is None:
            continue
        if case_id in result:
            raise ValueError(f"duplicate internal case_id: {case_id}")
        result[str(case_id)] = payload
    return result


def mean_ci_clustered(
    rows: list[dict[str, Any]], field: str, seed: int, samples: int
) -> dict[str, Any]:
    selected = [row for row in rows if finite(row.get(field)) is not None]
    if not selected:
        return {"case_support": 0, "cluster_support": 0, "mean": None, "ci95": None}
    clusters: dict[str, list[float]] = defaultdict(list)
    for row in selected:
        clusters[str(row["cluster"])].append(float(row[field]))
    names = sorted(clusters)
    point = float(np.mean([float(row[field]) for row in selected]))
    rng = np.random.default_rng(seed)
    distribution = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        chosen = rng.integers(0, len(names), size=len(names))
        values = [value for index in chosen for value in clusters[names[int(index)]]]
        distribution[sample] = float(np.mean(values))
    return {
        "case_support": len(selected),
        "cluster_support": len(names),
        "mean": point,
        "ci95": [
            float(np.quantile(distribution, 0.025)),
            float(np.quantile(distribution, 0.975)),
        ],
        "bootstrap_samples": samples,
    }


def summarize_group(
    rows: list[dict[str, Any]], seed: int, samples: int
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for index, metric in enumerate(sorted({str(row["metric"]) for row in rows})):
        selected = [row for row in rows if row["metric"] == metric]
        paired = [
            row for row in selected
            if finite(row.get("online")) is not None
            and finite(row.get("internal")) is not None
        ]
        result[metric] = {
            "online": mean_ci_clustered(paired, "online", seed + index * 11, samples),
            "internal": mean_ci_clustered(paired, "internal", seed + index * 11 + 1, samples),
            "internal_advantage": mean_ci_clustered(
                paired, "internal_advantage", seed + index * 11 + 2, samples
            ),
            "advantage_sign": (
                "online_minus_internal_for_lower_is_better_metrics; "
                "internal_minus_online_for_higher_is_better_metrics"
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online-aggregate", type=Path, required=True)
    parser.add_argument("--internal-metrics-dir", type=Path, required=True)
    parser.add_argument("--internal-file-glob", default="*.json")
    parser.add_argument("--method", action="append", type=parse_method, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260905)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()

    online = read_json(args.online_aggregate.resolve())
    if int(online.get("missing_case_count", -1)) != 0:
        raise RuntimeError("paired comparison requires a complete OnlineHMR aggregate")
    protocol = str(online["protocol"])
    harmony = protocol == "Bridge3R-Harmony4D-MultiCut-v1"
    if protocol not in {"CS150", "MVH150", "MC150-3", "MC150-4"} and not harmony:
        raise ValueError(f"unsupported paired protocol: {protocol}")
    metrics = (
        HARMONY_METRICS if harmony
        else MULTICUT_METRICS if protocol.startswith("MC150")
        else SINGLE_METRICS
    )
    internal = load_internal(args.internal_metrics_dir.resolve(), args.internal_file_glob)
    online_ids = [str(row["case_id"]) for row in online["cases"]]
    missing = [case_id for case_id in online_ids if case_id not in internal]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} internal cases; first={missing[0]}")

    rows: list[dict[str, Any]] = []
    for online_case in online["cases"]:
        case_id = str(online_case["case_id"])
        internal_case = internal[case_id]
        cluster = online_case.get("subject") or case_id
        stratum = online_case.get("angle_stratum")
        for method_label, method_key in args.method:
            method = internal_case.get("methods", {}).get(method_key)
            if not isinstance(method, dict):
                raise KeyError(f"{case_id}: missing internal method {method_key}")
            for display, source in metrics.items():
                online_value = finite(online_case.get(display))
                internal_value = (
                    finite(nested(method, *source))
                    if harmony
                    else finite(method.get("metrics", {}).get(source, {}).get("mean"))
                )
                rows.append({
                    "case_id": case_id,
                    "cluster": cluster,
                    "angle_stratum": stratum,
                    "internal_method": method_label,
                    "internal_method_key": method_key,
                    "metric": display,
                    "online": online_value,
                    "internal": internal_value,
                    "internal_advantage": (
                        None if online_value is None or internal_value is None
                        else (
                            internal_value - online_value
                            if display in HIGHER_IS_BETTER
                            else online_value - internal_value
                        )
                    ),
                })
            if harmony:
                seams = [
                    finite(nested(value, "cut_seam", "root_excess_m"))
                    for _, value in sorted(
                        method.get("cut_seams", {}).items(), key=lambda item: int(item[0])
                    )
                ]
                seams = [value for value in seams if value is not None]
                online_seam = finite(online_case.get("Seam-root_m"))
                internal_seam = None if not seams else float(np.mean(seams))
                rows.append({
                    "case_id": case_id,
                    "cluster": cluster,
                    "angle_stratum": stratum,
                    "internal_method": method_label,
                    "internal_method_key": method_key,
                    "metric": "Seam-root_m",
                    "online": online_seam,
                    "internal": internal_seam,
                    "internal_advantage": (
                        None if online_seam is None or internal_seam is None
                        else online_seam - internal_seam
                    ),
                })
            online_coverage = finite(online_case.get("Coverage"))
            internal_coverage = finite(
                method.get("coverage", {}).get("coverage" if harmony else "valid_frame_coverage")
            )
            rows.append({
                "case_id": case_id,
                "cluster": cluster,
                "angle_stratum": stratum,
                "internal_method": method_label,
                "internal_method_key": method_key,
                "metric": "Coverage",
                "online": online_coverage,
                "internal": internal_coverage,
                "internal_advantage": (
                    None if online_coverage is None or internal_coverage is None
                    else internal_coverage - online_coverage
                ),
            })

    by_method: dict[str, Any] = {}
    for method_index, (method_label, method_key) in enumerate(args.method):
        selected = [row for row in rows if row["internal_method_key"] == method_key]
        strata = {}
        for stratum in sorted({str(row["angle_stratum"]) for row in selected if row.get("angle_stratum")}):
            stratum_rows = [row for row in selected if row.get("angle_stratum") == stratum]
            strata[stratum] = summarize_group(
                stratum_rows,
                args.bootstrap_seed + 1000 * (method_index + 1) + len(strata) * 100,
                args.bootstrap_samples,
            )
        by_method[method_key] = {
            "label": method_label,
            "overall": summarize_group(
                selected,
                args.bootstrap_seed + 1000 * method_index,
                args.bootstrap_samples,
            ),
            "angle_strata": strata,
        }

    payload = {
        "schema_version": "Bridge3R-OnlineHMR-extension-paired-comparison-v1",
        "dataset": online["dataset"],
        "protocol": protocol,
        "fixed_manifest_denominator": online["fixed_manifest_denominator"],
        "online_method": online["method"],
        "internal_methods": by_method,
        "pairing_contract": {
            "same_case_ids": True,
            "same_frozen_evaluator_metric_definitions": True,
            "conditional_geometry_uses_pairwise_finite_support": True,
            "coverage_uses_full_fixed_denominator": True,
            "cluster": "subject when present; otherwise case",
        },
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "onlinehmr_internal_paired.json", payload)
    fields = [
        "case_id", "cluster", "angle_stratum", "internal_method",
        "internal_method_key", "metric", "online", "internal", "internal_advantage",
    ]
    with (output / "onlinehmr_internal_paired_cases.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        "output": str(output),
        "protocol": protocol,
        "case_count": len(online_ids),
        "internal_methods": [label for label, _ in args.method],
    }, indent=2))


if __name__ == "__main__":
    main()
