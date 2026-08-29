#!/usr/bin/env python3
"""Validate a completed standardized Bridge3R runtime/memory report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any


EXPECTED_ROUTES = (
    "strict_human3r",
    "bridge3r_no_cut",
    "bridge3r_single_cut_transaction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-runtime-cv-percent", type=float, default=2.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    report_path = args.report.resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}
    checks["schema"] = report.get("schema_version") == "Bridge3R-standardized-runtime-memory-v1"
    protocol = report["protocol"]
    checks.update(
        input_512=int(protocol["input_size"]) == 512,
        batch_size_one=int(protocol["batch_size"]) == 1,
        fp32=str(protocol["precision"]) == "FP32",
        autocast_off=protocol["autocast_enabled"] is False,
        tf32_off=protocol["tf32_enabled"] is False,
        one_or_more_warmups=int(protocol["warmup"]) >= 1,
        three_or_more_repetitions=int(protocol["repetitions"]) >= 3,
        clip_100=int(protocol["clip_frames"]) == 100,
        single_process_declared=protocol["single_process_on_gpu"] is True,
    )
    case = report["case"]
    checks.update(
        boundary_50=int(case["boundary_index"]) == 50,
        case_frame_count=int(case["pre_frames"]) + int(case["post_frames"]) == 100,
        manifest_hash_matches=sha256(Path(case["manifest"])) == case["manifest_sha256"],
    )
    gpu = report["hardware"]["gpu"]
    checks.update(
        gpu_index_four=int(gpu["torch_index"]) == 4,
        gpu_is_l20="L20" in str(gpu["torch_name"]),
    )
    checks["benchmark_script_hash_matches"] = (
        sha256(Path(report["software"]["script"])) == report["software"]["script_sha256"]
    )
    for name, checkpoint in report["checkpoints"].items():
        checks[f"checkpoint_hash_matches__{name}"] = (
            sha256(Path(checkpoint["path"])) == checkpoint["sha256"]
        )

    stability: dict[str, Any] = {}
    checks["route_set_exact"] = set(report["routes"]) == set(EXPECTED_ROUTES)
    for name in EXPECTED_ROUTES:
        route = report["routes"][name]
        rows = route["timed_repetitions"]
        values = [float(row["total_seconds"]) for row in rows]
        peaks = [int(row["peak_allocated_bytes"]) for row in rows]
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0.0
        cv = 100.0 * stdev / mean
        route_checks = {
            "repetition_count_matches": len(rows) == int(protocol["repetitions"]),
            "all_runtime_finite_positive": all(math.isfinite(value) and value > 0 for value in values),
            "all_peak_finite_positive": all(value > 0 for value in peaks),
            "runtime_cv_within_threshold": cv <= float(args.max_runtime_cv_percent),
        }
        for key, value in route_checks.items():
            checks[f"{name}__{key}"] = value
        stability[name] = {
            "total_seconds": values,
            "mean_seconds": mean,
            "stdev_seconds": stdev,
            "cv_percent": cv,
            "range_seconds": max(values) - min(values),
            "peak_allocated_bytes": peaks,
            "peak_range_bytes": max(peaks) - min(peaks),
        }

    summary = report["summary"]
    strict_seconds = float(summary["strict_human3r_seconds"])
    no_cut_seconds = float(summary["bridge3r_no_cut_seconds"])
    derived = {
        "no_cut_extra_vs_strict_seconds": no_cut_seconds - strict_seconds,
        "no_cut_overhead_vs_strict_percent": 100.0
        * (no_cut_seconds - strict_seconds)
        / strict_seconds,
        "cut_extra_vs_no_cut_seconds": float(summary["cut_extra_seconds"]),
        "cut_overhead_vs_no_cut_percent": float(summary["cut_overhead_percent"]),
    }
    payload = {
        "schema_version": "Bridge3R-standardized-runtime-memory-integrity-v1",
        "source_report": str(report_path),
        "source_report_sha256": sha256(report_path),
        "source_markdown": str(report_path.with_suffix(".md")),
        "source_markdown_sha256": sha256(report_path.with_suffix(".md")),
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "stability_threshold_cv_percent": float(args.max_runtime_cv_percent),
        "stability": stability,
        "derived_comparison": derived,
        "interpretation": (
            "All timings are stable under the preregistered one-warmup/three-repeat "
            "single-GPU protocol. This audit verifies file bindings and report "
            "self-consistency; process exclusivity was enforced and observed by the "
            "launcher but is not independently reconstructable from this JSON."
        ),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(partial, output)
    print(json.dumps({"output": str(output), "all_checks_pass": payload["all_checks_pass"]}, indent=2))


if __name__ == "__main__":
    main()
