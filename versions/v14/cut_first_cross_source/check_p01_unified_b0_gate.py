#!/usr/bin/env python3
"""Apply the preregistered P0.1 unified-B0 selection gate without writing outputs.

This is deliberately a *selection* utility, not an optimizer: the P0.1 final
checkpoint is fixed before it is called.  It makes the comparison auditable
and prevents a later external EgoHumans confirmation from silently deciding
between P0 and P0.1.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_P0_FOUR = REPO_ROOT / "output/v14_cut_first_cross_source/eval_multihuman_p0_180/four_source_b0_evaluation.json"
DEFAULT_P0_DEV = REPO_ROOT / "output/v14_cut_first_cross_source/eval_multihuman_camera_dev/p0_e6/multihuman_camera_dev_evaluation.json"
DEFAULT_OLD_DEV = REPO_ROOT / "output/v14_cut_first_cross_source/eval_multihuman_camera_dev/old_b0/multihuman_camera_dev_evaluation.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-four", type=Path, required=True)
    parser.add_argument("--candidate-dev", type=Path, required=True)
    parser.add_argument("--p0-four", type=Path, default=DEFAULT_P0_FOUR)
    parser.add_argument("--p0-dev", type=Path, default=DEFAULT_P0_DEV)
    parser.add_argument("--old-dev", type=Path, default=DEFAULT_OLD_DEV)
    parser.add_argument("--raw-tolerance", type=float, default=1e-9)
    parser.add_argument("--parity-limit", type=float, default=1e-5)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def value(report: dict[str, Any], method: str, metric: str, stat: str = "mean") -> float:
    return float(report["summary"]["methods"][method][metric][stat])


def case_key(row: dict[str, Any]) -> str:
    record = row.get("record", {})
    return str(record.get("pattern_id", record.get("event_id", "")))


def raw_case_delta(candidate: dict[str, Any], reference: dict[str, Any]) -> float:
    ref_rows = {case_key(row): row for row in reference.get("cases", []) if row.get("status") == "ok"}
    candidate_rows = {case_key(row): row for row in candidate.get("cases", []) if row.get("status") == "ok"}
    if candidate_rows.keys() != ref_rows.keys():
        missing = sorted(ref_rows.keys() - candidate_rows.keys())[:3]
        extra = sorted(candidate_rows.keys() - ref_rows.keys())[:3]
        raise RuntimeError(f"raw-reset case sets differ; missing={missing}, extra={extra}")
    maximum = 0.0
    for key, candidate_row in candidate_rows.items():
        candidate_metrics = candidate_row["methods"]["raw_reset"]
        reference_metrics = ref_rows[key]["methods"]["raw_reset"]
        if candidate_metrics.keys() != reference_metrics.keys():
            raise RuntimeError(f"raw-reset metric keys differ for {key}")
        for metric in candidate_metrics:
            candidate_value = candidate_metrics[metric]
            reference_value = reference_metrics[metric]
            if isinstance(candidate_value, bool):
                if candidate_value != reference_value:
                    return math.inf
            elif isinstance(candidate_value, (int, float)):
                maximum = max(maximum, abs(float(candidate_value) - float(reference_value)))
    return maximum


def main() -> None:
    args = parse_args()
    candidate_four = load(args.candidate_four)
    candidate_dev = load(args.candidate_dev)
    p0_four = load(args.p0_four)
    p0_dev = load(args.p0_dev)
    old_dev = load(args.old_dev)

    candidate_four_b0 = candidate_four["summary"]["methods"]["b0_runtime"]
    p0_four_b0 = p0_four["summary"]["methods"]["b0_runtime"]
    candidate_dev_b0 = candidate_dev["summary"]["methods"]["b0_runtime"]
    old_dev_b0 = old_dev["summary"]["methods"]["b0_runtime"]
    raw_four_delta = raw_case_delta(candidate_four, p0_four)
    raw_dev_delta = raw_case_delta(candidate_dev, p0_dev)
    parity_p95 = float(candidate_dev["summary"]["camera_parity"]["matrix_max_abs"]["p95"])

    checks = {
        "four_case_count_180": candidate_four["summary"]["case_count"] == 180,
        "four_no_failures": len(candidate_four.get("failures", [])) == 0,
        "four_p95_retains_p0": float(candidate_four_b0["camera_composite"]["p95"])
        <= float(p0_four_b0["camera_composite"]["p95"]),
        "four_catastrophic_retains_p0": int(candidate_four_b0["catastrophic_count"])
        <= int(p0_four_b0["catastrophic_count"]),
        "dev_case_count_36": candidate_dev["summary"]["case_count"] == 36,
        "dev_no_failures": len(candidate_dev.get("failures", [])) == 0,
        "dev_camera_nonregresses_old": value(candidate_dev, "b0_runtime", "camera_composite")
        <= value(old_dev, "b0_runtime", "camera_composite"),
        "dev_fixed_root_nonregresses_old": value(candidate_dev, "b0_runtime", "human_root_error_m")
        <= value(old_dev, "b0_runtime", "human_root_error_m"),
        "dev_pair_vector_nonregresses_old": value(candidate_dev, "b0_runtime", "pairwise_vector_error_m")
        <= value(old_dev, "b0_runtime", "pairwise_vector_error_m"),
        "dev_shadow_b0_parity": parity_p95 <= args.parity_limit,
        "four_raw_reset_parity": raw_four_delta <= args.raw_tolerance,
        "dev_raw_reset_parity": raw_dev_delta <= args.raw_tolerance,
    }
    result = {
        "candidate_four": str(args.candidate_four),
        "candidate_dev": str(args.candidate_dev),
        "checks": checks,
        "pass": all(checks.values()),
        "values": {
            "four_p95_candidate": float(candidate_four_b0["camera_composite"]["p95"]),
            "four_p95_p0": float(p0_four_b0["camera_composite"]["p95"]),
            "four_catastrophic_candidate": int(candidate_four_b0["catastrophic_count"]),
            "four_catastrophic_p0": int(p0_four_b0["catastrophic_count"]),
            "dev_camera_candidate": value(candidate_dev, "b0_runtime", "camera_composite"),
            "dev_camera_old": value(old_dev, "b0_runtime", "camera_composite"),
            "dev_root_candidate": value(candidate_dev, "b0_runtime", "human_root_error_m"),
            "dev_root_old": value(old_dev, "b0_runtime", "human_root_error_m"),
            "dev_pair_vector_candidate": value(candidate_dev, "b0_runtime", "pairwise_vector_error_m"),
            "dev_pair_vector_old": value(old_dev, "b0_runtime", "pairwise_vector_error_m"),
            "dev_camera_parity_p95": parity_p95,
            "four_raw_reset_max_abs_delta": raw_four_delta,
            "dev_raw_reset_max_abs_delta": raw_dev_delta,
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
