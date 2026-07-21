#!/usr/bin/env python3
"""Probe fixed multi-window explicit rotation fallbacks on development caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from v25_holdout_rotation_validation import angle_deg, capped_rotation
from v29_cross_holdout_explicit_rule_audit import (
    DEFAULT_SETS,
    background_tail_trigger,
    load_cases,
    load_shards,
    low_torso_trigger,
    metrics,
)
from v29_frozen_explicit_rule_validation import BACKGROUND_CONFIG, LOW_CONFIG


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "output" / "v30_multiwindow_rule_probe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def enrich_cases(cases: list[dict]) -> None:
    wide_by_set = {
        set_name: load_shards(v15_dir, "v15_candidates")
        for set_name, v15_dir, _ in DEFAULT_SETS
    }
    for case in cases:
        wide = wide_by_set[case["set"]][case["case_name"]]
        full3 = wide["windows"]["full_rgb_3p3"]
        background = wide["windows"]["background_only_1p1"]
        full3_rotation = np.asarray(
            full3["candidates"]["coarse"]["transform"], dtype=np.float32
        )[:3, :3]
        case["full3_rotation"] = full3_rotation
        case["full3_residual"] = angle_deg(full3_rotation, case["fixed"])
        case["full3_spread"] = float(full3["rotation_consensus"]["spread_deg"])
        case["full_full3_agreement"] = angle_deg(
            case["full_rotation"], full3_rotation
        )
        case["full_background_agreement"] = angle_deg(
            case["full_rotation"], case["background_rotation"]
        )
        case["background_count"] = int(
            background["candidates"]["metric_full_full"]["correspondence_count"]
        )


def positive_near_spread(case: dict) -> bool:
    return bool(
        not case["v24_accepted"]
        and 10.0 <= case["torso_residual"] <= 30.0
        and case["direction_cosine"] >= 0.0
        and case["texture"] <= 0.02
        and 30.0 <= case["full_residual"] <= 100.0
        and 5.0 < case["full_spread"] <= 6.0
        and case["full_fit"] <= 0.30
        and case["full_inlier"] >= 0.90
        and case["full_count"] >= 100
    )


def large_near_spread(case: dict) -> bool:
    return bool(
        not case["v24_accepted"]
        and case["torso_residual"] >= 30.0
        and case["full_residual"] >= case["torso_residual"] + 5.0
        and case["full_residual"] <= 100.0
        and 15.0 < case["full_spread"] <= 20.0
    )


def large_multiframe_consensus(case: dict) -> bool:
    return bool(
        not case["v24_accepted"]
        and case["torso_residual"] >= 30.0
        and 100.0 < case["full_residual"] <= 145.0
        and 100.0 < case["full3_residual"] <= 145.0
        and case["full3_spread"] <= 5.0
        and case["full_full3_agreement"] <= 30.0
        and case["texture"] <= 0.03
    )


def geometry_failure_mask_consensus(case: dict) -> bool:
    return bool(
        not case["v24_accepted"]
        and 5.0 <= case["torso_residual"] <= 15.0
        and -0.5 <= case["direction_cosine"] < 0.0
        and case["texture"] <= 0.02
        and 40.0 <= case["full_residual"] <= 100.0
        and case["full_spread"] <= 5.0
        and case["full_fit"] >= 1.0
        and case["full_inlier"] <= 0.40
        and case["background_residual"] <= 100.0
        and case["background_spread"] <= 6.0
        and case["full_background_agreement"] <= 20.0
        and case["background_count"] >= 100
    )


def select_rotation(case: dict) -> tuple[np.ndarray, str]:
    rotation = case["base"]
    if low_torso_trigger(case, LOW_CONFIG):
        return capped_rotation(rotation, case["full_rotation"], 60.0), "v29_low_torso"
    if background_tail_trigger(case, BACKGROUND_CONFIG):
        return (
            capped_rotation(rotation, case["background_rotation"], 45.0),
            "v29_background_tail",
        )
    if positive_near_spread(case):
        return capped_rotation(rotation, case["full_rotation"], 60.0), "positive_near_spread"
    if large_near_spread(case):
        return capped_rotation(rotation, case["full_rotation"], 25.0), "large_near_spread"
    if large_multiframe_consensus(case):
        return (
            capped_rotation(rotation, case["full3_rotation"], 60.0),
            "large_multiframe_consensus",
        )
    if geometry_failure_mask_consensus(case):
        return (
            capped_rotation(rotation, case["background_rotation"], 60.0),
            "geometry_failure_mask_consensus",
        )
    return rotation, "v24"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases()
    enrich_cases(cases)
    rows = []
    for case in cases:
        rotation, selected = select_rotation(case)
        rows.append(
            {
                **case,
                "selected": selected,
                "proposal_error": angle_deg(rotation, case["gt"]),
            }
        )
    overall = metrics(rows, "proposal_error")
    by_set = {
        set_name: metrics(
            [row for row in rows if row["set"] == set_name], "proposal_error"
        )
        for set_name in sorted({row["set"] for row in rows})
    }
    trigger_counts = {
        selected: sum(row["selected"] == selected for row in rows)
        for selected in sorted({row["selected"] for row in rows})
        if selected != "v24"
    }
    triggered = [
        {
            "set": row["set"],
            "case_name": row["case_name"],
            "source": row["source"],
            "selected": row["selected"],
            "v24_rotation_error_deg": row["base_error"],
            "v30_rotation_error_deg": row["proposal_error"],
        }
        for row in rows
        if row["selected"] != "v24"
    ]
    report = {
        "experiment": "V30 development multi-window explicit rule probe",
        "protocol": {
            "case_count": len(rows),
            "gt_runtime_information": False,
            "holdout3_reserved_for_frozen_validation": True,
        },
        "trigger_counts": trigger_counts,
        "overall": overall,
        "by_set": by_set,
        "triggered_cases": triggered,
    }
    output = args.output_dir / "v30_development_multiwindow_rule_probe.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    print(f">> wrote {output}")


if __name__ == "__main__":
    main()
