#!/usr/bin/env python3
"""Validate a frozen low-torso explicit metric-rotation fallback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from v25_holdout_rotation_validation import (
    angle_deg,
    capped_rotation,
    safe_gravity,
    v24_rotation,
)
from v29_frozen_explicit_rule_validation import load_shards
from v30_frozen_multiwindow_rule_validation import aggregate


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "output" / "v31_metric_fit_holdout4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v15_dir", type=Path, default=DEFAULT_ROOT / "v15")
    parser.add_argument("--v16_dir", type=Path, default=DEFAULT_ROOT / "v16")
    parser.add_argument(
        "--output_dir", type=Path, default=DEFAULT_ROOT / "frozen_rule_validation"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    v15 = load_shards(args.v15_dir, "v15_candidates")
    v16 = load_shards(args.v16_dir, "v16_candidates")
    names = sorted(set(v15) & set(v16))
    if len(names) != len(v15) or len(names) != len(v16):
        raise RuntimeError(f"V15/V16 mismatch: {len(v15)}/{len(v16)}/{len(names)}")

    rows = []
    for name in names:
        wide = v15[name]
        human = v16[name]
        fixed = np.asarray(
            wide["baselines"]["fixed_explicit"]["transform"], dtype=np.float32
        )[:3, :3]
        gt = np.asarray(
            wide["baselines"]["boundary_oracle"]["transform"], dtype=np.float32
        )[:3, :3]
        torso, _ = safe_gravity(human)
        base, diagnostics = v24_rotation(fixed, torso, wide)
        full = wide["windows"]["full_rgb_1p1"]
        coarse = np.asarray(
            full["candidates"]["coarse"]["transform"], dtype=np.float32
        )[:3, :3]
        metric_row = full["candidates"]["metric_full_full"]
        metric = np.asarray(metric_row["transform"], dtype=np.float32)[:3, :3]
        camera_metric_agreement = angle_deg(coarse, metric)
        metric_residual = angle_deg(metric, fixed)
        trigger = bool(
            not diagnostics["v24_accepted"]
            and float(diagnostics["torso_residual_deg"]) < 5.0
            and float(wide["texture_score"]) < 0.01
            and float(full["rotation_consensus"]["spread_deg"]) <= 2.0
            and 30.0 <= float(diagnostics["vggt_residual_deg"]) <= 100.0
            and 30.0 <= metric_residual <= 100.0
            and camera_metric_agreement <= 30.0
            and float(metric_row["fit_residual_median_m"]) <= 0.50
            and float(metric_row["robust_inlier_ratio"]) >= 0.80
            and int(metric_row["correspondence_count"]) >= 100
        )
        final = capped_rotation(base, metric, 45.0) if trigger else base
        rows.append(
            {
                "case_name": name,
                "source": human["record"]["source"],
                "selected": "low_torso_metric_full" if trigger else "v24",
                "v24_rotation_error_deg": angle_deg(base, gt),
                "v30_rotation_error_deg": angle_deg(final, gt),
                "trigger_diagnostics": {
                    "torso_residual_deg": float(
                        diagnostics["torso_residual_deg"]
                    ),
                    "texture_score": float(wide["texture_score"]),
                    "coarse_residual_deg": float(
                        diagnostics["vggt_residual_deg"]
                    ),
                    "coarse_spread_deg": float(
                        full["rotation_consensus"]["spread_deg"]
                    ),
                    "metric_residual_deg": metric_residual,
                    "camera_metric_agreement_deg": camera_metric_agreement,
                    "metric_fit_residual_median_m": float(
                        metric_row["fit_residual_median_m"]
                    ),
                    "metric_robust_inlier_ratio": float(
                        metric_row["robust_inlier_ratio"]
                    ),
                    "metric_correspondence_count": int(
                        metric_row["correspondence_count"]
                    ),
                },
                "v31_transform_rotation": final.tolist(),
            }
        )

    report = {
        "experiment": "V31 frozen low-torso metric-fit validation",
        "protocol": {
            "thresholds_frozen_before_this_cache_was_evaluated": True,
            "gt_runtime_information": False,
            "post_cut_frames": 1,
            "metric_rotation_source": "VGGT 2D tracks + Human3R 3D rigid fit",
        },
        "trigger_count": int(sum(row["selected"] != "v24" for row in rows)),
        "overall": aggregate(rows),
        "by_source": {
            source: aggregate([row for row in rows if row["source"] == source])
            for source in sorted({row["source"] for row in rows})
        },
        "triggered_cases": [row for row in rows if row["selected"] != "v24"],
        "cases": rows,
    }
    output = args.output_dir / "v31_low_torso_metric_fit_validation.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "trigger_count": report["trigger_count"],
                "overall": report["overall"],
                "by_source": report["by_source"],
                "triggered_cases": [
                    {
                        "case_name": row["case_name"],
                        "source": row["source"],
                        "v24_rotation_error_deg": row["v24_rotation_error_deg"],
                        "v31_rotation_error_deg": row["v30_rotation_error_deg"],
                    }
                    for row in report["triggered_cases"]
                ],
            },
            indent=2,
        )
    )
    print(f">> wrote {output}")


if __name__ == "__main__":
    main()
