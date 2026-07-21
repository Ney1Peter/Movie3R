#!/usr/bin/env python3
"""Validate frozen V30 multi-window explicit rules on an unseen cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from v25_holdout_rotation_validation import angle_deg, safe_gravity, v24_rotation
from v29_frozen_explicit_rule_validation import load_shards
from v30_development_multiwindow_rule_probe import select_rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "output" / "v29_rotation_rule_holdout3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v15_dir", type=Path, default=DEFAULT_ROOT / "v15")
    parser.add_argument("--v16_dir", type=Path, default=DEFAULT_ROOT / "v16")
    parser.add_argument(
        "--output_dir", type=Path, default=DEFAULT_ROOT / "v30_frozen_validation"
    )
    return parser.parse_args()


def distribution(values: np.ndarray) -> dict:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
    }


def aggregate(rows: list[dict]) -> dict:
    base = np.asarray([row["v24_rotation_error_deg"] for row in rows])
    final = np.asarray([row["v30_rotation_error_deg"] for row in rows])
    return {
        "count": len(rows),
        "v24_rotation_deg": distribution(base),
        "v30_rotation_deg": distribution(final),
        "v24_catastrophic_rate": float(np.mean(base > 45.0)),
        "v30_catastrophic_rate": float(np.mean(final > 45.0)),
        "rescued_catastrophic_count": int(
            np.sum((base > 45.0) & (final <= 45.0))
        ),
        "introduced_catastrophic_count": int(
            np.sum((base <= 45.0) & (final > 45.0))
        ),
        "harmful_over_5deg_count": int(np.sum(final > base + 5.0)),
        "good_case_harmful_count": int(
            np.sum((base < 10.0) & (final > base + 5.0))
        ),
    }


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
        v24, diagnostics = v24_rotation(fixed, torso, wide)
        full = wide["windows"]["full_rgb_1p1"]
        full3 = wide["windows"]["full_rgb_3p3"]
        background = wide["windows"]["background_only_1p1"]
        full_rotation = np.asarray(
            full["candidates"]["coarse"]["transform"], dtype=np.float32
        )[:3, :3]
        full3_rotation = np.asarray(
            full3["candidates"]["coarse"]["transform"], dtype=np.float32
        )[:3, :3]
        background_rotation = np.asarray(
            background["candidates"]["coarse"]["transform"], dtype=np.float32
        )[:3, :3]
        full_metric = full["candidates"]["metric_full_full"]
        background_metric = background["candidates"]["metric_full_full"]
        case = {
            "case_name": name,
            "source": human["record"]["source"],
            "fixed": fixed,
            "gt": gt,
            "base": v24,
            "v24_accepted": bool(diagnostics["v24_accepted"]),
            "torso_residual": float(diagnostics["torso_residual_deg"]),
            "direction_cosine": float(diagnostics["residual_direction_cosine"]),
            "texture": float(wide["texture_score"]),
            "full_rotation": full_rotation,
            "full_residual": float(diagnostics["vggt_residual_deg"]),
            "full_spread": float(full["rotation_consensus"]["spread_deg"]),
            "full_fit": float(full_metric["fit_residual_median_m"]),
            "full_inlier": float(full_metric["robust_inlier_ratio"]),
            "full_count": int(full_metric["correspondence_count"]),
            "full3_rotation": full3_rotation,
            "full3_residual": angle_deg(full3_rotation, fixed),
            "full3_spread": float(full3["rotation_consensus"]["spread_deg"]),
            "full_full3_agreement": angle_deg(full_rotation, full3_rotation),
            "background_rotation": background_rotation,
            "background_residual": angle_deg(background_rotation, fixed),
            "background_spread": float(
                background["rotation_consensus"]["spread_deg"]
            ),
            "background_fit": float(background_metric["fit_residual_median_m"]),
            "background_inlier": float(background_metric["robust_inlier_ratio"]),
            "background_count": int(background_metric["correspondence_count"]),
            "full_background_agreement": angle_deg(
                full_rotation, background_rotation
            ),
        }
        final, selected = select_rotation(case)
        rows.append(
            {
                "case_name": name,
                "source": case["source"],
                "selected": selected,
                "v24_rotation_error_deg": angle_deg(v24, gt),
                "v30_rotation_error_deg": angle_deg(final, gt),
                "trigger_diagnostics": {
                    key: value
                    for key, value in case.items()
                    if key
                    not in {
                        "fixed",
                        "gt",
                        "base",
                        "full_rotation",
                        "full3_rotation",
                        "background_rotation",
                    }
                },
                "v30_transform_rotation": final.tolist(),
            }
        )

    trigger_counts = {
        selected: sum(row["selected"] == selected for row in rows)
        for selected in sorted({row["selected"] for row in rows})
        if selected != "v24"
    }
    report = {
        "experiment": "V30 frozen multi-window explicit rule validation",
        "protocol": {
            "thresholds_frozen_before_this_cache_was_evaluated": True,
            "gt_runtime_information": False,
            "post_cut_window": "1+1 except one 3+3 consensus fallback",
        },
        "trigger_counts": trigger_counts,
        "overall": aggregate(rows),
        "by_source": {
            source: aggregate([row for row in rows if row["source"] == source])
            for source in sorted({row["source"] for row in rows})
        },
        "triggered_cases": [row for row in rows if row["selected"] != "v24"],
        "cases": rows,
    }
    output = args.output_dir / "v30_frozen_multiwindow_rule_validation.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "trigger_counts": report["trigger_counts"],
                "overall": report["overall"],
                "by_source": report["by_source"],
                "triggered_cases": [
                    {
                        "case_name": row["case_name"],
                        "source": row["source"],
                        "selected": row["selected"],
                        "v24_rotation_error_deg": row["v24_rotation_error_deg"],
                        "v30_rotation_error_deg": row["v30_rotation_error_deg"],
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
