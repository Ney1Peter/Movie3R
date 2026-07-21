#!/usr/bin/env python3
"""Test explicit 3D-consistency vetoes for the V32 consensus branch."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from v25_holdout_rotation_validation import angle_deg, safe_gravity
from v32_consensus_texture_safety_audit import selected_rotation
from v34_consensus_cap_safety_audit import SETS, load_shards


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "output/v35_consensus_metric_safety"
TEXTURE_BOUND = 0.05


def distribution(values: np.ndarray) -> dict:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
    }


def metrics(rows: list[dict]) -> dict:
    fixed = np.asarray([row["fixed_error"] for row in rows])
    torso = np.asarray([row["torso_error"] for row in rows])
    baseline = np.asarray([row["baseline_error"] for row in rows])
    final = np.asarray([row["final_error"] for row in rows])
    return {
        "count": len(rows),
        "rotation_deg": distribution(final),
        "catastrophic_count": int(np.sum(final > 45.0)),
        "catastrophic_rate": float(np.mean(final > 45.0)),
        "rescued_vs_fixed": int(np.sum((fixed > 45.0) & (final <= 45.0))),
        "introduced_vs_fixed": int(np.sum((fixed <= 45.0) & (final > 45.0))),
        "rescued_vs_torso": int(np.sum((torso > 45.0) & (final <= 45.0))),
        "introduced_vs_torso": int(np.sum((torso <= 45.0) & (final > 45.0))),
        "harmful_over_5deg_vs_torso": int(np.sum(final > torso + 5.0)),
        "improved_over_5deg_vs_torso": int(np.sum(final + 5.0 < torso)),
        "rescued_vs_baseline": int(np.sum((baseline > 45.0) & (final <= 45.0))),
        "introduced_vs_baseline": int(np.sum((baseline <= 45.0) & (final > 45.0))),
        "harmful_over_5deg_vs_baseline": int(np.sum(final > baseline + 5.0)),
        "improved_over_5deg_vs_baseline": int(np.sum(final + 5.0 < baseline)),
    }


def accepted(metric: dict, config: dict) -> bool:
    if metric.get("fit_failed") or metric.get("transform") is None:
        return False
    return bool(
        float(metric.get("fit_residual_median_m", float("inf")))
        <= config["max_fit_m"]
        and float(metric.get("robust_inlier_ratio", 0.0))
        >= config["min_inlier_ratio"]
        and float(metric.get("epipolar_median_px", float("inf")))
        <= config["max_epipolar_px"]
        and int(metric.get("correspondence_count", 0)) >= 100
    )


def ranking_key(result: dict) -> tuple:
    overall = result["overall"]
    return (
        max(row["introduced_vs_torso"] for row in result["by_set"].values()),
        overall["introduced_vs_torso"],
        max(row["introduced_vs_baseline"] for row in result["by_set"].values()),
        overall["introduced_vs_baseline"],
        overall["catastrophic_count"],
        max(row["harmful_over_5deg_vs_torso"] for row in result["by_set"].values()),
        overall["harmful_over_5deg_vs_torso"],
        overall["rotation_deg"]["p95"],
        overall["rotation_deg"]["mean"],
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases = []
    for set_name, v15_dir, v16_dir in SETS:
        v15 = load_shards(v15_dir, "v15_candidates")
        v16 = load_shards(v16_dir, "v16_candidates")
        names = sorted(set(v15) & set(v16))
        if len(names) != len(v15) or len(names) != len(v16):
            raise RuntimeError(f"Mismatch in {set_name}: {len(v15)}/{len(v16)}")
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
            baseline, branch, diagnostics = selected_rotation(
                fixed, torso, wide, TEXTURE_BOUND, consensus_cap_deg=60.0
            )
            cases.append(
                {
                    "set": set_name,
                    "case_name": name,
                    "source": human["record"]["source"],
                    "fixed": fixed,
                    "torso": torso,
                    "gt": gt,
                    "baseline": baseline,
                    "baseline_branch": branch,
                    "diagnostics": diagnostics,
                    "metric": wide["windows"]["full_rgb_1p1"]["candidates"][
                        "metric_full_full"
                    ],
                    "fixed_error": angle_deg(fixed, gt),
                    "torso_error": angle_deg(torso, gt),
                    "baseline_error": angle_deg(baseline, gt),
                }
            )

    configs = [
        {
            "max_fit_m": max_fit,
            "min_inlier_ratio": min_inlier,
            "max_epipolar_px": max_epipolar,
        }
        for max_fit, min_inlier, max_epipolar in itertools.product(
            (0.50, 0.70, 1.00, 1.25, 1.50, 2.00),
            (0.15, 0.25, 0.35, 0.45, 0.55),
            (8.0, 10.0, 15.0, 20.0, 1000.0),
        )
    ]
    results = []
    for config in configs:
        rows = []
        vetoed = []
        for case in cases:
            veto = case["baseline_branch"] == "consensus" and not accepted(
                case["metric"], config
            )
            final = case["torso"] if veto else case["baseline"]
            final_error = angle_deg(final, case["gt"])
            rows.append({**case, "final_error": final_error})
            if veto:
                metric = case["metric"]
                vetoed.append(
                    {
                        "set": case["set"],
                        "case_name": case["case_name"],
                        "source": case["source"],
                        "baseline_error": case["baseline_error"],
                        "final_error": final_error,
                        "fit_m": float(metric.get("fit_residual_median_m", float("inf"))),
                        "inlier_ratio": float(metric.get("robust_inlier_ratio", 0.0)),
                        "epipolar_px": float(metric.get("epipolar_median_px", float("inf"))),
                    }
                )
        results.append(
            {
                "config": config,
                "vetoed_count": len(vetoed),
                "vetoed_cases": vetoed,
                "overall": metrics(rows),
                "by_set": {
                    set_name: metrics([row for row in rows if row["set"] == set_name])
                    for set_name in sorted({row["set"] for row in rows})
                },
            }
        )
    results.sort(key=ranking_key)
    report = {
        "experiment": "V35 explicit metric safety veto for positive consensus",
        "protocol": {
            "case_count": len(cases),
            "texture_bound": TEXTURE_BOUND,
            "consensus_cap_deg": 60.0,
            "gt_used_for_offline_development_only": True,
            "holdout6_reserved_for_frozen_validation": True,
        },
        "top_rules": results[:100],
    }
    output = OUTPUT / "v35_consensus_metric_safety_audit.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"best": results[0], "top10": results[:10]}, indent=2))
    print(f">> wrote {output}")


if __name__ == "__main__":
    main()
