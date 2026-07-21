#!/usr/bin/env python3
"""Search conservative low-torso metric-fit fallbacks before H4 evaluation."""

from __future__ import annotations

import glob
import itertools
import json
from pathlib import Path

import numpy as np

from v25_holdout_rotation_validation import (
    angle_deg,
    capped_rotation,
    safe_gravity,
    v24_rotation,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "output" / "v31_metric_fit_rule_search"
SETS = (
    (
        "original180",
        REPO_ROOT / "output" / "v15_wide_baseline_boundary_bridge" / "candidate_cache",
        REPO_ROOT / "output" / "v16_human_aware_rotation_residual" / "candidate_cache",
    ),
    (
        "holdout1",
        REPO_ROOT / "output" / "v25_holdout_rotation_validation" / "v15",
        REPO_ROOT / "output" / "v25_holdout_rotation_validation" / "v16",
    ),
    (
        "holdout2",
        REPO_ROOT / "output" / "v27_consensus_holdout2" / "v15",
        REPO_ROOT / "output" / "v27_consensus_holdout2" / "v16",
    ),
    (
        "holdout3",
        REPO_ROOT / "output" / "v29_rotation_rule_holdout3" / "v15",
        REPO_ROOT / "output" / "v29_rotation_rule_holdout3" / "v16",
    ),
)


def load_shards(root: Path, prefix: str) -> dict[str, dict]:
    rows: list[dict] = []
    for path in sorted(glob.glob(str(root / f"{prefix}_shard_*_of_*.json"))):
        rows.extend(json.loads(Path(path).read_text(encoding="utf-8"))["cases"])
    output = {row["case_name"]: row for row in rows}
    if not rows or len(output) != len(rows):
        raise RuntimeError(f"Invalid cache {root}: {len(rows)}/{len(output)}")
    return output


def load_cases() -> list[dict]:
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
            base, diagnostics = v24_rotation(fixed, torso, wide)
            full = wide["windows"]["full_rgb_1p1"]
            coarse = np.asarray(
                full["candidates"]["coarse"]["transform"], dtype=np.float32
            )[:3, :3]
            metric_row = full["candidates"]["metric_full_full"]
            metric = np.asarray(metric_row["transform"], dtype=np.float32)[:3, :3]
            cases.append(
                {
                    "set": set_name,
                    "case_name": name,
                    "source": human["record"]["source"],
                    "fixed": fixed,
                    "gt": gt,
                    "base": base,
                    "metric": metric,
                    "base_error": angle_deg(base, gt),
                    "v24_accepted": bool(diagnostics["v24_accepted"]),
                    "torso_residual": float(diagnostics["torso_residual_deg"]),
                    "texture": float(wide["texture_score"]),
                    "coarse_residual": float(diagnostics["vggt_residual_deg"]),
                    "spread": float(full["rotation_consensus"]["spread_deg"]),
                    "metric_residual": angle_deg(metric, fixed),
                    "agreement": angle_deg(coarse, metric),
                    "fit": float(metric_row["fit_residual_median_m"]),
                    "inlier": float(metric_row["robust_inlier_ratio"]),
                    "count": int(metric_row["correspondence_count"]),
                }
            )
    return cases


def trigger(case: dict, config: dict) -> bool:
    return bool(
        not case["v24_accepted"]
        and case["torso_residual"] < config["max_torso"]
        and case["texture"] < config["max_texture"]
        and case["spread"] <= config["max_spread"]
        and 30.0 <= case["coarse_residual"] <= 100.0
        and 30.0 <= case["metric_residual"] <= 100.0
        and case["agreement"] <= config["max_agreement"]
        and case["fit"] <= config["max_fit"]
        and case["inlier"] >= config["min_inlier"]
        and case["count"] >= 100
    )


def metrics(rows: list[dict]) -> dict:
    base = np.asarray([row["base_error"] for row in rows])
    final = np.asarray([row["final_error"] for row in rows])
    return {
        "count": len(rows),
        "mean": float(final.mean()),
        "p90": float(np.quantile(final, 0.90)),
        "p95": float(np.quantile(final, 0.95)),
        "catastrophic_count": int(np.sum(final > 45.0)),
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


def evaluate(cases: list[dict], config: dict) -> dict:
    rows = []
    active = []
    for case in cases:
        accepted = trigger(case, config)
        final = (
            capped_rotation(case["base"], case["metric"], config["cap_deg"])
            if accepted
            else case["base"]
        )
        final_error = angle_deg(final, case["gt"])
        rows.append({**case, "final_error": final_error})
        if accepted:
            active.append(
                {
                    "set": case["set"],
                    "case_name": case["case_name"],
                    "source": case["source"],
                    "base_error": case["base_error"],
                    "final_error": final_error,
                }
            )
    return {
        "config": config,
        "active_count": len(active),
        "active_cases": active,
        "overall": metrics(rows),
        "by_set": {
            set_name: metrics([row for row in rows if row["set"] == set_name])
            for set_name in sorted({row["set"] for row in rows})
        },
    }


def ranking_key(result: dict) -> tuple:
    overall = result["overall"]
    return (
        max(
            value["introduced_catastrophic_count"]
            for value in result["by_set"].values()
        ),
        max(value["good_case_harmful_count"] for value in result["by_set"].values()),
        overall["introduced_catastrophic_count"],
        overall["good_case_harmful_count"],
        -overall["rescued_catastrophic_count"],
        max(value["harmful_over_5deg_count"] for value in result["by_set"].values()),
        overall["harmful_over_5deg_count"],
        overall["mean"],
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases = load_cases()
    configs = [
        {
            "max_torso": max_torso,
            "max_texture": max_texture,
            "max_spread": max_spread,
            "max_agreement": max_agreement,
            "max_fit": max_fit,
            "min_inlier": min_inlier,
            "cap_deg": cap_deg,
        }
        for max_torso, max_texture, max_spread, max_agreement, max_fit, min_inlier, cap_deg in itertools.product(
            (5.0, 7.5, 10.0),
            (0.01, 0.02, 0.03),
            (2.0, 3.0, 5.0),
            (15.0, 20.0, 30.0),
            (0.3, 0.5, 0.7),
            (0.8, 0.9),
            (45.0, 60.0),
        )
    ]
    results = sorted((evaluate(cases, config) for config in configs), key=ranking_key)
    report = {
        "experiment": "V31 metric-fit rule search before H4",
        "protocol": {
            "case_count": len(cases),
            "sets": [name for name, _, _ in SETS],
            "gt_used_for_offline_development_only": True,
            "holdout4_reserved_for_frozen_validation": True,
        },
        "top_rules": results[:100],
    }
    output = OUTPUT / "v31_metric_fit_rule_search.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"case_count": len(cases), "best": results[0]}, indent=2))
    print(f">> wrote {output}")


if __name__ == "__main__":
    main()
