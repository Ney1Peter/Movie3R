#!/usr/bin/env python3
"""Search conservative explicit VGGT fallback rules across development caches."""

from __future__ import annotations

import argparse
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
DEFAULT_OUTPUT = REPO_ROOT / "output" / "v29_explicit_rule_audit"
DEFAULT_SETS = (
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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_shards(root: Path, prefix: str) -> dict[str, dict]:
    rows: list[dict] = []
    for path in sorted(glob.glob(str(root / f"{prefix}_shard_*_of_*.json"))):
        rows.extend(json.loads(Path(path).read_text(encoding="utf-8"))["cases"])
    output = {row["case_name"]: row for row in rows}
    if not rows or len(output) != len(rows):
        raise RuntimeError(f"Invalid {prefix} cache at {root}: {len(rows)}/{len(output)}")
    return output


def load_cases() -> list[dict]:
    cases: list[dict] = []
    for set_name, v15_dir, v16_dir in DEFAULT_SETS:
        v15 = load_shards(v15_dir, "v15_candidates")
        v16 = load_shards(v16_dir, "v16_candidates")
        names = sorted(set(v15) & set(v16))
        if len(names) != len(v15) or len(names) != len(v16):
            raise RuntimeError(f"V15/V16 mismatch for {set_name}: {len(v15)}/{len(v16)}")
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
            background = wide["windows"]["background_only_1p1"]
            full_rotation = np.asarray(
                full["candidates"]["coarse"]["transform"], dtype=np.float32
            )[:3, :3]
            background_rotation = np.asarray(
                background["candidates"]["coarse"]["transform"], dtype=np.float32
            )[:3, :3]
            full_metric = full["candidates"]["metric_full_full"]
            background_metric = background["candidates"]["metric_full_full"]
            cases.append(
                {
                    "set": set_name,
                    "case_name": name,
                    "source": human["record"]["source"],
                    "fixed": fixed,
                    "gt": gt,
                    "base": base,
                    "base_error": angle_deg(base, gt),
                    "full_rotation": full_rotation,
                    "background_rotation": background_rotation,
                    "texture": float(wide["texture_score"]),
                    "torso_residual": float(diagnostics["torso_residual_deg"]),
                    "direction_cosine": float(diagnostics["residual_direction_cosine"]),
                    "full_residual": float(diagnostics["vggt_residual_deg"]),
                    "full_spread": float(full["rotation_consensus"]["spread_deg"]),
                    "background_residual": angle_deg(background_rotation, fixed),
                    "background_spread": float(
                        background["rotation_consensus"]["spread_deg"]
                    ),
                    "full_fit": float(full_metric["fit_residual_median_m"]),
                    "full_inlier": float(full_metric["robust_inlier_ratio"]),
                    "full_count": int(full_metric["correspondence_count"]),
                    "background_fit": float(
                        background_metric["fit_residual_median_m"]
                    ),
                    "background_inlier": float(
                        background_metric["robust_inlier_ratio"]
                    ),
                    "v24_accepted": bool(diagnostics["v24_accepted"]),
                }
            )
    return cases


def low_torso_trigger(case: dict, config: dict) -> bool:
    return bool(
        not case["v24_accepted"]
        and config["min_torso"] <= case["torso_residual"] <= 30.0
        and config.get("min_direction_cosine", -1.0)
        <= case["direction_cosine"]
        < 0.0
        and case["texture"] <= config["max_texture"]
        and case["full_spread"] <= config["max_spread"]
        and config.get("min_full_residual", 30.0)
        <= case["full_residual"]
        <= 100.0
        and case["full_fit"] <= config["max_fit"]
        and case["full_inlier"] >= config["min_inlier"]
        and case["full_count"] >= 100
    )


def background_tail_trigger(case: dict, config: dict) -> bool:
    return bool(
        not case["v24_accepted"]
        and case["torso_residual"] >= config["min_torso"]
        and case["full_residual"] >= config["min_full_residual"]
        and case["background_residual"] <= config["max_background_residual"]
        and case["background_residual"] >= case["torso_residual"] + 5.0
        and case["background_spread"] <= config["max_background_spread"]
        and case["background_fit"] <= 1.0
        and case["background_inlier"] >= 0.35
    )


def metrics(rows: list[dict], method: str) -> dict:
    errors = np.asarray([row[method] for row in rows], dtype=np.float64)
    baseline = np.asarray([row["base_error"] for row in rows], dtype=np.float64)
    return {
        "count": int(errors.size),
        "mean": float(errors.mean()),
        "median": float(np.median(errors)),
        "p90": float(np.quantile(errors, 0.90)),
        "p95": float(np.quantile(errors, 0.95)),
        "catastrophic_count": int(np.sum(errors > 45.0)),
        "rescued_catastrophic_count": int(
            np.sum((baseline > 45.0) & (errors <= 45.0))
        ),
        "introduced_catastrophic_count": int(
            np.sum((baseline <= 45.0) & (errors > 45.0))
        ),
        "harmful_over_5deg_count": int(np.sum(errors > baseline + 5.0)),
        "good_case_harmful_count": int(
            np.sum((baseline < 10.0) & (errors > baseline + 5.0))
        ),
    }


def evaluate(cases: list[dict], low: dict | None, background: dict | None) -> dict:
    rows = []
    active_low = []
    active_background = []
    for case in cases:
        rotation = case["base"]
        low_active = low is not None and low_torso_trigger(case, low)
        if low_active:
            rotation = capped_rotation(rotation, case["full_rotation"], low["cap_deg"])
            active_low.append(case["case_name"])
        background_active = (
            background is not None
            and not low_active
            and background_tail_trigger(case, background)
        )
        if background_active:
            rotation = capped_rotation(
                rotation, case["background_rotation"], background["cap_deg"]
            )
            active_background.append(case["case_name"])
        rows.append({**case, "proposal_error": angle_deg(rotation, case["gt"])})
    by_set = {
        set_name: metrics(
            [row for row in rows if row["set"] == set_name], "proposal_error"
        )
        for set_name in sorted({row["set"] for row in rows})
    }
    overall = metrics(rows, "proposal_error")
    return {
        "low_config": low,
        "background_config": background,
        "active_low_count": len(active_low),
        "active_background_count": len(active_background),
        "active_low_cases": active_low,
        "active_background_cases": active_background,
        "overall": overall,
        "by_set": by_set,
    }


def ranking_key(result: dict) -> tuple:
    overall = result["overall"]
    worst_introduced = max(
        value["introduced_catastrophic_count"] for value in result["by_set"].values()
    )
    worst_good_harm = max(
        value["good_case_harmful_count"] for value in result["by_set"].values()
    )
    worst_harm = max(
        value["harmful_over_5deg_count"] for value in result["by_set"].values()
    )
    return (
        worst_introduced,
        worst_good_harm,
        overall["introduced_catastrophic_count"],
        overall["good_case_harmful_count"],
        -overall["rescued_catastrophic_count"],
        worst_harm,
        overall["harmful_over_5deg_count"],
        overall["mean"],
        overall["p95"],
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases()
    baseline_rows = [{**case, "proposal_error": case["base_error"]} for case in cases]
    baseline = {
        "overall": metrics(baseline_rows, "proposal_error"),
        "by_set": {
            set_name: metrics(
                [row for row in baseline_rows if row["set"] == set_name],
                "proposal_error",
            )
            for set_name in sorted({row["set"] for row in baseline_rows})
        },
    }

    low_configs = [
        {
            "min_torso": min_torso,
            "max_texture": max_texture,
            "max_spread": max_spread,
            "max_fit": max_fit,
            "min_inlier": min_inlier,
            "cap_deg": cap_deg,
        }
        for min_torso, max_texture, max_spread, max_fit, min_inlier, cap_deg in itertools.product(
            (5.0, 7.5, 10.0),
            (0.02, 0.03, 0.05),
            (5.0, 8.0, 10.0),
            (0.3, 0.5, 0.7),
            (0.7, 0.8, 0.9),
            (45.0, 60.0),
        )
    ]
    background_configs = [
        {
            "min_torso": min_torso,
            "min_full_residual": min_full,
            "max_background_residual": max_background,
            "max_background_spread": max_spread,
            "cap_deg": cap_deg,
        }
        for min_torso, min_full, max_background, max_spread, cap_deg in itertools.product(
            (30.0, 40.0),
            (100.0, 110.0),
            (105.0, 110.0, 120.0),
            (1.0, 2.0, 5.0),
            (25.0, 45.0, 60.0),
        )
    ]

    low_results = sorted(
        (evaluate(cases, config, None) for config in low_configs), key=ranking_key
    )
    background_results = sorted(
        (evaluate(cases, None, config) for config in background_configs),
        key=ranking_key,
    )
    combined_results = sorted(
        (
            evaluate(cases, low["low_config"], background["background_config"])
            for low, background in itertools.product(
                low_results[:30], background_results[:30]
            )
        ),
        key=ranking_key,
    )
    report = {
        "experiment": "V29 cross-holdout explicit fallback rule audit",
        "protocol": {
            "case_count": len(cases),
            "sets": [name for name, _, _ in DEFAULT_SETS],
            "gt_used_for_offline_rule_development_only": True,
            "holdout3_reserved_for_frozen_validation": True,
        },
        "baseline_v24": baseline,
        "top_low_torso_rules": low_results[:50],
        "top_background_tail_rules": background_results[:50],
        "top_combined_rules": combined_results[:50],
    }
    output = args.output_dir / "v29_cross_holdout_explicit_rule_audit.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "case_count": len(cases),
                "baseline": baseline,
                "best_low": low_results[0],
                "best_background": background_results[0],
                "best_combined": combined_results[0],
            },
            indent=2,
        )
    )
    print(f">> wrote {output}")


if __name__ == "__main__":
    main()
