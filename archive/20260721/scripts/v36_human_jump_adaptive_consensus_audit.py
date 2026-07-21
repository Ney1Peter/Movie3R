#!/usr/bin/env python3
"""Use residual human-orientation discontinuity to cap VGGT consensus updates."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from v25_holdout_rotation_validation import angle_deg, safe_gravity
from v32_consensus_texture_safety_audit import selected_rotation
from v34_consensus_cap_safety_audit import SETS, load_shards


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "output/v36_human_jump_adaptive_consensus"
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
            torso, gravity = safe_gravity(human)
            candidate_key = (
                "fixed_torso_motion_gravity_1f_resolve_t"
                if gravity["accepted"]
                else "fixed_torso_motion_1f_resolve_t"
            )
            human_jump = float(
                human["fixed_candidates"][candidate_key]["human_torso_jump_deg"]
            )
            baseline, branch, _ = selected_rotation(
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
                    "wide": wide,
                    "baseline": baseline,
                    "baseline_branch": branch,
                    "human_torso_jump_deg": human_jump,
                    "fixed_error": angle_deg(fixed, gt),
                    "torso_error": angle_deg(torso, gt),
                    "baseline_error": angle_deg(baseline, gt),
                }
            )

    configs = [
        {"min_human_jump_deg": threshold, "low_jump_cap_deg": cap}
        for threshold, cap in itertools.product(
            (20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 60.0),
            (0.0, 15.0, 20.0, 25.0, 30.0, 35.0, 45.0),
        )
    ]
    results = []
    for config in configs:
        rows = []
        changed = []
        for case in cases:
            adapt = bool(
                case["baseline_branch"] == "consensus"
                and case["human_torso_jump_deg"] < config["min_human_jump_deg"]
            )
            if adapt:
                final, _, diagnostics = selected_rotation(
                    case["fixed"],
                    case["torso"],
                    case["wide"],
                    TEXTURE_BOUND,
                    consensus_cap_deg=config["low_jump_cap_deg"],
                )
            else:
                final = case["baseline"]
                diagnostics = {}
            final_error = angle_deg(final, case["gt"])
            rows.append({**case, "final_error": final_error})
            if adapt:
                changed.append(
                    {
                        "set": case["set"],
                        "case_name": case["case_name"],
                        "source": case["source"],
                        "human_torso_jump_deg": case["human_torso_jump_deg"],
                        "baseline_error": case["baseline_error"],
                        "final_error": final_error,
                        "diagnostics": diagnostics,
                    }
                )
        results.append(
            {
                "config": config,
                "changed_count": len(changed),
                "changed_cases": changed,
                "overall": metrics(rows),
                "by_set": {
                    set_name: metrics([row for row in rows if row["set"] == set_name])
                    for set_name in sorted({row["set"] for row in rows})
                },
            }
        )
    results.sort(key=ranking_key)
    report = {
        "experiment": "V36 human-jump adaptive consensus cap audit",
        "protocol": {
            "case_count": len(cases),
            "texture_bound": TEXTURE_BOUND,
            "default_consensus_cap_deg": 60.0,
            "gt_used_for_offline_development_only": True,
            "holdout6_reserved_for_frozen_validation": True,
        },
        "top_rules": results,
    }
    output = OUTPUT / "v36_human_jump_adaptive_consensus_audit.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"top10": results[:10]}, indent=2))
    print(f">> wrote {output}")


if __name__ == "__main__":
    main()
