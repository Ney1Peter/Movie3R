#!/usr/bin/env python3
"""Select a safer cap for the V32 positive torso/VGGT consensus branch."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

from v25_holdout_rotation_validation import angle_deg, safe_gravity
from v32_consensus_texture_safety_audit import selected_rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "output/v34_consensus_cap_safety"
TEXTURE_BOUND = 0.05
SETS = (
    (
        "original180",
        REPO_ROOT / "output/v15_wide_baseline_boundary_bridge/candidate_cache",
        REPO_ROOT / "output/v16_human_aware_rotation_residual/candidate_cache",
    ),
    (
        "holdout1",
        REPO_ROOT / "output/v25_holdout_rotation_validation/v15",
        REPO_ROOT / "output/v25_holdout_rotation_validation/v16",
    ),
    (
        "holdout2",
        REPO_ROOT / "output/v27_consensus_holdout2/v15",
        REPO_ROOT / "output/v27_consensus_holdout2/v16",
    ),
    (
        "holdout3",
        REPO_ROOT / "output/v29_rotation_rule_holdout3/v15",
        REPO_ROOT / "output/v29_rotation_rule_holdout3/v16",
    ),
    (
        "holdout4",
        REPO_ROOT / "output/v31_metric_fit_holdout4/v15",
        REPO_ROOT / "output/v31_metric_fit_holdout4/v16",
    ),
    (
        "holdout5",
        REPO_ROOT / "output/v32_texture_safe_holdout5/v15",
        REPO_ROOT / "output/v32_texture_safe_holdout5/v16",
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
    cap60 = np.asarray([row["cap60_error"] for row in rows])
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
        "rescued_vs_cap60": int(np.sum((cap60 > 45.0) & (final <= 45.0))),
        "introduced_vs_cap60": int(np.sum((cap60 <= 45.0) & (final > 45.0))),
        "harmful_over_5deg_vs_cap60": int(np.sum(final > cap60 + 5.0)),
        "improved_over_5deg_vs_cap60": int(np.sum(final + 5.0 < cap60)),
    }


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
            cap60, _, _ = selected_rotation(
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
                    "fixed_error": angle_deg(fixed, gt),
                    "torso_error": angle_deg(torso, gt),
                    "cap60_error": angle_deg(cap60, gt),
                }
            )

    results = []
    for cap in (10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 45.0, 60.0):
        rows = []
        changed = []
        branches: dict[str, int] = {}
        for case in cases:
            final, branch, diagnostics = selected_rotation(
                case["fixed"],
                case["torso"],
                case["wide"],
                TEXTURE_BOUND,
                consensus_cap_deg=cap,
            )
            final_error = angle_deg(final, case["gt"])
            branches[branch] = branches.get(branch, 0) + 1
            row = {**case, "final_error": final_error}
            rows.append(row)
            if abs(final_error - case["cap60_error"]) > 1e-4:
                changed.append(
                    {
                        "set": case["set"],
                        "case_name": case["case_name"],
                        "source": case["source"],
                        "cap60_error": case["cap60_error"],
                        "final_error": final_error,
                        "diagnostics": diagnostics,
                    }
                )
        results.append(
            {
                "consensus_cap_deg": cap,
                "branch_counts": branches,
                "changed_count": len(changed),
                "changed_cases": changed,
                "overall": metrics(rows),
                "by_set": {
                    set_name: metrics([row for row in rows if row["set"] == set_name])
                    for set_name in sorted({row["set"] for row in rows})
                },
            }
        )

    report = {
        "experiment": "V34 positive-consensus cap safety audit",
        "protocol": {
            "case_count": len(cases),
            "texture_bound": TEXTURE_BOUND,
            "gt_used_for_offline_development_only": True,
            "holdout6_reserved_for_frozen_validation": True,
        },
        "results": results,
    }
    output = OUTPUT / "v34_consensus_cap_safety_audit.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            [
                {
                    "cap": row["consensus_cap_deg"],
                    "changed": row["changed_count"],
                    "overall": row["overall"],
                }
                for row in results
            ],
            indent=2,
        )
    )
    print(f">> wrote {output}")


if __name__ == "__main__":
    main()
