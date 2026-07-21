#!/usr/bin/env python3
"""Validate the frozen V36 adaptive consensus cap on holdout 6."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from v25_holdout_rotation_validation import angle_deg, safe_gravity
from v29_frozen_explicit_rule_validation import load_shards
from v32_consensus_texture_safety_audit import selected_rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "output/v36_human_jump_holdout6"
TEXTURE_BOUND = 0.05
HUMAN_JUMP_BOUND_DEG = 30.0
LOW_JUMP_CONSENSUS_CAP_DEG = 20.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v15_dir", type=Path, default=DEFAULT_ROOT / "v15")
    parser.add_argument("--v16_dir", type=Path, default=DEFAULT_ROOT / "v16")
    parser.add_argument(
        "--output_dir", type=Path, default=DEFAULT_ROOT / "frozen_rule_validation"
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
    fixed = np.asarray([row["fixed_error"] for row in rows])
    torso = np.asarray([row["torso_error"] for row in rows])
    v32 = np.asarray([row["v32_error"] for row in rows])
    final = np.asarray([row["v36_error"] for row in rows])
    return {
        "count": len(rows),
        "fixed_rotation_deg": distribution(fixed),
        "torso_rotation_deg": distribution(torso),
        "v32_rotation_deg": distribution(v32),
        "v36_rotation_deg": distribution(final),
        "v32_catastrophic_rate": float(np.mean(v32 > 45.0)),
        "v36_catastrophic_rate": float(np.mean(final > 45.0)),
        "v36_rescued_vs_fixed": int(np.sum((fixed > 45.0) & (final <= 45.0))),
        "v36_introduced_vs_fixed": int(np.sum((fixed <= 45.0) & (final > 45.0))),
        "v36_rescued_vs_torso": int(np.sum((torso > 45.0) & (final <= 45.0))),
        "v36_introduced_vs_torso": int(np.sum((torso <= 45.0) & (final > 45.0))),
        "v36_rescued_vs_v32": int(np.sum((v32 > 45.0) & (final <= 45.0))),
        "v36_introduced_vs_v32": int(np.sum((v32 <= 45.0) & (final > 45.0))),
        "v36_harmful_over_5deg_vs_v32": int(np.sum(final > v32 + 5.0)),
        "v36_improved_over_5deg_vs_v32": int(np.sum(final + 5.0 < v32)),
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
        torso, gravity = safe_gravity(human)
        candidate_key = (
            "fixed_torso_motion_gravity_1f_resolve_t"
            if gravity["accepted"]
            else "fixed_torso_motion_1f_resolve_t"
        )
        human_jump = float(
            human["fixed_candidates"][candidate_key]["human_torso_jump_deg"]
        )
        v32, branch, diagnostics = selected_rotation(
            fixed, torso, wide, TEXTURE_BOUND, consensus_cap_deg=60.0
        )
        adapted = bool(branch == "consensus" and human_jump < HUMAN_JUMP_BOUND_DEG)
        final = (
            selected_rotation(
                fixed,
                torso,
                wide,
                TEXTURE_BOUND,
                consensus_cap_deg=LOW_JUMP_CONSENSUS_CAP_DEG,
            )[0]
            if adapted
            else v32
        )
        rows.append(
            {
                "case_name": name,
                "source": human["record"]["source"],
                "fixed_error": angle_deg(fixed, gt),
                "torso_error": angle_deg(torso, gt),
                "v32_error": angle_deg(v32, gt),
                "v36_error": angle_deg(final, gt),
                "v32_branch": branch,
                "adapted": adapted,
                "changed": angle_deg(v32, final) > 1e-4,
                "human_torso_jump_deg": human_jump,
                "diagnostics": diagnostics,
                "v32_transform_rotation": v32.tolist(),
                "v36_transform_rotation": final.tolist(),
            }
        )

    report = {
        "experiment": "V36 frozen human-jump adaptive consensus validation",
        "protocol": {
            "texture_bound": TEXTURE_BOUND,
            "human_jump_bound_deg": HUMAN_JUMP_BOUND_DEG,
            "low_jump_consensus_cap_deg": LOW_JUMP_CONSENSUS_CAP_DEG,
            "thresholds_frozen_before_holdout6": True,
            "gt_runtime_information": False,
            "post_cut_frames": 1,
        },
        "adapted_count": int(sum(row["adapted"] for row in rows)),
        "changed_count": int(sum(row["changed"] for row in rows)),
        "overall": aggregate(rows),
        "by_source": {
            source: aggregate([row for row in rows if row["source"] == source])
            for source in sorted({row["source"] for row in rows})
        },
        "adapted_cases": [row for row in rows if row["adapted"]],
        "cases": rows,
    }
    output = args.output_dir / "v36_frozen_human_jump_consensus_validation.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "adapted_count": report["adapted_count"],
                "changed_count": report["changed_count"],
                "overall": report["overall"],
                "by_source": report["by_source"],
                "adapted_cases": [
                    {
                        "case_name": row["case_name"],
                        "source": row["source"],
                        "v32_error": row["v32_error"],
                        "v36_error": row["v36_error"],
                        "human_torso_jump_deg": row["human_torso_jump_deg"],
                    }
                    for row in report["adapted_cases"]
                ],
            },
            indent=2,
        )
    )
    print(f">> wrote {output}")


if __name__ == "__main__":
    main()
