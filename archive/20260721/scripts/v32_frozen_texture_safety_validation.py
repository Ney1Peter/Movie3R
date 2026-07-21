#!/usr/bin/env python3
"""Validate the frozen V32 texture-safe consensus rule on H5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from v25_holdout_rotation_validation import angle_deg, safe_gravity
from v29_frozen_explicit_rule_validation import load_shards
from v32_consensus_texture_safety_audit import selected_rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "output" / "v32_texture_safe_holdout5"
TEXTURE_BOUND = 0.05


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
    v24 = np.asarray([row["v24_error"] for row in rows])
    final = np.asarray([row["v32_error"] for row in rows])
    return {
        "count": len(rows),
        "fixed_rotation_deg": distribution(fixed),
        "torso_rotation_deg": distribution(torso),
        "v24_rotation_deg": distribution(v24),
        "v32_rotation_deg": distribution(final),
        "v24_catastrophic_rate": float(np.mean(v24 > 45.0)),
        "v32_catastrophic_rate": float(np.mean(final > 45.0)),
        "v32_rescued_vs_fixed": int(np.sum((fixed > 45.0) & (final <= 45.0))),
        "v32_introduced_vs_fixed": int(np.sum((fixed <= 45.0) & (final > 45.0))),
        "v32_rescued_vs_torso": int(np.sum((torso > 45.0) & (final <= 45.0))),
        "v32_introduced_vs_torso": int(np.sum((torso <= 45.0) & (final > 45.0))),
        "v32_harmful_over_5deg_vs_v24": int(np.sum(final > v24 + 5.0)),
        "v32_improved_over_5deg_vs_v24": int(np.sum(final + 5.0 < v24)),
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
        v24, v24_branch, _ = selected_rotation(fixed, torso, wide, None)
        final, branch, diagnostics = selected_rotation(
            fixed, torso, wide, TEXTURE_BOUND
        )
        rows.append(
            {
                "case_name": name,
                "source": human["record"]["source"],
                "fixed_error": angle_deg(fixed, gt),
                "torso_error": angle_deg(torso, gt),
                "v24_error": angle_deg(v24, gt),
                "v32_error": angle_deg(final, gt),
                "v24_branch": v24_branch,
                "v32_branch": branch,
                "changed": angle_deg(final, v24) > 1e-4,
                "diagnostics": diagnostics,
                "v32_transform_rotation": final.tolist(),
            }
        )

    report = {
        "experiment": "V32 frozen texture-safe consensus validation",
        "protocol": {
            "texture_bound": TEXTURE_BOUND,
            "threshold_frozen_before_h5_evaluation": True,
            "gt_runtime_information": False,
            "post_cut_frames": 1,
        },
        "changed_count": int(sum(row["changed"] for row in rows)),
        "overall": aggregate(rows),
        "by_source": {
            source: aggregate([row for row in rows if row["source"] == source])
            for source in sorted({row["source"] for row in rows})
        },
        "changed_cases": [row for row in rows if row["changed"]],
        "cases": rows,
    }
    output = args.output_dir / "v32_frozen_texture_safety_validation.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "changed_count": report["changed_count"],
                "overall": report["overall"],
                "by_source": report["by_source"],
                "changed_cases": [
                    {
                        "case_name": row["case_name"],
                        "source": row["source"],
                        "v24_error": row["v24_error"],
                        "v32_error": row["v32_error"],
                    }
                    for row in report["changed_cases"]
                ],
            },
            indent=2,
        )
    )
    print(f">> wrote {output}")


if __name__ == "__main__":
    main()
