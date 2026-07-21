#!/usr/bin/env python3
"""Validate the frozen V29 explicit fallback rule on an unseen cache."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from v25_holdout_rotation_validation import (
    angle_deg,
    capped_rotation,
    safe_gravity,
    v24_rotation,
)
from v29_cross_holdout_explicit_rule_audit import (
    background_tail_trigger,
    low_torso_trigger,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "output" / "v29_rotation_rule_holdout3"
LOW_CONFIG = {
    "min_torso": 5.0,
    "max_texture": 0.03,
    "max_spread": 10.0,
    "max_fit": 0.5,
    "min_inlier": 0.8,
    "cap_deg": 60.0,
    "min_direction_cosine": -0.5,
    "min_full_residual": 40.0,
}
BACKGROUND_CONFIG = {
    "min_torso": 30.0,
    "min_full_residual": 100.0,
    "max_background_residual": 105.0,
    "max_background_spread": 1.0,
    "cap_deg": 45.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v15_dir", type=Path, default=DEFAULT_ROOT / "v15")
    parser.add_argument("--v16_dir", type=Path, default=DEFAULT_ROOT / "v16")
    parser.add_argument(
        "--output_dir", type=Path, default=DEFAULT_ROOT / "frozen_rule_validation"
    )
    return parser.parse_args()


def load_shards(root: Path, prefix: str) -> dict[str, dict]:
    rows: list[dict] = []
    for path in sorted(glob.glob(str(root / f"{prefix}_shard_*_of_*.json"))):
        rows.extend(json.loads(Path(path).read_text(encoding="utf-8"))["cases"])
    output = {row["case_name"]: row for row in rows}
    if not rows or len(output) != len(rows):
        raise RuntimeError(f"Invalid {prefix} cache: {len(rows)}/{len(output)}")
    return output


def distribution(values: np.ndarray) -> dict:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
    }


def aggregate(rows: list[dict]) -> dict:
    base = np.asarray([row["v24_rotation_error_deg"] for row in rows])
    final = np.asarray([row["v29_rotation_error_deg"] for row in rows])
    return {
        "count": len(rows),
        "v24_rotation_deg": distribution(base),
        "v29_rotation_deg": distribution(final),
        "v24_catastrophic_rate": float(np.mean(base > 45.0)),
        "v29_catastrophic_rate": float(np.mean(final > 45.0)),
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
        background = wide["windows"]["background_only_1p1"]
        full_rotation = np.asarray(
            full["candidates"]["coarse"]["transform"], dtype=np.float32
        )[:3, :3]
        background_rotation = np.asarray(
            background["candidates"]["coarse"]["transform"], dtype=np.float32
        )[:3, :3]
        full_metric = full["candidates"]["metric_full_full"]
        background_metric = background["candidates"]["metric_full_full"]
        trigger_case = {
            "v24_accepted": bool(diagnostics["v24_accepted"]),
            "torso_residual": float(diagnostics["torso_residual_deg"]),
            "direction_cosine": float(diagnostics["residual_direction_cosine"]),
            "texture": float(wide["texture_score"]),
            "full_spread": float(full["rotation_consensus"]["spread_deg"]),
            "full_residual": float(diagnostics["vggt_residual_deg"]),
            "full_fit": float(full_metric["fit_residual_median_m"]),
            "full_inlier": float(full_metric["robust_inlier_ratio"]),
            "full_count": int(full_metric["correspondence_count"]),
            "background_residual": angle_deg(background_rotation, fixed),
            "background_spread": float(
                background["rotation_consensus"]["spread_deg"]
            ),
            "background_fit": float(background_metric["fit_residual_median_m"]),
            "background_inlier": float(background_metric["robust_inlier_ratio"]),
        }
        low_active = low_torso_trigger(trigger_case, LOW_CONFIG)
        background_active = (
            not low_active
            and background_tail_trigger(trigger_case, BACKGROUND_CONFIG)
        )
        final = v24
        selected = "v24"
        if low_active:
            final = capped_rotation(v24, full_rotation, LOW_CONFIG["cap_deg"])
            selected = "low_torso_full_rgb_1p1"
        elif background_active:
            final = capped_rotation(
                v24, background_rotation, BACKGROUND_CONFIG["cap_deg"]
            )
            selected = "large_torso_background_1p1"
        rows.append(
            {
                "case_name": name,
                "source": human["record"]["source"],
                "selected": selected,
                "low_trigger": low_active,
                "background_trigger": background_active,
                "v24_rotation_error_deg": angle_deg(v24, gt),
                "v29_rotation_error_deg": angle_deg(final, gt),
                "trigger_diagnostics": trigger_case,
                "v29_transform_rotation": final.tolist(),
            }
        )

    report = {
        "experiment": "V29 frozen explicit fallback validation",
        "protocol": {
            "thresholds_frozen_before_this_cache_was_evaluated": True,
            "gt_runtime_information": False,
            "low_config": LOW_CONFIG,
            "background_config": BACKGROUND_CONFIG,
        },
        "trigger_counts": {
            "low_torso_full_rgb_1p1": int(sum(row["low_trigger"] for row in rows)),
            "large_torso_background_1p1": int(
                sum(row["background_trigger"] for row in rows)
            ),
        },
        "overall": aggregate(rows),
        "by_source": {
            source: aggregate([row for row in rows if row["source"] == source])
            for source in sorted({row["source"] for row in rows})
        },
        "triggered_cases": [row for row in rows if row["selected"] != "v24"],
        "cases": rows,
    }
    output = args.output_dir / "v29_frozen_explicit_rule_validation.json"
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
                        "v29_rotation_error_deg": row["v29_rotation_error_deg"],
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
