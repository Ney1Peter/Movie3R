#!/usr/bin/env python3
"""Audit DA3 metric translation with a fixed VGGT rotation."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "output" / "v29_rotation_rule_holdout3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v15_dir", type=Path, default=DEFAULT_ROOT / "v15")
    parser.add_argument(
        "--output_dir", type=Path, default=DEFAULT_ROOT / "da3_translation_audit"
    )
    return parser.parse_args()


def load_rows(root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(glob.glob(str(root / "v15_candidates_shard_*_of_*.json"))):
        rows.extend(json.loads(Path(path).read_text(encoding="utf-8"))["cases"])
    if not rows or len(rows) != len({row["case_name"] for row in rows}):
        raise RuntimeError(f"Invalid V15 cache: {len(rows)} rows")
    return rows


def distribution(values: np.ndarray) -> dict:
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
    }


def aggregate(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    coarse = np.asarray([row["coarse_translation_m"] for row in rows])
    da3 = np.asarray([row["da3_translation_m"] for row in rows])
    human3r = np.asarray([row["human3r_translation_m"] for row in rows])
    return {
        "coarse_translation_m": distribution(coarse),
        "da3_fixed_rotation_translation_m": distribution(da3),
        "human3r_fixed_rotation_translation_m": distribution(human3r),
        "da3_vs_coarse": {
            "mean_delta_m": float(np.mean(da3 - coarse)),
            "improved_rate": float(np.mean(da3 < coarse)),
            "harmful_over_0_1m_rate": float(np.mean(da3 > coarse + 0.1)),
            "improved_over_0_1m_rate": float(np.mean(da3 + 0.1 < coarse)),
        },
        "da3_vs_human3r": {
            "mean_delta_m": float(np.mean(da3 - human3r)),
            "improved_rate": float(np.mean(da3 < human3r)),
            "harmful_over_0_1m_rate": float(np.mean(da3 > human3r + 0.1)),
        },
        "oracle_best_of_three_m": distribution(
            np.minimum(np.minimum(coarse, da3), human3r)
        ),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_rows(args.v15_dir)
    rows = []
    for case in cases:
        for window_name, window in case["windows"].items():
            candidates = window["candidates"]
            coarse = candidates.get("coarse")
            da3 = candidates.get("da3_wide_rotation_metric_translation")
            human3r = candidates.get("wide_rotation_metric_translation_full")
            if any(
                candidate is None
                or candidate.get("fit_failed")
                or candidate.get("transform") is None
                for candidate in (coarse, da3, human3r)
            ):
                continue
            rows.append(
                {
                    "case_name": case["case_name"],
                    "source": case["record"]["source"],
                    "window": window_name,
                    "rotation_error_deg": float(
                        coarse["camera_rotation_error_deg"]
                    ),
                    "coarse_translation_m": float(
                        coarse["camera_translation_error_m"]
                    ),
                    "da3_translation_m": float(da3["camera_translation_error_m"]),
                    "human3r_translation_m": float(
                        human3r["camera_translation_error_m"]
                    ),
                    "da3_fit_residual_median_m": (
                        None
                        if "da3_fit_residual_median_m" not in da3
                        else float(da3["da3_fit_residual_median_m"])
                    ),
                    "da3_inlier_ratio": (
                        None
                        if "da3_robust_inlier_ratio" not in da3
                        else float(da3["da3_robust_inlier_ratio"])
                    ),
                }
            )

    windows = sorted({row["window"] for row in rows})
    sources = sorted({row["source"] for row in rows})
    report = {
        "experiment": "V30 DA3 fixed-rotation translation stability audit",
        "protocol": {
            "gt_used_for_evaluation_and_rotation_error_grouping_only": True,
            "da3_rotation_frozen_to_vggt_window_rotation": True,
            "da3_free_rotation_not_used": True,
        },
        "case_count": len(cases),
        "row_count": len(rows),
        "by_window": {
            window: {
                "all": aggregate([row for row in rows if row["window"] == window]),
                "rotation_le_30deg": aggregate(
                    [
                        row
                        for row in rows
                        if row["window"] == window
                        and row["rotation_error_deg"] <= 30.0
                    ]
                ),
                "rotation_le_45deg": aggregate(
                    [
                        row
                        for row in rows
                        if row["window"] == window
                        and row["rotation_error_deg"] <= 45.0
                    ]
                ),
            }
            for window in windows
        },
        "by_source_full_rgb_1p1_rotation_le_45deg": {
            source: aggregate(
                [
                    row
                    for row in rows
                    if row["source"] == source
                    and row["window"] == "full_rgb_1p1"
                    and row["rotation_error_deg"] <= 45.0
                ]
            )
            for source in sources
        },
        "rows": rows,
    }
    output = args.output_dir / "v30_da3_translation_stability_audit.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "case_count": report["case_count"],
                "by_window": report["by_window"],
                "by_source_full_rgb_1p1_rotation_le_45deg": report[
                    "by_source_full_rgb_1p1_rotation_le_45deg"
                ],
            },
            indent=2,
        )
    )
    print(f">> wrote {output}")


if __name__ == "__main__":
    main()
