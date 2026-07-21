#!/usr/bin/env python3
"""Select a conservative DA3 rigid translation residual from the V50 probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from v19_da3_explicit_geometry_correction_probe import distribution  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v50_report",
        type=Path,
        default=(
            ROOT
            / "output/v50_da3_rigid_translation_prior"
            / "v50_da3_rigid_translation_prior_probe.json"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ROOT / "output/v51_safe_da3_translation_bridge",
    )
    parser.add_argument("--prior_threshold_m", type=float, default=0.50)
    parser.add_argument("--candidate", default="vector_cap020")
    return parser.parse_args()


def selected_value(row: dict, args: argparse.Namespace) -> tuple[dict, bool]:
    use_da3 = float(row["mapped_prior_delta_m"]) >= float(args.prior_threshold_m)
    key = str(args.candidate) if use_da3 else "v47_raw"
    return row["variants"][key], use_da3


def summarize(rows: list[dict], args: argparse.Namespace) -> dict:
    selected = [selected_value(row, args) for row in rows]
    values = [item[0] for item in selected]
    base = [row["variants"]["v47_raw"] for row in rows]
    delta = np.asarray(
        [
            value["camera"]["translation_m"] - old["camera"]["translation_m"]
            for value, old in zip(values, base)
        ],
        dtype=np.float64,
    )
    return {
        "count": len(rows),
        "selected_count": int(np.sum([item[1] for item in selected])),
        "selected_rate": float(np.mean([item[1] for item in selected])),
        "camera_translation_m": distribution(
            [value["camera"]["translation_m"] for value in values]
        ),
        "camera_viewing_direction_m": distribution(
            [value["camera"]["viewing_direction_m"] for value in values]
        ),
        "camera_transverse_m": distribution(
            [value["camera"]["transverse_m"] for value in values]
        ),
        "camera_rotation_deg": distribution(
            [value["camera"]["rotation_deg"] for value in values]
        ),
        "human_motion_error_m": distribution(
            [value["human"]["root_motion_error_m"] for value in values]
        ),
        "scene_trimmed_mean_m": distribution(
            [value["scene"]["trimmed_mean_m"] for value in values]
        ),
        "correction_m": distribution([value["correction_m"] for value in values]),
        "improved_over_005m": int(np.sum(delta < -0.05)),
        "harmed_over_005m": int(np.sum(delta > 0.05)),
        "mean_translation_delta_m": float(np.mean(delta)),
        "camera_catastrophic_rate": float(
            np.mean(
                [
                    value["camera"]["translation_m"] > 2.0
                    or value["camera"]["rotation_deg"] > 45.0
                    for value in values
                ]
            )
        ),
        "joint_failure_rate": float(
            np.mean(
                [
                    value["camera"]["translation_m"] > 2.0
                    or value["camera"]["rotation_deg"] > 45.0
                    or value["human"]["root_motion_error_m"] > 0.50
                    or value["scene"]["trimmed_mean_m"] > 1.0
                    for value in values
                ]
            )
        ),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.v50_report.read_text(encoding="utf-8"))
    rows = payload["cases"]
    exported = []
    for row in rows:
        value, use_da3 = selected_value(row, args)
        exported.append(
            {
                "case_name": row["case_name"],
                "source": row["source"],
                "use_da3": use_da3,
                "mapped_prior_delta_m": row["mapped_prior_delta_m"],
                "selected_variant": str(args.candidate) if use_da3 else "v47_raw",
                "result": value,
            }
        )
    report = {
        "experiment": "V51 safe DA3 rigid translation bridge",
        "case_count": len(rows),
        "protocol": {
            "base": "V47 raw Human3R geometry with V32 rotation",
            "da3_role": "camera translation prior only",
            "prior_disagreement_threshold_m": float(args.prior_threshold_m),
            "maximum_translation_residual_m": 0.20,
            "post_cut_frames": 1,
            "local_geometry_scaling": False,
            "shot_transform": "one rigid SE3",
            "runtime_gt": False,
        },
        "overall": summarize(rows, args),
        "by_source": {
            source: summarize([row for row in rows if row["source"] == source], args)
            for source in sorted({row["source"] for row in rows})
        },
        "cases": exported,
    }
    output = args.output_dir / "v51_safe_da3_translation_bridge.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"overall": report["overall"], "by_source": report["by_source"]}, indent=2))
    print(f">> wrote {output}")


if __name__ == "__main__":
    main()
