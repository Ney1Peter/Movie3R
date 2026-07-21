#!/usr/bin/env python3
"""Audit whether the V32 VGGT trigger represents difficulty or dataset identity."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from v25_holdout_rotation_validation import angle_deg, safe_gravity  # noqa: E402
from v32_consensus_texture_safety_audit import selected_rotation  # noqa: E402


SETS = (
    (
        "original180",
        ROOT / "output/v15_wide_baseline_boundary_bridge/candidate_cache",
        ROOT / "output/v16_human_aware_rotation_residual/candidate_cache",
    ),
    (
        "holdout1",
        ROOT / "output/v25_holdout_rotation_validation/v15",
        ROOT / "output/v25_holdout_rotation_validation/v16",
    ),
    (
        "holdout2",
        ROOT / "output/v27_consensus_holdout2/v15",
        ROOT / "output/v27_consensus_holdout2/v16",
    ),
    (
        "holdout3",
        ROOT / "output/v29_rotation_rule_holdout3/v15",
        ROOT / "output/v29_rotation_rule_holdout3/v16",
    ),
    (
        "holdout4",
        ROOT / "output/v31_metric_fit_holdout4/v15",
        ROOT / "output/v31_metric_fit_holdout4/v16",
    ),
    (
        "holdout5",
        ROOT / "output/v32_texture_safe_holdout5/v15",
        ROOT / "output/v32_texture_safe_holdout5/v16",
    ),
    (
        "holdout6",
        ROOT / "output/v36_human_jump_holdout6/v15",
        ROOT / "output/v36_human_jump_holdout6/v16",
    ),
    (
        "holdout7_valid",
        ROOT / "output/v37_human_jump_holdout7/v15_valid",
        ROOT / "output/v37_human_jump_holdout7/v16_valid",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ROOT / "output/v49_vggt_difficulty_trigger_audit",
    )
    parser.add_argument("--texture_bound", type=float, default=0.05)
    return parser.parse_args()


def load_shards(root: Path, prefix: str) -> dict[str, dict]:
    rows = []
    for path in sorted(glob.glob(str(root / f"{prefix}_shard_*_of_*.json"))):
        rows.extend(json.loads(Path(path).read_text(encoding="utf-8"))["cases"])
    return {row["case_name"]: row for row in rows}


def fixed_bucket(error: float) -> str:
    if error < 10.0:
        return "lt10"
    if error < 30.0:
        return "10_30"
    if error < 60.0:
        return "30_60"
    return "ge60"


def summarize(rows: list[dict]) -> dict:
    triggered = [row for row in rows if row["triggered"]]
    return {
        "count": len(rows),
        "triggered": len(triggered),
        "trigger_rate": float(len(triggered) / max(len(rows), 1)),
        "branch_counts": dict(Counter(row["branch"] for row in rows)),
        "view_angle_mean_deg": float(np.mean([row["view_angle_deg"] for row in rows])),
        "triggered_view_angle_mean_deg": (
            float(np.mean([row["view_angle_deg"] for row in triggered]))
            if triggered
            else float("nan")
        ),
        "fixed_rotation_mean_deg": float(
            np.mean([row["fixed_error_deg"] for row in rows])
        ),
        "base_rotation_mean_deg": float(
            np.mean([row["base_error_deg"] for row in rows])
        ),
        "final_rotation_mean_deg": float(
            np.mean([row["final_error_deg"] for row in rows])
        ),
        "triggered_base_rotation_mean_deg": (
            float(np.mean([row["base_error_deg"] for row in triggered]))
            if triggered
            else float("nan")
        ),
        "triggered_final_rotation_mean_deg": (
            float(np.mean([row["final_error_deg"] for row in triggered]))
            if triggered
            else float("nan")
        ),
        "triggered_improved_over_5deg": int(
            np.sum(
                [row["final_error_deg"] + 5.0 < row["base_error_deg"] for row in triggered]
            )
        ),
        "triggered_harmed_over_5deg": int(
            np.sum(
                [row["final_error_deg"] > row["base_error_deg"] + 5.0 for row in triggered]
            )
        ),
        "pure_vggt_better_over_5deg_without_trigger": int(
            np.sum(
                [
                    row["pure_vggt_error_deg"] + 5.0 < row["base_error_deg"]
                    for row in rows
                    if not row["triggered"]
                ]
            )
        ),
    }


def markdown(report: dict) -> str:
    lines = [
        "# V49 VGGT Difficulty Trigger Audit",
        "",
        "The runtime trigger reads no source ID and no GT view angle. It uses torso residual, VGGT residual, torso/VGGT direction agreement, VGGT spread, and image texture.",
        "",
        "## By Source",
        "",
        "| Source | Count | Triggered | Rate | View angle | Triggered angle | Base R | Final R | Triggered R before/after | I/H >5deg |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for source, value in report["by_source"].items():
        lines.append(
            "| {source} | {count} | {triggered} | {rate:.1%} | {angle:.1f} | {ta:.1f} | "
            "{base:.2f} | {final:.2f} | {tb:.2f}/{tf:.2f} | {improved}/{harmed} |".format(
                source=source,
                count=value["count"],
                triggered=value["triggered"],
                rate=value["trigger_rate"],
                angle=value["view_angle_mean_deg"],
                ta=value["triggered_view_angle_mean_deg"],
                base=value["base_rotation_mean_deg"],
                final=value["final_rotation_mean_deg"],
                tb=value["triggered_base_rotation_mean_deg"],
                tf=value["triggered_final_rotation_mean_deg"],
                improved=value["triggered_improved_over_5deg"],
                harmed=value["triggered_harmed_over_5deg"],
            )
        )
    lines.extend(
        [
            "",
            "## By View-Angle Bucket",
            "",
            "| Angle | Count | Trigger rate | Base R | Final R | Triggered R before/after |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for bucket, value in report["by_angle_bucket"].items():
        lines.append(
            "| {bucket} | {count} | {rate:.1%} | {base:.2f} | {final:.2f} | {tb:.2f}/{tf:.2f} |".format(
                bucket=bucket,
                count=value["count"],
                rate=value["trigger_rate"],
                base=value["base_rotation_mean_deg"],
                final=value["final_rotation_mean_deg"],
                tb=value["triggered_base_rotation_mean_deg"],
                tf=value["triggered_final_rotation_mean_deg"],
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for set_name, v15_dir, v16_dir in SETS:
        v15 = load_shards(v15_dir, "v15_candidates")
        v16 = load_shards(v16_dir, "v16_candidates")
        names = sorted(set(v15) & set(v16))
        for name in names:
            wide = v15[name]
            human = v16[name]
            fixed = np.asarray(
                human["baselines"]["fixed_explicit"]["transform"], dtype=np.float32
            )[:3, :3]
            gt = np.asarray(
                human["baselines"]["boundary_oracle"]["transform"], dtype=np.float32
            )[:3, :3]
            base, _ = safe_gravity(human)
            final, branch, diagnostics = selected_rotation(
                fixed,
                base,
                wide,
                float(args.texture_bound),
                consensus_cap_deg=60.0,
            )
            pure_vggt = np.asarray(
                wide["windows"]["full_rgb_1p1"]["candidates"]["coarse"]["transform"],
                dtype=np.float32,
            )[:3, :3]
            record = human["record"]
            fixed_error = angle_deg(fixed, gt)
            rows.append(
                {
                    "set": set_name,
                    "case_name": name,
                    "source": str(record["source"]),
                    "angle_bucket": str(record.get("angle_bucket", "unknown")),
                    "view_angle_deg": float(record.get("view_angle_deg", float("nan"))),
                    "fixed_error_bucket": fixed_bucket(fixed_error),
                    "fixed_error_deg": fixed_error,
                    "base_error_deg": angle_deg(base, gt),
                    "final_error_deg": angle_deg(final, gt),
                    "pure_vggt_error_deg": angle_deg(pure_vggt, gt),
                    "branch": branch,
                    "triggered": branch != "torso",
                    "diagnostics": diagnostics,
                }
            )

    report = {
        "experiment": "V49 VGGT difficulty trigger audit",
        "protocol": {
            "case_count": len(rows),
            "sets": [item[0] for item in SETS],
            "runtime_source_id_used": False,
            "runtime_gt_view_angle_used": False,
            "gt_camera_use": "offline audit only",
            "texture_bound": float(args.texture_bound),
        },
        "overall": summarize(rows),
        "by_source": {
            source: summarize([row for row in rows if row["source"] == source])
            for source in sorted({row["source"] for row in rows})
        },
        "by_angle_bucket": {
            bucket: summarize([row for row in rows if row["angle_bucket"] == bucket])
            for bucket in sorted({row["angle_bucket"] for row in rows})
        },
        "by_fixed_error": {
            bucket: summarize([row for row in rows if row["fixed_error_bucket"] == bucket])
            for bucket in ("lt10", "10_30", "30_60", "ge60")
        },
        "by_source_angle": {
            source: {
                bucket: summarize(
                    [
                        row
                        for row in rows
                        if row["source"] == source and row["angle_bucket"] == bucket
                    ]
                )
                for bucket in sorted({row["angle_bucket"] for row in rows})
            }
            for source in sorted({row["source"] for row in rows})
        },
        "cases": rows,
    }
    json_path = args.output_dir / "v49_vggt_difficulty_trigger_audit.json"
    md_path = args.output_dir / "v49_vggt_difficulty_trigger_audit.md"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    md_path.write_text(markdown(report), encoding="utf-8")
    print(markdown(report))
    print(f">> wrote {json_path}")
    print(f">> wrote {md_path}")


if __name__ == "__main__":
    main()
