#!/usr/bin/env python3
"""Decompose residual V36 failures into rotation and metric-translation limits."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()) if array.size else 0.0,
        "median": float(np.median(array)) if array.size else 0.0,
        "p90": float(np.percentile(array, 90)) if array.size else 0.0,
        "p95": float(np.percentile(array, 95)) if array.size else 0.0,
        "max": float(array.max()) if array.size else 0.0,
    }


def classify(case: dict) -> str:
    v36 = case["variants"]["v36"]["camera"]
    gt_rotation = case["variants"]["gt_rotation"]["camera"]
    translation = float(v36["translation_m"])
    rotation = float(v36["rotation_deg"])
    gt_rotation_translation = float(gt_rotation["translation_m"])
    rotation_gain = translation - gt_rotation_translation

    if translation <= 0.5 and rotation <= 30.0:
        return "acceptable"
    if rotation > 30.0 and rotation_gain > 0.2:
        return "rotation_dominated"
    if gt_rotation_translation > 0.5:
        if float(gt_rotation["viewing_direction_m"]) >= float(gt_rotation["transverse_m"]):
            return "metric_depth_dominated"
        return "metric_transverse_dominated"
    if rotation > 30.0:
        return "rotation_tail"
    return "mixed_moderate"


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    rows = []
    source_classes: dict[str, Counter] = defaultdict(Counter)

    for case in payload["cases"]:
        v36 = case["variants"]["v36"]
        gt_rotation = case["variants"]["gt_rotation"]
        classification = classify(case)
        source_classes[case["source"]][classification] += 1
        old_scale = float(case["root_scales"]["old"])
        new_scale = float(case["root_scales"]["new"])
        rows.append({
            "case_name": case["case_name"],
            "source": case["source"],
            "classification": classification,
            "v32_branch": case["v32_branch"],
            "adapted": bool(case["adapted"]),
            "translation_m": float(v36["camera"]["translation_m"]),
            "rotation_deg": float(v36["camera"]["rotation_deg"]),
            "viewing_direction_m": float(v36["camera"]["viewing_direction_m"]),
            "transverse_m": float(v36["camera"]["transverse_m"]),
            "gt_rotation_translation_m": float(gt_rotation["camera"]["translation_m"]),
            "gt_rotation_viewing_direction_m": float(gt_rotation["camera"]["viewing_direction_m"]),
            "gt_rotation_transverse_m": float(gt_rotation["camera"]["transverse_m"]),
            "translation_gain_from_gt_rotation_m": float(
                v36["camera"]["translation_m"] - gt_rotation["camera"]["translation_m"]
            ),
            "scene_trimmed_mean_m": float(v36["scene"]["trimmed_mean_m"]),
            "root_scale_old": old_scale,
            "root_scale_new": new_scale,
            "root_scale_ratio": new_scale / max(old_scale, 1e-8),
            "texture": float(case["diagnostics"]["texture"]),
            "torso_residual_deg": float(case["diagnostics"]["torso_residual"]),
            "vggt_residual_deg": float(case["diagnostics"]["vggt_residual"]),
            "human_torso_jump_deg": float(case["human_torso_jump_deg"]),
        })

    classification_counts = Counter(row["classification"] for row in rows)
    hard_rows = [row for row in rows if row["classification"] != "acceptable"]
    gt_rotation_limited = [row for row in rows if row["gt_rotation_translation_m"] > 0.5]
    rotation_limited = [row for row in rows if row["translation_gain_from_gt_rotation_m"] > 0.2]

    summary = {
        "experiment": "V38 final explicit metric bridge residual audit",
        "case_count": len(rows),
        "classification_counts": dict(classification_counts),
        "source_classification_counts": {
            source: dict(counts) for source, counts in sorted(source_classes.items())
        },
        "residual_metrics": {
            "v36_translation_m": stats([row["translation_m"] for row in rows]),
            "v36_rotation_deg": stats([row["rotation_deg"] for row in rows]),
            "translation_with_gt_rotation_m": stats([row["gt_rotation_translation_m"] for row in rows]),
            "translation_gain_from_gt_rotation_m": stats([
                row["translation_gain_from_gt_rotation_m"] for row in rows
            ]),
            "gt_rotation_viewing_direction_m": stats([
                row["gt_rotation_viewing_direction_m"] for row in rows
            ]),
            "gt_rotation_transverse_m": stats([row["gt_rotation_transverse_m"] for row in rows]),
        },
        "diagnostic_counts": {
            "non_acceptable": len(hard_rows),
            "translation_over_050m_even_with_gt_rotation": len(gt_rotation_limited),
            "translation_gain_over_020m_from_gt_rotation": len(rotation_limited),
        },
        "worst_translation": sorted(rows, key=lambda row: row["translation_m"], reverse=True)[:20],
        "worst_rotation": sorted(rows, key=lambda row: row["rotation_deg"], reverse=True)[:20],
        "worst_metric_translation_with_gt_rotation": sorted(
            rows, key=lambda row: row["gt_rotation_translation_m"], reverse=True
        )[:20],
        "cases": rows,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_dir / "v38_final_bridge_residual_audit.json"
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    lines = [
        "# V38 Final Bridge Residual Audit",
        "",
        f"Cases: {len(rows)}",
        "",
        "## Failure classes",
        "",
    ]
    for name, count in classification_counts.most_common():
        lines.append(f"- {name}: {count}")
    lines.extend([
        "",
        "## Diagnostic limits",
        "",
        f"- Non-acceptable cases: {len(hard_rows)}",
        f"- Translation >0.5 m even with GT rotation: {len(gt_rotation_limited)}",
        f"- Translation gain >0.2 m from GT rotation: {len(rotation_limited)}",
        "",
        "## Worst current translation",
        "",
    ])
    for row in summary["worst_translation"][:10]:
        lines.append(
            f"- {row['case_name']}: T={row['translation_m']:.3f} m, "
            f"R={row['rotation_deg']:.2f} deg, GT-R T={row['gt_rotation_translation_m']:.3f} m, "
            f"class={row['classification']}"
        )
    (args.output_dir / "V38_FINAL_BRIDGE_RESIDUAL_AUDIT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(output_json),
        "classification_counts": dict(classification_counts),
        "diagnostic_counts": summary["diagnostic_counts"],
    }))


if __name__ == "__main__":
    main()
