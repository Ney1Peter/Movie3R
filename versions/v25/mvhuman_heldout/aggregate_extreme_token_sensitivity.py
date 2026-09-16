#!/usr/bin/env python3
"""Aggregate the controlled MVHuman extreme-view token sensitivity study."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


MODES = (
    "full",
    "no_semantic",
    "no_alignment",
    "no_momentum",
    "semantic_only",
    "alignment_only",
    "semantic_alignment",
)
METHOD = "m15_bridge3r_full"
METRICS = (
    ("PA-MPJPE (mm)", "pa_mpjpe_body12_mm"),
    ("Anchor-MPJPE (mm)", "first_shot_anchor_mpjpe_body12_mm"),
    ("Anchor root (mm)", "first_shot_anchor_root_error_mm"),
    ("Seam root (mm)", "seam_root_excess_mm"),
    ("Camera rot. (deg)", "post_camera_relative_rotation_deg"),
    ("Camera trans. (m)", "post_camera_relative_translation_m"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    return parser.parse_args()


def finite_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float("nan")


def main() -> None:
    args = parse_args()
    root = args.input_root.resolve()
    summary: dict[str, object] = {
        "schema_version": "Shot3R-MVHuman-extreme-token-sensitivity-aggregate-v1",
        "diagnostic_only": True,
        "token_masks_are_inference_time": True,
        "transition_source": "fixed boundary at frame index 74",
        "aggregation": "case-macro mean over the 10 pre-defined extreme-view cases",
        "method_key": METHOD,
        "modes": {},
    }
    csv_rows = []
    for mode in MODES:
        paths = sorted((root / mode / "metrics").glob("mvh150_*.json"))
        if len(paths) != 10:
            raise ValueError(f"{mode}: expected 10 metric files, found {len(paths)}")
        collected = {key: [] for _, key in METRICS}
        cases = []
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("angle_stratum") != "extreme":
                raise ValueError(f"{path} is not an extreme-view case")
            method = payload["methods"][METHOD]
            if method.get("status") != "ok":
                raise ValueError(f"{path}: {METHOD} status is not ok")
            cases.append(payload["case_id"])
            for _, key in METRICS:
                value = method["metrics"][key]["mean"]
                collected[key].append(float(value))
        aggregate = {key: finite_mean(values) for key, values in collected.items()}
        summary["modes"][mode] = {"case_count": len(cases), "case_ids": cases, "metrics": aggregate}
        row = {"mode": mode, "case_count": len(cases)}
        row.update(aggregate)
        csv_rows.append(row)

    json_path = root / "aggregate.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    csv_path = root / "aggregate.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    labels = {
        "full": "Full three-token representation",
        "no_semantic": "Without semantic token",
        "no_alignment": "Without alignment token",
        "no_momentum": "Without temporal-context token",
        "semantic_only": "Semantic token only",
        "alignment_only": "Alignment token only",
        "semantic_alignment": "Semantic + alignment tokens",
    }
    tex = [
        "% Inference-time sensitivity of one frozen checkpoint; not an independently trained ablation.",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Token input & PA & Anchor & Root & Seam & Cam. rot. & Cam. trans. \\\\",
        "\\midrule",
    ]
    for row in csv_rows:
        values = [row[key] for _, key in METRICS]
        tex.append(
            f"{labels[row['mode']]} & {values[0]:.1f} & {values[1]:.1f} & {values[2]:.1f} & "
            f"{values[3]:.1f} & {values[4]:.1f} & {values[5]:.3f} \\\\"
        )
    tex.extend(["\\bottomrule", "\\end{tabular}"])
    tex_path = root / "aggregate_table.tex"
    tex_path.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "tex": str(tex_path)}, indent=2))


if __name__ == "__main__":
    main()
