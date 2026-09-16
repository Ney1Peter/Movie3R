#!/usr/bin/env python3
"""Aggregate the EgoBody alignment-plus-continued-state counterfactual.

The script consumes only completed evaluator JSON files, preserves the
43-recording macro used by the frozen EgoBody protocol, and compares the new
counterfactual with the existing alignment-plus-reinitialized-state row on
exactly the same 129 cases.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
METHOD = "m3_alignment_continued"
REFERENCE = "m3_b0_only"
METRICS = {
    "W-MPJPE_mm": ("multi_thumbs_named_provisional", "w_mpjpe_mm", "mean"),
    "WA-MPJPE_mm": ("multi_thumbs_named_provisional", "wa_mpjpe_mm", "mean"),
    "MPJPE_mm": ("multi_thumbs_named_provisional", "mpjpe_mm", "mean"),
    "PA-MPJPE_mm": ("multi_thumbs_named_provisional", "pa_mpjpe_mm", "mean"),
    "MPVPE_mm": ("multi_thumbs_named_provisional", "mpvpe_mm", "mean"),
    "ATE_Sim3_m": ("multi_thumbs_named_provisional", "ate_sim3_m", "mean"),
    "ATE_SE3_m": ("multi_thumbs_named_provisional", "ate_se3_m", "mean"),
    "Boundary_camera_t_m": ("camera", "first_post_translation_m"),
    "Boundary_camera_R_deg": ("camera", "first_post_rotation_deg"),
    "Boundary_root_m": ("fixed_world", "first_post_root_m", "mean"),
    "Post_root_m": ("fixed_world", "post_root_m", "mean"),
    "Seam_camera_t_m": ("cut_seam", "camera_translation_excess_m"),
    "Seam_camera_R_deg": ("cut_seam", "camera_rotation_excess_deg"),
    "Seam_root_m": ("cut_seam", "root_excess_m"),
    "IDF1": ("identity", "idf1"),
    "IDs": ("identity", "ids_total"),
    "Coverage": ("coverage", "coverage"),
}
CORE = ("W-MPJPE_mm", "WA-MPJPE_mm", "ATE_Sim3_m", "IDF1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", type=Path, required=True)
    parser.add_argument("--reference-case-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260908)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nested(value: dict[str, Any], path: tuple[str, ...]) -> float | None:
    current: Any = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    if current is None:
        return None
    number = float(current)
    return number if math.isfinite(number) else None


def angle_stratum(case_id: str) -> str:
    for value in ("small", "medium", "extreme"):
        if f"_{value}_" in case_id:
            return value
    raise ValueError(f"cannot infer angle stratum from {case_id}")


def mean(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def recording_macro(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["recording"])].append(row)
    if len(grouped) != 43 or any(len(values) != 3 for values in grouped.values()):
        raise ValueError("expected 43 recordings with exactly three cases each")
    return [
        {
            "recording": recording,
            "case_count": len(values),
            **{
                metric: mean([value.get(metric) for value in values])
                for metric in METRICS
            },
        }
        for recording, values in sorted(grouped.items())
    ]


def bootstrap(values: list[float], draws: int, rng: np.random.Generator) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    sampled = array[rng.integers(0, len(array), size=(draws, len(array)))].mean(axis=1)
    return [float(np.percentile(sampled, 2.5)), float(np.percentile(sampled, 97.5))]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.bootstrap_draws < 1000:
        raise ValueError("bootstrap-draws must be at least 1000")
    paths = sorted(args.evaluations.resolve().glob("*.evaluation.json"))
    if len(paths) != 129:
        raise ValueError(f"expected 129 evaluations, found {len(paths)}")
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("errors") or set(payload.get("methods", {})) != {METHOD}:
            raise ValueError(f"incomplete evaluation: {path}")
        case_id = str(payload["case_id"])
        record = payload["record_runtime_fields"]
        result = payload["methods"][METHOD]
        rows.append({
            "case_id": case_id,
            "recording": str(record["capture"]),
            "angle_stratum": angle_stratum(case_id),
            **{metric: nested(result, key_path) for metric, key_path in METRICS.items()},
        })
    if len({row["case_id"] for row in rows}) != 129:
        raise ValueError("duplicate evaluated case")

    with args.reference_case_csv.resolve().open(newline="", encoding="utf-8") as handle:
        reference_rows = [
            row for row in csv.DictReader(handle) if row.get("name") == REFERENCE
        ]
    if len(reference_rows) != 129:
        raise ValueError(f"expected 129 {REFERENCE} rows, found {len(reference_rows)}")
    references: dict[str, dict[str, Any]] = {}
    for row in reference_rows:
        references[row["case_id"]] = {
            "case_id": row["case_id"],
            "recording": row["recording"],
            "angle_stratum": row["angle_stratum"],
            **{
                metric: (
                    float(row[metric]) if row.get(metric) not in {None, ""} else None
                )
                for metric in METRICS
            },
        }
    if set(references) != {row["case_id"] for row in rows}:
        raise ValueError("new and reference case sets differ")

    new_recordings = recording_macro(rows)
    reference_recordings = recording_macro(list(references.values()))
    new_by_recording = {row["recording"]: row for row in new_recordings}
    reference_by_recording = {row["recording"]: row for row in reference_recordings}
    rng = np.random.default_rng(args.seed)
    summary: dict[str, Any] = {}
    for metric_index, metric in enumerate(METRICS):
        new_values = [float(row[metric]) for row in new_recordings if row[metric] is not None]
        reference_values = [
            float(row[metric]) for row in reference_recordings if row[metric] is not None
        ]
        paired = []
        for recording in sorted(new_by_recording):
            new_value = new_by_recording[recording].get(metric)
            reference_value = reference_by_recording[recording].get(metric)
            if new_value is None or reference_value is None:
                continue
            # Positive gain always favours the reinitialized-state reference.
            paired.append(
                float(new_value - reference_value)
                if metric != "IDF1"
                else float(reference_value - new_value)
            )
        metric_rng = np.random.default_rng(rng.integers(0, 2**63 - 1) + metric_index)
        summary[metric] = {
            "alignment_continued_mean": float(np.mean(new_values)) if new_values else None,
            "alignment_reinitialized_mean": (
                float(np.mean(reference_values)) if reference_values else None
            ),
            "paired_recording_count": len(paired),
            "reinitialized_state_gain_mean": float(np.mean(paired)) if paired else None,
            "reinitialized_state_gain_ci95": (
                bootstrap(paired, args.bootstrap_draws, metric_rng) if paired else [None, None]
            ),
        }

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "alignment_continued_case_metrics.csv", rows)
    write_csv(output / "alignment_continued_recording_metrics.csv", new_recordings)
    payload = {
        "schema_version": "Shot3R-EgoBody-alignment-continued-aggregate-v1",
        "status": "complete",
        "method": METHOD,
        "reference": REFERENCE,
        "aggregation": "mean over three cases within each recording, then equal-weight mean over 43 recordings",
        "case_count": len(rows),
        "recording_count": len(new_recordings),
        "bootstrap": {
            "draws": args.bootstrap_draws,
            "seed": args.seed,
            "unit": "recording",
        },
        "inputs": {
            "evaluation_directory": str(args.evaluations.resolve()),
            "evaluation_sha256": {path.name: sha256(path) for path in paths},
            "reference_case_csv": str(args.reference_case_csv.resolve()),
            "reference_case_csv_sha256": sha256(args.reference_case_csv.resolve()),
        },
        "summary": summary,
    }
    temporary = (output / "alignment_continued_summary.json.partial")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, output / "alignment_continued_summary.json")

    lines = [
        "% Generated from the completed 129-case EgoBody counterfactual.",
        "Configuration & W (mm) $\\downarrow$ & WA (mm) $\\downarrow$ & ATE (m) $\\downarrow$ & IDF1 $\\uparrow$ \\\\",
        "\\midrule",
        (
            "Alignment + continued state & "
            f"{summary['W-MPJPE_mm']['alignment_continued_mean']:.1f} & "
            f"{summary['WA-MPJPE_mm']['alignment_continued_mean']:.1f} & "
            f"{summary['ATE_Sim3_m']['alignment_continued_mean']:.3f} & "
            f"{summary['IDF1']['alignment_continued_mean']:.3f} \\\\"
        ),
        (
            "Alignment + reinitialized state & "
            f"{summary['W-MPJPE_mm']['alignment_reinitialized_mean']:.1f} & "
            f"{summary['WA-MPJPE_mm']['alignment_reinitialized_mean']:.1f} & "
            f"{summary['ATE_Sim3_m']['alignment_reinitialized_mean']:.3f} & "
            f"{summary['IDF1']['alignment_reinitialized_mean']:.3f} \\\\"
        ),
    ]
    (output / "alignment_state_pair.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "core": {key: summary[key] for key in CORE}}, indent=2))


if __name__ == "__main__":
    main()
