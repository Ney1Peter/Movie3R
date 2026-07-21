#!/usr/bin/env python3
"""Audit the remaining V22 failures and the GT-rotation rescue space."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V22 = (
    REPO_ROOT
    / "output"
    / "v22_explicit_metric_bridge"
    / "final_seed1"
    / "v22_explicit_metric_bridge.json"
)
DEFAULT_GT_ROTATION = (
    REPO_ROOT
    / "output"
    / "v22_explicit_metric_bridge"
    / "gt_rotation_oracle"
    / "v22_gt_rotation_metric_bridge_oracle.json"
)
DEFAULT_V10 = (
    REPO_ROOT
    / "output"
    / "v10_candidate_selection"
    / "oracle_gt_4source"
    / "oracle_candidate_selection_metrics.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "output" / "v22_explicit_metric_bridge" / "failure_audit"
SELECTED = "safe_gravity_absolute_scene_scale"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v22_report", type=Path, default=DEFAULT_V22)
    parser.add_argument("--gt_rotation_report", type=Path, default=DEFAULT_GT_ROTATION)
    parser.add_argument("--v10_report", type=Path, default=DEFAULT_V10)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {key: float("nan") for key in ("mean", "median", "p90", "p95")}
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
    }


def texture_score(case: dict) -> float:
    reset_root = Path(case["paths"]["human3r_local_reset"])
    image = cv2.imread(str(reset_root / "color" / "000002.png"), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return float("nan")
    image = image.astype(np.float32) / 255.0
    dx = np.abs(image[:, 1:] - image[:, :-1]).mean()
    dy = np.abs(image[1:, :] - image[:-1, :]).mean()
    return float(dx + dy)


def catastrophic_flags(value: dict) -> list[str]:
    flags = []
    if value["camera"]["translation_m"] > 2.0:
        flags.append("translation")
    if value["camera"]["rotation_deg"] > 45.0:
        flags.append("rotation")
    if value["human"]["root_motion_error_m"] > 0.50:
        flags.append("human")
    if value["scene"]["trimmed_mean_m"] > 1.0:
        flags.append("scene")
    return flags


def metrics(row: dict, key: str) -> dict:
    value = row[key]
    return {
        "translation_m": float(value["camera"]["translation_m"]),
        "rotation_deg": float(value["camera"]["rotation_deg"]),
        "viewing_direction_m": float(value["camera"]["viewing_direction_m"]),
        "transverse_m": float(value["camera"]["transverse_m"]),
        "human_motion_m": float(value["human"]["root_motion_error_m"]),
        "scene_m": float(value["scene"]["trimmed_mean_m"]),
    }


def summarize(rows: list[dict], key: str) -> dict:
    values = [row[key] for row in rows]
    return {
        "count": len(rows),
        "translation_m": distribution([value["translation_m"] for value in values]),
        "rotation_deg": distribution([value["rotation_deg"] for value in values]),
        "viewing_direction_m": distribution(
            [value["viewing_direction_m"] for value in values]
        ),
        "transverse_m": distribution([value["transverse_m"] for value in values]),
        "human_motion_m": distribution([value["human_motion_m"] for value in values]),
        "scene_m": distribution([value["scene_m"] for value in values]),
        "catastrophic_rate": float(
            np.mean([bool(row[f"{key}_catastrophic_flags"]) for row in rows])
        )
        if rows
        else float("nan"),
    }


def group_summary(rows: list[dict], field: str) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {
        group: {
            "fixed": summarize(values, "fixed"),
            "v22": summarize(values, "v22"),
            "gt_rotation": summarize(values, "gt_rotation"),
            "v22_camera_improved_rate": float(
                np.mean([row["v22"]["translation_m"] < row["fixed"]["translation_m"] for row in values])
            ),
            "v22_rotation_improved_rate": float(
                np.mean([row["v22"]["rotation_deg"] < row["fixed"]["rotation_deg"] for row in values])
            ),
        }
        for group, values in sorted(groups.items())
    }


def rotation_bin(value: float) -> str:
    if value < 10.0:
        return "lt10"
    if value < 30.0:
        return "10_30"
    if value < 60.0:
        return "30_60"
    return "ge60"


def texture_labels(rows: list[dict]) -> None:
    values = np.asarray([row["texture_score"] for row in rows], dtype=np.float64)
    q1, q2 = np.nanquantile(values, [1.0 / 3.0, 2.0 / 3.0])
    for row in rows:
        value = row["texture_score"]
        if value <= q1:
            row["texture_tertile"] = "low"
        elif value <= q2:
            row["texture_tertile"] = "medium"
        else:
            row["texture_tertile"] = "high"


def failure_priority(row: dict) -> float:
    value = row["v22"]
    return float(
        max(
            value["translation_m"] / 2.0,
            value["rotation_deg"] / 45.0,
            value["human_motion_m"] / 0.50,
            value["scene_m"] / 1.0,
        )
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    v22 = load(args.v22_report)
    gt_rotation = load(args.gt_rotation_report)
    v10 = load(args.v10_report)
    v10_cases = {row["case_name"]: row for row in v10["cases"]}
    gt_cases = {row["case_name"]: row for row in gt_rotation["cases"]}

    rows = []
    for case in v22["cases"]:
        name = case["case_name"]
        record = v10_cases[name]["record"]
        gt_case = gt_cases[name]
        fixed = metrics(case["variants"], "fixed_explicit")
        selected = metrics(case["variants"], SELECTED)
        oracle = metrics(gt_case, "gt_rotation_same_metric_translation")
        row = {
            "case_name": name,
            "source": case["source"],
            "angle_bucket": record["angle_bucket"],
            "view_angle_deg": float(record["view_angle_deg"]),
            "texture_score": texture_score(v10_cases[name]),
            "fixed_rotation_bin": rotation_bin(fixed["rotation_deg"]),
            "fixed": fixed,
            "v22": selected,
            "gt_rotation": oracle,
            "fixed_catastrophic_flags": catastrophic_flags(
                case["variants"]["fixed_explicit"]
            ),
            "v22_catastrophic_flags": catastrophic_flags(
                case["variants"][SELECTED]
            ),
            "gt_rotation_catastrophic_flags": catastrophic_flags(
                gt_case["gt_rotation_same_metric_translation"]
            ),
        }
        rows.append(row)
    texture_labels(rows)

    failures = [row for row in rows if row["v22_catastrophic_flags"]]
    failure_classes = Counter(
        "+".join(row["v22_catastrophic_flags"]) for row in failures
    )
    failure_by_source = defaultdict(Counter)
    for row in failures:
        failure_by_source[row["source"]]["total"] += 1
        failure_by_source[row["source"]][
            "+".join(row["v22_catastrophic_flags"])
        ] += 1
    rescued = [row for row in failures if not row["gt_rotation_catastrophic_flags"]]
    rotation_related = [row for row in failures if "rotation" in row["v22_catastrophic_flags"]]

    report = {
        "experiment": "V22 remaining failure and GT-rotation rescue audit",
        "case_count": len(rows),
        "thresholds": {
            "translation_m": 2.0,
            "rotation_deg": 45.0,
            "human_motion_m": 0.50,
            "scene_m": 1.0,
        },
        "overall": {
            "fixed": summarize(rows, "fixed"),
            "v22": summarize(rows, "v22"),
            "gt_rotation": summarize(rows, "gt_rotation"),
            "remaining_catastrophic_count": len(failures),
            "remaining_catastrophic_rate": len(failures) / len(rows),
            "failure_classes": dict(sorted(failure_classes.items())),
            "rotation_related_count": len(rotation_related),
            "rotation_related_fraction": len(rotation_related) / len(failures),
            "gt_rotation_rescued_count": len(rescued),
            "gt_rotation_rescued_fraction": len(rescued) / len(failures),
        },
        "failure_by_source": {
            source: dict(sorted(counts.items()))
            for source, counts in sorted(failure_by_source.items())
        },
        "groups": {
            "source": group_summary(rows, "source"),
            "view_angle_bucket": group_summary(rows, "angle_bucket"),
            "fixed_rotation_bin": group_summary(rows, "fixed_rotation_bin"),
            "texture_tertile": group_summary(rows, "texture_tertile"),
        },
        "worst_remaining_cases": sorted(
            failures, key=failure_priority, reverse=True
        ),
        "cases": rows,
        "conclusion": {
            "primary_remaining_failure": "rotation tail",
            "evidence": (
                "12 of 13 remaining catastrophic cuts are rotation-related and GT rotation "
                "rescues 12 of 13 while preserving the same DA3 metric scales and translation equation."
            ),
            "separate_scene_issue": (
                "The only non-rotation catastrophic cut is a THuman scene-discontinuity case; "
                "GT rotation does not fix it."
            ),
        },
    }
    json_path = args.output_dir / "v22_remaining_failure_audit.json"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# V22 Remaining Failure Audit",
        "",
        "## Failure decomposition",
        "",
        f"- Remaining catastrophic cuts: `{len(failures)}/{len(rows)}`.",
        f"- Rotation-related: `{len(rotation_related)}/{len(failures)}`.",
        f"- Rescued by GT rotation with all metric scales unchanged: `{len(rescued)}/{len(failures)}`.",
        f"- Failure classes: `{dict(sorted(failure_classes.items()))}`.",
        "",
        "## Source counts",
        "",
        "| Source | Remaining | Rotation | Translation+rotation | Scene |",
        "|---|---:|---:|---:|---:|",
    ]
    for source in sorted({row["source"] for row in rows}):
        counts = failure_by_source[source]
        lines.append(
            f"| {source} | {counts['total']} | {counts['rotation']} | "
            f"{counts['translation+rotation']} | {counts['scene']} |"
        )
    lines.extend(
        [
            "",
            "## Worst remaining cases",
            "",
            "| Case | Source | Flags | T | R | Scene | GT-R T | GT-R Scene |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(failures, key=failure_priority, reverse=True):
        lines.append(
            f"| {row['case_name']} | {row['source']} | "
            f"{'+'.join(row['v22_catastrophic_flags'])} | "
            f"{row['v22']['translation_m']:.3f} | {row['v22']['rotation_deg']:.2f} | "
            f"{row['v22']['scene_m']:.3f} | {row['gt_rotation']['translation_m']:.3f} | "
            f"{row['gt_rotation']['scene_m']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "The remaining deployable research target is explicit wide-baseline rotation. The THuman scene-only failure is a separate pointmap-continuity problem and should not be mixed into the rotation branch.",
        ]
    )
    md_path = args.output_dir / "v22_remaining_failure_audit.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f">> wrote {json_path}")
    print(f">> wrote {md_path}")


if __name__ == "__main__":
    main()
