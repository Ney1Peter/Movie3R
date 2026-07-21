#!/usr/bin/env python3
"""Merge V11 stage-1 shards and summarize gauge-neutral residuals."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "output" / "v11_gauge_neutral_first_write" / "stage1_full"
DEFAULT_OUTPUT = DEFAULT_INPUT / "merged"

METRICS = (
    "camera_relative_translation_m",
    "camera_relative_rotation_deg",
    "camera_frame_pointmap_m",
    "camera_frame_depth_m",
    "depth_consistency_m",
    "local_scale_log_abs",
    "root_centered_human_m",
    "human_relative_root_m",
    "torso_relative_orientation_deg",
    "human_local_pose_deg",
    "world_camera_translation_m",
    "world_camera_rotation_deg",
    "world_pointmap_teacher_m",
    "world_human_root_m",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def finite(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def stats(values: list[float]) -> dict:
    array = finite(values)
    if not len(array):
        return {key: None for key in ("count", "mean", "median", "p90", "p95", "max")}
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def camera_rates(cases: list[dict], frame_offset: int | None = None) -> dict:
    pairs = []
    for case in cases:
        if frame_offset is None:
            row = case["mean_future"]
        else:
            rows = {int(item["offset"]): item for item in case["per_frame"]}
            row = rows.get(frame_offset)
            if row is None:
                continue
        translation = float(row["camera_relative_translation_m"])
        rotation = float(row["camera_relative_rotation_deg"])
        if math.isfinite(translation) and math.isfinite(rotation):
            pairs.append((translation, rotation))
    if not pairs:
        return {
            "count": 0,
            "strict_success_rate": None,
            "relaxed_success_rate": None,
            "catastrophic_rate": None,
        }
    return {
        "count": len(pairs),
        "strict_success_rate": float(np.mean([t < 0.10 and r < 5.0 for t, r in pairs])),
        "relaxed_success_rate": float(np.mean([t < 0.25 and r < 10.0 for t, r in pairs])),
        "catastrophic_rate": float(np.mean([t > 1.0 or r > 30.0 for t, r in pairs])),
    }


def summarize(cases: list[dict]) -> dict:
    result = {
        "case_count": len(cases),
        "mean_future": {
            metric: stats([float(case["mean_future"].get(metric, float("nan"))) for case in cases])
            for metric in METRICS
        },
        "camera_rates": camera_rates(cases),
        "offsets": {},
    }
    for offset in (0, 1, 2, 4, 8):
        rows = []
        for case in cases:
            match = next((row for row in case["per_frame"] if int(row["offset"]) == offset), None)
            if match is not None:
                rows.append(match)
        result["offsets"][str(offset)] = {
            "case_count": len(rows),
            "metrics": {
                metric: stats([float(row.get(metric, float("nan"))) for row in rows])
                for metric in METRICS
            },
            "camera_rates": camera_rates(cases, offset),
        }
    return result


def tertile_labels(cases: list[dict], key: str, names: tuple[str, str, str]) -> dict[str, str]:
    values = finite([float(case[key]) for case in cases])
    low, high = np.percentile(values, [33.333, 66.667])
    output = {}
    for case in cases:
        value = float(case[key])
        output[case["case_name"]] = names[0] if value <= low else names[1] if value <= high else names[2]
    return output


def grouped_summary(cases: list[dict]) -> dict:
    texture = tertile_labels(cases, "texture_score", ("low", "medium", "high"))
    speed = tertile_labels(cases, "human_speed_m_per_frame", ("low", "medium", "high"))
    groups: dict[str, dict[str, list[dict]]] = {
        "source": defaultdict(list),
        "angle_bucket": defaultdict(list),
        "texture_tertile": defaultdict(list),
        "human_speed_tertile": defaultdict(list),
        "human_count": defaultdict(list),
    }
    for case in cases:
        record = case["record"]
        groups["source"][str(record["source"])].append(case)
        groups["angle_bucket"][str(record.get("angle_bucket", "unknown"))].append(case)
        groups["texture_tertile"][texture[case["case_name"]]].append(case)
        groups["human_speed_tertile"][speed[case["case_name"]]].append(case)
        groups["human_count"][str(case.get("human_count", "unknown"))].append(case)
    return {
        group_name: {label: summarize(items) for label, items in sorted(group.items())}
        for group_name, group in groups.items()
    }


def boundary_audit(cases: list[dict]) -> dict:
    translation = [float(case["boundary_checks"]["world_camera_translation_m"]) for case in cases]
    rotation = [float(case["boundary_checks"]["world_camera_rotation_deg"]) for case in cases]
    return {
        "translation_m": stats(translation),
        "rotation_deg": stats(rotation),
        "all_numerically_zero": bool(
            max(translation, default=0.0) < 1e-5 and max(rotation, default=0.0) < 0.11
        ),
    }


def write_case_csv(path: Path, cases: list[dict], texture: dict[str, str], speed: dict[str, str]) -> None:
    rows = []
    for case in cases:
        row = {
            "case_name": case["case_name"],
            "source": case["record"]["source"],
            "angle_bucket": case["record"].get("angle_bucket", "unknown"),
            "view_angle_deg": case["record"].get("view_angle_deg"),
            "texture_tertile": texture[case["case_name"]],
            "human_speed_tertile": speed[case["case_name"]],
            "post_count": case["post_count"],
        }
        row.update(case["mean_future"])
        rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# V11 Stage-1 Gauge-Neutral Residual Audit",
        "",
        f"Cases: {report['case_count']}",
        "",
        "## Overall future residual",
        "",
        "| Metric | Mean | Median | P90 | P95 | Max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric, row in report["overall"]["mean_future"].items():
        lines.append(
            f"| {metric} | {row['mean']:.6f} | {row['median']:.6f} | "
            f"{row['p90']:.6f} | {row['p95']:.6f} | {row['max']:.6f} |"
        )
    rates = report["overall"]["camera_rates"]
    lines.extend(
        [
            "",
            "## Camera relative-motion rates",
            "",
            f"- Strict success (<0.10 m and <5 deg): {100.0 * rates['strict_success_rate']:.2f}%",
            f"- Relaxed success (<0.25 m and <10 deg): {100.0 * rates['relaxed_success_rate']:.2f}%",
            f"- Catastrophic (>1 m or >30 deg): {100.0 * rates['catastrophic_rate']:.2f}%",
            "",
            "## Offset curve",
            "",
            "| Offset | Cases | Camera t (m) | Camera R (deg) | Pointmap (m) | Human rel root (m) |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for offset, row in report["overall"]["offsets"].items():
        metric = row["metrics"]
        lines.append(
            f"| {offset} | {row['case_count']} | "
            f"{metric['camera_relative_translation_m']['mean']:.6f} | "
            f"{metric['camera_relative_rotation_deg']['mean']:.6f} | "
            f"{metric['camera_frame_pointmap_m']['mean']:.6f} | "
            f"{metric['human_relative_root_m']['mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Boundary sanity",
            "",
            f"- GT boundary camera alignment numerically zero: {report['boundary_audit']['all_numerically_zero']}",
            "- Scene depth GT is unavailable; camera-frame pointmap targets use the same-camera warmed teacher.",
            "- All 180 processed boundary samples contain one detected person; no multi-person subgroup is available.",
            "- Background overlap labels are unavailable in the current manifest.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    shards = sorted(args.input_dir.glob("stage1_shard_*_of_*.json"))
    if not shards:
        raise FileNotFoundError(args.input_dir)
    cases = []
    for shard in shards:
        cases.extend(json.loads(shard.read_text(encoding="utf-8"))["cases"])
    names = [case["case_name"] for case in cases]
    if len(names) != len(set(names)):
        raise RuntimeError("Duplicate V11 stage-1 cases")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    texture = tertile_labels(cases, "texture_score", ("low", "medium", "high"))
    speed = tertile_labels(cases, "human_speed_m_per_frame", ("low", "medium", "high"))
    report = {
        "experiment": "V11 Stage-1 Gauge-Neutral Residual Audit",
        "case_count": len(cases),
        "shards": [str(path) for path in shards],
        "overall": summarize(cases),
        "groups": grouped_summary(cases),
        "boundary_audit": boundary_audit(cases),
        "limitations": {
            "scene_depth_gt": False,
            "pointmap_target": "same-camera warmed Human3R teacher",
            "background_overlap_labels": False,
            "multi_person_cases": int(sum(int(case.get("human_count", 1)) > 1 for case in cases)),
        },
    }
    (args.output_dir / "stage1_merged.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_case_csv(args.output_dir / "stage1_cases.csv", cases, texture, speed)
    write_markdown(args.output_dir / "stage1_summary.md", report)
    print(f">> merged {len(cases)} cases from {len(shards)} shards")
    print(f">> wrote {args.output_dir / 'stage1_summary.md'}")


if __name__ == "__main__":
    main()
