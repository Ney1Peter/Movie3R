#!/usr/bin/env python3
"""Build a compact V10 end-to-end evaluation report.

The report is intentionally file-based.  It can be rerun after any alignment
probe finishes and will summarize:

- V10 oracle-boundary alignment metrics from ``metrics_flat.csv``.
- Detector metrics from the existing detector probe CSVs.

Additional alignment runs can be passed with repeated ``--alignment_run`` values
formatted as ``name:path``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ALIGNMENT_RUNS = [
    (
        "v10_w5_smoke_s2",
        REPO_ROOT / "output" / "v10_static_alignment_probe" / "large_4source_angle60_w5_s2",
    ),
    (
        "v10_w5_medium_s200",
        REPO_ROOT / "output" / "v10_static_alignment_probe" / "medium_4source_angle60_w5_s200",
    ),
    (
        "v10_w5_large_s2000",
        REPO_ROOT / "output" / "v10_static_alignment_probe" / "large_4source_angle60_w5_s2000",
    ),
]

DEFAULT_DETECTOR_RUNS = [
    (
        "image_only",
        REPO_ROOT / "output" / "v10_detector_probe" / "image_feature_round1" / "detector_method_results.csv",
    ),
    (
        "human3r_pose",
        REPO_ROOT / "output" / "v10_detector_probe" / "human3r_pose_round1" / "detector_human3r_pose_method_results.csv",
    ),
    (
        "combined",
        REPO_ROOT / "output" / "v10_detector_probe" / "combined_round1" / "combined_method_results.csv",
    ),
]

ALIGNMENT_METRICS = [
    ("cam_rot_deg", "Camera rot (deg)"),
    ("cam_trans_m", "Camera trans (m)"),
    ("human_post_m", "Human post (m)"),
    ("Amean_B0_m", "Amean-B0 (m)"),
    ("Amean_B1_m", "Amean-B1 (m)"),
    ("BB_m", "B0-B1 (m)"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=REPO_ROOT / "output" / "v10_end_to_end_eval",
    )
    parser.add_argument(
        "--alignment_run",
        action="append",
        default=[],
        help="Extra alignment run in name:path format.",
    )
    parser.add_argument(
        "--detector_run",
        action="append",
        default=[],
        help="Extra detector run in name:path-to-method-results.csv format.",
    )
    parser.add_argument("--top_k_detectors", type=int, default=8)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_named_path(value: str) -> tuple[str, Path]:
    if ":" not in value:
        raise ValueError(f"Expected name:path, got {value!r}")
    name, path = value.split(":", 1)
    return name.strip(), Path(path).expanduser()


def alignment_runs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    runs = list(DEFAULT_ALIGNMENT_RUNS)
    runs.extend(parse_named_path(value) for value in args.alignment_run)
    return runs


def detector_runs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    runs = list(DEFAULT_DETECTOR_RUNS)
    runs.extend(parse_named_path(value) for value in args.detector_run)
    return runs


def mean_float(rows: list[dict], key: str) -> float:
    vals = [float(row[key]) for row in rows if row.get(key, "") not in {"", "nan", "None"}]
    return float(np.mean(vals)) if vals else float("nan")


def summarize_alignment_run(name: str, run_dir: Path) -> tuple[list[dict], list[dict]]:
    metrics_path = run_dir / "metrics_flat.csv"
    if not metrics_path.is_file():
        return [], []
    rows = read_csv(metrics_path)
    summary_rows = []
    per_source_rows = []
    groups = {"overall": rows}
    for source in sorted({row["source"] for row in rows}):
        groups[source] = [row for row in rows if row["source"] == source]

    for group, group_rows in groups.items():
        out = {
            "run": name,
            "group": group,
            "samples": len(group_rows),
            "path": str(run_dir),
        }
        for metric, _ in ALIGNMENT_METRICS:
            raw_key = f"raw_{metric}"
            aligned_key = f"aligned_{metric}"
            gain_key = f"gain_{metric}"
            out[f"raw_{metric}"] = mean_float(group_rows, raw_key)
            out[f"aligned_{metric}"] = mean_float(group_rows, aligned_key)
            out[f"gain_{metric}"] = mean_float(group_rows, gain_key)
        if group == "overall":
            summary_rows.append(out)
        else:
            per_source_rows.append(out)
    return summary_rows, per_source_rows


def summarize_detectors(args: argparse.Namespace) -> list[dict]:
    out = []
    for name, path in detector_runs(args):
        if not path.is_file():
            continue
        rows = read_csv(path)
        rows = sorted(rows, key=lambda row: float(row.get("f1", 0.0)), reverse=True)
        for rank, row in enumerate(rows[: int(args.top_k_detectors)], start=1):
            out.append(
                {
                    "detector_run": name,
                    "rank": rank,
                    "method": row.get("method", ""),
                    "f1": float(row.get("f1", 0.0)),
                    "precision": float(row.get("precision", 0.0)),
                    "recall": float(row.get("recall", 0.0)),
                    "stable_fpr": float(row.get("false_positive_rate", 0.0)),
                    "accuracy": float(row.get("accuracy", 0.0)),
                    "path": str(path),
                }
            )
    return out


def fmt(value: float, digits: int = 3) -> str:
    if value != value:
        return "n/a"
    return f"{value:.{digits}f}"


def write_markdown(
    path: Path,
    alignment_summary: list[dict],
    alignment_by_source: list[dict],
    detector_summary: list[dict],
) -> None:
    lines = [
        "# V10 End-to-End Evaluation Report",
        "",
        "This report is generated from saved CSV outputs. Lower alignment metrics are better; detector F1/precision/recall are higher-is-better, while stable FPR should be low.",
        "",
        "## Alignment Runs",
        "",
        "| Run | Samples | Cam Rot raw->aligned | Cam Trans raw->aligned | Human raw->aligned | Amean-B0 raw->aligned | Amean-B1 raw->aligned | B0-B1 raw->aligned |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if alignment_summary:
        for row in alignment_summary:
            lines.append(
                "| {run} | {samples} | {rrot}->{arot} | {rt}->{at} | {rh}->{ah} | {ra0}->{aa0} | {ra1}->{aa1} | {rbb}->{abb} |".format(
                    run=row["run"],
                    samples=row["samples"],
                    rrot=fmt(row["raw_cam_rot_deg"], 2),
                    arot=fmt(row["aligned_cam_rot_deg"], 2),
                    rt=fmt(row["raw_cam_trans_m"]),
                    at=fmt(row["aligned_cam_trans_m"]),
                    rh=fmt(row["raw_human_post_m"]),
                    ah=fmt(row["aligned_human_post_m"]),
                    ra0=fmt(row["raw_Amean_B0_m"]),
                    aa0=fmt(row["aligned_Amean_B0_m"]),
                    ra1=fmt(row["raw_Amean_B1_m"]),
                    aa1=fmt(row["aligned_Amean_B1_m"]),
                    rbb=fmt(row["raw_BB_m"]),
                    abb=fmt(row["aligned_BB_m"]),
                )
            )
    else:
        lines.append("| n/a | 0 | n/a | n/a | n/a | n/a | n/a | n/a |")

    lines += [
        "",
        "## Detector Top Results",
        "",
        "| Detector set | Method | F1 | Precision | Recall | Stable FPR | Accuracy |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in detector_summary:
        if int(row["rank"]) > 5:
            continue
        lines.append(
            f"| {row['detector_run']} | {row['method']} | {row['f1']:.3f} | {row['precision']:.3f} | {row['recall']:.3f} | {row['stable_fpr']:.3f} | {row['accuracy']:.3f} |"
        )

    if alignment_by_source:
        lines += [
            "",
            "## Alignment By Source",
            "",
            "| Run | Source | Samples | Cam Rot aligned | Cam Trans aligned | Human aligned |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for row in alignment_by_source:
            lines.append(
                f"| {row['run']} | {row['group']} | {row['samples']} | {fmt(row['aligned_cam_rot_deg'], 2)} | {fmt(row['aligned_cam_trans_m'])} | {fmt(row['aligned_human_post_m'])} |"
            )

    lines += [
        "",
        "## How To Read This",
        "",
        "- `raw` is strict original Human3R local-reset output before V10 segment-to-global alignment.",
        "- `aligned` is the V10 oracle-boundary aligned output.",
        "- `Amean-B0/B1` measures whether the post-boundary B frames are pulled into the historical A-gauge human anchor.",
        "- Detector stable FPR is important because false positives would reset/align continuous frames that should have stayed original.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    alignment_summary = []
    alignment_by_source = []
    for name, path in alignment_runs(args):
        summary, by_source = summarize_alignment_run(name, path)
        alignment_summary.extend(summary)
        alignment_by_source.extend(by_source)

    detector_summary = summarize_detectors(args)

    write_csv(args.output_dir / "alignment_summary.csv", alignment_summary)
    write_csv(args.output_dir / "alignment_by_source.csv", alignment_by_source)
    write_csv(args.output_dir / "detector_summary.csv", detector_summary)
    write_markdown(args.output_dir / "report.md", alignment_summary, alignment_by_source, detector_summary)
    (args.output_dir / "report_manifest.json").write_text(
        json.dumps(
            {
                "alignment_runs": [(name, str(path)) for name, path in alignment_runs(args)],
                "detector_runs": [(name, str(path)) for name, path in detector_runs(args)],
                "outputs": {
                    "report": str(args.output_dir / "report.md"),
                    "alignment_summary": str(args.output_dir / "alignment_summary.csv"),
                    "detector_summary": str(args.output_dir / "detector_summary.csv"),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
