#!/usr/bin/env python3
"""Collect V8.4 pose benchmark summaries into one comparison table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRIC_KEYS = [
    "v82_raw_trans_err_mean",
    "v82_trans_err_mean",
    "v82_trans_improvement_mean",
    "v82_raw_rot_err_deg_mean",
    "v82_rot_err_deg_mean",
    "v82_rot_improvement_deg_mean",
    "v82_gate_mean_mean",
    "v82_drift_target_mean_mean",
    "v82_delta_norm_mean",
    "loss_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval_root",
        type=Path,
        default=Path("output/v8_4_pose_benchmark/eval"),
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("output/v8_4_pose_benchmark/eval/run_comparison.csv"),
    )
    parser.add_argument(
        "--output_md",
        type=Path,
        default=Path("output/v8_4_pose_benchmark/eval/run_comparison.md"),
    )
    return parser.parse_args()


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def add_row(rows: list[dict], run_name: str, summary: dict, subset_name: str, subset_summary: dict) -> None:
    row = {
        "run": run_name,
        "subset": subset_name,
        "count": subset_summary.get("count", 0),
        "model_path": summary.get("model_path", ""),
    }
    for key in METRIC_KEYS:
        value = subset_summary.get(key, "")
        if isinstance(value, float):
            value = round(value, 6)
        row[key] = value
    rows.append(row)


def collect_rows(eval_root: Path) -> list[dict]:
    rows = []
    for summary_path in sorted(eval_root.glob("*/summary.json")):
        run_name = summary_path.parent.name
        summary = load_summary(summary_path)
        for subset_name, subset_summary in sorted(summary.get("subsets", {}).items()):
            add_row(rows, run_name, summary, subset_name, subset_summary)
        if "overall" in summary:
            add_row(rows, run_name, summary, "overall", summary["overall"])
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run", "subset", "count", *METRIC_KEYS, "model_path"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "run",
        "subset",
        "count",
        "v82_raw_trans_err_mean",
        "v82_trans_err_mean",
        "v82_trans_improvement_mean",
        "v82_raw_rot_err_deg_mean",
        "v82_rot_err_deg_mean",
        "v82_rot_improvement_deg_mean",
        "v82_gate_mean_mean",
        "v82_delta_norm_mean",
    ]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = collect_rows(args.eval_root)
    write_csv(args.output_csv, rows)
    write_markdown(args.output_md, rows)
    print(f"Wrote {len(rows)} rows to {args.output_csv}")
    print(f"Wrote markdown table to {args.output_md}")


if __name__ == "__main__":
    main()
