#!/usr/bin/env python3
"""Emit auditable AIST++ CS150 paper artifacts from final result ledgers.

This utility deliberately accepts only the completed 100-case internal
aggregate and the sealed 100-case PromptHMR ledger.  It does not choose cases,
metrics, or predictions, and makes the causal/offline distinction explicit in
the emitted table.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


INTERNAL_METHODS = (
    ("m0_strict_human3r", "Strict Human3R", "causal"),
    ("m15_bridge3r_fixed_v19", r"Bridge3R (fixed, causal)", "causal"),
)
METRICS = (
    "pa_mpjpe_body12_mm",
    "first_shot_anchor_mpjpe_body12_mm",
    "seam_root_excess_mm",
    "seam_orientation_excess_deg",
    "post_camera_relative_rotation_deg",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--internal-aggregate", type=Path, required=True)
    parser.add_argument("--prompthmr-ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object: {path}")
    return value


def finite(value: Any, *, name: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"non-finite {name}")
    return float(value)


def internal_row(aggregate: dict[str, Any], method: str, display: str, regime: str) -> dict[str, Any]:
    if aggregate.get("formal_manifest_case_count") != 100 or aggregate.get("metric_report_case_count") != 100:
        raise ValueError("internal aggregate is not a complete 100-case AIST CS150 result")
    row = aggregate.get("methods", {}).get(method)
    if not isinstance(row, dict) or int(row.get("reported_case_count", -1)) != 100:
        raise ValueError(f"internal method is incomplete: {method}")
    metric = row.get("metrics", {})
    values = {name: finite(metric.get(name, {}).get("case_macro_mean"), name=f"{method}.{name}") for name in METRICS}
    coverage = finite(row.get("coverage", {}).get("case_macro_mean"), name=f"{method}.coverage")
    completion = finite(row.get("coverage", {}).get("completion_case_macro_mean"), name=f"{method}.completion")
    if completion != 1.0:
        raise ValueError(f"internal completion is not 100%: {method}")
    return {"method": method, "display": display, "regime": regime, "metrics": values, "coverage": coverage}


def prompthmr_row(ledger: dict[str, Any]) -> dict[str, Any]:
    if ledger.get("formal_manifest_case_count") != 100 or ledger.get("reported_case_count") != 100:
        raise ValueError("PromptHMR ledger is not a complete 100-case AIST CS150 result")
    if len(ledger.get("records", [])) != 100:
        raise ValueError("PromptHMR ledger lacks 100 per-case records")
    metric = ledger.get("metrics", {})
    values = {name: finite(metric.get(name, {}).get("case_macro_mean"), name=f"prompthmr.{name}") for name in METRICS}
    coverage = finite(ledger.get("coverage", {}).get("case_macro_mean"), name="prompthmr.coverage")
    completion = finite(ledger.get("coverage", {}).get("completion_case_macro_mean"), name="prompthmr.completion")
    if completion != 1.0:
        raise ValueError("PromptHMR completion is not 100%")
    return {"method": "prompthmr_official", "display": "PromptHMR (official, offline)", "regime": "offline", "metrics": values, "coverage": coverage}


def fmt(value: float) -> str:
    return f"{value:.1f}"


def tex_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "% Auto-generated from the sealed AIST++ CS150 result ledgers; do not hand-edit.",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Execution & Method & PA-MPJPE $\downarrow$ & Anchor-MPJPE $\downarrow$ & Seam-root $\downarrow$ & Seam-orient. $\downarrow$ & Rel. cam. rot. $\downarrow$ & Coverage $\uparrow$ \\",
        r"\midrule",
    ]
    for index, row in enumerate(rows):
        if index == 2:
            lines.append(r"\midrule")
        label = "Causal streaming" if row["regime"] == "causal" else "Offline full-video"
        metric = row["metrics"]
        lines.append(
            f"{label} & {row['display']} & {fmt(metric[METRICS[0]])} & {fmt(metric[METRICS[1]])} & "
            f"{fmt(metric[METRICS[2]])} & {fmt(metric[METRICS[3]])} & {fmt(metric[METRICS[4]])} & {fmt(100.0 * row['coverage'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def markdown_report(rows: list[dict[str, Any]], internal_path: Path, external_path: Path) -> str:
    header = "| Execution | Method | PA-MPJPE | Anchor-MPJPE | Seam-root | Seam-orient. | Rel. camera rot. | Coverage |"
    divider = "|---|---|---:|---:|---:|---:|---:|---:|"
    table = []
    for row in rows:
        metric = row["metrics"]
        table.append(
            f"| {row['regime']} | {row['display']} | {fmt(metric[METRICS[0]])} | {fmt(metric[METRICS[1]])} | "
            f"{fmt(metric[METRICS[2]])} | {fmt(metric[METRICS[3]])} | {fmt(metric[METRICS[4]])} | {fmt(100.0 * row['coverage'])}% |"
        )
    return "\n".join([
        "# AIST++ CS150 formal single-person result", "",
        "All rows use the same frozen 100-source official `pose_test` RGB manifest and a 150-frame timeline. "
        "Bridge3R and Strict Human3R are causal streaming routes; PromptHMR is its unchanged official offline full-video pipeline. "
        "The rows are therefore not presented as a single latency-equivalent ranking.", "",
        header, divider, *table, "",
        "## Provenance", "",
        f"- Internal aggregate: `{internal_path.resolve()}`", 
        f"- PromptHMR sealed ledger: `{external_path.resolve()}`", 
        "- Every row has a 100-case macro denominator and 100% completion. "
        "The approximately 99.993% internal frame coverage is preserved before one-decimal display rounding.",
        "- GVHMR is intentionally absent: its pre-registered 12-case availability audit found one-sided raw tracker support on 2/12 hard-cut pilot cases, so it did not enter the global Test table.",
        "",
    ])


def main() -> None:
    args = parse_args()
    internal_path, external_path = args.internal_aggregate.resolve(), args.prompthmr_ledger.resolve()
    internal, external = load_json(internal_path), load_json(external_path)
    rows = [internal_row(internal, *item) for item in INTERNAL_METHODS] + [prompthmr_row(external)]
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refuse to overwrite paper artifacts: {output}")
    output.mkdir(parents=True)
    (output / "aist_cs150_formal_table.tex").write_text(tex_table(rows), encoding="utf-8")
    (output / "AIST_CS150_FORMAL_REPORT.md").write_text(markdown_report(rows, internal_path, external_path), encoding="utf-8")
    (output / "aist_cs150_formal_rows.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "rows": [row["method"] for row in rows]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
