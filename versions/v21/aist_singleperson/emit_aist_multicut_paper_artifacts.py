#!/usr/bin/env python3
"""Emit sealed paper artifacts from complete AIST++ MC150-3/MC150-4 tests."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


METHODS = (
    ("m0_strict_human3r", "Strict Human3R"),
    ("m1_clean_reset", "Clean reset"),
    ("m3_b0_only", "Coarse alignment only"),
    ("m4_b0_identity", "Coarse alignment + identity"),
    ("m15_bridge3r_fixed_v19", "Bridge3R (fixed, causal)"),
)
METRICS = (
    "pa_mpjpe_body12_mm",
    "first_shot_anchor_mpjpe_body12_mm",
    "mean_boundary_seam_root_excess_mm",
    "mean_boundary_seam_orientation_excess_deg",
    "post_first_cut_camera_relative_rotation_deg",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mc150-3-aggregate", type=Path, required=True)
    parser.add_argument("--mc150-4-aggregate", type=Path, required=True)
    parser.add_argument("--mc150-3-audit", type=Path, required=True)
    parser.add_argument("--mc150-4-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def finite(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"non-finite {name}")
    return float(value)


def rows(aggregate: dict[str, Any], audit: dict[str, Any], protocol: str) -> list[dict[str, Any]]:
    if (
        aggregate.get("protocol") != protocol
        or aggregate.get("formal_manifest_case_count") != 100
        or aggregate.get("metric_report_case_count") != 100
    ):
        raise ValueError(f"{protocol} aggregate is not a complete 100-case result")
    if (
        audit.get("protocol") != protocol
        or audit.get("formal_manifest_case_count") != 100
        or not audit.get("audit_passed")
    ):
        raise ValueError(f"{protocol} campaign audit did not pass")
    output = []
    for method, display in METHODS:
        row = aggregate.get("methods", {}).get(method)
        if not isinstance(row, dict) or row.get("reported_case_count") != 100:
            raise ValueError(f"{protocol} incomplete method: {method}")
        if finite(
            row.get("coverage", {}).get("completion_case_macro_mean"),
            f"{method}.completion",
        ) != 1.0:
            raise ValueError(f"{protocol} incomplete inference: {method}")
        values = {
            metric: finite(
                row.get("metrics", {}).get(metric, {}).get("case_macro_mean"),
                f"{method}.{metric}",
            )
            for metric in METRICS
        }
        output.append(
            {
                "protocol": protocol,
                "method": method,
                "display": display,
                "metrics": values,
                "coverage": finite(
                    row.get("coverage", {}).get("case_macro_mean"),
                    f"{method}.coverage",
                ),
            }
        )
    return output


def fmt(value: float) -> str:
    return f"{value:.1f}"


def tex(protocol: str, values: list[dict[str, Any]]) -> str:
    lines = [
        f"% Auto-generated from sealed AIST++ {protocol} ledgers; do not hand-edit.",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Method & PA-MPJPE $\downarrow$ & Anchor-MPJPE $\downarrow$ & Mean seam-root $\downarrow$ & Mean seam-orient. $\downarrow$ & Post-cut cam. rot. $\downarrow$ & Coverage $\uparrow$ \\",
        r"\midrule",
    ]
    for row in values:
        score = row["metrics"]
        lines.append(
            f"{row['display']} & {fmt(score[METRICS[0]])} & "
            f"{fmt(score[METRICS[1]])} & {fmt(score[METRICS[2]])} & "
            f"{fmt(score[METRICS[3]])} & {fmt(score[METRICS[4]])} & "
            f"{fmt(100 * row['coverage'])} " + r"\\"
        )
    return "\n".join([*lines, r"\bottomrule", r"\end{tabular}", ""])


def main() -> None:
    args = parse_args()
    aggregates = {
        "MC150-3": load(args.mc150_3_aggregate),
        "MC150-4": load(args.mc150_4_aggregate),
    }
    audits = {
        "MC150-3": load(args.mc150_3_audit),
        "MC150-4": load(args.mc150_4_audit),
    }
    all_rows = {
        protocol: rows(aggregates[protocol], audits[protocol], protocol)
        for protocol in aggregates
    }
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    output.mkdir(parents=True)
    for protocol, values in all_rows.items():
        (output / f"aist_{protocol.lower()}_formal_table.tex").write_text(
            tex(protocol, values), encoding="utf-8"
        )
    (output / "aist_multicut_formal_rows.json").write_text(
        json.dumps(all_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report = [
        "# AIST++ multi-cut formal result",
        "",
        "Both tables use their own frozen 100-source official pose_test RGB manifests. "
        "All rows are causal internal routes; the study is an event-scaling and component analysis, "
        "not a latency-equivalent ranking against the offline PromptHMR CS150 row.",
        "",
    ]
    for protocol, values in all_rows.items():
        report.extend(
            [
                f"## {protocol}",
                "",
                "| Method | PA-MPJPE | Anchor-MPJPE | Mean seam-root | Mean seam-orient. | Post-cut camera rot. | Coverage |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in values:
            score = row["metrics"]
            report.append(
                f"| {row['display']} | {fmt(score[METRICS[0]])} | "
                f"{fmt(score[METRICS[1]])} | {fmt(score[METRICS[2]])} | "
                f"{fmt(score[METRICS[3]])} | {fmt(score[METRICS[4]])} | "
                f"{fmt(100 * row['coverage'])}% |"
            )
        report.append("")
    (output / "AIST_MULTICUT_FORMAL_REPORT.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(
        json.dumps(
            {"output": str(output), "protocols": list(all_rows)},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
