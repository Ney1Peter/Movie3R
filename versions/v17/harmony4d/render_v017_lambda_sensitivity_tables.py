#!/usr/bin/env python3
"""Render complete supplementary TeX tables from the frozen lambda summary.

This reporting-only helper is deliberately separate from the preregistered
runner.  It consumes its immutable summary JSON and never materialises a
candidate, invokes a model, or makes a configuration-selection decision.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


CANDIDATES = (
    ("v16_0_m15_geometry", r"Parent ($\lambda$--)"),
    ("bridge3r_lambda_025", "0.25"),
    ("bridge3r_lambda_050", "0.50 (fixed)"),
    ("bridge3r_lambda_075", "0.75"),
    ("bridge3r_lambda_100", "1.00"),
)
METRICS = (
    ("W-MPJPE_mm", "W", 1),
    ("WA-MPJPE_mm", "WA", 1),
    ("MPJPE_mm", "MPJPE", 1),
    ("MPVPE_mm", "MPVPE", 1),
    ("Accel_mm_frame2", "Accel.", 1),
    ("ATE_Sim3_m", "ATE", 3),
    ("Seam_root_m", "Seam", 3),
    ("IDF1", "IDF1", 3),
    ("Coverage", "Cov.", 3),
)


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def finite(value: object) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def tex(value: object, digits: int) -> str:
    number = finite(value)
    return "--" if number is None else f"{number:.{digits}f}"


def row(summary: dict, candidate: str, display: str, aggregate: str) -> str:
    metrics = summary["candidates"][candidate]["metrics"]
    values = [tex(metrics[name][aggregate], digits) for name, _, digits in METRICS]
    return "    " + display + " & " + " & ".join(values) + " \\\\"


def table(summary: dict, aggregate: str, title: str, label: str) -> list[str]:
    metric_counts = {
        name: int(summary["candidates"][CANDIDATES[0][0]]["metrics"][name]["count"])
        for name, _, _ in METRICS
    }
    complete = int(summary["case_count_complete"])
    unavailable = int(summary["evaluator_unavailable_count"])
    manifest_count = complete + unavailable
    seam_count = metric_counts["Seam_root_m"]
    if any(metric_counts[name] != complete for name, _, _ in METRICS if name != "Seam_root_m"):
        raise ValueError(f"Unexpected non-seam metric denominators: {metric_counts}")
    if manifest_count != 12:
        raise ValueError(f"Expected the frozen 12-case manifest, got {manifest_count}")
    header = " & ".join([r"Shared-translation blend $\lambda$"] + [name for _, name, _ in METRICS]) + " \\\\"
    return [
        "% Auto-generated from the frozen train-only sensitivity summary; do not hand-edit values.",
        r"\begin{table*}[t]",
        r"  \centering",
        "  \\caption{" + title + " on the frozen Harmony4D training-only development "
        f"manifest (12 cases). Values are case-macro {aggregate}s over the {complete} "
        f"evaluator-complete cases; {unavailable} cases are evaluator-unavailable uniformly "
        "for every row before candidate-specific scoring. W, WA, MPJPE, MPVPE, and "
        "acceleration are in mm (acceleration per frame$^2$); ATE and seam-root are "
        f"in m. Seam-root has {seam_count} finite cases. All rows reuse identical base "
        "predictions and differ only in the shared-translation blend. The publication "
        "configuration remains $\\lambda=0.50$ irrespective of this descriptive table; "
        "the table does not select a configuration or change any primary result.}",
        "  \\label{" + label + "}",
        r"  \scriptsize",
        r"  \resizebox{\textwidth}{!}{%",
        r"  \begin{tabular}{lrrrrrrrrr}",
        r"    \toprule",
        "    " + header,
        r"    \midrule",
        *[row(summary, candidate, display, aggregate) for candidate, display in CANDIDATES],
        r"    \bottomrule",
        r"  \end{tabular}}",
        r"\end{table*}",
        "",
    ]


def main() -> None:
    parsed = args()
    summary = json.loads(parsed.summary.read_text(encoding="utf-8"))
    observed = tuple(summary.get("candidates", {}).keys())
    expected = tuple(name for name, _ in CANDIDATES)
    if observed != expected:
        raise ValueError(f"Unexpected candidate order: {observed}")
    if summary.get("reporting_scope") != "train-only sensitivity; does not select or alter the publication configuration":
        raise ValueError("Summary scope is not the frozen reporting scope")
    text = table(
        summary,
        "mean",
        "Case-macro mean blend sensitivity",
        "tab:harmony-lambda-sensitivity-mean",
    )
    text.extend(table(
        summary,
        "median",
        "Case-macro median blend sensitivity",
        "tab:harmony-lambda-sensitivity-median",
    ))
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    parsed.output.write_text("\n".join(text), encoding="utf-8")


if __name__ == "__main__":
    main()
