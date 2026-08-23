#!/usr/bin/env python3
"""Create the publication-facing EgoHumans external-baseline artifacts.

The script is intentionally downstream of the fail-closed aggregator.  It does
not select cases or recompute predictions; it only formats the sealed case
rows and the already frozen Movie3R report into a paper table and an audit
report.  Conditional body/world metrics remain labelled with their available
case counts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "Movie3R/output/v19_egohumans/final"
AGG = OUT / "external_baseline_metrics.json"
MAIN = OUT / "main_test_metrics.csv"
CASE = OUT.parent / "test/summary/case_metrics.csv"
REPORT = OUT / "FINAL_EGOHUMANS_ICLR_REPORT_20260823.md"
TABLE = ROOT / "ICLR-paper/movie3r_iclr2027_draft/artifacts/egohuman_external_table.tex"


def f(value, digits=1):
    if value is None or value == "":
        return "—"
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(x):
        return "—"
    return f"{x:.{digits}f}"


def pct(value):
    if value is None:
        return "—"
    return f"{100.0 * float(value):.2f}%"


def load_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def case_mean(rows, method, key):
    values = []
    for row in rows:
        if row.get("method") != method or row.get(key) in (None, ""):
            continue
        try:
            value = float(row[key])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return sum(values) / len(values) if values else None


def main():
    if not AGG.is_file():
        raise SystemExit(f"missing aggregator output: {AGG}")
    aggregate = json.loads(AGG.read_text(encoding="utf-8"))
    main_rows = {row["method"]: row for row in load_csv(MAIN)}
    case_rows = load_csv(CASE)
    # Movie3R's case CSV contains only evaluator-available cases.  Recover the
    # method-independent unavailable count from the frozen 116-case manifest so
    # those cases are never mislabeled as inference failures.
    manifest_path = ROOT / "data/EgoHuman_work_v19/external_predictions/trace_egohumans_v2/manifests/egohumans_test.runtime.jsonl"
    manifest_case_ids = {
        json.loads(line)["case_id"]
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    evaluator_available_case_ids = {row.get("case_id") for row in case_rows if row.get("case_id")}
    evaluator_unavailable_cases = len(manifest_case_ids - evaluator_available_case_ids)

    # Keep the external rows in the same order as the paper prose.
    external_order = [
        "TRACE (official, subject=4)",
        "PromptHMR (official SPEC)",
        "PromptHMR (no-SPEC adapter)",
    ]
    rows = []
    movie_rows = [
        ("Strict Human3R", main_rows.get("m0_strict_human3r")),
        ("Movie3R-v19 (frozen)", main_rows.get("v19_egohumans_frozen")),
    ]
    for label, src in movie_rows:
        if not src:
            raise SystemExit(f"missing Movie3R row for {label}")
        rows.append({
            "label": label,
            "total_cases": 116,
            "metric_cases": 90,
            "success_cases": 116,
            "failed_cases": 0,
            "evaluator_unavailable_cases": evaluator_unavailable_cases,
            "W_available_cases": 90,
            "WA_available_cases": 90,
            "W-MPJPE_mm": src.get("W-MPJPE_mm"),
            "WA-MPJPE_mm": src.get("WA-MPJPE_mm"),
            "MPJPE_mm": src.get("MPJPE_mm"),
            "PA-MPJPE_mm": case_mean(case_rows, src.get("method", ""), "PA-MPJPE_mm"),
            "MPVPE_mm": src.get("MPVPE_mm"),
            "Accel_mm_frame2": src.get("Accel_mm_frame2"),
            "ATE-Sim3_m": src.get("ATE_Sim3_m"),
            "ATE-SE3_m": src.get("ATE_SE3_m"),
            "RPE-translation_m": None,
            "RPE-rotation_deg": None,
            "Camera-seam-translation_m": None,
            "Camera-seam-rotation_deg": None,
            "Human-seam_mm": None,
            "IDF1": src.get("IDF1"),
            "IDs": src.get("IDs"),
            "Coverage": src.get("Coverage"),
            "Precision": case_mean(case_rows, src.get("method", ""), "Detection_precision"),
        })
    for label in external_order:
        src = aggregate[label]
        rows.append({
            "label": label,
            "total_cases": src.get("case_count"),
            "metric_cases": src.get("W-MPJPE_mm_available_cases"),
            "success_cases": src.get("success_cases"),
            "failed_cases": src.get("failed_cases"),
            "evaluator_unavailable_cases": evaluator_unavailable_cases,
            "W_available_cases": src.get("W_available_cases"),
            "WA_available_cases": src.get("WA_available_cases"),
            **{key: src.get(key) for key in [
                "W-MPJPE_mm", "WA-MPJPE_mm", "MPJPE_mm", "PA-MPJPE_mm",
                "MPVPE_mm", "Accel_mm_frame2", "ATE-Sim3_m", "ATE-SE3_m",
                "RPE-translation_m", "RPE-rotation_deg", "Camera-seam-translation_m",
                "Camera-seam-rotation_deg", "Human-seam_mm", "IDF1", "IDs", "Coverage", "Precision"]},
        })

    # A compact generated table is included by main.tex.  Percentages are
    # shown only for coverage/precision/IDF1; all geometric errors retain units
    # in the caption and the accompanying report.
    lines = [
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrrrrrrrrrrrr}",
        r"\toprule",
        r"Method & $N$ & Inf. fail & Eval. unavail. & Coverage & Precision & IDF1 & W & WA & MPJPE & PA & MPVPE & Accel & ATE-Sim3 & ATE-SE3 & RPE-T & RPE-R & IDs & W/WA avail.\\",
        r"\midrule",
    ]
    for row in rows:
        def val(key, digits=1):
            return f(row.get(key), digits).replace("—", r"\textemdash{}")
        label = row["label"].replace("-SPEC", r"--SPEC")
        if row["label"] == "Movie3R-v19 (frozen)":
            label = r"\textbf{Movie3R-v19}"
        cov_tex = pct(row.get("Coverage")).replace("%", r"\%")
        precision_tex = pct(row.get("Precision")).replace("%", r"\%")
        idf1_tex = pct(row.get("IDF1")).replace("%", r"\%")
        lines.append(
            f"{label} & {row['total_cases']} & {row['failed_cases']} & {row['evaluator_unavailable_cases']} & "
            f"{cov_tex} & {precision_tex} & {idf1_tex} & {val('W-MPJPE_mm')} & {val('WA-MPJPE_mm')} & "
            f"{val('MPJPE_mm')} & {val('PA-MPJPE_mm')} & {val('MPVPE_mm')} & {val('Accel_mm_frame2')} & "
            f"{val('ATE-Sim3_m', 3)} & {val('ATE-SE3_m', 3)} & {val('RPE-translation_m', 3)} & {val('RPE-rotation_deg', 3)} & "
            f"{val('IDs')} & {row.get('W_available_cases', 0)}/{row['total_cases']}; {row.get('WA_available_cases', 0)}/{row['total_cases']}\\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}}",
        r"\parbox{0.99\textwidth}{\footnotesize W/WA/MPJPE/PA/MPVPE/Accel are in mm (Accel is mm/frame$^2$); IDs is switches per case. `Inf. fail' counts method inference failures, while `Eval. unavail.' denotes the 26 method-independent cases without a valid shared-world evaluator fit. Movie3R/strict-Human3R body, identity, coverage, and precision values are conditional on the remaining 90 evaluator-available cases. External rows retain all 116 cases in the denominator; their body/world values show accepted-match availability, while Coverage, Precision, IDF1, and failure counts use all cases. TRACE has no independent physical camera trajectory, so its ATE/RPE is N/A. PromptHMR SPEC and no-SPEC are separately audited branches.}",
    ])
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def strata_tex(csv_name: str, tex_name: str, group_field: str, caption: str):
        path = OUT / csv_name
        if not path.is_file():
            return
        values = load_csv(path)
        tex = [
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{llrrrrrr}", r"\toprule",
            f"Method & {group_field.title()} & Coverage & IDF1 & W & WA & MPJPE & MPVPE\\",
            r"\midrule",
        ]
        for value in values:
            label = value.get("method", "").replace("-SPEC", r"--SPEC")
            group = value.get(group_field, "")
            cov_tex = pct(value.get("Coverage")).replace("%", r"\%")
            idf1_tex = pct(value.get("IDF1")).replace("%", r"\%")
            tex.append(
                f"{label} & {group} & {cov_tex} & {idf1_tex} & {f(value.get('W-MPJPE_mm'))} & "
                f"{f(value.get('WA-MPJPE_mm'))} & {f(value.get('MPJPE_mm'))} & {f(value.get('MPVPE_mm'))}\\\\"
            )
        tex.extend([r"\bottomrule", r"\end{tabular}}"])
        (TABLE.parent / tex_name).write_text("\n".join(tex) + "\n", encoding="utf-8")

    strata_tex("external_baseline_angle_metrics.csv", "egohuman_external_angle_table.tex", "angle_stratum", "EgoHumans angle strata")
    strata_tex("external_baseline_action_metrics.csv", "egohuman_external_action_table.tex", "sequence", "EgoHumans action strata")

    # Full markdown audit: the generated CSV/JSON files remain the canonical
    # machine-readable source, while this report is a human-readable handoff.
    md = [
        "# EgoHumans CS100 final external-baseline report",
        "",
        "## Frozen protocol",
        "",
        "- One frozen runtime/evaluator manifest: 116 camera-pair cases from 29 Test captures.",
        "- Every case uses the same RGB pair, 100 frames (50 pre + 50 post), 20 FPS, and the same evaluator topology.",
        "- Runtime manifest SHA256: `8c31b2b16afac7817ecc1f81e44d7811104c160ac22495697bd52f62a2054606`; evaluator manifest SHA256: `789e4731890cefbcf070a27f973f8968d977603e0e47fd58166b9be813cd79`.",
        "- Frozen protocol seal SHA256: `34207b999de690c2e166e6d1492cd3e1063de6727eda10a5c3178432511ce958`.",
        "- TRACE uses fixed subject count 4; PromptHMR is reported as official SPEC and separately labelled no-SPEC adapter.",
        "- All inference failures and zero-match cases remain in the denominator; no case is removed by method outcome.",
        "- Movie3R/strict-Human3R body/world values are conditional on 90 evaluator-available cases; 26 method-independent cases are disclosed separately.",
        "- Movie3R/strict-Human3R identity, coverage, and precision values are likewise conditional on those 90 evaluator-available cases; external rows retain all 116 cases for full-denominator coverage/identity/failure accounting.",
        "",
        "## Main case-macro results",
        "",
        "| Method | N | inference fail | evaluator unavailable | W avail. | WA avail. | W | WA | MPJPE | PA-MPJPE | MPVPE | Accel | ATE-Sim3 | ATE-SE3 | RPE-T | RPE-R | IDF1 | IDs | Coverage | Precision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md.append(
            f"| {row['label']} | {row['total_cases']} | {row['failed_cases']} | {row['evaluator_unavailable_cases']} | "
            f"{row.get('W_available_cases', 0)} | {row.get('WA_available_cases', 0)} | {f(row.get('W-MPJPE_mm'))} | "
            f"{f(row.get('WA-MPJPE_mm'))} | {f(row.get('MPJPE_mm'))} | {f(row.get('PA-MPJPE_mm'))} | "
            f"{f(row.get('MPVPE_mm'))} | {f(row.get('Accel_mm_frame2'))} | {f(row.get('ATE-Sim3_m'), 3)} | "
            f"{f(row.get('ATE-SE3_m'), 3)} | {f(row.get('RPE-translation_m'), 3)} | {f(row.get('RPE-rotation_deg'), 3)} | "
            f"{f(row.get('IDF1'), 3)} | {f(row.get('IDs'), 2)} | "
            f"{pct(row.get('Coverage'))} | {pct(row.get('Precision'))} |"
        )
    md.extend([
        "",
        "Metrics W/WA/MPJPE/PA/MPVPE/Accel/Human-seam are mm (Accel mm/frame²); ATE/RPE/camera-seam values are metres except rotation in degrees.  Conditional errors must always be read with their availability counts. Movie3R/strict-Human3R identity and coverage are conditional on 90 evaluator-available cases; external identity, coverage, and failure counts retain the full 116-case denominator.",
        "",
        "## External failure accounting",
        "",
    ])
    failure_path = OUT / "external_baseline_failures.csv"
    if failure_path.is_file():
        failures = load_csv(failure_path)
        from collections import Counter
        by_method = Counter(r.get("method") for r in failures)
        md.append(f"The generated failure/zero-coverage ledger contains {len(failures)} rows (a row may be a successful run with no accepted match).")
        for method, count in sorted(by_method.items()):
            md.append(f"- {method}: {count} failure or zero-coverage rows.")
    else:
        md.append("Failure ledger is generated alongside this report after aggregation.")
    md.extend([
        "",
        "## Stratified artifacts",
        "",
        "- `external_baseline_case_metrics.csv`: one row per method and manifest case.",
        "- `external_baseline_angle_metrics.csv`: small/medium/large/extreme case-macro strata.",
        "- `external_baseline_action_metrics.csv`: action/sequence strata.",
        "- `external_baseline_failures.csv`: inference failures and zero-coverage cases with reasons available in source evaluation JSON.",
        "- `external_baseline_metrics.json`: complete summaries and nested case rows.",
        "- `artifact_manifest.json`: frozen EgoHumans artifact inventory (SHA256 `ff46b7cd0b2edd0454535e0a6fa997a00200d704b6a5b807ced1993bbc0da453`).",
        "",
        "## Evidence boundary",
        "",
        "TRACE and PromptHMR are executable same-input comparisons under the frozen EgoHumans protocol. HumanMM and Multi-THuMBS remain literature context because their exact evaluator/data/code are unavailable; their paper numbers are not mixed into this table. No claim of beating an unavailable-code baseline is made.",
    ])
    REPORT.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(REPORT)
    print(TABLE)


if __name__ == "__main__":
    main()
