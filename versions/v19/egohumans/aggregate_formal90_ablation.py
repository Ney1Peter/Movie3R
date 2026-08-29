#!/usr/bin/env python3
"""Fail-closed aggregation for the sealed EgoHumans formal-90 ablations.

This utility accepts no model predictions or GT directly.  It reads only the
per-capture evaluator reports emitted by ``run_protocol.py`` and the immutable
formal case manifest, verifies exact case parity, then writes auditable JSON,
CSV, Markdown and paper-ready TeX fragments.  Availability is never converted
into a favourable geometric average: every metric carries its finite support.
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


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PARENT = REPO_ROOT / "output/bridge3r_egohumans_ablation_v1"
DEFAULT_MANIFEST = (
    REPO_ROOT.parents[0]
    / "ICLR-paper/bridge3r_iclr2027/private_audit/egohumans_formal90_manifest.jsonl"
)
CANDIDATE = "v19_egohumans_frozen"
SYSTEM_METHODS = (
    "m0_strict_human3r",
    "m1_clean_reset",
    "m3_b0_only",
    "m4_b0_identity",
    "m15_safe_boundary_permutation_causal_gru",
)
ROUTES: dict[str, str] = {
    "native": "No learned correction branch",
    "semantic_only": "Semantic only",
    "alignment_only": "Alignment only",
    "semantic_alignment": "Semantic + alignment",
    "lora_off": "Full token, LoRA off",
    "camera_residual_off": "Full token, camera residual off",
    "human_residual_off": "Full token, human residual off",
    "full_replay": "Bridge3R (full)",
}
CORE_METRICS: tuple[tuple[str, str, str, bool], ...] = (
    ("W-MPJPE_mm", "W-MPJPE", "mm", False),
    ("WA-MPJPE_mm", "WA-MPJPE", "mm", False),
    ("MPJPE_mm", "MPJPE", "mm", False),
    ("MPVPE_mm", "MPVPE", "mm", False),
    ("ATE_SE3_m", "ATE-SE3", "m", False),
    ("Seam_root_m", "Human seam", "m", False),
    ("IDF1", "IDF1", "unitless", True),
    ("Coverage", "Coverage", "unitless", True),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260828)
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(partial, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def formal_cases(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    cases = {str(row["case_id"]): row for row in rows}
    if len(cases) != len(rows) or len(cases) != 90:
        raise ValueError("Formal manifest must contain exactly 90 unique cases")
    if {str(row.get("formal_protocol")) for row in rows} != {"Bridge3R-EgoHumans-formal90-v1"}:
        raise ValueError("Unexpected formal manifest protocol")
    return cases


def report_files(root: Path) -> list[Path]:
    files = []
    for path in sorted(root.glob("test/captures/*/candidate_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if payload.get("schema_version") == "Movie3R-v16-Harmony4D-causal-stabilization-probe-v1":
            files.append(path)
    return files


def collect_route(
    root: Path, expected_cases: set[str], required_methods: set[str]
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    files = report_files(root)
    if len(files) != 27:
        raise ValueError(f"{root}: expected 27 capture reports, found {len(files)}")
    methods: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    sources: list[str] = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("errors"):
            raise ValueError(f"{path}: evaluator errors {payload['errors']}")
        if payload.get("skipped_cases"):
            raise ValueError(f"{path}: formal evaluator-unavailable cases {payload['skipped_cases']}")
        sources.append(str(path.resolve()))
        for kind, rows in (("reference", payload.get("reference_rows", [])), ("candidate", payload.get("rows", []))):
            for row in rows:
                name = str(row.get("method" if kind == "reference" else "candidate", ""))
                if name not in required_methods:
                    continue
                if row.get("status") != "complete":
                    raise ValueError(f"{path}: {name} is not complete for {row.get('case_id')}")
                case_id = str(row.get("case_id", ""))
                if case_id not in expected_cases:
                    raise ValueError(f"{path}: report has case outside formal manifest: {case_id}")
                if case_id in methods[name]:
                    raise ValueError(f"{path}: duplicate {name} row for {case_id}")
                methods[name][case_id] = row
    expected_methods = set(required_methods)
    missing_methods = expected_methods - set(methods)
    if missing_methods:
        raise ValueError(f"{root}: missing expected methods {sorted(missing_methods)}")
    result = {}
    for method in expected_methods:
        observed = set(methods[method])
        if observed != expected_cases:
            raise ValueError(
                f"{root}: {method} formal case mismatch; missing={len(expected_cases-observed)}, "
                f"extra={len(observed-expected_cases)}"
            )
        result[method] = [methods[method][case] for case in sorted(expected_cases)]
    return result, sources


def metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"case_count": len(rows)}
    for key, _, _, _ in CORE_METRICS:
        values = [finite(row.get("metrics", {}).get(key)) for row in rows]
        values = [value for value in values if value is not None]
        result[key] = float(np.mean(values)) if values else None
        result[f"{key}_available_cases"] = len(values)
    return result


def stratified(
    rows: list[dict[str, Any]], formal: dict[str, dict[str, Any]], selector: str
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        case = formal[str(row["case_id"])]
        if selector == "angle_stratum":
            group = str(case["angle_stratum"])
        elif selector == "ge150":
            group = "ge150" if float(case["camera_rotation_span_deg_evaluator_only"]) >= 150.0 else "lt150"
        else:
            raise ValueError(selector)
        groups[group].append(row)
    return {key: metric_summary(value) for key, value in sorted(groups.items())}


def capture_paired_bootstrap(
    full_rows: list[dict[str, Any]], other_rows: list[dict[str, Any]], metric: str,
    samples: int, rng: np.random.Generator,
) -> dict[str, Any]:
    full_by_case = {str(row["case_id"]): row for row in full_rows}
    other_by_case = {str(row["case_id"]): row for row in other_rows}
    groups: dict[str, list[float]] = defaultdict(list)
    for case_id in sorted(set(full_by_case) & set(other_by_case)):
        a = finite(full_by_case[case_id].get("metrics", {}).get(metric))
        b = finite(other_by_case[case_id].get("metrics", {}).get(metric))
        if a is None or b is None:
            continue
        capture = str(full_by_case[case_id].get("capture"))
        groups[capture].append(a - b)  # negative benefits a lower-is-better error.
    values = [float(np.mean(group)) for _, group in sorted(groups.items()) if group]
    if not values:
        return {"capture_count": 0, "mean_full_minus_route": None, "ci95": [None, None]}
    array = np.asarray(values, dtype=np.float64)
    draws = rng.integers(0, len(array), size=(samples, len(array)))
    estimates = array[draws].mean(axis=1)
    return {
        "capture_count": len(array),
        "mean_full_minus_route": float(array.mean()),
        "ci95": [float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))],
        "unit": "full_minus_route; negative favours full for error metrics",
    }


def tex_number(value: Any, digits: int = 1) -> str:
    number = finite(value)
    return "--" if number is None else f"{number:.{digits}f}"


def compact_table(summary: dict[str, Any], rows: list[tuple[str, dict[str, Any]]], caption: str, label: str) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Method & W-MPJPE$\downarrow$ & WA-MPJPE$\downarrow$ & MPJPE$\downarrow$ & Human seam$\downarrow$ & IDF1$\uparrow$ & Coverage$\uparrow$\\",
        r"\midrule",
    ]
    for name, values in rows:
        lines.append(
            f"{name} & {tex_number(values.get('W-MPJPE_mm'))} & {tex_number(values.get('WA-MPJPE_mm'))} & "
            f"{tex_number(values.get('MPJPE_mm'))} & {tex_number(values.get('Seam_root_m'), 3)} & "
            f"{tex_number(values.get('IDF1'), 3)} & {tex_number(values.get('Coverage'), 3)}\\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}}",
        rf"\caption{{{caption} Every row uses the immutable 90-case EgoHumans protocol; geometric metrics retain their finite support in the accompanying ledger.}}",
        rf"\label{{{label}}}",
        r"\end{table}",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    parent = args.parent.resolve()
    formal = formal_cases(args.manifest.resolve())
    expected_cases = set(formal)
    routes: dict[str, dict[str, list[dict[str, Any]]]] = {}
    sources: dict[str, list[str]] = {}
    for name in ROUTES:
        root = parent / f"formal90_{name}"
        required = (
            set(SYSTEM_METHODS) | {CANDIDATE}
            if name == "full_replay"
            else {"m0_strict_human3r", "m15_safe_boundary_permutation_causal_gru", CANDIDATE}
        )
        routes[name], sources[name] = collect_route(root, expected_cases, required)

    # Strict Human3R must be exactly the same frozen reference across all
    # inference masks.  Compare the metrics, rather than assuming it from the
    # command line, to detect accidental preprocessing divergence.
    strict_signature = None
    for name, methods in routes.items():
        signature = [
            (row["case_id"], json.dumps(row.get("metrics", {}), sort_keys=True))
            for row in methods["m0_strict_human3r"]
        ]
        if strict_signature is None:
            strict_signature = signature
        elif signature != strict_signature:
            raise ValueError(f"Strict Human3R differs between formal routes: {name}")

    named_rows: dict[str, list[dict[str, Any]]] = {
        "Strict Human3R": routes["full_replay"]["m0_strict_human3r"],
        "Clean reset": routes["full_replay"]["m1_clean_reset"],
        "Coarse gauge": routes["full_replay"]["m3_b0_only"],
        "Coarse + association": routes["full_replay"]["m4_b0_identity"],
        "Causal transaction": routes["full_replay"]["m15_safe_boundary_permutation_causal_gru"],
        "No learned correction branch": routes["native"][CANDIDATE],
        "Semantic only": routes["semantic_only"][CANDIDATE],
        "Alignment only": routes["alignment_only"][CANDIDATE],
        "Semantic + alignment": routes["semantic_alignment"][CANDIDATE],
        "Full token, LoRA off": routes["lora_off"][CANDIDATE],
        "Full token, camera residual off": routes["camera_residual_off"][CANDIDATE],
        "Full token, human residual off": routes["human_residual_off"][CANDIDATE],
        "Bridge3R (full)": routes["full_replay"][CANDIDATE],
    }
    summaries = {name: metric_summary(rows) for name, rows in named_rows.items()}
    strata = {
        name: {
            "angle_stratum": stratified(rows, formal, "angle_stratum"),
            "ge150": stratified(rows, formal, "ge150"),
        }
        for name, rows in named_rows.items()
    }
    rng = np.random.default_rng(int(args.seed))
    full_rows = named_rows["Bridge3R (full)"]
    paired = {
        name: {
            metric: capture_paired_bootstrap(full_rows, rows, metric, int(args.bootstrap), rng)
            for metric in ("W-MPJPE_mm", "WA-MPJPE_mm", "Seam_root_m", "IDF1", "Coverage")
        }
        for name, rows in named_rows.items()
        if name != "Bridge3R (full)"
    }
    result = {
        "schema_version": "Bridge3R-EgoHumans-formal90-ablation-summary-v1",
        "protocol": "Bridge3R-EgoHumans-formal90-v1",
        "formal_manifest": str(args.manifest.resolve()),
        "formal_manifest_sha256": sha256(args.manifest.resolve()),
        "case_count": len(formal),
        "capture_count": len({str(row["capture"]) for row in formal.values()}),
        "angle_stratum_counts": {
            key: sum(str(row["angle_stratum"]) == key for row in formal.values())
            for key in ("small", "medium", "large", "extreme")
        },
        "ge150_case_count": sum(
            float(row["camera_rotation_span_deg_evaluator_only"]) >= 150.0 for row in formal.values()
        ),
        "sources": sources,
        "methods": summaries,
        "stratified": strata,
        "capture_paired_bootstrap": paired,
        "metric_contract": {
            "coverage_and_idf1": "all 90 formal cases; no success-only denominator",
            "geometry": "finite evaluator values with per-metric available-case count",
            "uncertainty": "capture-level paired bootstrap; camera pairs remain clustered within capture",
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "formal90_ablation_summary.json", result)
    fields = ["method", "case_id", "capture", "sequence", "angle_stratum", "angle_deg", *[key for key, _, _, _ in CORE_METRICS]]
    with (args.output / "formal90_ablation_case_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, rows in named_rows.items():
            for row in rows:
                case = formal[str(row["case_id"])]
                writer.writerow({
                    "method": name,
                    "case_id": row["case_id"],
                    "capture": row.get("capture"),
                    "sequence": row.get("sequence"),
                    "angle_stratum": case["angle_stratum"],
                    "angle_deg": case["camera_rotation_span_deg_evaluator_only"],
                    **{key: row.get("metrics", {}).get(key) for key, _, _, _ in CORE_METRICS},
                })
    token_rows = [
        ("Strict Human3R", summaries["Strict Human3R"]),
        ("No learned correction branch", summaries["No learned correction branch"]),
        ("Semantic only", summaries["Semantic only"]),
        ("Alignment only", summaries["Alignment only"]),
        ("Semantic + alignment", summaries["Semantic + alignment"]),
        (r"\textbf{Bridge3R (full)}", summaries["Bridge3R (full)"]),
    ]
    (args.output / "formal90_token_ablation.tex").write_text(
        compact_table(result, token_rows, "Correction-token ablation on EgoHumans.", "tab:egohumans-token"),
        encoding="utf-8",
    )
    system_rows = [
        ("Strict Human3R", summaries["Strict Human3R"]),
        ("Clean reset", summaries["Clean reset"]),
        ("Coarse gauge", summaries["Coarse gauge"]),
        ("Coarse + association", summaries["Coarse + association"]),
        ("Causal boundary operation", summaries["Causal transaction"]),
        (r"\textbf{Bridge3R (full)}", summaries["Bridge3R (full)"]),
    ]
    (args.output / "formal90_system_ablation.tex").write_text(
        compact_table(result, system_rows, "Causal boundary-operation controls on EgoHumans.", "tab:egohumans-system"),
        encoding="utf-8",
    )
    head_rows = [
        ("Full token, LoRA off", summaries["Full token, LoRA off"]),
        ("Full token, camera residual off", summaries["Full token, camera residual off"]),
        ("Full token, human residual off", summaries["Full token, human residual off"]),
        (r"\textbf{Bridge3R (full)}", summaries["Bridge3R (full)"]),
    ]
    (args.output / "formal90_head_ablation.tex").write_text(
        compact_table(result, head_rows, "Fixed-checkpoint component masking on EgoHumans.", "tab:egohumans-head"),
        encoding="utf-8",
    )
    lines = [
        "# EgoHumans formal-90 ablation summary",
        "",
        f"- Manifest SHA-256: `{result['formal_manifest_sha256']}`",
        f"- Formal cases: {result['case_count']}; captures: {result['capture_count']}; ≥150°: {result['ge150_case_count']}",
        "- Coverage and IDF1 retain the 90-case denominator; every conditional geometry metric states its finite support in JSON/CSV.",
        "",
        "| Method | W-MPJPE | WA-MPJPE | MPJPE | MPVPE | ATE-SE3 | Seam | IDF1 | Coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, summary in summaries.items():
        lines.append(
            f"| {name} | {tex_number(summary['W-MPJPE_mm'])} | {tex_number(summary['WA-MPJPE_mm'])} | "
            f"{tex_number(summary['MPJPE_mm'])} | {tex_number(summary['MPVPE_mm'])} | "
            f"{tex_number(summary['ATE_SE3_m'], 3)} | {tex_number(summary['Seam_root_m'], 3)} | "
            f"{tex_number(summary['IDF1'], 3)} | {tex_number(summary['Coverage'], 3)} |"
        )
    (args.output / "FORMAL90_ABLATION_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "cases": len(formal), "methods": list(summaries)}, indent=2))


if __name__ == "__main__":
    main()
