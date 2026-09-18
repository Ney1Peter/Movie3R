#!/usr/bin/env python3
"""Build private denominator/support ledgers for BRIDGE3R v021.

This script is intentionally outside every manuscript/version directory.  Its
outputs are internal audit material and must not be copied into an Overleaf or
submission archive.
"""

from __future__ import annotations

import csv
import hashlib
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent

DATASETS = {
    "EgoHumans-CS100": {
        "formal_n": 90,
        "cluster_n": 27,
        "aggregation": "case macro; paired statistics clustered by capture",
        "internal_csv": ROOT / "Movie3R/output/v19_egohumans/test/summary/case_metrics.csv",
        "external_csv": ROOT / "Movie3R/output/v19_egohumans/final/external_baseline_case_metrics.csv",
        "strict_method": "m0_strict_human3r",
        "bridge_method": "v19_egohumans_frozen",
    },
    "Harmony4D-CS150": {
        "formal_n": 88,
        "cluster_n": 25,
        "aggregation": "case macro; paired statistics clustered by capture",
        "internal_csv": ROOT / "Movie3R/output/v17_harmony4d/unified_half_translation_audit/paper/case_metrics.csv",
        "external_csv": ROOT / "data/Harmony4D_work_v17_full_test/external_predictions/harmony4d_external_aggregate/external_baseline_case_metrics.csv",
        "strict_method": "m0_strict_human3r",
        "bridge_method": "bridge3r_unified_half_translation",
    },
}

DISPLAY = {
    "m0_strict_human3r": "Strict Human3R",
    "v19_egohumans_frozen": "BRIDGE3R",
    "bridge3r_unified_half_translation": "BRIDGE3R",
    "trace_official_adapted": "TRACE (official adapter)",
    "prompthmr_spec": "PromptHMR (official SPEC)",
    "prompthmr_nospec": "PromptHMR (no-SPEC adapter)",
}

METRICS = [
    "W-MPJPE_mm",
    "WA-MPJPE_mm",
    "MPJPE_mm",
    "PA-MPJPE_mm",
    "MPVPE_mm",
    "Accel_mm_frame2",
    "ATE-Sim3_m",
    "ATE-SE3_m",
    "IDF1",
    "Coverage",
    "Precision",
    "Detection_precision",
]

PAIRWISE_METRICS = [
    "W-MPJPE_mm",
    "WA-MPJPE_mm",
    "MPJPE_mm",
    "PA-MPJPE_mm",
    "MPVPE_mm",
    "ATE-Sim3_m",
    "ATE-SE3_m",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def fmt(value: float | None, digits: int = 6) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_value(row: dict[str, str], metric: str) -> float | None:
    if metric == "Precision":
        value = finite(row.get("Precision"))
        return value if value is not None else finite(row.get("Detection_precision"))
    if metric == "ATE-Sim3_m":
        value = finite(row.get("ATE-Sim3_m"))
        return value if value is not None else finite(row.get("ATE_Sim3_m"))
    if metric == "ATE-SE3_m":
        value = finite(row.get("ATE-SE3_m"))
        return value if value is not None else finite(row.get("ATE_SE3_m"))
    return finite(row.get(metric))


def method_summary(
    dataset: str,
    scope: str,
    rows: list[dict[str, str]],
    formal_n: int,
    cluster_n: int,
    aggregation: str,
    source: Path,
) -> dict[str, str]:
    assert len(rows) == formal_n, (dataset, scope, len(rows), formal_n)
    case_ids = [row["case_id"] for row in rows]
    assert len(set(case_ids)) == formal_n, (dataset, scope, "duplicate case IDs")
    statuses = Counter(row.get("status", "internal_complete") or "internal_complete" for row in rows)
    failure_statuses = {"failed", "inference_failed", "error"}
    method = rows[0]["method"]
    result: dict[str, str] = {
        "dataset": dataset,
        "method_key": method,
        "method": DISPLAY.get(method, method),
        "scope": scope,
        "formal_case_n": str(formal_n),
        "cluster_n": str(cluster_n),
        "aggregation": aggregation,
        "inference_attempt_n": str(formal_n),
        "inference_failure_n": str(sum(statuses.get(status, 0) for status in failure_statuses)),
        "status_counts": ";".join(f"{key}:{value}" for key, value in sorted(statuses.items())),
        "coverage_denominator_n": str(formal_n),
        "idf1_denominator_n": str(formal_n),
        "source": str(source.relative_to(ROOT)),
        "source_sha256": sha256(source),
    }
    for metric in METRICS:
        values = [value for row in rows if (value := metric_value(row, metric)) is not None]
        result[f"{metric}_support_n"] = str(len(values))
        # Availability/identity/precision are full-protocol quantities.  A
        # failed or empty prediction contributes zero instead of disappearing
        # from the denominator.  Geometry remains conditional on finite support.
        if metric in {"IDF1", "Coverage", "Precision", "Detection_precision"}:
            result[f"{metric}_mean"] = fmt(sum(values) / formal_n)
        else:
            result[f"{metric}_mean"] = fmt(mean(values))
    return result


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_dataset(dataset: str, config: dict[str, object]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    internal_path = Path(config["internal_csv"])
    external_path = Path(config["external_csv"])
    internal = read_csv(internal_path)
    external = read_csv(external_path)
    strict_key = str(config["strict_method"])
    bridge_key = str(config["bridge_method"])
    formal_n = int(config["formal_n"])
    cluster_n = int(config["cluster_n"])
    aggregation = str(config["aggregation"])

    formal_ids = sorted({row["case_id"] for row in internal if row["method"] == bridge_key})
    assert len(formal_ids) == formal_n, (dataset, "formal ID count", len(formal_ids), formal_n)
    formal_set = set(formal_ids)
    (OUT / f"{dataset.lower().replace('+', 'plus').replace('-', '_')}_formal_case_ids.txt").write_text(
        "\n".join(formal_ids) + "\n", encoding="utf-8"
    )

    summaries: list[dict[str, str]] = []
    by_method: dict[str, dict[str, dict[str, str]]] = {}
    for key in [strict_key, bridge_key]:
        rows = [row for row in internal if row["method"] == key and row["case_id"] in formal_set]
        summaries.append(method_summary(dataset, "internal formal", rows, formal_n, cluster_n, aggregation, internal_path))
        by_method[key] = {row["case_id"]: row for row in rows}

    external_keys = sorted({row["method"] for row in external})
    for key in external_keys:
        rows = [row for row in external if row["method"] == key and row["case_id"] in formal_set]
        summaries.append(method_summary(dataset, "external executable", rows, formal_n, cluster_n, aggregation, external_path))
        by_method[key] = {row["case_id"]: row for row in rows}

    pairwise: list[dict[str, str]] = []
    bridge_by_id = by_method[bridge_key]
    for external_key in external_keys:
        external_by_id = by_method[external_key]
        for metric in PAIRWISE_METRICS:
            ids = [
                case_id
                for case_id in formal_ids
                if metric_value(external_by_id[case_id], metric) is not None
                and metric_value(bridge_by_id[case_id], metric) is not None
            ]
            external_values = [metric_value(external_by_id[case_id], metric) for case_id in ids]
            bridge_values = [metric_value(bridge_by_id[case_id], metric) for case_id in ids]
            assert all(value is not None for value in external_values + bridge_values)
            ext_mean = mean([float(value) for value in external_values])
            bridge_mean = mean([float(value) for value in bridge_values])
            pairwise.append(
                {
                    "dataset": dataset,
                    "external_method_key": external_key,
                    "external_method": DISPLAY.get(external_key, external_key),
                    "reference_method": "BRIDGE3R",
                    "metric": metric,
                    "formal_case_n": str(formal_n),
                    "paired_support_n": str(len(ids)),
                    "external_mean": fmt(ext_mean),
                    "bridge3r_same_cases_mean": fmt(bridge_mean),
                    "external_minus_bridge3r": fmt(ext_mean - bridge_mean) if ext_mean is not None and bridge_mean is not None else "",
                    "case_id_manifest": f"{dataset.lower().replace('+', 'plus').replace('-', '_')}_formal_case_ids.txt",
                }
            )
    return summaries, pairwise


def compact_method_row(row: dict[str, str]) -> dict[str, str]:
    """Project the machine-reaggregated rows into the human audit ledger."""
    return {
        "experiment_id": row["dataset"],
        "dataset": row["dataset"].split("-")[0],
        "protocol": row["dataset"],
        "method_or_component": row["method"],
        "role": row["scope"],
        "formal_case_n": row["formal_case_n"],
        "cluster_n": row["cluster_n"],
        "aggregation": row["aggregation"],
        "inference_attempt_n": row["inference_attempt_n"],
        "inference_failure_n": row["inference_failure_n"],
        "coverage_denominator_n": row["coverage_denominator_n"],
        "idf1_denominator_n": row["idf1_denominator_n"],
        "w_support_case_n": row["W-MPJPE_mm_support_n"],
        "w_support_cluster_n": "",
        "wa_support_case_n": row["WA-MPJPE_mm_support_n"],
        "wa_support_cluster_n": "",
        "local_support_case_n": row["MPJPE_mm_support_n"],
        "local_support_cluster_n": "",
        "ate_sim3_support_case_n": row["ATE-Sim3_m_support_n"],
        "ate_sim3_support_cluster_n": "",
        "other_support": row["status_counts"],
        "source": row["source"],
        "publication_status": "candidate after denominator-aware table rewrite",
        "notes": "Automatically reaggregated on the formal case-ID manifest; geometry is conditional on finite support.",
    }


def additional_ledger_rows() -> list[dict[str, str]]:
    """Static rows for retained reports whose raw per-case artifacts are absent.

    These rows deliberately contain only denominator facts that are stated in
    the retained audit reports.  Blank cells mean that the retained report does
    not establish that cardinality; they must not be guessed or imputed.
    """
    artifact_root = "ICLR-paper/bridge3r_iclr2027/versions/v021_20260828_nine_page_expansion/manuscript/artifacts"
    rows = [
        {
            "experiment_id": "EgoBody-CS150",
            "dataset": "EgoBody",
            "protocol": "43 recordings x 3 angle strata; 150 frames",
            "method_or_component": "Strict Human3R",
            "role": "internal formal",
            "formal_case_n": "129",
            "cluster_n": "43",
            "aggregation": "mean 3 cases within recording, then recording macro",
            "inference_attempt_n": "129",
            "inference_failure_n": "0",
            "coverage_denominator_n": "129",
            "idf1_denominator_n": "129",
            "w_support_case_n": "129",
            "w_support_cluster_n": "43",
            "wa_support_case_n": "129",
            "wa_support_cluster_n": "43",
            "local_support_case_n": "129",
            "local_support_cluster_n": "43",
            "ate_sim3_support_case_n": "129",
            "ate_sim3_support_cluster_n": "43",
            "other_support": "paired-statistics unit: 43 recordings",
            "source": f"{artifact_root}/egobody_v20/EGOBODY_FINAL_RESULTS.md",
            "publication_status": "primary candidate",
            "notes": "Raw per-case artifact was deleted after sealing; retained report and hashes remain.",
        },
        {
            "experiment_id": "EgoBody-CS150",
            "dataset": "EgoBody",
            "protocol": "43 recordings x 3 angle strata; 150 frames",
            "method_or_component": "BRIDGE3R",
            "role": "internal formal",
            "formal_case_n": "129",
            "cluster_n": "43",
            "aggregation": "mean 3 cases within recording, then recording macro",
            "inference_attempt_n": "129",
            "inference_failure_n": "0",
            "coverage_denominator_n": "129",
            "idf1_denominator_n": "129",
            "w_support_case_n": "129",
            "w_support_cluster_n": "43",
            "wa_support_case_n": "129",
            "wa_support_cluster_n": "43",
            "local_support_case_n": "129",
            "local_support_cluster_n": "43",
            "ate_sim3_support_case_n": "129",
            "ate_sim3_support_cluster_n": "43",
            "other_support": "paired-statistics unit: 43 recordings",
            "source": f"{artifact_root}/egobody_v20/EGOBODY_FINAL_RESULTS.md",
            "publication_status": "primary candidate",
            "notes": "Raw per-case artifact was deleted after sealing; retained report and hashes remain.",
        },
        {
            "experiment_id": "EgoBody-CS150",
            "dataset": "EgoBody",
            "protocol": "43 recordings x 3 angle strata; 150 frames",
            "method_or_component": "TRACE (official adapter)",
            "role": "external executable",
            "formal_case_n": "129",
            "cluster_n": "43",
            "aggregation": "mean 3 cases within recording, then recording macro",
            "inference_attempt_n": "129",
            "inference_failure_n": "0",
            "coverage_denominator_n": "129",
            "idf1_denominator_n": "129",
            "w_support_case_n": "22",
            "w_support_cluster_n": "15",
            "wa_support_case_n": "35",
            "wa_support_cluster_n": "17",
            "local_support_case_n": "35",
            "local_support_cluster_n": "17",
            "ate_sim3_support_case_n": "0",
            "ate_sim3_support_cluster_n": "0",
            "other_support": "zero-match cases 94/129; ATE is N/A by output contract",
            "source": "external_baselines/bridge3r_eval/TRACE_EGOBODY_TEST.md",
            "publication_status": "external summary candidate",
            "notes": "Geometry is conditional on accepted matches; camera trajectory is not independently estimated.",
        },
        {
            "experiment_id": "EgoBody-CS150",
            "dataset": "EgoBody",
            "protocol": "43 recordings x 3 angle strata; 150 frames",
            "method_or_component": "PromptHMR (official SPEC)",
            "role": "external executable",
            "formal_case_n": "129",
            "cluster_n": "43",
            "aggregation": "mean 3 cases within recording, then recording macro",
            "inference_attempt_n": "129",
            "inference_failure_n": "0",
            "coverage_denominator_n": "129",
            "idf1_denominator_n": "129",
            "w_support_case_n": "64",
            "w_support_cluster_n": "34",
            "wa_support_case_n": "87",
            "wa_support_cluster_n": "36",
            "local_support_case_n": "87",
            "local_support_cluster_n": "36",
            "ate_sim3_support_case_n": "129",
            "ate_sim3_support_cluster_n": "43",
            "other_support": "zero-match cases 42/129",
            "source": "external_baselines/bridge3r_eval/PROMPTHMR_SPEC_EGOBODY_TEST.md",
            "publication_status": "external summary candidate",
            "notes": "Geometry is conditional on accepted matches; all availability metrics retain the full manifest.",
        },
        {
            "experiment_id": "EgoBody-CS150",
            "dataset": "EgoBody",
            "protocol": "43 recordings x 3 angle strata; 150 frames",
            "method_or_component": "PromptHMR (no-SPEC adapter)",
            "role": "external executable adapter",
            "formal_case_n": "129",
            "cluster_n": "43",
            "aggregation": "mean 3 cases within recording, then recording macro",
            "inference_attempt_n": "129",
            "inference_failure_n": "0",
            "coverage_denominator_n": "129",
            "idf1_denominator_n": "129",
            "w_support_case_n": "",
            "w_support_cluster_n": "33",
            "wa_support_case_n": "",
            "wa_support_cluster_n": "36",
            "local_support_case_n": "",
            "local_support_cluster_n": "36",
            "ate_sim3_support_case_n": "129",
            "ate_sim3_support_cluster_n": "43",
            "other_support": "zero-match cases 38/129; case-level W/WA support not retained in report",
            "source": "external_baselines/bridge3r_eval/PROMPTHMR_EGOBODY_TEST.md",
            "publication_status": "supplement/adapter diagnostic",
            "notes": "Blank case supports must remain blank unless the sealed per-case artifact is restored.",
        },
    ]

    for protocol in ["AIST++-CS150", "AIST++-MC150-3", "AIST++-MC150-4"]:
        if protocol == "AIST++-CS150":
            methods = ["Strict Human3R", "BRIDGE3R", "PromptHMR (official offline)"]
            source = f"{artifact_root}/aist_cs150_formal/AIST_CS150_FORMAL_REPORT.md"
        else:
            methods = ["Strict Human3R", "Clean reset", "Coarse alignment only", "Coarse alignment + identity", "BRIDGE3R"]
            source = f"{artifact_root}/aist_multicut_formal/AIST_MULTICUT_FORMAL_REPORT.md"
        for method in methods:
            rows.append(
                {
                    "experiment_id": protocol,
                    "dataset": "AIST++",
                    "protocol": protocol,
                    "method_or_component": method,
                    "role": "single-person diagnostic" if protocol == "AIST++-CS150" else "component/event-scaling diagnostic",
                    "formal_case_n": "100",
                    "cluster_n": "100",
                    "aggregation": "source macro",
                    "inference_attempt_n": "100",
                    "inference_failure_n": "0",
                    "coverage_denominator_n": "100",
                    "idf1_denominator_n": "",
                    "w_support_case_n": "",
                    "w_support_cluster_n": "",
                    "wa_support_case_n": "",
                    "wa_support_cluster_n": "",
                    "local_support_case_n": "100",
                    "local_support_cluster_n": "100",
                    "ate_sim3_support_case_n": "",
                    "ate_sim3_support_cluster_n": "",
                    "other_support": "all reported components N=100",
                    "source": source,
                    "publication_status": "supplement candidate",
                    "notes": "CS150 and each multi-cut protocol have separate frozen official pose_test manifests and must not be pooled.",
                }
            )

    rows.extend(
        [
            {
                "experiment_id": "AIST++-GVHMR-pilot",
                "dataset": "AIST++",
                "protocol": "pre-registered hard-cut pilot",
                "method_or_component": "GVHMR",
                "role": "availability gate only",
                "formal_case_n": "12",
                "cluster_n": "12",
                "aggregation": "case count",
                "inference_attempt_n": "12",
                "inference_failure_n": "",
                "coverage_denominator_n": "12",
                "idf1_denominator_n": "",
                "w_support_case_n": "",
                "w_support_cluster_n": "",
                "wa_support_case_n": "",
                "wa_support_cluster_n": "",
                "local_support_case_n": "2",
                "local_support_cluster_n": "2",
                "ate_sim3_support_case_n": "",
                "ate_sim3_support_cluster_n": "",
                "other_support": "one-sided raw tracker support 2/12",
                "source": f"{artifact_root}/aist_cs150_formal/AIST_CS150_FORMAL_REPORT.md",
                "publication_status": "private pilot; exclude from formal ranking",
                "notes": "Did not pass availability gate and therefore was not expanded to the 100-source Test table.",
            },
            {
                "experiment_id": "Harmony4D-association",
                "dataset": "Harmony4D",
                "protocol": "boundary association audit",
                "method_or_component": "BRIDGE3R association",
                "role": "mechanism audit",
                "formal_case_n": "88",
                "cluster_n": "25",
                "aggregation": "pair micro plus case macro",
                "inference_attempt_n": "88",
                "inference_failure_n": "0",
                "coverage_denominator_n": "",
                "idf1_denominator_n": "88",
                "w_support_case_n": "",
                "w_support_cluster_n": "",
                "wa_support_case_n": "",
                "wa_support_cluster_n": "",
                "local_support_case_n": "",
                "local_support_cluster_n": "",
                "ate_sim3_support_case_n": "",
                "ate_sim3_support_cluster_n": "",
                "other_support": "84/88 cases evaluable; 142/147 endpoint pairs correct; continuation 142/176",
                "source": f"{artifact_root}/harmony4d_boundary_association_table.tex",
                "publication_status": "supplement candidate",
                "notes": "Pair, case, continuation, and IDF1 denominators are distinct and must not be interchanged.",
            },
            {
                "experiment_id": "Harmony4D-multicut",
                "dataset": "Harmony4D",
                "protocol": "4 captures / 8 boundaries",
                "method_or_component": "Strict Human3R and BRIDGE3R",
                "role": "repeated-transition diagnostic",
                "formal_case_n": "4",
                "cluster_n": "4",
                "aggregation": "capture macro",
                "inference_attempt_n": "4",
                "inference_failure_n": "0",
                "coverage_denominator_n": "4",
                "idf1_denominator_n": "4",
                "w_support_case_n": "4",
                "w_support_cluster_n": "4",
                "wa_support_case_n": "4",
                "wa_support_cluster_n": "4",
                "local_support_case_n": "4",
                "local_support_cluster_n": "4",
                "ate_sim3_support_case_n": "4",
                "ate_sim3_support_cluster_n": "4",
                "other_support": "8 cut boundaries",
                "source": f"{artifact_root}/harmony4d_multicut_table.tex",
                "publication_status": "limited supplement evidence",
                "notes": "Small diagnostic set; do not describe as a broad multi-cut benchmark.",
            },
            {
                "experiment_id": "Harmony4D-lambda",
                "dataset": "Harmony4D",
                "protocol": "training-only sensitivity",
                "method_or_component": "shared-translation blend",
                "role": "hyperparameter sensitivity",
                "formal_case_n": "12",
                "cluster_n": "12",
                "aggregation": "case macro",
                "inference_attempt_n": "12",
                "inference_failure_n": "0",
                "coverage_denominator_n": "9",
                "idf1_denominator_n": "9",
                "w_support_case_n": "9",
                "w_support_cluster_n": "9",
                "wa_support_case_n": "9",
                "wa_support_cluster_n": "9",
                "local_support_case_n": "9",
                "local_support_cluster_n": "9",
                "ate_sim3_support_case_n": "9",
                "ate_sim3_support_cluster_n": "9",
                "other_support": "9 evaluator-complete; seam support 8; 3 uniformly evaluator-unavailable",
                "source": f"{artifact_root}/harmony4d_lambda_sensitivity.tex",
                "publication_status": "supplement candidate",
                "notes": "Training-only descriptive sweep; fixed lambda=0.5 is not selected from Test.",
            },
            {
                "experiment_id": "EgoBody-runtime-existing",
                "dataset": "EgoBody",
                "protocol": "existing component timing on Test",
                "method_or_component": "pipeline components",
                "role": "runtime diagnostic",
                "formal_case_n": "129",
                "cluster_n": "43",
                "aggregation": "case mean; mixed component timing",
                "inference_attempt_n": "129",
                "inference_failure_n": "0",
                "coverage_denominator_n": "",
                "idf1_denominator_n": "",
                "w_support_case_n": "",
                "w_support_cluster_n": "",
                "wa_support_case_n": "",
                "wa_support_cluster_n": "",
                "local_support_case_n": "",
                "local_support_cluster_n": "",
                "ate_sim3_support_case_n": "",
                "ate_sim3_support_cluster_n": "",
                "other_support": "host RSS only; GPU peak unavailable; deployed single-method FPS unavailable",
                "source": f"{artifact_root}/egobody_v20/runtime_summary.json",
                "publication_status": "private diagnostic; insufficient for formal runtime claim",
                "notes": "Do not report as standardized end-to-end throughput or peak GPU memory.",
            },
            {
                "experiment_id": "Harmony4D-legacy-five-method-conflict",
                "dataset": "Harmony4D",
                "protocol": "legacy 100-case table with 88 internal geometry supports",
                "method_or_component": "BRIDGE3R legacy row",
                "role": "conflict sentinel",
                "formal_case_n": "100",
                "cluster_n": "25",
                "aggregation": "legacy case macro",
                "inference_attempt_n": "100",
                "inference_failure_n": "0",
                "coverage_denominator_n": "100",
                "idf1_denominator_n": "100",
                "w_support_case_n": "88",
                "w_support_cluster_n": "",
                "wa_support_case_n": "88",
                "wa_support_cluster_n": "",
                "local_support_case_n": "88",
                "local_support_cluster_n": "",
                "ate_sim3_support_case_n": "88",
                "ate_sim3_support_cluster_n": "",
                "other_support": "legacy W/WA/ATE = 584.0/266.9/0.019; unified 88-case = 519.6/247.0/0.0168",
                "source": f"{artifact_root}/harmony4d_final/harmony4d_five_method_table.tex",
                "publication_status": "DO NOT USE; must regenerate",
                "notes": "Historical result binding conflicts with the current unified frozen 88-case method row.",
            },
        ]
    )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_summaries: list[dict[str, str]] = []
    all_pairwise: list[dict[str, str]] = []
    for dataset, config in DATASETS.items():
        summaries, pairwise = build_dataset(dataset, config)
        all_summaries.extend(summaries)
        all_pairwise.extend(pairwise)
    write_csv(OUT / "FORMAL_METHOD_SUPPORT.csv", all_summaries)
    write_csv(OUT / "PAIRWISE_EXTERNAL_GEOMETRY.csv", all_pairwise)
    ledger = [compact_method_row(row) for row in all_summaries]
    ledger.extend(additional_ledger_rows())
    write_csv(OUT / "DENOMINATOR_LEDGER.csv", ledger)
    print(
        f"wrote {len(all_summaries)} support rows, {len(all_pairwise)} pairwise rows, "
        f"and {len(ledger)} ledger rows to {OUT}"
    )


if __name__ == "__main__":
    main()
