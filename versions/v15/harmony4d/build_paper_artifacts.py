#!/usr/bin/env python3
"""Build paper tables, stratified analyses, and figures from frozen H4D reports."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


PRIMARY = "m15_safe_boundary_permutation_causal_gru"
SELECTED_METHODS = (
    "m0_strict_human3r",
    "m1_clean_reset",
    "m2_no_v9_raw_se3",
    "m3_b0_only",
    "m4_b0_identity",
    "m5_b0_identity_brtc",
    "m6_b0_identity_brtc_c1",
    "m7_full_v15_oracle",
    "m13_b0_boundary_permutation_id",
    "m15_safe_boundary_permutation_causal_gru",
)
DISPLAY = {
    "m0_strict_human3r": "Strict Human3R",
    "m1_clean_reset": "Clean reset",
    "m2_no_v9_raw_se3": "No-V9 raw SE(3)",
    "m3_b0_only": "B0 only",
    "m4_b0_identity": "B0 + frozen ID",
    "m5_b0_identity_brtc": "B0 + ID + BRTC",
    "m6_b0_identity_brtc_c1": "B0 + ID + BRTC + C1",
    "m7_full_v15_oracle": "Full v15 (oracle cut)",
    "m13_b0_boundary_permutation_id": "B0 + boundary ID",
    "m15_safe_boundary_permutation_causal_gru": "Movie3R (causal)",
}
ANGLE_ORDER = ("small", "medium", "large", "extreme")
LITERATURE = {
    "Multi-THuMBS": {
        "W-MPJPE_mm": 221.0, "WA-MPJPE_mm": 116.9, "MPJPE_mm": 215.9,
        "MPVPE_mm": 278.3, "Accel": 17.4, "ATE_m": 0.7, "IDs": 0.46,
    },
    "HSfM_dagger": {"MPVPE_mm": 257.6},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, nargs="+", required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--primary", default=PRIMARY)
    return parser.parse_args()


def nested(value: dict[str, Any], *path: str) -> float | None:
    current: Any = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    if current is None:
        return None
    try:
        number = float(current)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def mean(values: list[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(valid)) if valid else None


def distribution(values: list[float | None]) -> dict[str, Any]:
    array = np.asarray([value for value in values if value is not None], dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {key: None for key in ("count", "mean", "median", "p90", "p95", "std")}
    return {
        "count": int(len(array)), "mean": float(array.mean()),
        "median": float(np.median(array)), "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)), "std": float(array.std()),
    }


def load_reports(roots: list[Path]) -> list[dict[str, Any]]:
    output = []
    seen = set()
    for root in roots:
        for path in sorted(root.rglob("h4d_test_*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if payload.get("schema_version") != "Movie3R-Harmony4D-evaluation-v1":
                continue
            if payload["case_id"] in seen:
                raise ValueError(f"Duplicate report: {payload['case_id']}")
            seen.add(payload["case_id"])
            output.append(payload)
    return output


def load_runtimes(roots: list[Path]) -> list[dict[str, Any]]:
    output = []
    seen = set()
    for root in roots:
        for path in sorted(root.rglob("*.runtime.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != "Movie3R-Harmony4D-runtime-cache-v1":
                continue
            case_id = payload["record"]["case_id"]
            if case_id in seen:
                raise ValueError(f"Duplicate runtime: {case_id}")
            seen.add(case_id)
            output.append(payload)
    return output


def result_metrics(result: dict[str, Any]) -> dict[str, float | None]:
    named = result["multi_thumbs_named_provisional"]
    return {
        "W-MPJPE_mm": nested(named, "w_mpjpe_mm", "mean"),
        "WA-MPJPE_mm": nested(named, "wa_mpjpe_mm", "mean"),
        "MPJPE_mm": nested(named, "mpjpe_mm", "mean"),
        "MPVPE_mm": nested(named, "mpvpe_mm", "mean"),
        "Accel_mm_frame2": nested(named, "accel_delta2_mm_per_frame2", "mean"),
        "ATE_Sim3_m": nested(named, "ate_sim3_m", "mean"),
        "ATE_SE3_m": nested(named, "ate_se3_m", "mean"),
        "ATE_metric_initial_SE3_m": nested(named, "ate_metric_initial_se3_m", "mean"),
        "Boundary_camera_t_m": nested(result, "camera", "first_post_translation_m"),
        "Boundary_camera_R_deg": nested(result, "camera", "first_post_rotation_deg"),
        "Boundary_root_m": nested(result, "fixed_world", "first_post_root_m", "mean"),
        "Boundary_CHRGE_m": nested(result, "camera_human_relative", "first_post_root_gauge_m", "mean"),
        "Boundary_pair_vector_m": nested(result, "pairwise_layout", "first_post_vector_m", "mean"),
        "Seam_camera_t_m": nested(result, "cut_seam", "camera_translation_excess_m"),
        "Seam_camera_R_deg": nested(result, "cut_seam", "camera_rotation_excess_deg"),
        "Seam_root_m": nested(result, "cut_seam", "root_excess_m"),
        "Seam_CHRGE_m": nested(result, "cut_seam", "camera_human_relative_excess_m"),
        "IDs": nested(result, "identity", "ids_total"),
        "IDF1": nested(result, "identity", "idf1"),
        "Coverage": nested(result, "coverage", "coverage"),
        "Detection_precision": nested(result, "coverage", "precision"),
    }


def tex_value(value: float | None, digits: int = 1, percent: bool = False) -> str:
    if value is None or not math.isfinite(float(value)):
        return "--"
    number = 100.0 * float(value) if percent else float(value)
    return f"{number:.{digits}f}"


def write_tex_table(path: Path, headers: list[str], rows: list[list[str]], align: str | None = None) -> None:
    alignment = align or ("l" + "r" * (len(headers) - 1))
    lines = [
        "% Auto-generated; Movie3R-Harmony4D-CrossShot-v1 unless noted.",
        f"\\begin{{tabular}}{{{alignment}}}", "\\toprule",
        " & ".join(headers) + " \\\\", "\\midrule",
    ]
    lines.extend(" & ".join(row) + " \\\\" for row in rows)
    lines.extend(("\\bottomrule", "\\end{tabular}", ""))
    path.write_text("\n".join(lines), encoding="utf-8")


def runtime_artifacts(runtimes: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    rows = []
    for payload in runtimes:
        runtime = payload["runtime"]
        m0 = runtime["m0_forward"]
        shadow = runtime["oracle_shadow_forward"]
        raw = runtime["oracle_raw_post_forward"]
        peak_vram = max(float(m0["peak_vram_bytes"]), float(shadow["peak_vram_bytes"]), float(raw["peak_vram_bytes"]))
        row = {
            "case_id": payload["record"]["case_id"],
            "sequence": payload["record"]["sequence"],
            "angle_stratum": payload["record"]["angle_stratum"],
            "causal_detector_s": float(runtime["causal_gru_detector"]["seconds"]),
            "static_detector_s": float(runtime["static_logistic_detector"]["seconds"]),
            "strict_forward_s": float(m0["seconds"]),
            "strict_forward_fps": float(m0["fps"]),
            "shadow_forward_s": float(shadow["seconds"]),
            "raw_post_forward_s": float(raw["seconds"]),
            "current_model_s": float(shadow["seconds"] + raw["seconds"]),
            "explicit_geometry_s": float(runtime["oracle_explicit_geometry_seconds"]),
            "total_process_s": float(payload["total_process_seconds"]),
            "end_to_end_fps_including_detector_cache": 150.0 / max(float(payload["total_process_seconds"]), 1e-9),
            "peak_vram_gib": peak_vram / 2**30,
            "peak_rss_gib": float(payload["environment"]["process_peak_rss_bytes"]) / 2**30,
            "gpu": payload["environment"].get("gpu"),
        }
        rows.append(row)
    with (output / "runtime_cases.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    numeric = [key for key in rows[0] if key not in {"case_id", "sequence", "angle_stratum", "gpu"}]
    summary = {
        "case_count": len(rows),
        "gpu_names": sorted({str(row["gpu"]) for row in rows}),
        "metrics": {key: distribution([float(row[key]) for row in rows]) for key in numeric},
        "scope_note": (
            "total_process_s includes RGB detector feature extraction, all frozen ablation forwards, "
            "SMPL-X-to-SMPL packing, compressed cache serialization, and SHA256; it is not single-method deployment latency."
        ),
    }
    (output / "runtime_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    means = {key: summary["metrics"][key]["mean"] for key in numeric}
    write_tex_table(output / "efficiency_table.tex", ["Component", "Mean", "Unit"], [
        ["Causal RGB detector", tex_value(means["causal_detector_s"], 2), "s / 150 frames"],
        ["Strict Human3R forward", tex_value(means["strict_forward_s"], 2), "s / 150 frames"],
        ["Shadow proposal forward", tex_value(means["shadow_forward_s"], 2), "s / 76 frames"],
        ["Clean post forward", tex_value(means["raw_post_forward_s"], 2), "s / 75 frames"],
        ["Explicit geometry", tex_value(1000.0 * means["explicit_geometry_s"], 2), "ms / boundary"],
        ["Full experiment process", tex_value(means["total_process_s"], 2), "s / case"],
        ["Peak VRAM", tex_value(means["peak_vram_gib"], 2), "GiB"],
        ["Peak process RAM", tex_value(means["peak_rss_gib"], 2), "GiB"],
    ])
    return {"rows": rows, "summary": summary}


def runtime_detector_summary(runtimes: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for label, key in (("causal_gru", "causal_gru_detector"), ("static_logistic", "static_logistic_detector")):
        tp = fp = fn = first_boundary = 0
        seconds, squared_errors = [], []
        for payload in runtimes:
            boundary = int(payload["record"]["boundary_index"])
            detector = payload["runtime"][key]
            labels = np.asarray(detector["labels"], dtype=np.int64)
            target = np.zeros_like(labels); target[boundary] = 1
            tp += int(((labels == 1) & (target == 1)).sum())
            fp += int(((labels == 1) & (target == 0)).sum())
            fn += int(((labels == 0) & (target == 1)).sum())
            first_boundary += int(detector.get("first_positive_index") == boundary)
            seconds.append(float(detector["seconds"]))
            for row in detector.get("rows", []):
                index = int(row["pair_idx"])
                squared_errors.append((float(row["prob"]) - float(target[index])) ** 2)
        output[label] = {
            "case_count": len(runtimes), "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "first_positive_boundary_rate": first_boundary / max(len(runtimes), 1),
            "latency_seconds_mean": float(np.mean(seconds)),
            "brier": float(np.mean(squared_errors)) if squared_errors else None,
        }
    return output


def group_rows(reports: list[dict[str, Any]], methods: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = []
    for report in reports:
        for method in methods:
            rows.append({
                "case_id": report["case_id"], "sequence": report["record"]["sequence"],
                "angle_stratum": report["record"]["angle_stratum"], "method": method,
                **result_metrics(report["methods"][method]),
            })
    return rows


def grouped_csv(rows: list[dict[str, Any]], group: str, output: Path) -> list[dict[str, Any]]:
    metrics = [key for key in rows[0] if key not in {"case_id", "sequence", "angle_stratum", "method"}]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row[group]), str(row["method"]))].append(row)
    result = []
    order = ANGLE_ORDER if group == "angle_stratum" else sorted({str(row[group]) for row in rows})
    for value in order:
        for method in dict.fromkeys(row["method"] for row in rows):
            values = grouped.get((value, method), [])
            if not values:
                continue
            result.append({group: value, "method": method, "case_count": len(values), **{
                metric: mean([row[metric] for row in values]) for metric in metrics
            }})
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result[0]))
        writer.writeheader(); writer.writerows(result)
    return result


def plot_angle(rows: list[dict[str, Any]], output: Path) -> None:
    methods = ("m0_strict_human3r", "m3_b0_only", "m13_b0_boundary_permutation_id", PRIMARY)
    metrics = (("W-MPJPE_mm", "W-MPJPE (mm)"), ("ATE_Sim3_m", "ATE Sim(3) (m)"),
               ("IDs", "ID switches / clip"), ("IDF1", "IDF1"))
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6), constrained_layout=True)
    for axis, (metric, label) in zip(axes.flat, metrics):
        for method in methods:
            points = [next((row[metric] for row in rows if row["angle_stratum"] == angle and row["method"] == method), np.nan) for angle in ANGLE_ORDER]
            axis.plot(ANGLE_ORDER, points, marker="o", linewidth=1.8, label=DISPLAY.get(method, method))
        axis.set_ylabel(label); axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7.5, ncol=2)
    fig.savefig(output / "angle_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(output / "angle_sensitivity.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_sequences(rows: list[dict[str, Any]], output: Path) -> None:
    subset = [row for row in rows if row["method"] in {"m0_strict_human3r", PRIMARY}]
    sequences = sorted({row["sequence"] for row in subset})
    x = np.arange(len(sequences)); width = 0.38
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), constrained_layout=True)
    for axis, metric, ylabel in ((axes[0], "W-MPJPE_mm", "W-MPJPE (mm)"), (axes[1], "IDF1", "IDF1")):
        for offset, method in ((-width / 2, "m0_strict_human3r"), (width / 2, PRIMARY)):
            values = [next((row[metric] for row in subset if row["sequence"] == sequence and row["method"] == method), np.nan) for sequence in sequences]
            axis.bar(x + offset, values, width, label=DISPLAY[method])
        axis.set_ylabel(ylabel); axis.grid(axis="y", alpha=0.25)
    axes[0].set_xticks(x, [])
    axes[1].set_xticks(x, sequences, rotation=25, ha="right"); axes[0].legend()
    fig.savefig(output / "sequence_generalization.pdf", bbox_inches="tight")
    fig.savefig(output / "sequence_generalization.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    aggregate = json.loads(args.aggregate.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    reports = load_reports([path.resolve() for path in args.metrics])
    runtimes = load_runtimes([path.resolve() for path in args.predictions])
    if len(reports) != aggregate["case_count"] or len(runtimes) != audit["runtime_cases"]:
        raise ValueError({"reports": len(reports), "aggregate": aggregate["case_count"], "runtimes": len(runtimes), "audit": audit["runtime_cases"]})
    summary = aggregate["summary"]
    selected = [method for method in SELECTED_METHODS if method in summary]

    human_rows, camera_rows, boundary_rows = [], [], []
    for method in selected:
        values = summary[method]["clip_macro"]
        label = DISPLAY.get(method, method).replace("_", "\\_")
        human_rows.append([label, tex_value(values["W-MPJPE_mm"]), tex_value(values["WA-MPJPE_mm"]),
                           tex_value(values["MPJPE_mm"]), tex_value(values["MPVPE_mm"]),
                           tex_value(values["Accel_mm_frame2"], 2), tex_value(values["Coverage"], 1, True)])
        camera_rows.append([label, tex_value(values["ATE_Sim3_m"], 3), tex_value(values["ATE_SE3_m"], 3),
                            tex_value(values["Boundary_camera_t_m"], 3), tex_value(values["Boundary_camera_R_deg"], 2),
                            tex_value(values["IDs"], 2), tex_value(values["IDF1"], 3)])
        boundary_rows.append([label, tex_value(values["Boundary_root_m"], 3), tex_value(values["Boundary_CHRGE_m"], 3),
                              tex_value(values["Boundary_pair_vector_m"], 3), tex_value(values["Seam_camera_t_m"], 3),
                              tex_value(values["Seam_camera_R_deg"], 2), tex_value(values["Seam_root_m"], 3)])
    write_tex_table(output / "main_human_table.tex", ["Method", "W", "WA", "MPJPE", "MPVPE", "Accel", "Cov.(\\%)"], human_rows)
    write_tex_table(output / "camera_identity_table.tex", ["Method", "ATE-Sim3", "ATE-SE3", "Bnd.-t", "Bnd.-R", "IDs", "IDF1"], camera_rows)
    write_tex_table(output / "boundary_table.tex", ["Method", "Root", "CHRGE", "Pair", "Seam-t", "Seam-R", "Seam-root"], boundary_rows)

    runtime_payload = runtime_artifacts(runtimes, output)
    rows = group_rows(reports, tuple(selected))
    angle_rows = grouped_csv(rows, "angle_stratum", output / "angle_metrics.csv")
    sequence_rows = grouped_csv(rows, "sequence", output / "sequence_metrics.csv")
    plot_angle(angle_rows, output); plot_sequences(sequence_rows, output)

    detectors = runtime_detector_summary(runtimes)
    detector_rows = []
    for name, values in detectors.items():
        detector_rows.append([name.replace("_", "\\_"), str(values["tp"]), str(values["fp"]), str(values["fn"]),
                              tex_value(values["precision"], 3), tex_value(values["recall"], 3), tex_value(values["f1"], 3)])
    write_tex_table(output / "detector_table.tex", ["Detector", "TP", "FP", "FN", "Prec.", "Recall", "F1"], detector_rows)

    os_brtc_accepted = sum(bool(item["geometry"]["observability_safe_brtc"].get("accepted")) for item in runtimes)
    adaptive_accepted = sum(any(bool(row.get("accepted")) for row in item["geometry"].get("boundary_permutation_safe_adaptive", [])) for item in runtimes)
    gate_rows = aggregate.get("gate_rows", [])
    gate = {
        "runtime_case_count": len(runtimes), "evaluable_case_count": len(gate_rows),
        "os_brtc_accepted": int(os_brtc_accepted),
        "boundary_permutation_adaptive_accepted": int(adaptive_accepted),
        "evaluator_safe_gate_accepted": int(sum(bool(row.get("accepted")) for row in gate_rows)),
        "evaluator_any_metric_worse": int(sum(bool(row.get("any_metric_worse")) for row in gate_rows)),
        "evaluator_catastrophic_harm": int(sum(bool(row.get("catastrophic_harm")) for row in gate_rows)),
    }
    (output / "gate_summary.json").write_text(json.dumps(gate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_tex_table(output / "gate_table.tex", ["Gate statistic", "Count", "Denominator"], [
        ["OS-BRTC accepted", str(gate["os_brtc_accepted"]), str(gate["runtime_case_count"])],
        ["Shared adaptive accepted", str(gate["boundary_permutation_adaptive_accepted"]), str(gate["runtime_case_count"])],
        ["Any accepted metric worse", str(gate["evaluator_any_metric_worse"]), str(gate["evaluable_case_count"])],
        ["Catastrophic accepted harm", str(gate["evaluator_catastrophic_harm"]), str(gate["evaluable_case_count"])],
    ])

    literature_rows = [["Multi-THuMBS (literature)", "221.0", "116.9", "215.9", "278.3", "17.4", "0.700", "0.46"],
                       ["HSfM$^{\\dagger}$ (literature)", "--", "--", "225.6", "257.6", "28.3", "3.200", "1.58"]]
    primary_values = summary[args.primary]["clip_macro"]
    literature_rows.append(["Movie3R (ours; different protocol)", tex_value(primary_values["W-MPJPE_mm"]),
                            tex_value(primary_values["WA-MPJPE_mm"]), tex_value(primary_values["MPJPE_mm"]),
                            tex_value(primary_values["MPVPE_mm"]), tex_value(primary_values["Accel_mm_frame2"], 2),
                            tex_value(primary_values["ATE_Sim3_m"], 3), tex_value(primary_values["IDs"], 2)])
    write_tex_table(output / "literature_reference_table.tex", ["Method", "W", "WA", "MPJPE", "MPVPE", "Accel", "ATE", "IDs"], literature_rows)

    comparisons = {}
    for baseline in ("m0_strict_human3r", "m1_clean_reset", "m3_b0_only", "m13_b0_boundary_permutation_id"):
        if baseline not in summary:
            continue
        comparisons[baseline] = {}
        for metric, higher in (("W-MPJPE_mm", False), ("WA-MPJPE_mm", False), ("ATE_Sim3_m", False),
                               ("IDs", False), ("IDF1", True), ("Coverage", True), ("Seam_root_m", False)):
            primary_value = primary_values.get(metric); baseline_value = summary[baseline]["clip_macro"].get(metric)
            comparisons[baseline][metric] = {
                "primary": primary_value, "baseline": baseline_value,
                "primary_better": None if primary_value is None or baseline_value is None else (
                    primary_value > baseline_value if higher else primary_value < baseline_value
                ),
                "paired_significance": aggregate.get("paired_significance_primary_vs_baseline", {}).get(baseline, {}).get(metric),
            }
    final = {
        "schema_version": "Movie3R-Harmony4D-paper-artifacts-v1",
        "primary_method": args.primary,
        "runtime_cases": len(runtimes), "evaluable_cases": len(reports),
        "audit_status_counts": audit["status_counts"],
        "protocol_local_primary": {
            "clip_macro": primary_values,
            "sequence_macro": summary[args.primary]["sequence_macro"],
            "person_frame_weighted_micro": summary[args.primary]["person_frame_weighted_micro"],
            "confidence_intervals_95": summary[args.primary]["confidence_intervals_95"],
        },
        "comparisons": comparisons,
        "detectors": detectors, "gates": gate,
        "runtime": runtime_payload["summary"], "literature_reference": LITERATURE,
        "literature_caveat": "Multi-THuMBS exact Harmony4D manifest/evaluator is not public; literature values are not a direct leaderboard comparison.",
    }
    (output / "paper_results.json").write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "runtime_cases": len(runtimes), "evaluable_cases": len(reports), "tables": 7, "figures": 4}, indent=2))


if __name__ == "__main__":
    main()
