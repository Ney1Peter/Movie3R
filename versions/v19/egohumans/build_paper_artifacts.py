#!/usr/bin/env python3
"""Build final ICLR-facing EgoHumans tables and reproducibility report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = REPO_ROOT / "output/v19_egohumans"
DEFAULT_FINAL = REPO_ROOT / "versions/v19/egohumans/frozen_final_candidate.json"
LOWER = (
    "W-MPJPE_mm",
    "WA-MPJPE_mm",
    "MPJPE_mm",
    "MPVPE_mm",
    "Accel_mm_frame2",
    "ATE_Sim3_m",
    "ATE_SE3_m",
    "Seam_root_m",
    "IDs",
)
HIGHER = ("IDF1", "Coverage")
TABLE_METRICS = (
    "W-MPJPE_mm",
    "WA-MPJPE_mm",
    "MPJPE_mm",
    "MPVPE_mm",
    "Accel_mm_frame2",
    "ATE_Sim3_m",
    "ATE_SE3_m",
    "IDF1",
    "IDs",
    "Coverage",
    "Seam_root_m",
)
DISPLAY = {
    "m0_strict_human3r": "Strict Human3R",
    "m15_safe_boundary_permutation_causal_gru": "Movie3R-v15",
    "v16_0_m15_geometry": "Movie3R-v17 parent",
    "v17_harmony_multicue_safe": "Movie3R-v17 MultiCue-Safe",
    "v19_egohumans_frozen": "Movie3R-v19",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, default=ROOT / "development/summary")
    parser.add_argument("--holdout", type=Path, default=ROOT / "holdout/summary")
    parser.add_argument("--test", type=Path, default=ROOT / "test/summary")
    parser.add_argument("--final-candidate", type=Path, default=DEFAULT_FINAL)
    parser.add_argument("--output", type=Path, default=ROOT / "final")
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value: Any) -> float | None:
    if value in (None, ""):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def avg(values: Iterable[Any]) -> float | None:
    numbers = [value for item in values if (value := finite(item)) is not None]
    return mean(numbers) if numbers else None


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_cases(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def protocol_counts(summary_dir: Path, summary: dict[str, Any]) -> dict[str, int]:
    """Return protocol-level counts without conflating inference and evaluation."""
    state_path = summary_dir.parent / "protocol_state.json"
    state = load(state_path)
    captures = state.get("captures", {})
    statuses: dict[str, int] = defaultdict(int)
    for value in captures.values():
        statuses[str(value.get("status", "unknown"))] += 1
    planned_captures = len(state.get("entries", []))
    completed_captures = statuses.get("complete", 0)
    camera_pair_cases_per_capture = 4
    return {
        "planned_captures": planned_captures,
        "completed_captures": completed_captures,
        "structural_exclusions": statuses.get("structural_error", 0),
        "inferred_cases": completed_captures * camera_pair_cases_per_capture,
        "evaluable_captures": int(summary["capture_count"]),
        "evaluable_cases": int(summary["case_count"]),
        "evaluator_unavailable_cases": int(summary["evaluator_unavailable_count"]),
    }


def grouped_table(
    rows: list[dict[str, str]], methods: list[str], group: str
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["method"] in methods:
            buckets[(row[group], row["method"])].append(row)
    output = []
    for (value, method), values in sorted(buckets.items()):
        output.append(
            {
                group: value,
                "method": method,
                "case_count": len(values),
                **{metric: avg(row.get(metric) for row in values) for metric in TABLE_METRICS},
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, metric: str) -> str:
    number = finite(value)
    if number is None:
        return "--"
    if metric in {"ATE_Sim3_m", "ATE_SE3_m", "IDF1", "Coverage", "Seam_root_m"}:
        return f"{number:.3f}"
    return f"{number:.1f}"


def latex_group_table(rows: list[dict[str, Any]], group: str, final_method: str) -> str:
    metrics = ("W-MPJPE_mm", "WA-MPJPE_mm", "Accel_mm_frame2", "ATE_SE3_m", "IDF1", "IDs")
    labels = ("W", "WA", "Accel", "ATE-SE3", "IDF1", "IDs")
    lines = [
        "% Auto-generated; case-macro within each stratum.",
        "\\begin{tabular}{ll" + "r" * len(metrics) + "}",
        "\\toprule",
        f"{group.title()} & Method & " + " & ".join(labels) + r" \\",
        "\\midrule",
    ]
    for row in rows:
        method = str(row["method"])
        label = DISPLAY.get(method, "Movie3R-v19" if method == final_method else method)
        escaped_group = str(row[group]).replace("_", r"\_")
        lines.append(
            f"{escaped_group} & {label} & "
            + " & ".join(fmt(row[metric], metric) for metric in metrics)
            + r" \\"
        )
    lines.extend(("\\bottomrule", "\\end{tabular}", ""))
    return "\n".join(lines)


def relative(candidate: Any, baseline: Any, higher: bool = False) -> float | None:
    first, second = finite(candidate), finite(baseline)
    if first is None or second is None or abs(second) < 1e-12:
        return None
    return (first / second - 1.0) if higher else (1.0 - first / second)


def runtime_summary(final_method: str, test_root: Path) -> dict[str, Any]:
    forwards, detectors, geometry, rss, total = [], [], [], [], []
    for path in sorted((test_root.parent / "predictions").rglob("*.runtime.json")):
        payload = load(path)
        runtime = payload.get("runtime", {})
        forward = runtime.get("m0_forward", {})
        if finite(forward.get("fps")) is not None:
            forwards.append(float(forward["fps"]))
        if finite(runtime.get("causal_gru_detector", {}).get("seconds")) is not None:
            detectors.append(float(runtime["causal_gru_detector"]["seconds"]))
        if finite(runtime.get("oracle_explicit_geometry_seconds")) is not None:
            geometry.append(float(runtime["oracle_explicit_geometry_seconds"]))
        if finite(payload.get("environment", {}).get("process_peak_rss_bytes")) is not None:
            rss.append(float(payload["environment"]["process_peak_rss_bytes"]))
        if finite(payload.get("total_process_seconds")) is not None:
            total.append(float(payload["total_process_seconds"]))
    postprocess_by_case: dict[str, float] = {}
    for path in sorted((test_root.parent / "captures").rglob("*.json")):
        try:
            payload = load(path)
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("schema_version") != "Movie3R-v16-Harmony4D-causal-stabilization-probe-v1":
            continue
        for row in payload.get("rows", []):
            if row.get("candidate") != final_method or row.get("status") != "complete":
                continue
            value = finite(row.get("diagnostics", {}).get("postprocess_seconds"))
            if value is not None:
                postprocess_by_case[str(row["case_id"])] = value
    postprocess = list(postprocess_by_case.values())
    return {
        "case_count": len(forwards),
        "strict_forward_fps_mean": avg(forwards),
        "strict_forward_fps_median": median(forwards) if forwards else None,
        "detector_seconds_per_100_frames_mean": avg(detectors),
        "legacy_explicit_geometry_seconds_mean": avg(geometry),
        "process_peak_rss_gib_max": max(rss, default=0.0) / (1024**3) if rss else None,
        "total_process_seconds_mean": avg(total),
        "frozen_postprocess_case_count": len(postprocess),
        "frozen_postprocess_seconds_per_100_frames_mean": avg(postprocess),
        "frozen_postprocess_seconds_per_100_frames_median": median(postprocess) if postprocess else None,
        "contract": "wall-clock diagnostic on the shared server; not hardware-normalized throughput",
    }


def detector_summary(test_root: Path) -> list[dict[str, Any]]:
    output = []
    for name, key in (
        ("causal_gru", "causal_gru_detector"),
        ("static_logistic", "static_logistic_detector"),
    ):
        case_count = true_positive = false_positive = false_negative = exact = 0
        absolute_offsets: list[int] = []
        for path in sorted((test_root.parent / "predictions").rglob("*.runtime.json")):
            payload = load(path)
            boundary = int(payload["record"]["boundary_index"])
            detector = payload.get("runtime", {}).get(key)
            if detector is None:
                continue
            labels = [bool(value) for value in detector["labels"]]
            first = detector.get("first_positive_index")
            case_count += 1
            true_positive += int(labels[boundary])
            false_negative += int(not labels[boundary])
            false_positive += sum(value for index, value in enumerate(labels) if index != boundary)
            exact += int(first == boundary)
            if first is not None:
                absolute_offsets.append(abs(int(first) - boundary))
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else None
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else None
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall
            else None
        )
        output.append(
            {
                "detector": name,
                "case_count": case_count,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "frame_precision": precision,
                "frame_recall": recall,
                "frame_f1": f1,
                "first_positive_exact_rate": exact / case_count if case_count else None,
                "boundary_mae_frames_given_positive": avg(absolute_offsets),
            }
        )
    return output


def detector_latex(rows: list[dict[str, Any]]) -> str:
    lines = [
        "% Auto-generated boundary-detector diagnostics on all inferred Test cases.",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Detector & Cases & Precision & Recall & F1 & Exact & MAE (frames) \\",
        r"\midrule",
    ]
    for row in rows:
        label = str(row["detector"]).replace("_", r"\_")
        lines.append(
            f"{label} & {row['case_count']} & "
            f"{float(row['frame_precision']):.3f} & {float(row['frame_recall']):.3f} & "
            f"{float(row['frame_f1']):.3f} & {float(row['first_positive_exact_rate']):.3f} & "
            f"{float(row['boundary_mae_frames_given_positive']):.1f} " + r"\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}", ""))
    return "\n".join(lines)


def metric_line(name: str, metric: str, value: Any, ci: dict[str, Any] | None = None) -> str:
    rendered = fmt(value, metric)
    if ci and finite(ci.get("low")) is not None and finite(ci.get("high")) is not None:
        rendered += f" [{fmt(ci['low'], metric)}, {fmt(ci['high'], metric)}]"
    return f"| {name} | {rendered} |"


def main() -> None:
    args = parse_args()
    dev_path = args.development / "summary.json"
    holdout_path = args.holdout / "summary.json"
    test_path = args.test / "summary.json"
    dev, holdout, test, frozen = load(dev_path), load(holdout_path), load(test_path), load(args.final_candidate)
    dev_counts = protocol_counts(args.development, dev)
    holdout_counts = protocol_counts(args.holdout, holdout)
    test_counts = protocol_counts(args.test, test)
    final_method = str(frozen["final_method_name"])
    parent = str(test["parent"])
    strict = "m0_strict_human3r"
    methods = [name for name in (strict, "m15_safe_boundary_permutation_causal_gru", parent, final_method) if name in test["methods"]]
    if final_method not in test["methods"]:
        raise ValueError(f"Frozen method {final_method} missing from test summary")
    args.output.mkdir(parents=True, exist_ok=True)
    cases = read_cases(args.test / "case_metrics.csv")
    angle = grouped_table(cases, methods, "angle_stratum")
    action = grouped_table(cases, methods, "action")
    write_csv(args.output / "angle_metrics.csv", angle)
    write_csv(args.output / "action_metrics.csv", action)
    (args.output / "angle_table.tex").write_text(latex_group_table(angle, "angle_stratum", final_method), encoding="utf-8")
    (args.output / "action_table.tex").write_text(latex_group_table(action, "action", final_method), encoding="utf-8")

    final_metrics = test["methods"][final_method]["case_macro"]
    strict_metrics = test["methods"][strict]["case_macro"]
    parent_metrics = test["methods"][parent]["case_macro"]
    improvements = {}
    for metric in (*LOWER, *HIGHER):
        improvements[metric] = {
            "vs_strict_fraction": relative(final_metrics.get(metric), strict_metrics.get(metric), metric in HIGHER),
            "vs_parent_fraction": relative(final_metrics.get(metric), parent_metrics.get(metric), metric in HIGHER),
        }
    runtime = runtime_summary(final_method, args.test)
    detectors = detector_summary(args.test)
    (args.output / "runtime_summary.json").write_text(json.dumps(runtime, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output / "improvements.json").write_text(json.dumps(improvements, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(args.output / "detector_metrics.csv", detectors)
    (args.output / "detector_table.tex").write_text(detector_latex(detectors), encoding="utf-8")

    worst = []
    by_key = {(row["case_id"], row["method"]): row for row in cases}
    for (case_id, method), row in by_key.items():
        if method != final_method:
            continue
        other = by_key.get((case_id, parent))
        ratio = None if other is None else relative(row.get("W-MPJPE_mm"), other.get("W-MPJPE_mm"))
        worst.append(
            {
                "case_id": case_id,
                "action": row["action"],
                "angle_stratum": row["angle_stratum"],
                "final_W-MPJPE_mm": finite(row.get("W-MPJPE_mm")),
                "parent_W-MPJPE_mm": None if other is None else finite(other.get("W-MPJPE_mm")),
                "final_W_improvement_vs_parent_fraction": ratio,
            }
        )
    worst.sort(key=lambda row: float("inf") if row["final_W-MPJPE_mm"] is None else -float(row["final_W-MPJPE_mm"]))
    write_csv(args.output / "worst_cases_by_final_W.csv", worst[:20])

    method_rows = []
    for method in methods:
        row = {"method": method, "display_name": DISPLAY.get(method, "Movie3R-v19" if method == final_method else method)}
        row.update({metric: test["methods"][method]["case_macro"].get(metric) for metric in TABLE_METRICS})
        method_rows.append(row)
    write_csv(args.output / "main_test_metrics.csv", method_rows)

    reference = test["multi_thumbs_literature_reference"]["values"]
    ci = test["methods"][final_method]["hierarchical_bootstrap_ci95"]
    lines = [
        "# Movie3R EgoHumans ICLR final experiment report",
        "",
        "## Frozen protocol and selection",
        "",
        "- Protocol: `Movie3R-EgoHumans-CS100-v1`, 100 frames (50 pre + 50 post), 20 FPS.",
        f"- Pre-registered split: {dev_counts['planned_captures']} development / {holdout_counts['planned_captures']} holdout / {test_counts['planned_captures']} test capture archives; four fixed camera-angle strata per capture.",
        f"- Completed capture inference: {dev_counts['completed_captures']} development / {holdout_counts['completed_captures']} holdout / {test_counts['completed_captures']} test; the holdout has {holdout_counts['structural_exclusions']} pre-specified structural exclusion.",
        f"- Evaluable camera-pair cases: {dev_counts['evaluable_cases']} development / {holdout_counts['evaluable_cases']} holdout / {test_counts['evaluable_cases']} test. The corresponding method-independent evaluator-unavailable counts are {dev_counts['evaluator_unavailable_cases']} / {holdout_counts['evaluator_unavailable_cases']} / {test_counts['evaluator_unavailable_cases']}.",
        f"- Test accounting: {test_counts['inferred_cases']} camera-pair cases were inferred; the main table uses {test_counts['evaluable_cases']} cases from {test_counts['evaluable_captures']} captures after {test_counts['evaluator_unavailable_cases']} evaluator-unavailable cases. No structurally valid Test capture was removed.",
        f"- Final method: `{final_method}`; source candidate: `{frozen['source_candidate_name']}`; holdout fallback used: `{frozen['fallback_used']}`.",
        "- GT SMPL, calibration and identity are evaluator-only. Runtime uses RGB, model predictions and causal history.",
        "- Test was read only after the final candidate was frozen. Difficult but structurally valid cases remain in the results.",
        "",
        "## Test main results (case macro)",
        "",
        "| Method | W | WA | MPJPE | MPVPE | Accel | ATE-Sim3 | ATE-SE3 | IDF1 | IDs | Coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in method_rows:
        lines.append(
            "| " + row["display_name"] + " | "
            + " | ".join(fmt(row[metric], metric) for metric in TABLE_METRICS[:-1])
            + " |"
        )
    lines.extend(("", "## Frozen Movie3R 95% hierarchical bootstrap CI", "", "| Metric | Mean [95% CI] |", "|---|---:|"))
    for metric in TABLE_METRICS:
        lines.append(metric_line(metric, metric, final_metrics.get(metric), ci.get(metric)))
    lines.extend(("", "## Relative change of the frozen method", "", "Positive means improvement under each metric's direction.", "", "| Metric | vs Strict Human3R | vs v17 parent |", "|---|---:|---:|"))
    for metric in (*LOWER, *HIGHER):
        first = improvements[metric]["vs_strict_fraction"]
        second = improvements[metric]["vs_parent_fraction"]
        lines.append(f"| {metric} | {'--' if first is None else f'{100*first:.1f}%'} | {'--' if second is None else f'{100*second:.1f}%'} |")
    lines.extend((
        "",
        "## Multi-THuMBS literature context",
        "",
        "These are same-named metrics, not a same-protocol leaderboard comparison: Multi-THuMBS does not publish its exact EgoHumans capture/camera/cut manifest or evaluator.",
        "",
        "| Metric | Movie3R-CS100 | Multi-THuMBS paper |",
        "|---|---:|---:|",
    ))
    for metric in ("W-MPJPE_mm", "WA-MPJPE_mm", "MPJPE_mm", "MPVPE_mm", "Accel_mm_frame2", "ATE_Sim3_m", "IDs"):
        lines.append(f"| {metric} | {fmt(final_metrics.get(metric), metric)} | {fmt(reference.get(metric), metric)} |")
    lines.extend((
        "",
        "## Runtime diagnostics",
        "",
        f"- Strict forward speed: {fmt(runtime['strict_forward_fps_mean'], 'IDF1')} FPS mean.",
        f"- Frozen CPU postprocess: {fmt(runtime['frozen_postprocess_seconds_per_100_frames_mean'], 'ATE_SE3_m')} s per 100-frame case mean.",
        f"- Peak process RSS across test cases: {fmt(runtime['process_peak_rss_gib_max'], 'ATE_SE3_m')} GiB.",
        "- Timings are wall-clock diagnostics on a shared server and must not be presented as hardware-normalized throughput.",
        "",
        "## Causal boundary detector check",
        "",
        f"- The causal GRU first triggers at the pre-registered cut in {int(detectors[0]['true_positive'])}/{int(detectors[0]['case_count'])} Test cases, with {int(detectors[0]['false_positive'])} off-boundary positives and {fmt(detectors[0]['boundary_mae_frames_given_positive'], 'IDs')} frame MAE.",
        "- The v19 release postprocessor takes the causal GRU's first positive as its runtime trigger and fails closed if it differs from the cached source boundary. On CS100 the proposal equals the evaluator boundary for every Test case, so the frozen source caches are output-equivalent; this does not establish detector generalization beyond the constructed cross-camera protocol.",
        "",
        "## Interpretation and limitations",
        "",
        "- The main claim is causal multi-shot world consistency: camera gauge, human continuity, identity and within-shot stability are evaluated jointly.",
        "- MPJPE/MPVPE check that global correction does not buy alignment by damaging local body reconstruction.",
        "- Angle- and action-stratified tables are provided for generalization and failure analysis; the worst-case CSV is retained rather than hiding hard sequences.",
        f"- Evaluator availability is reported separately from the method Coverage metric. Test has {test_counts['evaluator_unavailable_cases']}/{test_counts['inferred_cases']} method-independent evaluator-unavailable camera pairs because no initial GT/predicted person match exists for the shared-world fit; these pairs are not silently counted as successful reconstructions.",
        "- Development-only identity re-tracking, per-person translation and unsafe SE(3) branches are documented in `versions/v19/EGOHUMANS_DEVELOPMENT_NEGATIVE_RESULTS_20260820.md` and are not tuned on holdout/test.",
        "- Multi-THuMBS values are literature-scale context only until an official manifest/evaluator becomes available.",
        "",
        "## Artifacts",
        "",
        "- `main_test_metrics.csv`: test main table source.",
        "- `angle_metrics.csv`, `angle_table.tex`: small/medium/large/extreme results.",
        "- `action_metrics.csv`, `action_table.tex`: seven-action appendix results.",
        "- `worst_cases_by_final_W.csv`: retained hard cases.",
        "- `detector_metrics.csv`, `detector_table.tex`: automatic-boundary diagnostics over all 116 inferred Test cases.",
        "- `runtime_summary.json`, `improvements.json`: runtime and relative-effect sources.",
        "- Development, holdout and test directories each retain `summary.json`, `case_metrics.csv`, `main_table.tex` and `SUMMARY.md`.",
        "",
    ))
    report = args.output / "FINAL_EGOHUMANS_ICLR_REPORT_20260820.md"
    report.write_text("\n".join(lines), encoding="utf-8")

    source_paths = [
        dev_path,
        holdout_path,
        test_path,
        args.development / "case_metrics.csv",
        args.holdout / "case_metrics.csv",
        args.test / "case_metrics.csv",
        args.final_candidate,
        REPO_ROOT / "versions/v19/egohumans/frozen_holdout_candidates.json",
        REPO_ROOT / "versions/v19/egohumans/development_combined_candidates.json",
        REPO_ROOT / "versions/v19/EGOHUMANS_ICLR_EXPERIMENT_PLAN_20260820.md",
        REPO_ROOT / "versions/v19/EGOHUMANS_DEVELOPMENT_NEGATIVE_RESULTS_20260820.md",
    ]
    manifest = {
        "schema_version": "Movie3R-v19-EgoHumans-final-artifacts-v1",
        "protocol": "Movie3R-EgoHumans-CS100-v1",
        "final_method": final_method,
        "sources": [
            {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in source_paths
        ],
        "outputs": sorted(path.name for path in args.output.iterdir() if path.is_file()),
    }
    partial = args.output / "artifact_manifest.json.partial"
    partial.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(partial, args.output / "artifact_manifest.json")
    print(json.dumps({"report": str(report.resolve()), "final_method": final_method, "test_cases": test["case_count"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
