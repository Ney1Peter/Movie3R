#!/usr/bin/env python3
"""Aggregate MVH150 reports with capture-macro and angle-stratified scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "Bridge3R-MVHuman-Heldout-MVH150-aggregate-v1"
REPORT_SCHEMA = "Bridge3R-MVHuman-Heldout-MVH150-evaluation-v1"
RUNTIME_SCHEMA = "Bridge3R-MVHuman-Heldout-MVH150-runtime-cache-v1"
EXPECTED_CASES = 50
EXPECTED_FRAMES = 150
EXPECTED_METHODS = (
    "m0_strict_human3r",
    "m1_clean_reset",
    "m3_coarse_gauge",
    "m4_coarse_gauge_identity",
    "m6_fine_alignment",
    "m14_gated_parent",
    "m15_bridge3r_full",
)
METRICS = (
    "pa_mpjpe_body12_mm",
    "first_shot_anchor_mpjpe_body12_mm",
    "first_shot_anchor_root_error_mm",
    "first_shot_anchor_orientation_proxy_deg",
    "seam_root_excess_mm",
    "seam_orientation_excess_deg",
    "post_camera_relative_rotation_deg",
    "post_camera_relative_translation_m",
)
DISPLAY = {
    "m0_strict_human3r": "Strict Human3R",
    "m1_clean_reset": "Clean reset",
    "m3_coarse_gauge": "Learned coarse gauge",
    "m4_coarse_gauge_identity": "Coarse gauge + association",
    "m6_fine_alignment": "Fine alignment transaction",
    "m14_gated_parent": "Gated parent",
    "m15_bridge3r_full": "BRIDGE3R",
    "prompthmr_official": "PromptHMR (offline)",
    "gvhmr_official": "GVHMR (offline)",
    "trace_official": "TRACE",
    "spec_official": "SPEC",
}

# Several case reports bind the same immutable model files.  Re-hashing a
# multi-gigabyte checkpoint once per case adds no audit strength, so cache the
# observed digest for the lifetime of this fail-closed aggregation process.
_VERIFIED_FILE_HASHES: dict[Path, str] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--evaluator-manifest", type=Path, required=True)
    parser.add_argument("--texture-statistics", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tex-output", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260830)
    return parser.parse_args()


@lru_cache(maxsize=None)
def sha256(path: Path) -> str:
    path = path.resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_sha256(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("content_sha256", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def require_file_hash(path: Path, expected: str, label: str) -> str:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    observed = _VERIFIED_FILE_HASHES.get(path)
    if observed is None:
        observed = sha256(path)
        _VERIFIED_FILE_HASHES[path] = observed
    if observed != expected:
        raise ValueError(f"{label} hash mismatch: expected {expected}, observed {observed}")
    return observed


def verify_video(path: Path) -> dict[str, Any]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Derived video is not decodable: {path}")
    reported_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    decoded_frames = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        decoded_frames += 1
    capture.release()
    if reported_frames != EXPECTED_FRAMES or decoded_frames != EXPECTED_FRAMES or abs(fps - 30.0) > 1e-3:
        raise ValueError(
            f"Derived-video contract drifted for {path}: "
            f"reported={reported_frames}, decoded={decoded_frames}, fps={fps}"
        )
    return {"sha256": sha256(path), "reported_frames": reported_frames, "decoded_frames": decoded_frames, "fps": fps}


def infer_audit_root(cases: dict[str, dict[str, Any]]) -> Path:
    archives = [Path(str(row["archive"])).resolve() for row in cases.values()]
    common = Path(os.path.commonpath([str(path) for path in archives]))
    for candidate in (common, *common.parents):
        if (candidate / "metadata").is_dir():
            return candidate
    raise FileNotFoundError("Could not infer the MVHuman audit root containing metadata/")


def verify_runtime_and_artifacts(
    reports: dict[str, dict[str, Any]],
    cases: dict[str, dict[str, Any]],
    evaluator_manifest: Path,
) -> dict[str, Any]:
    protocol_root = evaluator_manifest.resolve().parent.parent
    runtime_manifest = protocol_root / "manifests" / "test_runtime.jsonl"
    protocol_freeze = protocol_root / "protocol_freeze.json"
    materialization_ledger = protocol_root / "materialization_ledger.json"
    for path in (runtime_manifest, protocol_freeze, materialization_ledger):
        if not path.is_file():
            raise FileNotFoundError(path)
    ledger = json.loads(materialization_ledger.read_text(encoding="utf-8"))
    expected_ledger_hashes = {
        runtime_manifest: str(ledger.get("runtime_manifest_sha256")),
        evaluator_manifest.resolve(): str(ledger.get("evaluator_manifest_sha256")),
        protocol_freeze: str(ledger.get("protocol_freeze_sha256")),
    }
    ledger_hashes = {
        str(path): require_file_hash(path, expected, path.name)
        for path, expected in expected_ledger_hashes.items()
    }
    if int(ledger.get("videos", -1)) != EXPECTED_CASES:
        raise ValueError("Materialization ledger does not bind exactly 50 videos")

    runtime_rows = [
        json.loads(line) for line in runtime_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    runtime_by_case = {str(row.get("case_id")): row for row in runtime_rows}
    if len(runtime_rows) != EXPECTED_CASES or set(runtime_by_case) != set(cases):
        raise ValueError("Runtime manifest is not a one-to-one match for the 50 evaluator cases")

    video_hashes, video_contracts, report_hashes, cache_hashes = {}, {}, {}, {}
    checkpoint_hashes: dict[str, str] = {}
    for case_id in sorted(cases):
        runtime_row = runtime_by_case[case_id]
        if runtime_row.get("protocol") != "MVH150" or runtime_row.get("role") != "test":
            raise ValueError(f"Runtime manifest contract drifted for {case_id}")
        relative = Path(str(runtime_row.get("input_video", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe derived-video path for {case_id}: {relative}")
        video = (protocol_root / "derived" / relative).resolve()
        if (protocol_root / "derived").resolve() not in video.parents:
            raise ValueError(f"Derived-video path escapes protocol root for {case_id}")
        video_contracts[case_id] = verify_video(video)
        video_hashes[case_id] = video_contracts[case_id]["sha256"]

        inputs = reports[case_id].get("inputs", {})
        runtime_report = Path(str(inputs.get("runtime_report", ""))).resolve()
        require_file_hash(runtime_report, str(inputs.get("runtime_report_sha256")), f"runtime report {case_id}")
        runtime = json.loads(runtime_report.read_text(encoding="utf-8"))
        if runtime.get("schema_version") != RUNTIME_SCHEMA:
            raise ValueError(f"Runtime report schema drifted for {case_id}")
        record = runtime.get("record", {})
        if record != runtime_row:
            raise ValueError(f"Runtime report/manifest mismatch for {case_id}")
        if tuple(runtime.get("methods", [])) != EXPECTED_METHODS:
            raise ValueError(f"Runtime method contract drifted for {case_id}")
        report_hashes[case_id] = sha256(runtime_report)

        cache = Path(str(inputs.get("cache", ""))).resolve()
        cache_hashes[case_id] = require_file_hash(
            cache, str(inputs.get("cache_sha256")), f"prediction cache {case_id}"
        )
        if Path(str(runtime.get("cache", ""))).resolve() != cache:
            raise ValueError(f"Runtime-report cache path mismatch for {case_id}")
        if runtime.get("cache_sha256") != cache_hashes[case_id]:
            raise ValueError(f"Runtime-report cache hash mismatch for {case_id}")
        for name, path_key, hash_key in (
            ("bridge3r", "current", "current_sha256"),
            ("human3r", "original", "original_sha256"),
            ("detector", "detector", "detector_sha256"),
        ):
            checkpoint = runtime.get("checkpoint", {})
            observed = require_file_hash(
                Path(str(checkpoint.get(path_key, ""))),
                str(checkpoint.get(hash_key, "")),
                f"{name} checkpoint",
            )
            previous = checkpoint_hashes.setdefault(name, observed)
            if previous != observed:
                raise ValueError(f"Inconsistent {name} checkpoint across cases")

    protocol_payload = json.loads(protocol_freeze.read_text(encoding="utf-8"))
    archive_rows = {str(row["subject"]): row for row in protocol_payload.get("subjects", [])}
    case_subjects = {str(row["subject"]) for row in cases.values()}
    if set(archive_rows) != case_subjects:
        raise ValueError("Protocol-freeze archive subjects do not match evaluator subjects")
    archive_hashes = {
        subject: require_file_hash(
            Path(str(archive_rows[subject]["archive"])),
            str(archive_rows[subject]["archive_sha256"]),
            f"source archive {subject}",
        )
        for subject in sorted(case_subjects)
    }

    audit_root = infer_audit_root(cases)
    gt_hashes: dict[str, dict[str, str]] = {}
    for subject in sorted({str(row["subject"]) for row in cases.values()}):
        subject_rows = [row for row in cases.values() if str(row["subject"]) == subject]
        subject_root = audit_root / "metadata" / subject
        paths = [subject_root / "camera_scale.pkl", subject_root / "camera_extrinsics.json"]
        indices = sorted({int(index) for row in subject_rows for index in row["source_frame_indices"]})
        paths.extend(subject_root / "smplx" / "keypoints3d" / f"{index:06d}.json" for index in indices)
        gt_hashes[subject] = {str(path.relative_to(audit_root)): sha256(path) for path in paths}

    gt_aggregate_hashes = {
        subject: hashlib.sha256(
            json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        for subject, values in gt_hashes.items()
    }

    code_paths = [
        Path(__file__).resolve(),
        Path(__file__).with_name("build_protocol.py"),
        Path(__file__).with_name("run_protocol.py"),
        Path(__file__).with_name("run_case.py"),
        Path(__file__).with_name("evaluate_case.py"),
        Path(__file__).resolve().parents[2] / "v21" / "aist_singleperson" / "evaluate_aist.py",
    ]
    return {
        "protocol_root": str(protocol_root),
        "materialization_ledger": str(materialization_ledger),
        "materialization_ledger_sha256": sha256(materialization_ledger),
        "ledger_bound_hashes": ledger_hashes,
        "derived_video_sha256": video_hashes,
        "derived_video_contract": video_contracts,
        "runtime_report_sha256": report_hashes,
        "prediction_cache_sha256": cache_hashes,
        "checkpoint_sha256": checkpoint_hashes,
        "source_archive_sha256": archive_hashes,
        "code_sha256": {str(path): sha256(path) for path in code_paths},
        "gt_metadata_root": str(audit_root),
        "gt_metadata_sha256": gt_hashes,
        "gt_metadata_aggregate_sha256": gt_aggregate_hashes,
    }


def validate_report(path: Path, value: dict[str, Any], case: dict[str, Any], evaluator_sha: str) -> None:
    case_id = str(case["case_id"])
    if value.get("schema_version") != REPORT_SCHEMA or value.get("protocol") != "MVH150":
        raise ValueError(f"Evaluation schema/protocol drifted in {path}")
    if str(value.get("case_id")) != case_id:
        raise ValueError(f"Evaluation case ID mismatch in {path}")
    if str(value.get("subject")) != str(case["subject"]) or value.get("angle_stratum") != case["angle_stratum"]:
        raise ValueError(f"Evaluation metadata mismatch for {case_id}")
    if value.get("errors") != {}:
        raise ValueError(f"Evaluator errors are not empty for {case_id}: {value.get('errors')}")
    if set(value.get("methods", {})) != set(EXPECTED_METHODS):
        raise ValueError(f"Expected exactly seven internal methods for {case_id}")
    if not value.get("detector", {}).get("available"):
        raise ValueError(f"Detector output unavailable for {case_id}")
    if value.get("inputs", {}).get("evaluator_manifest_sha256") != evaluator_sha:
        raise ValueError(f"Evaluator-manifest binding mismatch for {case_id}")
    if value.get("content_sha256") != content_sha256(value):
        raise ValueError(f"Evaluation content hash mismatch for {case_id}")
    for method in EXPECTED_METHODS:
        row = value["methods"][method]
        if row.get("status") != "ok" or row.get("method") != method:
            raise ValueError(f"Method {method} is not successful for {case_id}")
        coverage = row.get("coverage", {}).get("valid_frame_coverage")
        if coverage is None or not np.isfinite(float(coverage)) or not 0.0 <= float(coverage) <= 1.0:
            raise ValueError(f"Invalid coverage for {case_id}/{method}")
        if set(row.get("metrics", {})) != set(METRICS):
            raise ValueError(f"Metric contract drifted for {case_id}/{method}")
        for metric in METRICS:
            item = row["metrics"][metric]
            if int(item.get("count", 0)) <= 0 or item.get("mean") is None or not np.isfinite(float(item["mean"])):
                raise ValueError(f"No finite support for {case_id}/{method}/{metric}")


def metric_value(row: dict[str, Any], name: str) -> float | None:
    value = row.get("metrics", {}).get(name, {}).get("mean")
    return None if value is None else float(value)


def summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return {
        "count": int(len(array)),
        "mean": None if not len(array) else float(array.mean()),
        "median": None if not len(array) else float(np.median(array)),
        "std": None if not len(array) else float(array.std()),
    }


def subject_macro(method_rows: dict[str, dict[str, Any]], cases: dict[str, dict[str, Any]], metric: str, stratum: str | None = None) -> dict[str, Any]:
    by_subject: dict[str, list[float]] = defaultdict(list)
    for case_id, row in method_rows.items():
        case = cases[case_id]
        if stratum is not None and case["angle_stratum"] != stratum:
            continue
        value = metric_value(row, metric)
        if value is not None:
            by_subject[str(case["subject"])].append(value)
    values = [float(np.mean(items)) for items in by_subject.values() if items]
    result = summary(values)
    result["capture_count"] = len(by_subject)
    result["case_support"] = sum(len(items) for items in by_subject.values())
    return result


def paired_bootstrap(
    rows: dict[str, dict[str, dict[str, Any]]],
    cases: dict[str, dict[str, Any]],
    metric: str,
    stratum: str | None,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    baseline, method = rows.get("m0_strict_human3r", {}), rows.get("m15_bridge3r_full", {})
    by_subject: dict[str, list[float]] = defaultdict(list)
    for case_id in sorted(set(baseline).intersection(method)):
        case = cases[case_id]
        if stratum is not None and case["angle_stratum"] != stratum:
            continue
        before, after = metric_value(baseline[case_id], metric), metric_value(method[case_id], metric)
        if before is not None and after is not None:
            by_subject[str(case["subject"])].append(before - after)
    subjects = sorted(by_subject)
    values = np.asarray([np.mean(by_subject[subject]) for subject in subjects], dtype=np.float64)
    if not len(values):
        return {"capture_count": 0, "mean_absolute_reduction": None, "ci95": None}
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(replicates, len(values)))].mean(axis=1)
    baseline_subject = []
    method_subject = []
    for subject in subjects:
        case_ids = [case_id for case_id in baseline if str(cases[case_id]["subject"]) == subject and (stratum is None or cases[case_id]["angle_stratum"] == stratum) and case_id in method]
        before = [metric_value(baseline[case_id], metric) for case_id in case_ids]
        after = [metric_value(method[case_id], metric) for case_id in case_ids]
        before = [value for value in before if value is not None]
        after = [value for value in after if value is not None]
        if before and after:
            baseline_subject.append(float(np.mean(before))); method_subject.append(float(np.mean(after)))
    baseline_mean = float(np.mean(baseline_subject)) if baseline_subject else None
    method_mean = float(np.mean(method_subject)) if method_subject else None
    relative = None if baseline_mean in (None, 0.0) or method_mean is None else 100.0 * (baseline_mean - method_mean) / baseline_mean
    return {
        "capture_count": len(values),
        "mean_absolute_reduction": float(values.mean()),
        "relative_reduction_percent": relative,
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "definition": "Strict Human3R minus BRIDGE3R; positive favours BRIDGE3R; capture-cluster bootstrap",
    }


def tex(value: float | None, digits: int = 1) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def make_tex(payload: dict[str, Any]) -> str:
    methods = ["m0_strict_human3r", "m15_bridge3r_full"]
    rows = []
    for method in methods:
        result = payload["methods"][method]
        overall = result["overall"]
        coverage = result["coverage"]["capture_macro_mean"] * 100.0
        rows.append(
            f"{DISPLAY[method]} & {tex(overall['pa_mpjpe_body12_mm']['mean'])} & "
            f"{tex(overall['first_shot_anchor_mpjpe_body12_mm']['mean'])} & "
            f"{tex(overall['seam_root_excess_mm']['mean'])} & "
            f"{tex(overall['post_camera_relative_rotation_deg']['mean'])} & "
            f"{tex(overall['post_camera_relative_translation_m']['mean'], 3)} & {coverage:.1f} \\\\"
        )
    return "\n".join(
        [
            "% Auto-generated from frozen MVHuman MVH150 case reports; do not hand-edit.",
            "\\begin{table}[t]", "\\centering", "\\small",
            "\\caption{Held-out MVHuman captures in a controlled weak-texture studio. Human errors are in mm, relative camera rotation in degrees, and translation in m. All entries are capture-macro means over the same 50 frozen cases.}",
            "\\label{tab:mvhuman-heldout}",
            "\\resizebox{\\columnwidth}{!}{%", "\\begin{tabular}{lrrrrrr}", "\\toprule",
            "Method & PA & Anchor & Seam-root & Cam. rot. & Cam. trans. & Cov. \\\\", "\\midrule",
            *rows, "\\bottomrule", "\\end{tabular}}", "\\end{table}", "",
        ]
    )


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 100:
        raise ValueError("At least 100 bootstrap replicates are required")
    manifest_rows = [json.loads(line) for line in args.evaluator_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    cases = {str(row["case_id"]): row for row in manifest_rows}
    if len(manifest_rows) != EXPECTED_CASES or len(cases) != EXPECTED_CASES:
        raise ValueError(f"Evaluator manifest must contain exactly {EXPECTED_CASES} unique cases")
    evaluator_sha = sha256(args.evaluator_manifest.resolve())
    reports = {}
    for path in sorted(args.metrics_dir.glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        case_id = str(value.get("case_id"))
        if case_id in reports:
            raise ValueError(f"Duplicate report for {case_id}")
        if case_id not in cases:
            raise ValueError(f"Report outside frozen manifest: {case_id}")
        validate_report(path, value, cases[case_id], evaluator_sha)
        reports[case_id] = value
    if set(reports) != set(cases):
        missing = sorted(set(cases).difference(reports))
        extra = sorted(set(reports).difference(cases))
        raise ValueError(f"Formal aggregation requires 50/50 paired reports; missing={missing}, extra={extra}")
    provenance = verify_runtime_and_artifacts(reports, cases, args.evaluator_manifest)
    method_rows: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for case_id, report in reports.items():
        for method, row in report.get("methods", {}).items():
            method_rows[str(method)][case_id] = row
    strata = ["small", "medium", "large", "very_large", "extreme"]
    methods = {}
    if set(method_rows) != set(EXPECTED_METHODS):
        raise ValueError("Aggregate method set does not match the frozen seven-method contract")
    for method, rows in sorted(method_rows.items()):
        if set(rows) != set(cases):
            raise ValueError(f"Method {method} is not paired on all 50 cases")
        overall = {metric: subject_macro(rows, cases, metric) for metric in METRICS}
        by_angle = {stratum: {metric: subject_macro(rows, cases, metric, stratum) for metric in METRICS} for stratum in strata}
        coverage_subject: dict[str, list[float]] = defaultdict(list)
        for case_id, row in rows.items():
            coverage_subject[str(cases[case_id]["subject"])].append(float(row.get("coverage", {}).get("valid_frame_coverage", 0.0)))
        coverage_values = [float(np.mean(values)) for values in coverage_subject.values()]
        methods[method] = {
            "display": DISPLAY.get(method, method),
            "reported_cases": len(rows),
            "formal_denominator": len(cases),
            "overall": overall,
            "by_angle": by_angle,
            "coverage": {"capture_count": len(coverage_values), "capture_macro_mean": float(np.mean(coverage_values)) if coverage_values else 0.0},
        }
    paired = {
        "overall": {metric: paired_bootstrap(method_rows, cases, metric, None, args.bootstrap_seed, args.bootstrap_replicates) for metric in METRICS},
        "by_angle": {
            stratum: {metric: paired_bootstrap(method_rows, cases, metric, stratum, args.bootstrap_seed + index + 1, args.bootstrap_replicates) for metric in METRICS}
            for index, stratum in enumerate(strata)
        },
    }
    detector_rows = [report["detector"] for report in reports.values()]
    detector = {
        "reported_cases": len(detector_rows), "formal_denominator": len(cases),
        "tp": sum(int(row.get("tp", 0)) for row in detector_rows if row.get("available")),
        "fp": sum(int(row.get("fp", 0)) for row in detector_rows if row.get("available")),
        "fn": sum(int(row.get("fn", 0)) for row in detector_rows if row.get("available")),
    }
    detector["precision"] = detector["tp"] / max(detector["tp"] + detector["fp"], 1)
    detector["recall"] = detector["tp"] / max(detector["tp"] + detector["fn"], 1)
    detector["f1"] = 2 * detector["tp"] / max(2 * detector["tp"] + detector["fp"] + detector["fn"], 1)
    texture = None
    if args.texture_statistics and args.texture_statistics.is_file():
        texture = json.loads(args.texture_statistics.read_text(encoding="utf-8"))
    payload = {
        "schema_version": SCHEMA,
        "formal_case_count": len(cases), "formal_capture_count": len({row["subject"] for row in cases.values()}),
        "metric_report_count": len(reports), "missing_reports": sorted(set(cases).difference(reports)),
        "evaluator_manifest": str(args.evaluator_manifest.resolve()), "evaluator_manifest_sha256": evaluator_sha,
        "methods": methods, "paired_bridge3r_vs_strict": paired, "detector": detector, "texture_statistics": texture,
        "provenance": provenance,
        "aggregation": "Within-case frame means, then equal weighting within capture and across ten held-out captures. Failures remain visible through the frozen denominator and coverage.",
    }
    payload["content_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    if args.tex_output:
        args.tex_output.parent.mkdir(parents=True, exist_ok=True)
        args.tex_output.write_text(make_tex(payload), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "cases": len(reports), "methods": list(methods), "missing": payload["missing_reports"]}, indent=2))


if __name__ == "__main__":
    main()
