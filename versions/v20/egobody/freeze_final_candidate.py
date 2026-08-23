#!/usr/bin/env python3
"""Select one final source after holdout without reading test metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


PARENT = "v16_0_m15_geometry"
CORE = ("W-MPJPE_mm", "WA-MPJPE_mm", "Accel_mm_frame2", "ATE_SE3_m", "Seam_root_m")
AGGREGATE_SCHEMA = "Bridge3R-EgoBody-CS150-aggregate-v1"
HOLDOUT_SCHEMA = "Bridge3R-EgoBody-frozen-holdout-candidates-v1"
OUTPUT_SCHEMA = "Bridge3R-EgoBody-frozen-final-candidate-v1"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def recording_nonworse(path: Path, candidate: str) -> dict[str, Any]:
    rows = read_csv(path)
    keys = [(row.get("name", ""), row.get("recording", "")) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate name/recording rows in holdout recording metrics")
    by_key = {(row["name"], row["recording"]): row for row in rows}
    recordings = sorted({row["recording"] for row in rows if row["name"] == PARENT})
    ratios = {}
    for recording in recordings:
        if (candidate, recording) not in by_key:
            continue
        values = []
        for metric in CORE:
            base = float(by_key[(PARENT, recording)][metric])
            value = float(by_key[(candidate, recording)][metric])
            if math.isfinite(base) and math.isfinite(value) and base > 0 and value > 0:
                values.append(value / base)
        if values:
            ratios[recording] = math.exp(sum(math.log(value) for value in values) / len(values))
    return {
        "recording_core_error_ratios": ratios,
        "parent_recordings": len(recordings),
        "available_recordings": len(ratios),
        "missing_recordings": sorted(set(recordings) - set(ratios)),
        "nonworse_recordings": sum(value <= 1.0 for value in ratios.values()),
        "nonworse_fraction": sum(value <= 1.0 for value in ratios.values()) / max(len(ratios), 1),
        "worst_recording_core_ratio": max(ratios.values()) if ratios else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-summary", type=Path, required=True)
    parser.add_argument("--holdout-summary", type=Path, required=True)
    parser.add_argument("--holdout-recording-metrics", type=Path, required=True)
    parser.add_argument("--holdout-candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def freeze_json(path: Path, value: dict[str, Any]) -> bool:
    encoded = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise FileExistsError(f"refusing to replace frozen artifact: {path}")
        return True
    partial = path.with_suffix(path.suffix + f".{os.getpid()}.partial")
    partial.write_bytes(encoded)
    try:
        os.link(partial, path)
    except FileExistsError:
        if path.read_bytes() != encoded:
            raise FileExistsError(f"concurrent incompatible freeze: {path}")
        return True
    finally:
        partial.unlink(missing_ok=True)
    return False


def main() -> None:
    args = parse_args()
    development = json.loads(args.development_summary.read_text(encoding="utf-8"))
    holdout = json.loads(args.holdout_summary.read_text(encoding="utf-8"))
    source = json.loads(args.holdout_candidates.read_text(encoding="utf-8"))
    if (
        development.get("schema_version") != AGGREGATE_SCHEMA
        or development.get("split") != "development"
        or development.get("parent") != PARENT
    ):
        raise ValueError("development summary has incompatible schema/split/parent")
    if (
        holdout.get("schema_version") != AGGREGATE_SCHEMA
        or holdout.get("split") != "holdout"
        or holdout.get("parent") != PARENT
    ):
        raise ValueError("holdout summary has incompatible schema/split/parent")
    if (
        source.get("schema_version") != HOLDOUT_SCHEMA
        or source.get("frozen_before_holdout") is not True
        or source.get("holdout_metrics_read") is not False
    ):
        raise ValueError("holdout candidates were not frozen before holdout")
    source_path = args.holdout_candidates.resolve()
    development_path = args.development_summary.resolve()
    recording_path = args.holdout_recording_metrics.resolve()
    if Path(str(source.get("frozen_artifact_path", ""))).resolve() != source_path:
        raise ValueError("holdout candidates moved from their frozen path")
    if (
        source.get("provenance", {}).get("development_summary_sha256")
        != digest(development_path)
    ):
        raise ValueError("holdout candidates are not bound to this development summary")
    if (
        Path(str(holdout.get("candidate_source", ""))).resolve() != source_path
        or holdout.get("candidate_source_sha256") != digest(source_path)
    ):
        raise ValueError("holdout summary is not bound to these frozen candidates")
    if (
        Path(str(holdout.get("recording_metrics", ""))).resolve() != recording_path
        or holdout.get("recording_metrics_sha256") != digest(recording_path)
    ):
        raise ValueError("holdout recording CSV is not bound to its summary")
    candidate_rows = source.get("candidates", [])
    names = [str(row.get("name", "")) for row in candidate_rows]
    if not names or not all(names) or len(names) != len(set(names)):
        raise ValueError("holdout candidate names are missing/duplicate")
    configs = {str(row["name"]): row for row in candidate_rows}
    if PARENT not in configs:
        raise ValueError("holdout candidates lack the frozen parent fallback")
    candidates = [name for name in configs if name != PARENT]
    parent_metrics = holdout["methods"][PARENT]["metrics"]
    decisions: dict[str, Any] = {}
    qualified = []
    for name in candidates:
        comparison = holdout["paired_to_parent"].get(name, {})
        metric = comparison.get("metrics", {})
        safety = holdout["safety"].get(name, {})
        consistency = development["paired_to_parent"].get(name, {}).get("core_geometric_mean_error_ratio")
        stratified = recording_nonworse(args.holdout_recording_metrics, name)
        candidate_metrics = holdout["methods"].get(name, {}).get("metrics", {})
        coverage = candidate_metrics.get("Coverage", {}).get("mean")
        idf1 = candidate_metrics.get("IDF1", {}).get("mean")
        checks = {
            "complete_case_pairing": comparison.get("paired_case_count") == holdout.get("case_count") and all(
                metric.get(value, {}).get("paired_count") == holdout.get("case_count")
                for value in (*CORE, "MPJPE_mm", "MPVPE_mm")
            ),
            "all_core_metrics_defined": all(metric.get(value, {}).get("mean_candidate_to_parent_error_ratio") is not None for value in CORE),
            "development_direction_consistent": consistency is not None and consistency <= 1.0,
            "holdout_core_improvement_ge_2pct": comparison.get("core_geometric_mean_error_ratio") is not None and comparison["core_geometric_mean_error_ratio"] <= 0.98,
            "mpjpe_noninferior_2pct": metric.get("MPJPE_mm", {}).get("mean_candidate_to_parent_error_ratio") is not None and metric["MPJPE_mm"]["mean_candidate_to_parent_error_ratio"] <= 1.02,
            "mpvpe_noninferior_2pct": metric.get("MPVPE_mm", {}).get("mean_candidate_to_parent_error_ratio") is not None and metric["MPVPE_mm"]["mean_candidate_to_parent_error_ratio"] <= 1.02,
            "coverage_drop_le_1pp": coverage is not None and coverage >= parent_metrics["Coverage"]["mean"] - 0.01,
            "idf1_drop_le_0p01": idf1 is not None and idf1 >= parent_metrics["IDF1"]["mean"] - 0.01,
            "all_holdout_recordings_available": stratified["available_recordings"] == holdout.get("recording_count") and stratified["parent_recordings"] == holdout.get("recording_count"),
            "majority_recordings_nonworse": stratified["nonworse_fraction"] >= 0.5,
            "gate_diagnostics_complete": not safety.get("missing_gate_cases"),
            "fallback_arrays_exact": safety.get("fallback_array_exactness_passed") is True,
            "fallback_metrics_exact": safety.get("fallback_metric_exactness_passed") is True,
            "worst_accepted_W_harm_le_20pct": safety.get("worst_accepted_W_ratio") is None or safety["worst_accepted_W_ratio"] <= 1.20,
        }
        passed = all(checks.values())
        decisions[name] = {"passed": passed, "checks": checks, "comparison": comparison, "recording_safety": stratified, "gate_safety": safety}
        if passed:
            qualified.append(name)
    qualified.sort(key=lambda name: (
        holdout["paired_to_parent"][name]["core_geometric_mean_error_ratio"],
        -decisions[name]["recording_safety"]["nonworse_fraction"],
        name,
    ))
    selected = qualified[0] if qualified else PARENT
    selected_config = configs[selected]
    output = {
        "schema_version": OUTPUT_SCHEMA,
        "protocol": "Bridge3R-EgoBody-CS150-v1",
        "frozen_artifact_path": str(args.output.resolve()),
        "frozen_before_test": True,
        "test_metrics_read": False,
        "source_candidate_name": selected,
        "fallback_to_parent": selected == PARENT,
        "qualified_holdout_candidates": qualified,
        "decisions": decisions,
        "selection_rule": {
            "holdout_core_improvement": "recording-macro geometric mean error ratio <= 0.98",
            "development_direction_consistency": True,
            "local_pose_noninferiority": "MPJPE and MPVPE ratios <= 1.02",
            "coverage_identity": "coverage drop <= 0.01 and IDF1 drop <= 0.01",
            "recording_safety": "at least half of holdout recordings nonworse",
            "case_safety": "bit-exact fallback arrays and metrics, and worst accepted W harm <= 20%",
            "tie_break": "lowest holdout core ratio, highest nonworse fraction, lexical name",
        },
        "candidates": [selected_config],
        "provenance": {
            "development_summary": str(args.development_summary.resolve()),
            "development_summary_sha256": digest(args.development_summary.resolve()),
            "holdout_summary": str(args.holdout_summary.resolve()),
            "holdout_summary_sha256": digest(args.holdout_summary.resolve()),
            "holdout_recording_metrics": str(args.holdout_recording_metrics.resolve()),
            "holdout_recording_metrics_sha256": digest(args.holdout_recording_metrics.resolve()),
            "holdout_candidates": str(args.holdout_candidates.resolve()),
            "holdout_candidates_sha256": digest(args.holdout_candidates.resolve()),
        },
    }
    reused = freeze_json(args.output, output)
    print(json.dumps({"output": str(args.output.resolve()), "qualified": qualified, "selected": selected, "reused": reused}, indent=2))


if __name__ == "__main__":
    main()
