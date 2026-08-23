#!/usr/bin/env python3
"""Freeze at most two v20 candidates after development and before holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


PARENT = "v16_0_m15_geometry"
TRANSFER = ("v17_harmony_multicue_safe", "v19_ungated_translation_b050")
AGGREGATE_SCHEMA = "Bridge3R-EgoBody-CS150-aggregate-v1"
DEVELOPMENT_SCHEMA = "Bridge3R-EgoBody-development-candidates-v1"
OUTPUT_SCHEMA = "Bridge3R-EgoBody-frozen-holdout-candidates-v1"
CORE = ("W-MPJPE_mm", "WA-MPJPE_mm", "Accel_mm_frame2", "ATE_SE3_m", "Seam_root_m")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-summary", type=Path, required=True)
    parser.add_argument("--development-candidates", type=Path, required=True)
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
    summary = json.loads(args.development_summary.read_text(encoding="utf-8"))
    source = json.loads(args.development_candidates.read_text(encoding="utf-8"))
    if (
        summary.get("schema_version") != AGGREGATE_SCHEMA
        or summary.get("split") != "development"
        or summary.get("parent") != PARENT
    ):
        raise ValueError("development summary has incompatible schema/split/parent")
    if (
        source.get("schema_version") != DEVELOPMENT_SCHEMA
        or source.get("frozen_before_development_inference") is not True
    ):
        raise ValueError("development candidate source was not frozen before inference")
    source_path = args.development_candidates.resolve()
    if (
        Path(str(summary.get("candidate_source", ""))).resolve() != source_path
        or summary.get("candidate_source_sha256") != digest(source_path)
    ):
        raise ValueError("development summary is not bound to this candidate source")
    candidate_rows = source.get("candidates", [])
    names = [str(row.get("name", "")) for row in candidate_rows]
    if not names or not all(names) or len(names) != len(set(names)):
        raise ValueError("development candidate names are missing/duplicate")
    configs = {str(row["name"]): row for row in candidate_rows}
    methods = summary["methods"]
    paired = summary["paired_to_parent"]
    safety = summary["safety"]
    if PARENT not in configs or any(name not in configs for name in TRANSFER):
        raise ValueError("development candidate file lacks required transfer controls")
    decisions: dict[str, Any] = {}
    qualified = []
    parent_idf1 = methods[PARENT]["metrics"]["IDF1"]["mean"]
    parent_coverage = methods[PARENT]["metrics"]["Coverage"]["mean"]
    for name in sorted(value for value in configs if value.startswith("v20_")):
        comparison = paired.get(name, {})
        metric = comparison.get("metrics", {})
        gate = safety.get(name, {})
        candidate_idf1 = methods.get(name, {}).get("metrics", {}).get("IDF1", {}).get("mean")
        candidate_coverage = methods.get(name, {}).get("metrics", {}).get("Coverage", {}).get("mean")
        checks = {
            "complete_case_pairing": comparison.get("paired_case_count") == summary.get("case_count") and all(
                metric.get(value, {}).get("paired_count") == summary.get("case_count")
                for value in (*CORE, "MPJPE_mm", "MPVPE_mm")
            ),
            "all_core_metrics_defined": all(
                metric.get(value, {}).get("mean_candidate_to_parent_error_ratio")
                is not None
                for value in CORE
            ),
            "core_improvement_ge_2pct": comparison.get("core_geometric_mean_error_ratio") is not None and comparison["core_geometric_mean_error_ratio"] <= 0.98,
            "mpjpe_noninferior_2pct": metric.get("MPJPE_mm", {}).get("mean_candidate_to_parent_error_ratio") is not None and metric["MPJPE_mm"]["mean_candidate_to_parent_error_ratio"] <= 1.02,
            "mpvpe_noninferior_2pct": metric.get("MPVPE_mm", {}).get("mean_candidate_to_parent_error_ratio") is not None and metric["MPVPE_mm"]["mean_candidate_to_parent_error_ratio"] <= 1.02,
            "coverage_drop_le_1pp": candidate_coverage is not None and parent_coverage is not None and candidate_coverage >= parent_coverage - 0.01,
            "idf1_drop_le_0p01": candidate_idf1 is not None and parent_idf1 is not None and candidate_idf1 >= parent_idf1 - 0.01,
            "prediction_only_gate_enabled": gate.get("gate_enabled") is True,
            "gate_diagnostics_complete": not gate.get("missing_gate_cases"),
            "exact_fallback_arrays": gate.get("fallback_array_exactness_passed") is True,
            "exact_fallback_metrics": gate.get("fallback_metric_exactness_passed") is True,
            "worst_accepted_W_harm_le_20pct": gate.get("worst_accepted_W_ratio") is None or gate["worst_accepted_W_ratio"] <= 1.20,
        }
        passed = all(checks.values())
        decisions[name] = {"passed": passed, "checks": checks, "comparison": comparison, "safety": gate}
        if passed:
            qualified.append(name)
    qualified.sort(key=lambda name: (
        paired[name]["core_geometric_mean_error_ratio"],
        safety[name].get("worst_accepted_W_ratio") or 0.0,
        name,
    ))
    selected_new = qualified[:2]
    selected_names = [PARENT, *TRANSFER, *selected_new]
    output = {
        "schema_version": OUTPUT_SCHEMA,
        "protocol": "Bridge3R-EgoBody-CS150-v1",
        "frozen_artifact_path": str(args.output.resolve()),
        "frozen_before_holdout": True,
        "holdout_metrics_read": False,
        "selection_rule": {
            "maximum_new_candidates": 2,
            "core_improvement": "development recording-macro geometric mean error ratio <= 0.98",
            "local_pose_noninferiority": "MPJPE and MPVPE ratios <= 1.02",
            "coverage_identity": "coverage drop <= 0.01 and IDF1 drop <= 0.01",
            "safety": "prediction-only gate, bit-exact fallback arrays and metrics, worst accepted W harm <= 20%",
            "required_transfer_controls": list(TRANSFER),
        },
        "qualified_new_candidates": qualified,
        "selected_new_candidates": selected_new,
        "decisions": decisions,
        "candidates": [configs[name] for name in selected_names],
        "provenance": {
            "development_summary": str(args.development_summary.resolve()),
            "development_summary_sha256": digest(args.development_summary.resolve()),
            "development_candidates": str(args.development_candidates.resolve()),
            "development_candidates_sha256": digest(args.development_candidates.resolve()),
        },
    }
    reused = freeze_json(args.output, output)
    print(json.dumps({"output": str(args.output.resolve()), "qualified": qualified, "selected": selected_names, "reused": reused}, indent=2))


if __name__ == "__main__":
    main()
