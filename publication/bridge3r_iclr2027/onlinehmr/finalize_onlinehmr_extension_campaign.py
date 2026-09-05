#!/usr/bin/env python3
"""Finalize every protocol in a completed OnlineHMR extension campaign."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Any


SCRIPT = Path(__file__).resolve()
MOVIE_PYTHON = SCRIPT.parents[3] / ".venv/bin/python"
AGGREGATOR = SCRIPT.with_name("aggregate_onlinehmr_extensions.py")
ORDER = (
    "harmony4d_multicut",
    "aist_cs150",
    "mvhuman_mvh150",
    "aist_mc150_3",
    "aist_mc150_4",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def mean_and_support(aggregate: dict[str, Any], metric: str) -> tuple[Any, int]:
    record = aggregate.get("overall", {}).get(metric, {})
    return record.get("mean"), int(record.get("support", 0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.campaign_root.resolve()
    inventory = read_json(root / "manifests/inventory.json")

    missing_protocols = [name for name in ORDER if name not in inventory["protocols"]]
    if missing_protocols:
        raise ValueError(f"inventory is missing protocols: {missing_protocols}")

    validated: list[tuple[str, dict[str, Any], Path]] = []
    for name in ORDER:
        protocol = inventory["protocols"][name]
        expected = int(protocol["case_count"])
        run_root = root / "runs" / f"{name}_attempt01"
        state_path = run_root / "protocol_state.json"
        if not state_path.is_file():
            raise FileNotFoundError(state_path)
        state = read_json(state_path)
        if (
            state.get("status") != "complete"
            or int(state.get("fixed_denominator", -1)) != expected
            or int(state.get("completed_cases", -1)) != expected
        ):
            raise RuntimeError(
                f"{name} is not complete on its fixed denominator: "
                f"status={state.get('status')}, completed={state.get('completed_cases')}, "
                f"expected={expected}"
            )

        validated.append((name, protocol, run_root))

    summaries: list[dict[str, Any]] = []
    for name, protocol, run_root in validated:
        expected = int(protocol["case_count"])
        output_dir = root / "formal" / name
        command = [
            str(MOVIE_PYTHON),
            str(AGGREGATOR),
            "--runtime-manifest",
            str(Path(protocol["runtime_manifest"]).resolve()),
            "--evaluator-manifest",
            str(Path(protocol["evaluator_manifest"]).resolve()),
            "--run-root",
            str(run_root),
            "--output-dir",
            str(output_dir),
            "--require-complete",
        ]
        subprocess.run(command, check=True, cwd=SCRIPT.parents[4])
        aggregate_path = output_dir / "onlinehmr_extension_aggregate.json"
        aggregate = read_json(aggregate_path)
        if (
            int(aggregate.get("fixed_manifest_denominator", -1)) != expected
            or int(aggregate.get("reported_case_count", -1)) != expected
            or int(aggregate.get("missing_case_count", -1)) != 0
        ):
            raise RuntimeError(f"invalid fixed-denominator aggregate: {aggregate_path}")

        coverage, coverage_n = mean_and_support(aggregate, "Coverage")
        completion, completion_n = mean_and_support(aggregate, "Completion")
        row: dict[str, Any] = {
            "protocol_key": name,
            "dataset": aggregate["dataset"],
            "protocol": aggregate["protocol"],
            "fixed_denominator": expected,
            "successful_inference_cases": aggregate["successful_inference_cases"],
            "failed_inference_cases": aggregate["failed_inference_cases"],
            "valid_geometry_cases": aggregate["valid_geometry_cases"],
            "coverage": coverage,
            "coverage_support": coverage_n,
            "completion": completion,
            "completion_support": completion_n,
            "aggregate_path": str(aggregate_path),
        }
        for metric in (
            "W-MPJPE_mm",
            "WA-MPJPE_mm",
            "ATE-Sim3_m",
            "IDF1",
            "PA-MPJPE_mm",
            "Anchor-MPJPE_mm",
            "Seam-root_mm",
            "Seam-orientation_deg",
            "Camera-rotation_deg",
            "Camera-translation_m",
            "Boundary-camera-rotation_deg",
            "Boundary-camera-translation_m",
        ):
            value, support = mean_and_support(aggregate, metric)
            if metric in aggregate.get("overall", {}):
                row[metric] = value
                row[f"{metric}_support"] = support
        summaries.append(row)

    summary = {
        "schema_version": "Bridge3R-OnlineHMR-extension-campaign-summary-v1",
        "method": "onlinehmr_official",
        "protocol_count": len(summaries),
        "fixed_manifest_denominator": sum(row["fixed_denominator"] for row in summaries),
        "reported_case_count": sum(row["fixed_denominator"] for row in summaries),
        "all_protocols_complete": True,
        "runtime_gt_access": False,
        "aggregation_contract": {
            "failure_and_empty_output_retained_in_fixed_denominator": True,
            "conditional_geometry_reports_finite_support": True,
            "coverage_and_completion_use_full_denominator": True,
        },
        "protocols": summaries,
    }
    atomic_json(root / "formal/onlinehmr_extension_campaign_summary.json", summary)
    fields = sorted({key for row in summaries for key in row})
    with (root / "formal/onlinehmr_extension_campaign_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps({
        "all_protocols_complete": True,
        "protocol_count": len(summaries),
        "fixed_manifest_denominator": summary["fixed_manifest_denominator"],
        "summary": str(root / "formal/onlinehmr_extension_campaign_summary.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
