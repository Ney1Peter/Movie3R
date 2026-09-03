#!/usr/bin/env python3
"""Audit the pre-specified 12-case OnlineHMR availability pilot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


METHOD = "onlinehmr_official"
SCHEMA = "Bridge3R-OnlineHMR-pilot-audit-v1"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--egobody-attempt", default="attempt01")
    parser.add_argument("--egohumans-attempt", default="attempt01")
    parser.add_argument("--harmony4d-attempt", default="attempt02")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pilot = json.loads(args.pilot.read_text(encoding="utf-8"))
    manifests = {
        "egobody": args.manifest_root / "egobody_cs150_test.runtime.jsonl",
        "egohumans": args.manifest_root / "egohumans_formal90.runtime.jsonl",
        "harmony4d": args.manifest_root / "harmony4d_formal88.runtime.jsonl",
    }
    attempts = {
        "egobody": [value for value in args.egobody_attempt.split(",") if value],
        "egohumans": [value for value in args.egohumans_attempt.split(",") if value],
        "harmony4d": [value for value in args.harmony4d_attempt.split(",") if value],
    }
    lookup = {}
    records = {}
    for dataset, path in manifests.items():
        rows = read_jsonl(path)
        records[dataset] = rows
        lookup[dataset] = {str(row["case_id"]): line for line, row in enumerate(rows, 1)}

    audited = []
    for selected in pilot["cases"]:
        dataset, case_id = str(selected["dataset"]), str(selected["case_id"])
        line = lookup[dataset][case_id]
        record = records[dataset][line - 1]
        root = None
        chosen_attempt = None
        for attempt in attempts[dataset]:
            candidate = args.runs_root / dataset / attempt / f"line{line:03d}"
            if (
                (candidate / "onlinehmr.runtime.json").is_file()
                and (candidate / "onlinehmr.evaluation.json").is_file()
            ):
                root, chosen_attempt = candidate, attempt
                break
        if root is None:
            chosen_attempt = attempts[dataset][0]
            root = args.runs_root / dataset / chosen_attempt / f"line{line:03d}"
        raw_path = root / "onlinehmr.runtime.json"
        cache_path = root / "prediction.npz"
        conversion_path = root / "prediction.json"
        evaluation_path = root / "onlinehmr.evaluation.json"
        item: dict[str, Any] = {
            **selected, "manifest_line": line, "attempt": chosen_attempt,
            "attempted": raw_path.is_file() and evaluation_path.is_file(),
            "prediction_artifacts_available": cache_path.is_file() and conversion_path.is_file(),
        }
        if item["attempted"]:
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            value = evaluation["methods"][METHOD]
            item.update({
                "raw_status": raw.get("status"),
                "camera_trajectory_rows": raw.get("camera_trajectory_rows"),
                "expected_camera_trajectory_rows": int(record["clip_length"]) - 1,
                "native_track_count": raw.get("native_track_count"),
                "coverage": value["coverage"]["coverage"],
            })
            if item["prediction_artifacts_available"]:
                conversion = json.loads(conversion_path.read_text(encoding="utf-8"))
                with np.load(cache_path, allow_pickle=False) as cache:
                    valid = np.asarray(cache[f"{METHOD}__valid"], dtype=bool)
                boundary = int(record["boundary_index"])
                item.update({
                    "pre_cut_predicted_person_frames": int(valid[:boundary].sum()),
                    "post_cut_predicted_person_frames": int(valid[boundary:].sum()),
                    "coordinate_roundtrip_residual_m": conversion["summary"].get(
                        "camera_world_roundtrip_max_residual_m"
                    ),
                })
            residual = item.get("coordinate_roundtrip_residual_m")
            item["passes_case_gate"] = bool(
                item["raw_status"] == "success"
                and item["prediction_artifacts_available"]
                and item["camera_trajectory_rows"] == item["expected_camera_trajectory_rows"]
                and item["pre_cut_predicted_person_frames"] > 0
                and item["post_cut_predicted_person_frames"] > 0
                and float(item["coverage"]) > 0.0
                and residual is not None
                and float(residual) <= 2e-4
            )
        else:
            item["passes_case_gate"] = False
        audited.append(item)

    available = sum(item["attempted"] for item in audited)
    passed = sum(item["passes_case_gate"] for item in audited)
    status = "waiting_for_all_12"
    if available == 12:
        status = "pass" if passed >= 10 else "fail"
    payload = {
        "schema_version": SCHEMA,
        "status": status,
        "expected_cases": 12,
        "available_cases": available,
        "passed_case_gates": passed,
        "minimum_required_passed_cases": 10,
        "gate_changes_configuration": False,
        "cases": audited,
    }
    atomic_json(args.output.resolve(), payload)
    lines = [
        "# OnlineHMR 12-case pilot audit", "",
        f"Status: **{status}** ({passed}/12 case gates passed; {available}/12 available).", "",
        "| Dataset | Stratum | Case | Available | Pass | Coverage | Pre/Post predicted person-frames |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for item in audited:
        lines.append(
            f"| {item['dataset']} | {item['angle_stratum']} | `{item['case_id']}` | "
            f"{int(item['attempted'])} | {int(item['passes_case_gate'])} | "
            f"{item.get('coverage', '—')} | "
            f"{item.get('pre_cut_predicted_person_frames', '—')}/"
            f"{item.get('post_cut_predicted_person_frames', '—')} |"
        )
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": status, "available_cases": available, "passed_case_gates": passed,
    }, indent=2))


if __name__ == "__main__":
    main()
