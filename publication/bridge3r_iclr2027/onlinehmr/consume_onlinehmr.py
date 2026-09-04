#!/usr/bin/env python3
"""Convert and GT-only evaluate frozen OnlineHMR outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT = Path(__file__).resolve()
WORKSPACE = SCRIPT.parents[4]
EVAL_ROOT = WORKSPACE / "external_baselines/bridge3r_eval"
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))
if str(WORKSPACE / "Movie3R") not in sys.path:
    sys.path.insert(0, str(WORKSPACE / "Movie3R"))

CONVERTER = SCRIPT.with_name("convert_onlinehmr_result.py")
METHOD = "onlinehmr_official"
SCHEMA = "Bridge3R-OnlineHMR-consumption-v1"


def line_tag(lines: list[int]) -> str:
    explicit = "-".join(f"{line:03d}" for line in lines)
    if len(explicit) <= 120:
        return explicit
    digest = hashlib.sha256(",".join(map(str, lines)).encode("ascii")).hexdigest()[:16]
    return f"{len(lines):03d}_sha256_{digest}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def normalize_w_contract(path: Path) -> None:
    """Correct stale prose metadata without changing frozen metric values."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    contract = payload.get("evaluation_contract")
    if not isinstance(contract, dict):
        return
    stale = contract.pop(
        "w_unavailable_when_fewer_than_two_pre_cut_matched_times", None
    )
    if stale is None:
        return
    contract.update({
        "w_fit_uses_only_physical_frames_zero_and_one": True,
        "w_fit_may_use_one_or_two_initial_frames_with_accepted_matches": True,
        "later_detections_never_replace_a_missed_initial_frame": True,
        "w_unavailable_when_no_accepted_match_in_fixed_initial_window": True,
    })
    atomic_json(path, payload)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def failed_result(dataset: str, record: dict[str, Any], gt_root: Path, reason: str) -> dict[str, Any]:
    import evaluate_prompthmr_egobody as base
    if dataset == "egohumans":
        from evaluate_prompthmr_egohumans import _egohumans_gt
        gt, identities = _egohumans_gt(record, gt_root)
    elif dataset == "harmony4d":
        from evaluate_prompthmr_harmony4d import _harmony4d_gt
        gt, identities = _harmony4d_gt(record, gt_root)
    elif dataset == "egobody":
        gt, identities = base.load_gt(record, gt_root)
    else:
        raise ValueError(dataset)
    return base.failed_inference_result(METHOD, gt, identities, reason)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("egobody", "egohumans", "harmony4d"), required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--evaluator-manifest", type=Path)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--lines", required=True)
    parser.add_argument("--adapter-python", type=Path, required=True)
    parser.add_argument("--evaluator-python", type=Path, required=True)
    parser.add_argument("--adapter-device", default="cpu")
    args = parser.parse_args()

    runtime_manifest = args.runtime_manifest.resolve()
    evaluator_manifest = (
        args.evaluator_manifest.resolve() if args.evaluator_manifest else runtime_manifest
    )
    runtime_rows = read_jsonl(runtime_manifest)
    evaluator_rows = read_jsonl(evaluator_manifest)
    if [row["case_id"] for row in runtime_rows] != [row["case_id"] for row in evaluator_rows]:
        raise ValueError("runtime/evaluator manifests differ")
    lines = [int(value) for value in args.lines.split(",") if value.strip()]
    if not lines or len(lines) != len(set(lines)) or any(line < 1 or line > len(runtime_rows) for line in lines):
        raise ValueError("invalid --lines")
    evaluator_script = EVAL_ROOT / f"evaluate_prompthmr_{args.dataset}.py"
    run_root = args.run_root.resolve()
    outputs = []
    for line in lines:
        runtime_row = runtime_rows[line - 1]
        record = evaluator_rows[line - 1]
        case_id = str(runtime_row["case_id"])
        root = run_root / f"line{line:03d}"
        raw_path = root / "onlinehmr.runtime.json"
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        if (
            raw.get("case_id") != case_id
            or raw.get("manifest_sha256") != sha256(runtime_manifest)
            or raw.get("runtime_gt_access") is not False
        ):
            raise ValueError(f"raw runtime provenance mismatch at line {line}")
        prediction = root / "prediction.npz"
        conversion = root / "prediction.json"
        eval_runtime = root / "onlinehmr.eval_runtime.json"
        evaluation = root / "onlinehmr.evaluation.json"
        status = str(raw.get("status"))
        if status == "success":
            if not prediction.is_file() or not conversion.is_file():
                subprocess.run([
                    os.path.abspath(args.adapter_python), str(CONVERTER),
                    "--native-root", str(Path(raw["native_root"]).resolve()),
                    "--camera-trajectory", str(Path(raw["camera_trajectory"]).resolve()),
                    "--manifest", str(runtime_manifest), "--line", str(line),
                    "--output", str(prediction), "--metadata-output", str(conversion),
                    "--method", METHOD, "--device", args.adapter_device,
                ], cwd=WORKSPACE, check=True)
            eval_payload = {
                "schema_version": "Bridge3R-OnlineHMR-evaluator-runtime-v1",
                "case_id": case_id,
                "split": record["split"],
                "record": record,
                "methods": [METHOD],
                "manifest_line": line,
                "runtime_manifest": str(runtime_manifest),
                "runtime_manifest_sha256": sha256(runtime_manifest),
                "evaluator_manifest": str(evaluator_manifest),
                "evaluator_manifest_sha256": sha256(evaluator_manifest),
                "raw_inference_status": status,
                "raw_inference_runtime": str(raw_path),
                "raw_inference_runtime_sha256": sha256(raw_path),
                "prediction_cache": str(prediction),
                "prediction_cache_sha256": sha256(prediction),
                "conversion_metadata": str(conversion),
                "conversion_metadata_sha256": sha256(conversion),
                "runtime_gt_access": False,
                "gt_used_only_after_inference": True,
            }
            if eval_runtime.is_file():
                if json.loads(eval_runtime.read_text(encoding="utf-8")) != eval_payload:
                    raise RuntimeError(f"stale evaluator runtime at line {line}")
            else:
                atomic_json(eval_runtime, eval_payload)
            if not evaluation.is_file():
                subprocess.run([
                    os.path.abspath(args.evaluator_python), str(evaluator_script),
                    "--cache", str(prediction), "--runtime-report", str(eval_runtime),
                    "--gt-root", str(args.gt_root.resolve()), "--output", str(evaluation),
                ], cwd=WORKSPACE, check=True)
        elif status == "failed":
            value = failed_result(
                args.dataset, record, args.gt_root.resolve(),
                str(raw.get("failure_reason") or "OnlineHMR inference failure"),
            )
            atomic_json(evaluation, {
                "schema_version": f"Bridge3R-OnlineHMR-{args.dataset}-evaluation-v1",
                "protocol": record["protocol"],
                "case_id": case_id,
                "record_runtime_fields": record,
                "methods": {METHOD: value},
                "inputs": {
                    "prediction_cache": None,
                    "runtime_report": str(raw_path),
                    "runtime_report_sha256": sha256(raw_path),
                },
                "evaluation_contract": {
                    "runtime_gt_access": False,
                    "gt_used_only_in_evaluator": True,
                    "whole_case_failure_counts_as_zero_coverage": True,
                    "test_tuning": False,
                },
            })
        else:
            raise ValueError(f"unknown raw status {status!r}")
        normalize_w_contract(evaluation)
        payload = json.loads(evaluation.read_text(encoding="utf-8"))
        value = payload["methods"][METHOD]
        outputs.append({
            "line": line, "case_id": case_id, "raw_status": status,
            "evaluation": str(evaluation), "evaluation_sha256": sha256(evaluation),
            "coverage": value["coverage"]["coverage"],
            "precision": value["coverage"]["precision"],
            "idf1": value["identity"]["idf1"],
            "w_available": value["world_alignment"]["w_available"],
        })
        print(
            f"[OnlineHMR consume {args.dataset} line {line:03d}] "
            f"{status} coverage={outputs[-1]['coverage']:.6f}", flush=True,
        )
    summary = {
        "schema_version": SCHEMA,
        "dataset": args.dataset,
        "method": METHOD,
        "selected_lines": lines,
        "attempted_cases": len(outputs),
        "successful_inference_cases": sum(row["raw_status"] == "success" for row in outputs),
        "failed_inference_cases": sum(row["raw_status"] == "failed" for row in outputs),
        "runtime_manifest": str(runtime_manifest),
        "runtime_manifest_sha256": sha256(runtime_manifest),
        "evaluator_manifest": str(evaluator_manifest),
        "evaluator_manifest_sha256": sha256(evaluator_manifest),
        "runtime_gt_access": False,
        "cases": outputs,
    }
    name = "consumption.lines_" + line_tag(lines) + ".json"
    atomic_json(run_root / name, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
