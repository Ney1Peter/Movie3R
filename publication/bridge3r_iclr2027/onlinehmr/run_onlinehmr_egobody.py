#!/usr/bin/env python3
"""Prepare, run, and evaluate OnlineHMR on the frozen EgoBody Test cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any


SCRIPT = Path(__file__).resolve()
WORKSPACE = SCRIPT.parents[4]
MOVIE_PYTHON = WORKSPACE / "Movie3R/.venv/bin/python"
ONLINE_PYTHON = WORKSPACE / "external_baselines/.venvs/onlinehmr-py311-pt25-cu118/bin/python"
ASSET_STAGER = SCRIPT.with_name("stage_egobody_assets.py")
IMAGE_STAGER = WORKSPACE / "Movie3R/versions/v20/egobody/stage_images.py"
MANIFEST_BUILDER = WORKSPACE / "Movie3R/versions/v20/egobody/build_manifest.py"
GT_BUILDER = WORKSPACE / "Movie3R/versions/v20/egobody/prepare_gt.py"
CASE_STAGER = WORKSPACE / "external_baselines/bridge3r_eval/stage_egobody_case.py"
RUNNER = SCRIPT.with_name("run_onlinehmr_lines.py")
CONSUMER = SCRIPT.with_name("consume_onlinehmr.py")
SCHEMA = "Bridge3R-OnlineHMR-EgoBody-run-v1"
EXPECTED_RUNTIME_SHA256 = "8a5861bd3e4ee55dd1639c86526d21c96a73bb44fe07ff9848ef7b6b7645b02b"
EXPECTED_EVALUATOR_SHA256 = "87144f01a8dc7b0630b1e7e9613a8b11904847a97c438e9ac9693b3453ac534f"


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


def run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, cwd=WORKSPACE, capture_output=True, text=True)
    log.write_text(
        "COMMAND " + json.dumps(command) + "\n" + completed.stdout
        + "\nSTDERR\n" + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}); see {log}")


def lines_from_text(value: str, total: int) -> list[int]:
    if value.strip().lower() == "all":
        return list(range(1, total + 1))
    output = [int(item) for item in value.split(",") if item.strip()]
    if not output or len(output) != len(set(output)) or any(line < 1 or line > total for line in output):
        raise ValueError("invalid --lines")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--attempt", default="attempt01")
    parser.add_argument("--lines", default="all")
    parser.add_argument("--gpus", default="0,1,2,3,4")
    parser.add_argument("--reserve-gib", type=float, default=50.0)
    parser.add_argument("--gt-device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    outer = args.outer.resolve()
    runtime = args.runtime_manifest.resolve()
    work = args.work_root.resolve()
    output = args.output_root.resolve()
    if not re.fullmatch(r"attempt[0-9]{2}", args.attempt):
        raise ValueError("--attempt must match attemptNN")
    if not outer.is_file() or not runtime.is_file():
        raise FileNotFoundError(outer if not outer.is_file() else runtime)
    rows = [
        json.loads(line) for line in runtime.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 129 or sha256(runtime) != EXPECTED_RUNTIME_SHA256:
        raise ValueError("EgoBody runtime manifest is not the frozen 129-case Test protocol")
    lines = lines_from_text(args.lines, len(rows))
    line_text = ",".join(str(line) for line in lines)
    state_path = output / "egobody" / f"protocol_summary.{args.attempt}.json"
    state: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "preflight",
        "dataset": "egobody",
        "attempt": args.attempt,
        "selected_lines": lines,
        "runtime_manifest": str(runtime),
        "runtime_manifest_sha256": sha256(runtime),
        "outer": str(outer),
        "outer_size_bytes": outer.stat().st_size,
        "runtime_gt_access": False,
    }
    atomic_json(state_path, state)

    assets = work / "outer"
    asset_command = [
        str(MOVIE_PYTHON), str(ASSET_STAGER), "--outer", str(outer),
        "--output-root", str(assets), "--official-split", "test",
        "--reserve-gib", str(args.reserve_gib),
    ]
    if args.dry_run:
        asset_command.append("--dry-run")
        run(asset_command, work / "logs/preflight.assets.log")
        state["status"] = "dry_run_complete"
        atomic_json(state_path, state)
        print(json.dumps(state, indent=2, ensure_ascii=False))
        return

    state["status"] = "materializing_evaluator_assets"
    atomic_json(state_path, state)
    run(asset_command, work / "logs/assets.log")

    manifests = work / "rebuilt_manifests"
    state["status"] = "rebuilding_evaluator_manifest"
    atomic_json(state_path, state)
    run([
        str(MOVIE_PYTHON), str(MANIFEST_BUILDER),
        "--data-info", str(assets / "data_info_release.csv"),
        "--data-splits", str(assets / "data_splits.csv"),
        "--calibrations-root", str(assets / "expanded/calibrations"),
        "--kinect-params-root", str(assets / "expanded/kinect_cam_params"),
        "--output-dir", str(manifests), "--split", "test",
    ], work / "logs/build_manifest.log")
    rebuilt_runtime = manifests / "egobody_cs150_test.runtime.jsonl"
    evaluator = manifests / "egobody_cs150_test.evaluator.jsonl"
    if sha256(rebuilt_runtime) != EXPECTED_RUNTIME_SHA256 or rebuilt_runtime.read_bytes() != runtime.read_bytes():
        raise ValueError("rebuilt EgoBody runtime manifest differs from frozen manifest")
    if sha256(evaluator) != EXPECTED_EVALUATOR_SHA256:
        raise ValueError("rebuilt EgoBody evaluator manifest differs from frozen evaluator")
    state.update({
        "evaluator_manifest": str(evaluator),
        "evaluator_manifest_sha256": sha256(evaluator),
    })

    explicit_lines = "-".join(f"{line:03d}" for line in lines)
    if len(explicit_lines) > 120:
        digest = hashlib.sha256(line_text.encode("ascii")).hexdigest()[:16]
        explicit_lines = f"{len(lines):03d}_sha256_{digest}"
    selection = work / "selections" / f"{args.attempt}.lines_{explicit_lines}.jsonl"
    selection.parent.mkdir(parents=True, exist_ok=True)
    selected_payload = "".join(
        json.dumps(rows[line - 1], sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
        for line in lines
    ).encode("utf-8")
    if selection.is_file() and selection.read_bytes() != selected_payload:
        raise RuntimeError(f"existing stage-only selection differs: {selection}")
    if not selection.exists():
        selection.write_bytes(selected_payload)

    staged = work / "staging"
    stage_metadata = work / "stage_metadata" / args.attempt
    state["status"] = "streaming_selected_rgb"
    atomic_json(state_path, state)
    run([
        str(MOVIE_PYTHON), str(IMAGE_STAGER), "--outer", str(outer),
        "--manifest", str(selection), "--output-root", str(staged),
        "--provenance-output", str(stage_metadata / "stage_images.provenance.json"),
        "--staged-manifest-output", str(stage_metadata / "stage_images.runtime.jsonl"),
        "--expected-images-per-case", "150", "--reserve-gib", str(args.reserve_gib),
    ], stage_metadata / "stage_images.log")

    gt_root = work / "gt_cache"
    state["status"] = "building_evaluator_only_gt"
    atomic_json(state_path, state)
    gt_command = [
        str(MOVIE_PYTHON), str(GT_BUILDER), "--manifest", str(evaluator),
        "--outer-root", str(assets), "--output-root", str(gt_root),
        "--model-root", str(WORKSPACE / "Movie3R/src/models"),
        "--device", args.gt_device, "--batch-size", "16", "--fail-fast",
    ]
    for line in lines:
        gt_command.extend(("--line", str(line)))
    run(gt_command, work / "logs" / f"prepare_gt.{args.attempt}.log")

    input_root = work / "runtime_inputs/egobody"
    state["status"] = "materializing_prediction_only_inputs"
    atomic_json(state_path, state)
    for line in lines:
        run([
            str(MOVIE_PYTHON), str(CASE_STAGER), "--manifest", str(runtime),
            "--line", str(line), "--staged-root", str(staged),
            "--output-root", str(input_root),
        ], work / "logs/inputs" / f"line{line:03d}.log")

    run_root = output / "egobody" / args.attempt
    state["status"] = "inference"
    atomic_json(state_path, state)
    run([
        str(MOVIE_PYTHON), str(RUNNER), "--manifest", str(runtime),
        "--lines", line_text, "--input-root", str(input_root),
        "--output-root", str(run_root), "--gpus", args.gpus,
        "--python", str(MOVIE_PYTHON), "--allow-failures",
    ], output / "egobody/logs" / f"{args.attempt}.inference.log")

    state["status"] = "evaluation"
    atomic_json(state_path, state)
    run([
        str(MOVIE_PYTHON), str(CONSUMER), "--dataset", "egobody",
        "--runtime-manifest", str(runtime), "--evaluator-manifest", str(evaluator),
        "--run-root", str(run_root), "--gt-root", str(gt_root),
        "--lines", line_text, "--adapter-python", str(ONLINE_PYTHON),
        "--evaluator-python", str(MOVIE_PYTHON), "--adapter-device", "cpu",
    ], output / "egobody/logs" / f"{args.attempt}.evaluation.log")

    missing = [
        str(run_root / f"line{line:03d}/onlinehmr.evaluation.json")
        for line in lines
        if not (run_root / f"line{line:03d}/onlinehmr.evaluation.json").is_file()
    ]
    if missing:
        raise FileNotFoundError(missing[0])
    state.update({"status": "complete", "completed_lines": sorted(lines)})
    atomic_json(state_path, state)
    print(json.dumps(state, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
