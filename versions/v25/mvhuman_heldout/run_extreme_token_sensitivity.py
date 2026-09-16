#!/usr/bin/env python3
"""Run inference-time token sensitivity on the frozen MVHuman extreme stratum.

This is a controlled component diagnostic, not a deployable causal result:
all routes use the protocol's fixed transition index so that detector errors do
not confound the comparison.  The checkpoint, inputs, and evaluator are fixed;
only the inference-time correction-token mask changes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
RUN_CASE = HERE / "run_case.py"
EVALUATE_CASE = HERE / "evaluate_case.py"
MODES = ("full", "semantic_only", "alignment_only", "semantic_alignment")
ALLOWED_MODES = MODES + ("no_semantic", "no_alignment", "no_momentum")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=[2, 4, 5, 6, 7])
    parser.add_argument("--python", type=Path, default=REPO_ROOT / ".venv/bin/python")
    parser.add_argument("--modes", nargs="+", choices=ALLOWED_MODES)
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument("--summary-name", default="run_summary.json")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_task(
    task: tuple[str, int, str],
    gpu: int,
    args: argparse.Namespace,
    runtime_manifest: Path,
    evaluator_manifest: Path,
) -> dict:
    mode, line, case_id = task
    root = args.output_root / mode
    cache = root / "predictions" / f"{case_id}.npz"
    report = cache.with_suffix(".runtime.json")
    metric = root / "metrics" / f"{case_id}.json"
    log = root / "logs" / f"{case_id}.log"
    work = root / "work" / f"gpu{gpu}"
    for parent in (cache.parent, metric.parent, log.parent, work):
        parent.mkdir(parents=True, exist_ok=True)
    commands: list[list[str]] = []
    if not (cache.is_file() and report.is_file()):
        commands.append([
            str(args.python), str(RUN_CASE),
            "--runtime-manifest", str(runtime_manifest),
            "--line", str(line),
            "--derived-root", str(args.protocol_root / "derived"),
            "--work-dir", str(work),
            "--output", str(cache),
            "--device", f"cuda:{gpu}",
            "--ablation-token-mode", mode,
            "--controlled-transition-index", "74",
        ])
    if not metric.is_file():
        commands.append([
            str(args.python), str(EVALUATE_CASE),
            "--cache", str(cache),
            "--runtime-report", str(report),
            "--evaluator-manifest", str(evaluator_manifest),
            "--case-id", case_id,
            "--audit-root", str(args.audit_root),
            "--output", str(metric),
        ])
    started = time.time()
    with log.open("a", encoding="utf-8") as handle:
        for command in commands:
            handle.write("COMMAND " + json.dumps(command) + "\n")
            handle.flush()
            completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, text=True)
            if completed.returncode:
                return {
                    "mode": mode, "case_id": case_id, "gpu": gpu,
                    "status": "failed", "returncode": completed.returncode,
                    "seconds": time.time() - started, "log": str(log),
                }
    return {
        "mode": mode, "case_id": case_id, "gpu": gpu,
        "status": "ok", "resumed": not commands,
        "seconds": time.time() - started, "log": str(log),
    }


def worker(
    gpu: int,
    tasks: list[tuple[str, int, str]],
    args: argparse.Namespace,
    runtime_manifest: Path,
    evaluator_manifest: Path,
) -> list[dict]:
    outcomes = []
    for task in tasks:
        result = run_task(task, gpu, args, runtime_manifest, evaluator_manifest)
        outcomes.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    return outcomes


def main() -> None:
    args = parse_args()
    args.protocol_root = args.protocol_root.resolve()
    args.audit_root = args.audit_root.resolve()
    args.output_root = args.output_root.resolve()
    args.python = args.python.expanduser()
    if not args.python.is_absolute():
        args.python = (Path.cwd() / args.python).absolute()
    runtime_manifest = args.protocol_root / "manifests/test_runtime.jsonl"
    evaluator_manifest = args.protocol_root / "manifests/test_evaluator.jsonl"
    runtime_rows = load_rows(runtime_manifest)
    evaluator_rows = {row["case_id"]: row for row in load_rows(evaluator_manifest)}
    selected = [
        (index + 1, row["case_id"])
        for index, row in enumerate(runtime_rows)
        if evaluator_rows[row["case_id"]]["angle_stratum"] == "extreme"
    ]
    if len(selected) != 10:
        raise ValueError(f"Expected 10 frozen extreme cases, found {len(selected)}")
    if args.case_ids:
        requested = set(args.case_ids)
        selected = [(line, case_id) for line, case_id in selected if case_id in requested]
        missing = requested - {case_id for _, case_id in selected}
        if missing:
            raise ValueError(f"Requested cases are not in the extreme stratum: {sorted(missing)}")
    modes = tuple(args.modes) if args.modes else MODES
    tasks = [(mode, line, case_id) for mode in modes for line, case_id in selected]
    shards = {gpu: tasks[offset::len(args.gpus)] for offset, gpu in enumerate(args.gpus)}
    started = time.time()
    outcomes = []
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = {
            executor.submit(worker, gpu, shard, args, runtime_manifest, evaluator_manifest): gpu
            for gpu, shard in shards.items()
        }
        for future in as_completed(futures):
            outcomes.extend(future.result())
    summary = {
        "schema_version": "Shot3R-MVHuman-extreme-token-sensitivity-run-v1",
        "diagnostic_only": True,
        "token_masks_are_inference_time": True,
        "transition_index": 74,
        "transition_source": "fixed evaluator boundary to isolate token sensitivity",
        "angle_stratum": "extreme",
        "modes": list(modes),
        "case_count_per_mode": len(selected),
        "gpus": args.gpus,
        "seconds": time.time() - started,
        "outcomes": sorted(outcomes, key=lambda row: (row["mode"], row["case_id"])),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / args.summary_name
    temporary = summary_path.with_suffix(summary_path.suffix + ".partial")
    temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(summary_path)
    failed = [row for row in outcomes if row["status"] != "ok"]
    print(json.dumps({"tasks": len(tasks), "failed": failed, "seconds": summary["seconds"]}, indent=2), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
