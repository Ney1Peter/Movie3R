#!/usr/bin/env python3
"""Run and evaluate OnlineHMR one verified archive group at a time."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any


SCRIPT = Path(__file__).resolve()
WORKSPACE = SCRIPT.parents[4]
MOVIE_PYTHON = WORKSPACE / "Movie3R/.venv/bin/python"
ONLINE_PYTHON = (
    WORKSPACE / "external_baselines/.venvs/onlinehmr-py311-pt25-cu118/bin/python"
)
CASE_STAGER = WORKSPACE / "external_baselines/bridge3r_eval/stage_trace_crossdataset_case.py"
H4D_STAGER = WORKSPACE / "external_baselines/bridge3r_eval/stage_trace_harmony4d_group.py"
EGOHUMAN_STAGER = WORKSPACE / "Movie3R/versions/v19/egohumans/stage_capture.py"
RUNNER = SCRIPT.with_name("run_onlinehmr_lines.py")
CONSUMER = SCRIPT.with_name("consume_onlinehmr.py")
SCHEMA = "Bridge3R-OnlineHMR-diskbounded-v1"


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


def run(command: list[str], log: Path, *, allow_failure: bool = False) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, cwd=WORKSPACE, capture_output=True, text=True)
    log.write_text(
        "COMMAND " + json.dumps(command) + "\n" + completed.stdout
        + "\nSTDERR\n" + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode and not allow_failure:
        raise RuntimeError(f"command failed ({completed.returncode}); see {log}")
    return int(completed.returncode)


def token(entry: str) -> str:
    path = PurePosixPath(entry)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe archive entry: {entry}")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(path).replace("/", "__"))


def remove_inside(path: Path, parent: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    resolved, root = path.resolve(), parent.resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"unsafe cleanup target {resolved}")
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def parse_lines(value: str | None, total: int) -> list[int]:
    if value is None:
        return list(range(1, total + 1))
    lines = [int(item) for item in value.split(",") if item.strip()]
    if not lines or len(lines) != len(set(lines)) or any(item < 1 or item > total for item in lines):
        raise ValueError("invalid --lines")
    return lines


def egohuman_paths(work: Path, entry: str) -> tuple[Path, Path, Path]:
    name = Path(entry).name
    stem = name[:-7] if name.endswith(".tar.gz") else name
    capture = re.sub(r"-\d{3}$", "", stem)
    slug = re.sub(
        r"[^A-Za-z0-9_.-]+", "_",
        (entry[:-7] if entry.endswith(".tar.gz") else entry).replace("/", "__"),
    )
    archive = work / "archives" / f"{slug}.tar.gz"
    stage = work / "staging" / slug
    return archive, stage, stage / capture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("egohumans", "harmony4d"), required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--evaluator-manifest", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--attempt", default="attempt01")
    parser.add_argument("--lines")
    parser.add_argument("--gpus", default="0,1,2,3,4")
    parser.add_argument("--reserve-gib", type=float, default=50.0)
    parser.add_argument("--keep-staging", action="store_true")
    args = parser.parse_args()

    runtime_manifest = args.runtime_manifest.resolve()
    evaluator_manifest = args.evaluator_manifest.resolve()
    rows = [
        json.loads(line)
        for line in runtime_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    evaluator_rows = [
        json.loads(line)
        for line in evaluator_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if [row["case_id"] for row in rows] != [row["case_id"] for row in evaluator_rows]:
        raise ValueError("runtime/evaluator manifests differ")
    lines = parse_lines(args.lines, len(rows))
    groups: dict[str, list[int]] = {}
    for line in lines:
        groups.setdefault(str(rows[line - 1]["archive_entry"]), []).append(line)

    work = args.work_root.resolve()
    output = args.output_root.resolve()
    input_root = work / "runtime_inputs" / args.dataset
    if not re.fullmatch(r"attempt[0-9]{2}", args.attempt):
        raise ValueError("--attempt must match attemptNN")
    run_root = output / args.dataset / args.attempt
    state_path = output / args.dataset / "protocol_summary.json"
    state: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "running",
        "dataset": args.dataset,
        "archive": str(args.archive.resolve()),
        "runtime_manifest": str(runtime_manifest),
        "runtime_manifest_sha256": sha256(runtime_manifest),
        "evaluator_manifest": str(evaluator_manifest),
        "evaluator_manifest_sha256": sha256(evaluator_manifest),
        "selected_lines": lines,
        "attempt": args.attempt,
        "gpus": args.gpus,
        "groups": {},
        "runtime_gt_access": False,
    }
    if state_path.is_file():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        if (
            previous.get("schema_version") == SCHEMA
            and previous.get("runtime_manifest_sha256") == state["runtime_manifest_sha256"]
            and previous.get("evaluator_manifest_sha256") == state["evaluator_manifest_sha256"]
        ):
            state["groups"] = previous.get("groups", {})
    atomic_json(state_path, state)

    for group_index, (entry, group_lines) in enumerate(groups.items(), 1):
        group_token = token(entry)
        metadata = output / args.dataset / "stage_metadata" / group_token
        line_text = ",".join(str(line) for line in group_lines)
        group_state: dict[str, Any] = {
            "entry": entry, "lines": group_lines, "status": "staging",
        }
        state["groups"][entry] = group_state
        atomic_json(state_path, state)
        print(f"[{group_index}/{len(groups)}] staging {entry}: {line_text}", flush=True)

        if args.dataset == "egohumans":
            inner_archive, stage_root, _ = egohuman_paths(work, entry)
            run([
                str(MOVIE_PYTHON), str(EGOHUMAN_STAGER),
                "--outer", str(args.archive.resolve()), "--entry", entry,
                "--work-root", str(work),
                "--audit-output", str(metadata / "audit.json"),
                "--ledger-output", str(metadata / "stage_ledger.json"),
                "--reserve-gib", str(args.reserve_gib),
            ], metadata / "stage.log")
        else:
            inner_archive = work / "nested" / entry
            h4d_token = "_".join(PurePosixPath(entry).with_suffix("").parts)
            stage_root = work / "staging" / h4d_token
            run([
                str(MOVIE_PYTHON), str(H4D_STAGER),
                "--outer", str(args.archive.resolve()), "--entry", entry,
                "--runtime-manifest", str(runtime_manifest),
                "--work-root", str(work), "--ledger", str(metadata / "stage_ledger.json"),
                "--reserve-gib", str(args.reserve_gib), "--lines", line_text,
            ], metadata / "stage.log")

        group_state["status"] = "input_materialization"
        atomic_json(state_path, state)
        for line in group_lines:
            run([
                str(MOVIE_PYTHON), str(CASE_STAGER),
                "--manifest", str(runtime_manifest), "--line", str(line),
                "--extracted-root", str(stage_root), "--output-root", str(input_root),
            ], metadata / f"line{line:03d}.input.log")

        group_state["status"] = "inference"
        atomic_json(state_path, state)
        run([
            str(MOVIE_PYTHON), str(RUNNER),
            "--manifest", str(runtime_manifest), "--lines", line_text,
            "--input-root", str(input_root), "--output-root", str(run_root),
            "--gpus", args.gpus, "--python", str(MOVIE_PYTHON),
            "--allow-failures",
        ], output / args.dataset / "logs" / f"{group_token}.inference.log")

        group_state["status"] = "evaluation"
        atomic_json(state_path, state)
        run([
            str(MOVIE_PYTHON), str(CONSUMER), "--dataset", args.dataset,
            "--runtime-manifest", str(runtime_manifest),
            "--evaluator-manifest", str(evaluator_manifest),
            "--run-root", str(run_root), "--gt-root", str(stage_root),
            "--lines", line_text, "--adapter-python", str(ONLINE_PYTHON),
            "--evaluator-python", str(MOVIE_PYTHON), "--adapter-device", "cpu",
        ], output / args.dataset / "logs" / f"{group_token}.evaluation.log")

        evaluation_paths = [run_root / f"line{line:03d}/onlinehmr.evaluation.json" for line in group_lines]
        missing = [str(path) for path in evaluation_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(missing[0])
        group_state.update({
            "status": "complete",
            "evaluations": [
                {"path": str(path), "sha256": sha256(path)} for path in evaluation_paths
            ],
        })
        if not args.keep_staging:
            for line in group_lines:
                remove_inside(input_root / str(rows[line - 1]["case_id"]), input_root)
            remove_inside(stage_root, work / "staging")
            remove_inside(inner_archive, work / ("archives" if args.dataset == "egohumans" else "nested"))
            group_state["reproducible_staging_removed"] = True
        atomic_json(state_path, state)

    completed = sorted(
        line
        for group in state["groups"].values()
        if group.get("status") == "complete"
        for line in group["lines"]
        if line in lines
    )
    state["completed_lines"] = completed
    state["status"] = "complete" if completed == sorted(lines) else "partial"
    atomic_json(state_path, state)
    print(json.dumps(state, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
