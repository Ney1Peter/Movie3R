#!/usr/bin/env python3
"""Run a frozen EgoBody CS150 manifest with strict resumability/provenance.

The driver never extracts RGB and never generates GT.  It consumes an already
staged RGB root, an already prepared GT-cache root, and the paired immutable
runtime/evaluator manifests.  Only runtime rows are passed to GPU inference;
the evaluator manifest is opened by this CPU orchestration process solely to
verify the case join and the frozen ``runtime_row_sha256`` contract.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from versions.v20.egobody.build_manifest import (  # noqa: E402
    EVALUATOR_SCHEMA,
    RUNTIME_SCHEMA,
)
from versions.v20.egobody.dataset import canonical_json, file_sha256, value_sha256  # noqa: E402


DRIVER = Path(__file__).resolve()
RUNNER = REPO_ROOT / "versions/v20/egobody/run_egobody_case.py"
EVALUATOR = REPO_ROOT / "versions/v20/egobody/evaluate_egobody.py"
PROBE = REPO_ROOT / "versions/v20/egobody/probe_candidates.py"
RUNTIME_CACHE_SCHEMA = "Bridge3R-EgoBody-CS150-runtime-cache-v1"
EVALUATION_SCHEMA = "Bridge3R-EgoBody-CS150-evaluation-v1"
PROBE_SCHEMA = "Movie3R-v16-Harmony4D-causal-stabilization-probe-v1"
STATE_SCHEMA = "Bridge3R-EgoBody-CS150-protocol-state-v1"
DEVELOPMENT_CANDIDATE_SCHEMA = "Bridge3R-EgoBody-development-candidates-v1"
HOLDOUT_CANDIDATE_SCHEMA = "Bridge3R-EgoBody-frozen-holdout-candidates-v1"
FINAL_CANDIDATE_SCHEMA = "Bridge3R-EgoBody-frozen-final-candidate-v1"
TEST_LEDGER_SCHEMA = "Bridge3R-EgoBody-test-consumption-v1"
ARRAY_KEYS = (
    "cameras_c2w",
    "vertices_world",
    "joints_world",
    "persistent_ids",
    "native_ids",
    "valid",
)
CORE_METHODS = (
    "m0_strict_human3r",
    "m0r_original_clean_reset",
    "m1_current_clean_reset",
    "m3_b0_only",
    "m15_v17_gated_parent",
)
CANDIDATE_BASELINE = "v16_0_m15_geometry"
KNOWN_EVALUATOR_UNAVAILABLE = (
    "ValueError: No initial matched people for shared world fit",
    "ValueError: Fewer than two valid pre-cut time points for shared W fit",
    "ValueError: No matched people for shared CS150 fit",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--evaluator-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "holdout", "test"), required=True)
    parser.add_argument("--staged-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--devices", required=True, help="Comma-separated free CUDA devices.")
    parser.add_argument("--candidate-json", type=Path, action="append", default=[])
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--current-checkpoint", type=Path)
    parser.add_argument("--original-checkpoint", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate/freeze inputs and write a plan without launching subprocesses.",
    )
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    output = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank JSONL row {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Non-object JSONL row {path}:{line_number}")
            output.append(value)
    if not output:
        raise ValueError(f"No rows in {path}")
    return output


def validate_manifest_pair(
    runtime_path: Path,
    evaluator_path: Path,
    split: str,
) -> list[tuple[int, dict[str, Any], dict[str, Any]]]:
    """Return original runtime line numbers and byte-contract joined rows."""

    runtime_rows = read_jsonl(runtime_path)
    evaluator_rows = read_jsonl(evaluator_path)
    runtime_by_case: dict[str, tuple[int, dict[str, Any]]] = {}
    evaluator_by_case: dict[str, dict[str, Any]] = {}
    for line, row in enumerate(runtime_rows, start=1):
        case_id = str(row.get("case_id", ""))
        if not case_id or case_id in runtime_by_case:
            raise ValueError(f"Missing/duplicate runtime case ID: {case_id!r}")
        if row.get("schema_version") != RUNTIME_SCHEMA:
            raise ValueError(f"Unexpected runtime schema in {case_id}")
        if row.get("split") != split:
            raise ValueError(f"Runtime split mismatch in {case_id}: {row.get('split')}")
        if any(str(key).endswith("_evaluator_only") for key in row):
            raise ValueError(f"Evaluator-only field leaked into runtime row {case_id}")
        members = row.get("image_members")
        frames = list(row.get("pre_frame_numbers", [])) + list(row.get("post_frame_numbers", []))
        if not isinstance(members, list) or len(members) != len(frames):
            raise ValueError(f"RGB member/frame count mismatch in {case_id}")
        if int(row.get("boundary_index", -1)) != len(row.get("pre_frame_numbers", [])):
            raise ValueError(f"Boundary/pre count mismatch in {case_id}")
        if not row["pre_frame_numbers"] or not row["post_frame_numbers"]:
            raise ValueError(f"Empty shot in {case_id}")
        if int(row["post_frame_numbers"][0]) != int(row["pre_frame_numbers"][-1]) + 1:
            raise ValueError(f"Non-causal source-time step in {case_id}")
        runtime_by_case[case_id] = (line, row)
    for row in evaluator_rows:
        case_id = str(row.get("case_id", ""))
        if not case_id or case_id in evaluator_by_case:
            raise ValueError(f"Missing/duplicate evaluator case ID: {case_id!r}")
        if row.get("schema_version") != EVALUATOR_SCHEMA:
            raise ValueError(f"Unexpected evaluator schema in {case_id}")
        if row.get("split") != split:
            raise ValueError(f"Evaluator split mismatch in {case_id}: {row.get('split')}")
        evaluator_by_case[case_id] = row
    if set(runtime_by_case) != set(evaluator_by_case):
        missing_eval = sorted(set(runtime_by_case) - set(evaluator_by_case))
        missing_runtime = sorted(set(evaluator_by_case) - set(runtime_by_case))
        raise ValueError(
            f"Runtime/evaluator cases differ; missing evaluator={missing_eval}, "
            f"missing runtime={missing_runtime}"
        )
    output = []
    for case_id, (line, runtime) in runtime_by_case.items():
        evaluator = evaluator_by_case[case_id]
        observed = str(evaluator.get("runtime_row_sha256", ""))
        expected = value_sha256(runtime)
        if observed != expected:
            raise ValueError(
                f"runtime_row_sha256 mismatch in {case_id}: {observed} vs {expected}"
            )
        if evaluator.get("protocol") != runtime.get("protocol"):
            raise ValueError(f"Protocol mismatch in joined case {case_id}")
        output.append((line, runtime, evaluator))
    output.sort(key=lambda value: value[0])
    return output


def candidate_metadata(paths: Iterable[Path], split: str) -> list[dict[str, Any]]:
    resolved = [path.resolve() for path in paths]
    if len(resolved) != len(set(resolved)):
        raise ValueError("Duplicate candidate JSON paths")
    if split == "test" and len(resolved) != 1:
        raise ValueError("Test requires exactly one frozen candidate JSON")
    output = []
    for path in resolved:
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = read_json(path)
        schema = str(payload.get("schema_version", ""))
        expected_schema = {
            "development": DEVELOPMENT_CANDIDATE_SCHEMA,
            "holdout": HOLDOUT_CANDIDATE_SCHEMA,
            "test": FINAL_CANDIDATE_SCHEMA,
        }[split]
        if schema != expected_schema:
            raise ValueError(
                f"{split} requires candidate schema {expected_schema}, got {schema!r}"
            )
        is_development = (
            path.name == "development_candidates.json"
            or "development-candidates" in schema.lower()
            or bool(payload.get("frozen_before_development_inference"))
        )
        if split in {"holdout", "test"} and is_development:
            raise ValueError(
                f"{split} may not consume development_candidates: {path}"
            )
        if split == "development" and payload.get(
            "frozen_before_development_inference"
        ) is not True:
            raise ValueError(f"Development candidates were not frozen: {path}")
        if split == "holdout":
            if payload.get("frozen_before_holdout") is not True:
                raise ValueError(f"Holdout candidates were not frozen: {path}")
            if payload.get("holdout_metrics_read") is not False:
                raise ValueError(
                    f"Holdout candidate does not certify unread Holdout metrics: {path}"
                )
            if Path(str(payload.get("frozen_artifact_path", ""))).resolve() != path:
                raise ValueError("Holdout candidates must run from their frozen path")
        if split == "test":
            if payload.get("frozen_before_test") is not True:
                raise ValueError(f"Test candidate was not frozen before Test: {path}")
            if payload.get("test_metrics_read") is not False:
                raise ValueError(f"Test candidate does not certify unread Test metrics: {path}")
        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"No candidates in {path}")
        names = [str(row.get("name", "")) for row in candidates if isinstance(row, dict)]
        if len(names) != len(candidates) or not all(names) or len(names) != len(set(names)):
            raise ValueError(f"Invalid/duplicate candidate names in {path}")
        if split == "test":
            if len(candidates) != 1:
                raise ValueError("Test final-candidate JSON must contain exactly one candidate")
            if str(payload.get("source_candidate_name", "")) != names[0]:
                raise ValueError("Test source_candidate_name does not match its candidate")
            if Path(str(payload.get("frozen_artifact_path", ""))).resolve() != path:
                raise ValueError("Test candidate must run from its original frozen path")
        output.append(
            {
                "path": str(path),
                "sha256": file_sha256(path),
                "schema_version": schema,
                "candidate_names": names,
                "frozen_before_test": payload.get("frozen_before_test"),
                "test_metrics_read": payload.get("test_metrics_read"),
                "source_candidate_name": payload.get("source_candidate_name"),
            }
        )
    return output


def reserve_test_consumption(
    candidate: dict[str, Any], run_identity_sha256: str, output_root: Path
) -> Path:
    candidate_path = Path(str(candidate["path"])).resolve()
    ledger = candidate_path.with_suffix(candidate_path.suffix + ".test-consumption.json")
    payload = {
        "schema_version": TEST_LEDGER_SCHEMA,
        "candidate_json": str(candidate_path),
        "candidate_json_sha256": candidate["sha256"],
        "run_identity_sha256": run_identity_sha256,
        "output_root": str(output_root.resolve()),
    }
    if ledger.is_file():
        if canonical_json(read_json(ledger)) != canonical_json(payload):
            raise ValueError(
                f"Frozen Test candidate was already consumed by another run: {ledger}"
            )
        return ledger
    ledger.parent.mkdir(parents=True, exist_ok=True)
    partial = ledger.with_suffix(ledger.suffix + f".{os.getpid()}.partial")
    partial.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    try:
        os.link(partial, ledger)
    except FileExistsError:
        if canonical_json(read_json(ledger)) != canonical_json(payload):
            raise ValueError(
                f"Frozen Test candidate was concurrently consumed: {ledger}"
            )
    finally:
        partial.unlink(missing_ok=True)
    return ledger


def _shape_error(label: str, observed: tuple[int, ...], expected: tuple[Any, ...]) -> None:
    raise ValueError(f"{label} shape {observed}, expected {expected}")


def validate_cache_arrays(cache: Path, methods: list[str], frame_count: int) -> None:
    with np.load(cache, allow_pickle=False) as data:
        required = {f"{method}__{key}" for method in methods for key in ARRAY_KEYS}
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"Cache misses {sorted(missing)}")
        unexpected = set(data.files) - required
        if unexpected:
            raise KeyError(f"Cache has undeclared arrays {sorted(unexpected)}")
        shared_people = None
        for method in methods:
            prefix = method + "__"
            cameras = np.asarray(data[prefix + "cameras_c2w"])
            vertices = np.asarray(data[prefix + "vertices_world"])
            joints = np.asarray(data[prefix + "joints_world"])
            persistent = np.asarray(data[prefix + "persistent_ids"])
            native = np.asarray(data[prefix + "native_ids"])
            valid_raw = np.asarray(data[prefix + "valid"])
            if cameras.shape != (frame_count, 4, 4):
                _shape_error(prefix + "cameras_c2w", cameras.shape, (frame_count, 4, 4))
            if vertices.ndim != 4 or vertices.shape[:1] != (frame_count,) or vertices.shape[2:] != (6890, 3):
                _shape_error(prefix + "vertices_world", vertices.shape, (frame_count, "P", 6890, 3))
            people = vertices.shape[1]
            if shared_people is None:
                shared_people = people
            elif people != shared_people:
                raise ValueError(f"Method person-axis mismatch in {method}")
            if joints.shape != (frame_count, people, 24, 3):
                _shape_error(prefix + "joints_world", joints.shape, (frame_count, people, 24, 3))
            if persistent.shape != (frame_count, people):
                _shape_error(prefix + "persistent_ids", persistent.shape, (frame_count, people))
            if native.shape != (frame_count, people):
                _shape_error(prefix + "native_ids", native.shape, (frame_count, people))
            if valid_raw.shape != (frame_count, people):
                _shape_error(prefix + "valid", valid_raw.shape, (frame_count, people))
            if not np.issubdtype(persistent.dtype, np.integer):
                raise ValueError(f"Non-integer persistent IDs in {method}")
            if not np.issubdtype(native.dtype, np.integer):
                raise ValueError(f"Non-integer native IDs in {method}")
            if not (
                np.issubdtype(valid_raw.dtype, np.bool_)
                or np.issubdtype(valid_raw.dtype, np.integer)
            ):
                raise ValueError(f"Non-boolean/integer validity mask in {method}")
            if not np.isin(valid_raw, [0, 1]).all():
                raise ValueError(f"Non-binary validity mask in {method}")
            valid = valid_raw.astype(bool)
            if not np.isfinite(cameras).all():
                raise ValueError(f"Non-finite cameras in {method}")
            if valid.any():
                if not np.isfinite(vertices[valid]).all() or not np.isfinite(joints[valid]).all():
                    raise ValueError(f"Non-finite valid human geometry in {method}")
                if (persistent[valid] < 0).any() or (native[valid] < 0).any():
                    raise ValueError(f"Negative ID on a valid detection in {method}")


def validate_prediction(
    cache: Path,
    runtime_path: Path,
    runtime_manifest: Path,
    manifest_sha256: str,
    manifest_line: int,
    row: dict[str, Any],
    expected_checkpoints: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    if not cache.is_file() or not runtime_path.is_file():
        raise FileNotFoundError(cache if not cache.is_file() else runtime_path)
    report = read_json(runtime_path)
    if report.get("schema_version") != RUNTIME_CACHE_SCHEMA:
        raise ValueError(f"Unexpected runtime-cache schema in {runtime_path}")
    if canonical_json(report.get("record")) != canonical_json(row):
        raise ValueError(f"Runtime record differs from frozen manifest for {row['case_id']}")
    methods = [str(value) for value in report.get("methods", [])]
    if methods[: len(CORE_METHODS)] != list(CORE_METHODS):
        raise ValueError(f"Missing/reordered core methods in {row['case_id']}: {methods}")
    if len(methods) != len(set(methods)):
        raise ValueError(f"Duplicate runtime methods in {row['case_id']}")
    actual_cache_sha = file_sha256(cache)
    if str(report.get("cache_sha256")) != actual_cache_sha:
        raise ValueError(f"Cache SHA mismatch in {row['case_id']}")
    if Path(str(report.get("cache", ""))).resolve() != cache.resolve():
        raise ValueError(f"Runtime cache path mismatch in {row['case_id']}")
    provenance = report.get("provenance", {})
    if Path(str(provenance.get("manifest", ""))).resolve() != runtime_manifest.resolve():
        raise ValueError(f"Runtime manifest path mismatch in {row['case_id']}")
    if str(provenance.get("manifest_sha256")) != manifest_sha256:
        raise ValueError(f"Runtime manifest SHA mismatch in {row['case_id']}")
    if int(provenance.get("manifest_line", -1)) != int(manifest_line):
        raise ValueError(f"Runtime manifest line mismatch in {row['case_id']}")
    contract = report.get("runtime_contract", {})
    if contract.get("gt_in_runtime") is not False or int(contract.get("future_frames_at_boundary", -1)) != 0:
        raise ValueError(f"Runtime causal/GT contract failed in {row['case_id']}")
    checkpoint = report.get("checkpoint", {})
    for key in ("current_sha256", "original_sha256", "detector_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(checkpoint.get(key, ""))):
            raise ValueError(f"Invalid checkpoint provenance {key} in {row['case_id']}")
    for label, expected in (expected_checkpoints or {}).items():
        if Path(str(checkpoint.get(label, ""))).resolve() != Path(expected["path"]):
            raise ValueError(f"{label} checkpoint path mismatch in {row['case_id']}")
        if str(checkpoint.get(f"{label}_sha256")) != expected["sha256"]:
            raise ValueError(f"{label} checkpoint SHA mismatch in {row['case_id']}")
    validate_cache_arrays(
        cache,
        methods,
        len(row["pre_frame_numbers"]) + len(row["post_frame_numbers"]),
    )
    return {
        "cache_sha256": actual_cache_sha,
        "runtime_report_sha256": file_sha256(runtime_path),
        "methods": methods,
        "checkpoint": checkpoint,
    }


def prediction_valid(*args: Any, **kwargs: Any) -> dict[str, Any] | None:
    try:
        return validate_prediction(*args, **kwargs)
    except (
        OSError, ValueError, KeyError, TypeError, AttributeError,
        json.JSONDecodeError,
    ):
        return None


def validate_evaluation(
    output: Path,
    cache: Path,
    runtime_path: Path,
    case_id: str,
    expected_methods: list[str],
    gt_root: Path,
) -> dict[str, Any]:
    payload = read_json(output)
    if payload.get("schema_version") != EVALUATION_SCHEMA:
        raise ValueError(f"Unexpected evaluation schema in {output}")
    if str(payload.get("case_id")) != case_id:
        raise ValueError(f"Evaluation case mismatch in {output}")
    inputs = payload.get("inputs", {})
    if not isinstance(inputs, dict):
        raise ValueError(f"Invalid evaluation inputs in {case_id}")
    if Path(str(inputs.get("prediction_cache", ""))).resolve() != cache.resolve():
        raise ValueError(f"Evaluation prediction path mismatch in {case_id}")
    if str(inputs.get("prediction_cache_sha256")) != file_sha256(cache):
        raise ValueError(f"Evaluation prediction SHA mismatch in {case_id}")
    if Path(str(inputs.get("runtime_report", ""))).resolve() != runtime_path.resolve():
        raise ValueError(f"Evaluation runtime path mismatch in {case_id}")
    if str(inputs.get("runtime_report_sha256")) != file_sha256(runtime_path):
        raise ValueError(f"Evaluation runtime SHA mismatch in {case_id}")
    expected_gt = (gt_root.resolve() / f"{case_id}.gt.npz").resolve()
    if Path(str(inputs.get("gt_cache", ""))).resolve() != expected_gt:
        raise ValueError(f"Evaluation GT path mismatch in {case_id}")
    methods = payload.get("methods", {})
    if not isinstance(methods, dict):
        raise ValueError(f"Invalid evaluation methods in {case_id}")
    results = set(methods)
    errors = payload.get("errors", {})
    if not isinstance(errors, dict):
        raise ValueError(f"Invalid evaluation errors in {case_id}")
    expected = set(expected_methods)
    if results | set(errors) != expected:
        raise ValueError(f"Evaluation method coverage mismatch in {case_id}")
    if not errors:
        status = "complete"
    elif (
        not results
        and len(set(map(str, errors.values()))) == 1
        and str(next(iter(errors.values()))).startswith(KNOWN_EVALUATOR_UNAVAILABLE)
    ):
        status = "evaluator_unavailable"
    else:
        raise ValueError(f"Method-dependent evaluator errors in {case_id}: {errors}")
    return {
        "status": status,
        "output_sha256": file_sha256(output),
        "complete_methods": sorted(results),
        "errors": errors,
    }


def evaluation_valid(*args: Any, **kwargs: Any) -> dict[str, Any] | None:
    try:
        return validate_evaluation(*args, **kwargs)
    except (
        OSError, ValueError, KeyError, TypeError, AttributeError,
        json.JSONDecodeError,
    ):
        return None


def _quarantine(paths: Iterable[Path], root: Path, case_id: str) -> list[str]:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return []
    destination = root / "quarantine" / case_id / str(time.time_ns())
    destination.mkdir(parents=True, exist_ok=False)
    output = []
    for path in existing:
        target = destination / path.name
        os.replace(path, target)
        output.append(str(target.resolve()))
    return output


def _finish_log(handle: Any, partial: Path, final: Path) -> None:
    handle.flush()
    handle.close()
    os.replace(partial, final)


def launch_inference(
    cases: list[tuple[int, dict[str, Any], dict[str, Any]]],
    runtime_manifest: Path,
    staged_root: Path,
    output_root: Path,
    devices: list[str],
    size: int,
    current_checkpoint: Path | None,
    original_checkpoint: Path | None,
    state: dict[str, Any],
    state_path: Path,
    expected_checkpoints: dict[str, dict[str, str]],
) -> None:
    predictions = output_root / "predictions"
    logs = output_root / "logs/inference"
    predictions.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    manifest_sha = file_sha256(runtime_manifest)
    pending: deque[tuple[int, dict[str, Any]]] = deque()
    progress = dict(state.get("inference", {}))
    for line, row, _ in cases:
        case_id = str(row["case_id"])
        cache = predictions / f"{case_id}.npz"
        runtime = predictions / f"{case_id}.runtime.json"
        valid = prediction_valid(
            cache, runtime, runtime_manifest, manifest_sha, line, row,
            expected_checkpoints,
        )
        if valid is not None:
            progress[case_id] = {"status": "complete", "cache_reused": True, **valid}
        else:
            quarantined = _quarantine(
                (cache, runtime, cache.with_suffix(".npz.partial")), output_root, case_id
            )
            progress[case_id] = {"status": "pending", "quarantined": quarantined}
            pending.append((line, row))
    state["inference"] = progress
    atomic_json(state_path, state)

    active: dict[str, tuple[subprocess.Popen[Any], Any, Path, Path, int, dict[str, Any], float, list[str]]] = {}
    failures = []
    while pending or active:
        while pending and len(active) < len(devices):
            device = next(value for value in devices if value not in active)
            line, row = pending.popleft()
            case_id = str(row["case_id"])
            cache = predictions / f"{case_id}.npz"
            command = [
                sys.executable,
                str(RUNNER),
                "--manifest",
                str(runtime_manifest),
                "--line",
                str(line),
                "--staged-root",
                str(staged_root),
                "--output",
                str(cache),
                "--device",
                device,
                "--size",
                str(size),
            ]
            if current_checkpoint is not None:
                command.extend(("--current-checkpoint", str(current_checkpoint)))
            if original_checkpoint is not None:
                command.extend(("--original-checkpoint", str(original_checkpoint)))
            final_log = logs / f"{case_id}.log"
            partial_log = final_log.with_suffix(final_log.suffix + ".partial")
            handle = partial_log.open("w", encoding="utf-8")
            handle.write("COMMAND " + json.dumps(command, ensure_ascii=False) + "\n")
            handle.flush()
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            active[device] = (
                process, handle, partial_log, final_log, line, row,
                time.perf_counter(), command,
            )
            progress[case_id] = {
                "status": "running", "device": device, "pid": process.pid,
                "command": command,
            }
            state["inference"] = progress
            atomic_json(state_path, state)
        if not active:
            continue
        time.sleep(0.5)
        for device, values in list(active.items()):
            process, handle, partial_log, final_log, line, row, started, command = values
            returncode = process.poll()
            if returncode is None:
                continue
            _finish_log(handle, partial_log, final_log)
            case_id = str(row["case_id"])
            cache = predictions / f"{case_id}.npz"
            runtime = predictions / f"{case_id}.runtime.json"
            valid = (
                prediction_valid(
                    cache, runtime, runtime_manifest, manifest_sha, line, row,
                    expected_checkpoints,
                )
                if returncode == 0
                else None
            )
            progress[case_id] = {
                "status": "complete" if valid is not None else "error",
                "device": device,
                "returncode": returncode,
                "seconds": time.perf_counter() - started,
                "command": command,
                "log": str(final_log.resolve()),
                **(valid or {}),
            }
            if valid is None:
                failures.append({"case_id": case_id, **progress[case_id]})
            del active[device]
            state["inference"] = progress
            atomic_json(state_path, state)
    if failures:
        raise RuntimeError(f"Inference failures: {failures}")


def run_logged(command: list[str], log: Path) -> tuple[int, float]:
    log.parent.mkdir(parents=True, exist_ok=True)
    partial = log.with_suffix(log.suffix + ".partial")
    started = time.perf_counter()
    with partial.open("w", encoding="utf-8") as handle:
        handle.write("COMMAND " + json.dumps(command, ensure_ascii=False) + "\n")
        handle.flush()
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    os.replace(partial, log)
    return completed.returncode, time.perf_counter() - started


def run_evaluations(
    cases: list[tuple[int, dict[str, Any], dict[str, Any]]],
    gt_root: Path,
    output_root: Path,
    state: dict[str, Any],
    state_path: Path,
) -> None:
    predictions = output_root / "predictions"
    evaluations = output_root / "evaluations"
    logs = output_root / "logs/evaluation"
    evaluations.mkdir(parents=True, exist_ok=True)
    progress = dict(state.get("evaluation", {}))
    failures = []
    for _, row, _ in cases:
        case_id = str(row["case_id"])
        cache = predictions / f"{case_id}.npz"
        runtime_path = predictions / f"{case_id}.runtime.json"
        runtime = read_json(runtime_path)
        methods = [str(value) for value in runtime["methods"]]
        output = evaluations / f"{case_id}.evaluation.json"
        valid = evaluation_valid(
            output, cache, runtime_path, case_id, methods, gt_root
        )
        if valid is not None:
            progress[case_id] = {"status": valid["status"], "reused": True, **valid}
            state["evaluation"] = progress
            atomic_json(state_path, state)
            continue
        quarantined = _quarantine((output,), output_root, case_id)
        command = [
            sys.executable,
            str(EVALUATOR),
            "--cache",
            str(cache),
            "--runtime-report",
            str(runtime_path),
            "--gt-root",
            str(gt_root),
            "--output",
            str(output),
        ]
        progress[case_id] = {
            "status": "running", "command": command, "quarantined": quarantined
        }
        state["evaluation"] = progress
        atomic_json(state_path, state)
        returncode, seconds = run_logged(command, logs / f"{case_id}.log")
        valid = (
            evaluation_valid(
                output, cache, runtime_path, case_id, methods, gt_root
            )
            if returncode == 0
            else None
        )
        progress[case_id] = {
            "status": valid["status"] if valid is not None else "error",
            "returncode": returncode,
            "seconds": seconds,
            "command": command,
            "log": str((logs / f"{case_id}.log").resolve()),
            **(valid or {}),
        }
        state["evaluation"] = progress
        atomic_json(state_path, state)
        if valid is None:
            failures.append({"case_id": case_id, **progress[case_id]})
    if failures:
        raise RuntimeError(f"Evaluation failures: {failures}")


def validate_candidate_report(
    path: Path,
    expected_cases: set[str],
    candidate_path: Path,
) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("schema_version") != PROBE_SCHEMA:
        raise ValueError(f"Unexpected candidate-probe schema in {path}")
    errors = payload.get("errors")
    if not isinstance(errors, list) or errors:
        raise ValueError(f"Candidate probe errors in {path}: {errors}")
    if Path(str(payload.get("candidate_source", ""))).resolve() != candidate_path.resolve():
        raise ValueError(f"Candidate source mismatch in {path}")
    spec = read_json(candidate_path)
    expected_candidates = {
        str(row.get("name", "")) for row in spec.get("candidates", [])
    } | {CANDIDATE_BASELINE}
    if "" in expected_candidates:
        raise ValueError(f"Candidate source has an empty name: {candidate_path}")
    rows, references = payload.get("rows"), payload.get("reference_rows")
    skipped_rows = payload.get("skipped_cases")
    if not isinstance(rows, list) or not isinstance(references, list) or not isinstance(skipped_rows, list):
        raise ValueError(f"Malformed candidate rows in {path}")
    skipped = set()
    for row in skipped_rows:
        case_id = str(row.get("case_id", ""))
        if (
            case_id not in expected_cases
            or case_id in skipped
            or row.get("status") != "evaluator_unavailable"
            or row.get("method_independent") is not True
        ):
            raise ValueError(f"Invalid skipped candidate case in {path}: {row}")
        skipped.add(case_id)

    def index(
        values: list[dict[str, Any]], field: str
    ) -> dict[str, dict[str, dict[str, Any]]]:
        result: dict[str, dict[str, dict[str, Any]]] = {}
        for row in values:
            if not isinstance(row, dict):
                raise ValueError(f"Non-object candidate row in {path}")
            case_id, name = str(row.get("case_id", "")), str(row.get(field, ""))
            if case_id not in expected_cases or not name:
                raise ValueError(f"Unexpected candidate row in {path}: {(case_id, name)}")
            case = result.setdefault(case_id, {})
            if name in case:
                raise ValueError(f"Duplicate candidate row in {path}: {(case_id, name)}")
            case[name] = row
        return result

    candidate_index = index(rows, "candidate")
    reference_index = index(references, "method")
    for case_id in expected_cases:
        if set(candidate_index.get(case_id, {})) != expected_candidates:
            raise ValueError(f"Candidate coverage mismatch in {case_id}")
        if set(reference_index.get(case_id, {})) != set(CORE_METHODS):
            raise ValueError(f"Reference coverage mismatch in {case_id}")
        expected_status = "error" if case_id in skipped else "complete"
        case_rows = [
            *candidate_index[case_id].values(), *reference_index[case_id].values()
        ]
        if any(row.get("status") != expected_status for row in case_rows):
            raise ValueError(f"Candidate status mismatch in {case_id}")
    if int(payload.get("candidate_count", -1)) != len(expected_candidates):
        raise ValueError(f"Candidate count mismatch in {path}")
    if int(payload.get("case_count", -1)) != len(expected_cases):
        raise ValueError(f"Candidate case count mismatch in {path}")
    if int(payload.get("complete_case_count", -1)) != len(expected_cases - skipped):
        raise ValueError(f"Candidate complete-case count mismatch in {path}")
    contract = payload.get("contract", {})
    if (
        not isinstance(contract, dict)
        or contract.get("candidate_runtime_uses_gt") is not False
        or int(contract.get("future_frames_at_boundary", -1)) != 0
        or contract.get("source_cache_immutable") is not True
    ):
        raise ValueError(f"Candidate runtime contract mismatch in {path}")
    return {
        "status": "complete",
        "output_sha256": file_sha256(path),
        "complete_case_count": int(payload.get("complete_case_count", 0)),
        "evaluator_unavailable": len(skipped),
    }


def run_candidate_probes(
    cases: list[tuple[int, dict[str, Any], dict[str, Any]]],
    candidates: list[dict[str, Any]],
    gt_root: Path,
    output_root: Path,
    state: dict[str, Any],
    state_path: Path,
) -> None:
    if not candidates:
        return
    predictions = output_root / "predictions"
    reports = output_root / "candidate_reports"
    logs = output_root / "logs/candidates"
    reports.mkdir(parents=True, exist_ok=True)
    expected = {str(row["case_id"]) for _, row, _ in cases}
    include = "^(?:" + "|".join(re.escape(value) for value in sorted(expected)) + ")$"
    progress = dict(state.get("candidate_reports", {}))
    for metadata in candidates:
        candidate_path = Path(metadata["path"])
        tag = f"{candidate_path.stem}_{metadata['sha256'][:12]}"
        output = reports / f"{tag}.json"
        try:
            valid = validate_candidate_report(output, expected, candidate_path)
        except (
            OSError, ValueError, KeyError, TypeError, AttributeError,
            json.JSONDecodeError,
        ):
            valid = None
        if valid is not None:
            progress[tag] = {"reused": True, **valid}
            state["candidate_reports"] = progress
            atomic_json(state_path, state)
            continue
        quarantined = _quarantine(
            (output, output.with_suffix(".csv")), output_root, f"candidate_{tag}"
        )
        command = [
            sys.executable,
            str(PROBE),
            "--prediction-roots",
            str(predictions),
            "--extracted-root",
            str(gt_root),
            "--output",
            str(output),
            "--candidate-json",
            str(candidate_path),
            "--reference-methods",
            *CORE_METHODS,
            "--include-case-regex",
            include,
        ]
        progress[tag] = {
            "status": "running", "command": command, "quarantined": quarantined
        }
        state["candidate_reports"] = progress
        atomic_json(state_path, state)
        returncode, seconds = run_logged(command, logs / f"{tag}.log")
        try:
            valid = (
                validate_candidate_report(output, expected, candidate_path)
                if returncode == 0
                else None
            )
        except (
            OSError, ValueError, KeyError, TypeError, AttributeError,
            json.JSONDecodeError,
        ):
            valid = None
        progress[tag] = {
            "status": "complete" if valid is not None else "error",
            "returncode": returncode,
            "seconds": seconds,
            "command": command,
            "log": str((logs / f"{tag}.log").resolve()),
            **(valid or {}),
        }
        state["candidate_reports"] = progress
        atomic_json(state_path, state)
        if valid is None:
            raise RuntimeError(f"Candidate probe failed: {tag}")


def main() -> None:
    args = parse_args()
    runtime_manifest = args.runtime_manifest.resolve()
    evaluator_manifest = args.evaluator_manifest.resolve()
    staged_root = args.staged_root.resolve()
    gt_root = args.gt_root.resolve()
    output_root = args.output_dir.resolve()
    for path in (runtime_manifest, evaluator_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (staged_root, gt_root):
        if not path.is_dir():
            raise FileNotFoundError(path)
    if not RUNNER.is_file() or not EVALUATOR.is_file() or not PROBE.is_file():
        raise FileNotFoundError("EgoBody runner/evaluator/probe is incomplete")
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    if not devices or len(devices) != len(set(devices)):
        raise ValueError("--devices must contain distinct device names")
    if not all(re.fullmatch(r"cuda:\d+", value) for value in devices):
        raise ValueError("Every device must be an explicit cuda:<index>")
    if args.max_cases is not None and int(args.max_cases) <= 0:
        raise ValueError("--max-cases must be positive")
    if int(args.size) <= 0:
        raise ValueError("--size must be positive")
    cases = validate_manifest_pair(runtime_manifest, evaluator_manifest, args.split)
    if args.max_cases is not None:
        cases = cases[: int(args.max_cases)]
    candidate_rows = candidate_metadata(args.candidate_json, args.split)
    explicit_checkpoints = {
        label: {"path": str(path.resolve()), "sha256": file_sha256(path.resolve())}
        for label, path in (
            ("current", args.current_checkpoint),
            ("original", args.original_checkpoint),
        )
        if path is not None
    }
    run_identity = {
        "split": args.split,
        "runtime_manifest": str(runtime_manifest),
        "runtime_manifest_sha256": file_sha256(runtime_manifest),
        "evaluator_manifest": str(evaluator_manifest),
        "evaluator_manifest_sha256": file_sha256(evaluator_manifest),
        "staged_root": str(staged_root),
        "gt_root": str(gt_root),
        "output_root": str(output_root),
        "selected_case_ids": [row["case_id"] for _, row, _ in cases],
        "candidate_json": candidate_rows,
        "size": int(args.size),
        "explicit_checkpoints": explicit_checkpoints,
        "driver_sha256": file_sha256(DRIVER),
        "runner_sha256": file_sha256(RUNNER),
        "evaluator_sha256": file_sha256(EVALUATOR),
        "probe_sha256": file_sha256(PROBE),
    }
    identity_sha = value_sha256(run_identity)
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "protocol_state.json"
    state: dict[str, Any]
    if state_path.is_file():
        previous = read_json(state_path)
        if previous.get("schema_version") != STATE_SCHEMA:
            raise ValueError("Existing output has an incompatible protocol-state schema")
        if previous.get("run_identity_sha256") != identity_sha:
            raise ValueError(
                "Existing output has a different frozen run identity; use a new output directory"
            )
        if canonical_json(previous.get("run_identity")) != canonical_json(run_identity):
            raise ValueError("Existing protocol state has inconsistent run-identity contents")
        state = previous
    else:
        state = {
            "schema_version": STATE_SCHEMA,
            "run_identity": run_identity,
            "run_identity_sha256": identity_sha,
            "runtime_manifest": str(runtime_manifest),
            "evaluator_manifest": str(evaluator_manifest),
            "staged_root": str(staged_root),
            "gt_root": str(gt_root),
            "split": args.split,
            "devices": devices,
            "max_cases": args.max_cases,
            "smoke_subset": args.max_cases is not None,
            "selected_case_count": len(cases),
            "candidate_json": candidate_rows,
            "status": "planned",
            "started_at": time.time(),
            "inference": {},
            "evaluation": {},
            "candidate_reports": {},
        }
    state["devices"] = devices
    state["dry_run"] = bool(args.dry_run)
    if args.split == "test" and state.get("status") == "complete":
        raise ValueError("This frozen Test run is already complete and may not be rerun")
    if args.dry_run:
        state.update(status="dry_run_complete", completed_at=time.time())
        atomic_json(state_path, state)
        print(
            json.dumps(
                {
                    "status": state["status"],
                    "state": str(state_path),
                    "cases": len(cases),
                    "split": args.split,
                    "candidate_files": len(candidate_rows),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    if args.split == "test":
        ledger = reserve_test_consumption(candidate_rows[0], identity_sha, output_root)
        state["test_consumption_ledger"] = str(ledger)
        state["test_consumption_ledger_sha256"] = file_sha256(ledger)
    state.update(status="running", current_stage="inference")
    atomic_json(state_path, state)
    launch_inference(
        cases,
        runtime_manifest,
        staged_root,
        output_root,
        devices,
        int(args.size),
        args.current_checkpoint.resolve() if args.current_checkpoint else None,
        args.original_checkpoint.resolve() if args.original_checkpoint else None,
        state,
        state_path,
        explicit_checkpoints,
    )
    state["current_stage"] = "evaluation"
    atomic_json(state_path, state)
    run_evaluations(cases, gt_root, output_root, state, state_path)
    state["current_stage"] = "candidate_probes"
    atomic_json(state_path, state)
    run_candidate_probes(
        cases, candidate_rows, gt_root, output_root, state, state_path
    )
    state.update(status="complete", current_stage=None, completed_at=time.time())
    atomic_json(state_path, state)
    print(
        json.dumps(
            {
                "status": state["status"],
                "state": str(state_path),
                "cases": len(cases),
                "split": args.split,
                "candidate_reports": len(state.get("candidate_reports", {})),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
