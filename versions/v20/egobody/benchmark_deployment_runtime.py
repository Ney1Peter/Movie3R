#!/usr/bin/env python3
"""Execute the frozen 9-case x 3-repeat EgoBody deployment benchmark plan.

The preregistration chooses RGB-only Development rows and one separate warmup
row.  It is bound by a hard-coded SHA-256, and its source manifest is opened
only after its own byte digest is checked.  No command-line option can supply
supervision, an evaluator, or a Holdout/Test split.

Measured throughput remains explicitly non-reportable until a separate frozen
numerical equivalence audit demonstrates that ``deployment_runtime.py`` matches
the selected formal prediction path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from versions.v20.egobody.deployment_runtime import (
    PROTOCOL,
    DeploymentRuntime,
    FrozenCandidate,
    _write_new_json,
    deployment_input_from_record,
    file_sha256,
    resolve_staged_paths,
    validate_frozen_candidate,
)


PREREGISTRATION_SCHEMA = "Bridge3R-EgoBody-runtime-benchmark-preregistration-v1"
BENCHMARK_SCHEMA = "Bridge3R-EgoBody-deployment-runtime-benchmark-v1"
POST_TEST_SUPPORT_SCHEMA = "Bridge3R-EgoBody-deployment-post-test-support-v1"
EXPECTED_PREREGISTRATION_SHA256 = (
    "c3d615b5ce0ce95a07556818e323e0ab5186497f44753ace89f2ae6a56871d5c"
)
CASE_COUNT = 9
REPEATS = 3
FRAMES_PER_CASE = 150
WARMUP_MANIFEST_LINE = 4
REQUIRED_MEASUREMENT_FIELDS = (
    "case_id",
    "phase",
    "repeat_index",
    "seconds",
    "cuda_max_memory_allocated_bytes",
    "cuda_max_memory_reserved_bytes",
    "process_peak_rss_bytes",
    "branch",
    "forward_calls",
    "forward_frames",
    "input_frames",
    "output_frames",
)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = PROJECT_ROOT.parent
POST_TEST_DIRECT_SOURCE_PATHS = (
    "versions/v20/egobody/deployment_runtime.py",
    "versions/v20/egobody/benchmark_deployment_runtime.py",
    "versions/v20/egobody/tests/test_deployment_runtime.py",
    "versions/v13/gt_id_consensus.py",
    "versions/v14/b0_person_triangulation.py",
    "versions/v14/causal_image_detector.py",
    "versions/v14/eval_streaming_within_shot_stability.py",
    "versions/v14/probe_p1_foot_scene_observability.py",
    "versions/v14/run_v14_2_single_sequence.py",
    "versions/v15/harmony4d/run_harmony_case.py",
    "versions/v15/harmony4d/topology.py",
    "versions/v15/FINAL_RUNTIME_SPEC.json",
    "versions/v16/harmony4d/causal_stabilization.py",
    "scripts/v10_detector_feature_probe.py",
    "scripts/v10_image_only_detector.py",
)
POST_TEST_RECURSIVE_SOURCE_ROOTS = (
    "src/dust3r",
    "src/croco",
    "src/mhmr",
    "src/models",
)
POST_TEST_TOPOLOGY_ASSETS = (
    "src/models/smplx/smplx2smpl.pkl",
    "src/models/smpl/SMPL_NEUTRAL.pkl",
    "src/models/smplx/SMPLX_NEUTRAL.npz",
)


@dataclass(frozen=True)
class FrozenPreregistration:
    path: Path
    sha256: str
    payload: dict[str, Any]
    manifest_path: Path
    manifest_sha256: str


@dataclass(frozen=True)
class DeploymentCase:
    manifest_line: int
    case_id: str
    recording: str
    angle_stratum: str
    deployment_input: dict[str, Any]

    def metadata(self) -> dict[str, Any]:
        return {
            "manifest_line": self.manifest_line,
            "case_id": self.case_id,
            "recording": self.recording,
            "angle_stratum": self.angle_stratum,
            "frame_count": int(self.deployment_input["frame_count"]),
        }


@dataclass(frozen=True)
class BenchmarkSelection:
    warmup: DeploymentCase
    cases: tuple[DeploymentCase, ...]


def build_support_hash_inventory(
    role_paths: Mapping[Path, set[str]],
    *,
    workspace_root: Path,
) -> dict[str, Any]:
    """Hash an explicit supporting-file set without interpreting any contents."""

    workspace = Path(workspace_root).resolve()
    if not workspace.is_dir() or Path(workspace_root).is_symlink():
        raise FileNotFoundError(workspace)
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_path, roles in role_paths.items():
        raw = Path(raw_path)
        if raw.is_symlink():
            raise ValueError(f"Supporting provenance rejects symlinks: {raw}")
        path = raw.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            relative = path.relative_to(workspace).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"Supporting provenance path escapes workspace: {path}"
            ) from exc
        if relative in seen:
            raise ValueError(f"Duplicate supporting provenance path: {relative}")
        seen.add(relative)
        entries.append(
            {
                "path": relative,
                "roles": sorted(str(role) for role in roles),
                "bytes": int(path.stat().st_size),
                "sha256": file_sha256(path),
            }
        )
    if not entries:
        raise ValueError("Post-Test supporting provenance may not be empty")
    entries.sort(key=lambda row: row["path"])
    tree_tuples = [
        [row["path"], row["bytes"], row["sha256"], row["roles"]]
        for row in entries
    ]
    root_sha = hashlib.sha256(
        (
            json.dumps(
                tree_tuples,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": POST_TEST_SUPPORT_SCHEMA,
        "provenance_timing": "post_test_supporting",
        "claim_scope": "deployment_benchmark_support_only",
        "entered_protocol_run_identity": False,
        "replaces_pre_test_lock": False,
        "test_metric_artifacts_read_or_inventoried": False,
        "test_metric_artifact_roles": [],
        "path_policy": "POSIX paths relative to the supplied workspace root",
        "tree_tuple_fields": ["path", "bytes", "sha256", "roles"],
        "file_count": len(entries),
        "total_bytes": sum(int(row["bytes"]) for row in entries),
        "root_sha256": root_sha,
        "files": entries,
    }


def build_post_test_support_provenance(
    *,
    candidate: FrozenCandidate,
    preregistration: FrozenPreregistration,
    runtime_artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the real deployment support closure after all timed measurements.

    Formal Test aggregates, per-case metrics, candidate reports, and prediction
    reports are intentionally absent.  Final-candidate provenance references
    only its already-frozen Development/Holdout selection inputs.
    """

    roles: dict[Path, set[str]] = {}

    def add(path: Path, role: str) -> None:
        roles.setdefault(Path(path), set()).add(role)

    add(candidate.path, "frozen_final_candidate")
    for row in candidate.provenance_files:
        add(Path(row["path"]), f"final_candidate_provenance:{row['role']}")
    add(preregistration.path, "runtime_benchmark_preregistration")
    add(preregistration.manifest_path, "preregistered_development_runtime_manifest")
    for relative in POST_TEST_DIRECT_SOURCE_PATHS:
        add(PROJECT_ROOT / relative, "deployment_direct_source")
    for relative_root in POST_TEST_RECURSIVE_SOURCE_ROOTS:
        root = PROJECT_ROOT / relative_root
        if not root.is_dir():
            raise FileNotFoundError(root)
        for path in sorted(root.rglob("*.py"), key=lambda value: value.as_posix()):
            if any(part in {"__pycache__", ".pytest_cache"} for part in path.parts):
                continue
            add(path, f"recursive_python_source:{relative_root}")
    for relative in POST_TEST_TOPOLOGY_ASSETS:
        add(PROJECT_ROOT / relative, "topology_or_body_model_asset")
    expected_runtime_hashes: dict[Path, str] = {}
    for path_key, role in (
        ("current_checkpoint", "current_model_checkpoint"),
        ("detector", "causal_detector_checkpoint"),
    ):
        value = runtime_artifacts.get(path_key)
        expected = str(runtime_artifacts.get(path_key + "_sha256", ""))
        if not isinstance(value, str) or not value or len(expected) != 64:
            raise ValueError(f"Runtime artifacts lack {path_key} provenance")
        path = Path(value).resolve()
        expected_runtime_hashes[path] = expected
        add(path, role)
    inventory = build_support_hash_inventory(roles, workspace_root=WORKSPACE_ROOT)
    observed_by_path = {
        (WORKSPACE_ROOT / row["path"]).resolve(): str(row["sha256"])
        for row in inventory["files"]
    }
    for path, expected in expected_runtime_hashes.items():
        if observed_by_path.get(path) != expected:
            raise ValueError(f"Runtime artifact SHA differs for {path}")
    return inventory


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object: {path}")
    return value


def validate_preregistration(
    path: Path,
    *,
    expected_sha256: str = EXPECTED_PREREGISTRATION_SHA256,
) -> FrozenPreregistration:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    observed_sha = file_sha256(resolved)
    if observed_sha != expected_sha256:
        raise ValueError(
            f"Runtime preregistration SHA-256 mismatch: {observed_sha} vs {expected_sha256}"
        )
    payload = _read_json_object(resolved)
    if payload.get("schema_version") != PREREGISTRATION_SCHEMA:
        raise ValueError("Runtime preregistration has an incompatible schema")
    if payload.get("protocol") != PROTOCOL:
        raise ValueError("Runtime preregistration has an incompatible protocol")
    required_flags = {
        "frozen_before_holdout_and_test": True,
        "holdout_metrics_read": False,
        "test_metrics_read": False,
        "final_candidate_available_at_freeze": False,
        "source_split": "development",
    }
    for key, expected in required_flags.items():
        if payload.get(key) != expected:
            raise ValueError(f"Invalid preregistration flag {key}: {payload.get(key)!r}")

    rule = payload.get("selection_rule")
    if not isinstance(rule, dict):
        raise ValueError("Runtime preregistration lacks a selection rule")
    if int(rule.get("case_count", -1)) != CASE_COUNT:
        raise ValueError("Runtime preregistration must select exactly nine cases")
    if rule.get("no_replacement") is not True:
        raise ValueError("Runtime preregistration must select without replacement")
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != CASE_COUNT:
        raise ValueError("Runtime preregistration must contain exactly nine case rows")
    if not all(isinstance(row, dict) for row in cases):
        raise ValueError("Runtime preregistration case rows must be objects")
    lines = [int(row.get("manifest_line", -1)) for row in cases]
    case_ids = [str(row.get("case_id", "")) for row in cases]
    if min(lines, default=-1) < 1 or len(lines) != len(set(lines)):
        raise ValueError("Preregistered manifest lines are invalid or duplicated")
    if not all(case_ids) or len(case_ids) != len(set(case_ids)):
        raise ValueError("Preregistered case IDs are missing or duplicated")
    recordings: dict[str, set[str]] = {}
    for row in cases:
        recordings.setdefault(str(row.get("recording", "")), set()).add(
            str(row.get("angle_stratum", ""))
        )
    if len(recordings) != 3 or any(
        strata != {"extreme", "medium", "small"} for strata in recordings.values()
    ):
        raise ValueError("Preregistration must contain three strata for three recordings")

    warmup = payload.get("warmup")
    if not isinstance(warmup, dict):
        raise ValueError("Runtime preregistration lacks its warmup row")
    if int(warmup.get("manifest_line", -1)) != WARMUP_MANIFEST_LINE:
        raise ValueError("The frozen warmup must use manifest line 4")
    if warmup.get("included_in_reported_cases") is not False:
        raise ValueError("Warmup may not enter reported cases")
    if warmup.get("included_in_steady_state_timing") is not False:
        raise ValueError("Warmup may not enter steady-state timing")
    if str(warmup.get("case_id", "")) in set(case_ids):
        raise ValueError("Warmup case overlaps a steady-state case")

    execution = payload.get("execution_contract")
    if not isinstance(execution, dict):
        raise ValueError("Runtime preregistration lacks its execution contract")
    if int(execution.get("steady_state_repeats_per_case", -1)) != REPEATS:
        raise ValueError("Runtime preregistration must use three repeats per case")
    if execution.get("ground_truth_access") is not False:
        raise ValueError("Runtime preregistration does not prohibit supervision access")
    if execution.get("evaluator_access") is not False:
        raise ValueError("Runtime preregistration does not prohibit evaluator access")

    source = payload.get("source")
    if not isinstance(source, dict):
        raise ValueError("Runtime preregistration lacks a source manifest binding")
    manifest = Path(str(source.get("development_runtime_manifest", ""))).resolve()
    manifest_sha = str(source.get("development_runtime_manifest_sha256", ""))
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if len(manifest_sha) != 64 or file_sha256(manifest) != manifest_sha:
        raise ValueError("Development runtime manifest differs from preregistration")
    return FrozenPreregistration(
        path=resolved,
        sha256=observed_sha,
        payload=payload,
        manifest_path=manifest,
        manifest_sha256=manifest_sha,
    )


def _manifest_rows(path: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank runtime-manifest row at line {line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Non-object runtime-manifest row at line {line_number}")
            output.append(row)
    if not output:
        raise ValueError("Development runtime manifest is empty")
    return output


def _select_one(rows: list[dict[str, Any]], frozen: Mapping[str, Any]) -> DeploymentCase:
    line = int(frozen["manifest_line"])
    if line < 1 or line > len(rows):
        raise IndexError(f"Preregistered line {line} is outside the runtime manifest")
    row = rows[line - 1]
    for key in ("case_id", "recording"):
        if str(row.get(key, "")) != str(frozen.get(key, "")):
            raise ValueError(f"Preregistered {key} differs at manifest line {line}")
    frozen_stratum = str(frozen.get("angle_stratum", ""))
    row_stratum = row.get("angle_stratum")
    if row_stratum is not None:
        if str(row_stratum) != frozen_stratum:
            raise ValueError(
                f"Preregistered angle_stratum differs at manifest line {line}"
            )
    elif f"_{frozen_stratum}_kinect_" not in str(row.get("case_id", "")):
        raise ValueError(
            f"Preregistered angle_stratum is not encoded in case_id at line {line}"
        )
    deployment_input = deployment_input_from_record(row)
    if int(deployment_input["frame_count"]) != FRAMES_PER_CASE:
        raise ValueError(f"Preregistered case {row['case_id']} does not have 150 RGB frames")
    return DeploymentCase(
        manifest_line=line,
        case_id=str(row["case_id"]),
        recording=str(row["recording"]),
        angle_stratum=frozen_stratum,
        deployment_input=deployment_input,
    )


def select_preregistered_cases(preregistration: FrozenPreregistration) -> BenchmarkSelection:
    # Recheck immediately before parse so a concurrent source replacement
    # cannot silently change selection after initial provenance validation.
    if file_sha256(preregistration.manifest_path) != preregistration.manifest_sha256:
        raise ValueError("Development runtime manifest changed after preregistration validation")
    rows = _manifest_rows(preregistration.manifest_path)
    cases = tuple(
        _select_one(rows, row) for row in preregistration.payload["cases"]
    )
    warmup = _select_one(rows, preregistration.payload["warmup"])
    return BenchmarkSelection(warmup=warmup, cases=cases)


def rotated_execution_order(
    cases: Sequence[DeploymentCase], repeats: int = REPEATS
) -> list[tuple[int, DeploymentCase]]:
    if len(cases) != CASE_COUNT or repeats != REPEATS:
        raise ValueError("Frozen benchmark requires nine cases and three repeats")
    return [
        (repeat, cases[(offset + repeat) % len(cases)])
        for repeat in range(repeats)
        for offset in range(len(cases))
    ]


def _validate_measurement(
    measurement: Mapping[str, Any],
    *,
    case: DeploymentCase,
    phase: str,
    repeat_index: int | None,
) -> dict[str, Any]:
    missing = [key for key in REQUIRED_MEASUREMENT_FIELDS if key not in measurement]
    if missing:
        raise ValueError(f"Runtime measurement misses fields: {missing}")
    output = dict(measurement)
    if output["case_id"] != case.case_id or output["phase"] != phase:
        raise ValueError("Runtime measurement case/phase differs from the execution plan")
    if output["repeat_index"] != repeat_index:
        raise ValueError("Runtime measurement repeat differs from the execution plan")
    if not math.isfinite(float(output["seconds"])) or float(output["seconds"]) <= 0:
        raise ValueError("Runtime latency must be finite and positive")
    if int(output["input_frames"]) != FRAMES_PER_CASE:
        raise ValueError("Runtime input frame count differs from the frozen contract")
    if int(output["output_frames"]) != FRAMES_PER_CASE:
        raise ValueError("Runtime output frame count differs from the frozen contract")
    if int(output["forward_calls"]) not in {1, 2}:
        raise ValueError("Deployment branch must use one or two model forwards")
    if int(output["forward_frames"]) not in {FRAMES_PER_CASE, FRAMES_PER_CASE + 1}:
        raise ValueError("Deployment forward-frame accounting is inconsistent")
    for key in (
        "cuda_max_memory_allocated_bytes",
        "cuda_max_memory_reserved_bytes",
        "process_peak_rss_bytes",
    ):
        if int(output[key]) < 0:
            raise ValueError(f"Negative resource metric: {key}")
    if not str(output["branch"]):
        raise ValueError("Deployment branch is missing")
    output.update(case.metadata())
    output["phase"] = phase
    output["repeat_index"] = repeat_index
    return output


def execute_benchmark_plan(
    runtime: Any,
    selection: BenchmarkSelection,
    staged_root: Path,
    *,
    path_resolver: Callable[[Mapping[str, Any], Path], list[Path]] = resolve_staged_paths,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    warmup_paths = path_resolver(selection.warmup.deployment_input, staged_root)
    warmup_raw = runtime.measure_case(
        case_id=selection.warmup.case_id,
        paths=warmup_paths,
        phase="warmup",
        repeat_index=None,
    )
    warmup = _validate_measurement(
        warmup_raw,
        case=selection.warmup,
        phase="warmup",
        repeat_index=None,
    )

    resolved = {
        case.case_id: path_resolver(case.deployment_input, staged_root)
        for case in selection.cases
    }
    runs: list[dict[str, Any]] = []
    for execution_index, (repeat, case) in enumerate(
        rotated_execution_order(selection.cases)
    ):
        raw = runtime.measure_case(
            case_id=case.case_id,
            paths=resolved[case.case_id],
            phase="steady",
            repeat_index=repeat,
        )
        row = _validate_measurement(
            raw,
            case=case,
            phase="steady",
            repeat_index=repeat,
        )
        row["execution_index"] = execution_index
        runs.append(row)
    return warmup, runs


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("Cannot compute a quantile of no values")
    position = (len(ordered) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def aggregate_steady_runs(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(runs) != CASE_COUNT * REPEATS:
        raise ValueError("Steady benchmark requires exactly 27 measurements")
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in runs:
        grouped.setdefault(str(row["case_id"]), []).append(row)
    if len(grouped) != CASE_COUNT or any(len(rows) != REPEATS for rows in grouped.values()):
        raise ValueError("Every preregistered case must have exactly three repeats")

    per_case: list[dict[str, Any]] = []
    for case_id, rows in grouped.items():
        repeat_indices = sorted(int(row["repeat_index"]) for row in rows)
        if repeat_indices != list(range(REPEATS)):
            raise ValueError(f"Case {case_id} has invalid repeat indices")
        branches = {str(row["branch"]) for row in rows}
        if len(branches) != 1:
            raise ValueError(f"Execution branch changed across repeats for {case_id}")
        seconds = [float(row["seconds"]) for row in rows]
        first = rows[0]
        per_case.append(
            {
                "case_id": case_id,
                "recording": str(first["recording"]),
                "angle_stratum": str(first["angle_stratum"]),
                "manifest_line": int(first["manifest_line"]),
                "branch": next(iter(branches)),
                "repeat_seconds": seconds,
                "median_seconds": float(statistics.median(seconds)),
                "max_cuda_memory_allocated_bytes": max(
                    int(row["cuda_max_memory_allocated_bytes"]) for row in rows
                ),
                "max_cuda_memory_reserved_bytes": max(
                    int(row["cuda_max_memory_reserved_bytes"]) for row in rows
                ),
                "max_process_peak_rss_bytes": max(
                    int(row["process_peak_rss_bytes"]) for row in rows
                ),
                "forward_calls_per_repeat": sorted(
                    {int(row["forward_calls"]) for row in rows}
                ),
                "forward_frames_per_repeat": sorted(
                    {int(row["forward_frames"]) for row in rows}
                ),
            }
        )
    per_case.sort(key=lambda row: int(row["manifest_line"]))
    medians = [float(row["median_seconds"]) for row in per_case]
    denominator = sum(medians)
    diagnostic_fps = CASE_COUNT * FRAMES_PER_CASE / denominator

    strata: dict[str, list[dict[str, Any]]] = {}
    for row in per_case:
        strata.setdefault(str(row["branch"]), []).append(row)
    branch_rows = []
    for branch, rows in sorted(strata.items()):
        seconds = sum(float(row["median_seconds"]) for row in rows)
        branch_rows.append(
            {
                "branch": branch,
                "case_count": len(rows),
                "case_ids": [row["case_id"] for row in rows],
                "nonreportable_diagnostic_fps": (
                    len(rows) * FRAMES_PER_CASE / seconds if seconds > 0 else None
                ),
            }
        )
    return {
        "case_count": CASE_COUNT,
        "repeats_per_case": REPEATS,
        "frames_per_case": FRAMES_PER_CASE,
        "per_case": per_case,
        "case_median_latency_seconds": {
            "median": float(statistics.median(medians)),
            "q1": _quantile(medians, 0.25),
            "q3": _quantile(medians, 0.75),
            "iqr": _quantile(medians, 0.75) - _quantile(medians, 0.25),
        },
        "aggregate_formula": "9*150/sum(each_case_median_of_3_seconds)",
        "aggregate_frame_count": CASE_COUNT * FRAMES_PER_CASE,
        "sum_case_median_seconds": denominator,
        "nonreportable_diagnostic_fps": diagnostic_fps,
        "reported_fps": None,
        "branch_strata": branch_rows,
        "max_cuda_memory_allocated_bytes": max(
            int(row["cuda_max_memory_allocated_bytes"]) for row in runs
        ),
        "max_cuda_memory_reserved_bytes": max(
            int(row["cuda_max_memory_reserved_bytes"]) for row in runs
        ),
        "max_process_peak_rss_bytes": max(
            int(row["process_peak_rss_bytes"]) for row in runs
        ),
        "total_forward_calls": sum(int(row["forward_calls"]) for row in runs),
        "total_forward_frames": sum(int(row["forward_frames"]) for row in runs),
    }


def audit_gpu_isolation(device_name: str) -> dict[str, Any]:
    if not device_name.startswith("cuda:") or not device_name[5:].isdigit():
        raise ValueError("Benchmark device must be an explicit cuda:<index>")
    index = device_name.split(":", 1)[1]
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                index,
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return {
            "verified": False,
            "unrelated_processes": [],
            "reason": f"GPU process query unavailable: {type(exc).__name__}",
        }
    rows = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        parts = [value.strip() for value in line.split(",", 2)]
        if len(parts) != 3 or not parts[0].isdigit():
            return {
                "verified": False,
                "unrelated_processes": [],
                "reason": "Could not parse nvidia-smi compute-process output",
            }
        pid = int(parts[0])
        if pid != os.getpid():
            rows.append(
                {"pid": pid, "process_name": parts[1], "used_gpu_memory_mib": parts[2]}
            )
    return {
        "verified": not rows,
        "unrelated_processes": rows,
        "reason": None if not rows else "Unrelated compute processes use the selected GPU",
    }


def build_benchmark_report(
    *,
    candidate: FrozenCandidate,
    preregistration: FrozenPreregistration,
    selection: BenchmarkSelection,
    load_metrics: Mapping[str, Any],
    warmup: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
    gpu_isolation: Mapping[str, Any],
    runtime_artifacts: Mapping[str, Any],
    post_test_support_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    aggregate = aggregate_steady_runs(runs)
    equivalence = {
        "verified": False,
        "status": "not_proven",
        "reason": (
            "No frozen prediction-level numerical audit demonstrates equivalence "
            "between the deployment-only and formal selected-candidate paths."
        ),
    }
    reportable = bool(equivalence["verified"] and gpu_isolation.get("verified"))
    if reportable:
        raise AssertionError("Reportability must remain closed until audit support exists")
    return {
        "schema_version": BENCHMARK_SCHEMA,
        "protocol": PROTOCOL,
        "status": "complete_nonreportable",
        "reportable_fps": False,
        "reporting_gate": {
            "equivalence_audit": equivalence,
            "gpu_isolation": dict(gpu_isolation),
            "reported_fps": None,
        },
        "contract": {
            "rgb_only": True,
            "evaluation_annotation_consumed": False,
            "evaluator_invoked": False,
            "source_split": "development",
            "case_count": CASE_COUNT,
            "repeats_per_case": REPEATS,
            "frames_per_case": FRAMES_PER_CASE,
            "warmup_manifest_line": WARMUP_MANIFEST_LINE,
            "warmup_in_steady_timing": False,
            "batch_size": 1,
            "precision": "FP32",
            "input_size": 512,
        },
        "provenance": {
            "final_candidate": str(candidate.path),
            "final_candidate_sha256": candidate.sha256,
            "final_candidate_name": candidate.name,
            "final_candidate_bound_files": list(candidate.provenance_files),
            "runtime_preregistration": str(preregistration.path),
            "runtime_preregistration_sha256": preregistration.sha256,
            "development_runtime_manifest": str(preregistration.manifest_path),
            "development_runtime_manifest_sha256": preregistration.manifest_sha256,
            "deployment_runtime": str(Path(__file__).with_name("deployment_runtime.py").resolve()),
            "deployment_runtime_sha256": file_sha256(
                Path(__file__).with_name("deployment_runtime.py").resolve()
            ),
            "benchmark_driver": str(Path(__file__).resolve()),
            "benchmark_driver_sha256": file_sha256(Path(__file__).resolve()),
        },
        "selection": {
            "warmup": selection.warmup.metadata(),
            "steady_cases": [case.metadata() for case in selection.cases],
            "execution_order": [
                {
                    "execution_index": index,
                    "repeat_index": repeat,
                    "case_id": case.case_id,
                }
                for index, (repeat, case) in enumerate(
                    rotated_execution_order(selection.cases)
                )
            ],
        },
        "load": dict(load_metrics),
        "warmup": dict(warmup),
        "steady_state": {
            "runs": [dict(row) for row in runs],
            "aggregate": aggregate,
        },
        "runtime_artifacts": dict(runtime_artifacts),
        "post_test_support_provenance": dict(post_test_support_provenance),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
        },
    }


def validate_benchmark_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != BENCHMARK_SCHEMA or report.get("protocol") != PROTOCOL:
        raise ValueError("Benchmark report schema/protocol mismatch")
    if report.get("reportable_fps") is not False:
        raise ValueError("Unaudited deployment FPS must not be reportable")
    gate = report.get("reporting_gate")
    if not isinstance(gate, dict) or gate.get("reported_fps") is not None:
        raise ValueError("Unaudited benchmark must not populate reported_fps")
    equivalence = gate.get("equivalence_audit")
    if not isinstance(equivalence, dict) or equivalence.get("verified") is not False:
        raise ValueError("Deployment equivalence gate must remain fail-closed")
    contract = report.get("contract")
    if not isinstance(contract, dict) or any(
        contract.get(key) != value
        for key, value in {
            "rgb_only": True,
            "evaluation_annotation_consumed": False,
            "evaluator_invoked": False,
            "case_count": CASE_COUNT,
            "repeats_per_case": REPEATS,
            "warmup_manifest_line": WARMUP_MANIFEST_LINE,
            "warmup_in_steady_timing": False,
        }.items()
    ):
        raise ValueError("Benchmark report contract is incomplete")
    selection = report.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("Benchmark report lacks selection metadata")
    if int(selection.get("warmup", {}).get("manifest_line", -1)) != WARMUP_MANIFEST_LINE:
        raise ValueError("Benchmark report warmup differs from preregistration")
    if len(selection.get("steady_cases", [])) != CASE_COUNT:
        raise ValueError("Benchmark report does not contain nine steady cases")
    if len(selection.get("execution_order", [])) != CASE_COUNT * REPEATS:
        raise ValueError("Benchmark report does not contain 27 ordered executions")
    steady = report.get("steady_state")
    if not isinstance(steady, dict) or len(steady.get("runs", [])) != CASE_COUNT * REPEATS:
        raise ValueError("Benchmark report does not contain 27 measurements")
    aggregate = steady.get("aggregate")
    if not isinstance(aggregate, dict):
        raise ValueError("Benchmark report lacks its aggregate")
    if aggregate.get("reported_fps") is not None:
        raise ValueError("Unaudited benchmark aggregate must not report FPS")
    diagnostic = float(aggregate.get("nonreportable_diagnostic_fps", math.nan))
    if not math.isfinite(diagnostic) or diagnostic <= 0:
        raise ValueError("Benchmark report diagnostic throughput is invalid")
    load = report.get("load")
    if not isinstance(load, dict) or any(
        key not in load
        for key in (
            "seconds",
            "cuda_max_memory_allocated_bytes",
            "cuda_max_memory_reserved_bytes",
            "process_peak_rss_bytes",
            "forward_calls",
            "forward_frames",
            "branch",
        )
    ):
        raise ValueError("Benchmark report lacks model-load resource accounting")
    support = report.get("post_test_support_provenance")
    if (
        not isinstance(support, dict)
        or support.get("schema_version") != POST_TEST_SUPPORT_SCHEMA
        or support.get("provenance_timing") != "post_test_supporting"
        or support.get("claim_scope") != "deployment_benchmark_support_only"
        or support.get("entered_protocol_run_identity") is not False
        or support.get("replaces_pre_test_lock") is not False
        or support.get("test_metric_artifacts_read_or_inventoried") is not False
        or support.get("test_metric_artifact_roles") != []
        or int(support.get("file_count", 0)) < 1
        or len(str(support.get("root_sha256", ""))) != 64
    ):
        raise ValueError("Benchmark report lacks fail-closed post-Test provenance")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-candidate", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--staged-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--current-checkpoint", type=Path)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    preregistration = validate_preregistration(args.preregistration)
    candidate = validate_frozen_candidate(args.final_candidate)
    selection = select_preregistered_cases(preregistration)
    gpu_isolation = audit_gpu_isolation(args.device)
    if gpu_isolation["unrelated_processes"]:
        raise RuntimeError("Selected GPU has unrelated compute processes; refusing benchmark")
    runtime = DeploymentRuntime.load(
        candidate,
        device_name=args.device,
        size=args.size,
        current_checkpoint=args.current_checkpoint,
    )
    warmup, runs = execute_benchmark_plan(runtime, selection, args.staged_root)
    support_provenance = build_post_test_support_provenance(
        candidate=candidate,
        preregistration=preregistration,
        runtime_artifacts=runtime.artifacts,
    )
    report = build_benchmark_report(
        candidate=candidate,
        preregistration=preregistration,
        selection=selection,
        load_metrics=runtime.load_metrics,
        warmup=warmup,
        runs=runs,
        gpu_isolation=gpu_isolation,
        runtime_artifacts=runtime.artifacts,
        post_test_support_provenance=support_provenance,
    )
    validate_benchmark_report(report)
    _write_new_json(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "reportable_fps": False,
                "nonreportable_diagnostic_fps": report["steady_state"]["aggregate"][
                    "nonreportable_diagnostic_fps"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
