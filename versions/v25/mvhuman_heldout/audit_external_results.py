#!/usr/bin/env python3
"""Fail-closed MVH150 external-baseline audit and aggregation.

Two deliberately different contracts are exposed:

* ``prompthmr-formal`` seals exactly all 50 frozen cases and produces fixed-
  denominator, case- and capture-macro aggregates.
* ``gvhmr-pilot`` seals exactly the twelve pre-registered pilot rows and emits
  an outcome-independent full-test gate.  The gate uses only artifact,
  timeline, native-track, raw-tracker, and declared camera-availability
  predicates; it never thresholds pose or camera accuracy.

Use ``--diagnose`` while jobs are incomplete.  Diagnostic mode is read-only,
never writes a formal ledger, and cannot approve a result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
FRAMES, FPS, CUT = 150, 30, 74
EXPECTED_CASES = 50
PILOT_LINES = (1, 5, 6, 10, 21, 25, 26, 30, 31, 35, 46, 50)
EXPECTED_RUNTIME_KEYS = {"case_id", "dataset", "protocol", "role", "input_video", "fps", "num_frames"}
REPORT_SCHEMA = "Bridge3R-MVHuman-Heldout-MVH150-evaluation-v1"
PROMPT_RUNTIME_SCHEMA = "Bridge3R-MVHuman-PromptHMR-official-runtime-v1"
GVHMR_RUNTIME_SCHEMAS = {
    "Bridge3R-MVHuman-GVHMR-official-runtime-v1",
    "Bridge3R-MVHuman-GVHMR-official-native-recovery-v1",
}
PROMPT_ADAPTER_SCHEMA = "Bridge3R-MVHuman-PromptHMR-adapter-v1"
GVHMR_ADAPTER_SCHEMA = "Bridge3R-MVHuman-GVHMR-adapter-v1"
PROMPT_METHOD, GVHMR_METHOD = "prompthmr_official", "gvhmr_official"
PROMPT_SPEC_CHECKPOINT = "data/pretrain/camcalib_sa_biased_l2.ckpt"
PROMPT_SPEC_SHA256 = "e4480cdd546ff8322978ef76e93c7a70a7d5e82c1390cbdd7a95473ac4595b48"
METRICS = (
    "pa_mpjpe_body12_mm",
    "first_shot_anchor_mpjpe_body12_mm",
    "first_shot_anchor_root_error_mm",
    "first_shot_anchor_orientation_proxy_deg",
    "seam_root_excess_mm",
    "seam_orientation_excess_deg",
    "post_camera_relative_rotation_deg",
    "post_camera_relative_translation_m",
)
HUMAN_METRICS = METRICS[:6]
CAMERA_METRICS = METRICS[6:]
GVHMR_CHECKPOINTS = (
    "gvhmr/gvhmr_siga24_release.ckpt",
    "hmr2/epoch=10-step=25000.ckpt",
    "vitpose/vitpose-h-multi-coco.pth",
    "yolo/yolov8x.pt",
    "dpvo/dpvo.pth",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prompthmr-formal", "gvhmr-pilot"), required=True)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Fresh formal ledger/gate JSON (required unless --diagnose)")
    parser.add_argument("--tex-output", type=Path, help="Optional fresh PromptHMR aggregate table")
    parser.add_argument("--diagnose", action="store_true", help="Print inventory only; never approve or write")
    return parser.parse_args()


@lru_cache(maxsize=None)
def sha256(path: Path) -> str:
    path = path.resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("content_sha256", None)
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not values or not all(isinstance(value, dict) for value in values):
        raise ValueError(f"empty or malformed JSONL: {path}")
    return values


def require_hash(path: Path, expected: Any, label: str) -> str:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"{label} has no valid recorded SHA-256")
    observed = sha256(path)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, observed {observed}")
    return observed


def verify_content_hash(value: dict[str, Any], label: str) -> None:
    if value.get("content_sha256") != canonical_digest(value):
        raise ValueError(f"{label} canonical content SHA-256 mismatch")


def protocol(args: argparse.Namespace) -> dict[str, Any]:
    root = args.protocol_root.resolve()
    runtime_path = root / "manifests" / "test_runtime.jsonl"
    evaluator_path = root / "manifests" / "test_evaluator.jsonl"
    ledger_path = root / "materialization_ledger.json"
    freeze_path = root / "protocol_freeze.json"
    for path in (runtime_path, evaluator_path, ledger_path, freeze_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    runtime, evaluator, ledger = read_jsonl(runtime_path), read_jsonl(evaluator_path), read_json(ledger_path)
    if len(runtime) != EXPECTED_CASES or len(evaluator) != EXPECTED_CASES:
        raise ValueError("MVH150 formal manifests must each contain exactly 50 rows")
    runtime_ids = [str(row.get("case_id")) for row in runtime]
    evaluator_ids = [str(row.get("case_id")) for row in evaluator]
    if runtime_ids != evaluator_ids or len(set(runtime_ids)) != EXPECTED_CASES:
        raise ValueError("runtime/evaluator manifests are not an aligned unique 50-case sequence")
    for row in runtime:
        if set(row) != EXPECTED_RUNTIME_KEYS:
            raise ValueError(f"runtime schema drifted for {row.get('case_id')}")
        contract = (row.get("dataset"), row.get("protocol"), row.get("role"), int(row.get("fps", -1)), int(row.get("num_frames", -1)))
        if contract != ("MVHuman", "MVH150", "test", FPS, FRAMES):
            raise ValueError(f"runtime contract drifted for {row.get('case_id')}")
    for runtime_row, evaluator_row in zip(runtime, evaluator):
        if evaluator_row.get("dataset") != "MVHuman" or evaluator_row.get("protocol") != "MVH150" or evaluator_row.get("role") != "test":
            raise ValueError(f"evaluator contract drifted for {evaluator_row.get('case_id')}")
        if evaluator_row.get("cut_indices_evaluator_only") != [CUT] or len(evaluator_row.get("source_frame_indices", [])) != FRAMES:
            raise ValueError(f"evaluator timeline drifted for {evaluator_row.get('case_id')}")
        if runtime_row["case_id"] != evaluator_row["case_id"]:
            raise ValueError("runtime/evaluator row order drifted")
    require_hash(runtime_path, ledger.get("runtime_manifest_sha256"), "runtime manifest")
    require_hash(evaluator_path, ledger.get("evaluator_manifest_sha256"), "evaluator manifest")
    require_hash(freeze_path, ledger.get("protocol_freeze_sha256"), "protocol freeze")
    if int(ledger.get("videos", -1)) != EXPECTED_CASES:
        raise ValueError("materialization ledger does not bind exactly 50 videos")
    return {
        "root": root,
        "runtime_path": runtime_path,
        "evaluator_path": evaluator_path,
        "ledger_path": ledger_path,
        "freeze_path": freeze_path,
        "runtime": runtime,
        "evaluator": evaluator,
        "runtime_sha256": sha256(runtime_path),
        "evaluator_sha256": sha256(evaluator_path),
    }


def selected_contract(mode: str, runtime: list[dict[str, Any]]) -> tuple[str, list[int], list[str]]:
    if mode == "prompthmr-formal":
        lines = list(range(1, EXPECTED_CASES + 1)); method = PROMPT_METHOD
    else:
        lines = list(PILOT_LINES); method = GVHMR_METHOD
    return method, lines, [str(runtime[line - 1]["case_id"]) for line in lines]


def artifact_names(root: Path, pattern: str, suffix: str) -> set[str]:
    return {path.name[: -len(suffix)] for path in root.glob(pattern) if path.is_file()}


def inventory(result_root: Path, expected: set[str]) -> dict[str, Any]:
    observed = {
        "cache": artifact_names(result_root / "predictions", "*.npz", ".npz"),
        "runtime_report": artifact_names(result_root / "predictions", "*.runtime.json", ".runtime.json"),
        "adapter": artifact_names(result_root / "predictions", "*.adapter.json", ".adapter.json"),
        "metric": artifact_names(result_root / "metrics", "*.json", ".json"),
        "log": artifact_names(result_root / "logs", "*.log", ".log"),
        "official": {path.name for path in (result_root / "official").iterdir()} if (result_root / "official").is_dir() else set(),
    }
    details = {
        name: {"count": len(values), "missing": sorted(expected - values), "extra": sorted(values - expected)}
        for name, values in observed.items()
    }
    expected_prediction_files = {
        *(f"{case_id}.npz" for case_id in expected),
        *(f"{case_id}.runtime.json" for case_id in expected),
        *(f"{case_id}.adapter.json" for case_id in expected),
    }
    unexpected_files = {
        "predictions": sorted(
            path.name for path in (result_root / "predictions").iterdir()
            if path.is_file() and path.name not in expected_prediction_files
        ) if (result_root / "predictions").is_dir() else [],
        "metrics": sorted(
            path.name for path in (result_root / "metrics").iterdir()
            if path.is_file() and path.name not in {f"{case_id}.json" for case_id in expected}
        ) if (result_root / "metrics").is_dir() else [],
        "official_root_files": sorted(
            path.name for path in (result_root / "official").iterdir() if path.is_file()
        ) if (result_root / "official").is_dir() else [],
    }
    complete = all(not item["missing"] and not item["extra"] for item in details.values()) and not any(unexpected_files.values())
    return {"complete": complete, "artifacts": details, "unexpected_files": unexpected_files}


def verify_run_summary(path: Path, mode: str, lines: list[int], case_ids: list[str]) -> dict[str, Any]:
    value = read_json(path)
    expected_method, expected_pilot = ("prompthmr", False) if mode == "prompthmr-formal" else ("gvhmr", True)
    if value.get("method") != expected_method or value.get("pilot") is not expected_pilot:
        raise ValueError("external run summary method/pilot contract drifted")
    if value.get("predeclared_lines") != lines or int(value.get("case_count", -1)) != len(lines):
        raise ValueError("external run summary selected a different fixed case set")
    outcomes = value.get("outcomes")
    if not isinstance(outcomes, list) or len(outcomes) != len(lines):
        raise ValueError("external run summary does not contain one outcome per fixed case")
    by_line = {int(row.get("line", -1)): row for row in outcomes if isinstance(row, dict)}
    if sorted(by_line) != lines:
        raise ValueError("external run summary outcome lines are incomplete or duplicated")
    for line, case_id in zip(lines, case_ids):
        row = by_line[line]
        if row.get("case_id") != case_id or row.get("status") != "ok":
            raise ValueError(f"external run summary is not successful for fixed line {line}/{case_id}")
    if int(value.get("ok", -1)) != len(lines) or value.get("failed") != []:
        raise ValueError("external run summary is not fail-closed complete")
    return {"path": str(path.resolve()), "sha256": sha256(path.resolve())}


def verify_gvhmr_pilot_outcomes(
    *, path: Path, lines: list[int], case_ids: list[str],
    protocol_info: dict[str, Any], result_root: Path,
) -> dict[str, Any]:
    """Seal both successful recoveries and native-inference failures.

    The first pilot runner used the old AIST-only converter, so every original
    outcome is marked failed. Ten cases nevertheless contain complete native
    official results and were subsequently converted by the narrow recovery
    utility. Two cases failed inside official SimpleVO before a native result
    existed. A valid availability audit must bind both outcomes instead of
    either fabricating artifacts for the latter or rejecting the failed gate
    before it can be recorded.
    """
    value = read_json(path)
    if value.get("method") != "gvhmr" or value.get("pilot") is not True:
        raise ValueError("GVHMR pilot summary method/pilot contract drifted")
    if value.get("predeclared_lines") != lines or int(value.get("case_count", -1)) != len(lines):
        raise ValueError("GVHMR pilot summary selected a different fixed case set")
    outcomes = value.get("outcomes")
    if not isinstance(outcomes, list) or len(outcomes) != len(lines):
        raise ValueError("GVHMR pilot summary is incomplete")
    by_line = {int(row.get("line", -1)): row for row in outcomes if isinstance(row, dict)}
    if sorted(by_line) != lines:
        raise ValueError("GVHMR pilot outcome lines are incomplete or duplicated")
    observed_ok = sum(row.get("status") == "ok" for row in outcomes)
    observed_failed = [row for row in outcomes if row.get("status") != "ok"]
    recorded_failed = value.get("failed")
    if not isinstance(recorded_failed, list):
        raise ValueError("GVHMR pilot summary has no failed-outcome list")
    failed_by_line = sorted(observed_failed, key=lambda item: int(item.get("line", -1)))
    recorded_by_line = sorted(recorded_failed, key=lambda item: int(item.get("line", -1)) if isinstance(item, dict) else -1)
    if int(value.get("ok", -1)) != observed_ok or recorded_by_line != failed_by_line:
        raise ValueError("GVHMR pilot summary counters do not match its fixed outcomes")

    sealed, successful, unavailable = [], [], []
    for line, case_id in zip(lines, case_ids):
        row = by_line[line]
        if row.get("case_id") != case_id:
            raise ValueError(f"GVHMR pilot case binding failed at line {line}")
        log_path = result_root / "logs" / f"{case_id}.log"
        if not log_path.is_file():
            raise FileNotFoundError(log_path)
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        official_dir = result_root / "official" / case_id
        if case_id not in log_text or str(official_dir) not in log_text:
            raise ValueError(f"GVHMR failure log is not bound to {case_id}")
        video = safe_video(protocol_info["root"], protocol_info["runtime"][line - 1])
        native = official_dir / video.stem / "hmr4d_results.pt"
        artifacts = {
            "cache": result_root / "predictions" / f"{case_id}.npz",
            "runtime_report": result_root / "predictions" / f"{case_id}.runtime.json",
            "adapter": result_root / "predictions" / f"{case_id}.adapter.json",
            "metric": result_root / "metrics" / f"{case_id}.json",
        }
        recovery_path = result_root / "recovery" / f"{case_id}.json"
        complete_recovery = native.is_file() and recovery_path.is_file() and all(item.is_file() for item in artifacts.values())
        if complete_recovery:
            recovery = read_json(recovery_path)
            verify_content_hash(recovery, f"GVHMR recovery record {case_id}")
            if (
                recovery.get("method") != GVHMR_METHOD
                or recovery.get("case_id") != case_id
                or int(recovery.get("manifest_line", -1)) != line
                or Path(str(recovery.get("native_result", ""))).resolve() != native.resolve()
                or recovery.get("native_result_sha256") != sha256(native)
                or Path(str(recovery.get("failure_log", ""))).resolve() != log_path.resolve()
                or recovery.get("failure_log_sha256") != sha256(log_path)
            ):
                raise ValueError(f"GVHMR recovery provenance mismatch for {case_id}")
            recovery_artifacts = recovery.get("artifacts", {})
            expected_hashes = {
                "cache_sha256": sha256(artifacts["cache"]),
                "runtime_report_sha256": sha256(artifacts["runtime_report"]),
                "adapter_sha256": sha256(artifacts["adapter"]),
                "metric_sha256": sha256(artifacts["metric"]),
            }
            if recovery_artifacts != expected_hashes:
                raise ValueError(f"GVHMR recovery artifact hashes mismatch for {case_id}")
            item = {
                "line": line, "case_id": case_id, "status": "recovered_native_and_evaluated",
                "original_runner_status": row.get("status"),
                "native_result": str(native.resolve()), "native_result_sha256": sha256(native),
                "failure_log": str(log_path.resolve()), "failure_log_sha256": sha256(log_path),
                "recovery_record": str(recovery_path.resolve()), "recovery_record_sha256": sha256(recovery_path),
            }
            successful.append(case_id)
        else:
            present = [str(item) for item in (*artifacts.values(), recovery_path) if item.exists()]
            if native.exists() or present:
                raise ValueError(f"GVHMR pilot has a partial/unsealed recovery for {case_id}: {present}")
            signature = "AttributeError: 'NoneType' object has no attribute 'matrix'"
            if row.get("status") != "failed" or "simple_vo.py" not in log_text or signature not in log_text:
                raise ValueError(f"GVHMR native-unavailable outcome is not the documented SimpleVO failure for {case_id}")
            item = {
                "line": line, "case_id": case_id, "status": "native_unavailable",
                "original_runner_status": row.get("status"),
                "native_result": str(native.resolve()), "native_result_absent": True,
                "failure_stage": "official SimpleVO before hmr4d_results.pt",
                "failure_signature": signature,
                "failure_log": str(log_path.resolve()), "failure_log_sha256": sha256(log_path),
            }
            unavailable.append(case_id)
        sealed.append(item)
    return {
        "summary": {"path": str(path.resolve()), "sha256": sha256(path.resolve())},
        "outcomes": sealed, "successful_case_ids": successful,
        "native_unavailable_case_ids": unavailable,
    }


def verify_video(path: Path) -> dict[str, Any]:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("formal video audit requires OpenCV") from error
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"video is not decodable: {path}")
    reported, fps, decoded = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))), float(capture.get(cv2.CAP_PROP_FPS)), 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        decoded += 1
    capture.release()
    if reported != FRAMES or decoded != FRAMES or abs(fps - FPS) > 1e-3:
        raise ValueError(f"video contract drifted: {path}, reported={reported}, decoded={decoded}, fps={fps}")
    return {"path": str(path), "sha256": sha256(path), "reported_frames": reported, "decoded_frames": decoded, "fps": fps}


def safe_video(protocol_root: Path, row: dict[str, Any]) -> Path:
    relative = Path(str(row["input_video"]))
    derived = (protocol_root / "derived").resolve()
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"unsafe input video for {row['case_id']}")
    path = (derived / relative).resolve()
    if derived not in path.parents or not path.is_file():
        raise FileNotFoundError(path)
    return path


def git_commit_exists(repo: Path, commit: Any) -> None:
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValueError("runtime report has no full Git commit")
    completed = subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode:
        raise ValueError(f"recorded external repository commit is unavailable: {commit}")


def summary_item(item: Any, *, available: bool, label: str) -> float | None:
    if not isinstance(item, dict):
        raise ValueError(f"{label} summary missing")
    count, mean = item.get("count"), item.get("mean")
    if available:
        if not isinstance(count, int) or count <= 0 or not isinstance(mean, (int, float)) or not math.isfinite(float(mean)):
            raise ValueError(f"{label} has no finite non-empty support")
        return float(mean)
    if count != 0 or mean is not None:
        raise ValueError(f"{label} must be explicitly unavailable (count=0, mean=null)")
    return None


def verify_rotation(cameras: np.ndarray, label: str) -> dict[str, float]:
    rotations = cameras[:, :3, :3]
    identity = np.eye(3)
    orthogonality = float(np.max(np.abs(rotations @ np.swapaxes(rotations, -1, -2) - identity)))
    determinant = float(np.max(np.abs(np.linalg.det(rotations) - 1.0)))
    bottom = float(np.max(np.abs(cameras[:, 3, :] - np.asarray([0.0, 0.0, 0.0, 1.0]))))
    if orthogonality >= 2e-3 or determinant >= 2e-3 or bottom >= 2e-4:
        raise ValueError(f"{label} cameras are not valid homogeneous SE(3)")
    return {"orthogonality_max": orthogonality, "determinant_error_max": determinant, "bottom_row_error_max": bottom}


def verify_schema_layer(adapter: dict[str, Any], method: str) -> dict[str, str]:
    expected_schema = PROMPT_ADAPTER_SCHEMA if method == PROMPT_METHOD else GVHMR_ADAPTER_SCHEMA
    if adapter.get("schema_version") != expected_schema:
        raise ValueError(f"{method} adapter did not use the strict MVHuman schema layer")
    layer = adapter.get("schema_layer")
    if not isinstance(layer, dict) or layer.get("numerical_conversion_changed") is not False:
        raise ValueError(f"{method} adapter schema-layer provenance is absent")
    base = Path(str(layer.get("base_converter", ""))).resolve()
    wrapper = Path(str(layer.get("mvhuman_wrapper", ""))).resolve()
    require_hash(base, layer.get("base_converter_sha256"), f"{method} base converter")
    require_hash(wrapper, layer.get("mvhuman_wrapper_sha256"), f"{method} MVHuman converter wrapper")
    expected_wrapper = HERE / ("convert_prompthmr_result.py" if method == PROMPT_METHOD else "convert_gvhmr_result.py")
    if wrapper != expected_wrapper.resolve():
        raise ValueError(f"{method} used an unexpected MVHuman converter wrapper")
    expected_base = (
        HERE.parents[3] / "external_baselines" / "bridge3r_eval" / "convert_prompthmr_result.py"
        if method == PROMPT_METHOD
        else HERE.parents[1] / "v21" / "aist_singleperson" / "convert_gvhmr_result.py"
    )
    if base != expected_base.resolve():
        raise ValueError(f"{method} used an unexpected frozen base converter")
    return {"base_converter": str(base), "base_converter_sha256": sha256(base), "wrapper": str(wrapper), "wrapper_sha256": sha256(wrapper)}


def inspect_case(
    *, method: str, line: int, runtime_row: dict[str, Any], evaluator_row: dict[str, Any],
    protocol_info: dict[str, Any], result_root: Path, repo: Path,
) -> dict[str, Any]:
    case_id = str(runtime_row["case_id"])
    cache_path = result_root / "predictions" / f"{case_id}.npz"
    report_path = result_root / "predictions" / f"{case_id}.runtime.json"
    adapter_path = result_root / "predictions" / f"{case_id}.adapter.json"
    metric_path = result_root / "metrics" / f"{case_id}.json"
    report, adapter, metric = read_json(report_path), read_json(adapter_path), read_json(metric_path)
    verify_content_hash(report, f"runtime report {case_id}")
    verify_content_hash(metric, f"evaluator report {case_id}")
    allowed_runtime = {PROMPT_RUNTIME_SCHEMA} if method == PROMPT_METHOD else GVHMR_RUNTIME_SCHEMAS
    if report.get("schema_version") not in allowed_runtime or report.get("record") != runtime_row or report.get("methods") != [method]:
        raise ValueError(f"runtime report protocol/case/method mismatch for {case_id}")
    if report.get("runtime_gt_access") is not False:
        raise ValueError(f"runtime GT-access contract failed for {case_id}")
    runtime = report.get("runtime", {})
    video_path = safe_video(protocol_info["root"], runtime_row)
    video = verify_video(video_path)
    if Path(str(runtime.get("input_video", ""))).resolve() != video_path or runtime.get("input_video_sha256") != video["sha256"]:
        raise ValueError(f"runtime input-video binding failed for {case_id}")
    expected_native = result_root / "official" / case_id
    expected_native = expected_native / ("results.pkl" if method == PROMPT_METHOD else f"{video_path.stem}/hmr4d_results.pt")
    native = Path(str(runtime.get("native_result", ""))).resolve()
    if native != expected_native.resolve():
        raise ValueError(f"native result path/case binding failed for {case_id}")
    native_sha = require_hash(native, runtime.get("native_result_sha256"), f"native result {case_id}")
    converter = Path(str(runtime.get("adapter_converter", ""))).resolve()
    expected_converter = HERE / ("convert_prompthmr_result.py" if method == PROMPT_METHOD else "convert_gvhmr_result.py")
    if converter != expected_converter.resolve():
        raise ValueError(f"runtime converter path mismatch for {case_id}")
    require_hash(converter, runtime.get("adapter_converter_sha256"), f"runtime converter {case_id}")
    if Path(str(runtime.get("adapter_metadata", ""))).resolve() != adapter_path.resolve():
        raise ValueError(f"runtime adapter path mismatch for {case_id}")
    adapter_sha = require_hash(adapter_path, runtime.get("adapter_metadata_sha256"), f"adapter metadata {case_id}")
    schema_layer = verify_schema_layer(adapter, method)
    if adapter.get("case_id") != case_id or adapter.get("protocol") != "MVH150" or adapter.get("runtime_gt_access") is not False:
        raise ValueError(f"adapter protocol/provenance mismatch for {case_id}")
    adapter_inputs = adapter.get("inputs", {})
    adapter_manifest = adapter_inputs.get("manifest", adapter_inputs.get("runtime_manifest"))
    if Path(str(adapter_manifest)).resolve() != protocol_info["runtime_path"].resolve():
        raise ValueError(f"adapter runtime-manifest path mismatch for {case_id}")
    if adapter_inputs.get("manifest_sha256", adapter_inputs.get("runtime_manifest_sha256")) != protocol_info["runtime_sha256"]:
        raise ValueError(f"adapter runtime-manifest hash mismatch for {case_id}")
    if int(adapter_inputs.get("manifest_line", adapter_inputs.get("runtime_line", -1))) != line:
        raise ValueError(f"adapter runtime line mismatch for {case_id}")
    adapter_native = adapter_inputs.get("result", adapter_inputs.get("native_result"))
    adapter_native_hash = adapter_inputs.get("result_sha256", adapter_inputs.get("native_result_sha256"))
    if Path(str(adapter_native)).resolve() != native or adapter_native_hash != native_sha:
        raise ValueError(f"adapter/native-result binding mismatch for {case_id}")

    metric_inputs = metric.get("inputs", {})
    if metric.get("schema_version") != REPORT_SCHEMA or metric.get("case_id") != case_id or metric.get("protocol") != "MVH150":
        raise ValueError(f"evaluator schema/case mismatch for {case_id}")
    if metric.get("subject") != evaluator_row.get("subject") or metric.get("angle_stratum") != evaluator_row.get("angle_stratum"):
        raise ValueError(f"evaluator case metadata mismatch for {case_id}")
    if metric.get("errors") != {} or set(metric.get("methods", {})) != {method}:
        raise ValueError(f"evaluator did not complete exactly {method} for {case_id}")
    if Path(str(metric_inputs.get("cache", ""))).resolve() != cache_path.resolve():
        raise ValueError(f"evaluator cache path mismatch for {case_id}")
    if Path(str(metric_inputs.get("runtime_report", ""))).resolve() != report_path.resolve():
        raise ValueError(f"evaluator runtime-report path mismatch for {case_id}")
    if Path(str(metric_inputs.get("evaluator_manifest", ""))).resolve() != protocol_info["evaluator_path"].resolve():
        raise ValueError(f"evaluator manifest path mismatch for {case_id}")
    cache_sha = require_hash(cache_path, metric_inputs.get("cache_sha256"), f"cache {case_id}")
    require_hash(report_path, metric_inputs.get("runtime_report_sha256"), f"runtime report evaluator binding {case_id}")
    if metric_inputs.get("evaluator_manifest_sha256") != protocol_info["evaluator_sha256"]:
        raise ValueError(f"evaluator manifest hash mismatch for {case_id}")
    if adapter.get("output_sha256") != cache_sha:
        raise ValueError(f"adapter/cache output hash mismatch for {case_id}")
    if Path(str(adapter.get("output", ""))).resolve() != cache_path.resolve():
        raise ValueError(f"adapter/cache output path mismatch for {case_id}")

    method_row = metric["methods"][method]
    if method_row.get("method") != method or method_row.get("status") != "ok" or set(method_row.get("metrics", {})) != set(METRICS):
        raise ValueError(f"metric method contract drifted for {case_id}")
    coverage = method_row.get("coverage", {})
    coverage_value = coverage.get("valid_frame_coverage")
    completion = coverage.get("completion")
    if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0 for value in (coverage_value, completion)):
        raise ValueError(f"coverage/completion invalid for {case_id}")
    track_id = method_row.get("track", {}).get("chosen_id")
    if not isinstance(track_id, int):
        raise ValueError(f"native persistent track was not selected for {case_id}")
    metric_values = {
        name: summary_item(method_row["metrics"][name], available=(method == PROMPT_METHOD or name in HUMAN_METRICS), label=f"{case_id}/{name}")
        for name in METRICS
    }

    with np.load(cache_path, allow_pickle=False) as archive:
        prefix = method + "__"
        keys = tuple(prefix + name for name in ("cameras_c2w", "joints_world", "persistent_ids", "valid"))
        if any(key not in archive.files for key in keys):
            raise KeyError(f"cache schema incomplete for {case_id}")
        cameras = np.asarray(archive[keys[0]], dtype=np.float64)
        joints = np.asarray(archive[keys[1]], dtype=np.float64)
        ids = np.asarray(archive[keys[2]], dtype=np.int64)
        valid = np.asarray(archive[keys[3]], dtype=bool)
    if cameras.shape != (FRAMES, 4, 4) or joints.ndim != 4 or joints.shape[0] != FRAMES or joints.shape[2:] != (24, 3) or ids.shape != valid.shape or ids.shape != joints.shape[:2]:
        raise ValueError(f"cache array shape drifted for {case_id}")
    selected = np.any(valid & (ids == track_id), axis=1)
    pre, post = int(selected[: CUT + 1].sum()), int(selected[CUT + 1 :].sum())
    if pre < 3 or post < 3 or not np.isfinite(joints[selected]).all():
        raise ValueError(f"cross-cut native track support/geometry failed for {case_id}: pre={pre}, post={post}")
    tracker_record: dict[str, Any] | None = None
    prompt_native_record: dict[str, Any] | None = None
    if method == PROMPT_METHOD:
        if (
            runtime.get("offline") is not True
            or runtime.get("static_cam") is not False
            or runtime.get("cut_or_calibration_supplied") is not False
            or runtime.get("recovered_without_reinference") is not False
        ):
            raise ValueError(f"official PromptHMR runtime contract drifted for {case_id}")
        pipeline_exception = runtime.get("pipeline_exception_after_result_write")
        if pipeline_exception is not None:
            expected_signature = "SMPLX_neutral_array_f32_slim.npz does not exist"
            traceback_text = pipeline_exception.get("traceback", "") if isinstance(pipeline_exception, dict) else ""
            if (
                not isinstance(pipeline_exception, dict)
                or pipeline_exception.get("type") != "PanicException"
                or expected_signature not in str(pipeline_exception.get("message", ""))
                or "export_scene_with_camera" not in traceback_text
                or "convert_mcs_to_gltf" not in traceback_text
            ):
                raise ValueError(f"PromptHMR raised an unapproved inference/export exception for {case_id}")
        if not np.isfinite(cameras).all():
            raise ValueError(f"PromptHMR camera is unavailable/non-finite for {case_id}")
        camera_audit: dict[str, Any] = verify_rotation(cameras, case_id)
        try:
            import joblib
        except ImportError as error:
            raise RuntimeError("formal PromptHMR native audit requires joblib") from error
        native_payload = joblib.load(native)
        spec = native_payload.get("spec_calib") if isinstance(native_payload, dict) else None
        first = spec.get("first_frame") if isinstance(spec, dict) else None
        spec_keys = ("vfov", "f_pix", "pitch", "roll")
        if not isinstance(first, dict) or any(
            not isinstance(first.get(key), (int, float, np.integer, np.floating))
            or isinstance(first.get(key), (bool, np.bool_))
            or not math.isfinite(float(first[key]))
            for key in spec_keys
        ):
            raise ValueError(f"official PromptHMR native result has no finite SPEC calibration for {case_id}")
        prompt_native_record = {
            "spec_calib_present": True,
            "first_frame_fields": {key: float(first[key]) for key in spec_keys},
            "contract": "unchanged official PromptHMR pipeline executed bundled SPEC calibration",
            "post_result_exception": (
                None if pipeline_exception is None else {
                    "type": pipeline_exception["type"],
                    "stage": "visualization-only GLB export after native results.pkl write",
                    "evaluator_inputs_affected": False,
                }
            ),
        }
    else:
        if not np.isnan(cameras).all():
            raise ValueError(f"GVHMR camera must be wholly and explicitly unavailable for {case_id}")
        camera_audit = {"available": False, "contract": "all cache entries NaN; both camera metrics count=0, mean=null"}
        tracker = adapter.get("tracker_audit")
        if not isinstance(tracker, dict) or tracker.get("available") is not True or tracker.get("audit_only") is not True:
            raise ValueError(f"raw official tracker audit unavailable for {case_id}")
        frames = tracker.get("raw_selected_frame_indices")
        if not isinstance(frames, list) or not frames or not all(isinstance(frame, int) and 0 <= frame < FRAMES for frame in frames):
            raise ValueError(f"raw tracker frame list invalid for {case_id}")
        if len(frames) != len(set(frames)) or frames != sorted(frames):
            raise ValueError(f"raw tracker frames are duplicated or unsorted for {case_id}")
        raw_pre = sum(frame <= CUT for frame in frames); raw_post = sum(frame > CUT for frame in frames)
        if (
            tracker.get("selected_track_id") is None
            or int(tracker.get("raw_selected_frame_count", -1)) != len(frames)
            or int(tracker.get("raw_selected_pre_cut_frame_count", -1)) != raw_pre
            or int(tracker.get("raw_selected_post_cut_frame_count", -1)) != raw_post
            or raw_pre < 3 or raw_post < 3
        ):
            raise ValueError(f"raw tracker lacks the pre-registered >=3-frame support on both cut sides for {case_id}")
        tracker_record = {
            "selected_track_id": int(tracker["selected_track_id"]), "raw_frames": len(frames),
            "raw_pre_cut_frames": raw_pre, "raw_post_cut_frames": raw_post,
        }

    git_key = "prompthmr_git" if method == PROMPT_METHOD else "gvhmr_git"
    provenance = report.get("provenance", {})
    git_value = provenance.get(git_key)
    if not isinstance(git_value, dict) or git_value.get("status_porcelain") != "":
        raise ValueError(f"external repository was not cleanly commit-bound for {case_id}")
    repo_key = "prompthmr_repo" if method == PROMPT_METHOD else "gvhmr_repo"
    if Path(str(provenance.get(repo_key, ""))).resolve() != repo.resolve():
        raise ValueError(f"external repository path mismatch for {case_id}")
    git_commit_exists(repo, git_value.get("commit"))
    license_record = None
    if method == PROMPT_METHOD:
        attestation = Path(str(provenance.get("license_attestation", ""))).resolve()
        attestation_sha = require_hash(attestation, provenance.get("license_attestation_sha256"), f"PromptHMR/SPEC license attestation {case_id}")
        license_record = {"path": str(attestation), "sha256": attestation_sha}
    return {
        "line": line, "case_id": case_id, "capture_id": str(evaluator_row["subject"]),
        "angle_stratum": str(evaluator_row["angle_stratum"]),
        "camera_rotation_geodesic_deg": float(evaluator_row["camera_rotation_geodesic_deg"]),
        "hashes": {
            "video": video["sha256"], "native": native_sha, "cache": cache_sha,
            "runtime_report": sha256(report_path), "adapter": adapter_sha, "metric": sha256(metric_path),
        },
        "video": video, "track": {"chosen_id": track_id, "pre_valid_frames": pre, "post_valid_frames": post},
        "raw_tracker": tracker_record, "camera": camera_audit,
        "prompt_native": prompt_native_record,
        "coverage": float(coverage_value), "completion": float(completion), "metrics": metric_values,
        "repo_git": git_value, "converter": schema_layer, "license_attestation": license_record,
        "runtime_report_path": str(report_path.resolve()),
    }


def verify_external_assets(method: str, repo: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    commits = {record["repo_git"]["commit"] for record in records}
    if len(commits) != 1:
        raise ValueError("external repository commit changed across fixed cases")
    if method == PROMPT_METHOD:
        manifest = repo / "PROMPTHMR_ASSET_MANIFEST.json"
        value = read_json(manifest)
        required = value.get("required_files")
        if not isinstance(required, list) or not required:
            raise ValueError("PromptHMR asset manifest has no required files")
        hashes = {}
        resolved_targets = {}
        for item in required:
            if not isinstance(item, dict):
                raise ValueError("malformed PromptHMR required asset row")
            relative = Path(str(item.get("path", "")))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("unsafe PromptHMR asset path")
            # Enforce repository containment on the manifest path itself.
            # Three licensed body-model entries are intentional repository
            # symlinks to the shared Movie3R model store; their resolved
            # targets are accepted only by the exact manifest SHA-256 below.
            lexical_path = (repo / relative).absolute()
            if repo.resolve() not in lexical_path.parents:
                raise ValueError("PromptHMR asset escapes repository")
            hashes[str(relative)] = require_hash(lexical_path, item.get("sha256"), f"PromptHMR asset {relative}")
            resolved_targets[str(relative)] = str(lexical_path.resolve())
        spec_path = repo / PROMPT_SPEC_CHECKPOINT
        spec_hash = require_hash(spec_path, PROMPT_SPEC_SHA256, "PromptHMR official SPEC checkpoint")
        spec_provenance_path = spec_path.with_suffix(spec_path.suffix + ".provenance.json")
        spec_provenance = read_json(spec_provenance_path)
        if (
            spec_provenance.get("artifact") != spec_path.name
            or spec_provenance.get("sha256") != spec_hash
            or int(spec_provenance.get("bytes", -1)) != spec_path.stat().st_size
            or spec_provenance.get("use") != "PromptHMR official SPEC camera-calibration path"
        ):
            raise ValueError("PromptHMR SPEC checkpoint provenance drifted")
        return {
            "repo": str(repo), "commit": next(iter(commits)),
            "asset_manifest": str(manifest), "asset_manifest_sha256": sha256(manifest),
            "required_asset_sha256": hashes, "required_asset_resolved_path": resolved_targets,
            "official_spec_checkpoint": {
                "path": str(spec_path.resolve()), "sha256": spec_hash,
                "bytes": spec_path.stat().st_size,
                "provenance": str(spec_provenance_path.resolve()),
                "provenance_sha256": sha256(spec_provenance_path),
            },
        }
    # The per-report checkpoint maps are checked directly below.  Keep this
    # branch explicit so GVHMR cannot inherit PromptHMR's manifest contract.
    checkpoint_root = repo / "inputs" / "checkpoints"
    first_map: dict[str, str] | None = None
    for record in records:
        report_path = Path(record["runtime_report_path"])
        report = read_json(report_path)
        current = report.get("provenance", {}).get("checkpoint_sha256")
        if not isinstance(current, dict) or set(current) != set(GVHMR_CHECKPOINTS):
            raise ValueError(f"GVHMR checkpoint map missing for {record['case_id']}")
        current = {str(key): str(value) for key, value in current.items()}
        if first_map is None:
            first_map = current
        elif current != first_map:
            raise ValueError("GVHMR checkpoint map changed across pilot cases")
    assert first_map is not None
    observed = {name: require_hash(checkpoint_root / name, first_map[name], f"GVHMR checkpoint {name}") for name in GVHMR_CHECKPOINTS}
    return {"repo": str(repo), "commit": next(iter(commits)), "checkpoint_sha256": observed}


def gt_metadata_hashes(audit_root: Path, evaluator_rows: list[dict[str, Any]]) -> dict[str, Any]:
    root = audit_root.resolve()
    by_subject: dict[str, set[Path]] = defaultdict(set)
    for row in evaluator_rows:
        subject = str(row["subject"]); subject_root = root / "metadata" / subject
        by_subject[subject].update((subject_root / "camera_scale.pkl", subject_root / "camera_extrinsics.json"))
        by_subject[subject].update(subject_root / "smplx" / "keypoints3d" / f"{int(index):06d}.json" for index in row["source_frame_indices"])
    result = {}
    for subject, paths in sorted(by_subject.items()):
        entries = {}
        for path in sorted(paths):
            if not path.is_file():
                raise FileNotFoundError(path)
            entries[str(path.relative_to(root))] = sha256(path)
        result[subject] = {
            "file_count": len(entries),
            "aggregate_sha256": hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
        }
    return result


def scalar_summary(values: list[float], expected: int) -> dict[str, Any]:
    if len(values) != expected or not np.isfinite(np.asarray(values, dtype=np.float64)).all():
        raise ValueError(f"fixed-denominator aggregate expected {expected} finite values, received {len(values)}")
    array = np.asarray(values, dtype=np.float64)
    return {"count": expected, "mean": float(array.mean()), "median": float(np.median(array)), "std": float(array.std())}


def prompt_aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    if len(records) != EXPECTED_CASES:
        raise ValueError("PromptHMR formal aggregation requires exactly 50 records")
    captures = sorted({record["capture_id"] for record in records})
    strata = sorted({record["angle_stratum"] for record in records})
    if len(captures) != 10 or any(sum(record["capture_id"] == capture for record in records) != 5 for capture in captures):
        raise ValueError("PromptHMR formal cases are not exactly 10 captures x 5 strata")

    def aggregate_field(field: str, selected: list[dict[str, Any]]) -> dict[str, Any]:
        case_values = [float(record["metrics"][field]) for record in selected]
        by_capture: dict[str, list[float]] = defaultdict(list)
        for record, value in zip(selected, case_values):
            by_capture[record["capture_id"]].append(value)
        capture_values = [float(np.mean(values)) for values in by_capture.values()]
        return {
            "case_macro": scalar_summary(case_values, len(selected)),
            "capture_macro": scalar_summary(capture_values, len(by_capture)),
        }

    return {
        "denominator_contract": "Each of the 50 frozen cases contributes exactly one finite within-case mean; no missing case or metric is dropped. Capture macro first averages the five strata within each held-out capture.",
        "metrics": {
            metric: {
                "overall": aggregate_field(metric, records),
                "by_angle_stratum": {
                    stratum: aggregate_field(metric, [record for record in records if record["angle_stratum"] == stratum])
                    for stratum in strata
                },
            }
            for metric in METRICS
        },
        "coverage": {
            "case_macro": scalar_summary([record["coverage"] for record in records], EXPECTED_CASES),
            "completion_case_macro": scalar_summary([record["completion"] for record in records], EXPECTED_CASES),
        },
    }


def build_tex(aggregate: dict[str, Any]) -> str:
    values = aggregate["metrics"]
    mean = lambda name, digits=1: f"{values[name]['overall']['capture_macro']['mean']:.{digits}f}"
    coverage = aggregate["coverage"]["case_macro"]["mean"] * 100.0
    return "\n".join([
        "% Auto-generated only after the fail-closed 50-case PromptHMR MVH150 audit.",
        "\\begin{tabular}{lrrrrrr}", "\\toprule",
        "Method & PA-MPJPE $\\downarrow$ & Anchor-MPJPE $\\downarrow$ & Seam-root $\\downarrow$ & Seam-orient. $\\downarrow$ & Cam. rot. $\\downarrow$ & Coverage $\\uparrow$ \\\\",
        "\\midrule",
        f"PromptHMR (official, offline) & {mean('pa_mpjpe_body12_mm')} & {mean('first_shot_anchor_mpjpe_body12_mm')} & {mean('seam_root_excess_mm')} & {mean('seam_orientation_excess_deg')} & {mean('post_camera_relative_rotation_deg')} & {coverage:.1f} \\\\",
        "\\bottomrule", "\\end{tabular}", "",
    ])


def main() -> None:
    args = parse_args()
    if args.diagnose and (args.output is not None or args.tex_output is not None):
        raise ValueError("--diagnose never writes; omit --output/--tex-output")
    if not args.diagnose and args.output is None:
        raise ValueError("formal audit requires a fresh --output")
    if args.mode == "gvhmr-pilot" and args.tex_output is not None:
        raise ValueError("GVHMR pilot gate deliberately emits no performance table")
    protocol_info = protocol(args)
    method, lines, case_ids = selected_contract(args.mode, protocol_info["runtime"])
    result_root, repo = args.result_root.resolve(), args.repo.resolve()
    if not result_root.is_dir() or not repo.is_dir():
        raise FileNotFoundError("result root or external repository is missing")
    status = inventory(result_root, set(case_ids))
    diagnostic = {
        "mode": args.mode, "formal_approved": False, "diagnostic_only": True,
        "expected_lines": lines, "expected_case_ids": case_ids, "inventory": status,
        "note": "Inventory completeness alone is never an approval; rerun without --diagnose for all hashes and semantic predicates.",
    }
    if args.diagnose:
        print(json.dumps(diagnostic, ensure_ascii=False, indent=2)); return
    if method == PROMPT_METHOD and not status["complete"]:
        raise RuntimeError("fixed-case inventory is incomplete or contains extras:\n" + json.dumps(status, indent=2))
    gvhmr_outcomes = None
    if method == PROMPT_METHOD:
        run_summary = verify_run_summary(result_root / "run_summary.json", args.mode, lines, case_ids)
        inspect_ids = set(case_ids)
    else:
        gvhmr_outcomes = verify_gvhmr_pilot_outcomes(
            path=result_root / "run_summary.json", lines=lines, case_ids=case_ids,
            protocol_info=protocol_info, result_root=result_root,
        )
        run_summary = gvhmr_outcomes["summary"]
        inspect_ids = set(gvhmr_outcomes["successful_case_ids"])
    evaluator_by_id = {str(row["case_id"]): row for row in protocol_info["evaluator"]}
    records = []
    for line, case_id in zip(lines, case_ids):
        if case_id not in inspect_ids:
            continue
        record = inspect_case(
            method=method, line=line, runtime_row=protocol_info["runtime"][line - 1],
            evaluator_row=evaluator_by_id[case_id], protocol_info=protocol_info,
            result_root=result_root, repo=repo,
        )
        record["runtime_report_path"] = str(result_root / "predictions" / f"{case_id}.runtime.json")
        records.append(record)
    assets = verify_external_assets(method, repo, records)
    selected_evaluator = [evaluator_by_id[case_id] for case_id in case_ids]
    provenance = {
        "runtime_manifest": str(protocol_info["runtime_path"]), "runtime_manifest_sha256": protocol_info["runtime_sha256"],
        "evaluator_manifest": str(protocol_info["evaluator_path"]), "evaluator_manifest_sha256": protocol_info["evaluator_sha256"],
        "materialization_ledger": str(protocol_info["ledger_path"]), "materialization_ledger_sha256": sha256(protocol_info["ledger_path"]),
        "protocol_freeze": str(protocol_info["freeze_path"]), "protocol_freeze_sha256": sha256(protocol_info["freeze_path"]),
        "run_summary": run_summary, "external_assets": assets,
        "evaluator_metadata_aggregate_sha256": gt_metadata_hashes(args.audit_root, selected_evaluator),
        "code_sha256": {
            str(path): sha256(path)
            for path in (
                Path(__file__).resolve(), HERE / "run_external_protocol.py", HERE / "run_prompthmr_case.py",
                HERE / "run_gvhmr_case.py", HERE / "convert_prompthmr_result.py",
                HERE / "convert_gvhmr_result.py", HERE / "recover_external_native.py", HERE / "evaluate_case.py",
                HERE.parents[1] / "v21" / "aist_singleperson" / "evaluate_aist.py",
            )
        },
    }
    base = {
        "schema_version": "Bridge3R-MVHuman-PromptHMR-formal-ledger-v1" if method == PROMPT_METHOD else "Bridge3R-MVHuman-GVHMR-pilot-gate-v1",
        "mode": args.mode, "method": method, "fixed_lines": lines, "fixed_case_ids": case_ids,
        "case_count": len(case_ids), "verified_record_count": len(records),
        "records": records, "provenance": provenance,
    }
    if method == PROMPT_METHOD:
        base["formal_approved"] = True
        base["aggregate"] = prompt_aggregate(records)
        base["approval_contract"] = "exactly 50/50 fixed cases; every metric finite; full artifact and provenance chain verified"
    else:
        assert gvhmr_outcomes is not None
        gate_approved = not gvhmr_outcomes["native_unavailable_case_ids"] and len(records) == len(case_ids)
        base["pilot_outcomes"] = gvhmr_outcomes["outcomes"]
        base["verified_success_count"] = len(records)
        base["native_unavailable_count"] = len(gvhmr_outcomes["native_unavailable_case_ids"])
        base["test_gate"] = {
            "approved": gate_approved,
            "rule": "All 12 pre-registered cases must have a native result, valid converted/evaluated artifacts, one evaluator-native track with >=3 frames on each side, and the independently repeated official raw tracker must support >=3 frames on each side. GVHMR camera quantities must be explicitly unavailable.",
            "accuracy_outcome_independent": True,
            "admission_signals": "pre-registered artifact and cross-cut availability predicates only",
            "pose_or_camera_accuracy_values_consulted_for_admission": False,
            "full_test_authorization": "forbidden" if not gate_approved else "authorized only after verifying this exact gate JSON and content SHA-256",
        }
        # Accuracy values are intentionally removed from a gate whose decision
        # must be independent of result quality.  Hashes still bind the sealed
        # evaluator reports for later audit.
        for record in base["records"]:
            record.pop("metrics", None)
    base["content_sha256"] = canonical_digest(base)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"formal audit refuses to overwrite {output}")
    if args.tex_output is not None and args.tex_output.resolve().exists():
        raise FileExistsError(f"formal audit refuses to overwrite {args.tex_output.resolve()}")
    atomic_json(output, base)
    if args.tex_output is not None:
        tex_path = args.tex_output.resolve(); tex_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = tex_path.with_suffix(tex_path.suffix + ".partial")
        temporary.write_text(build_tex(base["aggregate"]), encoding="utf-8"); os.replace(temporary, tex_path)
    approved = base.get("formal_approved", base.get("test_gate", {}).get("approved", False))
    print(json.dumps({"output": str(output), "mode": args.mode, "cases": len(records), "approved": approved, "content_sha256": base["content_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
