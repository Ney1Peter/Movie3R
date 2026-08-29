#!/usr/bin/env python3
"""Recover one completed external native result without re-running inference.

This utility is intentionally narrow.  It accepts only a frozen MVH150 Test
row, refuses to overwrite any converted/evaluated artifact, verifies that an
immutable failure log belongs to the case, and requires the native official
artifact to already exist.  SimpleVO failures that produced no native GVHMR
result therefore cannot be "recovered" with this path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PROMPT_RUNNER = HERE / "run_prompthmr_case.py"
PROMPT_CONVERTER = HERE / "convert_prompthmr_result.py"
GVHMR_CONVERTER = HERE / "convert_gvhmr_result.py"
EVALUATOR = HERE / "evaluate_case.py"
EXPECTED_KEYS = {"case_id", "dataset", "protocol", "role", "input_video", "fps", "num_frames"}
FRAMES, FPS = 150, 30
PROMPT_METHOD, GVHMR_METHOD = "prompthmr_official", "gvhmr_official"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("prompthmr", "gvhmr"), required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--evaluator-manifest", type=Path, required=True)
    parser.add_argument("--line", type=int, required=True)
    parser.add_argument("--derived-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--failure-log", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", dest="python_executable", type=Path, required=True)
    parser.add_argument("--physical-device", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--license-attestation", type=Path, help="Required for PromptHMR recovery")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def runtime_row(path: Path, line: int) -> dict[str, Any]:
    rows = [json.loads(value) for value in path.read_text(encoding="utf-8").splitlines() if value.strip()]
    if len(rows) != 50 or line < 1 or line > len(rows):
        raise ValueError("recovery requires one of exactly 50 frozen MVH150 runtime rows")
    row = rows[line - 1]
    if not isinstance(row, dict) or set(row) != EXPECTED_KEYS:
        raise ValueError("MVH150 runtime schema drifted")
    if (row.get("dataset"), row.get("protocol"), row.get("role"), int(row.get("fps", -1)), int(row.get("num_frames", -1))) != (
        "MVHuman", "MVH150", "test", FPS, FRAMES
    ):
        raise ValueError("recovery row is outside the frozen MVH150 Test contract")
    return row


def safe_video(root: Path, relative: str) -> Path:
    item = Path(relative)
    root = root.resolve()
    if item.is_absolute() or not item.parts or ".." in item.parts:
        raise ValueError("unsafe runtime video path")
    path = (root / item).resolve()
    if root not in path.parents or not path.is_file():
        raise FileNotFoundError(path)
    return path


def git_state(repo: Path) -> dict[str, str]:
    values = {}
    for name, command in (("commit", ("rev-parse", "HEAD")), ("status_porcelain", ("status", "--porcelain"))):
        completed = subprocess.run(["git", *command], cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if completed.returncode:
            raise RuntimeError(f"cannot record {repo} git {name}: {completed.stderr}")
        values[name] = completed.stdout.strip()
    return values


def gvhmr_checkpoint_hashes(repo: Path) -> dict[str, str]:
    names = (
        "gvhmr/gvhmr_siga24_release.ckpt", "hmr2/epoch=10-step=25000.ckpt",
        "vitpose/vitpose-h-multi-coco.pth", "yolo/yolov8x.pt", "dpvo/dpvo.pth",
    )
    root = repo / "inputs" / "checkpoints"
    missing = [str(root / name) for name in names if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"GVHMR checkpoint contract incomplete: {missing}")
    return {name: sha256(root / name) for name in names}


def require_fresh(paths: tuple[Path, ...]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"native recovery refuses to overwrite artifacts: {existing}")


def run(command: list[str], *, cwd: Path, env: dict[str, str], label: str) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode:
        raise RuntimeError(f"{label} failed (returncode={completed.returncode}):\n{completed.stderr}")
    return completed


def main() -> None:
    args = parse_args()
    if args.line < 1 or args.batch_size <= 0 or args.physical_device < 0:
        raise ValueError("line, batch-size, and physical-device are invalid")
    runtime_path = args.runtime_manifest.resolve()
    evaluator_path = args.evaluator_manifest.resolve()
    output_root, repo = args.output_root.resolve(), args.repo.resolve()
    failure_log = args.failure_log.resolve()
    python_executable = args.python_executable.absolute()
    if not all(path.is_file() for path in (runtime_path, evaluator_path, failure_log, python_executable)) or not repo.is_dir():
        raise FileNotFoundError("runtime/evaluator manifest, failure log, interpreter, or repository is missing")
    row = runtime_row(runtime_path, int(args.line))
    case_id = str(row["case_id"])
    video = safe_video(args.derived_root.resolve(), str(row["input_video"]))
    log_text = failure_log.read_text(encoding="utf-8", errors="replace")
    official_dir = output_root / "official" / case_id
    if case_id not in log_text or str(official_dir) not in log_text:
        raise ValueError("failure log is not textually bound to the requested case/output directory")
    cache = output_root / "predictions" / f"{case_id}.npz"
    report = output_root / "predictions" / f"{case_id}.runtime.json"
    adapter = output_root / "predictions" / f"{case_id}.adapter.json"
    metric = output_root / "metrics" / f"{case_id}.json"
    recovery_record = output_root / "recovery" / f"{case_id}.json"
    require_fresh((cache, report, adapter, metric, recovery_record))
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.physical_device)
    started = time.perf_counter()

    if args.method == "prompthmr":
        if args.license_attestation is None or not args.license_attestation.resolve().is_file():
            raise FileNotFoundError("PromptHMR recovery requires --license-attestation")
        native = official_dir / "results.pkl"
        if not native.is_file():
            raise FileNotFoundError(f"PromptHMR inference did not leave a recoverable native result: {native}")
        command = [
            str(python_executable), str(PROMPT_RUNNER), "--runtime-manifest", str(runtime_path),
            "--line", str(args.line), "--derived-root", str(args.derived_root.resolve()),
            "--repo", str(repo), "--license-attestation", str(args.license_attestation.resolve()),
            "--output-dir", str(official_dir), "--cache-output", str(cache),
            "--runtime-report", str(report), "--adapter-metadata", str(adapter),
            "--converter", str(PROMPT_CONVERTER), "--python", str(python_executable),
            "--device", "cuda:0", "--batch-size", str(args.batch_size),
            "--recover-native-result", str(native), "--recovery-log", str(failure_log),
        ]
        converted = run(command, cwd=HERE.parents[2], env=environment, label="PromptHMR native recovery")
        method = PROMPT_METHOD
    else:
        native = official_dir / video.stem / "hmr4d_results.pt"
        if not native.is_file():
            raise FileNotFoundError(
                "GVHMR official inference left no native result (e.g. a SimpleVO pre-inference failure); "
                f"recovery is forbidden: {native}"
            )
        command = [
            str(python_executable), str(GVHMR_CONVERTER), "--result", str(native),
            "--manifest", str(runtime_path), "--line", str(args.line),
            "--derived-root", str(args.derived_root.resolve()), "--repo", str(repo),
            "--output", str(cache), "--metadata-output", str(adapter),
            "--method", GVHMR_METHOD, "--device", "cuda:0", "--batch-size", str(args.batch_size),
            "--audit-raw-tracker",
        ]
        converted = run(command, cwd=repo, env=environment, label="GVHMR native conversion recovery")
        payload: dict[str, Any] = {
            "schema_version": "Bridge3R-MVHuman-GVHMR-official-native-recovery-v1",
            "record": row, "methods": [GVHMR_METHOD], "runtime_gt_access": False,
            "runtime": {
                "method": "GVHMR official unchanged whole-video demo; recovered conversion only",
                "offline": True, "static_cam": False, "use_dpvo": False, "f_mm": None,
                "cut_or_calibration_supplied": False, "input_video": str(video),
                "input_video_sha256": sha256(video), "output_dir": str(official_dir),
                "native_result": str(native), "native_result_sha256": sha256(native),
                "recovered_without_reinference": True,
                "original_failure_log": str(failure_log), "original_failure_log_sha256": sha256(failure_log),
                "adapter_converter": str(GVHMR_CONVERTER), "adapter_converter_sha256": sha256(GVHMR_CONVERTER),
                "adapter_metadata": str(adapter), "adapter_metadata_sha256": sha256(adapter),
                "conversion_command": command, "conversion_stdout": converted.stdout,
                "conversion_stderr": converted.stderr, "physical_cuda_device": int(args.physical_device),
                "logical_cuda_device": "cuda:0", "seconds": time.perf_counter() - started,
            },
            "provenance": {
                "gvhmr_repo": str(repo), "gvhmr_git": git_state(repo),
                "checkpoint_sha256": gvhmr_checkpoint_hashes(repo),
                "license": "GVHMR repository LICENSE: educational, research and non-profit use; no commercial use.",
                "python": sys.version, "platform": platform.platform(),
            },
        }
        payload["content_sha256"] = canonical_digest(payload)
        atomic_json(report, payload)
        method = GVHMR_METHOD

    evaluation = [
        str(python_executable), str(EVALUATOR), "--cache", str(cache),
        "--runtime-report", str(report), "--evaluator-manifest", str(evaluator_path),
        "--case-id", case_id, "--audit-root", str(args.audit_root.resolve()), "--output", str(metric),
    ]
    evaluated = run(evaluation, cwd=HERE.parents[2], env=environment, label="MVH150 evaluator")
    recovery = {
        "schema_version": "Bridge3R-MVHuman-external-native-recovery-record-v1",
        "method": method, "case_id": case_id, "manifest_line": int(args.line),
        "native_result": str(native), "native_result_sha256": sha256(native),
        "failure_log": str(failure_log), "failure_log_sha256": sha256(failure_log),
        "converter_command": command, "evaluator_command": evaluation,
        "artifacts": {
            "cache_sha256": sha256(cache), "runtime_report_sha256": sha256(report),
            "adapter_sha256": sha256(adapter), "metric_sha256": sha256(metric),
        },
        "seconds": time.perf_counter() - started,
        "contract": "native result pre-existed this process; conversion/evaluation only; no inference rerun and no overwrite",
        "stdout": {"converter": converted.stdout, "evaluator": evaluated.stdout},
    }
    recovery["content_sha256"] = canonical_digest(recovery)
    atomic_json(recovery_record, recovery)
    print(json.dumps({"case_id": case_id, "method": method, "recovered": True, "record": str(recovery_record)}, indent=2))


if __name__ == "__main__":
    main()
