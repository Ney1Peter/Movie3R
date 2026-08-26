#!/usr/bin/env python3
"""Run the unmodified official PromptHMR video pipeline on one AIST CS150 row.

The process deliberately receives *only* one compact runtime-manifest row.
It never opens the AIST evaluator manifest, labels, official calibration,
camera identifiers, or cut position.  PromptHMR therefore processes the
already-rendered 150-frame RGB video as one continuous video, exactly as its
official video entry point does.

The adapter conversion is run only after official inference has written its
``results.pkl``.  The conversion maintains PromptHMR native track IDs and
uses its world-coordinate output; evaluation is a separate process.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

try:
    from .protocol import DEFAULT_DERIVED_ROOT, atomic_json, canonical_json_digest
except ImportError:
    from protocol import DEFAULT_DERIVED_ROOT, atomic_json, canonical_json_digest  # type: ignore


SCHEMA = "Bridge3R-AIST-PromptHMR-official-runtime-v1"
EXPECTED_KEYS = {"case_id", "dataset", "protocol", "role", "input_video", "fps", "num_frames"}
EXPECTED_FPS, EXPECTED_FRAMES = 30, 150


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--line", type=int, default=1)
    parser.add_argument("--derived-root", type=Path, default=DEFAULT_DERIVED_ROOT)
    parser.add_argument("--repo", type=Path, required=True, help="PromptHMR checkout")
    parser.add_argument("--license-attestation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="fresh official Pipeline output directory")
    parser.add_argument("--cache-output", type=Path, required=True, help="fresh converted evaluator cache NPZ")
    parser.add_argument("--runtime-report", type=Path, required=True)
    parser.add_argument("--adapter-metadata", type=Path, required=True)
    parser.add_argument("--converter", type=Path, required=True)
    parser.add_argument("--python", dest="python_executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--recover-native-result", type=Path,
        help=("Existing official results.pkl written before a documented post-export failure. "
              "This skips inference, never overwrites the native artifact, and only runs conversion/reporting."),
    )
    parser.add_argument(
        "--recovery-log", type=Path,
        help="Required with --recover-native-result; immutable stdout/stderr evidence for the post-export failure.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_runtime(path: Path, line_number: int) -> dict[str, Any]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if line_number < 1 or line_number > len(lines):
        raise IndexError(f"--line {line_number} outside manifest with {len(lines)} rows")
    row = json.loads(lines[line_number - 1])
    if not isinstance(row, dict) or set(row) != EXPECTED_KEYS:
        raise ValueError("AIST runtime row schema drifted")
    if row["dataset"] != "AIST++" or row["protocol"] != "CS150":
        raise ValueError("this runner accepts only AIST++ CS150 rows")
    if int(row["fps"]) != EXPECTED_FPS or int(row["num_frames"]) != EXPECTED_FRAMES:
        raise ValueError("AIST CS150 temporal contract drifted")
    return row


def safe_video(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise ValueError(f"unsafe runtime video path: {relative!r}")
    root = root.resolve()
    path = (root / candidate).resolve()
    if root not in path.parents or not path.is_file():
        raise FileNotFoundError(path)
    return path


def external_git(repo: Path) -> dict[str, str | None]:
    def query(*command: str) -> str | None:
        completed = subprocess.run(["git", *command], cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return completed.stdout.strip() if completed.returncode == 0 else None
    return {"commit": query("rev-parse", "HEAD"), "status_porcelain": query("status", "--porcelain")}


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    runtime = read_runtime(args.runtime_manifest.resolve(), int(args.line))
    derived_root, repo = args.derived_root.resolve(), args.repo.resolve()
    video = safe_video(derived_root, str(runtime["input_video"]))
    attestation, converter = args.license_attestation.resolve(), args.converter.resolve()
    # Do not call Path.resolve() here: a virtualenv interpreter is commonly a
    # symlink to the system Python binary.  Resolving it would silently drop
    # the virtualenv's site-packages for the adapter subprocess.
    python_executable = args.python_executable.expanduser()
    if not python_executable.is_absolute():
        python_executable = (Path.cwd() / python_executable).absolute()
    output_dir, cache, report, metadata = (path.resolve() for path in (args.output_dir, args.cache_output, args.runtime_report, args.adapter_metadata))
    required = (repo, attestation, converter, python_executable)
    if not repo.is_dir() or any(not path.is_file() for path in required[1:]):
        raise FileNotFoundError("PromptHMR repository, interpreter, converter, or license attestation is missing")
    recovering = args.recover_native_result is not None
    recovered_result = args.recover_native_result.resolve() if recovering else None
    recovery_log = args.recovery_log.resolve() if args.recovery_log else None
    if recovering:
        if recovery_log is None or not recovery_log.is_file():
            raise ValueError("--recover-native-result requires an existing --recovery-log")
        if not recovered_result.is_file() or recovered_result.parent != output_dir:
            raise ValueError("recovered native result must be output-dir/results.pkl")
        if cache.exists() or report.exists() or metadata.exists():
            raise FileExistsError("recovery refuses to overwrite converted outputs or runtime report")
    elif output_dir.exists() or cache.exists() or report.exists() or metadata.exists():
        raise FileExistsError("PromptHMR AIST runner refuses to reuse or overwrite any output")
    if not torch.cuda.is_available() or not args.device.startswith("cuda:"):
        raise RuntimeError("official PromptHMR AIST inference requires an explicit CUDA device")
    device_index = int(args.device.split(":", 1)[1])
    torch.cuda.set_device(device_index)

    # PromptHMR was installed with audited, repository-local upstream caches.
    # Pin them here rather than falling back to a user's default Torch/HF
    # cache: formal inference must fail closed if a pinned dependency is
    # missing, never silently download a different version during a run.
    cache_root = repo / "data" / "cache"
    torch_home, hf_home = cache_root / "torch", cache_root / "huggingface"
    cache_required = (
        torch_home / "hub" / "pytorch_vision_v0.10.0" / "hubconf.py",
        torch_home / "hub" / "checkpoints" / "deeplabv3_resnet50_coco-cd0a2569.pth",
        hf_home / "hub",
    )
    if any(not path.exists() for path in cache_required):
        missing = [str(path) for path in cache_required if not path.exists()]
        raise FileNotFoundError(f"PromptHMR audited local cache is incomplete: {missing}")
    os.environ.update({
        "TORCH_HOME": str(torch_home),
        "HF_HOME": str(hf_home),
        "HF_HUB_CACHE": str(hf_home / "hub"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    })

    # The license gate is imported and checked before importing Pipeline,
    # because Pipeline imports bundled SPEC at module-import time.
    sys.path.insert(0, str((Path(__file__).resolve().parents[4] / "external_baselines" / "bridge3r_eval").resolve()))
    from prompthmr_license_gate import validate_attestation  # noqa: PLC0415
    license_payload = validate_attestation(attestation, repo)

    started = time.perf_counter()
    inference_exception: dict[str, str] | None = None
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    pipeline = None
    if recovering:
        result = recovered_result
        inference_exception = {
            "type": "recovered_native_result_after_post_export_failure",
            "message": "Native results.pkl predates this recovery process; see immutable recovery log.",
            "recovery_log": str(recovery_log),
            "recovery_log_sha256": sha256(recovery_log),
        }
    else:
        try:
            from pipeline import Pipeline  # noqa: PLC0415
            pipeline = Pipeline(static_cam=False)
            # Do not provide a cut, calibration, static-camera hint, or track.
            pipeline(str(video), str(output_dir), static_cam=False, save_only_essential=True)
        # pyo3_runtime.PanicException inherits BaseException in this release.
        # It is only recovered when the official pipeline has already written
        # the native result; otherwise the missing-result guard below fails.
        except BaseException as error:  # noqa: BLE001
            inference_exception = {
                "type": type(error).__name__, "message": str(error),
                "traceback": traceback.format_exc(),
            }
        finally:
            del pipeline
            gc.collect()
            torch.cuda.empty_cache()
        result = output_dir / "results.pkl"
    if not result.is_file():
        raise RuntimeError(
            "official PromptHMR pipeline did not produce results.pkl; "
            f"exception={None if inference_exception is None else inference_exception['type']}"
        )

    # An MCS export error can occur *after* the official pipeline has
    # atomically written its essential native result.  Preserve that error for
    # audit but do not discard a complete inference artifact because the MCS
    # visualizer asset is not an evaluator input.
    conversion_command = [
        str(python_executable), str(converter), "--result", str(result),
        "--manifest", str(args.runtime_manifest.resolve()), "--line", str(args.line),
        "--output", str(cache), "--metadata-output", str(metadata),
        "--method", "prompthmr_official", "--device", args.device,
        "--batch-size", str(args.batch_size),
    ]
    converted = subprocess.run(conversion_command, cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if converted.returncode:
        raise RuntimeError(f"PromptHMR adapter conversion failed:\n{converted.stderr}")
    elapsed = time.perf_counter() - started
    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "record": runtime,
        "methods": ["prompthmr_official"],
        "runtime": {
            "method": "PromptHMR official unchanged video pipeline",
            "offline": True,
            "static_cam": False,
            "cut_or_calibration_supplied": False,
            "input_video": str(video),
            "input_video_sha256": sha256(video),
            "output_dir": str(output_dir),
            "native_result": str(result),
            "native_result_sha256": sha256(result),
            "adapter_converter": str(converter),
            "adapter_converter_sha256": sha256(converter),
            "adapter_metadata": str(metadata),
            "adapter_metadata_sha256": sha256(metadata),
            "conversion": {"command": conversion_command, "stdout": converted.stdout, "stderr": converted.stderr},
            "pipeline_exception_after_result_write": inference_exception,
            "recovered_without_reinference": recovering,
            "seconds": elapsed,
            "cache_environment": {
                "TORCH_HOME": str(torch_home), "HF_HOME": str(hf_home),
                "HF_HUB_CACHE": str(hf_home / "hub"), "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            },
        },
        "provenance": {
            "prompthmr_repo": str(repo), "prompthmr_git": external_git(repo),
            "license_attestation": str(attestation), "license_attestation_sha256": sha256(attestation),
            "license_confirmation": {key: license_payload[key] for key in ("confirmed_by", "confirmed_at", "spec_authorization_basis")},
            "python": sys.version, "platform": platform.platform(),
            "torch": torch.__version__, "cuda": torch.version.cuda, "device": args.device,
        },
        "runtime_gt_access": False,
    }
    payload["content_sha256"] = canonical_json_digest(payload)
    atomic_json(report, payload)
    print(json.dumps({"cache": str(cache), "report": str(report), "case_id": runtime["case_id"], "seconds": elapsed}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
