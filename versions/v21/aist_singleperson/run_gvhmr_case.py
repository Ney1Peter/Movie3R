#!/usr/bin/env python3
"""Run unchanged official GVHMR on one prediction-only AIST CS150 runtime row."""

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

try:
    from .protocol import DEFAULT_DERIVED_ROOT, atomic_json, canonical_json_digest
except ImportError:
    from protocol import DEFAULT_DERIVED_ROOT, atomic_json, canonical_json_digest  # type: ignore


SCHEMA = "Bridge3R-AIST-GVHMR-official-runtime-v1"
EXPECTED_KEYS = {"case_id", "dataset", "protocol", "role", "input_video", "fps", "num_frames"}
EXPECTED_FPS, EXPECTED_FRAMES = 30, 150


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--line", type=int, required=True)
    parser.add_argument("--derived-root", type=Path, default=DEFAULT_DERIVED_ROOT)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="fresh parent for the official demo output")
    parser.add_argument("--cache-output", type=Path, required=True)
    parser.add_argument("--runtime-report", type=Path, required=True)
    parser.add_argument("--adapter-metadata", type=Path, required=True)
    parser.add_argument("--converter", type=Path, required=True)
    parser.add_argument("--python", dest="python_executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--physical-device", type=int, required=True, help="physical CUDA index, isolated through CUDA_VISIBLE_DEVICES")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_runtime(path: Path, line_number: int) -> dict[str, Any]:
    rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if line_number < 1 or line_number > len(rows):
        raise IndexError(f"--line {line_number} outside manifest with {len(rows)} rows")
    row = json.loads(rows[line_number - 1])
    if not isinstance(row, dict) or set(row) != EXPECTED_KEYS:
        raise ValueError("AIST runtime schema drifted")
    if row["dataset"] != "AIST++" or row["protocol"] != "CS150" or int(row["fps"]) != EXPECTED_FPS or int(row["num_frames"]) != EXPECTED_FRAMES:
        raise ValueError("this runner accepts only 150-frame AIST++ CS150 runtime rows")
    return row


def safe_video(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise ValueError(f"unsafe runtime video path: {relative!r}")
    root = root.resolve(); path = (root / candidate).resolve()
    if root not in path.parents or not path.is_file():
        raise FileNotFoundError(path)
    return path


def external_git(repo: Path) -> dict[str, str | None]:
    def query(*command: str) -> str | None:
        completed = subprocess.run(["git", *command], cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return completed.stdout.strip() if completed.returncode == 0 else None
    return {"commit": query("rev-parse", "HEAD"), "status_porcelain": query("status", "--porcelain")}


def checkpoint_hashes(repo: Path) -> dict[str, str]:
    names = (
        "gvhmr/gvhmr_siga24_release.ckpt", "hmr2/epoch=10-step=25000.ckpt",
        "vitpose/vitpose-h-multi-coco.pth", "yolo/yolov8x.pt", "dpvo/dpvo.pth",
    )
    root = repo / "inputs" / "checkpoints"
    missing = [str(root / name) for name in names if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"GVHMR checkpoint contract is incomplete: {missing}")
    return {name: sha256(root / name) for name in names}


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.physical_device < 0:
        raise ValueError("--batch-size must be positive and --physical-device nonnegative")
    runtime = read_runtime(args.runtime_manifest.resolve(), int(args.line))
    repo, derived = args.repo.resolve(), args.derived_root.resolve()
    video = safe_video(derived, str(runtime["input_video"]))
    # Preserve venv resolution: virtualenv binaries are symlinks on this host.
    python_executable = args.python_executable.expanduser()
    if not python_executable.is_absolute():
        python_executable = (Path.cwd() / python_executable).absolute()
    converter = args.converter.resolve()
    output_dir, cache, report, metadata = (item.resolve() for item in (args.output_dir, args.cache_output, args.runtime_report, args.adapter_metadata))
    if not repo.is_dir() or not python_executable.is_file() or not converter.is_file():
        raise FileNotFoundError("GVHMR repo, interpreter, or converter is missing")
    if output_dir.exists() or cache.exists() or report.exists() or metadata.exists():
        raise FileExistsError("GVHMR runner refuses to reuse or overwrite official, converted, or report outputs")
    checkpoints = checkpoint_hashes(repo)
    official_case = output_dir / video.stem
    native_result = official_case / "hmr4d_results.pt"
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.physical_device)
    # The unchanged official demo sees a single logical CUDA:0.  No cut,
    # calibration, focal length, static-camera flag, target person, or label
    # is supplied.
    command = [str(python_executable), "tools/demo/demo.py", "--video", str(video), "--output_root", str(output_dir)]
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment)
    if completed.returncode:
        raise RuntimeError(f"official GVHMR demo failed (returncode={completed.returncode}):\n{completed.stderr}")
    if not native_result.is_file():
        raise RuntimeError(f"official GVHMR demo returned success without native result: {native_result}")
    conversion = [
        str(python_executable), str(converter), "--result", str(native_result),
        "--manifest", str(args.runtime_manifest.resolve()), "--line", str(args.line),
        "--derived-root", str(derived), "--repo", str(repo), "--output", str(cache),
        "--metadata-output", str(metadata), "--method", "gvhmr_official", "--device", "cuda:0",
        "--batch-size", str(args.batch_size), "--audit-raw-tracker",
    ]
    converted = subprocess.run(conversion, cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment)
    if converted.returncode:
        raise RuntimeError(f"GVHMR adapter conversion failed:\n{converted.stderr}")
    elapsed = time.perf_counter() - started
    payload: dict[str, Any] = {
        "schema_version": SCHEMA, "record": runtime, "methods": ["gvhmr_official"], "runtime_gt_access": False,
        "runtime": {
            "method": "GVHMR official unchanged whole-video demo", "offline": True,
            "static_cam": False, "use_dpvo": False, "f_mm": None,
            "cut_or_calibration_supplied": False, "input_video": str(video), "input_video_sha256": sha256(video),
            "output_dir": str(output_dir), "native_result": str(native_result), "native_result_sha256": sha256(native_result),
            "official_command": command, "official_stdout": completed.stdout, "official_stderr": completed.stderr,
            "adapter_converter": str(converter), "adapter_converter_sha256": sha256(converter),
            "adapter_metadata": str(metadata), "adapter_metadata_sha256": sha256(metadata),
            "conversion_command": conversion, "conversion_stdout": converted.stdout, "conversion_stderr": converted.stderr,
            "physical_cuda_device": int(args.physical_device), "logical_cuda_device": "cuda:0", "seconds": elapsed,
        },
        "provenance": {
            "gvhmr_repo": str(repo), "gvhmr_git": external_git(repo), "checkpoint_sha256": checkpoints,
            "license": "GVHMR repository LICENSE: educational, research and non-profit use; no commercial use.",
            "python": sys.version, "platform": platform.platform(),
        },
    }
    payload["content_sha256"] = canonical_json_digest(payload)
    atomic_json(report, payload)
    print(json.dumps({"cache": str(cache), "report": str(report), "case_id": runtime["case_id"], "seconds": elapsed}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
