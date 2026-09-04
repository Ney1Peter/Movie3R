#!/usr/bin/env python3
"""Invoke frozen RGB-only OnlineHMR inference for one manifest row."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


SCHEMA = "Bridge3R-OnlineHMR-native-runtime-v1"
SCRIPT = Path(__file__).resolve()
WORKSPACE = SCRIPT.parents[4]
DEFAULT_REPO = WORKSPACE / "external_baselines/Video-OnlineHMR"
DEFAULT_PYTHON = (
    WORKSPACE
    / "external_baselines/.venvs/onlinehmr-py311-pt25-cu118/bin/python"
)
WEIGHT_MANIFEST = DEFAULT_REPO / "data/asset_manifests/onlinehmr_20260903/manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_row(path: Path, line_number: int) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if line_number < 1 or line_number > len(rows):
        raise IndexError(f"line {line_number} outside manifest with {len(rows)} rows")
    row = rows[line_number - 1]
    if row.get("runtime_gt_access") is not False:
        raise ValueError("runtime row is not prediction-only")
    return row


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def git_commit(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def validate_images(image_dir: Path, frame_count: int) -> None:
    images = sorted(image_dir.glob("*.jpg"))
    names = [f"{index:06d}.jpg" for index in range(frame_count)]
    if [path.name for path in images] != names:
        raise ValueError(
            f"staged JPEG names/count differ from exact 0..{frame_count - 1} timeline"
        )
    if any(path.stat().st_size <= 0 for path in images):
        raise ValueError("staged input contains an empty JPEG")


def validate_trajectory(path: Path, frame_count: int) -> tuple[bool, str | None]:
    """Lightweight validation before declaring a native output usable."""

    try:
        rows = [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != frame_count - 1 or any(len(row) != 8 for row in rows):
            return False, "invalid_camera_trajectory_shape"
        values = [[float(token) for token in row] for row in rows]
        if any(not math.isfinite(token) for row in values for token in row):
            return False, "nonfinite_camera_trajectory"
        if any(row[0] <= 0 for row in values):
            return False, "nonpositive_camera_scale"
    except (OSError, UnicodeError, ValueError):
        return False, "unreadable_camera_trajectory"
    return True, None


def remove_temporary_case(path: Path, parent: Path) -> int:
    """Remove only one named, reproducible OnlineHMR intermediate directory."""

    if not path.exists():
        return 0
    resolved, root = path.resolve(), parent.resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"unsafe temporary cleanup target: {resolved}")
    bytes_before = sum(item.stat().st_size for item in resolved.rglob("*") if item.is_file())
    shutil.rmtree(resolved)
    return int(bytes_before)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--line", type=int, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument(
        "--stall-timeout-seconds", type=int, default=900,
        help="fail a case after this many seconds without camera-log progress",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = args.manifest.resolve()
    row = read_row(manifest, int(args.line))
    case_id = str(row["case_id"])
    frame_count = int(row["clip_length"])
    image_dir = args.image_dir.resolve()
    validate_images(image_dir, frame_count)
    repo = args.repo.resolve()
    # Keep the virtual-environment launcher path intact.  Resolving its symlink
    # to the base interpreter drops pyvenv.cfg discovery and therefore the
    # environment's installed packages.
    python = Path(os.path.abspath(args.python))
    for path in (repo / "scripts/run_custom_mt.py", python, WEIGHT_MANIFEST):
        if not path.is_file():
            raise FileNotFoundError(path)
    weight_manifest = json.loads(WEIGHT_MANIFEST.read_text(encoding="utf-8"))
    if weight_manifest.get("complete") is not True or weight_manifest.get("file_count") != 15:
        raise ValueError("OnlineHMR weight manifest is not complete")

    root = args.output_root.resolve() / f"line{int(args.line):03d}"
    native = root / "native"
    runtime_path = root / "onlinehmr.runtime.json"
    stdout_path = root / "onlinehmr.stdout.log"
    stderr_path = root / "onlinehmr.stderr.log"
    manifest_hash = sha256(manifest)
    commit = git_commit(repo)
    identity = {
        "schema_version": SCHEMA,
        "case_id": case_id,
        "manifest_sha256": manifest_hash,
        "manifest_line": int(args.line),
        "repo_commit": commit,
        "weight_manifest_sha256": sha256(WEIGHT_MANIFEST),
    }
    if runtime_path.is_file():
        previous = json.loads(runtime_path.read_text(encoding="utf-8"))
        if any(previous.get(key) != value for key, value in identity.items()):
            raise RuntimeError(f"existing OnlineHMR runtime identity differs: {runtime_path}")
        if previous.get("status") == "success":
            print(json.dumps({"status": "reused", "case_id": case_id, "runtime": str(runtime_path)}))
            return
        raise RuntimeError(
            f"a retained failed attempt exists at {runtime_path}; use a new attempt root"
        )
    root.mkdir(parents=True, exist_ok=True)
    native.mkdir(parents=True, exist_ok=False)

    temporary_parent = repo / "results"
    temporary_case = temporary_parent / case_id
    stale_intermediate_bytes = remove_temporary_case(temporary_case, temporary_parent)
    source_trajectory = repo / "logs" / f"{case_id}_images_incremental_all.txt"
    if source_trajectory.is_file():
        source_trajectory.unlink()

    command = [
        str(python), "scripts/run_custom_mt.py",
        "--image-dir", str(image_dir),
        "--save_dir", str(native),
        "--no-viz", "--no-render", "--depth-mask",
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(int(args.gpu)),
            "CUDA_HOME": "/usr/local/cuda-11.8",
            "PATH": "/usr/local/cuda-11.8/bin:" + environment.get("PATH", ""),
            "PYTHONUNBUFFERED": "1",
            # The upstream custom-video script contains a debugging breakpoint
            # before its documented invalid-box skip.  Disable interactive
            # debugging without changing the skip or any model computation.
            "PYTHONBREAKPOINT": "0",
        }
    )
    started = time.time()
    timed_out = False
    stalled = False
    last_progress = started
    last_trajectory_size = -1
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_handle:
        process = subprocess.Popen(
            command,
            cwd=repo,
            env=environment,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            start_new_session=True,
        )
        while process.poll() is None:
            now = time.time()
            if source_trajectory.is_file():
                size = source_trajectory.stat().st_size
                if size != last_trajectory_size:
                    last_trajectory_size = size
                    last_progress = now
            if now - started > int(args.timeout_seconds):
                timed_out = True
            elif now - last_progress > int(args.stall_timeout_seconds):
                stalled = True
            if timed_out or stalled:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=30)
                break
            time.sleep(5)
        returncode = int(process.returncode if process.returncode is not None else 124)
    seconds = time.time() - started

    frozen_trajectory = root / "camera_incremental_all.txt"
    if source_trajectory.is_file():
        shutil.copy2(source_trajectory, frozen_trajectory)
    track_files = sorted(native.glob(f"*_{case_id}.npz"))
    global_results = native / "global_results"
    pointclouds = sorted(global_results.glob("*.ply")) if global_results.is_dir() else []
    trajectory_rows = 0
    trajectory_valid = False
    trajectory_failure_reason = "missing_camera_trajectory"
    if frozen_trajectory.is_file():
        trajectory_rows = sum(
            bool(line.strip())
            for line in frozen_trajectory.read_text(encoding="utf-8").splitlines()
        )
        trajectory_valid, trajectory_failure_reason = validate_trajectory(
            frozen_trajectory, frame_count
        )
    success = bool(
        returncode == 0
        and frozen_trajectory.is_file()
        and trajectory_rows == frame_count - 1
        and trajectory_valid
        and pointclouds
    )
    cleanup_errors = []
    reclaimed_intermediate_bytes = 0
    try:
        reclaimed_intermediate_bytes = remove_temporary_case(
            temporary_case, temporary_parent
        )
    except Exception as error:  # cleanup never changes the prediction outcome
        cleanup_errors.append(f"{type(error).__name__}: {error}")
    try:
        if source_trajectory.is_file():
            source_trajectory.unlink()
    except Exception as error:
        cleanup_errors.append(f"{type(error).__name__}: {error}")
    payload = {
        **identity,
        "status": "success" if success else "failed",
        "failure_reason": None if success else (
            "wall_time_timeout" if timed_out else
            "camera_progress_stall_timeout" if stalled else
            trajectory_failure_reason if not trajectory_valid else
            "nonzero_exit_or_incomplete_native_outputs"
        ),
        "dataset": row.get("dataset"),
        "split": row.get("split"),
        "record": row,
        "manifest": str(manifest),
        "repo": str(repo),
        "weight_manifest": str(WEIGHT_MANIFEST),
        "command": command,
        "CUDA_VISIBLE_DEVICES": str(int(args.gpu)),
        "returncode": returncode,
        "timed_out": timed_out,
        "stalled": stalled,
        "stall_timeout_seconds": int(args.stall_timeout_seconds),
        "wall_time_seconds": seconds,
        "image_dir": str(image_dir),
        "native_root": str(native),
        "native_track_files": [
            {"path": str(path), "bytes": path.stat().st_size}
            for path in track_files
        ],
        "native_track_count": len(track_files),
        "camera_trajectory": str(frozen_trajectory) if frozen_trajectory.is_file() else None,
        "camera_trajectory_bytes": (
            frozen_trajectory.stat().st_size if frozen_trajectory.is_file() else None
        ),
        "camera_trajectory_rows": trajectory_rows,
        "camera_trajectory_valid": trajectory_valid,
        "scene_pointclouds": [
            {"path": str(path), "bytes": path.stat().st_size}
            for path in pointclouds
        ],
        "reproducible_intermediate_cleanup": {
            "temporary_case_root": str(temporary_case),
            "stale_bytes_removed_before_run": stale_intermediate_bytes,
            "bytes_removed_after_freeze": reclaimed_intermediate_bytes,
            "source_camera_log_removed_after_copy": not source_trajectory.exists(),
            "errors": cleanup_errors,
        },
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "stdout_bytes": stdout_path.stat().st_size,
        "stderr_bytes": stderr_path.stat().st_size,
        "input_contract": "exact ordered staged JPEGs; no video re-encoding",
        "runtime_gt_access": False,
        "gt_camera_used": False,
        "gt_identity_used": False,
        "gt_boxes_masks_depth_used": False,
        "cut_label_used": False,
    }
    atomic_json(runtime_path, payload)
    print(json.dumps({
        "status": payload["status"],
        "case_id": case_id,
        "wall_time_seconds": seconds,
        "native_track_count": len(track_files),
        "camera_trajectory_rows": trajectory_rows,
        "runtime": str(runtime_path),
    }, indent=2))
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
