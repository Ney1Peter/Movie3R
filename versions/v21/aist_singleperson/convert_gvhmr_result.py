#!/usr/bin/env python3
"""Convert one official GVHMR demo result into the frozen AIST CS150 cache.

This adapter is deliberately prediction-only.  It receives exactly one
compact runtime-manifest row and the native ``hmr4d_results.pt`` written by
the unchanged GVHMR demo.  It never opens an evaluator manifest, official
calibration, view identity, cut annotation, or 3-D ground truth.

GVHMR exposes a gravity-view human trajectory, but no AIST-compatible metric
camera trajectory.  The cache therefore encodes its human prediction and
marks every camera transform as unavailable (NaN), rather than inventing a
camera translation or quietly reporting a non-comparable camera score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCHEMA = "Bridge3R-AIST-GVHMR-adapter-v1"
EXPECTED_KEYS = {"case_id", "dataset", "protocol", "role", "input_video", "fps", "num_frames"}
EXPECTED_FRAMES, EXPECTED_FPS = 150, 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True, help="native GVHMR hmr4d_results.pt")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--line", type=int, required=True)
    parser.add_argument("--derived-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True, help="GVHMR checkout")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--method", default="gvhmr_official")
    parser.add_argument("--device", default="cuda:0", help="logical CUDA device visible to this subprocess")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--audit-raw-tracker", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_runtime(path: Path, line_number: int) -> dict[str, Any]:
    rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if line_number < 1 or line_number > len(rows):
        raise IndexError(f"--line {line_number} is outside {len(rows)} runtime rows")
    value = json.loads(rows[line_number - 1])
    if not isinstance(value, dict) or set(value) != EXPECTED_KEYS:
        raise ValueError("runtime schema drifted or evaluator-only data leaked")
    if value.get("dataset") != "AIST++" or value.get("protocol") != "CS150":
        raise ValueError("only AIST++ CS150 compact runtime rows are accepted")
    if int(value.get("num_frames", -1)) != EXPECTED_FRAMES or int(value.get("fps", -1)) != EXPECTED_FPS:
        raise ValueError("AIST CS150 temporal contract drifted")
    return value


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
    import subprocess

    def query(*command: str) -> str | None:
        completed = subprocess.run(["git", *command], cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return completed.stdout.strip() if completed.returncode == 0 else None

    return {"commit": query("rev-parse", "HEAD"), "status_porcelain": query("status", "--porcelain")}


def as_finite_tensor(value: Any, *, name: str, frames: int, width: int, device: torch.device) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    value = value.detach().cpu().float()
    if value.ndim == 1 and width == 1:
        value = value[:, None]
    if value.ndim != 2 or value.shape[1] != width:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected (F,{width})")
    if value.shape[0] == 1:
        value = value.expand(frames, -1)
    if value.shape[0] != frames:
        raise ValueError(f"{name} has {value.shape[0]} frames, expected {frames}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return value.to(device)


def tracker_audit(repo: Path, video: Path) -> dict[str, Any]:
    """Record raw YOLO support without changing the official demo's inference.

    The official demo writes only a smoothed/interpolated one-track box file.
    This repeat of its own ``Tracker.track`` call is an audit-only operation:
    it has no access to labels and its output is never supplied back to GVHMR.
    """
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from hmr4d.utils.preproc import Tracker  # noqa: PLC0415

    tracker = Tracker()
    try:
        history = tracker.track(str(video))
        frame_ids, boxes, order = Tracker.sort_track_length(history, str(video))
    finally:
        del tracker
        torch.cuda.empty_cache()
    if not order:
        return {"available": True, "raw_track_count": 0, "selected_track_id": None, "reason": "official_tracker_found_no_human_track"}
    selected = int(order[0])
    frames = [int(frame) for frame in frame_ids[selected]]
    if any(frame < 0 or frame >= EXPECTED_FRAMES for frame in frames):
        raise ValueError("official tracker emitted a raw frame outside CS150 timeline")
    areas = np.asarray(boxes[selected], dtype=np.float64)
    if areas.ndim != 2 or areas.shape[1] != 4 or len(areas) != len(frames) or not np.isfinite(areas).all():
        raise ValueError("official tracker raw selected boxes are malformed")
    widths, heights = areas[:, 2] - areas[:, 0], areas[:, 3] - areas[:, 1]
    if np.any(widths <= 0) or np.any(heights <= 0):
        raise ValueError("official tracker raw selected boxes have non-positive area")
    pre = sum(frame <= 74 for frame in frames)
    post = sum(frame >= 75 for frame in frames)
    return {
        "available": True,
        "audit_only": True,
        "selected_track_policy": "unchanged official Tracker.sort_track_length: largest summed normalized box area",
        "raw_track_count": int(len(order)),
        "selected_track_id": selected,
        "raw_selected_frame_count": int(len(frames)),
        "raw_selected_pre_cut_frame_count": int(pre),
        "raw_selected_post_cut_frame_count": int(post),
        "raw_selected_pre_cut_coverage": float(pre / 75.0),
        "raw_selected_post_cut_coverage": float(post / 75.0),
        "raw_selected_frame_indices": frames,
        "warning": "GVHMR demo itself consumes an interpolated-and-smoothed version of the selected raw track; this audit does not alter inference or repair cross-cut identity.",
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not args.device.startswith("cuda:") or not torch.cuda.is_available():
        raise RuntimeError("GVHMR conversion requires an explicit CUDA device")
    repo, result, output, metadata = args.repo.resolve(), args.result.resolve(), args.output.resolve(), args.metadata_output.resolve()
    if not repo.is_dir() or not result.is_file():
        raise FileNotFoundError("GVHMR checkout or native hmr4d_results.pt is missing")
    if output.exists() or metadata.exists():
        raise FileExistsError("GVHMR converter refuses to overwrite an evaluator cache or metadata")
    runtime = read_runtime(args.manifest.resolve(), int(args.line))
    video = safe_video(args.derived_root.resolve(), str(runtime["input_video"]))
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    torch.cuda.set_device(int(args.device.split(":", 1)[1]))
    device = torch.device(args.device)
    from hmr4d.utils.body_model.smplx_lite import SmplxLiteSmplN24  # noqa: PLC0415

    # ``weights_only=False`` is required because the official native artifact
    # stores a nested Python dictionary.  It is produced locally by the
    # immediately preceding official demo and is never loaded from the web.
    try:
        native = torch.load(result, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch releases before the keyword existed.
        native = torch.load(result, map_location="cpu")
    if not isinstance(native, dict) or not isinstance(native.get("smpl_params_global"), dict):
        raise ValueError("native GVHMR result lacks smpl_params_global")
    params = native["smpl_params_global"]
    fields = {
        "body_pose": as_finite_tensor(params.get("body_pose"), name="smpl_params_global.body_pose", frames=EXPECTED_FRAMES, width=63, device=device),
        "betas": as_finite_tensor(params.get("betas"), name="smpl_params_global.betas", frames=EXPECTED_FRAMES, width=10, device=device),
        "global_orient": as_finite_tensor(params.get("global_orient"), name="smpl_params_global.global_orient", frames=EXPECTED_FRAMES, width=3, device=device),
        "transl": as_finite_tensor(params.get("transl"), name="smpl_params_global.transl", frames=EXPECTED_FRAMES, width=3, device=device),
    }
    model = SmplxLiteSmplN24().to(device).eval()
    joint_chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, EXPECTED_FRAMES, int(args.batch_size)):
            stop = min(start + int(args.batch_size), EXPECTED_FRAMES)
            output_joints = model(**{key: value[start:stop] for key, value in fields.items()})
            joint_chunks.append(output_joints.detach().cpu().numpy().astype(np.float32))
    del model, fields
    torch.cuda.empty_cache()
    joints = np.concatenate(joint_chunks, axis=0)
    if joints.shape != (EXPECTED_FRAMES, 24, 3) or not np.isfinite(joints).all():
        raise ValueError(f"GVHMR SMPL-X-to-SMPL24 conversion is invalid: {joints.shape}")
    # GVHMR's public demo is intentionally single-track.  ID 0 denotes that
    # one native output slot; it must not be interpreted as an external GT ID.
    joints_cache = joints[:, None]
    valid = np.ones((EXPECTED_FRAMES, 1), dtype=bool)
    persistent_ids = np.zeros((EXPECTED_FRAMES, 1), dtype=np.int64)
    cameras = np.full((EXPECTED_FRAMES, 4, 4), np.nan, dtype=np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    np.savez_compressed(
        temporary,
        **{
            f"{args.method}__cameras_c2w": cameras,
            f"{args.method}__joints_world": joints_cache,
            f"{args.method}__persistent_ids": persistent_ids,
            f"{args.method}__valid": valid,
        },
    )
    # numpy adds .npz when its path does not already carry that suffix.
    temporary_npz = temporary if temporary.exists() else Path(str(temporary) + ".npz")
    os.replace(temporary_npz, output)
    tracker = {"available": False, "reason": "raw_tracker_audit_not_requested"}
    if args.audit_raw_tracker:
        tracker = tracker_audit(repo, video)
    payload = {
        "schema_version": SCHEMA,
        "case_id": runtime["case_id"],
        "protocol": runtime["protocol"],
        "runtime_gt_access": False,
        "inputs": {
            "native_result": str(result), "native_result_sha256": sha256(result),
            "runtime_manifest": str(args.manifest.resolve()), "runtime_manifest_sha256": sha256(args.manifest.resolve()),
            "runtime_line": int(args.line), "input_video": str(video), "input_video_sha256": sha256(video),
            "gvhmr_repo": str(repo), "gvhmr_git": external_git(repo),
            "adapter": str(Path(__file__).resolve()), "adapter_sha256": sha256(Path(__file__).resolve()),
        },
        "coordinate_contract": {
            "human": "GVHMR official smpl_params_global gravity-view global gauge; evaluator may fit one first-shot Sim(3), then holds it fixed after the RGB view cut.",
            "joints": "official GVHMR SmplxLiteSmplN24 conversion: SMPL-X vertices mapped to neutral SMPL and regressed to 24 joints.",
            "camera": "not exported: GVHMR demo does not provide an AIST-comparable metric camera-to-world trajectory; all cameras_c2w entries are NaN and camera metrics must remain unavailable.",
            "persistent_id": "single official GVHMR output slot encoded as ID 0; no GT identity selection or cross-cut re-association is performed.",
        },
        "tracker_audit": tracker,
        "summary": {
            "method": args.method, "frame_count": EXPECTED_FRAMES,
            "valid_person_frames": EXPECTED_FRAMES,
            "single_output_slot": True,
            "camera_metrics_available": False,
            "joint_topology": "neutral-SMPL24 via GVHMR SmplxLiteSmplN24",
        },
        "output": str(output), "output_sha256": sha256(output),
    }
    atomic_json(metadata, payload)
    print(json.dumps({"output": str(output), "metadata": str(metadata), "case_id": runtime["case_id"], "tracker_audit": tracker.get("available")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
