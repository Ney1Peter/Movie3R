#!/usr/bin/env python3
"""Export the frozen C1-EMA25 correction in the original Human3R demo format.

The source is the CPU-rendered 5-pre + 25-post three-person B0+BRTC payload.
Its BRTC geometry is required to be numerically equal to the runtime-first
within-shot cache before C1 residuals are applied.  Thus this exporter changes
only post-shot person vertices; RGB, depth, confidence, camera, pre frames,
B0 and BRTC remain exact source artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
SOURCE = (
    REPO_ROOT / "output/v14/brtc_lc_smallspan_three_30frame_demo_payload/"
    "three_t1100_c4_c5_k0/b0_brtc_lc"
)
CACHE = REPO_ROOT / "output/v14/within_shot_stability/cache/three_t1100_c4_c5_pre5_post25.npz"
OUTPUT = (
    REPO_ROOT / "output/v14/within_shot_stability/demo_payload/"
    "three_t1100_c4_c5_k0/b0_brtc_lc_c1_ema25"
)
POLICY = REPO_ROOT / "versions/v14/frozen/WITHIN_SHOT_STATIC_GATE_EMA25_V1_20260804.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--policy", type=Path, default=POLICY)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def inside_repo(path: Path) -> Path:
    resolved = path.resolve()
    if resolved != REPO_ROOT and REPO_ROOT not in resolved.parents:
        raise ValueError(f"Path must stay under Movie3R: {resolved}")
    return resolved


def c1_residuals(cache_path: Path, policy_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    # Import only the GT-free runtime path.  The selected function reads no
    # evaluator target, label or identity field from this cache dictionary.
    from versions.v14.eval_streaming_within_shot_stability import POLICIES, runtime_c1

    policy_payload = json.loads(policy_path.read_text(encoding="utf-8"))
    name = str(policy_payload["policy"]["name"])
    policy = next((item for item in POLICIES if item.name == name), None)
    if policy is None:
        raise KeyError(f"Frozen policy {name} is absent from runtime implementation")
    with np.load(cache_path, allow_pickle=False) as payload:
        runtime = {
            "b0_cameras": np.asarray(payload["b0_cameras"], dtype=np.float64),
            "b0_roots": np.asarray(payload["b0_roots"], dtype=np.float64),
            "b0_joints": np.asarray(payload["b0_joints"], dtype=np.float64),
            "brtc_shifts_by_track": np.asarray(payload["brtc_shifts_by_track"], dtype=np.float64),
            "native_ids": np.asarray(payload["native_ids"], dtype=np.int64),
            "b0_vertices": np.asarray(payload["b0_vertices"], dtype=np.float64),
        }
    outcome = runtime_c1(runtime, policy)
    return np.asarray(outcome["residuals"], dtype=np.float64), {
        "policy": policy_payload,
        "gates": np.asarray(outcome["gates"], dtype=np.int8),
        "reasons": np.asarray(outcome["reasons"]),
        "camera_max_abs_change": float(outcome["camera_max_abs_change"]),
        "runtime": runtime,
    }


def reset_destination(destination: Path, overwrite: bool) -> None:
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"Destination exists: {destination}; pass --overwrite")
        if destination == REPO_ROOT or REPO_ROOT not in destination.parents:
            raise ValueError(f"Unsafe overwrite target: {destination}")
        shutil.rmtree(destination)
    destination.mkdir(parents=True)


def hardlink_source(source: Path, destination: Path, cut_index: int) -> None:
    for item in sorted(source.rglob("*")):
        relative = item.relative_to(source)
        output = destination / relative
        if item.is_dir():
            output.mkdir(exist_ok=True)
            continue
        if relative.parent == Path("smpl") and int(relative.stem) >= int(cut_index):
            # These are written as distinct files below; do not hardlink then overwrite.
            continue
        os.link(item, output)


def write_post_smpl(
    source: Path, destination: Path, residuals: np.ndarray, native_ids: np.ndarray,
    b0_vertices: np.ndarray, shifts: np.ndarray, cut_index: int,
) -> float:
    max_base_error = 0.0
    for post_index in range(len(residuals)):
        frame = int(cut_index) + post_index
        source_path = source / "smpl" / f"{frame:06d}.npz"
        destination_path = destination / "smpl" / f"{frame:06d}.npz"
        with np.load(source_path, allow_pickle=False) as payload:
            values = {key: payload[key] for key in payload.files}
        ids = np.asarray(values["smpl_id"], dtype=np.int64)
        if not np.array_equal(ids, native_ids[post_index]):
            raise RuntimeError(f"Native IDs disagree at post frame {post_index}: {ids} vs {native_ids[post_index]}")
        expected = b0_vertices[post_index] + shifts[:, None, :]
        current = np.asarray(values["verts_world"], dtype=np.float64)
        error = float(np.max(np.abs(current - expected)))
        max_base_error = max(max_base_error, error)
        if error > 1e-4:
            raise RuntimeError(f"Source BRTC payload/cache mismatch at post {post_index}: {error}")
        values["verts_world"] = (current + residuals[post_index, :, None, :]).astype(np.float32)
        np.savez(destination_path, **values)
    return max_base_error


def main() -> None:
    args = parse_args()
    source, cache, policy, destination = (inside_repo(path) for path in (args.source, args.cache, args.policy, args.output))
    if not source.is_dir() or not cache.is_file() or not policy.is_file():
        raise FileNotFoundError({"source": source, "cache": cache, "policy": policy})
    source_frames = sorted((source / "camera").glob("*.npz"))
    if len(source_frames) != 30:
        raise ValueError(f"Expected exactly 30 source frames, got {len(source_frames)}")
    cut_index, post_count = 5, 25
    residuals, debug = c1_residuals(cache, policy)
    runtime = debug["runtime"]
    if residuals.shape != (post_count, 3, 3):
        raise ValueError(f"Unexpected residual shape {residuals.shape}")
    reset_destination(destination, bool(args.overwrite))
    hardlink_source(source, destination, cut_index)
    base_error = write_post_smpl(
        source, destination, residuals, runtime["native_ids"], runtime["b0_vertices"],
        runtime["brtc_shifts_by_track"], cut_index,
    )
    manifest = {
        "format": "standard demo.py --save compatible; 5 pre + 25 post",
        "source": str(source),
        "cache": str(cache),
        "policy": str(policy),
        "cut_index": cut_index,
        "runtime_contract": {
            "gt_used": False,
            "future_post_frames_used": 0,
            "camera_update": "none",
            "camera_max_abs_change": debug["camera_max_abs_change"],
        },
        "source_brtc_cache_max_abs_error_m": base_error,
        "post_static_filtered_frames_by_native_id": {
            str(native_id): int(debug["gates"][:, person].sum())
            for person, native_id in enumerate(runtime["native_ids"][0])
        },
        "post_gate_reasons_by_native_id": {
            str(native_id): debug["reasons"][:, person].tolist()
            for person, native_id in enumerate(runtime["native_ids"][0])
        },
        "max_c1_residual_m_by_native_id": {
            str(native_id): float(np.linalg.norm(residuals[:, person], axis=1).max())
            for person, native_id in enumerate(runtime["native_ids"][0])
        },
    }
    manifest_path = destination.parent / "C1_EMA25_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f">> wrote C1 payload: {destination}")
    print(f">> manifest: {manifest_path}")
    print(f">> source/cache base max error: {base_error:.3e} m")


if __name__ == "__main__":
    main()
