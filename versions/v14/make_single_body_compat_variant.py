#!/usr/bin/env python3
"""Build a singleton diagnostic payload with current cameras and original bodies.

This is deliberately a visualization ablation.  It keeps the current
B0+BRTC+C1 background/camera payload and replaces only post-cut cached meshes
with strict-Human3R post meshes transported into the current B0 world gauge.
It tests whether a singleton failure is caused by the current human head,
without changing the current camera estimate.  ``SMPL_Layer`` emits camera-
space vertices, so the default path applies the current camera exactly once.
The old relative-pose operation is retained only as an explicit legacy
ablation for reproducing the historical 8116 payload.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--current", type=Path, required=True)
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--boundary", type=int, default=5)
    p.add_argument(
        "--original-space", choices=("camera", "world"), default="camera",
        help="Coordinate space of original/smpl/verts_world in the source payload.",
    )
    p.add_argument(
        "--transport", choices=("current_camera", "relative_legacy"),
        default="current_camera",
        help="How to place the original body in the current viewer world.",
    )
    p.add_argument(
        "--post-shift",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("DX", "DY", "DZ"),
        help="Optional BRTC world translation to apply after transporting the original body.",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def pose(path: Path, index: int) -> np.ndarray:
    with np.load(path / "camera" / f"{index:06d}.npz") as z:
        return np.asarray(z["pose"], dtype=np.float64)


def main() -> None:
    args = parse_args()
    current, original, output = (p.resolve() for p in (args.current, args.original, args.output))
    if not current.is_dir() or not original.is_dir():
        raise FileNotFoundError((current, original))
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(output)
        shutil.rmtree(output)
    shutil.copytree(current, output)

    boundary = int(args.boundary)
    post_shift = np.asarray(args.post_shift, dtype=np.float64)
    for index in range(boundary, len(list((output / "camera").glob("*.npz")))):
        src_path = original / "smpl" / f"{index:06d}.npz"
        dst_path = output / "smpl" / f"{index:06d}.npz"
        with np.load(src_path, allow_pickle=True) as src:
            values = {key: src[key] for key in src.files}
        vertices = np.asarray(values["verts_world"], dtype=np.float64)
        current_camera = pose(current, index)
        original_camera = pose(original, index)
        if args.original_space == "camera":
            if args.transport == "current_camera":
                # Correct demo.py contract: V_world = C_current @ V_cam.
                world_vertices = (
                    vertices @ current_camera[:3, :3].T + current_camera[:3, 3]
                )
            else:
                # Historical 8116 operation.  This is invalid when vertices
                # are camera-space, but useful to reproduce the old payload.
                legacy = current_camera @ np.linalg.inv(original_camera)
                world_vertices = vertices @ legacy[:3, :3].T + legacy[:3, 3]
        else:
            # Source is already world-space; only a world-gauge bridge is
            # appropriate in this branch.
            bridge = current_camera @ np.linalg.inv(original_camera)
            world_vertices = vertices @ bridge[:3, :3].T + bridge[:3, 3]
        values["verts_world"] = (world_vertices + post_shift).astype(np.float32)
        np.savez(dst_path, **values)

    report = {
        "variant": "current_camera_background_plus_original_post_body_diagnostic",
        "current": str(current),
        "original": str(original),
        "boundary_index": boundary,
        "original_space": args.original_space,
        "transport": args.transport,
        "post_shift_world": post_shift.tolist(),
        "camera_changed": False,
        "future_post_frames_used": False,
        "runtime_status": "diagnostic_only_not_mainline",
    }
    (output / "body_compat_variant.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
