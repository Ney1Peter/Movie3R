#!/usr/bin/env python3
"""Create a two-frame, B0-only demo payload from a frozen BRTC+C1 payload.

The first post-cut frame is before C1's warmup window, so its saved geometry is
the BRTC geometry.  We remove the frozen BRTC shift from that frame and keep
the B0 camera/background unchanged.  This is a visualization-only diagnostic;
it never reruns inference and does not use GT.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--boundary", type=int, default=5)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    source = args.input.resolve()
    destination = args.output.resolve()
    boundary = int(args.boundary)
    if not source.is_dir():
        raise FileNotFoundError(source)
    if boundary < 1:
        raise ValueError("boundary must leave at least one pre frame")
    if destination.exists():
        if not args.overwrite:
            raise FileExistsError(destination)
        shutil.rmtree(destination)
    shutil.copytree(source, destination)

    manifest_path = source.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    people = manifest["movie3r"]["brtc"]["people"]
    shifts = {
        int(row["post_index"]): np.asarray(row["final_shift_world"], dtype=np.float32)
        for row in people
    }
    if len(shifts) == 0:
        raise RuntimeError("No BRTC shifts in manifest")

    # The viewer payload stores meshes in world coordinates and preserves the
    # native smpl_id order.  At the first post frame C1 is still in warmup, so
    # removing BRTC's shift recovers exact B0-only geometry.
    post_index = boundary
    smpl_path = destination / "smpl" / f"{post_index:06d}.npz"
    with np.load(smpl_path, allow_pickle=True) as payload:
        values = {key: payload[key] for key in payload.files}
    vertices = np.asarray(values["verts_world"], dtype=np.float32).copy()
    ids = np.asarray(values.get("smpl_id", np.arange(len(vertices))), dtype=np.int64)
    for row, native_id in enumerate(ids.tolist()):
        # Single-person and multi-person payloads both use one BRTC shift per
        # native track.  A missing track is an invariant violation.
        if int(native_id) not in shifts:
            raise RuntimeError(f"Missing BRTC shift for native id {native_id}")
        vertices[row] -= shifts[int(native_id)]
    values["verts_world"] = vertices
    np.savez(smpl_path, **values)

    # Keep exactly the last pre and first post frames.  This applies to every
    # standard demo.py payload subdirectory (camera/depth/conf/color/smpl).
    kept = {boundary - 1, boundary}
    for subdir in ("camera", "depth", "conf", "color", "smpl"):
        directory = destination / subdir
        for path in directory.glob("*"):
            if path.is_file() and path.stem.isdigit() and int(path.stem) not in kept:
                path.unlink()
        # ``scripts/view_human3r_saved_output.py`` expects a zero-based,
        # contiguous payload, so rename last-pre/first-post to 000000/000001.
        for source_index, target_index in ((boundary - 1, 0), (boundary, 1)):
            source_path = directory / f"{source_index:06d}{next(directory.glob(f'{source_index:06d}.*')).suffix}"
            target_path = directory / f"{target_index:06d}{source_path.suffix}"
            if source_path.exists():
                source_path.rename(target_path)

    report = {
        "source": str(source),
        "variant": "B0-only two-frame diagnostic",
        "source_frames": [boundary - 1, boundary],
        "frames": [0, 1],
        "cut_index_in_payload": 1,
        "removed_brtc_shift_from_post_frame": {
            str(native_id): shift.tolist() for native_id, shift in shifts.items()
        },
        "c1_status": "exact warmup fallback at first post frame",
        "camera_changed": False,
        "gt_used": False,
        "future_frames_used": False,
    }
    (destination / "B0_ONLY_TWO_FRAME.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
