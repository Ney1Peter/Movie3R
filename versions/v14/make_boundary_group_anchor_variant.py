#!/usr/bin/env python3
"""Build a visualization-only causal group-anchor variant from a saved payload.

The saved B0+BRTC+C1 payload already contains the predicted pre/post people.
At the first post frame, estimate one common world translation from matched
mesh centers (last pre -> first post), then apply it to every post human mesh.
This is an ablation of the missing absolute human anchor; it does not change
cameras, RGB, depth, or the original payload.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--boundary", type=int, default=5)
    parser.add_argument(
        "--estimator",
        choices=("mean", "median", "individual"),
        default="mean",
        help="Robustness of the common shift estimator across matched people.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Fraction of the causal boundary shift to apply (0..1).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def centers(path: Path, index: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path / "smpl" / f"{index:06d}.npz", allow_pickle=True) as payload:
        vertices = np.asarray(payload["verts_world"], dtype=np.float64)
        ids = np.asarray(payload["smpl_id"], dtype=np.int64)
    if vertices.ndim != 3 or vertices.shape[-1] != 3:
        raise ValueError(f"Invalid verts_world at {path}: {vertices.shape}")
    return ids, vertices.mean(axis=1)


def main() -> None:
    args = parse_args()
    source = args.input.resolve()
    destination = args.output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if destination.exists():
        if not args.overwrite:
            raise FileExistsError(destination)
        shutil.rmtree(destination)
    shutil.copytree(source, destination)

    boundary = int(args.boundary)
    if not 0.0 <= float(args.alpha) <= 1.0:
        raise ValueError("--alpha must be in [0, 1]")
    pre_ids, pre_centers = centers(source, boundary - 1)
    post_ids, post_centers = centers(source, boundary)
    pre_by_id = {int(identity): value for identity, value in zip(pre_ids, pre_centers)}
    post_by_id = {int(identity): value for identity, value in zip(post_ids, post_centers)}
    common = [pre_by_id[int(identity)] - post_by_id[int(identity)]
              for identity in post_ids if int(identity) in pre_by_id]
    if not common:
        raise RuntimeError("No shared native IDs at the boundary")
    common_array = np.stack(common)
    if args.estimator == "median":
        raw_shift = np.median(common_array, axis=0)
    elif args.estimator == "individual":
        # Kept for reporting; each native track receives its own causal shift
        # below.  The common value is only a compact summary of the proposal.
        raw_shift = np.mean(common_array, axis=0)
    else:
        raw_shift = np.mean(common_array, axis=0)
    shift = float(args.alpha) * raw_shift
    individual_shifts = {
        int(identity): float(args.alpha) * (pre_by_id[int(identity)] - post_by_id[int(identity)])
        for identity in post_ids if int(identity) in pre_by_id
    }

    frame_files = sorted((destination / "camera").glob("*.npz"))
    for camera_file in frame_files:
        index = int(camera_file.stem)
        if index < boundary:
            continue
        smpl_path = destination / "smpl" / f"{index:06d}.npz"
        with np.load(smpl_path, allow_pickle=True) as payload:
            values = {key: payload[key] for key in payload.files}
        verts_world = np.asarray(values["verts_world"], dtype=np.float32)
        if args.estimator == "individual":
            frame_ids = np.asarray(values["smpl_id"], dtype=np.int64)
            for row, identity in enumerate(frame_ids.tolist()):
                if int(identity) in individual_shifts:
                    verts_world[row] += individual_shifts[int(identity)].astype(np.float32)
        else:
            verts_world += shift.astype(np.float32)
        values["verts_world"] = verts_world
        np.savez(smpl_path, **values)

    report = {
        "source": str(source),
        "variant": "causal_boundary_group_anchor_visual_ablation",
        "boundary_index": boundary,
        "anchor_ids": [int(identity) for identity in post_ids if int(identity) in pre_by_id],
        "per_person_shifts_world": {
            str(int(identity)): (pre_by_id[int(identity)] - post_by_id[int(identity)]).tolist()
            for identity in post_ids if int(identity) in pre_by_id
        },
        "estimator": args.estimator,
        "alpha": float(args.alpha),
        "raw_shift_world": raw_shift.tolist(),
        "common_shift_world": shift.tolist(),
        "individual_shift_world": {str(k): v.tolist() for k, v in individual_shifts.items()},
        "camera_changed": False,
        "future_post_frames_used": False,
    }
    (destination / "boundary_group_anchor.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
