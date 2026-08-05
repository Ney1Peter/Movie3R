#!/usr/bin/env python3
"""Adaptive post-baseline shared human--camera boundary correction.

The input is a saved B0+BRTC+C1 payload.  At each supplied candidate cut the
last pre-cut and first post-cut predicted meshes are matched anonymously and a
single post-to-pre SE(3) is estimated.  A conservative, GT-free gate decides
whether to apply it.  An accepted transform is held causally for every later
post frame and is applied identically to camera poses, saved world meshes and
background geometry (background depth itself remains camera-local).

This is the deployable *post-gate* path.  The companion front-detector path
(``streaming_detector_joint_boundary.py``) first proposes cut indices from RGB
only and then invokes the same geometry gate.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

from src.dust3r.adaptive_joint import (
    AdaptiveJointConfig,
    apply_to_arrays,
    apply_with_raw_reference,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument(
        "--raw-source",
        type=Path,
        default=None,
        help="Optional same-checkpoint raw Human3R payload for root-ray camera refinement.",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--boundary", type=int, action="append", required=True)
    p.add_argument("--min-rotation-deg", type=float, default=20.0)
    p.add_argument("--max-vertex-rms-m", type=float, default=0.20)
    p.add_argument("--max-normalized-rms", type=float, default=0.20)
    p.add_argument("--min-permutation-margin-m", type=float, default=0.01)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def frame_count(payload: Path) -> int:
    values = sorted((payload / "camera").glob("*.npz"))
    if not values:
        raise FileNotFoundError(f"No camera frames under {payload}")
    expected = [f"{i:06d}.npz" for i in range(len(values))]
    if [p.name for p in values] != expected:
        raise ValueError(f"Payload camera files are not contiguous: {payload}")
    return len(values)


def load_camera(path: Path, index: int) -> np.ndarray:
    with np.load(path / "camera" / f"{index:06d}.npz") as z:
        return np.asarray(z["pose"], dtype=np.float64)


def load_mesh(path: Path, index: int) -> np.ndarray:
    with np.load(path / "smpl" / f"{index:06d}.npz", allow_pickle=True) as z:
        if "verts_world" not in z.files:
            raise RuntimeError(
                f"{path}/smpl/{index:06d}.npz lacks verts_world. Run "
                "cache_payload_verts_world.py first; no model or GT is used."
            )
        return np.asarray(z["verts_world"], dtype=np.float64)


def replace_npz(path: Path, values: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(path.name + ".adaptive_tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **values)
    os.replace(temporary, path)


def write_payload(source: Path, output: Path, cameras: np.ndarray, meshes: list[np.ndarray]) -> None:
    for index, camera in enumerate(cameras):
        camera_path = output / "camera" / f"{index:06d}.npz"
        with np.load(camera_path) as z:
            values = {key: z[key] for key in z.files}
        values["pose"] = camera.astype(np.float32)
        replace_npz(camera_path, values)

        smpl_path = output / "smpl" / f"{index:06d}.npz"
        with np.load(smpl_path, allow_pickle=True) as z:
            values = {key: z[key] for key in z.files}
        values["verts_world"] = meshes[index].astype(np.float32)
        replace_npz(smpl_path, values)


def main() -> None:
    a = parse_args()
    source = a.source.resolve()
    output = a.output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.exists():
        if not a.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite: {output}")
        if output in {Path("/"), Path("/data"), Path("/data/wangzheng")}:  # safety
            raise ValueError(f"Refusing broad output deletion: {output}")
        shutil.rmtree(output)
    shutil.copytree(source, output)

    n = frame_count(source)
    cameras = np.stack([load_camera(source, i) for i in range(n)], axis=0)
    meshes = [load_mesh(source, i) for i in range(n)]
    cfg = AdaptiveJointConfig(
        min_rotation_deg=a.min_rotation_deg,
        max_vertex_rms_m=a.max_vertex_rms_m,
        max_normalized_rms=a.max_normalized_rms,
        min_permutation_margin_m=a.min_permutation_margin_m,
        alpha=a.alpha,
    )
    raw_cameras = raw_meshes = None
    if a.raw_source is not None:
        raw_source = a.raw_source.resolve()
        raw_n = frame_count(raw_source)
        if raw_n != n:
            raise ValueError(f"Raw payload frame count {raw_n} != source frame count {n}")
        raw_cameras = np.stack([load_camera(raw_source, i) for i in range(n)], axis=0)
        raw_meshes = [load_mesh(raw_source, i) for i in range(n)]
    if raw_cameras is None or raw_meshes is None:
        cameras_new, meshes_new, _, records = apply_to_arrays(
            cameras, meshes, None, a.boundary, cfg
        )
    else:
        cameras_new, meshes_new, _, records = apply_with_raw_reference(
            cameras, meshes, raw_cameras, raw_meshes, None, a.boundary, cfg
        )
    write_payload(source, output, cameras_new, meshes_new)
    diagnostics = {
        "method": "adaptive_post_shared_human_camera_boundary_v1",
        "source": str(source),
        "raw_source": str(a.raw_source.resolve()) if a.raw_source is not None else None,
        "output": str(output),
        "frame_count": n,
        "candidate_boundaries": [int(v) for v in a.boundary],
        "config": {
            "min_rotation_deg": float(a.min_rotation_deg),
            "max_vertex_rms_m": float(a.max_vertex_rms_m),
            "max_normalized_rms": float(a.max_normalized_rms),
            "min_permutation_margin_m": float(a.min_permutation_margin_m),
            "alpha": float(a.alpha),
        },
        "runtime_contract": "GT-free; pre frames unchanged; one accepted transform held causally over each post shot",
        "records": records,
    }
    (output / "adaptive_joint_boundary.json").write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(diagnostics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
