#!/usr/bin/env python3
"""Ablate V9/B0 with a raw Human3R camera SE(3) proposal.

This evaluator-only payload transform is the controlled ``no-V9`` baseline:
the post camera is mapped to the pre camera by ``G=C_pre@inv(C_post)``;
post meshes follow the same gauge, and an optional shared Kabsch residual can
be committed.  The selected boundary permutation is also written back to
``smpl_id`` so the comparison does not confuse removing V9 with removing ID
matching.  No GT is read.
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--boundary", type=int, default=5)
    p.add_argument("--human-residual", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def camera(path: Path, index: int) -> np.ndarray:
    with np.load(path / "camera" / f"{index:06d}.npz") as z: return np.asarray(z["pose"], dtype=np.float64)


def mesh(path: Path, index: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path / "smpl" / f"{index:06d}.npz", allow_pickle=True) as z:
        return np.asarray(z["verts_world"], dtype=np.float64), np.asarray(z.get("smpl_id", np.arange(len(z["verts_world"]))), dtype=np.int64).reshape(-1)


def transform_points(g: np.ndarray, x: np.ndarray) -> np.ndarray:
    return x @ g[:3, :3].T + g[:3, 3]


def kabsch(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    a, b = source - source.mean(0), target - target.mean(0)
    u, _, vh = np.linalg.svd(a.T @ b); r = vh.T @ u.T
    if np.linalg.det(r) < 0: vh[-1] *= -1; r = vh.T @ u.T
    t = target.mean(0) - r @ source.mean(0); g = np.eye(4); g[:3, :3] = r; g[:3, 3] = t
    mapped = transform_points(g, source); return g, float(np.sqrt(np.mean(np.sum((mapped - target) ** 2, axis=1))))


def main() -> None:
    a = parse_args(); source, output = a.source.resolve(), a.output.resolve()
    if output.exists():
        if not a.overwrite: raise FileExistsError(output)
        shutil.rmtree(output)
    shutil.copytree(source, output)
    n = len(list((source / "camera").glob("*.npz"))); b = int(a.boundary)
    pre_cam, post_cam = camera(source, b - 1), camera(source, b)
    g_raw = pre_cam @ np.linalg.inv(post_cam)
    pre, pre_ids = mesh(source, b - 1); post, post_ids = mesh(source, b)
    post_raw = transform_points(g_raw, post)
    candidates = []
    for perm in itertools.permutations(range(len(post_raw))):
        h, rms = kabsch(np.concatenate([post_raw[i] for i in perm]), np.concatenate([pre[i] for i in range(len(pre))]))
        candidates.append((rms, perm, h))
    candidates.sort(key=lambda x: x[0]); rms, perm, h = candidates[0]
    committed = h @ g_raw if a.human_residual else g_raw
    id_map = {int(post_ids[post_index]): int(pre_ids[pre_index]) for pre_index, post_index in enumerate(perm)}
    for i in range(b, n):
        cp = output / "camera" / f"{i:06d}.npz"
        with np.load(cp) as z: vals = {k: z[k] for k in z.files}
        vals["pose"] = (committed @ np.asarray(vals["pose"], dtype=np.float64)).astype(np.float32); np.savez(cp, **vals)
        sp = output / "smpl" / f"{i:06d}.npz"
        with np.load(sp, allow_pickle=True) as z: vals = {k: z[k] for k in z.files}
        vals["verts_world"] = transform_points(committed, np.asarray(vals["verts_world"], dtype=np.float64)).astype(np.float32)
        if "smpl_id" in vals:
            vals["smpl_id"] = np.asarray([id_map.get(int(x), int(x)) for x in np.asarray(vals["smpl_id"]).reshape(-1)], dtype=np.int64)
        np.savez(sp, **vals)
    report = {
        "source": str(source), "output": str(output), "boundary": b,
        "method": "raw_human3r_camera_se3_plus_optional_shared_human_kabsch",
        "human_residual_committed": bool(a.human_residual),
        "raw_camera_rotation_deg": float(np.degrees(np.linalg.norm(Rotation.from_matrix(g_raw[:3,:3]).as_rotvec()))),
        "raw_camera_translation_m": float(np.linalg.norm(g_raw[:3,3])),
        "human_residual_rotation_deg": float(np.degrees(np.linalg.norm(Rotation.from_matrix(h[:3,:3]).as_rotvec()))),
        "human_residual_rms_m": float(rms), "selected_permutation_post_index_by_pre_index": list(perm),
        "pre_native_ids": pre_ids.tolist(), "post_native_ids": post_ids.tolist(),
        "post_to_pre_id": {str(k): int(v) for k,v in id_map.items()},
        "runtime_contract": "GT-free; raw Human3R camera proposal, no V9/B0 checkpoint",
    }
    (output / "raw_se3_human_boundary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
