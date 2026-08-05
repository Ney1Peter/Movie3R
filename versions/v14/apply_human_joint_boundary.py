#!/usr/bin/env python3
"""Apply an online human-derived residual Boundary to a saved B0 payload.

This is an evaluator/deployment probe, not a GT-based method.  It estimates one
shared SE(3) from the last pre-cut and first post-cut predicted SMPL meshes,
then applies a bounded interpolation of that residual to both post cameras and
post meshes.  The same residual is used for every later post frame, preserving
the streaming contract.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True, help="Saved B0/BRTC payload")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--boundary", type=int, default=5, help="First post frame index")
    p.add_argument("--alpha", type=float, default=1.0, help="Residual human Boundary strength")
    p.add_argument(
        "--anchor-root", action="store_true",
        help="After the shared camera/body update, preserve each post-frame predicted SMPL root.",
    )
    p.add_argument(
        "--remap-ids", action="store_true",
        help="Commit the selected boundary permutation as a persistent smpl_id mapping.",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load(path: Path, index: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path / "smpl" / f"{index:06d}.npz", allow_pickle=True) as z:
        vertices = np.asarray(z["verts_world"], dtype=np.float64)
        ids = np.asarray(z.get("smpl_id", np.arange(len(vertices))), dtype=np.int64)
    return vertices, ids


def pose(path: Path, index: int) -> np.ndarray:
    with np.load(path / "camera" / f"{index:06d}.npz") as z:
        return np.asarray(z["pose"], dtype=np.float64)


def kabsch(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    a = source - source_center
    b = target - target_center
    left, _, right = np.linalg.svd(a.T @ b)
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right[-1] *= -1.0
        rotation = right.T @ left.T
    translation = target_center - rotation @ source_center
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    mapped = source @ rotation.T + translation
    rms = float(np.sqrt(np.mean(np.sum((mapped - target) ** 2, axis=1))))
    return transform, rms


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]


def residual_interp(transform: np.ndarray, alpha: float) -> np.ndarray:
    """Bounded identity-to-SE3 interpolation for a safe online update."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    rotation = Rotation.from_matrix(np.asarray(transform)[:3, :3])
    output = np.eye(4, dtype=np.float64)
    output[:3, :3] = Rotation.from_rotvec(alpha * rotation.as_rotvec()).as_matrix()
    output[:3, 3] = alpha * np.asarray(transform, dtype=np.float64)[:3, 3]
    return output


def choose_shared_boundary(pre: np.ndarray, post: np.ndarray) -> tuple[np.ndarray, dict]:
    if pre.ndim != 3 or post.ndim != 3 or pre.shape[0] != post.shape[0] or pre.shape[0] < 1:
        raise ValueError(f"Expected equal non-empty human arrays, got pre={pre.shape}, post={post.shape}")
    count = int(pre.shape[0])
    candidates = []
    for permutation in itertools.permutations(range(count)):
        source = np.concatenate([post[index] for index in permutation], axis=0)
        target = np.concatenate([pre[index] for index in range(count)], axis=0)
        transform, rms = kabsch(source, target)
        candidates.append((rms, tuple(int(value) for value in permutation), transform))
    candidates.sort(key=lambda value: value[0])
    rms, permutation, transform = candidates[0]
    rotation_deg = float(np.degrees(np.linalg.norm(Rotation.from_matrix(transform[:3, :3]).as_rotvec())))
    return transform, {
        "selected_permutation_post_index_by_pre_index": list(permutation),
        "shared_vertex_rms_m": float(rms),
        "shared_rotation_deg": rotation_deg,
        "candidate_count": len(candidates),
        "second_best_rms_m": float(candidates[1][0]) if len(candidates) > 1 else None,
        "permutation_margin_m": float(candidates[1][0] - rms) if len(candidates) > 1 else None,
    }


def main() -> None:
    a = args()
    source = a.source.resolve()
    output = a.output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.exists():
        if not a.overwrite:
            raise FileExistsError(output)
        shutil.rmtree(output)
    shutil.copytree(source, output)

    frame_count = len(list((source / "camera").glob("*.npz")))
    boundary = int(a.boundary)
    if boundary <= 0 or boundary >= frame_count:
        raise ValueError(f"Invalid boundary={boundary} for frame_count={frame_count}")
    pre, pre_ids = load(source, boundary - 1)
    post, post_ids = load(source, boundary)
    boundary_human, diagnostics = choose_shared_boundary(pre, post)
    applied = residual_interp(boundary_human, float(a.alpha))
    post_to_pre_id: dict[int, int] = {}
    if a.remap_ids:
        permutation = diagnostics["selected_permutation_post_index_by_pre_index"]
        post_to_pre_id = {
            int(post_ids[post_index]): int(pre_ids[pre_index])
            for pre_index, post_index in enumerate(permutation)
        }

    regressor = None
    if a.anchor_root:
        # The runtime already has an SMPL layer; this import is only needed by
        # this payload probe.  Root anchoring is a per-person residual after
        # the shared camera update and does not use GT.
        import smplx

        model_root = Path(__file__).resolve().parents[2] / "src" / "models"
        body_model = smplx.create(
            str(model_root), "smplx", gender="neutral", use_pca=False,
            flat_hand_mean=True, num_betas=10,
        ).eval()
        regressor = body_model.J_regressor.detach().cpu().numpy().astype(np.float64)

    for index in range(boundary, frame_count):
        camera_path = output / "camera" / f"{index:06d}.npz"
        with np.load(camera_path) as z:
            values = {key: z[key] for key in z.files}
        values["pose"] = (applied @ np.asarray(values["pose"], dtype=np.float64)).astype(np.float32)
        np.savez(camera_path, **values)

        smpl_path = output / "smpl" / f"{index:06d}.npz"
        with np.load(smpl_path, allow_pickle=True) as z:
            values = {key: z[key] for key in z.files}
        if "verts_world" in values:
            before_vertices = np.asarray(values["verts_world"], dtype=np.float64)
            after_vertices = transform_points(applied, before_vertices)
            if regressor is not None and len(before_vertices) == len(after_vertices):
                before_roots = np.einsum("jv,nvk->njk", regressor[[0]], before_vertices)[:, 0]
                after_roots = np.einsum("jv,nvk->njk", regressor[[0]], after_vertices)[:, 0]
                after_vertices = after_vertices + (before_roots - after_roots)[:, None, :]
            values["verts_world"] = after_vertices.astype(np.float32)
        if a.remap_ids and "smpl_id" in values:
            ids = np.asarray(values["smpl_id"], dtype=np.int64).reshape(-1)
            values["smpl_id"] = np.asarray(
                [post_to_pre_id.get(int(value), int(value)) for value in ids],
                dtype=np.int64,
            )
        np.savez(smpl_path, **values)

    diagnostics.update({
        "source": str(source),
        "output": str(output),
        "boundary_index": boundary,
        "frame_count": frame_count,
        "pre_native_ids": pre_ids.tolist(),
        "post_native_ids": post_ids.tolist(),
        "alpha": float(np.clip(a.alpha, 0.0, 1.0)),
        "anchor_root": bool(a.anchor_root),
        "remap_ids": bool(a.remap_ids),
        "post_to_pre_id": {str(key): int(value) for key, value in post_to_pre_id.items()},
        "applied_rotation_deg": float(np.degrees(np.linalg.norm(Rotation.from_matrix(applied[:3, :3]).as_rotvec()))),
        "applied_translation_m": float(np.linalg.norm(applied[:3, 3])),
        "runtime_contract": "GT-free; one residual estimated at first post and reused causally for all post frames",
    })
    (output / "human_joint_boundary.json").write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(diagnostics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
