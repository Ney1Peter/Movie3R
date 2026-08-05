#!/usr/bin/env python3
"""Apply a causal, confidence-gated camera--human boundary correction.

The input is the frozen ``B0 + BRTC + C1`` demo payload.  ``human-source``
is the raw Human3R payload produced by the same checkpoint/forward pass (it is
not an additional pretrained model).  At the first post frame we compare the
current B0 body with the raw body, estimate a bounded world rotation, and use
the two body-root rays as a robust camera-depth candidate.  The candidate is
accepted only when the B0 body itself has a large cross-shot rotation residual;
otherwise the frozen B0 camera/body output is copied exactly.  The correction
is then reused causally for the rest of the post shot.

This is a deployable probe: GT is never read.  The evaluator may compare the
saved payload against GT afterwards.
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True, help="B0+BRTC+C1 payload")
    p.add_argument("--human-source", type=Path, default=None,
                   help="raw Human3R payload from the same input frames")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--boundary", type=int, default=5)
    p.add_argument("--human-boundary", type=int, default=None,
                   help="Boundary index in a shorter raw payload (for first-cut audit only).")
    p.add_argument("--rotation-alpha", type=float, default=1.0)
    p.add_argument("--rotation-source", choices=("raw_human", "b0_boundary"), default="b0_boundary")
    p.add_argument("--b0-rotation-gate-deg", type=float, default=25.0)
    p.add_argument("--human-rms-gate-m", type=float, default=0.15)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_npz(path: Path, index: int, allow_pickle: bool = True) -> dict[str, np.ndarray]:
    with np.load(path / "smpl" / f"{index:06d}.npz", allow_pickle=allow_pickle) as z:
        return {key: z[key] for key in z.files}


def load_pose(path: Path, index: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path / "camera" / f"{index:06d}.npz") as z:
        return np.asarray(z["pose"], dtype=np.float64), np.asarray(z["intrinsics"])


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def kabsch(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    left, _, right = np.linalg.svd((source - source_center).T @ (target - target_center))
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right[-1] *= -1.0
        rotation = right.T @ left.T
    translation = target_center - rotation @ source_center
    mapped = source @ rotation.T + translation
    rms = float(np.sqrt(np.mean(np.sum((mapped - target) ** 2, axis=1))))
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rotation
    out[:3, 3] = translation
    return out, rms


def root_regressor() -> np.ndarray:
    # Import lazily so the no-source fallback remains a light payload copy.
    import smplx

    repo_root = Path(__file__).resolve().parents[2]
    model = smplx.create(
        str(repo_root / "src/models"), "smplx", gender="neutral", use_pca=False,
        flat_hand_mean=True, num_betas=10,
    ).eval()
    return model.J_regressor.detach().cpu().numpy().astype(np.float64)[0]


def root_from_vertices(vertices: np.ndarray, regressor: np.ndarray) -> np.ndarray:
    return np.einsum("v,nvk->nk", regressor, vertices)


def choose_permutation(current: np.ndarray, raw: np.ndarray) -> tuple[tuple[int, ...], float, float]:
    """Match raw detections to B0 detections using shape-only evidence."""
    if current.ndim != 3 or raw.ndim != 3 or current.shape[0] != raw.shape[0]:
        raise ValueError(f"person count mismatch current={current.shape} raw={raw.shape}")
    candidates: list[tuple[float, tuple[int, ...], float]] = []
    for permutation in itertools.permutations(range(raw.shape[0])):
        pieces = []
        for i, j in enumerate(permutation):
            a, b = current[i], raw[j]
            _, rms = kabsch(a - a.mean(0), b - b.mean(0))
            pieces.append(rms)
        candidates.append((float(np.mean(pieces)), permutation, float(np.max(pieces))))
    candidates.sort(key=lambda row: row[0])
    best = candidates[0]
    return best[1], best[0], float(candidates[1][0] - best[0]) if len(candidates) > 1 else float("inf")


def choose_global_permutation(current: np.ndarray, post: np.ndarray) -> tuple[tuple[int, ...], float, float]:
    """Choose identity ordering with one shared world Kabsch residual."""
    rows: list[tuple[float, tuple[int, ...]]] = []
    for permutation in itertools.permutations(range(post.shape[0])):
        source = np.concatenate([post[j] for j in permutation], axis=0)
        target = np.concatenate([current[i] for i in range(current.shape[0])], axis=0)
        _, rms = kabsch(source, target)
        rows.append((rms, permutation))
    rows.sort(key=lambda row: row[0])
    return rows[0][1], float(rows[0][0]), float(rows[1][0] - rows[0][0]) if len(rows) > 1 else float("inf")


def shared_boundary_rotation(current_pre: np.ndarray, current_post: np.ndarray) -> tuple[np.ndarray, float, float]:
    permutation, rms, margin = choose_global_permutation(current_pre, current_post)
    source = np.concatenate([current_post[j] for j in permutation], axis=0)
    target = np.concatenate([current_pre[i] for i in range(current_pre.shape[0])], axis=0)
    boundary, _ = kabsch(source, target)
    angle = float(np.degrees(np.linalg.norm(Rotation.from_matrix(boundary[:3, :3]).as_rotvec())))
    return boundary, angle, rms


def apply(args: argparse.Namespace) -> dict:
    source = args.source.resolve()
    output = args.output.resolve()
    raw = args.human_source.resolve() if args.human_source is not None else None
    if not source.is_dir():
        raise FileNotFoundError(source)
    if raw is not None and not raw.is_dir():
        raise FileNotFoundError(raw)
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(output)
        shutil.rmtree(output)
    shutil.copytree(source, output)

    frame_count = len(list((source / "camera").glob("*.npz")))
    boundary = int(args.boundary)
    if boundary <= 0 or boundary >= frame_count:
        raise ValueError(f"Invalid boundary={boundary} for frame_count={frame_count}")

    diagnostics: dict = {
        "source": str(source), "human_source": str(raw) if raw is not None else None,
        "output": str(output), "boundary_index": boundary, "frame_count": frame_count,
        "rotation_alpha": float(np.clip(args.rotation_alpha, 0.0, 1.0)),
        "b0_rotation_gate_deg": float(args.b0_rotation_gate_deg),
        "human_rms_gate_m": float(args.human_rms_gate_m),
        "runtime_contract": "GT-free; raw and B0 branches are same-checkpoint causal outputs; one boundary update reused post-cut",
    }

    # No raw branch means no human camera evidence: safe B0 fallback.
    if raw is None:
        diagnostics.update({"accepted": False, "reason": "missing_human_source"})
        (output / "joint_camera_human.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
        return diagnostics

    raw_boundary = boundary if args.human_boundary is None else int(args.human_boundary)
    current_pre = np.asarray(load_npz(source, boundary - 1)["verts_world"], dtype=np.float64)
    current_post = np.asarray(load_npz(source, boundary)["verts_world"], dtype=np.float64)
    raw_post = np.asarray(load_npz(raw, raw_boundary)["verts_world"], dtype=np.float64)
    if current_pre.shape != current_post.shape or current_post.shape != raw_post.shape:
        diagnostics.update({"accepted": False, "reason": "person_count_or_mesh_shape_mismatch",
                            "current_pre_shape": list(current_pre.shape),
                            "current_post_shape": list(current_post.shape),
                            "raw_post_shape": list(raw_post.shape)})
        (output / "joint_camera_human.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
        return diagnostics

    boundary_b0, b0_angle, b0_rms = shared_boundary_rotation(current_pre, current_post)
    permutation, human_rms, human_margin = choose_permutation(current_post, raw_post)
    # A single shared shape rotation is the camera candidate.  The translation
    # is deliberately not taken from Kabsch; it is solved from the root ray.
    current_centred = np.concatenate([current_post[i] - current_post[i].mean(0) for i in range(current_post.shape[0])])
    raw_centred = np.concatenate([raw_post[j] - raw_post[j].mean(0) for j in permutation])
    human_rotation, _ = kabsch(current_centred, raw_centred)
    selected_rotation = human_rotation if args.rotation_source == "raw_human" else boundary_b0
    human_angle = float(np.degrees(np.linalg.norm(Rotation.from_matrix(human_rotation[:3, :3]).as_rotvec())))
    accepted = bool(b0_angle >= float(args.b0_rotation_gate_deg) and human_rms <= float(args.human_rms_gate_m))
    diagnostics.update({
        "b0_boundary_rotation_deg": b0_angle, "b0_boundary_shape_rms_m": b0_rms,
        "human_candidate_rotation_deg": human_angle, "human_candidate_shape_rms_m": human_rms,
        "human_candidate_permutation_raw_index_by_current_index": list(permutation),
        "human_candidate_permutation_margin_m": human_margin,
        "rotation_source": str(args.rotation_source),
        "accepted": accepted,
        "reason": "large_b0_human_rotation_and_reliable_raw_shape" if accepted else "confidence_gate_reject_b0_kept",
    })
    if not accepted:
        (output / "joint_camera_human.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
        return diagnostics

    reg = root_regressor()
    alpha = float(np.clip(args.rotation_alpha, 0.0, 1.0))
    rdelta = Rotation.from_rotvec(alpha * Rotation.from_matrix(selected_rotation[:3, :3]).as_rotvec()).as_matrix()
    diagnostics["applied_rotation_deg"] = float(np.degrees(np.linalg.norm(Rotation.from_matrix(rdelta).as_rotvec())))
    per_frame = []
    for index in range(boundary, frame_count):
        c_path = output / "camera" / f"{index:06d}.npz"
        c_pose, c_intrinsics = load_pose(source, index)
        current_values = load_npz(source, index)
        raw_index = index if args.human_boundary is None else raw_boundary + (index - boundary)
        if not (raw / "smpl" / f"{raw_index:06d}.npz").is_file() or not (raw / "camera" / f"{raw_index:06d}.npz").is_file():
            per_frame.append({"index": index, "accepted": False, "reason": "raw_frame_unavailable"})
            continue
        raw_values = load_npz(raw, raw_index)
        current_vertices = np.asarray(current_values["verts_world"], dtype=np.float64)
        raw_vertices = np.asarray(raw_values["verts_world"], dtype=np.float64)
        if current_vertices.ndim != 3 or raw_vertices.ndim != 3 or current_vertices.shape[0] != raw_vertices.shape[0]:
            # A later visibility change cannot be silently remapped.  Keep the
            # B0 frame and record the safe fallback.
            per_frame.append({"index": index, "accepted": False, "reason": "later_person_count_mismatch"})
            continue
        roots_current = root_from_vertices(current_vertices, reg)
        roots_raw = root_from_vertices(raw_vertices, reg)
        raw_perm, _, _ = choose_permutation(current_vertices, raw_vertices)
        c_new = np.array(c_pose, copy=True)
        c_new[:3, :3] = rdelta @ c_pose[:3, :3]
        q_current = np.einsum("ij,nj->ni", c_pose[:3, :3].T, roots_current - c_pose[:3, 3])
        raw_pose, _ = load_pose(raw, raw_index)
        q_raw = np.einsum("ij,nj->ni", raw_pose[:3, :3].T, roots_raw - raw_pose[:3, 3])
        q_raw_reordered = q_raw[list(raw_perm)]
        q_human = 0.5 * q_current + 0.5 * q_raw_reordered
        # The root is the already corrected BRTC anchor.  This makes camera
        # translation and body motion share exactly the same world anchor.
        c_new[:3, 3] = np.mean(roots_current - np.einsum("ij,nj->ni", c_new[:3, :3], q_human), axis=0)
        np.savez(c_path, pose=c_new.astype(np.float32), intrinsics=c_intrinsics)

        smpl_path = output / "smpl" / f"{index:06d}.npz"
        values = load_npz(output, index)
        before = np.asarray(values["verts_world"], dtype=np.float64)
        roots = root_from_vertices(before, reg)
        after = before @ rdelta.T + (roots[:, None, :] - (roots @ rdelta.T)[:, None, :])
        values["verts_world"] = after.astype(np.float32)
        np.savez(smpl_path, **values)
        per_frame.append({"index": index, "accepted": True, "mean_root_ray_m": float(np.linalg.norm(q_human, axis=1).mean())})
    diagnostics["per_frame"] = per_frame
    (output / "joint_camera_human.json").write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return diagnostics


def main() -> None:
    result = apply(parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
