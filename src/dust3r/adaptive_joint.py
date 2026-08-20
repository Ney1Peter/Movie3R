"""Causal, adaptive shared human--camera boundary correction.

This module contains the small geometry core used by both the saved-payload
auditor and ``demo.py``.  It deliberately has no GT or learned component:
the only signal is the predicted human mesh immediately before and after a
candidate shot boundary.  If a reliable rigid residual is found, the same
world SE(3) is applied to the post-cut camera, background point map and human
mesh.  Otherwise the baseline is returned byte-for-byte in spirit (the
in-memory values are left unchanged).

The operation is causal.  At boundary ``b`` only frames ``b-1`` and ``b`` are
read to estimate the update; the update is then held for later frames in the
shot.  A later candidate can be evaluated independently.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class AdaptiveJointConfig:
    """Deployment thresholds for the conservative residual gate."""

    min_rotation_deg: float = 20.0
    max_vertex_rms_m: float = 0.20
    max_normalized_rms: float = 0.20
    min_permutation_margin_m: float = 0.01
    alpha: float = 1.0


def _as_vertices(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.asarray(value, dtype=np.float64)
    if value.size == 0:
        return np.empty((0, 0, 3), dtype=np.float64)
    if value.ndim != 3 or value.shape[1] < 3 or value.shape[2] != 3:
        raise ValueError(f"Expected [people, vertices, 3], got {value.shape}")
    return value


def _transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _kabsch(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"Kabsch shape mismatch: {source.shape} vs {target.shape}")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    left, _, right = np.linalg.svd(
        (source - source_center).T @ (target - target_center)
    )
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right[-1] *= -1.0
        rotation = right.T @ left.T
    translation = target_center - rotation @ source_center
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    mapped = _transform_points(transform, source)
    rms = float(np.sqrt(np.mean(np.sum((mapped - target) ** 2, axis=-1))))
    return transform, rms


def estimate_shared_boundary(
    pre_vertices: Any, post_vertices: Any
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Estimate post-to-pre shared SE(3), including anonymous ID matching."""

    pre = _as_vertices(pre_vertices)
    post = _as_vertices(post_vertices)
    if pre.shape != post.shape or pre.shape[0] == 0:
        return None, {
            "valid": False,
            "reason": "person_count_or_mesh_shape_mismatch",
            "pre_shape": list(pre.shape),
            "post_shape": list(post.shape),
        }

    count = int(pre.shape[0])
    rows: list[tuple[float, tuple[int, ...], np.ndarray]] = []
    for permutation in itertools.permutations(range(count)):
        source = np.concatenate([post[j] for j in permutation], axis=0)
        target = np.concatenate([pre[i] for i in range(count)], axis=0)
        transform, rms = _kabsch(source, target)
        rows.append((rms, tuple(int(j) for j in permutation), transform))
    rows.sort(key=lambda row: row[0])
    rms, permutation, transform = rows[0]
    angle = float(
        np.degrees(np.linalg.norm(Rotation.from_matrix(transform[:3, :3]).as_rotvec()))
    )
    # The robust body-scale normalization prevents a fixed metre threshold
    # from changing behaviour between AvatarReX and multi-person captures.
    extent = np.percentile(pre.reshape(-1, 3), 95, axis=0) - np.percentile(
        pre.reshape(-1, 3), 5, axis=0
    )
    scale = float(max(np.linalg.norm(extent), 1e-6))
    return transform, {
        "valid": True,
        "selected_permutation_post_index_by_pre_index": list(permutation),
        "shared_vertex_rms_m": float(rms),
        "shared_normalized_rms": float(rms / scale),
        "body_extent_m": scale,
        "shared_rotation_deg": angle,
        "candidate_count": len(rows),
        "second_best_rms_m": float(rows[1][0]) if len(rows) > 1 else None,
        "permutation_margin_m": float(rows[1][0] - rms) if len(rows) > 1 else None,
    }


def gate_boundary(
    diagnostics: dict[str, Any], config: AdaptiveJointConfig
) -> tuple[bool, str]:
    """Apply the conservative geometry gate without looking at GT."""

    if not diagnostics.get("valid", False):
        return False, str(diagnostics.get("reason", "invalid_geometry"))
    rms = float(diagnostics["shared_vertex_rms_m"])
    norm_rms = float(diagnostics["shared_normalized_rms"])
    angle = float(diagnostics["shared_rotation_deg"])
    margin = diagnostics.get("permutation_margin_m")
    if angle < float(config.min_rotation_deg):
        return False, "small_boundary_residual_baseline_kept"
    if rms > float(config.max_vertex_rms_m):
        return False, "human_shape_residual_too_large_baseline_kept"
    if norm_rms > float(config.max_normalized_rms):
        return False, "scale_normalized_residual_too_large_baseline_kept"
    if margin is not None and float(margin) < float(config.min_permutation_margin_m):
        return False, "ambiguous_person_matching_baseline_kept"
    return True, "accepted_shared_human_boundary"


def _interp_se3(transform: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    out = np.eye(4, dtype=np.float64)
    rot = Rotation.from_matrix(transform[:3, :3])
    out[:3, :3] = Rotation.from_rotvec(alpha * rot.as_rotvec()).as_matrix()
    out[:3, 3] = alpha * transform[:3, 3]
    return out


def apply_to_arrays(
    cameras: np.ndarray,
    meshes: list[Any],
    pointmaps: list[Any] | None,
    boundary_indices: list[int] | tuple[int, ...],
    config: AdaptiveJointConfig | None = None,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray] | None, list[dict[str, Any]]]:
    """Apply accepted boundaries to in-memory demo output arrays.

    ``cameras`` are camera-to-world [N,4,4].  ``meshes`` and ``pointmaps`` may
    contain torch tensors or NumPy arrays.  The returned values are NumPy
    arrays, making the function easy to use both in the demo and in CPU-only
    payload post-processing.
    """

    cfg = config or AdaptiveJointConfig()
    cameras_out = np.asarray(cameras, dtype=np.float64).copy()
    meshes_out = [_as_vertices(value).copy() for value in meshes]
    pointmaps_out = None
    if pointmaps is not None:
        pointmaps_out = [np.asarray(value, dtype=np.float64).copy() for value in pointmaps]
    records: list[dict[str, Any]] = []
    for boundary in sorted(set(int(v) for v in boundary_indices)):
        if boundary <= 0 or boundary >= len(meshes_out):
            records.append({"boundary_index": boundary, "accepted": False, "reason": "invalid_boundary"})
            continue
        transform, diag = estimate_shared_boundary(meshes_out[boundary - 1], meshes_out[boundary])
        accepted, reason = gate_boundary(diag, cfg)
        record = dict(diag)
        record.update({"boundary_index": boundary, "accepted": bool(accepted), "reason": reason, "alpha": float(cfg.alpha)})
        if accepted and transform is not None:
            applied = _interp_se3(transform, cfg.alpha)
            record["applied_rotation_deg"] = float(
                np.degrees(np.linalg.norm(Rotation.from_matrix(applied[:3, :3]).as_rotvec()))
            )
            record["applied_translation_m"] = float(np.linalg.norm(applied[:3, 3]))
            for index in range(boundary, len(meshes_out)):
                cameras_out[index] = applied @ cameras_out[index]
                meshes_out[index] = _transform_points(applied, meshes_out[index])
                if pointmaps_out is not None:
                    pointmaps_out[index] = _transform_points(applied, pointmaps_out[index])
        records.append(record)
    return cameras_out, meshes_out, pointmaps_out, records


def _root_regressor() -> np.ndarray:
    """Load the local SMPL-X root regressor lazily for the raw-reference path."""

    import smplx
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    model = smplx.create(
        str(repo_root / "src/models"),
        "smplx",
        gender="neutral",
        use_pca=False,
        flat_hand_mean=True,
        num_betas=10,
    ).eval()
    return model.J_regressor.detach().cpu().numpy().astype(np.float64)[0]


def _roots(vertices: np.ndarray, regressor: np.ndarray) -> np.ndarray:
    return np.einsum("v,nvk->nk", regressor, vertices)


def _match_raw_people(current: np.ndarray, raw: np.ndarray) -> tuple[tuple[int, ...], float, float]:
    if current.shape != raw.shape:
        raise ValueError(f"Raw/current shape mismatch: {current.shape} vs {raw.shape}")
    rows: list[tuple[float, tuple[int, ...]]] = []
    for permutation in itertools.permutations(range(current.shape[0])):
        residuals = []
        for i, j in enumerate(permutation):
            _, rms = _kabsch(
                current[i] - current[i].mean(0),
                raw[j] - raw[j].mean(0),
            )
            residuals.append(rms)
        rows.append((float(np.mean(residuals)), tuple(int(v) for v in permutation)))
    rows.sort(key=lambda row: row[0])
    margin = float(rows[1][0] - rows[0][0]) if len(rows) > 1 else float("inf")
    return rows[0][1], float(rows[0][0]), margin


def apply_with_raw_reference(
    cameras: np.ndarray,
    meshes: list[Any],
    raw_cameras: np.ndarray,
    raw_meshes: list[Any],
    pointmaps: list[Any] | None,
    boundary_indices: list[int] | tuple[int, ...],
    config: AdaptiveJointConfig | None = None,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray] | None, list[dict[str, Any]]]:
    """Joint camera/body update using a same-checkpoint raw shadow path.

    The B0 body supplies the cross-shot orientation and the raw shadow body
    supplies a second, independent root ray.  The body is rotated around its
    current BRTC root, while camera translation is solved from the average of
    B0/raw root rays.  This is the numerically stable singleton path used when
    low-texture B0 camera depth is unreliable.  A B0 geometry gate still
    decides whether the transaction happens; no GT or future frame is used.
    """

    cfg = config or AdaptiveJointConfig()
    cameras_out = np.asarray(cameras, dtype=np.float64).copy()
    raw_cameras = np.asarray(raw_cameras, dtype=np.float64)
    meshes_out = [_as_vertices(value).copy() for value in meshes]
    raw_values = [_as_vertices(value).copy() for value in raw_meshes]
    pointmaps_out = None if pointmaps is None else [np.asarray(value, dtype=np.float64).copy() for value in pointmaps]
    regressor = _root_regressor()
    records: list[dict[str, Any]] = []
    for boundary in sorted(set(int(v) for v in boundary_indices)):
        if boundary <= 0 or boundary >= len(meshes_out):
            records.append({"boundary_index": boundary, "accepted": False, "reason": "invalid_boundary"})
            continue
        transform, diagnostics = estimate_shared_boundary(meshes_out[boundary - 1], meshes_out[boundary])
        accepted, reason = gate_boundary(diagnostics, cfg)
        record = dict(diagnostics)
        record.update({"boundary_index": boundary, "accepted": bool(accepted), "reason": reason, "alpha": float(cfg.alpha), "update": "root_anchored_joint_camera_human"})
        if accepted and transform is not None:
            if boundary >= len(raw_values) or raw_values[boundary].shape != meshes_out[boundary].shape:
                record.update({"accepted": False, "reason": "raw_shadow_shape_mismatch_baseline_kept"})
                records.append(record)
                continue
            permutation, raw_rms, raw_margin = _match_raw_people(meshes_out[boundary], raw_values[boundary])
            if (
                raw_rms > float(cfg.max_vertex_rms_m)
                or raw_rms / max(float(diagnostics.get("body_extent_m", 1.0)), 1e-6)
                > float(cfg.max_normalized_rms)
                or raw_margin < float(cfg.min_permutation_margin_m)
            ):
                record.update({"accepted": False, "reason": "raw_shadow_human_residual_or_match_ambiguous_baseline_kept", "raw_shadow_rms_m": raw_rms, "raw_shadow_permutation_margin_m": raw_margin})
                records.append(record)
                continue
            alpha = float(np.clip(cfg.alpha, 0.0, 1.0))
            delta_r = Rotation.from_rotvec(alpha * Rotation.from_matrix(transform[:3, :3]).as_rotvec()).as_matrix()
            record.update({"raw_shadow_rms_m": float(raw_rms), "raw_shadow_permutation_margin_m": float(raw_margin), "applied_rotation_deg": float(np.degrees(np.linalg.norm(Rotation.from_matrix(delta_r).as_rotvec())))})
            for index in range(boundary, len(meshes_out)):
                if meshes_out[index].shape[0] != raw_values[min(index, len(raw_values) - 1)].shape[0]:
                    record["later_shape_fallback_index"] = index
                    continue
                current_vertices = meshes_out[index]
                if current_vertices.shape[0] == 0:
                    record.setdefault("later_empty_prediction_indices", []).append(index)
                    continue
                roots_current = _roots(current_vertices, regressor)
                root_raw = _roots(raw_values[index], regressor) if index < len(raw_values) else roots_current.copy()
                raw_perm, _, _ = _match_raw_people(current_vertices, raw_values[index]) if index < len(raw_values) and raw_values[index].shape == current_vertices.shape else (tuple(range(current_vertices.shape[0])), 0.0, 0.0)
                camera_old = cameras_out[index].copy()
                camera_new = camera_old.copy()
                camera_new[:3, :3] = delta_r @ camera_old[:3, :3]
                q_current = (camera_old[:3, :3].T @ (roots_current - camera_old[:3, 3]).T).T
                raw_pose = raw_cameras[index] if index < len(raw_cameras) else camera_old
                q_raw = (raw_pose[:3, :3].T @ (root_raw - raw_pose[:3, 3]).T).T
                q_raw = q_raw[list(raw_perm)]
                q_mean = 0.5 * (q_current + q_raw)
                camera_new[:3, 3] = np.mean(roots_current - (camera_new[:3, :3] @ q_mean.T).T, axis=0)
                cameras_out[index] = camera_new
                meshes_out[index] = (current_vertices - roots_current[:, None, :]) @ delta_r.T + roots_current[:, None, :]
                # Background follows the actual camera world update.  The
                # body is root-anchored by design; this is the joint camera/
                # human compromise for low-texture scenes.
                if pointmaps_out is not None:
                    world_update = camera_new @ np.linalg.inv(camera_old)
                    pointmaps_out[index] = _transform_points(world_update, pointmaps_out[index])
            records.append(record)
        else:
            records.append(record)
    return cameras_out, meshes_out, pointmaps_out, records
