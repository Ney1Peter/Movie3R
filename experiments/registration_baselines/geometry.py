#!/usr/bin/env python3
"""Prediction-only rigid registration and shared-transform utilities.

The functions in this module never read ground truth.  They consume one
per-shot Human3R cache and estimate a single transform from shot B into the
coordinate system established by shot A.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


STABLE_JOINTS = (0, 1, 2, 3, 6, 9, 12, 13, 14, 16, 17)
BONES = (
    (0, 1), (0, 2), (0, 3), (3, 6), (6, 9), (9, 12),
    (12, 13), (12, 14), (13, 16), (14, 17),
)


@dataclass(frozen=True)
class RegistrationResult:
    transform: np.ndarray
    status: str
    failure_reason: str | None
    diagnostics: dict[str, Any]
    id_map: dict[int, int]


def identity_result(status: str, reason: str, **diagnostics: Any) -> RegistrationResult:
    return RegistrationResult(
        transform=np.eye(4, dtype=np.float64),
        status=status,
        failure_reason=reason,
        diagnostics=diagnostics,
        id_map={},
    )


def validate_transform(transform: np.ndarray, atol: float = 1e-5) -> np.ndarray:
    value = np.asarray(transform, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError("nonfinite or non-4x4 transform")
    if not np.allclose(value[3], (0.0, 0.0, 0.0, 1.0), atol=atol):
        raise ValueError("invalid homogeneous row")
    rotation = value[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=atol):
        raise ValueError("rotation is not orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=atol):
        raise ValueError("rotation is improper")
    return value


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    value = np.asarray(points)
    matrix = validate_transform(transform)
    return value @ matrix[:3, :3].T + matrix[:3, 3]


def apply_shared_transform(
    arrays: dict[str, np.ndarray], transform: np.ndarray, boundary: int,
    id_map: dict[int, int] | None = None,
) -> dict[str, np.ndarray]:
    """Apply one B-to-A transform to every post-cut geometric quantity."""

    matrix = validate_transform(transform)
    output = {key: np.asarray(value).copy() for key, value in arrays.items()}
    frame_count = int(output["cameras_c2w"].shape[0])
    if boundary <= 0 or boundary >= frame_count:
        raise ValueError(f"invalid boundary {boundary}/{frame_count}")
    output["cameras_c2w"][boundary:] = np.einsum(
        "ij,tjk->tik", matrix, output["cameras_c2w"][boundary:]
    )
    for key in ("joints_world", "vertices_world"):
        values = output[key][boundary:]
        finite = np.isfinite(values).all(axis=-1)
        mapped = values @ matrix[:3, :3].T + matrix[:3, 3]
        output[key][boundary:] = np.where(finite[..., None], mapped, values)
    if id_map:
        ids = output["persistent_ids"]
        for old, new in id_map.items():
            region = ids[boundary:]
            region[region == int(old)] = int(new)
    return output


def _kabsch(source: np.ndarray, target: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Kabsch inputs must be Nx3 with equal shape")
    if len(source) < 3 or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("insufficient/nonfinite Kabsch correspondences")
    if weights is None:
        weight = np.full(len(source), 1.0 / len(source), dtype=np.float64)
    else:
        weight = np.maximum(np.asarray(weights, dtype=np.float64), 1e-8)
        weight /= weight.sum()
    source_mean = (weight[:, None] * source).sum(axis=0)
    target_mean = (weight[:, None] * target).sum(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = (weight[:, None] * target_centered).T @ source_centered
    u, _, vt = np.linalg.svd(covariance)
    sign = np.ones(3, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0:
        sign[-1] = -1.0
    rotation = u @ np.diag(sign) @ vt
    translation = target_mean - rotation @ source_mean
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return validate_transform(transform)


def _shape_cost(first: np.ndarray, second: np.ndarray) -> float:
    ids = np.asarray(STABLE_JOINTS, dtype=np.int64)
    first = np.asarray(first)
    second = np.asarray(second)
    a, b = first[ids], second[ids]
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return float("inf")
    a = a - a[0]
    b = b - b[0]
    a_bones = np.asarray([np.linalg.norm(first[j] - first[i]) for i, j in BONES])
    b_bones = np.asarray([np.linalg.norm(second[j] - second[i]) for i, j in BONES])
    a_scale = max(float(np.median(a_bones)), 1e-4)
    b_scale = max(float(np.median(b_bones)), 1e-4)
    a_norm, b_norm = a / a_scale, b / b_scale
    try:
        transform = _kabsch(b_norm, a_norm)
        procrustes = float(np.linalg.norm(transform_points(b_norm, transform) - a_norm, axis=1).mean())
    except ValueError:
        return float("inf")
    bone_shape = float(np.mean(np.abs(a_bones / a_scale - b_bones / b_scale)))
    stature = abs(math.log(max(b_scale, 1e-5) / max(a_scale, 1e-5)))
    return 0.55 * procrustes + 0.30 * bone_shape + 0.15 * stature


def match_boundary_people(
    arrays: dict[str, np.ndarray], boundary: int, max_cost: float,
) -> tuple[list[tuple[int, int]], np.ndarray, dict[int, int]]:
    pre_valid = np.flatnonzero(np.asarray(arrays["valid"])[boundary - 1].astype(bool))
    post_valid = np.flatnonzero(np.asarray(arrays["valid"])[boundary].astype(bool))
    if not len(pre_valid) or not len(post_valid):
        return [], np.empty((len(pre_valid), len(post_valid))), {}
    joints = np.asarray(arrays["joints_world"])
    costs = np.asarray([
        [_shape_cost(joints[boundary - 1, i], joints[boundary, j]) for j in post_valid]
        for i in pre_valid
    ], dtype=np.float64)
    finite = np.isfinite(costs)
    safe = np.where(finite, costs, 1e6)
    rows, cols = linear_sum_assignment(safe)
    pairs = [
        (int(pre_valid[r]), int(post_valid[c]))
        for r, c in zip(rows.tolist(), cols.tolist())
        if finite[r, c] and costs[r, c] <= float(max_cost)
    ]
    ids = np.asarray(arrays["persistent_ids"])
    id_map = {
        int(ids[boundary, post]): int(ids[boundary - 1, pre])
        for pre, post in pairs
        if int(ids[boundary, post]) >= 0 and int(ids[boundary - 1, pre]) >= 0
    }
    return pairs, costs, id_map


def pelvis_translation(
    arrays: dict[str, np.ndarray], boundary: int, max_pair_cost: float,
) -> RegistrationResult:
    started = time.perf_counter()
    pairs, costs, id_map = match_boundary_people(arrays, boundary, max_pair_cost)
    if not pairs:
        return identity_result(
            "insufficient_people", "no reliable prediction-only boundary match",
            matched_person_count=0, person_match_costs=costs.tolist(),
            runtime_seconds=time.perf_counter() - started,
        )
    joints = np.asarray(arrays["joints_world"])
    shifts = np.stack([
        joints[boundary - 1, pre, 0] - joints[boundary, post, 0]
        for pre, post in pairs
    ])
    translation = np.median(shifts, axis=0)
    if not np.isfinite(translation).all():
        return identity_result(
            "nonfinite_transform", "pelvis translation is nonfinite",
            matched_person_count=len(pairs), runtime_seconds=time.perf_counter() - started,
        )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = translation
    return RegistrationResult(
        transform=transform, status="ok", failure_reason=None, id_map=id_map,
        diagnostics={
            "matched_person_count": len(pairs),
            "matched_pairs": [list(value) for value in pairs],
            "person_match_costs": costs.tolist(),
            "fit_residual": float(np.linalg.norm(shifts - translation, axis=1).mean()),
            "runtime_seconds": time.perf_counter() - started,
        },
    )


def human_joint_se3(
    arrays: dict[str, np.ndarray], boundary: int, max_pair_cost: float,
    inlier_threshold_m: float, ransac_trials: int, min_inlier_ratio: float,
    seed: int,
) -> RegistrationResult:
    started = time.perf_counter()
    pairs, costs, id_map = match_boundary_people(arrays, boundary, max_pair_cost)
    if not pairs:
        return identity_result(
            "insufficient_people", "no reliable prediction-only boundary match",
            matched_person_count=0, person_match_costs=costs.tolist(),
            runtime_seconds=time.perf_counter() - started,
        )
    joints = np.asarray(arrays["joints_world"])
    source, target = [], []
    for pre, post in pairs:
        source.append(joints[boundary, post, list(STABLE_JOINTS)])
        target.append(joints[boundary - 1, pre, list(STABLE_JOINTS)])
    source = np.concatenate(source, axis=0)
    target = np.concatenate(target, axis=0)
    finite = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    source, target = source[finite], target[finite]
    if len(source) < 6:
        return identity_result(
            "insufficient_joints", "fewer than six finite stable-joint correspondences",
            matched_person_count=len(pairs), correspondence_count=len(source),
            runtime_seconds=time.perf_counter() - started,
        )
    generator = np.random.default_rng(int(seed))
    best: tuple[int, float, np.ndarray, np.ndarray] | None = None
    for _ in range(int(ransac_trials)):
        sample = generator.choice(len(source), size=min(4, len(source)), replace=False)
        try:
            candidate = _kabsch(source[sample], target[sample])
        except ValueError:
            continue
        residual = np.linalg.norm(transform_points(source, candidate) - target, axis=1)
        inliers = residual <= float(inlier_threshold_m)
        score = (int(inliers.sum()), -float(np.median(residual)))
        if best is None or score > (best[0], best[1]):
            best = (score[0], score[1], candidate, inliers)
    if best is None or best[0] < 6:
        return identity_result(
            "ransac_failed", "no RANSAC hypothesis has six inliers",
            matched_person_count=len(pairs), correspondence_count=len(source),
            runtime_seconds=time.perf_counter() - started,
        )
    inliers = best[3]
    inlier_ratio = float(inliers.mean())
    if inlier_ratio < float(min_inlier_ratio):
        return identity_result(
            "ransac_failed", "prediction-only inlier ratio below frozen threshold",
            matched_person_count=len(pairs), correspondence_count=len(source),
            inlier_ratio=inlier_ratio, runtime_seconds=time.perf_counter() - started,
        )
    try:
        transform = _kabsch(source[inliers], target[inliers])
    except ValueError as exc:
        return identity_result(
            "ransac_failed", str(exc), matched_person_count=len(pairs),
            correspondence_count=len(source), inlier_ratio=inlier_ratio,
            runtime_seconds=time.perf_counter() - started,
        )
    residual = np.linalg.norm(transform_points(source, transform) - target, axis=1)
    return RegistrationResult(
        transform=transform, status="ok", failure_reason=None, id_map=id_map,
        diagnostics={
            "matched_person_count": len(pairs),
            "matched_pairs": [list(value) for value in pairs],
            "person_match_costs": costs.tolist(),
            "correspondence_count": len(source),
            "inlier_count": int(inliers.sum()),
            "inlier_ratio": inlier_ratio,
            "fit_residual": float(residual[inliers].mean()),
            "fit_residual_all": float(residual.mean()),
            "runtime_seconds": time.perf_counter() - started,
        },
    )


def scene_fpfh_icp(
    source_points: np.ndarray, target_points: np.ndarray, voxel_size: float,
    ransac_distance_factor: float, icp_distance_factor: float,
    min_scene_points: int, min_correspondences: int, min_fitness: float,
    max_inlier_rmse_m: float, random_seed: int = 20260915,
) -> RegistrationResult:
    """Register B (source) to A (target) with FPFH/RANSAC and point-to-plane ICP."""

    started = time.perf_counter()
    source_raw = np.asarray(source_points, dtype=np.float64)
    target_raw = np.asarray(target_points, dtype=np.float64)
    source_raw = source_raw[np.isfinite(source_raw).all(axis=1)]
    target_raw = target_raw[np.isfinite(target_raw).all(axis=1)]
    if min(len(source_raw), len(target_raw)) < int(min_scene_points):
        return identity_result(
            "insufficient_scene_points", "raw prediction-only scene cloud is too small",
            scene_point_count_pre=len(target_raw), scene_point_count_post=len(source_raw),
            runtime_seconds=time.perf_counter() - started,
        )
    try:
        import open3d as o3d

        # Open3D's feature RANSAC owns an internal RNG.  Seed it for each
        # independent case/candidate so reruns do not depend on process order.
        o3d.utility.random.seed(int(random_seed))

        def prepare(points: np.ndarray):
            cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            cloud = cloud.voxel_down_sample(float(voxel_size))
            cloud.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=float(voxel_size) * 2.5, max_nn=40
                )
            )
            feature = o3d.pipelines.registration.compute_fpfh_feature(
                cloud,
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=float(voxel_size) * 5.0, max_nn=100
                ),
            )
            return cloud, feature

        source, source_feature = prepare(source_raw)
        target, target_feature = prepare(target_raw)
        source_count, target_count = len(source.points), len(target.points)
        if min(source_count, target_count) < int(min_scene_points):
            return identity_result(
                "insufficient_scene_points", "voxelized scene cloud is too small",
                scene_point_count_pre=target_count, scene_point_count_post=source_count,
                runtime_seconds=time.perf_counter() - started,
            )
        ransac_threshold = float(voxel_size) * float(ransac_distance_factor)
        global_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source, target, source_feature, target_feature, True, ransac_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3,
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(ransac_threshold),
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 0.999),
        )
        global_correspondences = len(global_result.correspondence_set)
        if global_correspondences < int(min_correspondences):
            return identity_result(
                "ransac_failed", "FPFH/RANSAC correspondence set below frozen threshold",
                scene_point_count_pre=target_count, scene_point_count_post=source_count,
                correspondence_count=global_correspondences,
                inlier_ratio=float(global_result.fitness),
                fit_residual=float(global_result.inlier_rmse),
                runtime_seconds=time.perf_counter() - started,
            )
        icp_threshold = float(voxel_size) * float(icp_distance_factor)
        refined = o3d.pipelines.registration.registration_icp(
            source, target, icp_threshold, global_result.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
        )
        transform = validate_transform(np.asarray(refined.transformation))
        correspondence_count = len(refined.correspondence_set)
        fitness = float(refined.fitness)
        residual = float(refined.inlier_rmse)
        if correspondence_count < int(min_correspondences):
            return identity_result(
                "icp_failed", "ICP correspondence set below frozen threshold",
                scene_point_count_pre=target_count, scene_point_count_post=source_count,
                correspondence_count=correspondence_count, inlier_ratio=fitness,
                fit_residual=residual, runtime_seconds=time.perf_counter() - started,
            )
        if fitness < float(min_fitness) or residual > float(max_inlier_rmse_m):
            return identity_result(
                "icp_failed", "prediction-only ICP fitness/residual gate failed",
                scene_point_count_pre=target_count, scene_point_count_post=source_count,
                correspondence_count=correspondence_count, inlier_ratio=fitness,
                fit_residual=residual, runtime_seconds=time.perf_counter() - started,
            )
        return RegistrationResult(
            transform=transform, status="ok", failure_reason=None, id_map={},
            diagnostics={
                "scene_point_count_pre_raw": len(target_raw),
                "scene_point_count_post_raw": len(source_raw),
                "scene_point_count_pre": target_count,
                "scene_point_count_post": source_count,
                "global_correspondence_count": global_correspondences,
                "global_fitness": float(global_result.fitness),
                "global_residual": float(global_result.inlier_rmse),
                "correspondence_count": correspondence_count,
                "inlier_ratio": fitness,
                "fit_residual": residual,
                "runtime_seconds": time.perf_counter() - started,
            },
        )
    except ImportError as exc:
        return identity_result(
            "runtime_error", f"Open3D unavailable: {exc}",
            runtime_seconds=time.perf_counter() - started,
        )
    except Exception as exc:
        return identity_result(
            "runtime_error", f"{type(exc).__name__}: {exc}",
            runtime_seconds=time.perf_counter() - started,
        )


def rotation_degrees(transform: np.ndarray) -> float:
    rotation = validate_transform(transform)[:3, :3]
    return float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))))
