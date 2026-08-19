#!/usr/bin/env python3
"""Evaluator-only Harmony4D metrics for one compact Movie3R cache."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from versions.v15.harmony4d.dataset import (  # noqa: E402
    load_exo_calibrations,
    load_gt_people,
    projected_visibility,
)
from versions.v15.harmony4d.topology import CommonTopology  # noqa: E402


MIN_VISIBLE_VERTEX_FRACTION = 0.01
MAX_ASSIGNMENT_COST_M = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--runtime-report", type=Path)
    parser.add_argument("--extracted-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.asarray(points) @ np.asarray(transform)[:3, :3].T + np.asarray(transform)[:3, 3]


def rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first)[:3, :3].T @ np.asarray(second)[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def fit_similarity(target: np.ndarray, prediction: np.ndarray, allow_scale: bool = True) -> tuple[float, np.ndarray, np.ndarray]:
    target = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1, 3)
    if target.shape != prediction.shape or len(target) < 3:
        raise ValueError(f"Similarity fit shape mismatch: {target.shape}, {prediction.shape}")
    target_mean, prediction_mean = target.mean(0), prediction.mean(0)
    target_centered = target - target_mean
    prediction_centered = prediction - prediction_mean
    left, singular, right = np.linalg.svd(prediction_centered.T @ target_centered)
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0:
        right[-1] *= -1
        rotation = right.T @ left.T
    denominator = float(np.sum(prediction_centered**2))
    scale = float(singular.sum() / max(denominator, 1e-12)) if allow_scale else 1.0
    translation = target_mean - scale * (rotation @ prediction_mean)
    return scale, rotation, translation


def apply_similarity(points: np.ndarray, fit: tuple[float, np.ndarray, np.ndarray]) -> np.ndarray:
    scale, rotation, translation = fit
    return scale * (np.asarray(points) @ rotation.T) + translation


def pelvis(joints: np.ndarray) -> np.ndarray:
    return np.asarray(joints)[..., [1, 2], :].mean(axis=-2)


def procrustes_mpjpe(target: np.ndarray, prediction: np.ndarray) -> float:
    """Per-frame PA-MPJPE in metres, following Human3R's local protocol."""

    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    fit = fit_similarity(target, prediction, allow_scale=True)
    return float(np.linalg.norm(apply_similarity(prediction, fit) - target, axis=-1).mean())


def human3r_rte_percent(target_roots: np.ndarray, prediction_roots: np.ndarray) -> np.ndarray:
    """Human3R/WHAM RTE: rigid alignment and GT path-length normalization.

    This is the NumPy equivalent of ``eval/global_human/utils.py::compute_rte``:
    the full root trajectory is aligned without scale, per-frame translation
    error is divided by the sum of consecutive GT root displacements, and the
    result is reported in percent.
    """

    target = np.asarray(target_roots, dtype=np.float64).reshape(-1, 3)
    prediction = np.asarray(prediction_roots, dtype=np.float64).reshape(-1, 3)
    if len(target) < 3 or target.shape != prediction.shape:
        return np.empty(0, dtype=np.float64)
    aligned = apply_similarity(
        prediction, fit_similarity(target, prediction, allow_scale=False)
    )
    displacement = float(np.linalg.norm(np.diff(target, axis=0), axis=1).sum())
    if displacement <= 1e-12:
        return np.empty(0, dtype=np.float64)
    return 100.0 * np.linalg.norm(target - aligned, axis=1) / displacement


def human3r_jitter(joints: np.ndarray, fps: float) -> np.ndarray:
    """Human3R/WHAM third-difference jitter in their ``m/s^3 / 10`` unit."""

    value = np.asarray(joints, dtype=np.float64)
    if len(value) < 4:
        return np.empty(0, dtype=np.float64)
    third = value[3:] - 3.0 * value[2:-1] + 3.0 * value[1:-2] - value[:-3]
    return np.linalg.norm(third * float(fps) ** 3, axis=-1).mean(axis=-1) / 10.0


def foot_sliding_cm(
    target_vertices: np.ndarray,
    prediction_vertices: np.ndarray,
    contact_threshold_m_per_frame: float = 1e-2,
) -> np.ndarray:
    """WHAM/Human3R four-vertex foot sliding, expressed in centimetres.

    GT consecutive foot displacement below 1 cm/frame defines contact.  The
    reported error is the predicted displacement at those same contacts.  The
    common Harmony4D topology is SMPL-6890, so the official SMPL foot indices
    are directly applicable.
    """

    target = np.asarray(target_vertices, dtype=np.float64)
    prediction = np.asarray(prediction_vertices, dtype=np.float64)
    if target.shape != prediction.shape or len(target) < 2 or target.shape[-2] != 6890:
        return np.empty(0, dtype=np.float64)
    foot_indices = np.asarray([3216, 3387, 6617, 6787], dtype=np.int64)
    target_step = np.linalg.norm(
        np.diff(target[:, foot_indices], axis=0), axis=-1
    )
    prediction_step = np.linalg.norm(
        np.diff(prediction[:, foot_indices], axis=0), axis=-1
    )
    return 100.0 * prediction_step[target_step < float(contact_threshold_m_per_frame)]


def summarize(values: list[float] | np.ndarray, scale: float = 1.0) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)] * float(scale)
    if not len(array):
        return {"count": 0, "mean": None, "median": None, "p90": None, "p95": None, "std": None}
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "std": float(array.std()),
    }


def method_arrays(cache: np.lib.npyio.NpzFile, method: str) -> dict[str, np.ndarray]:
    prefix = method + "__"
    required = ("cameras_c2w", "vertices_world", "joints_world", "persistent_ids", "native_ids", "valid")
    missing = [prefix + key for key in required if prefix + key not in cache.files]
    if missing:
        raise KeyError(missing)
    return {key: np.asarray(cache[prefix + key]) for key in required}


def camera_coordinates(camera_c2w: np.ndarray, points_world: np.ndarray) -> np.ndarray:
    rotation = np.asarray(camera_c2w)[:3, :3]
    centre = np.asarray(camera_c2w)[:3, 3]
    return (np.asarray(points_world) - centre) @ rotation


def frame_assignment(
    pred_camera: np.ndarray,
    pred_joints_world: np.ndarray,
    gt_camera: np.ndarray,
    gt_joints_world: np.ndarray,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    if not len(pred_joints_world) or not len(gt_joints_world):
        return [], np.empty((len(pred_joints_world), len(gt_joints_world)))
    pred = camera_coordinates(pred_camera, pred_joints_world)
    target = camera_coordinates(gt_camera, gt_joints_world)
    pred_root, target_root = pelvis(pred), pelvis(target)
    pred_body = pred - pred_root[:, None]
    target_body = target - target_root[:, None]
    cost = np.zeros((len(pred), len(target)), dtype=np.float64)
    for row in range(len(pred)):
        for column in range(len(target)):
            root = float(np.linalg.norm(pred_root[row] - target_root[column]))
            body = float(np.linalg.norm(pred_body[row] - target_body[column], axis=1).mean())
            cost[row, column] = root + 0.25 * body
    rows, columns = linear_sum_assignment(cost)
    return [(int(row), int(column)) for row, column in zip(rows, columns)], cost


def shared_initial_fit(
    arrays: dict[str, np.ndarray],
    gt: dict[str, np.ndarray],
    assignments: list[list[tuple[int, int]]],
    allow_scale: bool = True,
) -> tuple[float, np.ndarray, np.ndarray]:
    target, prediction = [], []
    for frame in range(min(2, len(assignments))):
        for pred_index, gt_index in assignments[frame]:
            target.append(gt["joints_world"][frame, gt_index])
            prediction.append(arrays["joints_world"][frame, pred_index])
    if not target:
        raise ValueError("No initial matched people for shared world fit")
    return fit_similarity(np.stack(target), np.stack(prediction), allow_scale=allow_scale)


def identity_metrics(
    arrays: dict[str, np.ndarray],
    assignments: list[list[tuple[int, int]]],
    identities: list[str],
    gt_visible: np.ndarray,
) -> dict[str, Any]:
    tracks: dict[int, list[int | None]] = {index: [None] * len(assignments) for index in range(len(identities))}
    confusion: dict[tuple[int, int], int] = {}
    matched = 0
    for frame, pairs in enumerate(assignments):
        for pred_index, gt_index in pairs:
            track_id = int(arrays["persistent_ids"][frame, pred_index])
            if track_id < 0:
                continue
            tracks[gt_index][frame] = track_id
            confusion[(gt_index, track_id)] = confusion.get((gt_index, track_id), 0) + 1
            matched += 1
    switches = {}
    continuity_numerator = continuity_denominator = 0
    for gt_index, values in tracks.items():
        previous = None
        count = 0
        for value in values:
            if value is None:
                continue
            if previous is not None:
                continuity_denominator += 1
                if value == previous:
                    continuity_numerator += 1
                else:
                    count += 1
            previous = value
        switches[identities[gt_index]] = count
    predicted_ids = sorted({key[1] for key in confusion})
    matrix = np.zeros((len(identities), len(predicted_ids)), dtype=np.int64)
    for (gt_index, predicted_id), value in confusion.items():
        matrix[gt_index, predicted_ids.index(predicted_id)] = value
    if matrix.size:
        rows, columns = linear_sum_assignment(-matrix)
        idtp = int(matrix[rows, columns].sum())
    else:
        idtp = 0
    total_gt = int(np.asarray(gt_visible, dtype=bool).sum())
    total_pred = int(np.asarray(arrays["valid"]).sum())
    idfp, idfn = total_pred - idtp, total_gt - idtp
    idf1 = 2.0 * idtp / max(2 * idtp + idfp + idfn, 1)
    return {
        "ids_total": int(sum(switches.values())),
        "ids_per_gt": switches,
        "id_continuity": continuity_numerator / max(continuity_denominator, 1),
        "association_accuracy_best_global_mapping": idtp / max(matched, 1),
        "idf1": float(idf1),
        "idtp": idtp,
        "idfp": int(idfp),
        "idfn": int(idfn),
        "visible_gt_person_frames": total_gt,
    }


def evaluate_method(
    method: str,
    arrays: dict[str, np.ndarray],
    gt: dict[str, np.ndarray],
    identities: list[str],
    boundary: int,
    fps: float,
) -> dict[str, Any]:
    frame_count = len(gt["cameras_c2w"])
    assignments: list[list[tuple[int, int]]] = []
    assignment_costs = []
    matched_count = 0
    for frame in range(frame_count):
        valid = np.flatnonzero(arrays["valid"][frame].astype(bool))
        gt_valid = np.flatnonzero(gt["visible"][frame].astype(bool))
        pairs_local, costs = frame_assignment(
            arrays["cameras_c2w"][frame], arrays["joints_world"][frame, valid],
            gt["cameras_c2w"][frame], gt["joints_world"][frame, gt_valid],
        )
        pairs_local = [
            (row, column) for row, column in pairs_local
            if float(costs[row, column]) <= MAX_ASSIGNMENT_COST_M
        ]
        pairs = [(int(valid[row]), int(gt_valid[column])) for row, column in pairs_local]
        assignments.append(pairs)
        assignment_costs.extend(float(costs[row, column]) for row, column in pairs_local)
        matched_count += len(pairs)
    fit = shared_initial_fit(arrays, gt, assignments)
    fit_initial_se3 = shared_initial_fit(arrays, gt, assignments, allow_scale=False)
    aligned_centre = apply_similarity(arrays["cameras_c2w"][:, :3, 3], fit)
    aligned_rotation = np.einsum("ij,tjk->tik", fit[1], arrays["cameras_c2w"][:, :3, :3])
    aligned_camera = np.repeat(np.eye(4)[None], frame_count, axis=0)
    aligned_camera[:, :3, :3] = aligned_rotation
    aligned_camera[:, :3, 3] = aligned_centre

    mpjpe, pa_mpjpe, mpvpe = [], [], []
    fixed_root, fixed_joint, fixed_vertex = [], [], []
    fixed_root_by_frame = [[] for _ in range(frame_count)]
    fixed_joint_by_frame = [[] for _ in range(frame_count)]
    fixed_vertex_by_frame = [[] for _ in range(frame_count)]
    chrge, orientation = [], []
    chrge_by_frame = [[] for _ in range(frame_count)]
    pair_distance, pair_vector = [], []
    pair_distance_by_frame = [[] for _ in range(frame_count)]
    pair_vector_by_frame = [[] for _ in range(frame_count)]
    per_identity: dict[int, dict[str, list[Any]]] = {
        index: {"frames": [], "pred_joints": [], "gt_joints": [], "pred_vertices": [], "gt_vertices": []}
        for index in range(len(identities))
    }
    aligned_pred_root_by_frame: list[dict[int, np.ndarray]] = []
    aligned_pred_joint_by_frame: list[dict[int, np.ndarray]] = []
    aligned_pred_vertex_by_frame: list[dict[int, np.ndarray]] = []
    for frame, pairs in enumerate(assignments):
        roots_this, joints_this, vertices_this = {}, {}, {}
        pred_camera = arrays["cameras_c2w"][frame]
        gt_camera = gt["cameras_c2w"][frame]
        pred_camera_joints = camera_coordinates(pred_camera, arrays["joints_world"][frame])
        pred_camera_vertices = camera_coordinates(pred_camera, arrays["vertices_world"][frame])
        gt_camera_joints = camera_coordinates(gt_camera, gt["joints_world"][frame])
        gt_camera_vertices = camera_coordinates(gt_camera, gt["vertices_world"][frame])
        for pred_index, gt_index in pairs:
            pred_pelvis = pelvis(pred_camera_joints[pred_index])
            gt_pelvis = pelvis(gt_camera_joints[gt_index])
            mpjpe.append(float(np.linalg.norm(
                (pred_camera_joints[pred_index] - pred_pelvis)
                - (gt_camera_joints[gt_index] - gt_pelvis), axis=1
            ).mean()))
            pa_mpjpe.append(procrustes_mpjpe(
                gt_camera_joints[gt_index] - gt_pelvis,
                pred_camera_joints[pred_index] - pred_pelvis,
            ))
            mpvpe.append(float(np.linalg.norm(
                (pred_camera_vertices[pred_index] - pred_pelvis)
                - (gt_camera_vertices[gt_index] - gt_pelvis), axis=1
            ).mean()))
            pred_joints_aligned = apply_similarity(arrays["joints_world"][frame, pred_index], fit)
            pred_vertices_aligned = apply_similarity(arrays["vertices_world"][frame, pred_index], fit)
            pred_root_aligned = pelvis(pred_joints_aligned)
            gt_root = pelvis(gt["joints_world"][frame, gt_index])
            fixed_root.append(float(np.linalg.norm(pred_root_aligned - gt_root)))
            fixed_joint.append(float(np.linalg.norm(pred_joints_aligned - gt["joints_world"][frame, gt_index], axis=1).mean()))
            fixed_vertex.append(float(np.linalg.norm(pred_vertices_aligned - gt["vertices_world"][frame, gt_index], axis=1).mean()))
            fixed_root_by_frame[frame].append(fixed_root[-1])
            fixed_joint_by_frame[frame].append(fixed_joint[-1])
            fixed_vertex_by_frame[frame].append(fixed_vertex[-1])
            q_pred = aligned_camera[frame, :3, :3].T @ (pred_root_aligned - aligned_camera[frame, :3, 3])
            q_gt = gt_camera[:3, :3].T @ (gt_root - gt_camera[:3, 3])
            chrge.append(float(np.linalg.norm(q_pred - q_gt)))
            chrge_by_frame[frame].append(chrge[-1])
            pred_body = pred_joints_aligned - pred_root_aligned
            gt_body = gt["joints_world"][frame, gt_index] - gt_root
            covariance = pred_body.T @ gt_body
            left, _, right = np.linalg.svd(covariance)
            body_rotation = right.T @ left.T
            if np.linalg.det(body_rotation) < 0:
                right[-1] *= -1
                body_rotation = right.T @ left.T
            orientation.append(rotation_error_deg(np.eye(4), np.block([[body_rotation, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]])))
            roots_this[gt_index] = pred_root_aligned
            joints_this[gt_index] = pred_joints_aligned
            vertices_this[gt_index] = pred_vertices_aligned
            track = per_identity[gt_index]
            track["frames"].append(frame)
            track["pred_joints"].append(arrays["joints_world"][frame, pred_index])
            track["gt_joints"].append(gt["joints_world"][frame, gt_index])
            track["pred_vertices"].append(arrays["vertices_world"][frame, pred_index])
            track["gt_vertices"].append(gt["vertices_world"][frame, gt_index])
        for first in roots_this:
            for second in roots_this:
                if first >= second:
                    continue
                pred_vector = roots_this[first] - roots_this[second]
                gt_vector = pelvis(gt["joints_world"][frame, first]) - pelvis(gt["joints_world"][frame, second])
                pair_distance.append(abs(float(np.linalg.norm(pred_vector) - np.linalg.norm(gt_vector))))
                pair_vector.append(float(np.linalg.norm(pred_vector - gt_vector)))
                pair_distance_by_frame[frame].append(pair_distance[-1])
                pair_vector_by_frame[frame].append(pair_vector[-1])
        aligned_pred_root_by_frame.append(roots_this)
        aligned_pred_joint_by_frame.append(joints_this)
        aligned_pred_vertex_by_frame.append(vertices_this)

    w_errors, w_one_errors, wa_errors, accel_discrete, accel_physical = [], [], [], [], []
    rte_h3r, jitter_h3r, sliding_cm = [], [], []
    motion_rows = []
    for gt_index, track in per_identity.items():
        if not track["frames"]:
            continue
        frames = np.asarray(track["frames"], dtype=np.int64)
        prediction = np.stack(track["pred_joints"])
        target = np.stack(track["gt_joints"])
        initial_fit = fit_similarity(target[: min(2, len(target))], prediction[: min(2, len(prediction))])
        one_fit = fit_similarity(target[:1], prediction[:1]) if target[:1].size >= 9 else initial_fit
        full_fit = fit_similarity(target, prediction)
        w_errors.extend(np.linalg.norm(apply_similarity(prediction, initial_fit) - target, axis=2).mean(axis=1))
        w_one_errors.extend(np.linalg.norm(apply_similarity(prediction, one_fit) - target, axis=2).mean(axis=1))
        wa_errors.extend(np.linalg.norm(apply_similarity(prediction, full_fit) - target, axis=2).mean(axis=1))
        rte_h3r.extend(human3r_rte_percent(pelvis(target), pelvis(prediction)))
        aligned = apply_similarity(prediction, initial_fit)
        if len(frames) >= 3:
            contiguous = (frames[1:-1] - frames[:-2] == 1) & (frames[2:] - frames[1:-1] == 1)
            pred_acc = aligned[:-2] - 2 * aligned[1:-1] + aligned[2:]
            gt_acc = target[:-2] - 2 * target[1:-1] + target[2:]
            values = np.linalg.norm(pred_acc - gt_acc, axis=2).mean(axis=1)[contiguous]
            accel_discrete.extend(values)
            accel_physical.extend(values * fps**2)
        if len(frames) >= 4:
            contiguous4 = (
                (frames[1:-2] - frames[:-3] == 1)
                & (frames[2:-1] - frames[1:-2] == 1)
                & (frames[3:] - frames[2:-1] == 1)
            )
            jitter_h3r.extend(human3r_jitter(prediction, fps)[contiguous4])
        if len(frames) >= 2:
            contiguous2 = frames[1:] - frames[:-1] == 1
            predicted_vertices = np.stack(track["pred_vertices"])
            target_vertices = np.stack(track["gt_vertices"])
            # Evaluate each contiguous run so a detection gap is never treated
            # as one physical video frame by the contact test.
            starts = np.flatnonzero(np.r_[True, ~contiguous2])
            ends = np.r_[starts[1:], len(frames)]
            for start, end in zip(starts, ends):
                if end - start >= 2:
                    sliding_cm.extend(foot_sliding_cm(
                        target_vertices[start:end], predicted_vertices[start:end]
                    ))
        post_mask = frames >= boundary
        if int(post_mask.sum()) >= 2:
            pred_roots = pelvis(aligned[post_mask])
            gt_roots = pelvis(target[post_mask])
            pred_net = float(np.linalg.norm(pred_roots[-1] - pred_roots[0]))
            gt_net = float(np.linalg.norm(gt_roots[-1] - gt_roots[0]))
            pred_steps = np.linalg.norm(np.diff(pred_roots, axis=0), axis=1)
            gt_steps = np.linalg.norm(np.diff(gt_roots, axis=0), axis=1)
            pred_path = float(pred_steps.sum())
            gt_path = float(gt_steps.sum())
            pred_direction = pred_roots[-1] - pred_roots[0]
            gt_direction = gt_roots[-1] - gt_roots[0]
            direction_cosine = float(
                np.dot(pred_direction, gt_direction)
                / max(np.linalg.norm(pred_direction) * np.linalg.norm(gt_direction), 1e-9)
            )
            label = "static" if gt_net <= 0.05 else ("moving" if gt_net >= 0.10 else "intermediate")
            motion_rows.append({
                "identity": identities[gt_index], "label_evaluator_only": label,
                "gt_net_displacement_m": gt_net, "pred_net_displacement_m": pred_net,
                "motion_retention": pred_net / max(gt_net, 1e-9) if label == "moving" else None,
                "pred_drift_m": pred_net if label == "static" else None,
                "gt_path_length_m": gt_path,
                "pred_path_length_m": pred_path,
                "path_retention": pred_path / max(gt_path, 1e-9) if label == "moving" else None,
                "trajectory_direction_cosine": direction_cosine if label == "moving" else None,
                "pred_mean_speed_m_per_s": float(pred_steps.mean() * fps) if len(pred_steps) else None,
                "gt_mean_speed_m_per_s": float(gt_steps.mean() * fps) if len(gt_steps) else None,
                "pred_max_deviation_from_first_m": float(np.linalg.norm(pred_roots - pred_roots[0], axis=1).max()),
            })

    camera_translation = np.linalg.norm(aligned_centre - gt["cameras_c2w"][:, :3, 3], axis=1)
    camera_rotation = np.asarray([
        rotation_error_deg(aligned_camera[frame], gt["cameras_c2w"][frame])
        for frame in range(frame_count)
    ])
    camera_fit_sim3 = fit_similarity(gt["cameras_c2w"][:, :3, 3], arrays["cameras_c2w"][:, :3, 3])
    camera_fit_se3 = fit_similarity(
        gt["cameras_c2w"][:, :3, 3], arrays["cameras_c2w"][:, :3, 3], allow_scale=False
    )
    ate_sim3 = np.linalg.norm(apply_similarity(arrays["cameras_c2w"][:, :3, 3], camera_fit_sim3) - gt["cameras_c2w"][:, :3, 3], axis=1)
    ate_se3 = np.linalg.norm(apply_similarity(arrays["cameras_c2w"][:, :3, 3], camera_fit_se3) - gt["cameras_c2w"][:, :3, 3], axis=1)
    ate_metric_initial_se3 = np.linalg.norm(
        apply_similarity(arrays["cameras_c2w"][:, :3, 3], fit_initial_se3)
        - gt["cameras_c2w"][:, :3, 3], axis=1
    )
    camera_rpe_translation, camera_rpe_rotation = [], []
    for frame in range(1, frame_count):
        pred_relative = np.linalg.inv(aligned_camera[frame - 1]) @ aligned_camera[frame]
        gt_relative = np.linalg.inv(gt["cameras_c2w"][frame - 1]) @ gt["cameras_c2w"][frame]
        camera_rpe_translation.append(float(np.linalg.norm(pred_relative[:3, 3] - gt_relative[:3, 3])))
        camera_rpe_rotation.append(rotation_error_deg(pred_relative, gt_relative))

    seam = {"available": boundary > 0 and boundary < frame_count}
    if seam["available"]:
        pred_camera_delta = aligned_centre[boundary] - aligned_centre[boundary - 1]
        gt_camera_delta = gt["cameras_c2w"][boundary, :3, 3] - gt["cameras_c2w"][boundary - 1, :3, 3]
        seam["camera_translation_excess_m"] = float(np.linalg.norm(pred_camera_delta - gt_camera_delta))
        pred_relative = aligned_camera[boundary - 1, :3, :3].T @ aligned_camera[boundary, :3, :3]
        gt_relative = gt["cameras_c2w"][boundary - 1, :3, :3].T @ gt["cameras_c2w"][boundary, :3, :3]
        seam["camera_rotation_excess_deg"] = rotation_error_deg(
            np.block([[pred_relative, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]]),
            np.block([[gt_relative, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]]),
        )
        root_values, joint_values, vertex_values = [], [], []
        for gt_index in set(aligned_pred_root_by_frame[boundary - 1]).intersection(aligned_pred_root_by_frame[boundary]):
            pred_delta = aligned_pred_root_by_frame[boundary][gt_index] - aligned_pred_root_by_frame[boundary - 1][gt_index]
            gt_delta = pelvis(gt["joints_world"][boundary, gt_index]) - pelvis(gt["joints_world"][boundary - 1, gt_index])
            root_values.append(float(np.linalg.norm(pred_delta - gt_delta)))
            pred_joint_delta = aligned_pred_joint_by_frame[boundary][gt_index] - aligned_pred_joint_by_frame[boundary - 1][gt_index]
            gt_joint_delta = gt["joints_world"][boundary, gt_index] - gt["joints_world"][boundary - 1, gt_index]
            joint_values.append(float(np.linalg.norm(pred_joint_delta - gt_joint_delta, axis=1).mean()))
            pred_vertex_delta = aligned_pred_vertex_by_frame[boundary][gt_index] - aligned_pred_vertex_by_frame[boundary - 1][gt_index]
            gt_vertex_delta = gt["vertices_world"][boundary, gt_index] - gt["vertices_world"][boundary - 1, gt_index]
            vertex_values.append(float(np.linalg.norm(pred_vertex_delta - gt_vertex_delta, axis=1).mean()))
        seam["root_excess_m"] = float(np.mean(root_values)) if root_values else None
        seam["joint_excess_m"] = float(np.mean(joint_values)) if joint_values else None
        seam["vertex_excess_m"] = float(np.mean(vertex_values)) if vertex_values else None
        relative_values = []
        for gt_index in set(aligned_pred_root_by_frame[boundary - 1]).intersection(aligned_pred_root_by_frame[boundary]):
            pred_before = aligned_camera[boundary - 1, :3, :3].T @ (
                aligned_pred_root_by_frame[boundary - 1][gt_index] - aligned_camera[boundary - 1, :3, 3]
            )
            pred_after = aligned_camera[boundary, :3, :3].T @ (
                aligned_pred_root_by_frame[boundary][gt_index] - aligned_camera[boundary, :3, 3]
            )
            gt_before = gt["cameras_c2w"][boundary - 1, :3, :3].T @ (
                pelvis(gt["joints_world"][boundary - 1, gt_index]) - gt["cameras_c2w"][boundary - 1, :3, 3]
            )
            gt_after = gt["cameras_c2w"][boundary, :3, :3].T @ (
                pelvis(gt["joints_world"][boundary, gt_index]) - gt["cameras_c2w"][boundary, :3, 3]
            )
            relative_values.append(float(np.linalg.norm((pred_after - pred_before) - (gt_after - gt_before))))
        seam["camera_human_relative_excess_m"] = float(np.mean(relative_values)) if relative_values else None

    visible_gt = int(np.asarray(gt["visible"], dtype=bool).sum())
    predicted = int(arrays["valid"].sum())
    coverage = {
        "visible_gt_person_frames": visible_gt,
        "matched_person_frames": matched_count,
        "missed_person_frames": visible_gt - matched_count,
        "predicted_person_frames": predicted,
        "false_positive_detections": max(predicted - matched_count, 0),
        "coverage": matched_count / max(visible_gt, 1),
        "precision": matched_count / max(predicted, 1),
        "recall": matched_count / max(visible_gt, 1),
        "minimum_visible_vertex_fraction": MIN_VISIBLE_VERTEX_FRACTION,
        "maximum_assignment_cost_m": MAX_ASSIGNMENT_COST_M,
    }
    visibility_strata = {}
    fractions = np.asarray(gt["visible_fraction"], dtype=np.float64)
    for name, mask in {
        "high_visibility": fractions >= 0.50,
        "partial_visibility": (fractions >= 0.10) & (fractions < 0.50),
        "severe_occlusion_or_truncation": (fractions >= MIN_VISIBLE_VERTEX_FRACTION) & (fractions < 0.10),
    }.items():
        total = int(mask.sum())
        matched = sum(bool(mask[frame, gt_index]) for frame, pairs in enumerate(assignments) for _, gt_index in pairs)
        visibility_strata[name] = {
            "visible_gt_person_frames": total,
            "matched_person_frames": int(matched),
            "coverage": matched / max(total, 1),
        }
    coverage["visibility_strata"] = visibility_strata
    return {
        "method": method,
        "multi_thumbs_named_provisional": {
            "w_mpjpe_mm": summarize(w_errors, 1000.0),
            "w_mpjpe_one_frame_fit_mm": summarize(w_one_errors, 1000.0),
            "wa_mpjpe_mm": summarize(wa_errors, 1000.0),
            "mpjpe_mm": summarize(mpjpe, 1000.0),
            "pa_mpjpe_mm": summarize(pa_mpjpe, 1000.0),
            "mpvpe_mm": summarize(mpvpe, 1000.0),
            "accel_delta2_mm_per_frame2": summarize(accel_discrete, 1000.0),
            "accel_physical_m_per_s2": summarize(accel_physical, 1.0),
            "rte_h3r_percent": summarize(rte_h3r),
            # Compact Harmony caches do not retain SMPL root-orientation
            # parameters.  This prediction/GT torso Kabsch angle is therefore
            # an explicit joint-derived proxy, not an official HumanMM ROE.
            "roe_joint_proxy_deg": summarize(orientation),
            "jitter_h3r_m_per_s3_div10": summarize(jitter_h3r),
            "foot_sliding_cm": summarize(sliding_cm),
            "ate_sim3_m": summarize(ate_sim3),
            "ate_se3_m": summarize(ate_se3),
            "ate_metric_initial_se3_m": summarize(ate_metric_initial_se3),
        },
        "coverage": coverage,
        "identity": identity_metrics(arrays, assignments, identities, gt["visible"]),
        "camera": {
            "translation_m": summarize(camera_translation),
            "rotation_deg": summarize(camera_rotation),
            "first_post_translation_m": float(camera_translation[boundary]),
            "first_post_rotation_deg": float(camera_rotation[boundary]),
            "post_translation_m": summarize(camera_translation[boundary:]),
            "post_rotation_deg": summarize(camera_rotation[boundary:]),
            "rpe_translation_m": summarize(camera_rpe_translation),
            "rpe_rotation_deg": summarize(camera_rpe_rotation),
            "boundary_rpe_translation_m": camera_rpe_translation[boundary - 1] if boundary > 0 else None,
            "boundary_rpe_rotation_deg": camera_rpe_rotation[boundary - 1] if boundary > 0 else None,
        },
        "fixed_world": {
            "root_m": summarize(fixed_root),
            "joint_m": summarize(fixed_joint),
            "vertex_m": summarize(fixed_vertex),
            "first_post_root_m": summarize(fixed_root_by_frame[boundary]),
            "first_post_joint_m": summarize(fixed_joint_by_frame[boundary]),
            "first_post_vertex_m": summarize(fixed_vertex_by_frame[boundary]),
            "post_root_m": summarize([value for values in fixed_root_by_frame[boundary:] for value in values]),
            "post_joint_m": summarize([value for values in fixed_joint_by_frame[boundary:] for value in values]),
            "post_vertex_m": summarize([value for values in fixed_vertex_by_frame[boundary:] for value in values]),
        },
        "camera_human_relative": {
            "root_gauge_m": summarize(chrge),
            "body_orientation_deg": summarize(orientation),
            "first_post_root_gauge_m": summarize(chrge_by_frame[boundary]),
            "post_root_gauge_m": summarize([value for values in chrge_by_frame[boundary:] for value in values]),
        },
        "pairwise_layout": {
            "distance_m": summarize(pair_distance),
            "vector_m": summarize(pair_vector),
            "first_post_distance_m": summarize(pair_distance_by_frame[boundary]),
            "first_post_vector_m": summarize(pair_vector_by_frame[boundary]),
            "post_distance_m": summarize([value for values in pair_distance_by_frame[boundary:] for value in values]),
            "post_vector_m": summarize([value for values in pair_vector_by_frame[boundary:] for value in values]),
        },
        "cut_seam": seam,
        "within_shot_motion": motion_rows,
        "assignment_cost": summarize(assignment_costs),
        "shared_initial_sim3": {"scale": fit[0], "rotation": fit[1], "translation": fit[2]},
        "shared_initial_se3": {
            "scale": fit_initial_se3[0], "rotation": fit_initial_se3[1], "translation": fit_initial_se3[2]
        },
    }


def load_gt(record: dict[str, Any], extracted_root: Path, topology: CommonTopology) -> tuple[dict[str, np.ndarray], list[str]]:
    sequence_root = extracted_root.resolve() / str(record["capture_relative"])
    calibrations = load_exo_calibrations(sequence_root)
    frames = list(record["pre_frame_numbers"]) + list(record["post_frame_numbers"])
    cameras = [record["pre_camera"]] * len(record["pre_frame_numbers"]) + [record["post_camera"]] * len(record["post_frame_numbers"])
    first = load_gt_people(sequence_root, frames[0])
    identities = sorted(first)
    vertices, joints, visibility = [], [], []
    for frame, camera_name in zip(frames, cameras):
        people = load_gt_people(sequence_root, int(frame))
        if sorted(people) != identities:
            raise ValueError(f"GT identity set changes at frame {frame}: {sorted(people)} vs {identities}")
        frame_vertices = np.stack([people[identity]["vertices"] for identity in identities])
        vertices.append(frame_vertices)
        joints.append(topology.joints_from_smpl(frame_vertices))
        visibility.append([
            projected_visibility(people[identity]["vertices"], calibrations[camera_name])["visible_vertex_fraction"]
            for identity in identities
        ])
    visibility_fraction = np.asarray(visibility, dtype=np.float64)
    return {
        "cameras_c2w": np.stack([calibrations[camera].camera_to_world for camera in cameras]),
        "vertices_world": np.stack(vertices),
        "joints_world": np.stack(joints),
        "frames": np.asarray(frames),
        "visible_fraction": visibility_fraction,
        "visible": visibility_fraction >= MIN_VISIBLE_VERTEX_FRACTION,
    }, identities


def detector_metrics(runtime: dict[str, Any], boundary: int, detector_key: str) -> dict[str, Any]:
    detector = runtime["runtime"][detector_key]
    labels = np.asarray(detector["labels"], dtype=np.int64)
    target = np.zeros_like(labels)
    target[boundary] = 1
    tp = int(((labels == 1) & (target == 1)).sum())
    fp = int(((labels == 1) & (target == 0)).sum())
    fn = int(((labels == 0) & (target == 1)).sum())
    future = np.flatnonzero(labels[boundary:])
    delay = int(future[0]) if len(future) else None
    probabilities = np.zeros(len(labels), dtype=np.float64)
    probability_valid = np.zeros(len(labels), dtype=bool)
    for row in detector.get("rows", []):
        index = int(row["pair_idx"])
        probabilities[index] = float(row["prob"])
        probability_valid[index] = True
    brier = float(np.mean((probabilities[probability_valid] - target[probability_valid]) ** 2)) if probability_valid.any() else None
    return {
        "kind": detector_key,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "false_positive_rate_per_noncut_pair": fp / max(len(labels) - 1, 1),
        "detection_delay_frames": delay,
        "boundary_detected": bool(labels[boundary]),
        "first_positive_index": detector.get("first_positive_index"),
        "first_positive_is_boundary": detector.get("first_positive_index") == boundary,
        "brier": brier,
        "latency_seconds": detector.get("seconds"),
    }


def self_test() -> None:
    rng = np.random.default_rng(20260818)
    source = rng.normal(size=(4, 24, 3))
    angle = 0.4
    rotation = np.asarray([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    scale, translation = 1.7, np.asarray([0.4, -0.2, 1.1])
    target = scale * (source @ rotation.T) + translation
    fit = fit_similarity(target, source)
    np.testing.assert_allclose(apply_similarity(source, fit), target, atol=1e-10)
    camera = np.eye(4)
    camera[:3, :3] = rotation
    camera[:3, 3] = translation
    points = rng.normal(size=(10, 3))
    np.testing.assert_allclose(
        camera_coordinates(camera, transform_points(camera, points)), points, atol=1e-10
    )
    arrays = {
        "persistent_ids": np.asarray([[3, 4], [3, 4], [3, 5]]),
        "valid": np.ones((3, 2), dtype=np.uint8),
    }
    assignments = [[(0, 0), (1, 1)] for _ in range(3)]
    identity = identity_metrics(arrays, assignments, ["a", "b"], np.ones((3, 2), dtype=bool))
    assert identity["ids_total"] == 1
    assert 0 < identity["idf1"] < 1
    np.testing.assert_allclose(procrustes_mpjpe(target[0], source[0]), 0.0, atol=1e-10)
    straight = np.stack([np.arange(5), np.zeros(5), np.zeros(5)], axis=1)
    np.testing.assert_allclose(human3r_rte_percent(straight, straight), 0.0, atol=1e-10)
    np.testing.assert_allclose(human3r_jitter(np.repeat(straight[:, None], 24, axis=1), 30), 0.0)
    feet = np.zeros((3, 6890, 3), dtype=np.float64)
    np.testing.assert_allclose(foot_sliding_cm(feet, feet), 0.0)
    print("Harmony evaluator self-test passed")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    required = (args.cache, args.runtime_report, args.extracted_root, args.output)
    if any(value is None for value in required):
        raise ValueError("--cache, --runtime-report, --extracted-root and --output are required")
    runtime = json.loads(args.runtime_report.read_text(encoding="utf-8"))
    record = runtime["record"]
    topology = CommonTopology.load()
    gt, identities = load_gt(record, args.extracted_root, topology)
    results = {}
    with np.load(args.cache, allow_pickle=False) as cache:
        for method in runtime["methods"]:
            results[method] = evaluate_method(
                method, method_arrays(cache, method), gt, identities,
                int(record["boundary_index"]), float(record["fps"]),
            )
    adaptive = runtime["geometry"]["adaptive"]
    accepted = bool(adaptive and adaptive[0].get("accepted", False))
    parent = results["m6_b0_identity_brtc_c1"]
    candidate = results["m7_full_v15_oracle"]
    gate_harm = {}
    for key, path in {
        "camera_translation": ("camera", "translation_m", "mean"),
        "camera_rotation": ("camera", "rotation_deg", "mean"),
        "fixed_root": ("fixed_world", "root_m", "mean"),
        "fixed_vertex": ("fixed_world", "vertex_m", "mean"),
        "pair_layout": ("pairwise_layout", "vector_m", "mean"),
    }.items():
        first: Any = parent
        second: Any = candidate
        for component in path:
            first, second = first[component], second[component]
        gate_harm[key] = None if first is None or second is None else float(second - first)
    safe_gate = runtime["geometry"].get("observability_safe_brtc", {})
    safe_accepted = bool(safe_gate.get("accepted", False))
    safe_parent = results["m4_b0_identity"]
    safe_candidate = results.get("m10_observability_safe_oracle")
    safe_harm = {}
    if safe_candidate is not None:
        for key, path in {
            "w_mpjpe": ("multi_thumbs_named_provisional", "w_mpjpe_mm", "mean"),
            "camera_translation": ("camera", "translation_m", "mean"),
            "camera_rotation": ("camera", "rotation_deg", "mean"),
            "fixed_root": ("fixed_world", "root_m", "mean"),
            "fixed_vertex": ("fixed_world", "vertex_m", "mean"),
            "pair_layout": ("pairwise_layout", "vector_m", "mean"),
            "seam_root": ("cut_seam", "root_excess_m"),
        }.items():
            first: Any = safe_parent
            second: Any = safe_candidate
            for component in path:
                first, second = first[component], second[component]
            safe_harm[key] = None if first is None or second is None else float(second - first)
    report = {
        "schema_version": "Movie3R-Harmony4D-evaluation-v1",
        "protocol": record["protocol"],
        "case_id": record["case_id"],
        "record": record,
        "identities": identities,
        "methods": results,
        "detector": detector_metrics(runtime, int(record["boundary_index"]), "causal_gru_detector"),
        "detectors": {
            "causal_gru": detector_metrics(runtime, int(record["boundary_index"]), "causal_gru_detector"),
            "static_logistic": detector_metrics(runtime, int(record["boundary_index"]), "static_logistic_detector"),
        },
        "adaptive_gate": {
            "accepted": accepted,
            "diagnostics": adaptive,
            "harm_candidate_minus_parent": gate_harm,
            "harmful_accept": bool(accepted and any(value is not None and value > 0.05 for value in gate_harm.values())),
        },
        "observability_safe_gate": {
            "accepted": safe_accepted,
            "runtime_diagnostics": safe_gate,
            "candidate_minus_b0_identity": safe_harm,
            "any_metric_worse": bool(
                safe_accepted and any(value is not None and value > 0.0 for value in safe_harm.values())
            ),
            "catastrophic_harm": bool(
                safe_accepted and any(value is not None and value > 0.05 for value in safe_harm.values())
            ),
        },
        "evaluation_contract": {
            "gt_used_only_in_evaluator": True,
            "common_topology": topology.metadata(),
            "matching": "per-frame Hungarian in camera coordinates; GT identity never enters runtime",
            "literature_protocol_status": "Multi-THuMBS-named provisional; not official reproduction",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    concise = {
        method: {
            "W": value["multi_thumbs_named_provisional"]["w_mpjpe_mm"]["mean"],
            "WA": value["multi_thumbs_named_provisional"]["wa_mpjpe_mm"]["mean"],
            "MPJPE": value["multi_thumbs_named_provisional"]["mpjpe_mm"]["mean"],
            "MPVPE": value["multi_thumbs_named_provisional"]["mpvpe_mm"]["mean"],
            "ATE": value["multi_thumbs_named_provisional"]["ate_sim3_m"]["mean"],
            "IDs": value["identity"]["ids_total"],
            "coverage": value["coverage"]["coverage"],
        }
        for method, value in results.items()
    }
    print(json.dumps({"output": str(args.output), "case_id": record["case_id"], "metrics": concise}, indent=2))


if __name__ == "__main__":
    main()
