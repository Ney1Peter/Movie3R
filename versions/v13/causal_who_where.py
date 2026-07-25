"""Frozen V13 geometry utilities for causal WHO-WHERE hypothesis probes.

Identity hypotheses only relabel anonymous Human3R detections.  Boundary
candidates and the final shared transform remain the frozen Phase-2 solver.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.spatial import cKDTree

from versions.v13 import gt_id_consensus as geometry


def transform_detection(boundary: np.ndarray, detection: dict) -> dict:
    boundary = np.asarray(boundary, dtype=np.float64)
    output = dict(detection)
    for name in ("root", "joints", "vertices"):
        value = np.asarray(detection[name], dtype=np.float64)
        points = value.reshape(-1, 3)
        output[name] = geometry.transform_points(boundary, points).reshape(value.shape)
    for name in ("torso", "root_rotation"):
        output[name] = boundary[:3, :3] @ np.asarray(
            detection[name], dtype=np.float64
        )
    return output


def transform_shot(shot: dict, boundary: np.ndarray) -> dict:
    """Apply one shared Boundary to camera, cloud, and every human."""
    output = dict(shot)
    output["frames"] = []
    for source in shot["frames"]:
        row = dict(source)
        row["pose"] = np.asarray(boundary, dtype=np.float64) @ np.asarray(
            source["pose"], dtype=np.float64
        )
        row["cloud"] = geometry.transform_points(
            boundary, np.asarray(source["cloud"], dtype=np.float64)
        ).astype(np.float32)
        row["detections"] = [
            transform_detection(boundary, detection)
            for detection in source["detections"]
        ]
        output["frames"].append(row)
    return output


def detections_by_index(frame: dict) -> dict[int, dict]:
    return {
        int(row["detection_index"]): dict(row) for row in frame["detections"]
    }


def update_geometry_history(
    history: dict[int, list[dict]], shot: dict, history_size: int = 5
) -> None:
    """Commit aligned per-track geometry with constant memory."""
    for frame in shot["frames"]:
        detections = detections_by_index(frame)
        external = np.asarray(frame["external_ids"], dtype=np.int64)
        for detection_index, track_id in enumerate(external):
            if int(track_id) < 0 or detection_index not in detections:
                continue
            values = history.setdefault(int(track_id), [])
            values.append(
                {
                    "dataset_frame": int(frame["dataset_frame"]),
                    "detection": dict(detections[detection_index]),
                }
            )
            history[int(track_id)] = values[-int(history_size) :]


def hypothesis_cache(
    previous_shot: dict,
    post_frame: dict,
    result: dict,
    slot_names: tuple[str, ...],
) -> dict:
    """Relabel geometry by hypothesis track slots without reading GT identity."""
    bank_ids = np.asarray(result["bank"]["track_ids"], dtype=np.int64)
    pairs = sorted(
        result["accepted_pairs"], key=lambda row: int(row["source_index"])
    )
    track_to_slot = {
        int(bank_ids[int(pair["source_index"])]): slot_names[index]
        for index, pair in enumerate(pairs[: len(slot_names)])
    }
    humans = []
    for frame in previous_shot["frames"]:
        detections = detections_by_index(frame)
        external = np.asarray(frame["external_ids"], dtype=np.int64)
        assigned = {}
        for detection_index, track_id in enumerate(external):
            slot = track_to_slot.get(int(track_id))
            if slot is not None and detection_index in detections:
                assigned[slot] = detections[detection_index]
        humans.append(assigned)
    post_detections = detections_by_index(post_frame)
    post = {}
    accepted = []
    for pair in pairs:
        source = int(pair["source_index"])
        target = int(pair["target_index"])
        track_id = int(bank_ids[source])
        slot = track_to_slot.get(track_id)
        if slot is not None and target in post_detections:
            post[slot] = post_detections[target]
            accepted.append(
                {
                    "slot": slot,
                    "track_id": track_id,
                    "post_detection_index": target,
                }
            )
    humans.append(post)
    pre_frames = [int(row["dataset_frame"]) for row in previous_shot["frames"]]
    return {
        "case": {
            "pre_frames": pre_frames,
            "post_frame": int(post_frame["dataset_frame"]),
        },
        "poses": [
            np.asarray(row["pose"], dtype=np.float64)
            for row in previous_shot["frames"]
        ]
        + [np.asarray(post_frame["pose"], dtype=np.float64)],
        "clouds": [
            np.asarray(row["cloud"], dtype=np.float32)
            for row in previous_shot["frames"]
        ]
        + [np.asarray(post_frame["cloud"], dtype=np.float32)],
        "humans": humans,
        "track_to_slot": track_to_slot,
        "accepted": accepted,
    }


def hypothesis_state_cache(
    geometry_history: dict[int, list[dict]],
    target_shot: dict,
    post_frame: dict,
    result: dict,
    slot_names: tuple[str, ...],
) -> dict:
    """Build a Boundary cache from persistent aligned human histories."""
    bank_ids = np.asarray(result["bank"]["track_ids"], dtype=np.int64)
    pairs = sorted(
        result["accepted_pairs"], key=lambda row: int(row["source_index"])
    )
    track_to_slot = {
        int(bank_ids[int(pair["source_index"])]): slot_names[index]
        for index, pair in enumerate(pairs[: len(slot_names)])
    }
    by_frame: dict[int, dict[str, dict]] = {}
    for track_id, slot in track_to_slot.items():
        for observation in geometry_history.get(track_id, []):
            by_frame.setdefault(int(observation["dataset_frame"]), {})[slot] = dict(
                observation["detection"]
            )
    pre_frames = sorted(by_frame)
    humans = [by_frame[frame] for frame in pre_frames]
    post_detections = detections_by_index(post_frame)
    post = {}
    accepted = []
    for pair in pairs:
        source = int(pair["source_index"])
        target = int(pair["target_index"])
        track_id = int(bank_ids[source])
        slot = track_to_slot.get(track_id)
        if slot is not None and target in post_detections:
            post[slot] = post_detections[target]
            accepted.append(
                {
                    "slot": slot,
                    "track_id": track_id,
                    "post_detection_index": target,
                }
            )
    humans.append(post)
    target_frames = target_shot["frames"]
    return {
        "case": {
            "pre_frames": pre_frames,
            "post_frame": int(post_frame["dataset_frame"]),
        },
        "poses": [
            np.asarray(row["pose"], dtype=np.float64) for row in target_frames
        ]
        + [np.asarray(post_frame["pose"], dtype=np.float64)],
        "clouds": [
            np.asarray(row["cloud"], dtype=np.float32) for row in target_frames
        ]
        + [np.asarray(post_frame["cloud"], dtype=np.float32)],
        "humans": humans,
        "track_to_slot": track_to_slot,
        "accepted": accepted,
    }


def fallback_solution(cache: dict) -> dict:
    initial = np.asarray(cache["poses"][-2]) @ np.linalg.inv(
        np.asarray(cache["poses"][-1])
    )
    valid = [cloud for cloud in cache["clouds"][:-1] if len(cloud)]
    target = np.concatenate(valid) if valid else np.empty((0, 3))
    refined, debug = geometry.fixed_refine(initial, cache["clouds"][-1], target)
    return {
        "rotation": refined[:3, :3],
        "translation": refined[:3, 3],
        "identities": (),
        "fallback": "identity-free camera-continuity Fixed Explicit",
        "fixed_debug": debug,
    }


def uniform_solution(cache: dict) -> tuple[dict, dict[str, dict]]:
    candidates = geometry.human_candidates(cache)
    identities = tuple(candidates)
    if len(identities) >= 2:
        rotation, translation = geometry.solve_consensus(
            candidates, identities, "mean_raw_t"
        )
        fallback = "multi-human"
    elif len(identities) == 1:
        candidate = candidates[identities[0]]
        rotation = candidate["rotation"]
        translation = candidate["translation"]
        fallback = "single-human"
    else:
        return fallback_solution(cache), candidates
    return {
        "rotation": rotation,
        "translation": translation,
        "identities": identities,
        "fallback": fallback,
    }, candidates


def _pairwise_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def pointmap_residual(cache: dict, solution: dict) -> float:
    valid = [cloud for cloud in cache["clouds"][:-1] if len(cloud)]
    source = np.asarray(cache["clouds"][-1], dtype=np.float64)
    if not valid or not len(source):
        return float("nan")
    target = np.concatenate(valid).astype(np.float64)
    boundary = geometry.make_transform(
        solution["rotation"], solution["translation"]
    )
    aligned = geometry.transform_points(boundary, source)
    forward = cKDTree(target).query(aligned, k=1)[0]
    backward = cKDTree(aligned).query(target, k=1)[0]
    return float(0.5 * (np.median(forward) + np.median(backward)))


def hypothesis_geometry(cache: dict) -> dict:
    solution, candidates = uniform_solution(cache)
    identities = tuple(candidates)
    rotations = [candidates[name]["rotation"] for name in identities]
    translations = [candidates[name]["translation"] for name in identities]
    rotation_dispersion = _pairwise_mean(
        [
            geometry.rotation_distance_deg(first, second)
            for first, second in combinations(rotations, 2)
        ]
    )
    translation_dispersion = _pairwise_mean(
        [
            float(np.linalg.norm(first - second))
            for first, second in combinations(translations, 2)
        ]
    )
    loo = []
    if len(identities) >= 2:
        for held_out in identities:
            support = tuple(name for name in identities if name != held_out)
            if len(support) == 1:
                candidate = candidates[support[0]]
                rotation = candidate["rotation"]
                translation = candidate["translation"]
            else:
                rotation, translation = geometry.solve_consensus(
                    candidates, support, "mean_raw_t"
                )
            residual = geometry.per_identity_residuals(
                candidates, identities, rotation, translation
            )[held_out]
            loo.append(
                residual["translation_m"] / 0.25
                + residual["rotation_deg"] / 10.0
                + residual["layout_m"] / 0.25
            )
    pointmap = pointmap_residual(cache, solution)
    missing_candidates = max(len(cache.get("accepted", [])) - len(candidates), 0)
    score = (
        rotation_dispersion / 20.0
        + translation_dispersion / 0.50
        + _pairwise_mean(loo)
        + (pointmap / 0.20 if np.isfinite(pointmap) else 5.0)
        + 20.0 * missing_candidates
    )
    return {
        "solution": solution,
        "candidate_count": len(candidates),
        "missing_candidate_count": int(missing_candidates),
        "rotation_dispersion_deg": rotation_dispersion,
        "translation_dispersion_m": translation_dispersion,
        "leave_one_out_score": _pairwise_mean(loo),
        "pointmap_residual_m": pointmap,
        "geometry_score": float(score),
    }


def camera_metrics(
    cache: dict,
    solution: dict,
    gt_pre_c2w: np.ndarray,
    gt_post_c2w: np.ndarray,
) -> dict:
    boundary = geometry.make_transform(
        solution["rotation"], solution["translation"]
    )
    pre_pose = np.asarray(cache["poses"][-2], dtype=np.float64)
    post_pose = np.asarray(cache["poses"][-1], dtype=np.float64)
    gauge = pre_pose @ np.linalg.inv(np.asarray(gt_pre_c2w, dtype=np.float64))
    target = gauge @ np.asarray(gt_post_c2w, dtype=np.float64)
    final = boundary @ post_pose
    translation = float(np.linalg.norm(final[:3, 3] - target[:3, 3]))
    rotation = geometry.rotation_error_deg(final, target)
    return {
        "camera_translation_error_m": translation,
        "camera_rotation_error_deg": rotation,
        "camera_composite": translation + 0.02 * rotation,
        "catastrophic": bool(translation > 2.0 or rotation > 45.0),
    }
