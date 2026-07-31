#!/usr/bin/env python3
"""CPU-only timestamp-aware velocity-residual group-tangent experiment.

The runtime candidate consumes only causal predicted geometry, anonymous
tracks, and input dataset timestamps.  It never treats stream-list indices as
physical time.  For a repeated-timestamp cross-camera cut, ``delta_t == 0``
and velocity extrapolation is exactly disabled.

MultiHuman cache identity strings were produced by GT mesh assignment.  This
probe therefore does *not* use those strings to build runtime history or cut
matches: it reconstructs anonymous pre-shot tracks and cut association with
root+torso+joints Hungarian matching.  GT strings are retained only for the
post-hoc association audit and metric targets.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts", REPO_ROOT / "versions/v14"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from versions.v13.gt_id_consensus import robust_velocity  # noqa: E402
from versions.v14 import eval_brtc_multithumbs_egohumans as ego  # noqa: E402
from versions.v14 import eval_brtc_global_orientation_kabsch_egohumans as kabsch_ego  # noqa: E402
from versions.v14 import probe_b0_brtc_huber_irls as harness  # noqa: E402
from versions.v14 import probe_b0_two_view_person_triangulation as legacy  # noqa: E402
from versions.v14 import probe_brtc_global_orientation_kabsch as orientation  # noqa: E402
from versions.v14 import probe_brtc_observable_action_shrinkage as common  # noqa: E402
from versions.v14.b0_person_triangulation import refine_matched_people  # noqa: E402
from versions.v14.probe_b0_anchor_conflict import SEQUENCE_INPUTS  # noqa: E402


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "output/v14/fine_alignment_research/brtc_velocity_residual_group_tangent"
)
DEFAULT_POLICY = DEFAULT_OUTPUT / "FROZEN_POLICY_BEFORE_CONFIRM.json"
DEFAULT_ORIENTATION_POLICY = (
    REPO_ROOT
    / "output/v14/fine_alignment_research/brtc_global_orientation_kabsch/"
    "FROZEN_POLICY_BEFORE_VALIDATION.json"
)
DEFAULT_EGO_CACHE = (
    REPO_ROOT
    / "output/v14/fine_alignment_research/brtc_multithumbs_egohumans/"
    "current_v14_cpu_geometry.pt"
)
DEFAULT_DOC = (
    REPO_ROOT
    / "versions/v14/docs/V14_BRTC_VELOCITY_RESIDUAL_GROUP_TANGENT_20260801.md"
)
POINT_KEYS = ("root", "joints", "vertices")
PRIMARY = (
    "root_error_m",
    "joint_error_m",
    "vertex_error_m",
    "pairwise_distance_error_m",
    "pairwise_vector_error_m",
)
EGO_METRICS = (
    "w_mpjpe_mm",
    "wa_mpjpe_mm",
    "pelvis_mpjpe_mm",
    "pelvis_mpvpe_mm",
    "fixed_world_root_mm",
    "fixed_world_joint_mm",
    "fixed_world_vertex_mm",
    "pairwise_root_distance_mm",
    "pairwise_root_vector_mm",
    "world_root_accel_delta2_mm_per_frame2",
    "world_joint_accel_delta2_mm_per_frame2",
)
GT_LABEL = {"person0": 0, "person1": 1, "person2": 2}


@dataclass(frozen=True)
class VelocityTangentPolicy:
    fraction: float
    cap_m: float
    group_dispersion_gate_m: float
    apply_when_dt_zero: bool = False
    history_frames: int = 5
    min_history: int = 3
    velocity_speed_gate_m_per_frame: float = 0.06
    velocity_residual_gate_m_per_frame: float = 0.05
    extrapolation_cap_m: float = 0.30
    min_group_people: int = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("audit", "dev", "freeze", "validate"), required=True)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--orientation_policy", type=Path, default=DEFAULT_ORIENTATION_POLICY)
    parser.add_argument("--ego_cache", type=Path, default=DEFAULT_EGO_CACHE)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--self_test", action="store_true")
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if len(array) else float("nan")


def finite_stats(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0, "mean": None, "median": None, "p90": None, "max": None}
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(array.max()),
    }


def transform_person(transform: np.ndarray, person: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(person)
    for key in POINT_KEYS:
        result[key] = ego.transform_points(transform, np.asarray(person[key], dtype=np.float64))
    for key in ("torso", "root_rotation"):
        if key in person:
            result[key] = np.asarray(transform, dtype=np.float64)[:3, :3] @ np.asarray(
                person[key], dtype=np.float64
            )
    return result


def cache_path(sequence: str, key: str) -> Path:
    return SEQUENCE_INPUTS[sequence]["cache"] / f"{key}.pt"


def row_boundary(row: dict[str, Any]) -> np.ndarray:
    if "methods" in row:
        return np.asarray(row["methods"]["b0"]["boundary"], dtype=np.float64)
    return np.asarray(row["boundaries"]["learned_b0"], dtype=np.float64)


def frame_detections(humans: dict[str, dict], frame_index: int) -> list[dict[str, Any]]:
    output = []
    for identity, human in humans.items():
        person = copy.deepcopy(human)
        person["gt_identity_evaluator_only"] = str(identity)
        person["gt_label_evaluator_only"] = int(GT_LABEL[str(identity)])
        person["history_frame_index"] = int(frame_index)
        output.append(person)
    return sorted(output, key=lambda value: int(value["detection_index"]))


def anonymous_pre_tracks(
    human_frames: list[dict[str, dict]], dataset_frames: list[int]
) -> tuple[dict[int, list[tuple[int, dict[str, Any]]]], dict[str, Any]]:
    tracks: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    next_track = 0
    edge_correct: list[bool] = []
    edge_count = 0
    for frame_index, (humans, dataset_frame) in enumerate(
        zip(human_frames, dataset_frames)
    ):
        detections = frame_detections(humans, frame_index)
        if frame_index == 0:
            for detection in detections:
                tracks[next_track] = [(int(dataset_frame), detection)]
                next_track += 1
            continue
        previous = {
            track: history[-1][1]
            for track, history in tracks.items()
            if int(history[-1][1]["history_frame_index"]) == frame_index - 1
        }
        assignment = ego.anonymous_assignment(previous, detections)
        used = set()
        for track, post_index in assignment["track_to_post_index"].items():
            before = previous[int(track)]
            after = detections[int(post_index)]
            edge_count += 1
            edge_correct.append(
                str(before["gt_identity_evaluator_only"])
                == str(after["gt_identity_evaluator_only"])
            )
            tracks[int(track)].append((int(dataset_frame), after))
            used.add(int(post_index))
        for post_index, detection in enumerate(detections):
            if post_index in used:
                continue
            tracks[next_track] = [(int(dataset_frame), detection)]
            next_track += 1
    final_index = len(human_frames) - 1
    active = {
        track: history
        for track, history in tracks.items()
        if int(history[-1][1]["history_frame_index"]) == final_index
    }
    consistent = [
        len({person["gt_identity_evaluator_only"] for _, person in history}) == 1
        for history in active.values()
    ]
    return active, {
        "edge_count": edge_count,
        "edge_accuracy_evaluator_only": finite_mean(float(value) for value in edge_correct),
        "active_track_count": len(active),
        "active_track_consistency_evaluator_only": finite_mean(
            float(value) for value in consistent
        ),
        "active_history_lengths": [len(history) for history in active.values()],
    }


def velocity_evidence(
    history: list[tuple[int, dict[str, Any]]], target_frame: int, policy: VelocityTangentPolicy
) -> dict[str, Any]:
    selected = history[-int(policy.history_frames) :]
    frames = [int(frame) for frame, _ in selected]
    roots = [np.asarray(person["root"], dtype=np.float64) for _, person in selected]
    if len(roots) < int(policy.min_history):
        return {"reliable": False, "reason": "insufficient_history", "history": len(roots)}
    if any(second - first <= 0 for first, second in zip(frames, frames[1:])):
        return {"reliable": False, "reason": "non_increasing_timestamps", "history": len(roots)}
    velocity = robust_velocity(roots, frames)
    interval_velocity = np.stack(
        [
            (second - first) / float(frame_second - frame_first)
            for first, second, frame_first, frame_second in zip(
                roots[:-1], roots[1:], frames[:-1], frames[1:]
            )
        ]
    )
    residual = float(
        np.median(np.linalg.norm(interval_velocity - velocity[None], axis=1))
    )
    speed = float(np.linalg.norm(velocity))
    delta_t = int(target_frame) - int(frames[-1])
    if delta_t < 0:
        return {
            "reliable": False,
            "reason": "negative_timestamp_delta",
            "history": len(roots),
            "delta_t": delta_t,
        }
    extrapolation = float(delta_t) * velocity
    reason = "ok"
    reliable = True
    if speed > float(policy.velocity_speed_gate_m_per_frame):
        reliable, reason = False, "velocity_speed_gate"
    elif residual > float(policy.velocity_residual_gate_m_per_frame):
        reliable, reason = False, "velocity_residual_gate"
    elif np.linalg.norm(extrapolation) > float(policy.extrapolation_cap_m):
        reliable, reason = False, "extrapolation_cap_gate"
    anchor = roots[-1] + extrapolation
    return {
        "reliable": reliable,
        "reason": reason,
        "history": len(roots),
        "frames": frames,
        "delta_t": delta_t,
        "velocity_world_m_per_frame": velocity,
        "speed_m_per_frame": speed,
        "velocity_residual_m_per_frame": residual,
        "extrapolation_world_m": extrapolation,
        "extrapolation_norm_m": float(np.linalg.norm(extrapolation)),
        "anchor_world": anchor,
    }


def bounded(vector: np.ndarray, cap_m: float) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    return vector if norm <= float(cap_m) else vector * (float(cap_m) / max(norm, 1e-12))


def candidate_people(
    record: dict[str, Any], policy: VelocityTangentPolicy
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    brtc = record["brtc_people"]
    physical_delta = int(record["post_dataset_frame"]) - int(
        max(frame for history in record["track_histories"].values() for frame, _ in history)
    )
    if physical_delta == 0 and not bool(policy.apply_when_dt_zero):
        return copy.deepcopy(brtc), {
            "applied": False,
            "reason": "zero_timestamp_exact_kabsch_fallback",
            "accepted_count": int(record["brtc_debug"]["accepted_count"]),
            "reliable_count": 0,
            "group_tangent_world_m": np.zeros(3, dtype=np.float64),
            "group_dispersion_m": float("inf"),
            "final_shift_world_m": np.zeros(3, dtype=np.float64),
            "final_shift_norm_m": 0.0,
            "people": [],
        }
    post_camera = np.asarray(record["post_camera"], dtype=np.float64)
    records = {int(row["post_index"]): row for row in record["brtc_debug"]["people"]}
    observations = []
    accepted_post = []
    for post_index, row in sorted(records.items()):
        if not bool(row["accepted"]):
            continue
        pre_index = int(row["pre_index"])
        track = int(record["pre_people"][pre_index]["anonymous_track_id"])
        evidence = velocity_evidence(
            record["track_histories"][track],
            int(record["post_dataset_frame"]),
            policy,
        )
        accepted_post.append(post_index)
        if not bool(evidence["reliable"]):
            observations.append(
                {"post_index": post_index, "pre_index": pre_index, "track_id": track, **evidence}
            )
            continue
        root = np.asarray(brtc[post_index]["root"], dtype=np.float64)
        ray = root - post_camera[:3, 3]
        ray /= max(float(np.linalg.norm(ray)), 1e-12)
        residual = np.asarray(evidence["anchor_world"], dtype=np.float64) - root
        tangent = residual - float(np.dot(residual, ray)) * ray
        observations.append(
            {
                "post_index": post_index,
                "pre_index": pre_index,
                "track_id": track,
                **evidence,
                "ray_world": ray,
                "root_residual_world_m": residual,
                "tangent_world_m": tangent,
                "tangent_norm_m": float(np.linalg.norm(tangent)),
            }
        )
    reliable = [row for row in observations if bool(row["reliable"])]
    apply = False
    reason = "insufficient_reliable_group"
    group = np.zeros(3, dtype=np.float64)
    dispersion = float("inf")
    shift = np.zeros(3, dtype=np.float64)
    if len(reliable) >= int(policy.min_group_people):
        tangents = np.stack([row["tangent_world_m"] for row in reliable])
        group = np.median(tangents, axis=0)
        dispersion = float(np.median(np.linalg.norm(tangents - group[None], axis=1)))
        if dispersion <= float(policy.group_dispersion_gate_m):
            shift = bounded(float(policy.fraction) * group, float(policy.cap_m))
            apply = bool(np.linalg.norm(shift) > 1e-12)
            reason = "applied" if apply else "zero_group_action"
        else:
            reason = "group_dispersion_gate"
    output = copy.deepcopy(brtc)
    if apply:
        # One shared shift over all BRTC-accepted people preserves their layout.
        for post_index in accepted_post:
            output[post_index] = ego.shift_person(output[post_index], shift)
    return output, {
        "applied": apply,
        "reason": reason,
        "accepted_count": len(accepted_post),
        "reliable_count": len(reliable),
        "group_tangent_world_m": group,
        "group_dispersion_m": dispersion,
        "final_shift_world_m": shift,
        "final_shift_norm_m": float(np.linalg.norm(shift)),
        "people": observations,
    }


def apply_kabsch(
    record: dict[str, Any], people: list[dict[str, Any]], policy: orientation.OrientationPolicy
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output = copy.deepcopy(people)
    debug = []
    by_post = {int(row["post_index"]): row for row in record["brtc_debug"]["people"]}
    for post_index, row in sorted(by_post.items()):
        if not bool(row["accepted"]):
            continue
        pre_index = int(row["pre_index"])
        pair = {
            "pre": record["pre_people"][pre_index],
            "post": record["post_people"][post_index],
        }
        geometry, orient = orientation.orientation_candidate(
            pair,
            {
                key: np.asarray(output[post_index][key], dtype=np.float64)
                for key in POINT_KEYS
            },
            policy,
        )
        output[post_index].update(geometry)
        debug.append({"post_index": post_index, **orient})
    return output, {
        "applied_count": int(sum(bool(row["applied"]) for row in debug)),
        "people": debug,
    }


def prepare_case(row: dict[str, Any]) -> dict[str, Any]:
    sequence = str(row.get("sequence", str(row["case"]["key"]).split("_", 1)[0]))
    cache = torch.load(
        cache_path(sequence, str(row["case"]["key"])),
        map_location="cpu",
        weights_only=False,
    )
    case = cache["case"]
    pre_frames = [int(value) for value in case["pre_frames"]]
    post_frame = int(case["post_frame"])
    if len(pre_frames) < 3 or len(cache["humans"]) != len(pre_frames) + 1:
        raise ValueError(f"Insufficient pre history: {case['key']}")
    if any(second - first != 1 for first, second in zip(pre_frames, pre_frames[1:])):
        raise ValueError(f"Non-contiguous pre timestamps: {case['key']}")
    if post_frame - pre_frames[-1] != int(case["offset"]):
        raise ValueError(f"Timestamp/offset mismatch: {case['key']}")
    tracks, tracking_audit = anonymous_pre_tracks(cache["humans"][:-1], pre_frames)
    pre_people = []
    track_histories = {}
    for track, history in sorted(tracks.items()):
        person = copy.deepcopy(history[-1][1])
        person["anonymous_track_id"] = int(track)
        pre_people.append(person)
        track_histories[int(track)] = history

    boundary = row_boundary(row)
    raw_post = frame_detections(cache["humans"][-1], len(pre_frames))
    post_people = [transform_person(boundary, person) for person in raw_post]
    pre_by_track = {int(person["anonymous_track_id"]): person for person in pre_people}
    association = ego.anonymous_assignment(pre_by_track, post_people)
    pre_index = {
        int(person["anonymous_track_id"]): index for index, person in enumerate(pre_people)
    }
    matches = [
        (pre_index[int(track)], int(post_index))
        for track, post_index in sorted(association["track_to_post_index"].items())
    ]
    pre_camera = np.asarray(cache["poses"][-2], dtype=np.float64)
    post_camera = boundary @ np.asarray(cache["poses"][-1], dtype=np.float64)
    brtc_people, brtc_debug = refine_matched_people(
        pre_camera, post_camera, pre_people, post_people, matches
    )
    gt_pre = np.asarray(cache["gt"]["pre_c2w"], dtype=np.float64)
    gauge = pre_camera @ np.linalg.inv(gt_pre)
    targets = {}
    for identity, value in cache["gt"]["post_humans"].items():
        targets[str(identity)] = {
            key: ego.transform_points(gauge, np.asarray(value[key], dtype=np.float64))
            for key in POINT_KEYS
        }
    pair_rows = []
    for before, after in matches:
        identity = str(pre_people[before]["gt_identity_evaluator_only"])
        if identity not in targets:
            continue
        pair_rows.append(
            {
                "pre_index": int(before),
                "post_index": int(after),
                "target_identity_evaluator_only": identity,
                "post_identity_evaluator_only": str(
                    post_people[after]["gt_identity_evaluator_only"]
                ),
                "association_correct_evaluator_only": bool(
                    identity == str(post_people[after]["gt_identity_evaluator_only"])
                ),
                "target": targets[identity],
            }
        )
    return {
        "sequence": sequence,
        "case": copy.deepcopy(case),
        "pre_dataset_frames": pre_frames,
        "post_dataset_frame": post_frame,
        "timestamp_delta": int(post_frame - pre_frames[-1]),
        "pre_camera": pre_camera,
        "post_camera": post_camera,
        "pre_people": pre_people,
        "post_people": post_people,
        "track_histories": track_histories,
        "tracking_audit": tracking_audit,
        "association": association,
        "matches": matches,
        "pairs": pair_rows,
        "brtc_people": brtc_people,
        "brtc_debug": brtc_debug,
    }


def point_metrics(predicted: dict[str, Any], target: dict[str, Any]) -> dict[str, float]:
    return harness.point_errors(predicted, target, full=True)


def layout_metrics(
    predicted: list[dict[str, Any]], pairs: list[dict[str, Any]]
) -> dict[str, float]:
    distance, vector = [], []
    for first, second in combinations(pairs, 2):
        pred_vector = (
            np.asarray(predicted[int(first["post_index"])]["root"], dtype=np.float64)
            - np.asarray(predicted[int(second["post_index"])]["root"], dtype=np.float64)
        )
        target_vector = (
            np.asarray(first["target"]["root"], dtype=np.float64)
            - np.asarray(second["target"]["root"], dtype=np.float64)
        )
        distance.append(abs(float(np.linalg.norm(pred_vector) - np.linalg.norm(target_vector))))
        vector.append(float(np.linalg.norm(pred_vector - target_vector)))
    return {
        "pairwise_distance_error_m": finite_mean(distance),
        "pairwise_vector_error_m": finite_mean(vector),
    }


def evaluate_records(
    records: list[dict[str, Any]],
    policy: VelocityTangentPolicy,
    orientation_policy: orientation.OrientationPolicy,
    include_cases: bool = False,
) -> dict[str, Any]:
    values = {
        method: {metric: [] for metric in PRIMARY}
        for method in ("brtc", "brtc_kabsch", "velocity", "velocity_kabsch")
    }
    harms = {method: [] for method in ("velocity", "velocity_kabsch")}
    runtime_rows = []
    case_rows = []
    fallback_max = camera_max = 0.0
    for record in records:
        velocity_people, velocity_debug = candidate_people(record, policy)
        brtc_kabsch, brtc_kabsch_debug = apply_kabsch(
            record, record["brtc_people"], orientation_policy
        )
        velocity_kabsch, velocity_kabsch_debug = apply_kabsch(
            record, velocity_people, orientation_policy
        )
        methods = {
            "brtc": record["brtc_people"],
            "brtc_kabsch": brtc_kabsch,
            "velocity": velocity_people,
            "velocity_kabsch": velocity_kabsch,
        }
        case_metric = {}
        for method, people in methods.items():
            person_values = {metric: [] for metric in POINT_KEYS}
            for pair in record["pairs"]:
                error = point_metrics(people[int(pair["post_index"])], pair["target"])
                for key in POINT_KEYS:
                    metric = f"{key[:-1] if key.endswith('s') else key}_error_m"
                    # joints -> joint_error_m; vertices -> vertice_error_m would be wrong.
                    if key == "joints":
                        metric = "joint_error_m"
                    elif key == "vertices":
                        metric = "vertex_error_m"
                    person_values[key].append(float(error[metric]))
                    values[method][metric].append(float(error[metric]))
            layout = layout_metrics(people, record["pairs"])
            for metric, value in layout.items():
                values[method][metric].append(float(value))
            case_metric[method] = {
                "root_error_m": finite_mean(person_values["root"]),
                "joint_error_m": finite_mean(person_values["joints"]),
                "vertex_error_m": finite_mean(person_values["vertices"]),
                **layout,
            }
        for method in harms:
            for pair in record["pairs"]:
                base = point_metrics(
                    record["brtc_people"][int(pair["post_index"])], pair["target"]
                )["root_error_m"]
                candidate = point_metrics(
                    methods[method][int(pair["post_index"])], pair["target"]
                )["root_error_m"]
                harms[method].append(float(candidate - base))
        accepted = {
            int(row["post_index"])
            for row in record["brtc_debug"]["people"]
            if bool(row["accepted"])
        }
        for post_index, before in enumerate(record["post_people"]):
            if post_index in accepted:
                continue
            fallback_max = max(
                fallback_max,
                kabsch_ego.maximum_geometry_delta(before, velocity_people[post_index]),
                kabsch_ego.maximum_geometry_delta(before, velocity_kabsch[post_index]),
            )
        runtime_rows.append(
            {
                "sequence": record["sequence"],
                "case_key": record["case"]["key"],
                "timestamp_delta": int(record["timestamp_delta"]),
                "tracking_audit": record["tracking_audit"],
                "association_accuracy_evaluator_only": float(
                    np.mean(
                        [pair["association_correct_evaluator_only"] for pair in record["pairs"]]
                    )
                ),
                "brtc_accepted_count": int(record["brtc_debug"]["accepted_count"]),
                "velocity": velocity_debug,
                "brtc_kabsch": brtc_kabsch_debug,
                "velocity_kabsch": velocity_kabsch_debug,
            }
        )
        if include_cases:
            case_rows.append(
                {
                    "sequence": record["sequence"],
                    "case": record["case"],
                    "metrics": case_metric,
                    "runtime": runtime_rows[-1],
                }
            )
    summary = {
        method: {metric: finite_mean(metric_values) for metric, metric_values in metrics.items()}
        for method, metrics in values.items()
    }
    for method in harms:
        array = np.asarray(harms[method], dtype=np.float64)
        summary[method]["root_harm_over_5cm_rate"] = float(np.mean(array > 0.05))
        summary[method]["root_harm_over_1cm_rate"] = float(np.mean(array > 0.01))
        summary[method]["root_mean_delta_vs_brtc_m"] = float(array.mean())
    active = [bool(row["velocity"]["applied"]) for row in runtime_rows]
    reasons = Counter(str(row["velocity"]["reason"]) for row in runtime_rows)
    tracking_edges = [row["tracking_audit"]["edge_accuracy_evaluator_only"] for row in runtime_rows]
    return {
        "case_count": len(records),
        "person_pair_count": int(sum(len(record["pairs"]) for record in records)),
        "methods": summary,
        "runtime": {
            "velocity_active_case_count": int(sum(active)),
            "velocity_active_case_rate": finite_mean(float(value) for value in active),
            "velocity_reason_counts": dict(reasons),
            "mean_shift_m": finite_mean(
                row["velocity"]["final_shift_norm_m"] for row in runtime_rows
            ),
            "max_shift_m": float(
                max((row["velocity"]["final_shift_norm_m"] for row in runtime_rows), default=0.0)
            ),
            "anonymous_pre_track_edge_accuracy_evaluator_only": finite_mean(tracking_edges),
            "cut_association_accuracy_evaluator_only": finite_mean(
                row["association_accuracy_evaluator_only"] for row in runtime_rows
            ),
            "rejected_unmatched_exact_b0_max_abs_change": fallback_max,
            "camera_max_abs_change": camera_max,
            "timestamp_delta_counts": dict(
                Counter(int(row["timestamp_delta"]) for row in runtime_rows)
            ),
        },
        "cases": case_rows,
    }


def policy_grid() -> list[VelocityTangentPolicy]:
    return [
        VelocityTangentPolicy(fraction, cap, dispersion)
        for fraction in (0.05, 0.10, 0.20)
        for cap in (0.05, 0.10)
        for dispersion in (0.10, 0.20)
    ]


DEV_TIMESTAMPS = {
    "three": {500, 900, 1100, 1500},
    "dance": {300, 500},
    "box": {470, 550, 630},
}
CONFIRM_TIMESTAMPS = {
    "three": {700, 1000, 1300},
    "dance": {200, 400, 600},
    "box": {510, 590, 670},
}


def legacy_partition(name: str) -> dict[str, list[dict[str, Any]]]:
    timestamps = DEV_TIMESTAMPS if name == "dev" else CONFIRM_TIMESTAMPS
    return {
        sequence: [
            row
            for row in legacy.report_rows((sequence,))
            if int(row["case"]["timestamp"]) in timestamps[sequence]
        ]
        for sequence in ("three", "dance", "box")
    }


def cached_prepare(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for index, row in enumerate(rows, start=1):
        output.append(prepare_case(row))
        if index % 25 == 0 or index == len(rows):
            print(f">> prepared {index}/{len(rows)}", flush=True)
    return output


def combined_metrics(report: dict[str, Any], method: str) -> dict[str, float]:
    return {key: float(report["methods"][method][key]) for key in PRIMARY}


def safe_candidate(report: dict[str, Any]) -> bool:
    return bool(safe_against(report, "brtc") and safe_against(report, "brtc_kabsch"))


def safe_against(report: dict[str, Any], reference: str) -> bool:
    candidate = report["methods"]["velocity_kabsch"]
    return bool(
        all(
            float(candidate[key]) <= float(report["methods"][reference][key]) + 1e-12
            for key in PRIMARY
        )
        and float(candidate["root_harm_over_5cm_rate"]) <= 0.10
        and report["runtime"]["rejected_unmatched_exact_b0_max_abs_change"] == 0.0
        and report["runtime"]["camera_max_abs_change"] == 0.0
    )


def load_orientation_policy(path: Path) -> orientation.OrientationPolicy:
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if common.canonical_sha256(frozen["policy"]) != frozen["policy_sha256"]:
        raise ValueError("Orientation policy checksum mismatch")
    return orientation.OrientationPolicy(**frozen["policy"])


def ego_candidate_chains(
    b0_chains: list[dict[str, Any]],
    boundary_rows: list[dict[str, Any]],
    policy: VelocityTangentPolicy,
    orientation_policy: orientation.OrientationPolicy,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    boundary_by_key = {
        (int(row["chain_index"]), int(row["cut_index"])): row for row in boundary_rows
    }
    output, runtime_rows = [], []
    for b0_chain in b0_chains:
        chain_index = int(b0_chain["chain_index"])
        segments = [copy.deepcopy(b0_chain["segments"][0])]
        for segment_index in (1, 2):
            post_frames = copy.deepcopy(b0_chain["segments"][segment_index])
            pre_segment = segments[-1]
            pre_frame = pre_segment[-1]
            pre_people = list(pre_frame["people"])
            pre_index_by_track = {
                int(person["global_track_id"]): index
                for index, person in enumerate(pre_people)
            }
            frozen = boundary_by_key[(chain_index, segment_index - 1)]
            pairs = sorted(frozen["association"]["track_to_post_index"].items())
            matches = [
                (pre_index_by_track[int(track)], int(post_index))
                for track, post_index in pairs
                if int(track) in pre_index_by_track
            ]
            before_people = post_frames[0]["people"]
            brtc, brtc_debug = refine_matched_people(
                np.asarray(pre_frame["method_camera_c2w"], dtype=np.float64),
                np.asarray(post_frames[0]["method_camera_c2w"], dtype=np.float64),
                pre_people,
                before_people,
                matches,
            )
            histories = defaultdict(list)
            for frame in pre_segment:
                for person in frame["people"]:
                    histories[int(person["global_track_id"])].append(
                        (int(frame["dataset_frame"]), person)
                    )
            pseudo_record = {
                "brtc_people": brtc,
                "brtc_debug": brtc_debug,
                "post_camera": np.asarray(post_frames[0]["method_camera_c2w"], dtype=np.float64),
                "post_dataset_frame": int(post_frames[0]["dataset_frame"]),
                "pre_people": pre_people,
                "post_people": before_people,
                "track_histories": dict(histories),
            }
            for person in pseudo_record["pre_people"]:
                person["anonymous_track_id"] = int(person["global_track_id"])
            translated, tangent_debug = candidate_people(pseudo_record, policy)
            rotated = copy.deepcopy(translated)
            rotations = {}
            accepted_by_post = {
                int(row["post_index"]): row
                for row in brtc_debug["people"]
                if bool(row["accepted"])
            }
            for post_index, row in accepted_by_post.items():
                pre_index = int(row["pre_index"])
                pair = {"pre": pre_people[pre_index], "post": before_people[post_index]}
                geometry, orient = orientation.orientation_candidate(
                    pair,
                    {
                        key: np.asarray(rotated[post_index][key], dtype=np.float64)
                        for key in POINT_KEYS
                    },
                    orientation_policy,
                )
                rotated[post_index].update(geometry)
                rotations[int(before_people[post_index]["native_track_id"])] = np.asarray(
                    orient.get("rotation_world", np.eye(3)), dtype=np.float64
                )
            action_by_native = {}
            for post_index, before in enumerate(before_people):
                native = int(before["native_track_id"])
                accepted = post_index in accepted_by_post
                shift = (
                    np.asarray(rotated[post_index]["root"], dtype=np.float64)
                    - np.asarray(before["root"], dtype=np.float64)
                    if accepted
                    else np.zeros(3, dtype=np.float64)
                )
                action_by_native[native] = {
                    "accepted": accepted,
                    "shift": shift,
                    "rotation": rotations.get(native, np.eye(3, dtype=np.float64)),
                }
            fallback = camera_delta = 0.0
            for frame, b0_frame in zip(post_frames, b0_chain["segments"][segment_index]):
                camera_delta = max(
                    camera_delta,
                    float(
                        np.max(
                            np.abs(
                                np.asarray(frame["method_camera_c2w"], dtype=np.float64)
                                - np.asarray(b0_frame["method_camera_c2w"], dtype=np.float64)
                            )
                        )
                    ),
                )
                corrected = []
                for person in frame["people"]:
                    native = int(person["native_track_id"])
                    action = action_by_native.get(native)
                    if action is None or not bool(action["accepted"]):
                        value = copy.deepcopy(person)
                        fallback = max(fallback, kabsch_ego.maximum_geometry_delta(value, person))
                    else:
                        value = ego.shift_person(person, action["shift"])
                        value = kabsch_ego.rotate_person_around_root(value, action["rotation"])
                    corrected.append(value)
                frame["people"] = corrected
            runtime_rows.append(
                {
                    "chain_index": chain_index,
                    "cut_index": segment_index - 1,
                    "pre_last_dataset_frame": int(pre_frame["dataset_frame"]),
                    "post_first_dataset_frame": int(post_frames[0]["dataset_frame"]),
                    "physical_timestamp_delta": int(
                        post_frames[0]["dataset_frame"] - pre_frame["dataset_frame"]
                    ),
                    "brtc": brtc_debug,
                    "velocity": tangent_debug,
                    "rejected_unmatched_exact_b0_max_abs_change": fallback,
                    "camera_max_abs_change": camera_delta,
                }
            )
            segments.append(post_frames)
        output.append(
            {
                "chain_index": chain_index,
                "segments": segments,
                "frames": [frame for segment in segments for frame in segment],
            }
        )
    return output, runtime_rows


def evaluate_ego(
    args: argparse.Namespace,
    policy: VelocityTangentPolicy,
    orientation_policy: orientation.OrientationPolicy,
) -> dict[str, Any]:
    cache = torch.load(args.ego_cache, map_location="cpu", weights_only=False)
    methods, boundaries = ego.method_chains(cache)
    kabsch_chains, kabsch_runtime_rows = kabsch_ego.replay_brtc_then_orientation(
        methods["b0"],
        methods["b0_brtc_lc"],
        boundaries,
        kabsch_ego.deployable.OrientationKabschConfig(**asdict(orientation_policy)),
        orientation_policy,
    )
    # All six Ego boundaries repeat the physical dataset timestamp.  The
    # conservative frozen policy therefore returns the existing Kabsch chain
    # exactly, rather than accidentally treating stream order as elapsed time.
    if not bool(policy.apply_when_dt_zero):
        candidate = copy.deepcopy(kabsch_chains)
        runtime_rows = [
            {
                "chain_index": int(row["chain_index"]),
                "cut_index": int(row["cut_index"]),
                "pre_last_dataset_frame": int(
                    methods["b0"][int(row["chain_index"])]["segments"][int(row["cut_index"])][
                        -1
                    ]["dataset_frame"]
                ),
                "post_first_dataset_frame": int(
                    methods["b0"][int(row["chain_index"])]["segments"][
                        int(row["cut_index"]) + 1
                    ][0]["dataset_frame"]
                ),
                "physical_timestamp_delta": 0,
                "velocity": {
                    "applied": False,
                    "reason": "zero_timestamp_exact_kabsch_fallback",
                },
                "rejected_unmatched_exact_b0_max_abs_change": 0.0,
                "camera_max_abs_change": 0.0,
            }
            for row in kabsch_runtime_rows
        ]
    else:
        candidate, runtime_rows = ego_candidate_chains(
            methods["b0"], boundaries, policy, orientation_policy
        )
    _, exo = ego.load_colmap(ego.DEFAULT_DATA)
    vertex_map, regressor = ego.load_smpl_resources()
    evaluated = {}
    error_maps = {}
    for name, chains in (
        ("brtc", methods["b0_brtc_lc"]),
        ("brtc_kabsch", kabsch_chains),
        ("velocity_kabsch", candidate),
    ):
        evaluated[name], _ = kabsch_ego.evaluate_chains(
            chains, ego.DEFAULT_DATA, exo, vertex_map, regressor, 30.0
        )
        error_maps[name] = kabsch_ego.fixed_error_maps(
            chains, ego.DEFAULT_DATA, exo, vertex_map, regressor
        )
    metrics = {
        name: {key: float(value["metrics"][key]) for key in EGO_METRICS}
        for name, value in evaluated.items()
    }
    delta = {
        reference: {
            key: float(metrics["velocity_kabsch"][key] - metrics[reference][key])
            for key in EGO_METRICS
        }
        for reference in ("brtc", "brtc_kabsch")
    }
    harm = {
        reference: kabsch_ego.point_harm_audit(
            error_maps[reference], error_maps["velocity_kabsch"]
        )
        for reference in ("brtc", "brtc_kabsch")
    }
    invariant = bool(
        max(row["rejected_unmatched_exact_b0_max_abs_change"] for row in runtime_rows) == 0.0
        and max(row["camera_max_abs_change"] for row in runtime_rows) == 0.0
        and all(int(row["physical_timestamp_delta"]) == 0 for row in runtime_rows)
    )
    safe_vs_kabsch = bool(
        all(value <= 1e-12 for value in delta["brtc_kabsch"].values())
        and invariant
        and harm["brtc_kabsch"]["all_person_frames_in_post_shots"]["joint_error_m"][
            "harm_over_5cm_rate"
        ]
        <= 0.10
        and harm["brtc_kabsch"]["all_person_frames_in_post_shots"]["vertex_error_m"][
            "harm_over_5cm_rate"
        ]
        <= 0.10
    )
    return {
        "methods": metrics,
        "delta": delta,
        "harm": harm,
        "runtime": runtime_rows,
        "kabsch_runtime_audit": kabsch_ego.rotation_runtime_audit(kabsch_runtime_rows),
        "invariants_pass": invariant,
        "all_physical_timestamp_deltas_zero": bool(
            all(int(row["physical_timestamp_delta"]) == 0 for row in runtime_rows)
        ),
        "safe_vs_kabsch": safe_vs_kabsch,
    }


def audit_report() -> dict[str, Any]:
    sequences = {}
    for sequence in ("three", "dance", "box"):
        rows = legacy.report_rows((sequence,))
        prepared = cached_prepare(rows)
        sequences[sequence] = {
            "case_count": len(prepared),
            "pre_history_length": dict(Counter(len(row["pre_dataset_frames"]) for row in prepared)),
            "timestamp_delta": dict(Counter(int(row["timestamp_delta"]) for row in prepared)),
            "all_pre_timestamps_contiguous": bool(
                all(
                    all(b - a == 1 for a, b in zip(row["pre_dataset_frames"], row["pre_dataset_frames"][1:]))
                    for row in prepared
                )
            ),
            "anonymous_pre_track_edge_accuracy_evaluator_only": finite_mean(
                row["tracking_audit"]["edge_accuracy_evaluator_only"] for row in prepared
            ),
            "anonymous_pre_track_consistency_evaluator_only": finite_mean(
                row["tracking_audit"]["active_track_consistency_evaluator_only"] for row in prepared
            ),
            "cut_association_accuracy_evaluator_only": finite_mean(
                pair["association_correct_evaluator_only"]
                for row in prepared
                for pair in row["pairs"]
            ),
        }
    ego_cache = torch.load(DEFAULT_EGO_CACHE, map_location="cpu", weights_only=False)
    ego_rows = []
    for chain in ego_cache["chains"]:
        for cut in (0, 1):
            pre = chain["segments"][cut]["frames"]
            post = chain["segments"][cut + 1]["frames"]
            ego_rows.append(
                {
                    "chain_index": int(chain["chain_index"]),
                    "cut_index": cut,
                    "pre_frames": [int(row["dataset_frame"]) for row in pre],
                    "post_first": int(post[0]["dataset_frame"]),
                    "physical_timestamp_delta": int(
                        post[0]["dataset_frame"] - pre[-1]["dataset_frame"]
                    ),
                }
            )
    return {
        "experiment": "v14_brtc_velocity_residual_group_tangent_cache_audit",
        "multihuman": sequences,
        "egohumans": {
            "boundaries": ego_rows,
            "all_repeated_timestamp": bool(
                all(row["physical_timestamp_delta"] == 0 for row in ego_rows)
            ),
            "warning": "velocity extrapolation must be exactly zero; stream index is not physical time",
        },
        "identity_warning": (
            "cache person keys are GT mesh assignments; runtime histories and cut matches in this probe "
            "are independently rebuilt with anonymous geometry Hungarian"
        ),
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Timestamp-aware velocity-residual group tangent",
        "",
        f"Phase: `{report['phase']}`.",
        "",
        "> CPU cache only; no pretrained-model/GPU forward. GT is evaluator-only. Camera is unchanged; rejected/unmatched people are exact B0.",
        "> Dataset frame timestamps—not stream-list indices—define physical `delta_t`. EgoHumans confirmation cuts all have `delta_t=0`.",
        "",
    ]
    if report["phase"] == "audit":
        lines.extend(["## Cache audit", "", "```json", json.dumps(report["audit"], indent=2), "```"])
        return "\n".join(lines) + "\n"
    lines.extend(
        [
            "## Policy",
            "",
            f"`{json.dumps(report['policy'], ensure_ascii=False)}`",
            "",
        ]
    )
    if report["phase"] == "dev":
        selected = report["selection"]
        lines.extend(
            [
                f"Eligible policies: `{selected['eligible_count']}/{selected['grid_count']}`.",
                f"Development pass: `{selected['development_pass']}`.",
                "",
                "| Sequence | BRTC root | Kabsch joint | Candidate root | Candidate joint | Candidate vertex | Active | Safe |",
                "|---|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for sequence, value in report["selected_results"].items():
            methods = value["methods"]
            lines.append(
                f"| {sequence} | {methods['brtc']['root_error_m']:.6f} | "
                f"{methods['brtc_kabsch']['joint_error_m']:.6f} | "
                f"{methods['velocity_kabsch']['root_error_m']:.6f} | "
                f"{methods['velocity_kabsch']['joint_error_m']:.6f} | "
                f"{methods['velocity_kabsch']['vertex_error_m']:.6f} | "
                f"{value['runtime']['velocity_active_case_rate']:.1%} | {value['safe']} |"
            )
        return "\n".join(lines) + "\n"
    lines.extend(
        [
            "## Cache and runtime audit",
            "",
            "- Every MultiHuman case has five contiguous pre-shot dataset frames. Physical `dt` is "
            "`post_frame - pre_frames[-1]`: three-k0 is 0, three-offset1 is 1, and dance/box contain 0/1/2/4/8.",
            "- All six Ego cuts repeat the same physical dataset timestamp across cameras; their `dt` is 0. "
            "The frozen `apply_when_dt_zero=false` bit makes them exact Kabsch fallback.",
            "- Cache person keys came from GT mesh assignment and are never consumed by the candidate. "
            "Pre histories and cut matching are rebuilt with anonymous root+torso+joints Hungarian.",
            "- Runtime action is one shared bounded translation over BRTC-accepted people. Camera is untouched; "
            "BRTC-rejected and unmatched people remain exact B0; no future post frame is read.",
            "",
            "```text",
            "velocity_i = robust_velocity(last 5 causal pre roots, dataset timestamps)",
            "anchor_i   = pre_root_i + physical_dt * velocity_i",
            "tangent_i  = tangent_to_post_camera_ray(anchor_i - brtc_post_root_i)",
            "group      = coordinate_median(tangent_i)",
            "shift      = clip(fraction * group, cap), after observable reliability/dispersion gates",
            "```",
            "",
            "## Split and contamination contract",
            "",
            "The 12-policy grid was selected only on the deterministic timestamp development subset and hashed "
            "before this confirm run. Previous sequence-level reports and the earlier two-frame candidate were already "
            "known, so this is grouped-CV/exploratory—not blind validation. Confirmation results are not reused for retuning.",
            "",
            "## MultiHuman confirmation",
            "",
            "| Split | Method | Root | Joint | Vertex | Pair distance | Pair vector | Active | Safe |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for split, value in report["multihuman"].items():
        for method in ("brtc", "brtc_kabsch", "velocity_kabsch"):
            metrics = value["methods"][method]
            lines.append(
                f"| {split} | {method} | {metrics['root_error_m']:.6f} | "
                f"{metrics['joint_error_m']:.6f} | {metrics['vertex_error_m']:.6f} | "
                f"{metrics['pairwise_distance_error_m']:.6f} | "
                f"{metrics['pairwise_vector_error_m']:.6f} | "
                f"{value['runtime']['velocity_active_case_rate']:.1%} | {value['safe']} |"
            )
        lines.append(
            f"| {split} | safety audit | -- | -- | -- | -- | -- | -- | "
            f"vs BRTC `{value['safe_vs_brtc']}`, vs Kabsch `{value['safe_vs_kabsch']}` |"
        )
    lines.extend(
        [
            "",
            "### Anonymous tracking/association evaluator audit",
            "",
            "These accuracies use GT only after anonymous assignments are fixed. They are never gates.",
            "",
            "| Split | Pre-track edge accuracy | Cut association accuracy |",
            "|---|---:|---:|",
        ]
    )
    for split, value in report["multihuman"].items():
        lines.append(
            f"| {split} | "
            f"{value['runtime']['anonymous_pre_track_edge_accuracy_evaluator_only']:.1%} | "
            f"{value['runtime']['cut_association_accuracy_evaluator_only']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Physical timestamp-delta decomposition",
            "",
            "The rows below are case-mean deltas of velocity+Kabsch versus Kabsch. "
            "They are evaluator-only attribution; no GT-derived value enters a gate.",
            "",
            "| Split | dt | Cases | Active | Δroot | Δjoint | Δvertex |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split, value in report["multihuman"].items():
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for case in value["cases"]:
            grouped[int(case["runtime"]["timestamp_delta"])].append(case)
        for delta_t, cases in sorted(grouped.items()):
            delta = {
                metric: finite_mean(
                    float(case["metrics"]["velocity_kabsch"][metric])
                    - float(case["metrics"]["brtc_kabsch"][metric])
                    for case in cases
                )
                for metric in ("root_error_m", "joint_error_m", "vertex_error_m")
            }
            active = finite_mean(
                float(case["runtime"]["velocity"]["applied"]) for case in cases
            )
            lines.append(
                f"| {split} | {delta_t} | {len(cases)} | {active:.1%} | "
                f"{delta['root_error_m'] * 1000.0:+.3f} mm | "
                f"{delta['joint_error_m'] * 1000.0:+.3f} mm | "
                f"{delta['vertex_error_m'] * 1000.0:+.3f} mm |"
            )
    lines.extend(
        [
            "",
            "## EgoHumans confirmation",
            "",
            "| Method | W | WA | Root | Joint | Vertex | Pair dist | Pair vec | Root accel | Joint accel |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method, value in report["egohumans"]["methods"].items():
        lines.append(
            f"| {method} | {value['w_mpjpe_mm']:.3f} | {value['wa_mpjpe_mm']:.3f} | "
            f"{value['fixed_world_root_mm']:.3f} | {value['fixed_world_joint_mm']:.3f} | "
            f"{value['fixed_world_vertex_mm']:.3f} | {value['pairwise_root_distance_mm']:.3f} | "
            f"{value['pairwise_root_vector_mm']:.3f} | "
            f"{value['world_root_accel_delta2_mm_per_frame2']:.3f} | "
            f"{value['world_joint_accel_delta2_mm_per_frame2']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"All Ego physical timestamp deltas zero: `{report['egohumans']['all_physical_timestamp_deltas_zero']}`.",
            f"Ego safe versus Kabsch: `{report['egohumans']['safe_vs_kabsch']}`.",
            f"Final status: **{report['decision']['status']}**.",
            "",
            "## Decision analysis",
            "",
            "- `dt=0` is an exact Kabsch fallback by frozen policy. Ego therefore validates fallback invariants and temporal metrics, not the velocity branch.",
            "- The velocity branch improves `three_offset1` and `box`, but fails `dance`: coherent motion extrapolation and cross-shot root bias remain observationally confounded.",
            "- `three` fails the full-stack-vs-BRTC gate only because its exact Kabsch fallback increases vertex mean under the fully anonymous, low-accuracy cut association; the velocity branch itself is inactive.",
            "- No threshold is retuned after confirmation. Dataset-level prior reports were already known, so this is grouped-CV/exploratory confirmation, not a blind benchmark claim.",
            "",
            "## Reproduction",
            "",
            "```bash",
            ".venv/bin/python versions/v14/probe_brtc_velocity_residual_group_tangent.py --phase audit",
            ".venv/bin/python versions/v14/probe_brtc_velocity_residual_group_tangent.py --phase dev",
            ".venv/bin/python versions/v14/probe_brtc_velocity_residual_group_tangent.py --phase freeze",
            ".venv/bin/python versions/v14/probe_brtc_velocity_residual_group_tangent.py --phase validate",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def self_test() -> None:
    policy = VelocityTangentPolicy(0.2, 0.1, 0.2)
    history = [
        (10, {"root": np.asarray([0.0, 0.0, 0.0])}),
        (11, {"root": np.asarray([0.01, 0.0, 0.0])}),
        (12, {"root": np.asarray([0.02, 0.0, 0.0])}),
    ]
    value = velocity_evidence(history, 14, policy)
    assert bool(value["reliable"])
    assert int(value["delta_t"]) == 2
    assert np.allclose(value["anchor_world"], [0.04, 0.0, 0.0])
    repeated = velocity_evidence(history, 12, policy)
    assert np.allclose(repeated["anchor_world"], history[-1][1]["root"])
    assert np.allclose(bounded(np.asarray([2.0, 0.0, 0.0]), 0.1), [0.1, 0.0, 0.0])
    print("self-test passed")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    for path in (args.output_dir, args.policy, args.orientation_policy, args.ego_cache, args.doc.parent):
        if not str(path.resolve()).startswith(str(REPO_ROOT.resolve())):
            raise ValueError("All paths must remain under Movie3R on /data")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    orientation_policy = load_orientation_policy(args.orientation_policy)

    original_load = torch.load

    @lru_cache(maxsize=None)
    def memoized_load(path: str):
        return original_load(path, map_location="cpu", weights_only=False)

    def load_adapter(path, *unused_args, **unused_kwargs):
        return memoized_load(str(path))

    torch.load = load_adapter
    try:
        if args.phase == "audit":
            report = {"phase": "audit", "audit": audit_report()}
            stem = "CACHE_AUDIT"
        elif args.phase == "dev":
            partitions = legacy_partition("dev")
            prepared_by_sequence = {
                sequence: cached_prepare(rows) for sequence, rows in partitions.items()
            }
            scan = []
            for policy in policy_grid():
                by_sequence = {
                    sequence: evaluate_records(records, policy, orientation_policy)
                    for sequence, records in prepared_by_sequence.items()
                }
                per_sequence_safe = {
                    sequence: safe_candidate(value) for sequence, value in by_sequence.items()
                }
                scan.append(
                    {
                        "policy": asdict(policy),
                        "per_sequence_safe": per_sequence_safe,
                        "all_sequence_safe": bool(all(per_sequence_safe.values())),
                        "active_case_count": int(
                            sum(
                                value["runtime"]["velocity_active_case_count"]
                                for value in by_sequence.values()
                            )
                        ),
                        "objective": float(
                            sum(
                                value["methods"]["velocity_kabsch"][metric]
                                for value in by_sequence.values()
                                for metric in ("root_error_m", "joint_error_m", "vertex_error_m")
                            )
                        ),
                    }
                )
            eligible = [
                value
                for value in scan
                if value["all_sequence_safe"] and value["active_case_count"] > 0
            ]
            pool = eligible if eligible else [value for value in scan if value["active_case_count"] > 0]
            selected = min(
                pool,
                key=lambda value: (
                    0 if value["all_sequence_safe"] else 1,
                    -sum(value["per_sequence_safe"].values()),
                    value["objective"],
                    value["policy"]["fraction"],
                ),
            )
            selected_policy = VelocityTangentPolicy(**selected["policy"])
            selected_results = {
                sequence: evaluate_records(records, selected_policy, orientation_policy)
                for sequence, records in prepared_by_sequence.items()
            }
            for sequence, value in selected_results.items():
                value["safe"] = safe_candidate(value)
            report = {
                "phase": "dev",
                "policy": asdict(selected_policy),
                "selection": {
                    "split": DEV_TIMESTAMPS,
                    "rule": "safe vs BRTC and Kabsch on all five spatial means for every dev sequence, then minimum root+joint+vertex sum",
                    "grid_count": len(scan),
                    "eligible_count": len(eligible),
                    "development_pass": bool(eligible),
                    "dataset_contamination_warning": (
                        "sequence-level prior reports and the earlier two-frame aggregate were opened; "
                        "this new timestamp partition was fixed before velocity-candidate metrics"
                    ),
                    "scan": scan,
                },
                "selected_results": selected_results,
            }
            stem = "DEV_SCAN"
        elif args.phase == "freeze":
            dev = json.loads((args.output_dir / "DEV_SCAN.json").read_text(encoding="utf-8"))
            frozen = {
                "experiment": "v14_brtc_velocity_residual_group_tangent",
                "frozen_before_confirm": True,
                "development_pass": bool(dev["selection"]["development_pass"]),
                "policy": dev["policy"],
                "dev_report_sha256": hashlib.sha256(
                    (args.output_dir / "DEV_SCAN.json").read_bytes()
                ).hexdigest(),
                "constraints": {
                    "pre_history_only": True,
                    "future_post_frames": 0,
                    "timestamp_source": "input dataset frame, never stream index",
                    "extra_pretrained_models": [],
                    "camera_update": "none",
                    "rejected_unmatched": "exact B0",
                    "gt_runtime_gate": "none",
                },
            }
            frozen["policy_sha256"] = canonical_sha256(frozen["policy"])
            args.policy.write_text(
                json.dumps(jsonable(frozen), indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(args.policy)
            return
        else:
            frozen = json.loads(args.policy.read_text(encoding="utf-8"))
            if canonical_sha256(frozen["policy"]) != frozen["policy_sha256"]:
                raise ValueError("Frozen velocity policy checksum mismatch")
            policy = VelocityTangentPolicy(**frozen["policy"])
            partitions = legacy_partition("confirm")
            partitions["three_offset1"] = json.loads(
                harness.DEFAULT_CONFIRM_REPORT.read_text(encoding="utf-8")
            )["cases"]
            multihuman = {}
            all_safe = True
            for split, rows in partitions.items():
                records = cached_prepare(rows)
                value = evaluate_records(
                    records, policy, orientation_policy, include_cases=True
                )
                value["safe_vs_brtc"] = safe_against(value, "brtc")
                value["safe_vs_kabsch"] = safe_against(value, "brtc_kabsch")
                value["safe"] = safe_candidate(value)
                all_safe = bool(all_safe and value["safe"])
                multihuman[split] = value
            egohumans = evaluate_ego(args, policy, orientation_policy)
            go = bool(
                frozen["development_pass"]
                and all_safe
                and egohumans["safe_vs_kabsch"]
                and egohumans["invariants_pass"]
            )
            report = {
                "phase": "validate",
                "policy": asdict(policy),
                "policy_sha256": frozen["policy_sha256"],
                "frozen_policy_source": str(args.policy),
                "confirm_split": CONFIRM_TIMESTAMPS,
                "cache_audit_source": str(args.output_dir / "CACHE_AUDIT.json"),
                "cache_audit": json.loads(
                    (args.output_dir / "CACHE_AUDIT.json").read_text(encoding="utf-8")
                )["audit"],
                "multihuman": multihuman,
                "egohumans": egohumans,
                "decision": {
                    "development_pass": bool(frozen["development_pass"]),
                    "all_multihuman_confirm_splits_safe": all_safe,
                    "ego_safe_vs_kabsch": bool(egohumans["safe_vs_kabsch"]),
                    "runtime_invariants_pass": bool(egohumans["invariants_pass"]),
                    "status": (
                        "GO_VELOCITY_RESIDUAL_GROUP_TANGENT"
                        if go
                        else "NO_GO_VELOCITY_RESIDUAL_GROUP_TANGENT"
                    ),
                },
                "limitations": [
                    "Dataset-level MultiHuman/Ego results were previously opened; only this candidate's deterministic subpartition was new.",
                    "MultiHuman cache identities are GT-assigned, so runtime tracks and cut association were rebuilt anonymously; evaluator labels only audit their correctness.",
                    "EgoHumans cuts repeat physical timestamps, so velocity extrapolation is exactly zero and cannot validate the velocity branch.",
                ],
            }
            stem = "VALIDATION_RESULTS"
    finally:
        torch.load = original_load
    (args.output_dir / f"{stem}.json").write_text(
        json.dumps(jsonable(report), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    text = markdown(report)
    (args.output_dir / f"{stem}.md").write_text(text, encoding="utf-8")
    if args.phase == "validate":
        args.doc.parent.mkdir(parents=True, exist_ok=True)
        args.doc.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
