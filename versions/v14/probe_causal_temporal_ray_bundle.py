#!/usr/bin/env python3
"""Strictly causal temporal ray-bundle probe on frozen-B0 MultiHuman caches.

The probe never changes a camera and never reads a future post frame.  At post
arrival k, it may use the five already-seen pre frames and post arrivals <= k.
GT is used only after prediction for metrics.  Development is dance; the policy
is serialized before the box confirmation can be run.

This is a cache-only experiment: it does not rerun Human3R or another model.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from versions.v14.probe_b0_anchor_conflict import SEQUENCE_INPUTS
from versions.v14.probe_b0_two_view_person_triangulation import (
    accepted_action,
    person_evidence,
    transform_points,
)


OUTPUT_DIR = (
    REPO_ROOT
    / "output/v14/fine_alignment_research/causal_temporal_ray_bundle"
)
DEV_REPORT = (
    REPO_ROOT
    / "output/v14/b0_identity_matching_extended/dance/v14_b0_identity_matching.json"
)
CONFIRM_REPORT = (
    REPO_ROOT
    / "output/v14/b0_identity_matching_extended/box/v14_b0_identity_matching.json"
)
BRTC_POLICY = (
    REPO_ROOT
    / "output/v14/fine_alignment_research/two_view_person_triangulation/"
    "FROZEN_TRIANGULATION_POLICY_BEFORE_CONFIRM.json"
)
FROZEN_POLICY = OUTPUT_DIR / "FROZEN_POLICY_BEFORE_CONFIRM.json"
OFFSETS = (0, 1, 2)
METHODS = ("b0", "brtc_lc", "causal_bundle")
POINT_KEYS = ("root", "joints", "vertices")


@dataclass(frozen=True)
class TemporalPolicy:
    pre_history: int
    pre_aggregate: str
    post_filter: str
    post_history: int
    ema_alpha: float
    carry_on_reject: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("dev", "freeze", "confirm"))
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--self_test", action="store_true")
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def finite_stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {key: float("nan") for key in ("mean", "median", "p90", "p95", "max")}
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def point_errors(predicted: dict[str, np.ndarray], target: dict[str, np.ndarray]) -> dict[str, float]:
    joints = min(len(predicted["joints"]), len(target["joints"]))
    vertices = min(len(predicted["vertices"]), len(target["vertices"]))
    return {
        "root_error_m": float(np.linalg.norm(predicted["root"] - target["root"])),
        "joint_error_m": float(np.linalg.norm(
            predicted["joints"][:joints] - target["joints"][:joints], axis=1
        ).mean()),
        "vertex_error_m": float(np.linalg.norm(
            predicted["vertices"][:vertices] - target["vertices"][:vertices], axis=1
        ).mean()),
    }


def policy_grid() -> list[TemporalPolicy]:
    policies = []
    for pre_history in (1, 3, 5):
        for pre_aggregate in ("median", "huber"):
            for post_filter in ("median", "ema"):
                for post_history in (1, 2, 3):
                    alphas = (1.0,) if post_filter == "median" else (0.50, 0.75)
                    for alpha in alphas:
                        for carry in (False, True):
                            policies.append(TemporalPolicy(
                                pre_history=pre_history,
                                pre_aggregate=pre_aggregate,
                                post_filter=post_filter,
                                post_history=post_history,
                                ema_alpha=alpha,
                                carry_on_reject=carry,
                            ))
    return policies


def robust_center(values: list[float], mode: str) -> float:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return float("nan")
    if mode == "median":
        return float(np.median(array))
    if mode != "huber":
        raise ValueError(mode)
    center = float(np.median(array))
    scale = max(float(1.4826 * np.median(np.abs(array - center))), 0.02)
    for _ in range(8):
        residual = array - center
        weights = np.minimum(1.0, (1.345 * scale) / np.maximum(np.abs(residual), 1e-12))
        updated = float(np.sum(weights * array) / np.sum(weights))
        if abs(updated - center) < 1e-8:
            break
        center = updated
    return center


def case_group(row: dict[str, Any]) -> tuple[int, int, int]:
    case = row["case"]
    return (int(case["timestamp"]), int(case["source_camera"]), int(case["target_camera"]))


def complete_streams(report: dict[str, Any]) -> list[list[dict[str, Any]]]:
    groups: dict[tuple[int, int, int], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in report["cases"]:
        offset = int(row["case"]["offset"])
        if offset in OFFSETS:
            groups[case_group(row)][offset] = row
    return [
        [by_offset[offset] for offset in OFFSETS]
        for _, by_offset in sorted(groups.items())
        if all(offset in by_offset for offset in OFFSETS)
    ]


def prepare_arrival(
    sequence: str,
    row: dict[str, Any],
) -> dict[str, Any]:
    case = row["case"]
    cache_path = SEQUENCE_INPUTS[sequence]["cache"] / f"{case['key']}.pt"
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    boundary = np.asarray(row["boundaries"]["learned_b0"], dtype=np.float64)
    pre_camera = np.asarray(cache["poses"][-2], dtype=np.float64)
    raw_post_camera = np.asarray(cache["poses"][-1], dtype=np.float64)
    post_camera = boundary @ raw_post_camera
    gt_pre = np.asarray(cache["gt"]["pre_c2w"], dtype=np.float64)
    gt_post = np.asarray(cache["gt"]["post_c2w"], dtype=np.float64)
    gauge = pre_camera @ np.linalg.inv(gt_pre)
    target_camera = gauge @ gt_post
    association = row["matching"]["learned_b0"]["matchers"][
        "root_torso_joints"
    ]["predicted_identity_by_pre_identity"]
    identities = tuple(
        identity for identity in sorted(association)
        if identity in cache["humans"][-2]
        and association[identity] in cache["humans"][-1]
        and identity in cache["gt"]["post_humans"]
    )
    people = {}
    for identity in identities:
        post_identity = association[identity]
        post_raw = cache["humans"][-1][post_identity]
        target_raw = cache["gt"]["post_humans"][identity]
        post = {
            key: transform_points(boundary, np.asarray(post_raw[key], dtype=np.float64))
            for key in POINT_KEYS
        }
        target = {
            key: transform_points(gauge, np.asarray(target_raw[key], dtype=np.float64))
            for key in POINT_KEYS
        }
        pre_people = []
        pre_cameras = []
        for pre_person_frame, pose in zip(cache["humans"][:-1], cache["poses"][:-1]):
            if identity not in pre_person_frame:
                continue
            raw = pre_person_frame[identity]
            pre_people.append({
                key: np.asarray(raw[key], dtype=np.float64) for key in POINT_KEYS
            })
            pre_cameras.append(np.asarray(pose, dtype=np.float64))
        people[identity] = {
            "post_identity": post_identity,
            "association_correct": bool(post_identity == identity),
            "post": post,
            "target": target,
            "pre_people": pre_people,
            "pre_cameras": pre_cameras,
        }
    return {
        "case": case,
        "cache_path": cache_path,
        "boundary": boundary,
        "raw_post_camera": raw_post_camera,
        "post_camera": post_camera,
        "target_camera": target_camera,
        "people": people,
    }


def evidence_action(
    person: dict[str, Any],
    post_camera: np.ndarray,
    brtc_policy: dict[str, Any],
    history: int,
    aggregate: str,
) -> tuple[np.ndarray, bool, dict[str, Any]]:
    rows = []
    start = max(0, len(person["pre_people"]) - history)
    for pre, camera in zip(person["pre_people"][start:], person["pre_cameras"][start:]):
        evidence = person_evidence(pre, person["post"], camera, post_camera)
        action, accepted = accepted_action(evidence, brtc_policy)
        rows.append((evidence, action, accepted))
    accepted_actions = [action for _, action, accepted in rows if accepted]
    accepted = bool(accepted_actions)
    action = robust_center(accepted_actions, aggregate) if accepted else 0.0
    ray = person["post"]["root"] - post_camera[:3, 3]
    ray = ray / max(float(np.linalg.norm(ray)), 1e-12)
    debug = {
        "pre_evidence_count": len(rows),
        "accepted_pre_count": len(accepted_actions),
        "raw_actions_m": accepted_actions,
        "aggregated_action_m": action,
    }
    return action * ray, accepted, debug


def observable_consensus(
    people: dict[str, dict[str, Any]],
    proposals: dict[str, np.ndarray],
    accepted: dict[str, bool],
) -> tuple[dict[str, np.ndarray], float, np.ndarray]:
    accepted_shifts = [proposals[i] for i in people if accepted[i]]
    group = np.median(np.stack(accepted_shifts), axis=0) if accepted_shifts else np.zeros(3)
    objectives = {}
    for residual_lambda in (0.0, 0.25, 0.50, 0.75, 1.0):
        roots = {}
        for identity, person in people.items():
            shift = (
                group + residual_lambda * (proposals[identity] - group)
                if accepted[identity] else np.zeros(3)
            )
            roots[identity] = person["post"]["root"] + shift
        errors = []
        identities = tuple(people)
        for index, first in enumerate(identities):
            for second in identities[index + 1:]:
                first_pre = people[first]["pre_people"][-1]["root"]
                second_pre = people[second]["pre_people"][-1]["root"]
                errors.append(float(np.linalg.norm(
                    (roots[first] - roots[second]) - (first_pre - second_pre)
                )))
        objectives[residual_lambda] = float(np.mean(errors)) if errors else 0.0
    selected = min(objectives, key=objectives.get)
    output = {
        identity: (
            group + selected * (proposals[identity] - group)
            if accepted[identity] else np.zeros(3)
        )
        for identity in people
    }
    return output, float(selected), group


def temporal_filter(
    current: np.ndarray,
    accepted: bool,
    state: list[np.ndarray],
    policy: TemporalPolicy,
) -> tuple[np.ndarray, list[np.ndarray], bool]:
    next_state = list(state)
    if accepted:
        if policy.post_filter == "ema":
            if next_state:
                filtered = (
                    policy.ema_alpha * current
                    + (1.0 - policy.ema_alpha) * next_state[-1]
                )
            else:
                filtered = current.copy()
            next_state.append(filtered)
            output = filtered
        else:
            next_state.append(current.copy())
            output = np.median(np.stack(next_state[-policy.post_history:]), axis=0)
        next_state = next_state[-policy.post_history:]
        return output, next_state, True
    if policy.carry_on_reject and next_state:
        return next_state[-1].copy(), next_state, True
    return np.zeros(3, dtype=np.float64), next_state, False


def shifted(person: dict[str, Any], shift: np.ndarray) -> dict[str, np.ndarray]:
    return {key: person["post"][key] + shift for key in POINT_KEYS}


def evaluate_stream(
    sequence: str,
    rows: list[dict[str, Any]],
    temporal_policy: TemporalPolicy,
    brtc_policy: dict[str, Any],
    prepared_arrivals: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    arrivals = (
        prepared_arrivals
        if prepared_arrivals is not None
        else [prepare_arrival(sequence, row) for row in rows]
    )
    temporal_state: dict[str, list[np.ndarray]] = defaultdict(list)
    frames = []
    for arrival in arrivals:
        people = arrival["people"]
        independent, independent_ok, independent_debug = {}, {}, {}
        bundled, bundled_ok, bundled_debug = {}, {}, {}
        for identity, person in people.items():
            shift, accepted, debug = evidence_action(
                person, arrival["post_camera"], brtc_policy, 1, "median"
            )
            independent[identity], independent_ok[identity] = shift, accepted
            independent_debug[identity] = debug
            shift, accepted, debug = evidence_action(
                person,
                arrival["post_camera"],
                brtc_policy,
                temporal_policy.pre_history,
                temporal_policy.pre_aggregate,
            )
            bundled[identity], bundled_ok[identity] = shift, accepted
            bundled_debug[identity] = debug
        independent, independent_lambda, independent_group = observable_consensus(
            people, independent, independent_ok
        )
        bundled, bundled_lambda, bundled_group = observable_consensus(
            people, bundled, bundled_ok
        )
        candidate, candidate_ok = {}, {}
        for identity in people:
            value, state, valid = temporal_filter(
                bundled[identity], bundled_ok[identity], temporal_state[identity], temporal_policy
            )
            temporal_state[identity] = state
            candidate[identity], candidate_ok[identity] = value, valid
        predictions = {method: {} for method in METHODS}
        person_rows = []
        for identity, person in people.items():
            predictions["b0"][identity] = shifted(person, np.zeros(3))
            predictions["brtc_lc"][identity] = shifted(person, independent[identity])
            predictions["causal_bundle"][identity] = shifted(person, candidate[identity])
            errors = {
                method: point_errors(predictions[method][identity], person["target"])
                for method in METHODS
            }
            person_rows.append({
                "identity": identity,
                "predicted_post_identity_evaluation_only": person["post_identity"],
                "association_correct_evaluation_only": person["association_correct"],
                "brtc_accepted": independent_ok[identity],
                "candidate_accepted_or_carried": candidate_ok[identity],
                "brtc_shift_world": independent[identity],
                "candidate_shift_world": candidate[identity],
                "independent_debug": independent_debug[identity],
                "bundle_debug": bundled_debug[identity],
                "errors": errors,
            })
        layouts = {}
        identities = tuple(people)
        for method in METHODS:
            distance, vector = [], []
            for index, first in enumerate(identities):
                for second in identities[index + 1:]:
                    predicted = (
                        predictions[method][first]["root"]
                        - predictions[method][second]["root"]
                    )
                    target = people[first]["target"]["root"] - people[second]["target"]["root"]
                    distance.append(abs(float(np.linalg.norm(predicted) - np.linalg.norm(target))))
                    vector.append(float(np.linalg.norm(predicted - target)))
            layouts[method] = {
                "pairwise_distance_error_m": float(np.mean(distance)) if distance else float("nan"),
                "pairwise_vector_error_m": float(np.mean(vector)) if vector else float("nan"),
            }
        frames.append({
            "case": arrival["case"],
            "camera": {
                "b0_translation_error_m": float(np.linalg.norm(
                    arrival["post_camera"][:3, 3] - arrival["target_camera"][:3, 3]
                )),
                "candidate_max_abs_change": float(np.max(np.abs(
                    arrival["post_camera"]
                    - arrival["boundary"] @ arrival["raw_post_camera"]
                ))),
            },
            "consensus": {
                "brtc_lambda": independent_lambda,
                "brtc_group_shift": independent_group,
                "bundle_lambda": bundled_lambda,
                "bundle_group_shift": bundled_group,
            },
            "people": person_rows,
            "layouts": layouts,
            "_predictions": predictions,
            "_targets": {identity: people[identity]["target"] for identity in people},
        })
    acceleration = {method: {"root": [], "joints": [], "vertices": []} for method in METHODS}
    correction_jitter = {"brtc_lc": [], "causal_bundle": []}
    common = set.intersection(*[set(frame["_targets"]) for frame in frames]) if frames else set()
    for identity in sorted(common):
        for method in METHODS:
            for key in POINT_KEYS:
                pred = [frame["_predictions"][method][identity][key] for frame in frames]
                target = [frame["_targets"][identity][key] for frame in frames]
                count = min(*(len(item) for item in pred), *(len(item) for item in target)) if key != "root" else 3
                if key == "root":
                    pred2 = pred[2] - 2.0 * pred[1] + pred[0]
                    target2 = target[2] - 2.0 * target[1] + target[0]
                    value = float(np.linalg.norm(pred2 - target2))
                else:
                    pred2 = pred[2][:count] - 2.0 * pred[1][:count] + pred[0][:count]
                    target2 = target[2][:count] - 2.0 * target[1][:count] + target[0][:count]
                    value = float(np.linalg.norm(pred2 - target2, axis=1).mean())
                acceleration[method][key].append(value)
        for method in ("brtc_lc", "causal_bundle"):
            shifts = []
            for frame in frames:
                row = next(person for person in frame["people"] if person["identity"] == identity)
                shifts.append(np.asarray(row[
                    "brtc_shift_world" if method == "brtc_lc" else "candidate_shift_world"
                ], dtype=np.float64))
            correction_jitter[method].append(float(np.linalg.norm(
                shifts[2] - 2.0 * shifts[1] + shifts[0]
            )))
    for frame in frames:
        frame.pop("_predictions")
        frame.pop("_targets")
    return {
        "stream": {
            "timestamp": rows[0]["case"]["timestamp"],
            "source_camera": rows[0]["case"]["source_camera"],
            "target_camera": rows[0]["case"]["target_camera"],
            "offsets": list(OFFSETS),
        },
        "frames": frames,
        "acceleration_m_per_frame2": acceleration,
        "correction_jitter_m_per_frame2": correction_jitter,
        "acceleration_identity_count": len(common),
    }


def summarize(streams: list[dict[str, Any]]) -> dict[str, Any]:
    frames = [frame for stream in streams for frame in stream["frames"]]
    people = [person for frame in frames for person in frame["people"]]
    output: dict[str, Any] = {
        "stream_count": len(streams),
        "frame_count": len(frames),
        "person_frame_count": len(people),
        "acceleration_identity_triple_count": int(sum(
            stream["acceleration_identity_count"] for stream in streams
        )),
        "association_accuracy": float(np.mean([
            person["association_correct_evaluation_only"] for person in people
        ])) if people else float("nan"),
        "camera_candidate_max_abs_change": float(max(
            (frame["camera"]["candidate_max_abs_change"] for frame in frames), default=0.0
        )),
    }
    for method in METHODS:
        method_rows = []
        for person in people:
            method_rows.append(person["errors"][method])
        output[method] = {
            metric: finite_stats([row[metric] for row in method_rows])
            for metric in ("root_error_m", "joint_error_m", "vertex_error_m")
        }
        for metric in ("pairwise_distance_error_m", "pairwise_vector_error_m"):
            output[method][metric] = finite_stats([
                frame["layouts"][method][metric] for frame in frames
            ])
        for key in POINT_KEYS:
            values = [
                value
                for stream in streams
                for value in stream["acceleration_m_per_frame2"][method][key]
            ]
            label = {"root": "root", "joints": "joint", "vertices": "vertex"}[key]
            output[method][f"{label}_acceleration_error_m_per_frame2"] = finite_stats(values)
        if method != "b0":
            output[method]["coverage"] = float(np.mean([
                person[
                    "brtc_accepted" if method == "brtc_lc" else "candidate_accepted_or_carried"
                ] for person in people
            ])) if people else float("nan")
            output[method]["root_harm_over_5cm_rate"] = float(np.mean([
                person["errors"][method]["root_error_m"]
                - person["errors"]["b0"]["root_error_m"] > 0.05
                for person in people
            ])) if people else float("nan")
            output[method]["correction_jitter_m_per_frame2"] = finite_stats([
                value
                for stream in streams
                for value in stream["correction_jitter_m_per_frame2"][method]
            ])
    for method in ("brtc_lc", "causal_bundle"):
        output[method]["relative_gain_vs_b0"] = {
            metric: float(1.0 - output[method][metric]["mean"] / output["b0"][metric]["mean"])
            for metric in (
                "root_error_m", "joint_error_m", "vertex_error_m",
                "pairwise_vector_error_m", "joint_acceleration_error_m_per_frame2",
            )
        }
    return output


def evaluate(
    sequence: str,
    report_path: Path,
    temporal_policy: TemporalPolicy,
    brtc_policy: dict[str, Any],
    prepared: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] | None = None,
) -> dict[str, Any]:
    rows = complete_streams(read_json(report_path))
    if prepared is None:
        prepared = []
        for index, stream_rows in enumerate(rows, start=1):
            print(
                f"[prepare {sequence} {index:03d}/{len(rows):03d}] "
                f"t={stream_rows[0]['case']['timestamp']} "
                f"c{stream_rows[0]['case']['source_camera']}->"
                f"c{stream_rows[0]['case']['target_camera']}",
                flush=True,
            )
            prepared.append((
                stream_rows,
                [prepare_arrival(sequence, row) for row in stream_rows],
            ))
    streams = []
    for stream_rows, arrivals in prepared:
        streams.append(evaluate_stream(
            sequence, stream_rows, temporal_policy, brtc_policy, arrivals
        ))
    return {
        "sequence": sequence,
        "report_source": str(report_path),
        "cache_source": str(SEQUENCE_INPUTS[sequence]["cache"]),
        "temporal_policy": asdict(temporal_policy),
        "summary": summarize(streams),
        "streams": streams,
    }


def selection_key(summary: dict[str, Any]) -> tuple[float, ...]:
    baseline = summary["brtc_lc"]
    candidate = summary["causal_bundle"]
    constraints = (
        candidate["root_error_m"]["mean"] <= 1.03 * baseline["root_error_m"]["mean"]
        and candidate["joint_error_m"]["mean"] <= 1.03 * baseline["joint_error_m"]["mean"]
        and candidate["vertex_error_m"]["mean"] <= 1.03 * baseline["vertex_error_m"]["mean"]
        and candidate["pairwise_vector_error_m"]["mean"]
        <= 1.05 * baseline["pairwise_vector_error_m"]["mean"]
        and candidate["coverage"] >= baseline["coverage"] - 0.05
        and candidate["root_harm_over_5cm_rate"]
        <= baseline["root_harm_over_5cm_rate"] + 0.02
    )
    return (
        0.0 if constraints else 1.0,
        candidate["joint_acceleration_error_m_per_frame2"]["mean"],
        candidate["root_error_m"]["mean"],
        candidate["pairwise_vector_error_m"]["mean"],
    )


def compact_dev_row(policy: TemporalPolicy, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": asdict(policy),
        "selection_key": selection_key(summary),
        "candidate": summary["causal_bundle"],
        "brtc_lc": summary["brtc_lc"],
        "b0": summary["b0"],
        "counts": {
            key: summary[key] for key in (
                "stream_count", "frame_count", "person_frame_count",
                "acceleration_identity_triple_count", "association_accuracy",
                "camera_candidate_max_abs_change",
            )
        },
    }


def markdown(report: dict[str, Any], phase: str) -> str:
    summary = report["summary"]
    lines = [
        f"# Causal Temporal Ray Bundle — {phase}",
        "",
        "Camera is bit-exact frozen B0. Each trajectory uses only k=0,1,2 in arrival order.",
        "",
        "| Method | Root mm | Joint mm | Vertex mm | Layout-vector mm | Joint Accel mm/frame² | Coverage | Harm >5cm | Shift jitter mm/frame² |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        value = summary[method]
        lines.append(
            f"| {method} | {value['root_error_m']['mean']*1000:.1f} | "
            f"{value['joint_error_m']['mean']*1000:.1f} | "
            f"{value['vertex_error_m']['mean']*1000:.1f} | "
            f"{value['pairwise_vector_error_m']['mean']*1000:.1f} | "
            f"{value['joint_acceleration_error_m_per_frame2']['mean']*1000:.2f} | "
            f"{value.get('coverage', 1.0):.1%} | "
            f"{value.get('root_harm_over_5cm_rate', 0.0):.1%} | "
            f"{value.get('correction_jitter_m_per_frame2', {'mean': 0.0})['mean']*1000:.2f} |"
        )
    lines.extend([
        "",
        f"Streams: {summary['stream_count']}; frames: {summary['frame_count']}; "
        f"person-frame: {summary['person_frame_count']}; Accel triples: "
        f"{summary['acceleration_identity_triple_count']}.",
        "",
        f"Automatic association accuracy: {summary['association_accuracy']:.1%}. "
        f"Camera max change: {summary['camera_candidate_max_abs_change']:.3e}.",
    ])
    return "\n".join(lines) + "\n"


def self_test() -> None:
    assert robust_center([0.0, 1.0, 100.0], "median") == 1.0
    huber = robust_center([0.0, 1.0, 100.0], "huber")
    assert 0.0 < huber < 10.0
    policy = TemporalPolicy(1, "median", "ema", 2, 0.5, True)
    output, state, valid = temporal_filter(np.ones(3), True, [], policy)
    assert valid and np.allclose(output, 1.0) and len(state) == 1
    output, _, valid = temporal_filter(np.zeros(3), False, state, policy)
    assert valid and np.allclose(output, 1.0)
    print("self-test passed")


def main() -> None:
    args = parse_args()
    if not str(args.output_dir.resolve()).startswith(str(REPO_ROOT.resolve())):
        raise ValueError("Output must remain under Movie3R")
    if args.self_test:
        self_test()
        return
    brtc_payload = read_json(BRTC_POLICY)
    if not brtc_payload.get("dev_pass", False):
        raise RuntimeError("The upstream BRTC policy is not a passed frozen policy")
    brtc_policy = brtc_payload["policy"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.phase == "dev":
        dev_rows = complete_streams(read_json(DEV_REPORT))
        prepared_dev = []
        for index, stream_rows in enumerate(dev_rows, start=1):
            print(f"[prepare dance {index:03d}/{len(dev_rows):03d}]", flush=True)
            prepared_dev.append((
                stream_rows,
                [prepare_arrival("dance", row) for row in stream_rows],
            ))
        rows = []
        for index, policy in enumerate(policy_grid(), start=1):
            print(f"[policy {index:03d}/{len(policy_grid()):03d}] {policy}", flush=True)
            result = evaluate(
                "dance", DEV_REPORT, policy, brtc_policy, prepared=prepared_dev
            )
            rows.append(compact_dev_row(policy, result["summary"]))
        rows.sort(key=lambda row: tuple(row["selection_key"]))
        report = {
            "experiment": "causal_temporal_ray_bundle",
            "phase": "development",
            "protocol": {
                "development": "dance k=0,1,2 complete streams",
                "confirmation": "box k=0,1,2 complete streams; not read in development",
                "causal": "pre frames and post arrivals <= current k only",
                "camera": "bit-exact frozen B0",
                "gt_candidate_use": "none; GT only after predictions for metrics",
                "historical_exposure": "dance and box were used by earlier V14 post-hoc studies",
            },
            "brtc_policy_source": str(BRTC_POLICY),
            "policy_count": len(rows),
            "selection_rule": (
                "First satisfy <=3% root/joint/vertex regression, <=5% layout regression, "
                "coverage loss <=5 points and harm increase <=2 points vs BRTC-LC; then "
                "lexicographically minimize joint Accel, root, layout."
            ),
            "ranked_policies": rows,
        }
        (args.output_dir / "DEV_SCAN.json").write_text(
            json.dumps(jsonable(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(jsonable(rows[0]), indent=2, ensure_ascii=False))
        return
    if args.phase == "freeze":
        dev_path = args.output_dir / "DEV_SCAN.json"
        dev = read_json(dev_path)
        if dev.get("phase") != "development" or not dev.get("ranked_policies"):
            raise RuntimeError("A complete development scan is required before freeze")
        best = dev["ranked_policies"][0]
        frozen = {
            "experiment": "causal_temporal_ray_bundle",
            "phase": "frozen_before_confirmation",
            "policy": best["policy"],
            "development_selection_key": best["selection_key"],
            "development_candidate": best["candidate"],
            "development_brtc_lc": best["brtc_lc"],
            "development_scan": str(dev_path),
            "confirmation_source_declared_but_not_read": str(CONFIRM_REPORT),
            "warning": "box is procedurally held out here but historically exposed in prior V14 work",
        }
        FROZEN_POLICY.parent.mkdir(parents=True, exist_ok=True)
        FROZEN_POLICY.write_text(
            json.dumps(jsonable(frozen), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(jsonable(frozen), indent=2, ensure_ascii=False))
        return
    frozen = read_json(FROZEN_POLICY)
    if frozen.get("phase") != "frozen_before_confirmation":
        raise RuntimeError("Policy must be serialized before confirmation")
    policy = TemporalPolicy(**frozen["policy"])
    result = evaluate("box", CONFIRM_REPORT, policy, brtc_policy)
    report = {
        "experiment": "causal_temporal_ray_bundle",
        "phase": "confirmation",
        "policy_source": str(FROZEN_POLICY),
        "policy": asdict(policy),
        "protocol": {
            "confirmation": "box k=0,1,2 complete streams",
            "confirmation_tuning": "none",
            "causal": "strict arrival order; no future frame",
            "camera": "bit-exact frozen B0",
            "historical_exposure": "not a pristine holdout because box was used in earlier V14 post-hoc work",
        },
        **result,
    }
    (args.output_dir / "CONFIRM_RESULTS.json").write_text(
        json.dumps(jsonable(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "RESULTS.md").write_text(
        markdown(report, "box frozen confirmation"), encoding="utf-8"
    )
    print(markdown(report, "box frozen confirmation"))


if __name__ == "__main__":
    main()
