#!/usr/bin/env python3
"""V13 Phase 3: deployable cross-shot identity bridge.

The script has two physically separated paths:

1. feature extraction writes only inference-time Human3R outputs;
2. analysis attaches GT labels in the evaluator, probes WHO, and applies the
   accepted matches to the frozen Phase-2 Uniform Multi-Human Consensus.

Tokens never predict rotation, translation, scale, fusion weights, or a
Boundary transform.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import versions.v13.gt_id_consensus as geometry  # noqa: E402
from versions.v12.experiments.v14_1_shot_aware_state_routing_probe import (  # noqa: E402
    build_model,
)
from versions.v13.identity_bridge import (  # noqa: E402
    BASE_FEATURES,
    FEATURE_COMPONENTS,
    MatchConfig,
    evaluate_assignment,
    feature_distance,
    frame_feature,
    labeled_pair_distances,
    majority_track_labels,
    match_identity_bank,
)
from versions.v13.native_token_probe import (  # noqa: E402
    jsonable,
    person_feature,
    tensor_numpy,
)


FEATURES = tuple(FEATURE_COMPONENTS)
PROTOTYPES = ("last", "mean", "medoid")
DISTANCES = ("raw_l2", "normalized_l2", "cosine")
MATCHERS = ("hungarian", "sinkhorn")

DEFAULT_PROTOCOLS = {
    "three": {
        "timestamps": (500, 700, 900, 1000, 1100, 1300, 1500),
        "camera_pairs": ("0-1", "1-2", "2-3", "3-4", "4-5", "5-0", "0-3", "1-4", "2-5"),
        "offsets": (0, 1, 2, 4, 8),
        "geometry_cache": ROOT / "output/v20_phase1_gt_id_multihuman_consensus/case_cache",
    },
    "dance": {
        "timestamps": (200, 300, 400, 500, 600, 700),
        "camera_pairs": ("0-1", "0-3", "1-4"),
        "offsets": (0, 4),
        "geometry_cache": ROOT / "output/v13/dance_phase2/case_cache",
    },
    "box": {
        "timestamps": (470, 510, 550, 590, 630, 670),
        "camera_pairs": ("0-1", "0-3", "1-4"),
        "offsets": (0, 4),
        "geometry_cache": ROOT / "output/v13/box_phase3/case_cache",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", choices=tuple(DEFAULT_PROTOCOLS), default="three")
    parser.add_argument("--role", choices=("development", "evaluation"), default="development")
    parser.add_argument("--mode", choices=("extract", "analyze", "all"), default="all")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("/data/wangzheng/iJCV-CODE/data/MultiHuman/Real-World-Capture/extracted"),
    )
    parser.add_argument("--model_path", type=Path, default=ROOT / "src/human3r_896L.pth")
    parser.add_argument("--output_root", type=Path, default=ROOT / "output/v13/phase3_identity")
    parser.add_argument("--geometry_cache_dir", type=Path)
    parser.add_argument("--frozen_config", type=Path)
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--history_frames", type=int, default=5)
    parser.add_argument("--point_samples", type=int, default=1024)
    parser.add_argument("--timestamps", type=int, nargs="+")
    parser.add_argument("--camera_pairs", nargs="+")
    parser.add_argument("--offsets", type=int, nargs="+")
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--overwrite_features", action="store_true")
    parser.add_argument("--overwrite_geometry", action="store_true")
    return parser.parse_args()


def configure(args: argparse.Namespace) -> argparse.Namespace:
    protocol = DEFAULT_PROTOCOLS[str(args.sequence)]
    args.timestamps = tuple(args.timestamps or protocol["timestamps"])
    args.camera_pairs = tuple(args.camera_pairs or protocol["camera_pairs"])
    args.offsets = tuple(args.offsets or protocol["offsets"])
    args.output_dir = args.output_root / str(args.sequence)
    args.geometry_cache_dir = args.geometry_cache_dir or protocol["geometry_cache"]
    args.feature_cache_dir = args.output_dir / "feature_cache"
    if args.frozen_config is None:
        args.frozen_config = args.output_root / "three/frozen_identity_config.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.feature_cache_dir.mkdir(parents=True, exist_ok=True)
    args.geometry_cache_dir.mkdir(parents=True, exist_ok=True)
    geometry.IDENTITIES = geometry.SEQUENCE_IDENTITIES[str(args.sequence)]
    return args


def config_key(config: MatchConfig) -> str:
    return "/".join((config.feature, config.prototype, config.distance, config.matcher))


def feature_frame(prediction: dict, debug: dict) -> dict:
    count = int(debug["num_humans"])
    track_ids = (
        np.full(count, -1, dtype=np.int64)
        if debug.get("smpl_ids") is None
        else tensor_numpy(debug["smpl_ids"])[0].astype(np.int64)
    )
    features = {
        name: person_feature(debug, prediction, name).astype(np.float32)
        for name in BASE_FEATURES
    }
    return {
        "count": count,
        "track_ids": track_ids,
        "head_scores": tensor_numpy(debug.get("head_scores", np.ones(count))).reshape(-1),
        "head_locations": tensor_numpy(debug.get("head_locations", np.empty((0, 2)))),
        "features": features,
    }


def build_geometry_cache_from_stream(
    args: argparse.Namespace,
    spec: geometry.CaseSpec,
    predictions: list[dict],
    views: list[dict],
    debug: list[dict],
    layer,
    regressor: np.ndarray,
    started: float,
) -> dict:
    pre_frames = list(range(spec.timestamp - args.history_frames + 1, spec.timestamp + 1))
    post_frame = spec.timestamp + spec.offset
    all_frames = pre_frames + [post_frame]
    all_cameras = [spec.source_camera] * len(pre_frames) + [spec.target_camera]
    frame_humans, assignment_rows, clouds = [], [], []
    for prediction, view, debug_row, frame, camera in zip(
        predictions, views, debug, all_frames, all_cameras
    ):
        humans = geometry.layer_humans(prediction, view, debug_row, layer)
        height, width = [int(value) for value in tensor_numpy(view["true_shape"])[0]]
        assigned, assignment = geometry.assign_gt_identities(
            args, humans, geometry.camera_matrix(prediction), camera, frame, width, height
        )
        frame_humans.append(assigned)
        assignment_rows.append(assignment)
        clouds.append(
            geometry.sampled_background_cloud(
                prediction, view, humans, int(args.point_samples)
            )
        )
    return {
        "case": {
            "key": spec.key,
            "timestamp": spec.timestamp,
            "source_camera": spec.source_camera,
            "target_camera": spec.target_camera,
            "offset": spec.offset,
            "pre_frames": pre_frames,
            "post_frame": post_frame,
        },
        "poses": [geometry.camera_matrix(row).astype(np.float64) for row in predictions],
        "humans": frame_humans,
        "assignment": assignment_rows,
        "clouds": clouds,
        "gt": {
            "pre_c2w": np.linalg.inv(
                geometry.gt_w2c(args, spec.source_camera, pre_frames[-1])
            ),
            "post_c2w": np.linalg.inv(
                geometry.gt_w2c(args, spec.target_camera, post_frame)
            ),
            "post_humans": geometry.gt_human_payload(args, post_frame, regressor),
        },
        "runtime_seconds": time.perf_counter() - started,
        "inference_contract": {
            "input": "2048x2048 full frames resized to 512x512 without crop",
            "cut_reset": "pre-decode fresh scene/camera state",
            "human_memory": False,
            "da3": False,
            "keypoint_rcnn": False,
            "v11_4_scale": False,
            "vggt": False,
        },
    }


def extract(args: argparse.Namespace, specs: list[geometry.CaseSpec]) -> None:
    missing_features = [
        spec
        for spec in specs
        if args.overwrite_features
        or not (args.feature_cache_dir / f"{spec.key}.pt").is_file()
    ]
    missing_geometry = [
        spec
        for spec in specs
        if args.overwrite_geometry
        or not (args.geometry_cache_dir / f"{spec.key}.pt").is_file()
    ]
    required = [spec for spec in specs if spec in set(missing_features + missing_geometry)]
    if not required:
        print(">> Phase 3 feature and geometry caches are complete", flush=True)
        return

    torch.cuda.set_device(torch.device(args.device))
    model = build_model(args)
    layer = regressor = None
    if missing_geometry:
        _, layer = geometry.build_smpl_models(model, torch.device(args.device))
        regressor = geometry.joint_regressor(layer)
    missing_geometry_keys = {spec.key for spec in missing_geometry}
    missing_feature_keys = {spec.key for spec in missing_features}
    for index, spec in enumerate(required):
        print(f">> [{index + 1}/{len(required)}] Phase 3 infer {spec.key}", flush=True)
        pre_frames = list(
            range(spec.timestamp - args.history_frames + 1, spec.timestamp + 1)
        )
        post_frame = spec.timestamp + spec.offset
        image_paths = [
            geometry.extract_video_frame(args, spec.source_camera, frame)
            for frame in pre_frames
        ] + [geometry.extract_video_frame(args, spec.target_camera, post_frame)]
        started = time.perf_counter()
        predictions, views, debug = geometry.run_fresh_stream(
            model, image_paths, len(pre_frames), args
        )
        if spec.key in missing_feature_keys:
            payload = {
                "case": {
                    "key": spec.key,
                    "sequence": str(args.sequence),
                    "timestamp": spec.timestamp,
                    "source_camera": spec.source_camera,
                    "target_camera": spec.target_camera,
                    "offset": spec.offset,
                    "pre_frames": pre_frames,
                    "post_frame": post_frame,
                },
                "frames": [
                    feature_frame(prediction, debug_row)
                    for prediction, debug_row in zip(predictions, debug)
                ],
                "runtime_seconds": time.perf_counter() - started,
                "candidate_gt_usage": False,
            }
            torch.save(payload, args.feature_cache_dir / f"{spec.key}.pt")
        if spec.key in missing_geometry_keys:
            payload = build_geometry_cache_from_stream(
                args, spec, predictions, views, debug, layer, regressor, started
            )
            torch.save(payload, args.geometry_cache_dir / f"{spec.key}.pt")
    del model, layer
    torch.cuda.empty_cache()


def detection_labels(strict_cache: dict, feature_cache: dict) -> list[np.ndarray]:
    identity_index = {name: index for index, name in enumerate(geometry.IDENTITIES)}
    output = []
    for assignment, frame in zip(strict_cache["assignment"], feature_cache["frames"]):
        labels = np.full(int(frame["count"]), -1, dtype=np.int64)
        for row in assignment.get("assignments", []):
            detection = int(row["detection_index"])
            if detection < len(labels):
                labels[detection] = identity_index[str(row["identity"])]
        output.append(labels)
    return output


def aggregate_identity(rows: list[dict]) -> dict:
    totals = {
        key: int(sum(row[key] for row in rows))
        for key in (
            "true_positive",
            "false_positive",
            "false_negative",
            "accepted",
            "matchable",
            "id_switches",
            "dustbin_true_positive",
            "dustbin_false_positive",
            "dustbin_false_negative",
        )
    }
    tp, fp, fn = totals["true_positive"], totals["false_positive"], totals["false_negative"]
    dtp = totals["dustbin_true_positive"]
    dfp = totals["dustbin_false_positive"]
    dfn = totals["dustbin_false_negative"]
    return {
        **totals,
        "case_count": len(rows),
        "assignment_accuracy": tp / max(totals["accepted"], 1),
        "recall_at_1": tp / max(totals["matchable"], 1),
        "idf1": 2 * tp / max(2 * tp + fp + fn, 1),
        "dustbin_precision": dtp / max(dtp + dfp, 1),
        "dustbin_recall": dtp / max(dtp + dfn, 1),
    }


def tracking_stats(cases: list[dict]) -> dict:
    pair_correct = pair_total = id_switches = fragmentations = 0
    assigned = expected = 0
    seen_segments = set()
    feature_distances = {
        feature: {"same": [], "different": []} for feature in FEATURES
    }
    id_true_positive = id_labeled_detections = 0
    for case in cases:
        key = (
            case["case"]["timestamp"],
            case["case"]["source_camera"],
        )
        if key in seen_segments:
            continue
        seen_segments.add(key)
        frames = case["features"]["frames"][:-1]
        labels = case["labels"][:-1]
        expected += len(geometry.IDENTITIES) * len(frames)
        assigned += sum(int(np.sum(row >= 0)) for row in labels)
        by_identity: dict[int, list[int]] = defaultdict(list)
        confusion: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        for frame, frame_labels in zip(frames, labels):
            for detection_index, label in enumerate(frame_labels):
                if int(label) >= 0:
                    track_id = int(frame["track_ids"][detection_index])
                    by_identity[int(label)].append(track_id)
                    if track_id >= 0:
                        confusion[track_id][int(label)] += 1
                        id_labeled_detections += 1
        if confusion:
            track_ids = sorted(confusion)
            identity_ids = list(range(len(geometry.IDENTITIES)))
            score = np.asarray(
                [
                    [confusion[track_id].get(identity, 0) for identity in identity_ids]
                    for track_id in track_ids
                ],
                dtype=np.float64,
            )
            source, target = geometry.linear_sum_assignment(-score)
            id_true_positive += int(score[source, target].sum())
        for values in by_identity.values():
            clean = [value for value in values if value >= 0]
            if clean:
                id_switches += sum(first != second for first, second in zip(clean[:-1], clean[1:]))
                fragmentations += max(len(set(clean)) - 1, 0)
        for first_frame, second_frame, first_labels, second_labels in zip(
            frames[:-1], frames[1:], labels[:-1], labels[1:]
        ):
            first_map = {
                int(label): int(first_frame["track_ids"][index])
                for index, label in enumerate(first_labels)
                if int(label) >= 0
            }
            second_map = {
                int(label): int(second_frame["track_ids"][index])
                for index, label in enumerate(second_labels)
                if int(label) >= 0
            }
            for identity in set(first_map) & set(second_map):
                pair_total += 1
                pair_correct += int(first_map[identity] == second_map[identity])
            for feature in FEATURES:
                first_values = frame_feature(first_frame, feature)
                second_values = frame_feature(second_frame, feature)
                distance = feature_distance(first_values, second_values, "normalized_l2")
                for first_index, first_label in enumerate(first_labels):
                    for second_index, second_label in enumerate(second_labels):
                        if int(first_label) < 0 or int(second_label) < 0:
                            continue
                        bucket = "same" if int(first_label) == int(second_label) else "different"
                        feature_distances[feature][bucket].append(
                            float(distance[first_index, second_index])
                        )
    id_errors = id_labeled_detections - id_true_positive
    return {
        "unique_no_cut_segments": len(seen_segments),
        "detection_recall": assigned / max(expected, 1),
        "adjacent_assignment_accuracy": pair_correct / max(pair_total, 1),
        "adjacent_shared_identity_count": pair_total,
        "id_switches": id_switches,
        "track_fragmentation": fragmentations,
        "idf1_on_labeled_detections": (
            2 * id_true_positive
            / max(2 * id_true_positive + 2 * id_errors, 1)
        ),
        "adjacent_feature_distance_normalized_l2": {
            feature: {
                "same": geometry.finite_distribution(values["same"]),
                "different": geometry.finite_distribution(values["different"]),
                "mean_margin": (
                    float(np.mean(values["different"]) - np.mean(values["same"]))
                    if values["same"] and values["different"]
                    else float("nan")
                ),
            }
            for feature, values in feature_distances.items()
        },
    }


def probe_configs() -> list[MatchConfig]:
    return [
        MatchConfig(feature, prototype, distance, matcher)
        for feature in FEATURES
        for prototype in PROTOTYPES
        for distance in DISTANCES
        for matcher in MATCHERS
    ]


def evaluate_config(cases: list[dict], config: MatchConfig) -> dict:
    rows, same, different = [], [], []
    by_offset: dict[int, list[dict]] = defaultdict(list)
    for case in cases:
        frames = case["features"]["frames"]
        labels = case["labels"]
        bank_labels = majority_track_labels(labels[:-1], frames[:-1])
        result = match_identity_bank(frames[:-1], frames[-1], config)
        metrics = evaluate_assignment(result, bank_labels, labels[-1])
        rows.append(metrics)
        by_offset[int(case["case"]["offset"])].append(metrics)
        first, second = labeled_pair_distances(result, bank_labels, labels[-1])
        same.extend(first)
        different.extend(second)
    return {
        "config": asdict(config),
        "identity": aggregate_identity(rows),
        "by_offset": {
            str(offset): aggregate_identity(values)
            for offset, values in sorted(by_offset.items())
        },
        "same_distance": geometry.finite_distribution(same),
        "different_distance": geometry.finite_distribution(different),
        "distance_margin_mean": (
            float(np.mean(different) - np.mean(same))
            if same and different
            else float("nan")
        ),
    }


def rank_probe(rows: list[dict]) -> list[dict]:
    def key(row: dict):
        offsets = [value["idf1"] for value in row["by_offset"].values()]
        config = row["config"]
        distance_preference = {
            "normalized_l2": 2,
            "cosine": 1,
            "raw_l2": 0,
        }[config["distance"]]
        prototype_preference = {"mean": 2, "medoid": 1, "last": 0}[
            config["prototype"]
        ]
        matcher_preference = {"hungarian": 1, "sinkhorn": 0}[config["matcher"]]
        return (
            row["identity"]["idf1"],
            min(offsets) if offsets else -1.0,
            row["identity"]["assignment_accuracy"],
            row["identity"]["recall_at_1"],
            distance_preference,
            prototype_preference,
            matcher_preference,
        )

    return sorted(rows, key=key, reverse=True)


def threshold_candidates(cases: list[dict], config: MatchConfig) -> list[float]:
    if config.matcher == "sinkhorn":
        return np.linspace(0.0, 0.9, 37).tolist()
    values = []
    unrestricted = replace(config, max_cost=float("inf"))
    for case in cases:
        frames = case["features"]["frames"]
        result = match_identity_bank(frames[:-1], frames[-1], unrestricted)
        values.extend(float(row["cost"]) for row in result["pairs"])
    if not values:
        return [float("inf")]
    quantiles = np.quantile(values, np.linspace(0.02, 1.0, 50)).tolist()
    return sorted(set(float(value) for value in quantiles)) + [float("inf")]


def select_dustbin(cases: list[dict], base: MatchConfig) -> tuple[MatchConfig, list[dict]]:
    candidates = []
    for threshold in threshold_candidates(cases, base):
        config = (
            replace(base, sinkhorn_score_threshold=threshold)
            if base.matcher == "sinkhorn"
            else replace(base, max_cost=threshold)
        )
        row = evaluate_config(cases, config)
        identity = row["identity"]
        normalized_fp = identity["false_positive"] / max(identity["matchable"], 1)
        row["selection_score"] = identity["idf1"] - 0.5 * normalized_fp
        candidates.append(row)
    candidates.sort(
        key=lambda row: (
            row["selection_score"],
            row["identity"]["idf1"],
            -row["identity"]["false_positive"],
            row["identity"]["recall_at_1"],
        ),
        reverse=True,
    )
    return MatchConfig(**candidates[0]["config"]), candidates


def geometry_caches(args: argparse.Namespace, spec: geometry.CaseSpec) -> tuple[dict, dict]:
    path = args.geometry_cache_dir / f"{spec.key}.pt"
    raw = torch.load(path, map_location="cpu", weights_only=False)
    strict = geometry.reassign_cache_gt_identities(args, raw)
    return raw, strict


def detections_by_index(frame: dict) -> dict[int, dict]:
    detections = {}
    for row in frame.values():
        # Geometry caches were originally keyed by evaluator-only GT identity.
        # Rebuild a pure detection payload before automatic association so the
        # deployable solver cannot accidentally consume that annotation later.
        detection = {
            key: value for key, value in row.items() if key != "identity"
        }
        detections[int(detection["detection_index"])] = detection
    return detections


def automatic_cache(raw_cache: dict, feature_frames: list[dict], result: dict) -> tuple[dict, dict]:
    """Relabel geometry by deployable pre-shot track ID, never by GT ID."""
    bank_track_ids = [int(value) for value in result["bank"]["track_ids"]]
    slot_names = list(geometry.IDENTITIES)
    track_to_slot = {
        track_id: slot_names[index]
        for index, track_id in enumerate(bank_track_ids[: len(slot_names)])
    }
    humans = []
    for frame_index, (feature_frame_row, detection_frame) in enumerate(
        zip(feature_frames[:-1], raw_cache["humans"][:-1])
    ):
        detections = detections_by_index(detection_frame)
        relabeled = {}
        for detection_index, track_id in enumerate(feature_frame_row["track_ids"]):
            slot = track_to_slot.get(int(track_id))
            if slot is not None and detection_index in detections:
                relabeled[slot] = detections[detection_index]
        humans.append(relabeled)
    post_detections = detections_by_index(raw_cache["humans"][-1])
    post = {}
    accepted = []
    for row in result["accepted_pairs"]:
        track_id = bank_track_ids[int(row["source_index"])]
        slot = track_to_slot.get(track_id)
        detection = int(row["target_index"])
        if slot is not None and detection in post_detections:
            post[slot] = post_detections[detection]
            accepted.append(
                {
                    "slot": slot,
                    "track_id": track_id,
                    "post_detection_index": detection,
                    "cost": float(row["cost"]),
                    "score": float(row["score"]),
                }
            )
    humans.append(post)
    cache = dict(raw_cache)
    cache["humans"] = humans
    return cache, {"track_to_slot": track_to_slot, "accepted": accepted}


def fallback_solution(cache: dict) -> dict:
    initial = np.asarray(cache["poses"][-2]) @ np.linalg.inv(np.asarray(cache["poses"][-1]))
    valid_target = [cloud for cloud in cache["clouds"][:-1] if len(cloud)]
    target = np.concatenate(valid_target) if valid_target else np.empty((0, 3))
    refined, debug = geometry.fixed_refine(initial, cache["clouds"][-1], target)
    return {
        "rotation": refined[:3, :3],
        "translation": refined[:3, 3],
        "identities": (),
        "fallback": "identity-free camera-continuity Fixed Explicit",
        "fixed_debug": debug,
    }


def uniform_solution(cache: dict) -> tuple[dict, dict]:
    candidates = geometry.human_candidates(cache)
    identities = tuple(identity for identity in geometry.IDENTITIES if identity in candidates)
    if len(identities) >= 2:
        rotation, translation = geometry.solve_consensus(candidates, identities, "mean_raw_t")
        solution = {
            "rotation": rotation,
            "translation": translation,
            "identities": identities,
            "fallback": "multi-human",
        }
    elif len(identities) == 1:
        candidate = candidates[identities[0]]
        solution = {
            "rotation": candidate["rotation"],
            "translation": candidate["translation"],
            "identities": identities,
            "selected_identity": identities[0],
            "fallback": "single-human",
        }
    else:
        solution = fallback_solution(cache)
    return solution, candidates


def cyclic_wrong_result(result: dict) -> dict:
    output = copy.deepcopy(result)
    accepted = output["accepted_pairs"]
    if len(accepted) >= 2:
        targets = [int(row["target_index"]) for row in accepted]
        shifted = targets[1:] + targets[:1]
        for row, target in zip(accepted, shifted):
            row["target_index"] = target
    return output


def controlled_frames(frames: list[dict], feature: str, mode: str, seed: int) -> list[dict]:
    output = copy.deepcopy(frames)
    components = FEATURE_COMPONENTS[feature]
    rng = np.random.default_rng(seed)
    for frame in output:
        count = int(frame["count"])
        if count == 0:
            continue
        permutation = rng.permutation(count)
        for component in components:
            value = np.asarray(frame["features"][component])
            if mode == "zero":
                frame["features"][component] = np.zeros_like(value)
            elif mode == "shuffle":
                frame["features"][component] = value[permutation]
            else:
                raise ValueError(mode)
    return output


def geometry_verify(result: dict, auto_cache: dict) -> tuple[dict, dict]:
    """Reject only gross identity conflicts; never reweight normal candidates."""
    solution, candidates = uniform_solution(auto_cache)
    identities = tuple(candidates)
    if len(identities) < 2:
        return result, {"triggered": False, "reason": "insufficient_multi_support"}
    translations = [candidates[name]["translation"] for name in identities]
    rotations = [candidates[name]["rotation"] for name in identities]
    translation_max = max(
        np.linalg.norm(first - second)
        for index, first in enumerate(translations)
        for second in translations[index + 1 :]
    )
    rotation_max = max(
        geometry.rotation_distance_deg(first, second)
        for index, first in enumerate(rotations)
        for second in rotations[index + 1 :]
    )
    gross = translation_max > 3.0 or rotation_max > 100.0
    diagnostics = {
        "triggered": bool(gross),
        "translation_candidate_max_m": float(translation_max),
        "rotation_candidate_max_deg": float(rotation_max),
        "threshold_translation_m": 3.0,
        "threshold_rotation_deg": 100.0,
    }
    if not gross:
        return result, diagnostics
    # At two-person support there is no causal evidence for which side is
    # correct. Sending both to dustbin is safer than committing a swapped ID.
    output = copy.deepcopy(result)
    output["accepted_pairs"] = []
    output["unmatched_source"] = np.arange(len(result["bank"]["track_ids"]), dtype=np.int64)
    output["unmatched_target"] = np.arange(len(result["target"]), dtype=np.int64)
    diagnostics["action"] = "all_matches_to_dustbin_fixed_fallback"
    return output, diagnostics


def endpoint_case(
    raw_cache: dict,
    strict_cache: dict,
    features: dict,
    labels: list[np.ndarray],
    selected: MatchConfig,
    seed: int,
) -> dict:
    frames = features["frames"]
    bank_labels = majority_track_labels(labels[:-1], frames[:-1])
    selected_result = match_identity_bank(frames[:-1], frames[-1], selected)
    native_result = match_identity_bank(
        frames[:-1],
        frames[-1],
        MatchConfig(
            feature="refined_human_tokens",
            prototype="last",
            distance="raw_l2",
            matcher="sinkhorn",
            sinkhorn_score_threshold=0.2,
            sinkhorn_alpha=-10.0,
            sinkhorn_iterations=20,
        ),
    )
    hungarian_result = match_identity_bank(
        frames[:-1], frames[-1], replace(selected, matcher="hungarian")
    )
    no_dustbin = (
        replace(selected, sinkhorn_score_threshold=0.0)
        if selected.matcher == "sinkhorn"
        else replace(selected, max_cost=float("inf"))
    )
    result_no_dustbin = match_identity_bank(frames[:-1], frames[-1], no_dustbin)
    shuffled_frames = controlled_frames(frames[:-1], selected.feature, "shuffle", seed)
    zero_frames = controlled_frames(frames[:-1], selected.feature, "zero", seed)
    controls = {
        "automatic_id": selected_result,
        "automatic_without_dustbin": result_no_dustbin,
        "wrong_person": cyclic_wrong_result(selected_result),
        "shuffled_memory": match_identity_bank(shuffled_frames, frames[-1], selected),
        "zero_memory": match_identity_bank(zero_frames, frames[-1], selected),
    }

    gt_candidates = geometry.human_candidates(strict_cache)
    gt_methods = geometry.method_solutions(gt_candidates)
    gt_name = "naive_mean" if "naive_mean" in gt_methods else "single_highest_confidence"
    methods = {
        "single_highest_confidence": geometry.evaluate_solution(
            strict_cache, gt_methods["single_highest_confidence"]
        ) if "single_highest_confidence" in gt_methods else None,
        "gt_id_uniform": geometry.evaluate_solution(strict_cache, gt_methods[gt_name])
        if gt_name in gt_methods
        else None,
    }
    matching = {}
    for name, result in controls.items():
        auto_cache, mapping = automatic_cache(raw_cache, frames, result)
        solution, candidates = uniform_solution(auto_cache)
        methods[name] = geometry.evaluate_solution(strict_cache, solution)
        methods[name]["fallback"] = solution.get("fallback")
        matching[name] = {
            "identity": evaluate_assignment(result, bank_labels, labels[-1]),
            "mapping": mapping,
            "candidate_count": len(candidates),
        }

    initial_cache, _ = automatic_cache(raw_cache, frames, selected_result)
    verified_result, verification = geometry_verify(selected_result, initial_cache)
    verified_cache, verified_mapping = automatic_cache(raw_cache, frames, verified_result)
    verified_solution, verified_candidates = uniform_solution(verified_cache)
    methods["automatic_geometry_verified"] = geometry.evaluate_solution(
        strict_cache, verified_solution
    )
    methods["automatic_geometry_verified"]["fallback"] = verified_solution.get("fallback")
    matching["automatic_geometry_verified"] = {
        "identity": evaluate_assignment(verified_result, bank_labels, labels[-1]),
        "mapping": verified_mapping,
        "candidate_count": len(verified_candidates),
        "verification": verification,
    }
    hungarian_matrix = np.zeros_like(hungarian_result["cost"], dtype=np.float32)
    for row in hungarian_result["accepted_pairs"]:
        hungarian_matrix[int(row["source_index"]), int(row["target_index"])] = 1.0
    return {
        "identity": evaluate_assignment(selected_result, bank_labels, labels[-1]),
        "bank_gt_labels_evaluator_only": bank_labels,
        "post_gt_labels_evaluator_only": labels[-1],
        "cost": selected_result["cost"],
        "hungarian_assignment_matrix": hungarian_matrix,
        "native_sinkhorn_soft_assignment_matrix": np.exp(
            np.asarray(native_result.get("log_transport", np.empty((0, 0))))
        ),
        "native_sinkhorn_identity": evaluate_assignment(
            native_result, bank_labels, labels[-1]
        ),
        "accepted_pairs": selected_result["accepted_pairs"],
        "unmatched_source": selected_result["unmatched_source"],
        "unmatched_target": selected_result["unmatched_target"],
        "matching": matching,
        "methods": {key: value for key, value in methods.items() if value is not None},
    }


def metric_summary(cases: list[dict], method: str) -> dict:
    rows = [case["endpoint"]["methods"].get(method) for case in cases]
    rows = [row for row in rows if row is not None]
    fields = (
        "camera_translation_error_m",
        "camera_rotation_error_deg",
        "camera_composite",
        "human_root_error_m",
        "human_joint_error_m",
        "human_vertex_error_m",
        "pairwise_distance_error_m",
        "pairwise_vector_error_m",
    )
    return {
        "case_count": len(rows),
        **{
            field: geometry.finite_distribution([float(row[field]) for row in rows])
            for field in fields
        },
        "catastrophic_rate": float(np.mean([row["catastrophic"] for row in rows]))
        if rows
        else float("nan"),
        "fallback_counts": {
            name: sum(row.get("fallback") == name for row in rows)
            for name in ("multi-human", "single-human", "identity-free camera-continuity Fixed Explicit")
        },
    }


def endpoint_summary(cases: list[dict]) -> dict:
    names = (
        "single_highest_confidence",
        "gt_id_uniform",
        "automatic_id",
        "automatic_without_dustbin",
        "automatic_geometry_verified",
        "wrong_person",
        "shuffled_memory",
        "zero_memory",
    )
    methods = {name: metric_summary(cases, name) for name in names}
    identity = aggregate_identity([case["endpoint"]["identity"] for case in cases])
    single = methods["single_highest_confidence"]["camera_composite"]["mean"]
    oracle = methods["gt_id_uniform"]["camera_composite"]["mean"]
    automatic = methods["automatic_id"]["camera_composite"]["mean"]
    denominator = single - oracle
    retention = (single - automatic) / denominator if denominator > 1e-12 else float("nan")
    return {
        "identity": identity,
        "methods": methods,
        "gt_id_gain_retention": float(retention),
        "common_case_definition": "all listed means use the same Phase-3 case list",
    }


def endpoint_breakdown(cases: list[dict]) -> dict:
    by_offset: dict[int, list[dict]] = defaultdict(list)
    by_camera_pair: dict[str, list[dict]] = defaultdict(list)
    by_matched_count: dict[int, list[dict]] = defaultdict(list)
    for case in cases:
        spec = case["case"]
        by_offset[int(spec["offset"])].append(case)
        by_camera_pair[
            f"{int(spec['source_camera'])}->{int(spec['target_camera'])}"
        ].append(case)
        count = int(
            case["endpoint"]["matching"]["automatic_id"]["candidate_count"]
        )
        by_matched_count[count].append(case)
    return {
        "by_offset": {
            str(key): endpoint_summary(value) for key, value in sorted(by_offset.items())
        },
        "by_camera_pair": {
            key: endpoint_summary(value) for key, value in sorted(by_camera_pair.items())
        },
        "by_automatic_matched_humans": {
            str(key): endpoint_summary(value)
            for key, value in sorted(by_matched_count.items())
        },
    }


def plot_probe(ranked: list[dict], report_dir: Path) -> None:
    top = ranked[:12]
    figure, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    names = [
        "/".join((row["config"]["feature"], row["config"]["prototype"], row["config"]["distance"], row["config"]["matcher"]))
        for row in top
    ]
    values = [row["identity"]["idf1"] for row in top]
    axis.barh(range(len(top)), values, color="#2f6f5e")
    axis.set_yticks(range(len(top)), names)
    axis.invert_yaxis()
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("Cross-shot IDF1")
    axis.set_title("V13 Phase 3 identity feature probe")
    figure.savefig(report_dir / "feature_probe_top12.png", dpi=160)
    plt.close(figure)


def plot_selected_matrices(cases: list[dict], report_dir: Path) -> None:
    worst = sorted(
        cases, key=lambda row: row["endpoint"]["identity"]["idf1"]
    )[:4]
    best = sorted(
        cases, key=lambda row: row["endpoint"]["identity"]["idf1"], reverse=True
    )[:4]
    dustbin = [
        row
        for row in cases
        if len(row["endpoint"]["unmatched_source"])
        or len(row["endpoint"]["unmatched_target"])
    ][:4]
    selected = []
    seen = set()
    for row in worst + best + dustbin:
        key = row["case"]["key"]
        if key not in seen:
            selected.append(row)
            seen.add(key)
    selected = selected[:12]
    columns = 4
    rows = max(1, math.ceil(len(selected) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.5 * rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for axis, case in zip(axes, selected):
        cost = np.asarray(case["endpoint"]["cost"])
        image = axis.imshow(cost, cmap="magma") if cost.size else None
        identity = case["endpoint"]["identity"]
        axis.set_title(
            f"{case['case']['key']}\nIDF1={identity['idf1']:.2f}, "
            f"dustbin={len(case['endpoint']['unmatched_target'])}",
            fontsize=8,
        )
        axis.set_xlabel("post detection")
        axis.set_ylabel("pre track prototype")
        if image is not None:
            figure.colorbar(image, ax=axis, fraction=0.046)
    for axis in axes[len(selected) :]:
        axis.axis("off")
    figure.savefig(report_dir / "selected_distance_matrices.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.5 * rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for axis, case in zip(axes, selected):
        matrix = np.asarray(case["endpoint"]["native_sinkhorn_soft_assignment_matrix"])
        image = axis.imshow(matrix, cmap="viridis") if matrix.size else None
        axis.set_title(case["case"]["key"], fontsize=8)
        axis.set_xlabel("post + dustbin")
        axis.set_ylabel("pre + dustbin")
        if image is not None:
            figure.colorbar(image, ax=axis, fraction=0.046)
    for axis in axes[len(selected) :]:
        axis.axis("off")
    figure.savefig(report_dir / "native_sinkhorn_soft_matrices.png", dpi=160)
    plt.close(figure)


def markdown(report: dict) -> str:
    selected = report["selected_config"]
    identity = report["endpoint"]["identity"]
    methods = report["endpoint"]["methods"]

    def mean(method: str, metric: str) -> float:
        return methods[method][metric]["mean"]

    lines = [
        f"# V13 Phase 3: Cross-Shot Identity Bridge ({report['sequence']})",
        "",
        f"Role: `{report['role']}`",
        "",
        "## Frozen identity rule",
        "",
        "```json",
        json.dumps(selected, indent=2),
        "```",
        "",
        "## Identity",
        "",
        f"- cases: {identity['case_count']}",
        f"- assignment accuracy: {identity['assignment_accuracy']:.4f}",
        f"- Recall@1: {identity['recall_at_1']:.4f}",
        f"- IDF1: {identity['idf1']:.4f}",
        f"- ID switches: {identity['id_switches']}",
        f"- dustbin precision/recall: {identity['dustbin_precision']:.4f} / {identity['dustbin_recall']:.4f}",
        "",
        "## End-to-end shared Boundary",
        "",
        "| Method | Camera T | Rotation | Composite | Catastrophic |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in (
        "single_highest_confidence",
        "gt_id_uniform",
        "automatic_id",
        "automatic_geometry_verified",
        "wrong_person",
        "shuffled_memory",
    ):
        lines.append(
            f"| {name} | {mean(name, 'camera_translation_error_m'):.3f} | "
            f"{mean(name, 'camera_rotation_error_deg'):.2f} | "
            f"{mean(name, 'camera_composite'):.3f} | "
            f"{methods[name]['catastrophic_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"GT-ID multi-human gain retention: `{report['endpoint']['gt_id_gain_retention']:.4f}`",
            "",
            "## GT boundary",
            "",
            "Feature extraction and candidate generation do not read GT identity, camera, SMPL-X, source ID, or camera-pair ID. GT identity is attached only in this evaluator for feature selection/scoring and the GT-ID upper bound.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(args: argparse.Namespace, specs: list[geometry.CaseSpec]) -> dict:
    cases = []
    for index, spec in enumerate(specs):
        feature_path = args.feature_cache_dir / f"{spec.key}.pt"
        geometry_path = args.geometry_cache_dir / f"{spec.key}.pt"
        if not feature_path.is_file() or not geometry_path.is_file():
            raise FileNotFoundError(f"Missing Phase 3 cache for {spec.key}")
        features = torch.load(feature_path, map_location="cpu", weights_only=False)
        raw, strict = geometry_caches(args, spec)
        labels = detection_labels(strict, features)
        for frame_index, (frame, raw_humans) in enumerate(
            zip(features["frames"], raw["humans"])
        ):
            detection_indices = sorted(
                int(row["detection_index"]) for row in raw_humans.values()
            )
            expected_indices = list(range(int(frame["count"])))
            if detection_indices != expected_indices:
                raise ValueError(
                    f"{spec.key} frame {frame_index}: geometry detections "
                    f"{detection_indices} != feature detections {expected_indices}"
                )
        cases.append(
            {
                "case": features["case"],
                "features": features,
                "raw": raw,
                "strict": strict,
                "labels": labels,
            }
        )
        if (index + 1) % 50 == 0:
            print(f">> loaded {index + 1}/{len(specs)} Phase 3 cases", flush=True)

    probe_cache_path = args.output_dir / "feature_probe_cache.json"
    case_keys = [case["case"]["key"] for case in cases]
    probe_rows = []
    if probe_cache_path.is_file():
        probe_cache = json.loads(probe_cache_path.read_text(encoding="utf-8"))
        if probe_cache.get("case_keys") == case_keys:
            probe_rows = probe_cache["rows"]
            print(f">> reused {len(probe_rows)} cached identity probe configurations", flush=True)
    if not probe_rows:
        for index, config in enumerate(probe_configs()):
            probe_rows.append(evaluate_config(cases, config))
            if (index + 1) % 30 == 0:
                print(f">> identity probe {index + 1}/{len(FEATURES) * len(PROTOTYPES) * len(DISTANCES) * len(MATCHERS)}", flush=True)
        probe_cache_path.write_text(
            json.dumps(
                jsonable({"case_keys": case_keys, "rows": probe_rows}),
                indent=2,
                allow_nan=True,
            )
            + "\n",
            encoding="utf-8",
        )
    ranked = rank_probe(probe_rows)

    if args.role == "development":
        base = MatchConfig(**ranked[0]["config"])
        selected, threshold_rows = select_dustbin(cases, base)
        frozen = {
            "schema_version": 1,
            "selected_on": "MultiHuman three only",
            "selection_scope": "identity metrics; no camera or Boundary GT used",
            "match_config": asdict(selected),
            "geometry_verification": {
                "translation_candidate_max_m": 3.0,
                "rotation_candidate_max_deg": 100.0,
                "action": "all tentative matches to dustbin, then Fixed fallback",
                "passes": 1,
            },
            "track_ttl": 8,
        }
        args.frozen_config.write_text(
            json.dumps(jsonable(frozen), indent=2, allow_nan=True) + "\n",
            encoding="utf-8",
        )
    else:
        if not args.frozen_config.is_file():
            raise FileNotFoundError(f"Frozen three config not found: {args.frozen_config}")
        frozen = json.loads(args.frozen_config.read_text(encoding="utf-8"))
        selected = MatchConfig(**frozen["match_config"])
        threshold_rows = []

    endpoint_cases = []
    for index, case in enumerate(cases):
        endpoint = endpoint_case(
            case["raw"],
            case["strict"],
            case["features"],
            case["labels"],
            selected,
            int(args.seed) + index,
        )
        endpoint_cases.append({"case": case["case"], "endpoint": endpoint})
    report = {
        "experiment": "V13 Phase 3 Cross-Shot Identity Bridge",
        "sequence": str(args.sequence),
        "role": str(args.role),
        "case_count": len(cases),
        "protocol": {
            "timestamps": args.timestamps,
            "camera_pairs": args.camera_pairs,
            "offsets": args.offsets,
            "history_frames": args.history_frames,
            "full_frame_no_crop": True,
        },
        "candidate_gt_usage": {
            "feature_extraction": False,
            "identity_bank": False,
            "matching": False,
            "dustbin": False,
            "boundary_candidate": False,
            "probe_scoring": True,
            "gt_id_upper_bound": True,
            "camera_human_metrics": True,
            "automatic_geometry_uses_identity_keys": False,
            "automatic_geometry_source": "raw detection multiset sorted only by detection_index",
        },
        "geometry_contract": {
            "phase2_uniform_consensus_frozen": True,
            "rotation": "equal-weight SO(3) mean of accepted per-human candidates",
            "translation": "arithmetic mean of accepted raw per-human translations",
            "shot_scale": 1.0,
            "da3": False,
            "vggt": False,
            "keypoint_rcnn": False,
            "token_predicts_se3": False,
        },
        "no_cut_tracker": tracking_stats(cases),
        "native_human3r_cross_shot_tracker": evaluate_config(
            cases,
            MatchConfig(
                feature="refined_human_tokens",
                prototype="last",
                distance="raw_l2",
                matcher="sinkhorn",
                sinkhorn_score_threshold=0.2,
                sinkhorn_alpha=-10.0,
                sinkhorn_iterations=20,
            ),
        ),
        "feature_probe_ranked": ranked,
        "threshold_probe": threshold_rows,
        "selected_config": asdict(selected),
        "frozen_rule": frozen,
        "endpoint": endpoint_summary(endpoint_cases),
        "endpoint_breakdown": endpoint_breakdown(endpoint_cases),
        "cases": endpoint_cases,
    }
    json_path = args.output_dir / "v13_phase3_identity_bridge.json"
    md_path = args.output_dir / "v13_phase3_identity_bridge.md"
    json_path.write_text(
        json.dumps(jsonable(report), indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    md_path.write_text(markdown(report), encoding="utf-8")
    plot_probe(ranked, args.output_dir)
    plot_selected_matrices(endpoint_cases, args.output_dir)
    print(f">> Phase 3 JSON: {json_path}", flush=True)
    print(f">> Phase 3 report: {md_path}", flush=True)
    return report


def main() -> None:
    args = configure(parse_args())
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    specs = geometry.case_specs(args)
    if not specs:
        raise ValueError("No valid Phase 3 cases")
    if args.mode in {"extract", "all"}:
        extract(args, specs)
    if args.mode in {"analyze", "all"}:
        analyze(args, specs)


if __name__ == "__main__":
    main()
