#!/usr/bin/env python3
"""V13 Phase 5 Stage 0/1: causal shot-persistent identity feasibility.

Human3R scene/camera state is freshly reset for every shot cache.  The only
state that crosses composed shots is the external identity state implemented in
``shot_persistent_identity.py``.  GT identity is attached after matching for
evaluation and never enters a candidate cost, gate, state update, or commit.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from versions.v13 import gt_id_consensus as geometry  # noqa: E402
from versions.v13.appearance_identity import (  # noqa: E402
    FrozenDinoAppearance,
    PrecisionGateConfig,
    apply_precision_gate,
    augment_frame,
    crop_predicted_bbox,
    precision_signals,
)
from versions.v13.experiments import phase3_cross_shot_identity as phase3  # noqa: E402
from versions.v13.identity_bridge import (  # noqa: E402
    MatchConfig,
    evaluate_assignment,
    majority_track_labels,
    match_identity_bank,
)
from versions.v13.native_token_probe import jsonable  # noqa: E402
from versions.v13.shot_persistent_identity import (  # noqa: E402
    ShotPersistentIdentityState,
    assignment_result,
    enumerate_topk_hypotheses,
)


FEATURE = "appearance_beta_pose"
PROTOTYPES = ("last", "mean", "medoid")
DISPERSION_FLOORS = (0.03, 0.05, 0.08, 0.12)


@dataclass(frozen=True)
class StreamSpec:
    name: str
    start_frame: int
    cameras: tuple[int, ...]
    shot_length: int = 5

    @property
    def shot_starts(self) -> tuple[int, ...]:
        return tuple(
            self.start_frame + index * self.shot_length
            for index in range(len(self.cameras))
        )


@dataclass(frozen=True)
class StateMethod:
    name: str
    prototype: str
    track_normalized: bool = False
    dispersion_floor: float = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", choices=("three", "dance", "box"), default="three")
    parser.add_argument("--role", choices=("development", "evaluation"), default="development")
    parser.add_argument("--mode", choices=("extract", "analyze", "all"), default="all")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path(
            "/data/wangzheng/iJCV-CODE/data/MultiHuman/Real-World-Capture/extracted"
        ),
    )
    parser.add_argument("--model_path", type=Path, default=ROOT / "src/human3r_896L.pth")
    parser.add_argument(
        "--output_root", type=Path, default=ROOT / "output/v13/phase5_identity"
    )
    parser.add_argument(
        "--phase4_config",
        type=Path,
        default=ROOT / "versions/v13/configs/phase4_precision_config.json",
    )
    parser.add_argument(
        "--encoder_hub_dir",
        type=Path,
        default=Path("/data/wangzheng/.cache/torch/hub/facebookresearch_dinov2_main"),
    )
    parser.add_argument(
        "--encoder_checkpoint",
        type=Path,
        default=Path(
            "/data/wangzheng/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth"
        ),
    )
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--point_samples", type=int, default=1024)
    parser.add_argument("--state_ttl", type=int, default=60)
    parser.add_argument("--history_size", type=int, default=5)
    parser.add_argument("--max_tracks", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=6)
    parser.add_argument("--max_streams", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=20260725)
    return parser.parse_args()


def configure(args: argparse.Namespace) -> argparse.Namespace:
    args.output_dir = args.output_root / str(args.sequence)
    args.shot_cache_dir = args.output_dir / "shot_cache"
    args.input_frame_dir = args.output_dir / "input_frames"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.shot_cache_dir.mkdir(parents=True, exist_ok=True)
    geometry.IDENTITIES = geometry.SEQUENCE_IDENTITIES[str(args.sequence)]
    return args


def stream_specs(sequence: str) -> list[StreamSpec]:
    if sequence == "three":
        starts = (500, 700, 900, 1100, 1300)
        routes = {
            "cycle": (0, 1, 2, 3, 4, 5, 0),
            "wide": (0, 3, 1, 4, 2, 5, 0),
            "return": (0, 3, 0, 3, 0, 3, 0),
        }
    elif sequence == "dance":
        starts = (200, 400, 600)
        routes = {
            "cycle": (0, 1, 2, 3, 4, 5),
            "wide": (0, 3, 1, 4, 2, 5),
            "return": (0, 3, 0, 3, 0, 3),
        }
    else:
        starts = (470, 540, 620)
        routes = {
            "cycle": (0, 1, 2, 3, 4, 5),
            "wide": (0, 3, 1, 4, 2, 5),
            "return": (0, 3, 0, 3, 0, 3),
        }
    return [
        StreamSpec(f"{sequence}_{route}_f{start}", start, cameras)
        for start in starts
        for route, cameras in routes.items()
    ]


def selected_specs(args: argparse.Namespace) -> list[StreamSpec]:
    specs = stream_specs(str(args.sequence))
    return specs[: int(args.max_streams)] if int(args.max_streams) > 0 else specs


def shot_key(sequence: str, camera: int, start: int, length: int) -> str:
    return f"{sequence}_cam{camera}_f{start:04d}_n{length}"


def shot_path(args: argparse.Namespace, camera: int, start: int, length: int) -> Path:
    return args.shot_cache_dir / f"{shot_key(args.sequence, camera, start, length)}.pt"


def required_shots(specs: list[StreamSpec]) -> list[tuple[int, int, int]]:
    return sorted(
        {
            (int(camera), int(start), int(spec.shot_length))
            for spec in specs
            for camera, start in zip(spec.cameras, spec.shot_starts)
        }
    )


def label_row(assignment: dict, count: int) -> np.ndarray:
    index = {name: value for value, name in enumerate(geometry.IDENTITIES)}
    labels = np.full(int(count), -1, dtype=np.int64)
    for row in assignment.get("assignments", []):
        detection = int(row["detection_index"])
        identity = str(row["identity"])
        if detection < len(labels) and identity in index:
            labels[detection] = int(index[identity])
    return labels


def predicted_appearance(
    encoder: FrozenDinoAppearance,
    frame: dict,
    humans: list[dict],
    image_path: Path,
    processed_shape: tuple[int, int],
) -> tuple[dict, list[dict]]:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(image_path)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    count = int(frame["count"])
    rows, crops, indices = [], [], []
    by_detection = {int(row["detection_index"]): row for row in humans}
    for detection in range(count):
        human = by_detection.get(detection)
        if human is None:
            rows.append({"valid": False, "reason": "missing_detection_geometry"})
            continue
        crop, row = crop_predicted_bbox(
            image, human["bbox"], processed_shape, padding_ratio=0.08
        )
        rows.append(row)
        if crop is not None:
            indices.append(detection)
            crops.append(crop)
    appearance = np.zeros((count, 768), dtype=np.float32)
    valid = np.zeros(count, dtype=bool)
    if crops:
        encoded = encoder.encode(crops)
        appearance[np.asarray(indices, dtype=np.int64)] = encoded
        valid[np.asarray(indices, dtype=np.int64)] = True
    return augment_frame(frame, appearance, valid), rows


def extract(args: argparse.Namespace, specs: list[StreamSpec]) -> None:
    missing = [
        row
        for row in required_shots(specs)
        if args.overwrite or not shot_path(args, *row).is_file()
    ]
    if not missing:
        print(">> Phase 5 shot caches are complete", flush=True)
        return
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    model = phase3.build_model(args)
    _, layer = geometry.build_smpl_models(model, device)
    encoder = FrozenDinoAppearance(
        str(args.device), args.encoder_hub_dir, args.encoder_checkpoint, batch_size=32
    )
    started = time.perf_counter()
    for shot_index, (camera, start, length) in enumerate(missing):
        print(
            f">> Phase 5 shot {shot_index + 1}/{len(missing)} "
            f"cam{camera} f{start}",
            flush=True,
        )
        paths = [
            geometry.extract_video_frame(args, camera, frame)
            for frame in range(start, start + length)
        ]
        predictions, views, debug = geometry.run_fresh_stream(
            model, paths, len(paths), args
        )
        frames = []
        for offset, (prediction, view, debug_row, image_path) in enumerate(
            zip(predictions, views, debug, paths)
        ):
            dataset_frame = start + offset
            humans = geometry.layer_humans(prediction, view, debug_row, layer)
            humans.sort(key=lambda row: int(row["detection_index"]))
            assigned, assignment = geometry.assign_gt_identities(
                args,
                humans,
                geometry.camera_matrix(prediction),
                camera,
                dataset_frame,
                int(args.size),
                int(args.size),
            )
            feature = phase3.feature_frame(prediction, debug_row)
            processed_shape = tuple(
                int(value) for value in view["true_shape"][0].detach().cpu().numpy()
            )
            feature, crops = predicted_appearance(
                encoder, feature, humans, image_path, processed_shape
            )
            detections = [
                {key: value for key, value in human.items() if key != "identity"}
                for human in humans
            ]
            frames.append(
                {
                    "dataset_frame": int(dataset_frame),
                    "camera": int(camera),
                    "feature": feature,
                    "detections": detections,
                    "pose": geometry.camera_matrix(prediction).astype(np.float64),
                    "cloud": geometry.sampled_background_cloud(
                        prediction, view, humans, int(args.point_samples)
                    ),
                    "assignment_evaluator_only": assignment,
                    "labels_evaluator_only": label_row(assignment, int(feature["count"])),
                    "predicted_crops": crops,
                    "image_path": str(image_path),
                }
            )
        torch.save(
            {
                "schema_version": 1,
                "sequence": str(args.sequence),
                "camera": int(camera),
                "start_frame": int(start),
                "length": int(length),
                "frames": frames,
                "candidate_gt_usage": False,
                "inference_contract": {
                    "fresh_scene_camera_state": True,
                    "human_identity_state": False,
                    "full_frame_resize_only": True,
                    "da3": False,
                    "vggt": False,
                    "v11_4_scale": False,
                },
            },
            shot_path(args, camera, start, length),
        )
    del encoder, layer, model
    torch.cuda.empty_cache()
    print(f">> Phase 5 extraction seconds={time.perf_counter() - started:.2f}")


def load_stream(args: argparse.Namespace, spec: StreamSpec) -> list[dict]:
    return [
        torch.load(
            shot_path(args, camera, start, spec.shot_length),
            map_location="cpu",
            weights_only=False,
        )
        for camera, start in zip(spec.cameras, spec.shot_starts)
    ]


def majority(votes: dict[int, list[int]]) -> dict[int, int]:
    output = {}
    for track_id, labels in votes.items():
        valid = np.asarray([value for value in labels if int(value) >= 0], dtype=np.int64)
        if len(valid):
            values, counts = np.unique(valid, return_counts=True)
            output[int(track_id)] = int(values[int(np.argmax(counts))])
    return output


def native_ids(frame: dict) -> np.ndarray:
    values = np.asarray(frame["feature"]["track_ids"], dtype=np.int64).reshape(-1)
    return values


def no_cut_external_ids(
    state: ShotPersistentIdentityState,
    frame: dict,
    native_to_external: dict[int, int],
) -> np.ndarray:
    output = []
    for detection, native in enumerate(native_ids(frame)):
        key = int(native) if int(native) >= 0 else -(detection + 1)
        if key not in native_to_external:
            native_to_external[key] = state.allocate()
        output.append(native_to_external[key])
    return np.asarray(output, dtype=np.int64)


def update_votes(
    votes: dict[int, list[int]], external: np.ndarray, labels: np.ndarray
) -> None:
    for track_id, label in zip(external, labels):
        if int(track_id) >= 0 and int(label) >= 0:
            votes[int(track_id)].append(int(label))


def accepted_pair_signature(result: dict, target_map: np.ndarray | None = None) -> tuple:
    bank = np.asarray(result["bank"]["track_ids"], dtype=np.int64)
    target_map = (
        np.arange(len(result["target"]), dtype=np.int64)
        if target_map is None
        else np.asarray(target_map, dtype=np.int64)
    )
    return tuple(
        sorted(
            (
                int(bank[int(row["source_index"])]),
                int(target_map[int(row["target_index"])]),
            )
            for row in result["accepted_pairs"]
        )
    )


def permute_frame(frame: dict, order: np.ndarray) -> dict:
    output = copy.deepcopy(frame)
    count = int(frame["count"])
    order = np.asarray(order, dtype=np.int64)
    if sorted(order.tolist()) != list(range(count)):
        raise ValueError("Invalid detection permutation")
    output["track_ids"] = np.asarray(frame["track_ids"])[order]
    output["head_scores"] = np.asarray(frame["head_scores"])[order]
    if "appearance_valid" in frame:
        output["appearance_valid"] = np.asarray(frame["appearance_valid"])[order]
    output["features"] = {
        name: np.asarray(value)[order] for name, value in frame["features"].items()
    }
    return output


def permute_shots(shots: list[dict], mode: str, seed: int) -> list[dict]:
    output = []
    rng = np.random.default_rng(seed)
    for source_shot in shots:
        shot = dict(source_shot)
        shot["frames"] = []
        for source_row in source_shot["frames"]:
            row = dict(source_row)
            count = int(row["feature"]["count"])
            if mode == "reverse":
                order = np.arange(count, dtype=np.int64)[::-1]
            elif mode == "random":
                order = rng.permutation(count)
            else:
                raise ValueError(f"Unsupported stream permutation: {mode}")
            row["feature"] = permute_frame(row["feature"], order)
            row["labels_evaluator_only"] = np.asarray(
                row["labels_evaluator_only"], dtype=np.int64
            )[order]
            shot["frames"].append(row)
        output.append(shot)
    return output


def semantic_pair_signature(
    result: dict, bank_labels: dict[int, int], target_labels: np.ndarray
) -> tuple[tuple[int, int], ...]:
    bank_ids = np.asarray(result["bank"]["track_ids"], dtype=np.int64)
    target_labels = np.asarray(target_labels, dtype=np.int64)
    return tuple(
        sorted(
            (
                int(bank_labels.get(int(bank_ids[int(row["source_index"])]), -1)),
                int(target_labels[int(row["target_index"])]),
            )
            for row in result["accepted_pairs"]
        )
    )


def cost_result(payload: dict, cost: np.ndarray) -> dict:
    pairs = []
    if cost.size:
        sources, targets = linear_sum_assignment(cost)
        pairs = [
            {
                "source_index": int(source),
                "target_index": int(target),
                "cost": float(cost[source, target]),
                "score": float(-cost[source, target]),
                "accepted": True,
            }
            for source, target in zip(sources, targets)
        ]
    return assignment_result(payload, cost, pairs)


def controlled_cost(cost: np.ndarray, mode: str, seed: int) -> np.ndarray:
    """Apply a control to the state cost before matching and committing it."""
    cost = np.asarray(cost, dtype=np.float64)
    if mode == "none" or len(cost) < 2:
        return cost
    if mode == "wrong_person_state":
        return np.roll(cost, shift=1, axis=0)
    if mode == "shuffled_state":
        return cost[np.random.default_rng(seed).permutation(len(cost))]
    if mode == "zero_state":
        return np.ones_like(cost)
    raise ValueError(f"Unsupported state control: {mode}")


def state_methods() -> list[StateMethod]:
    methods = [StateMethod(f"state_{mode}", mode) for mode in PROTOTYPES]
    methods.extend(
        StateMethod(f"state_mean_variance_floor{floor:.2f}", "mean", True, floor)
        for floor in DISPERSION_FLOORS
    )
    return methods


def aggregate(rows: list[dict]) -> dict:
    return phase3.aggregate_identity(rows)


def empty_metrics(matchable: int) -> dict:
    return {
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": int(matchable),
        "accepted": 0,
        "matchable": int(matchable),
        "id_switches": 0,
        "dustbin_true_positive": 0,
        "dustbin_false_positive": 0,
        "dustbin_false_negative": 0,
    }


def risk_gate(rows: list[dict]) -> tuple[dict, list[dict]]:
    margins = np.asarray([float(row["hypothesis_margin"]) for row in rows])
    finite = margins[np.isfinite(margins)]
    thresholds = [0.0]
    if len(finite):
        thresholds.extend(np.quantile(finite, np.linspace(0.05, 1.0, 40)).tolist())
    thresholds.append(float("inf"))
    candidates = []
    for threshold in sorted(set(float(value) for value in thresholds)):
        metrics = [
            row["identity"]
            if float(row["hypothesis_margin"]) >= threshold
            else empty_metrics(int(row["identity"]["matchable"]))
            for row in rows
        ]
        summary = aggregate(metrics)
        accepted = int(summary["accepted"])
        wrong = int(summary["false_positive"])
        active = [
            row
            for row in rows
            if float(row["hypothesis_margin"]) >= threshold
            and int(row["identity"]["accepted"]) >= 2
        ]
        candidates.append(
            {
                "threshold": float(threshold),
                "identity": summary,
                "accepted_precision": (
                    (accepted - wrong) / accepted if accepted else float("nan")
                ),
                "wrong_accept_rate": wrong / max(accepted, 1),
                "multi_activation_coverage": len(active) / max(len(rows), 1),
                "cut_all_matches_correct": (
                    float(
                        np.mean(
                            [row["identity"]["false_positive"] == 0 for row in active]
                        )
                    )
                    if active
                    else float("nan")
                ),
            }
        )
    candidates.sort(
        key=lambda row: (
            row["identity"]["false_positive"] == 0,
            row["multi_activation_coverage"],
            row["identity"]["recall_at_1"],
            row["identity"]["idf1"],
        ),
        reverse=True,
    )
    return candidates[0], candidates


def evaluate_stateless(
    previous_shot: dict,
    post_frame: dict,
    phase4_config: PrecisionGateConfig,
) -> dict[str, dict]:
    pre_frames = [copy.deepcopy(row["feature"]) for row in previous_shot["frames"]]
    pre_labels = [row["labels_evaluator_only"] for row in previous_shot["frames"]]
    post_feature = post_frame["feature"]
    bank_labels = majority_track_labels(pre_labels, pre_frames)
    unfiltered = match_identity_bank(
        pre_frames,
        post_feature,
        MatchConfig(FEATURE, "mean", "cosine", "hungarian"),
    )
    base, signals = precision_signals(pre_frames, post_feature, phase4_config)
    gated = apply_precision_gate(base, signals, phase4_config)
    target_labels = post_frame["labels_evaluator_only"]
    return {
        "stateless_unfiltered": evaluate_assignment(
            unfiltered, bank_labels, target_labels
        ),
        "phase4_stateless_precision": evaluate_assignment(
            gated, bank_labels, target_labels
        ),
    }


def evaluate_stream_method(
    args: argparse.Namespace,
    spec: StreamSpec,
    shots: list[dict],
    method: StateMethod,
    acceptance_margin: float | None = None,
    control_mode: str = "none",
) -> dict:
    state = ShotPersistentIdentityState(
        ttl=int(args.state_ttl),
        history_size=int(args.history_size),
        max_tracks=int(args.max_tracks),
    )
    votes: dict[int, list[int]] = defaultdict(list)
    native_to_external: dict[int, int] = {}
    cuts = []
    global_time = 0
    for shot_index, shot in enumerate(shots):
        for frame_index, row in enumerate(shot["frames"]):
            feature = row["feature"]
            labels = row["labels_evaluator_only"]
            if shot_index == 0 and frame_index == 0:
                external = state.bootstrap(feature, global_time)
                native_to_external = {
                    int(native): int(track)
                    for native, track in zip(native_ids(row), external)
                    if int(native) >= 0
                }
            elif frame_index == 0:
                bank_labels = majority(votes)
                payload, cost = state.cost_matrix(
                    feature,
                    FEATURE,
                    method.prototype,
                    global_time,
                    method.track_normalized,
                    method.dispersion_floor,
                )
                cost = controlled_cost(
                    cost,
                    control_mode,
                    int(args.seed) + len(cuts),
                )
                hypotheses = enumerate_topk_hypotheses(
                    payload, cost, top_k=int(args.top_k)
                )
                winner = hypotheses[0]
                candidate_result = winner["result"]
                topk_metrics = [
                    evaluate_assignment(row["result"], bank_labels, labels)
                    for row in hypotheses
                ]
                topk_gt = any(
                    metric["false_positive"] == 0
                    and metric["true_positive"] == metric["matchable"]
                    for metric in topk_metrics
                )
                margin = (
                    float(hypotheses[1]["identity_cost"] - winner["identity_cost"])
                    if len(hypotheses) > 1
                    else float("inf")
                )
                accepted = bool(
                    acceptance_margin is None or margin >= float(acceptance_margin)
                )
                result = (
                    candidate_result
                    if accepted
                    else assignment_result(payload, cost, [])
                )
                identity = evaluate_assignment(result, bank_labels, labels)
                permutation_equal = []
                orders = [
                    np.arange(int(feature["count"]), dtype=np.int64)[::-1],
                    np.random.default_rng(
                        int(args.seed) + len(cuts)
                    ).permutation(int(feature["count"])),
                ]
                direct_signature = accepted_pair_signature(candidate_result)
                for order in orders:
                    permuted = permute_frame(feature, order)
                    perm_payload, perm_cost = state.cost_matrix(
                        permuted,
                        FEATURE,
                        method.prototype,
                        global_time,
                        method.track_normalized,
                        method.dispersion_floor,
                    )
                    perm_cost = controlled_cost(
                        perm_cost,
                        control_mode,
                        int(args.seed) + len(cuts),
                    )
                    perm_winner = enumerate_topk_hypotheses(
                        perm_payload, perm_cost, top_k=int(args.top_k)
                    )[0]["result"]
                    permutation_equal.append(
                        accepted_pair_signature(perm_winner, order) == direct_signature
                    )
                before = state.snapshot()
                external = state.commit(feature, result, global_time)
                after = state.snapshot()
                current_native = native_ids(row)
                native_to_external = {
                    int(native): int(track)
                    for native, track in zip(current_native, external)
                    if int(native) >= 0
                }
                cuts.append(
                    {
                        "stream": spec.name,
                        "cut_index": shot_index,
                        "source_camera": int(spec.cameras[shot_index - 1]),
                        "target_camera": int(spec.cameras[shot_index]),
                        "dataset_frame": int(row["dataset_frame"]),
                        "identity": identity,
                        "candidate_identity": evaluate_assignment(
                            candidate_result, bank_labels, labels
                        ),
                        "identity_accepted": accepted,
                        "topk_gt_assignment": bool(topk_gt),
                        "topk_metrics": topk_metrics,
                        "hypothesis_margin": margin,
                        "winner_identity_cost": float(winner["identity_cost"]),
                        "hypothesis_count": len(hypotheses),
                        "permutation_equal": permutation_equal,
                        "state_before": before,
                        "state_after": after,
                        "accepted_pairs": result["accepted_pairs"],
                        "semantic_pair_signature": semantic_pair_signature(
                            result, bank_labels, labels
                        ),
                    }
                )
            else:
                external = no_cut_external_ids(state, row, native_to_external)
                valid = np.asarray(
                    feature.get(
                        "appearance_valid", np.ones(int(feature["count"]), dtype=bool)
                    ),
                    dtype=bool,
                )
                state.observe(feature, external, global_time, valid=valid)
            update_votes(votes, external, labels)
            global_time += 1
    identity = aggregate([row["identity"] for row in cuts])
    candidate_identity = aggregate([row["candidate_identity"] for row in cuts])
    risk_selected, risk_rows = risk_gate(
        [{**row, "identity": row["candidate_identity"]} for row in cuts]
    )
    permutation = [value for row in cuts for value in row["permutation_equal"]]
    return {
        "method": asdict(method),
        "stream": spec.name,
        "control_mode": str(control_mode),
        "acceptance_margin": (
            None if acceptance_margin is None else float(acceptance_margin)
        ),
        "identity": identity,
        "candidate_identity": candidate_identity,
        "multi_activation_coverage": float(
            np.mean([row["identity"]["accepted"] >= 2 for row in cuts])
        ),
        "wrong_accept_rate": (
            identity["false_positive"] / max(identity["accepted"], 1)
        ),
        "topk_gt_assignment_recall": float(
            np.mean([row["topk_gt_assignment"] for row in cuts])
        ),
        "permutation_agreement": float(np.mean(permutation)) if permutation else 1.0,
        "risk_selected": risk_selected,
        "risk_rows": risk_rows,
        "cuts": cuts,
        "final_state": state.snapshot(),
    }


def merge_method_streams(rows: list[dict]) -> dict:
    cuts = [cut for row in rows for cut in row["cuts"]]
    identity = aggregate([cut["identity"] for cut in cuts])
    selected, frontier = risk_gate(cuts)
    permutation = [value for cut in cuts for value in cut["permutation_equal"]]
    return {
        "method": rows[0]["method"],
        "stream_count": len(rows),
        "cut_count": len(cuts),
        "identity": identity,
        "topk_gt_assignment_recall": float(
            np.mean([cut["topk_gt_assignment"] for cut in cuts])
        ),
        "permutation_agreement": float(np.mean(permutation)) if permutation else 1.0,
        "risk_selected": selected,
        "risk_frontier": frontier,
        "streams": rows,
    }


def merge_operational_streams(rows: list[dict]) -> dict:
    cuts = [cut for row in rows for cut in row["cuts"]]
    identity = aggregate([cut["identity"] for cut in cuts])
    permutation = [value for cut in cuts for value in cut["permutation_equal"]]
    return {
        "method": rows[0]["method"],
        "control_mode": rows[0]["control_mode"],
        "acceptance_margin": rows[0]["acceptance_margin"],
        "stream_count": len(rows),
        "cut_count": len(cuts),
        "identity": identity,
        "multi_activation_coverage": float(
            np.mean([cut["identity"]["accepted"] >= 2 for cut in cuts])
        ),
        "wrong_accept_rate": (
            identity["false_positive"] / max(identity["accepted"], 1)
        ),
        "cut_all_matches_correct": float(
            np.mean(
                [
                    cut["identity"]["false_positive"] == 0
                    for cut in cuts
                    if cut["identity"]["accepted"] >= 2
                ]
            )
        )
        if any(cut["identity"]["accepted"] >= 2 for cut in cuts)
        else float("nan"),
        "topk_gt_assignment_recall": float(
            np.mean([cut["topk_gt_assignment"] for cut in cuts])
        ),
        "permutation_agreement": float(np.mean(permutation)) if permutation else 1.0,
        "streams": rows,
    }


def operational_summary(report: dict) -> dict:
    return {
        key: report[key]
        for key in (
            "acceptance_margin",
            "identity",
            "multi_activation_coverage",
            "wrong_accept_rate",
            "cut_all_matches_correct",
            "topk_gt_assignment_recall",
            "permutation_agreement",
        )
    }


def candidate_margins(report: dict, count: int = 32) -> list[float]:
    values = np.asarray(
        [
            float(cut["hypothesis_margin"])
            for stream in report["streams"]
            for cut in stream["cuts"]
        ],
        dtype=np.float64,
    )
    finite = values[np.isfinite(values)]
    thresholds = [0.0, float("inf")]
    if len(finite):
        thresholds.extend(
            np.quantile(finite, np.linspace(0.0, 1.0, min(count, len(finite))))
        )
    return sorted(set(float(value) for value in thresholds))


def causal_margin_search(
    args: argparse.Namespace,
    specs: list[StreamSpec],
    shots_by_stream: dict[str, list[dict]],
    method: StateMethod,
    discovery: dict,
) -> tuple[dict, list[dict]]:
    candidates = []
    for threshold in candidate_margins(discovery):
        rows = [
            evaluate_stream_method(
                args,
                spec,
                shots_by_stream[spec.name],
                method,
                acceptance_margin=threshold,
            )
            for spec in specs
        ]
        report = merge_operational_streams(rows)
        candidates.append(report)
    candidates.sort(
        key=lambda row: (
            row["identity"]["false_positive"] == 0,
            row["multi_activation_coverage"],
            row["identity"]["recall_at_1"],
            row["identity"]["idf1"],
        ),
        reverse=True,
    )
    return candidates[0], [operational_summary(row) for row in candidates]


def stream_permutation_audit(
    args: argparse.Namespace,
    specs: list[StreamSpec],
    shots_by_stream: dict[str, list[dict]],
    method: StateMethod,
    acceptance_margin: float,
    baseline: dict,
) -> dict:
    baseline_signatures = {
        (stream["stream"], int(cut["cut_index"])): tuple(
            tuple(pair) for pair in cut["semantic_pair_signature"]
        )
        for stream in baseline["streams"]
        for cut in stream["cuts"]
    }
    rows = []
    for mode_index, mode in enumerate(("reverse", "random")):
        equal = []
        for spec in specs:
            shots = permute_shots(
                shots_by_stream[spec.name],
                mode,
                int(args.seed) + 1000 * (mode_index + 1),
            )
            replay = evaluate_stream_method(
                args,
                spec,
                shots,
                method,
                acceptance_margin=acceptance_margin,
            )
            for cut in replay["cuts"]:
                key = (spec.name, int(cut["cut_index"]))
                signature = tuple(
                    tuple(pair) for pair in cut["semantic_pair_signature"]
                )
                equal.append(signature == baseline_signatures[key])
        rows.append(
            {
                "mode": mode,
                "cut_count": len(equal),
                "agreement": float(np.mean(equal)) if equal else 1.0,
            }
        )
    return {
        "audits": rows,
        "agreement": min(row["agreement"] for row in rows) if rows else 1.0,
    }


def analyze(args: argparse.Namespace, specs: list[StreamSpec]) -> dict:
    if str(args.role) != "development":
        raise RuntimeError(
            "This entrypoint selects the Phase-5 state method and margin and is "
            "development-only. Frozen evaluation was not run because the joint "
            "WHO-WHERE development gate failed."
        )
    frozen = json.loads(args.phase4_config.read_text(encoding="utf-8"))
    phase4_config = PrecisionGateConfig(**frozen["precision_gate"])
    stateless_rows: dict[str, list[dict]] = defaultdict(list)
    method_rows: dict[str, list[dict]] = defaultdict(list)
    shots_by_stream = {spec.name: load_stream(args, spec) for spec in specs}
    for stream_index, spec in enumerate(specs):
        print(f">> Stage 0 stream {stream_index + 1}/{len(specs)} {spec.name}")
        shots = shots_by_stream[spec.name]
        for shot_index in range(1, len(shots)):
            for name, metrics in evaluate_stateless(
                shots[shot_index - 1], shots[shot_index]["frames"][0], phase4_config
            ).items():
                stateless_rows[name].append(metrics)
        for method in state_methods():
            method_rows[method.name].append(
                evaluate_stream_method(args, spec, shots, method)
            )
    stateless = {
        name: {
            "identity": aggregate(rows),
            "multi_activation_coverage": float(
                np.mean([row["accepted"] >= 2 for row in rows])
            ),
            "wrong_accept_rate": (
                sum(row["false_positive"] for row in rows)
                / max(sum(row["accepted"] for row in rows), 1)
            ),
        }
        for name, rows in sorted(stateless_rows.items())
    }
    methods = {
        name: merge_method_streams(rows) for name, rows in method_rows.items()
    }
    ranked = sorted(
        methods.values(),
        key=lambda row: (
            row["identity"]["idf1"],
            row["risk_selected"]["identity"]["false_positive"] == 0,
            row["risk_selected"]["multi_activation_coverage"],
            row["topk_gt_assignment_recall"],
        ),
        reverse=True,
    )
    best = ranked[0]
    selected_method = StateMethod(**best["method"])
    operational, causal_risk_frontier = causal_margin_search(
        args, specs, shots_by_stream, selected_method, best
    )
    selected_margin = float(operational["acceptance_margin"])
    order_audit = stream_permutation_audit(
        args,
        specs,
        shots_by_stream,
        selected_method,
        selected_margin,
        operational,
    )
    controls = {}
    for control_mode in ("wrong_person_state", "shuffled_state", "zero_state"):
        rows = [
            evaluate_stream_method(
                args,
                spec,
                shots_by_stream[spec.name],
                selected_method,
                acceptance_margin=selected_margin,
                control_mode=control_mode,
            )
            for spec in specs
        ]
        controls[control_mode] = merge_operational_streams(rows)
    stateless_idf1 = stateless["stateless_unfiltered"]["identity"]["idf1"]
    phase4_coverage = stateless["phase4_stateless_precision"][
        "multi_activation_coverage"
    ]
    state_gain = float(best["identity"]["idf1"] - stateless_idf1)
    coverage_gain = float(
        operational["multi_activation_coverage"] - phase4_coverage
    )
    stage0_pass = bool(
        state_gain >= 0.03
        and operational["identity"]["false_positive"] == 0
        and coverage_gain >= 0.05
        and operational["permutation_agreement"] >= 0.999
        and order_audit["agreement"] >= 0.999
    )
    stage1_pass = bool(
        stage0_pass and operational["topk_gt_assignment_recall"] >= 0.90
    )
    decision = (
        "continue_to_joint_who_where"
        if stage1_pass
        else "stop_before_joint_search"
    )
    report = {
        "experiment": "V13 Phase 5 Causal Shot-Persistent Identity Stage 0/1",
        "sequence": str(args.sequence),
        "role": str(args.role),
        "stream_count": len(specs),
        "cut_count": sum(len(spec.cameras) - 1 for spec in specs),
        "protocol": {
            "streams": [asdict(spec) for spec in specs],
            "human3r_scene_camera_reset_every_shot": True,
            "external_identity_state_persists": True,
            "normal_frame_updates_use_native_tracking": True,
            "stage0_identity_only": True,
            "boundary_solver_run": False,
            "commit_after_fixed_identity_margin": True,
            "top_k": int(args.top_k),
            "ttl": int(args.state_ttl),
            "fixed_history_size": int(args.history_size),
            "future_frames": False,
        },
        "candidate_gt_usage": {
            "feature": False,
            "state": False,
            "matching": False,
            "hypothesis_generation": False,
            "commit": False,
            "evaluation": True,
        },
        "stateless": stateless,
        "stateful_discovery_ranked": ranked,
        "stateful_operational": operational,
        "causal_risk_frontier": causal_risk_frontier,
        "detection_order_audit": order_audit,
        "state_controls": controls,
        "selected_method": best["method"],
        "selected_acceptance_margin": selected_margin,
        "gates": {
            "stage0_required_idf1_gain": 0.03,
            "stage0_required_coverage_gain": 0.05,
            "stage0_pass": stage0_pass,
            "stage1_required_topk_recall": 0.90,
            "stage1_pass": stage1_pass,
        },
        "observed": {
            "stateful_idf1_gain": state_gain,
            "precision_coverage_gain": coverage_gain,
            "topk_gt_assignment_recall": operational[
                "topk_gt_assignment_recall"
            ],
            "permutation_agreement": operational["permutation_agreement"],
            "full_stream_permutation_agreement": order_audit["agreement"],
        },
        "decision": decision,
    }
    output = args.output_dir / "v13_phase5_stage0_state.json"
    output.write_text(
        json.dumps(jsonable(report), indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    markdown = [
        "# V13 Phase 5: Causal Identity State Stage 0/1",
        "",
        f"Decision: **{decision}**",
        "",
        f"- streams / cuts: `{report['stream_count']} / {report['cut_count']}`",
        f"- stateless IDF1: `{stateless_idf1:.4f}`",
        f"- best discovery stateful IDF1: `{best['identity']['idf1']:.4f}`",
        f"- discovery stateful gain: `{state_gain:+.4f}`",
        f"- fixed causal margin: `{selected_margin:.6f}`",
        f"- causal replay IDF1: `{operational['identity']['idf1']:.4f}`",
        f"- Phase 4 precision multi coverage: `{phase4_coverage:.4f}`",
        f"- state precision multi coverage: "
        f"`{operational['multi_activation_coverage']:.4f}`",
        f"- state wrong accepted: "
        f"`{operational['identity']['false_positive']}`",
        f"- Top-{args.top_k} GT assignment recall: "
        f"`{operational['topk_gt_assignment_recall']:.4f}`",
        f"- detection-order permutation agreement: "
        f"`{operational['permutation_agreement']:.4f}`",
        f"- full-stream permutation agreement: `{order_audit['agreement']:.4f}`",
        "",
        "GT identity was attached only after candidate generation for evaluation.",
    ]
    (args.output_dir / "v13_phase5_stage0_state.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    print(f">> Phase 5 Stage 0 report: {output}")
    print(f">> Decision: {decision}")
    return report


def main() -> None:
    args = configure(parse_args())
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    specs = selected_specs(args)
    if args.mode in {"extract", "all"}:
        extract(args, specs)
    if args.mode in {"analyze", "all"}:
        analyze(args, specs)


if __name__ == "__main__":
    main()
