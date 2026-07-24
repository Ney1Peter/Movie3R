#!/usr/bin/env python3
"""V13 Phase 4A: precision-first frozen appearance identity feasibility.

The deployable path uses only RGB, Human3R-predicted detection geometry, and
frozen person features.  GT identity/camera/body data are attached only by the
development selector and evaluator.  The Phase-2 Boundary solver is imported
unchanged.
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
from itertools import product
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import versions.v13.gt_id_consensus as geometry  # noqa: E402
import versions.v13.experiments.phase3_cross_shot_identity as phase3  # noqa: E402
from versions.v13.appearance_identity import (  # noqa: E402
    ENCODER_CHECKPOINT_SHA256,
    ENCODER_DIM,
    ENCODER_INPUT_SIZE,
    ENCODER_NAME,
    FrozenDinoAppearance,
    PrecisionGateConfig,
    apply_precision_gate,
    augment_frame,
    crop_predicted_bbox,
    precision_signals,
    signal_is_accepted,
)
from versions.v13.identity_bridge import (  # noqa: E402
    MatchConfig,
    evaluate_assignment,
    match_identity_bank,
    majority_track_labels,
)
from versions.v13.native_token_probe import jsonable  # noqa: E402


APPEARANCE_FEATURES = (
    "appearance",
    "appearance_beta",
    "appearance_pose",
    "appearance_beta_pose",
    "appearance_native",
)
PROBE_FEATURES = (
    "refined_human_tokens",
    "cut3r_head_tokens",
    "mhmr_head_tokens",
    "fused_human_prompts",
    "smpl_beta",
    "local_pose",
    *APPEARANCE_FEATURES,
)
PROTOTYPES = ("last", "mean", "medoid")
DISTANCES = ("raw_l2", "normalized_l2", "cosine")
MATCHERS = ("hungarian", "sinkhorn")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", choices=tuple(phase3.DEFAULT_PROTOCOLS), default="three")
    parser.add_argument("--role", choices=("development", "evaluation"), default="development")
    parser.add_argument("--mode", choices=("extract", "analyze", "all"), default="all")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("/data/wangzheng/iJCV-CODE/data/MultiHuman/Real-World-Capture/extracted"),
    )
    parser.add_argument(
        "--phase3_root", type=Path, default=ROOT / "output/v13/phase3_identity"
    )
    parser.add_argument(
        "--output_root", type=Path, default=ROOT / "output/v13/phase4_identity"
    )
    parser.add_argument("--geometry_cache_dir", type=Path)
    parser.add_argument(
        "--encoder_hub_dir",
        type=Path,
        default=Path("/data/wangzheng/.cache/torch/hub/facebookresearch_dinov2_main"),
    )
    parser.add_argument(
        "--encoder_checkpoint",
        type=Path,
        default=Path("/data/wangzheng/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth"),
    )
    parser.add_argument("--frozen_config", type=Path)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--bbox_padding", type=float, default=0.08)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--reuse_feature_probe",
        action="store_true",
        help="Reuse feature_probe.json while recomputing the frozen gate report.",
    )
    parser.add_argument("--seed", type=int, default=20260725)
    return parser.parse_args()


def configure(args: argparse.Namespace) -> argparse.Namespace:
    protocol = phase3.DEFAULT_PROTOCOLS[str(args.sequence)]
    args.timestamps = tuple(protocol["timestamps"])
    args.camera_pairs = tuple(protocol["camera_pairs"])
    args.offsets = tuple(protocol["offsets"])
    args.history_frames = 5
    args.geometry_cache_dir = args.geometry_cache_dir or protocol["geometry_cache"]
    args.phase3_dir = args.phase3_root / str(args.sequence)
    args.phase3_feature_dir = args.phase3_dir / "feature_cache"
    args.output_dir = args.output_root / str(args.sequence)
    args.appearance_cache_dir = args.output_dir / "appearance_cache"
    args.lattice_cache_dir = args.output_dir / "endpoint_lattice"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.appearance_cache_dir.mkdir(parents=True, exist_ok=True)
    args.lattice_cache_dir.mkdir(parents=True, exist_ok=True)
    args.frozen_config = args.frozen_config or (
        args.output_root / "three/frozen_precision_config.json"
        if str(args.role) == "development"
        else ROOT / "versions/v13/configs/phase4_precision_config.json"
    )
    geometry.IDENTITIES = geometry.SEQUENCE_IDENTITIES[str(args.sequence)]
    return args


def specs(args: argparse.Namespace) -> list[geometry.CaseSpec]:
    rows = geometry.case_specs(args)
    return rows[: int(args.max_cases)] if int(args.max_cases) > 0 else rows


def appearance_cache_path(args: argparse.Namespace, spec: geometry.CaseSpec) -> Path:
    return args.appearance_cache_dir / f"{spec.key}.pt"


def image_paths(args: argparse.Namespace, case: dict) -> list[Path]:
    pre_frames = [int(value) for value in case["pre_frames"]]
    source = int(case["source_camera"])
    target = int(case["target_camera"])
    post = int(case["post_frame"])
    return [
        args.phase3_dir / f"input_frames/cam{source}/{frame:06d}.jpg"
        for frame in pre_frames
    ] + [args.phase3_dir / f"input_frames/cam{target}/{post:06d}.jpg"]


def extract(args: argparse.Namespace, case_specs: list[geometry.CaseSpec]) -> None:
    missing = [
        spec
        for spec in case_specs
        if args.overwrite or not appearance_cache_path(args, spec).is_file()
    ]
    if not missing:
        print(">> Phase 4 appearance caches are complete", flush=True)
        return
    encoder = FrozenDinoAppearance(
        args.device,
        args.encoder_hub_dir,
        args.encoder_checkpoint,
        batch_size=int(args.batch_size),
    )
    valid_crops = invalid_crops = 0
    started = time.perf_counter()
    for case_index, spec in enumerate(missing):
        if case_index % 25 == 0 or case_index + 1 == len(missing):
            print(f">> appearance {case_index + 1}/{len(missing)} {spec.key}", flush=True)
        feature_path = args.phase3_feature_dir / f"{spec.key}.pt"
        geometry_path = args.geometry_cache_dir / f"{spec.key}.pt"
        feature = torch.load(feature_path, map_location="cpu", weights_only=False)
        raw = torch.load(geometry_path, map_location="cpu", weights_only=False)
        paths = image_paths(args, feature["case"])
        frame_payloads = []
        for frame_index, (frame, detection_frame, path) in enumerate(
            zip(feature["frames"], raw["humans"], paths)
        ):
            image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise FileNotFoundError(path)
            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            detections = phase3.detections_by_index(detection_frame)
            count = int(frame["count"])
            if set(detections) != set(range(count)):
                raise ValueError(f"{spec.key} frame {frame_index}: detection order mismatch")
            crops, crop_rows, crop_indices = [], [], []
            for detection_index in range(count):
                crop, row = crop_predicted_bbox(
                    image,
                    detections[detection_index]["bbox"],
                    (int(args.size), int(args.size)),
                    padding_ratio=float(args.bbox_padding),
                )
                crop_rows.append(row)
                if crop is not None:
                    crop_indices.append(detection_index)
                    crops.append(crop)
            embedding = np.zeros((count, ENCODER_DIM), dtype=np.float32)
            valid = np.zeros(count, dtype=bool)
            if crops:
                encoded = encoder.encode(crops)
                embedding[np.asarray(crop_indices, dtype=np.int64)] = encoded
                valid[np.asarray(crop_indices, dtype=np.int64)] = True
            valid_crops += int(valid.sum())
            invalid_crops += int((~valid).sum())
            frame_payloads.append(
                {
                    "appearance": embedding,
                    "valid": valid,
                    "crops": crop_rows,
                    "image_path": str(path),
                    "candidate_gt_usage": False,
                }
            )
        torch.save(
            {
                "case": feature["case"],
                "frames": frame_payloads,
                "encoder": {
                    "name": ENCODER_NAME,
                    "dimension": ENCODER_DIM,
                    "input_size": ENCODER_INPUT_SIZE,
                    "checkpoint": str(args.encoder_checkpoint),
                    "sha256": ENCODER_CHECKPOINT_SHA256,
                    "bbox_source": "Human3R predicted SMPL-X projection",
                    "bbox_padding": float(args.bbox_padding),
                },
                "candidate_gt_usage": False,
            },
            appearance_cache_path(args, spec),
        )
    del encoder
    torch.cuda.empty_cache()
    print(
        f">> appearance extraction {time.perf_counter() - started:.2f}s, "
        f"valid/invalid crops={valid_crops}/{invalid_crops}",
        flush=True,
    )


def load_cases(args: argparse.Namespace, case_specs: list[geometry.CaseSpec]) -> list[dict]:
    output = []
    for index, spec in enumerate(case_specs):
        feature = torch.load(
            args.phase3_feature_dir / f"{spec.key}.pt",
            map_location="cpu",
            weights_only=False,
        )
        appearance = torch.load(
            appearance_cache_path(args, spec), map_location="cpu", weights_only=False
        )
        raw, strict = phase3.geometry_caches(args, spec)
        frames = [
            augment_frame(frame, row["appearance"], row["valid"])
            for frame, row in zip(feature["frames"], appearance["frames"])
        ]
        labels = phase3.detection_labels(strict, feature)
        output.append(
            {
                "case": feature["case"],
                "features": {"frames": frames},
                "labels": labels,
                "raw": raw,
                "strict": strict,
            }
        )
        if (index + 1) % 50 == 0:
            print(f">> loaded {index + 1}/{len(case_specs)} Phase 4 cases", flush=True)
    return output


def probe_configs() -> list[MatchConfig]:
    return [
        MatchConfig(feature, prototype, distance, matcher)
        for feature, prototype, distance, matcher in product(
            PROBE_FEATURES, PROTOTYPES, DISTANCES, MATCHERS
        )
    ]


def feature_probe(cases: list[dict]) -> list[dict]:
    rows = []
    for index, config in enumerate(probe_configs()):
        rows.append(phase3.evaluate_config(cases, config))
        if (index + 1) % 50 == 0:
            print(f">> feature probe {index + 1}/{len(probe_configs())}", flush=True)
    return phase3.rank_probe(rows)


def select_primary(ranked: list[dict]) -> PrecisionGateConfig:
    appearance = [row for row in ranked if row["config"]["feature"] in APPEARANCE_FEATURES]
    if not appearance:
        raise ValueError("No appearance feature result")
    config = appearance[0]["config"]
    return PrecisionGateConfig(
        feature=str(config["feature"]),
        prototype=str(config["prototype"]),
        distance=str(config["distance"]),
        min_valid_observations=3,
        require_mutual=True,
    )


def prepare_identity_cases(cases: list[dict], config: PrecisionGateConfig) -> list[dict]:
    output = []
    for case in cases:
        frames = case["features"]["frames"]
        base, signals = precision_signals(frames[:-1], frames[-1], config)
        bank_labels = majority_track_labels(case["labels"][:-1], frames[:-1])
        track_ids = np.asarray(base["bank"]["track_ids"], dtype=np.int64)
        target_labels = np.asarray(case["labels"][-1], dtype=np.int64)
        for signal in signals:
            source_label = bank_labels.get(int(signal["track_id"]), -1)
            target_label = int(target_labels[int(signal["target_index"])])
            signal["correct_evaluator_only"] = bool(
                source_label >= 0 and source_label == target_label
            )
            signal["source_gt_evaluator_only"] = int(source_label)
            signal["target_gt_evaluator_only"] = int(target_label)
        output.append(
            {
                "case": case,
                "base": base,
                "signals": signals,
                "config": config,
                "bank_labels": bank_labels,
                "target_labels": target_labels,
                "track_ids": track_ids,
            }
        )
    return output


def finite_quantiles(values: list[float], quantiles: tuple[float, ...]) -> list[float]:
    array = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if not len(array):
        return [float("inf")]
    return sorted({float(np.quantile(array, value)) for value in quantiles})


def gate_candidates(prepared: list[dict], base: PrecisionGateConfig) -> list[PrecisionGateConfig]:
    signals = [signal for case in prepared for signal in case["signals"]]
    distances = finite_quantiles(
        [row["primary_distance"] for row in signals],
        (0.35, 0.50, 0.65, 0.75, 0.85, 0.92, 0.97, 1.0),
    )
    margins = [0.0] + finite_quantiles(
        [row["primary_margin"] for row in signals], (0.10, 0.25, 0.40, 0.55)
    )
    betas = finite_quantiles(
        [row["beta_distance"] for row in signals], (0.75, 0.90, 1.0)
    ) + [float("inf")]
    poses = finite_quantiles(
        [row["pose_distance"] for row in signals], (0.75, 0.90, 1.0)
    ) + [float("inf")]
    votes = (0.6, 0.8, 1.0)
    output = []
    seen = set()
    for distance, margin, vote, beta, pose in product(
        distances, margins, votes, sorted(set(betas)), sorted(set(poses))
    ):
        config = replace(
            base,
            max_primary_distance=float(distance),
            min_primary_margin=float(margin),
            min_vote_fraction=float(vote),
            max_beta_distance=float(beta),
            max_pose_distance=float(pose),
        )
        key = tuple(asdict(config).items())
        if key not in seen:
            output.append(config)
            seen.add(key)
    return output


def gate_identity_summary(prepared: list[dict], config: PrecisionGateConfig) -> dict:
    identity_rows = []
    any_active = all_correct = multi_active = multi_correct = 0
    accepted_counts = []
    masks = []
    grouped: dict[str, dict[str, list]] = {
        "offset": defaultdict(list),
        "camera_pair": defaultdict(list),
    }
    for case in prepared:
        result = apply_precision_gate(case["base"], case["signals"], config)
        metrics = evaluate_assignment(result, case["bank_labels"], case["target_labels"])
        identity_rows.append(metrics)
        accepted = len(result["accepted_pairs"])
        accepted_counts.append(accepted)
        spec = case["case"]["case"]
        grouped["offset"][str(int(spec["offset"]))].append((metrics, accepted))
        grouped["camera_pair"][
            f"{int(spec['source_camera'])}->{int(spec['target_camera'])}"
        ].append((metrics, accepted))
        mask = 0
        accepted_keys = {
            (int(row["source_index"]), int(row["target_index"]))
            for row in result["accepted_pairs"]
        }
        for signal in case["signals"]:
            key = (int(signal["source_index"]), int(signal["target_index"]))
            if key in accepted_keys:
                mask |= 1 << int(signal["pair_index"])
        masks.append(mask)
        if accepted:
            any_active += 1
            all_correct += int(metrics["false_positive"] == 0)
        if accepted >= 2:
            multi_active += 1
            multi_correct += int(metrics["false_positive"] == 0)
    aggregate = phase3.aggregate_identity(identity_rows)
    accepted = int(aggregate["accepted"])
    wrong = int(aggregate["false_positive"])

    def group_summary(rows: list[tuple[dict, int]]) -> dict:
        identity = phase3.aggregate_identity([row[0] for row in rows])
        group_accepted = int(identity["accepted"])
        group_wrong = int(identity["false_positive"])
        return {
            "case_count": len(rows),
            "identity": identity,
            "accepted_precision": (
                (group_accepted - group_wrong) / group_accepted
                if group_accepted
                else float("nan")
            ),
            "wrong_accept_rate": group_wrong / max(group_accepted, 1),
            "multi_activation_coverage": float(
                np.mean([accepted_count >= 2 for _, accepted_count in rows])
            ),
            "fallback_counts": {
                "multi": int(sum(value >= 2 for _, value in rows)),
                "single": int(sum(value == 1 for _, value in rows)),
                "fixed": int(sum(value == 0 for _, value in rows)),
            },
        }

    return {
        "config": asdict(config),
        "identity": aggregate,
        "accepted_precision": (accepted - wrong) / max(accepted, 1),
        "wrong_accept_rate": wrong / max(accepted, 1),
        "wrong_accept_per_matchable": wrong / max(int(aggregate["matchable"]), 1),
        "accepted_coverage": accepted / max(int(aggregate["matchable"]), 1),
        "cut_all_matches_correct": all_correct / max(any_active, 1),
        "multi_activation_coverage": multi_active / max(len(prepared), 1),
        "multi_all_matches_correct": multi_correct / max(multi_active, 1),
        "fallback_counts": {
            "multi": int(sum(value >= 2 for value in accepted_counts)),
            "single": int(sum(value == 1 for value in accepted_counts)),
            "fixed": int(sum(value == 0 for value in accepted_counts)),
        },
        "by_offset": {
            key: group_summary(rows)
            for key, rows in sorted(grouped["offset"].items(), key=lambda row: int(row[0]))
        },
        "by_camera_pair": {
            key: group_summary(rows)
            for key, rows in sorted(grouped["camera_pair"].items())
        },
        "masks": masks,
    }


def select_gate(prepared: list[dict], base: PrecisionGateConfig) -> tuple[dict, list[dict]]:
    rows = []
    configs = gate_candidates(prepared, base)
    for index, config in enumerate(configs):
        rows.append(gate_identity_summary(prepared, config))
        if (index + 1) % 500 == 0:
            print(f">> precision gate {index + 1}/{len(configs)}", flush=True)
    rows.sort(
        key=lambda row: (
            row["identity"]["false_positive"] == 0,
            row["accepted_precision"],
            row["multi_activation_coverage"],
            row["accepted_coverage"],
            row["identity"]["idf1"],
        ),
        reverse=True,
    )
    return rows[0], rows


def solution_from_candidates(candidates: dict, identities: tuple[str, ...], cache: dict) -> dict:
    if len(identities) >= 2:
        rotation, translation = geometry.solve_consensus(candidates, identities, "mean_raw_t")
        return {
            "rotation": rotation,
            "translation": translation,
            "identities": identities,
            "fallback": "multi-human",
        }
    if len(identities) == 1:
        candidate = candidates[identities[0]]
        return {
            "rotation": candidate["rotation"],
            "translation": candidate["translation"],
            "identities": identities,
            "fallback": "single-human",
        }
    return phase3.fallback_solution(cache)


def controlled_primary_result(
    frames: list[dict], config: PrecisionGateConfig, mode: str, seed: int
) -> dict:
    controlled = copy.deepcopy(frames[:-1])
    rng = np.random.default_rng(int(seed))
    for frame in controlled:
        count = int(frame["count"])
        if not count:
            continue
        value = np.asarray(frame["features"][config.feature])
        if mode == "shuffle":
            frame["features"][config.feature] = value[rng.permutation(count)]
        elif mode == "zero":
            frame["features"][config.feature] = np.zeros_like(value)
        else:
            raise ValueError(mode)
    return match_identity_bank(
        controlled,
        frames[-1],
        MatchConfig(
            feature=config.feature,
            prototype=config.prototype,
            distance=config.distance,
            matcher="hungarian",
            max_cost=float("inf"),
        ),
    )


def evaluate_control_result(raw: dict, strict: dict, frames: list[dict], result: dict) -> dict:
    cache, _ = phase3.automatic_cache(raw, frames, result)
    candidates = geometry.human_candidates(cache)
    identities = tuple(identity for identity in geometry.IDENTITIES if identity in candidates)
    solution = solution_from_candidates(candidates, identities, cache)
    metrics = geometry.evaluate_solution(strict, solution)
    metrics["fallback"] = solution.get("fallback")
    return metrics


def build_endpoint_lattice(args: argparse.Namespace, prepared: list[dict]) -> None:
    for index, row in enumerate(prepared):
        key = row["case"]["case"]["key"]
        path = args.lattice_cache_dir / f"{key}.pt"
        if path.is_file() and not args.overwrite:
            cached = torch.load(path, map_location="cpu", weights_only=False)
            if int(cached.get("schema_version", 1)) >= 2:
                continue
        raw = row["case"]["raw"]
        strict = row["case"]["strict"]
        auto_cache, mapping = phase3.automatic_cache(
            raw, row["case"]["features"]["frames"], row["base"]
        )
        candidates = geometry.human_candidates(auto_cache)
        slots = [item["slot"] for item in mapping["accepted"]]
        subset = {}
        for mask in range(1 << len(slots)):
            identities = tuple(
                slot
                for pair_index, slot in enumerate(slots)
                if mask & (1 << pair_index) and slot in candidates
            )
            solution = solution_from_candidates(candidates, identities, auto_cache)
            metrics = geometry.evaluate_solution(strict, solution)
            metrics["fallback"] = solution.get("fallback")
            subset[mask] = metrics
        gt_candidates = geometry.human_candidates(strict)
        gt_methods = geometry.method_solutions(gt_candidates)
        gt_name = "naive_mean" if "naive_mean" in gt_methods else "single_highest_confidence"
        frames = row["case"]["features"]["frames"]
        base_result = row["base"]
        wrong_result = phase3.cyclic_wrong_result(base_result)
        shuffled_result = controlled_primary_result(
            frames, row["config"], "shuffle", int(args.seed) + index
        )
        zero_result = controlled_primary_result(
            frames, row["config"], "zero", int(args.seed) + index
        )
        controls = {
            "automatic_unfiltered": base_result,
            "wrong_person": wrong_result,
            "shuffled_memory": shuffled_result,
            "zero_memory": zero_result,
        }
        control_metrics = {
            name: evaluate_control_result(raw, strict, frames, result)
            for name, result in controls.items()
        }
        control_identity = {
            name: evaluate_assignment(result, row["bank_labels"], row["target_labels"])
            for name, result in controls.items()
        }
        torch.save(
            {
                "schema_version": 2,
                "case": row["case"]["case"],
                "slots": slots,
                "subset": subset,
                "single": geometry.evaluate_solution(
                    strict, gt_methods["single_highest_confidence"]
                ),
                "gt_id": geometry.evaluate_solution(strict, gt_methods[gt_name]),
                "controls": control_metrics,
                "control_identity": control_identity,
            },
            path,
        )
        if (index + 1) % 25 == 0 or index + 1 == len(prepared):
            print(f">> endpoint lattice {index + 1}/{len(prepared)}", flush=True)


METRIC_FIELDS = (
    "camera_translation_error_m",
    "camera_rotation_error_deg",
    "camera_composite",
    "human_root_error_m",
    "human_joint_error_m",
    "human_vertex_error_m",
    "pairwise_distance_error_m",
    "pairwise_vector_error_m",
)


def metric_rows_summary(rows: list[dict]) -> dict:
    return {
        "case_count": len(rows),
        **{
            field: geometry.finite_distribution([float(row[field]) for row in rows])
            for field in METRIC_FIELDS
        },
        "catastrophic_rate": float(np.mean([row["catastrophic"] for row in rows])),
        "fallback_counts": {
            name: int(sum(row.get("fallback") == name for row in rows))
            for name in (
                "multi-human",
                "single-human",
                "identity-free camera-continuity Fixed Explicit",
            )
        },
    }


def attach_endpoint(args: argparse.Namespace, gate_row: dict, prepared: list[dict]) -> dict:
    selected, single, gt_id, fixed = [], [], [], []
    selected_active, single_active, gt_active = [], [], []
    controls: dict[str, list[dict]] = defaultdict(list)
    control_identity: dict[str, list[dict]] = defaultdict(list)
    for row, mask in zip(prepared, gate_row["masks"]):
        key = row["case"]["case"]["key"]
        lattice = torch.load(
            args.lattice_cache_dir / f"{key}.pt", map_location="cpu", weights_only=False
        )
        selected.append(lattice["subset"][int(mask)])
        fixed.append(lattice["subset"][0])
        single.append(lattice["single"])
        gt_id.append(lattice["gt_id"])
        if int(mask).bit_count() >= 2:
            selected_active.append(lattice["subset"][int(mask)])
            single_active.append(lattice["single"])
            gt_active.append(lattice["gt_id"])
        for name, metrics in lattice.get("controls", {}).items():
            controls[name].append(metrics)
        for name, metrics in lattice.get("control_identity", {}).items():
            control_identity[name].append(metrics)
    methods = {
        "single_highest_confidence": metric_rows_summary(single),
        "gt_id_uniform": metric_rows_summary(gt_id),
        "precision_first": metric_rows_summary(selected),
        "no_identity_fixed": metric_rows_summary(fixed),
    }
    methods.update(
        {name: metric_rows_summary(rows) for name, rows in controls.items()}
    )
    single_mean = methods["single_highest_confidence"]["camera_composite"]["mean"]
    gt_mean = methods["gt_id_uniform"]["camera_composite"]["mean"]
    selected_mean = methods["precision_first"]["camera_composite"]["mean"]
    denominator = single_mean - gt_mean
    active_methods = {
        "single_highest_confidence": metric_rows_summary(single_active),
        "gt_id_uniform": metric_rows_summary(gt_active),
        "precision_first": metric_rows_summary(selected_active),
    }
    active_single = active_methods["single_highest_confidence"]["camera_composite"]["mean"]
    active_gt = active_methods["gt_id_uniform"]["camera_composite"]["mean"]
    active_selected = active_methods["precision_first"]["camera_composite"]["mean"]
    active_denominator = active_single - active_gt
    return {
        **{key: value for key, value in gate_row.items() if key != "masks"},
        "methods": methods,
        "control_identity": {
            name: phase3.aggregate_identity(rows)
            for name, rows in control_identity.items()
        },
        "activated_multi": {
            "case_count": len(selected_active),
            "methods": active_methods,
            "gt_id_gain_retention": (
                float((active_single - active_selected) / active_denominator)
                if active_denominator > 1e-12
                else float("nan")
            ),
        },
        "gt_id_gain_retention": (
            float((single_mean - selected_mean) / denominator)
            if denominator > 1e-12
            else float("nan")
        ),
    }


def risk_frontier(rows: list[dict], limit: int = 16) -> list[dict]:
    ordered = sorted(
        rows,
        key=lambda row: (
            row["multi_activation_coverage"],
            -row["wrong_accept_rate"],
            row["accepted_precision"],
        ),
    )
    if len(ordered) <= limit:
        return ordered
    indices = np.linspace(0, len(ordered) - 1, limit).round().astype(int)
    return [ordered[index] for index in sorted(set(indices.tolist()))]


def plot_feature_probe(ranked: list[dict], output: Path) -> None:
    best = {}
    for row in ranked:
        best.setdefault(row["config"]["feature"], row)
    rows = sorted(best.values(), key=lambda row: row["identity"]["idf1"], reverse=True)
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    names = [row["config"]["feature"] for row in rows]
    values = [row["identity"]["idf1"] for row in rows]
    colors = ["#2f6f5e" if name.startswith("appearance") else "#6d7480" for name in names]
    axis.barh(range(len(rows)), values, color=colors)
    axis.set_yticks(range(len(rows)), names)
    axis.invert_yaxis()
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("Cross-shot IDF1")
    axis.set_title("V13 Phase 4A frozen appearance feature probe")
    figure.savefig(output, dpi=160)
    plt.close(figure)


def plot_risk_coverage(rows: list[dict], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    coverage = np.asarray([row["multi_activation_coverage"] for row in rows])
    risk = np.asarray([row["wrong_accept_rate"] for row in rows])
    precision = np.asarray([row["accepted_precision"] for row in rows])
    image = axis.scatter(coverage, risk, c=precision, cmap="viridis", s=14, alpha=0.8)
    axis.set_xlabel("Multi-activation coverage")
    axis.set_ylabel("Wrong-accept rate")
    axis.set_title("Precision-first risk-coverage candidates")
    figure.colorbar(image, ax=axis, label="Accepted precision")
    figure.savefig(output, dpi=160)
    plt.close(figure)


def plot_identity_distances(prepared: list[dict], output: Path) -> None:
    same, different = [], []
    for row in prepared:
        cost = np.asarray(row["base"]["cost"], dtype=np.float64)
        track_ids = np.asarray(row["base"]["bank"]["track_ids"], dtype=np.int64)
        for source_index, track_id in enumerate(track_ids):
            source_label = row["bank_labels"].get(int(track_id), -1)
            for target_index, target_label in enumerate(row["target_labels"]):
                if source_label < 0 or int(target_label) < 0:
                    continue
                bucket = same if source_label == int(target_label) else different
                bucket.append(float(cost[source_index, target_index]))
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    bins = np.linspace(0.0, max(same + different + [1.0]), 50)
    axis.hist(same, bins=bins, density=True, alpha=0.65, label="same identity")
    axis.hist(different, bins=bins, density=True, alpha=0.55, label="different identity")
    axis.set_xlabel("Primary feature distance")
    axis.set_ylabel("Density")
    axis.set_title("Shot-invariant identity distance overlap")
    axis.legend()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def plot_match_matrices(
    prepared: list[dict], config: PrecisionGateConfig, output: Path
) -> None:
    accepted_multi = []
    wrong_unfiltered = []
    for row in prepared:
        result = apply_precision_gate(row["base"], row["signals"], config)
        if len(result["accepted_pairs"]) >= 2:
            accepted_multi.append((row, result, "accepted multi"))
        if any(not signal["correct_evaluator_only"] for signal in row["signals"]):
            wrong_unfiltered.append((row, result, "unfiltered has wrong pair"))
    examples = (accepted_multi[:3] + wrong_unfiltered[:3])[:6]
    if not examples:
        return
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for axis in axes.reshape(-1):
        axis.axis("off")
    for axis, (row, result, category) in zip(axes.reshape(-1), examples):
        axis.axis("on")
        cost = np.asarray(row["base"]["cost"], dtype=np.float64)
        image = axis.imshow(cost, cmap="magma")
        accepted = {
            (int(pair["source_index"]), int(pair["target_index"]))
            for pair in result["accepted_pairs"]
        }
        for signal in row["signals"]:
            source = int(signal["source_index"])
            target = int(signal["target_index"])
            marker = "o" if (source, target) in accepted else "x"
            color = "#39d98a" if signal["correct_evaluator_only"] else "#ff5c5c"
            axis.scatter(target, source, marker=marker, s=100, facecolors="none", color=color)
        axis.set_title(f"{category}\n{row['case']['case']['key']}", fontsize=9)
        axis.set_xlabel("post detection")
        axis.set_ylabel("identity bank")
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def markdown(report: dict) -> str:
    identity = report["selected"]["identity"]
    methods = report["selected"]["methods"]

    def mean(method: str, field: str) -> float:
        return methods[method][field]["mean"]

    lines = [
        f"# V13 Phase 4A: Precision-First Identity ({report['sequence']})",
        "",
        f"Role: `{report['role']}`",
        "",
        "## Frozen appearance encoder",
        "",
        f"- encoder: `{report['appearance_encoder']['name']}`",
        f"- checkpoint SHA-256: `{report['appearance_encoder']['sha256']}`",
        "- crop source: Human3R predicted full-body bbox, no GT bbox",
        "",
        "## Selected precision gate",
        "",
        "```json",
        json.dumps(report["selected"]["config"], indent=2),
        "```",
        "",
        "## Identity risk",
        "",
        f"- accepted precision: {report['selected']['accepted_precision']:.4f}",
        f"- wrong-accept rate: {report['selected']['wrong_accept_rate']:.4f}",
        f"- accepted coverage: {report['selected']['accepted_coverage']:.4f}",
        f"- multi-activation coverage: {report['selected']['multi_activation_coverage']:.4f}",
        f"- cut all matches correct: {report['selected']['cut_all_matches_correct']:.4f}",
        f"- IDF1: {identity['idf1']:.4f}",
        f"- accepted/matchable: {identity['accepted']} / {identity['matchable']}",
        f"- wrong accepted: {identity['false_positive']}",
        "",
        "## Shared Boundary",
        "",
        "| Method | Camera T | Rotation | Composite | P90 | P95 | Catastrophic |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in (
        "single_highest_confidence",
        "gt_id_uniform",
        "automatic_unfiltered",
        "precision_first",
        "wrong_person",
        "shuffled_memory",
        "zero_memory",
        "no_identity_fixed",
    ):
        if name not in methods:
            continue
        row = methods[name]
        lines.append(
            f"| {name} | {mean(name, 'camera_translation_error_m'):.3f} | "
            f"{mean(name, 'camera_rotation_error_deg'):.2f} | "
            f"{mean(name, 'camera_composite'):.3f} | "
            f"{row['camera_composite']['p90']:.3f} | {row['camera_composite']['p95']:.3f} | "
            f"{row['catastrophic_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"GT-ID gain retention: `{report['selected']['gt_id_gain_retention']:.4f}`",
            "",
            "## GT boundary",
            "",
            "Appearance crops, feature extraction, matching, gate, dustbin and Boundary candidate "
            "do not read GT identity, GT bbox, GT camera, GT body, source ID or camera-pair ID. "
            "GT is attached only for development selection and final evaluation.",
            "",
        ]
    )
    active = report["selected"]["activated_multi"]
    lines.extend(
        [
            "## Activated multi cuts only",
            "",
            f"Cases: `{active['case_count']}`",
            "",
            "| Method | Composite | P90 | Catastrophic |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in ("single_highest_confidence", "gt_id_uniform", "precision_first"):
        row = active["methods"][name]
        lines.append(
            f"| {name} | {row['camera_composite']['mean']:.3f} | "
            f"{row['camera_composite']['p90']:.3f} | {row['catastrophic_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Activated-cut GT-ID gain retention: `{active['gt_id_gain_retention']:.4f}`",
            "",
        ]
    )
    lines.extend(
        [
            "## Temporal offset groups",
            "",
            "| Offset | Cuts | Accepted | Wrong | Precision | Multi coverage |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for offset, row in report["selected"]["by_offset"].items():
        precision = row["accepted_precision"]
        lines.append(
            f"| {offset} | {row['case_count']} | {row['identity']['accepted']} | "
            f"{row['identity']['false_positive']} | "
            f"{precision:.3f} | {row['multi_activation_coverage']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def analyze(args: argparse.Namespace, case_specs: list[geometry.CaseSpec]) -> dict:
    cases = load_cases(args, case_specs)
    probe_path = args.output_dir / "feature_probe.json"
    if str(args.role) == "development":
        if args.reuse_feature_probe and probe_path.is_file():
            ranked = json.loads(probe_path.read_text(encoding="utf-8"))
        else:
            ranked = feature_probe(cases)
            probe_path.write_text(
                json.dumps(jsonable(ranked), indent=2, allow_nan=True) + "\n",
                encoding="utf-8",
            )
        primary = select_primary(ranked)
        prepared = prepare_identity_cases(cases, primary)
        selected_identity, gate_rows = select_gate(prepared, primary)
        frozen = {
            "schema_version": 1,
            "selected_on": "MultiHuman three only",
            "selection_scope": "identity precision and coverage; no camera or Boundary metric",
            "appearance_encoder": {
                "name": ENCODER_NAME,
                "dimension": ENCODER_DIM,
                "input_size": ENCODER_INPUT_SIZE,
                "sha256": ENCODER_CHECKPOINT_SHA256,
                "bbox_source": "Human3R predicted full-body bbox",
                "bbox_padding": float(args.bbox_padding),
            },
            "precision_gate": selected_identity["config"],
            "gate_objective": (
                "zero wrong accepted first, then maximum multi activation coverage, "
                "accepted coverage and IDF1"
            ),
            "geometry": "frozen V13 Phase 2 uniform consensus",
        }
        args.frozen_config.write_text(
            json.dumps(frozen, indent=2, allow_nan=True) + "\n", encoding="utf-8"
        )
    else:
        if not args.frozen_config.is_file():
            raise FileNotFoundError(args.frozen_config)
        frozen = json.loads(args.frozen_config.read_text(encoding="utf-8"))
        primary = PrecisionGateConfig(**frozen["precision_gate"])
        if args.reuse_feature_probe and probe_path.is_file():
            ranked = json.loads(probe_path.read_text(encoding="utf-8"))
        else:
            ranked = feature_probe(cases)
            probe_path.write_text(
                json.dumps(jsonable(ranked), indent=2, allow_nan=True) + "\n",
                encoding="utf-8",
            )
        prepared = prepare_identity_cases(cases, primary)
        selected_identity = gate_identity_summary(prepared, primary)
        gate_rows = [selected_identity]

    build_endpoint_lattice(args, prepared)
    selected = attach_endpoint(args, selected_identity, prepared)
    if str(args.role) == "development":
        frontier = risk_frontier(gate_rows)
        risk_endpoint = [attach_endpoint(args, row, prepared) for row in frontier]
    else:
        risk_endpoint = [selected]
    report = {
        "experiment": "V13 Phase 4A Precision-First Frozen Appearance Feasibility",
        "sequence": str(args.sequence),
        "role": str(args.role),
        "case_count": len(cases),
        "protocol": {
            "timestamps": args.timestamps,
            "camera_pairs": args.camera_pairs,
            "offsets": args.offsets,
            "history_frames": args.history_frames,
            "full_frame_no_crop_before_predicted_bbox": True,
        },
        "appearance_encoder": frozen["appearance_encoder"],
        "candidate_gt_usage": {
            "appearance_crop": False,
            "feature": False,
            "matching": False,
            "gate": False,
            "boundary_candidate": False,
            "development_selection": str(args.role) == "development",
            "evaluation": True,
        },
        "geometry_contract": {
            "phase2_uniform_consensus_frozen": True,
            "shot_scale": 1.0,
            "da3": False,
            "vggt": False,
            "continuity": False,
            "identity_predicts_se3": False,
        },
        "feature_probe_ranked": ranked,
        "selected": selected,
        "risk_coverage": [
            {key: value for key, value in row.items() if key != "masks"}
            for row in risk_endpoint
        ],
    }
    output_json = args.output_dir / "v13_phase4_precision_identity.json"
    output_md = args.output_dir / "v13_phase4_precision_identity.md"
    output_json.write_text(
        json.dumps(jsonable(report), indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    output_md.write_text(markdown(report), encoding="utf-8")
    plot_feature_probe(ranked, args.output_dir / "feature_probe.png")
    plot_risk_coverage(gate_rows, args.output_dir / "risk_coverage.png")
    plot_identity_distances(prepared, args.output_dir / "same_different_distance.png")
    plot_match_matrices(
        prepared, primary, args.output_dir / "precision_match_matrices.png"
    )
    print(f">> Phase 4 JSON: {output_json}", flush=True)
    print(f">> Phase 4 report: {output_md}", flush=True)
    return report


def main() -> None:
    args = configure(parse_args())
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    case_specs = specs(args)
    if args.mode in {"extract", "all"}:
        extract(args, case_specs)
    if args.mode in {"analyze", "all"}:
        analyze(args, case_specs)


if __name__ == "__main__":
    main()
