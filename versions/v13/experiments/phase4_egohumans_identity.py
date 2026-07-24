#!/usr/bin/env python3
"""Frozen V13 Phase-4 appearance identity audit on EgoHumans multi-cut streams.

Predicted person boxes are reconstructed from Human3R's compact SMPL-X output.
GT boxes and identities never enter appearance extraction, matching, or memory.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from demo import prepare_input  # noqa: E402
from dust3r.utils.smpl_layer import SMPL_Layer  # noqa: E402
from versions.v13 import gt_id_consensus as geometry  # noqa: E402
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
)
from versions.v13.experiments.phase3_cross_shot_identity import (  # noqa: E402
    aggregate_identity,
    feature_frame,
)
from versions.v13.identity_bridge import (  # noqa: E402
    CausalIdentityMemory,
    evaluate_assignment,
)
from versions.v13.native_token_probe import jsonable, tensor_numpy  # noqa: E402


DEFAULT_PROBES = (
    "egobody",
    "egobody_cam02_cam05_cam08",
    "egobody_cam03_cam04_cam01",
)
IDENTITIES = ("aria01", "aria02", "aria03")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("/data/wangzheng/iJCV-CODE/data/EgoBody/001_legoassemble"),
    )
    parser.add_argument("--probe_root", type=Path, default=ROOT / "output/v13")
    parser.add_argument("--probe_names", nargs="+", default=DEFAULT_PROBES)
    parser.add_argument(
        "--frozen_config",
        type=Path,
        default=ROOT / "versions/v13/configs/phase4_precision_config.json",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ROOT / "output/v13/phase4_identity/egohumans_001_legoassemble",
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
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--bbox_padding", type=float, default=0.08)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--mhmr_img_res", type=int, default=896)
    parser.add_argument("--track_ttl", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def native_ids(debug: dict, count: int) -> np.ndarray:
    return (
        np.full(count, -1, dtype=np.int64)
        if debug.get("smpl_ids") is None
        else tensor_numpy(debug["smpl_ids"])[0].astype(np.int64)
    )


def stream_spec(source_report: dict) -> tuple[list[str], list[int], list[Path], set[int]]:
    cameras, frames = [], []
    root = Path(source_report["dataset"]["root"])
    for segment in source_report["stream"]["segments"]:
        camera = str(segment["camera"])
        for frame in segment["frames"]:
            cameras.append(camera)
            frames.append(int(frame))
    paths = [
        root / f"exo/{camera}/images/{frame:05d}.jpg"
        for camera, frame in zip(cameras, frames)
    ]
    return cameras, frames, paths, {int(value) for value in source_report["stream"]["cuts"]}


def predicted_boxes(
    predictions: list[dict],
    debug: list[dict],
    image_paths: list[Path],
    size: int,
    mhmr_img_res: int,
    layer: SMPL_Layer,
) -> tuple[list[list[dict]], list[tuple[int, int]]]:
    views = prepare_input(
        img_paths=[str(path) for path in image_paths],
        img_mask=[True] * len(image_paths),
        size=int(size),
        revisit=1,
        update=True,
        img_res=int(mhmr_img_res),
        reset_interval=10_000_000,
    )
    humans, shapes = [], []
    for prediction, view, debug_row in zip(predictions, views, debug):
        rows = geometry.layer_humans(prediction, view, debug_row, layer)
        rows.sort(key=lambda row: int(row["detection_index"]))
        count = int(prediction["smpl_transl"].shape[1])
        if [int(row["detection_index"]) for row in rows] != list(range(count)):
            raise ValueError("Predicted SMPL-X bbox order differs from detection order")
        humans.append(rows)
        shapes.append(tuple(int(value) for value in tensor_numpy(view["true_shape"])[0]))
    return humans, shapes


def build_appearance_cache(
    args: argparse.Namespace,
    probe_name: str,
    compact: dict,
    source_report: dict,
    encoder: FrozenDinoAppearance,
    layer: SMPL_Layer,
) -> dict:
    cache_path = args.output_dir / f"{probe_name}_appearance.pt"
    if cache_path.is_file() and not args.overwrite:
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    predictions = compact["predictions"]
    debug = compact["token_debug"]
    cameras, frames, image_paths, cuts = stream_spec(source_report)
    humans, processed_shapes = predicted_boxes(
        predictions,
        debug,
        image_paths,
        int(args.size),
        int(args.mhmr_img_res),
        layer,
    )
    crop_payloads = []
    all_crops = []
    for image_path, frame_humans, processed_shape in zip(
        image_paths, humans, processed_shapes
    ):
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(image_path)
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        rows, indices = [], []
        for detection_index, human in enumerate(frame_humans):
            crop, row = crop_predicted_bbox(
                image,
                human["bbox"],
                processed_shape,
                padding_ratio=float(args.bbox_padding),
            )
            row["detection_index"] = detection_index
            rows.append(row)
            if crop is not None:
                indices.append((len(all_crops), detection_index))
                all_crops.append(crop)
        crop_payloads.append({"rows": rows, "indices": indices, "count": len(frame_humans)})

    encoded = encoder.encode(all_crops)
    appearance_frames = []
    for payload, image_path, processed_shape in zip(
        crop_payloads, image_paths, processed_shapes
    ):
        count = int(payload["count"])
        embedding = np.zeros((count, ENCODER_DIM), dtype=np.float32)
        valid = np.zeros(count, dtype=bool)
        for crop_index, detection_index in payload["indices"]:
            embedding[int(detection_index)] = encoded[int(crop_index)]
            valid[int(detection_index)] = True
        appearance_frames.append(
            {
                "appearance": embedding,
                "valid": valid,
                "crops": payload["rows"],
                "image_path": str(image_path),
                "processed_shape": processed_shape,
                "candidate_gt_usage": False,
            }
        )
    cache = {
        "probe_name": probe_name,
        "cameras": cameras,
        "frames": frames,
        "cuts": sorted(cuts),
        "appearance_frames": appearance_frames,
        "encoder": {
            "name": ENCODER_NAME,
            "dimension": ENCODER_DIM,
            "input_size": ENCODER_INPUT_SIZE,
            "sha256": ENCODER_CHECKPOINT_SHA256,
            "bbox_source": "Human3R predicted SMPL-X projection",
            "bbox_padding": float(args.bbox_padding),
        },
        "candidate_gt_usage": False,
    }
    torch.save(cache, cache_path)
    return cache


def majority(votes: dict[int, list[int]]) -> dict[int, int]:
    output = {}
    for track_id, labels in votes.items():
        valid = np.asarray([value for value in labels if value >= 0], dtype=np.int64)
        if not len(valid):
            continue
        values, counts = np.unique(valid, return_counts=True)
        output[int(track_id)] = int(values[int(np.argmax(counts))])
    return output


def external_by_gt(track_ids: np.ndarray, labels: np.ndarray) -> dict[int, int]:
    return {
        int(label): int(track_id)
        for track_id, label in zip(track_ids, labels)
        if int(track_id) >= 0 and int(label) >= 0
    }


def gate_failures(signals: list[dict], config: PrecisionGateConfig) -> dict[str, int]:
    checks = {
        "mutual": lambda row: row["mutual_nearest"] or not config.require_mutual,
        "target_appearance_valid": lambda row: row["target_appearance_valid"],
        "valid_observations": lambda row: (
            row["valid_observations"] >= int(config.min_valid_observations)
        ),
        "primary_distance": lambda row: (
            row["primary_distance"] <= float(config.max_primary_distance)
        ),
        "primary_margin": lambda row: (
            row["primary_margin"] >= float(config.min_primary_margin)
        ),
        "vote_fraction": lambda row: (
            row["vote_fraction"] >= float(config.min_vote_fraction)
        ),
        "beta_distance": lambda row: (
            row["beta_distance"] <= float(config.max_beta_distance)
        ),
        "pose_distance": lambda row: (
            row["pose_distance"] <= float(config.max_pose_distance)
        ),
    }
    return {
        name: int(sum(not predicate(row) for row in signals))
        for name, predicate in checks.items()
    }


def evaluate_stream(
    compact: dict,
    appearance_cache: dict,
    config: PrecisionGateConfig,
    ttl: int,
) -> dict:
    labels = [np.asarray(value, dtype=np.int64) for value in compact["labels"]]
    native_frames = [
        feature_frame(prediction, debug)
        for prediction, debug in zip(compact["predictions"], compact["token_debug"])
    ]
    frames = [
        augment_frame(frame, appearance["appearance"], appearance["valid"])
        for frame, appearance in zip(native_frames, appearance_cache["appearance_frames"])
    ]
    cuts = {int(value) for value in appearance_cache["cuts"]}
    memory = CausalIdentityMemory(ttl=int(ttl), prototype_window=5)
    identity_votes: dict[int, list[int]] = defaultdict(list)
    native_to_external: dict[int, int] = {}
    cut_rows, timeline = [], []
    initial_external_by_gt: dict[int, int] = {}

    for frame_index, (frame, frame_labels, debug_row) in enumerate(
        zip(frames, labels, compact["token_debug"])
    ):
        current_native = native_ids(debug_row, int(frame["count"]))
        if frame_index == 0:
            external = memory.bootstrap(frame, timestamp=frame_index, use_native_ids=True)
            initial_external_by_gt = external_by_gt(external, frame_labels)
            native_to_external = {
                int(native): int(track)
                for native, track in zip(current_native, external)
                if int(native) >= 0
            }
        elif frame_index in cuts:
            prototype_frames = memory.prototype_frames(timestamp=frame_index)
            base, signals = precision_signals(prototype_frames, frame, config)
            result = apply_precision_gate(base, signals, config)
            bank_labels = majority(identity_votes)
            metrics = evaluate_assignment(result, bank_labels, frame_labels)
            before = memory.snapshot()
            external = memory.commit(frame, result, timestamp=frame_index)
            after = memory.snapshot()
            native_to_external = {
                int(native): int(track)
                for native, track in zip(current_native, external)
                if int(native) >= 0
            }
            current_external_by_gt = external_by_gt(external, frame_labels)
            visible_recovered = {
                IDENTITIES[label]: bool(
                    label in initial_external_by_gt
                    and current_external_by_gt.get(label) == initial_external_by_gt[label]
                )
                for label in current_external_by_gt
            }
            previous_labels = {
                int(value) for value in labels[frame_index - 1] if int(value) >= 0
            }
            current_labels = {
                int(value) for value in frame_labels if int(value) >= 0
            }
            reappeared_labels = sorted(current_labels - previous_labels)
            reappearance_recovered = {
                IDENTITIES[label]: bool(
                    label in initial_external_by_gt
                    and current_external_by_gt.get(label) == initial_external_by_gt[label]
                )
                for label in reappeared_labels
            }
            cut_rows.append(
                {
                    "frame_index": frame_index,
                    "camera": appearance_cache["cameras"][frame_index],
                    "frame": appearance_cache["frames"][frame_index],
                    "pre_human_count": int(frames[frame_index - 1]["count"]),
                    "post_human_count": int(frame["count"]),
                    "identity": metrics,
                    "accepted_precision": (
                        metrics["true_positive"] / max(metrics["accepted"], 1)
                    ),
                    "wrong_accept_count": int(metrics["false_positive"]),
                    "activation": (
                        "multi" if metrics["accepted"] >= 2 else
                        "single" if metrics["accepted"] == 1 else "fixed"
                    ),
                    "signals": signals,
                    "gate_failure_counts": gate_failures(signals, config),
                    "accepted_pairs": result["accepted_pairs"],
                    "unmatched_source": result["unmatched_source"],
                    "unmatched_target": result["unmatched_target"],
                    "memory_before_commit": before,
                    "memory_after_commit": after,
                    "initial_external_by_gt_evaluator_only": initial_external_by_gt,
                    "current_external_by_gt_evaluator_only": current_external_by_gt,
                    "visible_initial_track_recovered_evaluator_only": visible_recovered,
                    "reappeared_identities_evaluator_only": [
                        IDENTITIES[label] for label in reappeared_labels
                    ],
                    "reappearance_recovered_evaluator_only": reappearance_recovered,
                }
            )
        else:
            external_values = []
            for native in current_native:
                native = int(native)
                if native not in native_to_external:
                    native_to_external[native] = memory.next_track_id
                    memory.next_track_id += 1
                external_values.append(native_to_external[native])
            external = np.asarray(external_values, dtype=np.int64)
            memory.observe(frame, external, timestamp=frame_index)

        for track_id, label in zip(external, frame_labels):
            if int(label) >= 0:
                identity_votes[int(track_id)].append(int(label))
        timeline.append(
            {
                "frame_index": frame_index,
                "camera": appearance_cache["cameras"][frame_index],
                "frame": appearance_cache["frames"][frame_index],
                "native_ids": current_native,
                "external_ids": external,
                "gt_labels_evaluator_only": frame_labels,
                "memory": memory.snapshot(),
            }
        )

    aggregate = aggregate_identity([row["identity"] for row in cut_rows])
    accepted = int(aggregate["accepted"])
    wrong = int(aggregate["false_positive"])
    reappearance = [
        recovered
        for row in cut_rows
        for recovered in row["reappearance_recovered_evaluator_only"].values()
    ]
    appearance_valid = [
        bool(value)
        for row in appearance_cache["appearance_frames"]
        for value in row["valid"]
    ]
    return {
        "identity": aggregate,
        "accepted_precision": (
            (accepted - wrong) / accepted if accepted else float("nan")
        ),
        "wrong_accept_rate": wrong / max(accepted, 1),
        "multi_activation_coverage": float(
            np.mean([row["activation"] == "multi" for row in cut_rows])
        ),
        "fallback_counts": {
            name: int(sum(row["activation"] == name for row in cut_rows))
            for name in ("multi", "single", "fixed")
        },
        "appearance_crop_valid": int(sum(appearance_valid)),
        "appearance_crop_total": len(appearance_valid),
        "gate_failure_counts": {
            name: int(sum(row["gate_failure_counts"][name] for row in cut_rows))
            for name in cut_rows[0]["gate_failure_counts"]
        },
        "inactive_reappearance_recovered": int(sum(reappearance)),
        "inactive_reappearance_total": len(reappearance),
        "cuts": cut_rows,
        "timeline": timeline,
    }


def plot_cut_matrices(streams: list[dict], output: Path) -> None:
    rows = [(stream["probe_name"], cut) for stream in streams for cut in stream["cuts"]]
    figure, axes = plt.subplots(
        1,
        max(len(rows), 1),
        figsize=(4.5 * max(len(rows), 1), 4),
        constrained_layout=True,
        squeeze=False,
    )
    for axis, (probe_name, cut) in zip(axes.reshape(-1), rows):
        signals = cut["signals"]
        source_count = max((int(row["source_index"]) for row in signals), default=-1) + 1
        target_count = max((int(row["target_index"]) for row in signals), default=-1) + 1
        matrix = np.full((source_count, target_count), np.nan, dtype=np.float64)
        for signal in signals:
            matrix[int(signal["source_index"]), int(signal["target_index"])] = float(
                signal["primary_distance"]
            )
        image = axis.imshow(matrix, cmap="magma") if matrix.size else None
        axis.set_title(f"{probe_name}\ncut {cut['frame_index']} {cut['activation']}")
        axis.set_xlabel("post detection")
        axis.set_ylabel("external track")
        if image is not None:
            figure.colorbar(image, ax=axis, fraction=0.046)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def plot_predicted_bbox_overlays(streams: list[dict], output: Path) -> None:
    rows = []
    for stream in streams:
        cut_indices = {int(cut["frame_index"]) for cut in stream["cuts"]}
        selected = sorted(cut_indices | {max(index - 1, 0) for index in cut_indices})
        for frame_index in selected:
            rows.append((stream["probe_name"], stream["appearance_frames"][frame_index]))
    columns = 4
    figure_rows = max(1, int(np.ceil(len(rows) / columns)))
    figure, axes = plt.subplots(
        figure_rows,
        columns,
        figsize=(5 * columns, 3.2 * figure_rows),
        constrained_layout=True,
        squeeze=False,
    )
    colors = ((40, 220, 130), (255, 190, 60), (255, 90, 90), (90, 170, 255))
    for axis in axes.reshape(-1):
        axis.axis("off")
    for axis, (probe_name, frame) in zip(axes.reshape(-1), rows):
        image = cv2.imread(frame["image_path"], cv2.IMREAD_COLOR)
        if image is None:
            continue
        original_height, original_width = image.shape[:2]
        image = cv2.resize(image, (512, 288), interpolation=cv2.INTER_AREA)
        for detection_index, crop in enumerate(frame["crops"]):
            if not crop.get("valid") or "bbox_rgb" not in crop:
                continue
            box = np.asarray(crop["bbox_rgb"], dtype=np.float64)
            box[[0, 2]] *= 512.0 / original_width
            box[[1, 3]] *= 288.0 / original_height
            x0, y0, x1, y1 = np.round(box).astype(int)
            color = colors[detection_index % len(colors)]
            cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
            cv2.putText(
                image,
                f"det{detection_index}",
                (x0, max(14, y0 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axis.set_title(f"{probe_name}\n{Path(frame['image_path']).parent.parent.name}/{Path(frame['image_path']).stem}")
        axis.axis("off")
    figure.savefig(output, dpi=150)
    plt.close(figure)


def markdown(report: dict) -> str:
    aggregate = report["aggregate"]
    accepted_precision = (
        f"{aggregate['accepted_precision']:.4f}"
        if np.isfinite(aggregate["accepted_precision"])
        else "undefined (no accepted matches)"
    )
    lines = [
        "# V13 Phase 4A: EgoHumans Precision-First Multi-Cut Audit",
        "",
        "Frozen on `MultiHuman three`; no EgoHumans threshold tuning.",
        "",
        "## Aggregate",
        "",
        f"- accepted precision: {accepted_precision}",
        f"- wrong-accept rate: {aggregate['wrong_accept_rate']:.4f}",
        f"- accepted/matchable: {aggregate['identity']['accepted']} / {aggregate['identity']['matchable']}",
        f"- wrong accepted: {aggregate['identity']['false_positive']}",
        f"- multi-activation coverage: {aggregate['multi_activation_coverage']:.4f}",
        f"- fallback counts: {aggregate['fallback_counts']}",
        f"- appearance crop valid: {aggregate['appearance_crop_valid']} / {aggregate['appearance_crop_total']}",
        f"- inactive reappearance recovered: {aggregate['inactive_reappearance_recovered']} / {aggregate['inactive_reappearance_total']}",
        f"- gate failure counts: {aggregate['gate_failure_counts']}",
        "",
        "## Streams",
        "",
        "| Stream | Counts | Accepted | Wrong | Precision | Multi cuts | Fallback |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for stream in report["streams"]:
        cuts = " -> ".join(
            f"{row['pre_human_count']}:{row['post_human_count']}" for row in stream["cuts"]
        )
        precision = (
            f"{stream['accepted_precision']:.3f}"
            if np.isfinite(stream["accepted_precision"])
            else "N/A"
        )
        lines.append(
            f"| {stream['probe_name']} | {cuts} | {stream['identity']['accepted']} | "
            f"{stream['identity']['false_positive']} | {precision} | "
            f"{stream['fallback_counts']['multi']} | {stream['fallback_counts']} |"
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "This is a cross-data WHO, inactive-track and fallback audit. The compact "
            "EgoHumans caches do not provide the frozen Phase-2 MultiHuman geometry evaluator, "
            "so no camera or Boundary metric is claimed here.",
            "",
            "Appearance crops use Human3R-predicted SMPL-X projection only. GT identity is "
            "attached after matching for evaluation; GT bbox is never loaded.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.frozen_config.is_file():
        raise FileNotFoundError(args.frozen_config)
    frozen = json.loads(args.frozen_config.read_text(encoding="utf-8"))
    config = PrecisionGateConfig(**frozen["precision_gate"])
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    layer = SMPL_Layer(
        type="smplx",
        gender="neutral",
        num_betas=10,
        kid=False,
        person_center="head",
    ).to(device).eval()
    encoder = FrozenDinoAppearance(
        str(args.device),
        args.encoder_hub_dir,
        args.encoder_checkpoint,
        batch_size=int(args.batch_size),
    )

    streams = []
    for probe_name in args.probe_names:
        probe_dir = args.probe_root / probe_name
        compact_path = probe_dir / "v13_egobody_compact_tokens.pt"
        report_path = probe_dir / "v13_egobody_three_person_probe.json"
        if not compact_path.is_file() or not report_path.is_file():
            raise FileNotFoundError(f"Missing EgoHumans probe cache: {probe_dir}")
        compact = torch.load(compact_path, map_location="cpu", weights_only=False)
        source_report = json.loads(report_path.read_text(encoding="utf-8"))
        appearance = build_appearance_cache(
            args, probe_name, compact, source_report, encoder, layer
        )
        result = evaluate_stream(compact, appearance, config, int(args.track_ttl))
        streams.append(
            {
                "probe_name": probe_name,
                "appearance_frames": appearance["appearance_frames"],
                **result,
            }
        )

    identity = aggregate_identity([stream["identity"] for stream in streams])
    accepted = int(identity["accepted"])
    wrong = int(identity["false_positive"])
    all_cuts = [cut for stream in streams for cut in stream["cuts"]]
    reappearance = [
        recovered
        for cut in all_cuts
        for recovered in cut["reappearance_recovered_evaluator_only"].values()
    ]
    aggregate = {
        "identity": identity,
        "accepted_precision": (
            (accepted - wrong) / accepted if accepted else float("nan")
        ),
        "wrong_accept_rate": wrong / max(accepted, 1),
        "multi_activation_coverage": float(
            np.mean([cut["activation"] == "multi" for cut in all_cuts])
        ),
        "fallback_counts": {
            name: int(sum(cut["activation"] == name for cut in all_cuts))
            for name in ("multi", "single", "fixed")
        },
        "appearance_crop_valid": int(sum(stream["appearance_crop_valid"] for stream in streams)),
        "appearance_crop_total": int(sum(stream["appearance_crop_total"] for stream in streams)),
        "gate_failure_counts": {
            name: int(sum(stream["gate_failure_counts"][name] for stream in streams))
            for name in streams[0]["gate_failure_counts"]
        },
        "inactive_reappearance_recovered": int(sum(reappearance)),
        "inactive_reappearance_total": len(reappearance),
    }
    serializable_streams = [
        {key: value for key, value in stream.items() if key != "appearance_frames"}
        for stream in streams
    ]
    report = {
        "experiment": "V13 Phase 4A EgoHumans precision-first multi-cut identity audit",
        "dataset": "EgoHumans 001_legoassemble",
        "frozen_config": frozen,
        "track_ttl": int(args.track_ttl),
        "appearance_encoder": streams and appearance["encoder"],
        "candidate_gt_usage": {
            "predicted_bbox": False,
            "appearance": False,
            "matching": False,
            "memory": False,
            "evaluation": True,
        },
        "scope": "WHO, inactive tracklet, reappearance and fallback; no final Boundary metric",
        "streams": serializable_streams,
        "aggregate": aggregate,
    }
    output_json = args.output_dir / "v13_phase4_egohumans_identity.json"
    output_md = args.output_dir / "v13_phase4_egohumans_identity.md"
    output_json.write_text(
        json.dumps(jsonable(report), indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    output_md.write_text(markdown(report), encoding="utf-8")
    plot_cut_matrices(streams, args.output_dir / "egohumans_precision_matrices.png")
    plot_predicted_bbox_overlays(
        streams, args.output_dir / "egohumans_predicted_bbox_overlay.png"
    )
    print(f">> EgoHumans Phase 4 JSON: {output_json}", flush=True)
    print(f">> EgoHumans Phase 4 report: {output_md}", flush=True)


if __name__ == "__main__":
    main()
