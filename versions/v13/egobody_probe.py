#!/usr/bin/env python3
"""Legacy-named V20 EgoHumans probe retained inside Movie3R-Multi V13."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from versions.v12.experiments.v14_1_shot_aware_state_routing_probe import build_model, camera_matrix  # noqa: E402
from versions.v13.native_token_probe import (  # noqa: E402
    FEATURES,
    feature_cost,
    jsonable,
    person_feature,
    processed_head_locations,
    rotation_error_deg,
    run_stream,
    tensor_numpy,
)


IDENTITIES = ("aria01", "aria02", "aria03")
DEFAULT_SEGMENTS = (
    ("cam01", tuple(range(296, 301))),
    ("cam06", tuple(range(300, 305))),
    ("cam07", tuple(range(304, 309))),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("/data/wangzheng/iJCV-CODE/data/EgoBody/001_legoassemble"),
    )
    parser.add_argument("--model_path", type=Path, default=ROOT / "src/human3r_896L.pth")
    parser.add_argument(
        "--output_dir", type=Path, default=ROOT / "output/v13/egobody"
    )
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--assignment_threshold_px", type=float, default=24.0)
    parser.add_argument(
        "--segments",
        nargs="+",
        metavar="CAM:START-END",
        help=(
            "Camera segments in stream order, for example "
            "cam01:296-300 cam06:300-304 cam07:304-308. "
            "A hard reset is inserted before every segment after the first."
        ),
    )
    return parser.parse_args()


def parse_segments(values: list[str] | None) -> tuple[tuple[str, tuple[int, ...]], ...]:
    if not values:
        return DEFAULT_SEGMENTS
    output = []
    for value in values:
        try:
            camera, interval = value.split(":", maxsplit=1)
            start_text, end_text = interval.split("-", maxsplit=1)
            start, end = int(start_text), int(end_text)
        except ValueError as error:
            raise ValueError(
                f"Invalid segment {value!r}; expected CAM:START-END"
            ) from error
        if not camera.startswith("cam") or start < 1 or end < start:
            raise ValueError(f"Invalid segment {value!r}")
        output.append((camera, tuple(range(start, end + 1))))
    if len(output) < 2:
        raise ValueError("At least two camera segments are required")
    return tuple(output)


def segment_cuts(segments: tuple[tuple[str, tuple[int, ...]], ...]) -> tuple[int, ...]:
    lengths = [len(frames) for _, frames in segments]
    if any(length == 0 for length in lengths):
        raise ValueError("Every camera segment must contain at least one frame")
    return tuple(np.cumsum(lengths[:-1]).astype(int).tolist())


def load_colmap(data_root: Path) -> tuple[dict, dict]:
    cameras = {}
    camera_path = data_root / "colmap/workplace/cameras.txt"
    for line in camera_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields and fields[0].isdigit():
            cameras[int(fields[0])] = {
                "model": fields[1],
                "width": int(fields[2]),
                "height": int(fields[3]),
                "params": np.asarray(fields[4:], dtype=np.float64),
            }

    records = defaultdict(list)
    with (data_root / "colmap/workplace/images.txt").open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) != 10 or not fields[0].isdigit() or "/" not in fields[-1]:
                continue
            camera_name = fields[-1].split("/")[0]
            if not camera_name.startswith("cam"):
                continue
            quaternion = np.asarray(fields[1:5], dtype=np.float64)
            world_to_camera = Rotation.from_quat(
                [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]
            ).as_matrix()
            translation = np.asarray(fields[5:8], dtype=np.float64)
            records[camera_name].append(
                {
                    "rotation": world_to_camera,
                    "translation": translation,
                    "camera_id": int(fields[8]),
                }
            )

    colmap_from_aria = pickle.load(
        (data_root / "colmap/workplace/colmap_from_aria_transforms.pkl").open("rb")
    )["aria01"]
    linear = np.asarray(colmap_from_aria[:3, :3], dtype=np.float64)
    similarity_scale = float(np.cbrt(np.linalg.det(linear)))
    aria_to_colmap_rotation = linear / similarity_scale
    aria_to_colmap_translation = np.asarray(colmap_from_aria[:3, 3], dtype=np.float64)

    exo = {}
    for camera_name, rows in records.items():
        rotations = np.stack([row["rotation"] for row in rows])
        world_to_camera = Rotation.from_matrix(rotations).mean().as_matrix()
        centers = np.stack(
            [-row["rotation"].T @ row["translation"] for row in rows]
        )
        center_colmap = centers.mean(axis=0)
        camera_to_colmap = world_to_camera.T
        center_aria = np.linalg.solve(
            linear, center_colmap - aria_to_colmap_translation
        )
        camera_to_aria = aria_to_colmap_rotation.T @ camera_to_colmap
        camera_pose = np.eye(4, dtype=np.float64)
        camera_pose[:3, :3] = camera_to_aria
        camera_pose[:3, 3] = center_aria
        exo[camera_name] = {
            "camera_id": rows[0]["camera_id"],
            "c2w_aria01": camera_pose,
            "w2c_colmap_rotation": world_to_camera,
            "w2c_colmap_translation": -world_to_camera @ center_colmap,
            "center_spread_max_m_colmap": float(
                np.linalg.norm(centers - center_colmap, axis=1).max()
            ),
            "pose_record_count": len(rows),
        }
    return {
        "cameras": cameras,
        "colmap_from_aria01": np.asarray(colmap_from_aria, dtype=np.float64),
        "similarity_scale": similarity_scale,
    }, exo


def fisheye_project(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray, params: np.ndarray):
    camera = points @ rotation.T + translation
    normalized = camera[:, :2] / camera[:, 2:3]
    radius = np.linalg.norm(normalized, axis=1)
    theta = np.arctan(radius)
    theta2 = theta * theta
    k1, k2, k3, k4 = params[4:8]
    distorted_theta = theta * (
        1.0 + k1 * theta2 + k2 * theta2**2 + k3 * theta2**3 + k4 * theta2**4
    )
    radial = np.divide(
        distorted_theta,
        radius,
        out=np.ones_like(distorted_theta),
        where=radius > 1e-12,
    )
    pixels = normalized * radial[:, None]
    pixels[:, 0] = params[0] * pixels[:, 0] + params[2]
    pixels[:, 1] = params[1] * pixels[:, 1] + params[3]
    return pixels, camera[:, 2]


def bbox_iou(first: np.ndarray, second: np.ndarray) -> float:
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = width * height
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return float(intersection / max(first_area + second_area - intersection, 1e-9))


def projection_audit(data_root: Path, colmap: dict, exo: dict, frame: int = 300) -> dict:
    bodies = np.load(
        data_root / f"processed_data/smpl/{frame:05d}.npy", allow_pickle=True
    ).item()
    transform = colmap["colmap_from_aria01"]
    output = {}
    for camera_name, camera in sorted(exo.items()):
        annotations = {
            row["human_name"]: np.asarray(row["bbox"][:4], dtype=np.float64)
            for row in np.load(
                data_root / f"processed_data/bboxes/{camera_name}/rgb/{frame:05d}.npy",
                allow_pickle=True,
            )
        }
        params = colmap["cameras"][camera["camera_id"]]["params"]
        per_identity = {}
        for identity in IDENTITIES:
            vertices = np.asarray(bodies[identity]["vertices"], dtype=np.float64)
            vertices_colmap = vertices @ transform[:3, :3].T + transform[:3, 3]
            pixels, depth = fisheye_project(
                vertices_colmap,
                camera["w2c_colmap_rotation"],
                camera["w2c_colmap_translation"],
                params,
            )
            valid = depth > 0
            projected_bbox = np.r_[pixels[valid].min(axis=0), pixels[valid].max(axis=0)]
            per_identity[identity] = {
                "bbox_iou": bbox_iou(projected_bbox, annotations[identity]),
                "projected_bbox": projected_bbox,
                "annotated_bbox": annotations[identity],
            }
        output[camera_name] = {
            "mean_bbox_iou": float(
                np.mean([row["bbox_iou"] for row in per_identity.values()])
            ),
            "identities": per_identity,
        }
    return output


def processed_point(point: np.ndarray, original_shape: tuple[int, int], target_shape: tuple[int, int], size: int):
    original_height, original_width = original_shape
    target_height, target_width = target_shape
    scale = float(size) / float(max(original_height, original_width))
    resized_width = int(round(original_width * scale))
    resized_height = int(round(original_height * scale))
    offset = np.asarray(
        [(resized_width - target_width) // 2, (resized_height - target_height) // 2],
        dtype=np.float64,
    )
    return np.asarray(point, dtype=np.float64) * scale - offset


def gt_pelvis(row: dict) -> tuple[np.ndarray, float]:
    keypoints = np.asarray(row["keypoints"], dtype=np.float64)
    hips = keypoints[[11, 12]]
    valid = hips[:, 2] > 0.05
    if valid.any():
        weights = hips[valid, 2]
        point = np.average(hips[valid, :2], axis=0, weights=weights)
        confidence = float(weights.mean())
    else:
        bbox = np.asarray(row["bbox"], dtype=np.float64)
        point = np.asarray([(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5])
        confidence = 0.0
    return point, confidence


def points_to_bboxes_distance(points: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
    output = np.zeros((len(points), len(bboxes)), dtype=np.float64)
    for column, bbox in enumerate(bboxes):
        dx = np.maximum(np.maximum(bbox[0] - points[:, 0], 0.0), points[:, 0] - bbox[2])
        dy = np.maximum(np.maximum(bbox[1] - points[:, 1], 0.0), points[:, 1] - bbox[3])
        output[:, column] = np.sqrt(dx * dx + dy * dy)
    return output


def assign_identities(
    data_root: Path,
    cameras: list[str],
    frames: list[int],
    views: list[dict],
    debug: list[dict],
    size: int,
    threshold: float,
) -> tuple[list[np.ndarray], list[dict]]:
    labels, diagnostics = [], []
    identity_index = {name: index for index, name in enumerate(IDENTITIES)}
    for camera, frame, view, token_row in zip(cameras, frames, views, debug):
        height, width = [int(value) for value in tensor_numpy(view["true_shape"])[0]]
        detections = processed_head_locations(token_row, height, width)
        pose_annotations = np.load(
            data_root / f"processed_data/poses2d/{camera}/rgb/{frame:05d}.npy",
            allow_pickle=True,
        )
        bbox_annotations = {
            str(row["human_name"]): row
            for row in np.load(
                data_root / f"processed_data/bboxes/{camera}/rgb/{frame:05d}.npy",
                allow_pickle=True,
            )
        }
        targets, target_ids, target_confidence = [], [], []
        target_bboxes = []
        for annotation in pose_annotations:
            point, confidence = gt_pelvis(annotation)
            targets.append(processed_point(point, (2160, 3840), (height, width), size))
            identity_name = str(annotation["human_name"])
            target_ids.append(identity_index[identity_name])
            target_confidence.append(confidence)
            bbox = np.asarray(bbox_annotations[identity_name]["bbox"][:4], dtype=np.float64)
            top_left = processed_point(bbox[:2], (2160, 3840), (height, width), size)
            bottom_right = processed_point(bbox[2:], (2160, 3840), (height, width), size)
            target_bboxes.append(np.r_[top_left, bottom_right])
        targets = np.asarray(targets, dtype=np.float64)
        target_bboxes = np.asarray(target_bboxes, dtype=np.float64)
        pelvis_cost = (
            cdist(detections, targets)
            if len(detections) and len(targets)
            else np.empty((len(detections), len(targets)))
        )
        bbox_cost = points_to_bboxes_distance(detections, target_bboxes)
        cost = bbox_cost * 100.0 + pelvis_cost
        frame_labels = np.full(len(detections), -1, dtype=np.int64)
        if cost.size:
            rows, columns = linear_sum_assignment(cost)
            for row, column in zip(rows, columns):
                if bbox_cost[row, column] <= threshold:
                    frame_labels[row] = target_ids[column]
        labels.append(frame_labels)
        diagnostics.append(
            {
                "camera": camera,
                "frame": frame,
                "human_count": len(detections),
                "detection_locations": detections,
                "gt_pelvis_locations": targets,
                "gt_pelvis_confidence": np.asarray(target_confidence),
                "gt_bboxes": target_bboxes,
                "assignment_cost": cost,
                "bbox_distance_cost": bbox_cost,
                "pelvis_distance_cost": pelvis_cost,
                "labels": frame_labels,
                "assigned_count": int((frame_labels >= 0).sum()),
            }
        )
    return labels, diagnostics


def token_probe(
    predictions: list[dict],
    debug: list[dict],
    labels: list[np.ndarray],
    cuts: tuple[int, ...],
) -> dict:
    output = {}
    shot_starts = (0, *cuts)
    for feature_name in FEATURES:
        output[feature_name] = {}
        for normalized in (False, True):
            mode = "normalized_l2" if normalized else "raw_l2"
            boundaries = []
            for boundary_number, cut in enumerate(cuts):
                source_start = shot_starts[boundary_number]
                prototypes, prototype_ids = [], []
                for identity in range(len(IDENTITIES)):
                    rows = []
                    for frame_index in range(source_start, cut):
                        matches = np.flatnonzero(labels[frame_index] == identity)
                        if len(matches):
                            features = person_feature(
                                debug[frame_index], predictions[frame_index], feature_name
                            )
                            rows.append(features[matches[0]])
                    if rows:
                        prototypes.append(np.mean(np.stack(rows), axis=0))
                        prototype_ids.append(identity)
                target = person_feature(debug[cut], predictions[cut], feature_name)
                cost = feature_cost(np.stack(prototypes), target, normalized)
                source_indices, target_indices = linear_sum_assignment(cost)
                correct = [
                    int(prototype_ids[source] == labels[cut][target_index])
                    for source, target_index in zip(source_indices, target_indices)
                ]
                boundaries.append(
                    {
                        "cut": cut,
                        "prototype_ids": np.asarray(prototype_ids),
                        "target_labels": labels[cut],
                        "cost": cost,
                        "assignment": np.stack([source_indices, target_indices], axis=1),
                        "correct": int(sum(correct)),
                        "total": int(len(correct)),
                    }
                )
            total = sum(row["total"] for row in boundaries)
            output[feature_name][mode] = {
                "accuracy": float(sum(row["correct"] for row in boundaries) / max(total, 1)),
                "correct": int(sum(row["correct"] for row in boundaries)),
                "total": int(total),
                "boundaries": boundaries,
            }
    return output


def tracker_probe(debug: list[dict], labels: list[np.ndarray]) -> dict:
    identity_tracks = {identity: [] for identity in range(len(IDENTITIES))}
    frames = []
    for frame_index, (row, frame_labels) in enumerate(zip(debug, labels)):
        track_ids = (
            np.full(len(frame_labels), -1, dtype=np.int64)
            if row["smpl_ids"] is None
            else tensor_numpy(row["smpl_ids"])[0].astype(np.int64)
        )
        frames.append(
            {"frame_index": frame_index, "gt_labels": frame_labels, "track_ids": track_ids}
        )
        for detection_index, identity in enumerate(frame_labels):
            if identity >= 0:
                identity_tracks[int(identity)].append(int(track_ids[detection_index]))
    return {
        "frames": frames,
        "track_ids_by_identity": identity_tracks,
        "stable": {
            identity: bool(track_ids) and len(set(track_ids)) == 1
            for identity, track_ids in identity_tracks.items()
        },
    }


def weighted_rotation_mean(rotations: list[np.ndarray], weights: np.ndarray | None = None):
    scipy_rotation = Rotation.from_matrix(np.stack(rotations))
    return scipy_rotation.mean(weights=weights).as_matrix()


def rotation_medoid(rotations: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    pairwise = np.zeros((len(rotations), len(rotations)), dtype=np.float64)
    for row, first in enumerate(rotations):
        for column, second in enumerate(rotations):
            relative = first.T @ second
            cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
            pairwise[row, column] = np.arccos(cosine)
    return rotations[int(np.argmin(pairwise @ weights))]


def geometric_median(points: np.ndarray, weights: np.ndarray, iterations: int = 64):
    estimate = np.average(points, axis=0, weights=weights)
    for _ in range(iterations):
        distance = np.linalg.norm(points - estimate, axis=1)
        if float(distance.min()) < 1e-8:
            return points[int(np.argmin(distance))]
        effective = weights / np.maximum(distance, 1e-8)
        updated = np.average(points, axis=0, weights=effective)
        if np.linalg.norm(updated - estimate) < 1e-9:
            break
        estimate = updated
    return estimate


def geometry_probe(
    predictions: list[dict],
    debug: list[dict],
    labels: list[np.ndarray],
    cameras: list[str],
    frames: list[int],
    exo: dict,
    cuts: tuple[int, ...],
) -> dict:
    output = []
    for cut in cuts:
        previous_index, current_index = cut - 1, cut
        previous_pose = camera_matrix(predictions[previous_index]).astype(np.float64)
        current_pose = camera_matrix(predictions[current_index]).astype(np.float64)
        previous_map = {
            int(identity): index
            for index, identity in enumerate(labels[previous_index])
            if identity >= 0
        }
        current_map = {
            int(identity): index
            for index, identity in enumerate(labels[current_index])
            if identity >= 0
        }
        shared = sorted(set(previous_map) & set(current_map))
        previous_roots = tensor_numpy(predictions[previous_index]["smpl_transl"])[0]
        current_roots = tensor_numpy(predictions[current_index]["smpl_transl"])[0]
        previous_body = tensor_numpy(predictions[previous_index]["smpl_rotmat"])[0, :, 0]
        current_body = tensor_numpy(predictions[current_index]["smpl_rotmat"])[0, :, 0]
        previous_scores = tensor_numpy(debug[previous_index]["head_scores"])
        current_scores = tensor_numpy(debug[current_index]["head_scores"])

        anchors, local_roots, candidate_rotations, weights = [], [], [], []
        for identity in shared:
            previous_detection = previous_map[identity]
            current_detection = current_map[identity]
            anchors.append(
                previous_pose[:3, :3] @ previous_roots[previous_detection]
                + previous_pose[:3, 3]
            )
            local_roots.append(
                current_pose[:3, :3] @ current_roots[current_detection]
                + current_pose[:3, 3]
            )
            old_body = previous_pose[:3, :3] @ previous_body[previous_detection]
            new_body = current_pose[:3, :3] @ current_body[current_detection]
            candidate_rotations.append(old_body @ new_body.T)
            weights.append(
                max(
                    1e-4,
                    float(previous_scores[previous_detection] * current_scores[current_detection]),
                )
            )
        anchors = np.stack(anchors)
        local_roots = np.stack(local_roots)
        weights = np.asarray(weights, dtype=np.float64)
        weights /= weights.sum()

        methods = {}
        for local_index, identity in enumerate(shared):
            rotation = candidate_rotations[local_index]
            translation = anchors[local_index] - rotation @ local_roots[local_index]
            methods[f"single_{IDENTITIES[identity]}"] = (rotation, translation)

        mean_rotation = weighted_rotation_mean(candidate_rotations)
        mean_translations = anchors - local_roots @ mean_rotation.T
        methods["mean_consensus"] = (mean_rotation, mean_translations.mean(axis=0))

        weighted_rotation = weighted_rotation_mean(candidate_rotations, weights)
        weighted_translations = anchors - local_roots @ weighted_rotation.T
        methods["confidence_weighted"] = (
            weighted_rotation,
            np.average(weighted_translations, axis=0, weights=weights),
        )

        robust_rotation = rotation_medoid(candidate_rotations, weights)
        robust_translations = anchors - local_roots @ robust_rotation.T
        methods["rotation_medoid_translation_geomedian"] = (
            robust_rotation,
            geometric_median(robust_translations, weights),
        )

        gt_previous = exo[cameras[previous_index]]["c2w_aria01"]
        gt_current = exo[cameras[current_index]]["c2w_aria01"]
        evaluation_gauge = previous_pose @ np.linalg.inv(gt_previous)
        target_camera = evaluation_gauge @ gt_current
        method_rows = {}
        for method_name, (rotation, translation) in methods.items():
            boundary = np.eye(4, dtype=np.float64)
            boundary[:3, :3] = rotation
            boundary[:3, 3] = translation
            final_camera = boundary @ current_pose
            translation_candidates = anchors - local_roots @ rotation.T
            layout_residuals = []
            for first in range(len(shared)):
                for second in range(first + 1, len(shared)):
                    layout_residuals.append(
                        np.linalg.norm(
                            (anchors[first] - anchors[second])
                            - rotation @ (local_roots[first] - local_roots[second])
                        )
                    )
            method_rows[method_name] = {
                "boundary": boundary,
                "camera_translation_error_m": float(
                    np.linalg.norm(final_camera[:3, 3] - target_camera[:3, 3])
                ),
                "camera_rotation_error_deg": rotation_error_deg(final_camera, target_camera),
                "translation_candidate_dispersion_max_m": float(
                    np.max(cdist(translation_candidates, translation_candidates))
                ),
                "pairwise_layout_residual_mean_m": (
                    float(np.mean(layout_residuals)) if layout_residuals else float("nan")
                ),
                "pairwise_layout_residual_available": bool(layout_residuals),
            }
        single_names = [name for name in method_rows if name.startswith("single_")]
        best_single = min(
            single_names,
            key=lambda name: method_rows[name]["camera_translation_error_m"]
            + 0.02 * method_rows[name]["camera_rotation_error_deg"],
        )
        output.append(
            {
                "cut": cut,
                "previous_camera": cameras[previous_index],
                "current_camera": cameras[current_index],
                "synchronized_frame": frames[current_index],
                "shared_identities": [IDENTITIES[index] for index in shared],
                "head_score_weights": weights,
                "best_single_oracle": best_single,
                "methods": method_rows,
            }
        )
    return {"boundaries": output}


def plot_token_matrices(report: dict, cuts: tuple[int, ...], output_path: Path) -> None:
    features = ("refined_human_tokens", "fused_human_prompts", "smpl_beta")
    figure, axes = plt.subplots(
        len(features),
        len(cuts),
        figsize=(5 * len(cuts), 10),
        constrained_layout=True,
        squeeze=False,
    )
    for row, feature in enumerate(features):
        boundaries = report[feature]["normalized_l2"]["boundaries"]
        for column, boundary in enumerate(boundaries):
            cost = np.asarray(boundary["cost"])
            image = axes[row, column].imshow(cost, cmap="magma")
            prototype_names = [IDENTITIES[int(value)] for value in boundary["prototype_ids"]]
            target_names = [
                "unmatched" if int(value) < 0 else IDENTITIES[int(value)]
                for value in boundary["target_labels"]
            ]
            axes[row, column].set_yticks(range(len(prototype_names)), prototype_names)
            axes[row, column].set_xticks(
                range(len(target_names)), target_names, rotation=30, ha="right"
            )
            axes[row, column].set_title(f"{feature}\ncut {boundary['cut']}")
            figure.colorbar(image, ax=axes[row, column], fraction=0.046)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(torch.device(args.device))

    segments = parse_segments(args.segments)
    cuts = segment_cuts(segments)
    colmap, exo = load_colmap(args.data_root)
    projection = projection_audit(args.data_root, colmap, exo)
    unknown_cameras = sorted({camera for camera, _ in segments} - set(exo))
    if unknown_cameras:
        raise ValueError(f"Unknown exo cameras: {unknown_cameras}")
    cameras = [camera for camera, segment_frames in segments for _ in segment_frames]
    frames = [frame for _, segment_frames in segments for frame in segment_frames]
    image_paths = [
        args.data_root / f"exo/{camera}/images/{frame:05d}.jpg"
        for camera, frame in zip(cameras, frames)
    ]

    model = build_model(args)
    started = time.perf_counter()
    predictions, views, debug = run_stream(model, image_paths, list(cuts), args)
    labels, assignment = assign_identities(
        args.data_root,
        cameras,
        frames,
        views,
        debug,
        int(args.size),
        float(args.assignment_threshold_px),
    )
    tokens = token_probe(predictions, debug, labels, cuts)
    tracking = tracker_probe(debug, labels)
    geometry = geometry_probe(predictions, debug, labels, cameras, frames, exo, cuts)

    report = {
        "experiment": "Movie3R-Multi V13 EgoHumans feasibility probe",
        "legacy_experiment": "V20 EgoBody three-person feasibility probe (legacy name)",
        "scope": "Lite smoke test; no DA3, keypoint scale, VGGT, V11.4, or scene refinement",
        "candidate_gt_usage": {
            "human3r_inference": False,
            "token_matching_cost": False,
            "identity_labels_for_scoring_and_prototypes": True,
            "camera_pose_for_candidate": False,
            "camera_pose_for_evaluation_only": True,
        },
        "dataset": {
            "root": str(args.data_root),
            "identities": IDENTITIES,
            "frame_count": 601,
            "exo_camera_count": 8,
            "gt_body_model": "SMPL, 6890 vertices, 45 joints",
            "world_gauge": "aria01 metric world",
            "colmap_from_aria01_scale": colmap["similarity_scale"],
            "projection_audit": projection,
            "camera_pose_record_spread": {
                name: {
                    "record_count": row["pose_record_count"],
                    "center_spread_max_m_colmap": row["center_spread_max_m_colmap"],
                }
                for name, row in exo.items()
            },
        },
        "stream": {
            "segments": [
                {"camera": camera, "frames": segment_frames}
                for camera, segment_frames in segments
            ],
            "cuts": cuts,
            "same_timestamp_at_boundary": all(
                frames[cut - 1] == frames[cut] for cut in cuts
            ),
            "human_counts": [int(row["num_humans"]) for row in debug],
            "labels_by_detection": labels,
            "assignment": assignment,
        },
        "native_tracker": tracking,
        "token_probe": tokens,
        "gt_id_geometry_smoke": geometry,
        "runtime_seconds_excluding_model_load": time.perf_counter() - started,
    }
    report_path = args.output_dir / "v13_egobody_three_person_probe.json"
    report_path.write_text(
        json.dumps(jsonable(report), indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    plot_token_matrices(
        tokens, cuts, args.output_dir / "egobody_token_distance_matrices.png"
    )
    compact_predictions = [
        {
            key: prediction[key]
            for key in ("camera_pose", "smpl_shape", "smpl_transl", "smpl_rotmat", "smpl_id")
            if key in prediction
        }
        for prediction in predictions
    ]
    torch.save(
        {"predictions": compact_predictions, "token_debug": debug, "labels": labels},
        args.output_dir / "v13_egobody_compact_tokens.pt",
    )
    print(f">> EgoHumans V13 report: {report_path}", flush=True)
    print(f">> Runtime excluding model load: {report['runtime_seconds_excluding_model_load']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
