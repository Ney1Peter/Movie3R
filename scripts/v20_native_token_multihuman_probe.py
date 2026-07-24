#!/usr/bin/env python3
"""Minimal V20 native-token Re-ID and GT-ID multi-human geometry probe."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from scipy.ndimage import distance_transform_edt
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from demo import prepare_input  # noqa: E402
from scripts.v14_1_shot_aware_state_routing_probe import (  # noqa: E402
    build_model,
    camera_matrix,
)


FEATURES = (
    "refined_human_tokens",
    "fused_human_prompts",
    "mhmr_head_tokens",
    "cut3r_head_tokens",
    "smpl_beta",
    "local_pose",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--multihuman_root",
        type=Path,
        default=Path("/data/wangzheng/iJCV-CODE/data/dataset/multihuman"),
    )
    parser.add_argument(
        "--rich_root",
        type=Path,
        default=Path("/data/wangzheng/iJCV-CODE/data/RICH"),
    )
    parser.add_argument(
        "--model_path", type=Path, default=ROOT / "src/human3r_896L.pth"
    )
    parser.add_argument(
        "--output_dir", type=Path, default=ROOT / "output/v20_native_token_multihuman"
    )
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def tensor_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)


def person_rows(value, num_humans: int, name: str) -> np.ndarray:
    if num_humans == 0:
        return np.empty((0, 0), dtype=np.float32)
    array = tensor_numpy(value)
    while array.ndim > 1 and array.shape[0] == 1 and array.shape[0] != num_humans:
        array = array[0]
    if array.ndim == 1 and num_humans == 1:
        array = array[None]
    if array.ndim < 2 or array.shape[0] != num_humans:
        raise ValueError(
            f"Cannot interpret {name} shape {array.shape} as {num_humans} person rows"
        )
    return array.reshape(num_humans, -1)


def person_feature(debug: dict, prediction: dict, name: str) -> np.ndarray:
    num_humans = int(debug["num_humans"])
    if name in debug:
        return person_rows(debug[name], num_humans, name)
    if name == "smpl_beta":
        return person_rows(prediction["smpl_shape"], num_humans, name)
    if name == "local_pose":
        rotations = tensor_numpy(prediction["smpl_rotmat"])
        while rotations.ndim > 4 and rotations.shape[0] == 1:
            rotations = rotations[0]
        if rotations.ndim != 4 or rotations.shape[0] != num_humans:
            raise ValueError(
                f"Cannot interpret local_pose shape {rotations.shape} for {num_humans} humans"
            )
        rotations = rotations[:, 1:]
        return rotations.reshape(rotations.shape[0], -1)
    raise KeyError(name)


def normalized_rows(value: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(value, axis=1, keepdims=True)
    return value / np.maximum(norm, 1e-8)


def feature_cost(first: np.ndarray, second: np.ndarray, normalized: bool) -> np.ndarray:
    if normalized:
        first = normalized_rows(first)
        second = normalized_rows(second)
    return cdist(first, second, metric="euclidean")


def processed_head_locations(debug: dict, height: int, width: int) -> np.ndarray:
    locations = tensor_numpy(debug["head_locations"])
    scale = max(height, width) / 896.0
    locations = locations * scale
    locations[:, 0] -= (max(height, width) - width) // 2
    locations[:, 1] -= (max(height, width) - height) // 2
    locations[:, 0] = np.clip(locations[:, 0], 0, width - 1)
    locations[:, 1] = np.clip(locations[:, 1], 0, height - 1)
    return locations


def resize_crop_mask(mask: np.ndarray, height: int, width: int, size: int) -> np.ndarray:
    original_height, original_width = mask.shape
    scale = float(size) / float(max(original_width, original_height))
    resized_width = int(round(original_width * scale))
    resized_height = int(round(original_height * scale))
    resized = cv2.resize(
        mask.astype(np.uint8),
        (resized_width, resized_height),
        interpolation=cv2.INTER_NEAREST,
    )
    left = (resized_width - width) // 2
    top = (resized_height - height) // 2
    return resized[top : top + height, left : left + width] > 0


def assign_static_identities(
    debug: dict,
    masks: list[np.ndarray],
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    locations = processed_head_locations(debug, height, width)
    distances = np.zeros((len(locations), len(masks)), dtype=np.float64)
    for identity, mask in enumerate(masks):
        distance = distance_transform_edt(~mask)
        xy = np.rint(locations).astype(int)
        distances[:, identity] = distance[xy[:, 1], xy[:, 0]]
    labels = np.full(len(locations), -1, dtype=np.int64)
    if distances.size:
        rows, columns = linear_sum_assignment(distances)
        labels[rows] = columns
    return labels, distances


def load_opencv_camera(path: Path) -> tuple[np.ndarray, np.ndarray]:
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    extrinsic = storage.getNode("CameraMatrix").mat().astype(np.float64)
    intrinsic = storage.getNode("Intrinsics").mat().astype(np.float64)
    storage.release()
    return extrinsic, intrinsic


def c2w_from_w2c(extrinsic: np.ndarray) -> np.ndarray:
    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[:3] = extrinsic
    return np.linalg.inv(world_to_camera)


def rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first[:3, :3].T @ second[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return jsonable(value.detach().float().cpu().numpy())
    return value


def run_stream(model, image_paths: list[Path], cuts: list[int], args) -> tuple[list, list, list]:
    views = prepare_input(
        img_paths=[str(path) for path in image_paths],
        img_mask=[True] * len(image_paths),
        size=int(args.size),
        revisit=1,
        update=True,
        img_res=getattr(model, "mhmr_img_res", None),
        reset_interval=10_000_000,
    )
    for view in views:
        view["reset"] = torch.zeros_like(view["reset"], dtype=torch.bool)
    routing = {
        "enabled": True,
        "mode": "tracklet",
        "cut_indices": cuts,
        "consume_view_reset_at_cut": True,
    }
    with torch.no_grad():
        predictions, output_views, debug = model.forward_recurrent_lighter(
            views,
            str(args.device),
            ret_state=False,
            use_ttt3r=False,
            return_token_debug=True,
            shot_routing=routing,
        )
    return predictions, output_views, debug


def static_identity_labels(
    multihuman_root: Path,
    views: list[dict],
    debug: list[dict],
    angles: list[int],
    size: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    all_labels, all_costs = [], []
    for view, row, angle in zip(views, debug, angles):
        height, width = [int(value) for value in tensor_numpy(view["true_shape"])[0]]
        masks = []
        for identity in (0, 1):
            mask = cv2.imread(
                str(multihuman_root / f"mask/20_{identity}/{angle}.jpg"),
                cv2.IMREAD_GRAYSCALE,
            ) > 127
            masks.append(resize_crop_mask(mask, height, width, size))
        labels, costs = assign_static_identities(row, masks, height, width)
        all_labels.append(labels)
        all_costs.append(costs)
    return all_labels, all_costs


def assignment_probe(
    predictions: list[dict],
    debug: list[dict],
    labels: list[np.ndarray],
) -> dict:
    output = {}
    for feature_name in FEATURES:
        output[feature_name] = {}
        for normalized in (False, True):
            mode = "normalized_l2" if normalized else "raw_l2"
            rows = []
            same_distances, different_distances = [], []
            for boundary in range(1, len(predictions)):
                source_ids, target_ids = labels[boundary - 1], labels[boundary]
                if len(source_ids) == 0 or len(target_ids) == 0:
                    rows.append(
                        {
                            "boundary": boundary,
                            "cost": np.empty((len(source_ids), len(target_ids))),
                            "assignment": np.empty((0, 2), dtype=np.int64),
                            "correct": 0,
                            "total": 0,
                            "evaluable": False,
                        }
                    )
                    continue
                first = person_feature(debug[boundary - 1], predictions[boundary - 1], feature_name)
                second = person_feature(debug[boundary], predictions[boundary], feature_name)
                cost = feature_cost(first, second, normalized)
                source, target = linear_sum_assignment(cost)
                correct = [
                    int(source_ids[a] >= 0 and source_ids[a] == target_ids[b])
                    for a, b in zip(source, target)
                ]
                for source_index in range(len(source_ids)):
                    for target_index in range(len(target_ids)):
                        if source_ids[source_index] < 0 or target_ids[target_index] < 0:
                            continue
                        bucket = (
                            same_distances
                            if source_ids[source_index] == target_ids[target_index]
                            else different_distances
                        )
                        bucket.append(float(cost[source_index, target_index]))
                rows.append(
                    {
                        "boundary": boundary,
                        "cost": cost,
                        "assignment": np.stack([source, target], axis=1),
                        "correct": int(sum(correct)),
                        "total": int(len(correct)),
                        "evaluable": True,
                    }
                )
            total = sum(row["total"] for row in rows)
            output[feature_name][mode] = {
                "accuracy": float(sum(row["correct"] for row in rows) / max(total, 1)),
                "same_distance_mean": float(np.mean(same_distances)) if same_distances else math.nan,
                "different_distance_mean": float(np.mean(different_distances)) if different_distances else math.nan,
                "boundaries": rows,
            }
    return output


def tracker_identity_probe(debug: list[dict], labels: list[np.ndarray]) -> dict:
    by_identity = {0: [], 1: []}
    frame_rows = []
    for frame, (row, frame_labels) in enumerate(zip(debug, labels)):
        smpl_ids = (
            np.full(len(frame_labels), -1, dtype=np.int64)
            if row["smpl_ids"] is None
            else tensor_numpy(row["smpl_ids"])[0].astype(np.int64)
        )
        frame_rows.append({"frame": frame, "gt_labels": frame_labels, "track_ids": smpl_ids})
        for index, identity in enumerate(frame_labels):
            if identity in by_identity:
                by_identity[int(identity)].append(int(smpl_ids[index]))
    return {
        "frames": frame_rows,
        "track_ids_by_gt_identity": by_identity,
        "stable": {
            identity: len(set(values)) <= 1 and bool(values)
            for identity, values in by_identity.items()
        },
    }


def static_geometry_probe(
    multihuman_root: Path,
    predictions: list[dict],
    labels: list[np.ndarray],
    angles: list[int],
) -> dict:
    identity_to_index = []
    for frame_labels in labels:
        identity_to_index.append(
            {int(identity): int(index) for index, identity in enumerate(frame_labels) if identity >= 0}
        )
    valid_frames = [
        frame for frame, identities in enumerate(identity_to_index) if set(identities) == {0, 1}
    ]
    if len(valid_frames) < 2:
        return {
            "available": False,
            "reason": "Fewer than two views contain both GT-assigned people",
            "valid_two_person_frames": valid_frames,
            "boundaries": [],
            "summary": {},
        }

    anchor_frame = valid_frames[0]
    first_pose = camera_matrix(predictions[anchor_frame])
    first_extrinsic = np.load(
        multihuman_root / f"parameter/20_0/{angles[anchor_frame]}_extrinsic.npy"
    )
    first_gt_camera = c2w_from_w2c(first_extrinsic)
    gauge = first_pose @ np.linalg.inv(first_gt_camera)

    anchors, body_world = {}, {}
    first_transl = tensor_numpy(predictions[anchor_frame]["smpl_transl"])[0]
    first_rotmat = tensor_numpy(predictions[anchor_frame]["smpl_rotmat"])[0, :, 0]
    for identity in (0, 1):
        index = identity_to_index[anchor_frame][identity]
        anchors[identity] = first_pose[:3, :3] @ first_transl[index] + first_pose[:3, 3]
        body_world[identity] = first_pose[:3, :3] @ first_rotmat[index]

    boundaries = []
    for frame in valid_frames[1:]:
        target_extrinsic = np.load(
            multihuman_root / f"parameter/20_0/{angles[frame]}_extrinsic.npy"
        )
        target = gauge @ c2w_from_w2c(target_extrinsic)
        transl = tensor_numpy(predictions[frame]["smpl_transl"])[0]
        rotmat = tensor_numpy(predictions[frame]["smpl_rotmat"])[0, :, 0]
        candidate_rotations = {}
        for identity in (0, 1):
            index = identity_to_index[frame][identity]
            candidate_rotations[identity] = body_world[identity] @ rotmat[index].T

        methods = {}
        for identity in (0, 1):
            rotation = candidate_rotations[identity]
            index = identity_to_index[frame][identity]
            translation = anchors[identity] - rotation @ transl[index]
            camera = np.eye(4, dtype=np.float64)
            camera[:3, :3] = rotation
            camera[:3, 3] = translation
            methods[f"single_gt_{identity}"] = {
                "camera": camera,
                "translation_candidates": {identity: translation},
            }

        rotation = Rotation.from_matrix(
            np.stack([candidate_rotations[0], candidate_rotations[1]])
        ).mean().as_matrix()
        translation_candidates = {}
        for identity in (0, 1):
            index = identity_to_index[frame][identity]
            translation_candidates[identity] = anchors[identity] - rotation @ transl[index]
        translation = np.mean(np.stack(list(translation_candidates.values())), axis=0)
        camera = np.eye(4, dtype=np.float64)
        camera[:3, :3] = rotation
        camera[:3, 3] = translation
        methods["two_human_consensus"] = {
            "camera": camera,
            "translation_candidates": translation_candidates,
        }

        for method in methods.values():
            camera = method["camera"]
            method["camera_translation_error_m"] = float(
                np.linalg.norm(camera[:3, 3] - target[:3, 3])
            )
            method["camera_rotation_error_deg"] = rotation_error_deg(camera, target)
            candidates = list(method["translation_candidates"].values())
            method["translation_candidate_dispersion_m"] = (
                float(np.linalg.norm(candidates[0] - candidates[1]))
                if len(candidates) == 2
                else 0.0
            )

        first_index = identity_to_index[frame][0]
        second_index = identity_to_index[frame][1]
        local_layout = transl[first_index] - transl[second_index]
        world_layout = anchors[0] - anchors[1]
        methods["two_human_consensus"]["pairwise_layout_residual_m"] = float(
            np.linalg.norm(world_layout - rotation @ local_layout)
        )
        boundaries.append({"frame": frame, "angle": angles[frame], "methods": methods})

    summary = {}
    for name in ("single_gt_0", "single_gt_1", "two_human_consensus"):
        summary[name] = {
            "camera_translation_error_m": float(
                np.mean([row["methods"][name]["camera_translation_error_m"] for row in boundaries])
            ),
            "camera_rotation_error_deg": float(
                np.mean([row["methods"][name]["camera_rotation_error_deg"] for row in boundaries])
            ),
        }
    return {
        "available": True,
        "anchor_frame": anchor_frame,
        "valid_two_person_frames": valid_frames,
        "boundaries": boundaries,
        "summary": summary,
    }


def rich_gt_bbox(
    rich_root: Path,
    frame: int,
    camera: int,
    height: int,
    width: int,
    size: int,
) -> tuple[np.ndarray, float, float]:
    mesh = trimesh.load(
        rich_root / f"body/ParkingLot1_002_stretching1/{frame:05d}/002.ply",
        process=False,
    )
    vertices = np.asarray(mesh.vertices)
    extrinsic, intrinsic = load_opencv_camera(
        rich_root / f"cam/ParkingLot1/calibration/{camera:03d}.xml"
    )
    points = vertices @ extrinsic[:, :3].T + extrinsic[:, 3]
    pixels = (points[:, :2] / points[:, 2:3]) @ intrinsic[:2, :2].T + intrinsic[:2, 2]
    original_width, original_height = 4112, 3008
    scale = float(size) / float(max(original_width, original_height))
    resized_width = int(round(original_width * scale))
    resized_height = int(round(original_height * scale))
    left = (resized_width - width) // 2
    top = (resized_height - height) // 2
    pixels[:, 0] = pixels[:, 0] * scale - left
    pixels[:, 1] = pixels[:, 1] * scale - top
    bbox = np.r_[pixels.min(axis=0), pixels.max(axis=0)]
    inside = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    clipped_width = max(0.0, min(float(width), bbox[2]) - max(0.0, bbox[0]))
    clipped_height = max(0.0, min(float(height), bbox[3]) - max(0.0, bbox[1]))
    area = clipped_width * clipped_height / float(width * height)
    return bbox, float(inside.mean()), float(area)


def point_bbox_distance(points: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    dx = np.maximum(np.maximum(bbox[0] - points[:, 0], 0.0), points[:, 0] - bbox[2])
    dy = np.maximum(np.maximum(bbox[1] - points[:, 1], 0.0), points[:, 1] - bbox[3])
    return np.sqrt(dx * dx + dy * dy)


def rich_identity_probe(
    rich_root: Path,
    predictions: list[dict],
    views: list[dict],
    debug: list[dict],
    frames: list[int],
    cameras: list[int],
    cuts: list[int],
    size: int,
) -> dict:
    actor_indices, quality_rows = [], []
    for index, (view, row, frame, camera) in enumerate(zip(views, debug, frames, cameras)):
        height, width = [int(value) for value in tensor_numpy(view["true_shape"])[0]]
        bbox, visible_fraction, area = rich_gt_bbox(
            rich_root, frame, camera, height, width, size
        )
        locations = processed_head_locations(row, height, width)
        distances = point_bbox_distance(locations, bbox)
        with (
            rich_root
            / f"body/ParkingLot1_002_stretching1/{frame:05d}/002.pkl"
        ).open("rb") as handle:
            gt_body = pickle.load(handle)
        extrinsic, _ = load_opencv_camera(
            rich_root / f"cam/ParkingLot1/calibration/{camera:03d}.xml"
        )
        gt_world_root = np.asarray(gt_body["transl"], dtype=np.float64).reshape(-1, 3)[0]
        gt_camera_root = extrinsic[:, :3] @ gt_world_root + extrinsic[:, 3]
        predicted_roots = tensor_numpy(predictions[index]["smpl_transl"])[0]
        root_depth_cost = np.abs(predicted_roots[:, 2] - gt_camera_root[2])
        actor_index = int(np.argmin(root_depth_cost))
        actor_indices.append(actor_index)
        scores = tensor_numpy(row["head_scores"])
        smpl_ids = tensor_numpy(row["smpl_ids"])[0].astype(int)
        quality_rows.append(
            {
                "frame_index": index,
                "source_frame": frame,
                "camera": camera,
                "human_count": len(locations),
                "gt_actor_index": actor_index,
                "index0_is_gt_actor": actor_index == 0,
                "gt_actor_track_id": int(smpl_ids[actor_index]),
                "gt_actor_head_score": float(scores[actor_index]),
                "gt_camera_root_depth_m": float(gt_camera_root[2]),
                "predicted_root_depths_m": predicted_roots[:, 2],
                "root_depth_label_cost_m": root_depth_cost,
                "gt_bbox": bbox,
                "gt_visible_vertex_fraction": visible_fraction,
                "gt_bbox_area_ratio": area,
                "head_to_gt_bbox_distance": distances,
            }
        )

    retrieval = {}
    for feature_name in FEATURES:
        retrieval[feature_name] = {}
        for normalized in (False, True):
            mode = "normalized_l2" if normalized else "raw_l2"
            rows = []
            for cut in cuts:
                source_start = max([0, *[value for value in cuts if value < cut]])
                source_tokens = []
                for frame_index in range(source_start, cut):
                    values = person_feature(
                        debug[frame_index], predictions[frame_index], feature_name
                    )
                    source_tokens.append(values[actor_indices[frame_index]])
                prototype = np.mean(np.stack(source_tokens), axis=0, keepdims=True)
                target = person_feature(debug[cut], predictions[cut], feature_name)
                cost = feature_cost(prototype, target, normalized)[0]
                order = np.argsort(cost)
                expected = actor_indices[cut]
                rank = int(np.flatnonzero(order == expected)[0]) + 1
                rows.append(
                    {
                        "cut": cut,
                        "expected_actor_index": expected,
                        "retrieved_index": int(order[0]),
                        "rank": rank,
                        "correct": int(order[0] == expected),
                        "cost": cost,
                    }
                )
            retrieval[feature_name][mode] = {
                "accuracy": float(np.mean([row["correct"] for row in rows])),
                "mean_rank": float(np.mean([row["rank"] for row in rows])),
                "boundaries": rows,
            }

    actor_track_ids = [row["gt_actor_track_id"] for row in quality_rows]
    return {
        "quality": quality_rows,
        "retrieval": retrieval,
        "actor_track_ids": actor_track_ids,
        "actor_track_stable": len(set(actor_track_ids)) == 1,
        "index0_actor_rate": float(np.mean([row["index0_is_gt_actor"] for row in quality_rows])),
    }


def plot_static_matrices(report: dict, path: Path) -> None:
    features = ("refined_human_tokens", "fused_human_prompts", "smpl_beta")
    fig, axes = plt.subplots(len(features), 3, figsize=(9, 9), constrained_layout=True)
    for row, feature in enumerate(features):
        boundaries = report[feature]["normalized_l2"]["boundaries"]
        for column, boundary in enumerate(boundaries):
            cost = np.asarray(boundary["cost"])
            if cost.size == 0:
                axes[row, column].text(
                    0.5, 0.5, "not evaluable\n(no detection)", ha="center", va="center"
                )
                axes[row, column].set_axis_off()
                continue
            image = axes[row, column].imshow(cost, cmap="magma")
            axes[row, column].set_title(f"{feature}\ncut {boundary['boundary']}", fontsize=8)
            axes[row, column].set_xlabel("post detection")
            axes[row, column].set_ylabel("pre detection")
            fig.colorbar(image, ax=axes[row, column], fraction=0.046)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_rich_matrices(report: dict, path: Path) -> None:
    features = ("refined_human_tokens", "fused_human_prompts", "smpl_beta")
    fig, axes = plt.subplots(len(features), 2, figsize=(8, 9), constrained_layout=True)
    for row, feature in enumerate(features):
        boundaries = report[feature]["normalized_l2"]["boundaries"]
        for column, boundary in enumerate(boundaries):
            cost = np.asarray(boundary["cost"])[None]
            image = axes[row, column].imshow(cost, cmap="magma", aspect="auto")
            axes[row, column].axvline(
                boundary["expected_actor_index"], color="cyan", linewidth=2
            )
            axes[row, column].set_title(f"{feature}\ncut {boundary['cut']}", fontsize=8)
            axes[row, column].set_xlabel("post detection (cyan=GT 002)")
            axes[row, column].set_yticks([0])
            fig.colorbar(image, ax=axes[row, column], fraction=0.046)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("V20 token probe requires CUDA")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(torch.device(args.device))
    model = build_model(args)
    started = time.perf_counter()

    angles = [0, 90, 180, 270]
    static_paths = [args.multihuman_root / f"img/20_0/{angle}.jpg" for angle in angles]
    static_predictions, static_views, static_debug = run_stream(
        model, static_paths, [1, 2, 3], args
    )
    static_labels, static_label_costs = static_identity_labels(
        args.multihuman_root, static_views, static_debug, angles, int(args.size)
    )
    static_assignments = assignment_probe(static_predictions, static_debug, static_labels)
    static_tracker = tracker_identity_probe(static_debug, static_labels)
    static_geometry = static_geometry_probe(
        args.multihuman_root, static_predictions, static_labels, angles
    )

    rich_frames = list(range(190, 205))
    rich_cameras = [3] * 5 + [4] * 5 + [6] * 5
    rich_paths = [
        args.rich_root
        / f"ParkingLot1_002_stretching1/cam_{camera:02d}/{frame:05d}_{camera:02d}.bmp"
        for frame, camera in zip(rich_frames, rich_cameras)
    ]
    rich_predictions, rich_views, rich_debug = run_stream(
        model, rich_paths, [5, 10], args
    )
    rich_report = rich_identity_probe(
        args.rich_root,
        rich_predictions,
        rich_views,
        rich_debug,
        rich_frames,
        rich_cameras,
        [5, 10],
        int(args.size),
    )

    report = {
        "experiment": "V20 native-token two-example feasibility probe",
        "scope": (
            "Smoke test only: static two-person GT plus RICH GT-002 with unannotated distractors"
        ),
        "routing": {
            "scene_camera": "fresh pre-decode state at each cut",
            "human_tracklet": "preserved raw refined-token bank",
            "token_directly_predicts_geometry": False,
        },
        "static_multihuman": {
            "angles": angles,
            "human_counts": [int(row["num_humans"]) for row in static_debug],
            "gt_labels_by_detection": static_labels,
            "gt_label_costs": static_label_costs,
            "feature_assignment": static_assignments,
            "native_tracker": static_tracker,
            "gt_id_geometry": static_geometry,
        },
        "rich": {
            "frames": rich_frames,
            "cameras": rich_cameras,
            "cuts": [5, 10],
            **rich_report,
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    report_path = args.output_dir / "v20_two_example_probe.json"
    report_path.write_text(
        json.dumps(jsonable(report), indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    torch.save(
        {
            "static": {"predictions": static_predictions, "token_debug": static_debug},
            "rich": {"predictions": rich_predictions, "token_debug": rich_debug},
        },
        args.output_dir / "v20_two_example_token_dump.pt",
    )
    plot_static_matrices(
        jsonable(static_assignments), args.output_dir / "static_token_distance_matrices.png"
    )
    plot_rich_matrices(
        jsonable(rich_report["retrieval"]), args.output_dir / "rich_token_retrieval.png"
    )
    print(f">> V20 report: {report_path}", flush=True)
    print(f">> Runtime: {report['runtime_seconds']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
