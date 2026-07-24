#!/usr/bin/env python3
"""Movie3R-Multi V13 GT-ID shared-Boundary feasibility study.

Ground-truth identity is used only to associate Human3R detections across the
cut. Candidate generation uses predicted roots, torso frames, confidences and
pointmaps. GT cameras and SMPL-X meshes are read only by the evaluator.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, permutations
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import roma
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.boundary_human3r_reset_support import build_smpl_models, torso_frame  # noqa: E402
from scripts.v10_implicit_explicit_cross_shot_probe import (  # noqa: E402
    robust_local_pointmap_refinement,
)
from versions.v12.experiments.v14_1_shot_aware_state_routing_probe import (  # noqa: E402
    build_model,
    camera_matrix,
)
from versions.v13.native_token_probe import jsonable, tensor_numpy  # noqa: E402


IDENTITIES = ("person0", "person1", "person2")
VIDEO_NAMES = (
    "SaveToAvi-MJPG-18181923-0000.mp4",
    "SaveToAvi-MJPG-18181924-0000.mp4",
    "SaveToAvi-MJPG-18307701-0000.mp4",
    "SaveToAvi-MJPG-18307863-0000.mp4",
    "SaveToAvi-MJPG-18307864-0000.mp4",
    "SaveToAvi-MJPG-18307870-0000.mp4",
)


@dataclass(frozen=True)
class CaseSpec:
    timestamp: int
    source_camera: int
    target_camera: int
    offset: int

    @property
    def key(self) -> str:
        return (
            f"three_t{self.timestamp:04d}_c{self.source_camera}"
            f"_c{self.target_camera}_k{self.offset}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path(
            "/data/wangzheng/iJCV-CODE/data/MultiHuman/Real-World-Capture/extracted"
        ),
    )
    parser.add_argument("--model_path", type=Path, default=ROOT / "src/human3r_896L.pth")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ROOT / "output/v13/multihuman",
    )
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--history_frames", type=int, default=5)
    parser.add_argument("--point_samples", type=int, default=1024)
    parser.add_argument("--timestamps", type=int, nargs="+", default=(500, 700, 900, 1100, 1300, 1500))
    parser.add_argument(
        "--camera_pairs",
        nargs="+",
        default=("0-1", "1-2", "2-3", "3-4", "4-5", "5-0", "0-3", "1-4", "2-5"),
    )
    parser.add_argument("--offsets", type=int, nargs="+", default=(0,))
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--evaluation_only", action="store_true")
    return parser.parse_args()


def parse_camera_pairs(values: list[str]) -> tuple[tuple[int, int], ...]:
    output = []
    for value in values:
        first, second = value.split("-", maxsplit=1)
        pair = (int(first), int(second))
        if pair[0] == pair[1] or min(pair) < 0 or max(pair) >= len(VIDEO_NAMES):
            raise ValueError(f"Invalid camera pair: {value}")
        output.append(pair)
    return tuple(output)


def case_specs(args: argparse.Namespace) -> list[CaseSpec]:
    pairs = parse_camera_pairs(list(args.camera_pairs))
    specs = [
        CaseSpec(int(timestamp), source, target, int(offset))
        for offset in args.offsets
        for timestamp in args.timestamps
        for source, target in pairs
    ]
    specs = [
        spec
        for spec in specs
        if spec.timestamp - int(args.history_frames) + 1 >= 379
        and spec.timestamp + spec.offset <= 1555
    ]
    return specs[: int(args.max_cases)] if int(args.max_cases) > 0 else specs


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.asarray(points) @ np.asarray(transform)[:3, :3].T + np.asarray(transform)[:3, 3]


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    output = np.eye(4, dtype=np.float64)
    output[:3, :3] = np.asarray(rotation, dtype=np.float64)
    output[:3, 3] = np.asarray(translation, dtype=np.float64)
    return output


def rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first)[:3, :3].T @ np.asarray(second)[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def rotation_distance_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first).T @ np.asarray(second)
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def signed_angle_about_axis(
    source: np.ndarray, target: np.ndarray, axis: np.ndarray
) -> float:
    source = normalize(source - np.dot(source, axis) * axis)
    target = normalize(target - np.dot(target, axis) * axis)
    return float(
        math.atan2(np.dot(axis, np.cross(source, target)), np.dot(source, target))
    )


def yaw_residual(
    coarse_rotation: np.ndarray,
    source_frames: list[np.ndarray],
    target_frames: list[np.ndarray],
    maximum_deg: float,
) -> tuple[np.ndarray, dict]:
    angles = []
    for source, target in zip(source_frames, target_frames):
        target_up = normalize(target[:, 1])
        mapped_heading = coarse_rotation @ normalize(source[:, 2])
        angles.append(
            signed_angle_about_axis(mapped_heading, target[:, 2], target_up)
        )
    if not angles:
        return coarse_rotation.copy(), {"status": "no_valid_torso", "angle_count": 0}
    angle = float(np.median(np.unwrap(np.asarray(angles, dtype=np.float64))))
    bound = math.radians(float(maximum_deg))
    bounded = float(np.clip(angle, -bound, bound))
    axis = normalize(target_frames[0][:, 1])
    correction = Rotation.from_rotvec(bounded * axis).as_matrix()
    corrected = correction @ coarse_rotation
    return corrected.astype(np.float64), {
        "status": "ok",
        "raw_residual_deg": math.degrees(angle),
        "bounded_residual_deg": math.degrees(bounded),
        "clipped": bool(abs(angle) > bound),
        "angle_count": len(angles),
        "angle_median_abs_deviation_deg": float(
            np.degrees(np.median(np.abs(angles - np.median(angles))))
        ),
    }


@lru_cache(maxsize=256)
def load_obj_vertices(path: Path) -> np.ndarray:
    vertices = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                fields = line.split()
                vertices.append((float(fields[1]), float(fields[2]), float(fields[3])))
    output = np.asarray(vertices, dtype=np.float32)
    if output.shape != (10475, 3):
        raise ValueError(f"Unexpected SMPL-X vertices in {path}: {output.shape}")
    return output


def video_path(args: argparse.Namespace, camera: int) -> Path:
    return args.data_root / "three_original_video/three_new" / VIDEO_NAMES[camera]


def frame_cache_path(args: argparse.Namespace, camera: int, frame: int) -> Path:
    return args.output_dir / "input_frames" / f"cam{camera}" / f"{frame:06d}.jpg"


def extract_video_frame(args: argparse.Namespace, camera: int, frame: int) -> Path:
    output = frame_cache_path(args, camera, frame)
    if output.is_file():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path(args, camera)))
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
    ok, image = capture.read()
    capture.release()
    if not ok or image is None:
        raise RuntimeError(f"Cannot read camera {camera} frame {frame}")
    if image.shape[:2] != (2048, 2048):
        raise ValueError(f"Unexpected full-frame shape {image.shape} for camera {camera}")
    if not cv2.imwrite(str(output), image):
        raise OSError(f"Could not write {output}")
    return output


def parameter_root(args: argparse.Namespace, identity: str, frame: int) -> Path:
    return args.data_root / "three/three" / identity / "parameter" / str(frame)


def mesh_path(args: argparse.Namespace, identity: str, frame: int) -> Path:
    return args.data_root / "three/three" / identity / "smplx" / str(frame) / "smplx.obj"


def gt_w2c(args: argparse.Namespace, camera: int, frame: int) -> np.ndarray:
    value = np.load(parameter_root(args, "person0", frame) / f"{camera}_extrinsic.npy")
    output = np.eye(4, dtype=np.float64)
    output[:3] = np.asarray(value, dtype=np.float64)
    return output


def full_intrinsics(args: argparse.Namespace, camera: int) -> np.ndarray:
    payload = json.loads(
        (args.data_root / "three_original_video/calibration_new.json").read_text(
            encoding="utf-8"
        )
    )
    return np.asarray(payload[str(camera)]["K"], dtype=np.float64).reshape(3, 3)


def project_bbox(
    vertices: np.ndarray,
    w2c: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    camera = transform_points(w2c, vertices)
    valid = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 1e-5)
    pixels = camera[valid, :2] / camera[valid, 2:3]
    pixels = pixels @ intrinsics[:2, :2].T + intrinsics[:2, 2]
    pixels[:, 0] *= width / 2048.0
    pixels[:, 1] *= height / 2048.0
    lower = np.maximum(pixels.min(axis=0), (0.0, 0.0))
    upper = np.minimum(pixels.max(axis=0), (width - 1.0, height - 1.0))
    return np.r_[lower, upper].astype(np.float64)


def project_points(
    points: np.ndarray,
    w2c: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    camera = transform_points(w2c, points)
    pixels = np.full((len(camera), 2), np.nan, dtype=np.float64)
    valid = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 1e-5)
    pixels[valid] = camera[valid, :2] / camera[valid, 2:3]
    pixels[valid] = pixels[valid] @ intrinsics[:2, :2].T + intrinsics[:2, 2]
    pixels[:, 0] *= width / 2048.0
    pixels[:, 1] *= height / 2048.0
    return pixels, valid


def mesh_projection_distance(
    predicted_vertices: np.ndarray,
    predicted_w2c: np.ndarray,
    gt_vertices: np.ndarray,
    gt_camera_w2c: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
) -> float:
    if len(predicted_vertices) != len(gt_vertices):
        return float("inf")
    sample = np.arange(0, len(gt_vertices), 20, dtype=np.int64)
    predicted_pixels, predicted_valid = project_points(
        predicted_vertices[sample], predicted_w2c, intrinsics, width, height
    )
    gt_pixels, gt_valid = project_points(
        gt_vertices[sample], gt_camera_w2c, intrinsics, width, height
    )
    valid = predicted_valid & gt_valid
    if int(valid.sum()) < 32:
        return float("inf")
    distance = np.linalg.norm(predicted_pixels[valid] - gt_pixels[valid], axis=1)
    return float(np.median(distance))


def bbox_iou(first: np.ndarray, second: np.ndarray) -> float:
    lower = np.maximum(first[:2], second[:2])
    upper = np.minimum(first[2:], second[2:])
    intersection = float(np.prod(np.maximum(upper - lower, 0.0)))
    area_first = float(np.prod(np.maximum(first[2:] - first[:2], 0.0)))
    area_second = float(np.prod(np.maximum(second[2:] - second[:2], 0.0)))
    return intersection / max(area_first + area_second - intersection, 1e-8)


def layer_humans(prediction: dict, view: dict, debug: dict, layer) -> list[dict]:
    count = int(prediction["smpl_transl"].shape[1])
    if count == 0:
        return []
    device = next(layer.parameters()).device
    rotmat = prediction["smpl_rotmat"][0, :count].to(device=device, dtype=torch.float32)
    rotvec = roma.rotmat_to_rotvec(rotmat)
    shape = prediction["smpl_shape"][0, :count].to(device=device, dtype=torch.float32)
    transl = prediction["smpl_transl"][0, :count].to(device=device, dtype=torch.float32)
    expression = prediction.get("smpl_expression")
    expression = (
        torch.zeros((count, 10), device=device, dtype=torch.float32)
        if expression is None
        else expression[0, :count].to(device=device, dtype=torch.float32)
    )
    intrinsic = view["K_mhmr"][0].to(device=device, dtype=torch.float32).clone()
    height, width = [int(value) for value in tensor_numpy(view["true_shape"])[0]]
    padded_size = int(view["img_mhmr"].shape[-1])
    intrinsic[0, 2] -= 0.5 * (padded_size - width)
    intrinsic[1, 2] -= 0.5 * (padded_size - height)
    intrinsic = intrinsic.unsqueeze(0).expand(count, -1, -1)
    with torch.no_grad():
        body = layer(rotvec, shape, transl, None, None, K=intrinsic, expression=expression)
    joints = body["smpl_j3d"].detach().float().cpu().numpy()
    vertices = body["smpl_v3d"].detach().float().cpu().numpy()
    pixels = body["smpl_v2d"].detach().float().cpu().numpy()
    scores = tensor_numpy(debug.get("head_scores", np.ones(count))).reshape(-1)
    pose = camera_matrix(prediction).astype(np.float64)
    output = []
    for index in range(count):
        finite = np.isfinite(pixels[index]).all(axis=1)
        visible = finite & (vertices[index, :, 2] > 0.0)
        visible &= (pixels[index, :, 0] >= 0.0) & (pixels[index, :, 0] < width)
        visible &= (pixels[index, :, 1] >= 0.0) & (pixels[index, :, 1] < height)
        valid_pixels = pixels[index, finite]
        bbox = np.r_[valid_pixels.min(axis=0), valid_pixels.max(axis=0)]
        bbox[:2] = np.maximum(bbox[:2], (0.0, 0.0))
        bbox[2:] = np.minimum(bbox[2:], (width - 1.0, height - 1.0))
        torso_camera = torso_frame(joints[index])
        output.append(
            {
                "detection_index": index,
                "score": float(scores[index]) if index < len(scores) else 1.0,
                "completeness": float(np.mean(visible)),
                "bbox": bbox.astype(np.float64),
                "bbox_area": float(np.prod(np.maximum(bbox[2:] - bbox[:2], 0.0))),
                "root": transform_points(pose, joints[index, :1])[0],
                "torso": pose[:3, :3] @ torso_camera,
                "root_rotation": pose[:3, :3] @ tensor_numpy(rotmat[index, 0]),
                "joints": transform_points(pose, joints[index]),
                "vertices": transform_points(pose, vertices[index]),
            }
        )
    return output


def assign_gt_identities(
    args: argparse.Namespace,
    humans: list[dict],
    predicted_pose: np.ndarray,
    camera: int,
    frame: int,
    width: int,
    height: int,
) -> tuple[dict[str, dict], dict]:
    meshes = {identity: load_obj_vertices(mesh_path(args, identity, frame)) for identity in IDENTITIES}
    w2c = gt_w2c(args, camera, frame)
    intrinsic = full_intrinsics(args, camera)
    gt_boxes = {
        identity: project_bbox(mesh, w2c, intrinsic, width, height)
        for identity, mesh in meshes.items()
    }
    if not humans:
        return {}, {
            "status": "no_detection",
            "version": "gt_mesh_projection_v2",
            "gt_bboxes": gt_boxes,
        }
    costs = np.zeros((len(humans), len(IDENTITIES)), dtype=np.float64)
    ious = np.zeros_like(costs)
    projection_distances = np.zeros_like(costs)
    diagonal = math.hypot(width, height)
    predicted_w2c = np.linalg.inv(np.asarray(predicted_pose, dtype=np.float64))
    for row, human in enumerate(humans):
        for column, identity in enumerate(IDENTITIES):
            ious[row, column] = bbox_iou(human["bbox"], gt_boxes[identity])
            projection_distances[row, column] = mesh_projection_distance(
                human["vertices"], predicted_w2c, meshes[identity], w2c,
                intrinsic, width, height
            )
            costs[row, column] = (
                projection_distances[row, column] / diagonal
                + 0.05 * (1.0 - ious[row, column])
            )
    rows, columns = linear_sum_assignment(costs)
    feasible_totals = sorted(
        float(sum(costs[row, column] for row, column in enumerate(columns_candidate)))
        for columns_candidate in permutations(range(len(IDENTITIES)), len(humans))
    )
    assigned = {}
    assignment_rows = []
    for row, column in zip(rows, columns):
        identity = IDENTITIES[int(column)]
        human = dict(humans[int(row)])
        human["identity"] = identity
        assigned[identity] = human
        assignment_rows.append(
            {
                "identity": identity,
                "detection_index": int(human["detection_index"]),
                "cost": float(costs[row, column]),
                "bbox_iou": float(ious[row, column]),
                "mesh_projection_median_px": float(
                    projection_distances[row, column]
                ),
            }
        )
    return assigned, {
        "status": "ok",
        "version": "gt_mesh_projection_v2",
        "detection_count": len(humans),
        "assignments": assignment_rows,
        "cost": costs,
        "iou": ious,
        "mesh_projection_median_px": projection_distances,
        "global_assignment_cost": feasible_totals[0],
        "global_assignment_margin": (
            feasible_totals[1] - feasible_totals[0]
            if len(feasible_totals) > 1
            else float("inf")
        ),
        "gt_bboxes": gt_boxes,
    }


def sampled_background_cloud(
    prediction: dict,
    view: dict,
    humans: list[dict],
    count: int,
) -> np.ndarray:
    points = tensor_numpy(prediction["pts3d_in_self_view"]).reshape(-1, 3)
    shape = tuple(int(value) for value in prediction["pts3d_in_self_view"].shape[-3:-1])
    height, width = shape
    confidence = tensor_numpy(prediction.get("conf_self", np.ones((height, width)))).reshape(-1)
    valid = np.isfinite(points).all(axis=1) & np.isfinite(confidence)
    valid &= (points[:, 2] > 0.05) & (points[:, 2] < 50.0)
    true_height, true_width = [int(value) for value in tensor_numpy(view["true_shape"])[0]]
    yy, xx = np.indices((height, width))
    for human in humans:
        bbox = np.asarray(human["bbox"], dtype=np.float64).copy()
        bbox[[0, 2]] *= width / max(true_width, 1)
        bbox[[1, 3]] *= height / max(true_height, 1)
        margin_x = 0.08 * max(bbox[2] - bbox[0], 1.0)
        margin_y = 0.08 * max(bbox[3] - bbox[1], 1.0)
        inside = (
            (xx >= bbox[0] - margin_x)
            & (xx <= bbox[2] + margin_x)
            & (yy >= bbox[1] - margin_y)
            & (yy <= bbox[3] + margin_y)
        )
        valid &= ~inside.reshape(-1)
    cells_y, cells_x = 24, 24
    cell = np.minimum(yy.reshape(-1) * cells_y // height, cells_y - 1) * cells_x
    cell += np.minimum(xx.reshape(-1) * cells_x // width, cells_x - 1)
    selected = []
    for cell_id in range(cells_y * cells_x):
        ids = np.flatnonzero(valid & (cell == cell_id))
        if len(ids):
            selected.append(int(ids[np.argmax(confidence[ids])]))
    selected = np.asarray(selected, dtype=np.int64)
    if len(selected) > count:
        selected = selected[np.argsort(confidence[selected])[-count:]]
    return transform_points(camera_matrix(prediction), points[selected]).astype(np.float32)


def prepare_full_square_input(model, image_paths: list[Path], args: argparse.Namespace):
    from dust3r.utils.geometry import get_camera_parameters
    from src.dust3r.utils.image import load_images, pad_image

    images = load_images(
        [str(path) for path in image_paths],
        size=int(args.size),
        square_ok=True,
    )
    img_res = int(model.mhmr_img_res)
    intrinsic = get_camera_parameters(img_res, device="cpu")
    views = []
    for index, image in enumerate(images):
        view = {
            "img": image["img"],
            "ray_map": torch.full(
                (
                    image["img"].shape[0],
                    6,
                    image["img"].shape[-2],
                    image["img"].shape[-1],
                ),
                torch.nan,
            ),
            "true_shape": torch.from_numpy(image["true_shape"]),
            "idx": index,
            "instance": str(index),
            "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
            "img_mask": torch.tensor(True).unsqueeze(0),
            "ray_mask": torch.tensor(False).unsqueeze(0),
            "update": torch.tensor(True).unsqueeze(0),
            "update_state": torch.tensor(True).unsqueeze(0),
            "update_mem": torch.tensor(True).unsqueeze(0),
            "update_v8_history": torch.tensor(True).unsqueeze(0),
            "reset": torch.tensor(False).unsqueeze(0),
        }
        view["img_mhmr"] = pad_image(view["img"], img_res)
        view["K_mhmr"] = intrinsic
        views.append(view)
    return views


def run_fresh_stream(model, image_paths: list[Path], cut: int, args: argparse.Namespace):
    views = prepare_full_square_input(model, image_paths, args)
    for view in views:
        view["reset"] = torch.zeros_like(view["reset"], dtype=torch.bool)
    routing = {
        "enabled": True,
        "mode": "fresh",
        "cut_indices": [int(cut)],
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


def joint_regressor(layer) -> np.ndarray:
    value = layer.bm_x.J_regressor
    if getattr(value, "is_sparse", False):
        value = value.to_dense()
    return value.detach().float().cpu().numpy().astype(np.float32)


def gt_human_payload(args: argparse.Namespace, frame: int, regressor: np.ndarray) -> dict:
    output = {}
    for identity in IDENTITIES:
        vertices = load_obj_vertices(mesh_path(args, identity, frame))
        joints = regressor @ vertices
        output[identity] = {"vertices": vertices, "joints": joints, "root": joints[0]}
    return output


def build_case_cache(
    args: argparse.Namespace,
    spec: CaseSpec,
    model,
    layer,
    regressor: np.ndarray,
) -> dict:
    pre_frames = list(
        range(spec.timestamp - int(args.history_frames) + 1, spec.timestamp + 1)
    )
    post_frame = spec.timestamp + spec.offset
    image_paths = [extract_video_frame(args, spec.source_camera, frame) for frame in pre_frames]
    image_paths.append(extract_video_frame(args, spec.target_camera, post_frame))
    started = time.perf_counter()
    predictions, views, debug = run_fresh_stream(model, image_paths, len(pre_frames), args)
    all_frames = pre_frames + [post_frame]
    all_cameras = [spec.source_camera] * len(pre_frames) + [spec.target_camera]
    frame_humans, assignment_rows, clouds = [], [], []
    for prediction, view, debug_row, frame, camera in zip(
        predictions, views, debug, all_frames, all_cameras
    ):
        humans = layer_humans(prediction, view, debug_row, layer)
        height, width = [int(value) for value in tensor_numpy(view["true_shape"])[0]]
        assigned, assignment = assign_gt_identities(
            args, humans, camera_matrix(prediction), camera, frame, width, height
        )
        frame_humans.append(assigned)
        assignment_rows.append(assignment)
        clouds.append(sampled_background_cloud(prediction, view, humans, int(args.point_samples)))
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
        "poses": [camera_matrix(prediction).astype(np.float64) for prediction in predictions],
        "humans": frame_humans,
        "assignment": assignment_rows,
        "clouds": clouds,
        "gt": {
            "pre_c2w": np.linalg.inv(gt_w2c(args, spec.source_camera, pre_frames[-1])),
            "post_c2w": np.linalg.inv(gt_w2c(args, spec.target_camera, post_frame)),
            "post_humans": gt_human_payload(args, post_frame, regressor),
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


def reassign_cache_gt_identities(args: argparse.Namespace, cache: dict) -> dict:
    """Rebuild oracle IDs from GT/predicted mesh projection without rerunning Human3R."""
    spec = cache["case"]
    frames = [int(value) for value in spec["pre_frames"]] + [
        int(spec["post_frame"])
    ]
    cameras = [int(spec["source_camera"])] * len(spec["pre_frames"]) + [
        int(spec["target_camera"])
    ]
    reassigned_humans = []
    reassigned_rows = []
    audit_frames = []
    changed_count = 0
    for index, (frame, camera) in enumerate(zip(frames, cameras)):
        old_humans = cache["humans"][index]
        detections = sorted(
            (dict(value) for value in old_humans.values()),
            key=lambda value: int(value["detection_index"]),
        )
        if len(detections) > len(IDENTITIES):
            raise ValueError(
                f"{spec['key']} frame {index} has {len(detections)} detections but "
                f"only {len(IDENTITIES)} GT identities"
            )
        assigned, assignment = assign_gt_identities(
            args,
            detections,
            np.asarray(cache["poses"][index], dtype=np.float64),
            camera,
            frame,
            int(args.size),
            int(args.size),
        )
        old_by_detection = {
            int(value["detection_index"]): identity
            for identity, value in old_humans.items()
        }
        new_by_detection = {
            int(value["detection_index"]): identity
            for identity, value in assigned.items()
        }
        changes = {
            str(detection): {
                "old": old_by_detection.get(detection),
                "new": new_by_detection.get(detection),
            }
            for detection in sorted(set(old_by_detection) | set(new_by_detection))
            if old_by_detection.get(detection) != new_by_detection.get(detection)
        }
        changed_count += len(changes)
        audit_frames.append(
            {
                "frame_index": index,
                "dataset_frame": frame,
                "camera": camera,
                "changed": changes,
                "assignment_margin": assignment.get(
                    "global_assignment_margin", float("nan")
                ),
            }
        )
        reassigned_humans.append(assigned)
        reassigned_rows.append(assignment)
    output = dict(cache)
    output["humans"] = reassigned_humans
    output["assignment"] = reassigned_rows
    output["identity_audit"] = {
        "version": "gt_mesh_projection_v2",
        "changed_detection_assignments": changed_count,
        "changed_any": bool(changed_count),
        "frames": audit_frames,
    }
    return output


def robust_velocity(values: list[np.ndarray], frames: list[int]) -> np.ndarray:
    if len(values) < 2:
        return np.zeros(3, dtype=np.float64)
    rows = []
    for first, second, frame_first, frame_second in zip(
        values[:-1], values[1:], frames[:-1], frames[1:]
    ):
        rows.append((np.asarray(second) - np.asarray(first)) / max(frame_second - frame_first, 1))
    rows = np.stack(rows)
    center = np.median(rows, axis=0)
    distance = np.linalg.norm(rows - center, axis=1)
    threshold = max(0.03, float(np.median(distance) + 2.5 * np.median(np.abs(distance - np.median(distance)))))
    keep = distance <= threshold
    return np.mean(rows[keep], axis=0) if keep.any() else center


def predicted_rotation_frame(
    values: list[np.ndarray], frames: list[int], target_frame: int
) -> tuple[np.ndarray, dict]:
    if len(values) == 1:
        return values[-1], {"angular_speed_deg_per_frame": 0.0, "history": 1}
    velocities = []
    for first, second, frame_first, frame_second in zip(
        values[:-1], values[1:], frames[:-1], frames[1:]
    ):
        relative = np.asarray(second) @ np.asarray(first).T
        velocities.append(
            Rotation.from_matrix(relative).as_rotvec() / max(frame_second - frame_first, 1)
        )
    velocity = np.median(np.stack(velocities), axis=0)
    delta = target_frame - frames[-1]
    predicted = Rotation.from_rotvec(delta * velocity).as_matrix() @ values[-1]
    return predicted, {
        "angular_speed_deg_per_frame": float(np.degrees(np.linalg.norm(velocity))),
        "history": len(values),
        "prediction_delta_frames": int(delta),
    }


def fixed_refine(
    initial: np.ndarray, source_cloud: np.ndarray, target_cloud: np.ndarray
) -> tuple[np.ndarray, dict]:
    if len(source_cloud) < 32 or len(target_cloud) < 32:
        return initial, {"status": "too_few_background_points"}
    return robust_local_pointmap_refinement(
        initial.astype(np.float32),
        source_cloud.astype(np.float32),
        target_cloud.astype(np.float32),
        SimpleNamespace(
            refine_iters=8,
            refine_max_distance=0.60,
            refine_min_distance=0.12,
        ),
    )


def human_candidates(cache: dict) -> dict[str, dict]:
    case = cache["case"]
    pre_frames = [int(value) for value in case["pre_frames"]]
    post_frame = int(case["post_frame"])
    pre_humans = cache["humans"][:-1]
    post_humans = cache["humans"][-1]
    valid_target_clouds = [cloud for cloud in cache["clouds"][:-1] if len(cloud)]
    target_cloud = (
        np.concatenate(valid_target_clouds)
        if valid_target_clouds
        else np.empty((0, 3), dtype=np.float32)
    )
    source_cloud = cache["clouds"][-1]
    shared = [
        identity
        for identity in IDENTITIES
        if identity in post_humans and any(identity in row for row in pre_humans)
    ]
    output = {}
    for identity in shared:
        history = [
            (frame, row[identity])
            for frame, row in zip(pre_frames, pre_humans)
            if identity in row
        ]
        frames = [row[0] for row in history]
        humans = [row[1] for row in history]
        current = post_humans[identity]
        root_velocity = robust_velocity([row["root"] for row in humans], frames)
        anchor = humans[-1]["root"] + (post_frame - frames[-1]) * root_velocity
        target_torso, torso_motion = predicted_rotation_frame(
            [row["torso"] for row in humans], frames, post_frame
        )
        target_root_rotation = Rotation.from_matrix(
            np.stack([row["root_rotation"] for row in humans[-3:]])
        ).mean().as_matrix()
        initial_rotation = target_root_rotation @ current["root_rotation"].T
        initial_translation = anchor - initial_rotation @ current["root"]
        initial = make_transform(initial_rotation, initial_translation)
        fixed, fixed_debug = fixed_refine(initial, source_cloud, target_cloud)
        rotation, v16_debug = yaw_residual(
            fixed[:3, :3], [current["torso"]], [target_torso], 20.0
        )
        translation = anchor - rotation @ current["root"]
        pre_score = float(np.mean([max(row["score"], 1e-6) for row in humans]))
        motion_dispersion = float(
            np.median(
                np.linalg.norm(
                    np.diff(np.stack([row["root"] for row in humans]), axis=0)
                    - root_velocity,
                    axis=1,
                )
            )
        ) if len(humans) >= 2 else 0.0
        quality = math.sqrt(pre_score * max(current["score"], 1e-6))
        quality *= max(current["completeness"], 0.05)
        quality *= math.exp(-min(motion_dispersion / 0.10, 4.0))
        output[identity] = {
            "identity": identity,
            "rotation": rotation.astype(np.float64),
            "translation": translation.astype(np.float64),
            "anchor": anchor.astype(np.float64),
            "post_root": current["root"].astype(np.float64),
            "post_torso": current["torso"].astype(np.float64),
            "target_torso": target_torso.astype(np.float64),
            "quality": float(max(quality, 1e-8)),
            "post_score": float(current["score"]),
            "post_completeness": float(current["completeness"]),
            "post_bbox_area": float(current["bbox_area"]),
            "post_detection_index": int(current["detection_index"]),
            "history_count": len(humans),
            "root_velocity_m_per_frame": root_velocity,
            "motion_dispersion_m": motion_dispersion,
            "fixed_initial": initial,
            "fixed_refined": fixed,
            "fixed_debug": fixed_debug,
            "v16_debug": v16_debug,
            "torso_motion": torso_motion,
        }
    return output


def so3_mean(rotations: list[np.ndarray], weights: np.ndarray | None = None) -> np.ndarray:
    return Rotation.from_matrix(np.stack(rotations)).mean(weights=weights).as_matrix()


def so3_geometric_median(
    rotations: list[np.ndarray], weights: np.ndarray | None = None, iterations: int = 32
) -> np.ndarray:
    if len(rotations) == 1:
        return rotations[0]
    base_weights = np.ones(len(rotations)) if weights is None else np.asarray(weights, dtype=np.float64)
    center = so3_mean(rotations, base_weights)
    for _ in range(iterations):
        tangent = np.stack(
            [Rotation.from_matrix(center.T @ rotation).as_rotvec() for rotation in rotations]
        )
        distance = np.linalg.norm(tangent, axis=1)
        if float(distance.max()) < 1e-9:
            break
        effective = base_weights / np.maximum(distance, 1e-5)
        delta = np.average(tangent, axis=0, weights=effective)
        center = center @ Rotation.from_rotvec(delta).as_matrix()
        if float(np.linalg.norm(delta)) < 1e-8:
            break
    return center


def so3_huber(
    rotations: list[np.ndarray], weights: np.ndarray | None = None, delta_deg: float = 10.0
) -> np.ndarray:
    if len(rotations) == 1:
        return rotations[0]
    base_weights = np.ones(len(rotations)) if weights is None else np.asarray(weights, dtype=np.float64)
    center = so3_mean(rotations, base_weights)
    delta_limit = math.radians(delta_deg)
    for _ in range(24):
        tangent = np.stack(
            [Rotation.from_matrix(center.T @ rotation).as_rotvec() for rotation in rotations]
        )
        distance = np.linalg.norm(tangent, axis=1)
        robust = np.minimum(1.0, delta_limit / np.maximum(distance, 1e-8))
        delta = np.average(tangent, axis=0, weights=base_weights * robust)
        center = center @ Rotation.from_rotvec(delta).as_matrix()
        if float(np.linalg.norm(delta)) < 1e-8:
            break
    return center


def geometric_median(
    points: np.ndarray, weights: np.ndarray | None = None, iterations: int = 64
) -> np.ndarray:
    base_weights = np.ones(len(points)) if weights is None else np.asarray(weights, dtype=np.float64)
    estimate = np.average(points, axis=0, weights=base_weights)
    for _ in range(iterations):
        distance = np.linalg.norm(points - estimate, axis=1)
        if float(distance.min()) < 1e-9:
            return points[int(np.argmin(distance))]
        effective = base_weights / np.maximum(distance, 1e-8)
        updated = np.average(points, axis=0, weights=effective)
        if float(np.linalg.norm(updated - estimate)) < 1e-9:
            break
        estimate = updated
    return estimate


def huber_mean(
    points: np.ndarray, weights: np.ndarray | None = None, delta: float = 0.25
) -> np.ndarray:
    base_weights = np.ones(len(points)) if weights is None else np.asarray(weights, dtype=np.float64)
    estimate = np.average(points, axis=0, weights=base_weights)
    for _ in range(32):
        distance = np.linalg.norm(points - estimate, axis=1)
        robust = np.minimum(1.0, delta / np.maximum(distance, 1e-8))
        updated = np.average(points, axis=0, weights=base_weights * robust)
        if float(np.linalg.norm(updated - estimate)) < 1e-9:
            break
        estimate = updated
    return estimate


def coordinate_trimmed_mean(points: np.ndarray) -> np.ndarray:
    """Drop one coordinate-wise extreme at each side when support permits."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return np.mean(points, axis=0)
    trim = max(1, int(math.floor(0.2 * len(points))))
    if 2 * trim >= len(points):
        trim = (len(points) - 1) // 2
    ordered = np.sort(points, axis=0)
    return np.mean(ordered[trim : len(points) - trim], axis=0)


def translation_candidates(
    candidates: dict[str, dict], identities: tuple[str, ...], rotation: np.ndarray
) -> np.ndarray:
    return np.stack(
        [candidates[identity]["anchor"] - rotation @ candidates[identity]["post_root"] for identity in identities]
    )


def solve_consensus(
    candidates: dict[str, dict], identities: tuple[str, ...], mode: str
) -> tuple[np.ndarray, np.ndarray]:
    rotations = [candidates[identity]["rotation"] for identity in identities]
    quality = np.asarray([candidates[identity]["quality"] for identity in identities])
    quality /= quality.sum()
    if mode == "mean_raw_t":
        rotation = so3_mean(rotations)
        translation = np.mean(
            np.stack([candidates[identity]["translation"] for identity in identities]), axis=0
        )
    elif mode == "mean":
        rotation = so3_mean(rotations)
        translation = np.mean(translation_candidates(candidates, identities, rotation), axis=0)
    elif mode == "weighted":
        rotation = so3_mean(rotations, quality)
        translation = np.average(
            translation_candidates(candidates, identities, rotation), axis=0, weights=quality
        )
    elif mode == "median":
        rotation = so3_geometric_median(rotations)
        translation = np.median(translation_candidates(candidates, identities, rotation), axis=0)
    elif mode == "geomedian":
        rotation = so3_geometric_median(rotations, quality)
        translation = geometric_median(
            translation_candidates(candidates, identities, rotation), quality
        )
    elif mode == "huber":
        rotation = so3_huber(rotations, quality)
        translation = huber_mean(
            translation_candidates(candidates, identities, rotation), quality
        )
    elif mode == "trimmed":
        rotation = so3_geometric_median(rotations)
        translation = coordinate_trimmed_mean(
            translation_candidates(candidates, identities, rotation)
        )
    else:
        raise ValueError(mode)
    return rotation, translation


def per_identity_residuals(
    candidates: dict[str, dict], identities: tuple[str, ...], rotation: np.ndarray, translation: np.ndarray
) -> dict[str, dict]:
    output = {}
    for identity in identities:
        candidate = candidates[identity]
        translation_residual = float(
            np.linalg.norm(candidate["anchor"] - (rotation @ candidate["post_root"] + translation))
        )
        rotation_residual = rotation_distance_deg(
            rotation @ candidate["post_torso"], candidate["target_torso"]
        )
        layout = []
        for other in identities:
            if other == identity:
                continue
            layout.append(
                np.linalg.norm(
                    (candidate["anchor"] - candidates[other]["anchor"])
                    - rotation @ (candidate["post_root"] - candidates[other]["post_root"])
                )
            )
        layout_residual = float(np.mean(layout)) if layout else 0.0
        score = translation_residual / 0.25 + rotation_residual / 10.0 + layout_residual / 0.25
        output[identity] = {
            "translation_m": translation_residual,
            "rotation_deg": rotation_residual,
            "layout_m": layout_residual,
            "normalized_score": float(score),
        }
    return output


def layout_candidate_selection(
    candidates: dict[str, dict], identities: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray, str, dict]:
    rows = []
    for source_identity in identities:
        rotation = candidates[source_identity]["rotation"]
        translations = translation_candidates(candidates, identities, rotation)
        translation = geometric_median(translations)
        residuals = per_identity_residuals(
            candidates, identities, rotation, translation
        )
        layout_values = [residuals[identity]["layout_m"] for identity in identities]
        torso_values = [residuals[identity]["rotation_deg"] for identity in identities]
        translation_values = [
            residuals[identity]["translation_m"] for identity in identities
        ]
        score = (
            float(np.median(layout_values)) / 0.25
            + float(np.median(translation_values)) / 0.25
            + float(np.median(torso_values)) / 10.0
        )
        rows.append(
            {
                "source_identity": source_identity,
                "rotation": rotation,
                "translation": translation,
                "score": score,
                "layout_median_m": float(np.median(layout_values)),
                "translation_median_m": float(np.median(translation_values)),
                "torso_median_deg": float(np.median(torso_values)),
                "residuals": residuals,
            }
        )
    selected = min(rows, key=lambda row: row["score"])
    diagnostics = [
        {
            key: value
            for key, value in row.items()
            if key not in {"rotation", "translation"}
        }
        for row in rows
    ]
    return (
        selected["rotation"],
        selected["translation"],
        selected["source_identity"],
        {"candidate_scores": diagnostics, "selected_score": selected["score"]},
    )


def method_solutions(candidates: dict[str, dict]) -> dict[str, dict]:
    identities = tuple(identity for identity in IDENTITIES if identity in candidates)
    methods = {}
    for identity in identities:
        candidate = candidates[identity]
        methods[f"single_{identity}"] = {
            "rotation": candidate["rotation"],
            "translation": candidate["translation"],
            "identities": (identity,),
        }
    if not identities:
        return methods
    first = min(identities, key=lambda identity: candidates[identity]["post_detection_index"])
    largest = max(identities, key=lambda identity: candidates[identity]["post_bbox_area"])
    confidence = max(identities, key=lambda identity: candidates[identity]["quality"])
    for name, identity in (
        ("single_first", first),
        ("single_largest", largest),
        ("single_highest_confidence", confidence),
    ):
        methods[name] = {
            "rotation": candidates[identity]["rotation"],
            "translation": candidates[identity]["translation"],
            "identities": (identity,),
            "selected_identity": identity,
        }
    if len(identities) >= 2:
        for name, mode in (
            ("naive_mean", "mean_raw_t"),
            ("shared_rotation_mean", "mean"),
            ("confidence_weighted", "weighted"),
            ("rotation_geomedian_translation_median", "median"),
            ("rotation_geomedian_translation_geomedian", "geomedian"),
            ("rotation_geomedian_translation_trimmed", "trimmed"),
            ("robust_huber", "huber"),
        ):
            rotation, translation = solve_consensus(candidates, identities, mode)
            methods[name] = {
                "rotation": rotation,
                "translation": translation,
                "identities": identities,
            }
        layout_rotation, layout_translation, layout_source, layout_debug = (
            layout_candidate_selection(candidates, identities)
        )
        methods["layout_candidate_select"] = {
            "rotation": layout_rotation,
            "translation": layout_translation,
            "identities": identities,
            "selected_identity": layout_source,
            "layout_selection": layout_debug,
        }
        layout_residuals = per_identity_residuals(
            candidates, identities, layout_rotation, layout_translation
        )
        layout_scores = np.asarray(
            [layout_residuals[identity]["normalized_score"] for identity in identities]
        )
        layout_worst = int(np.argmax(layout_scores))
        layout_median = float(np.median(layout_scores))
        layout_reject = (
            len(identities) >= 3
            and layout_scores[layout_worst] > max(1.0, 1.5 * layout_median)
        )
        layout_kept = tuple(
            identity
            for index, identity in enumerate(identities)
            if not (layout_reject and index == layout_worst)
        )
        (
            layout_final_rotation,
            layout_final_translation,
            layout_final_source,
            layout_final_debug,
        ) = layout_candidate_selection(candidates, layout_kept)
        methods["layout_select_one_reject"] = {
            "rotation": layout_final_rotation,
            "translation": layout_final_translation,
            "identities": layout_kept,
            "selected_identity": layout_final_source,
            "rejected_identity": identities[layout_worst] if layout_reject else None,
            "initial_residuals": layout_residuals,
            "reject_triggered": bool(layout_reject),
            "layout_selection": {
                "initial": layout_debug,
                "final": layout_final_debug,
            },
        }
        rotation, translation = solve_consensus(candidates, identities, "huber")
        residuals = per_identity_residuals(candidates, identities, rotation, translation)
        scores = np.asarray([residuals[identity]["normalized_score"] for identity in identities])
        worst_index = int(np.argmax(scores))
        median_score = float(np.median(scores))
        reject = len(identities) >= 3 and scores[worst_index] > max(1.0, 1.5 * median_score)
        kept = tuple(identity for index, identity in enumerate(identities) if not (reject and index == worst_index))
        final_rotation, final_translation = solve_consensus(candidates, kept, "huber")
        methods["layout_aware_one_reject"] = {
            "rotation": final_rotation,
            "translation": final_translation,
            "identities": kept,
            "rejected_identity": identities[worst_index] if reject else None,
            "initial_residuals": residuals,
            "reject_triggered": bool(reject),
        }
        always_kept = tuple(identity for index, identity in enumerate(identities) if index != worst_index)
        always_rotation, always_translation = solve_consensus(candidates, always_kept, "huber")
        methods["diagnostic_always_reject_max"] = {
            "rotation": always_rotation,
            "translation": always_translation,
            "identities": always_kept,
            "rejected_identity": identities[worst_index],
            "initial_residuals": residuals,
        }
    for size in range(1, len(identities) + 1):
        for subset in combinations(identities, size):
            rotation, translation = solve_consensus(candidates, subset, "huber")
            suffix = "-".join(identity[-1] for identity in subset)
            methods[f"subset_huber_n{size}_{suffix}"] = {
                "rotation": rotation,
                "translation": translation,
                "identities": subset,
            }
    return methods


def evaluate_solution(cache: dict, solution: dict) -> dict:
    rotation = np.asarray(solution["rotation"], dtype=np.float64)
    translation = np.asarray(solution["translation"], dtype=np.float64)
    boundary = make_transform(rotation, translation)
    pre_pose = np.asarray(cache["poses"][-2], dtype=np.float64)
    post_pose = np.asarray(cache["poses"][-1], dtype=np.float64)
    gt_pre = np.asarray(cache["gt"]["pre_c2w"], dtype=np.float64)
    gt_post = np.asarray(cache["gt"]["post_c2w"], dtype=np.float64)
    gauge = pre_pose @ np.linalg.inv(gt_pre)
    target_camera = gauge @ gt_post
    final_camera = boundary @ post_pose
    post_humans = cache["humans"][-1]
    root_errors, joint_errors, vertex_errors = [], [], []
    predicted_roots, target_roots = {}, {}
    per_person = {}
    for identity in IDENTITIES:
        if identity not in post_humans:
            continue
        predicted = post_humans[identity]
        target = cache["gt"]["post_humans"][identity]
        final_root = transform_points(boundary, predicted["root"][None])[0]
        final_joints = transform_points(boundary, predicted["joints"])
        final_vertices = transform_points(boundary, predicted["vertices"])
        target_root = transform_points(gauge, target["root"][None])[0]
        target_joints = transform_points(gauge, target["joints"])
        target_vertices = transform_points(gauge, target["vertices"])
        joint_count = min(len(final_joints), len(target_joints))
        vertex_count = min(len(final_vertices), len(target_vertices))
        root_error = float(np.linalg.norm(final_root - target_root))
        joint_error = float(
            np.linalg.norm(final_joints[:joint_count] - target_joints[:joint_count], axis=1).mean()
        )
        vertex_error = float(
            np.linalg.norm(final_vertices[:vertex_count] - target_vertices[:vertex_count], axis=1).mean()
        )
        root_errors.append(root_error)
        joint_errors.append(joint_error)
        vertex_errors.append(vertex_error)
        predicted_roots[identity] = final_root
        target_roots[identity] = target_root
        per_person[identity] = {
            "root_error_m": root_error,
            "joint_error_m": joint_error,
            "vertex_error_m": vertex_error,
        }
    pair_distance, pair_vector = [], []
    for first, second in combinations(sorted(predicted_roots), 2):
        predicted_vector = predicted_roots[first] - predicted_roots[second]
        target_vector = target_roots[first] - target_roots[second]
        pair_distance.append(abs(np.linalg.norm(predicted_vector) - np.linalg.norm(target_vector)))
        pair_vector.append(np.linalg.norm(predicted_vector - target_vector))
    camera_translation = float(np.linalg.norm(final_camera[:3, 3] - target_camera[:3, 3]))
    camera_rotation = rotation_error_deg(final_camera, target_camera)
    return {
        "boundary": boundary,
        "selected_identities": solution["identities"],
        "selected_identity": solution.get("selected_identity"),
        "rejected_identity": solution.get("rejected_identity"),
        "reject_triggered": solution.get("reject_triggered", False),
        "initial_residuals": solution.get("initial_residuals"),
        "layout_selection": solution.get("layout_selection"),
        "camera_translation_error_m": camera_translation,
        "camera_rotation_error_deg": camera_rotation,
        "camera_composite": camera_translation + 0.02 * camera_rotation,
        "catastrophic": bool(camera_translation > 2.0 or camera_rotation > 45.0),
        "human_root_error_m": float(np.mean(root_errors)) if root_errors else float("nan"),
        "human_joint_error_m": float(np.mean(joint_errors)) if joint_errors else float("nan"),
        "human_vertex_error_m": float(np.mean(vertex_errors)) if vertex_errors else float("nan"),
        "pairwise_distance_error_m": float(np.mean(pair_distance)) if pair_distance else float("nan"),
        "pairwise_vector_error_m": float(np.mean(pair_vector)) if pair_vector else float("nan"),
        "per_person": per_person,
    }


def evaluate_case(cache: dict) -> dict:
    candidates = human_candidates(cache)
    solutions = method_solutions(candidates)
    methods = {name: evaluate_solution(cache, solution) for name, solution in solutions.items()}
    single_names = [f"single_{identity}" for identity in IDENTITIES if f"single_{identity}" in methods]
    if single_names:
        best = min(single_names, key=lambda name: methods[name]["camera_composite"])
        methods["oracle_best_single"] = {**methods[best], "oracle_source": best}
    rotations = [row["rotation"] for row in candidates.values()]
    translations = [row["translation"] for row in candidates.values()]
    rotation_dispersion = [rotation_distance_deg(a, b) for a, b in combinations(rotations, 2)]
    translation_dispersion = [np.linalg.norm(a - b) for a, b in combinations(translations, 2)]
    worst_single = None
    if single_names:
        worst_single = max(single_names, key=lambda name: methods[name]["camera_composite"]).removeprefix("single_")
    rejected = methods.get("layout_select_one_reject", {}).get("rejected_identity")
    return {
        "case": cache["case"],
        "shared_identities": tuple(candidates),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "candidate_dispersion": {
            "translation_pairwise_mean_m": float(np.mean(translation_dispersion)) if translation_dispersion else 0.0,
            "translation_pairwise_max_m": float(np.max(translation_dispersion)) if translation_dispersion else 0.0,
            "rotation_pairwise_mean_deg": float(np.mean(rotation_dispersion)) if rotation_dispersion else 0.0,
            "rotation_pairwise_max_deg": float(np.max(rotation_dispersion)) if rotation_dispersion else 0.0,
        },
        "methods": methods,
        "worst_single_identity": worst_single,
        "layout_rejected_worst_single": bool(rejected is not None and rejected == worst_single),
        "assignment": cache["assignment"],
        "identity_audit": cache.get("identity_audit"),
        "runtime_seconds": cache["runtime_seconds"],
    }


def finite_distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {key: float("nan") for key in ("mean", "median", "p90", "p95")}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
    }


def method_summary(cases: list[dict], method: str) -> dict:
    rows = [case["methods"][method] for case in cases if method in case["methods"]]
    metrics = (
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
        "valid_cases": len(rows),
        **{metric: finite_distribution([row[metric] for row in rows]) for metric in metrics},
        "catastrophic_rate": float(np.mean([row["catastrophic"] for row in rows])) if rows else float("nan"),
    }


def paired_comparison(cases: list[dict], first: str, second: str) -> dict:
    rows = [case for case in cases if first in case["methods"] and second in case["methods"]]
    output = {"valid_cases": len(rows)}
    for metric in ("camera_translation_error_m", "camera_rotation_error_deg", "camera_composite"):
        first_values = np.asarray([row["methods"][first][metric] for row in rows])
        second_values = np.asarray([row["methods"][second][metric] for row in rows])
        delta = second_values - first_values
        try:
            p_value = float(wilcoxon(second_values, first_values).pvalue) if len(rows) >= 2 else float("nan")
        except ValueError:
            p_value = 1.0
        output[metric] = {
            "first_mean": float(first_values.mean()) if len(rows) else float("nan"),
            "second_mean": float(second_values.mean()) if len(rows) else float("nan"),
            "second_minus_first_mean": float(delta.mean()) if len(rows) else float("nan"),
            "second_improvement_rate": float(np.mean(delta < 0.0)) if len(rows) else float("nan"),
            "second_harmful_rate": float(np.mean(delta > 0.0)) if len(rows) else float("nan"),
            "wilcoxon_p": p_value,
        }
    return output


def aggregate(cases: list[dict]) -> dict:
    method_names = sorted({name for case in cases for name in case["methods"] if not name.startswith("subset_")})
    summaries = {name: method_summary(cases, name) for name in method_names}
    by_offset = {}
    for offset in sorted({int(case["case"]["offset"]) for case in cases}):
        subset = [case for case in cases if int(case["case"]["offset"]) == offset]
        by_offset[str(offset)] = {name: method_summary(subset, name) for name in method_names}
    subset_rows = {1: [], 2: [], 3: []}
    for case in cases:
        for name, row in case["methods"].items():
            if name.startswith("subset_huber_n"):
                size = int(name.split("_n", 1)[1].split("_", 1)[0])
                subset_rows[size].append(row)
    number_ablation = {
        str(size): {
            "evaluations": len(rows),
            "camera_translation_error_m": finite_distribution([row["camera_translation_error_m"] for row in rows]),
            "camera_rotation_error_deg": finite_distribution([row["camera_rotation_error_deg"] for row in rows]),
            "camera_composite": finite_distribution([row["camera_composite"] for row in rows]),
            "catastrophic_rate": float(np.mean([row["catastrophic"] for row in rows])) if rows else float("nan"),
        }
        for size, rows in subset_rows.items()
    }
    common_three_cases = [case for case in cases if case["candidate_count"] == 3]
    common_three_rows = {1: [], 2: [], 3: []}
    for case in common_three_cases:
        for name, row in case["methods"].items():
            if name.startswith("subset_huber_n"):
                size = int(name.split("_n", 1)[1].split("_", 1)[0])
                common_three_rows[size].append(row)
    number_ablation_common_support = {
        str(size): {
            "case_count": len(common_three_cases),
            "subset_evaluations": len(rows),
            "camera_translation_error_m": finite_distribution(
                [row["camera_translation_error_m"] for row in rows]
            ),
            "camera_rotation_error_deg": finite_distribution(
                [row["camera_rotation_error_deg"] for row in rows]
            ),
            "camera_composite": finite_distribution(
                [row["camera_composite"] for row in rows]
            ),
            "catastrophic_rate": float(np.mean([row["catastrophic"] for row in rows]))
            if rows
            else float("nan"),
        }
        for size, rows in common_three_rows.items()
    }
    leave_one_out = {}
    for removed in range(3):
        kept = "-".join(str(index) for index in range(3) if index != removed)
        name = f"subset_huber_n2_{kept}"
        rows = [case["methods"][name] for case in common_three_cases]
        leave_one_out[f"minus_person{removed}"] = {
            "valid_cases": len(rows),
            "camera_translation_error_m": finite_distribution(
                [row["camera_translation_error_m"] for row in rows]
            ),
            "camera_rotation_error_deg": finite_distribution(
                [row["camera_rotation_error_deg"] for row in rows]
            ),
            "camera_composite": finite_distribution(
                [row["camera_composite"] for row in rows]
            ),
            "catastrophic_rate": float(np.mean([row["catastrophic"] for row in rows]))
            if rows
            else float("nan"),
        }
    eligible = [case for case in cases if case["candidate_count"] >= 2]
    reject_rows = [
        case
        for case in eligible
        if case["methods"].get("layout_select_one_reject", {}).get("reject_triggered")
    ]
    layout_diagnostics = {
        "eligible_cases": len(eligible),
        "reject_trigger_count": len(reject_rows),
        "rejected_worst_single_rate": float(
            np.mean([case["layout_rejected_worst_single"] for case in reject_rows])
        ) if reject_rows else float("nan"),
        "candidate_translation_dispersion_m": finite_distribution(
            [case["candidate_dispersion"]["translation_pairwise_mean_m"] for case in eligible]
        ),
        "candidate_rotation_dispersion_deg": finite_distribution(
            [case["candidate_dispersion"]["rotation_pairwise_mean_deg"] for case in eligible]
        ),
    }
    selector_diagnostics = {}
    for name in (
        "single_first",
        "single_largest",
        "single_highest_confidence",
        "layout_candidate_select",
        "layout_select_one_reject",
    ):
        selector_rows = [case for case in eligible if name in case["methods"]]
        hits = [
            case["methods"][name].get("selected_identity")
            == case["methods"]["oracle_best_single"]["oracle_source"].removeprefix(
                "single_"
            )
            for case in selector_rows
        ]
        selector_diagnostics[name] = {
            "valid_cases": len(selector_rows),
            "oracle_best_identity_hit_rate": float(np.mean(hits))
            if hits
            else float("nan"),
        }
    by_camera_pair = {}
    for source, target in sorted(
        {
            (int(case["case"]["source_camera"]), int(case["case"]["target_camera"]))
            for case in cases
        }
    ):
        subset = [
            case
            for case in cases
            if int(case["case"]["source_camera"]) == source
            and int(case["case"]["target_camera"]) == target
        ]
        by_camera_pair[f"{source}->{target}"] = {
            name: method_summary(subset, name)
            for name in (
                "single_first",
                "single_highest_confidence",
                "oracle_best_single",
                "naive_mean",
                "robust_huber",
                "layout_select_one_reject",
            )
            if any(name in case["methods"] for case in subset)
        }
    assignment_ious = [
        float(row["bbox_iou"])
        for case in cases
        for row in case["assignment"][-1].get("assignments", [])
    ]
    assignment_projection = [
        float(row["mesh_projection_median_px"])
        for case in cases
        for frame in case["assignment"]
        for row in frame.get("assignments", [])
        if "mesh_projection_median_px" in row
    ]
    assignment_margins = [
        float(frame["global_assignment_margin"])
        for case in cases
        for frame in case["assignment"]
        if np.isfinite(frame.get("global_assignment_margin", float("nan")))
    ]
    changed_assignments = [
        int((case.get("identity_audit") or {}).get("changed_detection_assignments", 0))
        for case in cases
    ]
    assignment_diagnostics = {
        "version": "gt_mesh_projection_v2",
        "post_assignment_count": len(assignment_ious),
        "bbox_iou": finite_distribution(assignment_ious),
        "bbox_iou_below_0_1_rate": float(np.mean(np.asarray(assignment_ious) < 0.1))
        if assignment_ious
        else float("nan"),
        "bbox_iou_below_0_2_rate": float(np.mean(np.asarray(assignment_ious) < 0.2))
        if assignment_ious
        else float("nan"),
        "all_frame_mesh_projection_median_px": finite_distribution(
            assignment_projection
        ),
        "all_frame_global_assignment_margin": finite_distribution(
            assignment_margins
        ),
        "cases_changed_from_legacy_bbox_assignment": int(
            np.sum(np.asarray(changed_assignments) > 0)
        ),
        "changed_detection_assignments": int(np.sum(changed_assignments)),
    }
    catastrophic_cases = []
    for case in cases:
        primary_row = case["methods"].get("layout_select_one_reject")
        if primary_row is None or not primary_row["catastrophic"]:
            continue
        oracle_row = case["methods"]["oracle_best_single"]
        catastrophic_cases.append(
            {
                "key": case["case"]["key"],
                "candidate_count": case["candidate_count"],
                "selected_identity": primary_row.get("selected_identity"),
                "rejected_identity": primary_row.get("rejected_identity"),
                "primary_translation_m": primary_row["camera_translation_error_m"],
                "primary_rotation_deg": primary_row["camera_rotation_error_deg"],
                "oracle_source": oracle_row["oracle_source"],
                "oracle_translation_m": oracle_row["camera_translation_error_m"],
                "oracle_rotation_deg": oracle_row["camera_rotation_error_deg"],
                "rotation_dispersion_deg": case["candidate_dispersion"][
                    "rotation_pairwise_mean_deg"
                ],
            }
        )
    catastrophic_cases.sort(key=lambda row: row["primary_rotation_deg"], reverse=True)
    comparisons = {
        "oracle_best_single_to_primary": paired_comparison(
            cases, "oracle_best_single", "layout_select_one_reject"
        ),
        "naive_mean_to_primary": paired_comparison(
            cases, "naive_mean", "layout_select_one_reject"
        ),
        "single_first_to_primary": paired_comparison(
            cases, "single_first", "layout_select_one_reject"
        ),
        "naive_mean_to_huber": paired_comparison(cases, "naive_mean", "robust_huber"),
    }
    oracle = summaries.get("oracle_best_single", {})
    primary = summaries.get("layout_select_one_reject", {})
    naive = summaries.get("naive_mean", {})
    geometry_gate = False
    robust_gate = False
    if oracle and primary:
        geometry_gate = (
            primary["camera_composite"]["mean"] < oracle["camera_composite"]["mean"]
            and primary["camera_composite"]["p90"] < oracle["camera_composite"]["p90"]
        )
    if naive and primary:
        robust_gate = (
            primary["camera_composite"]["mean"] < naive["camera_composite"]["mean"]
            and primary["camera_composite"]["p90"] < naive["camera_composite"]["p90"]
        )
    return {
        "case_count": len(cases),
        "candidate_count_distribution": finite_distribution([case["candidate_count"] for case in cases]),
        "methods": summaries,
        "by_offset": by_offset,
        "number_ablation": number_ablation,
        "number_ablation_common_three_person_support": number_ablation_common_support,
        "leave_one_out_common_three_person_support": leave_one_out,
        "layout_diagnostics": layout_diagnostics,
        "selector_diagnostics": selector_diagnostics,
        "assignment_diagnostics": assignment_diagnostics,
        "by_camera_pair": by_camera_pair,
        "catastrophic_primary_cases": catastrophic_cases,
        "paired_comparisons": comparisons,
        "decision": {
            "multi_human_exceeds_oracle_best_single_mean_and_p90": bool(geometry_gate),
            "robust_primary_exceeds_naive_mean_mean_and_p90": bool(robust_gate),
            "phase1_geometry_gate_pass": bool(geometry_gate and robust_gate),
            "rule": "Primary must beat Oracle Best Single and Naive Mean on camera composite mean and P90.",
        },
    }


def fmt(value: float) -> str:
    return "nan" if not np.isfinite(value) else f"{value:.3f}"


def markdown_report(report: dict) -> str:
    aggregate_report = report["aggregate"]
    methods = aggregate_report["methods"]
    order = (
        "single_first",
        "single_largest",
        "single_highest_confidence",
        "oracle_best_single",
        "naive_mean",
        "confidence_weighted",
        "rotation_geomedian_translation_geomedian",
        "rotation_geomedian_translation_median",
        "rotation_geomedian_translation_trimmed",
        "shared_rotation_mean",
        "robust_huber",
        "layout_candidate_select",
        "layout_select_one_reject",
    )
    lines = [
        "# V20 Phase 1 GT-ID Multi-Human Consensus Alignment v2",
        "",
        "## 实验协议",
        "",
        f"- 数据：MultiHuman Real-World-Capture `three`，{aggregate_report['case_count']} 个 cuts。",
        "- 输入：完整 2048x2048 画面仅缩放到 512x512，不裁剪。",
        "- 开启：Human3R fresh hard reset、Fixed Explicit coarse、V16 torso 20 deg、显式 root translation。",
        "- 关闭：DA3、Keypoint R-CNN、V11.4 scale、VGGT、额外 scene refinement、continuity memory。",
        "- GT identity：predicted/GT 同拓扑 SMPL-X 逐顶点 2D 投影距离 + Hungarian；不再使用 bbox-only association。",
        "- GT SMPL-X/camera 只用于 Oracle identity association 和 evaluator，不进入 Boundary 数值求解。",
        "",
        "## 主要结果",
        "",
        "| Method | N | Camera T mean/med/P90 | Camera R mean/med/P90 | Root | Joints | Vertices | Catastrophic |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in order:
        if name not in methods:
            continue
        row = methods[name]
        translation = row["camera_translation_error_m"]
        rotation = row["camera_rotation_error_deg"]
        lines.append(
            f"| {name} | {row['valid_cases']} | "
            f"{fmt(translation['mean'])}/{fmt(translation['median'])}/{fmt(translation['p90'])} | "
            f"{fmt(rotation['mean'])}/{fmt(rotation['median'])}/{fmt(rotation['p90'])} | "
            f"{fmt(row['human_root_error_m']['mean'])} | "
            f"{fmt(row['human_joint_error_m']['mean'])} | "
            f"{fmt(row['human_vertex_error_m']['mean'])} | "
            f"{100.0 * row['catastrophic_rate']:.1f}% |"
        )
    decision = aggregate_report["decision"]
    layout = aggregate_report["layout_diagnostics"]
    assignment = aggregate_report["assignment_diagnostics"]
    lines.extend(
        [
            "",
            "## 人数消融",
            "",
            "| Humans | Evaluations | Camera T mean/P90 | Camera R mean/P90 | Composite mean/P90 | Catastrophic |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for size, row in aggregate_report[
        "number_ablation_common_three_person_support"
    ].items():
        lines.append(
            f"| {size} | {row['subset_evaluations']} | "
            f"{fmt(row['camera_translation_error_m']['mean'])}/{fmt(row['camera_translation_error_m']['p90'])} | "
            f"{fmt(row['camera_rotation_error_deg']['mean'])}/{fmt(row['camera_rotation_error_deg']['p90'])} | "
            f"{fmt(row['camera_composite']['mean'])}/{fmt(row['camera_composite']['p90'])} | "
            f"{100.0 * row['catastrophic_rate']:.1f}% |"
        )
    lines.extend(
        [
            "",
            f"以上三行严格使用同一批 {len([case for case in report['cases'] if case['candidate_count'] == 3])} 个三人均可用 cuts；Evaluations 包含该人数的全部组合。",
            "",
            "## Selector 命中 Oracle Best Single",
            "",
            "| Selector | Valid | Hit rate |",
            "|---|---:|---:|",
        ]
    )
    for name, row in aggregate_report["selector_diagnostics"].items():
        lines.append(
            f"| {name} | {row['valid_cases']} | "
            f"{100.0 * row['oracle_best_identity_hit_rate']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Layout 与异常人物",
            "",
            f"- 可做多人共识的 cases：{layout['eligible_cases']}。",
            f"- 固定规则触发一次 reject：{layout['reject_trigger_count']}。",
            f"- 被剔除者恰为 GT-evaluated worst single 的比例：{fmt(layout['rejected_worst_single_rate'])}。",
            f"- translation candidate 平均 dispersion：{fmt(layout['candidate_translation_dispersion_m']['mean'])} m。",
            f"- rotation candidate 平均 dispersion：{fmt(layout['candidate_rotation_dispersion_deg']['mean'])} deg。",
            "",
            "## GT-ID 审计",
            "",
            f"- Assignment version：`{assignment['version']}`。",
            f"- 相比旧 bbox-only assignment，发生变化的 cases：{assignment['cases_changed_from_legacy_bbox_assignment']}。",
            f"- 被改正的 detection identity 数：{assignment['changed_detection_assignments']}。",
            f"- assigned mesh projection median：{fmt(assignment['all_frame_mesh_projection_median_px']['median'])} px。",
            f"- global assignment margin median：{fmt(assignment['all_frame_global_assignment_margin']['median'])}。",
            "",
            "## 决策",
            "",
            f"- 多人 primary 在 composite mean 和 P90 同时超过 Oracle Best Single：**{decision['multi_human_exceeds_oracle_best_single_mean_and_p90']}**。",
            f"- Robust primary 在 composite mean 和 P90 同时超过 Naive Mean：**{decision['robust_primary_exceeds_naive_mean_mean_and_p90']}**。",
            f"- Phase 1 geometry gate：**{'PASS' if decision['phase1_geometry_gate_pass'] else 'FAIL'}**。",
            "",
            "该 gate 只判断多人几何是否值得继续，不评价 token Re-ID。详细逐 cut 结果见同目录 JSON。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    specs = case_specs(args)
    if not specs:
        raise ValueError("No valid cases")
    cache_dir = args.output_dir / "case_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    missing = [
        spec
        for spec in specs
        if args.overwrite_cache or not (cache_dir / f"{spec.key}.pt").is_file()
    ]
    model = layer = regressor = None
    if missing:
        if args.evaluation_only:
            raise FileNotFoundError(f"Missing {len(missing)} case caches")
        torch.cuda.set_device(torch.device(args.device))
        model = build_model(args)
        _, layer = build_smpl_models(model, torch.device(args.device))
        regressor = joint_regressor(layer)
    cases = []
    started = time.perf_counter()
    for index, spec in enumerate(specs):
        cache_path = cache_dir / f"{spec.key}.pt"
        if cache_path.is_file() and not args.overwrite_cache:
            cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        else:
            print(f">> [{index + 1}/{len(specs)}] infer {spec.key}", flush=True)
            cache = build_case_cache(args, spec, model, layer, regressor)
            torch.save(cache, cache_path)
        cache = reassign_cache_gt_identities(args, cache)
        case = evaluate_case(cache)
        cases.append(case)
        print(
            f">> {spec.key}: matched={case['candidate_count']} "
            f"primary={case['methods'].get('layout_select_one_reject', {}).get('camera_composite', float('nan')):.3f}",
            flush=True,
        )
    aggregate_report = aggregate(cases)
    report = {
        "experiment": "Movie3R-Multi V13 GT-ID Consensus",
        "legacy_experiment": "V20 Phase 1 GT-ID Multi-Human Consensus Alignment v2",
        "scope": "Lite geometry feasibility; no Re-ID and no external depth/rotation/scale model",
        "candidate_gt_usage": {
            "gt_identity_for_oracle_association": True,
            "gt_smplx_projection_for_oracle_association": True,
            "gt_camera_for_candidate": False,
            "gt_smplx_geometry_for_candidate": False,
            "gt_camera_and_smplx_for_evaluation": True,
        },
        "frozen_components": {
            "human3r": str(args.model_path.resolve()),
            "fixed_explicit_pointmap_refinement": "8 iterations, 0.60m to 0.12m correspondence bound",
            "v16_torso_bound_deg": 20.0,
            "outlier_passes": 1,
            "vggt": False,
            "da3": False,
            "keypoint_rcnn": False,
            "v11_4_scale": False,
        },
        "cases": cases,
        "aggregate": aggregate_report,
        "wall_seconds": time.perf_counter() - started,
    }
    suffix = "_".join(str(value) for value in sorted(set(args.offsets)))
    json_path = args.output_dir / f"v13_gtid_offsets_{suffix}.json"
    markdown_path = args.output_dir / f"v13_gtid_offsets_{suffix}.md"
    json_path.write_text(
        json.dumps(jsonable(report), indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(f">> JSON: {json_path}", flush=True)
    print(f">> report: {markdown_path}", flush=True)
    print(f">> decision: {aggregate_report['decision']}", flush=True)


if __name__ == "__main__":
    main()
