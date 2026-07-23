#!/usr/bin/env python3
"""Run a true recurrent A/B multi-cut audit for the frozen V14.5 methods.

Human3R receives one chronological A/B/A/B stream.  A fresh local state is
created before every post-cut decode, while every explicit world transform is
chained from the previous prediction.  Ground truth is read only after all
candidate scales and boundaries have been generated.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from torchvision.models.detection import (
    KeypointRCNN_ResNet50_FPN_Weights,
    keypointrcnn_resnet50_fpn,
)


ROOT = Path(__file__).resolve().parents[1]
for path in (
    ROOT,
    ROOT / "src",
    ROOT / "scripts",
    ROOT / "scripts/archive_v2_v6",
    ROOT / "scripts/archive_v7",
    ROOT / "archive/20260721/scripts",
    ROOT / "output/v14_5_final_audit/tmp/frozen_pyc_modules",
    ROOT.parent / "Movie3R-dataset/Depth-Anything-3/src",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from torch_cache_support import configure_torch_cache  # noqa: E402

configure_torch_cache()

import dust3r  # noqa: E402

_ARCHIVED_DUST3R = ROOT / "archive/20260721/src/dust3r"
if str(_ARCHIVED_DUST3R) not in dust3r.__path__:
    dust3r.__path__.append(str(_ARCHIVED_DUST3R))

from demo import prepare_output  # noqa: E402
from dust3r.inference import inference_recurrent_lighter  # noqa: E402
from dust3r.utils.device import to_cpu  # noqa: E402
from scripts.boundary_human3r_reset_support import gt_human, predicted_human  # noqa: E402
from scripts.v10_1_fixed_explicit_candidate_probe import (  # noqa: E402
    FIXED_EXPLICIT_NAME,
    combine_clouds,
    human_initial,
    refine_candidate,
)
from scripts.v14_1_multicut_state_routing_rollout import (  # noqa: E402
    load_rollout_views,
)
from scripts.v14_1_shot_aware_state_routing_probe import (  # noqa: E402
    DEFAULT_RECORDS,
    build_model,
    build_smpl_models,
    read_jsonl,
    select_records,
)
from scripts.v9_learned_stream_alignment_overfit import gt_pose_from_view  # noqa: E402
from v10_implicit_explicit_cross_shot_probe import (  # noqa: E402
    background_cloud,
    history_background_cloud,
)
from v10_latent_activation_patching_probe import camera_matrix  # noqa: E402
from v15_wide_baseline_boundary_bridge_candidates import (  # noqa: E402
    aggregate_coarse,
    build_vggt,
    pair_specs,
    predicted_human_summary,
    run_vggt_pairs,
    square_view,
    view_rgb,
)
from v16_human_torso_candidates import predict_torso_frames, yaw_residual  # noqa: E402
from v18_cache_2d_keypoints import select_person  # noqa: E402
from v18_da3_metric_depth_probe import (  # noqa: E402
    DepthAnything3,
    estimate_frame_roots,
    metric_inference,
)
from v21_absolute_shot_background_scale_probe import (  # noqa: E402
    bounded_scene_scale,
    frame_calibration,
)
from v32_consensus_texture_safety_audit import selected_rotation  # noqa: E402


METHODS = ("continue", "hard_reset_fixed", "v11_1", "v11_4", "unified_da3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ROOT / "output/v14_5_final_audit/true_recurrent_multicut",
    )
    parser.add_argument("--data_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data"))
    parser.add_argument("--model_path", type=Path, default=ROOT / "src/human3r_896L.pth")
    parser.add_argument(
        "--da3_model_path",
        type=Path,
        default=ROOT.parent / "Movie3R-dataset/Depth-Anything-3/checkpoints/DA3Metric-Large",
    )
    parser.add_argument("--vggt_root", type=Path, default=Path("/data/wangzheng/Movie3R/vggt"))
    parser.add_argument(
        "--vggt_weights",
        type=Path,
        default=Path("/data/wangzheng/Movie3R/vggt/vggt_weights/model.pt"),
    )
    parser.add_argument(
        "--enable_vggt",
        action="store_true",
        help="Opt in to Conditional VGGT rotation-tail rescue (disabled by default).",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolution", type=int, nargs=2, default=(512, 288))
    parser.add_argument("--resize_mode", default="human3r_demo")
    parser.add_argument("--cases_per_source", type=int, default=1)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--max_cuts", type=int, default=8)
    parser.add_argument("--frames_per_shot", type=int, default=2)
    parser.add_argument("--process_res", type=int, default=504)
    parser.add_argument("--point_samples", type=int, default=5000)
    parser.add_argument("--scene_samples", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def scale_pose(pose: np.ndarray, scale: float) -> np.ndarray:
    output = np.asarray(pose, dtype=np.float64).copy()
    output[:3, 3] *= float(scale)
    return output


def transform_points(pose: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.asarray(points) @ np.asarray(pose)[:3, :3].T + np.asarray(pose)[:3, 3]


def rotation_error(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first)[:3, :3].T @ np.asarray(second)[:3, :3]
    return float(
        np.degrees(np.arccos(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)))
    )


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


def save_predictions(outputs: dict, model, output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cpu = to_cpu(outputs)
    payload = {"pred": cpu["pred"], "views": [dict(value) for value in cpu["views"]]}
    for view in payload["views"]:
        view["reset"] = torch.zeros_like(view["reset"], dtype=torch.bool)
    prepare_output(
        payload,
        str(output_dir),
        revisit=1,
        use_pose=True,
        save=True,
        render=False,
        render_video=False,
        img_res=getattr(model, "mhmr_img_res", None),
        subsample=1,
    )


def fixed_boundary(local_dir: Path, cut: int, shot_start: int, args, seed: int) -> np.ndarray:
    history = list(range(shot_start, cut))
    initial = human_initial(local_dir, history, cut, mode="mean")
    target, target_debug = history_background_cloud(
        local_dir, history, int(args.point_samples)
    )
    source, source_debug = combine_clouds(
        local_dir, [cut], int(args.point_samples), seed
    )
    result = refine_candidate(
        FIXED_EXPLICIT_NAME,
        initial,
        source,
        target,
        "standard",
        {"source": source_debug, "target": target_debug},
    )
    return np.asarray(result.transform, dtype=np.float64)


def raw_frame(local_dir: Path, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth = np.load(local_dir / "depth" / f"{index:06d}.npy").astype(np.float32)
    confidence = np.load(local_dir / "conf" / f"{index:06d}.npy").astype(np.float32)
    with np.load(local_dir / "smpl" / f"{index:06d}.npz", allow_pickle=True) as payload:
        mask = np.asarray(payload["msk"][0] > 0.10, dtype=np.uint8)
    return depth, confidence, mask


def detect_person(model, image: np.ndarray, device: torch.device):
    tensor = (
        torch.from_numpy(np.asarray(image).copy())
        .permute(2, 0, 1)
        .float()
        .div(255.0)
        .to(device)
    )
    with torch.no_grad():
        output = model([tensor])[0]
    return select_person(output, 0.50)


def shot_metric_scale(
    index: int,
    prediction: dict,
    view: dict,
    human: dict | None,
    local_dir: Path,
    keypoint_model,
    da3_model,
    device: torch.device,
    args,
) -> dict:
    image = view_rgb(view)
    keypoints, confidence_2d, _, detection_score = detect_person(
        keypoint_model, image, device
    )
    intrinsics = view["camera_intrinsics"][0].detach().float().cpu().numpy().astype(np.float32)
    depth, processed_intrinsics, elapsed = metric_inference(
        da3_model, [image], intrinsics[None], int(args.process_res)
    )
    metric_depth = depth[0]
    raw_depth, raw_confidence, human_mask = raw_frame(local_dir, index)
    root_scale = 1.0
    calibrated_root = None
    if human is not None:
        _, calibrated_root, _ = estimate_frame_roots(
            metric_depth,
            intrinsics,
            processed_intrinsics[0],
            keypoints,
            confidence_2d,
            human["joints"],
            0.30,
            3,
        )
        raw_z = max(float(human["root"][2]), 1e-4)
        if np.isfinite(calibrated_root).all():
            root_scale = float(np.clip(float(calibrated_root[2]) / raw_z, 0.35, 3.0))
        else:
            calibrated_root = None
    kernel = np.ones((11, 11), dtype=np.uint8)
    calibration = frame_calibration(
        raw_depth,
        metric_depth,
        raw_confidence,
        cv2.dilate(human_mask, kernel, iterations=1),
        root_scale,
        SimpleNamespace(
            raw_confidence_threshold=1.0,
            min_background_pixels=512,
            lowfreq_sigma=25.0,
        ),
    )
    median_ratio = float(calibration["scales"]["median_ratio"])
    bounded = bounded_scene_scale(median_ratio, root_scale, 0.15)
    scene_scale = bounded if median_ratio / max(root_scale, 1e-6) < 0.95 else root_scale
    if calibrated_root is None and human is not None:
        calibrated_root = np.asarray(human["root"], dtype=np.float32) * root_scale
    return {
        "root_scale": root_scale,
        "scene_scale": float(scene_scale),
        "calibrated_root": None
        if calibrated_root is None
        else np.asarray(calibrated_root, dtype=np.float64),
        "median_ratio": median_ratio,
        "background_status": calibration["status"],
        "valid_background_pixels": int(calibration["valid_background_pixels"]),
        "detection_score": float(detection_score),
        "da3_seconds": float(elapsed),
    }


def conditional_rotation(
    fixed: np.ndarray,
    old_predictions: list[dict],
    old_views: list[dict],
    new_prediction: dict,
    new_view: dict,
    pred_layer,
    vggt,
    args,
) -> tuple[np.ndarray, dict]:
    old_humans = [
        predicted_human_summary(prediction, view, pred_layer)
        for prediction, view in zip(old_predictions, old_views)
    ]
    new_human = predicted_human_summary(new_prediction, new_view, pred_layer)
    predicted_frames, motion = predict_torso_frames(old_humans, 1)
    if new_human is None or predicted_frames is None:
        return fixed[:3, :3].copy(), {"branch": "fixed_no_torso", "motion": motion}
    torso, torso_diag = yaw_residual(
        fixed[:3, :3], [new_human["torso"]], predicted_frames, 20.0
    )
    residual = rotation_error(
        np.block([[fixed[:3, :3], np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]]),
        np.block([[torso, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]]),
    )
    if residual < 10.0 or not bool(args.enable_vggt):
        return torso, {
            "branch": "torso",
            "pretrigger": False,
            "vggt_enabled": bool(args.enable_vggt),
            "torso_residual_deg": residual,
            "torso": torso_diag,
        }

    old_meta = [square_view(old_views[-1], False, 20, 16)]
    new_meta = [square_view(new_view, False, 20, 16)]
    pairs, seconds = run_vggt_pairs(vggt, old_meta, new_meta, pair_specs(1, 1), args)
    coarse, consensus = aggregate_coarse(
        pairs,
        [camera_matrix(old_predictions[-1])],
        [camera_matrix(new_prediction)],
    )
    wide = {
        "texture_score": float(
            (
                (new_view["img"][0].mean(0)[:, 1:] - new_view["img"][0].mean(0)[:, :-1])
                .abs()
                .mean()
                + (new_view["img"][0].mean(0)[1:] - new_view["img"][0].mean(0)[:-1])
                .abs()
                .mean()
            )
            .detach()
            .cpu()
        ),
        "windows": {
            "full_rgb_1p1": {
                "rotation_consensus": consensus,
                "candidates": {"coarse": {"transform": coarse.astype(float).tolist()}},
            }
        },
    }
    selected, branch, diagnostics = selected_rotation(
        fixed[:3, :3], torso, wide, 0.05, consensus_cap_deg=60.0
    )
    return selected, {
        "branch": branch,
        "pretrigger": True,
        "torso_residual_deg": residual,
        "vggt_seconds": float(seconds),
        **diagnostics,
    }


def scene_distance(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 32 or len(second) < 32:
        return float("nan")
    first_tree = cKDTree(first)
    second_tree = cKDTree(second)
    values = np.concatenate(
        [second_tree.query(first, k=1, workers=-1)[0], first_tree.query(second, k=1, workers=-1)[0]]
    )
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan")
    return float(np.mean(values[values <= np.percentile(values, 90)]))


def evaluate_method(
    method: str,
    predictions: list[dict],
    views: list[dict],
    local_dir: Path,
    poses: list[np.ndarray],
    humans: list[dict | None],
    gt_poses: list[np.ndarray],
    gt_humans: list[dict | None],
    target_gauge: np.ndarray,
    cuts: list[int],
    fixed: dict[int, np.ndarray],
    rotations: dict[int, np.ndarray],
    shot_scales: dict[int, dict],
    initial_scale: float,
    args,
) -> dict:
    frame_count = len(predictions)
    frame_world = [None] * frame_count
    root_world = [None] * frame_count
    joints_world = [None] * frame_count
    scene_cloud = [None] * frame_count
    shot_state = {}
    cut_set = set(cuts)
    shot_start = 0
    current_gauge = np.eye(4, dtype=np.float64)
    current_scale = initial_scale
    root_delta = np.zeros(3, dtype=np.float64)

    for index in range(frame_count):
        if index in cut_set:
            previous_start = shot_start
            shot_start = index
            if method == "continue":
                shot_state[index] = {
                    "previous_shot_start": previous_start,
                    "scale": float(current_scale),
                    "gauge": current_gauge.astype(float).tolist(),
                    "root_delta": root_delta.astype(float).tolist(),
                }
            elif method == "hard_reset_fixed":
                current_scale = current_scale
                boundary = fixed[index].copy()
                boundary[:3, 3] *= current_scale
                current_gauge = current_gauge @ boundary
                root_delta = np.zeros(3, dtype=np.float64)
            else:
                if method == "v11_1":
                    current_scale = initial_scale
                elif method == "v11_4":
                    current_scale = float(shot_scales[index]["scene_scale"])
                elif method == "unified_da3":
                    current_scale = float(shot_scales[index]["root_scale"])
                previous_root = root_world[index - 1]
                camera_rotation = (
                    current_gauge[:3, :3]
                    @ rotations[index]
                    @ poses[index][:3, :3]
                )
                raw_root = (
                    np.asarray(humans[index]["root"], dtype=np.float64)
                    if humans[index] is not None
                    else np.zeros(3, dtype=np.float64)
                )
                calibrated = raw_root * current_scale
                if method == "unified_da3" and shot_scales[index]["calibrated_root"] is not None:
                    calibrated = np.asarray(shot_scales[index]["calibrated_root"], dtype=np.float64)
                camera_pose = np.eye(4, dtype=np.float64)
                camera_pose[:3, :3] = camera_rotation
                camera_pose[:3, 3] = previous_root - camera_rotation @ calibrated
                current_gauge = camera_pose @ np.linalg.inv(scale_pose(poses[index], current_scale))
                root_delta = calibrated - raw_root * current_scale
            if method != "continue":
                shot_state[index] = {
                    "previous_shot_start": previous_start,
                    "scale": float(current_scale),
                    "gauge": current_gauge.astype(float).tolist(),
                    "root_delta": root_delta.astype(float).tolist(),
                }

        camera = current_gauge @ scale_pose(poses[index], current_scale)
        frame_world[index] = camera
        human = humans[index]
        if human is not None:
            root_camera = np.asarray(human["root"], dtype=np.float64) * current_scale + root_delta
            body = (np.asarray(human["joints"], dtype=np.float64) - human["root"]) * current_scale
            root_world[index] = camera[:3, :3] @ root_camera + camera[:3, 3]
            joints_world[index] = body @ camera[:3, :3].T + root_world[index]
        else:
            root_world[index] = camera[:3, 3].copy()
        if method == "continue":
            scene_cloud[index] = np.empty((0, 3), dtype=np.float64)
        else:
            cloud, _ = background_cloud(
                local_dir, index, int(args.scene_samples), int(args.seed) + 97 * index
            )
            scene_cloud[index] = transform_points(current_gauge, cloud * current_scale)

    target_camera = [target_gauge @ pose for pose in gt_poses]
    target_root = []
    target_joints = []
    for pose, human in zip(target_camera, gt_humans):
        if human is None:
            target_root.append(None)
            target_joints.append(None)
        else:
            target_root.append(transform_points(pose, human["root"][None])[0])
            target_joints.append(transform_points(pose, human["joints"]))

    per_frame = []
    for index in range(frame_count):
        root_error = (
            float(np.linalg.norm(root_world[index] - target_root[index]))
            if target_root[index] is not None and root_world[index] is not None
            else float("nan")
        )
        joint_error = (
            float(np.mean(np.linalg.norm(joints_world[index] - target_joints[index], axis=1)))
            if target_joints[index] is not None and joints_world[index] is not None
            else float("nan")
        )
        per_frame.append(
            {
                "frame": index,
                "camera_translation_m": float(
                    np.linalg.norm(frame_world[index][:3, 3] - target_camera[index][:3, 3])
                ),
                "camera_rotation_deg": rotation_error(frame_world[index], target_camera[index]),
                "human_root_m": root_error,
                "human_joints_m": joint_error,
            }
        )

    boundaries = []
    for cut in cuts:
        boundaries.append(
            {
                "cut": cut,
                "camera_translation_m": per_frame[cut]["camera_translation_m"],
                "camera_rotation_deg": per_frame[cut]["camera_rotation_deg"],
                "human_root_m": per_frame[cut]["human_root_m"],
                "human_joints_m": per_frame[cut]["human_joints_m"],
                "scene_discontinuity_m": scene_distance(scene_cloud[cut - 1], scene_cloud[cut]),
                "scale": float(shot_state[cut]["scale"]),
            }
        )

    prefixes = {}
    for count in (1, 2, 4, 8):
        if count > len(cuts):
            continue
        final_frame = min(cuts[count - 1] + int(args.frames_per_shot) - 1, frame_count - 1)
        rows = boundaries[:count]
        prefixes[str(count)] = {
            "final_frame": final_frame,
            "camera_cumulative_drift_m": per_frame[final_frame]["camera_translation_m"],
            "camera_cumulative_rotation_deg": per_frame[final_frame]["camera_rotation_deg"],
            "human_root_cumulative_drift_m": per_frame[final_frame]["human_root_m"],
            "human_joint_cumulative_drift_m": per_frame[final_frame]["human_joints_m"],
            "boundary_camera_translation_m": finite_distribution(
                [row["camera_translation_m"] for row in rows]
            ),
            "boundary_human_root_m": finite_distribution([row["human_root_m"] for row in rows]),
            "scene_discontinuity_m": finite_distribution(
                [row["scene_discontinuity_m"] for row in rows]
            ),
            "shot_scale_abs_log_drift": float(
                abs(math.log(max(rows[-1]["scale"], 1e-6) / max(initial_scale, 1e-6)))
            ),
        }
    return {
        "prefixes": prefixes,
        "boundaries": boundaries,
        "per_frame": per_frame,
        "shot_state": shot_state,
    }


def aggregate(cases: list[dict]) -> dict:
    output = {}
    for method in METHODS:
        output[method] = {}
        for count in ("1", "2", "4", "8"):
            rows = [case["methods"][method]["prefixes"][count] for case in cases]
            output[method][count] = {
                key: finite_distribution([row[key] for row in rows])
                for key in (
                    "camera_cumulative_drift_m",
                    "camera_cumulative_rotation_deg",
                    "human_root_cumulative_drift_m",
                    "human_joint_cumulative_drift_m",
                    "shot_scale_abs_log_drift",
                )
            }
            output[method][count]["scene_discontinuity_m"] = finite_distribution(
                [row["scene_discontinuity_m"]["mean"] for row in rows]
            )
    return output


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("V14.5 true recurrent audit requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = select_records(
        read_jsonl(args.records), int(args.cases_per_source), int(args.seed)
    )
    if args.max_cases > 0:
        records = records[: int(args.max_cases)]
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    human3r = build_model(args)
    gt_model, pred_layer = build_smpl_models(human3r, device)
    keypoint_model = keypointrcnn_resnet50_fpn(
        weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    ).to(device).eval()
    da3_model = DepthAnything3.from_pretrained(str(args.da3_model_path)).to(device).eval()
    args.query_rows = 20
    args.query_cols = 16
    args.pair_batch_size = 2
    vggt = build_vggt(args) if bool(args.enable_vggt) else None
    cases = []
    started = time.perf_counter()

    for case_index, record in enumerate(records):
        print(f">> recurrent [{case_index + 1}/{len(records)}] {record['pattern_id']}", flush=True)
        views, cuts = load_rollout_views(record, args, human3r, device)
        for view in views:
            view["reset"] = torch.zeros_like(view["reset"], dtype=torch.bool)
        continue_predictions, _ = human3r.forward_recurrent_lighter(
            views, str(device), ret_state=False, use_ttt3r=False
        )
        for cut in cuts:
            views[cut - 1]["reset"] = torch.ones_like(views[cut - 1]["reset"], dtype=torch.bool)
        with torch.no_grad():
            reset_outputs, _ = inference_recurrent_lighter(
                views, human3r, str(device), use_ttt3r=False
            )
        reset_predictions = reset_outputs["pred"]
        local_dir = args.output_dir / "cases" / str(record["pattern_id"]) / "human3r_true_reset"
        save_predictions(reset_outputs, human3r, local_dir, bool(args.overwrite))
        gt_model.update_smpl_gt(views)

        poses = [camera_matrix(row) for row in reset_predictions]
        continue_poses = [camera_matrix(row) for row in continue_predictions]
        humans = [
            predicted_human(row, view["camera_intrinsics"], pred_layer)
            for row, view in zip(reset_predictions, views)
        ]
        continue_humans = [
            predicted_human(row, view["camera_intrinsics"], pred_layer)
            for row, view in zip(continue_predictions, views)
        ]
        gt_poses = [
            gt_pose_from_view(view).detach().float().cpu().numpy().astype(np.float64)
            for view in views
        ]
        gt_humans = [gt_human(view) for view in views]

        shot_starts = [0, *cuts]
        scales = {}
        for start in shot_starts:
            scales[start] = shot_metric_scale(
                start,
                reset_predictions[start],
                views[start],
                humans[start],
                local_dir,
                keypoint_model,
                da3_model,
                device,
                args,
            )
        initial_scale = float(scales[0]["scene_scale"])
        target_gauge = scale_pose(poses[0], initial_scale) @ np.linalg.inv(gt_poses[0])

        fixed, rotations, rotation_diagnostics = {}, {}, {}
        previous_start = 0
        for cut in cuts:
            fixed[cut] = fixed_boundary(
                local_dir,
                cut,
                previous_start,
                args,
                int(args.seed) + case_index * 10000 + cut,
            )
            old_start = max(previous_start, cut - 5)
            rotations[cut], rotation_diagnostics[cut] = conditional_rotation(
                fixed[cut],
                reset_predictions[old_start:cut],
                views[old_start:cut],
                reset_predictions[cut],
                views[cut],
                pred_layer,
                vggt,
                args,
            )
            previous_start = cut

        methods = {}
        methods["continue"] = evaluate_method(
            "continue",
            continue_predictions,
            views,
            local_dir,
            continue_poses,
            continue_humans,
            gt_poses,
            gt_humans,
            target_gauge,
            cuts,
            fixed,
            rotations,
            scales,
            initial_scale,
            args,
        )
        for method in METHODS[1:]:
            methods[method] = evaluate_method(
                method,
                reset_predictions,
                views,
                local_dir,
                poses,
                humans,
                gt_poses,
                gt_humans,
                target_gauge,
                cuts,
                fixed,
                rotations,
                scales,
                initial_scale,
                args,
            )
        cases.append(
            {
                "case_name": str(record["pattern_id"]),
                "source": str(record["source"]),
                "record": record,
                "cuts": cuts,
                "shot_scales": {
                    str(key): {
                        name: (
                            value.astype(float).tolist()
                            if isinstance(value, np.ndarray)
                            else value
                        )
                        for name, value in row.items()
                    }
                    for key, row in scales.items()
                },
                "rotation_diagnostics": {
                    str(key): value for key, value in rotation_diagnostics.items()
                },
                "methods": methods,
            }
        )
        torch.cuda.empty_cache()

    report = {
        "experiment": "V14.5 true recurrent multi-cut audit",
        "case_count": len(cases),
        "protocol": {
            "single_chronological_human3r_stream_per_case": True,
            "actual_pre_decode_reset_at_every_cut": True,
            "previous_predicted_world_is_next_anchor": True,
            "gt_gauge_restart_per_cut": False,
            "future_shot_access": False,
            "frames_per_shot": int(args.frames_per_shot),
            "max_cuts": int(args.max_cuts),
            "v16_bound_deg": 20.0,
            "vggt_pretrigger": "torso residual >= 10 deg",
            "vggt_enabled": bool(args.enable_vggt),
            "v11_4_scale": "DA3 background median_ratio_q15_gate_lt95, once per shot",
            "gt_use": "evaluation only after candidate generation",
        },
        "elapsed_seconds": time.perf_counter() - started,
        "summary": aggregate(cases),
        "cases": cases,
    }
    output = args.output_dir / "v14_5_true_recurrent_multicut.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
    print(f">> wrote {output}", flush=True)


if __name__ == "__main__":
    main()
