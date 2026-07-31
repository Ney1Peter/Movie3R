#!/usr/bin/env python3
"""Provisional Multi-THuMBS-style evaluation for saved EgoHumans caches.

This evaluator intentionally does not claim reproduction of the unpublished
Multi-THuMBS supplementary protocol.  It implements local Human3R/GVHMR-style
diagnostics for the metric names reported by the paper, while keeping each
stable GT identity as a separate trajectory.  Formula and unit assumptions are
recorded explicitly rather than attributed to Multi-THuMBS.

The default inputs are the three 15-frame V13 Human3R *raw* EgoHumans probes.
They contain neither B0 nor DA3 output.  Their cut pairs also duplicate the
same dataset timestamp from two synchronized cameras.  These facts are
recorded prominently in every generated report.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from dust3r.utils.smpl_layer import SMPL_Layer  # noqa: E402
from versions.v13.egobody_probe import load_colmap  # noqa: E402

import roma  # noqa: E402
import smplx  # noqa: E402


DEFAULT_DATA_ROOT = Path("/data/wangzheng/iJCV-CODE/data/EgoBody/001_legoassemble")
SMPLX_DIR = REPO_ROOT / "src/models"
SMPLX2SMPL = SMPLX_DIR / "smplx/smplx2smpl.pkl"
DEFAULT_RUN_DIRS = (
    REPO_ROOT / "output/v13/egobody",
    REPO_ROOT / "output/v13/egobody_cam02_cam05_cam08",
    REPO_ROOT / "output/v13/egobody_cam03_cam04_cam01",
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output/v14/fine_alignment_research/multithumbs_protocol"

PAPER_EGOHUMANS_REFERENCE = {
    "source": "Multi-THuMBS Table 1/2, EgoHumans, arXiv:2607.01626v1",
    "w_mpjpe_mm": 279.0,
    "wa_mpjpe_mm": 166.0,
    "mpjpe_mm": 228.3,
    "mpvpe_mm": 262.2,
    "accel_reported_unit_unspecified": 27.3,
    "ate_m_assumed_from_convention": 0.7,
    "ids_per_sequence_assumed_from_table_aggregation": 0.97,
    "comparison_status": (
        "reference_only_not_comparable: exact benchmark construction, split, cut list, "
        "visibility rules, aggregation, and supplementary protocol were not released"
    ),
}


@dataclass
class SparseVertexMap:
    indices: np.ndarray
    weights: np.ndarray

    def apply(self, vertices: np.ndarray) -> np.ndarray:
        """Map (..., 10475, 3) SMPL-X vertices to (..., 6890, 3) SMPL."""
        safe_indices = np.maximum(self.indices, 0)
        selected = vertices[..., safe_indices, :]
        return (selected * self.weights[..., None]).sum(axis=-2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--run_dirs", type=Path, nargs="+", default=list(DEFAULT_RUN_DIRS))
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--self_test", action="store_true")
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_sparse_vertex_map(path: Path) -> SparseVertexMap:
    with path.open("rb") as handle:
        dense = np.asarray(pickle.load(handle)["matrix"])
    if dense.shape != (6890, 10475):
        raise ValueError(f"Unexpected SMPL-X-to-SMPL matrix shape: {dense.shape}")
    rows, columns = np.nonzero(dense)
    counts = np.bincount(rows, minlength=dense.shape[0])
    width = int(counts.max())
    indices = np.full((dense.shape[0], width), -1, dtype=np.int64)
    weights = np.zeros((dense.shape[0], width), dtype=np.float32)
    offsets = np.zeros(dense.shape[0], dtype=np.int64)
    for row, column in zip(rows.tolist(), columns.tolist()):
        slot = int(offsets[row])
        indices[row, slot] = column
        weights[row, slot] = float(dense[row, column])
        offsets[row] += 1
    del dense, rows, columns
    gc.collect()
    return SparseVertexMap(indices=indices, weights=weights)


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def pose_encoding_to_camera_matrix(pose_encoding: torch.Tensor) -> np.ndarray:
    """Decode Human3R absT_quaR (translation + real-first quaternion)."""
    value = pose_encoding.detach().float().cpu().numpy().reshape(-1)
    if value.shape != (7,):
        raise ValueError(f"Expected one 7D absT_quaR pose, got {value.shape}")
    quaternion = value[3:7].astype(np.float64)
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("Degenerate camera quaternion")
    real, i, j, k = quaternion / norm
    rotation = np.asarray(
        [
            [1.0 - 2.0 * (j * j + k * k), 2.0 * (i * j - k * real), 2.0 * (i * k + j * real)],
            [2.0 * (i * j + k * real), 1.0 - 2.0 * (i * i + k * k), 2.0 * (j * k - i * real)],
            [2.0 * (i * k - j * real), 2.0 * (j * k + i * real), 1.0 - 2.0 * (i * i + j * j)],
        ],
        dtype=np.float64,
    )
    camera = np.eye(4, dtype=np.float64)
    camera[:3, :3] = rotation
    camera[:3, 3] = value[:3]
    return camera


def fit_similarity(target: np.ndarray, prediction: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit target ~= scale * R * prediction + translation (Umeyama)."""
    target = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1, 3)
    if target.shape != prediction.shape or len(target) < 3:
        raise ValueError(f"Similarity inputs must match and contain >=3 points: {target.shape}, {prediction.shape}")
    target_mean = target.mean(axis=0)
    prediction_mean = prediction.mean(axis=0)
    target_zero = target - target_mean
    prediction_zero = prediction - prediction_mean
    variance = float(np.square(prediction_zero).sum() / len(prediction))
    if variance <= 1e-14:
        raise ValueError("Degenerate prediction variance for similarity alignment")
    correlation = target_zero.T @ prediction_zero / len(prediction)
    left, singular, right_t = np.linalg.svd(correlation)
    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(left) * np.linalg.det(right_t.T) < 0:
        correction[-1, -1] = -1.0
    rotation = left @ correction @ right_t
    scale = float(np.sum(singular * np.diag(correction)) / variance)
    translation = target_mean - scale * rotation @ prediction_mean
    return scale, rotation, translation


def apply_similarity(points: np.ndarray, fit: tuple[float, np.ndarray, np.ndarray]) -> np.ndarray:
    scale, rotation, translation = fit
    return scale * (points @ rotation.T) + translation


def mean_point_error(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.linalg.norm(first - second, axis=-1).mean(axis=-1)


def pelvis_center(joints: np.ndarray, vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Public GVHMR convention used by Human3R evaluation: average SMPL joints 1 and 2.
    pelvis = joints[..., [1, 2], :].mean(axis=-2, keepdims=True)
    return joints - pelvis, vertices - pelvis


def initial_frame_aligned_errors(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """GVHMR W-MPJPE: Sim(3) from the first two frames, applied to the track."""
    initial_count = min(2, len(target))
    fit = fit_similarity(target[:initial_count], prediction[:initial_count])
    return mean_point_error(target, apply_similarity(prediction, fit))


def trajectory_aligned_errors(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """GVHMR WA-MPJPE: one Sim(3) fitted over the complete identity track."""
    fit = fit_similarity(target, prediction)
    return mean_point_error(target, apply_similarity(prediction, fit))


def acceleration_second_difference_errors(
    target: np.ndarray,
    prediction: np.ndarray,
    frame_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Joint second-difference error, in m/frame^2 for meter-valued input.

    Multi-THuMBS does not publish its Accel formula or unit.  Keeping the raw
    discrete second difference separate from the optional ``fps**2`` conversion
    prevents the paper's unitless table value from being silently compared with
    a physical acceleration in m/s^2.
    """
    if len(target) < 3:
        return np.empty(0, dtype=np.float64)
    target_acceleration = target[:-2] - 2.0 * target[1:-1] + target[2:]
    prediction_acceleration = prediction[:-2] - 2.0 * prediction[1:-1] + prediction[2:]
    errors = np.linalg.norm(
        prediction_acceleration - target_acceleration, axis=-1
    ).mean(axis=-1)
    if frame_indices is not None:
        frame_indices = np.asarray(frame_indices, dtype=np.int64)
        contiguous = (frame_indices[1:-1] - frame_indices[:-2] == 1) & (
            frame_indices[2:] - frame_indices[1:-1] == 1
        )
        errors = errors[contiguous]
    return errors


def acceleration_errors(
    target: np.ndarray,
    prediction: np.ndarray,
    fps: float,
    frame_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Physical acceleration error in m/s^2, derived from discrete samples."""
    return acceleration_second_difference_errors(
        target, prediction, frame_indices
    ) * float(fps) ** 2


def identity_switches(gt_to_track_ids: dict[str, list[int | None]]) -> tuple[int, dict[str, int]]:
    per_identity = {}
    for identity, values in gt_to_track_ids.items():
        switches = 0
        previous = None
        for value in values:
            if value is None:
                continue
            if previous is not None and int(value) != int(previous):
                switches += 1
            previous = int(value)
        per_identity[identity] = switches
    return int(sum(per_identity.values())), per_identity


def predicted_frame_vertices(
    prediction: dict,
    layer: SMPL_Layer,
    vertex_map: SparseVertexMap,
    device: torch.device,
) -> np.ndarray:
    rotmat = prediction["smpl_rotmat"][0].to(device=device, dtype=torch.float32)
    shape = prediction["smpl_shape"][0].to(device=device, dtype=torch.float32)
    translation = prediction["smpl_transl"][0].to(device=device, dtype=torch.float32)
    if rotmat.shape[0] == 0:
        return np.empty((0, 6890, 3), dtype=np.float32)
    rotvec = roma.rotmat_to_rotvec(rotmat)
    intrinsics = torch.eye(3, device=device, dtype=torch.float32).expand(len(rotvec), -1, -1)
    with torch.no_grad():
        result = layer(
            rotvec,
            shape,
            translation,
            None,
            None,
            K=intrinsics,
            expression=None,
        )
    smplx_vertices = result["smpl_v3d"].detach().float().cpu().numpy()
    return vertex_map.apply(smplx_vertices).astype(np.float32)


def run_paths(run_dir: Path) -> tuple[Path, Path]:
    cache = run_dir / "v13_egobody_compact_tokens.pt"
    report = run_dir / "v13_egobody_three_person_probe.json"
    if not cache.is_file() or not report.is_file():
        raise FileNotFoundError(f"Missing V13 cache/report under {run_dir}")
    return cache, report


def unavailable(reason: str) -> dict:
    return {"status": "unavailable", "reason": reason}


def evaluate_run(
    run_dir: Path,
    data_root: Path,
    exo: dict,
    layer: SMPL_Layer,
    vertex_map: SparseVertexMap,
    joint_regressor: np.ndarray,
    device: torch.device,
    fps: float,
) -> tuple[dict, dict[str, np.ndarray]]:
    cache_path, source_report_path = run_paths(run_dir)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    predictions = cache["predictions"]
    labels = cache["labels"]
    segments = source_report["stream"]["segments"]
    cameras = [row["camera"] for row in segments for _ in row["frames"]]
    frames = [int(frame) for row in segments for frame in row["frames"]]
    identities = [str(value) for value in source_report["dataset"]["identities"]]
    if not (len(predictions) == len(labels) == len(cameras) == len(frames)):
        raise ValueError(f"Inconsistent stream lengths in {run_dir}")

    tracks: dict[str, dict[str, list[np.ndarray]]] = {
        identity: {
            "frame_indices": [],
            "pred_camera_joints": [],
            "target_camera_joints": [],
            "pred_camera_vertices": [],
            "target_camera_vertices": [],
            "pred_world_joints": [],
            "target_world_joints": [],
        }
        for identity in identities
    }
    gt_to_track_ids: dict[str, list[int | None]] = {
        identity: [None] * len(predictions) for identity in identities
    }
    pred_cameras, target_cameras = [], []
    matched_observations = 0

    for frame_index, (prediction, frame_labels, camera_name, frame_number) in enumerate(
        zip(predictions, labels, cameras, frames)
    ):
        predicted_camera = pose_encoding_to_camera_matrix(prediction["camera_pose"][0])
        target_camera = np.asarray(exo[camera_name]["c2w_aria01"], dtype=np.float64)
        pred_cameras.append(predicted_camera)
        target_cameras.append(target_camera)
        predicted_vertices = predicted_frame_vertices(prediction, layer, vertex_map, device)
        predicted_track_ids = prediction["smpl_id"][0].detach().cpu().numpy().astype(int)
        frame_labels = np.asarray(frame_labels).astype(int)
        gt_frame = np.load(
            data_root / f"processed_data/smpl/{frame_number:05d}.npy", allow_pickle=True
        ).item()
        world_to_target_camera = np.linalg.inv(target_camera)

        for detection_index, gt_index in enumerate(frame_labels.tolist()):
            if gt_index < 0 or gt_index >= len(identities) or detection_index >= len(predicted_vertices):
                continue
            identity = identities[gt_index]
            if tracks[identity]["frame_indices"] and tracks[identity]["frame_indices"][-1] == frame_index:
                # Reject duplicate association in one frame rather than silently corrupting a track.
                raise ValueError(
                    f"Duplicate GT identity at {run_dir}, frame {frame_index}, {identity}"
                )
            pred_vertices_camera = predicted_vertices[detection_index].astype(np.float64)
            target_vertices_world = np.asarray(gt_frame[identity]["vertices"], dtype=np.float64)
            target_vertices_camera = transform_points(world_to_target_camera, target_vertices_world)
            pred_joints_camera = joint_regressor @ pred_vertices_camera
            target_joints_camera = joint_regressor @ target_vertices_camera
            pred_vertices_world = transform_points(predicted_camera, pred_vertices_camera)
            pred_joints_world = joint_regressor @ pred_vertices_world
            target_joints_world = joint_regressor @ target_vertices_world
            tracks[identity]["frame_indices"].append(frame_index)
            tracks[identity]["pred_camera_joints"].append(pred_joints_camera)
            tracks[identity]["target_camera_joints"].append(target_joints_camera)
            tracks[identity]["pred_camera_vertices"].append(pred_vertices_camera)
            tracks[identity]["target_camera_vertices"].append(target_vertices_camera)
            tracks[identity]["pred_world_joints"].append(pred_joints_world)
            tracks[identity]["target_world_joints"].append(target_joints_world)
            gt_to_track_ids[identity][frame_index] = int(predicted_track_ids[detection_index])
            matched_observations += 1

    arrays: dict[str, list[np.ndarray]] = {
        "mpjpe_m": [],
        "mpvpe_m": [],
        "w_mpjpe_m": [],
        "wa_mpjpe_m": [],
        "accel_second_difference_m_per_frame2": [],
        "accel_physical_m_per_s2": [],
    }
    per_identity = {}
    for identity, values in tracks.items():
        frame_indices = np.asarray(values["frame_indices"], dtype=np.int64)
        stacked = {key: np.stack(item) for key, item in values.items() if key != "frame_indices"}
        pred_joints_centered, pred_vertices_centered = pelvis_center(
            stacked["pred_camera_joints"], stacked["pred_camera_vertices"]
        )
        target_joints_centered, target_vertices_centered = pelvis_center(
            stacked["target_camera_joints"], stacked["target_camera_vertices"]
        )
        identity_arrays = {
            "mpjpe_m": mean_point_error(target_joints_centered, pred_joints_centered),
            "mpvpe_m": mean_point_error(target_vertices_centered, pred_vertices_centered),
            "w_mpjpe_m": initial_frame_aligned_errors(
                stacked["target_world_joints"], stacked["pred_world_joints"]
            ),
            "wa_mpjpe_m": trajectory_aligned_errors(
                stacked["target_world_joints"], stacked["pred_world_joints"]
            ),
            "accel_second_difference_m_per_frame2": (
                acceleration_second_difference_errors(
                    target_joints_centered,
                    pred_joints_centered,
                    frame_indices,
                )
            ),
            "accel_physical_m_per_s2": acceleration_errors(
                target_joints_centered,
                pred_joints_centered,
                fps,
                frame_indices,
            ),
        }
        for key, value in identity_arrays.items():
            arrays[key].append(value)
        per_identity[identity] = {
            "frame_count": len(stacked["pred_world_joints"]),
            "stream_frame_indices": frame_indices,
            "w_mpjpe_mm": float(identity_arrays["w_mpjpe_m"].mean() * 1000.0),
            "wa_mpjpe_mm": float(identity_arrays["wa_mpjpe_m"].mean() * 1000.0),
            "mpjpe_mm": float(identity_arrays["mpjpe_m"].mean() * 1000.0),
            "mpvpe_mm": float(identity_arrays["mpvpe_m"].mean() * 1000.0),
            "accel_second_difference_mm_per_frame2": (
                float(
                    identity_arrays[
                        "accel_second_difference_m_per_frame2"
                    ].mean()
                    * 1000.0
                )
                if len(identity_arrays["accel_second_difference_m_per_frame2"])
                else None
            ),
            "accel_physical_m_per_s2": (
                float(identity_arrays["accel_physical_m_per_s2"].mean())
                if len(identity_arrays["accel_physical_m_per_s2"])
                else None
            ),
        }

    flat_arrays = {key: np.concatenate(value) for key, value in arrays.items()}
    pred_centers = np.stack(pred_cameras)[:, :3, 3]
    target_centers = np.stack(target_cameras)[:, :3, 3]
    camera_fit = fit_similarity(target_centers, pred_centers)
    aligned_centers = apply_similarity(pred_centers, camera_fit)
    ate_per_frame = np.linalg.norm(target_centers - aligned_centers, axis=-1)
    ids, ids_per_identity = identity_switches(gt_to_track_ids)

    same_timestamp = bool(source_report["stream"].get("same_timestamp_at_boundary", False))
    result = {
        "name": run_dir.name,
        "source_cache": str(cache_path),
        "source_report": str(source_report_path),
        "scope": "Human3R raw diagnostic; no B0 and no DA3",
        "frames": len(predictions),
        "cuts": len(source_report["stream"]["cuts"]),
        "matched_person_frames": matched_observations,
        "possible_gt_person_frames": len(predictions) * len(identities),
        "matched_gt_person_frame_fraction": float(
            matched_observations / (len(predictions) * len(identities))
        ),
        "gt_identity_assignment_for_pose_metrics": True,
        "same_dataset_timestamp_repeated_at_every_cut": same_timestamp,
        "metrics": {
            "w_mpjpe_mm": float(flat_arrays["w_mpjpe_m"].mean() * 1000.0),
            "wa_mpjpe_mm": float(flat_arrays["wa_mpjpe_m"].mean() * 1000.0),
            "mpjpe_mm": float(flat_arrays["mpjpe_m"].mean() * 1000.0),
            "mpvpe_mm": float(flat_arrays["mpvpe_m"].mean() * 1000.0),
            "accel_second_difference_mm_per_frame2": float(
                flat_arrays["accel_second_difference_m_per_frame2"].mean()
                * 1000.0
            ),
            "accel_physical_m_per_s2": float(
                flat_arrays["accel_physical_m_per_s2"].mean()
            ),
            "ate_m_sim3_translation_rmse": float(np.sqrt(np.mean(np.square(ate_per_frame)))),
            "identity_switches": ids,
            "identity_switches_per_cut": float(ids / max(len(source_report["stream"]["cuts"]), 1)),
        },
        "identity_switches_per_gt_identity": ids_per_identity,
        "per_identity": per_identity,
        "camera_alignment_diagnostics": {
            "sim3_scale": float(camera_fit[0]),
            "predicted_center_variance": float(np.square(pred_centers - pred_centers.mean(0)).sum()),
            "predicted_center_rank": int(np.linalg.matrix_rank(pred_centers - pred_centers.mean(0))),
            "target_center_rank": int(np.linalg.matrix_rank(target_centers - target_centers.mean(0))),
        },
    }
    return result, flat_arrays | {"ate_m": ate_per_frame, "ids": np.asarray([ids], dtype=np.float64)}


def aggregate_runs(results: list[dict], arrays: list[dict[str, np.ndarray]]) -> dict:
    concatenated = {
        key: np.concatenate([row[key] for row in arrays])
        for key in (
            "w_mpjpe_m",
            "wa_mpjpe_m",
            "mpjpe_m",
            "mpvpe_m",
            "accel_second_difference_m_per_frame2",
            "accel_physical_m_per_s2",
            "ate_m",
        )
    }
    ids = np.asarray([result["metrics"]["identity_switches"] for result in results], dtype=np.float64)
    total_cuts = sum(int(result["cuts"]) for result in results)
    return {
        "aggregation": {
            "pose_and_accel": "person-frame weighted over stable GT-identity tracks",
            "ate": "camera-frame weighted after an independent sequence-level Sim(3)",
            "ids": "mean raw count per 15-frame stream; total and per-cut also reported",
        },
        "metrics": {
            "w_mpjpe_mm": float(concatenated["w_mpjpe_m"].mean() * 1000.0),
            "wa_mpjpe_mm": float(concatenated["wa_mpjpe_m"].mean() * 1000.0),
            "mpjpe_mm": float(concatenated["mpjpe_m"].mean() * 1000.0),
            "mpvpe_mm": float(concatenated["mpvpe_m"].mean() * 1000.0),
            "accel_second_difference_mm_per_frame2": float(
                concatenated["accel_second_difference_m_per_frame2"].mean()
                * 1000.0
            ),
            "accel_physical_m_per_s2": float(
                concatenated["accel_physical_m_per_s2"].mean()
            ),
            "ate_m_sim3_translation_rmse": float(np.sqrt(np.mean(np.square(concatenated["ate_m"])))),
            "identity_switches_mean_per_stream": float(ids.mean()),
            "identity_switches_total": int(ids.sum()),
            "identity_switches_per_cut": float(ids.sum() / max(total_cuts, 1)),
        },
    }


def metric_protocol() -> dict:
    return {
        "status": "provisional_not_official_Multi-THuMBS_protocol",
        "evidence": {
            "paper": (
                "Multi-THuMBS names W-MPJPE (initial-frame alignment), WA-MPJPE "
                "(trajectory-level alignment), MPJPE, MPVPE, Accel, ATE, and IDs."
            ),
            "formula_source": (
                "W/WA and pelvis-aligned MPJPE/MPVPE follow the local Human3R/GVHMR-style "
                "evaluation utilities. Accel is only a provisional second-difference "
                "diagnostic because the Multi-THuMBS formula and unit are unpublished."
            ),
            "missing_information": (
                "arXiv:2607.01626v1 says details are in supplementary material, but its PDF "
                "and 26-file arXiv source package contain no supplementary material; no author "
                "evaluation code was found as of 2026-07-31."
            ),
        },
        "w_mpjpe": "Per GT identity: fit Sim(3) on its first two frames/all 24 joints, apply to full world track, mean joint error (mm).",
        "wa_mpjpe": "Per GT identity: fit one Sim(3) on all frames/all 24 joints, mean world joint error (mm).",
        "mpjpe": "Per matched person-frame in GT camera coordinates, subtract mean of SMPL hip joints 1/2, mean 24-joint error (mm).",
        "mpvpe": "Same pelvis translation alignment as MPJPE after SMPL-X-to-SMPL conversion, mean 6890-vertex error (mm).",
        "accel": (
            "Per GT identity on pelvis-centered camera-coordinate joints, report both the "
            "raw discrete second-difference error (mm/frame^2) and its fps^2-scaled physical "
            "version (m/s^2). Neither is claimed to be the unpublished paper protocol."
        ),
        "ate": "Sequence-level Sim(3)-aligned camera-center translation RMSE (m), matching common evo ATE translation convention.",
        "ids": "After oracle per-frame GT/detection matching, count every predicted native track-ID change along each stable GT identity.",
        "visibility": (
            "No paper visibility threshold is known. Pose metrics use only detections assigned to a GT "
            "identity; misses are reported through matched-person-frame coverage and are not charged as pose error."
        ),
    }


def limitations() -> list[str]:
    return [
        "The evaluated data are three hand-selected 15-frame streams from one EgoHumans capture, not the unreleased Multi-THuMBS benchmark split or cut construction.",
        "The caches are Human3R raw shot-reset predictions. They contain no B0 boundary and no DA3 refinement, so the numbers are not B0+DA3 results.",
        "Each stream has two camera cuts where the same dataset timestamp appears on both sides. The discrete Accel column treats them as adjacent samples, and the physical column additionally assumes 30 fps, so both are only cut-jump diagnostics.",
        "GT bboxes/labels associate each detection to a stable identity for metric computation. This is evaluator-side oracle matching, not deployable Re-ID input.",
        "Some streams miss people in difficult views. Pose errors average matched observations only, and Accel uses only contiguous three-observation windows; this can favor low-recall predictions.",
        "The local EgoHumans GT is SMPL while Human3R predicts SMPL-X. MPVPE first applies the repository's fixed SMPL-X-to-SMPL vertex transfer and then uses a neutral-SMPL 24-joint regressor for both prediction and GT.",
        "The exact Multi-THuMBS visibility, miss/false-positive, aggregation, model topology, ATE alignment, and IDs conventions are unavailable; table values cannot be used for winner/loser claims against this report.",
    ]


def write_markdown(report: dict, path: Path) -> None:
    aggregate = report["aggregate"]["metrics"]
    lines = [
        "# Provisional Multi-THuMBS-style EgoHumans diagnostic",
        "",
        "> **Human3R raw only: no B0, no DA3. Not the Multi-THuMBS paper split or official protocol.**",
        "",
        "## Aggregate diagnostic",
        "",
        "| W-MPJPE (mm) | WA-MPJPE (mm) | MPJPE (mm) | MPVPE (mm) | Accel Δ² (mm/frame²) | Accel physical (m/s²) | ATE (m) | IDs / stream |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {aggregate['w_mpjpe_mm']:.1f} | {aggregate['wa_mpjpe_mm']:.1f} | "
            f"{aggregate['mpjpe_mm']:.1f} | {aggregate['mpvpe_mm']:.1f} | "
            f"{aggregate['accel_second_difference_mm_per_frame2']:.2f} | "
            f"{aggregate['accel_physical_m_per_s2']:.2f} | "
            f"{aggregate['ate_m_sim3_translation_rmse']:.3f} | "
            f"{aggregate['identity_switches_mean_per_stream']:.2f} |"
        ),
        "",
        "Pose metrics are person-frame weighted after separating the three stable GT identities. "
        "ATE is aligned independently per 15-frame stream. IDs are native Human3R track-ID changes "
        "after evaluator-side GT association.",
        "",
        "## Per-stream diagnostic",
        "",
        "| Stream | W | WA | MPJPE | MPVPE | Accel Δ² | Accel physical | ATE | IDs | IDs/cut |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["runs"]:
        metric = row["metrics"]
        lines.append(
            f"| {row['name']} | {metric['w_mpjpe_mm']:.1f} | {metric['wa_mpjpe_mm']:.1f} | "
            f"{metric['mpjpe_mm']:.1f} | {metric['mpvpe_mm']:.1f} | "
            f"{metric['accel_second_difference_mm_per_frame2']:.2f} | "
            f"{metric['accel_physical_m_per_s2']:.2f} | "
            f"{metric['ate_m_sim3_translation_rmse']:.3f} | "
            f"{metric['identity_switches']} | {metric['identity_switches_per_cut']:.2f} |"
        )
    reference = report["paper_reference_only"]["EgoHumans"]
    lines.extend(
        [
            "",
            "## Paper reference — not directly comparable",
            "",
            "Multi-THuMBS reports `279.0 / 166.0 / 228.3 / 262.2 / 27.3` for "
            "W-MPJPE / WA-MPJPE / MPJPE / MPVPE / Accel on EgoHumans, plus ATE `0.7` "
            "and IDs `0.97`. The exact benchmark and supplementary protocol are unavailable, "
            "so these values are reference targets only and no superiority conclusion is valid.",
            "",
            f"Protocol status: `{report['protocol']['status']}`.",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {value}" for value in report["limitations"])
    lines.extend(
        [
            "",
            "## Unavailable metrics",
            "",
            f"- PCK*: {report['unavailable_metrics']['pck_star']['reason']}",
            f"- Jitter: {report['unavailable_metrics']['jitter']['reason']}",
            f"- Foot Sliding: {report['unavailable_metrics']['foot_sliding']['reason']}",
            "",
            f"Paper reference status: `{reference['comparison_status']}`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_self_test() -> None:
    rng = np.random.default_rng(7)
    prediction = rng.normal(size=(5, 24, 3))
    angle = 0.4
    rotation = np.asarray(
        [[math.cos(angle), -math.sin(angle), 0.0], [math.sin(angle), math.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    target = 1.7 * (prediction @ rotation.T) + np.asarray([2.0, -3.0, 0.5])
    assert initial_frame_aligned_errors(target, prediction).max() < 1e-10
    assert trajectory_aligned_errors(target, prediction).max() < 1e-10
    zero_second_difference = acceleration_second_difference_errors(target, target)
    zero_acceleration = acceleration_errors(target, target, 30.0)
    assert np.max(np.abs(zero_second_difference)) == 0.0
    assert np.max(np.abs(zero_acceleration)) == 0.0
    total, per_identity = identity_switches({"a": [1, 1, 2, 2, 1], "b": [3, None, 3, 4]})
    assert total == 3 and per_identity == {"a": 2, "b": 1}


def main() -> None:
    args = parse_args()
    run_self_test()
    if args.self_test:
        print(">> self-test passed")
        return
    if not args.data_root.is_dir():
        raise FileNotFoundError(args.data_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    _, exo = load_colmap(args.data_root)
    vertex_map = load_sparse_vertex_map(Path(SMPLX2SMPL))
    layer = SMPL_Layer(
        type="smplx", gender="neutral", num_betas=10, kid=False, person_center="head"
    ).to(device).eval()
    smpl_model = smplx.create(SMPLX_DIR, "smpl", gender="neutral").to("cpu")
    joint_regressor_tensor = smpl_model.J_regressor[:24]
    if getattr(joint_regressor_tensor, "is_sparse", False):
        joint_regressor_tensor = joint_regressor_tensor.to_dense()
    joint_regressor = joint_regressor_tensor.detach().float().cpu().numpy().astype(np.float64)

    results, all_arrays = [], []
    for run_dir in args.run_dirs:
        print(f">> evaluating {run_dir}", flush=True)
        result, arrays = evaluate_run(
            run_dir=run_dir,
            data_root=args.data_root,
            exo=exo,
            layer=layer,
            vertex_map=vertex_map,
            joint_regressor=joint_regressor,
            device=device,
            fps=float(args.fps),
        )
        results.append(result)
        all_arrays.append(arrays)

    report = {
        "title": "Provisional Multi-THuMBS-style evaluation of saved EgoHumans Human3R raw caches",
        "generated_scope": "read-only input evaluation; output report only",
        "protocol": metric_protocol(),
        "limitations": limitations(),
        "paper_reference_only": {"EgoHumans": PAPER_EGOHUMANS_REFERENCE},
        "runs": results,
        "aggregate": aggregate_runs(results, all_arrays),
        "unavailable_metrics": {
            "pck_star": unavailable(
                "Requires edited AVA/Friends/Big-Bang clips and the unreleased cross-shot PCK protocol; those data/caches are absent."
            ),
            "jitter": unavailable(
                "Paper Table 4 evaluates edited real videos; the local synchronized-view diagnostic is not that benchmark."
            ),
            "foot_sliding": unavailable(
                "Paper Table 4 contact/topology/aggregation details and edited-video predictions are unavailable."
            ),
        },
    }
    json_path = args.output_dir / "human3r_raw_egohumans_provisional.json"
    markdown_path = args.output_dir / "README.md"
    json_path.write_text(json.dumps(jsonable(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_markdown(report, markdown_path)
    print(f">> wrote {json_path}", flush=True)
    print(f">> wrote {markdown_path}", flush=True)


if __name__ == "__main__":
    main()
