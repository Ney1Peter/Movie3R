#!/usr/bin/env python3
"""Audit the 8116 singleton body in one explicit coordinate convention.

The saved ``verts_world`` in the hand-written viewer payloads are not always
world coordinates.  ``SMPL_Layer`` returns ``smpl_v3d`` in the current camera
coordinate system; demo.py converts it with the corresponding camera-to-world
pose before saving.  This script reconstructs the 8116 body both as it was
saved and with the missing/incorrect camera transform fixed, then compares
both variants directly with AvatarReX GT.

GT is opened only by this diagnostic.  It is never part of runtime inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import smplx
import torch
from smplx.joint_names import JOINT_NAMES


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASE = REPO_ROOT / "output/v14/report_comparison_viewers/avatarrex_t1836_c22070935_c22053912_pre5_post25"
DEFAULT_CALIB = Path("/data/wangzheng/iJCV-CODE/data/AvatarReX_raw_meta/lbn1/calibration_full.json")
DEFAULT_SMPL = Path("/data/wangzheng/iJCV-CODE/data/AvatarReX_raw_meta/lbn1/smpl_params.npz")
DEFAULT_OUTPUT = DEFAULT_CASE / "SINGLETON_8116_COORDINATE_AUDIT.json"


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", type=Path, default=DEFAULT_CASE)
    p.add_argument("--calibration", type=Path, default=DEFAULT_CALIB)
    p.add_argument("--smpl-params", type=Path, default=DEFAULT_SMPL)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--frame0", type=int, default=1836)
    p.add_argument("--boundary", type=int, default=5)
    return p.parse_args()


def homogeneous_camera(calibration: dict, camera: str) -> np.ndarray:
    value = calibration[str(camera)]
    # AvatarReX convention: X_cam = R_w2c X_world + T_w2c.
    r = np.asarray(value["R"], dtype=np.float64).reshape(3, 3)
    t = np.asarray(value["T"], dtype=np.float64).reshape(3)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = r.T
    out[:3, 3] = -r.T @ t
    return out


def load_pose(path: Path) -> np.ndarray:
    with np.load(path) as z:
        return np.asarray(z["pose"], dtype=np.float64)


def load_vertices(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=True) as z:
        return np.asarray(z["verts_world"], dtype=np.float64)[0]


def transform(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def kabsch_rotation_angle(pred: np.ndarray, target: np.ndarray) -> float:
    """Best rigid rotation angle after the two sets are independently centered."""
    a = pred - pred.mean(axis=0)
    b = target - target.mean(axis=0)
    u, _, vt = np.linalg.svd(a.T @ b)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0.0:
        vt[-1] *= -1.0
        r = vt.T @ u.T
    value = np.clip((np.trace(r) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(value)))


def rigid_aligned_mpvpe(pred: np.ndarray, target: np.ndarray) -> float:
    a = pred - pred.mean(axis=0)
    b = target - target.mean(axis=0)
    u, _, vt = np.linalg.svd(a.T @ b)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0.0:
        vt[-1] *= -1.0
        r = vt.T @ u.T
    return float(np.linalg.norm(a @ r.T - b, axis=1).mean())


def frame_metrics(pred: np.ndarray, target: np.ndarray, regressor: np.ndarray) -> dict[str, float]:
    pred_joints = regressor @ pred
    target_joints = regressor @ target
    pred_root = pred_joints[0]
    target_root = target_joints[0]
    delta_joints = pred_joints - target_joints
    centered_delta = (pred_joints - pred_root) - (target_joints - target_root)
    return {
        "root_error_m": float(np.linalg.norm(pred_root - target_root)),
        "mean_joint_error_m": float(np.linalg.norm(delta_joints, axis=1).mean()),
        "p95_joint_error_m": float(np.percentile(np.linalg.norm(delta_joints, axis=1), 95)),
        "max_joint_error_m": float(np.linalg.norm(delta_joints, axis=1).max()),
        "centered_joint_error_m": float(np.linalg.norm(centered_delta, axis=1).mean()),
        "mpvpe_m": float(np.linalg.norm(pred - target, axis=1).mean()),
        "centered_mpvpe_m": float(
            np.linalg.norm((pred - pred.mean(0)) - (target - target.mean(0)), axis=1).mean()
        ),
        "rigid_aligned_mpvpe_m": rigid_aligned_mpvpe(pred, target),
        "best_global_rotation_deg": kabsch_rotation_angle(pred_joints, target_joints),
    }


def camera_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    relative = pred @ np.linalg.inv(target)
    angle = np.degrees(np.arccos(np.clip((np.trace(relative[:3, :3]) - 1.0) * 0.5, -1.0, 1.0)))
    return {
        "translation_error_m": float(np.linalg.norm(pred[:3, 3] - target[:3, 3])),
        "rotation_error_deg": float(angle),
    }


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def main() -> None:
    a = args()
    case = a.case.resolve()
    calibration = json.loads(a.calibration.read_text(encoding="utf-8"))
    with np.load(a.smpl_params) as payload:
        gt_params = {key: np.asarray(payload[key]) for key in payload.files}

    model = smplx.create(
        str(REPO_ROOT / "src/models"), "smplx", gender="neutral", use_pca=False,
        flat_hand_mean=True, num_betas=10,
    ).eval()
    regressor = model.J_regressor.detach().cpu().numpy().astype(np.float64)
    if regressor.shape[0] != 55:
        raise RuntimeError(f"Expected the 55-joint SMPL-X regressor, got {regressor.shape}")

    # The last pre camera fixes the predicted gauge used by the viewer.
    pre_pred = load_pose(case / "original_human3r" / "camera" / f"{a.boundary - 1:06d}.npz")
    gt_pre = homogeneous_camera(calibration, "22070935")
    gt_post = homogeneous_camera(calibration, "22053912")
    gauge = pre_pred @ np.linalg.inv(gt_pre)
    gt_post_gauge_camera = gauge @ gt_post

    # The old 8116 payload and its source original Human3R payload.
    old_dir = case / "movie3r_current_camera_original_body"
    original_dir = case / "original_human3r"
    current_dir = case / "movie3r_b0_brtc_c1"
    manifest = json.loads((case / "manifest.json").read_text(encoding="utf-8"))
    shift = np.asarray(manifest["movie3r"]["brtc"]["group_shift_world"], dtype=np.float64)

    methods: dict[str, list[dict[str, float]]] = {
        "8116_saved_transform_Ccur_inv_Corig": [],
        "8116_coordinate_correct_Ccur_times_Vcam": [],
        "8117_saved_transform_plus_BRTC_shift": [],
    }
    per_frame: list[dict] = []
    camera_rows: list[dict[str, float]] = []

    for local_index in range(a.boundary, a.boundary + 25):
        frame = a.frame0 + local_index - a.boundary
        # Build GT SMPL-X mesh for this post timestamp.
        index = frame
        kwargs = {
            "betas": torch.from_numpy(gt_params["betas"][0:1]).float(),
            "global_orient": torch.from_numpy(gt_params["global_orient"][index:index + 1]).float(),
            "body_pose": torch.from_numpy(gt_params["body_pose"][index:index + 1]).float(),
            "jaw_pose": torch.from_numpy(gt_params["jaw_pose"][index:index + 1]).float(),
            "left_hand_pose": torch.from_numpy(gt_params["left_hand_pose"][index:index + 1]).float(),
            "right_hand_pose": torch.from_numpy(gt_params["right_hand_pose"][index:index + 1]).float(),
            "expression": torch.from_numpy(gt_params["expression"][index:index + 1]).float(),
            "transl": torch.from_numpy(gt_params["transl"][index:index + 1]).float(),
        }
        with torch.no_grad():
            gt_local = model(**kwargs).vertices[0].detach().cpu().numpy().astype(np.float64)
        gt_world = transform(gauge, gt_local)

        current_camera = load_pose(current_dir / "camera" / f"{local_index:06d}.npz")
        original_camera = load_pose(original_dir / "camera" / f"{local_index:06d}.npz")
        old_saved = load_vertices(old_dir / "smpl" / f"{local_index:06d}.npz")
        original_vcam = load_vertices(original_dir / "smpl" / f"{local_index:06d}.npz")

        # 8116's actual construction: it treated V_cam as if it were V_world.
        old_variant = old_saved
        # Coordinate-correct construction: SMPL_Layer returns V_cam, so apply
        # the camera-to-world pose exactly once.  This is the operation used
        # by demo.py before writing a world-space mesh.
        corrected_variant = transform(current_camera, original_vcam)
        old_plus_shift = old_variant + shift

        methods["8116_saved_transform_Ccur_inv_Corig"].append(frame_metrics(old_variant, gt_world, regressor))
        methods["8116_coordinate_correct_Ccur_times_Vcam"].append(frame_metrics(corrected_variant, gt_world, regressor))
        methods["8117_saved_transform_plus_BRTC_shift"].append(frame_metrics(old_plus_shift, gt_world, regressor))
        camera_rows.append(camera_metrics(current_camera, gt_post_gauge_camera))

        first = local_index == a.boundary
        if first:
            first_deltas = {}
            p_j = regressor @ corrected_variant
            g_j = regressor @ gt_world
            names = list(JOINT_NAMES[:55])
            for joint, name in enumerate(names):
                first_deltas[name] = {
                    "offset_vector_m": (p_j[joint] - g_j[joint]).tolist(),
                    "offset_norm_m": float(np.linalg.norm(p_j[joint] - g_j[joint])),
                }
        per_frame.append({
            "local_index": int(local_index), "dataset_frame": int(frame),
            "camera": camera_metrics(current_camera, gt_post_gauge_camera),
            "methods": {
                "8116_saved_transform_Ccur_inv_Corig": methods["8116_saved_transform_Ccur_inv_Corig"][-1],
                "8116_coordinate_correct_Ccur_times_Vcam": methods["8116_coordinate_correct_Ccur_times_Vcam"][-1],
                "8117_saved_transform_plus_BRTC_shift": methods["8117_saved_transform_plus_BRTC_shift"][-1],
            },
        })

    report = {
        "case": str(case),
        "coordinate_contract": {
            "smpl_layer_output": "camera-space V_cam/J_cam",
            "demo_world_export": "V_world = C_cam_to_world @ V_cam",
            "8116_operation": "V_saved = (C_current @ inv(C_original)) @ V_cam (invalid for this payload)",
            "coordinate_correct_operation": "V_world = C_current @ V_cam",
            "gt_camera_convention": "AvatarReX X_cam = R_w2c X_world + T_w2c; C_gt = [R^T, -R^T T]",
            "gt_used": "diagnostic only",
        },
        "gauge": {"definition": "C_pred_pre_last @ inv(C_gt_pre)", "matrix": gauge.tolist()},
        "brtc_shift_world_from_manifest": shift.tolist(),
        "camera_first_post": camera_rows[0],
        "camera_mean_post25": mean_metrics(camera_rows),
        "first_post_joint_offsets_coordinate_correct_variant": first_deltas,
        "methods_first_post": {key: rows[0] for key, rows in methods.items()},
        "methods_mean_post25": {key: mean_metrics(rows) for key, rows in methods.items()},
        "per_frame": per_frame,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(a.output),
        "camera_first_post": report["camera_first_post"],
        "methods_first_post": report["methods_first_post"],
        "methods_mean_post25": report["methods_mean_post25"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
