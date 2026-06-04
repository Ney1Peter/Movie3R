#!/usr/bin/env python3
"""Visualize THUman camera/SMPL coordinates through the training dataloader.

This script intentionally uses ``AvatarReX_AABB`` and ``SMPLModel.update_smpl_gt``
instead of manually projecting raw files.  The goal is to verify what the V8
training path actually sees after resize/crop, SMPL translation conversion, and
camera handling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import roma
import torch

from dust3r.datasets.avatarrex import AvatarReX_AABB
from dust3r.smpl_model import SMPLModel
from dust3r.utils.device import todevice


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data"))
    parser.add_argument("--split", default="training")
    parser.add_argument("--output_dir", type=Path, default=Path("output/v8_thuman_coordinate_check"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resolution", type=int, nargs=2, default=(512, 288), metavar=("W", "H"))
    parser.add_argument(
        "--samples",
        default=(
            "thuman00/cam00,thuman00/cam04,0;"
            "thuman00/cam10,thuman00/cam22,0;"
            "thuman02/cam00,thuman02/cam20,0;"
            "thuman02/cam10,thuman02/cam22,0"
        ),
        help="Semicolon-separated seqA,seqB,start_frame records.",
    )
    return parser.parse_args()


def parse_samples(text: str) -> list[tuple[str, str, int]]:
    samples = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        seq_a, seq_b, start = [part.strip() for part in chunk.split(",")]
        samples.append((seq_a, seq_b, int(start)))
    return samples


def tensor_image_to_bgr(img: torch.Tensor) -> np.ndarray:
    arr = img.detach().cpu().float().permute(1, 2, 0).numpy()
    arr = ((arr * 0.5 + 0.5).clip(0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def mask_to_uint8(mask: torch.Tensor | None, hw: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.zeros(hw, dtype=np.uint8)
    arr = mask.detach().cpu().float().numpy()
    while arr.ndim > 2:
        arr = arr[0]
    return (arr > 0.1).astype(np.uint8) * 255


def bbox_from_points(points: np.ndarray, width: int, height: int) -> tuple[np.ndarray | None, np.ndarray]:
    valid = np.isfinite(points).all(axis=1)
    valid &= points[:, 0] >= 0
    valid &= points[:, 0] < width
    valid &= points[:, 1] >= 0
    valid &= points[:, 1] < height
    pts = points[valid]
    if len(pts) == 0:
        return None, valid
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    return np.asarray([lo[0], lo[1], hi[0], hi[1]], dtype=np.float32), valid


def bbox_from_mask(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return np.asarray([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def bbox_iou(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1 + 1.0) * max(0.0, y2 - y1 + 1.0)
    area_a = max(0.0, float(a[2] - a[0] + 1.0)) * max(0.0, float(a[3] - a[1] + 1.0))
    area_b = max(0.0, float(b[2] - b[0] + 1.0)) * max(0.0, float(b[3] - b[1] + 1.0))
    denom = area_a + area_b - inter
    return None if denom <= 0 else inter / denom


def draw_bbox(img: np.ndarray, bbox: np.ndarray | None, color: tuple[int, int, int], label: str) -> None:
    if bbox is None:
        return
    x1, y1, x2, y2 = [int(round(x)) for x in bbox]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def draw_overlay(img_bgr: np.ndarray, mask: np.ndarray, joints: np.ndarray, verts: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    height, width = img_bgr.shape[:2]
    overlay = img_bgr.copy()
    if mask.max() > 0:
        color_mask = np.zeros_like(overlay)
        color_mask[:, :, 1] = mask
        overlay = cv2.addWeighted(overlay, 1.0, color_mask, 0.28, 0)

    points_for_bbox = joints
    if verts is not None and len(verts) > 0:
        step = max(1, len(verts) // 1800)
        verts_draw = verts[::step]
        valid = np.isfinite(verts_draw).all(axis=1)
        valid &= verts_draw[:, 0] >= 0
        valid &= verts_draw[:, 0] < width
        valid &= verts_draw[:, 1] >= 0
        valid &= verts_draw[:, 1] < height
        for x, y in verts_draw[valid].astype(int):
            cv2.circle(overlay, (int(x), int(y)), 1, (255, 170, 0), -1)
        points_for_bbox = verts

    joint_bbox, joint_valid = bbox_from_points(joints, width, height)
    smpl_bbox, smpl_valid = bbox_from_points(points_for_bbox, width, height)
    mask_bbox = bbox_from_mask(mask)
    draw_bbox(overlay, mask_bbox, (0, 255, 0), "mask")
    draw_bbox(overlay, smpl_bbox, (0, 0, 255), "smpl")
    for idx, (x, y) in enumerate(joints):
        if not np.isfinite([x, y]).all():
            continue
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(overlay, (int(round(x)), int(round(y))), 3, (0, 0, 255), -1)
            if idx in {0, 1, 2, 5, 8, 10, 11, 12, 15, 19, 20, 21, 22}:
                cv2.putText(overlay, str(idx), (int(x) + 3, int(y) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 255), 1)

    metrics = {
        "mask_bbox": None if mask_bbox is None else mask_bbox.tolist(),
        "smpl_bbox": None if smpl_bbox is None else smpl_bbox.tolist(),
        "joint_bbox": None if joint_bbox is None else joint_bbox.tolist(),
        "bbox_iou_mask_smpl": bbox_iou(mask_bbox, smpl_bbox),
        "visible_joint_ratio": float(joint_valid.mean()) if len(joint_valid) else 0.0,
        "visible_smpl_point_ratio": float(smpl_valid.mean()) if len(smpl_valid) else 0.0,
    }
    return overlay, metrics


def camera_angle_deg(pose_a: np.ndarray, pose_b: np.ndarray) -> float:
    da = pose_a[:3, 2] / np.linalg.norm(pose_a[:3, 2])
    db = pose_b[:3, 2] / np.linalg.norm(pose_b[:3, 2])
    return float(np.degrees(np.arccos(np.clip(float(np.dot(da, db)), -1.0, 1.0))))


def clone_views(views):
    cloned = []
    for view in views:
        new_view = {}
        for key, value in view.items():
            if torch.is_tensor(value):
                new_view[key] = value.clone()
            else:
                new_view[key] = value
        cloned.append(new_view)
    return cloned


def transform_root_pose_to_camera(views) -> None:
    """Legacy diagnostic: force root pose to camera coordinates in-place.

    This is intentionally kept only as a wrong-path comparison.  For THUman,
    SMPL-X must be generated in world space and then transformed to camera
    space as a mesh/joint cloud.
    """
    for view in views:
        if "smplx_root_pose" not in view or "camera_pose" not in view:
            continue
        root = view["smplx_root_pose"]
        pose = view["camera_pose"]
        # root: [B, max_humans, 1, 3], pose: [B, 4, 4]
        flat_root = root.reshape(-1, 3)
        root_mat = roma.rotvec_to_rotmat(flat_root)
        R_w2c = pose[:, :3, :3].transpose(-1, -2)
        expand_shape = root.shape[:-2]
        R_w2c = R_w2c[:, None].expand(*expand_shape, 3, 3).reshape(-1, 3, 3)
        root_cam = roma.rotmat_to_rotvec(R_w2c @ root_mat).reshape_as(root)
        view["smplx_root_pose"] = root_cam


def process_views_with_mode(views, smpl_model: SMPLModel, mode: str):
    work = clone_views(views)
    if mode == "legacy_camera_root_wrong":
        transform_root_pose_to_camera(work)
    elif mode != "loader_current":
        raise ValueError(mode)
    smpl_model.update_smpl_gt(work)
    return work


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = parse_samples(args.samples)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    dataset = AvatarReX_AABB(
        split=args.split,
        ROOT=str(args.training_root),
        resolution=tuple(args.resolution),
        num_views=4,
        aug_crop=0,
        allow_repeat=True,
        seed=101,
        n_corres=0,
        fixed_samples=samples,
        load_da3_depth=False,
        raw_calibration_root=None,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    smpl_model = SMPLModel(
        device,
        model_args={"patch_size": 16, "mhmr_img_res": 896, "bb_patch_size": 14},
    )

    summary: dict = {
        "purpose": "THUman coordinate check through AvatarReX_AABB dataloader and SMPLModel.update_smpl_gt",
        "training_root": str(args.training_root / args.split),
        "samples": [],
        "coordinate_rule": {
            "camera_pose": "stored c2w from THUman calibration X_cam = R_w2c @ X_world + T_w2c",
            "loader_current": "THUman SMPL root/transl stay in world coordinates; SMPLModel transforms generated mesh/joints to camera",
            "legacy_camera_root_wrong": "diagnostic only: forcing root/transl into camera before SMPL-X is not equivalent and should not be used",
            "raw_camera_pose": "None for current THUman conversion unless a raw_calibration source is added",
        },
    }

    for sample_idx, views in enumerate(loader):
        sample = samples[sample_idx]
        views = todevice(views, device)
        sample_dir = args.output_dir / f"sample_{sample_idx:02d}_{sample[0].replace('/', '_')}_{sample[1].replace('/', '_')}_{sample[2]}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_record = {"sample": list(sample), "frames": []}

        poses = [view["camera_pose"][0].detach().cpu().numpy().astype(np.float32) for view in views]
        sample_record["view_angle_0_to_2_deg"] = camera_angle_deg(poses[0], poses[2])
        sample_record["camera_centers"] = [pose[:3, 3].tolist() for pose in poses]
        sample_record["camera_z_axes"] = [pose[:3, 2].tolist() for pose in poses]

        mode_views = {
            mode: process_views_with_mode(views, smpl_model, mode)
            for mode in ("loader_current", "legacy_camera_root_wrong")
        }

        for view_idx in range(len(views)):
            view = mode_views["loader_current"][view_idx]
            img = tensor_image_to_bgr(view["img"][0])
            mask = mask_to_uint8(view.get("msk", None), img.shape[:2])
            combined = []
            mode_metrics = {}
            for mode, processed_views in mode_views.items():
                mode_view = processed_views[view_idx]
                smpl_mask = mode_view["smpl_mask"][0].detach().cpu().bool().numpy()
                if not smpl_mask.any():
                    out = img.copy()
                    metrics = {"has_smpl": False}
                else:
                    human_idx = int(np.where(smpl_mask)[0][0])
                    joints = mode_view["smpl_j2d"][0, human_idx].detach().cpu().numpy().astype(np.float32)
                    verts = None
                    if "smpl_v2d" in mode_view:
                        verts = mode_view["smpl_v2d"][0, human_idx].detach().cpu().numpy().astype(np.float32)
                    out, metrics = draw_overlay(img, mask, joints, verts)
                    metrics["has_smpl"] = True
                    metrics["smpl_mask_index"] = human_idx
                    j3d = mode_view["smpl_j3d"][0, human_idx].detach().cpu().numpy()
                    metrics["mean_smpl_cam_z"] = float(np.nanmean(j3d[:, 2]))
                    metrics["min_smpl_cam_z"] = float(np.nanmin(j3d[:, 2]))
                cv2.putText(out, mode, (8, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                combined.append(out)
                mode_metrics[mode] = metrics

            label = view.get("label", [""])[0] if isinstance(view.get("label"), list) else str(view.get("label", ""))
            out = np.concatenate(combined, axis=1)
            cv2.putText(out, f"sample {sample_idx} view {view_idx} {label}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(out, "green=mask red=SMPL bbox red dots=joints orange=vertices", (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            out_path = sample_dir / f"view_{view_idx:02d}_overlay.png"
            cv2.imwrite(str(out_path), out)
            sample_record["frames"].append(
                {
                    "view_idx": view_idx,
                    "label": label,
                    "overlay": str(out_path),
                    "camera_center": poses[view_idx][:3, 3].tolist(),
                    "camera_z_axis": poses[view_idx][:3, 2].tolist(),
                    "modes": mode_metrics,
                }
            )

        summary["samples"].append(sample_record)

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "summary": str(summary_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
