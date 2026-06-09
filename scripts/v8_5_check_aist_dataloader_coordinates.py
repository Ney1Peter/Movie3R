#!/usr/bin/env python3
"""Check AIST/ASIT samples through the actual Movie3R dataloader path.

This verifies the training path, not only the raw dataset convention:
  converted Training/asit files -> AvatarReX dataloader resize -> SMPLModel.update_smpl_gt

If the AIST native SMPL/camera convention is wired correctly, the projected
orange vertices and red joints should align with the person in the resized
Human3R-demo input images.

Note:
  V8.6 mainline currently disables ASIT/native-SMPL training and uses only
  AvatarReX + THuman SMPL-X data. This diagnostic is kept for the future ASIT
  re-enable step and requires the native-SMPL dataloader/model branch to be
  restored first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from dust3r.datasets.avatarrex import AvatarReX_AABB, AvatarReX_Video
from dust3r.smpl_model import SMPLModel
from dust3r.utils.device import todevice


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data"))
    parser.add_argument("--split", default="Training")
    parser.add_argument("--seq_a", default="asit/gBR_sBM_cAll_d04_mBR0_ch01/c01")
    parser.add_argument("--seq_b", default="asit/gBR_sBM_cAll_d04_mBR0_ch01/c09")
    parser.add_argument("--video_seq", default="asit/gBR_sBM_cAll_d04_mBR0_ch01/c01")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--resize_mode", default="human3r_demo")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output/v8_5_aist_dataloader_coordinate_check"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def tensor_image_to_bgr(img: torch.Tensor) -> np.ndarray:
    arr = img.detach().cpu().float().permute(1, 2, 0).numpy()
    arr = ((arr * 0.5 + 0.5).clip(0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def draw_points(
    img: np.ndarray,
    joints: np.ndarray,
    verts: np.ndarray,
    title_lines: list[str],
) -> tuple[np.ndarray, dict]:
    h, w = img.shape[:2]
    out = img.copy()

    step = max(1, len(verts) // 1600)
    verts_draw = verts[::step]
    valid_v = np.isfinite(verts_draw).all(axis=1)
    valid_v &= verts_draw[:, 0] >= 0
    valid_v &= verts_draw[:, 0] < w
    valid_v &= verts_draw[:, 1] >= 0
    valid_v &= verts_draw[:, 1] < h
    for x, y in verts_draw[valid_v].astype(np.int32):
        cv2.circle(out, (int(x), int(y)), 1, (0, 170, 255), -1)

    valid_j = np.isfinite(joints).all(axis=1)
    valid_j &= joints[:, 0] >= 0
    valid_j &= joints[:, 0] < w
    valid_j &= joints[:, 1] >= 0
    valid_j &= joints[:, 1] < h
    for x, y in joints[valid_j]:
        cv2.circle(out, (int(round(x)), int(round(y))), 3, (0, 0, 255), -1)

    shade = out.copy()
    cv2.rectangle(shade, (0, 0), (w, 72), (0, 0, 0), -1)
    out = cv2.addWeighted(shade, 0.55, out, 0.45, 0.0)
    y = 18
    for line in title_lines:
        cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
        y += 18

    return out, {
        "visible_joint_ratio": float(valid_j.mean()) if len(valid_j) else 0.0,
        "visible_vertex_ratio_sampled": float(valid_v.mean()) if len(valid_v) else 0.0,
    }


def camera_angle_deg(pose_a: np.ndarray, pose_b: np.ndarray) -> float:
    da = pose_a[:3, 2] / max(np.linalg.norm(pose_a[:3, 2]), 1e-12)
    db = pose_b[:3, 2] / max(np.linalg.norm(pose_b[:3, 2]), 1e-12)
    return float(np.degrees(np.arccos(np.clip(float(np.dot(da, db)), -1.0, 1.0))))


def run_dataset(name: str, dataset, output_dir: Path, device: torch.device, smpl_model: SMPLModel) -> dict:
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    views = next(iter(loader))
    views = todevice(views, device)
    smpl_model.update_smpl_gt(views)

    panels = []
    frame_metrics = []
    camera_poses = []
    for view_idx, view in enumerate(views):
        img = tensor_image_to_bgr(view["img"][0])
        camera_poses.append(view["camera_pose"][0].detach().cpu().numpy().astype(np.float32))
        smpl_mask = view["smpl_mask"][0].detach().cpu().bool().numpy()
        label = view.get("label", [""])[0] if isinstance(view.get("label", ""), list) else str(view.get("label", ""))
        shot_label = int(view["shot_label"][0].detach().cpu().item())
        world_flag = bool(view["human_params_are_world"][0].detach().cpu().item())
        if not smpl_mask.any():
            panel = img
            metrics = {"has_smpl": False}
        else:
            human_idx = int(np.where(smpl_mask)[0][0])
            joints = view["smpl_j2d"][0, human_idx].detach().cpu().numpy().astype(np.float32)
            verts = view["smpl_v2d"][0, human_idx].detach().cpu().numpy().astype(np.float32)
            j3d = view["smpl_j3d"][0, human_idx].detach().cpu().numpy().astype(np.float32)
            panel, metrics = draw_points(
                img,
                joints,
                verts,
                [
                    f"{name} v{view_idx} shot={shot_label} world_smpl={int(world_flag)}",
                    label,
                    "orange vertices | red joints",
                ],
            )
            metrics.update(
                {
                    "has_smpl": True,
                    "mean_smpl_cam_z": float(np.nanmean(j3d[:, 2])),
                    "min_smpl_cam_z": float(np.nanmin(j3d[:, 2])),
                }
            )
        panels.append(panel)
        frame_metrics.append(
            {
                "view_idx": view_idx,
                "label": label,
                "shot_label": shot_label,
                "human_params_are_world": world_flag,
                **metrics,
            }
        )

    max_h = max(panel.shape[0] for panel in panels)
    padded = []
    for panel in panels:
        if panel.shape[0] < max_h:
            panel = np.concatenate(
                [panel, np.zeros((max_h - panel.shape[0], panel.shape[1], 3), dtype=np.uint8)],
                axis=0,
            )
        padded.append(panel)

    sheet = np.concatenate(padded, axis=1)
    header = np.zeros((78, sheet.shape[1], 3), dtype=np.uint8)
    angle = camera_angle_deg(camera_poses[0], camera_poses[2])
    center_dist = float(np.linalg.norm(camera_poses[2][:3, 3] - camera_poses[0][:3, 3]))
    cv2.putText(header, f"{name}: AIST dataloader coordinate check", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(header, f"v0-v2 camera angle={angle:.1f} deg, center dist={center_dist:.2f}", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (190, 220, 255), 1, cv2.LINE_AA)
    output = np.concatenate([header, sheet], axis=0)
    out_path = output_dir / f"{name}_overlay.png"
    cv2.imwrite(str(out_path), output)
    return {
        "name": name,
        "overlay": str(out_path),
        "camera_v0_to_v2_angle_deg": angle,
        "camera_v0_to_v2_center_dist": center_dist,
        "frames": frame_metrics,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    aabb = AvatarReX_AABB(
        split=args.split,
        ROOT=str(args.data_root),
        resolution=args.resolution,
        resize_mode=args.resize_mode,
        num_views=4,
        aug_crop=0,
        allow_repeat=True,
        seed=101,
        n_corres=0,
        fixed_samples=[(args.seq_a, args.seq_b, args.start_frame)],
        load_da3_depth=False,
        raw_calibration_root=None,
    )
    aaaa = AvatarReX_Video(
        split=args.split,
        ROOT=str(args.data_root),
        resolution=args.resolution,
        resize_mode=args.resize_mode,
        num_views=4,
        aug_crop=0,
        allow_repeat=True,
        seed=101,
        n_corres=0,
        fixed_samples=[(args.video_seq, args.start_frame)],
        load_da3_depth=False,
        raw_calibration_root=None,
    )
    smpl_model = SMPLModel(
        device,
        model_args={"patch_size": 16, "mhmr_img_res": 896, "bb_patch_size": 14},
    )

    summary = {
        "purpose": "AIST native SMPL + camera validation through the actual training dataloader.",
        "data_root": str(args.data_root),
        "split": args.split,
        "resize_mode": args.resize_mode,
        "checks": [
            run_dataset("asit_aabb", aabb, args.output_dir, device, smpl_model),
            run_dataset("asit_aaaa", aaaa, args.output_dir, device, smpl_model),
        ],
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
