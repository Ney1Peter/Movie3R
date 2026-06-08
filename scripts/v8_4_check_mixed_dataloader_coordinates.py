#!/usr/bin/env python3
"""Check V8.4 mixed AABB/AAAA dataloader SMPL/camera coordinates.

This script uses the same dataloader path as training:
  dataloader view dict -> SMPLModel.update_smpl_gt -> projected SMPL overlay.

If SMPL/camera coordinates are wrong, the orange vertices/red joints will not
line up with the person/mask in the resized training image.
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
    parser.add_argument("--split", default="Test/v8_4_mixed_aabb_aaaa")
    parser.add_argument(
        "--manifest_root",
        type=Path,
        default=Path("output/v8_4_mixed_aabb_aaaa_manifests_no_zxc"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output/v8_4_mixed_dataloader_coordinate_check"),
    )
    parser.add_argument("--resize_mode", default="human3r_demo")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def choose_aabb(records: list[dict]) -> dict:
    # This record was selected once because all four AABB views keep visible
    # SMPL after the training crop and SMPLModel visibility filter.
    for record in records:
        if (
            record.get("seqA") == "thuman02/cam17"
            and record.get("seqB") == "thuman02/cam00"
            and int(record.get("start_frame", -1)) == 3073
        ):
            return record
    for record in records:
        if str(record.get("group", "")).startswith("thuman"):
            return record
    return records[0]


def choose_aaaa(records: list[dict]) -> dict:
    for record in records:
        if str(record.get("group", "")).startswith("thuman"):
            return record
    return records[0]


def tensor_image_to_bgr(img: torch.Tensor) -> np.ndarray:
    arr = img.detach().cpu().float().permute(1, 2, 0).numpy()
    arr = ((arr * 0.5 + 0.5).clip(0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def mask_to_uint8(mask: torch.Tensor | None, hw: tuple[int, int]) -> np.ndarray:
    if mask is None or isinstance(mask, bool):
        return np.zeros(hw, dtype=np.uint8)
    arr = mask.detach().cpu().float().numpy()
    while arr.ndim > 2:
        arr = arr[0]
    return (arr > 0.1).astype(np.uint8) * 255


def bbox_from_mask(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


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
    return np.array([lo[0], lo[1], hi[0], hi[1]], dtype=np.float32), valid


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


def draw_bbox(img: np.ndarray, bbox: np.ndarray | None, color: tuple[int, int, int], text: str) -> None:
    if bbox is None:
        return
    x1, y1, x2, y2 = [int(round(x)) for x in bbox]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, text, (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def camera_angle_deg(pose_a: np.ndarray, pose_b: np.ndarray) -> float:
    da = pose_a[:3, 2] / max(np.linalg.norm(pose_a[:3, 2]), 1e-12)
    db = pose_b[:3, 2] / max(np.linalg.norm(pose_b[:3, 2]), 1e-12)
    return float(np.degrees(np.arccos(np.clip(float(np.dot(da, db)), -1.0, 1.0))))


def relative_pose_stats(poses: list[np.ndarray]) -> dict:
    centers = [pose[:3, 3] for pose in poses]
    return {
        "centers": [center.tolist() for center in centers],
        "view0_to_view2_angle_deg": camera_angle_deg(poses[0], poses[2]),
        "view0_to_view2_center_dist": float(np.linalg.norm(centers[2] - centers[0])),
        "step_center_dists": [
            float(np.linalg.norm(centers[i] - centers[i - 1])) for i in range(1, len(centers))
        ],
    }


def rotation_error_deg(pose_a: np.ndarray, pose_b: np.ndarray) -> float:
    r = pose_a[:3, :3].T @ pose_b[:3, :3]
    cos = (np.trace(r) - 1.0) * 0.5
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def pose_diff_stats(camera_poses: list[np.ndarray], raw_poses: list[np.ndarray | None]) -> list[dict]:
    out = []
    for cam, raw in zip(camera_poses, raw_poses):
        if raw is None:
            out.append({"has_raw_camera_pose": False})
            continue
        out.append(
            {
                "has_raw_camera_pose": True,
                "camera_vs_raw_center_dist": float(np.linalg.norm(cam[:3, 3] - raw[:3, 3])),
                "camera_vs_raw_rot_deg": rotation_error_deg(cam, raw),
            }
        )
    return out


def draw_overlay(
    img: np.ndarray,
    mask: np.ndarray,
    joints: np.ndarray,
    verts: np.ndarray,
    title_lines: list[str],
) -> tuple[np.ndarray, dict]:
    h, w = img.shape[:2]
    out = img.copy()
    if mask.max() > 0:
        color = np.zeros_like(out)
        color[:, :, 1] = mask
        out = cv2.addWeighted(out, 1.0, color, 0.25, 0.0)

    step = max(1, len(verts) // 1700)
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
    for idx, (x, y) in enumerate(joints):
        if valid_j[idx]:
            cv2.circle(out, (int(round(x)), int(round(y))), 3, (0, 0, 255), -1)

    mask_bbox = bbox_from_mask(mask)
    smpl_bbox, valid_all_v = bbox_from_points(verts, w, h)
    draw_bbox(out, mask_bbox, (0, 255, 0), "mask")
    draw_bbox(out, smpl_bbox, (0, 0, 255), "smpl")

    panel = out.copy()
    shade = panel.copy()
    cv2.rectangle(shade, (0, 0), (w, 66), (0, 0, 0), -1)
    panel = cv2.addWeighted(shade, 0.58, panel, 0.42, 0.0)
    y = 18
    for line in title_lines:
        cv2.putText(panel, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
        y += 19

    metrics = {
        "bbox_iou_mask_smpl": bbox_iou(mask_bbox, smpl_bbox),
        "visible_joint_ratio": float(valid_j.mean()) if len(valid_j) else 0.0,
        "visible_vertex_ratio": float(valid_all_v.mean()) if len(valid_all_v) else 0.0,
    }
    return panel, metrics


def run_one_clip(
    name: str,
    dataset,
    output_dir: Path,
    device: torch.device,
    smpl_model: SMPLModel,
) -> dict:
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    views = next(iter(loader))
    views = todevice(views, device)

    camera_poses = [view["camera_pose"][0].detach().cpu().numpy().astype(np.float32) for view in views]
    raw_poses = [
        None if "raw_camera_pose" not in view else view["raw_camera_pose"][0].detach().cpu().numpy().astype(np.float32)
        for view in views
    ]
    labels = [
        view.get("label", [""])[0] if isinstance(view.get("label", ""), list) else str(view.get("label", ""))
        for view in views
    ]
    shot_labels = [int(view["shot_label"][0].detach().cpu().item()) for view in views]
    world_flags = [bool(view["human_params_are_world"][0].detach().cpu().item()) for view in views]

    smpl_model.update_smpl_gt(views)

    panels = []
    frame_metrics = []
    for view_idx, view in enumerate(views):
        img = tensor_image_to_bgr(view["img"][0])
        mask = mask_to_uint8(view.get("msk", None), img.shape[:2])
        smpl_mask = view["smpl_mask"][0].detach().cpu().bool().numpy()
        if not smpl_mask.any():
            panel = img
            metrics = {"has_smpl": False}
        else:
            human_idx = int(np.where(smpl_mask)[0][0])
            joints = view["smpl_j2d"][0, human_idx].detach().cpu().numpy().astype(np.float32)
            verts = view["smpl_v2d"][0, human_idx].detach().cpu().numpy().astype(np.float32)
            j3d = view["smpl_j3d"][0, human_idx].detach().cpu().numpy().astype(np.float32)
            title = [
                f"v{view_idx} shot={shot_labels[view_idx]} world_smpl={int(world_flags[view_idx])}",
                labels[view_idx],
                "green mask | orange verts | red joints/bbox",
            ]
            panel, metrics = draw_overlay(img, mask, joints, verts, title)
            metrics.update(
                {
                    "has_smpl": True,
                    "human_idx": human_idx,
                    "mean_smpl_cam_z": float(np.nanmean(j3d[:, 2])),
                    "min_smpl_cam_z": float(np.nanmin(j3d[:, 2])),
                }
            )
        panels.append(panel)
        frame_metrics.append({"view_idx": view_idx, "label": labels[view_idx], **metrics})

    max_h = max(panel.shape[0] for panel in panels)
    padded = []
    for panel in panels:
        if panel.shape[0] < max_h:
            pad = np.zeros((max_h - panel.shape[0], panel.shape[1], 3), dtype=np.uint8)
            panel = np.concatenate([panel, pad], axis=0)
        padded.append(panel)

    sheet = np.concatenate(padded, axis=1)
    header = np.zeros((82, sheet.shape[1], 3), dtype=np.uint8)
    raw_stats = relative_pose_stats([pose if pose is not None else camera_poses[i] for i, pose in enumerate(raw_poses)])
    cam_stats = relative_pose_stats(camera_poses)
    cv2.putText(header, f"{name}: dataloader coordinate check", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(header, f"camera_pose v0-v2 angle={cam_stats['view0_to_view2_angle_deg']:.1f}, raw_pose v0-v2 angle={raw_stats['view0_to_view2_angle_deg']:.1f}", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (190, 220, 255), 1, cv2.LINE_AA)
    output = np.concatenate([header, sheet], axis=0)
    output_path = output_dir / f"{name}_overlay.png"
    cv2.imwrite(str(output_path), output)

    return {
        "name": name,
        "overlay": str(output_path),
        "labels": labels,
        "shot_labels": shot_labels,
        "human_params_are_world": world_flags,
        "camera_pose_stats": cam_stats,
        "raw_pose_stats": raw_stats,
        "camera_vs_raw_pose": pose_diff_stats(camera_poses, raw_poses),
        "frames": frame_metrics,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    raw_roots = {
        "lbn1": "/data/wangzheng/iJCV-CODE/data/AvatarReX_raw_meta/lbn1",
        "zzr": "/data/wangzheng/iJCV-CODE/data/AvatarReX_raw_meta/zzr",
    }
    aabb_record = choose_aabb(load_jsonl(args.manifest_root / "test_aabb_no_zxc.jsonl"))
    aaaa_record = choose_aaaa(load_jsonl(args.manifest_root / "test_aaaa_no_zxc.jsonl"))

    aabb_dataset = AvatarReX_AABB(
        split=args.split,
        ROOT=str(args.data_root),
        resolution=512,
        resize_mode=args.resize_mode,
        num_views=4,
        aug_crop=0,
        allow_repeat=True,
        seed=101,
        n_corres=0,
        fixed_samples=[(aabb_record["seqA"], aabb_record["seqB"], int(aabb_record["start_frame"]))],
        load_da3_depth=False,
        raw_calibration_root=raw_roots,
    )
    aaaa_dataset = AvatarReX_Video(
        split=args.split,
        ROOT=str(args.data_root),
        resolution=512,
        resize_mode=args.resize_mode,
        num_views=4,
        aug_crop=0,
        allow_repeat=True,
        seed=101,
        n_corres=0,
        fixed_samples=[(aaaa_record["seq"], int(aaaa_record["start_frame"]))],
        load_da3_depth=False,
        raw_calibration_root=raw_roots,
    )

    smpl_model = SMPLModel(
        device,
        model_args={"patch_size": 16, "mhmr_img_res": 896, "bb_patch_size": 14},
    )
    summary = {
        "purpose": "Verify V8.4 mixed dataloader SMPL/camera coordinates before GT metric training.",
        "data_root": str(args.data_root),
        "split": args.split,
        "selected_records": {"aabb": aabb_record, "aaaa": aaaa_record},
        "checks": [
            run_one_clip("test_aabb", aabb_dataset, args.output_dir, device, smpl_model),
            run_one_clip("test_aaaa", aaaa_dataset, args.output_dir, device, smpl_model),
        ],
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
