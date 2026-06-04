#!/usr/bin/env python3
"""Diagnose THUman SMPL projection before using it for V8 training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import smplx


SMPLX_DIR = Path(__file__).resolve().parents[1] / "src" / "models"


def load_calibration(subject_root: Path, cam: str) -> dict:
    with (subject_root / "calibration.json").open("r", encoding="utf-8") as f:
        calibration = json.load(f)
    return calibration[cam]


def bbox_from_mask(mask: np.ndarray) -> np.ndarray | None:
    if mask.ndim == 3:
        mask = mask.max(axis=2)
    ys, xs = np.where(mask > 10)
    if len(xs) == 0:
        return None
    return np.asarray([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def bbox_from_points(points: np.ndarray, width: int, height: int) -> np.ndarray | None:
    valid = np.isfinite(points).all(axis=1)
    valid &= points[:, 0] >= 0
    valid &= points[:, 0] < width
    valid &= points[:, 1] >= 0
    valid &= points[:, 1] < height
    pts = points[valid]
    if len(pts) == 0:
        return None
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    return np.asarray([lo[0], lo[1], hi[0], hi[1]], dtype=np.float32)


def bbox_iou(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1 + 1.0) * max(0.0, y2 - y1 + 1.0)
    area_a = max(0.0, float(a[2] - a[0] + 1.0)) * max(0.0, float(a[3] - a[1] + 1.0))
    area_b = max(0.0, float(b[2] - b[0] + 1.0)) * max(0.0, float(b[3] - b[1] + 1.0))
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0 else inter / denom


def draw_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    pts: np.ndarray,
    mask_bbox: np.ndarray | None,
    smpl_bbox: np.ndarray | None,
    title: str,
) -> np.ndarray:
    out = image.copy()
    if mask.ndim == 3:
        mask_u8 = mask.max(axis=2)
    else:
        mask_u8 = mask
    green = np.zeros_like(out)
    green[:, :, 1] = (mask_u8 > 10).astype(np.uint8) * 255
    out = cv2.addWeighted(out, 1.0, green, 0.25, 0)

    h, w = out.shape[:2]
    step = max(1, len(pts) // 2200)
    draw_pts = pts[::step]
    valid = np.isfinite(draw_pts).all(axis=1)
    valid &= draw_pts[:, 0] >= 0
    valid &= draw_pts[:, 0] < w
    valid &= draw_pts[:, 1] >= 0
    valid &= draw_pts[:, 1] < h
    for x, y in draw_pts[valid].astype(int):
        cv2.circle(out, (int(x), int(y)), 1, (0, 0, 255), -1)

    if mask_bbox is not None:
        x1, y1, x2, y2 = [int(round(x)) for x in mask_bbox]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, "mask", (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    if smpl_bbox is not None:
        x1, y1, x2, y2 = [int(round(x)) for x in smpl_bbox]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(out, "smpl", (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    cv2.putText(out, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(out, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def rotvec_to_mat(rotvec: np.ndarray) -> np.ndarray:
    mat, _ = cv2.Rodrigues(rotvec.reshape(3).astype(np.float32))
    return mat.astype(np.float32)


def mat_to_rotvec(mat: np.ndarray) -> np.ndarray:
    rotvec, _ = cv2.Rodrigues(mat.astype(np.float32))
    return rotvec.reshape(3).astype(np.float32)


def make_smplx_vertices(model, smpl_data: dict[str, np.ndarray], frame_idx: int, device: torch.device) -> np.ndarray:
    betas = np.concatenate([smpl_data["betas"][0], [0.0]]).astype(np.float32)
    kwargs = {
        "global_orient": torch.from_numpy(smpl_data["global_orient"][frame_idx : frame_idx + 1]).to(device),
        "body_pose": torch.from_numpy(smpl_data["body_pose"][frame_idx : frame_idx + 1, : 63]).to(device),
        "jaw_pose": torch.from_numpy(smpl_data["jaw_pose"][frame_idx : frame_idx + 1]).to(device),
        "leye_pose": torch.zeros((1, 3), dtype=torch.float32, device=device),
        "reye_pose": torch.zeros((1, 3), dtype=torch.float32, device=device),
        "left_hand_pose": torch.from_numpy(smpl_data["left_hand_pose"][frame_idx : frame_idx + 1, : 45]).to(device),
        "right_hand_pose": torch.from_numpy(smpl_data["right_hand_pose"][frame_idx : frame_idx + 1, : 45]).to(device),
        "betas": torch.from_numpy(betas[None]).to(device),
        "transl": torch.from_numpy(smpl_data["transl"][frame_idx : frame_idx + 1]).to(device),
        "expression": torch.zeros((1, 10), dtype=torch.float32, device=device),
    }
    with torch.no_grad():
        out = model(**kwargs)
    return out.vertices[0].detach().cpu().numpy().astype(np.float32)


def project_points(
    verts_world: np.ndarray,
    R_w2c: np.ndarray,
    T_w2c: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray | None,
    world_delta: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    points_world = verts_world
    if world_delta is not None:
        points_world = points_world + world_delta.reshape(1, 3).astype(np.float32)
    points_cam = points_world @ R_w2c.T + T_w2c.reshape(1, 3)
    if dist is None:
        z = np.maximum(points_cam[:, 2:3], 1e-6)
        uv = points_cam[:, :2] / z
        uv = uv @ K[:2, :2].T + K[:2, 2]
        return uv.astype(np.float32), points_cam
    uv, _ = cv2.projectPoints(
        points_cam.astype(np.float32),
        np.zeros(3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        K.astype(np.float32),
        dist.astype(np.float32),
    )
    return uv.reshape(-1, 2).astype(np.float32), points_cam


def center_delta_from_bbox(mask_bbox: np.ndarray | None, smpl_bbox: np.ndarray | None, mean_z: float, K: np.ndarray) -> np.ndarray:
    if mask_bbox is None or smpl_bbox is None:
        return np.zeros(3, dtype=np.float32)
    mask_c = np.asarray([(mask_bbox[0] + mask_bbox[2]) * 0.5, (mask_bbox[1] + mask_bbox[3]) * 0.5], dtype=np.float32)
    smpl_c = np.asarray([(smpl_bbox[0] + smpl_bbox[2]) * 0.5, (smpl_bbox[1] + smpl_bbox[3]) * 0.5], dtype=np.float32)
    delta_uv = mask_c - smpl_c
    return np.asarray([delta_uv[0] * mean_z / K[0, 0], delta_uv[1] * mean_z / K[1, 1], 0.0], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default="subject00")
    parser.add_argument("--cam", default="cam04")
    parser.add_argument("--frame", type=int, default=2)
    parser.add_argument("--input_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data/THUman"))
    parser.add_argument("--output_dir", type=Path, default=Path("output/v8_thuman_smpl_projection_debug"))
    parser.add_argument("--offset_radius", type=int, default=20)
    args = parser.parse_args()

    subject_root = args.input_root / args.subject
    out_dir = args.output_dir / f"{args.subject}_{args.cam}_{args.frame:08d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(subject_root / "images" / args.cam / f"{args.frame:08d}.jpg"), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(subject_root / "masks" / args.cam / f"{args.frame:08d}.jpg"), cv2.IMREAD_COLOR)
    if image is None or mask is None:
        raise FileNotFoundError(f"missing image or mask for {args.subject}/{args.cam}/{args.frame:08d}")

    cal = load_calibration(subject_root, args.cam)
    R_w2c = np.asarray(cal["R"], dtype=np.float32).reshape(3, 3)
    T_w2c = np.asarray(cal["T"], dtype=np.float32).reshape(3)
    K = np.asarray(cal["K"], dtype=np.float32).reshape(3, 3)
    dist = np.asarray(cal.get("distCoeff", []), dtype=np.float32).reshape(-1)
    smpl_data = {k: v.copy() for k, v in np.load(subject_root / "smpl_params.npz").items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smplx.create(
        str(SMPLX_DIR),
        "smplx",
        gender="neutral",
        use_pca=False,
        flat_hand_mean=True,
        num_betas=11,
    ).to(device).eval()

    h, w = image.shape[:2]
    mask_bbox = bbox_from_mask(mask)
    frame_records = []
    best_offset = None
    for offset in range(-args.offset_radius, args.offset_radius + 1):
        frame_idx = int(np.clip(args.frame + offset, 0, smpl_data["global_orient"].shape[0] - 1))
        verts = make_smplx_vertices(model, smpl_data, frame_idx, device)
        uv, cam = project_points(verts, R_w2c, T_w2c, K, None)
        bbox = bbox_from_points(uv, w, h)
        iou = bbox_iou(mask_bbox, bbox)
        frame_records.append(
            {
                "smpl_frame": frame_idx,
                "frame_offset": offset,
                "bbox_iou": iou,
                "smpl_bbox": None if bbox is None else bbox.tolist(),
                "mean_cam_z": float(np.nanmean(cam[:, 2])),
            }
        )
        if best_offset is None or iou > best_offset["bbox_iou"]:
            best_offset = frame_records[-1]

    variants = []
    verts = make_smplx_vertices(model, smpl_data, args.frame, device)
    for name, dist_arg in (("pinhole", None), ("with_distortion", dist if len(dist) else None)):
        uv, cam = project_points(verts, R_w2c, T_w2c, K, dist_arg)
        bbox = bbox_from_points(uv, w, h)
        variants.append(
            {
                "name": name,
                "bbox_iou": bbox_iou(mask_bbox, bbox),
                "smpl_bbox": None if bbox is None else bbox.tolist(),
                "mean_cam_z": float(np.nanmean(cam[:, 2])),
                "center_delta_cam": center_delta_from_bbox(mask_bbox, bbox, float(np.nanmean(cam[:, 2])), K).tolist(),
            }
        )
        cv2.imwrite(
            str(out_dir / f"{name}.png"),
            draw_overlay(image, mask, uv, mask_bbox, bbox, f"{args.subject}/{args.cam} f{args.frame} {name} IoU={variants[-1]['bbox_iou']:.3f}"),
        )

    # Apply a camera-space center correction converted to world coordinates. This
    # is diagnostic only, used to see whether a simple translation explains the mismatch.
    uv, cam = project_points(verts, R_w2c, T_w2c, K, None)
    bbox = bbox_from_points(uv, w, h)
    delta_cam = center_delta_from_bbox(mask_bbox, bbox, float(np.nanmean(cam[:, 2])), K)
    delta_world = R_w2c.T @ delta_cam
    uv_shifted, cam_shifted = project_points(verts, R_w2c, T_w2c, K, None, world_delta=delta_world)
    bbox_shifted = bbox_from_points(uv_shifted, w, h)
    shifted = {
        "name": "center_shift_diagnostic",
        "bbox_iou": bbox_iou(mask_bbox, bbox_shifted),
        "smpl_bbox": None if bbox_shifted is None else bbox_shifted.tolist(),
        "delta_cam": delta_cam.tolist(),
        "delta_world": delta_world.tolist(),
        "mean_cam_z": float(np.nanmean(cam_shifted[:, 2])),
    }
    cv2.imwrite(
        str(out_dir / "center_shift_diagnostic.png"),
        draw_overlay(image, mask, uv_shifted, mask_bbox, bbox_shifted, f"center shift IoU={shifted['bbox_iou']:.3f}"),
    )

    summary = {
        "subject": args.subject,
        "cam": args.cam,
        "frame": args.frame,
        "image_shape": list(image.shape),
        "mask_bbox": None if mask_bbox is None else mask_bbox.tolist(),
        "calibration": {
            "K": K.tolist(),
            "R_w2c": R_w2c.tolist(),
            "T_w2c": T_w2c.tolist(),
            "distCoeff": dist.tolist(),
        },
        "variants": variants,
        "center_shift_diagnostic": shifted,
        "best_frame_offset": best_offset,
        "frame_offset_scan": frame_records,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), "summary": str(out_dir / "summary.json")}, indent=2))


if __name__ == "__main__":
    main()
