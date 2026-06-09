#!/usr/bin/env python3
"""Check AIST camera/SMPL projection conventions on a few frames.

This is a diagnostic script. It does not convert data. It renders several
candidate coordinate formulas onto RGB frames so we can visually identify the
right convention before writing the actual preprocessor.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import smplx
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SMPL_MODEL_ROOT = REPO_ROOT / "src" / "models"


def parse_csv(value: str, cast=str):
    return [cast(v.strip()) for v in value.split(",") if v.strip()]


def load_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"failed to read frame {frame_idx} from {video_path}")
    return frame


def rodrigues(rotvec) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rotvec, dtype=np.float64).reshape(3, 1))
    return R.astype(np.float32)


def make_smpl_vertices(model, smpl_data: dict, frame_idx: int, device: torch.device) -> np.ndarray:
    pose = smpl_data["smpl_poses"][frame_idx].astype(np.float32)
    global_orient = torch.from_numpy(pose[:3]).reshape(1, 3).to(device)
    body_pose = torch.from_numpy(pose[3:]).reshape(1, 69).to(device)
    betas = torch.zeros((1, 10), dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model(global_orient=global_orient, body_pose=body_pose, betas=betas)
    return out.vertices[0].detach().cpu().numpy().astype(np.float32)


def project_points(points_cam: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = points_cam[:, 2]
    valid = np.isfinite(points_cam).all(axis=1) & (z > 1e-6)
    uv = np.full((len(points_cam), 2), np.nan, dtype=np.float32)
    proj = (points_cam[valid] / z[valid, None]) @ K.T
    uv[valid] = proj[:, :2]
    return uv[:, :2], valid


def draw_projection(image_bgr: np.ndarray, uv: np.ndarray, valid: np.ndarray, title: str) -> np.ndarray:
    out = image_bgr.copy()
    h, w = out.shape[:2]
    in_img = (
        valid
        & np.isfinite(uv).all(axis=1)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < w)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < h)
    )
    pts = np.round(uv[in_img]).astype(np.int32)
    if len(pts) > 0:
        for p in pts[:: max(1, len(pts) // 4000)]:
            cv2.circle(out, (int(p[0]), int(p[1])), 1, (255, 255, 255), -1, cv2.LINE_AA)
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 255), 2)
    ratio = float(in_img.mean()) if len(in_img) else 0.0
    cv2.putText(
        out,
        f"{title}  in_img={ratio:.3f}",
        (24, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def build_variants(verts_smpl: np.ndarray, scale: float, trans: np.ndarray, R: np.ndarray, T: np.ndarray):
    world_scale_then_trans = verts_smpl * scale + trans[None, :]
    world_trans_then_scale = (verts_smpl + trans[None, :]) * scale
    world_no_scale = verts_smpl + trans[None, :]

    variants = {}
    variants["A_scale_v_plus_trans__R_X_plus_T"] = world_scale_then_trans @ R.T + T[None, :]
    variants["B_scale_v_plus_trans__Rinv_X_minus_T"] = (world_scale_then_trans - T[None, :]) @ R
    variants["C_v_plus_trans_no_scale__R_X_plus_T"] = world_no_scale @ R.T + T[None, :]
    variants["D_scale_v_plus_scale_trans__R_X_plus_T"] = world_trans_then_scale @ R.T + T[None, :]
    variants["E_scale_v_plus_trans__R_X_plus_T_over100"] = world_scale_then_trans @ R.T + (T[None, :] / 100.0)
    return variants


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data/asit"))
    parser.add_argument("--sequence", default="gBR_sBM_cAll_d04_mBR0_ch01")
    parser.add_argument("--cams", default="c01,c05,c09")
    parser.add_argument("--frames", default="0,60,120")
    parser.add_argument("--output_dir", type=Path, default=Path("output/v8_5_aist_projection_check"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    seq_root = args.root / args.sequence
    with (seq_root / "gt" / "smpl.pkl").open("rb") as f:
        smpl_data = pickle.load(f)

    device = torch.device(args.device)
    model = smplx.create(str(SMPL_MODEL_ROOT), "smpl", gender="neutral").to(device)
    model.eval()

    out_root = args.output_dir / args.sequence
    out_root.mkdir(parents=True, exist_ok=True)

    summary = []
    for cam in parse_csv(args.cams, str):
        with (seq_root / "camera" / f"{cam}_camera.json").open("r", encoding="utf-8") as f:
            cam_data = json.load(f)
        K = np.asarray(cam_data["matrix"], dtype=np.float32)
        R = rodrigues(cam_data["rotation"])
        T = np.asarray(cam_data["translation"], dtype=np.float32).reshape(3)
        video_path = seq_root / "videos" / f"{cam}.mp4"

        for frame_idx in parse_csv(args.frames, int):
            image = load_frame(video_path, frame_idx)
            verts = make_smpl_vertices(model, smpl_data, frame_idx, device)
            scale = float(np.asarray(smpl_data["smpl_scaling"]).reshape(-1)[0])
            trans = np.asarray(smpl_data["smpl_trans"][frame_idx], dtype=np.float32).reshape(3)
            variants = build_variants(verts, scale, trans, R, T)

            panels = []
            metrics = {"cam": cam, "frame": frame_idx}
            for name, pts_cam in variants.items():
                uv, valid = project_points(pts_cam, K)
                panel = draw_projection(image, uv, valid, name)
                panel = cv2.resize(panel, (640, 360), interpolation=cv2.INTER_AREA)
                panels.append(panel)

                h, w = image.shape[:2]
                in_img = (
                    valid
                    & np.isfinite(uv).all(axis=1)
                    & (uv[:, 0] >= 0)
                    & (uv[:, 0] < w)
                    & (uv[:, 1] >= 0)
                    & (uv[:, 1] < h)
                )
                metrics[name + "_in_img_ratio"] = float(in_img.mean())

            row1 = np.concatenate(panels[:3], axis=1)
            row2 = np.concatenate(panels[3:], axis=1)
            blank = np.zeros_like(panels[0])
            if row2.shape[1] < row1.shape[1]:
                row2 = np.concatenate([row2, blank], axis=1)
            collage = np.concatenate([row1, row2], axis=0)
            out_path = out_root / f"{cam}_frame{frame_idx:06d}_projection_variants.png"
            cv2.imwrite(str(out_path), collage)
            metrics["output"] = str(out_path)
            summary.append(metrics)

    summary_path = out_root / "projection_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
