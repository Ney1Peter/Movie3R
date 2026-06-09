#!/usr/bin/env python3
"""Draw the verified AIST projection formula for all cameras in one sheet."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import smplx
import torch

from v8_5_check_aist_projection import (
    SMPL_MODEL_ROOT,
    draw_projection,
    load_frame,
    make_smpl_vertices,
    parse_csv,
    project_points,
    rodrigues,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data/asit"))
    parser.add_argument("--sequence", default="gBR_sBM_cAll_d04_mBR0_ch01")
    parser.add_argument("--cams", default="c01,c02,c03,c04,c05,c06,c07,c08,c09")
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

    out_root = args.output_dir / args.sequence / "all_camera"
    out_root.mkdir(parents=True, exist_ok=True)

    for frame_idx in parse_csv(args.frames, int):
        panels = []
        for cam in parse_csv(args.cams, str):
            video_path = seq_root / "videos" / f"{cam}.mp4"
            calib_path = seq_root / "camera" / f"{cam}_camera.json"
            if not video_path.is_file() or not calib_path.is_file():
                continue
            with calib_path.open("r", encoding="utf-8") as f:
                cam_data = json.load(f)
            K = np.asarray(cam_data["matrix"], dtype=np.float32)
            R = rodrigues(cam_data["rotation"])
            T = np.asarray(cam_data["translation"], dtype=np.float32).reshape(3)
            image = load_frame(video_path, frame_idx)
            verts = make_smpl_vertices(model, smpl_data, frame_idx, device)
            scale = float(np.asarray(smpl_data["smpl_scaling"]).reshape(-1)[0])
            trans = np.asarray(smpl_data["smpl_trans"][frame_idx], dtype=np.float32).reshape(3)
            world = verts * scale + trans[None, :]
            pts_cam = world @ R.T + T[None, :]
            uv, valid = project_points(pts_cam, K)
            panel = draw_projection(image, uv, valid, f"{cam} video={frame_idx} smpl={frame_idx}")
            panels.append(cv2.resize(panel, (480, 270), interpolation=cv2.INTER_AREA))

        if not panels:
            continue
        cols = 3
        blank = np.zeros_like(panels[0])
        rows = []
        for i in range(0, len(panels), cols):
            row = panels[i : i + cols]
            while len(row) < cols:
                row.append(blank)
            rows.append(np.concatenate(row, axis=1))
        out = np.concatenate(rows, axis=0)
        out_path = out_root / f"frame{frame_idx:06d}_all_camera_projection.png"
        cv2.imwrite(str(out_path), out)
        print(out_path)


if __name__ == "__main__":
    main()
