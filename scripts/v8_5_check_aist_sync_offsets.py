#!/usr/bin/env python3
"""Visualize AIST video-frame vs SMPL-frame synchronization offsets."""

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
    REPO_ROOT,
    SMPL_MODEL_ROOT,
    draw_projection,
    load_frame,
    make_smpl_vertices,
    parse_csv,
    project_points,
    rodrigues,
)


def draw_panel(
    image: np.ndarray,
    model,
    smpl_data: dict,
    video_frame: int,
    smpl_frame: int,
    K: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    smpl_frame = int(np.clip(smpl_frame, 0, len(smpl_data["smpl_poses"]) - 1))
    verts = make_smpl_vertices(model, smpl_data, smpl_frame, device)
    scale = float(np.asarray(smpl_data["smpl_scaling"]).reshape(-1)[0])
    trans = np.asarray(smpl_data["smpl_trans"][smpl_frame], dtype=np.float32).reshape(3)
    world = verts * scale + trans[None, :]
    cam = world @ R.T + T[None, :]
    uv, valid = project_points(cam, K)
    title = f"video={video_frame} smpl={smpl_frame} offset={smpl_frame - video_frame:+d}"
    panel = draw_projection(image, uv, valid, title)
    return cv2.resize(panel, (480, 270), interpolation=cv2.INTER_AREA)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data/asit"))
    parser.add_argument("--sequence", default="gBR_sBM_cAll_d04_mBR0_ch01")
    parser.add_argument("--cams", default="c05")
    parser.add_argument("--frames", default="0,60,120")
    parser.add_argument("--offsets", default="-120,-90,-60,-30,0,30,60,90,120,180,240")
    parser.add_argument("--output_dir", type=Path, default=Path("output/v8_5_aist_projection_check"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    seq_root = args.root / args.sequence
    with (seq_root / "gt" / "smpl.pkl").open("rb") as f:
        smpl_data = pickle.load(f)

    device = torch.device(args.device)
    model = smplx.create(str(SMPL_MODEL_ROOT), "smpl", gender="neutral").to(device)
    model.eval()

    out_root = args.output_dir / args.sequence / "sync_offsets"
    out_root.mkdir(parents=True, exist_ok=True)

    for cam in parse_csv(args.cams, str):
        with (seq_root / "camera" / f"{cam}_camera.json").open("r", encoding="utf-8") as f:
            cam_data = json.load(f)
        K = np.asarray(cam_data["matrix"], dtype=np.float32)
        R = rodrigues(cam_data["rotation"])
        T = np.asarray(cam_data["translation"], dtype=np.float32).reshape(3)
        video_path = seq_root / "videos" / f"{cam}.mp4"

        for frame_idx in parse_csv(args.frames, int):
            image = load_frame(video_path, frame_idx)
            panels = []
            for off in parse_csv(args.offsets, int):
                panels.append(
                    draw_panel(
                        image,
                        model,
                        smpl_data,
                        frame_idx,
                        frame_idx + off,
                        K,
                        R,
                        T,
                        device,
                    )
                )
            cols = 4
            rows = []
            blank = np.zeros_like(panels[0])
            for i in range(0, len(panels), cols):
                row_panels = panels[i : i + cols]
                while len(row_panels) < cols:
                    row_panels.append(blank)
                rows.append(np.concatenate(row_panels, axis=1))
            out = np.concatenate(rows, axis=0)
            out_path = out_root / f"{cam}_video{frame_idx:06d}_smpl_offset_sweep.png"
            cv2.imwrite(str(out_path), out)
            print(out_path)


if __name__ == "__main__":
    main()
