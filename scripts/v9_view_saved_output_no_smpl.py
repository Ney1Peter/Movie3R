#!/usr/bin/env python3
"""Launch a saved-output viewer without constructing SMPL-X meshes."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from scripts.view_human3r_saved_output import align_cam_dict_to_reference, infer_num_frames, load_cam_dict
from viser_utils import SceneHumanViewer


def debug_print(message: str) -> None:
    if os.environ.get("V9_VIEW_DEBUG", "0") == "1":
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--raw_output_dir", type=Path, default=None)
    parser.add_argument("--align_raw_to_output0", action="store_true")
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--viewer_port", type=int, default=8080)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--vis_threshold", type=float, default=1.5)
    parser.add_argument("--msk_threshold", type=float, default=0.1)
    parser.add_argument("--mask_morph", type=int, default=10)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--downsample_factor", type=int, default=1)
    parser.add_argument("--camera_downsample", type=int, default=1)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def depth_to_world_fast(depth: np.ndarray, intrinsics: np.ndarray, pose: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    u = np.arange(width, dtype=np.float32)[None, :]
    v = np.arange(height, dtype=np.float32)[:, None]
    z = depth.astype(np.float32, copy=False)
    x = (u - float(intrinsics[0, 2])) * z / float(intrinsics[0, 0])
    y = (v - float(intrinsics[1, 2])) * z / float(intrinsics[1, 1])
    rot = pose[:3, :3].astype(np.float32, copy=False)
    trans = pose[:3, 3].astype(np.float32, copy=False)
    points = np.empty((height, width, 3), dtype=np.float32)
    points[..., 0] = rot[0, 0] * x + rot[0, 1] * y + rot[0, 2] * z + trans[0]
    points[..., 1] = rot[1, 0] * x + rot[1, 1] * y + rot[1, 2] * z + trans[1]
    points[..., 2] = rot[2, 0] * x + rot[2, 1] * y + rot[2, 2] * z + trans[2]
    return points


def load_payload_no_smpl(output_dir: Path, num_frames: int):
    pts3ds, colors, confs, msks = [], [], [], []
    verts, smpl_ids = [], []
    faces = np.empty((0, 3), dtype=np.int32)
    for idx in range(num_frames):
        debug_print(f"frame {idx}: load camera")
        cam = np.load(output_dir / "camera" / f"{idx:06d}.npz")
        pose = cam["pose"].astype(np.float32)
        intrinsics = cam["intrinsics"].astype(np.float32)
        debug_print(f"frame {idx}: load depth/conf/color")
        depth = np.load(output_dir / "depth" / f"{idx:06d}.npy").astype(np.float32)
        conf = np.load(output_dir / "conf" / f"{idx:06d}.npy").astype(np.float32)
        color_bgr = cv2.imread(str(output_dir / "color" / f"{idx:06d}.png"), cv2.IMREAD_COLOR)
        if color_bgr is None:
            raise FileNotFoundError(output_dir / "color" / f"{idx:06d}.png")
        color = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        debug_print(f"frame {idx}: depth to pointcloud")
        points = depth_to_world_fast(depth, intrinsics, pose)

        smpl_path = output_dir / "smpl" / f"{idx:06d}.npz"
        debug_print(f"frame {idx}: load smpl")
        if smpl_path.is_file():
            smpl = np.load(smpl_path, allow_pickle=True)
            msk = smpl["msk"].astype(np.float32) if "msk" in smpl.files else np.zeros((1, *depth.shape), dtype=np.float32)
            frame_verts = (
                smpl["verts_world"].astype(np.float32)
                if "verts_world" in smpl.files
                else np.empty((0, 0, 3), dtype=np.float32)
            )
            if "faces" in smpl.files and smpl["faces"].size > 0:
                faces = smpl["faces"].astype(np.int32)
            frame_ids = smpl["smpl_id"].astype(np.int64) if "smpl_id" in smpl.files else np.arange(frame_verts.shape[0], dtype=np.int64)
        else:
            msk = np.zeros((1, *depth.shape), dtype=np.float32)
            frame_verts = np.empty((0, 0, 3), dtype=np.float32)
            frame_ids = np.empty((0,), dtype=np.int64)
        if msk.shape[-2:] != depth.shape:
            msk = np.zeros((1, *depth.shape), dtype=np.float32)

        pts3ds.append(points[None].astype(np.float32))
        colors.append(color[None].astype(np.float32))
        confs.append(conf[None].astype(np.float32))
        msks.append(msk.astype(np.float32))
        verts.append(frame_verts)
        smpl_ids.append(frame_ids)
        debug_print(f"frame {idx}: done")

    return pts3ds, colors, confs, verts, faces, smpl_ids, msks


def main() -> None:
    args = parse_args()
    num_frames = infer_num_frames(args.output_dir, None, args.num_frames)
    print(f"Loading {num_frames} frames from {args.output_dir}", flush=True)
    pts3ds, colors, confs, verts, faces, smpl_ids, msks = load_payload_no_smpl(args.output_dir, num_frames)
    cam_dict = load_cam_dict(args.output_dir, num_frames)

    raw_cam_dict = None
    if args.raw_output_dir is not None:
        raw_cam_dict = load_cam_dict(args.raw_output_dir, num_frames)
        if args.align_raw_to_output0:
            raw_cam_dict = align_cam_dict_to_reference(raw_cam_dict, cam_dict)
        print(f"Raw camera overlay: {args.raw_output_dir}", flush=True)

    if args.dry_run:
        print("Dry run passed: no-SMPL viewer payload is readable.", flush=True)
        return

    print(f"Launching no-SMPL viewer on port {args.viewer_port}", flush=True)
    print(f"Open http://127.0.0.1:{args.viewer_port} after forwarding this port.", flush=True)
    viewer = SceneHumanViewer(
        pts3ds,
        colors,
        confs,
        cam_dict,
        verts,
        faces,
        smpl_ids,
        msks,
        gt_cam_dict=raw_cam_dict,
        device=args.device,
        port=args.viewer_port,
        edge_color_list=[None] * len(pts3ds),
        show_camera=True,
        show_gt_camera=raw_cam_dict is not None,
        vis_threshold=args.vis_threshold,
        msk_threshold=args.msk_threshold,
        mask_morph=args.mask_morph,
        size=args.size,
        downsample_factor=args.downsample_factor,
        smpl_downsample_factor=1,
        camera_downsample_factor=args.camera_downsample,
    )
    viewer.run()


if __name__ == "__main__":
    main()
