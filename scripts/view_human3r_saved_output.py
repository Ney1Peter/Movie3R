#!/usr/bin/env python3
"""Launch the Human3R SceneHumanViewer from a saved ``demo.py --save`` directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import roma

from dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates, geotrf
from dust3r.utils.smpl_layer import SMPL_Layer
from viser_utils import SceneHumanViewer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True, help="Saved Human3R output directory.")
    parser.add_argument("--raw_output_dir", type=Path, default=None, help="Optional raw output directory to show raw cameras as gray GT cameras.")
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--source_video", type=Path, default=None, help="Optional source mp4 used to infer frame count.")
    parser.add_argument("--viewer_port", type=int, default=8080)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--vis_threshold", type=float, default=1.5)
    parser.add_argument("--msk_threshold", type=float, default=0.1)
    parser.add_argument("--mask_morph", type=int, default=10)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--downsample_factor", type=int, default=1)
    parser.add_argument("--smpl_downsample", type=int, default=1)
    parser.add_argument("--camera_downsample", type=int, default=1)
    parser.add_argument("--normal_debug_json", type=Path, default=None, help="Optional line/label overlay JSON for plane normal debugging.")
    parser.add_argument("--dry_run", action="store_true", help="Load all viewer payloads and exit without starting the server.")
    return parser.parse_args()


def infer_num_frames(output_dir: Path, source_video: Path | None, explicit_num_frames: int | None) -> int:
    if explicit_num_frames is not None:
        return int(explicit_num_frames)
    if source_video is not None:
        cap = cv2.VideoCapture(str(source_video))
        if not cap.isOpened():
            raise ValueError(f"Could not open source video: {source_video}")
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if n > 0:
            return n
    files = sorted((output_dir / "camera").glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No camera npz files found under {output_dir / 'camera'}")
    return len(files)


def load_cam_dict(output_dir: Path, num_frames: int) -> dict:
    focal, pp, R, t = [], [], [], []
    for i in range(num_frames):
        cam = np.load(output_dir / "camera" / f"{i:06d}.npz")
        K = cam["intrinsics"].astype(np.float32)
        pose = cam["pose"].astype(np.float32)
        focal.append(float(0.5 * (K[0, 0] + K[1, 1])))
        pp.append(K[:2, 2])
        R.append(pose[:3, :3])
        t.append(pose[:3, 3])
    return {
        "focal": np.asarray(focal, dtype=np.float32),
        "pp": np.asarray(pp, dtype=np.float32),
        "R": np.asarray(R, dtype=np.float32),
        "t": np.asarray(t, dtype=np.float32),
    }


def load_viewer_payload(output_dir: Path, num_frames: int, device: str):
    pts3ds, colors, confs, msks = [], [], [], []
    smpl_shapes, smpl_rotvecs, smpl_transls, smpl_exprs, poses, intrinsics, smpl_ids = [], [], [], [], [], [], []

    for i in range(num_frames):
        cam = np.load(output_dir / "camera" / f"{i:06d}.npz")
        pose = cam["pose"].astype(np.float32)
        K = cam["intrinsics"].astype(np.float32)
        depth = np.load(output_dir / "depth" / f"{i:06d}.npy").astype(np.float32)
        conf = np.load(output_dir / "conf" / f"{i:06d}.npy").astype(np.float32)
        color_bgr = cv2.imread(str(output_dir / "color" / f"{i:06d}.png"), cv2.IMREAD_COLOR)
        if color_bgr is None:
            raise FileNotFoundError(output_dir / "color" / f"{i:06d}.png")
        color = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        pc_world, _ = depthmap_to_absolute_camera_coordinates(depth, K, pose)

        smpl = np.load(output_dir / "smpl" / f"{i:06d}.npz", allow_pickle=True)
        msk = smpl["msk"]
        if msk is None:
            msk = np.zeros((1, depth.shape[0], depth.shape[1]), dtype=np.float32)
        smpl_shape = smpl["shape"].astype(np.float32)
        smpl_rotvec = smpl["rotvec"].astype(np.float32)
        smpl_transl = smpl["transl"].astype(np.float32)
        smpl_expr = smpl["expression"]
        if smpl_expr is not None:
            smpl_expr = smpl_expr.astype(np.float32)
        smpl_id = smpl["smpl_id"] if "smpl_id" in smpl.files else np.arange(smpl_shape.shape[0], dtype=np.int64)

        pts3ds.append(pc_world[None].astype(np.float32))
        colors.append(color[None].astype(np.float32))
        confs.append(conf[None].astype(np.float32))
        msks.append(msk.astype(np.float32))
        smpl_shapes.append(smpl_shape)
        smpl_rotvecs.append(smpl_rotvec)
        smpl_transls.append(smpl_transl)
        smpl_exprs.append(smpl_expr)
        poses.append(pose)
        intrinsics.append(K)
        smpl_ids.append(smpl_id)

    beta_dim = next((s.shape[-1] for s in smpl_shapes if s.shape[0] > 0), 10)
    smpl_layer = SMPL_Layer(type="smplx", gender="neutral", num_betas=beta_dim, kid=False, person_center="head").to(device)
    smpl_faces = smpl_layer.bm_x.faces
    all_verts = []
    with torch.no_grad():
        for i in range(num_frames):
            n_humans = smpl_shapes[i].shape[0]
            if n_humans == 0:
                all_verts.append(np.empty((0, 0, 3), dtype=np.float32))
                continue
            expr = None if smpl_exprs[i] is None else torch.from_numpy(smpl_exprs[i]).to(device=device, dtype=torch.float32)
            out = smpl_layer(
                torch.from_numpy(smpl_rotvecs[i]).to(device=device, dtype=torch.float32),
                torch.from_numpy(smpl_shapes[i]).to(device=device, dtype=torch.float32),
                torch.from_numpy(smpl_transls[i]).to(device=device, dtype=torch.float32),
                None,
                None,
                K=torch.from_numpy(intrinsics[i]).to(device=device, dtype=torch.float32).expand(n_humans, -1, -1),
                expression=expr,
            )
            verts_world = geotrf(
                torch.from_numpy(poses[i]).to(device=device, dtype=torch.float32).unsqueeze(0),
                out["smpl_v3d"].unsqueeze(0),
            )[0]
            all_verts.append(verts_world.detach().cpu().numpy().astype(np.float32))
    return pts3ds, colors, confs, all_verts, smpl_faces, smpl_ids, msks


def add_normal_debug_overlay(viewer: SceneHumanViewer, path: Path | None) -> None:
    if path is None:
        return
    data = json.loads(path.read_text())
    segments = data.get("segments", [])
    if segments:
        points = np.asarray([[seg["start"], seg["end"]] for seg in segments], dtype=np.float32)
        colors = np.asarray([[seg.get("color", [255, 255, 0]), seg.get("color", [255, 255, 0])] for seg in segments], dtype=np.uint8)
        viewer.server.scene.add_line_segments(
            "/debug_floor_normals",
            points=points,
            colors=colors,
            line_width=float(data.get("line_width", 6.0)),
        )
    for label in data.get("labels", []):
        viewer.server.scene.add_label(
            label.get("name", "/debug_label"),
            label.get("text", ""),
            position=np.asarray(label["position"], dtype=np.float32),
            font_size_mode="scene",
            font_scene_height=float(label.get("height", 0.08)),
            depth_test=False,
        )


def main() -> None:
    args = parse_args()
    num_frames = infer_num_frames(args.output_dir, args.source_video, args.num_frames)
    print(f"Loading {num_frames} frames from {args.output_dir}")
    pts3ds, colors, confs, verts, faces, smpl_ids, msks = load_viewer_payload(args.output_dir, num_frames, args.device)
    cam_dict = load_cam_dict(args.output_dir, num_frames)

    gt_cam_dict = None
    show_gt_camera = False
    if args.raw_output_dir is not None:
        gt_cam_dict = load_cam_dict(args.raw_output_dir, num_frames)
        show_gt_camera = True
        print(f"Raw cameras loaded as gray GT camera frustums from {args.raw_output_dir}")

    if args.dry_run:
        print("Dry run passed: saved output can be loaded by the Human3R viewer payload path.")
        return

    print(f"Launching Human3R viewer on port {args.viewer_port}")
    print(f"Open http://127.0.0.1:{args.viewer_port} after forwarding this port.")
    viewer = SceneHumanViewer(
        pts3ds,
        colors,
        confs,
        cam_dict,
        verts,
        faces,
        smpl_ids,
        msks,
        gt_cam_dict=gt_cam_dict,
        device=args.device,
        port=args.viewer_port,
        edge_color_list=[None] * len(pts3ds),
        show_camera=True,
        show_gt_camera=show_gt_camera,
        vis_threshold=args.vis_threshold,
        msk_threshold=args.msk_threshold,
        mask_morph=args.mask_morph,
        size=args.size,
        downsample_factor=args.downsample_factor,
        smpl_downsample_factor=args.smpl_downsample,
        camera_downsample_factor=args.camera_downsample,
    )
    add_normal_debug_overlay(viewer, args.normal_debug_json)
    viewer.run()


if __name__ == "__main__":
    main()
