#!/usr/bin/env python3
"""View saved ASIT Human3R outputs with GT/raw/corrected camera overlays.

The saved Human3R payload lives in the model's arbitrary viewer gauge. ASIT GT
camera poses are official c2w matrices in the dataset gauge. For visualization
we use only the GT relative trajectory and anchor it to the original Human3R
frame-0 camera:

    GT_view_i = RawHuman3R_0 @ inv(GT_0) @ GT_i

This mirrors the coordinate convention used by the V8/V9 benchmark viewers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import viser.transforms as tf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.view_human3r_saved_output import infer_num_frames, load_cam_dict, load_viewer_payload
from viser_utils import SceneHumanViewer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True, help="Saved output used for scene/SMPL payload.")
    parser.add_argument("--raw_output_dir", type=Path, required=True, help="Original Human3R saved output.")
    parser.add_argument("--meta_json", type=Path, required=True, help="selected_clip_meta.json containing ASIT rgb_paths.")
    parser.add_argument("--viewer_port", type=int, required=True)
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vis_threshold", type=float, default=2.0)
    parser.add_argument("--msk_threshold", type=float, default=0.1)
    parser.add_argument("--mask_morph", type=int, default=10)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--downsample_factor", type=int, default=1)
    parser.add_argument("--smpl_downsample", type=int, default=1)
    parser.add_argument("--camera_downsample", type=int, default=1)
    return parser.parse_args()


def cam_pose(cam_dict: dict[str, np.ndarray], index: int) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = cam_dict["R"][index]
    pose[:3, 3] = cam_dict["t"][index]
    return pose


def poses_to_cam_dict(poses: np.ndarray, intrinsics_ref_dir: Path) -> dict[str, np.ndarray]:
    focal, pp, R, t = [], [], [], []
    for i, pose in enumerate(poses):
        # Use saved viewer intrinsics so the frustum scale/aspect matches the
        # resized Human3R payload rather than the original ASIT video size.
        K = np.load(intrinsics_ref_dir / "camera" / f"{i:06d}.npz")["intrinsics"].astype(np.float32)
        focal.append(float(0.5 * (K[0, 0] + K[1, 1])))
        pp.append(K[:2, 2])
        R.append(pose[:3, :3].astype(np.float32))
        t.append(pose[:3, 3].astype(np.float32))
    return {
        "focal": np.asarray(focal, dtype=np.float32),
        "pp": np.asarray(pp, dtype=np.float32),
        "R": np.asarray(R, dtype=np.float32),
        "t": np.asarray(t, dtype=np.float32),
    }


def load_asit_gt_viewer_cam_dict(meta_path: Path, raw_output_dir: Path, num_frames: int) -> dict[str, np.ndarray]:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    rgb_paths = [Path(p) for p in meta["rgb_paths"][:num_frames]]
    if len(rgb_paths) != num_frames:
        raise ValueError(f"Expected {num_frames} rgb paths in {meta_path}, got {len(rgb_paths)}")

    gt_poses = []
    for rgb_path in rgb_paths:
        cam_path = rgb_path.parent.parent / "cam" / f"{rgb_path.stem}.npz"
        data = np.load(cam_path)
        gt_poses.append(data["pose"].astype(np.float32))
    gt_poses = np.stack(gt_poses, axis=0)

    raw_cam_dict = load_cam_dict(raw_output_dir, num_frames)
    raw0 = cam_pose(raw_cam_dict, 0)
    gt_rel = np.einsum("ij,njk->nik", np.linalg.inv(gt_poses[0]), gt_poses)
    gt_viewer = np.einsum("ij,njk->nik", raw0, gt_rel).astype(np.float32)
    return poses_to_cam_dict(gt_viewer, raw_output_dir)


def add_camera_set(
    viewer: SceneHumanViewer,
    cam_dict: dict[str, np.ndarray],
    color: tuple[int, int, int],
    prefix: str,
) -> None:
    for step in range(len(cam_dict["R"])):
        focal = float(cam_dict["focal"][step])
        pp = cam_dict["pp"][step]
        R = cam_dict["R"][step]
        t = cam_dict["t"][step]
        q = tf.SO3.from_matrix(R).wxyz
        fov = 2 * np.arctan(float(pp[0]) / max(focal, 1e-6))
        aspect = float(pp[0]) / max(float(pp[1]), 1e-6)
        viewer.server.add_camera_frustum(
            name=f"/frames/{step}/{prefix}_camera",
            fov=fov,
            aspect=aspect,
            wxyz=q,
            position=t,
            scale=0.14,
            line_width=2.5,
            color=color,
        )


def main() -> None:
    args = parse_args()
    num_frames = infer_num_frames(args.output_dir, None, args.num_frames)
    print(f"Loading scene payload from {args.output_dir}")
    pts3ds, colors, confs, verts, faces, smpl_ids, msks = load_viewer_payload(
        args.output_dir, num_frames, args.device
    )
    output_cam_dict = load_cam_dict(args.output_dir, num_frames)
    raw_cam_dict = load_cam_dict(args.raw_output_dir, num_frames)
    gt_cam_dict = load_asit_gt_viewer_cam_dict(args.meta_json, args.raw_output_dir, num_frames)

    print("Color legend: GT camera=red, original Human3R raw=gray, current output=yellow.")
    print(f"Open http://127.0.0.1:{args.viewer_port} after forwarding this port.")
    viewer = SceneHumanViewer(
        pts3ds,
        colors,
        confs,
        output_cam_dict,
        verts,
        faces,
        smpl_ids,
        msks,
        device=args.device,
        port=args.viewer_port,
        edge_color_list=[None] * len(pts3ds),
        show_camera=False,
        show_gt_camera=False,
        vis_threshold=args.vis_threshold,
        msk_threshold=args.msk_threshold,
        mask_morph=args.mask_morph,
        size=args.size,
        downsample_factor=args.downsample_factor,
        smpl_downsample_factor=args.smpl_downsample,
        camera_downsample_factor=args.camera_downsample,
    )
    add_camera_set(viewer, gt_cam_dict, color=(255, 40, 40), prefix="gt")
    add_camera_set(viewer, raw_cam_dict, color=(150, 150, 150), prefix="human3r_raw")
    add_camera_set(viewer, output_cam_dict, color=(255, 220, 0), prefix="current_output")
    viewer.server.scene.add_label(
        "/legend",
        "GT red | Human3R raw gray | current output yellow",
        position=np.asarray([0.0, -0.45, 0.15], dtype=np.float32),
        font_size_mode="scene",
        font_scene_height=0.07,
        depth_test=False,
    )
    viewer.run()


if __name__ == "__main__":
    main()
