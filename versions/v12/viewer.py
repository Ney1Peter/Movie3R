#!/usr/bin/env python3
"""Launch a demo.py-style viewer for a Movie3R-Single V12 result."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dust3r.utils.smpl_layer import SMPL_Layer  # noqa: E402
from versions.v12.experiments.v14_3_human_continuity_visualization import (  # noqa: E402
    camera_points,
    load_frame,
    transform_points,
)
from versions.v12.experiments.v14_5_true_recurrent_multicut_audit import scale_pose  # noqa: E402
from viser_utils import SceneHumanViewer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result_dir", type=Path, required=True)
    parser.add_argument("--method", choices=("lite", "full"), required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--downsample_factor", type=int, default=10)
    parser.add_argument("--vis_threshold", type=float, default=1.5)
    parser.add_argument("--mask_morph", type=int, default=10)
    return parser.parse_args()


def state_for_frame(states: dict, frame: int) -> tuple[float, np.ndarray]:
    start = max(int(value) for value in states if int(value) <= frame)
    state = states[str(start)]
    return float(state["scale"]), np.asarray(state["gauge"], dtype=np.float32)


def frame_humans(
    layer: SMPL_Layer,
    local_dir: Path,
    index: int,
    intrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(local_dir / "smpl" / f"{index:06d}.npz", allow_pickle=True) as payload:
        rotvec = np.asarray(payload["rotvec"], dtype=np.float32)
        shape = np.asarray(payload["shape"], dtype=np.float32)
        transl = np.asarray(payload["transl"], dtype=np.float32)
        expression = np.asarray(payload["expression"], dtype=np.float32)
    count = len(shape)
    if count == 0:
        return (
            np.empty((0, 0, 3), dtype=np.float32),
            np.empty((0, 0, 3), dtype=np.float32),
        )
    if expression.ndim != 2 or len(expression) != count:
        expression = np.zeros((count, 10), dtype=np.float32)
    with torch.no_grad():
        output = layer(
            torch.from_numpy(rotvec),
            torch.from_numpy(shape),
            torch.from_numpy(transl),
            None,
            None,
            K=torch.from_numpy(intrinsic).unsqueeze(0).expand(count, -1, -1),
            expression=torch.from_numpy(expression),
        )
    return (
        output["smpl_j3d"].detach().float().cpu().numpy().astype(np.float32),
        output["smpl_v3d"].detach().float().cpu().numpy().astype(np.float32),
    )


def main() -> None:
    args = parse_args()
    report_path = args.result_dir / "v12_result.json"
    if not report_path.is_file():
        report_path = args.result_dir / "v14_7_custom_multicut.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    states = report[args.method]["shot_state"]
    local_dir = args.result_dir / "human3r_true_reset"
    count = int(report["frames"])
    frames = [load_frame(local_dir, index) for index in range(count)]

    layer = SMPL_Layer(
        type="smplx", gender="neutral", num_betas=10, kid=False, person_center="head"
    ).to(torch.device("cpu")).eval()
    intrinsics = np.stack([frame["K"] for frame in frames]).astype(np.float32)
    humans_camera = [
        frame_humans(layer, local_dir, index, intrinsics[index])
        for index in range(count)
    ]

    points_world, colors, confidences, masks, vertices_world = [], [], [], [], []
    camera_poses = []
    for index, frame in enumerate(frames):
        scale, gauge = state_for_frame(states, index)
        camera_pose = gauge @ scale_pose(frame["pose"], scale)
        camera_poses.append(camera_pose.astype(np.float32))

        points = camera_points(frame["depth"], frame["K"])
        points_local = transform_points(frame["pose"], points.reshape(-1, 3)).reshape(points.shape)
        points_world.append(
            transform_points(gauge, points_local.reshape(-1, 3) * scale)
            .reshape(points.shape)
            .astype(np.float32)
        )
        colors.append((frame["image"].astype(np.float32) / 255.0)[None])
        confidences.append(frame["confidence"].astype(np.float32)[None])
        masks.append(frame["mask"].astype(np.float32)[None])

        joints_camera, frame_vertices_camera = humans_camera[index]
        frame_vertices_world = []
        for human_index in range(len(joints_camera)):
            raw_root = joints_camera[human_index, 0]
            root_camera = raw_root * scale
            root_world = camera_pose[:3, :3] @ root_camera + camera_pose[:3, 3]
            centered = (frame_vertices_camera[human_index] - raw_root) * scale
            world_vertices = centered @ camera_pose[:3, :3].T + root_world
            frame_vertices_world.append(world_vertices.astype(np.float32))
        vertices_world.append(
            np.stack(frame_vertices_world)
            if frame_vertices_world
            else np.empty((0, 0, 3), dtype=np.float32)
        )

    camera_poses = np.stack(camera_poses)
    cam_dict = {
        "focal": intrinsics[:, 0, 0],
        "pp": intrinsics[:, :2, 2],
        "R": camera_poses[:, :3, :3],
        "t": camera_poses[:, :3, 3],
    }
    faces = np.asarray(layer.bm_x.faces, dtype=np.int32)
    smpl_ids = [np.arange(len(value), dtype=np.int64) for value in vertices_world]
    print(
        f">> V12 {args.method.upper()} viewer: http://127.0.0.1:{args.port}",
        flush=True,
    )
    viewer = SceneHumanViewer(
        points_world,
        colors,
        confidences,
        cam_dict,
        vertices_world,
        faces,
        smpl_ids,
        masks,
        device="cpu",
        port=int(args.port),
        edge_color_list=[None] * count,
        show_camera=True,
        vis_threshold=float(args.vis_threshold),
        msk_threshold=0.1,
        mask_morph=int(args.mask_morph),
        size=512,
        downsample_factor=int(args.downsample_factor),
        smpl_downsample_factor=1,
        camera_downsample_factor=1,
    )
    viewer.run()


if __name__ == "__main__":
    main()
