#!/usr/bin/env python3
"""Launch one interactive ``demo.py``-style viewer from an exported payload."""

from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

import cv2
import numpy as np


MOVIE_ROOT = Path(__file__).resolve().parents[2]
for item in (MOVIE_ROOT, MOVIE_ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from viser_utils import SceneHumanViewer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--cut-index", type=int, default=5)
    parser.add_argument("--up-direction", choices=("+y", "-y"), default="-y")
    parser.add_argument("--hide-camera", action="store_true")
    return parser.parse_args()


def load_payload(root: Path) -> dict[str, object]:
    root = root.resolve()
    colour_paths = sorted((root / "color").glob("*.png"))
    camera_paths = sorted((root / "camera").glob("*.npz"))
    smpl_paths = sorted((root / "smpl").glob("*.npz"))
    counts = {len(colour_paths), len(camera_paths), len(smpl_paths)}
    if len(counts) != 1 or not colour_paths:
        raise ValueError(
            f"Payload frame mismatch: color={len(colour_paths)}, "
            f"camera={len(camera_paths)}, smpl={len(smpl_paths)}"
        )

    pc_list = []
    colour_list = []
    confidence_list = []
    mask_list = []
    vertices = []
    identities = []
    cameras = {"focal": [], "pp": [], "R": [], "t": []}
    faces = None
    all_display_vertices = []
    depth_paths = sorted((root / "depth").glob("*.npy"))
    conf_paths = sorted((root / "conf").glob("*.npy"))
    if depth_paths and len(depth_paths) != len(colour_paths):
        raise ValueError(f"Depth/frame mismatch: depth={len(depth_paths)}, color={len(colour_paths)}")
    if conf_paths and len(conf_paths) != len(colour_paths):
        raise ValueError(f"Confidence/frame mismatch: conf={len(conf_paths)}, color={len(colour_paths)}")
    scene_frames = 0
    for frame_index, (colour_path, camera_path, smpl_path) in enumerate(
        zip(colour_paths, camera_paths, smpl_paths)
    ):
        image_bgr = cv2.imread(str(colour_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise OSError(f"Could not read {colour_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        height, width = image.shape[:2]
        with np.load(camera_path, allow_pickle=False) as camera:
            pose = np.asarray(camera["pose"], dtype=np.float32)
            intrinsics = np.asarray(camera["intrinsics"], dtype=np.float32)
        # ``demo.py --save`` stores camera-space z depth.  Reconstruct the
        # camera-space pointmap and then use the payload's own camera pose to
        # place it in the same world frame as that method's meshes.  External
        # baselines have explicit zero depth/confidence files, so they remain
        # honestly mesh-only instead of borrowing Bridge3R scene geometry.
        if depth_paths and conf_paths:
            depth = np.asarray(np.load(depth_paths[frame_index]), dtype=np.float32)
            confidence = np.asarray(np.load(conf_paths[frame_index]), dtype=np.float32)
            depth = np.squeeze(depth)
            confidence = np.squeeze(confidence)
            if depth.shape != (height, width):
                depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
            if confidence.shape != (height, width):
                confidence = cv2.resize(confidence, (width, height), interpolation=cv2.INTER_NEAREST)
            yy, xx = np.indices((height, width), dtype=np.float32)
            fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
            cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
            if not np.isfinite([fx, fy]).all() or min(abs(fx), abs(fy)) < 1e-6:
                raise ValueError(f"Invalid intrinsics in {camera_path}: {intrinsics}")
            camera_points = np.stack(
                ((xx - cx) * depth / fx, (yy - cy) * depth / fy, depth), axis=-1
            )
            world_points = camera_points @ pose[:3, :3].T + pose[:3, 3]
            valid_scene = (
                np.isfinite(world_points).all(axis=-1)
                & np.isfinite(depth)
                & np.isfinite(confidence)
                & (depth > 1e-5)
            )
            confidence = np.where(valid_scene, confidence, 0.0).astype(np.float32)
            world_points = np.where(valid_scene[..., None], world_points, 0.0).astype(np.float32)
            if np.any(confidence > 1.0):
                scene_frames += 1
        else:
            world_points = np.zeros((height, width, 3), dtype=np.float32)
            confidence = np.zeros((height, width), dtype=np.float32)
        pc_list.append(world_points)
        colour_list.append(image[None])
        confidence_list.append(confidence[None])
        cameras["focal"].append(float(intrinsics[0, 0]))
        cameras["pp"].append(np.asarray([intrinsics[0, 2], intrinsics[1, 2]], dtype=np.float32))
        cameras["R"].append(pose[:3, :3])
        cameras["t"].append(pose[:3, 3])
        with np.load(smpl_path, allow_pickle=False) as smpl:
            frame_vertices = np.asarray(smpl["verts_world"], dtype=np.float32)
            frame_ids = np.asarray(smpl["smpl_id"], dtype=np.int64)
            frame_faces = np.asarray(smpl["faces"], dtype=np.int32)
            frame_mask = np.asarray(smpl["msk"], dtype=np.float32) if "msk" in smpl.files else None
        if frame_mask is not None:
            frame_mask = np.squeeze(frame_mask)
            if frame_mask.shape != (height, width):
                frame_mask = cv2.resize(frame_mask, (width, height), interpolation=cv2.INTER_NEAREST)
            mask_list.append(frame_mask[None])
        else:
            mask_list.append(None)
        if len(frame_vertices) != len(frame_ids):
            raise ValueError(f"Mesh/ID mismatch in {smpl_path}")
        if faces is None:
            faces = frame_faces
        elif not np.array_equal(faces, frame_faces):
            raise ValueError(f"Topology mismatch in {smpl_path}")
        vertices.append(frame_vertices)
        identities.append(frame_ids)
        if len(frame_vertices):
            all_display_vertices.append(frame_vertices.reshape(-1, 3)[::100])
    for key in cameras:
        cameras[key] = np.asarray(cameras[key])
    if faces is None or not all_display_vertices:
        raise ValueError(f"Payload has no renderable human mesh: {root}")
    sampled = np.concatenate(all_display_vertices, axis=0)
    lower, upper = np.nanpercentile(sampled, [1.0, 99.0], axis=0)
    centre = 0.5 * (lower + upper)
    span = max(float(np.max(upper - lower)), 1.5)
    return {
        "pc_list": pc_list,
        "colour_list": colour_list,
        "confidence_list": confidence_list,
        "mask_list": mask_list,
        "vertices": vertices,
        "identities": identities,
        "cameras": cameras,
        "faces": faces,
        "centre": centre,
        "span": span,
        "frames": len(colour_paths),
        "scene_frames": scene_frames,
    }


def main() -> None:
    args = parse_args()
    payload = load_payload(args.payload)
    initial = int(np.clip(args.cut_index, 0, int(payload["frames"]) - 1))
    print(json.dumps({
        "title": args.title,
        "payload": str(args.payload.resolve()),
        "port": int(args.port),
        "url": f"http://0.0.0.0:{args.port}",
        "frames": int(payload["frames"]),
        "initial_frame": initial,
        "scene_pointcloud_frames": int(payload["scene_frames"]),
        "same_sequence_contract": "the payload RGB and frame indices are frozen by the two-dataset manifest",
    }, indent=2, ensure_ascii=False), flush=True)
    viewer = SceneHumanViewer(
        payload["pc_list"],
        payload["colour_list"],
        payload["confidence_list"],
        payload["cameras"],
        payload["vertices"],
        payload["faces"],
        payload["identities"],
        payload["mask_list"],
        device="cpu",
        port=int(args.port),
        edge_color_list=[None] * int(payload["frames"]),
        show_camera=not bool(args.hide_camera),
        vis_threshold=1.0,
        msk_threshold=0.1,
        mask_morph=0,
        size=512,
        downsample_factor=20,
        smpl_downsample_factor=1,
        camera_downsample_factor=2,
        initial_timestep=initial,
    )
    viewer.server.set_up_direction(args.up_direction)
    viewer.fourd = True
    # Centre the initial view on the first post-cut humans rather than on the
    # union of the entire trajectory. Large viewpoint cuts can otherwise put
    # the currently displayed people close to a screen edge.
    initial_vertices = np.asarray(payload["vertices"][initial], dtype=np.float64)
    finite_vertices = initial_vertices[np.isfinite(initial_vertices).all(axis=-1)]
    if len(finite_vertices):
        lower, upper = np.nanpercentile(finite_vertices.reshape(-1, 3), [1.0, 99.0], axis=0)
        centre = 0.5 * (lower + upper)
        span = max(float(np.max(upper - lower)), 1.5)
    else:
        centre = np.asarray(payload["centre"], dtype=np.float64)
        span = float(payload["span"])
    up = np.asarray([0.0, 1.0 if args.up_direction == "+y" else -1.0, 0.0])

    def _centre_camera(client) -> None:
        client.camera.look_at = centre
        client.camera.position = centre + span * np.asarray([1.35, 0.70 * up[1], 1.35])
        client.camera.up_direction = up

    @viewer.server.on_client_connect
    def _initial_camera(client) -> None:
        _centre_camera(client)
        # Browsers may apply their default camera once after the websocket is
        # established. Re-apply the same view after that initialization turn.
        threading.Timer(0.35, lambda: _centre_camera(client)).start()

    viewer.run()


if __name__ == "__main__":
    main()
