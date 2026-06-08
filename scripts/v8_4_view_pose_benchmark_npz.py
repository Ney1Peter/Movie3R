#!/usr/bin/env python3
"""Interactive Viser viewer for V8.4 pose benchmark dumps.

The benchmark dump only contains camera poses and RGB frame paths, not full
Human3R pointmaps. This viewer focuses on the pose correction behavior:

- GT cameras: red
- raw Human3R cameras: gray
- corrected cameras: yellow, with input images attached
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import viser
import viser.transforms as tf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--entry", type=int, default=0, help="0-based entry index in the manifest.")
    parser.add_argument(
        "--eval_dir",
        type=Path,
        default=Path("output/v8_4_pose_benchmark/eval/pose_only_final_test_only_gpu"),
    )
    parser.add_argument("--port", type=int, default=8140)
    parser.add_argument("--camera_scale", type=float, default=0.16)
    parser.add_argument("--image_width", type=int, default=320)
    parser.add_argument(
        "--external_raw_dir",
        type=Path,
        default=None,
        help=(
            "Optional saved Human3R output. If set, gray raw cameras are loaded "
            "from this directory, while GT/corrected relative poses from the pose "
            "dump are anchored to external raw frame 0."
        ),
    )
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a list manifest: {path}")
    return data


def load_rgb(path: Path, width: int) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    if width > 0 and w != width:
        scale = width / max(w, 1)
        image = cv2.resize(image, (width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
    return image


def add_camera(
    server: viser.ViserServer,
    pose: np.ndarray,
    name: str,
    label: str,
    color: tuple[int, int, int],
    image: np.ndarray | None,
    scale: float,
    label_offset: np.ndarray,
) -> None:
    rotation = pose[:3, :3]
    position = pose[:3, 3]
    q_wxyz = tf.SO3.from_matrix(rotation).wxyz
    server.scene.add_camera_frustum(
        name=name,
        fov=np.deg2rad(55.0),
        aspect=16.0 / 9.0,
        scale=scale,
        line_width=2.5,
        color=color,
        image=image,
        wxyz=q_wxyz,
        position=position,
    )
    server.scene.add_label(
        name=f"{name}_label",
        text=label,
        position=position + label_offset,
        font_size_mode="scene",
        font_scene_height=0.045,
        depth_test=False,
    )


def add_trajectory(
    server: viser.ViserServer,
    poses: np.ndarray,
    name: str,
    color: tuple[int, int, int],
) -> None:
    points = poses[:, :3, 3].astype(np.float32)
    if len(points) < 2:
        return
    starts = points[:-1]
    ends = points[1:]
    segments = np.stack([starts, ends], axis=1)
    server.scene.add_line_segments(
        name=name,
        points=segments,
        colors=np.asarray(color, dtype=np.uint8)[None, None, :].repeat(len(segments), axis=0).repeat(2, axis=1),
        line_width=3.0,
    )


def load_saved_camera_poses(output_dir: Path, num_frames: int) -> np.ndarray:
    poses = []
    for i in range(num_frames):
        path = output_dir / "camera" / f"{i:06d}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        poses.append(np.load(path)["pose"].astype(np.float32))
    return np.stack(poses, axis=0)


def add_axis_points(server: viser.ViserServer, poses: np.ndarray) -> None:
    points = poses[:, :3, 3].astype(np.float32)
    center = points.mean(axis=0)
    server.scene.add_grid(
        "/ground_grid",
        width=2.0,
        height=2.0,
        cell_size=0.1,
        position=(float(center[0]), float(center[1]), float(center[2] - 0.15)),
    )


def make_title(record: dict) -> str:
    clip_type = record.get("clip_type", "clip").upper()
    if clip_type == "AABB":
        source = f"{record.get('seqA')} -> {record.get('seqB')}"
        angle = f"{float(record.get('view_angle_deg', 0.0)):.1f} deg"
    else:
        source = str(record.get("seq"))
        angle = "same camera"
    return (
        f"{clip_type} #{int(record.get('benchmark_index', -1)):04d} | "
        f"{record.get('group')} | {record.get('angle_bucket', angle)} | {source}"
    )


def main() -> None:
    args = parse_args()
    records = load_manifest(args.manifest)
    if args.entry < 0 or args.entry >= len(records):
        raise IndexError(f"--entry {args.entry} out of range for {args.manifest} ({len(records)} entries)")
    record = records[args.entry]
    pose_path = args.eval_dir / record["pose_npz"]
    pose_data = np.load(pose_path)

    if args.external_raw_dir is None:
        gt = pose_data["gt_c2w_abs"].astype(np.float32)
        raw = pose_data["raw_c2w_abs_gt0"].astype(np.float32)
        corrected = pose_data["corrected_c2w_abs_gt0"].astype(np.float32)
        coordinate_note = "pose dump GT-frame0 gauge"
    else:
        raw = load_saved_camera_poses(args.external_raw_dir, len(pose_data["gt_c2w_rel"]))
        anchor0 = raw[0]
        gt = np.einsum("ij,njk->nik", anchor0, pose_data["gt_c2w_rel"].astype(np.float32))
        corrected = np.einsum("ij,njk->nik", anchor0, pose_data["corrected_c2w_rel"].astype(np.float32))
        coordinate_note = f"external raw frame-0 gauge: {args.external_raw_dir}"
    gates = pose_data["gate"].reshape(-1)

    images = [load_rgb(Path(path), args.image_width) for path in record.get("rgb_paths", [])]
    while len(images) < len(gt):
        images.append(None)

    server = viser.ViserServer(port=args.port, label=f"V8.4 pose viewer {args.entry}")
    server.scene.set_up_direction("+z")
    server.scene.world_axes.visible = True

    colors = {
        "gt": (255, 40, 40),
        "raw": (150, 150, 150),
        "corrected": (255, 220, 0),
    }
    add_axis_points(server, gt)
    add_trajectory(server, gt, "/trajectory/gt", colors["gt"])
    add_trajectory(server, raw, "/trajectory/raw", colors["raw"])
    add_trajectory(server, corrected, "/trajectory/corrected", colors["corrected"])

    for frame_idx in range(len(gt)):
        add_camera(
            server,
            gt[frame_idx],
            f"/cameras/{frame_idx}/gt",
            f"GT {frame_idx}",
            colors["gt"],
            None,
            args.camera_scale,
            np.asarray([0.0, 0.08, 0.0], dtype=np.float32),
        )
        add_camera(
            server,
            raw[frame_idx],
            f"/cameras/{frame_idx}/raw",
            f"raw {frame_idx}",
            colors["raw"],
            None,
            args.camera_scale * 0.92,
            np.asarray([0.0, 0.02, 0.0], dtype=np.float32),
        )
        add_camera(
            server,
            corrected[frame_idx],
            f"/cameras/{frame_idx}/corrected",
            f"corr {frame_idx} gate={gates[frame_idx]:.2f}",
            colors["corrected"],
            images[frame_idx],
            args.camera_scale,
            np.asarray([0.0, -0.06, 0.0], dtype=np.float32),
        )

    title = make_title(record)
    metrics = (
        f"trans {record.get('v82_raw_trans_err', float('nan')):.4f} -> {record.get('v82_trans_err', float('nan')):.4f}; "
        f"rot {record.get('v82_raw_rot_err_deg', float('nan')):.3f} -> {record.get('v82_rot_err_deg', float('nan')):.3f}; "
        f"gate mean {record.get('v82_gate_mean', float('nan')):.3f}"
    )
    anchor = gt[:, :3, 3].mean(axis=0)
    server.scene.add_label(
        "/title",
        f"{title}\n{metrics}\nGT=red | Human3R raw=gray | corrected=yellow",
        position=anchor + np.asarray([0.0, -0.35, 0.25], dtype=np.float32),
        font_size_mode="scene",
        font_scene_height=0.06,
        depth_test=False,
    )

    print(title)
    print(metrics)
    print(f"Coordinate mode: {coordinate_note}")
    print("Input frames:")
    for idx, path in enumerate(record.get("rgb_paths", [])):
        print(f"  {idx}: {path}")
    print(f"Open http://127.0.0.1:{args.port}")

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
