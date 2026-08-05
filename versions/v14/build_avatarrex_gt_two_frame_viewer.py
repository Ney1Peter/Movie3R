#!/usr/bin/env python3
"""Build a two-frame AvatarReX GT payload in the B0 viewer gauge."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import smplx


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/data/wangzheng/iJCV-CODE/data")
RAW_ROOT = DATA_ROOT / "AvatarReX_raw_meta" / "lbn1"
TRAIN_ROOT = DATA_ROOT / "Training" / "lbn1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--b0-payload", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pre-index", type=int, default=4)
    p.add_argument("--post-index", type=int, default=5)
    p.add_argument("--pre-frame", type=int, default=1835)
    p.add_argument("--post-frame", type=int, default=1836)
    p.add_argument("--depth-mode", choices=("raw", "empty"), default="raw")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def c2w(calibration: dict) -> np.ndarray:
    r = np.asarray(calibration["R"], dtype=np.float64).reshape(3, 3)
    t = np.asarray(calibration["T"], dtype=np.float64).reshape(3)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = r.T
    pose[:3, 3] = -r.T @ t
    return pose


def load_pose(path: Path, index: int) -> np.ndarray:
    with np.load(path / "camera" / f"{index:06d}.npz") as z:
        return np.asarray(z["pose"], dtype=np.float64)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(args.output)
        shutil.rmtree(args.output)
    for subdir in ("camera", "color", "depth", "conf", "smpl"):
        (args.output / subdir).mkdir(parents=True, exist_ok=True)

    calibration = json.loads((RAW_ROOT / "calibration_full.json").read_text())
    smpl_data = np.load(RAW_ROOT / "smpl_params.npz")
    model = smplx.create(
        str(REPO_ROOT / "src" / "models"), "smplx", gender="neutral",
        use_pca=False, flat_hand_mean=True, num_betas=10,
    ).eval()
    faces = np.asarray(model.faces, dtype=np.int32)

    # The prediction viewer uses the B0 pre camera as its world gauge.  Mapping
    # raw GT into that gauge makes a GT-vs-pred viewer comparison meaningful.
    gt_pre_pose = c2w(calibration["22070935"])
    predicted_pre_pose = load_pose(args.b0_payload, args.pre_index)
    gauge = predicted_pre_pose @ np.linalg.inv(gt_pre_pose)

    frame_specs = [
        (0, "22070935", int(args.pre_frame)),
        (1, "22053912", int(args.post_frame)),
    ]
    summary_frames = []
    for out_index, camera_key, frame in frame_specs:
        seq_root = TRAIN_ROOT / camera_key
        stem = f"{frame:08d}"
        color = cv2.imread(str(seq_root / "rgb" / f"{stem}.png"), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(seq_root / "mask" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        if color is None or mask is None:
            raise FileNotFoundError(f"missing RGB/mask for {camera_key} frame {frame}")
        mask_float = (mask > 10).astype(np.float32)
        if args.depth_mode == "raw":
            depth = np.asarray(np.load(seq_root / "depth" / f"{stem}.npy"), dtype=np.float32)
            if np.issubdtype(depth.dtype, np.integer):
                depth = depth / 1000.0
            conf = np.ones_like(depth, dtype=np.float32) * 10.0
        else:
            depth = np.zeros(color.shape[:2], dtype=np.float32)
            conf = np.zeros(color.shape[:2], dtype=np.float32)

        gt_pose = c2w(calibration[camera_key])
        viewer_pose = gauge @ gt_pose
        k = np.asarray(calibration[camera_key]["K"], dtype=np.float32).reshape(3, 3)

        # Generate canonical GT SMPL-X in raw world, then put mesh in the same
        # viewer gauge as the B0 payload.  This bypasses predicted camera or
        # predicted SMPL parameters entirely.
        def tensor(name: str, index: int) -> torch.Tensor:
            return torch.from_numpy(np.asarray(smpl_data[name][index:index + 1])).float()

        kwargs = {
            "global_orient": tensor("global_orient", frame),
            "body_pose": tensor("body_pose", frame).reshape(1, 63),
            "jaw_pose": tensor("jaw_pose", frame).reshape(1, 3),
            "left_hand_pose": tensor("left_hand_pose", frame).reshape(1, 45),
            "right_hand_pose": tensor("right_hand_pose", frame).reshape(1, 45),
            "betas": torch.from_numpy(np.asarray(smpl_data["betas"][0:1])).float(),
            "expression": tensor("expression", frame).reshape(1, 10),
            "transl": tensor("transl", frame).reshape(1, 3),
        }
        with torch.no_grad():
            world_out = model(**kwargs)
        raw_vertices = world_out.vertices[0].cpu().numpy().astype(np.float64)
        vertices_world = raw_vertices @ gauge[:3, :3].T + gauge[:3, 3]

        # The viewer uses cached verts_world and therefore does not need to
        # reconstruct the body from camera-local parameters.
        smpl = {
            "scores": np.ones((512, 368), dtype=np.float32),
            "msk": mask_float[None].astype(np.float32),
            "shape": np.asarray(smpl_data["betas"][0:1], dtype=np.float32),
            "rotvec": np.zeros((1, 53, 3), dtype=np.float32),
            "transl": np.zeros((1, 3), dtype=np.float32),
            "expression": np.asarray(smpl_data["expression"][frame:frame + 1], dtype=np.float32),
            "smpl_id": np.asarray([0], dtype=np.int64),
            "verts_world": vertices_world[None].astype(np.float32),
            "faces": faces,
        }
        cv2.imwrite(str(args.output / "color" / f"{out_index:06d}.png"), color)
        np.save(args.output / "depth" / f"{out_index:06d}.npy", depth)
        np.save(args.output / "conf" / f"{out_index:06d}.npy", conf)
        np.savez(args.output / "camera" / f"{out_index:06d}.npz", pose=viewer_pose.astype(np.float32), intrinsics=k)
        np.savez(args.output / "smpl" / f"{out_index:06d}.npz", **smpl)
        summary_frames.append({"viewer_index": out_index, "camera": camera_key, "raw_frame": frame, "pose": viewer_pose.tolist()})

    summary = {
        "output": str(args.output.resolve()),
        "coordinate_contract": "GT raw world mapped by gauge = B0_pre_c2w @ inv(GT_pre_c2w)",
        "gauge": gauge.tolist(),
        "depth_mode": args.depth_mode,
        "frames": summary_frames,
        "gt_only": True,
    }
    (args.output / "gt_two_frame_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
