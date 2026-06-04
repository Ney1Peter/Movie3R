#!/usr/bin/env python3
"""Build a pure AvatarReX GT saved-output directory for coordinate sanity checks.

This does not use Human3R predictions.  It writes color/depth/camera/SMPL in one
explicit coordinate convention:

  raw AvatarReX world -- calibration R/T --> RGB/depth camera coordinates
  RGB/depth camera coordinates -- true c2w --> viewer world

Use this before comparing V8.1 predicted cameras, otherwise GT SMPL debugging can
be confused with Human3R's predicted pointmaps or learned gauge.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import roma
import smplx
from smplx.joint_names import JOINT_NAMES

REPO_ROOT = Path(__file__).resolve().parents[1]
SMPLX_DIR = REPO_ROOT / "src" / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--avatarrex_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data"))
    parser.add_argument("--avatarrex_raw_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data/avatarrex_lbn1"))
    parser.add_argument("--split", default="Training")
    parser.add_argument("--seq_a", default="lbn1/22010710")
    parser.add_argument("--seq_b", default="lbn1/22053923")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--depth_mode",
        choices=["raw", "empty"],
        default="raw",
        help="raw uses converted depth files, which are DA3 pseudo-depth here; empty shows only cameras and SMPL.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def avatarrex_specs(seq_a: str, seq_b: str, start_frame: int) -> list[tuple[str, int]]:
    return [
        (seq_a, start_frame),
        (seq_a, start_frame + 1),
        (seq_b, start_frame + 2),
        (seq_b, start_frame + 3),
    ]


def raw_calibration_key(seq: str) -> str:
    return seq.split("/", 1)[1] if "/" in seq else seq


def true_c2w_from_calibration(cal: dict) -> np.ndarray:
    """Calibration convention: X_cam = R @ X_world + T."""
    R_w2c = np.asarray(cal["R"], dtype=np.float32).reshape(3, 3)
    T_w2c = np.asarray(cal["T"], dtype=np.float32).reshape(3)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R_w2c.T
    pose[:3, 3] = -R_w2c.T @ T_w2c
    return pose


def load_depth_meters(path: Path) -> np.ndarray:
    depth = np.load(path)
    depth_m = depth.astype(np.float32)
    if np.issubdtype(depth.dtype, np.integer):
        depth_m /= 1000.0
    depth_m[~np.isfinite(depth_m)] = 0.0
    return depth_m


def build_gt_smpl(
    raw_smplx: torch.nn.Module,
    smpl_data: np.lib.npyio.NpzFile,
    cal: dict,
    frame: int,
    mask: np.ndarray,
    device: torch.device,
) -> dict[str, np.ndarray]:
    R_w2c = torch.tensor(np.asarray(cal["R"], dtype=np.float32).reshape(3, 3), device=device)
    T_w2c = torch.tensor(np.asarray(cal["T"], dtype=np.float32).reshape(3), device=device)
    head_idx = JOINT_NAMES.index("head")

    root_world = torch.tensor(smpl_data["global_orient"][frame], dtype=torch.float32, device=device).reshape(1, 3)
    body = torch.tensor(smpl_data["body_pose"][frame], dtype=torch.float32, device=device).reshape(1, 21, 3)
    left_hand = torch.tensor(smpl_data["left_hand_pose"][frame], dtype=torch.float32, device=device).reshape(1, 15, 3)
    right_hand = torch.tensor(smpl_data["right_hand_pose"][frame], dtype=torch.float32, device=device).reshape(1, 15, 3)
    jaw = torch.tensor(smpl_data["jaw_pose"][frame], dtype=torch.float32, device=device).reshape(1, 1, 3)
    shape = torch.tensor(smpl_data["betas"][0], dtype=torch.float32, device=device).reshape(1, 10)
    transl_world = torch.tensor(smpl_data["transl"][frame], dtype=torch.float32, device=device).reshape(1, 3)
    expr = torch.tensor(smpl_data["expression"][frame], dtype=torch.float32, device=device).reshape(1, 10)

    with torch.no_grad():
        world_out = raw_smplx(
            global_orient=root_world,
            body_pose=body.reshape(1, 21 * 3),
            jaw_pose=jaw.reshape(1, 3),
            leye_pose=torch.zeros(1, 3, device=device),
            reye_pose=torch.zeros(1, 3, device=device),
            left_hand_pose=left_hand.reshape(1, 15 * 3),
            right_hand_pose=right_hand.reshape(1, 15 * 3),
            betas=shape,
            transl=transl_world,
            expression=expr,
        )

    joints_cam = world_out.joints @ R_w2c.T + T_w2c.reshape(1, 1, 3)
    head_cam = joints_cam[:, head_idx]
    root_cam = roma.rotmat_to_rotvec(R_w2c @ roma.rotvec_to_rotmat(root_world)[0]).reshape(1, 1, 3)
    rotvec = torch.cat([root_cam, body, left_hand, right_hand, jaw], dim=1)

    return {
        "scores": np.ones((1, 1), dtype=np.float32),
        "msk": mask[None].astype(np.float32),
        "shape": shape.detach().cpu().numpy().astype(np.float32),
        "rotvec": rotvec.detach().cpu().numpy().astype(np.float32),
        "transl": head_cam.detach().cpu().numpy().astype(np.float32),
        "expression": expr.detach().cpu().numpy().astype(np.float32),
        "smpl_id": np.asarray([0], dtype=np.int64),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} exists; pass --overwrite")
        shutil.rmtree(args.output_dir)
    for sub in ["color", "depth", "conf", "camera", "smpl"]:
        (args.output_dir / sub).mkdir(parents=True, exist_ok=True)

    with open(args.avatarrex_raw_root / "calibration_full.json", "r", encoding="utf-8") as f:
        calibration = json.load(f)
    smpl_data = np.load(args.avatarrex_raw_root / "smpl_params.npz")
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    raw_smplx = smplx.create(
        SMPLX_DIR,
        "smplx",
        gender="neutral",
        use_pca=False,
        flat_hand_mean=True,
        num_betas=10,
    ).to(device).eval()

    frame_metrics = []
    for out_idx, (seq, frame) in enumerate(avatarrex_specs(args.seq_a, args.seq_b, int(args.start_frame))):
        seq_key = raw_calibration_key(seq)
        root = args.avatarrex_root / args.split / seq
        stem = f"{frame:08d}"
        color = cv2.imread(str(root / "rgb" / f"{stem}.png"), cv2.IMREAD_COLOR)
        if color is None:
            raise FileNotFoundError(root / "rgb" / f"{stem}.png")
        mask_bgr = cv2.imread(str(root / "mask" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        if mask_bgr is None:
            raise FileNotFoundError(root / "mask" / f"{stem}.png")
        mask = (mask_bgr > 10).astype(np.float32)
        if args.depth_mode == "raw":
            depth = load_depth_meters(root / "depth" / f"{stem}.npy")
            conf = np.ones_like(depth, dtype=np.float32) * 10.0
        else:
            depth = np.zeros(color.shape[:2], dtype=np.float32)
            conf = np.zeros(color.shape[:2], dtype=np.float32)
        K = np.asarray(calibration[seq_key]["K"], dtype=np.float32).reshape(3, 3)
        pose = true_c2w_from_calibration(calibration[seq_key])
        smpl = build_gt_smpl(raw_smplx, smpl_data, calibration[seq_key], frame, mask, device)

        cv2.imwrite(str(args.output_dir / "color" / f"{out_idx:06d}.png"), color)
        np.save(args.output_dir / "depth" / f"{out_idx:06d}.npy", depth)
        np.save(args.output_dir / "conf" / f"{out_idx:06d}.npy", conf)
        np.savez(args.output_dir / "camera" / f"{out_idx:06d}.npz", pose=pose, intrinsics=K)
        np.savez(args.output_dir / "smpl" / f"{out_idx:06d}.npz", **smpl)
        frame_metrics.append({"idx": out_idx, "seq": seq, "frame": frame, "pose": pose.tolist()})

    summary = {
        "output_dir": str(args.output_dir),
        "coordinate_convention": "X_cam = R_w2c @ X_world + T_w2c; viewer pose = inv(raw calibration)",
        "depth_mode": args.depth_mode,
        "frames": frame_metrics,
    }
    with open(args.output_dir / "raw_gt_viewer_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
