#!/usr/bin/env python3
"""Build diagnostic saved-output dirs that disentangle camera and SMPL errors.

Given one saved Human3R/V8.1 output and the corresponding AvatarReX AABB sample,
this script writes:

  pred_camera_gt_smpl/  : predicted camera + GT SMPL
  gt_camera_pred_smpl/  : GT camera + predicted SMPL
  gt_camera_gt_smpl/    : GT camera + GT SMPL

The GT cameras are aligned to the prediction's frame-0 coordinate system so the
outputs can be compared in the same viewer world frame.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from smplx.joint_names import JOINT_NAMES

from dust3r.datasets.avatarrex import AvatarReX_AABB
from dust3r.smpl_model import SMPLModel
from dust3r.utils.device import todevice
from dust3r.utils.smpl_layer import SMPL_Layer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred_output_dir", type=Path, required=True)
    parser.add_argument("--avatarrex_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data/Avatarrex_output"))
    parser.add_argument("--split", default="Training")
    parser.add_argument("--seq_a", default="22010710")
    parser.add_argument("--seq_b", default="22053923")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--resolution", type=int, nargs=2, default=(512, 288), metavar=("W", "H"))
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def copy_tree(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"{dst} exists; pass --overwrite")
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def load_saved_cameras(output_dir: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    poses, intrinsics = [], []
    for path in sorted((output_dir / "camera").glob("*.npz")):
        cam = np.load(path)
        poses.append(cam["pose"].astype(np.float32))
        intrinsics.append(cam["intrinsics"].astype(np.float32))
    if not poses:
        raise FileNotFoundError(output_dir / "camera")
    return poses, intrinsics


def write_cameras(output_dir: Path, poses: list[np.ndarray], intrinsics: list[np.ndarray]) -> None:
    camera_dir = output_dir / "camera"
    camera_dir.mkdir(parents=True, exist_ok=True)
    for i, (pose, K) in enumerate(zip(poses, intrinsics)):
        np.savez(camera_dir / f"{i:06d}.npz", pose=pose.astype(np.float32), intrinsics=K.astype(np.float32))


def avatarrex_specs(seq_a: str, seq_b: str, start_frame: int) -> list[tuple[str, int]]:
    return [
        (seq_a, start_frame),
        (seq_a, start_frame + 1),
        (seq_b, start_frame + 2),
        (seq_b, start_frame + 3),
    ]


def load_gt_cameras(root: Path, split: str, specs: list[tuple[str, int]]) -> list[np.ndarray]:
    gt_poses = []
    for seq, frame in specs:
        cam = np.load(root / split / seq / "cam" / f"{frame:08d}.npz")
        gt_poses.append(cam["pose"].astype(np.float32))
    return gt_poses


def rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    rel = a[:3, :3] @ b[:3, :3].T
    angle = np.arccos(np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(angle))


def build_gt_smpl_npzs(args: argparse.Namespace, ref_output_dir: Path) -> list[dict[str, np.ndarray]]:
    dataset = AvatarReX_AABB(
        split=args.split,
        ROOT=str(args.avatarrex_root),
        resolution=tuple(args.resolution),
        num_views=4,
        aug_crop=0,
        allow_repeat=True,
        seed=args.seed,
        n_corres=0,
        fixed_samples=[(args.seq_a, args.seq_b, int(args.start_frame))],
    )
    views = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)))
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    views = todevice(views, device)
    smpl_model = SMPLModel(device, model_args={"patch_size": 16, "mhmr_img_res": 896, "bb_patch_size": 14})
    head_idx = JOINT_NAMES.index("head")

    gt_smpls = []
    with torch.no_grad():
        for i, view in enumerate(views):
            smpl_mask = view["smpl_mask"][0].bool()
            if not smpl_mask.any():
                raise RuntimeError(f"No raw GT SMPL in view {i} before visibility filtering")
            h = int(torch.where(smpl_mask)[0][0])
            shape = view["smplx_shape"][0, h].reshape(1, -1)
            root = view["smplx_root_pose"][0, h].reshape(1, 1, 3)
            body = view["smplx_body_pose"][0, h].reshape(1, 21, 3)
            left_hand = view["smplx_left_hand_pose"][0, h].reshape(1, 15, 3)
            right_hand = view["smplx_right_hand_pose"][0, h].reshape(1, 15, 3)
            jaw = view["smplx_jaw_pose"][0, h].reshape(1, 1, 3)
            transl = view["smplx_transl"][0, h].reshape(1, 3)
            expr = smpl_model.smplx_neutral_11.expression.repeat(1, 1)

            raw_out = smpl_model.smplx_neutral_11(
                global_orient=root.reshape(1, 3),
                body_pose=body.reshape(1, 21 * 3),
                jaw_pose=jaw.reshape(1, 3),
                leye_pose=view["smplx_leye_pose"][0, h].reshape(1, 3),
                reye_pose=view["smplx_reye_pose"][0, h].reshape(1, 3),
                left_hand_pose=left_hand.reshape(1, 15 * 3),
                right_hand_pose=right_hand.reshape(1, 15 * 3),
                betas=shape.reshape(1, -1),
                transl=transl,
                expression=expr,
            )
            head_cam = raw_out.joints[:, head_idx].detach().cpu().numpy().astype(np.float32)
            rotvec = torch.cat([root, body, left_hand, right_hand, jaw], dim=1)

            src_smpl = np.load(ref_output_dir / "smpl" / f"{i:06d}.npz", allow_pickle=True)
            gt_smpls.append(
                {
                    "scores": src_smpl["scores"].astype(np.float32),
                    "msk": src_smpl["msk"].astype(np.float32) if src_smpl["msk"] is not None else None,
                    "shape": shape.detach().cpu().numpy().astype(np.float32),
                    "rotvec": rotvec.detach().cpu().numpy().astype(np.float32),
                    "transl": head_cam,
                    "expression": expr.detach().cpu().numpy().astype(np.float32),
                    "smpl_id": np.asarray([0], dtype=np.int64),
                }
            )
    return gt_smpls


def write_smpl_npzs(output_dir: Path, smpls: list[dict[str, np.ndarray]]) -> None:
    smpl_dir = output_dir / "smpl"
    smpl_dir.mkdir(parents=True, exist_ok=True)
    for i, smpl in enumerate(smpls):
        np.savez(smpl_dir / f"{i:06d}.npz", **smpl)


def load_saved_smpl_world(output_dir: Path, device: torch.device):
    poses, _ = load_saved_cameras(output_dir)
    layer = None
    heads_world, pelvis_world, joint_world = [], [], []
    with torch.no_grad():
        for i, pose in enumerate(poses):
            smpl = np.load(output_dir / "smpl" / f"{i:06d}.npz", allow_pickle=True)
            if smpl["shape"].shape[0] == 0:
                heads_world.append(None)
                pelvis_world.append(None)
                joint_world.append(None)
                continue
            cam = np.load(output_dir / "camera" / f"{i:06d}.npz")
            K = torch.from_numpy(cam["intrinsics"].astype(np.float32)).to(device)
            shape = torch.from_numpy(smpl["shape"].astype(np.float32)).to(device)
            rotvec = torch.from_numpy(smpl["rotvec"].astype(np.float32)).to(device)
            transl = torch.from_numpy(smpl["transl"].astype(np.float32)).to(device)
            expr_np = smpl["expression"]
            expr = None if expr_np is None else torch.from_numpy(expr_np.astype(np.float32)).to(device)
            if layer is None or layer.num_betas != shape.shape[-1]:
                layer = SMPL_Layer(
                    type="smplx",
                    gender="neutral",
                    num_betas=shape.shape[-1],
                    kid=False,
                    person_center="head",
                ).to(device)
            out = layer(rotvec, shape, transl, None, None, K=K.expand(shape.shape[0], -1, -1), expression=expr)
            joints = out["smpl_j3d"][0].detach().cpu().numpy().astype(np.float32)
            pose_np = pose.astype(np.float32)
            joints_w = joints @ pose_np[:3, :3].T + pose_np[:3, 3]
            heads_world.append((out["smpl_transl"][0].detach().cpu().numpy() @ pose_np[:3, :3].T + pose_np[:3, 3]).astype(np.float32))
            pelvis_world.append(joints_w[0].astype(np.float32))
            joint_world.append(joints_w.astype(np.float32))
    return heads_world, pelvis_world, joint_world


def compute_metrics(pred_dir: Path, gt_smpl_dir: Path, aligned_gt_poses: list[np.ndarray]) -> dict:
    pred_poses, _ = load_saved_cameras(pred_dir)
    camera = []
    for i, (pred_pose, gt_pose) in enumerate(zip(pred_poses, aligned_gt_poses)):
        camera.append(
            {
                "frame": i,
                "center_err": float(np.linalg.norm(pred_pose[:3, 3] - gt_pose[:3, 3])),
                "rot_err_deg": rotation_error_deg(pred_pose, gt_pose),
                "pred_center": pred_pose[:3, 3].tolist(),
                "gt_center_aligned": gt_pose[:3, 3].tolist(),
            }
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred_head_w, pred_pelvis_w, pred_joints_w = load_saved_smpl_world(pred_dir, device)
    gt_head_w, gt_pelvis_w, gt_joints_w = load_saved_smpl_world(gt_smpl_dir, device)
    smpl = []
    for i in range(len(pred_head_w)):
        if pred_head_w[i] is None or gt_head_w[i] is None:
            continue
        n = min(pred_joints_w[i].shape[0], gt_joints_w[i].shape[0])
        smpl.append(
            {
                "frame": i,
                "head_world_err": float(np.linalg.norm(pred_head_w[i] - gt_head_w[i])),
                "pelvis_world_err": float(np.linalg.norm(pred_pelvis_w[i] - gt_pelvis_w[i])),
                "mean_joint_world_err": float(np.linalg.norm(pred_joints_w[i][:n] - gt_joints_w[i][:n], axis=1).mean()),
                "pred_head_world": pred_head_w[i].tolist(),
                "gt_head_world": gt_head_w[i].tolist(),
                "pred_pelvis_world": pred_pelvis_w[i].tolist(),
                "gt_pelvis_world": gt_pelvis_w[i].tolist(),
            }
        )
    return {"camera": camera, "smpl_world": smpl}


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    pred_poses, pred_intrinsics = load_saved_cameras(args.pred_output_dir)
    specs = avatarrex_specs(args.seq_a, args.seq_b, int(args.start_frame))
    gt_poses = load_gt_cameras(args.avatarrex_root, args.split, specs)
    align = pred_poses[0] @ np.linalg.inv(gt_poses[0])
    aligned_gt_poses = [(align @ gt_pose).astype(np.float32) for gt_pose in gt_poses]
    gt_smpls = build_gt_smpl_npzs(args, args.pred_output_dir)

    pred_camera_gt_smpl = args.output_root / "pred_camera_gt_smpl"
    gt_camera_pred_smpl = args.output_root / "gt_camera_pred_smpl"
    gt_camera_gt_smpl = args.output_root / "gt_camera_gt_smpl"

    copy_tree(args.pred_output_dir, pred_camera_gt_smpl, args.overwrite)
    write_smpl_npzs(pred_camera_gt_smpl, gt_smpls)

    copy_tree(args.pred_output_dir, gt_camera_pred_smpl, args.overwrite)
    write_cameras(gt_camera_pred_smpl, aligned_gt_poses, pred_intrinsics)

    copy_tree(args.pred_output_dir, gt_camera_gt_smpl, args.overwrite)
    write_cameras(gt_camera_gt_smpl, aligned_gt_poses, pred_intrinsics)
    write_smpl_npzs(gt_camera_gt_smpl, gt_smpls)

    metrics = {
        "pred_output_dir": str(args.pred_output_dir),
        "gt_camera_alignment": "T_gt_aligned_i = T_pred_0 @ inv(T_gt_0) @ T_gt_i",
        "outputs": {
            "pred_camera_gt_smpl": str(pred_camera_gt_smpl),
            "gt_camera_pred_smpl": str(gt_camera_pred_smpl),
            "gt_camera_gt_smpl": str(gt_camera_gt_smpl),
        },
        "pred_vs_gt": compute_metrics(args.pred_output_dir, gt_camera_gt_smpl, aligned_gt_poses),
    }
    metrics_path = args.output_root / "diagnostic_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({"output_root": str(args.output_root), "metrics": str(metrics_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
