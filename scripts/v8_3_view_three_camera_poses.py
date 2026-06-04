#!/usr/bin/env python3
"""Show GT, raw Human3R, and corrected camera poses in one Viser viewer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import viser.transforms as tf

from dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates, geotrf
from dust3r.utils.smpl_layer import SMPL_Layer
from viser_utils import SceneHumanViewer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case_dir", type=Path, required=True)
    parser.add_argument("--raw_dir", type=Path, default=None)
    parser.add_argument("--corrected_dir", type=Path, default=None)
    parser.add_argument("--data_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data"))
    parser.add_argument("--split", type=str, default="training")
    parser.add_argument(
        "--raw_calibration_root",
        type=Path,
        default=None,
        help="Raw AvatarReX root with calibration_full.json. Defaults to data/avatarrex_<group>.",
    )
    parser.add_argument("--frame_ids", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--viewer_port", type=int, default=8135)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--vis_threshold", type=float, default=1.5)
    parser.add_argument("--downsample_factor", type=int, default=1)
    parser.add_argument("--smpl_downsample", type=int, default=1)
    parser.add_argument("--camera_downsample", type=int, default=1)
    return parser.parse_args()


def load_pose_k(output_dir: Path, frame_id: int) -> tuple[np.ndarray, np.ndarray]:
    cam = np.load(output_dir / "camera" / f"{frame_id:06d}.npz")
    return cam["pose"].astype(np.float32), cam["intrinsics"].astype(np.float32)


def load_cam_dict_frames(output_dir: Path, frame_ids: list[int]) -> dict[str, np.ndarray]:
    focal, pp, R, t = [], [], [], []
    for frame_id in frame_ids:
        pose, K = load_pose_k(output_dir, frame_id)
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


def load_viewer_payload_frames(output_dir: Path, frame_ids: list[int], device: str):
    pts3ds, colors, confs, msks = [], [], [], []
    smpl_shapes, smpl_rotvecs, smpl_transls, smpl_exprs, poses, intrinsics, smpl_ids = [], [], [], [], [], [], []

    for frame_id in frame_ids:
        pose, K = load_pose_k(output_dir, frame_id)
        depth = np.load(output_dir / "depth" / f"{frame_id:06d}.npy").astype(np.float32)
        conf = np.load(output_dir / "conf" / f"{frame_id:06d}.npy").astype(np.float32)
        color_bgr = cv2.imread(str(output_dir / "color" / f"{frame_id:06d}.png"), cv2.IMREAD_COLOR)
        if color_bgr is None:
            raise FileNotFoundError(output_dir / "color" / f"{frame_id:06d}.png")
        color = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        pc_world, _ = depthmap_to_absolute_camera_coordinates(depth, K, pose)

        smpl = np.load(output_dir / "smpl" / f"{frame_id:06d}.npz", allow_pickle=True)
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
        for idx in range(len(frame_ids)):
            n_humans = smpl_shapes[idx].shape[0]
            if n_humans == 0:
                all_verts.append(np.empty((0, 0, 3), dtype=np.float32))
                continue
            expr = None if smpl_exprs[idx] is None else torch.from_numpy(smpl_exprs[idx]).to(device=device, dtype=torch.float32)
            out = smpl_layer(
                torch.from_numpy(smpl_rotvecs[idx]).to(device=device, dtype=torch.float32),
                torch.from_numpy(smpl_shapes[idx]).to(device=device, dtype=torch.float32),
                torch.from_numpy(smpl_transls[idx]).to(device=device, dtype=torch.float32),
                None,
                None,
                K=torch.from_numpy(intrinsics[idx]).to(device=device, dtype=torch.float32).expand(n_humans, -1, -1),
                expression=expr,
            )
            verts_world = geotrf(
                torch.from_numpy(poses[idx]).to(device=device, dtype=torch.float32).unsqueeze(0),
                out["smpl_v3d"].unsqueeze(0),
            )[0]
            all_verts.append(verts_world.detach().cpu().numpy().astype(np.float32))

    return pts3ds, colors, confs, all_verts, smpl_faces, smpl_ids, msks


def load_manifest(case_dir: Path) -> dict:
    return json.loads((case_dir / "input_manifest.json").read_text())


def load_raw_calibration(root: Path) -> dict:
    path = root / "calibration_full.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def raw_calibration_c2w(calibration: dict, seq: str) -> np.ndarray:
    """Match AvatarReX_AABB raw_camera_pose target: X_cam = R @ X_world + T."""
    seq_key = seq.split("/", 1)[1] if "/" in seq else seq
    if seq_key not in calibration:
        raise KeyError(f"{seq_key} not found in raw calibration")
    cal = calibration[seq_key]
    R_w2c = np.asarray(cal["R"], dtype=np.float32).reshape(3, 3)
    T_w2c = np.asarray(cal["T"], dtype=np.float32).reshape(3)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R_w2c.T
    pose[:3, 3] = -R_w2c.T @ T_w2c
    return pose


def rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    rel = a[:3, :3] @ b[:3, :3].T
    angle = np.arccos(np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(angle))


def pose_errors(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    return rotation_error_deg(pred, gt), float(np.linalg.norm(pred[:3, 3] - gt[:3, 3]))


def add_camera_set(viewer: SceneHumanViewer, cam_dict: dict[str, np.ndarray], frame_ids: list[int], color: tuple[int, int, int], prefix: str) -> None:
    for step, original_frame_id in enumerate(frame_ids):
        focal = cam_dict["focal"][step]
        pp = cam_dict["pp"][step]
        R = cam_dict["R"][step]
        t = cam_dict["t"][step]
        q = tf.SO3.from_matrix(R).wxyz
        fov = 2 * np.arctan(pp[0] / focal)
        aspect = pp[0] / pp[1]
        viewer.server.add_camera_frustum(
            name=f"/frames/{step}/{prefix}_camera",
            fov=fov,
            aspect=aspect,
            wxyz=q,
            position=t,
            scale=0.14,
            color=color,
        )
        viewer.server.scene.add_label(
            f"/frames/{step}/{prefix}_label",
            f"{prefix.upper()} f{original_frame_id}",
            position=t + np.array([0.0, 0.08, 0.0], dtype=np.float32),
            font_size_mode="scene",
            font_scene_height=0.055,
            depth_test=False,
        )


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir or args.case_dir / "raw_human3r"
    corrected_dir = args.corrected_dir or args.case_dir / "v831_corrected"
    frame_ids = [int(x) for x in args.frame_ids]

    manifest = load_manifest(args.case_dir)
    group = manifest.get("group") or str(manifest["frames"][0]["seq"]).split("/", 1)[0]
    raw_calibration_root = args.raw_calibration_root or (args.data_root / f"avatarrex_{group}")
    calibration = load_raw_calibration(raw_calibration_root)
    gt_abs = []
    for item in manifest["frames"]:
        gt_abs.append(raw_calibration_c2w(calibration, item["seq"]))

    raw0, _ = load_pose_k(raw_dir, 0)
    # V8 losses supervise raw_camera_pose as inv(raw_gt_0) @ raw_gt_i.
    # The saved viewer output is in the same canonical frame-0 gauge, with only
    # a tiny nonzero frame-0 offset, so left-align relative GT to raw frame 0.
    gt0_inv = np.linalg.inv(gt_abs[0])
    gt_aligned = [(raw0 @ gt0_inv @ pose).astype(np.float32) for pose in gt_abs]
    gt_cam_dict = {
        "focal": [],
        "pp": [],
        "R": [],
        "t": [],
    }
    for idx in frame_ids:
        pose = gt_aligned[idx]
        _, K = load_pose_k(raw_dir, idx)
        gt_cam_dict["focal"].append(float(0.5 * (K[0, 0] + K[1, 1])))
        gt_cam_dict["pp"].append(K[:2, 2])
        gt_cam_dict["R"].append(pose[:3, :3])
        gt_cam_dict["t"].append(pose[:3, 3])
    gt_cam_dict = {k: np.asarray(v, dtype=np.float32) for k, v in gt_cam_dict.items()}

    corrected_cam_dict = load_cam_dict_frames(corrected_dir, frame_ids)
    raw_cam_dict = load_cam_dict_frames(raw_dir, frame_ids)
    payload = load_viewer_payload_frames(corrected_dir, frame_ids, args.device)

    print(f"GT source: raw calibration target from {raw_calibration_root}")
    print("Pose errors against raw-calibration relative GT, aligned to raw frame 0:")
    for local_idx, frame_id in enumerate(frame_ids):
        gt_pose = np.eye(4, dtype=np.float32)
        gt_pose[:3, :3] = gt_cam_dict["R"][local_idx]
        gt_pose[:3, 3] = gt_cam_dict["t"][local_idx]
        raw_pose, _ = load_pose_k(raw_dir, frame_id)
        corrected_pose, _ = load_pose_k(corrected_dir, frame_id)
        raw_rot, raw_trans = pose_errors(raw_pose, gt_pose)
        corr_rot, corr_trans = pose_errors(corrected_pose, gt_pose)
        print(
            f"  frame {frame_id}: raw rot={raw_rot:.3f} deg trans={raw_trans:.4f}; "
            f"corrected rot={corr_rot:.3f} deg trans={corr_trans:.4f}"
        )

    print("Color legend: GT=red, Human3R raw=gray, corrected=yellow.")
    print(f"Open http://127.0.0.1:{args.viewer_port} after forwarding this port.")
    viewer = SceneHumanViewer(
        payload[0],
        payload[1],
        payload[2],
        corrected_cam_dict,
        payload[3],
        payload[4],
        payload[5],
        payload[6],
        device=args.device,
        port=args.viewer_port,
        edge_color_list=[None] * len(frame_ids),
        show_camera=False,
        show_gt_camera=False,
        vis_threshold=args.vis_threshold,
        downsample_factor=args.downsample_factor,
        smpl_downsample_factor=args.smpl_downsample,
        camera_downsample_factor=args.camera_downsample,
    )
    add_camera_set(viewer, gt_cam_dict, frame_ids, color=(255, 40, 40), prefix="gt")
    add_camera_set(viewer, raw_cam_dict, frame_ids, color=(150, 150, 150), prefix="human3r")
    add_camera_set(viewer, corrected_cam_dict, frame_ids, color=(255, 220, 0), prefix="corrected")
    viewer.server.scene.add_label(
        "/legend",
        "GT red | Human3R raw gray | corrected yellow",
        position=np.asarray([0.0, -0.4, 0.0], dtype=np.float32),
        font_size_mode="scene",
        font_scene_height=0.07,
        depth_test=False,
    )
    viewer.run()


if __name__ == "__main__":
    main()
