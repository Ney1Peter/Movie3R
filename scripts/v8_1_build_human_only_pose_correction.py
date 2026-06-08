#!/usr/bin/env python3
"""Build a human-only oracle pose-correction saved-output directory.

This is a V8.1 diagnostic, not the final pose-correction module.  It answers:

    If the correction branch only used explicit human anchors, what would the
    Human3R viewer look like?

The script starts from an existing Human3R ``raw/`` saved-output directory.  It
keeps depth, confidence, RGB, and SMPL parameters unchanged, and replaces only
``camera/*.npz``.  The new camera poses are estimated by fitting Human3R's
predicted SMPL camera-space body joints to AvatarReX GT SMPL world joints.

No GT camera pose, pointmap overlap, scene/background feature, or confidence map
is used for the correction itself.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import smplx
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dust3r.utils.geometry import geotrf  # noqa: E402
from dust3r.utils.smpl_layer import SMPL_Layer  # noqa: E402


FULL_BODY_ANCHOR_NAMES = {
    0: "pelvis",
    1: "left_hip",
    2: "right_hip",
    3: "spine1",
    4: "left_knee",
    5: "right_knee",
    6: "spine2",
    7: "left_ankle",
    8: "right_ankle",
    9: "spine3",
    10: "left_foot",
    11: "right_foot",
    12: "neck",
    13: "left_collar",
    14: "right_collar",
    16: "left_shoulder",
    17: "right_shoulder",
}

TOKEN_BODY_ANCHOR_SPECS = [
    ("pelvis", [0]),
    ("torso", [6, 9, 12]),
    ("left_foot", [10]),
    ("right_foot", [11]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare_root", type=Path, default=REPO_ROOT / "output" / "v8_1_human3r_aabb_compare")
    parser.add_argument("--raw_dir", type=Path, default=None)
    parser.add_argument("--output_name", default="human_only")
    parser.add_argument("--avatarrex_raw_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data/AvatarReX_raw_meta/lbn1"))
    parser.add_argument(
        "--anchor_set",
        choices=["full_body", "token_body"],
        default="full_body",
        help="full_body uses many stable SMPL joints; token_body uses only anchors validated by token probes.",
    )
    parser.add_argument(
        "--anchor_indices",
        default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17",
        help="Comma-separated SMPL-X joint indices used when --anchor_set full_body.",
    )
    parser.add_argument(
        "--target_similarity",
        action="store_true",
        default=True,
        help="Use a frame-0 similarity alignment from AvatarReX GT human space to Human3R raw world space.",
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def build_anchor_specs(args: argparse.Namespace) -> list[tuple[str, list[int]]]:
    if args.anchor_set == "token_body":
        return TOKEN_BODY_ANCHOR_SPECS
    anchor_indices = [int(x) for x in args.anchor_indices.split(",") if x.strip()]
    return [(FULL_BODY_ANCHOR_NAMES.get(i, f"joint_{i}"), [i]) for i in anchor_indices]


def anchor_indices_from_specs(anchor_specs: list[tuple[str, list[int]]]) -> list[int]:
    return sorted({idx for _, indices in anchor_specs for idx in indices})


def select_anchor_points(joints: np.ndarray, anchor_specs: list[tuple[str, list[int]]]) -> np.ndarray:
    points = []
    for _, indices in anchor_specs:
        pts = joints[np.asarray(indices, dtype=np.int64)]
        points.append(pts.mean(axis=0))
    return np.stack(points, axis=0).astype(np.float64)


def read_case_manifest(compare_root: Path) -> list[dict]:
    manifest_path = compare_root / "case_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    specs = data["input_frames"]
    for required in ("idx", "seq", "frame"):
        if any(required not in item for item in specs):
            raise KeyError(f"case_manifest input_frames missing {required}")
    return specs


def copy_saved_output_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    copied_video = dst / "output_video.mp4"
    if copied_video.exists():
        copied_video.unlink()


def load_camera_poses(output_dir: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    cam_files = sorted((output_dir / "camera").glob("*.npz"))
    poses, intrinsics = [], []
    for path in cam_files:
        data = np.load(path)
        poses.append(data["pose"].astype(np.float64))
        intrinsics.append(data["intrinsics"].astype(np.float64))
    return poses, intrinsics


def write_camera_poses(output_dir: Path, poses: list[np.ndarray], intrinsics: list[np.ndarray]) -> None:
    cam_dir = output_dir / "camera"
    for i, (pose, K) in enumerate(zip(poses, intrinsics)):
        np.savez(cam_dir / f"{i:06d}.npz", pose=pose.astype(np.float32), intrinsics=K.astype(np.float32))


def transform_points(T: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    return xyz @ T[:3, :3].T + T[:3, 3]


def fit_rigid(src: np.ndarray, dst: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Return T with dst ~= src @ R.T + t."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"Expected Nx3 src/dst with same shape, got {src.shape} and {dst.shape}")
    if weights is None:
        weights = np.ones(src.shape[0], dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / np.maximum(weights.sum(), 1e-12)
    src_mu = (src * weights[:, None]).sum(axis=0)
    dst_mu = (dst * weights[:, None]).sum(axis=0)
    src_c = src - src_mu
    dst_c = dst - dst_mu
    H = (src_c * weights[:, None]).T @ dst_c
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = dst_mu - R @ src_mu
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def fit_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return s, R, t with dst ~= s * (src @ R.T) + t."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    src_mu = src.mean(axis=0)
    dst_mu = dst.mean(axis=0)
    src_c = src - src_mu
    dst_c = dst - dst_mu
    H = src_c.T @ dst_c / max(len(src), 1)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    denom = np.sum(src_c * src_c) / max(len(src), 1)
    scale = float(np.trace(np.diag(np.linalg.svd(H, compute_uv=False))) / max(denom, 1e-12))
    # Recompute scale after the reflection-safe rotation. This is more stable
    # for nearly-planar body anchor sets.
    scale = float(np.sum(dst_c * (src_c @ R.T)) / max(np.sum(src_c * src_c), 1e-12))
    t = dst_mu - scale * (R @ src_mu)
    return scale, R, t


def apply_similarity(src: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return scale * (src @ R.T) + t


def load_gt_smpl_joints(raw_root: Path, frames: list[int], anchor_specs: list[tuple[str, list[int]]]) -> list[np.ndarray]:
    smpl_path = raw_root / "smpl_params.npz"
    if not smpl_path.is_file():
        raise FileNotFoundError(smpl_path)
    smpl_data = np.load(smpl_path)
    model = smplx.create(
        str(SRC_ROOT / "models"),
        "smplx",
        gender="neutral",
        use_pca=False,
        flat_hand_mean=True,
        num_betas=10,
    ).eval()
    joints = []
    with torch.no_grad():
        for frame in frames:
            out = model(
                global_orient=torch.tensor(smpl_data["global_orient"][frame], dtype=torch.float32).reshape(1, 3),
                body_pose=torch.tensor(smpl_data["body_pose"][frame], dtype=torch.float32).reshape(1, 63),
                jaw_pose=torch.tensor(smpl_data["jaw_pose"][frame], dtype=torch.float32).reshape(1, 3),
                leye_pose=torch.zeros(1, 3),
                reye_pose=torch.zeros(1, 3),
                left_hand_pose=torch.tensor(smpl_data["left_hand_pose"][frame], dtype=torch.float32).reshape(1, 45),
                right_hand_pose=torch.tensor(smpl_data["right_hand_pose"][frame], dtype=torch.float32).reshape(1, 45),
                betas=torch.tensor(smpl_data["betas"][0], dtype=torch.float32).reshape(1, 10),
                transl=torch.tensor(smpl_data["transl"][frame], dtype=torch.float32).reshape(1, 3),
                expression=torch.tensor(smpl_data["expression"][frame], dtype=torch.float32).reshape(1, 10),
            )
            full_joints = out.joints[0].cpu().numpy().astype(np.float64)
            joints.append(select_anchor_points(full_joints, anchor_specs))
    return joints


def load_human3r_smpl_joints(raw_dir: Path, intrinsics: list[np.ndarray], anchor_specs: list[tuple[str, list[int]]], device: str) -> list[np.ndarray]:
    smpl_paths = sorted((raw_dir / "smpl").glob("*.npz"))
    if len(smpl_paths) != len(intrinsics):
        raise RuntimeError(f"SMPL files ({len(smpl_paths)}) and camera files ({len(intrinsics)}) differ")

    beta_dim = None
    for path in smpl_paths:
        data = np.load(path, allow_pickle=True)
        if data["shape"].shape[0] > 0:
            beta_dim = data["shape"].shape[-1]
            break
    if beta_dim is None:
        raise RuntimeError("No Human3R SMPL predictions found")

    layer = SMPL_Layer(type="smplx", gender="neutral", num_betas=int(beta_dim), kid=False, person_center="head").to(device)
    joints = []
    with torch.no_grad():
        for i, path in enumerate(smpl_paths):
            data = np.load(path, allow_pickle=True)
            shape = data["shape"].astype(np.float32)
            if shape.shape[0] == 0:
                raise RuntimeError(f"No human in {path}")
            # First version: single-person diagnostic. Use the highest-scoring
            # person if there are multiple predictions later.
            person_idx = 0
            rotvec = data["rotvec"][person_idx : person_idx + 1].astype(np.float32)
            transl = data["transl"][person_idx : person_idx + 1].astype(np.float32)
            shape = shape[person_idx : person_idx + 1]
            expr = data["expression"]
            expr_t = None
            if expr is not None:
                expr_t = torch.from_numpy(expr[person_idx : person_idx + 1].astype(np.float32)).to(device)
            out = layer(
                torch.from_numpy(rotvec).to(device),
                torch.from_numpy(shape).to(device),
                torch.from_numpy(transl).to(device),
                None,
                None,
                K=torch.from_numpy(intrinsics[i].astype(np.float32)).to(device).unsqueeze(0),
                expression=expr_t,
            )
            full_joints = out["smpl_j3d"][0].detach().cpu().numpy().astype(np.float64)
            joints.append(select_anchor_points(full_joints, anchor_specs))
    return joints


def rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(a * a, axis=-1))))


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir or (args.compare_root / "raw")
    out_dir = args.compare_root / args.output_name
    specs = read_case_manifest(args.compare_root)
    anchor_specs = build_anchor_specs(args)
    anchor_indices = anchor_indices_from_specs(anchor_specs)
    anchor_names = [name for name, _ in anchor_specs]

    copy_saved_output_tree(raw_dir, out_dir)
    raw_poses, intrinsics = load_camera_poses(raw_dir)
    frames = [int(item["frame"]) for item in specs]

    pred_cam_joints = load_human3r_smpl_joints(raw_dir, intrinsics, anchor_specs, args.device)
    gt_world_joints = load_gt_smpl_joints(args.avatarrex_raw_root, frames, anchor_specs)

    raw_world_joints = [transform_points(pose, joints) for pose, joints in zip(raw_poses, pred_cam_joints)]
    if args.target_similarity:
        scale, R_align, t_align = fit_similarity(gt_world_joints[0], raw_world_joints[0])
    else:
        T_align = fit_rigid(gt_world_joints[0], raw_world_joints[0])
        scale, R_align, t_align = 1.0, T_align[:3, :3], T_align[:3, 3]
    target_world_joints = [apply_similarity(j, scale, R_align, t_align) for j in gt_world_joints]

    human_only_poses = []
    per_frame = []
    for i, (src_cam, target, raw_pose, raw_world) in enumerate(zip(pred_cam_joints, target_world_joints, raw_poses, raw_world_joints)):
        T = fit_rigid(src_cam, target)
        if i == 0:
            # Keep the viewer coordinate frame exactly identical to raw frame 0.
            T = raw_pose.copy()
        human_world = transform_points(T, src_cam)
        human_only_poses.append(T)
        per_frame.append(
            {
                "idx": i,
                "seq": specs[i]["seq"],
                "frame": frames[i],
                "raw_anchor_rmse": rms(raw_world - target),
                "human_only_anchor_rmse": rms(human_world - target),
                "raw_camera_center": raw_pose[:3, 3].tolist(),
                "human_only_camera_center": T[:3, 3].tolist(),
            }
        )

    write_camera_poses(out_dir, human_only_poses, intrinsics)
    summary = {
        "correction_type": "human_only_explicit_smpl_anchor_fit",
        "note": "Diagnostic only. Depth/conf/color/SMPL parameters are unchanged; only camera poses are replaced.",
        "used_information": [
            "Human3R predicted SMPL camera-space joints from raw/smpl/*.npz",
            "AvatarReX GT SMPL world joints from smpl_params.npz",
            "Frame-0 human-anchor alignment to put GT human space into raw Human3R world space",
        ],
        "not_used_information": [
            "AvatarReX GT camera extrinsics",
            "Human3R pointmap/depth for correction",
            "background/scene tokens",
            "confidence map",
            "feature matching",
        ],
        "anchor_indices": anchor_indices,
        "anchor_set": args.anchor_set,
        "anchor_specs": [{"name": name, "joint_indices": indices} for name, indices in anchor_specs],
        "anchor_names": anchor_names,
        "target_alignment": {
            "type": "similarity_gt_human_frame0_to_raw_human_frame0" if args.target_similarity else "rigid_gt_human_frame0_to_raw_human_frame0",
            "scale": float(scale),
            "rotation": R_align.tolist(),
            "translation": t_align.tolist(),
        },
        "per_frame": per_frame,
    }
    with open(out_dir / "human_only_correction_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(json.dumps({"output_dir": str(out_dir), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
