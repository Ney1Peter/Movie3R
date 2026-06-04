#!/usr/bin/env python3
"""Build a token-aligned human+local-scene pose-correction diagnostic.

This script is an explicit proxy for the future token-based correction module.
It only uses explicit regions that were validated in the V8.1 token probe:

  human anchors: pelvis, torso, left_foot, right_foot
  scene anchors: near_foot, optionally near_human background ring

It keeps Human3R depth/color/SMPL outputs unchanged and replaces only saved
camera poses.  Unlike the camera-oracle comparison, it does not copy GT camera
poses as the correction target.  It fits camera pose from Human3R camera-space
human anchors plus local scene depth points to GT human/scene anchors aligned to
the raw Human3R frame-0 coordinate system.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import smplx
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dust3r.utils.smpl_layer import SMPL_Layer  # noqa: E402


TOKEN_BODY_ANCHOR_SPECS = [
    ("pelvis", [0]),
    ("torso", [6, 9, 12]),
    ("left_foot", [10]),
    ("right_foot", [11]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare_root", type=Path, default=REPO_ROOT / "output" / "v8_1_human3r_aabb_compare")
    parser.add_argument("--output_name", default="token_aligned_human_nearfoot_scene")
    parser.add_argument("--avatarrex_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data"))
    parser.add_argument("--avatarrex_raw_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data/avatarrex_lbn1"))
    parser.add_argument("--split", default="Training")
    parser.add_argument(
        "--scene_regions",
        default="near_foot",
        help="Comma-separated scene regions from {near_foot,near_human}.",
    )
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--scene_samples_per_frame", type=int, default=512)
    parser.add_argument("--human_total_weight", type=float, default=0.65)
    parser.add_argument("--scene_total_weight", type=float, default=0.35)
    parser.add_argument(
        "--mode",
        choices=["joint_fit", "human_then_scene_translation"],
        default="joint_fit",
        help="joint_fit fits all anchors together; human_then_scene_translation preserves human rotation and applies a clipped scene translation residual.",
    )
    parser.add_argument("--scene_translation_alpha", type=float, default=0.35)
    parser.add_argument("--scene_translation_clip", type=float, default=0.25)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def read_case_manifest(compare_root: Path) -> list[dict]:
    manifest_path = compare_root / "case_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return data["input_frames"]


def copy_saved_output_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    copied_video = dst / "output_video.mp4"
    if copied_video.exists():
        copied_video.unlink()


def load_camera_poses(output_dir: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    poses, intrinsics = [], []
    for path in sorted((output_dir / "camera").glob("*.npz")):
        data = np.load(path)
        poses.append(data["pose"].astype(np.float64))
        intrinsics.append(data["intrinsics"].astype(np.float64))
    return poses, intrinsics


def write_camera_poses(output_dir: Path, poses: list[np.ndarray], intrinsics: list[np.ndarray]) -> None:
    for i, (pose, K) in enumerate(zip(poses, intrinsics)):
        np.savez(output_dir / "camera" / f"{i:06d}.npz", pose=pose.astype(np.float32), intrinsics=K.astype(np.float32))


def read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return (mask > 10).astype(np.uint8)


def resize_crop_params(width: int, height: int, size: int) -> dict:
    scale = size / max(width, height)
    resized_w = int(round(width * scale))
    resized_h = int(round(height * scale))
    cx, cy = resized_w // 2, resized_h // 2
    halfw = ((2 * cx) // 16) * 8
    halfh = ((2 * cy) // 16) * 8
    if resized_w == resized_h:
        halfh = int(3 * halfw / 4)
    x0, x1 = int(cx - halfw), int(cx + halfw)
    y0, y1 = int(cy - halfh), int(cy + halfh)
    return {
        "scale": scale,
        "resized_w": resized_w,
        "resized_h": resized_h,
        "crop_x0": x0,
        "crop_y0": y0,
        "crop_x1": x1,
        "crop_y1": y1,
        "out_w": x1 - x0,
        "out_h": y1 - y0,
    }


def resize_crop_image(img: np.ndarray, params: dict, interpolation: int) -> np.ndarray:
    resized = cv2.resize(img, (params["resized_w"], params["resized_h"]), interpolation=interpolation)
    return resized[params["crop_y0"] : params["crop_y1"], params["crop_x0"] : params["crop_x1"]]


def update_intrinsics(K: np.ndarray, params: dict) -> np.ndarray:
    out = K.astype(np.float64).copy()
    out[0, :] *= params["scale"]
    out[1, :] *= params["scale"]
    out[0, 2] -= params["crop_x0"]
    out[1, 2] -= params["crop_y0"]
    return out


def resize_crop_uv(uv_orig: np.ndarray, params: dict) -> np.ndarray:
    uv = uv_orig.astype(np.float64).copy()
    valid = np.isfinite(uv).all(axis=1)
    uv[valid, 0] = uv[valid, 0] * params["scale"] - params["crop_x0"]
    uv[valid, 1] = uv[valid, 1] * params["scale"] - params["crop_y0"]
    return uv


def add_circle(mask: np.ndarray, uv: tuple[float, float], radius: int) -> np.ndarray:
    out = mask.copy()
    if np.isfinite(uv).all():
        cv2.circle(out, (int(round(uv[0])), int(round(uv[1]))), radius, 1, -1)
    return out.astype(bool)


def mean_valid_uv(joints_uv: np.ndarray, indices: list[int]) -> tuple[float, float]:
    pts = joints_uv[indices]
    valid = np.isfinite(pts).all(axis=1)
    if not valid.any():
        raise RuntimeError(f"Invalid projected joints: {indices}")
    return float(pts[valid, 0].mean()), float(pts[valid, 1].mean())


def make_token_aligned_regions(human_mask: np.ndarray, joints_uv: np.ndarray) -> dict[str, np.ndarray]:
    pelvis = tuple(joints_uv[0].tolist())
    torso = mean_valid_uv(joints_uv, [6, 9, 12])
    left_foot = tuple(joints_uv[10].tolist())
    right_foot = tuple(joints_uv[11].tolist())
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (49, 49))
    dilated = cv2.dilate(human_mask.astype(np.uint8), kernel).astype(bool)
    near_human = dilated & (~human_mask)
    near_foot = np.zeros(human_mask.shape, dtype=bool)
    for foot in [left_foot, right_foot]:
        near_foot = add_circle(near_foot.astype(np.uint8), foot, 34)
    near_foot = near_foot & (~human_mask)
    return {
        "pelvis": add_circle(np.zeros(human_mask.shape, dtype=np.uint8), pelvis, 20) & human_mask,
        "torso": add_circle(np.zeros(human_mask.shape, dtype=np.uint8), torso, 24) & human_mask,
        "left_foot": add_circle(np.zeros(human_mask.shape, dtype=np.uint8), left_foot, 18) & human_mask,
        "right_foot": add_circle(np.zeros(human_mask.shape, dtype=np.uint8), right_foot, 18) & human_mask,
        "near_foot": near_foot,
        "near_human": near_human,
    }


class AvatarReXHelper:
    def __init__(self, raw_root: Path):
        self.raw_root = raw_root
        with open(raw_root / "calibration_full.json", "r", encoding="utf-8") as f:
            self.calibration = json.load(f)
        self.smpl_data = np.load(raw_root / "smpl_params.npz")
        self.model = smplx.create(
            str(SRC_ROOT / "models"),
            "smplx",
            gender="neutral",
            use_pca=False,
            flat_hand_mean=True,
            num_betas=10,
        ).eval()

    def c2w(self, seq: str) -> np.ndarray:
        cal = self.calibration[seq]
        R_w2c = np.asarray(cal["R"], dtype=np.float64).reshape(3, 3)
        T_w2c = np.asarray(cal["T"], dtype=np.float64).reshape(3)
        out = np.eye(4, dtype=np.float64)
        out[:3, :3] = R_w2c.T
        out[:3, 3] = -R_w2c.T @ T_w2c
        return out

    def intrinsics(self, seq: str) -> np.ndarray:
        return np.asarray(self.calibration[seq]["K"], dtype=np.float64).reshape(3, 3)

    def project_world(self, seq: str, xyz_world: np.ndarray) -> np.ndarray:
        cal = self.calibration[seq]
        R = np.asarray(cal["R"], dtype=np.float64).reshape(3, 3)
        T = np.asarray(cal["T"], dtype=np.float64).reshape(3)
        K = self.intrinsics(seq)
        xyz_cam = xyz_world @ R.T + T
        uv = np.full((xyz_world.shape[0], 2), np.nan, dtype=np.float64)
        valid = np.isfinite(xyz_cam).all(axis=1) & (xyz_cam[:, 2] > 1e-4)
        if valid.any():
            proj = xyz_cam[valid, :2] / xyz_cam[valid, 2:3]
            proj[:, 0] = proj[:, 0] * K[0, 0] + K[0, 2]
            proj[:, 1] = proj[:, 1] * K[1, 1] + K[1, 2]
            uv[valid] = proj
        return uv

    def smpl_joints(self, frame: int) -> np.ndarray:
        with torch.no_grad():
            out = self.model(
                global_orient=torch.tensor(self.smpl_data["global_orient"][frame], dtype=torch.float32).reshape(1, 3),
                body_pose=torch.tensor(self.smpl_data["body_pose"][frame], dtype=torch.float32).reshape(1, 63),
                jaw_pose=torch.tensor(self.smpl_data["jaw_pose"][frame], dtype=torch.float32).reshape(1, 3),
                leye_pose=torch.zeros(1, 3),
                reye_pose=torch.zeros(1, 3),
                left_hand_pose=torch.tensor(self.smpl_data["left_hand_pose"][frame], dtype=torch.float32).reshape(1, 45),
                right_hand_pose=torch.tensor(self.smpl_data["right_hand_pose"][frame], dtype=torch.float32).reshape(1, 45),
                betas=torch.tensor(self.smpl_data["betas"][0], dtype=torch.float32).reshape(1, 10),
                transl=torch.tensor(self.smpl_data["transl"][frame], dtype=torch.float32).reshape(1, 3),
                expression=torch.tensor(self.smpl_data["expression"][frame], dtype=torch.float32).reshape(1, 10),
            )
        return out.joints[0].cpu().numpy().astype(np.float64)


def select_anchor_points(joints: np.ndarray) -> np.ndarray:
    points = []
    for _, indices in TOKEN_BODY_ANCHOR_SPECS:
        points.append(joints[np.asarray(indices, dtype=np.int64)].mean(axis=0))
    return np.stack(points, axis=0).astype(np.float64)


def transform_points(T: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    return xyz @ T[:3, :3].T + T[:3, 3]


def fit_rigid(src: np.ndarray, dst: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if weights is None:
        weights = np.ones(src.shape[0], dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / np.maximum(weights.sum(), 1e-12)
    src_mu = (src * weights[:, None]).sum(axis=0)
    dst_mu = (dst * weights[:, None]).sum(axis=0)
    H = ((src - src_mu) * weights[:, None]).T @ (dst - dst_mu)
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
    scale = float(np.sum(dst_c * (src_c @ R.T)) / max(np.sum(src_c * src_c), 1e-12))
    t = dst_mu - scale * (R @ src_mu)
    return scale, R, t


def apply_similarity(src: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return scale * (src @ R.T) + t


def backproject(depth: np.ndarray, K: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    z = depth[ys, xs].astype(np.float64)
    x = (xs.astype(np.float64) - K[0, 2]) / K[0, 0] * z
    y = (ys.astype(np.float64) - K[1, 2]) / K[1, 1] * z
    return np.stack([x, y, z], axis=1)


def sample_scene_points(
    args: argparse.Namespace,
    spec: dict,
    raw_dir: Path,
    raw_K: np.ndarray,
    helper: AvatarReXHelper,
    scene_regions: list[str],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray]:
    seq = spec["seq"]
    frame = int(spec["frame"])
    root = args.avatarrex_root / args.split / seq
    rgb = cv2.imread(str(root / "rgb" / f"{frame:08d}.png"), cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(root / "rgb" / f"{frame:08d}.png")
    h0, w0 = rgb.shape[:2]
    params = resize_crop_params(w0, h0, args.size)
    mask = resize_crop_image(read_mask(root / "mask" / f"{frame:08d}.png"), params, cv2.INTER_NEAREST).astype(bool)
    depth_gt_raw = np.load(root / "depth" / f"{frame:08d}.npy")
    depth_gt = depth_gt_raw.astype(np.float64)
    if np.issubdtype(depth_gt_raw.dtype, np.integer):
        depth_gt /= 1000.0
    depth_gt = resize_crop_image(depth_gt, params, cv2.INTER_NEAREST)
    gt_K = update_intrinsics(helper.intrinsics(seq), params)

    joints_gt = helper.smpl_joints(frame)
    joints_uv = resize_crop_uv(helper.project_world(seq, joints_gt), params)
    region_masks = make_token_aligned_regions(mask, joints_uv)

    selected = np.zeros(mask.shape, dtype=bool)
    region_counts = {}
    for name in scene_regions:
        region = region_masks[name]
        selected |= region
        region_counts[name] = int(region.sum())

    raw_depth = np.load(raw_dir / "depth" / f"{int(spec['idx']):06d}.npy").astype(np.float64)
    valid = (
        selected
        & np.isfinite(raw_depth)
        & np.isfinite(depth_gt)
        & (raw_depth > 0.05)
        & (raw_depth < 20.0)
        & (depth_gt > 0.05)
        & (depth_gt < 20.0)
    )
    ys, xs = np.where(valid)
    if len(xs) > args.scene_samples_per_frame:
        keep = rng.choice(len(xs), size=args.scene_samples_per_frame, replace=False)
        ys, xs = ys[keep], xs[keep]
    src_cam = backproject(raw_depth, raw_K, ys, xs)
    gt_cam = backproject(depth_gt, gt_K, ys, xs)
    gt_world = transform_points(helper.c2w(seq), gt_cam)

    debug = cv2.cvtColor(resize_crop_image(rgb, params, cv2.INTER_LANCZOS4), cv2.COLOR_BGR2RGB)
    overlay = debug.copy()
    overlay[selected] = np.array([255, 80, 20], dtype=np.uint8)
    if len(xs) > 0:
        overlay[ys, xs] = np.array([20, 255, 255], dtype=np.uint8)
    debug_img = cv2.addWeighted(debug, 0.55, overlay, 0.45, 0)
    return src_cam, gt_world, {"valid_scene_points": int(len(xs)), **region_counts}, debug_img


def load_human3r_smpl_joints(raw_dir: Path, intrinsics: list[np.ndarray], device: str) -> list[np.ndarray]:
    smpl_paths = sorted((raw_dir / "smpl").glob("*.npz"))
    beta_dim = next((np.load(p, allow_pickle=True)["shape"].shape[-1] for p in smpl_paths), 10)
    layer = SMPL_Layer(type="smplx", gender="neutral", num_betas=int(beta_dim), kid=False, person_center="head").to(device)
    out_joints = []
    with torch.no_grad():
        for i, path in enumerate(smpl_paths):
            data = np.load(path, allow_pickle=True)
            rotvec = data["rotvec"][:1].astype(np.float32)
            shape = data["shape"][:1].astype(np.float32)
            transl = data["transl"][:1].astype(np.float32)
            expr = data["expression"]
            expr_t = None if expr is None else torch.from_numpy(expr[:1].astype(np.float32)).to(device)
            out = layer(
                torch.from_numpy(rotvec).to(device),
                torch.from_numpy(shape).to(device),
                torch.from_numpy(transl).to(device),
                None,
                None,
                K=torch.from_numpy(intrinsics[i].astype(np.float32)).to(device).unsqueeze(0),
                expression=expr_t,
            )
            out_joints.append(select_anchor_points(out["smpl_j3d"][0].detach().cpu().numpy().astype(np.float64)))
    return out_joints


def rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(a * a, axis=-1)))) if len(a) else float("nan")


def clipped_scene_translation(
    current_scene: np.ndarray,
    target_scene: np.ndarray,
    alpha: float,
    clip: float,
) -> np.ndarray:
    if len(current_scene) == 0:
        return np.zeros(3, dtype=np.float64)
    residual = target_scene - current_scene
    delta = np.median(residual, axis=0) * float(alpha)
    norm = float(np.linalg.norm(delta))
    if norm > clip > 0:
        delta = delta * (clip / norm)
    return delta.astype(np.float64)


def main() -> None:
    args = parse_args()
    raw_dir = args.compare_root / "raw"
    out_dir = args.compare_root / args.output_name
    scene_regions = [x.strip() for x in args.scene_regions.split(",") if x.strip()]
    for name in scene_regions:
        if name not in {"near_foot", "near_human"}:
            raise ValueError(f"Unknown scene region: {name}")

    copy_saved_output_tree(raw_dir, out_dir)
    debug_dir = out_dir / "scene_region_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    specs = read_case_manifest(args.compare_root)
    raw_poses, intrinsics = load_camera_poses(raw_dir)
    helper = AvatarReXHelper(args.avatarrex_raw_root)
    pred_human_cam = load_human3r_smpl_joints(raw_dir, intrinsics, args.device)
    gt_human_world = [select_anchor_points(helper.smpl_joints(int(spec["frame"]))) for spec in specs]

    raw_human_world0 = transform_points(raw_poses[0], pred_human_cam[0])
    scale, R_align, t_align = fit_similarity(gt_human_world[0], raw_human_world0)
    target_human_world = [apply_similarity(j, scale, R_align, t_align) for j in gt_human_world]

    rng = np.random.default_rng(42)
    corrected_poses = []
    per_frame = []
    for i, spec in enumerate(specs):
        scene_src, scene_gt_world, scene_info, debug_img = sample_scene_points(
            args, spec, raw_dir, intrinsics[i], helper, scene_regions, rng
        )
        cv2.imwrite(str(debug_dir / f"{i:06d}_{'_'.join(scene_regions)}.png"), cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
        scene_target = apply_similarity(scene_gt_world, scale, R_align, t_align)

        human_pose = fit_rigid(pred_human_cam[i], target_human_world[i])
        if args.mode == "human_then_scene_translation":
            T = human_pose
            if len(scene_src) >= 8 and args.scene_total_weight > 0:
                delta_t = clipped_scene_translation(
                    transform_points(T, scene_src),
                    scene_target,
                    alpha=args.scene_translation_alpha,
                    clip=args.scene_translation_clip,
                )
                T = T.copy()
                T[:3, 3] += delta_t
        else:
            src = [pred_human_cam[i]]
            dst = [target_human_world[i]]
            human_w = np.full(len(pred_human_cam[i]), args.human_total_weight / max(len(pred_human_cam[i]), 1), dtype=np.float64)
            weights = [human_w]
            if len(scene_src) >= 8 and args.scene_total_weight > 0:
                src.append(scene_src)
                dst.append(scene_target)
                weights.append(np.full(len(scene_src), args.scene_total_weight / len(scene_src), dtype=np.float64))

            src_all = np.concatenate(src, axis=0)
            dst_all = np.concatenate(dst, axis=0)
            weights_all = np.concatenate(weights, axis=0)
            T = fit_rigid(src_all, dst_all, weights_all)
        if i == 0:
            T = raw_poses[0].copy()
        corrected_poses.append(T)

        raw_human_err = rms(transform_points(raw_poses[i], pred_human_cam[i]) - target_human_world[i])
        corr_human_err = rms(transform_points(T, pred_human_cam[i]) - target_human_world[i])
        raw_scene_err = rms(transform_points(raw_poses[i], scene_src) - scene_target) if len(scene_src) else float("nan")
        corr_scene_err = rms(transform_points(T, scene_src) - scene_target) if len(scene_src) else float("nan")
        per_frame.append(
            {
                "idx": i,
                "seq": spec["seq"],
                "frame": int(spec["frame"]),
                "raw_human_anchor_rmse": raw_human_err,
                "corrected_human_anchor_rmse": corr_human_err,
                "raw_scene_rmse": raw_scene_err,
                "corrected_scene_rmse": corr_scene_err,
                "raw_camera_center": raw_poses[i][:3, 3].tolist(),
                "corrected_camera_center": T[:3, 3].tolist(),
                **scene_info,
            }
        )

    write_camera_poses(out_dir, corrected_poses, intrinsics)
    summary = {
        "correction_type": "token_aligned_human_local_scene_explicit_fit",
        "used_information": [
            "Human3R predicted SMPL camera-space anchors for pelvis/torso/left_foot/right_foot",
            "Human3R predicted camera-space depth points only inside token-validated local scene regions",
            "AvatarReX GT SMPL anchors aligned to raw frame-0 Human3R world",
            "AvatarReX GT depth in the same local scene pixels, transformed into the same aligned world",
        ],
        "not_used_information": [
            "Direct GT camera pose replacement",
            "Full-image background",
            "Human mask interior for scene points",
            "Unvalidated body parts",
        ],
        "human_anchor_names": [name for name, _ in TOKEN_BODY_ANCHOR_SPECS],
        "scene_regions": scene_regions,
        "weights": {
            "human_total_weight": args.human_total_weight,
            "scene_total_weight": args.scene_total_weight,
        },
        "mode": args.mode,
        "scene_translation_alpha": args.scene_translation_alpha,
        "scene_translation_clip": args.scene_translation_clip,
        "target_alignment": {
            "type": "similarity_gt_human_frame0_to_raw_human_frame0",
            "scale": float(scale),
            "rotation": R_align.tolist(),
            "translation": t_align.tolist(),
        },
        "per_frame": per_frame,
        "debug_dir": str(debug_dir),
    }
    with open(out_dir / "token_aligned_human_scene_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps({"output_dir": str(out_dir), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
