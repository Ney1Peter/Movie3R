#!/usr/bin/env python3
"""Floor-locked hybrid human/scene alignment for saved Human3R outputs.

The input directory is expected to be floor-leveled already.  This script keeps
the floor normal fixed and only optimizes yaw around that normal plus 3D
translation.  Human joints provide the actor anchor; background chamfer provides
an optional scene anchor whose weight is gated by a constrained scene-only probe.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from align_saved_output_floor_human import (
    build_joint_selection,
    compute_normal_translation,
    joint_metrics,
    plane_center,
    project_to_plane,
    solve_yaw_and_plane_translation,
    transform_overlay,
    transform_pose,
)
from dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates
from overfit_human_anchor_pose_correction import load_sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True, help="Floor-parallel saved-output directory.")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--normal_debug_json", type=Path, required=True)
    parser.add_argument("--reference_viewer_frame", type=int, default=0)
    parser.add_argument("--align_viewer_frames", type=int, nargs="*", default=None)
    parser.add_argument("--stable_weight", type=float, default=1.0)
    parser.add_argument("--foot_weight", type=float, default=2.0)
    parser.add_argument("--human_weight", type=float, default=1.0)
    parser.add_argument("--scene_weight", type=float, default=2.0)
    # **========== 原始代码 ==========**
    # parser.add_argument("--lr", type=float, default=5e-2)
    # **========== 新代码 ==========**
    # --lr is declared above next to the loss weights to avoid duplicate argparse entries.
    # **========== 结束 ==========**
    parser.add_argument("--scene_weight_override", type=float, default=None, help="Use this gate instead of the auto scene reliability gate.")
    parser.add_argument("--probe_steps", type=int, default=350)
    parser.add_argument("--steps", type=int, default=650)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--max_yaw_deg", type=float, default=120.0)
    parser.add_argument("--max_t_plane", type=float, default=6.0)
    parser.add_argument("--max_t_normal", type=float, default=2.0)
    parser.add_argument("--scene_chamfer_cap", type=float, default=0.8)
    parser.add_argument("--scene_reliability_scale", type=float, default=0.35)
    parser.add_argument("--min_scene_gate", type=float, default=0.0)
    parser.add_argument("--max_scene_gate", type=float, default=1.0)
    parser.add_argument("--conf_threshold", type=float, default=1.5)
    parser.add_argument("--mask_threshold", type=float, default=0.1)
    parser.add_argument("--bg_sample_points", type=int, default=2200)
    parser.add_argument("--seed", type=int, default=10601)
    parser.add_argument("--line_length", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_np(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def infer_num_frames(input_dir: Path) -> int:
    files = sorted((input_dir / "camera").glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No camera npz files under {input_dir / 'camera'}")
    return len(files)


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    for subdir in ["camera", "camera_raw", "color", "conf", "depth", "smpl"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)


def link_or_symlink(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        rel_src = os.path.relpath(src, dst.parent)
        os.symlink(rel_src, dst)


def copy_payload(input_dir: Path, output_dir: Path, frame_id: int) -> None:
    for subdir, ext in [("camera_raw", ".npz"), ("color", ".png"), ("conf", ".npy"), ("depth", ".npy"), ("smpl", ".npz")]:
        src = input_dir / subdir / f"{frame_id:06d}{ext}"
        if not src.is_file():
            if subdir == "camera_raw":
                continue
            raise FileNotFoundError(src)
        link_or_symlink(src, output_dir / subdir / f"{frame_id:06d}{ext}")


def load_background_points(output_dir: Path, frame: int, conf_threshold: float, mask_threshold: float, sample_points: int, seed: int) -> np.ndarray:
    cam = np.load(output_dir / "camera" / f"{frame:06d}.npz")
    pose = cam["pose"].astype(np.float32)
    intrinsics = cam["intrinsics"].astype(np.float32)
    depth = np.load(output_dir / "depth" / f"{frame:06d}.npy").astype(np.float32)
    conf = np.load(output_dir / "conf" / f"{frame:06d}.npy").astype(np.float32)
    points_world, _ = depthmap_to_absolute_camera_coordinates(depth, intrinsics, pose)
    valid = np.isfinite(points_world).all(axis=-1) & np.isfinite(depth) & (depth > 0.0) & np.isfinite(conf) & (conf >= float(conf_threshold))

    smpl_path = output_dir / "smpl" / f"{frame:06d}.npz"
    if smpl_path.is_file():
        smpl = np.load(smpl_path, allow_pickle=True)
        if "msk" in smpl.files:
            msk = smpl["msk"]
            if msk is not None and np.size(msk) > 0:
                # **========== 原始代码 ==========**
                # human_mask = np.max(mskul := mku := msk.astype(np.float32), axis=0) > float(mask_threshold)
                # valid &= ~human_mask
                # del mskul, mku

                # **========== 新代码 ==========**
                human_mask = np.max(msk.astype(np.float32), axis=0) > float(mask_threshold)
                valid &= ~human_mask
                # **========== 结束 ==========**

    points = points_world[valid].astype(np.float32)
    if points.shape[0] < 16:
        raise ValueError(f"Too few background points for frame {frame}: {points.shape[0]}")
    if points.shape[0] > int(sample_points):
        rng = np.random.default_rng(seed)
        points = points[rng.choice(points.shape[0], size=int(sample_points), replace=False)]
    return points.astype(np.float32)


def make_plane_basis(normal: np.ndarray) -> np.ndarray:
    n = normalize_np(normal)
    seed = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(seed, n))) > 0.9:
        seed = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    u = normalize_np(seed - n * float(np.dot(seed, n)))
    v = normalize_np(np.cross(n, u))
    return np.stack([u, v, n], axis=0)


def inv_tanh_clamped(value: float, max_value: float) -> float:
    if max_value <= 0.0:
        return 0.0
    x = float(np.clip(value / max_value, -0.999, 0.999))
    return 0.5 * math.log((1.0 + x) / (1.0 - x))


def rotation_about_axis(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    axis = axis / axis.norm().clamp_min(1e-8)
    x, y, z = axis.unbind(dim=0)
    zero = torch.zeros((), device=axis.device, dtype=axis.dtype)
    K = torch.stack(
        [zero, -z, y, z, zero, -x, -y, x, zero], dim=0
    ).reshape(3, 3)
    eye = torch.eye(3, device=axis.device, dtype=axis.dtype)
    return eye + torch.sin(angle) * K + (1.0 - torch.cos(angle)) * (K @ K)


def constrained_transform(
    raw_yaw: torch.Tensor,
    raw_coeffs: torch.Tensor,
    normal: torch.Tensor,
    basis: torch.Tensor,
    max_yaw: float,
    max_t_plane: float,
    max_t_normal: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    yaw = float(max_yaw) * torch.tanh(raw_yaw)
    coeff_0 = float(max_t_plane) * torch.tanh(raw_coeffs[0])
    coeff_1 = float(max_t_plane) * torch.tanh(raw_coeffs[1])
    coeff_2 = float(max_t_normal) * torch.tanh(raw_coeffs[2])
    coeffs = torch.stack([coeff_0, coeff_1, coeff_2])
    R = rotation_about_axis(normal, yaw)
    t = coeffs @ basis
    return R, t, yaw, coeffs


def transform_points_torch(points: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return points @ R.transpose(0, 1) + t[None]


def chamfer_metric(points: torch.Tensor, ref_points: torch.Tensor) -> torch.Tensor:
    dist = torch.cdist(points[None], ref_points[None], p=2)[0]
    return 0.5 * (dist.min(dim=1).values.mean() + dist.min(dim=0).values.mean())


def robust_chamfer(points: torch.Tensor, ref_points: torch.Tensor, cap: float) -> torch.Tensor:
    dist = torch.cdist(points[None], ref_points[None], p=2)[0]
    a = dist.min(dim=1).values.clamp_max(float(cap)).mean()
    b = dist.min(dim=0).values.clamp_max(float(cap)).mean()
    return 0.5 * (a + b)


def human_loss(ref_joints: torch.Tensor, cur_joints: torch.Tensor, joint_ids: np.ndarray, weights: torch.Tensor) -> torch.Tensor:
    diff = cur_joints[joint_ids] - ref_joints[joint_ids]
    return (torch.linalg.norm(diff, dim=-1) * weights).sum()


def init_raw_params(yaw: float, translation: np.ndarray, basis_np: np.ndarray, max_yaw: float, max_t_plane: float, max_t_normal: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    coeffs = basis_np @ translation.astype(np.float64)
    raw_yaw = torch.tensor(inv_tanh_clamped(float(yaw), max_yaw), device=device, dtype=torch.float32)
    raw_coeffs = torch.tensor(
        [
            inv_tanh_clamped(float(coeffs[0]), max_t_plane),
            inv_tanh_clamped(float(coeffs[1]), max_t_plane),
            inv_tanh_clamped(float(coeffs[2]), max_t_normal),
        ],
        device=device,
        dtype=torch.float32,
    )
    return raw_yaw, raw_coeffs


def optimize_scene_probe(
    bg_ref: torch.Tensor,
    bg_cur: torch.Tensor,
    normal: torch.Tensor,
    basis: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    raw_yaw = torch.nn.Parameter(torch.zeros((), device=device))
    raw_coeffs = torch.nn.Parameter(torch.zeros(3, device=device))
    optimizer = torch.optim.AdamW([raw_yaw, raw_coeffs], lr=float(args.lr), weight_decay=1e-4)
    max_yaw = math.radians(float(args.max_yaw_deg))
    initial = float(chamfer_metric(bg_cur, bg_ref).detach().cpu())
    for _ in range(int(args.probe_steps)):
        optimizer.zero_grad(set_to_none=True)
        R, t, _, _ = constrained_transform(raw_yaw, raw_coeffs, normal, basis, max_yaw, args.max_t_plane, args.max_t_normal)
        scene_loss = robust_chamfer(transform_points_torch(bg_cur, R, t), bg_ref, args.scene_chamfer_cap)
        prior = 0.002 * (raw_yaw.pow(2) + raw_coeffs.pow(2).mean())
        (scene_loss + prior).backward()
        optimizer.step()
    with torch.no_grad():
        R, t, yaw, coeffs = constrained_transform(raw_yaw, raw_coeffs, normal, basis, max_yaw, args.max_t_plane, args.max_t_normal)
        final = float(chamfer_metric(transform_points_torch(bg_cur, R, t), bg_ref).cpu())
    return {
        "initial_chamfer": initial,
        "final_chamfer": final,
        "improvement": initial - final,
        "yaw": float(yaw.detach().cpu()),
        "coeffs": coeffs.detach().cpu().numpy().astype(np.float32).tolist(),
        "translation": t.detach().cpu().numpy().astype(np.float32).tolist(),
    }


def optimize_hybrid_frame(
    frame: int,
    ref_joints_np: np.ndarray,
    cur_joints_np: np.ndarray,
    bg_ref_np: np.ndarray,
    bg_cur_np: np.ndarray,
    joint_ids: np.ndarray,
    joint_weights_np: np.ndarray,
    normal_np: np.ndarray,
    planes: dict[int, dict],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict]:
    basis_np = make_plane_basis(normal_np)
    normal = torch.from_numpy(normal_np.astype(np.float32)).to(device)
    basis = torch.from_numpy(basis_np.astype(np.float32)).to(device)
    bg_ref = torch.from_numpy(bg_ref_np).to(device=device, dtype=torch.float32)
    bg_cur = torch.from_numpy(bg_cur_np).to(device=device, dtype=torch.float32)
    ref_joints = torch.from_numpy(ref_joints_np.astype(np.float32)).to(device)
    cur_joints = torch.from_numpy(cur_joints_np.astype(np.float32)).to(device)
    joint_weights = torch.from_numpy(joint_weights_np.astype(np.float32)).to(device)
    max_yaw = math.radians(float(args.max_yaw_deg))

    R_human, t_plane, yaw_human = solve_yaw_and_plane_translation(ref_joints_np[joint_ids], cur_joints_np[joint_ids], normal_np, joint_weights_np)
    t_normal = compute_normal_translation(
        "human_centroid",
        ref_joints_np[joint_ids],
        cur_joints_np[joint_ids],
        joint_weights_np,
        normal_np,
        R_human,
        plane_center(planes[int(args.reference_viewer_frame)]),
        plane_center(planes[frame]),
    )
    human_init_t = t_plane + t_normal

    scene_probe = optimize_scene_probe(bg_ref, bg_cur, normal, basis, args, device)
    ref_ratio = float(planes[int(args.reference_viewer_frame)].get("inlier_ratio", 0.0))
    cur_ratio = float(planes[frame].get("inlier_ratio", 0.0))
    floor_gate = float(np.clip(min(ref_ratio, cur_ratio), 0.0, 1.0))
    residual_gate = math.exp(-scene_probe["final_chamfer"] / max(float(args.scene_reliability_scale), 1e-6))
    improve_gate = float(np.clip(scene_probe["improvement"] / max(scene_probe["initial_chamfer"], 1e-6), 0.0, 1.0))
    auto_gate = floor_gate * residual_gate * (0.35 + 0.65 * improve_gate)
    scene_gate = float(args.scene_weight_override) if args.scene_weight_override is not None else auto_gate
    scene_gate = float(np.clip(scene_gate, float(args.min_scene_gate), float(args.max_scene_gate)))

    raw_yaw_value, raw_coeff_values = init_raw_params(yaw_human, human_init_t, basis_np, max_yaw, args.max_t_plane, args.max_t_normal, device)
    raw_yaw = torch.nn.Parameter(raw_yaw_value)
    raw_coeffs = torch.nn.Parameter(raw_coeff_values)
    optimizer = torch.optim.AdamW([raw_yaw, raw_coeffs], lr=float(args.lr), weight_decay=1e-4)

    history = []
    for step in range(int(args.steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        R, t, yaw, coeffs = constrained_transform(raw_yaw, raw_coeffs, normal, basis, max_yaw, args.max_t_plane, args.max_t_normal)
        joints_corr = transform_points_torch(cur_joints, R, t)
        bg_corr = transform_points_torch(bg_cur, R, t)
        h_loss = human_loss(ref_joints, joints_corr, joint_ids, joint_weights)
        s_loss = robust_chamfer(bg_corr, bg_ref, args.scene_chamfer_cap)
        prior = 0.002 * (raw_yaw.pow(2) + raw_coeffs.pow(2).mean())
        loss = float(args.human_weight) * h_loss + float(args.scene_weight) * scene_gate * s_loss + prior
        loss.backward()
        optimizer.step()
        if step in {0, int(args.steps)}:
            history.append(
                {
                    "step": int(step),
                    "loss": float(loss.detach().cpu()),
                    "human_loss": float(h_loss.detach().cpu()),
                    "scene_loss": float(s_loss.detach().cpu()),
                    "scene_metric": float(chamfer_metric(bg_corr.detach(), bg_ref).detach().cpu()),
                    "yaw_deg": math.degrees(float(yaw.detach().cpu())),
                    "coeffs": coeffs.detach().cpu().numpy().astype(np.float32).tolist(),
                }
            )

    with torch.no_grad():
        R, t, yaw, coeffs = constrained_transform(raw_yaw, raw_coeffs, normal, basis, max_yaw, args.max_t_plane, args.max_t_normal)
        joints_corr = transform_points_torch(cur_joints, R, t)
        bg_corr = transform_points_torch(bg_cur, R, t)
        before_human = joint_metrics(ref_joints_np, cur_joints_np, normal_np, joint_ids, joint_weights_np)
        human_init_joints = cur_joints_np @ R_human.T + human_init_t[None]
        human_init_metrics = joint_metrics(ref_joints_np, human_init_joints, normal_np, joint_ids, joint_weights_np)
        after_human = joint_metrics(ref_joints_np, joints_corr.cpu().numpy(), normal_np, joint_ids, joint_weights_np)
        R_np = R.cpu().numpy().astype(np.float64)
        t_np = t.cpu().numpy().astype(np.float64)
        debug = {
            "viewer_frame": int(frame),
            "raw_frame": int(planes[frame].get("raw_frame", frame)),
            "scene_gate": scene_gate,
            "auto_scene_gate": auto_gate,
            "floor_gate": floor_gate,
            "scene_probe": scene_probe,
            "human_init": {
                "yaw_deg": math.degrees(float(yaw_human)),
                "translation": human_init_t.astype(np.float32).tolist(),
                "metrics": human_init_metrics,
                "scene_metric": float(chamfer_metric(torch.from_numpy((bg_cur_np @ R_human.T + human_init_t[None]).astype(np.float32)).to(device), bg_ref).cpu()),
            },
            "final": {
                "yaw_deg": math.degrees(float(yaw.cpu())),
                "translation": t_np.astype(np.float32).tolist(),
                "translation_norm": float(np.linalg.norm(t_np)),
                "translation_normal_component": float(np.dot(t_np, normal_np)),
                "floor_normal_after_dot": float(np.dot(R_np @ normal_np, normal_np)),
                "scene_metric": float(chamfer_metric(bg_corr, bg_ref).cpu()),
                "human_metrics": after_human,
            },
            "before": {
                "human_metrics": before_human,
                "scene_metric": float(chamfer_metric(bg_cur, bg_ref).cpu()),
            },
            "history": history,
        }
    return R_np, t_np, debug


def main() -> None:
    args = parse_args()
    torch.manual_seed(107)
    np.random.seed(107)
    device = torch.device(args.device)
    num_frames = infer_num_frames(args.input_dir)
    debug_json = json.loads(args.normal_debug_json.read_text())
    planes = {int(p["viewer_frame"]): p for p in debug_json.get("planes", [])}
    ref_frame = int(args.reference_viewer_frame)
    if ref_frame not in planes:
        raise KeyError(f"Reference viewer frame {ref_frame} missing from normal debug JSON")
    normal_np = normalize_np(np.asarray(planes[ref_frame]["normal"], dtype=np.float64)).astype(np.float32)
    # **========== 原始代码 ==========**
    # align_frames = args.align_viewu_frames if False else args.align_viewer_frames

    # **========== 新代码 ==========**
    align_frames = args.align_viewer_frames
    # **========== 结束 ==========**
    if align_frames is None:
        align_frames = sorted(frame for frame in planes if frame != ref_frame)
    align_frames = [int(x) for x in align_frames]

    data = load_sequence(args.input_dir, num_frames, device)
    joint_ids, joint_weights = build_joint_selection(args.stable_weight, args.foot_weight)
    ref_joints = data.joints_world[ref_frame].astype(np.float64)
    bg_ref = load_background_points(args.input_dir, ref_frame, args.conf_threshold, args.mask_threshold, args.bg_sample_points, args.seed + ref_frame)

    corrected_poses = data.poses.copy()
    transforms: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    records = []
    for frame in align_frames:
        if frame not in planes:
            raise KeyError(f"Viewer frame {frame} missing from normal debug JSON")
        cur_joints = data.joints_world[frame].astype(np.float64)
        bg_cur = load_background_points(args.input_dir, frame, args.conf_threshold, args.mask_threshold, args.bg_sample_points, args.seed + frame)
        R, t, record = optimize_hybrid_frame(frame, ref_joints, cur_joints, bg_ref, bg_cur, joint_ids, joint_weights, normal_np, planes, args, device)
        corrected_poses[frame] = transform_pose(data.poses[frame], R, t)
        transforms[frame] = (R, t)
        records.append(record)
        print(json.dumps(record["final"], sort_keys=True), flush=True)

    prepare_output_dir(args.output_dir, args.overwrite)
    for frame in range(num_frames):
        cam = np.load(args.input_dir / "camera" / f"{frame:06d}.npz")
        np.savez(args.output_dir / "camera" / f"{frame:06d}.npz", pose=corrected_poses[frame].astype(np.float32), intrinsics=cam["intrinsics"].astype(np.float32))
        copy_payload(args.input_dir, args.output_dir, frame)

    overlay = transform_overlay(debug_json, transforms, args.line_length)
    overlay.update(
        {
            "description": "Floor-locked hybrid human/scene alignment overlay.",
            "input_dir": str(args.input_dir),
            "output_dir": str(args.output_dir),
            "source_normal_debug_json": str(args.normal_debug_json),
            # **========== 原始代码 ==========**
            # "reference_viewu_frame": ref_frame if False else ref_frame,

            # **========== 新代码 ==========**
            "reference_viewer_frame": ref_frame,
            # **========== 结束 ==========**
            "aligned_viewer_frames": align_frames,
        }
    )
    overlay_path = args.output_dir / "floor_human_scene_alignment_debug.json"
    overlay_path.write_text(json.dumps(overlay, indent=2, sort_keys=True), encoding="utf-8")

    metrics = {
        "teacher_type": "floor_locked_human_scene_hybrid_debug",
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "normal_debug_json": str(args.normal_debug_json),
        "reference_viewer_frame": ref_frame,
        "reference_normal": normal_np.astype(np.float32).tolist(),
        "aligned_viewer_frames": align_frames,
        "human_weight": float(args.human_weight),
        "scene_weight": float(args.scene_weight),
        "scene_weight_override": None if args.scene_weight_override is None else float(args.scene_weight_override),
        "records": records,
        "overlay_json": str(overlay_path),
    }
    metrics_path = args.output_dir / "floor_human_scene_alignment_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
