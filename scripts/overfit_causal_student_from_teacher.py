#!/usr/bin/env python3
"""Overfit a causal student to imitate an offline local-gauge teacher.

The teacher may have used future frames to produce pseudo targets, but this
student is only given causal features from frame t and t-1. This is a smoke test
for whether the teacher correction is predictable from online cues.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from overfit_human_anchor_pose_correction import (
    FOOT_JOINTS,
    STABLE_JOINTS,
    apply_correction,
    compute_metrics,
    infer_num_frames,
    load_sequence,
)
from overfit_single_boundary_frame_human_anchor import write_outputs_with_links
from overfit_single_boundary_frame_scene_geometry import estimate_top_planes, match_planes, sample_points
from overfit_single_boundary_frame_scene_normal import load_background_points
from overfit_transient_gate_human_correction import compute_transition_metrics


class CausalStudentMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, max_delta_t: float, max_delta_r_deg: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 6),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.max_delta_t = float(max_delta_t)
        self.max_delta_r = math.radians(float(max_delta_r_deg))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.net(x)
        delta_t = self.max_delta_t * torch.tanh(raw[:, :3])
        delta_r = self.max_delta_r * torch.tanh(raw[:, 3:])
        return delta_t, delta_r


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--teacher_metrics", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--source_video", type=Path, default=None)
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--boundary", type=int, required=True)
    parser.add_argument("--train_start", type=int, default=None)
    parser.add_argument("--train_end", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=192)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--max_delta_t", type=float, default=5.0)
    parser.add_argument("--max_delta_r_deg", type=float, default=90.0)
    parser.add_argument("--target_weight", type=float, default=4.0)
    parser.add_argument("--noop_weight", type=float, default=0.4)
    parser.add_argument("--smooth_weight", type=float, default=0.1)
    parser.add_argument("--rot_loss_weight", type=float, default=4.0)
    parser.add_argument("--conf_threshold", type=float, default=1.5)
    parser.add_argument("--mask_threshold", type=float, default=0.1)
    parser.add_argument("--plane_max_points", type=int, default=12000)
    parser.add_argument("--plane_iterations", type=int, default=768)
    parser.add_argument("--plane_threshold", type=float, default=0.04)
    parser.add_argument("--num_planes", type=int, default=3)
    parser.add_argument("--min_plane_inlier_ratio", type=float, default=0.03)
    parser.add_argument("--min_plane_dot", type=float, default=0.9)
    parser.add_argument("--bg_chamfer_points", type=int, default=800)
    parser.add_argument("--bg_chamfer_cap", type=float, default=0.5)
    parser.add_argument("--subset_output_dir", type=Path, default=None)
    parser.add_argument("--raw_subset_output_dir", type=Path, default=None)
    parser.add_argument("--teacher_subset_output_dir", type=Path, default=None)
    parser.add_argument("--subset_start", type=int, default=None)
    parser.add_argument("--subset_count", type=int, default=31)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log_every", type=int, default=500)
    return parser.parse_args()


def rotation_log_vector(R: np.ndarray) -> np.ndarray:
    cos = np.clip((float(np.trace(R)) - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cos))
    if theta < 1e-6:
        return np.zeros(3, dtype=np.float32)
    scale = theta / (2.0 * np.sin(theta))
    return np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ], dtype=np.float32) * np.float32(scale)


def normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def torso_frame_np(joints: np.ndarray) -> np.ndarray:
    left_hip, right_hip = joints[:, 1], joints[:, 2]
    left_shoulder, right_shoulder = joints[:, 16], joints[:, 17]
    hip_mid = 0.5 * (left_hip + right_hip)
    shoulder_mid = 0.5 * (left_shoulder + right_shoulder)
    up = normalize(shoulder_mid - hip_mid)
    right = normalize(right_shoulder - left_shoulder)
    forward = normalize(np.cross(right, up))
    return np.stack([right, up, forward], axis=1)


def chamfer_np(points: np.ndarray, ref_points: np.ndarray, cap: float | None = None) -> float:
    d = np.linalg.norm(points[:, None, :] - ref_points[None, :, :], axis=-1)
    a = d.min(axis=1)
    b = d.min(axis=0)
    if cap is not None:
        a = np.minimum(a, float(cap))
        b = np.minimum(b, float(cap))
    return float(0.5 * (a.mean() + b.mean()))


def build_scene(input_dir: Path, frame_id: int, pose: np.ndarray, intrinsics: np.ndarray, args: argparse.Namespace) -> dict | None:
    try:
        points = load_background_points(input_dir, frame_id, pose, intrinsics, args.conf_threshold, args.mask_threshold)
        plane_args = SimpleNamespace(
            plane_max_points=args.plane_max_points,
            plane_iterations=args.plane_iterations,
            plane_threshold=args.plane_threshold,
            num_planes=args.num_planes,
            min_plane_inlier_ratio=args.min_plane_inlier_ratio,
        )
        planes = estimate_top_planes(points, plane_args, seed=8101 + 13 * frame_id)
        bg = sample_points(points, args.bg_chamfer_points, seed=8201 + 13 * frame_id)
        return {"planes": planes, "bg": bg.astype(np.float32), "num_points": int(points.shape[0])}
    except Exception:
        return None


def scene_pair_features(prev_scene: dict | None, cur_scene: dict | None, args: argparse.Namespace) -> list[float]:
    if prev_scene is None or cur_scene is None:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    matches = match_planes(prev_scene["planes"], cur_scene["planes"], args.min_plane_dot)
    dots = [float(m["dot_abs"]) for m in matches]
    offsets = [abs(float(m["cur"]["d"]) - float(m["ref"]["d"])) for m in matches]
    return [
        float(len(matches)) / max(float(args.num_planes), 1.0),
        float(np.mean(dots)) if dots else 0.0,
        float(max(offsets)) if offsets else 0.0,
        float(sum(float(m["weight"]) for m in matches)),
        chamfer_np(cur_scene["bg"], prev_scene["bg"], cap=args.bg_chamfer_cap),
    ]


def load_teacher_targets(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    data = json.loads(path.read_text())
    targets = {}
    for record in data["frames"]:
        if record.get("skipped", False):
            continue
        frame = int(record["frame"])
        targets[frame] = (
            np.asarray(record["delta_t"], dtype=np.float32),
            np.asarray(record["delta_r"], dtype=np.float32),
        )
    return targets


def build_features(data, scenes: dict[int, dict | None], frame_ids: list[int], boundary: int, args: argparse.Namespace) -> np.ndarray:
    torso_cam = torso_frame_np(data.joints_cam)
    feats = []
    for frame_id in frame_ids:
        prev = max(0, frame_id - 1)
        prev2 = max(0, frame_id - 2)
        dt = frame_id - int(boundary)
        t_step = data.poses[frame_id, :3, 3] - data.poses[prev, :3, 3]
        t_prev_step = data.poses[prev, :3, 3] - data.poses[prev2, :3, 3]
        R_step = data.poses[frame_id, :3, :3] @ data.poses[prev, :3, :3].T
        R_prev_step = data.poses[prev, :3, :3] @ data.poses[prev2, :3, :3].T
        pelvis_cam_step = data.joints_cam[frame_id, 0] - data.joints_cam[prev, 0]
        stable_cam = chamfer_np(data.joints_cam[frame_id, STABLE_JOINTS], data.joints_cam[prev, STABLE_JOINTS])
        foot_cam = chamfer_np(data.joints_cam[frame_id, FOOT_JOINTS], data.joints_cam[prev, FOOT_JOINTS])
        torso_dots = np.sum(torso_cam[frame_id] * torso_cam[prev], axis=-1).clip(-1.0, 1.0)
        scene_feat = scene_pair_features(scenes.get(prev), scenes.get(frame_id), args)
        feats.append(
            np.concatenate(
                [
                    np.array([
                        float(dt),
                        float(max(dt, 0)),
                        float(np.exp(-max(dt, 0) / 2.0)),
                        1.0 if dt == 0 else 0.0,
                        1.0 if 0 <= dt <= 2 else 0.0,
                        1.0 if dt >= 3 else 0.0,
                    ], dtype=np.float32),
                    t_step.astype(np.float32),
                    t_prev_step.astype(np.float32),
                    np.array([np.linalg.norm(t_step), np.linalg.norm(t_prev_step), np.linalg.norm(t_step - t_prev_step)], dtype=np.float32),
                    rotation_log_vector(R_step),
                    rotation_log_vector(R_prev_step),
                    pelvis_cam_step.astype(np.float32),
                    np.array([np.linalg.norm(pelvis_cam_step), stable_cam, foot_cam], dtype=np.float32),
                    torso_dots.astype(np.float32),
                    np.asarray(scene_feat, dtype=np.float32),
                ]
            )
        )
    feats_np = np.stack(feats).astype(np.float32)
    mean = feats_np.mean(axis=0, keepdims=True)
    std = feats_np.std(axis=0, keepdims=True) + 1e-6
    return (feats_np - mean) / std


def apply_predicted_deltas(data, frame_ids: list[int], delta_t: np.ndarray, delta_r: np.ndarray) -> np.ndarray:
    corrected = data.poses.copy()
    device = torch.device("cpu")
    joints = torch.from_numpy(data.joints_world[frame_ids]).to(device=device, dtype=torch.float32)
    poses = torch.from_numpy(data.poses[frame_ids]).to(device=device, dtype=torch.float32)
    dt = torch.from_numpy(delta_t).to(device=device, dtype=torch.float32)
    dr = torch.from_numpy(delta_r).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        _, R_corr, t_corr = apply_correction(joints, poses, dt, dr)
    for local_i, frame_id in enumerate(frame_ids):
        corrected[frame_id, :3, :3] = R_corr[local_i].numpy().astype(np.float32)
        corrected[frame_id, :3, 3] = t_corr[local_i].numpy().astype(np.float32)
    return corrected


def main() -> None:
    args = parse_args()
    torch.manual_seed(61)
    np.random.seed(61)
    device = torch.device(args.device)
    num_frames = infer_num_frames(args.input_dir, args.source_video, args.num_frames)
    boundary = int(args.boundary)
    train_start = max(0, boundary - 1) if args.train_start is None else int(args.train_start)
    train_end = min(num_frames - 1, boundary + 29) if args.train_end is None else min(int(args.train_end), num_frames - 1)
    frame_ids = list(range(train_start, train_end + 1))
    data = load_sequence(args.input_dir, num_frames, device)
    teacher_targets = load_teacher_targets(args.teacher_metrics)

    scenes = {}
    for frame_id in frame_ids:
        scenes[frame_id] = build_scene(args.input_dir, frame_id, data.poses[frame_id], data.intrinsics[frame_id], args)
    features_np = build_features(data, scenes, frame_ids, boundary, args)
    target_t = np.zeros((len(frame_ids), 3), dtype=np.float32)
    target_r = np.zeros((len(frame_ids), 3), dtype=np.float32)
    sample_w = np.full((len(frame_ids),), float(args.noop_weight), dtype=np.float32)
    for i, frame_id in enumerate(frame_ids):
        if frame_id in teacher_targets:
            target_t[i], target_r[i] = teacher_targets[frame_id]
            sample_w[i] = float(args.target_weight)

    features = torch.from_numpy(features_np).to(device=device, dtype=torch.float32)
    target_t_torch = torch.from_numpy(target_t).to(device=device, dtype=torch.float32)
    target_r_torch = torch.from_numpy(target_r).to(device=device, dtype=torch.float32)
    sample_w_torch = torch.from_numpy(sample_w).to(device=device, dtype=torch.float32)

    model = CausalStudentMLP(features.shape[1], args.hidden_dim, args.max_delta_t, args.max_delta_r_deg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    for step in range(int(args.steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        pred_t, pred_r = model(features)
        per_frame = (pred_t - target_t_torch).pow(2).mean(dim=1) + float(args.rot_loss_weight) * (pred_r - target_r_torch).pow(2).mean(dim=1)
        fit_loss = (sample_w_torch * per_frame).mean()
        smooth_loss = (pred_t[1:] - pred_t[:-1]).pow(2).mean() + float(args.rot_loss_weight) * (pred_r[1:] - pred_r[:-1]).pow(2).mean()
        loss = fit_loss + float(args.smooth_weight) * smooth_loss
        loss.backward()
        optimizer.step()
        if step % int(args.log_every) == 0 or step == int(args.steps):
            with torch.no_grad():
                target_idx = [i for i, f in enumerate(frame_ids) if f in teacher_targets]
                target_err_t = (pred_t[target_idx] - target_t_torch[target_idx]).norm(dim=1).mean() if target_idx else torch.tensor(0.0, device=device)
                target_err_r = torch.rad2deg((pred_r[target_idx] - target_r_torch[target_idx]).norm(dim=1)).mean() if target_idx else torch.tensor(0.0, device=device)
                noop_idx = [i for i, f in enumerate(frame_ids) if f not in teacher_targets]
                noop_norm = pred_t[noop_idx].norm(dim=1).mean() if noop_idx else torch.tensor(0.0, device=device)
                record = {
                    "step": int(step),
                    "loss": float(loss.detach().cpu()),
                    "fit_loss": float(fit_loss.detach().cpu()),
                    "smooth_loss": float(smooth_loss.detach().cpu()),
                    "target_err_t": float(target_err_t.detach().cpu()),
                    "target_err_r_deg": float(target_err_r.detach().cpu()),
                    "noop_delta_t_norm": float(noop_norm.detach().cpu()),
                }
                print(json.dumps(record, sort_keys=True), flush=True)
                history.append(record)

    with torch.no_grad():
        pred_t, pred_r = model(features)
    pred_t_np = pred_t.detach().cpu().numpy().astype(np.float32)
    pred_r_np = pred_r.detach().cpu().numpy().astype(np.float32)
    corrected_poses = apply_predicted_deltas(data, frame_ids, pred_t_np, pred_r_np)

    raw_metrics = compute_metrics(data, data.poses, boundary)
    raw_transition = compute_transition_metrics(data, data.poses, boundary)
    student_metrics = compute_metrics(data, corrected_poses, boundary)
    student_transition = compute_transition_metrics(data, corrected_poses, boundary)
    teacher_metrics = json.loads(args.teacher_metrics.read_text())

    write_outputs_with_links(args.input_dir, args.output_dir, corrected_poses, data.intrinsics, list(range(num_frames)), args.overwrite)
    if args.subset_output_dir is not None:
        subset_start = max(0, boundary - 1) if args.subset_start is None else int(args.subset_start)
        subset_frames = list(range(subset_start, min(num_frames, subset_start + int(args.subset_count))))
        write_outputs_with_links(args.input_dir, args.subset_output_dir, corrected_poses, data.intrinsics, subset_frames, args.overwrite)
    if args.raw_subset_output_dir is not None:
        subset_start = max(0, boundary - 1) if args.subset_start is None else int(args.subset_start)
        subset_frames = list(range(subset_start, min(num_frames, subset_start + int(args.subset_count))))
        write_outputs_with_links(args.input_dir, args.raw_subset_output_dir, data.poses, data.intrinsics, subset_frames, args.overwrite)
    if args.teacher_subset_output_dir is not None:
        teacher_output_dir = Path(teacher_metrics["output_dir"])
        teacher_num_frames = infer_num_frames(teacher_output_dir, args.source_video, args.num_frames)
        teacher_data = load_sequence(teacher_output_dir, teacher_num_frames, device)
        subset_start = max(0, boundary - 1) if args.subset_start is None else int(args.subset_start)
        subset_frames = list(range(subset_start, min(teacher_num_frames, subset_start + int(args.subset_count))))
        write_outputs_with_links(teacher_output_dir, args.teacher_subset_output_dir, teacher_data.poses, teacher_data.intrinsics, subset_frames, args.overwrite)

    frame_records = []
    for i, frame_id in enumerate(frame_ids):
        frame_records.append(
            {
                "frame": int(frame_id),
                "target_delta_t": target_t[i].tolist(),
                "target_delta_r": target_r[i].tolist(),
                "pred_delta_t": pred_t_np[i].tolist(),
                "pred_delta_r": pred_r_np[i].tolist(),
                "is_teacher_target": bool(frame_id in teacher_targets),
            }
        )
    metrics = {
        "input_dir": str(args.input_dir),
        "teacher_metrics": str(args.teacher_metrics),
        "teacher_output_dir": teacher_metrics.get("output_dir"),
        "output_dir": str(args.output_dir),
        "boundary": int(boundary),
        "train_start": int(train_start),
        "train_end": int(train_end),
        "causal_input_only": True,
        "note": "Student targets come from offline teacher, but features use only frame t and t-1.",
        "history": history,
        "frames": frame_records,
        "raw": raw_metrics,
        "raw_transition": raw_transition,
        "teacher_transition": teacher_metrics.get("corrected_transition"),
        "student": student_metrics,
        "student_transition": student_transition,
    }
    with open(args.output_dir / "causal_student_from_teacher_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    print("raw_transition_metrics", json.dumps(raw_transition, indent=2, sort_keys=True))
    print("teacher_transition_metrics", json.dumps(teacher_metrics.get("corrected_transition"), indent=2, sort_keys=True))
    print("student_transition_metrics", json.dumps(student_transition, indent=2, sort_keys=True))
    print(f"Wrote causal student output to {args.output_dir}")


if __name__ == "__main__":
    main()
