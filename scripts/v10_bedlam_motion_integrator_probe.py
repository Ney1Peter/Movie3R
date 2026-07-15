#!/usr/bin/env python3
"""Strict causal V10 motion/global-state integrator probe on BEDLAM GT.

This script verifies the core V10 question without detector noise or Human3R
noise first:

Given a clean BEDLAM trajectory, split it into local shots, perturb later shots
with random SE(3) gauges, and test whether a causal module can attach each new
local shot back to the historical global state.

The forward path is intentionally streaming:

* the boundary is oracle for this probe;
* at a new segment, the model sees only historical global outputs and the
  current segment's first local frame;
* one segment-to-global transform is predicted and cached;
* later frames in the same segment reuse the cached transform;
* no future frame is used to build the feature for the boundary transform.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "config" / "manifests" / "bedlam_seq000021_good_6fps" / "metadata.json"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "v10_bedlam_motion_integrator_probe" / "gt_synthetic"


@dataclass
class Trajectory:
    frames: np.ndarray
    root_t: torch.Tensor  # [T, P, 3]
    root_R: torch.Tensor  # [T, P, 3, 3]
    cam_t: torch.Tensor  # [T, 3]
    cam_R: torch.Tensor  # [T, 3, 3]


@dataclass
class Episode:
    local_root_t: torch.Tensor
    local_root_R: torch.Tensor
    local_cam_t: torch.Tensor
    local_cam_R: torch.Tensor
    target_root_t: torch.Tensor
    target_root_R: torch.Tensor
    target_cam_t: torch.Tensor
    target_cam_R: torch.Tensor
    boundaries: list[int]
    segment_ends: list[int]


@dataclass
class BoundaryItem:
    feature_current: torch.Tensor
    feature_history: torch.Tensor
    feature_backward: torch.Tensor
    feature_reverse_current: torch.Tensor
    feature_reverse_prev: torch.Tensor
    feature_residual: torch.Tensor
    local_root_t: torch.Tensor
    local_root_R: torch.Tensor
    local_cam_t: torch.Tensor
    local_cam_R: torch.Tensor
    target_root_t: torch.Tensor
    target_root_R: torch.Tensor
    target_cam_t: torch.Tensor
    target_cam_R: torch.Tensor
    prev_root_t: torch.Tensor
    prev_root_R: torch.Tensor
    prev_cam_t: torch.Tensor
    prev_cam_R: torch.Tensor
    geo_R: torch.Tensor
    geo_t: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory_source",
        choices=["bedlam_gt", "human3r_saved"],
        default="bedlam_gt",
        help="bedlam_gt uses the filtered BEDLAM manifest; human3r_saved uses a demo.py --save directory as pseudo-global target.",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--human3r_output_dir", type=Path, default=None)
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument(
        "--max_people",
        type=int,
        default=None,
        help="Use the first N detected humans from a saved Human3R output. Default: minimum count over frames.",
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train_episodes", type=int, default=512)
    parser.add_argument("--val_episodes", type=int, default=128)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--segment_boundaries", type=int, nargs="+", default=[0, 10, 20])
    parser.add_argument("--max_rot_deg", type=float, default=160.0)
    parser.add_argument("--max_trans", type=float, default=4.0)
    parser.add_argument("--perturb_rot_deg", type=float, default=120.0)
    parser.add_argument("--perturb_trans", type=float, default=2.5)
    parser.add_argument(
        "--global_rot_deg",
        type=float,
        default=180.0,
        help="Random whole-episode world-gauge rotation. This prevents current-only models from memorizing one fixed BEDLAM world.",
    )
    parser.add_argument(
        "--global_trans",
        type=float,
        default=5.0,
        help="Random whole-episode world-gauge translation.",
    )
    parser.add_argument("--residual_max_rot_deg", type=float, default=25.0)
    parser.add_argument("--residual_max_trans", type=float, default=0.8)
    parser.add_argument("--reverse_max_rot_deg", type=float, default=30.0)
    parser.add_argument("--reverse_max_trans", type=float, default=2.0)
    parser.add_argument(
        "--cycle_refine_steps",
        type=int,
        default=4,
        help="Runtime gradient steps for Cycle-World style boundary transform refinement.",
    )
    parser.add_argument("--cycle_refine_lr", type=float, default=0.2)
    parser.add_argument("--cycle_feature_weight", type=float, default=10.0)
    parser.add_argument("--cycle_refine_prior_weight", type=float, default=0.05)
    parser.add_argument("--cycle_refine_smooth_weight", type=float, default=0.2)
    parser.add_argument(
        "--bidir_output_weight",
        type=float,
        default=0.25,
        help="Weight for matching the frozen backward teacher's aligned boundary output.",
    )
    parser.add_argument(
        "--bidir_transform_weight",
        type=float,
        default=0.05,
        help="Weight for matching the frozen backward teacher's boundary SE(3).",
    )
    parser.add_argument("--log_every", type=int, default=250)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def skew(v: torch.Tensor) -> torch.Tensor:
    x, y, z = v.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        [zero, -z, y, z, zero, -x, -y, x, zero],
        dim=-1,
    ).reshape(*v.shape[:-1], 3, 3)


def so3_exp(rotvec: torch.Tensor) -> torch.Tensor:
    theta = rotvec.norm(dim=-1, keepdim=True)
    axis = rotvec / theta.clamp_min(1e-8)
    K = skew(axis)
    eye = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype).expand(*rotvec.shape[:-1], 3, 3)
    sin_t = torch.sin(theta)[..., None]
    cos_t = torch.cos(theta)[..., None]
    R = eye + sin_t * K + (1.0 - cos_t) * (K @ K)
    R_small = eye + skew(rotvec)
    return torch.where((theta < 1e-6)[..., None], R_small, R)


def rotation_geodesic(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    rel = R_pred.transpose(-1, -2) @ R_gt
    cos = ((rel.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cos)


def rot6d(R: torch.Tensor) -> torch.Tensor:
    return R[..., :, :2].reshape(*R.shape[:-2], 6)


def solve_rigid_transform(src: torch.Tensor, dst: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return R, t with R @ src + t ~= dst."""
    src_mean = src.mean(dim=0, keepdim=True)
    dst_mean = dst.mean(dim=0, keepdim=True)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    H = src_c.transpose(0, 1) @ dst_c
    U, _, Vh = torch.linalg.svd(H)
    R = Vh.transpose(0, 1) @ U.transpose(0, 1)
    if torch.det(R) < 0:
        Vh = Vh.clone()
        Vh[-1] *= -1
        R = Vh.transpose(0, 1) @ U.transpose(0, 1)
    t = dst_mean.reshape(3) - R @ src_mean.reshape(3)
    return R, t


def apply_transform(
    root_t: torch.Tensor,
    root_R: torch.Tensor,
    cam_t: torch.Tensor,
    cam_R: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    out_root_t = torch.einsum("ij,...j->...i", R, root_t) + t
    out_root_R = torch.einsum("ij,...jk->...ik", R, root_R)
    out_cam_t = torch.einsum("ij,...j->...i", R, cam_t) + t
    out_cam_R = torch.einsum("ij,...jk->...ik", R, cam_R)
    return out_root_t, out_root_R, out_cam_t, out_cam_R


def apply_transform_batch(
    root_t: torch.Tensor,
    root_R: torch.Tensor,
    cam_t: torch.Tensor,
    cam_R: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    out_root_t = torch.einsum("bij,bpj->bpi", R, root_t) + t[:, None, :]
    out_root_R = torch.einsum("bij,bpjk->bpik", R, root_R)
    out_cam_t = torch.einsum("bij,bj->bi", R, cam_t) + t
    out_cam_R = torch.einsum("bij,bjk->bik", R, cam_R)
    return out_root_t, out_root_R, out_cam_t, out_cam_R


def load_bedlam_trajectory(manifest_path: Path) -> Trajectory:
    meta = json.loads(manifest_path.read_text(encoding="utf-8"))
    npz = np.load(meta["npz_path"], allow_pickle=True)
    frames = np.asarray(meta["kept_frames"], dtype=np.int64)
    root_t = []
    root_R = []
    cam_t = []
    cam_R = []
    for frame in frames:
        key = f"{int(frame):04d}"
        indices = meta["npz_indices_by_frame"][key]
        pose_world = torch.from_numpy(np.asarray(npz["pose_world"][indices, :3], dtype=np.float32))
        root_R.append(so3_exp(pose_world))
        root_t.append(torch.from_numpy(np.asarray(npz["trans_world"][indices], dtype=np.float32)))
        cam = torch.from_numpy(np.asarray(npz["cam_ext"][indices[0]], dtype=np.float32))
        cam_R.append(cam[:3, :3])
        cam_t.append(cam[:3, 3])
    return Trajectory(
        frames=frames,
        root_t=torch.stack(root_t, dim=0),
        root_R=torch.stack(root_R, dim=0),
        cam_t=torch.stack(cam_t, dim=0),
        cam_R=torch.stack(cam_R, dim=0),
    )


def load_human3r_saved_trajectory(output_dir: Path, num_frames: int | None, max_people: int | None) -> Trajectory:
    """Load a demo.py --save directory as a pseudo-global trajectory.

    This is for the second probe stage: run original Human3R continuously, then
    use its own saved output as the clean target trajectory.  The loader keeps
    the first N humans consistently across frames.  It does not use future
    frames during model inference; it only prepares the pseudo target.
    """
    camera_dir = output_dir / "camera"
    smpl_dir = output_dir / "smpl"
    if num_frames is None:
        num_frames = len(sorted(camera_dir.glob("*.npz")))
    if num_frames <= 0:
        raise ValueError(f"No saved frames found in {output_dir}")

    cams = []
    smpls = []
    counts = []
    for idx in range(num_frames):
        cam_path = camera_dir / f"{idx:06d}.npz"
        smpl_path = smpl_dir / f"{idx:06d}.npz"
        if not cam_path.is_file():
            raise FileNotFoundError(cam_path)
        if not smpl_path.is_file():
            raise FileNotFoundError(smpl_path)
        cam = np.load(cam_path)
        smpl = np.load(smpl_path, allow_pickle=True)
        pose = torch.from_numpy(np.asarray(cam["pose"], dtype=np.float32))
        transl = torch.from_numpy(np.asarray(smpl["transl"], dtype=np.float32))
        rotvec = torch.from_numpy(np.asarray(smpl["rotvec"], dtype=np.float32))
        if transl.ndim != 2 or transl.shape[0] == 0:
            raise ValueError(f"No detected human in {smpl_path}")
        cams.append(pose)
        smpls.append((rotvec, transl))
        counts.append(int(transl.shape[0]))

    people = min(counts)
    if max_people is not None:
        people = min(people, int(max_people))
    if people <= 0:
        raise ValueError(f"No common detected humans across {num_frames} frames in {output_dir}")

    root_t = []
    root_R = []
    cam_t = []
    cam_R = []
    for pose, (rotvec, transl) in zip(cams, smpls):
        R_cam = pose[:3, :3]
        t_cam = pose[:3, 3]
        cur_transl = transl[:people]
        if rotvec.ndim == 3:
            root_rotvec = rotvec[:people, 0]
        elif rotvec.ndim == 2:
            root_rotvec = rotvec[:people, :3]
        else:
            raise ValueError(f"Unsupported rotvec shape: {tuple(rotvec.shape)}")
        local_root_R = so3_exp(root_rotvec.float())
        root_t.append(torch.einsum("ij,pj->pi", R_cam, cur_transl.float()) + t_cam[None, :])
        root_R.append(torch.einsum("ij,pjk->pik", R_cam, local_root_R))
        cam_t.append(t_cam)
        cam_R.append(R_cam)

    return Trajectory(
        frames=np.arange(num_frames, dtype=np.int64),
        root_t=torch.stack(root_t, dim=0),
        root_R=torch.stack(root_R, dim=0),
        cam_t=torch.stack(cam_t, dim=0),
        cam_R=torch.stack(cam_R, dim=0),
    )


def random_rotation(max_deg: float, rng: np.random.Generator) -> torch.Tensor:
    axis = rng.normal(size=3).astype(np.float32)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-8)
    angle = float(rng.uniform(-math.radians(max_deg), math.radians(max_deg)))
    return so3_exp(torch.from_numpy(axis * angle))


def make_episode(
    traj: Trajectory,
    boundaries: list[int],
    max_rot_deg: float,
    max_trans: float,
    global_rot_deg: float,
    global_trans: float,
    rng: np.random.Generator,
) -> Episode:
    T = traj.root_t.shape[0]
    segment_ends = list(boundaries[1:]) + [T]
    global_R = random_rotation(global_rot_deg, rng) if global_rot_deg > 0 else torch.eye(3)
    global_t = (
        torch.from_numpy(rng.uniform(-global_trans, global_trans, size=3).astype(np.float32))
        if global_trans > 0
        else torch.zeros(3)
    )
    target_root_t, target_root_R, target_cam_t, target_cam_R = apply_transform(
        traj.root_t,
        traj.root_R,
        traj.cam_t,
        traj.cam_R,
        global_R,
        global_t,
    )
    local_root_t = target_root_t.clone()
    local_root_R = target_root_R.clone()
    local_cam_t = target_cam_t.clone()
    local_cam_R = target_cam_R.clone()

    for seg_idx, start in enumerate(boundaries):
        end = segment_ends[seg_idx]
        if start == 0:
            R = torch.eye(3)
            t = torch.zeros(3)
        else:
            R = random_rotation(max_rot_deg, rng)
            t = torch.from_numpy(rng.uniform(-max_trans, max_trans, size=3).astype(np.float32))
        rt, rR, ct, cR = apply_transform(
            target_root_t[start:end],
            target_root_R[start:end],
            target_cam_t[start:end],
            target_cam_R[start:end],
            R,
            t,
        )
        local_root_t[start:end] = rt
        local_root_R[start:end] = rR
        local_cam_t[start:end] = ct
        local_cam_R[start:end] = cR

    return Episode(
        local_root_t=local_root_t,
        local_root_R=local_root_R,
        local_cam_t=local_cam_t,
        local_cam_R=local_cam_R,
        target_root_t=target_root_t,
        target_root_R=target_root_R,
        target_cam_t=target_cam_t,
        target_cam_R=target_cam_R,
        boundaries=boundaries,
        segment_ends=segment_ends,
    )


def flatten_feature(parts: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.cat([p.reshape(-1).float() for p in parts], dim=0)


def pairwise_distances(x: torch.Tensor) -> torch.Tensor:
    diffs = x[:, None, :] - x[None, :, :]
    return torch.linalg.norm(diffs, dim=-1)


def build_features(
    local_root_t: torch.Tensor,
    local_root_R: torch.Tensor,
    local_cam_t: torch.Tensor,
    local_cam_R: torch.Tensor,
    hist_root_t: torch.Tensor,
    hist_root_R: torch.Tensor,
    hist_cam_t: torch.Tensor,
    hist_cam_R: torch.Tensor,
    root_vel: torch.Tensor,
    cam_vel: torch.Tensor,
    geo_R: torch.Tensor,
    geo_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    current_centroid = local_root_t.mean(dim=0)
    expected_root_t = hist_root_t + root_vel
    expected_cam_t = hist_cam_t + cam_vel
    expected_centroid = expected_root_t.mean(dim=0)
    local_centered = local_root_t - current_centroid
    expected_centered = expected_root_t - expected_centroid
    current = flatten_feature(
        [
            local_root_t,
            local_centered,
            rot6d(local_root_R),
            local_cam_t,
            rot6d(local_cam_R),
            pairwise_distances(local_root_t),
        ]
    )
    history = flatten_feature(
        [
            local_root_t,
            local_centered,
            rot6d(local_root_R),
            local_cam_t,
            rot6d(local_cam_R),
            hist_root_t,
            expected_root_t,
            expected_centered,
            root_vel,
            rot6d(hist_root_R),
            hist_cam_t,
            expected_cam_t,
            cam_vel,
            rot6d(hist_cam_R),
            expected_centroid - current_centroid,
            pairwise_distances(local_root_t),
            pairwise_distances(expected_root_t),
        ]
    )
    geo_root_t, geo_root_R, geo_cam_t, geo_cam_R = apply_transform(
        local_root_t,
        local_root_R,
        local_cam_t,
        local_cam_R,
        geo_R,
        geo_t,
    )
    residual = flatten_feature(
        [
            history,
            geo_root_t,
            geo_root_t - expected_root_t,
            rot6d(geo_root_R),
            geo_cam_t,
            geo_cam_t - expected_cam_t,
            rot6d(geo_cam_R),
            geo_R,
            geo_t,
        ]
    )
    return current, history, residual


def build_reverse_state_features(
    root_t: torch.Tensor,
    root_R: torch.Tensor,
    cam_t: torch.Tensor,
    cam_R: torch.Tensor,
) -> torch.Tensor:
    centroid = root_t.mean(dim=0)
    return flatten_feature(
        [
            root_t,
            root_t - centroid,
            rot6d(root_R),
            cam_t,
            rot6d(cam_R),
            pairwise_distances(root_t),
        ]
    )


def make_items(episodes: list[Episode]) -> list[BoundaryItem]:
    items = []
    for episode in episodes:
        for boundary in episode.boundaries[1:]:
            hist_idx = boundary - 1
            prev_idx = max(boundary - 2, 0)
            root_vel = episode.target_root_t[hist_idx] - episode.target_root_t[prev_idx]
            cam_vel = episode.target_cam_t[hist_idx] - episode.target_cam_t[prev_idx]
            expected_root_t = episode.target_root_t[hist_idx] + root_vel
            geo_R, geo_t = solve_rigid_transform(episode.local_root_t[boundary], expected_root_t)
            feat_current, feat_history, feat_residual = build_features(
                episode.local_root_t[boundary],
                episode.local_root_R[boundary],
                episode.local_cam_t[boundary],
                episode.local_cam_R[boundary],
                episode.target_root_t[hist_idx],
                episode.target_root_R[hist_idx],
                episode.target_cam_t[hist_idx],
                episode.target_cam_R[hist_idx],
                root_vel,
                cam_vel,
                geo_R,
                geo_t,
            )
            seg_idx = episode.boundaries.index(boundary)
            seg_end = episode.segment_ends[seg_idx]
            future_idx = min(boundary + 1, seg_end - 1)
            future_next_idx = min(boundary + 2, seg_end - 1)
            if future_next_idx == future_idx:
                backward_root_vel = torch.zeros_like(episode.target_root_t[future_idx])
                backward_cam_vel = torch.zeros_like(episode.target_cam_t[future_idx])
            else:
                backward_root_vel = episode.target_root_t[future_idx] - episode.target_root_t[future_next_idx]
                backward_cam_vel = episode.target_cam_t[future_idx] - episode.target_cam_t[future_next_idx]
            backward_expected_root_t = episode.target_root_t[future_idx] + backward_root_vel
            backward_geo_R, backward_geo_t = solve_rigid_transform(
                episode.local_root_t[boundary],
                backward_expected_root_t,
            )
            _, feat_backward, _ = build_features(
                episode.local_root_t[boundary],
                episode.local_root_R[boundary],
                episode.local_cam_t[boundary],
                episode.local_cam_R[boundary],
                episode.target_root_t[future_idx],
                episode.target_root_R[future_idx],
                episode.target_cam_t[future_idx],
                episode.target_cam_R[future_idx],
                backward_root_vel,
                backward_cam_vel,
                backward_geo_R,
                backward_geo_t,
            )
            feat_reverse_current = build_reverse_state_features(
                episode.target_root_t[boundary],
                episode.target_root_R[boundary],
                episode.target_cam_t[boundary],
                episode.target_cam_R[boundary],
            )
            feat_reverse_prev = build_reverse_state_features(
                episode.target_root_t[hist_idx],
                episode.target_root_R[hist_idx],
                episode.target_cam_t[hist_idx],
                episode.target_cam_R[hist_idx],
            )
            items.append(
                BoundaryItem(
                    feature_current=feat_current,
                    feature_history=feat_history,
                    feature_backward=feat_backward,
                    feature_reverse_current=feat_reverse_current,
                    feature_reverse_prev=feat_reverse_prev,
                    feature_residual=feat_residual,
                    local_root_t=episode.local_root_t[boundary],
                    local_root_R=episode.local_root_R[boundary],
                    local_cam_t=episode.local_cam_t[boundary],
                    local_cam_R=episode.local_cam_R[boundary],
                    target_root_t=episode.target_root_t[boundary],
                    target_root_R=episode.target_root_R[boundary],
                    target_cam_t=episode.target_cam_t[boundary],
                    target_cam_R=episode.target_cam_R[boundary],
                    prev_root_t=episode.target_root_t[hist_idx],
                    prev_root_R=episode.target_root_R[hist_idx],
                    prev_cam_t=episode.target_cam_t[hist_idx],
                    prev_cam_R=episode.target_cam_R[hist_idx],
                    geo_R=geo_R,
                    geo_t=geo_t,
                )
            )
    return items


class TransformHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, max_rot_deg: float, max_trans: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 6),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.max_rot = math.radians(float(max_rot_deg))
        self.max_trans = float(max_trans)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.net(x)
        rotvec = self.max_rot * torch.tanh(raw[:, :3])
        trans = self.max_trans * torch.tanh(raw[:, 3:])
        return so3_exp(rotvec), trans


class ReverseFeatureHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, in_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


def item_batch(items: list[BoundaryItem], indices: list[int], key: str, device: torch.device) -> torch.Tensor:
    return torch.stack([getattr(items[i], key) for i in indices], dim=0).to(device)


def batch_loss(
    items: list[BoundaryItem],
    indices: list[int],
    model: TransformHead,
    feature_key: str,
    device: torch.device,
    residual: bool,
) -> torch.Tensor:
    features = item_batch(items, indices, feature_key, device)
    pred_R, pred_t = model(features)
    local_root_t = item_batch(items, indices, "local_root_t", device)
    local_root_R = item_batch(items, indices, "local_root_R", device)
    local_cam_t = item_batch(items, indices, "local_cam_t", device)
    local_cam_R = item_batch(items, indices, "local_cam_R", device)
    target_root_t = item_batch(items, indices, "target_root_t", device)
    target_root_R = item_batch(items, indices, "target_root_R", device)
    target_cam_t = item_batch(items, indices, "target_cam_t", device)
    target_cam_R = item_batch(items, indices, "target_cam_R", device)

    if residual:
        geo_R = item_batch(items, indices, "geo_R", device)
        geo_t = item_batch(items, indices, "geo_t", device)
        geo_root_t, geo_root_R, geo_cam_t, geo_cam_R = apply_transform_batch(
            local_root_t, local_root_R, local_cam_t, local_cam_R, geo_R, geo_t
        )
        out_root_t, out_root_R, out_cam_t, out_cam_R = apply_transform_batch(
            geo_root_t, geo_root_R, geo_cam_t, geo_cam_R, pred_R, pred_t
        )
        prior = pred_t.square().mean() + rotation_geodesic(pred_R, torch.eye(3, device=device)[None]).square().mean()
    else:
        out_root_t, out_root_R, out_cam_t, out_cam_R = apply_transform_batch(
            local_root_t, local_root_R, local_cam_t, local_cam_R, pred_R, pred_t
        )
        prior = torch.zeros((), device=device)

    root_pos = F.mse_loss(out_root_t, target_root_t)
    root_rot = rotation_geodesic(out_root_R, target_root_R).square().mean()
    cam_pos = F.mse_loss(out_cam_t, target_cam_t)
    cam_rot = rotation_geodesic(out_cam_R, target_cam_R).square().mean()
    return 8.0 * root_pos + 1.5 * root_rot + 2.0 * cam_pos + 0.5 * cam_rot + 1e-3 * prior


def build_history_direct_residual_features(
    history_features: torch.Tensor,
    coarse_R: torch.Tensor,
    coarse_t: torch.Tensor,
    coarse_root_t: torch.Tensor,
    coarse_root_R: torch.Tensor,
    coarse_cam_t: torch.Tensor,
    coarse_cam_R: torch.Tensor,
) -> torch.Tensor:
    return torch.cat(
        [
            history_features,
            coarse_R.reshape(coarse_R.shape[0], -1),
            coarse_t,
            coarse_root_t.flatten(1),
            rot6d(coarse_root_R).flatten(1),
            coarse_cam_t,
            rot6d(coarse_cam_R),
        ],
        dim=-1,
    )


def history_direct_residual_batch_loss(
    items: list[BoundaryItem],
    indices: list[int],
    direct_model: TransformHead,
    residual_model: TransformHead,
    device: torch.device,
) -> torch.Tensor:
    features = item_batch(items, indices, "feature_history", device)
    local_root_t = item_batch(items, indices, "local_root_t", device)
    local_root_R = item_batch(items, indices, "local_root_R", device)
    local_cam_t = item_batch(items, indices, "local_cam_t", device)
    local_cam_R = item_batch(items, indices, "local_cam_R", device)
    target_root_t = item_batch(items, indices, "target_root_t", device)
    target_root_R = item_batch(items, indices, "target_root_R", device)
    target_cam_t = item_batch(items, indices, "target_cam_t", device)
    target_cam_R = item_batch(items, indices, "target_cam_R", device)

    direct_model.eval()
    with torch.no_grad():
        coarse_R, coarse_t = direct_model(features)
        coarse_root_t, coarse_root_R, coarse_cam_t, coarse_cam_R = apply_transform_batch(
            local_root_t, local_root_R, local_cam_t, local_cam_R, coarse_R, coarse_t
        )

    residual_features = build_history_direct_residual_features(
        features,
        coarse_R,
        coarse_t,
        coarse_root_t,
        coarse_root_R,
        coarse_cam_t,
        coarse_cam_R,
    )
    res_R, res_t = residual_model(residual_features)
    out_root_t, out_root_R, out_cam_t, out_cam_R = apply_transform_batch(
        coarse_root_t, coarse_root_R, coarse_cam_t, coarse_cam_R, res_R, res_t
    )

    root_pos = F.mse_loss(out_root_t, target_root_t)
    root_rot = rotation_geodesic(out_root_R, target_root_R).square().mean()
    cam_pos = F.mse_loss(out_cam_t, target_cam_t)
    cam_rot = rotation_geodesic(out_cam_R, target_cam_R).square().mean()
    prior = res_t.square().mean() + rotation_geodesic(res_R, torch.eye(3, device=device)[None]).square().mean()
    return 8.0 * root_pos + 1.5 * root_rot + 2.0 * cam_pos + 0.5 * cam_rot + 1e-3 * prior


def reverse_state_batch_loss(
    items: list[BoundaryItem],
    indices: list[int],
    model: TransformHead,
    device: torch.device,
) -> torch.Tensor:
    features = item_batch(items, indices, "feature_reverse_current", device)
    pred_R, pred_t = model(features)
    current_root_t = item_batch(items, indices, "target_root_t", device)
    current_root_R = item_batch(items, indices, "target_root_R", device)
    current_cam_t = item_batch(items, indices, "target_cam_t", device)
    current_cam_R = item_batch(items, indices, "target_cam_R", device)
    prev_root_t = item_batch(items, indices, "prev_root_t", device)
    prev_root_R = item_batch(items, indices, "prev_root_R", device)
    prev_cam_t = item_batch(items, indices, "prev_cam_t", device)
    prev_cam_R = item_batch(items, indices, "prev_cam_R", device)

    out_root_t, out_root_R, out_cam_t, out_cam_R = apply_transform_batch(
        current_root_t,
        current_root_R,
        current_cam_t,
        current_cam_R,
        pred_R,
        pred_t,
    )
    root_pos = F.mse_loss(out_root_t, prev_root_t)
    root_rot = rotation_geodesic(out_root_R, prev_root_R).square().mean()
    cam_pos = F.mse_loss(out_cam_t, prev_cam_t)
    cam_rot = rotation_geodesic(out_cam_R, prev_cam_R).square().mean()
    prior = pred_t.square().mean() + rotation_geodesic(pred_R, torch.eye(3, device=device)[None]).square().mean()
    return 8.0 * root_pos + 1.5 * root_rot + 2.0 * cam_pos + 0.5 * cam_rot + 1e-3 * prior


def reverse_feature_batch_loss(
    items: list[BoundaryItem],
    indices: list[int],
    model: ReverseFeatureHead,
    device: torch.device,
) -> torch.Tensor:
    current_features = item_batch(items, indices, "feature_reverse_current", device)
    prev_features = item_batch(items, indices, "feature_reverse_prev", device)
    pred_prev = model(current_features)
    return F.mse_loss(pred_prev, prev_features)


def bidir_consistency_batch_loss(
    items: list[BoundaryItem],
    indices: list[int],
    model: TransformHead,
    backward_model: TransformHead,
    device: torch.device,
    output_weight: float,
    transform_weight: float,
) -> torch.Tensor:
    features = item_batch(items, indices, "feature_history", device)
    backward_features = item_batch(items, indices, "feature_backward", device)
    pred_R, pred_t = model(features)
    local_root_t = item_batch(items, indices, "local_root_t", device)
    local_root_R = item_batch(items, indices, "local_root_R", device)
    local_cam_t = item_batch(items, indices, "local_cam_t", device)
    local_cam_R = item_batch(items, indices, "local_cam_R", device)
    target_root_t = item_batch(items, indices, "target_root_t", device)
    target_root_R = item_batch(items, indices, "target_root_R", device)
    target_cam_t = item_batch(items, indices, "target_cam_t", device)
    target_cam_R = item_batch(items, indices, "target_cam_R", device)

    out_root_t, out_root_R, out_cam_t, out_cam_R = apply_transform_batch(
        local_root_t, local_root_R, local_cam_t, local_cam_R, pred_R, pred_t
    )
    root_pos = F.mse_loss(out_root_t, target_root_t)
    root_rot = rotation_geodesic(out_root_R, target_root_R).square().mean()
    cam_pos = F.mse_loss(out_cam_t, target_cam_t)
    cam_rot = rotation_geodesic(out_cam_R, target_cam_R).square().mean()
    supervised = 8.0 * root_pos + 1.5 * root_rot + 2.0 * cam_pos + 0.5 * cam_rot

    backward_model.eval()
    with torch.no_grad():
        teacher_R, teacher_t = backward_model(backward_features)
        teacher_root_t, teacher_root_R, teacher_cam_t, teacher_cam_R = apply_transform_batch(
            local_root_t, local_root_R, local_cam_t, local_cam_R, teacher_R, teacher_t
        )

    output_consistency = (
        8.0 * F.mse_loss(out_root_t, teacher_root_t)
        + 1.5 * rotation_geodesic(out_root_R, teacher_root_R).square().mean()
        + 2.0 * F.mse_loss(out_cam_t, teacher_cam_t)
        + 0.5 * rotation_geodesic(out_cam_R, teacher_cam_R).square().mean()
    )
    transform_consistency = F.mse_loss(pred_t, teacher_t) + rotation_geodesic(pred_R, teacher_R).square().mean()
    return supervised + output_weight * output_consistency + transform_weight * transform_consistency


def evaluate_item_loss(
    items: list[BoundaryItem],
    model: TransformHead,
    feature_key: str,
    device: torch.device,
    residual: bool,
    batch_size: int,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            idx = list(range(start, min(start + batch_size, len(items))))
            losses.append(float(batch_loss(items, idx, model, feature_key, device, residual).detach().cpu()))
    return float(np.mean(losses))


def evaluate_bidir_consistency_loss(
    items: list[BoundaryItem],
    model: TransformHead,
    backward_model: TransformHead,
    device: torch.device,
    batch_size: int,
    output_weight: float,
    transform_weight: float,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            idx = list(range(start, min(start + batch_size, len(items))))
            losses.append(
                float(
                    bidir_consistency_batch_loss(
                        items,
                        idx,
                        model,
                        backward_model,
                        device,
                        output_weight,
                        transform_weight,
                    ).detach().cpu()
                )
            )
    return float(np.mean(losses))


def evaluate_reverse_state_loss(
    items: list[BoundaryItem],
    model: TransformHead,
    device: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            idx = list(range(start, min(start + batch_size, len(items))))
            losses.append(float(reverse_state_batch_loss(items, idx, model, device).detach().cpu()))
    return float(np.mean(losses))


def evaluate_reverse_feature_loss(
    items: list[BoundaryItem],
    model: ReverseFeatureHead,
    device: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            idx = list(range(start, min(start + batch_size, len(items))))
            losses.append(float(reverse_feature_batch_loss(items, idx, model, device).detach().cpu()))
    return float(np.mean(losses))


def evaluate_history_direct_residual_loss(
    items: list[BoundaryItem],
    direct_model: TransformHead,
    residual_model: TransformHead,
    device: torch.device,
    batch_size: int,
) -> float:
    residual_model.eval()
    losses = []
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            idx = list(range(start, min(start + batch_size, len(items))))
            losses.append(
                float(
                    history_direct_residual_batch_loss(
                        items, idx, direct_model, residual_model, device
                    ).detach().cpu()
                )
            )
    return float(np.mean(losses))


def train_head(
    name: str,
    train_items: list[BoundaryItem],
    val_items: list[BoundaryItem],
    feature_key: str,
    residual: bool,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> TransformHead:
    in_dim = int(getattr(train_items[0], feature_key).numel())
    max_rot = args.residual_max_rot_deg if residual else args.max_rot_deg
    max_trans = args.residual_max_trans if residual else args.max_trans
    model = TransformHead(in_dim, args.hidden_dim, max_rot, max_trans).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    seed_offsets = {
        "current_only_mlp": 101,
        "history_current_integrator": 202,
        "backward_history_integrator": 303,
        "explicit_se3_residual_integrator": 303,
        "history_direct_residual_integrator": 404,
        "history_bidir_consistency_integrator": 505,
        "history_bidir_direct_residual_integrator": 606,
    }
    rng = random.Random(args.seed + seed_offsets.get(name, 0))
    log_path = output_dir / f"{name}_train_log.jsonl"
    best_state = None
    best_val = float("inf")
    with log_path.open("w", encoding="utf-8") as f:
        for step in range(1, args.steps + 1):
            model.train()
            idx = [rng.randrange(len(train_items)) for _ in range(args.batch_size)]
            loss = batch_loss(train_items, idx, model, feature_key, device, residual)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                val_loss = evaluate_item_loss(val_items, model, feature_key, device, residual, args.batch_size)
                row = {"step": step, "train_loss": float(loss.detach().cpu()), "val_loss": val_loss}
                f.write(json.dumps(row) + "\n")
                f.flush()
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                print(f"[{name}] step {step:04d} train={row['train_loss']:.6f} val={val_loss:.6f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt = {
        "name": name,
        "feature_key": feature_key,
        "residual": residual,
        "state_dict": model.state_dict(),
        "best_val_loss": best_val,
        "args": vars(args),
    }
    torch.save(ckpt, output_dir / f"{name}.pth")
    return model


def train_reverse_state_predictor(
    train_items: list[BoundaryItem],
    val_items: list[BoundaryItem],
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> TransformHead:
    name = "reverse_state_predictor"
    in_dim = int(train_items[0].feature_reverse_current.numel())
    model = TransformHead(in_dim, args.hidden_dim, args.reverse_max_rot_deg, args.reverse_max_trans).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = random.Random(args.seed + 707)
    log_path = output_dir / f"{name}_train_log.jsonl"
    best_state = None
    best_val = float("inf")
    with log_path.open("w", encoding="utf-8") as f:
        for step in range(1, args.steps + 1):
            model.train()
            idx = [rng.randrange(len(train_items)) for _ in range(args.batch_size)]
            loss = reverse_state_batch_loss(train_items, idx, model, device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                val_loss = evaluate_reverse_state_loss(val_items, model, device, args.batch_size)
                row = {"step": step, "train_loss": float(loss.detach().cpu()), "val_loss": val_loss}
                f.write(json.dumps(row) + "\n")
                f.flush()
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                print(f"[{name}] step {step:04d} train={row['train_loss']:.6f} val={val_loss:.6f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt = {
        "name": name,
        "feature_key": "feature_reverse_current",
        "state_dict": model.state_dict(),
        "best_val_loss": best_val,
        "args": vars(args),
        "reverse_max_rot_deg": args.reverse_max_rot_deg,
        "reverse_max_trans": args.reverse_max_trans,
    }
    torch.save(ckpt, output_dir / f"{name}.pth")
    return model


def train_reverse_feature_predictor(
    train_items: list[BoundaryItem],
    val_items: list[BoundaryItem],
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> ReverseFeatureHead:
    name = "reverse_feature_predictor"
    in_dim = int(train_items[0].feature_reverse_current.numel())
    model = ReverseFeatureHead(in_dim, args.hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = random.Random(args.seed + 808)
    log_path = output_dir / f"{name}_train_log.jsonl"
    best_state = None
    best_val = float("inf")
    with log_path.open("w", encoding="utf-8") as f:
        for step in range(1, args.steps + 1):
            model.train()
            idx = [rng.randrange(len(train_items)) for _ in range(args.batch_size)]
            loss = reverse_feature_batch_loss(train_items, idx, model, device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                val_loss = evaluate_reverse_feature_loss(val_items, model, device, args.batch_size)
                row = {"step": step, "train_loss": float(loss.detach().cpu()), "val_loss": val_loss}
                f.write(json.dumps(row) + "\n")
                f.flush()
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                print(f"[{name}] step {step:04d} train={row['train_loss']:.6f} val={val_loss:.6f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt = {
        "name": name,
        "feature_key": "feature_reverse_current_to_prev",
        "state_dict": model.state_dict(),
        "best_val_loss": best_val,
        "args": vars(args),
    }
    torch.save(ckpt, output_dir / f"{name}.pth")
    return model


def train_bidir_consistency_head(
    train_items: list[BoundaryItem],
    val_items: list[BoundaryItem],
    backward_model: TransformHead,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> TransformHead:
    in_dim = int(train_items[0].feature_history.numel())
    name = "history_bidir_consistency_integrator"
    model = TransformHead(in_dim, args.hidden_dim, args.max_rot_deg, args.max_trans).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = random.Random(args.seed + 505)
    log_path = output_dir / f"{name}_train_log.jsonl"
    best_state = None
    best_val = float("inf")
    with log_path.open("w", encoding="utf-8") as f:
        for step in range(1, args.steps + 1):
            model.train()
            idx = [rng.randrange(len(train_items)) for _ in range(args.batch_size)]
            loss = bidir_consistency_batch_loss(
                train_items,
                idx,
                model,
                backward_model,
                device,
                args.bidir_output_weight,
                args.bidir_transform_weight,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                val_loss = evaluate_bidir_consistency_loss(
                    val_items,
                    model,
                    backward_model,
                    device,
                    args.batch_size,
                    args.bidir_output_weight,
                    args.bidir_transform_weight,
                )
                row = {"step": step, "train_loss": float(loss.detach().cpu()), "val_loss": val_loss}
                f.write(json.dumps(row) + "\n")
                f.flush()
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                print(f"[{name}] step {step:04d} train={row['train_loss']:.6f} val={val_loss:.6f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt = {
        "name": name,
        "feature_key": "feature_history",
        "teacher": "backward_history_integrator",
        "bidir_output_weight": args.bidir_output_weight,
        "bidir_transform_weight": args.bidir_transform_weight,
        "state_dict": model.state_dict(),
        "best_val_loss": best_val,
        "args": vars(args),
    }
    torch.save(ckpt, output_dir / f"{name}.pth")
    return model


def train_history_direct_residual_head(
    train_items: list[BoundaryItem],
    val_items: list[BoundaryItem],
    direct_model: TransformHead,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
    name: str = "history_direct_residual_integrator",
    direct_name: str = "history_current_integrator",
) -> TransformHead:
    direct_model.eval()
    with torch.no_grad():
        sample_features = item_batch(train_items, [0], "feature_history", device)
        coarse_R, coarse_t = direct_model(sample_features)
        coarse_root_t, coarse_root_R, coarse_cam_t, coarse_cam_R = apply_transform_batch(
            item_batch(train_items, [0], "local_root_t", device),
            item_batch(train_items, [0], "local_root_R", device),
            item_batch(train_items, [0], "local_cam_t", device),
            item_batch(train_items, [0], "local_cam_R", device),
            coarse_R,
            coarse_t,
        )
        sample_residual_features = build_history_direct_residual_features(
            sample_features,
            coarse_R,
            coarse_t,
            coarse_root_t,
            coarse_root_R,
            coarse_cam_t,
            coarse_cam_R,
        )

    model = TransformHead(
        int(sample_residual_features.shape[-1]),
        args.hidden_dim,
        args.residual_max_rot_deg,
        args.residual_max_trans,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    seed_offsets = {
        "history_direct_residual_integrator": 404,
        "history_bidir_direct_residual_integrator": 606,
    }
    rng = random.Random(args.seed + seed_offsets.get(name, 404))
    log_path = output_dir / f"{name}_train_log.jsonl"
    best_state = None
    best_val = float("inf")
    with log_path.open("w", encoding="utf-8") as f:
        for step in range(1, args.steps + 1):
            model.train()
            idx = [rng.randrange(len(train_items)) for _ in range(args.batch_size)]
            loss = history_direct_residual_batch_loss(train_items, idx, direct_model, model, device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                val_loss = evaluate_history_direct_residual_loss(
                    val_items, direct_model, model, device, args.batch_size
                )
                row = {"step": step, "train_loss": float(loss.detach().cpu()), "val_loss": val_loss}
                f.write(json.dumps(row) + "\n")
                f.flush()
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                print(f"[{name}] step {step:04d} train={row['train_loss']:.6f} val={val_loss:.6f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt = {
        "name": name,
        "feature_key": "feature_history_direct_residual",
        "residual": True,
        "residual_base": direct_name,
        "state_dict": model.state_dict(),
        "best_val_loss": best_val,
        "args": vars(args),
    }
    torch.save(ckpt, output_dir / f"{name}.pth")
    return model


def make_episodes(traj: Trajectory, count: int, args: argparse.Namespace, seed_offset: int) -> list[Episode]:
    rng = np.random.default_rng(args.seed + seed_offset)
    boundaries = sorted(set(int(x) for x in args.segment_boundaries))
    if boundaries[0] != 0:
        boundaries = [0] + boundaries
    if any(x < 0 or x >= traj.root_t.shape[0] for x in boundaries):
        raise ValueError(f"segment_boundaries must be within [0, {traj.root_t.shape[0] - 1}]")
    return [
        make_episode(
            traj,
            boundaries,
            args.perturb_rot_deg,
            args.perturb_trans,
            args.global_rot_deg,
            args.global_trans,
            rng,
        )
        for _ in range(count)
    ]


def refine_transform_with_cycle_guidance(
    local_root_t: torch.Tensor,
    local_root_R: torch.Tensor,
    local_cam_t: torch.Tensor,
    local_cam_R: torch.Tensor,
    base_R: torch.Tensor,
    base_t: torch.Tensor,
    prev_root_t: torch.Tensor,
    prev_root_R: torch.Tensor,
    prev_cam_t: torch.Tensor,
    prev_cam_R: torch.Tensor,
    expected_root_t: torch.Tensor,
    expected_cam_t: torch.Tensor,
    reverse_model: TransformHead,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if int(args.cycle_refine_steps) <= 0:
        return base_R.detach().cpu(), base_t.detach().cpu()

    reverse_model.eval()
    local_root_t_b = local_root_t[None].to(device)
    local_root_R_b = local_root_R[None].to(device)
    local_cam_t_b = local_cam_t[None].to(device)
    local_cam_R_b = local_cam_R[None].to(device)
    base_R = base_R.detach().to(device)
    base_t = base_t.detach().to(device)
    prev_root_t_b = prev_root_t[None].to(device)
    prev_root_R_b = prev_root_R[None].to(device)
    prev_cam_t_b = prev_cam_t[None].to(device)
    prev_cam_R_b = prev_cam_R[None].to(device)
    expected_root_t_b = expected_root_t[None].to(device)
    expected_cam_t_b = expected_cam_t[None].to(device)
    eye = torch.eye(3, device=device)

    raw_rot = torch.zeros(1, 3, device=device, requires_grad=True)
    raw_t = torch.zeros(1, 3, device=device, requires_grad=True)
    opt = torch.optim.Adam([raw_rot, raw_t], lr=float(args.cycle_refine_lr))
    max_rot = math.radians(float(args.residual_max_rot_deg))
    max_trans = float(args.residual_max_trans)

    for _ in range(int(args.cycle_refine_steps)):
        R_delta = so3_exp(max_rot * torch.tanh(raw_rot))[0]
        t_delta = max_trans * torch.tanh(raw_t)[0]
        R_cur = R_delta @ base_R
        t_cur = R_delta @ base_t + t_delta
        cur_root_t, cur_root_R, cur_cam_t, cur_cam_R = apply_transform_batch(
            local_root_t_b,
            local_root_R_b,
            local_cam_t_b,
            local_cam_R_b,
            R_cur[None],
            t_cur[None],
        )
        reverse_features = build_reverse_state_features(
            cur_root_t[0],
            cur_root_R[0],
            cur_cam_t[0],
            cur_cam_R[0],
        )[None]
        rev_R, rev_t = reverse_model(reverse_features)
        prev_hat_root_t, prev_hat_root_R, prev_hat_cam_t, prev_hat_cam_R = apply_transform_batch(
            cur_root_t,
            cur_root_R,
            cur_cam_t,
            cur_cam_R,
            rev_R,
            rev_t,
        )
        cycle = (
            8.0 * F.mse_loss(prev_hat_root_t, prev_root_t_b)
            + 1.5 * rotation_geodesic(prev_hat_root_R, prev_root_R_b).square().mean()
            + 2.0 * F.mse_loss(prev_hat_cam_t, prev_cam_t_b)
            + 0.5 * rotation_geodesic(prev_hat_cam_R, prev_cam_R_b).square().mean()
        )
        smooth = F.mse_loss(cur_root_t, expected_root_t_b) + 0.25 * F.mse_loss(cur_cam_t, expected_cam_t_b)
        prior = t_delta.square().mean() + rotation_geodesic(R_delta[None], eye[None]).square().mean()
        loss = (
            cycle
            + float(args.cycle_refine_prior_weight) * prior
            + float(args.cycle_refine_smooth_weight) * smooth
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with torch.no_grad():
        R_delta = so3_exp(max_rot * torch.tanh(raw_rot))[0]
        t_delta = max_trans * torch.tanh(raw_t)[0]
        final_R = R_delta @ base_R
        final_t = R_delta @ base_t + t_delta
    return final_R.cpu(), final_t.cpu()


def refine_transform_with_feature_cycle_guidance(
    local_root_t: torch.Tensor,
    local_root_R: torch.Tensor,
    local_cam_t: torch.Tensor,
    local_cam_R: torch.Tensor,
    base_R: torch.Tensor,
    base_t: torch.Tensor,
    prev_root_t: torch.Tensor,
    prev_root_R: torch.Tensor,
    prev_cam_t: torch.Tensor,
    prev_cam_R: torch.Tensor,
    expected_root_t: torch.Tensor,
    expected_cam_t: torch.Tensor,
    reverse_feature_model: ReverseFeatureHead,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if int(args.cycle_refine_steps) <= 0:
        return base_R.detach().cpu(), base_t.detach().cpu()

    reverse_feature_model.eval()
    local_root_t_b = local_root_t[None].to(device)
    local_root_R_b = local_root_R[None].to(device)
    local_cam_t_b = local_cam_t[None].to(device)
    local_cam_R_b = local_cam_R[None].to(device)
    base_R = base_R.detach().to(device)
    base_t = base_t.detach().to(device)
    target_prev_feature = build_reverse_state_features(
        prev_root_t.to(device),
        prev_root_R.to(device),
        prev_cam_t.to(device),
        prev_cam_R.to(device),
    )[None]
    expected_root_t_b = expected_root_t[None].to(device)
    expected_cam_t_b = expected_cam_t[None].to(device)
    eye = torch.eye(3, device=device)

    raw_rot = torch.zeros(1, 3, device=device, requires_grad=True)
    raw_t = torch.zeros(1, 3, device=device, requires_grad=True)
    opt = torch.optim.Adam([raw_rot, raw_t], lr=float(args.cycle_refine_lr))
    max_rot = math.radians(float(args.residual_max_rot_deg))
    max_trans = float(args.residual_max_trans)

    for _ in range(int(args.cycle_refine_steps)):
        R_delta = so3_exp(max_rot * torch.tanh(raw_rot))[0]
        t_delta = max_trans * torch.tanh(raw_t)[0]
        R_cur = R_delta @ base_R
        t_cur = R_delta @ base_t + t_delta
        cur_root_t, cur_root_R, cur_cam_t, cur_cam_R = apply_transform_batch(
            local_root_t_b,
            local_root_R_b,
            local_cam_t_b,
            local_cam_R_b,
            R_cur[None],
            t_cur[None],
        )
        cur_feature = build_reverse_state_features(
            cur_root_t[0],
            cur_root_R[0],
            cur_cam_t[0],
            cur_cam_R[0],
        )[None]
        pred_prev_feature = reverse_feature_model(cur_feature)
        cycle = F.mse_loss(pred_prev_feature, target_prev_feature)
        smooth = F.mse_loss(cur_root_t, expected_root_t_b) + 0.25 * F.mse_loss(cur_cam_t, expected_cam_t_b)
        prior = t_delta.square().mean() + rotation_geodesic(R_delta[None], eye[None]).square().mean()
        loss = (
            float(args.cycle_feature_weight) * cycle
            + float(args.cycle_refine_prior_weight) * prior
            + float(args.cycle_refine_smooth_weight) * smooth
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with torch.no_grad():
        R_delta = so3_exp(max_rot * torch.tanh(raw_rot))[0]
        t_delta = max_trans * torch.tanh(raw_t)[0]
        final_R = R_delta @ base_R
        final_t = R_delta @ base_t + t_delta
    return final_R.cpu(), final_t.cpu()


def stream_apply_variant(
    episode: Episode,
    variant: str,
    models: dict[str, TransformHead],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    T = episode.target_root_t.shape[0]
    out_root_t = torch.empty_like(episode.target_root_t)
    out_root_R = torch.empty_like(episode.target_root_R)
    out_cam_t = torch.empty_like(episode.target_cam_t)
    out_cam_R = torch.empty_like(episode.target_cam_R)
    boundaries = episode.boundaries
    segment_ends = episode.segment_ends

    for seg_idx, start in enumerate(boundaries):
        end = segment_ends[seg_idx]
        if start == 0:
            R_cache = torch.eye(3)
            t_cache = torch.zeros(3)
        elif variant == "raw_perturbed":
            R_cache = torch.eye(3)
            t_cache = torch.zeros(3)
        else:
            hist_idx = start - 1
            prev_idx = max(start - 2, 0)
            root_vel = out_root_t[hist_idx] - out_root_t[prev_idx]
            cam_vel = out_cam_t[hist_idx] - out_cam_t[prev_idx]
            expected_root_t = out_root_t[hist_idx] + root_vel
            expected_cam_t = out_cam_t[hist_idx] + cam_vel
            geo_R, geo_t = solve_rigid_transform(episode.local_root_t[start], expected_root_t)

            if variant == "fixed_explicit_se3":
                R_cache, t_cache = geo_R, geo_t
            elif variant == "oracle_se3_upper":
                R_cache, t_cache = solve_rigid_transform(episode.local_root_t[start], episode.target_root_t[start])
            else:
                feat_current, feat_history, feat_residual = build_features(
                    episode.local_root_t[start],
                    episode.local_root_R[start],
                    episode.local_cam_t[start],
                    episode.local_cam_R[start],
                    out_root_t[hist_idx],
                    out_root_R[hist_idx],
                    out_cam_t[hist_idx],
                    out_cam_R[hist_idx],
                    root_vel,
                    cam_vel,
                    geo_R,
                    geo_t,
                )
                if variant == "current_only_mlp":
                    model = models["current_only_mlp"]
                    features = feat_current[None].to(device)
                    with torch.no_grad():
                        R_pred, t_pred = model(features)
                    R_cache, t_cache = R_pred[0].cpu(), t_pred[0].cpu()
                elif variant == "history_current_integrator":
                    model = models["history_current_integrator"]
                    features = feat_history[None].to(device)
                    with torch.no_grad():
                        R_pred, t_pred = model(features)
                    R_cache, t_cache = R_pred[0].cpu(), t_pred[0].cpu()
                elif variant == "history_bidir_consistency_integrator":
                    model = models["history_bidir_consistency_integrator"]
                    features = feat_history[None].to(device)
                    with torch.no_grad():
                        R_pred, t_pred = model(features)
                    R_cache, t_cache = R_pred[0].cpu(), t_pred[0].cpu()
                elif variant == "history_direct_residual_integrator":
                    direct_model = models["history_current_integrator"]
                    residual_model = models["history_direct_residual_integrator"]
                    features = feat_history[None].to(device)
                    local_root_t = episode.local_root_t[start : start + 1].to(device)
                    local_root_R = episode.local_root_R[start : start + 1].to(device)
                    local_cam_t = episode.local_cam_t[start : start + 1].to(device)
                    local_cam_R = episode.local_cam_R[start : start + 1].to(device)
                    with torch.no_grad():
                        R_direct, t_direct = direct_model(features)
                        coarse_root_t, coarse_root_R, coarse_cam_t, coarse_cam_R = apply_transform_batch(
                            local_root_t, local_root_R, local_cam_t, local_cam_R, R_direct, t_direct
                        )
                        residual_features = build_history_direct_residual_features(
                            features,
                            R_direct,
                            t_direct,
                            coarse_root_t,
                            coarse_root_R,
                            coarse_cam_t,
                            coarse_cam_R,
                        )
                        R_res, t_res = residual_model(residual_features)
                    R_cache = (R_res[0] @ R_direct[0]).cpu()
                    t_cache = (R_res[0] @ t_direct[0] + t_res[0]).cpu()
                elif variant == "history_direct_residual_cycle_guided":
                    direct_model = models["history_current_integrator"]
                    residual_model = models["history_direct_residual_integrator"]
                    reverse_model = models["reverse_state_predictor"]
                    features = feat_history[None].to(device)
                    local_root_t = episode.local_root_t[start : start + 1].to(device)
                    local_root_R = episode.local_root_R[start : start + 1].to(device)
                    local_cam_t = episode.local_cam_t[start : start + 1].to(device)
                    local_cam_R = episode.local_cam_R[start : start + 1].to(device)
                    with torch.no_grad():
                        R_direct, t_direct = direct_model(features)
                        coarse_root_t, coarse_root_R, coarse_cam_t, coarse_cam_R = apply_transform_batch(
                            local_root_t, local_root_R, local_cam_t, local_cam_R, R_direct, t_direct
                        )
                        residual_features = build_history_direct_residual_features(
                            features,
                            R_direct,
                            t_direct,
                            coarse_root_t,
                            coarse_root_R,
                            coarse_cam_t,
                            coarse_cam_R,
                        )
                        R_res, t_res = residual_model(residual_features)
                        R_base = R_res[0] @ R_direct[0]
                        t_base = R_res[0] @ t_direct[0] + t_res[0]
                    R_cache, t_cache = refine_transform_with_cycle_guidance(
                        episode.local_root_t[start],
                        episode.local_root_R[start],
                        episode.local_cam_t[start],
                        episode.local_cam_R[start],
                        R_base,
                        t_base,
                        out_root_t[hist_idx],
                        out_root_R[hist_idx],
                        out_cam_t[hist_idx],
                        out_cam_R[hist_idx],
                        expected_root_t,
                        expected_cam_t,
                        reverse_model,
                        args,
                        device,
                    )
                elif variant == "history_direct_residual_feature_cycle_guided":
                    direct_model = models["history_current_integrator"]
                    residual_model = models["history_direct_residual_integrator"]
                    reverse_feature_model = models["reverse_feature_predictor"]
                    features = feat_history[None].to(device)
                    local_root_t = episode.local_root_t[start : start + 1].to(device)
                    local_root_R = episode.local_root_R[start : start + 1].to(device)
                    local_cam_t = episode.local_cam_t[start : start + 1].to(device)
                    local_cam_R = episode.local_cam_R[start : start + 1].to(device)
                    with torch.no_grad():
                        R_direct, t_direct = direct_model(features)
                        coarse_root_t, coarse_root_R, coarse_cam_t, coarse_cam_R = apply_transform_batch(
                            local_root_t, local_root_R, local_cam_t, local_cam_R, R_direct, t_direct
                        )
                        residual_features = build_history_direct_residual_features(
                            features,
                            R_direct,
                            t_direct,
                            coarse_root_t,
                            coarse_root_R,
                            coarse_cam_t,
                            coarse_cam_R,
                        )
                        R_res, t_res = residual_model(residual_features)
                        R_base = R_res[0] @ R_direct[0]
                        t_base = R_res[0] @ t_direct[0] + t_res[0]
                    R_cache, t_cache = refine_transform_with_feature_cycle_guidance(
                        episode.local_root_t[start],
                        episode.local_root_R[start],
                        episode.local_cam_t[start],
                        episode.local_cam_R[start],
                        R_base,
                        t_base,
                        out_root_t[hist_idx],
                        out_root_R[hist_idx],
                        out_cam_t[hist_idx],
                        out_cam_R[hist_idx],
                        expected_root_t,
                        expected_cam_t,
                        reverse_feature_model,
                        args,
                        device,
                    )
                elif variant == "history_bidir_direct_residual_integrator":
                    direct_model = models["history_bidir_consistency_integrator"]
                    residual_model = models["history_bidir_direct_residual_integrator"]
                    features = feat_history[None].to(device)
                    local_root_t = episode.local_root_t[start : start + 1].to(device)
                    local_root_R = episode.local_root_R[start : start + 1].to(device)
                    local_cam_t = episode.local_cam_t[start : start + 1].to(device)
                    local_cam_R = episode.local_cam_R[start : start + 1].to(device)
                    with torch.no_grad():
                        R_direct, t_direct = direct_model(features)
                        coarse_root_t, coarse_root_R, coarse_cam_t, coarse_cam_R = apply_transform_batch(
                            local_root_t, local_root_R, local_cam_t, local_cam_R, R_direct, t_direct
                        )
                        residual_features = build_history_direct_residual_features(
                            features,
                            R_direct,
                            t_direct,
                            coarse_root_t,
                            coarse_root_R,
                            coarse_cam_t,
                            coarse_cam_R,
                        )
                        R_res, t_res = residual_model(residual_features)
                    R_cache = (R_res[0] @ R_direct[0]).cpu()
                    t_cache = (R_res[0] @ t_direct[0] + t_res[0]).cpu()
                elif variant == "explicit_se3_residual_integrator":
                    model = models["explicit_se3_residual_integrator"]
                    features = feat_residual[None].to(device)
                    with torch.no_grad():
                        R_res, t_res = model(features)
                    R_res = R_res[0].cpu()
                    t_res = t_res[0].cpu()
                    R_cache = R_res @ geo_R
                    t_cache = R_res @ geo_t + t_res
                else:
                    raise ValueError(f"Unknown variant: {variant}")

        rt, rR, ct, cR = apply_transform(
            episode.local_root_t[start:end],
            episode.local_root_R[start:end],
            episode.local_cam_t[start:end],
            episode.local_cam_R[start:end],
            R_cache,
            t_cache,
        )
        out_root_t[start:end] = rt
        out_root_R[start:end] = rR
        out_cam_t[start:end] = ct
        out_cam_R[start:end] = cR

    return out_root_t, out_root_R, out_cam_t, out_cam_R


def compute_metrics(
    episode: Episode,
    pred: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, float]:
    root_t, root_R, cam_t, cam_R = pred
    target_root_t = episode.target_root_t
    target_root_R = episode.target_root_R
    target_cam_t = episode.target_cam_t
    target_cam_R = episode.target_cam_R

    root_trans = torch.linalg.norm(root_t - target_root_t, dim=-1)
    root_rot = rotation_geodesic(root_R, target_root_R) * (180.0 / math.pi)
    cam_trans = torch.linalg.norm(cam_t - target_cam_t, dim=-1)
    cam_rot = rotation_geodesic(cam_R, target_cam_R) * (180.0 / math.pi)

    pred_root_vel = root_t[1:] - root_t[:-1]
    gt_root_vel = target_root_t[1:] - target_root_t[:-1]
    vel = torch.linalg.norm(pred_root_vel - gt_root_vel, dim=-1)
    if root_t.shape[0] > 2:
        pred_acc = root_t[2:] - 2 * root_t[1:-1] + root_t[:-2]
        gt_acc = target_root_t[2:] - 2 * target_root_t[1:-1] + target_root_t[:-2]
        accel = torch.linalg.norm(pred_acc - gt_acc, dim=-1)
    else:
        accel = torch.zeros(1)

    boundary_errors = []
    for boundary in episode.boundaries[1:]:
        pred_jump = root_t[boundary] - root_t[boundary - 1]
        gt_jump = target_root_t[boundary] - target_root_t[boundary - 1]
        boundary_errors.append(torch.linalg.norm(pred_jump - gt_jump, dim=-1))
    boundary = torch.cat(boundary_errors, dim=0) if boundary_errors else torch.zeros(1)

    non_boundary = []
    boundaries = set(episode.boundaries[1:])
    for idx in range(1, root_t.shape[0]):
        if idx not in boundaries:
            pred_jump = root_t[idx] - root_t[idx - 1]
            gt_jump = target_root_t[idx] - target_root_t[idx - 1]
            non_boundary.append(torch.linalg.norm(pred_jump - gt_jump, dim=-1))
    non_boundary_t = torch.cat(non_boundary, dim=0) if non_boundary else torch.zeros(1)

    dist_pred = torch.linalg.norm(root_t[:, :, None, :] - root_t[:, None, :, :], dim=-1)
    dist_gt = torch.linalg.norm(target_root_t[:, :, None, :] - target_root_t[:, None, :, :], dim=-1)
    person_dist = torch.abs(dist_pred - dist_gt)

    return {
        "root_trans_m": float(root_trans.mean()),
        "root_rot_deg": float(root_rot.mean()),
        "cam_trans_m": float(cam_trans.mean()),
        "cam_rot_deg": float(cam_rot.mean()),
        "boundary_jump_m": float(boundary.mean()),
        "velocity_m": float(vel.mean()),
        "accel_m": float(accel.mean()),
        "non_boundary_motion_m": float(non_boundary_t.mean()),
        "inter_person_dist_m": float(person_dist.mean()),
    }


def evaluate_variants(
    episodes: list[Episode],
    models: dict[str, TransformHead],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    variants = [
        "raw_perturbed",
        "fixed_explicit_se3",
        "current_only_mlp",
        "history_current_integrator",
        "history_direct_residual_integrator",
        "history_direct_residual_cycle_guided",
        "history_direct_residual_feature_cycle_guided",
        "history_bidir_consistency_integrator",
        "history_bidir_direct_residual_integrator",
        "explicit_se3_residual_integrator",
        "oracle_se3_upper",
    ]
    rows: dict[str, list[dict[str, float]]] = {name: [] for name in variants}
    for episode in episodes:
        for variant in variants:
            pred = stream_apply_variant(episode, variant, models, args, device)
            rows[variant].append(compute_metrics(episode, pred))
    summary: dict[str, dict[str, float]] = {}
    for variant, values in rows.items():
        keys = values[0].keys()
        summary[variant] = {key: float(np.mean([v[key] for v in values])) for key in keys}
    return summary


def write_summary(output_dir: Path, summary: dict[str, dict[str, float]], args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    (output_dir / "metrics_summary.json").write_text(
        json.dumps({"summary": summary, "args": serializable_args}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    fieldnames = ["variant"] + list(next(iter(summary.values())).keys())
    with (output_dir / "metrics_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for variant, metrics in summary.items():
            row = {"variant": variant}
            row.update(metrics)
            writer.writerow(row)

    lines = [
        "# V10 BEDLAM Motion / Global-State Integrator Probe",
        "",
        "All variants are evaluated with a strict causal streaming loop. Lower is better.",
        "",
        "| Variant | Root Trans | Root Rot | Cam Trans | Cam Rot | Boundary Jump | Velocity | Non-boundary |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, metrics in summary.items():
        lines.append(
            "| {variant} | {root_trans_m:.4f} | {root_rot_deg:.2f} | {cam_trans_m:.4f} | "
            "{cam_rot_deg:.2f} | {boundary_jump_m:.4f} | {velocity_m:.4f} | "
            "{non_boundary_motion_m:.4f} |".format(variant=variant, **metrics)
        )
    (output_dir / "metrics_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    if args.trajectory_source == "bedlam_gt":
        traj = load_bedlam_trajectory(args.manifest)
        source_desc = str(args.manifest)
    else:
        if args.human3r_output_dir is None:
            raise ValueError("--human3r_output_dir is required when --trajectory_source human3r_saved")
        traj = load_human3r_saved_trajectory(args.human3r_output_dir, args.num_frames, args.max_people)
        source_desc = str(args.human3r_output_dir)
    train_episodes = make_episodes(traj, args.train_episodes, args, seed_offset=0)
    val_episodes = make_episodes(traj, args.val_episodes, args, seed_offset=100000)
    train_items = make_items(train_episodes)
    val_items = make_items(val_episodes)
    print(
        f"Loaded {args.trajectory_source} from {source_desc} frames={len(traj.frames)} people={traj.root_t.shape[1]} "
        f"train_items={len(train_items)} val_items={len(val_items)} device={device}"
    )

    models = {
        "current_only_mlp": train_head(
            "current_only_mlp",
            train_items,
            val_items,
            "feature_current",
            residual=False,
            args=args,
            device=device,
            output_dir=args.output_dir,
        ),
        "history_current_integrator": train_head(
            "history_current_integrator",
            train_items,
            val_items,
            "feature_history",
            residual=False,
            args=args,
            device=device,
            output_dir=args.output_dir,
        ),
        "backward_history_integrator": train_head(
            "backward_history_integrator",
            train_items,
            val_items,
            "feature_backward",
            residual=False,
            args=args,
            device=device,
            output_dir=args.output_dir,
        ),
    }
    models["history_direct_residual_integrator"] = train_history_direct_residual_head(
        train_items,
        val_items,
        models["history_current_integrator"],
        args=args,
        device=device,
        output_dir=args.output_dir,
        name="history_direct_residual_integrator",
        direct_name="history_current_integrator",
    )
    models["history_bidir_consistency_integrator"] = train_bidir_consistency_head(
        train_items,
        val_items,
        models["backward_history_integrator"],
        args=args,
        device=device,
        output_dir=args.output_dir,
    )
    models["history_bidir_direct_residual_integrator"] = train_history_direct_residual_head(
        train_items,
        val_items,
        models["history_bidir_consistency_integrator"],
        args=args,
        device=device,
        output_dir=args.output_dir,
        name="history_bidir_direct_residual_integrator",
        direct_name="history_bidir_consistency_integrator",
    )
    models["reverse_state_predictor"] = train_reverse_state_predictor(
        train_items,
        val_items,
        args=args,
        device=device,
        output_dir=args.output_dir,
    )
    models["reverse_feature_predictor"] = train_reverse_feature_predictor(
        train_items,
        val_items,
        args=args,
        device=device,
        output_dir=args.output_dir,
    )
    models["explicit_se3_residual_integrator"] = train_head(
        "explicit_se3_residual_integrator",
        train_items,
        val_items,
        "feature_residual",
        residual=True,
        args=args,
        device=device,
        output_dir=args.output_dir,
    )

    for model in models.values():
        model.eval()
    summary = evaluate_variants(val_episodes, models, args, device)
    write_summary(args.output_dir, summary, args)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
