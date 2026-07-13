#!/usr/bin/env python3
"""V10 static-human streaming alignment probe with body-frame supervision.

This keeps the oracle-boundary setup and frozen original Human3R from the V9
alignment probe, but tests whether body orientation cues help a learnable
segment-to-global alignment module.  The target setting is near-static humans:
the model should align a new local shot segment to the historical global gauge
without only matching root translation.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
ARCHIVE_V7 = SCRIPTS_ROOT / "archive_v7"
for path in (str(REPO_ROOT), str(SRC_ROOT), str(SCRIPTS_ROOT), str(ARCHIVE_V7)):
    if path not in sys.path:
        sys.path.insert(0, path)

from overfit_human_anchor_pose_correction import FOOT_JOINTS, STABLE_JOINTS
from v9_learned_stream_alignment_4source_probe import (
    DEFAULT_BAD_SAMPLE_REGISTRY,
    SOURCE_ORDER,
    apply_transform_batch,
    cache_samples,
    evaluate_and_write_outputs,
    select_records,
)
from v9_learned_stream_alignment_overfit import (
    StreamingAlignmentMLP,
    build_feature,
    rotation_geodesic,
    so3_exp,
)


BODY_FRAME_JOINTS = {
    "pelvis": 0,
    "left_hip": 1,
    "right_hip": 2,
    "spine": 9,
    "head": 15,
    "left_shoulder": 16,
    "right_shoulder": 17,
    "left_foot": 10,
    "right_foot": 11,
}

BODY_VECTOR_PAIRS = (
    (0, 15),  # pelvis -> head
    (0, 9),   # pelvis -> upper torso
    (1, 2),   # hip left/right axis
    (16, 17), # shoulder left/right axis
    (0, 10),  # pelvis -> left foot
    (0, 11),  # pelvis -> right foot
)

BODY_ANCHOR_JOINTS = (
    0,   # pelvis/root
    1,   # left hip
    2,   # right hip
    9,   # spine/torso
    10,  # left foot
    11,  # right foot
    15,  # head
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=REPO_ROOT / "output" / "v10_static_alignment_probe" / "base",
    )
    parser.add_argument(
        "--manifest_root",
        type=Path,
        default=REPO_ROOT / "config" / "manifests" / "v9_4source_pattern_probe",
    )
    parser.add_argument(
        "--manifest_map",
        type=Path,
        default=None,
        help="Optional JSON source_manifests map for reusing large V9 manifest sets.",
    )
    parser.add_argument("--manifest_name", default="train_aabb.jsonl")
    parser.add_argument(
        "--bad_sample_registry",
        type=Path,
        default=DEFAULT_BAD_SAMPLE_REGISTRY,
        help="JSONL blacklist for samples whose saved Human3R output is unusable.",
    )
    parser.add_argument("--model_path", type=Path, default=REPO_ROOT / "src" / "human3r_896L.pth")
    parser.add_argument("--data_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data"))
    parser.add_argument("--sources", nargs="+", default=list(SOURCE_ORDER), choices=list(SOURCE_ORDER))
    parser.add_argument("--samples_per_source", type=int, default=2)
    parser.add_argument("--boundary", type=int, default=2)
    parser.add_argument("--resolution", type=int, nargs=2, default=(512, 288), metavar=("W", "H"))
    parser.add_argument("--resize_mode", default="human3r_demo")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--max_rot_deg", type=float, default=180.0)
    parser.add_argument("--max_trans", type=float, default=12.0)
    parser.add_argument(
        "--alignment_mode",
        choices=["direct_se3", "geo_only", "geo_residual"],
        default="direct_se3",
        help="direct_se3 is the old black-box SE(3) head; geo modes use explicit human-anchor geometry first.",
    )
    parser.add_argument("--residual_max_rot_deg", type=float, default=10.0)
    parser.add_argument("--residual_max_trans", type=float, default=0.5)
    parser.add_argument("--residual_target_weight", type=float, default=1.0)
    parser.add_argument("--residual_prior_weight", type=float, default=0.1)
    parser.add_argument("--proposal_improvement_weight", type=float, default=1.0)
    parser.add_argument("--human_weight", type=float, default=5.0)
    parser.add_argument("--camera_t_weight", type=float, default=2.0)
    parser.add_argument("--camera_r_weight", type=float, default=1.0)
    parser.add_argument("--body_frame_weight", type=float, default=0.0)
    parser.add_argument("--body_vector_weight", type=float, default=0.0)
    parser.add_argument("--body_anchor_weight", type=float, default=0.0)
    parser.add_argument("--body_vertical_weight", type=float, default=0.0)
    parser.add_argument("--prior_weight", type=float, default=1e-4)
    parser.add_argument("--use_body_frame_input", action="store_true")
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--skip_inference", action="store_true")
    parser.add_argument(
        "--skip_bad_samples",
        action="store_true",
        help="Skip cached samples whose saved Human3R output has no detected human.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_vec(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def safe_joint(joints: torch.Tensor, idx: int) -> torch.Tensor:
    max_idx = joints.shape[-2] - 1
    return joints[..., min(int(idx), max_idx), :]


def body_frame_from_joints(joints: torch.Tensor) -> torch.Tensor:
    """Construct an orthonormal body frame from predicted/GT joints.

    Columns are roughly [right, up, forward].  This is intentionally simple and
    deterministic so the ablation isolates whether body-frame information helps.
    """
    pelvis = safe_joint(joints, BODY_FRAME_JOINTS["pelvis"])
    head = safe_joint(joints, BODY_FRAME_JOINTS["head"])
    spine = safe_joint(joints, BODY_FRAME_JOINTS["spine"])
    left_hip = safe_joint(joints, BODY_FRAME_JOINTS["left_hip"])
    right_hip = safe_joint(joints, BODY_FRAME_JOINTS["right_hip"])
    left_shoulder = safe_joint(joints, BODY_FRAME_JOINTS["left_shoulder"])
    right_shoulder = safe_joint(joints, BODY_FRAME_JOINTS["right_shoulder"])

    up = normalize_vec(head - pelvis)
    fallback_up = normalize_vec(spine - pelvis)
    up_norm = (head - pelvis).norm(dim=-1, keepdim=True)
    up = torch.where(up_norm > 1e-4, up, fallback_up)

    right = right_hip - left_hip
    shoulder_right = right_shoulder - left_shoulder
    right = torch.where(right.norm(dim=-1, keepdim=True) > 1e-4, right, shoulder_right)
    right = normalize_vec(right)

    forward = normalize_vec(torch.cross(right, up, dim=-1))
    right = normalize_vec(torch.cross(up, forward, dim=-1))
    return torch.stack([right, up, forward], dim=-1)


def body_vectors(joints: torch.Tensor) -> torch.Tensor:
    vecs = []
    for a, b in BODY_VECTOR_PAIRS:
        if a >= joints.shape[-2] or b >= joints.shape[-2]:
            continue
        vecs.append(normalize_vec(joints[..., b, :] - joints[..., a, :]))
    if not vecs:
        return joints.new_zeros((*joints.shape[:-2], 0, 3))
    return torch.stack(vecs, dim=-2)


def valid_joint_ids(joints: torch.Tensor, joint_ids: tuple[int, ...]) -> torch.Tensor:
    valid = [idx for idx in joint_ids if idx < joints.shape[-2]]
    if not valid:
        valid = [0]
    return torch.as_tensor(valid, device=joints.device, dtype=torch.long)


def build_v10_feature(
    pred_joints: torch.Tensor,
    boundary: int,
    joint_ids: torch.Tensor,
    use_body_frame_input: bool,
) -> torch.Tensor:
    base = build_feature(pred_joints, boundary, joint_ids)
    if not use_body_frame_input:
        return base

    hist_joints = pred_joints[:boundary].mean(dim=0, keepdim=True)
    cur_joints = pred_joints[boundary : boundary + 1]
    hist_frame = body_frame_from_joints(hist_joints)
    cur_frame = body_frame_from_joints(cur_joints)
    rel_frame = hist_frame.transpose(-1, -2) @ cur_frame
    body_feat = torch.cat(
        [
            hist_frame.flatten(),
            cur_frame.flatten(),
            rel_frame.flatten(),
            (cur_frame - hist_frame).flatten(),
        ],
        dim=0,
    ).reshape(1, -1)
    return torch.cat([base, body_feat], dim=-1)


def fixed_geo_anchor_weights(joint_ids: torch.Tensor) -> torch.Tensor:
    weights = torch.ones_like(joint_ids, dtype=torch.float32)
    priority = {
        0: 3.0,   # pelvis/root
        1: 2.0,   # left hip
        2: 2.0,   # right hip
        9: 2.5,   # torso/spine
        10: 1.5,  # left foot
        11: 1.5,  # right foot
        15: 2.0,  # head
    }
    for i, joint_id in enumerate(joint_ids.detach().cpu().tolist()):
        weights[i] = float(priority.get(int(joint_id), 1.0))
    return weights.to(device=joint_ids.device)


def solve_weighted_rigid_transform_batch(
    src: torch.Tensor,
    dst: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return R,t with R @ src + t ~= dst, no scale."""
    weights = weights.to(device=src.device, dtype=src.dtype)
    if weights.ndim == 1:
        weights = weights[None, :, None].expand(src.shape[0], -1, -1)
    elif weights.ndim == 2:
        weights = weights[:, :, None]
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    src_mean = (weights * src).sum(dim=1, keepdim=True)
    dst_mean = (weights * dst).sum(dim=1, keepdim=True)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    H = torch.einsum("bnc,bnd->bcd", weights * src_c, dst_c)
    U, _, Vh = torch.linalg.svd(H)
    R = Vh.transpose(-1, -2) @ U.transpose(-1, -2)
    det = torch.det(R)
    if torch.any(det < 0):
        Vh = Vh.clone()
        Vh[det < 0, -1, :] *= -1.0
        R = Vh.transpose(-1, -2) @ U.transpose(-1, -2)
    t = dst_mean[:, 0] - torch.einsum("bij,bj->bi", R, src_mean[:, 0])
    return R, t


def compute_geo_proposal(
    pred_joints: torch.Tensor,
    boundary: int,
    joint_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hist = pred_joints[:, :boundary, joint_ids].mean(dim=1)
    cur = pred_joints[:, boundary, joint_ids]
    weights = fixed_geo_anchor_weights(joint_ids)
    R_geo, t_geo = solve_weighted_rigid_transform_batch(cur, hist, weights)
    return R_geo, t_geo, weights


def transform_joints(joints: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bij,bfkj->bfki", R, joints) + t[:, None, None, :]


def build_geo_residual_features(
    pred_joints: torch.Tensor,
    R_geo: torch.Tensor,
    t_geo: torch.Tensor,
    boundary: int,
    joint_ids: torch.Tensor,
) -> torch.Tensor:
    hist = pred_joints[:, :boundary, joint_ids].mean(dim=1)
    cur = pred_joints[:, boundary, joint_ids]
    cur_geo = torch.einsum("bij,bnj->bni", R_geo, cur) + t_geo[:, None, :]
    hist_center = hist.mean(dim=1, keepdim=True)
    cur_geo_center = cur_geo.mean(dim=1, keepdim=True)
    hist_shape = hist - hist_center
    cur_geo_shape = cur_geo - cur_geo_center
    residual = cur_geo - hist
    center_residual = cur_geo_center - hist_center
    return torch.cat(
        [
            hist_shape.flatten(start_dim=1),
            cur_geo_shape.flatten(start_dim=1),
            residual.flatten(start_dim=1),
            hist_center.flatten(start_dim=1),
            cur_geo_center.flatten(start_dim=1),
            center_residual.flatten(start_dim=1),
            R_geo.flatten(start_dim=1),
            t_geo,
        ],
        dim=-1,
    )


class GeometryResidualAlignmentMLP(nn.Module):
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
            nn.Linear(hidden_dim, 7),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.max_rot = np.deg2rad(float(max_rot_deg))
        self.max_trans = float(max_trans)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.net(x)
        gate = torch.sigmoid(raw[:, 6:7])
        delta_rotvec = float(self.max_rot) * torch.tanh(raw[:, :3]) * gate
        delta_trans = float(self.max_trans) * torch.tanh(raw[:, 3:6]) * gate
        return delta_rotvec, delta_trans, gate.squeeze(-1)


def compose_residual_with_geo(
    delta_rotvec: torch.Tensor,
    delta_trans: torch.Tensor,
    R_geo: torch.Tensor,
    t_geo: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    R_delta = so3_exp(delta_rotvec)
    R_final = R_delta @ R_geo
    t_final = torch.einsum("bij,bj->bi", R_delta, t_geo) + delta_trans
    return R_final, t_final, R_delta


def target_transform_from_boundary_pose(
    pred_poses: torch.Tensor,
    target_poses: torch.Tensor,
    boundary: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred = pred_poses[:, boundary]
    target = target_poses[:, boundary]
    R_target = target[:, :3, :3] @ pred[:, :3, :3].transpose(-1, -2)
    t_target = target[:, :3, 3] - torch.einsum("bij,bj->bi", R_target, pred[:, :3, 3])
    return R_target, t_target


def residual_target_from_geo(
    R_target: torch.Tensor,
    t_target: torch.Tensor,
    R_geo: torch.Tensor,
    t_geo: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    R_res = R_target @ R_geo.transpose(-1, -2)
    t_res = t_target - torch.einsum("bij,bj->bi", R_res, t_geo)
    return R_res, t_res


def body_frame_metrics(
    joints: np.ndarray,
    target_joints: np.ndarray,
    frame_ids: list[int],
) -> dict:
    joints_t = torch.from_numpy(joints).float()
    target_t = torch.from_numpy(target_joints).float()
    frame = body_frame_from_joints(joints_t[frame_ids])
    target = body_frame_from_joints(target_t[frame_ids])
    err = rotation_geodesic(frame.reshape(-1, 3, 3), target.reshape(-1, 3, 3))
    return {
        "mean_deg": float(torch.rad2deg(err).mean().item()),
        "max_deg": float(torch.rad2deg(err).max().item()),
    }


def write_v10_extra_metrics(
    samples,
    aligned_joints: np.ndarray,
    args: argparse.Namespace,
    debug: dict,
) -> dict:
    boundary = int(args.boundary)
    rows = []
    for i, sample in enumerate(samples):
        raw_body = body_frame_metrics(
            sample.pred_joints,
            sample.target_joints,
            list(range(boundary, 4)),
        )
        aligned_body = body_frame_metrics(
            aligned_joints[i],
            sample.target_joints,
            list(range(boundary, 4)),
        )
        rows.append(
            {
                "index": int(sample.index),
                "source": sample.source,
                "pattern_id": sample.pattern_id,
                "raw_body_frame_deg": raw_body["mean_deg"],
                "aligned_body_frame_deg": aligned_body["mean_deg"],
                "gain_body_frame_deg": raw_body["mean_deg"] - aligned_body["mean_deg"],
            }
        )

    aggregate = {
        "raw_body_frame_deg": float(np.mean([r["raw_body_frame_deg"] for r in rows])),
        "aligned_body_frame_deg": float(np.mean([r["aligned_body_frame_deg"] for r in rows])),
        "gain_body_frame_deg": float(np.mean([r["gain_body_frame_deg"] for r in rows])),
    }
    by_source = {}
    for source in sorted({r["source"] for r in rows}):
        source_rows = [r for r in rows if r["source"] == source]
        by_source[source] = {
            "raw_body_frame_deg": float(np.mean([r["raw_body_frame_deg"] for r in source_rows])),
            "aligned_body_frame_deg": float(np.mean([r["aligned_body_frame_deg"] for r in source_rows])),
            "gain_body_frame_deg": float(np.mean([r["gain_body_frame_deg"] for r in source_rows])),
        }

    output = {
        "variant": {
            "use_body_frame_input": bool(args.use_body_frame_input),
            "body_frame_weight": float(args.body_frame_weight),
            "body_vector_weight": float(args.body_vector_weight),
            "body_anchor_weight": float(args.body_anchor_weight),
            "body_vertical_weight": float(args.body_vertical_weight),
        },
        "aggregate": aggregate,
        "by_source": by_source,
        "samples": rows,
        "train_history": debug.get("history", []),
    }
    (args.output_dir / "v10_body_frame_metrics.json").write_text(
        json.dumps(output, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with (args.output_dir / "v10_body_frame_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return output


def train_v10_alignment(samples, args: argparse.Namespace, device: torch.device):
    pred_poses = torch.from_numpy(np.stack([sample.pred_poses for sample in samples])).to(device=device, dtype=torch.float32)
    pred_joints = torch.from_numpy(np.stack([sample.pred_joints for sample in samples])).to(device=device, dtype=torch.float32)
    target_poses = torch.from_numpy(np.stack([sample.target_poses for sample in samples])).to(device=device, dtype=torch.float32)
    target_joints = torch.from_numpy(np.stack([sample.target_joints for sample in samples])).to(device=device, dtype=torch.float32)
    joint_ids_np = np.asarray(sorted(set(STABLE_JOINTS + FOOT_JOINTS)), dtype=np.int64)
    joint_ids = torch.from_numpy(joint_ids_np).to(device=device, dtype=torch.long)
    boundary = int(args.boundary)
    if str(args.alignment_mode) in {"geo_only", "geo_residual"}:
        return train_geometry_alignment(pred_poses, pred_joints, target_poses, target_joints, joint_ids, samples, args, device)

    features = torch.cat(
        [
            build_v10_feature(
                pred_joints[i],
                boundary,
                joint_ids,
                bool(args.use_body_frame_input),
            )
            for i in range(pred_joints.shape[0])
        ],
        dim=0,
    ).to(device)

    aligner = StreamingAlignmentMLP(
        in_dim=features.shape[-1],
        hidden_dim=int(args.hidden_dim),
        max_rot_deg=float(args.max_rot_deg),
        max_trans=float(args.max_trans),
    ).to(device)
    optim = torch.optim.AdamW(aligner.parameters(), lr=float(args.lr), weight_decay=1e-4)
    post = slice(boundary, pred_poses.shape[1])
    history = []
    log_path = args.output_dir / "alignment_train_steps.jsonl"
    if log_path.exists() and args.overwrite:
        log_path.unlink()

    for step in range(int(args.steps) + 1):
        optim.zero_grad(set_to_none=True)
        rotvec, trans = aligner(features)
        R = so3_exp(rotvec)
        aligned_poses, aligned_joints = apply_transform_batch(pred_poses, pred_joints, R, trans, boundary)

        human_loss = F.smooth_l1_loss(
            aligned_joints[:, post][:, :, joint_ids],
            target_joints[:, post][:, :, joint_ids],
            beta=0.05,
        )
        camera_t_loss = F.smooth_l1_loss(aligned_poses[:, post, :3, 3], target_poses[:, post, :3, 3], beta=0.05)
        camera_r_loss = rotation_geodesic(
            aligned_poses[:, post, :3, :3].reshape(-1, 3, 3),
            target_poses[:, post, :3, :3].reshape(-1, 3, 3),
        ).mean()
        aligned_body = body_frame_from_joints(aligned_joints[:, post])
        target_body = body_frame_from_joints(target_joints[:, post])
        body_frame_loss = rotation_geodesic(
            aligned_body.reshape(-1, 3, 3),
            target_body.reshape(-1, 3, 3),
        ).mean()
        aligned_vec = body_vectors(aligned_joints[:, post])
        target_vec = body_vectors(target_joints[:, post])
        body_vector_loss = (
            aligned_vec.new_tensor(0.0)
            if aligned_vec.shape[-2] == 0
            else F.smooth_l1_loss(aligned_vec, target_vec, beta=0.05)
        )
        anchor_ids = valid_joint_ids(aligned_joints, BODY_ANCHOR_JOINTS)
        aligned_anchor = aligned_joints[:, post][:, :, anchor_ids]
        target_anchor = target_joints[:, post][:, :, anchor_ids]
        aligned_center = aligned_anchor.mean(dim=-2)
        target_center = target_anchor.mean(dim=-2)
        body_anchor_loss = F.smooth_l1_loss(aligned_center, target_center, beta=0.05)
        body_anchor_loss = body_anchor_loss + 0.5 * F.smooth_l1_loss(
            aligned_anchor,
            target_anchor,
            beta=0.05,
        )
        target_up = target_body[..., :, 1]
        vertical_offset = ((aligned_center - target_center) * target_up).sum(dim=-1, keepdim=True)
        body_vertical_loss = F.smooth_l1_loss(
            vertical_offset,
            torch.zeros_like(vertical_offset),
            beta=0.02,
        )
        prior_loss = rotvec.pow(2).mean() + trans.pow(2).mean()
        loss = (
            float(args.human_weight) * human_loss
            + float(args.camera_t_weight) * camera_t_loss
            + float(args.camera_r_weight) * camera_r_loss
            + float(args.body_frame_weight) * body_frame_loss
            + float(args.body_vector_weight) * body_vector_loss
            + float(args.body_anchor_weight) * body_anchor_loss
            + float(args.body_vertical_weight) * body_vertical_loss
            + float(args.prior_weight) * prior_loss
        )
        loss.backward()
        optim.step()

        if step % int(args.log_every) == 0 or step == int(args.steps):
            row = {
                "step": int(step),
                "loss": float(loss.detach().cpu()),
                "human_loss": float(human_loss.detach().cpu()),
                "camera_t_loss": float(camera_t_loss.detach().cpu()),
                "camera_r_deg": float(torch.rad2deg(camera_r_loss.detach()).cpu()),
                "body_frame_deg": float(torch.rad2deg(body_frame_loss.detach()).cpu()),
                "body_vector_loss": float(body_vector_loss.detach().cpu()),
                "body_anchor_loss": float(body_anchor_loss.detach().cpu()),
                "body_vertical_loss": float(body_vertical_loss.detach().cpu()),
                "rotvec_deg_mean": float(torch.rad2deg(rotvec.norm(dim=-1).detach()).mean().cpu()),
                "trans_norm_mean": float(trans.norm(dim=-1).detach().mean().cpu()),
            }
            print(json.dumps(row, sort_keys=True), flush=True)
            history.append(row)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")

    with torch.no_grad():
        rotvec, trans = aligner(features)
        R = so3_exp(rotvec)
        aligned_poses, aligned_joints = apply_transform_batch(pred_poses, pred_joints, R, trans, boundary)

    checkpoint = {
        "model": aligner.state_dict(),
        "features": features.detach().cpu(),
        "joint_ids": joint_ids.detach().cpu(),
        "rotvec": rotvec.detach().cpu(),
        "trans": trans.detach().cpu(),
        "args": vars(args),
        "samples": [
            {
                "source": sample.source,
                "pattern_id": sample.pattern_id,
                "record": sample.record,
            }
            for sample in samples
        ],
    }
    torch.save(checkpoint, args.output_dir / "alignment_head_4source_probe.pth")
    debug = {
        "history": history,
        "joint_ids": joint_ids_np.astype(int).tolist(),
        "alignment_mode": "direct_se3",
        "learned_rotvec_deg_norm": torch.rad2deg(rotvec.norm(dim=-1)).detach().cpu().numpy().astype(float).tolist(),
        "learned_trans_norm": trans.norm(dim=-1).detach().cpu().numpy().astype(float).tolist(),
        "use_body_frame_input": bool(args.use_body_frame_input),
        "body_frame_weight": float(args.body_frame_weight),
        "body_vector_weight": float(args.body_vector_weight),
        "body_anchor_weight": float(args.body_anchor_weight),
        "body_vertical_weight": float(args.body_vertical_weight),
    }
    return (
        aligned_poses.detach().cpu().numpy().astype(np.float32),
        aligned_joints.detach().cpu().numpy().astype(np.float32),
        debug,
    )


def train_geometry_alignment(
    pred_poses: torch.Tensor,
    pred_joints: torch.Tensor,
    target_poses: torch.Tensor,
    target_joints: torch.Tensor,
    joint_ids: torch.Tensor,
    samples,
    args: argparse.Namespace,
    device: torch.device,
):
    boundary = int(args.boundary)
    joint_ids_np = joint_ids.detach().cpu().numpy().astype(np.int64)
    post = slice(boundary, pred_poses.shape[1])
    R_geo, t_geo, geo_weights = compute_geo_proposal(pred_joints, boundary, joint_ids)
    geo_poses, geo_joints = apply_transform_batch(pred_poses, pred_joints, R_geo, t_geo, boundary)

    def compute_losses(aligned_poses: torch.Tensor, aligned_joints: torch.Tensor) -> dict[str, torch.Tensor]:
        human_loss = F.smooth_l1_loss(
            aligned_joints[:, post][:, :, joint_ids],
            target_joints[:, post][:, :, joint_ids],
            beta=0.05,
        )
        camera_t_loss = F.smooth_l1_loss(aligned_poses[:, post, :3, 3], target_poses[:, post, :3, 3], beta=0.05)
        camera_r_loss = rotation_geodesic(
            aligned_poses[:, post, :3, :3].reshape(-1, 3, 3),
            target_poses[:, post, :3, :3].reshape(-1, 3, 3),
        ).mean()
        aligned_body = body_frame_from_joints(aligned_joints[:, post])
        target_body = body_frame_from_joints(target_joints[:, post])
        body_frame_loss = rotation_geodesic(
            aligned_body.reshape(-1, 3, 3),
            target_body.reshape(-1, 3, 3),
        ).mean()
        aligned_vec = body_vectors(aligned_joints[:, post])
        target_vec = body_vectors(target_joints[:, post])
        body_vector_loss = (
            aligned_vec.new_tensor(0.0)
            if aligned_vec.shape[-2] == 0
            else F.smooth_l1_loss(aligned_vec, target_vec, beta=0.05)
        )
        anchor_ids = valid_joint_ids(aligned_joints, BODY_ANCHOR_JOINTS)
        aligned_anchor = aligned_joints[:, post][:, :, anchor_ids]
        target_anchor = target_joints[:, post][:, :, anchor_ids]
        aligned_center = aligned_anchor.mean(dim=-2)
        target_center = target_anchor.mean(dim=-2)
        body_anchor_loss = F.smooth_l1_loss(aligned_center, target_center, beta=0.05)
        body_anchor_loss = body_anchor_loss + 0.5 * F.smooth_l1_loss(aligned_anchor, target_anchor, beta=0.05)
        target_up = target_body[..., :, 1]
        vertical_offset = ((aligned_center - target_center) * target_up).sum(dim=-1, keepdim=True)
        body_vertical_loss = F.smooth_l1_loss(vertical_offset, torch.zeros_like(vertical_offset), beta=0.02)
        return {
            "human_loss": human_loss,
            "camera_t_loss": camera_t_loss,
            "camera_r_loss": camera_r_loss,
            "body_frame_loss": body_frame_loss,
            "body_vector_loss": body_vector_loss,
            "body_anchor_loss": body_anchor_loss,
            "body_vertical_loss": body_vertical_loss,
        }

    def weighted_final_loss(losses: dict[str, torch.Tensor]) -> torch.Tensor:
        return (
            float(args.human_weight) * losses["human_loss"]
            + float(args.camera_t_weight) * losses["camera_t_loss"]
            + float(args.camera_r_weight) * losses["camera_r_loss"]
            + float(args.body_frame_weight) * losses["body_frame_loss"]
            + float(args.body_vector_weight) * losses["body_vector_loss"]
            + float(args.body_anchor_weight) * losses["body_anchor_loss"]
            + float(args.body_vertical_weight) * losses["body_vertical_loss"]
        )

    geo_losses = compute_losses(geo_poses, geo_joints)
    history = []
    log_path = args.output_dir / "alignment_train_steps.jsonl"
    if log_path.exists() and args.overwrite:
        log_path.unlink()

    if str(args.alignment_mode) == "geo_only":
        loss = weighted_final_loss(geo_losses)
        eye = torch.eye(3, device=device, dtype=R_geo.dtype).unsqueeze(0).expand_as(R_geo)
        row = {
            "step": 0,
            "mode": "geo_only",
            "loss": float(loss.detach().cpu()),
            "human_loss": float(geo_losses["human_loss"].detach().cpu()),
            "camera_t_loss": float(geo_losses["camera_t_loss"].detach().cpu()),
            "camera_r_deg": float(torch.rad2deg(geo_losses["camera_r_loss"].detach()).cpu()),
            "body_frame_deg": float(torch.rad2deg(geo_losses["body_frame_loss"].detach()).cpu()),
            "body_vector_loss": float(geo_losses["body_vector_loss"].detach().cpu()),
            "body_anchor_loss": float(geo_losses["body_anchor_loss"].detach().cpu()),
            "body_vertical_loss": float(geo_losses["body_vertical_loss"].detach().cpu()),
            "geo_rot_deg_mean": float(torch.rad2deg(rotation_geodesic(R_geo, eye)).mean().detach().cpu()),
            "geo_trans_norm_mean": float(t_geo.norm(dim=-1).mean().detach().cpu()),
        }
        print(json.dumps(row, sort_keys=True), flush=True)
        history.append(row)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        torch.save(
            {
                "args": vars(args),
                "joint_ids": joint_ids.detach().cpu(),
                "geo_weights": geo_weights.detach().cpu(),
                "R_geo": R_geo.detach().cpu(),
                "t_geo": t_geo.detach().cpu(),
            },
            args.output_dir / "alignment_head_4source_probe.pth",
        )
        debug = {
            "history": history,
            "joint_ids": joint_ids_np.astype(int).tolist(),
            "alignment_mode": "geo_only",
            "geo_weights": geo_weights.detach().cpu().numpy().astype(float).tolist(),
            "geo_rot_deg_norm": torch.rad2deg(rotation_geodesic(R_geo, eye)).detach().cpu().numpy().astype(float).tolist(),
            "geo_trans_norm": t_geo.norm(dim=-1).detach().cpu().numpy().astype(float).tolist(),
        }
        return geo_poses.detach().cpu().numpy().astype(np.float32), geo_joints.detach().cpu().numpy().astype(np.float32), debug

    features = build_geo_residual_features(pred_joints, R_geo.detach(), t_geo.detach(), boundary, joint_ids).to(device)
    aligner = GeometryResidualAlignmentMLP(
        in_dim=features.shape[-1],
        hidden_dim=int(args.hidden_dim),
        max_rot_deg=float(args.residual_max_rot_deg),
        max_trans=float(args.residual_max_trans),
    ).to(device)
    optim = torch.optim.AdamW(aligner.parameters(), lr=float(args.lr), weight_decay=1e-4)
    R_target, t_target = target_transform_from_boundary_pose(pred_poses, target_poses, boundary)
    R_res_target, t_res_target = residual_target_from_geo(R_target, t_target, R_geo.detach(), t_geo.detach())
    geo_anchor_error = torch.linalg.norm(
        geo_joints[:, post][:, :, joint_ids] - target_joints[:, post][:, :, joint_ids],
        dim=-1,
    ).detach()

    for step in range(int(args.steps) + 1):
        optim.zero_grad(set_to_none=True)
        delta_rotvec, delta_trans, gate = aligner(features)
        R_final, t_final, R_delta = compose_residual_with_geo(delta_rotvec, delta_trans, R_geo.detach(), t_geo.detach())
        aligned_poses, aligned_joints = apply_transform_batch(pred_poses, pred_joints, R_final, t_final, boundary)
        losses = compute_losses(aligned_poses, aligned_joints)
        final_loss = weighted_final_loss(losses)
        residual_r_loss = rotation_geodesic(R_delta, R_res_target.detach()).mean()
        residual_t_loss = F.smooth_l1_loss(delta_trans, t_res_target.detach(), beta=0.05)
        residual_target_loss = residual_r_loss + residual_t_loss
        residual_prior_loss = delta_rotvec.pow(2).mean() + delta_trans.pow(2).mean()
        final_anchor_error = torch.linalg.norm(
            aligned_joints[:, post][:, :, joint_ids] - target_joints[:, post][:, :, joint_ids],
            dim=-1,
        )
        proposal_improvement_loss = F.relu(final_anchor_error - geo_anchor_error).mean()
        loss = (
            final_loss
            + float(args.residual_target_weight) * residual_target_loss
            + float(args.residual_prior_weight) * residual_prior_loss
            + float(args.proposal_improvement_weight) * proposal_improvement_loss
        )
        loss.backward()
        optim.step()

        if step % int(args.log_every) == 0 or step == int(args.steps):
            row = {
                "step": int(step),
                "mode": "geo_residual",
                "loss": float(loss.detach().cpu()),
                "final_loss": float(final_loss.detach().cpu()),
                "human_loss": float(losses["human_loss"].detach().cpu()),
                "camera_t_loss": float(losses["camera_t_loss"].detach().cpu()),
                "camera_r_deg": float(torch.rad2deg(losses["camera_r_loss"].detach()).cpu()),
                "body_frame_deg": float(torch.rad2deg(losses["body_frame_loss"].detach()).cpu()),
                "body_vector_loss": float(losses["body_vector_loss"].detach().cpu()),
                "body_anchor_loss": float(losses["body_anchor_loss"].detach().cpu()),
                "body_vertical_loss": float(losses["body_vertical_loss"].detach().cpu()),
                "residual_r_deg": float(torch.rad2deg(residual_r_loss.detach()).cpu()),
                "residual_t_loss": float(residual_t_loss.detach().cpu()),
                "residual_prior_loss": float(residual_prior_loss.detach().cpu()),
                "proposal_improvement_loss": float(proposal_improvement_loss.detach().cpu()),
                "delta_rot_deg_mean": float(torch.rad2deg(delta_rotvec.norm(dim=-1).detach()).mean().cpu()),
                "delta_trans_norm_mean": float(delta_trans.norm(dim=-1).detach().mean().cpu()),
                "gate_mean": float(gate.detach().mean().cpu()),
                "geo_loss": float(weighted_final_loss(geo_losses).detach().cpu()),
                "geo_human_loss": float(geo_losses["human_loss"].detach().cpu()),
                "geo_camera_t_loss": float(geo_losses["camera_t_loss"].detach().cpu()),
                "geo_camera_r_deg": float(torch.rad2deg(geo_losses["camera_r_loss"].detach()).cpu()),
            }
            print(json.dumps(row, sort_keys=True), flush=True)
            history.append(row)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")

    with torch.no_grad():
        delta_rotvec, delta_trans, gate = aligner(features)
        R_final, t_final, R_delta = compose_residual_with_geo(delta_rotvec, delta_trans, R_geo, t_geo)
        aligned_poses, aligned_joints = apply_transform_batch(pred_poses, pred_joints, R_final, t_final, boundary)
    torch.save(
        {
            "model": aligner.state_dict(),
            "args": vars(args),
            "features": features.detach().cpu(),
            "joint_ids": joint_ids.detach().cpu(),
            "geo_weights": geo_weights.detach().cpu(),
            "R_geo": R_geo.detach().cpu(),
            "t_geo": t_geo.detach().cpu(),
            "delta_rotvec": delta_rotvec.detach().cpu(),
            "delta_trans": delta_trans.detach().cpu(),
            "gate": gate.detach().cpu(),
            "samples": [
                {
                    "source": sample.source,
                    "pattern_id": sample.pattern_id,
                    "record": sample.record,
                }
                for sample in samples
            ],
        },
        args.output_dir / "alignment_head_4source_probe.pth",
    )
    debug = {
        "history": history,
        "joint_ids": joint_ids_np.astype(int).tolist(),
        "alignment_mode": "geo_residual",
        "geo_weights": geo_weights.detach().cpu().numpy().astype(float).tolist(),
        "geo_rot_deg_norm": torch.rad2deg(
            rotation_geodesic(R_geo, torch.eye(3, device=device, dtype=R_geo.dtype).unsqueeze(0).expand_as(R_geo))
        ).detach().cpu().numpy().astype(float).tolist(),
        "geo_trans_norm": t_geo.norm(dim=-1).detach().cpu().numpy().astype(float).tolist(),
        "delta_rot_deg_norm": torch.rad2deg(delta_rotvec.norm(dim=-1)).detach().cpu().numpy().astype(float).tolist(),
        "delta_trans_norm": delta_trans.norm(dim=-1).detach().cpu().numpy().astype(float).tolist(),
        "gate": gate.detach().cpu().numpy().astype(float).tolist(),
        "residual_max_rot_deg": float(args.residual_max_rot_deg),
        "residual_max_trans": float(args.residual_max_trans),
    }
    return (
        aligned_poses.detach().cpu().numpy().astype(np.float32),
        aligned_joints.detach().cpu().numpy().astype(np.float32),
        debug,
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(23)
    np.random.seed(23)
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_args.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    records = select_records(args)
    (args.output_dir / "selected_records.json").write_text(
        json.dumps(records, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    selected_text = ", ".join(f"{r['source']}:{r['pattern_id']}" for r in records)
    print(f">> selected {len(records)} samples: {selected_text}")
    samples = cache_samples(records, args, device)
    aligned_poses, aligned_joints, debug = train_v10_alignment(samples, args, device)
    summary = evaluate_and_write_outputs(samples, aligned_poses, aligned_joints, debug, args)
    extra = write_v10_extra_metrics(samples, aligned_joints, args, debug)
    print(
        json.dumps(
            {
                "aggregate": summary["aggregate"],
                "body_frame": extra["aggregate"],
                "outputs": summary["outputs"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
