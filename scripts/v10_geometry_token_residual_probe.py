#!/usr/bin/env python3
"""V10 geometry proposal plus token residual alignment probe.

This is Route C from the V10 notes:

    explicit human-anchor T_geo
    + compact Human3R token pair features
    + small gated residual around T_geo

The goal is to test whether token/state information can improve a strong
streaming human-geometry proposal without letting the learned head predict a
large unconstrained SE(3) transform.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

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
from v10_static_alignment_4source_probe import (
    BODY_ANCHOR_JOINTS,
    build_geo_residual_features,
    body_frame_from_joints,
    body_vectors,
    compute_geo_proposal,
    compose_residual_with_geo,
    residual_target_from_geo,
    target_transform_from_boundary_pose,
    valid_joint_ids,
)
from v10_token_alignment_4source_probe import TOKEN_FEATURE_SETS, build_pair_feature, cache_samples
from v9_learned_stream_alignment_4source_probe import (
    DEFAULT_BAD_SAMPLE_REGISTRY,
    SOURCE_ORDER,
    apply_transform_batch,
    bad_sample_key,
    evaluate_and_write_outputs,
    load_bad_sample_keys,
    read_jsonl,
    safe_name,
    source_manifest_paths,
)
from v9_learned_stream_alignment_overfit import rotation_geodesic, so3_exp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=REPO_ROOT
        / "output"
        / "v10_geometry_token_residual_probe"
        / "4source_s2_geo_human_token",
    )
    parser.add_argument(
        "--manifest_root",
        type=Path,
        default=REPO_ROOT / "config" / "manifests" / "v9_4source_pattern_probe",
    )
    parser.add_argument("--manifest_map", type=Path, default=None)
    parser.add_argument("--manifest_name", default="train_aabb.jsonl")
    parser.add_argument("--bad_sample_registry", type=Path, default=DEFAULT_BAD_SAMPLE_REGISTRY)
    parser.add_argument("--model_path", type=Path, default=REPO_ROOT / "src" / "human3r_896L.pth")
    parser.add_argument("--data_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data"))
    parser.add_argument("--sources", nargs="+", default=list(SOURCE_ORDER), choices=list(SOURCE_ORDER))
    parser.add_argument("--samples_per_source", type=int, default=2)
    parser.add_argument(
        "--source_offset",
        type=int,
        default=0,
        help="Skip this many usable records per source before selecting samples; useful for held-out eval.",
    )
    parser.add_argument("--boundary", type=int, default=2)
    parser.add_argument("--resolution", type=int, nargs=2, default=(512, 288), metavar=("W", "H"))
    parser.add_argument("--resize_mode", default="human3r_demo")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--token_feature_set", choices=list(TOKEN_FEATURE_SETS), default="human_only")
    parser.add_argument("--residual_max_rot_deg", type=float, default=10.0)
    parser.add_argument("--residual_max_trans", type=float, default=0.5)
    parser.add_argument("--human_weight", type=float, default=10.0)
    parser.add_argument("--camera_t_weight", type=float, default=2.0)
    parser.add_argument("--camera_r_weight", type=float, default=1.0)
    parser.add_argument("--residual_target_weight", type=float, default=1.0)
    parser.add_argument("--residual_prior_weight", type=float, default=0.1)
    parser.add_argument("--proposal_improvement_weight", type=float, default=10.0)
    parser.add_argument("--body_frame_weight", type=float, default=1.0)
    parser.add_argument("--body_vector_weight", type=float, default=1.0)
    parser.add_argument("--body_anchor_weight", type=float, default=5.0)
    parser.add_argument("--body_vertical_weight", type=float, default=5.0)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--skip_inference", action="store_true")
    parser.add_argument("--skip_bad_samples", action="store_true")
    parser.add_argument("--eval_checkpoint", type=Path, default=None)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def select_records(args: argparse.Namespace) -> list[dict]:
    selected = []
    manifest_paths = source_manifest_paths(args)
    bad_keys = load_bad_sample_keys(getattr(args, "bad_sample_registry", None))
    boundary = int(getattr(args, "boundary", 2))
    default_shot_labels = [0, 0, 0, 0]
    if 0 <= boundary < len(default_shot_labels):
        default_shot_labels[boundary] = 1
    source_offset = max(0, int(getattr(args, "source_offset", 0)))
    for source in args.sources:
        path = manifest_paths.get(source)
        if path is None:
            path = args.manifest_root / source / str(args.manifest_name)
        records = read_jsonl(path)
        seen_usable = 0
        picked = 0
        for manifest_idx, record in enumerate(records):
            record = dict(record)
            record["source"] = source
            if bad_sample_key(record) in bad_keys:
                continue
            if seen_usable < source_offset:
                seen_usable += 1
                continue
            record["source_local_index"] = seen_usable
            record.setdefault("oracle_boundary", boundary)
            record.setdefault("shot_labels", list(default_shot_labels))
            record.setdefault(
                "pattern_id",
                f"{source}_{record.get('group', 'group')}_{record.get('start_frame', 'start')}_{seen_usable}",
            )
            record["source_manifest_index"] = manifest_idx
            selected.append(record)
            picked += 1
            seen_usable += 1
            if picked >= int(args.samples_per_source):
                break
        if picked < int(args.samples_per_source):
            raise RuntimeError(
                f"{source} only has {picked}/{int(args.samples_per_source)} usable held-out records "
                f"after source_offset={source_offset} in {path}"
            )
    return selected


class GeoTokenResidualMLP(nn.Module):
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
        # Start exactly at the geometry proposal; the residual learns only if
        # the supervised losses can justify moving away from T_geo.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.max_rot = np.deg2rad(float(max_rot_deg))
        self.max_trans = float(max_trans)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.net(features)
        gate = torch.sigmoid(raw[:, 6:7])
        delta_rotvec = float(self.max_rot) * torch.tanh(raw[:, :3]) * gate
        delta_trans = float(self.max_trans) * torch.tanh(raw[:, 3:6]) * gate
        return delta_rotvec, delta_trans, gate.squeeze(-1)


def make_variant_samples(samples, variant_dir: Path, aligned_name: str):
    variant_samples = []
    for sample in samples:
        variant_samples.append(
            SimpleNamespace(
                index=sample.index,
                source=sample.source,
                pattern_id=sample.pattern_id,
                record=sample.record,
                local_dir=sample.local_dir,
                aligned_dir=variant_dir
                / "samples"
                / f"{sample.index:02d}_{safe_name(sample.pattern_id)}"
                / aligned_name,
                pred_poses=sample.pred_poses,
                pred_joints=sample.pred_joints,
                target_poses=sample.target_poses,
                target_joints=sample.target_joints,
                bridge_debug=sample.bridge_debug,
            )
        )
    return variant_samples


def evaluate_variant(samples, aligned_poses, aligned_joints, debug, args, variant_dir: Path, aligned_name: str) -> dict:
    eval_args = SimpleNamespace(**vars(args))
    eval_args.output_dir = variant_dir
    variant_samples = make_variant_samples(samples, variant_dir, aligned_name)
    return evaluate_and_write_outputs(variant_samples, aligned_poses, aligned_joints, debug, eval_args)


def tensor_stack(samples, attr: str, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.stack([getattr(sample, attr) for sample in samples])).to(
        device=device,
        dtype=torch.float32,
    )


def compute_basic_losses(
    aligned_poses: torch.Tensor,
    aligned_joints: torch.Tensor,
    target_poses: torch.Tensor,
    target_joints: torch.Tensor,
    joint_ids: torch.Tensor,
    boundary: int,
) -> dict[str, torch.Tensor]:
    post = slice(boundary, aligned_poses.shape[1])
    human_loss = F.smooth_l1_loss(
        aligned_joints[:, post][:, :, joint_ids],
        target_joints[:, post][:, :, joint_ids],
        beta=0.05,
    )
    camera_t_loss = F.smooth_l1_loss(
        aligned_poses[:, post, :3, 3],
        target_poses[:, post, :3, 3],
        beta=0.05,
    )
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


def weighted_final_loss(losses: dict[str, torch.Tensor], args: argparse.Namespace) -> torch.Tensor:
    return (
        float(args.human_weight) * losses["human_loss"]
        + float(args.camera_t_weight) * losses["camera_t_loss"]
        + float(args.camera_r_weight) * losses["camera_r_loss"]
        + float(args.body_frame_weight) * losses["body_frame_loss"]
        + float(args.body_vector_weight) * losses["body_vector_loss"]
        + float(args.body_anchor_weight) * losses["body_anchor_loss"]
        + float(args.body_vertical_weight) * losses["body_vertical_loss"]
    )


def train_geo_token_residual(samples, args: argparse.Namespace, device: torch.device):
    pred_poses = tensor_stack(samples, "pred_poses", device)
    pred_joints = tensor_stack(samples, "pred_joints", device)
    target_poses = tensor_stack(samples, "target_poses", device)
    target_joints = tensor_stack(samples, "target_joints", device)
    joint_ids_np = np.asarray(sorted(set(STABLE_JOINTS + FOOT_JOINTS)), dtype=np.int64)
    joint_ids = torch.from_numpy(joint_ids_np).to(device=device, dtype=torch.long)
    boundary = int(args.boundary)
    post = slice(boundary, pred_poses.shape[1])

    R_geo, t_geo, geo_weights = compute_geo_proposal(pred_joints, boundary, joint_ids)
    geo_poses, geo_joints = apply_transform_batch(pred_poses, pred_joints, R_geo, t_geo, boundary)
    geo_losses = compute_basic_losses(geo_poses, geo_joints, target_poses, target_joints, joint_ids, boundary)

    geo_features = build_geo_residual_features(
        pred_joints,
        R_geo.detach(),
        t_geo.detach(),
        boundary,
        joint_ids,
    ).to(device)
    token_features_np = np.stack(
        [build_pair_feature(sample.token_features, str(args.token_feature_set), boundary) for sample in samples]
    ).astype(np.float32)
    token_features = torch.from_numpy(token_features_np).to(device=device, dtype=torch.float32)
    features = torch.cat([geo_features, token_features], dim=-1)

    aligner = GeoTokenResidualMLP(
        in_dim=features.shape[-1],
        hidden_dim=int(args.hidden_dim),
        max_rot_deg=float(args.residual_max_rot_deg),
        max_trans=float(args.residual_max_trans),
    ).to(device)
    if args.eval_checkpoint is not None:
        checkpoint = torch.load(args.eval_checkpoint, map_location=device)
        aligner.load_state_dict(checkpoint["model"], strict=True)
    optim = torch.optim.AdamW(aligner.parameters(), lr=float(args.lr), weight_decay=1e-4)

    R_target, t_target = target_transform_from_boundary_pose(pred_poses, target_poses, boundary)
    R_res_target, t_res_target = residual_target_from_geo(R_target, t_target, R_geo.detach(), t_geo.detach())
    geo_anchor_error = torch.linalg.norm(
        geo_joints[:, post][:, :, joint_ids] - target_joints[:, post][:, :, joint_ids],
        dim=-1,
    ).detach()

    history = []
    log_path = args.output_dir / "alignment_train_steps.jsonl"
    if log_path.exists() and args.overwrite:
        log_path.unlink()

    train_steps = 0 if bool(args.eval_only) else int(args.steps)
    for step in range(train_steps + 1):
        optim.zero_grad(set_to_none=True)
        delta_rotvec, delta_trans, gate = aligner(features)
        R_final, t_final, R_delta = compose_residual_with_geo(
            delta_rotvec,
            delta_trans,
            R_geo.detach(),
            t_geo.detach(),
        )
        aligned_poses, aligned_joints = apply_transform_batch(pred_poses, pred_joints, R_final, t_final, boundary)
        losses = compute_basic_losses(aligned_poses, aligned_joints, target_poses, target_joints, joint_ids, boundary)
        final_loss = weighted_final_loss(losses, args)
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
        if not bool(args.eval_only):
            loss.backward()
            optim.step()

        if step % int(args.log_every) == 0 or step == train_steps:
            row = {
                "step": int(step),
                "mode": "geo_token_residual_eval_only" if bool(args.eval_only) else "geo_token_residual",
                "token_feature_set": str(args.token_feature_set),
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
                "geo_loss": float(weighted_final_loss(geo_losses, args).detach().cpu()),
                "geo_human_loss": float(geo_losses["human_loss"].detach().cpu()),
                "geo_camera_t_loss": float(geo_losses["camera_t_loss"].detach().cpu()),
                "geo_camera_r_deg": float(torch.rad2deg(geo_losses["camera_r_loss"].detach()).cpu()),
                "geo_body_frame_deg": float(torch.rad2deg(geo_losses["body_frame_loss"].detach()).cpu()),
                "geo_body_vector_loss": float(geo_losses["body_vector_loss"].detach().cpu()),
                "geo_body_anchor_loss": float(geo_losses["body_anchor_loss"].detach().cpu()),
                "geo_body_vertical_loss": float(geo_losses["body_vertical_loss"].detach().cpu()),
            }
            print(json.dumps(row, sort_keys=True), flush=True)
            history.append(row)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")

    with torch.no_grad():
        delta_rotvec, delta_trans, gate = aligner(features)
        R_final, t_final, R_delta = compose_residual_with_geo(delta_rotvec, delta_trans, R_geo, t_geo)
        aligned_poses, aligned_joints = apply_transform_batch(pred_poses, pred_joints, R_final, t_final, boundary)

    eye = torch.eye(3, device=device, dtype=R_geo.dtype).unsqueeze(0).expand_as(R_geo)
    geo_debug = {
        "history": [
            {
                "step": 0,
                "mode": "geo_only",
                "loss": float(weighted_final_loss(geo_losses, args).detach().cpu()),
                "human_loss": float(geo_losses["human_loss"].detach().cpu()),
                "camera_t_loss": float(geo_losses["camera_t_loss"].detach().cpu()),
                "camera_r_deg": float(torch.rad2deg(geo_losses["camera_r_loss"].detach()).cpu()),
                "body_frame_deg": float(torch.rad2deg(geo_losses["body_frame_loss"].detach()).cpu()),
                "body_vector_loss": float(geo_losses["body_vector_loss"].detach().cpu()),
                "body_anchor_loss": float(geo_losses["body_anchor_loss"].detach().cpu()),
                "body_vertical_loss": float(geo_losses["body_vertical_loss"].detach().cpu()),
            }
        ],
        "joint_ids": joint_ids_np.astype(int).tolist(),
        "alignment_mode": "geo_only",
        "geo_weights": geo_weights.detach().cpu().numpy().astype(float).tolist(),
        "geo_rot_deg_norm": torch.rad2deg(rotation_geodesic(R_geo, eye)).detach().cpu().numpy().astype(float).tolist(),
        "geo_trans_norm": t_geo.norm(dim=-1).detach().cpu().numpy().astype(float).tolist(),
    }
    residual_debug = {
        "history": history,
        "joint_ids": joint_ids_np.astype(int).tolist(),
        "alignment_mode": "geo_token_residual",
        "eval_checkpoint": str(args.eval_checkpoint) if args.eval_checkpoint is not None else None,
        "eval_only": bool(args.eval_only),
        "token_feature_set": str(args.token_feature_set),
        "token_feature_keys": list(TOKEN_FEATURE_SETS[str(args.token_feature_set)]),
        "geo_feature_dim": int(geo_features.shape[-1]),
        "token_feature_dim": int(token_features.shape[-1]),
        "feature_dim": int(features.shape[-1]),
        "geo_weights": geo_weights.detach().cpu().numpy().astype(float).tolist(),
        "geo_rot_deg_norm": torch.rad2deg(rotation_geodesic(R_geo, eye)).detach().cpu().numpy().astype(float).tolist(),
        "geo_trans_norm": t_geo.norm(dim=-1).detach().cpu().numpy().astype(float).tolist(),
        "delta_rot_deg_norm": torch.rad2deg(delta_rotvec.norm(dim=-1)).detach().cpu().numpy().astype(float).tolist(),
        "delta_trans_norm": delta_trans.norm(dim=-1).detach().cpu().numpy().astype(float).tolist(),
        "gate": gate.detach().cpu().numpy().astype(float).tolist(),
        "residual_max_rot_deg": float(args.residual_max_rot_deg),
        "residual_max_trans": float(args.residual_max_trans),
    }
    torch.save(
        {
            "model": aligner.state_dict(),
            "args": vars(args),
            "features": features.detach().cpu(),
            "geo_features": geo_features.detach().cpu(),
            "token_features": token_features.detach().cpu(),
            "joint_ids": joint_ids.detach().cpu(),
            "geo_weights": geo_weights.detach().cpu(),
            "R_geo": R_geo.detach().cpu(),
            "t_geo": t_geo.detach().cpu(),
            "delta_rotvec": delta_rotvec.detach().cpu(),
            "delta_trans": delta_trans.detach().cpu(),
            "gate": gate.detach().cpu(),
            "samples": [
                {"source": sample.source, "pattern_id": sample.pattern_id, "record": sample.record}
                for sample in samples
            ],
        },
        args.output_dir / "alignment_head_geo_token_residual.pth",
    )
    return (
        geo_poses.detach().cpu().numpy().astype(np.float32),
        geo_joints.detach().cpu().numpy().astype(np.float32),
        geo_debug,
        aligned_poses.detach().cpu().numpy().astype(np.float32),
        aligned_joints.detach().cpu().numpy().astype(np.float32),
        residual_debug,
    )


def overall_metric_row(variant: str, summary: dict) -> dict:
    row = {"variant": variant}
    for metric, values in summary["aggregate"]["overall"].items():
        row[f"raw_{metric}"] = values["raw_mean"]
        row[f"aligned_{metric}"] = values["aligned_mean"]
        row[f"gain_{metric}"] = values["gain_mean"]
    return row


def write_comparison(output_dir: Path, summaries: dict[str, dict]) -> None:
    rows = [overall_metric_row(name, summary) for name, summary in summaries.items()]
    if not rows:
        return
    with (output_dir / "geo_token_residual_comparison.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# V10 Geometry + Token Residual Probe",
        "",
        "Lower is better. Gain = raw local-reset - aligned.",
        "",
        "| Variant | Cam Rot | Cam Trans | Human | Amean-B0 | Amean-B1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {variant} | {raw_cam_rot_deg:.2f}->{aligned_cam_rot_deg:.2f} ({gain_cam_rot_deg:+.2f}) | "
            "{raw_cam_trans_m:.3f}->{aligned_cam_trans_m:.3f} ({gain_cam_trans_m:+.3f}) | "
            "{raw_human_post_m:.3f}->{aligned_human_post_m:.3f} ({gain_human_post_m:+.3f}) | "
            "{raw_Amean_B0_m:.3f}->{aligned_Amean_B0_m:.3f} ({gain_Amean_B0_m:+.3f}) | "
            "{raw_Amean_B1_m:.3f}->{aligned_Amean_B1_m:.3f} ({gain_Amean_B1_m:+.3f}) |".format(**row)
        )
    (output_dir / "geo_token_residual_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.manual_seed(31)
    np.random.seed(31)
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
    print(f">> selected {len(records)} samples: {selected_text}", flush=True)
    samples = cache_samples(records, args, device)
    (
        geo_poses,
        geo_joints,
        geo_debug,
        residual_poses,
        residual_joints,
        residual_debug,
    ) = train_geo_token_residual(samples, args, device)

    summaries = {}
    geo_dir = args.output_dir / "variants" / "geo_only"
    summaries["geo_only"] = evaluate_variant(samples, geo_poses, geo_joints, geo_debug, args, geo_dir, "geo_aligned")
    residual_dir = args.output_dir / "variants" / f"geo_{args.token_feature_set}_residual"
    summaries[f"geo_{args.token_feature_set}_residual"] = evaluate_variant(
        samples,
        residual_poses,
        residual_joints,
        residual_debug,
        args,
        residual_dir,
        "geo_token_residual_aligned",
    )
    write_comparison(args.output_dir, summaries)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "comparison": str(args.output_dir / "geo_token_residual_comparison.md"),
                "aggregate": {name: summary["aggregate"]["overall"] for name, summary in summaries.items()},
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
