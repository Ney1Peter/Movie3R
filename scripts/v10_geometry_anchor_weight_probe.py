#!/usr/bin/env python3
"""V10 conservative geometry alignment with learned anchor weights.

This probe keeps explicit geometry as the main alignment mechanism.  The learnable
part does not regress a free SE(3) residual.  Instead it predicts a conservative
re-weighting of human anchors, then a differentiable weighted Procrustes solve
produces the final segment-to-global transform.
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
from v10_geometry_token_residual_probe import (
    compute_basic_losses,
    evaluate_variant,
    select_records,
    tensor_stack,
    weighted_final_loss,
)
from v10_static_alignment_4source_probe import (
    build_geo_residual_features,
    compute_geo_proposal,
    fixed_geo_anchor_weights,
    solve_weighted_rigid_transform_batch,
)
from v10_token_alignment_4source_probe import TOKEN_FEATURE_SETS, build_pair_feature, cache_samples
from v9_learned_stream_alignment_4source_probe import (
    DEFAULT_BAD_SAMPLE_REGISTRY,
    SOURCE_ORDER,
    apply_transform_batch,
)
from v9_learned_stream_alignment_overfit import rotation_geodesic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=REPO_ROOT / "output" / "v10_geometry_anchor_weight_probe" / "4source_s2_human_token",
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
    parser.add_argument("--source_offset", type=int, default=0)
    parser.add_argument("--boundary", type=int, default=2)
    parser.add_argument("--resolution", type=int, nargs=2, default=(512, 288), metavar=("W", "H"))
    parser.add_argument("--resize_mode", default="human3r_demo")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--token_feature_set", choices=list(TOKEN_FEATURE_SETS), default="human_only")
    parser.add_argument("--max_logit_delta", type=float, default=2.0)
    parser.add_argument("--gate_init_bias", type=float, default=-2.0)
    parser.add_argument(
        "--disable_weight_gate",
        action="store_true",
        help="Use bounded anchor weight deltas directly instead of mixing them with base weights through a gate.",
    )
    parser.add_argument("--human_weight", type=float, default=10.0)
    parser.add_argument("--camera_t_weight", type=float, default=2.0)
    parser.add_argument("--camera_r_weight", type=float, default=1.0)
    parser.add_argument("--body_frame_weight", type=float, default=1.0)
    parser.add_argument("--body_vector_weight", type=float, default=1.0)
    parser.add_argument("--body_anchor_weight", type=float, default=5.0)
    parser.add_argument("--body_vertical_weight", type=float, default=5.0)
    parser.add_argument("--proposal_improvement_weight", type=float, default=10.0)
    parser.add_argument("--weight_prior_weight", type=float, default=1.0)
    parser.add_argument("--gate_prior_weight", type=float, default=0.05)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--skip_inference", action="store_true")
    parser.add_argument("--skip_bad_samples", action="store_true")
    parser.add_argument("--eval_checkpoint", type=Path, default=None)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


class AnchorWeightHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_anchors: int,
        max_logit_delta: float,
        gate_init_bias: float,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_anchors + 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            self.net[-1].bias[num_anchors] = float(gate_init_bias)
        self.num_anchors = int(num_anchors)
        self.max_logit_delta = float(max_logit_delta)

    def forward(
        self,
        features: torch.Tensor,
        base_weights: torch.Tensor,
        use_gate: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.net(features)
        delta_logits = self.max_logit_delta * torch.tanh(raw[:, : self.num_anchors])
        gate = torch.sigmoid(raw[:, self.num_anchors : self.num_anchors + 1])
        base = base_weights.to(device=features.device, dtype=features.dtype)
        base = base / base.sum().clamp_min(1e-8)
        learned = torch.softmax(torch.log(base.clamp_min(1e-8))[None, :] + delta_logits, dim=-1)
        if use_gate:
            weights = (1.0 - gate) * base[None, :] + gate * learned
        else:
            weights = learned
            gate = torch.ones_like(gate)
        return weights, gate.squeeze(-1)


def compare_summary_row(variant: str, summary: dict) -> dict:
    row = {"variant": variant}
    for metric, values in summary["aggregate"]["overall"].items():
        row[f"raw_{metric}"] = values["raw_mean"]
        row[f"aligned_{metric}"] = values["aligned_mean"]
        row[f"gain_{metric}"] = values["gain_mean"]
    return row


def write_comparison(output_dir: Path, summaries: dict[str, dict]) -> None:
    rows = [compare_summary_row(name, summary) for name, summary in summaries.items()]
    if not rows:
        return
    with (output_dir / "anchor_weight_comparison.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# V10 Learned Anchor Weight Probe",
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
    (output_dir / "anchor_weight_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_anchor_weight_alignment(samples, args: argparse.Namespace, device: torch.device):
    pred_poses = tensor_stack(samples, "pred_poses", device)
    pred_joints = tensor_stack(samples, "pred_joints", device)
    target_poses = tensor_stack(samples, "target_poses", device)
    target_joints = tensor_stack(samples, "target_joints", device)
    joint_ids_np = np.asarray(sorted(set(STABLE_JOINTS + FOOT_JOINTS)), dtype=np.int64)
    joint_ids = torch.from_numpy(joint_ids_np).to(device=device, dtype=torch.long)
    boundary = int(args.boundary)
    post = slice(boundary, pred_poses.shape[1])

    R_fixed, t_fixed, base_weights = compute_geo_proposal(pred_joints, boundary, joint_ids)
    fixed_poses, fixed_joints = apply_transform_batch(pred_poses, pred_joints, R_fixed, t_fixed, boundary)
    fixed_losses = compute_basic_losses(fixed_poses, fixed_joints, target_poses, target_joints, joint_ids, boundary)
    fixed_anchor_error = torch.linalg.norm(
        fixed_joints[:, post][:, :, joint_ids] - target_joints[:, post][:, :, joint_ids],
        dim=-1,
    ).detach()

    geo_features = build_geo_residual_features(pred_joints, R_fixed.detach(), t_fixed.detach(), boundary, joint_ids)
    token_features_np = np.stack(
        [build_pair_feature(sample.token_features, str(args.token_feature_set), boundary) for sample in samples]
    ).astype(np.float32)
    token_features = torch.from_numpy(token_features_np).to(device=device, dtype=torch.float32)
    features = torch.cat([geo_features.to(device), token_features], dim=-1)

    head = AnchorWeightHead(
        in_dim=features.shape[-1],
        hidden_dim=int(args.hidden_dim),
        num_anchors=int(joint_ids.numel()),
        max_logit_delta=float(args.max_logit_delta),
        gate_init_bias=float(args.gate_init_bias),
    ).to(device)
    if args.eval_checkpoint is not None:
        checkpoint = torch.load(args.eval_checkpoint, map_location=device)
        head.load_state_dict(checkpoint["model"], strict=True)
    optim = torch.optim.AdamW(head.parameters(), lr=float(args.lr), weight_decay=1e-4)
    hist = pred_joints[:, :boundary, joint_ids].mean(dim=1)
    cur = pred_joints[:, boundary, joint_ids]
    base_norm = base_weights.to(device=device, dtype=torch.float32)
    base_norm = base_norm / base_norm.sum().clamp_min(1e-8)

    history = []
    log_path = args.output_dir / "anchor_weight_train_steps.jsonl"
    if log_path.exists() and args.overwrite:
        log_path.unlink()

    train_steps = 0 if bool(args.eval_only) else int(args.steps)
    for step in range(train_steps + 1):
        optim.zero_grad(set_to_none=True)
        learned_weights, gate = head(features, base_weights, use_gate=not bool(args.disable_weight_gate))
        R_learned, t_learned = solve_weighted_rigid_transform_batch(cur, hist, learned_weights)
        aligned_poses, aligned_joints = apply_transform_batch(
            pred_poses,
            pred_joints,
            R_learned,
            t_learned,
            boundary,
        )
        losses = compute_basic_losses(aligned_poses, aligned_joints, target_poses, target_joints, joint_ids, boundary)
        final_loss = weighted_final_loss(losses, args)
        final_anchor_error = torch.linalg.norm(
            aligned_joints[:, post][:, :, joint_ids] - target_joints[:, post][:, :, joint_ids],
            dim=-1,
        )
        proposal_improvement_loss = F.relu(final_anchor_error - fixed_anchor_error).mean()
        weight_prior_loss = F.mse_loss(learned_weights, base_norm[None, :].expand_as(learned_weights))
        gate_prior_loss = gate.mean() if not bool(args.disable_weight_gate) else gate.new_tensor(0.0)
        loss = (
            final_loss
            + float(args.proposal_improvement_weight) * proposal_improvement_loss
            + float(args.weight_prior_weight) * weight_prior_loss
            + float(args.gate_prior_weight) * gate_prior_loss
        )
        if not bool(args.eval_only):
            loss.backward()
            optim.step()

        if step % int(args.log_every) == 0 or step == train_steps:
            eye = torch.eye(3, device=device, dtype=R_learned.dtype).unsqueeze(0).expand_as(R_learned)
            row = {
                "step": int(step),
                "mode": "anchor_weight_eval_only" if bool(args.eval_only) else "anchor_weight_train",
                "loss": float(loss.detach().cpu()),
                "final_loss": float(final_loss.detach().cpu()),
                "human_loss": float(losses["human_loss"].detach().cpu()),
                "camera_t_loss": float(losses["camera_t_loss"].detach().cpu()),
                "camera_r_deg": float(torch.rad2deg(losses["camera_r_loss"].detach()).cpu()),
                "proposal_improvement_loss": float(proposal_improvement_loss.detach().cpu()),
                "weight_prior_loss": float(weight_prior_loss.detach().cpu()),
                "gate_mean": float(gate.detach().mean().cpu()),
                "learned_geo_rot_deg_mean": float(torch.rad2deg(rotation_geodesic(R_learned, eye)).mean().cpu()),
                "learned_geo_trans_norm_mean": float(t_learned.norm(dim=-1).mean().detach().cpu()),
                "fixed_loss": float(weighted_final_loss(fixed_losses, args).detach().cpu()),
                "fixed_human_loss": float(fixed_losses["human_loss"].detach().cpu()),
                "fixed_camera_t_loss": float(fixed_losses["camera_t_loss"].detach().cpu()),
                "fixed_camera_r_deg": float(torch.rad2deg(fixed_losses["camera_r_loss"].detach()).cpu()),
            }
            print(json.dumps(row, sort_keys=True), flush=True)
            history.append(row)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")

    with torch.no_grad():
        learned_weights, gate = head(features, base_weights, use_gate=not bool(args.disable_weight_gate))
        R_learned, t_learned = solve_weighted_rigid_transform_batch(cur, hist, learned_weights)
        aligned_poses, aligned_joints = apply_transform_batch(pred_poses, pred_joints, R_learned, t_learned, boundary)

    eye = torch.eye(3, device=device, dtype=R_fixed.dtype).unsqueeze(0).expand_as(R_fixed)
    fixed_debug = {
        "history": [
            {
                "step": 0,
                "mode": "fixed_geo",
                "loss": float(weighted_final_loss(fixed_losses, args).detach().cpu()),
                "human_loss": float(fixed_losses["human_loss"].detach().cpu()),
                "camera_t_loss": float(fixed_losses["camera_t_loss"].detach().cpu()),
                "camera_r_deg": float(torch.rad2deg(fixed_losses["camera_r_loss"].detach()).cpu()),
            }
        ],
        "joint_ids": joint_ids_np.astype(int).tolist(),
        "alignment_mode": "fixed_geo",
        "geo_weights": base_weights.detach().cpu().numpy().astype(float).tolist(),
        "geo_rot_deg_norm": torch.rad2deg(rotation_geodesic(R_fixed, eye)).detach().cpu().numpy().astype(float).tolist(),
        "geo_trans_norm": t_fixed.norm(dim=-1).detach().cpu().numpy().astype(float).tolist(),
    }
    learned_debug = {
        "history": history,
        "joint_ids": joint_ids_np.astype(int).tolist(),
        "alignment_mode": "learned_anchor_weight_geo",
        "eval_checkpoint": str(args.eval_checkpoint) if args.eval_checkpoint is not None else None,
        "eval_only": bool(args.eval_only),
        "token_feature_set": str(args.token_feature_set),
        "token_feature_keys": list(TOKEN_FEATURE_SETS[str(args.token_feature_set)]),
        "feature_dim": int(features.shape[-1]),
        "base_weights": base_norm.detach().cpu().numpy().astype(float).tolist(),
        "learned_weights": learned_weights.detach().cpu().numpy().astype(float).tolist(),
        "gate": gate.detach().cpu().numpy().astype(float).tolist(),
        "max_logit_delta": float(args.max_logit_delta),
        "disable_weight_gate": bool(args.disable_weight_gate),
    }
    torch.save(
        {
            "model": head.state_dict(),
            "args": vars(args),
            "features": features.detach().cpu(),
            "token_features": token_features.detach().cpu(),
            "joint_ids": joint_ids.detach().cpu(),
            "base_weights": base_norm.detach().cpu(),
            "learned_weights": learned_weights.detach().cpu(),
            "gate": gate.detach().cpu(),
            "R_fixed": R_fixed.detach().cpu(),
            "t_fixed": t_fixed.detach().cpu(),
            "R_learned": R_learned.detach().cpu(),
            "t_learned": t_learned.detach().cpu(),
            "samples": [
                {"source": sample.source, "pattern_id": sample.pattern_id, "record": sample.record}
                for sample in samples
            ],
        },
        args.output_dir / "anchor_weight_head_4source_probe.pth",
    )
    return (
        fixed_poses.detach().cpu().numpy().astype(np.float32),
        fixed_joints.detach().cpu().numpy().astype(np.float32),
        fixed_debug,
        aligned_poses.detach().cpu().numpy().astype(np.float32),
        aligned_joints.detach().cpu().numpy().astype(np.float32),
        learned_debug,
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(37)
    np.random.seed(37)
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
        fixed_poses,
        fixed_joints,
        fixed_debug,
        learned_poses,
        learned_joints,
        learned_debug,
    ) = train_anchor_weight_alignment(samples, args, device)

    summaries = {}
    fixed_dir = args.output_dir / "variants" / "fixed_geo"
    summaries["fixed_geo"] = evaluate_variant(samples, fixed_poses, fixed_joints, fixed_debug, args, fixed_dir, "fixed_geo")
    learned_dir = args.output_dir / "variants" / f"learned_anchor_weight_{args.token_feature_set}"
    summaries[f"learned_anchor_weight_{args.token_feature_set}"] = evaluate_variant(
        samples,
        learned_poses,
        learned_joints,
        learned_debug,
        args,
        learned_dir,
        "learned_anchor_weight_geo",
    )
    write_comparison(args.output_dir, summaries)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "comparison": str(args.output_dir / "anchor_weight_comparison.md"),
                "aggregate": {name: summary["aggregate"]["overall"] for name, summary in summaries.items()},
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
