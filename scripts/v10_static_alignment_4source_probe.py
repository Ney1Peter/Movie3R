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
    parser.add_argument("--manifest_name", default="train_aabb.jsonl")
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
    parser.add_argument("--human_weight", type=float, default=5.0)
    parser.add_argument("--camera_t_weight", type=float, default=2.0)
    parser.add_argument("--camera_r_weight", type=float, default=1.0)
    parser.add_argument("--body_frame_weight", type=float, default=0.0)
    parser.add_argument("--body_vector_weight", type=float, default=0.0)
    parser.add_argument("--prior_weight", type=float, default=1e-4)
    parser.add_argument("--use_body_frame_input", action="store_true")
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--skip_inference", action="store_true")
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
        prior_loss = rotvec.pow(2).mean() + trans.pow(2).mean()
        loss = (
            float(args.human_weight) * human_loss
            + float(args.camera_t_weight) * camera_t_loss
            + float(args.camera_r_weight) * camera_r_loss
            + float(args.body_frame_weight) * body_frame_loss
            + float(args.body_vector_weight) * body_vector_loss
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
        "learned_rotvec_deg_norm": torch.rad2deg(rotvec.norm(dim=-1)).detach().cpu().numpy().astype(float).tolist(),
        "learned_trans_norm": trans.norm(dim=-1).detach().cpu().numpy().astype(float).tolist(),
        "use_body_frame_input": bool(args.use_body_frame_input),
        "body_frame_weight": float(args.body_frame_weight),
        "body_vector_weight": float(args.body_vector_weight),
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
