#!/usr/bin/env python3
"""Overfit a learnable streaming segment alignment module on one AABB clip.

This is a probe for the post-Human3R streaming design:

1. run strict original Human3R once on a 4-frame AABB clip;
2. reset Human3R local recurrent state after the A segment;
3. train a tiny MLP that sees only predicted human anchors from history A and
   current B1, then predicts one SE(3) transform for the whole B segment;
4. cache that transform for B2 and save a Human3R-compatible aligned output.

There is no hand-written yaw/translation rule here.  GT camera/SMPL is used only
as the overfit supervision target.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
ARCHIVE_V7 = REPO_ROOT / "scripts" / "archive_v7"
for path in (str(REPO_ROOT), str(SRC_ROOT), str(ARCHIVE_V7)):
    if path not in sys.path:
        sys.path.insert(0, path)

from add_ckpt_path import add_path_to_dust3r
from demo import prepare_output
from dust3r.datasets.avatarrex import AvatarReX_AABB
from dust3r.inference import inference_recurrent_lighter
from dust3r.model import ARCroco3DStereo
from dust3r.smpl_model import SMPLModel
from dust3r.utils.device import to_cpu, todevice
from overfit_human_anchor_pose_correction import FOOT_JOINTS, STABLE_JOINTS, load_sequence
from v9_online_stream_human3r_segment_align import strict_original_model
from v9_segment_human3r_yaw_align_probe import copy_np_payload, copy_smpl, save_camera, transform_pose


DEFAULT_RAW_ROOTS = {
    "lbn1": "/data/wangzheng/iJCV-CODE/data/AvatarReX_raw_meta/lbn1",
    "lbn2": "/data/wangzheng/iJCV-CODE/data/AvatarReX_raw_meta/lbn2",
    "zzr": "/data/wangzheng/iJCV-CODE/data/AvatarReX_raw_meta/zzr",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "output" / "v9_learned_stream_alignment_overfit" / "lbn1_1192")
    parser.add_argument("--model_path", type=Path, default=REPO_ROOT / "src" / "human3r_896L.pth")
    parser.add_argument("--data_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data"))
    parser.add_argument("--split", default="Training")
    parser.add_argument("--seq_a", default="lbn1/22053926")
    parser.add_argument("--seq_b", default="lbn1/22010716")
    parser.add_argument("--start_frame", type=int, default=1192)
    parser.add_argument("--boundary", type=int, default=2, help="First frame index of the new B segment.")
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
    parser.add_argument("--prior_weight", type=float, default=1e-4)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--skip_inference", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


class StreamingAlignmentMLP(nn.Module):
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
        return rotvec, trans


def skew(v: torch.Tensor) -> torch.Tensor:
    x, y, z = v.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        [zero, -z, y, z, zero, -x, -y, x, zero],
        dim=-1,
    ).reshape(-1, 3, 3)


def so3_exp(rotvec: torch.Tensor) -> torch.Tensor:
    theta = rotvec.norm(dim=-1, keepdim=True)
    axis = rotvec / theta.clamp_min(1e-7)
    K = skew(axis)
    eye = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype).expand(rotvec.shape[0], 3, 3)
    sin_t = torch.sin(theta).reshape(-1, 1, 1)
    cos_t = torch.cos(theta).reshape(-1, 1, 1)
    R = eye + sin_t * K + (1.0 - cos_t) * (K @ K)
    R_small = eye + skew(rotvec)
    return torch.where((theta < 1e-6).reshape(-1, 1, 1), R_small, R)


def rotation_geodesic(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    rel = R_pred.transpose(-1, -2) @ R_gt
    cos = ((rel.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cos)


def output_complete(path: Path) -> bool:
    return (path / "camera" / "000000.npz").is_file() and (path / "smpl" / "000003.npz").is_file()


def load_aabb_views(args: argparse.Namespace, device: torch.device) -> list[dict]:
    raw_roots = {k: str(v) for k, v in DEFAULT_RAW_ROOTS.items()}
    dataset = AvatarReX_AABB(
        split=args.split,
        ROOT=str(args.data_root),
        resolution=tuple(args.resolution),
        num_views=4,
        aug_crop=0,
        allow_repeat=True,
        seed=401,
        n_corres=0,
        fixed_samples=[(args.seq_a, args.seq_b, int(args.start_frame))],
        load_da3_depth=False,
        raw_calibration_root=raw_roots,
        resize_mode=str(args.resize_mode),
        max_humans=1,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    views = next(iter(loader))
    for view in views:
        view["img_mask"] = torch.ones_like(view["img_mask"], dtype=torch.bool)
        view["ray_mask"] = torch.zeros_like(view["ray_mask"], dtype=torch.bool)
        view["update"] = torch.ones_like(view["img_mask"], dtype=torch.bool)
        view["update_state"] = torch.ones_like(view["img_mask"], dtype=torch.bool)
        view["update_mem"] = torch.ones_like(view["img_mask"], dtype=torch.bool)
        view["update_v8_history"] = torch.ones_like(view["img_mask"], dtype=torch.bool)
        view["reset"] = torch.zeros_like(view["img_mask"], dtype=torch.bool)
    views[int(args.boundary) - 1]["reset"] = torch.ones_like(views[int(args.boundary) - 1]["img_mask"], dtype=torch.bool)
    views = todevice(views, device)
    return views


def attach_gt_smpl(views: list[dict], model: ARCroco3DStereo, device: torch.device) -> None:
    smpl_model = SMPLModel(
        device,
        model_args={
            "patch_size": model.croco_args["patch_size"],
            "mhmr_img_res": model.mhmr_img_res,
            "bb_patch_size": model.bb_patch_size,
        },
    )
    smpl_model.update_smpl_gt(views)


def run_local_reset_human3r(args: argparse.Namespace, views: list[dict], local_dir: Path, device: torch.device) -> None:
    if output_complete(local_dir) and not args.overwrite:
        return
    if local_dir.exists() and args.overwrite:
        shutil.rmtree(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    add_path_to_dust3r(str(args.model_path))
    model = ARCroco3DStereo.from_pretrained(str(args.model_path)).to(device).eval()
    strict_original_model(model)
    img_res = getattr(model, "mhmr_img_res", None)

    with torch.no_grad():
        outputs, _ = inference_recurrent_lighter(views, model, str(device), use_ttt3r=False)

    outputs_cpu = to_cpu(outputs)
    outputs_to_save = {"pred": outputs_cpu["pred"], "views": [dict(v) for v in outputs_cpu["views"]]}
    for view in outputs_to_save["views"]:
        view["reset"] = torch.zeros_like(view["reset"], dtype=torch.bool)
    prepare_output(
        outputs_to_save,
        str(local_dir),
        revisit=1,
        use_pose=True,
        save=True,
        render=False,
        render_video=False,
        img_res=img_res,
        subsample=1,
    )


def gt_pose_from_view(view: dict) -> torch.Tensor:
    pose = view.get("raw_camera_pose", None)
    if pose is None:
        pose = view["camera_pose"]
    return pose.detach().float()[0]


def solve_rigid_transform(src: torch.Tensor, dst: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return R,t with R @ src + t ~= dst, no scale."""
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


def extract_gt_world(
    views: list[dict],
    pred_poses: np.ndarray,
    pred_joints: np.ndarray,
    boundary: int,
    joint_ids: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict]:
    gt_poses = torch.stack([gt_pose_from_view(view) for view in views], dim=0).to(device)
    gt_joints_cam = torch.stack([view["smpl_j3d"].detach().float()[0, 0] for view in views], dim=0).to(device)
    R = gt_poses[:, :3, :3]
    t = gt_poses[:, :3, 3]
    gt_joints_world = torch.einsum("nij,nkj->nki", R, gt_joints_cam) + t[:, None, :]

    pred_joints_t = torch.from_numpy(pred_joints).to(device=device, dtype=torch.float32)
    joint_ids_t = torch.from_numpy(joint_ids).to(device=device, dtype=torch.long)
    src = gt_joints_world[:boundary, joint_ids_t].reshape(-1, 3)
    dst = pred_joints_t[:boundary, joint_ids_t].reshape(-1, 3)
    bridge_R, bridge_t = solve_rigid_transform(src, dst)

    target_poses = gt_poses.clone()
    target_poses[:, :3, :3] = torch.einsum("ij,njk->nik", bridge_R, gt_poses[:, :3, :3])
    target_poses[:, :3, 3] = torch.einsum("ij,nj->ni", bridge_R, gt_poses[:, :3, 3]) + bridge_t[None]
    target_joints = torch.einsum("ij,nkj->nki", bridge_R, gt_joints_world) + bridge_t[None, None, :]
    bridge_debug = {
        "bridge": "GT world to Human3R A-segment gauge estimated from A-segment human anchors only",
        "bridge_R": bridge_R.detach().cpu().numpy().astype(np.float32).tolist(),
        "bridge_t": bridge_t.detach().cpu().numpy().astype(np.float32).tolist(),
    }
    return (
        target_poses.detach().cpu().numpy().astype(np.float32),
        target_joints.detach().cpu().numpy().astype(np.float32),
        bridge_debug,
    )


def build_feature(pred_joints: torch.Tensor, boundary: int, joint_ids: torch.Tensor) -> torch.Tensor:
    hist = pred_joints[:boundary, joint_ids].mean(dim=0)
    cur = pred_joints[boundary, joint_ids]
    hist_center = hist.mean(dim=0, keepdim=True)
    cur_center = cur.mean(dim=0, keepdim=True)
    hist_shape = hist - hist_center
    cur_shape = cur - cur_center
    feature = torch.cat(
        [
            hist.flatten(),
            cur.flatten(),
            hist_shape.flatten(),
            cur_shape.flatten(),
            (cur - hist).flatten(),
            hist_center.flatten(),
            cur_center.flatten(),
            (cur_center - hist_center).flatten(),
        ],
        dim=0,
    )
    return feature.unsqueeze(0)


def apply_transform_to_post(
    pred_poses: torch.Tensor,
    pred_joints: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    boundary: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    out_poses = pred_poses.clone()
    out_joints = pred_joints.clone()
    post = slice(boundary, pred_poses.shape[0])
    out_joints[post] = torch.einsum("ij,nkj->nki", R, pred_joints[post]) + t[None, None, :]
    out_poses[post, :3, :3] = torch.einsum("ij,njk->nik", R, pred_poses[post, :3, :3])
    out_poses[post, :3, 3] = torch.einsum("ij,nj->ni", R, pred_poses[post, :3, 3]) + t[None, :]
    return out_poses, out_joints


def camera_metrics(poses: np.ndarray, target_poses: np.ndarray, frame_ids: list[int]) -> dict:
    out = {}
    for idx in frame_ids:
        t_err = float(np.linalg.norm(poses[idx, :3, 3] - target_poses[idx, :3, 3]))
        rel = poses[idx, :3, :3].T @ target_poses[idx, :3, :3]
        cos = float(np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0))
        r_err = float(math.degrees(math.acos(cos)))
        out[f"f{idx}_t_m"] = t_err
        out[f"f{idx}_r_deg"] = r_err
    out["mean_t_m"] = float(np.mean([out[f"f{idx}_t_m"] for idx in frame_ids]))
    out["mean_r_deg"] = float(np.mean([out[f"f{idx}_r_deg"] for idx in frame_ids]))
    return out


def human_metrics(joints: np.ndarray, target_joints: np.ndarray, frame_ids: list[int], joint_ids: np.ndarray) -> dict:
    vals = {}
    for idx in frame_ids:
        vals[f"f{idx}_m"] = float(np.linalg.norm(joints[idx, joint_ids] - target_joints[idx, joint_ids], axis=-1).mean())
    vals["mean_m"] = float(np.mean([vals[f"f{idx}_m"] for idx in frame_ids]))
    return vals


def train_alignment(
    local_dir: Path,
    target_poses: np.ndarray,
    target_joints: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict]:
    data = load_sequence(local_dir, 4, device)
    pred_poses = torch.from_numpy(data.poses).to(device=device, dtype=torch.float32)
    pred_joints = torch.from_numpy(data.joints_world).to(device=device, dtype=torch.float32)
    tgt_poses = torch.from_numpy(target_poses).to(device=device, dtype=torch.float32)
    tgt_joints = torch.from_numpy(target_joints).to(device=device, dtype=torch.float32)

    joint_ids_np = np.asarray(sorted(set(STABLE_JOINTS + FOOT_JOINTS)), dtype=np.int64)
    joint_ids = torch.from_numpy(joint_ids_np).to(device=device, dtype=torch.long)
    feature = build_feature(pred_joints, int(args.boundary), joint_ids)

    model = StreamingAlignmentMLP(
        in_dim=feature.shape[-1],
        hidden_dim=int(args.hidden_dim),
        max_rot_deg=float(args.max_rot_deg),
        max_trans=float(args.max_trans),
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)

    post_ids = torch.arange(int(args.boundary), pred_poses.shape[0], device=device)
    history = []
    log_path = args.output_dir / "alignment_train_steps.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists() and args.overwrite:
        log_path.unlink()

    for step in range(int(args.steps) + 1):
        optim.zero_grad(set_to_none=True)
        rotvec, trans = model(feature)
        R = so3_exp(rotvec)[0]
        t = trans[0]
        aligned_poses, aligned_joints = apply_transform_to_post(pred_poses, pred_joints, R, t, int(args.boundary))

        human_loss = F.smooth_l1_loss(
            aligned_joints[post_ids[:, None], joint_ids[None]],
            tgt_joints[post_ids[:, None], joint_ids[None]],
            beta=0.05,
        )
        camera_t_loss = F.smooth_l1_loss(aligned_poses[post_ids, :3, 3], tgt_poses[post_ids, :3, 3], beta=0.05)
        camera_r_loss = rotation_geodesic(aligned_poses[post_ids, :3, :3], tgt_poses[post_ids, :3, :3]).mean()
        prior_loss = rotvec.pow(2).mean() + trans.pow(2).mean()
        loss = (
            float(args.human_weight) * human_loss
            + float(args.camera_t_weight) * camera_t_loss
            + float(args.camera_r_weight) * camera_r_loss
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
                "rotvec_deg": float(torch.rad2deg(rotvec.norm(dim=-1).detach()).cpu()[0]),
                "trans_norm": float(trans.norm(dim=-1).detach().cpu()[0]),
            }
            print(json.dumps(row, sort_keys=True))
            history.append(row)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")

    with torch.no_grad():
        rotvec, trans = model(feature)
        R = so3_exp(rotvec)[0]
        t = trans[0]
        aligned_poses, aligned_joints = apply_transform_to_post(pred_poses, pred_joints, R, t, int(args.boundary))
    torch.save(
        {
            "model": model.state_dict(),
            "feature": feature.detach().cpu(),
            "joint_ids": joint_ids.detach().cpu(),
            "rotvec": rotvec.detach().cpu(),
            "trans": trans.detach().cpu(),
            "args": vars(args),
        },
        args.output_dir / "alignment_head_overfit.pth",
    )
    debug = {
        "history": history,
        "joint_ids": joint_ids_np.astype(int).tolist(),
        "learned_R": R.detach().cpu().numpy().astype(np.float32).tolist(),
        "learned_t": t.detach().cpu().numpy().astype(np.float32).tolist(),
        "learned_rotvec_deg_norm": float(torch.rad2deg(rotvec.norm(dim=-1)).detach().cpu()[0]),
        "learned_trans_norm": float(trans.norm(dim=-1).detach().cpu()[0]),
    }
    return aligned_poses.detach().cpu().numpy().astype(np.float32), aligned_joints.detach().cpu().numpy().astype(np.float32), debug


def write_aligned_output(local_dir: Path, aligned_dir: Path, aligned_poses: np.ndarray, boundary: int, overwrite: bool) -> None:
    if aligned_dir.exists() and overwrite:
        shutil.rmtree(aligned_dir)
    for sub in ["camera", "color", "conf", "depth", "smpl"]:
        (aligned_dir / sub).mkdir(parents=True, exist_ok=True)

    # Extract the learned transform from one post-boundary camera pair.
    src_pose = np.load(local_dir / "camera" / f"{boundary:06d}.npz")["pose"].astype(np.float64)
    dst_pose = aligned_poses[boundary].astype(np.float64)
    R = dst_pose[:3, :3] @ src_pose[:3, :3].T
    t = dst_pose[:3, 3] - R @ src_pose[:3, 3]

    for idx in range(4):
        for sub, ext in [("color", ".png"), ("conf", ".npy"), ("depth", ".npy")]:
            copy_np_payload(local_dir / sub / f"{idx:06d}{ext}", aligned_dir / sub / f"{idx:06d}{ext}")
        src_camera = local_dir / "camera" / f"{idx:06d}.npz"
        save_camera(src_camera, aligned_dir / "camera" / f"{idx:06d}.npz", aligned_poses[idx])
        copy_smpl(
            local_dir / "smpl" / f"{idx:06d}.npz",
            aligned_dir / "smpl" / f"{idx:06d}.npz",
            R if idx >= boundary else None,
            t if idx >= boundary else None,
        )


def main() -> None:
    args = parse_args()
    torch.manual_seed(13)
    np.random.seed(13)
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    add_path_to_dust3r(str(args.model_path))
    model_for_gt = ARCroco3DStereo.from_pretrained(str(args.model_path)).to(device).eval()
    strict_original_model(model_for_gt)
    views = load_aabb_views(args, device)
    attach_gt_smpl(views, model_for_gt, device)
    del model_for_gt
    if device.type == "cuda":
        torch.cuda.empty_cache()

    local_dir = args.output_dir / "original_human3r_local_reset"
    if not args.skip_inference:
        run_local_reset_human3r(args, views, local_dir, device)
    if not output_complete(local_dir):
        raise FileNotFoundError(f"Local Human3R output is incomplete: {local_dir}")

    pred_data = load_sequence(local_dir, 4, device)
    bridge_joint_ids = np.asarray(sorted(set(STABLE_JOINTS + FOOT_JOINTS)), dtype=np.int64)
    target_poses, target_joints, bridge_debug = extract_gt_world(
        views,
        pred_data.poses,
        pred_data.joints_world,
        int(args.boundary),
        bridge_joint_ids,
        device,
    )
    aligned_poses, aligned_joints, debug = train_alignment(local_dir, target_poses, target_joints, args, device)

    joint_ids = np.asarray(debug["joint_ids"], dtype=np.int64)
    raw_metrics = {
        "camera_post": camera_metrics(pred_data.poses, target_poses, list(range(args.boundary, 4))),
        "human_post": human_metrics(pred_data.joints_world, target_joints, list(range(args.boundary, 4)), joint_ids),
        "human_AA": human_metrics(pred_data.joints_world, target_joints, list(range(0, args.boundary)), joint_ids),
    }
    aligned_metrics = {
        "camera_post": camera_metrics(aligned_poses, target_poses, list(range(args.boundary, 4))),
        "human_post": human_metrics(aligned_joints, target_joints, list(range(args.boundary, 4)), joint_ids),
        "human_AA": human_metrics(aligned_joints, target_joints, list(range(0, args.boundary)), joint_ids),
    }

    aligned_dir = args.output_dir / "learned_stream_aligned"
    write_aligned_output(local_dir, aligned_dir, aligned_poses, int(args.boundary), bool(args.overwrite))

    summary = {
        "method": "learned post-Human3R streaming segment alignment; oracle boundary; no hand-written yaw/translation rule",
        "clip": {
            "split": args.split,
            "seq_a": args.seq_a,
            "seq_b": args.seq_b,
            "start_frame": int(args.start_frame),
            "boundary": int(args.boundary),
        },
        "strict_original_human3r": True,
        "streaming_semantics": {
            "local_state_reset_after_frame": int(args.boundary - 1),
            "alignment_head_runs_on_frame": int(args.boundary),
            "later_segment_frames_use_cached_transform": True,
            "uses_future_frames_as_input": False,
        },
        "outputs": {
            "local_reset": str(local_dir),
            "learned_aligned": str(aligned_dir),
            "checkpoint": str(args.output_dir / "alignment_head_overfit.pth"),
            "train_log": str(args.output_dir / "alignment_train_steps.jsonl"),
        },
        "raw_metrics": raw_metrics,
        "aligned_metrics": aligned_metrics,
        "alignment_debug": debug,
        "gt_bridge_debug": bridge_debug,
    }
    (args.output_dir / "learned_stream_alignment_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
