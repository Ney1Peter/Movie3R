#!/usr/bin/env python3
"""Apply the V10 learned anchor-weight alignment head to an image folder.

This is a strict streaming inference probe:

* Human3R runs frame by frame with an oracle reset before the boundary frame.
* The boundary transform uses only already-seen history frames plus the current
  boundary frame.
* Later frames reuse the cached transform.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
ARCHIVE_V7 = SCRIPTS_ROOT / "archive_v7"
for path in (str(REPO_ROOT), str(SRC_ROOT), str(SCRIPTS_ROOT), str(ARCHIVE_V7)):
    if path not in sys.path:
        sys.path.insert(0, path)

from add_ckpt_path import add_path_to_dust3r
from demo import parse_seq_path, prepare_input, prepare_output
from dust3r.model import ARCroco3DStereo
from dust3r.utils.device import to_cpu
from overfit_human_anchor_pose_correction import FOOT_JOINTS, STABLE_JOINTS, load_sequence
from v10_geometry_anchor_weight_probe import AnchorWeightHead
from v10_static_alignment_4source_probe import (
    body_frame_from_joints,
    build_geo_residual_features,
    compute_geo_proposal,
    solve_weighted_rigid_transform_batch,
)
from v10_token_alignment_4source_probe import build_pair_feature, token_debug_to_arrays
from v9_learned_stream_alignment_4source_probe import apply_transform_batch
from v9_learned_stream_alignment_overfit import rotation_geodesic
from v9_online_stream_human3r_segment_align import strict_original_model
from v9_segment_human3r_yaw_align_probe import copy_np_payload, copy_smpl, save_camera


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--case_name", default=None)
    parser.add_argument(
        "--output_root",
        type=Path,
        default=REPO_ROOT / "output" / "v10_anchor_weight_generalization_eval",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, default=REPO_ROOT / "src" / "human3r_896L.pth")
    parser.add_argument("--boundary", type=int, default=2, help="First frame index of the new segment.")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_inference", action="store_true")
    return parser.parse_args()


def list_images(input_dir: Path) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [p for p in sorted(input_dir.iterdir()) if p.suffix.lower() in suffixes]
    if len(images) < 4:
        raise ValueError(f"Need at least 4 images under {input_dir}, got {len(images)}")
    return images[:4]


def output_complete(output_dir: Path, num_frames: int) -> bool:
    return all((output_dir / "camera" / f"{i:06d}.npz").is_file() for i in range(num_frames))


def copy_inputs(input_dir: Path, case_dir: Path, overwrite: bool) -> Path:
    input_copy = case_dir / "input_frames"
    if input_copy.exists() and overwrite:
        shutil.rmtree(input_copy)
    input_copy.mkdir(parents=True, exist_ok=True)
    if len(list(input_copy.iterdir())) < 4 or overwrite:
        for old in input_copy.iterdir():
            old.unlink()
        for idx, image in enumerate(list_images(input_dir), start=1):
            shutil.copy2(image, input_copy / f"{idx:06d}{image.suffix.lower()}")
    return input_copy


def run_local_reset_with_tokens(
    input_copy: Path,
    local_dir: Path,
    token_path: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, np.ndarray]:
    if output_complete(local_dir, 4) and token_path.is_file() and not args.overwrite:
        data = np.load(token_path)
        return {key: data[key].astype(np.float32) for key in data.files}
    if local_dir.exists() and args.overwrite:
        shutil.rmtree(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    add_path_to_dust3r(str(args.model_path))
    img_paths, tmpdirname = parse_seq_path(str(input_copy))
    img_paths = img_paths[:4]
    model = ARCroco3DStereo.from_pretrained(str(args.model_path)).to(device).eval()
    disabled_lora = strict_original_model(model)
    img_res = getattr(model, "mhmr_img_res", None)
    views = prepare_input(
        img_paths=img_paths,
        img_mask=[True] * len(img_paths),
        size=args.size,
        revisit=1,
        update=True,
        img_res=img_res,
        reset_interval=10000000,
    )
    if tmpdirname is not None:
        shutil.rmtree(tmpdirname)

    for view in views:
        view["reset"] = torch.tensor(False).unsqueeze(0)
    if 0 < int(args.boundary) < len(views):
        views[int(args.boundary) - 1]["reset"] = torch.tensor(True).unsqueeze(0)

    with torch.no_grad():
        preds, batch, _, token_debug = model.forward_recurrent_lighter(
            views,
            str(device),
            ret_state=True,
            use_ttt3r=False,
            return_token_debug=True,
        )
    outputs_cpu = to_cpu({"pred": preds, "views": batch})
    outputs_to_save = {"pred": outputs_cpu["pred"], "views": [deepcopy(v) for v in outputs_cpu["views"]]}
    for view in outputs_to_save["views"]:
        view["reset"] = torch.tensor(False).unsqueeze(0)
    prepare_output(
        outputs_to_save,
        str(local_dir),
        1,
        True,
        True,
        False,
        False,
        img_res,
        1,
    )
    token_arrays = token_debug_to_arrays(token_debug)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(token_path, **token_arrays)
    summary = {
        "status": "ran",
        "model_path": str(args.model_path),
        "strict_original_human3r": True,
        "disabled_lora": disabled_lora,
        "streaming_reset_after_frames": [int(args.boundary) - 1],
        "boundaries": [int(args.boundary)],
        "num_frames": len(img_paths),
    }
    (local_dir / "online_stream_local_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return token_arrays


def write_aligned_outputs(local_dir: Path, aligned_dir: Path, poses: np.ndarray, R: np.ndarray, t: np.ndarray, boundary: int) -> None:
    if aligned_dir.exists():
        shutil.rmtree(aligned_dir)
    for sub in ["camera", "color", "conf", "depth", "smpl"]:
        (aligned_dir / sub).mkdir(parents=True, exist_ok=True)
    num_frames = poses.shape[0]
    for frame_idx in range(num_frames):
        cam_path = local_dir / "camera" / f"{frame_idx:06d}.npz"
        save_camera(cam_path, aligned_dir / "camera" / f"{frame_idx:06d}.npz", poses[frame_idx])
        if frame_idx >= boundary:
            copy_smpl(
                local_dir / "smpl" / f"{frame_idx:06d}.npz",
                aligned_dir / "smpl" / f"{frame_idx:06d}.npz",
                R,
                t,
            )
        else:
            copy_smpl(
                local_dir / "smpl" / f"{frame_idx:06d}.npz",
                aligned_dir / "smpl" / f"{frame_idx:06d}.npz",
                None,
                None,
            )
        for sub, ext in [("color", ".png"), ("conf", ".npy"), ("depth", ".npy")]:
            copy_np_payload(local_dir / sub / f"{frame_idx:06d}{ext}", aligned_dir / sub / f"{frame_idx:06d}{ext}")


def segment_metrics(joints: np.ndarray, boundary: int, joint_ids: np.ndarray) -> dict:
    hist = joints[:boundary, joint_ids].mean(axis=0)
    b0 = joints[boundary, joint_ids]
    b1 = joints[boundary + 1, joint_ids]
    return {
        "Amean_B0_m": float(np.linalg.norm(hist - b0, axis=-1).mean()),
        "Amean_B1_m": float(np.linalg.norm(hist - b1, axis=-1).mean()),
        "BB_m": float(np.linalg.norm(b0 - b1, axis=-1).mean()),
    }


def camera_delta(poses: np.ndarray, i: int, j: int) -> dict:
    rel = poses[j, :3, :3] @ poses[i, :3, :3].T
    cos = np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0)
    return {
        "r_deg": float(np.degrees(np.arccos(cos))),
        "t_m": float(np.linalg.norm(poses[j, :3, 3] - poses[i, :3, 3])),
    }


def body_frame_anchor_metrics(joints: np.ndarray, boundary: int) -> dict:
    joints_t = torch.from_numpy(joints.astype(np.float32))
    frames = body_frame_from_joints(joints_t)
    hist = frames[:boundary].mean(dim=0)
    hist_u, _, hist_vh = torch.linalg.svd(hist)
    hist = hist_u @ hist_vh
    post = frames[boundary : boundary + 2]
    err = rotation_geodesic(post, hist[None].expand_as(post))
    return {
        "B0_to_Amean_deg": float(torch.rad2deg(err[0]).item()),
        "B1_to_Amean_deg": float(torch.rad2deg(err[1]).item()),
        "Bmean_to_Amean_deg": float(torch.rad2deg(err).mean().item()),
    }


def pose_metrics_bundle(poses: np.ndarray, joints: np.ndarray, boundary: int, joint_ids: np.ndarray) -> dict:
    return {
        "segment_anchor": segment_metrics(joints, boundary, joint_ids),
        "body_frame_anchor": body_frame_anchor_metrics(joints, boundary),
        "camera_delta": {
            "AA_0_1": camera_delta(poses, 0, 1),
            "BB_2_3": camera_delta(poses, 2, 3),
            "boundary_1_2": camera_delta(poses, 1, 2),
        },
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    case_name = args.case_name or args.input_dir.name
    case_dir = args.output_root / case_name
    if case_dir.exists() and args.overwrite:
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    input_copy = copy_inputs(args.input_dir, case_dir, args.overwrite)
    local_dir = case_dir / "online_human3r_local_reset"
    token_path = case_dir / "token_features.npz"
    fixed_dir = case_dir / "online_human3r_fixed_geo_cached"
    aligned_dir = case_dir / "online_human3r_anchor_weight_aligned"

    if args.skip_inference and token_path.is_file():
        token_npz = np.load(token_path)
        token_features = {key: token_npz[key].astype(np.float32) for key in token_npz.files}
    else:
        token_features = run_local_reset_with_tokens(input_copy, local_dir, token_path, args, device)

    pred_data = load_sequence(local_dir, 4, torch.device("cpu"))
    pred_poses = torch.from_numpy(pred_data.poses[None]).to(device=device, dtype=torch.float32)
    pred_joints = torch.from_numpy(pred_data.joints_world[None]).to(device=device, dtype=torch.float32)
    joint_ids_np = np.asarray(sorted(set(STABLE_JOINTS + FOOT_JOINTS)), dtype=np.int64)
    joint_ids = torch.from_numpy(joint_ids_np).to(device=device, dtype=torch.long)
    boundary = int(args.boundary)

    R_fixed, t_fixed, base_weights = compute_geo_proposal(pred_joints, boundary, joint_ids)
    fixed_poses, fixed_joints = apply_transform_batch(
        pred_poses,
        pred_joints,
        R_fixed,
        t_fixed,
        boundary,
    )
    geo_features = build_geo_residual_features(pred_joints, R_fixed, t_fixed, boundary, joint_ids)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    ckpt_args = checkpoint.get("args", {})
    token_feature_set = str(ckpt_args.get("token_feature_set", "human_only"))
    token_piece = torch.from_numpy(build_pair_feature(token_features, token_feature_set, boundary)[None]).to(
        device=device,
        dtype=torch.float32,
    )
    features = torch.cat([geo_features, token_piece], dim=-1)
    num_anchors = int(joint_ids.numel())
    head = AnchorWeightHead(
        in_dim=features.shape[-1],
        hidden_dim=int(ckpt_args.get("hidden_dim", 256)),
        num_anchors=num_anchors,
        max_logit_delta=float(ckpt_args.get("max_logit_delta", 1.0)),
        gate_init_bias=float(ckpt_args.get("gate_init_bias", -2.0)),
    ).to(device)
    head.load_state_dict(checkpoint["model"], strict=True)
    head.eval()
    with torch.no_grad():
        learned_weights, gate = head(
            features,
            base_weights,
            use_gate=not bool(ckpt_args.get("disable_weight_gate", False)),
        )
        hist = pred_joints[:, :boundary, joint_ids].mean(dim=1)
        cur = pred_joints[:, boundary, joint_ids]
        R_learned, t_learned = solve_weighted_rigid_transform_batch(cur, hist, learned_weights)
        aligned_poses, aligned_joints = apply_transform_batch(
            pred_poses,
            pred_joints,
            R_learned,
            t_learned,
            boundary,
        )

    fixed_poses_np = fixed_poses[0].detach().cpu().numpy().astype(np.float32)
    fixed_joints_np = fixed_joints[0].detach().cpu().numpy().astype(np.float32)
    R_fixed_np = R_fixed[0].detach().cpu().numpy().astype(np.float64)
    t_fixed_np = t_fixed[0].detach().cpu().numpy().astype(np.float64)
    aligned_poses_np = aligned_poses[0].detach().cpu().numpy().astype(np.float32)
    aligned_joints_np = aligned_joints[0].detach().cpu().numpy().astype(np.float32)
    R_np = R_learned[0].detach().cpu().numpy().astype(np.float64)
    t_np = t_learned[0].detach().cpu().numpy().astype(np.float64)
    write_aligned_outputs(local_dir, fixed_dir, fixed_poses_np, R_fixed_np, t_fixed_np, boundary)
    write_aligned_outputs(local_dir, aligned_dir, aligned_poses_np, R_np, t_np, boundary)

    summary = {
        "case_name": case_name,
        "input_dir": str(args.input_dir),
        "local_dir": str(local_dir),
        "fixed_dir": str(fixed_dir),
        "aligned_dir": str(aligned_dir),
        "checkpoint": str(args.checkpoint),
        "boundary": boundary,
        "strict_streaming": {
            "uses_future_frames": False,
            "boundary_transform_input": "history frames before boundary + current boundary frame",
            "post_boundary_frames_use_cached_transform": True,
        },
        "token_feature_set": token_feature_set,
        "feature_dim": int(features.shape[-1]),
        "gate": float(gate.detach().cpu()[0]),
        "base_weights": base_weights.detach().cpu().numpy().astype(float).tolist(),
        "learned_weights": learned_weights.detach().cpu().numpy()[0].astype(float).tolist(),
        "transform": {
            "fixed_translation": t_fixed_np.astype(float).tolist(),
            "translation": t_np.astype(float).tolist(),
        },
        "metrics_no_gt": {
            "raw": pose_metrics_bundle(pred_data.poses, pred_data.joints_world, boundary, joint_ids_np),
            "fixed_geo_cached": pose_metrics_bundle(fixed_poses_np, fixed_joints_np, boundary, joint_ids_np),
            "learned_anchor_weight_cached": pose_metrics_bundle(aligned_poses_np, aligned_joints_np, boundary, joint_ids_np),
        },
    }
    (case_dir / "anchor_weight_stream_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
