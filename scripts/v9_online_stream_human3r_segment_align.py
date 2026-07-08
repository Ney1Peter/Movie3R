#!/usr/bin/env python3
"""True streaming Human3R segment alignment probe.

This differs from ``v9_segment_human3r_yaw_align_probe.py``:

* Human3R is run once on the full sequence through its recurrent streaming path.
* A shot boundary is represented as a reset after the last frame of the previous
  segment, so the next frame starts a fresh local state.
* The global transform for a new segment is estimated immediately from the first
  frame of that segment and cached for later frames.

The current script still uses an oracle boundary.  It tests whether the online
state/reset/alignment mechanics work without running each segment separately.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
ARCHIVE_V7 = REPO_ROOT / "scripts" / "archive_v7"
for path in (str(REPO_ROOT), str(SRC_ROOT), str(ARCHIVE_V7)):
    if path not in sys.path:
        sys.path.insert(0, path)

from add_ckpt_path import add_path_to_dust3r
from demo import parse_seq_path, prepare_input, prepare_output
from overfit_human_anchor_pose_correction import FOOT_JOINTS, STABLE_JOINTS, load_sequence
from v9_segment_human3r_yaw_align_probe import (
    copy_np_payload,
    copy_smpl,
    list_images,
    pose_delta_metrics,
    save_camera,
    transform_pose,
    weighted_joint_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--case_name", default=None)
    parser.add_argument("--output_root", type=Path, default=REPO_ROOT / "output" / "v9_online_stream_probe")
    parser.add_argument("--model_path", type=Path, default=REPO_ROOT / "src" / "human3r_896L.pth")
    parser.add_argument("--boundary", type=int, default=2, help="First frame index of the new segment.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_inference", action="store_true")
    parser.add_argument("--vis_threshold", type=float, default=1.5)
    return parser.parse_args()


def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / max(float(np.linalg.norm(v)), eps)


def rotation_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = normalize(axis)
    x, y, z = axis
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    eye = np.eye(3, dtype=np.float64)
    return eye + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def signed_yaw_between(src_dir: np.ndarray, dst_dir: np.ndarray, up: np.ndarray) -> float:
    up = normalize(up)
    src = normalize(src_dir - up * float(np.dot(src_dir, up)))
    dst = normalize(dst_dir - up * float(np.dot(dst_dir, up)))
    return math.atan2(float(np.dot(up, np.cross(src, dst))), float(np.dot(src, dst)))


def output_complete(output_dir: Path) -> bool:
    return (output_dir / "camera" / "000000.npz").is_file()


def strict_original_model(model) -> dict:
    from src.dust3r.v8_head_lora import set_lora_enabled

    disabled_flags = [
        "enable_shot_adaptation",
        "enable_shot_decoder_token",
        "enable_anchor_pose_adapter",
        "enable_anchor_decoder_tokens",
        "enable_anchor_pose_token_adapter",
        "enable_v7_pose_adapter",
        "enable_v8_pose_prompt",
        "enable_v8_human_trans_corr",
        "enable_v8_human_latent_corr",
        "enable_v8_head_lora",
        "enable_layerwise_pose_shot_adapter",
        "enable_pose_alignment_adapter",
        "enable_pose_translation_adapter",
        "enable_pose_lora",
        "enable_human_lora",
        "enable_world_lora",
    ]
    for name in disabled_flags:
        if hasattr(model, name):
            setattr(model, name, False)
    disabled_lora = {}
    if hasattr(model.downstream_head, "pose_head"):
        disabled_lora["pose_head"] = set_lora_enabled(model.downstream_head.pose_head, False)
    human_lora_count = 0
    for attr in ("deccam", "decpose", "decshape", "decexpression"):
        if hasattr(model.downstream_head, attr):
            human_lora_count += set_lora_enabled(getattr(model.downstream_head, attr), False)
    disabled_lora["human_head"] = human_lora_count
    return disabled_lora


def root_from_saved_frame(output_dir: Path, frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
    cam = np.load(output_dir / "camera" / f"{frame_idx:06d}.npz")["pose"].astype(np.float64)
    smpl = np.load(output_dir / "smpl" / f"{frame_idx:06d}.npz", allow_pickle=True)
    if smpl["rotvec"].shape[0] < 1:
        raise ValueError(f"No detected human in {output_dir / 'smpl' / f'{frame_idx:06d}.npz'}")
    root_R_cam, _ = cv2.Rodrigues(smpl["rotvec"][0, 0].astype(np.float64))
    root_R_world = cam[:3, :3] @ root_R_cam
    root_t_world = cam[:3, :3] @ smpl["transl"][0].astype(np.float64) + cam[:3, 3]
    return root_R_world[:, 2], root_t_world


def prepare_case_inputs(input_dir: Path, case_dir: Path, overwrite: bool) -> Path:
    input_copy = case_dir / "input_frames"
    if input_copy.exists() and overwrite:
        shutil.rmtree(input_copy)
    input_copy.mkdir(parents=True, exist_ok=True)
    images = list_images(input_dir)
    existing = sorted(input_copy.glob("*"))
    if len(existing) < 4 or overwrite:
        for p in existing:
            p.unlink()
        for i, src in enumerate(images, start=1):
            shutil.copy2(src, input_copy / f"{i:06d}{src.suffix.lower()}")
    return input_copy


def run_streaming_human3r(input_copy: Path, local_dir: Path, args: argparse.Namespace) -> dict:
    if output_complete(local_dir) and not args.overwrite:
        return {"status": "exists", "output_dir": str(local_dir)}
    if local_dir.exists() and args.overwrite:
        shutil.rmtree(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    add_path_to_dust3r(str(args.model_path))
    from src.dust3r.inference import inference_recurrent_lighter
    from src.dust3r.model import ARCroco3DStereo

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    device_t = torch.device(device)

    img_paths, tmpdirname = parse_seq_path(str(input_copy))
    img_paths = img_paths[:4]
    model = ARCroco3DStereo.from_pretrained(str(args.model_path)).to(device_t)
    disabled_lora = strict_original_model(model)
    model.eval()
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

    # Oracle boundary, online semantics: reset state after the last frame of A,
    # so B1 is processed from a fresh local Human3R state.
    for view in views:
        view["reset"] = torch.tensor(False).unsqueeze(0)
    views[int(args.boundary) - 1]["reset"] = torch.tensor(True).unsqueeze(0)

    with torch.no_grad():
        outputs, _ = inference_recurrent_lighter(views, model, str(device_t), use_ttt3r=False)

    # Save local segment gauges directly.  The reset has already affected the
    # model state; for saving we clear reset flags to avoid demo.py's overlap
    # post-processing path.
    outputs_to_save = {
        "pred": outputs["pred"],
        "views": [deepcopy(v) for v in outputs["views"]],
    }
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
    summary = {
        "status": "ran",
        "model_path": str(args.model_path),
        "strict_original_human3r": True,
        "disabled_lora": disabled_lora,
        "streaming_reset_after_frame": int(args.boundary - 1),
        "num_frames": len(img_paths),
        "output_dir": str(local_dir),
    }
    (local_dir / "online_stream_local_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def write_online_aligned(local_dir: Path, aligned_dir: Path, boundary: int, overwrite: bool) -> dict:
    if aligned_dir.exists() and overwrite:
        shutil.rmtree(aligned_dir)
    for sub in ["camera", "color", "conf", "depth", "smpl"]:
        (aligned_dir / sub).mkdir(parents=True, exist_ok=True)

    history_dirs: list[np.ndarray] = []
    history_trans: list[np.ndarray] = []
    segment_R = np.eye(3, dtype=np.float64)
    segment_t = np.zeros(3, dtype=np.float64)
    transform_records = []
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    n_frames = len(sorted((local_dir / "camera").glob("*.npz")))
    for frame_idx in range(n_frames):
        root_dir, root_t = root_from_saved_frame(local_dir, frame_idx)
        if frame_idx < boundary:
            segment_R = np.eye(3, dtype=np.float64)
            segment_t = np.zeros(3, dtype=np.float64)
            history_dirs.append(root_dir)
            history_trans.append(root_t)
            mode = "history_identity"
        elif frame_idx == boundary:
            if not history_dirs:
                raise ValueError("Boundary cannot be 0 for this online alignment probe.")
            hist_dir = normalize(np.mean(np.stack(history_dirs), axis=0))
            hist_t = np.mean(np.stack(history_trans), axis=0)
            yaw = signed_yaw_between(root_dir, hist_dir, up)
            segment_R = rotation_about_axis(up, yaw)
            segment_t = hist_t - segment_R @ root_t
            mode = "new_segment_first_frame_estimate"
            transform_records.append(
                {
                    "frame": int(frame_idx),
                    "yaw_deg": float(math.degrees(yaw)),
                    "translation": segment_t.astype(np.float32).tolist(),
                    "history_root_translation_mean": hist_t.astype(np.float32).tolist(),
                    "current_root_translation_raw": root_t.astype(np.float32).tolist(),
                    "current_root_translation_aligned": (segment_R @ root_t + segment_t).astype(np.float32).tolist(),
                }
            )
        else:
            mode = "cached_segment_transform"

        cam_path = local_dir / "camera" / f"{frame_idx:06d}.npz"
        cam = np.load(cam_path)
        pose = transform_pose(cam["pose"].astype(np.float32), segment_R, segment_t)
        save_camera(cam_path, aligned_dir / "camera" / f"{frame_idx:06d}.npz", pose)
        copy_smpl(
            local_dir / "smpl" / f"{frame_idx:06d}.npz",
            aligned_dir / "smpl" / f"{frame_idx:06d}.npz",
            segment_R if frame_idx >= boundary else None,
            segment_t if frame_idx >= boundary else None,
        )
        for sub, ext in [("color", ".png"), ("conf", ".npy"), ("depth", ".npy")]:
            copy_np_payload(local_dir / sub / f"{frame_idx:06d}{ext}", aligned_dir / sub / f"{frame_idx:06d}{ext}")

        if frame_idx >= boundary:
            aligned_root_t = segment_R @ root_t + segment_t
        else:
            aligned_root_t = root_t
        transform_records.append(
            {
                "frame": int(frame_idx),
                "mode": mode,
                "root_translation_raw": root_t.astype(np.float32).tolist(),
                "root_translation_aligned": aligned_root_t.astype(np.float32).tolist(),
            }
        )

    return {
        "local_dir": str(local_dir),
        "aligned_dir": str(aligned_dir),
        "boundary": int(boundary),
        "transform_records": transform_records,
    }


def joint_dist(a: np.ndarray, b: np.ndarray, joint_ids: np.ndarray) -> float:
    return float(np.linalg.norm(a[joint_ids] - b[joint_ids], axis=-1).mean())


def compute_metrics(local_dir: Path, aligned_dir: Path, boundary: int) -> dict:
    joint_ids, _ = weighted_joint_ids(1.0, 1.5)
    local = load_sequence(local_dir, 4, torch.device("cpu"))
    aligned = load_sequence(aligned_dir, 4, torch.device("cpu"))
    a_mean = aligned.joints_world[:boundary, joint_ids].mean(axis=0)
    return {
        "human_joint_mean_m": {
            "AA_0_1_aligned": joint_dist(aligned.joints_world[0], aligned.joints_world[1], joint_ids),
            "BB_2_3_local": joint_dist(local.joints_world[2], local.joints_world[3], joint_ids),
            "BB_2_3_aligned": joint_dist(aligned.joints_world[2], aligned.joints_world[3], joint_ids),
            "A_mean_to_B0_local": joint_dist(local.joints_world[:boundary, joint_ids].mean(axis=0), local.joints_world[boundary, joint_ids], np.arange(len(joint_ids))),
            "A_mean_to_B0_aligned": joint_dist(a_mean, aligned.joints_world[boundary, joint_ids], np.arange(len(joint_ids))),
            "A_mean_to_B1_aligned": joint_dist(a_mean, aligned.joints_world[boundary + 1, joint_ids], np.arange(len(joint_ids))),
        },
        "camera_delta": {
            "AA_0_1_aligned": pose_delta_metrics(aligned.poses, 0, 1),
            "BB_2_3_local": pose_delta_metrics(local.poses, 2, 3),
            "BB_2_3_aligned": pose_delta_metrics(aligned.poses, 2, 3),
            "boundary_1_2_local": pose_delta_metrics(local.poses, 1, 2),
            "boundary_1_2_aligned": pose_delta_metrics(aligned.poses, 1, 2),
        },
    }


def main() -> None:
    args = parse_args()
    case_name = args.case_name or args.input_dir.name
    case_dir = args.output_root / case_name
    if case_dir.exists() and args.overwrite:
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    input_copy = prepare_case_inputs(args.input_dir, case_dir, args.overwrite)
    local_dir = case_dir / "online_human3r_local_reset"
    aligned_dir = case_dir / "online_human3r_global_aligned"

    run_summary = {"status": "skip_inference"}
    if not args.skip_inference:
        run_summary = run_streaming_human3r(input_copy, local_dir, args)
    align_summary = write_online_aligned(local_dir, aligned_dir, int(args.boundary), args.overwrite)
    metrics = compute_metrics(local_dir, aligned_dir, int(args.boundary))
    summary = {
        "case_name": case_name,
        "input_dir": str(args.input_dir),
        "case_dir": str(case_dir),
        "run": run_summary,
        "alignment": align_summary,
        "metrics": metrics,
        "method": "true streaming full-sequence Human3R with oracle reset boundary and online first-frame segment transform cache",
    }
    summary_path = case_dir / "online_stream_alignment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
