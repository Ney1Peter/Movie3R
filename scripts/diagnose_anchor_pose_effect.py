#!/usr/bin/env python3
"""Diagnose whether V6-A AnchorPoseAdapter changes camera pose at inference.

This script intentionally does not launch the viewer. It runs the same video once
without injected anchors and once with injected anchors, then reports whether the
adapter produced a non-zero camera-pose residual. It also applies a manual pose
delta to the boundary frame to verify that downstream camera/point transforms are
actually sensitive to camera_pose changes.
"""

import argparse
import copy
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in [REPO_ROOT, REPO_ROOT / "src"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from add_ckpt_path import add_path_to_dust3r  # noqa: E402
from demo import inject_video_anchor, parse_seq_path, prepare_input  # noqa: E402


def parse_vec3(text):
    values = [float(v.strip()) for v in text.split(",")]
    if len(values) != 3:
        raise argparse.ArgumentTypeError("expected comma-separated x,y,z")
    return values


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--seq_path", required=True)
    parser.add_argument("--anchor_path", required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--max_frames", type=int, default=66)
    parser.add_argument("--reset_interval", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use_ttt3r", action="store_true")
    parser.add_argument("--manual_delta_t", type=parse_vec3, default="0.5,0,0")
    parser.add_argument("--out_json", default="output/anchor_pose_diagnostics/h36_new_a_b.json")
    return parser.parse_args()


def scalar_tensor(value):
    if value is None or not torch.is_tensor(value):
        return None
    return float(value.detach().float().reshape(-1).mean().cpu())


def rotation_angle_deg(r0, r1):
    rel = r0.T @ r1
    cos_angle = float(np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cos_angle))


def camera_mats(outputs, pose_encoding_to_camera):
    mats = []
    encs = []
    for pred in outputs["pred"]:
        enc = pred["camera_pose"].detach().float()
        mat = pose_encoding_to_camera(enc.clone()).detach().float()
        encs.append(enc.reshape(-1).cpu().numpy())
        mats.append(mat.reshape(-1, 4, 4)[0].cpu().numpy())
    return np.stack(mats, axis=0), np.stack(encs, axis=0)


def boundary_jump(mats, cur_idx):
    prev_idx = cur_idx - 1
    t_jump = float(np.linalg.norm(mats[cur_idx, :3, 3] - mats[prev_idx, :3, 3]))
    r_jump = rotation_angle_deg(mats[prev_idx, :3, :3], mats[cur_idx, :3, :3])
    return {"translation_norm": t_jump, "rotation_deg": r_jump}


def anchor_details(outputs):
    keys = [
        "anchor_pose_gate",
        "anchor_pose_delta_t_norm",
        "anchor_pose_delta_q_norm",
        "anchor_pose_valid",
        "anchor_pose_attn_max",
    ]
    details = []
    for frame_idx, pred in enumerate(outputs["pred"]):
        item = {"frame_idx": frame_idx}
        has_value = False
        for key in keys:
            value = scalar_tensor(pred.get(key))
            if value is not None:
                item[key] = value
                has_value = True
        if has_value:
            details.append(item)
    return details


def summarize_pair(no_anchor_outputs, with_anchor_outputs, cur_idx, pose_encoding_to_camera):
    no_mats, no_encs = camera_mats(no_anchor_outputs, pose_encoding_to_camera)
    with_mats, with_encs = camera_mats(with_anchor_outputs, pose_encoding_to_camera)
    enc_diff = np.linalg.norm(with_encs - no_encs, axis=1)
    trans_diff = np.linalg.norm(with_mats[:, :3, 3] - no_mats[:, :3, 3], axis=1)
    rot_diff = np.array(
        [rotation_angle_deg(no_mats[i, :3, :3], with_mats[i, :3, :3]) for i in range(len(no_mats))],
        dtype=np.float32,
    )
    return {
        "num_frames": int(len(no_mats)),
        "cur_idx": int(cur_idx),
        "with_anchor_details": anchor_details(with_anchor_outputs),
        "no_anchor_boundary_jump": boundary_jump(no_mats, cur_idx),
        "with_anchor_boundary_jump": boundary_jump(with_mats, cur_idx),
        "cur_frame_encoded_pose_diff_norm": float(enc_diff[cur_idx]),
        "cur_frame_translation_diff_norm": float(trans_diff[cur_idx]),
        "cur_frame_rotation_diff_deg": float(rot_diff[cur_idx]),
        "max_encoded_pose_diff_norm": float(enc_diff.max()),
        "max_translation_diff_norm": float(trans_diff.max()),
        "max_rotation_diff_deg": float(rot_diff.max()),
        "frames_with_translation_diff_gt_1e_6": np.flatnonzero(trans_diff > 1e-6).astype(int).tolist(),
    }


def manual_pose_test(outputs, cur_idx, manual_delta_t, pose_encoding_to_camera, geotrf):
    pred = outputs["pred"][cur_idx]
    pose_base = pred["camera_pose"].detach().float()
    pose_manual = pose_base.clone()
    delta = torch.tensor(manual_delta_t, device=pose_manual.device, dtype=pose_manual.dtype).view(1, 3)
    pose_manual[:, :3] = pose_manual[:, :3] + delta

    mat_base = pose_encoding_to_camera(pose_base.clone()).detach().float().reshape(-1, 4, 4)[0]
    mat_manual = pose_encoding_to_camera(pose_manual.clone()).detach().float().reshape(-1, 4, 4)[0]

    pts_self = pred["pts3d_in_self_view"].detach().float()
    world_base = geotrf(mat_base.unsqueeze(0), pts_self)
    world_manual = geotrf(mat_manual.unsqueeze(0), pts_self)
    shift = (world_manual - world_base).reshape(-1, 3)
    finite = torch.isfinite(shift).all(dim=-1)
    shift = shift[finite]
    shift_norm = shift.norm(dim=-1)

    return {
        "cur_idx": int(cur_idx),
        "manual_delta_t_encoded": [float(v) for v in manual_delta_t],
        "camera_translation_shift_norm": float((mat_manual[:3, 3] - mat_base[:3, 3]).norm().cpu()),
        "point_shift_mean_norm": float(shift_norm.mean().cpu()),
        "point_shift_median_norm": float(shift_norm.median().cpu()),
        "point_shift_max_norm": float(shift_norm.max().cpu()),
        "finite_points": int(shift.shape[0]),
    }


def run_case(label, views, model, device, use_ttt3r, inference_recurrent_lighter):
    print(f"Running {label} inference on {len(views)} frames...")
    start = time.time()
    with torch.no_grad():
        outputs, _ = inference_recurrent_lighter(
            copy.deepcopy(views), model, device, verbose=True, use_ttt3r=use_ttt3r
        )
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"{label} inference finished in {time.time() - start:.2f}s")
    return outputs


def main():
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; switching to CPU")
        device = "cpu"

    anchor_data = np.load(args.anchor_path)
    ref_idx = int(anchor_data["ref_view_idx"][0])
    cur_idx = int(anchor_data["cur_view_idx"][0])
    if args.max_frames is not None and args.max_frames <= cur_idx:
        raise ValueError(f"--max_frames must include cur_idx={cur_idx}; got {args.max_frames}")

    add_path_to_dust3r(args.model_path)
    from src.dust3r.inference import inference_recurrent_lighter  # noqa: E402
    from src.dust3r.model import ARCroco3DStereo  # noqa: E402
    from src.dust3r.utils.camera import pose_encoding_to_camera  # noqa: E402
    from src.dust3r.utils.geometry import geotrf  # noqa: E402

    print(f"Loading model from {args.model_path}...")
    model = ARCroco3DStereo.from_pretrained(args.model_path).to(device)
    model.eval()
    print(f"enable_anchor_pose_adapter={getattr(model, 'enable_anchor_pose_adapter', None)}")
    print(f"enable_anchor_pose_rotation={getattr(model, 'enable_anchor_pose_rotation', None)}")

    img_paths, tmpdirname = parse_seq_path(args.seq_path)
    try:
        if args.max_frames is not None:
            img_paths = img_paths[: args.max_frames]
        print(f"Preparing {len(img_paths)} frames; anchor boundary ref={ref_idx}, cur={cur_idx}")
        img_res = getattr(model, "mhmr_img_res", None)
        views = prepare_input(
            img_paths=img_paths,
            img_mask=[True] * len(img_paths),
            size=args.size,
            revisit=1,
            update=True,
            img_res=img_res,
            reset_interval=args.reset_interval,
        )
        views_with_anchor = inject_video_anchor(copy.deepcopy(views), args.anchor_path)
    finally:
        if tmpdirname is not None:
            shutil.rmtree(tmpdirname)

    no_anchor_outputs = run_case(
        "no-anchor", views, model, device, args.use_ttt3r, inference_recurrent_lighter
    )
    with_anchor_outputs = run_case(
        "with-anchor", views_with_anchor, model, device, args.use_ttt3r, inference_recurrent_lighter
    )

    report = {
        "model_path": args.model_path,
        "seq_path": args.seq_path,
        "anchor_path": args.anchor_path,
        "ref_idx": ref_idx,
        "cur_idx": cur_idx,
        "anchor_npz_quality_gate": float(np.asarray(anchor_data["quality_gate"]).reshape(-1)[0]),
        "anchor_npz_valid_count": int(np.asarray(anchor_data["anchor_mask"]).astype(bool).sum()),
        "A_anchor_vs_no_anchor": summarize_pair(
            no_anchor_outputs, with_anchor_outputs, cur_idx, pose_encoding_to_camera
        ),
        "B_manual_pose_delta": manual_pose_test(
            no_anchor_outputs, cur_idx, args.manual_delta_t, pose_encoding_to_camera, geotrf
        ),
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved diagnostics to {out_json}")


if __name__ == "__main__":
    main()
