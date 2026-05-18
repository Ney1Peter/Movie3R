#!/usr/bin/env python3
"""Estimate a hand-crafted shot-boundary pose correction from anchor matches.

The test uses the external anchor patch pairs as geometric constraints:
reference-frame anchor 3D points define the target world positions, and
current-frame anchor 3D points are aligned to them with a fitted rigid/similarity
transform. This is an oracle-style post-process diagnostic, not a trainable model.
"""

import argparse
import itertools
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import cv2


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in [REPO_ROOT, REPO_ROOT / "src"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from add_ckpt_path import add_path_to_dust3r  # noqa: E402
from demo import parse_seq_path, prepare_input  # noqa: E402


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
    parser.add_argument("--out_json", default="output/anchor_pose_diagnostics/h36_new_manual_anchor.json")
    return parser.parse_args()


def rotation_angle_deg(r0, r1):
    rel = r0.T @ r1
    cos_angle = float(np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cos_angle))


def transform_points(points, rotation, translation, scale=1.0):
    return scale * (points @ rotation.T) + translation[None]


def fit_umeyama(src, dst, with_scale=False):
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape[0] < 3:
        return None
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src0 = src - src_mean
    dst0 = dst - dst_mean
    if np.linalg.matrix_rank(src0) < 2 or np.linalg.matrix_rank(dst0) < 2:
        return None

    cov = src0.T @ dst0 / float(src.shape[0])
    u, singular_values, vt = np.linalg.svd(cov)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        singular_values[-1] *= -1
        rotation = vt.T @ u.T

    if with_scale:
        variance = np.sum(src0 * src0) / float(src.shape[0])
        if variance <= 1e-12:
            return None
        scale = float(np.sum(singular_values) / variance)
    else:
        scale = 1.0

    translation = dst_mean - scale * (rotation @ src_mean)
    return {
        "rotation": rotation.astype(np.float32),
        "translation": translation.astype(np.float32),
        "scale": float(scale),
    }


def evaluate_fit(src, dst, fit):
    aligned = transform_points(src, fit["rotation"], fit["translation"], fit["scale"])
    return np.linalg.norm(aligned - dst, axis=1)


def robust_fit(src, dst, with_scale=False):
    src = np.asarray(src, dtype=np.float32)
    dst = np.asarray(dst, dtype=np.float32)
    n = src.shape[0]
    best = None
    best_residuals = None
    for indices in itertools.combinations(range(n), 3):
        candidate = fit_umeyama(src[list(indices)], dst[list(indices)], with_scale=with_scale)
        if candidate is None:
            continue
        residuals = evaluate_fit(src, dst, candidate)
        score = (float(np.median(residuals)), float(np.mean(residuals)))
        if best is None or score < best["score"]:
            best = {**candidate, "score": score}
            best_residuals = residuals

    if best is None:
        best = fit_umeyama(src, dst, with_scale=with_scale)
        if best is None:
            raise RuntimeError("failed to fit anchor transform")
        best_residuals = evaluate_fit(src, dst, best)

    threshold = max(0.05, 2.5 * float(np.median(best_residuals)))
    inliers = best_residuals <= threshold
    if int(inliers.sum()) >= 3:
        refit = fit_umeyama(src[inliers], dst[inliers], with_scale=with_scale)
        if refit is not None:
            best = {**refit, "score": (float(np.median(evaluate_fit(src, dst, refit))), float(np.mean(evaluate_fit(src, dst, refit))))}
            best_residuals = evaluate_fit(src, dst, best)

    best["inlier_count"] = int(inliers.sum())
    best["inlier_threshold"] = float(threshold)
    best["residuals"] = best_residuals.astype(np.float32)
    return best


def residual_stats(residuals):
    residuals = np.asarray(residuals, dtype=np.float32)
    return {
        "mean": float(residuals.mean()),
        "median": float(np.median(residuals)),
        "max": float(residuals.max()),
        "min": float(residuals.min()),
        "values": [float(v) for v in residuals.tolist()],
    }


def patch_idx_to_xy(patch_idx, grid_hw, image_hw):
    grid_h, grid_w = int(grid_hw[0]), int(grid_hw[1])
    image_h, image_w = int(image_hw[0]), int(image_hw[1])
    y = patch_idx // grid_w
    x = patch_idx % grid_w
    px = int(np.clip(round((x + 0.5) * image_w / grid_w), 0, image_w - 1))
    py = int(np.clip(round((y + 0.5) * image_h / grid_h), 0, image_h - 1))
    return px, py


def pose_mats(outputs, pose_encoding_to_camera):
    mats = []
    for pred in outputs["pred"]:
        pose = pred["camera_pose"].detach().float()
        mat = pose_encoding_to_camera(pose.clone()).detach().float().reshape(-1, 4, 4)[0]
        mats.append(mat.cpu().numpy())
    return np.stack(mats, axis=0)


def apply_camera(points, c2w):
    rotation = c2w[:3, :3]
    translation = c2w[:3, 3]
    return points @ rotation.T + translation[None]


def boundary_jump(mats, cur_idx):
    return {
        "translation_norm": float(np.linalg.norm(mats[cur_idx, :3, 3] - mats[cur_idx - 1, :3, 3])),
        "rotation_deg": rotation_angle_deg(mats[cur_idx - 1, :3, :3], mats[cur_idx, :3, :3]),
    }


def matrix_from_fit(fit):
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = fit["rotation"]
    mat[:3, 3] = fit["translation"]
    return mat


def fit_pnp(ref_world_points, cur_image_points, image_hw, focal):
    image_h, image_w = int(image_hw[0]), int(image_hw[1])
    camera_matrix = np.array(
        [[float(focal), 0.0, image_w / 2.0], [0.0, float(focal), image_h / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    object_points = np.asarray(ref_world_points, dtype=np.float32)
    image_points = np.asarray(cur_image_points, dtype=np.float32)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points,
        image_points,
        camera_matrix,
        None,
        iterationsCount=2000,
        reprojectionError=16.0,
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or inliers is None or len(inliers) < 4:
        return None

    inlier_idx = inliers.reshape(-1)
    if len(inlier_idx) >= 6:
        ok_refine, rvec, tvec = cv2.solvePnP(
            object_points[inlier_idx],
            image_points[inlier_idx],
            camera_matrix,
            None,
            rvec,
            tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok_refine:
            return None

    rotation_w2c, _ = cv2.Rodrigues(rvec)
    translation_w2c = tvec.reshape(3)
    rotation_c2w = rotation_w2c.T
    translation_c2w = -rotation_c2w @ translation_w2c
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = rotation_c2w.astype(np.float32)
    c2w[:3, 3] = translation_c2w.astype(np.float32)

    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, None)
    projected = projected.reshape(-1, 2)
    reproj = np.linalg.norm(projected - image_points, axis=1)
    return {
        "c2w": c2w,
        "inliers": inlier_idx.astype(np.int64),
        "reprojection_residuals_px": reproj.astype(np.float32),
        "focal": float(focal),
    }


def run_inference(args, inference_recurrent_lighter, model, views):
    print(f"Running no-anchor inference on {len(views)} frames...")
    start = time.time()
    with torch.no_grad():
        outputs, _ = inference_recurrent_lighter(
            views, model, args.device, verbose=True, use_ttt3r=args.use_ttt3r
        )
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"Inference finished in {time.time() - start:.2f}s")
    return outputs


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; switching to CPU")
        args.device = "cpu"

    anchor = np.load(args.anchor_path)
    ref_idx = int(anchor["ref_view_idx"][0])
    cur_idx = int(anchor["cur_view_idx"][0])
    if args.max_frames is not None and args.max_frames <= cur_idx:
        raise ValueError(f"--max_frames must include cur_idx={cur_idx}; got {args.max_frames}")

    add_path_to_dust3r(args.model_path)
    from src.dust3r.inference import inference_recurrent_lighter  # noqa: E402
    from src.dust3r.model import ARCroco3DStereo  # noqa: E402
    from src.dust3r.utils.camera import pose_encoding_to_camera  # noqa: E402
    from src.dust3r.post_process import estimate_focal_knowing_depth  # noqa: E402

    print(f"Loading model from {args.model_path}...")
    model = ARCroco3DStereo.from_pretrained(args.model_path).to(args.device)
    model.eval()

    img_paths, tmpdirname = parse_seq_path(args.seq_path)
    try:
        if args.max_frames is not None:
            img_paths = img_paths[: args.max_frames]
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
    finally:
        if tmpdirname is not None:
            shutil.rmtree(tmpdirname)

    outputs = run_inference(args, inference_recurrent_lighter, model, views)
    mats = pose_mats(outputs, pose_encoding_to_camera)

    ref_pts = outputs["pred"][ref_idx]["pts3d_in_self_view"].detach().float().cpu().numpy()[0]
    cur_pts = outputs["pred"][cur_idx]["pts3d_in_self_view"].detach().float().cpu().numpy()[0]
    ref_h, ref_w = ref_pts.shape[:2]
    cur_h, cur_w = cur_pts.shape[:2]
    ref_grid_hw = anchor["ref_grid_hw"] if "ref_grid_hw" in anchor.files else np.array([ref_h // 16, ref_w // 16])
    cur_grid_hw = anchor["cur_grid_hw"] if "cur_grid_hw" in anchor.files else np.array([cur_h // 16, cur_w // 16])

    ref_patch_idx = np.asarray(anchor["ref_patch_idx"], dtype=np.int64)
    cur_patch_idx = np.asarray(anchor["cur_patch_idx"], dtype=np.int64)
    anchor_mask = np.asarray(anchor["anchor_mask"], dtype=bool)
    ref_patch_idx = ref_patch_idx[anchor_mask]
    cur_patch_idx = cur_patch_idx[anchor_mask]

    src_cur_self = []
    dst_ref_world = []
    cur_image_points = []
    used_pairs = []
    for ref_idx_patch, cur_idx_patch in zip(ref_patch_idx, cur_patch_idx):
        ref_x, ref_y = patch_idx_to_xy(int(ref_idx_patch), ref_grid_hw, (ref_h, ref_w))
        cur_x, cur_y = patch_idx_to_xy(int(cur_idx_patch), cur_grid_hw, (cur_h, cur_w))
        ref_point_self = ref_pts[ref_y, ref_x]
        cur_point_self = cur_pts[cur_y, cur_x]
        if not (np.isfinite(ref_point_self).all() and np.isfinite(cur_point_self).all()):
            continue
        ref_point_world = apply_camera(ref_point_self[None], mats[ref_idx])[0]
        src_cur_self.append(cur_point_self)
        dst_ref_world.append(ref_point_world)
        cur_image_points.append([float(cur_x), float(cur_y)])
        used_pairs.append(
            {
                "ref_patch_idx": int(ref_idx_patch),
                "cur_patch_idx": int(cur_idx_patch),
                "ref_xy": [int(ref_x), int(ref_y)],
                "cur_xy": [int(cur_x), int(cur_y)],
            }
        )

    src_cur_self = np.asarray(src_cur_self, dtype=np.float32)
    dst_ref_world = np.asarray(dst_ref_world, dtype=np.float32)
    cur_image_points = np.asarray(cur_image_points, dtype=np.float32)
    if src_cur_self.shape[0] < 3:
        raise RuntimeError(f"need at least 3 finite anchors; got {src_cur_self.shape[0]}")

    cur_world_original = apply_camera(src_cur_self, mats[cur_idx])
    before_residuals = np.linalg.norm(cur_world_original - dst_ref_world, axis=1)
    rigid = robust_fit(src_cur_self, dst_ref_world, with_scale=False)
    similarity = robust_fit(src_cur_self, dst_ref_world, with_scale=True)

    rigid_mat = matrix_from_fit(rigid)
    original_mats_rigid_corrected = mats.copy()
    original_mats_rigid_corrected[cur_idx] = rigid_mat

    correction = rigid_mat @ np.linalg.inv(mats[cur_idx])
    post_cut_corrected_mats = mats.copy()
    for frame_idx in range(cur_idx, len(post_cut_corrected_mats)):
        post_cut_corrected_mats[frame_idx] = correction @ post_cut_corrected_mats[frame_idx]

    pp = torch.tensor([[cur_w // 2, cur_h // 2]], dtype=torch.float32)
    focal = estimate_focal_knowing_depth(
        torch.from_numpy(cur_pts).float().unsqueeze(0),
        pp,
        focal_mode="weiszfeld",
    ).reshape(-1)[0].item()
    pnp = fit_pnp(dst_ref_world, cur_image_points, (cur_h, cur_w), focal)
    pnp_report = None
    if pnp is not None:
        pnp_mats = mats.copy()
        pnp_mats[cur_idx] = pnp["c2w"]
        pnp_cur_world = apply_camera(src_cur_self, pnp["c2w"])
        pnp_report = {
            "focal": pnp["focal"],
            "inlier_count": int(len(pnp["inliers"])),
            "inlier_indices": [int(v) for v in pnp["inliers"].tolist()],
            "reprojection_residual_px": residual_stats(pnp["reprojection_residuals_px"]),
            "anchor_3d_residual_after_pnp": residual_stats(
                np.linalg.norm(pnp_cur_world - dst_ref_world, axis=1)
            ),
            "boundary_jump": boundary_jump(pnp_mats, cur_idx),
            "current_pose_change": {
                "translation_norm": float(np.linalg.norm(pnp["c2w"][:3, 3] - mats[cur_idx, :3, 3])),
                "rotation_deg": rotation_angle_deg(mats[cur_idx, :3, :3], pnp["c2w"][:3, :3]),
            },
            "c2w": pnp["c2w"].tolist(),
        }

    report = {
        "model_path": args.model_path,
        "seq_path": args.seq_path,
        "anchor_path": args.anchor_path,
        "ref_idx": ref_idx,
        "cur_idx": cur_idx,
        "anchor_quality_gate": float(np.asarray(anchor["quality_gate"]).reshape(-1)[0]),
        "anchor_valid_count_npz": int(anchor_mask.sum()),
        "finite_anchor_count_used": int(src_cur_self.shape[0]),
        "used_pairs": used_pairs,
        "before_anchor_3d_residual": residual_stats(before_residuals),
        "manual_rigid_anchor_3d_residual": residual_stats(rigid["residuals"]),
        "manual_similarity_anchor_3d_residual": residual_stats(similarity["residuals"]),
        "original_boundary_jump": boundary_jump(mats, cur_idx),
        "rigid_frame63_only_boundary_jump": boundary_jump(original_mats_rigid_corrected, cur_idx),
        "rigid_post_cut_global_boundary_jump": boundary_jump(post_cut_corrected_mats, cur_idx),
        "rigid_current_pose_change": {
            "translation_norm": float(np.linalg.norm(rigid_mat[:3, 3] - mats[cur_idx, :3, 3])),
            "rotation_deg": rotation_angle_deg(mats[cur_idx, :3, :3], rigid_mat[:3, :3]),
        },
        "rigid_global_correction_from_original_world": {
            "translation_norm": float(np.linalg.norm(correction[:3, 3])),
            "rotation_deg": rotation_angle_deg(np.eye(3), correction[:3, :3]),
        },
        "manual_pnp_from_ref3d_cur2d": pnp_report,
        "similarity_scale": float(similarity["scale"]),
        "rigid_transform_cur_self_to_ref_world": rigid_mat.tolist(),
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved manual anchor correction report to {out_json}")


if __name__ == "__main__":
    main()
