#!/usr/bin/env python3
"""V10 minimal dual-state, same-frame bridge diagnostic.

The probe reuses strict original Human3R outputs from the existing AvatarReX
state-vs-gauge experiment.  At the first post-cut RGB frame it keeps two
causal branches:

* Continue: read the old recurrent state once to expose its old-world gauge.
* Reset: reconstruct the new shot with a fresh recurrent state.

No model is trained.  A single boundary transform maps the complete Reset
shot back to the Continue world.  Ground truth is used only for evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import roma
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts"
SRC_ROOT = REPO_ROOT / "src"
for path in (str(SCRIPT_ROOT), str(REPO_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from dust3r.utils.smpl_layer import SMPL_Layer  # noqa: E402
from v10_oracle_state_vs_gauge_probe import (  # noqa: E402
    load_gt_c2w,
    load_pose,
    mean_root,
    merge_reset_output,
    rotation_error_deg,
    sample_points,
    summarize_rpe,
    summarize_variant,
    threshold_stats,
    write_pose,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base_probe_dir",
        type=Path,
        default=REPO_ROOT / "output" / "v10_oracle_state_vs_gauge_probe" / "avatarrex_lbn1_1192_cut10",
        help="Existing strict-Human3R Continue/Fresh probe output.",
    )
    parser.add_argument(
        "--raw_meta_root",
        type=Path,
        default=Path("/data/wangzheng/iJCV-CODE/data/AvatarReX_raw_meta/lbn1"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=REPO_ROOT / "output" / "v10_dual_state_same_frame_bridge" / "avatarrex_lbn1_1192_cut10",
    )
    parser.add_argument("--conf_quantile", type=float, default=0.60)
    parser.add_argument("--human_mask_threshold", type=float, default=0.10)
    parser.add_argument("--human_mask_dilate", type=int, default=15)
    parser.add_argument("--max_bridge_pairs", type=int, default=50000)
    parser.add_argument("--max_metric_points", type=int, default=1200)
    parser.add_argument("--smpl_device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def require_base_probe(base: Path) -> dict:
    manifest_path = base / "case_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing base manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for rel in ("A_raw_continue", "raw_post_fresh_state"):
        path = base / rel
        if not (path / "camera").is_dir():
            raise FileNotFoundError(f"Missing strict Human3R output: {path}")
    return manifest


def build_target_poses(raw_meta_root: Path, records: list[dict], raw_continue: Path) -> list[np.ndarray]:
    gt = [load_gt_c2w(raw_meta_root, record["seq"]) for record in records]
    raw0, _ = load_pose(raw_continue, 0)
    # Evaluation gauge only.  This alignment is never used by either bridge.
    eval_align = raw0 @ np.linalg.inv(gt[0])
    return [(eval_align @ pose).astype(np.float32) for pose in gt]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_same_boundary_rgb(base_probe_dir: Path, cut_idx: int) -> dict:
    continue_images = sorted((base_probe_dir / "input_all").glob("*.png"))
    reset_images = sorted((base_probe_dir / "input_post").glob("*.png"))
    if cut_idx >= len(continue_images) or not reset_images:
        raise RuntimeError("Base probe does not contain both boundary RGB inputs")
    continue_path = continue_images[cut_idx]
    reset_path = reset_images[0]
    continue_rgb = cv2.imread(str(continue_path), cv2.IMREAD_UNCHANGED)
    reset_rgb = cv2.imread(str(reset_path), cv2.IMREAD_UNCHANGED)
    if continue_rgb is None or reset_rgb is None:
        raise RuntimeError("Could not load boundary RGB inputs")
    same_shape = continue_rgb.shape == reset_rgb.shape
    same_pixels = bool(same_shape and np.array_equal(continue_rgb, reset_rgb))
    if not same_pixels:
        raise RuntimeError("Continue and Reset boundary branches are not processing identical RGB pixels")
    return {
        "continue_path": str(continue_path),
        "reset_path": str(reset_path),
        "shape": list(continue_rgb.shape),
        "same_pixels": same_pixels,
        "continue_sha256": file_sha256(continue_path),
        "reset_sha256": file_sha256(reset_path),
    }


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    return (points @ T[:3, :3].T + T[:3, 3]).astype(np.float32)


def load_human_mask(output_dir: Path, idx: int, shape: tuple[int, int], threshold: float, dilate: int) -> np.ndarray:
    smpl = np.load(output_dir / "smpl" / f"{idx:06d}.npz", allow_pickle=True)
    msk = smpl["msk"]
    if msk.shape == () and msk.item() is None:
        body = np.zeros(shape, dtype=np.uint8)
    else:
        msk = np.asarray(msk, dtype=np.float32)
        if msk.ndim == 2:
            body = msk > threshold
        else:
            body = np.max(msk, axis=0) > threshold
        body = body.astype(np.uint8)
    if body.shape != shape:
        body = cv2.resize(body, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    if dilate > 1:
        kernel = np.ones((dilate, dilate), dtype=np.uint8)
        body = cv2.dilate(body, kernel, iterations=1)
    return body.astype(bool)


def dense_world_pointmap(output_dir: Path, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pose, K = load_pose(output_dir, idx)
    depth = np.load(output_dir / "depth" / f"{idx:06d}.npy").astype(np.float32)
    conf = np.load(output_dir / "conf" / f"{idx:06d}.npy").astype(np.float32)
    h, w = depth.shape
    ys, xs = np.indices((h, w), dtype=np.float32)
    z = depth
    x = (xs - K[0, 2]) / K[0, 0] * z
    y = (ys - K[1, 2]) / K[1, 1] * z
    points_cam = np.stack([x, y, z], axis=-1)
    points_world = transform_points(pose, points_cam.reshape(-1, 3)).reshape(h, w, 3)
    return points_world, depth, conf


def save_boundary_payload(dst: Path, output_dir: Path, idx: int, args: argparse.Namespace) -> None:
    pointmap_world, depth, conf = dense_world_pointmap(output_dir, idx)
    pose, K = load_pose(output_dir, idx)
    human_mask = load_human_mask(
        output_dir,
        idx,
        depth.shape,
        args.human_mask_threshold,
        dilate=1,
    )
    with np.load(output_dir / "smpl" / f"{idx:06d}.npz", allow_pickle=True) as smpl:
        payload = {f"smpl_{key}": smpl[key] for key in smpl.files}
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dst,
        camera_pose_c2w=pose,
        camera_intrinsics=K,
        pointmap_world=pointmap_world,
        depth_camera=depth,
        confidence=conf,
        human_mask=human_mask,
        **payload,
    )


def normalized_confidence(conf: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = conf[valid]
    if values.size == 0:
        return np.zeros_like(conf, dtype=np.float32)
    lo, hi = np.quantile(values, [0.05, 0.95])
    scale = max(float(hi - lo), 1e-6)
    return np.clip((conf - float(lo)) / scale, 0.0, 1.0).astype(np.float32)


def weighted_kabsch(src: np.ndarray, dst: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / max(float(weights.sum()), 1e-12)
    src64 = src.astype(np.float64)
    dst64 = dst.astype(np.float64)
    src_mean = (src64 * weights[:, None]).sum(axis=0)
    dst_mean = (dst64 * weights[:, None]).sum(axis=0)
    src_c = src64 - src_mean
    dst_c = dst64 - dst_mean
    covariance = (src_c * weights[:, None]).T @ dst_c
    U, _, Vt = np.linalg.svd(covariance)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1.0
        R = Vt.T @ U.T
    t = dst_mean - R @ src_mean
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R.astype(np.float32)
    T[:3, 3] = t.astype(np.float32)
    return T


def robust_fixed_correspondence_se3(
    src: np.ndarray,
    dst: np.ndarray,
    weights: np.ndarray,
    max_iters: int = 8,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Robust Kabsch with fixed pixel correspondences; no ICP or rematching."""
    active = np.ones(len(src), dtype=bool)
    history = []
    T = np.eye(4, dtype=np.float32)
    for iteration in range(max_iters):
        if int(active.sum()) < 64:
            break
        T = weighted_kabsch(src[active], dst[active], weights[active])
        residual = np.linalg.norm(transform_points(T, src) - dst, axis=1)
        active_residual = residual[active]
        median = float(np.median(active_residual))
        mad = float(np.median(np.abs(active_residual - median)))
        robust_sigma = max(1.4826 * mad, 1e-4)
        mad_limit = median + 3.0 * robust_sigma
        quantile_limit = float(np.quantile(active_residual, 0.85))
        threshold = max(0.005, min(mad_limit, quantile_limit))
        new_active = residual <= threshold
        history.append(
            {
                "iteration": iteration,
                "inliers": int(active.sum()),
                "median_residual": median,
                "mad": mad,
                "threshold": threshold,
            }
        )
        if np.array_equal(new_active, active):
            break
        active = new_active
    if int(active.sum()) >= 64:
        T = weighted_kabsch(src[active], dst[active], weights[active])
    return T, active, history


def estimate_pointmap_bridge(
    continue_dir: Path,
    continue_idx: int,
    reset_dir: Path,
    reset_idx: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict]:
    dst_world, dst_depth, dst_conf = dense_world_pointmap(continue_dir, continue_idx)
    src_world, src_depth, src_conf = dense_world_pointmap(reset_dir, reset_idx)
    if dst_world.shape != src_world.shape:
        raise RuntimeError(f"Pointmap shape mismatch: {dst_world.shape} vs {src_world.shape}")
    shape = src_depth.shape
    human = load_human_mask(continue_dir, continue_idx, shape, args.human_mask_threshold, args.human_mask_dilate)
    human |= load_human_mask(reset_dir, reset_idx, shape, args.human_mask_threshold, args.human_mask_dilate)

    finite = (
        np.isfinite(src_world).all(axis=-1)
        & np.isfinite(dst_world).all(axis=-1)
        & np.isfinite(src_depth)
        & np.isfinite(dst_depth)
        & np.isfinite(src_conf)
        & np.isfinite(dst_conf)
    )
    valid_depth = (src_depth > 0.05) & (src_depth < 50.0) & (dst_depth > 0.05) & (dst_depth < 50.0)
    base_valid = finite & valid_depth & ~human
    if int(base_valid.sum()) < 64:
        raise RuntimeError(f"Only {int(base_valid.sum())} valid non-human point correspondences")
    src_threshold = float(np.quantile(src_conf[base_valid], args.conf_quantile))
    dst_threshold = float(np.quantile(dst_conf[base_valid], args.conf_quantile))
    valid = base_valid & (src_conf >= src_threshold) & (dst_conf >= dst_threshold)
    ys, xs = np.where(valid)
    if len(xs) < 64:
        raise RuntimeError(f"Only {len(xs)} high-confidence point correspondences")
    rng = np.random.default_rng(20260715)
    if len(xs) > args.max_bridge_pairs:
        keep = rng.choice(len(xs), size=args.max_bridge_pairs, replace=False)
        ys, xs = ys[keep], xs[keep]

    src = src_world[ys, xs]
    dst = dst_world[ys, xs]
    src_conf_norm = normalized_confidence(src_conf, base_valid)[ys, xs]
    dst_conf_norm = normalized_confidence(dst_conf, base_valid)[ys, xs]
    weights = np.sqrt(np.maximum(src_conf_norm * dst_conf_norm, 1e-6)).astype(np.float32)
    T, inliers, history = robust_fixed_correspondence_se3(src, dst, weights)
    residual = np.linalg.norm(transform_points(T, src) - dst, axis=1)
    diagnostics = {
        "method": "same-pixel background-only confidence-filtered robust weighted Kabsch",
        "uses_future_frames": False,
        "uses_icp_or_ba": False,
        "continue_idx": continue_idx,
        "reset_idx": reset_idx,
        "base_valid_pixels": int(base_valid.sum()),
        "high_confidence_pairs_before_sampling": int(valid.sum()),
        "sampled_pairs": int(len(src)),
        "final_inliers": int(inliers.sum()),
        "final_inlier_rate": float(inliers.mean()),
        "src_conf_threshold": src_threshold,
        "dst_conf_threshold": dst_threshold,
        "human_excluded_pixels": int(human.sum()),
        "residual_all_median": float(np.median(residual)),
        "residual_all_p90": float(np.quantile(residual, 0.90)),
        "residual_inlier_mean": float(residual[inliers].mean()),
        "residual_inlier_median": float(np.median(residual[inliers])),
        "rotation_det": float(np.linalg.det(T[:3, :3])),
        "rotation_orthogonality_error": float(np.linalg.norm(T[:3, :3].T @ T[:3, :3] - np.eye(3))),
        "iterations": history,
    }
    return T, diagnostics


def apply_bridge_to_shot(src_dir: Path, dst_dir: Path, T: np.ndarray, cut_idx: int, frame_count: int) -> None:
    copy_tree(src_dir, dst_dir)
    for idx in range(cut_idx, frame_count):
        pose, K = load_pose(src_dir, idx)
        write_pose(dst_dir, idx, T @ pose, K)
    (dst_dir / "segment_bridge.json").write_text(
        json.dumps(
            {
                "cut_idx": cut_idx,
                "frame_count": frame_count,
                "reset_local_to_old_world_se3": T.tolist(),
                "applied_once_at_boundary": True,
                "reestimated_per_frame": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def add_world_smpl_payload(output_dir: Path, frame_count: int, device: str) -> None:
    """Cache world root/orientation/mesh under the corrected camera gauge."""
    layer_cache: dict[int, SMPL_Layer] = {}
    faces_cache: dict[int, np.ndarray] = {}
    for idx in range(frame_count):
        smpl_path = output_dir / "smpl" / f"{idx:06d}.npz"
        with np.load(smpl_path, allow_pickle=True) as smpl:
            payload = {key: smpl[key] for key in smpl.files}
        shape = np.asarray(payload["shape"], dtype=np.float32)
        rotvec = np.asarray(payload["rotvec"], dtype=np.float32)
        transl = np.asarray(payload["transl"], dtype=np.float32)
        pose, K = load_pose(output_dir, idx)
        num_people = len(shape)
        beta_dim = shape.shape[-1] if shape.ndim == 2 and shape.shape[-1] else 10
        if beta_dim not in layer_cache:
            layer_cache[beta_dim] = SMPL_Layer(
                type="smplx", gender="neutral", num_betas=beta_dim, kid=False, person_center="head"
            ).to(device)
            layer_cache[beta_dim].eval()
            faces_cache[beta_dim] = layer_cache[beta_dim].bm_x.faces.astype(np.int32)

        if num_people:
            expression = payload.get("expression")
            if isinstance(expression, np.ndarray) and expression.shape == () and expression.item() is None:
                expression = None
            expression_t = None if expression is None else torch.from_numpy(np.asarray(expression, dtype=np.float32)).to(device)
            with torch.no_grad():
                smpl_out = layer_cache[beta_dim](
                    torch.from_numpy(rotvec).to(device),
                    torch.from_numpy(shape).to(device),
                    torch.from_numpy(transl).to(device),
                    None,
                    None,
                    K=torch.from_numpy(K).to(device).expand(num_people, -1, -1),
                    expression=expression_t,
                )
            verts_cam = smpl_out["smpl_v3d"].detach().cpu().numpy().astype(np.float32)
            pelvis_cam = smpl_out["smpl_transl_pelvis"][:, 0].detach().cpu().numpy().astype(np.float32)
            verts_world = transform_points(pose, verts_cam.reshape(-1, 3)).reshape(verts_cam.shape)
            pelvis_world = transform_points(pose, pelvis_cam)
            anchor_world = transform_points(pose, transl)
            root_cam_R = roma.rotvec_to_rotmat(torch.from_numpy(rotvec[:, 0])).numpy().astype(np.float32)
            root_world_R = np.einsum("ij,njk->nik", pose[:3, :3], root_cam_R).astype(np.float32)
        else:
            verts_world = np.empty((0, 0, 3), dtype=np.float32)
            pelvis_world = np.empty((0, 3), dtype=np.float32)
            anchor_world = np.empty((0, 3), dtype=np.float32)
            root_world_R = np.empty((0, 3, 3), dtype=np.float32)
        payload.update(
            {
                "verts_world": verts_world,
                "faces": faces_cache[beta_dim],
                "pelvis_world": pelvis_world,
                "anchor_world": anchor_world,
                "root_orient_world": root_world_R,
            }
        )
        np.savez(smpl_path, **payload)


def bridge_pose_residual(T: np.ndarray, reset_pose: np.ndarray, continue_pose: np.ndarray) -> dict:
    bridged = T @ reset_pose
    return {
        "translation": float(np.linalg.norm(bridged[:3, 3] - continue_pose[:3, 3])),
        "rotation_deg": rotation_error_deg(bridged, continue_pose),
    }


def transform_difference(a: np.ndarray, b: np.ndarray) -> dict:
    delta = a @ np.linalg.inv(b)
    identity = np.eye(4, dtype=np.float32)
    return {
        "translation": float(np.linalg.norm(delta[:3, 3])),
        "rotation_deg": rotation_error_deg(delta, identity),
    }


def write_per_frame_csv(path: Path, variants: list[dict], rpe: dict[str, dict]) -> None:
    rows = []
    for item in variants:
        rpe_by_offset = {row["offset"]: row for row in rpe[item["name"]]["per_frame"]}
        for row in item["per_frame"]:
            rel = rpe_by_offset[row["offset"]]
            rows.append(
                {
                    "variant": item["name"],
                    "idx": row["idx"],
                    "offset": row["offset"],
                    "camera_t_error": row["camera_t_error"],
                    "camera_r_error_deg": row["camera_r_error_deg"],
                    "rpe_t_error": rel["rpe_t_error"],
                    "rpe_r_error_deg": rel["rpe_r_error_deg"],
                    "world_root_gap_to_pre_boundary": row["world_root_gap_to_pre_boundary"],
                    "pointmap_chamfer_to_pre_boundary": row["pointmap_chamfer_to_pre_boundary"],
                }
            )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(output_dir: Path, variants: list[dict], rpe: dict[str, dict], dirs: dict[str, Path], cut_idx: int, frame_count: int) -> dict:
    analysis = output_dir / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    saved = {}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for item in variants:
        name = item["name"]
        rows = item["per_frame"]
        rel = rpe[name]["per_frame"]
        xs = [row["offset"] for row in rows]
        axes[0, 0].plot(xs, [row["camera_t_error"] for row in rows], marker="o", label=name)
        axes[0, 1].plot(xs, [row["camera_r_error_deg"] for row in rows], marker="o", label=name)
        axes[1, 0].plot(xs, [row["rpe_t_error"] for row in rel], marker="o", label=name)
        axes[1, 1].plot(xs, [row["rpe_r_error_deg"] for row in rel], marker="o", label=name)
    titles = (
        "Camera translation error",
        "Camera rotation error (deg)",
        "Gauge-free RPE translation",
        "Gauge-free RPE rotation (deg)",
    )
    for ax, title in zip(axes.reshape(-1), titles):
        ax.set_title(title)
        ax.set_xlabel("post-cut offset")
        ax.grid(True, alpha=0.25)
    axes[0, 1].legend(fontsize=7)
    fig.tight_layout()
    path = analysis / "bridge_camera_error_curves.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved["error_curves"] = str(path)

    fig = plt.figure(figsize=(13, 6))
    ax_cam = fig.add_subplot(1, 2, 1, projection="3d")
    ax_human = fig.add_subplot(1, 2, 2, projection="3d")
    for name, output_path in dirs.items():
        poses = [load_pose(output_path, idx)[0] for idx in range(frame_count)]
        cameras = np.stack([pose[:3, 3] for pose in poses])
        roots = np.stack([mean_root(output_path, idx) for idx in range(frame_count)])
        ax_cam.plot(cameras[:, 0], cameras[:, 1], cameras[:, 2], marker="o", markersize=2, label=name)
        ax_human.plot(roots[:, 0], roots[:, 1], roots[:, 2], marker="o", markersize=2, label=name)
        ax_cam.scatter(*cameras[cut_idx], s=30)
        ax_human.scatter(*roots[cut_idx], s=30)
    ax_cam.set_title("Camera trajectories")
    ax_human.set_title("Mean human-root trajectories")
    for ax in (ax_cam, ax_human):
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    ax_cam.legend(fontsize=7)
    fig.tight_layout()
    path = analysis / "camera_human_trajectories.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved["trajectories"] = str(path)

    ref = sample_points(dirs["A_raw_continue"], cut_idx - 1, 1800)
    fig = plt.figure(figsize=(15, 4))
    for panel, name in enumerate(("R_reset_raw", "D0_reset_camera_bridge", "D1_reset_pointmap_bridge"), start=1):
        ax = fig.add_subplot(1, 3, panel, projection="3d")
        current = sample_points(dirs[name], cut_idx, 1800)
        ax.scatter(ref[:, 0], ref[:, 1], ref[:, 2], s=1, alpha=0.35, label="pre-cut")
        ax.scatter(current[:, 0], current[:, 1], current[:, 2], s=1, alpha=0.35, label=name)
        ax.set_title(name)
        ax.legend(fontsize=7)
    fig.tight_layout()
    path = analysis / "boundary_pointcloud_stitch.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved["boundary_pointcloud"] = str(path)
    return saved


def drift_summary(item: dict, rpe_item: dict) -> dict:
    rows = item["per_frame"]
    rel = rpe_item["per_frame"]
    xs = np.asarray([row["offset"] for row in rows], dtype=np.float32)
    t_errors = np.asarray([row["camera_t_error"] for row in rows], dtype=np.float32)
    slope = float(np.polyfit(xs, t_errors, 1)[0]) if len(xs) > 1 else 0.0
    return {
        "boundary_camera_t_error": rows[0]["camera_t_error"],
        "boundary_camera_r_error_deg": rows[0]["camera_r_error_deg"],
        "last_camera_t_error": rows[-1]["camera_t_error"],
        "last_camera_r_error_deg": rows[-1]["camera_r_error_deg"],
        "camera_t_error_linear_slope_per_frame": slope,
        "last_rpe_t_error": rel[-1]["rpe_t_error"],
        "last_rpe_r_error_deg": rel[-1]["rpe_r_error_deg"],
    }


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# V10 双状态同帧桥接实验",
        "",
        "本实验不训练网络，GT 只用于评测。Continue 与 Reset 在 cut 后第一帧处理同一张 RGB，桥接变换只计算一次并固定用于整个后续 shot。",
        "",
        "## 数据与因果约束",
        "",
        f"- 数据：AvatarReX `{report['case']['seq_a']}` -> `{report['case']['seq_b']}`",
        f"- cut_idx：`{report['case']['cut_idx']}`",
        "- Camera Bridge：只读取边界同帧的 Continue/Reset camera-to-world pose。",
        "- Pointmap Bridge：只读取边界同帧的同像素背景高置信 3D 对应，不使用 ICP、BA 或未来帧。",
        "- cut 后 Continue 新 state 被丢弃，后续 shot 的局部输出全部来自 fresh Reset state。",
        "",
        "## Camera 指标",
        "",
        "| Variant | Cam T mean ↓ | Cam R mean ↓ | Boundary T ↓ | Boundary R ↓ | Last T ↓ | Last R ↓ | RPE T mean ↓ | RPE R mean ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["variants"]:
        rpe = report["rpe"][item["name"]]
        drift = report["drift"][item["name"]]
        lines.append(
            f"| {item['name']} | {item['mean_camera_t_error']:.4f} | {item['mean_camera_r_error_deg']:.2f} | "
            f"{drift['boundary_camera_t_error']:.4f} | {drift['boundary_camera_r_error_deg']:.2f} | "
            f"{drift['last_camera_t_error']:.4f} | {drift['last_camera_r_error_deg']:.2f} | "
            f"{rpe['mean_rpe_t_error']:.4f} | {rpe['mean_rpe_r_error_deg']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Bridge 自检",
            "",
            f"- Camera Bridge 边界同帧残差：`{report['bridge_checks']['camera_bridge_boundary_residual']['translation']:.8f} m / {report['bridge_checks']['camera_bridge_boundary_residual']['rotation_deg']:.8f} deg`。",
            f"- Pointmap Bridge 固定对应内点：`{report['pointmap_bridge_diagnostics']['final_inliers']}/{report['pointmap_bridge_diagnostics']['sampled_pairs']}`。",
            f"- Pointmap Bridge 内点平均残差：`{report['pointmap_bridge_diagnostics']['residual_inlier_mean']:.4f} m`。",
            f"- D0/D1 变换差：`{report['bridge_checks']['pointmap_vs_camera_transform']['translation']:.4f} m / {report['bridge_checks']['pointmap_vs_camera_transform']['rotation_deg']:.2f} deg`。",
            "",
            "## 输出",
            "",
            "- 完整指标：`dual_state_same_frame_bridge_metrics.json`",
            "- 逐帧指标：`analysis/per_frame_bridge_metrics.csv`",
            "- Camera/RPE 曲线：`analysis/bridge_camera_error_curves.png`",
            "- 相机与人体轨迹：`analysis/camera_human_trajectories.png`",
            "- 边界点云拼接：`analysis/boundary_pointcloud_stitch.png`",
            "",
            "## 解释原则",
            "",
            "- D0 若明显优于 A，说明旧 state 的边界输出确实保留了可用于重接世界坐标的参考。",
            "- D0 若边界准确且后续 RPE 接近 fresh Reset/C，说明一次性 bridge + 新 shot state 可以同时保留旧世界 gauge 和干净局部轨迹。",
            "- D1 若优于 D0，说明同帧背景 pointmap 能补充 camera token 的边界误差；若更差，则 camera bridge 是更可靠的最小方案。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    manifest = require_base_probe(args.base_probe_dir)
    case = manifest["case"]
    records = manifest["frames"]
    cut_idx = int(case["cut_idx"])
    post_frames = int(case["post_frames"])
    frame_count = cut_idx + post_frames

    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = args.output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    raw_continue = args.base_probe_dir / "A_raw_continue"
    raw_post = args.base_probe_dir / "raw_post_fresh_state"
    reset_raw = args.output_dir / "R_reset_raw"
    camera_bridge_dir = args.output_dir / "D0_reset_camera_bridge"
    pointmap_bridge_dir = args.output_dir / "D1_reset_pointmap_bridge"
    merge_args = SimpleNamespace(pre_frames=cut_idx, post_frames=post_frames)
    merge_reset_output(merge_args, raw_continue, raw_post, reset_raw)

    boundary_rgb_check = verify_same_boundary_rgb(args.base_probe_dir, cut_idx)
    save_boundary_payload(args.output_dir / "boundary" / "continue_boundary_payload.npz", raw_continue, cut_idx, args)
    save_boundary_payload(args.output_dir / "boundary" / "reset_boundary_payload.npz", raw_post, 0, args)

    continue_boundary, _ = load_pose(raw_continue, cut_idx)
    reset_boundary, _ = load_pose(raw_post, 0)
    camera_bridge = (continue_boundary @ np.linalg.inv(reset_boundary)).astype(np.float32)
    pointmap_bridge, pointmap_diag = estimate_pointmap_bridge(raw_continue, cut_idx, raw_post, 0, args)

    apply_bridge_to_shot(reset_raw, camera_bridge_dir, camera_bridge, cut_idx, frame_count)
    apply_bridge_to_shot(reset_raw, pointmap_bridge_dir, pointmap_bridge, cut_idx, frame_count)
    add_world_smpl_payload(camera_bridge_dir, frame_count, args.smpl_device)
    add_world_smpl_payload(pointmap_bridge_dir, frame_count, args.smpl_device)

    target_poses = build_target_poses(args.raw_meta_root, records, raw_continue)
    dirs: dict[str, Path] = {
        "A_raw_continue": raw_continue,
        "R_reset_raw": reset_raw,
        "D0_reset_camera_bridge": camera_bridge_dir,
        "D1_reset_pointmap_bridge": pointmap_bridge_dir,
    }
    oracle_reset = args.base_probe_dir / "C_reset_oracle_output"
    if (oracle_reset / "camera").is_dir():
        dirs["C_reset_oracle_output"] = oracle_reset

    variants = [
        summarize_variant(name, output_dir, target_poses, cut_idx, frame_count, args.max_metric_points)
        for name, output_dir in dirs.items()
    ]
    rpe = {name: summarize_rpe(output_dir, target_poses, cut_idx, frame_count) for name, output_dir in dirs.items()}
    success = {item["name"]: threshold_stats(item["per_frame"]) for item in variants}
    drift = {item["name"]: drift_summary(item, rpe[item["name"]]) for item in variants}
    write_per_frame_csv(analysis_dir / "per_frame_bridge_metrics.csv", variants, rpe)
    plots = plot_results(args.output_dir, variants, rpe, dirs, cut_idx, frame_count)

    report = {
        "case": case,
        "constraints": {
            "trained_model": False,
            "gt_used_for_bridge": False,
            "future_frames_used": False,
            "per_frame_bridge_update": False,
            "trajectory_smoothing": False,
            "continue_state_after_cut_discarded": True,
            "saved_camera_pose_convention": "camera-to-world",
        },
        "boundary_rgb_check": boundary_rgb_check,
        "variant_dirs": {name: str(path) for name, path in dirs.items()},
        "camera_bridge_se3": camera_bridge.tolist(),
        "pointmap_bridge_se3": pointmap_bridge.tolist(),
        "pointmap_bridge_diagnostics": pointmap_diag,
        "bridge_checks": {
            "camera_bridge_boundary_residual": bridge_pose_residual(camera_bridge, reset_boundary, continue_boundary),
            "pointmap_bridge_boundary_camera_residual": bridge_pose_residual(pointmap_bridge, reset_boundary, continue_boundary),
            "pointmap_vs_camera_transform": transform_difference(pointmap_bridge, camera_bridge),
        },
        "variants": variants,
        "rpe": rpe,
        "threshold_success": success,
        "drift": drift,
        "plots": plots,
        "notes": [
            "GT camera poses are aligned to Human3R frame-0 gauge and used only after all bridge transforms are fixed.",
            "Changing saved c2w left-transforms the reconstructed world pointmap and camera-space SMPL-X consistently.",
            "D0/D1 additionally cache corrected world pelvis, root orientation, and SMPL-X mesh for audit and visualization.",
        ],
    }
    metrics_path = args.output_dir / "dual_state_same_frame_bridge_metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(args.output_dir / "dual_state_same_frame_bridge_metrics.md", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
