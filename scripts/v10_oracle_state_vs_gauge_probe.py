#!/usr/bin/env python3
"""Oracle diagnostic: final gauge error vs recurrent-state pollution.

This probe builds a real RGB A/B cut sequence, runs strict original Human3R in
two ways, and then applies the same style of oracle boundary SE(3) correction at
the output level:

* A_raw_continue: all frames run in one continuous Human3R recurrent state.
* B_continue_oracle_output: same continuous state, but post-cut outputs are
  transformed by an oracle boundary SE(3).
* C_reset_oracle_output: post-cut frames are reconstructed from a fresh
  Human3R state, then transformed by an oracle boundary SE(3).

If B and C are close, the main error is likely final gauge.  If C is clearly
better after the cut, resetting/forking the recurrent state is important.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from add_ckpt_path import add_path_to_dust3r  # noqa: E402
from demo import prepare_input, prepare_output  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data"))
    parser.add_argument("--raw_meta_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data/AvatarReX_raw_meta/lbn1"))
    parser.add_argument("--split", default="Training")
    parser.add_argument("--seq_a", default="lbn1/22053926")
    parser.add_argument("--seq_b", default="lbn1/22010716")
    parser.add_argument("--start_frame", type=int, default=1192)
    parser.add_argument("--pre_frames", type=int, default=10)
    parser.add_argument("--post_frames", type=int, default=11)
    parser.add_argument("--model_path", type=Path, default=REPO_ROOT / "src" / "human3r_896L.pth")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=REPO_ROOT / "output" / "v10_oracle_state_vs_gauge_probe" / "avatarrex_lbn1_1192_cut10",
    )
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_points", type=int, default=1200)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_inference", action="store_true")
    return parser.parse_args()


def clean_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def output_complete(path: Path, expected: int) -> bool:
    return (path / "camera").is_dir() and len(list((path / "camera").glob("*.npz"))) == expected


def copy_frame(src: Path, dst: Path) -> None:
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(src)
    cv2.imwrite(str(dst), img)


def build_inputs(args: argparse.Namespace) -> list[dict]:
    input_all = args.output_dir / "input_all"
    input_post = args.output_dir / "input_post"
    clean_dir(input_all, True)
    clean_dir(input_post, True)

    records = []
    idx = 0
    for local_i in range(args.pre_frames):
        frame = args.start_frame + local_i
        src = args.data_root / args.split / args.seq_a / "rgb" / f"{frame:08d}.png"
        dst = input_all / f"{idx:06d}_A_{args.seq_a.replace('/', '-')}_{frame:08d}.png"
        copy_frame(src, dst)
        records.append({"idx": idx, "segment": "A", "seq": args.seq_a, "frame": frame, "path": str(dst)})
        idx += 1
    for local_i in range(args.post_frames):
        frame = args.start_frame + args.pre_frames + local_i
        src = args.data_root / args.split / args.seq_b / "rgb" / f"{frame:08d}.png"
        dst_all = input_all / f"{idx:06d}_B_{args.seq_b.replace('/', '-')}_{frame:08d}.png"
        dst_post = input_post / f"{local_i:06d}_B_{args.seq_b.replace('/', '-')}_{frame:08d}.png"
        copy_frame(src, dst_all)
        copy_frame(src, dst_post)
        records.append({"idx": idx, "segment": "B", "seq": args.seq_b, "frame": frame, "path": str(dst_all)})
        idx += 1
    return records


def run_human3r(args: argparse.Namespace, input_dir: Path, output_dir: Path, expected: int) -> None:
    if args.skip_inference and output_complete(output_dir, expected):
        return
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    add_path_to_dust3r(str(args.model_path))
    from src.dust3r.inference import inference_recurrent_lighter
    from src.dust3r.model import ARCroco3DStereo

    img_paths = sorted(str(p) for p in input_dir.glob("*.png"))
    if len(img_paths) != expected:
        raise RuntimeError(f"{input_dir} has {len(img_paths)} images, expected {expected}")
    model = ARCroco3DStereo.from_pretrained(str(args.model_path)).to(args.device).eval()

    disabled = []
    for name in (
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
    ):
        if hasattr(model, name):
            setattr(model, name, False)
            disabled.append(name)
    if hasattr(model, "downstream_head"):
        try:
            from src.dust3r.v8_head_lora import set_lora_enabled

            if hasattr(model.downstream_head, "pose_head"):
                set_lora_enabled(model.downstream_head.pose_head, False)
            for attr in ("deccam", "decpose", "decshape", "decexpression"):
                if hasattr(model.downstream_head, attr):
                    set_lora_enabled(getattr(model.downstream_head, attr), False)
        except Exception as exc:  # pragma: no cover - diagnostic path only.
            print(f"Warning: could not disable LoRA cleanly: {exc}")
    print(f"Running strict Human3R on {input_dir} -> {output_dir}; disabled={disabled}")

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
    with torch.no_grad():
        outputs, _ = inference_recurrent_lighter(views, model, args.device, use_ttt3r=False)
    prepare_output(
        outputs,
        str(output_dir),
        revisit=1,
        use_pose=True,
        save=True,
        render=False,
        render_video=False,
        img_res=img_res,
        subsample=1,
    )


def load_gt_c2w(raw_root: Path, seq: str) -> np.ndarray:
    with (raw_root / "calibration_full.json").open("r", encoding="utf-8") as f:
        calibration = json.load(f)
    key = seq if seq in calibration else seq.split("/")[-1]
    cal = calibration[key]
    R_w2c = np.asarray(cal["R"], dtype=np.float32).reshape(3, 3)
    T_w2c = np.asarray(cal["T"], dtype=np.float32).reshape(3)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R_w2c.T
    c2w[:3, 3] = -R_w2c.T @ T_w2c
    return c2w


def load_pose(output_dir: Path, idx: int) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(output_dir / "camera" / f"{idx:06d}.npz")
    return data["pose"].astype(np.float32), data["intrinsics"].astype(np.float32)


def write_pose(output_dir: Path, idx: int, pose: np.ndarray, K: np.ndarray) -> None:
    np.savez(output_dir / "camera" / f"{idx:06d}.npz", pose=pose.astype(np.float32), intrinsics=K.astype(np.float32))


def copy_payload_frame(src_dir: Path, src_idx: int, dst_dir: Path, dst_idx: int) -> None:
    for name, suffix in (("camera", ".npz"), ("smpl", ".npz"), ("depth", ".npy"), ("conf", ".npy"), ("color", ".png")):
        src = src_dir / name / f"{src_idx:06d}{suffix}"
        dst = dst_dir / name / f"{dst_idx:06d}{suffix}"
        shutil.copyfile(src, dst)


def merge_reset_output(args: argparse.Namespace, all_dir: Path, post_dir: Path, out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for name in ("camera", "smpl", "depth", "conf", "color"):
        (out_dir / name).mkdir(parents=True, exist_ok=True)
    for idx in range(args.pre_frames):
        copy_payload_frame(all_dir, idx, out_dir, idx)
    for local_i in range(args.post_frames):
        copy_payload_frame(post_dir, local_i, out_dir, args.pre_frames + local_i)


def copy_output_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def oracle_gt_poses(args: argparse.Namespace, records: list[dict], reference_raw_dir: Path) -> list[np.ndarray]:
    gt = [load_gt_c2w(args.raw_meta_root, r["seq"]) for r in records]
    raw0, _ = load_pose(reference_raw_dir, 0)
    align = raw0 @ np.linalg.inv(gt[0])
    return [(align @ g).astype(np.float32) for g in gt]


def apply_boundary_oracle(
    src_dir: Path,
    dst_dir: Path,
    target_poses: list[np.ndarray],
    cut_idx: int,
) -> np.ndarray:
    copy_output_tree(src_dir, dst_dir)
    raw_boundary, _ = load_pose(src_dir, cut_idx)
    T = target_poses[cut_idx] @ np.linalg.inv(raw_boundary)
    for idx in range(cut_idx, len(target_poses)):
        pose, K = load_pose(src_dir, idx)
        write_pose(dst_dir, idx, T @ pose, K)
    return T.astype(np.float32)


def rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    rel = a[:3, :3].T @ b[:3, :3]
    cos = np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def inverse_pose(pose: np.ndarray) -> np.ndarray:
    inv = np.eye(4, dtype=np.float32)
    inv[:3, :3] = pose[:3, :3].T
    inv[:3, 3] = -pose[:3, :3].T @ pose[:3, 3]
    return inv


def compose_transform(R: np.ndarray, t: np.ndarray, scale: float = 1.0) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = scale * R.astype(np.float32)
    T[:3, 3] = t.astype(np.float32)
    return T


def load_pose_list(output_dir: Path, count: int) -> list[np.ndarray]:
    return [load_pose(output_dir, idx)[0] for idx in range(count)]


def pose_centers(poses: list[np.ndarray]) -> np.ndarray:
    return np.stack([p[:3, 3] for p in poses], axis=0).astype(np.float32)


def trajectory_extent(points: np.ndarray) -> float:
    if len(points) == 0:
        return 0.0
    center = points.mean(axis=0, keepdims=True)
    return float(np.linalg.norm(points - center, axis=1).max())


def pose_landmarks(poses: list[np.ndarray], axis_radius: float) -> np.ndarray:
    """Camera centers plus local basis endpoints for non-degenerate pose alignment."""
    pts = []
    for pose in poses:
        center = pose[:3, 3]
        axes = pose[:3, :3]
        pts.append(center)
        for axis_i in range(3):
            pts.append(center + axis_radius * axes[:, axis_i])
    return np.stack(pts, axis=0).astype(np.float32)


def solve_se3_points(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve R, t with R @ src + t ~= dst."""
    src_mean = src.mean(axis=0, keepdims=True)
    dst_mean = dst.mean(axis=0, keepdims=True)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    H = src_c.T @ dst_c
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = dst_mean.reshape(3) - R @ src_mean.reshape(3)
    return R.astype(np.float32), t.astype(np.float32)


def solve_sim3_points(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Umeyama-style similarity solve with scale * R @ src + t ~= dst."""
    src_mean = src.mean(axis=0, keepdims=True)
    dst_mean = dst.mean(axis=0, keepdims=True)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    H = src_c.T @ dst_c / max(1, len(src))
    U, S, Vt = np.linalg.svd(H)
    D = np.eye(3, dtype=np.float32)
    if np.linalg.det(Vt.T @ U.T) < 0:
        D[-1, -1] = -1.0
    R = Vt.T @ D @ U.T
    var_src = float((src_c * src_c).sum() / max(1, len(src)))
    scale = float(np.trace(np.diag(S) @ D) / max(var_src, 1e-8))
    t = dst_mean.reshape(3) - scale * R @ src_mean.reshape(3)
    return scale, R.astype(np.float32), t.astype(np.float32)


def apply_left_se3_to_pose(pose: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = pose.copy()
    out[:3, :3] = R @ pose[:3, :3]
    out[:3, 3] = R @ pose[:3, 3] + t
    return out.astype(np.float32)


def apply_left_sim3_to_pose(pose: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = pose.copy()
    # Camera orientation stays rotational. Translation is in the aligned Sim(3)
    # coordinate system. This is diagnostic only, not a physically valid camera
    # extrinsic replacement for rendering.
    out[:3, :3] = R @ pose[:3, :3]
    out[:3, 3] = scale * (R @ pose[:3, 3]) + t
    return out.astype(np.float32)


def camera_errors(output_dir: Path, target_poses: list[np.ndarray], idx: int) -> tuple[float, float]:
    pose, _ = load_pose(output_dir, idx)
    return float(np.linalg.norm(pose[:3, 3] - target_poses[idx][:3, 3])), rotation_error_deg(pose, target_poses[idx])


def smpl_world_roots(output_dir: Path, idx: int) -> np.ndarray:
    pose, _ = load_pose(output_dir, idx)
    smpl = np.load(output_dir / "smpl" / f"{idx:06d}.npz")
    transl = np.asarray(smpl["transl"], dtype=np.float32).reshape(-1, 3)
    return transl @ pose[:3, :3].T + pose[:3, 3]


def mean_root(output_dir: Path, idx: int) -> np.ndarray:
    roots = smpl_world_roots(output_dir, idx)
    return roots.mean(axis=0)


def sample_points(output_dir: Path, idx: int, max_points: int) -> np.ndarray:
    pose, K = load_pose(output_dir, idx)
    depth = np.load(output_dir / "depth" / f"{idx:06d}.npy").astype(np.float32)
    conf = np.load(output_dir / "conf" / f"{idx:06d}.npy").astype(np.float32)
    valid_conf = conf[np.isfinite(conf)]
    threshold = float(np.quantile(valid_conf, 0.70)) if valid_conf.size else 0.0
    valid = np.isfinite(depth) & (depth > 0.05) & (depth < 50.0) & np.isfinite(conf) & (conf >= threshold)
    ys, xs = np.where(valid)
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    rng = np.random.default_rng(20260715 + idx)
    if len(xs) > max_points:
        keep = rng.choice(len(xs), size=max_points, replace=False)
        ys, xs = ys[keep], xs[keep]
    z = depth[ys, xs]
    x = (xs.astype(np.float32) - K[0, 2]) / K[0, 0] * z
    y = (ys.astype(np.float32) - K[1, 2]) / K[1, 1] * z
    pts_cam = np.stack([x, y, z], axis=-1)
    return (pts_cam @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32)


def chamfer_proxy(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) == 0 or len(b) == 0:
        return None
    ta = torch.from_numpy(a.astype(np.float32))
    tb = torch.from_numpy(b.astype(np.float32))
    d = torch.cdist(ta, tb)
    return float(0.5 * (d.min(dim=1).values.mean() + d.min(dim=0).values.mean()).item())


def id_consistency_proxy(output_dir: Path, start: int, end: int) -> dict:
    counts = []
    nearest_same_index = []
    prev = None
    for idx in range(start, end):
        roots = smpl_world_roots(output_dir, idx)
        counts.append(int(len(roots)))
        if prev is not None and len(prev) and len(roots):
            d = np.linalg.norm(prev[:, None, :] - roots[None, :, :], axis=-1)
            assignment = d.argmin(axis=1)
            same = [int(j == assignment[j]) for j in range(min(len(assignment), len(roots)))]
            nearest_same_index.extend(same)
        prev = roots
    return {
        "counts": counts,
        "constant_count": bool(len(set(counts)) == 1),
        "nearest_same_index_rate": float(np.mean(nearest_same_index)) if nearest_same_index else None,
    }


def summarize_variant(
    name: str,
    output_dir: Path,
    target_poses: list[np.ndarray],
    cut_idx: int,
    end_idx: int,
    max_points: int,
) -> dict:
    ref_root = mean_root(output_dir, cut_idx - 1)
    ref_points = sample_points(output_dir, cut_idx - 1, max_points)
    rows = []
    for idx in range(cut_idx, end_idx):
        cam_t, cam_r = camera_errors(output_dir, target_poses, idx)
        root = mean_root(output_dir, idx)
        pts = sample_points(output_dir, idx, max_points)
        rows.append(
            {
                "idx": idx,
                "offset": idx - cut_idx,
                "camera_t_error": cam_t,
                "camera_r_error_deg": cam_r,
                "world_root_gap_to_pre_boundary": float(np.linalg.norm(root - ref_root)),
                "pointmap_chamfer_to_pre_boundary": chamfer_proxy(ref_points, pts),
            }
        )

    def mean_key(key: str) -> float | None:
        vals = [r[key] for r in rows if r[key] is not None]
        return float(np.mean(vals)) if vals else None

    # Boundary frame itself can be exactly matched by oracle SE(3), so recovery
    # should mean "from here onward the post-cut trajectory stays stable".
    recovery = None
    recovery_t_threshold = 0.05
    recovery_r_threshold = 1.0
    for pos, row in enumerate(rows):
        tail = rows[pos:]
        if all(r["camera_t_error"] < recovery_t_threshold and r["camera_r_error_deg"] < recovery_r_threshold for r in tail):
            recovery = int(row["offset"])
            break

    return {
        "name": name,
        "dir": str(output_dir),
        "cut_window": [cut_idx, end_idx - 1],
        "mean_camera_t_error": mean_key("camera_t_error"),
        "mean_camera_r_error_deg": mean_key("camera_r_error_deg"),
        "mean_world_root_gap_to_pre_boundary": mean_key("world_root_gap_to_pre_boundary"),
        "mean_pointmap_chamfer_to_pre_boundary": mean_key("pointmap_chamfer_to_pre_boundary"),
        "state_recovery_offset_camera_threshold": recovery,
        "state_recovery_threshold": {
            "camera_t_error": recovery_t_threshold,
            "camera_r_error_deg": recovery_r_threshold,
            "definition": "first post-cut offset whose full tail remains below both thresholds",
        },
        "id_consistency": id_consistency_proxy(output_dir, cut_idx - 1, end_idx),
        "per_frame": rows,
    }


def threshold_stats(rows: list[dict], t_threshold: float = 0.05, r_threshold: float = 1.0) -> dict:
    success = [
        bool(row["camera_t_error"] < t_threshold and row["camera_r_error_deg"] < r_threshold)
        for row in rows
    ]
    longest = 0
    current = 0
    for ok in success:
        current = current + 1 if ok else 0
        longest = max(longest, current)
    probes = {}
    for off in (0, 1, 2, 4, 8, 10, 16):
        match = next((row for row in rows if row["offset"] == off), None)
        if match is not None:
            probes[str(off)] = {
                "camera_t_error": match["camera_t_error"],
                "camera_r_error_deg": match["camera_r_error_deg"],
                "success": bool(match["camera_t_error"] < t_threshold and match["camera_r_error_deg"] < r_threshold),
            }
    return {
        "threshold": {"camera_t_error": t_threshold, "camera_r_error_deg": r_threshold},
        "success_rate": float(np.mean(success)) if success else None,
        "longest_success_run": int(longest),
        "probe_offsets": probes,
    }


def rpe_rows(output_dir: Path, target_poses: list[np.ndarray], cut_idx: int, end_idx: int) -> list[dict]:
    pred_cut, _ = load_pose(output_dir, cut_idx)
    tgt_cut = target_poses[cut_idx]
    inv_pred_cut = inverse_pose(pred_cut)
    inv_tgt_cut = inverse_pose(tgt_cut)
    rows = []
    for idx in range(cut_idx, end_idx):
        pred, _ = load_pose(output_dir, idx)
        pred_rel = inv_pred_cut @ pred
        tgt_rel = inv_tgt_cut @ target_poses[idx]
        rows.append(
            {
                "idx": idx,
                "offset": idx - cut_idx,
                "rpe_t_error": float(np.linalg.norm(pred_rel[:3, 3] - tgt_rel[:3, 3])),
                "rpe_r_error_deg": rotation_error_deg(pred_rel, tgt_rel),
            }
        )
    return rows


def summarize_rpe(output_dir: Path, target_poses: list[np.ndarray], cut_idx: int, end_idx: int) -> dict:
    rows = rpe_rows(output_dir, target_poses, cut_idx, end_idx)
    return {
        "mean_rpe_t_error": float(np.mean([r["rpe_t_error"] for r in rows])),
        "mean_rpe_r_error_deg": float(np.mean([r["rpe_r_error_deg"] for r in rows])),
        "max_rpe_t_error": float(np.max([r["rpe_t_error"] for r in rows])),
        "max_rpe_r_error_deg": float(np.max([r["rpe_r_error_deg"] for r in rows])),
        "per_frame": rows,
    }


def best_shot_alignment_summary(output_dir: Path, target_poses: list[np.ndarray], cut_idx: int, end_idx: int) -> dict:
    poses = load_pose_list(output_dir, end_idx)
    pred_post = poses[cut_idx:end_idx]
    tgt_post = target_poses[cut_idx:end_idx]
    pred_centers = pose_centers(pred_post)
    tgt_centers = pose_centers(tgt_post)

    pred_extent = trajectory_extent(pred_centers)
    tgt_extent = trajectory_extent(tgt_centers)
    axis_radius = max(0.25, pred_extent, tgt_extent)
    pred_pose_points = pose_landmarks(pred_post, axis_radius)
    tgt_pose_points = pose_landmarks(tgt_post, axis_radius)

    # Center-only fitting is reported as a degeneracy reference.  When a shot is
    # almost static, centers alone cannot determine rotation or scale, so the
    # main diagnostic uses pose landmarks.
    R_center_se3, t_center_se3 = solve_se3_points(pred_centers, tgt_centers)
    scale_center_sim3, R_center_sim3, t_center_sim3 = solve_sim3_points(pred_centers, tgt_centers)
    R_se3, t_se3 = solve_se3_points(pred_pose_points, tgt_pose_points)
    scale_sim3, R_sim3, t_sim3 = solve_sim3_points(pred_pose_points, tgt_pose_points)

    def eval_aligned(kind: str, *, center_only: bool = False) -> dict:
        rows = []
        for local_i, idx in enumerate(range(cut_idx, end_idx)):
            pred = poses[idx]
            if kind == "se3" and center_only:
                aligned = apply_left_se3_to_pose(pred, R_center_se3, t_center_se3)
            elif kind == "sim3" and center_only:
                aligned = apply_left_sim3_to_pose(pred, scale_center_sim3, R_center_sim3, t_center_sim3)
            elif kind == "se3":
                aligned = apply_left_se3_to_pose(pred, R_se3, t_se3)
            elif kind == "sim3":
                aligned = apply_left_sim3_to_pose(pred, scale_sim3, R_sim3, t_sim3)
            else:
                raise ValueError(kind)
            tgt = target_poses[idx]
            rows.append(
                {
                    "idx": idx,
                    "offset": local_i,
                    "camera_t_error": float(np.linalg.norm(aligned[:3, 3] - tgt[:3, 3])),
                    "camera_r_error_deg": rotation_error_deg(aligned, tgt),
                }
            )
        return {
            "mean_camera_t_error": float(np.mean([r["camera_t_error"] for r in rows])),
            "mean_camera_r_error_deg": float(np.mean([r["camera_r_error_deg"] for r in rows])),
            "max_camera_t_error": float(np.max([r["camera_t_error"] for r in rows])),
            "max_camera_r_error_deg": float(np.max([r["camera_r_error_deg"] for r in rows])),
            "per_frame": rows,
        }

    return {
        "fit_points": "camera centers plus local right/up/forward endpoints",
        "axis_radius": axis_radius,
        "center_extent_pred": pred_extent,
        "center_extent_target": tgt_extent,
        "center_only_reference": {
            "best_se3": {
                "R": R_center_se3.tolist(),
                "t": t_center_se3.tolist(),
                **eval_aligned("se3", center_only=True),
            },
            "best_sim3": {
                "scale": scale_center_sim3,
                "R": R_center_sim3.tolist(),
                "t": t_center_sim3.tolist(),
                "degenerate": bool(scale_center_sim3 < 0.1 or scale_center_sim3 > 10.0 or min(pred_extent, tgt_extent) < 1e-4),
                **eval_aligned("sim3", center_only=True),
            },
        },
        "best_se3": {
            "R": R_se3.tolist(),
            "t": t_se3.tolist(),
            **eval_aligned("se3"),
        },
        "best_sim3": {
            "scale": scale_sim3,
            "R": R_sim3.tolist(),
            "t": t_sim3.tolist(),
            **eval_aligned("sim3"),
        },
    }


def sample_camera_points(output_dir: Path, idx: int, max_points: int) -> np.ndarray:
    _, K = load_pose(output_dir, idx)
    depth = np.load(output_dir / "depth" / f"{idx:06d}.npy").astype(np.float32)
    conf = np.load(output_dir / "conf" / f"{idx:06d}.npy").astype(np.float32)
    valid_conf = conf[np.isfinite(conf)]
    threshold = float(np.quantile(valid_conf, 0.70)) if valid_conf.size else 0.0
    valid = np.isfinite(depth) & (depth > 0.05) & (depth < 50.0) & np.isfinite(conf) & (conf >= threshold)
    ys, xs = np.where(valid)
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    rng = np.random.default_rng(20260716 + idx)
    if len(xs) > max_points:
        keep = rng.choice(len(xs), size=max_points, replace=False)
        ys, xs = ys[keep], xs[keep]
    z = depth[ys, xs]
    x = (xs.astype(np.float32) - K[0, 2]) / K[0, 0] * z
    y = (ys.astype(np.float32) - K[1, 2]) / K[1, 1] * z
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def frame_camera_local_stats(output_dir: Path, idx: int, max_points: int) -> dict:
    depth = np.load(output_dir / "depth" / f"{idx:06d}.npy").astype(np.float32)
    conf = np.load(output_dir / "conf" / f"{idx:06d}.npy").astype(np.float32)
    smpl = np.load(output_dir / "smpl" / f"{idx:06d}.npz")
    transl = np.asarray(smpl["transl"], dtype=np.float32).reshape(-1, 3)
    rotvec = np.asarray(smpl["rotvec"], dtype=np.float32)
    msk = np.asarray(smpl["msk"], dtype=np.float32)
    valid_depth = depth[np.isfinite(depth) & (depth > 0.05) & (depth < 50.0)]
    pts_cam = sample_camera_points(output_dir, idx, max_points)
    return {
        "num_people": int(len(transl)),
        "smpl_transl": transl,
        "mean_smpl_transl": transl.mean(axis=0) if len(transl) else np.zeros(3, dtype=np.float32),
        "root_centered_pose": rotvec[:, 1:, :].reshape(len(rotvec), -1) if len(rotvec) else np.zeros((0, 1), dtype=np.float32),
        "depth_median": float(np.median(valid_depth)) if valid_depth.size else None,
        "depth_mean": float(np.mean(valid_depth)) if valid_depth.size else None,
        "depth_std": float(np.std(valid_depth)) if valid_depth.size else None,
        "conf_mean": float(np.nanmean(conf)),
        "conf_p70": float(np.nanquantile(conf, 0.70)),
        "mask_area": float(msk.mean()) if msk.size else 0.0,
        "camera_points": pts_cam,
    }


def summarize_camera_frame_compare(
    continue_dir: Path,
    fresh_post_dir: Path,
    cut_idx: int,
    post_frames: int,
    max_points: int,
) -> dict:
    rows = []
    for off in range(post_frames):
        cont = frame_camera_local_stats(continue_dir, cut_idx + off, max_points)
        fresh = frame_camera_local_stats(fresh_post_dir, off, max_points)
        cont_pose = cont["root_centered_pose"]
        fresh_pose = fresh["root_centered_pose"]
        pose_l2 = None
        if len(cont_pose) and len(fresh_pose) and cont_pose.shape == fresh_pose.shape:
            pose_l2 = float(np.linalg.norm(cont_pose - fresh_pose, axis=1).mean())
        depth_ratio = None
        if cont["depth_median"] is not None and fresh["depth_median"] not in (None, 0.0):
            depth_ratio = float(cont["depth_median"] / fresh["depth_median"])
        rows.append(
            {
                "offset": off,
                "idx_continue": cut_idx + off,
                "idx_fresh": off,
                "num_people_continue": cont["num_people"],
                "num_people_fresh": fresh["num_people"],
                "smpl_camera_root_mean_l2": float(np.linalg.norm(cont["mean_smpl_transl"] - fresh["mean_smpl_transl"])),
                "root_centered_pose_l2": pose_l2,
                "camera_frame_point_chamfer": chamfer_proxy(cont["camera_points"], fresh["camera_points"]),
                "depth_median_continue": cont["depth_median"],
                "depth_median_fresh": fresh["depth_median"],
                "depth_median_ratio_continue_over_fresh": depth_ratio,
                "depth_mean_abs_diff": None
                if cont["depth_mean"] is None or fresh["depth_mean"] is None
                else float(abs(cont["depth_mean"] - fresh["depth_mean"])),
                "conf_mean_continue": cont["conf_mean"],
                "conf_mean_fresh": fresh["conf_mean"],
                "conf_mean_abs_diff": float(abs(cont["conf_mean"] - fresh["conf_mean"])),
                "mask_area_continue": cont["mask_area"],
                "mask_area_fresh": fresh["mask_area"],
                "mask_area_abs_diff": float(abs(cont["mask_area"] - fresh["mask_area"])),
            }
        )

    def mean_optional(key: str) -> float | None:
        vals = [r[key] for r in rows if r[key] is not None]
        return float(np.mean(vals)) if vals else None

    return {
        "meaning": "Compare post-cut old-state Human3R camera-frame outputs against fresh-state Human3R outputs before any world-gauge correction.",
        "mean_smpl_camera_root_l2": mean_optional("smpl_camera_root_mean_l2"),
        "mean_root_centered_pose_l2": mean_optional("root_centered_pose_l2"),
        "mean_camera_frame_point_chamfer": mean_optional("camera_frame_point_chamfer"),
        "mean_depth_median_ratio_continue_over_fresh": mean_optional("depth_median_ratio_continue_over_fresh"),
        "mean_depth_mean_abs_diff": mean_optional("depth_mean_abs_diff"),
        "mean_conf_mean_abs_diff": mean_optional("conf_mean_abs_diff"),
        "mean_mask_area_abs_diff": mean_optional("mask_area_abs_diff"),
        "per_frame": rows,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_advanced_csvs(output_dir: Path, advanced: dict) -> None:
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    for name, item in advanced["per_frame_error_curves"].items():
        write_csv(analysis_dir / f"{name}_per_frame_errors.csv", item)
    for name, item in advanced["rpe"].items():
        write_csv(analysis_dir / f"{name}_rpe.csv", item["per_frame"])
    write_csv(analysis_dir / "camera_frame_continue_vs_fresh.csv", advanced["camera_frame_compare"]["per_frame"])


def plot_advanced_curves(output_dir: Path, advanced: dict) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    saved = {}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, rows in advanced["per_frame_error_curves"].items():
        xs = [r["offset"] for r in rows]
        axes[0].plot(xs, [r["camera_t_error"] for r in rows], marker="o", label=name)
        axes[1].plot(xs, [r["camera_r_error_deg"] for r in rows], marker="o", label=name)
    axes[0].set_title("Per-frame camera translation error")
    axes[0].set_xlabel("post-cut offset")
    axes[0].set_ylabel("translation error")
    axes[0].grid(True, alpha=0.25)
    axes[1].set_title("Per-frame camera rotation error")
    axes[1].set_xlabel("post-cut offset")
    axes[1].set_ylabel("rotation error (deg)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    path = analysis_dir / "per_frame_camera_error_curves.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved["per_frame_camera_error_curves"] = str(path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, item in advanced["rpe"].items():
        rows = item["per_frame"]
        xs = [r["offset"] for r in rows]
        axes[0].plot(xs, [r["rpe_t_error"] for r in rows], marker="o", label=name)
        axes[1].plot(xs, [r["rpe_r_error_deg"] for r in rows], marker="o", label=name)
    axes[0].set_title("Gauge-free RPE translation")
    axes[0].set_xlabel("post-cut offset")
    axes[0].set_ylabel("RPE translation")
    axes[0].grid(True, alpha=0.25)
    axes[1].set_title("Gauge-free RPE rotation")
    axes[1].set_xlabel("post-cut offset")
    axes[1].set_ylabel("RPE rotation (deg)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    path = analysis_dir / "rpe_curves.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved["rpe_curves"] = str(path)

    rows = advanced["camera_frame_compare"]["per_frame"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    xs = [r["offset"] for r in rows]
    axes[0].plot(xs, [r["smpl_camera_root_mean_l2"] for r in rows], marker="o")
    axes[0].set_title("SMPL camera-root diff")
    axes[1].plot(xs, [r["camera_frame_point_chamfer"] for r in rows], marker="o")
    axes[1].set_title("Camera-frame point chamfer")
    axes[2].plot(xs, [r["depth_median_ratio_continue_over_fresh"] for r in rows], marker="o")
    axes[2].axhline(1.0, color="black", linewidth=1, alpha=0.5)
    axes[2].set_title("Depth median ratio")
    for ax in axes:
        ax.set_xlabel("post-cut offset")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = analysis_dir / "camera_frame_continue_vs_fresh.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved["camera_frame_continue_vs_fresh"] = str(path)
    return saved


def build_advanced_analysis(
    output_dir: Path,
    target_poses: list[np.ndarray],
    raw_continue: Path,
    oracle_continue: Path,
    oracle_reset: Path,
    raw_post: Path,
    variants: list[dict],
    cut_idx: int,
    end_idx: int,
    max_points: int,
) -> dict:
    variant_dirs = {
        "A_raw_continue": raw_continue,
        "B_continue_oracle_output": oracle_continue,
        "C_reset_oracle_output": oracle_reset,
    }
    per_frame = {item["name"]: item["per_frame"] for item in variants}
    success = {item["name"]: threshold_stats(item["per_frame"]) for item in variants}
    rpe = {
        name: summarize_rpe(path, target_poses, cut_idx, end_idx)
        for name, path in variant_dirs.items()
    }
    best = {
        name: best_shot_alignment_summary(path, target_poses, cut_idx, end_idx)
        for name, path in variant_dirs.items()
    }
    camera_frame = summarize_camera_frame_compare(
        raw_continue,
        raw_post,
        cut_idx,
        end_idx - cut_idx,
        max_points,
    )
    advanced = {
        "per_frame_error_curves": per_frame,
        "threshold_success": success,
        "rpe": rpe,
        "best_shot_alignment": best,
        "camera_frame_compare": camera_frame,
        "interpretation_hints": [
            "B should have near-zero boundary-frame global error because boundary oracle SE(3) is fitted at cut_idx.",
            "A and B should have identical RPE up to numerical noise because B is a constant left SE(3) transform of A after the cut.",
            "If C has lower RPE than A/B, reset/fresh state improves post-cut relative trajectory, not only global gauge.",
            "If best-shot SE(3)/Sim(3) cannot make B approach C, the old-state trajectory has time-varying shape error.",
            "Camera-frame comparison uses raw_continue vs raw_post_fresh_state before any output gauge correction.",
        ],
    }
    write_advanced_csvs(output_dir, advanced)
    advanced["plots"] = plot_advanced_curves(output_dir, advanced)
    return advanced


def write_markdown(path: Path, report: dict) -> None:
    advanced = report.get("advanced_analysis", {})
    lines = [
        "# V10 Oracle State vs Gauge Probe",
        "",
        "这个诊断用于区分：分镜后错误主要是最终坐标 gauge 错，还是 Human3R recurrent state 被 cut 污染。",
        "",
        "## 设置",
        "",
        f"- seqA: `{report['case']['seq_a']}`",
        f"- seqB: `{report['case']['seq_b']}`",
        f"- start_frame: `{report['case']['start_frame']}`",
        f"- cut index: `{report['case']['cut_idx']}`",
        f"- post window: `{report['case']['cut_idx']}..{report['case']['eval_end_idx'] - 1}`",
        "",
        "## 结果概览",
        "",
        "| Variant | Cam T ↓ | Cam R ↓ | Root Gap ↓ | Point Chamfer ↓ | Recovery | ID const | ID same-rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["variants"]:
        idc = item["id_consistency"]
        lines.append(
            "| {name} | {cam_t:.4f} | {cam_r:.2f} | {root:.4f} | {point} | {rec} | {const} | {rate} |".format(
                name=item["name"],
                cam_t=item["mean_camera_t_error"],
                cam_r=item["mean_camera_r_error_deg"],
                root=item["mean_world_root_gap_to_pre_boundary"],
                point="None" if item["mean_pointmap_chamfer_to_pre_boundary"] is None else f"{item['mean_pointmap_chamfer_to_pre_boundary']:.4f}",
                rec="None" if item["state_recovery_offset_camera_threshold"] is None else item["state_recovery_offset_camera_threshold"],
                const=idc["constant_count"],
                rate="None" if idc["nearest_same_index_rate"] is None else f"{idc['nearest_same_index_rate']:.2f}",
            )
        )
    lines.extend(
        [
            "",
            "## 逐帧和阈值统计",
            "",
            "阈值成功定义：`camera_t_error < 0.05` 且 `camera_r_error < 1 deg`。",
            "",
            "| Variant | Success Rate | Longest Success Run | off0 | off1 | off2 | off4 | off8 | off10 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, stat in advanced.get("threshold_success", {}).items():
        probes = stat["probe_offsets"]
        def fmt_probe(off: str) -> str:
            if off not in probes:
                return "-"
            p = probes[off]
            return f"{p['camera_t_error']:.3f}/{p['camera_r_error_deg']:.2f}"
        lines.append(
            "| {name} | {rate:.2f} | {run} | {p0} | {p1} | {p2} | {p4} | {p8} | {p10} |".format(
                name=name,
                rate=stat["success_rate"],
                run=stat["longest_success_run"],
                p0=fmt_probe("0"),
                p1=fmt_probe("1"),
                p2=fmt_probe("2"),
                p4=fmt_probe("4"),
                p8=fmt_probe("8"),
                p10=fmt_probe("10"),
            )
        )
    lines.extend(
        [
            "",
            "表中 `offN` 的格式是 `translation_error/rotation_error_deg`。",
            "",
            "## Gauge-free RPE",
            "",
            "| Variant | RPE T mean ↓ | RPE R mean ↓ | RPE T max ↓ | RPE R max ↓ |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, item in advanced.get("rpe", {}).items():
        lines.append(
            f"| {name} | {item['mean_rpe_t_error']:.4f} | {item['mean_rpe_r_error_deg']:.2f} | "
            f"{item['max_rpe_t_error']:.4f} | {item['max_rpe_r_error_deg']:.2f} |"
        )
    lines.extend(
        [
            "",
            "RPE 只看 cut 帧到后续帧的相对运动，不受整段 global gauge 影响。B 是 A 的固定左乘 SE(3) 后处理，所以 A/B 的 RPE 理论上应相同。",
            "",
            "## Best-shot Offline Alignment",
            "",
            "| Variant | Best SE3 Cam T ↓ | Best SE3 Cam R ↓ | Best Sim3 Scale | Best Sim3 Cam T ↓ | Best Sim3 Cam R ↓ |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, item in advanced.get("best_shot_alignment", {}).items():
        se3 = item["best_se3"]
        sim3 = item["best_sim3"]
        lines.append(
            f"| {name} | {se3['mean_camera_t_error']:.4f} | {se3['mean_camera_r_error_deg']:.2f} | "
            f"{sim3['scale']:.4f} | {sim3['mean_camera_t_error']:.4f} | {sim3['mean_camera_r_error_deg']:.2f} |"
        )
    cam_frame = advanced.get("camera_frame_compare", {})
    if cam_frame:
        lines.extend(
            [
                "",
                "## Camera-frame Continue vs Fresh",
                "",
                "这里比较的是 output gauge correction 之前的 post-cut Human3R 局部输出：`A_raw_continue` 的 B 段 vs `raw_post_fresh_state`。",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| mean_smpl_camera_root_l2 | {cam_frame['mean_smpl_camera_root_l2']:.4f} |",
                f"| mean_root_centered_pose_l2 | {cam_frame['mean_root_centered_pose_l2']:.4f} |",
                f"| mean_camera_frame_point_chamfer | {cam_frame['mean_camera_frame_point_chamfer']:.4f} |",
                f"| mean_depth_median_ratio_continue_over_fresh | {cam_frame['mean_depth_median_ratio_continue_over_fresh']:.4f} |",
                f"| mean_depth_mean_abs_diff | {cam_frame['mean_depth_mean_abs_diff']:.4f} |",
                f"| mean_conf_mean_abs_diff | {cam_frame['mean_conf_mean_abs_diff']:.4f} |",
                f"| mean_mask_area_abs_diff | {cam_frame['mean_mask_area_abs_diff']:.4f} |",
            ]
        )
    if advanced.get("plots"):
        lines.extend(["", "## 输出文件", ""])
        for key, value in sorted(advanced["plots"].items()):
            lines.append(f"- `{key}`: `{value}`")
        lines.extend(
            [
                "- per-frame/RPE/camera-frame CSV: `analysis/*.csv`",
                "",
            ]
        )
    lines.extend(
        [
            "",
            "## 判读",
            "",
            "- `A_raw_continue` 是原版 Human3R 遇到 RGB cut 后继续旧 state 的输出。",
            "- `B_continue_oracle_output` 不 reset state，只在最终输出层对 cut 后 segment 施加 boundary oracle SE(3)。",
            "- `C_reset_oracle_output` cut 后从 fresh Human3R state 重建，再施加同样类型的 boundary oracle SE(3)。",
            "- 如果 B 和 C 接近，主要问题偏最终 gauge；如果 C 明显优于 B，说明 reset/fork state 是必要的。",
            "- 当前脚本没有实现 D 的 latent/state write-back correction；D 需要改 Human3R 内部 state 写入路径。",
            "- 当前 6.5 的 `Read-old/write-fresh` 真版本需要 fork 两套 recurrent state：旧 state 只读，新 shot 写 fresh state。这个不能只靠 saved-output 后处理表达。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.model_path.is_file():
        raise FileNotFoundError(args.model_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = build_inputs(args)
    manifest = {
        "case": {
            "seq_a": args.seq_a,
            "seq_b": args.seq_b,
            "start_frame": args.start_frame,
            "pre_frames": args.pre_frames,
            "post_frames": args.post_frames,
            "cut_idx": args.pre_frames,
            "eval_end_idx": args.pre_frames + args.post_frames,
        },
        "frames": records,
    }
    (args.output_dir / "case_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    raw_continue = args.output_dir / "A_raw_continue"
    raw_post = args.output_dir / "raw_post_fresh_state"
    reset_merged = args.output_dir / "C_reset_raw_merged"
    run_human3r(args, args.output_dir / "input_all", raw_continue, args.pre_frames + args.post_frames)
    run_human3r(args, args.output_dir / "input_post", raw_post, args.post_frames)
    merge_reset_output(args, raw_continue, raw_post, reset_merged)

    target_poses = oracle_gt_poses(args, records, raw_continue)
    oracle_continue = args.output_dir / "B_continue_oracle_output"
    oracle_reset = args.output_dir / "C_reset_oracle_output"
    T_b = apply_boundary_oracle(raw_continue, oracle_continue, target_poses, args.pre_frames)
    T_c = apply_boundary_oracle(reset_merged, oracle_reset, target_poses, args.pre_frames)

    variants = [
        summarize_variant("A_raw_continue", raw_continue, target_poses, args.pre_frames, args.pre_frames + args.post_frames, args.max_points),
        summarize_variant(
            "B_continue_oracle_output",
            oracle_continue,
            target_poses,
            args.pre_frames,
            args.pre_frames + args.post_frames,
            args.max_points,
        ),
        summarize_variant(
            "C_reset_oracle_output",
            oracle_reset,
            target_poses,
            args.pre_frames,
            args.pre_frames + args.post_frames,
            args.max_points,
        ),
    ]
    advanced = build_advanced_analysis(
        args.output_dir,
        target_poses,
        raw_continue,
        oracle_continue,
        oracle_reset,
        raw_post,
        variants,
        args.pre_frames,
        args.pre_frames + args.post_frames,
        args.max_points,
    )
    report = {
        "case": manifest["case"],
        "oracle_boundary_transform_B": T_b.tolist(),
        "oracle_boundary_transform_C": T_c.tolist(),
        "variants": variants,
        "advanced_analysis": advanced,
        "notes": [
            "B uses the old recurrent state and only changes final post-cut camera/world gauge.",
            "C uses a fresh post-cut recurrent state, then applies final post-cut camera/world gauge correction.",
            "Root and pointmap metrics are proxy consistency metrics derived from saved Human3R outputs, not dataset GT root/pointmap supervision.",
        ],
    }
    (args.output_dir / "oracle_state_vs_gauge_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(args.output_dir / "oracle_state_vs_gauge_metrics.md", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
