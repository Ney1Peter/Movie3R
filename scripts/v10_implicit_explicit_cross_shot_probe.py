#!/usr/bin/env python3
"""Training-free implicit coarse plus explicit fine cross-shot probe.

This experiment keeps strict original Human3R frozen and leaves its default
inference code unchanged.  It compares:

* Original Human3R continue-old-state.
* Fresh reset without alignment.
* Explicit-only human/pointmap alignment.
* Implicit-only frozen token matching.
* Token-initialized explicit refinement.
* Fresh reset plus oracle boundary SE(3).

Only clean pre-cut history and fresh post-cut information are used by the
alignment methods.  Ground truth is used after inference for evaluation and
difficulty labeling only.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MethodType, SimpleNamespace

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from scipy.stats import spearmanr


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPT_ROOT = REPO_ROOT / "scripts"
for path in (str(REPO_ROOT), str(SRC_ROOT), str(SCRIPT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from add_ckpt_path import add_path_to_dust3r  # noqa: E402
from demo import prepare_input, prepare_output  # noqa: E402
from dust3r.model import ARCroco3DStereo  # noqa: E402
from dust3r.utils.device import to_cpu  # noqa: E402
from v10_dual_state_same_frame_bridge_probe import (  # noqa: E402
    add_world_smpl_payload,
    apply_bridge_to_shot,
    load_human_mask,
    transform_points,
    weighted_kabsch,
)
from v10_oracle_state_vs_gauge_probe import (  # noqa: E402
    load_gt_c2w,
    load_pose,
    merge_reset_output,
    rotation_error_deg,
    summarize_rpe,
    summarize_variant,
    threshold_stats,
)
from v9_online_stream_human3r_segment_align import strict_original_model  # noqa: E402


DEFAULT_CASE_MANIFEST = REPO_ROOT / "config" / "manifests" / "archive" / "v9_early" / "v9_small10_avatarrex_train_aabb_7.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data"))
    parser.add_argument("--raw_meta_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data/AvatarReX_raw_meta"))
    parser.add_argument("--case_manifest", type=Path, default=DEFAULT_CASE_MANIFEST)
    parser.add_argument("--model_path", type=Path, default=REPO_ROOT / "src" / "human3r_896L.pth")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=REPO_ROOT / "output" / "v10_implicit_explicit_cross_shot_probe" / "avatarrex_training_free",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--pre_frames", type=int, default=3)
    parser.add_argument("--post_frames", type=int, default=6)
    parser.add_argument("--history_window", type=int, default=3)
    parser.add_argument("--max_cases", type=int, default=6)
    parser.add_argument("--case_indices", type=int, nargs="*", default=None)
    parser.add_argument("--token_ransac_iters", type=int, default=512)
    parser.add_argument("--token_inlier_threshold", type=float, default=0.20)
    parser.add_argument("--token_min_matches", type=int, default=8)
    parser.add_argument("--cloud_points_per_frame", type=int, default=6000)
    parser.add_argument("--refine_iters", type=int, default=8)
    parser.add_argument("--refine_max_distance", type=float, default=0.60)
    parser.add_argument("--refine_min_distance", type=float, default=0.12)
    parser.add_argument("--smpl_device", default="cpu")
    parser.add_argument("--skip_inference", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


@dataclass
class AlignmentCandidate:
    name: str
    transform: np.ndarray
    confidence: float
    diagnostics: dict


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def normalize_record(record: dict, index: int) -> dict:
    record = dict(record)
    if "seqA" not in record and "seqs" in record:
        record["seqA"] = record["seqs"][0]
        record["seqB"] = record["seqs"][2]
    if "start_frame" not in record and "frames" in record:
        record["start_frame"] = int(record["frames"][0])
    record.setdefault("pattern_id", f"avatarrex_probe_{index:02d}")
    record.setdefault("group", str(record["seqA"]).split("/", 1)[0])
    return record


def select_cases(args: argparse.Namespace) -> list[dict]:
    records = [normalize_record(record, idx) for idx, record in enumerate(read_jsonl(args.case_manifest))]
    if args.case_indices:
        records = [records[idx] for idx in args.case_indices]
    return records[: int(args.max_cases)]


def image_path(args: argparse.Namespace, seq: str, frame: int) -> Path:
    path = args.data_root / "Training" / seq / "rgb" / f"{frame:08d}.png"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def build_case_inputs(args: argparse.Namespace, record: dict, case_dir: Path) -> list[dict]:
    input_all = case_dir / "input_all"
    input_post = case_dir / "input_post"
    if args.overwrite:
        shutil.rmtree(input_all, ignore_errors=True)
        shutil.rmtree(input_post, ignore_errors=True)
    input_all.mkdir(parents=True, exist_ok=True)
    input_post.mkdir(parents=True, exist_ok=True)
    records = []
    start = int(record["start_frame"])
    for idx in range(int(args.pre_frames)):
        frame = start + idx
        src = image_path(args, str(record["seqA"]), frame)
        dst = input_all / f"{idx:06d}_A_{frame:08d}.png"
        if not dst.is_file() or args.overwrite:
            shutil.copy2(src, dst)
        records.append({"idx": idx, "segment": "A", "seq": record["seqA"], "frame": frame, "path": str(dst)})
    for offset in range(int(args.post_frames)):
        idx = int(args.pre_frames) + offset
        frame = start + int(args.pre_frames) + offset
        src = image_path(args, str(record["seqB"]), frame)
        dst_all = input_all / f"{idx:06d}_B_{frame:08d}.png"
        dst_post = input_post / f"{offset:06d}_B_{frame:08d}.png"
        if not dst_all.is_file() or args.overwrite:
            shutil.copy2(src, dst_all)
        if not dst_post.is_file() or args.overwrite:
            shutil.copy2(src, dst_post)
        records.append({"idx": idx, "segment": "B", "seq": record["seqB"], "frame": frame, "path": str(dst_all)})
    return records


def output_complete(path: Path, expected: int) -> bool:
    return len(list((path / "camera").glob("*.npz"))) == expected and len(list((path / "depth").glob("*.npy"))) == expected


def token_cache_complete(path: Path, expected: int) -> bool:
    if not path.is_file():
        return False
    with np.load(path) as data:
        return "encoder_scene" in data.files and data["encoder_scene"].shape[0] == expected


@contextmanager
def capture_spatial_tokens(model: ARCroco3DStereo):
    """Experiment-local wrappers; original methods are restored on exit."""
    frames: list[dict] = []
    original_encode = model._encode_image
    original_rollout = model._recurrent_rollout

    def encode_wrapper(_self, *args, **kwargs):
        result = original_encode(*args, **kwargs)
        image_layers, image_pos = result[0], result[1]
        frames.append(
            {
                "encoder_scene": image_layers[-1].detach(),
                "token_pos": image_pos.detach(),
            }
        )
        return result

    def rollout_wrapper(_self, *args, **kwargs):
        result = original_rollout(*args, **kwargs)
        decoder_layers = result[1]
        frame = frames[-1]
        num_scene = frame["encoder_scene"].shape[1]
        frame["decoder_scene"] = decoder_layers[-1][:, 1 : 1 + num_scene].detach()
        return result

    model._encode_image = MethodType(encode_wrapper, model)
    model._recurrent_rollout = MethodType(rollout_wrapper, model)
    try:
        yield frames
    finally:
        model._encode_image = original_encode
        model._recurrent_rollout = original_rollout


def token_debug_arrays(token_debug: list[dict]) -> dict[str, np.ndarray]:
    keys = (
        "pose_token_in",
        "pose_token_out",
        "human_token_in",
        "human_token_out",
        "state_summary_before",
        "state_summary_after",
        "pose_memory_summary_before",
        "pose_memory_summary_after",
    )
    arrays = {}
    for key in keys:
        values = []
        for row in token_debug:
            value = row.get(key)
            values.append(None if value is None else value.detach().cpu().numpy().astype(np.float32)[0])
        if all(value is None for value in values):
            continue
        dim = next(value.shape[0] for value in values if value is not None)
        values = [np.zeros(dim, dtype=np.float32) if value is None else value for value in values]
        arrays[key] = np.stack(values).astype(np.float16)
    return arrays


def run_segment_with_tokens(
    model: ARCroco3DStereo | None,
    input_dir: Path,
    output_dir: Path,
    token_path: Path,
    expected: int,
    args: argparse.Namespace,
) -> float:
    timing_path = token_path.with_suffix(".timing.json")

    def cached_timing() -> float:
        if not timing_path.is_file():
            return 0.0
        return float(json.loads(timing_path.read_text(encoding="utf-8")).get("elapsed_seconds", 0.0))

    if args.skip_inference:
        if output_complete(output_dir, expected) and token_cache_complete(token_path, expected):
            return cached_timing()
        raise FileNotFoundError(f"Missing cached Human3R output/token data for {output_dir}")
    if output_complete(output_dir, expected) and token_cache_complete(token_path, expected) and not args.overwrite:
        return cached_timing()
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(str(path) for path in input_dir.glob("*.png"))
    if len(image_paths) != expected:
        raise RuntimeError(f"{input_dir} has {len(image_paths)} images, expected {expected}")
    views = prepare_input(
        img_paths=image_paths,
        img_mask=[True] * expected,
        size=int(args.size),
        revisit=1,
        update=True,
        img_res=getattr(model, "mhmr_img_res", None),
        reset_interval=10000000,
    )
    start = time.perf_counter()
    with capture_spatial_tokens(model) as spatial_frames, torch.no_grad():
        preds, batch, _, token_debug = model.forward_recurrent_lighter(
            views,
            str(args.device),
            ret_state=True,
            use_ttt3r=False,
            return_token_debug=True,
        )
    elapsed = time.perf_counter() - start
    if len(spatial_frames) != expected:
        raise RuntimeError(f"Captured {len(spatial_frames)} token frames, expected {expected}")
    outputs_cpu = to_cpu({"pred": preds, "views": batch})
    prepare_output(
        outputs_cpu,
        str(output_dir),
        revisit=1,
        use_pose=True,
        save=True,
        render=False,
        render_video=False,
        img_res=getattr(model, "mhmr_img_res", None),
        subsample=1,
    )
    arrays = {
        "encoder_scene": np.stack([frame["encoder_scene"].float().cpu().numpy()[0] for frame in spatial_frames]).astype(np.float16),
        "decoder_scene": np.stack([frame["decoder_scene"].float().cpu().numpy()[0] for frame in spatial_frames]).astype(np.float16),
        "token_pos": np.stack([frame["token_pos"].float().cpu().numpy()[0] for frame in spatial_frames]).astype(np.float16),
    }
    arrays.update(token_debug_arrays(token_debug))
    token_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(token_path, **arrays)
    timing_path.write_text(
        json.dumps({"elapsed_seconds": elapsed, "frames": expected, "seconds_per_frame": elapsed / expected}, indent=2) + "\n",
        encoding="utf-8",
    )
    return elapsed


def load_tokens(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key].astype(np.float32) for key in data.files}


def build_model(args: argparse.Namespace) -> ARCroco3DStereo:
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    add_path_to_dust3r(str(args.model_path))
    model = ARCroco3DStereo.from_pretrained(str(args.model_path)).to(args.device).eval()
    strict_original_model(model)
    return model


def compose_se3(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return (left @ right).astype(np.float32)


def se3_magnitude(transform: np.ndarray) -> dict:
    identity = np.eye(4, dtype=np.float32)
    return {
        "translation": float(np.linalg.norm(transform[:3, 3])),
        "rotation_deg": rotation_error_deg(transform, identity),
    }


def transform_error(transform: np.ndarray, target: np.ndarray) -> dict:
    delta = transform @ np.linalg.inv(target)
    return se3_magnitude(delta)


def camera_pointmap(output_dir: Path, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pose, K = load_pose(output_dir, idx)
    depth = np.load(output_dir / "depth" / f"{idx:06d}.npy").astype(np.float32)
    conf = np.load(output_dir / "conf" / f"{idx:06d}.npy").astype(np.float32)
    h, w = depth.shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    z = depth
    x = (xx - K[0, 2]) / K[0, 0] * z
    y = (yy - K[1, 2]) / K[1, 1] * z
    points_cam = np.stack([x, y, z], axis=-1)
    points_world = transform_points(pose, points_cam.reshape(-1, 3)).reshape(h, w, 3)
    return points_world, depth, conf, K


def background_cloud(
    output_dir: Path,
    idx: int,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    points, depth, conf, _ = camera_pointmap(output_dir, idx)
    human = load_human_mask(output_dir, idx, depth.shape, threshold=0.10, dilate=15)
    finite = np.isfinite(points).all(axis=-1) & np.isfinite(depth) & np.isfinite(conf)
    valid = finite & (depth > 0.05) & (depth < 50.0) & ~human
    threshold = float(np.quantile(conf[valid], 0.65)) if valid.any() else float("inf")
    valid &= conf >= threshold
    yy, xx = np.where(valid)
    rng = np.random.default_rng(seed)
    if len(xx) > max_points:
        keep = rng.choice(len(xx), size=max_points, replace=False)
        yy, xx = yy[keep], xx[keep]
    return points[yy, xx].astype(np.float32), {
        "valid_background_points": int(valid.sum()),
        "sampled_points": int(len(xx)),
        "confidence_threshold": threshold,
        "human_pixels": int(human.sum()),
    }


def history_background_cloud(
    output_dir: Path,
    frame_indices: list[int],
    max_points_per_frame: int,
) -> tuple[np.ndarray, list[dict]]:
    clouds, diagnostics = [], []
    for idx in frame_indices:
        cloud, debug = background_cloud(output_dir, idx, max_points_per_frame, seed=20260715 + idx)
        clouds.append(cloud)
        diagnostics.append({"idx": idx, **debug})
    if not clouds or not any(len(cloud) for cloud in clouds):
        return np.empty((0, 3), dtype=np.float32), diagnostics
    return np.concatenate(clouds, axis=0), diagnostics


def root_pose_world(output_dir: Path, idx: int) -> tuple[np.ndarray, np.ndarray]:
    pose, _ = load_pose(output_dir, idx)
    smpl = np.load(output_dir / "smpl" / f"{idx:06d}.npz", allow_pickle=True)
    if len(smpl["transl"]) == 0:
        raise ValueError(f"No human in {output_dir} frame {idx}")
    root_cam, _ = cv2.Rodrigues(np.asarray(smpl["rotvec"][0, 0], dtype=np.float64))
    root_world = pose[:3, :3] @ root_cam.astype(np.float32)
    translation_world = pose[:3, :3] @ np.asarray(smpl["transl"][0], dtype=np.float32) + pose[:3, 3]
    return root_world.astype(np.float32), translation_world.astype(np.float32)


def average_rotations(rotations: list[np.ndarray]) -> np.ndarray:
    matrix = np.stack(rotations).sum(axis=0)
    U, _, Vt = np.linalg.svd(matrix)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return R.astype(np.float32)


def explicit_human_initial(history_dir: Path, history_indices: list[int], fresh_dir: Path) -> AlignmentCandidate:
    history_R, history_t = [], []
    for idx in history_indices:
        try:
            R, t = root_pose_world(history_dir, idx)
        except ValueError:
            continue
        history_R.append(R)
        history_t.append(t)
    try:
        current_R, current_t = root_pose_world(fresh_dir, 0)
    except ValueError:
        current_R, current_t = np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
    if not history_R:
        return AlignmentCandidate(
            name="explicit_identity_no_human",
            transform=np.eye(4, dtype=np.float32),
            confidence=0.0,
            diagnostics={"human_available": False},
        )
    target_R = average_rotations(history_R)
    target_t = np.median(np.stack(history_t), axis=0).astype(np.float32)
    R = target_R @ current_R.T
    t = target_t - R @ current_t
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = t
    return AlignmentCandidate(
        name="explicit_human_root_body_frame",
        transform=T,
        confidence=1.0,
        diagnostics={
            "human_available": True,
            "history_human_frames": len(history_R),
            "history_root_center": target_t.tolist(),
            "current_root_center": current_t.tolist(),
            "transform_magnitude": se3_magnitude(T),
        },
    )


def token_pixel_points(
    output_dir: Path,
    idx: int,
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points, depth, conf, _ = camera_pointmap(output_dir, idx)
    human = load_human_mask(output_dir, idx, depth.shape, threshold=0.10, dilate=7)
    patch_size = 16.0
    # CroCo PositionGetter stores patch coordinates as [row(y), col(x)].
    yy = np.clip(np.round((positions[:, 0] + 0.5) * patch_size).astype(np.int64), 0, depth.shape[0] - 1)
    xx = np.clip(np.round((positions[:, 1] + 0.5) * patch_size).astype(np.int64), 0, depth.shape[1] - 1)
    token_points = points[yy, xx]
    valid_conf = conf[np.isfinite(conf)]
    conf_threshold = float(np.quantile(valid_conf, 0.55)) if valid_conf.size else float("inf")
    valid = (
        np.isfinite(token_points).all(axis=-1)
        & np.isfinite(depth[yy, xx])
        & (depth[yy, xx] > 0.05)
        & (depth[yy, xx] < 50.0)
        & np.isfinite(conf[yy, xx])
        & (conf[yy, xx] >= conf_threshold)
        & ~human[yy, xx]
    )
    return token_points.astype(np.float32), valid, conf[yy, xx].astype(np.float32)


def mutual_token_matches(
    current_desc: np.ndarray,
    history_desc: np.ndarray,
    current_valid: np.ndarray,
    history_valid: np.ndarray,
    device: str,
) -> dict:
    current_ids = np.where(current_valid)[0]
    history_ids = np.where(history_valid)[0]
    if len(current_ids) < 2 or len(history_ids) < 2:
        return {"current_idx": np.empty(0, dtype=np.int64), "history_idx": np.empty(0, dtype=np.int64), "similarity": np.empty(0), "margin": np.empty(0)}
    cur = torch.from_numpy(current_desc[current_ids]).to(device=device, dtype=torch.float32)
    hist = torch.from_numpy(history_desc[history_ids]).to(device=device, dtype=torch.float32)
    cur = F.normalize(cur, dim=-1)
    hist = F.normalize(hist, dim=-1)
    with torch.no_grad():
        sim = cur @ hist.T
        top2 = torch.topk(sim, k=2, dim=1)
        cur_to_hist = top2.indices[:, 0]
        hist_to_cur = sim.argmax(dim=0)
        cur_arange = torch.arange(len(cur), device=sim.device)
        mutual = hist_to_cur[cur_to_hist] == cur_arange
        similarity = top2.values[:, 0]
        margin = top2.values[:, 0] - top2.values[:, 1]
        if mutual.any():
            mutual_similarity = similarity[mutual]
            similarity_floor = torch.quantile(mutual_similarity, 0.45)
            margin_floor = torch.quantile(margin[mutual], 0.25)
            keep = mutual & (similarity >= similarity_floor) & (margin >= margin_floor)
        else:
            keep = mutual
        kept_cur = cur_arange[keep].cpu().numpy()
        kept_hist = cur_to_hist[keep].cpu().numpy()
        return {
            "current_idx": current_ids[kept_cur],
            "history_idx": history_ids[kept_hist],
            "similarity": similarity[keep].cpu().numpy().astype(np.float32),
            "margin": margin[keep].cpu().numpy().astype(np.float32),
            "valid_current_tokens": int(len(current_ids)),
            "valid_history_tokens": int(len(history_ids)),
            "mutual_before_filter": int(mutual.sum().item()),
        }


def nondegenerate_triplet(points: np.ndarray) -> bool:
    if len(points) < 3:
        return False
    return float(np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0]))) > 1e-4


def token_ransac_se3(
    src: np.ndarray,
    dst: np.ndarray,
    similarity: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> tuple[np.ndarray | None, np.ndarray, dict]:
    count = len(src)
    if count < int(args.token_min_matches):
        return None, np.zeros(count, dtype=bool), {"reason": "too_few_matches", "matches": count}
    weights = np.maximum(similarity - similarity.min() + 1e-3, 1e-3).astype(np.float32)
    probabilities = weights / weights.sum()
    rng = np.random.default_rng(seed)
    best_T, best_inliers, best_score = None, np.zeros(count, dtype=bool), -float("inf")
    threshold = float(args.token_inlier_threshold)
    for _ in range(int(args.token_ransac_iters)):
        sample = rng.choice(count, size=3, replace=False, p=probabilities)
        if not nondegenerate_triplet(src[sample]) or not nondegenerate_triplet(dst[sample]):
            continue
        T = weighted_kabsch(src[sample], dst[sample], np.ones(3, dtype=np.float32))
        residual = np.linalg.norm(transform_points(T, src) - dst, axis=-1)
        inliers = residual < threshold
        if int(inliers.sum()) < 3:
            continue
        score = float(inliers.sum()) + float(np.clip(similarity[inliers], 0.0, 1.0).sum())
        if score > best_score:
            best_T, best_inliers, best_score = T, inliers, score
    if best_T is None or int(best_inliers.sum()) < 3:
        return None, best_inliers, {"reason": "ransac_failed", "matches": count}
    best_T = weighted_kabsch(src[best_inliers], dst[best_inliers], weights[best_inliers])
    residual = np.linalg.norm(transform_points(best_T, src) - dst, axis=-1)
    best_inliers = residual < threshold
    if int(best_inliers.sum()) >= 3:
        best_T = weighted_kabsch(src[best_inliers], dst[best_inliers], weights[best_inliers])
        residual = np.linalg.norm(transform_points(best_T, src) - dst, axis=-1)
    diagnostics = {
        "matches": count,
        "inliers": int(best_inliers.sum()),
        "inlier_rate": float(best_inliers.mean()),
        "similarity_mean": float(similarity.mean()),
        "similarity_inlier_mean": float(similarity[best_inliers].mean()) if best_inliers.any() else None,
        "residual_median": float(np.median(residual)),
        "residual_inlier_mean": float(residual[best_inliers].mean()) if best_inliers.any() else None,
    }
    return best_T, best_inliers, diagnostics


def build_token_candidates(
    continue_dir: Path,
    fresh_dir: Path,
    continue_tokens: dict[str, np.ndarray],
    fresh_tokens: dict[str, np.ndarray],
    history_indices: list[int],
    args: argparse.Namespace,
    case_seed: int,
) -> tuple[list[AlignmentCandidate], dict]:
    candidates = []
    all_debug = []
    current_positions = fresh_tokens["token_pos"][0]
    current_points, current_valid, _ = token_pixel_points(fresh_dir, 0, current_positions)
    human_similarity = None
    if "human_token_out" in continue_tokens and "human_token_out" in fresh_tokens:
        hist_human = continue_tokens["human_token_out"][history_indices].mean(axis=0)
        cur_human = fresh_tokens["human_token_out"][0]
        denom = max(float(np.linalg.norm(hist_human) * np.linalg.norm(cur_human)), 1e-8)
        human_similarity = float(np.dot(hist_human, cur_human) / denom)
    state_similarity = None
    if "state_summary_after" in continue_tokens and "state_summary_after" in fresh_tokens:
        hist_state = continue_tokens["state_summary_after"][history_indices].mean(axis=0)
        cur_state = fresh_tokens["state_summary_after"][0]
        denom = max(float(np.linalg.norm(hist_state) * np.linalg.norm(cur_state)), 1e-8)
        state_similarity = float(np.dot(hist_state, cur_state) / denom)

    for feature_key in ("encoder_scene", "decoder_scene"):
        current_desc = fresh_tokens[feature_key][0]
        for history_idx in history_indices:
            history_positions = continue_tokens["token_pos"][history_idx]
            history_points, history_valid, _ = token_pixel_points(continue_dir, history_idx, history_positions)
            matches = mutual_token_matches(
                current_desc,
                continue_tokens[feature_key][history_idx],
                current_valid,
                history_valid,
                args.device,
            )
            current_ids = matches["current_idx"]
            history_ids = matches["history_idx"]
            T, inliers, ransac_debug = token_ransac_se3(
                current_points[current_ids],
                history_points[history_ids],
                matches["similarity"],
                args,
                seed=case_seed * 100 + history_idx + (0 if feature_key == "encoder_scene" else 50),
            )
            debug = {
                "feature": feature_key,
                "history_idx": history_idx,
                "human_token_similarity": human_similarity,
                "state_similarity": state_similarity,
                **{key: value for key, value in matches.items() if key not in {"current_idx", "history_idx", "similarity", "margin"}},
                **ransac_debug,
            }
            if len(matches["similarity"]):
                debug["match_similarity_mean"] = float(matches["similarity"].mean())
                debug["match_margin_mean"] = float(matches["margin"].mean())
            all_debug.append(debug)
            if T is None:
                continue
            src_match = current_points[current_ids]
            dst_match = history_points[history_ids]
            cur_pos = current_positions[current_ids]
            hist_pos = history_positions[history_ids]
            grid_scale = np.maximum(
                np.asarray(
                    [
                        max(float(current_positions[:, 0].max()), float(history_positions[:, 0].max()), 1.0),
                        max(float(current_positions[:, 1].max()), float(history_positions[:, 1].max()), 1.0),
                    ],
                    dtype=np.float32,
                ),
                1.0,
            )
            displacement = np.linalg.norm((cur_pos - hist_pos) / grid_scale[None], axis=-1)
            displacement_patches = np.linalg.norm(cur_pos - hist_pos, axis=-1)
            same_patch_rate = float((displacement_patches <= 1.5).mean())
            positional_collapse = bool(same_patch_rate >= 0.50 and float(np.median(displacement_patches)) <= 1.5)
            coverage = min(1.0, float(ransac_debug["inliers"]) / 32.0)
            confidence = (
                float(ransac_debug["inlier_rate"])
                * max(float(ransac_debug["similarity_inlier_mean"] or 0.0), 0.0)
                * coverage
            )
            if human_similarity is not None:
                confidence *= float(np.clip(0.5 + 0.5 * human_similarity, 0.25, 1.0))
            candidates.append(
                AlignmentCandidate(
                    name=f"implicit_{feature_key}_hist{history_idx}",
                    transform=T,
                    confidence=float(confidence),
                    diagnostics={
                        **debug,
                        "transform_magnitude": se3_magnitude(T),
                        "normalized_patch_displacement_mean": float(displacement.mean()),
                        "normalized_patch_displacement_median": float(np.median(displacement)),
                        "patch_displacement_median": float(np.median(displacement_patches)),
                        "same_patch_rate_1p5": same_patch_rate,
                        "positional_collapse": positional_collapse,
                        "reliable_for_hybrid": not positional_collapse,
                        "_match_source_points": src_match,
                        "_match_target_points": dst_match,
                        "_ransac_inliers": inliers,
                        "_match_current_pos": cur_pos,
                        "_match_history_pos": hist_pos,
                    },
                )
            )
    candidates.sort(key=lambda candidate: candidate.confidence, reverse=True)
    return candidates, {
        "human_token_similarity": human_similarity,
        "state_similarity": state_similarity,
        "hypotheses": all_debug,
        "num_valid_candidates": len(candidates),
    }


def annotate_token_candidates_with_oracle(candidates: list[AlignmentCandidate], oracle_transform: np.ndarray) -> list[dict]:
    analysis = []
    for candidate in candidates:
        src = candidate.diagnostics.pop("_match_source_points", None)
        dst = candidate.diagnostics.pop("_match_target_points", None)
        ransac_inliers = candidate.diagnostics.pop("_ransac_inliers", None)
        current_pos = candidate.diagnostics.pop("_match_current_pos", None)
        history_pos = candidate.diagnostics.pop("_match_history_pos", None)
        if src is None or dst is None or len(src) == 0:
            continue
        residual = np.linalg.norm(transform_points(oracle_transform, src) - dst, axis=-1)
        correct_020 = residual < 0.20
        candidate.diagnostics["oracle_correspondence_analysis"] = {
            "median_3d_error": float(np.median(residual)),
            "mean_3d_error": float(residual.mean()),
            "correct_rate_020": float(correct_020.mean()),
            "ransac_inlier_correct_rate_020": None
            if ransac_inliers is None or not np.asarray(ransac_inliers).any()
            else float(correct_020[np.asarray(ransac_inliers, dtype=bool)].mean()),
            "note": "GT/oracle is used only here after candidate estimation to diagnose whether token matches are physically correct.",
        }
        if current_pos is not None and history_pos is not None:
            candidate.diagnostics["match_patch_pairs"] = [
                {
                    "current_yx": np.asarray(current_pos[idx], dtype=np.float32).tolist(),
                    "history_yx": np.asarray(history_pos[idx], dtype=np.float32).tolist(),
                    "ransac_inlier": False if ransac_inliers is None else bool(np.asarray(ransac_inliers)[idx]),
                    "oracle_3d_error": float(residual[idx]),
                    "oracle_correct_020": bool(correct_020[idx]),
                }
                for idx in range(len(residual))
            ]
        analysis.append(
            {
                "name": candidate.name,
                "confidence": candidate.confidence,
                **candidate.diagnostics["oracle_correspondence_analysis"],
            }
        )
    return analysis


def render_token_match_diagnostic(
    case_dir: Path,
    candidate: AlignmentCandidate,
    continue_dir: Path,
    fresh_dir: Path,
) -> str | None:
    pairs = candidate.diagnostics.get("match_patch_pairs", [])
    history_idx = candidate.diagnostics.get("history_idx")
    if not pairs or history_idx is None:
        return None
    history_bgr = cv2.imread(str(continue_dir / "color" / f"{int(history_idx):06d}.png"), cv2.IMREAD_COLOR)
    current_bgr = cv2.imread(str(fresh_dir / "color" / "000000.png"), cv2.IMREAD_COLOR)
    if history_bgr is None or current_bgr is None:
        return None
    height = max(history_bgr.shape[0], current_bgr.shape[0])
    width_a = history_bgr.shape[1]
    canvas = np.zeros((height + 54, width_a + current_bgr.shape[1], 3), dtype=np.uint8)
    canvas[:54] = 24
    canvas[54 : 54 + history_bgr.shape[0], :width_a] = history_bgr
    canvas[54 : 54 + current_bgr.shape[0], width_a:] = current_bgr
    ordered = sorted(pairs, key=lambda pair: (not pair["ransac_inlier"], pair["oracle_3d_error"]))[:60]
    for pair in ordered:
        hy, hx = pair["history_yx"]
        cy, cx = pair["current_yx"]
        p0 = (int(round((hx + 0.5) * 16.0)), int(round((hy + 0.5) * 16.0)) + 54)
        p1 = (width_a + int(round((cx + 0.5) * 16.0)), int(round((cy + 0.5) * 16.0)) + 54)
        if pair["oracle_correct_020"]:
            color = (60, 220, 60)
        elif pair["ransac_inlier"]:
            color = (0, 165, 255)
        else:
            color = (90, 90, 220)
        cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, p0, 2, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 2, color, -1, cv2.LINE_AA)
    title = (
        f"{candidate.name} conf={candidate.confidence:.3f} | "
        "green=oracle-correct, orange=RANSAC-inlier but wrong"
    )
    cv2.putText(canvas, title, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 2, cv2.LINE_AA)
    output = case_dir / "analysis" / "selected_token_matches_oracle_diagnostic.jpg"
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas)
    return str(output)


def fixed_geometry_score(transform: np.ndarray, source: np.ndarray, target: np.ndarray) -> dict:
    if len(source) == 0 or len(target) == 0:
        return {"score": -float("inf"), "overlap_020": 0.0, "median_distance": None, "trimmed_mean": None}
    tree = cKDTree(target)
    distances, _ = tree.query(transform_points(transform, source), k=1, workers=-1)
    finite = distances[np.isfinite(distances)]
    if finite.size == 0:
        return {"score": -float("inf"), "overlap_020": 0.0, "median_distance": None, "trimmed_mean": None}
    trimmed = finite[finite <= np.quantile(finite, 0.70)]
    median = float(np.median(finite))
    trimmed_mean = float(trimmed.mean()) if trimmed.size else median
    overlap = float((finite < 0.20).mean())
    score = overlap - 0.25 * min(trimmed_mean, 2.0)
    return {"score": float(score), "overlap_020": overlap, "median_distance": median, "trimmed_mean": trimmed_mean}


def robust_local_pointmap_refinement(
    initial: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict]:
    """Fixed-budget local point-to-point refinement shared by Explicit and Hybrid."""
    transform = initial.copy().astype(np.float32)
    tree = cKDTree(target)
    history = []
    start = time.perf_counter()
    for iteration in range(int(args.refine_iters)):
        transformed = transform_points(transform, source)
        distances, indices = tree.query(transformed, k=1, workers=-1)
        alpha = iteration / max(int(args.refine_iters) - 1, 1)
        max_distance = (1.0 - alpha) * float(args.refine_max_distance) + alpha * float(args.refine_min_distance)
        valid = np.isfinite(distances) & (distances < max_distance)
        if int(valid.sum()) < 32:
            history.append({"iteration": iteration, "pairs": int(valid.sum()), "status": "too_few_pairs"})
            break
        valid_ids = np.where(valid)[0]
        keep_limit = float(np.quantile(distances[valid], 0.70))
        valid_ids = valid_ids[distances[valid_ids] <= keep_limit]
        if len(valid_ids) < 32:
            break
        src_current = transformed[valid_ids]
        dst_match = target[indices[valid_ids]]
        weights = 1.0 / np.maximum(distances[valid_ids], 0.01)
        delta = weighted_kabsch(src_current, dst_match, weights.astype(np.float32))
        transform = compose_se3(delta, transform)
        history.append(
            {
                "iteration": iteration,
                "pairs": int(len(valid_ids)),
                "max_distance": max_distance,
                "trim_limit": keep_limit,
                "median_distance": float(np.median(distances[valid_ids])),
                "delta": se3_magnitude(delta),
            }
        )
    elapsed = time.perf_counter() - start
    return transform, {
        "iterations_budget": int(args.refine_iters),
        "iterations_run": len(history),
        "elapsed_seconds": elapsed,
        "history": history,
        "initial_score": fixed_geometry_score(initial, source, target),
        "final_score": fixed_geometry_score(transform, source, target),
        "residual_from_initial": se3_magnitude(transform @ np.linalg.inv(initial)),
    }


def choose_hybrid_initial(
    explicit: AlignmentCandidate,
    token_candidates: list[AlignmentCandidate],
    source_cloud: np.ndarray,
    target_cloud: np.ndarray,
) -> tuple[AlignmentCandidate, dict]:
    top_token_reliable = bool(
        token_candidates
        and token_candidates[0].diagnostics.get("reliable_for_hybrid", True)
    )
    reliable_tokens = (
        [
            candidate
            for candidate in token_candidates
            if bool(candidate.diagnostics.get("reliable_for_hybrid", True))
        ]
        if top_token_reliable
        else []
    )
    candidates = [explicit] + reliable_tokens[:4]
    scored = []
    for candidate in candidates:
        geometry = fixed_geometry_score(candidate.transform, source_cloud, target_cloud)
        token_bonus = 0.02 * candidate.confidence if candidate.name.startswith("implicit_") else 0.0
        total = float(geometry["score"] + token_bonus)
        scored.append(
            {
                "name": candidate.name,
                "geometry": geometry,
                "token_confidence": candidate.confidence,
                "total_score": total,
                "candidate": candidate,
            }
        )
    scored.sort(key=lambda item: item["total_score"], reverse=True)
    chosen = scored[0]["candidate"]
    return chosen, {
        "chosen": chosen.name,
        "scores": [{key: value for key, value in item.items() if key != "candidate"} for item in scored],
        "token_candidate_selected": chosen.name.startswith("implicit_"),
        "token_candidates_total": len(token_candidates),
        "token_candidates_reliable": len(reliable_tokens),
        "fallback_reason": None
        if reliable_tokens
        else (
            "top-confidence token candidate failed position-collapse reliability check"
            if token_candidates
            else "no valid token candidate"
        ),
    }


def target_poses_for_case(args: argparse.Namespace, frame_records: list[dict], raw_continue: Path) -> list[np.ndarray]:
    gt = []
    for frame in frame_records:
        group = str(frame["seq"]).split("/", 1)[0]
        gt.append(load_gt_c2w(args.raw_meta_root / group, frame["seq"]))
    pred0, _ = load_pose(raw_continue, 0)
    align = pred0 @ np.linalg.inv(gt[0])
    return [(align @ pose).astype(np.float32) for pose in gt]


def apply_oracle(reset_merged: Path, dst: Path, target_poses: list[np.ndarray], cut_idx: int, frame_count: int) -> np.ndarray:
    reset_boundary, _ = load_pose(reset_merged, cut_idx)
    transform = target_poses[cut_idx] @ np.linalg.inv(reset_boundary)
    apply_bridge_to_shot(reset_merged, dst, transform.astype(np.float32), cut_idx, frame_count)
    return transform.astype(np.float32)


def human_jump(output_dir: Path, cut_idx: int) -> dict:
    try:
        pre_R, pre_t = root_pose_world(output_dir, cut_idx - 1)
        post_R, post_t = root_pose_world(output_dir, cut_idx)
    except ValueError:
        return {"available": False}
    pre = np.eye(4, dtype=np.float32)
    post = np.eye(4, dtype=np.float32)
    pre[:3, :3], pre[:3, 3] = pre_R, pre_t
    post[:3, :3], post[:3, 3] = post_R, post_t
    return {
        "available": True,
        "world_root_jump": float(np.linalg.norm(post_t - pre_t)),
        "global_orientation_jump_deg": rotation_error_deg(pre, post),
    }


def pointcloud_stitch_metrics(output_dir: Path, cut_idx: int, points_per_frame: int) -> dict:
    target, _ = background_cloud(output_dir, cut_idx - 1, points_per_frame, 3000 + cut_idx)
    source, _ = background_cloud(output_dir, cut_idx, points_per_frame, 4000 + cut_idx)
    score = fixed_geometry_score(np.eye(4, dtype=np.float32), source, target)
    return {
        "overlap_020": score["overlap_020"],
        "median_nearest_distance": score["median_distance"],
        "trimmed_mean_nearest_distance": score["trimmed_mean"],
    }


def image_texture_score(path: Path) -> float:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return 0.0
    resized = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(resized, cv2.CV_32F).var())


def safe_case_name(record: dict, index: int) -> str:
    raw = f"{index:02d}_{record['group']}_{record['start_frame']}_{record['seqA'].split('/')[-1]}_{record['seqB'].split('/')[-1]}"
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in raw)


def evaluate_case_variants(
    dirs: dict[str, Path],
    target_poses: list[np.ndarray],
    cut_idx: int,
    frame_count: int,
    points_per_frame: int,
) -> tuple[list[dict], dict, dict, dict, dict]:
    variants = [
        summarize_variant(name, path, target_poses, cut_idx, frame_count, min(1200, points_per_frame))
        for name, path in dirs.items()
    ]
    rpe = {name: summarize_rpe(path, target_poses, cut_idx, frame_count) for name, path in dirs.items()}
    recovery = {item["name"]: threshold_stats(item["per_frame"]) for item in variants}
    human = {name: human_jump(path, cut_idx) for name, path in dirs.items()}
    pointcloud = {name: pointcloud_stitch_metrics(path, cut_idx, points_per_frame) for name, path in dirs.items()}
    return variants, rpe, recovery, human, pointcloud


def run_case(
    model: ARCroco3DStereo | None,
    record: dict,
    index: int,
    args: argparse.Namespace,
) -> dict:
    case_name = safe_case_name(record, index)
    case_dir = args.output_dir / "cases" / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    frame_records = build_case_inputs(args, record, case_dir)
    cut_idx = int(args.pre_frames)
    frame_count = cut_idx + int(args.post_frames)
    history_start = max(0, cut_idx - int(args.history_window))
    history_indices = list(range(history_start, cut_idx))

    raw_continue = case_dir / "A_original_continue"
    raw_fresh = case_dir / "raw_post_fresh"
    continue_tokens_path = case_dir / "tokens_continue.npz"
    fresh_tokens_path = case_dir / "tokens_fresh_post.npz"
    time_continue = run_segment_with_tokens(
        model,
        case_dir / "input_all",
        raw_continue,
        continue_tokens_path,
        frame_count,
        args,
    )
    time_fresh = run_segment_with_tokens(
        model,
        case_dir / "input_post",
        raw_fresh,
        fresh_tokens_path,
        int(args.post_frames),
        args,
    )

    reset_merged = case_dir / "R_reset_without_alignment"
    merge_reset_output(
        SimpleNamespace(pre_frames=cut_idx, post_frames=int(args.post_frames)),
        raw_continue,
        raw_fresh,
        reset_merged,
    )
    continue_tokens = load_tokens(continue_tokens_path)
    fresh_tokens = load_tokens(fresh_tokens_path)
    explicit_initial = explicit_human_initial(raw_continue, history_indices, raw_fresh)
    token_candidates, token_debug = build_token_candidates(
        raw_continue,
        raw_fresh,
        continue_tokens,
        fresh_tokens,
        history_indices,
        args,
        case_seed=index + 1,
    )
    if token_candidates:
        implicit_candidate = token_candidates[0]
    else:
        implicit_candidate = AlignmentCandidate(
            name="implicit_failed_identity",
            transform=np.eye(4, dtype=np.float32),
            confidence=0.0,
            diagnostics={"reason": "no_valid_token_candidate"},
        )

    target_cloud, target_cloud_debug = history_background_cloud(
        raw_continue,
        history_indices,
        int(args.cloud_points_per_frame),
    )
    source_cloud, source_cloud_debug = background_cloud(
        raw_fresh,
        0,
        int(args.cloud_points_per_frame),
        seed=20260715 + index,
    )
    explicit_transform, explicit_refine = robust_local_pointmap_refinement(
        explicit_initial.transform,
        source_cloud,
        target_cloud,
        args,
    )
    hybrid_initial, hybrid_select = choose_hybrid_initial(
        explicit_initial,
        token_candidates,
        source_cloud,
        target_cloud,
    )
    hybrid_transform, hybrid_refine = robust_local_pointmap_refinement(
        hybrid_initial.transform,
        source_cloud,
        target_cloud,
        args,
    )

    explicit_dir = case_dir / "E_reset_explicit_only"
    implicit_dir = case_dir / "I_reset_implicit_only"
    hybrid_dir = case_dir / "H_reset_implicit_to_explicit"
    oracle_dir = case_dir / "C_reset_oracle"
    apply_bridge_to_shot(reset_merged, explicit_dir, explicit_transform, cut_idx, frame_count)
    apply_bridge_to_shot(reset_merged, implicit_dir, implicit_candidate.transform, cut_idx, frame_count)
    apply_bridge_to_shot(reset_merged, hybrid_dir, hybrid_transform, cut_idx, frame_count)
    target_poses = target_poses_for_case(args, frame_records, raw_continue)
    oracle_transform = apply_oracle(reset_merged, oracle_dir, target_poses, cut_idx, frame_count)
    token_debug["candidate_oracle_correspondence_analysis"] = annotate_token_candidates_with_oracle(
        token_candidates,
        oracle_transform,
    )
    token_match_plot = render_token_match_diagnostic(
        case_dir,
        implicit_candidate,
        raw_continue,
        raw_fresh,
    )

    for output_path in (explicit_dir, implicit_dir, hybrid_dir, oracle_dir):
        add_world_smpl_payload(output_path, frame_count, args.smpl_device)

    dirs = {
        "A_original_continue": raw_continue,
        "R_reset_without_alignment": reset_merged,
        "E_reset_explicit_only": explicit_dir,
        "I_reset_implicit_only": implicit_dir,
        "H_reset_implicit_to_explicit": hybrid_dir,
        "C_reset_oracle": oracle_dir,
    }
    variants, rpe, recovery, human, pointcloud = evaluate_case_variants(
        dirs,
        target_poses,
        cut_idx,
        frame_count,
        int(args.cloud_points_per_frame),
    )
    variant_lookup = {item["name"]: item for item in variants}
    oracle_overlap = pointcloud["C_reset_oracle"]["overlap_020"]
    texture = 0.5 * (
        image_texture_score(Path(frame_records[cut_idx - 1]["path"]))
        + image_texture_score(Path(frame_records[cut_idx]["path"]))
    )
    target_transform = oracle_transform

    alignment_debug = {
        "explicit_initial": {
            "name": explicit_initial.name,
            "confidence": explicit_initial.confidence,
            "transform": explicit_initial.transform.tolist(),
            "transform_error_to_oracle": transform_error(explicit_initial.transform, target_transform),
            "diagnostics": explicit_initial.diagnostics,
        },
        "implicit_selected": {
            "name": implicit_candidate.name,
            "confidence": implicit_candidate.confidence,
            "transform": implicit_candidate.transform.tolist(),
            "transform_error_to_oracle": transform_error(implicit_candidate.transform, target_transform),
            "diagnostics": implicit_candidate.diagnostics,
        },
        "explicit_refinement": {
            "final_transform": explicit_transform.tolist(),
            "final_error_to_oracle": transform_error(explicit_transform, target_transform),
            **explicit_refine,
        },
        "hybrid_selection": hybrid_select,
        "hybrid_refinement": {
            "initial_name": hybrid_initial.name,
            "initial_transform": hybrid_initial.transform.tolist(),
            "initial_error_to_oracle": transform_error(hybrid_initial.transform, target_transform),
            "final_transform": hybrid_transform.tolist(),
            "final_error_to_oracle": transform_error(hybrid_transform, target_transform),
            **hybrid_refine,
        },
        "oracle_transform": target_transform.tolist(),
        "token_debug": token_debug,
        "target_cloud_debug": target_cloud_debug,
        "source_cloud_debug": source_cloud_debug,
    }
    report = {
        "case_name": case_name,
        "record": record,
        "frame_records": frame_records,
        "cut_idx": cut_idx,
        "frame_count": frame_count,
        "history_indices": history_indices,
        "difficulty_raw": {
            "view_angle_deg": float(record.get("view_angle_deg", 0.0)),
            "texture_laplacian_variance": texture,
            "oracle_background_overlap_020": oracle_overlap,
        },
        "timing": {
            "human3r_continue_seconds": time_continue,
            "human3r_fresh_seconds": time_fresh,
            "explicit_refinement_seconds": explicit_refine["elapsed_seconds"],
            "hybrid_refinement_seconds": hybrid_refine["elapsed_seconds"],
        },
        "variant_dirs": {name: str(path) for name, path in dirs.items()},
        "variants": variants,
        "rpe": rpe,
        "strict_recovery": recovery,
        "human_jump": human,
        "pointcloud_stitch": pointcloud,
        "alignment_debug": alignment_debug,
        "plots": {"selected_token_matches": token_match_plot},
        "success": {
            name: bool(
                variant_lookup[name]["mean_camera_t_error"] < 0.10
                and variant_lookup[name]["mean_camera_r_error_deg"] < 5.0
            )
            for name in dirs
        },
    }
    (case_dir / "case_metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def quantile_label(value: float, values: list[float], labels: tuple[str, str, str]) -> str:
    if len(values) < 3:
        return labels[1]
    q1, q2 = np.quantile(np.asarray(values, dtype=np.float32), [1.0 / 3.0, 2.0 / 3.0])
    if value <= q1:
        return labels[0]
    if value >= q2:
        return labels[2]
    return labels[1]


def assign_difficulty(cases: list[dict]) -> None:
    textures = [case["difficulty_raw"]["texture_laplacian_variance"] for case in cases]
    overlaps = [case["difficulty_raw"]["oracle_background_overlap_020"] for case in cases]
    for case in cases:
        raw = case["difficulty_raw"]
        angle = raw["view_angle_deg"]
        if angle < 60.0:
            angle_label = "small"
        elif angle >= 120.0:
            angle_label = "large"
        else:
            angle_label = "medium"
        case["difficulty"] = {
            "texture": quantile_label(raw["texture_laplacian_variance"], textures, ("low", "medium", "high")),
            "angle": angle_label,
            "background_overlap": quantile_label(
                raw["oracle_background_overlap_020"], overlaps, ("low", "medium", "high")
            ),
        }


def variant_metric(case: dict, name: str) -> dict:
    return next(item for item in case["variants"] if item["name"] == name)


def aggregate_variant(cases: list[dict], variant_name: str) -> dict:
    items = [variant_metric(case, variant_name) for case in cases]
    return {
        "count": len(items),
        "mean_camera_t_error": float(np.mean([item["mean_camera_t_error"] for item in items])),
        "mean_camera_r_error_deg": float(np.mean([item["mean_camera_r_error_deg"] for item in items])),
        "success_rate": float(np.mean([case["success"][variant_name] for case in cases])),
        "mean_boundary_t_error": float(np.mean([item["per_frame"][0]["camera_t_error"] for item in items])),
        "mean_boundary_r_error_deg": float(np.mean([item["per_frame"][0]["camera_r_error_deg"] for item in items])),
        "mean_root_jump": float(
            np.mean(
                [
                    case["human_jump"][variant_name]["world_root_jump"]
                    for case in cases
                    if case["human_jump"][variant_name].get("available")
                ]
            )
        ),
        "mean_orientation_jump_deg": float(
            np.mean(
                [
                    case["human_jump"][variant_name]["global_orientation_jump_deg"]
                    for case in cases
                    if case["human_jump"][variant_name].get("available")
                ]
            )
        ),
        "mean_pointcloud_overlap_020": float(
            np.mean([case["pointcloud_stitch"][variant_name]["overlap_020"] for case in cases])
        ),
    }


def group_summary(cases: list[dict], variant_names: list[str]) -> dict:
    grouped = {}
    for dimension in ("texture", "angle", "background_overlap"):
        grouped[dimension] = {}
        for label in sorted({case["difficulty"][dimension] for case in cases}):
            subset = [case for case in cases if case["difficulty"][dimension] == label]
            grouped[dimension][label] = {
                "count": len(subset),
                "variants": {name: aggregate_variant(subset, name) for name in variant_names},
            }
    return grouped


def token_value_analysis(cases: list[dict]) -> dict:
    confidence = np.asarray(
        [case["alignment_debug"]["implicit_selected"]["confidence"] for case in cases], dtype=np.float32
    )
    coarse_error = np.asarray(
        [
            case["alignment_debug"]["implicit_selected"]["transform_error_to_oracle"]["rotation_deg"]
            for case in cases
        ],
        dtype=np.float32,
    )
    hybrid_success = np.asarray([case["success"]["H_reset_implicit_to_explicit"] for case in cases], dtype=np.float32)
    match_correct_rate = np.asarray(
        [
            case["alignment_debug"]["implicit_selected"]["diagnostics"]
            .get("oracle_correspondence_analysis", {})
            .get("correct_rate_020", 0.0)
            for case in cases
        ],
        dtype=np.float32,
    )
    explicit_error = np.asarray(
        [variant_metric(case, "E_reset_explicit_only")["mean_camera_t_error"] for case in cases], dtype=np.float32
    )
    hybrid_error = np.asarray(
        [variant_metric(case, "H_reset_implicit_to_explicit")["mean_camera_t_error"] for case in cases], dtype=np.float32
    )
    confidence_error_corr = spearmanr(confidence, -coarse_error).statistic if len(cases) > 2 else None
    confidence_success_corr = spearmanr(confidence, hybrid_success).statistic if len(np.unique(hybrid_success)) > 1 else None
    confidence_match_corr = spearmanr(confidence, match_correct_rate).statistic if len(cases) > 2 else None
    explicit_fail_hybrid_success = int(
        sum(
            (not case["success"]["E_reset_explicit_only"])
            and case["success"]["H_reset_implicit_to_explicit"]
            for case in cases
        )
    )
    token_selected = int(
        sum(case["alignment_debug"]["hybrid_selection"]["token_candidate_selected"] for case in cases)
    )
    return {
        "token_confidence_vs_negative_coarse_rotation_error_spearman": None
        if confidence_error_corr is None or not np.isfinite(confidence_error_corr)
        else float(confidence_error_corr),
        "token_confidence_vs_hybrid_success_spearman": None
        if confidence_success_corr is None or not np.isfinite(confidence_success_corr)
        else float(confidence_success_corr),
        "token_confidence_vs_oracle_match_correct_rate_spearman": None
        if confidence_match_corr is None or not np.isfinite(confidence_match_corr)
        else float(confidence_match_corr),
        "selected_token_match_correct_rate_020_mean": float(match_correct_rate.mean()),
        "hybrid_token_candidate_selected_cases": token_selected,
        "explicit_fail_but_hybrid_success_cases": explicit_fail_hybrid_success,
        "hybrid_translation_improvement_over_explicit_mean": float(np.mean(explicit_error - hybrid_error)),
        "per_case": [
            {
                "case_name": case["case_name"],
                "token_confidence": case["alignment_debug"]["implicit_selected"]["confidence"],
                "token_match_correct_rate_020": case["alignment_debug"]["implicit_selected"]["diagnostics"]
                .get("oracle_correspondence_analysis", {})
                .get("correct_rate_020"),
                "implicit_coarse_error": case["alignment_debug"]["implicit_selected"]["transform_error_to_oracle"],
                "explicit_initial_error": case["alignment_debug"]["explicit_initial"]["transform_error_to_oracle"],
                "hybrid_initial_name": case["alignment_debug"]["hybrid_refinement"]["initial_name"],
                "hybrid_initial_error": case["alignment_debug"]["hybrid_refinement"]["initial_error_to_oracle"],
                "hybrid_residual": case["alignment_debug"]["hybrid_refinement"]["residual_from_initial"],
                "explicit_success": case["success"]["E_reset_explicit_only"],
                "hybrid_success": case["success"]["H_reset_implicit_to_explicit"],
            }
            for case in cases
        ],
    }


def write_summary_csv(path: Path, cases: list[dict], variant_names: list[str]) -> None:
    rows = []
    for case in cases:
        for name in variant_names:
            item = variant_metric(case, name)
            rows.append(
                {
                    "case_name": case["case_name"],
                    "variant": name,
                    "view_angle_deg": case["difficulty_raw"]["view_angle_deg"],
                    "texture": case["difficulty"]["texture"],
                    "angle": case["difficulty"]["angle"],
                    "background_overlap": case["difficulty"]["background_overlap"],
                    "camera_t_mean": item["mean_camera_t_error"],
                    "camera_r_mean_deg": item["mean_camera_r_error_deg"],
                    "boundary_t": item["per_frame"][0]["camera_t_error"],
                    "boundary_r_deg": item["per_frame"][0]["camera_r_error_deg"],
                    "strict_success_rate": case["strict_recovery"][name]["success_rate"],
                    "success": case["success"][name],
                    "root_jump": case["human_jump"][name].get("world_root_jump"),
                    "orientation_jump_deg": case["human_jump"][name].get("global_orientation_jump_deg"),
                    "pointcloud_overlap_020": case["pointcloud_stitch"][name]["overlap_020"],
                }
            )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(output_dir: Path, cases: list[dict], variant_names: list[str]) -> dict:
    analysis = output_dir / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    saved = {}
    short = {
        "A_original_continue": "Continue",
        "R_reset_without_alignment": "Reset",
        "E_reset_explicit_only": "Explicit",
        "I_reset_implicit_only": "Implicit",
        "H_reset_implicit_to_explicit": "Hybrid",
        "C_reset_oracle": "Oracle",
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    xs = np.arange(len(cases))
    for name in variant_names:
        axes[0].plot(xs, [variant_metric(case, name)["mean_camera_t_error"] for case in cases], marker="o", label=short[name])
        axes[1].plot(xs, [variant_metric(case, name)["mean_camera_r_error_deg"] for case in cases], marker="o", label=short[name])
    axes[0].set_title("Post-cut camera translation error")
    axes[1].set_title("Post-cut camera rotation error")
    for ax in axes:
        ax.set_xticks(xs, [f"{case['difficulty_raw']['view_angle_deg']:.0f}deg" for case in cases], rotation=35)
        ax.grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    path = analysis / "case_camera_errors.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved["case_camera_errors"] = str(path)

    confidence = [case["alignment_debug"]["implicit_selected"]["confidence"] for case in cases]
    implicit_rot = [
        case["alignment_debug"]["implicit_selected"]["transform_error_to_oracle"]["rotation_deg"]
        for case in cases
    ]
    hybrid_gain = [
        variant_metric(case, "E_reset_explicit_only")["mean_camera_t_error"]
        - variant_metric(case, "H_reset_implicit_to_explicit")["mean_camera_t_error"]
        for case in cases
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(confidence, implicit_rot)
    axes[0].set_xlabel("token confidence")
    axes[0].set_ylabel("implicit coarse rotation error (deg)")
    axes[1].scatter(confidence, hybrid_gain)
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_xlabel("token confidence")
    axes[1].set_ylabel("Explicit T error - Hybrid T error")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = analysis / "token_confidence_diagnostics.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved["token_confidence"] = str(path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name in variant_names:
        per_case_rows = [next(item for item in case["variants"] if item["name"] == name)["per_frame"] for case in cases]
        offsets = [row["offset"] for row in per_case_rows[0]]
        mean_t = np.mean([[row["camera_t_error"] for row in rows] for rows in per_case_rows], axis=0)
        mean_r = np.mean([[row["camera_r_error_deg"] for row in rows] for rows in per_case_rows], axis=0)
        axes[0].plot(offsets, mean_t, marker="o", label=short[name])
        axes[1].plot(offsets, mean_r, marker="o", label=short[name])
    axes[0].set_title("Mean per-frame camera translation error")
    axes[1].set_title("Mean per-frame camera rotation error")
    for ax in axes:
        ax.set_xlabel("post-cut offset")
        ax.grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    path = analysis / "mean_per_frame_camera_error_curves.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved["mean_per_frame_camera_errors"] = str(path)
    return saved


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# V10 隐式粗对齐 + 显式细对齐 Training-Free Probe",
        "",
        "Human3R 主体完全冻结，使用 GT cut_idx，但 GT 不参与任何 bridge 计算。",
        "",
        "## 总体结果",
        "",
        "| Variant | Cam T mean ↓ | Cam R mean ↓ | Boundary T ↓ | Boundary R ↓ | Success rate ↑ | Root jump ↓ | Orient jump ↓ | PC overlap ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in report["aggregate"].items():
        lines.append(
            f"| {name} | {item['mean_camera_t_error']:.4f} | {item['mean_camera_r_error_deg']:.2f} | "
            f"{item['mean_boundary_t_error']:.4f} | {item['mean_boundary_r_error_deg']:.2f} | "
            f"{item['success_rate']:.2f} | {item['mean_root_jump']:.4f} | "
            f"{item['mean_orientation_jump_deg']:.2f} | {item['mean_pointcloud_overlap_020']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Token 价值诊断",
            "",
            f"- Hybrid 选择 token 初值：`{report['token_analysis']['hybrid_token_candidate_selected_cases']}/{len(report['cases'])}` cases。",
            f"- Explicit 失败但 Hybrid 成功：`{report['token_analysis']['explicit_fail_but_hybrid_success_cases']}` cases。",
            f"- Hybrid 相比 Explicit 的平均 camera translation 改善：`{report['token_analysis']['hybrid_translation_improvement_over_explicit_mean']:.4f}`。",
            f"- token confidence 与 coarse 方向正确性的 Spearman：`{report['token_analysis']['token_confidence_vs_negative_coarse_rotation_error_spearman']}`。",
            f"- token confidence 与 Hybrid 成功率的 Spearman：`{report['token_analysis']['token_confidence_vs_hybrid_success_spearman']}`。",
            f"- 选中 token matches 的 oracle 3D 正确率（0.2 m）均值：`{report['token_analysis']['selected_token_match_correct_rate_020_mean']:.4f}`。",
            f"- token confidence 与真实 match 正确率的 Spearman：`{report['token_analysis']['token_confidence_vs_oracle_match_correct_rate_spearman']}`。",
            "",
            "## Case 明细",
            "",
            "| Case | Angle | Texture | Overlap | Implicit coarse T/R | Hybrid init | Explicit success | Hybrid success |",
            "|---|---:|---|---|---:|---|---:|---:|",
        ]
    )
    for case in report["cases"]:
        implicit = case["alignment_debug"]["implicit_selected"]
        err = implicit["transform_error_to_oracle"]
        lines.append(
            f"| {case['case_name']} | {case['difficulty_raw']['view_angle_deg']:.1f} | "
            f"{case['difficulty']['texture']} | {case['difficulty']['background_overlap']} | "
            f"{err['translation']:.3f}/{err['rotation_deg']:.1f} | "
            f"{case['alignment_debug']['hybrid_refinement']['initial_name']} | "
            f"{case['success']['E_reset_explicit_only']} | {case['success']['H_reset_implicit_to_explicit']} |"
        )
    lines.extend(["", "## 难度分组：Explicit 与 Hybrid", ""])
    for dimension, groups in report["by_difficulty"].items():
        lines.extend(
            [
                f"### {dimension}",
                "",
                "| Group | Count | Explicit T/R | Hybrid T/R | Hybrid-Explicit T |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for label, group in groups.items():
            explicit = group["variants"]["E_reset_explicit_only"]
            hybrid = group["variants"]["H_reset_implicit_to_explicit"]
            lines.append(
                f"| {label} | {group['count']} | {explicit['mean_camera_t_error']:.3f}/{explicit['mean_camera_r_error_deg']:.1f} | "
                f"{hybrid['mean_camera_t_error']:.3f}/{hybrid['mean_camera_r_error_deg']:.1f} | "
                f"{hybrid['mean_camera_t_error'] - explicit['mean_camera_t_error']:+.3f} |"
            )
    lines.extend(
        [
            "",
            "## 推理时间",
            "",
            "| Component | Measured cases | Mean seconds | Median seconds |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, timing in report["timing"].items():
        mean = "-" if timing["mean_seconds"] is None else f"{timing['mean_seconds']:.4f}"
        median = "-" if timing["median_seconds"] is None else f"{timing['median_seconds']:.4f}"
        lines.append(f"| {name} | {timing['measured_cases']} | {mean} | {median} |")
    lines.extend(
        [
            "",
            "## 公平性与限制",
            "",
            "- Explicit-only 与 Hybrid 使用完全相同的历史窗口、fresh pointmap、背景 mask、点数、8 次 refinement 和阈值。",
            "- Hybrid 唯一额外信息是冻结 token 提供的 coarse candidate 和 confidence。",
            "- Implicit-only 的 SE(3) 由 token mutual matches 决定对应关系，再读取对应像素的预测 3D；token 本身没有被当作三维坐标。",
            "- RICH 当前上传目录没有 camera calibration，因此没有混入定量成功率。",
            "- 成功定义为 post-cut 平均 camera translation < 0.10 且 rotation < 5 deg。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def aggregate_timing(cases: list[dict]) -> dict:
    keys = sorted({key for case in cases for key in case["timing"]})
    result = {}
    for key in keys:
        values = [float(case["timing"][key]) for case in cases if float(case["timing"][key]) > 0.0]
        result[key] = {
            "measured_cases": len(values),
            "mean_seconds": None if not values else float(np.mean(values)),
            "median_seconds": None if not values else float(np.median(values)),
        }
    return result


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = select_cases(args)
    model = None if args.skip_inference else build_model(args)
    case_reports = []
    for index, record in enumerate(cases):
        print(f"[{index + 1}/{len(cases)}] {record['pattern_id']} angle={record.get('view_angle_deg')}", flush=True)
        case_reports.append(run_case(model, record, index, args))
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()
    assign_difficulty(case_reports)
    variant_names = [item["name"] for item in case_reports[0]["variants"]]
    aggregate = {name: aggregate_variant(case_reports, name) for name in variant_names}
    report = {
        "settings": {
            "training": False,
            "human3r_frozen": True,
            "gt_cut_idx": True,
            "gt_used_for_alignment": False,
            "future_frames_used_for_bridge": False,
            "bridge_post_frames": 1,
            "history_window": int(args.history_window),
            "post_frames_evaluated": int(args.post_frames),
            "explicit_and_hybrid_same_refinement_budget": True,
            "refinement_iterations": int(args.refine_iters),
            "one_se3_per_shot": True,
        },
        "cases": case_reports,
        "aggregate": aggregate,
        "by_difficulty": group_summary(case_reports, variant_names),
        "token_analysis": token_value_analysis(case_reports),
        "timing": aggregate_timing(case_reports),
    }
    report["plots"] = plot_summary(args.output_dir, case_reports, variant_names)
    analysis = args.output_dir / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    write_summary_csv(analysis / "all_case_variant_metrics.csv", case_reports, variant_names)
    (args.output_dir / "implicit_explicit_probe_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_markdown(args.output_dir / "implicit_explicit_probe_metrics.md", report)
    print(json.dumps({"aggregate": aggregate, "token_analysis": report["token_analysis"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
