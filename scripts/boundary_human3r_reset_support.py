#!/usr/bin/env python3
"""Internal Human3R reset/inference support for active boundary experiments.

This module retains the validated data-loading and fresh-shot inference helpers
from the legacy gauge-neutral probe. It is support code, not a new version.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import roma
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from add_ckpt_path import add_path_to_dust3r  # noqa: E402
from dust3r.datasets.avatarrex import AvatarReX_Pattern  # noqa: E402
from dust3r.model import ARCroco3DStereo  # noqa: E402
from dust3r.smpl_model import SMPLModel  # noqa: E402
from dust3r.utils.camera import pose_encoding_to_camera  # noqa: E402
from dust3r.utils.device import todevice  # noqa: E402
from dust3r.utils.image import pad_image  # noqa: E402
from dust3r.utils.smpl_layer import SMPL_Layer  # noqa: E402
from v10_oracle_state_vs_gauge_probe import rotation_error_deg  # noqa: E402
from v10_token_alignment_4source_probe import raw_roots_for_record, source_split_and_scope  # noqa: E402
from v9_learned_stream_alignment_overfit import gt_pose_from_view  # noqa: E402
from v9_online_stream_human3r_segment_align import strict_original_model  # noqa: E402


DEFAULT_RECORDS = (
    REPO_ROOT / "output" / "v10_candidate_selection" / "oracle_gt_4source" / "selected_records.jsonl"
)
DEFAULT_OUTPUT = REPO_ROOT / "output" / "v11_gauge_neutral_first_write" / "stage1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data"))
    parser.add_argument("--model_path", type=Path, default=REPO_ROOT / "src" / "human3r_896L.pth")
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--resolution", type=int, nargs=2, default=(512, 288))
    parser.add_argument("--resize_mode", default="human3r_demo")
    parser.add_argument("--max_post_frames", type=int, default=9)
    parser.add_argument("--warmup_frames", type=int, default=8)
    parser.add_argument("--point_sample", type=int, default=12000)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260718)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_model(args: argparse.Namespace) -> ARCroco3DStereo:
    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Human3R reset inference must run on CUDA")
    add_path_to_dust3r(str(args.model_path))
    model = ARCroco3DStereo.from_pretrained(str(args.model_path)).to(args.device).eval()
    strict_original_model(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def build_smpl_models(model: ARCroco3DStereo, device: torch.device) -> tuple[SMPLModel, SMPL_Layer]:
    gt_model = SMPLModel(
        device,
        model_args={
            "patch_size": model.croco_args["patch_size"],
            "mhmr_img_res": model.mhmr_img_res,
            "bb_patch_size": model.bb_patch_size,
        },
    )
    pred_layer = SMPL_Layer(
        type="smplx",
        gender="neutral",
        num_betas=10,
        kid=False,
        person_center="head",
    ).to(device).eval()
    return gt_model, pred_layer


def sequence_path(args: argparse.Namespace, record: dict, seq: str) -> Path:
    split, _ = source_split_and_scope(record)
    return args.data_root / split / seq


def record_spec(record: dict, args: argparse.Namespace) -> dict:
    seq_b = str(record["seqB"])
    start = int(record["start_frame"]) + 2
    rgb_dir = sequence_path(args, record, seq_b) / "rgb"
    frame_ids = sorted(int(path.stem) for path in rgb_dir.glob("*.png"))
    position = frame_ids.index(start)
    post = frame_ids[position : position + int(args.max_post_frames)]
    warmup_start = max(0, position - int(args.warmup_frames))
    warmup = frame_ids[warmup_start:position]
    return {
        "record": record,
        "post_frames": post,
        "warmup_frames": warmup,
        "post_count": len(post),
        "warmup_count": len(warmup),
    }


def pattern_record(spec: dict, teacher: bool) -> dict:
    record = spec["record"]
    seq_b = str(record["seqB"])
    frames = (spec["warmup_frames"] + spec["post_frames"]) if teacher else spec["post_frames"]
    return {
        "clip_type": "v11_teacher" if teacher else "v11_reset",
        "group": str(record.get("group", "")),
        "seqs": [seq_b] * len(frames),
        "frames": frames,
        "shot_labels": [0] * len(frames),
        "transition_angles_deg": [0.0] * len(frames),
        "view_angle_deg": float(record.get("view_angle_deg", 0.0)),
        "angle_bucket": str(record.get("angle_bucket", "unknown")),
        "pattern_id": f"{record['pattern_id']}_{'teacher' if teacher else 'reset'}",
    }


def build_dataset(specs: list[dict], teacher: bool, args: argparse.Namespace) -> AvatarReX_Pattern:
    record = specs[0]["record"]
    split, _ = source_split_and_scope(record)
    samples = [pattern_record(spec, teacher) for spec in specs]
    return AvatarReX_Pattern(
        split=split,
        ROOT=str(args.data_root),
        resolution=tuple(args.resolution),
        num_views=len(samples[0]["frames"]),
        aug_crop=0,
        allow_repeat=True,
        seed=401,
        n_corres=0,
        fixed_samples=samples,
        load_da3_depth=False,
        raw_calibration_root=raw_roots_for_record(record),
        resize_mode=str(args.resize_mode),
        max_humans=1,
    )


def configure_views(views: list[dict], device: torch.device, mhmr_img_res: int) -> list[dict]:
    for view in views:
        view["img_mhmr"] = pad_image(view["img"], int(mhmr_img_res))
        view["img_mask"] = torch.ones_like(view["img_mask"], dtype=torch.bool)
        view["ray_mask"] = torch.zeros_like(view["ray_mask"], dtype=torch.bool)
        view["update"] = torch.ones_like(view["img_mask"], dtype=torch.bool)
        view["update_state"] = torch.ones_like(view["img_mask"], dtype=torch.bool)
        view["update_mem"] = torch.ones_like(view["img_mask"], dtype=torch.bool)
        view["update_v8_history"] = torch.ones_like(view["img_mask"], dtype=torch.bool)
        view["reset"] = torch.zeros_like(view["img_mask"], dtype=torch.bool)
    return todevice(views, device)


def camera_matrix(prediction: dict) -> np.ndarray:
    value = pose_encoding_to_camera(prediction["camera_pose"].detach().float())
    return value[0].cpu().numpy().astype(np.float32)


def pointmap_camera(prediction: dict) -> np.ndarray:
    return prediction["pts3d_in_self_view"][0].detach().float().cpu().numpy().reshape(-1, 3).astype(np.float32)


def relative_pose(pose0: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return (np.linalg.inv(pose0) @ pose).astype(np.float32)


def finite_sample_indices(arrays: list[np.ndarray], count: int, seed: int) -> np.ndarray:
    size = min(len(array) for array in arrays)
    valid = np.ones(size, dtype=bool)
    for array in arrays:
        valid &= np.isfinite(array[:size]).all(axis=1)
    ids = np.flatnonzero(valid)
    if len(ids) > count:
        ids = np.random.default_rng(seed).choice(ids, size=count, replace=False)
    return ids


def torso_frame(joints: np.ndarray) -> np.ndarray:
    hip_mid = 0.5 * (joints[1] + joints[2])
    shoulder_mid = 0.5 * (joints[16] + joints[17])
    up = shoulder_mid - hip_mid
    right = joints[17] - joints[16]
    up /= max(float(np.linalg.norm(up)), 1e-8)
    right /= max(float(np.linalg.norm(right)), 1e-8)
    forward = np.cross(right, up)
    forward /= max(float(np.linalg.norm(forward)), 1e-8)
    right = np.cross(up, forward)
    right /= max(float(np.linalg.norm(right)), 1e-8)
    return np.stack([right, up, forward], axis=1).astype(np.float32)


def predicted_human(
    prediction: dict,
    intrinsic: torch.Tensor,
    layer: SMPL_Layer,
) -> dict | None:
    device = next(layer.parameters()).device
    rotmat = prediction.get("smpl_rotmat")
    shape = prediction.get("smpl_shape")
    transl = prediction.get("smpl_transl")
    if rotmat is None or shape is None or transl is None or rotmat.shape[1] == 0:
        return None
    rotmat = rotmat[:, 0].to(device=device, dtype=torch.float32)
    rotvec = roma.rotmat_to_rotvec(rotmat)
    expression = prediction.get("smpl_expression")
    if expression is None or expression.shape[1] == 0:
        expression = torch.zeros(transl.shape[0], 10, device=device, dtype=torch.float32)
    else:
        expression = expression[:, 0].to(device=device, dtype=torch.float32)
    output = layer(
        rotvec,
        shape[:, 0].to(device=device, dtype=torch.float32),
        transl[:, 0].to(device=device, dtype=torch.float32),
        None,
        None,
        K=intrinsic.to(device=device, dtype=torch.float32),
        expression=expression,
    )
    joints = output["smpl_j3d"][0].detach().float().cpu().numpy().astype(np.float32)
    return {
        "joints": joints,
        "root": joints[0],
        "torso": torso_frame(joints),
        "local_rotmat": rotmat[0, 1:].detach().cpu().numpy().astype(np.float32),
    }


def gt_human(view: dict) -> dict | None:
    mask = view.get("smpl_mask")
    if mask is None or not bool(mask[0, 0].detach().cpu()):
        return None
    joints = view["smpl_j3d"][0, 0].detach().float().cpu().numpy().astype(np.float32)
    rotmat = view.get("smpl_rotmat")
    local = None if rotmat is None else rotmat[0, 0, 1:].detach().float().cpu().numpy().astype(np.float32)
    return {"joints": joints, "root": joints[0], "torso": torso_frame(joints), "local_rotmat": local}


def rotation_batch_error(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return float("nan")
    count = min(len(a), len(b))
    relative = np.einsum("nij,njk->nik", a[:count], np.swapaxes(b[:count], -1, -2))
    trace = np.trace(relative, axis1=1, axis2=2)
    return float(np.degrees(np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))).mean())


def texture_score(view: dict) -> float:
    image = view["img"][0].detach().float()
    gray = image.mean(dim=0)
    dx = (gray[:, 1:] - gray[:, :-1]).abs().mean()
    dy = (gray[1:, :] - gray[:-1, :]).abs().mean()
    return float((dx + dy).detach().cpu())


def evaluate_case(
    spec: dict,
    reset_predictions: list[dict],
    teacher_predictions: list[dict],
    reset_views: list[dict],
    teacher_warmup: int,
    gt_model: SMPLModel,
    pred_layer: SMPL_Layer,
    args: argparse.Namespace,
    case_index: int,
    world_transform: np.ndarray | None = None,
    update_smpl_gt: bool = True,
) -> dict:
    teacher_post = teacher_predictions[teacher_warmup:]
    gt_poses = [gt_pose_from_view(view).detach().float().cpu().numpy().astype(np.float32) for view in reset_views]
    pred_poses = [camera_matrix(prediction) for prediction in reset_predictions]
    teacher_poses = [camera_matrix(prediction) for prediction in teacher_post]
    pred_points = [pointmap_camera(prediction) for prediction in reset_predictions]
    teacher_points = [pointmap_camera(prediction) for prediction in teacher_post]

    boundary_world = (
        gt_poses[0] @ np.linalg.inv(pred_poses[0])
        if world_transform is None
        else np.asarray(world_transform, dtype=np.float32)
    )
    teacher_world = gt_poses[0] @ np.linalg.inv(teacher_poses[0])
    aligned_pred_poses = [(boundary_world @ pose).astype(np.float32) for pose in pred_poses]
    aligned_teacher_poses = [(teacher_world @ pose).astype(np.float32) for pose in teacher_poses]

    if update_smpl_gt:
        with torch.no_grad():
            gt_model.update_smpl_gt(reset_views)
    with torch.no_grad():
        pred_humans = [
            predicted_human(prediction, reset_views[idx]["camera_intrinsics"], pred_layer)
            for idx, prediction in enumerate(reset_predictions)
        ]
    gt_humans = [gt_human(view) for view in reset_views]
    pred_root0 = None if pred_humans[0] is None else pred_humans[0]["root"]
    gt_root0 = None if gt_humans[0] is None else gt_humans[0]["root"]
    pred_torso0 = None if pred_humans[0] is None else pred_humans[0]["torso"]
    gt_torso0 = None if gt_humans[0] is None else gt_humans[0]["torso"]

    per_frame = []
    for offset in range(len(reset_predictions)):
        pred_rel = relative_pose(pred_poses[0], pred_poses[offset])
        gt_rel = relative_pose(gt_poses[0], gt_poses[offset])
        ids = finite_sample_indices(
            [pred_points[offset], teacher_points[offset], pred_points[0], teacher_points[0]],
            int(args.point_sample),
            args.seed + case_index * 31 + offset,
        )
        if len(ids):
            point_error = float(np.linalg.norm(pred_points[offset][ids] - teacher_points[offset][ids], axis=1).mean())
            depth_error = float(np.abs(pred_points[offset][ids, 2] - teacher_points[offset][ids, 2]).mean())
            pred_depth_delta = pred_points[offset][ids, 2] - pred_points[0][ids, 2]
            teacher_depth_delta = teacher_points[offset][ids, 2] - teacher_points[0][ids, 2]
            depth_consistency = float(np.abs(pred_depth_delta - teacher_depth_delta).mean())
            pred_scale = float(np.median(np.linalg.norm(pred_points[offset][ids], axis=1)))
            teacher_scale = float(np.median(np.linalg.norm(teacher_points[offset][ids], axis=1)))
            scale_error = abs(math.log(max(pred_scale, 1e-6) / max(teacher_scale, 1e-6)))
            pred_world_points = pred_points[offset][ids] @ aligned_pred_poses[offset][:3, :3].T + aligned_pred_poses[offset][:3, 3]
            teacher_world_points = teacher_points[offset][ids] @ aligned_teacher_poses[offset][:3, :3].T + aligned_teacher_poses[offset][:3, 3]
            world_point_error = float(np.linalg.norm(pred_world_points - teacher_world_points, axis=1).mean())
        else:
            point_error = depth_error = depth_consistency = scale_error = world_point_error = float("nan")

        human = pred_humans[offset]
        target_human = gt_humans[offset]
        if human is not None and target_human is not None and pred_root0 is not None and gt_root0 is not None:
            root_displacement = float(np.linalg.norm((human["root"] - pred_root0) - (target_human["root"] - gt_root0)))
            pred_centered = human["joints"] - human["root"]
            gt_centered = target_human["joints"] - target_human["root"]
            joint_count = min(len(pred_centered), len(gt_centered))
            root_centered = float(np.linalg.norm(pred_centered[:joint_count] - gt_centered[:joint_count], axis=1).mean())
            pred_torso_rel = pred_torso0.T @ human["torso"]
            gt_torso_rel = gt_torso0.T @ target_human["torso"]
            torso_relative = rotation_error_deg(pred_torso_rel, gt_torso_rel)
            local_pose = rotation_batch_error(human["local_rotmat"], target_human["local_rotmat"])
            pred_world_root = aligned_pred_poses[offset][:3, :3] @ human["root"] + aligned_pred_poses[offset][:3, 3]
            gt_world_root = gt_poses[offset][:3, :3] @ target_human["root"] + gt_poses[offset][:3, 3]
            world_root = float(np.linalg.norm(pred_world_root - gt_world_root))
        else:
            root_displacement = root_centered = torso_relative = local_pose = world_root = float("nan")

        per_frame.append(
            {
                "offset": offset,
                "camera_relative_translation_m": float(np.linalg.norm(pred_rel[:3, 3] - gt_rel[:3, 3])),
                "camera_relative_rotation_deg": rotation_error_deg(pred_rel, gt_rel),
                "camera_frame_pointmap_m": point_error,
                "camera_frame_depth_m": depth_error,
                "depth_consistency_m": depth_consistency,
                "local_scale_log_abs": float(scale_error),
                "root_centered_human_m": root_centered,
                "human_relative_root_m": root_displacement,
                "torso_relative_orientation_deg": torso_relative,
                "human_local_pose_deg": local_pose,
                "world_camera_translation_m": float(np.linalg.norm(aligned_pred_poses[offset][:3, 3] - gt_poses[offset][:3, 3])),
                "world_camera_rotation_deg": rotation_error_deg(aligned_pred_poses[offset], gt_poses[offset]),
                "world_pointmap_teacher_m": world_point_error,
                "world_human_root_m": world_root,
            }
        )

    future = per_frame[1:] if len(per_frame) > 1 else per_frame
    mean = {}
    for key in per_frame[0]:
        if key == "offset":
            continue
        values = np.asarray([row[key] for row in future], dtype=np.float64)
        mean[key] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")

    gt_roots = [human["root"] for human in gt_humans if human is not None]
    speed = float(np.mean([np.linalg.norm(gt_roots[i] - gt_roots[i - 1]) for i in range(1, len(gt_roots))])) if len(gt_roots) > 1 else 0.0
    record = spec["record"]
    return {
        "case_name": str(record["pattern_id"]),
        "record": record,
        "post_frames": list(spec["post_frames"]),
        "post_count": len(spec["post_frames"]),
        "warmup_count": teacher_warmup,
        "texture_score": texture_score(reset_views[0]),
        "human_count": 1,
        "human_speed_m_per_frame": speed,
        "boundary_transform_gt": boundary_world.tolist(),
        "boundary_checks": {
            "world_camera_translation_m": per_frame[0]["world_camera_translation_m"],
            "world_camera_rotation_deg": per_frame[0]["world_camera_rotation_deg"],
        },
        "mean_future": mean,
        "per_frame": per_frame,
    }


def run_group(
    specs: list[dict],
    model: ARCroco3DStereo,
    gt_model: SMPLModel,
    pred_layer: SMPL_Layer,
    args: argparse.Namespace,
    device: torch.device,
    start_index: int,
) -> list[dict]:
    reset_dataset = build_dataset(specs, False, args)
    teacher_dataset = build_dataset(specs, True, args)
    reset_loader = torch.utils.data.DataLoader(reset_dataset, batch_size=1, shuffle=False, num_workers=0)
    teacher_loader = torch.utils.data.DataLoader(teacher_dataset, batch_size=1, shuffle=False, num_workers=0)
    results = []
    for local_index, (reset_views, teacher_views) in enumerate(zip(reset_loader, teacher_loader)):
        reset_views = configure_views(reset_views, device, model.mhmr_img_res)
        teacher_views = configure_views(teacher_views, device, model.mhmr_img_res)
        with torch.no_grad():
            reset_predictions, _ = model.forward_recurrent_lighter(reset_views, str(device), ret_state=False, use_ttt3r=False)
            teacher_predictions, _ = model.forward_recurrent_lighter(teacher_views, str(device), ret_state=False, use_ttt3r=False)
        result = evaluate_case(
            specs[local_index],
            reset_predictions,
            teacher_predictions,
            reset_views,
            specs[local_index]["warmup_count"],
            gt_model,
            pred_layer,
            args,
            start_index + local_index,
        )
        results.append(result)
        print(
            f">> [{start_index + local_index + 1}] {result['case_name']} "
            f"RPE={result['mean_future']['camera_relative_rotation_deg']:.3f}deg "
            f"PM={result['mean_future']['camera_frame_pointmap_m']:.4f}m",
            flush=True,
        )
        del reset_views, teacher_views, reset_predictions, teacher_predictions
        torch.cuda.empty_cache()
    return results


def main() -> None:
    args = parse_args()
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("Invalid shard index")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(args.records)
    selected = [record for index, record in enumerate(records) if index % args.num_shards == args.shard_index]
    if args.max_cases > 0:
        selected = selected[: int(args.max_cases)]
    specs = [record_spec(record, args) for record in selected]
    groups: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for spec in specs:
        source = str(spec["record"]["source"])
        groups[(source, spec["post_count"], spec["warmup_count"])].append(spec)

    device = torch.device(args.device)
    model = build_model(args)
    gt_model, pred_layer = build_smpl_models(model, device)
    cases = []
    start = 0
    started = time.perf_counter()
    for key in sorted(groups):
        group = groups[key]
        print(f">> group={key} cases={len(group)}", flush=True)
        cases.extend(run_group(group, model, gt_model, pred_layer, args, device, start))
        start += len(group)
    report = {
        "experiment": "Legacy gauge-neutral audit retained as reset support",
        "shard_index": int(args.shard_index),
        "num_shards": int(args.num_shards),
        "case_count": len(cases),
        "elapsed_seconds": time.perf_counter() - started,
        "constraints": {
            "human3r_frozen": True,
            "gt_boundary_only_for_world_metrics": True,
            "gauge_neutral_metrics_use_no_boundary_transform": True,
            "teacher": "same B camera with pre-boundary warm-up",
        },
        "cases": cases,
    }
    output = args.output_dir / f"stage1_shard_{args.shard_index:02d}_of_{args.num_shards:02d}.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    print(f">> wrote {output}", flush=True)


if __name__ == "__main__":
    main()
