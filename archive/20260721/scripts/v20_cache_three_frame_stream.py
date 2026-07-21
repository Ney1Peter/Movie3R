#!/usr/bin/env python3
"""Cache five pre-cut frames and three fresh post-cut Human3R frames for V20."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from v18_cache_human_stream import (  # noqa: E402
    DEFAULT_RECORDS,
    build_dataset,
    build_model,
    build_smpl_models,
    configure_views,
    copy_gt_params,
    gt_pose_from_view,
    image_uint8,
    old_a_dataset,
    one_batch,
    prediction_detail,
    read_jsonl,
    record_spec,
    stack,
)
from v10_latent_activation_patching_probe import camera_matrix  # noqa: E402


DEFAULT_OUTPUT = REPO_ROOT / "output" / "v20_shot_scale_consistency" / "stream3_cache"


def optional_prediction_detail(prediction: dict, view: dict, pred_layer) -> tuple[dict, bool]:
    try:
        return prediction_detail(prediction, view, pred_layer), True
    except ValueError:
        pose = camera_matrix(prediction)
        intrinsics = view["camera_intrinsics"][0].detach().float().cpu().numpy().astype(np.float32)
        return {
            "pose": pose,
            "intrinsics": intrinsics,
            "rotvec": np.full((53, 3), np.nan, dtype=np.float32),
            "shape": np.full(10, np.nan, dtype=np.float32),
            "transl": np.full(3, np.nan, dtype=np.float32),
            "expression": np.full(10, np.nan, dtype=np.float32),
            "joints_camera": np.full((22, 3), np.nan, dtype=np.float32),
            "joints_world": np.full((22, 3), np.nan, dtype=np.float32),
        }, False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data"))
    parser.add_argument("--model_path", type=Path, default=REPO_ROOT / "src" / "human3r_896L.pth")
    parser.add_argument("--device", required=True)
    parser.add_argument("--resolution", type=int, nargs=2, default=(512, 288))
    parser.add_argument("--resize_mode", default="human3r_demo")
    parser.add_argument("--warmup_frames", type=int, default=8)
    parser.add_argument("--history_frames", type=int, default=5)
    parser.add_argument("--max_post_frames", type=int, default=3)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--sources", nargs="*", default=())
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_case(record: dict, model, gt_model, pred_layer, args: argparse.Namespace) -> dict:
    output_path = args.output_dir / "cases" / f"{record['pattern_id']}.npz"
    if output_path.exists() and not args.overwrite:
        try:
            with np.load(output_path) as cached:
                if cached["old_pose"].shape[0] == 5 and cached["new_pose"].shape[0] == 3:
                    return {
                        "case_name": record["pattern_id"],
                        "source": record["source"],
                        "record": record,
                        "cache_path": str(output_path),
                        "history_frames": 5,
                        "post_frames": 3,
                        "inference_seconds": 0.0,
                        "reused": True,
                    }
        except (OSError, ValueError, EOFError):
            output_path.unlink(missing_ok=True)
    spec = record_spec(record, args)
    device = torch.device(args.device)
    reset_views = configure_views(
        one_batch(build_dataset([spec], False, args)), device, model.mhmr_img_res
    )[:3]
    old_views_all = configure_views(one_batch(old_a_dataset(spec, args)), device, model.mhmr_img_res)
    started = time.perf_counter()
    with torch.no_grad():
        old_predictions_all, _ = model.forward_recurrent_lighter(
            old_views_all, str(device), ret_state=False, use_ttt3r=False
        )
        reset_predictions, _ = model.forward_recurrent_lighter(
            reset_views, str(device), ret_state=False, use_ttt3r=False
        )
    inference_seconds = time.perf_counter() - started
    count = min(int(args.history_frames), len(old_predictions_all))
    old_predictions = old_predictions_all[-count:]
    old_views = old_views_all[-count:]

    gt_old_params = [copy_gt_params(view) for view in old_views]
    gt_new_params = [copy_gt_params(view) for view in reset_views]
    gt_model.update_smpl_gt(old_views + reset_views)
    old_details = [
        prediction_detail(prediction, view, pred_layer)
        for prediction, view in zip(old_predictions, old_views)
    ]
    new_rows = [
        optional_prediction_detail(prediction, view, pred_layer)
        for prediction, view in zip(reset_predictions, reset_views)
    ]
    new_details = [row[0] for row in new_rows]
    new_human_valid = np.asarray([row[1] for row in new_rows], dtype=np.bool_)
    if not bool(new_human_valid[0]):
        raise ValueError("V20 requires a valid Human3R person in the first post-cut frame")
    template = next(detail for detail, valid in new_rows if valid)
    for detail, valid in new_rows:
        if valid:
            continue
        for key in ("rotvec", "shape", "transl", "expression", "joints_camera", "joints_world"):
            detail[key] = np.full_like(template[key], np.nan, dtype=np.float32)
    old_gt_poses = np.stack(
        [gt_pose_from_view(view).detach().float().cpu().numpy().astype(np.float32) for view in old_views]
    )
    new_gt_poses = np.stack(
        [gt_pose_from_view(view).detach().float().cpu().numpy().astype(np.float32) for view in reset_views]
    )
    old_gt_joints_camera = np.stack(
        [view["smpl_j3d"][0, 0].detach().float().cpu().numpy().astype(np.float32) for view in old_views]
    )
    new_gt_joints_camera = np.stack(
        [view["smpl_j3d"][0, 0].detach().float().cpu().numpy().astype(np.float32) for view in reset_views]
    )
    old_pose = old_details[-1]["pose"]
    old_from_gt = old_pose @ np.linalg.inv(old_gt_poses[-1])
    target_poses = np.stack([old_from_gt @ pose for pose in new_gt_poses]).astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        old_pose=stack(old_details, "pose"),
        old_intrinsics=stack(old_details, "intrinsics"),
        old_rotvec=stack(old_details, "rotvec"),
        old_shape=stack(old_details, "shape"),
        old_transl=stack(old_details, "transl"),
        old_expression=stack(old_details, "expression"),
        old_joints_camera=stack(old_details, "joints_camera"),
        old_joints_world=stack(old_details, "joints_world"),
        old_images=np.stack([image_uint8(view) for view in old_views]),
        new_pose=stack(new_details, "pose"),
        new_intrinsics=stack(new_details, "intrinsics"),
        new_rotvec=stack(new_details, "rotvec"),
        new_shape=stack(new_details, "shape"),
        new_transl=stack(new_details, "transl"),
        new_expression=stack(new_details, "expression"),
        new_joints_camera=stack(new_details, "joints_camera"),
        new_joints_world=stack(new_details, "joints_world"),
        new_images=np.stack([image_uint8(view) for view in reset_views]),
        new_human_valid=new_human_valid,
        old_gt_pose=old_gt_poses,
        new_gt_pose=new_gt_poses,
        old_gt_joints_camera=old_gt_joints_camera,
        new_gt_joints_camera=new_gt_joints_camera,
        old_gt_pose53_camera=np.stack([row["pose_camera"] for row in gt_old_params]),
        old_gt_shape=np.stack([row["shape"] for row in gt_old_params]),
        old_gt_world_scale=np.asarray([row["world_scale"] for row in gt_old_params], dtype=np.float32),
        new_gt_pose53_camera=np.stack([row["pose_camera"] for row in gt_new_params]),
        new_gt_shape=np.stack([row["shape"] for row in gt_new_params]),
        new_gt_world_scale=np.asarray([row["world_scale"] for row in gt_new_params], dtype=np.float32),
        target_pose=target_poses,
    )
    return {
        "case_name": record["pattern_id"],
        "source": record["source"],
        "record": record,
        "cache_path": str(output_path),
        "history_frames": count,
        "post_frames": len(new_details),
        "inference_seconds": inference_seconds,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("V20 three-frame stream cache requires CUDA")
    if not 0 <= int(args.shard_index) < int(args.num_shards):
        raise ValueError("Invalid shard index")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / f"v20_stream3_shard_{args.shard_index:02d}_of_{args.num_shards:02d}.json"
    if manifest.exists() and not args.overwrite:
        print(f">> exists {manifest}")
        return
    records = read_jsonl(args.records)
    if args.sources:
        allowed = set(args.sources)
        records = [record for record in records if str(record["source"]) in allowed]
    selected = [
        record
        for index, record in enumerate(records)
        if index % int(args.num_shards) == int(args.shard_index)
    ]
    if int(args.max_cases) > 0:
        selected = selected[: int(args.max_cases)]
    model = build_model(args)
    gt_model, pred_layer = build_smpl_models(model, torch.device(args.device))
    rows = []
    for index, record in enumerate(selected):
        row = run_case(record, model, gt_model, pred_layer, args)
        rows.append(row)
        print(
            f">> [{index + 1}/{len(selected)}] {record['pattern_id']} "
            f"history={row['history_frames']} post={row['post_frames']} "
            f"time={row['inference_seconds']:.2f}s",
            flush=True,
        )
        torch.cuda.empty_cache()
    payload = {
        "experiment": "V20 five-frame history and three-frame fresh Human3R cache",
        "case_count": len(rows),
        "shard_index": int(args.shard_index),
        "num_shards": int(args.num_shards),
        "protocol": {"human3r_frozen": True, "old_history_frames": 5, "post_cut_frames": 3},
        "cases": rows,
    }
    manifest.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f">> wrote {manifest}", flush=True)


if __name__ == "__main__":
    main()
