#!/usr/bin/env python3
"""Cache five-frame causal human history and one fresh frame for V18."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import roma
import torch
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from v11_gauge_neutral_first_write_probe import (  # noqa: E402
    DEFAULT_RECORDS,
    build_dataset,
    build_model,
    build_smpl_models,
    configure_views,
    predicted_human,
    read_jsonl,
    record_spec,
)
from v10_latent_activation_patching_probe import camera_matrix  # noqa: E402
from v12_build_gauge_neutral_teacher_cache import old_a_dataset  # noqa: E402
from v14_depth_free_world_memory_candidates import fixed_explicit_human3r_gauge  # noqa: E402
from v15_wide_baseline_boundary_bridge_candidates import predicted_human_summary  # noqa: E402
from v16_human_torso_candidates import predict_torso_frames, yaw_residual  # noqa: E402
from v9_learned_stream_alignment_overfit import gt_pose_from_view  # noqa: E402


DEFAULT_OUTPUT = REPO_ROOT / "output" / "v18_human_metric_translation" / "stream_cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--candidate_root",
        type=Path,
        default=REPO_ROOT / "output" / "v10_candidate_selection" / "oracle_gt_4source" / "cases",
    )
    parser.add_argument("--data_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data"))
    parser.add_argument("--model_path", type=Path, default=REPO_ROOT / "src" / "human3r_896L.pth")
    parser.add_argument("--device", required=True)
    parser.add_argument("--resolution", type=int, nargs=2, default=(512, 288))
    parser.add_argument("--resize_mode", default="human3r_demo")
    parser.add_argument("--warmup_frames", type=int, default=8)
    parser.add_argument("--history_frames", type=int, default=5)
    parser.add_argument("--max_post_frames", type=int, default=3)
    parser.add_argument("--max_yaw_residual_deg", type=float, default=45.0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--sources", nargs="*", default=())
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def one_batch(dataset) -> list[dict]:
    return next(iter(torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)))


def image_uint8(view: dict) -> np.ndarray:
    image = ((view["img"][0].detach().float().cpu() + 1.0) * 127.5).clamp(0.0, 255.0)
    return image.byte().permute(1, 2, 0).numpy()


def prediction_detail(prediction: dict, view: dict, pred_layer) -> dict:
    human = predicted_human(prediction, view["camera_intrinsics"], pred_layer)
    if human is None:
        raise ValueError("V18 requires one Human3R person per frame")
    rotmat = prediction["smpl_rotmat"][:, 0].detach().float()
    rotvec = roma.rotmat_to_rotvec(rotmat)[0].cpu().numpy().astype(np.float32)
    shape = prediction["smpl_shape"][0, 0].detach().float().cpu().numpy().astype(np.float32)
    transl = prediction["smpl_transl"][0, 0].detach().float().cpu().numpy().astype(np.float32)
    expression = prediction.get("smpl_expression")
    if expression is None or expression.shape[1] == 0:
        expression_np = np.zeros(10, dtype=np.float32)
    else:
        expression_np = expression[0, 0].detach().float().cpu().numpy().astype(np.float32)
    pose = camera_matrix(prediction)
    joints_cam = human["joints"].astype(np.float32)
    joints_world = (joints_cam @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32)
    return {
        "pose": pose,
        "intrinsics": view["camera_intrinsics"][0].detach().float().cpu().numpy().astype(np.float32),
        "rotvec": rotvec,
        "shape": shape,
        "transl": transl,
        "expression": expression_np,
        "joints_camera": joints_cam,
        "joints_world": joints_world,
    }


def copy_gt_params(view: dict) -> dict:
    def array(key: str) -> np.ndarray:
        return view[key][0, 0].detach().float().cpu().numpy().astype(np.float32)

    pose = np.concatenate(
        [
            array("smplx_root_pose").reshape(1, 3),
            array("smplx_body_pose").reshape(21, 3),
            array("smplx_left_hand_pose").reshape(15, 3),
            array("smplx_right_hand_pose").reshape(15, 3),
            array("smplx_jaw_pose").reshape(1, 3),
        ],
        axis=0,
    ).astype(np.float32)
    human_world = bool(view.get("human_params_are_world", torch.zeros(1, dtype=torch.bool))[0].detach().cpu())
    if human_world:
        world_to_camera = view["T_w2c"][0].detach().float().cpu().numpy().astype(np.float32)
        root_rotation = Rotation.from_rotvec(pose[0].astype(np.float64)).as_matrix()
        pose[0] = Rotation.from_matrix(world_to_camera[:3, :3] @ root_rotation).as_rotvec().astype(np.float32)
    return {
        "pose_camera": pose,
        "shape": array("smplx_shape"),
        "world_scale": float(array("smplx_world_scale").reshape(-1)[0]),
    }


def stack(details: list[dict], key: str) -> np.ndarray:
    return np.stack([detail[key] for detail in details])


def run_case(record: dict, model, gt_model, pred_layer, args: argparse.Namespace) -> dict:
    spec = record_spec(record, args)
    device = torch.device(args.device)
    reset_views = configure_views(one_batch(build_dataset([spec], False, args)), device, model.mhmr_img_res)[:1]
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
    gt_new_params = copy_gt_params(reset_views[0])
    old_gt_poses = np.stack(
        [gt_pose_from_view(view).detach().float().cpu().numpy().astype(np.float32) for view in old_views]
    )
    new_gt_pose = gt_pose_from_view(reset_views[0]).detach().float().cpu().numpy().astype(np.float32)
    gt_model.update_smpl_gt(old_views + reset_views)
    old_gt_joints_camera = np.stack(
        [view["smpl_j3d"][0, 0].detach().float().cpu().numpy().astype(np.float32) for view in old_views]
    )
    new_gt_joints_camera = reset_views[0]["smpl_j3d"][0, 0].detach().float().cpu().numpy().astype(np.float32)

    old_details = [
        prediction_detail(prediction, view, pred_layer)
        for prediction, view in zip(old_predictions, old_views)
    ]
    new_detail = prediction_detail(reset_predictions[0], reset_views[0], pred_layer)
    old_humans = [predicted_human_summary(prediction, view, pred_layer) for prediction, view in zip(old_predictions, old_views)]
    new_human = predicted_human_summary(reset_predictions[0], reset_views[0], pred_layer)
    predicted_frames, motion_diagnostics = predict_torso_frames(old_humans, 1)
    fixed, fixed_name = fixed_explicit_human3r_gauge(args.candidate_root, record["pattern_id"])
    torso_rotation, torso_diagnostics = yaw_residual(
        fixed[:3, :3], [new_human["torso"]], predicted_frames[:1], float(args.max_yaw_residual_deg)
    )
    torso = np.asarray(fixed, dtype=np.float32).copy()
    torso[:3, :3] = torso_rotation

    old_pose = old_details[-1]["pose"]
    new_pose = new_detail["pose"]
    old_from_raw = old_pose @ np.linalg.inv(old_gt_poses[-1])
    target_pose = old_from_raw @ new_gt_pose
    gt_boundary = target_pose @ np.linalg.inv(new_pose)
    gt_old_joints_world = np.stack(
        [joints @ pose[:3, :3].T + pose[:3, 3] for joints, pose in zip(old_gt_joints_camera, old_gt_poses)]
    )
    gt_new_joints_world = new_gt_joints_camera @ new_gt_pose[:3, :3].T + new_gt_pose[:3, 3]
    target_old_joints_world = np.stack(
        [joints @ old_from_raw[:3, :3].T + old_from_raw[:3, 3] for joints in gt_old_joints_world]
    )
    target_new_joints_world = gt_new_joints_world @ old_from_raw[:3, :3].T + old_from_raw[:3, 3]

    case_dir = args.output_dir / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    output_path = case_dir / f"{record['pattern_id']}.npz"
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
        new_pose=new_detail["pose"],
        new_intrinsics=new_detail["intrinsics"],
        new_rotvec=new_detail["rotvec"],
        new_shape=new_detail["shape"],
        new_transl=new_detail["transl"],
        new_expression=new_detail["expression"],
        new_joints_camera=new_detail["joints_camera"],
        new_joints_world=new_detail["joints_world"],
        new_image=image_uint8(reset_views[0]),
        old_gt_pose=old_gt_poses,
        new_gt_pose=new_gt_pose,
        old_gt_joints_camera=old_gt_joints_camera,
        new_gt_joints_camera=new_gt_joints_camera,
        old_gt_joints_target_world=target_old_joints_world,
        new_gt_joints_target_world=target_new_joints_world,
        old_gt_pose53_camera=np.stack([row["pose_camera"] for row in gt_old_params]),
        old_gt_shape=np.stack([row["shape"] for row in gt_old_params]),
        old_gt_world_scale=np.asarray([row["world_scale"] for row in gt_old_params], dtype=np.float32),
        new_gt_pose53_camera=gt_new_params["pose_camera"],
        new_gt_shape=gt_new_params["shape"],
        new_gt_world_scale=np.asarray(gt_new_params["world_scale"], dtype=np.float32),
        fixed_transform=np.asarray(fixed, dtype=np.float32),
        torso_transform=torso,
        gt_boundary=np.asarray(gt_boundary, dtype=np.float32),
        target_pose=np.asarray(target_pose, dtype=np.float32),
    )
    return {
        "case_name": record["pattern_id"],
        "source": record["source"],
        "record": record,
        "cache_path": str(output_path),
        "history_frames": count,
        "fixed_name": fixed_name,
        "motion_diagnostics": motion_diagnostics,
        "torso_diagnostics": torso_diagnostics,
        "inference_seconds": inference_seconds,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("V18 stream cache requires CUDA")
    if not 0 <= int(args.shard_index) < int(args.num_shards):
        raise ValueError("Invalid shard index")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / f"v18_stream_shard_{args.shard_index:02d}_of_{args.num_shards:02d}.json"
    if manifest_path.exists() and not args.overwrite:
        print(f">> exists {manifest_path}")
        return
    records = read_jsonl(args.records)
    if args.sources:
        allowed = set(args.sources)
        records = [record for record in records if str(record["source"]) in allowed]
    selected = [record for index, record in enumerate(records) if index % int(args.num_shards) == int(args.shard_index)]
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
            f"history={row['history_frames']} time={row['inference_seconds']:.2f}s",
            flush=True,
        )
        torch.cuda.empty_cache()
    payload = {
        "experiment": "V18 five-frame causal human stream cache",
        "shard_index": int(args.shard_index),
        "num_shards": int(args.num_shards),
        "case_count": len(rows),
        "protocol": {
            "human3r_frozen": True,
            "old_history_frames": int(args.history_frames),
            "post_cut_frames": 1,
            "raw_tokens_used": False,
            "gt_use": "diagnostics and projection partial oracles only",
        },
        "cases": rows,
    }
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f">> wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
