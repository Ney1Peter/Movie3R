#!/usr/bin/env python3
"""Compare V14.1 training and demo forward paths on one exact dataset sample."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument(
        "--manifest_path",
        type=Path,
        default=Path("config/manifests/v14_1_cut_event/single/lbn1_1192.jsonl"),
    )
    parser.add_argument(
        "--resize_mode",
        choices=("human3r_demo", "resize_only_16"),
        default="human3r_demo",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--force_event_only_routing",
        action="store_true",
        help="Force formal V9 weights through V14.1 event-only token/LoRA routing.",
    )
    return parser.parse_args()


def _tensor_comparison(train_value, demo_value):
    if not torch.is_tensor(train_value) or not torch.is_tensor(demo_value):
        return None
    train_value = train_value.detach().float().cpu()
    demo_value = demo_value.detach().float().cpu()
    if train_value.shape != demo_value.shape:
        return {
            "train_shape": list(train_value.shape),
            "demo_shape": list(demo_value.shape),
        }
    delta = train_value - demo_value
    return {
        "shape": list(train_value.shape),
        "train_abs_mean": float(train_value.abs().mean()),
        "demo_abs_mean": float(demo_value.abs().mean()),
        "mae": float(delta.abs().mean()),
        "rmse": float(delta.square().mean().sqrt()),
        "max_abs": float(delta.abs().max()),
    }


def main() -> None:
    args = parse_args()

    from dust3r.datasets.avatarrex import AvatarReX_Pattern
    from dust3r.inference import (
        _compose_v14_1_dual_path_scene,
        _make_v14_1_event_off_batch,
        _make_v8_image_only_model_batch,
    )
    from dust3r.losses import V82PoseRelationLoss
    from dust3r.model import ARCroco3DStereo
    from dust3r.smpl_model import SMPLModel
    from dust3r.utils.device import todevice
    from dust3r.utils.camera import pose_encoding_to_camera
    from dust3r.utils.geometry import geotrf, resize_camera_intrinsics
    from dust3r.utils.image import pad_image

    raw_calibration_root = {
        "lbn1": "/data/wangzheng/iJCV-CODE/data/AvatarReX_raw_meta/lbn1",
        "lbn2": "/data/wangzheng/iJCV-CODE/data/AvatarReX_raw_meta/lbn2",
        "zzr": "/data/wangzheng/iJCV-CODE/data/AvatarReX_raw_meta/zzr",
        "zxc": "/data/wangzheng/iJCV-CODE/data/AvatarReX_raw_meta/zxc",
    }
    dataset = AvatarReX_Pattern(
        allow_repeat=True,
        split="Training",
        ROOT="/data/wangzheng/iJCV-CODE/data",
        aug_crop=0,
        resolution=512,
        resize_mode=args.resize_mode,
        num_views=3,
        seed=14102,
        n_corres=0,
        manifest_path=str(args.manifest_path.resolve()),
        load_da3_depth=False,
        raw_calibration_root=raw_calibration_root,
        max_humans=1,
    )
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)))
    num_views = len(batch)
    images = torch.stack([view["img"] for view in batch], dim=0)
    images = images.view(-1, *images.shape[2:])
    intrinsics = torch.stack([view["camera_intrinsics"] for view in batch], dim=0)
    intrinsics = intrinsics.view(-1, *intrinsics.shape[2:])
    intrinsics_mhmr = resize_camera_intrinsics(
        intrinsics, *images.shape[2:], model_mhmr_resolution := 896
    )
    images_mhmr = pad_image(images, model_mhmr_resolution)
    for view, image_mhmr, intrinsic_mhmr in zip(
        batch,
        images_mhmr.chunk(num_views, dim=0),
        intrinsics_mhmr.chunk(num_views, dim=0),
    ):
        view["img_mhmr"] = image_mhmr
        view["K_mhmr"] = intrinsic_mhmr
    model_batch = _make_v8_image_only_model_batch(batch)

    device = torch.device(args.device)
    gt_batch = todevice(copy.deepcopy(batch), device)
    model_batch = todevice(model_batch, device)
    model = ARCroco3DStereo.from_pretrained(str(args.model_path)).to(device).eval()
    if args.force_event_only_routing:
        model.v9_oracle_correction_gate_enabled = True
        model.v9_oracle_correction_inference_only = False
        model.v9_oracle_correction_cache_enabled = False
        model.v14_1_event_only_head_lora = True
    smpl_model = SMPLModel(
        device,
        model_args={
            "patch_size": model.croco_args["patch_size"],
            "mhmr_img_res": model.mhmr_img_res,
            "bb_patch_size": model.bb_patch_size,
        },
    )
    smpl_model.update_smpl_gt(gt_batch)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
        train_output = model(copy.deepcopy(model_batch))
        train_preds = train_output.ress
        demo_preds, _, _ = model.forward_recurrent_lighter(
            copy.deepcopy(model_batch), str(device), ret_state=True, use_ttt3r=False
        )
        event_off_batch = _make_v14_1_event_off_batch(copy.deepcopy(model_batch))
        demo_shadow_preds, _, _ = model.forward_recurrent_lighter(
            copy.deepcopy(event_off_batch), str(device), ret_state=True, use_ttt3r=False
        )
        dual_path_preds = _compose_v14_1_dual_path_scene(
            model_batch, demo_preds, demo_shadow_preds
        )

    criterion = V82PoseRelationLoss(
        translation_weight=1.0,
        rotation_weight=5.0,
        residual_weight=1.0e-5,
        drift_weight=0.0,
        improvement_weight=0.0,
        pose_key="raw_camera_pose",
        human_trans_weight=10.0,
        human_trans_delta_weight=1.0e-6,
        human_trans_supervise_from_view=0,
        pose_lora_norm_weight=0.0,
        human_lora_norm_weight=0.0,
        supervise_shot_label_only=True,
    )
    train_loss, train_metrics = criterion.compute_loss(gt_batch, train_preds)
    demo_loss, demo_metrics = criterion.compute_loss(gt_batch, demo_preds)

    selected_keys = (
        "camera_pose",
        "v8_raw_camera_pose",
        "smpl_transl",
        "v8_human_latent_corr_smpl_transl_raw",
        "pts3d_in_self_view",
        "pts3d_in_other_view",
        "v8_pose_prompt_gate",
        "v8_pose_prompt_delta_applied",
        "v8_human_latent_corr_gate",
        "v8_human_latent_corr_delta_applied",
        "v9_pre_decoder_append",
        "v9_oracle_force_gate",
    )
    report = {
        "model_path": str(args.model_path.resolve()),
        "manifest_path": str(args.manifest_path.resolve()),
        "resize_mode": args.resize_mode,
        "force_event_only_routing": bool(args.force_event_only_routing),
        "num_views": len(train_preds),
        "metrics": {
            "train": {"loss": float(train_loss.detach()), **train_metrics},
            "demo": {"loss": float(demo_loss.detach()), **demo_metrics},
        },
        "views": [],
    }
    for view_idx, (train_pred, demo_pred, demo_shadow_pred, dual_path_pred) in enumerate(
        zip(train_preds, demo_preds, demo_shadow_preds, dual_path_preds)
    ):
        view_report = {
            "view_idx": view_idx,
            "keys": {},
            "event_on_vs_off": {},
            "dual_path_vs_raw": {},
        }
        for key in selected_keys:
            comparison = _tensor_comparison(train_pred.get(key), demo_pred.get(key))
            if comparison is not None:
                view_report["keys"][key] = comparison
            elif key in train_pred or key in demo_pred:
                view_report["keys"][key] = {
                    "train_present": key in train_pred,
                    "demo_present": key in demo_pred,
                }
        for key in (
            "camera_pose",
            "smpl_transl",
            "smpl_shape",
            "smpl_rotmat",
            "smpl_expression",
            "pts3d_in_self_view",
            "pts3d_in_other_view",
        ):
            comparison = _tensor_comparison(
                demo_pred.get(key), demo_shadow_pred.get(key)
            )
            if comparison is not None:
                view_report["event_on_vs_off"][key] = comparison
            dual_comparison = _tensor_comparison(
                dual_path_pred.get(key), demo_shadow_pred.get(key)
            )
            if dual_comparison is not None:
                view_report["dual_path_vs_raw"][key] = dual_comparison
        camera_pose = demo_pred.get("camera_pose")
        raw_camera_pose = demo_shadow_pred.get("camera_pose")
        raw_world_pointmap = demo_shadow_pred.get("pts3d_in_other_view")
        world_pointmap = demo_pred.get("pts3d_in_other_view")
        if (
            camera_pose is not None
            and raw_camera_pose is not None
            and raw_world_pointmap is not None
            and world_pointmap is not None
        ):
            corrected_camera = pose_encoding_to_camera(camera_pose.float())
            raw_camera = pose_encoding_to_camera(raw_camera_pose.float())
            boundary = corrected_camera @ torch.linalg.inv(raw_camera)
            shared_target = geotrf(
                boundary,
                raw_world_pointmap.float(),
            )
            view_report["shared_transform_consistency"] = _tensor_comparison(
                world_pointmap, shared_target
            )
        report["views"].append(view_report)

    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
