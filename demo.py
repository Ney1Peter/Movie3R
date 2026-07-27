#!/usr/bin/env python3
"""
Modified from CUT3R: https://github.com/CUT3R/CUT3R

Online Human-Scene Reconstruction Inference and Visualization Script

This script performs inference using the ARCroco3DStereo model and visualizes the
resulting 3D scene point clouds and SMPLX sequences with the SceneHumanViewer. 
Use the command-line arguments to adjust parameters 
such as the model checkpoint path, image sequence directory, image size, device, etc.

Example:
    python demo.py --size 512 \
        --seq_path examples/GoodMornin1.mp4 --subsample 1 --vis_threshold 2 \
        --downsample_factor 1 --use_ttt3r --reset_interval 100
"""

import os
import numpy as np
import torch
import time
import glob
import random
import cv2
import argparse
import tempfile
import shutil
from copy import deepcopy
from add_ckpt_path import add_path_to_dust3r
import imageio.v2 as iio
import roma

# Set random seed for reproducibility.
random.seed(42)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run 3D point cloud inference and visualization using ARCroco3DStereo."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="experiments/formal_training-4gpu/checkpoint-final.pth",
        help="Path to the pretrained model checkpoint.",
    )
    parser.add_argument(
        "--seq_path",
        type=str,
        default="",
        help="Path to the directory containing the image sequence.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (e.g., 'cuda' or 'cpu').",
    )
    parser.add_argument(
        "--size",
        type=int,
        default="512",
        help="Shape that input images will be rescaled to; if using 224+linear model, choose 224 otherwise 512",
    )
    parser.add_argument(
        "--vis_threshold",
        type=float,
        default=1.5,
        help="Visualization threshold for the viewer. Ranging from 1 to INF",
    )
    parser.add_argument(
        "--msk_threshold",
        type=float,
        default=0.1,
        help="Mask threshold. Ranging from 0 to 1",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./tmp",
        help="value for tempfile.tempdir",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save output results.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Save smpl mesh projection.",
    )
    parser.add_argument(
        "--render_video",
        action="store_true",
        help="Save smpl mesh projection video.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Max frames to use. Default is None (use all images).",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=1,
        help="Subsample factor for input images. Default is 1 (use all images).",
    )
    parser.add_argument(
        "--reset_interval", 
        type=int, 
        default=10000000
        )
    parser.add_argument(
        "--freeze_state_after",
        type=int,
        default=None,
        help=(
            "Inference-only state-write ablation. Frames with original index >= "
            "this value still run forward but do not update recurrent state, "
            "pose memory, or V9 correction history."
        ),
    )
    parser.add_argument(
        "--freeze_state_feat_after",
        type=int,
        default=None,
        help=(
            "Inference-only ablation. Frames with original index >= this value "
            "do not update the global recurrent state_feat."
        ),
    )
    parser.add_argument(
        "--freeze_pose_memory_after",
        type=int,
        default=None,
        help=(
            "Inference-only ablation. Frames with original index >= this value "
            "do not update the pose retriever memory."
        ),
    )
    parser.add_argument(
        "--freeze_v9_history_after",
        type=int,
        default=None,
        help=(
            "Inference-only ablation. Frames with original index >= this value "
            "do not update V9 correction-token history."
        ),
    )
    parser.add_argument(
        "--use_ttt3r",
        action="store_true",
        help="Use TTT3R.",
        default=False
    )
    parser.add_argument(
        "--downsample_factor",
        type=int,
        default=10,
        help="Point cloud downsample factor for the viewer",
    )
    parser.add_argument(
        "--smpl_downsample",
        type=int,
        default=1,
        help="SMPL sequence downsample factor for the viewer",
    )
    parser.add_argument(
        "--camera_downsample",
        type=int,
        default=1,
        help="Camera motion downsample factor for the viewer",
    )
    parser.add_argument(
        "--mask_morph",
        type=int,
        default=10,
        help="Mask morphology for the viewer",
    )
    parser.add_argument(
        "--disable_shot_adaptation",
        action="store_true",
        help="Disable ShotToken path after loading checkpoint for ablation.",
    )
    parser.add_argument(
        "--disable_shot_decoder_token",
        action="store_true",
        help="Do not append q_t to decoder tokens; keep it only as LoRA condition.",
    )
    parser.add_argument(
        "--disable_pose_lora",
        action="store_true",
        help="Disable PoseLoRA camera pose correction for ablation.",
    )
    parser.add_argument(
        "--disable_pose_translation_adapter",
        action="store_true",
        help="Disable V3 translation-only camera adapter for ablation.",
    )
    # **========== V3 当前代码备份：推理只支持关闭 translation-only adapter ==========**
    # parser.add_argument(
    #     "--disable_pose_translation_adapter",
    #     action="store_true",
    #     help="Disable V3 translation-only camera adapter for ablation.",
    # )
    # **========== 结束 ==========**
    parser.add_argument(
        "--disable_pose_alignment_adapter",
        action="store_true",
        help="Disable V4 pose-only ShotToken alignment adapter for ablation.",
    )
    parser.add_argument(
        "--disable_layerwise_pose_shot_adapter",
        action="store_true",
        help="Disable V5.1 layerwise pose-only ShotToken adapter for ablation.",
    )
    parser.add_argument(
        "--disable_pose_alignment_rotation",
        action="store_true",
        help="Keep camera rotation fixed in V4 pose alignment adapter.",
    )
    parser.add_argument(
        "--disable_human_lora",
        action="store_true",
        help="Disable HumanLoRA SMPL translation correction for ablation.",
    )
    parser.add_argument(
        "--disable_world_lora",
        action="store_true",
        help="Disable WorldLoRA pointmap correction for ablation.",
    )
    parser.add_argument(
        "--disable_anchor_pose_adapter",
        action="store_true",
        help="Disable V6-A/V6.1 decoder-after AnchorPoseAdapter for ablation.",
    )
    parser.add_argument(
        "--disable_v8_pose_prompt",
        action="store_true",
        help="Disable V8/V9 correct-token prompt branch for inference ablation.",
    )
    parser.add_argument(
        "--disable_v8_human_latent_corr",
        action="store_true",
        help="Disable V8/V9 human latent correction head for inference ablation.",
    )
    parser.add_argument(
        "--disable_v8_human_trans_corr",
        action="store_true",
        help="Disable V8/V9 human translation correction head for inference ablation.",
    )
    parser.add_argument(
        "--disable_v8_head_lora",
        action="store_true",
        help="Disable V8/V9 pose/human head LoRA modules for inference ablation.",
    )
    parser.add_argument(
        "--disable_v8_pose_head_lora",
        action="store_true",
        help="Disable only the V8/V9 pose-head LoRA modules for inference ablation.",
    )
    parser.add_argument(
        "--disable_v8_human_head_lora",
        action="store_true",
        help="Disable only the V8/V9 human-head LoRA modules for inference ablation.",
    )
    # V6.1: real-video validation path. External XFeat anchors are built outside
    # demo.py and injected here as patch-level fields consumed by AnchorPoseAdapter.
    parser.add_argument(
        "--anchor_path",
        type=str,
        default=None,
        help="Optional video anchor npz generated by scripts/build_video_anchor_cache.py.",
    )
    parser.add_argument(
        "--viewer_port",
        type=int,
        default=8080,
        help="Port for the Human3R/viser viewer.",
    )
    parser.add_argument(
        "--cut_indices",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Zero-based frame indices marked as explicit camera-cut events. "
            "V14.1 applies correction only at these frames."
        ),
    )
    # V6.1: camera-only viewer mode is a debugging convenience for checking
    # whether shot-boundary camera frustums move without loading full point clouds.
    parser.add_argument(
        "--viewer_camera_only",
        action="store_true",
        help="Only load camera frustums in the viewer to avoid browser overload.",
    )
    return parser.parse_args()


def prepare_input(
    img_paths, 
    img_mask, 
    size, 
    raymaps=None, 
    raymap_mask=None, 
    revisit=1, 
    update=True, 
    img_res=None, 
    reset_interval=100,
    freeze_state_after=None,
    freeze_state_feat_after=None,
    freeze_pose_memory_after=None,
    freeze_v9_history_after=None,
):
    """
    Prepare input views for inference from a list of image paths.

    Args:
        img_paths (list): List of image file paths.
        img_mask (list of bool): Flags indicating valid images.
        size (int): Target image size.
        raymaps (list, optional): List of ray maps.
        raymap_mask (list, optional): Flags indicating valid ray maps.
        revisit (int): How many times to revisit each view.
        update (bool): Whether to update the state on revisits.

    Returns:
        list: A list of view dictionaries.
    """
    # Import image loader (delayed import needed after adding ckpt path).
    from src.dust3r.utils.image import load_images, pad_image
    from dust3r.utils.geometry import get_camera_parameters

    images = load_images(img_paths, size=size)
    if img_res is not None:
        K_mhmr = get_camera_parameters(img_res, device="cpu") # if use pseudo K

    views = []

    def _should_update_frame(i, freeze_after):
        return not (freeze_after is not None and i >= int(freeze_after))

    def _update_flags(i):
        base_update = _should_update_frame(i, freeze_state_after)
        return {
            "update": base_update,
            "update_state": base_update and _should_update_frame(i, freeze_state_feat_after),
            "update_mem": base_update and _should_update_frame(i, freeze_pose_memory_after),
            "update_v8_history": base_update and _should_update_frame(i, freeze_v9_history_after),
        }

    if raymaps is None and raymap_mask is None:
        # Only images are provided.
        for i in range(len(images)):
            update_flags = _update_flags(i)
            view = {
                "img": images[i]["img"],
                "ray_map": torch.full(
                    (
                        images[i]["img"].shape[0],
                        6,
                        images[i]["img"].shape[-2],
                        images[i]["img"].shape[-1],
                    ),
                    torch.nan,
                ),
                "true_shape": torch.from_numpy(images[i]["true_shape"]),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.from_numpy(
                    np.eye(4, dtype=np.float32)
                    ).unsqueeze(0),
                "img_mask": torch.tensor(True).unsqueeze(0),
                "ray_mask": torch.tensor(False).unsqueeze(0),
                "update": torch.tensor(update_flags["update"]).unsqueeze(0),
                "update_state": torch.tensor(update_flags["update_state"]).unsqueeze(0),
                "update_mem": torch.tensor(update_flags["update_mem"]).unsqueeze(0),
                "update_v8_history": torch.tensor(update_flags["update_v8_history"]).unsqueeze(0),
                "reset": torch.tensor((i+1) % reset_interval == 0).unsqueeze(0),
            }
            if img_res is not None:
                view["img_mhmr"] = pad_image(view["img"], img_res)
                view["K_mhmr"] = K_mhmr
            views.append(view)
            if (i+1) % reset_interval == 0:
                overlap_view = deepcopy(view)
                overlap_view["reset"] = torch.tensor(False).unsqueeze(0)
                views.append(overlap_view)
    else:
        # Combine images and raymaps.
        num_views = len(images) + len(raymaps)
        assert len(img_mask) == len(raymap_mask) == num_views
        assert sum(img_mask) == len(images) and sum(raymap_mask) == len(raymaps)

        j = 0
        k = 0
        for i in range(num_views):
            update_flags = _update_flags(i)
            view = {
                "img": (
                    images[j]["img"]
                    if img_mask[i]
                    else torch.full_like(images[0]["img"], torch.nan)
                ),
                "ray_map": (
                    raymaps[k]
                    if raymap_mask[i]
                    else torch.full_like(raymaps[0], torch.nan)
                ),
                "true_shape": (
                    torch.from_numpy(images[j]["true_shape"])
                    if img_mask[i]
                    else torch.from_numpy(np.int32([raymaps[k].shape[1:-1][::-1]]))
                ),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.from_numpy(
                    np.eye(4, dtype=np.float32)
                    ).unsqueeze(0),
                "img_mask": torch.tensor(img_mask[i]).unsqueeze(0),
                "ray_mask": torch.tensor(raymap_mask[i]).unsqueeze(0),
                "update": torch.tensor(img_mask[i] and update_flags["update"]).unsqueeze(0),
                "update_state": torch.tensor(img_mask[i] and update_flags["update_state"]).unsqueeze(0),
                "update_mem": torch.tensor(img_mask[i] and update_flags["update_mem"]).unsqueeze(0),
                "update_v8_history": torch.tensor(img_mask[i] and update_flags["update_v8_history"]).unsqueeze(0),
                "reset": torch.tensor((i+1) % reset_interval == 0).unsqueeze(0),
            }
            if img_res is not None:
                view["img_mhmr"] = pad_image(view["img"], img_res)
                view["K_mhmr"] = K_mhmr
            if img_mask[i]:
                j += 1
            if raymap_mask[i]:
                k += 1
            views.append(view)
            if (i+1) % reset_interval == 0:
                overlap_view = deepcopy(view)
                overlap_view["reset"] = torch.tensor(False).unsqueeze(0)
                views.append(overlap_view)
        assert j == len(images) and k == len(raymaps)

    if revisit > 1:
        new_views = []
        for r in range(revisit):
            for i, view in enumerate(views):
                new_view = deepcopy(view)
                new_view["idx"] = r * len(views) + i
                new_view["instance"] = str(r * len(views) + i)
                if r > 0 and not update:
                    new_view["update"] = torch.tensor(False).unsqueeze(0)
                    new_view["update_state"] = torch.tensor(False).unsqueeze(0)
                    new_view["update_mem"] = torch.tensor(False).unsqueeze(0)
                    new_view["update_v8_history"] = torch.tensor(False).unsqueeze(0)
                new_views.append(new_view)
        return new_views

    return views


def _pad_anchor_array(array, shape, dtype):
    out = np.zeros(shape, dtype=dtype)
    if array is None:
        return out
    arr = np.asarray(array, dtype=dtype)
    n = min(shape[0], arr.shape[0])
    if n > 0:
        out[:n] = arr[:n]
    return out


def inject_video_anchor(views, anchor_path):
    """V6.1: inject external video anchor metadata into demo views.

    The npz is generated by scripts/build_video_anchor_cache.py. It contains
    RGB-space XFeat matches that have already been mapped to Human3R patch-grid
    indices, so this function only pads fields and attaches them to cur_view.
    """
    data = np.load(anchor_path)
    top_k = int(data["top_k_tokens"][0]) if "top_k_tokens" in data.files else 16
    ref_idx = int(data["ref_view_idx"][0])
    cur_idx = int(data["cur_view_idx"][0])
    if ref_idx < 0 or ref_idx >= len(views) or cur_idx < 0 or cur_idx >= len(views):
        raise ValueError(
            f"anchor view indices out of range: ref={ref_idx}, cur={cur_idx}, num_views={len(views)}"
        )

    ref_patch_idx = _pad_anchor_array(data["ref_patch_idx"], (top_k,), np.int64)
    cur_patch_idx = _pad_anchor_array(data["cur_patch_idx"], (top_k,), np.int64)
    ref_pos_norm = _pad_anchor_array(data["ref_pos_norm"], (top_k, 2), np.float32)
    cur_pos_norm = _pad_anchor_array(data["cur_pos_norm"], (top_k, 2), np.float32)
    if "local_residual_norm" in data.files:
        local_residual_norm = _pad_anchor_array(data["local_residual_norm"], (top_k, 2), np.float32)
    else:
        local_residual_norm = ref_pos_norm - cur_pos_norm
    confidence = _pad_anchor_array(data["confidence"], (top_k,), np.float32)
    if "anchor_mask" in data.files:
        anchor_mask = _pad_anchor_array(data["anchor_mask"], (top_k,), np.bool_)
    else:
        anchor_mask = np.arange(top_k) < min(top_k, len(data["ref_patch_idx"]))
    quality_gate = float(np.asarray(data["quality_gate"], dtype=np.float32).reshape(-1)[0])

    views[cur_idx].update(
        anchor_valid=torch.tensor([bool(anchor_mask.any() and quality_gate > 0.0)]),
        anchor_ref_view_idx=torch.tensor([ref_idx], dtype=torch.long),
        anchor_cur_view_idx=torch.tensor([cur_idx], dtype=torch.long),
        anchor_ref_patch_idx=torch.from_numpy(ref_patch_idx).long().unsqueeze(0),
        anchor_cur_patch_idx=torch.from_numpy(cur_patch_idx).long().unsqueeze(0),
        anchor_ref_pos_norm=torch.from_numpy(ref_pos_norm).float().unsqueeze(0),
        anchor_cur_pos_norm=torch.from_numpy(cur_pos_norm).float().unsqueeze(0),
        anchor_local_residual_norm=torch.from_numpy(local_residual_norm).float().unsqueeze(0),
        anchor_confidence=torch.from_numpy(confidence).float().unsqueeze(0),
        anchor_quality_gate=torch.tensor([[quality_gate]], dtype=torch.float32),
        anchor_mask=torch.from_numpy(anchor_mask).bool().unsqueeze(0),
        shot_label=torch.tensor([1.0], dtype=torch.float32),
    )
    print(
        f"Injected video anchors from {anchor_path}: ref_view={ref_idx}, cur_view={cur_idx}, "
        f"valid_anchors={int(anchor_mask.sum())}, quality_gate={quality_gate:.4f}"
    )
    return views

def prepare_output(
        outputs, outdir, revisit=1, use_pose=True, 
        save=False, render=False, render_video=False, img_res=None, subsample=1):
    """
    Process inference outputs to generate point clouds and camera parameters for visualization.

    Args:
        outputs (dict): Inference outputs.
        revisit (int): Number of revisits per view.
        use_pose (bool): Whether to transform points using camera pose.
        save (bool): Whether to save output results.
        render (bool): Whether to save smpl mesh projection.
        render_video (bool): Whether to save smpl mesh projection video.
    """
    from src.dust3r.utils.camera import pose_encoding_to_camera
    from src.dust3r.post_process import estimate_focal_knowing_depth
    from src.dust3r.utils.geometry import geotrf, matrix_cumprod
    from src.dust3r.utils import SMPL_Layer, vis_heatmap, render_meshes
    from src.dust3r.utils.image import unpad_image
    from viser_utils import get_color

    # Only keep the outputs corresponding to one full pass.
    valid_length = len(outputs["pred"]) // revisit
    outputs["pred"] = outputs["pred"][-valid_length:]
    outputs["views"] = outputs["views"][-valid_length:]

    # delet overlaps: reset_mask=True outputs["pred"] and outputs["views"]
    reset_mask = torch.cat([view["reset"] for view in outputs["views"]], 0)
    shifted_reset_mask = torch.cat([torch.tensor(False).unsqueeze(0), reset_mask[:-1]], dim=0)
    outputs["pred"] = [
        pred for pred, mask in zip(outputs["pred"], shifted_reset_mask) if not mask]
    outputs["views"] = [
        view for view, mask in zip(outputs["views"], shifted_reset_mask) if not mask]
    reset_mask = reset_mask[~shifted_reset_mask]

    pts3ds_self_ls = [output["pts3d_in_self_view"] for output in outputs["pred"]]
    pts3ds_other = [output["pts3d_in_other_view"] for output in outputs["pred"]]
    conf_self = [output["conf_self"] for output in outputs["pred"]]
    conf_other = [output["conf"] for output in outputs["pred"]]
    pts3ds_self = torch.cat(pts3ds_self_ls, 0)

    # Recover camera poses.
    pr_poses = [
        pose_encoding_to_camera(pred["camera_pose"].clone()).cpu()
        for pred in outputs["pred"]
    ]

    # reset_mask = torch.cat([view["reset"] for view in outputs["views"]], 0)
    if reset_mask.any():
        pr_poses = torch.cat(pr_poses, 0)
        identity = torch.eye(4, device=pr_poses.device)
        reset_poses = torch.where(reset_mask.unsqueeze(-1).unsqueeze(-1), pr_poses, identity)
        cumulative_bases = matrix_cumprod(reset_poses)
        shifted_bases = torch.cat([identity.unsqueeze(0), cumulative_bases[:-1]], dim=0)
        pr_poses = torch.einsum('bij,bjk->bik', shifted_bases, pr_poses)
        # keeps only reset_mask=False pr_poses
        pr_poses = list(pr_poses.unsqueeze(1).unbind(0))

    R_c2w = torch.cat([pr_pose[:, :3, :3] for pr_pose in pr_poses], 0)
    t_c2w = torch.cat([pr_pose[:, :3, 3] for pr_pose in pr_poses], 0)

    if use_pose:
        transformed_pts3ds_other = []
        for pose, pself in zip(pr_poses, pts3ds_self):
            transformed_pts3ds_other.append(geotrf(pose, pself.unsqueeze(0)))
        pts3ds_other = transformed_pts3ds_other
        conf_other = conf_self

    # Estimate focal length based on depth.
    B, H, W, _ = pts3ds_self.shape
    pp = torch.tensor([W // 2, H // 2], device=pts3ds_self.device).float().repeat(B, 1)
    focal = estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld")

    colors = [
        0.5 * (output["img"].permute(0, 2, 3, 1) + 1.0) for output in outputs["views"]
    ]

    cam_dict = {
        "focal": focal.numpy(),
        "pp": pp.numpy(),
        "R": R_c2w.numpy(),
        "t": t_c2w.numpy(),
    }

    pts3ds_self_tosave = pts3ds_self  # B, H, W, 3
    depths_tosave = pts3ds_self_tosave[..., 2]
    pts3ds_other_tosave = torch.cat(pts3ds_other)  # B, H, W, 3
    conf_self_tosave = torch.cat(conf_self)  # B, H, W
    conf_other_tosave = torch.cat(conf_other)  # B, H, W
    colors_tosave = torch.cat(
        [
            0.5 * (output["img"].permute(0, 2, 3, 1) + 1.0)
            for output in outputs["views"]
        ]
    )  # [B, H, W, 3]
    cam2world_tosave = torch.cat(pr_poses)  # B, 4, 4
    intrinsics_tosave = (
        torch.eye(3).unsqueeze(0).repeat(cam2world_tosave.shape[0], 1, 1)
    )  # B, 3, 3
    intrinsics_tosave[:, 0, 0] = focal.detach()
    intrinsics_tosave[:, 1, 1] = focal.detach()
    intrinsics_tosave[:, 0, 2] = pp[:, 0]
    intrinsics_tosave[:, 1, 2] = pp[:, 1]

    # get SMPL parameters from outputs
    smpl_shape = [output.get(
        "smpl_shape", torch.empty(1,0,10))[0] for output in outputs["pred"]]
    smpl_rotvec = [roma.rotmat_to_rotvec(
        output.get(
            "smpl_rotmat", torch.empty(1,0,53,3,3))[0]) for output in outputs["pred"]]
    smpl_transl = [output.get(
        "smpl_transl", torch.empty(1,0,3))[0] for output in outputs["pred"]]
    smpl_expression = [output.get(
        "smpl_expression", [None])[0] for output in outputs["pred"]]
    smpl_id = [output.get(
        "smpl_id", torch.empty(1,0))[0] for output in outputs["pred"]]
    # smpl_loc = [output.get(
    #     "smpl_loc", torch.empty(1,0,2))[0] for output in outputs["pred"]]
    # K_mhmr = [output.get(
    #     "K_mhmr", torch.empty(1,0,3))[0] for output in outputs["views"]]
        
    if render or save:
        smpl_scores = [
            output.get("smpl_scores", torch.zeros(1, H, W, 1))[...,0] for output in outputs["pred"]]
        if img_res is not None:
            smpl_scores = [
                unpad_image(s, [H, W])[0] for s in smpl_scores]

    has_mask = "msk" in outputs["pred"][0]
    if has_mask:
        msks = [output["msk"][...,0] for output in outputs["pred"]]
        if img_res is not None:
            msks = [unpad_image(m, [H, W]) for m in msks]
    else:
        msks = [torch.zeros(1, H, W) for _ in range(B)]

    # SMPL layer
    smpl_layer = SMPL_Layer(type='smplx', 
                            gender='neutral', 
                            num_betas=smpl_shape[0].shape[-1], 
                            kid=False, 
                            person_center='head')
    smpl_faces = smpl_layer.bm_x.faces

    if save:
        print(f"Saving output to {outdir}...")
        os.makedirs(os.path.join(outdir, "depth"), exist_ok=True)
        os.makedirs(os.path.join(outdir, "conf"), exist_ok=True)
        os.makedirs(os.path.join(outdir, "color"), exist_ok=True)
        os.makedirs(os.path.join(outdir, "camera"), exist_ok=True)
        os.makedirs(os.path.join(outdir, "smpl"), exist_ok=True)

    all_verts = []
    for f_id in range(B):
        n_humans_i = smpl_shape[f_id].shape[0]
        
        if n_humans_i > 0:
            with torch.no_grad():
                smpl_out = smpl_layer(
                    smpl_rotvec[f_id], 
                    smpl_shape[f_id], 
                    smpl_transl[f_id], 
                    None, None, 
                    K=intrinsics_tosave[f_id].expand(n_humans_i, -1 , -1), 
                    expression=smpl_expression[f_id])
        
        depth = depths_tosave[f_id].numpy()
        conf = conf_self_tosave[f_id].numpy()
        color = colors_tosave[f_id].numpy()
        c2w = cam2world_tosave[f_id].numpy()
        intrins = intrinsics_tosave[f_id].numpy()

        if n_humans_i > 0:
            # transform smpl verts to world coordinates
            all_verts.append(geotrf(pr_poses[f_id], smpl_out['smpl_v3d'].unsqueeze(0))[0])
            pr_verts = [t.numpy() for t in smpl_out['smpl_v3d'].unbind(0)]
            pr_faces = [smpl_faces] * n_humans_i
        else:
            pr_verts = []
            pr_faces = []
            all_verts.append(torch.empty(0))

        if render:
            hm = vis_heatmap(colors_tosave[f_id], smpl_scores[f_id]).numpy()
            img_array_np = (color * 255).astype(np.uint8)
            smpl_rend = render_meshes(img_array_np.copy(), pr_verts, pr_faces,
                                        {'focal': intrins[[0,1],[0,1]], 
                                        'princpt': intrins[[0,1],[-1,-1]]},
                                        color=[get_color(i)/255 for i in smpl_id[f_id]])
            if has_mask:
                msk_array_np = vis_heatmap(colors_tosave[f_id], msks[f_id][0]).numpy()
                color_smpl = np.concatenate([
                    img_array_np, 
                    (msk_array_np * 255).astype(np.uint8), 
                    (hm * 255).astype(np.uint8), 
                    smpl_rend], 1)
            else:
                color_smpl = np.concatenate([
                    img_array_np, 
                    (hm * 255).astype(np.uint8), 
                    smpl_rend], 1)
        
        if save:
            np.save(os.path.join(outdir, "depth", f"{f_id:06d}.npy"), depth)
            np.save(os.path.join(outdir, "conf", f"{f_id:06d}.npy"), conf)
            iio.imwrite(
                os.path.join(outdir, "color", f"{f_id:06d}.png"),
                (color * 255).astype(np.uint8),
            )
            np.savez(
                os.path.join(outdir, "camera", f"{f_id:06d}.npz"),
                pose=c2w,
                intrinsics=intrins,
            )
            np.savez(
                os.path.join(outdir, "smpl", f"{f_id:06d}.npz"),
                scores=smpl_scores[f_id].numpy(),
                msk=msks[f_id].numpy() if has_mask else None,
                shape=smpl_shape[f_id].numpy(),
                rotvec=smpl_rotvec[f_id].numpy(),
                transl=smpl_transl[f_id].numpy(),
                expression=smpl_expression[f_id].numpy() if smpl_expression[f_id] is not None else None
            )

        # Save smpl projection
        if render:
            os.makedirs(os.path.join(outdir, "color_smpl"), exist_ok=True)
            iio.imwrite(
                os.path.join(outdir, "color_smpl", f"{f_id:06d}.png"),
                color_smpl,
            )

    if render and render_video:
        print(f"Saving smpl mesh projection to {outdir}...")
        frames_dir = os.path.join(outdir, "color_smpl")
        video_path = os.path.join(outdir, "output_video.mp4")
        output_fps = 30 // subsample
        os.system(f'/usr/bin/ffmpeg -y -framerate {output_fps} -i "{frames_dir}/%06d.png" '
                f'-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
                f'-vcodec h264 -preset fast -profile:v baseline -pix_fmt yuv420p '
                f'-movflags +faststart -b:v 5000k "{video_path}"')
    
    return (
        pts3ds_other,
        colors, 
        conf_other, 
        cam_dict, 
        all_verts, 
        smpl_faces,
        smpl_id,
        msks
    )

def parse_seq_path(p):
    if os.path.isdir(p):
        img_paths = sorted(glob.glob(f"{p}/*"))
        tmpdirname = None
    else:
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file {p}")
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if video_fps == 0:
            cap.release()
            raise ValueError(f"Error: Video FPS is 0 for {p}")
        frame_interval = 1
        frame_indices = list(range(0, total_frames, frame_interval))
        print(
            f" - Video FPS: {video_fps}, Frame Interval: {frame_interval}, Total Frames to Read: {len(frame_indices)}"
        )
        img_paths = []
        tmpdirname = tempfile.mkdtemp()
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(tmpdirname, f"frame_{i}.jpg")
            cv2.imwrite(frame_path, frame)
            img_paths.append(frame_path)
        cap.release()
    return img_paths, tmpdirname


def run_inference(args):
    """
    Execute the full inference and visualization pipeline.

    Args:
        args: Parsed command-line arguments.
    """
    # Set up the computation device.
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Switching to CPU.")
        device = "cpu"

    # Add the checkpoint path (required for model imports in the dust3r package).
    add_path_to_dust3r(args.model_path)

    # Import model and inference functions after adding the ckpt path.
    from src.dust3r.inference import inference_recurrent_lighter
    from src.dust3r.model import ARCroco3DStereo
    from src.dust3r.v8_head_lora import set_lora_enabled
    from viser_utils import SceneHumanViewer

    # Prepare image file paths.
    img_paths, tmpdirname = parse_seq_path(args.seq_path)
    if not img_paths:
        print(f"No images found in {args.seq_path}. Please verify the path.")
        return
    
    if args.max_frames is not None:
        img_paths = img_paths[:args.max_frames]
    img_paths = img_paths[::args.subsample]

    print(f"Found {len(img_paths)} images in {args.seq_path}.")
    img_mask = [True] * len(img_paths)

    # Load and prepare the model.
    print(f"Loading model from {args.model_path}...")
    model = ARCroco3DStereo.from_pretrained(args.model_path).to(device)
    # **========== 原始代码备份：只支持整体关闭 Shot Adaptation ==========**
    # if args.disable_shot_adaptation and hasattr(model, "enable_shot_adaptation"):
    #     model.enable_shot_adaptation = False
    #     print("Shot adaptation disabled for ablation.")
    # **========== 结束 ==========**
    if args.disable_shot_adaptation and hasattr(model, "enable_shot_adaptation"):
        model.enable_shot_adaptation = False
        print("Shot adaptation disabled for ablation.")
    if args.disable_shot_decoder_token and hasattr(model, "enable_shot_decoder_token"):
        model.enable_shot_decoder_token = False
        print("Shot decoder token disabled for ablation.")
    if args.disable_pose_lora and hasattr(model, "enable_pose_lora"):
        model.enable_pose_lora = False
        print("PoseLoRA disabled for ablation.")
    if args.disable_pose_translation_adapter and hasattr(model, "enable_pose_translation_adapter"):
        model.enable_pose_translation_adapter = False
        print("PoseTranslationAdapter disabled for ablation.")
    # **========== V3 当前代码备份：关闭 translation-only adapter ==========**
    # if args.disable_pose_translation_adapter and hasattr(model, "enable_pose_translation_adapter"):
    #     model.enable_pose_translation_adapter = False
    #     print("PoseTranslationAdapter disabled for ablation.")
    # **========== 结束 ==========**
    if args.disable_pose_alignment_adapter and hasattr(model, "enable_pose_alignment_adapter"):
        model.enable_pose_alignment_adapter = False
        print("PoseAlignmentAdapter disabled for ablation.")
    if args.disable_layerwise_pose_shot_adapter and hasattr(model, "enable_layerwise_pose_shot_adapter"):
        model.enable_layerwise_pose_shot_adapter = False
        print("LayerwisePoseShotAdapter disabled for ablation.")
    if args.disable_pose_alignment_rotation and hasattr(model, "enable_pose_alignment_rotation"):
        model.enable_pose_alignment_rotation = False
        print("PoseAlignmentAdapter rotation disabled for ablation.")
    if args.disable_human_lora and hasattr(model, "enable_human_lora"):
        model.enable_human_lora = False
        print("HumanLoRA disabled for ablation.")
    if args.disable_world_lora and hasattr(model, "enable_world_lora"):
        model.enable_world_lora = False
        print("WorldLoRA disabled for ablation.")
    if args.disable_anchor_pose_adapter and hasattr(model, "enable_anchor_pose_adapter"):
        model.enable_anchor_pose_adapter = False
        print("AnchorPoseAdapter disabled for ablation.")
    if args.disable_v8_pose_prompt and hasattr(model, "enable_v8_pose_prompt"):
        model.enable_v8_pose_prompt = False
        print("V8/V9 pose prompt disabled for ablation.")
    if args.disable_v8_human_latent_corr and hasattr(model, "enable_v8_human_latent_corr"):
        model.enable_v8_human_latent_corr = False
        print("V8/V9 human latent correction disabled for ablation.")
    if args.disable_v8_human_trans_corr and hasattr(model, "enable_v8_human_trans_corr"):
        model.enable_v8_human_trans_corr = False
        print("V8/V9 human translation correction disabled for ablation.")
    disabled_lora = {}
    if args.disable_v8_head_lora or args.disable_v8_pose_head_lora:
        n = 0
        if hasattr(model.downstream_head, "pose_head"):
            n = set_lora_enabled(model.downstream_head.pose_head, False)
        disabled_lora["pose_head"] = n
    if args.disable_v8_head_lora or args.disable_v8_human_head_lora:
        n = 0
        for attr in ("deccam", "decpose", "decshape", "decexpression"):
            if hasattr(model.downstream_head, attr):
                n += set_lora_enabled(getattr(model.downstream_head, attr), False)
        disabled_lora["human_head"] = n
    if disabled_lora and hasattr(model, "enable_v8_head_lora"):
        model.enable_v8_head_lora = False
        print(f"V8/V9 head LoRA disabled for ablation: {disabled_lora}")
    model.eval()

    # Prepare input views.
    print("Preparing input views...")
    img_res = getattr(model, 'mhmr_img_res', None)
    views = prepare_input(
        img_paths=img_paths,
        img_mask=img_mask,
        size=args.size,
        revisit=1,
        update=True,
        img_res=img_res,
        reset_interval=args.reset_interval,
        freeze_state_after=args.freeze_state_after,
        freeze_state_feat_after=args.freeze_state_feat_after,
        freeze_pose_memory_after=args.freeze_pose_memory_after,
        freeze_v9_history_after=args.freeze_v9_history_after,
    )
    cut_indices = set(args.cut_indices or [])
    invalid_cut_indices = sorted(
        index for index in cut_indices if index < 0 or index >= len(views)
    )
    if invalid_cut_indices:
        raise ValueError(
            f"cut_indices outside input range [0, {len(views) - 1}]: "
            f"{invalid_cut_indices}"
        )
    for view_idx, view in enumerate(views):
        ref = view["img"]
        view["shot_label"] = torch.full(
            (ref.shape[0],),
            1.0 if view_idx in cut_indices else 0.0,
            device=ref.device,
            dtype=ref.dtype,
        )
    if cut_indices:
        print(f"Explicit cut events: {sorted(cut_indices)}")
    if args.freeze_state_after is not None:
        print(
            f"Freeze-state-after enabled: frames with original index >= "
            f"{args.freeze_state_after} will not write recurrent state, pose "
            f"memory, or V9 correction history."
        )
    if args.freeze_state_feat_after is not None:
        print(
            f"Freeze-state-feat-after enabled: frames with original index >= "
            f"{args.freeze_state_feat_after} will not write recurrent state_feat."
        )
    if args.freeze_pose_memory_after is not None:
        print(
            f"Freeze-pose-memory-after enabled: frames with original index >= "
            f"{args.freeze_pose_memory_after} will not write pose memory."
        )
    if args.freeze_v9_history_after is not None:
        print(
            f"Freeze-v9-history-after enabled: frames with original index >= "
            f"{args.freeze_v9_history_after} will not write V9 correction history."
        )
    # V6.1: ordinary mp4 inference has no dataset loader, so external anchor
    # fields are attached here before model.forward_recurrent_lighter().
    if args.anchor_path is not None:
        views = inject_video_anchor(views, args.anchor_path)

    if tmpdirname is not None:
        shutil.rmtree(tmpdirname)

    # Run inference.
    print("Running inference...")
    start_time = time.time()
    outputs, _ = inference_recurrent_lighter(
        views, model, device, use_ttt3r=args.use_ttt3r)
    total_time = time.time() - start_time
    per_frame_time = total_time / len(views)
    print(
        f"Inference completed in {total_time:.2f} seconds (average {per_frame_time:.2f} s per frame)."
    )

    # Process outputs for visualization.
    print("Preparing output for visualization...")
    (
        pts3ds_other, 
        colors, 
        conf, 
        cam_dict, 
        all_smpl_verts, 
        smpl_faces,
        smpl_id,
        msks,
        ) = prepare_output(
        outputs, args.output_dir, 1, True, 
        args.save, args.render, args.render_video, img_res, args.subsample
    )

    # Convert tensors to numpy arrays for visualization.
    pts3ds_to_vis = [p.cpu().numpy() for p in pts3ds_other]
    colors_to_vis = [c.cpu().numpy() for c in colors]
    msks_to_vis = [m.cpu().numpy() for m in msks]
    conf_to_vis = [c.cpu().numpy() for c in conf]
    edge_colors = [None] * len(pts3ds_to_vis)
    verts_to_vis = [p.cpu().numpy() for p in all_smpl_verts]

    if args.viewer_camera_only:
        dummy_pc = np.zeros((1, 1, 3), dtype=np.float32)
        dummy_color = np.zeros((1, 1, 3), dtype=np.float32)
        dummy_conf = np.zeros((1, 1), dtype=np.float32)
        pts3ds_to_vis = [dummy_pc.copy() for _ in pts3ds_to_vis]
        colors_to_vis = [dummy_color.copy() for _ in colors_to_vis]
        conf_to_vis = [dummy_conf.copy() for _ in conf_to_vis]
        msks_to_vis = [None for _ in msks_to_vis]
        verts_to_vis = [[] for _ in verts_to_vis]
        print("Viewer camera-only mode enabled: point clouds and SMPL meshes are hidden.")

    # Create and run the point cloud viewer.
    # **========== 原始代码备份：viewer 固定使用默认端口并加载完整场景 ==========**
    # print("Launching Human3R viewer...")
    # viewer = SceneHumanViewer(
    #     pts3ds_to_vis,
    #     colors_to_vis,
    #     conf_to_vis,
    #     cam_dict,
    #     verts_to_vis,
    #     smpl_faces,
    #     smpl_id,
    #     msks_to_vis,
    #     device=device,
    #     edge_color_list=edge_colors,
    #     show_camera=True,
    #     vis_threshold=args.vis_threshold,
    #     msk_threshold=args.msk_threshold,
    #     mask_morph=args.mask_morph,
    #     size = args.size,
    #     downsample_factor=args.downsample_factor,
    #     smpl_downsample_factor=args.smpl_downsample,
    #     camera_downsample_factor=args.camera_downsample
    # )
    # **========== 结束 ==========**
    print(f"Launching Human3R viewer on port {args.viewer_port}...")
    print(f"Open http://127.0.0.1:{args.viewer_port} after forwarding this port.")
    viewer = SceneHumanViewer(
        pts3ds_to_vis,
        colors_to_vis,
        conf_to_vis,
        cam_dict,
        verts_to_vis,
        smpl_faces,
        smpl_id,
        msks_to_vis,
        device=device,
        port=args.viewer_port,
        edge_color_list=edge_colors,
        show_camera=True,
        vis_threshold=args.vis_threshold,
        msk_threshold=args.msk_threshold,
        mask_morph=args.mask_morph,
        size=args.size,
        downsample_factor=args.downsample_factor,
        smpl_downsample_factor=args.smpl_downsample,
        camera_downsample_factor=args.camera_downsample,
    )
    viewer.run()


def main():
    args = parse_args()
    if not args.seq_path:
        print(
            "No inputs found! Please use our gradio demo if you would like to iteractively upload inputs."
        )
        return
    else:
        run_inference(args)


if __name__ == "__main__":
    main()



# cd /workspace/code/Movie3R

# export PYTHONPATH=src:. && .venv/bin/python demo.py \
#     --model_path experiments/formal_training-4gpu/checkpoint-final.pth \
#     --size 512 \
#     --seq_path data/h36.mp4 \
#     --subsample 1 \
#     --vis_threshold 2 \
#     --downsample_factor 1 \
#     --reset_interval 100 \
#     --output_dir ./output/h36_test \
#     --save \
#     --render \
#     --render_video
