#!/usr/bin/env python3
"""Run one V8.1 AABB token/proxy probe.

This script does not train or modify Human3R.  It uses original AvatarReX
calibration/SMPL to define human-centric regions, runs the frozen
Human3R/CUT3R encoder to dump image tokens, and writes visual checks for:

1. explicit region extraction,
2. encoder-token correspondence,
3. synthetic SE(3) pose correction from sampled anchors.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import smplx
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dust3r.model import ARCroco3DStereo  # noqa: E402
from dust3r.utils.image import ImgNorm  # noqa: E402


def install_cpu_attention_patch() -> None:
    """Keep RoPE in float32 on CPU.

    The upstream block casts q/k to fp16 before RoPE for CUDA speed.  The local
    CPU RoPE extension expects float tensors, so this patch is only installed for
    this offline validation script when no CUDA device is used.
    """

    from croco.models.blocks import Attention

    def forward_cpu(self, x, xpos):
        bsz, ntok, dim = x.shape
        qkv = (
            self.qkv(x)
            .reshape(bsz, ntok, 3, self.num_heads, dim // self.num_heads)
            .transpose(1, 3)
        )
        q, k, v = [qkv[:, :, i] for i in range(3)]
        q_type = q.dtype
        k_type = k.dtype
        if self.rope is not None:
            q = self.rope(q.float(), xpos)
            k = self.rope(k.float(), xpos)
            q = q.to(q_type)
            k = k.to(k_type)
        x_out = (
            F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                dropout_p=self.attn_drop.p,
                scale=self.scale,
            )
            .transpose(1, 2)
            .reshape(bsz, ntok, dim)
        )
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out

    Attention.forward = forward_cpu


@dataclass
class ProcessedView:
    view_idx: int
    seq: str
    frame: int
    label: str
    rgb_orig: np.ndarray
    rgb: np.ndarray
    mask: np.ndarray
    depth_m: np.ndarray
    pose: np.ndarray
    intrinsics: np.ndarray
    original_intrinsics: np.ndarray
    transform: dict
    anchors: dict[str, tuple[float, float]]
    masks: dict[str, np.ndarray]
    smpl_vertices_uv: np.ndarray
    smpl_joints_uv: np.ndarray
    smpl_bbox: np.ndarray
    smpl_projection_iou: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data/Avatarrex_output"))
    parser.add_argument("--avatarrex_raw_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data/avatarrex_lbn1"))
    parser.add_argument("--split", default="Training")
    parser.add_argument("--seq_a", default="22070932")
    parser.add_argument("--seq_b", default="22070935")
    parser.add_argument("--start_frame", type=int, default=820)
    parser.add_argument("--model_path", type=Path, default=REPO_ROOT / "src" / "human3r_896L.pth")
    parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "output" / "v8_1_probe_aabb_medium_case")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--drift_t", nargs=3, type=float, default=[0.35, -0.20, 0.25])
    parser.add_argument("--drift_r_deg", nargs=3, type=float, default=[0.0, 8.0, -6.0])
    parser.add_argument("--max_points", type=int, default=4096)
    parser.add_argument("--min_smpl_iou", type=float, default=0.50)
    parser.add_argument(
        "--skip_synthetic_correction",
        action="store_true",
        help="Only draw explicit overlays and token heatmaps. Useful for grouped RGB/mask/SMPL data without depth.",
    )
    return parser.parse_args()


class AvatarReXRawProjector:
    """Project original AvatarReX SMPL into the original RGB camera.

    AvatarReX calibration stores a world-to-camera transform:

        X_cam = R @ X_world + T

    Earlier V8.1 probes incorrectly treated the processed camera file as a
    standard c2w pose and guessed body anchors from the mask.  This helper keeps
    the convention explicit and uses original SMPL joints for all body anchors.
    """

    def __init__(self, raw_root: Path):
        self.raw_root = raw_root
        self.calibration_path = raw_root / "calibration_full.json"
        self.smpl_path = raw_root / "smpl_params.npz"
        if not self.calibration_path.is_file():
            raise FileNotFoundError(f"AvatarReX calibration not found: {self.calibration_path}")
        if not self.smpl_path.is_file():
            raise FileNotFoundError(f"AvatarReX SMPL params not found: {self.smpl_path}")

        with open(self.calibration_path, "r", encoding="utf-8") as f:
            self.calibration = json.load(f)
        self.smpl_data = np.load(self.smpl_path)
        self.model = smplx.create(
            str(SRC_ROOT / "models"),
            "smplx",
            gender="neutral",
            use_pca=False,
            flat_hand_mean=True,
            num_betas=10,
        ).eval()
        self._smpl_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def world_to_camera(self, seq: str, xyz_world: np.ndarray) -> np.ndarray:
        cal = self.calibration[seq]
        R = np.asarray(cal["R"], dtype=np.float64).reshape(3, 3)
        T = np.asarray(cal["T"], dtype=np.float64).reshape(3)
        return xyz_world @ R.T + T

    def camera_to_world_pose(self, seq: str) -> np.ndarray:
        cal = self.calibration[seq]
        R_w2c = np.asarray(cal["R"], dtype=np.float32).reshape(3, 3)
        T_w2c = np.asarray(cal["T"], dtype=np.float32).reshape(3)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R_w2c.T
        c2w[:3, 3] = -R_w2c.T @ T_w2c
        return c2w

    def intrinsics(self, seq: str) -> np.ndarray:
        return np.asarray(self.calibration[seq]["K"], dtype=np.float32).reshape(3, 3)

    def smpl_world(self, frame: int) -> tuple[np.ndarray, np.ndarray]:
        frame = int(frame)
        if frame not in self._smpl_cache:
            with torch.no_grad():
                out = self.model(
                    global_orient=torch.tensor(self.smpl_data["global_orient"][frame], dtype=torch.float32).reshape(1, 3),
                    body_pose=torch.tensor(self.smpl_data["body_pose"][frame], dtype=torch.float32).reshape(1, 63),
                    jaw_pose=torch.tensor(self.smpl_data["jaw_pose"][frame], dtype=torch.float32).reshape(1, 3),
                    leye_pose=torch.zeros(1, 3),
                    reye_pose=torch.zeros(1, 3),
                    left_hand_pose=torch.tensor(self.smpl_data["left_hand_pose"][frame], dtype=torch.float32).reshape(1, 45),
                    right_hand_pose=torch.tensor(self.smpl_data["right_hand_pose"][frame], dtype=torch.float32).reshape(1, 45),
                    betas=torch.tensor(self.smpl_data["betas"][0], dtype=torch.float32).reshape(1, 10),
                    transl=torch.tensor(self.smpl_data["transl"][frame], dtype=torch.float32).reshape(1, 3),
                )
            verts = out.vertices[0].cpu().numpy().astype(np.float64)
            joints = out.joints[0].cpu().numpy().astype(np.float64)
            self._smpl_cache[frame] = (verts, joints)
        return self._smpl_cache[frame]

    def project_world(self, seq: str, xyz_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        K = self.intrinsics(seq).astype(np.float64)
        xyz_cam = self.world_to_camera(seq, xyz_world)
        valid = np.isfinite(xyz_cam).all(axis=1) & (xyz_cam[:, 2] > 1e-4)
        uv = np.full((xyz_world.shape[0], 2), np.nan, dtype=np.float32)
        if valid.any():
            proj = xyz_cam[valid, :2] / xyz_cam[valid, 2:3]
            proj[:, 0] = proj[:, 0] * K[0, 0] + K[0, 2]
            proj[:, 1] = proj[:, 1] * K[1, 1] + K[1, 2]
            uv[valid] = proj.astype(np.float32)
        return uv, valid


def ensure_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "explicit": root / "explicit_overlays",
        "heatmaps": root / "token_heatmaps",
        "correction": root / "pose_correction",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return (mask > 10).astype(np.uint8)


def resize_crop_params(width: int, height: int, size: int) -> dict:
    scale = size / max(width, height)
    resized_w = int(round(width * scale))
    resized_h = int(round(height * scale))
    cx, cy = resized_w // 2, resized_h // 2
    halfw = ((2 * cx) // 16) * 8
    halfh = ((2 * cy) // 16) * 8
    if resized_w == resized_h:
        halfh = int(3 * halfw / 4)
    x0, x1 = int(cx - halfw), int(cx + halfw)
    y0, y1 = int(cy - halfh), int(cy + halfh)
    return {
        "scale": scale,
        "resized_w": resized_w,
        "resized_h": resized_h,
        "crop_x0": x0,
        "crop_y0": y0,
        "crop_x1": x1,
        "crop_y1": y1,
        "out_w": x1 - x0,
        "out_h": y1 - y0,
    }


def resize_crop_image(img: np.ndarray, params: dict, interpolation: int) -> np.ndarray:
    resized = cv2.resize(img, (params["resized_w"], params["resized_h"]), interpolation=interpolation)
    return resized[params["crop_y0"] : params["crop_y1"], params["crop_x0"] : params["crop_x1"]]


def update_intrinsics(K: np.ndarray, params: dict) -> np.ndarray:
    out = K.astype(np.float32).copy()
    out[0, :] *= params["scale"]
    out[1, :] *= params["scale"]
    out[0, 2] -= params["crop_x0"]
    out[1, 2] -= params["crop_y0"]
    return out


def resize_crop_uv(uv_orig: np.ndarray, params: dict) -> np.ndarray:
    uv = uv_orig.astype(np.float32).copy()
    valid = np.isfinite(uv).all(axis=1)
    uv[valid, 0] = uv[valid, 0] * params["scale"] - params["crop_x0"]
    uv[valid, 1] = uv[valid, 1] * params["scale"] - params["crop_y0"]
    return uv


def bbox_from_uv(uv: np.ndarray) -> np.ndarray:
    valid = np.isfinite(uv).all(axis=1)
    if not valid.any():
        return np.full((4,), np.nan, dtype=np.float32)
    pts = uv[valid]
    return np.array([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()], dtype=np.float32)


def bbox_from_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.full((4,), np.nan, dtype=np.float32)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    if not np.isfinite(box_a).all() or not np.isfinite(box_b).all():
        return 0.0
    ix0, iy0 = max(float(box_a[0]), float(box_b[0])), max(float(box_a[1]), float(box_b[1]))
    ix1, iy1 = min(float(box_a[2]), float(box_b[2])), min(float(box_a[3]), float(box_b[3]))
    iw, ih = max(0.0, ix1 - ix0 + 1.0), max(0.0, iy1 - iy0 + 1.0)
    inter = iw * ih
    area_a = max(0.0, float(box_a[2] - box_a[0] + 1.0)) * max(0.0, float(box_a[3] - box_a[1] + 1.0))
    area_b = max(0.0, float(box_b[2] - box_b[0] + 1.0)) * max(0.0, float(box_b[3] - box_b[1] + 1.0))
    return float(inter / (area_a + area_b - inter + 1e-8))


def add_circle(mask: np.ndarray, uv: tuple[float, float], radius: int) -> np.ndarray:
    out = mask.copy()
    if np.isfinite(uv).all():
        cv2.circle(out, (int(round(uv[0])), int(round(uv[1]))), radius, 1, -1)
    return out.astype(bool)


def valid_joint_uv(joints_uv: np.ndarray, idx: int, name: str) -> tuple[float, float]:
    uv = joints_uv[idx]
    if not np.isfinite(uv).all():
        raise RuntimeError(f"Invalid projected SMPL joint for {name} at index {idx}: {uv}")
    return float(uv[0]), float(uv[1])


def mean_valid_uv(joints_uv: np.ndarray, indices: list[int], name: str) -> tuple[float, float]:
    pts = joints_uv[indices]
    valid = np.isfinite(pts).all(axis=1)
    if not valid.any():
        raise RuntimeError(f"Invalid projected SMPL joints for {name}: {indices}")
    return float(pts[valid, 0].mean()), float(pts[valid, 1].mean())


def anchor_from_region(region: np.ndarray, preferred_uv: tuple[float, float]) -> tuple[float, float] | None:
    ys, xs = np.where(region)
    if len(xs) == 0:
        return None
    dx = xs.astype(np.float32) - float(preferred_uv[0])
    dy = ys.astype(np.float32) - float(preferred_uv[1])
    idx = int(np.argmin(dx * dx + dy * dy))
    return float(xs[idx]), float(ys[idx])


def make_smpl_anchors_and_regions(
    human_mask: np.ndarray,
    joints_uv: np.ndarray,
) -> tuple[dict[str, tuple[float, float]], dict[str, np.ndarray]]:
    h, w = human_mask.shape
    pelvis = valid_joint_uv(joints_uv, 0, "pelvis")
    torso = mean_valid_uv(joints_uv, [6, 9, 12], "torso")
    left_foot = valid_joint_uv(joints_uv, 10, "left_foot")
    right_foot = valid_joint_uv(joints_uv, 11, "right_foot")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (49, 49))
    dilated = cv2.dilate(human_mask.astype(np.uint8), kernel).astype(bool)
    near_human = dilated & (~human_mask)

    near_foot = np.zeros((h, w), dtype=bool)
    for foot in [left_foot, right_foot]:
        near_foot = add_circle(near_foot.astype(np.uint8), foot, 34)
    near_foot = near_foot & (~human_mask)

    local_masks = {
        "human": human_mask,
        "pelvis": add_circle(np.zeros((h, w), dtype=np.uint8), pelvis, 20) & human_mask,
        "torso": add_circle(np.zeros((h, w), dtype=np.uint8), torso, 24) & human_mask,
        "left_foot": add_circle(np.zeros((h, w), dtype=np.uint8), left_foot, 18) & human_mask,
        "right_foot": add_circle(np.zeros((h, w), dtype=np.uint8), right_foot, 18) & human_mask,
        "near_human": near_human,
        "near_foot": near_foot,
    }

    near_human_anchor = anchor_from_region(near_human, torso) or torso
    near_foot_preferred = (
        0.5 * (left_foot[0] + right_foot[0]),
        0.5 * (left_foot[1] + right_foot[1]),
    )
    near_foot_anchor = anchor_from_region(near_foot, near_foot_preferred) or near_foot_preferred

    anchors = {
        "pelvis": pelvis,
        "torso": torso,
        "left_foot": left_foot,
        "right_foot": right_foot,
        "near_human": near_human_anchor,
        "near_foot": near_foot_anchor,
    }
    return anchors, local_masks


def load_view(
    args: argparse.Namespace,
    projector: AvatarReXRawProjector,
    view_idx: int,
    seq: str,
    frame: int,
    label: str,
) -> ProcessedView:
    root = args.root / args.split / seq
    stem = f"{frame:08d}"
    rgb_orig = read_rgb(root / "rgb" / f"{stem}.png")
    mask_orig = read_mask(root / "mask" / f"{stem}.png")
    depth_path = root / "depth" / f"{stem}.npy"
    if depth_path.is_file():
        depth_raw = np.load(depth_path)
        depth_m = depth_raw.astype(np.float32)
        if np.issubdtype(depth_raw.dtype, np.integer):
            depth_m /= 1000.0
    else:
        depth_m = np.zeros(mask_orig.shape, dtype=np.float32)
    pose = projector.camera_to_world_pose(seq)
    K_orig = projector.intrinsics(seq)
    params = resize_crop_params(rgb_orig.shape[1], rgb_orig.shape[0], args.size)
    rgb = resize_crop_image(rgb_orig, params, cv2.INTER_LANCZOS4)
    mask = resize_crop_image(mask_orig, params, cv2.INTER_NEAREST).astype(bool)
    depth = resize_crop_image(depth_m, params, cv2.INTER_NEAREST)
    K = update_intrinsics(K_orig, params)
    verts_world, joints_world = projector.smpl_world(frame)
    verts_uv_orig, _ = projector.project_world(seq, verts_world)
    joints_uv_orig, _ = projector.project_world(seq, joints_world)
    verts_uv = resize_crop_uv(verts_uv_orig, params)
    joints_uv = resize_crop_uv(joints_uv_orig, params)
    smpl_bbox = bbox_from_uv(verts_uv)
    anchors, region_masks = make_smpl_anchors_and_regions(mask, joints_uv)
    smpl_projection_iou = bbox_iou(smpl_bbox, bbox_from_mask(mask))
    return ProcessedView(
        view_idx=view_idx,
        seq=seq,
        frame=frame,
        label=label,
        rgb_orig=rgb_orig,
        rgb=rgb,
        mask=mask,
        depth_m=depth,
        pose=pose,
        intrinsics=K,
        original_intrinsics=K_orig,
        transform=params,
        anchors=anchors,
        masks=region_masks,
        smpl_vertices_uv=verts_uv,
        smpl_joints_uv=joints_uv,
        smpl_bbox=smpl_bbox,
        smpl_projection_iou=smpl_projection_iou,
    )


def make_mask_anchors(mask: np.ndarray) -> tuple[dict[str, tuple[float, float]], dict[str, np.ndarray]]:
    raise RuntimeError(
        "Mask-bbox anchor extraction is deprecated for V8.1. "
        "Use original AvatarReX calibration/SMPL projection via make_smpl_anchors_and_regions()."
    )


def draw_overlay(view: ProcessedView, output_path: Path) -> None:
    img = view.rgb.copy()
    overlay = img.copy()
    overlay[view.mask] = (0.65 * overlay[view.mask] + 0.35 * np.array([255, 40, 40])).astype(np.uint8)
    overlay[view.masks["near_human"]] = (0.6 * overlay[view.masks["near_human"]] + 0.4 * np.array([40, 180, 255])).astype(np.uint8)
    overlay[view.masks["near_foot"]] = (0.6 * overlay[view.masks["near_foot"]] + 0.4 * np.array([40, 255, 120])).astype(np.uint8)

    valid_verts = np.isfinite(view.smpl_vertices_uv).all(axis=1)
    if valid_verts.any():
        pts = view.smpl_vertices_uv[valid_verts]
        in_img = (
            (pts[:, 0] >= 0)
            & (pts[:, 0] < overlay.shape[1])
            & (pts[:, 1] >= 0)
            & (pts[:, 1] < overlay.shape[0])
        )
        pts = pts[in_img]
        step = max(1, len(pts) // 2200) if len(pts) else 1
        for x, y in pts[::step].astype(int):
            cv2.circle(overlay, (int(x), int(y)), 1, (255, 120, 20), -1)
        if np.isfinite(view.smpl_bbox).all():
            x0, y0, x1, y1 = np.round(view.smpl_bbox).astype(int)
            cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 120, 20), 2)

    colors = {
        "pelvis": (255, 255, 0),
        "torso": (255, 160, 0),
        "left_foot": (0, 255, 255),
        "right_foot": (0, 180, 255),
        "near_human": (0, 80, 255),
        "near_foot": (0, 255, 80),
    }
    for name, uv in view.anchors.items():
        x, y = int(round(uv[0])), int(round(uv[1]))
        cv2.circle(overlay, (x, y), 5, colors[name], -1)
        cv2.putText(overlay, name, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, colors[name], 1, cv2.LINE_AA)
        px0, py0 = (x // 16) * 16, (y // 16) * 16
        cv2.rectangle(overlay, (px0, py0), (px0 + 15, py0 + 15), colors[name], 1)

    joint_labels = {0: "pelvis", 10: "L_foot", 11: "R_foot", 15: "head"}
    for idx, name in joint_labels.items():
        if idx >= len(view.smpl_joints_uv):
            continue
        x, y = view.smpl_joints_uv[idx]
        if np.isfinite([x, y]).all() and 0 <= x < overlay.shape[1] and 0 <= y < overlay.shape[0]:
            cv2.circle(overlay, (int(round(x)), int(round(y))), 4, (0, 170, 255), -1)
            cv2.putText(overlay, name, (int(round(x)) + 5, int(round(y)) + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 170, 255), 1, cv2.LINE_AA)

    ys, xs = np.where(view.mask)
    if len(xs):
        cv2.rectangle(overlay, (xs.min(), ys.min()), (xs.max(), ys.max()), (255, 255, 255), 1)
    cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 44), (0, 0, 0), -1)
    cv2.putText(overlay, f"{view.label}: {view.seq}@{view.frame}", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(overlay, f"SMPL-mask IoU={view.smpl_projection_iou:.3f}", (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def image_tensor_from_rgb(rgb: np.ndarray) -> torch.Tensor:
    img = Image.fromarray(rgb)
    return ImgNorm(img)[None]


def encode_views(args: argparse.Namespace, views: list[ProcessedView]) -> tuple[np.ndarray, tuple[int, int]]:
    print(f"Loading frozen encoder from {args.model_path} on {args.device}...")
    if str(args.device).startswith("cpu"):
        install_cpu_attention_patch()
    model = ARCroco3DStereo.from_pretrained(str(args.model_path)).to(args.device).float().eval()
    imgs = torch.cat([image_tensor_from_rgb(v.rgb) for v in views], dim=0).to(args.device)
    shapes = torch.tensor([[v.rgb.shape[0], v.rgb.shape[1]] for v in views], dtype=torch.int32, device=args.device)
    with torch.no_grad():
        feats, _, _ = model._encode_image(imgs, shapes)
    tokens = feats[-1].detach().float().cpu().numpy()
    grid_hw = (views[0].rgb.shape[0] // 16, views[0].rgb.shape[1] // 16)
    return tokens, grid_hw


def patch_index(uv: tuple[float, float], grid_hw: tuple[int, int]) -> tuple[int, int, int]:
    gh, gw = grid_hw
    x = int(np.clip(math.floor(uv[0] / 16.0), 0, gw - 1))
    y = int(np.clip(math.floor(uv[1] / 16.0), 0, gh - 1))
    return y * gw + x, x, y


def token_similarity_heatmap(tokens: np.ndarray, token_idx: int, grid_hw: tuple[int, int], out_hw: tuple[int, int]) -> np.ndarray:
    tok = tokens[token_idx]
    feat = tokens
    tok = tok / max(np.linalg.norm(tok), 1e-8)
    feat = feat / np.maximum(np.linalg.norm(feat, axis=1, keepdims=True), 1e-8)
    sim = feat @ tok
    sim_grid = sim.reshape(grid_hw)
    sim_grid = (sim_grid - sim_grid.min()) / max(float(sim_grid.max() - sim_grid.min()), 1e-8)
    heat = cv2.resize(sim_grid.astype(np.float32), (out_hw[1], out_hw[0]), interpolation=cv2.INTER_CUBIC)
    return heat


def save_heatmap(rgb: np.ndarray, heat: np.ndarray, output_path: Path, title: str) -> None:
    heat_u8 = np.clip(heat * 255.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    overlay = (0.52 * rgb + 0.48 * color).astype(np.uint8)
    cv2.putText(overlay, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def region_scores(heat: np.ndarray, view: ProcessedView, anchor: str) -> dict[str, float]:
    human = view.masks["human"]
    near_human = view.masks["near_human"]
    near_foot = view.masks["near_foot"]
    target = view.masks.get(anchor, human)
    outside = ~target
    return {
        "target_mean_sim": float(heat[target].mean()) if target.any() else 0.0,
        "outside_mean_sim": float(heat[outside].mean()) if outside.any() else 0.0,
        "human_mean_sim": float(heat[human].mean()) if human.any() else 0.0,
        "near_human_mean_sim": float(heat[near_human].mean()) if near_human.any() else 0.0,
        "near_foot_mean_sim": float(heat[near_foot].mean()) if near_foot.any() else 0.0,
    }


def backproject_points(view: ProcessedView, region: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    valid = region & np.isfinite(view.depth_m) & (view.depth_m > 0.05) & (view.depth_m < 20.0)
    ys, xs = np.where(valid)
    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
    if len(xs) > max_points:
        rng = np.random.default_rng(7)
        keep = rng.choice(len(xs), size=max_points, replace=False)
        xs, ys = xs[keep], ys[keep]
    z = view.depth_m[ys, xs].astype(np.float32)
    K = view.intrinsics
    x = (xs.astype(np.float32) - K[0, 2]) / K[0, 0] * z
    y = (ys.astype(np.float32) - K[1, 2]) / K[1, 1] * z
    pts_cam = np.stack([x, y, z], axis=-1).astype(np.float32)
    uv = np.stack([xs, ys], axis=-1).astype(np.float32)
    return pts_cam, uv


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ T[:3, :3].T + T[:3, 3]


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float32)


def so3_exp(rotvec_deg: list[float]) -> np.ndarray:
    r = np.radians(np.asarray(rotvec_deg, dtype=np.float32))
    theta = float(np.linalg.norm(r))
    if theta < 1e-8:
        return np.eye(3, dtype=np.float32)
    axis = r / theta
    K = skew(axis)
    return (np.eye(3, dtype=np.float32) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)).astype(np.float32)


def make_transform(t: list[float], r_deg: list[float]) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = so3_exp(r_deg)
    T[:3, 3] = np.asarray(t, dtype=np.float32)
    return T


def rigid_align(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src_c = src.mean(axis=0)
    dst_c = dst.mean(axis=0)
    X = src - src_c
    Y = dst - dst_c
    U, _, Vt = np.linalg.svd(X.T @ Y)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = dst_c - R @ src_c
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R.astype(np.float32)
    T[:3, 3] = t.astype(np.float32)
    return T


def rotation_error_deg(A: np.ndarray, B: np.ndarray) -> float:
    R = A[:3, :3] @ B[:3, :3].T
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def pose_error(T: np.ndarray, T_gt: np.ndarray) -> dict[str, float]:
    return {
        "translation_error": float(np.linalg.norm(T[:3, 3] - T_gt[:3, 3])),
        "rotation_error_deg": rotation_error_deg(T, T_gt),
    }


def run_synthetic_correction(args: argparse.Namespace, views: list[ProcessedView], out_dir: Path) -> dict:
    view = views[2]
    region = view.masks["near_human"] | view.masks["near_foot"]
    pts_cam, _ = backproject_points(view, region, args.max_points)
    if pts_cam.shape[0] < 8:
        region = view.masks["human"]
        pts_cam, _ = backproject_points(view, region, args.max_points)
    if pts_cam.shape[0] < 8:
        raise RuntimeError("Not enough valid depth points for synthetic correction")

    drift = make_transform(args.drift_t, args.drift_r_deg)
    T_gt = view.pose
    T_raw = drift @ T_gt
    pts_gt = transform_points(T_gt, pts_cam)
    pts_raw = transform_points(T_raw, pts_cam)
    delta = rigid_align(pts_raw, pts_gt)
    T_corr = delta @ T_raw
    pts_corr = transform_points(T_corr, pts_cam)

    before_res = np.linalg.norm(pts_raw - pts_gt, axis=-1)
    after_res = np.linalg.norm(pts_corr - pts_gt, axis=-1)
    summary = {
        "case_view": {"seq": view.seq, "frame": view.frame, "view_idx": view.view_idx},
        "num_anchor_points": int(pts_cam.shape[0]),
        "injected_drift": {"translation": args.drift_t, "rotation_deg": args.drift_r_deg},
        "raw_pose_error": pose_error(T_raw, T_gt),
        "corrected_pose_error": pose_error(T_corr, T_gt),
        "anchor_residual_before_mean": float(before_res.mean()),
        "anchor_residual_after_mean": float(after_res.mean()),
        "anchor_residual_before_median": float(np.median(before_res)),
        "anchor_residual_after_median": float(np.median(after_res)),
        "estimated_delta": delta.tolist(),
    }

    with open(out_dir / "synthetic_correction_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    step = max(1, pts_gt.shape[0] // 1200)
    ax.scatter(pts_gt[::step, 0], pts_gt[::step, 2], s=3, c="black", alpha=0.45, label="GT anchors")
    ax.scatter(pts_raw[::step, 0], pts_raw[::step, 2], s=3, c="red", alpha=0.35, label="raw drifted")
    ax.scatter(pts_corr[::step, 0], pts_corr[::step, 2], s=3, c="cyan", alpha=0.35, label="corrected")
    ax.set_xlabel("world X")
    ax.set_ylabel("world Z")
    ax.set_title("Synthetic drift correction on view2 anchors")
    ax.legend()
    ax.axis("equal")
    fig.tight_layout()
    fig.savefig(out_dir / "synthetic_correction_xz.png", dpi=180)
    plt.close(fig)

    centroids_gt, centroids_raw = [], []
    for i, v in enumerate(views):
        pts_i, _ = backproject_points(v, v.masks["human"], args.max_points)
        if pts_i.shape[0] == 0:
            centroids_gt.append([np.nan, np.nan, np.nan])
            centroids_raw.append([np.nan, np.nan, np.nan])
            continue
        T_i = T_raw if i == 2 else v.pose
        centroids_gt.append(transform_points(v.pose, pts_i).mean(axis=0))
        centroids_raw.append(transform_points(T_i, pts_i).mean(axis=0))
    centroids_gt = np.asarray(centroids_gt)
    centroids_raw = np.asarray(centroids_raw)
    centroids_corr = centroids_raw.copy()
    centroids_corr[2] = transform_points(T_corr, backproject_points(views[2], views[2].masks["human"], args.max_points)[0]).mean(axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    x = np.arange(len(views))
    axes[0].plot(x, centroids_gt[:, 0], "-o", label="GT X", color="black")
    axes[0].plot(x, centroids_raw[:, 0], "-o", label="raw X", color="red")
    axes[0].plot(x, centroids_corr[:, 0], "-o", label="corr X", color="cyan")
    axes[0].axvline(2, color="orange", linestyle="--", label="A->B")
    axes[0].set_ylabel("centroid world X")
    axes[0].legend()
    jump_gt = np.linalg.norm(np.diff(centroids_gt, axis=0), axis=1)
    jump_raw = np.linalg.norm(np.diff(centroids_raw, axis=0), axis=1)
    jump_corr = np.linalg.norm(np.diff(centroids_corr, axis=0), axis=1)
    axes[1].plot(np.arange(1, len(views)), jump_gt, "-o", label="GT jump", color="black")
    axes[1].plot(np.arange(1, len(views)), jump_raw, "-o", label="raw jump", color="red")
    axes[1].plot(np.arange(1, len(views)), jump_corr, "-o", label="corr jump", color="cyan")
    axes[1].axvline(2, color="orange", linestyle="--")
    axes[1].set_xlabel("view index")
    axes[1].set_ylabel("human centroid jump")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "synthetic_proxy_curves.png", dpi=180)
    plt.close(fig)
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dirs = ensure_dirs(args.output_dir)
    projector = AvatarReXRawProjector(args.avatarrex_raw_root)

    case = [
        (args.seq_a, args.start_frame, "view0_A_t"),
        (args.seq_a, args.start_frame + 1, "view1_A_t1"),
        (args.seq_b, args.start_frame + 2, "view2_B_t2_boundary"),
        (args.seq_b, args.start_frame + 3, "view3_B_t3"),
    ]
    views = [load_view(args, projector, i, seq, frame, label) for i, (seq, frame, label) in enumerate(case)]
    low_iou = [
        (v.label, v.seq, v.frame, v.smpl_projection_iou)
        for v in views
        if v.smpl_projection_iou < args.min_smpl_iou
    ]
    if low_iou:
        raise RuntimeError(
            "SMPL projection sanity check failed. "
            f"Expected bbox IoU >= {args.min_smpl_iou}, got {low_iou}. "
            "AvatarReX convention must be X_cam = R @ X_smpl + T."
        )

    manifest = {
        "dataset_root": str(args.root),
        "avatarrex_raw_root": str(args.avatarrex_raw_root),
        "camera_convention": "AvatarReX calibration is world-to-camera: X_cam = R @ X_world + T; saved c2w = inverse([R|T]).",
        "split": args.split,
        "seq_a": args.seq_a,
        "seq_b": args.seq_b,
        "start_frame": args.start_frame,
        "views": [
            {
                "idx": v.view_idx,
                "seq": v.seq,
                "frame": v.frame,
                "label": v.label,
                "smpl_projection_iou": v.smpl_projection_iou,
            }
            for v in views
        ],
        "shot_label": [0, 0, 1, 0],
        "size": args.size,
    }
    with open(args.output_dir / "case_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    for v in views:
        draw_overlay(v, dirs["explicit"] / f"view{v.view_idx}_{v.label}.png")

    tokens, grid_hw = encode_views(args, views)
    metrics_rows = []
    anchors_to_check = ["pelvis", "torso", "left_foot", "right_foot", "near_human", "near_foot"]
    for v in views:
        for anchor in anchors_to_check:
            idx, px, py = patch_index(v.anchors[anchor], grid_hw)
            heat = token_similarity_heatmap(tokens[v.view_idx], idx, grid_hw, v.rgb.shape[:2])
            heat_path = dirs["heatmaps"] / f"view{v.view_idx}_{anchor}_heatmap.png"
            save_heatmap(v.rgb, heat, heat_path, f"{v.label} {anchor} token similarity")
            scores = region_scores(heat, v, anchor)
            uv = v.anchors[anchor]
            in_image = 0 <= uv[0] < v.rgb.shape[1] and 0 <= uv[1] < v.rgb.shape[0]
            in_human = bool(v.mask[int(np.clip(round(uv[1]), 0, v.rgb.shape[0] - 1)), int(np.clip(round(uv[0]), 0, v.rgb.shape[1] - 1))]) if in_image else False
            metrics_rows.append(
                {
                    "view_idx": v.view_idx,
                    "label": v.label,
                    "seq": v.seq,
                    "frame": v.frame,
                    "anchor": anchor,
                    "uv_x": float(uv[0]),
                    "uv_y": float(uv[1]),
                    "patch_x": px,
                    "patch_y": py,
                    "patch_idx": idx,
                    "in_image": int(in_image),
                    "in_human_mask": int(in_human),
                    "target_pixels": int(v.masks.get(anchor, v.mask).sum()),
                    "human_pixels": int(v.mask.sum()),
                    "near_human_pixels": int(v.masks["near_human"].sum()),
                    "near_foot_pixels": int(v.masks["near_foot"].sum()),
                    "smpl_projection_iou": float(v.smpl_projection_iou),
                    **scores,
                }
            )

    with open(args.output_dir / "token_extraction_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metrics_rows)

    correction_summary = None
    if not args.skip_synthetic_correction:
        correction_summary = run_synthetic_correction(args, views, dirs["correction"])
    print(json.dumps({"output_dir": str(args.output_dir), "grid_hw": grid_hw, "correction": correction_summary}, indent=2))


if __name__ == "__main__":
    main()
