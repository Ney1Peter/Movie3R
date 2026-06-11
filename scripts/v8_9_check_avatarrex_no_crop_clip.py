#!/usr/bin/env python3
"""Sanity-check one AvatarReX clip with crop vs no-crop dataloader resize."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from dust3r.datasets.avatarrex import AvatarReX_AABB


DEFAULT_RAW_ROOTS = {
    "lbn1": "/data/wangzheng/iJCV-CODE/data/AvatarReX_raw_meta/lbn1",
    "zxc": "/data/wangzheng/iJCV-CODE/data/AvatarReX_raw_meta/zxc",
    "zzr": "/data/wangzheng/iJCV-CODE/data/AvatarReX_raw_meta/zzr",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data"))
    parser.add_argument("--split", default="Training")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--raw_roots", default=repr(DEFAULT_RAW_ROOTS))
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def parse_raw_roots(text: str):
    value = ast.literal_eval(text) if text.strip().startswith("{") else text
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items()}
    return str(value)


def tensor_image_to_bgr(img: torch.Tensor) -> np.ndarray:
    arr = img.detach().cpu().float().permute(1, 2, 0).numpy()
    arr = ((arr * 0.5 + 0.5).clip(0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def build_dataset(args: argparse.Namespace, resize_mode: str) -> AvatarReX_AABB:
    return AvatarReX_AABB(
        allow_repeat=True,
        split=args.split,
        ROOT=str(args.data_root),
        aug_crop=0,
        resolution=args.resolution,
        resize_mode=resize_mode,
        num_views=4,
        seed=901,
        n_corres=0,
        manifest_path=str(args.manifest),
        load_da3_depth=False,
        raw_calibration_root=parse_raw_roots(args.raw_roots),
    )


def camera_angle_deg(pose_a: np.ndarray, pose_b: np.ndarray) -> float:
    za = pose_a[:3, 2] / max(np.linalg.norm(pose_a[:3, 2]), 1e-12)
    zb = pose_b[:3, 2] / max(np.linalg.norm(pose_b[:3, 2]), 1e-12)
    return float(np.degrees(np.arccos(np.clip(float(np.dot(za, zb)), -1.0, 1.0))))


def to_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def load_views(args: argparse.Namespace, resize_mode: str):
    dataset = build_dataset(args, resize_mode)
    return dataset[0]


def stack_row(images: list[np.ndarray], label: str) -> np.ndarray:
    height = max(img.shape[0] for img in images)
    padded = []
    for img in images:
        top = (height - img.shape[0]) // 2
        bottom = height - img.shape[0] - top
        padded_img = cv2.copyMakeBorder(img, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
        cv2.putText(padded_img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        padded.append(padded_img)
    return np.concatenate(padded, axis=1)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    summary: dict[str, object] = {"manifest": str(args.manifest)}
    for mode in ("human3r_demo", "resize_only_16"):
        views = load_views(args, mode)
        images = [tensor_image_to_bgr(view["img"]) for view in views]
        mode_dir = args.output_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        for view_idx, (view, image) in enumerate(zip(views, images)):
            frame_name = str(view["label"]).replace("/", "_")
            cv2.imwrite(str(mode_dir / f"{view_idx:02d}_{frame_name}.png"), image)
        rows.append(stack_row(images, mode))
        poses = [to_numpy(view["raw_camera_pose"]).astype(np.float32) for view in views]
        summary[mode] = {
            "image_shapes_hwc": [list(img.shape) for img in images],
            "labels": [str(view["label"]) for view in views],
            "instances": [str(view["instance"]) for view in views],
            "has_raw_camera_pose": [bool("raw_camera_pose" in view) for view in views],
            "view0_to_view2_angle_deg": camera_angle_deg(poses[0], poses[2]),
            "view0_to_view2_center_dist": float(np.linalg.norm(poses[2][:3, 3] - poses[0][:3, 3])),
        }

    width = max(row.shape[1] for row in rows)
    padded_rows = []
    for row in rows:
        right = width - row.shape[1]
        padded_rows.append(cv2.copyMakeBorder(row, 0, 0, 0, right, cv2.BORDER_CONSTANT, value=(20, 20, 20)))
    montage = np.concatenate(padded_rows, axis=0)

    cv2.imwrite(str(args.output_dir / "crop_vs_no_crop_inputs.png"), montage)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
