#!/usr/bin/env python3
"""Inspect V8.1 AvatarReX AABB samples before training.

The script does not run Human3R. It checks that the dataloader can select
large-view-change AABB samples and saves a reproducible manifest plus simple
RGB/mask montages for visual inspection.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from dust3r.datasets.avatarrex import AvatarReX_AABB, _avatarrex_load_camera_pose


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/data/wangzheng/iJCV-CODE/data"),
    )
    parser.add_argument("--split", default="training")
    parser.add_argument("--output_dir", type=Path, default=Path("output/v8_1_aabb_dataloader_check"))
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--min_view_angle_deg", type=float, default=60.0)
    parser.add_argument("--max_view_angle_deg", type=float, default=None)
    parser.add_argument("--pair_strategy", default="top_angle", choices=["all", "top_angle", "fixed"])
    parser.add_argument("--resolution", default="512,288", help="width,height")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_rgb(path: Path, size: tuple[int, int]) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img.resize(size, Image.BILINEAR)


def load_mask_overlay(rgb: Image.Image, mask_path: Path) -> Image.Image:
    if not mask_path.exists():
        return rgb.copy()
    mask = Image.open(mask_path).convert("L").resize(rgb.size, Image.NEAREST)
    overlay = Image.new("RGB", rgb.size, (255, 60, 40))
    return Image.composite(overlay, rgb, mask.point(lambda v: int(v * 0.35)))


def draw_label(img: Image.Image, text: str) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, out.size[0], 22), fill=(0, 0, 0))
    draw.text((5, 4), text, fill=(255, 255, 255))
    return out


def save_montage(root: Path, split: str, record: dict, output_path: Path, tile_size: tuple[int, int]):
    tiles = []
    mask_tiles = []
    for view in record["views"]:
        frame = int(view["frame"])
        seq = view["seq"]
        rgb_path = root / split / seq / "rgb" / f"{frame:08d}.png"
        mask_path = root / split / seq / "mask" / f"{frame:08d}.png"
        rgb = load_rgb(rgb_path, tile_size)
        label = f"v{view['view_idx']} {seq} f{frame} shot={view['shot_label']}"
        tiles.append(draw_label(rgb, label))
        mask_tiles.append(draw_label(load_mask_overlay(rgb, mask_path), "mask overlay"))

    w, h = tile_size
    montage = Image.new("RGB", (w * 4, h * 2), (20, 20, 20))
    for idx, tile in enumerate(tiles):
        montage.paste(tile, (idx * w, 0))
    for idx, tile in enumerate(mask_tiles):
        montage.paste(tile, (idx * w, h))
    montage.save(output_path)


def required_files(root: Path, split: str, seq: str, frame: int) -> dict:
    base = root / split / seq
    frame_name = f"{int(frame):08d}"
    return {
        "rgb": (base / "rgb" / f"{frame_name}.png").is_file(),
        "mask": (base / "mask" / f"{frame_name}.png").is_file(),
        "depth": (base / "depth" / f"{frame_name}.npy").is_file(),
        "smpl": (base / "smpl" / f"{frame_name}.pkl").is_file(),
        "cam": (base / "cam" / f"{frame_name}.npz").is_file(),
    }


def build_record(dataset: AvatarReX_AABB, idx: int, root: Path, split: str) -> dict:
    meta = dataset.get_sample_metadata(idx)
    frames = meta["frames"]
    view_specs = [
        (0, meta["seqA"], frames[0], 0),
        (1, meta["seqA"], frames[1], 0),
        (2, meta["seqB"], frames[2], 1),
        (3, meta["seqB"], frames[3], 0),
    ]
    split_path = osp.join(str(root), split)
    views = []
    for view_idx, seq, frame, shot_label in view_specs:
        pose = _avatarrex_load_camera_pose(split_path, seq, frame)
        views.append(
            {
                "view_idx": view_idx,
                "seq": seq,
                "frame": int(frame),
                "shot_label": int(shot_label),
                "files": required_files(root, split, seq, frame),
                "camera_c2w": pose.tolist(),
            }
        )
    return {
        "seqA": meta["seqA"],
        "seqB": meta["seqB"],
        "start_frame": int(meta["start_frame"]),
        "frames": [int(x) for x in frames],
        "view_angle_deg": float(meta["view_angle_deg"]),
        "views": views,
    }


def main():
    args = parse_args()
    width, height = [int(x) for x in args.resolution.split(",")]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = AvatarReX_AABB(
        split=args.split,
        ROOT=str(args.root),
        num_views=4,
        resolution=(width, height),
        seed=args.seed,
        min_view_angle_deg=args.min_view_angle_deg,
        max_view_angle_deg=args.max_view_angle_deg,
        pair_strategy=args.pair_strategy,
        max_samples=args.num_samples,
    )

    records = []
    for idx in range(min(args.num_samples, len(dataset))):
        record = build_record(dataset, idx, args.root, args.split)
        records.append(record)
        save_montage(
            args.root,
            args.split,
            record,
            args.output_dir / f"sample_{idx:03d}_angle_{record['view_angle_deg']:.1f}.jpg",
            (width, height),
        )

    manifest_path = args.output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    fixed_samples_path = args.output_dir / "fixed_samples.json"
    fixed_records = [
        {
            "seqA": r["seqA"],
            "seqB": r["seqB"],
            "start_frame": r["start_frame"],
            "frames": r["frames"],
            "view_angle_deg": r["view_angle_deg"],
        }
        for r in records
    ]
    with open(fixed_samples_path, "w", encoding="utf-8") as f:
        json.dump(fixed_records, f, indent=2)

    print(f"Selected {len(records)} samples from {len(dataset)} filtered samples")
    print(f"Manifest: {manifest_path}")
    for idx, record in enumerate(records):
        print(
            f"[{idx:03d}] {record['seqA']} -> {record['seqB']} "
            f"start={record['start_frame']} angle={record['view_angle_deg']:.2f}"
        )


if __name__ == "__main__":
    main()
