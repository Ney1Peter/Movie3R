#!/usr/bin/env python3
"""Export an AvatarReX AABB boundary sample as a shot-change mp4."""

import argparse
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/data/wangzheng/iJCV-CODE/data/RICH_4Human3R")
    parser.add_argument("--split", default="Training")
    parser.add_argument("--source_sequence", default="BBQ_001_guitar")
    parser.add_argument("--cam_a", type=int, default=1)
    parser.add_argument("--cam_b", type=int, default=2)
    parser.add_argument("--start_frame", type=int, default=35)
    parser.add_argument("--before_frames", type=int, default=30)
    parser.add_argument("--after_frames", type=int, default=30)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument(
        "--output",
        default="examples/guitar_AABB_overfit_cam01_cam02_start00000035_60frames_20fps.mp4",
    )
    return parser.parse_args()


def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read frame: {path}")
    return image


def main():
    args = parse_args()
    split_root = Path(args.root) / args.split
    seq_a = f"{args.source_sequence}_cam_{args.cam_a:02d}"
    seq_b = f"{args.source_sequence}_cam_{args.cam_b:02d}"

    ref_frame = args.start_frame + 1
    cur_frame = args.start_frame + 2
    first_ref_frame = ref_frame - args.before_frames + 1
    if first_ref_frame < 0:
        raise ValueError("before_frames reaches before frame 0")

    frame_specs = [
        (seq_a, frame_idx) for frame_idx in range(first_ref_frame, ref_frame + 1)
    ] + [
        (seq_b, frame_idx) for frame_idx in range(cur_frame, cur_frame + args.after_frames)
    ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    first_image = read_image(split_root / frame_specs[0][0] / "rgb" / f"{frame_specs[0][1]:08d}.png")
    height, width = first_image.shape[:2]
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output}")

    try:
        for seq_name, frame_idx in frame_specs:
            image = read_image(split_root / seq_name / "rgb" / f"{frame_idx:08d}.png")
            if image.shape[:2] != (height, width):
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(image)
    finally:
        writer.release()

    print(f"wrote: {output}")
    print(f"boundary video frames: {args.before_frames - 1} -> {args.before_frames}")
    print(f"AABB training frames: {seq_a} {args.start_frame},{ref_frame} -> {seq_b} {cur_frame},{args.start_frame + 3}")


if __name__ == "__main__":
    main()
