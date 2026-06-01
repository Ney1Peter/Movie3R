#!/usr/bin/env python3
"""Export one grouped AvatarReX AABB clip as a plain RGB image sequence.

The output is intentionally just four images plus metadata. It is meant for
pure demo-style inference, where the model sees only RGB frames:

    A_t, A_{t+1}, B_{t+2}, B_{t+3}

No GT camera, SMPL, mask, or depth file is copied.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training_root",
        type=Path,
        default=Path("/data/wangzheng/iJCV-CODE/data/Avatarrex_output/Training"),
    )
    parser.add_argument("--group", required=True, choices=("lbn1", "zxc", "zzr"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--min_angle", type=float, default=60.0)
    parser.add_argument("--seq_a", default=None)
    parser.add_argument("--seq_b", default=None)
    parser.add_argument("--start_frame", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_pose(seq_dir: Path, frame_id: int) -> np.ndarray:
    return np.load(seq_dir / "cam" / f"{frame_id:08d}.npz")["pose"].astype(np.float64)


def camera_angle_deg(pose_a: np.ndarray, pose_b: np.ndarray) -> float:
    dir_a = pose_a[:3, 2]
    dir_b = pose_b[:3, 2]
    dir_a = dir_a / max(np.linalg.norm(dir_a), 1e-12)
    dir_b = dir_b / max(np.linalg.norm(dir_b), 1e-12)
    return math.degrees(math.acos(float(np.clip(np.dot(dir_a, dir_b), -1.0, 1.0))))


def frame_ids(seq_dir: Path) -> list[int]:
    return sorted(int(path.stem) for path in (seq_dir / "rgb").glob("*.png"))


def has_rgb(seq_dir: Path, frame_id: int) -> bool:
    return (seq_dir / "rgb" / f"{frame_id:08d}.png").is_file()


def valid_start_frames(seq_a_dir: Path, seq_b_dir: Path) -> list[int]:
    ids = frame_ids(seq_a_dir)
    valid = []
    for t in ids:
        if (
            has_rgb(seq_a_dir, t)
            and has_rgb(seq_a_dir, t + 1)
            and has_rgb(seq_b_dir, t + 2)
            and has_rgb(seq_b_dir, t + 3)
        ):
            valid.append(t)
    return valid


def choose_sample(group_dir: Path, seed: int, min_angle: float):
    rng = random.Random(seed)
    seqs = sorted(path.name for path in group_dir.iterdir() if path.is_dir())
    if len(seqs) < 2:
        raise ValueError(f"Need at least two sequence folders under {group_dir}")

    candidates = []
    probe_frame = frame_ids(group_dir / seqs[0])[0]
    for seq_a in seqs:
        for seq_b in seqs:
            if seq_a == seq_b:
                continue
            seq_a_dir = group_dir / seq_a
            seq_b_dir = group_dir / seq_b
            starts = valid_start_frames(seq_a_dir, seq_b_dir)
            if not starts:
                continue
            angle = camera_angle_deg(
                load_pose(seq_a_dir, probe_frame),
                load_pose(seq_b_dir, probe_frame),
            )
            if angle >= min_angle:
                candidates.append((seq_a, seq_b, angle, starts))

    if not candidates:
        raise ValueError(f"No AABB pair in {group_dir} satisfies min_angle={min_angle}")
    seq_a, seq_b, angle, starts = rng.choice(candidates)
    start_frame = rng.choice(starts)
    return seq_a, seq_b, start_frame, angle


def main() -> None:
    args = parse_args()
    group_dir = args.training_root / args.group
    if not group_dir.is_dir():
        raise FileNotFoundError(group_dir)

    if args.seq_a is None or args.seq_b is None or args.start_frame is None:
        seq_a, seq_b, start_frame, angle = choose_sample(group_dir, args.seed, args.min_angle)
    else:
        seq_a, seq_b, start_frame = args.seq_a, args.seq_b, int(args.start_frame)
        angle = camera_angle_deg(
            load_pose(group_dir / seq_a, start_frame),
            load_pose(group_dir / seq_b, start_frame),
        )
        if start_frame not in valid_start_frames(group_dir / seq_a, group_dir / seq_b):
            raise ValueError(f"Invalid AABB start_frame={start_frame} for {seq_a}->{seq_b}")

    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        (seq_a, start_frame),
        (seq_a, start_frame + 1),
        (seq_b, start_frame + 2),
        (seq_b, start_frame + 3),
    ]
    copied = []
    for out_idx, (seq, frame_id) in enumerate(specs):
        src = group_dir / seq / "rgb" / f"{frame_id:08d}.png"
        dst = args.output_dir / f"{out_idx:06d}.png"
        shutil.copy2(src, dst)
        copied.append(
            {
                "view_idx": out_idx,
                "seq": seq,
                "frame_id": int(frame_id),
                "source_rgb": str(src),
                "output_rgb": str(dst),
            }
        )

    metadata = {
        "group": args.group,
        "seq_a": seq_a,
        "seq_b": seq_b,
        "start_frame": int(start_frame),
        "frames": [int(start_frame + i) for i in range(4)],
        "view_angle_deg": float(angle),
        "layout": "A_t, A_t+1, B_t+2, B_t+3",
        "rgb_only": True,
        "copied": copied,
    }
    (args.output_dir / "sample.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
