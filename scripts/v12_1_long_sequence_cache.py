#!/usr/bin/env python3
"""V12.1 cache 10+10 Human3R frames for retained-method visualization."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from dust3r.datasets.avatarrex import AvatarReX_Pattern  # noqa: E402
from v10_token_alignment_4source_probe import (  # noqa: E402
    raw_roots_for_record,
    source_split_and_scope,
)
from boundary_human3r_reset_support import (  # noqa: E402
    build_model,
    configure_views,
)
from v9_learned_stream_alignment_4source_probe import run_local_reset_human3r  # noqa: E402


DEFAULT_CASES = (
    "avatarrex_120_150_lbn2_1651_22010714_22070932",
    "thuman_060_090_thuman02_2770_cam12_cam07",
    "mvhuman100_090_120_100003_338_CC32871A035_CC32871A008",
    "mvhuman200_120_150_200002_410_22327109_22236235",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v10_report",
        type=Path,
        default=(
            ROOT
            / "output/v10_candidate_selection/oracle_gt_4source"
            / "oracle_candidate_selection_metrics.json"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ROOT / "output/v52_long_sequence_visualization/cache",
    )
    parser.add_argument("--data_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data"))
    parser.add_argument("--model_path", type=Path, default=ROOT / "src/human3r_896L.pth")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--resolution", type=int, nargs=2, default=(512, 288))
    parser.add_argument("--resize_mode", default="human3r_demo")
    parser.add_argument("--frames_per_shot", type=int, default=10)
    parser.add_argument("--cases", nargs="*", default=DEFAULT_CASES)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def report_cases(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["case_name"]): row for row in payload["cases"]}


def available_frames(args: argparse.Namespace, record: dict, seq: str) -> list[int]:
    split, _ = source_split_and_scope(record)
    rgb_dir = args.data_root / split / seq / "rgb"
    frames = sorted(int(path.stem) for path in rgb_dir.glob("*.png"))
    if not frames:
        raise FileNotFoundError(rgb_dir)
    return frames


def shot_frames(args: argparse.Namespace, record: dict) -> tuple[list[int], list[int]]:
    boundary = int(record["start_frame"]) + 2
    pre_available = available_frames(args, record, str(record["seqA"]))
    post_available = available_frames(args, record, str(record["seqB"]))
    pre = [frame for frame in pre_available if frame < boundary][-int(args.frames_per_shot) :]
    post = [frame for frame in post_available if frame >= boundary][: int(args.frames_per_shot)]
    if len(pre) != int(args.frames_per_shot) or len(post) != int(args.frames_per_shot):
        raise RuntimeError(
            f"Insufficient frames for {record['pattern_id']}: pre={len(pre)} post={len(post)}"
        )
    return pre, post


def build_shot_dataset(
    args: argparse.Namespace,
    record: dict,
    seq: str,
    frames: list[int],
    side: str,
) -> AvatarReX_Pattern:
    split, _ = source_split_and_scope(record)
    sample = {
        "clip_type": f"v12_long_{side}",
        "group": str(record.get("group", "")),
        "seqs": [seq] * len(frames),
        "frames": frames,
        "shot_labels": [0] * len(frames),
        "transition_angles_deg": [0.0] * len(frames),
        "view_angle_deg": float(record.get("view_angle_deg", 0.0)),
        "angle_bucket": str(record.get("angle_bucket", "unknown")),
        "pattern_id": f"{record['pattern_id']}_v12_{side}",
    }
    return AvatarReX_Pattern(
        split=split,
        ROOT=str(args.data_root),
        resolution=tuple(args.resolution),
        num_views=len(frames),
        aug_crop=0,
        allow_repeat=True,
        seed=401,
        n_corres=0,
        fixed_samples=[sample],
        load_da3_depth=False,
        raw_calibration_root=raw_roots_for_record(record),
        resize_mode=str(args.resize_mode),
        max_humans=1,
    )


def one_batch(dataset: AvatarReX_Pattern) -> list[dict]:
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return next(iter(loader))


def shot_complete(path: Path, expected: int) -> bool:
    required = (("camera", ".npz"), ("depth", ".npy"), ("color", ".png"), ("smpl", ".npz"))
    return all(len(list((path / folder).glob(f"*{suffix}"))) == expected for folder, suffix in required)


def load_pose(path: Path, index: int) -> np.ndarray:
    with np.load(path / "camera" / f"{index:06d}.npz") as data:
        return data["pose"].astype(np.float32)


def rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    relative = a[:3, :3] @ b[:3, :3].T
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def pose_error(a: np.ndarray, b: np.ndarray) -> dict:
    return {
        "translation_m": float(np.linalg.norm(a[:3, 3] - b[:3, 3])),
        "rotation_deg": rotation_error_deg(a, b),
    }


def image_mae(a: Path, b: Path) -> float:
    image_a = np.asarray(Image.open(a).convert("RGB"), dtype=np.float32)
    image_b = np.asarray(Image.open(b).convert("RGB"), dtype=np.float32)
    if image_a.shape != image_b.shape:
        return float("inf")
    return float(np.mean(np.abs(image_a - image_b)))


def depth_mae(a: Path, b: Path) -> float:
    depth_a = np.load(a).astype(np.float32)
    depth_b = np.load(b).astype(np.float32)
    valid = np.isfinite(depth_a) & np.isfinite(depth_b) & (depth_a > 0.05) & (depth_b > 0.05)
    return float(np.mean(np.abs(depth_a[valid] - depth_b[valid]))) if bool(valid.any()) else float("nan")


def gauge_manifest(
    old_dir: Path,
    pre_dir: Path,
    post_dir: Path,
    count: int,
) -> dict:
    old_poses = [load_pose(old_dir, index) for index in range(4)]
    pre_poses = [load_pose(pre_dir, index) for index in range(count)]
    post_poses = [load_pose(post_dir, index) for index in range(count)]
    pre_gauge = old_poses[1] @ np.linalg.inv(pre_poses[-1])
    post_gauge = old_poses[2] @ np.linalg.inv(post_poses[0])
    aligned_pre_previous = pre_gauge @ pre_poses[-2]
    aligned_post_next = post_gauge @ post_poses[1]
    color_pairs = (
        (pre_dir / "color" / f"{count - 2:06d}.png", old_dir / "color/000000.png"),
        (pre_dir / "color" / f"{count - 1:06d}.png", old_dir / "color/000001.png"),
        (post_dir / "color/000000.png", old_dir / "color/000002.png"),
        (post_dir / "color/000001.png", old_dir / "color/000003.png"),
    )
    image_errors = [image_mae(a, b) for a, b in color_pairs]
    if max(image_errors) > 0.5:
        raise RuntimeError(f"Long-sequence overlap frames do not match the V10 cache: {image_errors}")
    return {
        "pre_to_v10_gauge": pre_gauge.astype(np.float32).tolist(),
        "post_to_v10_gauge": post_gauge.astype(np.float32).tolist(),
        "overlap_checks": {
            "image_mae_0_1_2_3": image_errors,
            "pre_previous_pose": pose_error(aligned_pre_previous, old_poses[0]),
            "post_next_pose": pose_error(aligned_post_next, old_poses[3]),
            "pre_last_depth_mae_m": depth_mae(
                pre_dir / "depth" / f"{count - 1:06d}.npy",
                old_dir / "depth/000001.npy",
            ),
            "post_first_depth_mae_m": depth_mae(
                post_dir / "depth/000000.npy",
                old_dir / "depth/000002.npy",
            ),
        },
    }


def run_case(
    args: argparse.Namespace,
    model,
    device: torch.device,
    case: dict,
) -> dict:
    record = case["record"]
    case_dir = args.output_dir / str(case["case_name"])
    pre_dir = case_dir / "pre_human3r"
    post_dir = case_dir / "post_human3r"
    pre_frames, post_frames = shot_frames(args, record)
    count = int(args.frames_per_shot)

    for side, seq, frames, output in (
        ("pre", str(record["seqA"]), pre_frames, pre_dir),
        ("post", str(record["seqB"]), post_frames, post_dir),
    ):
        if not args.overwrite and shot_complete(output, count):
            continue
        if output.exists() and not shot_complete(output, count):
            shutil.rmtree(output)
        dataset = build_shot_dataset(args, record, seq, frames, side)
        views = configure_views(one_batch(dataset), device, model.mhmr_img_res)
        run_local_reset_human3r(model, views, output, args, device)
        if not shot_complete(output, count):
            raise RuntimeError(f"Incomplete {side} cache: {output}")

    gauge = gauge_manifest(
        Path(case["paths"]["human3r_local_reset"]),
        pre_dir,
        post_dir,
        count,
    )
    manifest = {
        "case_name": str(case["case_name"]),
        "record": record,
        "frames_per_shot": count,
        "pre_frames": pre_frames,
        "post_frames": post_frames,
        "pre_dir": str(pre_dir),
        "post_dir": str(post_dir),
        "v10_local_dir": str(case["paths"]["human3r_local_reset"]),
        **gauge,
    }
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    args = parse_args()
    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Long-sequence Human3R caching requires CUDA")
    cases = report_cases(args.v10_report)
    missing = sorted(set(args.cases) - set(cases))
    if missing:
        raise KeyError(f"Cases missing from V10 report: {missing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = build_model(args)
    manifests = []
    for index, name in enumerate(args.cases):
        print(f">> [{index + 1}/{len(args.cases)}] cache 10+10 {name}", flush=True)
        manifest = run_case(args, model, device, cases[name])
        manifests.append(manifest)
        checks = manifest["overlap_checks"]
        print(
            f">> overlap pre={checks['pre_previous_pose']} post={checks['post_next_pose']} ",
            f"depth={checks['pre_last_depth_mae_m']:.4f}/{checks['post_first_depth_mae_m']:.4f}",
            flush=True,
        )
    summary = {
        "case_count": len(manifests),
        "frames_per_case": 2 * int(args.frames_per_shot),
        "cases": [row["case_name"] for row in manifests],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
