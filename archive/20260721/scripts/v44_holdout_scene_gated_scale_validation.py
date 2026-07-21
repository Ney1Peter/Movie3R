#!/usr/bin/env python3
"""Frozen V43 background-scale validation on an independent V10 holdout cache."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from v11_gauge_neutral_first_write_probe import build_dataset, read_jsonl, record_spec
from v12_build_gauge_neutral_teacher_cache import old_a_dataset
from dust3r.utils.smpl_layer import SMPL_Layer
from v18_da3_metric_depth_probe import (
    DepthAnything3,
    boundary_from_camera_pose,
    camera_pose_from_human,
    evaluate,
    metric_inference,
    sample_depth,
)
from v19_da3_depth_correction_ablation import load_raw_pair
from v19_da3_explicit_geometry_correction_probe import (
    distribution,
    sample_cloud,
    scene_alignment_metrics,
    transform_points,
)
from v20_shot_scale_consistency_probe import scale_pose
from v21_absolute_shot_background_scale_probe import bounded_scene_scale, frame_calibration
from v9_learned_stream_alignment_overfit import gt_pose_from_view


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "output/v37_human_jump_holdout7"
DEFAULT_MODEL = (
    REPO_ROOT.parent
    / "Movie3R-dataset/Depth-Anything-3/checkpoints/DA3Metric-Large"
)
SCENE_GAIN_THRESHOLD_M = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_ROOT / "records/holdout_records_v10_valid.jsonl")
    parser.add_argument("--v10_report", type=Path, default=DEFAULT_ROOT / "v10_merged/merged_cases.json")
    parser.add_argument(
        "--rotation_report",
        type=Path,
        default=DEFAULT_ROOT / "frozen_rule_validation_valid/v36_frozen_human_jump_consensus_validation.json",
    )
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_ROOT / "v44_scale_validation")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data"))
    parser.add_argument("--resolution", type=int, nargs=2, default=(512, 288))
    parser.add_argument("--resize_mode", default="human3r_demo")
    parser.add_argument("--warmup_frames", type=int, default=8)
    parser.add_argument("--max_post_frames", type=int, default=3)
    parser.add_argument("--process_res", type=int, default=504)
    parser.add_argument("--point_samples", type=int, default=6000)
    parser.add_argument("--raw_confidence_threshold", type=float, default=1.0)
    parser.add_argument("--mask_dilate", type=int, default=11)
    parser.add_argument("--lowfreq_sigma", type=float, default=25.0)
    parser.add_argument("--min_background_pixels", type=int, default=512)
    parser.add_argument("--background_correction_bound", type=float, default=0.15)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--merge", action="store_true")
    return parser.parse_args()


def one_batch(dataset) -> list[dict]:
    return next(iter(torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)))


def load_v10(path: Path) -> dict[str, dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))["cases"]
    return {row["case_name"]: row for row in rows}


def load_rotations(path: Path) -> dict[str, dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))["cases"]
    return {row["case_name"]: row for row in rows}


def load_frame(
    local: Path,
    index: int,
    layer: SMPL_Layer,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(local / "camera" / f"{index:06d}.npz") as camera:
        pose = np.asarray(camera["pose"], dtype=np.float32)
        intrinsics = np.asarray(camera["intrinsics"], dtype=np.float32)
    with np.load(local / "smpl" / f"{index:06d}.npz") as smpl:
        rotvec = torch.from_numpy(np.asarray(smpl["rotvec"], dtype=np.float32)).to(device)
        shape = torch.from_numpy(np.asarray(smpl["shape"], dtype=np.float32)).to(device)
        transl = torch.from_numpy(np.asarray(smpl["transl"], dtype=np.float32)).to(device)
        expression = torch.from_numpy(np.asarray(smpl["expression"], dtype=np.float32)).to(device)
    with torch.no_grad():
        human = layer(
            rotvec,
            shape,
            transl,
            None,
            None,
            K=torch.from_numpy(intrinsics[None]).to(device),
            expression=expression,
        )
    root = human["smpl_j3d"][0, 0].detach().float().cpu().numpy().astype(np.float32)
    return pose, intrinsics, root


def root_scale(root: np.ndarray, metric_depth: np.ndarray, intrinsics: np.ndarray) -> float:
    z = max(float(root[2]), 1e-4)
    pixel = np.asarray([
        float(intrinsics[0, 0] * root[0] / z + intrinsics[0, 2]),
        float(intrinsics[1, 1] * root[1] / z + intrinsics[1, 2]),
    ], dtype=np.float32)
    depth = sample_depth(metric_depth, pixel, radius=4)
    if not np.isfinite(depth):
        return 1.0
    return float(np.clip(depth / z, 0.35, 3.0))


def view_image(view: dict) -> np.ndarray:
    image = ((view["img"][0].detach().float().cpu() + 1.0) * 127.5).clamp(0.0, 255.0)
    return image.byte().permute(1, 2, 0).numpy()


def gt_camera_pair(record: dict, args: argparse.Namespace) -> dict:
    spec = record_spec(record, args)
    new_views = one_batch(build_dataset([spec], False, args))[:1]
    old_views = one_batch(old_a_dataset(spec, args))
    old_gt = gt_pose_from_view(old_views[-1]).detach().float().cpu().numpy().astype(np.float32)
    new_gt = gt_pose_from_view(new_views[0]).detach().float().cpu().numpy().astype(np.float32)
    return {
        "old_gt": old_gt,
        "new_gt": new_gt,
        "images": [view_image(old_views[-1]), view_image(new_views[0])],
        "intrinsics": np.stack([
            old_views[-1]["camera_intrinsics"][0].detach().float().cpu().numpy(),
            new_views[0]["camera_intrinsics"][0].detach().float().cpu().numpy(),
        ]).astype(np.float32),
    }


def variant(
    raw_poses: list[np.ndarray],
    roots: list[np.ndarray],
    scales: tuple[float, float],
    rotation: np.ndarray,
    old_gt: np.ndarray,
    new_gt: np.ndarray,
    raw: dict,
    intrinsics: list[np.ndarray],
    scene_scales: tuple[float, float],
    masks: list[np.ndarray],
    args: argparse.Namespace,
    seed: int,
) -> dict:
    poses = [scale_pose(raw_poses[index], scales[index]) for index in range(2)]
    scaled_roots = [roots[index] * scales[index] for index in range(2)]
    old_anchor = transform_points(poses[0], scaled_roots[0][None])[0]
    camera_rotation = rotation @ poses[1][:3, :3]
    camera_pose = camera_pose_from_human(camera_rotation, old_anchor, scaled_roots[1])
    transform = boundary_from_camera_pose(camera_pose, poses[1])
    target_pose = poses[0] @ np.linalg.inv(old_gt) @ new_gt
    rng = np.random.default_rng(seed)
    clouds = [
        sample_cloud(
            raw["depth"][index] * scene_scales[index],
            intrinsics[index],
            poses[index],
            masks[index],
            raw["confidence"][index],
            float(args.raw_confidence_threshold),
            int(args.point_samples),
            rng,
        )
        for index in range(2)
    ]
    return {
        "scales": {"old": scales[0], "new": scales[1]},
        "transform": transform.astype(float).tolist(),
        "camera": evaluate(transform, poses[1], target_pose),
        "scene": scene_alignment_metrics(transform, clouds[1], clouds[0]),
    }


def run_case(
    record: dict,
    v10: dict,
    rotation_row: dict,
    model: DepthAnything3,
    layer: SMPL_Layer,
    args: argparse.Namespace,
    index: int,
) -> dict:
    local = Path(v10["paths"]["human3r_local_reset"])
    device = torch.device(args.device)
    frames = [load_frame(local, frame, layer, device) for frame in (1, 2)]
    raw_poses = [frame[0] for frame in frames]
    raw_intrinsics = np.stack([frame[1] for frame in frames]).astype(np.float32)
    roots = [frame[2] for frame in frames]
    views = gt_camera_pair(record, args)
    images = views["images"]
    input_intrinsics = views["intrinsics"]
    metric_depths = []
    metric_intrinsics = []
    inference_seconds = 0.0
    for frame in range(2):
        depth, processed_intrinsics, elapsed = metric_inference(
            model,
            [images[frame]],
            input_intrinsics[frame : frame + 1],
            int(args.process_res),
        )
        metric_depths.append(depth[0])
        metric_intrinsics.append(processed_intrinsics[0])
        inference_seconds += float(elapsed)

    human_scales = tuple(
        root_scale(roots[frame], metric_depths[frame], metric_intrinsics[frame])
        for frame in range(2)
    )
    raw = load_raw_pair(local)
    kernel = np.ones((int(args.mask_dilate), int(args.mask_dilate)), dtype=np.uint8)
    masks = [cv2.dilate(value.astype(np.uint8), kernel, iterations=1) for value in raw["mask"]]
    calibrations = [
        frame_calibration(
            raw["depth"][frame],
            metric_depths[frame],
            raw["confidence"][frame],
            masks[frame],
            human_scales[frame],
            args,
        )
        for frame in range(2)
    ]
    background_scales = []
    for frame in range(2):
        scene_scale = float(calibrations[frame]["scales"]["median_ratio"])
        bounded = bounded_scene_scale(
            scene_scale,
            human_scales[frame],
            float(args.background_correction_bound),
        )
        ratio = scene_scale / max(human_scales[frame], 1e-6)
        background_scales.append(bounded if ratio < 0.95 else human_scales[frame])
    background_scales = tuple(background_scales)

    old_gt, new_gt = views["old_gt"], views["new_gt"]
    rotation_value = rotation_row.get("v32_transform_rotation")
    if rotation_value is None:
        rotation_value = rotation_row.get("v36_transform_rotation")
    if rotation_value is None:
        rotation_value = np.asarray(
            rotation_row["variants"]["v36"]["transform"], dtype=np.float32
        )[:3, :3]
    rotation = np.asarray(rotation_value, dtype=np.float32)
    seed = int(args.seed) + 1009 * index
    baseline = variant(
        raw_poses,
        roots,
        human_scales,
        rotation,
        old_gt,
        new_gt,
        raw,
        [frame[1] for frame in frames],
        background_scales,
        masks,
        args,
        seed,
    )
    candidate_scales = (human_scales[0], background_scales[1])
    candidate = variant(
        raw_poses,
        roots,
        candidate_scales,
        rotation,
        old_gt,
        new_gt,
        raw,
        [frame[1] for frame in frames],
        background_scales,
        masks,
        args,
        seed,
    )
    scene_delta = float(candidate["scene"]["trimmed_mean_m"] - baseline["scene"]["trimmed_mean_m"])
    camera_delta = float(candidate["camera"]["translation_m"] - baseline["camera"]["translation_m"])
    accepted = bool(scene_delta < -SCENE_GAIN_THRESHOLD_M)
    return {
        "case_name": record["pattern_id"],
        "source": record["source"],
        "accepted": accepted,
        "scene_delta_m": scene_delta,
        "camera_delta_m": camera_delta,
        "human_scales": {"old": human_scales[0], "new": human_scales[1]},
        "background_scales": {"old": background_scales[0], "new": background_scales[1]},
        "background_over_human": {
            "old": background_scales[0] / max(human_scales[0], 1e-6),
            "new": background_scales[1] / max(human_scales[1], 1e-6),
        },
        "calibration_status": {
            "old": calibrations[0]["status"],
            "new": calibrations[1]["status"],
            "old_pixels": calibrations[0]["valid_background_pixels"],
            "new_pixels": calibrations[1]["valid_background_pixels"],
        },
        "variants": {"v36": baseline, "v44": candidate if accepted else baseline},
        "raw_candidate": candidate,
        "da3_inference_seconds": inference_seconds,
    }


def summarize(rows: list[dict], variant: str) -> dict:
    values = [row["variants"][variant] for row in rows]
    baseline = [row["variants"]["v36"] for row in rows]
    translation = np.asarray([row["camera"]["translation_m"] for row in values])
    base_translation = np.asarray([row["camera"]["translation_m"] for row in baseline])
    rotation = np.asarray([row["camera"]["rotation_deg"] for row in values])
    scene = np.asarray([row["scene"]["trimmed_mean_m"] for row in values])
    base_scene = np.asarray([row["scene"]["trimmed_mean_m"] for row in baseline])
    return {
        "camera_translation_m": distribution(translation.tolist()),
        "camera_rotation_deg": distribution(rotation.tolist()),
        "scene_trimmed_mean_m": distribution(scene.tolist()),
        "translation_improved_005m": int(np.sum(translation + 0.05 < base_translation)),
        "translation_harmful_005m": int(np.sum(translation > base_translation + 0.05)),
        "translation_improved_010m": int(np.sum(translation + 0.10 < base_translation)),
        "translation_harmful_010m": int(np.sum(translation > base_translation + 0.10)),
        "scene_harmful_005m": int(np.sum(scene > base_scene + 0.05)),
    }


def merge(args: argparse.Namespace) -> None:
    rows = []
    for path in sorted(glob.glob(str(args.output_dir / "v44_shard_*_of_*.json"))):
        rows.extend(json.loads(Path(path).read_text(encoding="utf-8"))["cases"])
    names = {row["case_name"] for row in rows}
    if not rows or len(names) != len(rows):
        raise RuntimeError(f"Invalid V44 shards: {len(rows)}/{len(names)}")
    report = {
        "experiment": "V44 frozen scene-gated background-scale holdout validation",
        "case_count": len(rows),
        "protocol": {
            "scene_gain_threshold_m": SCENE_GAIN_THRESHOLD_M,
            "background_correction_bound": float(args.background_correction_bound),
            "threshold_frozen_before_holdout": True,
            "gt_runtime_information": False,
            "post_cut_frames": 1,
        },
        "accepted_count": int(sum(row["accepted"] for row in rows)),
        "overall": {variant: summarize(rows, variant) for variant in ("v36", "v44")},
        "by_source": {
            source: {variant: summarize([row for row in rows if row["source"] == source], variant) for variant in ("v36", "v44")}
            for source in sorted({row["source"] for row in rows})
        },
        "accepted_cases": [row for row in rows if row["accepted"]],
        "cases": rows,
    }
    output = args.output_dir / "v44_holdout_scene_gated_scale_validation.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "accepted_count": report["accepted_count"],
        "overall": report["overall"],
        "by_source": report["by_source"],
        "accepted_cases": [
            {
                "case_name": row["case_name"],
                "source": row["source"],
                "camera_delta_m": row["camera_delta_m"],
                "scene_delta_m": row["scene_delta_m"],
            }
            for row in report["accepted_cases"]
        ],
    }, indent=2))
    print(f">> wrote {output}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.merge:
        merge(args)
        return
    if not args.rotation_report.exists():
        raise FileNotFoundError(args.rotation_report)
    records = read_jsonl(args.records)
    selected = [record for index, record in enumerate(records) if index % args.num_shards == args.shard_index]
    if args.max_cases > 0:
        selected = selected[: args.max_cases]
    v10 = load_v10(args.v10_report)
    rotations = load_rotations(args.rotation_report)
    model = DepthAnything3.from_pretrained(str(args.model_path)).to(args.device).eval()
    layer = SMPL_Layer(
        type="smplx",
        gender="neutral",
        num_betas=10,
        kid=False,
        person_center="head",
    ).to(args.device).eval()
    rows = []
    for index, record in enumerate(selected):
        name = record["pattern_id"]
        rows.append(run_case(record, v10[name], rotations[name], model, layer, args, index))
        print(f">> [{index + 1}/{len(selected)}] {name} accepted={rows[-1]['accepted']}", flush=True)
    output = args.output_dir / f"v44_shard_{args.shard_index:02d}_of_{args.num_shards:02d}.json"
    output.write_text(json.dumps({
        "experiment": "V44 holdout scene-gated scale shard",
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "case_count": len(rows),
        "cases": rows,
    }, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f">> wrote {output}")


if __name__ == "__main__":
    main()
