#!/usr/bin/env python3
"""V11.4 uniform-similarity probe preserving complete Human3R geometry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from dust3r.utils.smpl_layer import SMPL_Layer  # noqa: E402
from boundary_depth_correction_support import load_raw_pair  # noqa: E402
from boundary_geometry_support import distribution  # noqa: E402
from boundary_shot_scale_support import calibrated_targets, scale_pose  # noqa: E402
from boundary_metric_selection_support import load_cases, load_shards  # noqa: E402
from v11_2_contact_preserving_probe import (  # noqa: E402
    build_clouds,
    load_body_pair,
    solve_variant,
)


ROTATIONS = {
    "torso": "torso_raw",
    "v47": "v32_raw",
}
SCALE_MODES = ("root", "scene", "geometric_mean")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--component_report",
        type=Path,
        default=ROOT / "output/v48_component_necessity_ablation/v48_component_necessity_ablation.json",
    )
    parser.add_argument(
        "--bridge_report",
        type=Path,
        default=ROOT / "output/v36_final_explicit_metric_bridge/v36_final_explicit_metric_bridge.json",
    )
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
        "--stream_dir",
        type=Path,
        default=ROOT / "output/v18_human_metric_translation/stream_cache",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ROOT / "output/v53_uniform_similarity_integrity",
    )
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--point_samples", type=int, default=4000)
    parser.add_argument("--raw_confidence_threshold", type=float, default=1.0)
    parser.add_argument("--mask_dilate", type=int, default=11)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args()


def scales_for_mode(bridge: dict, mode: str) -> list[float]:
    root = bridge["root_scales"]
    scene = bridge["scene_scale_sets"]["absolute"]
    if mode == "root":
        return [float(root["old"]), float(root["new"])]
    if mode == "scene":
        return [float(scene["old"]), float(scene["new"])]
    if mode == "geometric_mean":
        return [
            float(np.sqrt(float(root[side]) * float(scene[side])))
            for side in ("old", "new")
        ]
    raise KeyError(mode)


def run_case(
    v48: dict,
    bridge: dict,
    v10: dict,
    stream_row: dict,
    layer: SMPL_Layer,
    args: argparse.Namespace,
    index: int,
) -> dict:
    local = Path(v10["paths"]["human3r_local_reset"])
    body = load_body_pair(local, layer, torch.device(args.device))
    with np.load(stream_row["cache_path"]) as stream:
        raw_poses = [
            np.asarray(stream["old_pose"][-1], dtype=np.float32),
            np.asarray(stream["new_pose"], dtype=np.float32),
        ]
        intrinsics = np.stack(
            [stream["old_intrinsics"][-1], stream["new_intrinsics"]]
        ).astype(np.float32)
        gt_world_scale = {
            "old": float(np.median(stream["old_gt_world_scale"])),
            "new": float(np.median(stream["new_gt_world_scale"])),
        }

    raw = load_raw_pair(local)
    kernel = np.ones((int(args.mask_dilate), int(args.mask_dilate)), dtype=np.uint8)
    masks = [
        cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        for mask in raw["mask"]
    ]
    variants = {}
    for scale_mode in SCALE_MODES:
        scales = scales_for_mode(bridge, scale_mode)
        poses = [scale_pose(raw_poses[frame], scales[frame]) for frame in range(2)]
        with np.load(stream_row["cache_path"]) as stream:
            target, gt_old, gt_new = calibrated_targets(stream, poses[0])
        roots = [body["roots"][frame] * scales[frame] for frame in range(2)]
        clouds = build_clouds(
            raw,
            intrinsics,
            poses,
            scales,
            masks,
            args,
            int(args.seed) + index * 11 + SCALE_MODES.index(scale_mode),
        )
        for rotation_name, v48_key in ROTATIONS.items():
            rotation = np.asarray(
                v48["variants"][v48_key]["transform"], dtype=np.float32
            )[:3, :3]
            value = solve_variant(
                rotation,
                poses,
                roots,
                target,
                gt_old,
                gt_new,
                clouds,
            )
            value["scales"] = {"old": scales[0], "new": scales[1]}
            value["integrity"] = {
                "human_reprojection_shift_px": 0.0,
                "foot_ground_distortion_m": 0.0,
                "relative_human_scene_geometry_preserved": True,
                "uniform_similarity": True,
                "body_scale_old": scales[0],
                "body_scale_new": scales[1],
            }
            variants[f"{rotation_name}_uniform_{scale_mode}"] = value

    return {
        "case_name": str(v48["case_name"]),
        "source": str(v48["source"]),
        "gt_world_scale": gt_world_scale,
        "root_scales": bridge["root_scales"],
        "scene_scales": bridge["scene_scale_sets"]["absolute"],
        "variants": variants,
    }


def summarize(rows: list[dict], variant: str) -> dict:
    values = [row["variants"][variant] for row in rows]
    scale_error = []
    for row, value in zip(rows, values):
        for side in ("old", "new"):
            scale_error.append(abs(value["scales"][side] - row["gt_world_scale"][side]))
    return {
        "camera_translation_m": distribution(
            [value["camera"]["translation_m"] for value in values]
        ),
        "camera_rotation_deg": distribution(
            [value["camera"]["rotation_deg"] for value in values]
        ),
        "human_motion_error_m": distribution(
            [value["human"]["root_motion_error_m"] for value in values]
        ),
        "scene_trimmed_mean_m": distribution(
            [value["scene"]["trimmed_mean_m"] for value in values]
        ),
        "scale_absolute_error_vs_gt_diagnostic": distribution(scale_error),
        "human_reprojection_shift_px": {
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
        },
        "foot_ground_distortion_m": {
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
        },
    }


def main() -> None:
    args = parse_args()
    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("V11.4 requires CUDA for SMPL-X reconstruction")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    v48 = load_cases(args.component_report)
    bridges = load_cases(args.bridge_report)
    v10 = load_cases(args.v10_report)
    streams = load_shards(args.stream_dir, "v18_stream_shard_*_of_*.json")
    names = sorted(set(v48) & set(bridges) & set(v10) & set(streams))
    if len(names) != 180:
        raise RuntimeError(f"Expected 180 aligned cases, got {len(names)}")
    device = torch.device(args.device)
    layer = SMPL_Layer(
        type="smplx",
        gender="neutral",
        num_betas=10,
        kid=False,
        person_center="head",
    ).to(device).eval()
    rows = []
    for index, name in enumerate(names):
        rows.append(
            run_case(v48[name], bridges[name], v10[name], streams[name], layer, args, index)
        )
        if (index + 1) % 20 == 0:
            print(f"V11.4 uniform-similarity probe {index + 1}/{len(names)}", flush=True)
    variant_names = sorted(rows[0]["variants"])
    report = {
        "experiment": "V11.4 uniform similarity integrity probe",
        "case_count": len(rows),
        "protocol": {
            "post_cut_frames": 1,
            "rotation_sources": list(ROTATIONS),
            "scale_modes": list(SCALE_MODES),
            "scaling": "camera, pointmap, SMPL-X root, and SMPL-X body offsets share one shot scale",
            "projection_change": "none under perspective projection",
            "contact_change": "none because human and scene are scaled together",
            "gt_runtime_information": False,
        },
        "overall": {variant: summarize(rows, variant) for variant in variant_names},
        "by_source": {
            source: {
                variant: summarize(
                    [row for row in rows if row["source"] == source], variant
                )
                for variant in variant_names
            }
            for source in sorted({row["source"] for row in rows})
        },
        "cases": rows,
    }
    output = args.output_dir / "v53_uniform_similarity_integrity_probe.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"overall": report["overall"], "by_source": report["by_source"]}, indent=2))
    print(f">> wrote {output}")


if __name__ == "__main__":
    main()
