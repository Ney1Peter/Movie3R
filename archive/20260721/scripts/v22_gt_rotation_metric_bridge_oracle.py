#!/usr/bin/env python3
"""Replace only V22 rotation with GT and keep the deployable metric translation equation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from v18_da3_metric_depth_probe import (  # noqa: E402
    boundary_from_camera_pose,
    camera_pose_from_human,
    evaluate,
)
from v19_da3_depth_correction_ablation import load_raw_pair  # noqa: E402
from v19_da3_explicit_geometry_correction_probe import (  # noqa: E402
    distribution,
    human_metrics,
    sample_cloud,
    scene_alignment_metrics,
    transform_points,
)
from v20_shot_scale_consistency_probe import calibrated_targets, scale_pose  # noqa: E402
from v22_explicit_metric_bridge_selection import (  # noqa: E402
    DEFAULT_V21,
    load_cases,
    load_shards,
)


DEFAULT_V22 = (
    REPO_ROOT
    / "output"
    / "v22_explicit_metric_bridge"
    / "final_seed1"
    / "v22_explicit_metric_bridge.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "output" / "v22_explicit_metric_bridge" / "gt_rotation_oracle"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v22_report", type=Path, default=DEFAULT_V22)
    parser.add_argument("--v21_report", type=Path, default=DEFAULT_V21)
    parser.add_argument(
        "--v10_report",
        type=Path,
        default=REPO_ROOT
        / "output"
        / "v10_candidate_selection"
        / "oracle_gt_4source"
        / "oracle_candidate_selection_metrics.json",
    )
    parser.add_argument(
        "--stream_dir",
        type=Path,
        default=REPO_ROOT / "output" / "v18_human_metric_translation" / "stream_cache",
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pointmap_variant", default="median_ratio_q15_gate_lt95")
    parser.add_argument("--point_samples", type=int, default=6000)
    parser.add_argument("--raw_confidence_threshold", type=float, default=1.0)
    parser.add_argument("--mask_dilate", type=int, default=11)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args()


def run_case(
    name: str,
    v22: dict,
    v21: dict,
    v10: dict,
    stream_row: dict,
    args: argparse.Namespace,
    index: int,
) -> dict:
    with np.load(stream_row["cache_path"]) as stream:
        raw_poses = np.stack([stream["old_pose"][-1], stream["new_pose"]]).astype(np.float32)
        intrinsics = np.stack([stream["old_intrinsics"][-1], stream["new_intrinsics"]]).astype(
            np.float32
        )
        translations = np.stack([stream["old_transl"][-1], stream["new_transl"]]).astype(
            np.float32
        )
        root_scales = [float(v21["root_scales"]["old"]), float(v21["root_scales"]["new"])]
        old_pose = scale_pose(raw_poses[0], root_scales[0])
        new_pose = scale_pose(raw_poses[1], root_scales[1])
        target_pose, gt_old_world, gt_new_world = calibrated_targets(stream, old_pose)
    old_root = translations[0] * root_scales[0]
    new_root = translations[1] * root_scales[1]
    old_anchor_world = transform_points(old_pose, old_root[None])[0]
    camera_pose = camera_pose_from_human(target_pose[:3, :3], old_anchor_world, new_root)
    transform = boundary_from_camera_pose(camera_pose, new_pose)

    raw = load_raw_pair(Path(v10["paths"]["human3r_local_reset"]))
    kernel = np.ones((int(args.mask_dilate), int(args.mask_dilate)), dtype=np.uint8)
    masks = [cv2.dilate(value.astype(np.uint8), kernel, iterations=1) for value in raw["mask"]]
    scene_variant = v21["variants"][str(args.pointmap_variant)]
    scene_scales = [
        float(scene_variant["old_scene_scale"]),
        float(scene_variant["new_scene_scale"]),
    ]
    poses = [old_pose, new_pose]
    rng = np.random.default_rng(int(args.seed) + 1009 * index)
    clouds = []
    for frame in range(2):
        clouds.append(
            sample_cloud(
                raw["depth"][frame] * scene_scales[frame],
                intrinsics[frame],
                poses[frame],
                masks[frame],
                raw["confidence"][frame],
                float(args.raw_confidence_threshold),
                int(args.point_samples),
                rng,
            )
        )
    return {
        "case_name": name,
        "source": v22["source"],
        "v22": v22["variants"]["safe_gravity_absolute_scene_scale"],
        "gt_rotation_same_metric_translation": {
            "transform": transform.astype(float).tolist(),
            "camera": evaluate(transform, new_pose, target_pose),
            "human": human_metrics(
                transform,
                old_root,
                new_root,
                old_pose,
                new_pose,
                gt_old_world,
                gt_new_world,
            ),
            "scene": scene_alignment_metrics(transform, clouds[1], clouds[0]),
        },
    }


def aggregate(rows: list[dict], key: str) -> dict:
    values = [row[key] for row in rows]
    return {
        "camera_translation_m": distribution([row["camera"]["translation_m"] for row in values]),
        "camera_rotation_deg": distribution([row["camera"]["rotation_deg"] for row in values]),
        "human_motion_error_m": distribution(
            [row["human"]["root_motion_error_m"] for row in values]
        ),
        "scene_trimmed_mean_m": distribution(
            [row["scene"]["trimmed_mean_m"] for row in values]
        ),
        "combined_catastrophic_rate": float(
            np.mean(
                [
                    row["camera"]["translation_m"] > 2.0
                    or row["camera"]["rotation_deg"] > 45.0
                    or row["human"]["root_motion_error_m"] > 0.50
                    or row["scene"]["trimmed_mean_m"] > 1.0
                    for row in values
                ]
            )
        ),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    v22 = load_cases(args.v22_report)
    v21 = load_cases(args.v21_report)
    v10 = load_cases(args.v10_report)
    streams = load_shards(args.stream_dir, "v18_stream_shard_*_of_*.json")
    rows = []
    for index, name in enumerate(sorted(v22)):
        rows.append(run_case(name, v22[name], v21[name], v10[name], streams[name], args, index))
        print(f"V22 GT rotation oracle {index + 1}/{len(v22)}", flush=True)
    report = {
        "experiment": "V22 GT rotation with unchanged deployable metric translation equation",
        "case_count": len(rows),
        "protocol": {
            "gt_information": "target camera rotation only",
            "gt_translation": False,
            "metric_scale": "unchanged V22 DA3 root/background scales",
            "human_motion": "unchanged V22 one-frame prediction",
            "translation_solver": "unchanged explicit human-root equation",
        },
        "overall": {
            "v22": aggregate(rows, "v22"),
            "gt_rotation_same_metric_translation": aggregate(
                rows, "gt_rotation_same_metric_translation"
            ),
        },
        "by_source": {
            source: {
                "v22": aggregate([row for row in rows if row["source"] == source], "v22"),
                "gt_rotation_same_metric_translation": aggregate(
                    [row for row in rows if row["source"] == source],
                    "gt_rotation_same_metric_translation",
                ),
            }
            for source in sorted({row["source"] for row in rows})
        },
        "cases": rows,
    }
    output = args.output_dir / "v22_gt_rotation_metric_bridge_oracle.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(f">> wrote {output}", flush=True)


if __name__ == "__main__":
    main()
