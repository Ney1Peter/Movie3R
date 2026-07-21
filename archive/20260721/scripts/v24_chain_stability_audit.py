#!/usr/bin/env python3
"""Audit V24 on the 38 available A-to-B-to-C scale-persistence chains."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

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
from v22_explicit_metric_bridge_selection import load_cases, load_shards  # noqa: E402
from v22_chain_scale_propagation_audit import aggregate  # noqa: E402


DEFAULT_V21 = (
    REPO_ROOT
    / "output"
    / "v21_absolute_shot_background_scale"
    / "gated_full180"
    / "v21_absolute_shot_background_scale.json"
)
DEFAULT_V22_CHAIN = (
    REPO_ROOT
    / "output"
    / "v22_explicit_metric_bridge"
    / "chain_audit"
    / "v22_chain_scale_propagation_audit.json"
)
DEFAULT_V24 = (
    REPO_ROOT
    / "output"
    / "v24_vggt_v22_rotation_bridge"
    / "v24_vggt_v22_rotation_bridge.json"
)
DEFAULT_V10 = (
    REPO_ROOT
    / "output"
    / "v10_candidate_selection"
    / "oracle_gt_4source"
    / "oracle_candidate_selection_metrics.json"
)
DEFAULT_STREAM = REPO_ROOT / "output" / "v18_human_metric_translation" / "stream_cache"
DEFAULT_OUTPUT = (
    REPO_ROOT / "output" / "v24_vggt_v22_rotation_bridge" / "chain_audit"
)
SELECTED = "safe_tiered_extension_vggt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v21_report", type=Path, default=DEFAULT_V21)
    parser.add_argument("--v22_chain", type=Path, default=DEFAULT_V22_CHAIN)
    parser.add_argument("--v24_report", type=Path, default=DEFAULT_V24)
    parser.add_argument("--v10_report", type=Path, default=DEFAULT_V10)
    parser.add_argument("--stream_dir", type=Path, default=DEFAULT_STREAM)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pointmap_variant", default="median_ratio_q15_gate_lt95")
    parser.add_argument("--point_samples", type=int, default=6000)
    parser.add_argument("--raw_confidence_threshold", type=float, default=1.0)
    parser.add_argument("--mask_dilate", type=int, default=11)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args()


def propagated_v24_case(
    previous_name: str,
    current_name: str,
    v21: dict[str, dict],
    v24: dict[str, dict],
    v10: dict[str, dict],
    streams: dict[str, dict],
    args: argparse.Namespace,
    index: int,
) -> dict:
    previous_scale = v21[previous_name]
    current_scale = v21[current_name]
    old_root_scale = float(previous_scale["root_scales"]["new"])
    new_root_scale = float(current_scale["root_scales"]["new"])
    old_scene_scale = float(
        previous_scale["variants"][str(args.pointmap_variant)]["new_scene_scale"]
    )
    new_scene_scale = float(
        current_scale["variants"][str(args.pointmap_variant)]["new_scene_scale"]
    )

    with np.load(streams[current_name]["cache_path"]) as stream:
        raw_poses = np.stack([stream["old_pose"][-1], stream["new_pose"]]).astype(
            np.float32
        )
        intrinsics = np.stack(
            [stream["old_intrinsics"][-1], stream["new_intrinsics"]]
        ).astype(np.float32)
        translations = np.stack(
            [stream["old_transl"][-1], stream["new_transl"]]
        ).astype(np.float32)
        old_pose = scale_pose(raw_poses[0], old_root_scale)
        new_pose = scale_pose(raw_poses[1], new_root_scale)
        target_pose, gt_old_world, gt_new_world = calibrated_targets(stream, old_pose)

    selected = v24[current_name]["variants"][SELECTED]
    selected_boundary = np.asarray(selected["transform"], dtype=np.float32)
    corrected_rotation = (selected_boundary @ raw_poses[1])[:3, :3]
    old_root = translations[0] * old_root_scale
    new_root = translations[1] * new_root_scale
    old_anchor_world = transform_points(old_pose, old_root[None])[0]
    camera_pose = camera_pose_from_human(corrected_rotation, old_anchor_world, new_root)
    transform = boundary_from_camera_pose(camera_pose, new_pose)

    raw = load_raw_pair(Path(v10[current_name]["paths"]["human3r_local_reset"]))
    kernel = np.ones((int(args.mask_dilate), int(args.mask_dilate)), dtype=np.uint8)
    masks = [cv2.dilate(value.astype(np.uint8), kernel, iterations=1) for value in raw["mask"]]
    poses = [old_pose, new_pose]
    depths = [raw["depth"][0] * old_scene_scale, raw["depth"][1] * new_scene_scale]
    rng = np.random.default_rng(int(args.seed) + 1009 * index)
    clouds = [
        sample_cloud(
            depths[frame],
            intrinsics[frame],
            poses[frame],
            masks[frame],
            raw["confidence"][frame],
            float(args.raw_confidence_threshold),
            int(args.point_samples),
            rng,
        )
        for frame in range(2)
    ]
    return {
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
    }


def metric_delta(rows: list[dict], left: str, right: str) -> dict:
    keys = {
        "camera_translation_m": ("camera", "translation_m"),
        "camera_rotation_deg": ("camera", "rotation_deg"),
        "human_motion_error_m": ("human", "root_motion_error_m"),
        "scene_trimmed_mean_m": ("scene", "trimmed_mean_m"),
    }
    output = {}
    for name, path in keys.items():
        values = [row[right][path[0]][path[1]] - row[left][path[0]][path[1]] for row in rows]
        output[name] = distribution(values)
        output[name]["improved_fraction"] = float(np.mean(np.asarray(values) < 0.0))
    return output


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    v21 = load_cases(args.v21_report)
    v24 = load_cases(args.v24_report)
    v10 = load_cases(args.v10_report)
    streams = load_shards(args.stream_dir, "v18_stream_shard_*_of_*.json")
    v22_chain = json.loads(args.v22_chain.read_text(encoding="utf-8"))

    rows = []
    for index, previous in enumerate(v22_chain["cases"]):
        previous_name = previous["previous_case"]
        current_name = previous["current_case"]
        row = {
            "previous_case": previous_name,
            "current_case": current_name,
            "source": previous["source"],
            "scale_state": previous["scale_state"],
            "v22_per_cut_reestimated": previous["per_cut_reestimated"],
            "v22_propagated": previous["propagated"],
            "v24_per_cut_reestimated": v24[current_name]["variants"][SELECTED],
            "v24_propagated": propagated_v24_case(
                previous_name,
                current_name,
                v21,
                v24,
                v10,
                streams,
                args,
                index,
            ),
        }
        rows.append(row)
        print(f">> V24 chain {index + 1}/{len(v22_chain['cases'])}", flush=True)

    methods = (
        "v22_per_cut_reestimated",
        "v22_propagated",
        "v24_per_cut_reestimated",
        "v24_propagated",
    )
    report = {
        "experiment": "V24 chained-cut scale-persistence and rotation-bridge audit",
        "chain_count": len(rows),
        "protocol": {
            "chain_set": "exactly the 38 V22 A-to-B-to-C chains",
            "middle_shot_human_and_scene_scales": "propagated from the previous cut",
            "v24_rotation": "selected safe conditional VGGT rotation",
            "translation": "explicitly re-solved after V24 rotation",
            "full_pose_accumulation": False,
            "scope": "two-boundary scale persistence, not long-horizon trajectory drift",
        },
        "overall": {method: aggregate(rows, method) for method in methods},
        "v24_gain_over_v22": {
            "per_cut_reestimated": metric_delta(
                rows, "v22_per_cut_reestimated", "v24_per_cut_reestimated"
            ),
            "propagated": metric_delta(rows, "v22_propagated", "v24_propagated"),
        },
        "by_source": {
            source: {
                method: aggregate(
                    [row for row in rows if row["source"] == source], method
                )
                for method in methods
            }
            for source in sorted({row["source"] for row in rows})
        },
        "cases": rows,
    }
    output = args.output_dir / "v24_chain_stability_audit.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f">> wrote {output}")


if __name__ == "__main__":
    main()
