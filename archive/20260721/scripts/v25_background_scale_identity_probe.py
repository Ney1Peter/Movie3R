#!/usr/bin/env python3
"""Re-evaluate V24 scene continuity with V22 identity-bounded scene scales."""

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

from v19_da3_depth_correction_ablation import load_raw_pair  # noqa: E402
from v19_da3_explicit_geometry_correction_probe import (  # noqa: E402
    distribution,
    sample_cloud,
    scene_alignment_metrics,
)
from v20_shot_scale_consistency_probe import scale_pose  # noqa: E402
from v22_explicit_metric_bridge_selection import load_cases, load_shards  # noqa: E402


DEFAULT_V22 = (
    REPO_ROOT
    / "output"
    / "v22_explicit_metric_bridge"
    / "final_seed1"
    / "v22_explicit_metric_bridge.json"
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
DEFAULT_OUTPUT = REPO_ROOT / "output" / "v25_background_scale_identity"
SELECTED = "safe_tiered_extension_vggt"
SCALE_SETS = ("absolute", "identity05", "identity10", "identity15", "identity20")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v22_report", type=Path, default=DEFAULT_V22)
    parser.add_argument("--v24_report", type=Path, default=DEFAULT_V24)
    parser.add_argument("--v10_report", type=Path, default=DEFAULT_V10)
    parser.add_argument("--stream_dir", type=Path, default=DEFAULT_STREAM)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--point_samples", type=int, default=6000)
    parser.add_argument("--raw_confidence_threshold", type=float, default=1.0)
    parser.add_argument("--mask_dilate", type=int, default=11)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    v22 = load_cases(args.v22_report)
    v24 = load_cases(args.v24_report)
    v10 = load_cases(args.v10_report)
    streams = load_shards(args.stream_dir, "v18_stream_shard_*_of_*.json")
    names = sorted(set(v22) & set(v24) & set(v10) & set(streams))
    if len(names) != 180:
        raise RuntimeError(f"Expected 180 common cases, got {len(names)}")

    rows = []
    kernel = np.ones((int(args.mask_dilate), int(args.mask_dilate)), dtype=np.uint8)
    for index, name in enumerate(names):
        selected = v24[name]["variants"][SELECTED]
        transform = np.asarray(selected["transform"], dtype=np.float32)
        scales = v22[name]["scene_scale_sets"]
        with np.load(streams[name]["cache_path"]) as stream:
            raw_poses = np.stack([stream["old_pose"][-1], stream["new_pose"]]).astype(
                np.float32
            )
            intrinsics = np.stack(
                [stream["old_intrinsics"][-1], stream["new_intrinsics"]]
            ).astype(np.float32)
        root_scales = [
            float(v22[name]["root_scales"]["old"]),
            float(v22[name]["root_scales"]["new"]),
        ]
        poses = [scale_pose(raw_poses[frame], root_scales[frame]) for frame in range(2)]
        raw = load_raw_pair(Path(v10[name]["paths"]["human3r_local_reset"]))
        masks = [cv2.dilate(value.astype(np.uint8), kernel, iterations=1) for value in raw["mask"]]
        variants = {}
        for scale_index, scale_name in enumerate(SCALE_SETS):
            rng = np.random.default_rng(int(args.seed) + 1009 * index)
            scene_scale = [
                float(scales[scale_name]["old"]),
                float(scales[scale_name]["new"]),
            ]
            clouds = [
                sample_cloud(
                    raw["depth"][frame] * scene_scale[frame],
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
            variants[scale_name] = {
                "old_scene_scale": scene_scale[0],
                "new_scene_scale": scene_scale[1],
                "scene": scene_alignment_metrics(transform, clouds[1], clouds[0]),
            }
        rows.append(
            {
                "case_name": name,
                "source": v24[name]["source"],
                "camera": selected["camera"],
                "human": selected["human"],
                "variants": variants,
            }
        )
        if (index + 1) % 20 == 0 or index + 1 == len(names):
            print(f">> V25 scene identity {index + 1}/{len(names)}", flush=True)

    baseline_scene = np.asarray(
        [row["variants"]["absolute"]["scene"]["trimmed_mean_m"] for row in rows]
    )

    def aggregate(scale_name: str, subset: list[dict]) -> dict:
        camera = np.asarray([row["camera"]["translation_m"] for row in subset])
        rotation = np.asarray([row["camera"]["rotation_deg"] for row in subset])
        human = np.asarray([row["human"]["root_motion_error_m"] for row in subset])
        scene = np.asarray(
            [row["variants"][scale_name]["scene"]["trimmed_mean_m"] for row in subset]
        )
        base = np.asarray(
            [row["variants"]["absolute"]["scene"]["trimmed_mean_m"] for row in subset]
        )
        return {
            "scene_trimmed_mean_m": distribution(scene.tolist()),
            "scene_catastrophic_rate": float(np.mean(scene > 1.0)),
            "combined_catastrophic_rate": float(
                np.mean((camera > 2.0) | (rotation > 45.0) | (human > 0.50) | (scene > 1.0))
            ),
            "strict_success_rate": float(
                np.mean((camera < 0.5) & (rotation < 10.0) & (human < 0.10) & (scene < 0.20))
            ),
            "scene_harmful_010m": float(np.mean(scene > base + 0.10)),
            "scene_improved_010m": float(np.mean(scene < base - 0.10)),
        }

    report = {
        "experiment": "V25 V24 rotation with identity-bounded V22 scene scales",
        "case_count": len(rows),
        "protocol": {
            "camera_human_transform": "unchanged V24 selected fixed shot-level SE(3)",
            "scene_scale_rules": list(SCALE_SETS),
            "source_specific_rule": False,
            "gt_runtime_information": False,
        },
        "overall": {scale_name: aggregate(scale_name, rows) for scale_name in SCALE_SETS},
        "by_source": {
            source: {
                scale_name: aggregate(
                    scale_name, [row for row in rows if row["source"] == source]
                )
                for scale_name in SCALE_SETS
            }
            for source in sorted({row["source"] for row in rows})
        },
        "absolute_scene_catastrophic_cases": [
            row["case_name"]
            for row, value in zip(rows, baseline_scene)
            if value > 1.0
        ],
        "cases": rows,
    }
    output = args.output_dir / "v25_background_scale_identity_probe.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["overall"], indent=2))
    print(f">> wrote {output}")


if __name__ == "__main__":
    main()
