#!/usr/bin/env python3
"""Audit scene-continuity gating for the V40 post-cut background scale."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from v19_da3_depth_correction_ablation import load_raw_pair
from v19_da3_explicit_geometry_correction_probe import sample_cloud, scene_alignment_metrics
from v20_shot_scale_consistency_probe import scale_pose
from v22_explicit_metric_bridge_selection import load_cases, load_shards


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V40 = REPO_ROOT / "output/v40_human_background_scale_fusion/v40_human_background_scale_fusion.json"
DEFAULT_V36 = REPO_ROOT / "output/v36_final_explicit_metric_bridge/v36_final_explicit_metric_bridge.json"
DEFAULT_V10 = REPO_ROOT / "output/v10_candidate_selection/oracle_gt_4source/oracle_candidate_selection_metrics.json"
DEFAULT_STREAM = REPO_ROOT / "output/v18_human_metric_translation/stream_cache"
DEFAULT_OUTPUT = REPO_ROOT / "output/v41_background_scale_scene_safety"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v40_report", type=Path, default=DEFAULT_V40)
    parser.add_argument("--v36_report", type=Path, default=DEFAULT_V36)
    parser.add_argument("--v10_report", type=Path, default=DEFAULT_V10)
    parser.add_argument("--stream_dir", type=Path, default=DEFAULT_STREAM)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--point_samples", type=int, default=6000)
    parser.add_argument("--raw_confidence_threshold", type=float, default=1.0)
    parser.add_argument("--mask_dilate", type=int, default=11)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args()


def run_case(
    v40: dict,
    v36: dict,
    v10: dict,
    stream_row: dict,
    args: argparse.Namespace,
    index: int,
) -> dict:
    with np.load(stream_row["cache_path"]) as stream:
        raw_poses = np.stack([stream["old_pose"][-1], stream["new_pose"]]).astype(np.float32)
        intrinsics = np.stack([stream["old_intrinsics"][-1], stream["new_intrinsics"]]).astype(np.float32)
    raw = load_raw_pair(Path(v10["paths"]["human3r_local_reset"]))
    kernel = np.ones((int(args.mask_dilate), int(args.mask_dilate)), dtype=np.uint8)
    masks = [cv2.dilate(value.astype(np.uint8), kernel, iterations=1) for value in raw["mask"]]
    scene_scales = [
        float(v36["scene_scale_sets"]["absolute"]["old"]),
        float(v36["scene_scale_sets"]["absolute"]["new"]),
    ]
    variants = {}
    for variant in ("human_root", "human_old_background_new"):
        # Use identical point samples so scene deltas only reflect the candidate transform.
        rng = np.random.default_rng(int(args.seed) + 1009 * index)
        value = v40["variants"][variant]
        scales = [float(value["scales"]["old"]), float(value["scales"]["new"])]
        poses = [scale_pose(raw_poses[frame], scales[frame]) for frame in range(2)]
        clouds = [
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
            for frame in range(2)
        ]
        transform = np.asarray(value["transform"], dtype=np.float32)
        variants[variant] = {
            "camera": value["camera"],
            "human": value["human"],
            "scene": scene_alignment_metrics(transform, clouds[1], clouds[0]),
        }
    baseline = variants["human_root"]
    candidate = variants["human_old_background_new"]
    return {
        "case_name": v40["case_name"],
        "source": v40["source"],
        "scale_ratio": v40["scale_ratio"],
        "camera_delta_m": float(
            candidate["camera"]["translation_m"] - baseline["camera"]["translation_m"]
        ),
        "scene_delta_m": float(
            candidate["scene"]["trimmed_mean_m"] - baseline["scene"]["trimmed_mean_m"]
        ),
        "variants": variants,
    }


def summarize(rows: list[dict], threshold: float) -> dict:
    selected = []
    for row in rows:
        use_candidate = bool(row["scene_delta_m"] < -threshold)
        selected.append(row["variants"]["human_old_background_new" if use_candidate else "human_root"])
    baseline = [row["variants"]["human_root"] for row in rows]
    translation = np.asarray([row["camera"]["translation_m"] for row in selected])
    base_translation = np.asarray([row["camera"]["translation_m"] for row in baseline])
    scene = np.asarray([row["scene"]["trimmed_mean_m"] for row in selected])
    base_scene = np.asarray([row["scene"]["trimmed_mean_m"] for row in baseline])
    return {
        "selected_count": int(sum(row["scene_delta_m"] < -threshold for row in rows)),
        "camera_translation_mean_m": float(translation.mean()),
        "camera_translation_median_m": float(np.median(translation)),
        "camera_translation_p90_m": float(np.quantile(translation, 0.90)),
        "camera_translation_p95_m": float(np.quantile(translation, 0.95)),
        "scene_mean_m": float(scene.mean()),
        "camera_improved_005m": int(np.sum(translation + 0.05 < base_translation)),
        "camera_harmful_005m": int(np.sum(translation > base_translation + 0.05)),
        "camera_improved_010m": int(np.sum(translation + 0.10 < base_translation)),
        "camera_harmful_010m": int(np.sum(translation > base_translation + 0.10)),
        "scene_harmful_005m": int(np.sum(scene > base_scene + 0.05)),
    }


def main() -> None:
    args = parse_args()
    v40 = load_cases(args.v40_report)
    v36 = load_cases(args.v36_report)
    v10 = load_cases(args.v10_report)
    streams = load_shards(args.stream_dir, "v18_stream_shard_*_of_*.json")
    names = sorted(set(v40) & set(v36) & set(v10) & set(streams))
    if len(names) != 180:
        raise RuntimeError(f"Expected 180 common cases, got {len(names)}")
    rows = []
    for index, name in enumerate(names):
        rows.append(run_case(v40[name], v36[name], v10[name], streams[name], args, index))
        if (index + 1) % 20 == 0:
            print(f"V41 scene audit {index + 1}/{len(names)}", flush=True)
    thresholds = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05)
    report = {
        "experiment": "V41 background-scale scene-continuity safety audit",
        "case_count": len(rows),
        "protocol": {
            "runtime_selection_signal": "cross-shot scene trimmed-mean improvement only",
            "gt_runtime_information": False,
            "point_samples": int(args.point_samples),
        },
        "thresholds": {
            f"scene_gain_{threshold:.3f}": summarize(rows, threshold)
            for threshold in thresholds
        },
        "by_source": {
            source: {
                f"scene_gain_{threshold:.3f}": summarize(
                    [row for row in rows if row["source"] == source], threshold
                )
                for threshold in thresholds
            }
            for source in sorted({row["source"] for row in rows})
        },
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "v41_background_scale_scene_safety_audit.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({"thresholds": report["thresholds"], "by_source": report["by_source"]}, indent=2))
    print(f">> wrote {output}")


if __name__ == "__main__":
    main()
