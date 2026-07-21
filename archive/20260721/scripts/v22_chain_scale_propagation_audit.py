#!/usr/bin/env python3
"""Audit V22 when the middle shot keeps the scale assigned at its first frame."""

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
from v22_explicit_metric_bridge_selection import (  # noqa: E402
    DEFAULT_GRAVITY,
    DEFAULT_TORSO,
    DEFAULT_V21,
    BOUND,
    METHOD,
    gravity_is_safe,
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
DEFAULT_OUTPUT = REPO_ROOT / "output" / "v22_explicit_metric_bridge" / "chain_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torso_report", type=Path, default=DEFAULT_TORSO)
    parser.add_argument("--gravity_report", type=Path, default=DEFAULT_GRAVITY)
    parser.add_argument("--v21_report", type=Path, default=DEFAULT_V21)
    parser.add_argument("--v22_report", type=Path, default=DEFAULT_V22)
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
    parser.add_argument(
        "--v16_dir",
        type=Path,
        default=REPO_ROOT / "output" / "v16_human_aware_rotation_residual" / "candidate_cache",
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pointmap_variant", default="median_ratio_q15_gate_lt95")
    parser.add_argument("--gravity_min_angle_deg", type=float, default=7.5)
    parser.add_argument("--gravity_min_inlier_ratio", type=float, default=0.5)
    parser.add_argument("--gravity_min_reference_alignment", type=float, default=0.8)
    parser.add_argument("--point_samples", type=int, default=6000)
    parser.add_argument("--raw_confidence_threshold", type=float, default=1.0)
    parser.add_argument("--mask_dilate", type=int, default=11)
    parser.add_argument("--max_frame_gap", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args()


def chain_pairs(rows: dict[str, dict], max_gap: int) -> list[tuple[str, str]]:
    values = list(rows.values())
    pairs = []
    for first in values:
        a = first.get("record", {})
        for second in values:
            if first is second or first["source"] != second["source"]:
                continue
            b = second.get("record", {})
            if a.get("group") != b.get("group") or a.get("seqB") != b.get("seqA"):
                continue
            if abs(int(a.get("start_frame", -100000)) - int(b.get("start_frame", 100000))) <= int(
                max_gap
            ):
                pairs.append((first["case_name"], second["case_name"]))
    return pairs


def propagated_case(
    previous_name: str,
    current_name: str,
    torso: dict[str, dict],
    gravity: dict[str, dict],
    v21: dict[str, dict],
    v10: dict[str, dict],
    streams: dict[str, dict],
    v16: dict[str, dict],
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
        raw_poses = np.stack([stream["old_pose"][-1], stream["new_pose"]]).astype(np.float32)
        intrinsics = np.stack([stream["old_intrinsics"][-1], stream["new_intrinsics"]]).astype(
            np.float32
        )
        translations = np.stack([stream["old_transl"][-1], stream["new_transl"]]).astype(
            np.float32
        )
        old_pose = scale_pose(raw_poses[0], old_root_scale)
        new_pose = scale_pose(raw_poses[1], new_root_scale)
        target_pose, gt_old_world, gt_new_world = calibrated_targets(stream, old_pose)

    use_gravity, diagnostics = gravity_is_safe(v16[current_name], args)
    source_report = gravity[current_name] if use_gravity else torso[current_name]
    source_value = source_report["methods"][METHOD]["variants"][BOUND]
    source_transform = np.asarray(source_value["transform"], dtype=np.float32)
    corrected_rotation = (source_transform @ raw_poses[1])[:3, :3]
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
    clouds = []
    for frame in range(2):
        clouds.append(
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
        )
    return {
        "previous_case": previous_name,
        "current_case": current_name,
        "source": torso[current_name]["source"],
        "gravity": diagnostics,
        "scale_state": {
            "propagated_old_root": old_root_scale,
            "reestimated_old_root": float(current_scale["root_scales"]["old"]),
            "propagated_old_scene": old_scene_scale,
            "reestimated_old_scene": float(
                current_scale["variants"][str(args.pointmap_variant)]["old_scene_scale"]
            ),
            "new_root": new_root_scale,
            "new_scene": new_scene_scale,
        },
        "propagated": {
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
    torso = load_cases(args.torso_report)
    gravity = load_cases(args.gravity_report)
    v21 = load_cases(args.v21_report)
    v22 = load_cases(args.v22_report)
    v10 = load_cases(args.v10_report)
    streams = load_shards(args.stream_dir, "v18_stream_shard_*_of_*.json")
    v16 = load_shards(args.v16_dir, "v16_candidates_shard_*_of_*.json")
    pairs = chain_pairs(v21, int(args.max_frame_gap))
    rows = []
    for index, (previous_name, current_name) in enumerate(pairs):
        row = propagated_case(
            previous_name,
            current_name,
            torso,
            gravity,
            v21,
            v10,
            streams,
            v16,
            args,
            index,
        )
        current = v22[current_name]["variants"]["safe_gravity_absolute_scene_scale"]
        row["per_cut_reestimated"] = current
        rows.append(row)
        print(f"V22 chain propagation {index + 1}/{len(pairs)}", flush=True)

    root_log_delta = [
        abs(
            np.log(
                row["scale_state"]["propagated_old_root"]
                / max(row["scale_state"]["reestimated_old_root"], 1e-6)
            )
        )
        for row in rows
    ]
    scene_log_delta = [
        abs(
            np.log(
                row["scale_state"]["propagated_old_scene"]
                / max(row["scale_state"]["reestimated_old_scene"], 1e-6)
            )
        )
        for row in rows
    ]
    report = {
        "experiment": "V22 middle-shot scale-state propagation audit",
        "chain_count": len(rows),
        "protocol": {
            "chain_definition": "seqB(first) == seqA(second), same group/source, start-frame gap <= 30",
            "old_shot_scale": "propagated from the previous cut new-shot first frame",
            "new_shot_scale": "estimated once at the current cut first frame",
            "full_pose_accumulation": False,
            "scope": "scale-state persistence, not cumulative trajectory drift",
        },
        "scale_stability": {
            "root_abs_log_difference": distribution(root_log_delta),
            "scene_abs_log_difference": distribution(scene_log_delta),
            "root_within_10_percent": float(np.mean(np.asarray(root_log_delta) <= np.log(1.10))),
            "scene_within_10_percent": float(np.mean(np.asarray(scene_log_delta) <= np.log(1.10))),
            "root_within_20_percent": float(np.mean(np.asarray(root_log_delta) <= np.log(1.20))),
            "scene_within_20_percent": float(np.mean(np.asarray(scene_log_delta) <= np.log(1.20))),
        },
        "overall": {
            "per_cut_reestimated": aggregate(rows, "per_cut_reestimated"),
            "propagated": aggregate(rows, "propagated"),
        },
        "by_source": {
            source: {
                "per_cut_reestimated": aggregate(
                    [row for row in rows if row["source"] == source], "per_cut_reestimated"
                ),
                "propagated": aggregate(
                    [row for row in rows if row["source"] == source], "propagated"
                ),
            }
            for source in sorted({row["source"] for row in rows})
        },
        "cases": rows,
    }
    output = args.output_dir / "v22_chain_scale_propagation_audit.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(f">> wrote {output}", flush=True)


if __name__ == "__main__":
    main()
