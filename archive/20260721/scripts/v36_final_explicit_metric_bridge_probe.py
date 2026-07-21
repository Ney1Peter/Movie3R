#!/usr/bin/env python3
"""Compose V36 rotation with the synchronized V22 metric bridge."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np

from v18_da3_metric_depth_probe import (
    boundary_from_camera_pose,
    camera_pose_from_human,
    evaluate,
)
from v19_da3_depth_correction_ablation import load_raw_pair
from v19_da3_explicit_geometry_correction_probe import (
    distribution,
    human_metrics,
    sample_cloud,
    scene_alignment_metrics,
    transform_points,
)
from v20_shot_scale_consistency_probe import calibrated_targets, scale_pose
from v22_explicit_metric_bridge_selection import load_cases, load_shards
from v25_holdout_rotation_validation import safe_gravity
from v32_consensus_texture_safety_audit import selected_rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V22 = REPO_ROOT / "output/v22_explicit_metric_bridge/final_seed1/v22_explicit_metric_bridge.json"
DEFAULT_V15 = REPO_ROOT / "output/v15_wide_baseline_boundary_bridge/candidate_cache"
DEFAULT_V16 = REPO_ROOT / "output/v16_human_aware_rotation_residual/candidate_cache"
DEFAULT_V10 = REPO_ROOT / "output/v10_candidate_selection/oracle_gt_4source/oracle_candidate_selection_metrics.json"
DEFAULT_STREAM = REPO_ROOT / "output/v18_human_metric_translation/stream_cache"
DEFAULT_OUTPUT = REPO_ROOT / "output/v36_final_explicit_metric_bridge"
V22_SELECTED = "safe_gravity_absolute_scene_scale"
TEXTURE_BOUND = 0.05
HUMAN_JUMP_BOUND_DEG = 30.0
LOW_JUMP_CONSENSUS_CAP_DEG = 20.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v22_report", type=Path, default=DEFAULT_V22)
    parser.add_argument("--v15_dir", type=Path, default=DEFAULT_V15)
    parser.add_argument("--v16_dir", type=Path, default=DEFAULT_V16)
    parser.add_argument("--v10_report", type=Path, default=DEFAULT_V10)
    parser.add_argument("--stream_dir", type=Path, default=DEFAULT_STREAM)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--point_samples", type=int, default=6000)
    parser.add_argument("--raw_confidence_threshold", type=float, default=1.0)
    parser.add_argument("--mask_dilate", type=int, default=11)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args()


def load_v15(path: Path) -> dict[str, dict]:
    rows = []
    for shard in sorted(glob.glob(str(path / "v15_candidates_shard_*_of_*.json"))):
        rows.extend(json.loads(Path(shard).read_text(encoding="utf-8"))["cases"])
    output = {row["case_name"]: row for row in rows}
    if len(output) != 180:
        raise RuntimeError(f"Expected 180 V15 cases, got {len(output)}")
    return output


def run_case(
    name: str,
    v22: dict,
    v15: dict,
    v16: dict,
    v10: dict,
    stream_row: dict,
    args: argparse.Namespace,
    index: int,
) -> dict:
    with np.load(stream_row["cache_path"]) as stream:
        raw_poses = np.stack([stream["old_pose"][-1], stream["new_pose"]]).astype(
            np.float32
        )
        intrinsics = np.stack(
            [stream["old_intrinsics"][-1], stream["new_intrinsics"]]
        ).astype(np.float32)
        translations = np.stack(
            [stream["old_transl"][-1], stream["new_transl"]]
        ).astype(np.float32)
        root_scales = [
            float(v22["root_scales"]["old"]),
            float(v22["root_scales"]["new"]),
        ]
        old_pose = scale_pose(raw_poses[0], root_scales[0])
        new_pose = scale_pose(raw_poses[1], root_scales[1])
        target_pose, gt_old_world, gt_new_world = calibrated_targets(stream, old_pose)
    old_root = translations[0] * root_scales[0]
    new_root = translations[1] * root_scales[1]
    old_anchor_world = transform_points(old_pose, old_root[None])[0]

    raw = load_raw_pair(Path(v10["paths"]["human3r_local_reset"]))
    kernel = np.ones((int(args.mask_dilate), int(args.mask_dilate)), dtype=np.uint8)
    masks = [cv2.dilate(value.astype(np.uint8), kernel, iterations=1) for value in raw["mask"]]
    scene_scales = [
        float(v22["scene_scale_sets"]["absolute"]["old"]),
        float(v22["scene_scale_sets"]["absolute"]["new"]),
    ]
    rng = np.random.default_rng(int(args.seed) + 1009 * index)
    clouds = []
    for frame, pose in enumerate((old_pose, new_pose)):
        clouds.append(
            sample_cloud(
                raw["depth"][frame] * scene_scales[frame],
                intrinsics[frame],
                pose,
                masks[frame],
                raw["confidence"][frame],
                float(args.raw_confidence_threshold),
                int(args.point_samples),
                rng,
            )
        )

    fixed = np.asarray(v22["variants"]["fixed_explicit"]["transform"], dtype=np.float32)[
        :3, :3
    ]
    torso = np.asarray(v22["variants"][V22_SELECTED]["transform"], dtype=np.float32)[
        :3, :3
    ]
    _, gravity = safe_gravity(v16)
    candidate_key = (
        "fixed_torso_motion_gravity_1f_resolve_t"
        if gravity["accepted"]
        else "fixed_torso_motion_1f_resolve_t"
    )
    human_jump = float(v16["fixed_candidates"][candidate_key]["human_torso_jump_deg"])
    v32, branch, diagnostics = selected_rotation(
        fixed, torso, v15, TEXTURE_BOUND, consensus_cap_deg=60.0
    )
    adapted = bool(branch == "consensus" and human_jump < HUMAN_JUMP_BOUND_DEG)
    v36 = (
        selected_rotation(
            fixed,
            torso,
            v15,
            TEXTURE_BOUND,
            consensus_cap_deg=LOW_JUMP_CONSENSUS_CAP_DEG,
        )[0]
        if adapted
        else v32
    )
    gt = np.asarray(v15["baselines"]["boundary_oracle"]["transform"], dtype=np.float32)[
        :3, :3
    ]
    rotations = {"v22": torso, "v32": v32, "v36": v36, "gt_rotation": gt}
    variants = {}
    for variant, boundary_rotation in rotations.items():
        camera_rotation = boundary_rotation @ new_pose[:3, :3]
        camera_pose = camera_pose_from_human(camera_rotation, old_anchor_world, new_root)
        transform = boundary_from_camera_pose(camera_pose, new_pose)
        variants[variant] = {
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
    return {
        "case_name": name,
        "source": v22["source"],
        "adapted": adapted,
        "human_torso_jump_deg": human_jump,
        "v32_branch": branch,
        "diagnostics": diagnostics,
        "root_scales": v22["root_scales"],
        "scene_scale_sets": v22["scene_scale_sets"],
        "variants": variants,
    }


def aggregate(rows: list[dict], variant: str, baseline: str) -> dict:
    values = [row["variants"][variant] for row in rows]
    references = [row["variants"][baseline] for row in rows]
    camera = np.asarray([row["camera"]["translation_m"] for row in values])
    rotation = np.asarray([row["camera"]["rotation_deg"] for row in values])
    human = np.asarray([row["human"]["root_motion_error_m"] for row in values])
    scene = np.asarray([row["scene"]["trimmed_mean_m"] for row in values])
    base_camera = np.asarray([row["camera"]["translation_m"] for row in references])
    base_rotation = np.asarray([row["camera"]["rotation_deg"] for row in references])
    base_human = np.asarray([row["human"]["root_motion_error_m"] for row in references])
    base_scene = np.asarray([row["scene"]["trimmed_mean_m"] for row in references])
    return {
        "camera_translation_m": distribution(camera.tolist()),
        "camera_rotation_deg": distribution(rotation.tolist()),
        "human_motion_error_m": distribution(human.tolist()),
        "scene_trimmed_mean_m": distribution(scene.tolist()),
        "combined_catastrophic_rate": float(
            np.mean((camera > 2.0) | (rotation > 45.0) | (human > 0.50) | (scene > 1.0))
        ),
        "strict_success_rate": float(
            np.mean((camera < 0.5) & (rotation < 10.0) & (human < 0.10) & (scene < 0.20))
        ),
        "camera_harmful_010m": float(np.mean(camera > base_camera + 0.10)),
        "rotation_harmful_5deg": float(np.mean(rotation > base_rotation + 5.0)),
        "human_harmful_010m": float(np.mean(human > base_human + 0.10)),
        "scene_harmful_010m": float(np.mean(scene > base_scene + 0.10)),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    v22 = load_cases(args.v22_report)
    v15 = load_v15(args.v15_dir)
    v16 = load_shards(args.v16_dir, "v16_candidates_shard_*_of_*.json")
    v10 = load_cases(args.v10_report)
    streams = load_shards(args.stream_dir, "v18_stream_shard_*_of_*.json")
    names = sorted(set(v22) & set(v15) & set(v16) & set(v10) & set(streams))
    if len(names) != 180:
        raise RuntimeError(f"Expected 180 common cases, got {len(names)}")
    rows = []
    for index, name in enumerate(names):
        rows.append(
            run_case(
                name,
                v22[name],
                v15[name],
                v16[name],
                v10[name],
                streams[name],
                args,
                index,
            )
        )
        if (index + 1) % 10 == 0 or index + 1 == len(names):
            print(f"V36 explicit metric bridge {index + 1}/{len(names)}", flush=True)
    variants = ("v22", "v32", "v36", "gt_rotation")
    report = {
        "experiment": "V36 adaptive wide rotation plus V22 synchronized metric bridge",
        "case_count": len(rows),
        "protocol": {
            "post_cut_frames": 1,
            "texture_bound": TEXTURE_BOUND,
            "human_jump_bound_deg": HUMAN_JUMP_BOUND_DEG,
            "low_jump_consensus_cap_deg": LOW_JUMP_CONSENSUS_CAP_DEG,
            "metric_scale": "unchanged V22 DA3 root/background shot scales",
            "translation": "unchanged V22 explicit human-root equation re-solved after rotation",
            "camera_pointmap_smplx_share_final_transform": True,
            "metric_scale_state": (
                "camera translation and SMPL-X root share the DA3 root scale; "
                "pointmap uses the bounded V22 background scale correction"
            ),
            "gt_runtime_information": False,
        },
        "adapted_count": int(sum(row["adapted"] for row in rows)),
        "overall_vs_v22": {
            variant: aggregate(rows, variant, "v22") for variant in variants
        },
        "overall_vs_v32": {
            variant: aggregate(rows, variant, "v32") for variant in variants
        },
        "by_source_vs_v22": {
            source: {
                variant: aggregate(
                    [row for row in rows if row["source"] == source], variant, "v22"
                )
                for variant in variants
            }
            for source in sorted({row["source"] for row in rows})
        },
        "cases": rows,
    }
    output = args.output_dir / "v36_final_explicit_metric_bridge.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "adapted_count": report["adapted_count"],
                "overall_vs_v22": report["overall_vs_v22"],
                "overall_vs_v32": report["overall_vs_v32"],
            },
            indent=2,
        )
    )
    print(f">> wrote {output}")


if __name__ == "__main__":
    main()
