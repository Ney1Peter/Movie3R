#!/usr/bin/env python3
"""Internal selection and loading helpers for explicit metric candidates."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from boundary_depth_correction_support import load_raw_pair  # noqa: E402
from boundary_geometry_support import (  # noqa: E402
    distribution,
    sample_cloud,
    scene_alignment_metrics,
)
from boundary_shot_scale_support import scale_pose  # noqa: E402


DEFAULT_TORSO = (
    REPO_ROOT
    / "output"
    / "v20_shot_scale_consistency"
    / "full180_independent_bound45_conf1_fair"
    / "v20_shot_scale_consistency.json"
)
DEFAULT_GRAVITY = (
    REPO_ROOT
    / "output"
    / "v20_shot_scale_consistency"
    / "full180_independent_gravity100_conf1_fair"
    / "v20_shot_scale_consistency.json"
)
DEFAULT_V21 = (
    REPO_ROOT
    / "output"
    / "v21_absolute_shot_background_scale"
    / "gated_full180"
    / "v21_absolute_shot_background_scale.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "output" / "v22_explicit_metric_bridge"
METHOD = "torso_first1"
BOUND = "b00"
POINTMAP_VARIANT = "median_ratio_q15_gate_lt95"
IDENTITY_BOUNDS = (0.05, 0.10, 0.15, 0.20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torso_report", type=Path, default=DEFAULT_TORSO)
    parser.add_argument("--gravity_report", type=Path, default=DEFAULT_GRAVITY)
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
    parser.add_argument(
        "--v16_dir",
        type=Path,
        default=REPO_ROOT / "output" / "v16_human_aware_rotation_residual" / "candidate_cache",
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pointmap_variant", default=POINTMAP_VARIANT)
    parser.add_argument("--gravity_min_angle_deg", type=float, default=7.5)
    parser.add_argument("--gravity_min_inlier_ratio", type=float, default=0.5)
    parser.add_argument("--gravity_min_reference_alignment", type=float, default=0.8)
    parser.add_argument("--point_samples", type=int, default=6000)
    parser.add_argument("--raw_confidence_threshold", type=float, default=1.0)
    parser.add_argument("--mask_dilate", type=int, default=11)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args()


def load_cases(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {row["case_name"]: row for row in payload["cases"]}


def load_shards(root: Path, pattern: str) -> dict[str, dict]:
    rows = []
    for path in sorted(glob.glob(str(root / pattern))):
        rows.extend(json.loads(Path(path).read_text(encoding="utf-8"))["cases"])
    return {row["case_name"]: row for row in rows}


def gravity_is_safe(row: dict, args: argparse.Namespace) -> tuple[bool, dict]:
    diagnostics = row["ground_diagnostics"]
    old_frames = diagnostics["old"]["frames"]
    new_frame = diagnostics["new_1f"]["frames"][0]
    inlier = min(
        float(np.mean([frame["inlier_ratio"] for frame in old_frames])),
        float(new_frame["inlier_ratio"]),
    )
    alignment = min(
        float(np.mean([frame["reference_alignment"] for frame in old_frames])),
        float(new_frame["reference_alignment"]),
    )
    angle = float(
        row["fixed_candidates"]["fixed_torso_motion_gravity_1f_resolve_t"]["gravity"][
            "bounded_residual_deg"
        ]
    )
    accepted = bool(
        angle >= float(args.gravity_min_angle_deg)
        and inlier >= float(args.gravity_min_inlier_ratio)
        and alignment >= float(args.gravity_min_reference_alignment)
    )
    return accepted, {
        "accepted": accepted,
        "angle_deg": angle,
        "inlier_ratio": inlier,
        "reference_alignment": alignment,
    }


def sample_pair(
    raw: dict,
    depths: list[np.ndarray],
    poses: np.ndarray,
    masks: list[np.ndarray],
    args: argparse.Namespace,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    clouds = []
    for frame in range(2):
        clouds.append(
            sample_cloud(
                depths[frame],
                raw["intrinsics"][frame],
                poses[frame],
                masks[frame],
                raw["confidence"][frame],
                float(args.raw_confidence_threshold),
                int(args.point_samples),
                rng,
            )
        )
    return clouds[0], clouds[1]


def run_case(
    case_name: str,
    torso: dict,
    gravity: dict,
    v21: dict,
    v10: dict,
    stream: dict,
    v16: dict,
    args: argparse.Namespace,
    index: int,
) -> dict:
    with np.load(stream["cache_path"]) as cache:
        raw_poses = np.stack([cache["old_pose"][-1], cache["new_pose"]]).astype(np.float32)
        intrinsics = np.stack([cache["old_intrinsics"][-1], cache["new_intrinsics"]]).astype(
            np.float32
        )
    raw = load_raw_pair(Path(v10["paths"]["human3r_local_reset"]))
    raw["intrinsics"] = intrinsics
    kernel = np.ones((int(args.mask_dilate), int(args.mask_dilate)), dtype=np.uint8)
    masks = [cv2.dilate(value.astype(np.uint8), kernel, iterations=1) for value in raw["mask"]]

    torso_value = torso["methods"][METHOD]["variants"][BOUND]
    gravity_value = gravity["methods"][METHOD]["variants"][BOUND]
    use_gravity, gravity_diagnostics = gravity_is_safe(v16, args)
    selected_value = gravity_value if use_gravity else torso_value
    selected_transform = np.asarray(selected_value["transform"], dtype=np.float32)
    torso_transform = np.asarray(torso_value["transform"], dtype=np.float32)

    root_scales = [float(v21["root_scales"]["old"]), float(v21["root_scales"]["new"])]
    scene_variant = v21["variants"][str(args.pointmap_variant)]
    scene_scales = [
        float(scene_variant["old_scene_scale"]),
        float(scene_variant["new_scene_scale"]),
    ]
    absolute_da3_scales = [
        float(v21["calibration"]["old"]["scales"]["median_ratio"]),
        float(v21["calibration"]["new"]["scales"]["median_ratio"]),
    ]
    scaled_poses = np.stack(
        [scale_pose(raw_poses[frame], root_scales[frame]) for frame in range(2)]
    )
    root_depths = [raw["depth"][frame] * root_scales[frame] for frame in range(2)]
    scene_scale_sets = {"absolute": scene_scales}
    for bound in IDENTITY_BOUNDS:
        key = f"identity{int(round(100 * bound)):02d}"
        lower = 1.0 / (1.0 + float(bound))
        upper = 1.0 + float(bound)
        scene_scale_sets[key] = [
            1.0 if lower <= absolute_da3_scales[frame] <= upper else scene_scales[frame]
            for frame in range(2)
        ]
    seed = int(args.seed) + 1009 * index
    root_old, root_new = sample_pair(raw, root_depths, scaled_poses, masks, args, seed)
    scene_clouds = {}
    for key, scales in scene_scale_sets.items():
        depths = [raw["depth"][frame] * scales[frame] for frame in range(2)]
        scene_clouds[key] = sample_pair(raw, depths, scaled_poses, masks, args, seed)

    fixed_transform = np.asarray(torso["fixed"]["transform"], dtype=np.float32)
    fixed_old, fixed_new = sample_pair(raw, raw["depth"], raw_poses, masks, args, seed)
    variants = {
        "fixed_explicit": {
            "camera": torso["fixed"]["camera"],
            "human": torso["fixed"]["human"],
            "scene": scene_alignment_metrics(fixed_transform, fixed_new, fixed_old),
            "transform": fixed_transform.astype(float).tolist(),
        },
        "torso_root_scale": {
            "camera": torso_value["camera"],
            "human": torso_value["human"],
            "scene": scene_alignment_metrics(torso_transform, root_new, root_old),
            "transform": torso_transform.astype(float).tolist(),
        },
        "safe_gravity_root_scale": {
            "camera": selected_value["camera"],
            "human": selected_value["human"],
            "scene": scene_alignment_metrics(selected_transform, root_new, root_old),
            "transform": selected_transform.astype(float).tolist(),
        },
        "torso_absolute_scene_scale": {
            "camera": torso_value["camera"],
            "human": torso_value["human"],
            "scene": scene_alignment_metrics(
                torso_transform, scene_clouds["absolute"][1], scene_clouds["absolute"][0]
            ),
            "transform": torso_transform.astype(float).tolist(),
        },
        "safe_gravity_absolute_scene_scale": {
            "camera": selected_value["camera"],
            "human": selected_value["human"],
            "scene": scene_alignment_metrics(
                selected_transform, scene_clouds["absolute"][1], scene_clouds["absolute"][0]
            ),
            "transform": selected_transform.astype(float).tolist(),
        },
    }
    for bound in IDENTITY_BOUNDS:
        key = f"identity{int(round(100 * bound)):02d}"
        variants[f"safe_gravity_scene_{key}"] = {
            "camera": selected_value["camera"],
            "human": selected_value["human"],
            "scene": scene_alignment_metrics(
                selected_transform, scene_clouds[key][1], scene_clouds[key][0]
            ),
            "transform": selected_transform.astype(float).tolist(),
        }
    return {
        "case_name": case_name,
        "source": torso["source"],
        "gravity": gravity_diagnostics,
        "root_scales": {"old": root_scales[0], "new": root_scales[1]},
        "scene_scales": {"old": scene_scales[0], "new": scene_scales[1]},
        "absolute_da3_scene_scales": {
            "old": absolute_da3_scales[0],
            "new": absolute_da3_scales[1],
        },
        "scene_scale_sets": {
            key: {"old": scales[0], "new": scales[1]} for key, scales in scene_scale_sets.items()
        },
        "variants": variants,
    }


def aggregate(rows: list[dict], variant: str) -> dict:
    values = [row["variants"][variant] for row in rows]
    fixed = [row["variants"]["fixed_explicit"] for row in rows]
    torso = [row["variants"]["torso_root_scale"] for row in rows]
    camera = np.asarray([row["camera"]["translation_m"] for row in values])
    rotation = np.asarray([row["camera"]["rotation_deg"] for row in values])
    human = np.asarray([row["human"]["root_motion_error_m"] for row in values])
    scene = np.asarray([row["scene"]["trimmed_mean_m"] for row in values])
    fixed_camera = np.asarray([row["camera"]["translation_m"] for row in fixed])
    fixed_human = np.asarray([row["human"]["root_motion_error_m"] for row in fixed])
    fixed_scene = np.asarray([row["scene"]["trimmed_mean_m"] for row in fixed])
    torso_camera = np.asarray([row["camera"]["translation_m"] for row in torso])
    torso_rotation = np.asarray([row["camera"]["rotation_deg"] for row in torso])
    torso_scene = np.asarray([row["scene"]["trimmed_mean_m"] for row in torso])
    return {
        "camera_translation_m": distribution(camera.tolist()),
        "camera_rotation_deg": distribution(rotation.tolist()),
        "human_motion_error_m": distribution(human.tolist()),
        "scene_trimmed_mean_m": distribution(scene.tolist()),
        "translation_catastrophic_rate_2m": float(np.mean(camera > 2.0)),
        "combined_catastrophic_rate": float(
            np.mean((camera > 2.0) | (rotation > 45.0) | (human > 0.50) | (scene > 1.0))
        ),
        "strict_success_rate": float(
            np.mean((camera < 0.5) & (rotation < 10.0) & (human < 0.10) & (scene < 0.20))
        ),
        "loose_success_rate": float(
            np.mean((camera < 1.0) & (rotation < 20.0) & (human < 0.10) & (scene < 0.50))
        ),
        "all_three_improved_over_fixed_rate": float(
            np.mean((camera < fixed_camera) & (human < fixed_human) & (scene < fixed_scene))
        ),
        "camera_harmful_over_torso_010m": float(np.mean(camera > torso_camera + 0.10)),
        "rotation_harmful_over_torso_5deg": float(np.mean(rotation > torso_rotation + 5.0)),
        "scene_harmful_over_torso_010m": float(np.mean(scene > torso_scene + 0.10)),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torso = load_cases(args.torso_report)
    gravity = load_cases(args.gravity_report)
    v21 = load_cases(args.v21_report)
    v10 = load_cases(args.v10_report)
    streams = load_shards(args.stream_dir, "v18_stream_shard_*_of_*.json")
    v16 = load_shards(args.v16_dir, "v16_candidates_shard_*_of_*.json")
    names = sorted(torso)
    if not all(len(mapping) == 180 for mapping in (torso, gravity, v21, v10, streams, v16)):
        raise RuntimeError("Expected 180 aligned cases in every input")
    rows = []
    for index, name in enumerate(names):
        rows.append(
            run_case(
                name,
                torso[name],
                gravity[name],
                v21[name],
                v10[name],
                streams[name],
                v16[name],
                args,
                index,
            )
        )
        print(f"Explicit metric selection {index + 1}/{len(names)}", flush=True)
    variants = sorted(rows[0]["variants"])
    report = {
        "experiment": "Safe explicit metric shot bridge composition",
        "case_count": len(rows),
        "protocol": {
            "hard_reset": True,
            "post_cut_frames": 1,
            "rotation": "torso-motion plus diagnostic-safe gravity",
            "translation": "DA3 torso root scale plus explicit human-root camera equation",
            "pointmap": str(args.pointmap_variant),
            "identity_if_da3_metric_scale_close_bounds": list(IDENTITY_BOUNDS),
            "gravity_rule": {
                "min_angle_deg": float(args.gravity_min_angle_deg),
                "min_inlier_ratio": float(args.gravity_min_inlier_ratio),
                "min_reference_alignment": float(args.gravity_min_reference_alignment),
            },
            "gravity_acceptance_rate": float(np.mean([row["gravity"]["accepted"] for row in rows])),
            "learned_components": False,
            "cross_cut_fitting": False,
            "fixed_shot_level_parameters": True,
        },
        "overall": {variant: aggregate(rows, variant) for variant in variants},
        "by_source": {
            source: {
                variant: aggregate([row for row in rows if row["source"] == source], variant)
                for variant in variants
            }
            for source in sorted({row["source"] for row in rows})
        },
        "v21_chain_stability": json.loads(args.v21_report.read_text(encoding="utf-8"))[
            "chain_stability"
        ][str(args.pointmap_variant)],
        "cases": rows,
    }
    output = args.output_dir / "v22_explicit_metric_bridge.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(f">> wrote {output}", flush=True)


if __name__ == "__main__":
    main()
