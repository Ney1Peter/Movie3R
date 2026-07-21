#!/usr/bin/env python3
"""Evaluate explicit multi-cue rotation and scene-scale consensus beyond V24."""

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

from v15_wide_baseline_boundary_bridge_candidates import project_rotation  # noqa: E402
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
from v24_vggt_v22_rotation_bridge_probe import (  # noqa: E402
    capped_rotation,
    candidate_rotations,
    rotation_angle_deg,
)


DEFAULT_V22 = (
    REPO_ROOT
    / "output"
    / "v22_explicit_metric_bridge"
    / "final_seed1"
    / "v22_explicit_metric_bridge.json"
)
DEFAULT_V15 = REPO_ROOT / "output" / "v15_wide_baseline_boundary_bridge" / "candidate_cache"
DEFAULT_V10 = (
    REPO_ROOT
    / "output"
    / "v10_candidate_selection"
    / "oracle_gt_4source"
    / "oracle_candidate_selection_metrics.json"
)
DEFAULT_STREAM = REPO_ROOT / "output" / "v18_human_metric_translation" / "stream_cache"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "v25_explicit_consensus_bridge"
V22_SELECTED = "safe_gravity_absolute_scene_scale"
V24_SELECTED = "safe_tiered_extension_vggt"
ROTATION_VARIANTS = (
    "v24",
    "background_1p1_fallback",
    "multiframe_dual_3p3_fallback",
    "low_torso_explicit_consensus",
    "v25_1p1_rotation",
    "v25_3p3_rotation",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v22_report", type=Path, default=DEFAULT_V22)
    parser.add_argument("--v15_dir", type=Path, default=DEFAULT_V15)
    parser.add_argument("--v10_report", type=Path, default=DEFAULT_V10)
    parser.add_argument("--stream_dir", type=Path, default=DEFAULT_STREAM)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--point_samples", type=int, default=6000)
    parser.add_argument("--raw_confidence_threshold", type=float, default=1.0)
    parser.add_argument("--mask_dilate", type=int, default=11)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args()


def load_v15(root: Path) -> dict[str, dict]:
    rows = []
    for path in sorted(glob.glob(str(root / "v15_candidates_shard_*_of_*.json"))):
        rows.extend(json.loads(Path(path).read_text(encoding="utf-8"))["cases"])
    output = {row["case_name"]: row for row in rows}
    if len(output) != 180:
        raise RuntimeError(f"Expected 180 V15 cases, got {len(output)}")
    return output


def boundary_rotation(window: dict, candidate: str = "coarse") -> np.ndarray:
    return np.asarray(window["candidates"][candidate]["transform"], dtype=np.float32)[
        :3, :3
    ]


def accepted_v24(diagnostics: dict) -> bool:
    return bool(
        diagnostics["trigger_safe_large_residual"]
        or diagnostics["trigger_safe_consensus"]
        or diagnostics["trigger_safe_low_texture_conflict"]
    )


def rotation_candidates(v22: dict, v15: dict) -> tuple[dict[str, np.ndarray], dict]:
    v24_rotations, diagnostics = candidate_rotations(v22, v15)
    base = v24_rotations[V24_SELECTED]
    fixed = np.asarray(v22["variants"]["fixed_explicit"]["transform"], dtype=np.float32)[
        :3, :3
    ]
    torso = np.asarray(v22["variants"][V22_SELECTED]["transform"], dtype=np.float32)[
        :3, :3
    ]
    full_1p1 = v15["windows"]["full_rgb_1p1"]
    background_1p1 = v15["windows"]["background_only_1p1"]
    full_3p3 = v15["windows"]["full_rgb_3p3"]
    background_3p3 = v15["windows"]["background_only_3p3"]

    background_rotation = boundary_rotation(background_1p1)
    background_residual = rotation_angle_deg(background_rotation, fixed)
    background_trigger = bool(
        not accepted_v24(diagnostics)
        and diagnostics["torso_residual_deg"] >= 30.0
        and float(full_1p1["rotation_consensus"]["spread_deg"]) > 15.0
        and float(background_1p1["rotation_consensus"]["spread_deg"]) <= 15.0
        and background_residual <= 100.0
        and background_residual >= diagnostics["torso_residual_deg"] + 5.0
    )
    background_selected = (
        capped_rotation(base, background_rotation, 60.0) if background_trigger else base
    )

    full_3_rotation = boundary_rotation(full_3p3)
    background_3_rotation = boundary_rotation(background_3p3)
    full_3_residual = rotation_angle_deg(full_3_rotation, fixed)
    background_3_residual = rotation_angle_deg(background_3_rotation, fixed)
    multiframe_agreement = rotation_angle_deg(full_3_rotation, background_3_rotation)
    multiframe_trigger = bool(
        not accepted_v24(diagnostics)
        and diagnostics["torso_residual_deg"] >= 30.0
        and float(full_1p1["rotation_consensus"]["spread_deg"]) > 15.0
        and float(full_3p3["rotation_consensus"]["spread_deg"]) <= 10.0
        and float(background_3p3["rotation_consensus"]["spread_deg"]) <= 10.0
        and multiframe_agreement <= 15.0
        and full_3_residual <= 100.0
        and background_3_residual <= 100.0
        and full_3_residual >= diagnostics["torso_residual_deg"] + 5.0
        and background_3_residual >= diagnostics["torso_residual_deg"] + 5.0
    )
    multiframe_target = project_rotation(full_3_rotation + background_3_rotation)
    multiframe_selected = (
        capped_rotation(base, multiframe_target, 60.0) if multiframe_trigger else base
    )

    coarse = boundary_rotation(full_1p1)
    metric = boundary_rotation(full_1p1, "metric_full_full")
    metric_row = full_1p1["candidates"]["metric_full_full"]
    coarse_residual = rotation_angle_deg(coarse, fixed)
    camera_metric_agreement = rotation_angle_deg(coarse, metric)
    explicit_consensus_trigger = bool(
        not accepted_v24(diagnostics)
        and diagnostics["torso_residual_deg"] < 10.0
        and float(full_1p1["rotation_consensus"]["spread_deg"]) <= 2.0
        and 30.0 <= coarse_residual <= 100.0
        and camera_metric_agreement <= 10.0
        and float(metric_row["fit_residual_median_m"]) <= 0.60
        and float(metric_row["robust_inlier_ratio"]) >= 0.50
        and int(metric_row["correspondence_count"]) >= 100
    )
    explicit_selected = (
        capped_rotation(base, coarse, 60.0) if explicit_consensus_trigger else base
    )

    one_frame_selected = explicit_selected if explicit_consensus_trigger else background_selected
    three_frame_selected = (
        explicit_selected if explicit_consensus_trigger else multiframe_selected
    )
    diagnostics.update(
        {
            "v24_accepted": accepted_v24(diagnostics),
            "background_1p1_residual_deg": background_residual,
            "background_1p1_spread_deg": float(
                background_1p1["rotation_consensus"]["spread_deg"]
            ),
            "trigger_background_1p1_fallback": background_trigger,
            "full_3p3_residual_deg": full_3_residual,
            "background_3p3_residual_deg": background_3_residual,
            "full_background_3p3_agreement_deg": multiframe_agreement,
            "trigger_multiframe_dual_3p3_fallback": multiframe_trigger,
            "camera_metric_rotation_agreement_deg": camera_metric_agreement,
            "metric_fit_residual_median_m": float(metric_row["fit_residual_median_m"]),
            "metric_epipolar_median_px": float(metric_row["epipolar_median_px"]),
            "metric_robust_inlier_ratio": float(metric_row["robust_inlier_ratio"]),
            "metric_correspondence_count": int(metric_row["correspondence_count"]),
            "trigger_low_torso_explicit_consensus": explicit_consensus_trigger,
        }
    )
    return {
        "v24": base,
        "background_1p1_fallback": background_selected,
        "multiframe_dual_3p3_fallback": multiframe_selected,
        "low_torso_explicit_consensus": explicit_selected,
        "v25_1p1_rotation": one_frame_selected,
        "v25_3p3_rotation": three_frame_selected,
    }, diagnostics


def run_case(
    name: str,
    v22: dict,
    v15: dict,
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
    clouds = {}
    for scale_name in ("absolute", "identity10"):
        scene_scale = [
            float(v22["scene_scale_sets"][scale_name]["old"]),
            float(v22["scene_scale_sets"][scale_name]["new"]),
        ]
        rng = np.random.default_rng(int(args.seed) + 1009 * index)
        clouds[scale_name] = [
            sample_cloud(
                raw["depth"][frame] * scene_scale[frame],
                intrinsics[frame],
                (old_pose, new_pose)[frame],
                masks[frame],
                raw["confidence"][frame],
                float(args.raw_confidence_threshold),
                int(args.point_samples),
                rng,
            )
            for frame in range(2)
        ]

    rotations, diagnostics = rotation_candidates(v22, v15)
    variants = {}
    for rotation_name, boundary_rotation in rotations.items():
        camera_rotation = boundary_rotation @ new_pose[:3, :3]
        camera_pose = camera_pose_from_human(camera_rotation, old_anchor_world, new_root)
        transform = boundary_from_camera_pose(camera_pose, new_pose)
        camera = evaluate(transform, new_pose, target_pose)
        human = human_metrics(
            transform,
            old_root,
            new_root,
            old_pose,
            new_pose,
            gt_old_world,
            gt_new_world,
        )
        scene_by_scale = {
            scale_name: scene_alignment_metrics(
                transform, values[1], values[0]
            )
            for scale_name, values in clouds.items()
        }
        selected_scale = min(
            scene_by_scale,
            key=lambda key: scene_by_scale[key]["trimmed_mean_m"],
        )
        margin_scale = (
            "identity10"
            if scene_by_scale["identity10"]["trimmed_mean_m"] + 0.10
            < scene_by_scale["absolute"]["trimmed_mean_m"]
            else "absolute"
        )
        for scale_policy, scale_name in (
            ("absolute", "absolute"),
            ("scene_consensus", selected_scale),
            ("scene_margin010", margin_scale),
        ):
            variants[f"{rotation_name}_{scale_policy}"] = {
                "transform": transform.astype(float).tolist(),
                "camera": camera,
                "human": human,
                "scene": scene_by_scale[scale_name],
                "scene_scale_rule": scale_name,
                "scene_scale_candidates": scene_by_scale,
            }
    return {
        "case_name": name,
        "source": v22["source"],
        "diagnostics": diagnostics,
        "variants": variants,
    }


def aggregate(rows: list[dict], variant: str, baseline: str) -> dict:
    values = [row["variants"][variant] for row in rows]
    base = [row["variants"][baseline] for row in rows]
    camera = np.asarray([row["camera"]["translation_m"] for row in values])
    rotation = np.asarray([row["camera"]["rotation_deg"] for row in values])
    human = np.asarray([row["human"]["root_motion_error_m"] for row in values])
    scene = np.asarray([row["scene"]["trimmed_mean_m"] for row in values])
    base_camera = np.asarray([row["camera"]["translation_m"] for row in base])
    base_rotation = np.asarray([row["camera"]["rotation_deg"] for row in base])
    base_scene = np.asarray([row["scene"]["trimmed_mean_m"] for row in base])
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
        "scene_harmful_010m": float(np.mean(scene > base_scene + 0.10)),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    v22 = load_cases(args.v22_report)
    v15 = load_v15(args.v15_dir)
    v10 = load_cases(args.v10_report)
    streams = load_shards(args.stream_dir, "v18_stream_shard_*_of_*.json")
    names = sorted(set(v22) & set(v15) & set(v10) & set(streams))
    if len(names) != 180:
        raise RuntimeError(f"Expected 180 common cases, got {len(names)}")
    rows = []
    for index, name in enumerate(names):
        rows.append(
            run_case(name, v22[name], v15[name], v10[name], streams[name], args, index)
        )
        if (index + 1) % 20 == 0 or index + 1 == len(names):
            print(f">> V25 explicit consensus {index + 1}/{len(names)}", flush=True)

    variants = [
        f"{rotation}_{scale}"
        for rotation in ROTATION_VARIANTS
        for scale in ("absolute", "scene_consensus", "scene_margin010")
    ]
    baseline = "v24_absolute"
    report = {
        "experiment": "V25 explicit multi-cue Boundary consensus probe",
        "case_count": len(rows),
        "protocol": {
            "base": "V24 safe conditional VGGT rotation plus V22 metric translation",
            "one_frame_fallback": "background-only 1+1 or camera/correspondence explicit consensus",
            "three_frame_fallback": "full/background 3+3 rotation consensus, fixed two-frame delay",
            "scene_scale": "choose absolute or identity10 by 1+1 pointcloud self-consistency; conservative variant requires 0.10 m improvement",
            "learned_gate": False,
            "source_specific_rule": False,
            "gt_runtime_information": False,
            "translation_resolved_after_every_rotation": True,
        },
        "trigger_counts": {
            "background_1p1": int(
                sum(row["diagnostics"]["trigger_background_1p1_fallback"] for row in rows)
            ),
            "multiframe_dual_3p3": int(
                sum(
                    row["diagnostics"]["trigger_multiframe_dual_3p3_fallback"]
                    for row in rows
                )
            ),
            "low_torso_explicit_consensus": int(
                sum(
                    row["diagnostics"]["trigger_low_torso_explicit_consensus"]
                    for row in rows
                )
            ),
        },
        "overall": {variant: aggregate(rows, variant, baseline) for variant in variants},
        "by_source": {
            source: {
                variant: aggregate(
                    [row for row in rows if row["source"] == source], variant, baseline
                )
                for variant in variants
            }
            for source in sorted({row["source"] for row in rows})
        },
        "catastrophic_cases": {
            variant: [
                row["case_name"]
                for row in rows
                if (
                    row["variants"][variant]["camera"]["translation_m"] > 2.0
                    or row["variants"][variant]["camera"]["rotation_deg"] > 45.0
                    or row["variants"][variant]["human"]["root_motion_error_m"] > 0.50
                    or row["variants"][variant]["scene"]["trimmed_mean_m"] > 1.0
                )
            ]
            for variant in variants
        },
        "cases": rows,
    }
    output = args.output_dir / "v25_explicit_consensus_bridge_probe.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    selected = {
        key: report["overall"][key]
        for key in (
            "v24_absolute",
            "v24_scene_consensus",
            "v24_scene_margin010",
            "v25_1p1_rotation_absolute",
            "v25_1p1_rotation_scene_consensus",
            "v25_1p1_rotation_scene_margin010",
            "v25_3p3_rotation_scene_consensus",
        )
    }
    print(json.dumps({"triggers": report["trigger_counts"], "selected": selected}, indent=2))
    print(f">> wrote {output}")


if __name__ == "__main__":
    main()
