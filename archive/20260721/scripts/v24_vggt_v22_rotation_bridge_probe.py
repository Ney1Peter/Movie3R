#!/usr/bin/env python3
"""Combine cached VGGT rotation with the deployable V22 metric translation bridge."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


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
from v22_explicit_metric_bridge_selection import load_cases, load_shards  # noqa: E402


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
DEFAULT_OUTPUT = REPO_ROOT / "output" / "v24_vggt_v22_rotation_bridge"
SELECTED = "safe_gravity_absolute_scene_scale"
VARIANTS = (
    "v22",
    "vggt_full_1p1",
    "vggt_cap5_all",
    "torso30_vggt_cap10",
    "torso30_vggt_cap30",
    "torso10_consensus_vggt_cap60",
    "tiered_torso30_cap20_else_consensus_cap60",
    "tiered_torso30_cap30_else_consensus_cap60",
    "tiered_extension5_torso30_cap20_else_consensus_cap60",
    "tiered_extension5_torso30_cap25_else_consensus_cap60",
    "tiered_extension5_cap25_low_texture_conflict_cap45",
    "safe_tiered_extension_vggt",
    "oracle_best_v22_vggt_rotation",
    "gt_rotation",
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


def load_v15(path: Path) -> dict[str, dict]:
    rows = []
    for shard in sorted(glob.glob(str(path / "v15_candidates_shard_*_of_*.json"))):
        rows.extend(json.loads(Path(shard).read_text(encoding="utf-8"))["cases"])
    output = {row["case_name"]: row for row in rows}
    if len(output) != 180:
        raise RuntimeError(f"Expected 180 V15 cases, got {len(output)}")
    return output


def relative_rotvec(target: np.ndarray, base: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix((target @ base.T).astype(np.float64)).as_rotvec()


def rotation_angle_deg(target: np.ndarray, base: np.ndarray) -> float:
    return float(np.degrees(np.linalg.norm(relative_rotvec(target, base))))


def capped_rotation(base: np.ndarray, target: np.ndarray, bound_deg: float) -> np.ndarray:
    residual = relative_rotvec(target, base)
    magnitude = float(np.linalg.norm(residual))
    bound = float(np.radians(bound_deg))
    if magnitude > bound > 0.0:
        residual *= bound / magnitude
    if bound_deg <= 0.0:
        residual[:] = 0.0
    return (Rotation.from_rotvec(residual).as_matrix() @ base).astype(np.float32)


def candidate_rotations(v22: dict, v15: dict) -> tuple[dict[str, np.ndarray], dict]:
    fixed = np.asarray(v22["variants"]["fixed_explicit"]["transform"], dtype=np.float32)[:3, :3]
    torso = np.asarray(v22["variants"][SELECTED]["transform"], dtype=np.float32)[:3, :3]
    vggt = np.asarray(
        v15["windows"]["full_rgb_1p1"]["candidates"]["coarse"]["transform"],
        dtype=np.float32,
    )[:3, :3]
    gt = np.asarray(v15["baselines"]["boundary_oracle"]["transform"], dtype=np.float32)[:3, :3]
    torso_residual = relative_rotvec(torso, fixed)
    vggt_residual = relative_rotvec(vggt, fixed)
    torso_residual_deg = float(np.degrees(np.linalg.norm(torso_residual)))
    residual_cosine = float(
        np.dot(torso_residual, vggt_residual)
        / max(np.linalg.norm(torso_residual) * np.linalg.norm(vggt_residual), 1e-9)
    )
    internal_spread = float(
        v15["windows"]["full_rgb_1p1"]["rotation_consensus"]["spread_deg"]
    )
    trigger30 = torso_residual_deg >= 30.0
    consensus = (
        torso_residual_deg >= 10.0
        and residual_cosine >= 0.0
        and internal_spread <= 5.0
    )
    tiered_cap20 = (
        capped_rotation(torso, vggt, 20.0)
        if trigger30
        else (capped_rotation(torso, vggt, 60.0) if consensus else torso)
    )
    tiered_cap30 = (
        capped_rotation(torso, vggt, 30.0)
        if trigger30
        else (capped_rotation(torso, vggt, 60.0) if consensus else torso)
    )
    extends_torso = (
        float(np.degrees(np.linalg.norm(vggt_residual)))
        >= torso_residual_deg + 5.0
    )
    tiered_extension5 = (
        capped_rotation(torso, vggt, 20.0)
        if trigger30 and extends_torso
        else (
            capped_rotation(torso, vggt, 60.0)
            if consensus and extends_torso
            else torso
        )
    )
    tiered_extension5_cap25 = (
        capped_rotation(torso, vggt, 25.0)
        if trigger30 and extends_torso
        else (
            capped_rotation(torso, vggt, 60.0)
            if consensus and extends_torso
            else torso
        )
    )
    low_texture_conflict = (
        torso_residual_deg >= 10.0
        and float(np.degrees(np.linalg.norm(vggt_residual)))
        >= torso_residual_deg + 10.0
        and internal_spread <= 5.0
        and residual_cosine < 0.0
        and float(v15["texture_score"]) < 0.05
    )
    tiered_low_texture_conflict = (
        capped_rotation(tiered_extension5_cap25, vggt, 45.0)
        if low_texture_conflict
        else tiered_extension5_cap25
    )
    vggt_residual_deg = float(np.degrees(np.linalg.norm(vggt_residual)))
    safe_large_residual = (
        trigger30
        and extends_torso
        and vggt_residual_deg <= 100.0
        and internal_spread <= 15.0
    )
    safe_consensus = (
        consensus
        and extends_torso
        and vggt_residual_deg <= 100.0
    )
    safe_base = (
        capped_rotation(torso, vggt, 25.0)
        if safe_large_residual
        else (capped_rotation(torso, vggt, 60.0) if safe_consensus else torso)
    )
    safe_low_texture_conflict = low_texture_conflict and vggt_residual_deg <= 100.0
    safe_selected = (
        capped_rotation(safe_base, vggt, 45.0)
        if safe_low_texture_conflict
        else safe_base
    )
    oracle = vggt if rotation_angle_deg(vggt, gt) < rotation_angle_deg(torso, gt) else torso
    return {
        "v22": torso,
        "vggt_full_1p1": vggt,
        "vggt_cap5_all": capped_rotation(torso, vggt, 5.0),
        "torso30_vggt_cap10": capped_rotation(torso, vggt, 10.0) if trigger30 else torso,
        "torso30_vggt_cap30": capped_rotation(torso, vggt, 30.0) if trigger30 else torso,
        "torso10_consensus_vggt_cap60": (
            capped_rotation(torso, vggt, 60.0) if consensus else torso
        ),
        "tiered_torso30_cap20_else_consensus_cap60": tiered_cap20,
        "tiered_torso30_cap30_else_consensus_cap60": tiered_cap30,
        "tiered_extension5_torso30_cap20_else_consensus_cap60": tiered_extension5,
        "tiered_extension5_torso30_cap25_else_consensus_cap60": tiered_extension5_cap25,
        "tiered_extension5_cap25_low_texture_conflict_cap45": tiered_low_texture_conflict,
        "safe_tiered_extension_vggt": safe_selected,
        "oracle_best_v22_vggt_rotation": oracle,
        "gt_rotation": gt,
    }, {
        "torso_residual_deg": torso_residual_deg,
        "vggt_residual_deg": float(np.degrees(np.linalg.norm(vggt_residual))),
        "residual_direction_cosine": residual_cosine,
        "vggt_internal_spread_deg": internal_spread,
        "trigger_torso30": trigger30,
        "trigger_consensus": consensus,
        "vggt_extends_torso_by_5deg": extends_torso,
        "trigger_low_texture_conflict": low_texture_conflict,
        "trigger_safe_large_residual": safe_large_residual,
        "trigger_safe_consensus": safe_consensus,
        "trigger_safe_low_texture_conflict": safe_low_texture_conflict,
    }


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
        raw_poses = np.stack([stream["old_pose"][-1], stream["new_pose"]]).astype(np.float32)
        intrinsics = np.stack([stream["old_intrinsics"][-1], stream["new_intrinsics"]]).astype(
            np.float32
        )
        translations = np.stack([stream["old_transl"][-1], stream["new_transl"]]).astype(
            np.float32
        )
        root_scales = [float(v22["root_scales"]["old"]), float(v22["root_scales"]["new"])]
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

    rotations, diagnostics = candidate_rotations(v22, v15)
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
        "diagnostics": diagnostics,
        "variants": variants,
    }


def aggregate(rows: list[dict], variant: str) -> dict:
    values = [row["variants"][variant] for row in rows]
    baseline = [row["variants"]["v22"] for row in rows]
    camera = np.asarray([row["camera"]["translation_m"] for row in values])
    rotation = np.asarray([row["camera"]["rotation_deg"] for row in values])
    human = np.asarray([row["human"]["root_motion_error_m"] for row in values])
    scene = np.asarray([row["scene"]["trimmed_mean_m"] for row in values])
    base_camera = np.asarray([row["camera"]["translation_m"] for row in baseline])
    base_rotation = np.asarray([row["camera"]["rotation_deg"] for row in baseline])
    base_scene = np.asarray([row["scene"]["trimmed_mean_m"] for row in baseline])
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
        rows.append(run_case(name, v22[name], v15[name], v10[name], streams[name], args, index))
        if (index + 1) % 10 == 0 or index + 1 == len(names):
            print(f"V24 VGGT/V22 rotation bridge {index + 1}/{len(names)}", flush=True)
    report = {
        "experiment": "V24 cached VGGT rotation plus V22 explicit metric translation",
        "case_count": len(rows),
        "protocol": {
            "vggt": "frozen V15 full-RGB 1+1 camera rotation",
            "metric_scale": "unchanged V22 DA3 root/background scales",
            "translation": "unchanged V22 explicit human-root equation, re-solved after rotation",
            "post_cut_frames": 1,
            "learned_gate": False,
            "source_specific_rule": False,
            "torso30_trigger_rate": float(
                np.mean([row["diagnostics"]["trigger_torso30"] for row in rows])
            ),
            "consensus_trigger_rate": float(
                np.mean([row["diagnostics"]["trigger_consensus"] for row in rows])
            ),
        },
        "overall": {variant: aggregate(rows, variant) for variant in VARIANTS},
        "by_source": {
            source: {
                variant: aggregate([row for row in rows if row["source"] == source], variant)
                for variant in VARIANTS
            }
            for source in sorted({row["source"] for row in rows})
        },
        "cases": rows,
    }
    output = args.output_dir / "v24_vggt_v22_rotation_bridge.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(f">> wrote {output}")


if __name__ == "__main__":
    main()
