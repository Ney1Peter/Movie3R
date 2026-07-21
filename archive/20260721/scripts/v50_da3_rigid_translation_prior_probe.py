#!/usr/bin/env python3
"""Use DA3 only as a translation prior while preserving raw Human3R geometry."""

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
from v18_da3_metric_depth_probe import evaluate  # noqa: E402
from v19_da3_depth_correction_ablation import load_raw_pair  # noqa: E402
from v19_da3_explicit_geometry_correction_probe import (  # noqa: E402
    distribution,
    human_metrics,
    scene_alignment_metrics,
)
from v20_shot_scale_consistency_probe import calibrated_targets, scale_pose  # noqa: E402
from v22_explicit_metric_bridge_selection import load_cases, load_shards  # noqa: E402
from v46_contact_preserving_metric_bridge_probe import build_clouds, load_body_pair  # noqa: E402


WEIGHTS = (0.25, 0.50, 0.75, 1.00)
CAPS_M = (0.10, 0.20, 0.30, 0.50)
MODES = ("vector", "old_view", "new_view")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v48_report",
        type=Path,
        default=(
            ROOT
            / "output/v48_component_necessity_ablation"
            / "v48_component_necessity_ablation.json"
        ),
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
        default=ROOT / "output/v50_da3_rigid_translation_prior",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--point_samples", type=int, default=4000)
    parser.add_argument("--raw_confidence_threshold", type=float, default=1.0)
    parser.add_argument("--mask_dilate", type=int, default=11)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args()


def normalize(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-8)


def camera_to_boundary(camera_pose: np.ndarray, local_pose: np.ndarray) -> np.ndarray:
    return (camera_pose @ np.linalg.inv(local_pose)).astype(np.float32)


def mapped_da3_camera_pose(
    da3_transform: np.ndarray,
    raw_old_pose: np.ndarray,
    raw_new_pose: np.ndarray,
    old_scale: float,
    new_scale: float,
) -> np.ndarray:
    metric_old_pose = scale_pose(raw_old_pose, float(old_scale))
    metric_new_pose = scale_pose(raw_new_pose, float(new_scale))
    camera_metric_old_world = da3_transform @ metric_new_pose
    camera_raw_old_world = camera_metric_old_world.astype(np.float32).copy()
    camera_raw_old_world[:3, 3] -= (
        metric_old_pose[:3, 3] - raw_old_pose[:3, 3]
    )
    return camera_raw_old_world


def corrected_camera_pose(
    raw_camera: np.ndarray,
    mapped_camera: np.ndarray,
    old_pose: np.ndarray,
    mode: str,
    weight: float,
) -> np.ndarray:
    output = raw_camera.astype(np.float32).copy()
    delta = mapped_camera[:3, 3] - raw_camera[:3, 3]
    if mode == "vector":
        correction = delta
    elif mode == "old_view":
        axis = normalize(old_pose[:3, 2])
        correction = float(np.dot(delta, axis)) * axis
    elif mode == "new_view":
        axis = normalize(raw_camera[:3, 2])
        correction = float(np.dot(delta, axis)) * axis
    else:
        raise ValueError(mode)
    output[:3, 3] += float(weight) * correction
    return output


def capped_camera_pose(
    raw_camera: np.ndarray,
    mapped_camera: np.ndarray,
    old_pose: np.ndarray,
    mode: str,
    cap_m: float,
) -> np.ndarray:
    full = corrected_camera_pose(raw_camera, mapped_camera, old_pose, mode, 1.0)
    correction = full[:3, 3] - raw_camera[:3, 3]
    magnitude = float(np.linalg.norm(correction))
    weight = min(1.0, float(cap_m) / max(magnitude, 1e-8))
    output = raw_camera.astype(np.float32).copy()
    output[:3, 3] += weight * correction
    return output


def evaluate_transform(
    transform: np.ndarray,
    poses: list[np.ndarray],
    roots: list[np.ndarray],
    target_pose: np.ndarray,
    gt_old_world: np.ndarray,
    gt_new_world: np.ndarray,
    clouds: list[np.ndarray],
    correction_m: float,
) -> dict:
    return {
        "transform": transform.astype(float).tolist(),
        "camera": evaluate(transform, poses[1], target_pose),
        "human": human_metrics(
            transform,
            roots[0],
            roots[1],
            poses[0],
            poses[1],
            gt_old_world,
            gt_new_world,
        ),
        "scene": scene_alignment_metrics(transform, clouds[1], clouds[0]),
        "correction_m": float(correction_m),
        "integrity": {
            "foot_ground_distortion_m": 0.0,
            "human_reprojection_shift_px": 0.0,
            "rigid_local_geometry": True,
        },
    }


def run_case(
    v48: dict,
    v10: dict,
    stream_row: dict,
    layer: SMPL_Layer,
    args: argparse.Namespace,
    index: int,
) -> dict:
    local = Path(v10["paths"]["human3r_local_reset"])
    body = load_body_pair(local, layer, torch.device(args.device))
    with np.load(stream_row["cache_path"]) as stream:
        poses = [
            np.asarray(stream["old_pose"][-1], dtype=np.float32),
            np.asarray(stream["new_pose"], dtype=np.float32),
        ]
        intrinsics = np.stack(
            [stream["old_intrinsics"][-1], stream["new_intrinsics"]]
        ).astype(np.float32)
        target, gt_old, gt_new = calibrated_targets(stream, poses[0])

    raw = load_raw_pair(local)
    kernel = np.ones((int(args.mask_dilate), int(args.mask_dilate)), dtype=np.uint8)
    masks = [
        cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) for mask in raw["mask"]
    ]
    clouds = build_clouds(
        raw,
        intrinsics,
        poses,
        [1.0, 1.0],
        masks,
        args,
        int(args.seed) + index,
    )
    roots = [body["roots"][0], body["roots"][1]]

    raw_transform = np.asarray(
        v48["variants"]["v32_raw"]["transform"], dtype=np.float32
    )
    da3_transform = np.asarray(
        v48["variants"]["v32_da3"]["transform"], dtype=np.float32
    )
    raw_camera = raw_transform @ poses[1]
    mapped_camera = mapped_da3_camera_pose(
        da3_transform,
        poses[0],
        poses[1],
        float(v48["root_scales"]["old"]),
        float(v48["root_scales"]["new"]),
    )
    variants = {
        "v47_raw": evaluate_transform(
            raw_transform,
            poses,
            roots,
            target,
            gt_old,
            gt_new,
            clouds,
            0.0,
        )
    }
    for mode in MODES:
        for weight in WEIGHTS:
            camera = corrected_camera_pose(raw_camera, mapped_camera, poses[0], mode, weight)
            transform = camera_to_boundary(camera, poses[1])
            variants[f"{mode}_{int(round(100 * weight)):03d}"] = evaluate_transform(
                transform,
                poses,
                roots,
                target,
                gt_old,
                gt_new,
                clouds,
                float(np.linalg.norm(camera[:3, 3] - raw_camera[:3, 3])),
            )
        for cap_m in CAPS_M:
            camera = capped_camera_pose(
                raw_camera, mapped_camera, poses[0], mode, cap_m
            )
            transform = camera_to_boundary(camera, poses[1])
            variants[f"{mode}_cap{int(round(100 * cap_m)):03d}"] = evaluate_transform(
                transform,
                poses,
                roots,
                target,
                gt_old,
                gt_new,
                clouds,
                float(np.linalg.norm(camera[:3, 3] - raw_camera[:3, 3])),
            )
    return {
        "case_name": v48["case_name"],
        "source": v48["source"],
        "v32_branch": v48["v32_branch"],
        "root_scale_ratio": float(v48["root_scales"]["new"])
        / max(float(v48["root_scales"]["old"]), 1e-8),
        "mapped_prior_delta_m": float(
            np.linalg.norm(mapped_camera[:3, 3] - raw_camera[:3, 3])
        ),
        "variants": variants,
    }


def summarize(rows: list[dict], variant: str) -> dict:
    values = [row["variants"][variant] for row in rows]
    return {
        "camera_translation_m": distribution(
            [value["camera"]["translation_m"] for value in values]
        ),
        "camera_viewing_direction_m": distribution(
            [value["camera"]["viewing_direction_m"] for value in values]
        ),
        "camera_transverse_m": distribution(
            [value["camera"]["transverse_m"] for value in values]
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
        "correction_m": distribution([value["correction_m"] for value in values]),
        "camera_catastrophic_rate": float(
            np.mean(
                [
                    value["camera"]["translation_m"] > 2.0
                    or value["camera"]["rotation_deg"] > 45.0
                    for value in values
                ]
            )
        ),
        "joint_failure_rate": float(
            np.mean(
                [
                    value["camera"]["translation_m"] > 2.0
                    or value["camera"]["rotation_deg"] > 45.0
                    or value["human"]["root_motion_error_m"] > 0.50
                    or value["scene"]["trimmed_mean_m"] > 1.0
                    for value in values
                ]
            )
        ),
    }


def paired(rows: list[dict], variant: str) -> dict:
    base = np.asarray(
        [row["variants"]["v47_raw"]["camera"]["translation_m"] for row in rows],
        dtype=np.float64,
    )
    value = np.asarray(
        [row["variants"][variant]["camera"]["translation_m"] for row in rows],
        dtype=np.float64,
    )
    delta = value - base
    return {
        "improved_count": int(np.sum(delta < -0.05)),
        "harmed_count": int(np.sum(delta > 0.05)),
        "mean_delta_m": float(np.mean(delta)),
        "p95_delta_m": float(np.quantile(delta, 0.95)),
    }


def markdown(report: dict) -> str:
    lines = [
        "# V50 DA3 Rigid Translation Prior Probe",
        "",
        "DA3 is converted to a camera-center translation prior after V32 rotation. Raw Human3R camera, pointmap, and SMPL-X geometry are never scaled.",
        "",
        "| Variant | T mean/P95 | View/Transverse | Human motion | Scene | Correction | Camera cat | Joint failure | I/H >5cm |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in report["overall"].items():
        pair = report["paired_vs_v47"].get(name, {})
        lines.append(
            "| {name} | {tm:.3f}/{tp:.3f} | {view:.3f}/{trans:.3f} | {human:.3f} | "
            "{scene:.3f} | {corr:.3f} | {cat:.1%} | {joint:.1%} | {improved}/{harmed} |".format(
                name=name,
                tm=value["camera_translation_m"]["mean"],
                tp=value["camera_translation_m"]["p95"],
                view=value["camera_viewing_direction_m"]["mean"],
                trans=value["camera_transverse_m"]["mean"],
                human=value["human_motion_error_m"]["mean"],
                scene=value["scene_trimmed_mean_m"]["mean"],
                corr=value["correction_m"]["mean"],
                cat=value["camera_catastrophic_rate"],
                joint=value["joint_failure_rate"],
                improved=pair.get("improved_count", 0),
                harmed=pair.get("harmed_count", 0),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    v48_payload = json.loads(args.v48_report.read_text(encoding="utf-8"))
    v48 = {row["case_name"]: row for row in v48_payload["cases"]}
    v10 = load_cases(args.v10_report)
    streams = load_shards(args.stream_dir, "v18_stream_shard_*_of_*.json")
    names = sorted(set(v48) & set(v10) & set(streams))
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
        rows.append(run_case(v48[name], v10[name], streams[name], layer, args, index))
        if (index + 1) % 20 == 0:
            print(f"V50 DA3 rigid prior {index + 1}/{len(names)}", flush=True)

    variants = ["v47_raw"] + [
        f"{mode}_{int(round(100 * weight)):03d}"
        for mode in MODES
        for weight in WEIGHTS
    ] + [
        f"{mode}_cap{int(round(100 * cap_m)):03d}"
        for mode in MODES
        for cap_m in CAPS_M
    ]
    report = {
        "experiment": "V50 DA3 rigid translation prior probe",
        "case_count": len(rows),
        "protocol": {
            "rotation": "frozen V32",
            "post_cut_frames": 1,
            "da3_role": "translation prior only",
            "local_geometry_scaling": False,
            "shot_transform": "one rigid SE3",
            "runtime_gt": False,
        },
        "overall": {name: summarize(rows, name) for name in variants},
        "by_source": {
            source: {
                name: summarize([row for row in rows if row["source"] == source], name)
                for name in variants
            }
            for source in sorted({row["source"] for row in rows})
        },
        "paired_vs_v47": {
            name: paired(rows, name) for name in variants if name != "v47_raw"
        },
        "cases": rows,
    }
    json_path = args.output_dir / "v50_da3_rigid_translation_prior_probe.json"
    md_path = args.output_dir / "v50_da3_rigid_translation_prior_probe.md"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    md_path.write_text(markdown(report), encoding="utf-8")
    print(markdown(report))
    print(f">> wrote {json_path}")
    print(f">> wrote {md_path}")


if __name__ == "__main__":
    main()
