#!/usr/bin/env python3
"""V11.2 contact-preserving boundary-alignment probe."""

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

from dust3r.utils.smpl_layer import SMPL_Layer
from boundary_metric_depth_support import boundary_from_camera_pose, camera_pose_from_human, evaluate
from boundary_depth_correction_support import load_raw_pair
from boundary_geometry_support import (
    distribution,
    human_metrics,
    sample_cloud,
    scene_alignment_metrics,
    transform_points,
)
from boundary_shot_scale_support import calibrated_targets, scale_pose
from boundary_metric_selection_support import load_cases, load_shards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bridge_report",
        type=Path,
        default=ROOT / "output/v36_final_explicit_metric_bridge/v36_final_explicit_metric_bridge.json",
    )
    parser.add_argument(
        "--v10_report",
        type=Path,
        default=ROOT / "output/v10_candidate_selection/oracle_gt_4source/oracle_candidate_selection_metrics.json",
    )
    parser.add_argument(
        "--stream_dir",
        type=Path,
        default=ROOT / "output/v18_human_metric_translation/stream_cache",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ROOT / "output/v46_contact_preserving_metric_bridge",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--point_samples", type=int, default=4000)
    parser.add_argument("--raw_confidence_threshold", type=float, default=1.0)
    parser.add_argument("--mask_dilate", type=int, default=11)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args()


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1e-8)


def load_body_pair(local: Path, layer: SMPL_Layer, device: torch.device) -> dict:
    arrays = {key: [] for key in ("rotvec", "shape", "transl", "expression")}
    intrinsics = []
    for frame in (1, 2):
        with np.load(local / "smpl" / f"{frame:06d}.npz", allow_pickle=True) as smpl:
            for key in arrays:
                arrays[key].append(np.asarray(smpl[key][0], dtype=np.float32))
        with np.load(local / "camera" / f"{frame:06d}.npz") as camera:
            intrinsics.append(np.asarray(camera["intrinsics"], dtype=np.float32))
    with torch.no_grad():
        output = layer(
            torch.from_numpy(np.stack(arrays["rotvec"])).to(device),
            torch.from_numpy(np.stack(arrays["shape"])).to(device),
            torch.from_numpy(np.stack(arrays["transl"])).to(device),
            None,
            None,
            K=torch.from_numpy(np.stack(intrinsics)).to(device),
            expression=torch.from_numpy(np.stack(arrays["expression"])).to(device),
        )
    joints = output["smpl_j3d"].detach().float().cpu().numpy().astype(np.float32)
    names = layer.joint_names
    pelvis = joints[:, names.index("pelvis")]
    foot_indices = [
        names.index("left_big_toe"),
        names.index("left_small_toe"),
        names.index("left_heel"),
        names.index("right_big_toe"),
        names.index("right_small_toe"),
        names.index("right_heel"),
    ]
    feet = np.median(joints[:, foot_indices], axis=1).astype(np.float32)
    roots = np.stack(arrays["transl"]).astype(np.float32)
    return {"roots": roots, "pelvis": pelvis, "feet": feet, "joints": joints}


def projection_shift_px(
    intrinsics: np.ndarray,
    original: np.ndarray,
    modified: np.ndarray,
) -> float:
    original = np.asarray(original, dtype=np.float32)
    modified = np.asarray(modified, dtype=np.float32)
    valid = (
        np.isfinite(original).all(axis=1)
        & np.isfinite(modified).all(axis=1)
        & (original[:, 2] > 0.05)
        & (modified[:, 2] > 0.05)
    )
    if not np.any(valid):
        return float("nan")

    def project(points: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                intrinsics[0, 0] * points[:, 0] / points[:, 2] + intrinsics[0, 2],
                intrinsics[1, 1] * points[:, 1] / points[:, 2] + intrinsics[1, 2],
            ],
            axis=1,
        )

    delta = project(modified[valid]) - project(original[valid])
    return float(np.mean(np.linalg.norm(delta, axis=1)))


def contact_correction(
    root: np.ndarray,
    pelvis: np.ndarray,
    foot: np.ndarray,
    root_scale: float,
    scene_scale: float,
) -> dict:
    offset = foot - root
    scaled_human_foot = root * root_scale + offset
    scaled_scene_contact = foot * scene_scale
    error = scaled_human_foot - scaled_scene_contact
    down = normalize(foot - pelvis)
    signed = float(np.dot(error, down))
    correction = (-signed * down).astype(np.float32)
    return {
        "signed_penetration_proxy_m": signed,
        "absolute_contact_distortion_m": float(abs(signed)),
        "correction": correction,
        "correction_m": float(np.linalg.norm(correction)),
    }


def build_clouds(
    raw: dict,
    intrinsics: np.ndarray,
    poses: list[np.ndarray],
    scales: list[float],
    masks: list[np.ndarray],
    args: argparse.Namespace,
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [
        sample_cloud(
            raw["depth"][frame] * float(scales[frame]),
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


def solve_variant(
    boundary_rotation: np.ndarray,
    poses: list[np.ndarray],
    roots: list[np.ndarray],
    target_pose: np.ndarray,
    gt_old_world: np.ndarray,
    gt_new_world: np.ndarray,
    clouds: list[np.ndarray],
) -> dict:
    old_anchor = transform_points(poses[0], roots[0][None])[0]
    camera_rotation = boundary_rotation @ poses[1][:3, :3]
    camera_pose = camera_pose_from_human(camera_rotation, old_anchor, roots[1])
    transform = boundary_from_camera_pose(camera_pose, poses[1])
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
    }


def run_case(
    bridge: dict,
    v10: dict,
    stream_row: dict,
    layer: SMPL_Layer,
    args: argparse.Namespace,
    index: int,
) -> dict:
    local = Path(v10["paths"]["human3r_local_reset"])
    body = load_body_pair(local, layer, torch.device(args.device))
    with np.load(stream_row["cache_path"]) as stream:
        raw_poses = [
            np.asarray(stream["old_pose"][-1], dtype=np.float32),
            np.asarray(stream["new_pose"], dtype=np.float32),
        ]
        intrinsics = np.stack(
            [stream["old_intrinsics"][-1], stream["new_intrinsics"]]
        ).astype(np.float32)
        raw_target, raw_gt_old, raw_gt_new = calibrated_targets(stream, raw_poses[0])

        root_scales = [
            float(bridge["root_scales"]["old"]),
            float(bridge["root_scales"]["new"]),
        ]
        scene_scales = [
            float(bridge["scene_scale_sets"]["absolute"]["old"]),
            float(bridge["scene_scale_sets"]["absolute"]["new"]),
        ]
        metric_poses = [scale_pose(raw_poses[i], root_scales[i]) for i in range(2)]
        metric_target, metric_gt_old, metric_gt_new = calibrated_targets(stream, metric_poses[0])

    raw = load_raw_pair(local)
    kernel = np.ones((int(args.mask_dilate), int(args.mask_dilate)), dtype=np.uint8)
    masks = [cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) for mask in raw["mask"]]
    raw_clouds = build_clouds(
        raw, intrinsics, raw_poses, [1.0, 1.0], masks, args, int(args.seed) + index
    )
    metric_clouds = build_clouds(
        raw,
        intrinsics,
        metric_poses,
        scene_scales,
        masks,
        args,
        int(args.seed) + index,
    )

    boundary_rotation = np.asarray(
        bridge["variants"]["v32"]["transform"], dtype=np.float32
    )[:3, :3]
    raw_roots = [body["roots"][0], body["roots"][1]]
    raw_v32 = solve_variant(
        boundary_rotation,
        raw_poses,
        raw_roots,
        raw_target,
        raw_gt_old,
        raw_gt_new,
        raw_clouds,
    )

    contacts = [
        contact_correction(
            body["roots"][frame],
            body["pelvis"][frame],
            body["feet"][frame],
            root_scales[frame],
            scene_scales[frame],
        )
        for frame in range(2)
    ]
    metric_roots = [
        body["roots"][frame] * root_scales[frame] for frame in range(2)
    ]
    corrected_roots = [
        metric_roots[frame] + contacts[frame]["correction"] for frame in range(2)
    ]
    current_projection_shifts = []
    contact_projection_shifts = []
    for frame in range(2):
        raw_joints = body["joints"][frame]
        root = body["roots"][frame]
        current_joints = raw_joints - root + metric_roots[frame]
        corrected_joints = raw_joints - root + corrected_roots[frame]
        current_projection_shifts.append(
            projection_shift_px(intrinsics[frame], raw_joints, current_joints)
        )
        contact_projection_shifts.append(
            projection_shift_px(intrinsics[frame], raw_joints, corrected_joints)
        )
    contact_v32 = solve_variant(
        boundary_rotation,
        metric_poses,
        corrected_roots,
        metric_target,
        metric_gt_old,
        metric_gt_new,
        metric_clouds,
    )
    contact_v32["contact"] = {
        "old": {key: value for key, value in contacts[0].items() if key != "correction"},
        "new": {key: value for key, value in contacts[1].items() if key != "correction"},
        "mean_correction_m": float(np.mean([c["correction_m"] for c in contacts])),
        "max_correction_m": float(np.max([c["correction_m"] for c in contacts])),
        "post_correction_contact_proxy_m": 0.0,
        "root_corrections_camera": {
            "old": contacts[0]["correction"].astype(float).tolist(),
            "new": contacts[1]["correction"].astype(float).tolist(),
        },
    }
    contact_v32["integrity"] = {
        "human_reprojection_shift_px": float(np.nanmean(contact_projection_shifts)),
        "rigid_local_geometry": False,
    }
    current = bridge["variants"]["v32"]
    current_contact = {
        "old": {key: value for key, value in contacts[0].items() if key != "correction"},
        "new": {key: value for key, value in contacts[1].items() if key != "correction"},
        "mean_absolute_contact_distortion_m": float(
            np.mean([c["absolute_contact_distortion_m"] for c in contacts])
        ),
    }
    return {
        "case_name": bridge["case_name"],
        "source": bridge["source"],
        "root_scales": bridge["root_scales"],
        "scene_scales": bridge["scene_scale_sets"]["absolute"],
        "current_v45": {
            **current,
            "contact": current_contact,
            "integrity": {
                "human_reprojection_shift_px": float(np.nanmean(current_projection_shifts)),
                "rigid_local_geometry": False,
            },
        },
        "raw_scale_v32": {
            **raw_v32,
            "contact": {
                "mean_absolute_contact_distortion_m": 0.0,
                "post_correction_contact_proxy_m": 0.0,
            },
            "integrity": {
                "human_reprojection_shift_px": 0.0,
                "rigid_local_geometry": True,
            },
        },
        "contact_v32": contact_v32,
    }


def summarize(rows: list[dict], variant: str) -> dict:
    values = [row[variant] for row in rows]

    def contact_value(value: dict) -> float:
        contact = value["contact"]
        if "post_correction_contact_proxy_m" in contact:
            return float(contact["post_correction_contact_proxy_m"])
        if "mean_absolute_contact_distortion_m" in contact:
            return float(contact["mean_absolute_contact_distortion_m"])
        return float(
            0.5
            * (
                contact["old"]["absolute_contact_distortion_m"]
                + contact["new"]["absolute_contact_distortion_m"]
            )
        )

    return {
        "camera_translation_m": distribution([v["camera"]["translation_m"] for v in values]),
        "camera_rotation_deg": distribution([v["camera"]["rotation_deg"] for v in values]),
        "human_motion_error_m": distribution([v["human"]["root_motion_error_m"] for v in values]),
        "scene_trimmed_mean_m": distribution([v["scene"]["trimmed_mean_m"] for v in values]),
        "contact_distortion_m": distribution([contact_value(v) for v in values]),
        "contact_correction_m": distribution([
            v["contact"].get("mean_correction_m", 0.0) for v in values
        ]),
        "human_reprojection_shift_px": distribution([
            v.get("integrity", {}).get("human_reprojection_shift_px", 0.0)
            for v in values
        ]),
        "rigid_local_geometry_rate": float(np.mean([
            bool(v.get("integrity", {}).get("rigid_local_geometry", False))
            for v in values
        ])),
        "combined_catastrophic_rate": float(np.mean([
            v["camera"]["translation_m"] > 2.0
            or v["camera"]["rotation_deg"] > 45.0
            or v["human"]["root_motion_error_m"] > 0.50
            or v["scene"]["trimmed_mean_m"] > 1.0
            or contact_value(v) > 0.10
            or v.get("integrity", {}).get("human_reprojection_shift_px", 0.0) > 25.0
            for v in values
        ])),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bridges = load_cases(args.bridge_report)
    v10 = load_cases(args.v10_report)
    streams = load_shards(args.stream_dir, "v18_stream_shard_*_of_*.json")
    names = sorted(set(bridges) & set(v10) & set(streams))
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
        rows.append(run_case(bridges[name], v10[name], streams[name], layer, args, index))
        if (index + 1) % 20 == 0:
            print(f"V11.2 contact probe {index + 1}/{len(names)}", flush=True)
    report = {
        "experiment": "V11.2 contact-preserving metric bridge probe",
        "case_count": len(rows),
        "protocol": {
            "rotation": "frozen conditional wide rotation",
            "contact_reference": "preserve raw Human3R foot/background relation",
            "contact_correction": "bounded direction is pelvis-to-feet; translation re-solved explicitly",
            "gt_runtime_information": False,
        },
        "overall": {
            variant: summarize(rows, variant)
            for variant in ("current_v45", "raw_scale_v32", "contact_v32")
        },
        "by_source": {
            source: {
                variant: summarize([row for row in rows if row["source"] == source], variant)
                for variant in ("current_v45", "raw_scale_v32", "contact_v32")
            }
            for source in sorted({row["source"] for row in rows})
        },
        "cases": rows,
    }
    output = args.output_dir / "v46_contact_preserving_metric_bridge_probe.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({"overall": report["overall"], "by_source": report["by_source"]}, indent=2))
    print(f">> wrote {output}")


if __name__ == "__main__":
    main()
