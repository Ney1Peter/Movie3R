#!/usr/bin/env python3
"""V11.3 component ablation for Fixed, torso, wide-baseline, and DA3 cues."""

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
from boundary_metric_depth_support import evaluate  # noqa: E402
from boundary_depth_correction_support import load_raw_pair  # noqa: E402
from boundary_geometry_support import (  # noqa: E402
    distribution,
    human_metrics,
    scene_alignment_metrics,
)
from boundary_shot_scale_support import calibrated_targets, scale_pose  # noqa: E402
from boundary_metric_selection_support import load_cases, load_shards  # noqa: E402
from v11_2_contact_preserving_probe import (  # noqa: E402
    build_clouds,
    contact_correction,
    load_body_pair,
    projection_shift_px,
    solve_variant,
)


ROTATION_KEYS = ("fixed", "torso", "v22", "vggt", "v32")
SCALE_KEYS = ("raw", "da3")


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
        default=(
            ROOT
            / "output/v10_candidate_selection/oracle_gt_4source"
            / "oracle_candidate_selection_metrics.json"
        ),
    )
    parser.add_argument(
        "--wide_candidate_dir",
        type=Path,
        default=ROOT / "output/v15_wide_baseline_boundary_bridge/candidate_cache",
    )
    parser.add_argument(
        "--torso_candidate_dir",
        type=Path,
        default=ROOT / "output/v16_human_aware_rotation_residual/candidate_cache",
    )
    parser.add_argument(
        "--stream_dir",
        type=Path,
        default=ROOT / "output/v18_human_metric_translation/stream_cache",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ROOT / "output/v48_component_necessity_ablation",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--point_samples", type=int, default=4000)
    parser.add_argument("--raw_confidence_threshold", type=float, default=1.0)
    parser.add_argument("--mask_dilate", type=int, default=11)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args()


def load_candidate_shards(directory: Path, pattern: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for path in sorted(directory.glob(pattern)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload["cases"]:
            rows[str(row["case_name"])] = row
    return rows


def rotation_sources(bridge: dict, v15: dict, v16: dict) -> dict[str, np.ndarray]:
    return {
        "fixed": np.asarray(
            v16["baselines"]["fixed_explicit"]["transform"], dtype=np.float32
        )[:3, :3],
        "torso": np.asarray(
            v16["fixed_candidates"]["fixed_torso_motion_1f_resolve_t"]["transform"],
            dtype=np.float32,
        )[:3, :3],
        "v22": np.asarray(bridge["variants"]["v22"]["transform"], dtype=np.float32)[
            :3, :3
        ],
        "vggt": np.asarray(
            v15["windows"]["full_rgb_1p1"]["candidates"]["coarse"]["transform"],
            dtype=np.float32,
        )[:3, :3],
        "v32": np.asarray(bridge["variants"]["v32"]["transform"], dtype=np.float32)[
            :3, :3
        ],
    }


def integrity_metrics(
    body: dict,
    intrinsics: np.ndarray,
    root_scales: list[float],
    scene_scales: list[float],
) -> dict:
    contact_values = []
    projection_values = []
    for frame in range(2):
        contact = contact_correction(
            body["roots"][frame],
            body["pelvis"][frame],
            body["feet"][frame],
            root_scales[frame],
            scene_scales[frame],
        )
        contact_values.append(float(contact["absolute_contact_distortion_m"]))
        raw_joints = body["joints"][frame]
        root = body["roots"][frame]
        modified = raw_joints - root + root * float(root_scales[frame])
        projection_values.append(
            projection_shift_px(intrinsics[frame], raw_joints, modified)
        )
    return {
        "foot_ground_distortion_m": float(np.mean(contact_values)),
        "human_reprojection_shift_px": float(np.nanmean(projection_values)),
        "rigid_local_geometry": bool(
            np.allclose(root_scales, [1.0, 1.0])
            and np.allclose(scene_scales, [1.0, 1.0])
        ),
    }


def actual_fixed_variant(
    transform: np.ndarray,
    poses: list[np.ndarray],
    roots: list[np.ndarray],
    target_pose: np.ndarray,
    gt_old_world: np.ndarray,
    gt_new_world: np.ndarray,
    clouds: list[np.ndarray],
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
        "integrity": {
            "foot_ground_distortion_m": 0.0,
            "human_reprojection_shift_px": 0.0,
            "rigid_local_geometry": True,
        },
    }


def run_case(
    bridge: dict,
    v10: dict,
    v15: dict,
    v16: dict,
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
        metric_target, metric_gt_old, metric_gt_new = calibrated_targets(
            stream, metric_poses[0]
        )

    raw = load_raw_pair(local)
    kernel = np.ones((int(args.mask_dilate), int(args.mask_dilate)), dtype=np.uint8)
    masks = [
        cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) for mask in raw["mask"]
    ]
    seed = int(args.seed) + index
    raw_clouds = build_clouds(
        raw, intrinsics, raw_poses, [1.0, 1.0], masks, args, seed
    )
    metric_clouds = build_clouds(
        raw, intrinsics, metric_poses, scene_scales, masks, args, seed
    )

    raw_roots = [body["roots"][0], body["roots"][1]]
    metric_roots = [
        body["roots"][frame] * root_scales[frame] for frame in range(2)
    ]
    sources = rotation_sources(bridge, v15, v16)
    variants = {
        "fixed_explicit": actual_fixed_variant(
            np.asarray(v16["baselines"]["fixed_explicit"]["transform"], dtype=np.float32),
            raw_poses,
            raw_roots,
            raw_target,
            raw_gt_old,
            raw_gt_new,
            raw_clouds,
        )
    }
    raw_integrity = integrity_metrics(
        body, intrinsics, [1.0, 1.0], [1.0, 1.0]
    )
    da3_integrity = integrity_metrics(
        body, intrinsics, root_scales, scene_scales
    )
    for rotation_key in ROTATION_KEYS:
        variants[f"{rotation_key}_raw"] = {
            **solve_variant(
                sources[rotation_key],
                raw_poses,
                raw_roots,
                raw_target,
                raw_gt_old,
                raw_gt_new,
                raw_clouds,
            ),
            "integrity": raw_integrity,
        }
        variants[f"{rotation_key}_da3"] = {
            **solve_variant(
                sources[rotation_key],
                metric_poses,
                metric_roots,
                metric_target,
                metric_gt_old,
                metric_gt_new,
                metric_clouds,
            ),
            "integrity": da3_integrity,
        }
    return {
        "case_name": bridge["case_name"],
        "source": bridge["source"],
        "v32_branch": bridge["v32_branch"],
        "root_scales": bridge["root_scales"],
        "scene_scales": bridge["scene_scale_sets"]["absolute"],
        "variants": variants,
    }


def summarize(rows: list[dict], variant: str) -> dict:
    values = [row["variants"][variant] for row in rows]
    return {
        "camera_translation_m": distribution(
            [value["camera"]["translation_m"] for value in values]
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
        "foot_ground_distortion_m": distribution(
            [value["integrity"]["foot_ground_distortion_m"] for value in values]
        ),
        "human_reprojection_shift_px": distribution(
            [value["integrity"]["human_reprojection_shift_px"] for value in values]
        ),
        "rigid_local_geometry_rate": float(
            np.mean([value["integrity"]["rigid_local_geometry"] for value in values])
        ),
        "camera_catastrophic_rate": float(
            np.mean(
                [
                    value["camera"]["translation_m"] > 2.0
                    or value["camera"]["rotation_deg"] > 45.0
                    for value in values
                ]
            )
        ),
        "joint_integrity_failure_rate": float(
            np.mean(
                [
                    value["camera"]["translation_m"] > 2.0
                    or value["camera"]["rotation_deg"] > 45.0
                    or value["scene"]["trimmed_mean_m"] > 1.0
                    or value["integrity"]["foot_ground_distortion_m"] > 0.10
                    or value["integrity"]["human_reprojection_shift_px"] > 25.0
                    for value in values
                ]
            )
        ),
    }


def paired_effect(rows: list[dict], before: str, after: str) -> dict:
    output = {}
    for metric, group, key, lower_is_better in (
        ("translation_m", "camera", "translation_m", True),
        ("rotation_deg", "camera", "rotation_deg", True),
        ("scene_m", "scene", "trimmed_mean_m", True),
    ):
        old = np.asarray(
            [row["variants"][before][group][key] for row in rows], dtype=np.float64
        )
        new = np.asarray(
            [row["variants"][after][group][key] for row in rows], dtype=np.float64
        )
        delta = new - old
        output[metric] = {
            "mean_before": float(np.mean(old)),
            "mean_after": float(np.mean(new)),
            "mean_delta": float(np.mean(delta)),
            "improved_count": int(np.sum(delta < 0.0 if lower_is_better else delta > 0.0)),
            "harmed_count": int(np.sum(delta > 0.0 if lower_is_better else delta < 0.0)),
        }
    return output


def markdown(report: dict) -> str:
    names = ["fixed_explicit"] + [
        f"{rotation}_{scale}" for rotation in ROTATION_KEYS for scale in SCALE_KEYS
    ]
    lines = [
        "# V11.3 Component Necessity Ablation",
        "",
        "| Variant | T mean/P95 | R mean/P95 | Scene | Contact | Reprojection | Camera cat | Joint failure |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        value = report["overall"][name]
        lines.append(
            "| {name} | {tm:.3f}/{tp:.3f} | {rm:.2f}/{rp:.2f} | {scene:.3f} | "
            "{contact:.3f} | {reproj:.1f}px | {cat:.1%} | {joint:.1%} |".format(
                name=name,
                tm=value["camera_translation_m"]["mean"],
                tp=value["camera_translation_m"]["p95"],
                rm=value["camera_rotation_deg"]["mean"],
                rp=value["camera_rotation_deg"]["p95"],
                scene=value["scene_trimmed_mean_m"]["mean"],
                contact=value["foot_ground_distortion_m"]["mean"],
                reproj=value["human_reprojection_shift_px"]["mean"],
                cat=value["camera_catastrophic_rate"],
                joint=value["joint_integrity_failure_rate"],
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bridges = load_cases(args.bridge_report)
    v10 = load_cases(args.v10_report)
    v15 = load_candidate_shards(args.wide_candidate_dir, "v15_candidates_shard_*.json")
    v16 = load_candidate_shards(args.torso_candidate_dir, "v16_candidates_shard_*.json")
    streams = load_shards(args.stream_dir, "v18_stream_shard_*_of_*.json")
    names = sorted(set(bridges) & set(v10) & set(v15) & set(v16) & set(streams))
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
        rows.append(
            run_case(
                bridges[name],
                v10[name],
                v15[name],
                v16[name],
                streams[name],
                layer,
                args,
                index,
            )
        )
        if (index + 1) % 20 == 0:
            print(f"V11.3 component ablation {index + 1}/{len(names)}", flush=True)

    variant_names = ["fixed_explicit"] + [
        f"{rotation}_{scale}" for rotation in ROTATION_KEYS for scale in SCALE_KEYS
    ]
    report = {
        "experiment": "V11.3 Fixed/Torso/Wide/DA3 component necessity ablation",
        "case_count": len(rows),
        "protocol": {
            "rotation_sources": list(ROTATION_KEYS),
            "scale_sources": list(SCALE_KEYS),
            "shared_translation_solver": "explicit human-root camera equation",
            "shared_windows": "1+1",
            "runtime_gt": False,
            "gt_cut_index": True,
        },
        "overall": {name: summarize(rows, name) for name in variant_names},
        "by_source": {
            source: {
                name: summarize([row for row in rows if row["source"] == source], name)
                for name in variant_names
            }
            for source in sorted({row["source"] for row in rows})
        },
        "paired_effects": {
            "torso_over_fixed_raw": paired_effect(rows, "fixed_raw", "torso_raw"),
            "safe_gravity_over_torso_raw": paired_effect(rows, "torso_raw", "v22_raw"),
            "pure_vggt_over_fixed_raw": paired_effect(rows, "fixed_raw", "vggt_raw"),
            "conditional_vggt_over_v22_raw": paired_effect(rows, "v22_raw", "v32_raw"),
            **{
                f"da3_over_raw_with_{rotation}": paired_effect(
                    rows, f"{rotation}_raw", f"{rotation}_da3"
                )
                for rotation in ROTATION_KEYS
            },
        },
        "cases": rows,
    }
    json_path = args.output_dir / "v48_component_necessity_ablation.json"
    md_path = args.output_dir / "v48_component_necessity_ablation.md"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    md_path.write_text(markdown(report), encoding="utf-8")
    print(markdown(report))
    print(f">> wrote {json_path}")
    print(f">> wrote {md_path}")


if __name__ == "__main__":
    main()
