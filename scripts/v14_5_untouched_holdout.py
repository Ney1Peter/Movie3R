#!/usr/bin/env python3
"""Run frozen shot-scale inference and V14.4 evaluation on the V14.5 holdout."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "output/v14_5_final_audit/untouched_holdout"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("scale", "evaluate"), required=True)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--process_res", type=int, default=504)
    parser.add_argument("--point_samples", type=int, default=6000)
    parser.add_argument("--scene_samples", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args()


def load_shards(root: Path, pattern: str) -> dict[str, dict]:
    rows = []
    for path in sorted(glob.glob(str(root / pattern))):
        rows.extend(json.loads(Path(path).read_text(encoding="utf-8"))["cases"])
    output = {str(row["case_name"]): row for row in rows}
    if not rows or len(rows) != len(output):
        raise RuntimeError(f"Invalid shards at {root}: {len(rows)}/{len(output)}")
    return output


def load_cases(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["case_name"]): row for row in payload["cases"]}


def numeric_transform(row: dict) -> np.ndarray:
    return np.asarray(row["transform"], dtype=np.float32)


def workspace_path(path: str | Path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return str(candidate.resolve())


def fixed_case(v10: dict) -> dict:
    return {"methods": {"fixed_raw_geometry": v10["fixed_explicit"]}}


def scale_stage(args: argparse.Namespace) -> None:
    import v20_shot_scale_consistency_probe as v20
    import v21_absolute_shot_background_scale_probe as v21

    streams = load_shards(args.root / "stream_cache", "v18_stream_shard_*_of_*.json")
    keypoints = load_shards(
        args.root / "keypoint_cache", "v18_keypoints_shard_*_of_*.json"
    )
    scene = load_shards(args.root / "v16", "v16_candidates_shard_*_of_*.json")
    v10 = load_cases(args.root / "v10_merged/merged_cases.json")
    names = sorted(set(streams) & set(keypoints) & set(scene) & set(v10))
    if len(names) != 60:
        raise RuntimeError(
            f"Expected 60 untouched stream/keypoint/V16/V10 cases, got {len(names)}"
        )

    scale_root = args.root / "scale"
    scale_root.mkdir(parents=True, exist_ok=True)
    model_path = (
        ROOT.parent
        / "Movie3R-dataset/Depth-Anything-3/checkpoints/DA3Metric-Large"
    )
    model = v20.DepthAnything3.from_pretrained(str(model_path)).to(args.device).eval()
    v20_args = SimpleNamespace(
        output_dir=scale_root / "v20",
        da3_cache_dir=scale_root / "v20_da3_cache",
        overwrite_da3_cache=False,
        da3_mode="independent_single",
        process_res=int(args.process_res),
        keypoint_threshold=0.30,
        sample_radius=3,
        point_samples=int(args.point_samples),
        raw_confidence_threshold=1.0,
        mask_dilate=11,
        scene_iters=4,
        scene_max_distance=1.0,
        scene_min_distance=0.05,
        rotation_candidate_key="fixed_torso_motion_1f_resolve_t",
        rotation_blend_candidate_key=None,
        rotation_blend_alpha=0.0,
        seed=int(args.seed),
    )
    v20_args.output_dir.mkdir(parents=True, exist_ok=True)
    v20_rows = []
    enriched = {}
    for index, name in enumerate(names):
        case = {
            **streams[name],
            "cache_path": workspace_path(streams[name]["cache_path"]),
            "keypoint_path": workspace_path(keypoints[name]["cache_path"]),
            "scene_case": scene[name],
        }
        enriched[name] = case
        v20_rows.append(
            v20.run_case(case, v10[name], fixed_case(v10[name]), model, v20_args, index)
        )
        if (index + 1) % 5 == 0:
            print(f">> untouched V20 scale {index + 1}/{len(names)}", flush=True)
    v20_report = {
        "experiment": "V14.5 untouched holdout frozen V20 scale",
        "case_count": len(v20_rows),
        "protocol": {
            "frozen_v20_run_case": True,
            "da3_mode": "independent_single",
            "scale_source": "torso_first1",
            "scale_clip": [0.35, 3.0],
        },
        "cases": v20_rows,
    }
    v20_path = scale_root / "v20/v20_shot_scale_consistency.json"
    v20_path.write_text(json.dumps(v20_report, indent=2, allow_nan=True) + "\n")
    v20_map = {row["case_name"]: row for row in v20_rows}

    v21_args = SimpleNamespace(
        output_dir=scale_root / "v21",
        cache_dir=scale_root / "v21_da3_dense_cache",
        overwrite_cache=False,
        process_res=int(args.process_res),
        point_samples=int(args.point_samples),
        raw_confidence_threshold=1.0,
        mask_dilate=11,
        lowfreq_sigma=25.0,
        min_background_pixels=512,
        seed=int(args.seed),
    )
    v21_args.output_dir.mkdir(parents=True, exist_ok=True)
    v21_rows = []
    for index, name in enumerate(names):
        v21_rows.append(
            v21.run_case(enriched[name], v10[name], v20_map[name], model, v21_args, index)
        )
        if (index + 1) % 5 == 0:
            print(f">> untouched V21 background scale {index + 1}/{len(names)}", flush=True)
    v21_report = v21.build_report(v21_rows, v21_args)
    v21_report["experiment"] = "V14.5 untouched holdout frozen V21 background scale"
    v21_path = scale_root / "v21/v21_absolute_shot_background_scale.json"
    v21_path.write_text(json.dumps(v21_report, indent=2, allow_nan=True) + "\n")

    selected = []
    for name in names:
        root = v20_map[name]["methods"]["torso_first1"]
        background = next(row for row in v21_rows if row["case_name"] == name)[
            "variants"
        ]["median_ratio_q15_gate_lt95"]
        with np.load(scale_root / "v20_da3_cache" / f"{name}.npz") as cache:
            da3_new_root_depth = float(cache["pelvis_roots"][-1, 2])
        selected.append(
            {
                "case_name": name,
                "source": v20_map[name]["source"],
                "root_scales": {
                    "old": float(root["old_scale"]),
                    "new": float(root["new_scale"]),
                },
                "scene_scale_sets": {
                    "absolute": {
                        "old": float(background["old_scene_scale"]),
                        "new": float(background["new_scene_scale"]),
                    }
                },
                "da3_new_root_depth_m": da3_new_root_depth,
            }
        )
    selected_path = scale_root / "selected_scales.json"
    selected_path.write_text(
        json.dumps(
            {
                "experiment": "V14.5 frozen V11.4 scale cue on untouched holdout",
                "case_count": len(selected),
                "cases": selected,
            },
            indent=2,
        )
        + "\n"
    )
    print(f">> wrote {selected_path}", flush=True)


def component_rows(root: Path) -> dict[str, dict]:
    from v32_consensus_texture_safety_audit import selected_rotation

    v15 = load_shards(root / "v15", "v15_candidates_shard_*_of_*.json")
    v16 = load_shards(root / "v16", "v16_candidates_shard_*_of_*.json")
    output = {}
    for name in sorted(set(v15) & set(v16)):
        fixed_transform = numeric_transform(v15[name]["baselines"]["fixed_explicit"])
        torso_transform = numeric_transform(
            v16[name]["fixed_candidates"]["fixed_torso_motion_1f_resolve_t"]
        )
        rotation, branch, diagnostics = selected_rotation(
            fixed_transform[:3, :3],
            torso_transform[:3, :3],
            v15[name],
            0.05,
            consensus_cap_deg=60.0,
        )
        conditional = torso_transform.copy()
        conditional[:3, :3] = rotation
        output[name] = {
            "case_name": name,
            "source": v15[name].get("source", v15[name]["record"]["source"]),
            "v32_branch": branch,
            "diagnostics": diagnostics,
            "variants": {
                "fixed_raw": {"transform": fixed_transform.astype(float).tolist()},
                "torso_raw": {"transform": torso_transform.astype(float).tolist()},
                "v32_raw": {"transform": conditional.astype(float).tolist()},
            },
        }
    return output


def distribution(values: list[float]) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
    }


def method_summary(rows: list[dict], method: str, scene_names: set[str]) -> dict:
    methods = [row["methods"][method] for row in rows]
    camera = np.asarray([row["camera"]["translation_m"] for row in methods])
    rotation = np.asarray([row["camera"]["rotation_deg"] for row in methods])
    root = np.asarray([row["human"]["world_root_error_m"] for row in methods])
    joints = np.asarray([row["human"]["world_joint_mean_error_m"] for row in methods])
    vertices = np.asarray([row["human"]["world_vertex_mean_error_m"] for row in methods])
    scene = [
        row["methods"][method]["scene"]["trimmed_mean_m"]
        for row in rows
        if row["case_name"] in scene_names
    ]
    return {
        "camera_translation_m": distribution(camera.tolist()),
        "camera_rotation_deg": distribution(rotation.tolist()),
        "human_root_m": distribution(root.tolist()),
        "human_joints_m": distribution(joints.tolist()),
        "human_vertices_m": distribution(vertices.tolist()),
        "scene_m": distribution(scene),
        "scene_valid_count": len(scene),
        "catastrophic_rate": float(np.mean((camera > 2.0) | (rotation > 45.0))),
        "success_rate": float(np.mean((camera < 0.5) & (rotation < 20.0))),
    }


def evaluate_stage(args: argparse.Namespace) -> None:
    import v14_4_unified_similarity_reanchoring_probe as v14
    from dust3r.utils.smpl_layer import SMPL_Layer

    streams = load_shards(args.root / "stream_cache", "v18_stream_shard_*_of_*.json")
    keypoints = load_shards(
        args.root / "keypoint_cache", "v18_keypoints_shard_*_of_*.json"
    )
    v10 = load_cases(args.root / "v10_merged/merged_cases.json")
    scales = load_cases(args.root / "scale/selected_scales.json")
    components = component_rows(args.root)
    names = sorted(set(streams) & set(keypoints) & set(v10) & set(scales) & set(components))
    if len(names) != 60:
        raise RuntimeError(f"Expected 60 common untouched cases, got {len(names)}")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    layer10 = SMPL_Layer(
        type="smplx", gender="neutral", num_betas=10, kid=False, person_center="head"
    ).to(device).eval()
    layer11 = SMPL_Layer(
        type="smplx", gender="neutral", num_betas=11, kid=False, person_center="head"
    ).to(device).eval()
    foot_indices = np.asarray(
        [layer10.joint_names.index(name) for name in v14.FOOT_NAMES], dtype=np.int64
    )
    eval_args = SimpleNamespace(
        scene_samples=int(args.scene_samples),
        confidence_threshold=1.5,
        mask_dilate=11,
        keypoint_threshold=0.30,
        oracle_scale_min=0.20,
        oracle_scale_max=2.50,
    )
    rows = []
    for index, name in enumerate(names):
        stream_path = workspace_path(streams[name]["cache_path"])
        keypoint_path = workspace_path(keypoints[name]["cache_path"])
        with np.load(stream_path) as stream:
            beta = np.median(stream["old_shape"], axis=0).astype(float).tolist()
            roots = np.asarray(stream["old_joints_camera"], dtype=np.float32)
            body = roots - roots[:, :1]
            physical = float(np.median([v14.physical_scale(value) for value in body]))
        case = {
            **streams[name],
            "cache_path": Path(stream_path),
            "keypoint_path": Path(keypoint_path),
            "v14_2": {
                "memory": {
                    "canonical_beta": beta,
                    "canonical_physical_scale": physical,
                }
            },
            "da3": {"depth": {"da3_pelvis_m": scales[name]["da3_new_root_depth_m"]}},
            "local_dir": Path(workspace_path(v10[name]["paths"]["human3r_local_reset"])),
        }
        rows.append(
            v14.run_case(
                case,
                scales[name],
                components[name],
                layer10,
                layer11,
                foot_indices,
                device,
                eval_args,
            )
        )
        if (index + 1) % 5 == 0:
            print(f">> untouched V14.4 evaluator {index + 1}/{len(names)}", flush=True)

    methods = (
        "fixed_explicit",
        "v11_1_conditional_wide_raw_scale",
        "v11_4_uniform_similarity",
        "v11_4_uniform_similarity_conditional_vggt",
        "unified_da3_absolute_scale_da3_coupled_root_conditional_vggt",
        "boundary_oracle",
    )
    scene_names = {
        row["case_name"]
        for row in rows
        if all(row["methods"][method]["scene"]["valid"] for method in methods)
    }
    fixed_camera = np.asarray(
        [row["methods"]["fixed_explicit"]["camera"]["translation_m"] for row in rows]
    )
    v11_camera = np.asarray(
        [
            row["methods"]["v11_4_uniform_similarity_conditional_vggt"]["camera"][
                "translation_m"
            ]
            for row in rows
        ]
    )
    no_vggt_rotation = np.asarray(
        [
            row["methods"]["v11_4_uniform_similarity"]["camera"]["rotation_deg"]
            for row in rows
        ]
    )
    conditional_rotation = np.asarray(
        [
            row["methods"]["v11_4_uniform_similarity_conditional_vggt"]["camera"][
                "rotation_deg"
            ]
            for row in rows
        ]
    )
    ranked_captures = sorted(
        (
            {
                "case_name": row["case_name"],
                "capture_key": [
                    row["record"].get("source"),
                    row["record"].get("group"),
                    row["record"].get("start_frame"),
                ],
                "source": row["source"],
                "fixed_camera_m": float(fixed_camera[index]),
                "v11_4_camera_m": float(v11_camera[index]),
                "v11_4_minus_fixed_m": float(v11_camera[index] - fixed_camera[index]),
            }
            for index, row in enumerate(rows)
        ),
        key=lambda row: row["v11_4_camera_m"],
        reverse=True,
    )
    report = {
        "experiment": "V14.5 untouched capture-disjoint holdout",
        "case_count": len(rows),
        "protocol": {
            "selection_frozen_before_inference": True,
            "capture_disjoint": True,
            "methods_and_thresholds_frozen": True,
            "common_v14_4_evaluator": True,
        },
        "scene_common_valid_count": len(scene_names),
        "overall": {method: method_summary(rows, method, scene_names) for method in methods},
        "by_source": {
            source: {
                method: method_summary(
                    [row for row in rows if row["source"] == source], method, scene_names
                )
                for method in methods
            }
            for source in sorted({row["source"] for row in rows})
        },
        "paired_v11_4_vs_fixed": {
            "camera_mean_gain_m": float(
                np.mean(fixed_camera - v11_camera)
            ),
            "camera_improved_rate": float(
                np.mean(v11_camera < fixed_camera)
            ),
            "camera_harmed_rate": float(np.mean(v11_camera > fixed_camera)),
            "camera_harmful_over_005m_rate": float(
                np.mean(v11_camera > fixed_camera + 0.05)
            ),
            "camera_paired_wilcoxon_p": float(
                wilcoxon(v11_camera, fixed_camera).pvalue
            ),
        },
        "conditional_vggt": {
            "trigger_count": int(sum(row["conditional_vggt_triggered"] for row in rows)),
            "trigger_rate": float(np.mean([row["conditional_vggt_triggered"] for row in rows])),
            "rotation_mean_gain_deg": float(
                np.mean(no_vggt_rotation - conditional_rotation)
            ),
            "rotation_improved_rate": float(
                np.mean(conditional_rotation < no_vggt_rotation)
            ),
            "rotation_harmed_over_5deg_count": int(
                np.sum(conditional_rotation > no_vggt_rotation + 5.0)
            ),
            "rotation_improved_over_5deg_count": int(
                np.sum(conditional_rotation + 5.0 < no_vggt_rotation)
            ),
            "paired_wilcoxon_p": float(
                wilcoxon(conditional_rotation, no_vggt_rotation).pvalue
            ),
        },
        "worst_captures_v11_4": ranked_captures[:10],
        "cases": rows,
    }
    output_dir = args.root / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "v14_5_untouched_holdout.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
    print(f">> wrote {output}", flush=True)


def main() -> None:
    args = parse_args()
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("V14.5 untouched holdout stages require CUDA")
    if args.stage == "scale":
        scale_stage(args)
    else:
        evaluate_stage(args)


if __name__ == "__main__":
    main()
