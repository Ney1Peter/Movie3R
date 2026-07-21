#!/usr/bin/env python3
"""Fuse independent DA3 human-root and background shot scales under V36 rotation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from v18_da3_metric_depth_probe import boundary_from_camera_pose, camera_pose_from_human, evaluate
from v19_da3_explicit_geometry_correction_probe import distribution, human_metrics, transform_points
from v20_shot_scale_consistency_probe import calibrated_targets, scale_pose
from v22_explicit_metric_bridge_selection import load_shards


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V36 = REPO_ROOT / "output/v36_final_explicit_metric_bridge/v36_final_explicit_metric_bridge.json"
DEFAULT_STREAM = REPO_ROOT / "output/v18_human_metric_translation/stream_cache"
DEFAULT_OUTPUT = REPO_ROOT / "output/v40_human_background_scale_fusion"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v36_report", type=Path, default=DEFAULT_V36)
    parser.add_argument("--stream_dir", type=Path, default=DEFAULT_STREAM)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rotation_variant", choices=("v32", "v36"), default="v32")
    return parser.parse_args()


def load_cases(path: Path) -> dict[str, dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))["cases"]
    return {row["case_name"]: row for row in rows}


def log_blend(human: float, background: float, alpha: float) -> float:
    return float(np.exp((1.0 - alpha) * np.log(human) + alpha * np.log(background)))


def scale_candidates(case: dict) -> dict[str, tuple[float, float]]:
    human = case["root_scales"]
    background = case["scene_scale_sets"]["absolute"]
    human_pair = (float(human["old"]), float(human["new"]))
    background_pair = (float(background["old"]), float(background["new"]))
    candidates = {
        "human_root": human_pair,
        "background": background_pair,
        "human_old_background_new": (human_pair[0], background_pair[1]),
    }
    for alpha in (0.25, 0.50, 0.75):
        key = f"log_blend_{int(100 * alpha):02d}"
        candidates[key] = tuple(
            log_blend(human_value, background_value, alpha)
            for human_value, background_value in zip(human_pair, background_pair)
        )
    return candidates


def run_case(case: dict, stream_row: dict, rotation_variant: str) -> dict:
    boundary_rotation = np.asarray(
        case["variants"][rotation_variant]["transform"], dtype=np.float32
    )[:3, :3]
    with np.load(stream_row["cache_path"]) as stream:
        raw_poses = np.stack([stream["old_pose"][-1], stream["new_pose"]]).astype(np.float32)
        translations = np.stack([stream["old_transl"][-1], stream["new_transl"]]).astype(np.float32)
        variants = {}
        for name, (old_scale, new_scale) in scale_candidates(case).items():
            old_pose = scale_pose(raw_poses[0], old_scale)
            new_pose = scale_pose(raw_poses[1], new_scale)
            target_pose, gt_old_world, gt_new_world = calibrated_targets(stream, old_pose)
            old_root = translations[0] * old_scale
            new_root = translations[1] * new_scale
            old_anchor_world = transform_points(old_pose, old_root[None])[0]
            camera_rotation = boundary_rotation @ new_pose[:3, :3]
            camera_pose = camera_pose_from_human(camera_rotation, old_anchor_world, new_root)
            transform = boundary_from_camera_pose(camera_pose, new_pose)
            variants[name] = {
                "scales": {"old": old_scale, "new": new_scale},
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
            }
    human = case["root_scales"]
    background = case["scene_scale_sets"]["absolute"]
    return {
        "case_name": case["case_name"],
        "source": case["source"],
        "scale_ratio": {
            "old_background_over_human": float(background["old"] / human["old"]),
            "new_background_over_human": float(background["new"] / human["new"]),
        },
        "variants": variants,
    }


def summarize(rows: list[dict], variant: str) -> dict:
    values = [row["variants"][variant] for row in rows]
    baseline = [row["variants"]["human_root"] for row in rows]
    translation = np.asarray([row["camera"]["translation_m"] for row in values])
    base_translation = np.asarray([row["camera"]["translation_m"] for row in baseline])
    viewing = np.asarray([row["camera"]["viewing_direction_m"] for row in values])
    transverse = np.asarray([row["camera"]["transverse_m"] for row in values])
    human_motion = np.asarray([row["human"]["root_motion_error_m"] for row in values])
    return {
        "camera_translation_m": distribution(translation.tolist()),
        "viewing_direction_m": distribution(viewing.tolist()),
        "transverse_m": distribution(transverse.tolist()),
        "human_motion_error_m": distribution(human_motion.tolist()),
        "translation_catastrophic_rate": float(np.mean(translation > 2.0)),
        "improved_005m": float(np.mean(translation + 0.05 < base_translation)),
        "harmful_005m": float(np.mean(translation > base_translation + 0.05)),
        "improved_010m": float(np.mean(translation + 0.10 < base_translation)),
        "harmful_010m": float(np.mean(translation > base_translation + 0.10)),
        "mean_delta_m": float(np.mean(translation - base_translation)),
    }


def main() -> None:
    args = parse_args()
    cases = load_cases(args.v36_report)
    streams = load_shards(args.stream_dir, "v18_stream_shard_*_of_*.json")
    names = sorted(set(cases) & set(streams))
    if len(names) != 180:
        raise RuntimeError(f"Expected 180 common cases, got {len(names)}")
    rows = [run_case(cases[name], streams[name], args.rotation_variant) for name in names]
    variants = tuple(rows[0]["variants"])
    report = {
        "experiment": "V40 human/background DA3 shot-scale fusion under frozen V36 rotation",
        "case_count": len(rows),
        "protocol": {
            "rotation": f"frozen {args.rotation_variant}",
            "translation": "explicit human-root equation",
            "scale_sources": "independent DA3 human-root and static-background estimates",
            "gt_runtime_information": False,
        },
        "scale_disagreement_count": int(sum(
            abs(row["scale_ratio"]["old_background_over_human"] - 1.0) > 1e-5
            or abs(row["scale_ratio"]["new_background_over_human"] - 1.0) > 1e-5
            for row in rows
        )),
        "overall": {variant: summarize(rows, variant) for variant in variants},
        "by_source": {
            source: {
                variant: summarize([row for row in rows if row["source"] == source], variant)
                for variant in variants
            }
            for source in sorted({row["source"] for row in rows})
        },
        "cases": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "v40_human_background_scale_fusion.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "scale_disagreement_count": report["scale_disagreement_count"],
        "overall": report["overall"],
        "by_source": report["by_source"],
    }, indent=2))
    print(f">> wrote {output}")


if __name__ == "__main__":
    main()
