#!/usr/bin/env python3
"""Test robust DA3 human-root shot scales under the frozen V36 rotation."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from v18_da3_metric_depth_probe import boundary_from_camera_pose, camera_pose_from_human, evaluate
from v19_da3_explicit_geometry_correction_probe import distribution, human_metrics, transform_points
from v20_shot_scale_consistency_probe import calibrated_targets, scale_pose
from v22_explicit_metric_bridge_selection import load_shards


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V36 = REPO_ROOT / "output/v36_final_explicit_metric_bridge/v36_final_explicit_metric_bridge.json"
DEFAULT_V20 = (
    REPO_ROOT
    / "output/v20_shot_scale_consistency/full180_independent_gravity_conf1_fair/v20_shot_scale_consistency.json"
)
DEFAULT_STREAM = REPO_ROOT / "output/v18_human_metric_translation/stream_cache"
DEFAULT_OUTPUT = REPO_ROOT / "output/v39_da3_root_scale_robustness"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v36_report", type=Path, default=DEFAULT_V36)
    parser.add_argument("--v20_report", type=Path, default=DEFAULT_V20)
    parser.add_argument("--stream_dir", type=Path, default=DEFAULT_STREAM)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_cases(path: Path) -> dict[str, dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))["cases"]
    output = {row["case_name"]: row for row in rows}
    if len(output) != len(rows):
        raise RuntimeError(f"Duplicate cases in {path}")
    return output


def root_scale_candidates(v20: dict) -> dict[str, tuple[float, float]]:
    methods = v20["methods"]
    torso_first = methods["torso_first1"]
    pelvis_first = methods["pelvis_first1"]
    torso_median = methods["torso_median5"]
    pelvis_median = methods["pelvis_median5"]

    def pair(method: dict) -> tuple[float, float]:
        return float(method["old_scale"]), float(method["new_scale"])

    torso_first_pair = pair(torso_first)
    pelvis_first_pair = pair(pelvis_first)
    torso_median_pair = pair(torso_median)
    pelvis_median_pair = pair(pelvis_median)

    def geometric_mean(first: tuple[float, float], second: tuple[float, float]) -> tuple[float, float]:
        return tuple(float(np.sqrt(a * b)) for a, b in zip(first, second))

    return {
        "torso_first1": torso_first_pair,
        "pelvis_first1": pelvis_first_pair,
        "torso_median5": torso_median_pair,
        "pelvis_median5": pelvis_median_pair,
        "torso_pelvis_first1_geomean": geometric_mean(torso_first_pair, pelvis_first_pair),
        "torso_pelvis_median5_geomean": geometric_mean(torso_median_pair, pelvis_median_pair),
    }


def run_case(v36: dict, v20: dict, stream_row: dict) -> dict:
    boundary_rotation = np.asarray(v36["variants"]["v36"]["transform"], dtype=np.float32)[:3, :3]
    with np.load(stream_row["cache_path"]) as stream:
        raw_poses = np.stack([stream["old_pose"][-1], stream["new_pose"]]).astype(np.float32)
        translations = np.stack([stream["old_transl"][-1], stream["new_transl"]]).astype(np.float32)
        variants = {}
        for name, (old_scale, new_scale) in root_scale_candidates(v20).items():
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
                "root_scales": {"old": old_scale, "new": new_scale},
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
    return {
        "case_name": v36["case_name"],
        "source": v36["source"],
        "variants": variants,
    }


def summarize(rows: list[dict], variant: str, baseline: str = "torso_first1") -> dict:
    values = [row["variants"][variant] for row in rows]
    references = [row["variants"][baseline] for row in rows]
    translation = np.asarray([row["camera"]["translation_m"] for row in values])
    rotation = np.asarray([row["camera"]["rotation_deg"] for row in values])
    human = np.asarray([row["human"]["root_motion_error_m"] for row in values])
    viewing = np.asarray([row["camera"]["viewing_direction_m"] for row in values])
    transverse = np.asarray([row["camera"]["transverse_m"] for row in values])
    base_translation = np.asarray([row["camera"]["translation_m"] for row in references])
    return {
        "camera_translation_m": distribution(translation.tolist()),
        "camera_rotation_deg": distribution(rotation.tolist()),
        "viewing_direction_m": distribution(viewing.tolist()),
        "transverse_m": distribution(transverse.tolist()),
        "human_motion_error_m": distribution(human.tolist()),
        "translation_catastrophic_rate": float(np.mean(translation > 2.0)),
        "translation_improved_005m": float(np.mean(translation + 0.05 < base_translation)),
        "translation_harmful_005m": float(np.mean(translation > base_translation + 0.05)),
        "translation_harmful_010m": float(np.mean(translation > base_translation + 0.10)),
    }


def main() -> None:
    args = parse_args()
    v36 = load_cases(args.v36_report)
    v20 = load_cases(args.v20_report)
    streams = load_shards(args.stream_dir, "v18_stream_shard_*_of_*.json")
    names = sorted(set(v36) & set(v20) & set(streams))
    if len(names) != 180:
        raise RuntimeError(f"Expected 180 common cases, got {len(names)}")

    rows = [run_case(v36[name], v20[name], streams[name]) for name in names]
    variants = tuple(rows[0]["variants"])
    report = {
        "experiment": "V39 robust DA3 root-scale probe under frozen V36 rotation",
        "case_count": len(rows),
        "protocol": {
            "rotation": "frozen V36",
            "translation": "explicit human-root equation",
            "post_cut_frames": 1,
            "gt_runtime_information": False,
        },
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
    output = args.output_dir / "v39_da3_root_scale_robustness.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({"overall": report["overall"], "by_source": report["by_source"]}, indent=2))
    print(f">> wrote {output}")


if __name__ == "__main__":
    main()
