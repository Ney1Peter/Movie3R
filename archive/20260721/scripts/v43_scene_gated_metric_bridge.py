#!/usr/bin/env python3
"""Compose V36 with the stable V41 scene-gated post-cut background scale."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from v19_da3_explicit_geometry_correction_probe import distribution


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V36 = REPO_ROOT / "output/v36_final_explicit_metric_bridge/v36_final_explicit_metric_bridge.json"
DEFAULT_V40 = REPO_ROOT / "output/v40_human_background_scale_fusion/v40_human_background_scale_fusion.json"
DEFAULT_V41 = REPO_ROOT / "output/v41_background_scale_scene_safety/v41_background_scale_scene_safety_audit.json"
DEFAULT_OUTPUT = REPO_ROOT / "output/v43_scene_gated_metric_bridge"
SCENE_GAIN_THRESHOLD_M = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v36_report", type=Path, default=DEFAULT_V36)
    parser.add_argument("--v40_report", type=Path, default=DEFAULT_V40)
    parser.add_argument("--v41_report", type=Path, default=DEFAULT_V41)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_cases(path: Path, key: str = "cases") -> dict[str, dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))[key]
    return {row["case_name"]: row for row in rows}


def summarize(rows: list[dict], variant: str, baseline: str = "v36") -> dict:
    values = [row["variants"][variant] for row in rows]
    references = [row["variants"][baseline] for row in rows]
    translation = np.asarray([row["camera"]["translation_m"] for row in values])
    rotation = np.asarray([row["camera"]["rotation_deg"] for row in values])
    human = np.asarray([row["human"]["root_motion_error_m"] for row in values])
    scene = np.asarray([row["scene"]["trimmed_mean_m"] for row in values])
    base_translation = np.asarray([row["camera"]["translation_m"] for row in references])
    base_rotation = np.asarray([row["camera"]["rotation_deg"] for row in references])
    base_human = np.asarray([row["human"]["root_motion_error_m"] for row in references])
    base_scene = np.asarray([row["scene"]["trimmed_mean_m"] for row in references])
    return {
        "camera_translation_m": distribution(translation.tolist()),
        "camera_rotation_deg": distribution(rotation.tolist()),
        "human_motion_error_m": distribution(human.tolist()),
        "scene_trimmed_mean_m": distribution(scene.tolist()),
        "combined_catastrophic_rate": float(np.mean(
            (translation > 2.0) | (rotation > 45.0) | (human > 0.50) | (scene > 1.0)
        )),
        "strict_success_rate": float(np.mean(
            (translation < 0.5) & (rotation < 10.0) & (human < 0.10) & (scene < 0.20)
        )),
        "translation_improved_005m": int(np.sum(translation + 0.05 < base_translation)),
        "translation_harmful_005m": int(np.sum(translation > base_translation + 0.05)),
        "rotation_harmful_5deg": int(np.sum(rotation > base_rotation + 5.0)),
        "human_harmful_005m": int(np.sum(human > base_human + 0.05)),
        "scene_harmful_005m": int(np.sum(scene > base_scene + 0.05)),
    }


def main() -> None:
    args = parse_args()
    v36 = load_cases(args.v36_report)
    v40 = load_cases(args.v40_report)
    v41 = load_cases(args.v41_report, key="rows")
    names = sorted(set(v36) & set(v40) & set(v41))
    if len(names) != 180:
        raise RuntimeError(f"Expected 180 common cases, got {len(names)}")

    rows = []
    for name in names:
        use_background = bool(v41[name]["scene_delta_m"] < -SCENE_GAIN_THRESHOLD_M)
        selected_key = "human_old_background_new" if use_background else "human_root"
        selected = {
            "transform": v40[name]["variants"][selected_key]["transform"],
            "camera": v40[name]["variants"][selected_key]["camera"],
            "human": v40[name]["variants"][selected_key]["human"],
            "scene": v41[name]["variants"][selected_key]["scene"],
            "root_scales": v40[name]["variants"][selected_key]["scales"],
        }
        baseline = {
            "transform": v40[name]["variants"]["human_root"]["transform"],
            "camera": v40[name]["variants"]["human_root"]["camera"],
            "human": v40[name]["variants"]["human_root"]["human"],
            "scene": v41[name]["variants"]["human_root"]["scene"],
            "root_scales": v40[name]["variants"]["human_root"]["scales"],
        }
        rows.append({
            "case_name": name,
            "source": v36[name]["source"],
            "scale_branch": "scene_gated_background" if use_background else "human_root",
            "scene_delta_m": v41[name]["scene_delta_m"],
            "camera_delta_m": v41[name]["camera_delta_m"],
            "variants": {"v36": baseline, "v43": selected},
        })

    report = {
        "experiment": "V43 scene-gated explicit metric bridge",
        "case_count": len(rows),
        "protocol": {
            "base": "V36 explicit metric bridge",
            "candidate": "post-cut DA3 background scale in the explicit human-root equation",
            "acceptance": f"candidate scene trimmed-mean improves by more than {SCENE_GAIN_THRESHOLD_M:.2f} m",
            "post_cut_frames": 1,
            "gt_runtime_information": False,
            "shot_transform_count": 1,
        },
        "selected_count": int(sum(row["scale_branch"] == "scene_gated_background" for row in rows)),
        "overall": {
            "v36": summarize(rows, "v36"),
            "v43": summarize(rows, "v43"),
        },
        "by_source": {
            source: {
                "v36": summarize([row for row in rows if row["source"] == source], "v36"),
                "v43": summarize([row for row in rows if row["source"] == source], "v43"),
            }
            for source in sorted({row["source"] for row in rows})
        },
        "selected_cases": [row for row in rows if row["scale_branch"] == "scene_gated_background"],
        "cases": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "v43_scene_gated_metric_bridge.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    export = {
        "method": "V43 scene-gated explicit metric bridge",
        "case_count": len(rows),
        "scene_gain_threshold_m": SCENE_GAIN_THRESHOLD_M,
        "cases": [
            {
                "case_name": row["case_name"],
                "source": row["source"],
                "scale_branch": row["scale_branch"],
                "transform": row["variants"]["v43"]["transform"],
                "root_scales": row["variants"]["v43"]["root_scales"],
            }
            for row in rows
        ],
    }
    export_path = args.output_dir / "v43_selected_explicit_metric_bridge.json"
    export_path.write_text(json.dumps(export, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "selected_count": report["selected_count"],
        "overall": report["overall"],
        "by_source": report["by_source"],
    }, indent=2))
    print(f">> wrote {output}")
    print(f">> wrote {export_path}")


if __name__ == "__main__":
    main()
