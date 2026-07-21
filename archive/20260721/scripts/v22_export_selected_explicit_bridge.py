#!/usr/bin/env python3
"""Export compact per-cut parameters for the selected V22 explicit bridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    REPO_ROOT
    / "output"
    / "v22_explicit_metric_bridge"
    / "final_seed1"
    / "v22_explicit_metric_bridge.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "output" / "v22_explicit_metric_bridge" / "selected_candidates"
SELECTED = "safe_gravity_absolute_scene_scale"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_report", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads(args.input_report.read_text(encoding="utf-8"))
    rows = []
    for case in report["cases"]:
        selected = case["variants"][SELECTED]
        transform = np.asarray(selected["transform"], dtype=np.float32)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise RuntimeError(f"Invalid transform for {case['case_name']}")
        rows.append(
            {
                "case_name": case["case_name"],
                "source": case["source"],
                "boundary_transform": transform.astype(float).tolist(),
                "old_shot_human_camera_scale": float(case["root_scales"]["old"]),
                "new_shot_human_camera_scale": float(case["root_scales"]["new"]),
                "old_shot_background_depth_scale": float(case["scene_scales"]["old"]),
                "new_shot_background_depth_scale": float(case["scene_scales"]["new"]),
                "gravity_residual_accepted": bool(case["gravity"]["accepted"]),
                "gravity_diagnostics": case["gravity"],
                "evaluation": {
                    "camera": selected["camera"],
                    "human": selected["human"],
                    "scene": selected["scene"],
                },
            }
        )
    output = {
        "experiment": "V22 selected explicit metric bridge export",
        "case_count": len(rows),
        "runtime_order": [
            "hard reset Human3R at cut",
            "infer DA3 once on the first new-shot frame",
            "multiply new-shot camera translations and SMPL-X root translations by new_shot_human_camera_scale",
            "multiply new-shot background pointmap depth by new_shot_background_depth_scale",
            "apply boundary_transform to camera, pointmap and SMPL-X with the same left-multiplied SE(3)",
            "keep all scales and boundary_transform fixed for the rest of the shot",
        ],
        "constraints": {
            "post_cut_frames": 1,
            "learned_alignment": False,
            "cross_cut_pointcloud_fit": False,
            "per_frame_boundary_update": False,
        },
        "gravity_acceptance_rate": float(
            np.mean([row["gravity_residual_accepted"] for row in rows])
        ),
        "cases": rows,
    }
    path = args.output_dir / "v22_selected_explicit_bridge.json"
    path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(f">> wrote {path}")


if __name__ == "__main__":
    main()
