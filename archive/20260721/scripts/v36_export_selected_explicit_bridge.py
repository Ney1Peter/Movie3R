#!/usr/bin/env python3
"""Export the selected V36 synchronized metric bridge for runtime use."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = (
    REPO_ROOT
    / "output/v36_final_explicit_metric_bridge/v36_final_explicit_metric_bridge.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "output/v36_final_explicit_metric_bridge/selected_candidates"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    rows = []
    invalid = []
    for case in report["cases"]:
        selected = case["variants"]["v36"]
        transform = np.asarray(selected["transform"], dtype=np.float64)
        rotation = transform[:3, :3] if transform.shape == (4, 4) else np.zeros((3, 3))
        diagnostics = {
            "shape": list(transform.shape),
            "finite": bool(np.isfinite(transform).all()),
            "bottom_row_error": (
                float(np.max(np.abs(transform[3] - [0.0, 0.0, 0.0, 1.0])))
                if transform.shape == (4, 4)
                else float("inf")
            ),
            "orthogonality_error": float(
                np.max(np.abs(rotation.T @ rotation - np.eye(3)))
            ),
            "determinant": float(np.linalg.det(rotation)),
        }
        valid = bool(
            diagnostics["shape"] == [4, 4]
            and diagnostics["finite"]
            and diagnostics["bottom_row_error"] <= 1e-5
            and diagnostics["orthogonality_error"] <= 1e-4
            and abs(diagnostics["determinant"] - 1.0) <= 1e-4
        )
        if not valid:
            invalid.append({"case_name": case["case_name"], **diagnostics})
        scene_scale = case["scene_scale_sets"]["absolute"]
        rows.append(
            {
                "case_name": case["case_name"],
                "source": case["source"],
                "boundary_transform": selected["transform"],
                "old_shot_human_camera_scale": case["root_scales"]["old"],
                "new_shot_human_camera_scale": case["root_scales"]["new"],
                "old_shot_background_depth_scale": scene_scale["old"],
                "new_shot_background_depth_scale": scene_scale["new"],
                "v32_branch": case["v32_branch"],
                "adaptive_consensus_cap_used": case["adapted"],
                "human_torso_jump_deg": case["human_torso_jump_deg"],
                "rotation_diagnostics": case["diagnostics"],
                "evaluation": {
                    "camera": selected["camera"],
                    "human": selected["human"],
                    "scene": selected["scene"],
                },
            }
        )
    if invalid or len(rows) != 180 or len({row["case_name"] for row in rows}) != 180:
        raise RuntimeError(
            f"Invalid V36 export: rows={len(rows)}, unique={len({row['case_name'] for row in rows})}, invalid={len(invalid)}"
        )
    payload = {
        "experiment": "V36 selected synchronized explicit metric bridge",
        "case_count": len(rows),
        "runtime_order": [
            "hard reset Human3R at the cut",
            "estimate independent DA3 human/background metric shot scales",
            "apply the DA3 root scale to Human3R camera translation and SMPL-X root",
            "apply the bounded V22 background scale correction to the pointmap",
            "compute bounded torso/gravity rotation",
            "conditionally compute frozen VGGT full-RGB 1+1 rotation",
            "apply texture-safe and human-jump-adaptive physical safety rules",
            "re-solve camera translation with the explicit metric human-root equation",
            "store one fixed shot-level scale state and SE(3)",
        ],
        "constraints": {
            "post_cut_frames": 1,
            "future_shot_access": False,
            "human3r_frozen": True,
            "learned_gate": False,
            "source_specific_rule": False,
            "gt_runtime_information": False,
            "camera_pointmap_smplx_share_final_transform": True,
        },
        "integrity": {
            "all_transforms_valid_se3": True,
            "unique_case_names": True,
            "adaptive_consensus_count": int(
                sum(row["adaptive_consensus_cap_used"] for row in rows)
            ),
        },
        "cases": rows,
    }
    output = args.output_dir / "v36_selected_explicit_metric_bridge.json"
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["integrity"], indent=2))
    print(f">> wrote {output}")


if __name__ == "__main__":
    main()
