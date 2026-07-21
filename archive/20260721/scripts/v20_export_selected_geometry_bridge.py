#!/usr/bin/env python3
"""Export the selected V20 shot-level geometry corrections and audit SE(3) validity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BOUNDARY = (
    REPO_ROOT
    / "output"
    / "v20_shot_scale_consistency"
    / "full180_independent_bound45_conf1_fair"
    / "v20_shot_scale_consistency.json"
)
DEFAULT_SCENE = (
    REPO_ROOT
    / "output"
    / "v20_shot_scale_consistency"
    / "scene_depth_scale_independent_safe"
    / "v20_scene_depth_scale_refinement.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "output"
    / "v20_shot_scale_consistency"
    / "selected_candidates"
    / "v20_selected_geometry_bridge.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundary_report", type=Path, default=DEFAULT_BOUNDARY)
    parser.add_argument("--scene_report", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--method", default="torso_first1")
    parser.add_argument("--pointmap_variant", default="q30")
    return parser.parse_args()


def rotation_audit(transform: np.ndarray) -> dict:
    rotation = transform[:3, :3]
    return {
        "determinant": float(np.linalg.det(rotation)),
        "orthogonality_max_abs": float(np.max(np.abs(rotation.T @ rotation - np.eye(3)))),
        "bottom_row_max_abs": float(np.max(np.abs(transform[3] - np.asarray([0.0, 0.0, 0.0, 1.0])))),
    }


def main() -> None:
    args = parse_args()
    boundary = json.loads(args.boundary_report.read_text(encoding="utf-8"))
    scene = json.loads(args.scene_report.read_text(encoding="utf-8"))
    boundary_rows = {row["case_name"]: row for row in boundary["cases"]}
    scene_rows = {row["case_name"]: row for row in scene["cases"]}
    if len(boundary_rows) != 180 or len(scene_rows) != 180 or set(boundary_rows) != set(scene_rows):
        raise RuntimeError("Selected V20 export requires matching 180-cut reports")
    rows = []
    audits = []
    for case_name in sorted(boundary_rows):
        base = boundary_rows[case_name]
        geometry = base["methods"][args.method]
        transform = np.asarray(geometry["variants"]["b00"]["transform"], dtype=np.float64)
        pointmap = scene_rows[case_name]["methods"][args.method][args.pointmap_variant]
        audit = rotation_audit(transform)
        values = np.concatenate(
            [
                transform.reshape(-1),
                np.asarray(
                    [geometry["old_scale"], geometry["new_scale"], pointmap["pointmap_scale"]],
                    dtype=np.float64,
                ),
            ]
        )
        if not np.isfinite(values).all():
            raise RuntimeError(f"Non-finite V20 candidate: {case_name}")
        if not (0.35 <= float(geometry["old_scale"]) <= 3.0):
            raise RuntimeError(f"Invalid old shot scale: {case_name}")
        if not (0.35 <= float(geometry["new_scale"]) <= 3.0):
            raise RuntimeError(f"Invalid new shot scale: {case_name}")
        if not (0.70 <= float(pointmap["pointmap_scale"]) <= 1.30):
            raise RuntimeError(f"Invalid pointmap residual scale: {case_name}")
        rows.append(
            {
                "case_name": case_name,
                "source": base["source"],
                "old_shot_metric_scale": float(geometry["old_scale"]),
                "new_shot_metric_scale": float(geometry["new_scale"]),
                "boundary_transform": transform.astype(float).tolist(),
                "background_pointmap_depth_scale": float(pointmap["pointmap_scale"]),
                "background_refinement_accepted": bool(pointmap["fit"].get("accepted", True)),
                "application": {
                    "camera_translation": "multiply by new_shot_metric_scale, then apply boundary_transform",
                    "smplx_root_translation": "multiply by new_shot_metric_scale, then apply boundary_transform",
                    "background_pointmap_depth": "multiply by new_shot_metric_scale and background_pointmap_depth_scale, then apply boundary_transform",
                    "rotation": "apply boundary_transform rotation without scale",
                },
            }
        )
        audits.append(audit)
    report = {
        "experiment": "V20 selected strict-causal explicit geometry bridge",
        "case_count": len(rows),
        "protocol": {
            "hard_reset": True,
            "post_cut_frames": 1,
            "da3_inference": "independent single frame at each shot start",
            "rotation": "V16 torso motion, 45-degree residual bound",
            "one_boundary_se3_per_shot": True,
            "normal_frames_unchanged": True,
            "learned_gate_or_selector": False,
        },
        "audit": {
            "max_rotation_determinant_error": float(
                max(abs(row["determinant"] - 1.0) for row in audits)
            ),
            "max_rotation_orthogonality_error": float(
                max(row["orthogonality_max_abs"] for row in audits)
            ),
            "max_bottom_row_error": float(max(row["bottom_row_max_abs"] for row in audits)),
            "accepted_background_refinement_rate": float(
                np.mean([row["background_refinement_accepted"] for row in rows])
            ),
        },
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f">> wrote {args.output}")
    print(json.dumps(report["audit"], indent=2))


if __name__ == "__main__":
    main()
