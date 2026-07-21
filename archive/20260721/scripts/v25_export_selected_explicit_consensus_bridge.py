#!/usr/bin/env python3
"""Export the conservative selected V25 1+1 explicit-consensus bridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V25 = (
    REPO_ROOT
    / "output"
    / "v25_explicit_consensus_bridge"
    / "v25_explicit_consensus_bridge_probe.json"
)
DEFAULT_V24_EXPORT = (
    REPO_ROOT
    / "output"
    / "v24_vggt_v22_rotation_bridge"
    / "selected_candidates"
    / "v24_selected_rotation_bridge.json"
)
DEFAULT_V22 = (
    REPO_ROOT
    / "output"
    / "v22_explicit_metric_bridge"
    / "final_seed1"
    / "v22_explicit_metric_bridge.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "output" / "v25_explicit_consensus_bridge" / "selected_candidates"
SELECTED = "v25_1p1_rotation_scene_margin010"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v25_report", type=Path, default=DEFAULT_V25)
    parser.add_argument("--v24_export", type=Path, default=DEFAULT_V24_EXPORT)
    parser.add_argument("--v22_report", type=Path, default=DEFAULT_V22)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    v25 = load(args.v25_report)
    v24 = {row["case_name"]: row for row in load(args.v24_export)["cases"]}
    v22 = {row["case_name"]: row for row in load(args.v22_report)["cases"]}
    rows = []
    for case in v25["cases"]:
        name = case["case_name"]
        selected = case["variants"][SELECTED]
        inherited = v24[name]
        scale_rule = selected["scene_scale_rule"]
        scene_scales = v22[name]["scene_scale_sets"][scale_rule]
        diagnostics = case["diagnostics"]
        if diagnostics["trigger_low_torso_explicit_consensus"]:
            final_rotation_rule = "low_torso_camera_correspondence_consensus_cap60"
        elif diagnostics["trigger_background_1p1_fallback"]:
            final_rotation_rule = "background_1p1_fallback_cap60"
        else:
            final_rotation_rule = inherited["accepted_rotation_rule"]
        rows.append(
            {
                "case_name": name,
                "source": case["source"],
                "boundary_transform": selected["transform"],
                "old_shot_human_camera_scale": inherited[
                    "old_shot_human_camera_scale"
                ],
                "new_shot_human_camera_scale": inherited[
                    "new_shot_human_camera_scale"
                ],
                "old_shot_background_depth_scale": float(scene_scales["old"]),
                "new_shot_background_depth_scale": float(scene_scales["new"]),
                "gravity_residual_accepted": inherited["gravity_residual_accepted"],
                "vggt_required": inherited["vggt_required"],
                "v24_rotation_accepted": inherited["vggt_rotation_accepted"],
                "v25_background_fallback_accepted": diagnostics[
                    "trigger_background_1p1_fallback"
                ],
                "v25_explicit_consensus_accepted": diagnostics[
                    "trigger_low_torso_explicit_consensus"
                ],
                "accepted_rotation_rule": final_rotation_rule,
                "scene_scale_rule": scale_rule,
                "scene_scale_improvement_margin_m": 0.10,
                "diagnostics": diagnostics,
                "evaluation": {
                    "camera": selected["camera"],
                    "human": selected["human"],
                    "scene": selected["scene"],
                },
            }
        )

    if len(rows) != 180 or len({row["case_name"] for row in rows}) != 180:
        raise RuntimeError("V25 export must contain 180 unique cases")
    for row in rows:
        transform = np.asarray(row["boundary_transform"], dtype=np.float64)
        rotation = transform[:3, :3]
        scales = np.asarray(
            [
                row["old_shot_human_camera_scale"],
                row["new_shot_human_camera_scale"],
                row["old_shot_background_depth_scale"],
                row["new_shot_background_depth_scale"],
            ],
            dtype=np.float64,
        )
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise RuntimeError(f"Invalid transform for {row['case_name']}")
        if np.max(np.abs(rotation.T @ rotation - np.eye(3))) > 1e-4:
            raise RuntimeError(f"Non-orthogonal rotation for {row['case_name']}")
        if abs(np.linalg.det(rotation) - 1.0) > 1e-4:
            raise RuntimeError(f"Invalid determinant for {row['case_name']}")
        if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
            raise RuntimeError(f"Invalid homogeneous row for {row['case_name']}")
        if not np.isfinite(scales).all() or np.any(scales <= 0.0):
            raise RuntimeError(f"Invalid scale state for {row['case_name']}")

    payload = {
        "experiment": "V25 selected conservative explicit-consensus metric bridge",
        "case_count": len(rows),
        "selected_variant": SELECTED,
        "runtime_order": [
            "hard reset Human3R",
            "compute V22 DA3 metric human/background scales",
            "compute V16 torso-motion rotation and safe gravity residual",
            "run V24 full-RGB 1+1 VGGT when pretriggered",
            "if rejected-large-torso, optionally run background-only 1+1 VGGT",
            "if low-torso, require VGGT camera/correspondence explicit consensus",
            "re-solve metric camera translation after the final rotation",
            "use identity10 scene scale only when 1+1 residual improves by more than 0.10 m",
            "store one fixed shot-level SE(3) and scale state",
        ],
        "constraints": {
            "post_cut_frames": 1,
            "future_shot_access": False,
            "learned_gate": False,
            "source_specific_rule": False,
            "gt_runtime_information": False,
            "camera_pointmap_smplx_share_transform": True,
        },
        "trigger_counts": {
            "v24_rotation": int(sum(row["v24_rotation_accepted"] for row in rows)),
            "background_1p1_fallback": int(
                sum(row["v25_background_fallback_accepted"] for row in rows)
            ),
            "low_torso_explicit_consensus": int(
                sum(row["v25_explicit_consensus_accepted"] for row in rows)
            ),
            "identity10_scene_margin010": int(
                sum(row["scene_scale_rule"] == "identity10" for row in rows)
            ),
        },
        "cases": rows,
    }
    output = args.output_dir / "v25_selected_explicit_consensus_bridge.json"
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f">> wrote {output}")


if __name__ == "__main__":
    main()
