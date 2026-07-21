#!/usr/bin/env python3
"""Export the selected V24 transform together with the persistent V22 scale state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V24 = (
    REPO_ROOT
    / "output"
    / "v24_vggt_v22_rotation_bridge"
    / "v24_vggt_v22_rotation_bridge.json"
)
DEFAULT_V22_EXPORT = (
    REPO_ROOT
    / "output"
    / "v22_explicit_metric_bridge"
    / "selected_candidates"
    / "v22_selected_explicit_bridge.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "output" / "v24_vggt_v22_rotation_bridge" / "selected_candidates"
SELECTED = "safe_tiered_extension_vggt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v24_report", type=Path, default=DEFAULT_V24)
    parser.add_argument("--v22_export", type=Path, default=DEFAULT_V22_EXPORT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    v24 = load(args.v24_report)
    v22 = load(args.v22_export)
    scale_cases = {row["case_name"]: row for row in v22["cases"]}
    rows = []
    for case in v24["cases"]:
        name = case["case_name"]
        scale = scale_cases[name]
        diagnostics = case["diagnostics"]
        selected = case["variants"][SELECTED]
        vggt_required = diagnostics["torso_residual_deg"] >= 10.0
        if diagnostics["trigger_safe_low_texture_conflict"]:
            accepted_rule = "low_texture_conflict_cap45"
        elif diagnostics["trigger_safe_large_residual"]:
            accepted_rule = "large_torso_residual_cap25"
        elif diagnostics["trigger_safe_consensus"]:
            accepted_rule = "torso_vggt_consensus_cap60"
        else:
            accepted_rule = "keep_v22"
        rows.append(
            {
                "case_name": name,
                "source": case["source"],
                "boundary_transform": selected["transform"],
                "old_shot_human_camera_scale": scale["old_shot_human_camera_scale"],
                "new_shot_human_camera_scale": scale["new_shot_human_camera_scale"],
                "old_shot_background_depth_scale": scale["old_shot_background_depth_scale"],
                "new_shot_background_depth_scale": scale["new_shot_background_depth_scale"],
                "gravity_residual_accepted": scale["gravity_residual_accepted"],
                "vggt_required": vggt_required,
                "vggt_rotation_accepted": accepted_rule != "keep_v22",
                "accepted_rotation_rule": accepted_rule,
                "rotation_diagnostics": diagnostics,
                "evaluation": {
                    "camera": selected["camera"],
                    "human": selected["human"],
                    "scene": selected["scene"],
                },
            }
        )
    if len(rows) != 180 or len({row["case_name"] for row in rows}) != 180:
        raise RuntimeError("V24 export must contain 180 unique cases")
    for row in rows:
        transform = np.asarray(row["boundary_transform"], dtype=np.float64)
        rotation = transform[:3, :3]
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise RuntimeError(f"Invalid transform for {row['case_name']}")
        if np.max(np.abs(rotation.T @ rotation - np.eye(3))) > 1e-4:
            raise RuntimeError(f"Non-orthogonal rotation for {row['case_name']}")
        if abs(np.linalg.det(rotation) - 1.0) > 1e-4:
            raise RuntimeError(f"Invalid rotation determinant for {row['case_name']}")
        if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
            raise RuntimeError(f"Invalid homogeneous row for {row['case_name']}")
    payload = {
        "experiment": "V24 selected safe conditional wide-rotation metric bridge",
        "case_count": len(rows),
        "runtime_order": [
            "hard reset Human3R",
            "compute V22 DA3 metric root/background scales",
            "compute V16 torso-motion rotation and safe gravity residual",
            "run frozen VGGT 1+1 only when torso residual is at least 10 degrees",
            "apply fixed extension/consensus/low-texture safety rules",
            "re-solve camera translation from the metric human-root equation",
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
        "vggt_required_rate": float(np.mean([row["vggt_required"] for row in rows])),
        "vggt_acceptance_rate": float(
            np.mean([row["vggt_rotation_accepted"] for row in rows])
        ),
        "cases": rows,
    }
    output = args.output_dir / "v24_selected_rotation_bridge.json"
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f">> wrote {output}")


if __name__ == "__main__":
    main()
