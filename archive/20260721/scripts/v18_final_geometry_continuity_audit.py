#!/usr/bin/env python3
"""Audit visible Human3R root continuity after V18/DA3 boundary transforms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v10-report",
        type=Path,
        default=REPO_ROOT
        / "output"
        / "v10_candidate_selection"
        / "oracle_gt_4source"
        / "oracle_candidate_selection_metrics.json",
    )
    parser.add_argument(
        "--v18-report",
        type=Path,
        default=REPO_ROOT
        / "output"
        / "v18_human_metric_translation"
        / "final_candidates"
        / "v18_human_metric_translation_eval.json",
    )
    parser.add_argument(
        "--da3-report",
        type=Path,
        default=REPO_ROOT
        / "output"
        / "v18_human_metric_translation"
        / "da3_metric_depth"
        / "v18_da3_metric_depth_probe.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "output"
        / "v18_human_metric_translation"
        / "geometry_continuity_audit"
        / "v18_final_geometry_continuity_audit.json",
    )
    parser.add_argument("--boundary", type=int, default=2)
    return parser.parse_args()


def case_map(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["case_name"]): row for row in payload["cases"]}


def predicted_root_world(local_dir: Path, frame: int) -> np.ndarray:
    with np.load(local_dir / "camera" / f"{frame:06d}.npz") as camera:
        pose = np.asarray(camera["pose"], dtype=np.float32)
    with np.load(local_dir / "smpl" / f"{frame:06d}.npz", allow_pickle=True) as smpl:
        if len(smpl["transl"]) == 0:
            raise ValueError(f"Missing SMPL-X person in {local_dir}, frame {frame}")
        root_camera = np.asarray(smpl["transl"][0], dtype=np.float32)
    return (pose[:3, :3] @ root_camera + pose[:3, 3]).astype(np.float32)


def transformed_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    return transform[:3, :3] @ point + transform[:3, 3]


def summarize(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "mean_m": float(np.mean(array)),
        "median_m": float(np.median(array)),
        "p90_m": float(np.quantile(array, 0.90)),
        "p95_m": float(np.quantile(array, 0.95)),
        "over_1m_rate": float(np.mean(array > 1.0)),
    }


def main() -> None:
    args = parse_args()
    v10 = case_map(args.v10_report)
    v18 = case_map(args.v18_report)
    da3 = case_map(args.da3_report)
    rows = []

    for case_name, case in v18.items():
        if case_name not in v10 or case_name not in da3:
            continue
        local_dir = Path(v10[case_name]["paths"]["human3r_local_reset"])
        pre_root = predicted_root_world(local_dir, int(args.boundary) - 1)
        post_root = predicted_root_world(local_dir, int(args.boundary))
        candidates = {
            "hard_reset": np.eye(4, dtype=np.float32),
            "fixed_explicit": np.asarray(
                case["candidates"]["fixed_explicit"]["transform"], dtype=np.float32
            ),
            "v18_human_projection": np.asarray(
                case["candidates"]["human_no_calibration"]["transform"], dtype=np.float32
            ),
            "da3_metric_camera": np.asarray(
                da3[case_name]["candidates"]["da3_pelvis_depth"]["transform"],
                dtype=np.float32,
            ),
            "boundary_oracle": np.asarray(
                case["candidates"]["boundary_oracle"]["transform"], dtype=np.float32
            ),
        }
        jumps = {
            name: float(np.linalg.norm(transformed_point(transform, post_root) - pre_root))
            for name, transform in candidates.items()
        }
        rows.append(
            {
                "case_name": case_name,
                "source": str(case["source"]),
                "visible_human3r_root_jump_m": jumps,
            }
        )

    methods = tuple(rows[0]["visible_human3r_root_jump_m"])
    overall = {
        method: summarize([row["visible_human3r_root_jump_m"][method] for row in rows])
        for method in methods
    }
    sources = sorted({row["source"] for row in rows})
    by_source = {
        source: {
            method: summarize(
                [
                    row["visible_human3r_root_jump_m"][method]
                    for row in rows
                    if row["source"] == source
                ]
            )
            for method in methods
        }
        for source in sources
    }
    payload = {
        "metric_definition": (
            "Distance between the last pre-cut and first post-cut Human3R-predicted "
            "SMPL-X translation roots after applying one candidate Boundary SE(3) to "
            "the post-cut root. This includes real one-frame human motion and is a "
            "final-output continuity diagnostic, not camera-pose GT error."
        ),
        "overall": overall,
        "by_source": by_source,
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(overall, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
