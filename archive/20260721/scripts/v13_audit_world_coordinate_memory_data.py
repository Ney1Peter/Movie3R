#!/usr/bin/env python3
"""Audit V13 data sources for physically supervised world-coordinate memory probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path("/data/wangzheng/iJCV-CODE/data")
DEFAULT_OUTPUT = REPO_ROOT / "output" / "v13_world_coordinate_memory" / "data_audit.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def nonempty_directory(path: Path) -> bool:
    return path.is_dir() and next(path.iterdir(), None) is not None


def source_row(
    name: str,
    root: Path,
    camera_examples: tuple[Path, ...],
    mask_examples: tuple[Path, ...],
    depth_directories: tuple[Path, ...] = (),
    mesh_examples: tuple[Path, ...] = (),
) -> dict:
    available_depth_dirs = [str(path) for path in depth_directories if nonempty_directory(path)]
    available_meshes = [str(path) for path in mesh_examples if path.is_file()]
    available_cameras = [str(path) for path in camera_examples if path.is_file()]
    available_masks = [str(path) for path in mask_examples if path.is_file()]
    true_scene_geometry = bool(available_depth_dirs or available_meshes)
    return {
        "source": name,
        "root": str(root),
        "exists": root.is_dir(),
        "camera_examples": available_cameras,
        "mask_examples": available_masks,
        "nonempty_depth_directories": available_depth_dirs,
        "mesh_or_scan_examples": available_meshes,
        "true_scene_coordinate_oracle_available": true_scene_geometry and bool(available_cameras),
        "offline_teacher_pseudo_oracle_available": root.is_dir() and bool(available_cameras),
    }


def main() -> None:
    args = parse_args()
    sources = [
        source_row(
            "avatarrex",
            args.data_root / "Training",
            (
                args.data_root / "Training" / "lbn1" / "22053912" / "cam" / "00001645.npz",
                args.data_root / "AvatarReX_raw_meta" / "lbn1" / "calibration_full.json",
            ),
            (args.data_root / "Training" / "lbn1" / "22053912" / "mask" / "00001645.png",),
            tuple((args.data_root / "Training" / "lbn2").glob("*/depth")),
        ),
        source_row(
            "thuman",
            args.data_root / "Training",
            (args.data_root / "Training" / "thuman00" / "cam19" / "cam" / "00002461.npz",),
            (args.data_root / "Training" / "thuman00" / "cam19" / "mask" / "00002461.png",),
        ),
        source_row(
            "mvhuman",
            args.data_root / "MVHuman",
            (args.data_root / "MVHuman" / "200001" / "annots" / "22327117" / "0105_img.json",),
            (args.data_root / "MVHuman" / "200001" / "fmask_lr" / "22327117" / "0105_img_fmask.png",),
        ),
        source_row(
            "rich",
            args.data_root / "RICH",
            (),
            (),
        ),
        source_row(
            "bedlam21",
            args.data_root / "BEDLAM" / "21",
            (
                args.data_root / "BEDLAM" / "21" / "cam" / "seq_000021_camera.csv",
                args.data_root / "BEDLAM" / "21" / "20221010_3-10_500_batch01hand_zoom_suburb_d_6fps.npz",
            ),
            (args.data_root / "BEDLAM" / "21" / "mask" / "seq_000021" / "seq_000021_0006_env.png",),
        ),
    ]
    report = {
        "experiment": "V13 world-coordinate memory data audit",
        "sources": sources,
        "true_oracle_sources": [row["source"] for row in sources if row["true_scene_coordinate_oracle_available"]],
        "pseudo_oracle_sources": [row["source"] for row in sources if row["offline_teacher_pseudo_oracle_available"]],
        "conclusion": (
            "true_static_scene_depth_or_scan_unavailable_in_current_v13_sources"
            if not any(row["true_scene_coordinate_oracle_available"] for row in sources)
            else "true_scene_coordinate_subset_available"
        ),
        "protocol": {
            "true_oracle": "GT depth or scan plus calibrated camera; not substituted by model predictions",
            "same_view_pseudo_oracle": "warmed same-camera Human3R teacher pointmap anchored by GT camera",
            "historical_memory_pseudo_oracle": "pre-cut Human3R world anchors; correspondence assigned by offline teacher and GT camera",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
