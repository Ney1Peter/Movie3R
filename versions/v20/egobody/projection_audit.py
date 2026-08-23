#!/usr/bin/env python3
"""Render evaluator-only SMPL overlays to validate the EgoBody transform chain."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np


COLORS = ((40, 80, 255), (40, 220, 80))  # BGR


def jsonl_by_case(path: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    output = {str(row["case_id"]): row for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"duplicate case IDs in {path}")
    return output


def safe_path(root: Path, relative: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise ValueError(relative)
    result = (root.resolve() / value).resolve()
    if root.resolve() not in result.parents:
        raise ValueError(relative)
    return result


def project(vertices_world: np.ndarray, c2w: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotation = np.asarray(c2w, dtype=np.float64)[:3, :3]
    centre = np.asarray(c2w, dtype=np.float64)[:3, 3]
    camera = (np.asarray(vertices_world, dtype=np.float64) - centre) @ rotation
    pixels_h = camera @ np.asarray(intrinsic, dtype=np.float64).T
    pixels = pixels_h[:, :2] / np.maximum(pixels_h[:, 2:3], 1e-12)
    return pixels, camera[:, 2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--evaluator-manifest", type=Path, required=True)
    parser.add_argument("--staged-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--indices", default="0,74,75,149")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = jsonl_by_case(args.runtime_manifest)[args.case_id]
    evaluator = jsonl_by_case(args.evaluator_manifest)[args.case_id]
    indices = [int(value) for value in args.indices.split(",") if value.strip()]
    with np.load(args.gt_root / f"{args.case_id}.gt.npz", allow_pickle=False) as cache:
        vertices = np.asarray(cache["vertices_world"])
        gt_cameras = np.asarray(cache["cameras_c2w"])
        frames = np.asarray(cache["frames"])
    members = list(runtime["image_members"])
    cameras = (
        [str(runtime["pre_camera"])] * len(runtime["pre_frame_numbers"])
        + [str(runtime["post_camera"])] * len(runtime["post_frame_numbers"])
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in indices:
        if index < 0 or index >= len(members):
            raise IndexError(index)
        image_path = safe_path(args.staged_root, members[index])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"could not decode {image_path}")
        calibration = evaluator["camera_calibration_evaluator_only"][cameras[index]]
        intrinsic = np.asarray(calibration["intrinsic"], dtype=np.float64)
        distortion = np.asarray(calibration["distortion"], dtype=np.float64)
        image = cv2.undistort(image, intrinsic, distortion)
        c2w = np.asarray(calibration["camera_to_world"], dtype=np.float64)
        if not np.allclose(c2w, gt_cameras[index], atol=1e-5):
            raise ValueError(f"GT/evaluator camera mismatch at {index}")
        people_rows = []
        for person in range(vertices.shape[1]):
            pixels, depth = project(vertices[index, person], c2w, intrinsic)
            finite = np.isfinite(pixels).all(axis=1) & (depth > 0)
            inside = finite & (pixels[:, 0] >= 0) & (pixels[:, 0] < image.shape[1]) & (pixels[:, 1] >= 0) & (pixels[:, 1] < image.shape[0])
            points = np.rint(pixels[inside]).astype(np.int32)
            if len(points):
                for x, y in points[::20]:
                    cv2.circle(image, (int(x), int(y)), 1, COLORS[person % len(COLORS)], -1, cv2.LINE_AA)
                low, high = points.min(axis=0), points.max(axis=0)
                cv2.rectangle(image, tuple(low), tuple(high), COLORS[person % len(COLORS)], 2)
            people_rows.append({
                "person": person,
                "positive_depth_fraction": float((depth > 0).mean()),
                "inside_fraction": float(inside.mean()),
                "inside_vertex_count": int(inside.sum()),
            })
        output = args.output_dir / f"{args.case_id}_i{index:03d}_f{int(frames[index]):05d}.jpg"
        partial = output.with_suffix(output.suffix + ".partial.jpg")
        if not cv2.imwrite(str(partial), image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise OSError(partial)
        os.replace(partial, output)
        rows.append({
            "clip_index": index,
            "source_frame": int(frames[index]),
            "camera": cameras[index],
            "image": str(image_path.resolve()),
            "overlay": str(output.resolve()),
            "people": people_rows,
        })
    report = {
        "schema_version": "Bridge3R-EgoBody-projection-audit-v1",
        "case_id": args.case_id,
        "coordinate_chain": "C12 SMPL-X -> scene world; world -> selected Kinect color",
        "undistortion": "cv2.undistort with official Color.json",
        "rows": rows,
    }
    report_path = args.output_dir / f"{args.case_id}.projection_audit.json"
    partial_report = report_path.with_suffix(report_path.suffix + ".partial")
    partial_report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(partial_report, report_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
