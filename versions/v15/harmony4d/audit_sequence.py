#!/usr/bin/env python3
"""Create a compact, evaluator-only schema/calibration audit for one capture."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from versions.v15.harmony4d.dataset import (
    frame_numbers,
    image_path,
    infer_fps,
    load_exo_calibrations,
    load_gt_people,
    locate_sequence_root,
    project_fisheye,
    projected_visibility,
)
from versions.v15.harmony4d.protocol import select_balanced_pairs
from versions.v15.harmony4d.topology import CommonTopology


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extracted-root", type=Path, required=True)
    parser.add_argument("--archive-entry", required=True)
    parser.add_argument(
        "--capture-relative",
        default=None,
        help="Capture path relative to extracted root; required when an archive contains multiple captures",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlay-cameras", default="cam01,cam08,cam15")
    return parser.parse_args()


def finite(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(item) for item in value]
    if isinstance(value, np.ndarray):
        return finite(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def camera_roundtrip(calibration, points_world: np.ndarray) -> dict[str, float | int]:
    pixels, depth_valid = project_fisheye(points_world, calibration)
    inside = depth_valid & np.isfinite(pixels).all(axis=1)
    inside &= (pixels[:, 0] >= 0) & (pixels[:, 0] < calibration.width)
    inside &= (pixels[:, 1] >= 0) & (pixels[:, 1] < calibration.height)
    indices = np.flatnonzero(inside)[::25]
    if not len(indices):
        return {"count": 0, "normalized_ray_median": float("nan"), "normalized_ray_p95": float("nan")}
    undistorted = cv2.fisheye.undistortPoints(
        pixels[indices, None, :], calibration.intrinsic, calibration.distortion
    )[:, 0]
    w2c = calibration.world_to_camera
    camera = points_world[indices] @ w2c[:3, :3].T + w2c[:3, 3]
    target = camera[:, :2] / camera[:, 2:3]
    error = np.linalg.norm(undistorted - target, axis=1)
    return {
        "count": int(len(error)),
        "normalized_ray_median": float(np.median(error)),
        "normalized_ray_p95": float(np.percentile(error, 95)),
    }


def overlay(path: Path, calibration, people: dict[str, dict], topology: CommonTopology, output: Path) -> None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    colours = [(32, 220, 32), (32, 128, 255), (255, 64, 64), (220, 32, 220)]
    for person_index, (identity, person) in enumerate(sorted(people.items())):
        colour = colours[person_index % len(colours)]
        vertices = np.asarray(person["vertices"], dtype=np.float64)
        pixels, valid_depth = project_fisheye(vertices, calibration)
        visible = valid_depth & np.isfinite(pixels).all(axis=1)
        visible &= (pixels[:, 0] >= 0) & (pixels[:, 0] < calibration.width)
        visible &= (pixels[:, 1] >= 0) & (pixels[:, 1] < calibration.height)
        for point in np.rint(pixels[np.flatnonzero(visible)[::20]]).astype(np.int32):
            cv2.circle(image, tuple(point), 2, colour, -1, lineType=cv2.LINE_AA)
        joints = topology.joints_from_smpl(vertices)
        joint_pixels, joint_valid = project_fisheye(joints, calibration)
        for point in np.rint(joint_pixels[joint_valid]).astype(np.int32):
            if 0 <= point[0] < calibration.width and 0 <= point[1] < calibration.height:
                cv2.circle(image, tuple(point), 8, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        visible_pixels = pixels[visible]
        if len(visible_pixels):
            lo = np.floor(visible_pixels.min(axis=0)).astype(int)
            hi = np.ceil(visible_pixels.max(axis=0)).astype(int)
            cv2.rectangle(image, tuple(lo), tuple(hi), colour, 5)
            cv2.putText(image, identity, (int(lo[0]), max(30, int(lo[1]) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, colour, 3)
    output.parent.mkdir(parents=True, exist_ok=True)
    resized = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_AREA)
    if not cv2.imwrite(str(output), resized):
        raise RuntimeError(f"Failed to save {output}")


def main() -> None:
    args = parse_args()
    extracted_root = args.extracted_root.resolve()
    if args.capture_relative is None:
        sequence_root = locate_sequence_root(extracted_root)
    else:
        sequence_root = (extracted_root / args.capture_relative).resolve()
        if extracted_root not in sequence_root.parents or not (
            sequence_root / "colmap/workplace/cameras.txt"
        ).is_file():
            raise ValueError(f"Invalid capture below extracted root: {args.capture_relative}")
    calibrations = load_exo_calibrations(sequence_root)
    frames = frame_numbers(sequence_root, sorted(calibrations))
    if len(frames) < 150:
        raise ValueError(f"Sequence is too short for H4D-CS150: {len(frames)} frames")
    audit_frame = frames[len(frames) // 2]
    people = load_gt_people(sequence_root, audit_frame)
    topology = CommonTopology.load()
    identity_rows = {}
    for identity, person in sorted(people.items()):
        regressed = topology.joints_from_smpl(person["vertices"])
        stored = np.asarray(person["joints_stored"], dtype=np.float64)[:24]
        errors = np.linalg.norm(regressed - stored, axis=1)
        identity_rows[identity] = {
            "vertices": list(person["vertices"].shape),
            "stored_joints": list(person["joints_stored"].shape),
            "regressed_vs_stored_joint_median_m": float(np.median(errors)),
            "regressed_vs_stored_joint_p95_m": float(np.percentile(errors, 95)),
        }
    camera_rows = {}
    visibility_score = {}
    all_vertices = np.concatenate([person["vertices"] for person in people.values()])
    for name, calibration in calibrations.items():
        per_person = {
            identity: projected_visibility(person["vertices"], calibration)
            for identity, person in sorted(people.items())
        }
        visibility_score[name] = min(
            row["visible_vertex_fraction"] for row in per_person.values()
        )
        camera_rows[name] = {
            **calibration.jsonable(),
            "audit_frame": audit_frame,
            "people_projection": per_person,
            "projection_roundtrip": camera_roundtrip(calibration, all_vertices),
            "inverse_roundtrip_max_abs": float(
                np.max(np.abs(calibration.world_to_camera @ calibration.camera_to_world - np.eye(4)))
            ),
        }
    selected_pairs = select_balanced_pairs(calibrations, visibility_score)
    reprojection_medians = np.asarray(
        [calibration.reprojection_median_px for calibration in calibrations.values()],
        dtype=np.float64,
    )
    reprojection_p95s = np.asarray(
        [calibration.reprojection_p95_px for calibration in calibrations.values()],
        dtype=np.float64,
    )
    projection_thresholds = {"median_px": 5.0, "p95_px": 15.0}
    projection_pass = bool(
        np.isfinite(reprojection_medians).all()
        and np.isfinite(reprojection_p95s).all()
        and np.all(reprojection_medians <= projection_thresholds["median_px"])
        and np.all(reprojection_p95s <= projection_thresholds["p95_px"])
    )
    overlays = []
    requested = [value.strip() for value in args.overlay_cameras.split(",") if value.strip()]
    for name in requested:
        if name not in calibrations:
            continue
        destination = (
            args.output.resolve().parent
            / "overlays"
            / f"{Path(args.archive_entry).stem}_{sequence_root.name}_{name}_f{audit_frame:05d}.jpg"
        )
        overlay(image_path(sequence_root, name, audit_frame), calibrations[name], people, topology, destination)
        overlays.append(str(destination))
    report = {
        "schema_version": "Harmony4D-Movie3R-index-v1",
        "archive_entry": args.archive_entry,
        "sequence_root_name": sequence_root.name,
        "capture_group_name": sequence_root.parent.name,
        "capture_relative": str(sequence_root.relative_to(extracted_root)),
        "frame_count": len(frames),
        "frame_min": min(frames),
        "frame_max": max(frames),
        "frames_contiguous": frames == list(range(min(frames), max(frames) + 1)),
        "fps": infer_fps(sequence_root),
        "camera_count": len(calibrations),
        "identities": sorted(people),
        "person_count_at_audit": len(people),
        "audit_frame": audit_frame,
        "units": "metres",
        "camera_convention": "COLMAP world_to_camera; adapter also stores inverse camera_to_world",
        "published_smpl_world_identity": sorted({camera.canonical_world for camera in calibrations.values()}),
        "extrinsic_sources": {
            source: sum(camera.extrinsic_source == source for camera in calibrations.values())
            for source in sorted({camera.extrinsic_source for camera in calibrations.values()})
        },
        "projection_audit": {
            "thresholds": projection_thresholds,
            "camera_count": len(calibrations),
            "pnp_fallback_count": sum(
                camera.extrinsic_source.endswith("static_pnp")
                for camera in calibrations.values()
            ),
            "median_px_across_cameras": float(np.median(reprojection_medians)),
            "max_camera_median_px": float(np.max(reprojection_medians)),
            "median_camera_p95_px": float(np.median(reprojection_p95s)),
            "max_camera_p95_px": float(np.max(reprojection_p95s)),
            "pass": projection_pass,
        },
        "distortion_model": "OPENCV_FISHEYE",
        "gt_runtime_isolation": "index/evaluator only",
        "topology": topology.metadata(),
        "identity_topology_audit": identity_rows,
        "cameras": camera_rows,
        "selected_protocol_pairs": selected_pairs,
        "overlays": overlays,
    }
    if not projection_pass:
        raise ValueError(
            "Harmony4D projection audit failed: "
            f"camera median max={np.max(reprojection_medians):.3f}px, "
            f"camera P95 max={np.max(reprojection_p95s):.3f}px"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(finite(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "frames": len(frames),
        "cameras": len(calibrations),
        "identities": sorted(people),
        "pairs": selected_pairs,
        "overlays": overlays,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
