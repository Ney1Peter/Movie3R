#!/usr/bin/env python3
"""Freeze and materialise the held-out MVHuman viewpoint-cut protocol.

Case selection is deliberately independent of model predictions.  For every
subject and angle stratum, the camera pair closest to the predeclared stratum
centre is selected, with a lexicographic tie break.  A centred 150-frame
window is then used.  Ground truth, camera identifiers, angles, and masks are
written only to the evaluator manifest; the runtime manifest contains only a
derived RGB video and its temporal shape.

The uploaded ZIPs contain one ``tar.gz`` per subject.  This script never
extracts a complete subject: it decompresses each subject archive once and
materialises only RGB/mask members needed by the frozen cases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter


SCHEMA = "Bridge3R-MVHuman-Heldout-MVH150-protocol-v2"
EXPECTED_FRAMES = 150
CUT_INDEX = 74
DERIVED_FPS = 30
SEED = 20260830
ANGLE_STRATA = (
    ("small", 0.0, 60.0, 45.0),
    ("medium", 60.0, 90.0, 75.0),
    ("large", 90.0, 120.0, 105.0),
    ("very_large", 120.0, 150.0, 135.0),
    ("extreme", 150.0, 180.000001, 165.0),
)
BODY25_TO_COMMON12 = np.asarray([5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11], dtype=np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def camera_name(key: str) -> str:
    return key.rsplit(".", 1)[0].split("_")[-1]


def camera_rotation_geodesic_deg(first: dict[str, Any], second: dict[str, Any]) -> float:
    r_first = np.asarray(first["rotation"], dtype=np.float64)
    r_second = np.asarray(second["rotation"], dtype=np.float64)
    relative = r_first @ r_second.T
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def optical_axis_angle_deg(first: dict[str, Any], second: dict[str, Any]) -> float:
    """Return the angle between the calibrated camera optical axes.

    MVHuman stores world-to-camera rotations.  The camera-to-world optical
    axis is therefore ``R.T @ [0, 0, 1]``.  This diagnostic is recorded next
    to the SO(3) geodesic used by the three multi-person protocols, but it is
    not used to select cases because the 16-camera rig has no >=150-degree
    optical-axis pair.
    """
    axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    r_first = np.asarray(first["rotation"], dtype=np.float64)
    r_second = np.asarray(second["rotation"], dtype=np.float64)
    first_axis = r_first.T @ axis
    second_axis = r_second.T @ axis
    cosine = float(np.clip(np.dot(first_axis, second_axis), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def cameras_by_name(extrinsics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for key, value in extrinsics.items():
        name = camera_name(key)
        if name in result:
            raise ValueError(f"Duplicate normalized camera name {name}")
        result[name] = value
    return result


def choose_pairs(cameras: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    names = sorted(cameras)
    candidates = []
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            rotation_angle = camera_rotation_geodesic_deg(cameras[first], cameras[second])
            axis_angle = optical_axis_angle_deg(cameras[first], cameras[second])
            candidates.append((first, second, rotation_angle, axis_angle))
    chosen = []
    for stratum, low, high, target in ANGLE_STRATA:
        pool = [item for item in candidates if low <= item[2] < high]
        if not pool:
            raise ValueError(f"No camera pair in angle stratum {stratum}")
        first, second, rotation_angle, axis_angle = min(
            pool, key=lambda item: (abs(item[2] - target), item[0], item[1])
        )
        chosen.append(
            {
                "angle_stratum": stratum,
                "stratum_bounds_deg": [low, min(high, 180.0)],
                "target_angle_deg": target,
                "camera_a": first,
                "camera_b": second,
                # Backwards-compatible alias consumed by the evaluator.  Its
                # exact definition is made explicit by ``angle_measure``.
                "viewpoint_angle_deg": float(rotation_angle),
                "camera_rotation_geodesic_deg": float(rotation_angle),
                "optical_axis_angle_deg": float(axis_angle),
                "angle_measure": "SO(3) geodesic between calibrated camera rotations",
                "selection": "closest SO(3)-geodesic value to predeclared stratum centre; lexicographic tie break",
            }
        )
    return chosen


def c2w(value: dict[str, Any]) -> np.ndarray:
    rotation = np.asarray(value["rotation"], dtype=np.float64)
    translation = np.asarray(value["translation"], dtype=np.float64).reshape(3) / 1000.0
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation.T
    pose[:3, 3] = -(rotation.T @ translation)
    return pose


def frame_name(index: int) -> str:
    return f"{(int(index) + 1) * 5:04d}"


def body25(path: Path, camera_scale: float) -> tuple[np.ndarray, np.ndarray]:
    people = json.loads(path.read_text(encoding="utf-8"))
    if len(people) != 1:
        raise ValueError(f"Expected one annotated person in {path}, found {len(people)}")
    points = np.asarray(people[0]["keypoints3d"], dtype=np.float64)
    if points.shape[0] < 25 or points.shape[1] < 4:
        raise ValueError(f"Invalid BODY25 keypoints in {path}: {points.shape}")
    return points[:25, :3] / float(camera_scale), points[:25, 3]


def safe_extract_members(archive: Path, members: list[str], destination: Path) -> None:
    root = destination.resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name in members:
        if name.startswith("/") or ".." in Path(name).parts:
            raise ValueError(f"Unsafe requested archive member: {name}")
        target = (root / name).resolve()
        if root not in target.parents:
            raise ValueError(f"Archive member escapes destination: {name}")
    # GNU tar reads the gzip stream once.  Python TarFile.extractall after a
    # full getmembers() scan can repeatedly seek/decompress a compressed tar
    # when selected members are far apart, which is prohibitively slow here.
    with tempfile.NamedTemporaryFile("w", encoding="utf-8") as handle:
        handle.write("\n".join(members) + "\n")
        handle.flush()
        command = [
            "tar", "--extract", "--gzip", "--file", str(archive),
            "--directory", str(root), "--files-from", handle.name,
            "--verbatim-files-from", "--no-same-owner", "--no-same-permissions",
        ]
        subprocess.run(command, check=True)


def link_case_frames(case: dict[str, Any], extracted: Path, derived: Path) -> tuple[list[Path], list[Path]]:
    case_dir = derived / "cases" / case["case_id"]
    frames_dir, masks_dir = case_dir / "frames", case_dir / "masks"
    frames_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    rgb_paths, mask_paths = [], []
    for output_index, (camera, source_index) in enumerate(zip(case["camera_timeline"], case["source_frame_indices"])):
        raw = frame_name(source_index)
        rgb = extracted / case["subject"] / "images_lr" / camera / f"{raw}_img.jpg"
        mask = extracted / case["subject"] / "fmask_lr" / camera / f"{raw}_img_fmask.png"
        if not rgb.is_file() or not mask.is_file():
            raise FileNotFoundError(f"Frozen case source missing: {rgb} / {mask}")
        rgb_link = frames_dir / f"frame_{output_index + 1:06d}.jpg"
        mask_link = masks_dir / f"frame_{output_index + 1:06d}.png"
        for source, target in ((rgb, rgb_link), (mask, mask_link)):
            if target.exists() or target.is_symlink():
                target.unlink()
            os.symlink(os.path.relpath(source, target.parent), target)
        rgb_paths.append(rgb_link)
        mask_paths.append(mask_link)
    return rgb_paths, mask_paths


def encode_video(frames_dir: Path, output: Path) -> None:
    if output.is_file():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-framerate", str(DERIVED_FPS), "-i", str(frames_dir / "frame_%06d.jpg"),
        "-frames:v", str(EXPECTED_FRAMES), "-c:v", "libx264", "-preset", "medium",
        "-crf", "15", "-pix_fmt", "yuv420p", str(output),
    ]
    subprocess.run(command, check=True)


def texture_statistics(rgb_paths: list[Path], mask_paths: list[Path]) -> dict[str, Any]:
    gradients, laplacians, valid_fractions = [], [], []
    for rgb_path, mask_path in zip(rgb_paths[::15], mask_paths[::15]):
        gray_image = Image.open(rgb_path).convert("L")
        scale = min(1.0, 512.0 / max(gray_image.size))
        sample_size = (max(1, round(gray_image.width * scale)), max(1, round(gray_image.height * scale)))
        gray = np.asarray(gray_image.resize(sample_size, Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
        mask_image = Image.open(mask_path).convert("L").resize(sample_size, Image.Resampling.NEAREST)
        dilated = np.asarray(mask_image.filter(ImageFilter.MaxFilter(11)), dtype=np.uint8) > 127
        background = ~dilated
        gx = np.zeros_like(gray); gy = np.zeros_like(gray)
        gx[:, 1:-1] = 0.5 * (gray[:, 2:] - gray[:, :-2])
        gy[1:-1, :] = 0.5 * (gray[2:, :] - gray[:-2, :])
        lap = np.zeros_like(gray)
        lap[1:-1, 1:-1] = (
            gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
            - 4.0 * gray[1:-1, 1:-1]
        )
        valid_fractions.append(float(background.mean()))
        if int(background.sum()) < 1024:
            continue
        gradients.append(float(np.sqrt(gx[background] ** 2 + gy[background] ** 2).mean()))
        laplacians.append(float(np.var(lap[background])))
    return {
        "sample_stride_frames": 15,
        "analysis_long_side_pixels": 512,
        "mask_dilation_pixels_at_analysis_scale": 11,
        "mean_normalized_gradient": None if not gradients else float(np.mean(gradients)),
        "mean_laplacian_variance": None if not laplacians else float(np.mean(laplacians)),
        "mean_valid_background_fraction": None if not valid_fractions else float(np.mean(valid_fractions)),
        "sample_count": len(gradients),
    }


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    audit_root, output_root = args.audit_root.resolve(), args.output_root.resolve()
    metadata_root = audit_root / "metadata"
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty protocol root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    extraction: dict[str, set[str]] = defaultdict(set)
    subject_ledger = []
    for subject_dir in sorted(path for path in metadata_root.iterdir() if path.is_dir()):
        subject = subject_dir.name
        archive_group = "test1_archives" if subject.startswith("1") else "test2_archives"
        archive = audit_root / archive_group / f"{subject}.tar.gz"
        extrinsics = json.loads((subject_dir / "camera_extrinsics.json").read_text(encoding="utf-8"))
        cameras = cameras_by_name(extrinsics)
        gt_files = sorted((subject_dir / "smplx" / "keypoints3d").glob("*.json"))
        if len(gt_files) < EXPECTED_FRAMES:
            raise ValueError(f"{subject} has only {len(gt_files)} annotated frames")
        start = (len(gt_files) - EXPECTED_FRAMES) // 2
        source_indices = list(range(start, start + EXPECTED_FRAMES))
        scale = float(pickle.load(open(subject_dir / "camera_scale.pkl", "rb")))
        gt_valid_counts = []
        for index in source_indices:
            _, confidence = body25(subject_dir / "smplx" / "keypoints3d" / f"{index:06d}.json", scale)
            gt_valid_counts.append(int((confidence[BODY25_TO_COMMON12] > 0.2).sum()))
        if min(gt_valid_counts) < 8:
            raise ValueError(f"{subject} centred window has fewer than 8 valid common joints")
        subject_cases = []
        for pair in choose_pairs(cameras):
            case_index = len(cases) + 1
            case_id = f"mvh150_{case_index:04d}"
            camera_timeline = [pair["camera_a"]] * (CUT_INDEX + 1) + [pair["camera_b"]] * (EXPECTED_FRAMES - CUT_INDEX - 1)
            case = {
                "case_id": case_id,
                "dataset": "MVHuman",
                "protocol": "MVH150",
                "role": "test",
                "subject": subject,
                "archive": str(archive),
                "source_frame_start": start,
                "source_frame_indices": source_indices,
                "cut_indices_evaluator_only": [CUT_INDEX],
                "camera_timeline": camera_timeline,
                "camera_a": pair["camera_a"],
                "camera_b": pair["camera_b"],
                "viewpoint_angle_deg": pair["viewpoint_angle_deg"],
                "camera_rotation_geodesic_deg": pair["camera_rotation_geodesic_deg"],
                "optical_axis_angle_deg": pair["optical_axis_angle_deg"],
                "angle_measure": pair["angle_measure"],
                "angle_stratum": pair["angle_stratum"],
                "stratum_bounds_deg": pair["stratum_bounds_deg"],
                "selection": pair["selection"],
                "gt_common12_min_valid_joints": min(gt_valid_counts),
                "gt_common12_mean_valid_joints": float(np.mean(gt_valid_counts)),
            }
            for camera, index in zip(camera_timeline, source_indices):
                raw = frame_name(index)
                extraction[subject].add(f"{subject}/images_lr/{camera}/{raw}_img.jpg")
                extraction[subject].add(f"{subject}/fmask_lr/{camera}/{raw}_img_fmask.png")
            cases.append(case)
            subject_cases.append(case_id)
        subject_ledger.append(
            {
                "subject": subject,
                "archive": str(archive),
                "archive_sha256": sha256(archive),
                "camera_count": len(cameras),
                "annotated_frame_count": len(gt_files),
                "centred_window_start": start,
                "case_ids": subject_cases,
            }
        )

    freeze = {
        "schema_version": SCHEMA,
        "seed": SEED,
        "case_count": len(cases),
        "subject_count": len(subject_ledger),
        "angle_strata": [
            {"name": name, "low_inclusive_deg": low, "high_exclusive_deg": min(high, 180.0), "target_deg": target}
            for name, low, high, target in ANGLE_STRATA
        ],
        "angle_definition": {
            "selection_and_stratification": "SO(3) geodesic between calibrated camera rotations",
            "optical_axis_angle": "recorded as a secondary diagnostic and not used for selection",
            "reason": "matches the three multi-person protocols and preserves balanced strata on both MVHuman camera rigs",
        },
        "temporal_contract": {
            "frames": EXPECTED_FRAMES,
            "cut_index": CUT_INDEX,
            "pre_cut_frames": CUT_INDEX + 1,
            "post_cut_frames": EXPECTED_FRAMES - CUT_INDEX - 1,
            "time": "consecutive synchronized source indices; no temporal jump",
            "derived_video_fps": DERIVED_FPS,
        },
        "selection_independence": "GT completeness, calibrated angle strata, centred time window, and lexicographic tie breaks only; no model output was available.",
        "subjects": subject_ledger,
        "cases": cases,
    }
    freeze["content_sha256"] = canonical_digest(freeze)
    atomic_json(output_root / "protocol_freeze.json", freeze)
    evaluator_manifest = output_root / "manifests" / "test_evaluator.jsonl"
    evaluator_manifest.parent.mkdir(parents=True, exist_ok=True)
    evaluator_manifest.write_text("".join(json.dumps(case, sort_keys=True) + "\n" for case in cases), encoding="utf-8")
    for subject, members in extraction.items():
        member_file = output_root / "extraction_members" / f"{subject}.txt"
        member_file.parent.mkdir(parents=True, exist_ok=True)
        member_file.write_text("\n".join(sorted(members)) + "\n", encoding="utf-8")

    if args.materialize:
        extracted = output_root / "extracted"
        for subject in sorted(extraction):
            archive_group = "test1_archives" if subject.startswith("1") else "test2_archives"
            members = sorted(extraction[subject])
            if not all((extracted / member).is_file() for member in members):
                safe_extract_members(audit_root / archive_group / f"{subject}.tar.gz", members, extracted)
        runtime_rows = []
        texture_rows = []
        for case in cases:
            rgb_paths, mask_paths = link_case_frames(case, extracted, output_root / "derived")
            video = output_root / "derived" / "videos" / f"{case['case_id']}.mp4"
            encode_video(rgb_paths[0].parent, video)
            texture = texture_statistics(rgb_paths, mask_paths)
            texture_rows.append({"case_id": case["case_id"], "subject": case["subject"], "angle_stratum": case["angle_stratum"], **texture})
            runtime_rows.append(
                {
                    "case_id": case["case_id"], "dataset": "MVHuman", "protocol": "MVH150", "role": "test",
                    "input_video": str(video.relative_to(output_root / "derived")), "fps": DERIVED_FPS, "num_frames": EXPECTED_FRAMES,
                }
            )
        runtime_manifest = output_root / "manifests" / "test_runtime.jsonl"
        runtime_manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in runtime_rows), encoding="utf-8")
        atomic_json(output_root / "texture_statistics.json", {"schema_version": SCHEMA, "cases": texture_rows})
        materialization = {
            "runtime_manifest": str(runtime_manifest), "runtime_manifest_sha256": sha256(runtime_manifest),
            "evaluator_manifest": str(evaluator_manifest), "evaluator_manifest_sha256": sha256(evaluator_manifest),
            "protocol_freeze": str(output_root / "protocol_freeze.json"), "protocol_freeze_sha256": sha256(output_root / "protocol_freeze.json"),
            "videos": len(runtime_rows), "extracted_members": sum(len(value) for value in extraction.values()),
        }
        atomic_json(output_root / "materialization_ledger.json", materialization)
    print(json.dumps({"output_root": str(output_root), "subjects": len(subject_ledger), "cases": len(cases), "materialized": args.materialize}, indent=2))


if __name__ == "__main__":
    main()
