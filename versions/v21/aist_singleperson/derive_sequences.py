#!/usr/bin/env python3
"""Materialize frozen AIST++ CS150/MC150 videos, labels and manifests.

Each output clip uses the frame map produced by :mod:`build_frame_maps`: the
same 150 physical action times are retained while only the RGB camera changes
at fixed evaluator-only cuts.  Runtime manifests deliberately contain neither
cut labels nor calibration/GT fields; those live in a separate evaluator
manifest and per-case labels.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pickle
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .protocol import (
        DEFAULT_BUNDLE_ROOT,
        DEFAULT_DERIVED_ROOT,
        OUTPUT_FPS,
        OUTPUT_FRAMES,
        PROTOCOL_NAME,
        PROTOCOLS,
        atomic_json,
        camera_records,
        camera_to_world,
        camera_world_to_camera,
        canonical_json_digest,
        load_frozen_sources,
        output_gt_ticks,
        sha256_file,
        source_video_path,
        verify_input_manifest_freeze,
    )
except ImportError:  # Direct script execution from this directory.
    from protocol import (  # type: ignore
        DEFAULT_BUNDLE_ROOT,
        DEFAULT_DERIVED_ROOT,
        OUTPUT_FPS,
        OUTPUT_FRAMES,
        PROTOCOL_NAME,
        PROTOCOLS,
        atomic_json,
        camera_records,
        camera_to_world,
        camera_world_to_camera,
        canonical_json_digest,
        load_frozen_sources,
        output_gt_ticks,
        sha256_file,
        source_video_path,
        verify_input_manifest_freeze,
    )


CASE_SCHEMA = "Bridge3R-AIST-SinglePerson-derived-case-v1"
LABEL_SCHEMA = "Bridge3R-AIST-SinglePerson-label-v1"
MANIFEST_SCHEMA = "Bridge3R-AIST-SinglePerson-derived-manifest-v1"
METRE_PER_AIST_NATIVE_UNIT = 0.01  # Official camera/keypoint translations are centimetres.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
    parser.add_argument("--derived-root", type=Path, default=DEFAULT_DERIVED_ROOT)
    parser.add_argument("--roles", default="pilot", help="Comma-separated frozen roles: pilot,test.")
    # The workspace ffmpeg build currently requires a newer NVENC driver API
    # than the installed driver exposes, so the reproducible default is CPU
    # x264.  NVENC remains an explicit opt-in for a compatible environment.
    parser.add_argument("--encoder", choices=("nvenc", "x264"), default="x264")
    parser.add_argument("--gpu", type=int, default=0, help="NVENC GPU index; ignored for x264.")
    parser.add_argument("--source-id", action="append", default=[], help="Optional exact frozen source ID; repeatable.")
    parser.add_argument("--force", action="store_true", help="Replace completed generated artifacts.")
    return parser.parse_args()


def map_path(derived_root: Path, source: dict[str, Any]) -> Path:
    return derived_root / "frame_maps/aist" / str(source["role"]) / f"{str(source['source_id']).split(':', 1)[1]}.json"


def load_checked_frame_map(derived_root: Path, source: dict[str, Any], input_hashes: dict[str, str]) -> dict[str, Any]:
    path = map_path(derived_root, source)
    if not path.is_file():
        raise FileNotFoundError(f"Frame map must be built before deriving sequences: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    digest = payload.pop("content_sha256", None)
    if digest != canonical_json_digest(payload):
        raise ValueError(f"Frame-map content digest differs: {path}")
    payload["content_sha256"] = digest
    if payload.get("input_manifest_sha256") != input_hashes:
        raise ValueError(f"Frame map uses stale input manifest: {path}")
    if payload.get("source_id") != source["source_id"]:
        raise ValueError(f"Frame-map source mismatch: {path}")
    if payload.get("output_num_frames") != OUTPUT_FRAMES or payload.get("output_fps") != OUTPUT_FPS:
        raise ValueError(f"Frame-map output specification mismatch: {path}")
    if payload.get("target_gt_ticks") != output_gt_ticks(source).tolist():
        raise ValueError(f"Frame-map GT tick mismatch: {path}")
    # JSON is written with sorted keys, while the frozen camera tuple is an
    # ordered sequence used later for A/B/C/D composition.  Validate the
    # membership here and retain the source tuple as the only ordering source.
    if set(payload.get("videos", {})) != set(source["camera_ids"]):
        raise ValueError(f"Frame-map camera order mismatch: {path}")
    return payload


def select_expression(indices: list[int]) -> str:
    if len(indices) != OUTPUT_FRAMES or sorted(indices) != indices or len(set(indices)) != len(indices):
        raise ValueError("Selected video frame indices must be unique and strictly increasing")
    # The comma must be escaped for ffmpeg's filter-expression parser.
    return "+".join(f"eq(n\\,{int(index)})" for index in indices)


def shot_ranges(lengths: tuple[int, ...]) -> list[tuple[int, int]]:
    if sum(lengths) != OUTPUT_FRAMES:
        raise ValueError(f"Unexpected protocol lengths: {lengths}")
    start = 0
    ranges = []
    for length in lengths:
        end = start + int(length)
        ranges.append((start, end))
        start = end
    return ranges


def filter_graph(source: dict[str, Any], frame_map: dict[str, Any]) -> str:
    """Create the three protocol streams in one ffmpeg decode pass."""
    camera_ids = [str(value) for value in source["camera_ids"]]
    if len(camera_ids) != 4:
        raise ValueError("Each AIST source must carry four frozen views")
    chains: list[str] = []
    for index, camera_id in enumerate(camera_ids):
        indices = frame_map["videos"][camera_id]["output_decode_indices"]
        chains.append(
            f"[{index}:v]select='{select_expression(indices)}',setpts=N/(30*TB),format=yuv420p[base{index}]"
        )
        # A labelled ffmpeg stream can be consumed only once.  Split it for
        # exactly the protocols that use this camera; leaving unused split
        # outputs unconnected makes ffmpeg reject the graph.
        consumers = [name for name, spec in PROTOCOLS.items() if index < len(spec["shot_lengths"])]
        labels = "".join(f"[v{index}_{name}]" for name in consumers)
        if len(consumers) == 1:
            chains.append(f"[base{index}]null{labels}")
        else:
            chains.append(f"[base{index}]split={len(consumers)}{labels}")
    for protocol, spec in PROTOCOLS.items():
        lengths = tuple(spec["shot_lengths"])
        chunks: list[str] = []
        for shot_index, (start, end) in enumerate(shot_ranges(lengths)):
            chains.append(
                f"[v{shot_index}_{protocol}]trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS[{protocol}_s{shot_index}]"
            )
            chunks.append(f"[{protocol}_s{shot_index}]")
        chains.append(
            f"{''.join(chunks)}concat=n={len(chunks)}:v=1:a=0,format=yuv420p[{protocol}]"
        )
    return ";".join(chains)


def output_paths(derived_root: Path, source: dict[str, Any], protocol: str) -> tuple[Path, Path, Path]:
    name = str(source["source_id"]).split(":", 1)[1]
    role = str(source["role"])
    video = derived_root / "videos/aist" / protocol.lower() / role / f"{name}.mp4"
    label = derived_root / "labels/aist" / protocol.lower() / role / f"{name}.npz"
    metadata = derived_root / "labels/aist" / protocol.lower() / role / f"{name}.json"
    return video, label, metadata


def ffprobe_output(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames", "-of", "json", str(path),
    ]
    completed = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    streams = json.loads(completed.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"Expected exactly one video stream in derived output: {path}")
    stream = streams[0]
    numerator, denominator = (int(value) for value in str(stream["avg_frame_rate"]).split("/"))
    fps = numerator / denominator
    result = {
        "width": int(stream["width"]), "height": int(stream["height"]),
        "fps": fps, "frame_count": int(stream["nb_read_frames"]),
    }
    if result["frame_count"] != OUTPUT_FRAMES or abs(result["fps"] - OUTPUT_FPS) > 1e-6:
        raise ValueError(f"Derived video format mismatch: {path}: {result}")
    return result


def encode_source_videos(
    bundle_root: Path,
    derived_root: Path,
    source: dict[str, Any],
    frame_map: dict[str, Any],
    encoder: str,
    gpu: int,
    force: bool,
) -> dict[str, dict[str, Any]]:
    existing = {protocol: output_paths(derived_root, source, protocol)[0] for protocol in PROTOCOLS}
    if all(path.is_file() for path in existing.values()) and not force:
        return {
            protocol: {"path": path, "ffprobe": ffprobe_output(path), "reused": True}
            for protocol, path in existing.items()
        }
    if any(path.exists() for path in existing.values()) and not force:
        existing_paths = [str(path) for path in existing.values() if path.exists()]
        raise FileExistsError(
            "Only a subset of derived videos exists; refusing to overwrite without --force: "
            f"{existing_paths}"
        )
    for path in existing.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    partials = {protocol: path.with_suffix(".partial.mp4") for protocol, path in existing.items()}
    stale_partials = [str(path) for path in partials.values() if path.exists()]
    if stale_partials:
        raise FileExistsError(f"Refusing to overwrite stale partial outputs: {stale_partials}")

    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]
    for camera_id in source["camera_ids"]:
        command += ["-i", str(source_video_path(bundle_root, source, str(camera_id)))]
    command += ["-filter_complex", filter_graph(source, frame_map)]
    for protocol in PROTOCOLS:
        command += ["-map", f"[{protocol}]", "-an", "-r", str(OUTPUT_FPS)]
        if encoder == "nvenc":
            command += ["-c:v", "h264_nvenc", "-gpu", str(gpu), "-preset", "p4", "-tune", "hq", "-rc", "vbr", "-cq", "19", "-b:v", "0"]
        else:
            command += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"]
        command += ["-pix_fmt", "yuv420p", "-movflags", "+faststart", str(partials[protocol])]
    subprocess.run(command, check=True)
    results: dict[str, dict[str, Any]] = {}
    for protocol, partial in partials.items():
        format_info = ffprobe_output(partial)
        os.replace(partial, existing[protocol])
        results[protocol] = {"path": existing[protocol], "ffprobe": format_info, "reused": False}
    return results


def protocol_camera_timeline(source: dict[str, Any], protocol: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    lengths = tuple(PROTOCOLS[protocol]["shot_lengths"])
    camera_ids = [str(value) for value in source["camera_ids"]]
    timeline: list[str] = []
    shot_ids: list[int] = []
    for shot, length in enumerate(lengths):
        timeline.extend([camera_ids[shot]] * int(length))
        shot_ids.extend([shot] * int(length))
    if len(timeline) != OUTPUT_FRAMES:
        raise AssertionError(protocol)
    return timeline, np.asarray(shot_ids, dtype=np.int16), np.asarray(PROTOCOLS[protocol]["cut_indices"], dtype=np.int16)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError(f"Refusing to overwrite stale partial label: {temporary}")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def load_official_labels(bundle_root: Path, source: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    with (bundle_root / source["motion_path"]).open("rb") as stream:
        motion = pickle.load(stream)
    with (bundle_root / source["keypoints3d_path"]).open("rb") as stream:
        keypoints = pickle.load(stream)
    required_motion = {"smpl_poses", "smpl_scaling", "smpl_trans"}
    missing = required_motion - set(motion)
    if missing or "keypoints3d" not in keypoints:
        raise ValueError(f"Official AIST labels missing fields for {source['source_id']}: {missing}")
    return motion, keypoints


def write_labels(
    bundle_root: Path,
    derived_root: Path,
    source: dict[str, Any],
    frame_map: dict[str, Any],
    protocol: str,
    video_info: dict[str, Any],
    input_hashes: dict[str, str],
    force: bool,
) -> dict[str, Any]:
    _, label_path, metadata_path = output_paths(derived_root, source, protocol)
    if label_path.is_file() and metadata_path.is_file() and not force:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return {"label_path": label_path, "metadata_path": metadata_path, "metadata": metadata, "reused": True}
    if (label_path.exists() or metadata_path.exists()) and not force:
        raise FileExistsError(f"Partial derived labels exist without --force: {label_path}, {metadata_path}")

    motion, keypoints = load_official_labels(bundle_root, source)
    ticks = output_gt_ticks(source)
    pose = np.asarray(motion["smpl_poses"], dtype=np.float32)
    translation = np.asarray(motion["smpl_trans"], dtype=np.float32)
    joints = np.asarray(keypoints["keypoints3d"], dtype=np.float32)
    if max(ticks) >= len(pose) or max(ticks) >= len(translation) or max(ticks) >= len(joints):
        raise ValueError(f"GT tick exceeds official label length for {source['source_id']}")
    timeline, shot_ids, cut_indices = protocol_camera_timeline(source, protocol)
    records = camera_records(bundle_root, source)
    camera_K = np.empty((OUTPUT_FRAMES, 3, 3), dtype=np.float32)
    camera_w2c = np.empty((OUTPUT_FRAMES, 4, 4), dtype=np.float32)
    camera_c2w = np.empty((OUTPUT_FRAMES, 4, 4), dtype=np.float32)
    rgb_pts = np.empty((OUTPUT_FRAMES,), dtype=np.float64)
    decode_indices = np.empty((OUTPUT_FRAMES,), dtype=np.int32)
    for frame, camera_id in enumerate(timeline):
        K, rotation, translation_native = camera_world_to_camera(records[camera_id])
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = rotation
        w2c[:3, 3] = translation_native * METRE_PER_AIST_NATIVE_UNIT
        c2w = camera_to_world(records[camera_id])
        c2w[:3, 3] *= METRE_PER_AIST_NATIVE_UNIT
        camera_K[frame] = K.astype(np.float32)
        camera_w2c[frame] = w2c.astype(np.float32)
        camera_c2w[frame] = c2w.astype(np.float32)
        video_map = frame_map["videos"][camera_id]
        rgb_pts[frame] = float(video_map["output_pts_seconds"][frame])
        decode_indices[frame] = int(video_map["output_decode_indices"][frame])

    scale = float(np.asarray(motion["smpl_scaling"], dtype=np.float32).reshape(-1)[0])
    arrays = {
        "gt_tick_60fps": ticks.astype(np.int32),
        "target_time_seconds": np.asarray(frame_map["target_times_seconds"], dtype=np.float64),
        "rgb_pts_seconds": rgb_pts,
        "source_decode_index": decode_indices,
        "shot_id": shot_ids,
        "camera_id": np.asarray(timeline),
        "cut_indices_evaluator_only": cut_indices,
        "world_keypoints_m": joints[ticks] * METRE_PER_AIST_NATIVE_UNIT,
        "smpl_root_translation_m": translation[ticks] * METRE_PER_AIST_NATIVE_UNIT,
        "smpl_root_orientation_axis_angle": pose[ticks, :3],
        "smpl_body_pose_axis_angle": pose[ticks, 3:],
        "smpl_scaling_native": np.asarray(scale, dtype=np.float32),
        "camera_intrinsics": camera_K,
        "camera_world_to_camera_m": camera_w2c,
        "camera_camera_to_world_m": camera_c2w,
    }
    if not label_path.exists() or force:
        atomic_npz(label_path, arrays)
    metadata = {
        "schema_version": LABEL_SCHEMA,
        "protocol": PROTOCOL_NAME,
        "input_manifest_sha256": input_hashes,
        "source_id": source["source_id"],
        "role": source["role"],
        "split": source["split"],
        "sequence_name": source["sequence_name"],
        "derived_protocol": protocol,
        "label_npz": str(label_path.relative_to(derived_root)),
        "label_npz_sha256": sha256_file(label_path),
        "video": str(video_info["path"].relative_to(derived_root)),
        "video_sha256": sha256_file(video_info["path"]),
        "video_format": video_info["ffprobe"],
        "output_fps": OUTPUT_FPS,
        "output_num_frames": OUTPUT_FRAMES,
        "world_length_unit": "metre",
        "native_aist_length_unit": "centimetre",
        "smpl_scaling_native_unit": "native SMPL mesh to AIST-centimetre scale",
        "joint_convention": "official AIST++ keypoints3d (17 joints; exact order retained pending common-joint evaluator freeze)",
        "camera_convention": "official AIST calibration; R,t map world centimetres to camera centimetres; exported translations are converted to metres",
        "runtime_disclosure": "Runtime MP4 manifest contains no camera IDs, calibration, GT ticks or cut labels.",
        "evaluator_disclosure": "cut indices, camera parameters and GT are evaluator-only.",
        "frame_map": str(map_path(derived_root, source).relative_to(derived_root)),
        "frame_map_sha256": frame_map["content_sha256"],
        "camera_ids_per_frame": timeline,
        "cut_indices_evaluator_only": cut_indices.tolist(),
    }
    metadata["content_sha256"] = canonical_json_digest(metadata)
    atomic_json(metadata_path, metadata)
    return {"label_path": label_path, "metadata_path": metadata_path, "metadata": metadata, "reused": False}


def write_case_artifact(
    derived_root: Path,
    source: dict[str, Any],
    frame_map: dict[str, Any],
    outputs: dict[str, dict[str, Any]],
    labels: dict[str, dict[str, Any]],
    input_hashes: dict[str, str],
) -> Path:
    name = str(source["source_id"]).split(":", 1)[1]
    path = derived_root / "cases/aist" / str(source["role"]) / f"{name}.json"
    payload = {
        "schema_version": CASE_SCHEMA,
        "protocol": PROTOCOL_NAME,
        "input_manifest_sha256": input_hashes,
        "source_id": source["source_id"],
        "role": source["role"],
        "split": source["split"],
        "source_sequence": source["sequence_name"],
        "camera_tuple_evaluator_only": source["camera_ids"],
        "camera_transition_angles_degrees_evaluator_only": source["camera_selection"]["actual_transition_angles_degrees"],
        "camera_transition_target_strata_evaluator_only": source["camera_selection"]["target_class_order"],
        "frame_map": str(map_path(derived_root, source).relative_to(derived_root)),
        "frame_map_sha256": frame_map["content_sha256"],
        "protocols": {
            protocol: {
                "case_id": f"aist__{name}__{protocol.lower()}",
                "video": str(outputs[protocol]["path"].relative_to(derived_root)),
                "label": str(labels[protocol]["label_path"].relative_to(derived_root)),
                "label_metadata": str(labels[protocol]["metadata_path"].relative_to(derived_root)),
                "cut_indices_evaluator_only": labels[protocol]["metadata"]["cut_indices_evaluator_only"],
            }
            for protocol in PROTOCOLS
        },
    }
    payload["content_sha256"] = canonical_json_digest(payload)
    atomic_json(path, payload)
    return path


def build_role_manifests(
    derived_root: Path, sources: list[dict[str, Any]], role: str, input_hashes: dict[str, str]
) -> list[Path]:
    selected = [source for source in sources if source["role"] == role]
    case_paths = [derived_root / "cases/aist" / role / f"{str(source['source_id']).split(':', 1)[1]}.json" for source in selected]
    missing = [str(path) for path in case_paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"Cannot freeze {role} manifests before all source artifacts exist: {missing[:3]}")
    outputs: list[Path] = []
    for protocol in PROTOCOLS:
        runtime_rows = []
        evaluator_rows = []
        for case_path in case_paths:
            case = json.loads(case_path.read_text(encoding="utf-8"))
            item = case["protocols"][protocol]
            runtime_rows.append({
                "case_id": item["case_id"], "dataset": "AIST++", "protocol": protocol,
                "role": role, "input_video": item["video"], "fps": OUTPUT_FPS, "num_frames": OUTPUT_FRAMES,
            })
            evaluator_rows.append({
                "case_id": item["case_id"], "dataset": "AIST++", "protocol": protocol,
                "role": role, "source_id": case["source_id"], "label": item["label"],
                "label_metadata": item["label_metadata"], "cut_indices_evaluator_only": item["cut_indices_evaluator_only"],
                "camera_tuple_evaluator_only": case["camera_tuple_evaluator_only"],
                "camera_transition_angles_degrees_evaluator_only": case["camera_transition_angles_degrees_evaluator_only"],
                "camera_transition_target_strata_evaluator_only": case["camera_transition_target_strata_evaluator_only"],
            })
        runtime_rows.sort(key=lambda row: row["case_id"])
        evaluator_rows.sort(key=lambda row: row["case_id"])
        for kind, rows in (("runtime", runtime_rows), ("evaluator", evaluator_rows)):
            path = derived_root / "manifests" / f"aist_{protocol.lower()}_{role}.{kind}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".partial")
            temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
            os.replace(temporary, path)
            outputs.append(path)
        spec = {
            "schema_version": MANIFEST_SCHEMA,
            "protocol": PROTOCOL_NAME,
            "derived_protocol": protocol,
            "role": role,
            "input_manifest_sha256": input_hashes,
            "case_count": len(runtime_rows),
            "runtime_manifest": str(outputs[-2].relative_to(derived_root)),
            "runtime_manifest_sha256": sha256_file(outputs[-2]),
            "evaluator_manifest": str(outputs[-1].relative_to(derived_root)),
            "evaluator_manifest_sha256": sha256_file(outputs[-1]),
            "runtime_manifest_contract": "RGB path, case ID, role, FPS and length only; no GT/camera/cut fields.",
            "evaluator_manifest_contract": "GT labels and cut indices are evaluation-only; never passed to a method.",
        }
        spec["content_sha256"] = canonical_json_digest(spec)
        spec_path = derived_root / "manifests" / f"aist_{protocol.lower()}_{role}.spec.json"
        atomic_json(spec_path, spec)
        outputs.append(spec_path)
    return outputs


def main() -> None:
    args = parse_args()
    roles = tuple(value.strip() for value in args.roles.split(",") if value.strip())
    bundle_root = args.bundle_root.resolve()
    derived_root = args.derived_root.resolve()
    input_hashes = verify_input_manifest_freeze(bundle_root)
    all_sources = load_frozen_sources(bundle_root, roles)
    sources = all_sources
    if args.source_id:
        requested = set(args.source_id)
        unknown = requested - {str(source["source_id"]) for source in sources}
        if unknown:
            raise SystemExit(f"--source-id not in selected roles: {sorted(unknown)}")
        sources = [source for source in sources if str(source["source_id"]) in requested]
    completed = []
    for index, source in enumerate(sources, start=1):
        frame_map = load_checked_frame_map(derived_root, source, input_hashes)
        videos = encode_source_videos(bundle_root, derived_root, source, frame_map, args.encoder, args.gpu, args.force)
        labels = {
            protocol: write_labels(bundle_root, derived_root, source, frame_map, protocol, videos[protocol], input_hashes, args.force)
            for protocol in PROTOCOLS
        }
        case_path = write_case_artifact(derived_root, source, frame_map, videos, labels, input_hashes)
        completed.append({"source_id": source["source_id"], "case": str(case_path), "video_reused": all(v["reused"] for v in videos.values())})
        print(f"derived sources: {index}/{len(sources)} complete ({source['source_id']})", flush=True)
    manifests = []
    if not args.source_id:
        for role in roles:
            manifests.extend(build_role_manifests(derived_root, all_sources, role, input_hashes))
    summary = {
        "schema_version": "Bridge3R-AIST-SinglePerson-derivation-run-v1",
        "protocol": PROTOCOL_NAME,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input_manifest_sha256": input_hashes,
        "roles": list(roles),
        "derived_sources": completed,
        "manifests": [str(path) for path in manifests],
    }
    summary["content_sha256"] = canonical_json_digest(summary)
    summary_path = derived_root / "derivation_runs" / f"derive_{'_'.join(sorted(roles))}.json"
    atomic_json(summary_path, summary)
    print(json.dumps({"summary": str(summary_path), "derived_source_count": len(completed)}, indent=2))


if __name__ == "__main__":
    main()
