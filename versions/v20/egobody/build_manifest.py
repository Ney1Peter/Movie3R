#!/usr/bin/env python3
"""Build deterministic, GT-isolated EgoBody-CS150 runtime/evaluator manifests."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from versions.v20.egobody.dataset import (  # noqa: E402
    BOUNDARY_INDEX,
    CASES_PER_RECORDING,
    CLIP_LENGTH,
    FPS,
    KINECT_ID_TO_INTRINSIC_ROLE,
    PAIR_STRATA,
    POST_COUNT,
    PRE_COUNT,
    PROTOCOL_NAME,
    PROTOCOL_SEED,
    RecordingInfo,
    audit_recording,
    canonical_json,
    file_sha256,
    jsonable,
    load_recording_metadata,
    value_sha256,
)


RUNTIME_SCHEMA = "Bridge3R-v20-EgoBody-runtime-manifest-row-v1"
EVALUATOR_SCHEMA = "Bridge3R-v20-EgoBody-evaluator-manifest-row-v1"
SPEC_SCHEMA = "Bridge3R-v20-EgoBody-manifest-spec-v1"


def _case_id(recording: str, stratum: str, pre: str, post: str, boundary: int) -> str:
    return f"egobody_{recording}_{stratum}_{pre}_{post}_b{boundary:05d}"


def _runtime_row(audit: dict[str, Any], pair: dict[str, Any]) -> dict[str, Any]:
    frames = [int(value) for value in audit["clip_frames"]]
    pre, post = str(pair["pre_camera"]), str(pair["post_camera"])
    boundary_frame = frames[BOUNDARY_INDEX]
    pre_role = str(audit["cameras"][pre]["intrinsic_role"])
    post_role = str(audit["cameras"][post]["intrinsic_role"])
    pre_role = pre_role[len("kinect_") :] if pre_role.startswith("kinect_") else pre_role
    post_role = post_role[len("kinect_") :] if post_role.startswith("kinect_") else post_role
    image_members = [
        f"kinect_color/{audit['recording']}/{pre_role}/frame_{frame:05d}.jpg"
        for frame in frames[:PRE_COUNT]
    ] + [
        f"kinect_color/{audit['recording']}/{post_role}/frame_{frame:05d}.jpg"
        for frame in frames[PRE_COUNT:]
    ]
    return {
        "schema_version": RUNTIME_SCHEMA,
        "protocol": PROTOCOL_NAME,
        "protocol_seed": PROTOCOL_SEED,
        "split": audit["protocol_split"],
        "case_id": _case_id(
            str(audit["recording"]), str(pair["angle_stratum"]), pre, post, boundary_frame
        ),
        "recording": audit["recording"],
        "capture": audit["recording"],
        "capture_relative": audit["recording"],
        "pre_camera": pre,
        "post_camera": post,
        "pre_frame_numbers": frames[:PRE_COUNT],
        "post_frame_numbers": frames[PRE_COUNT:],
        "image_members": image_members,
        "image_member_layout": (
            "kinect_color/<recording>/<master_or_sub_N>/frame_<source_id:05d>.jpg"
        ),
        "boundary_frame": boundary_frame,
        "boundary_index": BOUNDARY_INDEX,
        "clip_length": CLIP_LENGTH,
        "fps": FPS,
        "source_time_contract": "post[0] = pre[-1] + 1",
        "runtime_gt_access": False,
        "selection_depends_on_model_result": False,
    }


def _evaluator_row(
    recording: RecordingInfo,
    audit: dict[str, Any],
    pair: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    cameras = audit["cameras"]
    pre, post = runtime["pre_camera"], runtime["post_camera"]
    return {
        "schema_version": EVALUATOR_SCHEMA,
        "protocol": PROTOCOL_NAME,
        "protocol_seed": PROTOCOL_SEED,
        "split": recording.protocol_split,
        "official_split": recording.official_split,
        "case_id": runtime["case_id"],
        "runtime_row_sha256": value_sha256(runtime),
        "recording": recording.recording,
        "scene_name": recording.scene,
        "subjects_evaluator_only": [recording.subject0, recording.subject1],
        "fpv_subject_evaluator_only": recording.fpv_subject,
        "interactee_subject_evaluator_only": recording.interactee_subject,
        "body_descriptors_evaluator_only": [
            {"index": recording.body0.index, "gender": recording.body0.gender},
            {"index": recording.body1.index, "gender": recording.body1.gender},
        ],
        "fpv_body_index_evaluator_only": recording.fpv_body.index,
        "metadata_start_frame_evaluator_only": recording.start_frame,
        "metadata_end_frame_evaluator_only": recording.end_frame,
        "pre_camera": pre,
        "post_camera": post,
        "pre_frame_numbers": runtime["pre_frame_numbers"],
        "post_frame_numbers": runtime["post_frame_numbers"],
        "boundary_index": BOUNDARY_INDEX,
        "angle_stratum_evaluator_only": pair["angle_stratum"],
        "camera_rotation_span_deg_evaluator_only": pair["rotation_span_deg"],
        "camera_center_baseline_m_evaluator_only": pair["camera_center_baseline_m"],
        "camera_pair_selection_rank_evaluator_only": pair["selection_rank"],
        "available_camera_pair_count_evaluator_only": pair["available_pair_count"],
        "person_count_evaluator_only": 2,
        "camera_calibration_evaluator_only": {
            pre: cameras[pre],
            post: cameras[post],
        },
        "camera_chain_contract": audit["camera_chain_contract"],
        "gt_sources_evaluator_only": {
            "camera_wearer": f"smplx_camera_wearer_{recording.official_split}",
            "interactee": f"smplx_interactee_{recording.official_split}",
            "calibration": "calibrations",
        },
        "gt_available_to_runtime": False,
    }


def rows_for_recording(
    recording: RecordingInfo,
    calibrations_root: Path,
    kinect_params_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    audit = audit_recording(recording, calibrations_root, kinect_params_root)
    runtime_rows, evaluator_rows = [], []
    for pair in audit["selected_camera_pairs"]:
        runtime = _runtime_row(audit, pair)
        evaluator = _evaluator_row(recording, audit, pair, runtime)
        runtime_rows.append(runtime)
        evaluator_rows.append(evaluator)
    if len(runtime_rows) != CASES_PER_RECORDING:
        raise AssertionError(recording.recording)
    return runtime_rows, evaluator_rows, audit


def build_rows(
    recordings: Iterable[RecordingInfo],
    calibrations_root: Path,
    kinect_params_root: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    runtime_by_split = {key: [] for key in ("development", "holdout", "test")}
    evaluator_by_split = {key: [] for key in ("development", "holdout", "test")}
    audits = []
    for recording in sorted(recordings, key=lambda value: value.recording):
        runtime, evaluator, audit = rows_for_recording(
            recording, calibrations_root, kinect_params_root
        )
        runtime_by_split[recording.protocol_split].extend(runtime)
        evaluator_by_split[recording.protocol_split].extend(evaluator)
        audits.append(audit)
    for split in runtime_by_split:
        runtime_by_split[split].sort(key=lambda row: str(row["case_id"]))
        evaluator_by_split[split].sort(key=lambda row: str(row["case_id"]))
        if [row["case_id"] for row in runtime_by_split[split]] != [
            row["case_id"] for row in evaluator_by_split[split]
        ]:
            raise AssertionError(f"Runtime/evaluator case mismatch in {split}")
    return runtime_by_split, evaluator_by_split, audits


def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return (
        "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")
    )


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_bytes(value)
    os.replace(partial, path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(
        path,
        (json.dumps(jsonable(value), sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode(
            "utf-8"
        ),
    )


def write_split_manifests(
    split: str,
    runtime_rows: list[dict[str, Any]],
    evaluator_rows: list[dict[str, Any]],
    output_dir: Path,
    source_metadata: dict[str, Any],
) -> dict[str, Any]:
    runtime_path = output_dir / f"egobody_cs150_{split}.runtime.jsonl"
    evaluator_path = output_dir / f"egobody_cs150_{split}.evaluator.jsonl"
    _atomic_bytes(runtime_path, jsonl_bytes(runtime_rows))
    _atomic_bytes(evaluator_path, jsonl_bytes(evaluator_rows))
    recording_counts: dict[str, int] = {}
    for row in runtime_rows:
        recording = str(row["recording"])
        recording_counts[recording] = recording_counts.get(recording, 0) + 1
    if set(recording_counts.values()) - {CASES_PER_RECORDING}:
        raise ValueError(f"Recording-macro imbalance in {split}: {recording_counts}")
    spec = {
        "schema_version": SPEC_SCHEMA,
        "protocol": PROTOCOL_NAME,
        "protocol_seed": PROTOCOL_SEED,
        "split": split,
        "runtime_manifest": str(runtime_path.resolve()),
        "runtime_manifest_sha256": file_sha256(runtime_path),
        "runtime_row_schema": RUNTIME_SCHEMA,
        "evaluator_manifest": str(evaluator_path.resolve()),
        "evaluator_manifest_sha256": file_sha256(evaluator_path),
        "evaluator_row_schema": EVALUATOR_SCHEMA,
        "case_count": len(runtime_rows),
        "recording_count": len(recording_counts),
        "cases_per_recording": CASES_PER_RECORDING,
        "recording_macro_balanced": True,
        "clip_length": CLIP_LENGTH,
        "pre_frames": PRE_COUNT,
        "post_frames": POST_COUNT,
        "boundary_index": BOUNDARY_INDEX,
        "fps": FPS,
        "source_time_contract": "post[0] = pre[-1] + 1",
        "camera_pair_strata": list(PAIR_STRATA),
        "camera_mapping": {
            str(key): value for key, value in sorted(KINECT_ID_TO_INTRINSIC_ROLE.items())
        },
        "camera_chain_contract": (
            "world_from_camera = world_from_kinect12 @ kinect12_from_camera_color"
        ),
        "runtime_evaluator_case_ids_equal": True,
        "runtime_contains_evaluator_only_fields": False,
        "selection_depends_on_model_result": False,
        "source_metadata": source_metadata,
    }
    spec_path = output_dir / f"egobody_cs150_{split}.spec.json"
    _atomic_json(spec_path, spec)
    return {**spec, "spec": str(spec_path.resolve()), "spec_sha256": file_sha256(spec_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-info", type=Path, required=True)
    parser.add_argument("--data-splits", type=Path, required=True)
    parser.add_argument("--calibrations-root", type=Path, required=True)
    parser.add_argument("--kinect-params-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("development", "holdout", "test", "all"),
        default="all",
    )
    parser.add_argument(
        "--recordings",
        nargs="*",
        help="Optional exact recording subset; intended only for smoke manifests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recordings = load_recording_metadata(args.data_info, args.data_splits)
    by_name = {row.recording: row for row in recordings}
    if args.recordings:
        unknown = sorted(set(args.recordings) - set(by_name))
        if unknown:
            raise ValueError(f"Unknown recordings: {unknown}")
        recordings = [by_name[name] for name in sorted(set(args.recordings))]
    requested = (
        ("development", "holdout", "test")
        if args.split == "all"
        else (args.split,)
    )
    recordings = [row for row in recordings if row.protocol_split in requested]
    if not recordings:
        raise ValueError("No recordings selected")
    runtime, evaluator, audits = build_rows(
        recordings, args.calibrations_root, args.kinect_params_root
    )
    source_metadata = {
        "data_info": str(args.data_info.resolve()),
        "data_info_sha256": file_sha256(args.data_info),
        "data_splits": str(args.data_splits.resolve()),
        "data_splits_sha256": file_sha256(args.data_splits),
        "calibrations_root": str(args.calibrations_root.resolve()),
        "kinect_params_root": str(args.kinect_params_root.resolve()),
        "official_split_mapping": {
            "train": "development",
            "val": "holdout",
            "test": "test",
        },
    }
    output = {}
    for split in requested:
        if not runtime[split]:
            continue
        output[split] = write_split_manifests(
            split, runtime[split], evaluator[split], args.output_dir, source_metadata
        )
    index = {
        "schema_version": "Bridge3R-v20-EgoBody-manifest-index-v1",
        "protocol": PROTOCOL_NAME,
        "protocol_seed": PROTOCOL_SEED,
        "selected_recording_count": len(recordings),
        "selected_recordings_sha256": value_sha256(
            [row.recording for row in recordings]
        ),
        "audit_count": len(audits),
        "audits_sha256": value_sha256(audits),
        "splits": output,
        "source_metadata": source_metadata,
    }
    index_path = args.output_dir / "egobody_cs150_manifest_index.json"
    _atomic_json(index_path, index)
    print(
        json.dumps(
            {
                "index": str(index_path.resolve()),
                "index_sha256": file_sha256(index_path),
                "recordings": len(recordings),
                "cases": {key: value["case_count"] for key, value in output.items()},
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
