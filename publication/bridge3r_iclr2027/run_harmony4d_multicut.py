#!/usr/bin/env python3
"""Run one frozen Harmony4D multi-cut or no-cut publication case on GPU.

The three-shot path performs RGB-only Human3R forwards, constructs each B0
gauge from a read-only shadow prefix, and applies the single locked Bridge3R
transaction at boundaries 50 and 100. The no-cut path is a strict transaction
no-op control. Neither path opens calibration, SMPL annotations, evaluator
files, or GT identity labels.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
for candidate in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from dust3r.model import ARCroco3DStereo  # noqa: E402
from dust3r.utils.smpl_layer import SMPL_Layer  # noqa: E402
from publication.bridge3r_iclr2027.runtime_contract import apply_locked_transaction  # noqa: E402
from versions.v14.causal_image_detector import CausalGRUShotDetector  # noqa: E402
from versions.v14.probe_p1_foot_scene_observability import anonymous_match  # noqa: E402
from versions.v14.run_v14_2_single_sequence import configure_model, set_event_indices  # noqa: E402
from versions.v15.harmony4d.run_harmony_case import (  # noqa: E402
    DETECTOR_PATH,
    STATIC_DETECTOR_CSV,
    decode_sequence,
    default_checkpoints,
    map_frames,
    pack_methods,
    run_forward,
    strict_original,
)
from versions.v15.harmony4d.topology import CommonTopology  # noqa: E402
from versions.v13 import gt_id_consensus as gt_helpers  # noqa: E402
from v10_image_only_detector import StreamingImageOnlyShotDetector  # noqa: E402


STANDARD = (
    "cameras_c2w",
    "joints_world",
    "vertices_world",
    "valid",
    "native_ids",
    "persistent_ids",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--line", type=int, default=1)
    parser.add_argument("--extracted-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--no-cut", action="store_true")
    parser.add_argument("--current-checkpoint", type=Path, default=None)
    parser.add_argument("--original-checkpoint", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_record(path: Path, line: int, no_cut: bool) -> dict[str, Any]:
    rows = [item for item in path.read_text(encoding="utf-8").splitlines() if item.strip()]
    if line < 1 or line > len(rows):
        raise IndexError(f"requested line {line}; manifest contains {len(rows)} records")
    record = json.loads(rows[line - 1])
    required = (
        ("capture_relative", "camera", "frame_numbers", "boundaries", "clip_length")
        if no_cut else
        ("capture_relative", "shot_cameras", "shot_frame_numbers", "boundaries", "clip_length")
    )
    missing = set(required).difference(record)
    if missing:
        raise ValueError(f"manifest record misses {sorted(missing)}")
    expected_boundaries = [] if no_cut else [50, 100]
    if list(record["boundaries"]) != expected_boundaries or int(record["clip_length"]) != 150:
        raise ValueError("record does not match the frozen 150-frame protocol")
    return record


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def image_paths(root: Path, record: dict[str, Any], no_cut: bool) -> list[Path]:
    sequence_root = root.resolve() / str(record["capture_relative"])
    if no_cut:
        pairs = [(str(record["camera"]), int(frame)) for frame in record["frame_numbers"]]
    else:
        pairs = [
            (str(camera), int(frame))
            for camera, shot in zip(record["shot_cameras"], record["shot_frame_numbers"])
            for frame in shot
        ]
    paths = [
        sequence_root / "exo" / camera / "images" / f"{frame:05d}.jpg"
        for camera, frame in pairs
    ]
    if len(paths) != 150:
        raise ValueError(f"expected 150 RGB frames, found {len(paths)}")
    missing = next((path for path in paths if not path.is_file()), None)
    if missing is not None:
        raise FileNotFoundError(missing)
    return paths


def decode_forward(
    model: ARCroco3DStereo,
    layer: SMPL_Layer,
    topology: CommonTopology,
    paths: list[Path],
    events: set[int],
    device: torch.device,
    size: int,
    label: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    views = gt_helpers.prepare_full_square_input(
        model, paths, SimpleNamespace(size=int(size))
    )
    views = set_event_indices(views, events)
    predictions, returned, debug, timing = run_forward(model, views, device, label)
    frames = decode_sequence(predictions, returned, debug, layer, topology)
    del predictions, returned, debug, views
    return frames, timing


def standard_arrays(frames: list[dict[str, Any]], topology: CommonTopology) -> dict[str, np.ndarray]:
    packed = pack_methods({"single": frames}, topology)
    return {key: np.asarray(packed["single__" + key]) for key in STANDARD}


def combine_arrays(parts: Iterable[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    values = list(parts)
    if not values:
        raise ValueError("no array parts")
    people = max(int(value["valid"].shape[1]) for value in values)
    output: dict[str, list[np.ndarray]] = {key: [] for key in STANDARD}
    for value in values:
        slots = int(value["valid"].shape[1])
        for key in STANDARD:
            source = np.asarray(value[key])
            if key == "cameras_c2w":
                output[key].append(source)
                continue
            if slots == people:
                output[key].append(source)
                continue
            shape = (source.shape[0], people, *source.shape[2:])
            if key in {"joints_world", "vertices_world"}:
                padded = np.full(shape, np.nan, dtype=source.dtype)
            elif key in {"native_ids", "persistent_ids"}:
                padded = np.full(shape, -1, dtype=source.dtype)
            elif key == "valid":
                padded = np.zeros(shape, dtype=source.dtype)
            padded[:, :slots] = source
            output[key].append(padded)
    return {key: np.concatenate(chunks, axis=0) for key, chunks in output.items()}


def frame_people(arrays: dict[str, np.ndarray], frame: int) -> list[dict[str, Any]]:
    people = []
    for slot in np.flatnonzero(np.asarray(arrays["valid"][frame]).astype(bool)):
        joints = np.asarray(arrays["joints_world"][frame, slot], dtype=np.float64)
        people.append(
            {
                "native_id": int(arrays["native_ids"][frame, slot]),
                "persistent_id": int(arrays["persistent_ids"][frame, slot]),
                "root": joints[[1, 2]].mean(axis=0),
                "joints": joints,
                "vertices": np.asarray(arrays["vertices_world"][frame, slot], dtype=np.float64),
                "torso": gt_helpers.torso_frame(joints),
            }
        )
    return people


def b0_transform(reference_camera: np.ndarray, shadow_camera: np.ndarray, raw_camera: np.ndarray) -> np.ndarray:
    return np.asarray(reference_camera) @ np.linalg.inv(np.asarray(shadow_camera)) @ np.asarray(shadow_camera) @ np.linalg.inv(np.asarray(raw_camera))


def finite_equal(left: np.ndarray, right: np.ndarray) -> bool:
    left, right = np.asarray(left), np.asarray(right)
    return bool(
        left.shape == right.shape
        and left.dtype == right.dtype
        and np.array_equal(np.isnan(left), np.isnan(right))
        and np.array_equal(np.nan_to_num(left, nan=0.0), np.nan_to_num(right, nan=0.0))
    )


def run_detectors(paths: list[Path]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, detector in (
        ("causal_gru_detector", CausalGRUShotDetector(DETECTOR_PATH)),
        ("static_logistic_detector", StreamingImageOnlyShotDetector(STATIC_DETECTOR_CSV)),
    ):
        started = time.perf_counter()
        labels, rows = detector.predict_sequence(paths)
        result[key] = {
            "seconds": time.perf_counter() - started,
            "labels": [int(value) for value in labels],
            "rows": rows,
            "first_positive_index": next((index for index, value in enumerate(labels) if int(value)), None),
        }
    return result


def run_multicut(
    current_model: ARCroco3DStereo,
    layer: SMPL_Layer,
    topology: CommonTopology,
    paths: list[Path],
    device: torch.device,
    size: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    """Materialize two ordered locked transactions from RGB-only predictions."""

    shadow_first, timing_shadow_first = decode_forward(
        current_model, layer, topology, paths[:51], {50}, device, size, "bridge_shadow_cut50"
    )
    raw_second, timing_raw_second = decode_forward(
        current_model, layer, topology, paths[50:100], set(), device, size, "bridge_reset_shot2"
    )
    first_b0 = map_frames(
        raw_second,
        np.asarray(shadow_first[-1]["camera"]) @ np.linalg.inv(np.asarray(raw_second[0]["camera"])),
    )
    first_raw = combine_arrays([
        standard_arrays(shadow_first[:-1], topology),
        standard_arrays(first_b0, topology),
    ])
    first_pairs = anonymous_match(shadow_first[-2]["people"], first_b0[0]["people"])["pairs"]
    first_out, first_diagnostics = apply_locked_transaction(
        first_raw, boundary=50, pairs=first_pairs, cut_detected=True
    )

    shadow_second, timing_shadow_second = decode_forward(
        current_model, layer, topology, paths[:101], {50, 100}, device, size, "bridge_shadow_cut100"
    )
    raw_third, timing_raw_third = decode_forward(
        current_model, layer, topology, paths[100:], set(), device, size, "bridge_reset_shot3"
    )
    bridge_reference = np.asarray(first_out["cameras_c2w"][99])
    shadow_reference = np.asarray(shadow_second[99]["camera"])
    second_b0 = map_frames(
        raw_third,
        bridge_reference @ np.linalg.inv(shadow_reference)
        @ np.asarray(shadow_second[100]["camera"])
        @ np.linalg.inv(np.asarray(raw_third[0]["camera"])),
    )
    second_raw = combine_arrays([first_out, standard_arrays(second_b0, topology)])
    second_pairs = anonymous_match(frame_people(first_out, 99), second_b0[0]["people"])["pairs"]
    final_out, second_diagnostics = apply_locked_transaction(
        second_raw, boundary=100, pairs=second_pairs, cut_detected=True
    )
    return final_out, [first_diagnostics, second_diagnostics], {
        "shadow_cut50": timing_shadow_first,
        "reset_shot2": timing_raw_second,
        "shadow_cut100": timing_shadow_second,
        "reset_shot3": timing_raw_third,
        "association_pairs": {
            "50": [list(map(int, pair)) for pair in first_pairs],
            "100": [list(map(int, pair)) for pair in second_pairs],
        },
    }


def main() -> None:
    args = parse_args()
    record = read_record(args.manifest, int(args.line), bool(args.no_cut))
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(output)
    paths = image_paths(args.extracted_root, record, bool(args.no_cut))
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    default_current, default_original = default_checkpoints()
    current_path = (args.current_checkpoint or default_current).resolve()
    original_path = (args.original_checkpoint or default_original).resolve()
    for artifact in (current_path, original_path, DETECTOR_PATH, STATIC_DETECTOR_CSV):
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
    topology = CommonTopology.load()
    runtime = {"detectors": run_detectors(paths)}

    original_model = ARCroco3DStereo.from_pretrained(str(original_path)).to(device)
    strict_original(original_model)
    original_model.eval()
    original_layer = SMPL_Layer(
        type="smplx", gender="neutral", num_betas=10, kid=False, person_center="head"
    ).to(device).eval()
    strict_frames, strict_timing = decode_forward(
        original_model, original_layer, topology, paths, set(), device, int(args.size), "strict_human3r"
    )
    strict_arrays = standard_arrays(strict_frames, topology)
    del original_model, original_layer, strict_frames
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    current_model = ARCroco3DStereo.from_pretrained(str(current_path)).to(device)
    current_flags = configure_model(current_model)
    current_model.eval()
    current_layer = SMPL_Layer(
        type="smplx", gender="neutral", num_betas=10, kid=False, person_center="head"
    ).to(device).eval()
    if args.no_cut:
        clean_frames, clean_timing = decode_forward(
            current_model, current_layer, topology, paths, set(), device, int(args.size), "bridge_nocut_clean"
        )
        clean_arrays = standard_arrays(clean_frames, topology)
        bridge_arrays, no_cut_diagnostics = apply_locked_transaction(
            clean_arrays, boundary=None, pairs=[], cut_detected=False
        )
        exact = {key: finite_equal(clean_arrays[key], bridge_arrays[key]) for key in STANDARD}
        diagnostics: dict[str, Any] = {
            "no_cut": no_cut_diagnostics,
            "no_cut_array_equal": exact,
            "clean_forward": clean_timing,
        }
        method_name = "bridge3r_nocut"
    else:
        bridge_arrays, transactions, transaction_runtime = run_multicut(
            current_model, current_layer, topology, paths, device, int(args.size)
        )
        diagnostics = {"transactions": transactions, "transaction_runtime": transaction_runtime}
        method_name = "bridge3r"
    del current_model, current_layer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    output.parent.mkdir(parents=True, exist_ok=True)
    packed = {
        **{"strict_human3r__" + key: value for key, value in strict_arrays.items()},
        **{method_name + "__" + key: value for key, value in bridge_arrays.items()},
    }
    temporary = output.with_suffix(output.suffix + ".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **packed)
    os.replace(temporary, output)
    report = {
        "schema_version": "Bridge3R-Harmony4D-multicut-runtime-v1",
        "record": record,
        "methods": ["strict_human3r", method_name],
        "runtime": runtime["detectors"],
        "strict_forward": strict_timing,
        "bridge3r": diagnostics,
        "checkpoint": {
            "current": str(current_path),
            "current_sha256": sha256(current_path),
            "original": str(original_path),
            "original_sha256": sha256(original_path),
            "detector": str(DETECTOR_PATH),
            "detector_sha256": sha256(DETECTOR_PATH),
            "current_flags": current_flags,
        },
        "provenance": {
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256(args.manifest.resolve()),
            "manifest_line": int(args.line),
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": sha256(Path(__file__).resolve()),
            "argv": sys.argv,
        },
        "runtime_contract": {
            "gt_used": False,
            "pre_cut_frames_mutated": False,
            "no_cut": bool(args.no_cut),
            "boundaries": record["boundaries"],
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "cache": str(output),
        "cache_sha256": sha256(output),
    }
    report_path = output.with_suffix(".runtime.json")
    report_path.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "case_id": record["case_id"],
        "cache": str(output),
        "runtime": str(report_path),
        "methods": report["methods"],
    }, indent=2))


if __name__ == "__main__":
    main()
