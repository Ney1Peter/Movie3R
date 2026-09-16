#!/usr/bin/env python3
"""Run the official TRAM prediction stages on one fixed 150-frame RGB folder.

This adapter replaces only the demo's lossy video-to-JPEG conversion with the
already decoded benchmark JPEGs.  Detection/SAM/DEVA, camera calibration,
masked metric DROID-SLAM, gravity alignment, and VIMO are the released TRAM
functions.  Rendering is deliberately excluded.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pycocotools import mask as masktool


TRAM_ROOT = Path(__file__).resolve().parents[3] / "external_baselines/TRAM"
if str(TRAM_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAM_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-humans", type=int, default=20)
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    script_started = time.perf_counter()
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(input_dir.glob("*.jpg"))
    expected = [f"{index:06d}.jpg" for index in range(150)]
    if len(images) != 150 or [path.name for path in images] != expected:
        raise ValueError("TRAM runtime input must contain ordered 000000--000149.jpg")

    # Imports are deliberately inside the timed process.  They initialize the
    # released detector and DEVA models and therefore belong to cold-start
    # single-video wall time.
    from lib.camera import run_metric_slam, calibrate_intrinsics, align_cam_to_world
    from lib.pipeline import detect_segment_track
    from lib.models import get_hmr_vimo

    component: dict[str, float] = {}
    started = time.perf_counter()
    boxes, encoded_masks, track_array = detect_segment_track(
        [str(path) for path in images],
        str(output_dir),
        thresh=0.25,
        min_size=100,
        save_vos=False,
    )
    torch.cuda.synchronize()
    component["detection_segmentation_tracking_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    decoded_masks = np.asarray([masktool.decode(mask) for mask in encoded_masks])
    mask_tensor = torch.from_numpy(decoded_masks)
    camera_intrinsics, is_static = calibrate_intrinsics(
        str(input_dir), mask_tensor, is_static=False
    )
    camera_rotation, camera_translation = run_metric_slam(
        str(input_dir), masks=mask_tensor, calib=camera_intrinsics, is_static=is_static
    )
    world_rotation, world_translation, spec_focal = align_cam_to_world(
        str(images[0]), camera_rotation, camera_translation
    )
    camera = {
        "pred_cam_R": camera_rotation.numpy(),
        "pred_cam_T": camera_translation.numpy(),
        "world_cam_R": world_rotation.numpy(),
        "world_cam_T": world_translation.numpy(),
        "img_focal": camera_intrinsics[0],
        "img_center": camera_intrinsics[2:],
        "spec_focal": spec_focal,
    }
    np.save(output_dir / "camera.npy", camera)
    np.save(output_dir / "boxes.npy", boxes)
    np.save(output_dir / "masks.npy", encoded_masks)
    np.save(output_dir / "tracks.npy", track_array)
    torch.cuda.synchronize()
    component["camera_recovery_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    tracks = track_array.item()
    track_ids = list(tracks)
    rank = np.argsort([len(track) for track in tracks.values()])[::-1]
    ordered_tracks = [tracks[track_ids[index]] for index in rank]
    model = get_hmr_vimo(checkpoint="data/pretrain/vimo_checkpoint.pth.tar")
    hps_dir = output_dir / "hps"
    hps_dir.mkdir(exist_ok=True)
    saved_tracks = 0
    for index, track in enumerate(ordered_tracks):
        valid = np.asarray([item["det"] for item in track])
        track_boxes = np.concatenate([item["det_box"] for item in track])
        frames = np.asarray([item["frame"] for item in track])
        result = model.inference(
            [str(path) for path in images],
            track_boxes,
            valid=valid,
            frame=frames,
            img_focal=camera["img_focal"],
            img_center=camera["img_center"],
        )
        if result is not None:
            np.save(hps_dir / f"hps_track_{index}.npy", result)
            saved_tracks += 1
        if index + 1 >= int(args.max_humans):
            break
    torch.cuda.synchronize()
    component["vimo_human_recovery_seconds"] = time.perf_counter() - started

    native_outputs = [output_dir / "camera.npy", output_dir / "tracks.npy"] + sorted(hps_dir.glob("*.npy"))
    report = {
        "schema_version": "Shot3R-H4D-CS150-TRAM-internal-runtime-v1",
        "case_id": args.case_id,
        "frames": len(images),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "components": component,
        "script_seconds": time.perf_counter() - script_started,
        "is_static": bool(is_static),
        "native_track_count": len(ordered_tracks),
        "saved_hps_tracks": saved_tracks,
        "native_outputs": [str(path) for path in native_outputs],
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        },
    }
    report_path = output_dir / "tram.internal.json"
    report_path.write_text(json.dumps(jsonable(report), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
