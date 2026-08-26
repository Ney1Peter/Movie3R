#!/usr/bin/env python3
"""Build immutable RGB-PTS to AIST-60-Hz GT mappings for frozen sources.

The AIST MP4 containers run at 59.94 FPS while the official motion/keypoint
arrays use a 60-Hz time axis.  This builder probes decoded video-frame PTSs,
then records the nearest decodable RGB frame for every frozen 30-FPS output
time.  It never trusts container frame number as a GT index.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .protocol import (
        DEFAULT_BUNDLE_ROOT,
        DEFAULT_DERIVED_ROOT,
        GT_FPS,
        OUTPUT_FRAMES,
        PROTOCOL_NAME,
        atomic_json,
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
        GT_FPS,
        OUTPUT_FRAMES,
        PROTOCOL_NAME,
        atomic_json,
        canonical_json_digest,
        load_frozen_sources,
        output_gt_ticks,
        sha256_file,
        source_video_path,
        verify_input_manifest_freeze,
    )


SCHEMA = "Bridge3R-AIST-SinglePerson-frame-map-v1"
SUMMARY_SCHEMA = "Bridge3R-AIST-SinglePerson-frame-map-summary-v1"
# Half of one 59.94-FPS source interval plus a conservative 2 ms tolerance.
MAX_ABSOLUTE_PTS_ERROR_SECONDS = 0.0105


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
    parser.add_argument("--derived-root", type=Path, default=DEFAULT_DERIVED_ROOT)
    parser.add_argument(
        "--roles", default="pilot", help="Comma-separated frozen roles: pilot,test."
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Concurrent ffprobe processes (1--8)."
    )
    parser.add_argument(
        "--source-id", action="append", default=[], help="Optional exact frozen source ID; repeatable."
    )
    parser.add_argument("--force", action="store_true", help="Replace completed derived map artifacts only.")
    return parser.parse_args()


def ffprobe_pts(path: Path) -> np.ndarray:
    """Return a strictly monotone decoded-frame PTS vector in seconds."""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "frame=best_effort_timestamp_time",
        "-of",
        "csv=p=0",
        str(path),
    ]
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    values: list[float] = []
    for line in completed.stdout.splitlines():
        token = line.split(",", 1)[0].strip()
        if not token or token == "N/A":
            continue
        values.append(float(token))
    pts = np.asarray(values, dtype=np.float64)
    if pts.size < OUTPUT_FRAMES or not np.isfinite(pts).all():
        raise ValueError(f"Invalid decoded PTS vector for {path}: count={pts.size}")
    if not np.all(np.diff(pts) > 0):
        raise ValueError(f"Decoded video PTS are not strictly increasing: {path}")
    return pts


def nearest_indices(pts: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    right = np.searchsorted(pts, targets, side="left")
    right = np.clip(right, 0, len(pts) - 1)
    left = np.clip(right - 1, 0, len(pts) - 1)
    choose_left = np.abs(pts[left] - targets) <= np.abs(pts[right] - targets)
    indices = np.where(choose_left, left, right).astype(np.int64)
    return indices, pts[indices] - targets


def build_source_map(
    bundle_root: Path, derived_root: Path, source: dict[str, Any], input_hashes: dict[str, str]
) -> dict[str, Any]:
    source_id = str(source["source_id"])
    ticks = output_gt_ticks(source)
    targets = ticks.astype(np.float64) / float(GT_FPS)
    videos: dict[str, Any] = {}
    for camera_id in source["camera_ids"]:
        path = source_video_path(bundle_root, source, str(camera_id))
        pts = ffprobe_pts(path)
        indices, errors = nearest_indices(pts, targets)
        if not np.all(np.diff(indices) > 0):
            raise ValueError(f"Non-monotone selected decode index for {source_id}/{camera_id}")
        max_error = float(np.abs(errors).max())
        if max_error > MAX_ABSOLUTE_PTS_ERROR_SECONDS:
            raise ValueError(
                f"PTS mapping error too large for {source_id}/{camera_id}: "
                f"{max_error:.6f}s > {MAX_ABSOLUTE_PTS_ERROR_SECONDS:.6f}s"
            )
        videos[str(camera_id)] = {
            "source_video": str(path.relative_to(bundle_root)),
            "source_video_sha256": sha256_file(path),
            "decoded_frame_count": int(len(pts)),
            "first_pts_seconds": float(pts[0]),
            "last_pts_seconds": float(pts[-1]),
            "output_decode_indices": indices.tolist(),
            "output_pts_seconds": [round(float(value), 9) for value in pts[indices]],
            "signed_pts_error_seconds": [round(float(value), 9) for value in errors],
            "mean_absolute_pts_error_seconds": float(np.abs(errors).mean()),
            "max_absolute_pts_error_seconds": max_error,
        }
    result = {
        "schema_version": SCHEMA,
        "protocol": PROTOCOL_NAME,
        "input_manifest_sha256": input_hashes,
        "source_id": source_id,
        "role": source["role"],
        "split": source["split"],
        "sequence_name": source["sequence_name"],
        "camera_ids": source["camera_ids"],
        "output_fps": 30,
        "output_num_frames": OUTPUT_FRAMES,
        "gt_fps": GT_FPS,
        "target_gt_ticks": ticks.tolist(),
        "target_times_seconds": [round(float(value), 9) for value in targets],
        "mapping_rule": "nearest decoded RGB PTS to official_gt_tick / 60",
        "maximum_allowed_absolute_pts_error_seconds": MAX_ABSOLUTE_PTS_ERROR_SECONDS,
        "videos": videos,
    }
    result["content_sha256"] = canonical_json_digest(result)
    output = derived_root / "frame_maps/aist" / str(source["role"]) / f"{source_id.split(':', 1)[1]}.json"
    return {"output": output, "payload": result}


def main() -> None:
    args = parse_args()
    if not 1 <= args.workers <= 8:
        raise SystemExit("--workers must be between 1 and 8")
    roles = tuple(token.strip() for token in args.roles.split(",") if token.strip())
    bundle_root = args.bundle_root.resolve()
    derived_root = args.derived_root.resolve()
    input_hashes = verify_input_manifest_freeze(bundle_root)
    sources = load_frozen_sources(bundle_root, roles)
    if args.source_id:
        allowed = set(args.source_id)
        unknown = allowed - {str(row["source_id"]) for row in sources}
        if unknown:
            raise SystemExit(f"--source-id is not in selected roles: {sorted(unknown)}")
        sources = [row for row in sources if str(row["source_id"]) in allowed]

    completed: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for source in sources:
        path = derived_root / "frame_maps/aist" / str(source["role"]) / f"{str(source['source_id']).split(':', 1)[1]}.json"
        if path.is_file() and not args.force:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("input_manifest_sha256") != input_hashes:
                raise ValueError(f"Existing map has stale input hashes: {path}")
            completed.append({"source_id": source["source_id"], "output": str(path), "reused": True})
        else:
            pending.append(source)

    def worker(source: dict[str, Any]) -> dict[str, Any]:
        result = build_source_map(bundle_root, derived_root, source, input_hashes)
        atomic_json(result["output"], result["payload"])
        return {
            "source_id": source["source_id"],
            "role": source["role"],
            "output": str(result["output"]),
            "reused": False,
            "max_absolute_pts_error_seconds": max(
                row["max_absolute_pts_error_seconds"]
                for row in result["payload"]["videos"].values()
            ),
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(worker, source): source for source in pending}
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            completed.append(future.result())
            print(f"frame maps: {index}/{len(pending)} newly complete", flush=True)

    rows = sorted(completed, key=lambda row: str(row["source_id"]))
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "protocol": PROTOCOL_NAME,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input_manifest_sha256": input_hashes,
        "roles": list(roles),
        "source_count": len(rows),
        "source_ids": [row["source_id"] for row in rows],
        "maps": rows,
    }
    summary["content_sha256"] = canonical_json_digest(summary)
    summary_path = derived_root / "frame_maps/aist" / f"summary_{'_'.join(sorted(roles))}.json"
    atomic_json(summary_path, summary)
    print(json.dumps({"summary": str(summary_path), "source_count": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
