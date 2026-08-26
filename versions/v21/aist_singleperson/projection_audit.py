#!/usr/bin/env python3
"""Render calibration/GT overlays for the frozen AIST++ frame-map audit.

This evaluator-only utility confirms that the official 3D keypoints, camera
calibration, selected decoded RGB frame, coordinate axis and lens distortion
are mutually coherent.  It does not inspect a prediction and does not alter a
runtime manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from .protocol import (
        DEFAULT_BUNDLE_ROOT,
        DEFAULT_DERIVED_ROOT,
        camera_records,
        load_frozen_sources,
        output_gt_ticks,
        source_video_path,
        verify_input_manifest_freeze,
    )
    from .derive_sequences import load_checked_frame_map
except ImportError:  # Direct script execution from this directory.
    from protocol import (  # type: ignore
        DEFAULT_BUNDLE_ROOT,
        DEFAULT_DERIVED_ROOT,
        camera_records,
        load_frozen_sources,
        output_gt_ticks,
        source_video_path,
        verify_input_manifest_freeze,
    )
    from derive_sequences import load_checked_frame_map  # type: ignore


SCHEMA = "Bridge3R-AIST-SinglePerson-calibration-projection-audit-v1"
AUDIT_SALT = "bridge3r-aist-projection-audit-v1"
AUDIT_FRAMES = (0, 74, 75, 149)  # Both sides of the CS150 boundary.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
    parser.add_argument("--derived-root", type=Path, default=DEFAULT_DERIVED_ROOT)
    parser.add_argument("--roles", default="pilot", help="Comma-separated frozen roles: pilot,test.")
    parser.add_argument("--count", type=int, default=20, help="Number of source-level audits to select deterministically.")
    return parser.parse_args()


def source_rank(source: dict[str, Any]) -> str:
    return hashlib.sha256(f"{AUDIT_SALT}|{source['source_id']}".encode()).hexdigest()


def read_frame(path: Path, index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open RGB for projection audit: {path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, image = capture.read()
    capture.release()
    if not ok or image is None:
        raise RuntimeError(f"Cannot decode frame {index} from {path}")
    return image


def project_keypoints(points: np.ndarray, camera: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    K = np.asarray(camera["matrix"], dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(camera["distortions"], dtype=np.float64).reshape(-1, 1)
    rvec = np.asarray(camera["rotation"], dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(camera["translation"], dtype=np.float64).reshape(3, 1)
    projected, _ = cv2.projectPoints(points.astype(np.float64), rvec, tvec, K, distortion)
    rotation, _ = cv2.Rodrigues(rvec)
    depths = points @ rotation.T + tvec.reshape(1, 3)
    return projected.reshape(-1, 2), depths[:, 2]


def draw_overlay(image: np.ndarray, uv: np.ndarray, depth: np.ndarray, title: str) -> tuple[np.ndarray, dict[str, float]]:
    output = image.copy()
    height, width = output.shape[:2]
    finite = np.isfinite(uv).all(axis=1) & np.isfinite(depth) & (depth > 0)
    in_image = finite & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    for joint, (x, y) in enumerate(uv[in_image]):
        cv2.circle(output, (int(round(x)), int(round(y))), 5, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(output, str(joint), (int(round(x)) + 5, int(round(y)) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (20, 20, 20), 1, cv2.LINE_AA)
    mask = in_image
    if np.any(mask):
        lo = np.floor(uv[mask].min(axis=0)).astype(int)
        hi = np.ceil(uv[mask].max(axis=0)).astype(int)
        cv2.rectangle(output, tuple(lo), tuple(hi), (0, 255, 255), 2)
    band = output.copy()
    cv2.rectangle(band, (0, 0), (width, 52), (0, 0, 0), -1)
    output = cv2.addWeighted(band, 0.58, output, 0.42, 0)
    cv2.putText(output, title, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(output, f"official 3D keypoints: in-image={float(in_image.mean()):.3f}", (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 0), 1, cv2.LINE_AA)
    return output, {"in_image_joint_fraction": float(in_image.mean()), "positive_depth_joint_fraction": float(finite.mean())}


def audit_source(bundle_root: Path, derived_root: Path, source: dict[str, Any], input_hashes: dict[str, str]) -> dict[str, Any]:
    frame_map = load_checked_frame_map(derived_root, source, input_hashes)
    ticks = output_gt_ticks(source)
    with (bundle_root / source["keypoints3d_path"]).open("rb") as stream:
        keypoints = np.asarray(pickle.load(stream)["keypoints3d"], dtype=np.float64)
    records = camera_records(bundle_root, source)
    sequence = str(source["source_id"]).split(":", 1)[1]
    output_dir = derived_root / "audits/aist/projection" / str(source["role"]) / sequence
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    panels = []
    for output_frame in AUDIT_FRAMES:
        camera_id = str(source["camera_ids"][0 if output_frame < 75 else 1])
        mapping = frame_map["videos"][camera_id]
        decode_index = int(mapping["output_decode_indices"][output_frame])
        image = read_frame(source_video_path(bundle_root, source, camera_id), decode_index)
        uv, depth = project_keypoints(keypoints[int(ticks[output_frame])], records[camera_id])
        title = f"{sequence} | out={output_frame} GT={int(ticks[output_frame])} {camera_id} decode={decode_index}"
        overlay, metrics = draw_overlay(image, uv, depth, title)
        overlay_path = output_dir / f"cs150_f{output_frame:03d}_{camera_id}.png"
        cv2.imwrite(str(overlay_path), overlay)
        thumbnail = cv2.resize(overlay, (960, 540), interpolation=cv2.INTER_AREA)
        panels.append(thumbnail)
        outputs.append({
            "output_frame": output_frame, "gt_tick": int(ticks[output_frame]), "camera_id": camera_id,
            "decode_index": decode_index, "rgb_pts_seconds": mapping["output_pts_seconds"][output_frame],
            "overlay": str(overlay_path.relative_to(derived_root)), **metrics,
        })
    contact = np.concatenate([np.concatenate(panels[:2], axis=1), np.concatenate(panels[2:], axis=1)], axis=0)
    contact_path = output_dir / "cs150_boundary_contact_sheet.png"
    cv2.imwrite(str(contact_path), contact)
    return {
        "source_id": source["source_id"], "role": source["role"], "split": source["split"],
        "selection_hash": source_rank(source), "frames": outputs,
        "contact_sheet": str(contact_path.relative_to(derived_root)),
        "minimum_in_image_joint_fraction": min(row["in_image_joint_fraction"] for row in outputs),
    }


def main() -> None:
    args = parse_args()
    if args.count < 1:
        raise SystemExit("--count must be positive")
    roles = tuple(value.strip() for value in args.roles.split(",") if value.strip())
    bundle_root, derived_root = args.bundle_root.resolve(), args.derived_root.resolve()
    input_hashes = verify_input_manifest_freeze(bundle_root)
    candidates = load_frozen_sources(bundle_root, roles)
    selected = sorted(candidates, key=source_rank)[: min(args.count, len(candidates))]
    rows = [audit_source(bundle_root, derived_root, source, input_hashes) for source in selected]
    payload = {
        "schema_version": SCHEMA, "protocol": "Bridge3R-AIST-SinglePerson-v1", "input_manifest_sha256": input_hashes,
        "selection_salt": AUDIT_SALT, "requested_count": args.count, "selected_source_count": len(rows),
        "roles": list(roles), "selection_rule": "lowest SHA256(audit_salt|source_id), independent of model outputs",
        "checks": ["official 3D keypoints projected with official K/R/t/distortion", "boundary-adjacent RGB frames", "manual visual review required before acceptance"],
        "sources": rows,
    }
    output = derived_root / "audits/aist/projection" / f"projection_audit_{'_'.join(sorted(roles))}.json"
    payload["content_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"audit": str(output), "source_count": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
