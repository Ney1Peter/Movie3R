#!/usr/bin/env python3
"""Validate and freeze the derived AIST++ v1 dataset card and checksums."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    from .protocol import DEFAULT_BUNDLE_ROOT, DEFAULT_DERIVED_ROOT, PROTOCOLS, atomic_json, canonical_json_digest, load_frozen_sources, sha256_file, verify_input_manifest_freeze
    from .derive_sequences import output_paths
except ImportError:
    from protocol import DEFAULT_BUNDLE_ROOT, DEFAULT_DERIVED_ROOT, PROTOCOLS, atomic_json, canonical_json_digest, load_frozen_sources, sha256_file, verify_input_manifest_freeze  # type: ignore
    from derive_sequences import output_paths  # type: ignore


SCHEMA = "Bridge3R-AIST-SinglePerson-dataset-finalization-v1"


def check_case(root: Path, source: dict, protocol: str) -> list[Path]:
    video, label, metadata = output_paths(root, source, protocol)
    for path in (video, label, metadata):
        if not path.is_file():
            raise FileNotFoundError(path)
    with np.load(label, allow_pickle=False) as values:
        expected = {
            "gt_tick_60fps", "target_time_seconds", "rgb_pts_seconds", "source_decode_index", "shot_id", "camera_id",
            "cut_indices_evaluator_only", "world_keypoints_m", "smpl_root_translation_m", "smpl_root_orientation_axis_angle",
            "smpl_body_pose_axis_angle", "camera_intrinsics", "camera_world_to_camera_m", "camera_camera_to_world_m",
        }
        missing = expected.difference(values.files)
        if missing:
            raise KeyError(f"{label} misses {sorted(missing)}")
        if values["world_keypoints_m"].shape != (150, 17, 3) or values["camera_camera_to_world_m"].shape != (150, 4, 4):
            raise ValueError(f"Unexpected AIST label shape: {label}")
    name = str(source["source_id"]).split(":", 1)[1]
    case = root / "cases/aist" / str(source["role"]) / f"{name}.json"
    if not case.is_file():
        raise FileNotFoundError(case)
    return [video, label, metadata, case]


def main() -> None:
    bundle_root, root = DEFAULT_BUNDLE_ROOT.resolve(), DEFAULT_DERIVED_ROOT.resolve()
    input_hashes = verify_input_manifest_freeze(bundle_root)
    files: list[Path] = []
    counts = {}
    for role in ("pilot", "test"):
        sources = load_frozen_sources(bundle_root, (role,))
        counts[role] = len(sources)
        for source in sources:
            for protocol in PROTOCOLS:
                files.extend(check_case(root, source, protocol))
        for protocol in PROTOCOLS:
            for suffix in ("runtime.jsonl", "evaluator.jsonl", "spec.json"):
                path = root / "manifests" / f"aist_{protocol.lower()}_{role}.{suffix}"
                if not path.is_file():
                    raise FileNotFoundError(path)
                files.append(path)
    audit = root / "audits/aist/projection/projection_audit_pilot_test.json"
    if not audit.is_file():
        raise FileNotFoundError(audit)
    audit_data = json.loads(audit.read_text(encoding="utf-8"))
    if int(audit_data.get("selected_source_count", 0)) < 20:
        raise ValueError("Projection audit has fewer than 20 selected sources")
    if min(float(row["minimum_in_image_joint_fraction"]) for row in audit_data["sources"]) < 1.0:
        raise ValueError("Projection audit contains an out-of-image official keypoint")
    files.append(audit)
    for role in ("pilot", "test"):
        files.extend(sorted((root / "frame_maps/aist" / role).glob("*.json")))
    files = sorted(set(path.resolve() for path in files))
    checksum_lines = [f"{sha256_file(path)}  {path.relative_to(root)}" for path in files]
    checksums = root / "CHECKSUMS.sha256"
    checksums.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    inventory = {
        "schema_version": SCHEMA, "input_manifest_sha256": input_hashes, "case_counts": counts,
        "protocols": {name: {"shot_lengths": list(spec["shot_lengths"]), "cut_indices_evaluator_only": list(spec["cut_indices"])} for name, spec in PROTOCOLS.items()},
        "checked_file_count": len(files), "checksums": str(checksums.relative_to(root)), "checksums_sha256": sha256_file(checksums),
        "projection_audit": {"path": str(audit.relative_to(root)), "sha256": sha256_file(audit), "source_count": int(audit_data["selected_source_count"]), "minimum_in_image_joint_fraction": min(float(row["minimum_in_image_joint_fraction"]) for row in audit_data["sources"])},
    }
    inventory["content_sha256"] = canonical_json_digest(inventory)
    atomic_json(root / "DERIVED_INVENTORY_v1.json", inventory)
    card = f"""# Bridge3R AIST++ Derived Single-Person Benchmark v1

## Scope

This derived set evaluates continuous single-person motion under physical
camera changes in the same AIST Dance Video Database capture.  It supports
Camera--Human reconstruction assessment only; it does not provide a dense
scene-reconstruction benchmark or claim cross-scene editing.

## Frozen source protocol

- Official AIST++ `pose_test`: {counts['test']} source sequences, used only for final metrics.
- Official AIST++ `pose_val`: {counts['pilot']} disjoint source sequences, used only for runtime/interface pilot checks.
- Every source has one frozen 5-second window, 150 output frames at 30 FPS,
  and a pre-frozen four-camera tuple spanning low/mid/high relative-view
  transitions.  Source, window and camera tuples are immutable in the input
  manifest; no reserve source may replace a failed method case.
- `CS150` contains 75/75 frames with one evaluator-only boundary after frame
  74. `MC150-3` contains 50/50/50 frames, and `MC150-4` contains 38/38/37/37
  frames.  All protocols retain the same physical action timestamps.

## Time and geometry

The source MP4 containers are 59.94 FPS while official motion/keypoint labels
use a 60-Hz index.  Each derived frame is selected using the stored rule
`nearest decoded RGB PTS <-> official_gt_tick / 60`; container frame indices
are never treated as GT indices.  The maximum permitted PTS residual is
10.5 ms.  Labels store official 3-D keypoints, SMPL motion, camera intrinsics,
world-to-camera and camera-to-world transforms in metres.  Runtime manifests
contain only case ID, RGB MP4, FPS and length; GT, camera IDs and cuts are
evaluator-only.

## Calibration acceptance

The deterministic projection audit selected
{inventory['projection_audit']['source_count']} pilot/test sources.  At both
sides of the CS150 boundary and at clip endpoints, all 17 official 3-D
keypoints projected inside their corresponding official RGB image
(minimum in-image fraction = {inventory['projection_audit']['minimum_in_image_joint_fraction']:.3f}).
Manual visual checks additionally reviewed a frontal and a side-view contact
sheet; pose overlays align with the dancers and camera changes.

## Checksums and license

`CHECKSUMS.sha256` covers {inventory['checked_file_count']} derived frame-map,
video, label, case and manifest artifacts.  The original AIST RGB/video terms
remain controlling; neither RGB nor raw annotations belong in a paper source
archive.  Use requires the AIST Dance Video Database Terms of Use accepted by
the data controller.
"""
    (root / "DATASET_CARD_v1.md").write_text(card, encoding="utf-8")
    print(json.dumps({"inventory": str(root / "DERIVED_INVENTORY_v1.json"), "dataset_card": str(root / "DATASET_CARD_v1.md"), "checksums": str(checksums), "checked_files": len(files)}, indent=2))


if __name__ == "__main__":
    main()
