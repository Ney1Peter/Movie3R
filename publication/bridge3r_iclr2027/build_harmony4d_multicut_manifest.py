#!/usr/bin/env python3
"""Freeze the GT-only auxiliary Harmony4D three-shot manifest.

This builder consumes only the calibration/visibility audits emitted by the
existing Harmony4D staging code.  It is intentionally independent of every
model prediction and result file; see ``MULTICUT_PROTOCOL.md`` for the
pre-registered selection rule.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


PROTOCOL = "Bridge3R-Harmony4D-MultiCut-v1"
SEED = 20260825
SHOT_LENGTH = 50
SHOT_COUNT = 3
MIN_VISIBLE_VERTEX_FRACTION = 0.01


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rotation_angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first, dtype=np.float64).T @ np.asarray(second, dtype=np.float64)
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def camera_payload(audit: dict[str, Any], name: str) -> dict[str, Any]:
    payload = dict(audit["cameras"][name])
    matrix = np.asarray(payload["camera_to_world"], dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{audit['archive_entry']}:{name}: invalid camera_to_world")
    visible = [
        float(row["visible_vertex_fraction"])
        for row in payload.get("people_projection", {}).values()
    ]
    if not visible or min(visible) < MIN_VISIBLE_VERTEX_FRACTION:
        raise ValueError(f"{audit['archive_entry']}:{name}: inadequate audited visibility")
    return {"matrix": matrix, "minimum_visible_fraction": min(visible)}


def transition(first: dict[str, Any], second: dict[str, Any]) -> dict[str, float]:
    rotation = rotation_angle_degrees(first["matrix"][:3, :3], second["matrix"][:3, :3])
    baseline = float(np.linalg.norm(first["matrix"][:3, 3] - second["matrix"][:3, 3]))
    return {"rotation_deg": rotation, "baseline_m": baseline}


def select_triplet(audit: dict[str, Any]) -> tuple[list[str], list[dict[str, float]]]:
    candidates: dict[str, dict[str, Any]] = {}
    for name in sorted(audit.get("cameras", {})):
        try:
            candidates[name] = camera_payload(audit, name)
        except ValueError:
            continue
    feasible: list[tuple[tuple[float, float, str, str, str], list[str], list[dict[str, float]]]] = []
    for names in itertools.permutations(sorted(candidates), SHOT_COUNT):
        first = transition(candidates[names[0]], candidates[names[1]])
        second = transition(candidates[names[1]], candidates[names[2]])
        transitions = [first, second]
        if min(item["baseline_m"] for item in transitions) <= 0.0:
            continue
        # Negative values implement the required descending sort without
        # looking at any model output.  Names make ties deterministic.
        key = (
            -min(item["rotation_deg"] for item in transitions),
            -min(item["baseline_m"] for item in transitions),
            *names,
        )
        feasible.append((key, list(names), transitions))
    if not feasible:
        raise ValueError(f"{audit['archive_entry']}: no visibility-valid camera triplet")
    _, names, transitions = min(feasible, key=lambda item: item[0])
    return names, transitions


def build_record(audit_path: Path) -> dict[str, Any]:
    audit = json.loads(audit_path.read_text(encoding="utf-8-sig"))
    required = {
        "archive_entry", "capture_relative", "capture_group_name", "sequence_root_name",
        "frame_min", "frame_max", "frames_contiguous", "fps", "identities", "cameras",
        "projection_audit",
    }
    missing = required.difference(audit)
    if missing:
        raise ValueError(f"{audit_path}: missing {sorted(missing)}")
    if not audit["frames_contiguous"] or not audit["projection_audit"].get("pass"):
        raise ValueError(f"{audit_path}: capture did not pass structural/calibration audit")
    start = int(audit["frame_min"])
    total = SHOT_COUNT * SHOT_LENGTH
    if int(audit["frame_max"]) < start + total - 1:
        raise ValueError(f"{audit_path}: insufficient contiguous frames for {total}-frame multi-cut")
    cameras, transitions = select_triplet(audit)
    shots = [list(range(start + SHOT_LENGTH * index, start + SHOT_LENGTH * (index + 1))) for index in range(SHOT_COUNT)]
    case_id = "h4d_multicut_" + "_".join(
        [str(audit["capture_group_name"]), str(audit["sequence_root_name"]), *cameras]
    )
    return {
        "protocol": PROTOCOL,
        "protocol_seed": SEED,
        "case_id": case_id,
        "archive_entry": str(audit["archive_entry"]),
        "capture_relative": str(audit["capture_relative"]),
        "sequence": str(audit["capture_group_name"]),
        "capture": str(audit["sequence_root_name"]),
        "fps": float(audit["fps"]),
        "shot_cameras": cameras,
        "shot_frame_numbers": shots,
        "boundaries": [SHOT_LENGTH, SHOT_LENGTH * 2],
        "clip_length": total,
        "identities_evaluator_only": list(audit["identities"]),
        "camera_transitions_evaluator_only": transitions,
        "selection_depends_on_model_result": False,
        "gt_available_to_runtime": False,
        "selection": {
            "capture_rule": "first coordinate-valid capture in pre-existing SHA256 structural order",
            "camera_rule": (
                "maximum minimum adjacent rotation, then maximum minimum adjacent baseline, "
                "then lexicographic camera names; calibration and visibility only"
            ),
            "minimum_visible_vertex_fraction": MIN_VISIBLE_VERTEX_FRACTION,
            "audit": str(audit_path),
            "audit_sha256": sha256(audit_path),
        },
    }


def build_no_cut_record(record: dict[str, Any]) -> dict[str, Any]:
    """Derive a detector-negative control without inspecting any prediction.

    The control reuses the selected capture, first selected camera, and exact
    150-frame interval of its three-shot counterpart. It has no transaction
    boundary and is therefore a test of detector false positives and strict
    no-cut invariance, not an additional selection opportunity.
    """

    frames = [frame for shot in record["shot_frame_numbers"] for frame in shot]
    return {
        "protocol": PROTOCOL + "-NoCutControl",
        "protocol_seed": SEED,
        "case_id": str(record["case_id"]) + "_nocut",
        "archive_entry": record["archive_entry"],
        "capture_relative": record["capture_relative"],
        "sequence": record["sequence"],
        "capture": record["capture"],
        "fps": record["fps"],
        "camera": record["shot_cameras"][0],
        "frame_numbers": frames,
        "boundaries": [],
        "clip_length": len(frames),
        "identities_evaluator_only": record["identities_evaluator_only"],
        "selection_depends_on_model_result": False,
        "gt_available_to_runtime": False,
        "selection": {
            "control_rule": (
                "reuse the first selected multi-cut camera and the same contiguous "
                "150-frame interval; no model result consulted"
            ),
            "source_multicut_case_id": record["case_id"],
            "source_audit": record["selection"]["audit"],
            "source_audit_sha256": record["selection"]["audit_sha256"],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audits", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--no-cut-output", type=Path, default=None,
        help="Optional frozen detector-negative manifest derived from the same cases.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = [build_record(path.resolve()) for path in args.audits]
    records.sort(key=lambda row: (row["archive_entry"], row["capture_relative"]))
    if len({row["case_id"] for row in records}) != len(records):
        raise ValueError("duplicate multi-cut case IDs")
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    partial.write_text(text, encoding="utf-8")
    partial.replace(args.output)
    spec = {
        "schema_version": "Bridge3R-Harmony4D-MultiCut-manifest-v1",
        "protocol": PROTOCOL,
        "seed": SEED,
        "case_count": len(records),
        "shot_count": SHOT_COUNT,
        "shot_length": SHOT_LENGTH,
        "clip_length": SHOT_COUNT * SHOT_LENGTH,
        "boundaries": [SHOT_LENGTH, SHOT_LENGTH * 2],
        "manifest": str(args.output.resolve()),
        "manifest_sha256": sha256(args.output),
        "construction": "calibration/GT visibility only; no model result used",
        "runtime_gt_access": False,
    }
    spec_path = args.output.with_suffix(".spec.json")
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    if args.no_cut_output is not None:
        controls = [build_no_cut_record(record) for record in records]
        controls.sort(key=lambda row: row["case_id"])
        control_text = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in controls
        )
        control_path = args.no_cut_output
        control_path.parent.mkdir(parents=True, exist_ok=True)
        partial = control_path.with_suffix(control_path.suffix + ".partial")
        partial.write_text(control_text, encoding="utf-8")
        partial.replace(control_path)
        control_spec = {
            "schema_version": "Bridge3R-Harmony4D-NoCut-manifest-v1",
            "protocol": PROTOCOL + "-NoCutControl",
            "seed": SEED,
            "case_count": len(controls),
            "clip_length": SHOT_COUNT * SHOT_LENGTH,
            "boundaries": [],
            "manifest": str(control_path.resolve()),
            "manifest_sha256": sha256(control_path),
            "construction": (
                "first selected multi-cut camera and identical frame interval; "
                "no model result used"
            ),
            "runtime_gt_access": False,
        }
        control_path.with_suffix(".spec.json").write_text(
            json.dumps(control_spec, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(spec, indent=2))


if __name__ == "__main__":
    main()
