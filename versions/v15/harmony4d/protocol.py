#!/usr/bin/env python3
"""Frozen deterministic cross-shot protocol helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from versions.v15.harmony4d.dataset import CameraCalibration


PROTOCOL_NAME = "Movie3R-Harmony4D-CrossShot-v1"
PROTOCOL_SEED = 20260818
CLIP_LENGTH = 150
BOUNDARY_INDEX = 75
ANGLE_STRATA = {
    "small": (0.0, 30.0),
    "medium": (30.0, 60.0),
    "large": (60.0, 120.0),
    "extreme": (120.0, 180.000001),
}


def rotation_span_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first)[:3, :3].T @ np.asarray(second)[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def camera_pair_rows(calibrations: dict[str, CameraCalibration]) -> list[dict[str, Any]]:
    names = sorted(calibrations)
    rows = []
    for first in names:
        for second in names:
            if first == second:
                continue
            a, b = calibrations[first], calibrations[second]
            angle = rotation_span_deg(a.camera_to_world, b.camera_to_world)
            baseline = float(np.linalg.norm(a.camera_to_world[:3, 3] - b.camera_to_world[:3, 3]))
            stratum = next(
                (name for name, (low, high) in ANGLE_STRATA.items() if low <= angle < high),
                None,
            )
            if stratum is not None:
                rows.append(
                    {
                        "pre_camera": first,
                        "post_camera": second,
                        "angle_deg": angle,
                        "baseline_m": baseline,
                        "angle_stratum": stratum,
                    }
                )
    return rows


def select_balanced_pairs(
    calibrations: dict[str, CameraCalibration],
    visibility: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    visibility = visibility or {name: 1.0 for name in calibrations}
    output = []
    for stratum, (low, high) in ANGLE_STRATA.items():
        target = 0.5 * (low + min(high, 180.0))
        eligible = [
            row for row in camera_pair_rows(calibrations)
            if row["angle_stratum"] == stratum
            and visibility.get(row["pre_camera"], 0.0) >= 0.01
            and visibility.get(row["post_camera"], 0.0) >= 0.01
        ]
        if not eligible:
            continue
        eligible.sort(
            key=lambda row: (
                abs(float(row["angle_deg"]) - target),
                -min(visibility[row["pre_camera"]], visibility[row["post_camera"]]),
                row["pre_camera"],
                row["post_camera"],
            )
        )
        output.append(eligible[0])
    return output


def manifest_sha256(rows: list[dict[str, Any]]) -> str:
    canonical = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")
    return manifest_sha256(rows)

