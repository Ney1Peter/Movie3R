#!/usr/bin/env python3
"""Evaluator-only metrics for one frozen MVHuman MVH150 case."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from versions.v21.aist_singleperson import evaluate_aist as shared  # noqa: E402


SCHEMA = "Bridge3R-MVHuman-Heldout-MVH150-evaluation-v1"
EXPECTED_FRAMES = 150
BODY25_TO_COMMON12 = np.asarray([5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11], dtype=np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--runtime-report", type=Path, required=True)
    parser.add_argument("--evaluator-manifest", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
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
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def manifest_row(path: Path, case_id: str) -> dict[str, Any]:
    matches = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("case_id") == case_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one evaluator row for {case_id}, found {len(matches)}")
    row = matches[0]
    if row.get("dataset") != "MVHuman" or row.get("protocol") != "MVH150" or row.get("role") != "test":
        raise ValueError("Evaluator row is outside the frozen MVH150 Test protocol")
    if row.get("cut_indices_evaluator_only") != [74] or len(row.get("source_frame_indices", [])) != EXPECTED_FRAMES:
        raise ValueError("MVH150 evaluator timeline drifted")
    return row


def camera_c2w(value: dict[str, Any]) -> np.ndarray:
    rotation = np.asarray(value["rotation"], dtype=np.float64)
    translation = np.asarray(value["translation"], dtype=np.float64).reshape(3) / 1000.0
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation.T
    pose[:3, 3] = -(rotation.T @ translation)
    return pose


def camera_name(key: str) -> str:
    return key.rsplit(".", 1)[0].split("_")[-1]


def load_label(row: dict[str, Any], audit_root: Path) -> dict[str, np.ndarray]:
    subject_root = audit_root / "metadata" / str(row["subject"])
    scale = float(pickle.load(open(subject_root / "camera_scale.pkl", "rb")))
    extrinsics_raw = json.loads((subject_root / "camera_extrinsics.json").read_text(encoding="utf-8"))
    extrinsics = {camera_name(key): value for key, value in extrinsics_raw.items()}
    world = np.full((EXPECTED_FRAMES, 17, 3), np.nan, dtype=np.float64)
    cameras = np.empty((EXPECTED_FRAMES, 4, 4), dtype=np.float64)
    confidence = np.empty((EXPECTED_FRAMES, 12), dtype=np.float64)
    for output_index, (source_index, camera) in enumerate(zip(row["source_frame_indices"], row["camera_timeline"])):
        path = subject_root / "smplx" / "keypoints3d" / f"{int(source_index):06d}.json"
        people = json.loads(path.read_text(encoding="utf-8"))
        if len(people) != 1:
            raise ValueError(f"Expected one GT person in {path}")
        points = np.asarray(people[0]["keypoints3d"], dtype=np.float64)
        body12 = points[BODY25_TO_COMMON12, :3] / scale
        confidence[output_index] = points[BODY25_TO_COMMON12, 3]
        body12[confidence[output_index] <= 0.2] = np.nan
        world[output_index, shared.COCO_BODY12] = body12
        cameras[output_index] = camera_c2w(extrinsics[str(camera)])
    return {
        "world_keypoints_m": world,
        "camera_camera_to_world_m": cameras,
        "cut_indices_evaluator_only": np.asarray([74], dtype=np.int64),
        "common12_confidence": confidence,
    }


def load_arrays(cache: np.lib.npyio.NpzFile, method: str) -> dict[str, np.ndarray]:
    prefix = method + "__"
    required = ("cameras_c2w", "joints_world", "persistent_ids", "valid")
    missing = [prefix + key for key in required if prefix + key not in cache.files]
    if missing:
        raise KeyError(f"{method} cache lacks {missing}")
    return {key: np.asarray(cache[prefix + key]) for key in required}


def main() -> None:
    args = parse_args()
    row = manifest_row(args.evaluator_manifest.resolve(), args.case_id)
    report = json.loads(args.runtime_report.read_text(encoding="utf-8"))
    record = report.get("record", {})
    if record.get("case_id") != args.case_id or record.get("protocol") != "MVH150":
        raise ValueError("Runtime report/evaluator case mismatch")
    label = load_label(row, args.audit_root.resolve())
    methods, errors = {}, {}
    with np.load(args.cache, allow_pickle=False) as cache:
        for method in report.get("methods", []):
            try:
                methods[str(method)] = shared.evaluate_method(str(method), load_arrays(cache, str(method)), label, report)
            except Exception as error:
                errors[str(method)] = f"{type(error).__name__}: {error}"
    payload = {
        "schema_version": SCHEMA,
        "protocol": "MVH150",
        "case_id": args.case_id,
        "subject": row["subject"],
        "angle_stratum": row["angle_stratum"],
        "viewpoint_angle_deg": row["viewpoint_angle_deg"],
        "camera_rotation_geodesic_deg": row.get(
            "camera_rotation_geodesic_deg", row["viewpoint_angle_deg"]
        ),
        "optical_axis_angle_deg": row.get("optical_axis_angle_deg"),
        "angle_measure": row.get(
            "angle_measure", "SO(3) geodesic between calibrated camera rotations"
        ),
        "methods": methods,
        "errors": errors,
        "detector": shared.detector_metrics(report, label["cut_indices_evaluator_only"]),
        "inputs": {
            "cache": str(args.cache.resolve()), "cache_sha256": sha256(args.cache.resolve()),
            "runtime_report": str(args.runtime_report.resolve()), "runtime_report_sha256": sha256(args.runtime_report.resolve()),
            "evaluator_manifest": str(args.evaluator_manifest.resolve()), "evaluator_manifest_sha256": sha256(args.evaluator_manifest.resolve()),
        },
        "evaluation_contract": {
            "gt_used_only_in_evaluator": True,
            "joint_set": {
                "name": "MVHuman-Body25-common-body12-v1",
                "body25_indices": BODY25_TO_COMMON12,
                "common_smpl24_indices": shared.SMPL24_BODY12,
                "names": shared.BODY12_NAMES,
            },
            "anchor": "one first-shot-only Sim(3) per case/method; no post-cut re-alignment",
            "track_policy": "longest valid persistent track then lowest numeric ID; never GT selected",
            "camera": "calibrated camera-to-world; relative errors measured from the first input frame",
        },
    }
    payload["content_sha256"] = hashlib.sha256(
        json.dumps(jsonable(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "case_id": args.case_id, "methods": len(methods), "errors": errors}, indent=2))


if __name__ == "__main__":
    main()
