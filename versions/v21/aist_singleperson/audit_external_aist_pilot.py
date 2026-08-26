#!/usr/bin/env python3
"""Audit a completed external-method AIST CS150 pilot before opening test.

The audit does not select examples or tune a method.  It verifies that every
pre-registered pilot row has an evaluator report, uses the original 150-frame
timeline, retains its own persistent track across both sides of the view cut,
and declares any unavailable camera quantity explicitly.  Its result is an
auditable test-gate record, not a performance score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .protocol import atomic_json, canonical_json_digest, sha256_file
except ImportError:
    from protocol import atomic_json, canonical_json_digest, sha256_file  # type: ignore


SCHEMA = "Bridge3R-AIST-external-pilot-audit-v1"
FRAMES, CUT = 150, 74


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--camera-contract", choices=("required", "unavailable"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def safe_name(case_id: str) -> str:
    return case_id.replace("/", "_")


def finite_summary(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    count, mean = value.get("count"), value.get("mean")
    return isinstance(count, int) and count > 0 and isinstance(mean, (int, float)) and math.isfinite(float(mean))


def rotation_audit(cameras: np.ndarray) -> dict[str, Any]:
    finite = bool(np.isfinite(cameras).all())
    if not finite:
        return {"finite": False, "orthogonality_max": None, "determinant_error_max": None}
    rotations = cameras[:, :3, :3]
    identity = np.eye(3)
    orthogonality = float(np.max(np.abs(rotations @ np.swapaxes(rotations, -1, -2) - identity)))
    determinant = float(np.max(np.abs(np.linalg.det(rotations) - 1.0)))
    return {"finite": True, "orthogonality_max": orthogonality, "determinant_error_max": determinant}


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError("pilot audit refuses to overwrite its frozen output")
    runtime_rows = [json.loads(line) for line in args.runtime_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    case_ids = [str(row.get("case_id")) for row in runtime_rows]
    if not case_ids or len(case_ids) != len(set(case_ids)):
        raise ValueError("runtime manifest has no unique pilot case IDs")
    audits, failures, warnings = [], [], []
    for case_id in case_ids:
        name = safe_name(case_id)
        metric_path, prediction_path = args.metrics_dir / f"{name}.json", args.predictions_dir / f"{name}.npz"
        adapter_path = args.predictions_dir / f"{name}.adapter.json"
        item: dict[str, Any] = {"case_id": case_id, "files": {"metric": str(metric_path), "prediction": str(prediction_path), "adapter": str(adapter_path)}}
        required = [path for path in (metric_path, prediction_path, adapter_path) if not path.is_file()]
        if required:
            item["status"] = "missing_artifact"; item["missing"] = [str(path) for path in required]
            audits.append(item); failures.append({"case_id": case_id, "reason": item["status"]}); continue
        metric = json.loads(metric_path.read_text(encoding="utf-8")); method = metric.get("methods", {}).get(args.method)
        if not isinstance(method, dict) or method.get("status") != "ok" or args.method in metric.get("errors", {}):
            item["status"] = "evaluator_error"; item["evaluator"] = {"method": method, "errors": metric.get("errors", {})}
            audits.append(item); failures.append({"case_id": case_id, "reason": item["status"]}); continue
        with np.load(prediction_path, allow_pickle=False) as cache:
            prefix = args.method + "__"
            required_keys = ("cameras_c2w", "joints_world", "persistent_ids", "valid")
            absent = [prefix + key for key in required_keys if prefix + key not in cache.files]
            if absent:
                item["status"] = "cache_schema_error"; item["missing_cache_keys"] = absent
                audits.append(item); failures.append({"case_id": case_id, "reason": item["status"]}); continue
            cameras, joints = np.asarray(cache[prefix + "cameras_c2w"]), np.asarray(cache[prefix + "joints_world"])
            ids, valid = np.asarray(cache[prefix + "persistent_ids"]), np.asarray(cache[prefix + "valid"], dtype=bool)
        track = method.get("track", {}); chosen = track.get("chosen_id")
        shape_ok = cameras.shape == (FRAMES, 4, 4) and joints.ndim == 4 and joints.shape[0] == FRAMES and joints.shape[2:] == (24, 3) and ids.shape == valid.shape == joints.shape[:2]
        if not shape_ok or chosen is None:
            item["status"] = "cache_or_track_shape_error"; item["shapes"] = {"cameras": list(cameras.shape), "joints": list(joints.shape), "ids": list(ids.shape), "valid": list(valid.shape), "chosen_id": chosen}
            audits.append(item); failures.append({"case_id": case_id, "reason": item["status"]}); continue
        selected = np.any(valid & (ids == int(chosen)), axis=1)
        pre, post = int(selected[: CUT + 1].sum()), int(selected[CUT + 1 :].sum())
        human_finite = bool(np.isfinite(joints[selected]).all()) if selected.any() else False
        cameras_audit = rotation_audit(cameras)
        metric_values = method.get("metrics", {})
        local_ok = finite_summary(metric_values.get("pa_mpjpe_body12_mm"))
        global_human_ok = all(finite_summary(metric_values.get(name)) for name in ("first_shot_anchor_mpjpe_body12_mm", "seam_root_excess_mm", "seam_orientation_excess_deg"))
        if args.camera_contract == "required":
            camera_ok = cameras_audit["finite"] and cameras_audit["orthogonality_max"] < 2e-3 and cameras_audit["determinant_error_max"] < 2e-3 and finite_summary(metric_values.get("post_camera_relative_rotation_deg"))
        else:
            camera_ok = not cameras_audit["finite"] and metric_values.get("post_camera_relative_rotation_deg", {}).get("mean") is None and metric_values.get("post_camera_relative_translation_m", {}).get("mean") is None
        adapter = json.loads(adapter_path.read_text(encoding="utf-8")); tracker = adapter.get("tracker_audit", {})
        tracker_ok = True
        if args.method == "gvhmr_official":
            tracker_ok = bool(tracker.get("available")) and int(tracker.get("raw_selected_pre_cut_frame_count", 0)) > 0 and int(tracker.get("raw_selected_post_cut_frame_count", 0)) > 0
        item.update({
            "status": "ok" if all((pre >= 3, post >= 3, human_finite, local_ok, global_human_ok, camera_ok, tracker_ok)) else "audit_failed",
            "track": {"chosen_id": int(chosen), "pre_valid_frames": pre, "post_valid_frames": post, "total_valid_frames": int(selected.sum())},
            "human_finite": human_finite, "metric_availability": {"local": local_ok, "global_human": global_human_ok, "camera": camera_ok},
            "camera": cameras_audit, "tracker": tracker,
            "metric_sha256": sha256_file(metric_path), "prediction_sha256": sha256_file(prediction_path), "adapter_sha256": sha256_file(adapter_path),
        })
        if item["status"] != "ok":
            failures.append({"case_id": case_id, "reason": "audit_predicate_failed"})
        if args.method == "gvhmr_official":
            warnings.append({"case_id": case_id, "message": "GVHMR camera trajectory is deliberately unavailable; this method cannot populate relative-camera columns."})
        audits.append(item)
    payload = {
        "schema_version": SCHEMA, "method": args.method, "camera_contract": args.camera_contract,
        "runtime_manifest": str(args.runtime_manifest.resolve()), "runtime_manifest_sha256": sha256_file(args.runtime_manifest.resolve()),
        "case_count": len(case_ids), "audits": audits, "failures": failures, "warnings": warnings,
        "test_gate": {"approved": not failures, "reason": "all pre-registered pilot rows passed output, timeline, cross-cut track, metric-availability, and declared-camera-contract checks" if not failures else "one or more pilot rows failed; do not open test without documented resolution"},
    }
    payload["content_sha256"] = canonical_json_digest(payload)
    atomic_json(args.output, payload)
    print(json.dumps({"output": str(args.output), "method": args.method, "cases": len(case_ids), "approved": not failures, "failures": len(failures)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
