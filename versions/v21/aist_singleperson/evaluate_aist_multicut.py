#!/usr/bin/env python3
"""Evaluator-only metrics for one AIST++ MC150-3/MC150-4 cache.

This is deliberately separate from :mod:`evaluate_aist`, whose input contract
is exactly one CS150 cut.  The MC evaluator reads labels only after an
RGB-only multi-cut cache has been written.  It evaluates every official
boundary independently and never feeds a cut, calibration, camera ID, or GT
track back to the runner.
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
    from .evaluate_aist import (
        BODY12_NAMES,
        COCO_BODY12,
        EXPECTED_FRAMES,
        SMPL24_BODY12,
        apply_similarity,
        fit_similarity,
        jsonable,
        pick_track,
        procrustes_mpjpe,
        root_from_body12,
        rotation_error_deg,
        select_track,
        summary,
        torso_rotation,
    )
    from .protocol import DEFAULT_DERIVED_ROOT, PROTOCOLS, canonical_json_digest
except ImportError:
    from evaluate_aist import (  # type: ignore
        BODY12_NAMES,
        COCO_BODY12,
        EXPECTED_FRAMES,
        SMPL24_BODY12,
        apply_similarity,
        fit_similarity,
        jsonable,
        pick_track,
        procrustes_mpjpe,
        root_from_body12,
        rotation_error_deg,
        select_track,
        summary,
        torso_rotation,
    )
    from protocol import DEFAULT_DERIVED_ROOT, PROTOCOLS, canonical_json_digest  # type: ignore


SCHEMA = "Bridge3R-AIST-SinglePerson-MC150-evaluation-v1"
ALLOWED_PROTOCOLS = {"MC150-3", "MC150-4"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--runtime-report", type=Path)
    parser.add_argument("--label", type=Path)
    parser.add_argument("--derived-root", type=Path, default=DEFAULT_DERIVED_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def detector_metrics(report: dict[str, Any], cuts: np.ndarray) -> dict[str, Any]:
    detector = report.get("runtime", {}).get("causal_gru_detector", {})
    labels = np.asarray(detector.get("labels", []), dtype=np.int64)
    if labels.shape != (EXPECTED_FRAMES,):
        return {"available": False, "reason": f"labels_shape={labels.shape}"}
    target = np.zeros(EXPECTED_FRAMES, dtype=np.int64)
    target[np.asarray(cuts, dtype=np.int64) + 1] = 1
    positives = np.flatnonzero(labels == 1)
    tp = int(((labels == 1) & (target == 1)).sum())
    fp = int(((labels == 1) & (target == 0)).sum())
    fn = int(((labels == 0) & (target == 1)).sum())
    offsets = [int(np.min(np.abs(positives - (int(cut) + 1)))) for cut in cuts] if len(positives) else []
    return {
        "available": True,
        "target_count": int(target.sum()),
        "positive_count": int(len(positives)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "f1": 2.0 * tp / max(2 * tp + fp + fn, 1),
        "nearest_gt_cut_offset_frames": summary(offsets),
        "policy": "exact first-frame-after-cut matches; all detector positives are retained",
    }


def load_arrays(cache: np.lib.npyio.NpzFile, method: str) -> dict[str, np.ndarray]:
    prefix = method + "__"
    required = ("cameras_c2w", "joints_world", "persistent_ids", "valid")
    missing = [prefix + key for key in required if prefix + key not in cache.files]
    if missing:
        raise KeyError(f"{method} cache lacks {missing}")
    return {key: np.asarray(cache[prefix + key]) for key in required}


def evaluate_method(
    method: str, arrays: dict[str, np.ndarray], label: dict[str, np.ndarray], report: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate one track under a first-shot anchor retained across all cuts."""
    protocol = str(report["record"]["protocol"])
    cuts = np.asarray(label["cut_indices_evaluator_only"], dtype=np.int64)
    expected = np.asarray(PROTOCOLS[protocol]["cut_indices"], dtype=np.int64)
    if not np.array_equal(cuts, expected):
        raise ValueError(f"{protocol} cut indices differ from the frozen contract: {cuts.tolist()}")
    track_id, track = pick_track(arrays)
    pred, pred_valid, pred_cameras, selected_indices = select_track(arrays, track_id)
    gt_full = np.asarray(label["world_keypoints_m"], dtype=np.float64)
    gt = gt_full[:, COCO_BODY12]
    gt_cameras = np.asarray(label["camera_camera_to_world_m"], dtype=np.float64)
    if gt.shape != (EXPECTED_FRAMES, len(COCO_BODY12), 3):
        raise ValueError(f"Unexpected AIST body shape: {gt.shape}")
    if gt_cameras.shape != (EXPECTED_FRAMES, 4, 4):
        raise ValueError(f"Unexpected AIST camera shape: {gt_cameras.shape}")
    gt_valid = np.isfinite(gt).all(axis=(1, 2))
    metric_valid = pred_valid & gt_valid
    first_post = int(cuts[0]) + 1
    anchor_mask = metric_valid.copy()
    anchor_mask[first_post:] = False
    coverage = {
        "valid_frames": int(pred_valid.sum()),
        "total_frames": EXPECTED_FRAMES,
        "valid_frame_coverage": float(pred_valid.mean()),
        "completion": 1.0,
        "anchor_valid_frames": int(anchor_mask.sum()),
        "post_first_cut_valid_frames": int(metric_valid[first_post:].sum()),
        "evaluator_gt_valid_frames": int(gt_valid.sum()),
        "geometry_evaluable_frames": int(metric_valid.sum()),
    }
    if int(anchor_mask.sum()) < 3:
        return {
            "method": method,
            "status": "evaluator_unavailable_insufficient_first_shot_track_coverage",
            "track": track,
            "coverage": coverage,
            "metrics": {},
            "selected_detection_index_per_frame": selected_indices,
        }
    fit = fit_similarity(gt[anchor_mask], pred[anchor_mask], allow_scale=True)
    aligned = apply_similarity(pred, fit)
    aligned_root, gt_root = root_from_body12(aligned), root_from_body12(gt)
    pa = [procrustes_mpjpe(gt[index], pred[index]) for index in np.flatnonzero(metric_valid)]
    anchor_joint = np.linalg.norm(aligned[metric_valid] - gt[metric_valid], axis=-1).mean(axis=-1)
    root_error = np.linalg.norm(aligned_root[metric_valid] - gt_root[metric_valid], axis=-1)
    orientation = []
    for frame in np.flatnonzero(metric_valid):
        source, target = torso_rotation(aligned[frame]), torso_rotation(gt[frame])
        if source is not None and target is not None:
            orientation.append(rotation_error_deg(source, target))
    seam_root, seam_orientation, seam_camera_rotation, seam_camera_translation = [], [], [], []
    boundary_metrics = []
    for boundary_order, cut in enumerate(cuts, 1):
        post = int(cut) + 1
        boundary_row: dict[str, Any] = {
            "boundary_order": boundary_order,
            "cut_index": int(cut),
            "seam_root_excess_mm": None,
            "seam_orientation_excess_deg": None,
            "camera_relative_rotation_deg": None,
            "camera_relative_translation_m": None,
        }
        if metric_valid[int(cut)] and metric_valid[post]:
            root_value = float(np.linalg.norm(
                (aligned_root[post] - aligned_root[int(cut)]) - (gt_root[post] - gt_root[int(cut)])
            ))
            seam_root.append(root_value)
            boundary_row["seam_root_excess_mm"] = 1000.0 * root_value
            before_pred, after_pred = torso_rotation(aligned[int(cut)]), torso_rotation(aligned[post])
            before_gt, after_gt = torso_rotation(gt[int(cut)]), torso_rotation(gt[post])
            if all(value is not None for value in (before_pred, after_pred, before_gt, after_gt)):
                orientation_value = rotation_error_deg(
                    before_pred.T @ after_pred,  # type: ignore[union-attr]
                    before_gt.T @ after_gt,  # type: ignore[union-attr]
                )
                seam_orientation.append(orientation_value)
                boundary_row["seam_orientation_excess_deg"] = orientation_value
        if all(np.isfinite(value).all() for value in (pred_cameras[int(cut)], pred_cameras[post], gt_cameras[int(cut)], gt_cameras[post])):
            predicted_relative = np.linalg.inv(pred_cameras[int(cut)]) @ pred_cameras[post]
            gt_relative = np.linalg.inv(gt_cameras[int(cut)]) @ gt_cameras[post]
            rotation_value = rotation_error_deg(predicted_relative[:3, :3], gt_relative[:3, :3])
            translation_value = float(np.linalg.norm(predicted_relative[:3, 3] - gt_relative[:3, 3]))
            seam_camera_rotation.append(rotation_value)
            seam_camera_translation.append(translation_value)
            boundary_row["camera_relative_rotation_deg"] = rotation_value
            boundary_row["camera_relative_translation_m"] = translation_value
        boundary_metrics.append(boundary_row)
    anchor_camera = pred_cameras[0]
    post_camera_rotation, post_camera_translation = [], []
    for frame in range(first_post, EXPECTED_FRAMES):
        if not all(np.isfinite(value).all() for value in (anchor_camera, pred_cameras[frame], gt_cameras[0], gt_cameras[frame])):
            continue
        predicted_relative = np.linalg.inv(anchor_camera) @ pred_cameras[frame]
        gt_relative = np.linalg.inv(gt_cameras[0]) @ gt_cameras[frame]
        post_camera_rotation.append(rotation_error_deg(predicted_relative[:3, :3], gt_relative[:3, :3]))
        post_camera_translation.append(float(np.linalg.norm(predicted_relative[:3, 3] - gt_relative[:3, 3])))
    return {
        "method": method,
        "status": "ok",
        "track": track,
        "coverage": coverage,
        "metrics": {
            "pa_mpjpe_body12_mm": summary(pa, 1000.0),
            "first_shot_anchor_mpjpe_body12_mm": summary(anchor_joint, 1000.0),
            "first_shot_anchor_root_error_mm": summary(root_error, 1000.0),
            "first_shot_anchor_orientation_proxy_deg": summary(orientation),
            "mean_boundary_seam_root_excess_mm": summary(seam_root, 1000.0),
            "mean_boundary_seam_orientation_excess_deg": summary(seam_orientation),
            "mean_boundary_camera_relative_rotation_deg": summary(seam_camera_rotation),
            "mean_boundary_camera_relative_translation_m": summary(seam_camera_translation),
            "post_first_cut_camera_relative_rotation_deg": summary(post_camera_rotation),
            "post_first_cut_camera_relative_translation_m": summary(post_camera_translation),
        },
        "alignment": {
            "definition": "one Sim(3) fitted on valid first-shot body-12 points only; reused without re-fitting after every RGB cut",
            "scale": fit[0], "rotation": fit[1], "translation": fit[2],
        },
        "boundary_metrics": boundary_metrics,
        "selected_detection_index_per_frame": selected_indices,
    }


def self_test() -> None:
    # Purely numerical test: all imported geometry primitives remain usable.
    rng = np.random.default_rng(20260827)
    points = rng.normal(size=(8, 12, 3))
    fit = fit_similarity(points, points)
    np.testing.assert_allclose(apply_similarity(points, fit), points, atol=1e-10)
    report = {"runtime": {"causal_gru_detector": {"labels": [0] * 49 + [1] + [0] * 50 + [1] + [0] * 49}}}
    score = detector_metrics(report, np.asarray([48, 99], dtype=np.int64))
    assert score["tp"] == 2 and score["f1"] == 1.0
    print("AIST multi-cut evaluator self-test passed")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if any(value is None for value in (args.cache, args.runtime_report, args.label, args.output)):
        raise ValueError("--cache, --runtime-report, --label and --output are required")
    report = json.loads(args.runtime_report.read_text(encoding="utf-8"))
    record = report.get("record", {})
    protocol = str(record.get("protocol"))
    if protocol not in ALLOWED_PROTOCOLS or int(record.get("num_frames", -1)) != EXPECTED_FRAMES:
        raise ValueError("Runtime report is not an AIST++ MC150-3/MC150-4 cache")
    with np.load(args.label, allow_pickle=False) as archive:
        needed = {"world_keypoints_m", "camera_camera_to_world_m", "cut_indices_evaluator_only"}
        missing = needed.difference(archive.files)
        if missing:
            raise KeyError(f"AIST label misses {sorted(missing)}")
        label = {key: np.asarray(archive[key]) for key in needed}
    methods, errors = {}, {}
    with np.load(args.cache, allow_pickle=False) as cache:
        for method in report.get("methods", []):
            try:
                methods[str(method)] = evaluate_method(str(method), load_arrays(cache, str(method)), label, report)
            except Exception as error:  # Preserve every method failure in the frozen denominator.
                errors[str(method)] = f"{type(error).__name__}: {error}"
    payload = {
        "schema_version": SCHEMA,
        "protocol": protocol,
        "case_id": record.get("case_id"),
        "methods": methods,
        "errors": errors,
        "detector": detector_metrics(report, label["cut_indices_evaluator_only"]),
        "inputs": {
            "cache": str(args.cache.resolve()), "cache_sha256": sha256(args.cache.resolve()),
            "runtime_report": str(args.runtime_report.resolve()), "runtime_report_sha256": sha256(args.runtime_report.resolve()),
            "label": str(args.label.resolve()), "label_sha256": sha256(args.label.resolve()),
        },
        "evaluation_contract": {
            "gt_used_only_in_evaluator": True,
            "joint_set": {"name": "AIST-COCO-body12-v1", "coco17_indices": COCO_BODY12, "common_smpl24_indices": SMPL24_BODY12, "names": BODY12_NAMES, "excluded_coco17_indices": [0, 1, 2, 3, 4]},
            "anchor": "one first-shot-only Sim(3) per case/method; no later-shot re-alignment",
            "boundaries": "every frozen evaluator-only boundary contributes equally within a case to mean-boundary metrics",
            "label_availability": "A non-finite official body frame is excluded uniformly from geometry metrics only; prediction coverage keeps the 150-frame runtime denominator.",
            "track_policy": "longest valid persistent track then lowest numeric ID; never GT selected",
        },
    }
    payload["content_sha256"] = canonical_json_digest(jsonable(payload))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "case_id": record.get("case_id"), "protocol": protocol, "methods": len(methods), "errors": errors}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
