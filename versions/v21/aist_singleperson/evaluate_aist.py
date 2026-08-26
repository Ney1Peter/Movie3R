#!/usr/bin/env python3
"""Evaluator-only metrics for one derived AIST++ CS150 Bridge3R cache.

The official AIST++ 3-D keypoints are COCO-17.  The face landmarks cannot be
obtained from the common SMPL-24 cache with an exact shared regressor, so all
human scores below use the explicitly named COCO *body-12* subset (shoulders,
elbows, wrists, hips, knees, ankles).  This is intentional and is recorded in
every result: it prevents an unverified face-joint conversion from entering a
paper score.

The evaluator is the only component that reads AIST labels, calibration, and
ground-truth cut indices.  A single first-shot Sim(3) anchor is fitted for
each method/case and is retained after the view change; no post-cut re-fit is
performed.  If an official label has a non-finite body frame, that frame is
excluded from *all* geometry calculations for every method while prediction
coverage keeps its full 150-frame denominator.  This is a label-availability
rule, not a method-dependent filtering rule.
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
    from .protocol import DEFAULT_DERIVED_ROOT, canonical_json_digest
except ImportError:
    from protocol import DEFAULT_DERIVED_ROOT, canonical_json_digest  # type: ignore


SCHEMA = "Bridge3R-AIST-SinglePerson-CS150-evaluation-v1"
COCO_BODY12 = np.asarray([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], dtype=np.int64)
SMPL24_BODY12 = np.asarray([16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8], dtype=np.int64)
BODY12_NAMES = (
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
)
EXPECTED_FRAMES = 150


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


def summary(values: np.ndarray | list[float], scale: float = 1.0) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)] * float(scale)
    if not len(array):
        return {"count": 0, "mean": None, "median": None, "p90": None, "std": None}
    return {
        "count": int(len(array)), "mean": float(array.mean()), "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)), "std": float(array.std()),
    }


def fit_similarity(target: np.ndarray, prediction: np.ndarray, allow_scale: bool = True) -> tuple[float, np.ndarray, np.ndarray]:
    target = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1, 3)
    if target.shape != prediction.shape or len(target) < 3:
        raise ValueError(f"Similarity fit shape mismatch: {target.shape} vs {prediction.shape}")
    target_mean, prediction_mean = target.mean(axis=0), prediction.mean(axis=0)
    target_centered, prediction_centered = target - target_mean, prediction - prediction_mean
    left, singular, right = np.linalg.svd(prediction_centered.T @ target_centered)
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0:
        right[-1] *= -1
        rotation = right.T @ left.T
    denominator = float(np.sum(prediction_centered ** 2))
    scale = float(singular.sum() / max(denominator, 1e-12)) if allow_scale else 1.0
    translation = target_mean - scale * (rotation @ prediction_mean)
    return scale, rotation, translation


def apply_similarity(points: np.ndarray, fit: tuple[float, np.ndarray, np.ndarray]) -> np.ndarray:
    scale, rotation, translation = fit
    return scale * (np.asarray(points) @ rotation.T) + translation


def procrustes_mpjpe(target: np.ndarray, prediction: np.ndarray) -> float:
    fit = fit_similarity(target, prediction, allow_scale=True)
    return float(np.linalg.norm(apply_similarity(prediction, fit) - target, axis=-1).mean())


def rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first, dtype=np.float64).T @ np.asarray(second, dtype=np.float64)
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def root_from_body12(joints: np.ndarray) -> np.ndarray:
    # COCO body-12 order: left/right hips at indexes 6 and 7.
    return np.asarray(joints, dtype=np.float64)[..., [6, 7], :].mean(axis=-2)


def torso_rotation(joints: np.ndarray) -> np.ndarray | None:
    """Return an orthonormal body frame derived only from body-12 joints."""
    value = np.asarray(joints, dtype=np.float64)
    left_shoulder, right_shoulder = value[0], value[1]
    left_hip, right_hip = value[6], value[7]
    lateral = right_shoulder - left_shoulder
    vertical = (left_shoulder + right_shoulder - left_hip - right_hip) * 0.5
    if np.linalg.norm(lateral) < 1e-8 or np.linalg.norm(vertical) < 1e-8:
        return None
    lateral = lateral / np.linalg.norm(lateral)
    vertical = vertical - lateral * np.dot(vertical, lateral)
    if np.linalg.norm(vertical) < 1e-8:
        return None
    vertical = vertical / np.linalg.norm(vertical)
    forward = np.cross(lateral, vertical)
    if np.linalg.norm(forward) < 1e-8:
        return None
    forward = forward / np.linalg.norm(forward)
    return np.column_stack((lateral, vertical, forward))


def pick_track(arrays: dict[str, np.ndarray]) -> tuple[int | None, dict[str, Any]]:
    valid = np.asarray(arrays["valid"], dtype=bool)
    identifiers = np.asarray(arrays["persistent_ids"], dtype=np.int64)
    if valid.shape != identifiers.shape:
        raise ValueError("valid/persistent-id shape mismatch")
    counts: dict[int, int] = {}
    for value in identifiers[valid]:
        if int(value) >= 0:
            counts[int(value)] = counts.get(int(value), 0) + 1
    if not counts:
        return None, {"policy": "longest_valid_persistent_track_then_smallest_id", "reason": "no_nonnegative_persistent_id", "counts": {}}
    chosen = min(counts, key=lambda item: (-counts[item], item))
    return chosen, {"policy": "longest_valid_persistent_track_then_smallest_id", "chosen_id": chosen, "counts": {str(key): value for key, value in sorted(counts.items())}}


def select_track(arrays: dict[str, np.ndarray], track_id: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frames = len(arrays["valid"])
    joints = np.full((frames, len(COCO_BODY12), 3), np.nan, dtype=np.float64)
    cameras = np.asarray(arrays["cameras_c2w"], dtype=np.float64)
    if cameras.shape != (frames, 4, 4):
        raise ValueError(f"Predicted camera shape is {cameras.shape}, expected ({frames},4,4)")
    selected = np.full(frames, -1, dtype=np.int64)
    if track_id is None:
        return joints, np.zeros(frames, dtype=bool), cameras, selected
    valid = np.asarray(arrays["valid"], dtype=bool)
    ids = np.asarray(arrays["persistent_ids"], dtype=np.int64)
    source = np.asarray(arrays["joints_world"], dtype=np.float64)
    if source.shape[:2] != valid.shape or source.shape[-2:] != (24, 3):
        raise ValueError(f"Predicted joint shape is incompatible: {source.shape}")
    for frame in range(frames):
        choices = np.flatnonzero(valid[frame] & (ids[frame] == int(track_id)))
        if len(choices):
            index = int(choices[0])
            value = source[frame, index, SMPL24_BODY12]
            if np.isfinite(value).all():
                joints[frame] = value
                selected[frame] = index
    return joints, np.isfinite(joints).all(axis=(1, 2)), cameras, selected


def detector_metrics(report: dict[str, Any], cut_indices: np.ndarray) -> dict[str, Any]:
    detector = report.get("runtime", {}).get("causal_gru_detector", {})
    labels = np.asarray(detector.get("labels", []), dtype=np.int64)
    if labels.shape != (EXPECTED_FRAMES,):
        return {"available": False, "reason": f"labels_shape={labels.shape}"}
    target = np.zeros(EXPECTED_FRAMES, dtype=np.int64)
    target[np.asarray(cut_indices, dtype=np.int64) + 1] = 1
    tp = int(((labels == 1) & (target == 1)).sum())
    fp = int(((labels == 1) & (target == 0)).sum())
    fn = int(((labels == 0) & (target == 1)).sum())
    positives = np.flatnonzero(labels)
    offsets = [int(min(abs(int(value) - (int(cut) + 1)) for value in positives)) for cut in cut_indices] if len(positives) else []
    return {
        "available": True, "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
        "f1": 2.0 * tp / max(2 * tp + fp + fn, 1),
        "first_positive_index": detector.get("first_positive_index"),
        "nearest_gt_cut_offset_frames": None if not offsets else float(np.mean(offsets)),
        "latency_seconds": detector.get("seconds"),
    }


def evaluate_method(method: str, arrays: dict[str, np.ndarray], label: dict[str, np.ndarray], report: dict[str, Any]) -> dict[str, Any]:
    track_id, track = pick_track(arrays)
    pred, pred_valid, pred_cameras, selected_indices = select_track(arrays, track_id)
    gt_full = np.asarray(label["world_keypoints_m"], dtype=np.float64)
    gt = gt_full[:, COCO_BODY12]
    gt_cameras = np.asarray(label["camera_camera_to_world_m"], dtype=np.float64)
    cuts = np.asarray(label["cut_indices_evaluator_only"], dtype=np.int64)
    if gt.shape != (EXPECTED_FRAMES, len(COCO_BODY12), 3) or gt_cameras.shape != (EXPECTED_FRAMES, 4, 4):
        raise ValueError("AIST label shape differs from the frozen CS150 contract")
    if cuts.tolist() != [74]:
        raise ValueError(f"CS150 requires cut index [74], received {cuts.tolist()}")
    cut = int(cuts[0]); first_post = cut + 1
    # AIST labels occasionally mark an unavailable body frame with NaNs.  A
    # full body frame (rather than individual joints) is the smallest common
    # unit used by PA, root and orientation summaries, so one immutable
    # evaluator-only mask is shared by every metric and method.  Prediction
    # coverage intentionally remains independent of this annotation mask.
    gt_valid = np.isfinite(gt).all(axis=(1, 2))
    metric_valid = pred_valid & gt_valid
    anchor_mask = metric_valid.copy(); anchor_mask[first_post:] = False
    status = "ok"
    if int(anchor_mask.sum()) < 3:
        status = "evaluator_unavailable_insufficient_first_shot_track_coverage"
        return {
            "method": method, "status": status, "track": track,
            "coverage": {
                "valid_frames": int(pred_valid.sum()), "total_frames": EXPECTED_FRAMES,
                "valid_frame_coverage": float(pred_valid.mean()), "completion": 1.0,
                "evaluator_gt_valid_frames": int(gt_valid.sum()),
                "geometry_evaluable_frames": int(metric_valid.sum()),
            },
            "metrics": {}, "selected_detection_index_per_frame": selected_indices,
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
    seam_root = None
    seam_orientation = None
    if metric_valid[cut] and metric_valid[first_post]:
        seam_root = float(np.linalg.norm((aligned_root[first_post] - aligned_root[cut]) - (gt_root[first_post] - gt_root[cut])))
        before_pred, after_pred = torso_rotation(aligned[cut]), torso_rotation(aligned[first_post])
        before_gt, after_gt = torso_rotation(gt[cut]), torso_rotation(gt[first_post])
        if all(value is not None for value in (before_pred, after_pred, before_gt, after_gt)):
            pred_relative = before_pred.T @ after_pred  # type: ignore[union-attr]
            gt_relative = before_gt.T @ after_gt  # type: ignore[union-attr]
            seam_orientation = rotation_error_deg(pred_relative, gt_relative)
    # The camera relative rotation is gauge-invariant.  We report it after
    # the first true cut, but it is never supplied to the runner.
    anchor_camera = pred_cameras[0]
    camera_rotation = []
    camera_translation = []
    for frame in range(first_post, EXPECTED_FRAMES):
        if not (np.isfinite(anchor_camera).all() and np.isfinite(pred_cameras[frame]).all() and np.isfinite(gt_cameras[0]).all() and np.isfinite(gt_cameras[frame]).all()):
            continue
        predicted_relative = np.linalg.inv(anchor_camera) @ pred_cameras[frame]
        gt_relative = np.linalg.inv(gt_cameras[0]) @ gt_cameras[frame]
        camera_rotation.append(rotation_error_deg(predicted_relative[:3, :3], gt_relative[:3, :3]))
        camera_translation.append(float(np.linalg.norm(predicted_relative[:3, 3] - gt_relative[:3, 3])))
    return {
        "method": method, "status": status, "track": track,
        "coverage": {
            "valid_frames": int(pred_valid.sum()), "total_frames": EXPECTED_FRAMES,
            "valid_frame_coverage": float(pred_valid.mean()), "completion": 1.0,
            "anchor_valid_frames": int(anchor_mask.sum()), "post_valid_frames": int(metric_valid[first_post:].sum()),
            "evaluator_gt_valid_frames": int(gt_valid.sum()), "geometry_evaluable_frames": int(metric_valid.sum()),
        },
        "metrics": {
            "pa_mpjpe_body12_mm": summary(pa, 1000.0),
            "first_shot_anchor_mpjpe_body12_mm": summary(anchor_joint, 1000.0),
            "first_shot_anchor_root_error_mm": summary(root_error, 1000.0),
            "first_shot_anchor_orientation_proxy_deg": summary(orientation),
            "seam_root_excess_mm": summary([] if seam_root is None else [seam_root], 1000.0),
            "seam_orientation_excess_deg": summary([] if seam_orientation is None else [seam_orientation]),
            "post_camera_relative_rotation_deg": summary(camera_rotation),
            "post_camera_relative_translation_m": summary(camera_translation),
        },
        "alignment": {
            "definition": "one Sim(3) fitted on valid first-shot body-12 points only; reused after the RGB view cut",
            "scale": fit[0], "rotation": fit[1], "translation": fit[2],
        },
        "selected_detection_index_per_frame": selected_indices,
    }


def load_arrays(cache: np.lib.npyio.NpzFile, method: str) -> dict[str, np.ndarray]:
    prefix = method + "__"
    required = ("cameras_c2w", "joints_world", "persistent_ids", "valid")
    missing = [prefix + key for key in required if prefix + key not in cache.files]
    if missing:
        raise KeyError(f"{method} cache lacks {missing}")
    return {key: np.asarray(cache[prefix + key]) for key in required}


def self_test() -> None:
    rng = np.random.default_rng(20260826)
    source = rng.normal(size=(8, 12, 3))
    angle = 0.3
    rotation = np.asarray([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    target = 1.4 * (source @ rotation.T) + np.asarray([0.3, -0.4, 1.1])
    fit = fit_similarity(target, source)
    np.testing.assert_allclose(apply_similarity(source, fit), target, atol=1e-10)
    np.testing.assert_allclose([procrustes_mpjpe(target[index], source[index]) for index in range(8)], 0.0, atol=1e-10)
    assert torso_rotation(source[0]) is not None
    print("AIST evaluator self-test passed")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test(); return
    if any(value is None for value in (args.cache, args.runtime_report, args.label, args.output)):
        raise ValueError("--cache, --runtime-report, --label and --output are required")
    report = json.loads(args.runtime_report.read_text(encoding="utf-8"))
    record = report.get("record", {})
    if record.get("protocol") != "CS150" or int(record.get("num_frames", -1)) != EXPECTED_FRAMES:
        raise ValueError("Runtime report is not an AIST++ CS150 cache")
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
            except Exception as error:  # Each method remains in denominator; preserve its failure.
                errors[str(method)] = f"{type(error).__name__}: {error}"
    payload = {
        "schema_version": SCHEMA, "protocol": "CS150", "case_id": record.get("case_id"),
        "methods": methods, "errors": errors,
        "detector": detector_metrics(report, label["cut_indices_evaluator_only"]),
        "inputs": {"cache": str(args.cache.resolve()), "cache_sha256": sha256(args.cache.resolve()), "runtime_report": str(args.runtime_report.resolve()), "runtime_report_sha256": sha256(args.runtime_report.resolve()), "label": str(args.label.resolve()), "label_sha256": sha256(args.label.resolve())},
        "evaluation_contract": {
            "gt_used_only_in_evaluator": True,
            "joint_set": {"name": "AIST-COCO-body12-v1", "coco17_indices": COCO_BODY12, "common_smpl24_indices": SMPL24_BODY12, "names": BODY12_NAMES, "excluded_coco17_indices": [0, 1, 2, 3, 4], "reason": "face landmarks have no verified common-SMPL-24 equivalent"},
            "anchor": "one first-shot-only Sim(3) per case/method; no post-cut re-alignment",
            "label_availability": "A non-finite official body frame is excluded uniformly from geometry metrics only; prediction coverage retains the 150-frame runtime denominator.",
            "track_policy": "longest valid persistent track then lowest numeric ID; never GT selected",
        },
    }
    # Method diagnostics include compact NumPy arrays (the per-frame selected
    # detection slot), so canonicalise their JSON representation rather than
    # passing raw arrays to the shared digest helper.
    payload["content_sha256"] = canonical_json_digest(jsonable(payload))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "case_id": record.get("case_id"), "methods": len(methods), "errors": errors}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
