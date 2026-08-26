#!/usr/bin/env python3
"""Evaluator-only metrics for a frozen Bridge3R Harmony4D multi-cut cache.

Runtime receives only RGB paths and the fixed multi-cut manifest. Calibration,
SMPL annotations, visibility, and persistent ground-truth identities are read
only here, after the cache and runtime ledger exist. The implementation reuses
the established Harmony4D single-cut metric code, evaluates body/camera
quantities over all 150 frames, and retains a separate seam record for each
of the two fixed boundaries.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from versions.v15.harmony4d.dataset import (  # noqa: E402
    load_exo_calibrations,
    load_gt_people,
    projected_visibility,
)
from versions.v15.harmony4d.evaluate_harmony import (  # noqa: E402
    MIN_VISIBLE_VERTEX_FRACTION,
    CommonTopology,
    evaluate_method,
    jsonable,
    method_arrays,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--runtime-report", type=Path, required=True)
    parser.add_argument("--extracted-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_gt(record: dict[str, Any], extracted_root: Path, topology: CommonTopology) -> tuple[dict[str, np.ndarray], list[str]]:
    """Load GT only after inference for the three-shot record schema."""

    cameras = [str(value) for value in record["shot_cameras"] for _ in range(len(record["shot_frame_numbers"][0]))]
    frames = [int(frame) for shot in record["shot_frame_numbers"] for frame in shot]
    if len(cameras) != len(frames) or not frames:
        raise ValueError("inconsistent multi-cut camera/frame record")
    sequence_root = extracted_root.resolve() / str(record["capture_relative"])
    calibrations = load_exo_calibrations(sequence_root)
    first = load_gt_people(sequence_root, frames[0])
    identities = sorted(first)
    vertices, joints, visibility = [], [], []
    for frame, camera_name in zip(frames, cameras):
        people = load_gt_people(sequence_root, frame)
        if sorted(people) != identities:
            raise ValueError(f"GT identity set changes at frame {frame}")
        frame_vertices = np.stack([people[identity]["vertices"] for identity in identities])
        vertices.append(frame_vertices)
        joints.append(topology.joints_from_smpl(frame_vertices))
        visibility.append([
            projected_visibility(people[identity]["vertices"], calibrations[camera_name])["visible_vertex_fraction"]
            for identity in identities
        ])
    fractions = np.asarray(visibility, dtype=np.float64)
    return {
        "cameras_c2w": np.stack([calibrations[name].camera_to_world for name in cameras]),
        "vertices_world": np.stack(vertices),
        "joints_world": np.stack(joints),
        "frames": np.asarray(frames),
        "visible_fraction": fractions,
        "visible": fractions >= MIN_VISIBLE_VERTEX_FRACTION,
    }, identities


def detector_metrics(runtime: dict[str, Any], boundaries: list[int], key: str) -> dict[str, Any]:
    detector = runtime["runtime"][key]
    labels = np.asarray(detector["labels"], dtype=np.int64)
    target = np.zeros_like(labels)
    target[np.asarray(boundaries, dtype=np.int64)] = 1
    tp = int(((labels == 1) & (target == 1)).sum())
    fp = int(((labels == 1) & (target == 0)).sum())
    fn = int(((labels == 0) & (target == 1)).sum())
    probabilities = np.zeros(len(labels), dtype=np.float64)
    valid_probability = np.zeros(len(labels), dtype=bool)
    for row in detector.get("rows", []):
        index = int(row["pair_idx"])
        probabilities[index] = float(row["prob"])
        valid_probability[index] = True
    brier = (
        float(np.mean((probabilities[valid_probability] - target[valid_probability]) ** 2))
        if valid_probability.any() else None
    )
    return {
        "kind": key,
        "boundaries": boundaries,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "false_positive_rate_per_noncut_pair": fp / max(len(labels) - len(boundaries), 1),
        "boundary_detected": {str(boundary): bool(labels[boundary]) for boundary in boundaries},
        "first_positive_index": detector.get("first_positive_index"),
        "brier": brier,
        "latency_seconds": detector.get("seconds"),
    }


def main() -> None:
    args = parse_args()
    runtime = json.loads(args.runtime_report.read_text(encoding="utf-8"))
    record = runtime["record"]
    boundaries = [int(value) for value in record["boundaries"]]
    if boundaries != sorted(boundaries) or not boundaries or boundaries[0] <= 0:
        raise ValueError(f"invalid multi-cut boundaries: {boundaries}")
    topology = CommonTopology.load()
    gt, identities = load_gt(record, args.extracted_root, topology)
    if len(gt["frames"]) != int(record["clip_length"]):
        raise ValueError("GT frame count differs from frozen manifest")

    methods = {}
    with np.load(args.cache, allow_pickle=False) as cache:
        for method in runtime["methods"]:
            arrays = method_arrays(cache, method)
            aggregate = evaluate_method(
                method, arrays, gt, identities, boundaries[0], float(record["fps"])
            )
            seams = {}
            for boundary in boundaries:
                boundary_result = evaluate_method(
                    method, arrays, gt, identities, boundary, float(record["fps"])
                )
                seams[str(boundary)] = {
                    "cut_seam": boundary_result["cut_seam"],
                    "boundary_camera_rpe_translation_m": boundary_result["camera"]["boundary_rpe_translation_m"],
                    "boundary_camera_rpe_rotation_deg": boundary_result["camera"]["boundary_rpe_rotation_deg"],
                }
            aggregate["cut_seams"] = seams
            methods[method] = aggregate

    report = {
        "schema_version": "Bridge3R-Harmony4D-multicut-evaluation-v1",
        "protocol": record["protocol"],
        "case_id": record["case_id"],
        "record": record,
        "identities": identities,
        "methods": methods,
        "detectors": {
            "causal_gru": detector_metrics(runtime, boundaries, "causal_gru_detector"),
            "static_logistic": detector_metrics(runtime, boundaries, "static_logistic_detector"),
        },
        "evaluation_contract": {
            "gt_used_only_in_evaluator": True,
            "common_topology": topology.metadata(),
            "matching": "per-frame Hungarian in camera coordinates; GT identity never enters runtime",
            "aggregation_scope": "all 150 frames, with each frozen cut seam reported separately",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    concise = {
        method: {
            "W": value["multi_thumbs_named_provisional"]["w_mpjpe_mm"]["mean"],
            "WA": value["multi_thumbs_named_provisional"]["wa_mpjpe_mm"]["mean"],
            "ATE": value["multi_thumbs_named_provisional"]["ate_sim3_m"]["mean"],
            "IDF1": value["identity"]["idf1"],
            "coverage": value["coverage"]["coverage"],
        }
        for method, value in methods.items()
    }
    print(json.dumps({"output": str(args.output), "case_id": record["case_id"], "metrics": concise}, indent=2))


if __name__ == "__main__":
    main()
