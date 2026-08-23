#!/usr/bin/env python3
"""Evaluate one EgoBody prediction cache against an isolated GT cache.

The legacy metric implementation is reused for all unchanged quantities.  Its
per-identity W/WA values are replaced with the preregistered CS150 definition:
one shared Sim(3) per clip and all matched people.  This preserves multi-person
layout instead of allowing each identity to move independently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from versions.v15.harmony4d import evaluate_harmony as frozen  # noqa: E402
from versions.v15.harmony4d.topology import CommonTopology  # noqa: E402


SCHEMA = "Bridge3R-EgoBody-CS150-evaluation-v1"


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_gt(
    record: dict[str, Any],
    gt_root: Path,
    topology: CommonTopology | None = None,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Load a prepare_gt.py cache; topology is accepted for probe compatibility."""

    del topology
    case_id = str(record["case_id"])
    path = gt_root.resolve() / f"{case_id}.gt.npz"
    metadata_path = gt_root.resolve() / f"{case_id}.gt.json"
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(path if not path.is_file() else metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if str(metadata.get("case_id")) != case_id:
        raise ValueError(f"GT cache case mismatch: {metadata.get('case_id')} vs {case_id}")
    expected_frames = np.asarray(
        record["pre_frame_numbers"] + record["post_frame_numbers"], dtype=np.int64
    )
    with np.load(path, allow_pickle=False) as cache:
        required = {
            "cameras_c2w", "vertices_world", "joints_world", "frames",
            "visible_fraction", "visible", "identities",
        }
        missing = required.difference(cache.files)
        if missing:
            raise KeyError(f"GT cache misses {sorted(missing)}")
        gt = {key: np.asarray(cache[key]) for key in required if key != "identities"}
        identities = [str(value) for value in np.asarray(cache["identities"]).tolist()]
    if not np.array_equal(gt["frames"], expected_frames):
        raise ValueError("GT cache frames do not match runtime row")
    frame_count = len(expected_frames)
    if gt["cameras_c2w"].shape != (frame_count, 4, 4):
        raise ValueError(f"bad GT camera shape {gt['cameras_c2w'].shape}")
    if gt["vertices_world"].shape != (frame_count, len(identities), 6890, 3):
        raise ValueError(f"bad GT vertices shape {gt['vertices_world'].shape}")
    if gt["joints_world"].shape != (frame_count, len(identities), 24, 3):
        raise ValueError(f"bad GT joints shape {gt['joints_world'].shape}")
    return gt, identities


def frame_assignments(
    arrays: dict[str, np.ndarray], gt: dict[str, np.ndarray]
) -> list[list[tuple[int, int]]]:
    output: list[list[tuple[int, int]]] = []
    for frame in range(len(gt["frames"])):
        pred_valid = np.flatnonzero(arrays["valid"][frame].astype(bool))
        gt_valid = np.flatnonzero(gt["visible"][frame].astype(bool))
        local, costs = frozen.frame_assignment(
            arrays["cameras_c2w"][frame],
            arrays["joints_world"][frame, pred_valid],
            gt["cameras_c2w"][frame],
            gt["joints_world"][frame, gt_valid],
        )
        output.append([
            (int(pred_valid[row]), int(gt_valid[column]))
            for row, column in local
            if float(costs[row, column]) <= frozen.MAX_ASSIGNMENT_COST_M
        ])
    return output


def fit_from_frames(
    arrays: dict[str, np.ndarray],
    gt: dict[str, np.ndarray],
    assignments: list[list[tuple[int, int]]],
    frames: list[int],
) -> tuple[float, np.ndarray, np.ndarray]:
    target, prediction = [], []
    for frame in frames:
        for pred_index, gt_index in assignments[frame]:
            target.append(gt["joints_world"][frame, gt_index])
            prediction.append(arrays["joints_world"][frame, pred_index])
    if not target:
        raise ValueError("No matched people for shared CS150 fit")
    return frozen.fit_similarity(np.stack(target), np.stack(prediction), allow_scale=True)


def shared_cs150_alignment(
    arrays: dict[str, np.ndarray],
    gt: dict[str, np.ndarray],
    boundary: int,
) -> dict[str, Any]:
    assignments = frame_assignments(arrays, gt)
    valid_pre_times = [frame for frame in range(boundary) if assignments[frame]]
    if len(valid_pre_times) < 2:
        raise ValueError("Fewer than two valid pre-cut time points for shared W fit")
    initial_times = valid_pre_times[:2]
    all_times = [frame for frame, pairs in enumerate(assignments) if pairs]
    w_fit = fit_from_frames(arrays, gt, assignments, initial_times)
    wa_fit = fit_from_frames(arrays, gt, assignments, all_times)
    w_errors, wa_errors = [], []
    w_by_frame: list[list[float]] = [[] for _ in assignments]
    wa_by_frame: list[list[float]] = [[] for _ in assignments]
    for frame, pairs in enumerate(assignments):
        for pred_index, gt_index in pairs:
            prediction = arrays["joints_world"][frame, pred_index]
            target = gt["joints_world"][frame, gt_index]
            w_value = float(np.linalg.norm(
                frozen.apply_similarity(prediction, w_fit) - target, axis=-1
            ).mean())
            wa_value = float(np.linalg.norm(
                frozen.apply_similarity(prediction, wa_fit) - target, axis=-1
            ).mean())
            w_errors.append(w_value)
            wa_errors.append(wa_value)
            w_by_frame[frame].append(w_value)
            wa_by_frame[frame].append(wa_value)
    return {
        "definition": (
            "one clip-level shared Sim(3); W fits all matched identities at the "
            "earliest two valid pre-cut times; WA fits all matched identities/times"
        ),
        "w_mpjpe_mm": frozen.summarize(w_errors, 1000.0),
        "wa_mpjpe_mm": frozen.summarize(wa_errors, 1000.0),
        "w_first_post_mm": frozen.summarize(w_by_frame[boundary], 1000.0),
        "w_post_mm": frozen.summarize(
            [value for row in w_by_frame[boundary:] for value in row], 1000.0
        ),
        "wa_first_post_mm": frozen.summarize(wa_by_frame[boundary], 1000.0),
        "initial_fit_times": initial_times,
        "matched_person_frames": int(sum(len(row) for row in assignments)),
        "w_fit": {"scale": w_fit[0], "rotation": w_fit[1], "translation": w_fit[2]},
        "wa_fit": {"scale": wa_fit[0], "rotation": wa_fit[1], "translation": wa_fit[2]},
    }


def evaluate_method(
    method: str,
    arrays: dict[str, np.ndarray],
    gt: dict[str, np.ndarray],
    identities: list[str],
    boundary: int,
    fps: float,
) -> dict[str, Any]:
    result = frozen.evaluate_method(method, arrays, gt, identities, boundary, fps)
    shared = shared_cs150_alignment(arrays, gt, boundary)
    named = result["multi_thumbs_named_provisional"]
    result["legacy_per_identity_alignment"] = {
        "w_mpjpe_mm": named["w_mpjpe_mm"],
        "wa_mpjpe_mm": named["wa_mpjpe_mm"],
        "status": "diagnostic_only_not_the_Bridge3R_CS150_headline",
    }
    named["w_mpjpe_mm"] = shared["w_mpjpe_mm"]
    named["wa_mpjpe_mm"] = shared["wa_mpjpe_mm"]
    result["cs150_shared_alignment"] = shared
    result["metric_contract"] = {
        "w_wa_alignment_scope": "one shared Sim(3) per clip across both people",
        "multi_thumbs_official_evaluator": False,
        "literature_comparability": "protocol-different context only",
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--runtime-report", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = json.loads(args.runtime_report.read_text(encoding="utf-8"))
    record = runtime["record"]
    gt, identities = load_gt(record, args.gt_root)
    results, errors = {}, {}
    with np.load(args.cache, allow_pickle=False) as cache:
        for method in runtime["methods"]:
            try:
                results[method] = evaluate_method(
                    method,
                    frozen.method_arrays(cache, method),
                    gt,
                    identities,
                    int(record["boundary_index"]),
                    float(record["fps"]),
                )
            except Exception as error:
                errors[method] = f"{type(error).__name__}: {error}"
    detector = {}
    if "causal_gru_detector" in runtime.get("runtime", {}):
        detector["causal_gru"] = frozen.detector_metrics(
            runtime, int(record["boundary_index"]), "causal_gru_detector"
        )
    payload = {
        "schema_version": SCHEMA,
        "protocol": record["protocol"],
        "case_id": record["case_id"],
        "record_runtime_fields": record,
        "identities": identities,
        "methods": results,
        "errors": errors,
        "detectors": detector,
        "inputs": {
            "prediction_cache": str(args.cache.resolve()),
            "prediction_cache_sha256": sha256(args.cache.resolve()),
            "runtime_report": str(args.runtime_report.resolve()),
            "runtime_report_sha256": sha256(args.runtime_report.resolve()),
            "gt_cache": str((args.gt_root / f"{record['case_id']}.gt.npz").resolve()),
        },
        "gt_summary": {
            "coordinate_system": "metric EgoBody scene world",
            "source_body_model": "SMPL-X 10475",
            "evaluation_topology": "common SMPL 6890 / 24 joints",
            "frame_count": int(len(gt["frames"])),
            "visible_gt_person_frames": int(gt["visible"].sum()),
        },
        "evaluation_contract": {
            "gt_used_only_in_evaluator": True,
            "runtime_gt_access": False,
            "matching": "per-frame Hungarian in camera coordinates",
            "w_wa": "shared clip-level Sim(3), not per-identity",
            "test_tuning": False,
            "literature_protocol_status": (
                "custom preregistered CS150; Multi-THuMBS public numbers are context only"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    partial.write_text(
        json.dumps(jsonable(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, args.output)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "case_id": record["case_id"],
        "methods": len(results),
        "errors": errors,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
