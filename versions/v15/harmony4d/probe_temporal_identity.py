#!/usr/bin/env python3
"""Dev-only probe for causal adjacent-frame persistent identity propagation."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from versions.v15.harmony4d.evaluate_harmony import (  # noqa: E402
    MAX_ASSIGNMENT_COST_M,
    frame_assignment,
    identity_metrics,
    load_gt,
    method_arrays,
    pelvis,
)
from versions.v15.harmony4d.topology import CommonTopology  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        nargs=2,
        metavar=("PREDICTIONS", "EXTRACTED_ROOT"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-adjacent-cost-m", type=float, default=0.50)
    return parser.parse_args()


def matching_cost(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    previous_root, current_root = pelvis(previous), pelvis(current)
    previous_body = previous - previous_root[:, None]
    current_body = current - current_root[:, None]
    cost = np.zeros((len(previous), len(current)), dtype=np.float64)
    for row in range(len(previous)):
        for column in range(len(current)):
            root = np.linalg.norm(previous_root[row] - current_root[column])
            body = np.linalg.norm(previous_body[row] - current_body[column], axis=1).mean()
            cost[row, column] = float(root + 0.25 * body)
    return cost


def causal_temporal_ids(
    arrays: dict[str, np.ndarray], boundary: int, maximum_cost: float
) -> tuple[np.ndarray, dict[str, Any]]:
    valid = np.asarray(arrays["valid"], dtype=bool)
    native = np.asarray(arrays["native_ids"], dtype=np.int32)
    joints = np.asarray(arrays["joints_world"], dtype=np.float64)
    output = np.full(native.shape, -1, dtype=np.int32)
    first = np.flatnonzero(valid[0])
    output[0, first] = np.arange(len(first), dtype=np.int32)
    next_id = len(first)
    debug = []
    for frame in range(1, len(valid)):
        previous = np.flatnonzero(valid[frame - 1])
        current = np.flatnonzero(valid[frame])
        pairs = []
        rejected = []
        if len(previous) and len(current):
            cost = matching_cost(joints[frame - 1, previous], joints[frame, current])
            rows, columns = linear_sum_assignment(cost)
            # At the shot boundary B0 provides only a coarse gauge, so retain
            # the existing full association behavior.  Within a shot, a
            # dustbin prevents a new/false detection from stealing a track.
            threshold = float("inf") if frame == boundary else maximum_cost
            for row, column in zip(rows, columns):
                if float(cost[row, column]) <= threshold:
                    pred_previous, pred_current = int(previous[row]), int(current[column])
                    output[frame, pred_current] = output[frame - 1, pred_previous]
                    pairs.append((pred_previous, pred_current, float(cost[row, column])))
                else:
                    rejected.append((int(previous[row]), int(current[column]), float(cost[row, column])))
        for index in current:
            if output[frame, index] < 0:
                output[frame, index] = next_id
                next_id += 1
        debug.append({
            "frame": frame,
            "is_boundary": frame == boundary,
            "pairs": pairs,
            "rejected": rejected,
            "native_ids": native[frame, current].tolist(),
            "persistent_ids": output[frame, current].tolist(),
        })
    return output, {
        "maximum_adjacent_cost_m": maximum_cost,
        "new_track_count": next_id,
        "frames": debug,
    }


def boundary_permutation_ids(
    arrays: dict[str, np.ndarray],
    boundary: int,
    frozen_boundary_pairs: list[list[int]] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply one cross-shot permutation, then trust causal native slot state."""

    valid = np.asarray(arrays["valid"], dtype=bool)
    native = np.asarray(arrays["native_ids"], dtype=np.int32)
    joints = np.asarray(arrays["joints_world"], dtype=np.float64)
    output = native.copy()
    previous = np.flatnonzero(valid[boundary - 1])
    current = np.flatnonzero(valid[boundary])
    mapping: dict[int, int] = {}
    next_id = int(output[:boundary][output[:boundary] >= 0].max(initial=-1)) + 1
    if frozen_boundary_pairs is not None:
        pairs = [(int(row), int(column)) for row, column in frozen_boundary_pairs]
        for row, column in pairs:
            if row >= len(previous) or column >= len(current):
                continue
            mapping[int(native[boundary, current[column]])] = int(
                output[boundary - 1, previous[row]]
            )
    elif len(previous) and len(current):
        cost = matching_cost(joints[boundary - 1, previous], joints[boundary, current])
        rows, columns = linear_sum_assignment(cost)
        for row, column in zip(rows, columns):
            mapping[int(native[boundary, current[column]])] = int(
                output[boundary - 1, previous[row]]
            )
    for frame in range(boundary, len(valid)):
        for index in np.flatnonzero(valid[frame]):
            slot = int(native[frame, index])
            if slot not in mapping:
                mapping[slot] = next_id
                next_id += 1
            output[frame, index] = mapping[slot]
    return output, {
        "boundary_native_slot_to_persistent_id": mapping,
        "new_track_count": next_id,
    }


def evaluator_assignments(
    arrays: dict[str, np.ndarray], gt: dict[str, np.ndarray]
) -> list[list[tuple[int, int]]]:
    assignments = []
    for frame in range(len(gt["cameras_c2w"])):
        valid = np.flatnonzero(arrays["valid"][frame].astype(bool))
        gt_valid = np.flatnonzero(gt["visible"][frame].astype(bool))
        pairs_local, costs = frame_assignment(
            arrays["cameras_c2w"][frame], arrays["joints_world"][frame, valid],
            gt["cameras_c2w"][frame], gt["joints_world"][frame, gt_valid],
        )
        assignments.append([
            (int(valid[row]), int(gt_valid[column]))
            for row, column in pairs_local
            if float(costs[row, column]) <= MAX_ASSIGNMENT_COST_M
        ])
    return assignments


def main() -> None:
    args = parse_args()
    topology = CommonTopology.load()
    rows = []
    for predictions_text, root_text in args.dataset:
        predictions, extracted_root = Path(predictions_text), Path(root_text)
        for runtime_path in sorted(predictions.glob("*.runtime.json")):
            case_id = runtime_path.name.removesuffix(".runtime.json")
            cache_path = runtime_path.with_name(case_id + ".npz")
            runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
            record = runtime["record"]
            gt, identities = load_gt(record, extracted_root, topology)
            with np.load(cache_path, allow_pickle=False) as cache:
                m3 = method_arrays(cache, "m3_b0_only")
                m4 = method_arrays(cache, "m4_b0_identity")
            temporal_dustbin = copy.deepcopy(m3)
            temporal_dustbin["persistent_ids"], debug = causal_temporal_ids(
                temporal_dustbin, int(record["boundary_index"]), float(args.max_adjacent_cost_m)
            )
            temporal_unbounded = copy.deepcopy(m3)
            temporal_unbounded["persistent_ids"], unbounded_debug = causal_temporal_ids(
                temporal_unbounded, int(record["boundary_index"]), float("inf")
            )
            boundary_permutation = copy.deepcopy(m3)
            boundary_permutation["persistent_ids"], boundary_debug = boundary_permutation_ids(
                boundary_permutation, int(record["boundary_index"])
            )
            runtime_permutation = copy.deepcopy(m3)
            runtime_permutation["persistent_ids"], runtime_boundary_debug = boundary_permutation_ids(
                runtime_permutation,
                int(record["boundary_index"]),
                runtime["geometry"]["association"]["pairs"],
            )
            assignments = evaluator_assignments(m3, gt)
            candidates = {
                "m3_b0_native": m3,
                "m4_frozen_identity": m4,
                "temporal_dustbin": temporal_dustbin,
                "temporal_unbounded": temporal_unbounded,
                "boundary_permutation": boundary_permutation,
                "runtime_boundary_permutation": runtime_permutation,
            }
            values = {
                method: identity_metrics(value, assignments, identities, gt["visible"])
                for method, value in candidates.items()
            }
            rows.append({
                "case_id": case_id,
                "sequence": record["sequence"],
                "angle_stratum": record["angle_stratum"],
                "methods": {
                    method: {
                        "IDs": result["ids_total"],
                        "IDF1": result["idf1"],
                        "association_accuracy": result["association_accuracy_best_global_mapping"],
                        "coverage": sum(len(pairs) for pairs in assignments)
                        / max(int(gt["visible"].sum()), 1),
                    }
                    for method, result in values.items()
                },
                "runtime_debug": {
                    "temporal_dustbin": debug,
                    "temporal_unbounded": unbounded_debug,
                    "boundary_permutation": boundary_debug,
                    "runtime_boundary_permutation": runtime_boundary_debug,
                },
            })
            print(json.dumps({"case_id": case_id, "methods": rows[-1]["methods"]}), flush=True)
    summary = {}
    for method in (
        "m3_b0_native", "m4_frozen_identity", "temporal_dustbin",
        "temporal_unbounded", "boundary_permutation",
        "runtime_boundary_permutation",
    ):
        summary[method] = {
            key: float(np.mean([row["methods"][method][key] for row in rows]))
            for key in ("IDs", "IDF1", "association_accuracy", "coverage")
        }
        summary[method]["IDs_total"] = int(sum(row["methods"][method]["IDs"] for row in rows))
    report = {
        "schema_version": "Movie3R-Harmony4D-temporal-ID-probe-v1",
        "causal": True,
        "future_frames": 0,
        "uses_gt_at_runtime": False,
        "candidates": {
            "temporal_dustbin": (
                "adjacent-frame root+centered-pose Hungarian; unbounded only at B0 boundary; "
                "0.50m within-shot dustbin"
            ),
            "temporal_unbounded": "adjacent-frame causal Hungarian without dustbin",
            "boundary_permutation": (
                "one B0-boundary permutation followed by the backbone's causal native slot state"
            ),
            "runtime_boundary_permutation": (
                "the exact frozen anonymous_match boundary pairs followed by causal native slot state"
            ),
        },
        "case_count": len(rows),
        "summary": summary,
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
