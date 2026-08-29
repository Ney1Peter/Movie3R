#!/usr/bin/env python3
"""Evaluator-only first-post-cut association audit for EgoHumans caches.

The runtime correspondence signal is read before any ground truth is opened.
Ground-truth SMPL identities and camera calibration are used only to score the
frozen prediction-only pairs.  The implementation deliberately shares the
Harmony4D audit's endpoint assignment, identity diagnostic, and aggregation
logic so that the two datasets have the same evaluation semantics.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from publication.bridge3r_iclr2027 import evaluate_harmony4d_boundary_association as common  # noqa: E402
from versions.v15.harmony4d.topology import CommonTopology  # noqa: E402
from versions.v19.egohumans.evaluate_egohumans import load_gt  # noqa: E402


SCHEMA = "Bridge3R-EgoHumans-boundary-association-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--extracted-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", default="m3_b0_only")
    parser.add_argument("--include-case-regex", default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260829)
    return parser.parse_args()


def evaluate_case(
    runtime_path: Path,
    extracted_root: Path,
    topology: CommonTopology,
    method: str,
    binding: dict[str, Any],
) -> dict[str, Any]:
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    record = runtime.get("record")
    if not isinstance(record, dict):
        raise KeyError(f"{runtime_path}: no record")
    boundary = int(record["boundary_index"])
    cache_path = runtime_path.with_name(runtime_path.name.removesuffix(".runtime.json") + ".npz")
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)

    # The frozen runtime signal and arrays are loaded before GT is opened.
    arrays = common.minimal_arrays(cache_path, method)
    pairs = common.parse_runtime_pairs(runtime)
    expected_pairs = [tuple(map(int, value)) for value in binding["final_boundary_pairs"]]
    if pairs != expected_pairs:
        raise ValueError(
            f"{record['case_id']}: runtime pairs differ from frozen binding: "
            f"{pairs} != {expected_pairs}"
        )
    if boundary <= 0 or boundary >= len(arrays["valid"]):
        raise ValueError(f"{record['case_id']}: invalid boundary {boundary}")

    gt, identities = load_gt(record, extracted_root, topology)
    pre_map, pre_slots, pre_gt = common.accepted_frame_assignment(arrays, gt, boundary - 1)
    post_map, post_slots, post_gt = common.accepted_frame_assignment(arrays, gt, boundary)
    maximum_cardinality = min(len(pre_slots), len(post_slots))
    if len(pairs) > maximum_cardinality:
        raise ValueError(
            f"{record['case_id']}: {len(pairs)} pairs exceed {maximum_cardinality} endpoints"
        )

    pair_rows = []
    correct = evaluable = 0
    for pre_position, post_position in pairs:
        if pre_position >= len(pre_slots) or post_position >= len(post_slots):
            raise ValueError(
                f"{record['case_id']}: pair {(pre_position, post_position)} exceeds "
                f"endpoint lists ({len(pre_slots)}, {len(post_slots)})"
            )
        pre_slot, post_slot = pre_slots[pre_position], post_slots[post_position]
        pre_gt_id, post_gt_id = pre_map.get(pre_slot), post_map.get(post_slot)
        is_evaluable = pre_gt_id is not None and post_gt_id is not None
        is_correct = bool(is_evaluable and pre_gt_id == post_gt_id)
        evaluable += int(is_evaluable)
        correct += int(is_correct)
        pair_rows.append(
            {
                "pre_position": int(pre_position),
                "post_position": int(post_position),
                "evaluable": is_evaluable,
                "correct": is_correct,
            }
        )

    assignments = common.all_accepted_assignments(arrays, gt)
    runtime_arrays = copy.deepcopy(arrays)
    runtime_arrays["persistent_ids"] = common.reconstruct_runtime_ids(arrays, boundary, pairs)
    oracle_arrays = copy.deepcopy(arrays)
    oracle_arrays["persistent_ids"] = common.oracle_ids_from_evaluator(arrays, assignments)
    runtime_identity = common.identity_metrics(runtime_arrays, assignments, identities, gt["visible"])
    oracle_identity = common.identity_metrics(oracle_arrays, assignments, identities, gt["visible"])
    gt_continuations = len(set(pre_gt).intersection(post_gt))

    return {
        "case_id": str(record["case_id"]),
        "sequence": str(record.get("sequence", "")),
        "capture": str(record.get("capture", "")),
        "angle_stratum": str(record.get("angle_stratum", "")),
        "camera_rotation_span_deg_evaluator_only": record.get(
            "camera_rotation_span_deg_evaluator_only"
        ),
        "boundary_index": boundary,
        "cache": str(cache_path.resolve()),
        "cache_sha256": common.sha256(cache_path),
        "runtime": str(runtime_path.resolve()),
        "runtime_sha256": common.sha256(runtime_path),
        "endpoint": {
            "runtime_pair_count": len(pairs),
            "maximum_runtime_pair_count": maximum_cardinality,
            "runtime_abstention_count": maximum_cardinality - len(pairs),
            "evaluable_pair_count": evaluable,
            "evaluator_excluded_pair_count": len(pairs) - evaluable,
            "correct_pair_count": correct,
            "correspondence_accuracy": float(correct / evaluable) if evaluable else None,
            "gt_continuation_count": gt_continuations,
            "correct_continuation_coverage": (
                float(correct / gt_continuations) if gt_continuations else None
            ),
            "pairs": pair_rows,
        },
        "identity_evaluator_only": {
            "cache_reconstructed_idf1_diagnostic": float(runtime_identity["idf1"]),
            "final_audit_idf1": float(binding["final_idf1"]),
            "oracle_framewise_idf1_upper_bound": float(oracle_identity["idf1"]),
            "runtime_ids_total": int(runtime_identity["ids_total"]),
            "oracle_ids_total": int(oracle_identity["ids_total"]),
        },
    }


def main() -> None:
    args = parse_args()
    selector = re.compile(args.include_case_regex) if args.include_case_regex else None
    topology = CommonTopology.load()
    rows = []
    for runtime_path, binding in common.runtime_inputs(None, args.runtime_manifest):
        assert binding is not None
        case_id = str(binding["case_id"])
        if selector and not selector.search(case_id):
            continue
        row = evaluate_case(
            runtime_path,
            args.extracted_root.resolve(),
            topology,
            str(args.method),
            binding,
        )
        rows.append(row)
        print(
            json.dumps(
                {
                    "case_id": case_id,
                    "accuracy": row["endpoint"]["correspondence_accuracy"],
                    "evaluable_pairs": row["endpoint"]["evaluable_pair_count"],
                }
            ),
            flush=True,
        )
    if not rows:
        raise ValueError("No cases remain after selection")
    rows.sort(key=lambda row: row["case_id"])
    payload = {
        "schema_version": SCHEMA,
        "method": str(args.method),
        "runtime_signal": "frozen geometry.association.pairs",
        "runtime_gt_access": False,
        "future_post_frames_used_by_association": 0,
        "runtime_manifest": str(args.runtime_manifest.resolve()),
        "extracted_root": str(args.extracted_root.resolve()),
        "evaluator": {
            "assignment": "shared camera--joint Hungarian endpoint assignment",
            "maximum_assignment_cost_m": float(common.MAX_ASSIGNMENT_COST_M),
            "oracle": "framewise evaluator GT identity labels; never provided to runtime",
        },
        "bootstrap": {
            "samples": int(args.bootstrap_samples),
            "seed": int(args.seed),
            "unit": "case",
        },
        "summary": common.aggregate(rows, int(args.bootstrap_samples), int(args.seed)),
        "cases": rows,
    }
    common.atomic_json(args.output.resolve(), payload)
    case_text = "".join(json.dumps(common.jsonable(row), sort_keys=True) + "\n" for row in rows)
    args.output.with_suffix(".cases.jsonl").write_text(case_text, encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "case_count": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
