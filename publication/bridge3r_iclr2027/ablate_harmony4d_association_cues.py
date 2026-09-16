#!/usr/bin/env python3
"""Offline cue ablation for the frozen Harmony4D boundary matcher.

All variants reuse the same m3_b0_only predictions, robust per-case cue
normalization, and Hungarian assignment.  Ground truth is opened only after
each prediction-only assignment has been fixed and is used solely for scoring.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


REPO_ROOT = Path(__file__).resolve().parents[2]
for root in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from scripts.boundary_human3r_reset_support import torso_frame  # noqa: E402
from versions.v14.probe_b0_identity_matching import (  # noqa: E402
    identity_cost_components,
    matching_costs,
)
from versions.v15.harmony4d.topology import CommonTopology  # noqa: E402
from publication.bridge3r_iclr2027 import (  # noqa: E402
    evaluate_harmony4d_boundary_association as common,
)


METHOD = "m3_b0_only"
VARIANTS = (
    ("pelvis", "root"),
    ("pelvis_torso", "root_torso"),
    ("pelvis_torso_joints", "root_torso_joints"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260908)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(common.jsonable(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def prediction_only_pairs(
    arrays: dict[str, np.ndarray], boundary: int
) -> dict[str, list[tuple[int, int]]]:
    valid = np.asarray(arrays["valid"], dtype=bool)
    joints = np.asarray(arrays["joints_world"], dtype=np.float64)
    pre_slots = np.flatnonzero(valid[boundary - 1])
    post_slots = np.flatnonzero(valid[boundary])
    if not len(pre_slots) or not len(post_slots):
        return {name: [] for name, _ in VARIANTS}

    def person(value: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "root": np.asarray(value[0]),
            "joints": np.asarray(value),
            "torso": torso_frame(value),
        }

    pre = {
        str(position): person(joints[boundary - 1, slot])
        for position, slot in enumerate(pre_slots)
    }
    post = [
        (str(position), person(joints[boundary, slot]))
        for position, slot in enumerate(post_slots)
    ]
    components = identity_cost_components(
        pre, post, np.eye(4, dtype=np.float64), tuple(pre)
    )
    costs = matching_costs(components)
    output = {}
    for display, cost_name in VARIANTS:
        rows, columns = linear_sum_assignment(costs[cost_name])
        output[display] = [
            (int(row), int(column)) for row, column in zip(rows, columns)
        ]
    return output


def bootstrap_sequence_macro(
    rows: list[dict[str, Any]], key: str, draws: int, seed: int
) -> list[float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(key)
        if value is not None and math.isfinite(float(value)):
            grouped[row["sequence"]].append(float(value))
    names = sorted(grouped)
    if not names:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        selected = rng.integers(0, len(names), size=len(names))
        values = []
        for index in selected:
            local = grouped[names[int(index)]]
            local_indices = rng.integers(0, len(local), size=len(local))
            values.extend(local[int(value)] for value in local_indices)
        samples[draw] = np.mean(values)
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def main() -> None:
    args = parse_args()
    topology = CommonTopology.load()
    cases: list[dict[str, Any]] = []
    parity = 0
    inputs = common.runtime_inputs(None, args.runtime_manifest)
    for case_index, (runtime_path, binding) in enumerate(inputs, start=1):
        if binding is None:
            raise ValueError("the cue ablation requires a frozen binding manifest")
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        record = runtime["record"]
        boundary = int(record["boundary_index"])
        cache = Path(binding["cache"]).resolve()
        arrays = common.minimal_arrays(cache, METHOD)
        predicted = prediction_only_pairs(arrays, boundary)
        frozen_pairs = [tuple(map(int, pair)) for pair in binding["final_boundary_pairs"]]
        parity += int(predicted["pelvis_torso_joints"] == frozen_pairs)

        sequence = str(binding["sequence"])
        extracted_root = args.staging_root.resolve() / f"test_{sequence}"
        gt, identities = common.load_gt(record, extracted_root, topology)
        pre_map, pre_slots, pre_gt = common.accepted_frame_assignment(arrays, gt, boundary - 1)
        post_map, post_slots, post_gt = common.accepted_frame_assignment(arrays, gt, boundary)
        assignments = common.all_accepted_assignments(arrays, gt)
        gt_continuations = len(set(pre_gt).intersection(post_gt))

        case: dict[str, Any] = {
            "case_id": str(record["case_id"]),
            "sequence": sequence,
            "angle_stratum": str(record.get("angle_stratum", "")),
            "variants": {},
        }
        for display, _ in VARIANTS:
            pairs = predicted[display]
            correct = evaluable = 0
            for pre_position, post_position in pairs:
                if pre_position >= len(pre_slots) or post_position >= len(post_slots):
                    raise ValueError(f"{record['case_id']}: pair exceeds endpoint list")
                pre_gt_id = pre_map.get(pre_slots[pre_position])
                post_gt_id = post_map.get(post_slots[post_position])
                valid_pair = pre_gt_id is not None and post_gt_id is not None
                evaluable += int(valid_pair)
                correct += int(valid_pair and pre_gt_id == post_gt_id)
            variant_arrays = {key: np.asarray(value).copy() for key, value in arrays.items()}
            variant_arrays["persistent_ids"] = common.reconstruct_runtime_ids(
                arrays, boundary, pairs
            )
            identity = common.identity_metrics(
                variant_arrays, assignments, identities, gt["visible"]
            )
            case["variants"][display] = {
                "pairs": pairs,
                "pair_count": len(pairs),
                "evaluable_pair_count": evaluable,
                "correct_pair_count": correct,
                "gt_continuation_count": gt_continuations,
                "pair_accuracy": float(correct / evaluable) if evaluable else None,
                "identity_continuation": (
                    float(correct / gt_continuations) if gt_continuations else None
                ),
                "idf1": float(identity["idf1"]),
            }
        cases.append(case)
        print(f">> {case_index}/{len(inputs)} {record['case_id']}", flush=True)

    summaries: dict[str, Any] = {}
    flat_rows = []
    for display, _ in VARIANTS:
        rows = []
        for case in cases:
            value = case["variants"][display]
            rows.append({"sequence": case["sequence"], **value})
            flat_rows.append({
                "case_id": case["case_id"],
                "sequence": case["sequence"],
                "angle_stratum": case["angle_stratum"],
                "variant": display,
                **{key: value[key] for key in (
                    "pair_count", "evaluable_pair_count", "correct_pair_count",
                    "gt_continuation_count", "pair_accuracy", "identity_continuation", "idf1"
                )},
            })
        correct = sum(row["correct_pair_count"] for row in rows)
        evaluable = sum(row["evaluable_pair_count"] for row in rows)
        continuations = sum(row["gt_continuation_count"] for row in rows)
        idf1 = [row["idf1"] for row in rows]
        summaries[display] = {
            "case_count": len(rows),
            "sequence_count": len({row["sequence"] for row in rows}),
            "correct_pair_count": correct,
            "evaluable_pair_count": evaluable,
            "pair_micro_accuracy": float(correct / evaluable) if evaluable else None,
            "gt_continuation_count": continuations,
            "identity_continuation": float(correct / continuations) if continuations else None,
            "idf1_case_mean": float(np.mean(idf1)),
            "idf1_sequence_cluster_bootstrap_ci95": bootstrap_sequence_macro(
                rows, "idf1", args.bootstrap_draws, args.seed
            ),
        }

    payload = {
        "schema_version": "Shot3R-Harmony4D-association-cue-ablation-v1",
        "status": "complete",
        "protocol": (
            "same frozen m3_b0_only predictions; per-case robust cue normalization; "
            "Hungarian assignment; GT used only after assignments are fixed"
        ),
        "runtime_manifest": str(args.runtime_manifest.resolve()),
        "runtime_manifest_sha256": sha256(args.runtime_manifest.resolve()),
        "staging_root": str(args.staging_root.resolve()),
        "case_count": len(cases),
        "full_cue_runtime_pair_parity": {"matched_cases": parity, "total_cases": len(cases)},
        "bootstrap": {"draws": args.bootstrap_draws, "seed": args.seed, "cluster": "sequence"},
        "summary": summaries,
        "cases": cases,
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "association_cue_ablation.json", payload)
    with (output / "association_cue_ablation_cases.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    print(json.dumps({
        "output": str(output),
        "parity": payload["full_cue_runtime_pair_parity"],
        "summary": summaries,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
