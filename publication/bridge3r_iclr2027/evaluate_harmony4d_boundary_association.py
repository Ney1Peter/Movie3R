#!/usr/bin/env python3
"""Evaluator-only first-post-cut association audit for frozen Harmony4D caches.

The runtime signal is read exclusively from ``geometry.association.pairs`` in
each frozen RGB-only runtime report.  Harmony4D calibration and identities are
opened only after that signal has been loaded, to evaluate whether a pair links
the same GT identity immediately before and after the cut.  This script does
not run a model or modify cached predictions.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from versions.v15.harmony4d.evaluate_harmony import (  # noqa: E402
    MAX_ASSIGNMENT_COST_M,
    frame_assignment,
    identity_metrics,
    load_gt,
)
from versions.v15.harmony4d.topology import CommonTopology  # noqa: E402


REQUIRED_ARRAYS = ("cameras_c2w", "joints_world", "persistent_ids", "native_ids", "valid")
SCHEMA = "Bridge3R-Harmony4D-boundary-association-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction-roots", type=Path, nargs="+", default=None,
        help="Directories containing one frozen .npz and .runtime.json per case.",
    )
    parser.add_argument(
        "--runtime-manifest", type=Path, default=None,
        help=(
            "Frozen JSONL result-binding manifest. Each row names one exact "
            "runtime cache and its final-audit boundary pairs."
        ),
    )
    parser.add_argument("--extracted-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", default="m3_b0_only")
    parser.add_argument("--include-case-regex", default=None)
    parser.add_argument("--include-sequence", default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260825)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
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
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def runtime_inputs(
    roots: list[Path] | None, runtime_manifest: Path | None,
) -> list[tuple[Path, dict[str, Any] | None]]:
    output: list[tuple[Path, dict[str, Any] | None]] = []
    for root in roots or []:
        output.extend((path, None) for path in sorted(root.resolve().glob("*.runtime.json")))
    if runtime_manifest is not None:
        if not runtime_manifest.is_file():
            raise FileNotFoundError(runtime_manifest)
        for line in runtime_manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            binding = json.loads(line)
            path = Path(binding["runtime"]).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            output.append((path, binding))
    if not output:
        raise FileNotFoundError("No .runtime.json files in --prediction-roots/--runtime-manifest")
    seen = set()
    for path, binding in output:
        runtime = json.loads(path.read_text(encoding="utf-8"))
        case_id = str(runtime.get("record", {}).get("case_id", ""))
        if not case_id or case_id in seen:
            raise ValueError(f"Duplicate or missing runtime case id: {path}")
        if binding is not None and str(binding.get("case_id", "")) != case_id:
            raise ValueError(f"Manifest/runtime case mismatch: {binding.get('case_id')} != {case_id}")
        seen.add(case_id)
    return output


def minimal_arrays(cache_path: Path, method: str) -> dict[str, np.ndarray]:
    prefix = method + "__"
    with np.load(cache_path, allow_pickle=False) as cache:
        missing = [prefix + key for key in REQUIRED_ARRAYS if prefix + key not in cache.files]
        if missing:
            raise KeyError(f"{cache_path}: missing {missing}")
        return {key: np.asarray(cache[prefix + key]) for key in REQUIRED_ARRAYS}


def accepted_frame_assignment(
    arrays: dict[str, np.ndarray], gt: dict[str, np.ndarray], frame: int,
) -> tuple[dict[int, int], list[int], list[int]]:
    """Return only evaluator-accepted prediction-slot -> GT-slot matches."""

    predicted_slots = np.flatnonzero(np.asarray(arrays["valid"][frame]).astype(bool))
    gt_slots = np.flatnonzero(np.asarray(gt["visible"][frame]).astype(bool))
    pairs, costs = frame_assignment(
        arrays["cameras_c2w"][frame], arrays["joints_world"][frame, predicted_slots],
        gt["cameras_c2w"][frame], gt["joints_world"][frame, gt_slots],
    )
    output: dict[int, int] = {}
    for row, column in pairs:
        if float(costs[row, column]) <= MAX_ASSIGNMENT_COST_M:
            output[int(predicted_slots[row])] = int(gt_slots[column])
    return output, [int(value) for value in predicted_slots], [int(value) for value in gt_slots]


def all_accepted_assignments(
    arrays: dict[str, np.ndarray], gt: dict[str, np.ndarray],
) -> list[list[tuple[int, int]]]:
    output = []
    for frame in range(len(arrays["valid"])):
        mapping, _, _ = accepted_frame_assignment(arrays, gt, frame)
        output.append(sorted(mapping.items()))
    return output


def parse_runtime_pairs(runtime: dict[str, Any]) -> list[tuple[int, int]]:
    pairs = runtime.get("geometry", {}).get("association", {}).get("pairs")
    if not isinstance(pairs, list):
        raise KeyError("runtime report lacks geometry.association.pairs")
    output = []
    for value in pairs:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"Malformed runtime association pair: {value!r}")
        first, second = int(value[0]), int(value[1])
        if first < 0 or second < 0:
            raise ValueError(f"Negative runtime association pair: {value!r}")
        output.append((first, second))
    if len({first for first, _ in output}) != len(output):
        raise ValueError(f"Runtime association reuses a pre-cut endpoint: {output}")
    if len({second for _, second in output}) != len(output):
        raise ValueError(f"Runtime association reuses a post-cut endpoint: {output}")
    return output


def reconstruct_runtime_ids(
    arrays: dict[str, np.ndarray], boundary: int, pairs: list[tuple[int, int]],
) -> np.ndarray:
    """Reconstruct the frozen one-boundary permutation using native slots.

    It deliberately uses no GT.  The returned labels are only used to provide
    a same-detection runtime-IDF1 reference beside the evaluator-only oracle
    identity upper bound.
    """

    valid = np.asarray(arrays["valid"], dtype=bool)
    native = np.asarray(arrays["native_ids"], dtype=np.int64)
    output = native.copy()
    pre_slots = np.flatnonzero(valid[boundary - 1])
    post_slots = np.flatnonzero(valid[boundary])
    mapping: dict[int, int] = {}
    used = {int(value) for value in output[:boundary][output[:boundary] >= 0]}
    next_id = max(used, default=-1) + 1
    for pre_position, post_position in pairs:
        if pre_position >= len(pre_slots) or post_position >= len(post_slots):
            raise ValueError(
                f"Runtime pair {(pre_position, post_position)} exceeds valid endpoint "
                f"counts ({len(pre_slots)}, {len(post_slots)})"
            )
        pre_slot, post_slot = int(pre_slots[pre_position]), int(post_slots[post_position])
        mapping[int(native[boundary, post_slot])] = int(output[boundary - 1, pre_slot])
    for frame in range(boundary, len(valid)):
        for slot in np.flatnonzero(valid[frame]):
            native_id = int(native[frame, slot])
            if native_id not in mapping:
                mapping[native_id] = next_id
                next_id += 1
            output[frame, slot] = mapping[native_id]
    return output


def oracle_ids_from_evaluator(
    arrays: dict[str, np.ndarray], assignments: list[list[tuple[int, int]]],
) -> np.ndarray:
    """GT-ID labels for an explicit evaluator-only all-frame upper bound."""

    output = np.full(np.asarray(arrays["persistent_ids"]).shape, -1, dtype=np.int64)
    for frame, pairs in enumerate(assignments):
        for prediction_slot, gt_slot in pairs:
            output[frame, prediction_slot] = int(gt_slot)
    return output


def bootstrap_case_mean(values: list[float], samples: int, seed: int) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if not len(finite):
        return {"count": 0, "mean": None, "ci95_low": None, "ci95_high": None}
    if int(samples) <= 0:
        return {"count": int(len(finite)), "mean": float(finite.mean()), "ci95_low": None, "ci95_high": None}
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, len(finite), size=(int(samples), len(finite)))
    estimates = finite[draws].mean(axis=1)
    return {
        "count": int(len(finite)),
        "mean": float(finite.mean()),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
    }


def evaluate_case(
    runtime_path: Path, extracted_root: Path, topology: CommonTopology, method: str,
    binding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    record = runtime.get("record")
    if not isinstance(record, dict):
        raise KeyError(f"{runtime_path}: no record")
    boundary = int(record["boundary_index"])
    cache_path = runtime_path.with_name(runtime_path.name.removesuffix(".runtime.json") + ".npz")
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)
    arrays = minimal_arrays(cache_path, method)
    if boundary <= 0 or boundary >= len(arrays["valid"]):
        raise ValueError(f"{record['case_id']}: invalid boundary {boundary}")
    # Runtime data are fully loaded before this evaluator-only call opens GT.
    pairs = parse_runtime_pairs(runtime)
    if binding is not None:
        expected_pairs = [tuple(map(int, value)) for value in binding["final_boundary_pairs"]]
        if pairs != expected_pairs:
            raise ValueError(
                f"{record['case_id']}: runtime pairs do not match frozen final-audit binding "
                f"{pairs} != {expected_pairs}"
            )
    gt, identities = load_gt(record, extracted_root, topology)
    pre_map, pre_slots, pre_gt = accepted_frame_assignment(arrays, gt, boundary - 1)
    post_map, post_slots, post_gt = accepted_frame_assignment(arrays, gt, boundary)
    maximum_cardinality = min(len(pre_slots), len(post_slots))
    if len(pairs) > maximum_cardinality:
        raise ValueError(f"{record['case_id']}: {len(pairs)} pairs exceed {maximum_cardinality} endpoints")

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
        pair_rows.append({
            "pre_position": pre_position,
            "post_position": post_position,
            "evaluable": is_evaluable,
            "correct": is_correct,
        })

    gt_continuations = len(set(pre_gt).intersection(post_gt))
    assignments = all_accepted_assignments(arrays, gt)
    runtime_arrays = copy.deepcopy(arrays)
    runtime_arrays["persistent_ids"] = reconstruct_runtime_ids(arrays, boundary, pairs)
    oracle_arrays = copy.deepcopy(arrays)
    oracle_arrays["persistent_ids"] = oracle_ids_from_evaluator(arrays, assignments)
    runtime_identity = identity_metrics(runtime_arrays, assignments, identities, gt["visible"])
    oracle_identity = identity_metrics(oracle_arrays, assignments, identities, gt["visible"])
    return {
        "case_id": str(record["case_id"]),
        "sequence": str(record.get("sequence", "")),
        "angle_stratum": str(record.get("angle_stratum", "")),
        "boundary_index": boundary,
        "cache": str(cache_path.resolve()),
        "cache_sha256": sha256(cache_path),
        "runtime": str(runtime_path.resolve()),
        "runtime_sha256": sha256(runtime_path),
        "endpoint": {
            "runtime_pair_count": len(pairs),
            "maximum_runtime_pair_count": maximum_cardinality,
            "runtime_abstention_count": maximum_cardinality - len(pairs),
            "evaluable_pair_count": evaluable,
            "evaluator_excluded_pair_count": len(pairs) - evaluable,
            "correct_pair_count": correct,
            "correspondence_accuracy": (float(correct / evaluable) if evaluable else None),
            "gt_continuation_count": gt_continuations,
            "correct_continuation_coverage": (
                float(correct / gt_continuations) if gt_continuations else None
            ),
            "pairs": pair_rows,
        },
        "identity_evaluator_only": {
            "cache_reconstructed_idf1_diagnostic": float(runtime_identity["idf1"]),
            "final_audit_idf1": (None if binding is None else float(binding["final_idf1"])),
            "oracle_framewise_idf1_upper_bound": float(oracle_identity["idf1"]),
            "runtime_ids_total": int(runtime_identity["ids_total"]),
            "oracle_ids_total": int(oracle_identity["ids_total"]),
        },
    }


def aggregate(rows: list[dict[str, Any]], samples: int, seed: int) -> dict[str, Any]:
    endpoint_rows = [row["endpoint"] for row in rows]
    evaluable = int(sum(row["evaluable_pair_count"] for row in endpoint_rows))
    correct = int(sum(row["correct_pair_count"] for row in endpoint_rows))
    maximum = int(sum(row["maximum_runtime_pair_count"] for row in endpoint_rows))
    abstentions = int(sum(row["runtime_abstention_count"] for row in endpoint_rows))
    continuations = int(sum(row["gt_continuation_count"] for row in endpoint_rows))
    cache_idf1 = [float(row["identity_evaluator_only"]["cache_reconstructed_idf1_diagnostic"]) for row in rows]
    final_idf1 = [
        float(row["identity_evaluator_only"]["final_audit_idf1"])
        for row in rows if row["identity_evaluator_only"]["final_audit_idf1"] is not None
    ]
    oracle_idf1 = [float(row["identity_evaluator_only"]["oracle_framewise_idf1_upper_bound"]) for row in rows]
    return {
        "case_count": len(rows),
        "first_post_cut_correspondence": {
            "pair_micro_accuracy": float(correct / evaluable) if evaluable else None,
            "correct_pair_count": correct,
            "evaluable_pair_count": evaluable,
            "case_macro_accuracy": bootstrap_case_mean(
                [row["correspondence_accuracy"] for row in endpoint_rows if row["correspondence_accuracy"] is not None],
                samples, seed,
            ),
            "correct_continuation_coverage": float(correct / continuations) if continuations else None,
            "gt_continuation_count": continuations,
            "runtime_abstention_rate": float(abstentions / maximum) if maximum else None,
            "runtime_abstention_count": abstentions,
            "maximum_runtime_pair_count": maximum,
            "evaluator_excluded_pair_count": int(sum(row["evaluator_excluded_pair_count"] for row in endpoint_rows)),
        },
        "identity": {
            "cache_reconstructed_idf1_diagnostic": bootstrap_case_mean(cache_idf1, samples, seed + 1),
            "final_audit_idf1": bootstrap_case_mean(final_idf1, samples, seed + 2),
            "oracle_framewise_idf1_upper_bound": bootstrap_case_mean(oracle_idf1, samples, seed + 3),
        },
    }


def main() -> None:
    args = parse_args()
    if not args.prediction_roots and args.runtime_manifest is None:
        raise ValueError("Provide --prediction-roots and/or --runtime-manifest")
    topology = CommonTopology.load()
    selector = re.compile(args.include_case_regex) if args.include_case_regex else None
    rows = []
    seen = set()
    for runtime_path, binding in runtime_inputs(args.prediction_roots, args.runtime_manifest):
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        case_id = str(runtime.get("record", {}).get("case_id", ""))
        if selector and not selector.search(case_id):
            continue
        if args.include_sequence and str(runtime.get("record", {}).get("sequence", "")) != args.include_sequence:
            continue
        if not case_id:
            raise KeyError(f"{runtime_path}: missing record.case_id")
        if case_id in seen:
            raise ValueError(f"Duplicate case id: {case_id}")
        seen.add(case_id)
        row = evaluate_case(runtime_path, args.extracted_root.resolve(), topology, args.method, binding)
        rows.append(row)
        print(json.dumps({
            "case_id": case_id,
            "accuracy": row["endpoint"]["correspondence_accuracy"],
            "evaluable_pairs": row["endpoint"]["evaluable_pair_count"],
        }), flush=True)
    if not rows:
        raise ValueError("No cases remain after --include-case-regex")
    rows.sort(key=lambda row: row["case_id"])
    payload = {
        "schema_version": SCHEMA,
        "method": str(args.method),
        "runtime_signal": "frozen geometry.association.pairs",
        "runtime_gt_access": False,
        "future_post_frames_used_by_association": 0,
        "prediction_roots": [str(path.resolve()) for path in (args.prediction_roots or [])],
        "runtime_manifest": None if args.runtime_manifest is None else str(args.runtime_manifest.resolve()),
        "extracted_root": str(args.extracted_root.resolve()),
        "evaluator": {
            "assignment": "existing Harmony4D camera--joint Hungarian assignment",
            "maximum_assignment_cost_m": float(MAX_ASSIGNMENT_COST_M),
            "oracle": "framewise evaluator GT identity labels; never provided to runtime",
        },
        "bootstrap": {"samples": int(args.bootstrap_samples), "seed": int(args.seed), "unit": "case"},
        "summary": aggregate(rows, int(args.bootstrap_samples), int(args.seed)),
        "cases": rows,
    }
    atomic_json(args.output.resolve(), payload)
    lines = "".join(json.dumps(jsonable(row), sort_keys=True) + "\n" for row in rows)
    args.output.with_suffix(".cases.jsonl").write_text(lines, encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "case_count": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
