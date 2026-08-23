#!/usr/bin/env python3
"""Evaluate finite prediction-only candidates with isolated EgoBody GT."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from versions.v15.harmony4d.evaluate_harmony import method_arrays  # noqa: E402
from versions.v15.harmony4d.topology import CommonTopology  # noqa: E402
from versions.v16.harmony4d.causal_stabilization import Candidate, apply_candidate  # noqa: E402
from versions.v16.harmony4d.probe_causal_stabilization import metric_row, write_csv  # noqa: E402
from versions.v19.egohumans.causal_identity import IdentityConfig, retrack_causally  # noqa: E402
from versions.v19.egohumans.joint_correction import PersonCorrectionConfig, person_boundary_correct  # noqa: E402
from versions.v20.egobody.evaluate_egobody import evaluate_method, load_gt  # noqa: E402


BASELINE = "v16_0_m15_geometry"
# Preserve the frozen aggregate adapter's accepted schema identifier.  The
# v20 causal-detector contract is declared separately in protocol_extension.
SCHEMA = "Movie3R-v16-Harmony4D-causal-stabilization-probe-v1"
ARRAY_KEYS = (
    "cameras_c2w",
    "vertices_world",
    "joints_world",
    "persistent_ids",
    "native_ids",
    "valid",
)


class ExactFallbackMismatch(RuntimeError):
    """Raised when a candidate claiming cached-parent fallback is not bit exact."""

    def __init__(self, audit: dict[str, Any]):
        super().__init__("declared exact fallback differs from cached causal parent")
        self.audit = audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--extracted-root", type=Path, required=True, help="v20 GT-cache root")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-json", type=Path, required=True)
    parser.add_argument("--reference-methods", nargs="*", default=[])
    parser.add_argument("--include-case-regex")
    return parser.parse_args()


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
    return value


def runtime_paths(roots: list[Path]) -> list[Path]:
    output, seen = [], set()
    for root in roots:
        for path in sorted(root.resolve().glob("*.runtime.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            case_id = str(payload["record"]["case_id"])
            if case_id in seen:
                raise ValueError(f"duplicate case: {case_id}")
            seen.add(case_id)
            output.append(path)
    if not output:
        raise FileNotFoundError("no runtime reports")
    return output


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_for_runtime(path: Path, runtime: dict[str, Any]) -> Path:
    adjacent = path.with_name(path.name.removesuffix(".runtime.json") + ".npz")
    if adjacent.is_file():
        return adjacent.resolve()
    declared = runtime.get("cache")
    if declared is None or not Path(declared).is_file():
        raise FileNotFoundError(adjacent if declared is None else declared)
    return Path(declared).resolve()


def nested_mapping(payload: dict[str, Any], path: tuple[str, ...]) -> dict[str, Any]:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            raise KeyError(".".join(path))
        current = current[key]
    if not isinstance(current, dict):
        raise TypeError(f"{'.'.join(path)} is not an object")
    return current


def detector_context(runtime: dict[str, Any]) -> dict[str, Any]:
    """Resolve the deployment event without consulting evaluator-only GT arrays."""

    record = runtime["record"]
    evaluation_boundary = int(record["boundary_index"])
    detector = nested_mapping(runtime, ("runtime", "causal_gru_detector"))
    if "proposal_boundary" in detector:
        proposal_raw = detector["proposal_boundary"]
    elif "first_positive_index" in detector:
        proposal_raw = detector["first_positive_index"]
    else:
        raise KeyError("causal_gru_detector misses proposal_boundary")
    proposal_boundary = None if proposal_raw is None else int(proposal_raw)
    labels = [int(value) for value in detector.get("labels", [])]
    positive_indices = [index for index, value in enumerate(labels) if value]
    derived_first = positive_indices[0] if positive_indices else None
    if labels and derived_first != proposal_boundary:
        raise ValueError(
            "detector proposal is inconsistent with the first positive label: "
            f"{proposal_boundary} vs {derived_first}"
        )
    declared_first = detector.get("first_positive_index", proposal_boundary)
    if declared_first is not None and int(declared_first) != proposal_boundary:
        raise ValueError(
            "first_positive_index differs from proposal_boundary: "
            f"{declared_first} vs {proposal_boundary}"
        )

    if proposal_boundary is None:
        outcome, subtype = "missed", None
        source_method = "m3_causal_detector_b0"
        parent_method = "m15_causal_detector_parent"
        association_path = None
        association: dict[str, Any] = {}
        pairs: list[tuple[int, int]] = []
    elif proposal_boundary == evaluation_boundary:
        outcome, subtype = "exact", None
        source_method = "m3_b0_only"
        parent_method = "m15_v17_gated_parent"
        association_keys = ("geometry", "evaluation_boundary", "association")
        association_path = ".".join(association_keys)
        association = nested_mapping(runtime, association_keys)
        pairs = [tuple(map(int, pair)) for pair in association.get("pairs", [])]
    else:
        outcome = "wrong"
        subtype = "early" if proposal_boundary < evaluation_boundary else "late"
        source_method = "m3_causal_detector_b0"
        parent_method = "m15_causal_detector_parent"
        association_keys = (
            "geometry", "detector_driven", "geometry", "association"
        )
        association_path = ".".join(association_keys)
        association = nested_mapping(runtime, association_keys)
        pairs = [tuple(map(int, pair)) for pair in association.get("pairs", [])]

    frame_count = len(labels) or int(
        len(record.get("pre_frame_numbers", []))
        + len(record.get("post_frame_numbers", []))
    )
    detector_seconds_raw = detector.get("seconds")
    detector_seconds = (
        None if detector_seconds_raw is None else float(detector_seconds_raw)
    )
    offset = (
        None
        if proposal_boundary is None
        else int(proposal_boundary - evaluation_boundary)
    )
    return {
        "detector_outcome": outcome,
        "wrong_temporal_subtype": subtype,
        "evaluation_boundary": evaluation_boundary,
        "proposal_boundary": proposal_boundary,
        "proposal_offset_frames": offset,
        "detector_latency_frames": offset,
        "detector_seconds": detector_seconds,
        "source_method": source_method,
        "cached_parent_method": parent_method,
        "association_path": association_path,
        "association_pair_count": len(pairs),
        "association_pairs": pairs,
        "association_metadata": association,
        "frame_count": frame_count,
        "fps": float(record["fps"]),
        "detector_fields": {
            "proposal_boundary": proposal_boundary,
            "first_positive_index": (
                proposal_boundary if declared_first is None else int(declared_first)
            ),
            "matches_evaluation_boundary": detector.get(
                "matches_evaluation_boundary",
                proposal_boundary == evaluation_boundary,
            ),
            "deployment_policy": detector.get("deployment_policy"),
            "false_positive_indices": detector.get("false_positive_indices"),
            "positive_indices": positive_indices,
            "label_count": len(labels),
        },
    }


def fallback_audit(
    arrays: dict[str, np.ndarray],
    cached_parent: dict[str, np.ndarray],
    parent_method: str,
    declared_exact_fallback: bool,
) -> dict[str, Any]:
    """Compare every array property and raw byte string against the cached parent."""

    candidate_keys = set(arrays)
    parent_keys = set(cached_parent)
    keys = sorted(candidate_keys | parent_keys)
    per_key: dict[str, Any] = {}
    for key in keys:
        candidate_value = arrays.get(key)
        parent_value = cached_parent.get(key)
        candidate_array = (
            None if candidate_value is None else np.asarray(candidate_value)
        )
        parent_array = None if parent_value is None else np.asarray(parent_value)
        dtype_equal = bool(
            candidate_array is not None
            and parent_array is not None
            and candidate_array.dtype == parent_array.dtype
        )
        shape_equal = bool(
            candidate_array is not None
            and parent_array is not None
            and candidate_array.shape == parent_array.shape
        )
        bytes_equal = bool(
            dtype_equal
            and shape_equal
            and candidate_array.tobytes() == parent_array.tobytes()
        )
        per_key[key] = {
            "candidate_present": candidate_array is not None,
            "parent_present": parent_array is not None,
            "candidate_dtype": (
                None if candidate_array is None else str(candidate_array.dtype)
            ),
            "parent_dtype": None if parent_array is None else str(parent_array.dtype),
            "dtype_equal": dtype_equal,
            "candidate_shape": (
                None if candidate_array is None else list(candidate_array.shape)
            ),
            "parent_shape": None if parent_array is None else list(parent_array.shape),
            "shape_equal": shape_equal,
            "raw_tobytes_equal": bytes_equal,
        }
    key_set_equal = candidate_keys == parent_keys
    bit_exact = key_set_equal and all(
        bool(value["dtype_equal"])
        and bool(value["shape_equal"])
        and bool(value["raw_tobytes_equal"])
        for value in per_key.values()
    )
    return {
        "comparison": "candidate_vs_cached_causal_parent",
        "cached_parent_method": parent_method,
        "declared_exact_fallback": bool(declared_exact_fallback),
        "candidate_keys": sorted(candidate_keys),
        "parent_keys": sorted(parent_keys),
        "key_set_equal": key_set_equal,
        "per_key": per_key,
        "bit_exact": bool(bit_exact),
    }


def enforce_exact_fallback(audit: dict[str, Any]) -> None:
    if audit["declared_exact_fallback"] and not audit["bit_exact"]:
        raise ExactFallbackMismatch(audit)


def angle_stratum(record: dict[str, Any]) -> str:
    # The stratum is protocol metadata, not an inference input.  It is encoded
    # in the immutable case ID while the runtime row remains GT-isolated.
    if record.get("angle_stratum"):
        return str(record["angle_stratum"])
    match = re.search(r"_(small|medium|extreme)_kinect_", str(record["case_id"]))
    if match is None:
        raise ValueError(f"case ID lacks angle stratum: {record['case_id']}")
    return match.group(1)


def materialize(
    source: dict[str, np.ndarray],
    cached_parent: dict[str, np.ndarray],
    proposal_boundary: int | None,
    pairs: list[tuple[int, int]],
    candidate: dict[str, Any],
    detector_outcome: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    name = str(candidate["name"])
    if name == BASELINE or detector_outcome == "missed":
        is_baseline = name == BASELINE
        arrays = {key: np.asarray(value).copy() for key, value in cached_parent.items()}
        reason = "cached_causal_baseline" if is_baseline else "detector_missed_no_event"
        return arrays, {
            "candidate": candidate,
            "identity": None,
            "shot_gauge": {"policy": "exact_noop"},
            "boundary": {
                "policy": "none",
                "accepted": False,
                "reason": reason,
                "materialization_boundary": None if detector_outcome == "missed" else proposal_boundary,
            },
            "reliability_gate": {
                "enabled": False,
                "accepted": is_baseline,
                "reasons": [] if is_baseline else ["detector_missed"],
                "fallback": "self" if is_baseline else "cached_causal_parent",
                "gt_used": False,
            },
            "root_filter": None,
            "runtime_contract": {
                "gt_used": False,
                "future_frames_used": 0,
                "pre_frames_rewritten_after_emission": False,
                "exact_m15_fallback": True,
                "cached_parent_reused_directly": True,
                "materialization_boundary": (
                    None if detector_outcome == "missed" else proposal_boundary
                ),
            },
        }
    if proposal_boundary is None:
        raise AssertionError("missed proposals must return the cached no-event parent")
    geometry_spec = dict(candidate.get("geometry") or {"name": name})
    geometry_spec["name"] = name
    arrays, geometry_debug = apply_candidate(
        source, proposal_boundary, pairs, Candidate(**geometry_spec)
    )
    if geometry_debug.get("runtime_contract", {}).get("exact_m15_fallback"):
        # Downstream identity/person corrections must not mutate a path that
        # declares exact cached-parent fallback.
        arrays = {key: np.asarray(value).copy() for key, value in cached_parent.items()}
        geometry_debug["runtime_contract"]["cached_parent_reused_directly"] = True
        geometry_debug["runtime_contract"]["materialization_boundary"] = proposal_boundary
        return arrays, {
            **geometry_debug,
            "candidate": candidate,
            "causal_identity": None,
            "person_correction": None,
        }
    identity_debug = None
    if candidate.get("identity") is not None:
        arrays, identity_debug = retrack_causally(
            arrays, proposal_boundary, pairs, IdentityConfig(**candidate["identity"])
        )
    person_debug = None
    if candidate.get("person") is not None:
        if identity_debug is None:
            raise ValueError("person correction requires causal identity transport")
        arrays, person_debug = person_boundary_correct(
            arrays, proposal_boundary, pairs, PersonCorrectionConfig(**candidate["person"])
        )
    geometry_debug.setdefault("runtime_contract", {})["materialization_boundary"] = proposal_boundary
    return arrays, {
        **geometry_debug,
        "candidate": candidate,
        "causal_identity": identity_debug,
        "person_correction": person_debug,
    }


def result_row(
    record: dict[str, Any], name_key: str, name: str, result: dict[str, Any], diagnostics: dict[str, Any] | None = None
) -> dict[str, Any]:
    row = {
        "case_id": str(record["case_id"]),
        "sequence": str(record["recording"]),
        "capture": str(record["recording"]),
        "angle_stratum": angle_stratum(record),
        name_key: name,
        "status": "complete",
        "metrics": metric_row(result),
        "within_shot_motion": result["within_shot_motion"],
    }
    if diagnostics is not None:
        row["diagnostics"] = diagnostics
    return row


def error_row(
    record: dict[str, Any],
    name_key: str,
    name: str,
    error: Exception,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "case_id": str(record["case_id"]),
        "sequence": str(record["recording"]),
        "capture": str(record["recording"]),
        "angle_stratum": angle_stratum(record),
        name_key: name,
        "status": "error",
        "error": f"{type(error).__name__}: {error}",
    }
    if diagnostics is not None:
        row["diagnostics"] = diagnostics
    return row


def main() -> None:
    args = parse_args()
    spec = json.loads(args.candidate_json.read_text(encoding="utf-8"))
    candidates = list(spec["candidates"])
    if BASELINE not in {str(row["name"]) for row in candidates}:
        candidates.insert(0, {"name": BASELINE, "geometry": {"name": BASELINE}, "identity": None})
    topology = CommonTopology.load()
    rows, references, errors = [], [], []
    for runtime_path in runtime_paths(args.prediction_roots):
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        record = runtime["record"]
        case_id = str(record["case_id"])
        if args.include_case_regex and not re.search(args.include_case_regex, case_id):
            continue
        deployment = detector_context(runtime)
        cache_path = cache_for_runtime(runtime_path, runtime)
        with np.load(cache_path, allow_pickle=False) as cache:
            source = method_arrays(cache, deployment["source_method"])
            cached_parent = method_arrays(cache, deployment["cached_parent_method"])
            refs = {method: method_arrays(cache, method) for method in args.reference_methods}
        if set(source) != set(ARRAY_KEYS) or set(cached_parent) != set(ARRAY_KEYS):
            raise ValueError(f"{case_id}: source/parent array schema is incomplete")
        frame_count = int(deployment["frame_count"])
        if len(source["valid"]) != frame_count or len(cached_parent["valid"]) != frame_count:
            raise ValueError(
                f"{case_id}: detector/source frame counts disagree: "
                f"{frame_count}, {len(source['valid'])}, {len(cached_parent['valid'])}"
            )
        gt, identities = load_gt(record, args.extracted_root, topology)
        pairs = deployment["association_pairs"]
        evaluation_boundary = int(deployment["evaluation_boundary"])
        proposal_boundary = deployment["proposal_boundary"]
        fps = float(deployment["fps"])
        process_seconds_raw = runtime.get("total_process_seconds")
        process_seconds = (
            None if process_seconds_raw is None else float(process_seconds_raw)
        )
        detector_seconds = deployment["detector_seconds"]
        case_diagnostics = {
            "deployment": deployment,
            "timing": {
                "total_process_seconds": process_seconds,
                "total_process_fps": (
                    None
                    if process_seconds is None or process_seconds <= 0.0
                    else frame_count / process_seconds
                ),
                "detector_seconds": detector_seconds,
                "detector_fps": (
                    None
                    if detector_seconds is None or detector_seconds <= 0.0
                    else frame_count / detector_seconds
                ),
                "frame_count": frame_count,
                "source_fps": fps,
            },
            "provenance": {
                "runtime_report": str(runtime_path.resolve()),
                "runtime_report_sha256": file_sha256(runtime_path),
                "runtime_schema_version": runtime.get("schema_version"),
                "cache": str(cache_path),
                "cache_declared": runtime.get("cache"),
                "cache_sha256": runtime.get("cache_sha256"),
                "checkpoint": runtime.get("checkpoint"),
                "runtime_provenance": runtime.get("provenance"),
            },
        }
        for method, arrays in refs.items():
            try:
                result = evaluate_method(
                    method, arrays, gt, identities, evaluation_boundary, fps
                )
                references.append(result_row(record, "method", method, result))
            except Exception as error:
                row = error_row(record, "method", method, error)
                references.append(row)
                errors.append(f"{case_id}:{method}:{row['error']}")
        for candidate in candidates:
            name = str(candidate["name"])
            diagnostics: dict[str, Any] = dict(case_diagnostics)
            try:
                started = time.perf_counter()
                arrays, materialization = materialize(
                    source,
                    cached_parent,
                    proposal_boundary,
                    pairs,
                    candidate,
                    str(deployment["detector_outcome"]),
                )
                elapsed = time.perf_counter() - started
                diagnostics["materialization"] = materialization
                # Preserve the v16/v19 diagnostics layout consumed by the
                # frozen aggregate/safety code while retaining the explicit
                # v20 materialization namespace above.
                diagnostics.update(materialization)
                diagnostics["postprocess_seconds"] = elapsed
                diagnostics["postprocess_frames"] = int(len(arrays["valid"]))
                diagnostics["candidate_postprocess"] = {
                    "seconds": elapsed,
                    "frames": int(len(arrays["valid"])),
                    "fps": None if elapsed <= 0.0 else len(arrays["valid"]) / elapsed,
                }
                declared_exact = bool(
                    materialization.get("runtime_contract", {}).get(
                        "exact_m15_fallback", False
                    )
                )
                audit = fallback_audit(
                    arrays,
                    cached_parent,
                    str(deployment["cached_parent_method"]),
                    declared_exact,
                )
                diagnostics["fallback_audit"] = audit
                enforce_exact_fallback(audit)
                result = evaluate_method(
                    name, arrays, gt, identities, evaluation_boundary, fps
                )
                rows.append(result_row(record, "candidate", name, result, diagnostics))
            except Exception as error:
                if isinstance(error, ExactFallbackMismatch):
                    diagnostics["fallback_audit"] = error.audit
                row = error_row(record, "candidate", name, error, diagnostics)
                rows.append(row)
                errors.append(f"{case_id}:{name}:{row['error']}")
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in rows + references:
        by_case.setdefault(str(row["case_id"]), []).append(row)
    expected = len(candidates) + len(args.reference_methods)
    known = {
        "ValueError: No initial matched people for shared world fit",
        "ValueError: Fewer than two valid pre-cut time points for shared W fit",
        "ValueError: No matched people for shared CS150 fit",
    }
    skipped = []
    for case_id, values in sorted(by_case.items()):
        if len(values) == expected and all(row.get("status") == "error" for row in values) and {str(row.get("error")) for row in values} <= known:
            skipped.append({
                "case_id": case_id,
                "status": "evaluator_unavailable",
                "reason": sorted({str(row.get("error")) for row in values}),
                "method_independent": True,
                "evaluations_affected": expected,
            })
    skipped_ids = {row["case_id"] for row in skipped}
    errors = [value for value in errors if value.split(":", 1)[0] not in skipped_ids]
    payload = {
        "schema_version": SCHEMA,
        "protocol_extension": "Bridge3R-v20-EgoBody-shared-CS150-v1",
        "candidate_source": str(args.candidate_json.resolve()),
        "candidate_count": len(candidates),
        "case_count": len(by_case),
        "complete_case_count": len({str(row["case_id"]) for row in rows if row.get("status") == "complete"}),
        "skipped_cases": skipped,
        "errors": errors,
        "rows": rows,
        "reference_rows": references,
        "aggregate": {},
        "contract": {
            "candidate_runtime_uses_gt": False,
            "gt_scope": "frozen evaluator only",
            "evaluation_boundary_scope": "GT scoring only",
            "candidate_materialization_boundary": "causal detector proposal only",
            "missed_proposal_policy": "bit-exact cached no-event parent for every candidate",
            "baseline_policy": "cached causal parent; never recomputed from an oracle source",
            "future_frames_at_boundary": 0,
            "source_cache_immutable": True,
            "w_wa_alignment": "one shared clip-level Sim(3) across both people",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    partial.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(partial, args.output)
    write_csv(args.output.with_suffix(".csv"), rows)
    print(json.dumps({"output": str(args.output.resolve()), "cases": payload["case_count"], "candidates": len(candidates), "errors": len(errors)}, indent=2))


if __name__ == "__main__":
    main()
