#!/usr/bin/env python3
"""Evaluate finite v19 joint-geometry/causal-identity candidates."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from versions.v15.harmony4d.evaluate_harmony import evaluate_method, method_arrays
from versions.v15.harmony4d.topology import CommonTopology
from versions.v16.harmony4d.causal_stabilization import Candidate, apply_candidate
from versions.v16.harmony4d.probe_causal_stabilization import METRICS, metric_row, write_csv
from versions.v19.egohumans.causal_identity import IdentityConfig, retrack_causally
from versions.v19.egohumans.evaluate_egohumans import load_gt
from versions.v19.egohumans.joint_correction import (
    PersonCorrectionConfig,
    person_boundary_correct,
)


BASELINE = "v16_0_m15_geometry"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--extracted-root", type=Path, required=True)
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
    values = []
    seen = set()
    for root in roots:
        for path in sorted(root.resolve().glob("*.runtime.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            case_id = str(payload["record"]["case_id"])
            if case_id in seen:
                raise ValueError(f"duplicate case: {case_id}")
            seen.add(case_id)
            values.append(path)
    if not values:
        raise FileNotFoundError("no runtime reports")
    return values


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
        cache_path = runtime_path.with_name(runtime_path.name.removesuffix(".runtime.json") + ".npz")
        with np.load(cache_path, allow_pickle=False) as cache:
            source = method_arrays(cache, "m3_b0_only")
            refs = {method: method_arrays(cache, method) for method in args.reference_methods}
        gt, identities = load_gt(record, args.extracted_root, topology)
        pairs = [tuple(map(int, pair)) for pair in runtime.get("geometry", {}).get("association", {}).get("pairs", [])]
        for method, arrays in refs.items():
            try:
                result = evaluate_method(method, arrays, gt, identities, int(record["boundary_index"]), float(record["fps"]))
                references.append(
                    {
                        "case_id": case_id,
                        "sequence": record["sequence"],
                        "capture": record.get("capture"),
                        "angle_stratum": record["angle_stratum"],
                        "person_count": record.get("person_count_evaluator_only"),
                        "method": method,
                        "status": "complete",
                        "metrics": metric_row(result),
                        "within_shot_motion": result["within_shot_motion"],
                    }
                )
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                references.append(
                    {
                        "case_id": case_id,
                        "sequence": record["sequence"],
                        "capture": record.get("capture"),
                        "angle_stratum": record["angle_stratum"],
                        "person_count": record.get("person_count_evaluator_only"),
                        "method": method,
                        "status": "error",
                        "error": message,
                    }
                )
                errors.append(f"{case_id}:{method}:{message}")
        for candidate in candidates:
            name = str(candidate["name"])
            try:
                postprocess_started = time.perf_counter()
                geometry_spec = dict(candidate.get("geometry") or {"name": BASELINE})
                geometry_spec["name"] = name
                arrays, geometry_debug = apply_candidate(
                    source, int(record["boundary_index"]), pairs, Candidate(**geometry_spec)
                )
                identity_debug = None
                if candidate.get("identity") is not None:
                    arrays, identity_debug = retrack_causally(
                        arrays,
                        int(record["boundary_index"]),
                        pairs,
                        IdentityConfig(**candidate["identity"]),
                    )
                person_debug = None
                if candidate.get("person") is not None:
                    if identity_debug is None:
                        raise ValueError("person correction requires causal identity transport")
                    arrays, person_debug = person_boundary_correct(
                        arrays,
                        int(record["boundary_index"]),
                        pairs,
                        PersonCorrectionConfig(**candidate["person"]),
                    )
                postprocess_seconds = time.perf_counter() - postprocess_started
                result = evaluate_method(name, arrays, gt, identities, int(record["boundary_index"]), float(record["fps"]))
                rows.append(
                    {
                        "case_id": case_id,
                        "sequence": record["sequence"],
                        "capture": record.get("capture"),
                        "angle_stratum": record["angle_stratum"],
                        "person_count": record.get("person_count_evaluator_only"),
                        "candidate": name,
                        "status": "complete",
                        "metrics": metric_row(result),
                        "within_shot_motion": result["within_shot_motion"],
                        "diagnostics": {
                            **geometry_debug,
                            "candidate": candidate,
                            "causal_identity": identity_debug,
                            "person_correction": person_debug,
                            "postprocess_seconds": postprocess_seconds,
                            "postprocess_frames": int(len(arrays["valid"])),
                        },
                    }
                )
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                rows.append(
                    {
                        "case_id": case_id,
                        "sequence": record["sequence"],
                        "capture": record.get("capture"),
                        "angle_stratum": record["angle_stratum"],
                        "person_count": record.get("person_count_evaluator_only"),
                        "candidate": name,
                        "status": "error",
                        "error": message,
                    }
                )
                errors.append(f"{case_id}:{name}:{message}")
    by_case = {}
    for row in rows + references:
        by_case.setdefault(str(row["case_id"]), []).append(row)
    expected = len(candidates) + len(args.reference_methods)
    unavailable_message = "ValueError: No initial matched people for shared world fit"
    skipped = []
    for case_id, values in sorted(by_case.items()):
        if len(values) == expected and all(row.get("status") == "error" for row in values) and all(row.get("error") == unavailable_message for row in values):
            skipped.append(
                {
                    "case_id": case_id,
                    "status": "evaluator_unavailable",
                    "reason": unavailable_message,
                    "method_independent": True,
                    "evaluations_affected": expected,
                }
            )
    skipped_ids = {row["case_id"] for row in skipped}
    errors = [value for value in errors if value.split(":", 1)[0] not in skipped_ids]
    payload = {
        "schema_version": "Movie3R-v16-Harmony4D-causal-stabilization-probe-v1",
        "protocol_extension": "Movie3R-v19-EgoHumans-causal-identity-v1",
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
            "future_frames_at_boundary": 0,
            "source_cache_immutable": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    partial.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    partial.replace(args.output)
    write_csv(args.output.with_suffix(".csv"), rows)
    print(json.dumps({"output": str(args.output.resolve()), "cases": payload["case_count"], "candidates": len(candidates), "errors": len(errors)}, indent=2))


if __name__ == "__main__":
    main()
