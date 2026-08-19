#!/usr/bin/env python3
"""Evaluate causal v16 candidates from immutable v15 B0 prediction caches.

Candidate geometry never reads GT.  Harmony4D annotations are loaded only
after a candidate has been materialized and are used by the frozen v15
evaluator for train/dev model selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from versions.v15.harmony4d.evaluate_harmony import (
    evaluate_method,
    load_gt,
    method_arrays,
)
from versions.v15.harmony4d.topology import CommonTopology
from versions.v16.harmony4d.causal_stabilization import (
    Candidate,
    apply_candidate,
    exploration_candidates,
)


SOURCE_METHOD = "m3_b0_only"
BASELINE = "v16_0_m15_geometry"
METRICS = {
    "W-MPJPE_mm": ("multi_thumbs_named_provisional", "w_mpjpe_mm", "mean"),
    "WA-MPJPE_mm": ("multi_thumbs_named_provisional", "wa_mpjpe_mm", "mean"),
    "MPJPE_mm": ("multi_thumbs_named_provisional", "mpjpe_mm", "mean"),
    "PA-MPJPE_mm": ("multi_thumbs_named_provisional", "pa_mpjpe_mm", "mean"),
    "MPVPE_mm": ("multi_thumbs_named_provisional", "mpvpe_mm", "mean"),
    "Accel_mm_frame2": ("multi_thumbs_named_provisional", "accel_delta2_mm_per_frame2", "mean"),
    "RTE_H3R_percent": ("multi_thumbs_named_provisional", "rte_h3r_percent", "mean"),
    "ROE_joint_proxy_deg": ("multi_thumbs_named_provisional", "roe_joint_proxy_deg", "mean"),
    "Jitter_H3R": ("multi_thumbs_named_provisional", "jitter_h3r_m_per_s3_div10", "mean"),
    "Foot_sliding_cm": ("multi_thumbs_named_provisional", "foot_sliding_cm", "mean"),
    "ATE_Sim3_m": ("multi_thumbs_named_provisional", "ate_sim3_m", "mean"),
    "ATE_SE3_m": ("multi_thumbs_named_provisional", "ate_se3_m", "mean"),
    "Boundary_camera_t_m": ("camera", "first_post_translation_m"),
    "Boundary_camera_R_deg": ("camera", "first_post_rotation_deg"),
    "Boundary_root_m": ("fixed_world", "first_post_root_m", "mean"),
    "Post_root_m": ("fixed_world", "post_root_m", "mean"),
    "Seam_camera_t_m": ("cut_seam", "camera_translation_excess_m"),
    "Seam_camera_R_deg": ("cut_seam", "camera_rotation_excess_deg"),
    "Seam_root_m": ("cut_seam", "root_excess_m"),
    "Seam_CHRGE_m": ("cut_seam", "camera_human_relative_excess_m"),
    "CHRGE_m": ("camera_human_relative", "root_gauge_m", "mean"),
    "Pair_vector_m": ("pairwise_layout", "vector_m", "mean"),
    "IDs": ("identity", "ids_total"),
    "IDF1": ("identity", "idf1"),
    "Coverage": ("coverage", "coverage"),
    "Detection_precision": ("coverage", "precision"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--extracted-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-json", type=Path)
    parser.add_argument("--source-method", default=SOURCE_METHOD)
    parser.add_argument(
        "--reference-methods", nargs="*", default=[],
        help="Cache methods evaluated unchanged under the same GT/evaluator pass.",
    )
    parser.add_argument(
        "--include-case-regex",
        help="Evaluate only case IDs matching this regular expression.",
    )
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
    if isinstance(value, Path):
        return str(value)
    return value


def nested(value: dict[str, Any], path: tuple[str, ...]) -> float | None:
    current: Any = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    if current is None:
        return None
    number = float(current)
    return number if math.isfinite(number) else None


def metric_row(result: dict[str, Any]) -> dict[str, float | None]:
    return {name: nested(result, path) for name, path in METRICS.items()}


def load_candidates(path: Path | None) -> list[Candidate]:
    if path is None:
        return exploration_candidates()
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["candidates"] if isinstance(payload, dict) else payload
    values = [Candidate(**row) for row in rows]
    if BASELINE not in {value.name for value in values}:
        values.insert(0, Candidate(BASELINE))
    return values


def runtime_paths(roots: list[Path]) -> list[Path]:
    output = []
    seen = set()
    for root in roots:
        for path in sorted(root.resolve().glob("*.runtime.json")):
            runtime = json.loads(path.read_text(encoding="utf-8"))
            case_id = str(runtime["record"]["case_id"])
            if case_id in seen:
                raise ValueError(f"duplicate case: {case_id}")
            seen.add(case_id)
            output.append(path)
    if not output:
        raise FileNotFoundError("no runtime reports")
    return output


def cache_for_runtime(path: Path, runtime: dict[str, Any]) -> Path:
    adjacent = path.with_name(path.name.removesuffix(".runtime.json") + ".npz")
    if adjacent.is_file():
        return adjacent
    cache = Path(runtime["cache"])
    if not cache.is_file():
        raise FileNotFoundError(cache)
    return cache


def mean(values: list[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(valid)) if valid else None


def aggregate(rows: list[dict[str, Any]], candidates: list[Candidate]) -> dict[str, Any]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["status"] == "complete":
            by_method[row["candidate"]].append(row)
    summary: dict[str, Any] = {}
    for candidate in candidates:
        values = by_method.get(candidate.name, [])
        metrics = {name: mean([row["metrics"].get(name) for row in values]) for name in METRICS}
        summary[candidate.name] = {
            "candidate": candidate.__dict__,
            "complete_cases": len(values),
            "metrics": metrics,
        }
    baseline = summary[BASELINE]["metrics"]
    for name, value in summary.items():
        metrics = value["metrics"]
        ratios = {}
        for metric in METRICS:
            first, second = baseline.get(metric), metrics.get(metric)
            ratios[metric] = (
                None if first is None or second is None or abs(first) < 1e-12
                else float(second / first)
            )
        case_rows = by_method.get(name, [])
        baseline_by_case = {
            row["case_id"]: row for row in by_method.get(BASELINE, [])
        }
        comparable = [
            row for row in case_rows
            if row["case_id"] in baseline_by_case
            and row["metrics"].get("W-MPJPE_mm") is not None
            and baseline_by_case[row["case_id"]]["metrics"].get("W-MPJPE_mm") is not None
        ]
        w_wins = sum(
            row["metrics"]["W-MPJPE_mm"]
            <= baseline_by_case[row["case_id"]]["metrics"]["W-MPJPE_mm"] + 1e-3
            for row in comparable
        )
        objective_terms = [ratios.get(key) for key in ("W-MPJPE_mm", "Accel_mm_frame2", "Seam_root_m")]
        objective_valid = [term for term in objective_terms if term is not None]
        promotion = bool(
            ratios.get("W-MPJPE_mm") is not None and ratios["W-MPJPE_mm"] <= 0.90
            and ratios.get("Accel_mm_frame2") is not None and ratios["Accel_mm_frame2"] <= 0.85
            and ratios.get("Seam_root_m") is not None and ratios["Seam_root_m"] <= 0.85
            and (ratios.get("MPJPE_mm") is None or ratios["MPJPE_mm"] <= 1.05)
            and (ratios.get("MPVPE_mm") is None or ratios["MPVPE_mm"] <= 1.05)
            and metrics.get("Coverage") == baseline.get("Coverage")
        )
        value.update({
            "ratio_to_baseline": ratios,
            "lower_is_better_objective": mean(objective_valid),
            "w_nonworse_cases": w_wins,
            "w_comparable_cases": len(comparable),
            "w_nonworse_rate": w_wins / max(len(comparable), 1),
            "passes_exploration_gate": promotion,
        })
    ranking = sorted(
        summary,
        key=lambda name: (
            summary[name]["lower_is_better_objective"] is None,
            summary[name]["lower_is_better_objective"] or float("inf"),
        ),
    )
    return {"baseline": BASELINE, "summary": summary, "ranking": ranking}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["case_id", "sequence", "angle_stratum", "candidate", "status", *METRICS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "case_id": row["case_id"],
                "sequence": row["sequence"],
                "angle_stratum": row["angle_stratum"],
                "candidate": row["candidate"],
                "status": row["status"],
                **row.get("metrics", {}),
            })


def main() -> None:
    args = parse_args()
    candidates = load_candidates(args.candidate_json)
    topology = CommonTopology.load()
    rows = []
    reference_rows = []
    errors = []
    for runtime_path in runtime_paths(args.prediction_roots):
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        record = runtime["record"]
        case_id = str(record["case_id"])
        if args.include_case_regex and not re.search(args.include_case_regex, case_id):
            continue
        cache_path = cache_for_runtime(runtime_path, runtime)
        with np.load(cache_path, allow_pickle=False) as cache:
            source = method_arrays(cache, args.source_method)
            references = {
                method: method_arrays(cache, method) for method in args.reference_methods
            }
        geometry = runtime.get("geometry", {})
        pairs = [tuple(map(int, pair)) for pair in geometry.get("association", {}).get("pairs", [])]
        gt, identities = load_gt(record, args.extracted_root, topology)
        for method, arrays in references.items():
            print(f">> {case_id} reference:{method}", flush=True)
            try:
                result = evaluate_method(
                    method, arrays, gt, identities,
                    int(record["boundary_index"]), float(record["fps"]),
                )
                reference_rows.append({
                    "case_id": case_id,
                    "sequence": record["sequence"],
                    "capture": record.get("capture", Path(record["capture_relative"]).name),
                    "angle_stratum": record["angle_stratum"],
                    "person_count": record.get("person_count_evaluator_only"),
                    "method": method,
                    "status": "complete",
                    "metrics": metric_row(result),
                    "within_shot_motion": result["within_shot_motion"],
                })
            except Exception as error:
                reference_rows.append({
                    "case_id": case_id,
                    "sequence": record["sequence"],
                    "capture": record.get("capture", Path(record["capture_relative"]).name),
                    "angle_stratum": record["angle_stratum"],
                    "person_count": record.get("person_count_evaluator_only"),
                    "method": method,
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                })
                errors.append(f"{case_id}:{method}:{type(error).__name__}:{error}")
        for candidate in candidates:
            print(f">> {case_id} {candidate.name}", flush=True)
            try:
                arrays, diagnostics = apply_candidate(
                    source, int(record["boundary_index"]), pairs, candidate
                )
                result = evaluate_method(
                    candidate.name, arrays, gt, identities,
                    int(record["boundary_index"]), float(record["fps"]),
                )
                rows.append({
                    "case_id": case_id,
                    "sequence": record["sequence"],
                    "capture": record.get("capture", Path(record["capture_relative"]).name),
                    "angle_stratum": record["angle_stratum"],
                    "person_count": record.get("person_count_evaluator_only"),
                    "candidate": candidate.name,
                    "status": "complete",
                    "metrics": metric_row(result),
                    "within_shot_motion": result["within_shot_motion"],
                    "diagnostics": diagnostics,
                })
            except Exception as error:  # keep the finite grid auditable
                rows.append({
                    "case_id": case_id,
                    "sequence": record["sequence"],
                    "capture": record.get("capture", Path(record["capture_relative"]).name),
                    "angle_stratum": record["angle_stratum"],
                    "person_count": record.get("person_count_evaluator_only"),
                    "candidate": candidate.name,
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                })
                errors.append(f"{case_id}:{candidate.name}:{type(error).__name__}:{error}")
    all_evaluations = rows + reference_rows
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_evaluations:
        by_case[str(row["case_id"])].append(row)
    expected_per_case = len(candidates) + len(args.reference_methods)
    evaluator_unavailable_message = "ValueError: No initial matched people for shared world fit"
    skipped_cases = []
    for case_id, values in sorted(by_case.items()):
        if (
            len(values) == expected_per_case
            and all(row.get("status") == "error" for row in values)
            and all(row.get("error") == evaluator_unavailable_message for row in values)
        ):
            skipped_cases.append({
                "case_id": case_id,
                "status": "evaluator_unavailable",
                "reason": evaluator_unavailable_message,
                "method_independent": True,
                "evaluations_affected": expected_per_case,
            })
    skipped_ids = {row["case_id"] for row in skipped_cases}
    errors = [
        value for value in errors
        if value.split(":", 1)[0] not in skipped_ids
    ]
    aggregation = aggregate(rows, candidates)
    payload = {
        "schema_version": "Movie3R-v16-Harmony4D-causal-stabilization-probe-v1",
        "source_method": args.source_method,
        "reference_methods": list(args.reference_methods),
        "prediction_roots": [str(path.resolve()) for path in args.prediction_roots],
        "extracted_root": str(args.extracted_root.resolve()),
        "candidate_source": str(args.candidate_json.resolve()) if args.candidate_json else "frozen_exploration_grid_in_code",
        "include_case_regex": args.include_case_regex,
        "candidate_count": len(candidates),
        "case_count": len({row["case_id"] for row in rows}),
        "complete_case_count": len({
            row["case_id"] for row in rows if row["status"] == "complete"
        }),
        "skipped_cases": skipped_cases,
        "errors": errors,
        "aggregate": aggregation,
        "rows": rows,
        "reference_rows": reference_rows,
        "contract": {
            "candidate_runtime_uses_gt": False,
            "gt_scope": "frozen evaluator only",
            "future_frames_at_boundary": 0,
            "source_cache_immutable": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(args.output.with_suffix(".csv"), rows)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "cases": payload["case_count"],
        "candidates": len(candidates),
        "errors": len(errors),
        "top10": aggregation["ranking"][:10],
        "passing": [
            name for name in aggregation["ranking"]
            if aggregation["summary"][name]["passes_exploration_gate"]
        ],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
