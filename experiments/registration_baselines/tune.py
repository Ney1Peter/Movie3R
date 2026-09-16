#!/usr/bin/env python3
"""Select one R1/R2/R3 configuration per dataset on frozen Development data.

The program is intentionally Development-only.  It evaluates every explicit
candidate on every available Development case, stores the full candidate
ledger, and chooses a single dataset-level setting.  It never performs
per-case candidate selection.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.registration_baselines import evaluate, fit  # noqa: E402
from experiments.registration_baselines.geometry import (  # noqa: E402
    RegistrationResult,
    apply_shared_transform,
    human_joint_se3,
    pelvis_translation,
    rotation_degrees,
    scene_fpfh_icp,
)
from experiments.registration_baselines.prepare_inputs import value_sha256  # noqa: E402


SCHEMA = "Shot3R-traditional-registration-development-tuning-v1"
METHODS = ("r1", "r2", "r3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--evaluator-manifest", type=Path, required=True)
    parser.add_argument("--r0-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--candidate-grid", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("egobody", "egohumans"), required=True)
    parser.add_argument("--max-workers", type=int, default=12)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def finite(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def metric_mean(section: dict[str, Any], key: str) -> float | None:
    value = section.get(key, {}).get("mean")
    return finite(value)


def compact_metrics(result: dict[str, Any], dataset: str) -> dict[str, Any]:
    named = result["multi_thumbs_named_provisional"]
    camera = result.get("camera", {})
    seam = result.get("cut_seam", {})
    identity = result.get("identity", {})
    coverage = result.get("coverage", {})
    return {
        "w_mpjpe_mm": metric_mean(named, "w_mpjpe_mm"),
        "wa_mpjpe_mm": metric_mean(named, "wa_mpjpe_mm"),
        "ate_m": metric_mean(named, "ate_sim3_m" if dataset == "egobody" else "ate_se3_m"),
        "ate_sim3_m": metric_mean(named, "ate_sim3_m"),
        "ate_se3_m": metric_mean(named, "ate_se3_m"),
        "mpjpe_mm": metric_mean(named, "mpjpe_mm"),
        "mpvpe_mm": metric_mean(named, "mpvpe_mm"),
        "boundary_camera_translation_m": finite(camera.get("boundary_rpe_translation_m")),
        "boundary_camera_rotation_deg": finite(camera.get("boundary_rpe_rotation_deg")),
        "seam_root_m": finite(seam.get("root_excess_m")),
        "idf1": finite(identity.get("idf1")),
        "coverage": finite(coverage.get("coverage")),
        "matched_person_frames": int(coverage.get("matched_person_frames", 0)),
        "visible_gt_person_frames": int(coverage.get("visible_gt_person_frames", 0)),
    }


def estimate(
    method: str,
    arrays: dict[str, np.ndarray],
    scene: dict[str, np.ndarray],
    boundary: int,
    candidate: dict[str, Any],
    seed: int,
) -> RegistrationResult:
    if method == "r1":
        return pelvis_translation(arrays, boundary, float(candidate["max_pair_cost"]))
    if method == "r2":
        return human_joint_se3(
            arrays,
            boundary,
            float(candidate["max_pair_cost"]),
            float(candidate["inlier_threshold_m"]),
            int(candidate["ransac_trials"]),
            float(candidate["min_inlier_ratio"]),
            seed,
        )
    if method == "r3":
        return scene_fpfh_icp(
            scene["post"],
            scene["pre"],
            float(candidate["voxel_size_m"]),
            float(candidate["ransac_distance_factor"]),
            float(candidate["icp_distance_factor"]),
            int(candidate["min_scene_points"]),
            int(candidate["min_correspondences"]),
            float(candidate["min_fitness"]),
            float(candidate["max_inlier_rmse_m"]),
            seed,
        )
    raise ValueError(method)


def evaluate_arrays(
    method: str,
    arrays: dict[str, np.ndarray],
    gt: dict[str, np.ndarray],
    identities: list[str],
    record: dict[str, Any],
    dataset: str,
) -> tuple[dict[str, Any] | None, str | None]:
    evaluator_fn = evaluate.egobody_evaluate if dataset == "egobody" else evaluate.egohumans_evaluate
    try:
        result = evaluator_fn(
            method,
            arrays,
            gt,
            identities,
            int(record["boundary_index"]),
            float(record["fps"]),
        )
        return compact_metrics(result, dataset), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def process_case(task: dict[str, Any]) -> dict[str, Any]:
    # Keep OpenMP/OpenBLAS/Open3D from multiplying threads inside each worker.
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    record = task["record"]
    dataset = str(task["dataset"])
    case_id = str(record["case_id"])
    arrays, scene = fit.load_r0(Path(task["r0_path"]))
    gt, identities = evaluate.load_gt(Path(task["gt_path"]))
    rows: list[dict[str, Any]] = []
    r0_metrics, r0_error = evaluate_arrays("r0", arrays, gt, identities, record, dataset)
    rows.append({
        "case_id": case_id,
        "method": "r0",
        "candidate_id": "r0_identity",
        "candidate": {},
        "status": "ok",
        "failure_reason": None,
        "fallback_to_r0": False,
        "runtime_seconds": 0.0,
        "estimated_translation_m": 0.0,
        "estimated_rotation_deg": 0.0,
        "metrics": r0_metrics,
        "evaluation_error": r0_error,
    })
    boundary = int(record["boundary_index"])
    seed = int(task["seed"])
    for method in METHODS:
        for candidate in task["grid"][method]:
            candidate_id = str(candidate["candidate_id"])
            result = estimate(method, arrays, scene, boundary, candidate, seed)
            mapped = apply_shared_transform(arrays, result.transform, boundary, result.id_map)
            metrics, error = evaluate_arrays(
                candidate_id, mapped, gt, identities, record, dataset
            )
            rows.append({
                "case_id": case_id,
                "method": method,
                "candidate_id": candidate_id,
                "candidate": candidate,
                "status": result.status,
                "failure_reason": result.failure_reason,
                "fallback_to_r0": result.status != "ok",
                "runtime_seconds": finite(result.diagnostics.get("runtime_seconds")),
                "estimated_translation_m": float(np.linalg.norm(result.transform[:3, 3])),
                "estimated_rotation_deg": rotation_degrees(result.transform),
                "diagnostics": result.diagnostics,
                "metrics": metrics,
                "evaluation_error": error,
            })
    metadata = task["metadata"]
    return {
        "schema_version": "Shot3R-registration-development-case-candidates-v1",
        "dataset": dataset,
        "split": "development",
        "case_id": case_id,
        "record": record,
        "evaluator_metadata": {
            "angle_stratum": metadata.get("angle_stratum_evaluator_only", metadata.get("angle_stratum")),
            "camera_rotation_span_deg": metadata.get("camera_rotation_span_deg_evaluator_only"),
            "person_count": metadata.get("person_count_evaluator_only"),
        },
        "grid_sha256": task["grid_sha256"],
        "rows": rows,
    }


def macro(rows: list[dict[str, Any]], metric: str, unit_by_case: dict[str, str]) -> dict[str, Any]:
    by_unit: dict[str, list[float]] = defaultdict(list)
    valid_cases = 0
    for row in rows:
        metrics = row.get("metrics")
        value = None if metrics is None else finite(metrics.get(metric))
        if value is None:
            continue
        valid_cases += 1
        by_unit[unit_by_case[str(row["case_id"])]].append(value)
    unit_values = [float(np.mean(values)) for values in by_unit.values() if values]
    return {
        "mean": float(np.mean(unit_values)) if unit_values else None,
        "valid_case_count": valid_cases,
        "valid_unit_count": len(unit_values),
    }


def aggregate(
    cases: list[dict[str, Any]], dataset: str, grid: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = [row for case in cases for row in case["rows"]]
    unit_by_case = {}
    for case in cases:
        record = case["record"]
        unit = str(record["recording"] if dataset == "egobody" else record["capture"])
        unit_by_case[str(case["case_id"])] = unit
    expected_cases, expected_units = len(cases), len(set(unit_by_case.values()))
    candidates = {"r0_identity": ("r0", {})}
    for method in METHODS:
        candidates.update({str(value["candidate_id"]): (method, value) for value in grid[method]})
    summaries: dict[str, Any] = {}
    for candidate_id, (method, candidate) in candidates.items():
        selected = [row for row in rows if row["candidate_id"] == candidate_id]
        statuses: dict[str, int] = defaultdict(int)
        for row in selected:
            statuses[str(row["status"])] += 1
        summaries[candidate_id] = {
            "method": method,
            "candidate": candidate,
            "case_count": len(selected),
            "expected_case_count": expected_cases,
            "unit_count": expected_units,
            "w_mpjpe_mm": macro(selected, "w_mpjpe_mm", unit_by_case),
            "wa_mpjpe_mm": macro(selected, "wa_mpjpe_mm", unit_by_case),
            "ate_m": macro(selected, "ate_m", unit_by_case),
            "idf1": macro(selected, "idf1", unit_by_case),
            "coverage": macro(selected, "coverage", unit_by_case),
            "status_counts": dict(sorted(statuses.items())),
            "algorithmic_failure_count": sum(row["status"] != "ok" for row in selected),
            "algorithmic_failure_rate": sum(row["status"] != "ok" for row in selected) / max(len(selected), 1),
            "evaluation_error_count": sum(row.get("evaluation_error") is not None for row in selected),
            "mean_fit_runtime_seconds": float(np.mean([
                row["runtime_seconds"] for row in selected if row.get("runtime_seconds") is not None
            ])) if selected else None,
        }
    chosen = {}
    rankings = {}
    for method in METHODS:
        values = [value for value in summaries.values() if value["method"] == method]
        max_valid = max(value["w_mpjpe_mm"]["valid_case_count"] for value in values)
        eligible = [value for value in values if value["w_mpjpe_mm"]["valid_case_count"] == max_valid]
        eligible.sort(key=lambda value: (
            float("inf") if value["w_mpjpe_mm"]["mean"] is None else value["w_mpjpe_mm"]["mean"],
            float("inf") if value["ate_m"]["mean"] is None else value["ate_m"]["mean"],
            value["algorithmic_failure_rate"],
            str(value["candidate"]["candidate_id"]),
        ))
        ranking = [str(value["candidate"]["candidate_id"]) for value in eligible]
        rankings[method] = {
            "maximum_common_valid_case_count": max_valid,
            "ranking": ranking,
            "rule": "maximum valid support, then dataset macro W-MPJPE, ATE, algorithmic failure rate, candidate ID",
        }
        chosen[method] = eligible[0]
    return summaries, {"chosen": chosen, "rankings": rankings}


def main() -> None:
    args = parse_args()
    runtime_manifest = args.runtime_manifest.resolve()
    evaluator_manifest = args.evaluator_manifest.resolve()
    grid_path = args.candidate_grid.resolve()
    runtime_rows = read_rows(runtime_manifest)
    evaluator_rows = read_rows(evaluator_manifest)
    if not runtime_rows or any(str(row.get("split")) != "development" for row in runtime_rows):
        raise ValueError("tune.py accepts complete Development manifests only")
    runtime_by_case = {str(row["case_id"]): row for row in runtime_rows}
    evaluator_by_case = {str(row["case_id"]): row for row in evaluator_rows}
    if set(runtime_by_case) != set(evaluator_by_case):
        raise ValueError("runtime/evaluator case mismatch")
    for case_id, metadata in evaluator_by_case.items():
        expected = metadata.get("runtime_row_sha256")
        if expected is not None and str(expected) != value_sha256(runtime_by_case[case_id]):
            raise ValueError(f"runtime byte-contract mismatch: {case_id}")
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    if grid.get("test_access") is not False or grid.get("frozen_before_development_evaluation") is not True:
        raise ValueError("candidate grid is not declared Development-frozen")
    for method in METHODS:
        if not isinstance(grid.get(method), list) or not grid[method]:
            raise ValueError(f"candidate grid misses explicit {method} list")
    grid_hash = sha256(grid_path)
    case_root = args.output_root / "metrics/development_tuning" / args.dataset / "cases"
    tasks, cases = [], []
    for record in runtime_rows:
        case_id = str(record["case_id"])
        output = case_root / f"{case_id}.json"
        if output.is_file():
            try:
                payload = json.loads(output.read_text(encoding="utf-8"))
                if payload.get("case_id") == case_id and payload.get("grid_sha256") == grid_hash:
                    cases.append(payload)
                    continue
            except Exception:
                pass
        r0_path = args.r0_root / args.dataset / "development" / f"{case_id}.r0_scene.npz"
        gt_path = args.gt_root / args.dataset / "development" / f"{case_id}.gt.npz"
        if not r0_path.is_file() or not gt_path.is_file():
            raise FileNotFoundError(r0_path if not r0_path.is_file() else gt_path)
        tasks.append({
            "record": record,
            "metadata": evaluator_by_case[case_id],
            "dataset": args.dataset,
            "r0_path": str(r0_path),
            "gt_path": str(gt_path),
            "grid": {method: grid[method] for method in METHODS},
            "grid_sha256": grid_hash,
            "seed": int(grid["random_seed"]),
        })
    started = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, int(args.max_workers))) as pool:
        futures = {pool.submit(process_case, task): task for task in tasks}
        for position, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            task = futures[future]
            case_id = str(task["record"]["case_id"])
            try:
                payload = future.result()
            except Exception as exc:
                raise RuntimeError(f"Development tuning case failed: {case_id}") from exc
            atomic_json(case_root / f"{case_id}.json", payload)
            cases.append(payload)
            print(f"[{position}/{len(tasks)} new; {len(cases)}/{len(runtime_rows)} total] tuned {case_id}", flush=True)
    cases.sort(key=lambda value: str(value["case_id"]))
    if len(cases) != len(runtime_rows):
        raise ValueError(f"incomplete Development tuning cases: {len(cases)}/{len(runtime_rows)}")
    summaries, selection = aggregate(cases, args.dataset, grid)
    selected_config = {
        "schema_version": "Shot3R-registration-config-v1",
        "configuration_id": f"development-frozen-{args.dataset}-{grid_hash[:12]}",
        "dataset": args.dataset,
        "random_seed": int(grid["random_seed"]),
        "r1": {key: value for key, value in selection["chosen"]["r1"]["candidate"].items() if key != "candidate_id"},
        "r2": {key: value for key, value in selection["chosen"]["r2"]["candidate"].items() if key != "candidate_id"},
        "r3": {key: value for key, value in selection["chosen"]["r3"]["candidate"].items() if key != "candidate_id"},
        "selected_candidate_ids": {
            method: selection["chosen"][method]["candidate"]["candidate_id"] for method in METHODS
        },
        "selection_split": "development",
        "selection_is_dataset_level": True,
        "per_case_oracle": False,
        "candidate_grid": str(grid_path),
        "candidate_grid_sha256": grid_hash,
        "status": "frozen for one Holdout gate and subsequent Test runtime",
    }
    output_dir = args.output_root / "metrics/development_tuning" / args.dataset
    report = {
        "schema_version": SCHEMA,
        "dataset": args.dataset,
        "split": "development",
        "runtime_manifest": str(runtime_manifest),
        "runtime_manifest_sha256": sha256(runtime_manifest),
        "evaluator_manifest": str(evaluator_manifest),
        "evaluator_manifest_sha256": sha256(evaluator_manifest),
        "candidate_grid": str(grid_path),
        "candidate_grid_sha256": grid_hash,
        "case_count": len(cases),
        "aggregation_unit": "recording" if args.dataset == "egobody" else "capture",
        "candidate_summaries": summaries,
        "selection": selection,
        "frozen_config": selected_config,
        "elapsed_seconds": time.time() - started,
        "test_access": False,
        "per_case_selection": False,
    }
    atomic_json(output_dir / "summary.json", report)
    atomic_json(args.output_root / "config" / f"frozen_{args.dataset}.json", selected_config)
    print(json.dumps({
        "dataset": args.dataset,
        "cases": len(cases),
        "selected_candidate_ids": selected_config["selected_candidate_ids"],
        "config": str((args.output_root / "config" / f"frozen_{args.dataset}.json").resolve()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
