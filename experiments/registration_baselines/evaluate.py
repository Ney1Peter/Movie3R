#!/usr/bin/env python3
"""GT-only evaluator for sealed traditional-registration predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.registration_baselines.prepare_inputs import value_sha256  # noqa: E402
from versions.v15.harmony4d import evaluate_harmony as base_evaluator  # noqa: E402
from versions.v19.egohumans.evaluate_egohumans import evaluate_method as egohumans_evaluate  # noqa: E402
from versions.v20.egobody.evaluate_egobody import evaluate_method as egobody_evaluate  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--evaluator-manifest", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("egobody", "egohumans"), required=True)
    parser.add_argument("--split", choices=("development", "holdout", "test"), required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--lines")
    parser.add_argument(
        "--prediction-seal", type=Path,
        help="Required for Test; generated before the evaluator manifest is opened.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(base_evaluator.jsonable(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(partial, path)


def load_gt(path: Path) -> tuple[dict[str, np.ndarray], list[str]]:
    with np.load(path, allow_pickle=False) as cache:
        identities = [str(value) for value in np.asarray(cache["identities"]).tolist()]
        gt = {key: np.asarray(cache[key]) for key in (
            "cameras_c2w", "vertices_world", "joints_world", "frames",
            "visible_fraction", "visible",
        )}
    return gt, identities


def evaluator_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "angle_stratum": row.get("angle_stratum_evaluator_only", row.get("angle_stratum")),
        "camera_rotation_span_deg": row.get("camera_rotation_span_deg_evaluator_only"),
        "camera_center_baseline_m": row.get("camera_center_baseline_m_evaluator_only"),
        "person_count": row.get("person_count_evaluator_only"),
        "recording": row.get("recording"),
        "capture": row.get("capture"),
        "sequence": row.get("sequence"),
    }


def frame_assignments(
    arrays: dict[str, np.ndarray], gt: dict[str, np.ndarray]
) -> tuple[list[list[tuple[int, int]]], list[float]]:
    """Reproduce the frozen evaluator's GT-only per-frame matching."""

    assignments: list[list[tuple[int, int]]] = []
    assignment_costs: list[float] = []
    for frame in range(len(gt["cameras_c2w"])):
        pred_valid = np.flatnonzero(arrays["valid"][frame].astype(bool))
        gt_valid = np.flatnonzero(gt["visible"][frame].astype(bool))
        pairs_local, costs = base_evaluator.frame_assignment(
            arrays["cameras_c2w"][frame], arrays["joints_world"][frame, pred_valid],
            gt["cameras_c2w"][frame], gt["joints_world"][frame, gt_valid],
        )
        accepted = [
            (row, column) for row, column in pairs_local
            if float(costs[row, column]) <= base_evaluator.MAX_ASSIGNMENT_COST_M
        ]
        assignments.append([
            (int(pred_valid[row]), int(gt_valid[column])) for row, column in accepted
        ])
        assignment_costs.extend(float(costs[row, column]) for row, column in accepted)
    return assignments, assignment_costs


def empty_summary() -> dict[str, Any]:
    return base_evaluator.summarize([])


def partial_evaluation(
    method: str,
    arrays: dict[str, np.ndarray],
    gt: dict[str, np.ndarray],
    identities: list[str],
    boundary: int,
    reason: str,
) -> dict[str, Any]:
    """Keep metrics independent of the unavailable initial world-frame fit.

    When the first two frames contain no accepted person match, W and all
    initial-world quantities are mathematically undefined.  The case remains
    in the fixed denominator; local pose, detection/association, and
    camera-only ATE are still evaluated.
    """

    assignments, assignment_costs = frame_assignments(arrays, gt)
    matched_count = int(sum(len(row) for row in assignments))
    local_joint: list[float] = []
    local_pa_joint: list[float] = []
    local_vertex: list[float] = []
    for frame, pairs in enumerate(assignments):
        pred_camera_joints = base_evaluator.camera_coordinates(
            arrays["cameras_c2w"][frame], arrays["joints_world"][frame]
        )
        pred_camera_vertices = base_evaluator.camera_coordinates(
            arrays["cameras_c2w"][frame], arrays["vertices_world"][frame]
        )
        gt_camera_joints = base_evaluator.camera_coordinates(
            gt["cameras_c2w"][frame], gt["joints_world"][frame]
        )
        gt_camera_vertices = base_evaluator.camera_coordinates(
            gt["cameras_c2w"][frame], gt["vertices_world"][frame]
        )
        for pred_index, gt_index in pairs:
            pred_pelvis = base_evaluator.pelvis(pred_camera_joints[pred_index])
            gt_pelvis = base_evaluator.pelvis(gt_camera_joints[gt_index])
            pred_body = pred_camera_joints[pred_index] - pred_pelvis
            gt_body = gt_camera_joints[gt_index] - gt_pelvis
            local_joint.append(float(np.linalg.norm(pred_body - gt_body, axis=1).mean()))
            local_pa_joint.append(base_evaluator.procrustes_mpjpe(gt_body, pred_body))
            local_vertex.append(float(np.linalg.norm(
                (pred_camera_vertices[pred_index] - pred_pelvis)
                - (gt_camera_vertices[gt_index] - gt_pelvis), axis=1,
            ).mean()))

    visible_gt = int(np.asarray(gt["visible"], dtype=bool).sum())
    predicted = int(np.asarray(arrays["valid"], dtype=bool).sum())
    coverage = {
        "visible_gt_person_frames": visible_gt,
        "matched_person_frames": matched_count,
        "missed_person_frames": visible_gt - matched_count,
        "predicted_person_frames": predicted,
        "false_positive_detections": max(predicted - matched_count, 0),
        "coverage": matched_count / max(visible_gt, 1),
        "precision": matched_count / max(predicted, 1),
        "recall": matched_count / max(visible_gt, 1),
        "minimum_visible_vertex_fraction": base_evaluator.MIN_VISIBLE_VERTEX_FRACTION,
        "maximum_assignment_cost_m": base_evaluator.MAX_ASSIGNMENT_COST_M,
    }
    fractions = np.asarray(gt["visible_fraction"], dtype=np.float64)
    visibility_strata = {}
    for name, mask in {
        "high_visibility": fractions >= 0.50,
        "partial_visibility": (fractions >= 0.10) & (fractions < 0.50),
        "severe_occlusion_or_truncation": (
            (fractions >= base_evaluator.MIN_VISIBLE_VERTEX_FRACTION) & (fractions < 0.10)
        ),
    }.items():
        total = int(mask.sum())
        matched = sum(
            bool(mask[frame, gt_index])
            for frame, pairs in enumerate(assignments) for _, gt_index in pairs
        )
        visibility_strata[name] = {
            "visible_gt_person_frames": total,
            "matched_person_frames": int(matched),
            "coverage": matched / max(total, 1),
        }
    coverage["visibility_strata"] = visibility_strata

    pred_centres = np.asarray(arrays["cameras_c2w"][:, :3, 3])
    gt_centres = np.asarray(gt["cameras_c2w"][:, :3, 3])
    sim3_fit = base_evaluator.fit_similarity(gt_centres, pred_centres, allow_scale=True)
    se3_fit = base_evaluator.fit_similarity(gt_centres, pred_centres, allow_scale=False)
    ate_sim3 = np.linalg.norm(
        base_evaluator.apply_similarity(pred_centres, sim3_fit) - gt_centres, axis=1
    )
    ate_se3 = np.linalg.norm(
        base_evaluator.apply_similarity(pred_centres, se3_fit) - gt_centres, axis=1
    )

    unavailable = empty_summary()
    named = {
        "w_mpjpe_mm": unavailable,
        "w_mpjpe_one_frame_fit_mm": unavailable,
        "wa_mpjpe_mm": unavailable,
        "mpjpe_mm": base_evaluator.summarize(local_joint, 1000.0),
        "pa_mpjpe_mm": base_evaluator.summarize(local_pa_joint, 1000.0),
        "mpvpe_mm": base_evaluator.summarize(local_vertex, 1000.0),
        "accel_delta2_mm_per_frame2": empty_summary(),
        "accel_physical_m_per_s2": empty_summary(),
        "rte_h3r_percent": empty_summary(),
        "roe_joint_proxy_deg": empty_summary(),
        "jitter_h3r_m_per_s3_div10": empty_summary(),
        "foot_sliding_cm": empty_summary(),
        "ate_sim3_m": base_evaluator.summarize(ate_sim3),
        "ate_se3_m": base_evaluator.summarize(ate_se3),
        "ate_metric_initial_se3_m": empty_summary(),
    }
    return {
        "method": method,
        "multi_thumbs_named_provisional": named,
        "coverage": coverage,
        "identity": base_evaluator.identity_metrics(
            arrays, assignments, identities, gt["visible"]
        ),
        "camera": {
            "translation_m": empty_summary(), "rotation_deg": empty_summary(),
            "first_post_translation_m": None, "first_post_rotation_deg": None,
            "post_translation_m": empty_summary(), "post_rotation_deg": empty_summary(),
            "rpe_translation_m": empty_summary(), "rpe_rotation_deg": empty_summary(),
            "boundary_rpe_translation_m": None, "boundary_rpe_rotation_deg": None,
        },
        "fixed_world": {
            "root_m": empty_summary(), "joint_m": empty_summary(), "vertex_m": empty_summary(),
            "first_post_root_m": empty_summary(), "first_post_joint_m": empty_summary(),
            "first_post_vertex_m": empty_summary(), "post_root_m": empty_summary(),
            "post_joint_m": empty_summary(), "post_vertex_m": empty_summary(),
        },
        "camera_human_relative": {
            "root_gauge_m": empty_summary(), "body_orientation_deg": empty_summary(),
            "first_post_root_gauge_m": empty_summary(), "post_root_gauge_m": empty_summary(),
        },
        "pairwise_layout": {
            "distance_m": empty_summary(), "vector_m": empty_summary(),
            "first_post_distance_m": empty_summary(), "first_post_vector_m": empty_summary(),
            "post_distance_m": empty_summary(), "post_vector_m": empty_summary(),
        },
        "cut_seam": {
            "available": False, "root_excess_m": None, "joint_excess_m": None,
            "vertex_excess_m": None, "camera_translation_excess_m": None,
            "camera_rotation_excess_deg": None, "camera_human_relative_excess_m": None,
        },
        "within_shot_motion": [],
        "assignment_cost": base_evaluator.summarize(assignment_costs),
        "shared_initial_sim3": None,
        "shared_initial_se3": None,
        "metric_availability": {
            "world_alignment": False,
            "local_pose": True,
            "coverage_and_identity": True,
            "camera_ate": True,
            "reason": reason,
            "initial_matched_person_frames": int(sum(len(row) for row in assignments[:2])),
            "case_retained_in_fixed_denominator": True,
        },
        "partial_evaluation": True,
        "boundary_index": int(boundary),
    }


def main() -> None:
    args = parse_args()
    seal = None
    sealed_by_path: dict[str, str] = {}
    if args.split == "test":
        if args.prediction_seal is None or not args.prediction_seal.is_file():
            raise ValueError("Frozen Test evaluation requires --prediction-seal")
        seal = json.loads(args.prediction_seal.resolve().read_text(encoding="utf-8"))
        if (
            seal.get("schema_version") != "Shot3R-traditional-registration-Test-prediction-seal-v1"
            or seal.get("dataset") != args.dataset
            or seal.get("split") != "test"
            or seal.get("sealed_before_evaluator") is not True
            or seal.get("evaluator_manifest_opened") is not False
        ):
            raise ValueError("invalid or post-evaluation Test prediction seal")
        sealed_by_path = {str(Path(row["path"]).resolve()): str(row["sha256"]) for row in seal["entries"]}
    runtime_rows = read_rows(args.runtime_manifest.resolve())
    evaluator_rows = read_rows(args.evaluator_manifest.resolve())
    evaluator_by_case = {str(row["case_id"]): row for row in evaluator_rows}
    if {str(row["case_id"]) for row in runtime_rows} != set(evaluator_by_case):
        raise ValueError("runtime/evaluator case mismatch")
    requested = {int(value) for value in args.lines.split(",") if value.strip()} if args.lines else None
    selected = [
        (index, row) for index, row in enumerate(runtime_rows, start=1)
        if (requested is None or index in requested)
        and (index - 1) % args.num_shards == args.shard_index
    ]
    completed, reused, failures = [], [], []
    evaluator_fn = egobody_evaluate if args.dataset == "egobody" else egohumans_evaluate
    for position, (line, record) in enumerate(selected, start=1):
        case_id = str(record["case_id"])
        evaluator_row = evaluator_by_case[case_id]
        expected_hash = evaluator_row.get("runtime_row_sha256")
        if expected_hash is not None and str(expected_hash) != value_sha256(record):
            raise ValueError(f"runtime/evaluator byte-contract mismatch: {case_id}")
        prediction = args.prediction_root / args.dataset / args.split / f"{case_id}.npz"
        runtime_report = prediction.with_suffix(".runtime.json")
        gt_path = args.gt_root / args.dataset / args.split / f"{case_id}.gt.npz"
        output = args.output_root / "metrics/evaluations" / args.dataset / args.split / f"{case_id}.evaluation.json"
        if output.is_file():
            try:
                existing = json.loads(output.read_text(encoding="utf-8"))
                if existing.get("case_id") == case_id and set(existing.get("methods", {})) == {"r0", "r1", "r2", "r3"}:
                    reused.append(case_id)
                    print(f"[{position}/{len(selected)}] reusable evaluation {case_id}", flush=True)
                    continue
            except Exception:
                pass
        try:
            if not prediction.is_file() or not runtime_report.is_file() or not gt_path.is_file():
                raise FileNotFoundError(next(path for path in (prediction, runtime_report, gt_path) if not path.is_file()))
            if args.split == "test":
                for sealed_path in (prediction.resolve(), runtime_report.resolve()):
                    expected = sealed_by_path.get(str(sealed_path))
                    if expected is None or sha256(sealed_path) != expected:
                        raise ValueError(f"prediction differs from pre-evaluation seal: {sealed_path}")
            runtime = json.loads(runtime_report.read_text(encoding="utf-8"))
            if runtime.get("case_id") != case_id or runtime.get("record") != record:
                raise ValueError("runtime report does not bind exact manifest row")
            gt, identities = load_gt(gt_path)
            results, errors = {}, {}
            with np.load(prediction, allow_pickle=False) as cache:
                for method in runtime["methods"]:
                    arrays = base_evaluator.method_arrays(cache, method)
                    try:
                        results[method] = evaluator_fn(
                            method, arrays, gt,
                            identities, int(record["boundary_index"]), float(record["fps"]),
                        )
                    except Exception as exc:
                        message = f"{type(exc).__name__}: {exc}"
                        if isinstance(exc, ValueError) and (
                            "No initial matched people" in str(exc)
                            or "Fewer than two valid pre-cut time points" in str(exc)
                        ):
                            results[method] = partial_evaluation(
                                method, arrays, gt, identities,
                                int(record["boundary_index"]), message,
                            )
                        else:
                            errors[method] = message
            transform_status = {}
            for method in runtime["methods"]:
                path = args.output_root / "transforms" / args.dataset / args.split / case_id / f"{method}.json"
                transform_status[method] = json.loads(path.read_text(encoding="utf-8"))
            payload = {
                "schema_version": "Shot3R-traditional-registration-evaluation-v1",
                "dataset": args.dataset, "split": args.split, "case_id": case_id,
                "record_runtime_fields": record,
                "evaluator_metadata": evaluator_metadata(evaluator_row),
                "identities": identities, "methods": results, "errors": errors,
                "registration": transform_status,
                "inputs": {
                    "prediction": str(prediction.resolve()),
                    "prediction_sha256": sha256(prediction),
                    "runtime_report": str(runtime_report.resolve()),
                    "runtime_report_sha256": sha256(runtime_report),
                    "gt_cache": str(gt_path.resolve()),
                    "evaluator_manifest": str(args.evaluator_manifest.resolve()),
                    "prediction_seal": None if args.prediction_seal is None else str(args.prediction_seal.resolve()),
                    "prediction_seal_sha256": None if args.prediction_seal is None else sha256(args.prediction_seal.resolve()),
                    "evaluator_manifest_opened_after_prediction_seal": args.split == "test" and seal is not None,
                },
                "gt_summary": {
                    "frame_count": int(len(gt["frames"])),
                    "visible_gt_person_frames": int(gt["visible"].sum()),
                },
                "evaluation_contract": {
                    "gt_used_only_in_evaluator": True,
                    "registration_used_gt": False,
                    "failure_fallback_retained": True,
                    "fixed_case_denominator": True,
                },
            }
            atomic_json(output, payload)
            completed.append(case_id)
            print(f"[{position}/{len(selected)}] evaluated {case_id} errors={errors}", flush=True)
        except Exception as exc:
            failures.append({"case_id": case_id, "line": line, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{position}/{len(selected)}] FAILED {failures[-1]}", flush=True)
    ledger = args.output_root / "logs" / f"evaluate_{args.dataset}_{args.split}_shard{args.shard_index:02d}.json"
    atomic_json(ledger, {
        "schema_version": "Shot3R-registration-evaluation-shard-ledger-v1",
        "dataset": args.dataset, "split": args.split,
        "selected_count": len(selected), "completed": completed,
        "reused": reused, "failures": failures,
    })
    if failures:
        raise SystemExit(f"{len(failures)} evaluations failed; see {ledger}")


if __name__ == "__main__":
    main()
