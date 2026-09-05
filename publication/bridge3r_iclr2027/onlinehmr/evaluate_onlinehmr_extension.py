#!/usr/bin/env python3
"""Convert and evaluate one OnlineHMR extension case after RGB-only inference."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT = Path(__file__).resolve()
WORKSPACE = SCRIPT.parents[4]
MOVIE_ROOT = WORKSPACE / "Movie3R"
CONVERTER = SCRIPT.with_name("convert_onlinehmr_result.py")
AIST_EVALUATOR = MOVIE_ROOT / "versions/v21/aist_singleperson/evaluate_aist.py"
AIST_MULTICUT_EVALUATOR = MOVIE_ROOT / "versions/v21/aist_singleperson/evaluate_aist_multicut.py"
MVHUMAN_EVALUATOR = MOVIE_ROOT / "versions/v25/mvhuman_heldout/evaluate_case.py"
METHOD = "onlinehmr_official"


def read_row(path: Path, line_number: int) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if line_number < 1 or line_number > len(rows):
        raise IndexError(line_number)
    return rows[line_number - 1]


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(partial, path)


def remove_inside(path: Path, parent: Path) -> int:
    if not path.exists():
        return 0
    resolved, root = path.resolve(), parent.resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"unsafe cleanup target: {resolved}")
    size = sum(item.stat().st_size for item in resolved.rglob("*") if item.is_file())
    shutil.rmtree(resolved)
    return int(size)


def failure_payload(
    row: dict[str, Any],
    raw: dict[str, Any],
    *,
    status: str = "inference_failure",
    reason: str | None = None,
) -> dict[str, Any]:
    value = {
        "method": METHOD,
        "status": status,
        "failure_reason": reason or raw.get("failure_reason") or "OnlineHMR inference failure",
        "coverage": {
            "valid_frames": 0,
            "total_frames": int(row["clip_length"]),
            "valid_frame_coverage": 0.0,
            "completion": 0.0,
        },
        "metrics": {},
    }
    return {
        "schema_version": "Bridge3R-OnlineHMR-extension-evaluation-v1",
        "protocol": row["protocol"],
        "case_id": row["case_id"],
        "methods": {METHOD: value},
        "errors": {METHOD: value["failure_reason"]},
        "evaluation_contract": {
            "runtime_gt_access": False,
            "whole_case_failure_counts_as_zero_coverage": True,
            "fixed_denominator": True,
        },
    }


def evaluate_harmony(
    prediction: Path,
    evaluator: dict[str, Any],
    source_root: Path,
    output: Path,
) -> None:
    if str(MOVIE_ROOT) not in sys.path:
        sys.path.insert(0, str(MOVIE_ROOT))
    from publication.bridge3r_iclr2027.evaluate_harmony4d_multicut import (  # type: ignore
        CommonTopology,
        evaluate_method,
        jsonable,
        load_gt,
        method_arrays,
    )
    from versions.v15.harmony4d import evaluate_harmony as frozen  # type: ignore

    boundaries = [int(value) for value in evaluator["boundaries"]]
    topology = CommonTopology.load()
    extracted = source_root / f"train_{evaluator['sequence']}"
    gt, identities = load_gt(evaluator, extracted, topology)
    with np.load(prediction, allow_pickle=False) as cache:
        arrays = method_arrays(cache, METHOD)
        try:
            aggregate = evaluate_method(
                METHOD, arrays, gt, identities, boundaries[0], float(evaluator["fps"])
            )
            seams = {}
            for boundary in boundaries:
                boundary_result = evaluate_method(
                    METHOD, arrays, gt, identities, boundary, float(evaluator["fps"])
                )
                seams[str(boundary)] = {
                    "cut_seam": boundary_result["cut_seam"],
                    "boundary_camera_rpe_translation_m": boundary_result["camera"]["boundary_rpe_translation_m"],
                    "boundary_camera_rpe_rotation_deg": boundary_result["camera"]["boundary_rpe_rotation_deg"],
                }
            aggregate["cut_seams"] = seams
        except ValueError as error:
            if "No initial matched people for shared world fit" not in str(error):
                raise
            # Preserve a valid fixed-denominator record when OnlineHMR has no
            # accepted person in the two immutable initial frames.  W/WA and
            # human seam errors are genuinely unavailable, while coverage,
            # IDF1 and camera-centre ATE remain well-defined under the same
            # frozen matching threshold.
            assignments = []
            matched_count = 0
            for frame in range(len(gt["cameras_c2w"])):
                valid = np.flatnonzero(arrays["valid"][frame].astype(bool))
                gt_valid = np.flatnonzero(gt["visible"][frame].astype(bool))
                local, costs = frozen.frame_assignment(
                    arrays["cameras_c2w"][frame], arrays["joints_world"][frame, valid],
                    gt["cameras_c2w"][frame], gt["joints_world"][frame, gt_valid],
                )
                local = [
                    (row, column) for row, column in local
                    if float(costs[row, column]) <= frozen.MAX_ASSIGNMENT_COST_M
                ]
                pairs = [(int(valid[row]), int(gt_valid[column])) for row, column in local]
                assignments.append(pairs)
                matched_count += len(pairs)
            visible = int(np.asarray(gt["visible"], dtype=bool).sum())
            predicted = int(np.asarray(arrays["valid"], dtype=bool).sum())
            centres = np.asarray(arrays["cameras_c2w"][:, :3, 3], dtype=np.float64)
            target_centres = np.asarray(gt["cameras_c2w"][:, :3, 3], dtype=np.float64)
            camera_fit = frozen.fit_similarity(target_centres, centres)
            ate = np.linalg.norm(frozen.apply_similarity(centres, camera_fit) - target_centres, axis=1)
            seams = {}
            for boundary in boundaries:
                predicted_relative = np.linalg.inv(arrays["cameras_c2w"][boundary - 1]) @ arrays["cameras_c2w"][boundary]
                gt_relative = np.linalg.inv(gt["cameras_c2w"][boundary - 1]) @ gt["cameras_c2w"][boundary]
                seams[str(boundary)] = {
                    "cut_seam": {
                        "available": True,
                        "root_excess_m": None,
                        "joint_excess_m": None,
                        "vertex_excess_m": None,
                        "camera_translation_excess_m": None,
                        "camera_rotation_excess_deg": frozen.rotation_error_deg(predicted_relative, gt_relative),
                    },
                    "boundary_camera_rpe_translation_m": None,
                    "boundary_camera_rpe_rotation_deg": frozen.rotation_error_deg(predicted_relative, gt_relative),
                }
            aggregate = {
                "method": METHOD,
                "status": "evaluator_unavailable_insufficient_initial_match",
                "failure_reason": str(error),
                "multi_thumbs_named_provisional": {
                    "w_mpjpe_mm": frozen.summarize([]),
                    "wa_mpjpe_mm": frozen.summarize([]),
                    "ate_sim3_m": frozen.summarize(ate),
                },
                "coverage": {
                    "visible_gt_person_frames": visible,
                    "matched_person_frames": matched_count,
                    "missed_person_frames": visible - matched_count,
                    "predicted_person_frames": predicted,
                    "false_positive_detections": max(predicted - matched_count, 0),
                    "coverage": matched_count / max(visible, 1),
                    "precision": matched_count / max(predicted, 1),
                    "recall": matched_count / max(visible, 1),
                    "minimum_visible_vertex_fraction": frozen.MIN_VISIBLE_VERTEX_FRACTION,
                    "maximum_assignment_cost_m": frozen.MAX_ASSIGNMENT_COST_M,
                },
                "identity": frozen.identity_metrics(arrays, assignments, identities, gt["visible"]),
                "cut_seams": seams,
            }
    atomic_json(output, jsonable({
        "schema_version": "Bridge3R-OnlineHMR-Harmony4D-multicut-evaluation-v1",
        "protocol": evaluator["protocol"],
        "case_id": evaluator["case_id"],
        "record": evaluator,
        "identities": identities,
        "methods": {METHOD: aggregate},
        "evaluation_contract": {
            "gt_used_only_after_inference": True,
            "runtime_gt_access": False,
            "matching": "per-frame Hungarian in camera coordinates; GT identity never enters runtime",
            "aggregation_scope": "all 150 frames; both fixed RGB boundaries are evaluated",
            "common_topology": topology.metadata(),
        },
    }))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--evaluator-manifest", type=Path, required=True)
    parser.add_argument("--line", type=int, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--adapter-python", type=Path, required=True)
    parser.add_argument("--evaluator-python", type=Path, required=True)
    parser.add_argument("--mvhuman-audit-root", type=Path)
    args = parser.parse_args()
    runtime_manifest = args.runtime_manifest.resolve()
    evaluator_manifest = args.evaluator_manifest.resolve()
    row = read_row(runtime_manifest, int(args.line))
    evaluator = read_row(evaluator_manifest, int(args.line))
    if row["case_id"] != evaluator["case_id"]:
        raise ValueError("runtime/evaluator case mismatch")
    root = args.run_root.resolve() / f"line{int(args.line):03d}"
    raw_path = root / "onlinehmr.runtime.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    if raw.get("case_id") != row["case_id"] or raw.get("runtime_gt_access") is not False:
        raise ValueError("inference provenance mismatch")
    prediction = root / "prediction.npz"
    metadata = root / "prediction.json"
    eval_runtime = root / "onlinehmr.eval_runtime.json"
    evaluation = root / "onlinehmr.evaluation.json"
    if evaluation.is_file():
        print(json.dumps({"status": "reused", "case_id": row["case_id"], "evaluation": str(evaluation)}))
        return
    if raw.get("status") != "success":
        atomic_json(evaluation, failure_payload(row, raw))
        print(json.dumps({"status": "recorded_failure", "case_id": row["case_id"]}))
        return
    command = [
        os.path.abspath(args.adapter_python), str(CONVERTER),
        "--native-root", str(Path(raw["native_root"]).resolve()),
        "--camera-trajectory", str(Path(raw["camera_trajectory"]).resolve()),
        "--manifest", str(runtime_manifest), "--line", str(int(args.line)),
        "--output", str(prediction), "--metadata-output", str(metadata),
        "--method", METHOD, "--device", "cpu",
    ]
    if row["dataset"] != "Harmony4D":
        command.append("--omit-vertices")
    try:
        subprocess.run(command, cwd=WORKSPACE, check=True)
    except Exception as error:
        reason = f"prediction_conversion_failed: {type(error).__name__}: {error}"
        atomic_json(
            evaluation,
            failure_payload(row, raw, status="invalid_output", reason=reason),
        )
        native = root / "native"
        removed = remove_inside(native, root)
        atomic_json(root / "post_evaluation_cleanup.json", {
            "schema_version": "Bridge3R-OnlineHMR-extension-cleanup-v1",
            "case_id": row["case_id"],
            "removed_reproducible_native_bytes": removed,
            "prediction_retained": None,
            "evaluation_retained": str(evaluation),
            "conversion_failure": reason,
        })
        print(json.dumps({"status": "recorded_invalid_output", "case_id": row["case_id"], "reason": reason}))
        return
    atomic_json(eval_runtime, {
        "schema_version": "Bridge3R-OnlineHMR-extension-evaluator-runtime-v1",
        "case_id": row["case_id"],
        "record": row,
        "methods": [METHOD],
        "manifest_line": int(args.line),
        "runtime_gt_access": False,
        "gt_used_only_after_inference": True,
    })
    source_root = args.source_root.resolve()
    protocol = str(row["protocol"])
    if protocol == "CS150":
        script = AIST_EVALUATOR
        command = [
            os.path.abspath(args.evaluator_python), str(script),
            "--cache", str(prediction), "--runtime-report", str(eval_runtime),
            "--label", str(source_root / str(evaluator["label"])),
            "--output", str(evaluation),
        ]
        subprocess.run(command, cwd=WORKSPACE, check=True)
    elif protocol in {"MC150-3", "MC150-4"}:
        command = [
            os.path.abspath(args.evaluator_python), str(AIST_MULTICUT_EVALUATOR),
            "--cache", str(prediction), "--runtime-report", str(eval_runtime),
            "--label", str(source_root / str(evaluator["label"])),
            "--output", str(evaluation),
        ]
        subprocess.run(command, cwd=WORKSPACE, check=True)
    elif protocol == "MVH150":
        if args.mvhuman_audit_root is None:
            raise ValueError("--mvhuman-audit-root is required for MVH150")
        command = [
            os.path.abspath(args.evaluator_python), str(MVHUMAN_EVALUATOR),
            "--cache", str(prediction), "--runtime-report", str(eval_runtime),
            "--evaluator-manifest", str(evaluator_manifest),
            "--case-id", str(row["case_id"]),
            "--audit-root", str(args.mvhuman_audit_root.resolve()),
            "--output", str(evaluation),
        ]
        subprocess.run(command, cwd=WORKSPACE, check=True)
    elif protocol == "Bridge3R-Harmony4D-MultiCut-v1":
        evaluate_harmony(prediction, evaluator, source_root, evaluation)
    else:
        raise ValueError(f"unsupported extension protocol: {protocol}")
    native = root / "native"
    removed = remove_inside(native, root)
    atomic_json(root / "post_evaluation_cleanup.json", {
        "schema_version": "Bridge3R-OnlineHMR-extension-cleanup-v1",
        "case_id": row["case_id"],
        "removed_reproducible_native_bytes": removed,
        "prediction_retained": str(prediction),
        "evaluation_retained": str(evaluation),
    })
    print(json.dumps({"status": "evaluated", "case_id": row["case_id"], "evaluation": str(evaluation), "removed_bytes": removed}))


if __name__ == "__main__":
    main()
