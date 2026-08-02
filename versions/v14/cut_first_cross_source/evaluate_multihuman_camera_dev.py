#!/usr/bin/env python3
"""Evaluate one B0 checkpoint on the frozen MultiHuman camera-only dev split.

The P0 intervention trains a *single* cut-first B0 on full-frame MultiHuman
camera supervision.  This evaluator deliberately keeps its decision path
strictly causal:

    shadow(A(t-1), A(t), B(t))  -> predicted post camera
    raw(B(t))                   -> clean recurrent state
    B0 = C_shadow_post @ inv(C_raw_post)
    deployed post camera = B0 @ C_raw_post

Official calibration and SMPL-X meshes are loaded only after those three
predictions exist.  Calibration defines the reporting gauge; meshes give a
fixed-human audit, including root/joint/vertex and layout errors.  They do
not enter B0 construction, routing, or the model forward.

The split comes from ``v14_multihuman_camera_supervision_20260803.json``:
36 opposite-camera events on camera pairs disjoint from P0 training.  It is
development data (the ``three`` sequence was opened before), never a final
confirmation benchmark.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from dust3r.model import ARCroco3DStereo  # noqa: E402
from dust3r.utils.smpl_layer import SMPL_Layer  # noqa: E402
from dust3r.v14_outputs import boundary_from_camera_predictions  # noqa: E402
from versions.v13 import gt_id_consensus as gt  # noqa: E402
from versions.v14.run_v14_2_multihuman_sequence import assign_prediction_ids  # noqa: E402
from versions.v14.run_v14_2_single_sequence import (  # noqa: E402
    boundary_error,
    camera_matrix,
    configure_model,
    set_event_indices,
)


DEFAULT_MANIFEST = REPO_ROOT / "config/manifests/v14_multihuman_camera_supervision_20260803.json"
DEFAULT_OUTPUT = REPO_ROOT / "output/v14_cut_first_cross_source/eval_multihuman_camera_dev"
DEFAULT_DATA = Path("/data/wangzheng/iJCV-CODE/data/MultiHuman/Real-World-Capture/extracted")
METHODS = ("raw_reset", "b0_runtime")
METRICS = (
    "camera_translation_error_m",
    "camera_rotation_error_deg",
    "camera_composite",
    "human_root_error_m",
    "human_joint_error_m",
    "human_vertex_error_m",
    "pairwise_distance_error_m",
    "pairwise_vector_error_m",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frame-cache-dir", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    return gt.jsonable(value)


def finite_summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0, "mean": float("nan"), "median": float("nan"), "p90": float("nan"), "p95": float("nan")}
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
    }


def summarize(cases: list[dict]) -> dict:
    output: dict[str, Any] = {"case_count": len(cases), "methods": {}}
    for method in METHODS:
        rows = [case["methods"][method] for case in cases]
        output["methods"][method] = {
            **{metric: finite_summary([row[metric] for row in rows]) for metric in METRICS},
            "catastrophic_count": int(sum(bool(row["catastrophic"]) for row in rows)),
            "human_assignment_count": int(sum(int(row["human_assignment_count"]) for row in rows)),
            "human_assignment_coverage": float(
                np.mean([float(row["human_assignment_count"]) / 3.0 for row in rows])
            ) if rows else float("nan"),
        }
    output["camera_parity"] = {
        "matrix_max_abs": finite_summary([case["b0_camera_parity"]["matrix_max_abs"] for case in cases]),
        "translation_m": finite_summary([case["b0_camera_parity"]["translation_m"] for case in cases]),
    }
    output["timing_seconds"] = finite_summary([case["timing_seconds"] for case in cases])
    return output


def evaluate_geometry(
    boundary: np.ndarray,
    raw_pose: np.ndarray,
    pre_pose: np.ndarray,
    gt_pre: np.ndarray,
    gt_post: np.ndarray,
    post_humans: dict[str, dict],
    target_humans: dict[str, dict],
) -> dict:
    """Evaluate one fixed boundary; GT is consumed only in this function."""
    gauge = pre_pose @ np.linalg.inv(gt_pre)
    target_camera = gauge @ gt_post
    final_camera = boundary @ raw_pose
    camera_translation = float(np.linalg.norm(final_camera[:3, 3] - target_camera[:3, 3]))
    camera_rotation = gt.rotation_error_deg(final_camera, target_camera)

    root_errors, joint_errors, vertex_errors = [], [], []
    final_roots, target_roots, per_person = {}, {}, {}
    for identity, human in post_humans.items():
        target = target_humans[identity]
        final_root = gt.transform_points(boundary, human["root"][None])[0]
        final_joints = gt.transform_points(boundary, human["joints"])
        final_vertices = gt.transform_points(boundary, human["vertices"])
        target_root = gt.transform_points(gauge, target["root"][None])[0]
        target_joints = gt.transform_points(gauge, target["joints"])
        target_vertices = gt.transform_points(gauge, target["vertices"])
        joint_count = min(len(final_joints), len(target_joints))
        vertex_count = min(len(final_vertices), len(target_vertices))
        root_error = float(np.linalg.norm(final_root - target_root))
        joint_error = float(np.mean(np.linalg.norm(final_joints[:joint_count] - target_joints[:joint_count], axis=1)))
        vertex_error = float(np.mean(np.linalg.norm(final_vertices[:vertex_count] - target_vertices[:vertex_count], axis=1)))
        root_errors.append(root_error)
        joint_errors.append(joint_error)
        vertex_errors.append(vertex_error)
        final_roots[identity] = final_root
        target_roots[identity] = target_root
        per_person[identity] = {
            "root_error_m": root_error,
            "joint_error_m": joint_error,
            "vertex_error_m": vertex_error,
        }

    pair_distance, pair_vector = [], []
    identities = sorted(final_roots)
    for offset, first in enumerate(identities):
        for second in identities[offset + 1 :]:
            predicted_vector = final_roots[first] - final_roots[second]
            target_vector = target_roots[first] - target_roots[second]
            pair_distance.append(abs(float(np.linalg.norm(predicted_vector)) - float(np.linalg.norm(target_vector))))
            pair_vector.append(float(np.linalg.norm(predicted_vector - target_vector)))
    return {
        "camera_translation_error_m": camera_translation,
        "camera_rotation_error_deg": camera_rotation,
        "camera_composite": camera_translation + 0.02 * camera_rotation,
        "catastrophic": bool(camera_translation > 2.0 or camera_rotation > 45.0),
        "human_assignment_count": len(post_humans),
        "human_root_error_m": float(np.mean(root_errors)) if root_errors else float("nan"),
        "human_joint_error_m": float(np.mean(joint_errors)) if joint_errors else float("nan"),
        "human_vertex_error_m": float(np.mean(vertex_errors)) if vertex_errors else float("nan"),
        "pairwise_distance_error_m": float(np.mean(pair_distance)) if pair_distance else float("nan"),
        "pairwise_vector_error_m": float(np.mean(pair_vector)) if pair_vector else float("nan"),
        "per_person": per_person,
    }


def run_case(
    model: ARCroco3DStereo,
    layer: SMPL_Layer,
    record: dict,
    gt_args: SimpleNamespace,
    device: torch.device,
) -> dict:
    """Run all model paths before loading any calibration or mesh supervision."""
    frame, pre_camera, post_camera = (int(record[key]) for key in ("frame", "pre_camera", "post_camera"))
    input_paths = [
        gt.extract_video_frame(gt_args, pre_camera, frame - 1),
        gt.extract_video_frame(gt_args, pre_camera, frame),
        gt.extract_video_frame(gt_args, post_camera, frame),
    ]
    views = gt.prepare_full_square_input(model, input_paths, gt_args)
    shadow_views = set_event_indices(copy.deepcopy(views), {2})
    raw_views = set_event_indices(copy.deepcopy(views[2:]), set())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
        shadow_predictions, shadow_returned, _ = model.forward_recurrent_lighter(
            shadow_views, str(device), ret_state=False, use_ttt3r=False, return_token_debug=True
        )
        raw_predictions, raw_returned, raw_debug = model.forward_recurrent_lighter(
            raw_views, str(device), ret_state=False, use_ttt3r=False, return_token_debug=True
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    shadow_pre_pose = camera_matrix(shadow_predictions[1]).astype(np.float64)
    shadow_post_pose = camera_matrix(shadow_predictions[2]).astype(np.float64)
    raw_pose = camera_matrix(raw_predictions[0]).astype(np.float64)
    boundary = boundary_from_camera_predictions(shadow_predictions[2], raw_predictions[0])[0].detach().float().cpu().numpy().astype(np.float64)
    b0_pose = boundary @ raw_pose
    parity = {
        "matrix_max_abs": float(np.max(np.abs(shadow_post_pose - b0_pose))),
        "translation_m": float(np.linalg.norm(shadow_post_pose[:3, 3] - b0_pose[:3, 3])),
    }
    if parity["matrix_max_abs"] > 1e-5:
        raise RuntimeError(f"B0/shadow camera parity failed: {parity}")

    # Runtime is complete.  GT identities/meshes below are evaluator-only.
    height, width = [int(value) for value in gt.tensor_numpy(raw_returned[0]["true_shape"])[0]]
    detections = gt.layer_humans(raw_predictions[0], raw_returned[0], raw_debug[0], layer)
    assigned, assignment = gt.assign_gt_identities(
        gt_args, detections, raw_pose, post_camera, frame, width, height
    )
    gt_pre = np.linalg.inv(gt.gt_w2c(gt_args, pre_camera, frame))
    gt_post = np.linalg.inv(gt.gt_w2c(gt_args, post_camera, frame))
    target_humans = gt.gt_human_payload(gt_args, frame, gt.joint_regressor(layer))

    return {
        "record": record,
        "timing_seconds": elapsed,
        "boundary_error_against_gt": boundary_error(
            boundary,
            (shadow_pre_pose @ np.linalg.inv(gt_pre)) @ gt_post @ np.linalg.inv(raw_pose),
        ),
        "b0_camera_parity": parity,
        "assignment": assignment,
        "methods": {
            "raw_reset": evaluate_geometry(
                np.eye(4, dtype=np.float64), raw_pose, shadow_pre_pose, gt_pre, gt_post, assigned, target_humans
            ),
            "b0_runtime": evaluate_geometry(
                boundary, raw_pose, shadow_pre_pose, gt_pre, gt_post, assigned, target_humans
            ),
        },
    }


def markdown(report: dict) -> str:
    summary = report["summary"]
    lines = [
        "# P0 MultiHuman Camera-Only Development Evaluation",
        "",
        f"Checkpoint: `{report['checkpoint']}`",
        "",
        "The B0 decision uses only `A(t-1), A(t), B(t)`. Official camera/SMPL-X are read only after all runtime predictions; this is a development split, not a final test.",
        "",
        "| Method | Camera T (m) | Camera R (deg) | Composite | Root (m) | Joint (m) | Vertex (m) | Pair vector (m) | Cat. | Assigned humans |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = summary["methods"][method]
        lines.append(
            f"| {method} | {row['camera_translation_error_m']['mean']:.4f} | "
            f"{row['camera_rotation_error_deg']['mean']:.3f} | {row['camera_composite']['mean']:.4f} | "
            f"{row['human_root_error_m']['mean']:.4f} | {row['human_joint_error_m']['mean']:.4f} | "
            f"{row['human_vertex_error_m']['mean']:.4f} | {row['pairwise_vector_error_m']['mean']:.4f} | "
            f"{row['catastrophic_count']} | {row['human_assignment_count']} |"
        )
    parity = summary["camera_parity"]
    lines.extend([
        "",
        f"Cases: `{summary['case_count']}`; failures: `{len(report['failures'])}`. "
        f"B0/shadow maximum camera discrepancy: `{parity['matrix_max_abs']['mean']:.3e}` mean, `{parity['matrix_max_abs']['p95']:.3e}` P95.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not args.model_path.is_file():
        raise FileNotFoundError(args.model_path)
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = list(payload["dev"])
    if args.max_cases:
        records = records[: int(args.max_cases)]
    if not records:
        raise ValueError("No development records selected")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = args.output_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    frame_cache_dir = args.frame_cache_dir or (args.output_dir / "frame_cache")
    frame_cache_dir.mkdir(parents=True, exist_ok=True)
    gt_args = SimpleNamespace(
        data_root=args.data_root,
        sequence="three",
        output_dir=frame_cache_dir,
        size=int(args.size),
    )
    gt.IDENTITIES = gt.SEQUENCE_IDENTITIES["three"]
    device = torch.device(args.device)
    model = ARCroco3DStereo.from_pretrained(str(args.model_path)).to(device)
    flags = configure_model(model)
    layer = SMPL_Layer(type="smplx", gender="neutral", num_betas=10, kid=False, person_center="head").to(device).eval()

    cases, failures = [], []
    for index, record in enumerate(records, start=1):
        name = str(record["event_id"])
        cache_path = cases_dir / f"{name}.json"
        if cache_path.is_file() and not args.overwrite:
            row = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            try:
                row = {"status": "ok", **run_case(model, layer, record, gt_args, device)}
            except Exception as error:
                row = {"status": "failed", "record": record, "error": repr(error), "traceback": traceback.format_exc()}
            cache_path.write_text(json.dumps(jsonable(row), indent=2, allow_nan=True) + "\n", encoding="utf-8")
        if row["status"] == "ok":
            cases.append(row)
            metric = row["methods"]["b0_runtime"]
            print(f"[{index:02d}/{len(records):02d}] {name} comp={metric['camera_composite']:.4f} root={metric['human_root_error_m']:.4f} n={metric['human_assignment_count']}", flush=True)
        else:
            failures.append(row)
            print(f"[{index:02d}/{len(records):02d}] {name} FAILED {row['error']}", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report = {
        "experiment": "v14_p0_multihuman_camera_only_dev",
        "checkpoint": str(args.model_path),
        "manifest": str(args.manifest),
        "manifest_content_sha256": payload.get("content_sha256"),
        "model_flags": flags,
        "constraints": {
            "runtime_inputs": "full-frame A(t-1), A(t), B(t)",
            "causal": True,
            "clean_raw_reset_is_deployed_state": True,
            "gt_camera_or_mesh_used_by_runtime": False,
            "human_correction_applied": False,
            "b0_shadow_camera_parity_required": True,
        },
        "summary": summarize(cases),
        "failures": failures,
        "cases": cases,
    }
    json_path = args.output_dir / "multihuman_camera_dev_evaluation.json"
    md_path = args.output_dir / "multihuman_camera_dev_evaluation.md"
    json_path.write_text(json.dumps(jsonable(report), indent=2, allow_nan=True) + "\n", encoding="utf-8")
    md_path.write_text(markdown(report), encoding="utf-8")
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
