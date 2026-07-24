#!/usr/bin/env python3
"""V14.1 causal shape-memory strength sweep.

This probe reuses a single Hard Reset Human3R forward pass per sample and then
applies the deterministic human-only memory update offline.  Since the memory
does not touch scene/camera state or raw human tokens in this experiment, this
is equivalent to the online geometry-memory path while making a development
and held-out sweep substantially cheaper.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
for path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dust3r.shot_human_memory import project_to_rotation  # noqa: E402
from versions.v12.experiments.v14_1_shot_aware_state_routing_probe import (  # noqa: E402
    DEFAULT_OUTPUT,
    DEFAULT_RECORDS,
    aggregate,
    build_model,
    build_smpl_models,
    evaluate_method,
    gt_boundary_transform,
    method_routing,
    prepare_views,
    read_jsonl,
    run_method,
    select_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--selection_report", type=Path)
    parser.add_argument("--exclude_report", type=Path)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT / "shape_sweep")
    parser.add_argument("--data_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data"))
    parser.add_argument("--model_path", type=Path, default=REPO_ROOT / "src" / "human3r_896L.pth")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolution", type=int, nargs=2, default=(512, 288))
    parser.add_argument("--resize_mode", default="human3r_demo")
    parser.add_argument("--boundary", type=int, default=2)
    parser.add_argument("--cases_per_source", type=int, default=4)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--shape_alphas", type=float, nargs="+", default=(0.0, 0.25, 0.5, 0.75, 1.0))
    parser.add_argument("--local_pose_alpha", type=float, default=0.15)
    parser.add_argument("--update_alpha", type=float, default=0.2)
    return parser.parse_args()


def report_pattern_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    report = json.loads(path.read_text(encoding="utf-8"))
    return {str(case["record"]["pattern_id"]) for case in report["cases"]}


def selected_records(args: argparse.Namespace) -> list[dict]:
    records = read_jsonl(args.records)
    by_id = {str(record["pattern_id"]): record for record in records}
    if args.selection_report is not None:
        selected = []
        report = json.loads(args.selection_report.read_text(encoding="utf-8"))
        for case in report["cases"]:
            pattern_id = str(case["record"]["pattern_id"])
            selected.append(by_id.get(pattern_id, case["record"]))
    else:
        excluded = report_pattern_ids(args.exclude_report)
        pool = [record for record in records if str(record["pattern_id"]) not in excluded]
        selected = select_records(pool, int(args.cases_per_source), int(args.seed))
    if args.max_cases > 0:
        selected = selected[: int(args.max_cases)]
    return selected


def clone_human_prediction(prediction: dict) -> dict:
    result = dict(prediction)
    for key in ("smpl_shape", "smpl_rotmat"):
        value = prediction.get(key)
        if value is not None:
            result[key] = value.detach().clone()
    return result


def blend_human_geometry(
    predictions: list[dict],
    boundary: int,
    shape_alpha: float,
    local_pose_alpha: float,
    update_alpha: float,
) -> list[dict]:
    """Apply the one-person causal geometry memory used by the online router."""

    shape_memory = None
    pose_memory = None
    output = []
    shape_alpha = float(np.clip(shape_alpha, 0.0, 1.0))
    local_pose_alpha = float(np.clip(local_pose_alpha, 0.0, 1.0))
    update_alpha = float(np.clip(update_alpha, 0.0, 1.0))

    for index, raw_prediction in enumerate(predictions):
        prediction = clone_human_prediction(raw_prediction)
        shape = prediction.get("smpl_shape")
        rotmat = prediction.get("smpl_rotmat")

        if index >= boundary and shape is not None and shape_memory is not None:
            prediction["smpl_shape"] = shape + shape_alpha * (shape_memory.to(shape) - shape)
            shape = prediction["smpl_shape"]
        if (
            index >= boundary
            and rotmat is not None
            and rotmat.shape[-3] > 1
            and pose_memory is not None
        ):
            local = rotmat[..., 1:, :, :]
            blended = local + local_pose_alpha * (pose_memory.to(local) - local)
            prediction["smpl_rotmat"] = torch.cat(
                [rotmat[..., :1, :, :], project_to_rotation(blended)], dim=-3
            )
            rotmat = prediction["smpl_rotmat"]

        if shape is not None:
            if shape_memory is None or shape_memory.shape != shape.shape:
                shape_memory = shape.detach().clone()
            else:
                shape_memory = shape_memory + update_alpha * (shape.detach() - shape_memory)
        if rotmat is not None and rotmat.shape[-3] > 1:
            local = rotmat[..., 1:, :, :]
            if pose_memory is None or pose_memory.shape != local.shape:
                pose_memory = local.detach().clone()
            else:
                pose_memory = project_to_rotation(
                    pose_memory + update_alpha * (local.detach() - pose_memory)
                )
        output.append(prediction)
    return output


def summarize_tradeoff(summary: dict) -> dict:
    result = {}
    for method, sources in summary.items():
        all_metrics = sources["all"]
        result[method] = {
            key: all_metrics[key]["mean"]
            for key in (
                "shape_jump_l2",
                "shape_gt_l2",
                "body_scale_jump_abs",
                "body_scale_gt_abs",
                "local_pose_jump_residual_deg",
                "local_pose_deg",
            )
        }
    return result


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = selected_records(args)
    device = torch.device(args.device)
    model = build_model(args)
    gt_model, pred_layer = build_smpl_models(model, device)
    cases = []
    started = time.perf_counter()

    for case_index, record in enumerate(records):
        print(f">> [{case_index + 1}/{len(records)}] {record['pattern_id']}", flush=True)
        views = prepare_views(record, args, model, device)
        hard_predictions = run_method(
            model,
            views,
            device,
            method_routing("hard_reset", int(args.boundary)),
        )
        gt_transform, _, target_poses = gt_boundary_transform(
            hard_predictions, views, int(args.boundary)
        )
        with torch.no_grad():
            gt_model.update_smpl_gt(views)

        results = {}
        for shape_alpha in args.shape_alphas:
            predictions = blend_human_geometry(
                hard_predictions,
                boundary=int(args.boundary),
                shape_alpha=float(shape_alpha),
                local_pose_alpha=float(args.local_pose_alpha),
                update_alpha=float(args.update_alpha),
            )
            label = f"shape_alpha_{float(shape_alpha):g}"
            results[label] = evaluate_method(
                predictions,
                hard_predictions,
                views,
                pred_layer,
                target_poses,
                gt_transform,
                int(args.boundary),
            )
        cases.append({"record": record, "results": results})
        print(
            ">> "
            + " ".join(
                f"a={alpha:g}:{results[f'shape_alpha_{float(alpha):g}']['shape_jump_l2']:.3f}"
                for alpha in args.shape_alphas
            ),
            flush=True,
        )
        del views, hard_predictions
        torch.cuda.empty_cache()

    summary = aggregate(cases)
    report = {
        "experiment": "V14.1 causal shape-memory strength sweep",
        "case_count": len(cases),
        "elapsed_seconds": time.perf_counter() - started,
        "shape_alphas": [float(value) for value in args.shape_alphas],
        "local_pose_alpha": float(args.local_pose_alpha),
        "update_alpha": float(args.update_alpha),
        "selection_report": None if args.selection_report is None else str(args.selection_report),
        "exclude_report": None if args.exclude_report is None else str(args.exclude_report),
        "cases": cases,
        "summary": summary,
        "tradeoff": summarize_tradeoff(summary),
    }
    output = args.output_dir / "v14_1_shape_memory_sweep.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    print(f">> wrote {output}", flush=True)


if __name__ == "__main__":
    main()
