#!/usr/bin/env python3
"""V14.1 deterministic shot-aware state-routing probe.

The experiment keeps Human3R frozen, resets scene/camera state before the first
post-cut frame, and optionally lets only the SMPL-X branch read a separate
cross-shot human memory.  Every routing method is evaluated with the same GT or
Fixed Explicit boundary transform so state routing is not credited for SE(3).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
for path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from add_ckpt_path import add_path_to_dust3r  # noqa: E402
from dust3r.model import ARCroco3DStereo  # noqa: E402
from dust3r.smpl_model import SMPLModel  # noqa: E402
from dust3r.utils.camera import pose_encoding_to_camera  # noqa: E402
from dust3r.utils.image import pad_image  # noqa: E402
from dust3r.utils.smpl_layer import SMPL_Layer  # noqa: E402
from scripts.boundary_human3r_reset_support import (  # noqa: E402
    gt_human,
    predicted_human,
    rotation_batch_error,
)
from scripts.v10_oracle_state_vs_gauge_probe import rotation_error_deg  # noqa: E402
from scripts.v10_token_alignment_4source_probe import load_aabb_views_for_record  # noqa: E402
from scripts.v9_learned_stream_alignment_overfit import gt_pose_from_view  # noqa: E402
from scripts.v9_online_stream_human3r_segment_align import strict_original_model  # noqa: E402


DEFAULT_RECORDS = (
    REPO_ROOT / "output" / "v10_candidate_selection" / "oracle_gt_4source" / "selected_records.jsonl"
)
DEFAULT_CASE_ROOT = REPO_ROOT / "output" / "v10_candidate_selection" / "oracle_gt_4source" / "cases"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "v14_1_shot_aware_state_routing"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--case_root", type=Path, default=DEFAULT_CASE_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data"))
    parser.add_argument("--model_path", type=Path, default=REPO_ROOT / "src" / "human3r_896L.pth")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--resolution", type=int, nargs=2, default=(512, 288))
    parser.add_argument("--resize_mode", default="human3r_demo")
    parser.add_argument("--boundary", type=int, default=2)
    parser.add_argument("--cases_per_source", type=int, default=2)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--skip_wrong_memory", action="store_true")
    parser.add_argument("--recommended_only", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_records(records: list[dict], per_source: int, seed: int) -> list[dict]:
    if per_source <= 0:
        return records
    rng = np.random.default_rng(seed)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[str(record["source"])].append(record)
    selected = []
    for source in sorted(grouped):
        by_angle: dict[str, list[dict]] = defaultdict(list)
        for record in grouped[source]:
            by_angle[str(record.get("angle_bucket", "unknown"))].append(record)
        buckets = sorted(by_angle)
        cursor = 0
        while len([row for row in selected if row["source"] == source]) < per_source:
            bucket = buckets[cursor % len(buckets)]
            pool = by_angle[bucket]
            if pool:
                index = int(rng.integers(0, len(pool)))
                selected.append(pool.pop(index))
            if not any(by_angle.values()):
                break
            cursor += 1
    return sorted(selected, key=lambda row: (str(row["source"]), str(row["pattern_id"])))


def build_model(args: argparse.Namespace) -> ARCroco3DStereo:
    add_path_to_dust3r(str(args.model_path))
    model = ARCroco3DStereo.from_pretrained(str(args.model_path)).to(args.device).eval()
    strict_original_model(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def build_smpl_models(model: ARCroco3DStereo, device: torch.device) -> tuple[SMPLModel, SMPL_Layer]:
    gt_model = SMPLModel(
        device,
        model_args={
            "patch_size": model.croco_args["patch_size"],
            "mhmr_img_res": model.mhmr_img_res,
            "bb_patch_size": model.bb_patch_size,
        },
    )
    pred_layer = SMPL_Layer(
        type="smplx", gender="neutral", num_betas=10, kid=False, person_center="head"
    ).to(device).eval()
    return gt_model, pred_layer


def prepare_views(record: dict, args: argparse.Namespace, model: ARCroco3DStereo, device: torch.device):
    loader_args = SimpleNamespace(
        data_root=args.data_root,
        resolution=tuple(args.resolution),
        resize_mode=args.resize_mode,
        boundary=int(args.boundary),
    )
    views = load_aabb_views_for_record(record, loader_args, device)
    for view in views:
        view["img_mhmr"] = pad_image(view["img"], int(model.mhmr_img_res))
        view["reset"] = torch.zeros_like(view["reset"], dtype=torch.bool)
    return views


def camera_matrix(prediction: dict) -> np.ndarray:
    return pose_encoding_to_camera(prediction["camera_pose"].float())[0].cpu().numpy().astype(np.float32)


def fixed_transform(record: dict, case_root: Path) -> np.ndarray | None:
    path = case_root / str(record["pattern_id"]) / "case_metrics.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return np.asarray(data["fixed_explicit"]["transform"], dtype=np.float32)


def gt_boundary_transform(
    hard_predictions: list[dict], views: list[dict], boundary: int
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    predicted = [camera_matrix(prediction) for prediction in hard_predictions]
    gt = [gt_pose_from_view(view).detach().cpu().numpy().astype(np.float32) for view in views]
    gauge = predicted[0] @ np.linalg.inv(gt[0])
    target = [(gauge @ pose).astype(np.float32) for pose in gt]
    transform = target[boundary] @ np.linalg.inv(predicted[boundary])
    return transform.astype(np.float32), predicted, target


def transform_pose(transform: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return (transform @ pose).astype(np.float32)


def skeleton_scale(joints: np.ndarray | None) -> float:
    if joints is None or len(joints) < 18:
        return float("nan")
    pairs = ((1, 2), (16, 17), (1, 16), (2, 17), (0, 1), (0, 2), (1, 4), (2, 5))
    lengths = [np.linalg.norm(joints[a] - joints[b]) for a, b in pairs]
    return float(np.mean(lengths))


def world_human(human: dict | None, pose: np.ndarray) -> dict | None:
    if human is None:
        return None
    rotation = pose[:3, :3]
    translation = pose[:3, 3]
    return {
        **human,
        "joints_world": human["joints"] @ rotation.T + translation,
        "root_world": rotation @ human["root"] + translation,
        "torso_world": rotation @ human["torso"],
    }


def mean_finite(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(array)) if np.isfinite(array).any() else float("nan")


def method_routing(
    name: str,
    boundary: int,
    boundary_transform: np.ndarray | None = None,
    memory_override: dict | None = None,
) -> dict:
    base = {"enabled": True, "cut_indices": [boundary]}
    if name == "original_continue":
        return {**base, "mode": "continue"}
    if name == "full_old_state_read":
        return {**base, "mode": "full_old_state"}
    if name == "hard_reset":
        return {**base, "mode": "hard_reset"}
    if name == "hard_reset_tracklet":
        return {**base, "mode": "tracklet"}

    config = {
        **base,
        "mode": "selective_full",
        "token_alpha": 0.0,
        "shape_alpha": 0.0,
        "local_pose_alpha": 0.0,
        "update_alpha": 0.2,
        "commit_mode": "none",
        "min_detection_score": 0.35,
        "max_shape_delta": 3.0,
        "max_world_root_jump": 1.5,
    }
    if name == "selective_token":
        config["token_alpha"] = 0.05
    elif name == "selective_shape":
        config["shape_alpha"] = 0.75
    elif name == "selective_shape_pose":
        config.update(shape_alpha=0.75, local_pose_alpha=0.15)
    elif name == "selective_recommended_align":
        config.update(
            token_alpha=0.0,
            shape_alpha=0.25,
            local_pose_alpha=0.15,
            commit_mode="align",
        )
    elif name in {
        "selective_full_immediate",
        "selective_full_align",
        "selective_full_verify",
        "selective_zero_memory",
        "selective_wrong_video",
    }:
        config.update(token_alpha=0.05, shape_alpha=0.75, local_pose_alpha=0.15)
        if name == "selective_full_immediate":
            config["commit_mode"] = "immediate"
        elif name == "selective_full_align":
            config["commit_mode"] = "align"
        else:
            config["commit_mode"] = "verify"
    else:
        raise KeyError(name)
    if boundary_transform is not None:
        config["boundary_transforms"] = {boundary: boundary_transform}
    if memory_override is not None:
        config["memory_override"] = memory_override
    return config


def run_method(model, views, device, routing) -> list[dict]:
    with torch.no_grad():
        predictions, _ = model.forward_recurrent_lighter(
            views, str(device), ret_state=False, use_ttt3r=False, shot_routing=routing
        )
    return predictions


def memory_snapshot(prediction: dict) -> dict | None:
    snapshot = {}
    for name in ("token", "shape", "local_rotmat", "world_root"):
        value = prediction.get(f"v14_1_memory_{name}")
        if value is not None:
            snapshot[name] = value.detach().cpu().numpy().astype(np.float32)
    return snapshot or None


def zero_snapshot(snapshot: dict | None) -> dict | None:
    return None if snapshot is None else {key: np.zeros_like(value) for key, value in snapshot.items()}


def evaluate_method(
    predictions: list[dict],
    hard_predictions: list[dict],
    views: list[dict],
    pred_layer: SMPL_Layer,
    target_poses: list[np.ndarray],
    boundary_transform: np.ndarray,
    boundary: int,
) -> dict:
    local_poses = [camera_matrix(prediction) for prediction in predictions]
    world_poses = [
        pose if index < boundary else transform_pose(boundary_transform, pose)
        for index, pose in enumerate(local_poses)
    ]
    predicted_humans = [
        predicted_human(prediction, views[index]["camera_intrinsics"], pred_layer)
        for index, prediction in enumerate(predictions)
    ]
    target_humans = [gt_human(view) for view in views]
    predicted_world = [world_human(human, world_poses[index]) for index, human in enumerate(predicted_humans)]
    target_world = [world_human(human, target_poses[index]) for index, human in enumerate(target_humans)]

    camera_translation = []
    camera_rotation = []
    world_root_error = []
    torso_error = []
    shape_error = []
    scale_error = []
    local_pose_error = []
    for index in range(boundary, len(predictions)):
        camera_translation.append(float(np.linalg.norm(world_poses[index][:3, 3] - target_poses[index][:3, 3])))
        camera_rotation.append(rotation_error_deg(world_poses[index], target_poses[index]))
        pred_human = predicted_world[index]
        target_human = target_world[index]
        if pred_human is not None and target_human is not None:
            world_root_error.append(float(np.linalg.norm(pred_human["root_world"] - target_human["root_world"])))
            torso_error.append(rotation_error_deg(pred_human["torso_world"], target_human["torso_world"]))
            scale_error.append(abs(skeleton_scale(pred_human["joints"]) - skeleton_scale(target_human["joints"])))
            local_pose_error.append(rotation_batch_error(pred_human["local_rotmat"], target_human["local_rotmat"]))
        pred_shape = predictions[index].get("smpl_shape")
        gt_shape = views[index].get("smpl_shape")
        gt_mask = views[index].get("smpl_mask")
        if pred_shape is not None and pred_shape.shape[1] and gt_shape is not None and bool(gt_mask[0, 0]):
            count = min(pred_shape.shape[-1], gt_shape.shape[-1])
            shape_error.append(
                float(torch.linalg.vector_norm(pred_shape[0, 0, :count].cpu() - gt_shape[0, 0, :count].cpu()))
            )

    pre_index = boundary - 1
    pred_before = predicted_humans[pre_index]
    pred_after = predicted_humans[boundary]
    target_before = target_humans[pre_index]
    target_after = target_humans[boundary]
    shape_jump = float("nan")
    if predictions[pre_index].get("smpl_shape") is not None and predictions[boundary].get("smpl_shape") is not None:
        before = predictions[pre_index]["smpl_shape"]
        after = predictions[boundary]["smpl_shape"]
        if before.shape[1] and after.shape[1]:
            count = min(before.shape[-1], after.shape[-1])
            shape_jump = float(torch.linalg.vector_norm(before[0, 0, :count] - after[0, 0, :count]))
    scale_jump = abs(skeleton_scale(None if pred_before is None else pred_before["joints"]) - skeleton_scale(None if pred_after is None else pred_after["joints"]))
    pose_jump_residual = float("nan")
    if pred_before is not None and pred_after is not None and target_before is not None and target_after is not None:
        pred_jump = rotation_batch_error(pred_after["local_rotmat"], pred_before["local_rotmat"])
        target_jump = rotation_batch_error(target_after["local_rotmat"], target_before["local_rotmat"])
        pose_jump_residual = abs(pred_jump - target_jump)
    root_jump_residual = float("nan")
    if all(item is not None for item in (predicted_world[pre_index], predicted_world[boundary], target_world[pre_index], target_world[boundary])):
        pred_delta = predicted_world[boundary]["root_world"] - predicted_world[pre_index]["root_world"]
        target_delta = target_world[boundary]["root_world"] - target_world[pre_index]["root_world"]
        root_jump_residual = float(np.linalg.norm(pred_delta - target_delta))

    scene_camera_max = 0.0
    scene_pointmap_max = 0.0
    for index in range(boundary, len(predictions)):
        scene_camera_max = max(
            scene_camera_max,
            float((predictions[index]["camera_pose"] - hard_predictions[index]["camera_pose"]).abs().max()),
        )
        scene_pointmap_max = max(
            scene_pointmap_max,
            float(
                (
                    predictions[index]["pts3d_in_self_view"]
                    - hard_predictions[index]["pts3d_in_self_view"]
                ).abs().max()
            ),
        )
    ids_before = predictions[pre_index].get("smpl_id")
    ids_after = predictions[boundary].get("smpl_id")
    track_id_continuous = float("nan")
    if ids_before is not None and ids_after is not None and ids_before.numel() and ids_after.numel():
        track_id_continuous = float(ids_before.flatten()[0] == ids_after.flatten()[0])

    memory_commit = [
        float(prediction["v14_1_memory_commit"].float().mean())
        for prediction in predictions[boundary:]
        if "v14_1_memory_commit" in prediction
    ]
    memory_root_error = []
    for index in range(boundary, len(predictions)):
        memory_root = predictions[index].get("v14_1_memory_world_root")
        target_human = target_world[index]
        if memory_root is not None and target_human is not None and memory_root.shape[1]:
            memory_root_error.append(
                float(np.linalg.norm(memory_root[0, 0].cpu().numpy() - target_human["root_world"]))
            )

    return {
        "camera_translation_m": mean_finite(camera_translation),
        "camera_rotation_deg": mean_finite(camera_rotation),
        "world_root_m": mean_finite(world_root_error),
        "torso_orientation_deg": mean_finite(torso_error),
        "shape_gt_l2": mean_finite(shape_error),
        "body_scale_gt_abs": mean_finite(scale_error),
        "local_pose_deg": mean_finite(local_pose_error),
        "shape_jump_l2": shape_jump,
        "body_scale_jump_abs": scale_jump,
        "local_pose_jump_residual_deg": pose_jump_residual,
        "world_root_jump_residual_m": root_jump_residual,
        "scene_camera_max_abs_vs_hard_reset": scene_camera_max,
        "scene_pointmap_max_abs_vs_hard_reset": scene_pointmap_max,
        "track_id_continuous": track_id_continuous,
        "memory_commit_rate": mean_finite(memory_commit),
        "memory_world_root_m": mean_finite(memory_root_error),
    }


def aggregate(cases: list[dict]) -> dict:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for case in cases:
        source = str(case["record"]["source"])
        for key, result in case["results"].items():
            grouped[(key, "all")].append(result)
            grouped[(key, source)].append(result)
    summary = {}
    for (method, source), rows in grouped.items():
        metrics = {}
        for name in rows[0]:
            values = np.asarray([row[name] for row in rows], dtype=np.float64)
            finite = values[np.isfinite(values)]
            metrics[name] = {
                "mean": float(np.mean(finite)) if len(finite) else float("nan"),
                "median": float(np.median(finite)) if len(finite) else float("nan"),
                "p90": float(np.percentile(finite, 90)) if len(finite) else float("nan"),
                "count": int(len(finite)),
            }
        summary.setdefault(method, {})[source] = metrics
    return summary


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# V14.1 Shot-Aware State Routing",
        "",
        f"Cases: {report['case_count']}",
        "",
        "| Method / Boundary | Shape jump | Scale jump | Local-pose residual | World-root jump | Scene camera diff | Scene pointmap diff |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, sources in report["summary"].items():
        metrics = sources.get("all")
        if metrics is None:
            continue
        lines.append(
            "| {} | {:.4f} | {:.4f} | {:.2f} | {:.4f} | {:.2e} | {:.2e} |".format(
                method,
                metrics["shape_jump_l2"]["mean"],
                metrics["body_scale_jump_abs"]["mean"],
                metrics["local_pose_jump_residual_deg"]["mean"],
                metrics["world_root_jump_residual_m"]["mean"],
                metrics["scene_camera_max_abs_vs_hard_reset"]["mean"],
                metrics["scene_pointmap_max_abs_vs_hard_reset"]["mean"],
            )
        )
    lines.extend(
        [
            "",
            "`max_humans=1` is used by this four-source protocol, so numeric ID continuity is only a plumbing check; it is not a formal multi-person IDF1 result.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("Invalid shard index")
    records = select_records(read_jsonl(args.records), int(args.cases_per_source), int(args.seed))
    records = [row for index, row in enumerate(records) if index % args.num_shards == args.shard_index]
    if args.max_cases > 0:
        records = records[: int(args.max_cases)]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = build_model(args)
    gt_model, pred_layer = build_smpl_models(model, device)
    cases = []
    wrong_memory_by_source: dict[str, dict] = {}
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
        deploy_transform = fixed_transform(record, args.case_root)
        if deploy_transform is None:
            deploy_transform = gt_transform.copy()

        with torch.no_grad():
            gt_model.update_smpl_gt(views)

        results = {}
        prediction_cache = {"hard_reset": hard_predictions}
        base_methods = () if args.recommended_only else (
            "original_continue",
            "full_old_state_read",
            "hard_reset_tracklet",
            "selective_token",
            "selective_shape",
            "selective_shape_pose",
        )
        for method in base_methods:
            prediction_cache[method] = run_method(
                model, views, device, method_routing(method, int(args.boundary), gt_transform)
            )

        correct_snapshot = (
            None
            if args.recommended_only
            else memory_snapshot(prediction_cache["selective_shape"][int(args.boundary) - 1])
        )
        if args.recommended_only:
            controls = {
                "selective_recommended_align@gt": (
                    "selective_recommended_align",
                    gt_transform,
                    None,
                ),
                "selective_recommended_align@fixed": (
                    "selective_recommended_align",
                    deploy_transform,
                    None,
                ),
            }
        else:
            controls = {
                "selective_full_immediate@gt": ("selective_full_immediate", gt_transform, None),
                "selective_full_align@gt": ("selective_full_align", gt_transform, None),
                "selective_full_verify@gt": ("selective_full_verify", gt_transform, None),
                "selective_full_verify@fixed": ("selective_full_verify", deploy_transform, None),
                "selective_zero_memory@gt": (
                    "selective_zero_memory",
                    gt_transform,
                    zero_snapshot(correct_snapshot),
                ),
            }
        source = str(record["source"])
        if not args.recommended_only and not args.skip_wrong_memory and source in wrong_memory_by_source:
            controls["selective_wrong_video@gt"] = (
                "selective_wrong_video",
                gt_transform,
                wrong_memory_by_source[source],
            )
        for label, (method, transform, override) in controls.items():
            prediction_cache[label] = run_method(
                model,
                views,
                device,
                method_routing(method, int(args.boundary), transform, override),
            )
        if not args.recommended_only and correct_snapshot is not None:
            wrong_memory_by_source[source] = correct_snapshot

        for method, predictions in prediction_cache.items():
            if method.endswith("@fixed"):
                evaluations = ((method, deploy_transform),)
            elif "@" in method:
                evaluations = ((method, gt_transform),)
            else:
                evaluations = (
                    (f"{method}@gt", gt_transform),
                    (f"{method}@fixed", deploy_transform),
                )
            for label, transform in evaluations:
                results[label] = evaluate_method(
                    predictions,
                    hard_predictions,
                    views,
                    pred_layer,
                    target_poses,
                    transform,
                    int(args.boundary),
                )

        cases.append(
            {
                "record": record,
                "gt_boundary_transform": gt_transform.tolist(),
                "fixed_boundary_transform": deploy_transform.tolist(),
                "results": results,
            }
        )
        if args.recommended_only:
            print(
                ">> shape jump hard={:.3f} recommended={:.3f}".format(
                    results["hard_reset@gt"]["shape_jump_l2"],
                    results["selective_recommended_align@fixed"]["shape_jump_l2"],
                ),
                flush=True,
            )
        else:
            print(
                ">> shape jump hard={:.3f} shape={:.3f} full={:.3f}".format(
                    results["hard_reset@gt"]["shape_jump_l2"],
                    results["selective_shape@gt"]["shape_jump_l2"],
                    results["selective_full_verify@fixed"]["shape_jump_l2"],
                ),
                flush=True,
            )
        del views, hard_predictions, prediction_cache
        torch.cuda.empty_cache()

    report = {
        "experiment": "V14.1 Shot-Aware Modality-Selective State Routing",
        "case_count": len(cases),
        "elapsed_seconds": time.perf_counter() - started,
        "constraints": {
            "human3r_frozen": True,
            "gt_cut_idx": True,
            "scene_camera_predecode_reset": True,
            "human_memory_isolated_from_scene_state": True,
            "learned_router": False,
            "max_humans": 1,
            "recommended_only": bool(args.recommended_only),
        },
        "cases": cases,
        "summary": aggregate(cases),
    }
    suffix = f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
    output_json = args.output_dir / f"v14_1_state_routing_{suffix}.json"
    output_md = args.output_dir / f"v14_1_state_routing_{suffix}.md"
    output_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(report, output_md)
    print(f">> wrote {output_json}", flush=True)


if __name__ == "__main__":
    main()
