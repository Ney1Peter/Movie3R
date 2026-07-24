#!/usr/bin/env python3
"""V14.1 multi-cut rollout for deterministic human-memory commit rules."""

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

from dust3r.datasets.avatarrex import AvatarReX_Pattern  # noqa: E402
from dust3r.utils.device import todevice  # noqa: E402
from dust3r.utils.image import pad_image  # noqa: E402
from scripts.boundary_human3r_reset_support import gt_human, predicted_human  # noqa: E402
from scripts.v10_token_alignment_4source_probe import (  # noqa: E402
    raw_roots_for_record,
    source_split_and_scope,
)
from versions.v12.experiments.v14_1_shot_aware_state_routing_probe import (  # noqa: E402
    DEFAULT_OUTPUT,
    DEFAULT_RECORDS,
    build_model,
    build_smpl_models,
    camera_matrix,
    mean_finite,
    read_jsonl,
    select_records,
    skeleton_scale,
    world_human,
)
from scripts.v9_learned_stream_alignment_overfit import gt_pose_from_view  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT / "multicut")
    parser.add_argument("--data_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data"))
    parser.add_argument("--model_path", type=Path, default=REPO_ROOT / "src" / "human3r_896L.pth")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--resolution", type=int, nargs=2, default=(512, 288))
    parser.add_argument("--resize_mode", default="human3r_demo")
    parser.add_argument("--cases_per_source", type=int, default=1)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--max_cuts", type=int, default=8)
    parser.add_argument("--frames_per_shot", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--token_alpha", type=float, default=0.0)
    parser.add_argument("--shape_alpha", type=float, default=0.75)
    parser.add_argument("--local_pose_alpha", type=float, default=0.15)
    parser.add_argument("--verify_thresholds", type=float, nargs="+", default=(1.5,))
    return parser.parse_args()


def rollout_record(record: dict, max_cuts: int, frames_per_shot: int) -> tuple[dict, list[int]]:
    seq_a = str(record["seqA"])
    seq_b = str(record["seqB"])
    start = int(record["start_frame"])
    frame_count = (max_cuts + 1) * frames_per_shot
    rollout_start = start - max(frame_count - 4, 0)
    seqs = []
    cuts = []
    for index in range(frame_count):
        shot = index // frames_per_shot
        seqs.append(seq_a if shot % 2 == 0 else seq_b)
        if index > 0 and index % frames_per_shot == 0:
            cuts.append(index)
    return (
        {
            "clip_type": "v14_1_multicut",
            "group": str(record.get("group", "")),
            "seqs": seqs,
            "frames": [rollout_start + index for index in range(frame_count)],
            "shot_labels": [1 if index in cuts else 0 for index in range(frame_count)],
            "transition_angles_deg": [
                float(record.get("view_angle_deg", 0.0)) if index in cuts else 0.0
                for index in range(frame_count)
            ],
            "view_angle_deg": float(record.get("view_angle_deg", 0.0)),
            "angle_bucket": str(record.get("angle_bucket", "unknown")),
            "pattern_id": f"{record['pattern_id']}_multicut_{max_cuts}",
        },
        cuts,
    )


def load_rollout_views(record: dict, args, model, device):
    pattern, cuts = rollout_record(record, int(args.max_cuts), int(args.frames_per_shot))
    split, _ = source_split_and_scope(record)
    dataset = AvatarReX_Pattern(
        split=split,
        ROOT=str(args.data_root),
        resolution=tuple(args.resolution),
        num_views=len(pattern["frames"]),
        aug_crop=0,
        allow_repeat=True,
        seed=401,
        n_corres=0,
        fixed_samples=[pattern],
        load_da3_depth=False,
        raw_calibration_root=raw_roots_for_record(record),
        resize_mode=str(args.resize_mode),
        max_humans=1,
    )
    views = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)))
    for view in views:
        view["img_mhmr"] = pad_image(view["img"], int(model.mhmr_img_res))
        view["img_mask"] = torch.ones_like(view["img_mask"], dtype=torch.bool)
        view["ray_mask"] = torch.zeros_like(view["ray_mask"], dtype=torch.bool)
        view["update"] = torch.ones_like(view["img_mask"], dtype=torch.bool)
        view["update_state"] = torch.ones_like(view["img_mask"], dtype=torch.bool)
        view["update_mem"] = torch.ones_like(view["img_mask"], dtype=torch.bool)
        view["update_v8_history"] = torch.ones_like(view["img_mask"], dtype=torch.bool)
        view["reset"] = torch.zeros_like(view["img_mask"], dtype=torch.bool)
    return todevice(views, device), cuts


def run(model, views, device, routing):
    with torch.no_grad():
        if routing is None:
            predictions, _ = model.forward_recurrent_lighter(views, str(device))
        else:
            predictions, _ = model.forward_recurrent_lighter(
                views, str(device), shot_routing=routing
            )
    return predictions


def gt_transforms(hard_predictions, views, cuts):
    local = [camera_matrix(prediction) for prediction in hard_predictions]
    gt = [gt_pose_from_view(view).detach().cpu().numpy().astype(np.float32) for view in views]
    gauge = local[0] @ np.linalg.inv(gt[0])
    target = [(gauge @ pose).astype(np.float32) for pose in gt]
    transforms = {index: (target[index] @ np.linalg.inv(local[index])).astype(np.float32) for index in cuts}
    return transforms, target


def routing_config(
    mode: str,
    cuts: list[int],
    transforms: dict[int, np.ndarray],
    *,
    shape_alpha: float = 0.75,
    local_pose_alpha: float = 0.15,
    max_world_root_jump: float = 1.5,
    token_alpha: float = 0.0,
):
    if mode == "hard_reset":
        return {"enabled": True, "mode": "hard_reset", "cut_indices": cuts}
    if mode not in {"immediate", "align", "verify"}:
        raise ValueError(f"Unsupported commit mode: {mode}")
    return {
        "enabled": True,
        "mode": "selective_full",
        "cut_indices": cuts,
        "boundary_transforms": transforms,
        "token_alpha": float(token_alpha),
        "shape_alpha": float(shape_alpha),
        "local_pose_alpha": float(local_pose_alpha),
        "update_alpha": 0.2,
        "commit_mode": mode,
        "min_detection_score": 0.35,
        "max_shape_delta": 3.0,
        "max_world_root_jump": float(max_world_root_jump),
    }


def active_transform(index: int, cuts: list[int], transforms: dict[int, np.ndarray]) -> np.ndarray:
    eligible = [cut for cut in cuts if cut <= index]
    return np.eye(4, dtype=np.float32) if not eligible else transforms[max(eligible)]


def evaluate_prefixes(predictions, hard_predictions, views, pred_layer, cuts, transforms, target_poses):
    humans = [
        predicted_human(prediction, views[index]["camera_intrinsics"], pred_layer)
        for index, prediction in enumerate(predictions)
    ]
    target_humans = [gt_human(view) for view in views]
    local_poses = [camera_matrix(prediction) for prediction in predictions]
    world = []
    target_world = []
    for index, human in enumerate(humans):
        pose = active_transform(index, cuts, transforms) @ local_poses[index]
        world.append(world_human(human, pose))
        target_world.append(world_human(target_humans[index], target_poses[index]))

    results = {}
    for cut_count in (1, 2, 4, 8):
        if cut_count > len(cuts):
            continue
        prefix_cuts = cuts[:cut_count]
        shape_jumps = []
        shape_drifts = []
        scale_jumps = []
        world_root_errors = []
        memory_root_errors = []
        commit_values = []
        for cut in prefix_cuts:
            before_shape = predictions[cut - 1].get("smpl_shape")
            after_shape = predictions[cut].get("smpl_shape")
            initial_shape = predictions[0].get("smpl_shape")
            if before_shape is not None and after_shape is not None and before_shape.shape[1] and after_shape.shape[1]:
                count = min(before_shape.shape[-1], after_shape.shape[-1])
                shape_jumps.append(
                    float(torch.linalg.vector_norm(before_shape[0, 0, :count] - after_shape[0, 0, :count]))
                )
                shape_drifts.append(
                    float(torch.linalg.vector_norm(initial_shape[0, 0, :count] - after_shape[0, 0, :count]))
                )
            if humans[cut - 1] is not None and humans[cut] is not None:
                scale_jumps.append(
                    abs(skeleton_scale(humans[cut - 1]["joints"]) - skeleton_scale(humans[cut]["joints"]))
                )
            if world[cut] is not None and target_world[cut] is not None:
                world_root_errors.append(
                    float(np.linalg.norm(world[cut]["root_world"] - target_world[cut]["root_world"]))
                )
            memory_root = predictions[cut].get("v14_1_memory_world_root")
            if memory_root is not None and memory_root.shape[1] and target_world[cut] is not None:
                memory_root_errors.append(
                    float(np.linalg.norm(memory_root[0, 0].cpu().numpy() - target_world[cut]["root_world"]))
                )
            commit = predictions[cut].get("v14_1_memory_commit")
            if commit is not None:
                commit_values.append(float(commit.float().mean()))

        end = prefix_cuts[-1] + 1
        scene_camera = 0.0
        scene_pointmap = 0.0
        for index in range(end):
            scene_camera = max(
                scene_camera,
                float((predictions[index]["camera_pose"] - hard_predictions[index]["camera_pose"]).abs().max()),
            )
            scene_pointmap = max(
                scene_pointmap,
                float(
                    (
                        predictions[index]["pts3d_in_self_view"]
                        - hard_predictions[index]["pts3d_in_self_view"]
                    ).abs().max()
                ),
            )
        results[str(cut_count)] = {
            "shape_jump_l2": mean_finite(shape_jumps),
            "shape_drift_from_first_l2": mean_finite(shape_drifts),
            "body_scale_jump_abs": mean_finite(scale_jumps),
            "world_root_m": mean_finite(world_root_errors),
            "memory_world_root_m": mean_finite(memory_root_errors),
            "memory_commit_rate": mean_finite(commit_values),
            "scene_camera_max_abs_vs_hard_reset": scene_camera,
            "scene_pointmap_max_abs_vs_hard_reset": scene_pointmap,
        }
    return results


def aggregate(cases):
    grouped = defaultdict(list)
    for case in cases:
        for method, prefixes in case["results"].items():
            for cut_count, metrics in prefixes.items():
                grouped[(method, cut_count)].append(metrics)
    summary = {}
    for (method, cut_count), rows in grouped.items():
        summary.setdefault(method, {})[cut_count] = {
            key: mean_finite([row[key] for row in rows]) for key in rows[0]
        }
    return summary


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = select_records(read_jsonl(args.records), int(args.cases_per_source), int(args.seed))
    if args.max_cases > 0:
        records = records[: int(args.max_cases)]
    device = torch.device(args.device)
    model = build_model(args)
    gt_model, pred_layer = build_smpl_models(model, device)
    cases = []
    no_cut_check = None
    started = time.perf_counter()
    for case_index, record in enumerate(records):
        print(f">> [{case_index + 1}/{len(records)}] {record['pattern_id']}", flush=True)
        views, cuts = load_rollout_views(record, args, model, device)
        hard = run(model, views, device, routing_config("hard_reset", cuts, {}))
        transforms, target_poses = gt_transforms(hard, views, cuts)
        with torch.no_grad():
            gt_model.update_smpl_gt(views)

        if no_cut_check is None:
            short_views = views[:4]
            original = run(model, short_views, device, None)
            routed = run(
                model,
                short_views,
                device,
                {
                    "enabled": True,
                    "mode": "selective_full",
                    "cut_indices": [],
                    "token_alpha": float(args.token_alpha),
                    "shape_alpha": float(args.shape_alpha),
                    "local_pose_alpha": float(args.local_pose_alpha),
                    "commit_mode": "verify",
                },
            )
            no_cut_check = {
                "camera_max_abs": max(
                    float((a["camera_pose"] - b["camera_pose"]).abs().max())
                    for a, b in zip(original, routed)
                ),
                "pointmap_max_abs": max(
                    float((a["pts3d_in_self_view"] - b["pts3d_in_self_view"]).abs().max())
                    for a, b in zip(original, routed)
                ),
                "smpl_shape_max_abs": max(
                    float((a["smpl_shape"] - b["smpl_shape"]).abs().max())
                    for a, b in zip(original, routed)
                ),
            }
        predictions = {"hard_reset": hard}
        for label, mode, threshold in (
            ("selective_immediate", "immediate", 1.5),
            ("selective_align", "align", 1.5),
        ):
            predictions[label] = run(
                model,
                views,
                device,
                routing_config(
                    mode,
                    cuts,
                    transforms,
                    shape_alpha=float(args.shape_alpha),
                    local_pose_alpha=float(args.local_pose_alpha),
                    max_world_root_jump=threshold,
                    token_alpha=float(args.token_alpha),
                ),
            )
        verify_labels = []
        for threshold in args.verify_thresholds:
            label = (
                "selective_verify"
                if len(args.verify_thresholds) == 1
                else f"selective_verify_{float(threshold):g}m"
            )
            verify_labels.append(label)
            predictions[label] = run(
                model,
                views,
                device,
                routing_config(
                    "verify",
                    cuts,
                    transforms,
                    shape_alpha=float(args.shape_alpha),
                    local_pose_alpha=float(args.local_pose_alpha),
                    max_world_root_jump=float(threshold),
                    token_alpha=float(args.token_alpha),
                ),
            )
        results = {
            method: evaluate_prefixes(
                output, hard, views, pred_layer, cuts, transforms, target_poses
            )
            for method, output in predictions.items()
        }
        cases.append({"record": record, "cuts": cuts, "results": results})
        largest_prefix = str(max(int(value) for value in results["selective_immediate"]))
        verify_text = " ".join(
            "{}={:.3f}".format(
                label,
                results[label][largest_prefix]["memory_world_root_m"],
            )
            for label in verify_labels
        )
        print(
            ">> {}-cut memory root immediate={:.3f} align={:.3f} {}".format(
                largest_prefix,
                results["selective_immediate"][largest_prefix]["memory_world_root_m"],
                results["selective_align"][largest_prefix]["memory_world_root_m"],
                verify_text,
            ),
            flush=True,
        )
        del views, predictions
        torch.cuda.empty_cache()

    report = {
        "experiment": "V14.1 multi-cut state-routing rollout",
        "case_count": len(cases),
        "elapsed_seconds": time.perf_counter() - started,
        "token_alpha": float(args.token_alpha),
        "shape_alpha": float(args.shape_alpha),
        "local_pose_alpha": float(args.local_pose_alpha),
        "verify_thresholds": [float(value) for value in args.verify_thresholds],
        "no_cut_check": no_cut_check,
        "cases": cases,
        "summary": aggregate(cases),
    }
    output = args.output_dir / "v14_1_multicut_rollout.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    print(f">> wrote {output}", flush=True)


if __name__ == "__main__":
    main()
