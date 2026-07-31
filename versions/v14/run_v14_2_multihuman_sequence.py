#!/usr/bin/env python3
"""Run one causal V14.2 + frozen GT-ID multi-human Boundary segment probe."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


from versions.v13 import gt_id_consensus as gt_consensus  # noqa: E402
from versions.v14.run_v14_2_single_sequence import (  # noqa: E402
    boundary_error,
    camera_matrix,
    configure_model,
    merged_predictions,
    merged_views,
    save_viewer_payload,
    set_event_indices,
    transformed_predictions,
)


DEFAULT_MODEL = Path(
    "/dev/shm/movie3r_v14_1/"
    "v14_1_v9_event_only_boundary_geometry_self20_fp32_e80/checkpoint-best.pth"
)
DEFAULT_OUTPUT = Path("/dev/shm/movie3r_v14_2/multihuman_three_t0900_c0_c3")
METHOD_ORDER = (
    "raw_reset",
    "v14_b0",
    "b0_v16_rotation_mean",
    "single_highest_quality",
    "gtid_uniform_multi",
    "gt_camera_only_boundary",
    "gt_rotation_multi_anchor_oracle",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path(
            "/data/wangzheng/iJCV-CODE/data/MultiHuman/Real-World-Capture/extracted"
        ),
    )
    parser.add_argument("--sequence", choices=("three",), default="three")
    parser.add_argument("--timestamp", type=int, default=900)
    parser.add_argument("--source_camera", type=int, default=0)
    parser.add_argument("--target_camera", type=int, default=3)
    parser.add_argument("--pre_frames", type=int, default=4)
    parser.add_argument("--post_frames", type=int, default=6)
    parser.add_argument("--point_samples", type=int, default=1024)
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_rollout(
    model,
    views: list[dict],
    device: str,
    name: str,
) -> tuple[list[dict], list[dict], list[dict], float]:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))
    started = time.perf_counter()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
        predictions, returned_views, debug = model.forward_recurrent_lighter(
            views,
            device,
            ret_state=False,
            use_ttt3r=False,
            return_token_debug=True,
        )
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))
    elapsed = time.perf_counter() - started
    print(f">> {name}: {len(views)} frames in {elapsed:.2f}s", flush=True)
    return predictions, returned_views, debug, elapsed


def assigned_frame(
    args: argparse.Namespace,
    prediction: dict,
    view: dict,
    debug: dict,
    layer,
    camera: int,
    frame: int,
) -> tuple[dict, dict, np.ndarray]:
    humans = gt_consensus.layer_humans(prediction, view, debug, layer)
    height, width = [int(value) for value in gt_consensus.tensor_numpy(view["true_shape"])[0]]
    assigned, assignment = gt_consensus.assign_gt_identities(
        args,
        humans,
        camera_matrix(prediction),
        camera,
        frame,
        width,
        height,
    )
    cloud = gt_consensus.sampled_background_cloud(
        prediction, view, humans, int(args.point_samples)
    )
    return assigned, assignment, cloud


def assign_prediction_ids(prediction: dict, humans: dict[str, dict]) -> dict:
    output = dict(prediction)
    current = prediction.get("smpl_id")
    if current is None:
        count = int(prediction["smpl_transl"].shape[1])
        current = torch.arange(count).reshape(1, count)
    ids = current.clone()
    for identity_index, identity in enumerate(gt_consensus.IDENTITIES):
        if identity in humans:
            ids[0, int(humans[identity]["detection_index"])] = identity_index
    output["smpl_id"] = ids
    return output


def persistent_root_anchor(
    history: list[tuple[int, dict]], target_frame: int
) -> tuple[np.ndarray, np.ndarray]:
    frames = [frame for frame, _ in history]
    humans = [human for _, human in history]
    velocity = gt_consensus.robust_velocity([human["root"] for human in humans], frames)
    delta = target_frame - frames[-1]
    return humans[-1]["root"] + delta * velocity, velocity


def b0_human_candidates(cache: dict, b0: np.ndarray) -> dict[str, dict]:
    pre_frames = [int(value) for value in cache["case"]["pre_frames"]]
    post_frame = int(cache["case"]["post_frame"])
    pre_humans = cache["humans"][:-1]
    post_humans = cache["humans"][-1]
    output = {}
    for identity in gt_consensus.IDENTITIES:
        history = [
            (frame, humans[identity])
            for frame, humans in zip(pre_frames, pre_humans)
            if identity in humans
        ]
        if identity not in post_humans or not history:
            continue
        current = post_humans[identity]
        anchor, velocity = persistent_root_anchor(history, post_frame)
        history_frames = [frame for frame, _ in history]
        history_humans = [human for _, human in history]
        target_torso, torso_motion = gt_consensus.predicted_rotation_frame(
            [human["torso"] for human in history_humans],
            history_frames,
            post_frame,
        )
        rotation, v16_debug = gt_consensus.yaw_residual(
            b0[:3, :3], [current["torso"]], [target_torso], 20.0
        )
        translation = anchor - rotation @ current["root"]
        quality = float(
            np.sqrt(
                max(current["score"], 1e-6)
                * max(float(np.mean([human["score"] for human in history_humans])), 1e-6)
            )
            * max(current["completeness"], 0.05)
        )
        output[identity] = {
            "identity": identity,
            "rotation": rotation,
            "translation": translation,
            "anchor": anchor,
            "post_root": current["root"],
            "quality": quality,
            "root_velocity_m_per_frame": velocity,
            "v16_debug": v16_debug,
            "torso_motion": torso_motion,
        }
    return output


def solution(rotation: np.ndarray, translation: np.ndarray, identities) -> dict:
    return {
        "rotation": np.asarray(rotation, dtype=np.float64),
        "translation": np.asarray(translation, dtype=np.float64),
        "identities": tuple(identities),
    }


def make_solutions(cache: dict, b0: np.ndarray, candidates: dict[str, dict]) -> dict:
    identities = tuple(candidates)
    if not identities:
        raise RuntimeError("No GT-ID human is shared across the cut")
    rotations = [candidates[identity]["rotation"] for identity in identities]
    translations = [candidates[identity]["translation"] for identity in identities]
    rotation_mean = gt_consensus.so3_mean(rotations)
    translation_mean = np.mean(np.stack(translations), axis=0)
    highest = max(identities, key=lambda identity: candidates[identity]["quality"])

    pre_pose = np.asarray(cache["poses"][-2], dtype=np.float64)
    post_pose = np.asarray(cache["poses"][-1], dtype=np.float64)
    gauge = pre_pose @ np.linalg.inv(np.asarray(cache["gt"]["pre_c2w"]))
    target_camera = gauge @ np.asarray(cache["gt"]["post_c2w"])
    gt_camera = target_camera @ np.linalg.inv(post_pose)
    gt_rotation = gt_camera[:3, :3]
    gt_anchor_translations = np.stack(
        [
            candidates[identity]["anchor"]
            - gt_rotation @ candidates[identity]["post_root"]
            for identity in identities
        ]
    )

    output = {
        "raw_reset": solution(np.eye(3), np.zeros(3), identities),
        "v14_b0": solution(b0[:3, :3], b0[:3, 3], identities),
        "b0_v16_rotation_mean": solution(rotation_mean, b0[:3, 3], identities),
        "single_highest_quality": solution(
            candidates[highest]["rotation"],
            candidates[highest]["translation"],
            (highest,),
        ),
        "gtid_uniform_multi": solution(rotation_mean, translation_mean, identities),
        "gt_camera_only_boundary": solution(
            gt_camera[:3, :3], gt_camera[:3, 3], identities
        ),
        "gt_rotation_multi_anchor_oracle": solution(
            gt_rotation, np.mean(gt_anchor_translations, axis=0), identities
        ),
    }
    output["single_highest_quality"]["selected_identity"] = highest
    return output


def solution_boundary(value: dict) -> np.ndarray:
    return gt_consensus.make_transform(value["rotation"], value["translation"])


def seam_metrics(cache: dict, boundary: np.ndarray) -> dict:
    pre_humans = cache["humans"][-2]
    post_humans = cache["humans"][-1]
    jumps = {}
    for identity in gt_consensus.IDENTITIES:
        if identity in pre_humans and identity in post_humans:
            mapped = gt_consensus.transform_points(
                boundary, post_humans[identity]["root"][None]
            )[0]
            jumps[identity] = float(np.linalg.norm(mapped - pre_humans[identity]["root"]))

    pre_cloud = np.asarray(cache["clouds"][-2], dtype=np.float64)
    post_cloud = gt_consensus.transform_points(
        boundary, np.asarray(cache["clouds"][-1], dtype=np.float64)
    )
    if len(pre_cloud) and len(post_cloud):
        pre_tree = cKDTree(pre_cloud)
        post_tree = cKDTree(post_cloud)
        distances = np.r_[
            post_tree.query(pre_cloud, k=1, workers=-1)[0],
            pre_tree.query(post_cloud, k=1, workers=-1)[0],
        ]
        cloud_median = float(np.median(distances))
        cloud_p90 = float(np.quantile(distances, 0.90))
    else:
        cloud_median = float("nan")
        cloud_p90 = float("nan")
    return {
        "root_jump_mean_m": float(np.mean(list(jumps.values()))) if jumps else float("nan"),
        "root_jump_max_m": float(np.max(list(jumps.values()))) if jumps else float("nan"),
        "root_jump_per_identity_m": jumps,
        "background_cloud_nn_median_m": cloud_median,
        "background_cloud_nn_p90_m": cloud_p90,
    }


def candidate_dispersion(candidates: dict[str, dict]) -> dict:
    identities = tuple(candidates)
    rotation = [
        gt_consensus.rotation_distance_deg(
            candidates[first]["rotation"], candidates[second]["rotation"]
        )
        for first, second in combinations(identities, 2)
    ]
    translation = [
        float(
            np.linalg.norm(
                candidates[first]["translation"] - candidates[second]["translation"]
            )
        )
        for first, second in combinations(identities, 2)
    ]
    return {
        "rotation_pairwise_deg": rotation,
        "rotation_mean_deg": float(np.mean(rotation)) if rotation else 0.0,
        "translation_pairwise_m": translation,
        "translation_mean_m": float(np.mean(translation)) if translation else 0.0,
    }


def markdown_report(report: dict) -> str:
    lines = [
        "# V14.2 Multi-Human Segment Probe",
        "",
        f"Case: `{report['case']['key']}` with {report['case']['human_count']} GT-ID humans.",
        "",
        "| Method | Camera T (m) | Camera R (deg) | Composite | Root (m) | Cut root jump (m) | Cloud NN (m) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in METHOD_ORDER:
        row = report["methods"][name]
        lines.append(
            f"| {name} | {row['camera_translation_error_m']:.3f} | "
            f"{row['camera_rotation_error_deg']:.2f} | {row['camera_composite']:.3f} | "
            f"{row['human_root_error_m']:.3f} | {row['seam']['root_jump_mean_m']:.3f} | "
            f"{row['seam']['background_cloud_nn_median_m']:.3f} |"
        )
    lines.extend(
        [
            "",
            "The experiment uses GT identity only for the controlled multi-human geometry probe. "
            "V14 produces one causal learned B0; every accepted human then produces one frozen "
            "V16/V12 candidate, and the multi method uses equal SO(3)/translation means.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not args.model_path.is_file():
        raise FileNotFoundError(args.model_path)
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gt_consensus.IDENTITIES = gt_consensus.SEQUENCE_IDENTITIES[args.sequence]

    from dust3r.model import ARCroco3DStereo
    from dust3r.utils.smpl_layer import SMPL_Layer
    from dust3r.v14_outputs import boundary_from_camera_predictions

    device = torch.device(args.device)
    model = ARCroco3DStereo.from_pretrained(str(args.model_path)).to(device)
    flags = configure_model(model)
    layer = SMPL_Layer(
        type="smplx",
        gender="neutral",
        num_betas=10,
        kid=False,
        person_center="head",
    ).to(device).eval()
    regressor = gt_consensus.joint_regressor(layer)

    timestamp = int(args.timestamp)
    pre_frames = list(range(timestamp - int(args.pre_frames) + 1, timestamp + 1))
    post_frames = list(range(timestamp, timestamp + int(args.post_frames)))
    pre_paths = [
        gt_consensus.extract_video_frame(args, int(args.source_camera), frame)
        for frame in pre_frames
    ]
    post_paths = [
        gt_consensus.extract_video_frame(args, int(args.target_camera), frame)
        for frame in post_frames
    ]
    all_paths = pre_paths + post_paths
    cut_index = len(pre_paths)

    continue_views = set_event_indices(
        gt_consensus.prepare_full_square_input(model, all_paths, args), set()
    )
    shadow_views = set_event_indices(
        gt_consensus.prepare_full_square_input(model, pre_paths + post_paths[:1], args),
        {cut_index},
    )
    raw_first_views = set_event_indices(
        gt_consensus.prepare_full_square_input(model, post_paths[:1], args), set()
    )
    raw_views = set_event_indices(
        gt_consensus.prepare_full_square_input(model, post_paths, args), set()
    )

    continue_predictions, continue_returned, continue_debug, continue_time = run_rollout(
        model, continue_views, str(device), "continue_event_off"
    )
    shadow_predictions, _, _, shadow_time = run_rollout(
        model, shadow_views, str(device), "shadow_pre_plus_first_post"
    )
    raw_first_predictions, _, _, raw_first_time = run_rollout(
        model, raw_first_views, str(device), "raw_reset_first_only"
    )
    raw_predictions, raw_returned, raw_debug, raw_time = run_rollout(
        model, raw_views, str(device), "raw_reset_full_post"
    )

    b0_tensor = boundary_from_camera_predictions(
        shadow_predictions[-1], raw_first_predictions[0]
    )
    b0 = b0_tensor[0].detach().cpu().numpy().astype(np.float64)

    pre_humans, pre_assignments, pre_clouds = [], [], []
    for prediction, view, debug, frame in zip(
        continue_predictions[:cut_index],
        continue_returned[:cut_index],
        continue_debug[:cut_index],
        pre_frames,
    ):
        humans, assignment, cloud = assigned_frame(
            args,
            prediction,
            view,
            debug,
            layer,
            int(args.source_camera),
            frame,
        )
        pre_humans.append(humans)
        pre_assignments.append(assignment)
        pre_clouds.append(cloud)

    post_humans, post_assignments, post_clouds = [], [], []
    for prediction, view, debug, frame in zip(
        raw_predictions, raw_returned, raw_debug, post_frames
    ):
        humans, assignment, cloud = assigned_frame(
            args,
            prediction,
            view,
            debug,
            layer,
            int(args.target_camera),
            frame,
        )
        post_humans.append(humans)
        post_assignments.append(assignment)
        post_clouds.append(cloud)

    pre_predictions = [
        assign_prediction_ids(prediction, humans)
        for prediction, humans in zip(continue_predictions[:cut_index], pre_humans)
    ]
    post_predictions = [
        assign_prediction_ids(prediction, humans)
        for prediction, humans in zip(raw_predictions, post_humans)
    ]

    cache = {
        "case": {
            "key": (
                f"{args.sequence}_t{timestamp:04d}_c{args.source_camera}"
                f"_c{args.target_camera}_k0_v14"
            ),
            "timestamp": timestamp,
            "source_camera": int(args.source_camera),
            "target_camera": int(args.target_camera),
            "offset": 0,
            "pre_frames": pre_frames,
            "post_frame": post_frames[0],
        },
        "poses": [camera_matrix(prediction).astype(np.float64) for prediction in pre_predictions]
        + [camera_matrix(post_predictions[0]).astype(np.float64)],
        "humans": pre_humans + [post_humans[0]],
        "assignment": pre_assignments + [post_assignments[0]],
        "clouds": pre_clouds + [post_clouds[0]],
        "gt": {
            "pre_c2w": np.linalg.inv(
                gt_consensus.gt_w2c(args, int(args.source_camera), pre_frames[-1])
            ),
            "post_c2w": np.linalg.inv(
                gt_consensus.gt_w2c(args, int(args.target_camera), post_frames[0])
            ),
            "post_humans": gt_consensus.gt_human_payload(
                args, post_frames[0], regressor
            ),
        },
        "runtime_seconds": continue_time + shadow_time + raw_first_time + raw_time,
    }
    candidates = b0_human_candidates(cache, b0)
    solutions = make_solutions(cache, b0, candidates)

    methods = {}
    boundaries = {}
    viewer_outputs = {}
    full_views = merged_views(continue_returned[:cut_index], raw_returned)
    for name in METHOD_ORDER:
        value = solutions[name]
        boundary = solution_boundary(value)
        boundaries[name] = boundary
        evaluated = gt_consensus.evaluate_solution(cache, value)
        evaluated["seam"] = seam_metrics(cache, boundary)
        methods[name] = evaluated
        if name == "raw_reset":
            merged = merged_predictions(pre_predictions, post_predictions)
        else:
            boundary_tensor = torch.from_numpy(boundary.astype(np.float32)).unsqueeze(0)
            merged = merged_predictions(
                pre_predictions,
                transformed_predictions(post_predictions, boundary_tensor),
            )
        viewer_outputs[name] = str(
            save_viewer_payload(name, merged, copy.deepcopy(full_views), args.output_dir)
        )

    gt_camera = boundaries["gt_camera_only_boundary"]
    report = {
        "experiment": "V14.2 causal learned B0 with frozen GT-ID multi-human consensus",
        "model_path": str(args.model_path.resolve()),
        "model_flags": flags,
        "case": {
            **cache["case"],
            "post_frames": post_frames,
            "human_count": len(candidates),
            "identities": tuple(candidates),
        },
        "constraints": {
            "causal": True,
            "future_frames_used_for_b0": False,
            "gt_identity_used_for_geometry_probe": True,
            "gt_camera_or_human_used_for_candidate_generation": False,
            "uniform_multi_fusion": True,
            "one_shared_boundary": True,
            "fixed_boundary_for_full_post_segment": True,
        },
        "timing_seconds": {
            "continue": continue_time,
            "shadow": shadow_time,
            "raw_first_only": raw_first_time,
            "raw_full": raw_time,
        },
        "b0_camera_boundary_error": boundary_error(b0, gt_camera),
        "candidate_dispersion": candidate_dispersion(candidates),
        "candidates": candidates,
        "methods": methods,
        "assignments": {
            "pre": pre_assignments,
            "post": post_assignments,
        },
        "viewer_outputs": viewer_outputs,
    }
    report_path = args.output_dir / "v14_2_multihuman_report.json"
    report_path.write_text(
        json.dumps(
            gt_consensus.jsonable(report),
            indent=2,
            ensure_ascii=False,
            allow_nan=True,
        )
        + "\n",
        encoding="utf-8",
    )
    markdown = markdown_report(gt_consensus.jsonable(report))
    markdown_path = args.output_dir / "v14_2_multihuman_report.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    print(markdown, flush=True)
    print(f">> wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
