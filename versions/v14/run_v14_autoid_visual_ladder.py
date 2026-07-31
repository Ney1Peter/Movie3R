#!/usr/bin/env python3
"""Export a five-step V14 B0/automatic-ID multi-human viewer ladder."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


from versions.v13 import gt_id_consensus as geometry  # noqa: E402
from versions.v14.probe_b0_identity_matching import (  # noqa: E402
    identity_cost_components,
    matching_costs,
)
from versions.v14.run_v14_2_multihuman_sequence import run_rollout  # noqa: E402
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


DEFAULT_BASE_MODEL = REPO_ROOT / "src/human3r_896L.pth"
DEFAULT_V14_MODEL = Path(
    "/dev/shm/movie3r_v14_1/"
    "v14_1_v9_event_only_boundary_geometry_self20_fp32_e80/checkpoint-best.pth"
)
DEFAULT_OUTPUT = Path("/dev/shm/movie3r_v14_visual/dance_t0300_c1_c4_k4")
METHODS = (
    "01_original_human3r_continuous",
    "02_human3r_hard_reset",
    "03_v14_learned_b0",
    "04_direct_autoid_multi",
    "05_b0_autoid_multi",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--v14_model", type=Path, default=DEFAULT_V14_MODEL)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path(
            "/data/wangzheng/iJCV-CODE/data/MultiHuman/Real-World-Capture/extracted"
        ),
    )
    parser.add_argument("--sequence", choices=tuple(geometry.SEQUENCE_IDENTITIES), default="dance")
    parser.add_argument("--timestamp", type=int, default=300)
    parser.add_argument("--source_camera", type=int, default=1)
    parser.add_argument("--target_camera", type=int, default=4)
    parser.add_argument("--offset", type=int, default=4)
    parser.add_argument("--pre_frames", type=int, default=4)
    parser.add_argument("--post_frames", type=int, default=6)
    parser.add_argument("--point_samples", type=int, default=1024)
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_model(path: Path, device: str, configure_v14: bool):
    from dust3r.model import ARCroco3DStereo

    model = ARCroco3DStereo.from_pretrained(str(path)).to(device)
    if configure_v14:
        flags = configure_model(model)
    else:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        flags = {
            "checkpoint": "original_human3r",
            "v9_oracle_correction_gate_enabled": bool(
                getattr(model, "v9_oracle_correction_gate_enabled", False)
            ),
        }
    return model, flags


def release_model(model) -> None:
    model.to("cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def native_track_id(prediction: dict, detection_index: int) -> int:
    value = prediction.get("smpl_id")
    if value is None:
        return int(detection_index)
    return int(value[0, detection_index].detach().cpu().item())


def tracked_humans(
    prediction: dict,
    view: dict,
    debug: dict,
    layer,
) -> tuple[dict[str, dict], list[dict]]:
    anonymous = geometry.layer_humans(prediction, view, debug, layer)
    tracked = {}
    for human in anonymous:
        track = f"track_{native_track_id(prediction, int(human['detection_index']))}"
        row = dict(human)
        row["track"] = track
        tracked[track] = row
    return tracked, anonymous


def sampled_cloud(
    prediction: dict, view: dict, humans: list[dict], point_samples: int
) -> np.ndarray:
    return geometry.sampled_background_cloud(prediction, view, humans, point_samples)


def automatic_assignment(
    pre_humans: dict[str, dict],
    post_humans: list[dict],
    matching_boundary: np.ndarray,
) -> dict:
    tracks = tuple(sorted(pre_humans))
    detections = [(f"det_{index}", human) for index, human in enumerate(post_humans)]
    components = identity_cost_components(
        pre_humans, detections, matching_boundary, tracks
    )
    cost = matching_costs(components)["root_torso"]
    rows, columns = linear_sum_assignment(cost)
    track_to_detection = {
        tracks[int(row)]: int(post_humans[int(column)]["detection_index"])
        for row, column in zip(rows, columns)
    }
    return {
        "tracks": tracks,
        "track_to_detection": track_to_detection,
        "cost": cost,
        "root_cost_m": components["root"],
        "torso_cost_deg": components["torso"],
    }


def assigned_post_humans(
    assignment: dict, post_humans: list[dict]
) -> dict[str, dict]:
    by_detection = {
        int(human["detection_index"]): human for human in post_humans
    }
    return {
        track: dict(by_detection[detection])
        for track, detection in assignment["track_to_detection"].items()
    }


def uniform_multi_boundary(cache: dict, tracks: tuple[str, ...]) -> tuple[np.ndarray, dict]:
    old_identities = geometry.IDENTITIES
    geometry.IDENTITIES = tracks
    try:
        candidates = geometry.human_candidates(cache)
    finally:
        geometry.IDENTITIES = old_identities
    if len(candidates) < 2:
        raise RuntimeError(f"Need at least two automatic-ID candidates, found {tuple(candidates)}")
    rotation = geometry.so3_mean(
        [candidates[track]["rotation"] for track in candidates]
    )
    translation = np.mean(
        np.stack([candidates[track]["translation"] for track in candidates]), axis=0
    )
    boundary = geometry.make_transform(rotation, translation)
    return boundary, candidates


def gt_detection_maps(
    args: argparse.Namespace,
    pre_prediction: dict,
    pre_view: dict,
    pre_humans: list[dict],
    post_prediction: dict,
    post_view: dict,
    post_humans: list[dict],
    pre_frame: int,
    post_frame: int,
) -> tuple[dict[int, str], dict[int, str]]:
    def one(prediction, view, humans, camera, frame):
        height, width = [int(value) for value in geometry.tensor_numpy(view["true_shape"])[0]]
        _, audit = geometry.assign_gt_identities(
            args,
            humans,
            camera_matrix(prediction),
            camera,
            frame,
            width,
            height,
        )
        return {
            int(row["detection_index"]): str(row["identity"])
            for row in audit.get("assignments", [])
        }

    return (
        one(
            pre_prediction,
            pre_view,
            pre_humans,
            int(args.source_camera),
            pre_frame,
        ),
        one(
            post_prediction,
            post_view,
            post_humans,
            int(args.target_camera),
            post_frame,
        ),
    )


def assignment_audit(
    assignment: dict,
    last_pre: dict[str, dict],
    pre_detection_gt: dict[int, str],
    post_detection_gt: dict[int, str],
) -> dict:
    rows = []
    for track, post_detection in assignment["track_to_detection"].items():
        pre_detection = int(last_pre[track]["detection_index"])
        pre_gt = pre_detection_gt.get(pre_detection)
        post_gt = post_detection_gt.get(int(post_detection))
        rows.append(
            {
                "track": track,
                "pre_detection": pre_detection,
                "pre_gt_identity": pre_gt,
                "post_detection": int(post_detection),
                "post_gt_identity": post_gt,
                "correct": bool(pre_gt is not None and pre_gt == post_gt),
            }
        )
    return {
        "rows": rows,
        "correct_count": int(sum(row["correct"] for row in rows)),
        "person_count": len(rows),
        "all_correct": bool(rows and all(row["correct"] for row in rows)),
    }


def remap_prediction_ids(
    predictions: list[dict], native_to_display: dict[int, int], unknown_offset: int
) -> list[dict]:
    output = []
    for prediction in predictions:
        row = copy.deepcopy(prediction)
        native = row.get("smpl_id")
        if native is not None:
            mapped = native.clone()
            for index, value in enumerate(native[0].detach().cpu().tolist()):
                mapped[0, index] = native_to_display.get(
                    int(value), unknown_offset + int(value)
                )
            row["smpl_id"] = mapped
        output.append(row)
    return output


def prediction_id_maps(
    pre_prediction: dict,
    post_prediction: dict,
    last_pre: dict[str, dict],
    assignment: dict,
) -> tuple[dict[int, int], dict[int, int]]:
    display_by_track = {track: index for index, track in enumerate(sorted(last_pre))}
    pre_native_to_display = {
        native_track_id(pre_prediction, int(human["detection_index"])): display_by_track[track]
        for track, human in last_pre.items()
    }
    post_native_to_display = {}
    for track, detection in assignment["track_to_detection"].items():
        native = native_track_id(post_prediction, int(detection))
        post_native_to_display[native] = display_by_track[track]
    return pre_native_to_display, post_native_to_display


def jsonable(value):
    return geometry.jsonable(value)


def main() -> None:
    args = parse_args()
    for path in (args.base_model, args.v14_model):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    geometry.IDENTITIES = geometry.SEQUENCE_IDENTITIES[str(args.sequence)]

    timestamp = int(args.timestamp)
    post_frame = timestamp + int(args.offset)
    pre_frames = list(range(timestamp - int(args.pre_frames) + 1, timestamp + 1))
    post_frames = list(range(post_frame, post_frame + int(args.post_frames)))
    pre_paths = [
        geometry.extract_video_frame(args, int(args.source_camera), frame)
        for frame in pre_frames
    ]
    post_paths = [
        geometry.extract_video_frame(args, int(args.target_camera), frame)
        for frame in post_frames
    ]
    cut_index = len(pre_paths)
    all_paths = pre_paths + post_paths

    device = str(args.device)
    base_model, base_flags = load_model(args.base_model, device, configure_v14=False)
    continuous_views = set_event_indices(
        geometry.prepare_full_square_input(base_model, all_paths, args), set()
    )
    raw_post_views = set_event_indices(
        geometry.prepare_full_square_input(base_model, post_paths, args), set()
    )
    continuous_predictions, continuous_returned, continuous_debug, continuous_time = run_rollout(
        base_model, continuous_views, device, "original_continuous"
    )
    raw_post_predictions, raw_post_returned, raw_post_debug, raw_post_time = run_rollout(
        base_model, raw_post_views, device, "original_hard_reset_post"
    )

    from dust3r.utils.smpl_layer import SMPL_Layer

    layer = SMPL_Layer(
        type="smplx",
        gender="neutral",
        num_betas=10,
        kid=False,
        person_center="head",
    ).to(device).eval()

    pre_predictions = continuous_predictions[:cut_index]
    pre_returned = continuous_returned[:cut_index]
    pre_debug = continuous_debug[:cut_index]
    pre_tracked, pre_anonymous, pre_clouds = [], [], []
    for prediction, view, debug in zip(pre_predictions, pre_returned, pre_debug):
        tracked, anonymous = tracked_humans(prediction, view, debug, layer)
        pre_tracked.append(tracked)
        pre_anonymous.append(anonymous)
        pre_clouds.append(sampled_cloud(prediction, view, anonymous, int(args.point_samples)))
    post_tracked, post_anonymous = tracked_humans(
        raw_post_predictions[0], raw_post_returned[0], raw_post_debug[0], layer
    )
    del post_tracked
    post_cloud = sampled_cloud(
        raw_post_predictions[0],
        raw_post_returned[0],
        post_anonymous,
        int(args.point_samples),
    )
    release_model(base_model)
    del base_model

    v14_model, v14_flags = load_model(args.v14_model, device, configure_v14=True)
    shadow_views = set_event_indices(
        geometry.prepare_full_square_input(v14_model, pre_paths + post_paths[:1], args),
        {cut_index},
    )
    shadow_predictions, _, _, shadow_time = run_rollout(
        v14_model, shadow_views, device, "v14_shadow_first_post"
    )
    shadow_camera = camera_matrix(shadow_predictions[-1]).astype(np.float64)
    raw_camera = camera_matrix(raw_post_predictions[0]).astype(np.float64)
    b0 = shadow_camera @ np.linalg.inv(raw_camera)
    release_model(v14_model)
    del v14_model

    active_tracks = tuple(sorted(pre_tracked[-1]))
    if len(active_tracks) < 2:
        raise RuntimeError(f"Need >=2 pre-cut Human3R tracks, found {active_tracks}")
    direct_assignment = automatic_assignment(
        pre_tracked[-1], post_anonymous, np.eye(4, dtype=np.float64)
    )
    b0_assignment = automatic_assignment(pre_tracked[-1], post_anonymous, b0)

    case = {
        "key": (
            f"{args.sequence}_t{timestamp:04d}_c{args.source_camera}"
            f"_c{args.target_camera}_k{args.offset}"
        ),
        "timestamp": timestamp,
        "source_camera": int(args.source_camera),
        "target_camera": int(args.target_camera),
        "offset": int(args.offset),
        "pre_frames": pre_frames,
        "post_frame": post_frame,
        "post_frames": post_frames,
    }

    def geometry_cache(assignment: dict) -> dict:
        return {
            "case": case,
            "poses": [camera_matrix(prediction).astype(np.float64) for prediction in pre_predictions]
            + [raw_camera],
            "humans": pre_tracked + [assigned_post_humans(assignment, post_anonymous)],
            "clouds": pre_clouds + [post_cloud],
        }

    direct_boundary, direct_candidates = uniform_multi_boundary(
        geometry_cache(direct_assignment), active_tracks
    )
    b0_multi_boundary, b0_candidates = uniform_multi_boundary(
        geometry_cache(b0_assignment), active_tracks
    )

    pre_detection_gt, post_detection_gt = gt_detection_maps(
        args,
        pre_predictions[-1],
        pre_returned[-1],
        pre_anonymous[-1],
        raw_post_predictions[0],
        raw_post_returned[0],
        post_anonymous,
        pre_frames[-1],
        post_frames[0],
    )
    direct_audit = assignment_audit(
        direct_assignment, pre_tracked[-1], pre_detection_gt, post_detection_gt
    )
    b0_audit = assignment_audit(
        b0_assignment, pre_tracked[-1], pre_detection_gt, post_detection_gt
    )

    gt_pre = np.linalg.inv(
        geometry.gt_w2c(args, int(args.source_camera), pre_frames[-1])
    )
    gt_post = np.linalg.inv(
        geometry.gt_w2c(args, int(args.target_camera), post_frames[0])
    )
    pre_camera = camera_matrix(pre_predictions[-1]).astype(np.float64)
    target_post_camera = pre_camera @ np.linalg.inv(gt_pre) @ gt_post
    gt_boundary = target_post_camera @ np.linalg.inv(raw_camera)

    full_reset_views = merged_views(pre_returned, raw_post_returned)
    reset_predictions = merged_predictions(pre_predictions, raw_post_predictions)
    boundary_tensor = {
        "b0": torch.from_numpy(b0.astype(np.float32)).unsqueeze(0),
        "direct": torch.from_numpy(direct_boundary.astype(np.float32)).unsqueeze(0),
        "b0_multi": torch.from_numpy(b0_multi_boundary.astype(np.float32)).unsqueeze(0),
    }

    direct_pre_ids, direct_post_ids = prediction_id_maps(
        pre_predictions[-1], raw_post_predictions[0], pre_tracked[-1], direct_assignment
    )
    b0_pre_ids, b0_post_ids = prediction_id_maps(
        pre_predictions[-1], raw_post_predictions[0], pre_tracked[-1], b0_assignment
    )
    direct_pre = remap_prediction_ids(pre_predictions, direct_pre_ids, 100)
    direct_post = remap_prediction_ids(raw_post_predictions, direct_post_ids, 100)
    b0_pre = remap_prediction_ids(pre_predictions, b0_pre_ids, 100)
    b0_post = remap_prediction_ids(raw_post_predictions, b0_post_ids, 100)

    payloads = {
        METHODS[0]: (continuous_predictions, continuous_returned),
        METHODS[1]: (reset_predictions, full_reset_views),
        METHODS[2]: (
            merged_predictions(
                pre_predictions,
                transformed_predictions(raw_post_predictions, boundary_tensor["b0"]),
            ),
            full_reset_views,
        ),
        METHODS[3]: (
            merged_predictions(
                direct_pre,
                transformed_predictions(direct_post, boundary_tensor["direct"]),
            ),
            full_reset_views,
        ),
        METHODS[4]: (
            merged_predictions(
                b0_pre,
                transformed_predictions(b0_post, boundary_tensor["b0_multi"]),
            ),
            full_reset_views,
        ),
    }
    viewer_outputs = {}
    for name, (predictions, views) in payloads.items():
        viewer_outputs[name] = str(
            save_viewer_payload(
                name,
                predictions,
                copy.deepcopy(views),
                args.output_dir,
            )
        )

    report = {
        "experiment": "V14 learned B0 before automatic-ID uniform multi-human alignment",
        "case": case,
        "models": {
            "base": str(args.base_model.resolve()),
            "v14": str(args.v14_model.resolve()),
            "base_flags": base_flags,
            "v14_flags": v14_flags,
        },
        "protocol": {
            "gt_identity_used_by_matcher": False,
            "gt_identity_used_by_geometry": False,
            "gt_identity_used_for_audit_only": True,
            "automatic_matcher": "anonymous root+torso Hungarian",
            "multi_geometry": "frozen Phase-2 per-human Fixed Explicit + V16; equal SO(3)/translation mean",
            "b0_role_in_method_05": "pre-match coordinate normalization only",
            "one_shared_boundary": True,
            "fixed_boundary_for_full_post_segment": True,
        },
        "timing_seconds": {
            "original_continuous": continuous_time,
            "original_hard_reset_post": raw_post_time,
            "v14_shadow": shadow_time,
        },
        "boundaries": {
            "b0": b0,
            "direct_autoid_multi": direct_boundary,
            "b0_autoid_multi": b0_multi_boundary,
            "gt_camera_audit": gt_boundary,
        },
        "boundary_errors": {
            "b0": boundary_error(b0, gt_boundary),
            "direct_autoid_multi": boundary_error(direct_boundary, gt_boundary),
            "b0_autoid_multi": boundary_error(b0_multi_boundary, gt_boundary),
        },
        "automatic_identity": {
            "direct": {
                "assignment": direct_assignment,
                "audit": direct_audit,
            },
            "after_b0": {
                "assignment": b0_assignment,
                "audit": b0_audit,
            },
        },
        "candidate_count": {
            "direct": len(direct_candidates),
            "after_b0": len(b0_candidates),
        },
        "viewer_outputs": viewer_outputs,
    }
    report_path = args.output_dir / "visual_ladder_report.json"
    report_path.write_text(
        json.dumps(jsonable(report), indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(jsonable(report["boundary_errors"]), indent=2), flush=True)
    print(
        f">> direct ID: {direct_audit['correct_count']}/{direct_audit['person_count']}; "
        f"B0 ID: {b0_audit['correct_count']}/{b0_audit['person_count']}",
        flush=True,
    )
    print(f">> wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
