#!/usr/bin/env python3
"""Cache a causally reconstructed long post-shot trajectory for stability work.

This is intentionally a *runtime-first, evaluator-second* builder:

    pre RGB history + first post RGB -> read-only shadow B0
    post RGB stream                  -> clean reset P0 stream
    B0 + automatic association + first-frame BRTC-LC -> current predictions
    ---------------------------------------------------------------
    only after all of the above: calibrated GT meshes for evaluation

The BRTC translation is estimated once at the boundary and carried by native
track ID.  It is therefore a correct baseline for testing a *shot-internal*
root residual.  The cache contains no viewer images or pointmaps: only the
geometry required for reproducible numerical experiments.  It runs on CPU by
design and never touches a GPU.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from dust3r.model import ARCroco3DStereo
from dust3r.utils.smpl_layer import SMPL_Layer
from versions.v13 import gt_id_consensus as gt
from versions.v14.b0_person_triangulation import (
    DEFAULT_CONFIG as BRTC_CONFIG,
    refine_matched_people,
)
from versions.v14.export_p5_brtc_demo_payload import CHECKPOINT, as_array, under_repo
from versions.v14.probe_p1_foot_scene_observability import (
    anonymous_match,
    configure_model,
    decode_people,
    set_event_indices,
    transform_person,
)
from versions.v14.run_v14_2_single_sequence import camera_matrix


DEFAULT_DATA = Path(
    "/data/wangzheng/iJCV-CODE/data/MultiHuman/Real-World-Capture/extracted"
)
DEFAULT_OUTPUT = REPO_ROOT / "output/v14/within_shot_stability/cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", default="three", choices=("three", "dance", "box"))
    parser.add_argument("--frame", type=int, required=True, help="last pre / first post dataset frame")
    parser.add_argument("--pre-camera", type=int, required=True)
    parser.add_argument("--post-camera", type=int, required=True)
    parser.add_argument("--pre-frames", type=int, default=5)
    parser.add_argument("--post-frames", type=int, default=25)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    value = np.asarray(points, dtype=np.float64)
    return value @ matrix[:3, :3].T + matrix[:3, 3]


def native_ids(prediction: dict[str, Any], count: int) -> list[int]:
    value = prediction.get("smpl_id")
    if value is None:
        return list(range(count))
    result = [int(item) for item in value[0, :count].detach().cpu().tolist()]
    return result if len(result) == count else list(range(count))


def cache_name(args: argparse.Namespace) -> str:
    return (
        f"{args.sequence}_t{args.frame:04d}_c{args.pre_camera}_c{args.post_camera}"
        f"_pre{args.pre_frames}_post{args.post_frames}"
    )


def gt_targets(
    gt_args: SimpleNamespace,
    layer: SMPL_Layer,
    sequence: str,
    frames: list[int],
    pre_camera: int,
    post_camera: int,
    evaluation_gauge: np.ndarray,
    ordered_identities: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return evaluator-only GT in persistent predicted-track order."""
    regressor = gt.joint_regressor(layer)
    gt_cameras, roots, joints, vertices = [], [], [], []
    for frame in frames:
        post_c2w = np.linalg.inv(gt.gt_w2c(gt_args, int(post_camera), int(frame)))
        payload = gt_payload(gt_args, int(frame), regressor, ordered_identities)
        gt_cameras.append(evaluation_gauge @ post_c2w)
        roots.append(np.stack([
            transform_points(evaluation_gauge, payload[identity]["root"][None])[0]
            for identity in ordered_identities
        ]))
        joints.append(np.stack([
            transform_points(evaluation_gauge, payload[identity]["joints"])
            for identity in ordered_identities
        ]))
        vertices.append(np.stack([
            transform_points(evaluation_gauge, payload[identity]["vertices"])
            for identity in ordered_identities
        ]))
    return (
        np.asarray(gt_cameras, dtype=np.float32),
        np.asarray(roots, dtype=np.float32),
        np.asarray(joints, dtype=np.float32),
        np.asarray(vertices, dtype=np.float32),
    )


def gt_payload(
    gt_args: SimpleNamespace,
    frame: int,
    regressor: np.ndarray,
    identities: tuple[str, ...] | list[str],
) -> dict[str, dict[str, np.ndarray]]:
    """Sequence-aware evaluator GT; ``box``/``dance`` have two people."""
    output: dict[str, dict[str, np.ndarray]] = {}
    for identity in identities:
        vertices = gt.load_obj_vertices(gt.mesh_path(gt_args, str(identity), int(frame)))
        joints = np.asarray(regressor, dtype=np.float32) @ vertices
        output[str(identity)] = {"vertices": vertices, "joints": joints, "root": joints[0]}
    return output


def first_frame_evaluator_assignment(
    predicted_roots: np.ndarray,
    target_roots: dict[str, np.ndarray],
) -> tuple[list[str], np.ndarray]:
    """Evaluator-only one-time assignment; it is never fed back to runtime."""
    names = sorted(target_roots)
    targets = np.stack([np.asarray(target_roots[name], dtype=np.float64) for name in names])
    cost = np.linalg.norm(predicted_roots[:, None] - targets[None], axis=2)
    rows, columns = linear_sum_assignment(cost)
    if len(rows) != len(predicted_roots) or len(columns) != len(names):
        raise RuntimeError("Expected a complete predicted/GT first-frame assignment")
    mapping = [""] * len(predicted_roots)
    for row, column in zip(rows, columns):
        mapping[int(row)] = names[int(column)]
    return mapping, cost


def main() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    args = parse_args()
    if int(args.pre_frames) < 1 or int(args.post_frames) < 3:
        raise ValueError("Need >=1 pre frame and >=3 post frames")
    if int(args.pre_camera) == int(args.post_camera):
        raise ValueError("This cache is for a cross-shot B0+BRTC trajectory; cameras must differ")
    checkpoint = under_repo(args.checkpoint)
    output_dir = under_repo(args.output_dir)
    if not args.data_root.is_dir():
        raise FileNotFoundError(args.data_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = cache_name(args)
    npz_path, meta_path = output_dir / f"{stem}.npz", output_dir / f"{stem}.json"
    if (npz_path.exists() or meta_path.exists()) and not bool(args.overwrite):
        raise FileExistsError(f"Cache already exists: {npz_path}; pass --overwrite explicitly")

    frame = int(args.frame)
    sequence_identities = tuple(gt.SEQUENCE_IDENTITIES[str(args.sequence)])
    pre_numbers = list(range(frame - int(args.pre_frames) + 1, frame + 1))
    post_numbers = list(range(frame, frame + int(args.post_frames)))
    gt_args = SimpleNamespace(
        data_root=args.data_root,
        sequence=str(args.sequence),
        output_dir=output_dir / "input_frames" / stem,
        size=int(args.size),
    )
    pre_paths = [gt.extract_video_frame(gt_args, int(args.pre_camera), value) for value in pre_numbers]
    post_paths = [gt.extract_video_frame(gt_args, int(args.post_camera), value) for value in post_numbers]

    print(f">> CPU loading frozen P0: {stem}; pre={len(pre_paths)}, post={len(post_paths)}", flush=True)
    model = ARCroco3DStereo.from_pretrained(str(checkpoint)).to("cpu")
    flags = configure_model(model)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layer = SMPL_Layer(
        type="smplx", gender="neutral", num_betas=10, kid=False, person_center="head"
    ).to("cpu").eval()
    try:
        # --------------------------- Runtime transaction ---------------------------
        pre_views = gt.prepare_full_square_input(model, pre_paths, SimpleNamespace(size=int(args.size)))
        post_views = gt.prepare_full_square_input(model, post_paths, SimpleNamespace(size=int(args.size)))
        shadow_views = set_event_indices(copy.deepcopy(pre_views + post_views[:1]), {len(pre_views)})
        raw_views = set_event_indices(copy.deepcopy(post_views), set())
        started = time.perf_counter()
        with torch.no_grad():
            shadow_predictions, shadow_returned, shadow_debug = model.forward_recurrent_lighter(
                shadow_views, "cpu", ret_state=False, use_ttt3r=False, return_token_debug=True
            )
            raw_predictions, raw_returned, raw_debug = model.forward_recurrent_lighter(
                raw_views, "cpu", ret_state=False, use_ttt3r=False, return_token_debug=True
            )
        runtime_seconds = time.perf_counter() - started

        pre_camera = as_array(camera_matrix(shadow_predictions[-2]))
        shadow_post_camera = as_array(camera_matrix(shadow_predictions[-1]))
        raw_first_camera = as_array(camera_matrix(raw_predictions[0]))
        b0 = shadow_post_camera @ np.linalg.inv(raw_first_camera)
        b0_camera = b0 @ raw_first_camera
        shadow_parity = float(np.max(np.abs(b0_camera - shadow_post_camera)))
        if shadow_parity > 1e-5:
            raise RuntimeError(f"B0/shadow parity failure: {shadow_parity}")

        pre_people = decode_people(
            shadow_predictions[-2], shadow_returned[-2], shadow_debug[-2], layer
        )
        raw_people = [
            decode_people(prediction, returned, debug, layer)
            for prediction, returned, debug in zip(raw_predictions, raw_returned, raw_debug)
        ]
        if not pre_people or not raw_people[0]:
            raise RuntimeError("No people available at the boundary")
        b0_first_people = [transform_person(b0, item) for item in raw_people[0]]
        association = anonymous_match(pre_people, b0_first_people)
        brtc_first_people, brtc_debug = refine_matched_people(
            pre_camera, b0_camera, pre_people, b0_first_people, association["pairs"], BRTC_CONFIG
        )
        if brtc_debug.get("camera_update") != "none":
            raise RuntimeError("BRTC attempted to modify a camera")

        initial_ids = native_ids(raw_predictions[0], len(b0_first_people))
        if len(set(initial_ids)) != len(initial_ids):
            raise RuntimeError(f"Duplicate native IDs at first post frame: {initial_ids}")
        b0_by_detection = {int(item["detection_index"]): item for item in b0_first_people}
        brtc_by_detection = {int(item["detection_index"]): item for item in brtc_first_people}
        shifts_by_id = {
            int(native_id): as_array(brtc_by_detection[index]["root"]) - as_array(b0_by_detection[index]["root"])
            for index, native_id in enumerate(initial_ids)
            if index in b0_by_detection and index in brtc_by_detection
        }
        if len(shifts_by_id) != len(initial_ids):
            raise RuntimeError("A first-post BRTC translation is missing for a native track")

        b0_cameras, b0_roots, b0_joints, b0_vertices, native_id_frames = [], [], [], [], []
        for index, (prediction, returned, debug, local_people) in enumerate(
            zip(raw_predictions, raw_returned, raw_debug, raw_people)
        ):
            ids = native_ids(prediction, len(local_people))
            if set(ids) != set(initial_ids) or len(ids) != len(initial_ids):
                raise RuntimeError(
                    f"Native tracks changed at post index {index}: expected {initial_ids}, got {ids}"
                )
            by_id = {int(native_id): person for native_id, person in zip(ids, local_people)}
            people = [transform_person(b0, by_id[int(native_id)]) for native_id in initial_ids]
            b0_cameras.append(b0 @ as_array(camera_matrix(prediction)))
            b0_roots.append(np.stack([as_array(item["root"]) for item in people]))
            b0_joints.append(np.stack([as_array(item["joints"]) for item in people]))
            b0_vertices.append(np.stack([as_array(item["vertices"]) for item in people]))
            native_id_frames.append(np.asarray(initial_ids, dtype=np.int64))

        # ------------------------ Evaluator-only transaction -----------------------
        gt_pre_camera = np.linalg.inv(gt.gt_w2c(gt_args, int(args.pre_camera), frame))
        evaluation_gauge = pre_camera @ np.linalg.inv(gt_pre_camera)
        first_payload = gt_payload(
            gt_args, frame, gt.joint_regressor(layer), sequence_identities
        )
        first_target_roots = {
            identity: transform_points(evaluation_gauge, payload["root"][None])[0]
            for identity, payload in first_payload.items()
        }
        brtc_first_roots = np.stack([
            as_array(b0_by_detection[index]["root"]) + shifts_by_id[int(native_id)]
            for index, native_id in enumerate(initial_ids)
        ])
        track_to_gt, first_assignment_cost = first_frame_evaluator_assignment(
            brtc_first_roots, first_target_roots
        )
        if any(identity not in sequence_identities for identity in track_to_gt):
            raise RuntimeError(f"Unexpected evaluator identities: {track_to_gt}")
        gt_cameras, gt_roots, gt_joints, gt_vertices = gt_targets(
            gt_args, layer, str(args.sequence), post_numbers, int(args.pre_camera),
            int(args.post_camera), evaluation_gauge, track_to_gt
        )

        b0_roots_array = np.asarray(b0_roots, dtype=np.float32)
        shift_array = np.stack([shifts_by_id[int(native_id)] for native_id in initial_ids]).astype(np.float32)
        brtc_roots = b0_roots_array + shift_array[None]
        first_root_error = np.linalg.norm(brtc_roots[0] - gt_roots[0], axis=1)
        print(
            f">> tracks={initial_ids}, evaluator IDs={track_to_gt}, "
            f"first BRTC root mean={float(first_root_error.mean()):.4f}m, "
            f"runtime={runtime_seconds:.1f}s",
            flush=True,
        )

        arrays = {
            "b0_cameras": np.asarray(b0_cameras, dtype=np.float32),
            "b0_roots": b0_roots_array,
            "b0_joints": np.asarray(b0_joints, dtype=np.float32),
            "b0_vertices": np.asarray(b0_vertices, dtype=np.float32),
            "brtc_shifts_by_track": shift_array,
            "native_ids": np.asarray(native_id_frames, dtype=np.int64),
            "gt_cameras_evaluator_only": gt_cameras,
            "gt_roots_evaluator_only": gt_roots,
            "gt_joints_evaluator_only": gt_joints,
            "gt_vertices_evaluator_only": gt_vertices,
        }
        temporary = npz_path.with_name(npz_path.name + ".new")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, npz_path)
        metadata = {
            "experiment": "within_shot_stability_runtime_first_cache",
            "cache": str(npz_path),
            "case": {
                "sequence": str(args.sequence), "frame": frame,
                "pre_camera": int(args.pre_camera), "post_camera": int(args.post_camera),
                "pre_frames": pre_numbers, "post_frames": post_numbers,
            },
            "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
            "checkpoint_flags": flags,
            "runtime": {
                "device": "cpu", "cuda_visible_devices": "", "runtime_seconds": runtime_seconds,
                "future_post_frames_for_boundary": 0,
                "future_post_frames_for_brtc": 0,
                "camera_update": "none", "b0_shadow_camera_max_abs_error": shadow_parity,
                "native_ids_initial": initial_ids,
                "native_ids_all_frames_equal": bool(all(np.array_equal(row, native_id_frames[0]) for row in native_id_frames)),
                "automatic_association": association,
                "brtc": brtc_debug,
                "gt_used": False,
            },
            "evaluator_only": {
                "evaluation_gauge": evaluation_gauge,
                "track_to_gt_identity": track_to_gt,
                "first_assignment_cost_m": first_assignment_cost,
                "first_brtc_root_error_m_by_track": first_root_error,
                "gt_read_after_runtime": True,
            },
            "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
        }
        meta_path.write_text(json.dumps(jsonable(metadata), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f">> wrote {npz_path}", flush=True)
        print(f">> wrote {meta_path}", flush=True)
    finally:
        del layer, model


if __name__ == "__main__":
    main()
