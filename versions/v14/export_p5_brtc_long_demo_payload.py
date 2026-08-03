#!/usr/bin/env python3
"""Create a 10-frame original-demo viewer payload around one frozen P5 cut.

Display protocol (not a new evaluation): two pre-cut frames and eight post-cut
frames are replayed causally from the frozen P0 checkpoint on CPU.  The frozen
cache supplies the exact BRTC/P5 correction at the first post frame.  The same
per-native-track rigid translation is carried over to later post frames for
visual continuity only.  No later-frame metric, training, or ICLR claim is
made by this exporter.
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

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from dust3r.model import ARCroco3DStereo
from dust3r.utils.smpl_layer import SMPL_Layer
from versions.v13 import gt_id_consensus as gt
from versions.v14.export_p5_brtc_demo_payload import (
    CHECKPOINT,
    CACHE_ROOT,
    FACES,
    FEATURES,
    RIDGE_MODEL,
    as_array,
    background_from_prediction,
    load_ridge,
    p5_from_cache,
    prepare_destination,
    sha256,
    under_repo,
)
from versions.v14.probe_p1_foot_scene_observability import (
    configure_model,
    decode_people,
    set_event_indices,
    transform_person,
)
from versions.v14.run_v14_2_single_sequence import camera_matrix


DEFAULT_CASE = "confirm_dance_t0400_c1_c4_k0"
DEFAULT_OUTPUT = REPO_ROOT / "output/v14/p5_brtc_ray_residual_long_demo_payload"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--ridge-model", type=Path, default=RIDGE_MODEL)
    parser.add_argument("--faces", type=Path, default=FACES)
    parser.add_argument("--data-root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data/MultiHuman/Real-World-Capture/extracted"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pre-frames", type=int, default=2)
    parser.add_argument("--post-frames", type=int, default=8)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def native_ids(prediction: dict[str, Any], count: int) -> list[int]:
    value = prediction.get("smpl_id")
    if value is None:
        return list(range(count))
    result = [int(item) for item in value[0, :count].detach().cpu().tolist()]
    return result if len(result) == count else list(range(count))


def translated(people: list[dict[str, Any]], shifts: dict[int, np.ndarray], ids: list[int]) -> list[dict[str, Any]]:
    output = []
    for person, native_id in zip(people, ids):
        row = dict(person)
        shift = shifts.get(int(native_id), np.zeros(3, dtype=np.float64))
        for key in ("root", "joints", "vertices"):
            row[key] = as_array(person[key]) + shift
        output.append(row)
    return output


def cache_first_post_shifts(runtime: dict[str, Any], p5_people: list[dict[str, Any]], first_native_ids: list[int]) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    b0 = {int(person["detection_index"]): person for person in runtime["b0_post_people"]}
    brtc = {int(person["detection_index"]): person for person in runtime["brtc_post_people"]}
    p5 = {int(person["detection_index"]): person for person in p5_people}
    brtc_shifts, p5_shifts = {}, {}
    for detection, native_id in enumerate(first_native_ids):
        if detection not in b0 or detection not in brtc or detection not in p5:
            continue
        brtc_shifts[int(native_id)] = as_array(brtc[detection]["root"]) - as_array(b0[detection]["root"])
        p5_shifts[int(native_id)] = as_array(p5[detection]["root"]) - as_array(b0[detection]["root"])
    return brtc_shifts, p5_shifts


def write_payload(destination: Path, frames: list[dict[str, Any]], faces: np.ndarray, overwrite: bool) -> None:
    prepare_destination(destination, overwrite)
    for index, frame in enumerate(frames):
        background = frame["background"]
        people = frame["people"]
        vertices = (np.stack([as_array(person["vertices"]).astype(np.float32) for person in people])
                    if people else np.empty((0, 10475, 3), dtype=np.float32))
        ids = np.asarray(frame["ids"], dtype=np.int64)
        count = len(vertices)
        color_bgr = cv2.cvtColor((background["color"] * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(destination / "color" / f"{index:06d}.png"), color_bgr):
            raise OSError(f"Could not write color frame {index}")
        np.save(destination / "depth" / f"{index:06d}.npy", background["depth"])
        np.save(destination / "conf" / f"{index:06d}.npy", background["conf"])
        np.savez(destination / "camera" / f"{index:06d}.npz", pose=as_array(frame["camera"]).astype(np.float32), intrinsics=background["intrinsics"])
        np.savez(
            destination / "smpl" / f"{index:06d}.npz",
            scores=np.zeros_like(background["depth"], dtype=np.float32),
            msk=background["mask"],
            shape=np.zeros((count, 10), dtype=np.float32),
            rotvec=np.zeros((count, 53, 3), dtype=np.float32),
            transl=np.zeros((count, 3), dtype=np.float32),
            expression=np.zeros((count, 10), dtype=np.float32),
            smpl_id=ids,
            verts_world=vertices,
            faces=faces,
        )


def prepare_paths(args: argparse.Namespace, record: dict[str, Any], root: Path) -> tuple[list[Path], list[Path]]:
    frame, pre_camera, post_camera = (int(record[key]) for key in ("frame", "pre_camera", "post_camera"))
    pre_numbers = list(range(frame - int(args.pre_frames) + 1, frame + 1))
    post_numbers = list(range(frame, frame + int(args.post_frames)))
    cache_args = SimpleNamespace(data_root=args.data_root, sequence=str(record["sequence"]), output_dir=root / "input_frames", size=int(args.size))
    pre = [gt.extract_video_frame(cache_args, pre_camera, item) for item in pre_numbers]
    post = [gt.extract_video_frame(cache_args, post_camera, item) for item in post_numbers]
    return pre, post


def main() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    args = parse_args()
    cache_root, checkpoint, ridge_path, faces_path, output_root = (
        under_repo(args.cache_root), under_repo(args.checkpoint), under_repo(args.ridge_model),
        under_repo(args.faces), under_repo(args.output_dir),
    )
    if not args.data_root.is_dir():
        raise FileNotFoundError(args.data_root)
    cache_path = cache_root / "cases" / f"{args.case}.pt"
    cached = torch.load(cache_path, map_location="cpu", weights_only=False)
    if cached.get("status") != "ok":
        raise RuntimeError(f"Invalid cache: {args.case}")
    runtime = cached["runtime"]
    ridge = load_ridge(ridge_path)
    p5_cached, residuals = p5_from_cache(runtime, ridge)
    with np.load(faces_path, allow_pickle=False) as payload:
        faces = np.asarray(payload["f"], dtype=np.int32)

    input_root = output_root / str(args.case)
    pre_paths, post_paths = prepare_paths(args, runtime["record"], input_root)
    print(f">> CPU loading frozen P0; pre={len(pre_paths)}, post={len(post_paths)}", flush=True)
    model = ARCroco3DStereo.from_pretrained(str(checkpoint)).to("cpu")
    flags = configure_model(model)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layer = SMPL_Layer(type="smplx", gender="neutral", num_betas=10, kid=False, person_center="head").to("cpu").eval()
    try:
        pre_views = gt.prepare_full_square_input(model, pre_paths, SimpleNamespace(size=int(args.size)))
        post_views = gt.prepare_full_square_input(model, post_paths, SimpleNamespace(size=int(args.size)))
        shadow_views = set_event_indices(copy.deepcopy(pre_views + post_views[:1]), {len(pre_views)})
        raw_views = set_event_indices(copy.deepcopy(post_views), set())
        started = time.perf_counter()
        with torch.no_grad():
            shadow_predictions, shadow_returned, shadow_debug = model.forward_recurrent_lighter(shadow_views, "cpu", ret_state=False, use_ttt3r=False, return_token_debug=True)
            raw_predictions, raw_returned, raw_debug = model.forward_recurrent_lighter(raw_views, "cpu", ret_state=False, use_ttt3r=False, return_token_debug=True)
        elapsed = time.perf_counter() - started

        last_pre_index = len(pre_paths) - 1
        computed_last_pre = as_array(camera_matrix(shadow_predictions[last_pre_index]))
        pre_alignment = as_array(runtime["pre_camera_c2w"]) @ np.linalg.inv(computed_last_pre)
        computed_b0 = as_array(camera_matrix(shadow_predictions[-1])) @ np.linalg.inv(as_array(camera_matrix(raw_predictions[0])))
        b0_cache_error = float(np.max(np.abs(computed_b0 - as_array(runtime["b0"]))))

        pre_frames = []
        for index in range(len(pre_paths)):
            people = [transform_person(pre_alignment, person) for person in decode_people(shadow_predictions[index], shadow_returned[index], shadow_debug[index], layer)]
            pre_frames.append({
                "background": background_from_prediction(shadow_predictions[index], shadow_returned[index]),
                "camera": pre_alignment @ as_array(camera_matrix(shadow_predictions[index])),
                "people": people,
                "ids": native_ids(shadow_predictions[index], len(people)),
            })
        # Enforce the frozen confirmation gauge at the displayed last-pre frame.
        pre_frames[-1]["camera"] = as_array(runtime["pre_camera_c2w"])

        raw_post_people = [decode_people(prediction, view, debug, layer) for prediction, view, debug in zip(raw_predictions, raw_returned, raw_debug)]
        first_native = native_ids(raw_predictions[0], len(raw_post_people[0]))
        brtc_shifts, p5_shifts = cache_first_post_shifts(runtime, p5_cached, first_native)
        post_brtc, post_p5 = [], []
        frozen_brtc = {int(person["detection_index"]): person for person in runtime["brtc_post_people"]}
        frozen_p5 = {int(person["detection_index"]): person for person in p5_cached}
        for index, (prediction, returned, debug, local_people) in enumerate(zip(raw_predictions, raw_returned, raw_debug, raw_post_people)):
            b0_people = [transform_person(as_array(runtime["b0"]), person) for person in local_people]
            ids = native_ids(prediction, len(b0_people))
            if index == 0:
                brtc_people = [dict(frozen_brtc.get(int(person["detection_index"]), person)) for person in b0_people]
                p5_people = [dict(frozen_p5.get(int(person["detection_index"]), person)) for person in b0_people]
                camera = as_array(runtime["b0_camera_c2w"])
            else:
                brtc_people = translated(b0_people, brtc_shifts, ids)
                p5_people = translated(b0_people, p5_shifts, ids)
                camera = as_array(runtime["b0"]) @ as_array(camera_matrix(prediction))
            common = {
                "background": background_from_prediction(prediction, returned),
                "camera": camera,
                "ids": ids,
            }
            post_brtc.append({**common, "people": brtc_people})
            post_p5.append({**common, "people": p5_people})
        case_root = output_root / str(args.case)
        write_payload(case_root / "b0_brtc_lc", pre_frames + post_brtc, faces, bool(args.overwrite))
        write_payload(case_root / "b0_brtc_lc_p5", pre_frames + post_p5, faces, bool(args.overwrite))
    finally:
        del layer, model

    manifest = {
        "format": "standard demo.py --save compatible 10-frame payload",
        "case": runtime["record"],
        "frame_layout": {"pre": len(pre_paths), "post": len(post_paths), "cut_index": len(pre_paths)},
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint), "checkpoint_flags": flags,
        "frozen_confirmation_cache": str(cache_path), "ridge_model": str(ridge_path),
        "cpu_forward_seconds": elapsed, "cpu_recomputed_b0_max_abs_error_vs_cache": b0_cache_error,
        "first_post_p5_residual_m_by_detection": {str(key): value for key, value in residuals.items()},
        "methods": {"brtc": str(case_root / "b0_brtc_lc"), "p5": str(case_root / "b0_brtc_lc_p5")},
        "critical_scope": "first post uses exact frozen confirmation meshes. Later post frames carry those first-post per-native-track translations for visual continuity only; they are not long-stream evaluated results.",
        "runtime_contract": {"device": "cpu", "cuda_visible_devices": "", "future_post_frames_for_correction": 0, "gt_used": False},
    }
    destination = case_root / "manifest.json"
    destination.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(destination, flush=True)


if __name__ == "__main__":
    main()
