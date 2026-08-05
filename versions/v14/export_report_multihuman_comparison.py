#!/usr/bin/env python3
"""Export strict-Human3R vs frozen Movie3R demo payloads for one MultiHuman cut.

This is a visualization-only exporter.  Both variants read exactly the same
pre/post RGB frames.  The baseline is strict original Human3R carrying its
native recurrent state across the cut.  The Movie3R variant performs the
causal transaction, B0, BRTC-LC and the frozen C1-EMA25 policy.  No GT field
is opened or used by either runtime path.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from dust3r.model import ARCroco3DStereo
from dust3r.utils.smpl_layer import SMPL_Layer
from versions.v13 import gt_id_consensus as gt
from versions.v14.b0_person_triangulation import DEFAULT_CONFIG as BRTC_CONFIG
from versions.v14.b0_person_triangulation import refine_matched_people
from versions.v14.eval_streaming_within_shot_stability import POLICIES, runtime_c1
from versions.v14.export_p5_brtc_demo_payload import (
    CHECKPOINT as CURRENT_CHECKPOINT,
    FACES,
    as_array,
    background_from_prediction,
    under_repo,
)
from versions.v14.export_p5_brtc_long_demo_payload import native_ids, translated, write_payload
from versions.v14.probe_p1_foot_scene_observability import (
    anonymous_match,
    configure_model,
    decode_people,
    set_event_indices,
    transform_person,
)
from versions.v14.run_v14_2_single_sequence import camera_matrix


ORIGINAL_CHECKPOINT = REPO_ROOT / "src/human3r_896L.pth"
DATA_ROOT = Path("/data/wangzheng/iJCV-CODE/data/MultiHuman/Real-World-Capture/extracted")
AVATARREX_ROOT = Path("/data/wangzheng/iJCV-CODE/data/Training")
OUTPUT_ROOT = REPO_ROOT / "output/v14/report_comparison_viewers"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", choices=("avatarrex", "dance", "box", "three"), required=True)
    parser.add_argument("--frame", type=int, required=True, help="last pre / first post frame")
    # MultiHuman uses integer camera indices; AvatarReX uses numeric stream IDs.
    parser.add_argument("--pre-camera", type=str, required=True)
    parser.add_argument("--post-camera", type=str, required=True)
    parser.add_argument("--pre-frames", type=int, default=5)
    parser.add_argument("--post-frames", type=int, default=25)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--avatarrex-root", type=Path, default=AVATARREX_ROOT)
    parser.add_argument("--avatarrex-group", default="lbn1")
    parser.add_argument("--original-checkpoint", type=Path, default=ORIGINAL_CHECKPOINT)
    parser.add_argument("--current-checkpoint", type=Path, default=CURRENT_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
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


def record(args: argparse.Namespace) -> dict[str, int | str]:
    return {
        "sequence": str(args.sequence), "frame": int(args.frame),
        "pre_camera": str(args.pre_camera), "post_camera": str(args.post_camera),
    }


def image_paths(args: argparse.Namespace, root: Path) -> tuple[list[Path], list[Path]]:
    info = record(args)
    pre_numbers = list(range(int(args.frame) - int(args.pre_frames) + 1, int(args.frame) + 1))
    post_numbers = list(range(int(args.frame), int(args.frame) + int(args.post_frames)))
    if str(args.sequence) == "avatarrex":
        # AvatarReX RGB streams are already decoded PNGs.  Read only these
        # images; calibration/SMPL/DA3 data are intentionally not opened.
        def frame_path(camera: str, frame: int) -> Path:
            path = (
                args.avatarrex_root / str(args.avatarrex_group) / str(camera)
                / "rgb" / f"{int(frame):08d}.png"
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            return path

        pre = [frame_path(str(args.pre_camera), frame) for frame in pre_numbers]
        post = [frame_path(str(args.post_camera), frame) for frame in post_numbers]
    else:
        extraction_args = SimpleNamespace(
            data_root=args.data_root, sequence=str(args.sequence), output_dir=root / "input_frames", size=int(args.size),
        )
        pre = [gt.extract_video_frame(extraction_args, int(args.pre_camera), frame) for frame in pre_numbers]
        post = [gt.extract_video_frame(extraction_args, int(args.post_camera), frame) for frame in post_numbers]
    if not all(path.is_file() for path in pre + post):
        raise FileNotFoundError({"record": info, "pre": [str(p) for p in pre], "post": [str(p) for p in post]})
    return pre, post


def disable_movie3r(model: ARCroco3DStereo) -> None:
    """Mirror ``--strict_original_human3r`` without loading demo.py."""
    for name in (
        "enable_shot_adaptation", "enable_shot_decoder_token", "enable_anchor_pose_adapter",
        "enable_anchor_decoder_tokens", "enable_anchor_pose_token_adapter", "enable_v7_pose_adapter",
        "enable_v8_pose_prompt", "enable_v8_human_trans_corr", "enable_v8_human_latent_corr",
        "enable_v8_head_lora", "enable_layerwise_pose_shot_adapter", "enable_pose_alignment_adapter",
        "enable_pose_translation_adapter", "enable_pose_lora", "enable_human_lora", "enable_world_lora",
    ):
        if hasattr(model, name):
            setattr(model, name, False)


def read_faces(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as source:
        return np.asarray(source["f"], dtype=np.int32)


def decode_frames(
    predictions: list[dict[str, Any]], returned: list[dict[str, Any]], debug: list[dict[str, Any]],
    layer: SMPL_Layer,
) -> list[dict[str, Any]]:
    frames = []
    for prediction, view, token_debug in zip(predictions, returned, debug):
        people = decode_people(prediction, view, token_debug, layer)
        frames.append({
            "background": background_from_prediction(prediction, view),
            "camera": as_array(camera_matrix(prediction)),
            "people": people,
            "ids": native_ids(prediction, len(people)),
            # decode_people/SMPL_Layer returns camera-space geometry.  Keep the
            # fact explicit until the final demo payload conversion below.
            "people_coordinate_space": "camera",
        })
    return frames


def write_demo_payload(
    destination: Path, frames: list[dict[str, Any]], faces: np.ndarray, overwrite: bool
) -> None:
    """Write a demo payload after enforcing the camera/world contract.

    ``demo.py`` saves ``verts_world = C_cam2world @ verts_cam``.  The custom
    viewer exporter historically wrote camera-space meshes directly, which
    made post-cut meshes appear to jump even when the camera was correct.
    Frames explicitly marked ``world`` (B0/BRTC output) are not transformed a
    second time.
    """
    converted = []
    for frame in frames:
        row = dict(frame)
        people = frame.get("people", [])
        if frame.get("people_coordinate_space", "camera") == "camera":
            people = [transform_person(as_array(frame["camera"]), person) for person in people]
        row["people"] = people
        row["people_coordinate_space"] = "world"
        converted.append(row)
    write_payload(destination, converted, faces, overwrite)


def strict_original(args: argparse.Namespace, paths: list[Path], destination: Path, faces: np.ndarray) -> dict[str, Any]:
    model_path = under_repo(args.original_checkpoint)
    print(f">> strict Human3R CPU forward: {len(paths)} frames", flush=True)
    model = ARCroco3DStereo.from_pretrained(str(model_path)).to("cpu")
    disable_movie3r(model)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layer = SMPL_Layer(type="smplx", gender="neutral", num_betas=10, kid=False, person_center="head").to("cpu").eval()
    try:
        views = gt.prepare_full_square_input(model, paths, SimpleNamespace(size=int(args.size)))
        views = set_event_indices(views, set())
        started = time.perf_counter()
        with torch.no_grad():
            predictions, returned, debug = model.forward_recurrent_lighter(
                views, "cpu", ret_state=False, use_ttt3r=False, return_token_debug=True,
            )
        elapsed = time.perf_counter() - started
        frames = decode_frames(predictions, returned, debug, layer)
        write_demo_payload(destination, frames, faces, bool(args.overwrite))
        return {
            "checkpoint": str(model_path), "cpu_forward_seconds": elapsed,
            "strict_original_human3r": True, "cut_event_given_to_model": False,
            "people_per_frame": [len(frame["people"]) for frame in frames],
        }
    finally:
        del layer, model


def c1_policy() -> Any:
    policy = next((item for item in POLICIES if item.name == "c1_ema25"), None)
    if policy is None:
        raise RuntimeError("Frozen C1 policy c1_ema25 is unavailable")
    return policy


def movie3r(args: argparse.Namespace, pre_paths: list[Path], post_paths: list[Path], destination: Path, faces: np.ndarray) -> dict[str, Any]:
    checkpoint = under_repo(args.current_checkpoint)
    print(f">> Movie3R CPU forward: pre={len(pre_paths)}, post={len(post_paths)}", flush=True)
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
            shadow_predictions, shadow_returned, shadow_debug = model.forward_recurrent_lighter(
                shadow_views, "cpu", ret_state=False, use_ttt3r=False, return_token_debug=True,
            )
            raw_predictions, raw_returned, raw_debug = model.forward_recurrent_lighter(
                raw_views, "cpu", ret_state=False, use_ttt3r=False, return_token_debug=True,
            )
        elapsed = time.perf_counter() - started

        pre_frames = decode_frames(shadow_predictions[:-1], shadow_returned[:-1], shadow_debug[:-1], layer)
        raw_post = decode_frames(raw_predictions, raw_returned, raw_debug, layer)
        if not pre_frames or not raw_post:
            raise RuntimeError("Missing pre or post predictions")
        b0 = as_array(camera_matrix(shadow_predictions[-1])) @ np.linalg.inv(as_array(camera_matrix(raw_predictions[0])))
        b0_camera = b0 @ as_array(camera_matrix(raw_predictions[0]))
        camera_parity = float(np.max(np.abs(b0_camera - as_array(camera_matrix(shadow_predictions[-1])))))

        b0_post = []
        for frame in raw_post:
            raw_camera = as_array(frame["camera"])
            # raw people are camera-space.  Apply the complete raw-camera to
            # current-world map, not B0 alone.
            world_from_raw = b0 @ raw_camera
            people = [transform_person(world_from_raw, person) for person in frame["people"]]
            b0_post.append({
                **frame,
                "camera": world_from_raw,
                "people": people,
                "people_coordinate_space": "world",
            })
        counts = [len(frame["people"]) for frame in b0_post]
        if counts[0] == 0:
            raise RuntimeError(f"No people at first post frame: counts={counts}")

        association = anonymous_match(pre_frames[-1]["people"], b0_post[0]["people"])
        brtc_first, brtc_debug = refine_matched_people(
            as_array(pre_frames[-1]["camera"]), as_array(b0_post[0]["camera"]),
            pre_frames[-1]["people"], b0_post[0]["people"], association["pairs"], BRTC_CONFIG,
        )
        if brtc_debug.get("camera_update") != "none":
            raise RuntimeError("BRTC attempted to update the frozen B0 camera")
        first_ids = list(b0_post[0]["ids"])
        before = {int(person["detection_index"]): person for person in b0_post[0]["people"]}
        after = {int(person["detection_index"]): person for person in brtc_first}
        shifts = {
            int(native_id): as_array(after[index]["root"]) - as_array(before[index]["root"])
            for index, native_id in enumerate(first_ids) if index in before and index in after
        }
        if set(shifts) != set(int(item) for item in first_ids):
            raise RuntimeError(f"Missing BRTC shifts for tracks: ids={first_ids}, shifts={sorted(shifts)}")

        variable_visibility = len(set(counts)) != 1
        post_brtc = []
        for index, frame in enumerate(b0_post):
            if index == 0:
                people, ids = brtc_first, first_ids
            else:
                # Causal external identity bank: missing detections are
                # absent for this frame, never silently assigned a new ID.
                local = anonymous_match(b0_post[0]["people"], frame["people"])
                matched = [(first_ids[row], frame["people"][column]) for row, column in local["pairs"]]
                matched.sort(key=lambda item: int(item[0]))
                ids = [int(item[0]) for item in matched]
                people = translated([item[1] for item in matched], shifts, ids)
            post_brtc.append({**frame, "people": people, "ids": ids, "people_coordinate_space": "world"})

        if variable_visibility:
            c1 = {"policy": "disabled_variable_visibility", "camera_max_abs_change": 0.0}
            post_final = post_brtc
        else:
            b0_roots = np.stack([[as_array(person["root"]) for person in frame["people"]] for frame in b0_post])
            b0_joints = np.stack([[as_array(person["joints"]) for person in frame["people"]] for frame in b0_post])
            b0_vertices = np.stack([[as_array(person["vertices"]) for person in frame["people"]] for frame in b0_post])
            shift_array = np.stack([shifts[int(native_id)] for native_id in first_ids])
            c1 = runtime_c1({
                "b0_cameras": np.stack([as_array(frame["camera"]) for frame in b0_post]),
                "b0_roots": b0_roots, "b0_joints": b0_joints, "b0_vertices": b0_vertices,
                "brtc_shifts_by_track": shift_array,
            }, c1_policy())
            post_final = []
            for index, frame in enumerate(post_brtc):
                residuals = np.asarray(c1["residuals"][index], dtype=np.float64)
                people = []
                for person, residual in zip(frame["people"], residuals):
                    corrected = dict(person)
                    for key in ("root", "joints", "vertices"):
                        corrected[key] = as_array(person[key]) + residual
                    people.append(corrected)
                post_final.append({**frame, "people": people})

        # Preserve the same-checkpoint clean-reset post branch for the v15
        # adaptive camera-human gate.  It is an audit artifact, not a third
        # method result: the final viewer still comes from ``post_final``.
        # Keeping it in the standard payload format lets the gate use raw
        # body roots/rays without rerunning the expensive model forward.
        raw_destination = destination.parent / "movie3r_raw_current_human3r"
        write_demo_payload(raw_destination, pre_frames + raw_post, faces, bool(args.overwrite))

        write_demo_payload(destination, pre_frames + post_final, faces, bool(args.overwrite))
        return {
            "checkpoint": str(checkpoint), "checkpoint_flags": flags, "cpu_forward_seconds": elapsed,
            "runtime": "clean reset + shadow B0 + BRTC-LC + C1-EMA25",
            "raw_shadow_payload": str(raw_destination),
            "b0_shadow_camera_max_abs_error": camera_parity,
            "association": {"pairs": association["pairs"], "matched_count": association["matched_count"]},
            "brtc": brtc_debug,
            "c1": {
                "policy": "c1_ema25", "camera_max_abs_change": float(c1["camera_max_abs_change"]),
                "static_filtered_frames_by_native_id": ({
                    str(native_id): int(c1["gates"][:, index].sum()) for index, native_id in enumerate(first_ids)
                } if "gates" in c1 else {}),
                "variable_visibility": variable_visibility,
            },
            "people_per_post_frame": counts,
        }
    finally:
        del layer, model


def main() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    args = parse_args()
    if int(args.pre_frames) < 1 or int(args.post_frames) < 3:
        raise ValueError("Need at least one pre and three post frames")
    if str(args.pre_camera) == str(args.post_camera):
        raise ValueError("A visualization case must cross two different cameras")
    data_root = args.avatarrex_root if str(args.sequence) == "avatarrex" else args.data_root
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    output_root = under_repo(args.output_root)
    case_name = f"{args.sequence}_t{args.frame:04d}_c{args.pre_camera}_c{args.post_camera}_pre{args.pre_frames}_post{args.post_frames}"
    case_root = output_root / case_name
    faces = read_faces(under_repo(FACES))
    pre_paths, post_paths = image_paths(args, case_root)
    original = strict_original(args, pre_paths + post_paths, case_root / "original_human3r", faces)
    current = movie3r(args, pre_paths, post_paths, case_root / "movie3r_b0_brtc_c1", faces)
    manifest = {
        "format": "standard demo.py --save compatible payloads",
        "case": record(args), "frame_layout": {"pre": int(args.pre_frames), "post": int(args.post_frames), "cut_index": int(args.pre_frames)},
        "input_paths": [str(path) for path in pre_paths + post_paths],
        "baseline": original, "movie3r": current,
        "contract": {"device": "cpu", "cuda_visible_devices": "", "gt_used": False, "future_post_frames_used": 0},
    }
    (case_root / "manifest.json").write_text(json.dumps(jsonable(manifest), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f">> original: {case_root / 'original_human3r'}")
    print(f">> raw current shadow: {case_root / 'movie3r_raw_current_human3r'}")
    print(f">> movie3r: {case_root / 'movie3r_b0_brtc_c1'}")
    print(f">> manifest: {case_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
