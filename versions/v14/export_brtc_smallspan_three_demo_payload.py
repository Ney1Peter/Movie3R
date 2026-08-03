#!/usr/bin/env python3
"""Export a CPU-only small-span three-person B0/BRTC-LC viewer.

This is a display exporter, not a new training or evaluation run.  It replays
the causal current-P0 transaction on RGB only:

    two pre-cut frames + first post frame -> shadow B0
    post-cut stream from reset             -> committed local reconstruction
    automatic anonymous matching           -> frozen BRTC-LC translations

The selected ``three_t1100_c4_c5_k0`` cut has a 42.66 degree calibrated
camera span, substantially smaller than the existing 125.16 degree three-person
stress-test viewer.  The same BRTC shift estimated at the first post frame is
carried by native Human3R track ID across later post frames solely to display
the online stream continuously.
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
from versions.v14.export_p5_brtc_demo_payload import (
    CHECKPOINT,
    FACES,
    as_array,
    background_from_prediction,
    sha256,
    under_repo,
)
from versions.v14.export_p5_brtc_long_demo_payload import (
    native_ids,
    translated,
    write_payload,
)
from versions.v14.probe_p1_foot_scene_observability import (
    anonymous_match,
    configure_model,
    decode_people,
    set_event_indices,
    transform_person,
)
from versions.v14.b0_person_triangulation import (
    DEFAULT_CONFIG as BRTC_CONFIG,
    refine_matched_people,
)
from versions.v14.run_v14_2_single_sequence import camera_matrix


DEFAULT_RECORD = {
    "event_id": "three_t1100_c4_c5_k0",
    "sequence": "three",
    "frame": 1100,
    "pre_camera": 4,
    "post_camera": 5,
}
DEFAULT_OUTPUT = REPO_ROOT / "output/v14/brtc_lc_smallspan_three_demo_payload"
DEFAULT_FROZEN_REPORT = (
    REPO_ROOT / "output/v14/fine_alignment_research/b0_two_view_person_triangulation/dev_three.json"
)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--faces", type=Path, default=FACES)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/data/wangzheng/iJCV-CODE/data/MultiHuman/Real-World-Capture/extracted"),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frozen-report", type=Path, default=DEFAULT_FROZEN_REPORT)
    parser.add_argument("--pre-frames", type=int, default=2)
    parser.add_argument("--post-frames", type=int, default=8)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def extract_paths(args: argparse.Namespace, record: dict[str, Any], root: Path) -> tuple[list[Path], list[Path]]:
    frame = int(record["frame"])
    pre_numbers = list(range(frame - int(args.pre_frames) + 1, frame + 1))
    post_numbers = list(range(frame, frame + int(args.post_frames)))
    gt_args = SimpleNamespace(
        data_root=args.data_root,
        sequence=str(record["sequence"]),
        output_dir=root / "input_frames",
        size=int(args.size),
    )
    pre = [gt.extract_video_frame(gt_args, int(record["pre_camera"]), value) for value in pre_numbers]
    post = [gt.extract_video_frame(gt_args, int(record["post_camera"]), value) for value in post_numbers]
    return pre, post


def shifts_by_native_id(
    b0_people: list[dict[str, Any]],
    brtc_people: list[dict[str, Any]],
    first_native_ids: list[int],
) -> dict[int, np.ndarray]:
    b0_by_detection = {int(person["detection_index"]): person for person in b0_people}
    brtc_by_detection = {int(person["detection_index"]): person for person in brtc_people}
    shifts: dict[int, np.ndarray] = {}
    for detection, native_id in enumerate(first_native_ids):
        before, after = b0_by_detection.get(detection), brtc_by_detection.get(detection)
        if before is not None and after is not None:
            shifts[int(native_id)] = as_array(after["root"]) - as_array(before["root"])
    return shifts


def frozen_selection(report_path: Path, key: str) -> dict[str, Any]:
    rows = json.loads(report_path.read_text(encoding="utf-8"))["cases"]
    matches = [row for row in rows if str(row["case"]["key"]) == key]
    if len(matches) != 1:
        raise KeyError(f"Expected exactly one frozen selection row for {key}")
    row = matches[0]
    people = row["people"]
    return {
        "camera_span_deg": float(row["camera_span_deg"]),
        "b0_camera_translation_error_m": float(row["camera"]["b0_translation_error_m"]),
        "b0_camera_rotation_error_deg": float(row["camera"]["b0_rotation_error_deg"]),
        "frozen_mean_root_error_m": {
            "b0": float(np.mean([person["baseline"]["root_error_m"] for person in people])),
            "brtc_lc": float(np.mean([person["corrected"]["root_error_m"] for person in people])),
        },
        "frozen_accepted_people": int(sum(bool(person["accepted"]) for person in people)),
    }


def main() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    args = parse_args()
    checkpoint, faces_path, output_root, report_path = (
        under_repo(args.checkpoint),
        under_repo(args.faces),
        under_repo(args.output_dir),
        under_repo(args.frozen_report),
    )
    if not args.data_root.is_dir():
        raise FileNotFoundError(args.data_root)
    if int(args.pre_frames) < 1 or int(args.post_frames) < 1:
        raise ValueError("pre/post frame counts must be positive")

    record = dict(DEFAULT_RECORD)
    selection = frozen_selection(report_path, str(record["event_id"]))
    with np.load(faces_path, allow_pickle=False) as payload:
        faces = np.asarray(payload["f"], dtype=np.int32)
    case_root = output_root / str(record["event_id"])
    pre_paths, post_paths = extract_paths(args, record, case_root)

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
            shadow_predictions, shadow_returned, shadow_debug = model.forward_recurrent_lighter(
                shadow_views, "cpu", ret_state=False, use_ttt3r=False, return_token_debug=True
            )
            raw_predictions, raw_returned, raw_debug = model.forward_recurrent_lighter(
                raw_views, "cpu", ret_state=False, use_ttt3r=False, return_token_debug=True
            )
        elapsed = time.perf_counter() - started

        b0 = as_array(camera_matrix(shadow_predictions[-1])) @ np.linalg.inv(as_array(camera_matrix(raw_predictions[0])))
        b0_camera = b0 @ as_array(camera_matrix(raw_predictions[0]))
        shadow_parity = float(np.max(np.abs(b0_camera - as_array(camera_matrix(shadow_predictions[-1])))))
        pre_frames = []
        for prediction, returned, debug in zip(shadow_predictions[:-1], shadow_returned[:-1], shadow_debug[:-1]):
            people = decode_people(prediction, returned, debug, layer)
            pre_frames.append({
                "background": background_from_prediction(prediction, returned),
                "camera": as_array(camera_matrix(prediction)),
                "people": people,
                "ids": native_ids(prediction, len(people)),
            })

        raw_post_people = [
            decode_people(prediction, returned, debug, layer)
            for prediction, returned, debug in zip(raw_predictions, raw_returned, raw_debug)
        ]
        if len(pre_frames[-1]["people"]) != 3 or len(raw_post_people[0]) != 3:
            raise RuntimeError(
                f"Expected three people at the cut, got pre={len(pre_frames[-1]['people'])}, post={len(raw_post_people[0])}"
            )
        b0_first_people = [transform_person(b0, person) for person in raw_post_people[0]]
        association = anonymous_match(pre_frames[-1]["people"], b0_first_people)
        if int(association["matched_count"]) != 3:
            raise RuntimeError(f"Expected 3 automatic matches, got {association['matched_count']}")
        brtc_first_people, brtc_debug = refine_matched_people(
            as_array(pre_frames[-1]["camera"]), b0_camera, pre_frames[-1]["people"],
            b0_first_people, association["pairs"], BRTC_CONFIG,
        )
        if brtc_debug.get("camera_update") != "none":
            raise RuntimeError("BRTC attempted to edit a camera")
        first_native_ids = native_ids(raw_predictions[0], len(b0_first_people))
        shifts = shifts_by_native_id(b0_first_people, brtc_first_people, first_native_ids)
        if len(shifts) != 3:
            raise RuntimeError(f"Expected BRTC shifts for three native tracks, got {sorted(shifts)}")

        post_b0, post_brtc = [], []
        post_counts = []
        for index, (prediction, returned, local_people) in enumerate(zip(raw_predictions, raw_returned, raw_post_people)):
            b0_people = [transform_person(b0, person) for person in local_people]
            ids = native_ids(prediction, len(b0_people))
            post_counts.append(len(b0_people))
            if len(b0_people) != 3:
                raise RuntimeError(f"Expected three people at post frame {index}, got {len(b0_people)}")
            camera = b0 @ as_array(camera_matrix(prediction))
            common = {"background": background_from_prediction(prediction, returned), "camera": camera, "ids": ids}
            post_b0.append({**common, "people": b0_people})
            corrected = brtc_first_people if index == 0 else translated(b0_people, shifts, ids)
            post_brtc.append({**common, "people": corrected})

        write_payload(case_root / "b0_frozen", pre_frames + post_b0, faces, bool(args.overwrite))
        write_payload(case_root / "b0_brtc_lc", pre_frames + post_brtc, faces, bool(args.overwrite))
    finally:
        del layer, model

    manifest = {
        "format": f"standard demo.py --save compatible {len(pre_paths) + len(post_paths)}-frame payload",
        "case": record,
        "frame_layout": {"pre": len(pre_paths), "post": len(post_paths), "cut_index": len(pre_paths)},
        "selection_from_frozen_three_person_report": selection,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_flags": flags,
        "cpu_forward_seconds": elapsed,
        "b0_shadow_camera_max_abs_error": shadow_parity,
        "first_post_runtime": {
            "automatic_association": {"pairs": association["pairs"], "matched_count": association["matched_count"]},
            "brtc": brtc_debug,
            "brtc_shift_world_by_native_track": {str(key): value.tolist() for key, value in shifts.items()},
        },
        "post_person_counts": post_counts,
        "methods": {"b0": str(case_root / "b0_frozen"), "brtc_lc": str(case_root / "b0_brtc_lc")},
        "scope": "At runtime only RGB, current-P0, automatic anonymous association and BRTC-LC are used. The frozen report is selection metadata only; no GT is read during payload creation. Later post frames carry the first-post native-track BRTC translation only for viewer continuity.",
        "runtime_contract": {"device": "cpu", "cuda_visible_devices": "", "future_post_frames_for_correction": 0, "gt_used": False},
    }
    destination = case_root / "manifest.json"
    destination.write_text(json.dumps(jsonable(manifest), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(destination, flush=True)


if __name__ == "__main__":
    main()
