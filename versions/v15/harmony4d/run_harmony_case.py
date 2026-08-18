#!/usr/bin/env python3
"""Run one RGB-only Harmony4D cross-shot case on GPU.

One strict Human3R forward and two same-checkpoint Movie3R forwards are
decoded into a compact common-SMPL cache.  M1--M7 are then derived from those
same forwards, so ablations cannot differ because of stochastic detections or
preprocessing.  This process never opens calibration, SMPL GT, or evaluator
files.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
for root in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from dust3r.model import ARCroco3DStereo  # noqa: E402
from dust3r.utils.smpl_layer import SMPL_Layer  # noqa: E402
from src.dust3r.adaptive_joint import AdaptiveJointConfig, apply_with_raw_reference  # noqa: E402
from versions.v13 import gt_id_consensus as gt_helpers  # noqa: E402
from versions.v14.b0_person_triangulation import DEFAULT_CONFIG as BRTC_CONFIG  # noqa: E402
from versions.v14.b0_person_triangulation import refine_matched_people  # noqa: E402
from versions.v14.causal_image_detector import CausalGRUShotDetector  # noqa: E402
from versions.v14.eval_streaming_within_shot_stability import POLICIES, runtime_c1  # noqa: E402
from versions.v14.probe_p1_foot_scene_observability import anonymous_match  # noqa: E402
from versions.v14.run_v14_2_single_sequence import camera_matrix, configure_model, set_event_indices  # noqa: E402
from versions.v15.harmony4d.topology import CommonTopology  # noqa: E402


SPEC_PATH = REPO_ROOT / "versions/v15/FINAL_RUNTIME_SPEC.json"
DETECTOR_PATH = REPO_ROOT / "output/v14/detector_learning_audit/SELECTED_MODEL.pt"
METHODS = (
    "m0_strict_human3r",
    "m1_clean_reset",
    "m2_no_v9_raw_se3",
    "m3_b0_only",
    "m4_b0_identity",
    "m5_b0_identity_brtc",
    "m6_b0_identity_brtc_c1",
    "m7_full_v15_oracle",
)
KNOWN_VERIFIED_ARTIFACTS = {
    "src/human3r_896L.pth": (4670554642, "1c5d89077d7734476ce74183df178c51ad172cad5e256081e61480cf231a9377"),
    "output/v14_cut_first_cross_source/v14_cut_first_cross_source_multihuman_p0_e6/checkpoint-final.pth": (
        4930639378, "de2430ed5adcfd9ba919d49f88364f964063b3d0b43848ffada709b444828265"
    ),
    "output/v14/detector_learning_audit/SELECTED_MODEL.pt": (
        24910, "cb84b0da620878515e94f08b30d757206b41c4de82e2ff4091fe2a6e519e498f"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--record-json")
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--line", type=int, default=1)
    parser.add_argument("--extracted-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--current-checkpoint", type=Path, default=None)
    parser.add_argument("--original-checkpoint", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_record(args: argparse.Namespace) -> dict[str, Any]:
    if args.record_json:
        value = json.loads(args.record_json)
    else:
        rows = [line for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        if args.line < 1 or args.line > len(rows):
            raise IndexError(f"Requested line {args.line}; manifest has {len(rows)} records")
        value = json.loads(rows[args.line - 1])
    required = {
        "case_id", "capture_relative", "pre_camera", "post_camera",
        "pre_frame_numbers", "post_frame_numbers", "boundary_index",
    }
    missing = required.difference(value)
    if missing:
        raise ValueError(f"Case misses fields: {sorted(missing)}")
    if int(value["boundary_index"]) != len(value["pre_frame_numbers"]):
        raise ValueError("boundary_index must equal the number of pre frames")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verified_artifact_sha256(path: Path) -> str:
    resolved = path.resolve()
    try:
        relative = str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return sha256(resolved)
    known = KNOWN_VERIFIED_ARTIFACTS.get(relative)
    if known is None:
        return sha256(resolved)
    expected_size, digest = known
    if resolved.stat().st_size != expected_size:
        raise RuntimeError(f"Frozen artifact size changed: {resolved}")
    return digest


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def default_checkpoints() -> tuple[Path, Path]:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    return (
        REPO_ROOT / spec["checkpoints"]["primary_multihuman"]["path"],
        REPO_ROOT / spec["checkpoints"]["original_human3r"]["path"],
    )


def strict_original(model: ARCroco3DStereo) -> None:
    for name in (
        "enable_shot_adaptation", "enable_shot_decoder_token", "enable_anchor_pose_adapter",
        "enable_anchor_decoder_tokens", "enable_anchor_pose_token_adapter", "enable_v7_pose_adapter",
        "enable_v8_pose_prompt", "enable_v8_human_trans_corr", "enable_v8_human_latent_corr",
        "enable_v8_head_lora", "enable_layerwise_pose_shot_adapter", "enable_pose_alignment_adapter",
        "enable_pose_translation_adapter", "enable_pose_lora", "enable_human_lora", "enable_world_lora",
    ):
        if hasattr(model, name):
            setattr(model, name, False)


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.asarray(points) @ np.asarray(transform)[:3, :3].T + np.asarray(transform)[:3, 3]


def transform_person(transform: np.ndarray, person: dict[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(person)
    for key in ("root", "joints", "vertices"):
        output[key] = transform_points(transform, person[key])
    output["torso"] = np.asarray(transform)[:3, :3] @ np.asarray(person["torso"])
    return output


def shift_person(person: dict[str, Any], shift: np.ndarray) -> dict[str, Any]:
    output = copy.deepcopy(person)
    for key in ("root", "joints", "vertices"):
        output[key] = np.asarray(person[key]) + np.asarray(shift)
    return output


def frame_image_paths(sequence_root: Path, record: dict[str, Any]) -> tuple[list[Path], list[Path]]:
    pre = [
        sequence_root / "exo" / str(record["pre_camera"]) / "images" / f"{int(frame):05d}.jpg"
        for frame in record["pre_frame_numbers"]
    ]
    post = [
        sequence_root / "exo" / str(record["post_camera"]) / "images" / f"{int(frame):05d}.jpg"
        for frame in record["post_frame_numbers"]
    ]
    missing = [path for path in pre + post if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    return pre, post


def run_forward(
    model: ARCroco3DStereo,
    views: list[dict[str, Any]],
    device: torch.device,
    label: str,
) -> tuple[list[dict], list[dict], list[dict], dict[str, float]]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        predictions, returned, debug = model.forward_recurrent_lighter(
            views, str(device), ret_state=False, use_ttt3r=False, return_token_debug=True
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak = float(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0.0
    print(f">> {label}: {len(views)} frames, {elapsed:.2f}s, peak={peak / 2**30:.2f}GiB", flush=True)
    return predictions, returned, debug, {
        "frames": len(views), "seconds": elapsed, "fps": len(views) / max(elapsed, 1e-9),
        "peak_vram_bytes": peak,
    }


def decode_frame(
    prediction: dict[str, Any], view: dict[str, Any], layer: SMPL_Layer, topology: CommonTopology
) -> dict[str, Any]:
    import roma

    camera = camera_matrix(prediction).astype(np.float64)
    count = int(prediction["smpl_transl"].shape[1])
    people = []
    native = prediction.get("smpl_id")
    native_ids = (
        list(range(count))
        if native is None
        else [int(value) for value in native[0, :count].detach().cpu().reshape(-1).tolist()]
    )
    if count:
        device = next(layer.parameters()).device
        rotmat = prediction["smpl_rotmat"][0, :count].to(device=device, dtype=torch.float32)
        rotvec = roma.rotmat_to_rotvec(rotmat)
        shape = prediction["smpl_shape"][0, :count].to(device=device, dtype=torch.float32)
        transl = prediction["smpl_transl"][0, :count].to(device=device, dtype=torch.float32)
        expression = prediction.get("smpl_expression")
        expression = (
            torch.zeros((count, 10), device=device)
            if expression is None
            else expression[0, :count].to(device=device, dtype=torch.float32)
        )
        intrinsic = view["K_mhmr"][0].to(device=device, dtype=torch.float32).expand(count, -1, -1)
        with torch.no_grad():
            body = layer(rotvec, shape, transl, None, None, K=intrinsic, expression=expression)
        smplx_camera = body["smpl_v3d"].detach().float().cpu().numpy()
        smpl_camera = topology.smplx_vertices_to_smpl(smplx_camera)
        joints_camera = topology.joints_from_smpl(smpl_camera)
        for index in range(count):
            joints_world = transform_points(camera, joints_camera[index])
            people.append({
                "detection_index": index,
                "native_id": native_ids[index],
                "persistent_id": native_ids[index],
                "root": joints_world[0],
                "joints": joints_world,
                "vertices": transform_points(camera, smplx_camera[index]),
                "torso": camera[:3, :3] @ gt_helpers.torso_frame(joints_camera[index]),
            })
    return {"camera": camera, "people": people}


def decode_sequence(predictions, returned, debug, layer, topology) -> list[dict[str, Any]]:
    frames = []
    for prediction, view in zip(predictions, returned):
        frames.append(decode_frame(prediction, view, layer, topology))
    return frames


def map_frames(frames: list[dict[str, Any]], transform: np.ndarray) -> list[dict[str, Any]]:
    return [
        {
            "camera": np.asarray(transform) @ np.asarray(frame["camera"]),
            "people": [transform_person(transform, person) for person in frame["people"]],
        }
        for frame in frames
    ]


def persistent_post(
    pre_last: dict[str, Any], b0_post: list[dict[str, Any]], shifts: dict[int, np.ndarray] | None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    association = anonymous_match(pre_last["people"], b0_post[0]["people"])
    used = {int(person["persistent_id"]) for person in pre_last["people"]}
    next_id = max(used, default=-1) + 1
    first_ids: dict[int, int] = {}
    for pre_index, post_index in association["pairs"]:
        first_ids[int(post_index)] = int(pre_last["people"][pre_index]["persistent_id"])
    for post_index in range(len(b0_post[0]["people"])):
        if post_index not in first_ids:
            first_ids[post_index] = next_id
            next_id += 1
    output = []
    first_reference = b0_post[0]["people"]
    for frame_index, frame in enumerate(b0_post):
        local = (
            {index: index for index in range(len(first_reference))}
            if frame_index == 0
            else {int(a): int(b) for a, b in anonymous_match(first_reference, frame["people"])["pairs"]}
        )
        people = []
        for first_index in sorted(local, key=lambda value: first_ids[value]):
            current_index = local[first_index]
            person = copy.deepcopy(frame["people"][current_index])
            persistent = int(first_ids[first_index])
            person["persistent_id"] = persistent
            if shifts is not None and persistent in shifts:
                person = shift_person(person, shifts[persistent])
            people.append(person)
        output.append({"camera": np.asarray(frame["camera"]).copy(), "people": people})
    association_out = {
        "pairs": association["pairs"],
        "matched_count": association["matched_count"],
        "cost": np.asarray(association.get("cost", [])).tolist(),
        "first_post_index_to_persistent_id": {str(k): int(v) for k, v in first_ids.items()},
    }
    return output, association_out


def apply_c1(b0_post: list[dict[str, Any]], brtc_post: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    counts = [len(frame["people"]) for frame in brtc_post]
    ids = [[int(person["persistent_id"]) for person in frame["people"]] for frame in brtc_post]
    if len(set(counts)) != 1 or any(value != ids[0] for value in ids[1:]) or not counts[0]:
        return copy.deepcopy(brtc_post), {"policy": "disabled_variable_visibility_or_identity", "gates": []}
    policy = next(item for item in POLICIES if item.name == "c1_ema25")
    b0_by_id = [
        {int(person["persistent_id"]): person for person in frame["people"]}
        for frame in b0_post
    ]
    ordered_ids = ids[0]
    b0_roots = np.stack([[b0_by_id[t][identity]["root"] for identity in ordered_ids] for t in range(len(b0_post))])
    b0_joints = np.stack([[b0_by_id[t][identity]["joints"] for identity in ordered_ids] for t in range(len(b0_post))])
    shifts = np.stack([
        np.asarray(brtc_post[0]["people"][index]["root"]) - np.asarray(b0_by_id[0][identity]["root"])
        for index, identity in enumerate(ordered_ids)
    ])
    result = runtime_c1({
        "b0_cameras": np.stack([frame["camera"] for frame in b0_post]),
        "b0_roots": b0_roots,
        "b0_joints": b0_joints,
        "brtc_shifts_by_track": shifts,
    }, policy)
    output = copy.deepcopy(brtc_post)
    for frame_index, frame in enumerate(output):
        for person_index, person in enumerate(frame["people"]):
            residual = np.asarray(result["residuals"][frame_index, person_index])
            frame["people"][person_index] = shift_person(person, residual)
    return output, {
        "policy": policy.name,
        "gates": np.asarray(result["gates"]).tolist(),
        "reasons": np.asarray(result["reasons"]).tolist(),
        "camera_max_abs_change": float(result["camera_max_abs_change"]),
    }


def pack_methods(methods: dict[str, list[dict[str, Any]]], topology: CommonTopology) -> dict[str, np.ndarray]:
    frame_count = {len(frames) for frames in methods.values()}
    if len(frame_count) != 1:
        raise ValueError(f"Method frame counts differ: {frame_count}")
    frames = next(iter(frame_count))
    people_max = max((len(frame["people"]) for value in methods.values() for frame in value), default=0)
    output: dict[str, np.ndarray] = {}
    for method, values in methods.items():
        cameras = np.stack([frame["camera"] for frame in values]).astype(np.float32)
        vertices = np.full((frames, people_max, 6890, 3), np.nan, dtype=np.float32)
        joints = np.full((frames, people_max, 24, 3), np.nan, dtype=np.float32)
        ids = np.full((frames, people_max), -1, dtype=np.int32)
        native_ids = np.full((frames, people_max), -1, dtype=np.int32)
        valid = np.zeros((frames, people_max), dtype=np.uint8)
        for frame_index, frame in enumerate(values):
            for person_index, person in enumerate(frame["people"]):
                smpl = topology.smplx_vertices_to_smpl(np.asarray(person["vertices"])[None])[0]
                vertices[frame_index, person_index] = smpl
                joints[frame_index, person_index] = topology.joints_from_smpl(smpl)
                ids[frame_index, person_index] = int(person["persistent_id"])
                native_ids[frame_index, person_index] = int(person["native_id"])
                valid[frame_index, person_index] = 1
        prefix = method + "__"
        output[prefix + "cameras_c2w"] = cameras
        output[prefix + "vertices_world"] = vertices
        output[prefix + "joints_world"] = joints
        output[prefix + "persistent_ids"] = ids
        output[prefix + "native_ids"] = native_ids
        output[prefix + "valid"] = valid
    return output


def main() -> None:
    args = parse_args()
    record = read_record(args)
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sequence_root = args.extracted_root.resolve() / str(record["capture_relative"])
    if not (sequence_root / "exo").is_dir():
        raise FileNotFoundError(sequence_root / "exo")
    pre_paths, post_paths = frame_image_paths(sequence_root, record)
    all_paths = pre_paths + post_paths
    boundary = int(record["boundary_index"])
    default_current, default_original = default_checkpoints()
    current_path = (args.current_checkpoint or default_current).resolve()
    original_path = (args.original_checkpoint or default_original).resolve()
    for path in (current_path, original_path, DETECTOR_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    topology = CommonTopology.load()
    runtime: dict[str, Any] = {}

    detector_started = time.perf_counter()
    detector = CausalGRUShotDetector(DETECTOR_PATH)
    detector_labels, detector_rows = detector.predict_sequence(all_paths)
    runtime["causal_gru_detector"] = {
        "seconds": time.perf_counter() - detector_started,
        "labels": detector_labels,
        "rows": detector_rows,
        "boundary_prediction": int(detector_labels[boundary]),
        "false_positive_indices": [i for i, value in enumerate(detector_labels) if value and i != boundary],
    }

    original_model = ARCroco3DStereo.from_pretrained(str(original_path)).to(device)
    strict_original(original_model)
    original_model.eval()
    original_layer = SMPL_Layer(type="smplx", gender="neutral", num_betas=10, kid=False, person_center="head").to(device).eval()
    original_views = set_event_indices(
        gt_helpers.prepare_full_square_input(original_model, all_paths, SimpleNamespace(size=int(args.size))),
        set(),
    )
    original_predictions, original_returned, original_debug, runtime["m0_forward"] = run_forward(
        original_model, original_views, device, "strict_original_human3r"
    )
    m0 = decode_sequence(original_predictions, original_returned, original_debug, original_layer, topology)
    del original_predictions, original_returned, original_debug, original_views, original_layer, original_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    current_model = ARCroco3DStereo.from_pretrained(str(current_path)).to(device)
    flags = configure_model(current_model)
    current_layer = SMPL_Layer(type="smplx", gender="neutral", num_betas=10, kid=False, person_center="head").to(device).eval()
    pre_views = gt_helpers.prepare_full_square_input(current_model, pre_paths, SimpleNamespace(size=int(args.size)))
    post_views = gt_helpers.prepare_full_square_input(current_model, post_paths, SimpleNamespace(size=int(args.size)))
    shadow_views = set_event_indices(copy.deepcopy(pre_views + post_views[:1]), {boundary})
    raw_post_views = set_event_indices(copy.deepcopy(post_views), set())
    shadow_predictions, shadow_returned, shadow_debug, runtime["shadow_forward"] = run_forward(
        current_model, shadow_views, device, "current_shadow"
    )
    shadow = decode_sequence(shadow_predictions, shadow_returned, shadow_debug, current_layer, topology)
    del shadow_predictions, shadow_returned, shadow_debug, shadow_views
    raw_predictions, raw_returned, raw_debug, runtime["raw_post_forward"] = run_forward(
        current_model, raw_post_views, device, "current_raw_post"
    )
    raw_post = decode_sequence(raw_predictions, raw_returned, raw_debug, current_layer, topology)
    del raw_predictions, raw_returned, raw_debug, raw_post_views, pre_views, post_views, current_layer, current_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    pre = shadow[:-1]
    shadow_first_post = shadow[-1]
    m1 = copy.deepcopy(pre + raw_post)
    raw_transform = np.asarray(pre[-1]["camera"]) @ np.linalg.inv(np.asarray(raw_post[0]["camera"]))
    raw_se3_post = map_frames(raw_post, raw_transform)
    m2 = copy.deepcopy(pre + raw_se3_post)
    b0_transform = np.asarray(shadow_first_post["camera"]) @ np.linalg.inv(np.asarray(raw_post[0]["camera"]))
    b0_post = map_frames(raw_post, b0_transform)
    m3 = copy.deepcopy(pre + b0_post)
    identity_post, association = persistent_post(pre[-1], b0_post, shifts=None)
    m4 = copy.deepcopy(pre + identity_post)

    brtc_first, brtc_debug = refine_matched_people(
        np.asarray(pre[-1]["camera"]), np.asarray(b0_post[0]["camera"]),
        pre[-1]["people"], b0_post[0]["people"], association["pairs"], BRTC_CONFIG,
    )
    first_persistent = {
        int(index): int(identity)
        for index, identity in association["first_post_index_to_persistent_id"].items()
    }
    shifts: dict[int, np.ndarray] = {}
    for post_index, (before, after) in enumerate(zip(b0_post[0]["people"], brtc_first)):
        shifts[first_persistent[post_index]] = np.asarray(after["root"]) - np.asarray(before["root"])
    brtc_post, _ = persistent_post(pre[-1], b0_post, shifts=shifts)
    m5 = copy.deepcopy(pre + brtc_post)
    c1_post, c1_debug = apply_c1(identity_post, brtc_post)
    m6 = copy.deepcopy(pre + c1_post)

    m6_cameras = np.stack([frame["camera"] for frame in m6])
    m6_meshes = [np.stack([person["vertices"] for person in frame["people"]]) for frame in m6]
    raw_reference = pre + raw_post
    raw_cameras = np.stack([frame["camera"] for frame in raw_reference])
    raw_meshes = [np.stack([person["vertices"] for person in frame["people"]]) for frame in raw_reference]
    adaptive_cameras, adaptive_meshes, _, adaptive_debug = apply_with_raw_reference(
        m6_cameras, m6_meshes, raw_cameras, raw_meshes, None, [boundary], AdaptiveJointConfig()
    )
    m7 = copy.deepcopy(m6)
    for frame_index, frame in enumerate(m7):
        frame["camera"] = adaptive_cameras[frame_index]
        if len(frame["people"]) == len(adaptive_meshes[frame_index]):
            for person, vertices in zip(frame["people"], adaptive_meshes[frame_index]):
                old_root = np.asarray(person["root"])
                smpl = topology.smplx_vertices_to_smpl(np.asarray(vertices)[None])[0]
                joints = topology.joints_from_smpl(smpl)
                person["vertices"] = np.asarray(vertices)
                person["joints"] = joints
                person["root"] = joints[0]
                # Torso is used only in the already-completed boundary gate;
                # keep a finite rotated approximation for diagnostics.
                if np.linalg.norm(old_root - joints[0]) > 0:
                    person["torso"] = np.asarray(person["torso"])

    methods = {
        "m0_strict_human3r": m0,
        "m1_clean_reset": m1,
        "m2_no_v9_raw_se3": m2,
        "m3_b0_only": m3,
        "m4_b0_identity": m4,
        "m5_b0_identity_brtc": m5,
        "m6_b0_identity_brtc_c1": m6,
        "m7_full_v15_oracle": m7,
    }
    arrays = pack_methods(methods, topology)
    temp = output.with_suffix(output.suffix + ".partial")
    with temp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temp, output)
    report = {
        "schema_version": "Movie3R-Harmony4D-runtime-cache-v1",
        "record": record,
        "methods": list(methods),
        "runtime": runtime,
        "checkpoint": {
            "current": str(current_path), "current_sha256": verified_artifact_sha256(current_path),
            "original": str(original_path), "original_sha256": verified_artifact_sha256(original_path),
            "detector": str(DETECTOR_PATH), "detector_sha256": verified_artifact_sha256(DETECTOR_PATH),
            "current_flags": flags,
        },
        "geometry": {
            "raw_se3": raw_transform,
            "b0": b0_transform,
            "association": association,
            "brtc": brtc_debug,
            "brtc_shifts_by_persistent_id": shifts,
            "c1": c1_debug,
            "adaptive": adaptive_debug,
        },
        "topology": topology.metadata(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "precision": "FP32",
        },
        "runtime_contract": {
            "gt_in_runtime": False,
            "future_frames_at_boundary": 0,
            "pre_cut_frames_mutated": False,
            "same_forward_ablations": True,
        },
        "cache": str(output),
        "cache_sha256": sha256(output),
    }
    report_path = output.with_suffix(".runtime.json")
    report_path.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "case_id": record["case_id"], "cache": str(output), "report": str(report_path),
        "methods": list(methods), "detector_boundary": detector_labels[boundary],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
