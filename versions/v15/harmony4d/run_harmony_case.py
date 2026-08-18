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
import resource
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
from v10_image_only_detector import StreamingImageOnlyShotDetector  # noqa: E402


SPEC_PATH = REPO_ROOT / "versions/v15/FINAL_RUNTIME_SPEC.json"
DETECTOR_PATH = REPO_ROOT / "output/v14/detector_learning_audit/SELECTED_MODEL.pt"
STATIC_DETECTOR_CSV = (
    REPO_ROOT
    / "output/archive/20260721/v10_detector_probe/image_feature_round1/detector_pair_features.csv"
)
METHODS = (
    "m0_strict_human3r",
    "m1_clean_reset",
    "m2_no_v9_raw_se3",
    "m3_b0_only",
    "m4_b0_identity",
    "m5_b0_identity_brtc",
    "m6_b0_identity_brtc_c1",
    "m7_full_v15_oracle",
    "m8_full_v15_causal_gru",
    "m9_full_v15_static_logistic",
    "m10_observability_safe_oracle",
    "m11_observability_safe_causal_gru",
    "m12_observability_safe_static_logistic",
    "m13_b0_boundary_permutation_id",
    "m14_safe_boundary_permutation_oracle",
    "m15_safe_boundary_permutation_causal_gru",
    "m16_safe_boundary_permutation_static_logistic",
)
KNOWN_VERIFIED_ARTIFACTS = {
    "src/human3r_896L.pth": (4670554642, "1c5d89077d7734476ce74183df178c51ad172cad5e256081e61480cf231a9377"),
    "output/v14_cut_first_cross_source/v14_cut_first_cross_source_multihuman_p0_e6/checkpoint-final.pth": (
        4930639378, "de2430ed5adcfd9ba919d49f88364f964063b3d0b43848ffada709b444828265"
    ),
    "output/v14/detector_learning_audit/SELECTED_MODEL.pt": (
        24910, "cb84b0da620878515e94f08b30d757206b41c4de82e2ff4091fe2a6e519e498f"
    ),
    "output/archive/20260721/v10_detector_probe/image_feature_round1/detector_pair_features.csv": (
        234688, "208a448257d6d5970f020beaa7a265c8f257fe227ab2dbd20bd9c07e78b10f4d"
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


def boundary_permutation_post(
    pre_last: dict[str, Any],
    b0_post: list[dict[str, Any]],
    association: dict[str, Any],
    shifts: dict[int, np.ndarray] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Permute slots once at the cut, then preserve causal backbone slots.

    Human3R's recurrent state already keeps detection slots reasonably stable
    inside a shot.  Re-solving a global assignment against the first post frame
    at every timestep caused identity churn and silently dropped unmatched
    detections under close interaction.  This branch uses B0 geometry for one
    cross-shot permutation only; all later frames retain their native causal
    slot, and every detection is preserved.
    """

    if not b0_post:
        return [], {"native_slot_to_persistent_id": {}, "new_track_count": 0}
    used = {int(person["persistent_id"]) for person in pre_last["people"]}
    next_id = max(used, default=-1) + 1
    native_to_persistent: dict[int, int] = {}
    for pre_index, post_index in association.get("pairs", []):
        native = int(b0_post[0]["people"][int(post_index)]["native_id"])
        native_to_persistent[native] = int(
            pre_last["people"][int(pre_index)]["persistent_id"]
        )
    output = []
    new_slots = []
    for frame_index, frame in enumerate(b0_post):
        people = []
        for person in frame["people"]:
            value = copy.deepcopy(person)
            native = int(value["native_id"])
            if native not in native_to_persistent:
                native_to_persistent[native] = next_id
                new_slots.append({
                    "frame_index": frame_index,
                    "native_id": native,
                    "persistent_id": next_id,
                })
                next_id += 1
            persistent = native_to_persistent[native]
            value["persistent_id"] = persistent
            if shifts is not None and persistent in shifts:
                value = shift_person(value, shifts[persistent])
            people.append(value)
        output.append({"camera": np.asarray(frame["camera"]).copy(), "people": people})
    return output, {
        "policy": "single_B0_boundary_permutation_then_causal_native_slot",
        "native_slot_to_persistent_id": {
            str(key): int(value) for key, value in sorted(native_to_persistent.items())
        },
        "new_slots": new_slots,
        "new_track_count": next_id,
        "detections_preserved": True,
    }


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


def observability_safe_brtc_gate(
    pre_count: int,
    post_count: int,
    association: dict[str, Any],
    brtc_debug: dict[str, Any],
) -> dict[str, Any]:
    """Train/dev-frozen guard for applying the explicit BRTC correction.

    The guard uses only runtime geometry.  It requires a bijective association,
    a bounded shared displacement, tight cross-view ray agreement, and a real
    reduction of the observable layout objective.  Rejection is an exact
    fallback to B0 plus persistent identity; it never drops the case.
    """

    matched = int(association.get("matched_count", len(association.get("pairs", []))))
    square_full = bool(pre_count > 0 and pre_count == post_count == matched)
    group_shift_norm = float(
        np.linalg.norm(np.asarray(brtc_debug.get("group_shift_world", [np.inf] * 3)))
    )
    people = list(brtc_debug.get("people", []))
    median_gaps = [
        float(person.get("evidence", {}).get("median_gap_m", np.inf))
        for person in people
    ]
    max_median_gap = max(median_gaps, default=float("inf"))
    objectives = {
        float(key): float(value)
        for key, value in brtc_debug.get("observable_layout_objective_by_lambda", {}).items()
    }
    baseline = objectives.get(0.0, float("inf"))
    selected_lambda = float(brtc_debug.get("selected_residual_lambda", 0.0))
    selected = objectives.get(selected_lambda, float("inf"))
    relative_gain = (
        float((baseline - selected) / baseline)
        if np.isfinite(baseline) and baseline > 1e-9 and np.isfinite(selected)
        else float("-inf")
    )
    all_people_accepted = bool(
        len(people) == matched
        and int(brtc_debug.get("accepted_count", 0)) == matched
        and all(bool(person.get("accepted", False)) for person in people)
    )
    checks = {
        "square_full_association": square_full,
        "all_people_accepted": all_people_accepted,
        "group_shift_norm_le_0p15m": group_shift_norm <= 0.15,
        "max_median_ray_gap_le_0p10m": max_median_gap <= 0.10,
        "observable_layout_relative_gain_ge_0p10": relative_gain >= 0.10,
    }
    return {
        "accepted": bool(all(checks.values())),
        "checks": checks,
        "pre_count": int(pre_count),
        "post_count": int(post_count),
        "matched_count": matched,
        "group_shift_norm_m": group_shift_norm,
        "max_median_ray_gap_m": max_median_gap,
        "observable_layout_baseline": baseline,
        "observable_layout_selected": selected,
        "observable_layout_relative_gain": relative_gain,
        "threshold_provenance": "frozen_on_H4D_train01_before_train03/train15_full_dev",
        "fallback": "exact_B0_plus_persistent_ID_then_C1_and_adaptive",
    }


def compose_transaction(
    pre: list[dict[str, Any]],
    shadow_first_post: dict[str, Any],
    raw_post: list[dict[str, Any]],
    boundary: int,
    topology: CommonTopology,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Compose all explicit branches from one event and one reset forward."""

    m1 = copy.deepcopy(pre + raw_post)
    raw_transform = np.asarray(pre[-1]["camera"]) @ np.linalg.inv(np.asarray(raw_post[0]["camera"]))
    raw_se3_post = map_frames(raw_post, raw_transform)
    m2 = copy.deepcopy(pre + raw_se3_post)
    b0_transform = np.asarray(shadow_first_post["camera"]) @ np.linalg.inv(np.asarray(raw_post[0]["camera"]))
    b0_post = map_frames(raw_post, b0_transform)
    m3 = copy.deepcopy(pre + b0_post)
    identity_post, association = persistent_post(pre[-1], b0_post, shifts=None)
    m4 = copy.deepcopy(pre + identity_post)
    boundary_post, boundary_identity_debug = boundary_permutation_post(
        pre[-1], b0_post, association, shifts=None
    )
    m13 = copy.deepcopy(pre + boundary_post)

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
                smpl = topology.smplx_vertices_to_smpl(np.asarray(vertices)[None])[0]
                joints = topology.joints_from_smpl(smpl)
                person["vertices"] = np.asarray(vertices)
                person["joints"] = joints
                person["root"] = joints[0]

    safe_gate = observability_safe_brtc_gate(
        len(pre[-1]["people"]), len(b0_post[0]["people"]), association, brtc_debug
    )
    safe_brtc_post = brtc_post if safe_gate["accepted"] else identity_post
    safe_c1_post, safe_c1_debug = apply_c1(identity_post, safe_brtc_post)
    safe_base = copy.deepcopy(pre + safe_c1_post)
    safe_cameras = np.stack([frame["camera"] for frame in safe_base])
    safe_meshes = [np.stack([person["vertices"] for person in frame["people"]]) for frame in safe_base]
    safe_adaptive_cameras, safe_adaptive_meshes, _, safe_adaptive_debug = apply_with_raw_reference(
        safe_cameras, safe_meshes, raw_cameras, raw_meshes, None, [boundary], AdaptiveJointConfig()
    )
    m10 = copy.deepcopy(safe_base)
    for frame_index, frame in enumerate(m10):
        frame["camera"] = safe_adaptive_cameras[frame_index]
        if len(frame["people"]) == len(safe_adaptive_meshes[frame_index]):
            for person, vertices in zip(frame["people"], safe_adaptive_meshes[frame_index]):
                smpl = topology.smplx_vertices_to_smpl(np.asarray(vertices)[None])[0]
                joints = topology.joints_from_smpl(smpl)
                person["vertices"] = np.asarray(vertices)
                person["joints"] = joints
                person["root"] = joints[0]

    boundary_brtc_post, _ = boundary_permutation_post(
        pre[-1], b0_post, association, shifts=shifts
    )
    boundary_safe_brtc_post = (
        boundary_brtc_post if safe_gate["accepted"] else boundary_post
    )
    boundary_safe_c1_post, boundary_safe_c1_debug = apply_c1(
        boundary_post, boundary_safe_brtc_post
    )
    boundary_safe_base = copy.deepcopy(pre + boundary_safe_c1_post)
    boundary_safe_cameras = np.stack([frame["camera"] for frame in boundary_safe_base])
    boundary_safe_meshes = [
        np.stack([person["vertices"] for person in frame["people"]])
        for frame in boundary_safe_base
    ]
    boundary_safe_adaptive_cameras, boundary_safe_adaptive_meshes, _, boundary_safe_adaptive_debug = (
        apply_with_raw_reference(
            boundary_safe_cameras, boundary_safe_meshes, raw_cameras, raw_meshes,
            None, [boundary], AdaptiveJointConfig()
        )
    )
    m14 = copy.deepcopy(boundary_safe_base)
    for frame_index, frame in enumerate(m14):
        frame["camera"] = boundary_safe_adaptive_cameras[frame_index]
        if len(frame["people"]) == len(boundary_safe_adaptive_meshes[frame_index]):
            for person, vertices in zip(
                frame["people"], boundary_safe_adaptive_meshes[frame_index]
            ):
                smpl = topology.smplx_vertices_to_smpl(np.asarray(vertices)[None])[0]
                joints = topology.joints_from_smpl(smpl)
                person["vertices"] = np.asarray(vertices)
                person["joints"] = joints
                person["root"] = joints[0]

    return {
        "m1_clean_reset": m1,
        "m2_no_v9_raw_se3": m2,
        "m3_b0_only": m3,
        "m4_b0_identity": m4,
        "m5_b0_identity_brtc": m5,
        "m6_b0_identity_brtc_c1": m6,
        "m7_full_v15_oracle": m7,
        "m10_observability_safe_oracle": m10,
        "m13_b0_boundary_permutation_id": m13,
        "m14_safe_boundary_permutation_oracle": m14,
    }, {
        "raw_se3": raw_transform,
        "b0": b0_transform,
        "association": association,
        "brtc": brtc_debug,
        "brtc_shifts_by_persistent_id": shifts,
        "c1": c1_debug,
        "adaptive": adaptive_debug,
        "observability_safe_brtc": safe_gate,
        "observability_safe_c1": safe_c1_debug,
        "observability_safe_adaptive": safe_adaptive_debug,
        "boundary_permutation_identity": boundary_identity_debug,
        "boundary_permutation_safe_c1": boundary_safe_c1_debug,
        "boundary_permutation_safe_adaptive": boundary_safe_adaptive_debug,
    }


def run_transaction(
    model: ARCroco3DStereo,
    layer: SMPL_Layer,
    topology: CommonTopology,
    paths: list[Path],
    boundary: int,
    device: torch.device,
    size: int,
    label: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], dict[str, Any]]:
    """Materialise one causal first-positive boundary proposal."""

    if boundary <= 0 or boundary >= len(paths):
        raise ValueError(f"Invalid transaction boundary {boundary} for {len(paths)} frames")
    pre_paths, post_paths = paths[:boundary], paths[boundary:]
    pre_views = gt_helpers.prepare_full_square_input(model, pre_paths, SimpleNamespace(size=int(size)))
    post_views = gt_helpers.prepare_full_square_input(model, post_paths, SimpleNamespace(size=int(size)))
    shadow_views = set_event_indices(copy.deepcopy(pre_views + post_views[:1]), {boundary})
    raw_post_views = set_event_indices(copy.deepcopy(post_views), set())
    shadow_predictions, shadow_returned, shadow_debug, shadow_runtime = run_forward(
        model, shadow_views, device, f"{label}_shadow"
    )
    shadow = decode_sequence(shadow_predictions, shadow_returned, shadow_debug, layer, topology)
    del shadow_predictions, shadow_returned, shadow_debug, shadow_views
    raw_predictions, raw_returned, raw_debug, raw_runtime = run_forward(
        model, raw_post_views, device, f"{label}_raw_post"
    )
    raw_post = decode_sequence(raw_predictions, raw_returned, raw_debug, layer, topology)
    del raw_predictions, raw_returned, raw_debug, raw_post_views, pre_views, post_views
    geometry_started = time.perf_counter()
    methods, geometry = compose_transaction(shadow[:-1], shadow[-1], raw_post, boundary, topology)
    geometry_seconds = time.perf_counter() - geometry_started
    return methods, geometry, {
        "shadow_forward": shadow_runtime,
        "raw_post_forward": raw_runtime,
        "explicit_geometry_seconds": geometry_seconds,
    }


def run_no_event(
    model: ARCroco3DStereo,
    layer: SMPL_Layer,
    topology: CommonTopology,
    paths: list[Path],
    device: torch.device,
    size: int,
    label: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    views = set_event_indices(
        gt_helpers.prepare_full_square_input(model, paths, SimpleNamespace(size=int(size))), set()
    )
    predictions, returned, debug, runtime = run_forward(model, views, device, label)
    frames = decode_sequence(predictions, returned, debug, layer, topology)
    del predictions, returned, debug, views
    return frames, runtime


def first_positive(labels: list[int]) -> int | None:
    return next((index for index, value in enumerate(labels) if int(value) == 1), None)


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
    process_started = time.perf_counter()
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
    for path in (current_path, original_path, DETECTOR_PATH, STATIC_DETECTOR_CSV):
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
        "first_positive_index": first_positive(detector_labels),
        "deployment_policy": "first_positive_only_for_preregistered_two-shot_protocol",
    }
    static_started = time.perf_counter()
    static_detector = StreamingImageOnlyShotDetector(STATIC_DETECTOR_CSV)
    static_labels, static_rows = static_detector.predict_sequence(all_paths)
    runtime["static_logistic_detector"] = {
        "seconds": time.perf_counter() - static_started,
        "labels": static_labels,
        "rows": static_rows,
        "boundary_prediction": int(static_labels[boundary]),
        "false_positive_indices": [i for i, value in enumerate(static_labels) if value and i != boundary],
        "first_positive_index": first_positive(static_labels),
        "deployment_policy": "first_positive_only_for_preregistered_two-shot_protocol",
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
    current_model.eval()
    current_layer = SMPL_Layer(type="smplx", gender="neutral", num_betas=10, kid=False, person_center="head").to(device).eval()
    oracle_methods, oracle_geometry, oracle_runtime = run_transaction(
        current_model, current_layer, topology, all_paths, boundary, device, int(args.size), "oracle"
    )
    runtime["oracle_shadow_forward"] = oracle_runtime["shadow_forward"]
    runtime["oracle_raw_post_forward"] = oracle_runtime["raw_post_forward"]
    runtime["oracle_explicit_geometry_seconds"] = oracle_runtime["explicit_geometry_seconds"]
    methods = {"m0_strict_human3r": m0, **oracle_methods}

    materialized: dict[
        int | None, tuple[dict[str, list[dict[str, Any]]], dict[str, Any], dict[str, Any]]
    ] = {
        boundary: (oracle_methods, oracle_geometry, {"reused_oracle": True})
    }
    detector_geometry: dict[str, Any] = {}
    for detector_name, labels, method_name in (
        ("causal_gru", detector_labels, "m8_full_v15_causal_gru"),
        ("static_logistic", static_labels, "m9_full_v15_static_logistic"),
    ):
        proposal = first_positive(labels)
        if proposal not in materialized:
            if proposal is None:
                frames, forward_runtime = run_no_event(
                    current_model, current_layer, topology, all_paths, device, int(args.size),
                    f"{detector_name}_no_event",
                )
                materialized[proposal] = (
                    {
                        "m7_full_v15_oracle": frames,
                        "m10_observability_safe_oracle": frames,
                        "m14_safe_boundary_permutation_oracle": frames,
                    },
                    {"event": None, "reason": "detector_emitted_no_positive"},
                    {"no_event_forward": forward_runtime},
                )
            else:
                candidate_methods, candidate_geometry, candidate_runtime = run_transaction(
                    current_model, current_layer, topology, all_paths, proposal, device,
                    int(args.size), detector_name,
                )
                materialized[proposal] = (
                    candidate_methods, candidate_geometry, candidate_runtime
                )
        candidate_methods, geometry, detector_runtime = materialized[proposal]
        methods[method_name] = copy.deepcopy(candidate_methods["m7_full_v15_oracle"])
        safe_method_name = {
            "causal_gru": "m11_observability_safe_causal_gru",
            "static_logistic": "m12_observability_safe_static_logistic",
        }[detector_name]
        methods[safe_method_name] = copy.deepcopy(
            candidate_methods["m10_observability_safe_oracle"]
        )
        boundary_safe_method_name = {
            "causal_gru": "m15_safe_boundary_permutation_causal_gru",
            "static_logistic": "m16_safe_boundary_permutation_static_logistic",
        }[detector_name]
        methods[boundary_safe_method_name] = copy.deepcopy(
            candidate_methods["m14_safe_boundary_permutation_oracle"]
        )
        detector_geometry[detector_name] = {
            "first_positive_index": proposal,
            "matches_oracle_boundary": proposal == boundary,
            "geometry": geometry,
            "runtime": detector_runtime,
        }
    del current_layer, current_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
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
            "static_detector_training_csv": str(STATIC_DETECTOR_CSV),
            "static_detector_training_csv_sha256": verified_artifact_sha256(STATIC_DETECTOR_CSV),
            "current_flags": flags,
        },
        "geometry": {**oracle_geometry, "detector_driven": detector_geometry},
        "topology": topology.metadata(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "precision": "FP32",
            "process_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        },
        "total_process_seconds": time.perf_counter() - process_started,
        "runtime_contract": {
            "gt_in_runtime": False,
            "future_frames_at_boundary": 0,
            "pre_cut_frames_mutated": False,
            "same_forward_ablations": True,
            "detector_deployment_policy": "first positive only; H4D-CS150 contains exactly two preregistered shots",
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
