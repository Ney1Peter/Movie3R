#!/usr/bin/env python3
"""P1 cache and falsifiable foot--scene root-residual observability probe.

The builder is deliberately ordered as a runtime transaction:

    RGB -> shadow/raw Human3R -> B0 -> anonymous match -> frozen BRTC
        -> compact local foot patches
        -> [only now] evaluator GT labels/meshes

The diagnostic separately computes every proposal from ``runtime`` fields
before opening ``evaluator``.  It never modifies a camera or Boundary, and its
only counterfactual applies a bounded translation to an already BRTC-accepted
post person.  See V14_P1_FOOT_SCENE_OBSERVABILITY_PROTOCOL_20260803.md.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import roma
import torch
from scipy.ndimage import binary_dilation
from scipy.optimize import linear_sum_assignment


REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from dust3r.model import ARCroco3DStereo  # noqa: E402
from dust3r.utils.smpl_layer import SMPL_Layer  # noqa: E402
from dust3r.v14_outputs import boundary_from_camera_predictions  # noqa: E402
from versions.v13 import gt_id_consensus as gt  # noqa: E402
from versions.v14.b0_person_triangulation import (  # noqa: E402
    DEFAULT_CONFIG as BRTC_CONFIG,
    refine_matched_people,
)
from versions.v14.probe_b0_identity_matching import (  # noqa: E402
    identity_cost_components,
    matching_costs,
)
from versions.v14.run_v14_2_single_sequence import (  # noqa: E402
    camera_matrix,
    configure_model,
    set_event_indices,
)


DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "output/v14_cut_first_cross_source/v14_cut_first_cross_source_multihuman_p0_e6"
    / "checkpoint-final.pth"
)
DEFAULT_MANIFEST = REPO_ROOT / "config/manifests/v14_multihuman_camera_supervision_20260803.json"
DEFAULT_DATA = Path("/data/wangzheng/iJCV-CODE/data/MultiHuman/Real-World-Capture/extracted")
DEFAULT_OUTPUT = REPO_ROOT / "output/v14/fine_alignment_research/p1_foot_scene_observability"
FOOT_JOINTS = (("left", 10), ("right", 11))
PATCH_SIZE = 33
PATCH_RADIUS = PATCH_SIZE // 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--diagnose-only", action="store_true")
    return parser.parse_args()


def ensure_workspace(path: Path) -> None:
    if not str(path.resolve()).startswith(str(REPO_ROOT.resolve())):
        raise ValueError(f"P1 artifact must stay in Movie3R workspace: {path}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    return hashlib.sha256(value.tobytes()).hexdigest()


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
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def finite_summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0, "mean": float("nan"), "median": float("nan"), "p90": float("nan"), "p95": float("nan")}
    return {
        "count": int(len(array)), "mean": float(array.mean()), "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)), "p95": float(np.percentile(array, 95)),
    }


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    transform, points = np.asarray(transform, dtype=np.float64), np.asarray(points, dtype=np.float64)
    return points @ transform[:3, :3].T + transform[:3, 3]


def transform_person(transform: np.ndarray, person: dict[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(person)
    for key in ("root", "joints", "vertices"):
        output[key] = transform_points(transform, person[key])
    for key in ("torso", "root_rotation"):
        output[key] = transform[:3, :3] @ np.asarray(person[key], dtype=np.float64)
    return output


def decode_people(
    prediction: dict[str, Any], view: dict[str, Any], debug: dict[str, Any], layer: SMPL_Layer
) -> list[dict[str, Any]]:
    """Mirror ``layer_humans`` while retaining local foot UV/depth for P1."""
    if "smpl_transl" not in prediction:
        return []
    count = int(prediction["smpl_transl"].shape[1])
    if not count:
        return []
    device = next(layer.parameters()).device
    rotmat = prediction["smpl_rotmat"][0, :count].to(device=device, dtype=torch.float32)
    rotvec = roma.rotmat_to_rotvec(rotmat)
    shape = prediction["smpl_shape"][0, :count].to(device=device, dtype=torch.float32)
    transl = prediction["smpl_transl"][0, :count].to(device=device, dtype=torch.float32)
    expression = prediction.get("smpl_expression")
    expression = (torch.zeros((count, 10), device=device, dtype=torch.float32) if expression is None
                  else expression[0, :count].to(device=device, dtype=torch.float32))
    intrinsic = view["K_mhmr"][0].to(device=device, dtype=torch.float32).clone()
    height, width = [int(value) for value in gt.tensor_numpy(view["true_shape"])[0]]
    padded_size = int(view["img_mhmr"].shape[-1])
    intrinsic[0, 2] -= 0.5 * (padded_size - width)
    intrinsic[1, 2] -= 0.5 * (padded_size - height)
    with torch.no_grad():
        body = layer(rotvec, shape, transl, None, None, K=intrinsic[None].expand(count, -1, -1), expression=expression)
    joints = body["smpl_j3d"].detach().float().cpu().numpy()
    vertices = body["smpl_v3d"].detach().float().cpu().numpy()
    pixels = body["smpl_v2d"].detach().float().cpu().numpy()
    joint_pixels = body["smpl_j2d"].detach().float().cpu().numpy()
    scores = gt.tensor_numpy(debug.get("head_scores", np.ones(count))).reshape(-1)
    pose = camera_matrix(prediction).astype(np.float64)
    output = []
    for index in range(count):
        finite = np.isfinite(pixels[index]).all(axis=1)
        visible = finite & (vertices[index, :, 2] > 0.0)
        visible &= (pixels[index, :, 0] >= 0.0) & (pixels[index, :, 0] < width)
        visible &= (pixels[index, :, 1] >= 0.0) & (pixels[index, :, 1] < height)
        valid_pixels = pixels[index, finite]
        if not len(valid_pixels):
            bbox = np.zeros(4, dtype=np.float64)
        else:
            bbox = np.r_[valid_pixels.min(axis=0), valid_pixels.max(axis=0)]
            bbox[:2] = np.maximum(bbox[:2], (0.0, 0.0))
            bbox[2:] = np.minimum(bbox[2:], (width - 1.0, height - 1.0))
        feet = {}
        for name, joint_id in FOOT_JOINTS:
            exists = joint_id < len(joints[index])
            local = joints[index, joint_id].astype(np.float64) if exists else np.full(3, np.nan)
            # ``smpl_v2d`` has one row per mesh vertex.  Foot anchors are
            # joints, so indexing it by a joint ID silently selects an
            # unrelated low-index vertex and destroys pixel correspondence.
            uv = joint_pixels[index, joint_id].astype(np.float64) if exists else np.full(2, np.nan)
            in_frame = bool(exists and np.isfinite(local).all() and np.isfinite(uv).all()
                            and local[2] > 0.05 and 0 <= uv[0] < width and 0 <= uv[1] < height)
            feet[name] = {"joint_id": joint_id, "local": local, "uv": uv, "in_frame": in_frame}
        output.append({
            "detection_index": int(index), "score": float(scores[index]) if index < len(scores) else 1.0,
            "completeness": float(np.mean(visible)), "bbox": bbox, "root": transform_points(pose, joints[index, :1])[0],
            "torso": pose[:3, :3] @ gt.torso_frame(joints[index]), "root_rotation": pose[:3, :3] @ gt.tensor_numpy(rotmat[index, 0]),
            "joints": transform_points(pose, joints[index]), "vertices": transform_points(pose, vertices[index]),
            "feet_local": feet,
        })
    return output


def prediction_arrays(prediction: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    points = np.asarray(gt.tensor_numpy(prediction["pts3d_in_self_view"]), dtype=np.float32).squeeze()
    confidence = prediction.get("conf_self")
    confidence = (np.ones(points.shape[:2], dtype=np.float32) if confidence is None
                  else np.asarray(gt.tensor_numpy(confidence), dtype=np.float32).squeeze())
    if points.ndim != 3 or points.shape[-1] != 3 or confidence.shape != points.shape[:2]:
        raise ValueError(f"invalid P1 pointmap/confidence {points.shape} {confidence.shape}")
    mask = prediction.get("msk")
    if mask is not None:
        mask = np.asarray(gt.tensor_numpy(mask)).squeeze()
        if mask.ndim != 2:
            raise ValueError(f"invalid Human3R mask {mask.shape}")
        mask = cv2.resize(mask.astype(np.uint8), (points.shape[1], points.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    return points, confidence, mask


def foot_patch(
    foot: dict[str, Any], points: np.ndarray, confidence: np.ndarray, human_mask: np.ndarray | None,
    true_shape: tuple[int, int],
) -> dict[str, Any]:
    height, width = points.shape[:2]
    true_height, true_width = true_shape
    uv = np.asarray(foot["uv"], dtype=np.float64)
    local = np.asarray(foot["local"], dtype=np.float64)
    if not bool(foot["in_frame"]):
        return {"status": "foot_not_in_frame", "foot_local": local, "foot_uv": uv}
    center = np.rint(np.array([uv[0] * width / max(true_width, 1), uv[1] * height / max(true_height, 1)])).astype(int)
    if center[0] < 0 or center[0] >= width or center[1] < 0 or center[1] >= height:
        return {"status": "foot_outside_pointmap", "foot_local": local, "foot_uv": uv}
    y_grid, x_grid = np.indices((PATCH_SIZE, PATCH_SIZE))
    xx = center[0] + x_grid - PATCH_RADIUS
    yy = center[1] + y_grid - PATCH_RADIUS
    inside = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
    patch_points = np.full((PATCH_SIZE, PATCH_SIZE, 3), np.nan, dtype=np.float32)
    patch_conf = np.full((PATCH_SIZE, PATCH_SIZE), np.nan, dtype=np.float32)
    patch_points[inside] = points[yy[inside], xx[inside]]
    patch_conf[inside] = confidence[yy[inside], xx[inside]]
    valid = inside & np.isfinite(patch_points).all(axis=2) & np.isfinite(patch_conf)
    valid &= (patch_points[:, :, 2] > 0.05) & (patch_points[:, :, 2] < 50.0)
    human = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=bool)
    if human_mask is not None:
        dilated = binary_dilation(human_mask, iterations=3)
        human[inside] = dilated[yy[inside], xx[inside]]
    radius = np.sqrt((x_grid - PATCH_RADIUS) ** 2 + (y_grid - PATCH_RADIUS) ** 2)
    annulus = (radius >= 4.0) & (radius <= 16.0)
    return {
        "status": "ok", "foot_local": local.astype(np.float32), "foot_uv": uv.astype(np.float32),
        "center_pointmap_uv": center.astype(np.int16), "points_local": patch_points,
        "confidence": patch_conf, "valid": valid, "nonhuman": ~human, "annulus": annulus,
        "patch_uv": np.stack((xx, yy), axis=-1).astype(np.int16),
    }


def pack_people(
    people: list[dict[str, Any]], prediction: dict[str, Any], view: dict[str, Any]
) -> list[dict[str, Any]]:
    points, confidence, mask = prediction_arrays(prediction)
    true_shape = tuple(int(value) for value in gt.tensor_numpy(view["true_shape"])[0])
    output = []
    for person in people:
        row = copy.deepcopy(person)
        row["foot_patches"] = {
            name: foot_patch(foot, points, confidence, mask, true_shape)
            for name, foot in person["feet_local"].items()
        }
        output.append(row)
    return output


def anonymous_match(pre_people: list[dict[str, Any]], post_people: list[dict[str, Any]]) -> dict[str, Any]:
    if not pre_people or not post_people:
        return {"pairs": [], "matched_count": 0, "cost": []}
    named_pre = {str(index): person for index, person in enumerate(pre_people)}
    detections = [(str(index), person) for index, person in enumerate(post_people)]
    components = identity_cost_components(named_pre, detections, np.eye(4, dtype=np.float64), tuple(named_pre))
    cost = matching_costs(components)["root_torso_joints"]
    rows, columns = linear_sum_assignment(cost)
    return {"pairs": [(int(row), int(column)) for row, column in zip(rows, columns)], "matched_count": int(len(rows)), "cost": cost}


def cache_case(
    model: ARCroco3DStereo, layer: SMPL_Layer, record: dict[str, Any], gt_args: SimpleNamespace,
    device: torch.device, size: int,
) -> dict[str, Any]:
    """Finish the complete non-GT action transaction before evaluator fields exist."""
    frame, pre_camera, post_camera = (int(record[key]) for key in ("frame", "pre_camera", "post_camera"))
    inputs = [
        gt.extract_video_frame(gt_args, pre_camera, frame - 1), gt.extract_video_frame(gt_args, pre_camera, frame),
        gt.extract_video_frame(gt_args, post_camera, frame),
    ]
    views = gt.prepare_full_square_input(model, inputs, SimpleNamespace(size=int(size)))
    shadow_views, raw_views = set_event_indices(copy.deepcopy(views), {2}), set_event_indices(copy.deepcopy(views[2:]), set())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
        shadow_predictions, shadow_returned, shadow_debug = model.forward_recurrent_lighter(
            shadow_views, str(device), ret_state=False, use_ttt3r=False, return_token_debug=True
        )
        raw_predictions, raw_returned, raw_debug = model.forward_recurrent_lighter(
            raw_views, str(device), ret_state=False, use_ttt3r=False, return_token_debug=True
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    pre_pose = camera_matrix(shadow_predictions[1]).astype(np.float64)
    shadow_pose = camera_matrix(shadow_predictions[2]).astype(np.float64)
    raw_pose = camera_matrix(raw_predictions[0]).astype(np.float64)
    boundary = boundary_from_camera_predictions(shadow_predictions[2], raw_predictions[0])[0].detach().float().cpu().numpy().astype(np.float64)
    b0_pose = boundary @ raw_pose
    parity = float(np.max(np.abs(shadow_pose - b0_pose)))
    if parity > 1e-5:
        raise RuntimeError(f"B0/shadow parity failure {parity}")
    pre_people = pack_people(decode_people(shadow_predictions[1], shadow_returned[1], shadow_debug[1], layer), shadow_predictions[1], shadow_returned[1])
    raw_people = pack_people(decode_people(raw_predictions[0], raw_returned[0], raw_debug[0], layer), raw_predictions[0], raw_returned[0])
    b0_people = [transform_person(boundary, person) for person in raw_people]
    association = anonymous_match(pre_people, b0_people)
    brtc_people, brtc_debug = refine_matched_people(pre_pose, b0_pose, pre_people, b0_people, association["pairs"], BRTC_CONFIG)
    if brtc_debug.get("camera_update") != "none":
        raise RuntimeError("BRTC attempted a camera update")
    runtime = {
        "record": dict(record), "timing_seconds": float(elapsed), "pre_camera_c2w": pre_pose,
        "raw_camera_c2w": raw_pose, "b0_camera_c2w": b0_pose, "b0": boundary,
        "b0_shadow_camera_max_abs": parity, "pre_people": pre_people, "raw_post_people": raw_people,
        "b0_post_people": b0_people, "brtc_post_people": brtc_people, "association": association,
        "b0_camera_sha256": array_sha256(b0_pose),
        "brtc": brtc_debug, "runtime_contract": {"gt_used": False, "future_post_frames_used": 0,
        "camera_update": "none", "shadow_state_committed": False},
    }
    # The only code below this line allowed to read calibrated meshes/identity.
    height, width = [int(value) for value in gt.tensor_numpy(raw_returned[0]["true_shape"])[0]]
    assigned, assignment = gt.assign_gt_identities(gt_args, raw_people, raw_pose, post_camera, frame, width, height)
    gt_pre = np.linalg.inv(gt.gt_w2c(gt_args, pre_camera, frame))
    gt_post = np.linalg.inv(gt.gt_w2c(gt_args, post_camera, frame))
    gauge = pre_pose @ np.linalg.inv(gt_pre)
    targets = gt.gt_human_payload(gt_args, frame, gt.joint_regressor(layer))
    target_by_detection = {}
    for identity, detected in assigned.items():
        if identity in targets:
            target_by_detection[int(detected["detection_index"])] = {
                "identity_evaluator_only": identity,
                "root_world": transform_points(gauge, targets[identity]["root"][None])[0],
                "joints_world": transform_points(gauge, targets[identity]["joints"]),
                "vertices_world": transform_points(gauge, targets[identity]["vertices"]),
            }
    return {"status": "ok", "runtime": runtime, "evaluator": {
        "assignment_evaluator_only": assignment, "target_by_detection": target_by_detection,
        "gt_camera_evaluator_only": gauge @ gt_post,
    }}


def build_cache(args: argparse.Namespace) -> Path:
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = list(payload["dev"])
    if args.max_cases:
        records = records[: int(args.max_cases)]
    cache_dir, cases_dir = args.output_dir / "cache", args.output_dir / "cases"
    cache_dir.mkdir(parents=True, exist_ok=True); cases_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = ARCroco3DStereo.from_pretrained(str(args.model_path)).to(device)
    flags = configure_model(model)
    layer = SMPL_Layer(type="smplx", gender="neutral", num_betas=10, kid=False, person_center="head").to(device).eval()
    gt_args = SimpleNamespace(data_root=args.data_root, sequence="three", output_dir=cache_dir / "frame_cache", size=int(args.size))
    gt.IDENTITIES = gt.SEQUENCE_IDENTITIES["three"]
    case_paths, failures = [], []
    try:
        for index, record in enumerate(records, start=1):
            path = cases_dir / f"{record['event_id']}.pt"
            if path.is_file() and not args.overwrite:
                row = torch.load(path, map_location="cpu", weights_only=False)
            else:
                try:
                    row = cache_case(model, layer, record, gt_args, device, int(args.size))
                except Exception as error:  # preserve failure evidence and continue the fixed split
                    row = {"status": "failed", "record": record, "error": repr(error), "traceback": traceback.format_exc()}
                torch.save(row, path)
            if row["status"] == "ok":
                case_paths.append(path)
                print(f"[{index:02d}/{len(records):02d}] {record['event_id']} people={len(row['runtime']['raw_post_people'])} matched={row['runtime']['association']['matched_count']}", flush=True)
            else:
                failures.append({"event_id": record["event_id"], "error": row["error"]})
                print(f"[{index:02d}/{len(records):02d}] {record['event_id']} FAILED {row['error']}", flush=True)
            if device.type == "cuda": torch.cuda.empty_cache()
            gc.collect()
    finally:
        del layer, model
        if device.type == "cuda": torch.cuda.empty_cache()
        gc.collect()
    index = {"schema": 2, "purpose": "P1 compact Human3R foot-local runtime cache", "checkpoint": str(args.model_path),
             "checkpoint_sha256": sha256(args.model_path), "manifest": str(args.manifest), "manifest_sha256": sha256(args.manifest),
             "model_flags": flags, "device": str(device), "records_requested": len(records),
             "case_paths": [str(path) for path in case_paths], "failures": failures,
             "runtime_gt_order": "runtime transaction complete before evaluator GT is loaded", "patch": {"size": PATCH_SIZE, "annulus_px": [4, 16], "human_mask_dilation_px": 3}}
    destination = args.output_dir / "P1_CACHE_INDEX.json"
    destination.write_text(json.dumps(jsonable(index), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return destination


def robust_plane(patch: dict[str, Any], transform: np.ndarray, foot_world: np.ndarray) -> tuple[dict[str, Any] | None, str]:
    if patch.get("status") != "ok": return None, str(patch.get("status", "invalid_patch"))
    select = np.asarray(patch["valid"], bool) & np.asarray(patch["nonhuman"], bool) & np.asarray(patch["annulus"], bool)
    points = np.asarray(patch["points_local"], dtype=np.float64)[select]
    conf = np.asarray(patch["confidence"], dtype=np.float64)[select]
    uv = np.asarray(patch["patch_uv"], dtype=np.int16)[select]
    if len(points) < 24: return None, "too_few_support"
    quadrants = ((uv[:, 0] >= int(patch["center_pointmap_uv"][0])) * 2 + (uv[:, 1] >= int(patch["center_pointmap_uv"][1]))).astype(int)
    if len(np.unique(quadrants)) < 3: return None, "uv_quadrants"
    points = transform_points(transform, points)
    weight = np.maximum(conf - np.nanmin(conf) + 1e-3, 1e-3); weight /= weight.sum()
    centroid = np.sum(weight[:, None] * points, axis=0)
    _, singular, vectors = np.linalg.svd((points - centroid) * np.sqrt(weight[:, None]), full_matrices=False)
    normal = vectors[-1]; normal /= max(float(np.linalg.norm(normal)), 1e-12)
    if float(np.dot(normal, foot_world - centroid)) < 0: normal = -normal
    residual = np.abs((points - centroid) @ normal)
    extent = float(np.percentile(np.linalg.norm(points - centroid, axis=1), 90))
    if extent < 0.05: return None, "small_extent"
    if float(np.median(residual)) > 0.02: return None, "plane_residual"
    return {"centroid": centroid, "normal": normal, "signed_offset": float(np.dot(foot_world - centroid, normal)),
            "support_count": int(len(points)), "quadrants": int(len(np.unique(quadrants))), "extent_m": extent,
            "median_plane_residual_m": float(np.median(residual)), "singular": singular}, "ok"


def proposal_rows(runtime: dict[str, Any]) -> list[dict[str, Any]]:
    """Prediction-only P1 action creation.  Do not add evaluator access here."""
    rows = []
    brtc_by_post = {int(item["post_index"]): item for item in runtime["brtc"]["people"]}
    pre_people, b0_people, brtc_people = runtime["pre_people"], runtime["b0_post_people"], runtime["brtc_post_people"]
    pre_camera, b0 = np.asarray(runtime["pre_camera_c2w"]), np.asarray(runtime["b0"])
    for pre_index, post_index in runtime["association"]["pairs"]:
        brtc = brtc_by_post.get(int(post_index), {})
        base = {"pre_index": int(pre_index), "post_index": int(post_index), "accepted_brtc": bool(brtc.get("accepted", False)),
                "fallback_reason": None, "per_foot": {}, "proposal_world": np.zeros(3), "action_world": np.zeros(3)}
        if not base["accepted_brtc"]:
            base["fallback_reason"] = "brtc_rejected"; rows.append(base); continue
        proposals = []
        for name, _ in FOOT_JOINTS:
            pre_patch, post_patch = pre_people[pre_index]["foot_patches"][name], b0_people[post_index]["foot_patches"][name]
            pre_foot = transform_points(pre_camera, np.asarray(pre_patch.get("foot_local", [np.nan] * 3))[None])[0]
            post_foot = transform_points(b0, np.asarray(post_patch.get("foot_local", [np.nan] * 3))[None])[0]
            post_foot += np.asarray(brtc.get("final_shift_world", np.zeros(3)), dtype=np.float64)
            pre_plane, pre_reason = robust_plane(pre_patch, pre_camera, pre_foot)
            post_plane, post_reason = robust_plane(post_patch, b0, post_foot)
            foot = {"pre_reason": pre_reason, "post_reason": post_reason}
            if pre_plane is None or post_plane is None:
                base["per_foot"][name] = foot; continue
            angle = math.degrees(math.acos(np.clip(float(np.dot(pre_plane["normal"], post_plane["normal"])), -1.0, 1.0)))
            foot.update({"pre": pre_plane, "post": post_plane, "normal_disagreement_deg": angle})
            if angle > 25.0 or max(abs(pre_plane["signed_offset"]), abs(post_plane["signed_offset"])) > 0.20:
                foot["proposal_reason"] = "normal_or_contact_range"; base["per_foot"][name] = foot; continue
            proposal = (pre_plane["signed_offset"] - post_plane["signed_offset"]) * post_plane["normal"]
            # Must reduce this prediction-only plane residual by >=10%; full correction does, then half is committed.
            if abs(float(np.dot(post_plane["normal"], proposal))) < 0.10 * abs(post_plane["signed_offset"] - pre_plane["signed_offset"]):
                foot["proposal_reason"] = "no_predicted_improvement"; base["per_foot"][name] = foot; continue
            foot["proposal_world"] = proposal; base["per_foot"][name] = foot; proposals.append(proposal)
        if len(proposals) != 2:
            base["fallback_reason"] = "incomplete_foot_observability"; rows.append(base); continue
        if float(np.linalg.norm(proposals[0] - proposals[1])) > 0.02:
            base["fallback_reason"] = "left_right_disagreement"; rows.append(base); continue
        proposal = np.median(np.stack(proposals), axis=0)
        norm = float(np.linalg.norm(0.5 * proposal))
        base["proposal_world"] = proposal
        base["action_world"] = 0.5 * proposal * min(1.0, 0.03 / max(norm, 1e-12))
        base["fallback_reason"] = "applied"
        rows.append(base)
    return rows


def bootstrap_mean(values: np.ndarray, seed: int = 20260803, draws: int = 10000) -> list[float]:
    if not len(values): return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    sample = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return [float(np.quantile(sample, 0.025)), float(np.quantile(sample, 0.975))]


def diagnose(args: argparse.Namespace) -> Path:
    index_path = args.output_dir / "P1_CACHE_INDEX.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    rows, fallback, foot_failures, cosine, sign, root_delta, joint_delta, vertex_delta = [], {}, {}, [], [], [], [], []
    camera_change = 0.0
    camera_hashes: dict[str, str] = {}
    for path_text in index["case_paths"]:
        cached = torch.load(path_text, map_location="cpu", weights_only=False)
        runtime = cached["runtime"]
        proposals = proposal_rows(runtime)  # mandatory: action is fixed before evaluator is accessed.
        if runtime["runtime_contract"]["gt_used"] or runtime["runtime_contract"]["future_post_frames_used"]:
            raise RuntimeError("invalid runtime cache contract")
        # Neither BRTC nor this root-only proposal accepts a camera argument by
        # reference.  Rehash immediately before evaluator access to make the
        # no-camera-edit invariant an explicit serialized audit.
        current_hash = array_sha256(np.asarray(runtime["b0_camera_c2w"]))
        if current_hash != runtime["b0_camera_sha256"]:
            raise RuntimeError("P1 runtime camera mutated before evaluator access")
        camera_hashes[str(runtime["record"]["event_id"])] = current_hash
        targets = cached["evaluator"]["target_by_detection"]
        for proposal in proposals:
            reason = str(proposal["fallback_reason"]); fallback[reason] = fallback.get(reason, 0) + 1
            for foot in proposal["per_foot"].values():
                for name in ("pre_reason", "post_reason", "proposal_reason"):
                    value = foot.get(name)
                    if value not in (None, "ok"):
                        key = f"{name}:{value}"
                        foot_failures[key] = foot_failures.get(key, 0) + 1
            post_index = int(proposal["post_index"])
            person = runtime["brtc_post_people"][post_index]
            target = targets.get(int(person["detection_index"]))
            row = {"event_id": runtime["record"]["event_id"], **proposal, "has_evaluator_target": target is not None}
            if target is not None:
                residual = np.asarray(target["root_world"], dtype=np.float64) - np.asarray(person["root"], dtype=np.float64)
                action = np.asarray(proposal["action_world"], dtype=np.float64)
                raw = np.asarray(proposal["proposal_world"], dtype=np.float64)
                row["root_residual_world_evaluator_only"] = residual
                row["root_error_brtc_m"] = float(np.linalg.norm(residual))
                row["root_error_p1_m"] = float(np.linalg.norm(residual - action))
                if reason == "applied" and np.linalg.norm(raw) > 1e-9 and np.linalg.norm(residual) > 1e-9:
                    value = float(np.dot(raw, residual) / (np.linalg.norm(raw) * np.linalg.norm(residual)))
                    cosine.append(value); sign.append(float(np.dot(raw, residual) > 0.0))
                    root_delta.append(row["root_error_p1_m"] - row["root_error_brtc_m"])
                    joints = np.asarray(person["joints"], dtype=np.float64); vertices = np.asarray(person["vertices"], dtype=np.float64)
                    target_joints, target_vertices = np.asarray(target["joints_world"]), np.asarray(target["vertices_world"])
                    joint_delta.append(float(np.mean(np.linalg.norm(joints + action - target_joints, axis=1)) - np.mean(np.linalg.norm(joints - target_joints, axis=1))))
                    vertex_delta.append(float(np.mean(np.linalg.norm(vertices + action - target_vertices, axis=1)) - np.mean(np.linalg.norm(vertices - target_vertices, axis=1))))
            rows.append(row)
    applied = [row for row in rows if row["fallback_reason"] == "applied"]
    targeted_accepted = [row for row in rows if row["accepted_brtc"] and row["has_evaluator_target"]]
    gate = {
        "valid_matched_person_count_at_least_24": len(cosine) >= 24,
        "coverage_at_least_20pct": len(cosine) / max(len(targeted_accepted), 1) >= 0.20,
        "mean_cosine_bootstrap_lower_gt_zero": bool(len(cosine) and bootstrap_mean(np.asarray(cosine))[0] > 0.0),
        "sign_agreement_at_least_60pct": bool(len(sign) and float(np.mean(sign)) >= 0.60),
        "mean_root_improvement_at_least_5mm": bool(len(root_delta) and float(np.mean(root_delta)) <= -0.005),
        "camera_bit_exact": camera_change == 0.0,
    }
    report = {"experiment": "v14_p1_human3r_foot_scene_observability", "status": "GO_TO_POLICY_FREEZE" if all(gate.values()) else "NO_GO_HUMAN3R_FOOT_SCENE_SIGNAL",
              "cache_index": str(index_path), "cache_index_sha256": sha256(index_path), "policy": {"patch_size": PATCH_SIZE, "annulus_px": [4, 16], "cap_m": 0.03, "shrinkage": 0.5},
              "runtime_invariants": {"camera_max_abs_change": camera_change, "camera_sha256_by_event": camera_hashes,
                                     "all_actions_before_gt": True, "external_pretrained_models": []},
              "counts": {"proposal_rows": len(rows), "brtc_accepted_with_target": len(targeted_accepted), "applied_with_direction_target": len(cosine), "fallback_reasons": fallback, "foot_gate_failures": foot_failures},
              "direction": {"cosine": finite_summary(cosine), "cosine_bootstrap_95pct": bootstrap_mean(np.asarray(cosine)), "sign_agreement": float(np.mean(sign)) if sign else float("nan")},
              "counterfactual_bounded_action": {"root_delta_m": finite_summary(root_delta), "joint_delta_m": finite_summary(joint_delta), "vertex_delta_m": finite_summary(vertex_delta)},
              "gate": gate, "rows": rows}
    destination = args.output_dir / "P1_FOOT_SCENE_SIGNAL_REPORT.json"
    destination.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return destination


def main() -> None:
    args = parse_args()
    for path in (args.model_path, args.manifest, args.output_dir): ensure_workspace(path)
    if args.build_only and args.diagnose_only: raise ValueError("choose at most one mode")
    if not args.diagnose_only:
        if not args.model_path.is_file(): raise FileNotFoundError(args.model_path)
        build_cache(args)
    if not args.build_only:
        report = diagnose(args); print(report, flush=True)


if __name__ == "__main__":
    main()
