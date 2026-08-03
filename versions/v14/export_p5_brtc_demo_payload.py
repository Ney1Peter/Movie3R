#!/usr/bin/env python3
"""Build standard demo.py viewer payloads for frozen BRTC-LC and P5 results.

The P5 confirmation cache deliberately stores only compact person geometry,
not the dense scene pointmaps required by the original Human3R viewer.  This
exporter therefore recomputes the causal three-RGB transaction on CPU from the
frozen current-P0 checkpoint, then writes its two relevant display frames
(``last-pre`` and ``first-post``) plus the already-confirmed cached BRTC/P5
world meshes into standard ``demo.py --save`` payloads.

It never uses evaluator data to produce BRTC/P5 output.  The two candidates
remain exactly those from the frozen cache; the CPU forward only restores
background/depth/confidence for visualisation.
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
from typing import Any

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from dust3r.model import ARCroco3DStereo
from dust3r.post_process import estimate_focal_knowing_depth
from dust3r.utils.image import unpad_image
from versions.v13 import gt_id_consensus as gt
from versions.v14.probe_p1_foot_scene_observability import configure_model, set_event_indices
from versions.v14.run_v14_2_single_sequence import camera_matrix


CACHE_ROOT = REPO_ROOT / "output/v14/fine_alignment_research/p5_brtc_ray_residual_confirmation_cache"
CHECKPOINT = REPO_ROOT / "output/v14_cut_first_cross_source/v14_cut_first_cross_source_multihuman_p0_e6/checkpoint-final.pth"
RIDGE_MODEL = REPO_ROOT / "output/v14/fine_alignment_research/p5_brtc_ray_residual_calibration/P5_FROZEN_MODEL_BEFORE_CONFIRM.json"
FACES = REPO_ROOT / "src/models/smplx/SMPLX_NEUTRAL.npz"
DEFAULT_OUTPUT = REPO_ROOT / "output/v14/p5_brtc_ray_residual_demo_payload"
DEFAULT_CASES = ("confirm_dance_t0400_c1_c4_k0", "confirm_box_t0510_c0_c3_k0")
FEATURES = ("raw_m", "valid_count", "median_gap_m", "max_gap_m", "median_sine", "min_sine", "mad_m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--ridge-model", type=Path, default=RIDGE_MODEL)
    parser.add_argument("--faces", type=Path, default=FACES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    parser.add_argument("--device", default="cpu", choices=("cpu",))
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def under_repo(path: Path) -> Path:
    resolved = path.resolve()
    if resolved != REPO_ROOT and REPO_ROOT not in resolved.parents:
        raise ValueError(f"Expected a Movie3R path, got {resolved}")
    return resolved


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def load_ridge(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if tuple(payload["features"]) != FEATURES or float(payload["alpha"]) != 1.0:
        raise ValueError(f"Unexpected P5 model: {path}")
    return (
        as_array(payload["scaler_mean"]), as_array(payload["scaler_scale"]),
        as_array(payload["ridge_coef"]), float(payload["ridge_intercept"]), float(payload["clip_m"]),
    )


def p5_from_cache(runtime: dict[str, Any], ridge: tuple[np.ndarray, np.ndarray, np.ndarray, float, float]) -> tuple[list[dict[str, Any]], dict[int, float]]:
    mean, scale, coef, intercept, cap = ridge
    output = [
        {key: (as_array(value).copy() if key in ("root", "joints", "vertices") else value)
         for key, value in person.items()}
        for person in runtime["brtc_post_people"]
    ]
    center = as_array(runtime["b0_camera_c2w"])[:3, 3]
    raw_by_detection = {int(item["detection_index"]): item for item in runtime["b0_post_people"]}
    debug_by_detection = {int(item["post_index"]): item for item in runtime["brtc"]["people"]}
    residuals: dict[int, float] = {}
    for person in output:
        detection = int(person["detection_index"])
        debug = debug_by_detection[detection]
        if not bool(debug["accepted"]):
            residuals[detection] = 0.0
            continue
        feature = as_array([debug["evidence"][feature_name] for feature_name in FEATURES])
        residual = float(np.clip(((feature - mean) / scale) @ coef + intercept, -cap, cap))
        ray = as_array(raw_by_detection[detection]["root"]) - center
        ray /= max(float(np.linalg.norm(ray)), 1e-12)
        for key in ("root", "joints", "vertices"):
            person[key] += residual * ray
        residuals[detection] = residual
    return output, residuals


def input_path(cache_root: Path, sequence: str, camera: int, frame: int) -> Path:
    return cache_root / "cache" / sequence / "frame_cache" / "input_frames" / f"cam{camera}" / f"{frame:06d}.jpg"


def make_views(cache_root: Path, record: dict[str, Any], model: ARCroco3DStereo, size: int) -> list[dict[str, Any]]:
    frame, pre_camera, post_camera = (int(record[key]) for key in ("frame", "pre_camera", "post_camera"))
    paths = [
        input_path(cache_root, str(record["sequence"]), pre_camera, frame - 1),
        input_path(cache_root, str(record["sequence"]), pre_camera, frame),
        input_path(cache_root, str(record["sequence"]), post_camera, frame),
    ]
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError([str(path) for path in paths if not path.is_file()])
    return gt.prepare_full_square_input(model, paths, argparse.Namespace(size=int(size)))


def cpu_backgrounds(model: ARCroco3DStereo, cache_root: Path, runtime: dict[str, Any], size: int) -> tuple[list[dict[str, np.ndarray]], list[np.ndarray], float, float]:
    """Return the actual display maps for last-pre and first-post only."""
    views = make_views(cache_root, runtime["record"], model, size)
    shadow_views = set_event_indices(copy.deepcopy(views), {2})
    raw_views = set_event_indices(copy.deepcopy(views[2:]), set())
    started = time.perf_counter()
    with torch.no_grad():
        shadow_predictions, shadow_returned = model.forward_recurrent_lighter(
            shadow_views, "cpu", ret_state=False, use_ttt3r=False, return_token_debug=False
        )
        raw_predictions, raw_returned = model.forward_recurrent_lighter(
            raw_views, "cpu", ret_state=False, use_ttt3r=False, return_token_debug=False
        )
    elapsed = time.perf_counter() - started
    rows = [
        (shadow_predictions[1], shadow_returned[1]),
        (raw_predictions[0], raw_returned[0]),
    ]
    backgrounds = [background_from_prediction(prediction, view) for prediction, view in rows]
    computed_last_pre = as_array(camera_matrix(shadow_predictions[1]))
    cached_last_pre = as_array(runtime["pre_camera_c2w"])
    computed_b0 = as_array(camera_matrix(shadow_predictions[2])) @ np.linalg.inv(as_array(camera_matrix(raw_predictions[0])))
    camera_error = float(np.max(np.abs(computed_b0 - as_array(runtime["b0"]))))
    cameras = [
        cached_last_pre,
        as_array(runtime["b0_camera_c2w"]),
    ]
    return backgrounds, cameras, float(elapsed), camera_error


def background_from_prediction(prediction: dict[str, torch.Tensor], view: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    pointmap = prediction["pts3d_in_self_view"][0].detach().float().cpu()
    confidence = prediction.get("conf_self", torch.ones(pointmap.shape[:2], dtype=pointmap.dtype))[0].detach().float().cpu()
    height, width = [int(value) for value in view["true_shape"][0].detach().cpu().tolist()]
    image = 0.5 * (view["img"][0].detach().float().cpu().permute(1, 2, 0) + 1.0)
    if tuple(image.shape[:2]) != (height, width):
        image = torch.nn.functional.interpolate(image.permute(2, 0, 1).unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
    if tuple(pointmap.shape[:2]) != (height, width):
        pointmap = torch.nn.functional.interpolate(pointmap.permute(2, 0, 1).unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
        confidence = torch.nn.functional.interpolate(confidence.unsqueeze(0).unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False)[0, 0]
    mask_value = prediction.get("msk")
    if mask_value is None:
        mask = torch.zeros((1, height, width), dtype=torch.float32)
    else:
        mask = mask_value[..., 0].detach().float().cpu()
        if tuple(mask.shape[-2:]) != (height, width):
            mask = unpad_image(mask, [height, width])
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
    pp = torch.tensor([width // 2, height // 2], dtype=torch.float32)
    focal = estimate_focal_knowing_depth(pointmap.unsqueeze(0), pp.unsqueeze(0), focal_mode="weiszfeld").detach().float().cpu().item()
    intrinsic = np.eye(3, dtype=np.float32)
    intrinsic[0, 0] = intrinsic[1, 1] = float(focal)
    intrinsic[0, 2], intrinsic[1, 2] = float(pp[0]), float(pp[1])
    return {
        "depth": pointmap[..., 2].numpy().astype(np.float32),
        "conf": confidence.numpy().astype(np.float32),
        "color": np.clip(image.numpy(), 0.0, 1.0).astype(np.float32),
        "mask": mask.numpy().astype(np.float32),
        "intrinsics": intrinsic,
    }


def prepare_destination(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {path}")
        if path == REPO_ROOT or REPO_ROOT not in path.parents:
            raise ValueError(f"Refusing unsafe overwrite: {path}")
        for child in sorted(path.rglob("*"), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
        path.rmdir()
    for name in ("depth", "conf", "color", "camera", "smpl"):
        (path / name).mkdir(parents=True, exist_ok=True)


def write_payload(destination: Path, backgrounds: list[dict[str, np.ndarray]], cameras: list[np.ndarray], people: list[list[dict[str, Any]]], faces: np.ndarray) -> None:
    if not (len(backgrounds) == len(cameras) == len(people) == 2):
        raise ValueError("A P5 payload must contain exactly last-pre and first-post")
    for index, (background, camera, frame_people) in enumerate(zip(backgrounds, cameras, people)):
        vertices = np.stack([as_array(person["vertices"]).astype(np.float32) for person in frame_people]) if frame_people else np.empty((0, 10475, 3), dtype=np.float32)
        ids = np.asarray([int(person["detection_index"]) for person in frame_people], dtype=np.int64)
        count = len(vertices)
        color_bgr = cv2.cvtColor((background["color"] * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(destination / "color" / f"{index:06d}.png"), color_bgr):
            raise OSError(f"Could not write RGB for frame {index}")
        np.save(destination / "depth" / f"{index:06d}.npy", background["depth"])
        np.save(destination / "conf" / f"{index:06d}.npy", background["conf"])
        np.savez(destination / "camera" / f"{index:06d}.npz", pose=as_array(camera).astype(np.float32), intrinsics=background["intrinsics"])
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


def export_case(model: ARCroco3DStereo, cache_root: Path, case: str, ridge: tuple[np.ndarray, np.ndarray, np.ndarray, float, float], faces: np.ndarray, output_root: Path, size: int, overwrite: bool) -> dict[str, Any]:
    source = cache_root / "cases" / f"{case}.pt"
    cache = torch.load(source, map_location="cpu", weights_only=False)
    if cache.get("status") != "ok":
        raise RuntimeError(f"Cached case failed: {case}")
    runtime = cache["runtime"]
    backgrounds, cameras, elapsed, recomputed_b0_error = cpu_backgrounds(model, cache_root, runtime, size)
    p5_people, residuals = p5_from_cache(runtime, ridge)
    pre_people = runtime["pre_people"]
    outputs = {
        "b0_brtc_lc": [pre_people, runtime["brtc_post_people"]],
        "b0_brtc_lc_p5": [pre_people, p5_people],
    }
    destinations = {}
    for method, people in outputs.items():
        destination = output_root / case / method
        prepare_destination(destination, overwrite)
        write_payload(destination, backgrounds, cameras, people, faces)
        destinations[method] = str(destination)
    return {
        "event_id": case,
        "record": runtime["record"],
        "cache": str(source),
        "cpu_background_forward_seconds": elapsed,
        "cpu_recomputed_b0_max_abs_error_vs_cache": recomputed_b0_error,
        "methods": destinations,
        "p5_residual_m_by_post_detection": {str(key): value for key, value in residuals.items()},
        "contract": {
            "device": "cpu",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "human_geometry": "frozen BRTC/P5 confirmation cache; no evaluator read by runtime export",
            "background": "same frozen P0 checkpoint, recomputed from the three cached RGB inputs only",
            "future_post_frames_used": 0,
            "camera": "cached B0 camera, bit-exact payload input",
        },
    }


def main() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    args = parse_args()
    cache_root, checkpoint, ridge_path, faces_path, output_root = (
        under_repo(args.cache_root), under_repo(args.checkpoint), under_repo(args.ridge_model),
        under_repo(args.faces), under_repo(args.output_dir),
    )
    if not checkpoint.is_file() or not ridge_path.is_file() or not faces_path.is_file():
        raise FileNotFoundError("Missing checkpoint, ridge model, or SMPL-X faces")
    ridge = load_ridge(ridge_path)
    with np.load(faces_path, allow_pickle=False) as payload:
        faces = np.asarray(payload["f"], dtype=np.int32)
    print(f">> loading frozen P0 on CPU: {checkpoint}", flush=True)
    model = ARCroco3DStereo.from_pretrained(str(checkpoint)).to("cpu")
    flags = configure_model(model)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    try:
        cases = [export_case(model, cache_root, str(case), ridge, faces, output_root, int(args.size), bool(args.overwrite)) for case in args.cases]
    finally:
        del model
    manifest = {
        "format": "standard demo.py --save compatible payload; view with scripts/view_human3r_saved_output.py",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_flags": flags,
        "ridge_model": str(ridge_path),
        "ridge_model_sha256": sha256(ridge_path),
        "cases": cases,
    }
    destination = output_root / "manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(destination, flush=True)


if __name__ == "__main__":
    main()
