#!/usr/bin/env python3
"""Export CPU-only, single-image Human3R-style overlays for selected P5 cases.

This utility deliberately reads the already frozen P5 confirmation cache.  It
does *not* load Human3R, run a forward pass, allocate CUDA tensors, or change
the saved experiment cache.  Each method/reference is written as an
independent image rather than a comparison grid:

    RGB -> B0 camera -> BRTC-LC bodies                (brtc_lc.png)
    RGB -> B0 camera -> BRTC-LC + frozen P5 bodies    (brtc_lc_p5.png)
    RGB -> evaluator-only GT bodies                   (gt_reference.png)

The GT image is a visual reference only.  It is never used to construct BRTC
or P5.  The manifest exposes the per-person evaluator numbers separately, so
the display cannot be mistaken for a runtime input.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = REPO_ROOT / "output/v14/fine_alignment_research/p5_brtc_ray_residual_confirmation_cache"
DEFAULT_MODEL = REPO_ROOT / "output/v14/fine_alignment_research/p5_brtc_ray_residual_calibration/P5_FROZEN_MODEL_BEFORE_CONFIRM.json"
DEFAULT_OUTPUT = REPO_ROOT / "output/v14/p5_brtc_ray_residual_original_demo/selected_multihuman"
FACES = REPO_ROOT / "src/models/smplx/SMPLX_NEUTRAL.npz"
DATA_ROOT = Path("/data/wangzheng/iJCV-CODE/data/MultiHuman/Real-World-Capture/extracted")

# These are the two confirmation events where both visible people benefit
# from the frozen residual.  They are selected before rendering, from the
# evaluator report, rather than by looking at the pictures.
DEFAULT_CASES = (
    "confirm_dance_t0400_c1_c4_k0",
    "confirm_box_t0510_c0_c3_k0",
)
FEATURES = (
    "raw_m", "valid_count", "median_gap_m", "max_gap_m", "median_sine",
    "min_sine", "mad_m",
)
COLORS_BGR = (
    (54, 106, 235),   # orange-red
    (186, 99, 42),    # blue
    (74, 175, 77),    # green
    (175, 70, 190),   # purple
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--faces", type=Path, default=FACES)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    parser.add_argument("--size", type=int, default=512)
    return parser.parse_args()


def ensure_workspace(path: Path) -> Path:
    resolved = path.resolve()
    if REPO_ROOT != resolved and REPO_ROOT not in resolved.parents:
        raise ValueError(f"Artifact must stay in Movie3R workspace: {resolved}")
    return resolved


def as_float(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def w2c_vertices(c2w: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    w2c = np.linalg.inv(as_float(c2w))
    return as_float(vertices) @ w2c[:3, :3].T + w2c[:3, 3]


def project(vertices_camera: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    depth = vertices_camera[:, 2]
    safe = np.maximum(depth, 1e-8)
    xy = np.empty((len(vertices_camera), 2), dtype=np.float64)
    xy[:, 0] = intrinsic[0, 0] * vertices_camera[:, 0] / safe + intrinsic[0, 2]
    xy[:, 1] = intrinsic[1, 1] * vertices_camera[:, 1] / safe + intrinsic[1, 2]
    return xy, depth


def render_people(
    image: np.ndarray,
    c2w: np.ndarray,
    people: list[tuple[str, np.ndarray]],
    faces: np.ndarray,
    intrinsic: np.ndarray,
    size: int,
) -> np.ndarray:
    """CPU painter renderer: sufficiently dense for an original-demo overlay."""
    base = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    overlay = base.copy()
    triangles: list[tuple[float, np.ndarray, tuple[int, int, int]]] = []
    labels: list[tuple[str, tuple[int, int], tuple[int, int, int]]] = []
    for order, (label, vertices_world) in enumerate(people):
        vertices_camera = w2c_vertices(c2w, vertices_world)
        xy, depth = project(vertices_camera, intrinsic)
        valid_vertices = (
            np.isfinite(xy).all(axis=1)
            & np.isfinite(depth)
            & (depth > 0.05)
            & (xy[:, 0] > -size)
            & (xy[:, 0] < 2 * size)
            & (xy[:, 1] > -size)
            & (xy[:, 1] < 2 * size)
        )
        color = COLORS_BGR[order % len(COLORS_BGR)]
        # Every third SMPL-X face is enough at 512px and keeps this a short
        # CPU-only export; the selection is deterministic and contains no
        # learned/render-time signal.
        selected = faces[::3]
        mask = valid_vertices[selected].all(axis=1)
        for face in selected[mask]:
            polygon = np.rint(xy[face]).astype(np.int32)
            area = abs(float(cv2.contourArea(polygon)))
            if area < 0.25 or area > float(size * size):
                continue
            triangles.append((float(np.mean(depth[face])), polygon, color))
        root_xy, root_depth = project(vertices_camera[:1], intrinsic)
        if root_depth[0] > 0.05 and np.isfinite(root_xy[0]).all():
            point = tuple(np.rint(root_xy[0]).astype(int))
            labels.append((label, point, color))

    # Far-to-near painter ordering produces deterministic inter-person
    # occlusion without EGL/OpenGL and therefore cannot occupy a GPU.
    for _, polygon, color in sorted(triangles, key=lambda value: value[0], reverse=True):
        cv2.fillConvexPoly(overlay, polygon, color, lineType=cv2.LINE_AA)
    rendered = cv2.addWeighted(overlay, 0.52, base, 0.48, 0.0)
    for label, point, color in labels:
        cv2.circle(rendered, point, 5, color, -1, lineType=cv2.LINE_AA)
        cv2.putText(rendered, label, (point[0] + 7, point[1] - 7), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.putText(rendered, label, (point[0] + 7, point[1] - 7), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color, 1, lineType=cv2.LINE_AA)
    return rendered


def errors(prediction: dict[str, Any], target: dict[str, Any]) -> dict[str, float]:
    joints = min(len(prediction["joints"]), len(target["joints_world"]))
    vertices = min(len(prediction["vertices"]), len(target["vertices_world"]))
    return {
        "root_m": float(np.linalg.norm(as_float(prediction["root"]) - as_float(target["root_world"]))),
        "joint_m": float(np.mean(np.linalg.norm(as_float(prediction["joints"])[:joints] - as_float(target["joints_world"])[:joints], axis=-1))),
        "vertex_m": float(np.mean(np.linalg.norm(as_float(prediction["vertices"])[:vertices] - as_float(target["vertices_world"])[:vertices], axis=-1))),
    }


def load_calibration(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if tuple(payload["features"]) != FEATURES or float(payload["alpha"]) != 1.0:
        raise ValueError(f"Not the frozen P5 ridge model: {path}")
    return (
        as_float(payload["scaler_mean"]), as_float(payload["scaler_scale"]),
        as_float(payload["ridge_coef"]), float(payload["ridge_intercept"]), float(payload["clip_m"]),
    )


def p5_people(runtime: dict[str, Any], calibration: tuple[np.ndarray, np.ndarray, np.ndarray, float, float]) -> tuple[list[dict[str, Any]], dict[int, float]]:
    mean, scale, coef, intercept, cap = calibration
    output = [{key: (as_float(value).copy() if key in ("root", "joints", "vertices") else value)
               for key, value in person.items()} for person in runtime["brtc_post_people"]]
    center = as_float(runtime["b0_camera_c2w"])[:3, 3]
    by_post = {int(item["post_index"]): item for item in runtime["brtc"]["people"]}
    by_raw_post = {int(item["detection_index"]): item for item in runtime["b0_post_people"]}
    residuals: dict[int, float] = {}
    for post_index, debug in by_post.items():
        if not bool(debug["accepted"]):
            residuals[post_index] = 0.0
            continue
        feature = as_float([debug["evidence"][name] for name in FEATURES])
        residual = float(np.clip(((feature - mean) / scale) @ coef + intercept, -cap, cap))
        raw_root = as_float(by_raw_post[post_index]["root"])
        ray = raw_root - center
        ray /= max(float(np.linalg.norm(ray)), 1e-12)
        for key in ("root", "joints", "vertices"):
            output[post_index][key] += residual * ray
        residuals[post_index] = residual
    return output, residuals


def cache_rgb(cache_root: Path, record: dict[str, Any]) -> Path:
    return (
        cache_root / "cache" / str(record["sequence"]) / "frame_cache" / "input_frames"
        / f"cam{int(record['post_camera'])}" / f"{int(record['frame']):06d}.jpg"
    )


def rgb_intrinsics(data_root: Path, record: dict[str, Any]) -> np.ndarray:
    calibration_path = data_root / f"{record['sequence']}_original_video" / "calibration_new.json"
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    intrinsic = as_float(payload[str(int(record["post_camera"]))]["K"]).reshape(3, 3)
    # Saved cache RGB frames are 512x512 versions of the 2048x2048 captures.
    intrinsic[:2] *= 0.25
    return intrinsic


def export_case(
    cache_root: Path, case: str, faces: np.ndarray, calibration: tuple[np.ndarray, np.ndarray, np.ndarray, float, float],
    output_dir: Path, data_root: Path, size: int,
) -> dict[str, Any]:
    cache_path = cache_root / "cases" / f"{case}.pt"
    row = torch.load(cache_path, map_location="cpu", weights_only=False)
    if row.get("status") != "ok":
        raise RuntimeError(f"Failed cached event: {case}")
    runtime, evaluator = row["runtime"], row["evaluator"]
    record = runtime["record"]
    source = cache_rgb(cache_root, record)
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    p5, residuals = p5_people(runtime, calibration)
    target_by_detection = evaluator["target_by_detection"]
    labels = evaluator["post_labels_by_detection"]
    c2w = as_float(runtime["b0_camera_c2w"])
    intrinsic = rgb_intrinsics(data_root, record)

    systems = {
        "brtc_lc": runtime["brtc_post_people"],
        "brtc_lc_p5": p5,
        "gt_reference": [
            {
                "detection_index": int(detection),
                "vertices": as_float(target["vertices_world"]),
                "root": as_float(target["root_world"]),
            }
            for detection, target in sorted(target_by_detection.items())
        ],
    }
    for name, people in systems.items():
        labeled = [
            (str(labels.get(int(person["detection_index"]), f"person{index}")), as_float(person["vertices"]))
            for index, person in enumerate(people)
        ]
        rendered = render_people(image, c2w, labeled, faces, intrinsic, size)
        if not cv2.imwrite(str(output_dir / f"{name}.png"), rendered):
            raise OSError(f"Could not write {output_dir / f'{name}.png'}")
    if not cv2.imwrite(str(output_dir / "rgb.png"), cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)):
        raise OSError(f"Could not write {output_dir / 'rgb.png'}")

    person_rows = []
    p5_by_detection = {int(person["detection_index"]): person for person in p5}
    for index, person in enumerate(runtime["brtc_post_people"]):
        detection = int(person["detection_index"])
        target = target_by_detection.get(detection)
        item: dict[str, Any] = {
            "post_detection": detection,
            "evaluator_identity": labels.get(detection),
            "p5_ray_residual_m": residuals.get(detection, 0.0),
        }
        if target is not None:
            before, after = errors(person, target), errors(p5_by_detection[detection], target)
            item["brtc_lc_error_m"] = before
            item["brtc_lc_p5_error_m"] = after
            item["p5_minus_brtc_m"] = {key: after[key] - before[key] for key in before}
        person_rows.append(item)
    return {
        "event_id": case,
        "record": record,
        "source_rgb": str(source),
        "display_intrinsics": intrinsic.tolist(),
        "runtime_methods": {
            "brtc_lc": "frozen B0 camera + frozen BRTC-LC person translation",
            "brtc_lc_p5": "same BRTC-LC output + frozen 7-feature ridge ray residual",
        },
        "gt_reference": "evaluator-only visual reference; not a runtime method/input",
        "runtime_invariants": {
            "model_forward_runs": 0,
            "cuda_allocated": False,
            "camera_changed": False,
            "future_post_frames_used": 0,
            "gt_used_for_runtime": False,
        },
        "people": person_rows,
        "files": {name: f"{name}.png" for name in ("rgb", "brtc_lc", "brtc_lc_p5", "gt_reference")},
    }


def main() -> None:
    # Keep accidental CUDA visibility from turning a torch.load into a GPU map.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    args = parse_args()
    cache_root, model, faces_path, output_dir = (
        ensure_workspace(args.cache_root), ensure_workspace(args.model),
        ensure_workspace(args.faces), ensure_workspace(args.output_dir),
    )
    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    if not (cache_root / "cases").is_dir():
        raise FileNotFoundError(cache_root / "cases")
    calibration = load_calibration(model)
    with np.load(faces_path, allow_pickle=False) as payload:
        faces = np.asarray(payload["f"], dtype=np.int32)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Invalid SMPL-X faces: {faces.shape}")
    manifest = {
        "format": "independent CPU-only Human3R-style post RGB overlays",
        "selection": "two multi-person P5 confirmation events with gains for both visible people",
        "cases": [export_case(cache_root, str(case), faces, calibration, output_dir / str(case), data_root, int(args.size)) for case in args.cases],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
