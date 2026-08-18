#!/usr/bin/env python3
"""Restore demo.py backgrounds for selected immutable Harmony4D test caches.

The compact paper caches contain exact cameras, common-SMPL meshes, and IDs but
omit dense pointmaps.  This exporter causally replays the frozen checkpoints to
restore RGB/depth/confidence only.  Displayed geometry is always read from the
immutable test cache; evaluator GT is never read.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import pickle
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
for item in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from dust3r.model import ARCroco3DStereo  # noqa: E402
from versions.v14.export_p5_brtc_demo_payload import background_from_prediction  # noqa: E402
from versions.v15.harmony4d.run_harmony_case import (  # noqa: E402
    configure_model,
    frame_image_paths,
    gt_helpers,
    run_forward,
    set_event_indices,
    strict_original,
)
from versions.v15.harmony4d.topology import SMPL_NEUTRAL  # noqa: E402


M0 = "m0_strict_human3r"
M15 = "m15_safe_boundary_permutation_causal_gru"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--pre-display", type=int, default=5)
    parser.add_argument("--post-display", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_destination(path: Path, overwrite: bool) -> Path:
    destination = path.resolve()
    output_root = (REPO_ROOT / "output").resolve()
    if destination != output_root and output_root not in destination.parents:
        raise ValueError(f"Qualitative output must stay under {output_root}: {destination}")
    if destination.exists():
        if not overwrite:
            raise FileExistsError(destination)
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    return destination


def make_payload_root(path: Path) -> None:
    for name in ("depth", "conf", "color", "camera", "smpl"):
        (path / name).mkdir(parents=True, exist_ok=True)


def load_faces() -> np.ndarray:
    with SMPL_NEUTRAL.open("rb") as handle:
        model = pickle.load(handle, encoding="latin1")
    return np.asarray(model["f"], dtype=np.int32)


def method_geometry(cache_path: Path, method: str, indices: list[int]) -> list[dict[str, np.ndarray]]:
    prefix = method + "__"
    with np.load(cache_path, allow_pickle=False) as cache:
        cameras = np.asarray(cache[prefix + "cameras_c2w"])[indices]
        vertices = np.asarray(cache[prefix + "vertices_world"])[indices]
        ids = np.asarray(cache[prefix + "persistent_ids"])[indices]
        valid = np.asarray(cache[prefix + "valid"])[indices].astype(bool)
    return [
        {
            "camera": cameras[index],
            "vertices": vertices[index, valid[index]],
            "ids": ids[index, valid[index]],
        }
        for index in range(len(indices))
    ]


def write_payload(
    destination: Path,
    backgrounds: list[dict[str, np.ndarray]],
    geometry: list[dict[str, np.ndarray]],
    faces: np.ndarray,
) -> None:
    if len(backgrounds) != len(geometry):
        raise ValueError(f"Background/geometry mismatch: {len(backgrounds)} vs {len(geometry)}")
    make_payload_root(destination)
    for index, (background, frame) in enumerate(zip(backgrounds, geometry)):
        color = np.clip(background["color"] * 255.0, 0, 255).astype(np.uint8)
        if not cv2.imwrite(str(destination / "color" / f"{index:06d}.png"), cv2.cvtColor(color, cv2.COLOR_RGB2BGR)):
            raise OSError(f"Could not write RGB {index}")
        np.save(destination / "depth" / f"{index:06d}.npy", np.asarray(background["depth"], dtype=np.float32))
        np.save(destination / "conf" / f"{index:06d}.npy", np.asarray(background["conf"], dtype=np.float32))
        np.savez(
            destination / "camera" / f"{index:06d}.npz",
            pose=np.asarray(frame["camera"], dtype=np.float32),
            intrinsics=np.asarray(background["intrinsics"], dtype=np.float32),
        )
        count = len(frame["vertices"])
        np.savez(
            destination / "smpl" / f"{index:06d}.npz",
            scores=np.zeros_like(background["depth"], dtype=np.float32),
            msk=np.asarray(background["mask"], dtype=np.float32),
            shape=np.zeros((count, 10), dtype=np.float32),
            rotvec=np.zeros((count, 53, 3), dtype=np.float32),
            transl=np.zeros((count, 3), dtype=np.float32),
            expression=np.zeros((count, 10), dtype=np.float32),
            smpl_id=np.asarray(frame["ids"], dtype=np.int64),
            verts_world=np.asarray(frame["vertices"], dtype=np.float32),
            faces=faces,
        )


def selected_paths(item: dict[str, Any], pre_display: int, post_display: int) -> tuple[dict[str, Any], list[Path], list[Path], list[int]]:
    runtime = json.loads(Path(item["runtime"]).read_text(encoding="utf-8"))
    record = runtime["record"]
    all_pre, all_post = frame_image_paths(Path(item["extracted_root"]) / str(record["capture_relative"]), record)
    if len(all_pre) != int(record["boundary_index"]):
        raise ValueError(f"Unexpected pre length for {record['case_id']}")
    if pre_display > len(all_pre) or post_display > len(all_post):
        raise ValueError(f"Display window too long for {record['case_id']}")
    pre = all_pre
    post = all_post[:post_display]
    indices = list(range(len(all_pre) - pre_display, len(all_pre))) + list(range(len(all_pre), len(all_pre) + post_display))
    return runtime, pre, post, indices


def backgrounds_from_rows(predictions: list[dict[str, Any]], returned: list[dict[str, Any]], indices: list[int]) -> list[dict[str, np.ndarray]]:
    return [background_from_prediction(predictions[index], returned[index]) for index in indices]


def release(*values: Any) -> None:
    del values
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    selection_path = args.selection.resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    cases = selection["cases"]
    destination = safe_destination(args.output, bool(args.overwrite))
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    faces = load_faces()
    prepared = []
    for item in cases:
        runtime, pre, post, indices = selected_paths(item, int(args.pre_display), int(args.post_display))
        cache_path = Path(runtime["cache"]).resolve()
        prepared.append({**item, "runtime_payload": runtime, "cache": cache_path, "pre": pre, "post": post, "indices": indices})
    current_paths = {row["runtime_payload"]["checkpoint"]["current"] for row in prepared}
    original_paths = {row["runtime_payload"]["checkpoint"]["original"] for row in prepared}
    if len(current_paths) != 1 or len(original_paths) != 1:
        raise ValueError("Selected cases do not share frozen checkpoints")

    timing: dict[str, dict[str, float]] = defaultdict(dict)
    original_path = Path(next(iter(original_paths)))
    original = ARCroco3DStereo.from_pretrained(str(original_path)).to(device)
    strict_original(original); original.eval()
    for row in prepared:
        case_id = row["runtime_payload"]["record"]["case_id"]
        paths = row["pre"] + row["post"]
        views = set_event_indices(gt_helpers.prepare_full_square_input(original, paths, SimpleNamespace(size=int(args.size))), set())
        started = time.perf_counter()
        predictions, returned, debug, _ = run_forward(original, views, device, "qualitative_strict_original")
        timing[case_id]["m0_background_seconds"] = time.perf_counter() - started
        start = len(row["pre"]) - int(args.pre_display)
        displayed = list(range(start, len(row["pre"]))) + list(range(len(row["pre"]), len(paths)))
        backgrounds = backgrounds_from_rows(predictions, returned, displayed)
        geometry = method_geometry(row["cache"], M0, row["indices"])
        root = destination / case_id / "strict_human3r"
        write_payload(root, backgrounds, geometry, faces)
        del predictions, returned, debug, views, backgrounds, geometry
        release()
    del original; release()

    current_path = Path(next(iter(current_paths)))
    current = ARCroco3DStereo.from_pretrained(str(current_path)).to(device)
    flags = configure_model(current); current.eval()
    for row in prepared:
        case_id = row["runtime_payload"]["record"]["case_id"]
        pre_views = gt_helpers.prepare_full_square_input(current, row["pre"], SimpleNamespace(size=int(args.size)))
        post_views = gt_helpers.prepare_full_square_input(current, row["post"], SimpleNamespace(size=int(args.size)))
        shadow_views = set_event_indices(copy.deepcopy(pre_views + post_views[:1]), {len(pre_views)})
        raw_views = set_event_indices(copy.deepcopy(post_views), set())
        started = time.perf_counter()
        shadow_predictions, shadow_returned, shadow_debug, _ = run_forward(current, shadow_views, device, "qualitative_shadow")
        raw_predictions, raw_returned, raw_debug, _ = run_forward(current, raw_views, device, "qualitative_raw_post")
        timing[case_id]["m15_background_seconds"] = time.perf_counter() - started
        pre_indices = list(range(len(row["pre"]) - int(args.pre_display), len(row["pre"])))
        backgrounds = backgrounds_from_rows(shadow_predictions, shadow_returned, pre_indices)
        backgrounds.extend(backgrounds_from_rows(raw_predictions, raw_returned, list(range(len(row["post"])))))
        geometry = method_geometry(row["cache"], M15, row["indices"])
        root = destination / case_id / "movie3r_m15_causal"
        write_payload(root, backgrounds, geometry, faces)
        del pre_views, post_views, shadow_views, raw_views
        del shadow_predictions, shadow_returned, shadow_debug, raw_predictions, raw_returned, raw_debug
        del backgrounds, geometry
        release()
    del current; release()

    manifest = {
        "schema_version": "Movie3R-Harmony4D-qualitative-v1",
        "format": "standard demo.py --save compatible",
        "selection": str(selection_path),
        "frame_layout": {"pre": int(args.pre_display), "post": int(args.post_display), "cut_index": int(args.pre_display)},
        "methods": {"baseline": M0, "primary": M15},
        "checkpoint": {"original": str(original_path), "current": str(current_path), "current_flags": flags},
        "device": str(device),
        "cases": [{
            "case_id": row["runtime_payload"]["record"]["case_id"],
            "categories": row["categories"],
            "cache": str(row["cache"]),
            "timing": timing[row["runtime_payload"]["record"]["case_id"]],
            "baseline_payload": str(destination / row["runtime_payload"]["record"]["case_id"] / "strict_human3r"),
            "primary_payload": str(destination / row["runtime_payload"]["record"]["case_id"] / "movie3r_m15_causal"),
        } for row in prepared],
        "contract": {
            "geometry_source": "immutable formal test caches",
            "background_source": "causal frozen-checkpoint replay for visualization only",
            "gt_used": False,
            "future_frames_used_for_geometry": 0,
            "metrics_or_gate_modified": False,
        },
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(destination), "cases": len(prepared), "payloads": 2 * len(prepared)}, indent=2))


if __name__ == "__main__":
    main()
