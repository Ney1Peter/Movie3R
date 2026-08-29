#!/usr/bin/env python3
"""Restore honest Human3R/Bridge3R scene point clouds for two demo cases.

The paper caches intentionally omit dense pointmaps.  This script replays the
frozen RGB-only models only to recover their own depth/confidence background;
all displayed camera, mesh, and identity geometry still comes from the formal
immutable cache.  External baselines are never given an internal point cloud.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import pickle
import shutil
import sys
import tarfile
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import torch


MOVIE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = MOVIE_ROOT.parent
for item in (MOVIE_ROOT, MOVIE_ROOT / "src", MOVIE_ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from dust3r.model import ARCroco3DStereo  # noqa: E402
from publication.bridge3r_iclr2027.runtime_contract import apply_locked_transaction  # noqa: E402
from versions.v14.export_p5_brtc_demo_payload import background_from_prediction  # noqa: E402
from versions.v15.harmony4d.run_harmony_case import (  # noqa: E402
    configure_model,
    gt_helpers,
    run_forward,
    set_event_indices,
    strict_original,
)


CASES = {
    "harmony4d": {
        "case_id": "h4d_test_03_grappling2_028_grappling2_extreme_cam14_cam16_b00076",
        "runtime": MOVIE_ROOT / "output/v15_harmony4d/predictions/test_03_grappling2/h4d_test_03_grappling2_028_grappling2_extreme_cam14_cam16_b00076.runtime.json",
        "cache": MOVIE_ROOT / "output/v15_harmony4d/predictions/test_03_grappling2/h4d_test_03_grappling2_028_grappling2_extreme_cam14_cam16_b00076.npz",
        "rgb_dir": WORKSPACE_ROOT / "data/Harmony4D_work_v17_full_test/external_predictions/trace_harmony4d_v2/test/runtime_inputs/h4d_test_03_grappling2_028_grappling2_extreme_cam14_cam16_b00076/images",
    },
    "egohumans": {
        "case_id": "ego_test_legoassemble_003_legoassemble_extreme_cam03_cam04_b00301",
        "runtime": MOVIE_ROOT / "output/v19_egohumans/test/predictions/legoassemble__003_legoassemble-002/ego_test_legoassemble_003_legoassemble_extreme_cam03_cam04_b00301.runtime.json",
        "cache": MOVIE_ROOT / "output/v19_egohumans/test/predictions/legoassemble__003_legoassemble-002/ego_test_legoassemble_003_legoassemble_extreme_cam03_cam04_b00301.npz",
        "outer_zip": WORKSPACE_ROOT / "data/EgoHuman.zip",
        "trace_manifest": WORKSPACE_ROOT / "data/EgoHuman_work_v19/external_predictions/trace_egohumans_v2/manifests/egohumans_test.runtime.jsonl",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(CASES), required=True)
    parser.add_argument("--method", choices=("strict", "bridge3r"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--pre-display", type=int, default=5)
    parser.add_argument("--post-display", type=int, default=25)
    parser.add_argument(
        "--output",
        type=Path,
        default=MOVIE_ROOT / "output/bridge3r_two_dataset_demo_v2",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ego_manifest_row(case_id: str, path: Path) -> dict[str, Any]:
    matches = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            row = json.loads(line)
            if row.get("case_id") == case_id:
                matches.append(row)
    if len(matches) != 1:
        raise ValueError(f"Expected one EgoHumans runtime row for {case_id}, found {len(matches)}")
    return matches[0]


def extract_ego_prefix(spec: dict[str, Any], runtime: dict[str, Any], count: int, root: Path) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    outputs = [root / f"{index:06d}.jpg" for index in range(count)]
    if all(path.is_file() and path.stat().st_size > 0 for path in outputs):
        return outputs
    row = ego_manifest_row(str(spec["case_id"]), Path(spec["trace_manifest"]))
    members = [str(value) for value in row["image_members"][:count]]
    wanted = {name: outputs[index] for index, name in enumerate(members)}
    found: set[str] = set()
    with zipfile.ZipFile(Path(spec["outer_zip"])) as outer:
        with outer.open(str(runtime["record"]["archive_entry"])) as nested:
            with tarfile.open(fileobj=nested, mode="r|gz") as archive:
                for member in archive:
                    parts = tuple(part for part in member.name.lstrip("./").split("/") if part)
                    logical = "/".join(parts[8:])
                    if logical not in wanted:
                        continue
                    source = archive.extractfile(member)
                    if source is None:
                        raise OSError(f"Could not read {logical}")
                    payload = source.read()
                    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if image is None or not cv2.imwrite(str(wanted[logical]), image, [cv2.IMWRITE_JPEG_QUALITY, 100]):
                        raise OSError(f"Could not decode/write {logical}")
                    found.add(logical)
                    if len(found) == len(wanted):
                        break
    missing = sorted(set(wanted) - found)
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} EgoHumans RGB frames: {missing[:3]}")
    return outputs


def input_paths(
    dataset: str,
    spec: dict[str, Any],
    runtime: dict[str, Any],
    post_count: int,
    output: Path,
) -> tuple[list[Path], list[Path]]:
    boundary = int(runtime["record"]["boundary_index"])
    if dataset == "harmony4d":
        all_paths = sorted(Path(spec["rgb_dir"]).glob("*.jpg"))
        if len(all_paths) != int(runtime["record"]["clip_length"]):
            raise ValueError(f"Unexpected Harmony4D RGB count: {len(all_paths)}")
        paths = all_paths[: boundary + post_count]
    else:
        paths = extract_ego_prefix(
            spec,
            runtime,
            boundary + post_count,
            output / dataset / "_model_replay_rgb",
        )
    return paths[:boundary], paths[boundary:]


def cache_arrays(path: Path, prefix: str) -> dict[str, np.ndarray]:
    keys = (
        "cameras_c2w",
        "vertices_world",
        "joints_world",
        "persistent_ids",
        "native_ids",
        "valid",
    )
    with np.load(path, allow_pickle=False) as source:
        return {key: np.asarray(source[f"{prefix}__{key}"]).copy() for key in keys}


def formal_geometry(
    method: str,
    cache: Path,
    runtime: dict[str, Any],
    indices: list[int],
) -> list[dict[str, np.ndarray]]:
    if method == "strict":
        arrays = cache_arrays(cache, "m0_strict_human3r")
    else:
        arrays = cache_arrays(cache, "m3_b0_only")
        pairs = [tuple(map(int, pair)) for pair in runtime["geometry"]["association"]["pairs"]]
        arrays, _ = apply_locked_transaction(
            arrays,
            boundary=int(runtime["record"]["boundary_index"]),
            pairs=pairs,
            cut_detected=True,
        )
    frames = []
    for index in indices:
        valid = np.asarray(arrays["valid"][index]).astype(bool)
        frames.append(
            {
                "camera": np.asarray(arrays["cameras_c2w"][index], dtype=np.float32),
                "vertices": np.asarray(arrays["vertices_world"][index, valid], dtype=np.float32),
                "ids": np.asarray(arrays["persistent_ids"][index, valid], dtype=np.int64),
            }
        )
    return frames


def load_faces() -> np.ndarray:
    with (MOVIE_ROOT / "src/models/smpl/SMPL_NEUTRAL.pkl").open("rb") as handle:
        model = pickle.load(handle, encoding="latin1")
    return np.asarray(model["f"], dtype=np.int32)


def prediction_backgrounds(
    method: str,
    model: ARCroco3DStereo,
    pre: list[Path],
    post: list[Path],
    size: int,
    device: torch.device,
    pre_display: int,
) -> tuple[list[dict[str, np.ndarray]], dict[str, Any]]:
    options = SimpleNamespace(size=int(size))
    if method == "strict":
        views = set_event_indices(
            gt_helpers.prepare_full_square_input(model, pre + post, options), set()
        )
        predictions, returned, debug, timing = run_forward(model, views, device, "strict_pointcloud_replay")
        selected = list(range(len(pre) - pre_display, len(pre) + len(post)))
        backgrounds = [background_from_prediction(predictions[index], returned[index]) for index in selected]
        del views, predictions, returned, debug
        return backgrounds, timing
    pre_views = gt_helpers.prepare_full_square_input(model, pre, options)
    post_views = gt_helpers.prepare_full_square_input(model, post, options)
    shadow_views = set_event_indices(copy.deepcopy(pre_views + post_views[:1]), {len(pre_views)})
    raw_views = set_event_indices(copy.deepcopy(post_views), set())
    shadow_predictions, shadow_returned, shadow_debug, shadow_timing = run_forward(
        model, shadow_views, device, "bridge3r_shadow_pointcloud_replay"
    )
    raw_predictions, raw_returned, raw_debug, raw_timing = run_forward(
        model, raw_views, device, "bridge3r_clean_reset_pointcloud_replay"
    )
    backgrounds = [
        background_from_prediction(shadow_predictions[index], shadow_returned[index])
        for index in range(len(pre) - pre_display, len(pre))
    ]
    backgrounds.extend(
        background_from_prediction(raw_predictions[index], raw_returned[index])
        for index in range(len(post))
    )
    del pre_views, post_views, shadow_views, raw_views
    del shadow_predictions, shadow_returned, shadow_debug, raw_predictions, raw_returned, raw_debug
    return backgrounds, {"shadow": shadow_timing, "clean_reset_post": raw_timing}


def write_payload(
    root: Path,
    backgrounds: list[dict[str, np.ndarray]],
    geometry: list[dict[str, np.ndarray]],
    faces: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    if len(backgrounds) != len(geometry):
        raise ValueError(f"Background/geometry mismatch: {len(backgrounds)} vs {len(geometry)}")
    if root.exists():
        resolved = root.resolve()
        allowed = (MOVIE_ROOT / "output/bridge3r_two_dataset_demo_v2").resolve()
        if resolved == allowed or allowed not in resolved.parents:
            raise ValueError(f"Unsafe payload replacement: {resolved}")
        shutil.rmtree(root)
    for name in ("depth", "conf", "color", "camera", "smpl"):
        (root / name).mkdir(parents=True, exist_ok=True)
    for index, (background, frame) in enumerate(zip(backgrounds, geometry)):
        color = np.clip(np.asarray(background["color"]) * 255.0, 0, 255).astype(np.uint8)
        if not cv2.imwrite(str(root / "color" / f"{index:06d}.png"), cv2.cvtColor(color, cv2.COLOR_RGB2BGR)):
            raise OSError(f"Could not write RGB {index}")
        np.save(root / "depth" / f"{index:06d}.npy", np.asarray(background["depth"], dtype=np.float32))
        np.save(root / "conf" / f"{index:06d}.npy", np.asarray(background["conf"], dtype=np.float32))
        np.savez(
            root / "camera" / f"{index:06d}.npz",
            pose=frame["camera"],
            intrinsics=np.asarray(background["intrinsics"], dtype=np.float32),
        )
        count = len(frame["vertices"])
        np.savez(
            root / "smpl" / f"{index:06d}.npz",
            scores=np.zeros_like(background["depth"], dtype=np.float32),
            msk=np.asarray(background["mask"], dtype=np.float32),
            shape=np.zeros((count, 10), dtype=np.float32),
            rotvec=np.zeros((count, 53, 3), dtype=np.float32),
            transl=np.zeros((count, 3), dtype=np.float32),
            expression=np.zeros((count, 10), dtype=np.float32),
            smpl_id=frame["ids"],
            verts_world=frame["vertices"],
            faces=faces,
        )
    (root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    allowed = (MOVIE_ROOT / "output").resolve()
    if output == allowed or allowed not in output.parents:
        raise ValueError(f"Output must stay below {allowed}: {output}")
    spec = CASES[args.dataset]
    runtime = read_json(Path(spec["runtime"]))
    record = runtime["record"]
    if record["case_id"] != spec["case_id"]:
        raise ValueError("Runtime case mismatch")
    boundary = int(record["boundary_index"])
    pre_display, post_display = int(args.pre_display), int(args.post_display)
    pre, post = input_paths(args.dataset, spec, runtime, post_display, output)
    if len(pre) != boundary or len(post) != post_display:
        raise ValueError((len(pre), len(post), boundary, post_display))
    indices = list(range(boundary - pre_display, boundary + post_display))
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    checkpoint_key = "original" if args.method == "strict" else "current"
    checkpoint = Path(runtime["checkpoint"][checkpoint_key])
    started = time.perf_counter()
    model = ARCroco3DStereo.from_pretrained(str(checkpoint)).to(device)
    flags: dict[str, Any] | None
    if args.method == "strict":
        strict_original(model)
        flags = None
    else:
        flags = configure_model(model)
    model.eval()
    backgrounds, timing = prediction_backgrounds(
        args.method, model, pre, post, int(args.size), device, pre_display
    )
    geometry = formal_geometry(args.method, Path(spec["cache"]), runtime, indices)
    payload = output / args.dataset / "payloads" / args.method
    write_payload(
        payload,
        backgrounds,
        geometry,
        load_faces(),
        {
            "schema_version": "Bridge3R-demo-pointcloud-replay-v1",
            "dataset": args.dataset,
            "case_id": spec["case_id"],
            "method": args.method,
            "scene_channel_available": True,
            "scene_source": "the method's own frozen checkpoint replay",
            "human_camera_geometry_source": "immutable formal test cache",
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_flags": flags,
            "frame_indices": indices,
            "cut_index_in_payload": pre_display,
            "gt_used": False,
            "external_baseline_geometry_borrowed": False,
            "timing": timing,
        },
    )
    del model, backgrounds, geometry
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(json.dumps({
        "payload": str(payload),
        "frames": len(indices),
        "method": args.method,
        "dataset": args.dataset,
        "seconds_total": time.perf_counter() - started,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
