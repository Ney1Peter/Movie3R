#!/usr/bin/env python3
"""Generate one shared per-shot Human3R cache for registration baselines.

This runtime process accepts only an RGB/runtime manifest.  It has no
evaluator-manifest or GT argument.  One original Human3R model is loaded per
worker and reused for every assigned case.
"""

from __future__ import annotations

import argparse
import gc
import json
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


REPO_ROOT = Path(__file__).resolve().parents[2]
for root in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from dust3r.model import ARCroco3DStereo  # noqa: E402
from dust3r.utils.smpl_layer import SMPL_Layer  # noqa: E402
from versions.v13 import gt_id_consensus as gt_helpers  # noqa: E402
from versions.v15.harmony4d import run_harmony_case as frozen  # noqa: E402
from versions.v15.harmony4d.topology import CommonTopology  # noqa: E402


SCHEMA = "Shot3R-traditional-registration-R0-scene-cache-v1"
EXPECTED_CHECKPOINT_SHA256 = "1c5d89077d7734476ce74183df178c51ad172cad5e256081e61480cf231a9377"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--staged-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("egobody", "egohumans"), required=True)
    parser.add_argument("--split", choices=("development", "holdout", "test", "smoke"), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "src/human3r_896L.pth")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--lines", help="Optional comma-separated one-based manifest lines")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--scene-grid-stride", type=int, default=8)
    parser.add_argument("--scene-cache-voxel", type=float, default=0.03)
    parser.add_argument("--scene-max-points", type=int, default=250000)
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(frozen.jsonable(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(partial, path)


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty runtime manifest: {path}")
    forbidden = [
        (index, key) for index, row in enumerate(rows, start=1) for key in row
        if str(key).endswith("_evaluator_only") or str(key).startswith("gt_")
    ]
    if forbidden:
        raise ValueError(f"evaluator/GT field leaked into runtime manifest: {forbidden[:3]}")
    return rows


def image_paths(record: dict[str, Any], staged_root: Path) -> list[Path]:
    values = record.get("image_paths") or record.get("image_members")
    if not isinstance(values, list):
        raise ValueError("runtime row has no image path list")
    expected = len(record["pre_frame_numbers"]) + len(record["post_frame_numbers"])
    if len(values) != expected:
        raise ValueError(f"RGB count {len(values)} != {expected}")
    root = staged_root.resolve()
    output = []
    for raw in values:
        path = Path(str(raw))
        candidate = path.resolve() if path.is_absolute() else (root / path).resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError(f"RGB path escapes staging root: {raw}")
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        output.append(candidate)
    return output


def _tensor_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)


def _mask_at_resolution(prediction: dict[str, Any], height: int, width: int) -> tuple[np.ndarray, bool]:
    raw = prediction.get("msk")
    if raw is None:
        return np.zeros((height, width), dtype=bool), False
    value = _tensor_numpy(raw).squeeze()
    while value.ndim > 2:
        value = value[..., 0]
    if value.ndim != 2:
        return np.zeros((height, width), dtype=bool), False
    if value.shape != (height, width):
        tensor = torch.as_tensor(value, dtype=torch.float32)[None, None]
        value = torch.nn.functional.interpolate(tensor, size=(height, width), mode="nearest")[0, 0].numpy()
    # Human3R's mask head returns sigmoid probabilities.  The repository's
    # released geometry/export utilities use 0.10 for the predicted foreground
    # mask; using merely ``>0`` would incorrectly remove every sigmoid pixel.
    return value > 0.10, True


def scene_frame(prediction: dict[str, Any], stride: int) -> tuple[np.ndarray, np.ndarray, bool]:
    raw_points = prediction.get("pts3d_in_self_view")
    if raw_points is None:
        return np.empty((0, 3), np.float32), np.empty((0,), np.float32), False
    points = _tensor_numpy(raw_points)
    if points.ndim == 4:
        points = points[0]
    if points.ndim != 3 or points.shape[-1] != 3:
        return np.empty((0, 3), np.float32), np.empty((0,), np.float32), False
    height, width = points.shape[:2]
    confidence = prediction.get("conf_self")
    if confidence is None:
        conf = np.ones((height, width), dtype=np.float32)
    else:
        conf = _tensor_numpy(confidence).squeeze()
        if conf.shape != (height, width):
            conf = torch.nn.functional.interpolate(
                torch.as_tensor(conf, dtype=torch.float32)[None, None],
                size=(height, width), mode="bilinear", align_corners=False,
            )[0, 0].numpy()
    mask, has_mask = _mask_at_resolution(prediction, height, width)
    offset = max(int(stride) // 2, 0)
    yy = np.arange(offset, height, max(int(stride), 1), dtype=np.int64)
    xx = np.arange(offset, width, max(int(stride), 1), dtype=np.int64)
    grid_y, grid_x = np.meshgrid(yy, xx, indexing="ij")
    selected_points = points[grid_y, grid_x].reshape(-1, 3).astype(np.float32)
    selected_conf = conf[grid_y, grid_x].reshape(-1).astype(np.float32)
    selected_mask = mask[grid_y, grid_x].reshape(-1)
    valid = np.isfinite(selected_points).all(axis=1) & np.isfinite(selected_conf)
    valid &= (selected_points[:, 2] > 0.05) & (selected_points[:, 2] < 50.0)
    valid &= ~selected_mask
    selected_points, selected_conf = selected_points[valid], selected_conf[valid]
    if len(selected_points):
        camera = np.asarray(frozen.camera_matrix(prediction), dtype=np.float64)
        selected_points = (
            selected_points @ camera[:3, :3].T + camera[:3, 3]
        ).astype(np.float32)
    return selected_points, selected_conf, has_mask


def fuse_scene(
    predictions: list[dict[str, Any]], grid_stride: int, cache_voxel: float,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    point_rows, confidence_rows, mask_rows = [], [], []
    for prediction in predictions:
        points, confidence, has_mask = scene_frame(prediction, grid_stride)
        point_rows.append(points)
        confidence_rows.append(confidence)
        mask_rows.append(has_mask)
    if not any(len(value) for value in point_rows):
        return (
            np.empty((0, 3), np.float32), np.empty((0,), np.float32),
            {"raw_selected_points": 0, "mask_available_frames": int(sum(mask_rows))},
        )
    points = np.concatenate(point_rows, axis=0)
    confidence = np.concatenate(confidence_rows, axis=0)
    raw_count = len(points)
    order = np.argsort(-confidence, kind="stable")
    quantized = np.floor(points[order] / float(cache_voxel)).astype(np.int64)
    _, unique_in_order = np.unique(quantized, axis=0, return_index=True)
    keep = order[np.sort(unique_in_order)]
    if len(keep) > int(max_points):
        top = np.argsort(-confidence[keep], kind="stable")[: int(max_points)]
        keep = keep[top]
    return (
        points[keep].astype(np.float32), confidence[keep].astype(np.float32),
        {
            "raw_selected_points": raw_count,
            "cached_points": len(keep),
            "mask_available_frames": int(sum(mask_rows)),
            "frame_count": len(predictions),
            "grid_stride": int(grid_stride),
            "cache_voxel_m": float(cache_voxel),
        },
    )


def run_shot(
    model: ARCroco3DStereo, layer: SMPL_Layer, topology: CommonTopology,
    paths: list[Path], device: torch.device, size: int, label: str,
    grid_stride: int, cache_voxel: float, max_points: int,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, dict[str, Any]]:
    views = frozen.set_event_indices(
        gt_helpers.prepare_full_square_input(model, paths, SimpleNamespace(size=int(size))),
        set(),
    )
    predictions, returned, debug, forward_runtime = frozen.run_forward(model, views, device, label)
    frames = frozen.decode_sequence(predictions, returned, debug, layer, topology)
    scene_points, scene_conf, scene_runtime = fuse_scene(
        predictions, grid_stride, cache_voxel, max_points
    )
    del predictions, returned, debug, views
    return frames, scene_points, scene_conf, {
        "forward": forward_runtime, "scene": scene_runtime
    }


def save_case(
    record: dict[str, Any], paths: list[Path], model: ARCroco3DStereo,
    layer: SMPL_Layer, topology: CommonTopology, args: argparse.Namespace,
    line_number: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    boundary = int(record["boundary_index"])
    pre, pre_points, pre_conf, pre_runtime = run_shot(
        model, layer, topology, paths[:boundary], torch.device(args.device), args.size,
        f"{record['case_id']}:shotA", args.scene_grid_stride,
        args.scene_cache_voxel, args.scene_max_points,
    )
    post, post_points, post_conf, post_runtime = run_shot(
        model, layer, topology, paths[boundary:], torch.device(args.device), args.size,
        f"{record['case_id']}:shotB", args.scene_grid_stride,
        args.scene_cache_voxel, args.scene_max_points,
    )
    packed = frozen.pack_methods({"r0": pre + post}, topology)
    packed.update({
        "scene_pre_points": pre_points,
        "scene_pre_confidence": pre_conf,
        "scene_post_points": post_points,
        "scene_post_confidence": post_conf,
    })
    cache = args.output_root / "predictions" / args.dataset / args.split / f"{record['case_id']}.r0_scene.npz"
    report = args.output_root / "logs" / "inference" / args.dataset / args.split / f"{record['case_id']}.runtime.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    partial = cache.with_suffix(cache.suffix + ".partial")
    with partial.open("wb") as handle:
        np.savez_compressed(handle, **packed)
    os.replace(partial, cache)
    runtime = {
        "schema_version": SCHEMA,
        "dataset": args.dataset,
        "split": args.split,
        "case_id": record["case_id"],
        "record": record,
        "runtime_manifest": str(args.manifest.resolve()),
        "manifest_line": int(line_number),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_expected_sha256": EXPECTED_CHECKPOINT_SHA256,
        "gt_used_for_runtime": False,
        "evaluator_manifest_opened": False,
        "boundary_index": boundary,
        "shot_A": pre_runtime,
        "shot_B": post_runtime,
        "scene_mask_source": "Human3R predicted msk; absent frames retain unmasked prediction points",
        "total_seconds": time.perf_counter() - started,
        "cache": str(cache.resolve()),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": args.device,
            "gpu": torch.cuda.get_device_name(torch.device(args.device)),
            "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        },
    }
    atomic_json(report, runtime)
    return {"case_id": record["case_id"], "cache": str(cache), "report": str(report), "seconds": runtime["total_seconds"]}


def reusable(cache: Path, report: Path, case_id: str) -> bool:
    if not cache.is_file() or not report.is_file():
        return False
    try:
        value = json.loads(report.read_text(encoding="utf-8"))
        if value.get("case_id") != case_id or value.get("schema_version") != SCHEMA:
            return False
        with np.load(cache, allow_pickle=False) as arrays:
            return all(key in arrays.files for key in (
                "r0__cameras_c2w", "r0__vertices_world", "r0__joints_world",
                "r0__persistent_ids", "r0__native_ids", "r0__valid",
                "scene_pre_points", "scene_post_points",
            ))
    except Exception:
        return False


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard specification")
    rows = read_rows(args.manifest.resolve())
    requested = None
    if args.lines:
        requested = {int(value) for value in args.lines.split(",") if value.strip()}
    selected = [
        (index, row) for index, row in enumerate(rows, start=1)
        if (requested is None or index in requested)
        and ((index - 1) % args.num_shards == args.shard_index)
    ]
    if args.max_cases is not None:
        selected = selected[: int(args.max_cases)]
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file() or checkpoint.stat().st_size != 4670554642:
        raise FileNotFoundError(f"missing/unexpected original Human3R checkpoint: {checkpoint}")
    topology = CommonTopology.load()
    model = ARCroco3DStereo.from_pretrained(str(checkpoint)).to(device)
    frozen.strict_original(model)
    model.eval()
    layer = SMPL_Layer(
        type="smplx", gender="neutral", num_betas=10, kid=False,
        person_center="head",
    ).to(device).eval()
    completed, failed, skipped = [], [], []
    for position, (line_number, record) in enumerate(selected, start=1):
        case_id = str(record["case_id"])
        cache = args.output_root / "predictions" / args.dataset / args.split / f"{case_id}.r0_scene.npz"
        report = args.output_root / "logs" / "inference" / args.dataset / args.split / f"{case_id}.runtime.json"
        if reusable(cache, report, case_id):
            skipped.append(case_id)
            print(f"[{position}/{len(selected)}] reusable {case_id}", flush=True)
            continue
        try:
            paths = image_paths(record, args.staged_root)
            result = save_case(record, paths, model, layer, topology, args, line_number)
            completed.append(result)
            print(f"[{position}/{len(selected)}] complete {case_id} {result['seconds']:.1f}s", flush=True)
        except Exception as exc:
            failed.append({"case_id": case_id, "line": line_number, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{position}/{len(selected)}] FAILED {case_id}: {failed[-1]['error']}", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    ledger = args.output_root / "logs" / f"infer_{args.dataset}_{args.split}_shard{args.shard_index:02d}.json"
    atomic_json(ledger, {
        "schema_version": "Shot3R-registration-inference-shard-ledger-v1",
        "dataset": args.dataset, "split": args.split,
        "shard_index": args.shard_index, "num_shards": args.num_shards,
        "selected_count": len(selected), "completed": completed,
        "reused": skipped, "failures": failed,
    })
    if failed:
        raise SystemExit(f"{len(failed)} inference cases failed; see {ledger}")


if __name__ == "__main__":
    main()
