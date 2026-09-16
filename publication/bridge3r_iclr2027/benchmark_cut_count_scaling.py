#!/usr/bin/env python3
"""Measure Shot3R core runtime and peak VRAM versus shot-transition count.

The protocol fixes one 300-frame RGB stream and evaluates 0, 1, 3, and 5
annotated transition schedules.  Preprocessing, checkpoint loading, and the
transition detector are excluded, while neural reconstruction, SMPL decoding,
prediction-only association, and the locked camera--human transaction are
included.  A transition is processed with one additional boundary-frame
evaluation, so a C-cut route evaluates exactly 300+C neural frames.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
for root in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from dust3r.model import ARCroco3DStereo  # noqa: E402
from dust3r.utils.smpl_layer import SMPL_Layer  # noqa: E402
from publication.bridge3r_iclr2027.runtime_contract import apply_locked_transaction  # noqa: E402
from versions.v13 import gt_id_consensus as gt_helpers  # noqa: E402
from versions.v14.run_v14_2_single_sequence import configure_model, set_event_indices  # noqa: E402
from versions.v15.harmony4d.run_harmony_case import (  # noqa: E402
    decode_sequence,
    frame_image_paths,
    map_frames,
    pack_methods,
    persistent_post,
)
from versions.v15.harmony4d.topology import CommonTopology  # noqa: E402


SCHEMA = "Shot3R-cut-count-runtime-memory-v1"
DEFAULT_MANIFEST = (
    WORKSPACE_ROOT
    / "ICLR-paper/bridge3r_iclr2027/private_audit/egohumans_formal90_manifest.jsonl"
)
DEFAULT_CASE = "ego_test_fencing_002_fencing_extreme_cam10_cam07_b00301"
DEFAULT_CURRENT = (
    REPO_ROOT
    / "output/v14_cut_first_cross_source/"
    "v14_cut_first_cross_source_multihuman_p0_e6/checkpoint-final.pth"
)
CUT_SCHEDULES = {
    0: [],
    1: [150],
    3: [75, 150, 225],
    5: [50, 100, 150, 200, 250],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--case-id", default=DEFAULT_CASE)
    parser.add_argument("--extracted-root", type=Path, required=True)
    parser.add_argument(
        "--source-layout",
        choices=("egohumans", "image-members"),
        default="egohumans",
        help="EgoHumans exo layout or a manifest row with rooted image_members.",
    )
    parser.add_argument("--current-checkpoint", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def load_record(manifest: Path, case_id: str, source_layout: str) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [row for row in rows if str(row.get("case_id")) == case_id]
    if len(selected) != 1:
        raise ValueError(f"expected one record for {case_id}, found {len(selected)}")
    expected = 100 if source_layout == "egohumans" else 150
    if int(selected[0]["clip_length"]) != expected:
        raise ValueError(f"the frozen source case must contain {expected} frames")
    return selected[0]


def synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def forward(
    model: ARCroco3DStereo,
    views: list[dict[str, Any]],
    device: torch.device,
) -> tuple[list[dict], list[dict], list[dict]]:
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
        return model.forward_recurrent_lighter(
            views,
            str(device),
            ret_state=False,
            use_ttt3r=False,
            return_token_debug=True,
        )


def segment_edges(cuts: list[int], frame_count: int) -> list[tuple[int, int]]:
    edges = [0, *cuts, frame_count]
    segments = list(zip(edges[:-1], edges[1:]))
    if any(start >= stop for start, stop in segments):
        raise ValueError(f"invalid cuts {cuts}")
    return segments


def prepare_route_views(
    model: ARCroco3DStereo,
    paths: list[Path],
    cuts: list[int],
    size: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    segments = segment_edges(cuts, len(paths))
    for index, (start, stop) in enumerate(segments):
        local_paths = paths[start:stop]
        local_event: set[int] = set()
        if index + 1 < len(segments):
            local_paths = [*local_paths, paths[stop]]
            local_event = {stop - start}
        views = gt_helpers.prepare_full_square_input(
            model, local_paths, SimpleNamespace(size=int(size))
        )
        output.append({
            "start": start,
            "stop": stop,
            "views": set_event_indices(views, local_event),
            "has_boundary_probe": bool(local_event),
        })
    expected = len(paths) + len(cuts)
    observed = sum(len(value["views"]) for value in output)
    if observed != expected:
        raise AssertionError(f"expected {expected} neural frames, prepared {observed}")
    return output


def boundary_transaction(
    pre_last: dict[str, Any],
    post: list[dict[str, Any]],
    topology: CommonTopology,
) -> tuple[list[dict[str, Any]], int]:
    persistent, association = persistent_post(pre_last, post, shifts=None)
    arrays = pack_methods({"route": [pre_last, *persistent]}, topology)
    prefix = "route__"
    source = {
        key: np.asarray(arrays[prefix + key])
        for key in (
            "cameras_c2w",
            "vertices_world",
            "joints_world",
            "persistent_ids",
            "native_ids",
            "valid",
        )
    }
    pairs = [tuple(map(int, pair)) for pair in association["pairs"]]
    _, diagnostics = apply_locked_transaction(
        source, boundary=1, pairs=pairs, cut_detected=True
    )
    transform = np.asarray(diagnostics["boundary"]["transform"], dtype=np.float64)
    return map_frames(persistent, transform), len(pairs)


def measure_route(
    model: ARCroco3DStereo,
    layer: SMPL_Layer,
    topology: CommonTopology,
    chunks: list[dict[str, Any]],
    cuts: list[int],
    device: torch.device,
) -> dict[str, Any]:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    baseline = int(torch.cuda.memory_allocated(device))
    synchronize(device)
    total_started = time.perf_counter()
    neural_seconds = 0.0
    decode_seconds = 0.0
    geometry_seconds = 0.0
    output_frames: list[dict[str, Any]] = []
    pending_shadow: dict[str, Any] | None = None
    pair_count = 0

    for index, chunk in enumerate(chunks):
        started = time.perf_counter()
        predictions, returned, debug = forward(model, chunk["views"], device)
        synchronize(device)
        neural_seconds += time.perf_counter() - started
        started = time.perf_counter()
        decoded = decode_sequence(predictions, returned, debug, layer, topology)
        synchronize(device)
        decode_seconds += time.perf_counter() - started
        del predictions, returned, debug

        segment_length = int(chunk["stop"] - chunk["start"])
        segment = decoded[:segment_length]
        next_shadow = decoded[-1] if chunk["has_boundary_probe"] else None
        if index == 0:
            placed = segment
            transform_to_world = np.eye(4, dtype=np.float64)
        else:
            if pending_shadow is None or not segment:
                raise AssertionError("missing boundary probe or clean segment")
            coarse = np.asarray(pending_shadow["camera"]) @ np.linalg.inv(
                np.asarray(segment[0]["camera"])
            )
            coarse_post = map_frames(segment, coarse)
            started = time.perf_counter()
            placed, count = boundary_transaction(output_frames[-1], coarse_post, topology)
            geometry_seconds += time.perf_counter() - started
            pair_count += count
            # The locked transaction is translation-only.  Recover its net
            # local-to-world transform from the first camera pair so the next
            # boundary probe follows the same placed coordinate system.
            transform_to_world = np.asarray(placed[0]["camera"]) @ np.linalg.inv(
                np.asarray(segment[0]["camera"])
            )
        output_frames.extend(placed)
        pending_shadow = (
            map_frames([next_shadow], transform_to_world)[0]
            if next_shadow is not None
            else None
        )
        del decoded, segment, placed

    synchronize(device)
    total_seconds = time.perf_counter() - total_started
    peak = int(torch.cuda.max_memory_allocated(device))
    if len(output_frames) != 300:
        raise AssertionError(f"route emitted {len(output_frames)} frames")
    neural_frames = sum(len(chunk["views"]) for chunk in chunks)
    del output_frames
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "cut_count": len(cuts),
        "neural_frame_evaluations": neural_frames,
        "output_frames": 300,
        "neural_seconds": neural_seconds,
        "decode_seconds": decode_seconds,
        "geometry_seconds": geometry_seconds,
        "total_seconds": total_seconds,
        "amortized_output_fps": 300.0 / total_seconds,
        "peak_allocated_bytes": peak,
        "baseline_allocated_bytes": baseline,
        "incremental_peak_bytes": peak - baseline,
        "boundary_pair_count": pair_count,
    }


def run_repeated(callback: Any, warmup: int, repetitions: int) -> dict[str, Any]:
    warmups = []
    for index in range(warmup):
        result = callback()
        warmups.append(result)
        print(json.dumps({"phase": "warmup", "iteration": index + 1, **result}), flush=True)
    timed = []
    for index in range(repetitions):
        result = callback()
        timed.append(result)
        print(json.dumps({"phase": "timed", "iteration": index + 1, **result}), flush=True)
    numeric = [
        key for key, value in timed[0].items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    aggregate = {}
    for key in numeric:
        values = [float(row[key]) for row in timed]
        aggregate[key] = {
            "median": statistics.median(values),
            "mean": statistics.mean(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return {"warmup": warmups, "timed": timed, "aggregate": aggregate}


def gpu_provenance(device: torch.device) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(device)
    query = subprocess.run(
        [
            "nvidia-smi",
            f"--id={int(device.index)}",
            "--query-gpu=index,name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
    )
    return {
        "name": props.name,
        "total_memory_bytes": int(props.total_memory),
        "compute_capability": f"{props.major}.{props.minor}",
        "nvidia_smi": query.stdout.strip() if query.returncode == 0 else None,
    }


def main() -> None:
    args = parse_args()
    if args.warmup < 1 or args.repetitions < 3:
        raise ValueError("formal protocol requires one warm-up and at least three repetitions")
    if args.size != 512:
        raise ValueError("formal protocol is fixed to 512-pixel input")
    device = torch.device(args.device)
    if device.type != "cuda" or device.index is None:
        raise ValueError("an explicit CUDA device is required")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    manifest = args.manifest.resolve()
    checkpoint = args.current_checkpoint.resolve()
    if not manifest.is_file() or not checkpoint.is_file():
        raise FileNotFoundError(manifest if not manifest.is_file() else checkpoint)
    record = load_record(manifest, args.case_id, args.source_layout)
    if args.source_layout == "egohumans":
        sequence_root = args.extracted_root.resolve() / str(record["capture_relative"])
        pre, post = frame_image_paths(sequence_root, record)
        source_paths = pre + post
    else:
        sequence_root = args.extracted_root.resolve()
        source_paths = [sequence_root / str(value) for value in record["image_members"]]
        missing = [path for path in source_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(missing[0])
    repetitions = 300 // len(source_paths)
    paths = source_paths * repetitions
    if len(paths) != 300:
        raise AssertionError("300-frame source construction failed")

    topology = CommonTopology.load()
    model = ARCroco3DStereo.from_pretrained(str(checkpoint)).to(device).eval()
    flags = configure_model(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layer = SMPL_Layer(
        type="smplx", gender="neutral", num_betas=10, kid=False, person_center="head"
    ).to(device).eval()
    prepared = {
        cut_count: prepare_route_views(model, paths, cuts, args.size)
        for cut_count, cuts in CUT_SCHEDULES.items()
    }
    routes = {}
    for cut_count, cuts in CUT_SCHEDULES.items():
        print(f">> route cuts={cut_count}, schedule={cuts}", flush=True)
        routes[str(cut_count)] = run_repeated(
            lambda cut_values=cuts, chunk_values=prepared[cut_count]: measure_route(
                model, layer, topology, chunk_values, cut_values, device
            ),
            args.warmup,
            args.repetitions,
        )

    no_cut = routes["0"]["aggregate"]["total_seconds"]["median"]
    summary = {}
    for cut_count in CUT_SCHEDULES:
        aggregate = routes[str(cut_count)]["aggregate"]
        seconds = aggregate["total_seconds"]["median"]
        summary[str(cut_count)] = {
            "cut_schedule": CUT_SCHEDULES[cut_count],
            "neural_frame_evaluations": int(
                aggregate["neural_frame_evaluations"]["median"]
            ),
            "median_seconds": seconds,
            "median_amortized_output_fps": aggregate["amortized_output_fps"]["median"],
            "maximum_peak_allocated_bytes": aggregate["peak_allocated_bytes"]["max"],
            "extra_seconds_over_no_cut": seconds - no_cut,
            "overhead_percent_over_no_cut": 100.0 * (seconds - no_cut) / no_cut,
        }
    script = Path(__file__).resolve()
    report = {
        "schema_version": SCHEMA,
        "status": "complete",
        "protocol": {
            "output_frames": 300,
            "source_construction": (
                f"the same frozen {len(source_paths)}-frame RGB case repeated "
                f"{repetitions} times for every route"
            ),
            "source_layout": args.source_layout,
            "input_size": 512,
            "batch_size": 1,
            "precision": "FP32",
            "tf32": False,
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "aggregation": "median time/FPS and maximum torch peak allocated over timed repetitions",
            "checkpoint_loading_timed": False,
            "rgb_decode_resize_timed": False,
            "model_output_decode_timed": True,
            "association_and_locked_geometry_timed": True,
            "transition_detector_timed": False,
            "neural_evaluations": "300 + cut count",
        },
        "case": {
            "case_id": record["case_id"],
            "capture": record["capture"],
            "source_frame_count": len(source_paths),
            "manifest": str(manifest),
            "manifest_sha256": sha256(manifest),
            "staged_sequence_root": str(sequence_root),
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256(checkpoint),
            "runtime_flags": flags,
        },
        "hardware": {"gpu": gpu_provenance(device), "platform": platform.platform()},
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "script": str(script),
            "script_sha256": sha256(script),
        },
        "routes": routes,
        "summary": summary,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, report)
    lines = [
        "# Shot3R cut-count runtime and memory scaling",
        "",
        "| Cuts | Neural frames | Median seconds | Output FPS | Peak allocated (GiB) | Overhead vs. 0 cuts |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for cut_count in CUT_SCHEDULES:
        row = summary[str(cut_count)]
        lines.append(
            f"| {cut_count} | {row['neural_frame_evaluations']} | "
            f"{row['median_seconds']:.3f} | {row['median_amortized_output_fps']:.3f} | "
            f"{row['maximum_peak_allocated_bytes'] / 2**30:.2f} | "
            f"{row['overhead_percent_over_no_cut']:.2f}% |"
        )
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
