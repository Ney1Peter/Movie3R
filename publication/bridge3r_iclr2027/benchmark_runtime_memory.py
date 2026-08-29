#!/usr/bin/env python3
"""Standardized single-GPU runtime and peak-memory benchmark for Bridge3R.

The benchmark preloads one frozen 100-frame EgoHumans case and measures only
model inference plus reconstruction/transaction materialization.  Checkpoint
loading and RGB decoding/resizing are outside timed regions.  The protocol is
fixed to batch size one, 512-pixel model input, explicit FP32 (including TF32
disabled), one warm-up iteration, and at least three timed repetitions.

Measured routes:

* strict Human3R, one 100-frame no-event rollout;
* Bridge3R no-cut, one 100-frame no-event rollout;
* Bridge3R one-cut transaction, comprising the 51-frame read-only shadow
  prefix, 50-frame clean-reset post-cut rollout, decoding, prediction-only
  boundary association, and the locked publication geometry transaction.

The causal cut detector is intentionally excluded: this benchmark isolates
the reconstruction transaction requested by the publication protocol.
"""

from __future__ import annotations

import argparse
import copy
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
from typing import Any, Callable

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
for root in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from dust3r.model import ARCroco3DStereo  # noqa: E402
from dust3r.utils.smpl_layer import SMPL_Layer  # noqa: E402
from publication.bridge3r_iclr2027.runtime_contract import (  # noqa: E402
    apply_locked_transaction,
)
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


SCHEMA = "Bridge3R-standardized-runtime-memory-v1"
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
DEFAULT_ORIGINAL = REPO_ROOT / "src/human3r_896L.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--case-id", default=DEFAULT_CASE)
    parser.add_argument("--extracted-root", type=Path, required=True)
    parser.add_argument("--current-checkpoint", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--original-checkpoint", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:4")
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
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def load_record(manifest: Path, case_id: str) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [row for row in rows if str(row.get("case_id")) == case_id]
    if len(selected) != 1:
        raise ValueError(f"Expected one record for {case_id}, found {len(selected)}")
    record = selected[0]
    if int(record["clip_length"]) != 100:
        raise ValueError(f"Runtime protocol requires 100 frames, got {record['clip_length']}")
    boundary = int(record["boundary_index"])
    if boundary != len(record["pre_frame_numbers"]):
        raise ValueError("boundary index does not equal pre-cut frame count")
    if len(record["post_frame_numbers"]) != 100 - boundary:
        raise ValueError("post-cut frame count does not complete the 100-frame clip")
    return record


def strict_original(model: ARCroco3DStereo) -> None:
    for name in (
        "enable_shot_adaptation",
        "enable_shot_decoder_token",
        "enable_anchor_pose_adapter",
        "enable_anchor_decoder_tokens",
        "enable_anchor_pose_token_adapter",
        "enable_v7_pose_adapter",
        "enable_v8_pose_prompt",
        "enable_v8_human_trans_corr",
        "enable_v8_human_latent_corr",
        "enable_v8_head_lora",
        "enable_layerwise_pose_shot_adapter",
        "enable_pose_alignment_adapter",
        "enable_pose_translation_adapter",
        "enable_pose_lora",
        "enable_human_lora",
        "enable_world_lora",
    ):
        if hasattr(model, name):
            setattr(model, name, False)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def forward(
    model: ARCroco3DStereo,
    views: list[dict[str, Any]],
    device: torch.device,
) -> tuple[list[dict], list[dict], list[dict]]:
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        return model.forward_recurrent_lighter(
            views,
            str(device),
            ret_state=False,
            use_ttt3r=False,
            return_token_debug=True,
        )


def measure_direct(
    model: ARCroco3DStereo,
    layer: SMPL_Layer,
    topology: CommonTopology,
    views: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    baseline = int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else 0
    synchronize(device)
    total_started = time.perf_counter()
    forward_started = total_started
    predictions, returned, debug = forward(model, views, device)
    synchronize(device)
    forward_seconds = time.perf_counter() - forward_started
    decode_started = time.perf_counter()
    decoded = decode_sequence(predictions, returned, debug, layer, topology)
    synchronize(device)
    decode_seconds = time.perf_counter() - decode_started
    total_seconds = time.perf_counter() - total_started
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    person_frames = int(sum(len(frame["people"]) for frame in decoded))
    del predictions, returned, debug, decoded
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "forward_seconds": forward_seconds,
        "decode_seconds": decode_seconds,
        "total_seconds": total_seconds,
        "fps_forward": len(views) / forward_seconds,
        "fps_end_to_end": len(views) / total_seconds,
        "peak_allocated_bytes": peak,
        "baseline_allocated_bytes": baseline,
        "incremental_peak_bytes": peak - baseline,
        "decoded_person_frames": person_frames,
    }


def minimal_locked_geometry(
    shadow: list[dict[str, Any]],
    raw_post: list[dict[str, Any]],
    boundary: int,
    topology: CommonTopology,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Materialize only the one publication route, not exploratory branches."""

    pre = shadow[:-1]
    shadow_first_post = shadow[-1]
    b0_transform = np.asarray(shadow_first_post["camera"]) @ np.linalg.inv(
        np.asarray(raw_post[0]["camera"])
    )
    b0_post = map_frames(raw_post, b0_transform)
    m3 = copy.deepcopy(pre + b0_post)
    _, association = persistent_post(pre[-1], b0_post, shifts=None)
    packed = pack_methods({"m3_b0_only": m3}, topology)
    prefix = "m3_b0_only__"
    source = {
        key: np.asarray(packed[prefix + key])
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
    output, diagnostics = apply_locked_transaction(
        source,
        boundary=boundary,
        pairs=pairs,
        cut_detected=True,
    )
    return output, {
        "b0_transform": b0_transform,
        "pairs": pairs,
        "association": association,
        "locked_transaction": diagnostics,
    }


def measure_transaction(
    model: ARCroco3DStereo,
    layer: SMPL_Layer,
    topology: CommonTopology,
    shadow_views: list[dict[str, Any]],
    raw_post_views: list[dict[str, Any]],
    boundary: int,
    device: torch.device,
) -> dict[str, Any]:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    baseline = int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else 0
    synchronize(device)
    total_started = time.perf_counter()

    started = time.perf_counter()
    shadow_predictions, shadow_returned, shadow_debug = forward(model, shadow_views, device)
    synchronize(device)
    shadow_forward_seconds = time.perf_counter() - started
    started = time.perf_counter()
    shadow = decode_sequence(
        shadow_predictions, shadow_returned, shadow_debug, layer, topology
    )
    synchronize(device)
    shadow_decode_seconds = time.perf_counter() - started
    del shadow_predictions, shadow_returned, shadow_debug

    started = time.perf_counter()
    raw_predictions, raw_returned, raw_debug = forward(model, raw_post_views, device)
    synchronize(device)
    raw_forward_seconds = time.perf_counter() - started
    started = time.perf_counter()
    raw_post = decode_sequence(raw_predictions, raw_returned, raw_debug, layer, topology)
    synchronize(device)
    raw_decode_seconds = time.perf_counter() - started
    del raw_predictions, raw_returned, raw_debug

    started = time.perf_counter()
    output, geometry = minimal_locked_geometry(shadow, raw_post, boundary, topology)
    geometry_seconds = time.perf_counter() - started
    synchronize(device)
    total_seconds = time.perf_counter() - total_started
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    valid = np.asarray(output["valid"]).astype(bool)
    pairs = geometry["pairs"]
    del shadow, raw_post, output, geometry
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    neural_seconds = shadow_forward_seconds + raw_forward_seconds
    decode_seconds = shadow_decode_seconds + raw_decode_seconds
    return {
        "shadow_forward_seconds": shadow_forward_seconds,
        "shadow_decode_seconds": shadow_decode_seconds,
        "clean_post_forward_seconds": raw_forward_seconds,
        "clean_post_decode_seconds": raw_decode_seconds,
        "neural_forward_seconds": neural_seconds,
        "decode_seconds": decode_seconds,
        "geometry_seconds": geometry_seconds,
        "total_seconds": total_seconds,
        "neural_frame_evaluations": len(shadow_views) + len(raw_post_views),
        "output_frames": int(valid.shape[0]),
        "fps_neural_frame_evaluations": (
            len(shadow_views) + len(raw_post_views)
        )
        / neural_seconds,
        "amortized_output_fps": int(valid.shape[0]) / total_seconds,
        "peak_allocated_bytes": peak,
        "baseline_allocated_bytes": baseline,
        "incremental_peak_bytes": peak - baseline,
        "boundary_pair_count": len(pairs),
        "valid_person_frames": int(valid.sum()),
    }


def run_protocol(
    name: str,
    callback: Callable[[], dict[str, Any]],
    warmup: int,
    repetitions: int,
) -> dict[str, Any]:
    for index in range(warmup):
        result = callback()
        print(
            json.dumps(
                {"route": name, "phase": "warmup", "iteration": index + 1, **result}
            ),
            flush=True,
        )
    timed = []
    for index in range(repetitions):
        result = callback()
        timed.append(result)
        print(
            json.dumps(
                {"route": name, "phase": "timed", "iteration": index + 1, **result}
            ),
            flush=True,
        )
    numeric_keys = [
        key
        for key, value in timed[0].items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    aggregate = {}
    for key in numeric_keys:
        values = [float(row[key]) for row in timed]
        aggregate[key] = {
            "median": statistics.median(values),
            "mean": statistics.mean(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return {"warmup_count": warmup, "timed_repetitions": timed, "aggregate": aggregate}


def git_provenance() -> dict[str, Any]:
    def query(*arguments: str) -> str | None:
        completed = subprocess.run(
            ["git", *arguments], cwd=REPO_ROOT, text=True, capture_output=True
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    status = query("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": query("rev-parse", "HEAD"),
        "tracked_worktree_dirty": None if status is None else bool(status),
    }


def gpu_provenance(device: torch.device) -> dict[str, Any]:
    index = int(device.index or 0)
    props = torch.cuda.get_device_properties(device)
    query = subprocess.run(
        [
            "nvidia-smi",
            f"--id={index}",
            "--query-gpu=index,name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
    )
    return {
        "torch_index": index,
        "torch_name": props.name,
        "total_memory_bytes": int(props.total_memory),
        "compute_capability": f"{props.major}.{props.minor}",
        "nvidia_smi": query.stdout.strip() if query.returncode == 0 else None,
    }


def format_gib(value: float) -> str:
    return f"{value / 2**30:.2f}"


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    routes = report["routes"]
    strict = routes["strict_human3r"]["aggregate"]
    no_cut = routes["bridge3r_no_cut"]["aggregate"]
    cut = routes["bridge3r_single_cut_transaction"]["aggregate"]
    summary = report["summary"]
    rows = [
        (
            "Strict Human3R",
            strict["total_seconds"]["median"],
            strict["fps_end_to_end"]["median"],
            strict["peak_allocated_bytes"]["max"],
        ),
        (
            "Bridge3R no-cut",
            no_cut["total_seconds"]["median"],
            no_cut["fps_end_to_end"]["median"],
            no_cut["peak_allocated_bytes"]["max"],
        ),
        (
            "Bridge3R single-cut transaction",
            cut["total_seconds"]["median"],
            cut["amortized_output_fps"]["median"],
            cut["peak_allocated_bytes"]["max"],
        ),
    ]
    lines = [
        "# Standardized runtime and peak-memory benchmark",
        "",
        f"- Case: `{report['case']['case_id']}` (100 frames, boundary 50)",
        f"- GPU: {report['hardware']['gpu']['torch_name']} (`{report['hardware']['gpu']['nvidia_smi']}`)",
        "- Input/precision: 512, batch size 1, FP32 with TF32 disabled",
        f"- Timing: {report['protocol']['warmup']} warm-up + {report['protocol']['repetitions']} measured repetitions",
        "- RGB decode/resize and checkpoint loading are excluded; model output decoding and locked geometry are included.",
        "",
        "| Route | Median seconds / 100 output frames | FPS | Max torch peak allocated (GiB) |",
        "|---|---:|---:|---:|",
    ]
    for name, seconds, fps, peak in rows:
        lines.append(f"| {name} | {seconds:.3f} | {fps:.3f} | {format_gib(peak)} |")
    lines.extend(
        [
            "",
            f"The single-cut transaction adds **{summary['cut_extra_seconds']:.3f} s** "
            f"({summary['cut_overhead_percent']:.2f}%) over the Bridge3R no-cut path, "
            f"yielding {summary['single_cut_amortized_fps']:.3f} amortized output FPS.",
            "",
            "The transaction consists of a 51-frame shadow prefix, a 50-frame clean-reset "
            "post-cut rollout, output decoding, prediction-only association, and the locked "
            "half-translation publication geometry. Detector latency is not included.",
            "",
            "All individual repetitions and complete hashes are recorded in the adjacent JSON.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.warmup < 1 or args.repetitions < 3:
        raise ValueError("Formal protocol requires >=1 warm-up and >=3 timed repetitions")
    if int(args.size) != 512:
        raise ValueError("Formal protocol is fixed at input size 512")
    device = torch.device(args.device)
    if device.type != "cuda" or device.index is None:
        raise ValueError("Formal protocol requires an explicit single CUDA device")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    manifest = args.manifest.resolve()
    checkpoint = args.current_checkpoint.resolve()
    original = args.original_checkpoint.resolve()
    extracted_root = args.extracted_root.resolve()
    output = args.output.resolve()
    for path in (manifest, checkpoint, original):
        if not path.is_file():
            raise FileNotFoundError(path)
    record = load_record(manifest, str(args.case_id))
    sequence_root = extracted_root / str(record["capture_relative"])
    pre_paths, post_paths = frame_image_paths(sequence_root, record)
    all_paths = pre_paths + post_paths
    boundary = int(record["boundary_index"])
    if len(all_paths) != 100:
        raise ValueError(f"Expected 100 input frames, got {len(all_paths)}")
    topology = CommonTopology.load()

    # Strict Human3R route.  Model/layer construction and preprocessing happen
    # before warm-up and are intentionally excluded from all timed regions.
    strict_model = ARCroco3DStereo.from_pretrained(str(original)).to(device).eval()
    strict_original(strict_model)
    for parameter in strict_model.parameters():
        parameter.requires_grad_(False)
    strict_layer = SMPL_Layer(
        type="smplx", gender="neutral", num_betas=10, kid=False, person_center="head"
    ).to(device).eval()
    strict_views = set_event_indices(
        gt_helpers.prepare_full_square_input(
            strict_model, all_paths, SimpleNamespace(size=int(args.size))
        ),
        set(),
    )
    strict_results = run_protocol(
        "strict_human3r",
        lambda: measure_direct(
            strict_model, strict_layer, topology, strict_views, device
        ),
        int(args.warmup),
        int(args.repetitions),
    )
    del strict_views, strict_layer, strict_model
    gc.collect()
    torch.cuda.empty_cache()

    # Current checkpoint, with the exact frozen inference configuration.
    current_model = ARCroco3DStereo.from_pretrained(str(checkpoint)).to(device).eval()
    model_flags = configure_model(current_model)
    for parameter in current_model.parameters():
        parameter.requires_grad_(False)
    current_layer = SMPL_Layer(
        type="smplx", gender="neutral", num_betas=10, kid=False, person_center="head"
    ).to(device).eval()
    no_cut_views = set_event_indices(
        gt_helpers.prepare_full_square_input(
            current_model, all_paths, SimpleNamespace(size=int(args.size))
        ),
        set(),
    )
    shadow_views = set_event_indices(
        gt_helpers.prepare_full_square_input(
            current_model, pre_paths + post_paths[:1], SimpleNamespace(size=int(args.size))
        ),
        {boundary},
    )
    raw_post_views = set_event_indices(
        gt_helpers.prepare_full_square_input(
            current_model, post_paths, SimpleNamespace(size=int(args.size))
        ),
        set(),
    )
    no_cut_results = run_protocol(
        "bridge3r_no_cut",
        lambda: measure_direct(
            current_model, current_layer, topology, no_cut_views, device
        ),
        int(args.warmup),
        int(args.repetitions),
    )
    transaction_results = run_protocol(
        "bridge3r_single_cut_transaction",
        lambda: measure_transaction(
            current_model,
            current_layer,
            topology,
            shadow_views,
            raw_post_views,
            boundary,
            device,
        ),
        int(args.warmup),
        int(args.repetitions),
    )

    no_cut_seconds = no_cut_results["aggregate"]["total_seconds"]["median"]
    cut_seconds = transaction_results["aggregate"]["total_seconds"]["median"]
    script_path = Path(__file__).resolve()
    report = {
        "schema_version": SCHEMA,
        "protocol": {
            "input_size": int(args.size),
            "batch_size": 1,
            "precision": "FP32",
            "autocast_enabled": False,
            "tf32_enabled": False,
            "clip_frames": 100,
            "warmup": int(args.warmup),
            "repetitions": int(args.repetitions),
            "aggregation": "median runtime/FPS; maximum peak allocated across timed repetitions",
            "checkpoint_loading_timed": False,
            "rgb_decode_resize_timed": False,
            "model_output_decode_timed": True,
            "locked_geometry_timed": True,
            "detector_timed": False,
            "single_process_on_gpu": True,
        },
        "case": {
            "case_id": record["case_id"],
            "capture": record["capture"],
            "capture_relative": record["capture_relative"],
            "angle_stratum": record.get("angle_stratum"),
            "camera_rotation_span_deg_evaluator_only": record.get(
                "camera_rotation_span_deg_evaluator_only"
            ),
            "pre_camera": record["pre_camera"],
            "post_camera": record["post_camera"],
            "boundary_index": boundary,
            "pre_frames": len(pre_paths),
            "post_frames": len(post_paths),
            "manifest": str(manifest),
            "manifest_sha256": sha256(manifest),
            "staged_sequence_root": str(sequence_root),
        },
        "checkpoints": {
            "strict_human3r": {
                "path": str(original),
                "bytes": original.stat().st_size,
                "sha256": sha256(original),
            },
            "bridge3r": {
                "path": str(checkpoint),
                "bytes": checkpoint.stat().st_size,
                "sha256": sha256(checkpoint),
                "runtime_flags": model_flags,
            },
        },
        "hardware": {
            "gpu": gpu_provenance(device),
            "cpu": platform.processor(),
            "platform": platform.platform(),
        },
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "git": git_provenance(),
            "script": str(script_path),
            "script_sha256": sha256(script_path),
        },
        "routes": {
            "strict_human3r": strict_results,
            "bridge3r_no_cut": no_cut_results,
            "bridge3r_single_cut_transaction": transaction_results,
        },
        "summary": {
            "strict_human3r_seconds": strict_results["aggregate"]["total_seconds"][
                "median"
            ],
            "strict_human3r_fps": strict_results["aggregate"]["fps_end_to_end"][
                "median"
            ],
            "bridge3r_no_cut_seconds": no_cut_seconds,
            "bridge3r_no_cut_fps": no_cut_results["aggregate"]["fps_end_to_end"][
                "median"
            ],
            "bridge3r_single_cut_seconds": cut_seconds,
            "single_cut_amortized_fps": transaction_results["aggregate"][
                "amortized_output_fps"
            ]["median"],
            "cut_extra_seconds": cut_seconds - no_cut_seconds,
            "cut_overhead_percent": 100.0 * (cut_seconds - no_cut_seconds) / no_cut_seconds,
            "strict_peak_allocated_bytes": strict_results["aggregate"][
                "peak_allocated_bytes"
            ]["max"],
            "bridge3r_no_cut_peak_allocated_bytes": no_cut_results["aggregate"][
                "peak_allocated_bytes"
            ]["max"],
            "bridge3r_single_cut_peak_allocated_bytes": transaction_results[
                "aggregate"
            ]["peak_allocated_bytes"]["max"],
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text(
        json.dumps(jsonable(report), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, output)
    markdown = output.with_suffix(".md")
    write_markdown(markdown, report)
    print(
        json.dumps(
            {
                "output": str(output),
                "markdown": str(markdown),
                "summary": report["summary"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
