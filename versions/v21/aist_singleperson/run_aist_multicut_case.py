#!/usr/bin/env python3
"""Run an RGB-only Bridge3R multi-cut AIST++ case without evaluator access.

Unlike the CS150 runner, this is an explicit multi-event policy: every causal
detector positive is processed left-to-right.  The recurrent coarse branch
receives the complete proposed-event set, while the fixed shared-translation
transaction is applied once per proposed event and only changes that event's
future suffix.  The runner sees a compact RGB manifest row only; labels,
camera identities, calibration, and true cut locations remain evaluator-only.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import platform
import resource
import shutil
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
from versions.v14.causal_image_detector import CausalGRUShotDetector  # noqa: E402
from versions.v14.probe_p1_foot_scene_observability import anonymous_match  # noqa: E402
from versions.v14.run_v14_2_single_sequence import configure_model, set_event_indices  # noqa: E402
from versions.v15.harmony4d import run_harmony_case as frozen  # noqa: E402
from versions.v15.harmony4d.topology import CommonTopology  # noqa: E402
from versions.v16.harmony4d.causal_stabilization import Candidate, apply_candidate, boundary_permutation_ids  # noqa: E402

try:
    from .protocol import PROTOCOLS
    from .run_aist_case import decode_frames, safe_video_path
except ImportError:
    from protocol import PROTOCOLS  # type: ignore
    from run_aist_case import decode_frames, safe_video_path  # type: ignore


SCHEMA = "Bridge3R-AIST-SinglePerson-MC150-runtime-cache-v1"
ALLOWED_PROTOCOLS = {"MC150-3", "MC150-4"}
EXPECTED_FRAMES, EXPECTED_FPS = 150, 30
METHOD_NAMES = (
    "m0_strict_human3r",
    "m1_clean_reset",
    "m3_b0_only",
    "m4_b0_identity",
    "m15_bridge3r_fixed_v19",
)
ARRAY_KEYS = ("cameras_c2w", "vertices_world", "joints_world", "persistent_ids", "native_ids", "valid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--line", type=int, default=1)
    parser.add_argument("--derived-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--current-checkpoint", type=Path)
    parser.add_argument("--original-checkpoint", type=Path)
    parser.add_argument("--keep-decoded-frames", action="store_true")
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
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def process_peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024))


def read_runtime(manifest: Path, line: int) -> dict[str, Any]:
    rows = [json.loads(value) for value in manifest.read_text(encoding="utf-8").splitlines() if value.strip()]
    if line < 1 or line > len(rows):
        raise IndexError(f"line {line} outside runtime manifest")
    row = rows[line - 1]
    required = {"case_id", "dataset", "protocol", "role", "input_video", "fps", "num_frames"}
    if not isinstance(row, dict) or set(row) != required:
        raise ValueError("multi-cut runtime schema drifted")
    if row["dataset"] != "AIST++" or row["protocol"] not in ALLOWED_PROTOCOLS:
        raise ValueError("runner accepts only AIST MC150 runtime rows")
    if int(row["fps"]) != EXPECTED_FPS or int(row["num_frames"]) != EXPECTED_FRAMES:
        raise ValueError("unexpected AIST multi-cut timeline")
    return row


def prepare_views(model: ARCroco3DStereo, paths: list[Path], events: set[int], size: int) -> list[dict[str, Any]]:
    views = frozen.gt_helpers.prepare_full_square_input(model, paths, SimpleNamespace(size=int(size)))
    return set_event_indices(views, {int(event) for event in events})


def run_with_events(
    model: ARCroco3DStereo, layer: SMPL_Layer, topology: CommonTopology, paths: list[Path], events: set[int],
    device: torch.device, size: int, label: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    views = prepare_views(model, paths, events, size)
    predictions, returned, debug, runtime = frozen.run_forward(model, views, device, label)
    frames = frozen.decode_sequence(predictions, returned, debug, layer, topology)
    del views, predictions, returned, debug
    return frames, runtime


def candidate_events(labels: list[int]) -> list[int]:
    events = [index for index, value in enumerate(labels) if index > 0 and int(value) == 1]
    if any(index >= EXPECTED_FRAMES for index in events):
        raise ValueError("detector emitted an out-of-range event")
    return events


def segments(events: list[int], frame_count: int) -> list[tuple[int, int]]:
    edges = [0, *events, frame_count]
    output = list(zip(edges[:-1], edges[1:]))
    if not output or any(start >= stop for start, stop in output):
        raise ValueError(f"invalid causal event segmentation: {events}")
    return output


def unprefix(packed: dict[str, np.ndarray], method: str) -> dict[str, np.ndarray]:
    return {key: np.asarray(packed[f"{method}__{key}"]).copy() for key in ARRAY_KEYS}


def append_arrays(destination: dict[str, np.ndarray], method: str, arrays: dict[str, np.ndarray]) -> None:
    for key in ARRAY_KEYS:
        destination[f"{method}__{key}"] = np.asarray(arrays[key])


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    if args.size != 512:
        raise ValueError("the frozen multi-cut contract uses --size 512")
    record = read_runtime(args.runtime_manifest.resolve(), int(args.line))
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"runner refuses to overwrite: {output}")
    video = safe_video_path(args.derived_root.resolve(), str(record["input_video"]))
    paths, decode_runtime = decode_frames(video, args.work_dir, str(record["case_id"]))
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("formal AIST multi-cut inference requires CUDA")
    torch.cuda.set_device(device)
    current_default, original_default = frozen.default_checkpoints()
    current = (args.current_checkpoint or current_default).resolve()
    original = (args.original_checkpoint or original_default).resolve()
    for required in (current, original, Path(frozen.DETECTOR_PATH)):
        if not required.is_file():
            raise FileNotFoundError(required)
    topology = CommonTopology.load()

    detector = CausalGRUShotDetector(Path(frozen.DETECTOR_PATH))
    detector_started = time.perf_counter()
    labels, detector_rows = detector.predict_sequence(paths)
    events = candidate_events(labels)
    runtime: dict[str, Any] = {
        "decode": decode_runtime,
        "causal_gru_detector": {
            "seconds": time.perf_counter() - detector_started,
            "labels": labels,
            "rows": detector_rows,
            "all_positive_indices": events,
            "deployment_policy": "all_positive_events_left_to_right; no evaluator boundary access",
        },
    }

    original_model = ARCroco3DStereo.from_pretrained(str(original)).to(device)
    frozen.strict_original(original_model)
    original_model.eval()
    original_layer = SMPL_Layer(type="smplx", gender="neutral", num_betas=10, kid=False, person_center="head").to(device).eval()
    strict, strict_runtime = run_with_events(original_model, original_layer, topology, paths, set(), device, args.size, "aist_multicut_strict")
    del original_model, original_layer
    gc.collect(); torch.cuda.empty_cache()

    model = ARCroco3DStereo.from_pretrained(str(current)).to(device)
    flags = configure_model(model)
    model.eval()
    layer = SMPL_Layer(type="smplx", gender="neutral", num_betas=10, kid=False, person_center="head").to(device).eval()
    shadow, shadow_runtime = run_with_events(model, layer, topology, paths, set(events), device, args.size, "aist_multicut_shadow")
    raw_segments, raw_runtimes = [], []
    for segment_index, (start, stop) in enumerate(segments(events, len(paths))):
        segment, segment_runtime = run_with_events(model, layer, topology, paths[start:stop], set(), device, args.size, f"aist_multicut_clean_reset_s{segment_index}")
        raw_segments.append(segment); raw_runtimes.append({"start": start, "stop": stop, "runtime": segment_runtime})
    clean_reset = [frame for segment in raw_segments for frame in segment]
    if len(clean_reset) != len(paths):
        raise AssertionError("clean-reset segments do not cover the RGB timeline")
    coarse: list[dict[str, Any]] = []
    first_stop = events[0] if events else len(paths)
    coarse.extend(copy.deepcopy(shadow[:first_stop]))
    for event, raw_post in zip(events, raw_segments[1:]):
        transform = np.asarray(shadow[event]["camera"]) @ np.linalg.inv(np.asarray(raw_post[0]["camera"]))
        coarse.extend(frozen.map_frames(raw_post, transform))
    if len(coarse) != len(paths):
        raise AssertionError("coarse multi-event composition does not cover the RGB timeline")
    del layer, model
    gc.collect(); torch.cuda.empty_cache()

    association_rows = []
    for event in events:
        association = anonymous_match(coarse[event - 1]["people"], coarse[event]["people"])
        association_rows.append({"boundary": int(event), "pairs": [list(map(int, pair)) for pair in association.get("pairs", [])], "matched_count": int(association.get("matched_count", 0)), "cost": np.asarray(association.get("cost", [])).tolist()})
    packed = frozen.pack_methods({"m0_strict_human3r": strict, "m1_clean_reset": clean_reset, "m3_b0_only": coarse}, topology)
    coarse_arrays = unprefix(packed, "m3_b0_only")
    identity_arrays = {key: value.copy() for key, value in coarse_arrays.items()}
    bridge_arrays = {key: value.copy() for key, value in coarse_arrays.items()}
    identity_debug, bridge_debug = [], []
    fixed = Candidate(name="v19_ungated_translation_b050", camera_alpha=1.0, boundary_kind="translation", boundary_blend=0.5)
    for association in association_rows:
        boundary, pairs = int(association["boundary"]), [tuple(pair) for pair in association["pairs"]]
        identity_arrays, identity_row = boundary_permutation_ids(identity_arrays, boundary, pairs)
        bridge_arrays, bridge_row = apply_candidate(bridge_arrays, boundary, pairs, fixed)
        if bridge_row.get("runtime_contract", {}).get("exact_m15_fallback"):
            raise RuntimeError("fixed Bridge3R multi-event transaction entered a gated fallback")
        identity_debug.append({"boundary": boundary, "debug": identity_row})
        bridge_debug.append({"boundary": boundary, "debug": bridge_row})
    append_arrays(packed, "m4_b0_identity", identity_arrays)
    append_arrays(packed, "m15_bridge3r_fixed_v19", bridge_arrays)
    temporary = output.with_suffix(output.suffix + ".partial")
    output.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **packed)
    os.replace(temporary, output)
    runtime["multicut_transaction"] = {
        "policy": "coarse_event_token_and_fixed_transaction_repeated_left_to_right",
        "events": events,
        "segments": [{"start": start, "stop": stop} for start, stop in segments(events, len(paths))],
        "shadow_forward": shadow_runtime,
        "clean_reset_forwards": raw_runtimes,
        "association": association_rows,
        "identity": identity_debug,
        "bridge3r_fixed_v19": bridge_debug,
    }
    report = {
        "schema_version": SCHEMA, "record": record, "methods": list(METHOD_NAMES), "runtime": runtime,
        "checkpoint": {
            "current": str(current), "current_sha256": frozen.verified_artifact_sha256(current),
            "original": str(original), "original_sha256": frozen.verified_artifact_sha256(original),
            "detector": str(Path(frozen.DETECTOR_PATH)), "detector_sha256": frozen.verified_artifact_sha256(Path(frozen.DETECTOR_PATH)),
            "current_flags": flags,
        },
        "provenance": {"runner": str(Path(__file__).resolve()), "runner_sha256": sha256(Path(__file__).resolve())},
        "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda, "device": str(device), "gpu": torch.cuda.get_device_name(device), "process_peak_rss_bytes": process_peak_rss_bytes()},
        "total_process_seconds": time.perf_counter() - started,
        "runtime_contract": {
            "gt_in_runtime": False, "camera_or_cut_in_runtime": False, "future_frames_at_boundary": 0,
            "transaction_boundary_source": "all CausalGRUShotDetector positives in increasing frame order",
            "pre_frames_rewritten_after_event": False, "protocol_scope": str(record["protocol"]),
        },
        "cache": str(output), "cache_sha256": sha256(output),
    }
    report_path = output.with_suffix(".runtime.json")
    report_path.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not args.keep_decoded_frames:
        shutil.rmtree(args.work_dir.resolve() / str(record["case_id"]))
    print(json.dumps({"case_id": record["case_id"], "events": events, "cache": str(output), "report": str(report_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
