#!/usr/bin/env python3
"""Run one AIST++ CS150 RGB-only Bridge3R case without evaluator access.

The runtime input is deliberately restricted to one row of an AIST derived
``runtime`` manifest: a case identifier, a derived RGB MP4, and its temporal
format.  In particular, this process never opens the evaluator manifest,
labels, calibration, camera tuple, or ground-truth cut index.  The causal RGB
detector supplies the only proposed boundary used by all reset/transaction
branches.

This runner is intentionally CS150-only.  Multi-cut materialisation needs a
separate, explicitly audited multi-event transaction policy and must not be
silently approximated by reusing the two-shot implementation here.
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
import subprocess
import sys
import time
from pathlib import Path
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
from versions.v14.run_v14_2_single_sequence import configure_model  # noqa: E402
from versions.v15.harmony4d import run_harmony_case as frozen  # noqa: E402
from versions.v15.harmony4d.topology import CommonTopology  # noqa: E402
from versions.v16.harmony4d.causal_stabilization import Candidate, apply_candidate  # noqa: E402


SCHEMA = "Bridge3R-AIST-SinglePerson-CS150-runtime-cache-v1"
EXPECTED_PROTOCOL = "CS150"
EXPECTED_FRAMES = 150
EXPECTED_FPS = 30
METHOD_NAMES = (
    "m0_strict_human3r",
    "m1_clean_reset",
    "m3_b0_only",
    "m4_b0_identity",
    "m6_b0_identity_brtc_c1",
    "m14_safe_boundary_permutation_causal_gru",
    "m15_bridge3r_fixed_v19",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--record-json", help="One runtime-manifest JSON object.")
    source.add_argument("--runtime-manifest", type=Path)
    parser.add_argument("--line", type=int, default=1)
    parser.add_argument("--derived-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--current-checkpoint", type=Path)
    parser.add_argument("--original-checkpoint", type=Path)
    parser.add_argument("--keep-decoded-frames", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
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
    return value


def read_runtime_record(args: argparse.Namespace) -> dict[str, Any]:
    if args.record_json is not None:
        record = json.loads(args.record_json)
    else:
        lines = [line for line in args.runtime_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        if args.line < 1 or args.line > len(lines):
            raise IndexError(f"--line {args.line} outside runtime manifest with {len(lines)} rows")
        record = json.loads(lines[args.line - 1])
    if not isinstance(record, dict):
        raise TypeError("Runtime row must be a JSON object")
    required = {"case_id", "dataset", "protocol", "role", "input_video", "fps", "num_frames"}
    missing = required.difference(record)
    if missing:
        raise ValueError(f"Runtime row misses {sorted(missing)}")
    if record["dataset"] != "AIST++" or record["protocol"] != EXPECTED_PROTOCOL:
        raise ValueError("This runner accepts only AIST++ CS150 runtime rows")
    if int(record["fps"]) != EXPECTED_FPS or int(record["num_frames"]) != EXPECTED_FRAMES:
        raise ValueError(f"Unexpected RGB timeline in {record['case_id']}")
    leaked = [key for key in record if key.endswith("_evaluator_only") or key in {"cut_index", "camera_id", "label"}]
    if leaked:
        raise ValueError(f"Evaluator-only fields leaked into runtime input: {leaked}")
    return record


def safe_video_path(derived_root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"Unsafe runtime video path: {value!r}")
    root = derived_root.resolve()
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file():
        raise FileNotFoundError(path)
    return path


def decode_frames(video: Path, work_dir: Path, case_id: str) -> tuple[list[Path], dict[str, Any]]:
    """Decode exactly the already-derived 30-FPS RGB timeline to JPEG frames."""

    case_dir = work_dir.resolve() / case_id
    if case_dir.exists():
        raise FileExistsError(f"Refusing to reuse or overwrite case work directory: {case_dir}")
    partial = case_dir.with_name(case_dir.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"Stale partial frame directory must be inspected first: {partial}")
    partial.mkdir(parents=True)
    started = time.perf_counter()
    try:
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(video),
            "-map", "0:v:0", "-vsync", "0", "-q:v", "2", str(partial / "frame_%06d.jpg"),
        ]
        subprocess.run(command, check=True)
        frames = sorted(partial.glob("frame_*.jpg"))
        if len(frames) != EXPECTED_FRAMES:
            raise ValueError(f"Decoded {len(frames)} frames from {video}; expected {EXPECTED_FRAMES}")
        if [path.name for path in frames] != [f"frame_{index:06d}.jpg" for index in range(1, EXPECTED_FRAMES + 1)]:
            raise ValueError("Decoded frame numbering is not contiguous")
        os.replace(partial, case_dir)
    except Exception:
        # The partial directory is retained deliberately for post-mortem; it
        # is never silently removed or reused by a later run.
        raise
    output = sorted(case_dir.glob("frame_*.jpg"))
    return output, {"seconds": time.perf_counter() - started, "frame_count": len(output), "directory": str(case_dir)}


def process_peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024))


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    if args.size != 512:
        raise ValueError("The frozen AIST++ pilot contract currently uses --size 512")
    record = read_runtime_record(args)
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    video = safe_video_path(args.derived_root, str(record["input_video"]))
    paths, decode_runtime = decode_frames(video, args.work_dir, str(record["case_id"]))
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Formal AIST++ Bridge3R inference requires an explicit CUDA device")
    torch.cuda.set_device(device)

    default_current, default_original = frozen.default_checkpoints()
    current = (args.current_checkpoint or default_current).resolve()
    original = (args.original_checkpoint or default_original).resolve()
    for required in (current, original, Path(frozen.DETECTOR_PATH)):
        if not required.is_file():
            raise FileNotFoundError(required)
    topology = CommonTopology.load()
    runtime: dict[str, Any] = {"decode": decode_runtime}

    detector_started = time.perf_counter()
    detector = CausalGRUShotDetector(Path(frozen.DETECTOR_PATH))
    detector_labels, detector_rows = detector.predict_sequence(paths)
    proposal = frozen.first_positive(detector_labels)
    runtime["causal_gru_detector"] = {
        "seconds": time.perf_counter() - detector_started,
        "labels": detector_labels,
        "rows": detector_rows,
        "first_positive_index": proposal,
        "deployment_policy": "first_positive_only; no evaluator boundary access",
    }

    original_model = ARCroco3DStereo.from_pretrained(str(original)).to(device)
    frozen.strict_original(original_model)
    original_model.eval()
    original_layer = SMPL_Layer(type="smplx", gender="neutral", num_betas=10, kid=False, person_center="head").to(device).eval()
    strict, runtime["strict_forward"] = frozen.run_no_event(
        original_model, original_layer, topology, paths, device, args.size, "aist_strict_human3r"
    )
    del original_layer, original_model
    gc.collect(); torch.cuda.empty_cache()

    current_model = ARCroco3DStereo.from_pretrained(str(current)).to(device)
    flags = configure_model(current_model)
    current_model.eval()
    current_layer = SMPL_Layer(type="smplx", gender="neutral", num_betas=10, kid=False, person_center="head").to(device).eval()
    if proposal is None:
        parent, parent_runtime = frozen.run_no_event(
            current_model, current_layer, topology, paths, device, args.size, "aist_detector_miss_parent"
        )
        # A detector miss must be visible in completion/coverage.  Every
        # event-dependent ablation is the exact unmodified parent, rather
        # than an oracle-reset substitute.
        methods = {
            "m0_strict_human3r": strict,
            "m1_clean_reset": copy.deepcopy(parent),
            "m3_b0_only": copy.deepcopy(parent),
            "m4_b0_identity": copy.deepcopy(parent),
            "m6_b0_identity_brtc_c1": copy.deepcopy(parent),
            "m14_safe_boundary_permutation_causal_gru": parent,
        }
        runtime["causal_transaction"] = {"event": None, "reason": "detector_emitted_no_positive", "runtime": {"parent": parent_runtime}}
    else:
        candidate, geometry, candidate_runtime = frozen.run_transaction(
            current_model, current_layer, topology, paths, int(proposal), device, args.size, "aist_causal_first_positive"
        )
        methods = {
            "m0_strict_human3r": strict,
            "m1_clean_reset": candidate["m1_clean_reset"],
            "m3_b0_only": candidate["m3_b0_only"],
            "m4_b0_identity": candidate["m4_b0_identity"],
            "m6_b0_identity_brtc_c1": candidate["m6_b0_identity_brtc_c1"],
            "m14_safe_boundary_permutation_causal_gru": candidate["m14_safe_boundary_permutation_oracle"],
        }
        runtime["causal_transaction"] = {"event": int(proposal), "geometry": geometry, "runtime": candidate_runtime}
    del current_layer, current_model
    gc.collect(); torch.cuda.empty_cache()

    arrays = frozen.pack_methods(methods, topology)
    # This is the frozen published operating point used by the previous
    # Camera--Human studies: prediction-only, ungated, translation-only
    # correction at blend 0.5.  It is not selected from AIST pilot/test
    # outcomes.  Its source is the detector-proposed B0 branch, not an
    # evaluator boundary or camera label.
    bridge_name = "m15_bridge3r_fixed_v19"
    if proposal is None:
        for key in ("cameras_c2w", "vertices_world", "joints_world", "persistent_ids", "native_ids", "valid"):
            arrays[f"{bridge_name}__{key}"] = arrays[f"m3_b0_only__{key}"].copy()
        runtime["bridge3r_fixed_v19"] = {
            "source": "exact_current_parent_after_detector_miss",
            "candidate": {"name": "v19_ungated_translation_b050", "camera_alpha": 1.0, "boundary_kind": "translation", "boundary_blend": 0.5},
            "runtime_contract": {"gt_used": False, "exact_parent_fallback": True},
        }
    else:
        source_arrays = {
            key: np.asarray(arrays[f"m3_b0_only__{key}"]).copy()
            for key in ("cameras_c2w", "vertices_world", "joints_world", "persistent_ids", "native_ids", "valid")
        }
        pairs = [tuple(map(int, pair)) for pair in runtime["causal_transaction"]["geometry"].get("association", {}).get("pairs", [])]
        bridge_arrays, bridge_debug = apply_candidate(
            source_arrays,
            int(proposal),
            pairs,
            Candidate(
                name="v19_ungated_translation_b050",
                camera_alpha=1.0,
                boundary_kind="translation",
                boundary_blend=0.5,
            ),
        )
        if bridge_debug.get("runtime_contract", {}).get("exact_m15_fallback"):
            raise RuntimeError("Frozen ungated v19 candidate unexpectedly entered a gated fallback")
        for key, value in bridge_arrays.items():
            arrays[f"{bridge_name}__{key}"] = np.asarray(value)
        runtime["bridge3r_fixed_v19"] = {
            "source": "m3_b0_only_at_detector_first_positive",
            "candidate": {"name": "v19_ungated_translation_b050", "camera_alpha": 1.0, "boundary_kind": "translation", "boundary_blend": 0.5},
            "association_pairs": pairs,
            "diagnostics": bridge_debug,
        }
    temporary = output.with_suffix(output.suffix + ".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, output)
    report = {
        "schema_version": SCHEMA,
        "record": record,
        "methods": list(METHOD_NAMES),
        "runtime": runtime,
        "checkpoint": {
            "current": str(current), "current_sha256": frozen.verified_artifact_sha256(current),
            "original": str(original), "original_sha256": frozen.verified_artifact_sha256(original),
            "detector": str(Path(frozen.DETECTOR_PATH)), "detector_sha256": frozen.verified_artifact_sha256(Path(frozen.DETECTOR_PATH)),
            "current_flags": flags,
        },
        "provenance": {"runner": str(Path(__file__).resolve()), "runner_sha256": sha256(Path(__file__).resolve())},
        "environment": {
            "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "device": str(device), "gpu": torch.cuda.get_device_name(device),
            "process_peak_rss_bytes": process_peak_rss_bytes(),
        },
        "total_process_seconds": time.perf_counter() - started,
        "runtime_contract": {
            "gt_in_runtime": False,
            "camera_or_cut_in_runtime": False,
            "future_frames_at_boundary": 0,
            "transaction_boundary_source": "CausalGRUShotDetector.first_positive",
            "detector_miss_policy": "exact_current_parent; no oracle boundary substitution",
            "protocol_scope": "CS150 single transition only",
        },
        "cache": str(output), "cache_sha256": sha256(output),
    }
    report_path = output.with_suffix(".runtime.json")
    report_path.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not args.keep_decoded_frames:
        shutil.rmtree(args.work_dir.resolve() / str(record["case_id"]))
    print(json.dumps({"case_id": record["case_id"], "cache": str(output), "report": str(report_path), "detector_first_positive": proposal}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
