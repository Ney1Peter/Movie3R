#!/usr/bin/env python3
"""Run an evaluator-timed Bridge3R control on one MVHuman MVH150 case.

This diagnostic deliberately receives a boundary index from the command line
to isolate alignment quality from shot-detector quality.  It is not a causal
deployment result and must never be mixed into the automatic baseline table.
Camera IDs, calibration, masks, angle strata, and 3-D labels remain evaluator
only and are never opened here.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import platform
import resource
import shutil
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
from versions.v21.aist_singleperson import run_aist_case as shared  # noqa: E402


SCHEMA = "Bridge3R-MVHuman-Heldout-MVH150-runtime-cache-v1"
EXPECTED_KEYS = {"case_id", "dataset", "protocol", "role", "input_video", "fps", "num_frames"}
EXPECTED_FRAMES = 150
METHOD_NAMES = (
    "m0_strict_human3r",
    "m1_clean_reset",
    "m3_coarse_gauge",
    "m4_coarse_gauge_identity",
    "m6_fine_alignment",
    "m14_gated_parent",
    "m15_bridge3r_full",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--line", type=int, required=True)
    parser.add_argument("--derived-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--current-checkpoint", type=Path)
    parser.add_argument("--original-checkpoint", type=Path)
    parser.add_argument("--control-event-index", type=int, required=True)
    parser.add_argument("--keep-decoded-frames", action="store_true")
    return parser.parse_args()


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


def read_runtime(path: Path, line: int) -> dict[str, Any]:
    rows = [json.loads(value) for value in path.read_text(encoding="utf-8").splitlines() if value.strip()]
    if line < 1 or line > len(rows):
        raise IndexError(f"--line {line} outside runtime manifest with {len(rows)} rows")
    row = rows[line - 1]
    if not isinstance(row, dict) or set(row) != EXPECTED_KEYS:
        raise ValueError("MVHuman runtime manifest schema drifted")
    if row["dataset"] != "MVHuman" or row["protocol"] != "MVH150" or row["role"] != "test":
        raise ValueError("This runner accepts only the frozen MVHuman MVH150 Test protocol")
    if int(row["num_frames"]) != EXPECTED_FRAMES or int(row["fps"]) != 30:
        raise ValueError("MVH150 temporal contract drifted")
    return row


def safe_video(root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"Unsafe runtime video path: {value}")
    root = root.resolve()
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file():
        raise FileNotFoundError(path)
    return path


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    if args.size != 512:
        raise ValueError("The frozen MVH150 contract uses 512-pixel inference")
    record = read_runtime(args.runtime_manifest.resolve(), int(args.line))
    output = args.output.resolve()
    if output.exists() or output.with_suffix(".runtime.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    video = safe_video(args.derived_root, str(record["input_video"]))
    paths, decode_runtime = shared.decode_frames(video, args.work_dir, str(record["case_id"]))
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Formal MVH150 inference requires an explicit CUDA device")
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
    automatic_proposal = frozen.first_positive(detector_labels)
    proposal = int(args.control_event_index)
    if proposal <= 0 or proposal >= len(paths):
        raise ValueError("The evaluator-timed control event must be inside the sequence")
    runtime["causal_gru_detector"] = {
        "seconds": time.perf_counter() - detector_started,
        "labels": detector_labels,
        "rows": detector_rows,
        "first_positive_index": automatic_proposal,
        "deployment_policy": "reported for diagnosis only; it does not time this control",
    }
    runtime["evaluator_timed_control"] = {
        "event": proposal,
        "automatic_first_positive_index": automatic_proposal,
        "purpose": "isolate boundary-alignment quality from automatic detector quality",
    }

    original_model = ARCroco3DStereo.from_pretrained(str(original)).to(device)
    frozen.strict_original(original_model); original_model.eval()
    original_layer = SMPL_Layer(type="smplx", gender="neutral", num_betas=10, kid=False, person_center="head").to(device).eval()
    strict, runtime["strict_forward"] = frozen.run_no_event(
        original_model, original_layer, topology, paths, device, args.size, "mvhuman_strict_human3r"
    )
    del original_layer, original_model
    gc.collect(); torch.cuda.empty_cache()

    current_model = ARCroco3DStereo.from_pretrained(str(current)).to(device)
    flags = configure_model(current_model); current_model.eval()
    current_layer = SMPL_Layer(type="smplx", gender="neutral", num_betas=10, kid=False, person_center="head").to(device).eval()
    if proposal is None:
        parent, parent_runtime = frozen.run_no_event(
            current_model, current_layer, topology, paths, device, args.size, "mvhuman_detector_miss_parent"
        )
        methods = {
            "m0_strict_human3r": strict,
            "m1_clean_reset": copy.deepcopy(parent),
            "m3_coarse_gauge": copy.deepcopy(parent),
            "m4_coarse_gauge_identity": copy.deepcopy(parent),
            "m6_fine_alignment": copy.deepcopy(parent),
            "m14_gated_parent": parent,
        }
        runtime["causal_transaction"] = {"event": None, "reason": "detector emitted no positive", "runtime": {"parent": parent_runtime}}
    else:
        candidate, geometry, candidate_runtime = frozen.run_transaction(
            current_model, current_layer, topology, paths, int(proposal), device, args.size, "mvhuman_evaluator_timed_control"
        )
        methods = {
            "m0_strict_human3r": strict,
            "m1_clean_reset": candidate["m1_clean_reset"],
            "m3_coarse_gauge": candidate["m3_b0_only"],
            "m4_coarse_gauge_identity": candidate["m4_b0_identity"],
            "m6_fine_alignment": candidate["m6_b0_identity_brtc_c1"],
            "m14_gated_parent": candidate["m14_safe_boundary_permutation_oracle"],
        }
        runtime["causal_transaction"] = {"event": int(proposal), "geometry": geometry, "runtime": candidate_runtime}
    del current_layer, current_model
    gc.collect(); torch.cuda.empty_cache()

    arrays = frozen.pack_methods(methods, topology)
    bridge_name = "m15_bridge3r_full"
    if proposal is None:
        for key in ("cameras_c2w", "vertices_world", "joints_world", "persistent_ids", "native_ids", "valid"):
            arrays[f"{bridge_name}__{key}"] = arrays[f"m3_coarse_gauge__{key}"].copy()
        runtime["bridge3r"] = {"source": "exact current parent after detector miss", "exact_parent_fallback": True}
    else:
        source_arrays = {
            key: np.asarray(arrays[f"m3_coarse_gauge__{key}"]).copy()
            for key in ("cameras_c2w", "vertices_world", "joints_world", "persistent_ids", "native_ids", "valid")
        }
        pairs = [tuple(map(int, pair)) for pair in runtime["causal_transaction"]["geometry"].get("association", {}).get("pairs", [])]
        bridge_arrays, diagnostics = apply_candidate(
            source_arrays,
            int(proposal),
            pairs,
            Candidate(name="bridge3r_frozen_operating_point", camera_alpha=1.0, boundary_kind="translation", boundary_blend=0.5),
        )
        if diagnostics.get("runtime_contract", {}).get("exact_m15_fallback"):
            raise RuntimeError("Frozen Bridge3R candidate unexpectedly entered a gated fallback")
        for key, value in bridge_arrays.items():
            arrays[f"{bridge_name}__{key}"] = np.asarray(value)
        runtime["bridge3r"] = {"source": "learned coarse gauge at evaluator-timed control boundary", "association_pairs": pairs, "diagnostics": diagnostics}

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
        "environment": {
            "python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "device": str(device), "gpu": torch.cuda.get_device_name(device),
            "process_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        },
        "total_process_seconds": time.perf_counter() - started,
        "runtime_contract": {
            "gt_in_runtime": True, "camera_or_cut_in_runtime": True, "future_frames_at_boundary": 0,
            "transaction_boundary_source": "evaluator-timed diagnostic control",
            "detector_miss_policy": "exact current parent; no oracle boundary substitution",
            "eligible_for_automatic_method_ranking": False,
        },
        "cache": str(output), "cache_sha256": shared.sha256(output),
    }
    report_path = output.with_suffix(".runtime.json")
    report_path.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not args.keep_decoded_frames:
        shutil.rmtree(args.work_dir.resolve() / str(record["case_id"]))
    print(json.dumps({"case_id": record["case_id"], "cache": str(output), "control_event": proposal, "detector_first_positive": automatic_proposal}, indent=2))


if __name__ == "__main__":
    main()
