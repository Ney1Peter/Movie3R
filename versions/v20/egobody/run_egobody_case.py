#!/usr/bin/env python3
"""Run one frozen EgoBody CS150 case without opening evaluator-only data.

This is a small dataset adapter around the audited v15 inference primitives.
Unlike the legacy runner it reads explicit RGB members from the runtime
manifest and stores only the methods needed by the EgoBody study.  The cache
also contains a detector-specific B0 source whenever the causal proposal does
not equal the evaluation cut, so detector errors are evaluated rather than
silently replaced by the oracle boundary.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import platform
import resource
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


SCHEMA = "Bridge3R-EgoBody-CS150-runtime-cache-v1"
CORE_METHODS = (
    "m0_strict_human3r",
    "m0r_original_clean_reset",
    "m1_current_clean_reset",
    "m3_b0_only",
    "m15_v17_gated_parent",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--record-json")
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--line", type=int, default=1)
    parser.add_argument("--staged-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--current-checkpoint", type=Path)
    parser.add_argument("--original-checkpoint", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_record(args: argparse.Namespace) -> dict[str, Any]:
    if args.record_json is not None:
        record = json.loads(args.record_json)
    else:
        rows = [
            line for line in args.manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if args.line < 1 or args.line > len(rows):
            raise IndexError(f"line {args.line} outside manifest with {len(rows)} rows")
        record = json.loads(rows[args.line - 1])
    required = {
        "case_id", "pre_frame_numbers", "post_frame_numbers", "boundary_index",
        "pre_camera", "post_camera",
    }
    missing = required.difference(record)
    if missing:
        raise ValueError(f"runtime row misses {sorted(missing)}")
    if int(record["boundary_index"]) != len(record["pre_frame_numbers"]):
        raise ValueError("boundary_index must equal pre-frame count")
    if "image_members" not in record and "image_paths" not in record:
        raise ValueError("runtime row must contain image_members or image_paths")
    return record


def safe_relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe staged RGB path: {value!r}")
    return path


def image_paths(record: dict[str, Any], staged_root: Path) -> list[Path]:
    values = record.get("image_paths") or record.get("image_members")
    expected = len(record["pre_frame_numbers"]) + len(record["post_frame_numbers"])
    if len(values) != expected:
        raise ValueError(f"expected {expected} image paths, received {len(values)}")
    root = staged_root.resolve()
    output = [(root / safe_relative(str(value))).resolve() for value in values]
    if any(root not in path.parents for path in output):
        raise ValueError("resolved RGB path escapes staged root")
    missing = [path for path in output if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    return output


def atomic_json(path: Path, payload: Any) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(frozen.jsonable(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def run_reset_pair(
    model: ARCroco3DStereo,
    layer: SMPL_Layer,
    topology: CommonTopology,
    paths: list[Path],
    boundary: int,
    device: torch.device,
    size: int,
    label: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pre, pre_runtime = frozen.run_no_event(
        model, layer, topology, paths[:boundary], device, size, f"{label}_pre"
    )
    post, post_runtime = frozen.run_no_event(
        model, layer, topology, paths[boundary:], device, size, f"{label}_post"
    )
    return pre + post, {"pre_forward": pre_runtime, "post_forward": post_runtime}


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    record = read_record(args)
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = image_paths(record, args.staged_root)
    boundary = int(record["boundary_index"])
    if boundary <= 0 or boundary >= len(paths):
        raise ValueError(f"invalid boundary {boundary} for {len(paths)} frames")
    default_current, default_original = frozen.default_checkpoints()
    current_path = (args.current_checkpoint or default_current).resolve()
    original_path = (args.original_checkpoint or default_original).resolve()
    for path in (current_path, original_path, frozen.DETECTOR_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    topology = CommonTopology.load()
    runtime: dict[str, Any] = {}

    detector_started = time.perf_counter()
    detector = CausalGRUShotDetector(frozen.DETECTOR_PATH)
    detector_labels, detector_rows = detector.predict_sequence(paths)
    proposal = frozen.first_positive(detector_labels)
    runtime["causal_gru_detector"] = {
        "seconds": time.perf_counter() - detector_started,
        "labels": detector_labels,
        "rows": detector_rows,
        "evaluation_boundary": boundary,
        "proposal_boundary": proposal,
        # Keep the frozen v15 evaluator adapter and the v20 deployment probe
        # on the same first-positive detector contract.
        "first_positive_index": proposal,
        "matches_evaluation_boundary": proposal == boundary,
        "false_positive_indices": [
            index for index, value in enumerate(detector_labels)
            if int(value) and index != boundary
        ],
        "deployment_policy": "first positive only",
    }

    original_model = ARCroco3DStereo.from_pretrained(str(original_path)).to(device)
    frozen.strict_original(original_model)
    original_model.eval()
    original_layer = SMPL_Layer(
        type="smplx", gender="neutral", num_betas=10, kid=False,
        person_center="head",
    ).to(device).eval()
    strict_frames, runtime["m0_forward"] = frozen.run_no_event(
        original_model, original_layer, topology, paths, device, int(args.size),
        "strict_original_human3r",
    )
    reset_frames, runtime["m0r_forward"] = run_reset_pair(
        original_model, original_layer, topology, paths, boundary, device,
        int(args.size), "original_clean_reset",
    )
    del original_layer, original_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    current_model = ARCroco3DStereo.from_pretrained(str(current_path)).to(device)
    flags = configure_model(current_model)
    current_model.eval()
    current_layer = SMPL_Layer(
        type="smplx", gender="neutral", num_betas=10, kid=False,
        person_center="head",
    ).to(device).eval()
    oracle_methods, oracle_geometry, oracle_runtime = frozen.run_transaction(
        current_model, current_layer, topology, paths, boundary, device,
        int(args.size), "evaluation_boundary",
    )
    runtime["oracle_transaction"] = oracle_runtime

    if proposal is None:
        detector_frames, detector_forward = frozen.run_no_event(
            current_model, current_layer, topology, paths, device, int(args.size),
            "detector_no_event",
        )
        detector_methods = {
            "m3_b0_only": detector_frames,
            "m14_safe_boundary_permutation_oracle": detector_frames,
        }
        detector_geometry: dict[str, Any] = {
            "event": None,
            "reason": "detector emitted no positive",
            "runtime": {"no_event_forward": detector_forward},
        }
    elif proposal == boundary:
        detector_methods = oracle_methods
        detector_geometry = {
            "event": proposal,
            "reused_evaluation_boundary": True,
            "geometry": oracle_geometry,
        }
    else:
        detector_methods, proposal_geometry, proposal_runtime = frozen.run_transaction(
            current_model, current_layer, topology, paths, proposal, device,
            int(args.size), "detector_proposal",
        )
        detector_geometry = {
            "event": proposal,
            "reused_evaluation_boundary": False,
            "geometry": proposal_geometry,
            "runtime": proposal_runtime,
        }
    del current_layer, current_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    methods = {
        "m0_strict_human3r": strict_frames,
        "m0r_original_clean_reset": reset_frames,
        "m1_current_clean_reset": oracle_methods["m1_clean_reset"],
        "m3_b0_only": oracle_methods["m3_b0_only"],
        "m15_v17_gated_parent": oracle_methods[
            "m14_safe_boundary_permutation_oracle"
        ],
    }
    # Exact detector proposals reuse the oracle arrays through provenance
    # instead of storing byte-identical 6890-vertex tensors twice.
    if proposal != boundary:
        methods["m3_causal_detector_b0"] = detector_methods["m3_b0_only"]
        methods["m15_causal_detector_parent"] = detector_methods[
            "m14_safe_boundary_permutation_oracle"
        ]
    if tuple(methods)[: len(CORE_METHODS)] != CORE_METHODS:
        raise AssertionError(tuple(methods))
    arrays = frozen.pack_methods(methods, topology)
    partial = output.with_suffix(output.suffix + ".partial")
    with partial.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(partial, output)

    report = {
        "schema_version": SCHEMA,
        "record": record,
        "methods": list(methods),
        "runtime": runtime,
        "checkpoint": {
            "current": str(current_path),
            "current_sha256": frozen.verified_artifact_sha256(current_path),
            "original": str(original_path),
            "original_sha256": frozen.verified_artifact_sha256(original_path),
            "detector": str(frozen.DETECTOR_PATH),
            "detector_sha256": frozen.verified_artifact_sha256(frozen.DETECTOR_PATH),
            "current_flags": flags,
        },
        "geometry": {
            # Compatibility copy for prediction-only v16/v19 candidate probes.
            # The canonical audited value remains under evaluation_boundary.
            "association": oracle_geometry.get("association", {}),
            "evaluation_boundary": oracle_geometry,
            "detector_driven": detector_geometry,
        },
        "topology": topology.metadata(),
        "provenance": {
            **frozen.git_provenance(),
            "manifest": str(args.manifest.resolve()) if args.manifest else None,
            "manifest_sha256": frozen.sha256(args.manifest.resolve()) if args.manifest else None,
            "manifest_line": int(args.line) if args.manifest else None,
            "protocol_seed": record.get("protocol_seed"),
            "argv": sys.argv,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "precision": "FP32",
            "process_peak_rss_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            ),
        },
        "runtime_contract": {
            "gt_in_runtime": False,
            "future_frames_at_boundary": 0,
            "pre_cut_frames_mutated": False,
            "evaluation_boundary": boundary,
            "proposal_boundary": proposal,
            "detector_specific_source_materialized": proposal != boundary,
            "detector_arrays_reuse_evaluation_boundary": proposal == boundary,
        },
        "total_process_seconds": time.perf_counter() - started,
        "cache": str(output),
        "cache_sha256": frozen.sha256(output),
    }
    report_path = output.with_suffix(".runtime.json")
    atomic_json(report_path, report)
    print(json.dumps({
        "case_id": record["case_id"],
        "cache": str(output),
        "runtime_report": str(report_path),
        "methods": list(methods),
        "evaluation_boundary": boundary,
        "proposal_boundary": proposal,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
