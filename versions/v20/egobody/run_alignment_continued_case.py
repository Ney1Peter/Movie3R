#!/usr/bin/env python3
"""Run the alignment-plus-continued-state EgoBody counterfactual.

This runner is intentionally separate from the frozen v20 protocol.  It uses
the same trained Shot3R checkpoint and the annotated transition, enables the
history-conditioned alignment path at that transition, and then commits the
state produced by that path for all subsequent frames.  Existing v20 caches
are read neither here nor overwritten.
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


REPO_ROOT = Path(__file__).resolve().parents[3]
for root in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from dust3r.model import ARCroco3DStereo  # noqa: E402
from dust3r.utils.smpl_layer import SMPL_Layer  # noqa: E402
from versions.v13 import gt_id_consensus as gt_helpers  # noqa: E402
from versions.v14.run_v14_2_single_sequence import (  # noqa: E402
    configure_model,
    set_event_indices,
)
from versions.v15.harmony4d import run_harmony_case as frozen  # noqa: E402
from versions.v15.harmony4d.topology import CommonTopology  # noqa: E402
from versions.v20.egobody import run_egobody_case as egobody  # noqa: E402


SCHEMA = "Shot3R-EgoBody-alignment-continued-runtime-v1"
METHOD = "m3_alignment_continued"


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
    parser.add_argument(
        "--verify-causal-prefix",
        action="store_true",
        help="Rerun the prefix through the boundary and verify causal equality.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def max_frame_difference(
    left: dict[str, Any], right: dict[str, Any]
) -> dict[str, float | int]:
    """Compare two decoded boundary frames without assuming a fixed count."""

    output: dict[str, float | int] = {
        "camera_max_abs": float(
            np.max(np.abs(np.asarray(left["camera"]) - np.asarray(right["camera"])))
        ),
        "left_people": len(left["people"]),
        "right_people": len(right["people"]),
    }
    if len(left["people"]) == len(right["people"]) and left["people"]:
        output["root_max_abs"] = float(max(
            np.max(np.abs(np.asarray(a["root"]) - np.asarray(b["root"])))
            for a, b in zip(left["people"], right["people"])
        ))
    else:
        output["root_max_abs"] = float("inf")
    return output


def atomic_json(path: Path, payload: Any) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(frozen.jsonable(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    record = egobody.read_record(args)
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    paths = egobody.image_paths(record, args.staged_root)
    boundary = int(record["boundary_index"])
    if boundary <= 0 or boundary >= len(paths):
        raise ValueError(f"invalid boundary {boundary} for {len(paths)} frames")

    default_current, _ = frozen.default_checkpoints()
    checkpoint = (args.current_checkpoint or default_current).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    topology = CommonTopology.load()
    model = ARCroco3DStereo.from_pretrained(str(checkpoint)).to(device)
    flags = configure_model(model)
    model.eval()
    layer = SMPL_Layer(
        type="smplx", gender="neutral", num_betas=10, kid=False,
        person_center="head",
    ).to(device).eval()

    views = gt_helpers.prepare_full_square_input(
        model, paths, SimpleNamespace(size=int(args.size))
    )
    views = set_event_indices(views, {boundary})
    predictions, returned, debug, forward_runtime = frozen.run_forward(
        model, views, device, "alignment_continued"
    )
    frames = frozen.decode_sequence(predictions, returned, debug, layer, topology)
    del predictions, returned, debug, views

    prefix_audit: dict[str, Any] | None = None
    if args.verify_causal_prefix:
        prefix_views = gt_helpers.prepare_full_square_input(
            model, paths[: boundary + 1], SimpleNamespace(size=int(args.size))
        )
        prefix_views = set_event_indices(prefix_views, {boundary})
        prefix_predictions, prefix_returned, prefix_debug, prefix_runtime = frozen.run_forward(
            model, prefix_views, device, "alignment_continued_prefix_audit"
        )
        prefix_frames = frozen.decode_sequence(
            prefix_predictions, prefix_returned, prefix_debug, layer, topology
        )
        difference = max_frame_difference(frames[boundary], prefix_frames[-1])
        if (
            difference["camera_max_abs"] > 1e-6
            or difference["root_max_abs"] > 1e-5
            or difference["left_people"] != difference["right_people"]
        ):
            raise AssertionError(f"causal prefix mismatch: {difference}")
        prefix_audit = {
            "runtime": prefix_runtime,
            "boundary_frame_difference": difference,
            "passed": True,
        }
        del prefix_predictions, prefix_returned, prefix_debug, prefix_views, prefix_frames

    arrays = frozen.pack_methods({METHOD: frames}, topology)
    partial = output.with_suffix(output.suffix + ".partial")
    with partial.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(partial, output)

    del frames, layer, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    report = {
        "schema_version": SCHEMA,
        "record": record,
        "methods": [METHOD],
        "runtime": {
            "alignment_continued_forward": forward_runtime,
            "causal_prefix_audit": prefix_audit,
        },
        "checkpoint": {
            "current": str(checkpoint),
            "current_sha256": frozen.verified_artifact_sha256(checkpoint),
            "current_flags": flags,
        },
        "topology": topology.metadata(),
        "provenance": {
            **frozen.git_provenance(),
            "manifest": str(args.manifest.resolve()) if args.manifest else None,
            "manifest_sha256": frozen.sha256(args.manifest.resolve()) if args.manifest else None,
            "manifest_line": int(args.line) if args.manifest else None,
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
        "counterfactual_contract": {
            "annotated_boundary": True,
            "history_conditioned_alignment_enabled": True,
            "boundary_path_state_propagated_after_boundary": True,
            "separate_post_shot_reinitialization": False,
            "association_or_shared_translation": False,
            "future_frames_at_boundary": 0,
            "gt_in_runtime": False,
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
        "method": METHOD,
        "boundary": boundary,
        "causal_prefix_audit": prefix_audit,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
