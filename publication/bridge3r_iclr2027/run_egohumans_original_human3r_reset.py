#!/usr/bin/env python3
"""Run a post-cut fresh-state pass with the original Human3R checkpoint.

This narrow runner complements the frozen EgoHumans formal90 cache.  Its
continuous-state branch was produced by strict original Human3R, whereas the
historical ``m1_clean_reset`` cache used the trained Shot3R checkpoint and is
therefore not a valid state-only control.  Here the exact original checkpoint
is reused and only the 50 post-cut RGB frames are processed from initial state.
Only camera poses are serialized because the state-isolation claim is tested
with gauge-invariant within-shot camera motion.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
import sys

for root in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from dust3r.model import ARCroco3DStereo  # noqa: E402
from versions.v13 import gt_id_consensus as gt_helpers  # noqa: E402
from versions.v14.run_v14_2_single_sequence import camera_matrix, set_event_indices  # noqa: E402
from versions.v15.harmony4d.run_harmony_case import (  # noqa: E402
    default_checkpoints,
    frame_image_paths,
    run_forward,
    sha256,
    strict_original,
    verified_artifact_sha256,
)


SCHEMA = "Shot3R-EgoHumans-original-Human3R-reset-v1"


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=REPO_ROOT
        / "output/bridge3r_egohumans_ablation_v1/formal90_native/test/predictions",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "output/egohumans_state_isolation_formal90/original_reset",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument(
        "--archive-entry",
        help="Optional exact EgoHumans capture entry; selection is checked against source runtimes.",
    )
    parser.add_argument(
        "--extracted-root",
        type=Path,
        help=(
            "Optional staged root replacing the historical --extracted-root recorded in "
            "the source runtime. Required when the original disk-bounded staging was removed."
        ),
    )
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def argv_value(argv: list[str], flag: str) -> str:
    try:
        return argv[argv.index(flag) + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"source runtime does not bind {flag}") from exc


def complete(npz_path: Path, runtime_path: Path, case_id: str, checkpoint_sha: str) -> bool:
    if not npz_path.is_file() or not runtime_path.is_file():
        return False
    try:
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        if (
            runtime.get("schema_version") != SCHEMA
            or runtime.get("case_id") != case_id
            or runtime.get("checkpoint_sha256") != checkpoint_sha
            or runtime.get("output_sha256") != sha256(npz_path)
        ):
            return False
        with np.load(npz_path, allow_pickle=False) as payload:
            cameras = payload["original_human3r_reset__cameras_c2w"]
        return cameras.shape == (50, 4, 4) and bool(np.all(np.isfinite(cameras)))
    except Exception:
        return False


def main() -> None:
    options = args()
    if options.num_shards <= 0 or not 0 <= options.shard_index < options.num_shards:
        raise ValueError("invalid shard")
    source_root = options.source_root.resolve(strict=True)
    output_root = options.output_root.resolve()
    all_sources = sorted(source_root.glob("*/*.npz"))
    if len(all_sources) != 90:
        raise ValueError(f"formal90 requires 90 source caches, found {len(all_sources)}")
    sources = all_sources
    if options.archive_entry is not None:
        selected = []
        for source in sources:
            runtime = json.loads(source.with_suffix(".runtime.json").read_text(encoding="utf-8"))
            if runtime["record"]["archive_entry"] == options.archive_entry:
                selected.append(source)
        sources = selected
        if not sources:
            raise ValueError(f"no formal90 cases found for {options.archive_entry}")
    sources = [path for index, path in enumerate(sources) if index % options.num_shards == options.shard_index]

    device = torch.device(options.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this audited runner requires CUDA")
    torch.cuda.set_device(device)
    _, original_checkpoint = default_checkpoints()
    original_checkpoint = original_checkpoint.resolve(strict=True)
    checkpoint_sha = verified_artifact_sha256(original_checkpoint)

    model = ARCroco3DStereo.from_pretrained(str(original_checkpoint)).to(device)
    strict_original(model)
    model.eval()

    completed = 0
    for ordinal, source in enumerate(sources, start=1):
        source_runtime_path = source.with_suffix(".runtime.json")
        source_runtime = json.loads(source_runtime_path.read_text(encoding="utf-8"))
        source_cache_sha = sha256(source)
        if source_cache_sha != source_runtime["cache_sha256"]:
            raise ValueError(f"source cache SHA mismatch for {source}")
        record: dict[str, Any] = source_runtime["record"]
        case_id = str(record["case_id"])
        if int(record["boundary_index"]) != 50 or int(record["clip_length"]) != 100:
            raise ValueError(f"unexpected formal90 record geometry for {case_id}")
        if source_runtime["checkpoint"]["original_sha256"] != checkpoint_sha:
            raise ValueError(f"original checkpoint mismatch for {case_id}")

        relative = source.relative_to(source_root)
        output = (output_root / relative).resolve()
        runtime_output = output.with_suffix(".runtime.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        if not options.overwrite and complete(output, runtime_output, case_id, checkpoint_sha):
            print(f"[{ordinal}/{len(sources)}] cached {case_id}", flush=True)
            completed += 1
            continue

        original_argv = list(source_runtime["provenance"]["argv"])
        recorded_extracted_root = Path(argv_value(original_argv, "--extracted-root"))
        extracted_root = (
            options.extracted_root if options.extracted_root is not None else recorded_extracted_root
        ).resolve(strict=True)
        sequence_root = extracted_root / str(record["capture_relative"])
        _, post_paths = frame_image_paths(sequence_root, record)
        if len(post_paths) != 50:
            raise ValueError(f"expected 50 post-cut frames for {case_id}")

        views = set_event_indices(
            gt_helpers.prepare_full_square_input(model, post_paths, SimpleNamespace(size=int(options.size))),
            set(),
        )
        predictions, returned, debug, forward = run_forward(
            model, views, device, f"original_human3r_reset:{case_id}"
        )
        cameras = np.stack([camera_matrix(prediction) for prediction in predictions]).astype(np.float32)
        if cameras.shape != (50, 4, 4) or not np.all(np.isfinite(cameras)):
            raise ValueError(f"invalid camera output for {case_id}")

        temporary = output.with_suffix(".partial.npz")
        np.savez_compressed(temporary, original_human3r_reset__cameras_c2w=cameras)
        os.replace(temporary, output)
        payload = {
            "schema_version": SCHEMA,
            "case_id": case_id,
            "source_continued_cache": str(source.resolve()),
            "source_continued_cache_sha256_recorded": source_runtime["cache_sha256"],
            "source_continued_cache_sha256_verified": source_cache_sha,
            "source_runtime": str(source_runtime_path.resolve()),
            "source_runtime_sha256": sha256(source_runtime_path),
            "checkpoint": str(original_checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "same_original_checkpoint_as_continued": True,
            "post_frame_count": len(post_paths),
            "archive_entry": str(record["archive_entry"]),
            "staged_extracted_root": str(extracted_root),
            "recorded_historical_extracted_root": str(recorded_extracted_root),
            "boundary_index": 50,
            "future_frames_before_reset": 0,
            "event_token_enabled": False,
            "forward": forward,
            "device": str(device),
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device),
            },
            "output": str(output),
            "output_sha256": sha256(output),
        }
        runtime_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        del predictions, returned, debug, views, cameras
        gc.collect()
        torch.cuda.empty_cache()
        completed += 1
        print(f"[{ordinal}/{len(sources)}] complete {case_id}", flush=True)

    print(json.dumps({"shard": options.shard_index, "completed": completed, "expected": len(sources)}))


if __name__ == "__main__":
    main()
