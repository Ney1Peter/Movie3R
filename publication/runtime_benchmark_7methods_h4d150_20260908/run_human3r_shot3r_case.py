#!/usr/bin/env python3
"""Run exactly one Human3R or Shot3R prediction on one fixed RGB directory.

The script materializes only the requested publication route.  In particular,
it does not run Human3R while timing Shot3R and does not construct exploratory
Shot3R ablations.  Process-level wall time is measured by ``run_internal.py``;
the component times recorded here are diagnostic checks.
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


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
for root in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from dust3r.model import ARCroco3DStereo  # noqa: E402
from dust3r.utils.smpl_layer import SMPL_Layer  # noqa: E402
from publication.bridge3r_iclr2027.benchmark_runtime_memory import (  # noqa: E402
    minimal_locked_geometry,
)
from versions.v13 import gt_id_consensus as input_helpers  # noqa: E402
from versions.v14.causal_image_detector import CausalGRUShotDetector  # noqa: E402
from versions.v14.run_v14_2_single_sequence import configure_model, set_event_indices  # noqa: E402
from versions.v15.harmony4d import run_harmony_case as frozen  # noqa: E402
from versions.v15.harmony4d.topology import CommonTopology  # noqa: E402


SCHEMA = "Shot3R-H4D-CS150-internal-runtime-v1"
EXPECTED_FRAMES = 150


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("human3r", "shot3r"), required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--size", type=int, default=512)
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


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def forward(model: ARCroco3DStereo, views: list[dict[str, Any]], device: torch.device):
    synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        outputs = model.forward_recurrent_lighter(
            views,
            str(device),
            ret_state=False,
            use_ttt3r=False,
            return_token_debug=True,
        )
    synchronize(device)
    return outputs, time.perf_counter() - started


def prepare(model: ARCroco3DStereo, paths: list[Path], size: int):
    return input_helpers.prepare_full_square_input(model, paths, SimpleNamespace(size=int(size)))


def save_arrays(path: Path, arrays: dict[str, np.ndarray]) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    started = time.perf_counter()
    with temporary.open("wb") as handle:
        # Uncompressed native arrays avoid making the reported speed depend on
        # a method-specific compression ratio.
        np.savez(handle, **arrays)
    os.replace(temporary, path)
    return time.perf_counter() - started


def run_human3r(
    paths: list[Path], device: torch.device, size: int, output: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    _, original = frozen.default_checkpoints()
    load_started = time.perf_counter()
    model = ARCroco3DStereo.from_pretrained(str(original)).to(device).eval()
    frozen.strict_original(model)
    layer = SMPL_Layer(
        type="smplx", gender="neutral", num_betas=10, kid=False, person_center="head"
    ).to(device).eval()
    topology = CommonTopology.load()
    model_load_seconds = time.perf_counter() - load_started

    preprocess_started = time.perf_counter()
    views = set_event_indices(prepare(model, paths, size), set())
    preprocess_seconds = time.perf_counter() - preprocess_started
    (predictions, returned, debug), forward_seconds = forward(model, views, device)
    decode_started = time.perf_counter()
    decoded = frozen.decode_sequence(predictions, returned, debug, layer, topology)
    synchronize(device)
    decode_seconds = time.perf_counter() - decode_started
    arrays = frozen.pack_methods({"human3r": decoded}, topology)
    serialization_seconds = save_arrays(output, arrays)
    valid = arrays["human3r__valid"]
    result = {
        "frames": len(paths),
        "valid_person_frames": int(valid.sum()),
        "output": str(output),
    }
    runtime = {
        "checkpoint_and_model_load_seconds": model_load_seconds,
        "rgb_decode_resize_seconds": preprocess_seconds,
        "neural_forward_seconds": forward_seconds,
        "decode_and_geometry_seconds": decode_seconds,
        "native_serialization_seconds": serialization_seconds,
    }
    del arrays, decoded, predictions, returned, debug, views, layer, model
    gc.collect()
    torch.cuda.empty_cache()
    return result, runtime


def run_shot3r(
    paths: list[Path], device: torch.device, size: int, output: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    detector_started = time.perf_counter()
    detector = CausalGRUShotDetector(Path(frozen.DETECTOR_PATH))
    labels, rows = detector.predict_sequence(paths)
    proposal = frozen.first_positive(labels)
    detector_seconds = time.perf_counter() - detector_started

    current, _ = frozen.default_checkpoints()
    load_started = time.perf_counter()
    model = ARCroco3DStereo.from_pretrained(str(current)).to(device).eval()
    flags = configure_model(model)
    layer = SMPL_Layer(
        type="smplx", gender="neutral", num_betas=10, kid=False, person_center="head"
    ).to(device).eval()
    topology = CommonTopology.load()
    model_load_seconds = time.perf_counter() - load_started

    if proposal is None:
        preprocess_started = time.perf_counter()
        views = set_event_indices(prepare(model, paths, size), set())
        preprocess_seconds = time.perf_counter() - preprocess_started
        (predictions, returned, debug), forward_seconds = forward(model, views, device)
        decode_started = time.perf_counter()
        decoded = frozen.decode_sequence(predictions, returned, debug, layer, topology)
        arrays0 = frozen.pack_methods({"shot3r": decoded}, topology)
        arrays = {key.removeprefix("shot3r__"): value for key, value in arrays0.items()}
        synchronize(device)
        decode_geometry_seconds = time.perf_counter() - decode_started
        neural_evaluations = len(paths)
    else:
        boundary = int(proposal)
        if boundary <= 0 or boundary >= len(paths):
            raise RuntimeError(f"invalid detector proposal {boundary}")
        preprocess_started = time.perf_counter()
        pre_views = prepare(model, paths[:boundary], size)
        post_views = prepare(model, paths[boundary:], size)
        shadow_views = set_event_indices(pre_views + post_views[:1], {boundary})
        raw_post_views = set_event_indices(post_views, set())
        preprocess_seconds = time.perf_counter() - preprocess_started

        (shadow_pred, shadow_ret, shadow_dbg), shadow_seconds = forward(
            model, shadow_views, device
        )
        (post_pred, post_ret, post_dbg), post_seconds = forward(
            model, raw_post_views, device
        )
        decode_started = time.perf_counter()
        shadow = frozen.decode_sequence(shadow_pred, shadow_ret, shadow_dbg, layer, topology)
        raw_post = frozen.decode_sequence(post_pred, post_ret, post_dbg, layer, topology)
        arrays, geometry = minimal_locked_geometry(shadow, raw_post, boundary, topology)
        synchronize(device)
        decode_geometry_seconds = time.perf_counter() - decode_started
        forward_seconds = shadow_seconds + post_seconds
        neural_evaluations = len(shadow_views) + len(raw_post_views)
        del shadow_pred, shadow_ret, shadow_dbg, post_pred, post_ret, post_dbg
        del shadow, raw_post, geometry, shadow_views, raw_post_views, pre_views, post_views

    serialization_seconds = save_arrays(output, arrays)
    result = {
        "frames": len(paths),
        "valid_person_frames": int(np.asarray(arrays["valid"]).sum()),
        "detector_first_positive": proposal,
        "detector_positive_indices": [i for i, value in enumerate(labels) if int(value)],
        "detector_pair_rows": len(rows),
        "neural_frame_evaluations": neural_evaluations,
        "model_flags": flags,
        "output": str(output),
    }
    runtime = {
        "transition_detector_seconds": detector_seconds,
        "checkpoint_and_model_load_seconds": model_load_seconds,
        "rgb_decode_resize_seconds": preprocess_seconds,
        "neural_forward_seconds": forward_seconds,
        "decode_and_geometry_seconds": decode_geometry_seconds,
        "native_serialization_seconds": serialization_seconds,
    }
    del arrays, layer, model
    gc.collect()
    torch.cuda.empty_cache()
    return result, runtime


def main() -> None:
    script_started = time.perf_counter()
    args = parse_args()
    if int(args.size) != 512:
        raise ValueError("publication protocol fixes the model input size to 512")
    input_dir = args.input_dir.resolve()
    paths = sorted(input_dir.glob("*.jpg"))
    if len(paths) != EXPECTED_FRAMES:
        raise ValueError(f"expected {EXPECTED_FRAMES} JPEGs in {input_dir}, found {len(paths)}")
    expected_names = [f"{index:06d}.jpg" for index in range(EXPECTED_FRAMES)]
    if [path.name for path in paths] != expected_names:
        raise ValueError("input JPEG names/order do not match the frozen 000000--000149 contract")
    device = torch.device(args.device)
    if device.type != "cuda" or device.index is None:
        raise ValueError("an explicit single CUDA device is required")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    if args.method == "human3r":
        result, runtime = run_human3r(paths, device, int(args.size), args.output.resolve())
    else:
        result, runtime = run_shot3r(paths, device, int(args.size), args.output.resolve())
    report = {
        "schema_version": SCHEMA,
        "method": args.method,
        "case_id": args.case_id,
        "input_dir": str(input_dir),
        "result": result,
        "components": runtime,
        "script_seconds": time.perf_counter() - script_started,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "precision": "FP32; TF32 disabled",
            "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        },
    }
    report_path = args.output.resolve().with_suffix(".internal.json")
    report_path.write_text(json.dumps(jsonable(report), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), **report}, indent=2))


if __name__ == "__main__":
    main()
