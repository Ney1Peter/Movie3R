#!/usr/bin/env python3
"""Export M15 and frozen v16 geometry in the standard demo.py payload format."""

from __future__ import annotations

import copy
import gc
import json
import sys
import time
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
for item in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from dust3r.model import ARCroco3DStereo  # noqa: E402
from versions.v15.harmony4d.evaluate_harmony import method_arrays  # noqa: E402
from versions.v15.harmony4d.export_harmony_qualitative import (  # noqa: E402
    backgrounds_from_rows,
    load_faces,
    release,
    safe_destination,
    selected_paths,
    write_payload,
)
from versions.v15.harmony4d.run_harmony_case import (  # noqa: E402
    configure_model,
    gt_helpers,
    run_forward,
    set_event_indices,
)
from versions.v16.harmony4d.causal_stabilization import (  # noqa: E402
    Candidate,
    apply_candidate,
)


SOURCE_METHOD = "m3_b0_only"
BASELINE = "v16_0_m15_geometry"
PRIMARY = "v16_harmony_safe"


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--candidate-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--pre-display", type=int, default=5)
    parser.add_argument("--post-display", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_candidate(path: Path) -> Candidate:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["candidates"] if isinstance(payload, dict) else payload
    selected = [Candidate(**row) for row in rows if row["name"] == PRIMARY]
    if len(selected) != 1:
        raise ValueError(f"Expected one {PRIMARY} candidate in {path}")
    return selected[0]


def selected_geometry(arrays: dict[str, np.ndarray], indices: list[int]) -> list[dict[str, np.ndarray]]:
    return [
        {
            "camera": arrays["cameras_c2w"][frame],
            "vertices": arrays["vertices_world"][frame, arrays["valid"][frame].astype(bool)],
            "ids": arrays["persistent_ids"][frame, arrays["valid"][frame].astype(bool)],
        }
        for frame in indices
    ]


def materialize_geometry(
    cache_path: Path,
    runtime: dict[str, Any],
    indices: list[int],
    candidate: Candidate,
) -> tuple[list[dict[str, np.ndarray]], dict[str, Any]]:
    with np.load(cache_path, allow_pickle=False) as cache:
        source = method_arrays(cache, SOURCE_METHOD)
    pairs = [
        tuple(map(int, pair))
        for pair in runtime.get("geometry", {}).get("association", {}).get("pairs", [])
    ]
    arrays, diagnostics = apply_candidate(
        source,
        int(runtime["record"]["boundary_index"]),
        pairs,
        candidate,
    )
    return selected_geometry(arrays, indices), diagnostics


def main() -> None:
    args = parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    candidate = load_candidate(args.candidate_json)
    destination = safe_destination(args.output, bool(args.overwrite))
    faces = load_faces()
    prepared = []
    for item in selection["cases"]:
        runtime, pre, post, indices = selected_paths(
            item, int(args.pre_display), int(args.post_display)
        )
        prepared.append({
            **item,
            "runtime_payload": runtime,
            "cache": Path(runtime["cache"]).resolve(),
            "pre": pre,
            "post": post,
            "indices": indices,
        })
    checkpoint_paths = {row["runtime_payload"]["checkpoint"]["current"] for row in prepared}
    if len(checkpoint_paths) != 1:
        raise ValueError("Selected cases do not share one current checkpoint")
    checkpoint = Path(next(iter(checkpoint_paths)))
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = ARCroco3DStereo.from_pretrained(str(checkpoint)).to(device)
    flags = configure_model(model)
    model.eval()
    rows = []
    for row in prepared:
        runtime = row["runtime_payload"]
        case_id = runtime["record"]["case_id"]
        pre_views = gt_helpers.prepare_full_square_input(
            model, row["pre"], SimpleNamespace(size=int(args.size))
        )
        post_views = gt_helpers.prepare_full_square_input(
            model, row["post"], SimpleNamespace(size=int(args.size))
        )
        shadow_views = set_event_indices(copy.deepcopy(pre_views + post_views[:1]), {len(pre_views)})
        raw_views = set_event_indices(copy.deepcopy(post_views), set())
        started = time.perf_counter()
        shadow_prediction, shadow_returned, shadow_debug, _ = run_forward(
            model, shadow_views, device, "qualitative_v16_shadow"
        )
        raw_prediction, raw_returned, raw_debug, _ = run_forward(
            model, raw_views, device, "qualitative_v16_raw_post"
        )
        pre_indices = list(range(len(row["pre"]) - int(args.pre_display), len(row["pre"])))
        backgrounds = backgrounds_from_rows(shadow_prediction, shadow_returned, pre_indices)
        backgrounds.extend(
            backgrounds_from_rows(raw_prediction, raw_returned, list(range(len(row["post"]))))
        )
        baseline_geometry, baseline_debug = materialize_geometry(
            row["cache"], runtime, row["indices"], Candidate(BASELINE)
        )
        primary_geometry, primary_debug = materialize_geometry(
            row["cache"], runtime, row["indices"], candidate
        )
        baseline_root = destination / case_id / "movie3r_m15"
        primary_root = destination / case_id / "movie3r_v16_harmony_safe"
        write_payload(baseline_root, backgrounds, baseline_geometry, faces)
        write_payload(primary_root, backgrounds, primary_geometry, faces)
        rows.append({
            "case_id": case_id,
            "categories": row["categories"],
            "runtime": str(Path(row["runtime"]).resolve()),
            "cache": str(row["cache"]),
            "seconds": time.perf_counter() - started,
            "baseline_payload": str(baseline_root),
            "primary_payload": str(primary_root),
            "baseline_diagnostics": baseline_debug,
            "primary_diagnostics": primary_debug,
        })
        del pre_views, post_views, shadow_views, raw_views
        del shadow_prediction, shadow_returned, shadow_debug
        del raw_prediction, raw_returned, raw_debug, backgrounds
        del baseline_geometry, primary_geometry
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    del model
    release()
    manifest = {
        "schema_version": "Movie3R-v16-Harmony4D-qualitative-v1",
        "format": "standard demo.py --save compatible",
        "selection": str(args.selection.resolve()),
        "candidate": candidate.__dict__,
        "frame_layout": {
            "pre": int(args.pre_display),
            "post": int(args.post_display),
            "cut_index": int(args.pre_display),
        },
        "checkpoint": str(checkpoint),
        "checkpoint_flags": flags,
        "device": str(device),
        "cases": rows,
        "contract": {
            "geometry_source": "immutable formal train09 cache plus frozen v16 causal transform",
            "background_source": "frozen-checkpoint replay for visualization only",
            "gt_used": False,
            "future_frames_used_for_geometry": 0,
            "metrics_or_gate_modified": False,
        },
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(destination), "cases": len(rows), "payloads": 2 * len(rows)}, indent=2))


if __name__ == "__main__":
    main()
