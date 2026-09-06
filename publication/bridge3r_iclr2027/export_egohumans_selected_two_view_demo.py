#!/usr/bin/env python3
"""Export the selected EgoHumans 0-to-177 degree six-frame demo payload.

The frozen BRIDGE3R model is causally replayed on every intervening RGB frame.
Only six publication-selected frames are exported: three from cam03 before the
cut and three from cam04 after it. Human, camera, and identity geometry comes
from the immutable formal 100-frame result for the same case.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch


MOVIE_ROOT = Path(__file__).resolve().parents[2]
for item in (MOVIE_ROOT, MOVIE_ROOT / "src", MOVIE_ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from dust3r.model import ARCroco3DStereo  # noqa: E402
from publication.bridge3r_iclr2027.restore_demo_pointcloud_backgrounds import (  # noqa: E402
    CASES,
    background_from_prediction,
    configure_model,
    formal_geometry,
    gt_helpers,
    input_paths,
    load_faces,
    read_json,
    run_forward,
    set_event_indices,
    write_payload,
)


DISPLAY_INDICES = [0, 10, 20, 60, 70, 80]
PRE_INDICES = DISPLAY_INDICES[:3]
POST_GLOBAL_INDICES = DISPLAY_INDICES[3:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            MOVIE_ROOT
            / "output/bridge3r_two_dataset_demo_v2/egohumans_selected_0_177"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = CASES["egohumans"]
    runtime = read_json(Path(spec["runtime"]))
    boundary = int(runtime["record"]["boundary_index"])
    if boundary != 50:
        raise ValueError(f"Unexpected boundary: {boundary}")

    # The latest displayed global frame is 80, i.e. post-cut local frame 30.
    pre, post = input_paths(
        "egohumans",
        spec,
        runtime,
        post_count=max(POST_GLOBAL_INDICES) - boundary + 1,
        output=args.output,
    )
    if len(pre) != boundary or len(post) != 31:
        raise ValueError((len(pre), len(post)))

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    checkpoint = Path(runtime["checkpoint"]["current"])
    started = time.perf_counter()
    model = ARCroco3DStereo.from_pretrained(str(checkpoint)).to(device)
    flags = configure_model(model)
    model.eval()

    options = SimpleNamespace(size=int(args.size))
    pre_views = gt_helpers.prepare_full_square_input(model, pre, options)
    post_views = gt_helpers.prepare_full_square_input(model, post, options)

    # Replay the causal pre-cut state plus the boundary event, and independently
    # replay the clean post-cut stream used by the frozen read-reset-register
    # transaction. No future frame changes any earlier prediction.
    shadow_views = set_event_indices(
        copy.deepcopy(pre_views + post_views[:1]), {boundary}
    )
    post_reset_views = set_event_indices(copy.deepcopy(post_views), set())
    shadow_pred, shadow_returned, shadow_debug, shadow_timing = run_forward(
        model, shadow_views, device, "bridge3r_selected_shadow_replay"
    )
    post_pred, post_returned, post_debug, post_timing = run_forward(
        model, post_reset_views, device, "bridge3r_selected_post_reset_replay"
    )

    backgrounds = [
        background_from_prediction(shadow_pred[index], shadow_returned[index])
        for index in PRE_INDICES
    ]
    backgrounds.extend(
        background_from_prediction(
            post_pred[index - boundary], post_returned[index - boundary]
        )
        for index in POST_GLOBAL_INDICES
    )
    geometry = formal_geometry(
        "bridge3r", Path(spec["cache"]), runtime, DISPLAY_INDICES
    )
    payload = args.output.resolve() / "payloads/bridge3r"
    write_payload(
        payload,
        backgrounds,
        geometry,
        load_faces(),
        {
            "schema_version": "Bridge3R-EgoHumans-selected-two-view-demo-v1",
            "dataset": "egohumans",
            "case_id": spec["case_id"],
            "method": "bridge3r",
            "view_sequence": ["cam03", "cam03", "cam03", "cam04", "cam04", "cam04"],
            "relative_view_degrees": [0.0, 0.0, 0.0, 176.74916252624004,
                                      176.74916252624004, 176.74916252624004],
            "formal_clip_indices": DISPLAY_INDICES,
            "source_frames": [251, 261, 271, 311, 321, 331],
            "relative_seconds": [0.0, 0.5, 1.0, 3.0, 3.5, 4.0],
            "cut_index_in_payload": 3,
            "scene_channel_available": True,
            "scene_source": "the method's own frozen checkpoint causal replay",
            "human_camera_geometry_source": "immutable formal 100-frame test cache",
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_flags": flags,
            "gt_used": False,
            "intervening_frames_processed": True,
            "future_frames_used": 0,
            "external_baseline_geometry_borrowed": False,
            "timing": {"shadow": shadow_timing, "post_reset": post_timing},
        },
    )

    del model, pre_views, post_views, shadow_views, post_reset_views
    del shadow_pred, shadow_returned, shadow_debug, post_pred, post_returned, post_debug
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "payload": str(payload),
                "display_frames": len(DISPLAY_INDICES),
                "formal_clip_indices": DISPLAY_INDICES,
                "source_frames": [251, 261, 271, 311, 321, 331],
                "seconds_total": time.perf_counter() - started,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
