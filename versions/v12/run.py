#!/usr/bin/env python3
"""Run Movie3R-Single V12 Lite/Full on an arbitrary image cut stream."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torchvision.models.detection import (
    KeypointRCNN_ResNet50_FPN_Weights,
    keypointrcnn_resnet50_fpn,
)


ROOT = Path(__file__).resolve().parents[2]
for path in (
    ROOT,
    ROOT / "src",
    ROOT / "scripts",
    ROOT / "scripts/archive_v2_v6",
    ROOT / "scripts/archive_v7",
    ROOT / "archive/20260721/scripts",
    ROOT.parent / "Movie3R-dataset/Depth-Anything-3/src",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from torch_cache_support import configure_torch_cache  # noqa: E402

configure_torch_cache()

from demo import parse_seq_path, prepare_input  # noqa: E402
from dust3r.inference import inference_recurrent_lighter  # noqa: E402
from scripts.boundary_human3r_reset_support import predicted_human  # noqa: E402
from versions.v12.experiments.v14_1_shot_aware_state_routing_probe import (  # noqa: E402
    build_model,
    build_smpl_models,
)
from versions.v12.experiments.v14_3_human_continuity_visualization import load_frame  # noqa: E402
from versions.v12.experiments.v14_5_true_recurrent_multicut_audit import (  # noqa: E402
    conditional_rotation,
    fixed_boundary,
    save_predictions,
    scale_pose,
    shot_metric_scale,
)
from v10_latent_activation_patching_probe import camera_matrix  # noqa: E402
from v18_da3_metric_depth_probe import DepthAnything3  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--cuts", type=int, nargs="+", required=True)
    parser.add_argument("--model_path", type=Path, default=ROOT / "src/human3r_896L.pth")
    parser.add_argument(
        "--da3_model_path",
        type=Path,
        default=ROOT.parent / "Movie3R-dataset/Depth-Anything-3/checkpoints/DA3Metric-Large",
    )
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--process_res", type=int, default=504)
    parser.add_argument("--point_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def build_states(
    poses: list[np.ndarray],
    humans: list[dict | None],
    cuts: list[int],
    rotations: dict[int, np.ndarray],
    scales: dict[int, float],
) -> dict:
    cut_set = set(cuts)
    current_scale = float(scales[0])
    current_gauge = np.eye(4, dtype=np.float64)
    states = {
        "0": {
            "scale": current_scale,
            "gauge": current_gauge.astype(float).tolist(),
        }
    }
    previous_root = None

    for index, pose in enumerate(poses):
        if index in cut_set:
            if previous_root is None:
                raise RuntimeError(f"Missing pre-cut human anchor at frame {index - 1}")
            human = humans[index]
            if human is None:
                raise RuntimeError(f"Missing post-cut human at frame {index}")
            current_scale = float(scales[index])
            camera_rotation = current_gauge[:3, :3] @ rotations[index] @ pose[:3, :3]
            raw_root = np.asarray(human["root"], dtype=np.float64)
            calibrated_root = raw_root * current_scale
            camera_pose = np.eye(4, dtype=np.float64)
            camera_pose[:3, :3] = camera_rotation
            camera_pose[:3, 3] = previous_root - camera_rotation @ calibrated_root
            current_gauge = camera_pose @ np.linalg.inv(scale_pose(pose, current_scale))
            states[str(index)] = {
                "scale": current_scale,
                "gauge": current_gauge.astype(float).tolist(),
            }

        camera = current_gauge @ scale_pose(pose, current_scale)
        human = humans[index]
        if human is not None:
            root_camera = np.asarray(human["root"], dtype=np.float64) * current_scale
            previous_root = camera[:3, :3] @ root_camera + camera[:3, 3]
        elif previous_root is None:
            previous_root = camera[:3, 3].copy()

    return states


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("Movie3R-Single V12 inference requires CUDA")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    img_paths, tmp_dir = parse_seq_path(str(args.seq_path))
    if tmp_dir is not None:
        raise ValueError("Use an image directory so frame/cut indexing remains explicit")
    cuts = sorted(set(int(value) for value in args.cuts))
    if not img_paths:
        raise FileNotFoundError(args.seq_path)
    if cuts[0] <= 0 or cuts[-1] >= len(img_paths):
        raise ValueError(f"Cuts must lie inside 1..{len(img_paths) - 1}: {cuts}")

    print(f">> Loading frozen Human3R on {device}", flush=True)
    model = build_model(args)
    _, pred_layer = build_smpl_models(model, device)
    views = prepare_input(
        img_paths=img_paths,
        img_mask=[True] * len(img_paths),
        size=int(args.size),
        revisit=1,
        update=True,
        img_res=getattr(model, "mhmr_img_res", None),
        reset_interval=10_000_000,
    )
    for view in views:
        view["reset"] = torch.zeros_like(view["reset"], dtype=torch.bool)
    for cut in cuts:
        # Human3R consumes reset after the previous frame and starts the cut
        # frame in a fresh recurrent state before decoding it.
        views[cut - 1]["reset"] = torch.ones_like(views[cut - 1]["reset"], dtype=torch.bool)

    started = time.perf_counter()
    with torch.no_grad():
        outputs, _ = inference_recurrent_lighter(
            views, model, str(device), use_ttt3r=False
        )
    human3r_seconds = time.perf_counter() - started
    predictions = outputs["pred"]
    local_dir = args.output_dir / "human3r_true_reset"
    save_predictions(outputs, model, local_dir, bool(args.overwrite))

    # The directory-input path has no calibration loader. Use the exact focal
    # estimated from each Human3R pointmap and saved by prepare_output.
    frames = [load_frame(local_dir, index) for index in range(len(predictions))]
    for view, frame in zip(views, frames):
        view["camera_intrinsics"] = torch.from_numpy(frame["K"]).unsqueeze(0)
    poses = [camera_matrix(row) for row in predictions]
    humans = [
        predicted_human(row, view["camera_intrinsics"], pred_layer)
        for row, view in zip(predictions, views)
    ]

    fixed, rotations, rotation_diagnostics = {}, {}, {}
    shot_start = 0
    for cut in cuts:
        fixed[cut] = fixed_boundary(
            local_dir,
            cut,
            shot_start,
            args,
            int(args.seed) + cut,
        )
        rotations[cut], rotation_diagnostics[cut] = conditional_rotation(
            fixed[cut],
            predictions[max(shot_start, cut - 5) : cut],
            views[max(shot_start, cut - 5) : cut],
            predictions[cut],
            views[cut],
            pred_layer,
            None,
            argparse.Namespace(enable_vggt=False),
        )
        shot_start = cut

    lite_scales = {start: 1.0 for start in [0, *cuts]}
    lite_states = build_states(poses, humans, cuts, rotations, lite_scales)

    print(f">> Loading Full-only Keypoint R-CNN and DA3 on {device}", flush=True)
    keypoint_model = keypointrcnn_resnet50_fpn(
        weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    ).to(device).eval()
    da3_model = DepthAnything3.from_pretrained(str(args.da3_model_path)).to(device).eval()
    full_scale_rows = {}
    scale_started = time.perf_counter()
    for start in [0, *cuts]:
        print(f">> Full shared-scale cue at frame {start}", flush=True)
        full_scale_rows[start] = shot_metric_scale(
            start,
            predictions[start],
            views[start],
            humans[start],
            local_dir,
            keypoint_model,
            da3_model,
            device,
            args,
        )
    scale_seconds = time.perf_counter() - scale_started
    full_scales = {
        start: float(row["scene_scale"]) for start, row in full_scale_rows.items()
    }
    full_states = build_states(poses, humans, cuts, rotations, full_scales)

    report = {
        "method": "Movie3R-Single V12 Lite/Full comparison",
        "legacy_method": "V14.7 custom multi-cut Lite/Full comparison",
        "input": str(args.seq_path.resolve()),
        "frames": len(img_paths),
        "cuts": cuts,
        "cut_semantics": "first frame of each new shot",
        "gpu": str(device),
        "vggt_enabled": False,
        "intrinsics": "Human3R pointmap focal estimate (demo directory-input convention)",
        "human_counts": [
            int(row.get("smpl_shape", torch.empty(1, 0, 10)).shape[1])
            for row in predictions
        ],
        "alignment_anchor_policy": (
            "Human3R person index 0; no cross-shot multi-person Re-ID or consensus"
        ),
        "timing_seconds": {
            "human3r_shared": human3r_seconds,
            "full_da3_keypoint_total": scale_seconds,
        },
        "lite": {
            "description": "Hard reset + Fixed Explicit + V16 + explicit root anchor; s=1; no DA3/Keypoint/VGGT",
            "shot_state": lite_states,
        },
        "full": {
            "description": "Lite + DA3/2D-keypoint V11.4 shared shot scale; VGGT off",
            "shot_state": full_states,
            "scale_cues": {
                str(start): {key: json_value(value) for key, value in row.items()}
                for start, row in full_scale_rows.items()
            },
        },
        "fixed": {str(key): value.astype(float).tolist() for key, value in fixed.items()},
        "rotation": {
            str(key): {
                "matrix": rotations[key].astype(float).tolist(),
                "diagnostics": rotation_diagnostics[key],
            }
            for key in cuts
        },
    }
    report_path = args.output_dir / "v12_result.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    print(f">> Report: {report_path}", flush=True)
    print(f">> Human3R: {human3r_seconds:.2f}s", flush=True)
    print(f">> Full DA3+Keypoint cues: {scale_seconds:.2f}s", flush=True)


if __name__ == "__main__":
    main()
