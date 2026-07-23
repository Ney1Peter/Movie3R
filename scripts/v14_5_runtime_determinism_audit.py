#!/usr/bin/env python3
"""Benchmark the frozen V11.4 deploy path and audit three-run determinism."""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from torchvision.models.detection import (
    KeypointRCNN_ResNet50_FPN_Weights,
    keypointrcnn_resnet50_fpn,
)


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from torch_cache_support import configure_torch_cache  # noqa: E402

configure_torch_cache()

from scripts.v14_1_shot_aware_state_routing_probe import (  # noqa: E402
    build_model,
    read_jsonl,
)
from v11_gauge_neutral_first_write_probe import (  # noqa: E402
    build_dataset,
    configure_views,
    record_spec,
)
from v12_build_gauge_neutral_teacher_cache import old_a_dataset  # noqa: E402
from v15_wide_baseline_boundary_bridge_candidates import aggregate_coarse, build_vggt  # noqa: E402
from v18_cache_2d_keypoints import select_person  # noqa: E402
from v18_da3_metric_depth_probe import (  # noqa: E402
    DepthAnything3,
    estimate_frame_roots,
    metric_inference,
)
from v19_da3_depth_correction_ablation import load_raw_pair  # noqa: E402
from v21_absolute_shot_background_scale_probe import (  # noqa: E402
    bounded_scene_scale,
    frame_calibration,
)
from v24_integrated_runtime_benchmark import (  # noqa: E402
    keypoint_inference,
    load_case_inputs,
    memory_snapshot,
    square_image,
    synchronize,
    timed_cuda,
    vggt_inference,
)
from v32_consensus_texture_safety_audit import selected_rotation  # noqa: E402


DEFAULT_STREAM = ROOT / "output/v18_human_metric_translation/stream_cache"
DEFAULT_V16 = ROOT / "output/v18_human_metric_translation/v16_bound20_scene"
DEFAULT_V10 = (
    ROOT
    / "output/v10_candidate_selection/oracle_gt_4source/"
    "oracle_candidate_selection_metrics.json"
)
DEFAULT_RECORDS = (
    ROOT / "output/v10_candidate_selection/oracle_gt_4source/selected_records.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream_dir", type=Path, default=DEFAULT_STREAM)
    parser.add_argument("--v16_dir", type=Path, default=DEFAULT_V16)
    parser.add_argument("--v10_report", type=Path, default=DEFAULT_V10)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--data_root", type=Path, default=Path("/data/wangzheng/iJCV-CODE/data"))
    parser.add_argument("--model_path", type=Path, default=ROOT / "src/human3r_896L.pth")
    parser.add_argument(
        "--da3_model_path",
        type=Path,
        default=ROOT.parent / "Movie3R-dataset/Depth-Anything-3/checkpoints/DA3Metric-Large",
    )
    parser.add_argument("--vggt_root", type=Path, default=Path("/data/wangzheng/Movie3R/vggt"))
    parser.add_argument(
        "--vggt_weights",
        type=Path,
        default=Path("/data/wangzheng/Movie3R/vggt/vggt_weights/model.pt"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ROOT / "output/v14_5_final_audit/runtime",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--process_res", type=int, default=504)
    parser.add_argument("--cases_per_source", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--normal_frames", type=int, default=8)
    parser.add_argument("--resolution", type=int, nargs=2, default=(512, 288))
    parser.add_argument("--resize_mode", default="human3r_demo")
    parser.add_argument("--warmup_frames", type=int, default=8)
    parser.add_argument("--max_post_frames", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args()


def load_shards(root: Path, pattern: str) -> dict[str, dict]:
    rows = []
    for path in sorted(glob.glob(str(root / pattern))):
        rows.extend(json.loads(Path(path).read_text(encoding="utf-8"))["cases"])
    return {str(row["case_name"]): row for row in rows}


def distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
    }


def scale_pose(pose: np.ndarray, scale: float) -> np.ndarray:
    output = np.asarray(pose, dtype=np.float64).copy()
    output[:3, 3] *= float(scale)
    return output


def texture_score(image: np.ndarray) -> float:
    tensor = torch.from_numpy(np.asarray(image).copy()).float().div(127.5).sub(1.0)
    gray = tensor.mean(dim=2)
    return float(
        (gray[:, 1:] - gray[:, :-1]).abs().mean()
        + (gray[1:] - gray[:-1]).abs().mean()
    )


def selected_cases(streams: dict[str, dict], count: int) -> list[dict]:
    rows = []
    for source in sorted({row["source"] for row in streams.values()}):
        group = sorted(
            [row for row in streams.values() if row["source"] == source],
            key=lambda row: str(row["case_name"]),
        )
        indices = np.linspace(0, len(group) - 1, min(count, len(group)), dtype=np.int64)
        for index in indices:
            row = dict(group[int(index)])
            cache_path = Path(row["cache_path"])
            if not cache_path.is_absolute():
                row["cache_path"] = str((ROOT / cache_path).resolve())
            rows.append(row)
    return rows


def final_candidate(
    case: dict,
    v10: dict,
    v16: dict,
    keypoint_model,
    da3_model,
    vggt_model,
    device: torch.device,
    args,
) -> tuple[dict, dict]:
    images, intrinsics, poses, joints, raw = load_case_inputs(case, v10)
    detections, keypoint_seconds = timed_cuda(
        lambda: keypoint_inference(keypoint_model, images, device), device
    )
    depths, processed, da3_seconds = [], [], []
    for index in range(2):
        output, elapsed = timed_cuda(
            lambda index=index: metric_inference(
                da3_model,
                [images[index]],
                intrinsics[index : index + 1],
                int(args.process_res),
            ),
            device,
        )
        depth, processed_intrinsics, _ = output
        depths.append(depth[0])
        processed.append(processed_intrinsics[0])
        da3_seconds.append(elapsed)

    post_started = time.perf_counter()
    root_scales, scene_scales = [], []
    kernel = np.ones((11, 11), dtype=np.uint8)
    for index in range(2):
        keypoints, confidence, _, _ = detections[index]
        _, calibrated_root, _ = estimate_frame_roots(
            depths[index],
            intrinsics[index],
            processed[index],
            keypoints,
            confidence,
            joints[index],
            0.30,
            3,
        )
        root_scale = float(
            np.clip(float(calibrated_root[2]) / max(float(joints[index, 0, 2]), 1e-4), 0.35, 3.0)
        )
        calibration = frame_calibration(
            raw["depth"][index],
            depths[index],
            raw["confidence"][index],
            cv2.dilate(raw["mask"][index].astype(np.uint8), kernel, iterations=1),
            root_scale,
            SimpleNamespace(
                raw_confidence_threshold=1.0,
                min_background_pixels=512,
                lowfreq_sigma=25.0,
            ),
        )
        median_ratio = float(calibration["scales"]["median_ratio"])
        scene_scale = (
            bounded_scene_scale(median_ratio, root_scale, 0.15)
            if median_ratio / max(root_scale, 1e-6) < 0.95
            else root_scale
        )
        root_scales.append(root_scale)
        scene_scales.append(float(scene_scale))

    name = str(case["case_name"])
    fixed = np.asarray(v10[name]["fixed_explicit"]["transform"], dtype=np.float32)
    torso = np.asarray(
        v16[name]["fixed_candidates"]["fixed_torso_motion_1f_resolve_t"]["transform"],
        dtype=np.float32,
    )
    relative = torso[:3, :3] @ fixed[:3, :3].T
    torso_residual = float(
        np.degrees(
            np.arccos(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
        )
    )
    trigger = bool(torso_residual >= 10.0)
    vggt_seconds = 0.0
    branch = "torso"
    rotation = torso[:3, :3]
    if trigger:
        vggt_output, vggt_seconds = timed_cuda(
            lambda: vggt_inference(vggt_model, images, args), device
        )
        pairs, _ = vggt_output
        coarse, consensus = aggregate_coarse(pairs, [poses[0]], [poses[1]])
        wide = {
            "texture_score": texture_score(images[1]),
            "windows": {
                "full_rgb_1p1": {
                    "rotation_consensus": consensus,
                    "candidates": {"coarse": {"transform": coarse.tolist()}},
                }
            },
        }
        rotation, branch, _ = selected_rotation(
            fixed[:3, :3], torso[:3, :3], wide, 0.05, consensus_cap_deg=60.0
        )

    old_pose = scale_pose(poses[0], scene_scales[0])
    old_root = joints[0, 0] * scene_scales[0]
    new_root = joints[1, 0] * scene_scales[1]
    old_anchor = old_pose[:3, :3] @ old_root + old_pose[:3, 3]
    camera_rotation = rotation @ poses[1, :3, :3]
    camera_pose = np.eye(4, dtype=np.float64)
    camera_pose[:3, :3] = camera_rotation
    camera_pose[:3, 3] = old_anchor - camera_rotation @ new_root
    boundary = camera_pose @ np.linalg.inv(scale_pose(poses[1], scene_scales[1]))
    post_seconds = time.perf_counter() - post_started
    total = keypoint_seconds + sum(da3_seconds) + vggt_seconds + post_seconds
    return (
        {
            "case_name": name,
            "source": case["source"],
            "trigger": trigger,
            "accepted_branch": branch,
            "old_scale": scene_scales[0],
            "new_scale": scene_scales[1],
            "root_scales": root_scales,
            "boundary": boundary.astype(float).tolist(),
        },
        {
            "keypoint_seconds": keypoint_seconds,
            "da3_seconds": float(sum(da3_seconds)),
            "vggt_seconds": float(vggt_seconds),
            "postprocess_seconds": post_seconds,
            "cut_total_seconds": total,
        },
    )


def normal_frame_benchmark(human3r, args, device: torch.device) -> dict:
    record = read_jsonl(args.records)[0]
    spec = record_spec(record, args)
    views = configure_views(
        next(
            iter(
                torch.utils.data.DataLoader(
                    old_a_dataset(spec, args), batch_size=1, num_workers=0
                )
            )
        ),
        device,
        human3r.mhmr_img_res,
    )[: int(args.normal_frames)]
    for view in views:
        view["reset"] = torch.zeros_like(view["reset"], dtype=torch.bool)
    with torch.no_grad():
        human3r.forward_recurrent_lighter(views, str(device), ret_state=False, use_ttt3r=False)
    synchronize(device)
    elapsed = []
    for _ in range(int(args.repeats)):
        synchronize(device)
        started = time.perf_counter()
        with torch.no_grad():
            human3r.forward_recurrent_lighter(
                views, str(device), ret_state=False, use_ttt3r=False
            )
        synchronize(device)
        elapsed.append(time.perf_counter() - started)
    return {
        "frame_count": len(views),
        "elapsed_seconds": elapsed,
        "fps": [len(views) / value for value in elapsed],
        "fps_distribution": distribution([len(views) / value for value in elapsed]),
    }


def maximum_repeat_difference(rows: list[dict]) -> dict:
    by_case = {}
    for row in rows:
        by_case.setdefault(row["case_name"], []).append(row)
    maximum = {"scale": 0.0, "boundary": 0.0, "trigger_change": False, "branch_change": False}
    for values in by_case.values():
        first = values[0]
        for value in values[1:]:
            maximum["scale"] = max(
                maximum["scale"],
                abs(float(first["old_scale"]) - float(value["old_scale"])),
                abs(float(first["new_scale"]) - float(value["new_scale"])),
            )
            maximum["boundary"] = max(
                maximum["boundary"],
                float(
                    np.max(
                        np.abs(
                            np.asarray(first["boundary"], dtype=np.float64)
                            - np.asarray(value["boundary"], dtype=np.float64)
                        )
                    )
                ),
            )
            maximum["trigger_change"] |= bool(first["trigger"] != value["trigger"])
            maximum["branch_change"] |= bool(
                first["accepted_branch"] != value["accepted_branch"]
            )
    return maximum


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("V14.5 runtime audit requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    args.query_rows = 20
    args.query_cols = 16
    args.pair_batch_size = 2

    streams = load_shards(args.stream_dir, "v18_stream_shard_*_of_*.json")
    v16 = load_shards(args.v16_dir, "v16_candidates_shard_*_of_*.json")
    v10 = {
        str(row["case_name"]): row
        for row in json.loads(args.v10_report.read_text(encoding="utf-8"))["cases"]
    }
    cases = selected_cases(
        {name: row for name, row in streams.items() if name in v16 and name in v10},
        int(args.cases_per_source),
    )

    torch.cuda.empty_cache()
    human3r = build_model(args)
    keypoint_model = keypointrcnn_resnet50_fpn(
        weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    ).to(device).eval()
    da3_model = DepthAnything3.from_pretrained(str(args.da3_model_path)).to(device).eval()
    vggt_model = build_vggt(args)
    synchronize(device)
    resident = memory_snapshot(device)

    # Warm every branch before timing.
    warm = cases[0]
    final_candidate(warm, v10, v16, keypoint_model, da3_model, vggt_model, device, args)
    torch.cuda.reset_peak_memory_stats(device)
    outputs, timings = [], []
    for repeat in range(int(args.repeats)):
        for index, case in enumerate(cases):
            candidate, timing = final_candidate(
                case, v10, v16, keypoint_model, da3_model, vggt_model, device, args
            )
            candidate["repeat"] = repeat
            timing.update(
                {"case_name": candidate["case_name"], "source": candidate["source"], "repeat": repeat, "trigger": candidate["trigger"]}
            )
            outputs.append(candidate)
            timings.append(timing)
            print(
                f">> runtime r{repeat + 1}/{args.repeats} {index + 1}/{len(cases)} "
                f"trigger={candidate['trigger']} total={timing['cut_total_seconds']:.3f}s",
                flush=True,
            )

    partial = args.output_dir / "v14_5_runtime_cut_runs.partial.json"
    partial.write_text(
        json.dumps(
            {"candidate_runs": outputs, "timing_runs": timings},
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    normal = normal_frame_benchmark(human3r, args, device)
    latency = [row["cut_total_seconds"] for row in timings]
    triggered = [row["cut_total_seconds"] for row in timings if row["trigger"]]
    untriggered = [row["cut_total_seconds"] for row in timings if not row["trigger"]]
    report = {
        "experiment": "V14.5 actual deployment runtime and determinism audit",
        "device": {
            "name": torch.cuda.get_device_name(device),
            "requested": str(device),
        },
        "protocol": {
            "repeats": int(args.repeats),
            "case_count_per_repeat": len(cases),
            "models_resident_together": ["Human3R", "Keypoint R-CNN", "DA3Metric-Large", "VGGT-1B"],
            "vggt_pretrigger": "torso residual >= 10 deg",
            "v11_4_scale": "median_ratio_q15_gate_lt95",
            "gt_or_scene_evaluator_in_timing": False,
            "visualization_in_timing": False,
        },
        "latency_seconds": distribution(latency),
        "triggered_latency_seconds": distribution(triggered) if triggered else None,
        "untriggered_latency_seconds": distribution(untriggered) if untriggered else None,
        "trigger_rate": float(np.mean([row["trigger"] for row in timings])),
        "normal_frame": normal,
        "gpu_memory_bytes": {
            "resident_after_all_models": resident,
            "peak_allocated": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved": int(torch.cuda.max_memory_reserved(device)),
        },
        "determinism": maximum_repeat_difference(outputs),
        "candidate_runs": outputs,
        "timing_runs": timings,
    }
    output = args.output_dir / "v14_5_runtime_determinism_audit.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f">> wrote {output}", flush=True)


if __name__ == "__main__":
    main()
