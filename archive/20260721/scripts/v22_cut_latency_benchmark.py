#!/usr/bin/env python3
"""Benchmark the cut-only latency and incremental memory of the V22 bridge."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights, keypointrcnn_resnet50_fpn


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "output" / "v22_explicit_metric_bridge" / "latency_benchmark"
DEFAULT_V10 = (
    REPO_ROOT
    / "output"
    / "v10_candidate_selection"
    / "oracle_gt_4source"
    / "oracle_candidate_selection_metrics.json"
)

import sys

for path in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from v18_cache_2d_keypoints import select_person  # noqa: E402
from v18_da3_metric_depth_probe import (  # noqa: E402
    DEFAULT_ROOT,
    DEFAULT_MODEL,
    DepthAnything3,
    estimate_frame_roots,
    load_cases,
    metric_inference,
    resolve,
)
from v19_da3_depth_correction_ablation import load_raw_pair  # noqa: E402
from v21_absolute_shot_background_scale_probe import frame_calibration  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream_dir", type=Path, default=DEFAULT_ROOT / "stream_cache")
    parser.add_argument("--keypoint_dir", type=Path, default=DEFAULT_ROOT / "keypoint_cache")
    parser.add_argument("--scene_dir", type=Path, default=DEFAULT_ROOT / "v16_bound20_scene")
    parser.add_argument(
        "--v18_report",
        type=Path,
        default=DEFAULT_ROOT / "final_candidates" / "v18_human_metric_translation_eval.json",
    )
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--v10_report", type=Path, default=DEFAULT_V10)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--process_res", type=int, default=504)
    parser.add_argument("--cases_per_source", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--keypoint_threshold", type=float, default=0.30)
    parser.add_argument("--sample_radius", type=int, default=3)
    parser.set_defaults(max_cases=0)
    return parser.parse_args()


def distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def timed_cuda(function, device: torch.device):
    synchronize(device)
    started = time.perf_counter()
    value = function()
    synchronize(device)
    return value, time.perf_counter() - started


def keypoint_inference(model: torch.nn.Module, images: list[np.ndarray], device: torch.device):
    batch = [
        torch.from_numpy(np.asarray(image).copy())
        .permute(2, 0, 1)
        .float()
        .div(255.0)
        .to(device)
        for image in images
    ]
    with torch.no_grad():
        outputs = model(batch)
    return [select_person(row, 0.50) for row in outputs]


def model_bytes(model: torch.nn.Module) -> int:
    return int(
        sum(value.numel() * value.element_size() for value in model.parameters())
        + sum(value.numel() * value.element_size() for value in model.buffers())
    )


def explicit_postprocess(
    images: list[np.ndarray],
    intrinsics: np.ndarray,
    poses: np.ndarray,
    joints: np.ndarray,
    detections: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]],
    depths: list[np.ndarray],
    processed_intrinsics: list[np.ndarray],
    raw: dict,
) -> dict:
    roots = []
    calibrations = []
    calibration_args = SimpleNamespace(
        raw_confidence_threshold=1.0,
        min_background_pixels=512,
        lowfreq_sigma=25.0,
    )
    kernel = np.ones((11, 11), dtype=np.uint8)
    for index in range(2):
        keypoints, confidence, _, _ = detections[index]
        _, torso, _ = estimate_frame_roots(
            depths[index],
            intrinsics[index],
            processed_intrinsics[index],
            keypoints,
            confidence,
            joints[index],
            0.30,
            3,
        )
        roots.append(torso)
        root_scale = float(np.clip(torso[2] / max(float(joints[index, 0, 2]), 1e-4), 0.35, 3.0))
        mask = cv2.dilate(raw["mask"][index].astype(np.uint8), kernel)
        calibrations.append(
            frame_calibration(
                raw["depth"][index],
                depths[index],
                raw["confidence"][index],
                mask,
                root_scale,
                calibration_args,
            )
        )
    old_world = poses[0, :3, :3] @ roots[0] + poses[0, :3, 3]
    rotation = poses[0, :3, :3]
    camera_translation = old_world - rotation @ roots[1]
    camera_pose = np.eye(4, dtype=np.float32)
    camera_pose[:3, :3] = rotation
    camera_pose[:3, 3] = camera_translation
    boundary = camera_pose @ np.linalg.inv(poses[1])
    return {
        "boundary": boundary,
        "scene_scales": [row["scales"]["median_ratio"] for row in calibrations],
    }


def choose_cases(cases: list[dict], count: int) -> list[dict]:
    selected = []
    for source in sorted({row["source"] for row in cases}):
        source_cases = [row for row in cases if row["source"] == source]
        ids = np.linspace(0, len(source_cases) - 1, count, dtype=np.int64)
        selected.extend(source_cases[int(index)] for index in ids)
    return selected


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("V22 latency benchmark requires CUDA")
    if not (args.model_path / "model.safetensors").exists():
        raise FileNotFoundError(args.model_path / "model.safetensors")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    cases = choose_cases(load_cases(args), int(args.cases_per_source))
    v10 = {
        row["case_name"]: row
        for row in json.loads(args.v10_report.read_text(encoding="utf-8"))["cases"]
    }

    torch.cuda.empty_cache()
    synchronize(device)
    load_started = time.perf_counter()
    keypoint_model = keypointrcnn_resnet50_fpn(
        weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    ).to(device).eval()
    keypoint_load_seconds = time.perf_counter() - load_started
    load_started = time.perf_counter()
    da3_model = DepthAnything3.from_pretrained(str(args.model_path)).to(device).eval()
    synchronize(device)
    da3_load_seconds = time.perf_counter() - load_started
    resident_allocated = int(torch.cuda.memory_allocated(device))
    resident_reserved = int(torch.cuda.memory_reserved(device))
    torch.cuda.reset_peak_memory_stats(device)

    records = []
    for case_index, case in enumerate(cases):
        with np.load(resolve(case["cache_path"])) as stream:
            images = [stream["old_images"][-1], stream["new_image"]]
            intrinsics = np.stack(
                [stream["old_intrinsics"][-1], stream["new_intrinsics"]]
            ).astype(np.float32)
            poses = np.stack([stream["old_pose"][-1], stream["new_pose"]]).astype(np.float32)
            joints = np.stack(
                [stream["old_joints_camera"][-1], stream["new_joints_camera"]]
            ).astype(np.float32)
        raw = load_raw_pair(Path(v10[case["case_name"]]["paths"]["human3r_local_reset"]))

        if case_index < int(args.warmup):
            keypoint_inference(keypoint_model, images, device)
            for index in range(2):
                metric_inference(
                    da3_model,
                    [images[index]],
                    intrinsics[index : index + 1],
                    int(args.process_res),
                )
            synchronize(device)

        detections, keypoint_seconds = timed_cuda(
            lambda: keypoint_inference(keypoint_model, images, device), device
        )
        depths = []
        processed = []
        da3_frame_seconds = []
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
            da3_frame_seconds.append(elapsed)
        post_started = time.perf_counter()
        explicit_postprocess(
            images,
            intrinsics,
            poses,
            joints,
            detections,
            depths,
            processed,
            raw,
        )
        post_seconds = time.perf_counter() - post_started
        total = keypoint_seconds + sum(da3_frame_seconds) + post_seconds
        records.append(
            {
                "case_name": case["case_name"],
                "source": case["source"],
                "keypoint_seconds_2frames_batch": keypoint_seconds,
                "da3_old_seconds": da3_frame_seconds[0],
                "da3_new_seconds": da3_frame_seconds[1],
                "da3_seconds_2frames_independent": sum(da3_frame_seconds),
                "explicit_postprocess_seconds": post_seconds,
                "cut_total_seconds": total,
            }
        )
        print(
            f">> [{case_index + 1}/{len(cases)}] {case['source']} "
            f"kp={keypoint_seconds:.3f}s da3={sum(da3_frame_seconds):.3f}s "
            f"post={post_seconds:.3f}s total={total:.3f}s",
            flush=True,
        )

    report = {
        "experiment": "V22 cut-only latency and incremental GPU-memory benchmark",
        "device": {
            "requested": str(device),
            "name": torch.cuda.get_device_name(device),
            "total_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory),
        },
        "protocol": {
            "case_count": len(records),
            "cases_per_source": int(args.cases_per_source),
            "process_res": int(args.process_res),
            "old_and_new_da3_are_independent_calls": True,
            "keypoint_frames_are_one_batch": True,
            "ordinary_frame_path_changed": False,
            "memory_scope": "incremental Keypoint R-CNN + DA3 only; Human3R excluded",
        },
        "model_load_seconds": {
            "keypoint_rcnn": keypoint_load_seconds,
            "da3_metric_large": da3_load_seconds,
        },
        "model_parameter_bytes": {
            "keypoint_rcnn": model_bytes(keypoint_model),
            "da3_metric_large": model_bytes(da3_model),
        },
        "gpu_memory_bytes": {
            "resident_allocated_after_model_load": resident_allocated,
            "resident_reserved_after_model_load": resident_reserved,
            "peak_allocated_during_benchmark": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_during_benchmark": int(torch.cuda.max_memory_reserved(device)),
        },
        "latency_seconds": {
            key: distribution([row[key] for row in records])
            for key in (
                "keypoint_seconds_2frames_batch",
                "da3_old_seconds",
                "da3_new_seconds",
                "da3_seconds_2frames_independent",
                "explicit_postprocess_seconds",
                "cut_total_seconds",
            )
        },
        "cases": records,
    }
    output = args.output_dir / "v22_cut_latency_benchmark.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f">> wrote {output}")


if __name__ == "__main__":
    main()
