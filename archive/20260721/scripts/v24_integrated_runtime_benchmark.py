#!/usr/bin/env python3
"""Benchmark V24 with Keypoint R-CNN, DA3, and VGGT resident together."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.detection import (
    KeypointRCNN_ResNet50_FPN_Weights,
    keypointrcnn_resnet50_fpn,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from v15_wide_baseline_boundary_bridge_candidates import (  # noqa: E402
    DEFAULT_VGGT_ROOT,
    DEFAULT_VGGT_WEIGHTS,
    build_vggt,
    pair_specs,
    run_vggt_pairs,
)
from v18_cache_2d_keypoints import select_person  # noqa: E402
from v18_da3_metric_depth_probe import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_ROOT,
    DepthAnything3,
    estimate_frame_roots,
    load_cases,
    metric_inference,
    resolve,
)
from v19_da3_depth_correction_ablation import load_raw_pair  # noqa: E402
from v21_absolute_shot_background_scale_probe import frame_calibration  # noqa: E402


DEFAULT_V10 = (
    REPO_ROOT
    / "output"
    / "v10_candidate_selection"
    / "oracle_gt_4source"
    / "oracle_candidate_selection_metrics.json"
)
DEFAULT_V24_EXPORT = (
    REPO_ROOT
    / "output"
    / "v24_vggt_v22_rotation_bridge"
    / "selected_candidates"
    / "v24_selected_rotation_bridge.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "output" / "v24_vggt_v22_rotation_bridge" / "runtime_benchmark"
)


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
    parser.add_argument("--v24_export", type=Path, default=DEFAULT_V24_EXPORT)
    parser.add_argument("--vggt_root", type=Path, default=DEFAULT_VGGT_ROOT)
    parser.add_argument("--vggt_weights", type=Path, default=DEFAULT_VGGT_WEIGHTS)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--process_res", type=int, default=504)
    parser.add_argument("--query_rows", type=int, default=20)
    parser.add_argument("--query_cols", type=int, default=16)
    parser.add_argument("--pair_batch_size", type=int, default=2)
    parser.add_argument("--cases_per_trigger_group", type=int, default=1)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keypoint_threshold", type=float, default=0.30)
    parser.add_argument("--sample_radius", type=int, default=3)
    parser.set_defaults(max_cases=0)
    return parser.parse_args()


def distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {key: float("nan") for key in ("mean", "median", "p90", "p95", "min", "max")}
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


def model_bytes(model: torch.nn.Module) -> int:
    return int(
        sum(value.numel() * value.element_size() for value in model.parameters())
        + sum(value.numel() * value.element_size() for value in model.buffers())
    )


def memory_snapshot(device: torch.device) -> dict:
    free, total = torch.cuda.mem_get_info(device)
    return {
        "allocated": int(torch.cuda.memory_allocated(device)),
        "reserved": int(torch.cuda.memory_reserved(device)),
        "device_free": int(free),
        "device_total": int(total),
    }


def keypoint_inference(
    model: torch.nn.Module, images: list[np.ndarray], device: torch.device
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
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


def square_image(
    image: np.ndarray, query_rows: int, query_cols: int
) -> dict:
    tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().div(255.0)
    height, width = int(tensor.shape[-2]), int(tensor.shape[-1])
    scale = 518.0 / max(height, width)
    resized_height = min(518, max(14, int(round(height * scale / 14.0)) * 14))
    resized_width = min(518, max(14, int(round(width * scale / 14.0)) * 14))
    tensor = F.interpolate(
        tensor[None],
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    )[0]
    pad_top = (518 - resized_height) // 2
    pad_left = (518 - resized_width) // 2
    tensor = F.pad(
        tensor,
        (
            pad_left,
            518 - resized_width - pad_left,
            pad_top,
            518 - resized_height - pad_top,
        ),
        value=1.0,
    )
    margin_y = min(12.0, max(height * 0.05, 2.0))
    margin_x = min(12.0, max(width * 0.05, 2.0))
    yy = np.linspace(margin_y, height - 1.0 - margin_y, query_rows)
    xx = np.linspace(margin_x, width - 1.0 - margin_x, query_cols)
    grid_x, grid_y = np.meshgrid(xx, yy)
    original_query = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1).astype(
        np.float32
    )
    square_query = np.stack(
        [
            original_query[:, 0] * resized_width / width + pad_left,
            original_query[:, 1] * resized_height / height + pad_top,
        ],
        axis=1,
    ).astype(np.float32)
    return {
        "image": tensor,
        "height": height,
        "width": width,
        "resized_height": resized_height,
        "resized_width": resized_width,
        "pad_top": pad_top,
        "pad_left": pad_left,
        "original_query": original_query,
        "square_query": square_query,
        "human_ratio": 0.0,
    }


def vggt_inference(model: torch.nn.Module, images: list[np.ndarray], args: argparse.Namespace):
    old_meta = [square_image(images[0], int(args.query_rows), int(args.query_cols))]
    new_meta = [square_image(images[1], int(args.query_rows), int(args.query_cols))]
    return run_vggt_pairs(model, old_meta, new_meta, pair_specs(1, 1), args)


def explicit_postprocess(
    intrinsics: np.ndarray,
    poses: np.ndarray,
    joints: np.ndarray,
    detections: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]],
    depths: list[np.ndarray],
    processed_intrinsics: list[np.ndarray],
    raw: dict,
) -> None:
    roots = []
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
        root_scale = float(
            np.clip(torso[2] / max(float(joints[index, 0, 2]), 1e-4), 0.35, 3.0)
        )
        mask = cv2.dilate(raw["mask"][index].astype(np.uint8), kernel)
        frame_calibration(
            raw["depth"][index],
            depths[index],
            raw["confidence"][index],
            mask,
            root_scale,
            calibration_args,
        )
    old_world = poses[0, :3, :3] @ roots[0] + poses[0, :3, 3]
    rotation = poses[0, :3, :3]
    camera_translation = old_world - rotation @ roots[1]
    camera_pose = np.eye(4, dtype=np.float32)
    camera_pose[:3, :3] = rotation
    camera_pose[:3, 3] = camera_translation
    camera_pose @ np.linalg.inv(poses[1])


def select_cases(cases: list[dict], export: dict, count: int) -> list[dict]:
    required = {row["case_name"]: bool(row["vggt_required"]) for row in export["cases"]}
    selected = []
    for source in sorted({row["source"] for row in cases}):
        for trigger in (False, True):
            group = [
                row
                for row in cases
                if row["source"] == source and required[row["case_name"]] == trigger
            ]
            if not group:
                continue
            indices = np.linspace(0, len(group) - 1, min(count, len(group)), dtype=np.int64)
            for index in indices:
                row = dict(group[int(index)])
                row["vggt_required"] = trigger
                selected.append(row)
    return selected


def load_case_inputs(case: dict, v10: dict[str, dict]) -> tuple:
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
    return images, intrinsics, poses, joints, raw


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("V24 integrated benchmark requires CUDA")
    if not (args.model_path / "model.safetensors").exists():
        raise FileNotFoundError(args.model_path / "model.safetensors")
    if not args.vggt_weights.exists():
        raise FileNotFoundError(args.vggt_weights)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    export = json.loads(args.v24_export.read_text(encoding="utf-8"))
    cases = select_cases(
        load_cases(args), export, int(args.cases_per_trigger_group)
    )
    v10 = {
        row["case_name"]: row
        for row in json.loads(args.v10_report.read_text(encoding="utf-8"))["cases"]
    }

    torch.cuda.empty_cache()
    synchronize(device)
    memory = {"before_model_load": memory_snapshot(device)}
    load_seconds = {}

    started = time.perf_counter()
    keypoint_model = keypointrcnn_resnet50_fpn(
        weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    ).to(device).eval()
    synchronize(device)
    load_seconds["keypoint_rcnn"] = time.perf_counter() - started
    memory["after_keypoint_load"] = memory_snapshot(device)

    started = time.perf_counter()
    da3_model = DepthAnything3.from_pretrained(str(args.model_path)).to(device).eval()
    synchronize(device)
    load_seconds["da3_metric_large"] = time.perf_counter() - started
    memory["after_da3_load"] = memory_snapshot(device)

    started = time.perf_counter()
    vggt_model = build_vggt(args)
    synchronize(device)
    load_seconds["vggt_1b_camera_track"] = time.perf_counter() - started
    memory["after_vggt_load"] = memory_snapshot(device)

    if args.warmup:
        warm_case = next(row for row in cases if row["vggt_required"])
        images, intrinsics, _, _, _ = load_case_inputs(warm_case, v10)
        keypoint_inference(keypoint_model, images, device)
        for index in range(2):
            metric_inference(
                da3_model,
                [images[index]],
                intrinsics[index : index + 1],
                int(args.process_res),
            )
        vggt_inference(vggt_model, images, args)
        synchronize(device)

    torch.cuda.reset_peak_memory_stats(device)
    records = []
    for case_index, case in enumerate(cases):
        images, intrinsics, poses, joints, raw = load_case_inputs(case, v10)
        detections, keypoint_seconds = timed_cuda(
            lambda: keypoint_inference(keypoint_model, images, device), device
        )
        depths = []
        processed = []
        da3_seconds = []
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

        started = time.perf_counter()
        explicit_postprocess(
            intrinsics,
            poses,
            joints,
            detections,
            depths,
            processed,
            raw,
        )
        explicit_seconds = time.perf_counter() - started

        vggt_seconds = 0.0
        if case["vggt_required"]:
            _, vggt_seconds = timed_cuda(
                lambda: vggt_inference(vggt_model, images, args), device
            )
        total_seconds = (
            keypoint_seconds + sum(da3_seconds) + explicit_seconds + vggt_seconds
        )
        records.append(
            {
                "case_name": case["case_name"],
                "source": case["source"],
                "vggt_required": case["vggt_required"],
                "keypoint_seconds_2frames_batch": keypoint_seconds,
                "da3_seconds_2frames_independent": sum(da3_seconds),
                "explicit_postprocess_seconds": explicit_seconds,
                "vggt_seconds_bidirectional_1p1": vggt_seconds,
                "cut_total_seconds": total_seconds,
            }
        )
        print(
            f">> [{case_index + 1}/{len(cases)}] {case['source']} "
            f"trigger={case['vggt_required']} kp={keypoint_seconds:.3f}s "
            f"da3={sum(da3_seconds):.3f}s vggt={vggt_seconds:.3f}s "
            f"total={total_seconds:.3f}s",
            flush=True,
        )

    memory["after_benchmark"] = memory_snapshot(device)
    memory["peak_during_benchmark"] = {
        "allocated": int(torch.cuda.max_memory_allocated(device)),
        "reserved": int(torch.cuda.max_memory_reserved(device)),
    }
    report = {
        "experiment": "V24 integrated cut-only runtime and GPU-memory benchmark",
        "device": {
            "requested": str(device),
            "name": torch.cuda.get_device_name(device),
            "total_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory),
        },
        "protocol": {
            "case_count": len(records),
            "balanced_by_source_and_vggt_pretrigger": True,
            "vggt_pretrigger_rate": float(np.mean([row["vggt_required"] for row in records])),
            "process_res": int(args.process_res),
            "vggt_window": "full-RGB bidirectional 1+1",
            "models_resident_together": [
                "Keypoint R-CNN ResNet50-FPN",
                "DA3Metric-Large",
                "VGGT-1B camera+track",
            ],
            "human3r_excluded_from_memory_scope": True,
            "ordinary_frame_path_changed": False,
        },
        "model_load_seconds": load_seconds,
        "model_parameter_bytes": {
            "keypoint_rcnn": model_bytes(keypoint_model),
            "da3_metric_large": model_bytes(da3_model),
            "vggt_1b_camera_track": model_bytes(vggt_model),
        },
        "gpu_memory_bytes": memory,
        "latency_seconds": {
            key: distribution([row[key] for row in records])
            for key in (
                "keypoint_seconds_2frames_batch",
                "da3_seconds_2frames_independent",
                "explicit_postprocess_seconds",
                "vggt_seconds_bidirectional_1p1",
                "cut_total_seconds",
            )
        },
        "triggered_latency_seconds": distribution(
            [row["cut_total_seconds"] for row in records if row["vggt_required"]]
        ),
        "untriggered_latency_seconds": distribution(
            [row["cut_total_seconds"] for row in records if not row["vggt_required"]]
        ),
        "cases": records,
    }
    output = args.output_dir / "v24_integrated_runtime_benchmark.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f">> wrote {output}")


if __name__ == "__main__":
    main()
