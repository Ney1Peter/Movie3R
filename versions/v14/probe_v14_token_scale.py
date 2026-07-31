#!/usr/bin/env python3
"""Probe frozen V14/Human3R tokens for cross-sequence shot-scale information."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


from versions.v13 import gt_id_consensus as geometry  # noqa: E402
from versions.v14.probe_v14_shot_scale import (  # noqa: E402
    DEFAULT_DATA,
    frame_metric_scales,
)
from versions.v14.run_v14_2_single_sequence import (  # noqa: E402
    camera_matrix,
    configure_model,
    set_event_indices,
)


DEFAULT_MODEL = Path(
    "/dev/shm/movie3r_v14_1/"
    "v14_1_v9_event_only_boundary_geometry_self20_fp32_e80/checkpoint-best.pth"
)
DEFAULT_SCALE_REPORT = REPO_ROOT / "output/v14/shot_scale_audit/v14_shot_scale_audit.json"
DEFAULT_OUTPUT = REPO_ROOT / "output/v14/token_scale_probe"
FEATURE_NAMES = (
    "pose_token",
    "scene_state",
    "human_global",
    "refined_human_mean",
    "fused_prompt_mean",
    "cut3r_head_mean",
    "mhmr_head_mean",
    "combined",
)
TARGETS = (
    "body_relative",
    "radial_relative",
    "depth_relative",
    "layout_metric_relative",
    "oracle_root",
)
RIDGE_ALPHA_SWEEP = (0.1, 1.0, 10.0, 100.0, 1000.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--scale_report", type=Path, default=DEFAULT_SCALE_REPORT)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--ridge_alpha", type=float, default=10.0)
    return parser.parse_args()


def tensor_vector(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy().reshape(-1).astype(np.float64)


def mean_human_token(value: torch.Tensor, dimension: int) -> np.ndarray:
    array = value.detach().float().cpu().numpy()
    if array.size == 0 or array.shape[-2] == 0:
        return np.zeros(dimension, dtype=np.float64)
    return np.asarray(array.reshape(-1, array.shape[-1]).mean(axis=0), dtype=np.float64)


def debug_features(debug: dict) -> dict[str, np.ndarray]:
    pose = tensor_vector(debug["pose_token_out"])
    scene = tensor_vector(debug["state_summary_new"])
    human = tensor_vector(debug["human_token_out"])
    refined = mean_human_token(debug["refined_human_tokens"], 768)
    fused = mean_human_token(debug["fused_human_prompts"], 768)
    cut3r = mean_human_token(debug["cut3r_head_tokens"], 1024)
    mhmr = mean_human_token(debug["mhmr_head_tokens"], 1024)
    output = {
        "pose_token": pose,
        "scene_state": scene,
        "human_global": human,
        "refined_human_mean": refined,
        "fused_prompt_mean": fused,
        "cut3r_head_mean": cut3r,
        "mhmr_head_mean": mhmr,
    }
    output["combined"] = np.concatenate([pose, scene, human, refined])
    return output


def frame_keys(scale_report: dict) -> list[tuple[str, int, int]]:
    keys = set()
    for row in scale_report["cases"]:
        case = row["case"]
        keys.add(
            (
                row["sequence"],
                int(case["source_camera"]),
                int(case["pre_frames"][-1]),
            )
        )
        keys.add(
            (
                row["sequence"],
                int(case["target_camera"]),
                int(case["post_frame"]),
            )
        )
    return sorted(keys)


def infer_frame(model, layer, args: argparse.Namespace, key: tuple[str, int, int]) -> dict:
    sequence, camera, frame = key
    args.sequence = sequence
    geometry.IDENTITIES = geometry.SEQUENCE_IDENTITIES[sequence]
    # Keep extraction caches sequence-disjoint because camera/frame numbers overlap.
    original_output = args.output_dir
    args.output_dir = original_output / "input_frames" / sequence
    path = geometry.extract_video_frame(args, camera, frame)
    views = set_event_indices(
        geometry.prepare_full_square_input(model, [path], args), set()
    )
    started = time.perf_counter()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
        predictions, returned_views, debug = model.forward_recurrent_lighter(
            views,
            str(args.device),
            ret_state=False,
            use_ttt3r=False,
            return_token_debug=True,
        )
    elapsed = time.perf_counter() - started
    prediction = predictions[0]
    view = returned_views[0]
    token_debug = debug[0]
    humans = geometry.layer_humans(prediction, view, token_debug, layer)
    height, width = [
        int(value) for value in geometry.tensor_numpy(view["true_shape"])[0]
    ]
    assigned, assignment = geometry.assign_gt_identities(
        args,
        humans,
        camera_matrix(prediction),
        camera,
        frame,
        width,
        height,
    )
    labels = frame_metric_scales(
        args,
        assigned,
        camera_matrix(prediction),
        camera,
        frame,
    )
    features = debug_features(token_debug)
    args.output_dir = original_output
    return {
        "key": {"sequence": sequence, "camera": camera, "frame": frame},
        "labels": {
            name: float(labels[name]) for name in ("body", "radial", "depth", "layout")
        },
        "features": features,
        "num_humans": len(assigned),
        "assignment_status": assignment.get("status", "unknown"),
        "runtime_seconds": elapsed,
    }


def save_feature_cache(path: Path, records: list[dict]) -> None:
    arrays = {}
    arrays["sequence"] = np.asarray(
        [record["key"]["sequence"] for record in records], dtype="U16"
    )
    arrays["camera"] = np.asarray(
        [record["key"]["camera"] for record in records], dtype=np.int32
    )
    arrays["frame"] = np.asarray(
        [record["key"]["frame"] for record in records], dtype=np.int32
    )
    for target in ("body", "radial", "depth", "layout"):
        arrays[f"label_{target}"] = np.asarray(
            [record["labels"][target] for record in records], dtype=np.float64
        )
    for name in FEATURE_NAMES:
        arrays[f"feature_{name}"] = np.stack(
            [record["features"][name] for record in records]
        ).astype(np.float32)
    np.savez_compressed(path, **arrays)


def load_feature_cache(path: Path) -> dict[tuple[str, int, int], dict]:
    with np.load(path) as values:
        output = {}
        for index in range(len(values["sequence"])):
            key = (
                str(values["sequence"][index]),
                int(values["camera"][index]),
                int(values["frame"][index]),
            )
            output[key] = {
                "labels": {
                    target: float(values[f"label_{target}"][index])
                    for target in ("body", "radial", "depth", "layout")
                },
                "features": {
                    name: np.asarray(values[f"feature_{name}"][index], dtype=np.float64)
                    for name in FEATURE_NAMES
                },
            }
    return output


def pair_rows(scale_report: dict, frames: dict) -> list[dict]:
    output = []
    for case_row in scale_report["cases"]:
        case = case_row["case"]
        pre_key = (
            case_row["sequence"],
            int(case["source_camera"]),
            int(case["pre_frames"][-1]),
        )
        post_key = (
            case_row["sequence"],
            int(case["target_camera"]),
            int(case["post_frame"]),
        )
        if pre_key not in frames or post_key not in frames:
            continue
        output.append(
            {
                "sequence": case_row["sequence"],
                "case_key": case["key"],
                "targets": {
                    name: float(case_row["scale_measurements"][name])
                    for name in TARGETS
                },
                "features": {
                    name: frames[post_key]["features"][name]
                    - frames[pre_key]["features"][name]
                    for name in FEATURE_NAMES
                },
            }
        )
    return output


def ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, float]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    keep = std > 1e-6
    train = (train_x[:, keep] - mean[keep]) / std[keep]
    test = (test_x[:, keep] - mean[keep]) / std[keep]
    target_mean = float(np.mean(train_y))
    centered = train_y - target_mean
    kernel = train @ train.T
    dual = np.linalg.solve(
        kernel + float(alpha) * np.eye(len(kernel), dtype=np.float64), centered
    )
    return target_mean + (test @ train.T) @ dual, target_mean


def regression_metrics(target: np.ndarray, predicted: np.ndarray, baseline: float) -> dict:
    finite = np.isfinite(target) & np.isfinite(predicted)
    target = target[finite]
    predicted = predicted[finite]
    if not len(target):
        return {name: float("nan") for name in ("mae", "baseline_mae", "r2", "pearson")}
    residual = float(np.sum((target - predicted) ** 2))
    total = float(np.sum((target - np.mean(target)) ** 2))
    pearson = (
        float(np.corrcoef(target, predicted)[0, 1])
        if len(target) > 1 and np.std(target) > 1e-8 and np.std(predicted) > 1e-8
        else float("nan")
    )
    return {
        "count": len(target),
        "mae": float(np.mean(np.abs(target - predicted))),
        "baseline_mae": float(np.mean(np.abs(target - baseline))),
        "mae_gain_vs_constant": float(
            np.mean(np.abs(target - baseline)) - np.mean(np.abs(target - predicted))
        ),
        "r2": float(1.0 - residual / max(total, 1e-12)),
        "pearson": pearson,
    }


def run_probe(rows: list[dict], alpha: float) -> dict:
    output = {}
    splits = {
        "three_to_dance": (("three",), ("dance",)),
        "three_to_box": (("three",), ("box",)),
        "three_to_dance_box": (("three",), ("dance", "box")),
        "leave_three_out": (("dance", "box"), ("three",)),
        "leave_dance_out": (("three", "box"), ("dance",)),
        "leave_box_out": (("three", "dance"), ("box",)),
    }
    for split, (train_sequences, test_sequences) in splits.items():
        train_rows = [row for row in rows if row["sequence"] in train_sequences]
        test_rows = [row for row in rows if row["sequence"] in test_sequences]
        output[split] = {}
        for feature in FEATURE_NAMES:
            train_x = np.stack([row["features"][feature] for row in train_rows])
            test_x = np.stack([row["features"][feature] for row in test_rows])
            output[split][feature] = {}
            for target in TARGETS:
                train_y = np.asarray(
                    [row["targets"][target] for row in train_rows], dtype=np.float64
                )
                test_y = np.asarray(
                    [row["targets"][target] for row in test_rows], dtype=np.float64
                )
                finite_train = np.isfinite(train_y) & np.isfinite(train_x).all(axis=1)
                finite_test = np.isfinite(test_y) & np.isfinite(test_x).all(axis=1)
                if int(finite_train.sum()) < 4 or not finite_test.any():
                    output[split][feature][target] = regression_metrics(
                        np.asarray([]), np.asarray([]), 1.0
                    )
                    continue
                predicted, baseline = ridge_predict(
                    train_x[finite_train],
                    train_y[finite_train],
                    test_x[finite_test],
                    alpha,
                )
                output[split][feature][target] = regression_metrics(
                    test_y[finite_test], predicted, baseline
                )
    return output


def markdown(report: dict) -> str:
    lines = [
        "# V14 Frozen Token Scale Probe",
        "",
        "A fixed ridge probe is trained on `three`; no V14/Human3R parameter is updated.",
        "Positive MAE gain means the token probe beats a constant train-mean predictor.",
        "",
    ]
    for split in ("three_to_dance", "three_to_box", "three_to_dance_box"):
        lines.extend(
            [
                f"## {split}",
                "",
                "| Feature | Body gain | Radial gain | Depth gain | Layout gain | Oracle-root gain |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for feature in FEATURE_NAMES:
            values = report["probe"][split][feature]
            lines.append(
                f"| {feature} | "
                + " | ".join(
                    f"{values[target]['mae_gain_vs_constant']:+.4f}"
                    for target in TARGETS
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if not args.model_path.is_file():
        raise FileNotFoundError(args.model_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scale_report = json.loads(args.scale_report.read_text(encoding="utf-8"))
    keys = frame_keys(scale_report)
    if int(args.max_frames) > 0:
        keys = keys[: int(args.max_frames)]
    feature_path = args.output_dir / "v14_token_scale_features.npz"

    if not feature_path.is_file() or int(args.max_frames) > 0:
        from dust3r.model import ARCroco3DStereo
        from dust3r.utils.smpl_layer import SMPL_Layer

        device = torch.device(args.device)
        model = ARCroco3DStereo.from_pretrained(str(args.model_path)).to(device).eval()
        flags = configure_model(model)
        layer = SMPL_Layer(
            type="smplx",
            gender="neutral",
            num_betas=10,
            kid=False,
            person_center="head",
        ).to(device).eval()
        records = []
        for index, key in enumerate(keys, start=1):
            record = infer_frame(model, layer, args, key)
            records.append(record)
            print(
                f"[{index:03d}/{len(keys):03d}] {key} "
                f"humans={record['num_humans']} {record['runtime_seconds']:.2f}s",
                flush=True,
            )
            gc.collect()
            if torch.cuda.is_available() and index % 10 == 0:
                torch.cuda.empty_cache()
        save_feature_cache(feature_path, records)
        model_flags = flags
    else:
        model_flags = {"loaded_from_existing_feature_cache": True}

    frames = load_feature_cache(feature_path)
    rows = pair_rows(scale_report, frames)
    report = {
        "experiment": "V14 frozen token shot-scale linear probe",
        "protocol": {
            "frame_count": len(frames),
            "pair_count": len(rows),
            "features": list(FEATURE_NAMES),
            "targets": list(TARGETS),
            "ridge_alpha": float(args.ridge_alpha),
            "model_frozen": True,
            "gt_runtime_information": False,
            "model_flags": model_flags,
        },
        "probe": run_probe(rows, float(args.ridge_alpha)),
        "alpha_sweep": {
            str(alpha): run_probe(rows, alpha) for alpha in RIDGE_ALPHA_SWEEP
        },
    }
    json_path = args.output_dir / "v14_token_scale_probe.json"
    md_path = args.output_dir / "v14_token_scale_probe.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n")
    md_path.write_text(markdown(report), encoding="utf-8")
    print(markdown(report), flush=True)
    print(f">> wrote {json_path}", flush=True)


if __name__ == "__main__":
    main()
