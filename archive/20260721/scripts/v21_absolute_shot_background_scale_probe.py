#!/usr/bin/env python3
"""V21: estimate each shot's background metric scale without cross-cut fitting."""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from v18_da3_metric_depth_probe import (  # noqa: E402
    DEFAULT_MODEL,
    DepthAnything3,
    load_cases,
    metric_inference,
    resolve,
)
from v19_da3_depth_correction_ablation import (  # noqa: E402
    load_raw_pair,
    robust_affine,
    robust_background_pairs,
    weighted_smooth,
)
from v19_da3_explicit_geometry_correction_probe import (  # noqa: E402
    distribution,
    sample_cloud,
    scene_alignment_metrics,
)
from v20_shot_scale_consistency_probe import case_map, scale_pose  # noqa: E402


DEFAULT_V20 = (
    REPO_ROOT
    / "output"
    / "v20_shot_scale_consistency"
    / "full180_independent_bound45_conf1_fair"
    / "v20_shot_scale_consistency.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "output" / "v21_absolute_shot_background_scale"
METHOD = "torso_first1"
VARIANT = "b00"
SCALAR_METHODS = (
    "median_ratio",
    "median_log_ratio",
    "depth_bin_median",
    "trimmed_l2_scale",
)
Q_BOUNDS = (0.15, 0.30, 0.50)
AFFINE_VARIANTS = (
    ("affine_q15_d025", 0.15, 0.25),
    ("affine_q30_d050", 0.30, 0.50),
)
PROFILE_BOUNDS = (0.15, 0.30, 0.50)
GATE_THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stream_dir",
        type=Path,
        default=REPO_ROOT / "output" / "v18_human_metric_translation" / "stream_cache",
    )
    parser.add_argument(
        "--keypoint_dir",
        type=Path,
        default=REPO_ROOT / "output" / "v18_human_metric_translation" / "keypoint_cache",
    )
    parser.add_argument(
        "--scene_dir",
        type=Path,
        default=REPO_ROOT / "output" / "v18_human_metric_translation" / "v16_bound20_scene",
    )
    parser.add_argument(
        "--v18_report",
        type=Path,
        default=REPO_ROOT
        / "output"
        / "v18_human_metric_translation"
        / "final_candidates"
        / "v18_human_metric_translation_eval.json",
    )
    parser.add_argument("--v20_report", type=Path, default=DEFAULT_V20)
    parser.add_argument(
        "--v10_report",
        type=Path,
        default=REPO_ROOT
        / "output"
        / "v10_candidate_selection"
        / "oracle_gt_4source"
        / "oracle_candidate_selection_metrics.json",
    )
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache_dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--process_res", type=int, default=504)
    parser.add_argument("--point_samples", type=int, default=6000)
    parser.add_argument("--raw_confidence_threshold", type=float, default=1.0)
    parser.add_argument("--mask_dilate", type=int, default=11)
    parser.add_argument("--lowfreq_sigma", type=float, default=25.0)
    parser.add_argument("--min_background_pixels", type=int, default=512)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--merge", action="store_true")
    return parser.parse_args()


def cache_path(args: argparse.Namespace, case_name: str) -> Path:
    root = args.cache_dir or (args.output_dir / "da3_dense_cache")
    return root / f"{case_name}.npz"


def infer_pair(
    case: dict,
    model: DepthAnything3,
    args: argparse.Namespace,
) -> tuple[list[np.ndarray], list[np.ndarray], float]:
    path = cache_path(args, str(case["case_name"]))
    if path.exists() and not args.overwrite_cache:
        with np.load(path) as cache:
            return (
                [cache["old_depth"].astype(np.float32), cache["new_depth"].astype(np.float32)],
                [cache["old_intrinsics"].astype(np.float32), cache["new_intrinsics"].astype(np.float32)],
                float(cache["inference_seconds"]),
            )

    with np.load(resolve(case["cache_path"])) as stream:
        images = [stream["old_images"][-1], stream["new_image"]]
        intrinsics = np.stack([stream["old_intrinsics"][-1], stream["new_intrinsics"]]).astype(
            np.float32
        )
    depths = []
    processed = []
    elapsed = 0.0
    for index in range(2):
        depth, k_processed, seconds = metric_inference(
            model,
            [images[index]],
            intrinsics[index : index + 1],
            int(args.process_res),
        )
        depths.append(depth[0])
        processed.append(k_processed[0])
        elapsed += float(seconds)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        old_depth=depths[0].astype(np.float16),
        new_depth=depths[1].astype(np.float16),
        old_intrinsics=processed[0].astype(np.float32),
        new_intrinsics=processed[1].astype(np.float32),
        inference_seconds=np.asarray(elapsed, dtype=np.float32),
    )
    return depths, processed, elapsed


def robust_scale_estimates(raw: np.ndarray, metric: np.ndarray) -> dict[str, float]:
    ratio = metric / np.maximum(raw, 1e-4)
    valid = np.isfinite(ratio) & (ratio >= 0.2) & (ratio <= 5.0)
    x = raw[valid].astype(np.float64)
    y = metric[valid].astype(np.float64)
    ratio = ratio[valid].astype(np.float64)
    if len(ratio) < 64:
        return {name: float("nan") for name in SCALAR_METHODS}

    low, high = np.quantile(ratio, [0.10, 0.90])
    trimmed = (ratio >= low) & (ratio <= high)
    denominator = float(np.sum(x[trimmed] * x[trimmed]))
    l2_scale = float(np.sum(x[trimmed] * y[trimmed]) / max(denominator, 1e-8))

    depth_edges = np.quantile(y, np.linspace(0.0, 1.0, 7))
    bin_scales = []
    for left, right in zip(depth_edges[:-1], depth_edges[1:]):
        in_bin = (y >= left) & (y <= right)
        if int(in_bin.sum()) >= 32:
            bin_scales.append(float(np.median(ratio[in_bin])))
    depth_bin = float(np.median(bin_scales)) if bin_scales else float(np.median(ratio))
    return {
        "median_ratio": float(np.median(ratio)),
        "median_log_ratio": float(np.exp(np.median(np.log(ratio)))),
        "depth_bin_median": depth_bin,
        "trimmed_l2_scale": l2_scale,
    }


def frame_calibration(
    raw_depth: np.ndarray,
    metric_depth: np.ndarray,
    confidence: np.ndarray,
    mask: np.ndarray,
    root_scale: float,
    args: argparse.Namespace,
) -> dict:
    resized = cv2.resize(
        metric_depth,
        (raw_depth.shape[1], raw_depth.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    source, target, valid = robust_background_pairs(
        raw_depth,
        resized,
        confidence,
        mask,
        float(args.raw_confidence_threshold),
    )
    if len(source) < int(args.min_background_pixels):
        estimates = {name: float(root_scale) for name in SCALAR_METHODS}
        affine = (float(root_scale), 0.0)
        profile_depth = np.asarray([0.1, 30.0], dtype=np.float32)
        profile_ratio = np.asarray([root_scale, root_scale], dtype=np.float32)
        status = "fallback_root_scale"
    else:
        estimates = robust_scale_estimates(source, target)
        estimates = {
            name: float(value) if np.isfinite(value) else float(root_scale)
            for name, value in estimates.items()
        }
        affine = robust_affine(source, target)
        edges = np.unique(np.quantile(source, np.linspace(0.0, 1.0, 9)))
        profile_depth_values = []
        profile_ratio_values = []
        ratio = target / np.maximum(source, 1e-4)
        for left, right in zip(edges[:-1], edges[1:]):
            in_bin = (source >= left) & (source <= right)
            if int(in_bin.sum()) < 32:
                continue
            profile_depth_values.append(float(np.median(source[in_bin])))
            profile_ratio_values.append(float(np.median(ratio[in_bin])))
        if len(profile_depth_values) < 2:
            profile_depth_values = [0.1, 30.0]
            profile_ratio_values = [estimates["median_ratio"], estimates["median_ratio"]]
        profile_depth = np.asarray(profile_depth_values, dtype=np.float32)
        profile_ratio = np.asarray(profile_ratio_values, dtype=np.float32)
        status = "ok"

    log_ratio = np.log(np.maximum(resized, 1e-4) / np.maximum(raw_depth, 1e-4))
    lowfreq = weighted_smooth(log_ratio, valid, float(args.lowfreq_sigma))
    lowfreq_depth = raw_depth * np.exp(np.clip(lowfreq, np.log(0.35), np.log(3.0)))
    return {
        "status": status,
        "valid_background_pixels": int(len(source)),
        "metric_depth": resized,
        "lowfreq_depth": np.clip(lowfreq_depth, 0.05, 30.0).astype(np.float32),
        "scales": {name: float(np.clip(value, 0.35, 3.0)) for name, value in estimates.items()},
        "affine_scale": float(affine[0]),
        "affine_shift_m": float(affine[1]),
        "profile_depth": profile_depth,
        "profile_ratio": profile_ratio,
    }


def affine_depth(
    raw_depth: np.ndarray,
    calibration: dict,
    root_scale: float,
    scale_bound: float,
    shift_bound_m: float,
) -> tuple[np.ndarray, float]:
    scale = float(
        np.clip(
            calibration["affine_scale"],
            float(root_scale) * (1.0 - float(scale_bound)),
            float(root_scale) * (1.0 + float(scale_bound)),
        )
    )
    shift = float(np.clip(calibration["affine_shift_m"], -shift_bound_m, shift_bound_m))
    corrected = np.clip(raw_depth * scale + shift, 0.05, 30.0).astype(np.float32)
    effective = float(np.median(corrected / np.maximum(raw_depth, 1e-4)))
    return corrected, effective


def profile_depth(
    raw_depth: np.ndarray,
    calibration: dict,
    root_scale: float,
    bound: float,
) -> tuple[np.ndarray, float]:
    ratio = np.interp(
        raw_depth.reshape(-1),
        calibration["profile_depth"],
        calibration["profile_ratio"],
    ).reshape(raw_depth.shape)
    ratio = np.clip(
        ratio,
        float(root_scale) * (1.0 - float(bound)),
        float(root_scale) * (1.0 + float(bound)),
    )
    corrected = np.clip(raw_depth * ratio, 0.05, 30.0).astype(np.float32)
    return corrected, float(np.median(ratio))


def bounded_scene_scale(scene_scale: float, root_scale: float, bound: float) -> float:
    q = float(scene_scale) / max(float(root_scale), 1e-6)
    return float(root_scale) * float(np.clip(q, 1.0 - float(bound), 1.0 + float(bound)))


def corrected_cloud(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    raw_pose: np.ndarray,
    root_scale: float,
    mask: np.ndarray,
    confidence: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> np.ndarray:
    return sample_cloud(
        depth,
        intrinsics,
        scale_pose(raw_pose, float(root_scale)),
        mask,
        confidence,
        float(args.raw_confidence_threshold),
        int(args.point_samples),
        rng,
    )


def run_case(
    case: dict,
    v10_case: dict,
    v20_case: dict,
    model: DepthAnything3,
    args: argparse.Namespace,
    index: int,
) -> dict:
    with np.load(resolve(case["cache_path"])) as stream:
        raw_poses = np.stack([stream["old_pose"][-1], stream["new_pose"]]).astype(np.float32)
        intrinsics = np.stack([stream["old_intrinsics"][-1], stream["new_intrinsics"]]).astype(
            np.float32
        )
    source = v20_case["methods"][METHOD]
    root_scales = [float(source["old_scale"]), float(source["new_scale"])]
    initial = source["variants"][VARIANT]
    transform = np.asarray(initial["transform"], dtype=np.float32)

    raw = load_raw_pair(Path(v10_case["paths"]["human3r_local_reset"]))
    kernel = np.ones((int(args.mask_dilate), int(args.mask_dilate)), dtype=np.uint8)
    masks = [cv2.dilate(value.astype(np.uint8), kernel, iterations=1) for value in raw["mask"]]
    metric_depths, _, elapsed = infer_pair(case, model, args)
    calibration = [
        frame_calibration(
            raw["depth"][frame],
            metric_depths[frame],
            raw["confidence"][frame],
            masks[frame],
            root_scales[frame],
            args,
        )
        for frame in range(2)
    ]

    depth_variants: dict[str, list[np.ndarray]] = {
        "root_scale": [raw["depth"][i] * root_scales[i] for i in range(2)],
        "da3_dense": [calibration[i]["metric_depth"] for i in range(2)],
        "lowfreq_scale": [calibration[i]["lowfreq_depth"] for i in range(2)],
    }
    scale_variants: dict[str, list[float]] = {"root_scale": root_scales}
    for method in SCALAR_METHODS:
        scene_scales = [calibration[i]["scales"][method] for i in range(2)]
        depth_variants[method] = [raw["depth"][i] * scene_scales[i] for i in range(2)]
        scale_variants[method] = scene_scales
        for bound in Q_BOUNDS:
            name = f"{method}_q{int(round(100 * bound)):02d}"
            bounded = [
                bounded_scene_scale(scene_scales[i], root_scales[i], bound) for i in range(2)
            ]
            depth_variants[name] = [raw["depth"][i] * bounded[i] for i in range(2)]
            scale_variants[name] = bounded
    for name, scale_bound, shift_bound in AFFINE_VARIANTS:
        corrected = [
            affine_depth(
                raw["depth"][i], calibration[i], root_scales[i], scale_bound, shift_bound
            )
            for i in range(2)
        ]
        depth_variants[name] = [value[0] for value in corrected]
        scale_variants[name] = [value[1] for value in corrected]
    for bound in PROFILE_BOUNDS:
        name = f"depth_profile_q{int(round(100 * bound)):02d}"
        corrected = [
            profile_depth(raw["depth"][i], calibration[i], root_scales[i], bound)
            for i in range(2)
        ]
        depth_variants[name] = [value[0] for value in corrected]
        scale_variants[name] = [value[1] for value in corrected]
    for threshold in GATE_THRESHOLDS:
        gate_key = int(round(100 * threshold))
        q = [
            calibration[i]["scales"]["median_ratio"] / max(root_scales[i], 1e-6)
            for i in range(2)
        ]
        for source_name in ("median_ratio_q15", "depth_profile_q15", "lowfreq_scale"):
            name = f"{source_name}_gate_lt{gate_key:02d}"
            depth_variants[name] = [
                depth_variants[source_name][i] if q[i] < threshold else depth_variants["root_scale"][i]
                for i in range(2)
            ]
            source_scales = scale_variants.get(source_name, root_scales)
            scale_variants[name] = [
                source_scales[i] if q[i] < threshold else root_scales[i] for i in range(2)
            ]

    variants = {}
    for name, depths in depth_variants.items():
        # Reuse the same sampling stream so scale variants are compared on the same pixels.
        rng = np.random.default_rng(int(args.seed) + 1009 * index)
        clouds = [
            corrected_cloud(
                depths[frame],
                intrinsics[frame],
                raw_poses[frame],
                root_scales[frame],
                masks[frame],
                raw["confidence"][frame],
                args,
                rng,
            )
            for frame in range(2)
        ]
        variants[name] = {
            "scene": scene_alignment_metrics(transform, clouds[1], clouds[0]),
            "old_scene_scale": float(scale_variants.get(name, [np.nan, np.nan])[0]),
            "new_scene_scale": float(scale_variants.get(name, [np.nan, np.nan])[1]),
        }

    return {
        "case_name": case["case_name"],
        "source": case["source"],
        "record": case.get("record", {}),
        "camera": initial["camera"],
        "human": initial["human"],
        "root_scales": {"old": root_scales[0], "new": root_scales[1]},
        "calibration": {
            "old": {
                "status": calibration[0]["status"],
                "valid_background_pixels": calibration[0]["valid_background_pixels"],
                "scales": calibration[0]["scales"],
                "affine_scale": calibration[0]["affine_scale"],
                "affine_shift_m": calibration[0]["affine_shift_m"],
            },
            "new": {
                "status": calibration[1]["status"],
                "valid_background_pixels": calibration[1]["valid_background_pixels"],
                "scales": calibration[1]["scales"],
                "affine_scale": calibration[1]["affine_scale"],
                "affine_shift_m": calibration[1]["affine_shift_m"],
            },
        },
        "variants": variants,
        "da3_inference_seconds": float(elapsed),
    }


def summarize(rows: list[dict], variant: str) -> dict:
    scene = [row["variants"][variant]["scene"]["trimmed_mean_m"] for row in rows]
    root_scene = [row["variants"]["root_scale"]["scene"]["trimmed_mean_m"] for row in rows]
    old_scale = np.asarray(
        [row["variants"][variant]["old_scene_scale"] for row in rows], dtype=np.float64
    )
    new_scale = np.asarray(
        [row["variants"][variant]["new_scene_scale"] for row in rows], dtype=np.float64
    )
    output = {
        "scene_trimmed_mean_m": distribution(scene),
        "scene_overlap_020": distribution(
            [row["variants"][variant]["scene"]["overlap_020"] for row in rows]
        ),
        "scene_improved_over_root_rate": float(np.mean(np.asarray(scene) < np.asarray(root_scene))),
        "scene_harmful_rate_010m": float(
            np.mean(np.asarray(scene) > np.asarray(root_scene) + 0.10)
        ),
    }
    if np.isfinite(old_scale).any() and np.isfinite(new_scale).any():
        output["old_scene_scale"] = distribution(old_scale.tolist())
        output["new_scene_scale"] = distribution(new_scale.tolist())
    return output


def chain_pairs(rows: list[dict], max_frame_gap: int = 30) -> list[tuple[dict, dict]]:
    pairs = []
    for first in rows:
        a = first.get("record", {})
        for second in rows:
            if first is second or first["source"] != second["source"]:
                continue
            b = second.get("record", {})
            if a.get("group") != b.get("group") or a.get("seqB") != b.get("seqA"):
                continue
            if abs(int(a.get("start_frame", -100000)) - int(b.get("start_frame", 100000))) <= int(
                max_frame_gap
            ):
                pairs.append((first, second))
    return pairs


def stability(values: list[tuple[float, float]]) -> dict:
    delta = np.asarray(
        [abs(float(np.log(max(a, 1e-6) / max(b, 1e-6)))) for a, b in values],
        dtype=np.float64,
    )
    return {
        "pair_count": int(len(delta)),
        "median_abs_log_difference": float(np.median(delta)) if len(delta) else float("nan"),
        "p90_abs_log_difference": float(np.quantile(delta, 0.90)) if len(delta) else float("nan"),
        "within_10_percent": float(np.mean(delta <= np.log(1.10))) if len(delta) else float("nan"),
        "within_20_percent": float(np.mean(delta <= np.log(1.20))) if len(delta) else float("nan"),
    }


def build_report(rows: list[dict], args: argparse.Namespace) -> dict:
    variants = sorted(rows[0]["variants"])
    pairs = chain_pairs(rows)
    chain = {
        "root_scale": stability(
            [(first["root_scales"]["new"], second["root_scales"]["old"]) for first, second in pairs]
        )
    }
    for variant in variants:
        if variant in ("da3_dense", "lowfreq_scale"):
            continue
        chain[variant] = stability(
            [
                (
                    first["variants"][variant]["new_scene_scale"],
                    second["variants"][variant]["old_scene_scale"],
                )
                for first, second in pairs
            ]
        )
    return {
        "experiment": "V21 independent absolute shot background metric-scale probe",
        "case_count": len(rows),
        "protocol": {
            "da3_inference": "independent single-frame per shot",
            "camera_human_boundary": f"V20 {METHOD}/{VARIANT}",
            "cross_cut_scene_fitting": False,
            "pointmap_only_absolute_calibration": True,
            "q_bounds_relative_to_human_root_scale": list(Q_BOUNDS),
            "independent_shot_gate_thresholds": list(GATE_THRESHOLDS),
            "chain_max_start_frame_gap": 30,
            "learned_components": False,
        },
        "overall": {variant: summarize(rows, variant) for variant in variants},
        "by_source": {
            source: {
                variant: summarize([row for row in rows if row["source"] == source], variant)
                for variant in variants
            }
            for source in sorted({row["source"] for row in rows})
        },
        "chain_stability": chain,
        "cases": rows,
    }


def merge(args: argparse.Namespace) -> None:
    paths = sorted(glob.glob(str(args.output_dir / "v21_shard_*_of_*.json")))
    rows = []
    for path in paths:
        rows.extend(json.loads(Path(path).read_text(encoding="utf-8"))["cases"])
    unique = {row["case_name"]: row for row in rows}
    if len(unique) != 180:
        raise RuntimeError(f"Expected 180 unique cases across shards, got {len(unique)}")
    report = build_report([unique[name] for name in sorted(unique)], args)
    output = args.output_dir / "v21_absolute_shot_background_scale.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(f">> wrote {output}", flush=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.merge:
        merge(args)
        return
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("V21 absolute shot scale probe requires CUDA")
    if not (0 <= int(args.shard_id) < int(args.num_shards)):
        raise ValueError("shard_id must be in [0, num_shards)")

    cases = load_cases(args)
    cases = [case for index, case in enumerate(cases) if index % int(args.num_shards) == int(args.shard_id)]
    v10 = case_map(args.v10_report)
    v20 = case_map(args.v20_report)
    model = DepthAnything3.from_pretrained(str(args.model_path)).to(args.device).eval()
    started = time.perf_counter()
    rows = []
    for index, case in enumerate(cases):
        rows.append(run_case(case, v10[case["case_name"]], v20[case["case_name"]], model, args, index))
        print(
            f"V21 shard {args.shard_id}/{args.num_shards} {index + 1}/{len(cases)}",
            flush=True,
        )
    payload = {
        "experiment": "V21 independent absolute shot background metric-scale shard",
        "shard_id": int(args.shard_id),
        "num_shards": int(args.num_shards),
        "elapsed_seconds": float(time.perf_counter() - started),
        "cases": rows,
    }
    output = args.output_dir / f"v21_shard_{int(args.shard_id):02d}_of_{int(args.num_shards):02d}.json"
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(f">> wrote {output}", flush=True)


if __name__ == "__main__":
    main()
